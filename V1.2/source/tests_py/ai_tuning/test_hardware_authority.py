from concurrent.futures import ThreadPoolExecutor
import os
from queue import Empty, Queue
import subprocess
import sys
from threading import Barrier, Event, Lock, Thread
import time

import pytest

from vcsel_ai_tuning.hardware_authority import (
    HardwareAuthority,
    HardwareAuthorityBusy,
)


class HandleProxy:
    def __init__(self, handle, *, write_error=None, close_error=None):
        self._handle = handle
        self._write_error = write_error
        self._close_error = close_error

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def write(self, payload):
        if self._write_error is not None:
            raise self._write_error
        return self._handle.write(payload)

    def close(self):
        self._handle.close()
        if self._close_error is not None:
            raise self._close_error

    def force_close(self):
        if not self._handle.closed:
            self._handle.close()


CHILD_SCRIPT = """
import sys
from vcsel_ai_tuning.hardware_authority import HardwareAuthority

authority = HardwareAuthority(sys.argv[1])
with authority.acquire(sys.argv[2]):
    print("READY", flush=True)
    sys.stdin.readline()
"""


PAUSED_WINNER_SCRIPT = """
import sys
from vcsel_ai_tuning.hardware_authority import HardwareAuthority

original_write_owner = HardwareAuthority._write_file_owner

def paused_write_owner(handle, owner):
    print("HARDWARE_LOCKED", flush=True)
    sys.stdin.readline()
    return original_write_owner(handle, owner)

HardwareAuthority._write_file_owner = staticmethod(paused_write_owner)
authority = HardwareAuthority(sys.argv[1])
with authority.acquire(sys.argv[2]):
    print("READY", flush=True)
    sys.stdin.readline()
"""


CONTENDER_SCRIPT = """
import sys
from vcsel_ai_tuning.hardware_authority import HardwareAuthority, HardwareAuthorityBusy

try:
    HardwareAuthority(sys.argv[1]).acquire(sys.argv[2])
except HardwareAuthorityBusy as error:
    print(f"BUSY:{error.current_owner}", flush=True)
else:
    print("ACQUIRED", flush=True)
"""


DELAYED_INITIALIZER_SCRIPT = """
import os
import sys
from vcsel_ai_tuning.hardware_authority import HardwareAuthority, HardwareAuthorityBusy

def delayed_initialize(handle):
    handle.seek(0, os.SEEK_END)
    missing = HardwareAuthority._OWNER_OFFSET - handle.tell()
    print("INITIALIZER_PAUSED", flush=True)
    sys.stdin.readline()
    if missing > 0:
        try:
            handle.write(b"\\0" * missing)
        except PermissionError:
            handle.seek(0, os.SEEK_END)
            if handle.tell() < HardwareAuthority._OWNER_OFFSET:
                raise
        handle.flush()

HardwareAuthority._initialize_file = staticmethod(delayed_initialize)
try:
    HardwareAuthority(sys.argv[1]).acquire("delayed-initializer")
except HardwareAuthorityBusy as error:
    print(f"BUSY:{error.current_owner}", flush=True)
else:
    print("ACQUIRED", flush=True)
"""


def spawn_authority_owner(path, owner):
    return subprocess.Popen(
        [sys.executable, "-c", CHILD_SCRIPT, str(path), owner],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )


def wait_until_ready(process):
    assert process.stdout is not None
    ready = read_line_queue(process.stdout).get(timeout=10)
    if ready.strip() != "READY":
        assert process.stderr is not None
        raise AssertionError(f"child did not acquire authority: {process.stderr.read()}")


def read_line_queue(stream):
    result = Queue(maxsize=1)

    def read_line():
        result.put(stream.readline())

    Thread(target=read_line, daemon=True).start()
    return result


def test_two_authorities_for_one_path_yield_exactly_one_winner(tmp_path):
    path = tmp_path / "hardware-authority.lock"
    authorities = (HardwareAuthority(path), HardwareAuthority(path))
    start = Barrier(2)
    loser_finished = Event()
    results = []
    results_lock = Lock()

    def contend(index):
        owner = f"contender-{index}"
        start.wait(timeout=5)
        try:
            with authorities[index].acquire(owner):
                with results_lock:
                    results.append(("winner", owner))
                assert loser_finished.wait(timeout=5)
        except HardwareAuthorityBusy as error:
            with results_lock:
                results.append(("busy", owner, error.current_owner))
            loser_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(contend, index) for index in range(2)]
        for future in futures:
            future.result(timeout=5)

    winners = [result for result in results if result[0] == "winner"]
    losers = [result for result in results if result[0] == "busy"]
    assert len(winners) == 1
    assert len(losers) == 1
    assert losers[0][2] == winners[0][1]


def test_release_is_idempotent_and_permits_a_later_owner(tmp_path):
    path = tmp_path / "hardware-authority.lock"
    first = HardwareAuthority(path)
    second = HardwareAuthority(path)

    lease = first.acquire("first-owner")
    lease.release()
    lease.release()

    with second.acquire("later-owner") as later:
        assert later.owner == "later-owner"
        assert later.released is False
    assert later.released is True


def test_metadata_write_failure_releases_os_lock(tmp_path, monkeypatch):
    path = tmp_path / "hardware-authority.lock"
    path.write_bytes(b"\0\0")
    opened = []
    original_open = HardwareAuthority._open_file

    def failing_open(candidate):
        handle = original_open(candidate)
        if candidate != path.resolve():
            return handle
        proxy = HandleProxy(handle, write_error=OSError("metadata write failed"))
        opened.append(proxy)
        return proxy

    authority = HardwareAuthority(path)
    with monkeypatch.context() as patcher:
        patcher.setattr(
            HardwareAuthority, "_open_file", staticmethod(failing_open)
        )
        with pytest.raises(OSError, match="metadata write failed"):
            authority.acquire("first-owner")

    try:
        with HardwareAuthority(path).acquire("later-owner"):
            pass
    finally:
        for handle in opened:
            handle.force_close()


def test_file_initialization_error_is_not_reported_as_busy(tmp_path, monkeypatch):
    path = (tmp_path / "hardware-authority.lock").resolve()
    original_open = HardwareAuthority._open_file

    def failing_open(candidate):
        handle = original_open(candidate)
        if candidate != path:
            return handle
        return HandleProxy(
            handle,
            write_error=PermissionError("initialize failed"),
            close_error=OSError("close failed"),
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            HardwareAuthority, "_open_file", staticmethod(failing_open)
        )
        with pytest.raises(PermissionError, match="initialize failed"):
            HardwareAuthority(path).acquire("owner")

    with HardwareAuthority(path).acquire("later-owner"):
        pass


def test_release_discards_process_owner_even_when_close_fails(tmp_path, monkeypatch):
    path = (tmp_path / "hardware-authority.lock").resolve()
    path.write_bytes(b"\0")
    original_open = HardwareAuthority._open_file

    def close_failing_open(candidate):
        handle = original_open(candidate)
        if candidate != path:
            return handle
        return HandleProxy(handle, close_error=OSError("close failed"))

    with monkeypatch.context() as patcher:
        patcher.setattr(
            HardwareAuthority, "_open_file", staticmethod(close_failing_open)
        )
        lease = HardwareAuthority(path).acquire("first-owner")
        with pytest.raises(OSError, match="close failed"):
            lease.release()

    with HardwareAuthority(path).acquire("later-owner"):
        pass


def test_independent_process_reports_exact_owner_and_releases_on_exit(tmp_path):
    path = tmp_path / "hardware-authority.lock"
    process = spawn_authority_owner(path, "child-owner")
    try:
        wait_until_ready(process)
        with pytest.raises(HardwareAuthorityBusy) as caught:
            HardwareAuthority(path).acquire("parent-owner")
        assert caught.value.current_owner == "child-owner"
        assert process.stdin is not None
        process.stdin.write("\n")
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

    with HardwareAuthority(path).acquire("later-owner"):
        pass


def test_process_termination_releases_os_authority_lock(tmp_path):
    path = tmp_path / "hardware-authority.lock"
    process = spawn_authority_owner(path, "terminated-owner")
    try:
        wait_until_ready(process)
        with pytest.raises(HardwareAuthorityBusy) as caught:
            HardwareAuthority(path).acquire("parent-owner")
        assert caught.value.current_owner == "terminated-owner"
    finally:
        process.terminate()
        process.wait(timeout=10)

    deadline = time.monotonic() + 2.0
    while True:
        try:
            lease = HardwareAuthority(path).acquire("later-owner")
        except HardwareAuthorityBusy:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
        else:
            break
    with lease:
        pass


def test_contender_waits_for_exact_owner_publication(tmp_path):
    path = tmp_path / "hardware-authority.lock"
    winner = subprocess.Popen(
        [sys.executable, "-c", PAUSED_WINNER_SCRIPT, str(path), "winner-owner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    contender = None
    try:
        assert winner.stdout is not None
        transition = read_line_queue(winner.stdout).get(timeout=10)
        assert transition.strip() == "HARDWARE_LOCKED"

        contender = subprocess.Popen(
            [sys.executable, "-c", CONTENDER_SCRIPT, str(path), "contender"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        assert contender.stdout is not None
        outcome_reader = read_line_queue(contender.stdout)
        try:
            outcome = outcome_reader.get(timeout=0.2)
        except Empty:
            outcome = None

        assert winner.stdin is not None
        winner.stdin.write("\n")
        winner.stdin.flush()
        ready = read_line_queue(winner.stdout).get(timeout=10)
        assert ready.strip() == "READY"
        if outcome is None:
            outcome = outcome_reader.get(timeout=10)
        assert outcome.strip() == "BUSY:winner-owner"
        assert contender.wait(timeout=10) == 0

        winner.stdin.write("\n")
        winner.stdin.flush()
        assert winner.wait(timeout=10) == 0
    finally:
        if contender is not None and contender.poll() is None:
            contender.terminate()
            contender.wait(timeout=10)
        if winner.poll() is None:
            winner.terminate()
            winner.wait(timeout=10)


def test_concurrent_first_initialization_cannot_corrupt_owner(tmp_path):
    path = tmp_path / "hardware-authority.lock"
    delayed = subprocess.Popen(
        [sys.executable, "-c", DELAYED_INITIALIZER_SCRIPT, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    winner = None
    try:
        assert delayed.stdout is not None
        paused = read_line_queue(delayed.stdout).get(timeout=10)
        assert paused.strip() == "INITIALIZER_PAUSED"

        winner = spawn_authority_owner(path, "winner-owner")
        wait_until_ready(winner)

        assert delayed.stdin is not None
        delayed.stdin.write("\n")
        delayed.stdin.flush()
        outcome = read_line_queue(delayed.stdout).get(timeout=10)
        if not outcome:
            assert delayed.stderr is not None
            raise AssertionError(delayed.stderr.read())
        assert outcome.strip() == "BUSY:winner-owner"
        assert delayed.wait(timeout=10) == 0

        assert winner.stdin is not None
        winner.stdin.write("\n")
        winner.stdin.flush()
        assert winner.wait(timeout=10) == 0
    finally:
        if winner is not None and winner.poll() is None:
            winner.terminate()
            winner.wait(timeout=10)
        if delayed.poll() is None:
            delayed.terminate()
            delayed.wait(timeout=10)
