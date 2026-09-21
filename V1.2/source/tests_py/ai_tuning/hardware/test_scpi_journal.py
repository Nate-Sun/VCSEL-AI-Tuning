from __future__ import annotations

from threading import Event, Thread

from vcsel_ai_tuning.hardware.scpi_journal import (
    JournaledScpiSession,
    ScpiOperationJournal,
)


class _BlockingSession:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.timeout = 60_000

    def query(self, command: str) -> str:
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return f"ok:{command}"


def test_started_query_is_fsynced_before_delegate_returns(tmp_path):
    delegate = _BlockingSession()
    journal = ScpiOperationJournal(tmp_path / "journal.jsonl")
    session = JournaledScpiSession(
        delegate,
        journal,
        lambda: {
            "task_id": "t1",
            "item_id": "p1",
            "round_index": 1,
            "remaining_deadline_s": 0.125,
        },
    )
    result: list[str] = []
    thread = Thread(target=lambda: result.append(session.query("*IDN?")))

    thread.start()
    assert delegate.entered.wait(timeout=1.0)
    events = journal.events()
    assert len(events) == 1
    assert events[0]["event"] == "STARTED"
    assert events[0]["operation"] == "query"
    assert events[0]["task_id"] == "t1"
    assert journal.last_in_flight() == events[0]
    assert 1 <= delegate.timeout <= 125

    delegate.release.set()
    thread.join(timeout=1.0)
    assert result == ["ok:*IDN?"]
    assert [event["event"] for event in journal.events()] == [
        "STARTED",
        "COMPLETED",
    ]
    assert journal.last_in_flight() is None


def test_failed_write_is_terminal_and_preserves_context(tmp_path):
    class _Broken:
        timeout = 1000

        def write(self, command: str) -> None:
            raise OSError(f"lost:{command}")

    journal = ScpiOperationJournal(tmp_path / "journal.jsonl")
    session = JournaledScpiSession(
        _Broken(), journal, lambda: {"task_id": "t2", "round_index": 4}
    )

    try:
        session.write(":ACQuire:SINGle")
    except OSError:
        pass

    events = journal.events()
    assert [event["event"] for event in events] == ["STARTED", "FAILED"]
    assert events[0]["operation_id"] == events[1]["operation_id"]
    assert events[1]["task_id"] == "t2"
    assert journal.last_in_flight() is None
