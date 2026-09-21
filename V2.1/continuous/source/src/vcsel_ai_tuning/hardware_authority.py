"""Atomic non-blocking ownership of the shared hardware control plane."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from threading import RLock
import time
from typing import BinaryIO


class HardwareAuthorityBusy(RuntimeError):
    """Raised when another identifiable operation owns the hardware authority."""

    def __init__(self, current_owner: str) -> None:
        self.current_owner = current_owner
        super().__init__(f"hardware authority is busy: {current_owner}")


class HardwareAuthorityLease:
    """One idempotently releasable authority lease."""

    def __init__(self, authority: "HardwareAuthority", owner: str, token: object):
        self._authority = authority
        self._token = token
        self.owner = owner
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self._authority._release(self._token)
        self.released = True

    def __enter__(self) -> "HardwareAuthorityLease":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()


class HardwareAuthority:
    """Thread- and process-safe non-blocking hardware authority."""

    _HARDWARE_LOCK_OFFSET = 0
    _METADATA_GUARD_OFFSET = 1
    _OWNER_OFFSET = 2
    _METADATA_GUARD_TIMEOUT_S = 1.0
    _METADATA_GUARD_RETRY_S = 0.005
    _process_lock = RLock()
    _process_owners: dict[str, tuple[str, object]] = {}

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path).resolve()
        self._path_key = None if self.path is None else os.path.normcase(str(self.path))
        self._active_owner: str | None = None
        self._active_token: object | None = None
        self._handle: BinaryIO | None = None

    @property
    def current_owner(self) -> str | None:
        with self._process_lock:
            return self._active_owner

    def acquire(self, owner: str) -> HardwareAuthorityLease:
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("hardware authority owner must be a nonblank string")
        token = object()
        with self._process_lock:
            if self._active_owner is not None:
                raise HardwareAuthorityBusy(self._active_owner)
            if self._path_key is not None:
                process_owner = self._process_owners.get(self._path_key)
                if process_owner is not None:
                    raise HardwareAuthorityBusy(process_owner[0])
                self._process_owners[self._path_key] = (owner, token)
            try:
                handle = self._acquire_file(owner)
            except Exception:
                self._discard_process_owner(token)
                raise
            self._active_owner = owner
            self._active_token = token
            self._handle = handle
        return HardwareAuthorityLease(self, owner, token)

    def _acquire_file(self, owner: str) -> BinaryIO | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._open_file(self.path)
        try:
            self._initialize_file(handle)
        except BaseException:
            self._close_safely(handle)
            raise
        try:
            self._acquire_metadata_guard(handle)
            guard_locked = True
        except BaseException:
            self._close_safely(handle)
            raise
        hardware_locked = False
        try:
            try:
                self._lock_file(handle)
                hardware_locked = True
            except OSError as error:
                if not self._is_lock_contention(error):
                    raise
                current_owner = self._read_file_owner_safely(handle)
                raise HardwareAuthorityBusy(current_owner) from error
            self._write_file_owner(handle, owner)
            self._unlock_metadata_guard(handle)
            guard_locked = False
        except BaseException:
            if hardware_locked:
                self._unlock_file_safely(handle)
            if guard_locked:
                self._unlock_metadata_guard_safely(handle)
            self._close_safely(handle)
            raise
        return handle

    @staticmethod
    def _open_file(path: Path) -> BinaryIO:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(path, flags, 0o600)
        try:
            return os.fdopen(descriptor, "r+b", buffering=0)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    @staticmethod
    def _initialize_file(handle: BinaryIO) -> None:
        handle.seek(0, os.SEEK_END)
        if handle.tell() >= HardwareAuthority._OWNER_OFFSET:
            return
        reserved = b"\0" * HardwareAuthority._OWNER_OFFSET
        handle.seek(0)
        try:
            written = handle.write(reserved)
        except OSError as error:
            handle.seek(0, os.SEEK_END)
            if (
                HardwareAuthority._is_lock_contention(error)
                and handle.tell() >= HardwareAuthority._OWNER_OFFSET
            ):
                return
            raise
        if written != len(reserved):
            raise OSError(errno.EIO, "hardware authority file initialization failed")
        handle.flush()

    @staticmethod
    def _lock_file(handle: BinaryIO) -> None:
        HardwareAuthority._lock_range(
            handle, HardwareAuthority._HARDWARE_LOCK_OFFSET
        )

    @staticmethod
    def _unlock_file(handle: BinaryIO) -> None:
        HardwareAuthority._unlock_range(
            handle, HardwareAuthority._HARDWARE_LOCK_OFFSET
        )

    @staticmethod
    def _lock_metadata_guard(handle: BinaryIO) -> None:
        HardwareAuthority._lock_range(
            handle, HardwareAuthority._METADATA_GUARD_OFFSET
        )

    @staticmethod
    def _unlock_metadata_guard(handle: BinaryIO) -> None:
        HardwareAuthority._unlock_range(
            handle, HardwareAuthority._METADATA_GUARD_OFFSET
        )

    @staticmethod
    def _lock_range(handle: BinaryIO, offset: int) -> None:
        handle.seek(offset)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.lockf(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
                1,
                offset,
                os.SEEK_SET,
            )

    @staticmethod
    def _unlock_range(handle: BinaryIO, offset: int) -> None:
        handle.seek(offset)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.lockf(
                handle.fileno(),
                fcntl.LOCK_UN,
                1,
                offset,
                os.SEEK_SET,
            )

    @staticmethod
    def _is_lock_contention(error: OSError) -> bool:
        return error.errno in {errno.EACCES, errno.EAGAIN}

    def _acquire_metadata_guard(self, handle: BinaryIO) -> None:
        deadline = time.monotonic() + self._METADATA_GUARD_TIMEOUT_S
        while True:
            try:
                self._lock_metadata_guard(handle)
                return
            except OSError as error:
                if not self._is_lock_contention(error):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HardwareAuthorityBusy(
                        "owner-metadata-transition"
                    ) from error
                time.sleep(min(self._METADATA_GUARD_RETRY_S, remaining))

    @staticmethod
    def _write_file_owner(handle: BinaryIO, owner: str) -> None:
        encoded = owner.encode("utf-8")
        handle.seek(HardwareAuthority._OWNER_OFFSET)
        handle.truncate()
        if handle.write(encoded) != len(encoded):
            raise OSError(errno.EIO, "hardware authority owner write was incomplete")
        handle.flush()
        os.fsync(handle.fileno())

    @staticmethod
    def _read_file_owner_safely(handle: BinaryIO) -> str:
        try:
            handle.seek(HardwareAuthority._OWNER_OFFSET)
            encoded = handle.read()
        except OSError:
            return "another-process"
        if not encoded:
            return "another-process"
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            return "another-process"

    @staticmethod
    def _close_safely(handle: BinaryIO) -> None:
        try:
            handle.close()
        except OSError:
            pass

    def _unlock_file_safely(self, handle: BinaryIO) -> None:
        try:
            self._unlock_file(handle)
        except OSError:
            pass

    def _unlock_metadata_guard_safely(self, handle: BinaryIO) -> None:
        try:
            self._unlock_metadata_guard(handle)
        except OSError:
            pass

    def _release(self, token: object) -> None:
        with self._process_lock:
            if self._active_token is not token:
                return
            handle = self._handle
            if handle is not None:
                self._acquire_metadata_guard(handle)
            self._handle = None
            self._active_owner = None
            self._active_token = None
            try:
                if handle is not None:
                    handle.seek(self._OWNER_OFFSET)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    self._unlock_file(handle)
            finally:
                try:
                    try:
                        if handle is not None:
                            self._unlock_metadata_guard(handle)
                    finally:
                        if handle is not None:
                            handle.close()
                finally:
                    self._discard_process_owner(token)

    def _discard_process_owner(self, token: object) -> None:
        if self._path_key is None:
            return
        owner = self._process_owners.get(self._path_key)
        if owner is not None and owner[1] is token:
            del self._process_owners[self._path_key]
