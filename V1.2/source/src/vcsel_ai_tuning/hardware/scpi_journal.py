"""Durable evidence for SCPI operations that may block in a driver call."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
from math import ceil, isfinite
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any
from uuid import uuid4


ContextProvider = Callable[[], Mapping[str, object]]


class ScpiOperationJournal:
    """Append-only JSONL journal with flush and fsync per event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append(self, event: Mapping[str, object]) -> None:
        line = json.dumps(
            dict(event),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())

    def events(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists():
            return ()
        events: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("SCPI journal event must be a JSON object")
            events.append(value)
        return tuple(events)

    def last_in_flight(self) -> dict[str, object] | None:
        active: dict[str, dict[str, object]] = {}
        for event in self.events():
            operation_id = str(event.get("operation_id", ""))
            if not operation_id:
                continue
            if event.get("event") == "STARTED":
                active[operation_id] = event
            elif event.get("event") in {"COMPLETED", "FAILED"}:
                active.pop(operation_id, None)
        if not active:
            return None
        return dict(active[next(reversed(active))])


class ScpiOperationRecorder:
    """Record an operation before and after delegating it."""

    def __init__(
        self,
        journal: ScpiOperationJournal,
        context: ContextProvider,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._journal = journal
        self._context = context
        self._monotonic = monotonic
        self._now_utc = now_utc

    def execute(
        self,
        operation: str,
        command: str | None,
        callable_: Callable[..., Any],
        *args: object,
        **kwargs: object,
    ) -> Any:
        operation_id = uuid4().hex
        started = self._monotonic()
        common: dict[str, object] = {
            "schema_version": "1.0",
            "operation_id": operation_id,
            "operation": operation,
            "command": command,
            **dict(self._context()),
        }
        self._journal.append(
            {
                **common,
                "event": "STARTED",
                "started_at_utc": self._now_utc().isoformat(),
            }
        )
        try:
            result = callable_(*args, **kwargs)
        except BaseException as error:
            self._journal.append(
                {
                    **common,
                    "event": "FAILED",
                    "finished_at_utc": self._now_utc().isoformat(),
                    "elapsed_s": max(0.0, self._monotonic() - started),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            raise
        self._journal.append(
            {
                **common,
                "event": "COMPLETED",
                "finished_at_utc": self._now_utc().isoformat(),
                "elapsed_s": max(0.0, self._monotonic() - started),
                "response": result if isinstance(result, str) else None,
            }
        )
        return result


class JournaledScpiSession:
    """Transparent SCPI session wrapper backed by a durable recorder."""

    def __init__(
        self,
        delegate: object,
        journal: ScpiOperationJournal,
        context: ContextProvider,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._delegate = delegate
        self._context = context
        self._recorder = ScpiOperationRecorder(
            journal, context, monotonic=monotonic, now_utc=now_utc
        )

    @property
    def timeout(self) -> object:
        return getattr(self._delegate, "timeout")

    @timeout.setter
    def timeout(self, value: object) -> None:
        setattr(self._delegate, "timeout", value)

    def _cap_timeout_to_remaining_deadline(self) -> None:
        remaining = self._context().get("remaining_deadline_s")
        if (
            isinstance(remaining, bool)
            or not isinstance(remaining, (int, float))
            or not isfinite(float(remaining))
        ):
            return
        remaining_ms = max(1, ceil(max(0.0, float(remaining)) * 1_000))
        current = getattr(self._delegate, "timeout", None)
        if (
            isinstance(current, (int, float))
            and not isinstance(current, bool)
            and isfinite(float(current))
        ):
            remaining_ms = min(remaining_ms, max(1, int(current)))
        setattr(self._delegate, "timeout", remaining_ms)

    def write(self, command: str) -> object:
        self._cap_timeout_to_remaining_deadline()
        return self._recorder.execute(
            "write", command, getattr(self._delegate, "write"), command
        )

    def query(self, command: str) -> str:
        self._cap_timeout_to_remaining_deadline()
        return self._recorder.execute(
            "query", command, getattr(self._delegate, "query"), command
        )

    def read(self) -> str:
        self._cap_timeout_to_remaining_deadline()
        return self._recorder.execute("read", None, getattr(self._delegate, "read"))

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)
