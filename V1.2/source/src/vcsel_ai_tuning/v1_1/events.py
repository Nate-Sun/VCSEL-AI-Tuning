"""Ordered, resumable SSE formatting for persisted V1.1 task events."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

from .domain import LifecycleState
from .repository import TaskEvent


HEARTBEAT_FRAME = ": heartbeat\n\n"
STREAM_END_STATES = frozenset(
    {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
        LifecycleState.ABORTED,
        LifecycleState.ABORTED_RECOVERED,
        LifecycleState.FAILED,
        LifecycleState.FAILED_INVALID_DCA_TAPS,
        LifecycleState.RECOVERY_REQUIRED,
    }
)
_CANONICAL_CURSOR = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MAX_SQLITE_INTEGER = (1 << 63) - 1


class InvalidLastEventId(ValueError):
    pass


class TaskEventSource(Protocol):
    def events(
        self, task_id: str, *, after_sequence: int = 0
    ) -> Sequence[TaskEvent]: ...

    def current_state(self, task_id: str) -> LifecycleState | None: ...


def parse_last_event_id(value: str | None) -> int:
    if value is None:
        return 0
    if not isinstance(value, str) or _CANONICAL_CURSOR.fullmatch(value) is None:
        raise InvalidLastEventId(
            "Last-Event-ID must be a canonical non-negative integer"
        )
    result = int(value, 10)
    if result > _MAX_SQLITE_INTEGER:
        raise InvalidLastEventId("Last-Event-ID exceeds SQLite integer range")
    return result


def format_task_event(event: TaskEvent) -> str:
    if not isinstance(event, TaskEvent):
        raise TypeError("event must be a TaskEvent")
    if event.sequence < 1:
        raise ValueError("event sequence must be positive")
    if not event.event_type or "\n" in event.event_type or "\r" in event.event_type:
        raise ValueError("event_type is not safe for SSE")
    payload = {
        "task_id": event.task_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "from_state": (
            None if event.from_state is None else event.from_state.value
        ),
        "to_state": event.to_state.value,
        "payload": event.payload,
        "product_version": event.product_version,
        "schema_version": event.schema_version,
        "campaign_type": event.campaign_type.value,
        "runtime_mode": event.runtime_mode.value,
        "b0_hash": event.b0_hash,
        "configuration_fingerprint": event.config_fingerprint,
        "created_at": event.created_at.isoformat(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {encoded}\n\n"
    )


def stream_task_events(
    source: TaskEventSource,
    task_id: str,
    *,
    last_event_id: str | None = None,
    heartbeat_interval_s: float = 2.0,
    wait: Callable[[float], None] = time.sleep,
    is_disconnected: Callable[[], bool] = lambda: False,
) -> Iterator[str]:
    cursor = parse_last_event_id(last_event_id)
    while not is_disconnected():
        batch = sorted(
            source.events(task_id, after_sequence=cursor),
            key=lambda event: event.sequence,
        )
        unique: dict[int, TaskEvent] = {}
        for event in batch:
            if event.sequence > cursor:
                unique.setdefault(event.sequence, event)
        for sequence in sorted(unique):
            event = unique[sequence]
            cursor = sequence
            yield format_task_event(event)
            if event.to_state in STREAM_END_STATES:
                return
        if source.current_state(task_id) in STREAM_END_STATES:
            return
        if unique:
            continue
        yield HEARTBEAT_FRAME
        wait(heartbeat_interval_s)
