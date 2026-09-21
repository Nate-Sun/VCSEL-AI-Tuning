"""Ordered, resumable Server-Sent Events for persisted campaign events."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import json
import re
import time
from typing import Protocol

from vcsel_ai_tuning.campaign.models import CampaignState
from vcsel_ai_tuning.campaign.store import CampaignEvent


HEARTBEAT_FRAME = ": heartbeat\n\n"
TERMINAL_STATES = frozenset(
    {
        CampaignState.COMPLETED,
        CampaignState.FAILED,
        CampaignState.RECOVERY_REQUIRED,
    }
)
_CANONICAL_CURSOR = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MAX_SQLITE_INTEGER_TEXT = str((1 << 63) - 1)


class InvalidLastEventId(ValueError):
    """Raised when an SSE resume cursor is not a canonical nonnegative integer."""


class EventSource(Protocol):
    """Read-only event source required by the SSE iterator."""

    def events(
        self, campaign_id: str, *, after_sequence: int = 0
    ) -> Sequence[CampaignEvent]: ...

    def current_state(self, campaign_id: str) -> CampaignState | None: ...


def parse_last_event_id(value: str | None) -> int:
    """Parse an HTTP Last-Event-ID header without permissive coercion."""

    if value is None:
        return 0
    if not isinstance(value, str) or _CANONICAL_CURSOR.fullmatch(value) is None:
        raise InvalidLastEventId(
            "Last-Event-ID must be a canonical non-negative integer"
        )
    if len(value) > len(_MAX_SQLITE_INTEGER_TEXT) or (
        len(value) == len(_MAX_SQLITE_INTEGER_TEXT)
        and value > _MAX_SQLITE_INTEGER_TEXT
    ):
        raise InvalidLastEventId(
            "Last-Event-ID must fit in a signed SQLite integer"
        )
    return int(value, 10)


def _state_value(state: CampaignState | None) -> str | None:
    return None if state is None else state.value


def format_sse_event(event: CampaignEvent) -> str:
    """Serialize one persisted event into a canonical SSE frame."""

    if type(event.sequence) is not int or event.sequence < 1:
        raise ValueError("campaign event sequence must be a positive integer")
    if (
        not isinstance(event.event_type, str)
        or not event.event_type
        or "\r" in event.event_type
        or "\n" in event.event_type
    ):
        raise ValueError("campaign event type must be a safe non-empty SSE field")
    data = {
        "campaign_id": event.campaign_id,
        "created_at": event.created_at.isoformat(),
        "event_type": event.event_type,
        "from_state": _state_value(event.from_state),
        "payload": event.payload,
        "sequence": event.sequence,
        "to_state": _state_value(event.to_state),
    }
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {encoded}\n\n"
    )


def stream_campaign_events(
    source: EventSource,
    campaign_id: str,
    *,
    last_event_id: str | None = None,
    heartbeat_interval_s: float = 15.0,
    wait: Callable[[float], None] = time.sleep,
    is_disconnected: Callable[[], bool] = lambda: False,
) -> Iterator[str]:
    """Yield ordered SSE frames without taking ownership of campaign execution."""

    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    if (
        isinstance(heartbeat_interval_s, bool)
        or not isinstance(heartbeat_interval_s, (int, float))
        or heartbeat_interval_s <= 0
    ):
        raise ValueError("heartbeat_interval_s must be positive")
    cursor = parse_last_event_id(last_event_id)

    while True:
        if is_disconnected():
            return
        batch = source.events(campaign_id, after_sequence=cursor)
        unseen_by_sequence: dict[int, CampaignEvent] = {}
        for event in batch:
            if event.campaign_id != campaign_id:
                raise ValueError("event source returned a different campaign")
            if type(event.sequence) is not int or event.sequence < 1:
                raise ValueError("event source returned an invalid sequence")
            if event.sequence > cursor:
                unseen_by_sequence.setdefault(event.sequence, event)

        if unseen_by_sequence:
            for sequence in sorted(unseen_by_sequence):
                event = unseen_by_sequence[sequence]
                yield format_sse_event(event)
                cursor = sequence
                if event.to_state in TERMINAL_STATES:
                    return

        if source.current_state(campaign_id) in TERMINAL_STATES:
            return

        if unseen_by_sequence:
            continue

        wait(float(heartbeat_interval_s))
        if is_disconnected():
            return
        yield HEARTBEAT_FRAME
