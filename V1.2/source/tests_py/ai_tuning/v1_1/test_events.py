from datetime import datetime, timezone
import json

import pytest

from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.events import (
    InvalidLastEventId,
    format_task_event,
    parse_last_event_id,
    stream_task_events,
)
from vcsel_ai_tuning.v1_1.repository import TaskEvent


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def event(sequence, state=LifecycleState.RUNNING):
    return TaskEvent(
        task_id="task-1",
        sequence=sequence,
        event_type="STATE_CHANGED",
        from_state=LifecycleState.AUTHORIZED,
        to_state=state,
        payload={"sequence": sequence},
        product_version="1.1.0",
        schema_version=1,
        campaign_type=TaskType.AI_OPTIMIZATION,
        runtime_mode=RuntimeMode.FAKE,
        b0_hash="b" * 64,
        config_fingerprint="c" * 64,
        created_at=NOW,
    )


@pytest.mark.parametrize("value, expected", [(None, 0), ("0", 0), ("42", 42)])
def test_parse_last_event_id(value, expected):
    assert parse_last_event_id(value) == expected


@pytest.mark.parametrize("value", ["", "-1", "+1", "01", "1.0", " 1", "one"])
def test_invalid_last_event_id(value):
    with pytest.raises(InvalidLastEventId):
        parse_last_event_id(value)


def test_sse_frame_contains_v11_audit_metadata():
    frame = format_task_event(event(3))
    data = json.loads(frame.split("data: ", 1)[1])

    assert frame.startswith("id: 3\nevent: STATE_CHANGED\n")
    assert data["product_version"] == "1.1.0"
    assert data["schema_version"] == 1
    assert data["campaign_type"] == "AI_OPTIMIZATION"
    assert data["runtime_mode"] == "FAKE"
    assert data["b0_hash"] == "b" * 64
    assert data["configuration_fingerprint"] == "c" * 64


class Source:
    def __init__(self):
        self.calls = []

    def events(self, task_id, *, after_sequence=0):
        self.calls.append((task_id, after_sequence))
        return [event(4), event(3), event(4, LifecycleState.COMPLETED)]

    def current_state(self, task_id):
        return LifecycleState.COMPLETED


def test_stream_resumes_after_cursor_orders_deduplicates_and_closes_terminal():
    source = Source()
    frames = list(stream_task_events(source, "task-1", last_event_id="2"))

    assert [frame.splitlines()[0] for frame in frames] == ["id: 3", "id: 4"]
    assert source.calls == [("task-1", 2)]
