from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
import json

import pytest

from vcsel_ai_tuning.campaign.models import CampaignState
from vcsel_ai_tuning.campaign.store import CampaignEvent


CREATED_AT = datetime(2026, 7, 22, 1, 2, 3, tzinfo=timezone.utc)
SQLITE_MAX_INTEGER = (1 << 63) - 1


@pytest.fixture
def events_module():
    try:
        return import_module("vcsel_ai_tuning.api.events")
    except ModuleNotFoundError as error:
        missing_error = error

        class MissingEventsModule:
            def __getattr__(self, name):
                raise AssertionError(
                    "vcsel_ai_tuning.api.events is missing; "
                    f"cannot access {name}"
                ) from missing_error

        return MissingEventsModule()


def _event(
    sequence: int,
    *,
    event_type: str = "STATE_TRANSITION",
    from_state: CampaignState | None = CampaignState.PREFLIGHT,
    to_state: CampaignState = CampaignState.AWAIT_BASELINE_APPROVAL,
    payload=None,
) -> CampaignEvent:
    return CampaignEvent(
        campaign_id="campaign-1",
        sequence=sequence,
        event_type=event_type,
        from_state=from_state,
        to_state=to_state,
        payload={} if payload is None else payload,
        created_at=CREATED_AT,
    )


class FakeEventSource:
    def __init__(self, batches, *, state=CampaignState.PREFLIGHT):
        self.batches = list(batches)
        self.state = state
        self.calls = []
        self.state_calls = []
        self.backend_running = True
        self.mutations = []

    def events(self, campaign_id, *, after_sequence=0):
        self.calls.append((campaign_id, after_sequence))
        if self.batches:
            return list(self.batches.pop(0))
        return []

    def current_state(self, campaign_id):
        self.state_calls.append(campaign_id)
        return self.state


class FakeWait:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("42", 42)])
def test_last_event_id_accepts_only_canonical_nonnegative_integers(
    events_module, value, expected
):
    assert events_module.parse_last_event_id(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "-1", "+1", "01", "1.0", " 1", "1 ", "one", 1, True],
)
def test_last_event_id_rejects_noncanonical_or_nonstring_values(
    events_module, value
):
    with pytest.raises(events_module.InvalidLastEventId, match="Last-Event-ID"):
        events_module.parse_last_event_id(value)


def test_last_event_id_accepts_maximum_signed_sqlite_integer(events_module):
    assert events_module.parse_last_event_id(str(SQLITE_MAX_INTEGER)) == (
        SQLITE_MAX_INTEGER
    )


@pytest.mark.parametrize(
    "value",
    [str(SQLITE_MAX_INTEGER + 1), "9" * 10_000],
    ids=["sqlite-max-plus-one", "extremely-long-digits"],
)
def test_last_event_id_rejects_values_outside_sqlite_integer_range(
    events_module, value
):
    with pytest.raises(events_module.InvalidLastEventId, match="Last-Event-ID"):
        events_module.parse_last_event_id(value)


def test_event_source_contract_exposes_current_state(events_module):
    assert callable(events_module.EventSource.current_state)


def test_canonical_sse_frame_contains_id_event_and_compact_json(events_module):
    event = _event(
        1,
        event_type="PREFLIGHT_COMPLETED",
        payload={"verified": True},
    )

    frame = events_module.format_sse_event(event)

    expected_data = {
        "campaign_id": "campaign-1",
        "created_at": "2026-07-22T01:02:03+00:00",
        "event_type": "PREFLIGHT_COMPLETED",
        "from_state": "PREFLIGHT",
        "payload": {"verified": True},
        "sequence": 1,
        "to_state": "AWAIT_BASELINE_APPROVAL",
    }
    expected_json = json.dumps(
        expected_data,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert frame == (
        "id: 1\n"
        "event: PREFLIGHT_COMPLETED\n"
        f"data: {expected_json}\n\n"
    )


def test_replay_after_last_event_id_is_ordered_and_deduplicated(events_module):
    source = FakeEventSource(
        [
            [
                _event(3, to_state=CampaignState.COMPLETED),
                _event(2),
                _event(2),
                _event(1),
            ]
        ]
    )

    frames = list(
        events_module.stream_campaign_events(
            source,
            "campaign-1",
            last_event_id="1",
            wait=lambda _seconds: None,
        )
    )

    assert [frame.splitlines()[0] for frame in frames] == ["id: 2", "id: 3"]
    assert source.calls == [("campaign-1", 1)]


def test_overlapping_poll_batches_never_repeat_an_event(events_module):
    source = FakeEventSource(
        [
            [_event(2)],
            [_event(2), _event(3, to_state=CampaignState.FAILED)],
        ]
    )

    frames = list(
        events_module.stream_campaign_events(
            source,
            "campaign-1",
            last_event_id="1",
            wait=lambda _seconds: None,
        )
    )

    assert [frame.splitlines()[0] for frame in frames] == ["id: 2", "id: 3"]
    assert source.calls == [("campaign-1", 1), ("campaign-1", 2)]


def test_idle_stream_emits_heartbeat_then_replays_new_event(events_module):
    source = FakeEventSource(
        [[], [_event(1, to_state=CampaignState.RECOVERY_REQUIRED)]]
    )
    wait = FakeWait()

    frames = list(
        events_module.stream_campaign_events(
            source,
            "campaign-1",
            heartbeat_interval_s=2.5,
            wait=wait,
        )
    )

    assert frames[0] == ": heartbeat\n\n"
    assert frames[1].startswith("id: 1\nevent: STATE_TRANSITION\n")
    assert wait.calls == [2.5]
    assert source.calls == [("campaign-1", 0), ("campaign-1", 0)]
    assert source.state_calls == ["campaign-1"]


@pytest.mark.parametrize("last_event_id", ["5", "6"], ids=["at", "past"])
@pytest.mark.parametrize(
    "terminal_state",
    [CampaignState.COMPLETED, CampaignState.FAILED, CampaignState.RECOVERY_REQUIRED],
)
def test_reconnect_at_or_past_terminal_event_closes_without_heartbeat(
    events_module, last_event_id, terminal_state
):
    source = FakeEventSource([[]], state=terminal_state)
    wait = FakeWait()
    disconnected = False

    def stop_if_unexpected_wait(seconds):
        nonlocal disconnected
        wait(seconds)
        disconnected = True

    frames = list(
        events_module.stream_campaign_events(
            source,
            "campaign-1",
            last_event_id=last_event_id,
            wait=stop_if_unexpected_wait,
            is_disconnected=lambda: disconnected,
        )
    )

    assert frames == []
    assert source.calls == [("campaign-1", int(last_event_id))]
    assert source.state_calls == ["campaign-1"]
    assert wait.calls == []


@pytest.mark.parametrize(
    "terminal_state",
    [CampaignState.COMPLETED, CampaignState.FAILED, CampaignState.RECOVERY_REQUIRED],
)
def test_terminal_campaign_state_closes_without_another_poll(
    events_module, terminal_state
):
    source = FakeEventSource(
        [[_event(1, to_state=terminal_state)], [_event(2)]]
    )

    frames = list(
        events_module.stream_campaign_events(
            source,
            "campaign-1",
            wait=lambda _seconds: None,
        )
    )

    assert len(frames) == 1
    assert frames[0].startswith("id: 1\n")
    assert source.calls == [("campaign-1", 0)]


def test_client_disconnect_stops_only_the_stream_not_backend(events_module):
    source = FakeEventSource([[]])
    disconnected = False

    def wait(_seconds):
        nonlocal disconnected
        disconnected = True

    frames = list(
        events_module.stream_campaign_events(
            source,
            "campaign-1",
            wait=wait,
            is_disconnected=lambda: disconnected,
        )
    )

    assert frames == []
    assert source.backend_running is True
    assert source.mutations == []
    assert source.calls == [("campaign-1", 0)]
