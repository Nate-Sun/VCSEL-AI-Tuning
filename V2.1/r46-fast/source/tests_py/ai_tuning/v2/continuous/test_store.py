from __future__ import annotations

import json

import pytest

from vcsel_ai_tuning.v2.continuous.store import (
    ContinuousCampaignStore,
    ContinuousStoreError,
)


def test_store_creates_isolated_layout_and_round_paths(tmp_path) -> None:
    store = ContinuousCampaignStore(tmp_path)
    store.initialize("campaign-1", {"campaign_id": "campaign-1", "mode": "FAKE"})
    store.save_state("campaign-1", {"state": "RUNNING", "round": 0})
    store.append_event("campaign-1", "STARTED", {"round": 0})
    store.save_round("campaign-1", 1, {"round": 1})
    store.save_acceptance("campaign-1", {"passed": True})
    store.save_recovery("campaign-1", {"restored": True})

    root = tmp_path / "campaign-1"
    assert json.loads((root / "campaign.json").read_text(encoding="utf-8"))["mode"] == "FAKE"
    assert store.load_state("campaign-1") == {"state": "RUNNING", "round": 0}
    assert (root / "events.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert json.loads((root / "rounds" / "0001" / "round.json").read_text(encoding="utf-8")) == {"round": 1}
    assert (root / "acceptance" / "acceptance.json").is_file()
    assert (root / "recovery" / "recovery.json").is_file()


def test_store_scrubs_secrets_recursively(tmp_path) -> None:
    store = ContinuousCampaignStore(tmp_path)
    store.initialize(
        "campaign-1",
        {
            "MOONSHOT_API_KEY": "top-secret",
            "nested": {
                "api_key": "secret",
                "authorization": "Bearer abc",
                "safe": "keep",
            },
        },
    )

    text = (tmp_path / "campaign-1" / "campaign.json").read_text(encoding="utf-8")
    assert "top-secret" not in text
    assert "Bearer abc" not in text
    assert "secret" not in text
    assert text.count("[REDACTED]") == 3
    assert "keep" in text


def test_store_rejects_path_escape_and_conflicting_reinitialization(tmp_path) -> None:
    store = ContinuousCampaignStore(tmp_path)
    with pytest.raises(ContinuousStoreError, match="campaign_id"):
        store.initialize("../outside", {"campaign_id": "outside"})

    store.initialize("campaign-1", {"campaign_id": "campaign-1", "mode": "FAKE"})
    store.initialize("campaign-1", {"campaign_id": "campaign-1", "mode": "FAKE"})
    with pytest.raises(ContinuousStoreError, match="conflicting"):
        store.initialize("campaign-1", {"campaign_id": "campaign-1", "mode": "LIVE"})


def test_store_rejects_historical_metadata_without_target_with_migration_error(
    tmp_path,
) -> None:
    store = ContinuousCampaignStore(tmp_path)
    store.initialize(
        "campaign-1",
        {"campaign_id": "campaign-1", "mode": "FAKE", "max_rounds": 3},
    )

    with pytest.raises(
        ContinuousStoreError,
        match="target_tdecq_db.*new campaign|migrat",
    ):
        store.initialize(
            "campaign-1",
            {
                "campaign_id": "campaign-1",
                "mode": "FAKE",
                "max_rounds": 3,
                "target_tdecq_db": 2.8,
            },
        )


def test_failed_atomic_replace_preserves_previous_state(tmp_path, monkeypatch) -> None:
    store = ContinuousCampaignStore(tmp_path)
    store.initialize("campaign-1", {"campaign_id": "campaign-1"})
    store.save_state("campaign-1", {"state": "RUNNING", "round": 1})

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("vcsel_ai_tuning.v2.continuous.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.save_state("campaign-1", {"state": "COMPLETED", "round": 2})

    assert store.load_state("campaign-1") == {"state": "RUNNING", "round": 1}


def test_truncated_event_log_fails_reload_deterministically(tmp_path) -> None:
    store = ContinuousCampaignStore(tmp_path)
    store.initialize("campaign-1", {"campaign_id": "campaign-1"})
    store.append_event("campaign-1", "STARTED", {"round": 0})
    path = tmp_path / "campaign-1" / "events.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"event_type":')

    with pytest.raises(ContinuousStoreError, match="event log"):
        store.load_events("campaign-1")
