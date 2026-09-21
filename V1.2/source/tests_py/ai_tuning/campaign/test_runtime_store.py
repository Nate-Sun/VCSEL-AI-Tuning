from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vcsel_ai_tuning.campaign.models import (
    AuthorizationMode,
    Campaign,
    CampaignState,
)
from vcsel_ai_tuning.campaign.store import (
    CampaignStore,
    ConcurrencyError,
    StoreError,
    UnknownCampaignError,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _campaign(
    campaign_id: str,
    *,
    created_at: datetime,
    baseline_hash: str = HASH_A,
) -> Campaign:
    return Campaign(
        campaign_id=campaign_id,
        baseline_id="B0",
        baseline_hash=baseline_hash,
        config_hash=HASH_B,
        authorization_mode=AuthorizationMode.PER_STAGE,
        created_at=created_at,
        updated_at=created_at,
    )


def test_create_campaign_atomically_persists_metadata_state_and_event(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    created_at = datetime(2026, 7, 24, 1, 2, 3, tzinfo=timezone.utc)
    campaign = _campaign("campaign-1", created_at=created_at)

    persisted = store.create_campaign(campaign)

    assert persisted == campaign
    assert store.current_state("campaign-1") is CampaignState.DRAFT
    assert store.get_campaign("campaign-1") == campaign
    event = store.events("campaign-1")[0]
    assert event.event_type == "CAMPAIGN_CREATED"
    assert event.payload == {
        "authorization_mode": "PER_STAGE",
        "baseline_hash": HASH_A,
        "baseline_id": "B0",
        "config_hash": HASH_B,
    }


def test_create_campaign_atomically_persists_immutable_tuning_path(
    tmp_path: Path,
):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    campaign = _campaign(
        "campaign-path",
        created_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    store.create_campaign(campaign, tuning_path="PATH_EXHAUSTIVE")

    assert store.get_campaign_tuning_path(campaign.campaign_id) == "PATH_EXHAUSTIVE"
    assert [event.event_type for event in store.events(campaign.campaign_id)] == [
        "CAMPAIGN_CREATED",
        "CAMPAIGN_PATH_SELECTED",
    ]
    with pytest.raises(ConcurrencyError, match="tuning path conflict"):
        store.create_campaign(campaign, tuning_path="PATH_TDECQ_BAYESIAN")


def test_create_campaign_is_idempotent_for_identical_metadata(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    campaign = _campaign(
        "campaign-1",
        created_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    store.create_campaign(campaign)
    store.save_transition(
        campaign.campaign_id,
        CampaignState.DRAFT,
        CampaignState.PREFLIGHT,
    )

    repeated = store.create_campaign(campaign)

    assert repeated.state is CampaignState.PREFLIGHT
    assert len(store.events(campaign.campaign_id)) == 2


def test_create_campaign_rejects_conflicting_immutable_metadata(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    created_at = datetime(2026, 7, 24, tzinfo=timezone.utc)
    store.create_campaign(_campaign("campaign-1", created_at=created_at))

    with pytest.raises(ConcurrencyError, match="metadata conflict"):
        store.create_campaign(
            _campaign(
                "campaign-1",
                created_at=created_at,
                baseline_hash="c" * 64,
            )
        )


def test_get_campaign_tracks_persisted_state_and_update_timestamp(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    created_at = datetime(2026, 7, 24, tzinfo=timezone.utc)
    updated_at = created_at + timedelta(minutes=1)
    store.create_campaign(_campaign("campaign-1", created_at=created_at))
    store.save_transition(
        "campaign-1",
        CampaignState.DRAFT,
        CampaignState.PREFLIGHT,
        created_at=updated_at,
    )

    campaign = store.get_campaign("campaign-1")

    assert campaign.state is CampaignState.PREFLIGHT
    assert campaign.created_at == created_at
    assert campaign.updated_at == updated_at


def test_get_campaign_rejects_unknown_campaign(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")

    with pytest.raises(UnknownCampaignError, match="unknown campaign"):
        store.get_campaign("missing")


def test_list_campaigns_has_stable_pagination_without_observations(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    created_at = datetime(2026, 7, 24, tzinfo=timezone.utc)
    for campaign_id, minutes in (
        ("campaign-c", 2),
        ("campaign-a", 0),
        ("campaign-b", 1),
    ):
        store.create_campaign(
            _campaign(campaign_id, created_at=created_at + timedelta(minutes=minutes))
        )

    first_page = store.list_campaigns(limit=2, offset=0)
    second_page = store.list_campaigns(limit=2, offset=2)

    assert [item.campaign_id for item in first_page] == ["campaign-c", "campaign-b"]
    assert [item.campaign_id for item in second_page] == ["campaign-a"]


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(0, 0), (101, 0), (1, -1), (True, 0), (1, False)],
)
def test_list_campaigns_validates_pagination(
    limit: object,
    offset: object,
    tmp_path: Path,
):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")

    with pytest.raises(StoreError, match="limit|offset"):
        store.list_campaigns(limit=limit, offset=offset)


def test_one_store_serializes_complete_transactions_across_worker_threads(
    tmp_path: Path,
):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    created_at = datetime(2026, 7, 24, tzinfo=timezone.utc)

    def create(index: int) -> str:
        campaign_id = f"campaign-{index:02d}"
        store.create_campaign(
            _campaign(campaign_id, created_at=created_at + timedelta(seconds=index))
        )
        store.append_event(
            campaign_id,
            "CONCURRENT_WRITE_VERIFIED",
            expected_state=CampaignState.DRAFT,
        )
        return store.get_campaign(campaign_id).campaign_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        created = tuple(executor.map(create, range(32)))

    assert created == tuple(f"campaign-{index:02d}" for index in range(32))
    assert len(store.list_campaigns(limit=100)) == 32
