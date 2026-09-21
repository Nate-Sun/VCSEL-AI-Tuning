from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from vcsel_ai_tuning.campaign.authorization import (
    rollback_hash_for,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    ParameterSpec,
    StageAuthorization,
)
from vcsel_ai_tuning.campaign.store import (
    CampaignStore,
    ConcurrencyError,
    StoreClosedError,
    StoreError,
    UnknownCampaignError,
)


ACTIVE_RESTART_STATES = (
    CampaignState.RESTORE_B0,
    CampaignState.BASELINE_MEASUREMENT,
    CampaignState.I2C_SCREENING,
    CampaignState.I2C_COARSE,
    CampaignState.I2C_FINE,
    CampaignState.FIR_APPLY,
    CampaignState.I2C_RETUNE,
    CampaignState.FINAL_VERIFICATION,
    CampaignState.ROLLING_BACK,
    CampaignState.ABORTING,
)

STABLE_RESTART_STATES = (
    CampaignState.DRAFT,
    CampaignState.PREFLIGHT,
    CampaignState.AWAIT_BASELINE_APPROVAL,
    CampaignState.AWAIT_COARSE_APPROVAL,
    CampaignState.AWAIT_FINE_APPROVAL,
    CampaignState.AWAIT_FIR_APPROVAL,
    CampaignState.PAUSED,
    CampaignState.COMPLETED,
    CampaignState.FAILED,
    CampaignState.RECOVERY_REQUIRED,
)

CAMPAIGN_ID = "campaign-authorization-1"
LEASE_OWNER = "campaign-worker-1"
NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def _stage_authorization(
    *,
    stage: CampaignState = CampaignState.I2C_COARSE,
    max_candidates: int = 2,
    max_duration_s: float = 10.0,
) -> StageAuthorization:
    rollback = AppliedSettings(
        requested_i2c={"ibias_dac": 70},
        quantized_i2c={"ibias_dac": 70},
        applied_i2c={"ibias_dac": 70},
        readback_i2c={"ibias_dac": 70},
    )
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=AuthorizationMode.PER_STAGE,
        stage=stage,
        baseline_hash="b" * 64,
        rollback_hash=rollback_hash_for(rollback),
        parameter_specs=(
            ParameterSpec(
                name="ibias_dac",
                encoding_min=0,
                encoding_max=127,
                tuning_min=68,
                tuning_max=74,
                max_step=2,
            ),
        ),
        max_candidates=max_candidates,
        max_duration_s=max_duration_s,
        issued_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(hours=1),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def test_save_state_and_transition_persist_state_and_ordered_events(tmp_path):
    database = tmp_path / "campaigns.sqlite3"

    with CampaignStore(database) as store:
        saved = store.save_state("campaign-1", CampaignState.DRAFT)
        transitioned = store.save_transition(
            "campaign-1",
            CampaignState.DRAFT,
            CampaignState.PREFLIGHT,
            payload={"operator": "reviewer-1"},
        )

        assert store.current_state("campaign-1") == CampaignState.PREFLIGHT
        assert [event.sequence for event in store.events("campaign-1")] == [1, 2]
        assert saved.event_type == "STATE_SAVED"
        assert saved.from_state is None
        assert saved.to_state == CampaignState.DRAFT
        assert transitioned.from_state == CampaignState.DRAFT
        assert transitioned.to_state == CampaignState.PREFLIGHT
        assert transitioned.payload == {"operator": "reviewer-1"}
        assert transitioned.created_at.utcoffset() == timedelta(0)


def test_append_event_is_atomic_ordered_and_keeps_campaign_state_unchanged(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", CampaignState.PREFLIGHT)

        appended = store.append_event(
            "campaign-1",
            "PREFLIGHT_READBACK_VERIFIED",
            payload={"verified": True},
            expected_state=CampaignState.PREFLIGHT,
        )

        assert store.current_state("campaign-1") == CampaignState.PREFLIGHT
        assert appended.sequence == 2
        assert appended.event_type == "PREFLIGHT_READBACK_VERIFIED"
        assert appended.from_state == CampaignState.PREFLIGHT
        assert appended.to_state == CampaignState.PREFLIGHT
        assert appended.payload == {"verified": True}
        assert [event.sequence for event in store.events("campaign-1")] == [1, 2]


def test_append_event_expected_state_conflict_appends_nothing(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", CampaignState.PREFLIGHT)

        with pytest.raises(ConcurrencyError, match="expected.*DRAFT.*PREFLIGHT"):
            store.append_event(
                "campaign-1",
                "STALE_AUDIT_EVENT",
                expected_state=CampaignState.DRAFT,
            )

        assert [event.event_type for event in store.events("campaign-1")] == [
            "STATE_SAVED"
        ]


def test_append_event_requires_existing_campaign_even_without_expected_state(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        with pytest.raises(UnknownCampaignError, match="missing"):
            store.append_event("missing", "ORPHAN_EVENT")


def test_event_sequences_start_at_one_and_are_independent_per_campaign(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", "DRAFT")
        store.save_transition("campaign-1", "DRAFT", "PREFLIGHT")
        store.save_state("campaign-2", "DRAFT")

        assert [event.sequence for event in store.events("campaign-1")] == [1, 2]
        assert [event.sequence for event in store.events("campaign-2")] == [1]
        assert store.current_state("campaign-2") == CampaignState.DRAFT


def test_events_after_sequence_are_strictly_ordered_for_sse_resume(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", "DRAFT")
        store.save_transition("campaign-1", "DRAFT", "PREFLIGHT")
        store.save_transition(
            "campaign-1", "PREFLIGHT", "AWAIT_BASELINE_APPROVAL"
        )

        events = store.events("campaign-1", after_sequence=1)

        assert [event.sequence for event in events] == [2, 3]
        assert [event.to_state for event in events] == [
            CampaignState.PREFLIGHT,
            CampaignState.AWAIT_BASELINE_APPROVAL,
        ]


def test_expected_current_state_conflict_prevents_lost_update(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", "DRAFT")

        with pytest.raises(ConcurrencyError, match="expected.*PREFLIGHT.*DRAFT"):
            store.save_transition(
                "campaign-1", "PREFLIGHT", "AWAIT_BASELINE_APPROVAL"
            )

        assert store.current_state("campaign-1") == CampaignState.DRAFT
        assert len(store.events("campaign-1")) == 1


@pytest.mark.parametrize(
    ("initial_state", "replacement_state"),
    [
        (CampaignState.DRAFT, CampaignState.COMPLETED),
        (CampaignState.RECOVERY_REQUIRED, CampaignState.FIR_APPLY),
    ],
)
def test_save_state_is_insert_only_bootstrap(
    tmp_path, initial_state, replacement_state
):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", initial_state)

        with pytest.raises(ConcurrencyError, match="exists|bootstrap"):
            store.save_state("campaign-1", replacement_state)

        assert store.current_state("campaign-1") == initial_state
        events = store.events("campaign-1")
        assert len(events) == 1
        assert events[0].event_type == "STATE_SAVED"
        assert events[0].sequence == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_payload_is_rejected_without_partial_state_update(tmp_path, invalid):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", "DRAFT")

        with pytest.raises(StoreError, match="payload.*JSON"):
            store.save_transition(
                "campaign-1", "DRAFT", "PREFLIGHT", payload={"metric": invalid}
            )

        assert store.current_state("campaign-1") == CampaignState.DRAFT
        assert len(store.events("campaign-1")) == 1


def test_event_insert_failure_rolls_back_state_update_in_same_transaction(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with CampaignStore(database) as store:
        store.save_state("campaign-1", "DRAFT")
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_test_event
                BEFORE INSERT ON campaign_events
                WHEN NEW.event_type = 'FAIL_EVENT'
                BEGIN
                    SELECT RAISE(ABORT, 'forced event failure');
                END
                """
            )

        with pytest.raises(StoreError, match="save transition"):
            store.save_transition(
                "campaign-1",
                "DRAFT",
                "PREFLIGHT",
                event_type="FAIL_EVENT",
            )

        assert store.current_state("campaign-1") == CampaignState.DRAFT
        assert len(store.events("campaign-1")) == 1


def test_payload_json_is_stably_serialized(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with CampaignStore(database) as store:
        store.save_state("campaign-1", "DRAFT")
        store.save_transition(
            "campaign-1",
            "DRAFT",
            "PREFLIGHT",
            payload={"z": 1, "a": [2, 3]},
        )

    with sqlite3.connect(database) as connection:
        payload_json = connection.execute(
            "SELECT payload_json FROM campaign_events "
            "WHERE campaign_id = ? AND sequence = ?",
            ("campaign-1", 2),
        ).fetchone()[0]

    assert payload_json == '{"a":[2,3],"z":1}'


@pytest.mark.parametrize("active_state", ACTIVE_RESTART_STATES)
def test_restart_recovery_moves_active_states_to_recovery_required(
    tmp_path, active_state
):
    with CampaignStore(tmp_path / f"{active_state}.sqlite3") as store:
        store.save_state("campaign-1", active_state)

        assert store.recover_state("campaign-1") == CampaignState.RECOVERY_REQUIRED
        assert store.current_state("campaign-1") == CampaignState.RECOVERY_REQUIRED
        recovery_event = store.events("campaign-1")[-1]
        assert recovery_event.event_type == "RESTART_RECOVERY"
        assert recovery_event.from_state == active_state
        assert recovery_event.to_state == CampaignState.RECOVERY_REQUIRED


@pytest.mark.parametrize("stable_state", STABLE_RESTART_STATES)
def test_restart_recovery_leaves_stable_states_unchanged_without_event(
    tmp_path, stable_state
):
    with CampaignStore(tmp_path / f"{stable_state}.sqlite3") as store:
        store.save_state("campaign-1", stable_state)
        before = store.events("campaign-1")

        assert store.recover_state("campaign-1") == stable_state
        assert store.events("campaign-1") == before


def test_restart_recovery_is_idempotent(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        store.save_state("campaign-1", "FIR_APPLY")

        assert store.recover_state("campaign-1") == "RECOVERY_REQUIRED"
        assert store.recover_state("campaign-1") == "RECOVERY_REQUIRED"
        assert [event.event_type for event in store.events("campaign-1")] == [
            "STATE_SAVED",
            "RESTART_RECOVERY",
        ]


def test_unknown_campaign_is_not_silently_recovered(tmp_path):
    with CampaignStore(tmp_path / "campaigns.sqlite3") as store:
        assert store.current_state("missing") is None
        with pytest.raises(UnknownCampaignError, match="missing"):
            store.recover_state("missing")


def test_close_is_idempotent_and_closed_store_rejects_operations(tmp_path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    store.close()
    store.close()

    with pytest.raises(StoreClosedError, match="closed"):
        store.current_state("campaign-1")


def test_database_state_and_events_survive_reopen(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with CampaignStore(database) as store:
        store.save_state("campaign-1", "DRAFT")
        store.save_transition("campaign-1", "DRAFT", "PREFLIGHT")

    with CampaignStore(database) as reopened:
        assert reopened.current_state("campaign-1") == CampaignState.PREFLIGHT
        assert [event.sequence for event in reopened.events("campaign-1")] == [1, 2]


def test_two_connections_allow_only_one_bootstrap_for_same_campaign(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with CampaignStore(database):
        pass
    barrier = Barrier(2)

    def save_from_connection(worker: int) -> tuple[str, int | None]:
        with CampaignStore(database) as store:
            barrier.wait()
            try:
                event = store.save_state(
                    "campaign-1", "DRAFT", payload={"worker": worker}
                )
            except ConcurrencyError:
                return ("conflict", None)
            return ("saved", event.sequence)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save_from_connection, (1, 2)))

    with CampaignStore(database) as store:
        assert sorted(status for status, _ in results) == ["conflict", "saved"]
        assert [sequence for _, sequence in results if sequence is not None] == [1]
        assert store.current_state("campaign-1") == CampaignState.DRAFT
        events = store.events("campaign-1")
        assert [event.sequence for event in events] == [1]
        assert [event.event_type for event in events] == ["STATE_SAVED"]


def test_observation_sequence_allocation_is_unique_across_connections(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with CampaignStore(database) as store:
        store.save_state("campaign-1", CampaignState.I2C_COARSE)
    barrier = Barrier(2)

    def allocate_from_connection(_: int) -> int:
        with CampaignStore(database) as store:
            barrier.wait()
            return store.allocate_observation_sequence("campaign-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        allocated = list(executor.map(allocate_from_connection, (1, 2)))

    assert sorted(allocated) == [1, 2]
    with CampaignStore(database) as reopened:
        assert reopened.allocate_observation_sequence("campaign-1") == 3


def test_authorized_transition_persists_full_authorization_and_usage(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    authorization = _stage_authorization()

    with CampaignStore(database) as store:
        store.save_state(
            CAMPAIGN_ID,
            CampaignState.AWAIT_COARSE_APPROVAL,
            created_at=NOW - timedelta(seconds=1),
        )
        activated = store.save_authorized_transition(
            CAMPAIGN_ID,
            CampaignState.AWAIT_COARSE_APPROVAL,
            CampaignState.I2C_COARSE,
            authorization,
            event_type="AUTHORIZED_STAGE_ENTERED",
            activated_at=NOW,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )
        ordinal = store.reserve_candidate(
            CAMPAIGN_ID,
            authorization.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=1),
        )

        assert ordinal == 0
        assert activated.from_state == CampaignState.AWAIT_COARSE_APPROVAL
        assert activated.to_state == CampaignState.I2C_COARSE
        assert activated.created_at == NOW
        assert activated.payload["authorization"] == authorization.model_dump(
            mode="json"
        )

    with CampaignStore(database) as reopened:
        usage = reopened.authorization_usage(
            CAMPAIGN_ID,
            authorization.authorization_id,
            now=NOW + timedelta(seconds=5),
        )
        assert usage.campaign_id == CAMPAIGN_ID
        assert usage.authorization == authorization
        assert usage.activated_at == NOW
        assert usage.candidate_count == 1
        assert usage.elapsed_s == 5.0

        audit_event = next(
            event
            for event in reopened.events(CAMPAIGN_ID)
            if event.event_type == "AUTHORIZED_STAGE_ENTERED"
        )
        assert audit_event.event_type == "AUTHORIZED_STAGE_ENTERED"
        assert audit_event.payload["authorization"] == authorization.model_dump(
            mode="json"
        )


def test_authorized_transition_rolls_back_authorization_if_event_insert_fails(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    authorization = _stage_authorization()

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.AWAIT_COARSE_APPROVAL)
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_authorization_event
                BEFORE INSERT ON campaign_events
                WHEN NEW.event_type = 'AUTHORIZED_STAGE_ENTERED'
                BEGIN
                    SELECT RAISE(ABORT, 'forced authorization event failure');
                END
                """
            )

        with pytest.raises(StoreError, match="authorized transition"):
            store.save_authorized_transition(
                CAMPAIGN_ID,
                CampaignState.AWAIT_COARSE_APPROVAL,
                CampaignState.I2C_COARSE,
                authorization,
                event_type="AUTHORIZED_STAGE_ENTERED",
                activated_at=NOW,
            )

        assert (
            store.current_state(CAMPAIGN_ID)
            == CampaignState.AWAIT_COARSE_APPROVAL
        )
        assert [event.event_type for event in store.events(CAMPAIGN_ID)] == [
            "STATE_SAVED"
        ]
        with pytest.raises(StoreError, match="authorization"):
            store.authorization_usage(
                CAMPAIGN_ID,
                authorization.authorization_id,
                now=NOW,
            )


def test_activate_authorization_in_active_stage_persists_exact_payload(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    authorization = _stage_authorization(stage=CampaignState.I2C_FINE)
    conflicting = _stage_authorization(
        stage=CampaignState.I2C_FINE,
        max_candidates=3,
    )

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.I2C_FINE)
        activated = store.activate_authorization(
            CAMPAIGN_ID,
            authorization,
            event_type="STAGE_AUTHORIZATION_ACTIVATED",
            activated_at=NOW,
        )

        assert activated.from_state == CampaignState.I2C_FINE
        assert activated.to_state == CampaignState.I2C_FINE
        assert activated.payload["authorization"] == authorization.model_dump(
            mode="json"
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            authorization.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=1),
        ) == 0

        store.activate_authorization(
            CAMPAIGN_ID,
            authorization,
            event_type="STAGE_AUTHORIZATION_REAFFIRMED",
            activated_at=NOW + timedelta(seconds=5),
        )
        with pytest.raises(ConcurrencyError, match="conflicting active authorization"):
            store.activate_authorization(
                CAMPAIGN_ID,
                conflicting,
                activated_at=NOW + timedelta(seconds=6),
            )

    with CampaignStore(database) as reopened:
        usage = reopened.authorization_usage(
            CAMPAIGN_ID,
            authorization.authorization_id,
            now=NOW + timedelta(seconds=10),
        )
        assert usage.authorization == authorization
        assert usage.activated_at == NOW
        assert usage.candidate_count == 1
        assert usage.elapsed_s == 10.0


def test_activate_authorization_explicitly_replaces_promoted_nested_auth(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    outer = _stage_authorization(stage=CampaignState.FIR_APPLY)
    nested = _stage_authorization(
        stage=CampaignState.I2C_RETUNE,
        max_candidates=3,
    )
    standalone = _stage_authorization(
        stage=CampaignState.I2C_RETUNE,
        max_candidates=4,
    )

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.FIR_APPLY)
        store.activate_authorization(CAMPAIGN_ID, outer, activated_at=NOW)
        store.activate_nested_authorization(
            CAMPAIGN_ID,
            outer.authorization_id,
            nested,
            activated_at=NOW,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            nested.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=1),
        ) == 0
        store.save_authorized_transition(
            CAMPAIGN_ID,
            CampaignState.FIR_APPLY,
            CampaignState.I2C_RETUNE,
            nested,
            activated_at=NOW + timedelta(seconds=2),
        )

        replacement_event = store.activate_authorization(
            CAMPAIGN_ID,
            standalone,
            replace_existing=True,
            activated_at=NOW + timedelta(seconds=3),
        )

        assert replacement_event.payload["authorization"] == standalone.model_dump(
            mode="json"
        )
        with pytest.raises(StoreError, match="active authorization"):
            store.reserve_candidate(
                CAMPAIGN_ID,
                nested.authorization_id,
                LEASE_OWNER,
                now=NOW + timedelta(seconds=4),
            )
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            standalone.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=4),
        ) == 0
        assert store.authorization_usage(
            CAMPAIGN_ID,
            nested.authorization_id,
            now=NOW + timedelta(seconds=4),
        ).candidate_count == 1
        standalone_usage = store.authorization_usage(
            CAMPAIGN_ID,
            standalone.authorization_id,
            now=NOW + timedelta(seconds=4),
        )
        assert standalone_usage.candidate_count == 1
        assert standalone_usage.activated_at == NOW + timedelta(seconds=3)


def test_candidate_reservation_enforces_persisted_authorization_budgets(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    candidate_limited = _stage_authorization(
        max_candidates=2,
        max_duration_s=10.0,
    )

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)
        store.activate_authorization(
            CAMPAIGN_ID,
            candidate_limited,
            activated_at=NOW,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )

        inactive = _stage_authorization(max_candidates=3)
        with pytest.raises(StoreError, match="active authorization"):
            store.reserve_candidate(
                CAMPAIGN_ID,
                inactive.authorization_id,
                LEASE_OWNER,
                now=NOW + timedelta(seconds=1),
            )

        assert store.reserve_candidate(
            CAMPAIGN_ID,
            candidate_limited.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=1),
        ) == 0
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            candidate_limited.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=2),
        ) == 1
        with pytest.raises(StoreError, match="max_candidates"):
            store.reserve_candidate(
                CAMPAIGN_ID,
                candidate_limited.authorization_id,
                LEASE_OWNER,
                now=NOW + timedelta(seconds=3),
            )
        assert store.authorization_usage(
            CAMPAIGN_ID,
            candidate_limited.authorization_id,
            now=NOW + timedelta(seconds=3),
        ).candidate_count == 2

    duration_database = tmp_path / "duration-campaigns.sqlite3"
    with CampaignStore(duration_database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)
        duration_limited = _stage_authorization(
            max_candidates=5,
            max_duration_s=1.0,
        )
        second_activation = NOW + timedelta(seconds=20)
        store.activate_authorization(
            CAMPAIGN_ID,
            duration_limited,
            activated_at=second_activation,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=second_activation,
        )
        with pytest.raises(StoreError, match="max_duration"):
            store.reserve_candidate(
                CAMPAIGN_ID,
                duration_limited.authorization_id,
                LEASE_OWNER,
                now=second_activation + timedelta(seconds=1),
            )
        duration_usage = store.authorization_usage(
            CAMPAIGN_ID,
            duration_limited.authorization_id,
            now=second_activation + timedelta(seconds=1),
        )
        assert duration_usage.candidate_count == 0
        assert duration_usage.elapsed_s == 1.0


def test_concurrent_candidate_reservations_get_unique_zero_based_ordinals(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    authorization = _stage_authorization(max_candidates=4)
    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)
        store.activate_authorization(
            CAMPAIGN_ID,
            authorization,
            activated_at=NOW,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )
    barrier = Barrier(2)

    def reserve_from_connection(_: int) -> int:
        with CampaignStore(database) as store:
            barrier.wait()
            return store.reserve_candidate(
                CAMPAIGN_ID,
                authorization.authorization_id,
                LEASE_OWNER,
                now=NOW + timedelta(seconds=1),
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        ordinals = list(executor.map(reserve_from_connection, (1, 2)))

    assert sorted(ordinals) == [0, 1]
    with CampaignStore(database) as reopened:
        with pytest.raises(ConcurrencyError, match="lease"):
            reopened.reserve_candidate(
                CAMPAIGN_ID,
                authorization.authorization_id,
                "different-worker",
                now=NOW + timedelta(seconds=2),
            )
        usage = reopened.authorization_usage(
            CAMPAIGN_ID,
            authorization.authorization_id,
            now=NOW + timedelta(seconds=2),
        )
        assert usage.candidate_count == 2


def test_execution_lease_is_exclusive_owner_released_and_cleared_by_recovery(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)
    barrier = Barrier(2)

    def acquire_from_connection(worker: int) -> tuple[str, str]:
        owner = f"campaign-worker-{worker}"
        with CampaignStore(database) as store:
            barrier.wait()
            try:
                store.acquire_execution_lease(
                    CAMPAIGN_ID,
                    owner,
                    acquired_at=NOW,
                )
            except ConcurrencyError:
                return ("conflict", owner)
            return ("acquired", owner)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(acquire_from_connection, (1, 2)))

    assert sorted(status for status, _ in results) == ["acquired", "conflict"]
    owner = next(owner for status, owner in results if status == "acquired")
    loser = next(owner for status, owner in results if status == "conflict")

    with CampaignStore(database) as store:
        with pytest.raises(ConcurrencyError, match="lease"):
            store.release_execution_lease(CAMPAIGN_ID, loser)
        store.release_execution_lease(CAMPAIGN_ID, owner)
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            loser,
            acquired_at=NOW + timedelta(seconds=1),
        )

        assert store.recover_state(CAMPAIGN_ID) == CampaignState.RECOVERY_REQUIRED
        with pytest.raises(ConcurrencyError, match="lease"):
            store.release_execution_lease(CAMPAIGN_ID, loser)


def test_nested_retune_and_outer_authorizations_track_independent_usage(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    outer = _stage_authorization(
        stage=CampaignState.FIR_APPLY,
        max_candidates=2,
    )
    nested = _stage_authorization(
        stage=CampaignState.I2C_RETUNE,
        max_candidates=3,
    )

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.FIR_APPLY)
        store.activate_authorization(CAMPAIGN_ID, outer, activated_at=NOW)
        nested_event = store.activate_nested_authorization(
            CAMPAIGN_ID,
            outer.authorization_id,
            nested,
            activated_at=NOW + timedelta(seconds=1),
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )

        assert store.reserve_candidate(
            CAMPAIGN_ID,
            outer.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=2),
        ) == 0
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            nested.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=2),
        ) == 0
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            nested.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=3),
        ) == 1
        assert nested_event.payload == {
            "authorization": nested.model_dump(mode="json"),
            "outer_authorization_id": outer.authorization_id,
        }

    with CampaignStore(database) as reopened:
        outer_usage = reopened.authorization_usage(
            CAMPAIGN_ID,
            outer.authorization_id,
            now=NOW + timedelta(seconds=5),
        )
        nested_usage = reopened.authorization_usage(
            CAMPAIGN_ID,
            nested.authorization_id,
            now=NOW + timedelta(seconds=5),
        )
        assert outer_usage.candidate_count == 1
        assert outer_usage.activated_at == NOW
        assert nested_usage.candidate_count == 2
        assert nested_usage.activated_at == NOW + timedelta(seconds=1)


def test_nested_authorization_replacement_preserves_outer_and_retires_old_nested(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    outer = _stage_authorization(
        stage=CampaignState.FIR_APPLY,
        max_candidates=3,
    )
    first_nested = _stage_authorization(
        stage=CampaignState.I2C_RETUNE,
        max_candidates=2,
    )
    replacement_nested = _stage_authorization(
        stage=CampaignState.I2C_RETUNE,
        max_candidates=3,
    )

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.FIR_APPLY)
        store.activate_authorization(CAMPAIGN_ID, outer, activated_at=NOW)
        store.activate_nested_authorization(
            CAMPAIGN_ID,
            outer.authorization_id,
            first_nested,
            activated_at=NOW,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            first_nested.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=1),
        ) == 0

        store.activate_nested_authorization(
            CAMPAIGN_ID,
            outer.authorization_id,
            replacement_nested,
            activated_at=NOW + timedelta(seconds=2),
        )

        with pytest.raises(StoreError, match="active authorization"):
            store.reserve_candidate(
                CAMPAIGN_ID,
                first_nested.authorization_id,
                LEASE_OWNER,
                now=NOW + timedelta(seconds=3),
            )
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            replacement_nested.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=3),
        ) == 0
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            outer.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=3),
        ) == 0
        assert store.authorization_usage(
            CAMPAIGN_ID,
            first_nested.authorization_id,
            now=NOW + timedelta(seconds=3),
        ).candidate_count == 1


def test_nested_authorization_rejects_illegal_or_missing_outer_context(tmp_path):
    nested = _stage_authorization(stage=CampaignState.I2C_RETUNE)
    missing_outer_database = tmp_path / "missing-outer.sqlite3"
    with CampaignStore(missing_outer_database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.FIR_APPLY)
        with pytest.raises(ConcurrencyError, match="outer.*FIR_APPLY"):
            store.activate_nested_authorization(
                CAMPAIGN_ID,
                "missing-outer-authorization",
                nested,
                activated_at=NOW,
            )

    invalid_nested_database = tmp_path / "invalid-nested.sqlite3"
    outer = _stage_authorization(stage=CampaignState.FIR_APPLY)
    with CampaignStore(invalid_nested_database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.FIR_APPLY)
        store.activate_authorization(CAMPAIGN_ID, outer, activated_at=NOW)
        invalid_stage = _stage_authorization(stage=CampaignState.I2C_FINE)
        with pytest.raises(StoreError, match="I2C_RETUNE"):
            store.activate_nested_authorization(
                CAMPAIGN_ID,
                outer.authorization_id,
                invalid_stage,
                activated_at=NOW,
            )
        with pytest.raises(ConcurrencyError, match="outer.*FIR_APPLY"):
            store.activate_nested_authorization(
                CAMPAIGN_ID,
                "wrong-outer-authorization",
                nested,
                activated_at=NOW,
            )

    wrong_state_database = tmp_path / "wrong-state.sqlite3"
    with CampaignStore(wrong_state_database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)
        with pytest.raises(ConcurrencyError, match="FIR_APPLY"):
            store.activate_nested_authorization(
                CAMPAIGN_ID,
                outer.authorization_id,
                nested,
                activated_at=NOW,
            )


def test_authorized_transition_deactivates_unrelated_outer_and_nested_auths(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    outer = _stage_authorization(stage=CampaignState.FIR_APPLY)
    nested = _stage_authorization(stage=CampaignState.I2C_RETUNE)
    next_authorization = _stage_authorization(stage=CampaignState.I2C_FINE)

    with CampaignStore(database) as store:
        store.save_state(CAMPAIGN_ID, CampaignState.FIR_APPLY)
        store.activate_authorization(CAMPAIGN_ID, outer, activated_at=NOW)
        store.activate_nested_authorization(
            CAMPAIGN_ID,
            outer.authorization_id,
            nested,
            activated_at=NOW,
        )
        store.acquire_execution_lease(
            CAMPAIGN_ID,
            LEASE_OWNER,
            acquired_at=NOW,
        )
        store.save_transition(
            CAMPAIGN_ID,
            CampaignState.FIR_APPLY,
            CampaignState.AWAIT_FINE_APPROVAL,
        )
        store.save_authorized_transition(
            CAMPAIGN_ID,
            CampaignState.AWAIT_FINE_APPROVAL,
            CampaignState.I2C_FINE,
            next_authorization,
            activated_at=NOW + timedelta(seconds=1),
        )

        for retired_id in (outer.authorization_id, nested.authorization_id):
            with pytest.raises(StoreError, match="active authorization"):
                store.reserve_candidate(
                    CAMPAIGN_ID,
                    retired_id,
                    LEASE_OWNER,
                    now=NOW + timedelta(seconds=2),
                )
        assert store.reserve_candidate(
            CAMPAIGN_ID,
            next_authorization.authorization_id,
            LEASE_OWNER,
            now=NOW + timedelta(seconds=2),
        ) == 0
