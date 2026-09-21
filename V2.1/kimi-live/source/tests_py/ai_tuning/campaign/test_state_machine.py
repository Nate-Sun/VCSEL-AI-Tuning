from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vcsel_ai_tuning.campaign.authorization import (
    AuthorizationError,
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
from vcsel_ai_tuning.campaign.state_machine import (
    CampaignStateMachine,
    IllegalTransition,
)


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
CAMPAIGN_ID = "campaign-1"
BASELINE_HASH = "a" * 64

MAIN_CHAIN = (
    CampaignState.DRAFT,
    CampaignState.PREFLIGHT,
    CampaignState.AWAIT_BASELINE_APPROVAL,
    CampaignState.RESTORE_B0,
    CampaignState.BASELINE_MEASUREMENT,
    CampaignState.AWAIT_COARSE_APPROVAL,
    CampaignState.I2C_SCREENING,
    CampaignState.I2C_COARSE,
    CampaignState.AWAIT_FINE_APPROVAL,
    CampaignState.I2C_FINE,
    CampaignState.AWAIT_FIR_APPROVAL,
    CampaignState.FIR_APPLY,
    CampaignState.I2C_RETUNE,
    CampaignState.FINAL_VERIFICATION,
    CampaignState.COMPLETED,
)

MUTATING_STATES = {
    CampaignState.RESTORE_B0,
    CampaignState.I2C_SCREENING,
    CampaignState.I2C_COARSE,
    CampaignState.I2C_FINE,
    CampaignState.FIR_APPLY,
    CampaignState.I2C_RETUNE,
}

HARDWARE_ACTIVE_STATES = (
    CampaignState.RESTORE_B0,
    CampaignState.BASELINE_MEASUREMENT,
    CampaignState.I2C_SCREENING,
    CampaignState.I2C_COARSE,
    CampaignState.I2C_FINE,
    CampaignState.FIR_APPLY,
    CampaignState.I2C_RETUNE,
    CampaignState.FINAL_VERIFICATION,
)

STABLE_NONTERMINAL_STATES = (
    CampaignState.DRAFT,
    CampaignState.PREFLIGHT,
    CampaignState.AWAIT_BASELINE_APPROVAL,
    CampaignState.AWAIT_COARSE_APPROVAL,
    CampaignState.AWAIT_FINE_APPROVAL,
    CampaignState.AWAIT_FIR_APPROVAL,
)


def _authorization(
    stage: CampaignState,
    *,
    campaign_id: str = CAMPAIGN_ID,
    baseline_hash: str = BASELINE_HASH,
    issued_at: datetime = NOW - timedelta(minutes=5),
    expires_at: datetime = NOW + timedelta(minutes=5),
) -> StageAuthorization:
    rollback_snapshot = AppliedSettings(readback_i2c={"ibias_dac": 70})
    authorization = StageAuthorization(
        authorization_id="pending",
        campaign_id=campaign_id,
        mode=AuthorizationMode.PER_STAGE,
        stage=stage,
        baseline_hash=baseline_hash,
        rollback_hash=rollback_hash_for(rollback_snapshot),
        parameter_specs=(
            ParameterSpec(
                name="ibias_dac",
                encoding_min=0,
                encoding_max=255,
                tuning_min=64,
                tuning_max=76,
                max_step=2,
            ),
        ),
        max_candidates=3,
        max_duration_s=60.0,
        issued_at=issued_at,
        expires_at=expires_at,
        rollback_snapshot=rollback_snapshot,
    )
    return with_authorization_id(authorization)


def _authorization_context(stage: CampaignState) -> dict[str, object]:
    return {
        "authorization": _authorization(stage),
        "now": NOW,
        "current_campaign_id": CAMPAIGN_ID,
        "current_baseline_hash": BASELINE_HASH,
    }


def test_complete_main_chain_is_explicit_and_authorizes_every_mutating_stage():
    machine = CampaignStateMachine()

    for target in MAIN_CHAIN[1:]:
        context = _authorization_context(target) if target in MUTATING_STATES else {}
        assert machine.transition(machine.state, target, **context) == target

    assert machine.state == CampaignState.COMPLETED


def test_skipping_or_reversing_a_main_chain_state_is_rejected():
    machine = CampaignStateMachine()

    with pytest.raises(IllegalTransition, match="DRAFT.*AWAIT_BASELINE_APPROVAL"):
        machine.transition(CampaignState.DRAFT, CampaignState.AWAIT_BASELINE_APPROVAL)

    assert machine.state == CampaignState.DRAFT
    machine.transition(CampaignState.DRAFT, CampaignState.PREFLIGHT)
    with pytest.raises(IllegalTransition, match="PREFLIGHT.*DRAFT"):
        machine.transition(CampaignState.PREFLIGHT, CampaignState.DRAFT)
    assert machine.state == CampaignState.PREFLIGHT


def test_supplied_current_state_must_match_machine_state():
    machine = CampaignStateMachine(CampaignState.I2C_COARSE)

    with pytest.raises(IllegalTransition, match="current.*I2C_COARSE"):
        machine.transition(
            CampaignState.AWAIT_FINE_APPROVAL,
            CampaignState.I2C_FINE,
            **_authorization_context(CampaignState.I2C_FINE),
        )

    assert machine.state == CampaignState.I2C_COARSE


@pytest.mark.parametrize(
    ("predecessor", "target"),
    [
        (CampaignState.AWAIT_BASELINE_APPROVAL, CampaignState.RESTORE_B0),
        (CampaignState.AWAIT_COARSE_APPROVAL, CampaignState.I2C_SCREENING),
        (CampaignState.I2C_SCREENING, CampaignState.I2C_COARSE),
        (CampaignState.AWAIT_FINE_APPROVAL, CampaignState.I2C_FINE),
        (CampaignState.AWAIT_FIR_APPROVAL, CampaignState.FIR_APPLY),
        (CampaignState.FIR_APPLY, CampaignState.I2C_RETUNE),
    ],
)
def test_entering_mutating_stage_without_authorization_fails_before_state_change(
    predecessor, target
):
    machine = CampaignStateMachine(predecessor)

    with pytest.raises(AuthorizationError, match="authorization.*required"):
        machine.transition(predecessor, target)

    assert machine.state == predecessor


@pytest.mark.parametrize(
    ("authorization", "context", "message"),
    [
        (
            _authorization(CampaignState.I2C_COARSE),
            {},
            "stage.*I2C_FINE",
        ),
        (
            _authorization(
                CampaignState.I2C_FINE,
                issued_at=NOW - timedelta(hours=2),
                expires_at=NOW - timedelta(hours=1),
            ),
            {},
            "expired",
        ),
        (
            _authorization(CampaignState.I2C_FINE),
            {"current_baseline_hash": "b" * 64},
            "baseline.*stale",
        ),
        (
            _authorization(CampaignState.I2C_FINE),
            {"current_campaign_id": "campaign-2"},
            "campaign.*scope",
        ),
    ],
)
def test_invalid_authorization_never_changes_state(authorization, context, message):
    machine = CampaignStateMachine(CampaignState.AWAIT_FINE_APPROVAL)
    full_context = {
        "authorization": authorization,
        "now": NOW,
        "current_campaign_id": CAMPAIGN_ID,
        "current_baseline_hash": BASELINE_HASH,
        **context,
    }

    with pytest.raises(AuthorizationError, match=message):
        machine.transition(
            CampaignState.AWAIT_FINE_APPROVAL,
            CampaignState.I2C_FINE,
            **full_context,
        )

    assert machine.state == CampaignState.AWAIT_FINE_APPROVAL


def test_core_coarse_to_fine_gate_requires_fresh_stage_authorization():
    machine = CampaignStateMachine(CampaignState.I2C_COARSE)

    machine.transition("I2C_COARSE", "AWAIT_FINE_APPROVAL")
    with pytest.raises(AuthorizationError):
        machine.transition("AWAIT_FINE_APPROVAL", "I2C_FINE")

    assert machine.state == CampaignState.AWAIT_FINE_APPROVAL
    assert (
        machine.transition(
            "AWAIT_FINE_APPROVAL",
            "I2C_FINE",
            **_authorization_context(CampaignState.I2C_FINE),
        )
        == CampaignState.I2C_FINE
    )


def test_measurement_states_do_not_claim_hardware_write_authority():
    baseline = CampaignStateMachine(CampaignState.RESTORE_B0)
    final = CampaignStateMachine(CampaignState.I2C_RETUNE)

    assert baseline.transition("RESTORE_B0", "BASELINE_MEASUREMENT") == (
        CampaignState.BASELINE_MEASUREMENT
    )
    assert final.transition("I2C_RETUNE", "FINAL_VERIFICATION") == (
        CampaignState.FINAL_VERIFICATION
    )


def test_paused_constructor_requires_known_resume_state():
    with pytest.raises(IllegalTransition, match="resume.*required"):
        CampaignStateMachine(CampaignState.PAUSED)


@pytest.mark.parametrize(
    "invalid_resume_state",
    (
        CampaignState.PAUSED,
        CampaignState.ABORTING,
        CampaignState.ROLLING_BACK,
        CampaignState.FAILED,
        CampaignState.RECOVERY_REQUIRED,
        CampaignState.COMPLETED,
    ),
)
def test_paused_constructor_rejects_exceptional_or_terminal_resume_state(
    invalid_resume_state,
):
    with pytest.raises(IllegalTransition, match="resume|PAUSED"):
        CampaignStateMachine(
            CampaignState.PAUSED, resume_state=invalid_resume_state
        )


@pytest.mark.parametrize("resume_state", MAIN_CHAIN[:-1])
def test_paused_constructor_accepts_real_main_chain_source(resume_state):
    machine = CampaignStateMachine(CampaignState.PAUSED, resume_state=resume_state)

    assert machine.state == CampaignState.PAUSED
    assert machine.resume_state == resume_state


def test_pause_can_only_resume_the_state_that_was_paused():
    machine = CampaignStateMachine(CampaignState.I2C_COARSE)

    assert machine.transition("I2C_COARSE", "PAUSED") == CampaignState.PAUSED
    assert machine.resume_state == CampaignState.I2C_COARSE
    with pytest.raises(IllegalTransition, match="resume.*I2C_COARSE"):
        machine.transition("PAUSED", "AWAIT_FINE_APPROVAL")
    assert machine.state == CampaignState.PAUSED
    assert (
        machine.transition(
            "PAUSED",
            "I2C_COARSE",
            **_authorization_context(CampaignState.I2C_COARSE),
        )
        == CampaignState.I2C_COARSE
    )
    assert machine.resume_state is None


def test_paused_campaign_can_abort_and_follow_explicit_rollback_path():
    machine = CampaignStateMachine(CampaignState.I2C_COARSE)

    machine.transition("I2C_COARSE", "PAUSED")
    machine.transition("PAUSED", "ABORTING")
    machine.transition("ABORTING", "ROLLING_BACK")
    assert machine.transition("ROLLING_BACK", "FAILED") == CampaignState.FAILED

    with pytest.raises(IllegalTransition, match="terminal"):
        machine.transition("FAILED", "DRAFT")


@pytest.mark.parametrize("active_state", HARDWARE_ACTIVE_STATES)
def test_hardware_active_state_cannot_fail_before_rollback(active_state):
    machine = CampaignStateMachine(active_state)

    with pytest.raises(IllegalTransition, match=f"{active_state.value}.*FAILED"):
        machine.transition(active_state, CampaignState.FAILED)
    assert machine.state == active_state


@pytest.mark.parametrize(
    "rollback_outcome", [CampaignState.FAILED, CampaignState.RECOVERY_REQUIRED]
)
def test_hardware_active_state_can_abort_and_finish_rollback(rollback_outcome):
    machine = CampaignStateMachine(CampaignState.I2C_COARSE)

    machine.transition(CampaignState.I2C_COARSE, CampaignState.ABORTING)
    machine.transition(CampaignState.ABORTING, CampaignState.ROLLING_BACK)
    assert (
        machine.transition(CampaignState.ROLLING_BACK, rollback_outcome)
        == rollback_outcome
    )


def test_aborting_cannot_skip_rollback_to_failed():
    machine = CampaignStateMachine(CampaignState.ABORTING)

    with pytest.raises(IllegalTransition, match="ABORTING.*FAILED"):
        machine.transition(CampaignState.ABORTING, CampaignState.FAILED)

    assert machine.state == CampaignState.ABORTING
    assert machine.transition("ABORTING", "RECOVERY_REQUIRED") == (
        CampaignState.RECOVERY_REQUIRED
    )


def test_paused_hardware_active_state_cannot_skip_rollback_to_failed():
    machine = CampaignStateMachine(CampaignState.I2C_COARSE)
    machine.transition(CampaignState.I2C_COARSE, CampaignState.PAUSED)

    with pytest.raises(IllegalTransition, match="PAUSED.*FAILED|rollback"):
        machine.transition(CampaignState.PAUSED, CampaignState.FAILED)

    assert machine.state == CampaignState.PAUSED
    assert machine.resume_state == CampaignState.I2C_COARSE
    machine.transition(CampaignState.PAUSED, CampaignState.ABORTING)
    machine.transition(CampaignState.ABORTING, CampaignState.ROLLING_BACK)
    assert machine.transition("ROLLING_BACK", "FAILED") == CampaignState.FAILED


def test_paused_stable_state_may_fail_without_rollback():
    machine = CampaignStateMachine(CampaignState.AWAIT_FINE_APPROVAL)
    machine.transition(CampaignState.AWAIT_FINE_APPROVAL, CampaignState.PAUSED)

    assert machine.transition(CampaignState.PAUSED, CampaignState.FAILED) == (
        CampaignState.FAILED
    )


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "RECOVERY_REQUIRED"])
def test_terminal_states_cannot_be_left(terminal):
    machine = CampaignStateMachine(terminal)

    with pytest.raises(IllegalTransition, match="terminal"):
        machine.transition(terminal, "PAUSED")

    assert machine.state == CampaignState(terminal)


@pytest.mark.parametrize("stable_state", STABLE_NONTERMINAL_STATES)
def test_stable_nonterminal_state_can_fail_without_rollback(stable_state):
    machine = CampaignStateMachine(stable_state)

    assert machine.transition(stable_state, CampaignState.FAILED) == CampaignState.FAILED


def test_aborting_can_require_manual_recovery_without_attempting_rollback():
    recovery = CampaignStateMachine(CampaignState.ABORTING)

    assert recovery.transition("ABORTING", "RECOVERY_REQUIRED") == (
        CampaignState.RECOVERY_REQUIRED
    )
