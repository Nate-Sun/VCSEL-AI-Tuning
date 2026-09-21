from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
)
from vcsel_ai_tuning.v2.continuous.models import (
    CampaignResult,
    ContinuousPreflightRequest,
    ContinuousRequest,
    ContinuousState,
    ExecutionMode,
    HumanDecision,
    ProviderAction,
    ProviderActionKind,
    ReasonCode,
    RoundRecord,
    StoppingPolicy,
)
from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc)
HASH = "a" * 64


def _eye() -> EyeSample:
    artifact = ArtifactReference(
        relative_path="eyes/before.png",
        sha256=HASH,
        media_type="image/png",
    )
    return EyeSample(
        measurement_id="m-1",
        campaign_id="campaign-1",
        sweep_point_id="baseline",
        raw_eye_density_ref=artifact,
        metrics=EyeMetrics(tdecq_db=3.1, outer_er_chan1a_db=4.5),
        requested_params={},
        applied_params={"eq": 22.0},
        readback_params={"eq": 22.0},
        dca_config_fingerprint="b" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=NOW,
        artifact_hashes={artifact.relative_path: artifact.sha256},
    )


def _authorization() -> StageAuthorization:
    return StageAuthorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_FINE,
        baseline_hash="c" * 64,
        rollback_hash="d" * 64,
        max_candidates=3,
        max_duration_s=300.0,
        issued_at=NOW,
        expires_at=datetime(2026, 8, 12, 10, 10, tzinfo=timezone.utc),
        rollback_snapshot=AppliedSettings(readback_i2c={"eq": 22}),
    )


def _request(**updates: object) -> ContinuousRequest:
    values = {
        "campaign_id": "campaign-1",
        "initial_eye": _eye(),
        "authorization": _authorization(),
        "policy": {"version": "i2c-v1"},
        "max_rounds": 3,
        "provider_config": {"provider_id": "fake"},
        "stopping_policy": StoppingPolicy(),
        "b0_path": Path("configs/ai_tuning/b0.yaml"),
        "result_dir": Path("artifacts/continuous"),
        "execution_mode": ExecutionMode.FAKE,
    }
    values.update(updates)
    return ContinuousRequest(**values)


def test_public_state_and_decision_vocabulary_is_exact() -> None:
    assert tuple(item.value for item in ContinuousState) == (
        "RUNNING",
        "AWAITING_HUMAN",
        "COMPLETED",
        "FAILED_SAFE",
        "RECOVERY_REQUIRED",
    )
    assert tuple(item.value for item in ProviderActionKind) == (
        "PROPOSE",
        "HOLD",
        "STOP",
        "REQUEST_HUMAN",
    )
    assert tuple(item.value for item in HumanDecision) == (
        "RESUME",
        "ACCEPT",
        "RESTORE_B0",
    )


def test_default_tdecq_target_is_two_db() -> None:
    assert StoppingPolicy().target_tdecq_db == pytest.approx(2.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("inf"), float("nan")])
def test_stopping_policy_rejects_non_positive_or_non_finite_targets(
    bad: float,
) -> None:
    with pytest.raises(ValidationError):
        StoppingPolicy(target_tdecq_db=bad)


@pytest.mark.parametrize("bad", [True, 0, -1, 3.0])
def test_requests_reject_non_positive_or_non_strict_round_counts(
    bad: object,
) -> None:
    with pytest.raises(ValidationError):
        _request(max_rounds=bad)
    with pytest.raises(ValidationError):
        ContinuousPreflightRequest(
            campaign_id="campaign-1",
            initial_eye=_eye(),
            policy={"version": "i2c-v1"},
            max_rounds=bad,
        )


@pytest.mark.parametrize("max_rounds", [1, 20, 10_000])
def test_requests_accept_positive_strict_round_counts(max_rounds: int) -> None:
    assert _request(max_rounds=max_rounds).max_rounds == max_rounds
    assert (
        ContinuousPreflightRequest(
            campaign_id="campaign-1",
            initial_eye=_eye(),
            policy={"version": "i2c-v1"},
            max_rounds=max_rounds,
        ).max_rounds
        == max_rounds
    )


def test_request_rejects_extra_fields_and_cross_campaign_inputs() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ContinuousRequest(
            **_request().model_dump(round_trip=True),
            surprise=True,
        )
    with pytest.raises(ValidationError, match="commission_deadline"):
        _request(commission_deadline=NOW)
    with pytest.raises(ValidationError, match="campaign"):
        _request(campaign_id="other")


def test_provider_action_enforces_sparse_action_shape() -> None:
    proposal = ProviderAction(
        action=ProviderActionKind.PROPOSE,
        requested_params={"eq": 23},
        reason_codes=(ReasonCode.EXPLORATION,),
        confidence=0.8,
        summary="Try one bounded equalizer step.",
    )
    assert proposal.requested_params == {"eq": 23}

    with pytest.raises(ValidationError, match="requires"):
        ProviderAction(
            action=ProviderActionKind.PROPOSE,
            requested_params=None,
            reason_codes=(ReasonCode.EXPLORATION,),
            confidence=0.8,
            summary="Missing proposal.",
        )
    with pytest.raises(ValidationError, match="forbids"):
        ProviderAction(
            action=ProviderActionKind.STOP,
            requested_params={"eq": 23},
            reason_codes=(ReasonCode.TARGET_MET,),
            confidence=0.8,
            summary="Stop.",
        )


def test_round_record_rejects_naive_time_and_non_finite_metric() -> None:
    with pytest.raises(ValidationError):
        RoundRecord(
            campaign_id="campaign-1",
            round_index=1,
            proposal_attempt=1,
            before_observation_id="before-1",
            after_observation_id="after-1",
            requested_params={"eq": 23},
            applied_params={"eq": 23},
            readback_params={"eq": 23},
            before_metrics={"tdecq_db": 3.1},
            after_metrics={"tdecq_db": float("nan")},
            started_at=datetime(2026, 8, 12, 10, 0),
            finished_at=NOW,
            evidence_persisted=True,
        )


def test_campaign_result_rejects_unproved_live_claim_and_fake_proof() -> None:
    base = {
        "campaign_id": "campaign-1",
        "state": ContinuousState.COMPLETED,
        "execution_mode": ExecutionMode.LIVE,
        "completed_rounds": 0,
        "target_tdecq_db": 2.8,
        "target_reached": False,
        "acceptance_passed": False,
        "closed_loop_proven": True,
        "stop_reason": "MAX_ROUNDS_REACHED",
        "started_at": NOW,
        "finished_at": NOW,
    }
    with pytest.raises(ValidationError, match="closed_loop_proven"):
        CampaignResult(**base)
    with pytest.raises(ValidationError, match="FAKE"):
        CampaignResult(
            **{
                **base,
                "execution_mode": ExecutionMode.FAKE,
                "completed_rounds": 1,
                "adapter_identities": {"i2c": "fake-i2c", "dca": "fake-dca"},
                "b0_restored": True,
                "evidence_complete": True,
            }
        )


def test_campaign_result_reads_json_written_before_provider_error_type() -> None:
    result = CampaignResult(
        campaign_id="campaign-1",
        state=ContinuousState.FAILED_SAFE,
        execution_mode=ExecutionMode.FAKE,
        completed_rounds=0,
        target_tdecq_db=2.8,
        target_reached=False,
        acceptance_passed=False,
        closed_loop_proven=False,
        stop_reason="PROVIDER_FAILURE",
        started_at=NOW,
        finished_at=NOW,
        b0_restored=True,
    )
    payload = result.model_dump(mode="json")
    payload.pop("provider_error_type")
    payload.pop("provider_error_message")

    loaded = CampaignResult.model_validate_json(json.dumps(payload))

    assert loaded.provider_error_type is None
    assert loaded.provider_error_message is None
    assert loaded.target_tdecq_db == 2.8


def test_campaign_result_requires_and_serializes_exact_target() -> None:
    base = {
        "campaign_id": "campaign-1",
        "state": ContinuousState.RUNNING,
        "execution_mode": ExecutionMode.FAKE,
        "completed_rounds": 0,
        "target_reached": False,
        "acceptance_passed": False,
        "closed_loop_proven": False,
        "started_at": NOW,
    }

    with pytest.raises(ValidationError, match="target_tdecq_db"):
        CampaignResult(**base)

    result = CampaignResult(**base, target_tdecq_db=2.8)

    assert result.target_tdecq_db == 2.8
    assert result.model_dump(mode="json")["target_tdecq_db"] == 2.8


def test_campaign_result_rejects_terminal_flag_inconsistency() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        CampaignResult(
            campaign_id="campaign-1",
            state=ContinuousState.RUNNING,
            execution_mode=ExecutionMode.FAKE,
            completed_rounds=0,
            target_tdecq_db=2.8,
            target_reached=False,
            acceptance_passed=False,
            closed_loop_proven=False,
            started_at=NOW,
            finished_at=NOW,
        )
