from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.v2.continuous.models import ProviderAction, ProviderActionKind
from vcsel_ai_tuning.v2.continuous.policy import DEFAULT_I2C_POLICY
from vcsel_ai_tuning.v2.continuous.provider import (
    ContinuousDecisionProvider,
    ProviderContext,
    ReasonCode,
)
from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc)


def _eye() -> EyeSample:
    artifact = ArtifactReference(
        relative_path="eye.png",
        sha256="a" * 64,
        media_type="image/png",
    )
    return EyeSample(
        measurement_id="m-1",
        campaign_id="campaign-1",
        sweep_point_id="baseline",
        raw_eye_density_ref=artifact,
        metrics=EyeMetrics(tdecq_db=3.1),
        requested_params={},
        applied_params={"eq": 22.0},
        readback_params={"eq": 22.0},
        dca_config_fingerprint="b" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=NOW,
        artifact_hashes={"eye.png": "a" * 64},
    )


def _context(**updates: object) -> ProviderContext:
    values = {
        "sample": _eye(),
        "verified_readback": {"eq": 22},
        "policy": DEFAULT_I2C_POLICY,
        "hardware_round": 1,
        "max_rounds": 3,
        "target_tdecq_db": 2.7,
        "recent_valid_summaries": (),
        "best_reference": None,
        "allowed_output_schema": {"version": "provider-action-v1"},
    }
    values.update(updates)
    return ProviderContext(**values)


def test_context_is_bounded_and_derives_remaining_rounds() -> None:
    context = _context(
        recent_valid_summaries=tuple(
            {"measurement_id": f"m-{index}"} for index in range(5)
        )
    )
    assert context.policy_version == "i2c-encoding-v1"
    assert context.remaining_rounds == 3
    assert context.verified_readback == {"eq": 22}
    with pytest.raises(TypeError):
        context.verified_readback["eq"] = 23

    with pytest.raises(ValidationError):
        _context(recent_valid_summaries=tuple({"x": i} for i in range(6)))


def test_context_reference_is_prior_measurement_from_same_campaign() -> None:
    current = _eye()
    same_measurement = current.model_copy()
    other_campaign = current.model_copy(
        update={"measurement_id": "m-reference", "campaign_id": "campaign-2"}
    )
    invalid = current.model_copy(
        update={
            "measurement_id": "m-invalid",
            "measurement_status": MeasurementStatus.INVALID,
        }
    )
    future = current.model_copy(
        update={
            "measurement_id": "m-future",
            "captured_at": current.captured_at + timedelta(seconds=1),
        }
    )

    with pytest.raises(ValidationError, match="prior measurement"):
        _context(best_reference=same_measurement)
    with pytest.raises(ValidationError, match="campaign"):
        _context(best_reference=other_campaign)
    with pytest.raises(ValidationError, match="valid"):
        _context(best_reference=invalid)
    with pytest.raises(ValidationError, match="postdate"):
        _context(best_reference=future)


@pytest.mark.parametrize(
    "forbidden",
    [
        lambda: None,
        object(),
        {"apply": "write"},
        {"setFir": "forbidden"},
        {"MOONSHOT_API_KEY": "secret"},
        {"bert": "forbidden"},
        Path("COM1"),
    ],
)
def test_context_rejects_executable_or_secret_values(forbidden: object) -> None:
    with pytest.raises(ValidationError, match="forbidden|safe provider data"):
        _context(allowed_output_schema={"bad": forbidden})


def test_reason_code_vocabulary_is_versioned_and_strict() -> None:
    assert {item.value for item in ReasonCode} == {
        "TARGET_MET",
        "TARGET_NOT_MET",
        "EXPECTED_IMPROVEMENT",
        "EXPLORATION",
        "LEARNED_NEGATIVE_RESULT",
        "EXHAUSTED_SEARCH",
        "LOW_CONFIDENCE",
        "OOD",
        "INVALID_OBSERVATION",
        "NEED_HUMAN",
    }
    with pytest.raises(ValidationError):
        ProviderAction(
            action=ProviderActionKind.HOLD,
            reason_codes=("INVENTED",),
            confidence=0.5,
            summary="Unknown code.",
        )


def test_provider_protocol_is_runtime_checkable() -> None:
    class FakeProvider:
        provider_id = "fake"
        provider_version = "1"

        def decide(self, context: ProviderContext) -> ProviderAction:
            assert context.sample.measurement_id == "m-1"
            return ProviderAction(
                action=ProviderActionKind.HOLD,
                reason_codes=(ReasonCode.EXPLORATION,),
                confidence=0.8,
                summary="Hold one observation.",
            )

    provider = FakeProvider()
    assert isinstance(provider, ContinuousDecisionProvider)
    assert provider.decide(_context()).action is ProviderActionKind.HOLD
