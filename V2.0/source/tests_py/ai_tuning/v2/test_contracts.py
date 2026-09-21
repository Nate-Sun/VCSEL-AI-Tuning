from __future__ import annotations

import json
from datetime import datetime
from math import inf, nan

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.optimization.protocol import OptimizationContext, TuningObservation
from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    BrainCandidate,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    EyeMetrics,
    EyeSample,
    PerceptionResult,
    V2Observation,
)


class DeceptiveParameterFloat(float):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class DeceptiveParameterKey(str):
    pass


def test_complete_perception_propose_and_brain_result_succeed(
    perception_result: PerceptionResult,
    brain_candidates: tuple[BrainCandidate, ...],
) -> None:
    decision = DecisionProposal(
        based_on_measurement_id=perception_result.measurement_id,
        action=DecisionAction.PROPOSE,
        candidate_id=brain_candidates[0].candidate_id,
        requested_params=brain_candidates[0].requested_params,
        reason_codes=("LOW_TDECQ",),
        expected_improvement=0.15,
        provider_id="decision-provider",
        provider_version="1.0.0",
    )

    result = BrainResult(
        perception=perception_result,
        decision=decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=18.5,
        request_hash="b" * 64,
        response_hash="c" * 64,
    )

    assert result.decision.action is DecisionAction.PROPOSE


def test_propose_requires_candidate_and_requested_params(
    perception_result: PerceptionResult,
) -> None:
    with pytest.raises(ValidationError, match="PROPOSE"):
        DecisionProposal(
            based_on_measurement_id=perception_result.measurement_id,
            action=DecisionAction.PROPOSE,
            reason_codes=("LOW_TDECQ",),
            provider_id="decision-provider",
            provider_version="1.0.0",
        )


def test_hold_forbids_executable_fields(perception_result: PerceptionResult) -> None:
    with pytest.raises(ValidationError, match="HOLD"):
        DecisionProposal(
            based_on_measurement_id=perception_result.measurement_id,
            action=DecisionAction.HOLD,
            candidate_id="candidate-001",
            requested_params={"bias_ma": 6.0},
            reason_codes=("NO_SAFE_CANDIDATE",),
            provider_id="decision-provider",
            provider_version="1.0.0",
        )


@pytest.mark.parametrize(
    "action",
    [DecisionAction.STOP, DecisionAction.REQUEST_HUMAN],
)
def test_no_safe_candidate_reason_is_reserved_for_hold(
    action: DecisionAction,
    perception_result: PerceptionResult,
) -> None:
    with pytest.raises(ValidationError, match="NO_SAFE_CANDIDATE"):
        DecisionProposal(
            based_on_measurement_id=perception_result.measurement_id,
            action=action,
            reason_codes=("NO_SAFE_CANDIDATE",),
            provider_id="decision-provider",
            provider_version="1.0.0",
        )


def test_hold_allows_no_safe_candidate_reason(
    perception_result: PerceptionResult,
) -> None:
    proposal = DecisionProposal(
        based_on_measurement_id=perception_result.measurement_id,
        action=DecisionAction.HOLD,
        reason_codes=("NO_SAFE_CANDIDATE",),
        provider_id="decision-provider",
        provider_version="1.0.0",
    )

    assert proposal.reason_codes == ("NO_SAFE_CANDIDATE",)


def test_eye_sample_rejects_artifact_hash_mismatch(eye_sample: EyeSample) -> None:
    payload = eye_sample.model_dump()
    payload["artifact_hashes"] = {"point-001/raw_eye_density.npy": "d" * 64}

    with pytest.raises(ValidationError, match="artifact_hashes"):
        EyeSample(**payload)


@pytest.mark.parametrize("score", [nan, inf, -inf])
def test_perception_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        PerceptionResult(
            measurement_id="measurement-001",
            quality_score=score,
            confidence=0.5,
            valid=True,
            ood=False,
            defects={},
            provider_id="vision-provider",
            provider_version="1.0.0",
            input_hash="a" * 64,
        )


def test_contracts_reject_extra_fields(perception_result: PerceptionResult) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PerceptionResult(**perception_result.model_dump(), unexpected=True)


def test_brain_result_rejects_measurement_mismatch(
    perception_result: PerceptionResult,
) -> None:
    decision = DecisionProposal(
        based_on_measurement_id="measurement-002",
        action=DecisionAction.HOLD,
        reason_codes=("WAIT_FOR_DATA",),
        provider_id="decision-provider",
        provider_version="1.0.0",
    )

    with pytest.raises(ValidationError, match="measurement_id"):
        BrainResult(
            perception=perception_result,
            decision=decision,
            brain_mode=BrainMode.PURE_API,
            latency_ms=18.5,
            request_hash="b" * 64,
            response_hash="c" * 64,
        )


def test_requested_params_are_deeply_immutable(
    brain_candidates: tuple[BrainCandidate, ...],
) -> None:
    requested_params = brain_candidates[0].requested_params

    with pytest.raises(TypeError):
        requested_params["equalizer"]["tap_1"] = 2.0


@pytest.mark.parametrize(
    "requested_params",
    [
        {"bias_ma": DeceptiveParameterFloat(999.0)},
        {"equalizer": {"tap_1": DeceptiveParameterFloat(999.0)}},
        {DeceptiveParameterKey("bias_ma"): 1.0},
    ],
)
def test_requested_params_reject_subclass_values_and_keys(requested_params) -> None:
    with pytest.raises(ValidationError):
        BrainCandidate(candidate_id="candidate-001", requested_params=requested_params)


@pytest.mark.parametrize(
    "nested_key",
    ["", 0],
)
def test_requested_params_reject_invalid_nested_mapping_keys(
    nested_key: object,
) -> None:
    with pytest.raises(ValidationError):
        BrainCandidate(
            candidate_id="candidate-001",
            requested_params={"top": {nested_key: 1.0}},
        )


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
@pytest.mark.parametrize(
    "unsafe_key",
    ["", " ", " leading", "trailing ", "x\x00y", "x\ny"],
    ids=["empty", "blank", "leading-space", "trailing-space", "nul", "newline"],
)
def test_brain_candidate_rejects_unsafe_mapping_names(
    unsafe_key: str,
    nested: bool,
) -> None:
    requested_params = (
        {"top": {unsafe_key: 1.0}} if nested else {unsafe_key: 1.0}
    )

    with pytest.raises(ValidationError):
        BrainCandidate(
            candidate_id="candidate-001",
            requested_params=requested_params,
        )


def test_requested_params_have_stable_json_serialization() -> None:
    candidate = BrainCandidate(
        candidate_id="candidate-001",
        requested_params={"top": {"tap_1": 1.0}},
    )

    first_dump = candidate.model_dump_json()
    second_dump = candidate.model_dump_json()

    assert first_dump == second_dump
    assert json.loads(first_dump) == {
        "candidate_id": "candidate-001",
        "requested_params": {"top": {"tap_1": 1.0}},
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "/absolute/raw.npy",
        "C:/raw.npy",
        "../raw.npy",
        "point/../raw.npy",
        r"point\raw.npy",
        "segment/file:bad.npy",
        "NUL",
        "nul.txt",
        "NUL.",
        "folder/CON ",
        "folder/file. ",
        "COM¹",
        "COM²",
        "COM³",
        "LPT¹",
        "LPT²",
        "LPT³",
        "folder/COM¹.npy",
        "folder/cOm².npy",
        "folder/lPt³.npy",
    ],
)
def test_artifact_reference_rejects_noncanonical_paths(relative_path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactReference(
            relative_path=relative_path,
            sha256="a" * 64,
            media_type="application/x-npy",
        )


def test_artifact_reference_accepts_normal_relative_path() -> None:
    artifact = ArtifactReference(
        relative_path="folder/file.npy",
        sha256="a" * 64,
        media_type="application/x-npy",
    )

    assert artifact.relative_path == "folder/file.npy"


@pytest.mark.parametrize(
    "field_name",
    ["requested_params", "applied_params", "readback_params"],
)
@pytest.mark.parametrize(
    "unsafe_key",
    ["", " ", "x\x00y", "x\ny"],
    ids=["empty", "blank", "nul", "newline"],
)
def test_eye_sample_parameter_mappings_reject_unsafe_names(
    field_name: str,
    unsafe_key: str,
    eye_sample: EyeSample,
) -> None:
    payload = eye_sample.model_dump()
    payload[field_name] = {unsafe_key: 1.0}

    with pytest.raises(ValidationError):
        EyeSample(**payload)


def test_eye_metrics_rejects_unsafe_other_metric_name() -> None:
    with pytest.raises(ValidationError):
        EyeMetrics(other_metrics={"x\x00y": 1.0})


def test_perception_rejects_unsafe_defect_name(
    perception_result: PerceptionResult,
) -> None:
    payload = perception_result.model_dump()
    payload["defects"] = {"x\ny": 0.1}

    with pytest.raises(ValidationError):
        PerceptionResult(**payload)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_perception_rejects_probability_outside_unit_interval(probability: float) -> None:
    with pytest.raises(ValidationError):
        PerceptionResult(
            measurement_id="measurement-001",
            quality_score=probability,
            confidence=0.5,
            valid=True,
            ood=False,
            defects={"ringing": 0.1},
            provider_id="vision-provider",
            provider_version="1.0.0",
            input_hash="a" * 64,
        )


def test_eye_sample_rejects_naive_timestamp(eye_sample: EyeSample) -> None:
    payload = eye_sample.model_dump()
    payload["captured_at"] = datetime(2026, 7, 30, 12, 0)

    with pytest.raises(ValidationError):
        EyeSample(**payload)


@pytest.mark.parametrize("sha256", ["A" * 64, "a" * 63, "g" * 64])
def test_contracts_require_lowercase_sha256(sha256: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactReference(
            relative_path="point-001/raw_eye_density.npy",
            sha256=sha256,
            media_type="application/x-npy",
        )


def test_v2_observation_keeps_distinct_observation_id_semantics(
    perception_result: PerceptionResult,
) -> None:
    tuning_observation = TuningObservation(
        observation_id="optimizer-observation-001",
        candidate_id="candidate-001",
        context=OptimizationContext(
            driver_profile_hash="d" * 64,
            vcsel_model="model-001",
            device_id="device-001",
            test_context_id="context-001",
        ),
        settings_requested={"bias": 6},
        settings_readback={"bias": 6},
        tdecq_db=1.2,
        tdecq_status="CORR",
        er_db=4.1,
        er_valid=True,
    )

    observation = V2Observation(
        measurement_id=perception_result.measurement_id,
        tuning_observation=tuning_observation,
        perception=perception_result,
    )

    assert observation.tuning_observation.observation_id != observation.measurement_id
