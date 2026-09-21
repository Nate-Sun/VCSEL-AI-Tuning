from __future__ import annotations

from math import inf, nan

import pytest

from vcsel_ai_tuning.optimization.protocol import OptimizationContext, TuningObservation
from vcsel_ai_tuning.v2 import (
    BrainCandidate,
    BrainMode,
    DecisionAction,
    DecisionProposal,
    MeasurementStatus,
    PerceptionResult,
    V2Observation,
)
from vcsel_ai_tuning.v2.brain import (
    ComposedBrain,
    DecisionProvider,
    EyeScorer,
    FusedBrainProvider,
    MultimodalApiBrain,
    OptimizationBrain,
)


def _perception(eye_sample, **changes: object) -> PerceptionResult:
    payload = {
        "measurement_id": eye_sample.measurement_id,
        "quality_score": 0.9,
        "confidence": 0.9,
        "valid": True,
        "ood": False,
        "defects": {},
        "provider_id": "local-vision",
        "provider_version": "1",
        "input_hash": eye_sample.raw_eye_density_ref.sha256,
    }
    payload.update(changes)
    return PerceptionResult(**payload)


def _proposal(eye_sample, candidate, **changes: object) -> DecisionProposal:
    payload = {
        "based_on_measurement_id": eye_sample.measurement_id,
        "action": DecisionAction.PROPOSE,
        "candidate_id": candidate.candidate_id,
        "requested_params": candidate.requested_params,
        "reason_codes": ("IMPROVE",),
        "provider_id": "local-decision",
        "provider_version": "1",
    }
    payload.update(changes)
    return DecisionProposal(**payload)


class SpyScorer:
    provider_id = "local-vision"
    provider_version = "1"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def score(self, sample):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class SpyDecision:
    provider_id = "local-decision"
    provider_version = "1"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def propose(self, sample, perception, history, safe_candidates):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class SpyFused:
    provider_id = "api"
    provider_version = "1"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def analyze_once(self, sample, history, safe_candidates):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class DeceptiveMinimumConfidence(float):
    def __ge__(self, other):
        return True

    def __le__(self, other):
        return True


def _history_observation(eye_sample, suffix: str) -> V2Observation:
    measurement_id = f"history-{suffix}"
    perception = _perception(eye_sample, measurement_id=measurement_id)
    return V2Observation(
        measurement_id=measurement_id,
        tuning_observation=TuningObservation(
            observation_id=f"observation-{suffix}",
            candidate_id="candidate-001",
            context=OptimizationContext(
                driver_profile_hash="d" * 64,
                vcsel_model="model-001",
                device_id="device-001",
                test_context_id="context-001",
            ),
            settings_requested={"bias": 6.0},
            settings_readback={"bias": 6.0},
            tdecq_db=1.0,
            tdecq_status="CORR",
            er_db=4.0,
            er_valid=True,
        ),
        perception=perception,
    )


def test_composed_calls_each_provider_once_and_can_propose(eye_sample, brain_candidates):
    scorer = SpyScorer(_perception(eye_sample))
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8)

    result = brain.analyze(eye_sample, (), brain_candidates)

    assert isinstance(brain, OptimizationBrain)
    assert result.brain_mode is BrainMode.LOCAL_VISION_LOCAL
    assert result.decision.action is DecisionAction.PROPOSE
    assert scorer.calls == decision.calls == 1


def test_protocol_identities_are_typed_properties_and_concrete_identities_are_provider_derived(
    eye_sample,
    brain_candidates,
):
    scorer = SpyScorer(_perception(eye_sample))
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))
    fused = SpyFused((_perception(eye_sample), _proposal(eye_sample, brain_candidates[0])))

    composed = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8)
    multimodal = MultimodalApiBrain(fused, 0.8)

    for protocol, names in (
        (OptimizationBrain, ("provider_key", "provider_version")),
        (EyeScorer, ("provider_id", "provider_version")),
        (DecisionProvider, ("provider_id", "provider_version")),
        (FusedBrainProvider, ("provider_id", "provider_version")),
    ):
        for name in names:
            assert isinstance(protocol.__dict__[name], property)

    assert composed.provider_key == "local-vision|local-decision"
    assert composed.provider_version == "1|1"
    assert multimodal.provider_key == "api"
    assert multimodal.provider_version == "1"


def test_fused_calls_once_and_uses_pure_api_mode(eye_sample, brain_candidates):
    provider = SpyFused((
        _perception(eye_sample, provider_id="api"),
        _proposal(eye_sample, brain_candidates[0], provider_id="api"),
    ))

    result = MultimodalApiBrain(provider, 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.brain_mode is BrainMode.PURE_API
    assert result.decision.action is DecisionAction.PROPOSE
    assert provider.calls == 1


def test_composed_rejects_pure_api_mode():
    with pytest.raises(ValueError, match="PURE_API"):
        ComposedBrain(BrainMode.PURE_API, object(), object(), 0.8)


@pytest.mark.parametrize("mode", ["LOCAL_VISION_LOCAL", "PURE_API", object()])
def test_composed_rejects_non_enum_modes_at_construction(eye_sample, brain_candidates, mode):
    with pytest.raises(TypeError, match="BrainMode"):
        ComposedBrain(
            mode,
            SpyScorer(_perception(eye_sample)),
            SpyDecision(_proposal(eye_sample, brain_candidates[0])),
            0.8,
        )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [({"valid": False}, "INVALID_EYE"), ({"ood": True}, "OOD"), ({"confidence": 0.79}, "LOW_CONFIDENCE")],
)
def test_composed_perception_gate_holds_without_decision(eye_sample, brain_candidates, changes, reason):
    scorer = SpyScorer(_perception(eye_sample, **changes))
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))

    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == (reason,)
    assert result.decision.candidate_id is None
    assert result.decision.requested_params is None
    assert result.perception == scorer.outcome
    assert result.perception is not scorer.outcome
    assert scorer.calls == 1
    assert decision.calls == 0


def test_minimum_confidence_validation_for_each_brain():
    for value in (-0.1, 1.1, nan, inf, True):
        with pytest.raises((TypeError, ValueError)):
            ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(None), SpyDecision(None), value)
        with pytest.raises((TypeError, ValueError)):
            MultimodalApiBrain(SpyFused(None), value)


def test_minimum_confidence_rejects_float_subclass_with_deceptive_comparisons():
    value = DeceptiveMinimumConfidence(-1.0)
    with pytest.raises(TypeError):
        ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(None), SpyDecision(None), value)
    with pytest.raises(TypeError):
        MultimodalApiBrain(SpyFused(None), value)


@pytest.mark.parametrize("status", [MeasurementStatus.INVALID, MeasurementStatus.INCOMPLETE])
def test_invalid_or_incomplete_sample_holds_without_provider_calls(eye_sample, brain_candidates, status):
    sample = eye_sample.model_copy(update={"measurement_status": status})
    scorer = SpyScorer(_perception(eye_sample))
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))

    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8).analyze(sample, (), brain_candidates)

    assert result.decision.reason_codes == ("INVALID_EYE",)
    assert scorer.calls == decision.calls == 0


def test_guard_perception_uses_raw_eye_hash_while_request_hash_keeps_full_request(eye_sample, brain_candidates):
    invalid = eye_sample.model_copy(update={"measurement_status": MeasurementStatus.INVALID})
    changed = invalid.model_copy(update={"requested_params": {"bias_ma": 7.0}})
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(None), SpyDecision(None), 0.8)

    first = brain.analyze(invalid, (), brain_candidates)
    second = brain.analyze(changed, (), brain_candidates)

    for result in (first, second):
        assert result.perception.input_hash == eye_sample.raw_eye_density_ref.sha256
        assert result.perception.ood is True
        assert result.perception.valid is False
        assert result.perception.quality_score == result.perception.confidence == 0.0
        assert result.perception.provider_id == "brain-guard"
        assert result.perception.provider_version == "1"
        assert result.request_hash != result.perception.input_hash
    assert first.request_hash != second.request_hash


def test_empty_or_duplicate_candidate_set_holds_before_providers(eye_sample, brain_candidates):
    scorer = SpyScorer(_perception(eye_sample))
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8)

    empty = brain.analyze(eye_sample, (), ())
    duplicate = brain.analyze(eye_sample, (), (brain_candidates[0], brain_candidates[0]))

    assert empty.decision.reason_codes == ("NO_SAFE_CANDIDATE",)
    assert duplicate.decision.reason_codes == ("DUPLICATE_CANDIDATE_ID",)
    assert scorer.calls == decision.calls == 0


def test_non_propose_safe_actions_are_preserved_after_validation(eye_sample, brain_candidates):
    for action in (DecisionAction.HOLD, DecisionAction.STOP, DecisionAction.REQUEST_HUMAN):
        scorer = SpyScorer(_perception(eye_sample))
        decision = SpyDecision(_proposal(eye_sample, brain_candidates[0], action=action, candidate_id=None, requested_params=None))

        result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8).analyze(eye_sample, (), brain_candidates)

        assert result.decision.action is action
        assert result.decision.candidate_id is None
        assert result.decision.requested_params is None


def test_canonical_hashes_are_lowercase_and_response_ignores_latency(eye_sample, brain_candidates):
    first = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(_proposal(eye_sample, brain_candidates[0])), 0.8).analyze(eye_sample, (), brain_candidates)
    second = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(_proposal(eye_sample, brain_candidates[0])), 0.8).analyze(eye_sample, (), brain_candidates)
    changed = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(_proposal(eye_sample, brain_candidates[1])), 0.8).analyze(eye_sample, (), brain_candidates[1:])

    assert first.response_hash == second.response_hash
    assert first.request_hash != changed.request_hash
    assert len(first.request_hash) == len(first.response_hash) == 64
    assert first.request_hash.islower() and first.response_hash.islower()


def test_request_hash_preserves_history_and_candidate_order(eye_sample, brain_candidates):
    history = (_history_observation(eye_sample, "one"), _history_observation(eye_sample, "two"))
    first = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(_proposal(eye_sample, brain_candidates[0])), 0.8).analyze(eye_sample, history, brain_candidates)
    reordered = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(_proposal(eye_sample, brain_candidates[1])), 0.8).analyze(eye_sample, (), tuple(reversed(brain_candidates)))
    reordered_history = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(_proposal(eye_sample, brain_candidates[0])), 0.8).analyze(eye_sample, tuple(reversed(history)), brain_candidates)

    assert first.request_hash != reordered.request_hash
    assert first.request_hash != reordered_history.request_hash
