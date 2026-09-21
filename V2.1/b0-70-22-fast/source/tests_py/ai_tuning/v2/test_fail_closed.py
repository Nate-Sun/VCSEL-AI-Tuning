from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from vcsel_ai_tuning.v2 import BrainCandidate, BrainMode, DecisionAction, DecisionProposal, PerceptionResult
from vcsel_ai_tuning.v2.brain import ComposedBrain, MultimodalApiBrain

from .test_brain import SpyDecision, SpyFused, SpyScorer, _perception, _proposal


class DeceptiveText(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class DeceptiveFloat(float):
    def __lt__(self, other):
        return False


class MutatingDecisionProvider:
    provider_id = "local-decision"
    provider_version = "1"

    def propose(self, sample, perception, history, safe_candidates):
        object.__setattr__(safe_candidates[0], "candidate_id", "outside")
        object.__setattr__(safe_candidates[0], "requested_params", {"bias_ma": 999.0})
        return DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=safe_candidates[0].candidate_id,
            requested_params=safe_candidates[0].requested_params,
            reason_codes=("MUTATED",),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


class MutatingFusedProvider:
    provider_id = "api"
    provider_version = "1"

    def analyze_once(self, sample, history, safe_candidates):
        object.__setattr__(safe_candidates[0], "candidate_id", "outside")
        object.__setattr__(safe_candidates[0], "requested_params", {"bias_ma": 999.0})
        return (
            _perception(sample, provider_id="api"),
            DecisionProposal(
                based_on_measurement_id=sample.measurement_id,
                action=DecisionAction.PROPOSE,
                candidate_id=safe_candidates[0].candidate_id,
                requested_params=safe_candidates[0].requested_params,
                reason_codes=("MUTATED",),
                provider_id="api",
                provider_version="1",
            ),
        )


def _constructed_evil_perception(eye_sample) -> PerceptionResult:
    return PerceptionResult.model_construct(
        measurement_id=DeceptiveText("foreign-measurement"),
        quality_score=0.9,
        confidence=DeceptiveFloat(0.1),
        valid=True,
        ood=False,
        defects={},
        provider_id=DeceptiveText("foreign-provider"),
        provider_version=DeceptiveText("foreign-version"),
        input_hash=DeceptiveText("b" * 64),
    )


@pytest.mark.parametrize("exception", [RuntimeError("token=very-secret"), ValueError("api-key=do-not-leak")])
def test_provider_exceptions_fail_closed_without_secret_leakage(eye_sample, brain_candidates, exception):
    scorer = SpyScorer(exception)
    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, SpyDecision(None), 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert result.decision.candidate_id is result.decision.requested_params is None
    assert "secret" not in result.model_dump_json()
    assert "api-key" not in result.model_dump_json()


def test_decision_and_fused_exceptions_fail_closed(eye_sample, brain_candidates):
    decision = SpyDecision(RuntimeError("decision secret"))
    composed = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), decision, 0.8)
    fused = MultimodalApiBrain(SpyFused(RuntimeError("fused secret")), 0.8)

    assert composed.analyze(eye_sample, (), brain_candidates).decision.reason_codes == ("PROVIDER_ERROR",)
    assert fused.analyze(eye_sample, (), brain_candidates).decision.reason_codes == ("PROVIDER_ERROR",)


def test_keyboard_interrupt_is_not_swallowed(eye_sample, brain_candidates):
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(KeyboardInterrupt()), SpyDecision(None), 0.8)

    with pytest.raises(KeyboardInterrupt):
        brain.analyze(eye_sample, (), brain_candidates)

    fused = MultimodalApiBrain(SpyFused(SystemExit()), 0.8)
    with pytest.raises(SystemExit):
        fused.analyze(eye_sample, (), brain_candidates)


@pytest.mark.parametrize("returned", [object(), {"measurement_id": "measurement-001"}, None])
def test_invalid_perception_schema_fails_closed(eye_sample, brain_candidates, returned):
    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(returned), SpyDecision(None), 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.decision.reason_codes == ("INVALID_PERCEPTION_SCHEMA",)


@pytest.mark.parametrize("returned", [object(), {"action": "PROPOSE"}, None])
def test_invalid_decision_schema_fails_closed(eye_sample, brain_candidates, returned):
    perception = _perception(eye_sample)
    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(perception), SpyDecision(returned), 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.decision.reason_codes == ("INVALID_DECISION_SCHEMA",)
    assert result.perception == perception
    assert result.perception is not perception


def test_foreign_perception_or_decision_measurement_fails_closed(eye_sample, brain_candidates):
    foreign_perception = _perception(eye_sample, measurement_id="other-measurement")
    perception_result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(foreign_perception), SpyDecision(None), 0.8).analyze(eye_sample, (), brain_candidates)
    foreign_decision = _proposal(eye_sample, brain_candidates[0], based_on_measurement_id="other-measurement")
    decision_result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(_perception(eye_sample)), SpyDecision(foreign_decision), 0.8).analyze(eye_sample, (), brain_candidates)

    assert perception_result.decision.reason_codes == ("MEASUREMENT_ID_MISMATCH",)
    assert decision_result.decision.reason_codes == ("MEASUREMENT_ID_MISMATCH",)
    assert decision_result.perception.provider_id == "local-vision"


@pytest.mark.parametrize("decision", [
    lambda sample, candidates: _proposal(sample, candidates[0], candidate_id="foreign", requested_params={"bias_ma": 9.0}),
    lambda sample, candidates: _proposal(sample, candidates[0], requested_params={"bias_ma": 7.0}),
    lambda sample, candidates: _proposal(sample, candidates[0], requested_params={"bias_ma": 6.0, "equalizer": {"tap_1": 2.0}}),
])
def test_proposal_must_exactly_match_supplied_candidate(eye_sample, brain_candidates, decision):
    perception = _perception(eye_sample)
    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, SpyScorer(perception), SpyDecision(decision(eye_sample, brain_candidates)), 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.decision.reason_codes == ("CANDIDATE_NOT_IN_INPUT_SET",)
    assert result.decision.candidate_id is result.decision.requested_params is None
    assert result.perception == perception
    assert result.perception is not perception


def test_fused_invalid_schema_uses_shared_final_validation(eye_sample, brain_candidates):
    perception = _perception(eye_sample, provider_id="api")
    result = MultimodalApiBrain(SpyFused((perception, object())), 0.8).analyze(eye_sample, (), brain_candidates)

    assert result.decision.reason_codes == ("INVALID_DECISION_SCHEMA",)
    assert result.perception == perception
    assert result.perception is not perception


class MetadataBomb:
    @property
    def provider_id(self):
        raise RuntimeError("metadata secret")

    @property
    def provider_version(self):
        raise RuntimeError("metadata secret")

    def score(self, sample):
        raise RuntimeError("provider secret")


def test_provider_metadata_cannot_break_fail_closed_result(eye_sample, brain_candidates):
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, MetadataBomb(), SpyDecision(None), 0.8)
    result = brain.analyze(eye_sample, (), brain_candidates)
    fused_brain = MultimodalApiBrain(MetadataBomb(), 0.8)
    fused_result = fused_brain.analyze(eye_sample, (), brain_candidates)

    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert "secret" not in result.model_dump_json()
    assert brain.provider_key == "brain-guard"
    assert brain.provider_version == "1"
    assert result.perception.input_hash == eye_sample.raw_eye_density_ref.sha256
    assert result.perception.ood is True
    assert fused_brain.provider_key == "brain-guard"
    assert fused_brain.provider_version == "1"
    assert fused_result.decision.reason_codes == ("PROVIDER_ERROR",)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("provider_id", None), ("provider_version", " ")],
)
def test_invalid_provider_metadata_falls_back_to_guard_identity(
    eye_sample,
    brain_candidates,
    attribute,
    value,
):
    scorer = SpyScorer(_perception(eye_sample))
    setattr(scorer, attribute, value)
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, SpyDecision(None), 0.8)

    result = brain.analyze(eye_sample, (), brain_candidates)

    assert brain.provider_key == "brain-guard"
    assert brain.provider_version == "1"
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert result.perception.provider_id == "brain-guard"


@pytest.mark.parametrize(
    ("scorer_id", "decision_id"),
    [("left|right", "tail"), ("left", "right|tail")],
)
def test_reserved_identity_separator_prevents_composition_collisions(
    eye_sample,
    brain_candidates,
    scorer_id,
    decision_id,
):
    scorer = SpyScorer(_perception(eye_sample))
    scorer.provider_id = scorer_id
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))
    decision.provider_id = decision_id
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8)

    result = brain.analyze(eye_sample, (), brain_candidates)

    assert brain.provider_key == "brain-guard"
    assert brain.provider_version == "1"
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert scorer.calls == decision.calls == 0
    assert "left" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("attribute", "unsafe_value"),
    [
        ("provider_id", " leading"),
        ("provider_id", "trailing "),
        ("provider_id", "secret\nvalue"),
        ("provider_id", "secret\x00value"),
        ("provider_id", "secret\u200bvalue"),
        ("provider_id", "secret\ud800value"),
        ("provider_id", "secret|value"),
        ("provider_version", "secret|version"),
    ],
)
def test_unsafe_provider_metadata_fails_closed_without_leakage(
    eye_sample,
    brain_candidates,
    attribute,
    unsafe_value,
):
    scorer = SpyScorer(_perception(eye_sample))
    setattr(scorer, attribute, unsafe_value)
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))
    brain = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8)

    result = brain.analyze(eye_sample, (), brain_candidates)
    rendered = result.model_dump_json()

    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert scorer.calls == decision.calls == 0
    assert "secret" not in rendered
    assert "\\n" not in rendered
    assert "\\u0000" not in rendered


@pytest.mark.parametrize("field", ["provider_id", "provider_version", "input_hash"])
def test_composed_rejects_spoofed_perception_before_decision(field, eye_sample, brain_candidates):
    changes = {field: "spoofed" if field != "input_hash" else "b" * 64}
    scorer = SpyScorer(_perception(eye_sample, **changes))
    decision = SpyDecision(_proposal(eye_sample, brain_candidates[0]))

    result = ComposedBrain(BrainMode.LOCAL_VISION_LOCAL, scorer, decision, 0.8).analyze(
        eye_sample, (), brain_candidates
    )

    assert result.decision.reason_codes == (
        "INPUT_HASH_MISMATCH" if field == "input_hash" else "PROVIDER_IDENTITY_MISMATCH",
    )
    assert result.perception.provider_id == "brain-guard"
    assert decision.calls == 0


@pytest.mark.parametrize("field", ["provider_id", "provider_version"])
def test_composed_rejects_spoofed_decision_but_keeps_verified_perception(field, eye_sample, brain_candidates):
    proposal = _proposal(eye_sample, brain_candidates[0], **{field: "spoofed"})
    perception = _perception(eye_sample)
    result = ComposedBrain(
        BrainMode.LOCAL_VISION_LOCAL, SpyScorer(perception), SpyDecision(proposal), 0.8
    ).analyze(eye_sample, (), brain_candidates)

    assert result.decision.reason_codes == ("PROVIDER_IDENTITY_MISMATCH",)
    assert result.perception == perception
    assert result.perception is not perception


@pytest.mark.parametrize("component", [0, 1])
def test_fused_rejects_spoofed_output_identity(component, eye_sample, brain_candidates):
    perception = _perception(eye_sample, provider_id="api")
    decision = _proposal(eye_sample, brain_candidates[0], provider_id="api")
    if component == 0:
        perception = perception.model_copy(update={"provider_id": "spoofed"})
    else:
        decision = decision.model_copy(update={"provider_id": "spoofed"})
    result = MultimodalApiBrain(SpyFused((perception, decision)), 0.8).analyze(
        eye_sample, (), brain_candidates
    )

    assert result.decision.reason_codes == ("PROVIDER_IDENTITY_MISMATCH",)
    assert result.perception.provider_id == ("api" if component else "brain-guard")


@pytest.mark.parametrize("fused", [False, True])
def test_constructed_subclass_perception_cannot_bypass_direct_brain_gates(
    fused, eye_sample, brain_candidates
):
    perception = _constructed_evil_perception(eye_sample)
    if fused:
        result = MultimodalApiBrain(
            SpyFused((perception, _proposal(eye_sample, brain_candidates[0], provider_id="api"))),
            0.8,
        ).analyze(eye_sample, (), brain_candidates)
    else:
        result = ComposedBrain(
            BrainMode.LOCAL_VISION_LOCAL,
            SpyScorer(perception),
            SpyDecision(_proposal(eye_sample, brain_candidates[0])),
            0.8,
        ).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.perception.provider_id == "brain-guard"


@pytest.mark.parametrize("fused", [False, True])
def test_direct_brains_gate_unexposed_candidates_against_provider_mutation(
    fused, eye_sample, brain_candidates
):
    original_candidate = brain_candidates[0].model_dump(round_trip=True)
    if fused:
        result = MultimodalApiBrain(MutatingFusedProvider(), 0.8).analyze(
            eye_sample, (), brain_candidates
        )
    else:
        result = ComposedBrain(
            BrainMode.LOCAL_VISION_LOCAL,
            SpyScorer(_perception(eye_sample)),
            MutatingDecisionProvider(),
            0.8,
        ).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert brain_candidates[0].model_dump(round_trip=True) == original_candidate


@pytest.mark.parametrize(("candidate_value", "decision_value"), [(1, 1.0), (1.0, 1)])
def test_direct_composed_requires_json_exact_candidate_parameters(
    candidate_value, decision_value, eye_sample
):
    candidate = BrainCandidate(candidate_id="numeric", requested_params={"bias_ma": candidate_value})
    decision = _proposal(eye_sample, candidate, requested_params={"bias_ma": decision_value})

    result = ComposedBrain(
        BrainMode.LOCAL_VISION_LOCAL,
        SpyScorer(_perception(eye_sample)),
        SpyDecision(decision),
        0.8,
    ).analyze(eye_sample, (), (candidate,))

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("CANDIDATE_NOT_IN_INPUT_SET",)


def test_v2_brain_import_has_no_hardware_network_or_optional_sdk_side_effects():
    source_root = Path(__file__).parents[3] / "src"
    script = textwrap.dedent(
        f"""
        import builtins
        import sys

        sys.path.insert(0, {str(source_root)!r})
        real_import = builtins.__import__
        forbidden = {{
            "fastapi", "pyvisa", "serial", "socket",
            "vcsel_ai_tuning.hardware", "vcsel_ai_tuning.persistence",
        }}

        def block_forbidden(name, globals=None, locals=None, fromlist=(), level=0):
            if name in forbidden or any(name.startswith(root + ".") for root in forbidden):
                raise ModuleNotFoundError("forbidden import", name=name)
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_forbidden
        import vcsel_ai_tuning.v2
        assert vcsel_ai_tuning.v2.ComposedBrain.__name__ == "ComposedBrain"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
