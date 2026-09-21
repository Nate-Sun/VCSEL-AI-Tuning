import pytest

from vcsel_ai_tuning.domain import Candidate, ParameterSpec
from vcsel_ai_tuning.safety import SafetyPolicy, SafetyViolation


def policy():
    return SafetyPolicy(
        (
            ParameterSpec(
                name="bias_ma",
                unit="mA",
                lower=4.0,
                upper=7.0,
                quantization=0.1,
                max_step=1.0,
                readback_tolerance=0.02,
            ),
            ParameterSpec(
                name="spacing_01_v",
                unit="V",
                lower=0.05,
                upper=0.4,
                quantization=0.01,
                max_step=0.05,
                readback_tolerance=0.005,
            ),
        )
    )


def test_candidate_is_quantized_before_it_passes():
    decision = policy().evaluate_candidate(
        Candidate(
            candidate_id="candidate-1",
            requested={"bias_ma": 5.04, "spacing_01_v": 0.123},
        ),
        reference={"bias_ma": 5.0, "spacing_01_v": 0.10},
    )

    assert decision.passed is True
    assert decision.applied == {"bias_ma": 5.0, "spacing_01_v": 0.12}


def test_candidate_step_is_measured_from_safe_reference():
    decision = policy().evaluate_candidate(
        Candidate(
            candidate_id="candidate-2",
            requested={"bias_ma": 6.2, "spacing_01_v": 0.10},
        ),
        reference={"bias_ma": 5.0, "spacing_01_v": 0.10},
    )

    assert decision.passed is False
    assert any(result.name == "bias_ma.max_step" for result in decision.constraints)


def test_missing_parameter_never_passes():
    decision = policy().evaluate_candidate(
        Candidate(candidate_id="candidate-3", requested={"bias_ma": 5.0}),
        reference={"bias_ma": 5.0, "spacing_01_v": 0.10},
    )

    assert decision.passed is False
    assert decision.applied == {}


def test_readback_mismatch_fails_and_require_raises():
    readback = policy().evaluate_readback(
        applied={"bias_ma": 5.0, "spacing_01_v": 0.10},
        readback={"bias_ma": 5.05, "spacing_01_v": 0.10},
    )

    assert readback.passed is False
    with pytest.raises(SafetyViolation, match="read-back"):
        readback.require("read-back")


def test_ffe_taps_are_l1_normalized_and_zero_norm_is_rejected():
    ffe_policy = SafetyPolicy(
        (
            ParameterSpec(name="ffe_tap_0", unit="1", lower=-1.0, upper=1.0),
            ParameterSpec(name="ffe_tap_1", unit="1", lower=-1.0, upper=1.0),
        )
    )
    normalized = ffe_policy.evaluate_candidate(
        Candidate(
            candidate_id="ffe-1",
            requested={"ffe_tap_0": 2.0 / 3.0, "ffe_tap_1": -1.0 / 3.0},
        ),
        reference={"ffe_tap_0": 1.0, "ffe_tap_1": 0.0},
    )
    zero = ffe_policy.evaluate_candidate(
        Candidate(
            candidate_id="ffe-2",
            requested={"ffe_tap_0": 0.0, "ffe_tap_1": 0.0},
        ),
        reference={"ffe_tap_0": 1.0, "ffe_tap_1": 0.0},
    )

    assert sum(abs(value) for value in normalized.applied.values()) == pytest.approx(1.0)
    assert zero.passed is False
    assert any(result.name == "ffe.l1_norm" for result in zero.constraints)
