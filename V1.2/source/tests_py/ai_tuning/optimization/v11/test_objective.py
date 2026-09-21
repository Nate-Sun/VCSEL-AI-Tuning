from __future__ import annotations

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.optimization.v11.models import (
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
)
from vcsel_ai_tuning.optimization.v11.objective import (
    Comparison,
    aggregate_repeated_observations,
    best_observation,
    compare_results,
)


def _fingerprint(*, main: int = 104) -> ExperimentFingerprint:
    return ExperimentFingerprint(
        fir_taps=(0, 9, -28, main, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint="a" * 64,
    )


def _observation(
    observation_id: str,
    *,
    requested: int = 7,
    readback: int = 7,
    tdecq_db: float | None = 2.5,
    er_db: float | None = 3.0,
    fingerprint: ExperimentFingerprint | None = None,
) -> I2CObservation:
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)
        if tdecq_db is not None
        else MetricResult.invalid(
            status="INV",
            reason="instrument returned INV",
            er_db=er_db,
        )
    )
    return I2CObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        requested_i2c={"eq": requested},
        readback_i2c={"eq": readback},
        result=result,
        fingerprint=fingerprint or _fingerprint(),
    )


def test_lower_tdecq_always_wins_even_with_much_lower_er():
    candidate = MetricResult.valid(tdecq_db=2.99, er_db=1.0)
    reference = MetricResult.valid(tdecq_db=3.0, er_db=100.0)

    assert compare_results(candidate, reference) is Comparison.BETTER


def test_any_nonzero_tdecq_improvement_is_meaningful_without_epsilon():
    candidate = MetricResult.valid(tdecq_db=2.999999999, er_db=1.0)
    reference = MetricResult.valid(tdecq_db=3.0, er_db=100.0)

    assert compare_results(candidate, reference) is Comparison.BETTER


def test_er_never_breaks_an_exact_tdecq_tie():
    candidate = MetricResult.valid(tdecq_db=2.5, er_db=3.0)
    reference = MetricResult.valid(tdecq_db=2.5, er_db=2.0)

    assert compare_results(candidate, reference) is Comparison.EQUAL
    assert compare_results(reference, candidate) is Comparison.EQUAL


def test_equal_tdecq_with_missing_er_is_equal():
    candidate = MetricResult.valid(tdecq_db=2.5, er_db=None)
    reference = MetricResult.valid(tdecq_db=2.5, er_db=99.0)

    assert compare_results(candidate, reference) is Comparison.EQUAL
    assert compare_results(reference, candidate) is Comparison.EQUAL


def test_invalid_candidate_is_never_replaced_by_a_numeric_penalty():
    invalid = MetricResult.invalid(
        status="INV",
        reason="instrument returned INV",
    )
    valid = MetricResult.valid(tdecq_db=9.0, er_db=None)

    assert invalid.tdecq_db is None
    assert compare_results(invalid, valid) is Comparison.INVALID
    assert compare_results(invalid, invalid) is Comparison.INVALID
    assert compare_results(valid, invalid) is Comparison.BETTER


def test_metric_result_enforces_status_value_and_reason_consistency():
    with pytest.raises(ValidationError, match="CORR"):
        MetricResult(
            tdecq_db=None,
            tdecq_status="CORR",
            tdecq_invalid_reason=None,
        )


def test_metric_result_enforces_er_status_and_value_consistency():
    with pytest.raises(ValidationError, match="ER|er_status"):
        MetricResult(
            tdecq_db=2.5,
            tdecq_status="CORR",
            er_db=99.0,
            er_status="INV",
        )
    with pytest.raises(ValidationError, match="ER|er_status"):
        MetricResult(
            tdecq_db=2.5,
            tdecq_status="CORR",
            er_db=None,
            er_status="CORR",
        )


def test_non_corr_er_never_breaks_an_exact_tdecq_tie():
    invalid_er = MetricResult(
        tdecq_db=2.5,
        tdecq_status="CORR",
        er_db=None,
        er_status="INV",
    )
    valid_er = MetricResult.valid(tdecq_db=2.5, er_db=3.0)

    assert invalid_er.has_valid_er is False
    assert compare_results(invalid_er, valid_er) is Comparison.EQUAL
    assert compare_results(valid_er, invalid_er) is Comparison.EQUAL

    with pytest.raises(ValidationError, match="non-CORR|invalid reason"):
        MetricResult(
            tdecq_db=None,
            tdecq_status="INV",
            tdecq_invalid_reason=None,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_result_rejects_nonfinite_values(value):
    with pytest.raises(ValidationError):
        MetricResult.valid(tdecq_db=value, er_db=3.0)


def test_observation_is_immutable_and_round_trips_as_json():
    observation = _observation("obs-1", requested=9, readback=7)

    restored = I2CObservation.model_validate_json(observation.model_dump_json())

    assert restored == observation
    assert restored.requested_i2c == {"eq": 9}
    assert restored.readback_i2c == {"eq": 7}
    with pytest.raises(TypeError):
        restored.readback_i2c["eq"] = 99
    with pytest.raises(ValidationError):
        restored.observation_id = "changed"


def test_fingerprint_requires_exactly_seven_taps_and_sha256():
    with pytest.raises(ValidationError):
        ExperimentFingerprint(
            fir_taps=(0, 0, 104),
            upper_nlc=6,
            lower_nlc=6,
            dca_config_fingerprint="a" * 64,
        )
    with pytest.raises(ValidationError):
        ExperimentFingerprint(
            fir_taps=(0, 9, -28, 104, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
            dca_config_fingerprint="not-a-sha256",
        )


def test_repeated_readback_uses_median_and_preserves_raw_observations():
    observations = (
        _observation("obs-3", requested=9, readback=7, tdecq_db=2.8, er_db=2.0),
        _observation("obs-1", requested=8, readback=7, tdecq_db=2.0, er_db=4.0),
        _observation("obs-2", requested=7, readback=7, tdecq_db=2.4, er_db=3.0),
    )

    aggregated = aggregate_repeated_observations(observations)

    assert len(aggregated) == 1
    assert aggregated[0].readback_i2c == {"eq": 7}
    assert aggregated[0].result.tdecq_db == 2.4
    assert aggregated[0].result.er_db == 3.0
    assert aggregated[0].raw_observations == tuple(
        sorted(observations, key=lambda item: item.observation_id)
    )


def test_all_invalid_repeats_remain_invalid_with_null_tdecq():
    observations = (
        _observation("obs-2", readback=7, tdecq_db=None, er_db=3.0),
        _observation("obs-1", readback=7, tdecq_db=None, er_db=None),
    )

    aggregated = aggregate_repeated_observations(observations)

    assert len(aggregated) == 1
    assert aggregated[0].result.tdecq_db is None
    assert aggregated[0].result.tdecq_status == "INVALID_AGGREGATE"
    assert aggregated[0].result.tdecq_invalid_reason == (
        "no valid TDECQ in repeated observations"
    )


def test_best_observation_uses_stable_id_for_exact_tdecq_tie():
    low_er = _observation("obs-a", tdecq_db=2.2, er_db=2.0)
    high_er_late_id = _observation("obs-z2", tdecq_db=2.2, er_db=3.0)
    high_er_early_id = _observation("obs-z1", tdecq_db=2.2, er_db=3.0)

    assert best_observation((low_er, high_er_late_id, high_er_early_id)) == (
        low_er
    )
