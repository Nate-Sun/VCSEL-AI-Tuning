from __future__ import annotations

import pytest

from vcsel_ai_tuning.optimization.v11.coarse import CoarsePlanner
from vcsel_ai_tuning.optimization.v11.grid import (
    DiscreteParameterSpace,
    TuningParameter,
)
from vcsel_ai_tuning.optimization.v11.models import (
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
    OptimizerStatus,
)


@pytest.fixture
def space() -> DiscreteParameterSpace:
    return DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=8,
                tuning_step=2,
            ),
            TuningParameter(
                name="ibias_dac",
                tuning_min=70,
                tuning_max=78,
                tuning_step=2,
            ),
        )
    )


def _fingerprint() -> ExperimentFingerprint:
    return ExperimentFingerprint(
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint="b" * 64,
    )


def _observation(
    observation_id: str,
    *,
    tdecq_db: float | None,
    er_db: float | None = None,
    eq: int = 2,
    ibias_dac: int = 72,
) -> I2CObservation:
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)
        if tdecq_db is not None
        else MetricResult.invalid(status="INV", reason="invalid acquisition")
    )
    point = {"eq": eq, "ibias_dac": ibias_dac}
    return I2CObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        requested_i2c=point,
        readback_i2c=point,
        result=result,
        fingerprint=_fingerprint(),
    )


def test_coarse_generates_a_seeded_sparse_replayable_subset(space):
    first = tuple(CoarsePlanner(space, budget=7, seed=19).candidates())
    second = tuple(CoarsePlanner(space, budget=7, seed=19).candidates())

    assert first == second
    assert len(first) == 7 < space.point_count
    assert len({tuple(item.requested_i2c.items()) for item in first}) == 7
    assert all(item.model_version == "i2c-coarse-v1.1" for item in first)


def test_coarse_does_not_materialize_a_huge_cartesian_product():
    huge = DiscreteParameterSpace(
        (
            TuningParameter(
                name="a",
                tuning_min=0,
                tuning_max=999_999,
                tuning_step=1,
            ),
            TuningParameter(
                name="b",
                tuning_min=0,
                tuning_max=999_999,
                tuning_step=1,
            ),
        )
    )

    candidates = tuple(CoarsePlanner(huge, budget=5, seed=7).candidates())

    assert len(candidates) == 5
    assert huge.point_count == 1_000_000_000_000


def test_budget_larger_than_space_visits_each_point_once(space):
    candidates = tuple(
        CoarsePlanner(space, budget=space.point_count + 10, seed=0).candidates()
    )

    assert len(candidates) == space.point_count
    assert len({tuple(item.requested_i2c.items()) for item in candidates}) == (
        space.point_count
    )


def test_coarse_completion_preserves_invalid_history_and_reports_no_valid(space):
    planner = CoarsePlanner(space, budget=2, seed=0)
    history = (
        _observation("obs-2", tdecq_db=None, eq=4),
        _observation("obs-1", tdecq_db=None, eq=2),
    )

    summary = planner.summarize(history)

    assert summary.status is OptimizerStatus.NO_VALID_TDECQ
    assert summary.best_observation is None
    assert summary.observations == history
    assert all(item.result.tdecq_db is None for item in summary.observations)


def test_coarse_best_ignores_er_for_exact_tdecq_tie(space):
    planner = CoarsePlanner(space, budget=3, seed=0)
    lower_tdecq_low_er = _observation(
        "obs-lower",
        tdecq_db=2.49,
        er_db=1.0,
        eq=2,
    )
    equal_tdecq_high_er = _observation(
        "obs-high-er",
        tdecq_db=2.5,
        er_db=100.0,
        eq=4,
    )
    equal_tdecq_higher_er = _observation(
        "obs-tie-er",
        tdecq_db=2.49,
        er_db=2.0,
        eq=6,
    )

    summary = planner.summarize(
        (lower_tdecq_low_er, equal_tdecq_high_er, equal_tdecq_higher_er)
    )

    assert summary.status is OptimizerStatus.COMPLETE
    assert summary.best_observation == lower_tdecq_low_er


@pytest.mark.parametrize("budget", [0, -1, True])
def test_coarse_requires_a_positive_integer_budget(space, budget):
    with pytest.raises(ValueError, match="budget"):
        CoarsePlanner(space, budget=budget, seed=0)


def test_coarse_requires_an_integer_seed(space):
    with pytest.raises(ValueError, match="seed"):
        CoarsePlanner(space, budget=2, seed=True)
