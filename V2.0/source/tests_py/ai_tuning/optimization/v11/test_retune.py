from __future__ import annotations

import pytest

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
from vcsel_ai_tuning.optimization.v11.retune import (
    RetunePlanner,
    build_local_space,
    select_retune_history,
)


def _fingerprint(**updates) -> ExperimentFingerprint:
    baseline = ExperimentFingerprint(
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint="d" * 64,
    )
    return baseline.model_copy(update=updates)


def _observation(
    observation_id: str,
    *,
    requested: int,
    readback: int,
    fingerprint: ExperimentFingerprint,
    tdecq_db: float,
) -> I2CObservation:
    return I2CObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        requested_i2c={"eq": requested},
        readback_i2c={"eq": readback},
        result=MetricResult.valid(tdecq_db=tdecq_db, er_db=None),
        fingerprint=fingerprint,
    )


def _space() -> DiscreteParameterSpace:
    return DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=9,
                tuning_step=3,
            ),
        )
    )


def test_retune_history_requires_complete_fingerprint_equality():
    current = _fingerprint()
    variants = (
        _fingerprint(fir_taps=(0, 9, -27, 104, -8, 1, -3)),
        _fingerprint(upper_nlc=7),
        _fingerprint(lower_nlc=7),
        _fingerprint(dca_config_fingerprint="e" * 64),
    )
    current_rows = (
        _observation(
            "obs-current-1",
            requested=0,
            readback=0,
            fingerprint=current,
            tdecq_db=2.6,
        ),
        _observation(
            "obs-current-2",
            requested=3,
            readback=3,
            fingerprint=current,
            tdecq_db=2.4,
        ),
    )
    foreign_rows = tuple(
        _observation(
            f"obs-foreign-{index}",
            requested=6,
            readback=6,
            fingerprint=variant,
            tdecq_db=1.0,
        )
        for index, variant in enumerate(variants)
    )
    history = (foreign_rows[0], current_rows[0], *foreign_rows[1:], current_rows[1])

    selected = select_retune_history(history, current)

    assert selected == current_rows


def test_local_retune_space_reuses_exact_global_grid_around_off_grid_readback():
    local = build_local_space(
        _space(),
        current_readback={"eq": 4},
        radius_steps=1,
    )

    assert local.names == ("eq",)
    assert local.parameters[0].tuning_min == 3
    assert local.parameters[0].tuning_max == 6
    assert local.parameters[0].tuning_step == 3
    assert tuple(local.iter_points()) == ({"eq": 3}, {"eq": 6})


def test_retune_preserves_all_history_but_models_only_current_context():
    current = _fingerprint()
    other = _fingerprint(fir_taps=(0, 0, 0, 104, 0, 0, 0))
    history = (
        _observation(
            "obs-current-0",
            requested=0,
            readback=0,
            fingerprint=current,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-other-3",
            requested=3,
            readback=3,
            fingerprint=other,
            tdecq_db=1.0,
        ),
        _observation(
            "obs-current-6",
            requested=6,
            readback=6,
            fingerprint=current,
            tdecq_db=2.4,
        ),
    )
    planner = RetunePlanner(
        _space(),
        fingerprint=current,
        current_readback={"eq": 3},
        radius_steps=1,
        seed=17,
    )

    result = planner.suggest(history)

    assert result.all_history == history
    assert tuple(row.observation_id for row in result.selected_history) == (
        "obs-current-0",
        "obs-current-6",
    )
    assert result.decision.status is OptimizerStatus.CANDIDATE
    assert result.decision.candidate is not None
    assert result.decision.candidate.requested_i2c == {"eq": 3}


def test_retune_with_no_matching_context_emits_explicit_bayesian_initial_design():
    current = _fingerprint()
    other = _fingerprint(upper_nlc=7)
    history = (
        _observation(
            "obs-other",
            requested=3,
            readback=3,
            fingerprint=other,
            tdecq_db=1.0,
        ),
    )
    planner = RetunePlanner(
        _space(),
        fingerprint=current,
        current_readback={"eq": 3},
        radius_steps=1,
        seed=17,
    )

    result = planner.suggest(history)

    assert result.all_history == history
    assert result.selected_history == ()
    assert result.decision.status is OptimizerStatus.CANDIDATE
    assert result.decision.candidate is not None
    assert result.decision.candidate.model_version == (
        "i2c-retune-initial-design-v1.1"
    )
    assert result.decision.candidate.requested_i2c in (
        {"eq": 0},
        {"eq": 3},
        {"eq": 6},
    )


def test_new_fir_repeats_at_one_readback_emit_a_second_initial_design_point():
    current = _fingerprint()
    history = tuple(
        _observation(
            f"obs-confirm-{index}",
            requested=3,
            readback=3,
            fingerprint=current,
            tdecq_db=tdecq,
        )
        for index, tdecq in enumerate((2.8, 2.6, 2.7))
    )
    planner = RetunePlanner(
        _space(),
        fingerprint=current,
        current_readback={"eq": 3},
        radius_steps=1,
        seed=17,
    )

    result = planner.suggest(history)

    assert result.decision.status is OptimizerStatus.CANDIDATE
    assert result.decision.candidate is not None
    assert result.decision.candidate.model_version == (
        "i2c-retune-initial-design-v1.1"
    )
    assert result.decision.candidate.requested_i2c != {"eq": 3}


def test_same_context_history_outside_local_candidate_window_can_still_train():
    current = _fingerprint()
    history = (
        _observation(
            "obs-current-0",
            requested=0,
            readback=0,
            fingerprint=current,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-current-6",
            requested=6,
            readback=6,
            fingerprint=current,
            tdecq_db=2.4,
        ),
        _observation(
            "obs-current-9-outside-local",
            requested=9,
            readback=9,
            fingerprint=current,
            tdecq_db=2.6,
        ),
    )
    planner = RetunePlanner(
        _space(),
        fingerprint=current,
        current_readback={"eq": 3},
        radius_steps=1,
        seed=17,
    )

    result = planner.suggest(history)

    assert result.selected_history == history
    assert result.decision.status is OptimizerStatus.CANDIDATE
    assert result.decision.candidate is not None
    assert result.decision.candidate.requested_i2c == {"eq": 3}


@pytest.mark.parametrize("radius_steps", [0, -1, True])
def test_local_space_requires_positive_integer_radius(radius_steps):
    with pytest.raises(ValueError, match="radius"):
        build_local_space(
            _space(),
            current_readback={"eq": 3},
            radius_steps=radius_steps,
        )
