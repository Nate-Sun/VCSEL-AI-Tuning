from __future__ import annotations

import pytest

from vcsel_ai_tuning.optimization.v11 import (
    DiscreteParameterSpace,
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
    OptimizerStatus,
    Pilot3Planner,
    PilotStrategy,
    TuningParameter,
)


@pytest.fixture
def space() -> DiscreteParameterSpace:
    return DiscreteParameterSpace(
        (
            TuningParameter(
                name="ibias_dac", tuning_min=69, tuning_max=73, tuning_step=2
            ),
            TuningParameter(
                name="res_gain", tuning_min=4, tuning_max=8, tuning_step=4
            ),
        )
    )


def _fingerprint(suffix: str = "a") -> ExperimentFingerprint:
    return ExperimentFingerprint(
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint=suffix * 64,
    )


def _observation(
    observation_id: str,
    candidate,
    *,
    tdecq_db: float | None,
    er_db: float | None = 3.0,
    fingerprint: ExperimentFingerprint | None = None,
) -> I2CObservation:
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)
        if tdecq_db is not None
        else MetricResult.invalid(status="INV", reason="DCA invalid", er_db=er_db)
    )
    return I2CObservation(
        observation_id=observation_id,
        candidate_id=candidate.candidate_id,
        requested_i2c=candidate.requested_i2c,
        readback_i2c=candidate.requested_i2c,
        result=result,
        fingerprint=fingerprint or _fingerprint(),
    )


def test_three_suggestions_are_observation_conditioned_and_replayable(space):
    planner = Pilot3Planner(space, seed=17)

    first = planner.suggest(())
    assert first.status is OptimizerStatus.CANDIDATE
    assert first.strategy is PilotStrategy.SEED
    assert first.candidate is not None

    observation_1 = _observation("obs-1", first.candidate, tdecq_db=3.1)
    second = planner.suggest((observation_1,))
    assert second.strategy is PilotStrategy.DIVERSITY_BOOTSTRAP
    assert second.observation_ids_used == ("obs-1",)
    assert second.candidate is not None

    observation_2 = _observation("obs-2", second.candidate, tdecq_db=2.8)
    history = (observation_1, observation_2)
    third = planner.suggest(history)
    assert third == planner.suggest(history)
    assert third.strategy is PilotStrategy.BAYESIAN_ACQUISITION
    assert third.observation_ids_used == ("obs-1", "obs-2")
    assert third.candidate is not None
    assert len(
        {
            tuple(first.candidate.requested_i2c.items()),
            tuple(second.candidate.requested_i2c.items()),
            tuple(third.candidate.requested_i2c.items()),
        }
    ) == 3


def test_pending_and_measured_points_are_never_reused(space):
    planner = Pilot3Planner(space, seed=17)
    original = planner.suggest(())
    assert original.candidate is not None

    replacement = planner.suggest(
        (), pending_i2c=(original.candidate.requested_i2c,)
    )

    assert replacement.candidate is not None
    assert replacement.candidate.requested_i2c != original.candidate.requested_i2c


def test_invalid_history_cannot_fit_the_bayesian_candidate(space):
    planner = Pilot3Planner(space, seed=17)
    first = planner.suggest(())
    assert first.candidate is not None
    invalid = _observation("obs-invalid", first.candidate, tdecq_db=None)
    second = planner.suggest((invalid,))
    assert second.candidate is not None
    valid = _observation("obs-valid", second.candidate, tdecq_db=2.9)

    third = planner.suggest((invalid, valid))

    assert third.status is OptimizerStatus.NOT_READY
    assert third.candidate is None


def test_er_does_not_change_a_tied_tdecq_bayesian_history(space):
    planner = Pilot3Planner(space, seed=17)
    first = planner.suggest(())
    assert first.candidate is not None
    first_low_er = _observation("obs-1", first.candidate, tdecq_db=2.5, er_db=1.0)
    first_high_er = _observation("obs-1", first.candidate, tdecq_db=2.5, er_db=99.0)
    second = planner.suggest((first_low_er,))
    assert second.candidate is not None
    second_low_er = _observation("obs-2", second.candidate, tdecq_db=2.5, er_db=1.0)
    second_high_er = _observation("obs-2", second.candidate, tdecq_db=2.5, er_db=99.0)

    low_er_decision = planner.suggest((first_low_er, second_low_er))
    high_er_decision = planner.suggest((first_high_er, second_high_er))

    assert low_er_decision.candidate == high_er_decision.candidate


def test_out_of_policy_requested_point_is_rejected(space):
    planner = Pilot3Planner(space, seed=17)
    candidate = planner.suggest(()).candidate
    assert candidate is not None
    observation = _observation("obs-1", candidate, tdecq_db=3.0)
    unsafe = observation.model_copy(
        update={"requested_i2c": {"ibias_dac": 75, "res_gain": 4}}
    )

    with pytest.raises(ValueError, match="grid"):
        planner.suggest((unsafe,))

