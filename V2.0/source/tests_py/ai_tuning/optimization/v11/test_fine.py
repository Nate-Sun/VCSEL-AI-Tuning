from __future__ import annotations

import pytest

from vcsel_ai_tuning.optimization import v11
from vcsel_ai_tuning.optimization.v11 import fine as fine_module
from vcsel_ai_tuning.optimization.v11.coarse import CoarseSummary
from vcsel_ai_tuning.optimization.v11.fine import DiscreteBayesianFine
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
                tuning_max=4,
                tuning_step=1,
            ),
        )
    )


@pytest.fixture
def fingerprint() -> ExperimentFingerprint:
    return ExperimentFingerprint(
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint="c" * 64,
    )


def _observation(
    observation_id: str,
    *,
    requested: int,
    readback: int,
    fingerprint: ExperimentFingerprint,
    tdecq_db: float | None,
    er_db: float | None = None,
) -> I2CObservation:
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)
        if tdecq_db is not None
        else MetricResult.invalid(
            status="INV",
            reason="DCA returned an invalid TDECQ",
        )
    )
    return I2CObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        requested_i2c={"eq": requested},
        readback_i2c={"eq": readback},
        result=result,
        fingerprint=fingerprint,
    )


def _coarse_summary(
    observations: tuple[I2CObservation, ...],
    *,
    best: I2CObservation | None,
    status: OptimizerStatus,
) -> CoarseSummary:
    return CoarseSummary(
        status=status,
        observations=observations,
        best_observation=best,
        candidate_count=len(observations),
        budget=max(len(observations), 1),
    )


def test_build_fine_from_coarse_uses_step_radius_and_global_features(
    fingerprint,
):
    global_space = DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=8,
                tuning_step=2,
            ),
        )
    )
    history = tuple(
        _observation(
            observation_id,
            requested=requested,
            readback=readback,
            fingerprint=fingerprint,
            tdecq_db=value,
        )
        for observation_id, requested, readback, value in (
            ("obs-0", 0, 0, 3.0),
            ("obs-best-request-2-read-4", 2, 4, 2.4),
            ("obs-8", 8, 8, 2.8),
        )
    )
    coarse = _coarse_summary(
        history,
        best=history[1],
        status=OptimizerStatus.COMPLETE,
    )

    bootstrap = fine_module.build_fine_from_coarse(
        global_space,
        coarse,
        radius_steps=1,
        seed=17,
        min_valid_observations=3,
        length_scale=0.2,
        xi=0.05,
        model_jitter=1e-7,
        candidate_chunk_size=4,
    )

    assert bootstrap.status is OptimizerStatus.CANDIDATE
    assert bootstrap.ready is True
    assert bootstrap.optimizer is not None
    assert tuple(bootstrap.optimizer.space.iter_points()) == (
        {"eq": 2},
        {"eq": 4},
        {"eq": 6},
    )
    assert bootstrap.optimizer.feature_space is global_space
    assert bootstrap.optimizer.seed == 17
    assert bootstrap.optimizer.min_valid_observations == 3
    assert bootstrap.optimizer.length_scale == pytest.approx(0.2)
    assert bootstrap.optimizer.xi == pytest.approx(0.05)
    assert bootstrap.optimizer.model_jitter == pytest.approx(1e-7)
    assert bootstrap.optimizer.candidate_chunk_size == 4

    decision = bootstrap.optimizer.suggest(history)

    assert decision.status is OptimizerStatus.CANDIDATE
    assert decision.candidate is not None
    assert decision.candidate.requested_i2c in ({"eq": 4}, {"eq": 6})


def test_build_fine_from_coarse_preserves_not_ready_without_fallback(
    space,
    fingerprint,
):
    best = _observation(
        "obs-best",
        requested=2,
        readback=2,
        fingerprint=fingerprint,
        tdecq_db=2.4,
    )
    coarse = _coarse_summary(
        (best,),
        best=best,
        status=OptimizerStatus.COMPLETE,
    )

    bootstrap = fine_module.build_fine_from_coarse(
        space,
        coarse,
        radius_steps=1,
        seed=3,
    )

    assert bootstrap.optimizer is not None
    decision = bootstrap.optimizer.suggest((best,))
    assert decision.status is OptimizerStatus.NOT_READY
    assert decision.candidate is None


def test_build_fine_from_coarse_reports_no_valid_tdecq(
    space,
    fingerprint,
):
    invalid = _observation(
        "obs-invalid",
        requested=2,
        readback=2,
        fingerprint=fingerprint,
        tdecq_db=None,
    )
    coarse = _coarse_summary(
        (invalid,),
        best=None,
        status=OptimizerStatus.NO_VALID_TDECQ,
    )

    bootstrap = fine_module.build_fine_from_coarse(
        space,
        coarse,
        radius_steps=1,
        seed=3,
    )

    assert bootstrap.status is OptimizerStatus.NO_VALID_TDECQ
    assert bootstrap.optimizer is None
    assert "valid tdecq" in bootstrap.reason.lower()


def test_build_fine_from_coarse_reports_not_ready_for_missing_complete_best(
    space,
):
    inconsistent = CoarseSummary.model_construct(
        status=OptimizerStatus.COMPLETE,
        observations=(),
        best_observation=None,
        candidate_count=0,
        budget=1,
    )

    bootstrap = fine_module.build_fine_from_coarse(
        space,
        inconsistent,
        radius_steps=1,
        seed=3,
    )

    assert bootstrap.status is OptimizerStatus.NOT_READY
    assert bootstrap.optimizer is None
    assert "best observation" in bootstrap.reason.lower()


def test_build_fine_from_coarse_is_exported_from_v11():
    assert v11.build_fine_from_coarse is fine_module.build_fine_from_coarse
    assert v11.FineBootstrap is fine_module.FineBootstrap


def test_fine_aggregates_on_actual_readback_and_uses_tdecq_median(
    space,
    fingerprint,
):
    history = (
        _observation(
            "obs-1",
            requested=4,
            readback=2,
            fingerprint=fingerprint,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-2",
            requested=4,
            readback=2,
            fingerprint=fingerprint,
            tdecq_db=2.0,
        ),
        _observation(
            "obs-3",
            requested=4,
            readback=2,
            fingerprint=fingerprint,
            tdecq_db=2.4,
        ),
    )

    aggregated = DiscreteBayesianFine(space, seed=5).aggregate(history)

    assert len(aggregated) == 1
    assert aggregated[0].readback_i2c == {"eq": 2}
    assert aggregated[0].result.tdecq_db == pytest.approx(2.4)


def test_fine_excludes_requested_points_not_readback_points(space, fingerprint):
    history = (
        _observation(
            "obs-request-0-read-1",
            requested=0,
            readback=1,
            fingerprint=fingerprint,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-request-2-read-2",
            requested=2,
            readback=2,
            fingerprint=fingerprint,
            tdecq_db=2.4,
        ),
        _observation(
            "obs-request-3-read-3",
            requested=3,
            readback=3,
            fingerprint=fingerprint,
            tdecq_db=2.5,
        ),
        _observation(
            "obs-request-4-read-4",
            requested=4,
            readback=4,
            fingerprint=fingerprint,
            tdecq_db=2.7,
        ),
    )

    decision = DiscreteBayesianFine(space, seed=7).suggest(history)

    assert decision.status is OptimizerStatus.CANDIDATE
    assert decision.candidate is not None
    assert decision.candidate.requested_i2c == {"eq": 1}


def test_fine_keeps_invalid_tdecq_null_and_scores_validity_separately(
    space,
    fingerprint,
):
    history = (
        _observation(
            "obs-valid-0",
            requested=0,
            readback=0,
            fingerprint=fingerprint,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-invalid-1",
            requested=1,
            readback=1,
            fingerprint=fingerprint,
            tdecq_db=None,
        ),
        _observation(
            "obs-valid-4",
            requested=4,
            readback=4,
            fingerprint=fingerprint,
            tdecq_db=2.4,
        ),
    )
    optimizer = DiscreteBayesianFine(space, seed=3)

    aggregated = optimizer.aggregate(history)
    decision = optimizer.suggest(history)

    invalid = next(row for row in aggregated if row.readback_i2c == {"eq": 1})
    assert invalid.result.tdecq_db is None
    assert decision.status is OptimizerStatus.CANDIDATE
    assert decision.candidate is not None
    assert 0.0 <= decision.candidate.predicted_metrics["valid_probability"] < 1.0


def test_fine_is_not_ready_without_enough_valid_tdecq_and_never_falls_back(
    space,
    fingerprint,
):
    history = (
        _observation(
            "obs-only-valid",
            requested=0,
            readback=0,
            fingerprint=fingerprint,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-invalid",
            requested=1,
            readback=1,
            fingerprint=fingerprint,
            tdecq_db=None,
        ),
    )

    decision = DiscreteBayesianFine(space, seed=3).suggest(history)

    assert decision.status is OptimizerStatus.NOT_READY
    assert decision.candidate is None
    assert "valid" in decision.reason.lower()
    assert "fallback" not in decision.reason.lower()


def test_fine_rejects_mixed_experiment_fingerprints(space, fingerprint):
    other = fingerprint.model_copy(update={"upper_nlc": 7})
    history = (
        _observation(
            "obs-current",
            requested=0,
            readback=0,
            fingerprint=fingerprint,
            tdecq_db=2.8,
        ),
        _observation(
            "obs-other",
            requested=1,
            readback=1,
            fingerprint=other,
            tdecq_db=2.4,
        ),
    )

    decision = DiscreteBayesianFine(space, seed=3).suggest(history)

    assert decision.status is OptimizerStatus.NOT_READY
    assert decision.candidate is None
    assert "fingerprint" in decision.reason.lower()


def test_fine_suggestion_is_deterministic_and_reports_model_metadata(
    space,
    fingerprint,
):
    history = tuple(
        _observation(
            f"obs-{eq}",
            requested=eq,
            readback=eq,
            fingerprint=fingerprint,
            tdecq_db=value,
        )
        for eq, value in ((0, 3.0), (2, 2.4), (4, 2.8))
    )

    first = DiscreteBayesianFine(space, seed=11).suggest(history)
    second = DiscreteBayesianFine(space, seed=11).suggest(tuple(reversed(history)))

    assert first == second
    assert first.candidate is not None
    assert first.candidate.model_version == "i2c-discrete-gp-v1.1"
    assert first.candidate.reason == (
        "maximum discrete TDECQ expected improvement times validity probability"
    )
    assert set(first.candidate.predicted_metrics) == {
        "predicted_tdecq_db",
        "predicted_sigma_db",
        "expected_improvement",
        "valid_probability",
        "acquisition",
    }


def test_fine_best_observation_uses_er_only_for_exact_tdecq_tie(
    space,
    fingerprint,
):
    history = (
        _observation(
            "obs-low-tdecq",
            requested=0,
            readback=0,
            fingerprint=fingerprint,
            tdecq_db=2.4,
            er_db=1.0,
        ),
        _observation(
            "obs-high-er-worse-tdecq",
            requested=1,
            readback=1,
            fingerprint=fingerprint,
            tdecq_db=2.5,
            er_db=100.0,
        ),
        _observation(
            "obs-tied-higher-er",
            requested=2,
            readback=2,
            fingerprint=fingerprint,
            tdecq_db=2.4,
            er_db=2.0,
        ),
    )

    decision = DiscreteBayesianFine(space, seed=11).suggest(history)

    assert decision.best_observation is not None
    assert decision.best_observation.readback_i2c == {"eq": 2}


def test_fine_reports_not_ready_when_numpy_model_fit_fails(
    monkeypatch,
    space,
    fingerprint,
):
    history = tuple(
        _observation(
            f"obs-{eq}",
            requested=eq,
            readback=eq,
            fingerprint=fingerprint,
            tdecq_db=value,
        )
        for eq, value in ((0, 3.0), (2, 2.4), (4, 2.8))
    )

    def fail_cholesky(*args, **kwargs):
        raise ValueError("synthetic model failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.v11.fine.np.linalg.cholesky",
        fail_cholesky,
    )

    decision = DiscreteBayesianFine(space, seed=11).suggest(history)

    assert decision.status is OptimizerStatus.NOT_READY
    assert decision.candidate is None
    assert "model" in decision.reason.lower()


def test_fine_reports_not_ready_when_numpy_is_unavailable(
    monkeypatch,
    space,
    fingerprint,
):
    history = tuple(
        _observation(
            f"obs-{eq}",
            requested=eq,
            readback=eq,
            fingerprint=fingerprint,
            tdecq_db=value,
        )
        for eq, value in ((0, 3.0), (2, 2.4), (4, 2.8))
    )
    monkeypatch.setattr("vcsel_ai_tuning.optimization.v11.fine.np", None)

    decision = DiscreteBayesianFine(space, seed=11).suggest(history)

    assert decision.status is OptimizerStatus.NOT_READY
    assert decision.candidate is None
    assert "numpy" in decision.reason.lower()


@pytest.mark.parametrize("seed", [True, 1.5])
def test_fine_requires_an_integer_seed(space, seed):
    with pytest.raises(ValueError, match="seed"):
        DiscreteBayesianFine(space, seed=seed)


def test_candidate_space_must_be_a_true_subset_of_the_feature_grid():
    feature_space = DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=6,
                tuning_step=2,
            ),
        )
    )
    misaligned_candidates = DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=6,
                tuning_step=3,
            ),
        )
    )

    with pytest.raises(ValueError, match="feature-space grid"):
        DiscreteBayesianFine(
            misaligned_candidates,
            seed=0,
            feature_space=feature_space,
        )


def test_validity_probability_is_exactly_replayable_under_history_reordering(
    space,
    fingerprint,
):
    history = tuple(
        _observation(
            f"obs-{index:04d}",
            requested=0,
            readback=index % 5,
            fingerprint=fingerprint,
            tdecq_db=(
                None
                if index % 7 == 0
                else 2.4 + ((index % 11) / 100.0)
            ),
        )
        for index in range(1001)
    )

    first = DiscreteBayesianFine(space, seed=23).suggest(history)
    reversed_history = DiscreteBayesianFine(space, seed=23).suggest(
        tuple(reversed(history))
    )

    assert first == reversed_history
