from __future__ import annotations

from collections.abc import Mapping
import json
from math import exp, pi, sqrt

import numpy as np
import pytest
from pydantic import ValidationError
import sklearn

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.bayesian import (
    BayesianSnapshot,
    CandidatePrediction,
    TdecqBayesianOptimizer,
)
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    PathComplete,
    TuningObservation,
    TuningPath,
)


class _IntSubclass(int):
    pass


def _parameter(name: str, grid_values: tuple[int, ...]) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        unit="count",
        lower=min(grid_values) - 1,
        upper=max(grid_values) + 1,
        max_step=1,
        grid_values=grid_values,
    )


def _observation(
    context,
    readback: Mapping[str, int],
    *,
    observation_id: str,
    tdecq_db: float | None,
    requested: Mapping[str, int] | None = None,
    candidate_id: str | None = None,
    er_db: float | None = 3.0,
) -> TuningObservation:
    valid = tdecq_db is not None
    return TuningObservation(
        observation_id=observation_id,
        candidate_id=candidate_id or f"candidate-{observation_id}",
        context=context,
        settings_requested=dict(requested or readback),
        settings_readback=dict(readback),
        tdecq_db=tdecq_db,
        tdecq_status="CORR" if valid else "INV",
        tdecq_invalid_reason=None if valid else "invalid TDECQ",
        er_db=er_db,
        er_valid=er_db is not None,
        er_status="CORR" if er_db is not None else "INV",
    )


def _optimizer(space, context, **updates) -> TdecqBayesianOptimizer:
    values = {
        "seed": 1124,
        "min_valid_observations": 3,
    }
    values.update(updates)
    return TdecqBayesianOptimizer(space, context=context, **values)


def _normal_pdf(value: float) -> float:
    return exp(-(value**2) / 2.0) / sqrt(2.0 * pi)


def _normal_cdf(value: float) -> float:
    return (1.0 + __import__("math").erf(value / sqrt(2.0))) / 2.0


def _snapshot_payload(snapshot: BayesianSnapshot) -> dict:
    return json.loads(snapshot.model_dump_json())


def _snapshot_document(payload: dict) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def test_candidate_prediction_is_frozen_finite_and_semantically_consistent():
    prediction = CandidatePrediction(
        key=(0, 1),
        predicted_tdecq_db=2.5,
        predicted_sigma_db=0.1,
        expected_improvement=0.2,
        valid_probability=0.5,
        acquisition=0.1,
    )

    with pytest.raises(ValidationError):
        CandidatePrediction(**prediction.model_dump(), extra_field=True)
    with pytest.raises(ValidationError, match="acquisition"):
        prediction.model_copy(
            update={"acquisition": 0.15}
        ).__class__.model_validate(
            {**prediction.model_dump(), "acquisition": 0.15}
        )
    with pytest.raises(ValidationError):
        CandidatePrediction(
            **{**prediction.model_dump(), "predicted_tdecq_db": float("nan")}
        )
    with pytest.raises(ValidationError):
        CandidatePrediction(
            **{**prediction.model_dump(), "valid_probability": 1.01}
        )
    with pytest.raises(ValidationError):
        prediction.acquisition = 0.2


@pytest.mark.parametrize("seed", [True, False, 1.0, "1", None, _IntSubclass(1)])
def test_seed_must_be_an_exact_integer(seed, context, two_by_two_space):
    with pytest.raises(ValueError, match="seed.*integer"):
        TdecqBayesianOptimizer(two_by_two_space, context, seed=seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_sklearn_seed_boundaries_round_trip_through_snapshot(
    seed,
    context,
    two_by_two_space,
):
    optimizer = TdecqBayesianOptimizer(
        two_by_two_space,
        context,
        seed=seed,
    )
    snapshot = optimizer.snapshot((), frozenset())

    restored = TdecqBayesianOptimizer.from_snapshot(
        two_by_two_space,
        snapshot.model_dump_json(),
    )

    assert optimizer.seed == snapshot.seed == restored.seed == seed


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_constructor_rejects_seed_outside_sklearn_range(
    seed,
    context,
    two_by_two_space,
):
    with pytest.raises(ValueError, match="seed.*0|seed.*range|seed.*32"):
        TdecqBayesianOptimizer(two_by_two_space, context, seed=seed)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_snapshot_restore_rejects_seed_outside_sklearn_range(
    seed,
    context,
    two_by_two_space,
):
    snapshot = TdecqBayesianOptimizer(
        two_by_two_space,
        context,
        seed=0,
    ).snapshot((), frozenset())
    payload = _snapshot_payload(snapshot)
    payload["seed"] = seed

    with pytest.raises(ValueError, match="seed|greater|less|range"):
        TdecqBayesianOptimizer.from_snapshot(
            two_by_two_space,
            _snapshot_document(payload),
        )


@pytest.mark.parametrize(
    "minimum",
    [True, False, 0, -1, 1.0, "3", None, _IntSubclass(3)],
)
def test_explicit_minimum_must_be_an_exact_positive_integer(
    minimum,
    context,
    two_by_two_space,
):
    kwargs = {} if minimum is None else {"min_valid_observations": minimum}
    if minimum is None:
        optimizer = TdecqBayesianOptimizer(
            two_by_two_space,
            context,
            seed=1,
            **kwargs,
        )
        assert optimizer.min_valid_observations == 3
    else:
        with pytest.raises(ValueError, match="min_valid_observations"):
            TdecqBayesianOptimizer(
                two_by_two_space,
                context,
                seed=1,
                **kwargs,
            )


@pytest.mark.parametrize(
    "xi",
    [True, False, -0.1, float("nan"), float("inf"), float("-inf"), "0.1", None],
)
def test_xi_must_be_finite_nonnegative_numeric(xi, context, two_by_two_space):
    with pytest.raises(ValueError, match="xi"):
        TdecqBayesianOptimizer(two_by_two_space, context, seed=1, xi=xi)


def test_xi_accepts_nonnegative_integer_and_float(context, two_by_two_space):
    assert TdecqBayesianOptimizer(two_by_two_space, context, seed=1, xi=0).xi == 0
    assert TdecqBayesianOptimizer(two_by_two_space, context, seed=1, xi=0.5).xi == 0.5


def test_features_are_canonical_normalized_and_handle_constant_axis(context):
    space = IntegerParameterSpace(
        parameters=(
            _parameter("varying", (10, 20, 30)),
            _parameter("constant", (7,)),
        )
    )
    optimizer = TdecqBayesianOptimizer(space, context, seed=7)
    points = list(space.enumerate_points())

    features = optimizer._features(points)

    assert features.shape == (3, 2)
    assert np.all((features >= 0.0) & (features <= 1.0))
    constant_index = space.names.index("constant")
    varying_index = space.names.index("varying")
    np.testing.assert_array_equal(features[:, constant_index], np.zeros(3))
    np.testing.assert_allclose(features[:, varying_index], (0.0, 0.5, 1.0))


def test_fit_uses_actual_readback_and_canonical_valid_training_order(
    three_by_three_space,
    mixed_surface,
):
    context = mixed_surface[0].context
    requested_mutation = tuple(
        item.model_copy(
            update={"settings_requested": {"eq": 2, "ibias_dac": 2}}
        )
        for item in reversed(mixed_surface)
    )
    first = _optimizer(three_by_three_space, context)
    second = _optimizer(three_by_three_space, context)

    first.fit(requested_mutation)
    second.fit(mixed_surface)

    expected = tuple(
        item
        for item in sorted(
            mixed_surface,
            key=lambda item: three_by_three_space.key(item.settings_readback),
        )
        if item.tdecq_status == "CORR" and item.tdecq_db is not None
    )
    assert first.tdecq_training_ids == tuple(item.observation_id for item in expected)
    assert second.tdecq_training_ids == first.tdecq_training_ids
    expected_features = first._features(
        [dict(item.settings_readback) for item in expected]
    )
    np.testing.assert_allclose(first._regressor.X_train_, expected_features)
    np.testing.assert_allclose(second._regressor.X_train_, expected_features)


def test_each_fit_resets_prior_model_state_even_when_new_history_is_invalid(
    three_by_three_space,
    mixed_surface,
    all_invalid_surface,
):
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
    )
    optimizer.fit(mixed_surface)
    assert optimizer._regressor is not None

    optimizer.fit(all_invalid_surface)

    assert optimizer.tdecq_training_ids == ()
    assert optimizer._regressor is None
    assert optimizer._classifier is None
    assert optimizer._validity_mode == "all_invalid"


def test_fit_validation_failure_also_clears_prior_model_state(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    optimizer.fit(mixed_surface)
    foreign = mixed_surface[0].model_copy(
        update={
            "context": mixed_surface[0].context.model_copy(
                update={"device_id": "vcsel-foreign"}
            )
        }
    )

    with pytest.raises(ValueError, match="context"):
        optimizer.fit((foreign,))

    assert optimizer.tdecq_training_ids == ()
    assert optimizer._regressor is None
    assert optimizer._classifier is None


def test_all_valid_history_uses_probability_one(
    three_by_three_space,
    mixed_surface,
):
    history = mixed_surface[:-1]
    optimizer = _optimizer(three_by_three_space, history[0].context)

    optimizer.fit(history)
    predictions = optimizer.score_unmeasured(history, frozenset())

    assert optimizer._validity_mode == "all_valid"
    assert predictions
    assert all(item.valid_probability == 1.0 for item in predictions)


def test_empty_and_all_invalid_validity_modes_return_probability_zero(
    three_by_three_space,
    context,
    all_invalid_surface,
):
    optimizer = _optimizer(three_by_three_space, context)
    features = np.zeros((2, len(three_by_three_space.names)))

    optimizer.fit(())
    np.testing.assert_array_equal(
        optimizer._valid_probability(features),
        np.zeros(2),
    )
    optimizer.fit(all_invalid_surface)
    np.testing.assert_array_equal(
        optimizer._valid_probability(features),
        np.zeros(2),
    )


def test_mixed_validity_uses_classifier_probability_and_locates_class_one(
    three_by_three_space,
    mixed_surface,
):
    class ReverseClassClassifier:
        classes_ = np.asarray([1, 0])

        @staticmethod
        def predict_proba(features):
            return np.tile((0.25, 0.75), (features.shape[0], 1))

    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    optimizer.fit(mixed_surface)
    optimizer._classifier = ReverseClassClassifier()
    features = np.zeros((3, len(three_by_three_space.names)))

    probabilities = optimizer._valid_probability(features)

    assert optimizer._validity_mode == "classifier"
    np.testing.assert_allclose(probabilities, (0.25, 0.25, 0.25))


def test_acquisition_uses_correct_minimization_ei_including_zero_sigma(
    context,
    two_by_two_space,
):
    points = tuple(two_by_two_space.enumerate_points())
    history = (
        _observation(
            context,
            points[0],
            observation_id="obs-a",
            tdecq_db=2.0,
        ),
        _observation(
            context,
            points[1],
            observation_id="obs-b",
            tdecq_db=2.2,
        ),
    )
    optimizer = _optimizer(
        two_by_two_space,
        context,
        min_valid_observations=2,
        xi=0.1,
    )
    optimizer.fit(history)

    class ControlledRegressor:
        @staticmethod
        def predict(features, return_std):
            assert return_std is True
            assert features.shape[0] == 2
            return np.asarray((2.5, 1.5)), np.asarray((1.0, 0.0))

    optimizer._regressor = ControlledRegressor()

    predictions = optimizer.score_unmeasured(history, frozenset())

    first_improvement = 2.0 - 2.5 - 0.1
    first_z = first_improvement / 1.0
    expected_first = (
        first_improvement * _normal_cdf(first_z) + _normal_pdf(first_z)
    )
    assert predictions[0].expected_improvement == pytest.approx(expected_first)
    assert predictions[1].expected_improvement == pytest.approx(0.4)
    assert all(
        item.acquisition
        == pytest.approx(item.expected_improvement * item.valid_probability)
        for item in predictions
    )


def test_mixed_classifier_probability_is_applied_to_acquisition(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    optimizer.fit(mixed_surface)

    predictions = optimizer.score_unmeasured(mixed_surface, frozenset())

    assert predictions
    assert any(0.0 < item.valid_probability < 1.0 for item in predictions)
    for item in predictions:
        assert item.acquisition == pytest.approx(
            item.expected_improvement * item.valid_probability
        )


def test_score_is_canonical_and_excludes_actual_readback_and_pending(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    optimizer.fit(mixed_surface)
    measured = {
        three_by_three_space.key(item.settings_readback) for item in mixed_surface
    }
    first_available = next(
        key
        for key in (
            three_by_three_space.key(point)
            for point in three_by_three_space.enumerate_points()
        )
        if key not in measured
    )

    predictions = optimizer.score_unmeasured(
        mixed_surface,
        frozenset({first_available}),
    )

    keys = tuple(item.key for item in predictions)
    assert keys == tuple(sorted(keys))
    assert not (set(keys) & measured)
    assert first_available not in keys


def test_score_rejects_stale_authoritative_history_but_ignores_er_and_requested(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    optimizer.fit(mixed_surface)
    audit_only = tuple(
        item.model_copy(
            update={
                "settings_requested": {"eq": 99, "ibias_dac": 99},
                "er_db": 1000.0 - index,
            }
        )
        for index, item in enumerate(reversed(mixed_surface))
    )
    authoritative_change = (
        mixed_surface[0].model_copy(update={"tdecq_db": 3.5}),
        *mixed_surface[1:],
    )

    assert optimizer.score_unmeasured(audit_only, frozenset())
    with pytest.raises(ValueError, match="stale|mismatch|fitted history"):
        optimizer.score_unmeasured(authoritative_change, frozenset())


def test_er_requested_and_input_order_mutations_do_not_change_suggestion_or_id(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    mutated = tuple(
        item.model_copy(
            update={
                "settings_requested": {"eq": 99, "ibias_dac": 99},
                "er_db": 1000.0 - index * 100.0,
            }
        )
        for index, item in enumerate(reversed(mixed_surface))
    )

    original = optimizer.suggest(mixed_surface)
    changed = optimizer.suggest(mutated)

    assert isinstance(original, CandidateSuggestion)
    assert changed == original


def test_authoritative_tdecq_and_readback_history_changes_candidate_id(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def deterministic_fit(model, features, targets):
        return model

    def deterministic_predict(model, features, return_std):
        mean = 2.0 + np.arange(features.shape[0], dtype=float) * 0.2
        sigma = np.ones(features.shape[0], dtype=float)
        return mean, sigma

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        deterministic_fit,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.predict",
        deterministic_predict,
    )
    history = mixed_surface[:-1]
    tdecq_changed = (
        history[0].model_copy(update={"tdecq_db": 3.5}),
        *history[1:],
    )
    first, second, *rest = history
    readback_changed = (
        first.model_copy(update={"settings_readback": second.settings_readback}),
        second.model_copy(update={"settings_readback": first.settings_readback}),
        *rest,
    )
    optimizer = _optimizer(three_by_three_space, history[0].context)

    original = optimizer.suggest(history)
    changed_value = optimizer.suggest(tdecq_changed)
    changed_readback = optimizer.suggest(readback_changed)

    assert isinstance(original, CandidateSuggestion)
    assert changed_value.requested == original.requested
    assert changed_readback.requested == original.requested
    assert len(
        {
            original.candidate_id,
            changed_value.candidate_id,
            changed_readback.candidate_id,
        }
    ) == 3


def test_seeded_and_observation_order_repeats_are_exact(
    three_by_three_space,
    mixed_surface,
):
    first = _optimizer(three_by_three_space, mixed_surface[0].context)
    second = _optimizer(three_by_three_space, mixed_surface[0].context)

    first_result = first.suggest(mixed_surface)
    repeat_result = first.suggest(mixed_surface)
    reordered_result = second.suggest(tuple(reversed(mixed_surface)))

    assert isinstance(first_result, CandidateSuggestion)
    assert repeat_result == first_result
    assert reordered_result == first_result


def test_pending_and_measured_points_are_never_suggested(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    first = optimizer.suggest(mixed_surface)
    pending = frozenset({three_by_three_space.key(first.requested)})

    second = optimizer.suggest(mixed_surface, pending)

    measured = {
        three_by_three_space.key(item.settings_readback) for item in mixed_surface
    }
    assert isinstance(second, CandidateSuggestion)
    assert three_by_three_space.key(second.requested) not in measured | pending


@pytest.mark.parametrize("entrypoint", ["suggest", "score_unmeasured"])
def test_invalid_pending_key_is_rejected(
    entrypoint,
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    if entrypoint == "score_unmeasured":
        optimizer.fit(mixed_surface)

    with pytest.raises(ValueError, match="pending key|valid safe key"):
        getattr(optimizer, entrypoint)(mixed_surface, frozenset({(9, 9)}))


def test_only_pending_remaining_points_report_no_safe_candidate(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    measured = {
        three_by_three_space.key(item.settings_readback) for item in mixed_surface
    }
    pending = frozenset(
        three_by_three_space.key(point)
        for point in three_by_three_space.enumerate_points()
        if three_by_three_space.key(point) not in measured
    )

    result = optimizer.suggest(mixed_surface, pending)

    assert isinstance(result, NoSafeCandidate)
    assert "pending" in result.reason.lower()


def test_all_invalid_history_requests_coordinate_fallback(
    three_by_three_space,
    all_invalid_surface,
):
    optimizer = _optimizer(three_by_three_space, all_invalid_surface[0].context)

    result = optimizer.suggest(all_invalid_surface)

    assert isinstance(result, FallbackRequested)
    assert "insufficient numeric TDECQ" in result.reason


def test_insufficient_valid_history_requests_coordinate_fallback(
    context,
    three_by_three_space,
):
    points = tuple(three_by_three_space.enumerate_points())
    history = tuple(
        _observation(
            context,
            points[index],
            observation_id=f"obs-{index}",
            tdecq_db=3.0 - index * 0.1,
        )
        for index in range(2)
    )
    optimizer = _optimizer(three_by_three_space, context)

    result = optimizer.suggest(history)

    assert isinstance(result, FallbackRequested)
    assert "insufficient numeric TDECQ" in result.reason


def test_empty_history_uses_first_compatible_effect_memory_anchor(
    context,
    two_by_two_space,
):
    optimizer = _optimizer(
        two_by_two_space,
        context,
        anchor_points=(
            {"eq": 1, "ibias_dac": 1},
            {"eq": 1, "ibias_dac": 0},
        ),
    )

    result = optimizer.suggest(())

    assert isinstance(result, CandidateSuggestion)
    assert result.requested == {"eq": 1, "ibias_dac": 1}
    assert result.reason == "compatible effect-memory anchor"
    assert result.acquisition is None


def test_anchor_points_are_canonical_defensive_immutable_and_deduplicated(
    context,
    two_by_two_space,
):
    first = {"ibias_dac": 1, "eq": 1}
    second = {"ibias_dac": 0, "eq": 1}
    optimizer = _optimizer(
        two_by_two_space,
        context,
        anchor_points=(first, second, dict(first)),
    )
    first["eq"] = 0

    assert optimizer.anchor_points == (
        {"eq": 1, "ibias_dac": 1},
        {"eq": 1, "ibias_dac": 0},
    )
    with pytest.raises(TypeError):
        optimizer.anchor_points[0]["eq"] = 0


@pytest.mark.parametrize(
    "anchor",
    [
        pytest.param({"eq": True, "ibias_dac": 1}, id="bool"),
        pytest.param({"eq": 0.5, "ibias_dac": 1}, id="fractional"),
        pytest.param({"eq": 1}, id="missing-name"),
        pytest.param({"eq": 1, "ibias_dac": 1, "extra": 0}, id="extra-name"),
        pytest.param({"eq": 9, "ibias_dac": 9}, id="unsafe"),
        pytest.param((1, 1), id="not-a-mapping"),
    ],
)
def test_malformed_or_unsafe_anchor_points_are_rejected(
    anchor,
    context,
    two_by_two_space,
):
    with pytest.raises(ValueError, match="anchor|point|mapping|integer|endpoint"):
        _optimizer(two_by_two_space, context, anchor_points=(anchor,))


def test_anchor_selection_skips_actual_readback_and_pending_in_priority_order(
    context,
    two_by_two_space,
):
    points = tuple(two_by_two_space.enumerate_points())
    anchors = (
        {"eq": 1, "ibias_dac": 1},
        {"eq": 1, "ibias_dac": 0},
        {"eq": 0, "ibias_dac": 1},
    )
    measured_first = _observation(
        context,
        anchors[0],
        requested=points[0],
        observation_id="obs-measured-anchor",
        tdecq_db=2.4,
    )
    optimizer = _optimizer(two_by_two_space, context, anchor_points=anchors)

    result = optimizer.suggest(
        (measured_first,),
        frozenset({two_by_two_space.key(anchors[1])}),
    )

    assert isinstance(result, CandidateSuggestion)
    assert result.requested == anchors[2]
    assert result.reason == "compatible effect-memory anchor"


def test_exhausted_anchors_preserve_normal_insufficient_history_fallback(
    context,
    two_by_two_space,
):
    anchors = (
        {"eq": 1, "ibias_dac": 1},
        {"eq": 1, "ibias_dac": 0},
    )
    measured = _observation(
        context,
        anchors[0],
        observation_id="obs-measured-anchor",
        tdecq_db=2.4,
    )
    optimizer = _optimizer(two_by_two_space, context, anchor_points=anchors)

    result = optimizer.suggest(
        (measured,),
        frozenset({two_by_two_space.key(anchors[1])}),
    )

    assert isinstance(result, FallbackRequested)
    assert "insufficient numeric TDECQ" in result.reason


def test_enough_current_device_data_uses_ei_and_never_anchor_order(
    three_by_three_space,
    mixed_surface,
):
    anchor = {"eq": 2, "ibias_dac": 2}
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
        anchor_points=(anchor,),
    )

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, CandidateSuggestion)
    assert result.reason == "maximum TDECQ EI multiplied by valid probability"
    assert result.acquisition is not None


def test_anchor_points_never_enter_current_device_regressor_training(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    fitted_targets = []

    def capture_fit(model, features, targets):
        fitted_targets.extend(float(value) for value in targets)
        return model

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        capture_fit,
    )
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
        anchor_points=({"eq": 2, "ibias_dac": 2},),
    )

    optimizer.fit(mixed_surface)

    assert optimizer.tdecq_training_ids == (
        "obs-00",
        "obs-01",
        "obs-10",
        "obs-11",
    )
    assert fitted_targets == pytest.approx([3.4, 3.0, 3.1, 2.4])


def test_anchor_candidate_id_is_branch_safe_but_ignores_er_and_requested(
    context,
    two_by_two_space,
):
    history = (
        _observation(
            context,
            {"eq": 0, "ibias_dac": 0},
            observation_id="obs-current",
            tdecq_db=3.0,
        ),
    )
    optimizer = _optimizer(
        two_by_two_space,
        context,
        anchor_points=({"eq": 1, "ibias_dac": 1},),
    )
    audit_only = (
        history[0].model_copy(
            update={
                "settings_requested": {"eq": 99, "ibias_dac": 99},
                "er_db": 1e100,
            }
        ),
    )
    authoritative_change = (
        history[0].model_copy(update={"tdecq_db": 3.1}),
    )

    original = optimizer.suggest(history)
    audit_changed = optimizer.suggest(audit_only)
    branch_changed = optimizer.suggest(authoritative_change)

    assert isinstance(original, CandidateSuggestion)
    assert audit_changed.candidate_id == original.candidate_id
    assert branch_changed.candidate_id != original.candidate_id


def test_anchor_candidate_id_includes_endpoint_priority_and_configuration(
    context,
    two_by_two_space,
):
    measured = _observation(
        context,
        {"eq": 0, "ibias_dac": 0},
        observation_id="obs-current",
        tdecq_db=3.0,
    )
    endpoint_a = _optimizer(
        two_by_two_space,
        context,
        anchor_points=({"eq": 1, "ibias_dac": 1},),
    ).suggest((measured,))
    endpoint_b = _optimizer(
        two_by_two_space,
        context,
        anchor_points=({"eq": 0, "ibias_dac": 1},),
    ).suggest((measured,))
    shifted_priority = _optimizer(
        two_by_two_space,
        context,
        anchor_points=(
            {"eq": 0, "ibias_dac": 0},
            {"eq": 1, "ibias_dac": 1},
        ),
    ).suggest((measured,))
    changed_config = _optimizer(
        two_by_two_space,
        context,
        min_valid_observations=4,
        anchor_points=({"eq": 1, "ibias_dac": 1},),
    ).suggest((measured,))

    assert endpoint_a.requested != endpoint_b.requested
    assert len(
        {
            endpoint_a.candidate_id,
            endpoint_b.candidate_id,
            shifted_priority.candidate_id,
            changed_config.candidate_id,
        }
    ) == 4


@pytest.mark.parametrize("failure", ["foreign", "duplicate", "unsafe"])
def test_history_validation_errors_propagate_from_suggest(
    failure,
    three_by_three_space,
    mixed_surface,
):
    if failure == "foreign":
        bad = mixed_surface[0].model_copy(
            update={
                "context": mixed_surface[0].context.model_copy(
                    update={"device_id": "vcsel-foreign"}
                )
            }
        )
        history = (bad,)
        match = "context"
    elif failure == "duplicate":
        bad = mixed_surface[0].model_copy(update={"observation_id": "obs-duplicate"})
        history = mixed_surface + (bad,)
        match = "duplicate|repeated"
    else:
        bad = mixed_surface[0].model_copy(
            update={"settings_readback": {"eq": 9, "ibias_dac": 9}}
        )
        history = (bad,)
        match = "readback|allowed endpoint"
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)

    with pytest.raises(ValueError, match=match):
        optimizer.suggest(history)


def test_regressor_fit_exception_requests_explicit_model_fit_fallback(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def fail_fit(model, features, targets):
        raise ValueError("synthetic regressor failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        fail_fit,
    )
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, FallbackRequested)
    assert "model fit failed" in result.reason.lower()
    assert "ValueError" in result.reason


def test_classifier_fit_exception_requests_explicit_model_fit_fallback(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def fail_fit(model, features, targets):
        raise np.linalg.LinAlgError("synthetic classifier failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessClassifier.fit",
        fail_fit,
    )
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, FallbackRequested)
    assert "model fit failed" in result.reason.lower()
    assert "LinAlgError" in result.reason


def test_predict_exception_requests_explicit_model_scoring_fallback(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def fail_predict(model, features, return_std):
        raise ValueError("synthetic prediction failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.predict",
        fail_predict,
    )
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, FallbackRequested)
    assert "model scoring failed" in result.reason.lower()
    assert "ValueError" in result.reason


def test_nonfinite_model_output_requests_coordinate_fallback(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def nonfinite_predict(model, features, return_std):
        return np.full(features.shape[0], np.nan), np.ones(features.shape[0])

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.predict",
        nonfinite_predict,
    )
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, FallbackRequested)
    assert "model scoring failed" in result.reason.lower()
    assert "nonfinite" in result.reason.lower()


def test_degenerate_zero_acquisition_requests_coordinate_fallback(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def zero_acquisition_predict(model, features, return_std):
        return np.full(features.shape[0], 100.0), np.zeros(features.shape[0])

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.predict",
        zero_acquisition_predict,
    )
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, FallbackRequested)
    assert "degenerate" in result.reason.lower()


@pytest.mark.parametrize(
    ("fixture_name", "expected_id", "expected_tdecq"),
    [
        ("complete_surface", "obs-11", 2.3),
        ("surface_with_invalid_best", "obs-10", 2.8),
    ],
)
def test_full_valid_or_mixed_pool_skips_fit_and_reports_truthful_global(
    monkeypatch,
    request,
    two_by_two_space,
    fixture_name,
    expected_id,
    expected_tdecq,
):
    observations = request.getfixturevalue(fixture_name)

    def fail_fit(*args, **kwargs):
        raise AssertionError("model fit must not run for a complete pool")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        fail_fit,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessClassifier.fit",
        fail_fit,
    )
    optimizer = _optimizer(two_by_two_space, observations[0].context)

    result = optimizer.suggest(observations)

    assert isinstance(result, PathComplete)
    assert result.path is TuningPath.TDECQ_BAYESIAN
    assert result.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert result.measured_count == result.candidate_count == 4
    assert result.best_observation_id == expected_id
    assert result.best_tdecq_db == pytest.approx(expected_tdecq)


def test_full_all_invalid_pool_skips_fit_and_reports_no_valid(
    monkeypatch,
    context,
    two_by_two_space,
):
    observations = tuple(
        _observation(
            context,
            point,
            observation_id=f"obs-{index}",
            tdecq_db=None,
        )
        for index, point in enumerate(two_by_two_space.enumerate_points())
    )

    def fail_fit(*args, **kwargs):
        raise AssertionError("model fit must not run for a complete pool")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        fail_fit,
    )
    optimizer = _optimizer(two_by_two_space, context)

    result = optimizer.suggest(observations)

    assert isinstance(result, PathComplete)
    assert result.status == "NO_VALID_TDECQ"
    assert result.measured_count == result.candidate_count == 4
    assert result.best_observation_id is None
    assert result.best_tdecq_db is None


def test_complete_pool_tdecq_tie_uses_observation_id_and_ignores_er(
    context,
    two_by_two_space,
):
    points = tuple(two_by_two_space.enumerate_points())
    observations = (
        _observation(
            context,
            points[0],
            observation_id="obs-z",
            tdecq_db=2.0,
            er_db=1e300,
        ),
        _observation(
            context,
            points[1],
            observation_id="obs-a",
            tdecq_db=2.0,
            er_db=-1e300,
        ),
        _observation(
            context,
            points[2],
            observation_id="obs-worse-1",
            tdecq_db=3.0,
        ),
        _observation(
            context,
            points[3],
            observation_id="obs-worse-2",
            tdecq_db=3.1,
        ),
    )
    optimizer = _optimizer(two_by_two_space, context)

    result = optimizer.suggest(tuple(reversed(observations)))

    assert isinstance(result, PathComplete)
    assert result.best_observation_id == "obs-a"
    assert result.best_tdecq_db == 2.0


def test_complete_pool_still_validates_pending_keys(
    context,
    two_by_two_space,
    complete_surface,
):
    optimizer = _optimizer(two_by_two_space, context)

    with pytest.raises(ValueError, match="pending key|valid safe key"):
        optimizer.suggest(complete_surface, frozenset({(9, 9)}))


def test_equal_acquisition_tie_uses_canonical_lowest_key(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    measured = {
        three_by_three_space.key(item.settings_readback) for item in mixed_surface
    }
    available = tuple(
        three_by_three_space.key(point)
        for point in three_by_three_space.enumerate_points()
        if three_by_three_space.key(point) not in measured
    )
    tied = tuple(
        CandidatePrediction(
            key=key,
            predicted_tdecq_db=2.5,
            predicted_sigma_db=0.2,
            expected_improvement=0.5,
            valid_probability=0.4,
            acquisition=0.2,
        )
        for key in (available[-1], available[0])
    )
    monkeypatch.setattr(
        optimizer,
        "score_unmeasured",
        lambda observations, pending_keys: tied,
    )

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, CandidateSuggestion)
    assert three_by_three_space.key(result.requested) == min(available)
    assert result.reason == "maximum TDECQ EI multiplied by valid probability"
    assert result.acquisition == pytest.approx(0.2)


def test_snapshot_json_round_trip_refits_exact_current_device_suggestion(
    three_by_three_space,
    mixed_surface,
):
    pending = frozenset({(2, 2)})
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
        anchor_points=(
            {"eq": 2, "ibias_dac": 1},
            {"eq": 2, "ibias_dac": 0},
        ),
    )
    before = optimizer.suggest(mixed_surface, pending)
    before_scores = optimizer.score_unmeasured(mixed_surface, pending)

    snapshot = optimizer.snapshot(mixed_surface, pending)
    document = snapshot.model_dump_json()
    loaded = BayesianSnapshot.model_validate_json(document)
    restored = TdecqBayesianOptimizer.from_snapshot(
        three_by_three_space,
        document,
    )
    assert restored.tdecq_training_ids == (
        "obs-00",
        "obs-01",
        "obs-10",
        "obs-11",
    )
    restored_pending = frozenset(loaded.pending_keys)
    after = restored.suggest(loaded.observations, restored_pending)
    after_scores = restored.score_unmeasured(
        loaded.observations,
        restored_pending,
    )

    assert loaded == snapshot
    assert snapshot.schema_version == "tdecq-bayesian-v1"
    assert snapshot.context == mixed_surface[0].context
    assert snapshot.space_fingerprint == three_by_three_space.fingerprint
    assert snapshot.parameter_names == three_by_three_space.names
    assert snapshot.seed == optimizer.seed
    assert snapshot.min_valid_observations == optimizer.min_valid_observations
    assert snapshot.xi == optimizer.xi
    assert snapshot.pending_keys == ((2, 2),)
    assert snapshot.anchor_points == optimizer.anchor_points
    assert snapshot.sklearn_version == sklearn.__version__
    assert tuple(item.observation_id for item in snapshot.observations) == (
        "obs-00",
        "obs-01",
        "obs-10",
        "obs-11",
        "obs-12",
    )
    assert isinstance(before, CandidateSuggestion)
    assert isinstance(after, CandidateSuggestion)
    assert after.candidate_id == before.candidate_id
    assert three_by_three_space.key(after.requested) == three_by_three_space.key(
        before.requested
    )
    assert after.acquisition == pytest.approx(before.acquisition)
    assert tuple(item.key for item in after_scores) == tuple(
        item.key for item in before_scores
    )
    assert tuple(item.acquisition for item in after_scores) == pytest.approx(
        tuple(item.acquisition for item in before_scores)
    )
    assert "GaussianProcess" not in document
    assert "_regressor" not in document
    assert "_classifier" not in document


def test_snapshot_needs_no_fitted_model_and_is_transitively_immutable_deterministic(
    three_by_three_space,
    mixed_surface,
):
    first_anchor = {"eq": 2, "ibias_dac": 2}
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
        anchor_points=(first_anchor,),
    )
    pending_a = frozenset({(2, 0), (2, 2)})
    pending_b = frozenset({(2, 2), (2, 0)})

    first = optimizer.snapshot(tuple(reversed(mixed_surface)), pending_a)
    second = optimizer.snapshot(mixed_surface, pending_b)
    first_anchor["eq"] = 0

    assert optimizer._regressor is None
    assert first.model_dump_json() == second.model_dump_json()
    assert first.pending_keys == ((2, 0), (2, 2))
    assert first.anchor_points == ({"eq": 2, "ibias_dac": 2},)
    with pytest.raises(TypeError):
        first.anchor_points[0]["eq"] = 0
    with pytest.raises(TypeError):
        first.observations[0].settings_readback["eq"] = 0
    with pytest.raises(ValidationError):
        first.seed = 1


def test_snapshot_json_canonicalizes_observation_setting_mapping_order(
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
    )
    canonical_observation = mixed_surface[0]
    payload = canonical_observation.model_dump(mode="python")
    payload["settings_requested"] = dict(
        reversed(tuple(canonical_observation.settings_requested.items()))
    )
    payload["settings_readback"] = dict(
        reversed(tuple(canonical_observation.settings_readback.items()))
    )
    reordered_observation = TuningObservation.model_validate(payload)

    canonical = optimizer.snapshot((canonical_observation,), frozenset())
    reordered = optimizer.snapshot((reordered_observation,), frozenset())

    assert canonical.model_dump_json() == reordered.model_dump_json()
    assert tuple(canonical.observations[0].settings_requested) == (
        "eq",
        "ibias_dac",
    )
    assert tuple(canonical.observations[0].settings_readback) == (
        "eq",
        "ibias_dac",
    )
    assert tuple(reordered_observation.settings_requested) == (
        "ibias_dac",
        "eq",
    )
    assert tuple(reordered_observation.settings_readback) == (
        "ibias_dac",
        "eq",
    )


def test_snapshot_rebuilds_mutable_model_copy_settings_defensively(
    context,
    two_by_two_space,
):
    original = _observation(
        context,
        {"eq": 0, "ibias_dac": 0},
        observation_id="obs-mutable-copy",
        tdecq_db=3.4,
    )
    requested = {"ibias_dac": 0, "eq": 0}
    readback = {"ibias_dac": 0, "eq": 0}
    mutable_copy = original.model_copy(
        update={
            "settings_requested": requested,
            "settings_readback": readback,
        }
    )
    optimizer = _optimizer(two_by_two_space, context)

    snapshot = optimizer.snapshot((mutable_copy,), frozenset())
    requested["eq"] = 1
    readback["eq"] = 1

    captured = snapshot.observations[0]
    assert captured is not mutable_copy
    assert captured.settings_requested == {"eq": 0, "ibias_dac": 0}
    assert captured.settings_readback == {"eq": 0, "ibias_dac": 0}
    with pytest.raises(TypeError):
        captured.settings_requested["eq"] = 1
    with pytest.raises(TypeError):
        captured.settings_readback["eq"] = 1


def test_direct_snapshot_construction_revalidates_nested_observation_semantics(
    context,
    two_by_two_space,
):
    observation = _observation(
        context,
        {"eq": 0, "ibias_dac": 0},
        observation_id="obs-direct-corrupt",
        tdecq_db=3.4,
    )
    corrupt = observation.model_copy(update={"tdecq_db": None})
    payload = _optimizer(two_by_two_space, context).snapshot(
        (),
        frozenset(),
    ).model_dump(mode="python")
    payload["observations"] = (corrupt,)

    with pytest.raises(ValidationError, match="CORR TDECQ|numeric value"):
        BayesianSnapshot(**payload)


def test_direct_snapshot_construction_rebuilds_aliased_observation_mappings(
    context,
    two_by_two_space,
):
    observation = _observation(
        context,
        {"eq": 0, "ibias_dac": 0},
        observation_id="obs-direct-aliased",
        tdecq_db=3.4,
    )
    requested = {"ibias_dac": 0, "eq": 0}
    readback = {"ibias_dac": 0, "eq": 0}
    aliased = observation.model_copy(
        update={
            "settings_requested": requested,
            "settings_readback": readback,
        }
    )
    payload = _optimizer(two_by_two_space, context).snapshot(
        (),
        frozenset(),
    ).model_dump(mode="python")
    payload["observations"] = (aliased,)

    snapshot = BayesianSnapshot(**payload)
    requested["eq"] = 1
    readback["eq"] = 1

    captured = snapshot.observations[0]
    assert captured is not aliased
    assert captured.settings_requested == {"eq": 0, "ibias_dac": 0}
    assert captured.settings_readback == {"eq": 0, "ibias_dac": 0}
    assert tuple(captured.settings_requested) == ("eq", "ibias_dac")
    assert tuple(captured.settings_readback) == ("eq", "ibias_dac")
    with pytest.raises(TypeError):
        captured.settings_requested["eq"] = 1
    with pytest.raises(TypeError):
        captured.settings_readback["eq"] = 1


@pytest.mark.parametrize("construction", ["model-copy", "model-construct"])
def test_snapshot_capture_rejects_coercive_corrupt_observation_models(
    construction,
    context,
    two_by_two_space,
):
    original = _observation(
        context,
        {"eq": 0, "ibias_dac": 0},
        observation_id="obs-corrupt-model",
        tdecq_db=3.4,
        er_db=None,
    )
    if construction == "model-copy":
        corrupt = original.model_copy(update={"er_valid": "false"})
    else:
        values = {
            name: getattr(original, name)
            for name in TuningObservation.model_fields
        }
        values["tdecq_db"] = "3.4"
        corrupt = TuningObservation.model_construct(**values)
    optimizer = _optimizer(two_by_two_space, context)

    with pytest.raises(ValueError, match="strict|exact|type|observation"):
        optimizer.snapshot((corrupt,), frozenset())


@pytest.mark.parametrize(
    ("field", "coercive_value"),
    [
        pytest.param("tdecq_db", "3.4", id="numeric-string"),
        pytest.param("er_valid", "false", id="boolean-string"),
    ],
)
def test_snapshot_restore_rejects_coercive_nested_observation_scalars(
    field,
    coercive_value,
    context,
    two_by_two_space,
):
    observation = _observation(
        context,
        {"eq": 0, "ibias_dac": 0},
        observation_id="obs-strict-restore",
        tdecq_db=3.4,
        er_db=None,
    )
    snapshot = _optimizer(two_by_two_space, context).snapshot(
        (observation,),
        frozenset(),
    )
    payload = _snapshot_payload(snapshot)
    payload["observations"][0][field] = coercive_value

    with pytest.raises(ValueError, match="strict|exact|type|observation"):
        TdecqBayesianOptimizer.from_snapshot(
            two_by_two_space,
            _snapshot_document(payload),
        )


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param("schema", id="schema"),
        pytest.param("sklearn-version", id="sklearn-version"),
        pytest.param("fingerprint", id="fingerprint"),
        pytest.param("parameter-names", id="parameter-names"),
        pytest.param("foreign-context", id="foreign-context"),
        pytest.param("duplicate-readback", id="duplicate-readback"),
        pytest.param("unsafe-readback", id="unsafe-readback"),
        pytest.param("unsafe-pending", id="unsafe-pending"),
        pytest.param("pending-measured-overlap", id="pending-measured-overlap"),
        pytest.param("unsafe-anchor", id="unsafe-anchor"),
        pytest.param("duplicate-anchor", id="duplicate-anchor"),
        pytest.param("extra-estimator", id="extra-estimator"),
    ],
)
def test_corrupt_or_tampered_snapshot_fails_without_partial_restore(
    corruption,
    three_by_three_space,
    mixed_surface,
):
    optimizer = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
        anchor_points=({"eq": 2, "ibias_dac": 2},),
    )
    payload = _snapshot_payload(optimizer.snapshot(mixed_surface, frozenset()))
    if corruption == "schema":
        payload["schema_version"] = "tdecq-bayesian-v2"
    elif corruption == "sklearn-version":
        payload["sklearn_version"] = "0.0-tampered"
    elif corruption == "fingerprint":
        payload["space_fingerprint"] = "f" * 64
    elif corruption == "parameter-names":
        payload["parameter_names"] = list(reversed(payload["parameter_names"]))
    elif corruption == "foreign-context":
        payload["observations"][0]["context"]["device_id"] = "vcsel-foreign"
    elif corruption == "duplicate-readback":
        duplicate = dict(payload["observations"][0])
        duplicate["observation_id"] = "obs-duplicate"
        duplicate["candidate_id"] = "candidate-duplicate"
        payload["observations"].append(duplicate)
    elif corruption == "unsafe-readback":
        payload["observations"][0]["settings_readback"]["eq"] = 9
    elif corruption == "unsafe-pending":
        payload["pending_keys"] = [[9, 9]]
    elif corruption == "pending-measured-overlap":
        payload["pending_keys"] = [[0, 0]]
    elif corruption == "unsafe-anchor":
        payload["anchor_points"] = [{"eq": 9, "ibias_dac": 9}]
    elif corruption == "duplicate-anchor":
        payload["anchor_points"].append(dict(payload["anchor_points"][0]))
    else:
        payload["regressor"] = {"unsafe": "serialized-object"}

    with pytest.raises(ValueError):
        TdecqBayesianOptimizer.from_snapshot(
            three_by_three_space,
            _snapshot_document(payload),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("seed", True, id="bool-seed"),
        pytest.param("min_valid_observations", 0, id="minimum-zero"),
        pytest.param("xi", -0.1, id="negative-xi"),
        pytest.param("xi", "0.1", id="coerced-xi"),
        pytest.param("sklearn_version", " ", id="blank-version"),
    ],
)
def test_snapshot_configuration_fields_are_strict_nonblank_and_finite(
    field,
    value,
    three_by_three_space,
    mixed_surface,
):
    snapshot = _optimizer(
        three_by_three_space,
        mixed_surface[0].context,
    ).snapshot(mixed_surface, frozenset())
    payload = _snapshot_payload(snapshot)
    payload[field] = value

    with pytest.raises(ValueError):
        TdecqBayesianOptimizer.from_snapshot(
            three_by_three_space,
            _snapshot_document(payload),
        )


def test_snapshot_and_restore_reject_space_pending_and_context_inconsistency(
    context,
    two_by_two_space,
    three_by_three_space,
    complete_surface,
):
    optimizer = _optimizer(two_by_two_space, context)
    snapshot = optimizer.snapshot(complete_surface[:2], frozenset())

    with pytest.raises(ValueError, match="space|fingerprint|parameter"):
        TdecqBayesianOptimizer.from_snapshot(
            three_by_three_space,
            snapshot.model_dump_json(),
        )
    with pytest.raises(ValueError, match="pending|overlap|measured"):
        optimizer.snapshot(complete_surface[:2], frozenset({(0, 0)}))
    with pytest.raises(ValueError, match="pending|safe|key"):
        optimizer.snapshot(complete_surface[:2], frozenset({(9, 9)}))
    foreign = complete_surface[0].model_copy(
        update={
            "context": context.model_copy(update={"device_id": "vcsel-foreign"})
        }
    )
    with pytest.raises(ValueError, match="context"):
        optimizer.snapshot((foreign,), frozenset())


@pytest.mark.parametrize("include_invalid", [False, True])
def test_restore_refits_all_valid_or_mixed_history_and_keeps_stale_guard(
    include_invalid,
    three_by_three_space,
    mixed_surface,
):
    history = mixed_surface if include_invalid else mixed_surface[:-1]
    optimizer = _optimizer(three_by_three_space, mixed_surface[0].context)
    snapshot = optimizer.snapshot(history, frozenset())

    restored = TdecqBayesianOptimizer.from_snapshot(
        three_by_three_space,
        snapshot.model_dump_json(),
    )

    assert restored.score_unmeasured(snapshot.observations, frozenset())
    assert restored._validity_mode == (
        "classifier" if include_invalid else "all_valid"
    )
    changed = (
        snapshot.observations[0].model_copy(update={"tdecq_db": 9.0}),
        *snapshot.observations[1:],
    )
    with pytest.raises(ValueError, match="stale|mismatch|fitted history"):
        restored.score_unmeasured(changed, frozenset())


def test_snapshot_contains_only_supplied_current_device_observations_not_anchor_values(
    context,
    two_by_two_space,
):
    optimizer = _optimizer(
        two_by_two_space,
        context,
        anchor_points=(
            {"eq": 1, "ibias_dac": 1},
            {"eq": 1, "ibias_dac": 0},
        ),
    )

    snapshot = optimizer.snapshot((), frozenset())
    payload = _snapshot_payload(snapshot)

    assert snapshot.observations == ()
    assert payload["anchor_points"] == [
        {"eq": 1, "ibias_dac": 1},
        {"eq": 1, "ibias_dac": 0},
    ]
    assert all("tdecq_db" not in anchor for anchor in payload["anchor_points"])
