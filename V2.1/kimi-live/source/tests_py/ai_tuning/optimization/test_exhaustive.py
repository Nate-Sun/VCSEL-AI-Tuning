from __future__ import annotations

from collections.abc import Mapping

import pytest

from vcsel_ai_tuning.optimization.exhaustive import ExhaustiveOptimizer
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    OptimizationContext,
    PathComplete,
    TuningObservation,
    TuningPath,
)


def _observation(
    context: OptimizationContext,
    readback: Mapping[str, int],
    *,
    observation_id: str,
    tdecq_db: float | None,
    requested: Mapping[str, int] | None = None,
    candidate_id: str | None = None,
    er_db: float | None = None,
) -> TuningObservation:
    valid_tdecq = tdecq_db is not None
    return TuningObservation(
        observation_id=observation_id,
        candidate_id=candidate_id or f"candidate-{observation_id}",
        context=context,
        settings_requested=dict(requested or readback),
        settings_readback=dict(readback),
        tdecq_db=tdecq_db,
        tdecq_status="CORR" if valid_tdecq else "INV",
        tdecq_invalid_reason=None if valid_tdecq else "instrument returned INV",
        er_db=er_db,
        er_valid=er_db is not None,
        er_status="CORR" if er_db is not None else None,
    )


def _complete_observations(
    context: OptimizationContext,
    points: tuple[dict[str, int], ...],
    values: tuple[float | None, ...],
) -> tuple[TuningObservation, ...]:
    return tuple(
        _observation(
            context,
            point,
            observation_id=f"obs-{index}",
            tdecq_db=value,
        )
        for index, (point, value) in enumerate(zip(points, values, strict=True))
    )


def test_two_by_two_suggests_every_endpoint_once_then_reports_global_best(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    points = tuple(two_by_two_space.enumerate_points())
    observations: tuple[TuningObservation, ...] = ()

    for index, (point, value) in enumerate(
        zip(points, (3.2, None, 2.4, 2.1), strict=True)
    ):
        suggestion = optimizer.suggest(observations)

        assert isinstance(suggestion, CandidateSuggestion)
        assert suggestion.path is TuningPath.EXHAUSTIVE
        assert suggestion.candidate_id == f"exhaustive-{index:08d}"
        assert dict(suggestion.requested) == point
        assert "canonical" in suggestion.reason.lower()
        observations += (
            _observation(
                context,
                point,
                observation_id=f"obs-{index}",
                candidate_id=suggestion.candidate_id,
                tdecq_db=value,
            ),
        )

    complete = optimizer.suggest(observations)

    assert isinstance(complete, PathComplete)
    assert complete.path is TuningPath.EXHAUSTIVE
    assert complete.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert complete.measured_count == 4
    assert complete.candidate_count == 4
    assert complete.best_observation_id == "obs-3"
    assert complete.best_tdecq_db == 2.1


def test_partial_explicit_budget_stop_preserves_best_result(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    first_point = next(two_by_two_space.enumerate_points())
    observations = (
        _observation(
            context,
            first_point,
            observation_id="obs-best",
            tdecq_db=2.5,
        ),
    )

    complete = optimizer.finish(observations, budget_exhausted=True)

    assert complete.status == "EXHAUSTIVE_INCOMPLETE"
    assert complete.measured_count == 1
    assert complete.candidate_count == 4
    assert complete.best_observation_id == "obs-best"
    assert complete.best_tdecq_db == 2.5


def test_complete_all_invalid_reports_no_valid_tdecq_without_best(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    points = tuple(two_by_two_space.enumerate_points())
    observations = _complete_observations(
        context,
        points,
        (None, None, None, None),
    )

    complete = optimizer.finish(observations, budget_exhausted=False)

    assert complete.status == "NO_VALID_TDECQ"
    assert complete.measured_count == complete.candidate_count == 4
    assert complete.best_observation_id is None
    assert complete.best_tdecq_db is None


@pytest.mark.parametrize("operation", ["suggest", "finish"])
def test_foreign_context_is_rejected_by_every_entrypoint(
    context,
    two_by_two_space,
    operation,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    foreign_context = context.model_copy(update={"device_id": "vcsel-foreign"})
    observation = _observation(
        foreign_context,
        next(two_by_two_space.enumerate_points()),
        observation_id="obs-foreign",
        tdecq_db=2.3,
    )

    with pytest.raises(ValueError, match="context"):
        if operation == "suggest":
            optimizer.suggest((observation,))
        else:
            optimizer.finish((observation,), budget_exhausted=True)


def test_actual_readback_not_requested_point_controls_measured_endpoint(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    points = tuple(two_by_two_space.enumerate_points())
    observation = _observation(
        context,
        points[1],
        requested=points[0],
        observation_id="obs-readback-drift",
        tdecq_db=2.7,
    )

    suggestion = optimizer.suggest((observation,))

    assert isinstance(suggestion, CandidateSuggestion)
    assert dict(suggestion.requested) == points[0]
    assert suggestion.candidate_id == "exhaustive-00000000"


def test_duplicate_actual_readback_is_rejected(context, two_by_two_space):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    point = next(two_by_two_space.enumerate_points())
    observations = (
        _observation(
            context,
            point,
            observation_id="obs-1",
            tdecq_db=2.5,
        ),
        _observation(
            context,
            point,
            observation_id="obs-2",
            tdecq_db=2.4,
        ),
    )

    with pytest.raises(ValueError, match="duplicate|repeated"):
        optimizer.suggest(observations)


def test_unsafe_actual_readback_is_rejected(context, two_by_two_space):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    observation = _observation(
        context,
        {"eq": 9, "ibias_dac": 9},
        observation_id="obs-unsafe",
        tdecq_db=2.2,
    )

    with pytest.raises(ValueError, match="readback|allowed endpoint"):
        optimizer.suggest((observation,))


def test_pending_keys_are_skipped_in_canonical_order(context, two_by_two_space):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    points = tuple(two_by_two_space.enumerate_points())
    pending = frozenset(
        {
            two_by_two_space.key(points[0]),
            two_by_two_space.key(points[2]),
        }
    )

    suggestion = optimizer.suggest((), pending)

    assert isinstance(suggestion, CandidateSuggestion)
    assert dict(suggestion.requested) == points[1]
    assert suggestion.candidate_id == "exhaustive-00000001"


def test_all_remaining_pending_returns_no_safe_candidate(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    points = tuple(two_by_two_space.enumerate_points())
    observation = _observation(
        context,
        points[0],
        observation_id="obs-0",
        tdecq_db=2.5,
    )
    pending = frozenset(two_by_two_space.key(point) for point in points[1:])

    result = optimizer.suggest((observation,), pending)

    assert isinstance(result, NoSafeCandidate)
    assert result.path is TuningPath.EXHAUSTIVE
    assert "pending" in result.reason.lower()


@pytest.mark.parametrize("invalid_key", [(9, 9), (0,)])
def test_invalid_pending_key_is_rejected(
    context,
    two_by_two_space,
    invalid_key,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)

    with pytest.raises(ValueError, match="key"):
        optimizer.suggest((), frozenset({invalid_key}))


def test_incomplete_finish_without_explicit_budget_exhaustion_is_rejected(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    observation = _observation(
        context,
        next(two_by_two_space.enumerate_points()),
        observation_id="obs-0",
        tdecq_db=2.5,
    )

    with pytest.raises(ValueError, match="incomplete|budget"):
        optimizer.finish((observation,), budget_exhausted=False)


def test_er_values_and_observation_order_do_not_affect_ranking_or_suggestion(
    context,
    two_by_two_space,
):
    points = tuple(two_by_two_space.enumerate_points())
    history_a = (
        _observation(
            context,
            points[0],
            observation_id="obs-a",
            tdecq_db=2.5,
            er_db=-1e300,
        ),
        _observation(
            context,
            points[2],
            observation_id="obs-b",
            tdecq_db=2.0,
            er_db=1e300,
        ),
    )
    history_b = (
        history_a[1].model_copy(update={"er_db": -1e300}),
        history_a[0].model_copy(update={"er_db": 1e300}),
    )
    optimizer_a = ExhaustiveOptimizer(two_by_two_space, context)
    optimizer_b = ExhaustiveOptimizer(two_by_two_space, context)

    suggestion_a = optimizer_a.suggest(history_a)
    suggestion_b = optimizer_b.suggest(history_b)
    result_a = optimizer_a.finish(history_a, budget_exhausted=True)
    result_b = optimizer_b.finish(history_b, budget_exhausted=True)

    assert suggestion_a == suggestion_b
    assert result_a == result_b
    assert result_a.best_observation_id == "obs-b"
    assert result_a.best_tdecq_db == 2.0


def test_equal_tdecq_tie_is_broken_by_observation_id(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    points = tuple(two_by_two_space.enumerate_points())
    observations = (
        _observation(
            context,
            points[0],
            observation_id="obs-z",
            tdecq_db=2.2,
        ),
        _observation(
            context,
            points[1],
            observation_id="obs-a",
            tdecq_db=2.2,
        ),
    )

    complete = optimizer.finish(observations, budget_exhausted=True)

    assert complete.best_observation_id == "obs-a"
    assert complete.best_tdecq_db == 2.2


def test_suggestion_ids_and_order_are_stable_across_instances_and_history_order(
    context,
    two_by_two_space,
):
    points = tuple(two_by_two_space.enumerate_points())
    measured = (
        _observation(
            context,
            points[3],
            observation_id="obs-3",
            tdecq_db=2.3,
        ),
        _observation(
            context,
            points[1],
            observation_id="obs-1",
            tdecq_db=2.5,
        ),
    )
    first = ExhaustiveOptimizer(two_by_two_space, context)
    second = ExhaustiveOptimizer(two_by_two_space, context)

    first_suggestion = first.suggest(measured)
    second_suggestion = second.suggest(tuple(reversed(measured)))

    assert first_suggestion == second_suggestion
    assert isinstance(first_suggestion, CandidateSuggestion)
    assert dict(first_suggestion.requested) == points[0]
    assert first_suggestion.candidate_id == "exhaustive-00000000"

    measured_with_first = measured + (
        _observation(
            context,
            points[0],
            observation_id="obs-0",
            candidate_id=first_suggestion.candidate_id,
            tdecq_db=2.4,
        ),
    )
    next_suggestion = first.suggest(measured_with_first)

    assert isinstance(next_suggestion, CandidateSuggestion)
    assert dict(next_suggestion.requested) == points[2]
    assert next_suggestion.candidate_id == "exhaustive-00000002"


def test_complete_mixed_result_satisfies_path_complete_invariants(
    context,
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context)
    observations = _complete_observations(
        context,
        tuple(two_by_two_space.enumerate_points()),
        (None, 2.9, None, 2.2),
    )

    result = optimizer.finish(observations, budget_exhausted=False)
    round_tripped = PathComplete.model_validate(result.model_dump())

    assert round_tripped == result
    assert result.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert result.measured_count == result.candidate_count == 4
    assert result.best_observation_id == "obs-3"
    assert result.best_tdecq_db == 2.2
