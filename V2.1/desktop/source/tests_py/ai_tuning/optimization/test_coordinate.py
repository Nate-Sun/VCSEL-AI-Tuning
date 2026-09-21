from __future__ import annotations

from collections.abc import Callable, Mapping
import re

import pytest

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.coordinate import CoordinateDescent
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    PathComplete,
    TuningObservation,
    TuningPath,
)


class _IntSubclass(int):
    pass


def _parameter(
    name: str,
    grid_values: tuple[int, ...],
    *,
    max_step: int | None = 1,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        unit="count",
        lower=min(grid_values),
        upper=max(grid_values),
        max_step=max_step,
        grid_values=grid_values,
    )


def _line_space(
    *,
    allowed: Callable[[Mapping[str, int]], bool] | None = None,
) -> IntegerParameterSpace:
    return IntegerParameterSpace(
        parameters=(_parameter("eq", (0, 1, 2)),),
        allowed=allowed,
    )


def _observation(
    context,
    readback: Mapping[str, int],
    *,
    observation_id: str,
    tdecq_db: float | None,
    candidate_id: str | None = None,
    requested: Mapping[str, int] | None = None,
    er_db: float | None = 3.0,
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
        tdecq_invalid_reason=None if valid_tdecq else "invalid TDECQ",
        er_db=er_db,
        er_valid=er_db is not None,
        er_status="CORR" if er_db is not None else "INV",
    )


def test_empty_history_uses_first_canonical_endpoint(context, two_by_two_space):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=17)
    first = next(two_by_two_space.enumerate_points())

    suggestion = optimizer.suggest(())

    assert isinstance(suggestion, CandidateSuggestion)
    assert suggestion.path is TuningPath.TDECQ_BAYESIAN
    assert dict(suggestion.requested) == first
    assert suggestion.reason == "deterministic canonical bootstrap"


def test_empty_history_honors_pending_and_uses_next_canonical_endpoint(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=17)
    points = tuple(two_by_two_space.enumerate_points())
    pending = frozenset({two_by_two_space.key(points[0])})

    suggestion = optimizer.suggest((), pending)

    assert isinstance(suggestion, CandidateSuggestion)
    assert dict(suggestion.requested) == points[1]
    assert suggestion.reason == "deterministic canonical bootstrap"


def test_observation_is_followed_by_one_safe_coordinate_step_within_max_step(
    context,
    three_by_three_space,
):
    optimizer = CoordinateDescent(three_by_three_space, context, seed=4)
    baseline = {"eq": 1, "ibias_dac": 1}
    observation = _observation(
        context,
        baseline,
        observation_id="obs-baseline",
        tdecq_db=2.5,
    )

    suggestion = optimizer.suggest((observation,))

    assert isinstance(suggestion, CandidateSuggestion)
    requested = dict(suggestion.requested)
    three_by_three_space.key(requested)
    changed = [
        name for name in three_by_three_space.names if requested[name] != baseline[name]
    ]
    assert len(changed) == 1
    assert abs(requested[changed[0]] - baseline[changed[0]]) <= 1


def test_best_minimizes_only_valid_tdecq(context, two_by_two_space):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
    points = tuple(two_by_two_space.enumerate_points())
    observations = (
        _observation(
            context,
            points[0],
            observation_id="obs-invalid",
            tdecq_db=None,
        ),
        _observation(
            context,
            points[1],
            observation_id="obs-high",
            tdecq_db=3.0,
        ),
        _observation(
            context,
            points[2],
            observation_id="obs-low",
            tdecq_db=2.0,
        ),
    )

    best = optimizer.best(observations)

    assert best is observations[2]


def test_equal_tdecq_tie_uses_observation_id_then_canonical_key(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
    points = tuple(two_by_two_space.enumerate_points())
    id_tie = (
        _observation(
            context,
            points[0],
            observation_id="obs-z",
            tdecq_db=2.0,
        ),
        _observation(
            context,
            points[2],
            observation_id="obs-a",
            tdecq_db=2.0,
        ),
    )
    key_tie = (
        _observation(
            context,
            points[3],
            observation_id="obs-same",
            tdecq_db=2.0,
        ),
        _observation(
            context,
            points[1],
            observation_id="obs-same",
            tdecq_db=2.0,
        ),
    )

    assert optimizer.best(tuple(reversed(id_tie))) is id_tie[1]
    assert optimizer.best(tuple(reversed(key_tie))) is key_tie[1]


def test_er_and_history_order_do_not_change_best_or_best_based_suggestion(
    context,
    three_by_three_space,
):
    points = tuple(three_by_three_space.enumerate_points())
    history = (
        _observation(
            context,
            points[0],
            observation_id="obs-worse",
            tdecq_db=3.0,
            er_db=-1e300,
        ),
        _observation(
            context,
            points[4],
            observation_id="obs-best",
            tdecq_db=2.0,
            er_db=1e300,
        ),
    )
    mutated = (
        history[1].model_copy(update={"er_db": -1e300}),
        history[0].model_copy(update={"er_db": 1e300}),
    )
    optimizer = CoordinateDescent(three_by_three_space, context, seed=11)

    assert optimizer.best(history).observation_id == "obs-best"
    assert optimizer.best(mutated).observation_id == "obs-best"
    assert optimizer.suggest(history) == optimizer.suggest(mutated)


def test_suggestion_and_id_repeat_across_calls_instances_and_history_order(
    context,
    three_by_three_space,
):
    points = tuple(three_by_three_space.enumerate_points())
    history = (
        _observation(
            context,
            points[0],
            observation_id="obs-worse",
            tdecq_db=3.0,
        ),
        _observation(
            context,
            points[4],
            observation_id="obs-best",
            tdecq_db=2.0,
        ),
    )
    first = CoordinateDescent(three_by_three_space, context, seed=29)
    second = CoordinateDescent(three_by_three_space, context, seed=29)

    suggestion = first.suggest(history)

    assert suggestion == first.suggest(history)
    assert suggestion == second.suggest(tuple(reversed(history)))
    assert isinstance(suggestion, CandidateSuggestion)
    assert suggestion.candidate_id == second.suggest(history).candidate_id


def test_candidate_id_contains_readable_endpoint_and_stable_history_digest(
    context,
):
    space = _line_space()
    optimizer = CoordinateDescent(space, context, seed=7)
    observation = _observation(
        context,
        {"eq": 0},
        observation_id="obs-a",
        tdecq_db=2.0,
    )

    suggestion = optimizer.suggest((observation,))

    assert isinstance(suggestion, CandidateSuggestion)
    assert dict(suggestion.requested) == {"eq": 1}
    assert re.fullmatch(
        r"coordinate-00000001-endpoint-00000001-history-[0-9a-f]{16}",
        suggestion.candidate_id,
    )


def test_candidate_id_disambiguates_authoritative_same_length_histories(
    context,
):
    space = _line_space()
    optimizer = CoordinateDescent(space, context, seed=7)
    histories = (
        (
            _observation(
                context,
                {"eq": 0},
                observation_id="obs-a",
                tdecq_db=2.0,
            ),
        ),
        (
            _observation(
                context,
                {"eq": 2},
                observation_id="obs-a",
                tdecq_db=2.0,
            ),
        ),
        (
            _observation(
                context,
                {"eq": 0},
                observation_id="obs-b",
                candidate_id="candidate-obs-a",
                tdecq_db=2.0,
            ),
        ),
        (
            _observation(
                context,
                {"eq": 0},
                observation_id="obs-a",
                candidate_id="candidate-other",
                tdecq_db=2.0,
            ),
        ),
        (
            _observation(
                context,
                {"eq": 0},
                observation_id="obs-a",
                tdecq_db=2.1,
            ),
        ),
        (
            _observation(
                context,
                {"eq": 0},
                observation_id="obs-a",
                tdecq_db=None,
            ),
        ),
    )

    suggestions = tuple(optimizer.suggest(history) for history in histories)

    assert all(
        isinstance(suggestion, CandidateSuggestion)
        for suggestion in suggestions
    )
    assert all(
        dict(suggestion.requested) == {"eq": 1}
        for suggestion in suggestions
    )
    assert len({suggestion.candidate_id for suggestion in suggestions}) == len(
        suggestions
    )


def test_requested_settings_do_not_participate_in_candidate_identity(context):
    space = _line_space()
    optimizer = CoordinateDescent(space, context, seed=7)
    authoritative = _observation(
        context,
        {"eq": 0},
        requested={"eq": 0},
        observation_id="obs-a",
        tdecq_db=2.0,
    )
    requested_only_mutation = _observation(
        context,
        {"eq": 0},
        requested={"eq": 2},
        observation_id="obs-a",
        tdecq_db=2.0,
    )

    assert optimizer.suggest((authoritative,)) == optimizer.suggest(
        (requested_only_mutation,)
    )


def test_same_history_length_different_endpoints_have_different_candidate_ids(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=3)
    points = tuple(two_by_two_space.enumerate_points())

    first = optimizer.suggest(())
    second = optimizer.suggest(
        (),
        frozenset({two_by_two_space.key(points[0])}),
    )

    assert isinstance(first, CandidateSuggestion)
    assert isinstance(second, CandidateSuggestion)
    assert dict(first.requested) != dict(second.requested)
    assert first.candidate_id != second.candidate_id


def test_local_exhaustion_restarts_canonically_in_disconnected_safe_region(
    context,
):
    space = _line_space(allowed=lambda point: point["eq"] != 1)
    optimizer = CoordinateDescent(space, context, seed=7)
    observation = _observation(
        context,
        {"eq": 0},
        observation_id="obs-left",
        tdecq_db=2.0,
    )

    result = optimizer.suggest((observation,))

    assert isinstance(result, CandidateSuggestion)
    assert dict(result.requested) == {"eq": 2}
    assert "global restart" in result.reason.lower()


def test_all_unmeasured_endpoints_pending_returns_no_safe_candidate(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
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
    assert result.path is TuningPath.TDECQ_BAYESIAN
    assert "pending" in result.reason.lower()
    assert "3" in result.reason


@pytest.mark.parametrize("invalid_key", [(9, 9), (0,), (False, 0)])
def test_invalid_pending_key_is_rejected(
    context,
    two_by_two_space,
    invalid_key,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)

    with pytest.raises(ValueError, match="pending key|valid safe key"):
        optimizer.suggest((), frozenset({invalid_key}))


def test_invalid_pending_key_is_rejected_even_after_full_measurement(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
    observations = tuple(
        _observation(
            context,
            point,
            observation_id=f"obs-{index}",
            tdecq_db=float(index + 1),
        )
        for index, point in enumerate(two_by_two_space.enumerate_points())
    )

    with pytest.raises(ValueError, match="pending key|valid safe key"):
        optimizer.suggest(observations, frozenset({(9, 9)}))


def test_full_mixed_pool_reports_truthful_global_tdecq_best_and_ignores_er(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=5)
    tdecq_values = (None, 2.9, None, 2.2)
    observations = tuple(
        _observation(
            context,
            point,
            observation_id=f"obs-{index}",
            tdecq_db=tdecq_values[index],
            er_db=-1e300 if index % 2 else 1e300,
        )
        for index, point in enumerate(two_by_two_space.enumerate_points())
    )
    mutated = tuple(
        observation.model_copy(update={"er_db": -observation.er_db})
        for observation in reversed(observations)
    )

    result = optimizer.suggest(observations)

    assert isinstance(result, PathComplete)
    assert result.path is TuningPath.TDECQ_BAYESIAN
    assert result.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert result.measured_count == result.candidate_count == 4
    assert result.best_observation_id == "obs-3"
    assert result.best_tdecq_db == 2.2
    assert optimizer.suggest(mutated) == result


def test_full_all_invalid_pool_reports_no_valid_tdecq(
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=5)
    observations = tuple(
        _observation(
            context,
            point,
            observation_id=f"obs-{index}",
            tdecq_db=None,
        )
        for index, point in enumerate(two_by_two_space.enumerate_points())
    )

    result = optimizer.suggest(observations)

    assert isinstance(result, PathComplete)
    assert result.status == "NO_VALID_TDECQ"
    assert result.measured_count == result.candidate_count == 4
    assert result.best_observation_id is None
    assert result.best_tdecq_db is None


@pytest.mark.parametrize("entrypoint", ["best", "suggest"])
def test_foreign_context_is_rejected(entrypoint, context, two_by_two_space):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
    foreign = context.model_copy(update={"device_id": "vcsel-foreign"})
    observation = _observation(
        foreign,
        next(two_by_two_space.enumerate_points()),
        observation_id="obs-foreign",
        tdecq_db=2.5,
    )

    with pytest.raises(ValueError, match="context"):
        getattr(optimizer, entrypoint)((observation,))


@pytest.mark.parametrize("entrypoint", ["best", "suggest"])
def test_duplicate_actual_readback_is_rejected(
    entrypoint,
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
    point = next(two_by_two_space.enumerate_points())
    observations = (
        _observation(
            context,
            point,
            observation_id="obs-a",
            tdecq_db=2.5,
        ),
        _observation(
            context,
            point,
            observation_id="obs-b",
            tdecq_db=2.4,
        ),
    )

    with pytest.raises(ValueError, match="duplicate|repeated"):
        getattr(optimizer, entrypoint)(observations)


@pytest.mark.parametrize("entrypoint", ["best", "suggest"])
def test_unsafe_actual_readback_is_rejected(
    entrypoint,
    context,
    two_by_two_space,
):
    optimizer = CoordinateDescent(two_by_two_space, context, seed=0)
    observation = _observation(
        context,
        {"eq": 9, "ibias_dac": 9},
        observation_id="obs-unsafe",
        tdecq_db=2.5,
    )

    with pytest.raises(ValueError, match="readback|allowed endpoint"):
        getattr(optimizer, entrypoint)((observation,))


def test_actual_readback_controls_measured_identity_and_local_center(context):
    space = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 2)),
            _parameter("ibias_dac", (0, 1, 2)),
        )
    )
    optimizer = CoordinateDescent(space, context, seed=13)
    requested = {"eq": 2, "ibias_dac": 2}
    readback = {"eq": 0, "ibias_dac": 0}
    observation = _observation(
        context,
        readback,
        requested=requested,
        observation_id="obs-drift",
        tdecq_db=2.0,
    )

    local = optimizer.suggest((observation,))

    assert isinstance(local, CandidateSuggestion)
    assert space.key(local.requested) in {
        space.key(point) for point in space.local_neighbors(readback)
    }
    assert space.key(local.requested) not in {
        space.key(point) for point in space.local_neighbors(requested)
    }

    requested_key = space.key(requested)
    pending = frozenset(
        space.key(point)
        for point in space.enumerate_points()
        if space.key(point) not in {space.key(readback), requested_key}
    )
    restart = optimizer.suggest((observation,), pending)

    assert isinstance(restart, CandidateSuggestion)
    assert dict(restart.requested) == requested
    assert "global restart" in restart.reason.lower()


def test_all_invalid_nonempty_history_still_has_deterministic_fallback(
    context,
    three_by_three_space,
):
    points = tuple(three_by_three_space.enumerate_points())
    history = (
        _observation(
            context,
            points[0],
            observation_id="obs-first",
            tdecq_db=None,
        ),
        _observation(
            context,
            points[4],
            observation_id="obs-last",
            tdecq_db=None,
        ),
    )
    optimizer = CoordinateDescent(three_by_three_space, context, seed=41)

    first = optimizer.suggest(history)
    second = optimizer.suggest(history)

    assert optimizer.best(history) is None
    assert isinstance(first, CandidateSuggestion)
    assert first == second
    assert three_by_three_space.key(first.requested) in {
        three_by_three_space.key(point)
        for point in three_by_three_space.local_neighbors(points[4])
    }


@pytest.mark.parametrize(
    "seed",
    [True, False, 1.0, "1", None, _IntSubclass(1)],
)
def test_seed_must_be_an_actual_integer(seed, context, two_by_two_space):
    with pytest.raises(ValueError, match="seed.*integer"):
        CoordinateDescent(two_by_two_space, context, seed=seed)
