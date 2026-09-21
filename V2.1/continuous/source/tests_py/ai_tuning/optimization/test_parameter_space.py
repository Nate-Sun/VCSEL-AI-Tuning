from __future__ import annotations

import pytest

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.parameter_space import (
    IntegerParameterSpace,
    NoTransitionPath,
    validated_observation_map,
)
from vcsel_ai_tuning.optimization.protocol import TuningObservation


def _parameter(
    name: str,
    grid_values: tuple[float, ...],
    *,
    max_step: float | None = 1,
    unit: str = "count",
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        unit=unit,
        lower=-100,
        upper=100,
        max_step=max_step,
        grid_values=grid_values,
    )


def _observation(
    context,
    *,
    observation_id: str = "obs-1",
    requested: dict[str, int] | None = None,
    readback: dict[str, int] | None = None,
) -> TuningObservation:
    return TuningObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        context=context,
        settings_requested=requested or {"eq": 1, "ibias_dac": 1},
        settings_readback=readback or {"eq": 0, "ibias_dac": 0},
        tdecq_db=2.4,
        tdecq_status="CORR",
    )


def test_enumeration_is_canonical_unique_and_not_filtered_by_max_step():
    space = IntegerParameterSpace(
        parameters=(
            _parameter(
                "ibias_dac",
                (72, 68, 70, 70),
                max_step=1,
                unit="DAC",
            ),
            _parameter("eq", (23, 21, 22), max_step=1, unit="tap"),
        )
    )

    points = list(space.enumerate_points())

    assert space.names == ("eq", "ibias_dac")
    assert space.candidate_count == 9
    assert points[0] == {"eq": 21, "ibias_dac": 68}
    assert points[-1] == {"eq": 23, "ibias_dac": 72}
    assert len({space.key(point) for point in points}) == 9


def test_cross_constraint_removes_only_unsafe_endpoint_and_rejects_its_key():
    space = IntegerParameterSpace(
        parameters=(
            _parameter("ibias_dac", (70, 72), max_step=2),
            _parameter("eq", (21, 23), max_step=2),
        ),
        allowed=lambda point: not (
            point["ibias_dac"] == 72 and point["eq"] == 23
        ),
    )

    safe_keys = tuple(space.key(point) for point in space.enumerate_points())

    assert space.candidate_count == 3
    assert safe_keys == ((21, 70), (21, 72), (23, 70))
    with pytest.raises(ValueError, match="allowed endpoint"):
        space.key({"eq": 23, "ibias_dac": 72})


def test_shortest_transition_uses_safe_single_coordinate_steps():
    space = IntegerParameterSpace(
        parameters=(
            _parameter("ibias_dac", (68, 70, 72), max_step=2),
            _parameter("eq", (21, 22, 23), max_step=1),
        )
    )
    source = {"ibias_dac": 68, "eq": 21}
    target = {"ibias_dac": 72, "eq": 23}

    path = space.shortest_transition(source, target)

    assert path == space.shortest_transition(source, target)
    assert len(path) == 4
    assert path[-1] == target
    previous = source
    limits = {"eq": 1, "ibias_dac": 2}
    for step in path:
        space.key(step)
        changed = [name for name in space.names if step[name] != previous[name]]
        assert len(changed) == 1
        name = changed[0]
        assert abs(step[name] - previous[name]) <= limits[name]
        previous = step


def test_shortest_transition_reports_disconnected_safe_regions():
    space = IntegerParameterSpace(
        parameters=(_parameter("eq", (20, 21, 22), max_step=1),),
        allowed=lambda point: point["eq"] != 21,
    )
    source = {"eq": 20}
    target = {"eq": 22}

    with pytest.raises(NoTransitionPath) as error:
        space.shortest_transition(source, target)

    message = str(error.value)
    assert repr(space.key(source)) in message
    assert repr(space.key(target)) in message


def test_adjacent_neighbors_do_not_skip_intermediate_grid_values():
    space = IntegerParameterSpace(
        parameters=(_parameter("eq", (0, 1, 2), max_step=2),)
    )

    assert list(space.adjacent_neighbors({"eq": 0})) == [{"eq": 1}]
    assert list(space.local_neighbors({"eq": 0})) == [
        {"eq": 1},
        {"eq": 2},
    ]


def test_adjacent_neighbors_do_not_skip_forbidden_cross_combination():
    space = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 2), max_step=2),
            _parameter("ibias_dac", (0, 1), max_step=1),
        ),
        allowed=lambda point: not (
            point["eq"] == 1 and point["ibias_dac"] == 0
        ),
    )

    assert list(space.adjacent_neighbors({"eq": 0, "ibias_dac": 0})) == [
        {"eq": 0, "ibias_dac": 1}
    ]
    assert {space.key(point) for point in space.local_neighbors(
        {"eq": 0, "ibias_dac": 0}
    )} == {(0, 1), (2, 0)}


def test_local_neighbors_generate_from_axes_without_scanning_all_endpoints(
    monkeypatch,
):
    space = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 2), max_step=2),
            _parameter("ibias_dac", (0, 1), max_step=1),
        ),
        allowed=lambda point: not (
            point["eq"] == 1 and point["ibias_dac"] == 0
        ),
    )

    class EndpointScanGuard:
        def __iter__(self):
            raise AssertionError("local neighbor discovery scanned all endpoints")

    monkeypatch.setattr(space, "_endpoint_keys", EndpointScanGuard())

    assert list(space.local_neighbors({"eq": 0, "ibias_dac": 0})) == [
        {"eq": 0, "ibias_dac": 1},
        {"eq": 2, "ibias_dac": 0},
    ]


def test_space_rejects_when_cross_constraint_allows_no_endpoint():
    with pytest.raises(ValueError, match="no safe endpoints"):
        IntegerParameterSpace(
            parameters=(_parameter("eq", (0, 1)),),
            allowed=lambda point: False,
        )


def test_space_rejects_empty_parameter_tuple():
    with pytest.raises(ValueError, match="at least one parameter"):
        IntegerParameterSpace(parameters=())


def test_space_rejects_duplicate_parameter_names():
    with pytest.raises(ValueError, match="unique"):
        IntegerParameterSpace(
            parameters=(
                _parameter("eq", (0, 1)),
                _parameter("eq", (2, 3)),
            )
        )


def test_space_rejects_missing_grid_values():
    parameter = ParameterSpec(name="eq", unit="tap", lower=0, upper=1)

    with pytest.raises(ValueError, match="grid_values"):
        IntegerParameterSpace(parameters=(parameter,))


def test_space_rejects_noninteger_prepared_grid_values():
    with pytest.raises(ValueError, match="integer"):
        IntegerParameterSpace(
            parameters=(_parameter("eq", (0, 0.5, 1)),)
        )


@pytest.mark.parametrize(
    "point",
    [
        pytest.param({"eq": 0}, id="missing-parameter"),
        pytest.param(
            {"eq": 0, "ibias_dac": 0, "extra": 0},
            id="extra-parameter",
        ),
    ],
)
def test_key_rejects_malformed_point(two_by_two_space, point):
    with pytest.raises(ValueError, match="exactly"):
        two_by_two_space.key(point)


def test_key_rejects_noninteger_point(two_by_two_space):
    with pytest.raises(ValueError, match="integer"):
        two_by_two_space.key({"eq": 0.5, "ibias_dac": 0})


def test_key_rejects_out_of_grid_point(two_by_two_space):
    with pytest.raises(ValueError, match="allowed endpoint"):
        two_by_two_space.key({"eq": 2, "ibias_dac": 0})


@pytest.mark.parametrize(
    "key",
    [
        pytest.param((0,), id="too-short"),
        pytest.param((0, 0, 0), id="too-long"),
        pytest.param((0.5, 0), id="noninteger"),
        pytest.param((2, 0), id="not-an-endpoint"),
    ],
)
def test_point_rejects_invalid_key(two_by_two_space, key):
    with pytest.raises(ValueError):
        two_by_two_space.point(key)


def test_same_endpoint_transition_is_empty(two_by_two_space):
    endpoint = {"eq": 0, "ibias_dac": 1}

    assert two_by_two_space.shortest_transition(endpoint, dict(endpoint)) == ()


def test_points_and_enumeration_are_defensive_copies(two_by_two_space):
    enumerated = list(two_by_two_space.enumerate_points())
    enumerated[0]["eq"] = 99
    reconstructed = two_by_two_space.point((0, 0))
    reconstructed["ibias_dac"] = 99

    assert next(two_by_two_space.enumerate_points()) == {
        "eq": 0,
        "ibias_dac": 0,
    }
    assert two_by_two_space.point((0, 0)) == {"eq": 0, "ibias_dac": 0}


def test_fingerprint_is_stable_under_parameter_and_grid_reordering():
    first = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (2, 0, 1, 1), max_step=1, unit="tap"),
            _parameter(
                "ibias_dac",
                (12, 10, 11, 10),
                max_step=1,
                unit="DAC",
            ),
        )
    )
    second = IntegerParameterSpace(
        parameters=(
            _parameter(
                "ibias_dac",
                (10, 11, 12),
                max_step=1,
                unit="DAC",
            ),
            _parameter("eq", (0, 1, 2), max_step=1, unit="tap"),
        )
    )

    assert first.fingerprint == second.fingerprint


def test_fingerprint_changes_with_endpoint_allowed_region_or_max_step():
    parameters = (
        _parameter("eq", (0, 1, 2), max_step=1, unit="tap"),
        _parameter("ibias_dac", (10, 11, 12), max_step=1, unit="DAC"),
    )
    base = IntegerParameterSpace(parameters=parameters)
    endpoint_changed = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 3), max_step=1, unit="tap"),
            parameters[1],
        )
    )
    allowed_region_changed = IntegerParameterSpace(
        parameters=parameters,
        allowed=lambda point: not (
            point["eq"] == 2 and point["ibias_dac"] == 12
        ),
    )
    max_step_changed = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 2), max_step=2, unit="tap"),
            parameters[1],
        )
    )

    assert len(
        {
            base.fingerprint,
            endpoint_changed.fingerprint,
            allowed_region_changed.fingerprint,
            max_step_changed.fingerprint,
        }
    ) == 4


def test_validated_observation_map_uses_actual_readback(context, two_by_two_space):
    observation = _observation(
        context,
        requested={"eq": 99, "ibias_dac": 99},
        readback={"eq": 0, "ibias_dac": 1},
    )

    result = validated_observation_map(
        two_by_two_space,
        context,
        (observation,),
    )

    assert result == {(0, 1): observation}


def test_validated_observation_map_rejects_mismatched_context(
    context,
    two_by_two_space,
):
    other_context = context.model_copy(update={"device_id": "vcsel-999"})
    observation = _observation(other_context)

    with pytest.raises(ValueError, match="context"):
        validated_observation_map(two_by_two_space, context, (observation,))


def test_validated_observation_map_rejects_duplicate_readback(
    context,
    two_by_two_space,
):
    observations = (
        _observation(context, observation_id="obs-1"),
        _observation(context, observation_id="obs-2"),
    )

    with pytest.raises(ValueError, match="duplicate|repeated"):
        validated_observation_map(two_by_two_space, context, observations)


def test_validated_observation_map_rejects_unsafe_readback(
    context,
    two_by_two_space,
):
    observation = _observation(
        context,
        readback={"eq": 2, "ibias_dac": 0},
    )

    with pytest.raises(ValueError, match="readback|allowed endpoint"):
        validated_observation_map(two_by_two_space, context, (observation,))
