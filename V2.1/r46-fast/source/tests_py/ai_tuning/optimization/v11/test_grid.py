from __future__ import annotations

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.optimization.v11.grid import (
    DiscreteParameterSpace,
    TuningParameter,
    seeded_permutation_indices,
)


def test_grid_does_not_force_an_unaligned_maximum():
    parameter = TuningParameter(
        name="eq",
        tuning_min=0,
        tuning_max=10,
        tuning_step=3,
    )

    assert parameter.grid_values == (0, 3, 6, 9)
    assert parameter.axis_length == 4


def test_baseline_outside_range_is_not_inserted_into_grid():
    baseline_eq = 1
    parameter = TuningParameter(
        name="eq",
        tuning_min=3,
        tuning_max=9,
        tuning_step=3,
    )

    assert baseline_eq not in parameter.grid_values
    assert parameter.grid_values == (3, 6, 9)


def test_disabled_parameter_is_not_a_tunable_axis():
    space = DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=2,
                tuning_step=1,
            ),
            TuningParameter(
                name="ibias_dac",
                tuning_min=70,
                tuning_max=74,
                tuning_step=2,
                enabled=False,
            ),
        )
    )

    assert space.names == ("eq",)
    assert space.point_count == 3
    assert tuple(space.iter_points()) == ({"eq": 0}, {"eq": 1}, {"eq": 2})


def test_space_uses_lazy_mixed_radix_indexing_for_huge_ranges():
    space = DiscreteParameterSpace(
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

    assert space.point_count == 1_000_000_000_000
    assert space.point_from_index(0) == {"a": 0, "b": 0}
    assert space.point_from_index(space.point_count - 1) == {
        "a": 999_999,
        "b": 999_999,
    }


def test_point_index_round_trip_is_exact_and_canonical():
    space = DiscreteParameterSpace(
        (
            TuningParameter(
                name="ibias_dac",
                tuning_min=70,
                tuning_max=74,
                tuning_step=2,
            ),
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=3,
                tuning_step=3,
            ),
        )
    )

    for original_index in range(space.point_count):
        point = space.point_from_index(original_index)
        assert space.index(point) == original_index


def test_space_rejects_off_grid_and_malformed_points():
    space = DiscreteParameterSpace(
        (
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=10,
                tuning_step=3,
            ),
        )
    )

    with pytest.raises(ValueError, match="grid"):
        space.index({"eq": 10})
    with pytest.raises(ValueError, match="exactly"):
        space.index({"eq": 3, "extra": 1})


def test_parameter_rejects_invalid_bounds_steps_and_unknown_max_step():
    with pytest.raises(ValidationError):
        TuningParameter(
            name="eq",
            tuning_min=10,
            tuning_max=0,
            tuning_step=1,
        )
    with pytest.raises(ValidationError):
        TuningParameter(
            name="eq",
            tuning_min=0,
            tuning_max=10,
            tuning_step=0,
        )
    with pytest.raises(ValidationError):
        TuningParameter(
            name="eq",
            tuning_min=0,
            tuning_max=10,
            tuning_step=3,
            max_step=3,
        )


def test_seeded_permutation_is_complete_unique_and_replayable():
    first = tuple(seeded_permutation_indices(17, seed=123))
    second = tuple(seeded_permutation_indices(17, seed=123))

    assert first == second
    assert len(first) == len(set(first)) == 17
    assert set(first) == set(range(17))


@pytest.mark.parametrize("count", [0, -1])
def test_seeded_permutation_rejects_nonpositive_count(count):
    with pytest.raises(ValueError, match="positive"):
        tuple(seeded_permutation_indices(count, seed=0))
