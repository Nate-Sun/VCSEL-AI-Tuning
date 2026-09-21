from __future__ import annotations

import itertools

import pytest

from vcsel_ai_tuning.optimization.v11.fir import (
    FirAuthorizationEnvelope,
    FirTarget,
)
from vcsel_ai_tuning.optimization.v11.grid import TuningParameter
from vcsel_ai_tuning.optimization.v11.sweep import (
    SweepMode,
    SweepPlanner,
    SweepPointStatus,
    SweepTask,
)


def _fir_baseline() -> FirTarget:
    return FirTarget(
        taps=(0, 0, 0, 100, 0, 0, 0),
        upper_nlc=6,
        lower_nlc=7,
    )


def _fir_authorization() -> FirAuthorizationEnvelope:
    return FirAuthorizationEnvelope(
        tap_ranges=(
            (-100, 100),
            (-100, 100),
            (-100, 100),
            (1, 120),
            (-100, 100),
            (-100, 100),
            (-100, 100),
        ),
        max_step=(100, 100, 100, 100, 100, 100, 100),
        max_total_change_l1=700,
    )


def test_sweep_task_and_mode_values_are_explicit():
    assert SweepTask.I2C_SWEEP.value == "I2C_SWEEP"
    assert SweepTask.BERT_FIR_SWEEP.value == "BERT_FIR_SWEEP"
    assert SweepMode.SINGLE_PARAMETER.value == "SINGLE_PARAMETER"
    assert SweepMode.FULL_COMBINATION.value == "FULL_COMBINATION"


def test_i2c_sweep_rejects_names_outside_the_backend_allowlist():
    with pytest.raises(ValueError, match="allowlist"):
        SweepPlanner(
            task=SweepTask.I2C_SWEEP,
            mode=SweepMode.SINGLE_PARAMETER,
            parameters=(
                TuningParameter(
                    name="unsafe_register",
                    tuning_min=0,
                    tuning_max=3,
                    tuning_step=1,
                ),
            ),
            baseline_i2c={"unsafe_register": 0},
            i2c_allowlist=("eq",),
            seed=0,
            max_points=10,
        )


def test_i2c_baseline_cannot_emit_fields_outside_the_backend_allowlist():
    with pytest.raises(ValueError, match="baseline_i2c.*allowlist"):
        SweepPlanner(
            task=SweepTask.I2C_SWEEP,
            mode=SweepMode.SINGLE_PARAMETER,
            parameters=(
                TuningParameter(
                    name="eq",
                    tuning_min=0,
                    tuning_max=3,
                    tuning_step=1,
                ),
            ),
            baseline_i2c={"eq": 0, "unsafe_register": 99},
            i2c_allowlist=("eq",),
            seed=0,
            max_points=10,
        )


def test_sweep_rejects_mixed_i2c_and_bert_inputs():
    with pytest.raises(ValueError, match="must not include FIR"):
        SweepPlanner(
            task=SweepTask.I2C_SWEEP,
            mode=SweepMode.SINGLE_PARAMETER,
            parameters=(
                TuningParameter(
                    name="eq",
                    tuning_min=0,
                    tuning_max=3,
                    tuning_step=1,
                ),
            ),
            baseline_i2c={"eq": 0},
            baseline_fir=_fir_baseline(),
            i2c_allowlist=("eq",),
            seed=0,
            max_points=10,
        )


def test_single_i2c_sweep_changes_one_parameter_and_holds_others_at_b0():
    planner = SweepPlanner(
        task=SweepTask.I2C_SWEEP,
        mode=SweepMode.SINGLE_PARAMETER,
        parameters=(
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=10,
                tuning_step=3,
            ),
            TuningParameter(
                name="ibias_dac",
                tuning_min=70,
                tuning_max=74,
                tuning_step=2,
            ),
        ),
        baseline_i2c={"eq": 1, "ibias_dac": 72},
        i2c_allowlist=("eq", "ibias_dac"),
        seed=99,
        max_points=20,
    )

    points = tuple(planner.points())
    eq_points = tuple(row for row in points if row.selected_parameter == "eq")
    ibias_points = tuple(
        row for row in points if row.selected_parameter == "ibias_dac"
    )

    assert planner.raw_point_count == 7
    assert tuple(row.requested_i2c["eq"] for row in eq_points) == (0, 3, 6, 9)
    assert all(row.requested_i2c["ibias_dac"] == 72 for row in eq_points)
    assert all(row.requested_i2c["eq"] == 1 for row in ibias_points)
    assert tuple(row.requested_i2c["ibias_dac"] for row in ibias_points) == (
        70,
        72,
        74,
    )
    assert 10 not in tuple(row.requested_i2c["eq"] for row in eq_points)


def _full_i2c(seed: int) -> SweepPlanner:
    return SweepPlanner(
        task=SweepTask.I2C_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        parameters=(
            TuningParameter(
                name="eq",
                tuning_min=0,
                tuning_max=2,
                tuning_step=1,
            ),
            TuningParameter(
                name="ibias_dac",
                tuning_min=70,
                tuning_max=76,
                tuning_step=2,
            ),
        ),
        baseline_i2c={"eq": 9, "ibias_dac": 72},
        i2c_allowlist=("eq", "ibias_dac"),
        seed=seed,
        max_points=20,
    )


def test_full_combination_has_replayable_seeded_order_and_both_indices():
    first = tuple(_full_i2c(seed=17).points())
    replay = tuple(_full_i2c(seed=17).points())
    other_seed = tuple(_full_i2c(seed=18).points())

    assert first == replay
    assert len(first) == 12
    assert tuple(row.execution_index for row in first) == tuple(range(12))
    assert set(row.original_index for row in first) == set(range(12))
    assert tuple(row.original_index for row in first) != tuple(
        row.original_index for row in other_seed
    )
    assert len({tuple(row.requested_i2c.items()) for row in first}) == 12


def test_full_combination_iteration_is_lazy_even_for_a_huge_space():
    planner = SweepPlanner(
        task=SweepTask.I2C_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        parameters=(
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
        ),
        baseline_i2c={"a": 0, "b": 0},
        i2c_allowlist=("a", "b"),
        seed=7,
        max_points=1_000_000_000_000,
    )

    first_two = tuple(itertools.islice(planner.points(), 2))

    assert planner.raw_point_count == 1_000_000_000_000
    assert len(first_two) == 2


def test_max_points_rejects_the_raw_count_upfront_without_sampling():
    with pytest.raises(ValueError, match="max_points"):
        _full_i2c(seed=17).__class__(
            task=SweepTask.I2C_SWEEP,
            mode=SweepMode.FULL_COMBINATION,
            parameters=(
                TuningParameter(
                    name="eq",
                    tuning_min=0,
                    tuning_max=99,
                    tuning_step=1,
                ),
                TuningParameter(
                    name="ibias_dac",
                    tuning_min=0,
                    tuning_max=99,
                    tuning_step=1,
                ),
            ),
            baseline_i2c={"eq": 0, "ibias_dac": 0},
            i2c_allowlist=("eq", "ibias_dac"),
            seed=0,
            max_points=9_999,
        )


def test_fir_sweep_only_accepts_seven_named_tap_axes():
    with pytest.raises(ValueError, match="FIR tap"):
        SweepPlanner(
            task=SweepTask.BERT_FIR_SWEEP,
            mode=SweepMode.SINGLE_PARAMETER,
            parameters=(
                TuningParameter(
                    name="upper_nlc",
                    tuning_min=0,
                    tuning_max=3,
                    tuning_step=1,
                ),
            ),
            baseline_fir=_fir_baseline(),
            fir_authorization=_fir_authorization(),
            seed=0,
            max_points=10,
        )


def test_fir_main_constraint_is_skipped_and_raw_taps_are_not_modified():
    planner = SweepPlanner(
        task=SweepTask.BERT_FIR_SWEEP,
        mode=SweepMode.SINGLE_PARAMETER,
        parameters=(
            TuningParameter(
                name="main",
                tuning_min=0,
                tuning_max=2,
                tuning_step=1,
            ),
        ),
        baseline_fir=_fir_baseline(),
        fir_authorization=_fir_authorization(),
        seed=0,
        max_points=10,
    )

    points = tuple(planner.points())

    assert points[0].status is SweepPointStatus.SKIPPED_CONSTRAINT
    assert points[0].requested_fir.taps == (0, 0, 0, 0, 0, 0, 0)
    assert "Main" in points[0].constraint_reason
    assert all(row.requested_fir.upper_nlc == 6 for row in points)
    assert all(row.requested_fir.lower_nlc == 7 for row in points)
    assert points[1].status is SweepPointStatus.READY


def test_fir_l1_violation_is_skipped_without_scaling_clipping_or_substitution():
    planner = SweepPlanner(
        task=SweepTask.BERT_FIR_SWEEP,
        mode=SweepMode.SINGLE_PARAMETER,
        parameters=(
            TuningParameter(
                name="pre3",
                tuning_min=0,
                tuning_max=100,
                tuning_step=100,
            ),
        ),
        baseline_fir=_fir_baseline(),
        fir_authorization=_fir_authorization(),
        seed=0,
        max_points=10,
    )

    points = tuple(planner.points())
    invalid = points[1]

    assert invalid.status is SweepPointStatus.SKIPPED_CONSTRAINT
    assert invalid.requested_fir.taps == (100, 0, 0, 100, 0, 0, 0)
    assert sum(abs(value) for value in invalid.requested_fir.taps) == 200
    assert "L1" in invalid.constraint_reason


def test_fir_authorization_violation_is_skipped_with_exact_requested_taps():
    restrictive = FirAuthorizationEnvelope(
        tap_ranges=_fir_authorization().tap_ranges,
        max_step=(5, 100, 100, 100, 100, 100, 100),
        max_total_change_l1=700,
    )
    planner = SweepPlanner(
        task=SweepTask.BERT_FIR_SWEEP,
        mode=SweepMode.SINGLE_PARAMETER,
        parameters=(
            TuningParameter(
                name="pre3",
                tuning_min=0,
                tuning_max=20,
                tuning_step=20,
            ),
        ),
        baseline_fir=_fir_baseline(),
        fir_authorization=restrictive,
        seed=0,
        max_points=10,
    )

    invalid = tuple(planner.points())[1]

    assert invalid.status is SweepPointStatus.SKIPPED_CONSTRAINT
    assert invalid.requested_fir.taps == (20, 0, 0, 100, 0, 0, 0)
    assert "max_step" in invalid.constraint_reason
