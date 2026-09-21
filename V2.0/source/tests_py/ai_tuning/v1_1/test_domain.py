import pytest

from vcsel_ai_tuning.v1_1.domain import (
    DiscreteRange,
    LifecycleState,
    RuntimeMode,
    TaskType,
)


def test_v11_task_types_and_runtime_modes_are_explicit():
    assert {item.value for item in TaskType} == {"AI_OPTIMIZATION", "SWEEP"}
    assert {item.value for item in RuntimeMode} == {"FAKE", "REPLAY", "LIVE"}


def test_v11_lifecycle_contains_operator_and_recovery_states():
    assert {item.value for item in LifecycleState} == {
        "DRAFT",
        "PREFLIGHTED",
        "B0_FROZEN",
        "AUTHORIZED",
        "RUNNING",
        "PAUSED",
        "ABORTING",
        "ROLLING_BACK",
        "COMPLETED",
        "COMPLETED_NO_IMPROVEMENT",
        "ABORTED",
        "ABORTED_RECOVERED",
        "FAILED",
        "FAILED_INVALID_DCA_TAPS",
        "RECOVERY_REQUIRED",
    }
    assert LifecycleState.RUNNING.is_nonterminal
    assert LifecycleState.RECOVERY_REQUIRED.is_nonterminal
    assert not LifecycleState.COMPLETED.is_nonterminal
    assert not LifecycleState.FAILED.is_nonterminal


def test_discrete_range_never_appends_a_non_aligned_maximum():
    parameter_range = DiscreteRange(minimum=0, maximum=10, step=4)

    assert parameter_range.values() == (0, 4, 8)
    assert 10 not in parameter_range.values()


def test_discrete_range_includes_an_aligned_maximum():
    assert DiscreteRange(minimum=-4, maximum=4, step=2).values() == (
        -4,
        -2,
        0,
        2,
        4,
    )


def test_discrete_range_rejects_invalid_bounds_instead_of_clipping():
    with pytest.raises(ValueError, match="minimum"):
        DiscreteRange(minimum=4, maximum=3, step=1)
    with pytest.raises(ValueError, match="step"):
        DiscreteRange(minimum=0, maximum=4, step=0)


def test_discrete_range_rejects_off_grid_candidate_instead_of_clipping():
    parameter_range = DiscreteRange(minimum=1, maximum=9, step=3)

    assert parameter_range.values() == (1, 4, 7)
    parameter_range.require_value(7)
    with pytest.raises(ValueError, match="aligned"):
        parameter_range.require_value(8)
    with pytest.raises(ValueError, match="range"):
        parameter_range.require_value(10)
