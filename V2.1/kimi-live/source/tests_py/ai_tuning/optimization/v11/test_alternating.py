from __future__ import annotations

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.optimization.v11.alternating import (
    AlternatingController,
    AlternationPolicy,
    AlternationStage,
    CompleteTuningCombination,
)
from vcsel_ai_tuning.optimization.v11.fir import FirTarget
from vcsel_ai_tuning.optimization.v11.models import MetricResult


def _enter_first_fir(controller: AlternatingController) -> None:
    assert controller.start().next_stage is AlternationStage.I2C_COARSE
    assert (
        controller.after_coarse(has_valid_tdecq=True).next_stage
        is AlternationStage.I2C_FINE
    )
    assert (
        controller.after_fine(has_valid_tdecq=True).next_stage
        is AlternationStage.FIR_FIRST
    )


@pytest.mark.parametrize(
    ("mode", "rounds"),
    [("FAST", 1), ("STANDARD", 3), ("DEEP", 5)],
)
def test_profile_round_limits(mode, rounds):
    assert AlternationPolicy.for_mode(mode).max_rounds == rounds


def test_advanced_requires_an_explicit_positive_round_count():
    with pytest.raises(ValueError, match="advanced_rounds"):
        AlternationPolicy.for_mode("ADVANCED")
    with pytest.raises(ValueError, match="advanced_rounds"):
        AlternationPolicy.for_mode("ADVANCED", advanced_rounds=0)

    assert (
        AlternationPolicy.for_mode("ADVANCED", advanced_rounds=7).max_rounds
        == 7
    )


def test_fixed_modes_reject_an_advanced_round_override():
    with pytest.raises(ValueError, match="only valid"):
        AlternationPolicy.for_mode("FAST", advanced_rounds=2)


def test_nominal_start_sequence_reaches_first_fir():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))

    _enter_first_fir(controller)


def test_after_fir_cannot_skip_start_coarse_or_fine():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))

    with pytest.raises(ValueError, match="transition"):
        controller.after_fir(round_index=0, improved=True)

    controller.start()
    with pytest.raises(ValueError, match="transition"):
        controller.after_fine(has_valid_tdecq=True)

    controller.after_coarse(has_valid_tdecq=True)
    with pytest.raises(ValueError, match="transition"):
        controller.after_fir(round_index=0, improved=True)


def test_after_retune_cannot_run_before_after_fir():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    _enter_first_fir(controller)

    with pytest.raises(ValueError, match="transition"):
        controller.after_retune(round_index=0, improved=True)


def test_round_index_cannot_jump_ahead_of_the_current_round():
    controller = AlternatingController(AlternationPolicy.for_mode("DEEP"))
    _enter_first_fir(controller)

    with pytest.raises(ValueError, match="round_index"):
        controller.after_fir(round_index=2, improved=True)


def test_no_fir_improvement_stops_before_retune():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    _enter_first_fir(controller)

    decision = controller.after_fir(round_index=0, improved=False)

    assert decision.next_stage is AlternationStage.COMPLETE
    assert "FIR" in decision.reason


def test_no_retune_improvement_stops_before_next_fir():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    _enter_first_fir(controller)
    controller.after_fir(round_index=0, improved=True)

    decision = controller.after_retune(round_index=0, improved=False)

    assert decision.next_stage is AlternationStage.COMPLETE
    assert "retune" in decision.reason


@pytest.mark.parametrize(
    ("mode", "rounds", "advanced_rounds"),
    [
        ("FAST", 1, None),
        ("STANDARD", 3, None),
        ("DEEP", 5, None),
        ("ADVANCED", 4, 4),
    ],
)
def test_both_improvements_follow_the_normal_flow_until_outer_limit(
    mode,
    rounds,
    advanced_rounds,
):
    controller = AlternatingController(
        AlternationPolicy.for_mode(mode, advanced_rounds=advanced_rounds)
    )
    _enter_first_fir(controller)

    for round_index in range(rounds):
        fir = controller.after_fir(round_index=round_index, improved=True)
        assert fir.next_stage is AlternationStage.I2C_RETUNE
        assert fir.round_index == round_index

        retune = controller.after_retune(
            round_index=round_index,
            improved=True,
        )
        if round_index + 1 == rounds:
            assert retune.next_stage is AlternationStage.COMPLETE
            assert retune.round_index == round_index
        else:
            assert retune.next_stage is AlternationStage.FIR_ITERATIVE
            assert retune.round_index == round_index + 1


def test_successful_retune_cannot_repeat_an_already_completed_round():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    _enter_first_fir(controller)
    controller.after_fir(round_index=0, improved=True)
    decision = controller.after_retune(round_index=0, improved=True)
    assert decision.round_index == 1

    with pytest.raises(ValueError, match="round_index"):
        controller.after_fir(round_index=0, improved=True)


def test_no_valid_coarse_or_fine_result_stops_explicitly():
    no_coarse = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    no_coarse.start()
    no_fine = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    no_fine.start()
    no_fine.after_coarse(has_valid_tdecq=True)

    assert (
        no_coarse.after_coarse(has_valid_tdecq=False).next_stage
        is AlternationStage.COMPLETE
    )
    assert (
        no_fine.after_fine(has_valid_tdecq=False).next_stage
        is AlternationStage.COMPLETE
    )


def test_complete_rejects_every_transition_including_start():
    controller = AlternatingController(AlternationPolicy.for_mode("FAST"))
    _enter_first_fir(controller)
    controller.after_fir(round_index=0, improved=True)
    decision = controller.after_retune(round_index=0, improved=True)
    assert decision.next_stage is AlternationStage.COMPLETE

    transitions = (
        controller.start,
        lambda: controller.after_coarse(has_valid_tdecq=True),
        lambda: controller.after_fine(has_valid_tdecq=True),
        lambda: controller.after_fir(round_index=0, improved=True),
        lambda: controller.after_retune(round_index=0, improved=True),
    )
    for transition in transitions:
        with pytest.raises(ValueError, match="COMPLETE"):
            transition()


def _fir(main: int = 104) -> FirTarget:
    return FirTarget(
        taps=(0, 6, -28, main, -8, -10, 0),
        upper_nlc=6,
        lower_nlc=6,
    )


def _combination(
    combination_id: str,
    *,
    eq: int,
    tdecq_db: float | None,
    er_db: float | None = None,
) -> CompleteTuningCombination:
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)
        if tdecq_db is not None
        else MetricResult.invalid(status="INV", reason="invalid acquisition")
    )
    return CompleteTuningCombination(
        combination_id=combination_id,
        readback_i2c={"eq": eq},
        fir=_fir(),
        result=result,
    )


def test_global_best_is_a_complete_i2c_fir_combination_with_strict_objective():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    rows = (
        _combination("invalid", eq=0, tdecq_db=None),
        _combination("higher-er-worse-tdecq", eq=1, tdecq_db=2.5, er_db=100.0),
        _combination("low-tdecq-low-er", eq=2, tdecq_db=2.4, er_db=1.0),
        _combination("tied-tdecq-high-er", eq=3, tdecq_db=2.4, er_db=2.0),
    )

    for row in rows:
        controller.record(row)

    assert controller.history == rows
    assert controller.global_best == rows[-1]
    assert controller.global_best.readback_i2c == {"eq": 3}
    assert controller.global_best.fir == _fir()


def test_global_best_exact_tie_is_deterministic_by_combination_id():
    first = _combination("combo-z", eq=0, tdecq_db=2.4, er_db=2.0)
    second = _combination("combo-a", eq=1, tdecq_db=2.4, er_db=2.0)
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))

    controller.record(first)
    controller.record(second)

    assert controller.global_best == second


@pytest.mark.parametrize("round_index", [-1, True])
def test_round_index_must_be_a_nonnegative_integer(round_index):
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    _enter_first_fir(controller)
    with pytest.raises(ValueError, match="round_index"):
        controller.after_fir(round_index=round_index, improved=True)


def test_combination_is_transitively_immutable():
    source = {"eq": 2}
    combination = CompleteTuningCombination(
        combination_id="immutable",
        readback_i2c=source,
        fir=_fir(),
        result=MetricResult.valid(tdecq_db=2.4, er_db=2.0),
    )
    source["eq"] = 9

    assert combination.readback_i2c == {"eq": 2}
    with pytest.raises(TypeError):
        combination.readback_i2c["eq"] = 3
    with pytest.raises(ValidationError):
        combination.fir = _fir()
