from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.v1_1.optimization_execution import (
    AlternatingCampaignExecutor,
    FailedInvalidDcaTaps,
    FinalVerificationExecutor,
    FirStageExecutor,
    StageOutcome,
)


class FirHardware:
    def __init__(self, measurements):
        self.measurements = list(measurements)
        self.calls = []

    def read_dca_taps(self):
        self.calls.append(("dca_taps",))
        return (0.1, 1.0, -0.1), 1

    def apply_fir(self, target):
        self.calls.append(("apply", target))

    def measure(self):
        self.calls.append(("measure",))
        return self.measurements.pop(0)

    def restore_fir(self, target):
        self.calls.append(("restore_fir", target))

    def restore_b0(self):
        self.calls.append(("restore_b0",))


class FirController:
    def __init__(self, final_status="ACCEPTED"):
        self.final_status = final_status
        self.candidate = SimpleNamespace(requested_fir="DIRECT_FORMULA_TARGET")
        self.evaluations = []

    def propose_first(self, taps, *, dca_main_index):
        return SimpleNamespace(status="CANDIDATE", candidate=self.candidate)

    def propose_iterative(self, taps, *, dca_main_index):
        return SimpleNamespace(status="CANDIDATE", candidate=self.candidate)

    def evaluate(self, candidate, measurements):
        self.evaluations.append(measurements)
        if len(measurements) == 1:
            return SimpleNamespace(
                status="NEEDS_CONFIRMATION",
                accepted=False,
                should_retune=False,
                recommended_fir=None,
                aggregate_result=measurements[0],
                no_improvement_count=0,
            )
        return SimpleNamespace(
            status=self.final_status,
            accepted=self.final_status == "ACCEPTED",
            should_retune=self.final_status == "ACCEPTED",
            recommended_fir="BEST_FIR",
            aggregate_result="MEDIAN_RESULT",
            no_improvement_count=0 if self.final_status == "ACCEPTED" else 1,
        )


def test_fir_stage_consumes_direct_controller_target_and_three_result_decision():
    controller = FirController()
    hardware = FirHardware(["first", "second", "third"])

    outcome = FirStageExecutor(controller=controller, hardware=hardware).run(
        mode="FIRST"
    )

    assert outcome.improved is True
    assert outcome.should_retune is True
    assert outcome.result == "MEDIAN_RESULT"
    assert [call for call in hardware.calls if call[0] == "apply"] == [
        ("apply", "DIRECT_FORMULA_TARGET")
    ]
    assert len([call for call in hardware.calls if call[0] == "measure"]) == 3
    assert hardware.calls == [
        ("dca_taps",),
        ("apply", "DIRECT_FORMULA_TARGET"),
        ("measure",),
        ("measure",),
        ("measure",),
    ]
    assert controller.evaluations == [
        ("first",),
        ("first", "second", "third"),
    ]


def test_invalid_dca_taps_fail_before_any_fir_write_or_restore():
    controller = FirController()
    controller.propose_first = lambda *args, **kwargs: SimpleNamespace(
        status="FAILED_INVALID_DCA_TAPS",
        candidate=None,
        reason="bad taps",
        recommended_fir="LATEST_BEST",
    )
    hardware = FirHardware([])

    with pytest.raises(FailedInvalidDcaTaps, match="bad taps"):
        FirStageExecutor(controller=controller, hardware=hardware).run(mode="FIRST")

    assert hardware.calls == [("dca_taps",)]


def test_rejected_fir_restores_controller_recommended_historical_best():
    controller = FirController(final_status="REJECTED_FIRST_PASS")
    hardware = FirHardware(["first", "second", "third"])

    outcome = FirStageExecutor(controller=controller, hardware=hardware).run(
        mode="FIRST"
    )

    assert outcome.improved is False
    assert outcome.should_retune is False
    assert ("restore_fir", "BEST_FIR") in hardware.calls


class FinalHardware:
    def __init__(self):
        self.calls = []

    def apply_complete(self, state):
        self.calls.append(("apply_complete", state))
        return state

    def measure(self):
        index = len([call for call in self.calls if call[0] == "measure"])
        self.calls.append(("measure",))
        return (2.2, 2.4, 2.3)[index]

    def restore_b0(self):
        self.calls.append(("restore_b0",))


def test_final_verification_measures_three_and_restores_b0_when_not_better():
    hardware = FinalHardware()
    verifier = FinalVerificationExecutor(
        hardware=hardware,
        aggregate_three=lambda values: sorted(values)[1],
        compare_results=lambda candidate, baseline: "EQUAL",
    )

    result = verifier.verify(best_state="GLOBAL_BEST", baseline_result=2.3)

    assert result.state == "COMPLETED_NO_IMPROVEMENT"
    assert result.raw_results == (2.2, 2.4, 2.3)
    assert result.aggregate_result == 2.3
    assert hardware.calls[-1] == ("restore_b0",)


class AlternatingController:
    def __init__(self, *, fir_improved):
        self.fir_improved = fir_improved

    def start(self):
        return SimpleNamespace(next_stage="I2C_COARSE", round_index=0)

    def after_coarse(self, *, has_valid_tdecq):
        assert has_valid_tdecq
        return SimpleNamespace(next_stage="I2C_FINE", round_index=0)

    def after_fine(self, *, has_valid_tdecq):
        assert has_valid_tdecq
        return SimpleNamespace(next_stage="FIR_FIRST", round_index=0)

    def after_fir(self, *, round_index, improved):
        assert improved is self.fir_improved
        return SimpleNamespace(
            next_stage="I2C_RETUNE" if improved else "COMPLETE",
            round_index=round_index,
        )

    def after_retune(self, *, round_index, improved):
        return SimpleNamespace(next_stage="COMPLETE", round_index=round_index)


class Stage:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        return self.outcome


def test_alternating_executor_stops_before_retune_when_fir_does_not_improve():
    coarse = Stage(StageOutcome(True, True, "coarse", False))
    fine = Stage(StageOutcome(True, True, "fine", False))
    fir = Stage(StageOutcome(False, True, "fir", False))
    retune = Stage(StageOutcome(True, True, "retune", False))
    final = Stage("final")
    executor = AlternatingCampaignExecutor(
        controller=AlternatingController(fir_improved=False),
        coarse=coarse,
        fine=fine,
        fir=fir,
        retune=retune,
        final=final,
    )

    result = executor.run()

    assert result == "final"
    assert coarse.calls == fine.calls == fir.calls == final.calls == 1
    assert retune.calls == 0
