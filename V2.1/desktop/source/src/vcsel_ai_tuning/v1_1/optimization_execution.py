"""Hardware execution adapters for the separate pure V1.1 optimizer package.

No objective, Bayesian, FIR formula, median, or alternation policy is
implemented here.  The injected optimizer controllers make those decisions;
this module executes their directives and enforces hardware rollback behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class FailedInvalidDcaTaps(RuntimeError):
    pass


def _value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    improved: bool
    has_valid_tdecq: bool
    result: Any
    should_retune: bool


class FirHardware(Protocol):
    def read_dca_taps(self) -> tuple[object, int]: ...

    def apply_fir(self, target: Any) -> None: ...

    def measure(self) -> Any: ...

    def restore_fir(self, target: Any) -> None: ...

    def restore_b0(self) -> None: ...


class FirStageExecutor:
    """Execute direct FIRST/ITERATIVE decisions from FirDecisionController."""

    def __init__(self, *, controller: Any, hardware: FirHardware) -> None:
        self._controller = controller
        self._hardware = hardware

    def run(self, *, mode: str) -> StageOutcome:
        dca_taps, dca_main_index = self._hardware.read_dca_taps()
        if mode == "FIRST":
            decision = self._controller.propose_first(
                dca_taps, dca_main_index=dca_main_index
            )
        elif mode == "ITERATIVE":
            decision = self._controller.propose_iterative(
                dca_taps, dca_main_index=dca_main_index
            )
        else:
            raise ValueError("FIR mode must be FIRST or ITERATIVE")

        status = _value(decision.status)
        if status == "FAILED_INVALID_DCA_TAPS":
            self._hardware.restore_b0()
            raise FailedInvalidDcaTaps(getattr(decision, "reason", status))
        candidate = getattr(decision, "candidate", None)
        if status != "CANDIDATE" or candidate is None:
            recommended = getattr(decision, "recommended_fir", None)
            if recommended is not None:
                self._hardware.restore_fir(recommended)
            return StageOutcome(
                improved=False,
                has_valid_tdecq=False,
                result=getattr(decision, "aggregate_result", None),
                should_retune=False,
            )

        self._hardware.apply_fir(candidate.requested_fir)
        first = self._hardware.measure()
        evaluated = self._controller.evaluate(candidate, (first,))
        if _value(evaluated.status) == "NEEDS_CONFIRMATION":
            second = self._hardware.measure()
            third = self._hardware.measure()
            evaluated = self._controller.evaluate(
                candidate, (first, second, third)
            )

        accepted = bool(getattr(evaluated, "accepted", False))
        recommended = getattr(evaluated, "recommended_fir", None)
        if not accepted:
            if recommended is not None:
                self._hardware.restore_fir(recommended)
            else:
                self._hardware.restore_b0()
        aggregate = getattr(evaluated, "aggregate_result", None)
        has_valid = aggregate is not None and bool(
            getattr(aggregate, "has_valid_tdecq", True)
        )
        return StageOutcome(
            improved=accepted,
            has_valid_tdecq=has_valid,
            result=aggregate,
            should_retune=bool(getattr(evaluated, "should_retune", False)),
        )


class FinalHardware(Protocol):
    def apply_complete(self, state: Any) -> Any: ...

    def measure(self) -> Any: ...

    def restore_b0(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FinalVerificationResult:
    state: str
    applied_readback: Any
    raw_results: tuple[Any, Any, Any]
    aggregate_result: Any


class FinalVerificationExecutor:
    """Execute three final reads; injected optimizer owns aggregate/compare."""

    def __init__(
        self,
        *,
        hardware: FinalHardware,
        aggregate_three: Callable[[tuple[Any, Any, Any]], Any],
        compare_results: Callable[[Any, Any], Any],
    ) -> None:
        self._hardware = hardware
        self._aggregate_three = aggregate_three
        self._compare_results = compare_results

    def verify(self, *, best_state: Any, baseline_result: Any) -> FinalVerificationResult:
        readback = self._hardware.apply_complete(best_state)
        raw = (
            self._hardware.measure(),
            self._hardware.measure(),
            self._hardware.measure(),
        )
        aggregate = self._aggregate_three(raw)
        comparison = _value(self._compare_results(aggregate, baseline_result))
        if comparison == "BETTER":
            state = "COMPLETED"
        else:
            self._hardware.restore_b0()
            state = "COMPLETED_NO_IMPROVEMENT"
        return FinalVerificationResult(
            state=state,
            applied_readback=readback,
            raw_results=raw,
            aggregate_result=aggregate,
        )


class ExecutableStage(Protocol):
    def run(self, **kwargs: Any) -> Any: ...


class AlternatingCampaignExecutor:
    """Follow AlternatingController directives without duplicating its policy."""

    def __init__(
        self,
        *,
        controller: Any,
        coarse: ExecutableStage,
        fine: ExecutableStage,
        fir: ExecutableStage,
        retune: ExecutableStage,
        final: ExecutableStage,
    ) -> None:
        self._controller = controller
        self._coarse = coarse
        self._fine = fine
        self._fir = fir
        self._retune = retune
        self._final = final

    def run(self) -> Any:
        decision = self._controller.start()
        while True:
            stage = _value(decision.next_stage)
            if stage == "I2C_COARSE":
                outcome = self._coarse.run()
                decision = self._controller.after_coarse(
                    has_valid_tdecq=outcome.has_valid_tdecq
                )
            elif stage == "I2C_FINE":
                outcome = self._fine.run()
                decision = self._controller.after_fine(
                    has_valid_tdecq=outcome.has_valid_tdecq
                )
            elif stage in {"FIR_FIRST", "FIR_ITERATIVE"}:
                outcome = self._fir.run(
                    mode="FIRST" if stage == "FIR_FIRST" else "ITERATIVE"
                )
                decision = self._controller.after_fir(
                    round_index=decision.round_index,
                    improved=outcome.improved,
                )
            elif stage == "I2C_RETUNE":
                outcome = self._retune.run(round_index=decision.round_index)
                decision = self._controller.after_retune(
                    round_index=decision.round_index,
                    improved=outcome.improved,
                )
            elif stage == "COMPLETE":
                return self._final.run()
            else:
                raise RuntimeError(f"optimizer returned unknown stage {stage}")
