from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, StrictInt, model_validator

from .fir import FirTarget
from .models import FrozenModel, Identifier, ImmutableIntMapping, MetricResult
from .objective import Comparison, compare_results


class AlternationMode(StrEnum):
    FAST = "FAST"
    STANDARD = "STANDARD"
    DEEP = "DEEP"
    ADVANCED = "ADVANCED"


class AlternationStage(StrEnum):
    I2C_COARSE = "I2C_COARSE"
    I2C_FINE = "I2C_FINE"
    FIR_FIRST = "FIR_FIRST"
    I2C_RETUNE = "I2C_RETUNE"
    FIR_ITERATIVE = "FIR_ITERATIVE"
    COMPLETE = "COMPLETE"


class AlternationPolicy(FrozenModel):
    mode: AlternationMode
    max_rounds: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_fixed_profile(self) -> Self:
        fixed = {
            AlternationMode.FAST: 1,
            AlternationMode.STANDARD: 3,
            AlternationMode.DEEP: 5,
        }
        expected = fixed.get(self.mode)
        if expected is not None and self.max_rounds != expected:
            raise ValueError(f"{self.mode} requires exactly {expected} rounds")
        return self

    @classmethod
    def for_mode(
        cls,
        mode: AlternationMode | str,
        *,
        advanced_rounds: int | None = None,
    ) -> "AlternationPolicy":
        try:
            normalized = (
                mode
                if isinstance(mode, AlternationMode)
                else AlternationMode(mode)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("mode must be FAST, STANDARD, DEEP, or ADVANCED") from error
        fixed = {
            AlternationMode.FAST: 1,
            AlternationMode.STANDARD: 3,
            AlternationMode.DEEP: 5,
        }
        if normalized is AlternationMode.ADVANCED:
            if (
                not isinstance(advanced_rounds, int)
                or isinstance(advanced_rounds, bool)
                or advanced_rounds <= 0
            ):
                raise ValueError("ADVANCED requires a positive integer advanced_rounds")
            return cls(mode=normalized, max_rounds=advanced_rounds)
        if advanced_rounds is not None:
            raise ValueError("advanced_rounds is only valid for ADVANCED mode")
        return cls(mode=normalized, max_rounds=fixed[normalized])


class AlternationDecision(FrozenModel):
    next_stage: AlternationStage
    reason: str = Field(min_length=1)
    round_index: StrictInt = Field(ge=0)


class CompleteTuningCombination(FrozenModel):
    combination_id: Identifier
    readback_i2c: ImmutableIntMapping
    fir: FirTarget
    result: MetricResult


class AlternatingController:
    """Pure outer-loop routing plus complete-setting global-best retention."""

    def __init__(self, policy: AlternationPolicy) -> None:
        if not isinstance(policy, AlternationPolicy):
            raise TypeError("policy must be an AlternationPolicy")
        self.policy = policy
        self._history: list[CompleteTuningCombination] = []
        self._next_stage: AlternationStage | None = None
        self._round_index = 0

    @property
    def history(self) -> tuple[CompleteTuningCombination, ...]:
        return tuple(self._history)

    @property
    def global_best(self) -> CompleteTuningCombination | None:
        best: CompleteTuningCombination | None = None
        for combination in sorted(
            self._history,
            key=lambda item: item.combination_id,
        ):
            if not combination.result.has_valid_tdecq:
                continue
            if best is None or compare_results(
                combination.result,
                best.result,
            ) is Comparison.BETTER:
                best = combination
        return best

    def record(self, combination: CompleteTuningCombination) -> None:
        if not isinstance(combination, CompleteTuningCombination):
            raise TypeError("combination must be a CompleteTuningCombination")
        self._history.append(combination)

    def start(self) -> AlternationDecision:
        if self._next_stage is not None:
            self._reject_transition("start", expected="not started")
        decision = AlternationDecision(
            next_stage=AlternationStage.I2C_COARSE,
            reason="start deterministic global I2C coarse coverage",
            round_index=0,
        )
        self._next_stage = decision.next_stage
        return decision

    def after_coarse(self, *, has_valid_tdecq: bool) -> AlternationDecision:
        self._require_stage("after_coarse", AlternationStage.I2C_COARSE)
        self._validate_bool(has_valid_tdecq, "has_valid_tdecq")
        decision = AlternationDecision(
            next_stage=(
                AlternationStage.I2C_FINE
                if has_valid_tdecq
                else AlternationStage.COMPLETE
            ),
            reason=(
                "coarse found valid TDECQ; continue to Bayesian fine"
                if has_valid_tdecq
                else "coarse exhausted without valid TDECQ"
            ),
            round_index=0,
        )
        self._next_stage = decision.next_stage
        return decision

    def after_fine(self, *, has_valid_tdecq: bool) -> AlternationDecision:
        self._require_stage("after_fine", AlternationStage.I2C_FINE)
        self._validate_bool(has_valid_tdecq, "has_valid_tdecq")
        decision = AlternationDecision(
            next_stage=(
                AlternationStage.FIR_FIRST
                if has_valid_tdecq
                else AlternationStage.COMPLETE
            ),
            reason=(
                "fine has a valid reference; start first-pass FIR"
                if has_valid_tdecq
                else "fine has no valid TDECQ reference"
            ),
            round_index=0,
        )
        self._next_stage = decision.next_stage
        return decision

    def after_fir(
        self,
        *,
        round_index: int,
        improved: bool,
    ) -> AlternationDecision:
        self._require_stage(
            "after_fir",
            AlternationStage.FIR_FIRST,
            AlternationStage.FIR_ITERATIVE,
        )
        self._validate_round(round_index)
        self._validate_bool(improved, "improved")
        decision = AlternationDecision(
            next_stage=(
                AlternationStage.I2C_RETUNE
                if improved
                else AlternationStage.COMPLETE
            ),
            reason=(
                "FIR improved; run context-isolated I2C retune"
                if improved
                else "FIR did not improve; stop before I2C retune"
            ),
            round_index=round_index,
        )
        self._next_stage = decision.next_stage
        return decision

    def after_retune(
        self,
        *,
        round_index: int,
        improved: bool,
    ) -> AlternationDecision:
        self._require_stage("after_retune", AlternationStage.I2C_RETUNE)
        self._validate_round(round_index)
        self._validate_bool(improved, "improved")
        at_limit = round_index + 1 >= self.policy.max_rounds
        if not improved:
            next_stage = AlternationStage.COMPLETE
            reason = "I2C retune did not improve; stop before the next FIR"
        elif at_limit:
            next_stage = AlternationStage.COMPLETE
            reason = "the configured outer alternation round limit is reached"
        else:
            next_stage = AlternationStage.FIR_ITERATIVE
            reason = "FIR and I2C retune improved; continue with iterative FIR"
        decision_round_index = (
            round_index + 1
            if next_stage is AlternationStage.FIR_ITERATIVE
            else round_index
        )
        decision = AlternationDecision(
            next_stage=next_stage,
            reason=reason,
            round_index=decision_round_index,
        )
        self._round_index = decision_round_index
        self._next_stage = decision.next_stage
        return decision

    def _validate_round(self, round_index: int) -> None:
        if (
            not isinstance(round_index, int)
            or isinstance(round_index, bool)
            or round_index < 0
            or round_index >= self.policy.max_rounds
            or round_index != self._round_index
        ):
            raise ValueError(
                "round_index must equal the current configured outer round"
            )

    def _require_stage(
        self,
        transition: str,
        *expected: AlternationStage,
    ) -> None:
        if self._next_stage is AlternationStage.COMPLETE:
            raise ValueError(
                f"{transition} transition is not allowed after COMPLETE"
            )
        if self._next_stage not in expected:
            expected_text = " or ".join(stage.value for stage in expected)
            self._reject_transition(transition, expected=expected_text)

    def _reject_transition(self, transition: str, *, expected: str) -> None:
        current = (
            "not started"
            if self._next_stage is None
            else self._next_stage.value
        )
        if self._next_stage is AlternationStage.COMPLETE:
            raise ValueError(
                f"{transition} transition is not allowed after COMPLETE"
            )
        raise ValueError(
            f"{transition} transition requires {expected}; current stage is {current}"
        )

    @staticmethod
    def _validate_bool(value: bool, name: str) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
