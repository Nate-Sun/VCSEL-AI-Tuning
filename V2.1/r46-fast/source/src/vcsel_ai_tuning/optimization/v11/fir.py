from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import StrEnum
from numbers import Real
from statistics import median
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from .models import (
    AggregatedObservation,
    ExperimentFingerprint,
    FrozenModel,
    I2CObservation,
    MetricResult,
)
from .objective import Comparison, compare_results


FIR_TAP_COUNT = 7
FIR_MAIN_INDEX = 3
FIR_L1_LIMIT = 170


class FirTarget(FrozenModel):
    taps: tuple[StrictInt, ...] = Field(min_length=FIR_TAP_COUNT, max_length=FIR_TAP_COUNT)
    upper_nlc: StrictInt = Field(ge=0, le=255)
    lower_nlc: StrictInt = Field(ge=0, le=255)
    pattern: Literal["SSPRQ"] = "SSPRQ"

    @model_validator(mode="after")
    def validate_main_and_l1(self) -> Self:
        if self.taps[FIR_MAIN_INDEX] <= 0:
            raise ValueError("FIR Main must be positive")
        if self.l1 > FIR_L1_LIMIT:
            raise ValueError(f"FIR L1 must be at most {FIR_L1_LIMIT}")
        return self

    @property
    def l1(self) -> int:
        return sum(abs(value) for value in self.taps)


class FirAuthorizationEnvelope(FrozenModel):
    tap_ranges: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        min_length=FIR_TAP_COUNT,
        max_length=FIR_TAP_COUNT,
    )
    max_step: tuple[StrictInt, ...] = Field(
        min_length=FIR_TAP_COUNT,
        max_length=FIR_TAP_COUNT,
    )
    max_total_change_l1: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        for index, (minimum, maximum) in enumerate(self.tap_ranges):
            if minimum > maximum:
                raise ValueError(f"tap range {index} has minimum above maximum")
        if self.tap_ranges[FIR_MAIN_INDEX][0] <= 0:
            raise ValueError("authorized FIR Main range must be positive")
        if any(step <= 0 for step in self.max_step):
            raise ValueError("FIR max_step values must be positive")
        return self

    def rejection_reason(
        self,
        target: FirTarget,
        current: FirTarget,
    ) -> str | None:
        if target.upper_nlc != current.upper_nlc or target.lower_nlc != current.lower_nlc:
            return "Upper/Lower NLC must remain fixed at B0"
        total_change = 0
        for index, (requested, previous, limits, max_step) in enumerate(
            zip(
                target.taps,
                current.taps,
                self.tap_ranges,
                self.max_step,
                strict=True,
            )
        ):
            minimum, maximum = limits
            if requested < minimum or requested > maximum:
                return f"FIR tap {index} is outside its authorized range"
            delta = abs(requested - previous)
            if delta > max_step:
                return f"FIR tap {index} exceeds max_step"
            total_change += delta
        if total_change > self.max_total_change_l1:
            return "FIR total change exceeds max_total_change_l1"
        return None


class FirCandidate(FrozenModel):
    candidate_id: Literal["fir-formula-direct"] = "fir-formula-direct"
    requested_fir: FirTarget
    mode: Literal["FIRST", "ITERATIVE"]
    model_version: Literal["fir-formula-direct-v1.1"] = "fir-formula-direct-v1.1"
    reason: Literal["direct FIR formula proposal"] = "direct FIR formula proposal"


class FirDecisionStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    FAILED_INVALID_DCA_TAPS = "FAILED_INVALID_DCA_TAPS"
    REJECTED_CONSTRAINT = "REJECTED_CONSTRAINT"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    ACCEPTED = "ACCEPTED"
    REJECTED_FIRST_PASS = "REJECTED_FIRST_PASS"
    REJECTED_ITERATIVE = "REJECTED_ITERATIVE"
    CONVERGED_NO_IMPROVEMENT = "CONVERGED_NO_IMPROVEMENT"


class FirDecision(FrozenModel):
    status: FirDecisionStatus
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    candidate: FirCandidate | None = None
    aggregate_result: MetricResult | None = None
    accepted: bool = False
    confirmation_remaining: StrictInt = Field(default=0, ge=0, le=2)
    recommended_fir: FirTarget | None = None
    should_retune: bool = False
    no_improvement_count: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.status is FirDecisionStatus.CANDIDATE and self.candidate is None:
            raise ValueError("CANDIDATE requires a direct FIR candidate")
        if self.status in {
            FirDecisionStatus.FAILED_INVALID_DCA_TAPS,
            FirDecisionStatus.REJECTED_CONSTRAINT,
            FirDecisionStatus.CONVERGED_NO_IMPROVEMENT,
        } and self.candidate is not None:
            raise ValueError(f"{self.status} cannot carry a new candidate")
        if self.accepted != (self.status is FirDecisionStatus.ACCEPTED):
            raise ValueError("accepted flag must agree with decision status")
        if self.should_retune and not self.accepted:
            raise ValueError("only an accepted FIR may request I2C retune")
        if self.status is FirDecisionStatus.NEEDS_CONFIRMATION:
            if self.confirmation_remaining != 2 or self.candidate is None:
                raise ValueError("NEEDS_CONFIRMATION requires one result and two repeats")
        elif self.confirmation_remaining != 0:
            raise ValueError("only NEEDS_CONFIRMATION may have repeats remaining")
        return self


class FirDecisionController:
    """Pure state machine around injected, unchanged FIR formula functions."""

    def __init__(
        self,
        *,
        baseline_fir: FirTarget,
        baseline_result: MetricResult,
        authorization: FirAuthorizationEnvelope,
        expected_dca_tap_count: int,
        calculate_first_pass: Callable[..., object],
        calculate_iterative_pass: Callable[..., object],
        dca_config_fingerprint: str,
    ) -> None:
        if not isinstance(baseline_fir, FirTarget):
            raise TypeError("baseline_fir must be a FirTarget")
        if not isinstance(baseline_result, MetricResult) or not baseline_result.has_valid_tdecq:
            raise ValueError("baseline_result must contain a valid TDECQ")
        if not isinstance(authorization, FirAuthorizationEnvelope):
            raise TypeError("authorization must be a FirAuthorizationEnvelope")
        if (
            not isinstance(expected_dca_tap_count, int)
            or isinstance(expected_dca_tap_count, bool)
            or expected_dca_tap_count <= 0
        ):
            raise ValueError("expected_dca_tap_count must be a positive integer")
        if not callable(calculate_first_pass) or not callable(calculate_iterative_pass):
            raise TypeError("FIR formula dependencies must be callable")
        baseline_rejection = authorization.rejection_reason(baseline_fir, baseline_fir)
        if baseline_rejection is not None:
            raise ValueError(f"B0 FIR is outside the authorization: {baseline_rejection}")
        baseline_fingerprint = ExperimentFingerprint(
            fir_taps=baseline_fir.taps,
            upper_nlc=baseline_fir.upper_nlc,
            lower_nlc=baseline_fir.lower_nlc,
            dca_config_fingerprint=dca_config_fingerprint,
        )

        self.baseline_fir = baseline_fir
        self.authorization = authorization
        self.expected_dca_tap_count = expected_dca_tap_count
        self._dca_config_fingerprint = baseline_fingerprint.dca_config_fingerprint
        self._calculate_first_pass = calculate_first_pass
        self._calculate_iterative_pass = calculate_iterative_pass
        self._best_fir = baseline_fir
        self._best_result = baseline_result
        self._first_accepted = False
        self._first_pass_stopped = False
        self._no_improvement_count = 0
        self._converged = False
        self._pending_confirmation: FirCandidate | None = None
        self._pending_first_result: MetricResult | None = None
        self._issued_candidate: FirCandidate | None = None

    @property
    def best_fir(self) -> FirTarget:
        return self._best_fir

    @property
    def best_result(self) -> MetricResult:
        return self._best_result

    @property
    def current_fingerprint(self) -> ExperimentFingerprint:
        return ExperimentFingerprint(
            fir_taps=self._best_fir.taps,
            upper_nlc=self._best_fir.upper_nlc,
            lower_nlc=self._best_fir.lower_nlc,
            dca_config_fingerprint=self._dca_config_fingerprint,
        )

    @property
    def retune_fingerprint(self) -> ExperimentFingerprint:
        """Return the exact experiment context required for a retune result."""
        return self.current_fingerprint

    @property
    def first_pass_stopped(self) -> bool:
        return self._first_pass_stopped

    @property
    def no_improvement_count(self) -> int:
        return self._no_improvement_count

    def propose_first(
        self,
        dca_taps: Sequence[Real] | None,
        *,
        dca_main_index: int,
    ) -> FirDecision:
        active = self._active_candidate_decision()
        if active is not None:
            return active
        if self._first_accepted:
            return self._failed_dca_decision(
                "first-pass is no longer allowed after an accepted FIR"
            )
        if self._first_pass_stopped:
            return FirDecision(
                status=FirDecisionStatus.REJECTED_FIRST_PASS,
                reason="first-pass already ended without improvement",
                recommended_fir=self.baseline_fir,
                no_improvement_count=self._no_improvement_count,
            )
        validated = self._validated_dca(dca_taps, dca_main_index)
        if isinstance(validated, FirDecision):
            return validated
        taps, main_index = validated
        try:
            formula = self._calculate_first_pass(
                taps,
                dca_main_index=main_index,
                manual_main=self.baseline_fir.taps[FIR_MAIN_INDEX],
            )
        except (TypeError, ValueError, ArithmeticError) as error:
            return self._failed_dca_decision(f"FIR first-pass formula failed: {error}")
        return self._candidate_from_formula(formula, mode="FIRST")

    def propose_iterative(
        self,
        dca_taps: Sequence[Real] | None,
        *,
        dca_main_index: int,
    ) -> FirDecision:
        active = self._active_candidate_decision()
        if active is not None:
            return active
        if self._converged:
            return FirDecision(
                status=FirDecisionStatus.CONVERGED_NO_IMPROVEMENT,
                reason="three consecutive iterative FIR candidates did not improve",
                recommended_fir=self._best_fir,
                no_improvement_count=self._no_improvement_count,
            )
        if not self._first_accepted:
            return self._failed_dca_decision(
                "iterative FIR requires an accepted first-pass candidate"
            )
        validated = self._validated_dca(dca_taps, dca_main_index)
        if isinstance(validated, FirDecision):
            return validated
        taps, main_index = validated
        try:
            formula = self._calculate_iterative_pass(
                self._best_fir.taps,
                taps,
                dca_main_index=main_index,
                manual_main=self.baseline_fir.taps[FIR_MAIN_INDEX],
            )
        except (TypeError, ValueError, ArithmeticError) as error:
            return self._failed_dca_decision(f"FIR iterative formula failed: {error}")
        return self._candidate_from_formula(formula, mode="ITERATIVE")

    def evaluate(
        self,
        candidate: FirCandidate,
        measurements: tuple[MetricResult, ...],
    ) -> FirDecision:
        if not isinstance(candidate, FirCandidate):
            raise TypeError("candidate must be a FirCandidate")
        if candidate != self._issued_candidate:
            raise ValueError("candidate was not issued by this FIR controller")
        self._validate_issued_candidate(candidate)
        if not isinstance(measurements, tuple) or len(measurements) not in {1, 3}:
            raise ValueError("FIR evaluation requires one or three measurements")
        if not all(isinstance(item, MetricResult) for item in measurements):
            raise TypeError("measurements must contain MetricResult values")
        if len(measurements) == 3 and self._pending_confirmation is None:
            raise ValueError(
                "the first measurement must be evaluated before its two confirmations"
            )
        if self._pending_confirmation is not None and len(measurements) == 1:
            raise ValueError("pending FIR confirmation requires the complete three-result batch")
        if self._pending_confirmation is not None and len(measurements) == 3:
            if candidate != self._pending_confirmation:
                raise ValueError("confirmation measurements must match the pending FIR")
            if measurements[0] != self._pending_first_result:
                raise ValueError(
                    "confirmation batch must include the original first result"
                )

        if len(measurements) == 1:
            first = measurements[0]
            if compare_results(first, self._best_result) is Comparison.BETTER:
                self._pending_confirmation = candidate
                self._pending_first_result = first
                return FirDecision(
                    status=FirDecisionStatus.NEEDS_CONFIRMATION,
                    reason="first result improved; two confirmation measurements remain",
                    candidate=candidate,
                    aggregate_result=first,
                    confirmation_remaining=2,
                    no_improvement_count=self._no_improvement_count,
                )
            return self._reject(candidate, first)

        if compare_results(measurements[0], self._best_result) is not Comparison.BETTER:
            return self._reject(candidate, measurements[0])
        aggregate = _confirmation_result(measurements)
        self._pending_confirmation = None
        self._pending_first_result = None
        if compare_results(aggregate, self._best_result) is Comparison.BETTER:
            self._best_result = aggregate
            self._best_fir = candidate.requested_fir
            self._first_accepted = True
            self._no_improvement_count = 0
            self._issued_candidate = None
            return FirDecision(
                status=FirDecisionStatus.ACCEPTED,
                reason="confirmed FIR median strictly improved the historical best",
                candidate=candidate,
                aggregate_result=aggregate,
                accepted=True,
                recommended_fir=self._best_fir,
                should_retune=True,
                no_improvement_count=0,
            )
        return self._reject(candidate, aggregate)

    def record_retune_result(
        self,
        observation: I2CObservation | AggregatedObservation,
    ) -> bool:
        """Advance the FIR comparison reference after same-FIR I2C retune."""
        if not isinstance(observation, (I2CObservation, AggregatedObservation)):
            raise TypeError(
                "retune result must be an I2CObservation or AggregatedObservation"
            )
        if (
            self._issued_candidate is not None
            or self._pending_confirmation is not None
        ):
            raise ValueError(
                "retune result cannot update while a FIR candidate is active or pending"
            )
        if not self._first_accepted:
            raise ValueError("retune results require an accepted FIR")
        if observation.fingerprint != self.retune_fingerprint:
            raise ValueError(
                "retune observation fingerprint must exactly match the current "
                "FIR, Upper/Lower NLC, and DCA config"
            )
        result = observation.result
        if compare_results(result, self._best_result) is Comparison.BETTER:
            self._best_result = result
            return True
        return False

    def _validated_dca(self, dca_taps, dca_main_index):
        if dca_taps is None or isinstance(dca_taps, (str, bytes, bytearray)):
            return self._failed_dca_decision("DCA Equalizer taps are missing")
        try:
            raw = tuple(dca_taps)
        except TypeError:
            return self._failed_dca_decision("DCA Equalizer taps are not a sequence")
        if len(raw) != self.expected_dca_tap_count:
            return self._failed_dca_decision(
                f"DCA Equalizer tap count must be {self.expected_dca_tap_count}"
            )
        if (
            not isinstance(dca_main_index, int)
            or isinstance(dca_main_index, bool)
            or dca_main_index < 0
            or dca_main_index >= len(raw)
        ):
            return self._failed_dca_decision("DCA Equalizer Main index is invalid")
        converted: list[float] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, Real):
                return self._failed_dca_decision("DCA Equalizer taps must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                return self._failed_dca_decision("DCA Equalizer taps must be finite")
            converted.append(numeric)
        return tuple(converted), dca_main_index

    def _candidate_from_formula(
        self,
        formula: object,
        *,
        mode: Literal["FIRST", "ITERATIVE"],
    ) -> FirDecision:
        if getattr(formula, "valid", None) is not True:
            formula_reason = getattr(formula, "reason", None)
            suffix = f": {formula_reason}" if formula_reason else ""
            return self._failed_dca_decision(
                f"DCA taps could not form a legal seven-tap FIR{suffix}"
            )
        raw_taps = getattr(formula, "taps", None)
        if (
            not isinstance(raw_taps, tuple)
            or len(raw_taps) != FIR_TAP_COUNT
            or any(not isinstance(tap, int) or isinstance(tap, bool) for tap in raw_taps)
        ):
            return self._failed_dca_decision(
                "DCA taps could not form exactly seven integer FIR taps"
            )
        try:
            target = FirTarget(
                taps=raw_taps,
                upper_nlc=self.baseline_fir.upper_nlc,
                lower_nlc=self.baseline_fir.lower_nlc,
            )
        except ValueError as error:
            return self._failed_dca_decision(f"formula FIR is invalid: {error}")
        if target.taps[FIR_MAIN_INDEX] != self.baseline_fir.taps[FIR_MAIN_INDEX]:
            return FirDecision(
                status=FirDecisionStatus.REJECTED_CONSTRAINT,
                reason="FIR Main must remain fixed at the B0 Main",
                no_improvement_count=self._no_improvement_count,
            )
        rejection = self.authorization.rejection_reason(target, self._best_fir)
        if rejection is not None:
            return FirDecision(
                status=FirDecisionStatus.REJECTED_CONSTRAINT,
                reason=rejection,
                no_improvement_count=self._no_improvement_count,
            )
        candidate = FirCandidate(requested_fir=target, mode=mode)
        self._issued_candidate = candidate
        return FirDecision(
            status=FirDecisionStatus.CANDIDATE,
            reason="direct FIR formula candidate is ready",
            candidate=candidate,
            no_improvement_count=self._no_improvement_count,
        )

    def _reject(
        self,
        candidate: FirCandidate,
        aggregate: MetricResult,
    ) -> FirDecision:
        self._pending_confirmation = None
        self._pending_first_result = None
        self._issued_candidate = None
        if candidate.mode == "FIRST":
            self._first_pass_stopped = True
            return FirDecision(
                status=FirDecisionStatus.REJECTED_FIRST_PASS,
                reason="first-pass FIR did not improve B0",
                candidate=candidate,
                aggregate_result=aggregate,
                recommended_fir=self.baseline_fir,
                no_improvement_count=self._no_improvement_count,
            )

        self._no_improvement_count += 1
        if self._no_improvement_count >= 3:
            self._converged = True
            return FirDecision(
                status=FirDecisionStatus.CONVERGED_NO_IMPROVEMENT,
                reason="three consecutive iterative FIR candidates did not improve",
                aggregate_result=aggregate,
                recommended_fir=self._best_fir,
                no_improvement_count=self._no_improvement_count,
            )
        return FirDecision(
            status=FirDecisionStatus.REJECTED_ITERATIVE,
            reason="iterative FIR did not improve the historical best",
            candidate=candidate,
            aggregate_result=aggregate,
            recommended_fir=self._best_fir,
            no_improvement_count=self._no_improvement_count,
        )

    def _failed_dca_decision(self, reason: str) -> FirDecision:
        return FirDecision(
            status=FirDecisionStatus.FAILED_INVALID_DCA_TAPS,
            reason=reason,
            no_improvement_count=self._no_improvement_count,
        )

    def _active_candidate_decision(self) -> FirDecision | None:
        if self._pending_confirmation is not None:
            assert self._pending_first_result is not None
            return FirDecision(
                status=FirDecisionStatus.NEEDS_CONFIRMATION,
                reason="first result improved; two confirmation measurements remain",
                candidate=self._pending_confirmation,
                aggregate_result=self._pending_first_result,
                confirmation_remaining=2,
                no_improvement_count=self._no_improvement_count,
            )
        if self._issued_candidate is not None:
            return FirDecision(
                status=FirDecisionStatus.CANDIDATE,
                reason="direct FIR formula candidate is ready",
                candidate=self._issued_candidate,
                no_improvement_count=self._no_improvement_count,
            )
        return None

    def _validate_issued_candidate(self, candidate: FirCandidate) -> None:
        target = candidate.requested_fir
        if (
            target.upper_nlc != self.baseline_fir.upper_nlc
            or target.lower_nlc != self.baseline_fir.lower_nlc
        ):
            raise ValueError("issued FIR NLC must remain fixed at B0")
        if target.taps[FIR_MAIN_INDEX] != self.baseline_fir.taps[FIR_MAIN_INDEX]:
            raise ValueError("issued FIR Main must remain fixed at B0")
        rejection = self.authorization.rejection_reason(target, self._best_fir)
        if rejection is not None:
            raise ValueError(f"issued FIR no longer satisfies authorization: {rejection}")


def _confirmation_result(
    measurements: tuple[MetricResult, ...],
) -> MetricResult:
    if any(not item.has_valid_tdecq for item in measurements):
        return MetricResult.invalid(
            status="INVALID_CONFIRMATION",
            reason="all three FIR confirmation measurements require valid TDECQ",
        )
    tdecq_values = tuple(
        float(item.tdecq_db)
        for item in measurements
        if item.tdecq_db is not None
    )
    er_values = tuple(
        float(item.er_db)
        for item in measurements
        if item.er_db is not None
    )
    return MetricResult.valid(
        tdecq_db=float(median(tdecq_values)),
        er_db=float(median(er_values)) if er_values else None,
    )
