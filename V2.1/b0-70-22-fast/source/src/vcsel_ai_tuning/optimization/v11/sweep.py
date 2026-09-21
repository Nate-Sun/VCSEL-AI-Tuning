from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import Self

from pydantic import Field, StrictInt, model_validator

from .fir import FIR_L1_LIMIT, FIR_MAIN_INDEX, FirAuthorizationEnvelope, FirTarget
from .grid import (
    DiscreteParameterSpace,
    TuningParameter,
    seeded_permutation_indices,
)
from .models import FrozenModel, Identifier, ImmutableIntMapping


FIR_TAP_NAMES = (
    "pre3",
    "pre2",
    "pre1",
    "main",
    "post1",
    "post2",
    "post3",
)
_FIR_INDEX = {name: index for index, name in enumerate(FIR_TAP_NAMES)}


class SweepTask(StrEnum):
    I2C_SWEEP = "I2C_SWEEP"
    BERT_FIR_SWEEP = "BERT_FIR_SWEEP"


class SweepMode(StrEnum):
    SINGLE_PARAMETER = "SINGLE_PARAMETER"
    FULL_COMBINATION = "FULL_COMBINATION"


class SweepPointStatus(StrEnum):
    READY = "READY"
    SKIPPED_CONSTRAINT = "SKIPPED_CONSTRAINT"


class SweepFirTarget(FrozenModel):
    """Raw requested taps; deliberately permits constraint-skipped values."""

    taps: tuple[StrictInt, ...] = Field(min_length=7, max_length=7)
    upper_nlc: StrictInt = Field(ge=0, le=255)
    lower_nlc: StrictInt = Field(ge=0, le=255)
    pattern: str = Field(default="SSPRQ", pattern=r"^SSPRQ$")


class SweepPoint(FrozenModel):
    task: SweepTask
    mode: SweepMode
    original_index: StrictInt = Field(ge=0)
    execution_index: StrictInt = Field(ge=0)
    status: SweepPointStatus
    selected_parameter: Identifier | None = None
    requested_i2c: ImmutableIntMapping | None = None
    requested_fir: SweepFirTarget | None = None
    constraint_reason: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if self.task is SweepTask.I2C_SWEEP:
            if self.requested_i2c is None or self.requested_fir is not None:
                raise ValueError("I2C sweep points require only requested_i2c")
        elif self.requested_fir is None or self.requested_i2c is not None:
            raise ValueError("BERT FIR sweep points require only requested_fir")

        if self.status is SweepPointStatus.SKIPPED_CONSTRAINT:
            if self.constraint_reason is None or not self.constraint_reason.strip():
                raise ValueError("SKIPPED_CONSTRAINT requires a reason")
        elif self.constraint_reason is not None:
            raise ValueError("READY points cannot have a constraint reason")
        if self.mode is SweepMode.SINGLE_PARAMETER:
            if self.selected_parameter is None:
                raise ValueError("single-parameter points require an axis name")
        elif self.selected_parameter is not None:
            raise ValueError("full-combination points do not have one selected axis")
        return self


class SweepPlanner:
    """Lazy independent I2C or FIR sweep point generation."""

    def __init__(
        self,
        *,
        task: SweepTask,
        mode: SweepMode,
        parameters: tuple[TuningParameter, ...],
        seed: int,
        max_points: int,
        baseline_i2c: Mapping[str, int] | None = None,
        i2c_allowlist: tuple[str, ...] = (),
        baseline_fir: FirTarget | None = None,
        fir_authorization: FirAuthorizationEnvelope | None = None,
    ) -> None:
        try:
            self.task = task if isinstance(task, SweepTask) else SweepTask(task)
            self.mode = mode if isinstance(mode, SweepMode) else SweepMode(mode)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid sweep task or mode") from error
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(max_points, int)
            or isinstance(max_points, bool)
            or max_points <= 0
        ):
            raise ValueError("max_points must be a positive integer")
        self.seed = seed
        self.max_points = max_points
        self.space = DiscreteParameterSpace(parameters)

        self._baseline_i2c: dict[str, int] | None = None
        self._baseline_fir: FirTarget | None = None
        self._fir_authorization: FirAuthorizationEnvelope | None = None
        if self.task is SweepTask.I2C_SWEEP:
            if baseline_fir is not None or fir_authorization is not None:
                raise ValueError("I2C_SWEEP must not include FIR inputs")
            self._configure_i2c(baseline_i2c, i2c_allowlist)
        else:
            if baseline_i2c is not None or i2c_allowlist:
                raise ValueError("BERT_FIR_SWEEP must not include I2C inputs")
            self._configure_fir(baseline_fir, fir_authorization)

        self.raw_point_count = (
            sum(parameter.axis_length for parameter in self.space.parameters)
            if self.mode is SweepMode.SINGLE_PARAMETER
            else self.space.point_count
        )
        if self.raw_point_count > self.max_points:
            raise ValueError(
                f"raw sweep point count {self.raw_point_count} exceeds max_points "
                f"{self.max_points}; automatic sampling is forbidden"
            )

    def points(self) -> Iterator[SweepPoint]:
        if self.mode is SweepMode.SINGLE_PARAMETER:
            yield from self._single_points()
            return
        for execution_index, original_index in enumerate(
            seeded_permutation_indices(self.space.point_count, self.seed)
        ):
            assignments = self.space.point_from_index(original_index)
            yield self._point(
                assignments,
                original_index=original_index,
                execution_index=execution_index,
                selected_parameter=None,
            )

    def _single_points(self) -> Iterator[SweepPoint]:
        original_index = 0
        for parameter in self.space.parameters:
            for position in range(parameter.axis_length):
                assignments = {parameter.name: parameter.value_at(position)}
                yield self._point(
                    assignments,
                    original_index=original_index,
                    execution_index=original_index,
                    selected_parameter=parameter.name,
                )
                original_index += 1

    def _point(
        self,
        assignments: Mapping[str, int],
        *,
        original_index: int,
        execution_index: int,
        selected_parameter: str | None,
    ) -> SweepPoint:
        if self.task is SweepTask.I2C_SWEEP:
            assert self._baseline_i2c is not None
            requested_i2c = dict(self._baseline_i2c)
            requested_i2c.update(assignments)
            return SweepPoint(
                task=self.task,
                mode=self.mode,
                original_index=original_index,
                execution_index=execution_index,
                selected_parameter=selected_parameter,
                status=SweepPointStatus.READY,
                requested_i2c=requested_i2c,
            )

        assert self._baseline_fir is not None
        taps = list(self._baseline_fir.taps)
        for name, value in assignments.items():
            taps[_FIR_INDEX[name]] = value
        raw_taps = tuple(taps)
        reason = self._fir_constraint_reason(raw_taps)
        return SweepPoint(
            task=self.task,
            mode=self.mode,
            original_index=original_index,
            execution_index=execution_index,
            selected_parameter=selected_parameter,
            status=(
                SweepPointStatus.READY
                if reason is None
                else SweepPointStatus.SKIPPED_CONSTRAINT
            ),
            requested_fir=SweepFirTarget(
                taps=raw_taps,
                upper_nlc=self._baseline_fir.upper_nlc,
                lower_nlc=self._baseline_fir.lower_nlc,
            ),
            constraint_reason=reason,
        )

    def _configure_i2c(
        self,
        baseline_i2c: Mapping[str, int] | None,
        allowlist: tuple[str, ...],
    ) -> None:
        if not isinstance(baseline_i2c, Mapping) or not baseline_i2c:
            raise ValueError("I2C_SWEEP requires a non-empty baseline_i2c mapping")
        if not isinstance(allowlist, tuple) or not allowlist:
            raise ValueError("I2C_SWEEP requires a backend i2c_allowlist")
        if len(set(allowlist)) != len(allowlist) or any(
            not isinstance(name, str) or not name for name in allowlist
        ):
            raise ValueError("i2c_allowlist must contain unique nonblank names")
        selected_names = {parameter.name for parameter in self.space.parameters}
        if not selected_names <= set(allowlist):
            raise ValueError("selected I2C parameter is outside the backend allowlist")
        if not selected_names <= set(baseline_i2c):
            raise ValueError("baseline_i2c is missing a selected parameter")
        if not set(baseline_i2c) <= set(allowlist):
            raise ValueError("baseline_i2c contains a field outside the backend allowlist")
        copied: dict[str, int] = {}
        for name, value in baseline_i2c.items():
            if not isinstance(name, str) or not name:
                raise ValueError("baseline_i2c names must be nonblank strings")
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("baseline_i2c values must be integers")
            copied[name] = value
        self._baseline_i2c = copied

    def _configure_fir(
        self,
        baseline_fir: FirTarget | None,
        authorization: FirAuthorizationEnvelope | None,
    ) -> None:
        if not isinstance(baseline_fir, FirTarget):
            raise ValueError("BERT_FIR_SWEEP requires baseline_fir")
        if not isinstance(authorization, FirAuthorizationEnvelope):
            raise ValueError("BERT_FIR_SWEEP requires fir_authorization")
        selected_names = {parameter.name for parameter in self.space.parameters}
        if not selected_names <= set(FIR_TAP_NAMES):
            raise ValueError("BERT FIR sweep parameters must be one of seven FIR tap names")
        baseline_rejection = authorization.rejection_reason(baseline_fir, baseline_fir)
        if baseline_rejection is not None:
            raise ValueError(f"B0 FIR is outside authorization: {baseline_rejection}")
        self._baseline_fir = baseline_fir
        self._fir_authorization = authorization

    def _fir_constraint_reason(self, taps: tuple[int, ...]) -> str | None:
        assert self._baseline_fir is not None
        assert self._fir_authorization is not None
        if taps[FIR_MAIN_INDEX] <= 0:
            return "FIR Main must be positive"
        l1 = sum(abs(value) for value in taps)
        if l1 > FIR_L1_LIMIT:
            return f"FIR L1 sum {l1} exceeds limit {FIR_L1_LIMIT}"

        total_change = 0
        for index, (requested, baseline, limits, max_step) in enumerate(
            zip(
                taps,
                self._baseline_fir.taps,
                self._fir_authorization.tap_ranges,
                self._fir_authorization.max_step,
                strict=True,
            )
        ):
            minimum, maximum = limits
            if requested < minimum or requested > maximum:
                return f"FIR tap {index} is outside its authorized range"
            delta = abs(requested - baseline)
            if delta > max_step:
                return f"FIR tap {index} exceeds max_step"
            total_change += delta
        if total_change > self._fir_authorization.max_total_change_l1:
            return "FIR total change exceeds max_total_change_l1"
        return None
