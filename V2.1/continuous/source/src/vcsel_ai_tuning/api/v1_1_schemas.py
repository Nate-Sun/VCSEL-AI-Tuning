"""Strict wire contracts for the AI Tuning V1.1 API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)

from vcsel_ai_tuning.v1_1.authorization import OptimizationDepth
from vcsel_ai_tuning.v1_1.domain import TaskType
from vcsel_ai_tuning.v1_1.sweep import SweepHardware, SweepMode


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"),
]


def _enum_wire(enum_type):
    def normalize(value: object) -> object:
        if isinstance(value, enum_type):
            return value
        if type(value) is str:
            return enum_type(value)
        return value

    return BeforeValidator(normalize)


WireTaskType = Annotated[TaskType, _enum_wire(TaskType)]
WireDepth = Annotated[OptimizationDepth, _enum_wire(OptimizationDepth)]
WireSweepHardware = Annotated[SweepHardware, _enum_wire(SweepHardware)]
WireSweepMode = Annotated[SweepMode, _enum_wire(SweepMode)]


class StrictRequest(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class ChannelMappingRequest(StrictRequest):
    i2c_channel: StrictInt = Field(ge=0, le=3)
    bert_port: str = Field(min_length=1, max_length=64)
    bert_channel: str = Field(min_length=1, max_length=64)
    dca_tdecq_source: Literal["FUNC1"]
    dca_outer_er_source: Literal["CHAN1A"]
    operator_confirmed: Literal[True]


class ParameterSelectionRequest(StrictRequest):
    name: str = Field(min_length=1, max_length=64)
    min: StrictInt
    max: StrictInt
    step: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "ParameterSelectionRequest":
        if self.min > self.max:
            raise ValueError("min must not exceed max")
        return self


def _json_array_to_tuple(value: object) -> object:
    """Normalize only JSON/Python array containers for strict tuple fields."""

    if type(value) in {list, tuple}:
        return tuple(value)
    return value


ParameterSelectionTuple = Annotated[
    tuple[ParameterSelectionRequest, ...],
    BeforeValidator(_json_array_to_tuple),
]


class CreateTaskRequest(StrictRequest):
    task_id: Identifier
    task_type: WireTaskType
    channel_mapping: ChannelMappingRequest
    parameters: ParameterSelectionTuple
    depth: WireDepth = OptimizationDepth.STANDARD
    candidate_budget: StrictInt = Field(gt=0)
    advanced_iteration_limit: StrictInt | None = Field(default=None, gt=0)
    sweep_hardware: WireSweepHardware | None = None
    sweep_mode: WireSweepMode | None = None
    seed: StrictInt = 0
    max_points: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_task_specific_fields(self) -> "CreateTaskRequest":
        if self.depth is OptimizationDepth.ADVANCED:
            if self.advanced_iteration_limit is None:
                raise ValueError(
                    "ADVANCED requires advanced_iteration_limit"
                )
        elif self.advanced_iteration_limit is not None:
            raise ValueError(
                "advanced_iteration_limit is only valid for ADVANCED"
            )
        if self.task_type is TaskType.SWEEP:
            if (
                self.sweep_hardware is None
                or self.sweep_mode is None
                or self.max_points is None
            ):
                raise ValueError(
                    "SWEEP requires sweep_hardware, sweep_mode, and max_points"
                )
        elif any(
            value is not None
            for value in (self.sweep_hardware, self.sweep_mode, self.max_points)
        ):
            raise ValueError("sweep fields are only valid for SWEEP tasks")
        return self


class SaveB0Request(StrictRequest):
    snapshot_token: Identifier
    operator_confirmed: Literal[True]


class AuthorizeTaskRequest(StrictRequest):
    authorization_id: Identifier
    expires_at: AwareDatetime
    operator_confirmed: Literal[True]


class RecoveryRequest(StrictRequest):
    action: Literal["RESUME", "ABORT_RECOVERED"]
    operator_confirmed: Literal[True]


class SweepCountRequest(StrictRequest):
    hardware: WireSweepHardware
    mode: WireSweepMode
    parameters: ParameterSelectionTuple
    seed: StrictInt
    max_points: StrictInt = Field(gt=0)


class ApplyHistoryResultRequest(StrictRequest):
    source_task_id: Identifier
    observation_id: Identifier


class HistoryRecoveryRequest(StrictRequest):
    operator_confirmed: Literal[True]


class AbortRequest(StrictRequest):
    reason: str = Field(min_length=1, max_length=1024)


class EmptyRequest(StrictRequest):
    pass
