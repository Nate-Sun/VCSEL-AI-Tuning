"""Strict wire contracts for the AI Tuning V1.1 API."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from vcsel_ai_tuning.v1_1.authorization import OptimizationDepth
from vcsel_ai_tuning.v1_1.domain import OptimizationScope, TaskType
from vcsel_ai_tuning.v1_1.live_ai_policy import (
    LiveAIProfile,
    validate_live_ai_payload,
)
from vcsel_ai_tuning.v1_1.sweep import SweepHardware, SweepMode


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"),
]


def _reject_observation_path_segments(value: str) -> str:
    if any(segment in {".", ".."} for segment in value.split(":")):
        raise ValueError("observation_id must not contain path traversal segments")
    return value


ObservationIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)*$",
    ),
    AfterValidator(_reject_observation_path_segments),
]


_RFC3339_AWARE_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _parse_rfc3339_aware_datetime(value: object) -> object:
    if type(value) is not str or _RFC3339_AWARE_DATETIME.fullmatch(value) is None:
        raise ValueError("expires_at must be a timezone-aware RFC3339 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "expires_at must be a timezone-aware RFC3339 string"
        ) from error


Rfc3339AwareDatetime = Annotated[
    AwareDatetime,
    BeforeValidator(_parse_rfc3339_aware_datetime),
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
WireOptimizationScope = Annotated[
    OptimizationScope,
    _enum_wire(OptimizationScope),
]
WireLiveAIProfile = Annotated[
    LiveAIProfile,
    _enum_wire(LiveAIProfile),
]
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
    dca_tdecq_equalizer_function: Literal[1]
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
    live_ai_profile: WireLiveAIProfile | None = None
    optimization_scope: WireOptimizationScope | None = None
    pilot_candidate_budget: StrictInt | None = Field(default=None, gt=0)
    fast_candidate_budget: StrictInt | None = Field(default=None, gt=0)
    coarse_candidate_budget: StrictInt | None = Field(default=None, gt=0)
    fine_candidate_budget: StrictInt | None = Field(default=None, ge=0)
    fir_first_candidate_budget: StrictInt | None = Field(default=None, gt=0)
    fir_iterative_candidate_budget: StrictInt | None = Field(default=None, gt=0)
    restore_after_each_candidate: StrictBool | None = None
    advanced_iteration_limit: StrictInt | None = Field(default=None, gt=0)
    sweep_hardware: WireSweepHardware | None = None
    sweep_mode: WireSweepMode | None = None
    seed: StrictInt = 0
    max_points: StrictInt | None = Field(default=None, gt=0)
    settle_time_s: FiniteFloat = Field(default=0.0, ge=0)
    save_dataset: StrictBool = False

    @model_serializer(mode="wrap")
    def serialize_staged_fields(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if self.live_ai_profile is None:
            for field in (
                "live_ai_profile",
                "optimization_scope",
                "pilot_candidate_budget",
                "fast_candidate_budget",
                "coarse_candidate_budget",
                "fine_candidate_budget",
                "fir_first_candidate_budget",
                "fir_iterative_candidate_budget",
                "restore_after_each_candidate",
            ):
                payload.pop(field, None)
        elif self.restore_after_each_candidate is None:
            payload.pop("restore_after_each_candidate", None)
        return payload

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
        staged_fields = (
            "live_ai_profile",
            "optimization_scope",
            "pilot_candidate_budget",
            "fast_candidate_budget",
            "coarse_candidate_budget",
            "fine_candidate_budget",
            "fir_first_candidate_budget",
            "fir_iterative_candidate_budget",
            "restore_after_each_candidate",
        )
        if self.live_ai_profile is None:
            supplied = [
                field
                for field in staged_fields
                if field != "live_ai_profile" and field in self.model_fields_set
            ]
            if supplied:
                raise ValueError(
                    f"{supplied[0]} requires live_ai_profile"
                )
            if self.task_type is not TaskType.SWEEP and self.settle_time_s != 0.0:
                raise ValueError("settle_time_s is only valid for SWEEP tasks")
            return self
        if self.task_type is not TaskType.AI_OPTIMIZATION:
            raise ValueError("live_ai_profile is only valid for AI_OPTIMIZATION")
        policy_payload = self.model_dump(mode="json")
        if self.restore_after_each_candidate is None:
            policy_payload.pop("restore_after_each_candidate", None)
        validate_live_ai_payload(self.live_ai_profile, policy_payload)
        return self


class SaveB0Request(StrictRequest):
    snapshot_token: Identifier
    operator_confirmed: Literal[True]


class SnapshotTokenRequest(StrictRequest):
    snapshot_token: Identifier


class ValidateParametersRequest(SnapshotTokenRequest):
    parameters: ParameterSelectionTuple


class AuthorizeTaskRequest(StrictRequest):
    authorization_id: Identifier
    expires_at: Rfc3339AwareDatetime
    operator_confirmed: Literal[True]


class RecoveryRequest(StrictRequest):
    action: Literal["RESUME", "ABORT_RECOVERED"]
    operator_confirmed: Literal[True]


class SweepCountRequest(StrictRequest):
    snapshot_token: Identifier
    hardware: WireSweepHardware
    mode: WireSweepMode
    parameters: ParameterSelectionTuple
    seed: StrictInt
    max_points: StrictInt = Field(gt=0)


class ApplyHistoryResultRequest(StrictRequest):
    source_task_id: Identifier
    observation_id: ObservationIdentifier


class HistoryRecoveryRequest(StrictRequest):
    operator_confirmed: Literal[True]


class AbortRequest(StrictRequest):
    reason: str = Field(min_length=1, max_length=1024)


class EmptyRequest(StrictRequest):
    pass
