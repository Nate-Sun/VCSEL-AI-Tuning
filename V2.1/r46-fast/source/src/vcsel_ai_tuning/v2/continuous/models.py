"""Strict immutable contracts for continuous eye optimization."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Mapping

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    PlainSerializer,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

from vcsel_ai_tuning.campaign.models import StageAuthorization
from vcsel_ai_tuning.v2.contracts import EyeSample, MeasurementStatus


NonBlankText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


def _mapping_input(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("mapping keys must be nonblank strings")
        result[key] = item
    return result


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _serialize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value)


ImmutableMapping = Annotated[
    Mapping[str, Any],
    BeforeValidator(_mapping_input),
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, Any]),
]


def _strict_int_mapping_input(value: object) -> dict[str, int]:
    raw = _mapping_input(value)
    for name, item in raw.items():
        if type(item) is not int:
            raise ValueError(f"{name} must be an integer")
    return raw


ImmutableIntMapping = Annotated[
    Mapping[str, StrictInt],
    BeforeValidator(_strict_int_mapping_input),
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, int]),
]


ImmutableMetricMapping = Annotated[
    Mapping[str, FiniteFloat | None],
    BeforeValidator(_mapping_input),
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float | None]),
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class ExecutionMode(StrEnum):
    FAKE = "FAKE"
    LIVE = "LIVE"


class ContinuousState(StrEnum):
    RUNNING = "RUNNING"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED_SAFE = "FAILED_SAFE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class ProviderActionKind(StrEnum):
    PROPOSE = "PROPOSE"
    HOLD = "HOLD"
    STOP = "STOP"
    REQUEST_HUMAN = "REQUEST_HUMAN"


class ReasonCode(StrEnum):
    TARGET_MET = "TARGET_MET"
    TARGET_NOT_MET = "TARGET_NOT_MET"
    EXPECTED_IMPROVEMENT = "EXPECTED_IMPROVEMENT"
    EXPLORATION = "EXPLORATION"
    LEARNED_NEGATIVE_RESULT = "LEARNED_NEGATIVE_RESULT"
    EXHAUSTED_SEARCH = "EXHAUSTED_SEARCH"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OOD = "OOD"
    INVALID_OBSERVATION = "INVALID_OBSERVATION"
    NEED_HUMAN = "NEED_HUMAN"


class HumanDecision(StrEnum):
    RESUME = "RESUME"
    ACCEPT = "ACCEPT"
    RESTORE_B0 = "RESTORE_B0"


class StoppingPolicy(FrozenModel):
    target_tdecq_db: FiniteFloat = Field(default=2.0, gt=0)
    acceptance_tdecq_delta_db: FiniteFloat = Field(default=0.2, ge=0)
    acceptance_outer_er_delta_db: FiniteFloat = Field(default=0.2, ge=0)


class ProviderAction(FrozenModel):
    action: ProviderActionKind
    requested_params: ImmutableIntMapping | None = None
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1, max_length=16)
    confidence: FiniteFloat = Field(ge=0, le=1)
    summary: NonBlankText = Field(max_length=512)

    @model_validator(mode="after")
    def validate_action_shape(self) -> "ProviderAction":
        if self.action is ProviderActionKind.PROPOSE:
            if not self.requested_params:
                raise ValueError("PROPOSE requires requested_params")
        elif self.requested_params is not None:
            raise ValueError(f"{self.action.value} forbids requested_params")
        return self


class ContinuousPreflightRequest(FrozenModel):
    """Provider-preview input with no authority to mutate the DUT."""

    campaign_id: NonBlankText
    initial_eye: EyeSample
    policy: ImmutableMapping
    max_rounds: StrictInt = Field(gt=0)
    stopping_policy: StoppingPolicy = Field(default_factory=StoppingPolicy)

    @model_validator(mode="after")
    def validate_campaign_binding(self) -> "ContinuousPreflightRequest":
        if self.initial_eye.campaign_id != self.campaign_id:
            raise ValueError("initial eye campaign does not match request campaign")
        return self


class ContinuousRequest(FrozenModel):
    campaign_id: NonBlankText
    initial_eye: EyeSample
    authorization: StageAuthorization
    policy: ImmutableMapping
    max_rounds: StrictInt = Field(gt=0)
    provider_config: ImmutableMapping
    stopping_policy: StoppingPolicy = Field(default_factory=StoppingPolicy)
    b0_path: Path
    b0_i2c: ImmutableIntMapping = Field(default_factory=dict)
    result_dir: Path
    execution_mode: ExecutionMode
    human_timeout_s: FiniteFloat = Field(default=900.0, gt=0)

    @model_validator(mode="after")
    def validate_campaign_binding(self) -> "ContinuousRequest":
        if self.initial_eye.campaign_id != self.campaign_id:
            raise ValueError("initial eye campaign does not match request campaign")
        if self.authorization.campaign_id != self.campaign_id:
            raise ValueError("authorization campaign does not match request campaign")
        return self


class RoundRecord(FrozenModel):
    campaign_id: NonBlankText
    round_index: StrictInt = Field(gt=0)
    proposal_attempt: StrictInt = Field(gt=0)
    before_observation_id: NonBlankText
    after_observation_id: NonBlankText
    requested_params: ImmutableIntMapping
    applied_params: ImmutableIntMapping
    readback_params: ImmutableIntMapping
    before_metrics: ImmutableMetricMapping
    after_metrics: ImmutableMetricMapping
    measurement_status: MeasurementStatus = MeasurementStatus.VALID
    started_at: AwareDatetime
    finished_at: AwareDatetime
    evidence_persisted: StrictBool
    stage_timings_s: ImmutableMetricMapping = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_round_evidence(self) -> "RoundRecord":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.before_observation_id == self.after_observation_id:
            raise ValueError("before and after observation IDs must differ")
        return self


class ExecutionOutcome(FrozenModel):
    sample: EyeSample
    requested_params: ImmutableIntMapping
    applied_params: ImmutableIntMapping
    readback_params: ImmutableIntMapping
    adapter_identities: ImmutableMapping = Field(default_factory=dict)
    evidence_complete: StrictBool

    @model_validator(mode="after")
    def validate_evidence(self) -> "ExecutionOutcome":
        if self.requested_params != self.applied_params:
            raise ValueError("applied_params must exactly match requested_params")
        if any(
            self.readback_params.get(name) != value
            for name, value in self.applied_params.items()
        ):
            raise ValueError("readback_params must exactly match applied values")
        sample_readback = {
            name: int(value) if float(value).is_integer() else value
            for name, value in self.sample.readback_params.items()
        }
        if sample_readback != dict(self.readback_params):
            raise ValueError("sample readback must match execution readback")
        return self


class CampaignResult(FrozenModel):
    campaign_id: NonBlankText
    state: ContinuousState
    execution_mode: ExecutionMode
    completed_rounds: StrictInt = Field(ge=0)
    target_tdecq_db: FiniteFloat = Field(gt=0)
    target_reached: StrictBool
    acceptance_passed: StrictBool
    closed_loop_proven: StrictBool
    stop_reason: NonBlankText | None = None
    provider_error_type: NonBlankText | None = None
    provider_error_message: NonBlankText | None = None
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    adapter_identities: ImmutableMapping = Field(default_factory=dict)
    b0_restored: StrictBool = False
    evidence_complete: StrictBool = False
    round_records: tuple[RoundRecord, ...] = ()

    @model_validator(mode="after")
    def validate_result_state(self) -> "CampaignResult":
        terminal = self.state in {
            ContinuousState.COMPLETED,
            ContinuousState.FAILED_SAFE,
            ContinuousState.RECOVERY_REQUIRED,
        }
        if terminal != (self.finished_at is not None):
            raise ValueError("finished_at must be present exactly for terminal states")
        if terminal and self.stop_reason is None:
            raise ValueError("terminal result requires stop_reason")
        if self.completed_rounds != len(self.round_records) and self.round_records:
            raise ValueError("completed_rounds must match round_records")
        if self.closed_loop_proven:
            if self.execution_mode is ExecutionMode.FAKE:
                raise ValueError("FAKE execution cannot set closed_loop_proven")
            if (
                self.completed_rounds < 1
                or not self.adapter_identities
                or not self.b0_restored
                or not self.evidence_complete
            ):
                raise ValueError("closed_loop_proven requires complete LIVE evidence")
        return self
