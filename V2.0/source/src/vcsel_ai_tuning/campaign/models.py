from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PlainSerializer,
    Strict,
    StrictInt,
    model_validator,
)


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
RegisterValue = Annotated[int, Strict(), Field(ge=0)]
MetricValue = FiniteFloat | None
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _serialize_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return dict(value)


ImmutableI2cMapping = Annotated[
    Mapping[str, StrictInt],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, int]),
]
ImmutableMetricMapping = Annotated[
    Mapping[str, MetricValue],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float | None]),
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        validate_default=True,
    )


class AuthorizationMode(StrEnum):
    PER_STAGE = "PER_STAGE"
    PER_CANDIDATE = "PER_CANDIDATE"
    FULL_CAMPAIGN = "FULL_CAMPAIGN"


class CampaignState(StrEnum):
    DRAFT = "DRAFT"
    PREFLIGHT = "PREFLIGHT"
    AWAIT_BASELINE_APPROVAL = "AWAIT_BASELINE_APPROVAL"
    RESTORE_B0 = "RESTORE_B0"
    BASELINE_MEASUREMENT = "BASELINE_MEASUREMENT"
    AWAIT_COARSE_APPROVAL = "AWAIT_COARSE_APPROVAL"
    I2C_SCREENING = "I2C_SCREENING"
    I2C_COARSE = "I2C_COARSE"
    AWAIT_FINE_APPROVAL = "AWAIT_FINE_APPROVAL"
    I2C_FINE = "I2C_FINE"
    AWAIT_FIR_APPROVAL = "AWAIT_FIR_APPROVAL"
    FIR_APPLY = "FIR_APPLY"
    I2C_RETUNE = "I2C_RETUNE"
    FINAL_VERIFICATION = "FINAL_VERIFICATION"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    ABORTING = "ABORTING"
    ROLLING_BACK = "ROLLING_BACK"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class FirSettings(FrozenModel):
    taps: tuple[StrictInt, ...] = Field(min_length=7, max_length=7)
    upper_nlc: RegisterValue
    lower_nlc: RegisterValue

    @model_validator(mode="after")
    def validate_main_and_l1(self) -> "FirSettings":
        if self.taps[3] <= 0:
            raise ValueError("FIR main tap must be positive")
        if self.l1 > 170:
            raise ValueError("FIR L1 must be at most 170")
        return self

    @property
    def l1(self) -> int:
        return sum(abs(value) for value in self.taps)


class FirAuthorizationSpec(FrozenModel):
    """Explicit FIR safety envelope frozen into a stage authorization."""

    tap_min: tuple[StrictInt, ...] = Field(min_length=7, max_length=7)
    tap_max: tuple[StrictInt, ...] = Field(min_length=7, max_length=7)
    max_step: tuple[StrictInt, ...] = Field(min_length=7, max_length=7)
    max_total_change_l1: StrictInt = Field(gt=0)
    fixed_upper_nlc: RegisterValue
    fixed_lower_nlc: RegisterValue

    @model_validator(mode="after")
    def validate_envelope(self) -> "FirAuthorizationSpec":
        for index, (minimum, maximum) in enumerate(
            zip(self.tap_min, self.tap_max, strict=True)
        ):
            if minimum > maximum:
                raise ValueError(
                    f"tap_min[{index}] must not exceed tap_max[{index}]"
                )
        if self.tap_min[3] <= 0:
            raise ValueError("FIR main tap authorization range must be positive")
        if any(step <= 0 for step in self.max_step):
            raise ValueError("FIR max_step values must all be positive")
        return self


class I2cSettings(FrozenModel):
    ibias_dac: RegisterValue
    vref: RegisterValue
    vbias_ref: RegisterValue
    vga_cur: RegisterValue
    res_gain: RegisterValue
    res_bw: RegisterValue
    gain_ctrl: RegisterValue
    efin_cur: RegisterValue
    eq: RegisterValue


class BaselineConfig(FrozenModel):
    baseline_id: Identifier
    version: StrictInt = Field(ge=1)
    pattern: Literal["SSPRQ"]
    fir: FirSettings
    i2c: I2cSettings


class ParameterSpec(FrozenModel):
    name: Identifier
    encoding_min: StrictInt
    encoding_max: StrictInt
    tuning_min: StrictInt
    tuning_max: StrictInt
    max_step: StrictInt = Field(gt=0)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_ranges(self) -> "ParameterSpec":
        if self.encoding_min >= self.encoding_max:
            raise ValueError("encoding_min must be less than encoding_max")
        if self.tuning_min > self.tuning_max:
            raise ValueError("tuning_min must not exceed tuning_max")
        if not (
            self.encoding_min <= self.tuning_min <= self.encoding_max
            and self.encoding_min <= self.tuning_max <= self.encoding_max
        ):
            raise ValueError("tuning range must be inside encoding range")
        return self


class Candidate(FrozenModel):
    candidate_id: Identifier
    stage: CampaignState
    requested_i2c: ImmutableI2cMapping = Field(default_factory=dict)
    requested_fir: FirSettings | None = None
    predicted_metrics: ImmutableMetricMapping = Field(default_factory=dict)
    model_version: str
    reason: str


class AppliedSettings(FrozenModel):
    requested_i2c: ImmutableI2cMapping = Field(default_factory=dict)
    quantized_i2c: ImmutableI2cMapping = Field(default_factory=dict)
    applied_i2c: ImmutableI2cMapping = Field(default_factory=dict)
    readback_i2c: ImmutableI2cMapping = Field(default_factory=dict)
    requested_fir: FirSettings | None = None
    applied_fir: FirSettings | None = None
    readback_fir: FirSettings | None = None


class Observation(FrozenModel):
    observation_id: Identifier
    campaign_id: Identifier
    candidate_id: Identifier | None
    stage: CampaignState
    settings: AppliedSettings
    metrics: ImmutableMetricMapping = Field(default_factory=dict)
    valid: bool
    feasible: bool
    elapsed_s: FiniteFloat = Field(ge=0)
    artifacts: tuple[str, ...] = ()
    measured_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class StageAuthorization(FrozenModel):
    authorization_id: Identifier
    campaign_id: Identifier
    mode: AuthorizationMode
    stage: CampaignState
    baseline_hash: Sha256
    rollback_hash: Sha256
    parameter_specs: tuple[ParameterSpec, ...] = ()
    fir_spec: FirAuthorizationSpec | None = None
    max_candidates: StrictInt = Field(gt=0)
    max_duration_s: FiniteFloat = Field(gt=0)
    issued_at: AwareDatetime
    expires_at: AwareDatetime
    rollback_snapshot: AppliedSettings

    @model_validator(mode="after")
    def validate_expiry(self) -> "StageAuthorization":
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


class StageSummary(FrozenModel):
    stage: CampaignState
    convergence_reason: str
    best_observation: Observation | None = None
    candidates_used: StrictInt = Field(ge=0)
    elapsed_s: FiniteFloat = Field(ge=0)
    warnings: tuple[str, ...] = ()


FinalOutcome = Literal[
    "SUCCESS",
    "PARTIAL_CONVERGENCE",
    "NO_IMPROVEMENT",
    "FAILED",
    "RECOVERY_REQUIRED",
]


class Campaign(FrozenModel):
    campaign_id: Identifier
    baseline_id: Identifier
    baseline_hash: Sha256
    config_hash: Sha256
    authorization_mode: AuthorizationMode = AuthorizationMode.PER_STAGE
    state: CampaignState = CampaignState.DRAFT
    stage_summaries: tuple[StageSummary, ...] = ()
    result: FinalOutcome | None = None
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Campaign":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


_DEFAULT_B0_PATH = Path(__file__).resolve().parents[3] / "configs" / "ai_tuning" / "b0.yaml"


def load_b0(path: str | Path | None = None) -> BaselineConfig:
    config_path = Path(path) if path is not None else _DEFAULT_B0_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    fir_data = data.get("fir")
    if isinstance(fir_data, dict) and isinstance(fir_data.get("taps"), list):
        data = dict(data)
        data["fir"] = {**fir_data, "taps": tuple(fir_data["taps"])}
    return BaselineConfig.model_validate(data)
