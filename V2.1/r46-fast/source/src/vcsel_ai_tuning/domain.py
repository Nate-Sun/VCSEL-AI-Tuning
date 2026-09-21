from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Fidelity(StrEnum):
    SIMULATION_PROXY = "simulation_proxy"
    HARDWARE_FAST = "hardware_fast"
    HARDWARE_STANDARD = "hardware_standard"


class RunState(StrEnum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    ARMED = "ARMED"
    APPLY = "APPLY"
    READBACK = "READBACK"
    SETTLE = "SETTLE"
    MEASURE_FAST = "MEASURE_FAST"
    VALIDATE = "VALIDATE"
    MEASURE_SLOW = "MEASURE_SLOW"
    COMMIT = "COMMIT"
    COMPLETE = "COMPLETE"
    ROLLBACK = "ROLLBACK"
    SAFE_OFF = "SAFE_OFF"
    ABORTED = "ABORTED"


class ParameterSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str
    unit: str
    lower: float
    upper: float
    quantization: float | None = Field(default=None, gt=0)
    max_step: float | None = Field(default=None, gt=0)
    readback_tolerance: float = Field(default=0.0, ge=0)
    grid_values: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_interval_and_grid(self) -> "ParameterSpec":
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")
        if any(value < self.lower or value > self.upper for value in self.grid_values):
            raise ValueError("grid value outside parameter bounds")
        return self

    def prepare(self, value: float) -> float:
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{self.name} must be finite")
        if value < self.lower or value > self.upper:
            raise ValueError(
                f"{self.name}={value} is outside [{self.lower}, {self.upper}] {self.unit}"
            )
        if self.quantization is None:
            return value
        lower = Decimal(str(self.lower))
        upper = Decimal(str(self.upper))
        quantum = Decimal(str(self.quantization))
        steps = round((Decimal(str(value)) - lower) / quantum)
        prepared = lower + steps * quantum
        return float(min(upper, max(lower, prepared)))


ContextValue = str | int | float | bool


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    requested: dict[str, float]
    context: dict[str, ContextValue] = Field(default_factory=dict)


class AppliedSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    requested: dict[str, float]
    applied: dict[str, float]
    readback: dict[str, float]


class MetricValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    value: float | None
    unit: str
    fidelity: Fidelity
    method_version: str
    uncertainty: float | None = Field(default=None, ge=0)
    valid: bool = True
    status: str | None = None
    invalid_reason: str | None = None
    measurement_definition_id: str | None = None

    @model_validator(mode="after")
    def require_standard_measurement_identity(self) -> "MetricValue":
        if self.valid and self.value is None:
            raise ValueError("valid metric requires a numeric value")
        if (
            self.fidelity is Fidelity.HARDWARE_STANDARD
            and not self.measurement_definition_id
        ):
            raise ValueError(
                "hardware_standard metric requires measurement_definition_id"
            )
        return self


class ConstraintResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str
    passed: bool
    value: float | None = None
    limit: str | None = None
    reason: str | None = None


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    relative_path: Path
    sha256: str
    media_type: str


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    state: RunState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: dict[str, Any] = Field(default_factory=dict)


class BerEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    error_count: int = Field(ge=0)
    bit_count: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)
    upper_bound: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "BerEvidence":
        if self.error_count > self.bit_count:
            raise ValueError("error_count cannot exceed bit_count")
        if self.error_count == 0 and self.upper_bound <= 0:
            raise ValueError("zero-error evidence requires a positive upper confidence bound")
        return self


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    state: RunState
    settings: AppliedSettings
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    constraints: list[ConstraintResult] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    ber_evidence: BerEvidence | None = None
    valid: bool
    failure_reason: str | None = None
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def validate_terminal_status(self) -> "Observation":
        if self.valid and self.state is not RunState.COMPLETE:
            raise ValueError("valid observation must finish in COMPLETE")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid observation requires failure_reason")
        for name, metric in self.metrics.items():
            if name == "tdecq_db" and metric.fidelity is not Fidelity.HARDWARE_STANDARD:
                raise ValueError("tdecq_db requires hardware_standard fidelity")
            if name == "tdecq_db" and not metric.valid:
                if metric.value is not None:
                    raise ValueError("invalid tdecq_db must keep value=None")
                if metric.status is None or not metric.status.strip():
                    raise ValueError("invalid tdecq_db requires status")
                if (
                    metric.invalid_reason is None
                    or not metric.invalid_reason.strip()
                ):
                    raise ValueError("invalid tdecq_db requires invalid_reason")
            if name == "tdecq_proxy" and metric.fidelity is Fidelity.HARDWARE_STANDARD:
                raise ValueError("tdecq_proxy cannot use hardware_standard fidelity")
        return self

    @property
    def feasible(self) -> bool:
        return self.valid and all(result.passed for result in self.constraints)


class BackendResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metrics: dict[str, MetricValue]
    raw_artifacts: dict[str, bytes] = Field(default_factory=dict)
    status: dict[str, ContextValue] = Field(default_factory=dict)
