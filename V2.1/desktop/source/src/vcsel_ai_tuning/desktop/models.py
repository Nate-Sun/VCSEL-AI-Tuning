"""Hardware-neutral contracts used by the desktop application."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    SecretStr,
    StrictInt,
    StringConstraints,
    field_validator,
)


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
    )


class B0Input(FrozenModel):
    ibias_dac: StrictInt = Field(ge=0)
    vref: StrictInt = Field(ge=0)
    vbias_ref: StrictInt = Field(ge=0)
    vga_cur: StrictInt = Field(ge=0)
    res_gain: StrictInt = Field(ge=0)
    res_bw: StrictInt = Field(ge=0)
    gain_ctrl: StrictInt = Field(ge=0)
    efin_cur: StrictInt = Field(ge=0)
    eq: StrictInt = Field(ge=0)


DEFAULT_B0 = B0Input(
    ibias_dac=60,
    vref=15,
    vbias_ref=15,
    vga_cur=13,
    res_gain=6,
    res_bw=0,
    gain_ctrl=2,
    efin_cur=15,
    eq=16,
)


class BenchSettings(FrozenModel):
    dca_visa_resource: NonBlank
    i2c_com_port: NonBlank
    i2c_address: StrictInt = Field(ge=0, le=0xFF)


class RunInputs(FrozenModel):
    api_key: SecretStr
    max_rounds: StrictInt = Field(gt=0)
    target_tdecq_db: FiniteFloat = Field(gt=0)
    b0: B0Input
    bench: BenchSettings

    @field_validator("api_key")
    @classmethod
    def require_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("api_key must not be blank")
        return value


class DesktopPhase(StrEnum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class PreflightSummary(FrozenModel):
    campaign_id: NonBlank
    dca_verified: bool
    i2c_readback_verified: bool
    kimi_proposal_valid: bool
    dut_write_performed: bool
    bert_contacted: bool
    details: str = ""

    @property
    def approvable(self) -> bool:
        return (
            self.dca_verified
            and self.i2c_readback_verified
            and self.kimi_proposal_valid
            and not self.dut_write_performed
            and not self.bert_contacted
        )


class ProgressSnapshot(FrozenModel):
    phase: DesktopPhase
    current_round: StrictInt = Field(ge=0)
    max_rounds: StrictInt = Field(gt=0)


class DesktopResult(FrozenModel):
    campaign_id: NonBlank
    phase: DesktopPhase
    campaign_root: Path
    b0_restored: bool
    message: str = ""
