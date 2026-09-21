"""Read-only provider boundary for continuous optimization decisions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, FiniteFloat, StrictInt, field_validator, model_validator

from vcsel_ai_tuning.v2.contracts import EyeSample, MeasurementStatus

from .models import (
    FrozenModel,
    ImmutableIntMapping,
    ImmutableMapping,
    ProviderAction,
    ReasonCode,
)
from .policy import I2cParameterPolicy


_FORBIDDEN_KEYS = {
    "api_key",
    "apply",
    "authorization",
    "bert",
    "fir",
    "i2c",
    "moonshot_api_key",
    "nlc",
    "serial",
    "setfir",
    "visa",
}


def _validate_safe_data(value: object, *, path: str = "context") -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Path) or callable(value):
        raise ValueError(f"{path} contains forbidden executable or secret value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).replace("-", "_").lower()
            if normalized in _FORBIDDEN_KEYS or normalized.startswith("bearer"):
                raise ValueError(f"{path}.{key} is forbidden provider data")
            _validate_safe_data(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_safe_data(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains value outside safe provider data")


class ProviderContext(FrozenModel):
    sample: EyeSample
    verified_readback: ImmutableIntMapping
    policy: I2cParameterPolicy
    hardware_round: StrictInt = Field(ge=1)
    max_rounds: StrictInt = Field(gt=0)
    target_tdecq_db: FiniteFloat
    recent_valid_summaries: tuple[ImmutableMapping, ...] = Field(
        default=(),
        max_length=5,
    )
    best_reference: EyeSample | None = None
    allowed_output_schema: ImmutableMapping
    correction_error_code: str | None = None

    @field_validator(
        "recent_valid_summaries",
        "allowed_output_schema",
        mode="after",
    )
    @classmethod
    def validate_safe_provider_values(cls, value: Any) -> Any:
        _validate_safe_data(value)
        return value

    @model_validator(mode="after")
    def validate_round_bounds(self) -> "ProviderContext":
        if self.hardware_round > self.max_rounds:
            raise ValueError("hardware_round must not exceed max_rounds")
        reference = self.best_reference
        if reference is not None:
            if reference.campaign_id != self.sample.campaign_id:
                raise ValueError("best_reference must belong to the current campaign")
            if reference.measurement_status is not MeasurementStatus.VALID:
                raise ValueError("best_reference must be a valid measurement")
            if reference.measurement_id == self.sample.measurement_id:
                raise ValueError("best_reference must be a prior measurement")
            if reference.captured_at > self.sample.captured_at:
                raise ValueError("best_reference must not postdate the current sample")
        return self

    @property
    def policy_version(self) -> str:
        return self.policy.version

    @property
    def remaining_rounds(self) -> int:
        return self.max_rounds - self.hardware_round + 1


@runtime_checkable
class ContinuousDecisionProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def decide(self, context: ProviderContext) -> ProviderAction: ...


__all__ = [
    "ContinuousDecisionProvider",
    "ProviderContext",
    "ReasonCode",
]
