"""Strict, hardware-neutral startup configuration for the API runtime."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)


class RuntimeMode(StrEnum):
    FAKE = "FAKE"
    READ_ONLY = "READ_ONLY"
    LIVE_TUNING = "LIVE_TUNING"


def _runtime_mode_from_wire(value: object) -> object:
    if isinstance(value, RuntimeMode):
        return value
    if type(value) is str:
        return RuntimeMode(value)
    return value


def _path_from_wire(value: object) -> object:
    if isinstance(value, Path):
        return value
    if type(value) is str:
        return Path(value)
    return value


def _resource_ids_from_wire(value: object) -> object:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


RuntimeModeValue = Annotated[
    RuntimeMode,
    BeforeValidator(_runtime_mode_from_wire),
]
ConfigPath = Annotated[Path, BeforeValidator(_path_from_wire)]
NonBlank = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ResourceIds = Annotated[
    tuple[NonBlank, ...],
    BeforeValidator(_resource_ids_from_wire),
]


class StrictRuntimeModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
    )


class ApiConfig(StrictRuntimeModel):
    host: NonBlank = "127.0.0.1"
    port: StrictInt = Field(default=8000, ge=1, le=65535)


class StorageConfig(StrictRuntimeModel):
    sqlite_path: ConfigPath = Path("state/ai_tuning/campaigns.sqlite3")
    artifact_root: ConfigPath = Path("artifacts/ai_tuning")


class HardwareConfig(StrictRuntimeModel):
    driver_profile: ConfigPath
    dca_visa_resource: NonBlank
    i2c_com_port: NonBlank
    i2c_address: StrictInt = Field(ge=0, le=0xFF)
    bert_host: NonBlank
    bert_port: StrictInt = Field(ge=1, le=65535)
    resource_ids: ResourceIds = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_resource_ids(self) -> "HardwareConfig":
        if len(self.resource_ids) != len(set(self.resource_ids)):
            raise ValueError("hardware resource_ids must be unique")
        return self


class RuntimeConfig(StrictRuntimeModel):
    mode: RuntimeModeValue = RuntimeMode.FAKE
    api: ApiConfig = Field(default_factory=ApiConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    b0_path: ConfigPath = Path("configs/ai_tuning/b0.yaml")
    hardware: HardwareConfig | None = None
    measurement_timeout_s: FiniteFloat = Field(default=30.0, gt=0)
    worker_count: StrictInt = Field(default=2, ge=1, le=32)

    @model_validator(mode="after")
    def require_explicit_hardware_configuration(self) -> "RuntimeConfig":
        if self.mode is not RuntimeMode.FAKE and self.hardware is None:
            raise ValueError(
                f"hardware configuration is required for {self.mode.value}"
            )
        return self


def _resolve_path(base: Path, value: Path) -> Path:
    return (value if value.is_absolute() else base / value).resolve()


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load strict YAML and resolve every filesystem path beside the config."""

    config_path = Path(path)
    data: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = RuntimeConfig.model_validate(data)
    base = config_path.resolve().parent
    storage = config.storage.model_copy(
        update={
            "sqlite_path": _resolve_path(base, config.storage.sqlite_path),
            "artifact_root": _resolve_path(base, config.storage.artifact_root),
        }
    )
    hardware = config.hardware
    if hardware is not None:
        hardware = hardware.model_copy(
            update={
                "driver_profile": _resolve_path(base, hardware.driver_profile),
            }
        )
    return config.model_copy(
        update={
            "storage": storage,
            "b0_path": _resolve_path(base, config.b0_path),
            "hardware": hardware,
        }
    )
