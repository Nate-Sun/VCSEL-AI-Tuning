from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .domain import ParameterSpec


_NonBlankIdentity = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["fake", "replay", "cadence", "hardware"]
    dry_run: bool = True
    options: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def forbid_live_hardware(self) -> "BackendConfig":
        if self.kind == "hardware" and not self.dry_run:
            raise ValueError("Milestone 1 hardware backend must remain dry-run")
        return self


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal[
        "grid",
        "random",
        "sobol",
        "PATH_EXHAUSTIVE",
        "PATH_TDECQ_BAYESIAN",
    ]
    budget: int = Field(gt=0)
    seed: int = 0


class MetricConstraintSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metric: str
    operator: Literal[">=", "<="]
    threshold: float


class TieBreakerSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metric: str
    direction: Literal["minimize", "maximize"]


class ObjectiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    primary_metric: str
    direction: Literal["minimize", "maximize"]
    metric_constraints: tuple[MetricConstraintSpec, ...] = ()
    tie_breakers: tuple[TieBreakerSpec, ...] = ()


class TuningIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    driver_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vcsel_model: _NonBlankIdentity
    device_id: _NonBlankIdentity
    test_context_id: _NonBlankIdentity


class CampaignConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    backend: BackendConfig
    optimizer: OptimizerConfig
    parameters: tuple[ParameterSpec, ...] = Field(min_length=1)
    safe_baseline: dict[str, float]
    objective: ObjectiveConfig
    slow_measurement_for_best: bool = False
    measurement_policy: Literal[
        "fast_proxy",
        "standard_best_only",
        "standard_every_candidate",
    ] = "fast_proxy"
    tuning_identity: TuningIdentity | None = None

    @model_validator(mode="after")
    def validate_parameters_and_baseline(self) -> "CampaignConfig":
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if set(self.safe_baseline) != set(names):
            raise ValueError("safe_baseline must define every parameter exactly once")
        for parameter in self.parameters:
            parameter.prepare(self.safe_baseline[parameter.name])

        dual_path = self.optimizer.kind in {
            "PATH_EXHAUSTIVE",
            "PATH_TDECQ_BAYESIAN",
        }
        if dual_path:
            if self.tuning_identity is None:
                raise ValueError("dual-path optimizer requires tuning_identity")
            if self.measurement_policy != "standard_every_candidate":
                raise ValueError(
                    "dual-path optimizer requires standard_every_candidate"
                )
            if (
                self.objective.primary_metric != "tdecq_db"
                or self.objective.direction != "minimize"
                or self.objective.metric_constraints
                or self.objective.tie_breakers
            ):
                raise ValueError(
                    "dual-path optimizer requires a TDECQ-only minimize objective"
                )

            endpoint_count = 1
            for parameter in self.parameters:
                if not parameter.grid_values:
                    raise ValueError(
                        "dual-path optimizer requires finite grid_values"
                    )
                prepared_values: set[int] = set()
                for raw_value in parameter.grid_values:
                    prepared = parameter.prepare(raw_value)
                    integer = int(prepared)
                    if float(integer) != prepared:
                        raise ValueError(
                            "dual-path optimizer requires integer grid values"
                        )
                    prepared_values.add(integer)
                endpoint_count *= len(prepared_values)

            if (
                self.optimizer.kind == "PATH_EXHAUSTIVE"
                and self.optimizer.budget < endpoint_count
            ):
                raise ValueError(
                    "exhaustive budget must cover the full unfiltered grid"
                )
        return self

    @property
    def parameter_map(self) -> dict[str, ParameterSpec]:
        return {parameter.name: parameter for parameter in self.parameters}


class SourceArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    manifest_id: str
    device_id: str
    temperature_c: float
    model_path: Path | None = None
    external_network_path: Path | None = None
    sources: tuple[SourceArtifact, ...] = ()
    model_parameters: dict[str, float] = Field(default_factory=dict)


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_model(path: Path, model_type: type[ModelT]) -> ModelT:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return model_type.model_validate(data)


def sha256_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_manifest_files(manifest: CalibrationManifest, base_dir: Path) -> None:
    for source in manifest.sources:
        resolved = (base_dir / source.path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual = sha256_file(resolved)
        if actual != source.sha256:
            raise ValueError(
                f"hash mismatch for {source.path}: expected {source.sha256}, got {actual}"
            )
