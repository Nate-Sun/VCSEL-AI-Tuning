"""Strict, hardware-neutral data contracts for V2 optimization brains."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Annotated
from unicodedata import category

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
    model_validator,
)

from vcsel_ai_tuning.optimization.protocol import TuningObservation


_WINDOWS_RESERVED_DEVICE_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {
        f"{prefix}{number}"
        for prefix in ("com", "lpt")
        for number in range(1, 10)
    }
    | {
        f"{prefix}{number}"
        for prefix in ("com", "lpt")
        for number in ("¹", "²", "³")
    }
)


def _require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _validate_relative_posix_path(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError("must be a canonical relative POSIX path")
    segments = value.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ValueError("must be a canonical relative POSIX path")
    if ":" in value:
        raise ValueError("must not contain a colon")
    for segment in segments:
        if segment.endswith((".", " ")):
            raise ValueError("path segments must not end with a dot or space")
        device_basename = segment.split(".", maxsplit=1)[0].casefold()
        if device_basename in _WINDOWS_RESERVED_DEVICE_BASENAMES:
            raise ValueError("path segments must not use Windows device names")
    return value


def _validate_mapping_name(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("mapping names must not be empty or blank")
    if value != value.strip():
        raise ValueError("mapping names must not have surrounding whitespace")
    if any(category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("mapping names must not contain control or format characters")
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _serialize_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return dict(value)


def _freeze_parameter_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_parameter_mapping(value)
    if type(value) not in (int, float):
        raise ValueError("parameter values must be finite numeric values or mappings")
    if not isfinite(value):
        raise ValueError("parameter values must be finite")
    return value


def _freeze_parameter_mapping(value: Mapping[object, object]) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, nested_value in value.items():
        if type(key) is not str:
            raise ValueError("mapping names must be strings")
        validated_key = _validate_mapping_name(key)
        frozen[validated_key] = _freeze_parameter_value(nested_value)
    return MappingProxyType(frozen)


def _validate_parameter_mapping_input(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("parameter values must be mappings")
    validated: dict[str, object] = {}
    for key, nested_value in value.items():
        if type(key) is not str:
            raise ValueError("mapping names must be strings")
        _validate_mapping_name(key)
        if isinstance(nested_value, Mapping):
            validated[key] = _validate_parameter_mapping_input(nested_value)
        elif type(nested_value) in (int, float) and isfinite(nested_value):
            validated[key] = nested_value
        else:
            raise ValueError("parameter values must be finite numeric values or mappings")
    return validated


def _serialize_parameter_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _serialize_parameter_value(nested_value)
            for key, nested_value in value.items()
        }
    return value


def _serialize_parameter_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _serialize_parameter_value(nested_value)
        for key, nested_value in value.items()
    }


NonBlankText = Annotated[str, AfterValidator(_require_nonblank)]
Identifier = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9_.-]+$"),
    AfterValidator(_require_nonblank),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CanonicalRelativePath = Annotated[
    str,
    AfterValidator(_validate_relative_posix_path),
]
MappingName = Annotated[str, AfterValidator(_validate_mapping_name)]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
NonnegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0)]
ImmutableFloatMapping = Annotated[
    Mapping[MappingName, FiniteFloat],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float]),
]
ImmutableOptionalFloatMapping = Annotated[
    Mapping[MappingName, FiniteFloat | None],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float | None]),
]
ImmutableProbabilityMapping = Annotated[
    Mapping[MappingName, Probability],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float]),
]
ImmutableArtifactHashMapping = Annotated[
    Mapping[CanonicalRelativePath, Sha256],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, str]),
]
ImmutableParameterMapping = Annotated[
    Mapping[MappingName, object],
    BeforeValidator(_validate_parameter_mapping_input),
    AfterValidator(_freeze_parameter_mapping),
    PlainSerializer(_serialize_parameter_mapping, return_type=dict[str, object]),
]


class FrozenModel(BaseModel):
    """Common strict and immutable model configuration for V2 contracts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        validate_default=True,
    )


class BrainMode(StrEnum):
    PURE_API = "PURE_API"
    LOCAL_VISION_API = "LOCAL_VISION_API"
    LOCAL_VISION_LOCAL = "LOCAL_VISION_LOCAL"
    DIRECT_FEATURES_LOCAL = "DIRECT_FEATURES_LOCAL"


class MeasurementStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"


class DecisionAction(StrEnum):
    PROPOSE = "PROPOSE"
    HOLD = "HOLD"
    STOP = "STOP"
    REQUEST_HUMAN = "REQUEST_HUMAN"


class ArtifactReference(FrozenModel):
    relative_path: CanonicalRelativePath
    sha256: Sha256
    media_type: NonBlankText


class BrainCandidate(FrozenModel):
    """Read-only V2 projection, independent of existing Candidate models."""

    candidate_id: Identifier
    requested_params: ImmutableParameterMapping


class EyeMetrics(FrozenModel):
    tdecq_db: FiniteFloat | None = None
    outer_er_chan1a_db: FiniteFloat | None = None
    rlm: FiniteFloat | None = None
    outer_oma: FiniteFloat | None = None
    other_metrics: ImmutableOptionalFloatMapping = Field(default_factory=dict)


class EyeSample(FrozenModel):
    measurement_id: Identifier
    campaign_id: Identifier
    sweep_point_id: Identifier
    raw_eye_density_ref: ArtifactReference
    waveform_ref: ArtifactReference | None = None
    metrics: EyeMetrics
    requested_params: ImmutableFloatMapping
    applied_params: ImmutableFloatMapping
    readback_params: ImmutableFloatMapping
    dca_config_fingerprint: Sha256
    measurement_status: MeasurementStatus
    er_status: NonBlankText | None = None
    tdecq_status: NonBlankText | None = None
    tdecq_invalid_reason: NonBlankText | None = None
    captured_at: AwareDatetime
    artifact_hashes: ImmutableArtifactHashMapping

    @model_validator(mode="after")
    def validate_artifact_hashes(self) -> "EyeSample":
        for artifact in (self.raw_eye_density_ref, self.waveform_ref):
            if artifact is not None and self.artifact_hashes.get(artifact.relative_path) != artifact.sha256:
                raise ValueError(
                    "artifact_hashes must contain every artifact reference with its exact hash"
                )
        return self


class PerceptionResult(FrozenModel):
    measurement_id: Identifier
    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    defects: ImmutableProbabilityMapping
    feature_vector: tuple[FiniteFloat, ...] | None = None
    provider_id: NonBlankText
    provider_version: NonBlankText
    input_hash: Sha256
    memory_revision: Sha256 | None = None
    memory_id: Sha256 | None = None
    reference_measurement_id: Identifier | None = None
    better_than_reference: StrictBool | None = None

    @model_validator(mode="after")
    def validate_comparison_provenance(self) -> "PerceptionResult":
        if (self.reference_measurement_id is None) != (self.better_than_reference is None):
            raise ValueError(
                "reference_measurement_id and better_than_reference must be supplied together"
            )
        if (self.memory_id is None) != (self.memory_revision is None):
            raise ValueError(
                "memory_id and memory_revision must be supplied together"
            )
        return self


class DecisionProposal(FrozenModel):
    based_on_measurement_id: Identifier
    action: DecisionAction
    candidate_id: Identifier | None = None
    requested_params: ImmutableParameterMapping | None = None
    reason_codes: tuple[NonBlankText, ...] = Field(min_length=1)
    expected_improvement: FiniteFloat | None = None
    provider_id: NonBlankText
    provider_version: NonBlankText

    @model_validator(mode="after")
    def validate_action_fields(self) -> "DecisionProposal":
        has_executable_fields = (
            self.candidate_id is not None or self.requested_params is not None
        )
        if (
            "NO_SAFE_CANDIDATE" in self.reason_codes
            and self.action is not DecisionAction.HOLD
        ):
            raise ValueError("NO_SAFE_CANDIDATE reason requires HOLD action")
        if self.action is DecisionAction.PROPOSE:
            if self.candidate_id is None or self.requested_params is None:
                raise ValueError("PROPOSE requires candidate_id and requested_params")
        elif has_executable_fields:
            raise ValueError(
                f"{self.action.value} forbids candidate_id and requested_params"
            )
        return self


class BrainResult(FrozenModel):
    perception: PerceptionResult
    decision: DecisionProposal
    brain_mode: BrainMode
    latency_ms: NonnegativeFiniteFloat
    request_hash: Sha256
    response_hash: Sha256

    @model_validator(mode="after")
    def validate_measurement_ids(self) -> "BrainResult":
        if self.perception.measurement_id != self.decision.based_on_measurement_id:
            raise ValueError("perception.measurement_id must equal decision.based_on_measurement_id")
        return self


class V2Observation(FrozenModel):
    measurement_id: Identifier
    tuning_observation: TuningObservation
    perception: PerceptionResult

    @model_validator(mode="after")
    def validate_measurement_id(self) -> "V2Observation":
        if self.measurement_id != self.perception.measurement_id:
            raise ValueError("measurement_id must equal perception.measurement_id")
        return self
