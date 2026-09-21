from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PlainSerializer,
    StrictInt,
    StringConstraints,
    model_validator,
)


def _freeze_int_mapping(value: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(value))


def _serialize_int_mapping(value: Mapping[str, int]) -> dict[str, int]:
    return dict(value)


def _freeze_metric_mapping(
    value: Mapping[str, float | None],
) -> Mapping[str, float | None]:
    return MappingProxyType(dict(value))


def _serialize_metric_mapping(
    value: Mapping[str, float | None],
) -> dict[str, float | None]:
    return dict(value)


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9_.-]+$",
    ),
]
StatusText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_upper=True, min_length=1),
]
ImmutableIntMapping = Annotated[
    Mapping[str, StrictInt],
    AfterValidator(_freeze_int_mapping),
    PlainSerializer(_serialize_int_mapping, return_type=dict[str, int]),
]
ImmutableMetricMapping = Annotated[
    Mapping[str, FiniteFloat | None],
    AfterValidator(_freeze_metric_mapping),
    PlainSerializer(
        _serialize_metric_mapping,
        return_type=dict[str, float | None],
    ),
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        validate_default=True,
    )


class OptimizerStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    COMPLETE = "COMPLETE"
    NO_VALID_TDECQ = "NO_VALID_TDECQ"
    NOT_READY = "NOT_READY"
    FAILED = "FAILED"


class MetricResult(FrozenModel):
    tdecq_db: FiniteFloat | None
    tdecq_status: StatusText
    tdecq_invalid_reason: str | None = None
    er_db: FiniteFloat | None = None
    er_status: StatusText | None = None

    @classmethod
    def valid(
        cls,
        tdecq_db: float,
        er_db: float | None,
    ) -> Self:
        return cls(
            tdecq_db=tdecq_db,
            tdecq_status="CORR",
            er_db=er_db,
            er_status="CORR" if er_db is not None else None,
        )

    @classmethod
    def invalid(
        cls,
        *,
        status: str,
        reason: str,
        er_db: float | None = None,
    ) -> Self:
        return cls(
            tdecq_db=None,
            tdecq_status=status,
            tdecq_invalid_reason=reason,
            er_db=er_db,
            er_status="CORR" if er_db is not None else None,
        )

    @property
    def has_valid_tdecq(self) -> bool:
        return self.tdecq_status == "CORR" and self.tdecq_db is not None

    @property
    def has_valid_er(self) -> bool:
        return self.er_status == "CORR" and self.er_db is not None

    @model_validator(mode="after")
    def validate_tdecq_state(self) -> Self:
        if self.tdecq_status == "CORR":
            if self.tdecq_db is None:
                raise ValueError("CORR TDECQ requires a numeric value")
            if self.tdecq_invalid_reason is not None:
                raise ValueError("CORR TDECQ cannot have an invalid reason")
        else:
            if self.tdecq_db is not None:
                raise ValueError("non-CORR TDECQ must keep value=None")
            if (
                self.tdecq_invalid_reason is None
                or not self.tdecq_invalid_reason.strip()
            ):
                raise ValueError("non-CORR TDECQ requires an invalid reason")

        if self.er_status == "CORR":
            if self.er_db is None:
                raise ValueError("CORR ER requires a numeric value")
        elif self.er_db is not None:
            raise ValueError("numeric ER requires er_status=CORR")
        return self


class ExperimentFingerprint(FrozenModel):
    fir_taps: tuple[StrictInt, ...] = Field(min_length=7, max_length=7)
    upper_nlc: StrictInt = Field(ge=0)
    lower_nlc: StrictInt = Field(ge=0)
    dca_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class I2CObservation(FrozenModel):
    observation_id: Identifier
    candidate_id: Identifier
    requested_i2c: ImmutableIntMapping
    readback_i2c: ImmutableIntMapping
    result: MetricResult
    fingerprint: ExperimentFingerprint


class AggregatedObservation(FrozenModel):
    readback_i2c: ImmutableIntMapping
    fingerprint: ExperimentFingerprint
    result: MetricResult
    raw_observations: tuple[I2CObservation, ...] = Field(min_length=1)


class AlgorithmCandidate(FrozenModel):
    candidate_id: Identifier
    requested_i2c: ImmutableIntMapping
    predicted_metrics: ImmutableMetricMapping = Field(default_factory=dict)
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    model_version: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
