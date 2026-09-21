from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Protocol

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    model_validator,
)


def _freeze_settings(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(values))


def _serialize_settings(values: Mapping[str, int]) -> dict[str, int]:
    return dict(values)


_ImmutableSettings = Annotated[
    Mapping[str, int],
    AfterValidator(_freeze_settings),
    PlainSerializer(_serialize_settings, return_type=dict[str, int]),
]

_NonBlankText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_TdecqStatus = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_upper=True, min_length=1),
]


class TuningPath(StrEnum):
    EXHAUSTIVE = "PATH_EXHAUSTIVE"
    TDECQ_BAYESIAN = "PATH_TDECQ_BAYESIAN"


class OptimizationContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    driver_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vcsel_model: _NonBlankText
    device_id: _NonBlankText
    test_context_id: _NonBlankText

    @property
    def compatibility_key(self) -> tuple[str, str, str]:
        return (
            self.driver_profile_hash,
            self.vcsel_model,
            self.test_context_id,
        )


class TuningObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    observation_id: _NonBlankText
    candidate_id: _NonBlankText
    context: OptimizationContext
    settings_requested: _ImmutableSettings
    settings_readback: _ImmutableSettings
    tdecq_db: float | None
    tdecq_status: _TdecqStatus
    tdecq_invalid_reason: str | None = None
    er_db: float | None = None
    er_valid: bool = False
    er_status: str | None = None
    artifact_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_metric_states(self) -> "TuningObservation":
        if self.tdecq_status == "CORR" and self.tdecq_db is None:
            raise ValueError("CORR TDECQ requires a numeric value")
        if self.tdecq_status == "CORR" and self.tdecq_invalid_reason is not None:
            raise ValueError("CORR TDECQ cannot have an invalid reason")
        if self.tdecq_status != "CORR" and self.tdecq_db is not None:
            raise ValueError("non-CORR TDECQ must keep value=None")
        if self.tdecq_status != "CORR" and (
            self.tdecq_invalid_reason is None
            or not self.tdecq_invalid_reason.strip()
        ):
            raise ValueError("non-CORR TDECQ requires an invalid reason")
        if self.er_valid != (self.er_db is not None):
            raise ValueError("ER validity must match the presence of a numeric value")
        return self


class CandidateSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["CANDIDATE"] = "CANDIDATE"
    path: TuningPath
    candidate_id: _NonBlankText
    requested: _ImmutableSettings
    reason: _NonBlankText
    acquisition: float | None = None


class FallbackRequested(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["FALLBACK_REQUESTED"] = "FALLBACK_REQUESTED"
    path: TuningPath
    reason: _NonBlankText


class NoSafeCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["NO_SAFE_CANDIDATE"] = "NO_SAFE_CANDIDATE"
    path: TuningPath
    reason: _NonBlankText


class PathComplete(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["PATH_COMPLETE"] = "PATH_COMPLETE"
    path: TuningPath
    status: Literal[
        "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
        "EXHAUSTIVE_INCOMPLETE",
        "BEST_TDECQ_FOUND_WITHIN_BUDGET",
        "NO_VALID_TDECQ",
    ]
    measured_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    best_observation_id: _NonBlankText | None = None
    best_tdecq_db: float | None = None

    @model_validator(mode="after")
    def validate_terminal_claim(self) -> "PathComplete":
        if self.measured_count > self.candidate_count:
            raise ValueError("measured_count cannot exceed candidate_count")

        has_best_id = self.best_observation_id is not None
        has_best_value = self.best_tdecq_db is not None
        if has_best_id != has_best_value:
            raise ValueError(
                "best observation identifier and TDECQ must form a pair"
            )
        if has_best_id and not self.best_observation_id.strip():
            raise ValueError("best_observation_id must be nonblank")

        has_best = has_best_id and has_best_value
        if has_best and self.measured_count == 0:
            raise ValueError(
                "best observation requires at least one measured candidate"
            )
        if (
            self.status == "EXHAUSTIVE_INCOMPLETE"
            and self.path is not TuningPath.EXHAUSTIVE
        ):
            raise ValueError(
                "EXHAUSTIVE_INCOMPLETE is only valid for PATH_EXHAUSTIVE"
            )
        if (
            self.status == "BEST_TDECQ_FOUND_WITHIN_BUDGET"
            and self.path is not TuningPath.TDECQ_BAYESIAN
        ):
            raise ValueError(
                "BEST_TDECQ_FOUND_WITHIN_BUDGET is only valid for "
                "PATH_TDECQ_BAYESIAN"
            )
        if self.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID" and (
            self.measured_count != self.candidate_count or not has_best
        ):
            raise ValueError(
                "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID requires full enumeration "
                "and a best observation"
            )
        if self.status == "BEST_TDECQ_FOUND_WITHIN_BUDGET" and not has_best:
            raise ValueError(
                "BEST_TDECQ_FOUND_WITHIN_BUDGET requires a best observation"
            )
        if self.status == "NO_VALID_TDECQ" and has_best:
            raise ValueError("NO_VALID_TDECQ cannot include a best observation")
        if (
            self.status == "EXHAUSTIVE_INCOMPLETE"
            and self.measured_count == self.candidate_count
        ):
            raise ValueError(
                "EXHAUSTIVE_INCOMPLETE requires partial enumeration"
            )
        return self


Suggestion = CandidateSuggestion | FallbackRequested | NoSafeCandidate | PathComplete


class OptimizerProtocol(Protocol):
    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[tuple[int, ...]] = frozenset(),
    ) -> Suggestion:
        raise NotImplementedError
