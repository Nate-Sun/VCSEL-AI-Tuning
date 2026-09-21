"""Strict HTTP request schemas for the campaign API."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    field_validator,
)

from vcsel_ai_tuning.campaign.models import (
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
)


CampaignId = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _authorization_mode_from_wire(value: object) -> object:
    if isinstance(value, AuthorizationMode):
        return value
    if type(value) is str:
        return AuthorizationMode(value)
    return value


def _campaign_state_from_wire(value: object) -> object:
    if isinstance(value, CampaignState):
        return value
    if type(value) is str:
        return CampaignState(value)
    return value


WireAuthorizationMode = Annotated[
    AuthorizationMode,
    BeforeValidator(_authorization_mode_from_wire),
]
WireCampaignState = Annotated[
    CampaignState,
    BeforeValidator(_campaign_state_from_wire),
]


class TuningRequestPath(StrEnum):
    PATH_EXHAUSTIVE = "PATH_EXHAUSTIVE"
    PATH_TDECQ_BAYESIAN = "PATH_TDECQ_BAYESIAN"
    PATH_HIERARCHICAL = "PATH_HIERARCHICAL"


def _tuning_path_from_wire(value: object) -> object:
    if isinstance(value, TuningRequestPath):
        return value
    if type(value) is str:
        return TuningRequestPath(value)
    return value


WireTuningRequestPath = Annotated[
    TuningRequestPath,
    BeforeValidator(_tuning_path_from_wire),
]


class StrictRequest(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class CreateCampaignRequest(StrictRequest):
    campaign_id: CampaignId
    baseline_id: CampaignId
    baseline_hash: Sha256
    config_hash: Sha256
    authorization_mode: WireAuthorizationMode = AuthorizationMode.PER_STAGE
    tuning_path: WireTuningRequestPath = Field(
        default=TuningRequestPath.PATH_HIERARCHICAL,
        exclude=True,
    )

    @field_validator("authorization_mode")
    @classmethod
    def require_per_stage_mode(
        cls,
        value: AuthorizationMode,
    ) -> AuthorizationMode:
        if value != AuthorizationMode.PER_STAGE:
            raise ValueError(
                f"{value.value} authorization is disabled; V1 requires PER_STAGE"
            )
        return value


class StagePlanRequest(StrictRequest):
    stage: WireCampaignState
    max_candidates: StrictInt = Field(gt=0)
    max_duration_s: FiniteFloat = Field(gt=0)


class StageApprovalRequest(StrictRequest):
    authorization: StageAuthorization

    @field_validator("authorization", mode="before")
    @classmethod
    def parse_strict_authorization_json(cls, value: object) -> object:
        if isinstance(value, StageAuthorization):
            return value
        return StageAuthorization.model_validate_json(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
