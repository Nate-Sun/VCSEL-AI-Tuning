"""Pure campaign configuration and domain contracts."""

from .models import (
    AppliedSettings,
    AuthorizationMode,
    BaselineConfig,
    Campaign,
    CampaignState,
    Candidate,
    FirSettings,
    I2cSettings,
    Observation,
    ParameterSpec,
    StageAuthorization,
    StageSummary,
    load_b0,
)

__all__ = [
    "AppliedSettings",
    "AuthorizationMode",
    "BaselineConfig",
    "Campaign",
    "CampaignState",
    "Candidate",
    "FirSettings",
    "I2cSettings",
    "Observation",
    "ParameterSpec",
    "StageAuthorization",
    "StageSummary",
    "load_b0",
]
