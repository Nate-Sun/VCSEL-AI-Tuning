"""Reusable continuous eye-optimization control contracts."""

from .models import (
    CampaignResult,
    ContinuousRequest,
    ContinuousState,
    ExecutionMode,
    ExecutionOutcome,
    HumanDecision,
    ProviderAction,
    ProviderActionKind,
    ReasonCode,
    RoundRecord,
    StoppingPolicy,
)
from .policy import (
    DEFAULT_I2C_POLICY,
    I2cParameterPolicy,
    ParameterRule,
    ProposalErrorCode,
    ProposalValidationError,
    materialize_candidate,
)

__all__ = [
    "CampaignResult",
    "ContinuousRequest",
    "ContinuousState",
    "ExecutionMode",
    "ExecutionOutcome",
    "HumanDecision",
    "ProviderAction",
    "ProviderActionKind",
    "ReasonCode",
    "RoundRecord",
    "StoppingPolicy",
    "DEFAULT_I2C_POLICY",
    "I2cParameterPolicy",
    "ParameterRule",
    "ProposalErrorCode",
    "ProposalValidationError",
    "materialize_candidate",
]
