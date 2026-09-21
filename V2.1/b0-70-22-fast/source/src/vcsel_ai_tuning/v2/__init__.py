"""V2 AI tuning brain data contracts."""

from .contracts import (
    ArtifactReference,
    BrainCandidate,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
    PerceptionResult,
    V2Observation,
)
from .brain import (
    ComposedBrain,
    DecisionProvider,
    EyeScorer,
    FusedBrainProvider,
    MultimodalApiBrain,
    OptimizationBrain,
    is_factory_gated_brain,
)
from .factory import (
    BrainCapability,
    BrainFactory,
    BrainFactoryConfig,
    BrainUnavailableError,
    load_v2_config,
)

__all__ = [
    "ArtifactReference",
    "BrainCandidate",
    "BrainMode",
    "BrainResult",
    "DecisionAction",
    "DecisionProposal",
    "EyeMetrics",
    "EyeSample",
    "MeasurementStatus",
    "PerceptionResult",
    "V2Observation",
    "ComposedBrain",
    "DecisionProvider",
    "EyeScorer",
    "FusedBrainProvider",
    "MultimodalApiBrain",
    "OptimizationBrain",
    "is_factory_gated_brain",
    "BrainCapability",
    "BrainFactory",
    "BrainFactoryConfig",
    "BrainUnavailableError",
    "load_v2_config",
]
