from importlib import import_module
from typing import TYPE_CHECKING

from .coordinate import CoordinateDescent
from .effects import EffectEdge, EffectMemory, PairInteraction, ResponseNode
from .exhaustive import ExhaustiveOptimizer
from .parameter_space import IntegerParameterSpace, NoTransitionPath
from .protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    OptimizationContext,
    OptimizerProtocol,
    PathComplete,
    Suggestion,
    TuningObservation,
    TuningPath,
)
from .replay import ReplayReport, ReplaySurface, replay

if TYPE_CHECKING:
    from .bayesian import (
        BayesianSnapshot,
        CandidatePrediction,
        TdecqBayesianOptimizer,
    )

_LAZY_EXPORTS = {
    "BayesianSnapshot": (".bayesian", "BayesianSnapshot"),
    "CandidatePrediction": (".bayesian", "CandidatePrediction"),
    "TdecqBayesianOptimizer": (".bayesian", "TdecqBayesianOptimizer"),
}

__all__ = [
    "BayesianSnapshot",
    "CandidatePrediction",
    "CandidateSuggestion",
    "CoordinateDescent",
    "EffectEdge",
    "EffectMemory",
    "ExhaustiveOptimizer",
    "FallbackRequested",
    "IntegerParameterSpace",
    "NoSafeCandidate",
    "NoTransitionPath",
    "OptimizationContext",
    "OptimizerProtocol",
    "PairInteraction",
    "PathComplete",
    "ReplayReport",
    "ReplaySurface",
    "ResponseNode",
    "Suggestion",
    "TdecqBayesianOptimizer",
    "TuningObservation",
    "TuningPath",
    "replay",
]


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing == "sklearn" or missing.startswith("sklearn."):
            raise ModuleNotFoundError(
                f"{name} requires the optional optimizer dependencies; install "
                "them with `python -m pip install -e '.[optimizer]'`."
            ) from error
        raise

    resolved = getattr(module, attribute_name)
    globals()[name] = resolved
    return resolved
