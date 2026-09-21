"""Lazy handoff to the separately developed pure V1.1 optimizer package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


class OptimizerInterfaceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class V11OptimizerApi:
    alternating_controller: Any
    alternation_policy: Any
    coarse_planner: Any
    fine_optimizer: Any
    fir_controller: Any
    retune_planner: Any
    aggregate_repeated_observations: Any
    compare_results: Any


def load_v11_optimizer_api(
    *,
    loader: Callable[[str], Any] = import_module,
) -> V11OptimizerApi:
    """Import pure optimizer interfaces only when an AI runner is assembled."""

    module_name = "vcsel_ai_tuning.optimization.v11"
    try:
        module = loader(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise OptimizerInterfaceUnavailable(
            "the separate optimization/v11 package is not installed"
        ) from error
    names = (
        "AlternatingController",
        "AlternationPolicy",
        "CoarsePlanner",
        "DiscreteBayesianFine",
        "FirDecisionController",
        "RetunePlanner",
        "aggregate_repeated_observations",
        "compare_results",
    )
    missing = tuple(name for name in names if not hasattr(module, name))
    if missing:
        raise OptimizerInterfaceUnavailable(
            "optimization/v11 handoff is missing: " + ", ".join(missing)
        )
    return V11OptimizerApi(
        alternating_controller=module.AlternatingController,
        alternation_policy=module.AlternationPolicy,
        coarse_planner=module.CoarsePlanner,
        fine_optimizer=module.DiscreteBayesianFine,
        fir_controller=module.FirDecisionController,
        retune_planner=module.RetunePlanner,
        aggregate_repeated_observations=module.aggregate_repeated_observations,
        compare_results=module.compare_results,
    )
