"""Lazy handoff to the separately developed pure V1.1 optimizer package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


class OptimizerInterfaceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class V11OptimizerApi:
    alternating_controller: type[Any]
    alternation_mode: type[Any]
    alternation_policy: type[Any]
    coarse_planner: type[Any]
    complete_tuning_combination: type[Any]
    discrete_parameter_space: type[Any]
    fine_optimizer: type[Any]
    experiment_fingerprint: type[Any]
    fir_authorization_envelope: type[Any]
    fir_controller: type[Any]
    fir_target: type[Any]
    i2c_observation: type[Any]
    metric_result: type[Any]
    optimizer_status: type[Any]
    pilot_planner: type[Any]
    retune_planner: type[Any]
    tuning_parameter: type[Any]
    aggregate_repeated_observations: Callable[..., Any]
    compare_results: Callable[..., Any]


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
    type_names = (
        "AlternatingController",
        "AlternationMode",
        "AlternationPolicy",
        "CoarsePlanner",
        "CompleteTuningCombination",
        "DiscreteParameterSpace",
        "DiscreteBayesianFine",
        "ExperimentFingerprint",
        "FirAuthorizationEnvelope",
        "FirDecisionController",
        "FirTarget",
        "I2CObservation",
        "MetricResult",
        "OptimizerStatus",
        "Pilot3Planner",
        "RetunePlanner",
        "TuningParameter",
    )
    callable_names = (
        "aggregate_repeated_observations",
        "compare_results",
    )
    names = type_names + callable_names
    missing = tuple(name for name in names if not hasattr(module, name))
    if missing:
        raise OptimizerInterfaceUnavailable(
            "optimization/v11 handoff is missing: " + ", ".join(missing)
        )
    invalid = tuple(
        name for name in type_names if not isinstance(getattr(module, name), type)
    ) + tuple(
        name for name in callable_names if not callable(getattr(module, name))
    )
    if invalid:
        raise OptimizerInterfaceUnavailable(
            "optimization/v11 handoff has invalid interfaces: "
            + ", ".join(invalid)
        )
    return V11OptimizerApi(
        alternating_controller=module.AlternatingController,
        alternation_mode=module.AlternationMode,
        alternation_policy=module.AlternationPolicy,
        coarse_planner=module.CoarsePlanner,
        complete_tuning_combination=module.CompleteTuningCombination,
        discrete_parameter_space=module.DiscreteParameterSpace,
        fine_optimizer=module.DiscreteBayesianFine,
        experiment_fingerprint=module.ExperimentFingerprint,
        fir_authorization_envelope=module.FirAuthorizationEnvelope,
        fir_controller=module.FirDecisionController,
        fir_target=module.FirTarget,
        i2c_observation=module.I2CObservation,
        metric_result=module.MetricResult,
        optimizer_status=module.OptimizerStatus,
        pilot_planner=module.Pilot3Planner,
        retune_planner=module.RetunePlanner,
        tuning_parameter=module.TuningParameter,
        aggregate_repeated_observations=module.aggregate_repeated_observations,
        compare_results=module.compare_results,
    )
