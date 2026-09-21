from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.v1_1.optimizer_bridge import (
    OptimizerInterfaceUnavailable,
    load_v11_optimizer_api,
)


REQUIRED = (
    "AlternatingController",
    "AlternationPolicy",
    "CoarsePlanner",
    "DiscreteBayesianFine",
    "FirDecisionController",
    "RetunePlanner",
    "aggregate_repeated_observations",
    "compare_results",
)


def test_bridge_loads_separate_public_optimizer_interfaces_lazily():
    module = SimpleNamespace(**{name: object() for name in REQUIRED})
    imports = []

    api = load_v11_optimizer_api(
        loader=lambda name: imports.append(name) or module
    )

    assert imports == ["vcsel_ai_tuning.optimization.v11"]
    assert api.compare_results is module.compare_results
    assert api.fir_controller is module.FirDecisionController


def test_bridge_fails_closed_when_optimizer_handoff_is_incomplete():
    with pytest.raises(OptimizerInterfaceUnavailable, match="RetunePlanner"):
        load_v11_optimizer_api(
            loader=lambda _name: SimpleNamespace(
                **{name: object() for name in REQUIRED if name != "RetunePlanner"}
            )
        )
