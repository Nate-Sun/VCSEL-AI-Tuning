from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.v1_1.optimizer_bridge import (
    OptimizerInterfaceUnavailable,
    load_v11_optimizer_api,
)


REQUIRED = (
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
    "aggregate_repeated_observations",
    "compare_results",
)
TYPE_NAMES = REQUIRED[:-2]
CALLABLE_NAMES = REQUIRED[-2:]

API_FIELD_BY_PUBLIC_NAME = {
    "AlternatingController": "alternating_controller",
    "AlternationMode": "alternation_mode",
    "AlternationPolicy": "alternation_policy",
    "CoarsePlanner": "coarse_planner",
    "CompleteTuningCombination": "complete_tuning_combination",
    "DiscreteParameterSpace": "discrete_parameter_space",
    "DiscreteBayesianFine": "fine_optimizer",
    "ExperimentFingerprint": "experiment_fingerprint",
    "FirAuthorizationEnvelope": "fir_authorization_envelope",
    "FirDecisionController": "fir_controller",
    "FirTarget": "fir_target",
    "I2CObservation": "i2c_observation",
    "MetricResult": "metric_result",
    "OptimizerStatus": "optimizer_status",
    "Pilot3Planner": "pilot_planner",
    "RetunePlanner": "retune_planner",
    "TuningParameter": "tuning_parameter",
    "aggregate_repeated_observations": "aggregate_repeated_observations",
    "compare_results": "compare_results",
}


def valid_optimizer_module() -> SimpleNamespace:
    values = {name: type(name, (), {}) for name in TYPE_NAMES}
    values.update(
        {
            name: (lambda *_args, **_kwargs: None)
            for name in CALLABLE_NAMES
        }
    )
    return SimpleNamespace(**values)


def test_bridge_loads_separate_public_optimizer_interfaces_lazily():
    module = valid_optimizer_module()
    imports = []

    api = load_v11_optimizer_api(
        loader=lambda name: imports.append(name) or module
    )

    assert imports == ["vcsel_ai_tuning.optimization.v11"]
    for public_name, api_field in API_FIELD_BY_PUBLIC_NAME.items():
        assert getattr(api, api_field) is getattr(module, public_name)


def test_bridge_fails_closed_when_optimizer_handoff_is_incomplete():
    module = valid_optimizer_module()
    del module.RetunePlanner

    with pytest.raises(OptimizerInterfaceUnavailable, match="RetunePlanner"):
        load_v11_optimizer_api(loader=lambda _name: module)


@pytest.mark.parametrize("name", ("FirTarget", "compare_results"))
def test_bridge_fails_closed_when_public_interface_has_wrong_kind(name):
    module = valid_optimizer_module()
    setattr(module, name, object())

    with pytest.raises(OptimizerInterfaceUnavailable, match=name):
        load_v11_optimizer_api(loader=lambda _name: module)


def test_importing_fake_ai_execution_does_not_import_real_hardware_modules():
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(source_root)!r})
original_import = builtins.__import__
forbidden = {{
    'pyvisa',
    'serial',
    'vcsel_ai_tuning.hardware.vd1124',
    'vcsel_ai_tuning.hardware.vit_bert',
    'vcsel_ai_tuning.hardware.dca_acquisition',
}}

def guarded_import(name, *args, **kwargs):
    if name in forbidden:
        raise AssertionError(f'FAKE optimization plant imported {{name}}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from datetime import datetime, timezone
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.fake_ai_execution import FakeOptimizationPlant

b0 = B0Snapshot.capture(
    b0_id='guard-b0',
    version=1,
    captured_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
    i2c_fields={{'eq': 22}},
    i2c_raw_registers={{0x10: 22}},
    fir_taps=(0, 9, -28, 104, -8, 1, -3),
    upper_nlc=6,
    lower_nlc=6,
    bert_pattern='SSPRQ',
    i2c_channel=0,
    bert_port='port1',
    bert_channel='0',
    dca_tdecq_source='CHAN1A',
    dca_tdecq_equalizer_function=1,
    dca_outer_er_source='CHAN1A',
    dca_fingerprint=DcaFingerprint.from_mappings(
        identity='FAKE,DCA,1.1',
        measurement_sources={{'outer_er': 'CHAN1A', 'tdecq': 'CHAN1A'}},
        configuration={{'pattern': 'SSPRQ', 'tdecq_equalizer_function': 'FUNC1'}},
    ),
    hardware_identities={{
        'i2c': HardwareIdentity('i2c', 'FAKE-I2C', 'FAKE-VD1124'),
        'bert': HardwareIdentity('bert', 'FAKE-BERT', 'FAKE-VIT'),
        'dca': HardwareIdentity('dca', 'FAKE-DCA', 'FAKE-FlexDCA'),
    }},
)
plant = FakeOptimizationPlant(b0)
assert plant.is_at_b0()
for name in forbidden:
    assert name not in sys.modules, name
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
