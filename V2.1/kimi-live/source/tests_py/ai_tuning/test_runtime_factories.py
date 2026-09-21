from __future__ import annotations

from importlib import import_module
import socket
import sys

import pytest

from vcsel_ai_tuning.campaign.models import FirSettings, load_b0
from vcsel_ai_tuning.optimization.exhaustive import ExhaustiveOptimizer
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import OptimizationContext
from vcsel_ai_tuning.runtime import (
    FakeFirProposer,
    OptimizerCapability,
    OptimizerCapabilityError,
    OptimizerFactory,
    ReadOnlyModeError,
    RuntimeAssemblyError,
    RuntimeCapabilities,
    build_fake_adapters,
    create_runtime,
    read_only_adapters,
)
from vcsel_ai_tuning.runtime_config import (
    HardwareConfig,
    RuntimeConfig,
    RuntimeMode,
    StorageConfig,
)
from vcsel_ai_tuning.domain import ParameterSpec


def _space() -> IntegerParameterSpace:
    return IntegerParameterSpace(
        (
            ParameterSpec(
                name="eq",
                unit="code",
                lower=21,
                upper=22,
                quantization=1,
                max_step=1,
                grid_values=(21, 22),
            ),
        )
    )


def _context() -> OptimizationContext:
    return OptimizationContext(
        driver_profile_hash="a" * 64,
        vcsel_model="fake-vcsel",
        device_id="fake-device-1",
        test_context_id="fake-context-1",
    )


def test_optimizer_factory_distinguishes_supported_paths():
    factory = OptimizerFactory()

    exhaustive = factory.create(
        OptimizerCapability.PATH_EXHAUSTIVE,
        space=_space(),
        context=_context(),
    )
    bayesian = factory.create(
        OptimizerCapability.PATH_TDECQ_BAYESIAN,
        space=_space(),
        context=_context(),
        seed=7,
    )

    assert isinstance(exhaustive, ExhaustiveOptimizer)
    assert bayesian.__class__.__name__ == "TdecqBayesianOptimizer"


def test_optimizer_factory_fails_closed_for_missing_hierarchical_optimizer():
    factory = OptimizerFactory()

    with pytest.raises(
        OptimizerCapabilityError,
        match="PATH_HIERARCHICAL.*not configured",
    ) as raised:
        factory.create(
            OptimizerCapability.PATH_HIERARCHICAL,
            space=_space(),
            context=_context(),
        )

    assert raised.value.code == "CAPABILITY_UNAVAILABLE"


def test_runtime_capabilities_do_not_claim_missing_production_features():
    capabilities = RuntimeCapabilities.for_mode(
        RuntimeMode.READ_ONLY,
        hierarchical_ready=False,
        fir_proposer_ready=False,
        hardware_connected=True,
    )

    assert capabilities.runtime_mode is RuntimeMode.READ_ONLY
    assert capabilities.hardware_connected is True
    assert capabilities.read_only is True
    assert capabilities.exhaustive_ready is True
    assert capabilities.tdecq_bayesian_ready is True
    assert capabilities.hierarchical_ready is False
    assert capabilities.fir_proposer_ready is False
    assert capabilities.live_tuning_enabled is False
    assert capabilities.authorization_modes == ("PER_STAGE",)
    assert capabilities.full_campaign is False


def test_non_fake_runtime_requires_explicit_human_approved_tuning_profile(
    tmp_path,
):
    adapter_factory_called = False

    def adapter_factory(_campaign_id, _baseline):
        nonlocal adapter_factory_called
        adapter_factory_called = True
        raise AssertionError("adapter factory must not run during startup")

    config = RuntimeConfig(
        mode=RuntimeMode.READ_ONLY,
        storage=StorageConfig(
            sqlite_path=tmp_path / "campaigns.sqlite3",
            artifact_root=tmp_path / "artifacts",
        ),
        hardware=HardwareConfig(
            driver_profile=tmp_path / "driver-profile.yaml",
            dca_visa_resource="TCPIP::example",
            i2c_com_port="COM1",
            i2c_address=0x50,
            bert_host="192.0.2.1",
            bert_port=9000,
            resource_ids=("bench-1",),
        ),
    )

    with pytest.raises(
        RuntimeAssemblyError,
        match="human-approved tuning profile factory",
    ):
        create_runtime(
            config,
            hardware_adapter_factory=adapter_factory,
        )

    assert adapter_factory_called is False


def test_fake_adapters_complete_deterministic_read_write_measurement_cycle():
    baseline = load_b0()
    adapters = build_fake_adapters(baseline)

    assert adapters.dca.preflight()["identity_verified"] is True
    assert adapters.i2c.preflight()["configuration_verified"] is True
    assert adapters.bert.preflight()["identity_verified"] is True
    assert adapters.i2c.apply({"eq": 21})["eq"] == 21
    target_fir = baseline.fir.model_copy(
        update={"taps": (1, 9, -28, 103, -8, 1, -3)}
    )
    assert adapters.bert.apply("SSPRQ", target_fir) == target_fir

    first = adapters.dca.measure_single()
    second = adapters.dca.measure_single()

    assert first.acquisition_valid is True
    assert first.tdecq_db is not None
    assert first == second
    assert adapters.i2c.readback()["eq"] == 21
    assert adapters.bert.readback() == {"pattern": "SSPRQ", "fir": target_fir}


def test_read_only_adapters_fail_closed_for_every_write():
    baseline = load_b0()
    fake = build_fake_adapters(baseline)
    adapters = read_only_adapters(fake)

    assert adapters.i2c.readback() == baseline.i2c.model_dump()
    assert adapters.bert.readback()["fir"] == baseline.fir
    assert adapters.dca.measure_single().acquisition_valid is True
    with pytest.raises(ReadOnlyModeError, match="READ_ONLY") as i2c_error:
        adapters.i2c.apply({"eq": 21})
    with pytest.raises(ReadOnlyModeError, match="READ_ONLY") as bert_error:
        adapters.bert.apply("SSPRQ", baseline.fir)

    assert i2c_error.value.code == "READ_ONLY_MODE"
    assert bert_error.value.code == "READ_ONLY_MODE"
    assert fake.i2c.readback() == baseline.i2c.model_dump()
    assert fake.bert.readback()["fir"] == baseline.fir


def test_fake_fir_proposer_is_deterministic_and_preserves_nlc():
    proposer = FakeFirProposer()
    current = FirSettings(
        taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )

    first = proposer.propose(current)
    second = proposer.propose(current)

    assert first == second
    assert first != current
    assert first.upper_nlc == current.upper_nlc
    assert first.lower_nlc == current.lower_nlc
    assert first.l1 <= 170


def test_importing_runtime_does_not_open_network_or_import_live_drivers(monkeypatch):
    def forbidden_socket(*_args, **_kwargs):
        raise AssertionError("runtime import attempted to open a socket")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    sys.modules.pop("vcsel_ai_tuning.runtime", None)

    module = import_module("vcsel_ai_tuning.runtime")

    assert module.RuntimeMode.FAKE == "FAKE"
    assert "pyvisa" not in sys.modules
    assert "serial" not in sys.modules
