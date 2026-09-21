from __future__ import annotations

import builtins
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import socket
import sys

import pytest
import yaml
import vcsel_ai_tuning.runtime as runtime_module

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
from vcsel_ai_tuning.v1_1.fake_runtime import FakeV11Gateway
from vcsel_ai_tuning.v1_1.domain import RuntimeMode as V11RuntimeMode
from vcsel_ai_tuning.v1_1.errors import ReadOnlyPreflightError
from vcsel_ai_tuning.v1_1.persistent_runner import FakeSweepExecutor
from vcsel_ai_tuning.v1_1.live_ai_policy import LiveAIProfile
from vcsel_ai_tuning.domain import ParameterSpec


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_live_profile(tmp_path: Path) -> Path:
    profile = {
        "dca": {
            "resource": "TCPIP0::example-dca::hislip0,4880::INSTR",
            "expected_idn_prefix": "Keysight Technologies,N1010A,",
            "function": 1,
            "timeout_ms": 60_000,
        },
        "bert": {
            "ip": "192.0.2.10",
            "local_ip": "192.0.2.11",
            "udp_port": 9100,
            "port": "port2",
            "channel": "3",
            "timeout_s": 2.0,
        },
        "i2c": {
            "port": "COM42",
            "baudrate": 230_400,
            "device_address": 0xA4,
            "chip_id_register": 0x0004,
            "expected_chip_id": 0x52,
            "chip_revision_register": 0x0005,
            "expected_chip_revision": 0x00,
        },
        "characterization": {
            "default_channel": 2,
            "baseline_repeats": 1,
            "measurement_timeout_s": 180.0,
            "poll_interval_s": 1.0,
            "meaningful_er_floor_db": None,
            "meaningful_tdecq_floor_db": None,
            "tdecq": {
                "source": "FUNC1",
                "input_channel": "CHAN1A",
            },
            "er": {
                "source": "CHAN1A",
                "value_query": ":MEASure:EYE:OER?",
                "source_query": ":MEASure:EYE:OER:SOURce?",
                "status_query": ":MEASure:EYE:OER:STATus?",
                "details_query": ":MEASure:EYE:OER:STATus:DETails?",
                "units_query": ":MEASure:EYE:OER:UNITs?",
            },
            "parameters": {
                "ibias_dac": {"address": 0x2002, "bit_start": 0, "bits": 7},
                "efin_cur": {"address": 0x2003, "bit_start": 0, "bits": 4},
                "vga_cur": {"address": 0x2005, "bit_start": 0, "bits": 4},
                "eq": {"address": 0x2006, "bit_start": 0, "bits": 5},
                "res_gain": {"address": 0x2007, "bit_start": 0, "bits": 4},
                "res_bw": {"address": 0x2008, "bit_start": 0, "bits": 4},
                "gain_ctrl": {"address": 0x2009, "bit_start": 0, "bits": 6},
                "vbias_ref": {"address": 0x200A, "bit_start": 0, "bits": 4},
                "vref": {"address": 0x200B, "bit_start": 0, "bits": 4},
            },
            "safety_limits": {},
        },
    }
    path = tmp_path / "reviewed-live-profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def _read_only_config(tmp_path: Path, profile_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        mode=RuntimeMode.READ_ONLY,
        storage=StorageConfig(
            sqlite_path=tmp_path / "state" / "campaigns.sqlite3",
            artifact_root=tmp_path / "artifacts",
        ),
        b0_path=REPO_ROOT / "configs" / "ai_tuning" / "b0.yaml",
        hardware=HardwareConfig(
            driver_profile=profile_path,
            dca_visa_resource="TCPIP0::example-dca::hislip0,4880::INSTR",
            i2c_com_port="COM42",
            i2c_address=0xA4,
            bert_host="192.0.2.10",
            bert_port=9100,
            resource_ids=("example-dca", "example-i2c", "example-bert"),
        ),
    )


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


def test_live_tuning_runtime_requires_explicit_human_approved_tuning_profile(
    tmp_path,
):
    adapter_factory_called = False

    def adapter_factory(_campaign_id, _baseline):
        nonlocal adapter_factory_called
        adapter_factory_called = True
        raise AssertionError("adapter factory must not run during startup")

    config = RuntimeConfig(
        mode=RuntimeMode.LIVE_TUNING,
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


def test_read_only_runtime_assembles_builtin_v11_without_opening_hardware(tmp_path):
    profile_path = _write_live_profile(tmp_path)
    config = _read_only_config(tmp_path, profile_path)
    driver_factory_calls = 0

    def forbidden_driver_factory(_profile, _remaining_s):
        nonlocal driver_factory_calls
        driver_factory_calls += 1
        raise AssertionError("READ_ONLY startup attempted to open hardware")

    runtime = create_runtime(
        config,
        v11_read_only_driver_factory=forbidden_driver_factory,
    )
    try:
        assert runtime.v11_service is not None
        assert runtime.v11_service.gateway.__class__.__name__ == "ReadOnlyV11Gateway"
        assert runtime.v11_service.gateway.total_timeout_s == 30.0
        assert runtime.v11_service.runner.__class__.__name__ == "ReadOnlyV11TaskRunner"
        health = runtime.v11_service.runtime_health()
        assert health["status"] == "ok"
        assert health["runtime_mode"] == "READ_ONLY"
        assert health["hardware_connected"] is False
        capabilities = runtime.v11_service.runtime_capabilities()
        assert capabilities["runtime_mode"] == "READ_ONLY"
        assert capabilities["dynamic_b0"] is True
        assert capabilities["hardware_connected"] is False
        assert capabilities["hardware_mutation"] is False
        assert capabilities["fake"] is False
        assert driver_factory_calls == 0
    finally:
        assert runtime.shutdown is not None
        runtime.shutdown()


def test_read_only_runtime_separates_platform_and_v11_databases(tmp_path):
    profile_path = _write_live_profile(tmp_path)
    platform_database = tmp_path / "state" / "v1.2" / "campaigns.sqlite3"
    v11_database = platform_database.with_name("v1_1_tasks.sqlite3")
    config = _read_only_config(tmp_path, profile_path).model_copy(
        update={
            "storage": StorageConfig(
                sqlite_path=platform_database,
                artifact_root=tmp_path / "artifacts" / "v1.2",
            )
        }
    )
    driver_factory_calls = 0

    def forbidden_driver_factory(_profile, _remaining_s):
        nonlocal driver_factory_calls
        driver_factory_calls += 1
        raise AssertionError("READ_ONLY startup attempted to open hardware")

    runtime = create_runtime(
        config,
        v11_read_only_driver_factory=forbidden_driver_factory,
    )
    try:
        assert platform_database.is_file()
        assert v11_database.is_file()
        assert platform_database.resolve() != v11_database.resolve()
        assert driver_factory_calls == 0
    finally:
        assert runtime.shutdown is not None
        runtime.shutdown()


def test_read_only_runtime_rejects_runtime_profile_endpoint_mismatch_before_io(
    tmp_path,
):
    profile_path = _write_live_profile(tmp_path)
    config = _read_only_config(tmp_path, profile_path)
    config = config.model_copy(
        update={
            "hardware": config.hardware.model_copy(
                update={"bert_host": "192.0.2.99"}
            )
        }
    )
    driver_factory_called = False

    def forbidden_driver_factory(_profile, _remaining_s):
        nonlocal driver_factory_called
        driver_factory_called = True
        raise AssertionError("endpoint mismatch attempted hardware I/O")

    with pytest.raises(RuntimeAssemblyError, match="endpoint mismatch.*BERT host"):
        create_runtime(
            config,
            v11_read_only_driver_factory=forbidden_driver_factory,
        )

    assert driver_factory_called is False


def test_read_only_runtime_rejects_custom_v11_components_before_io(tmp_path):
    profile_path = _write_live_profile(tmp_path)
    config = _read_only_config(tmp_path, profile_path)

    with pytest.raises(
        RuntimeAssemblyError,
        match="built-in V1.1 gateway and runner",
    ):
        create_runtime(
            config,
            v11_gateway=object(),
            v11_runner=object(),
        )

    assert not config.storage.sqlite_path.exists()


class _LiveV11Gateway(FakeV11Gateway):
    runtime_mode = RuntimeMode.LIVE_TUNING


def _live_config(
    tmp_path: Path,
    profile_path: Path | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        mode=RuntimeMode.LIVE_TUNING,
        storage=StorageConfig(
            sqlite_path=tmp_path / "live" / "campaigns.sqlite3",
            artifact_root=tmp_path / "live" / "artifacts",
        ),
        b0_path=REPO_ROOT / "configs" / "ai_tuning" / "b0.yaml",
        hardware=HardwareConfig(
            driver_profile=profile_path or tmp_path / "reviewed-profile.yaml",
            dca_visa_resource="TCPIP0::example-dca::hislip0,4880::INSTR",
            i2c_com_port="COM42",
            i2c_address=0xA4,
            bert_host="192.0.2.10",
            bert_port=9100,
            resource_ids=("live-bench",),
        ),
    )


def test_live_runtime_builds_persistent_runner_with_explicit_real_sweep_factory(
    tmp_path,
):
    config = _live_config(tmp_path)
    factory_calls = []

    def real_sweep_factory(b0, plan):
        factory_calls.append((b0, plan))
        raise AssertionError("startup must not construct or connect sweep hardware")

    runtime = create_runtime(
        config,
        hardware_adapter_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("startup opened legacy hardware")
        ),
        tuning_profile_factory=lambda _path, baseline: (
            runtime_module._built_in_fake_tuning_profile(baseline)
        ),
        v11_gateway=_LiveV11Gateway(),
        v11_sweep_executor_factory=real_sweep_factory,
    )
    try:
        runner = runtime.v11_service.runner
        assert runner._sweep_executor_factory is real_sweep_factory
        assert runner._sweep_executor_factory is not FakeSweepExecutor
        assert factory_calls == []
    finally:
        runtime.shutdown()


def test_live_runtime_registers_builtin_lazy_real_sweep_factory(tmp_path):
    profile_path = _write_live_profile(tmp_path)
    config = _live_config(tmp_path, profile_path)

    runtime = create_runtime(
        config,
        hardware_adapter_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("startup opened legacy hardware")
        ),
        tuning_profile_factory=lambda _path, baseline: (
            runtime_module._built_in_fake_tuning_profile(baseline)
        ),
        v11_gateway=_LiveV11Gateway(),
    )
    try:
        runner = runtime.v11_service.runner
        assert runner._sweep_executor_factory is not None
        assert runner._sweep_executor_factory is not FakeSweepExecutor
        assert runner.last_sweep_executor is None
    finally:
        runtime.shutdown()


def test_formal_live_runtime_assembles_v11_without_legacy_factories_or_io(
    tmp_path,
    monkeypatch,
):
    profile_path = _write_live_profile(tmp_path)
    config = _live_config(tmp_path, profile_path)
    socket_calls = []
    modules_before = set(sys.modules)

    def forbidden_socket(*args, **kwargs):
        socket_calls.append((args, kwargs))
        raise AssertionError("LIVE startup opened a socket")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {
            "pyvisa",
            "serial",
            "vcsel_ai_tuning.hardware.vit_bert",
        }:
            raise AssertionError(f"LIVE startup imported hardware driver {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    runtime = create_runtime(config)
    try:
        assert runtime.v11_service is not None
        assert runtime.v11_service.gateway.__class__.__name__ == "LiveV11Gateway"
        assert runtime.v11_service.gateway.runtime_mode is V11RuntimeMode.LIVE
        assert runtime.v11_service.gateway.capabilities()["hardware_mutation"] is True
        runner = runtime.v11_service.runner
        assert runner.__class__.__name__ == "PersistentV11TaskRunner"
        assert runner._sweep_executor_factory is not None
        assert runner._sweep_executor_factory is not FakeSweepExecutor
        assert runner.last_sweep_executor is None
        assert runtime.status_provider.capabilities.live_tuning_enabled is False
        with pytest.raises(RuntimeAssemblyError, match="legacy V1 campaigns are disabled"):
            runtime.service.create_campaign(object())
        assert socket_calls == []
        imported_during_startup = set(sys.modules) - modules_before
        assert "pyvisa" not in imported_during_startup
        assert "serial" not in imported_during_startup
        assert "vcsel_ai_tuning.hardware.vit_bert" not in imported_during_startup
    finally:
        runtime.shutdown()


def test_read_only_runtime_wires_spawned_dca_preflight_by_default(tmp_path):
    profile_path = _write_live_profile(tmp_path)
    config = _read_only_config(tmp_path, profile_path)

    runtime = create_runtime(config)
    try:
        factory = runtime.v11_service.gateway._driver_factory
        assert factory.func.__name__ == "open_isolated_read_only_drivers"
        assert factory.keywords["cleanup_grace_s"] == 5.0
        assert factory.keywords["artifact_root"] == (
            config.storage.artifact_root.parent / "v1.2" / "preflight-dca"
        )
    finally:
        runtime.shutdown()


def test_staged_native_v12_runtime_advertises_only_lazy_ai_without_io(
    tmp_path,
    monkeypatch,
):
    profile_path = _write_live_profile(tmp_path)
    config = _live_config(tmp_path, profile_path).model_copy(
        update={
            "live_ai_profile": LiveAIProfile.PILOT_3_THEN_FAST24,
            "measurement_timeout_s": 60.0,
            "cleanup_grace_s": 5.0,
        }
    )
    socket_calls = []
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: socket_calls.append((args, kwargs)),
    )

    runtime = create_runtime(config)
    try:
        capabilities = runtime.v11_service.runtime_capabilities()
        runner = runtime.v11_service.runner

        assert capabilities["product_version"] == "1.2.0"
        assert capabilities["campaign_types"] == ["AI_OPTIMIZATION"]
        assert capabilities["sweep_execution_ready"] is False
        assert capabilities["ai_execution_ready"] is True
        assert capabilities["live_ai_profile"] == "PILOT_3_THEN_FAST24"
        assert capabilities["candidate_budget"] == 27
        assert capabilities["pilot_candidate_budget"] == 3
        assert capabilities["fast_candidate_budget"] == 24
        assert capabilities["coarse_candidate_budget"] == 6
        assert capabilities["fine_candidate_budget"] == 16
        assert capabilities["fir_first_candidate_budget"] == 1
        assert capabilities["fir_iterative_candidate_budget"] == 1
        assert capabilities["measurement_timeout_s"] == 60.0
        assert capabilities["cleanup_grace_s"] == 5.0
        assert capabilities["dataset_required"] is True
        assert capabilities["final_restore"] == "B0_VERIFIED"
        assert runner._sweep_executor_factory is None
        assert runner._ai_execution_factory is not None
        preflight_factory = runtime.v11_service.gateway._driver_factory
        assert (
            preflight_factory.func.__name__
            == "open_isolated_read_only_drivers"
        )
        assert preflight_factory.keywords["cleanup_grace_s"] == 5.0
        assert socket_calls == []
    finally:
        runtime.shutdown()


def test_formal_live_runtime_rejects_profile_endpoint_mismatch_before_io(
    tmp_path,
    monkeypatch,
):
    profile_path = _write_live_profile(tmp_path)
    config = _live_config(tmp_path, profile_path)
    config = config.model_copy(
        update={
            "hardware": config.hardware.model_copy(
                update={"bert_host": "192.0.2.99"}
            )
        }
    )
    socket_calls = []
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: socket_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeAssemblyError, match="endpoint mismatch.*BERT host"):
        create_runtime(config)

    assert socket_calls == []
    assert not config.storage.sqlite_path.exists()


def test_formal_live_runtime_rejects_tdecq_function_mismatch_before_io(
    tmp_path,
):
    profile_path = _write_live_profile(tmp_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["dca"]["function"] = 2
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    config = _live_config(tmp_path, profile_path)
    driver_factory_calls = 0

    @contextmanager
    def forbidden_driver_factory(_profile, _remaining_s):
        nonlocal driver_factory_calls
        driver_factory_calls += 1
        raise AssertionError("invalid DCA profile attempted hardware I/O")
        yield  # pragma: no cover - contextmanager requires a generator

    with pytest.raises(
        RuntimeAssemblyError,
        match="tdecq.source must match.*FUNC2",
    ):
        create_runtime(
            config,
            v11_live_driver_factory=forbidden_driver_factory,
        )

    assert driver_factory_calls == 0
    assert not config.storage.sqlite_path.exists()


def test_formal_live_runtime_opens_query_drivers_only_for_explicit_preflight(
    tmp_path,
):
    profile_path = _write_live_profile(tmp_path)
    config = _live_config(tmp_path, profile_path)
    driver_factory_calls = 0

    @contextmanager
    def failing_driver_factory(_profile, _remaining_s):
        nonlocal driver_factory_calls
        driver_factory_calls += 1
        raise RuntimeError("explicit preflight driver sentinel")
        yield  # pragma: no cover - contextmanager requires a generator

    runtime = create_runtime(
        config,
        v11_live_driver_factory=failing_driver_factory,
    )
    try:
        assert driver_factory_calls == 0
        assert runtime.v11_service.runtime_health()["status"] == "ok"
        assert (
            runtime.v11_service.runtime_capabilities()["hardware_mutation"]
            is True
        )
        assert driver_factory_calls == 0

        with pytest.raises(
            ReadOnlyPreflightError,
            match="explicit preflight driver sentinel",
        ):
            runtime.v11_service.hardware_snapshot()
        assert driver_factory_calls == 1
    finally:
        runtime.shutdown()


def test_formal_live_runtime_environment_path_builds_fastapi_app(
    tmp_path,
    monkeypatch,
):
    from vcsel_ai_tuning.api import app as app_module

    profile_path = _write_live_profile(tmp_path)
    runtime_config_path = tmp_path / "runtime.live.yaml"
    runtime_config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE_TUNING",
                "api": {"host": "127.0.0.1", "port": 8765},
                "storage": {
                    "sqlite_path": "state/campaigns.sqlite3",
                    "artifact_root": "artifacts",
                },
                "b0_path": str(
                    REPO_ROOT / "configs" / "ai_tuning" / "b0.yaml"
                ),
                "hardware": {
                    "driver_profile": str(profile_path),
                    "dca_visa_resource": (
                        "TCPIP0::example-dca::hislip0,4880::INSTR"
                    ),
                    "i2c_com_port": "COM42",
                    "i2c_address": 0xA4,
                    "bert_host": "192.0.2.10",
                    "bert_port": 9100,
                    "resource_ids": ["live-bench"],
                },
                "measurement_timeout_s": 30.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    created_runtimes = []
    real_create_runtime = runtime_module.create_runtime

    def capture_runtime(config):
        runtime = real_create_runtime(config)
        created_runtimes.append(runtime)
        return runtime

    captured = {}
    monkeypatch.delenv("VCSEL_AI_CAMPAIGN_SERVICE_FACTORY", raising=False)
    monkeypatch.setenv("VCSEL_AI_RUNTIME_CONFIG", str(runtime_config_path))
    monkeypatch.setattr(runtime_module, "create_runtime", capture_runtime)
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("formal LIVE startup opened a socket")
        ),
    )
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, host, port: captured.update(
            {"app": app, "host": host, "port": port}
        ),
    )
    try:
        with pytest.raises(SystemExit) as raised:
            app_module.main()

        assert raised.value.code == 0
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 8765
        paths = set(captured["app"].openapi()["paths"])
        assert "/api/v1.1/tasks" in paths
        assert (
            created_runtimes[0].v11_service.gateway.runtime_mode
            is V11RuntimeMode.LIVE
        )
    finally:
        for runtime in created_runtimes:
            runtime.shutdown()


def test_fake_and_read_only_modes_reject_real_sweep_factory(tmp_path):
    marker = lambda _b0, _plan: None
    fake_config = RuntimeConfig(
        storage=StorageConfig(
            sqlite_path=tmp_path / "fake" / "campaigns.sqlite3",
            artifact_root=tmp_path / "fake" / "artifacts",
        ),
        b0_path=REPO_ROOT / "configs" / "ai_tuning" / "b0.yaml",
    )
    with pytest.raises(RuntimeAssemblyError, match="LIVE_TUNING"):
        create_runtime(fake_config, v11_sweep_executor_factory=marker)

    profile_path = _write_live_profile(tmp_path)
    with pytest.raises(RuntimeAssemblyError, match="LIVE_TUNING"):
        create_runtime(
            _read_only_config(tmp_path, profile_path),
            v11_sweep_executor_factory=marker,
        )


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
