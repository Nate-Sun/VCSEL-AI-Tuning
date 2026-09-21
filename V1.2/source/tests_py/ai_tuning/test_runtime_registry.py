from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import vcsel_ai_tuning.v1_1.fake_ai_execution as fake_ai_execution_module
import vcsel_ai_tuning.v1_1.service as v11_service_module

from vcsel_ai_tuning.api.v1_1_schemas import (
    AuthorizeTaskRequest,
    CreateTaskRequest,
    SaveB0Request,
)
from vcsel_ai_tuning.campaign.models import Campaign, CampaignState
from vcsel_ai_tuning.campaign.store import CampaignStore, StoreClosedError
from vcsel_ai_tuning.runtime import (
    AdapterBundle,
    CampaignEntry,
    CampaignRuntime,
    HardwareResourceBusyError,
    RuntimeClosedError,
    create_fake_runtime,
    create_runtime,
)
from vcsel_ai_tuning.campaign.authorization import canonical_json_sha256
from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.runtime_config import RuntimeConfig
from vcsel_ai_tuning.hardware_authority import HardwareAuthority
from vcsel_ai_tuning.v1_1.domain import (
    LifecycleState,
    RuntimeMode as V11RuntimeMode,
    TaskType,
)
from vcsel_ai_tuning.v1_1.optimizer_bridge import OptimizerInterfaceUnavailable
from vcsel_ai_tuning.v1_1.repository import (
    ActiveTaskConflict,
    TaskStore,
    TaskStoreError,
)
from vcsel_ai_tuning.v1_1.service import V11ApplicationService


@dataclass
class CloseSpy:
    closed: int = 0

    def close(self) -> None:
        self.closed += 1


def _campaign(campaign_id: str) -> Campaign:
    timestamp = datetime(2026, 7, 24, tzinfo=timezone.utc)
    return Campaign(
        campaign_id=campaign_id,
        baseline_id="B0",
        baseline_hash="a" * 64,
        config_hash="b" * 64,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _config(tmp_path: Path, *, worker_count: int = 2) -> RuntimeConfig:
    return RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts",
        },
        worker_count=worker_count,
    )


def _entry(
    campaign_id: str,
    *,
    repository: CloseSpy | None = None,
    adapters: tuple[CloseSpy, CloseSpy, CloseSpy] | None = None,
    resource_ids: tuple[str, ...] = ("bench-1",),
) -> CampaignEntry:
    repository = repository or CloseSpy()
    adapter_values = adapters or (CloseSpy(), CloseSpy(), CloseSpy())
    return CampaignEntry(
        campaign_id=campaign_id,
        repository=repository,
        adapters=AdapterBundle(*adapter_values),
        orchestrator=object(),
        resource_ids=resource_ids,
    )


def test_runtime_caches_one_dependency_entry_per_campaign(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    store.create_campaign(_campaign("campaign-1"))
    created: list[str] = []

    def factory(campaign_id: str) -> CampaignEntry:
        created.append(campaign_id)
        return _entry(campaign_id)

    runtime = CampaignRuntime(_config(tmp_path), store=store, entry_factory=factory)

    first = runtime.entry("campaign-1")
    second = runtime.entry("campaign-1")

    assert first is second
    assert created == ["campaign-1"]


def test_runtime_restart_fences_mutating_state_and_does_not_auto_resume(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    store.create_campaign(_campaign("active"))
    store.save_transition("active", CampaignState.DRAFT, CampaignState.PREFLIGHT)
    store.save_transition(
        "active",
        CampaignState.PREFLIGHT,
        CampaignState.AWAIT_BASELINE_APPROVAL,
    )
    store.save_transition(
        "active",
        CampaignState.AWAIT_BASELINE_APPROVAL,
        CampaignState.RESTORE_B0,
    )
    store.create_campaign(_campaign("safe"))
    runtime = CampaignRuntime(
        _config(tmp_path),
        store=store,
        entry_factory=lambda campaign_id: _entry(campaign_id),
    )

    recovered = runtime.recover()

    assert recovered == {
        "active": CampaignState.RECOVERY_REQUIRED,
        "safe": CampaignState.DRAFT,
    }
    assert store.current_state("active") is CampaignState.RECOVERY_REQUIRED
    assert [event.event_type for event in store.events("active")][-1] == (
        "RESTART_RECOVERY"
    )


def test_background_worker_persists_success_and_failure_events(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    store.create_campaign(_campaign("campaign-1"))
    runtime = CampaignRuntime(
        _config(tmp_path, worker_count=1),
        store=store,
        entry_factory=lambda campaign_id: _entry(campaign_id),
    )

    success = runtime.submit_stage("campaign-1", "TEST_SUCCESS", lambda _entry: 42)
    failure = runtime.submit_stage(
        "campaign-1",
        "TEST_FAILURE",
        lambda _entry: (_ for _ in ()).throw(ValueError("injected worker error")),
    )

    assert success.result(timeout=5) == 42
    with pytest.raises(ValueError, match="injected worker error"):
        failure.result(timeout=5)
    events = store.events("campaign-1")
    worker_events = [event for event in events if event.event_type.startswith("WORKER_")]
    assert [event.event_type for event in worker_events] == [
        "WORKER_STARTED",
        "WORKER_COMPLETED",
        "WORKER_STARTED",
        "WORKER_FAILED",
    ]
    assert worker_events[-1].payload == {
        "error_type": "ValueError",
        "operation": "TEST_FAILURE",
    }


def test_two_campaigns_cannot_use_the_same_hardware_resource_concurrently(
    tmp_path: Path,
):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    store.create_campaign(_campaign("campaign-1"))
    store.create_campaign(_campaign("campaign-2"))
    entered = Event()
    release = Event()
    runtime = CampaignRuntime(
        _config(tmp_path, worker_count=2),
        store=store,
        entry_factory=lambda campaign_id: _entry(
            campaign_id,
            resource_ids=("shared-bench",),
        ),
    )

    def blocking(_entry: CampaignEntry) -> str:
        entered.set()
        assert release.wait(timeout=5)
        return "released"

    first = runtime.submit_stage("campaign-1", "BLOCK", blocking)
    assert entered.wait(timeout=5)
    second = runtime.submit_stage("campaign-2", "CONTEND", lambda _entry: "unsafe")

    with pytest.raises(HardwareResourceBusyError, match="shared-bench"):
        second.result(timeout=5)
    release.set()
    assert first.result(timeout=5) == "released"
    third = runtime.submit_stage("campaign-2", "AFTER_RELEASE", lambda _entry: "ok")
    assert third.result(timeout=5) == "ok"


def test_v1_stage_and_v11_start_cannot_both_enter_hardware(tmp_path: Path):
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1.1").mkdir()
    store = CampaignStore(tmp_path / "v1" / "campaigns.sqlite3")
    store.create_campaign(_campaign("campaign-1"))
    entered = Event()
    release = Event()
    runtime = CampaignRuntime(
        _config(tmp_path / "v1", worker_count=1),
        store=store,
        entry_factory=lambda campaign_id: _entry(campaign_id),
        hardware_authority=authority,
    )
    v11_store = TaskStore(tmp_path / "v1.1" / "campaigns.sqlite3")
    v11_store.create_task(
        task_id="task-1",
        task_type=TaskType.AI_OPTIMIZATION,
        runtime_mode=V11RuntimeMode.FAKE,
        config_fingerprint="f" * 64,
        payload={"task_id": "task-1"},
    )
    v11_store.transition(
        "task-1",
        expected_state=LifecycleState.DRAFT,
        new_state=LifecycleState.AUTHORIZED,
        event_type="TEST_AUTHORIZED",
    )
    runner = SimpleNamespace(calls=[])
    runner.start = lambda task_id, task_type: runner.calls.append(
        ("start", task_id, task_type)
    )
    v11 = V11ApplicationService(
        store=v11_store,
        gateway=SimpleNamespace(runtime_mode=V11RuntimeMode.FAKE),
        runner=runner,
        hardware_authority=authority,
    )
    v11._verify_start_attestation = lambda _task_id: None

    def blocking(_entry: CampaignEntry) -> None:
        entered.set()
        assert release.wait(timeout=5)

    future = runtime.submit_stage("campaign-1", "BLOCK", blocking)
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(ActiveTaskConflict) as caught:
            v11.start_task("task-1")
        assert caught.value.active_task_id.startswith("v1:campaign-1:BLOCK")
        assert runner.calls == []
    finally:
        release.set()
        future.result(timeout=5)

    assert v11.start_task("task-1")["state"] == "RUNNING"
    assert runner.calls == [("start", "task-1", TaskType.AI_OPTIMIZATION)]
    v11.close()
    v11_store.close()
    runtime.close()


def test_runtime_shutdown_is_idempotent_and_releases_owned_resources(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    store.create_campaign(_campaign("campaign-1"))
    repository = CloseSpy()
    adapters = (CloseSpy(), CloseSpy(), CloseSpy())
    runtime = CampaignRuntime(
        _config(tmp_path),
        store=store,
        entry_factory=lambda campaign_id: _entry(
            campaign_id,
            repository=repository,
            adapters=adapters,
        ),
    )
    runtime.entry("campaign-1")

    runtime.close()
    runtime.close()

    assert repository.closed == 1
    assert [adapter.closed for adapter in adapters] == [1, 1, 1]
    with pytest.raises(StoreClosedError):
        store.current_state("campaign-1")
    with pytest.raises(RuntimeClosedError):
        runtime.submit_stage("campaign-1", "AFTER_CLOSE", lambda _entry: None)


def test_builtin_runtime_automatically_fences_active_state_on_restart(tmp_path: Path):
    config = _config(tmp_path)
    config = config.model_copy(
        update={"b0_path": Path("configs/ai_tuning/b0.yaml").resolve()}
    )
    first_api_runtime = create_fake_runtime(config)
    first_runtime = first_api_runtime.status_provider
    assert isinstance(first_runtime, CampaignRuntime)
    baseline_hash = canonical_json_sha256(load_b0(config.b0_path))
    timestamp = datetime(2026, 7, 24, tzinfo=timezone.utc)
    first_runtime.store.create_campaign(
        Campaign(
            campaign_id="interrupted",
            baseline_id="B0",
            baseline_hash=baseline_hash,
            config_hash="b" * 64,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    first_runtime.store.save_transition(
        "interrupted",
        CampaignState.DRAFT,
        CampaignState.PREFLIGHT,
    )
    first_runtime.store.save_transition(
        "interrupted",
        CampaignState.PREFLIGHT,
        CampaignState.AWAIT_BASELINE_APPROVAL,
    )
    first_runtime.store.save_transition(
        "interrupted",
        CampaignState.AWAIT_BASELINE_APPROVAL,
        CampaignState.RESTORE_B0,
    )
    first_runtime.close()

    restarted_api_runtime = create_fake_runtime(config)
    restarted = restarted_api_runtime.status_provider
    assert isinstance(restarted, CampaignRuntime)
    try:
        assert restarted.store.current_state("interrupted") is (
            CampaignState.RECOVERY_REQUIRED
        )
        assert [
            event.event_type for event in restarted.store.events("interrupted")
        ][-1] == "RESTART_RECOVERY"
    finally:
        restarted.close()


def test_runtime_assembly_injects_one_common_hardware_authority(tmp_path: Path):
    state_root = tmp_path / "state"
    config = RuntimeConfig(
        storage={
            "sqlite_path": state_root / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
    )
    history_hardware = object()
    api_runtime = create_runtime(
        config,
        v11_history_hardware=history_hardware,
    )
    runtime = api_runtime.status_provider
    v11 = api_runtime.v11_service
    assert isinstance(runtime, CampaignRuntime)
    assert isinstance(v11, V11ApplicationService)
    try:
        authority = runtime._hardware_authority
        assert v11._hardware_authority is authority
        assert v11.history_apply_controller._hardware_authority is authority
        assert authority.path == (state_root / "hardware-authority.lock").resolve()
    finally:
        api_runtime.shutdown()


def test_default_fake_runtime_v11_ai_execution_is_ready(tmp_path: Path):
    config = RuntimeConfig(
        storage={
            "sqlite_path": (
                tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3"
            ),
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
    )
    api_runtime = create_fake_runtime(config)
    try:
        capabilities = api_runtime.v11_service.runtime_capabilities()
        runner_capabilities = api_runtime.v11_service.runner.capabilities()

        assert capabilities["runtime_mode"] == "FAKE"
        assert capabilities["hardware_mutation"] is False
        assert capabilities["sweep_execution_ready"] is True
        assert capabilities["ai_execution_ready"] is True
        assert capabilities["hardware_connected"] is False
        assert runner_capabilities == {
            "sweep_execution_ready": True,
            "ai_execution_ready": True,
        }
    finally:
        api_runtime.shutdown()


def test_v11_ai_execution_factory_failure_keeps_default_fake_runtime_unavailable(
    tmp_path: Path,
    monkeypatch,
):
    config = RuntimeConfig(
        storage={
            "sqlite_path": (
                tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3"
            ),
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
    )
    build_calls = []

    def unavailable_factory(*, database_path, artifact_root):
        build_calls.append((database_path, artifact_root))
        raise OptimizerInterfaceUnavailable("injected optimizer interface failure")

    monkeypatch.setattr(
        fake_ai_execution_module,
        "build_fake_ai_execution_factory",
        unavailable_factory,
    )

    api_runtime = create_fake_runtime(config)
    service = api_runtime.v11_service
    assert service is not None
    try:
        expected_database = (
            config.storage.sqlite_path.parent.parent
                / "v1.2"
            / "campaigns.sqlite3"
        )
        expected_artifacts = config.storage.artifact_root.parent / "v1.2"
        assert build_calls == [(expected_database, expected_artifacts)]
        assert service.runtime_capabilities()["sweep_execution_ready"] is True
        assert service.runtime_capabilities()["ai_execution_ready"] is False

        task_id = "optimizer-interface-unavailable"
        service.create_task(
            CreateTaskRequest.model_validate(
                {
                    "task_id": task_id,
                    "task_type": "AI_OPTIMIZATION",
                    "channel_mapping": {
                        "i2c_channel": 0,
                        "bert_port": "port1",
                        "bert_channel": "0",
                        "dca_tdecq_source": "FUNC1",
                        "dca_tdecq_equalizer_function": 1,
                        "dca_outer_er_source": "CHAN1A",
                        "operator_confirmed": True,
                    },
                    "parameters": [
                        {"name": "eq", "min": 20, "max": 22, "step": 1}
                    ],
                    "candidate_budget": 3,
                    "seed": 17,
                }
            )
        )
        snapshot = service.hardware_snapshot()
        service.save_b0(
            task_id,
            SaveB0Request(
                snapshot_token=snapshot["snapshot_token"],
                operator_confirmed=True,
            ),
        )
        service.authorize_task(
            task_id,
            AuthorizeTaskRequest(
                authorization_id=f"auth-{task_id}",
                expires_at=(
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                ).isoformat(),
                operator_confirmed=True,
            ),
        )

        with pytest.raises(TaskStoreError, match="AI execution backend"):
            service.start_task(task_id)

        assert service.get_task(task_id)["state"] == "AUTHORIZED"
        assert service.store.observations(task_id) == ()
    finally:
        api_runtime.shutdown()


def test_v11_ai_execution_unexpected_factory_failure_releases_runtime_resources(
    tmp_path: Path,
    monkeypatch,
):
    config = RuntimeConfig(
        storage={
            "sqlite_path": (
                tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3"
            ),
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
    )
    real_builder = fake_ai_execution_module.build_fake_ai_execution_factory
    real_runtime_close = CampaignRuntime.close
    build_count = 0
    closed_runtimes = []

    def track_runtime_close(runtime):
        closed_runtimes.append(runtime)
        real_runtime_close(runtime)

    def fail_once(*, database_path, artifact_root):
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            raise RuntimeError("injected factory programming error")
        return real_builder(
            database_path=database_path,
            artifact_root=artifact_root,
        )

    monkeypatch.setattr(
        fake_ai_execution_module,
        "build_fake_ai_execution_factory",
        fail_once,
    )
    monkeypatch.setattr(CampaignRuntime, "close", track_runtime_close)

    with pytest.raises(RuntimeError, match="factory programming error"):
        create_fake_runtime(config)

    assert len(closed_runtimes) == 1
    api_runtime = create_fake_runtime(config)
    try:
        assert build_count == 2
        assert api_runtime.v11_service.runtime_capabilities()[
            "ai_execution_ready"
        ] is True
    finally:
        api_runtime.shutdown()

    database_path = config.storage.sqlite_path
    owner_lock_path = database_path.with_suffix(
        database_path.suffix + ".runtime.lock"
    )
    database_path.unlink()
    owner_lock_path.unlink()


def test_runtime_assembly_failure_closes_unowned_v11_runner(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path).model_copy(
        update={"b0_path": Path("configs/ai_tuning/b0.yaml").resolve()}
    )
    runner = CloseSpy()

    def fail_service_construction(**_kwargs):
        raise RuntimeError("injected V1.1 service construction failure")

    monkeypatch.setattr(
        v11_service_module,
        "V11ApplicationService",
        fail_service_construction,
    )

    with pytest.raises(RuntimeError, match="service construction failure"):
        create_runtime(
            config,
            v11_gateway=object(),
            v11_runner=runner,
        )

    assert runner.closed == 1


def test_second_runtime_cannot_recover_a_store_owned_by_a_live_process(
    tmp_path: Path,
):
    config = _config(tmp_path).model_copy(
        update={"b0_path": Path("configs/ai_tuning/b0.yaml").resolve()}
    )
    first_api_runtime = create_fake_runtime(config)
    first = first_api_runtime.status_provider
    assert isinstance(first, CampaignRuntime)
    first.store.create_campaign(_campaign("live-owner"))
    first.store.save_transition(
        "live-owner",
        CampaignState.DRAFT,
        CampaignState.PREFLIGHT,
    )
    state_before = first.store.current_state("live-owner")

    try:
        with pytest.raises(
            HardwareResourceBusyError,
            match="runtime store is already owned",
        ):
            create_fake_runtime(config)
        assert first.store.current_state("live-owner") is state_before
    finally:
        first.close()
