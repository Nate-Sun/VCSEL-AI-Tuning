from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Barrier, Event, Lock, Thread

import numpy as np
import pytest
import vcsel_ai_tuning.v1_1.fake_ai_execution as fake_ai_execution_module
import vcsel_ai_tuning.v1_1.persistent_runner as persistent_runner_module

from vcsel_ai_tuning.api.v1_1_schemas import (
    AbortRequest,
    AuthorizeTaskRequest,
    CreateTaskRequest,
    RecoveryRequest,
    SaveB0Request,
)
from vcsel_ai_tuning.runtime import create_fake_runtime, create_runtime
from vcsel_ai_tuning.runtime_config import RuntimeConfig
from vcsel_ai_tuning.hardware_authority import HardwareAuthority
from vcsel_ai_tuning.v1_1.domain import LifecycleState, TaskType
from vcsel_ai_tuning.v1_1.fake_runtime import FakeV11Gateway, FakeV11TaskRunner
from vcsel_ai_tuning.v1_1.fake_ai_execution import (
    FakeOptimizationPlant,
    FakeV11AIExecution,
)
from vcsel_ai_tuning.v1_1.hardware_control import RecoveryRequired
from vcsel_ai_tuning.v1_1.dataset import decode_png_dimensions
from vcsel_ai_tuning.v1_1.persistent_runner import (
    AIExecution,
    FakeSweepExecutor,
    PersistentV11TaskRunner,
    build_sweep_plan,
)
from vcsel_ai_tuning.v1_1.repository import (
    TaskConcurrencyError,
    TaskStore,
    TaskStoreError,
)
from vcsel_ai_tuning.v1_1.service import V11ApplicationService
from vcsel_ai_tuning.v1_1.sweep import (
    SweepPointRecord,
    SweepPointStatus,
    SweepRunner,
)


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "v1" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "v1" / "artifacts",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )


def _authorize_task(service, *, task_id: str, task_type: str) -> None:
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "channel_mapping": {
            "i2c_channel": 0,
            "bert_port": "port1",
            "bert_channel": "0",
            "dca_tdecq_source": "FUNC1",
            "dca_tdecq_equalizer_function": 1,
            "dca_outer_er_source": "CHAN1A",
            "operator_confirmed": True,
        },
        "parameters": (
            {"name": "eq", "min": 20, "max": 22, "step": 1},
        ),
        "candidate_budget": 3,
        "seed": 17,
    }
    if task_type == "SWEEP":
        payload.update(
            sweep_hardware="I2C_SWEEP",
            sweep_mode="SINGLE_PARAMETER",
            max_points=3,
        )
    service.create_task(CreateTaskRequest.model_validate(payload))
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


@pytest.mark.parametrize(
    ("stored_value", "expected"),
    [(None, False), (True, True)],
)
def test_build_sweep_plan_freezes_dataset_option(stored_value, expected):
    payload = {
        "task_id": "dataset-plan",
        "sweep_hardware": "I2C_SWEEP",
        "sweep_mode": "SINGLE_PARAMETER",
        "seed": 17,
        "max_points": 1,
        "parameters": [{"name": "eq", "min": 20, "max": 20, "step": 1}],
    }
    if stored_value is not None:
        payload["save_dataset"] = stored_value
    b0 = FakeV11Gateway().preflight_capture().b0

    plan = build_sweep_plan(payload, b0)

    assert plan.campaign_id == "dataset-plan"
    assert plan.save_dataset is expected


def _dataset_plan(*, save_dataset: bool):
    b0 = FakeV11Gateway().preflight_capture().b0
    payload = {
        "task_id": "fake-dataset-campaign",
        "sweep_hardware": "I2C_SWEEP",
        "sweep_mode": "SINGLE_PARAMETER",
        "seed": 17,
        "max_points": 1,
        "parameters": [{"name": "eq", "min": 21, "max": 21, "step": 1}],
        "save_dataset": save_dataset,
    }
    return b0, build_sweep_plan(payload, b0)


def test_fake_sweep_dataset_disabled_creates_no_bundle_or_capture(tmp_path):
    b0, plan = _dataset_plan(save_dataset=False)
    capture_calls = []
    executor = FakeSweepExecutor(
        b0,
        plan,
        artifact_root=tmp_path,
        dataset_capture_hook=lambda *_args: capture_calls.append("capture"),
    )
    point = next(item for item in plan.points if item.status.value == "PLANNED")

    executor.apply_point(point)
    measured = executor.measure_point()

    assert measured["dataset_status"] == "DISABLED"
    assert capture_calls == []
    assert not (tmp_path / plan.campaign_id).exists()


def test_fake_sweep_dataset_enabled_writes_replay_bundle(tmp_path):
    b0, plan = _dataset_plan(save_dataset=True)
    executor = FakeSweepExecutor(b0, plan, artifact_root=tmp_path)
    point = next(item for item in plan.points if item.status.value == "PLANNED")

    executor.apply_point(point)
    measured = executor.measure_point()

    point_id = f"point-{point.original_index:08d}"
    bundle = tmp_path / plan.campaign_id / point_id
    assert measured["measurement_status"] == "SUCCESS"
    assert measured["dataset_status"] == "COMPLETE"
    assert set(path.name for path in bundle.iterdir()) == {
        "record.json",
        "raw_eye_density.npy",
        "raw_eye.png",
        "tdecq_equalized_eye.png",
    }
    density = np.load(bundle / "raw_eye_density.npy", allow_pickle=False)
    assert density.shape == (6, 8)
    assert density.dtype == np.dtype("float32")
    assert decode_png_dimensions(bundle / "raw_eye.png") == (8, 6)
    assert decode_png_dimensions(bundle / "tdecq_equalized_eye.png") == (8, 6)
    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert record["measurement_id"] == measured["measurement_id"]
    assert record["acquisition_id"] == measured["acquisition_id"]
    assert record["bert"] == {
        "port": b0.bert_port,
        "channel": b0.bert_channel,
        "pattern": b0.bert_pattern,
    }
    assert record["equalizer"]["taps"] == list(b0.fir_taps)


def test_constraint_rejected_point_never_creates_dataset_directory(tmp_path):
    b0, plan = _dataset_plan(save_dataset=True)
    rejected = replace(
        plan.points[0],
        status=SweepPointStatus.SKIPPED_CONSTRAINT,
        constraint_reason="injected safety rejection",
    )
    plan = replace(plan, points=(rejected,))
    executor = FakeSweepExecutor(b0, plan, artifact_root=tmp_path)

    class Sink:
        def __init__(self):
            self.saved = []

        def save_sweep_point(self, record):
            self.saved.append(record)

        def update_sweep_point(self, record):
            self.saved[0] = record

    sink = Sink()
    SweepRunner(executor=executor, store=sink).run(plan)

    assert sink.saved[0].measurement_status == "REJECTED"
    assert sink.saved[0].dataset_status == "DISABLED"
    assert not (tmp_path / plan.campaign_id).exists()
    assert all(call[0] != "dataset_capture" for call in executor.calls)


class _CandidateBoundaryPlant(FakeOptimizationPlant):
    """Block only the first candidate measurement, never optimizer selection."""

    def __init__(
        self,
        b0,
        *,
        block_first_candidate: bool = True,
        fail_restore: bool = False,
    ) -> None:
        super().__init__(b0)
        self.block_first_candidate = block_first_candidate
        self.fail_restore = fail_restore
        self.candidate_measurement_started = Event()
        self.allow_candidate_measurement = Event()
        self.restore_calls = 0
        self.measure_calls = 0
        self.mutations: list[tuple[str, object]] = []
        self.constructed_at_b0 = self.is_at_b0()

    def apply_i2c(self, requested):
        result = super().apply_i2c(requested)
        self.mutations.append(("I2C", dict(result)))
        return result

    def apply_fir(self, target):
        result = super().apply_fir(target)
        self.mutations.append(("FIR", tuple(result)))
        return result

    def measure(self):
        self.measure_calls += 1
        if self.block_first_candidate and self.measure_calls == 2:
            self.candidate_measurement_started.set()
            if not self.allow_candidate_measurement.wait(timeout=5):
                raise TimeoutError("candidate boundary gate was not released")
        return super().measure()

    def restore_b0(self) -> None:
        self.restore_calls += 1
        if self.fail_restore:
            raise RuntimeError("injected real FAKE B0 restore failure")
        super().restore_b0()


class _StopObservedEvent(Event):
    def __init__(self) -> None:
        super().__init__()
        self.set_called = Event()

    def set(self) -> None:
        super().set()
        self.set_called.set()


def _observe_stop_request(runner: PersistentV11TaskRunner) -> _StopObservedEvent:
    stop = _StopObservedEvent()
    runner._stop_requested = stop
    return stop


def _real_fake_ai_factory(
    database_path: Path,
    artifact_root: Path,
    *,
    block_first_execution: bool = True,
    fail_restore: bool = False,
):
    executions: list[FakeV11AIExecution] = []
    plants: list[_CandidateBoundaryPlant] = []

    def factory(task_id, b0, payload):
        plant = _CandidateBoundaryPlant(
            b0,
            block_first_candidate=block_first_execution and not plants,
            fail_restore=fail_restore,
        )
        plants.append(plant)
        execution = FakeV11AIExecution(
            task_id,
            b0,
            payload,
            database_path,
            artifact_root,
            plant_factory=lambda _b0: plant,
        )
        executions.append(execution)
        return execution

    return factory, executions, plants


def _authorize_fir_constraint_sweep(service, *, task_id: str) -> None:
    request = CreateTaskRequest.model_validate(
        {
            "task_id": task_id,
            "task_type": "SWEEP",
            "channel_mapping": {
                "i2c_channel": 0,
                "bert_port": "port1",
                "bert_channel": "0",
                "dca_tdecq_source": "FUNC1",
                "dca_tdecq_equalizer_function": 1,
                "dca_outer_er_source": "CHAN1A",
                "operator_confirmed": True,
            },
            "parameters": (
                {"name": "main", "min": -1, "max": 1, "step": 1},
            ),
            "candidate_budget": 3,
            "sweep_hardware": "BERT_FIR_SWEEP",
            "sweep_mode": "SINGLE_PARAMETER",
            "seed": 23,
            "max_points": 3,
        }
    )
    service.create_task(request)
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


def test_default_fake_runtime_executes_sweep_to_persistent_completion(
    tmp_path: Path,
):
    config = _config(tmp_path)
    runtime = create_fake_runtime(config)
    service = runtime.v11_service
    assert service is not None
    try:
        request = CreateTaskRequest.model_validate(
            {
                "task_id": "fake-sweep",
                "task_type": "SWEEP",
                "channel_mapping": {
                    "i2c_channel": 0,
                    "bert_port": "port1",
                    "bert_channel": "0",
                    "dca_tdecq_source": "FUNC1",
                    "dca_tdecq_equalizer_function": 1,
                    "dca_outer_er_source": "CHAN1A",
                    "operator_confirmed": True,
                },
                "parameters": (
                    {"name": "eq", "min": 20, "max": 22, "step": 1},
                ),
                "candidate_budget": 3,
                "sweep_hardware": "I2C_SWEEP",
                "sweep_mode": "SINGLE_PARAMETER",
                "seed": 17,
                "max_points": 3,
            }
        )
        service.create_task(request)
        snapshot = service.hardware_snapshot()
        service.save_b0(
            "fake-sweep",
            SaveB0Request(
                snapshot_token=snapshot["snapshot_token"],
                operator_confirmed=True,
            ),
        )
        service.authorize_task(
            "fake-sweep",
            AuthorizeTaskRequest(
                authorization_id="auth-fake-sweep",
                expires_at=(
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                ).isoformat(),
                operator_confirmed=True,
            ),
        )

        assert service.start_task("fake-sweep")["state"] == "RUNNING"
        assert service.runner.wait_for_idle(timeout=5)

        assert service.get_task("fake-sweep")["state"] == "COMPLETED"
        observations = service.store.observations("fake-sweep")
        assert [(item.sequence, item.phase) for item in observations] == [(1, "B0")]
        points = service.store.sweep_points("fake-sweep", limit=10).items
        assert [item.original_index for item in points] == [0, 1, 2]
        assert all(item.status.value == "COMPLETED" for item in points)
        assert service.store.active_task_id() is None

        artifact_directory = service.artifact_root / "fake-sweep"
        assert (artifact_directory / "sweep_results.json").is_file()
        assert (artifact_directory / "sweep_results.csv").is_file()
    finally:
        runtime.shutdown()


class _SuccessfulAIExecution:
    def __init__(self, ran: Event) -> None:
        self._ran = ran
        self.restore_calls = 0

    def run(self, _stop_requested: Event):
        self._ran.set()
        return LifecycleState.COMPLETED

    def restore_b0(self) -> None:
        self.restore_calls += 1


class _ClosableAIExecution(_SuccessfulAIExecution):
    def __init__(self, ran: Event) -> None:
        super().__init__(ran)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _TerminalB0VerifiedAIExecution(_ClosableAIExecution):
    terminal_b0_verified = True

    def restore_b0(self) -> None:  # pragma: no cover - one-attempt assertion
        raise AssertionError("verified terminal B0 must not be restored twice")


class _AlreadyRolledBackAIExecution(_ClosableAIExecution):
    def run(self, _stop_requested: Event):
        error = RuntimeError("candidate failed after verified rollback")
        error.rollback_verified = True
        error.evidence = {"watchdog_status": "COMPLETED", "request_id": "dca-7"}
        raise error

    def restore_b0(self) -> None:  # pragma: no cover - one-attempt assertion
        raise AssertionError("verified rollback must not be repeated")


class _CooperativeAIExecution:
    def __init__(self, *, started: Event, wait_for_stop: bool) -> None:
        self.started = started
        self.wait_for_stop = wait_for_stop
        self.restore_calls = 0

    def run(self, stop_requested: Event):
        self.started.set()
        if self.wait_for_stop:
            assert stop_requested.wait(timeout=5)
            return None
        return LifecycleState.COMPLETED

    def restore_b0(self) -> None:
        self.restore_calls += 1


class _LegacyAIExecution:
    def run(self):
        return LifecycleState.COMPLETED

    def restore_b0(self) -> None:
        pass


class _TerminalOnStopAIExecution:
    def __init__(self, started: Event) -> None:
        self.started = started
        self.restore_calls = 0

    def run(self, stop_requested: Event):
        self.started.set()
        assert stop_requested.wait(timeout=5)
        return LifecycleState.COMPLETED

    def restore_b0(self) -> None:
        self.restore_calls += 1


class _LateAbortAIExecution:
    def __init__(self, *, started: Event, allow_return: Event) -> None:
        self.started = started
        self.allow_return = allow_return
        self.restore_calls = 0

    def run(self, stop_requested: Event):
        self.started.set()
        assert not stop_requested.is_set()
        assert self.allow_return.wait(timeout=5)
        assert not stop_requested.is_set()
        return LifecycleState.COMPLETED

    def restore_b0(self) -> None:
        self.restore_calls += 1


class _LateAbortRecoveryFailureExecution:
    def __init__(
        self,
        *,
        started: Event,
        allow_return: Event,
        direct_recovery: bool,
    ) -> None:
        self.started = started
        self.allow_return = allow_return
        self.direct_recovery = direct_recovery
        self.restore_calls = 0

    def run(self, stop_requested: Event):
        self.started.set()
        assert self.allow_return.wait(timeout=5)
        assert not stop_requested.is_set()
        if self.direct_recovery:
            raise RecoveryRequired(
                "injected recovery required",
                rollback_verified=False,
            )
        raise RuntimeError("injected AI run failure")

    def restore_b0(self) -> None:
        self.restore_calls += 1
        raise RuntimeError("injected AI restore failure")


class _AIResultExecution:
    def __init__(self, result: object) -> None:
        self.result = result
        self.restore_calls = 0

    def run(self, _stop_requested: Event):
        return self.result

    def restore_b0(self) -> None:
        self.restore_calls += 1


class _ArtifactAwareAIExecution(_AIResultExecution):
    def __init__(
        self,
        *,
        database_path: Path,
        task_id: str,
        refresh_error: Exception | None = None,
    ) -> None:
        super().__init__(LifecycleState.COMPLETED)
        self.database_path = database_path
        self.task_id = task_id
        self.refresh_error = refresh_error
        self.artifact_refreshes = []

    def refresh_artifact(self, terminal_lifecycle=None) -> None:
        with TaskStore(self.database_path) as store:
            durable_state = store.require_task(self.task_id).state
        self.artifact_refreshes.append((terminal_lifecycle, durable_state))
        if self.refresh_error is not None:
            raise self.refresh_error


class _RecoveryAIExecution:
    def __init__(self, *, restore_error: Exception | None = None) -> None:
        self.restore_error = restore_error
        self.run_calls = 0
        self.restore_calls = 0

    def run(self, _stop_requested: Event):
        self.run_calls += 1
        return LifecycleState.COMPLETED

    def restore_b0(self) -> None:
        self.restore_calls += 1
        if self.restore_error is not None:
            raise self.restore_error


class _LateAbortSweepRollbackFailureExecutor(FakeSweepExecutor):
    def __init__(self, b0, plan, *, started: Event, allow_return: Event):
        super().__init__(b0, plan)
        self.started = started
        self.allow_return = allow_return
        self.restore_calls = 0

    def apply_point(self, point):
        self.started.set()
        assert self.allow_return.wait(timeout=5)
        raise RuntimeError("injected sweep point failure")

    def restore_b0(self) -> None:
        self.restore_calls += 1
        raise RuntimeError("injected sweep rollback failure")


class _LateSignalAbortRunner(PersistentV11TaskRunner):
    def __init__(self, *args, allow_return: Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.allow_return = allow_return

    def abort_and_restore_b0(self, task_id: str) -> None:
        with self._lock:
            future = self._future
            if future is None:
                raise TaskStoreError(f"task {task_id} is not running")
        self.allow_return.set()
        future.result(timeout=5)
        self._stop_requested.set()


def test_injected_fake_ai_execution_reaches_terminal_state(tmp_path: Path):
    assert AIExecution.__protocol_attrs__ == {"run", "restore_b0"}
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ran = Event()
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=lambda _task_id, _b0, _payload: (
            _SuccessfulAIExecution(ran)
        ),
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="fake-ai",
            task_type="AI_OPTIMIZATION",
        )

        assert service.start_task("fake-ai")["state"] == "RUNNING"
        assert ran.wait(timeout=5)
        assert runner.wait_for_idle(timeout=5)
        assert service.get_task("fake-ai")["state"] == "COMPLETED"
        assert service.store.active_task_id() is None
    finally:
        runtime.shutdown()


def test_verified_terminal_b0_skips_duplicate_runner_restore(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent / "v1.1" / "campaigns.sqlite3"
    )
    executions = []

    def factory(_task_id, _b0, _payload):
        execution = _TerminalB0VerifiedAIExecution(Event())
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="terminal-b0-verified",
            task_type="AI_OPTIMIZATION",
        )
        assert service.start_task("terminal-b0-verified")["state"] == "RUNNING"
        assert runner.wait_for_idle(timeout=5)
        assert service.get_task("terminal-b0-verified")["state"] == "COMPLETED"
        assert executions[0].close_calls == 1
    finally:
        runtime.shutdown()


def test_cancelled_unpublished_ai_preparation_is_closed_once(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent / "v1.1" / "campaigns.sqlite3"
    )
    executions = []

    def factory(_task_id, _b0, _payload):
        execution = _ClosableAIExecution(Event())
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="prepared-close", task_type="AI_OPTIMIZATION")
        runner.start("prepared-close", TaskType.AI_OPTIMIZATION)

        runner.cancel_prepared("prepared-close")
        runner.cancel_prepared("prepared-close")

        assert executions[0].close_calls == 1
    finally:
        runtime.shutdown()


def test_ai_execution_is_closed_once_after_worker_terminal(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent / "v1.1" / "campaigns.sqlite3"
    )
    executions = []

    def factory(_task_id, _b0, _payload):
        execution = _ClosableAIExecution(Event())
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="terminal-close", task_type="AI_OPTIMIZATION")
        service.start_task("terminal-close")
        assert runner.wait_for_idle(timeout=5)

        assert executions[0].close_calls == 1
    finally:
        runtime.shutdown()
        assert executions[0].close_calls == 1


def test_runner_close_closes_prepared_ai_once(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent / "v1.1" / "campaigns.sqlite3"
    )
    executions = []

    def factory(_task_id, _b0, _payload):
        execution = _ClosableAIExecution(Event())
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    _authorize_task(service, task_id="shutdown-prepared", task_type="AI_OPTIMIZATION")
    runner.start("shutdown-prepared", TaskType.AI_OPTIMIZATION)

    runner.close()
    runner.close()

    assert executions[0].close_calls == 1
    service.close()


def test_runner_prepare_revalidates_persisted_staged_policy(tmp_path: Path):
    database_path = tmp_path / "campaigns.sqlite3"
    payload = {
        "task_id": "invalid-prepare",
        "task_type": "AI_OPTIMIZATION",
        "live_ai_profile": "PILOT_3_THEN_FAST24",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        "depth": "FAST",
        "candidate_budget": 27,
        "pilot_candidate_budget": 3,
        "fast_candidate_budget": 24,
        "coarse_candidate_budget": 6,
        "fine_candidate_budget": 15,
        "fir_first_candidate_budget": 1,
        "fir_iterative_candidate_budget": 1,
        "settle_time_s": 0.1,
        "save_dataset": True,
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
    }
    b0 = FakeV11Gateway().preflight_capture().b0
    with TaskStore(database_path) as store:
        store.create_task(
            task_id="invalid-prepare",
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=FakeV11Gateway().runtime_mode,
            config_fingerprint="c" * 64,
            payload=payload,
        )
        store.save_b0("invalid-prepare", b0)
    constructed = []
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=tmp_path / "artifacts",
        ai_execution_factory=lambda *_args: constructed.append(object()),
    )
    try:
        with pytest.raises(ValueError, match="fine_candidate_budget"):
            runner.start("invalid-prepare", TaskType.AI_OPTIMIZATION)
        assert constructed == []
    finally:
        runner.close()


def test_verified_execution_rollback_is_not_repeated_and_evidence_is_durable(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent / "v1.1" / "campaigns.sqlite3"
    )
    executions = []

    def factory(_task_id, _b0, _payload):
        execution = _AlreadyRolledBackAIExecution(Event())
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="verified-error-rollback",
            task_type="AI_OPTIMIZATION",
        )
        service.start_task("verified-error-rollback")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("verified-error-rollback")["state"] == "FAILED"
        terminal = service.store.events("verified-error-rollback")[-1]
        assert terminal.event_type == "AI_EXECUTION_FAILED_B0_RESTORED"
        assert terminal.payload["rollback_verified"] is True
        assert terminal.payload["evidence"] == {
            "watchdog_status": "COMPLETED",
            "request_id": "dca-7",
        }
        assert executions[0].close_calls == 1
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("action", ["RESUME", "ABORT_RECOVERED"])
def test_ai_recovery_reconstructs_fresh_execution_for_real_service_action(
    tmp_path: Path,
    action: str,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions: list[_RecoveryAIExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _RecoveryAIExecution()
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="recover-ai", task_type="AI_OPTIMIZATION")
        service.store.transition(
            "recover-ai",
            expected_state=LifecycleState.AUTHORIZED,
            new_state=LifecycleState.RECOVERY_REQUIRED,
            event_type="TEST_PROCESS_RESTART_DETECTED",
        )

        recovered = service.recover_task(
            "recover-ai",
            RecoveryRequest(action=action, operator_confirmed=True),
        )

        assert executions[0].restore_calls == 1
        assert executions[0].run_calls == 0
        if action == "RESUME":
            assert recovered["state"] == "RUNNING"
            assert runner.wait_for_idle(timeout=5)
            assert service.get_task("recover-ai")["state"] == "COMPLETED"
            assert len(executions) == 2
            assert executions[1].run_calls == 1
            assert executions[1].restore_calls == 1
        else:
            assert recovered["state"] == "ABORTED_RECOVERED"
            assert len(executions) == 1
            assert service.store.active_task_id() is None
            assert service._hardware_authority.current_owner is None
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("action", ["RESUME", "ABORT_RECOVERED"])
def test_ai_recovery_without_factory_fails_closed_and_retains_authority(
    tmp_path: Path,
    action: str,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="recover-ai", task_type="AI_OPTIMIZATION")
        service.store.transition(
            "recover-ai",
            expected_state=LifecycleState.AUTHORIZED,
            new_state=LifecycleState.RECOVERY_REQUIRED,
            event_type="TEST_PROCESS_RESTART_DETECTED",
        )

        with pytest.raises(TaskStoreError, match="AI execution backend"):
            service.recover_task(
                "recover-ai",
                RecoveryRequest(action=action, operator_confirmed=True),
            )

        assert service.get_task("recover-ai")["state"] == "RECOVERY_REQUIRED"
        assert service.store.active_task_id() == "recover-ai"
        assert service._hardware_authority.current_owner == "v1.1:task:recover-ai"
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("action", ["RESUME", "ABORT_RECOVERED"])
def test_ai_recovery_restore_failure_retains_recovery_and_authority(
    tmp_path: Path,
    action: str,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions: list[_RecoveryAIExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _RecoveryAIExecution(
            restore_error=RuntimeError("injected AI recovery restore failure")
        )
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="recover-ai", task_type="AI_OPTIMIZATION")
        service.store.transition(
            "recover-ai",
            expected_state=LifecycleState.AUTHORIZED,
            new_state=LifecycleState.RECOVERY_REQUIRED,
            event_type="TEST_PROCESS_RESTART_DETECTED",
        )

        with pytest.raises(RuntimeError, match="recovery restore failure"):
            service.recover_task(
                "recover-ai",
                RecoveryRequest(action=action, operator_confirmed=True),
            )

        assert executions[0].restore_calls == 1
        assert executions[0].run_calls == 0
        assert service.get_task("recover-ai")["state"] == "RECOVERY_REQUIRED"
        assert service.store.active_task_id() == "recover-ai"
        assert service._hardware_authority.current_owner == "v1.1:task:recover-ai"
    finally:
        runtime.shutdown()


def test_ai_pause_restores_b0_then_resume_completes(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    first_started = Event()
    resumed_started = Event()
    executions: list[_CooperativeAIExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _CooperativeAIExecution(
            started=(first_started if not executions else resumed_started),
            wait_for_stop=not executions,
        )
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="paused-ai", task_type="AI_OPTIMIZATION")
        assert service.start_task("paused-ai")["state"] == "RUNNING"
        assert first_started.wait(timeout=5)

        assert service.pause_task("paused-ai")["state"] == "PAUSED"
        assert executions[0].restore_calls == 1
        assert service.resume_task("paused-ai")["state"] == "RUNNING"
        assert resumed_started.wait(timeout=5)
        assert runner.wait_for_idle(timeout=5)
        assert service.get_task("paused-ai")["state"] == "COMPLETED"
        assert len(executions) == 2
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("audit_write_fails", (False, True))
def test_transient_candidate_artifact_failure_does_not_fail_task(
    tmp_path: Path,
    monkeypatch,
    audit_write_fails: bool,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ai_factory, _, _ = _real_fake_ai_factory(
        database_path,
        artifact_root,
        block_first_execution=False,
    )
    original_write = fake_ai_execution_module._write_atomic_json
    injected = Event()

    def fail_first_candidate_once(path, payload):
        if payload["candidates_consumed"] == 1 and not injected.is_set():
            injected.set()
            raise OSError("injected transient candidate artifact failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        fake_ai_execution_module,
        "_write_atomic_json",
        fail_first_candidate_once,
    )
    if audit_write_fails:
        original_append_audit = TaskStore.append_audit_event

        def fail_artifact_audit(store, *args, **kwargs):
            if kwargs.get("event_type") == "AI_ARTIFACT_EXPORT_FAILED":
                raise OSError("injected artifact audit failure")
            return original_append_audit(store, *args, **kwargs)

        monkeypatch.setattr(TaskStore, "append_audit_event", fail_artifact_audit)
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    task_id = "transient-artifact-ai"
    try:
        _authorize_task(service, task_id=task_id, task_type="AI_OPTIMIZATION")
        service.start_task(task_id)
        assert runner.wait_for_idle(timeout=5)

        assert injected.is_set()
        task = service.get_task(task_id)
        assert task["state"] in {"COMPLETED", "COMPLETED_NO_IMPROVEMENT"}
        failures = [
            event
            for event in service.store.events(task_id)
            if event.event_type == "AI_ARTIFACT_EXPORT_FAILED"
        ]
        if audit_write_fails:
            assert failures == []
        else:
            assert len(failures) == 1
            assert failures[0].payload["stage"] == "I2C_COARSE"
            assert "injected transient candidate artifact failure" in failures[
                0
            ].payload["error"]
        assert not any(
            event.event_type == "AI_EXECUTION_FAILED_B0_RESTORED"
            for event in service.store.events(task_id)
        )
        artifact = json.loads(
            (
                artifact_root
                / task_id
                / "optimization_result.json"
            ).read_text(encoding="utf-8")
        )
        assert artifact["terminal_lifecycle"] == task["state"]
        assert artifact["restored_to_b0"] is True
    finally:
        runtime.shutdown()


def test_ai_artifact_terminal_refresh_follows_restore_and_durable_terminal_cas(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions = []

    def ai_factory(task_id, _b0, _payload):
        execution = _ArtifactAwareAIExecution(
            database_path=database_path,
            task_id=task_id,
        )
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="artifact-ai", task_type="AI_OPTIMIZATION")
        service.start_task("artifact-ai")
        assert runner.wait_for_idle(timeout=5)

        assert executions[0].restore_calls == 1
        assert executions[0].artifact_refreshes == [
            (LifecycleState.COMPLETED, LifecycleState.COMPLETED)
        ]
    finally:
        runtime.shutdown()


def test_ai_artifact_failure_does_not_reclassify_verified_b0_restore(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions = []

    def ai_factory(task_id, _b0, _payload):
        execution = _ArtifactAwareAIExecution(
            database_path=database_path,
            task_id=task_id,
            refresh_error=OSError("injected artifact failure"),
        )
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="artifact-failure-ai",
            task_type="AI_OPTIMIZATION",
        )
        service.start_task("artifact-failure-ai")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("artifact-failure-ai")["state"] == "COMPLETED"
        assert executions[0].restore_calls == 1
        event_types = [
            event.event_type
            for event in service.store.events("artifact-failure-ai")
        ]
        assert "AI_ARTIFACT_EXPORT_FAILED" in event_types
        assert "AI_ROLLBACK_FAILED" not in event_types
        assert service.store.active_task_id() is None
    finally:
        runtime.shutdown()


def test_real_fake_ai_pause_waits_for_candidate_commit_then_resume_replays_progress(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ai_factory, executions, plants = _real_fake_ai_factory(
        database_path,
        artifact_root,
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    stop_observed = _observe_stop_request(runner)
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    pause_result: dict[str, object] = {}

    def pause() -> None:
        try:
            pause_result["task"] = service.pause_task("real-paused-ai")
        except Exception as error:  # pragma: no cover - asserted below
            pause_result["error"] = error

    try:
        _authorize_task(
            service,
            task_id="real-paused-ai",
            task_type="AI_OPTIMIZATION",
        )
        assert service.start_task("real-paused-ai")["state"] == "RUNNING"
        assert plants[0].candidate_measurement_started.wait(timeout=5)
        before_pause = service.store.observations("real-paused-ai")
        assert [row.phase for row in before_pause] == ["B0"]

        pause_thread = Thread(target=pause)
        pause_thread.start()
        assert stop_observed.set_called.wait(timeout=5)
        assert pause_thread.is_alive()
        assert service.get_task("real-paused-ai")["state"] == "RUNNING"

        plants[0].allow_candidate_measurement.set()
        pause_thread.join(timeout=5)
        assert not pause_thread.is_alive()
        assert "error" not in pause_result
        assert pause_result["task"]["state"] == "PAUSED"
        assert plants[0].restore_calls == 1
        assert plants[0].is_at_b0()

        committed = service.store.observations("real-paused-ai")
        assert [row.phase for row in committed] == ["B0", "I2C_COARSE"]
        committed_candidate = committed[-1]
        assert committed_candidate.raw_response["budget_index"] == 1

        assert service.resume_task("real-paused-ai")["state"] == "RUNNING"
        assert runner.wait_for_idle(timeout=10)
        terminal = service.get_task("real-paused-ai")["state"]
        assert terminal in {"COMPLETED", "COMPLETED_NO_IMPROVEMENT"}
        assert len(executions) == 2
        assert plants[1].constructed_at_b0
        assert plants[1].mutations[0][0] == "FIR"
        assert plants[1].mutations[1] == ("I2C", committed_candidate.readback)
        assert plants[1].restore_calls == 1
        assert plants[1].is_at_b0()

        observations = service.store.observations("real-paused-ai")
        assert [row.sequence for row in observations] == list(
            range(1, len(observations) + 1)
        )
        assert len({row.observation_id for row in observations}) == len(observations)
        candidate_rows = [
            row
            for row in observations
            if row.phase
            in {
                "I2C_COARSE",
                "I2C_FINE",
                "FIR_FIRST",
                "I2C_RETUNE",
                "FIR_ITERATIVE",
            }
        ]
        budget_indexes = [
            row.raw_response["budget_index"] for row in candidate_rows
        ]
        assert budget_indexes == list(range(1, len(candidate_rows) + 1))
        assert len(candidate_rows) <= 3
        assert len(
            {row.raw_response["candidate_id"] for row in candidate_rows}
        ) == len(candidate_rows)
        event_types = [
            event.event_type for event in service.store.events("real-paused-ai")
        ]
        assert "TASK_PAUSED_B0_RESTORED" in event_types
        assert "TASK_RESUMED_AFTER_PREFLIGHT" in event_types
    finally:
        plants[0].allow_candidate_measurement.set()
        runtime.shutdown()


def test_legacy_ai_run_signature_fails_before_running_transition(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=lambda _task_id, _b0, _payload: (
            _LegacyAIExecution()
        ),
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="legacy-ai", task_type="AI_OPTIMIZATION")

        with pytest.raises(TaskStoreError, match="cooperative"):
            service.start_task("legacy-ai")

        assert service.get_task("legacy-ai")["state"] == "AUTHORIZED"
        assert service._hardware_authority.current_owner is None
    finally:
        runtime.shutdown()


def test_ai_abort_wins_terminal_race_and_restores_b0(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    started = Event()
    executions: list[_TerminalOnStopAIExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _TerminalOnStopAIExecution(started)
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="aborted-ai", task_type="AI_OPTIMIZATION")
        service.start_task("aborted-ai")
        assert started.wait(timeout=5)

        result = service.abort_task(
            "aborted-ai",
            AbortRequest(reason="operator stop"),
        )

        assert result["state"] == "ABORTED"
        assert executions[0].restore_calls == 1
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner is None
    finally:
        runtime.shutdown()


def test_real_fake_ai_abort_waits_for_candidate_commit_restores_b0_and_stops(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ai_factory, _executions, plants = _real_fake_ai_factory(
        database_path,
        artifact_root,
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    stop_observed = _observe_stop_request(runner)
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    abort_result: dict[str, object] = {}

    def abort() -> None:
        try:
            abort_result["task"] = service.abort_task(
                "real-aborted-ai",
                AbortRequest(reason="operator stop at candidate boundary"),
            )
        except Exception as error:  # pragma: no cover - asserted below
            abort_result["error"] = error

    try:
        _authorize_task(
            service,
            task_id="real-aborted-ai",
            task_type="AI_OPTIMIZATION",
        )
        service.start_task("real-aborted-ai")
        assert plants[0].candidate_measurement_started.wait(timeout=5)

        abort_thread = Thread(target=abort)
        abort_thread.start()
        assert stop_observed.set_called.wait(timeout=5)
        assert abort_thread.is_alive()
        assert service.get_task("real-aborted-ai")["state"] == "ROLLING_BACK"

        plants[0].allow_candidate_measurement.set()
        abort_thread.join(timeout=5)
        assert not abort_thread.is_alive()
        assert "error" not in abort_result
        assert abort_result["task"]["state"] == "ABORTED"
        assert plants[0].restore_calls == 1
        assert plants[0].is_at_b0()
        observations = service.store.observations("real-aborted-ai")
        assert [row.phase for row in observations] == ["B0", "I2C_COARSE"]
        committed_ids = [row.observation_id for row in observations]
        assert runner.wait_for_idle(timeout=1)
        assert [
            row.observation_id
            for row in service.store.observations("real-aborted-ai")
        ] == committed_ids
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner is None
    finally:
        plants[0].allow_candidate_measurement.set()
        runtime.shutdown()


@pytest.mark.parametrize("action", ["pause", "abort"])
def test_real_fake_ai_control_restore_failure_requires_recovery_and_keeps_authority(
    tmp_path: Path,
    action: str,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ai_factory, _executions, plants = _real_fake_ai_factory(
        database_path,
        artifact_root,
        fail_restore=True,
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=ai_factory,
    )
    stop_observed = _observe_stop_request(runner)
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    result: dict[str, object] = {}

    def control() -> None:
        try:
            if action == "pause":
                result["task"] = service.pause_task("restore-failure-ai")
            else:
                result["task"] = service.abort_task(
                    "restore-failure-ai",
                    AbortRequest(reason="exercise restore failure"),
                )
        except Exception as error:
            result["error"] = error

    try:
        _authorize_task(
            service,
            task_id="restore-failure-ai",
            task_type="AI_OPTIMIZATION",
        )
        service.start_task("restore-failure-ai")
        assert plants[0].candidate_measurement_started.wait(timeout=5)
        thread = Thread(target=control)
        thread.start()
        assert stop_observed.set_called.wait(timeout=5)
        assert thread.is_alive()
        plants[0].allow_candidate_measurement.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        assert plants[0].restore_calls == 1
        assert service.get_task("restore-failure-ai")["state"] == "RECOVERY_REQUIRED"
        assert service.store.active_task_id() == "restore-failure-ai"
        assert service._hardware_authority.current_owner == (
            "v1.1:task:restore-failure-ai"
        )
        event_types = [
            event.event_type
            for event in service.store.events("restore-failure-ai")
        ]
        if action == "pause":
            assert isinstance(result.get("error"), RecoveryRequired)
            assert "AI_ROLLBACK_FAILED" in event_types
        else:
            assert result["task"]["state"] == "RECOVERY_REQUIRED"
            assert "B0_ROLLBACK_FAILED" in event_types
    finally:
        plants[0].allow_candidate_measurement.set()
        runtime.shutdown()


def test_real_fake_ai_terminal_restore_failure_requires_recovery(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ai_factory, _executions, plants = _real_fake_ai_factory(
        database_path,
        artifact_root,
        block_first_execution=False,
        fail_restore=True,
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="terminal-restore-failure-ai",
            task_type="AI_OPTIMIZATION",
        )
        service.start_task("terminal-restore-failure-ai")
        assert runner.wait_for_idle(timeout=10)

        assert plants[0].restore_calls == 1
        assert service.get_task("terminal-restore-failure-ai")["state"] == (
            "RECOVERY_REQUIRED"
        )
        assert service.store.active_task_id() == "terminal-restore-failure-ai"
        assert service._hardware_authority.current_owner == (
            "v1.1:task:terminal-restore-failure-ai"
        )
        events = service.store.events("terminal-restore-failure-ai")
        rollback = next(event for event in events if event.event_type == "AI_ROLLBACK_FAILED")
        assert "injected real FAKE B0 restore failure" in rollback.payload["rollback_error"]
    finally:
        runtime.shutdown()


def test_real_fake_ai_explicit_recovery_restore_failure_stays_recovery_required(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    ai_factory, _executions, plants = _real_fake_ai_factory(
        database_path,
        artifact_root,
        block_first_execution=False,
        fail_restore=True,
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="explicit-recovery-failure-ai",
            task_type="AI_OPTIMIZATION",
        )
        service.store.transition(
            "explicit-recovery-failure-ai",
            expected_state=LifecycleState.AUTHORIZED,
            new_state=LifecycleState.RECOVERY_REQUIRED,
            event_type="TEST_RECOVERY_REQUIRED",
        )

        with pytest.raises(RuntimeError, match="real FAKE B0 restore failure"):
            service.recover_task(
                "explicit-recovery-failure-ai",
                RecoveryRequest(action="ABORT_RECOVERED", operator_confirmed=True),
            )

        assert plants[0].restore_calls == 1
        assert service.get_task("explicit-recovery-failure-ai")["state"] == (
            "RECOVERY_REQUIRED"
        )
        assert service.store.active_task_id() == "explicit-recovery-failure-ai"
        assert service._hardware_authority.current_owner == (
            "v1.1:task:explicit-recovery-failure-ai"
        )
        recovery_failure = service.store.events(
            "explicit-recovery-failure-ai"
        )[-1]
        assert recovery_failure.event_type == "AI_RECOVERY_B0_FAILED"
        assert "injected real FAKE B0 restore failure" in (
            recovery_failure.payload["rollback_error"]
        )
    finally:
        runtime.shutdown()


def test_recovery_audit_failure_does_not_mask_real_fake_ai_restore_error(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    restore_error = RuntimeError("real restore failure")
    audit_attempts: list[tuple[str, object]] = []

    class RestoreFailurePlant(FakeOptimizationPlant):
        def restore_b0(self) -> None:
            raise restore_error

    def ai_factory(task_id, b0, payload):
        plant = RestoreFailurePlant(b0)
        return FakeV11AIExecution(
            task_id,
            b0,
            payload,
            database_path,
            artifact_root,
            plant_factory=lambda _b0: plant,
        )

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    original_append = TaskStore.append_audit_event
    try:
        _authorize_task(
            service,
            task_id="audit-failure-recovery-ai",
            task_type="AI_OPTIMIZATION",
        )
        service.store.transition(
            "audit-failure-recovery-ai",
            expected_state=LifecycleState.AUTHORIZED,
            new_state=LifecycleState.RECOVERY_REQUIRED,
            event_type="TEST_RECOVERY_REQUIRED",
        )
        events_before = service.store.events("audit-failure-recovery-ai")

        def fail_recovery_audit(store, task_id, *, event_type, payload, **kwargs):
            if event_type == "AI_RECOVERY_B0_FAILED":
                audit_attempts.append((task_id, payload))
                raise OSError("injected audit persistence failure")
            return original_append(
                store,
                task_id,
                event_type=event_type,
                payload=payload,
                **kwargs,
            )

        monkeypatch.setattr(TaskStore, "append_audit_event", fail_recovery_audit)

        with pytest.raises(RuntimeError) as caught:
            service.recover_task(
                "audit-failure-recovery-ai",
                RecoveryRequest(action="ABORT_RECOVERED", operator_confirmed=True),
            )

        assert caught.value is restore_error
        assert str(caught.value) == "real restore failure"
        assert audit_attempts == [
            (
                "audit-failure-recovery-ai",
                {"rollback_error": "real restore failure"},
            )
        ]
        assert service.store.events("audit-failure-recovery-ai") == events_before
        assert service.get_task("audit-failure-recovery-ai")["state"] == (
            "RECOVERY_REQUIRED"
        )
        assert service.store.active_task_id() == "audit-failure-recovery-ai"
        assert service._hardware_authority.current_owner == (
            "v1.1:task:audit-failure-recovery-ai"
        )
    finally:
        runtime.shutdown()


def test_real_fake_ai_restart_persists_progress_and_requires_explicit_recovery(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    first_factory, _first_executions, first_plants = _real_fake_ai_factory(
        database_path,
        artifact_root,
    )
    first_runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=first_factory,
    )
    stop_observed = _observe_stop_request(first_runner)
    first_runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=first_runner,
    )
    first_service = first_runtime.v11_service
    assert first_service is not None
    try:
        _authorize_task(
            first_service,
            task_id="restart-progress-ai",
            task_type="AI_OPTIMIZATION",
        )
        first_service.start_task("restart-progress-ai")
        assert first_plants[0].candidate_measurement_started.wait(timeout=5)
        pause_result: dict[str, object] = {}

        def pause() -> None:
            pause_result["task"] = first_service.pause_task("restart-progress-ai")

        pause_thread = Thread(target=pause)
        pause_thread.start()
        assert stop_observed.set_called.wait(timeout=5)
        first_plants[0].allow_candidate_measurement.set()
        pause_thread.join(timeout=5)
        assert not pause_thread.is_alive()
        assert pause_result["task"]["state"] == "PAUSED"
        before_observations = first_service.store.observations("restart-progress-ai")
        before_best = first_service.store.best_state("restart-progress-ai")
        before_b0 = first_service.store.load_b0("restart-progress-ai")
        before_authorization = first_service.store.load_authorization(
            "restart-progress-ai"
        )
        assert len(before_observations) == 2
        assert before_best is not None

        # Model the last durable lifecycle write before process loss.  There is
        # no worker after the already-safe PAUSED boundary, so shutdown performs
        # no execution or plant mutation.
        first_service.store.transition(
            "restart-progress-ai",
            expected_state=LifecycleState.PAUSED,
            new_state=LifecycleState.RUNNING,
            event_type="TEST_PROCESS_INTERRUPTED_AFTER_DURABLE_PROGRESS",
        )
        before_events = first_service.store.events("restart-progress-ai")
    finally:
        first_plants[0].allow_candidate_measurement.set()
        first_runtime.shutdown()

    restarted_factory, _restarted_executions, restarted_plants = (
        _real_fake_ai_factory(
            database_path,
            artifact_root,
            block_first_execution=False,
        )
    )
    restarted_runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=restarted_factory,
    )
    restarted_runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=restarted_runner,
    )
    restarted = restarted_runtime.v11_service
    assert restarted is not None
    try:
        assert restarted.get_task("restart-progress-ai")["state"] == (
            "RECOVERY_REQUIRED"
        )
        assert restarted_plants == []
        assert restarted.store.observations("restart-progress-ai") == (
            before_observations
        )
        restarted_events = restarted.store.events("restart-progress-ai")
        assert restarted_events[:-1] == before_events
        assert restarted_events[-1].event_type == "PROCESS_RESTART_DETECTED"
        assert restarted.store.best_state("restart-progress-ai") == before_best
        assert restarted.store.load_b0("restart-progress-ai") == before_b0
        assert restarted.store.load_authorization("restart-progress-ai") == (
            before_authorization
        )
        assert restarted.store.active_task_id() == "restart-progress-ai"
        assert restarted._hardware_authority.current_owner == (
            "v1.1:task:restart-progress-ai"
        )
        recovered = restarted.recover_task(
            "restart-progress-ai",
            RecoveryRequest(action="ABORT_RECOVERED", operator_confirmed=True),
        )
        assert recovered["state"] == "ABORTED_RECOVERED"
        assert len(restarted_plants) == 1
        assert restarted_plants[0].restore_calls == 1
        assert restarted_plants[0].is_at_b0()
        assert restarted.store.active_task_id() is None
    finally:
        restarted_runtime.shutdown()


def test_ai_terminal_after_rolling_back_is_cooperative_abort(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    started = Event()
    allow_return = Event()
    executions: list[_LateAbortAIExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _LateAbortAIExecution(
            started=started,
            allow_return=allow_return,
        )
        executions.append(execution)
        return execution

    runner = _LateSignalAbortRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
        allow_return=allow_return,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="late-abort-ai", task_type="AI_OPTIMIZATION")
        service.start_task("late-abort-ai")
        assert started.wait(timeout=5)

        result = service.abort_task(
            "late-abort-ai",
            AbortRequest(reason="operator stop"),
        )

        assert result["state"] == "ABORTED"
        assert executions[0].restore_calls == 1
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner is None
    finally:
        allow_return.set()
        runtime.shutdown()


@pytest.mark.parametrize(
    ("direct_recovery", "expected_restore_calls"),
    [(True, 0), (False, 1)],
)
def test_ai_recovery_failure_during_abort_is_not_swallowed(
    tmp_path: Path,
    direct_recovery: bool,
    expected_restore_calls: int,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    started = Event()
    allow_return = Event()
    executions: list[_LateAbortRecoveryFailureExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _LateAbortRecoveryFailureExecution(
            started=started,
            allow_return=allow_return,
            direct_recovery=direct_recovery,
        )
        executions.append(execution)
        return execution

    runner = _LateSignalAbortRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
        allow_return=allow_return,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="abort-recovery-ai", task_type="AI_OPTIMIZATION")
        service.start_task("abort-recovery-ai")
        assert started.wait(timeout=5)

        result = service.abort_task(
            "abort-recovery-ai",
            AbortRequest(reason="operator stop"),
        )

        assert result["state"] == "RECOVERY_REQUIRED"
        assert executions[0].restore_calls == expected_restore_calls
        assert service.store.active_task_id() == "abort-recovery-ai"
        assert service._hardware_authority.current_owner == (
            "v1.1:task:abort-recovery-ai"
        )
    finally:
        allow_return.set()
        runtime.shutdown()


def test_sweep_rollback_failure_during_abort_is_not_swallowed(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    started = Event()
    allow_return = Event()
    executors: list[_LateAbortSweepRollbackFailureExecutor] = []

    def executor_factory(b0, plan):
        executor = _LateAbortSweepRollbackFailureExecutor(
            b0,
            plan,
            started=started,
            allow_return=allow_return,
        )
        executors.append(executor)
        return executor

    runner = _LateSignalAbortRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=executor_factory,
        allow_return=allow_return,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="abort-recovery-sweep", task_type="SWEEP")
        service.start_task("abort-recovery-sweep")
        assert started.wait(timeout=5)

        result = service.abort_task(
            "abort-recovery-sweep",
            AbortRequest(reason="operator stop"),
        )

        assert result["state"] == "RECOVERY_REQUIRED"
        assert executors[0].restore_calls == 1
        assert service.store.active_task_id() == "abort-recovery-sweep"
        assert service._hardware_authority.current_owner == (
            "v1.1:task:abort-recovery-sweep"
        )
    finally:
        allow_return.set()
        runtime.shutdown()


@pytest.mark.parametrize(
    "ai_result",
    ["COMPLETED", LifecycleState.PAUSED, LifecycleState.FAILED],
)
def test_ai_result_boundary_restores_b0_and_never_leaves_running(
    tmp_path: Path,
    ai_result: object,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions: list[_AIResultExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _AIResultExecution(ai_result)
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="ai-result", task_type="AI_OPTIMIZATION")
        service.start_task("ai-result")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("ai-result")["state"] == "FAILED"
        assert executions[0].restore_calls == 1
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner is None
    finally:
        runtime.shutdown()


def test_ai_terminal_callback_failure_is_safe_after_b0_restore(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions: list[_AIResultExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _AIResultExecution(LifecycleState.COMPLETED)
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    runner.set_terminal_callback(
        lambda _task_id, _state: (_ for _ in ()).throw(
            RuntimeError("injected terminal callback failure")
        )
    )
    try:
        _authorize_task(service, task_id="callback-ai", task_type="AI_OPTIMIZATION")
        service.start_task("callback-ai")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("callback-ai")["state"] == "COMPLETED"
        assert executions[0].restore_calls == 1
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner == (
            "v1.1:task:callback-ai"
        )
    finally:
        runtime.shutdown()


def test_ai_terminal_persistence_failure_falls_back_to_failed(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executions: list[_AIResultExecution] = []

    def ai_factory(_task_id, _b0, _payload):
        execution = _AIResultExecution(LifecycleState.COMPLETED)
        executions.append(execution)
        return execution

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    original_transition = TaskStore.transition
    injected = Event()

    def fail_terminal_once(store, *args, **kwargs):
        if kwargs.get("event_type") == "AI_EXECUTION_TERMINAL" and not injected.is_set():
            injected.set()
            raise TaskStoreError("injected terminal persistence failure")
        return original_transition(store, *args, **kwargs)

    monkeypatch.setattr(TaskStore, "transition", fail_terminal_once)
    try:
        _authorize_task(service, task_id="persist-ai", task_type="AI_OPTIMIZATION")
        service.start_task("persist-ai")
        assert runner.wait_for_idle(timeout=5)

        assert injected.is_set()
        assert service.get_task("persist-ai")["state"] == "FAILED"
        assert executions[0].restore_calls == 1
        assert service.store.active_task_id() is None
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("failure_point", ["worker_store", "sweep_factory"])
def test_worker_setup_failure_does_not_leave_task_running(
    tmp_path: Path,
    monkeypatch,
    failure_point: str,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )

    def sweep_factory(b0, plan):
        if failure_point == "sweep_factory":
            raise RuntimeError("injected sweep factory failure")
        return FakeSweepExecutor(b0, plan)

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=sweep_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="setup-failure", task_type="SWEEP")
        if failure_point == "worker_store":
            real_task_store = persistent_runner_module.TaskStore
            calls = 0

            def fail_worker_open_once(path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise TaskStoreError("injected worker store open failure")
                return real_task_store(path)

            monkeypatch.setattr(
                persistent_runner_module,
                "TaskStore",
                fail_worker_open_once,
            )

        service.start_task("setup-failure")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("setup-failure")["state"] == "FAILED"
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner is None
    finally:
        runtime.shutdown()


class _BlockingSecondPointExecutor(FakeSweepExecutor):
    def __init__(
        self,
        b0,
        plan,
        *,
        entered: Event,
        release: Event,
    ) -> None:
        super().__init__(b0, plan)
        self._entered = entered
        self._release = release

    def apply_point(self, point):
        if point.original_index == 1:
            self._entered.set()
            assert self._release.wait(timeout=5)
        return super().apply_point(point)


class _FailingSecondPointExecutor(FakeSweepExecutor):
    def apply_point(self, point):
        if point.original_index == 1:
            self.calls.append(("apply_failed", point.original_index))
            raise RuntimeError("injected candidate failure")
        return super().apply_point(point)


class _RollbackFailingExecutor(_FailingSecondPointExecutor):
    def restore_b0(self) -> None:
        self.calls.append(("restore_b0_failed",))
        raise RuntimeError("injected rollback failure")


class _RecoveryEvidenceExecutor(FakeSweepExecutor):
    def apply_point(self, point):
        self.calls.append(("apply_recovery_required", point.original_index))
        observed = dict(self._current)
        first_name = next(iter(observed))
        observed[first_name] += 1
        raise RecoveryRequired(
            "injected rollback failure with observed state",
            rollback_verified=False,
            evidence={"last_observed_readback": observed},
        )

    def restore_b0(self) -> None:  # pragma: no cover - safety assertion
        raise AssertionError("runner must not repeat a failed rollback")


class _InvalidFirstMeasurementExecutor(FakeSweepExecutor):
    def __init__(self, b0, plan) -> None:
        super().__init__(b0, plan)
        self._measurement_count = 0

    def measure_point(self):
        self._measurement_count += 1
        measurement = dict(super().measure_point())
        if self._measurement_count == 1:
            measurement["tdecq"] = float("nan")
        return measurement


class _StartsAwayFromB0Executor(FakeSweepExecutor):
    def __init__(self, b0, plan) -> None:
        super().__init__(b0, plan)
        first_name = next(iter(self._current))
        self._current[first_name] += 1


class _PauseNotifyingRunner(PersistentV11TaskRunner):
    def __init__(self, *args, pause_requested: Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pause_requested = pause_requested

    def pause_and_restore_b0(self, task_id: str) -> None:
        self._stop_requested.set()
        self.pause_requested.set()
        self._request_stop_and_wait(task_id)


class _AbortNotifyingRunner(PersistentV11TaskRunner):
    def __init__(self, *args, abort_requested: Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.abort_requested = abort_requested

    def abort_and_restore_b0(self, task_id: str) -> None:
        self._stop_requested.set()
        self.abort_requested.set()
        self._request_stop_and_wait(task_id)


class _CloseNotifyingRunner(PersistentV11TaskRunner):
    def __init__(self, *args, close_requested: Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.close_requested = close_requested

    def close(self) -> None:
        self._stop_requested.set()
        self.close_requested.set()
        super().close()


class _ShutdownBlockedExecutor(FakeSweepExecutor):
    def __init__(self, b0, plan, *, entered: Event, close_requested: Event):
        super().__init__(b0, plan)
        self._entered = entered
        self._close_requested = close_requested

    def apply_point(self, point):
        if point.original_index == 1:
            self._entered.set()
            assert self._close_requested.wait(timeout=5)
        return super().apply_point(point)


class _ConcurrentStartRunner(PersistentV11TaskRunner):
    def __init__(
        self,
        *args,
        first_entered: Event,
        second_entered: Event,
        release_first: Event,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._start_count = 0
        self._start_count_lock = Lock()
        self._first_entered = first_entered
        self._second_entered = second_entered
        self._release_first = release_first

    def start(self, task_id: str, task_type: TaskType) -> None:
        with self._start_count_lock:
            self._start_count += 1
            count = self._start_count
        if count == 1:
            self._first_entered.set()
            assert self._release_first.wait(timeout=5)
        else:
            self._second_entered.set()
        super().start(task_id, task_type)


def test_artifacts_are_refreshed_before_a_later_point_finishes(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    entered = Event()
    release = Event()
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=lambda b0, plan: _BlockingSecondPointExecutor(
            b0,
            plan,
            entered=entered,
            release=release,
        ),
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="blocked-sweep", task_type="SWEEP")
        service.start_task("blocked-sweep")
        assert entered.wait(timeout=5)

        artifact_directory = artifact_root / "blocked-sweep"
        payload = json.loads(
            (artifact_directory / "sweep_results.json").read_text(
                encoding="utf-8"
            )
        )
        assert [point["original_index"] for point in payload["points"]] == [0]
        assert len(
            (artifact_directory / "sweep_results.csv").read_text(
                encoding="utf-8"
            ).splitlines()
        ) == 2
    finally:
        release.set()
        runtime.shutdown()


def test_ai_unavailable_stays_authorized_without_mutation(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(
            service,
            task_id="unavailable-ai",
            task_type="AI_OPTIMIZATION",
        )

        with pytest.raises(TaskStoreError, match="AI execution backend"):
            service.start_task("unavailable-ai")

        assert service.get_task("unavailable-ai")["state"] == "AUTHORIZED"
        assert service.store.observations("unavailable-ai") == ()
        assert service._hardware_authority.current_owner is None
    finally:
        runtime.shutdown()


def test_capabilities_report_sweep_ready_and_ai_unavailable(tmp_path: Path):
    config = _config(tmp_path)
    runner = PersistentV11TaskRunner(
        database_path=(
            config.storage.sqlite_path.parent.parent
            / "v1.1"
            / "campaigns.sqlite3"
        ),
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        capabilities = service.runtime_capabilities()
        health = service.runtime_health()

        assert capabilities["sweep_execution_ready"] is True
        assert capabilities["ai_execution_ready"] is False
        assert health["sweep_execution_ready"] is True
        assert health["ai_execution_ready"] is False
    finally:
        runtime.shutdown()


def test_pause_resume_skips_completed_points_and_preserves_plan_order(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    entered = Event()
    release = Event()
    pause_requested = Event()
    executors: list[FakeSweepExecutor] = []

    def executor_factory(b0, plan):
        executor = (
            _BlockingSecondPointExecutor(
                b0,
                plan,
                entered=entered,
                release=release,
            )
            if not executors
            else FakeSweepExecutor(b0, plan)
        )
        executors.append(executor)
        return executor

    runner = _PauseNotifyingRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=executor_factory,
        pause_requested=pause_requested,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    releaser = Thread(
        target=lambda: (
            pause_requested.wait(timeout=5) and release.set()
        ),
        name="release-paused-sweep-point",
    )
    try:
        _authorize_task(service, task_id="paused-sweep", task_type="SWEEP")
        service.start_task("paused-sweep")
        assert entered.wait(timeout=5)
        releaser.start()

        assert service.pause_task("paused-sweep")["state"] == "PAUSED"
        releaser.join(timeout=5)
        assert not releaser.is_alive()
        paused_points = service.store.sweep_points(
            "paused-sweep", limit=10
        ).items
        assert [point.original_index for point in paused_points] == [0, 1]

        assert service.resume_task("paused-sweep")["state"] == "RUNNING"
        assert runner.wait_for_idle(timeout=5)
        assert service.get_task("paused-sweep")["state"] == "COMPLETED"
        points = service.store.sweep_points("paused-sweep", limit=10).items
        assert [point.original_index for point in points] == [0, 1, 2]
        assert [point.seed for point in points] == [17, 17, 17]
        assert [
            call[1]
            for call in executors[0].calls
            if call[0] == "apply"
        ] == [0, 1]
        assert [
            call[1]
            for call in executors[1].calls
            if call[0] == "apply"
        ] == [2]
        assert len(service.store.observations("paused-sweep")) == 1
    finally:
        release.set()
        if releaser.is_alive():
            releaser.join(timeout=5)
        runtime.shutdown()


def test_all_persisted_sweep_points_reexport_final_artifacts(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    executors: list[FakeSweepExecutor] = []

    def executor_factory(b0, plan):
        executor = FakeSweepExecutor(b0, plan)
        executors.append(executor)
        return executor

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=executor_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="all-persisted", task_type="SWEEP")
        task = service.store.require_task("all-persisted")
        plan = persistent_runner_module.build_sweep_plan(
            task.payload,
            service.store.load_b0("all-persisted"),
        )
        for point in plan.points:
            skipped = point.status is SweepPointStatus.SKIPPED_CONSTRAINT
            service.store.save_sweep_point(
                "all-persisted",
                SweepPointRecord(
                    original_index=point.original_index,
                    execution_index=point.execution_index,
                    status=(
                        SweepPointStatus.SKIPPED_CONSTRAINT
                        if skipped
                        else SweepPointStatus.COMPLETED
                    ),
                    requested=dict(point.requested),
                    applied={} if skipped else dict(point.requested),
                    readback={} if skipped else dict(point.requested),
                    tdecq=None if skipped else 2.5,
                    er=None if skipped else 5.7,
                    raw_response=None,
                    error_reason=point.constraint_reason if skipped else None,
                    seed=plan.seed,
                    hardware=plan.hardware,
                    mode=plan.mode,
                    completed_at=datetime.now(timezone.utc),
                ),
            )

        artifact_directory = artifact_root / "all-persisted"
        assert not artifact_directory.exists()

        service.start_task("all-persisted")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("all-persisted")["state"] == "COMPLETED"
        assert (artifact_directory / "sweep_results.json").is_file()
        assert (artifact_directory / "sweep_results.csv").is_file()
        assert [call for call in executors[0].calls if call[0] == "apply"] == []
    finally:
        runtime.shutdown()


def test_abort_finishes_current_point_and_restores_b0(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    entered = Event()
    release = Event()
    abort_requested = Event()
    executors: list[FakeSweepExecutor] = []

    def executor_factory(b0, plan):
        executor = _BlockingSecondPointExecutor(
            b0,
            plan,
            entered=entered,
            release=release,
        )
        executors.append(executor)
        return executor

    runner = _AbortNotifyingRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=executor_factory,
        abort_requested=abort_requested,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    releaser = Thread(
        target=lambda: (
            abort_requested.wait(timeout=5) and release.set()
        ),
        name="release-aborted-sweep-point",
    )
    try:
        _authorize_task(service, task_id="aborted-sweep", task_type="SWEEP")
        service.start_task("aborted-sweep")
        assert entered.wait(timeout=5)
        releaser.start()

        aborted = service.abort_task(
            "aborted-sweep",
            AbortRequest(reason="operator stop"),
        )
        releaser.join(timeout=5)
        assert not releaser.is_alive()
        assert aborted["state"] == "ABORTED"
        assert service.store.active_task_id() is None
        assert service._hardware_authority.current_owner is None
        assert ("restore_b0",) in executors[0].calls
    finally:
        release.set()
        if releaser.is_alive():
            releaser.join(timeout=5)
        runtime.shutdown()


@pytest.mark.parametrize(
    ("executor_type", "expected_state", "authority_retained"),
    [
        (_FailingSecondPointExecutor, "FAILED", False),
        (_RollbackFailingExecutor, "RECOVERY_REQUIRED", True),
    ],
)
def test_failure_state_depends_on_verified_b0_rollback(
    tmp_path: Path,
    executor_type,
    expected_state: str,
    authority_retained: bool,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=executor_type,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="failed-sweep", task_type="SWEEP")
        service.start_task("failed-sweep")
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task("failed-sweep")["state"] == expected_state
        assert (service.store.active_task_id() is not None) is authority_retained
        assert (
            service._hardware_authority.current_owner is not None
        ) is authority_retained
        executor = runner.last_sweep_executor
        assert executor is not None
        restore_calls = [
            call
            for call in executor.calls
            if call[0] in {"restore_b0", "restore_b0_failed"}
        ]
        assert len(restore_calls) == 1
        terminal_event = service.store.events("failed-sweep")[-1]
        point_evidence = terminal_event.payload["point_evidence"]
        assert point_evidence["restore_status"] == (
            "VERIFIED"
            if expected_state == "FAILED"
            else "RECOVERY_REQUIRED"
        )
        assert point_evidence["candidate_round_time_s"] is not None
        assert point_evidence["screening_round_with_restore_s"] is not None
        assert "restore_readback_result" in point_evidence
    finally:
        runtime.shutdown()


def test_recovery_readback_is_preserved_in_point_and_task_event(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=_RecoveryEvidenceExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        task_id = "recovery-readback-evidence"
        _authorize_task(service, task_id=task_id, task_type="SWEEP")
        service.start_task(task_id)
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task(task_id)["state"] == "RECOVERY_REQUIRED"
        record = service.store.sweep_points(task_id, limit=10).items[0]
        assert record.restore_status == "RECOVERY_REQUIRED"
        assert record.restore_readback_result is not None
        terminal_event = service.store.events(task_id)[-1]
        assert terminal_event.payload["point_evidence"][
            "restore_readback_result"
        ] == dict(record.restore_readback_result)
        executor = runner.last_sweep_executor
        assert executor is not None
        assert not any(call[0] == "restore_b0" for call in executor.calls)
    finally:
        runtime.shutdown()


def test_persistent_point_update_failure_writes_complete_fallback_artifact(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        task_id = "point-evidence-fallback"
        _authorize_task(service, task_id=task_id, task_type="SWEEP")

        def fail_point_update(_store, _task_id, _record):
            raise OSError("injected persistent point update failure")

        monkeypatch.setattr(
            TaskStore,
            "update_sweep_point",
            fail_point_update,
        )
        service.start_task(task_id)
        assert runner.wait_for_idle(timeout=5)

        assert service.get_task(task_id)["state"] == "FAILED"
        executor = runner.last_sweep_executor
        assert executor is not None
        assert [
            call for call in executor.calls if call[0] == "restore_b0"
        ] == [("restore_b0",)]
        terminal_event = service.store.events(task_id)[-1]
        point_evidence = terminal_event.payload["point_evidence"]
        assert point_evidence["restore_status"] == "VERIFIED"
        assert point_evidence["result_persistence_duration_s"] is not None
        assert point_evidence["candidate_round_time_s"] is not None
        assert point_evidence["restore_duration_s"] is not None
        assert point_evidence["screening_round_with_restore_s"] is not None
        assert point_evidence["evidence_persistence_error"]

        fallback_path = (
            artifact_root / task_id / "sweep_failure_evidence.json"
        )
        fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
        assert fallback["task_id"] == task_id
        assert fallback["point_evidence"] == point_evidence
        assert terminal_event.payload["fallback_artifact"] == str(
            fallback_path
        )
    finally:
        runtime.shutdown()


@pytest.mark.parametrize(
    ("executor_type", "fir_constraints", "expected_statuses", "apply_indexes"),
    [
        (
            _InvalidFirstMeasurementExecutor,
            False,
            ["INVALID_TDECQ", "COMPLETED", "COMPLETED"],
            [0, 1, 2],
        ),
        (
            FakeSweepExecutor,
            True,
            ["SKIPPED_CONSTRAINT", "SKIPPED_CONSTRAINT", "COMPLETED"],
            [2],
        ),
    ],
)
def test_invalid_tdecq_continues_and_constraints_never_apply(
    tmp_path: Path,
    executor_type,
    fir_constraints: bool,
    expected_statuses: list[str],
    apply_indexes: list[int],
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executors: list[FakeSweepExecutor] = []

    def executor_factory(b0, plan):
        executor = executor_type(b0, plan)
        executors.append(executor)
        return executor

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=executor_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        if fir_constraints:
            _authorize_fir_constraint_sweep(service, task_id="edge-sweep")
        else:
            _authorize_task(service, task_id="edge-sweep", task_type="SWEEP")
        service.start_task("edge-sweep")
        assert runner.wait_for_idle(timeout=5)

        points = service.store.sweep_points("edge-sweep", limit=10).items
        assert [point.status.value for point in points] == expected_statuses
        assert [
            call[1]
            for call in executors[0].calls
            if call[0] == "apply"
        ] == apply_indexes
        assert service.get_task("edge-sweep")["state"] == "COMPLETED"
    finally:
        runtime.shutdown()


def test_shutdown_waits_for_worker_before_service_store_closes(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    entered = Event()
    close_requested = Event()
    runner = _CloseNotifyingRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=lambda b0, plan: _ShutdownBlockedExecutor(
            b0,
            plan,
            entered=entered,
            close_requested=close_requested,
        ),
        close_requested=close_requested,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    _authorize_task(service, task_id="shutdown-sweep", task_type="SWEEP")
    service.start_task("shutdown-sweep")
    assert entered.wait(timeout=5)

    runtime.shutdown()

    assert runner.wait_for_idle(timeout=0)
    with TaskStore(database_path) as reopened:
        assert reopened.require_task("shutdown-sweep").state in {
            LifecycleState.RUNNING,
            LifecycleState.RECOVERY_REQUIRED,
        }


def test_b0_is_verified_and_restored_before_baseline_measurement(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    executors: list[_StartsAwayFromB0Executor] = []

    def executor_factory(b0, plan):
        executor = _StartsAwayFromB0Executor(b0, plan)
        executors.append(executor)
        return executor

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=executor_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="b0-order", task_type="SWEEP")
        service.start_task("b0-order")
        assert runner.wait_for_idle(timeout=5)

        call_names = [call[0] for call in executors[0].calls]
        assert call_names[:6] == [
            "preflight",
            "is_at_b0",
            "restore_b0",
            "is_at_b0",
            "measure_b0",
            "apply",
        ]
        assert service.get_task("b0-order")["state"] == "COMPLETED"
    finally:
        runtime.shutdown()


def test_runner_rejects_second_prepare_before_dispatch(tmp_path: Path):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="double-prepare", task_type="SWEEP")
        runner.start("double-prepare", TaskType.SWEEP)

        with pytest.raises(TaskStoreError, match="prepared"):
            runner.start("double-prepare", TaskType.SWEEP)
    finally:
        runtime.shutdown()


def test_start_transition_loser_cancels_prepared_dispatch(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="prepare-race", task_type="SWEEP")
        real_transition = service.store.transition
        race_won = False

        def transition_with_abort_winner(task_id, **kwargs):
            nonlocal race_won
            if (
                not race_won
                and kwargs.get("expected_state") is LifecycleState.AUTHORIZED
                and kwargs.get("new_state") is LifecycleState.RUNNING
            ):
                race_won = True
                with TaskStore(database_path) as concurrent_store:
                    concurrent_store.transition(
                        task_id,
                        expected_state=LifecycleState.AUTHORIZED,
                        new_state=LifecycleState.ABORTED,
                        event_type="CONCURRENT_ABORT_WON",
                    )
            return real_transition(task_id, **kwargs)

        monkeypatch.setattr(service.store, "transition", transition_with_abort_winner)

        with pytest.raises(TaskConcurrencyError):
            service.start_task("prepare-race")

        assert service.get_task("prepare-race")["state"] == "ABORTED"
        assert service._hardware_authority.current_owner is None
        runner.start("prepare-race", TaskType.SWEEP)
        runner.cancel_prepared("prepare-race")
    finally:
        runtime.shutdown()


def test_dispatch_submit_failure_clears_prepared_and_fails_task(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=config.storage.artifact_root.parent / "v1.1",
        sweep_executor_factory=FakeSweepExecutor,
    )
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    service = runtime.v11_service
    assert service is not None
    try:
        _authorize_task(service, task_id="submit-failure", task_type="SWEEP")

        def fail_submit(*_args, **_kwargs):
            raise RuntimeError("injected submit failure")

        monkeypatch.setattr(runner._executor, "submit", fail_submit)

        with pytest.raises(RuntimeError, match="submit failure"):
            service.start_task("submit-failure")

        assert service.get_task("submit-failure")["state"] == "FAILED"
        assert service._hardware_authority.current_owner is None
        runner.start("submit-failure", TaskType.SWEEP)
        runner.cancel_prepared("submit-failure")
    finally:
        runtime.shutdown()


def test_shared_service_task_store_serializes_concurrent_operations(tmp_path: Path):
    config = _config(tmp_path)
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=FakeV11TaskRunner(),
    )
    service = runtime.v11_service
    assert service is not None
    writers = 8
    readers = 4
    iterations = 20
    start = Barrier(writers + readers)
    errors: list[Exception] = []
    error_lock = Lock()
    try:
        _authorize_task(service, task_id="shared-store", task_type="SWEEP")
        initial_events = len(service.events("shared-store"))

        def writer(worker_index: int) -> None:
            try:
                start.wait(timeout=5)
                for iteration in range(iterations):
                    service.store.append_audit_event(
                        "shared-store",
                        event_type=f"CONCURRENT_{worker_index}_{iteration}",
                        payload={"worker": worker_index, "iteration": iteration},
                    )
            except Exception as error:
                with error_lock:
                    errors.append(error)

        def reader() -> None:
            try:
                start.wait(timeout=5)
                for _ in range(iterations):
                    service.get_task("shared-store")
                    service.events("shared-store")
            except Exception as error:
                with error_lock:
                    errors.append(error)

        threads = [
            Thread(target=writer, args=(index,)) for index in range(writers)
        ] + [Thread(target=reader) for _ in range(readers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        events = service.events("shared-store")
        assert len(events) == initial_events + writers * iterations
        assert [event.sequence for event in events] == list(
            range(1, len(events) + 1)
        )
    finally:
        runtime.shutdown()


def test_concurrent_service_start_cannot_release_winner_authority(
    tmp_path: Path,
):
    config = _config(tmp_path)
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    authority_path = tmp_path / "hardware-authority.lock"
    setup_store = TaskStore(database_path)
    setup_service = V11ApplicationService(
        store=setup_store,
        gateway=FakeV11Gateway(),
        runner=FakeV11TaskRunner(),
        artifact_root=artifact_root,
        hardware_authority=HardwareAuthority(authority_path),
    )
    _authorize_task(
        setup_service,
        task_id="concurrent-start",
        task_type="SWEEP",
    )
    setup_service.close()
    setup_store.close()

    first_entered = Event()
    second_entered = Event()
    release_first = Event()
    point_entered = Event()
    release_point = Event()
    executors: list[FakeSweepExecutor] = []

    def executor_factory(b0, plan):
        executor = _BlockingSecondPointExecutor(
            b0,
            plan,
            entered=point_entered,
            release=release_point,
        )
        executors.append(executor)
        return executor

    runner = _ConcurrentStartRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        sweep_executor_factory=executor_factory,
        first_entered=first_entered,
        second_entered=second_entered,
        release_first=release_first,
    )
    authority = HardwareAuthority(authority_path)
    concurrent_store = TaskStore(database_path)
    service = V11ApplicationService(
        store=concurrent_store,
        gateway=FakeV11Gateway(),
        runner=runner,
        artifact_root=artifact_root,
        hardware_authority=authority,
    )
    outcomes: list[object] = []
    outcome_lock = Lock()

    def start_task() -> None:
        try:
            outcome: object = service.start_task("concurrent-start")
        except Exception as error:
            outcome = error
        with outcome_lock:
            outcomes.append(outcome)

    first = Thread(target=start_task, name="v11-start-first")
    second = Thread(target=start_task, name="v11-start-second")
    try:
        first.start()
        assert first_entered.wait(timeout=5)
        second.start()
        second_entered.wait(timeout=0.5)
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert not first.is_alive() and not second.is_alive()
        assert point_entered.wait(timeout=5)

        assert sum(isinstance(item, dict) for item in outcomes) == 1
        assert sum(isinstance(item, Exception) for item in outcomes) == 1
        assert len(executors) == 1
        assert authority.current_owner == "v1.1:task:concurrent-start"
    finally:
        release_first.set()
        release_point.set()
        runner.wait_for_idle(timeout=5)
        service.close()
        concurrent_store.close()
