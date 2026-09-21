from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Barrier, Event, Lock, Thread

import pytest
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
from vcsel_ai_tuning.v1_1.hardware_control import RecoveryRequired
from vcsel_ai_tuning.v1_1.persistent_runner import (
    FakeSweepExecutor,
    PersistentV11TaskRunner,
)
from vcsel_ai_tuning.v1_1.repository import (
    TaskConcurrencyError,
    TaskStore,
    TaskStoreError,
)
from vcsel_ai_tuning.v1_1.service import V11ApplicationService
from vcsel_ai_tuning.v1_1.sweep import SweepPointRecord, SweepPointStatus


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
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            operator_confirmed=True,
        ),
    )


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
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
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
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
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
    runtime = create_fake_runtime(_config(tmp_path))
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
