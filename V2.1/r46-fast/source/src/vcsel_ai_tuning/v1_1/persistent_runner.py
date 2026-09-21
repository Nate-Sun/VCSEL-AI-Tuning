"""Persistent single-worker execution for V1.1 tasks.

The runner owns no service ``TaskStore`` connection.  Every background run
opens and closes its own connection on the worker thread.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from threading import Event, RLock
from typing import Any, Protocol

from .baseline import B0Snapshot
from .constants import PRODUCT_VERSION, SCHEMA_VERSION
from .domain import DiscreteRange, LifecycleState, TaskType
from .hardware_control import RecoveryRequired
from .repository import (
    ObservationRecord,
    TaskConcurrencyError,
    TaskStore,
    TaskStoreError,
)
from .sweep import (
    FIR_TAP_NAMES,
    SweepExecutor,
    SweepHardware,
    SweepMode,
    SweepPlan,
    SweepPoint,
    SweepPointRecord,
    SweepRunState,
    SweepRunner,
    create_bert_fir_sweep,
    create_i2c_sweep,
    export_sweep_artifacts,
)


class BaselineMeasuringSweepExecutor(SweepExecutor, Protocol):
    def measure_b0(self) -> Mapping[str, object]: ...


SweepExecutorFactory = Callable[
    [B0Snapshot, SweepPlan], BaselineMeasuringSweepExecutor
]
TerminalCallback = Callable[[str, LifecycleState], None]


class AIExecution(Protocol):
    def run(self, stop_requested: Event) -> LifecycleState | None: ...

    def restore_b0(self) -> None: ...


AIExecutionFactory = Callable[[str, B0Snapshot, Mapping[str, Any]], AIExecution]


def build_sweep_plan(payload: Mapping[str, Any], b0: B0Snapshot) -> SweepPlan:
    """Rebuild the exact sweep solely from persisted request data and B0."""

    try:
        hardware = SweepHardware(payload["sweep_hardware"])
        mode = SweepMode(payload["sweep_mode"])
        seed = payload["seed"]
        max_points = payload["max_points"]
        raw_parameters = payload["parameters"]
    except (KeyError, TypeError, ValueError) as error:
        raise TaskStoreError("persisted sweep configuration is malformed") from error
    if not isinstance(raw_parameters, Sequence) or isinstance(
        raw_parameters, (str, bytes)
    ):
        raise TaskStoreError("persisted sweep parameters must be a sequence")
    try:
        ranges = tuple(
            (
                parameter["name"],
                DiscreteRange(
                    parameter["min"],
                    parameter["max"],
                    parameter["step"],
                ),
            )
            for parameter in raw_parameters
        )
        if hardware is SweepHardware.I2C_SWEEP:
            return create_i2c_sweep(
                ranges=ranges,
                b0=dict(b0.i2c_fields),
                mode=mode,
                seed=seed,
                max_points=max_points,
            )
        return create_bert_fir_sweep(
            ranges=ranges,
            b0=b0.fir_taps,
            mode=mode,
            seed=seed,
            max_points=max_points,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TaskStoreError("persisted sweep plan is invalid") from error


class FakeSweepExecutor:
    """Deterministic, connection-free FAKE sweep execution backend."""

    def __init__(self, b0: B0Snapshot, plan: SweepPlan) -> None:
        self._b0 = b0
        self._plan = plan
        self._baseline = self._state_from_b0()
        self._current = dict(self._baseline)
        self.calls: list[tuple[object, ...]] = []

    def _state_from_b0(self) -> dict[str, int]:
        if self._plan.hardware is SweepHardware.I2C_SWEEP:
            return dict(self._b0.i2c_fields)
        return dict(zip(FIR_TAP_NAMES, self._b0.fir_taps, strict=True))

    def preflight(self) -> None:
        self.calls.append(("preflight",))

    def is_at_b0(self) -> bool:
        self.calls.append(("is_at_b0",))
        return self._current == self._baseline

    def restore_b0(self) -> None:
        self.calls.append(("restore_b0",))
        self._current = dict(self._baseline)

    def restore_parameter(self, name: str) -> None:
        if name not in self._baseline:
            raise ValueError(f"unknown fake sweep parameter {name}")
        self.calls.append(("restore_parameter", name))
        self._current[name] = self._baseline[name]

    def apply_point(self, point: SweepPoint) -> Mapping[str, object]:
        requested = dict(point.requested)
        if requested.keys() != self._baseline.keys() or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in requested.values()
        ):
            raise ValueError("fake sweep point does not match the frozen B0 shape")
        self.calls.append(("apply", point.original_index, dict(requested)))
        self._current = requested
        return {
            "requested": requested,
            "applied": dict(requested),
            "readback": dict(self._current),
        }

    def _measurement(self, phase: str) -> dict[str, object]:
        delta = sum(
            abs(self._current[name] - baseline)
            for name, baseline in self._baseline.items()
        )
        tdecq = round(2.4 + delta * 0.017, 6)
        er = round(5.8 - min(delta, 100) * 0.004, 6)
        return {
            "requested": dict(self._current),
            "applied": dict(self._current),
            "readback": dict(self._current),
            "tdecq": tdecq,
            "er": er,
            "raw": {
                "fake": True,
                "phase": phase,
                "state": dict(self._current),
            },
        }

    def measure_b0(self) -> Mapping[str, object]:
        self.calls.append(("measure_b0",))
        return self._measurement("B0")

    def measure_point(self) -> Mapping[str, object]:
        self.calls.append(("measure_point",))
        return self._measurement("SWEEP")


@dataclass(frozen=True, slots=True)
class _PreparedSweep:
    task_id: str
    b0: B0Snapshot
    plan: SweepPlan


@dataclass(frozen=True, slots=True)
class _PreparedAI:
    task_id: str
    b0: B0Snapshot
    payload: Mapping[str, Any]
    execution: AIExecution


class _ArtifactRefreshingSink:
    def __init__(
        self,
        *,
        store: TaskStore,
        task_id: str,
        plan: SweepPlan,
        artifact_root: Path,
    ) -> None:
        self._store = store
        self._task_id = task_id
        self._plan = plan
        self._artifact_root = artifact_root

    def save_sweep_point(self, record: SweepPointRecord) -> None:
        self._store.save_sweep_point(self._task_id, record)
        self.refresh()

    def refresh(self) -> None:
        """Export the complete persisted result set, even after an all-skip resume."""

        task = self._store.require_task(self._task_id)
        records = self._store.sweep_points(
            self._task_id,
            limit=max(1, self._plan.point_count),
        ).items
        if task.b0_hash is None:
            raise TaskStoreError("sweep task has no persisted B0 hash")
        export_sweep_artifacts(
            directory=self._artifact_root / self._task_id,
            task_id=self._task_id,
            plan=self._plan,
            records=records,
            product_version=PRODUCT_VERSION,
            schema_version=SCHEMA_VERSION,
            runtime_mode=task.runtime_mode.value,
            b0_hash=task.b0_hash,
            configuration_fingerprint=task.config_fingerprint,
        )


class PersistentV11TaskRunner:
    """One-worker, no-queue persistent runner for V1.1 execution."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        artifact_root: str | Path,
        sweep_executor_factory: SweepExecutorFactory | None = None,
        ai_execution_factory: AIExecutionFactory | None = None,
    ) -> None:
        self._database_path = Path(database_path)
        self._artifact_root = Path(artifact_root)
        self._sweep_executor_factory = sweep_executor_factory
        self._ai_execution_factory = ai_execution_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="v1.1-task",
        )
        self._lock = RLock()
        self._stop_requested = Event()
        self._future: Future[None] | None = None
        self._prepared: _PreparedSweep | _PreparedAI | None = None
        self._terminal_callback: TerminalCallback = lambda _task_id, _state: None
        self._closed = False
        self.last_sweep_executor: BaselineMeasuringSweepExecutor | None = None

    def set_terminal_callback(self, callback: TerminalCallback) -> None:
        with self._lock:
            if self._closed:
                raise TaskStoreError("task runner is closed")
            self._terminal_callback = callback

    def capabilities(self) -> Mapping[str, object]:
        return {
            "sweep_execution_ready": self._sweep_executor_factory is not None,
            "ai_execution_ready": self._ai_execution_factory is not None,
        }

    def health(self) -> Mapping[str, object]:
        with self._lock:
            active = self._future is not None and not self._future.done()
            closed = self._closed
        return {
            **self.capabilities(),
            "task_worker_open": not closed,
            "task_worker_active": active,
        }

    def _ensure_available(self, *, allow_prepared: bool = False) -> None:
        if self._closed:
            raise TaskStoreError("task runner is closed")
        if self._future is not None and not self._future.done():
            raise TaskStoreError("V1.1 runner already has an active task")
        if self._prepared is not None and not allow_prepared:
            raise TaskStoreError("V1.1 runner already has a prepared task")

    def _validated_ai_execution(
        self,
        task_id: str,
        b0: B0Snapshot,
        payload: Mapping[str, Any],
    ) -> AIExecution:
        factory = self._ai_execution_factory
        if factory is None:
            raise TaskStoreError("AI execution backend is unavailable")
        execution = factory(task_id, b0, payload)
        run = getattr(execution, "run", None)
        restore = getattr(execution, "restore_b0", None)
        if not callable(run) or not callable(restore):
            raise TaskStoreError(
                "AI execution must provide cooperative run and restore_b0"
            )
        try:
            signature(run).bind(self._stop_requested)
        except (TypeError, ValueError) as error:
            raise TaskStoreError(
                "AI execution run must accept a cooperative stop Event"
            ) from error
        return execution

    def start(self, task_id: str, task_type: TaskType) -> None:
        """Synchronously validate and prepare; do not launch or mutate."""

        with self._lock:
            self._ensure_available()
        with TaskStore(self._database_path) as store:
            task = store.require_task(task_id)
            b0 = store.load_b0(task_id)
            if task_type is TaskType.SWEEP:
                if self._sweep_executor_factory is None:
                    raise TaskStoreError("sweep execution backend is unavailable")
                prepared: _PreparedSweep | _PreparedAI = _PreparedSweep(
                    task_id=task_id,
                    b0=b0,
                    plan=build_sweep_plan(task.payload, b0),
                )
            else:
                payload = dict(task.payload)
                prepared = _PreparedAI(
                    task_id=task_id,
                    b0=b0,
                    payload=payload,
                    execution=self._validated_ai_execution(
                        task_id,
                        b0,
                        payload,
                    ),
                )
        with self._lock:
            self._ensure_available()
            self._prepared = prepared
            self._stop_requested.clear()

    def run_started(self, task_id: str) -> None:
        """Launch only after the service has durably persisted RUNNING."""

        with self._lock:
            self._ensure_available(allow_prepared=True)
            prepared = self._prepared
            if prepared is None or prepared.task_id != task_id:
                raise TaskStoreError("task was not prepared for dispatch")
            self._prepared = None
            target = (
                self._run_sweep
                if isinstance(prepared, _PreparedSweep)
                else self._run_ai
            )
            self._future = self._executor.submit(target, prepared)

    def cancel_prepared(self, task_id: str) -> None:
        """Discard an unpublished preparation after lifecycle commit loses."""

        with self._lock:
            prepared = self._prepared
            if prepared is None:
                return
            if prepared.task_id != task_id:
                raise TaskStoreError(
                    f"prepared task {prepared.task_id} does not match {task_id}"
                )
            self._prepared = None
            self._stop_requested.clear()

    def _persist_b0_once(
        self,
        store: TaskStore,
        prepared: _PreparedSweep,
        executor: BaselineMeasuringSweepExecutor,
    ) -> None:
        if any(item.phase == "B0" for item in store.observations(prepared.task_id)):
            return
        measurement = executor.measure_b0()
        record = ObservationRecord.measurement(
            observation_id=f"{prepared.task_id}:B0",
            sequence=1,
            phase="B0",
            requested=dict(measurement.get("requested", {})),
            applied=dict(measurement.get("applied", {})),
            readback=dict(measurement.get("readback", {})),
            tdecq=measurement.get("tdecq"),
            er=measurement.get("er"),
            raw_response=measurement.get("raw"),
            created_at=datetime.now(timezone.utc),
        )
        store.append_observation(prepared.task_id, record)

    def _transition_terminal(
        self,
        store: TaskStore,
        task_id: str,
        state: LifecycleState,
        event_type: str,
        payload: Mapping[str, object] | None = None,
    ) -> bool:
        try:
            store.transition(
                task_id,
                expected_state=LifecycleState.RUNNING,
                new_state=state,
                event_type=event_type,
                payload={} if payload is None else payload,
            )
        except TaskConcurrencyError:
            return False
        try:
            self._terminal_callback(task_id, state)
        except Exception:
            # Durable terminal persistence already won the RUNNING CAS. Keep
            # authority fail-closed, but do not strand the worker Future.
            pass
        return True

    def _restore_ai_for_control(
        self,
        store: TaskStore,
        prepared: _PreparedAI,
        execution: AIExecution,
    ) -> None:
        try:
            execution.restore_b0()
        except Exception as rollback_error:
            self._transition_terminal(
                store,
                prepared.task_id,
                LifecycleState.RECOVERY_REQUIRED,
                "AI_ROLLBACK_FAILED",
                {"rollback_error": str(rollback_error)},
            )
            raise RecoveryRequired(
                "cooperative AI stop could not restore B0",
                rollback_verified=False,
            ) from rollback_error

    def _run_sweep(self, prepared: _PreparedSweep) -> None:
        factory = self._sweep_executor_factory
        assert factory is not None
        try:
            worker_store = TaskStore(self._database_path)
        except Exception as error:
            # No executor exists yet, so this path cannot have mutated hardware.
            # Use a fresh connection to keep the durable lifecycle from being
            # stranded at RUNNING if the worker connection failed to open.
            with TaskStore(self._database_path) as recovery_store:
                self._transition_terminal(
                    recovery_store,
                    prepared.task_id,
                    LifecycleState.FAILED,
                    "TASK_SETUP_FAILED_B0_UNCHANGED",
                    {"error": str(error), "failure_point": "worker_store"},
                )
            return
        with worker_store as store:
            try:
                executor = factory(prepared.b0, prepared.plan)
            except Exception as error:
                # Factory construction precedes every hardware operation.
                self._transition_terminal(
                    store,
                    prepared.task_id,
                    LifecycleState.FAILED,
                    "TASK_SETUP_FAILED_B0_UNCHANGED",
                    {"error": str(error), "failure_point": "sweep_factory"},
                )
                return
            self.last_sweep_executor = executor
            try:
                artifact_sink = _ArtifactRefreshingSink(
                    store=store,
                    task_id=prepared.task_id,
                    plan=prepared.plan,
                    artifact_root=self._artifact_root,
                )
                sweep_runner = SweepRunner(
                    executor=executor,
                    store=artifact_sink,
                )
                sweep_runner.prepare_at_b0()
                self._persist_b0_once(store, prepared, executor)
                existing = store.sweep_points(
                    prepared.task_id,
                    limit=max(1, prepared.plan.point_count),
                ).items
                result = sweep_runner.run(
                    prepared.plan,
                    completed_original_indexes=tuple(
                        point.original_index for point in existing
                    ),
                    pause_requested=self._stop_requested.is_set,
                    preflight_done=True,
                )
                if result.state is SweepRunState.COMPLETED:
                    artifact_sink.refresh()
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.COMPLETED,
                        "TASK_COMPLETED_B0_RESTORED",
                    )
            except RecoveryRequired as error:
                transitioned = self._transition_terminal(
                    store,
                    prepared.task_id,
                    LifecycleState.RECOVERY_REQUIRED,
                    "TASK_ROLLBACK_FAILED",
                    {"error": str(error)},
                )
                if not transitioned:
                    raise
            except Exception as error:
                try:
                    executor.restore_b0()
                except Exception as rollback_error:
                    transitioned = self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.RECOVERY_REQUIRED,
                        "TASK_ROLLBACK_FAILED",
                        {
                            "error": str(error),
                            "rollback_error": str(rollback_error),
                        },
                    )
                    if not transitioned:
                        raise RecoveryRequired(
                            "sweep execution failed and B0 restore failed",
                            rollback_verified=False,
                        ) from rollback_error
                else:
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.FAILED,
                        "TASK_EXECUTION_FAILED_B0_RESTORED",
                        {"error": str(error)},
                    )

    def _run_ai(self, prepared: _PreparedAI) -> None:
        with TaskStore(self._database_path) as store:
            execution = prepared.execution
            try:
                result = execution.run(self._stop_requested)
            except RecoveryRequired as error:
                transitioned = self._transition_terminal(
                    store,
                    prepared.task_id,
                    LifecycleState.RECOVERY_REQUIRED,
                    "AI_ROLLBACK_FAILED",
                    {"error": str(error)},
                )
                if not transitioned:
                    raise
                return
            except Exception as error:
                try:
                    execution.restore_b0()
                except Exception as rollback_error:
                    transitioned = self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.RECOVERY_REQUIRED,
                        "AI_ROLLBACK_FAILED",
                        {
                            "error": str(error),
                            "rollback_error": str(rollback_error),
                        },
                    )
                    if not transitioned:
                        raise RecoveryRequired(
                            "AI execution failed and B0 restore failed",
                            rollback_verified=False,
                        ) from rollback_error
                else:
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.FAILED,
                        "AI_EXECUTION_FAILED_B0_RESTORED",
                        {"error": str(error)},
                    )
                return
            if (
                self._stop_requested.is_set()
                or store.require_task(prepared.task_id).state
                is not LifecycleState.RUNNING
            ):
                self._restore_ai_for_control(store, prepared, execution)
                return
            terminal = LifecycleState.COMPLETED if result is None else result
            allowed_terminal = {
                LifecycleState.COMPLETED,
                LifecycleState.COMPLETED_NO_IMPROVEMENT,
                LifecycleState.FAILED,
                LifecycleState.FAILED_INVALID_DCA_TAPS,
                LifecycleState.RECOVERY_REQUIRED,
            }
            if (
                not isinstance(terminal, LifecycleState)
                or terminal not in allowed_terminal
            ):
                error = TaskStoreError(
                    "AI execution returned an invalid terminal lifecycle state"
                )
                try:
                    execution.restore_b0()
                except Exception as rollback_error:
                    transitioned = self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.RECOVERY_REQUIRED,
                        "AI_ROLLBACK_FAILED",
                        {
                            "error": str(error),
                            "rollback_error": str(rollback_error),
                        },
                    )
                    if not transitioned:
                        raise RecoveryRequired(
                            "invalid AI result and B0 restore failed",
                            rollback_verified=False,
                        ) from rollback_error
                else:
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.FAILED,
                        "AI_EXECUTION_FAILED_B0_RESTORED",
                        {"error": str(error)},
                    )
                return
            if terminal is LifecycleState.RECOVERY_REQUIRED:
                self._transition_terminal(
                    store,
                    prepared.task_id,
                    terminal,
                    "AI_EXECUTION_TERMINAL",
                )
                return
            try:
                execution.restore_b0()
            except Exception as rollback_error:
                transitioned = self._transition_terminal(
                    store,
                    prepared.task_id,
                    LifecycleState.RECOVERY_REQUIRED,
                    "AI_ROLLBACK_FAILED",
                    {"rollback_error": str(rollback_error)},
                )
                if not transitioned:
                    raise RecoveryRequired(
                        "AI terminal B0 restore failed",
                        rollback_verified=False,
                    ) from rollback_error
                return
            try:
                transitioned = self._transition_terminal(
                    store,
                    prepared.task_id,
                    terminal,
                    "AI_EXECUTION_TERMINAL",
                )
            except Exception as persistence_error:
                self._transition_terminal(
                    store,
                    prepared.task_id,
                    LifecycleState.FAILED,
                    "AI_TERMINAL_PERSISTENCE_FAILED_B0_RESTORED",
                    {"error": str(persistence_error)},
                )
                return
            if not transitioned:
                # B0 was restored before CAS; pause/abort owns lifecycle.
                return

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        with self._lock:
            future = self._future
        if future is None:
            return True
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            return False
        return True

    def pause_and_restore_b0(self, task_id: str) -> None:
        self._request_stop_and_wait(task_id)

    def abort_and_restore_b0(self, task_id: str) -> None:
        self._request_stop_and_wait(task_id)

    def _request_stop_and_wait(self, task_id: str) -> None:
        with self._lock:
            future = self._future
            if future is None:
                raise TaskStoreError(f"task {task_id} is not running")
            self._stop_requested.set()
        future.result()

    def resume_from_b0(self, task_id: str) -> None:
        with TaskStore(self._database_path) as store:
            task_type = store.require_task(task_id).task_type
        self.start(task_id, task_type)
        self.run_started(task_id)

    def recover_b0(self, task_id: str) -> None:
        with TaskStore(self._database_path) as store:
            task = store.require_task(task_id)
            b0 = store.load_b0(task_id)
            if task.task_type is TaskType.AI_OPTIMIZATION:
                # Reconstruct solely from durable task/B0 evidence.  A
                # pre-crash execution object is never trusted for recovery.
                execution = self._validated_ai_execution(
                    task_id,
                    b0,
                    dict(task.payload),
                )
                execution.restore_b0()
                return
            factory = self._sweep_executor_factory
            if factory is None:
                raise TaskStoreError("sweep execution backend is unavailable")
            plan = build_sweep_plan(task.payload, b0)
        executor = factory(b0, plan)
        executor.restore_b0()
        if not executor.is_at_b0():
            raise RecoveryRequired(
                "B0 recovery readback mismatch",
                rollback_verified=False,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_requested.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
