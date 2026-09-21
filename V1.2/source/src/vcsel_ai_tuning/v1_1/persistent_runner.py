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
import json
import logging
import os
from pathlib import Path
from threading import Event, RLock
from typing import Any, Protocol

import numpy as np

from .baseline import B0Snapshot
from .constants import PRODUCT_VERSION, SCHEMA_VERSION
from .dataset import (
    FrozenDatasetCapture,
    MeasurementArtifactBundleWriter,
    MeasurementArtifactContext,
    MeasurementStatus,
    NoOpDatasetRecorder,
    encode_density_png,
)
from .domain import DiscreteRange, LifecycleState, TaskType
from .hardware_control import RecoveryRequired
from .live_ai_policy import LiveAIProfile, validate_live_ai_payload
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


logger = logging.getLogger(__name__)


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
        settle_time_s = payload.get("settle_time_s", 0.0)
        campaign_id = payload["task_id"]
        save_dataset = payload.get("save_dataset", False)
        raw_parameters = payload["parameters"]
    except (KeyError, TypeError, ValueError) as error:
        raise TaskStoreError("persisted sweep configuration is malformed") from error
    if not isinstance(raw_parameters, Sequence) or isinstance(
        raw_parameters, (str, bytes)
    ):
        raise TaskStoreError("persisted sweep parameters must be a sequence")
    if type(save_dataset) is not bool:
        raise TaskStoreError("persisted save_dataset must be a boolean")
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
                settle_time_s=settle_time_s,
                campaign_id=campaign_id,
                save_dataset=save_dataset,
            )
        return create_bert_fir_sweep(
            ranges=ranges,
            b0=b0.fir_taps,
            mode=mode,
            seed=seed,
            max_points=max_points,
            settle_time_s=settle_time_s,
            campaign_id=campaign_id,
            save_dataset=save_dataset,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TaskStoreError("persisted sweep plan is invalid") from error


class FakeSweepExecutor:
    """Deterministic, connection-free FAKE sweep execution backend."""

    def __init__(
        self,
        b0: B0Snapshot,
        plan: SweepPlan,
        *,
        artifact_root: str | Path | None = None,
        dataset_capture_hook: Callable[..., object] | None = None,
    ) -> None:
        self._b0 = b0
        self._plan = plan
        self._baseline = self._state_from_b0()
        self._current = dict(self._baseline)
        if plan.save_dataset and artifact_root is None:
            raise ValueError(
                "save_dataset=true requires an explicit FAKE artifact root"
            )
        self._dataset_recorder = (
            MeasurementArtifactBundleWriter(artifact_root)
            if plan.save_dataset
            else NoOpDatasetRecorder()
        )
        self._dataset_capture_hook = dataset_capture_hook
        self._pending_point: SweepPoint | None = None
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
        self._pending_point = None

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
        self._pending_point = point
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
        point = self._pending_point
        if point is None:
            raise RuntimeError("no fake sweep point is awaiting measurement")
        measured = self._measurement("SWEEP")
        point_id = f"point-{point.original_index:08d}"
        measurement_id = f"{self._plan.campaign_id}:{point_id}:measurement"
        acquisition_id = f"{self._plan.campaign_id}:{point_id}:acquisition"
        identities = {
            item.kind: item for item in self._b0.hardware_identities
        }
        context = MeasurementArtifactContext(
            campaign_id=self._plan.campaign_id,
            sweep_point_id=point_id,
            measurement_id=measurement_id,
            acquisition_id=acquisition_id,
            captured_at_utc=datetime.now(timezone.utc),
            requested_params=measured["requested"],
            applied_params=measured["applied"],
            readback_params=measured["readback"],
            tdecq=float(measured["tdecq"]),
            outer_er=float(measured["er"]),
            outer_er_source=self._b0.dca_outer_er_source,
            dca_fingerprint=self._b0.dca_fingerprint.payload(),
            dca_identity=identities["dca"].identity,
            bert_port=self._b0.bert_port,
            bert_channel=self._b0.bert_channel,
            bert_pattern=self._b0.bert_pattern,
            measurement_status=MeasurementStatus.SUCCESS,
            dca_acquisition_duration_s=0.0,
        )

        def capture() -> FrozenDatasetCapture:
            self.calls.append(("dataset_capture", point.original_index))
            if self._dataset_capture_hook is not None:
                supplied = self._dataset_capture_hook(
                    acquisition_id,
                    measured,
                    dict(self._current),
                )
                if isinstance(supplied, FrozenDatasetCapture):
                    return supplied
            density = np.arange(48, dtype=np.float32).reshape(6, 8)
            density = density + np.float32(point.original_index)
            equalized = encode_density_png(np.square(density + 1.0))
            return FrozenDatasetCapture(
                acquisition_id=acquisition_id,
                same_acquisition_verified=True,
                raw_eye_density=density,
                density_axis_order=("amplitude", "time"),
                density_source="FAKE_REPLAY_CHAN1A",
                tdecq_equalized_eye_png=equalized,
                rlm=0.97,
                oma=0.91,
                average_power=0.50,
                equalizer_taps=tuple(self._b0.fir_taps),
                equalizer_main_index=3,
                equalizer_source=(
                    f"FUNC{self._b0.dca_tdecq_equalizer_function}"
                    if self._b0.dca_tdecq_equalizer_function is not None
                    else self._b0.dca_tdecq_source
                ),
                raw_responses={"fake": True, "replay": "fixed-v1"},
            )

        outcome = self._dataset_recorder.record(context, capture)
        measured.update(
            measurement_id=measurement_id,
            acquisition_id=acquisition_id,
            measurement_status=MeasurementStatus.SUCCESS.value,
            dataset_status=outcome.status.value,
            dataset_error=outcome.error,
            dataset_bundle_path=outcome.bundle_path,
        )
        raw = dict(measured["raw"])
        raw["dataset"] = outcome.payload()
        measured["raw"] = raw
        return measured


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
        executor: SweepExecutor,
    ) -> None:
        self._store = store
        self._task_id = task_id
        self._plan = plan
        self._artifact_root = artifact_root
        self._executor = executor

    def save_sweep_point(self, record: SweepPointRecord) -> None:
        self._store.save_sweep_point(self._task_id, record)
        self.refresh()

    def update_sweep_point(self, record: SweepPointRecord) -> None:
        self._store.update_sweep_point(self._task_id, record)
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
        evidence = getattr(self._executor, "audit_evidence", None)
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
            run_evidence=evidence() if callable(evidence) else None,
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
        self._closed_ai_executions: list[object] = []
        self._inflight_ai_execution: object | None = None
        self._terminal_callback: TerminalCallback = lambda _task_id, _state: None
        self._closed = False
        self.last_sweep_executor: BaselineMeasuringSweepExecutor | None = None

    @property
    def database_path(self) -> Path:
        """Return the task store shared with the application service."""

        return self._database_path

    @property
    def artifact_root(self) -> Path:
        """Return the artifact root shared with the application service."""

        return self._artifact_root

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

    def _close_ai_execution_once(self, execution: object) -> None:
        close = getattr(execution, "close", None)
        if not callable(close):
            return
        with self._lock:
            if any(item is execution for item in self._closed_ai_executions):
                return
            self._closed_ai_executions.append(execution)
        try:
            close()
        except Exception:
            logger.exception("failed to close AI execution")

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
                profile = payload.get("live_ai_profile")
                if profile is not None:
                    validate_live_ai_payload(LiveAIProfile(profile), payload)
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
        try:
            with self._lock:
                self._ensure_available()
                self._prepared = prepared
                self._stop_requested.clear()
        except Exception:
            if isinstance(prepared, _PreparedAI):
                self._close_ai_execution_once(prepared.execution)
            raise

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
            try:
                if isinstance(prepared, _PreparedAI):
                    self._inflight_ai_execution = prepared.execution
                self._future = self._executor.submit(target, prepared)
            except Exception:
                self._inflight_ai_execution = None
                if isinstance(prepared, _PreparedAI):
                    self._close_ai_execution_once(prepared.execution)
                raise

    def cancel_prepared(self, task_id: str) -> None:
        """Discard an unpublished preparation after lifecycle commit loses."""

        execution_to_close = None
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
            if isinstance(prepared, _PreparedAI):
                execution_to_close = prepared.execution
        if execution_to_close is not None:
            self._close_ai_execution_once(execution_to_close)

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

    def _write_sweep_failure_fallback(
        self,
        task_id: str,
        error: Exception,
    ) -> tuple[str | None, str | None]:
        """Durably preserve point evidence when its database UPDATE failed."""

        evidence = getattr(error, "evidence", None)
        if not isinstance(evidence, Mapping) or not evidence.get(
            "evidence_persistence_error"
        ):
            return None, None
        directory = self._artifact_root / task_id
        path = directory / "sweep_failure_evidence.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "product_version": PRODUCT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "campaign_type": "SWEEP",
            "task_id": task_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
            "rollback_verified": bool(
                getattr(error, "rollback_verified", False)
            ),
            "point_evidence": dict(evidence),
        }
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception as fallback_error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            logger.exception("failed to persist sweep failure fallback")
            return None, str(fallback_error)
        return str(path), None

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
        self._refresh_ai_artifact(
            store,
            prepared.task_id,
            execution,
            terminal_lifecycle=None,
        )

    @staticmethod
    def _refresh_ai_artifact(
        store: TaskStore,
        task_id: str,
        execution: AIExecution,
        *,
        terminal_lifecycle: LifecycleState | None,
    ) -> None:
        refresh = getattr(execution, "refresh_artifact", None)
        if not callable(refresh):
            return
        try:
            refresh(terminal_lifecycle)
        except Exception as artifact_error:
            try:
                store.append_audit_event(
                    task_id,
                    event_type="AI_ARTIFACT_EXPORT_FAILED",
                    payload={
                        "error": str(artifact_error),
                        "terminal_lifecycle": (
                            None
                            if terminal_lifecycle is None
                            else terminal_lifecycle.value
                        ),
                    },
                )
            except Exception:
                # Artifact evidence cannot redefine verified B0 restoration or
                # a lifecycle transition that already committed.
                pass

    @staticmethod
    def _ai_error_payload(error: BaseException) -> dict[str, object]:
        evidence = getattr(error, "evidence", None)
        return {
            "error": str(error),
            "error_type": type(error).__name__,
            "rollback_verified": bool(
                getattr(error, "rollback_verified", False)
            ),
            "evidence": dict(evidence) if isinstance(evidence, Mapping) else {},
        }

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
            artifact_sink: _ArtifactRefreshingSink | None = None
            try:
                artifact_sink = _ArtifactRefreshingSink(
                    store=store,
                    task_id=prepared.task_id,
                    plan=prepared.plan,
                    artifact_root=self._artifact_root,
                    executor=executor,
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
                fallback_artifact, fallback_artifact_error = (
                    self._write_sweep_failure_fallback(
                        prepared.task_id,
                        error,
                    )
                )
                if artifact_sink is not None:
                    try:
                        artifact_sink.refresh()
                    except Exception:
                        logger.exception(
                            "failed to refresh sweep artifact during recovery"
                        )
                transitioned = self._transition_terminal(
                    store,
                    prepared.task_id,
                    LifecycleState.RECOVERY_REQUIRED,
                    "TASK_ROLLBACK_FAILED",
                    {
                        "error": str(error),
                        "point_evidence": getattr(error, "evidence", None),
                        "fallback_artifact": fallback_artifact,
                        "fallback_artifact_error": fallback_artifact_error,
                    },
                )
                if not transitioned:
                    raise
            except Exception as error:
                fallback_artifact, fallback_artifact_error = (
                    self._write_sweep_failure_fallback(
                        prepared.task_id,
                        error,
                    )
                )
                rollback_already_verified = bool(
                    getattr(error, "rollback_verified", False)
                )
                rollback_error: Exception | None = None
                if not rollback_already_verified:
                    try:
                        executor.restore_b0()
                    except Exception as caught_rollback_error:
                        rollback_error = caught_rollback_error
                if rollback_error is not None:
                    transitioned = self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.RECOVERY_REQUIRED,
                        "TASK_ROLLBACK_FAILED",
                        {
                            "error": str(error),
                            "rollback_error": str(rollback_error),
                            "point_evidence": getattr(
                                error, "evidence", None
                            ),
                            "fallback_artifact": fallback_artifact,
                            "fallback_artifact_error": (
                                fallback_artifact_error
                            ),
                        },
                    )
                    if not transitioned:
                        raise RecoveryRequired(
                            "sweep execution failed and B0 restore failed",
                            rollback_verified=False,
                        ) from rollback_error
                else:
                    artifact_error: str | None = None
                    if artifact_sink is not None:
                        try:
                            artifact_sink.refresh()
                        except Exception as refresh_error:
                            artifact_error = str(refresh_error)
                            logger.exception(
                                "failed to refresh restored sweep artifact"
                            )
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.FAILED,
                        "TASK_EXECUTION_FAILED_B0_RESTORED",
                        {
                            "error": str(error),
                            "outcome": "FAILED_RESTORED",
                            "artifact_error": artifact_error,
                            "point_evidence": getattr(
                                error, "evidence", None
                            ),
                            "fallback_artifact": fallback_artifact,
                            "fallback_artifact_error": (
                                fallback_artifact_error
                            ),
                        },
                    )
            finally:
                close_executor = getattr(executor, "close", None)
                if callable(close_executor):
                    try:
                        close_executor()
                    except Exception:
                        logger.exception("failed to close live sweep executor")

    def _run_ai(self, prepared: _PreparedAI) -> None:
        try:
            self._run_ai_inner(prepared)
        finally:
            self._close_ai_execution_once(prepared.execution)
            with self._lock:
                if self._inflight_ai_execution is prepared.execution:
                    self._inflight_ai_execution = None

    def _run_ai_inner(self, prepared: _PreparedAI) -> None:
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
                    self._ai_error_payload(error),
                )
                if not transitioned:
                    raise
                return
            except Exception as error:
                error_payload = self._ai_error_payload(error)
                if error_payload["rollback_verified"] is True:
                    self._refresh_ai_artifact(
                        store,
                        prepared.task_id,
                        execution,
                        terminal_lifecycle=None,
                    )
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.FAILED,
                        "AI_EXECUTION_FAILED_B0_RESTORED",
                        error_payload,
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
                        {
                            **error_payload,
                            "rollback_error": str(rollback_error),
                        },
                    )
                    if not transitioned:
                        raise RecoveryRequired(
                            "AI execution failed and B0 restore failed",
                            rollback_verified=False,
                        ) from rollback_error
                else:
                    self._refresh_ai_artifact(
                        store,
                        prepared.task_id,
                        execution,
                        terminal_lifecycle=None,
                    )
                    self._transition_terminal(
                        store,
                        prepared.task_id,
                        LifecycleState.FAILED,
                        "AI_EXECUTION_FAILED_B0_RESTORED",
                        error_payload,
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
                    self._refresh_ai_artifact(
                        store,
                        prepared.task_id,
                        execution,
                        terminal_lifecycle=None,
                    )
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
            if getattr(execution, "terminal_b0_verified", False) is not True:
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
                self._refresh_ai_artifact(
                    store,
                    prepared.task_id,
                    execution,
                    terminal_lifecycle=None,
                )
                return
            if not transitioned:
                # B0 was restored before CAS; pause/abort owns lifecycle.
                self._refresh_ai_artifact(
                    store,
                    prepared.task_id,
                    execution,
                    terminal_lifecycle=None,
                )
                return
            self._refresh_ai_artifact(
                store,
                prepared.task_id,
                execution,
                terminal_lifecycle=terminal,
            )

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
                try:
                    execution.restore_b0()
                except Exception as rollback_error:
                    rollback_traceback = rollback_error.__traceback__
                    try:
                        store.append_audit_event(
                            task_id,
                            event_type="AI_RECOVERY_B0_FAILED",
                            payload={"rollback_error": str(rollback_error)},
                        )
                    except Exception:
                        logger.exception(
                            "failed to persist AI recovery B0 failure audit event"
                        )
                    raise rollback_error.with_traceback(rollback_traceback)
                return
            factory = self._sweep_executor_factory
            if factory is None:
                raise TaskStoreError("sweep execution backend is unavailable")
            plan = build_sweep_plan(task.payload, b0)
        executor = factory(b0, plan)
        try:
            executor.restore_b0()
            if not executor.is_at_b0():
                raise RecoveryRequired(
                    "B0 recovery readback mismatch",
                    rollback_verified=False,
                )
        finally:
            close_executor = getattr(executor, "close", None)
            if callable(close_executor):
                try:
                    close_executor()
                except Exception:
                    logger.exception("failed to close recovery sweep executor")

    def close(self) -> None:
        executions_to_close: list[object] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_requested.set()
            if isinstance(self._prepared, _PreparedAI):
                executions_to_close.append(self._prepared.execution)
            if self._inflight_ai_execution is not None:
                executions_to_close.append(self._inflight_ai_execution)
            self._prepared = None
        self._executor.shutdown(wait=True, cancel_futures=True)
        for execution in executions_to_close:
            self._close_ai_execution_once(execution)
