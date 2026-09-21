"""Application service for V1.1 lifecycle, authorization, and API calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from vcsel_ai_tuning.hardware_authority import (
    HardwareAuthority,
    HardwareAuthorityBusy,
    HardwareAuthorityLease,
)

from .authorization import (
    AuthorizationDriftError,
    AuthorizedParameter,
    FullCampaignAuthorization,
    OptimizationDepth,
    RollbackPolicy,
)
from .baseline import B0Snapshot, freeze_channel_mapping
from .constants import SCHEMA_VERSION
from .domain import DiscreteRange, LifecycleState, RuntimeMode, TaskType
from .history_apply import (
    HistoryApplyController,
    HistoryApplyRolledBackFailure,
    HistoryApplyRollbackFailure,
    HistoryHardwareState,
    HistoryResultTarget,
)
from .regmap import (
    ParameterMapping,
    ParameterSelection,
    validate_selected_parameters,
)
from .repository import RecoveryReasonCode, RecoveryStatus, TaskStore, TaskStoreError
from .repository import ActiveTaskConflict
from .sweep import (
    FIR_TAP_NAMES,
    SweepHardware,
    SweepMode,
    create_bert_fir_sweep,
    create_i2c_sweep,
)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hardware_state_hash(snapshot: B0Snapshot) -> str:
    payload = snapshot.payload()
    return _sha256(
        {
            "i2c_fields": payload["i2c_fields"],
            "i2c_raw_registers": payload["i2c_raw_registers"],
            "channel_mapping": payload["channel_mapping"],
            "fir": payload["fir"],
            "dca_fingerprint": payload["dca_fingerprint"],
            "hardware_identities": payload["hardware_identities"],
        }
    )


@dataclass(frozen=True, slots=True)
class PreflightCapture:
    snapshot_token: str
    b0: B0Snapshot
    mapping: ParameterMapping

    def __post_init__(self) -> None:
        if not self.snapshot_token:
            raise ValueError("snapshot_token must be nonblank")
        if not isinstance(self.b0, B0Snapshot):
            raise TypeError("b0 must be a B0Snapshot")
        if not isinstance(self.mapping, ParameterMapping):
            raise TypeError("mapping must be a ParameterMapping")


@dataclass(frozen=True, slots=True)
class GatewayAttestation:
    hardware_identities: Mapping[str, str]
    mapping_hash: str
    channel_mapping: Mapping[str, int | str]
    dca_config_fingerprint: str
    hardware_state_hash: str

    @classmethod
    def from_capture(cls, capture: PreflightCapture) -> "GatewayAttestation":
        return cls(
            hardware_identities={
                identity.kind: identity.identity
                for identity in capture.b0.hardware_identities
            },
            mapping_hash=capture.mapping.content_hash,
            channel_mapping=capture.b0.channel_mapping_payload(),
            dca_config_fingerprint=capture.b0.dca_fingerprint.content_hash,
            hardware_state_hash=_hardware_state_hash(capture.b0),
        )

    @classmethod
    def from_persisted_evidence(
        cls,
        b0: B0Snapshot,
        authorization: FullCampaignAuthorization,
    ) -> "GatewayAttestation":
        return cls(
            hardware_identities={
                identity.kind: identity.identity
                for identity in b0.hardware_identities
            },
            mapping_hash=authorization.mapping_hash,
            channel_mapping=b0.channel_mapping_payload(),
            dca_config_fingerprint=b0.dca_fingerprint.content_hash,
            hardware_state_hash=_hardware_state_hash(b0),
        )


class V11Gateway(Protocol):
    @property
    def runtime_mode(self) -> RuntimeMode: ...

    def health(self) -> Mapping[str, Any]: ...

    def capabilities(self) -> Mapping[str, Any]: ...

    def preflight_capture(self) -> PreflightCapture: ...

    def current_attestation(self) -> GatewayAttestation: ...


class TaskRunner(Protocol):
    def start(self, task_id: str, task_type: TaskType) -> None: ...

    def cancel_prepared(self, task_id: str) -> None: ...

    def pause_and_restore_b0(self, task_id: str) -> None: ...

    def resume_from_b0(self, task_id: str) -> None: ...

    def abort_and_restore_b0(self, task_id: str) -> None: ...

    def recover_b0(self, task_id: str) -> None: ...


class V11ApplicationService:
    """Synchronous control-plane service; execution is injected via TaskRunner."""

    def __init__(
        self,
        *,
        store: TaskStore,
        gateway: V11Gateway,
        runner: TaskRunner,
        artifact_root: str | Path | None = None,
        history_apply_controller: HistoryApplyController | None = None,
        external_active_task_id: Callable[[], str | None] | None = None,
        hardware_authority: HardwareAuthority | None = None,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.runner = runner
        self.artifact_root = Path(artifact_root or "artifacts/v1.1")
        self.history_apply_controller = history_apply_controller
        self.external_active_task_id = external_active_task_id or (lambda: None)
        self._hardware_authority = hardware_authority or HardwareAuthority()
        self._authority_lock = RLock()
        self._start_lock = RLock()
        self._task_lease: HardwareAuthorityLease | None = None
        self._history_recovery_lease: HardwareAuthorityLease | None = None
        self._history_recovery_id: str | None = None
        self._closed = False
        self._captures: dict[str, PreflightCapture] = {}
        self._task_b0: dict[str, B0Snapshot] = {}
        self._task_capture: dict[str, PreflightCapture] = {}
        self._authorizations: dict[str, FullCampaignAuthorization] = {}
        set_terminal_callback = getattr(
            self.runner, "set_terminal_callback", None
        )
        if callable(set_terminal_callback):
            set_terminal_callback(self._runner_terminal)
        # This is deliberately database-only.  No gateway method is called here.
        self.store.mark_interrupted_tasks_recovery_required()
        active_task_id = self.store.active_task_id()
        if active_task_id is not None:
            active_task = self.store.require_task(active_task_id)
            if active_task.b0_hash is not None:
                self._task_b0[active_task_id] = self.store.load_b0(active_task_id)
                if active_task.state in {
                    LifecycleState.AUTHORIZED,
                    LifecycleState.RECOVERY_REQUIRED,
                }:
                    self._authorizations[active_task_id] = (
                        self.store.load_authorization(active_task_id)
                    )
            if active_task.state is LifecycleState.RECOVERY_REQUIRED:
                try:
                    self._task_lease = self._hardware_authority.acquire(
                        self._task_owner(active_task_id)
                    )
                except HardwareAuthorityBusy:
                    # Recovery retries this acquisition and fails closed if the
                    # authority remains busy. No gateway method is called here.
                    pass
        open_history_recoveries = tuple(
            record
            for record in self.store.list_recovery_records(
                status=RecoveryStatus.OPEN
            )
            if record.reason_code
            is RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED
        )
        if open_history_recoveries:
            recovery = open_history_recoveries[0]
            self._history_recovery_id = recovery.recovery_id
            try:
                if self.history_apply_controller is not None:
                    self.history_apply_controller.reserve_recovery(
                        recovery.recovery_id
                    )
                else:
                    self._history_recovery_lease = self._hardware_authority.acquire(
                        f"history-recovery:{recovery.recovery_id}"
                    )
            except HardwareAuthorityBusy:
                # The OPEN record remains the durable fail-closed source of truth.
                pass

    @staticmethod
    def _task_owner(task_id: str) -> str:
        return f"v1.1:task:{task_id}"

    def _ensure_open(self) -> None:
        with self._authority_lock:
            if self._closed:
                raise TaskStoreError("service is closed")

    def _ensure_no_open_history_recovery(self) -> None:
        self._ensure_open()
        if (
            self.history_apply_controller is not None
            and self.history_apply_controller.has_unpersisted_recovery
        ):
            raise ActiveTaskConflict(
                self._hardware_authority.current_owner
                or "history-apply:unpersisted-recovery"
            )
        open_records = self.store.list_recovery_records(status=RecoveryStatus.OPEN)
        if open_records:
            raise ActiveTaskConflict(
                f"history-recovery:{open_records[0].recovery_id}"
            )

    @contextmanager
    def _hardware_access(self, owner: str):
        self._ensure_no_open_history_recovery()
        external = self.external_active_task_id()
        if external is not None:
            raise ActiveTaskConflict(f"v1:{external}")
        lease = self._acquire(owner)
        try:
            yield
        finally:
            lease.release()

    def _acquire(self, owner: str) -> HardwareAuthorityLease:
        with self._authority_lock:
            if self._closed:
                raise TaskStoreError("service is closed")
            try:
                return self._hardware_authority.acquire(owner)
            except HardwareAuthorityBusy as error:
                raise ActiveTaskConflict(error.current_owner) from error

    def _ensure_task_lease(self, task_id: str) -> HardwareAuthorityLease:
        self._ensure_no_open_history_recovery()
        with self._authority_lock:
            if self._task_lease is not None:
                if self._task_lease.owner != self._task_owner(task_id):
                    raise ActiveTaskConflict(self._task_lease.owner)
                return self._task_lease
            lease = self._acquire(self._task_owner(task_id))
            self._task_lease = lease
            return lease

    def _release_task_lease(self, task_id: str) -> None:
        with self._authority_lock:
            lease = self._task_lease
            if lease is None or lease.owner != self._task_owner(task_id):
                return
            self._task_lease = None
        lease.release()

    def _runner_terminal(
        self, task_id: str, state: LifecycleState
    ) -> None:
        """Receive worker terminal state without using the service DB connection."""

        if state is not LifecycleState.RECOVERY_REQUIRED:
            self._release_task_lease(task_id)

    @staticmethod
    def _request_payload(request: Any) -> dict[str, Any]:
        dumper = getattr(request, "model_dump", None)
        if not callable(dumper):
            raise TypeError("API request must provide model_dump")
        payload = dumper(mode="json")
        if not isinstance(payload, dict):
            raise TypeError("request payload must be an object")
        return payload

    @staticmethod
    def _task_payload(task: Any) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "state": task.state.value,
            "runtime_mode": task.runtime_mode.value,
            "product_version": task.product_version,
            "schema_version": task.schema_version,
            "b0_hash": task.b0_hash,
            "configuration_fingerprint": task.config_fingerprint,
            "config": task.payload,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    @staticmethod
    def _recovery_payload(record: Any) -> dict[str, Any]:
        return {
            "recovery_id": record.recovery_id,
            "task_id": record.task_id,
            "status": record.status.value,
            "resolved_at": (
                None
                if record.resolved_at is None
                else record.resolved_at.isoformat()
            ),
        }

    def runtime_health(self) -> dict[str, Any]:
        runner_health = getattr(self.runner, "health", None)
        payload = {
            "product_version": "1.1.0",
            "runtime_mode": self.gateway.runtime_mode.value,
            **dict(self.gateway.health()),
            **(
                dict(runner_health())
                if callable(runner_health)
                else {}
            ),
        }
        open_records = self.store.list_recovery_records(status=RecoveryStatus.OPEN)
        if open_records:
            payload.update(
                status="RECOVERY_REQUIRED",
                open_recovery_id=open_records[0].recovery_id,
            )
        elif (
            self.history_apply_controller is not None
            and self.history_apply_controller.has_unpersisted_recovery
        ):
            payload.update(
                status="RECOVERY_REQUIRED",
                open_recovery_id=None,
                recovery_persistence="FAILED",
            )
        return payload

    def runtime_capabilities(self) -> dict[str, Any]:
        runner_capabilities = getattr(self.runner, "capabilities", None)
        return {
            "product_version": "1.1.0",
            "schema_version": SCHEMA_VERSION,
            "runtime_mode": self.gateway.runtime_mode.value,
            **dict(self.gateway.capabilities()),
            **(
                dict(runner_capabilities())
                if callable(runner_capabilities)
                else {}
            ),
        }

    def create_task(self, request: Any) -> dict[str, Any]:
        with self._acquire(f"v1.1:create:{request.task_id}"):
            external = self.external_active_task_id()
            if external is not None:
                raise ActiveTaskConflict(f"v1:{external}")
            payload = self._request_payload(request)
            task = self.store.create_task(
                task_id=request.task_id,
                task_type=request.task_type,
                runtime_mode=self.gateway.runtime_mode,
                config_fingerprint=_sha256(payload),
                payload=payload,
            )
        return self._task_payload(task)

    def active_task(self) -> dict[str, Any] | None:
        task_id = self.store.active_task_id()
        return None if task_id is None else self._task_payload(
            self.store.require_task(task_id)
        )

    def hardware_snapshot(self) -> dict[str, Any]:
        with self._hardware_access("v1.1:hardware-snapshot"):
            capture = self.gateway.preflight_capture()
        self._captures[capture.snapshot_token] = capture
        return {
            "snapshot_token": capture.snapshot_token,
            "b0_hash": capture.b0.content_hash,
            "b0": capture.b0.payload(),
            "mapping_hash": capture.mapping.content_hash,
            "mapping": [binding.payload() for binding in capture.mapping.bindings],
        }

    def scan_parameter_mapping(self) -> dict[str, Any]:
        with self._hardware_access("v1.1:mapping-scan"):
            capture = self.gateway.preflight_capture()
        self._captures[capture.snapshot_token] = capture
        return {
            "snapshot_token": capture.snapshot_token,
            "mapping_hash": capture.mapping.content_hash,
            "parameters": [
                binding.payload() for binding in capture.mapping.bindings
            ],
        }

    def _latest_capture(self) -> PreflightCapture:
        if not self._captures:
            raise TaskStoreError("preflight/Script-RegMap scan is required")
        return tuple(self._captures.values())[-1]

    @staticmethod
    def _selections(parameters: Sequence[Any]) -> tuple[ParameterSelection, ...]:
        return tuple(
            ParameterSelection(
                name=parameter.name,
                minimum=parameter.min,
                maximum=parameter.max,
                step=parameter.step,
            )
            for parameter in parameters
        )

    def validate_parameters(self, parameters: Sequence[Any]) -> dict[str, Any]:
        capture = self._latest_capture()
        validated = validate_selected_parameters(
            self._selections(parameters), capture.mapping
        )
        return {
            "mapping_hash": capture.mapping.content_hash,
            "parameters": [
                {
                    "name": item.name,
                    "min": item.value_range.minimum,
                    "max": item.value_range.maximum,
                    "step": item.value_range.step,
                    "values": list(item.values),
                    "script_order": item.binding.script_order,
                }
                for item in validated
            ],
        }

    def save_b0(self, task_id: str, request: Any) -> dict[str, Any]:
        if request.operator_confirmed is not True:
            raise ValueError("operator confirmation is required")
        task = self.store.require_task(task_id)
        if task.state is not LifecycleState.DRAFT:
            raise TaskStoreError("B0 can only be saved from DRAFT")
        try:
            capture = self._captures[request.snapshot_token]
        except KeyError as error:
            raise TaskStoreError("unknown or expired snapshot token") from error
        raw_mapping = task.payload.get("channel_mapping")
        if not isinstance(raw_mapping, Mapping):
            raise TaskStoreError("task has no operator-confirmed channel mapping")
        confirmed_mapping = dict(raw_mapping)
        if confirmed_mapping.pop("operator_confirmed", None) is not True:
            raise TaskStoreError("task has no operator-confirmed channel mapping")
        try:
            frozen_mapping = freeze_channel_mapping(confirmed_mapping)
        except (TypeError, ValueError) as error:
            raise TaskStoreError("task channel mapping is invalid") from error
        if frozen_mapping != freeze_channel_mapping(
            capture.b0.channel_mapping_payload()
        ):
            raise TaskStoreError(
                "operator-confirmed channel mapping does not match hardware snapshot"
            )
        self.store.save_b0(task_id, capture.b0)
        self.store.transition(
            task_id,
            expected_state=LifecycleState.DRAFT,
            new_state=LifecycleState.PREFLIGHTED,
            event_type="PREFLIGHT_CONFIRMED",
            payload={"snapshot_token": capture.snapshot_token},
        )
        task = self.store.transition(
            task_id,
            expected_state=LifecycleState.PREFLIGHTED,
            new_state=LifecycleState.B0_FROZEN,
            event_type="B0_FROZEN",
            payload={"b0_hash": capture.b0.content_hash},
        )
        self._task_b0[task_id] = capture.b0
        self._task_capture[task_id] = capture
        return self._task_payload(task)

    def count_sweep(self, request: Any) -> dict[str, Any]:
        capture = self._latest_capture()
        parameters = self._selections(request.parameters)
        if request.hardware is SweepHardware.I2C_SWEEP:
            validated = validate_selected_parameters(parameters, capture.mapping)
            ranges = tuple((item.name, item.value_range) for item in validated)
            plan = create_i2c_sweep(
                ranges=ranges,
                b0=dict(capture.b0.i2c_fields),
                mode=request.mode,
                seed=request.seed,
                max_points=request.max_points,
            )
        else:
            unknown = sorted(
                selection.name
                for selection in parameters
                if selection.name not in FIR_TAP_NAMES
            )
            if unknown:
                raise ValueError(
                    "unknown FIR tap parameter(s): " + ", ".join(unknown)
                )
            ranges = tuple(
                (
                    selection.name,
                    DiscreteRange(
                        selection.minimum,
                        selection.maximum,
                        selection.step,
                    ),
                )
                for selection in parameters
            )
            plan = create_bert_fir_sweep(
                ranges=ranges,
                b0=capture.b0.fir_taps,
                mode=request.mode,
                seed=request.seed,
                max_points=request.max_points,
            )
        return {
            "hardware": plan.hardware.value,
            "mode": plan.mode.value,
            "seed": plan.seed,
            "point_count": plan.point_count,
            "skipped_constraint_count": sum(
                point.constraint_reason is not None for point in plan.points
            ),
        }

    def _task_parameters(self, task: Any, mapping: ParameterMapping):
        raw = task.payload.get("parameters", [])
        selections = tuple(ParameterSelection.from_payload(item) for item in raw)
        return validate_selected_parameters(selections, mapping)

    @staticmethod
    def _assert_attestation(
        expected: GatewayAttestation, current: GatewayAttestation
    ) -> None:
        mismatches: list[str] = []
        if dict(expected.hardware_identities) != dict(current.hardware_identities):
            mismatches.append("hardware_identities")
        if expected.mapping_hash != current.mapping_hash:
            mismatches.append("mapping_hash")
        if dict(expected.channel_mapping) != dict(current.channel_mapping):
            mismatches.append("channel_mapping")
        if expected.dca_config_fingerprint != current.dca_config_fingerprint:
            mismatches.append("dca_config_fingerprint")
        if expected.hardware_state_hash != current.hardware_state_hash:
            mismatches.append("b0_hash")
        if mismatches:
            raise AuthorizationDriftError(tuple(mismatches))

    def authorize_task(self, task_id: str, request: Any) -> dict[str, Any]:
        self._ensure_no_open_history_recovery()
        if request.operator_confirmed is not True:
            raise ValueError("operator confirmation is required")
        task = self.store.require_task(task_id)
        if task.state is not LifecycleState.B0_FROZEN:
            raise TaskStoreError("task must have a frozen B0 before authorization")
        try:
            baseline = self._task_b0[task_id]
        except KeyError as error:
            raise TaskStoreError("frozen B0 object is unavailable; re-run preflight") from error
        try:
            capture = self._task_capture[task_id]
        except KeyError as error:
            raise TaskStoreError(
                "frozen Script/RegMap evidence is unavailable; re-run preflight"
            ) from error
        expected = GatewayAttestation.from_capture(capture)
        with self._hardware_access(f"v1.1:authorize:{task_id}"):
            current = self.gateway.current_attestation()
        self._assert_attestation(expected, current)
        if (
            task.task_type is TaskType.SWEEP
            and task.payload.get("sweep_hardware")
            == SweepHardware.BERT_FIR_SWEEP.value
        ):
            selections = tuple(
                ParameterSelection.from_payload(item)
                for item in task.payload.get("parameters", [])
            )
            unknown = sorted(
                item.name for item in selections if item.name not in FIR_TAP_NAMES
            )
            if unknown:
                raise ValueError(
                    "unknown FIR tap parameter(s): " + ", ".join(unknown)
                )
            authorized_parameters = tuple(
                AuthorizedParameter(
                    item.name,
                    DiscreteRange(item.minimum, item.maximum, item.step),
                )
                for item in selections
            )
        else:
            parameters = self._task_parameters(task, capture.mapping)
            authorized_parameters = tuple(
                AuthorizedParameter(item.name, item.value_range)
                for item in parameters
            )
        depth = OptimizationDepth(task.payload.get("depth", "STANDARD"))
        authorization = FullCampaignAuthorization.freeze(
            authorization_id=request.authorization_id,
            task_id=task_id,
            task_type=task.task_type,
            b0_hash=baseline.content_hash,
            hardware_identities=dict(current.hardware_identities),
            mapping_hash=current.mapping_hash,
            channel_mapping=baseline.channel_mapping_payload(),
            dca_config_fingerprint=current.dca_config_fingerprint,
            parameters=authorized_parameters,
            depth=depth,
            candidate_budget=task.payload["candidate_budget"],
            seed=task.payload.get("seed", 0),
            max_points=task.payload.get("max_points"),
            rollback_policy=RollbackPolicy.RESTORE_B0,
            issued_at=datetime.now(timezone.utc),
            expires_at=request.expires_at,
            advanced_iteration_limit=task.payload.get("advanced_iteration_limit"),
        )
        self.store.save_authorization(task_id, authorization)
        task = self.store.transition(
            task_id,
            expected_state=LifecycleState.B0_FROZEN,
            new_state=LifecycleState.AUTHORIZED,
            event_type="FULL_CAMPAIGN_AUTHORIZED",
            payload={
                "authorization_id": authorization.authorization_id,
                "authorization_hash": authorization.content_hash,
            },
        )
        self._authorizations[task_id] = authorization
        return self._task_payload(task)

    def _verify_start_attestation(self, task_id: str) -> FullCampaignAuthorization:
        try:
            authorization = self._authorizations[task_id]
        except KeyError as error:
            raise TaskStoreError("authorization object is unavailable") from error
        if datetime.now(timezone.utc) >= authorization.expires_at:
            raise TaskStoreError("FULL_CAMPAIGN authorization has expired")
        try:
            b0 = self._task_b0[task_id]
        except KeyError as error:
            raise TaskStoreError("frozen B0 object is unavailable") from error
        expected = GatewayAttestation.from_persisted_evidence(b0, authorization)
        authorization.verify_current(
            b0_hash=b0.content_hash,
            hardware_identities=expected.hardware_identities,
            mapping_hash=expected.mapping_hash,
            channel_mapping=expected.channel_mapping,
            dca_config_fingerprint=expected.dca_config_fingerprint,
        )
        current = self.gateway.current_attestation()
        self._assert_attestation(expected, current)
        authorization.verify_current(
            b0_hash=b0.content_hash,
            hardware_identities=current.hardware_identities,
            mapping_hash=current.mapping_hash,
            channel_mapping=current.channel_mapping,
            dca_config_fingerprint=current.dca_config_fingerprint,
        )
        return authorization

    def start_task(self, task_id: str) -> dict[str, Any]:
        with self._start_lock:
            return self._start_task_locked(task_id)

    def _start_task_locked(self, task_id: str) -> dict[str, Any]:
        self._ensure_no_open_history_recovery()
        external = self.external_active_task_id()
        if external is not None:
            raise ActiveTaskConflict(f"v1:{external}")
        task = self.store.require_task(task_id)
        if task.state is not LifecycleState.AUTHORIZED:
            raise TaskStoreError("only an AUTHORIZED task can start")
        self._ensure_task_lease(task_id)
        running_persisted = False
        try:
            self._verify_start_attestation(task_id)
            self.runner.start(task_id, task.task_type)
            task = self.store.transition(
                task_id,
                expected_state=LifecycleState.AUTHORIZED,
                new_state=LifecycleState.RUNNING,
                event_type="TASK_STARTED",
            )
            running_persisted = True
            run_started = getattr(self.runner, "run_started", None)
            if callable(run_started):
                run_started(task_id)
        except Exception:
            cancel_prepared = getattr(self.runner, "cancel_prepared", None)
            if callable(cancel_prepared):
                try:
                    cancel_prepared(task_id)
                except Exception:
                    # Preserve the original commit/dispatch failure.  Runner
                    # implementations must fence mismatched prepared tasks.
                    pass
            if running_persisted:
                self.store.transition(
                    task_id,
                    expected_state=LifecycleState.RUNNING,
                    new_state=LifecycleState.FAILED,
                    event_type="TASK_DISPATCH_FAILED",
                )
            self._release_task_lease(task_id)
            raise
        return self._task_payload(task)

    def pause_task(self, task_id: str) -> dict[str, Any]:
        self._ensure_no_open_history_recovery()
        task = self.store.require_task(task_id)
        if task.state is not LifecycleState.RUNNING:
            raise TaskStoreError("only a RUNNING task can pause")
        self._ensure_task_lease(task_id)
        self.runner.pause_and_restore_b0(task_id)
        task = self.store.transition(
            task_id,
            expected_state=LifecycleState.RUNNING,
            new_state=LifecycleState.PAUSED,
            event_type="TASK_PAUSED_B0_RESTORED",
        )
        return self._task_payload(task)

    def resume_task(self, task_id: str) -> dict[str, Any]:
        self._ensure_no_open_history_recovery()
        task = self.store.require_task(task_id)
        if task.state is not LifecycleState.PAUSED:
            raise TaskStoreError("only a PAUSED task can resume")
        self._ensure_task_lease(task_id)
        self._verify_start_attestation(task_id)
        task = self._commit_running_then_resume(
            task_id,
            expected_state=LifecycleState.PAUSED,
            success_event="TASK_RESUMED_AFTER_PREFLIGHT",
            failure_event="TASK_RESUME_FAILED",
        )
        return self._task_payload(task)

    def _commit_running_then_resume(
        self,
        task_id: str,
        *,
        expected_state: LifecycleState,
        success_event: str,
        failure_event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        task = self.store.transition(
            task_id,
            expected_state=expected_state,
            new_state=LifecycleState.RUNNING,
            event_type=success_event,
            payload=payload,
        )
        try:
            self.runner.resume_from_b0(task_id)
        except Exception as resume_error:
            compensation_error: Exception | None = None
            transition_error: Exception | None = None
            try:
                self.runner.recover_b0(task_id)
            except Exception as error:
                compensation_error = error
            try:
                self.store.transition(
                    task_id,
                    expected_state=LifecycleState.RUNNING,
                    new_state=LifecycleState.RECOVERY_REQUIRED,
                    event_type=failure_event,
                    payload={
                        "error": str(resume_error),
                        "compensation_error": (
                            None
                            if compensation_error is None
                            else str(compensation_error)
                        ),
                    },
                )
            except Exception as error:
                transition_error = error
            details = ["resume failed after durable RUNNING transition"]
            if compensation_error is not None:
                details.append(f"B0 compensation failed: {compensation_error}")
            if transition_error is not None:
                details.append(
                    f"RECOVERY_REQUIRED transition failed: {transition_error}"
                )
            raise TaskStoreError("; ".join(details)) from resume_error
        return task

    def abort_task(self, task_id: str, request: Any) -> dict[str, Any]:
        self._ensure_open()
        task = self.store.require_task(task_id)
        if task.state in {
            LifecycleState.DRAFT,
            LifecycleState.PREFLIGHTED,
            LifecycleState.B0_FROZEN,
            LifecycleState.AUTHORIZED,
        }:
            task = self.store.transition(
                task_id,
                expected_state=task.state,
                new_state=LifecycleState.ABORTED,
                event_type="TASK_CANCELLED_BEFORE_START",
                payload={"reason": request.reason},
            )
            self._release_task_lease(task_id)
            return self._task_payload(task)
        if task.state not in {LifecycleState.RUNNING, LifecycleState.PAUSED}:
            raise TaskStoreError("only a RUNNING or PAUSED task can abort")
        self._ensure_no_open_history_recovery()
        self._ensure_task_lease(task_id)
        task = self.store.transition(
            task_id,
            expected_state=task.state,
            new_state=LifecycleState.ABORTING,
            event_type="TASK_ABORT_REQUESTED",
            payload={"reason": request.reason},
        )
        task = self.store.transition(
            task_id,
            expected_state=LifecycleState.ABORTING,
            new_state=LifecycleState.ROLLING_BACK,
            event_type="B0_ROLLBACK_STARTED",
        )
        try:
            self.runner.abort_and_restore_b0(task_id)
        except Exception as error:
            task = self.store.transition(
                task_id,
                expected_state=LifecycleState.ROLLING_BACK,
                new_state=LifecycleState.RECOVERY_REQUIRED,
                event_type="B0_ROLLBACK_FAILED",
                payload={"error": str(error)},
            )
            return self._task_payload(task)
        task = self.store.transition(
            task_id,
            expected_state=LifecycleState.ROLLING_BACK,
            new_state=LifecycleState.ABORTED,
            event_type="TASK_ABORTED_B0_RESTORED",
        )
        self._release_task_lease(task_id)
        return self._task_payload(task)

    def recover_task(self, task_id: str, request: Any) -> dict[str, Any]:
        self._ensure_no_open_history_recovery()
        task = self.store.require_task(task_id)
        if task.state is not LifecycleState.RECOVERY_REQUIRED:
            raise TaskStoreError("task is not RECOVERY_REQUIRED")
        if request.action not in {"RESUME", "ABORT_RECOVERED"}:
            raise TaskStoreError(
                "recovery action must be RESUME or ABORT_RECOVERED"
            )
        self._ensure_task_lease(task_id)
        self.runner.recover_b0(task_id)
        if request.action == "RESUME":
            self._verify_start_attestation(task_id)
            task = self._commit_running_then_resume(
                task_id,
                expected_state=LifecycleState.RECOVERY_REQUIRED,
                success_event="RECOVERY_B0_RESTORED",
                failure_event="RECOVERY_RESUME_FAILED",
                payload={"action": request.action},
            )
            return self._task_payload(task)
        task = self.store.transition(
            task_id,
            expected_state=LifecycleState.RECOVERY_REQUIRED,
            new_state=LifecycleState.ABORTED_RECOVERED,
            event_type="RECOVERY_B0_RESTORED",
            payload={"action": request.action},
        )
        self._release_task_lease(task_id)
        return self._task_payload(task)

    def close(self) -> None:
        with self._authority_lock:
            if self._closed:
                return
            self._closed = True
        close_runner = getattr(self.runner, "close", None)
        if callable(close_runner):
            close_runner()
        with self._authority_lock:
            lease = self._task_lease
            self._task_lease = None
            history_lease = self._history_recovery_lease
            self._history_recovery_lease = None
        if lease is not None:
            lease.release()
        if history_lease is not None:
            history_lease.release()
        if self.history_apply_controller is not None:
            self.history_apply_controller.close()

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self._task_payload(self.store.require_task(task_id))

    def events(self, task_id: str, *, after_sequence: int = 0):
        self.store.require_task(task_id)
        return self.store.events(task_id, after_sequence=after_sequence)

    def current_state(self, task_id: str) -> LifecycleState | None:
        task = self.store.get_task(task_id)
        return None if task is None else task.state

    def observations(self, task_id: str) -> list[dict[str, object]]:
        return [record.payload() for record in self.store.observations(task_id)]

    def best_result(self, task_id: str) -> Any:
        record = self.store.best_state(task_id)
        return None if record is None else record.payload()

    def sweep_points(self, task_id: str, *, offset: int, limit: int) -> Any:
        page = self.store.sweep_points(task_id, offset=offset, limit=limit)
        return {
            "items": [record.payload() for record in page.items],
            "total": page.total,
            "offset": page.offset,
            "limit": page.limit,
        }

    def final_report(self, task_id: str) -> dict[str, Any]:
        task = self.store.require_task(task_id)
        return {
            **self._task_payload(task),
            "observations": self.observations(task_id),
            "events": [event.payload for event in self.store.events(task_id)],
        }

    def sweep_json_path(self, task_id: str) -> Path:
        self.store.require_task(task_id)
        return self.artifact_root / task_id / "sweep_results.json"

    def sweep_csv_path(self, task_id: str) -> Path:
        self.store.require_task(task_id)
        return self.artifact_root / task_id / "sweep_results.csv"

    def apply_history_result(self, request: Any) -> Any:
        self._ensure_no_open_history_recovery()
        if self.history_apply_controller is None:
            raise TaskStoreError("history apply controller is not configured")
        external = self.external_active_task_id()
        if external is not None:
            raise ActiveTaskConflict(f"v1:{external}")
        record = self.store.best_state(request.source_task_id)
        if record is None or record.observation_id != request.observation_id:
            raise TaskStoreError(
                "historical observation has no persisted complete hardware state"
            )
        target = HistoryResultTarget(
            source_task_id=request.source_task_id,
            observation_id=request.observation_id,
            i2c=record.i2c,
            fir=record.fir,
            upper_nlc=record.upper_nlc,
            lower_nlc=record.lower_nlc,
            pattern=record.pattern,
            bert_port=record.bert_port,
            bert_channel=record.bert_channel,
            dca_fingerprint=record.dca_fingerprint,
        )
        try:
            receipt = self.history_apply_controller.apply(target).payload()
        except HistoryApplyRollbackFailure as error:
            recovery_id = str(uuid4())
            recovery = self.store.create_recovery_record(
                recovery_id=recovery_id,
                task_id=request.source_task_id,
                reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
                payload={
                    "source_task_id": request.source_task_id,
                    "observation_id": request.observation_id,
                    "temporary_rollback": error.temporary_rollback.payload(),
                    "apply_error": str(error.apply_error),
                    "rollback_error": str(error.rollback_error),
                },
            )
            self.history_apply_controller.bind_recovery(recovery.recovery_id)
            self._history_recovery_id = recovery.recovery_id
            raise
        except HistoryApplyRolledBackFailure as error:
            self.store.append_audit_event(
                request.source_task_id,
                event_type="HISTORY_RESULT_APPLY_FAILED_ROLLED_BACK",
                payload={
                    "source_task_id": request.source_task_id,
                    "observation_id": request.observation_id,
                    "temporary_rollback": error.temporary_rollback.payload(),
                    "apply_error": str(error.apply_error),
                    "rollback_verified": True,
                },
            )
            raise error.apply_error
        self.store.append_audit_event(
            request.source_task_id,
            event_type="HISTORY_RESULT_APPLIED",
            payload=receipt,
        )
        return receipt

    def recover_history_result(self, recovery_id: str, request: Any) -> Any:
        self._ensure_open()
        if request.operator_confirmed is not True:
            raise ValueError("operator confirmation is required")
        record = self.store.require_recovery_record(recovery_id)
        if (
            record.reason_code
            is not RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED
        ):
            raise TaskStoreError("recovery record is not a History Apply recovery")
        if record.status is RecoveryStatus.RESOLVED:
            resolved = self.store.resolve_recovery_record(recovery_id)
            if self.history_apply_controller is not None and (
                self.history_apply_controller.retained_recovery_id is not None
                or self.history_apply_controller.has_unpersisted_recovery
            ):
                self.history_apply_controller.release_recovery(recovery_id)
            self._history_recovery_id = None
            return self._recovery_payload(resolved)
        if record.status is not RecoveryStatus.OPEN:
            raise TaskStoreError(f"recovery record {recovery_id} is not OPEN")
        if self.history_apply_controller is None:
            raise TaskStoreError("history apply controller is not configured")
        try:
            if record.payload["source_task_id"] != record.task_id:
                raise ValueError("source task metadata mismatch")
            observation_id = record.payload["observation_id"]
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError("observation ID is invalid")
            snapshot = HistoryHardwareState.from_payload(
                record.payload["temporary_rollback"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TaskStoreError("history recovery payload is malformed") from error

        self.history_apply_controller.restore_recovery(recovery_id, snapshot)
        resolved = self.store.resolve_recovery_record(recovery_id)
        self.history_apply_controller.release_recovery(recovery_id)
        self._history_recovery_id = None
        return self._recovery_payload(resolved)
