"""Transactional V1.1 task state, audit events, and global hardware lock."""

from __future__ import annotations

import json
from functools import wraps
from math import isfinite
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Self

from .constants import PRODUCT_VERSION, SCHEMA_VERSION
from .authorization import FullCampaignAuthorization
from .baseline import B0Snapshot
from .domain import LifecycleState, RuntimeMode, TaskType
from .migrations import initialize_database
from .sweep import SweepPointRecord, SweepPointStatus


class TaskStoreError(RuntimeError):
    pass


class ActiveTaskConflict(TaskStoreError):
    def __init__(self, active_task_id: str) -> None:
        self.active_task_id = active_task_id
        super().__init__(f"hardware task {active_task_id} is already active")


class TaskConcurrencyError(TaskStoreError):
    pass


class UnknownTaskError(TaskStoreError):
    pass


class UnknownRecoveryError(TaskStoreError):
    pass


class RecoveryStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class RecoveryReasonCode(str, Enum):
    HISTORY_APPLY_ROLLBACK_FAILED = "HISTORY_APPLY_ROLLBACK_FAILED"


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    task_type: TaskType
    state: LifecycleState
    runtime_mode: RuntimeMode
    product_version: str
    schema_version: int
    b0_hash: str | None
    config_fingerprint: str
    payload: Any
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TaskEvent:
    task_id: str
    sequence: int
    event_type: str
    from_state: LifecycleState | None
    to_state: LifecycleState
    payload: Any
    product_version: str
    schema_version: int
    campaign_type: TaskType
    runtime_mode: RuntimeMode
    b0_hash: str | None
    config_fingerprint: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    recovery_id: str
    task_id: str
    reason_code: RecoveryReasonCode
    status: RecoveryStatus
    payload: Any
    product_version: str
    schema_version: int
    campaign_type: TaskType
    runtime_mode: RuntimeMode
    b0_hash: str | None
    config_fingerprint: str
    created_at: datetime
    resolved_at: datetime | None


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    observation_id: str
    sequence: int
    phase: str
    status: str
    valid: bool
    requested: dict[str, int]
    applied: dict[str, int]
    readback: dict[str, int]
    tdecq: float | None
    er: float | None
    raw_response: Any
    error_reason: str | None
    created_at: datetime

    @staticmethod
    def _metric(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        if not isfinite(result) or abs(result) >= 9.0e37:
            return None
        return result

    @classmethod
    def measurement(
        cls,
        *,
        observation_id: str,
        sequence: int,
        phase: str,
        requested: dict[str, int],
        applied: dict[str, int],
        readback: dict[str, int],
        tdecq: object,
        er: object,
        raw_response: Any,
        created_at: datetime,
        error_reason: str | None = None,
    ) -> "ObservationRecord":
        normalized_tdecq = cls._metric(tdecq)
        normalized_er = cls._metric(er)
        valid = normalized_tdecq is not None
        return cls(
            observation_id=observation_id,
            sequence=sequence,
            phase=phase,
            status="COMPLETED" if valid else "INVALID_TDECQ",
            valid=valid,
            requested=dict(requested),
            applied=dict(applied),
            readback=dict(readback),
            tdecq=normalized_tdecq,
            er=normalized_er,
            raw_response=raw_response,
            error_reason=(
                error_reason
                if error_reason is not None
                else (None if valid else "INVALID_TDECQ")
            ),
            created_at=_timestamp(created_at),
        )

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "sequence": self.sequence,
            "phase": self.phase,
            "status": self.status,
            "valid": self.valid,
            "requested": dict(self.requested),
            "applied": dict(self.applied),
            "readback": dict(self.readback),
            "tdecq": self.tdecq,
            "er": self.er,
            "raw_response": self.raw_response,
            "error_reason": self.error_reason,
            "created_at": _encode_time(self.created_at),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObservationRecord":
        return cls(
            observation_id=payload["observation_id"],
            sequence=payload["sequence"],
            phase=payload["phase"],
            status=payload["status"],
            valid=payload["valid"],
            requested=dict(payload["requested"]),
            applied=dict(payload["applied"]),
            readback=dict(payload["readback"]),
            tdecq=payload["tdecq"],
            er=payload["er"],
            raw_response=payload["raw_response"],
            error_reason=payload["error_reason"],
            created_at=_decode_time(payload["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class BestStateRecord:
    observation_id: str
    i2c: dict[str, int]
    fir: tuple[int, ...]
    upper_nlc: int
    lower_nlc: int
    pattern: str
    bert_port: str
    bert_channel: str
    dca_fingerprint: str
    result: Any
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "i2c", dict(self.i2c))
        object.__setattr__(self, "fir", tuple(self.fir))
        if not self.observation_id:
            raise ValueError("observation_id must be nonblank")
        if len(self.fir) != 7 or any(type(value) is not int for value in self.fir):
            raise ValueError("best FIR must contain seven integers")
        if self.fir[3] <= 0 or sum(abs(value) for value in self.fir) > 170:
            raise ValueError("best FIR violates Main/L1 constraints")
        if self.pattern != "SSPRQ":
            raise ValueError("best state pattern must be SSPRQ")
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at))

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "i2c": dict(self.i2c),
            "fir": list(self.fir),
            "upper_nlc": self.upper_nlc,
            "lower_nlc": self.lower_nlc,
            "pattern": self.pattern,
            "bert_port": self.bert_port,
            "bert_channel": self.bert_channel,
            "dca_fingerprint": self.dca_fingerprint,
            "result": self.result,
            "updated_at": _encode_time(self.updated_at),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BestStateRecord":
        return cls(
            observation_id=payload["observation_id"],
            i2c=dict(payload["i2c"]),
            fir=tuple(payload["fir"]),
            upper_nlc=payload["upper_nlc"],
            lower_nlc=payload["lower_nlc"],
            pattern=payload["pattern"],
            bert_port=payload["bert_port"],
            bert_channel=payload["bert_channel"],
            dca_fingerprint=payload["dca_fingerprint"],
            result=payload["result"],
            updated_at=_decode_time(payload["updated_at"]),
        )


@dataclass(frozen=True, slots=True)
class SweepPointPage:
    items: tuple[SweepPointRecord, ...]
    total: int
    offset: int
    limit: int


class _TaskSweepPointSink:
    def __init__(self, store: "TaskStore", task_id: str) -> None:
        self._store = store
        self._task_id = task_id

    def save_sweep_point(self, record: SweepPointRecord) -> None:
        self._store.save_sweep_point(self._task_id, record)


def _strict_json(value: Any) -> tuple[str, Any]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        restored = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload must be strict finite JSON: {exc}") from exc
    return encoded, restored


def _timestamp(value: datetime | None) -> datetime:
    result = datetime.now(timezone.utc) if value is None else value
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return result.astimezone(timezone.utc)


def _encode_time(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _decode_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise TaskStoreError("persisted timestamp must be an ISO datetime string")
    try:
        decoded = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TaskStoreError(
            "persisted timestamp must be an ISO datetime string"
        ) from error
    if decoded.tzinfo is None or decoded.utcoffset() is None:
        raise TaskStoreError("persisted timestamp is not timezone-aware")
    return decoded.astimezone(timezone.utc)


def _require_hash(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hash") from exc


def _serialize_task_store_methods(cls: type[Any]) -> type[Any]:
    """Serialize every public operation on the shared SQLite connection."""

    for name, method in tuple(vars(cls).items()):
        if name == "__init__" or not callable(method):
            continue
        if name.startswith("_") and name not in {"__enter__", "__exit__"}:
            continue

        @wraps(method)
        def locked(self, *args, __method=method, **kwargs):
            with self._lock:
                return __method(self, *args, **kwargs)

        setattr(cls, name, locked)
    return cls


@_serialize_task_store_methods
class TaskStore:
    def __init__(self, path: str | Path) -> None:
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = initialize_database(path)
        self._connection.row_factory = sqlite3.Row

    def __enter__(self) -> Self:
        self._require_connection()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        if connection is not None:
            self._connection = None
            connection.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise TaskStoreError("task store is closed")
        return self._connection

    def create_task(
        self,
        *,
        task_id: str,
        task_type: TaskType,
        runtime_mode: RuntimeMode,
        config_fingerprint: str,
        payload: Any,
        created_at: datetime | None = None,
    ) -> TaskRecord:
        if not task_id:
            raise ValueError("task_id must be nonblank")
        _require_hash("config_fingerprint", config_fingerprint)
        payload_json, restored_payload = _strict_json(payload)
        timestamp = _timestamp(created_at)
        encoded_time = _encode_time(timestamp)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            active = connection.execute(
                "SELECT task_id FROM active_hardware_task WHERE singleton = 1"
            ).fetchone()
            if active is not None:
                raise ActiveTaskConflict(active["task_id"])
            connection.execute(
                "INSERT INTO tasks("
                "task_id, task_type, state, runtime_mode, product_version, "
                "schema_version, b0_hash, config_fingerprint, payload_json, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                (
                    task_id,
                    task_type.value,
                    LifecycleState.DRAFT.value,
                    runtime_mode.value,
                    PRODUCT_VERSION,
                    SCHEMA_VERSION,
                    config_fingerprint,
                    payload_json,
                    encoded_time,
                    encoded_time,
                ),
            )
            connection.execute(
                "INSERT INTO active_hardware_task(singleton, task_id, acquired_at) "
                "VALUES (1, ?, ?)",
                (task_id, encoded_time),
            )
            self._insert_event(
                connection,
                task_id=task_id,
                sequence=1,
                event_type="TASK_CREATED",
                from_state=None,
                to_state=LifecycleState.DRAFT,
                payload_json=payload_json,
                task_type=task_type,
                runtime_mode=runtime_mode,
                b0_hash=None,
                config_fingerprint=config_fingerprint,
                created_at=encoded_time,
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return TaskRecord(
            task_id=task_id,
            task_type=task_type,
            state=LifecycleState.DRAFT,
            runtime_mode=runtime_mode,
            product_version=PRODUCT_VERSION,
            schema_version=SCHEMA_VERSION,
            b0_hash=None,
            config_fingerprint=config_fingerprint,
            payload=restored_payload,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def active_task_id(self) -> str | None:
        row = self._require_connection().execute(
            "SELECT task_id FROM active_hardware_task WHERE singleton = 1"
        ).fetchone()
        return None if row is None else row["task_id"]

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self._require_connection().execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        return None if row is None else self._task_from_row(row)

    def require_task(self, task_id: str) -> TaskRecord:
        task = self.get_task(task_id)
        if task is None:
            raise UnknownTaskError(f"unknown task {task_id}")
        return task

    def events(self, task_id: str, *, after_sequence: int = 0) -> list[TaskEvent]:
        rows = self._require_connection().execute(
            "SELECT * FROM task_events WHERE task_id = ? AND sequence > ? "
            "ORDER BY sequence",
            (task_id, after_sequence),
        ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def append_audit_event(
        self,
        task_id: str,
        *,
        event_type: str,
        payload: Any,
        occurred_at: datetime | None = None,
    ) -> TaskEvent:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be nonblank")
        payload_json, _ = _strict_json(payload)
        timestamp = _timestamp(occurred_at)
        encoded_time = _encode_time(timestamp)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            state = LifecycleState(row["state"])
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
            self._insert_event(
                connection,
                task_id=task_id,
                sequence=sequence,
                event_type=event_type,
                from_state=state,
                to_state=state,
                payload_json=payload_json,
                task_type=TaskType(row["task_type"]),
                runtime_mode=RuntimeMode(row["runtime_mode"]),
                b0_hash=row["b0_hash"],
                config_fingerprint=row["config_fingerprint"],
                created_at=encoded_time,
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return self.events(task_id, after_sequence=sequence - 1)[0]

    def create_recovery_record(
        self,
        *,
        recovery_id: str,
        task_id: str,
        reason_code: RecoveryReasonCode,
        payload: Any,
        created_at: datetime | None = None,
    ) -> RecoveryRecord:
        if not isinstance(recovery_id, str) or not recovery_id.strip():
            raise ValueError("recovery_id must be nonblank")
        if not isinstance(reason_code, RecoveryReasonCode):
            raise TypeError("reason_code must be a RecoveryReasonCode")
        payload_json, restored_payload = _strict_json(payload)
        if not isinstance(restored_payload, dict):
            raise ValueError("recovery payload must be a JSON object")
        if {"recovery_id", "reason_code"} & set(restored_payload):
            raise ValueError("recovery payload contains reserved audit fields")
        timestamp = _timestamp(created_at)
        encoded_time = _encode_time(timestamp)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        authoritative_id = recovery_id
        try:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            same_id = connection.execute(
                "SELECT * FROM recovery_records WHERE recovery_id = ?",
                (recovery_id,),
            ).fetchone()
            if same_id is not None and (
                same_id["task_id"] != task_id
                or same_id["reason_code"] != reason_code.value
                or same_id["payload_json"] != payload_json
            ):
                raise TaskStoreError("conflicting recovery ID or payload")
            existing_open = connection.execute(
                "SELECT * FROM recovery_records WHERE task_id = ? AND "
                "reason_code = ? AND status = ?",
                (task_id, reason_code.value, RecoveryStatus.OPEN.value),
            ).fetchone()
            existing = same_id if same_id is not None else existing_open
            if existing is not None:
                if (
                    existing["task_id"] != task_id
                    or existing["reason_code"] != reason_code.value
                    or existing["payload_json"] != payload_json
                ):
                    raise TaskStoreError("conflicting recovery ID or payload")
                authoritative_id = existing["recovery_id"]
                expected_audit = {
                    **restored_payload,
                    "recovery_id": authoritative_id,
                    "reason_code": reason_code.value,
                }
                matching_events = []
                for event_row in connection.execute(
                    "SELECT payload_json FROM task_events WHERE task_id = ? "
                    "AND event_type = ? ORDER BY sequence",
                    (task_id, "HISTORY_APPLY_ROLLBACK_FAILED"),
                ).fetchall():
                    if json.loads(event_row["payload_json"]) == expected_audit:
                        matching_events.append(event_row)
                if len(matching_events) != 1:
                    raise TaskStoreError(
                        "conflicting recovery audit event state"
                    )
            else:
                connection.execute(
                    "INSERT INTO recovery_records("
                    "recovery_id, task_id, reason_code, status, payload_json, "
                    "product_version, schema_version, campaign_type, runtime_mode, "
                    "b0_hash, config_fingerprint, created_at, resolved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        recovery_id,
                        task_id,
                        reason_code.value,
                        RecoveryStatus.OPEN.value,
                        payload_json,
                        PRODUCT_VERSION,
                        SCHEMA_VERSION,
                        task["task_type"],
                        task["runtime_mode"],
                        task["b0_hash"],
                        task["config_fingerprint"],
                        encoded_time,
                    ),
                )
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()[0]
                state = LifecycleState(task["state"])
                audit_json, _ = _strict_json(
                    {
                        **restored_payload,
                        "recovery_id": recovery_id,
                        "reason_code": reason_code.value,
                    }
                )
                self._insert_event(
                    connection,
                    task_id=task_id,
                    sequence=sequence,
                    event_type="HISTORY_APPLY_ROLLBACK_FAILED",
                    from_state=state,
                    to_state=state,
                    payload_json=audit_json,
                    task_type=TaskType(task["task_type"]),
                    runtime_mode=RuntimeMode(task["runtime_mode"]),
                    b0_hash=task["b0_hash"],
                    config_fingerprint=task["config_fingerprint"],
                    created_at=encoded_time,
                )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return self.require_recovery_record(authoritative_id)

    def get_recovery_record(self, recovery_id: str) -> RecoveryRecord | None:
        row = self._require_connection().execute(
            "SELECT * FROM recovery_records WHERE recovery_id = ?", (recovery_id,)
        ).fetchone()
        return None if row is None else self._recovery_from_row(row)

    def require_recovery_record(self, recovery_id: str) -> RecoveryRecord:
        record = self.get_recovery_record(recovery_id)
        if record is None:
            raise UnknownRecoveryError(f"unknown recovery record {recovery_id}")
        return record

    def list_recovery_records(
        self, *, status: RecoveryStatus | str | None = None
    ) -> tuple[RecoveryRecord, ...]:
        if status is None:
            rows = self._require_connection().execute(
                "SELECT * FROM recovery_records ORDER BY created_at, recovery_id"
            ).fetchall()
        else:
            try:
                normalized = RecoveryStatus(status)
            except (TypeError, ValueError) as error:
                raise ValueError("unknown recovery status") from error
            rows = self._require_connection().execute(
                "SELECT * FROM recovery_records WHERE status = ? "
                "ORDER BY created_at, recovery_id",
                (normalized.value,),
            ).fetchall()
        return tuple(self._recovery_from_row(row) for row in rows)

    def resolve_recovery_record(
        self, recovery_id: str, *, resolved_at: datetime | None = None
    ) -> RecoveryRecord:
        timestamp = _timestamp(resolved_at)
        encoded_time = _encode_time(timestamp)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM recovery_records WHERE recovery_id = ?",
                (recovery_id,),
            ).fetchone()
            if row is None:
                raise UnknownRecoveryError(f"unknown recovery record {recovery_id}")
            record = self._recovery_from_row(row)
            if (
                record.reason_code
                is not RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED
                or not isinstance(record.payload, dict)
                or record.payload.get("source_task_id") != record.task_id
                or not isinstance(record.payload.get("observation_id"), str)
                or not isinstance(record.payload.get("temporary_rollback"), dict)
            ):
                raise TaskStoreError("history recovery payload is malformed")
            recovered_payload = {
                "recovery_id": recovery_id,
                "reason_code": record.reason_code.value,
                "source_task_id": record.task_id,
                "observation_id": record.payload["observation_id"],
                "restored_state": record.payload["temporary_rollback"],
            }
            if record.status is RecoveryStatus.RESOLVED:
                matches = 0
                for event_row in connection.execute(
                    "SELECT payload_json FROM task_events WHERE task_id = ? "
                    "AND event_type = ?",
                    (record.task_id, "HISTORY_APPLY_RECOVERED"),
                ).fetchall():
                    if json.loads(event_row["payload_json"]) == recovered_payload:
                        matches += 1
                if matches != 1:
                    raise TaskStoreError(
                        "conflicting recovered audit event state"
                    )
            elif record.status is RecoveryStatus.OPEN:
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = ?", (record.task_id,)
                ).fetchone()
                if task is None:
                    raise TaskStoreError("persisted recovery owning task is missing")
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events "
                    "WHERE task_id = ?",
                    (record.task_id,),
                ).fetchone()[0]
                audit_json, _ = _strict_json(recovered_payload)
                state = LifecycleState(task["state"])
                self._insert_event(
                    connection,
                    task_id=record.task_id,
                    sequence=sequence,
                    event_type="HISTORY_APPLY_RECOVERED",
                    from_state=state,
                    to_state=state,
                    payload_json=audit_json,
                    task_type=TaskType(task["task_type"]),
                    runtime_mode=RuntimeMode(task["runtime_mode"]),
                    b0_hash=task["b0_hash"],
                    config_fingerprint=task["config_fingerprint"],
                    created_at=encoded_time,
                )
                connection.execute(
                    "UPDATE recovery_records SET status = ?, resolved_at = ? "
                    "WHERE recovery_id = ?",
                    (RecoveryStatus.RESOLVED.value, encoded_time, recovery_id),
                )
            else:
                raise TaskStoreError("unknown recovery status")
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return self.require_recovery_record(recovery_id)

    def save_b0(self, task_id: str, snapshot: B0Snapshot) -> None:
        if not isinstance(snapshot, B0Snapshot):
            raise TypeError("snapshot must be a B0Snapshot")
        payload_json, _ = _strict_json(snapshot.payload())
        frozen_at = _encode_time(snapshot.captured_at.astimezone(timezone.utc))
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            connection.execute(
                "INSERT INTO b0_snapshots("
                "task_id, version, b0_hash, payload_json, product_version, "
                "schema_version, campaign_type, runtime_mode, "
                "config_fingerprint, frozen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    snapshot.version,
                    snapshot.content_hash,
                    payload_json,
                    PRODUCT_VERSION,
                    SCHEMA_VERSION,
                    task["task_type"],
                    task["runtime_mode"],
                    task["config_fingerprint"],
                    frozen_at,
                ),
            )
            connection.execute(
                "UPDATE tasks SET b0_hash = ?, updated_at = ? WHERE task_id = ?",
                (snapshot.content_hash, frozen_at, task_id),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    def b0_payload(self, task_id: str) -> dict[str, Any]:
        self.require_task(task_id)
        row = self._require_connection().execute(
            "SELECT * FROM b0_snapshots WHERE task_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if row is None:
            raise TaskStoreError(f"task {task_id} has no frozen B0")
        return {
            "version": row["version"],
            "product_version": row["product_version"],
            "schema_version": row["schema_version"],
            "campaign_type": row["campaign_type"],
            "runtime_mode": row["runtime_mode"],
            "b0_hash": row["b0_hash"],
            "configuration_fingerprint": row["config_fingerprint"],
            "payload": json.loads(row["payload_json"]),
            "frozen_at": row["frozen_at"],
        }

    def load_b0(self, task_id: str) -> B0Snapshot:
        try:
            wrapper = self.b0_payload(task_id)
            snapshot = B0Snapshot.from_payload(wrapper["payload"])
            task = self.require_task(task_id)
            frozen_at = _decode_time(wrapper["frozen_at"])
        except TaskStoreError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise TaskStoreError("persisted B0 payload is malformed") from error
        if snapshot.content_hash != wrapper["b0_hash"]:
            raise TaskStoreError("persisted B0 content hash mismatch")
        if snapshot.version != wrapper["version"]:
            raise TaskStoreError("persisted B0 version metadata mismatch")
        if task.b0_hash != snapshot.content_hash:
            raise TaskStoreError("persisted task B0 hash mismatch")
        if frozen_at != snapshot.captured_at.astimezone(timezone.utc):
            raise TaskStoreError("persisted B0 timestamp metadata mismatch")
        if (
            wrapper["product_version"] != PRODUCT_VERSION
            or wrapper["schema_version"] != SCHEMA_VERSION
            or wrapper["campaign_type"] != task.task_type.value
            or wrapper["runtime_mode"] != task.runtime_mode.value
            or wrapper["configuration_fingerprint"] != task.config_fingerprint
        ):
            raise TaskStoreError("persisted B0 wrapper metadata mismatch")
        return snapshot

    def save_authorization(
        self,
        task_id: str,
        authorization: FullCampaignAuthorization,
    ) -> None:
        if not isinstance(authorization, FullCampaignAuthorization):
            raise TypeError("authorization must be FullCampaignAuthorization")
        if authorization.task_id != task_id:
            raise ValueError("authorization task_id does not match target task")
        payload_json, _ = _strict_json(authorization.payload())
        issued_at = _encode_time(authorization.issued_at.astimezone(timezone.utc))
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            if task["b0_hash"] != authorization.b0_hash:
                raise TaskStoreError("authorization B0 hash differs from frozen task B0")
            if task["task_type"] != authorization.task_type.value:
                raise TaskStoreError("authorization task type mismatch")
            connection.execute(
                "INSERT INTO task_authorizations("
                "authorization_id, task_id, content_hash, b0_hash, payload_json, "
                "product_version, schema_version, campaign_type, runtime_mode, "
                "config_fingerprint, issued_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    authorization.authorization_id,
                    task_id,
                    authorization.content_hash,
                    authorization.b0_hash,
                    payload_json,
                    PRODUCT_VERSION,
                    SCHEMA_VERSION,
                    task["task_type"],
                    task["runtime_mode"],
                    task["config_fingerprint"],
                    issued_at,
                ),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    def authorization_payload(self, task_id: str) -> dict[str, Any]:
        self.require_task(task_id)
        row = self._require_connection().execute(
            "SELECT * FROM task_authorizations WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise TaskStoreError(f"task {task_id} has no authorization")
        return {
            "authorization_id": row["authorization_id"],
            "content_hash": row["content_hash"],
            "b0_hash": row["b0_hash"],
            "product_version": row["product_version"],
            "schema_version": row["schema_version"],
            "campaign_type": row["campaign_type"],
            "runtime_mode": row["runtime_mode"],
            "configuration_fingerprint": row["config_fingerprint"],
            "payload": json.loads(row["payload_json"]),
            "issued_at": row["issued_at"],
        }

    def load_authorization(self, task_id: str) -> FullCampaignAuthorization:
        try:
            wrapper = self.authorization_payload(task_id)
            authorization = FullCampaignAuthorization.from_payload(wrapper["payload"])
            task = self.require_task(task_id)
            issued_at = _decode_time(wrapper["issued_at"])
        except TaskStoreError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise TaskStoreError("persisted authorization payload is malformed") from error
        if authorization.content_hash != wrapper["content_hash"]:
            raise TaskStoreError("persisted authorization content hash mismatch")
        if authorization.authorization_id != wrapper["authorization_id"]:
            raise TaskStoreError("persisted authorization identifier mismatch")
        if authorization.task_id != task_id:
            raise TaskStoreError("persisted authorization task identifier mismatch")
        if authorization.b0_hash != wrapper["b0_hash"]:
            raise TaskStoreError("persisted authorization B0 hash mismatch")
        if task.b0_hash != authorization.b0_hash:
            raise TaskStoreError("persisted task authorization B0 hash mismatch")
        if issued_at != authorization.issued_at.astimezone(timezone.utc):
            raise TaskStoreError("persisted authorization timestamp metadata mismatch")
        if (
            wrapper["product_version"] != PRODUCT_VERSION
            or wrapper["schema_version"] != SCHEMA_VERSION
            or wrapper["campaign_type"] != task.task_type.value
            or wrapper["runtime_mode"] != task.runtime_mode.value
            or wrapper["configuration_fingerprint"] != task.config_fingerprint
            or authorization.task_type is not task.task_type
        ):
            raise TaskStoreError("persisted authorization wrapper metadata mismatch")
        return authorization

    def append_observation(
        self, task_id: str, record: ObservationRecord
    ) -> None:
        if not isinstance(record, ObservationRecord):
            raise TypeError("record must be an ObservationRecord")
        payload_json, _ = _strict_json(record.payload())
        created_at = _encode_time(record.created_at)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            connection.execute(
                "INSERT INTO observations("
                "observation_id, task_id, sequence, phase, status, valid, "
                "payload_json, product_version, schema_version, campaign_type, "
                "runtime_mode, b0_hash, config_fingerprint, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.observation_id,
                    task_id,
                    record.sequence,
                    record.phase,
                    record.status,
                    int(record.valid),
                    payload_json,
                    PRODUCT_VERSION,
                    SCHEMA_VERSION,
                    task["task_type"],
                    task["runtime_mode"],
                    task["b0_hash"],
                    task["config_fingerprint"],
                    created_at,
                ),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    def observations(self, task_id: str) -> tuple[ObservationRecord, ...]:
        self.require_task(task_id)
        rows = self._require_connection().execute(
            "SELECT payload_json, status, valid FROM observations "
            "WHERE task_id = ? ORDER BY sequence",
            (task_id,),
        ).fetchall()
        result: list[ObservationRecord] = []
        for row in rows:
            record = ObservationRecord.from_payload(json.loads(row["payload_json"]))
            if record.status != row["status"] or record.valid != bool(row["valid"]):
                raise TaskStoreError("observation metadata mismatch")
            result.append(record)
        return tuple(result)

    def save_best_state(self, task_id: str, record: BestStateRecord) -> None:
        if not isinstance(record, BestStateRecord):
            raise TypeError("record must be a BestStateRecord")
        payload_json, _ = _strict_json(record.payload())
        updated_at = _encode_time(record.updated_at)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            observation = connection.execute(
                "SELECT 1 FROM observations WHERE task_id = ? AND observation_id = ?",
                (task_id, record.observation_id),
            ).fetchone()
            if observation is None:
                raise TaskStoreError(
                    "best state must reference a persisted task observation"
                )
            connection.execute(
                "INSERT INTO best_states("
                "task_id, observation_id, payload_json, product_version, "
                "schema_version, campaign_type, runtime_mode, b0_hash, "
                "config_fingerprint, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET "
                "observation_id=excluded.observation_id, "
                "payload_json=excluded.payload_json, "
                "product_version=excluded.product_version, "
                "schema_version=excluded.schema_version, "
                "campaign_type=excluded.campaign_type, "
                "runtime_mode=excluded.runtime_mode, "
                "b0_hash=excluded.b0_hash, "
                "config_fingerprint=excluded.config_fingerprint, "
                "updated_at=excluded.updated_at",
                (
                    task_id,
                    record.observation_id,
                    payload_json,
                    PRODUCT_VERSION,
                    SCHEMA_VERSION,
                    task["task_type"],
                    task["runtime_mode"],
                    task["b0_hash"],
                    task["config_fingerprint"],
                    updated_at,
                ),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    def best_state(self, task_id: str) -> BestStateRecord | None:
        self.require_task(task_id)
        row = self._require_connection().execute(
            "SELECT payload_json FROM best_states WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return BestStateRecord.from_payload(json.loads(row["payload_json"]))

    def mark_interrupted_tasks_recovery_required(self) -> tuple[str, ...]:
        interrupted_states = (
            LifecycleState.RUNNING,
            LifecycleState.PAUSED,
            LifecycleState.ABORTING,
            LifecycleState.ROLLING_BACK,
        )
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        changed: list[str] = []
        try:
            placeholders = ",".join("?" for _ in interrupted_states)
            rows = connection.execute(
                f"SELECT * FROM tasks WHERE state IN ({placeholders}) "
                "ORDER BY task_id",
                tuple(state.value for state in interrupted_states),
            ).fetchall()
            timestamp = _encode_time(_timestamp(None))
            for row in rows:
                task_id = row["task_id"]
                old_state = LifecycleState(row["state"])
                connection.execute(
                    "UPDATE tasks SET state = ?, updated_at = ? WHERE task_id = ?",
                    (LifecycleState.RECOVERY_REQUIRED.value, timestamp, task_id),
                )
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()[0]
                payload_json, _ = _strict_json(
                    {"reason": "unfinished task detected at process startup"}
                )
                self._insert_event(
                    connection,
                    task_id=task_id,
                    sequence=sequence,
                    event_type="PROCESS_RESTART_DETECTED",
                    from_state=old_state,
                    to_state=LifecycleState.RECOVERY_REQUIRED,
                    payload_json=payload_json,
                    task_type=TaskType(row["task_type"]),
                    runtime_mode=RuntimeMode(row["runtime_mode"]),
                    b0_hash=row["b0_hash"],
                    config_fingerprint=row["config_fingerprint"],
                    created_at=timestamp,
                )
                changed.append(task_id)
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return tuple(changed)

    def sweep_point_sink(self, task_id: str) -> _TaskSweepPointSink:
        task = self.require_task(task_id)
        if task.task_type is not TaskType.SWEEP:
            raise TaskStoreError(f"task {task_id} is not a SWEEP task")
        return _TaskSweepPointSink(self, task_id)

    def save_sweep_point(
        self, task_id: str, record: SweepPointRecord
    ) -> None:
        if not isinstance(record, SweepPointRecord):
            raise TypeError("record must be a SweepPointRecord")
        payload_json, _ = _strict_json(record.payload())
        completed_at = _encode_time(_timestamp(record.completed_at))
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            if TaskType(task["task_type"]) is not TaskType.SWEEP:
                raise TaskStoreError(f"task {task_id} is not a SWEEP task")
            connection.execute(
                "INSERT INTO sweep_points("
                "task_id, original_index, execution_index, status, payload_json, "
                "product_version, schema_version, campaign_type, runtime_mode, "
                "b0_hash, config_fingerprint, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    record.original_index,
                    record.execution_index,
                    record.status.value,
                    payload_json,
                    PRODUCT_VERSION,
                    SCHEMA_VERSION,
                    TaskType.SWEEP.value,
                    task["runtime_mode"],
                    task["b0_hash"],
                    task["config_fingerprint"],
                    completed_at,
                ),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    def sweep_points(
        self,
        task_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> SweepPointPage:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        self.require_task(task_id)
        connection = self._require_connection()
        total = connection.execute(
            "SELECT COUNT(*) FROM sweep_points WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT payload_json, status FROM sweep_points "
            "WHERE task_id = ? ORDER BY execution_index LIMIT ? OFFSET ?",
            (task_id, limit, offset),
        ).fetchall()
        items: list[SweepPointRecord] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            record = SweepPointRecord.from_payload(payload)
            if record.status is not SweepPointStatus(row["status"]):
                raise TaskStoreError("sweep point status metadata mismatch")
            items.append(record)
        return SweepPointPage(
            items=tuple(items), total=total, offset=offset, limit=limit
        )

    def transition(
        self,
        task_id: str,
        *,
        expected_state: LifecycleState,
        new_state: LifecycleState,
        event_type: str,
        payload: Any = None,
        occurred_at: datetime | None = None,
    ) -> TaskRecord:
        if not event_type:
            raise ValueError("event_type must be nonblank")
        payload_json, _ = _strict_json({} if payload is None else payload)
        timestamp = _timestamp(occurred_at)
        encoded_time = _encode_time(timestamp)
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise UnknownTaskError(f"unknown task {task_id}")
            current_state = LifecycleState(row["state"])
            if current_state is not expected_state:
                raise TaskConcurrencyError(
                    f"expected {expected_state.value}, found {current_state.value}"
                )
            active = connection.execute(
                "SELECT task_id FROM active_hardware_task WHERE singleton = 1"
            ).fetchone()
            if new_state.is_nonterminal:
                if active is None:
                    connection.execute(
                        "INSERT INTO active_hardware_task(singleton, task_id, acquired_at) "
                        "VALUES (1, ?, ?)",
                        (task_id, encoded_time),
                    )
                elif active["task_id"] != task_id:
                    raise ActiveTaskConflict(active["task_id"])
            connection.execute(
                "UPDATE tasks SET state = ?, updated_at = ? WHERE task_id = ?",
                (new_state.value, encoded_time, task_id),
            )
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
            self._insert_event(
                connection,
                task_id=task_id,
                sequence=sequence,
                event_type=event_type,
                from_state=current_state,
                to_state=new_state,
                payload_json=payload_json,
                task_type=TaskType(row["task_type"]),
                runtime_mode=RuntimeMode(row["runtime_mode"]),
                b0_hash=row["b0_hash"],
                config_fingerprint=row["config_fingerprint"],
                created_at=encoded_time,
            )
            if not new_state.is_nonterminal:
                connection.execute(
                    "DELETE FROM active_hardware_task WHERE singleton = 1 AND task_id = ?",
                    (task_id,),
                )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return self.require_task(task_id)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        task_id: str,
        sequence: int,
        event_type: str,
        from_state: LifecycleState | None,
        to_state: LifecycleState,
        payload_json: str,
        task_type: TaskType,
        runtime_mode: RuntimeMode,
        b0_hash: str | None,
        config_fingerprint: str,
        created_at: str,
    ) -> None:
        connection.execute(
            "INSERT INTO task_events("
            "task_id, sequence, event_type, from_state, to_state, payload_json, "
            "product_version, schema_version, campaign_type, runtime_mode, "
            "b0_hash, config_fingerprint, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                sequence,
                event_type,
                None if from_state is None else from_state.value,
                to_state.value,
                payload_json,
                PRODUCT_VERSION,
                SCHEMA_VERSION,
                task_type.value,
                runtime_mode.value,
                b0_hash,
                config_fingerprint,
                created_at,
            ),
        )

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            task_type=TaskType(row["task_type"]),
            state=LifecycleState(row["state"]),
            runtime_mode=RuntimeMode(row["runtime_mode"]),
            product_version=row["product_version"],
            schema_version=row["schema_version"],
            b0_hash=row["b0_hash"],
            config_fingerprint=row["config_fingerprint"],
            payload=json.loads(row["payload_json"]),
            created_at=_decode_time(row["created_at"]),
            updated_at=_decode_time(row["updated_at"]),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> TaskEvent:
        return TaskEvent(
            task_id=row["task_id"],
            sequence=row["sequence"],
            event_type=row["event_type"],
            from_state=(
                None
                if row["from_state"] is None
                else LifecycleState(row["from_state"])
            ),
            to_state=LifecycleState(row["to_state"]),
            payload=json.loads(row["payload_json"]),
            product_version=row["product_version"],
            schema_version=row["schema_version"],
            campaign_type=TaskType(row["campaign_type"]),
            runtime_mode=RuntimeMode(row["runtime_mode"]),
            b0_hash=row["b0_hash"],
            config_fingerprint=row["config_fingerprint"],
            created_at=_decode_time(row["created_at"]),
        )

    def _recovery_from_row(self, row: sqlite3.Row) -> RecoveryRecord:
        task = self._require_connection().execute(
            "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
        ).fetchone()
        if task is None:
            raise TaskStoreError("persisted recovery owning task is missing")
        if (
            row["product_version"] != PRODUCT_VERSION
            or row["schema_version"] != SCHEMA_VERSION
            or row["product_version"] != task["product_version"]
            or row["schema_version"] != task["schema_version"]
            or row["campaign_type"] != task["task_type"]
            or row["runtime_mode"] != task["runtime_mode"]
            or row["b0_hash"] != task["b0_hash"]
            or row["config_fingerprint"] != task["config_fingerprint"]
        ):
            raise TaskStoreError("persisted recovery wrapper metadata mismatch")
        try:
            return RecoveryRecord(
                recovery_id=row["recovery_id"],
                task_id=row["task_id"],
                reason_code=RecoveryReasonCode(row["reason_code"]),
                status=RecoveryStatus(row["status"]),
                payload=json.loads(row["payload_json"]),
                product_version=row["product_version"],
                schema_version=row["schema_version"],
                campaign_type=TaskType(row["campaign_type"]),
                runtime_mode=RuntimeMode(row["runtime_mode"]),
                b0_hash=row["b0_hash"],
                config_fingerprint=row["config_fingerprint"],
                created_at=_decode_time(row["created_at"]),
                resolved_at=(
                    None
                    if row["resolved_at"] is None
                    else _decode_time(row["resolved_at"])
                ),
            )
        except (TypeError, ValueError) as error:
            raise TaskStoreError("persisted recovery record is malformed") from error
