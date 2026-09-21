"""Deterministic V1.1 I2C/FIR sweep planning, execution, and export."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from itertools import product
from math import isfinite
from pathlib import Path
import time
from typing import Any, Protocol

from .domain import DiscreteRange
from .hardware_control import HardwareTransactionError, RecoveryRequired


FIR_TAP_NAMES = (
    "pre3",
    "pre2",
    "pre1",
    "main",
    "post1",
    "post2",
    "post3",
)


class SweepHardware(StrEnum):
    I2C_SWEEP = "I2C_SWEEP"
    BERT_FIR_SWEEP = "BERT_FIR_SWEEP"


class SweepMode(StrEnum):
    SINGLE_PARAMETER = "SINGLE_PARAMETER"
    FULL_COMBINATION = "FULL_COMBINATION"


class SweepPointStatus(StrEnum):
    PLANNED = "PLANNED"
    COMPLETED = "COMPLETED"
    INVALID_TDECQ = "INVALID_TDECQ"
    SKIPPED_CONSTRAINT = "SKIPPED_CONSTRAINT"
    FAILED = "FAILED"


class SweepRunState(StrEnum):
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"


class SweepLimitExceeded(ValueError):
    def __init__(self, *, point_count: int, max_points: int) -> None:
        self.point_count = point_count
        self.max_points = max_points
        super().__init__(
            f"sweep contains {point_count} points, exceeding max_points "
            f"{max_points}; automatic sampling is forbidden"
        )


@dataclass(frozen=True, slots=True)
class SweepPoint:
    original_index: int
    execution_index: int
    requested: Mapping[str, int]
    status: SweepPointStatus = SweepPointStatus.PLANNED
    constraint_reason: str | None = None
    group_parameter: str | None = None
    group_end: bool = False


@dataclass(frozen=True, slots=True)
class SweepPlan:
    hardware: SweepHardware
    mode: SweepMode
    seed: int
    max_points: int
    points: tuple[SweepPoint, ...]
    settle_time_s: float = 0.0
    campaign_id: str = "unbound"
    save_dataset: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise ValueError("campaign_id must be a nonblank string")
        if type(self.save_dataset) is not bool:
            raise TypeError("save_dataset must be a boolean")
        if (
            isinstance(self.settle_time_s, bool)
            or not isinstance(self.settle_time_s, (int, float))
            or not isfinite(float(self.settle_time_s))
            or self.settle_time_s < 0
        ):
            raise ValueError("settle_time_s must be a finite nonnegative number")
        object.__setattr__(self, "settle_time_s", float(self.settle_time_s))

    @property
    def point_count(self) -> int:
        return len(self.points)


def _validated_ranges(
    ranges: Sequence[tuple[str, DiscreteRange]],
) -> tuple[tuple[str, DiscreteRange], ...]:
    if not ranges:
        raise ValueError("sweep requires at least one parameter range")
    result: list[tuple[str, DiscreteRange]] = []
    seen: set[str] = set()
    for name, value_range in ranges:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("sweep parameter names must be nonblank")
        normalized = name.strip().lower()
        if normalized in seen:
            raise ValueError(f"duplicate sweep parameter {normalized}")
        if not isinstance(value_range, DiscreteRange):
            raise TypeError("sweep ranges must contain DiscreteRange values")
        seen.add(normalized)
        result.append((normalized, value_range))
    return tuple(result)


def _validate_budget(max_points: int, count: int) -> None:
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be an integer")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if count > max_points:
        raise SweepLimitExceeded(point_count=count, max_points=max_points)


def _planned_settings(
    *,
    ranges: tuple[tuple[str, DiscreteRange], ...],
    baseline: Mapping[str, int],
    mode: SweepMode,
    seed: int,
    max_points: int,
) -> list[tuple[int, int, dict[str, int], str | None, bool]]:
    unknown = sorted(name for name, _ in ranges if name not in baseline)
    if unknown:
        raise ValueError("sweep parameter missing from B0: " + ", ".join(unknown))

    natural: list[tuple[int, dict[str, int], str | None, bool]] = []
    if mode is SweepMode.SINGLE_PARAMETER:
        original_index = 0
        for name, value_range in ranges:
            values = value_range.values()
            for offset, value in enumerate(values):
                requested = dict(baseline)
                requested[name] = value
                natural.append(
                    (
                        original_index,
                        requested,
                        name,
                        offset == len(values) - 1,
                    )
                )
                original_index += 1
        _validate_budget(max_points, len(natural))
        return [
            (original, execution, requested, group, group_end)
            for execution, (original, requested, group, group_end) in enumerate(natural)
        ]

    if mode is not SweepMode.FULL_COMBINATION:
        raise ValueError(f"unsupported sweep mode {mode!r}")
    names = tuple(name for name, _ in ranges)
    grids = tuple(value_range.values() for _, value_range in ranges)
    for original_index, values in enumerate(product(*grids)):
        requested = dict(baseline)
        requested.update(zip(names, values, strict=True))
        natural.append((original_index, requested, None, False))
    _validate_budget(max_points, len(natural))
    shuffled = list(natural)
    random.Random(seed).shuffle(shuffled)
    return [
        (original, execution, requested, group, group_end)
        for execution, (original, requested, group, group_end) in enumerate(shuffled)
    ]


def create_i2c_sweep(
    *,
    ranges: Sequence[tuple[str, DiscreteRange]],
    b0: Mapping[str, int],
    mode: SweepMode,
    seed: int,
    max_points: int,
    settle_time_s: float = 0.0,
    campaign_id: str = "unbound",
    save_dataset: bool = False,
) -> SweepPlan:
    normalized = _validated_ranges(ranges)
    points = tuple(
        SweepPoint(
            original_index=original,
            execution_index=execution,
            requested=requested,
            group_parameter=group,
            group_end=group_end,
        )
        for original, execution, requested, group, group_end in _planned_settings(
            ranges=normalized,
            baseline=b0,
            mode=mode,
            seed=seed,
            max_points=max_points,
        )
    )
    return SweepPlan(
        hardware=SweepHardware.I2C_SWEEP,
        mode=mode,
        seed=seed,
        max_points=max_points,
        points=points,
        settle_time_s=settle_time_s,
        campaign_id=campaign_id,
        save_dataset=save_dataset,
    )


def _fir_constraint(taps: tuple[int, ...]) -> str | None:
    reasons: list[str] = []
    if taps[3] <= 0:
        reasons.append("FIR Main must be positive")
    l1 = sum(abs(value) for value in taps)
    if l1 > 170:
        reasons.append(f"FIR L1 {l1} exceeds 170")
    return "; ".join(reasons) or None


def create_bert_fir_sweep(
    *,
    ranges: Sequence[tuple[str, DiscreteRange]],
    b0: tuple[int, ...],
    mode: SweepMode,
    seed: int,
    max_points: int,
    settle_time_s: float = 0.0,
    campaign_id: str = "unbound",
    save_dataset: bool = False,
) -> SweepPlan:
    if len(b0) != 7 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in b0
    ):
        raise ValueError("B0 FIR must contain seven integers")
    normalized = _validated_ranges(ranges)
    unknown = sorted(name for name, _ in normalized if name not in FIR_TAP_NAMES)
    if unknown:
        raise ValueError("unknown FIR tap parameter(s): " + ", ".join(unknown))
    baseline = dict(zip(FIR_TAP_NAMES, b0, strict=True))
    points: list[SweepPoint] = []
    for original, execution, requested, group, group_end in _planned_settings(
        ranges=normalized,
        baseline=baseline,
        mode=mode,
        seed=seed,
        max_points=max_points,
    ):
        taps = tuple(requested[name] for name in FIR_TAP_NAMES)
        reason = _fir_constraint(taps)
        points.append(
            SweepPoint(
                original_index=original,
                execution_index=execution,
                requested=requested,
                status=(
                    SweepPointStatus.SKIPPED_CONSTRAINT
                    if reason
                    else SweepPointStatus.PLANNED
                ),
                constraint_reason=reason,
                group_parameter=group,
                group_end=group_end,
            )
        )
    return SweepPlan(
        hardware=SweepHardware.BERT_FIR_SWEEP,
        mode=mode,
        seed=seed,
        max_points=max_points,
        points=tuple(points),
        settle_time_s=settle_time_s,
        campaign_id=campaign_id,
        save_dataset=save_dataset,
    )


@dataclass(frozen=True, slots=True)
class SweepPointRecord:
    original_index: int
    execution_index: int
    status: SweepPointStatus
    requested: Mapping[str, int]
    applied: Mapping[str, int]
    readback: Mapping[str, int]
    tdecq: float | None
    er: float | None
    raw_response: Any
    error_reason: str | None
    seed: int
    hardware: SweepHardware
    mode: SweepMode
    completed_at: datetime
    write_readback_duration_s: float | None = None
    settle_duration_s: float | None = None
    dca_duration_s: float | None = None
    result_persistence_duration_s: float | None = None
    restore_duration_s: float | None = None
    candidate_round_time_s: float | None = None
    screening_round_with_restore_s: float | None = None
    restore_readback_result: Mapping[str, int] | None = None
    restore_status: str = "NOT_ATTEMPTED"
    evidence_persistence_error: str | None = None
    measurement_id: str | None = None
    acquisition_id: str | None = None
    measurement_status: str = "FAILED"
    dataset_status: str = "DISABLED"
    dataset_error: str | None = None
    dataset_bundle_path: str | None = None

    @property
    def point_id(self) -> str:
        return f"point-{self.original_index:08d}"

    def payload(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "original_index": self.original_index,
            "execution_index": self.execution_index,
            "status": self.status.value,
            "requested": dict(self.requested),
            "applied": dict(self.applied),
            "readback": dict(self.readback),
            "tdecq": self.tdecq,
            "er": self.er,
            "raw_response": self.raw_response,
            "error_reason": self.error_reason,
            "seed": self.seed,
            "hardware": self.hardware.value,
            "mode": self.mode.value,
            "completed_at": self.completed_at.isoformat(),
            "write_readback_duration_s": self.write_readback_duration_s,
            "settle_duration_s": self.settle_duration_s,
            "dca_duration_s": self.dca_duration_s,
            "result_persistence_duration_s": self.result_persistence_duration_s,
            "restore_duration_s": self.restore_duration_s,
            "candidate_round_time_s": self.candidate_round_time_s,
            "screening_round_with_restore_s": self.screening_round_with_restore_s,
            "restore_readback_result": (
                None
                if self.restore_readback_result is None
                else dict(self.restore_readback_result)
            ),
            "restore_status": self.restore_status,
            "evidence_persistence_error": self.evidence_persistence_error,
            "measurement_id": self.measurement_id,
            "acquisition_id": self.acquisition_id,
            "measurement_status": self.measurement_status,
            "dataset_status": self.dataset_status,
            "dataset_error": self.dataset_error,
            "dataset_bundle_path": self.dataset_bundle_path,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SweepPointRecord":
        return cls(
            original_index=payload["original_index"],
            execution_index=payload["execution_index"],
            status=SweepPointStatus(payload["status"]),
            requested=dict(payload["requested"]),
            applied=dict(payload["applied"]),
            readback=dict(payload["readback"]),
            tdecq=payload["tdecq"],
            er=payload["er"],
            raw_response=payload["raw_response"],
            error_reason=payload["error_reason"],
            seed=payload["seed"],
            hardware=SweepHardware(payload["hardware"]),
            mode=SweepMode(payload["mode"]),
            completed_at=datetime.fromisoformat(payload["completed_at"]),
            write_readback_duration_s=payload.get("write_readback_duration_s"),
            settle_duration_s=payload.get("settle_duration_s"),
            dca_duration_s=payload.get("dca_duration_s"),
            result_persistence_duration_s=payload.get(
                "result_persistence_duration_s"
            ),
            restore_duration_s=payload.get("restore_duration_s"),
            candidate_round_time_s=payload.get("candidate_round_time_s"),
            screening_round_with_restore_s=payload.get(
                "screening_round_with_restore_s"
            ),
            restore_readback_result=(
                None
                if payload.get("restore_readback_result") is None
                else dict(payload["restore_readback_result"])
            ),
            restore_status=str(payload.get("restore_status", "NOT_ATTEMPTED")),
            evidence_persistence_error=payload.get(
                "evidence_persistence_error"
            ),
            measurement_id=payload.get("measurement_id"),
            acquisition_id=payload.get("acquisition_id"),
            measurement_status=str(
                payload.get(
                    "measurement_status",
                    (
                        "SUCCESS"
                        if payload["status"] == SweepPointStatus.COMPLETED.value
                        else (
                            "REJECTED"
                            if payload["status"]
                            == SweepPointStatus.SKIPPED_CONSTRAINT.value
                            else "FAILED"
                        )
                    ),
                )
            ),
            dataset_status=str(payload.get("dataset_status", "DISABLED")),
            dataset_error=payload.get("dataset_error"),
            dataset_bundle_path=payload.get("dataset_bundle_path"),
        )


class PointEvidencePersistenceError(HardwareTransactionError):
    """A point INSERT succeeded but its complete evidence UPDATE failed."""

    def __init__(
        self,
        message: str,
        *,
        record: SweepPointRecord,
        persistence_error: Exception | str,
        rollback_verified: bool,
        hardware_error: Exception | None = None,
    ) -> None:
        self.record = record
        self.persistence_error = str(persistence_error)
        self.hardware_error = hardware_error
        super().__init__(
            message,
            rollback_verified=rollback_verified,
            evidence=record.payload(),
        )


class PointEvidenceRecoveryRequired(RecoveryRequired):
    """Point evidence failed after hardware recovery was already unsuccessful."""

    def __init__(
        self,
        message: str,
        *,
        record: SweepPointRecord,
        persistence_error: Exception | str,
        hardware_error: Exception,
    ) -> None:
        self.record = record
        self.persistence_error = str(persistence_error)
        self.hardware_error = hardware_error
        super().__init__(
            message,
            rollback_verified=False,
            evidence=record.payload(),
        )


class SweepExecutor(Protocol):
    def preflight(self) -> None: ...

    def is_at_b0(self) -> bool: ...

    def restore_b0(self) -> None: ...

    def restore_parameter(self, name: str) -> None: ...

    def apply_point(self, point: SweepPoint) -> Mapping[str, object]: ...

    def measure_point(self) -> Mapping[str, object]: ...


class SweepPointSink(Protocol):
    def save_sweep_point(self, record: SweepPointRecord) -> None: ...

    def update_sweep_point(self, record: SweepPointRecord) -> None: ...


@dataclass(frozen=True, slots=True)
class SweepRunResult:
    state: SweepRunState
    completed_original_indexes: tuple[int, ...]


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not isfinite(result) or abs(result) >= 9.0e37:
        return None
    return result


def best_sweep_point(
    records: Sequence[SweepPointRecord],
) -> SweepPointRecord | None:
    """Select the best valid point by TDECQ, then by Outer ER.

    TDECQ is the primary objective (lower is better).  Outer ER is the
    deterministic secondary objective (higher is better), and execution order
    breaks an otherwise exact tie.
    """

    candidates = [
        record
        for record in records
        if record.status is SweepPointStatus.COMPLETED
        and _finite_metric(record.tdecq) is not None
    ]
    if not candidates:
        return None

    def rank(record: SweepPointRecord) -> tuple[float, float, int]:
        tdecq = _finite_metric(record.tdecq)
        assert tdecq is not None
        er = _finite_metric(record.er)
        return (
            tdecq,
            -(er if er is not None else float("-inf")),
            record.execution_index,
        )

    return min(candidates, key=rank)


class SweepRunner:
    def __init__(
        self,
        *,
        executor: SweepExecutor,
        store: SweepPointSink,
        monotonic: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._executor = executor
        self._store = store
        self._monotonic = monotonic
        self._now_utc = now_utc

    @staticmethod
    def _duration(
        payload: Mapping[str, object],
        field: str,
        fallback: float,
    ) -> float:
        value = payload.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            and float(value) >= 0
        ):
            return float(value)
        return max(0.0, fallback)

    @staticmethod
    def _combined_persistence_error(
        existing: str | None,
        current: Exception | str,
    ) -> str:
        return "; ".join(
            item for item in (existing, str(current)) if item
        )

    @staticmethod
    def _readback_from_payload(
        payload: object,
    ) -> Mapping[str, int] | None:
        if not isinstance(payload, Mapping):
            return None
        for field in (
            "normalized_readback",
            "last_observed_readback",
            "partial_readback",
            "final_b0_readback",
        ):
            readback = payload.get(field)
            if not isinstance(readback, Mapping):
                continue
            try:
                return {
                    str(name): int(value) for name, value in readback.items()
                }
            except (TypeError, ValueError):
                continue
        return None

    def _restore_readback(
        self,
        error: Exception | None = None,
    ) -> Mapping[str, int] | None:
        if error is not None:
            readback = self._readback_from_payload(
                getattr(error, "evidence", None)
            )
            if readback is not None:
                return readback
        evidence = getattr(self._executor, "audit_evidence", None)
        if not callable(evidence):
            return None
        payload = evidence()
        return self._readback_from_payload(payload)

    def _restore_or_recovery(self, context: str) -> None:
        try:
            self._executor.restore_b0()
        except Exception as error:
            evidence = getattr(error, "evidence", None)
            raise RecoveryRequired(
                f"{context}; B0 restore failed: {error}",
                rollback_verified=False,
                evidence=(
                    dict(evidence)
                    if isinstance(evidence, Mapping)
                    else None
                ),
            ) from error
        if not self._executor.is_at_b0():
            raise RecoveryRequired(
                f"{context}; B0 restore could not be verified",
                rollback_verified=False,
            )

    def _persist_candidate(
        self,
        record: SweepPointRecord,
        *,
        round_started: float,
        hardware_error: Exception | None = None,
    ) -> SweepPointRecord:
        persistence_started = self._monotonic()
        self._store.save_sweep_point(record)
        persistence_duration_s = max(
            0.0, self._monotonic() - persistence_started
        )
        persisted = replace(
            record,
            result_persistence_duration_s=persistence_duration_s,
            candidate_round_time_s=max(
                0.0, self._monotonic() - round_started
            ),
        )
        try:
            self._store.update_sweep_point(persisted)
        except Exception as persistence_error:
            failed = replace(
                persisted,
                evidence_persistence_error=str(persistence_error),
            )
            raise PointEvidencePersistenceError(
                "point timing evidence could not be persisted",
                record=failed,
                persistence_error=persistence_error,
                rollback_verified=False,
                hardware_error=hardware_error,
            ) from persistence_error
        return persisted

    def _restore_point(
        self,
        record: SweepPointRecord,
        *,
        round_started: float,
        context: str,
    ) -> SweepPointRecord:
        restore_started = self._monotonic()
        try:
            self._restore_or_recovery(context)
        except Exception as error:
            failed = replace(
                record,
                restore_duration_s=max(
                    0.0, self._monotonic() - restore_started
                ),
                screening_round_with_restore_s=max(
                    0.0, self._monotonic() - round_started
                ),
                restore_readback_result=self._restore_readback(error),
                restore_status="RECOVERY_REQUIRED",
                completed_at=self._now_utc(),
            )
            if isinstance(error, RecoveryRequired):
                recovery = error
            else:
                recovery = RecoveryRequired(
                    f"{context}; B0 restore failed: {error}",
                    rollback_verified=False,
                )
            recovery.evidence = failed.payload()
            try:
                self._store.update_sweep_point(failed)
            except Exception as persistence_error:
                failed = replace(
                    failed,
                    evidence_persistence_error=(
                        self._combined_persistence_error(
                            failed.evidence_persistence_error,
                            persistence_error,
                        )
                    ),
                )
                recovery.evidence = failed.payload()
            raise recovery from error
        restored = replace(
            record,
            restore_duration_s=max(
                0.0, self._monotonic() - restore_started
            ),
            screening_round_with_restore_s=max(
                0.0, self._monotonic() - round_started
            ),
            restore_readback_result=self._restore_readback(),
            restore_status="VERIFIED",
            completed_at=self._now_utc(),
        )
        try:
            self._store.update_sweep_point(restored)
        except Exception as persistence_error:
            combined_error = self._combined_persistence_error(
                record.evidence_persistence_error,
                persistence_error,
            )
            failed = replace(
                restored,
                evidence_persistence_error=combined_error,
            )
            failure = PointEvidencePersistenceError(
                "B0 restore was verified but its point evidence could not be "
                f"persisted: {persistence_error}",
                record=failed,
                persistence_error=combined_error,
                rollback_verified=True,
            )
            raise failure from persistence_error
        return restored

    def _verify_point_restored(
        self,
        record: SweepPointRecord,
        *,
        round_started: float,
        context: str,
    ) -> SweepPointRecord:
        """Verify an executor-owned rollback without issuing another restore."""

        verification_started = self._monotonic()
        try:
            verified = self._executor.is_at_b0()
        except Exception as error:
            verified = False
            verification_error: Exception | None = error
        else:
            verification_error = None
        duration_s = max(0.0, self._monotonic() - verification_started)
        if not verified:
            failed = replace(
                record,
                restore_duration_s=duration_s,
                screening_round_with_restore_s=max(
                    0.0, self._monotonic() - round_started
                ),
                restore_readback_result=self._restore_readback(),
                restore_status="RECOVERY_REQUIRED",
                completed_at=self._now_utc(),
            )
            recovery = RecoveryRequired(
                f"{context}; executor rollback could not be verified"
                + (
                    ""
                    if verification_error is None
                    else f": {verification_error}"
                ),
                rollback_verified=False,
            )
            recovery.evidence = failed.payload()
            try:
                self._store.update_sweep_point(failed)
            except Exception as persistence_error:
                failed = replace(
                    failed,
                    evidence_persistence_error=(
                        self._combined_persistence_error(
                            failed.evidence_persistence_error,
                            persistence_error,
                        )
                    ),
                )
                recovery.evidence = failed.payload()
            raise recovery from verification_error
        restored = replace(
            record,
            restore_duration_s=duration_s,
            screening_round_with_restore_s=max(
                0.0, self._monotonic() - round_started
            ),
            restore_readback_result=self._restore_readback(),
            restore_status="VERIFIED_INTERNAL_ROLLBACK",
            completed_at=self._now_utc(),
        )
        try:
            self._store.update_sweep_point(restored)
        except Exception as persistence_error:
            combined_error = self._combined_persistence_error(
                record.evidence_persistence_error,
                persistence_error,
            )
            failed = replace(
                restored,
                evidence_persistence_error=combined_error,
            )
            failure = PointEvidencePersistenceError(
                "executor rollback was verified but its point evidence could "
                f"not be persisted: {persistence_error}",
                record=failed,
                persistence_error=combined_error,
                rollback_verified=True,
            )
            raise failure from persistence_error
        return restored

    def _record_failure(
        self,
        *,
        point: SweepPoint,
        plan: SweepPlan,
        application: Mapping[str, object],
        error: Exception,
        write_readback_duration_s: float,
        settle_duration_s: float,
        dca_duration_s: float,
    ) -> SweepPointRecord:
        evidence = getattr(error, "evidence", None)
        return SweepPointRecord(
            original_index=point.original_index,
            execution_index=point.execution_index,
            status=SweepPointStatus.FAILED,
            requested=dict(application.get("requested", point.requested)),
            applied=dict(application.get("applied", {})),
            readback=dict(application.get("readback", {})),
            tdecq=None,
            er=None,
            raw_response=(
                dict(evidence)
                if isinstance(evidence, Mapping)
                else {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "rollback_verified": getattr(
                        error, "rollback_verified", None
                    ),
                }
            ),
            error_reason=str(error),
            seed=plan.seed,
            hardware=plan.hardware,
            mode=plan.mode,
            completed_at=self._now_utc(),
            write_readback_duration_s=write_readback_duration_s,
            settle_duration_s=settle_duration_s,
            dca_duration_s=dca_duration_s,
            restore_status="PENDING",
            measurement_status="FAILED",
        )

    def prepare_at_b0(self) -> None:
        """Preflight and establish verified B0 before any measurement."""

        self._executor.preflight()
        if self._executor.is_at_b0():
            return
        self._restore_or_recovery("sweep requires original B0")

    def run(
        self,
        plan: SweepPlan,
        *,
        completed_original_indexes: Sequence[int] = (),
        pause_requested: Callable[[], bool] = lambda: False,
        preflight_done: bool = False,
    ) -> SweepRunResult:
        completed = set(completed_original_indexes)
        if not preflight_done:
            self.prepare_at_b0()
        try:
            for point in plan.points:
                if point.original_index in completed:
                    continue
                if pause_requested():
                    if not (
                        getattr(self._executor, "restore_after_each_point", False)
                        and self._executor.is_at_b0()
                    ):
                        self._restore_or_recovery("sweep paused")
                    return SweepRunResult(
                        state=SweepRunState.PAUSED,
                        completed_original_indexes=tuple(sorted(completed)),
                    )
                if point.status is SweepPointStatus.SKIPPED_CONSTRAINT:
                    record = SweepPointRecord(
                        original_index=point.original_index,
                        execution_index=point.execution_index,
                        status=SweepPointStatus.SKIPPED_CONSTRAINT,
                        requested=dict(point.requested),
                        applied={},
                        readback={},
                        tdecq=None,
                        er=None,
                        raw_response=None,
                        error_reason=point.constraint_reason,
                        seed=plan.seed,
                        hardware=plan.hardware,
                        mode=plan.mode,
                        completed_at=self._now_utc(),
                        restore_status="NOT_REQUIRED",
                        measurement_status="REJECTED",
                    )
                    self._store.save_sweep_point(record)
                    completed.add(point.original_index)
                    continue
                else:
                    application: Mapping[str, object] = {}
                    round_started = self._monotonic()
                    apply_started = round_started
                    try:
                        application = self._executor.apply_point(point)
                        apply_elapsed_s = max(
                            0.0, self._monotonic() - apply_started
                        )
                        write_readback_duration_s = self._duration(
                            application,
                            "write_readback_duration_s",
                            apply_elapsed_s,
                        )
                        measurement_started = self._monotonic()
                        measurement = self._executor.measure_point()
                        measurement_elapsed_s = max(
                            0.0, self._monotonic() - measurement_started
                        )
                        settle_duration_s = self._duration(
                            measurement,
                            "settle_duration_s",
                            0.0,
                        )
                        dca_duration_s = self._duration(
                            measurement,
                            "dca_duration_s",
                            max(0.0, measurement_elapsed_s - settle_duration_s),
                        )
                        tdecq = _finite_metric(measurement.get("tdecq"))
                        er = _finite_metric(measurement.get("er"))
                        status = (
                            SweepPointStatus.COMPLETED
                            if tdecq is not None
                            else SweepPointStatus.INVALID_TDECQ
                        )
                        record = SweepPointRecord(
                            original_index=point.original_index,
                            execution_index=point.execution_index,
                            status=status,
                            requested=dict(
                                application.get("requested", point.requested)
                            ),
                            applied=dict(application.get("applied", {})),
                            readback=dict(application.get("readback", {})),
                            tdecq=tdecq,
                            er=er,
                            raw_response=measurement.get("raw"),
                            error_reason=(
                                None if tdecq is not None else "INVALID_TDECQ"
                            ),
                            seed=plan.seed,
                            hardware=plan.hardware,
                            mode=plan.mode,
                            completed_at=self._now_utc(),
                            write_readback_duration_s=write_readback_duration_s,
                            settle_duration_s=settle_duration_s,
                            dca_duration_s=dca_duration_s,
                            restore_status="PENDING",
                            measurement_id=measurement.get("measurement_id"),
                            acquisition_id=measurement.get("acquisition_id"),
                            measurement_status=str(
                                measurement.get(
                                    "measurement_status",
                                    "SUCCESS" if tdecq is not None else "FAILED",
                                )
                            ),
                            dataset_status=str(
                                measurement.get("dataset_status", "DISABLED")
                            ),
                            dataset_error=measurement.get("dataset_error"),
                            dataset_bundle_path=measurement.get(
                                "dataset_bundle_path"
                            ),
                        )
                    except Exception as error:
                        apply_elapsed_s = max(
                            0.0, self._monotonic() - apply_started
                        )
                        evidence = getattr(error, "evidence", None)
                        evidence_payload = (
                            evidence if isinstance(evidence, Mapping) else {}
                        )
                        write_readback_duration_s = self._duration(
                            application,
                            "write_readback_duration_s",
                            apply_elapsed_s,
                        )
                        settle_duration_s = self._duration(
                            evidence_payload,
                            "settle_duration_s",
                            0.0,
                        )
                        dca_duration_s = self._duration(
                            evidence_payload,
                            "dca_duration_s",
                            0.0,
                        )
                        record = self._record_failure(
                            point=point,
                            plan=plan,
                            application=application,
                            error=error,
                            write_readback_duration_s=write_readback_duration_s,
                            settle_duration_s=settle_duration_s,
                            dca_duration_s=dca_duration_s,
                        )
                        record = self._persist_candidate(
                            record,
                            round_started=round_started,
                            hardware_error=error,
                        )
                        if isinstance(error, RecoveryRequired):
                            failed = replace(
                                record,
                                restore_duration_s=None,
                                screening_round_with_restore_s=max(
                                    0.0, self._monotonic() - round_started
                                ),
                                restore_readback_result=(
                                    self._restore_readback(error)
                                ),
                                restore_status="RECOVERY_REQUIRED",
                                completed_at=self._now_utc(),
                            )
                            error.evidence = failed.payload()
                            try:
                                self._store.update_sweep_point(failed)
                            except Exception as persistence_error:
                                failed = replace(
                                    failed,
                                    evidence_persistence_error=(
                                        self._combined_persistence_error(
                                            failed.evidence_persistence_error,
                                            persistence_error,
                                        )
                                    ),
                                )
                                error.evidence = failed.payload()
                            raise
                        if bool(getattr(error, "rollback_verified", False)):
                            restored_record = self._verify_point_restored(
                                record,
                                round_started=round_started,
                                context="sweep point failed",
                            )
                        else:
                            restored_record = self._restore_point(
                                record,
                                round_started=round_started,
                                context="sweep point failed",
                            )
                        restored_error = HardwareTransactionError(
                            f"sweep point failed; B0 restore verified: {error}",
                            rollback_verified=True,
                        )
                        restored_error.evidence = restored_record.payload()
                        raise restored_error from error
                record = self._persist_candidate(
                    record,
                    round_started=round_started,
                )
                completed.add(point.original_index)
                if (
                    point.status is not SweepPointStatus.SKIPPED_CONSTRAINT
                    and getattr(self._executor, "restore_after_each_point", False)
                ):
                    self._restore_point(
                        record,
                        round_started=round_started,
                        context="sweep point completed",
                    )
                elif point.group_end and point.group_parameter is not None:
                    self._executor.restore_parameter(point.group_parameter)
                    self._store.update_sweep_point(
                        replace(
                            record,
                            restore_duration_s=0.0,
                            screening_round_with_restore_s=(
                                record.candidate_round_time_s
                            ),
                            restore_status="NOT_REQUIRED",
                        )
                    )
                else:
                    self._store.update_sweep_point(
                        replace(
                            record,
                            restore_duration_s=0.0,
                            screening_round_with_restore_s=(
                                record.candidate_round_time_s
                            ),
                            restore_status="NOT_REQUIRED",
                        )
                    )
        except RecoveryRequired:
            raise
        except PointEvidencePersistenceError as error:
            if error.rollback_verified:
                raise
            hardware_error = error.hardware_error
            if isinstance(hardware_error, RecoveryRequired):
                failed = replace(
                    error.record,
                    screening_round_with_restore_s=max(
                        0.0, self._monotonic() - round_started
                    ),
                    restore_readback_result=self._restore_readback(
                        hardware_error
                    ),
                    restore_status="RECOVERY_REQUIRED",
                    completed_at=self._now_utc(),
                )
                try:
                    self._store.update_sweep_point(failed)
                except Exception as persistence_error:
                    failed = replace(
                        failed,
                        evidence_persistence_error=(
                            self._combined_persistence_error(
                                failed.evidence_persistence_error,
                                persistence_error,
                            )
                        ),
                    )
                raise PointEvidenceRecoveryRequired(
                    "point timing evidence failed after hardware recovery "
                    "had already failed; no second restore was attempted",
                    record=failed,
                    persistence_error=(
                        failed.evidence_persistence_error
                        or error.persistence_error
                    ),
                    hardware_error=hardware_error,
                ) from error
            if bool(getattr(hardware_error, "rollback_verified", False)):
                restored_record = self._verify_point_restored(
                    error.record,
                    round_started=round_started,
                    context="point evidence persistence failed after "
                    "executor rollback",
                )
                raise PointEvidencePersistenceError(
                    "point timing evidence initially failed after an "
                    "executor-owned rollback; B0 verification and "
                    "best-effort point update completed",
                    record=restored_record,
                    persistence_error=error.persistence_error,
                    rollback_verified=True,
                    hardware_error=hardware_error,
                ) from error
            restored_record = self._restore_point(
                error.record,
                round_started=round_started,
                context="point evidence persistence failed",
            )
            raise PointEvidencePersistenceError(
                "point timing evidence initially failed but B0 restore and "
                "best-effort point update completed",
                record=restored_record,
                persistence_error=error.persistence_error,
                rollback_verified=True,
                hardware_error=hardware_error,
            ) from error
        except HardwareTransactionError as error:
            if error.rollback_verified:
                raise
            self._restore_or_recovery("sweep point failed")
            raise HardwareTransactionError(
                f"sweep point failed; B0 restore verified: {error}",
                rollback_verified=True,
            ) from error
        except Exception as error:
            self._restore_or_recovery("sweep point failed")
            raise HardwareTransactionError(
                f"sweep point failed; B0 restore verified: {error}",
                rollback_verified=True,
            ) from error
        if getattr(self._executor, "restore_after_each_point", False):
            if not self._executor.is_at_b0():
                raise RecoveryRequired(
                    "sweep final B0 verification drifted after the per-point "
                    "restore",
                    rollback_verified=False,
                )
        else:
            self._restore_or_recovery("sweep completed")
        return SweepRunResult(
            state=SweepRunState.COMPLETED,
            completed_original_indexes=tuple(sorted(completed)),
        )


@dataclass(frozen=True, slots=True)
class SweepArtifactPaths:
    json_path: Path
    csv_path: Path


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def export_sweep_artifacts(
    *,
    directory: str | Path,
    task_id: str,
    plan: SweepPlan,
    records: Sequence[SweepPointRecord],
    product_version: str,
    schema_version: int,
    runtime_mode: str,
    b0_hash: str,
    configuration_fingerprint: str,
    run_evidence: Mapping[str, object] | None = None,
) -> SweepArtifactPaths:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    metadata = {
        "product_version": product_version,
        "schema_version": schema_version,
        "campaign_type": "SWEEP",
        "runtime_mode": runtime_mode,
        "task_id": task_id,
        "b0_hash": b0_hash,
        "configuration_fingerprint": configuration_fingerprint,
        "hardware": plan.hardware.value,
        "mode": plan.mode.value,
        "seed": plan.seed,
        "max_points": plan.max_points,
        "points": [record.payload() for record in records],
        "best_point": (
            None
            if (best := best_sweep_point(records)) is None
            else best.payload()
        ),
        "run_evidence": None if run_evidence is None else dict(run_evidence),
    }
    json_path = target / "sweep_results.json"
    _write_atomic(
        json_path,
        json.dumps(
            metadata,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )

    parameter_names = sorted(
        {
            name
            for record in records
            for values in (record.requested, record.applied, record.readback)
            for name in values
        }
    )
    fieldnames = [
        "point_id",
        "original_index",
        "execution_index",
        "status",
        "tdecq",
        "er",
        "error_reason",
        "seed",
        "hardware",
        "mode",
        "completed_at",
        "write_readback_duration_s",
        "settle_duration_s",
        "dca_duration_s",
        "result_persistence_duration_s",
        "restore_duration_s",
        "candidate_round_time_s",
        "screening_round_with_restore_s",
        "restore_readback_result",
        "restore_status",
        "evidence_persistence_error",
        "measurement_id",
        "acquisition_id",
        "measurement_status",
        "dataset_status",
        "dataset_error",
        "dataset_bundle_path",
        "raw_response",
    ]
    for prefix in ("requested", "applied", "readback"):
        fieldnames.extend(f"{prefix}.{name}" for name in parameter_names)
    rows: list[dict[str, object]] = []
    for record in records:
        row: dict[str, object] = {
            "point_id": record.point_id,
            "original_index": record.original_index,
            "execution_index": record.execution_index,
            "status": record.status.value,
            "tdecq": "" if record.tdecq is None else record.tdecq,
            "er": "" if record.er is None else record.er,
            "error_reason": record.error_reason or "",
            "seed": record.seed,
            "hardware": record.hardware.value,
            "mode": record.mode.value,
            "completed_at": record.completed_at.isoformat(),
            "write_readback_duration_s": (
                ""
                if record.write_readback_duration_s is None
                else record.write_readback_duration_s
            ),
            "settle_duration_s": (
                "" if record.settle_duration_s is None else record.settle_duration_s
            ),
            "dca_duration_s": (
                "" if record.dca_duration_s is None else record.dca_duration_s
            ),
            "result_persistence_duration_s": (
                ""
                if record.result_persistence_duration_s is None
                else record.result_persistence_duration_s
            ),
            "restore_duration_s": (
                ""
                if record.restore_duration_s is None
                else record.restore_duration_s
            ),
            "candidate_round_time_s": (
                ""
                if record.candidate_round_time_s is None
                else record.candidate_round_time_s
            ),
            "screening_round_with_restore_s": (
                ""
                if record.screening_round_with_restore_s is None
                else record.screening_round_with_restore_s
            ),
            "restore_readback_result": json.dumps(
                record.restore_readback_result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "restore_status": record.restore_status,
            "evidence_persistence_error": (
                ""
                if record.evidence_persistence_error is None
                else record.evidence_persistence_error
            ),
            "measurement_id": record.measurement_id or "",
            "acquisition_id": record.acquisition_id or "",
            "measurement_status": record.measurement_status,
            "dataset_status": record.dataset_status,
            "dataset_error": record.dataset_error or "",
            "dataset_bundle_path": record.dataset_bundle_path or "",
            "raw_response": json.dumps(
                record.raw_response,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for prefix, values in (
            ("requested", record.requested),
            ("applied", record.applied),
            ("readback", record.readback),
        ):
            for name in parameter_names:
                row[f"{prefix}.{name}"] = values.get(name, "")
        rows.append(row)
    csv_path = target / "sweep_results.csv"
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return SweepArtifactPaths(json_path=json_path, csv_path=csv_path)
