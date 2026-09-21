"""Deterministic V1.1 I2C/FIR sweep planning, execution, and export."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from .domain import DiscreteRange
from .hardware_control import RecoveryRequired


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

    def payload(self) -> dict[str, object]:
        return {
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


class SweepRunner:
    def __init__(self, *, executor: SweepExecutor, store: SweepPointSink) -> None:
        self._executor = executor
        self._store = store

    def _restore_or_recovery(self, context: str) -> None:
        try:
            self._executor.restore_b0()
        except Exception as error:
            raise RecoveryRequired(
                f"{context}; B0 restore failed: {error}",
                rollback_verified=False,
            ) from error

    def prepare_at_b0(self) -> None:
        """Preflight and establish verified B0 before any measurement."""

        self._executor.preflight()
        if self._executor.is_at_b0():
            return
        self._restore_or_recovery("sweep requires original B0")
        if not self._executor.is_at_b0():
            raise RecoveryRequired(
                "sweep B0 restore could not be verified",
                rollback_verified=False,
            )

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
                    self._restore_or_recovery("sweep paused")
                    return SweepRunResult(
                        state=SweepRunState.PAUSED,
                        completed_original_indexes=tuple(sorted(completed)),
                    )
                timestamp = datetime.now(timezone.utc)
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
                        completed_at=timestamp,
                    )
                else:
                    application = self._executor.apply_point(point)
                    measurement = self._executor.measure_point()
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
                            None
                            if tdecq is not None
                            else "INVALID_TDECQ"
                        ),
                        seed=plan.seed,
                        hardware=plan.hardware,
                        mode=plan.mode,
                        completed_at=timestamp,
                    )
                self._store.save_sweep_point(record)
                completed.add(point.original_index)
                if point.group_end and point.group_parameter is not None:
                    self._executor.restore_parameter(point.group_parameter)
        except Exception:
            self._restore_or_recovery("sweep point failed")
            raise
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
        "raw_response",
    ]
    for prefix in ("requested", "applied", "readback"):
        fieldnames.extend(f"{prefix}.{name}" for name in parameter_names)
    rows: list[dict[str, object]] = []
    for record in records:
        row: dict[str, object] = {
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
