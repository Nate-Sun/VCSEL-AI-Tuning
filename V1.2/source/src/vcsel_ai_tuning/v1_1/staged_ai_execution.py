"""Durable PILOT_3 safety gate followed by the frozen FAST24 campaign."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Protocol

from .domain import LifecycleState
from .repository import ObservationRecord, TaskStore


FAST24_PHASES = (
    ("I2C_COARSE",) * 6
    + ("I2C_FINE",) * 16
    + ("FIR_FIRST", "FIR_ITERATIVE")
)
_NON_CANDIDATE_PHASES = frozenset({"B0", "FINAL_VERIFICATION"})


class _Execution(Protocol):
    def run(self, stop_requested: Event) -> LifecycleState | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PilotSafetyGateEvidence:
    """Detached evidence required before any FAST24 execution is constructed."""

    valid_observation_count: int
    expected_observation_count: int
    every_restore_verified: bool
    final_b0_verified: bool
    dataset_complete: bool
    identities_unchanged: bool
    dca_fault_count: int

    def __post_init__(self) -> None:
        if (
            type(self.valid_observation_count) is not int
            or self.valid_observation_count < 0
            or type(self.expected_observation_count) is not int
            or self.expected_observation_count < 0
            or type(self.dca_fault_count) is not int
            or self.dca_fault_count < 0
        ):
            raise ValueError("PILOT safety gate counts must be non-negative integers")
        for name in (
            "every_restore_verified",
            "final_b0_verified",
            "dataset_complete",
            "identities_unchanged",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")

    def rejection_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.expected_observation_count != 3:
            reasons.append("expected_observation_count")
        if self.valid_observation_count != self.expected_observation_count:
            reasons.append("valid_observation_count")
        if not self.every_restore_verified:
            reasons.append("every_restore_verified")
        if not self.final_b0_verified:
            reasons.append("final_b0_verified")
        if not self.dataset_complete:
            reasons.append("dataset_complete")
        if not self.identities_unchanged:
            reasons.append("identities_unchanged")
        if self.dca_fault_count != 0:
            reasons.append("dca_fault_count")
        return tuple(reasons)


class PilotSafetyGateError(RuntimeError):
    """The durable PILOT qualification does not authorize FAST24."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = tuple(reasons)
        super().__init__(
            "PILOT_3 safety gate rejected FAST24: " + ", ".join(self.reasons)
        )


class StagedLiveAIExecution:
    """Run PILOT_3, durably cross its gate, then run exactly FAST24."""

    def __init__(
        self,
        task_id: str,
        database_path: str | Path,
        *,
        pilot_execution: _Execution,
        fast_execution_factory: Callable[[], _Execution],
        gate_evidence_factory: Callable[[], PilotSafetyGateEvidence],
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be nonblank")
        if not callable(fast_execution_factory):
            raise TypeError("fast_execution_factory must be callable")
        if not callable(gate_evidence_factory):
            raise TypeError("gate_evidence_factory must be callable")
        if not callable(now):
            raise TypeError("now must be callable")
        if not callable(getattr(pilot_execution, "run", None)) or not callable(
            getattr(pilot_execution, "close", None)
        ):
            raise TypeError("pilot_execution must implement run and close")

        self._task_id = task_id
        self._database_path = Path(database_path)
        self._pilot_execution = pilot_execution
        self._fast_execution_factory = fast_execution_factory
        self._gate_evidence_factory = gate_evidence_factory
        self._now = now
        self._fast_execution: _Execution | None = None
        self._closed = False

    @staticmethod
    def _pilot_dataset_complete(rows: tuple[ObservationRecord, ...]) -> bool:
        for row in rows:
            raw = row.raw_response
            if not isinstance(raw, Mapping) or raw.get("dataset_status") != "COMPLETE":
                return False
        return True

    def _durable_pilot_rows(self) -> tuple[ObservationRecord, ...]:
        with TaskStore(self._database_path) as store:
            return tuple(
                row
                for row in store.observations(self._task_id)
                if row.phase == "PILOT_3"
            )

    def _evaluate_gate(self) -> PilotSafetyGateEvidence:
        evidence = self._gate_evidence_factory()
        if not isinstance(evidence, PilotSafetyGateEvidence):
            raise TypeError(
                "gate_evidence_factory must return PilotSafetyGateEvidence"
            )
        rows = self._durable_pilot_rows()
        durable_valid = sum(row.valid and row.tdecq is not None for row in rows)
        reasons = list(evidence.rejection_reasons())
        if len(rows) != 3:
            reasons.append("durable_observation_count")
        if durable_valid != 3 or durable_valid != evidence.valid_observation_count:
            reasons.append("durable_valid_observation_count")
        if not self._pilot_dataset_complete(rows):
            reasons.append("durable_dataset_evidence")
        if reasons:
            unique_reasons = tuple(dict.fromkeys(reasons))
            with TaskStore(self._database_path) as store:
                store.append_audit_event(
                    self._task_id,
                    event_type="PILOT_SAFETY_GATE_FAILED",
                    payload={
                        **asdict(evidence),
                        "reasons": list(unique_reasons),
                    },
                    occurred_at=self._now(),
                )
            raise PilotSafetyGateError(unique_reasons)
        return evidence

    def _persist_passing_gate(self, evidence: PilotSafetyGateEvidence) -> None:
        with TaskStore(self._database_path) as store:
            store.append_audit_event(
                self._task_id,
                event_type="PILOT_SAFETY_GATE_PASSED",
                payload=asdict(evidence),
                occurred_at=self._now(),
            )

    def _verify_exact_fast24(self) -> None:
        with TaskStore(self._database_path) as store:
            rows = store.observations(self._task_id)
        phases = tuple(
            row.phase
            for row in rows
            if row.phase != "PILOT_3" and row.phase not in _NON_CANDIDATE_PHASES
        )
        if phases != FAST24_PHASES:
            raise RuntimeError(
                "FAST24 phase allocation mismatch: "
                f"expected {FAST24_PHASES!r}, got {phases!r}"
            )

    def run(self, stop_requested: Event) -> LifecycleState | None:
        if self._closed:
            raise RuntimeError("execution is closed")
        if not isinstance(stop_requested, Event):
            raise TypeError("stop_requested must be an Event")

        pilot_terminal = self._pilot_execution.run(stop_requested)
        if stop_requested.is_set() or pilot_terminal is None:
            return None
        if pilot_terminal not in {
            LifecycleState.COMPLETED,
            LifecycleState.COMPLETED_NO_IMPROVEMENT,
        }:
            return pilot_terminal

        evidence = self._evaluate_gate()
        if stop_requested.is_set():
            return None
        self._persist_passing_gate(evidence)
        if stop_requested.is_set():
            return None

        # Construction is intentionally after the committed gate transaction.
        fast_execution = self._fast_execution_factory()
        if not callable(getattr(fast_execution, "run", None)) or not callable(
            getattr(fast_execution, "close", None)
        ):
            raise TypeError("fast execution must implement run and close")
        self._fast_execution = fast_execution
        terminal = fast_execution.run(stop_requested)
        if terminal is None or stop_requested.is_set():
            return None
        self._verify_exact_fast24()
        return terminal

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._fast_execution is not None:
                self._fast_execution.close()
        finally:
            self._pilot_execution.close()

