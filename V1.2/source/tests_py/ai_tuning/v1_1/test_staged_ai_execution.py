from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pytest

from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.repository import ObservationRecord, TaskStore
from vcsel_ai_tuning.v1_1.staged_ai_execution import (
    PilotSafetyGateError,
    PilotSafetyGateEvidence,
    StagedLiveAIExecution,
)


NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)
FAST_PHASES = (
    ("I2C_COARSE",) * 6
    + ("I2C_FINE",) * 16
    + ("FIR_FIRST", "FIR_ITERATIVE")
)


def _create_task(database: Path, task_id: str = "staged") -> None:
    with TaskStore(database) as store:
        store.create_task(
            task_id=task_id,
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.LIVE,
            config_fingerprint="c" * 64,
            payload={"live_ai_profile": "PILOT_3_THEN_FAST24"},
        )


def _append_observation(
    database: Path,
    task_id: str,
    *,
    sequence: int,
    phase: str,
    valid: bool = True,
) -> None:
    row = ObservationRecord.measurement(
        observation_id=f"obs-{sequence:02d}",
        sequence=sequence,
        phase=phase,
        requested={"ibias_dac": 71, "res_gain": 6},
        applied={"ibias_dac": 71, "res_gain": 6},
        readback={"ibias_dac": 71, "res_gain": 6},
        tdecq=2.0 if valid else None,
        er=4.0,
        raw_response={"dataset_status": "COMPLETE"},
        error_reason=None if valid else "INVALID_TDECQ",
        created_at=NOW,
    )
    with TaskStore(database) as store:
        store.append_observation(task_id, row)


class _PilotExecution:
    def __init__(self, database: Path, task_id: str, recorder: list[str]) -> None:
        self.database = database
        self.task_id = task_id
        self.recorder = recorder
        self.closed = 0

    def run(self, stop_requested: Event):
        for sequence in range(1, 4):
            _append_observation(
                self.database,
                self.task_id,
                sequence=sequence,
                phase="PILOT_3",
            )
            self.recorder.append("PILOT_3")
        return LifecycleState.COMPLETED

    def close(self):
        self.closed += 1


class _FastExecution:
    def __init__(self, database: Path, task_id: str, recorder: list[str]) -> None:
        self.database = database
        self.task_id = task_id
        self.recorder = recorder
        self.closed = 0

    def run(self, stop_requested: Event):
        with TaskStore(self.database) as store:
            assert any(
                event.event_type == "PILOT_SAFETY_GATE_PASSED"
                for event in store.events(self.task_id)
            )
        self.recorder.append("FAST24_CANDIDATE_APPLIED")
        for offset, phase in enumerate(FAST_PHASES, start=4):
            _append_observation(
                self.database,
                self.task_id,
                sequence=offset,
                phase=phase,
            )
            self.recorder.append(phase)
        return LifecycleState.COMPLETED

    def close(self):
        self.closed += 1


def _valid_gate() -> PilotSafetyGateEvidence:
    return PilotSafetyGateEvidence(
        valid_observation_count=3,
        expected_observation_count=3,
        every_restore_verified=True,
        final_b0_verified=True,
        dataset_complete=True,
        identities_unchanged=True,
        dca_fault_count=0,
    )


def _execution(tmp_path: Path, evidence: PilotSafetyGateEvidence):
    database = tmp_path / "campaigns.sqlite3"
    _create_task(database)
    recorder: list[str] = []
    pilot = _PilotExecution(database, "staged", recorder)
    made_fast: list[_FastExecution] = []

    def make_fast():
        execution = _FastExecution(database, "staged", recorder)
        made_fast.append(execution)
        return execution

    execution = StagedLiveAIExecution(
        "staged",
        database,
        pilot_execution=pilot,
        fast_execution_factory=make_fast,
        gate_evidence_factory=lambda: evidence,
        now=lambda: NOW,
    )
    return execution, recorder, made_fast, database, pilot


def test_staged_execution_enters_fast24_only_after_durable_pilot_gate(tmp_path):
    execution, recorder, made_fast, database, _pilot = _execution(
        tmp_path, _valid_gate()
    )

    terminal = execution.run(Event())

    assert terminal is LifecycleState.COMPLETED
    assert recorder[:3] == ["PILOT_3"] * 3
    assert recorder[3:] == ["FAST24_CANDIDATE_APPLIED", *FAST_PHASES]
    assert len(made_fast) == 1
    with TaskStore(database) as store:
        events = store.events("staged")
        phases = tuple(row.phase for row in store.observations("staged"))
    gate = next(event for event in events if event.event_type == "PILOT_SAFETY_GATE_PASSED")
    assert gate.payload["valid_observation_count"] == 3
    assert phases == ("PILOT_3",) * 3 + FAST_PHASES


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("valid_observation_count", 2),
        ("expected_observation_count", 4),
        ("every_restore_verified", False),
        ("final_b0_verified", False),
        ("dataset_complete", False),
        ("identities_unchanged", False),
        ("dca_fault_count", 1),
    ),
)
def test_each_pilot_gate_failure_blocks_every_fast_write(tmp_path, field, value):
    evidence = replace(_valid_gate(), **{field: value})
    execution, recorder, made_fast, database, _pilot = _execution(tmp_path, evidence)

    with pytest.raises(PilotSafetyGateError, match="PILOT_3 safety gate"):
        execution.run(Event())

    assert recorder == ["PILOT_3"] * 3
    assert made_fast == []
    with TaskStore(database) as store:
        assert tuple(row.phase for row in store.observations("staged")) == (
            "PILOT_3",
        ) * 3
        assert not any(
            event.event_type == "PILOT_SAFETY_GATE_PASSED"
            for event in store.events("staged")
        )


def test_durable_invalid_pilot_row_blocks_fast_even_if_probe_claims_valid(tmp_path):
    execution, recorder, made_fast, database, _pilot = _execution(
        tmp_path, _valid_gate()
    )
    original = execution._pilot_execution

    def invalid_run(stop_requested):
        for sequence in range(1, 4):
            _append_observation(
                database,
                "staged",
                sequence=sequence,
                phase="PILOT_3",
                valid=sequence != 2,
            )
            recorder.append("PILOT_3")
        return LifecycleState.COMPLETED

    original.run = invalid_run

    with pytest.raises(PilotSafetyGateError):
        execution.run(Event())
    assert made_fast == []


def test_stop_after_pilot_skips_fast_and_close_is_idempotent(tmp_path):
    execution, recorder, made_fast, _database, pilot = _execution(
        tmp_path, _valid_gate()
    )
    stop = Event()
    original_run = pilot.run

    def run_then_stop(stop_requested):
        result = original_run(stop_requested)
        stop_requested.set()
        return result

    pilot.run = run_then_stop

    assert execution.run(stop) is None
    execution.close()
    execution.close()

    assert recorder == ["PILOT_3"] * 3
    assert made_fast == []
    assert pilot.closed == 1


def test_fast_return_with_wrong_phase_allocation_is_rejected(tmp_path):
    execution, _recorder, made_fast, database, _pilot = _execution(
        tmp_path, _valid_gate()
    )

    class WrongFast(_FastExecution):
        def run(self, stop_requested):
            for offset, phase in enumerate(FAST_PHASES[:-1], start=4):
                _append_observation(
                    self.database,
                    self.task_id,
                    sequence=offset,
                    phase=phase,
                )
            return LifecycleState.COMPLETED

    execution._fast_execution_factory = lambda: WrongFast(database, "staged", [])

    with pytest.raises(RuntimeError, match="FAST24 phase allocation"):
        execution.run(Event())
