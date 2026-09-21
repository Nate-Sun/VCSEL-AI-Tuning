from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pytest

import vcsel_ai_tuning.optimization.v11 as v11
from vcsel_ai_tuning.v1_1.ai_execution import (
    DcaObservationInvalidError,
    FinalSafetyGuard,
    OptimizationPlant,
    V11AIExecution,
)
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.repository import TaskStore


def _b0() -> B0Snapshot:
    return B0Snapshot.capture(
        b0_id="native-v12-pilot-b0",
        version=1,
        captured_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        i2c_fields={"ibias_dac": 70, "res_gain": 6},
        i2c_raw_registers={0x2000: 70, 0x2001: 6},
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        bert_pattern="SSPRQ",
        i2c_channel=0,
        bert_port="port1",
        bert_channel="0",
        dca_tdecq_source="FUNC1",
        dca_tdecq_equalizer_function=1,
        dca_outer_er_source="CHAN1A",
        dca_fingerprint=DcaFingerprint.from_mappings(
            identity="FAKE,DCA,Native-V1.2",
            measurement_sources={"outer_er": "CHAN1A", "tdecq": "FUNC1"},
            configuration={"tdecq_equalizer_function": "FUNC1"},
        ),
        hardware_identities={
            "i2c": HardwareIdentity("i2c", "FAKE-I2C", "VD1124"),
            "bert": HardwareIdentity("bert", "FAKE-BERT", "VIT"),
            "dca": HardwareIdentity("dca", "FAKE-DCA", "FlexDCA"),
        },
    )


def _config() -> dict[str, object]:
    return {
        "task_type": "AI_OPTIMIZATION",
        "optimization_scope": "I2C_ONLY",
        "depth": "STANDARD",
        "candidate_budget": 3,
        "coarse_candidate_budget": 3,
        "fine_candidate_budget": 0,
        "settle_time_s": 0.1,
        "restore_after_each_candidate": True,
        "save_dataset": True,
        "seed": 17,
        "parameters": (
            {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
            {"name": "res_gain", "min": 4, "max": 8, "step": 4},
        ),
    }


def _create_task(database: Path, task_id: str) -> None:
    with TaskStore(database) as store:
        store.create_task(
            task_id=task_id,
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.FAKE,
            config_fingerprint="a" * 64,
            payload=_config(),
        )
        store.save_b0(task_id, _b0())


class RecordingPlant:
    def __init__(
        self,
        database: Path,
        task_id: str,
        results: tuple[v11.MetricResult | Exception, ...],
        *,
        stop_after_measurement: Event | None = None,
        malformed_readback: bool = False,
        fail_restore: bool = False,
    ) -> None:
        self.database = database
        self.task_id = task_id
        self.results = iter(results)
        self.stop_after_measurement = stop_after_measurement
        self.malformed_readback = malformed_readback
        self.fail_restore = fail_restore
        self.current = dict(_b0().i2c_fields)
        self.fir = v11.FirTarget(
            taps=_b0().fir_taps,
            upper_nlc=_b0().upper_nlc,
            lower_nlc=_b0().lower_nlc,
        )
        self.events: list[tuple[object, ...]] = []
        self.apply_count = 0
        self.restore_count = 0

    @property
    def current_i2c(self) -> Mapping[str, int]:
        return dict(self.current)

    @property
    def current_fir(self) -> object:
        return self.fir

    def apply_i2c(self, requested: Mapping[str, int]) -> Mapping[str, object]:
        with TaskStore(self.database) as store:
            assert len(store.observations(self.task_id)) == self.apply_count
        self.apply_count += 1
        self.current = dict(requested)
        readback = dict(requested)
        if self.malformed_readback:
            readback["ibias_dac"] += 1
        self.events.append(("apply_i2c", dict(requested)))
        return {"applied": dict(requested), "readback": readback}

    def apply_fir(self, target: object) -> object:
        raise AssertionError("PILOT_3 must not write FIR")

    def read_dca_taps(self) -> tuple[object, int]:
        raise AssertionError("PILOT_3 must not request FIR proposals")

    def measure(self) -> object:
        self.events.append(("measure", dict(self.current)))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        if self.stop_after_measurement is not None:
            self.stop_after_measurement.set()
        return result

    def restore_b0(self) -> Mapping[str, int] | None:
        self.restore_count += 1
        self.events.append(("restore_b0",))
        if self.fail_restore:
            raise RuntimeError("injected restore failure")
        self.current = dict(_b0().i2c_fields)
        return dict(self.current)

    def is_at_b0(self) -> bool:
        self.events.append(("is_at_b0",))
        return self.current == dict(_b0().i2c_fields)

    def close(self) -> None:
        self.events.append(("close",))


class RecordingGuard:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.calls = 0

    def verify(self) -> Mapping[str, object]:
        self.calls += 1
        self.events.append(("final_guard",))
        return {"verified": True, "pattern": "SSPRQ"}


def _metric(tdecq: float, er: float = 3.0) -> v11.MetricResult:
    return v11.MetricResult.valid(tdecq_db=tdecq, er_db=er)


def _execution(tmp_path, results, **plant_options):
    database = tmp_path / "tasks.sqlite3"
    task_id = "native-v12-pilot"
    _create_task(database, task_id)
    plant = RecordingPlant(database, task_id, results, **plant_options)
    guard = RecordingGuard(plant.events)
    execution = V11AIExecution(
        task_id,
        _b0(),
        _config(),
        database,
        tmp_path / "artifacts",
        plant=plant,
        final_guard=guard,
        wait=lambda _seconds: None,
    )
    return execution, plant, guard, database, task_id


def test_constructor_requires_final_guard_before_any_side_effect(tmp_path):
    database = tmp_path / "guard.sqlite3"
    task_id = "missing-guard"
    _create_task(database, task_id)
    plant = RecordingPlant(database, task_id, (_metric(3.0),))

    with pytest.raises(ValueError, match="final_guard"):
        V11AIExecution(
            task_id,
            _b0(),
            _config(),
            database,
            tmp_path / "artifacts",
            plant=plant,
            wait=lambda _seconds: None,
        )

    assert plant.events == []


def test_pilot_persists_each_observation_before_next_suggestion(tmp_path):
    execution, plant, guard, database, task_id = _execution(
        tmp_path,
        (_metric(3.1), _metric(2.8), _metric(2.6, er=1.0)),
    )

    assert isinstance(plant, OptimizationPlant)
    assert isinstance(guard, FinalSafetyGuard)
    assert execution.run(Event()) is LifecycleState.COMPLETED

    with TaskStore(database) as store:
        rows = store.observations(task_id)
        decisions = [
            event
            for event in store.events(task_id)
            if event.event_type == "AI_PILOT_CANDIDATE_SELECTED"
        ]
    assert len(rows) == 3
    assert [row.phase for row in rows] == ["PILOT_3"] * 3
    assert [event.payload["strategy"] for event in decisions] == [
        "seed",
        "diversity_bootstrap",
        "bayesian_acquisition",
    ]
    assert [event.payload["observation_ids_used"] for event in decisions] == [
        [],
        [rows[0].observation_id],
        [rows[0].observation_id, rows[1].observation_id],
    ]
    assert all(row.requested == row.applied == row.readback for row in rows)
    assert plant.restore_count == 3
    assert guard.calls == 1
    assert plant.current_i2c == dict(_b0().i2c_fields)


def test_pilot_observation_ids_are_scoped_to_each_task_in_one_database(tmp_path):
    database = tmp_path / "shared-tasks.sqlite3"
    task_ids = ("native-v12-soak-first", "native-v12-soak-second")

    for task_id in task_ids:
        _create_task(database, task_id)
        plant = RecordingPlant(
            database,
            task_id,
            (_metric(3.1), _metric(2.8), _metric(2.6)),
        )
        execution = V11AIExecution(
            task_id,
            _b0(),
            _config(),
            database,
            tmp_path / "artifacts",
            plant=plant,
            final_guard=RecordingGuard(plant.events),
            wait=lambda _seconds: None,
        )

        assert execution.run(Event()) is LifecycleState.COMPLETED
        with TaskStore(database) as store:
            store.transition(
                task_id,
                expected_state=LifecycleState.DRAFT,
                new_state=LifecycleState.COMPLETED,
                event_type="TEST_RUNNER_COMPLETED",
            )

    with TaskStore(database) as store:
        first = store.observations(task_ids[0])
        second = store.observations(task_ids[1])

    assert [row.observation_id for row in first] == [
        f"{task_ids[0]}-pilot-{sequence:08d}" for sequence in range(1, 4)
    ]
    assert [row.observation_id for row in second] == [
        f"{task_ids[1]}-pilot-{sequence:08d}" for sequence in range(1, 4)
    ]
    assert {row.observation_id for row in first}.isdisjoint(
        row.observation_id for row in second
    )


def test_invalid_observation_is_persisted_then_stops_later_candidates(tmp_path):
    invalid = v11.MetricResult.invalid(status="INV", reason="DCA invalid")
    execution, plant, _guard, database, task_id = _execution(
        tmp_path,
        (_metric(3.1), invalid),
    )

    with pytest.raises(DcaObservationInvalidError, match="invalid"):
        execution.run(Event())

    with TaskStore(database) as store:
        rows = store.observations(task_id)
    assert len(rows) == 2
    assert rows[-1].tdecq is None
    assert rows[-1].valid is False
    assert plant.apply_count == 2
    assert plant.restore_count == 2
    assert plant.current_i2c == dict(_b0().i2c_fields)


def test_cooperative_stop_restores_before_returning(tmp_path):
    stop = Event()
    execution, plant, guard, database, task_id = _execution(
        tmp_path,
        (_metric(3.1),),
        stop_after_measurement=stop,
    )

    assert execution.run(stop) is None

    with TaskStore(database) as store:
        assert len(store.observations(task_id)) == 1
    assert plant.restore_count == 1
    assert plant.current_i2c == dict(_b0().i2c_fields)
    assert guard.calls == 0


def test_measurement_failure_makes_one_restore_attempt(tmp_path):
    execution, plant, _guard, _database, _task_id = _execution(
        tmp_path,
        (RuntimeError("injected measurement failure"),),
    )

    with pytest.raises(RuntimeError, match="measurement failure"):
        execution.run(Event())

    assert plant.restore_count == 1
    assert plant.current_i2c == dict(_b0().i2c_fields)


def test_restore_failure_is_not_retried_or_hidden(tmp_path):
    execution, plant, _guard, _database, _task_id = _execution(
        tmp_path,
        (_metric(3.1),),
        fail_restore=True,
    )

    with pytest.raises(RuntimeError, match="restore failure"):
        execution.run(Event())

    assert plant.restore_count == 1


def test_readback_mismatch_stops_before_measurement_and_restores(tmp_path):
    execution, plant, _guard, _database, _task_id = _execution(
        tmp_path,
        (_metric(3.1),),
        malformed_readback=True,
    )

    with pytest.raises(RuntimeError, match="readback"):
        execution.run(Event())

    assert not any(event[0] == "measure" for event in plant.events)
    assert plant.restore_count == 1


def test_close_is_idempotent(tmp_path):
    execution, plant, _guard, _database, _task_id = _execution(
        tmp_path,
        (_metric(3.1), _metric(2.8), _metric(2.6)),
    )

    execution.close()
    execution.close()

    assert [event for event in plant.events if event[0] == "close"] == [("close",)]
