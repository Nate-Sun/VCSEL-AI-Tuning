from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event

import pytest

import vcsel_ai_tuning.optimization.v11 as v11
from vcsel_ai_tuning.hardware.fir import calculate_iterative_pass
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.fake_ai_execution import (
    FakeOptimizationPlant,
    FakeV11AIExecution,
    build_fake_ai_execution_factory,
)
from vcsel_ai_tuning.v1_1.optimizer_bridge import load_v11_optimizer_api
from vcsel_ai_tuning.v1_1.repository import TaskStore


def make_b0() -> B0Snapshot:
    return B0Snapshot.capture(
        b0_id="fake-ai-b0",
        version=1,
        captured_at=datetime(2026, 7, 29, 9, 0, tzinfo=timezone.utc),
        i2c_fields={"eq": 22, "ibias_dac": 70},
        i2c_raw_registers={0x10: 22, 0x11: 70},
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        bert_pattern="SSPRQ",
        i2c_channel=0,
        bert_port="port1",
        bert_channel="0",
        dca_tdecq_source="CHAN1A",
        dca_tdecq_equalizer_function=1,
        dca_outer_er_source="CHAN1A",
        dca_fingerprint=DcaFingerprint.from_mappings(
            identity="FAKE,DCA,1.1",
            measurement_sources={"outer_er": "CHAN1A", "tdecq": "CHAN1A"},
            configuration={
                "pattern": "SSPRQ",
                "tdecq_equalizer_function": "FUNC1",
            },
        ),
        hardware_identities={
            "i2c": HardwareIdentity("i2c", "FAKE-I2C", "FAKE-VD1124"),
            "bert": HardwareIdentity("bert", "FAKE-BERT", "FAKE-VIT"),
            "dca": HardwareIdentity("dca", "FAKE-DCA", "FAKE-FlexDCA"),
        },
    )


def test_plant_starts_from_b0_changes_measurement_and_restores_exactly():
    b0 = make_b0()
    plant = FakeOptimizationPlant(b0)

    expected_b0_i2c = dict(b0.i2c_fields)
    assert plant.current_i2c == expected_b0_i2c
    assert plant.current_fir.taps == b0.fir_taps
    assert plant.is_at_b0()
    current_view = plant.current_i2c
    current_view["eq"] += 100
    assert plant.current_i2c == expected_b0_i2c
    baseline = plant.measure()
    assert baseline.tdecq_db == 2.8
    assert baseline.er_db == 5.4

    requested = dict(b0.i2c_fields)
    requested["eq"] += 1
    readback = plant.apply_i2c(requested)

    assert readback == requested
    assert plant.current_i2c == readback
    readback["eq"] += 100
    assert plant.current_i2c == requested
    changed = plant.measure()
    assert changed.tdecq_db == 2.78
    assert changed.er_db == 5.41
    assert changed != baseline

    plant.restore_b0()

    assert plant.is_at_b0()
    assert plant.measure() == baseline


def test_complete_state_measurement_is_deterministic_with_exact_fir_readback():
    b0 = make_b0()
    plant = FakeOptimizationPlant(b0)
    fir_target = load_v11_optimizer_api().fir_target(
        taps=(1, 9, -28, 104, -8, 1, -3),
        upper_nlc=b0.upper_nlc,
        lower_nlc=b0.lower_nlc,
    )

    readback = plant.apply_fir(fir_target)
    first = plant.measure()
    second = plant.measure()
    dca_taps, main_index = plant.read_dca_taps()

    assert readback == fir_target.taps
    assert isinstance(readback, tuple)
    assert plant.current_fir is fir_target
    assert first == second
    assert first.tdecq_db == 2.802
    assert first.er_db == 5.399
    norm = max(1, sum(abs(value) for value in fir_target.taps))
    assert len(dca_taps) == len(fir_target.taps) == 7
    assert dca_taps == pytest.approx(
        tuple(value / norm for value in fir_target.taps)
    )
    assert main_index == 3


def test_measure_only_generates_metrics_without_optimizer_comparison(monkeypatch):
    comparison_calls = []

    def reject_comparison(*args, **kwargs):
        comparison_calls.append((args, kwargs))
        raise AssertionError("FAKE plant must not compare or rank results")

    monkeypatch.setattr(v11, "compare_results", reject_comparison)
    plant = FakeOptimizationPlant(make_b0())

    result = plant.measure()

    assert isinstance(result, v11.MetricResult)
    assert result.has_valid_tdecq
    assert comparison_calls == []


@pytest.mark.parametrize(
    "requested",
    (
        {"eq": 23},
        {"eq": 23, "ibias_dac": 70, "unexpected": 1},
    ),
)
def test_apply_i2c_rejects_any_non_exact_field_set(requested):
    plant = FakeOptimizationPlant(make_b0())

    with pytest.raises(ValueError, match="field set"):
        plant.apply_i2c(requested)


def _task_config(*, budget=8, depth="FAST"):
    return {
        "parameters": (
            {"name": "eq", "min": 20, "max": 24, "step": 1},
            {"name": "ibias_dac", "min": 68, "max": 72, "step": 1},
        ),
        "candidate_budget": budget,
        "seed": 17,
        "depth": depth,
    }


def _create_task(database: Path, task_id: str, config):
    with TaskStore(database) as store:
        store.create_task(
            task_id=task_id,
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.FAKE,
            config_fingerprint="c" * 64,
            payload=config,
        )
        store.save_b0(task_id, make_b0())


def test_optional_ai_dataset_saves_every_committed_observation(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    artifact_root = tmp_path / "artifacts"
    config = {**_task_config(budget=2), "save_dataset": True}
    _create_task(database, "ai-dataset", config)
    execution = FakeV11AIExecution(
        "ai-dataset",
        make_b0(),
        config,
        database,
        artifact_root,
    )

    terminal = execution.run(Event())

    assert terminal in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }
    with TaskStore(database) as store:
        observations = store.observations("ai-dataset")
    bundles = sorted((artifact_root / "ai-dataset").glob("iteration-*"))
    assert len(bundles) == len(observations)
    for bundle, observation in zip(bundles, observations, strict=True):
        assert {path.name for path in bundle.iterdir()} == {
            "record.json",
            "raw_eye_density.npy",
            "raw_eye.png",
            "tdecq_equalized_eye.png",
        }
        record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
        assert record["observation_id"] == observation.observation_id
        assert record["measurement_id"].startswith("measurement-")
        assert record["acquisition_id"].startswith("acquisition-")
        assert record["dataset_status"] == "COMPLETE"


def test_ai_dataset_disabled_creates_no_iteration_bundles(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    artifact_root = tmp_path / "artifacts"
    config = {**_task_config(budget=1), "save_dataset": False}
    _create_task(database, "ai-no-dataset", config)

    FakeV11AIExecution(
        "ai-no-dataset",
        make_b0(),
        config,
        database,
        artifact_root,
    ).run(Event())

    assert list((artifact_root / "ai-no-dataset").glob("iteration-*")) == []


class ScriptedPlant(FakeOptimizationPlant):
    DCA_TAPS = (
        0.0,
        0.0,
        -0.1,
        1.0,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    def __init__(self, b0, results):
        super().__init__(b0)
        self.results = iter(results)
        self.restore_calls = 0
        self.applied = []
        self.fir_apply_count = 0

    def apply_i2c(self, requested):
        readback = super().apply_i2c(requested)
        self.applied.append(("I2C", dict(readback)))
        return readback

    def apply_fir(self, target):
        readback = super().apply_fir(target)
        self.fir_apply_count += 1
        self.applied.append(("FIR", tuple(readback)))
        return readback

    def read_dca_taps(self):
        if self.fir_apply_count:
            return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 3
        return self.DCA_TAPS, 3

    def measure(self):
        return next(self.results)

    def restore_b0(self):
        self.restore_calls += 1
        super().restore_b0()


class BoundaryStoppingPlant(ScriptedPlant):
    def __init__(self, b0, results, *, stop_event, stop_on_measure):
        super().__init__(b0, results)
        self.stop_event = stop_event
        self.stop_on_measure = stop_on_measure
        self.measure_count = 0

    def measure(self):
        result = super().measure()
        self.measure_count += 1
        if self.measure_count == self.stop_on_measure:
            self.stop_event.set()
        return result


class ClampedI2CPlant(ScriptedPlant):
    def apply_i2c(self, requested):
        clamped = dict(requested)
        clamped["eq"] = 20 + ((requested["eq"] - 20 + 1) % 5)
        clamped["ibias_dac"] = 68 + ((requested["ibias_dac"] - 68 + 1) % 5)
        readback = FakeOptimizationPlant.apply_i2c(self, clamped)
        self.applied.append(("I2C", dict(readback)))
        return readback


class InvalidEqualizerPlant(ScriptedPlant):
    def read_dca_taps(self):
        return (0.0, 0.0, "bad", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 3


def _metric(tdecq, er):
    return v11.MetricResult.valid(tdecq_db=tdecq, er_db=er)


@pytest.mark.parametrize(
    ("candidate_results", "expected_tdecq", "expected_er"),
    (
        ((_metric(2.0, 1.0), _metric(2.1, 100.0)), 2.0, 1.0),
        ((_metric(2.0, 1.0), _metric(2.0, 2.0)), 2.0, 1.0),
    ),
)
def test_persisted_best_uses_bridge_strict_objective(
    tmp_path, candidate_results, expected_tdecq, expected_er
):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=2)
    _create_task(database, "objective", config)
    compare_calls = []
    api = load_v11_optimizer_api()

    def compare(candidate, reference):
        compare_calls.append((candidate, reference))
        return api.compare_results(candidate, reference)

    bridged = replace(api, compare_results=compare)
    plant = ScriptedPlant(
        make_b0(),
        (
            _metric(3.0, 3.0),
            *candidate_results,
            _metric(expected_tdecq, expected_er),
            _metric(expected_tdecq, expected_er),
            _metric(expected_tdecq, expected_er),
        ),
    )
    execution = FakeV11AIExecution(
        "objective",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        optimizer_api=bridged,
        plant_factory=lambda _b0: plant,
    )

    terminal = execution.run(Event())

    with TaskStore(database) as store:
        best = store.best_state("objective")
        assert best is not None
        assert best.result["tdecq_db"] == expected_tdecq
        assert best.result["er_db"] == expected_er
        assert set(best.i2c) == set(dict(make_b0().i2c_fields))
        assert len(best.fir) == 7
        assert best.upper_nlc == make_b0().upper_nlc
        assert best.lower_nlc == make_b0().lower_nlc
        assert best.pattern == make_b0().bert_pattern
        assert (best.bert_port, best.bert_channel) == (
            make_b0().bert_port,
            make_b0().bert_channel,
        )
        assert best.dca_fingerprint == make_b0().dca_fingerprint.content_hash
        assert best.result["tdecq_status"] == "CORR"
        assert best.observation_id in {
            row.observation_id for row in store.observations("objective")
        }
    assert terminal is LifecycleState.COMPLETED
    assert compare_calls


def test_fine_suggest_receives_the_previous_actual_readback_and_metric(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=7)
    _create_task(database, "adaptive", config)
    api = load_v11_optimizer_api()
    suggest_histories = []

    class SpyFine:
        def __init__(self, *args, **kwargs):
            self._inner = api.fine_optimizer(*args, **kwargs)

        def suggest(self, observations, pending_i2c=()):
            suggest_histories.append(observations)
            return self._inner.suggest(observations, pending_i2c)

    plant = ScriptedPlant(make_b0(), tuple(_metric(3.0 - index / 10, 4.0) for index in range(20)))
    execution = FakeV11AIExecution(
        "adaptive",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        optimizer_api=replace(api, fine_optimizer=SpyFine),
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    assert len(suggest_histories) >= 2
    first_ids = {row.observation_id for row in suggest_histories[0]}
    added = [row for row in suggest_histories[1] if row.observation_id not in first_ids]
    assert len(added) == 1
    assert dict(added[0].readback_i2c) == dict(added[0].requested_i2c)
    assert isinstance(added[0].result, v11.MetricResult)


def test_fir_confirmation_batch_commits_one_observation_using_bridge_aggregate(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=8)
    _create_task(database, "fir-confirm", config)
    api = load_v11_optimizer_api()
    aggregate_sizes = []

    def aggregate(observations):
        aggregate_sizes.append(len(observations))
        return api.aggregate_repeated_observations(observations)

    results = (
        _metric(4.0, 4.0),
        _metric(3.8, 4.0),
        _metric(3.7, 4.0),
        _metric(3.6, 4.0),
        _metric(3.5, 4.0),
        _metric(3.4, 4.0),
        _metric(3.3, 4.0),
        _metric(2.0, 3.0),
        _metric(2.2, 3.2),
        _metric(2.1, 3.1),
        _metric(1.9, 3.0),
        _metric(1.9, 3.0),
        _metric(1.9, 3.0),
        _metric(1.9, 3.0),
    )
    plant = ScriptedPlant(make_b0(), results)
    execution = FakeV11AIExecution(
        "fir-confirm",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        optimizer_api=replace(api, aggregate_repeated_observations=aggregate),
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    with TaskStore(database) as store:
        fir_rows = [
            row for row in store.observations("fir-confirm")
            if row.phase == "FIR_FIRST"
        ]
    assert len(fir_rows) == 1
    assert len(fir_rows[0].raw_response["raw_measurements"]) == 3
    assert fir_rows[0].tdecq == pytest.approx(2.1)
    assert 3 in aggregate_sizes


def test_b0_observation_persists_equalizer_taps_and_main_index(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=1)
    _create_task(database, "b0-equalizer", config)
    plant = ScriptedPlant(
        make_b0(),
        (
            _metric(5.0, 4.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
        ),
    )
    execution = FakeV11AIExecution(
        "b0-equalizer",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    with TaskStore(database) as store:
        baseline = store.observations("b0-equalizer")[0]
    assert baseline.phase == "B0"
    assert baseline.raw_response["dca_tdecq_equalizer_taps"] == list(
        ScriptedPlant.DCA_TAPS
    )
    assert baseline.raw_response["dca_tdecq_equalizer_main_index"] == 3


def test_rejected_first_fir_runs_three_best_convolutions_before_stopping(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=20, depth="DEEP")
    _create_task(database, "first-three-misses", config)
    i2c_results = tuple(_metric(4.9 - index / 10, 4.0) for index in range(10))
    plant = ScriptedPlant(
        make_b0(),
        (
            _metric(5.0, 4.0),
            *i2c_results,
            _metric(4.5, 9.0),
            _metric(4.6, 9.0),
            _metric(4.7, 9.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
        ),
    )
    execution = FakeV11AIExecution(
        "first-three-misses",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    with TaskStore(database) as store:
        rows = store.observations("first-three-misses")
    fir_rows = [row for row in rows if row.phase == "FIR_FIRST"]
    expected_first = calculate_iterative_pass(
        make_b0().fir_taps,
        ScriptedPlant.DCA_TAPS,
        dca_main_index=3,
        manual_main=make_b0().fir_taps[3],
    )
    identity = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected_later = calculate_iterative_pass(
        make_b0().fir_taps,
        identity,
        dca_main_index=3,
        manual_main=make_b0().fir_taps[3],
    )

    assert len(fir_rows) == 3
    assert [
        tuple(row.requested[f"tap_{index}"] for index in range(7))
        for row in fir_rows
    ] == [expected_first.taps, expected_later.taps, expected_later.taps]
    assert [
        row.raw_response["convolution_base_fir"] for row in fir_rows
    ] == [list(make_b0().fir_taps)] * 3
    assert [
        row.raw_response["controller_no_improvement_count"] for row in fir_rows
    ] == [1, 2, 3]
    assert not any(row.phase in {"I2C_RETUNE", "FIR_ITERATIVE"} for row in rows)


def test_invalid_equalizer_is_audited_without_applying_fir(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=20, depth="DEEP")
    _create_task(database, "invalid-equalizer", config)
    plant = InvalidEqualizerPlant(
        make_b0(),
        (
            _metric(5.0, 4.0),
            *(_metric(4.9 - index / 10, 4.0) for index in range(10)),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
        ),
    )
    execution = FakeV11AIExecution(
        "invalid-equalizer",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    with TaskStore(database) as store:
        rows = store.observations("invalid-equalizer")
        invalid_events = [
            event
            for event in store.events("invalid-equalizer")
            if event.event_type == "AI_FIR_INVALID_DCA_TAPS"
        ]
    assert not any(row.phase in {"FIR_FIRST", "FIR_ITERATIVE"} for row in rows)
    assert all(
        state == make_b0().fir_taps
        for kind, state in plant.applied
        if kind == "FIR"
    )
    assert len(invalid_events) == 1
    assert "numeric" in invalid_events[0].payload["reason"]


def test_candidate_observations_never_exceed_global_budget_and_run_does_not_restore(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=6)
    _create_task(database, "budget", config)
    plant = ScriptedPlant(make_b0(), tuple(_metric(4.0 - index / 10, 4.0) for index in range(30)))
    execution = build_fake_ai_execution_factory(
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )("budget", make_b0(), config)

    terminal = execution.run(Event())

    with TaskStore(database) as store:
        rows = store.observations("budget")
    optimization_phases = {
        "I2C_COARSE",
        "I2C_FINE",
        "FIR_FIRST",
        "I2C_RETUNE",
        "FIR_ITERATIVE",
    }
    assert len([row for row in rows if row.phase in optimization_phases]) == config[
        "candidate_budget"
    ]
    assert terminal in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }
    assert plant.restore_calls == 0


def test_iterative_fir_stops_from_controller_three_no_improvement_state(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=20, depth="DEEP")
    _create_task(database, "fir-converge", config)
    results = [
        _metric(5.0, 4.0),
        *(_metric(4.0 - index / 100, 4.0) for index in range(10)),
        _metric(2.0, 3.0), _metric(2.1, 3.0), _metric(2.2, 3.0),
        _metric(1.8, 3.0),
        _metric(2.5, 9.0), _metric(2.5, 9.0), _metric(2.5, 9.0),
        _metric(1.8, 3.0), _metric(1.8, 3.0), _metric(1.8, 3.0),
    ]
    plant = ScriptedPlant(make_b0(), tuple(results))
    execution = FakeV11AIExecution(
        "fir-converge",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    with TaskStore(database) as store:
        phases = [row.phase for row in store.observations("fir-converge")]
        iterative = [
            row for row in store.observations("fir-converge")
            if row.phase == "FIR_ITERATIVE"
        ]
    stage_chain = (
        "I2C_COARSE",
        "I2C_FINE",
        "FIR_FIRST",
        "I2C_RETUNE",
        "FIR_ITERATIVE",
        "FINAL_VERIFICATION",
    )
    assert tuple(dict.fromkeys(phase for phase in phases if phase != "B0")) == stage_chain
    assert len(iterative) == 3
    assert iterative[-1].raw_response["controller_no_improvement_count"] == 3


def test_new_execution_resume_canonicalizes_persisted_fingerprint_tuples(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=7)
    _create_task(database, "tuple-resume", config)
    stop = Event()
    first_plant = BoundaryStoppingPlant(
        make_b0(),
        tuple(_metric(4.0 - index / 10, 4.0) for index in range(20)),
        stop_event=stop,
        stop_on_measure=3,
    )
    first = FakeV11AIExecution(
        "tuple-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: first_plant,
    )
    assert first.run(stop) is None

    with TaskStore(database) as store:
        before = store.observations("tuple-resume")
    committed_coarse = before[-1]
    assert committed_coarse.phase == "I2C_COARSE"
    resumed_plant = ScriptedPlant(
        make_b0(),
        tuple(_metric(3.0 - index / 10, 4.0) for index in range(30)),
    )
    resumed = FakeV11AIExecution(
        "tuple-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: resumed_plant,
    )

    assert resumed.run(Event()) in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }
    first_resumed_i2c = next(
        state for kind, state in resumed_plant.applied if kind == "I2C"
    )
    assert first_resumed_i2c == committed_coarse.readback
    with TaskStore(database) as store:
        after = store.observations("tuple-resume")
    assert after[len(before)].phase == "I2C_FINE"


def test_resume_after_fine_restores_last_complete_state_and_continues_next_candidate(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=7)
    _create_task(database, "fine-resume", config)
    stop = Event()
    first_plant = BoundaryStoppingPlant(
        make_b0(),
        tuple(_metric(4.0 - index / 10, 4.0) for index in range(20)),
        stop_event=stop,
        stop_on_measure=4,
    )
    first = FakeV11AIExecution(
        "fine-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: first_plant,
    )
    assert first.run(stop) is None
    with TaskStore(database) as store:
        before = store.observations("fine-resume")
    committed = before[-1]
    assert committed.phase == "I2C_FINE"

    resumed_plant = ScriptedPlant(
        make_b0(),
        tuple(_metric(3.0 - index / 10, 4.0) for index in range(30)),
    )
    resumed = FakeV11AIExecution(
        "fine-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: resumed_plant,
    )
    resumed.run(Event())

    first_resumed_i2c = next(
        state for kind, state in resumed_plant.applied if kind == "I2C"
    )
    assert first_resumed_i2c == committed.readback
    with TaskStore(database) as store:
        after = store.observations("fine-resume")
    assert after[len(before)].phase == "I2C_FINE"
    assert [row.sequence for row in after] == list(range(1, len(after) + 1))
    assert len({row.observation_id for row in after}) == len(after)
    i2c_candidate_ids = [
        row.raw_response["candidate_id"]
        for row in after
        if row.phase in {"I2C_COARSE", "I2C_FINE", "I2C_RETUNE"}
    ]
    assert len(i2c_candidate_ids) == len(set(i2c_candidate_ids))
    candidate_rows = [
        row
        for row in after
        if row.phase
        in {"I2C_COARSE", "I2C_FINE", "FIR_FIRST", "I2C_RETUNE", "FIR_ITERATIVE"}
    ]
    assert [row.raw_response["budget_index"] for row in candidate_rows] == list(
        range(1, len(candidate_rows) + 1)
    )


def test_resume_replays_iterative_miss_and_controller_reaches_three_without_restart(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=20, depth="DEEP")
    _create_task(database, "iterative-resume", config)
    stop = Event()
    first_results = [
        _metric(5.0, 4.0),
        *(_metric(4.0 - index / 100, 4.0) for index in range(10)),
        _metric(2.0, 3.0), _metric(2.1, 3.0), _metric(2.2, 3.0),
        _metric(1.8, 3.0),
        _metric(2.5, 9.0),
    ]
    first_plant = BoundaryStoppingPlant(
        make_b0(),
        tuple(first_results),
        stop_event=stop,
        stop_on_measure=16,
    )
    first = FakeV11AIExecution(
        "iterative-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: first_plant,
    )
    assert first.run(stop) is None
    with TaskStore(database) as store:
        before = store.observations("iterative-resume")
    first_misses = [row for row in before if row.phase == "FIR_ITERATIVE"]
    assert len(first_misses) == 1
    assert first_misses[0].raw_response["controller_no_improvement_count"] == 1

    resumed_plant = ScriptedPlant(
        make_b0(),
        (
            _metric(2.5, 9.0),
            _metric(2.5, 9.0),
            _metric(1.8, 3.0),
            _metric(1.8, 3.0),
            _metric(1.8, 3.0),
        ),
    )
    resumed = FakeV11AIExecution(
        "iterative-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: resumed_plant,
    )
    resumed.run(Event())

    with TaskStore(database) as store:
        after = store.observations("iterative-resume")
    iterative = [row for row in after if row.phase == "FIR_ITERATIVE"]
    assert len(iterative) == 3
    assert [
        row.raw_response["controller_no_improvement_count"] for row in iterative
    ] == [1, 2, 3]
    assert all(
        row.raw_response["controller_status"] != "CANDIDATE"
        for row in iterative
    )
    assert len({row.observation_id for row in after}) == len(after)
    assert after[len(before)].phase == "FIR_ITERATIVE"


def test_resume_replays_two_first_misses_and_continues_from_b0(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=20, depth="DEEP")
    _create_task(database, "first-miss-resume", config)
    stop = Event()
    first_plant = BoundaryStoppingPlant(
        make_b0(),
        (
            _metric(5.0, 4.0),
            *(_metric(4.9 - index / 10, 4.0) for index in range(10)),
            _metric(4.5, 9.0),
            _metric(4.6, 9.0),
        ),
        stop_event=stop,
        stop_on_measure=13,
    )
    first = FakeV11AIExecution(
        "first-miss-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: first_plant,
    )

    assert first.run(stop) is None
    with TaskStore(database) as store:
        before = store.observations("first-miss-resume")
    first_rows = [row for row in before if row.phase == "FIR_FIRST"]
    assert [
        row.raw_response["controller_no_improvement_count"] for row in first_rows
    ] == [1, 2]

    resumed_plant = ScriptedPlant(
        make_b0(),
        (
            _metric(4.7, 9.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
            _metric(4.0, 4.0),
        ),
    )
    resumed = FakeV11AIExecution(
        "first-miss-resume",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: resumed_plant,
    )

    resumed.run(Event())

    with TaskStore(database) as store:
        after = store.observations("first-miss-resume")
    fir_rows = [row for row in after if row.phase == "FIR_FIRST"]
    assert [
        row.raw_response["controller_no_improvement_count"] for row in fir_rows
    ] == [1, 2, 3]
    assert [
        row.raw_response["convolution_base_fir"] for row in fir_rows
    ] == [list(make_b0().fir_taps)] * 3
    assert after[len(before)].phase == "FIR_FIRST"


def test_fine_feedback_and_persistence_use_actual_clamped_readback(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=7)
    _create_task(database, "clamped-readback", config)
    api = load_v11_optimizer_api()
    histories = []

    class SpyFine:
        def __init__(self, *args, **kwargs):
            self._inner = api.fine_optimizer(*args, **kwargs)

        def suggest(self, observations, pending_i2c=()):
            histories.append(observations)
            return self._inner.suggest(observations, pending_i2c)

    plant = ClampedI2CPlant(
        make_b0(),
        tuple(_metric(4.0 - index / 10, 4.0) for index in range(30)),
    )
    execution = FakeV11AIExecution(
        "clamped-readback",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        optimizer_api=replace(api, fine_optimizer=SpyFine),
        plant_factory=lambda _b0: plant,
    )

    execution.run(Event())

    assert len(histories) >= 2
    prior_ids = {row.observation_id for row in histories[0]}
    feedback = [row for row in histories[1] if row.observation_id not in prior_ids]
    assert len(feedback) == 1
    assert dict(feedback[0].requested_i2c) != dict(feedback[0].readback_i2c)
    with TaskStore(database) as store:
        row = next(
            item
            for item in store.observations("clamped-readback")
            if item.observation_id.replace(":", ".") == feedback[0].observation_id
        )
    assert row.requested != row.readback
    assert row.raw_response["optimizer_readback"] == dict(feedback[0].readback_i2c)


def test_run_leaves_last_candidate_state_not_earlier_persisted_best_and_restore_is_separate(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=2)
    _create_task(database, "no-auto-best", config)
    plant = ScriptedPlant(
        make_b0(),
        (
            _metric(4.0, 4.0),
            _metric(1.0, 2.0),
            _metric(2.0, 9.0),
            _metric(3.0, 3.0),
            _metric(3.0, 3.0),
            _metric(3.0, 3.0),
        ),
    )
    execution = FakeV11AIExecution(
        "no-auto-best",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    terminal = execution.run(Event())

    with TaskStore(database) as store:
        observations = store.observations("no-auto-best")
        coarse = [row for row in observations if row.phase == "I2C_COARSE"]
        best = store.best_state("no-auto-best")
    assert terminal is LifecycleState.COMPLETED
    assert best is not None
    assert best.i2c == coarse[0].readback
    assert plant.current_i2c == coarse[-1].readback
    assert plant.current_i2c != best.i2c
    assert plant.current_fir.taps == tuple(
        coarse[-1].raw_response["experiment_fingerprint"]["fir_taps"]
    )
    assert plant.restore_calls == 0

    execution.restore_b0()

    assert plant.restore_calls == 1
    assert plant.is_at_b0()


def test_resume_repairs_best_from_committed_observation_after_save_gap(
    tmp_path,
    monkeypatch,
):
    database = tmp_path / "campaigns.sqlite3"
    config = _task_config(budget=2)
    _create_task(database, "best-gap", config)
    original_save = TaskStore.save_best_state
    injected = False

    def fail_first_candidate_best(store, task_id, record):
        nonlocal injected
        if not injected and ":i2c_coarse" in record.observation_id:
            injected = True
            raise RuntimeError("injected save_best_state gap")
        return original_save(store, task_id, record)

    monkeypatch.setattr(TaskStore, "save_best_state", fail_first_candidate_best)
    first = FakeV11AIExecution(
        "best-gap",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: ScriptedPlant(
            make_b0(),
            (_metric(4.0, 4.0), _metric(1.0, 1.0)),
        ),
    )

    with pytest.raises(RuntimeError, match="injected save_best_state gap"):
        first.run(Event())

    with TaskStore(database) as store:
        committed = store.observations("best-gap")
        stale = store.best_state("best-gap")
    assert committed[-1].phase == "I2C_COARSE"
    assert committed[-1].tdecq == 1.0
    assert stale is not None
    assert stale.observation_id != committed[-1].observation_id

    monkeypatch.setattr(TaskStore, "save_best_state", original_save)
    api = load_v11_optimizer_api()
    compare_calls = []

    def compare(candidate, reference):
        compare_calls.append((candidate, reference))
        return api.compare_results(candidate, reference)

    resumed = FakeV11AIExecution(
        "best-gap",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        optimizer_api=replace(api, compare_results=compare),
        plant_factory=lambda _b0: ScriptedPlant(
            make_b0(),
            (
                _metric(2.0, 9.0),
                _metric(1.5, 2.0),
                _metric(1.5, 2.0),
                _metric(1.5, 2.0),
            ),
        ),
    )

    assert resumed.run(Event()) is LifecycleState.COMPLETED

    with TaskStore(database) as store:
        after = store.observations("best-gap")
        repaired = store.best_state("best-gap")
    assert repaired is not None
    assert repaired.observation_id == committed[-1].observation_id
    assert repaired.result["tdecq_db"] == 1.0
    assert compare_calls
    assert len({row.observation_id for row in after}) == len(after)
    candidate_ids = [
        row.raw_response["candidate_id"]
        for row in after
        if row.phase in {"I2C_COARSE", "I2C_FINE", "FIR_FIRST", "I2C_RETUNE", "FIR_ITERATIVE"}
    ]
    assert len(candidate_ids) == len(set(candidate_ids))


@pytest.mark.parametrize("budget", (1, 2, 3))
def test_candidate_budget_excludes_b0_and_final_verification(tmp_path, budget):
    database = tmp_path / f"campaigns-{budget}.sqlite3"
    config = _task_config(budget=budget)
    task_id = f"budget-excludes-final-{budget}"
    _create_task(database, task_id, config)
    plant = ScriptedPlant(
        make_b0(),
        tuple(_metric(5.0 - index / 10, 4.0) for index in range(30)),
    )
    execution = FakeV11AIExecution(
        task_id,
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    assert execution.run(Event()) in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }

    with TaskStore(database) as store:
        rows = store.observations(task_id)
    optimization_phases = {
        "I2C_COARSE",
        "I2C_FINE",
        "FIR_FIRST",
        "I2C_RETUNE",
        "FIR_ITERATIVE",
    }
    candidates = [row for row in rows if row.phase in optimization_phases]
    final = [row for row in rows if row.phase == "FINAL_VERIFICATION"]
    assert len(candidates) == budget
    assert [row.raw_response["budget_index"] for row in candidates] == list(
        range(1, budget + 1)
    )
    assert len([row for row in rows if row.phase == "B0"]) == 1
    assert len(final) == 1
    assert final[0].raw_response["budget_index"] is None
    assert final[0].sequence == len(rows)


def test_invalid_b0_does_not_consume_candidate_budget_or_break_final_boundary(
    tmp_path,
):
    database = tmp_path / "invalid-b0.sqlite3"
    config = _task_config(budget=1)
    _create_task(database, "invalid-b0-budget", config)
    invalid = v11.MetricResult.invalid(status="INV", reason="synthetic invalid B0")
    plant = ScriptedPlant(
        make_b0(),
        (
            invalid,
            _metric(2.0, 2.0),
            _metric(2.0, 2.0),
            _metric(2.0, 2.0),
            _metric(2.0, 2.0),
        ),
    )
    execution = FakeV11AIExecution(
        "invalid-b0-budget",
        make_b0(),
        config,
        database,
        tmp_path / "artifacts",
        plant_factory=lambda _b0: plant,
    )

    terminal = execution.run(Event())

    with TaskStore(database) as store:
        rows = store.observations("invalid-b0-budget")
    assert terminal is LifecycleState.COMPLETED
    assert rows[0].phase == "B0"
    assert rows[0].valid is False
    assert len([row for row in rows if row.phase == "I2C_COARSE"]) == 1
    assert rows[-1].phase == "FINAL_VERIFICATION"
    assert rows[-1].raw_response["budget_index"] is None
