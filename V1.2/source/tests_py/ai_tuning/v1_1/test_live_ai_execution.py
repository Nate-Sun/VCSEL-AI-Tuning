from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration
from vcsel_ai_tuning.optimization.v11 import FirTarget, MetricResult
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.hardware_control import RecoveryRequired
from vcsel_ai_tuning.v1_1.dataset import (
    DatasetOutcome,
    DatasetStatus,
    FrozenDatasetCapture,
    MeasurementArtifactBundleWriter,
    encode_density_png,
)
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.live_ai_execution import (
    LiveAIExecutionError,
    LiveAlternatingOptimizationPlant,
    LiveI2cOptimizationPlant,
    LiveStagedAIExecution,
    build_live_ai_execution_factory,
    open_live_ai_drivers,
)
from vcsel_ai_tuning.v1_1.live_ai_policy import LiveAIProfile
from vcsel_ai_tuning.v1_1.live_sweep import LiveSweepDriverBundle
from vcsel_ai_tuning.v1_1.optimizer_bridge import load_v11_optimizer_api
from vcsel_ai_tuning.v1_1.repository import TaskStore


ROOT = Path(__file__).resolve().parents[3]


def _profile():
    return load_live_hardware_config(
        ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
    )


def _b0() -> B0Snapshot:
    profile = _profile()
    values = {field.name: 0 for field in profile.characterization.parameters}
    values.update(ibias_dac=70, res_gain=6)
    raw = {
        field.address_for_channel(profile.characterization.default_channel): (
            values[field.name] << field.bit_start
        )
        for field in profile.characterization.parameters
    }
    identity = "Keysight Technologies,N1010A,lab,A.08"
    fingerprint = DcaFingerprint.from_mappings(
        identity=identity,
        measurement_sources={"outer_er": "CHAN1A", "tdecq": "FUNC1"},
        configuration={"tdecq_equalizer_function": "FUNC1"},
    )
    return B0Snapshot.capture(
        b0_id="native-v12-live-ai-b0",
        version=1,
        captured_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        i2c_fields=values,
        i2c_raw_registers=raw,
        fir_taps=(0, -4, -8, 104, -8, -4, 0),
        upper_nlc=6,
        lower_nlc=6,
        bert_pattern="SSPRQ",
        i2c_channel=profile.characterization.default_channel,
        bert_port=profile.bert.port,
        bert_channel=profile.bert.channel,
        dca_tdecq_source="FUNC1",
        dca_tdecq_equalizer_function=1,
        dca_outer_er_source="CHAN1A",
        dca_fingerprint=fingerprint,
        hardware_identities={
            "dca": HardwareIdentity("dca", profile.dca.resource, identity),
            "i2c": HardwareIdentity(
                "i2c",
                f"{profile.i2c.port}/{profile.i2c.baudrate}/"
                f"0x{profile.i2c.device_address:02X}",
                "chip_id=0x52;revision=0x00",
            ),
            "bert": HardwareIdentity(
                "bert",
                f"{profile.bert.ip}:{profile.bert.udp_port}/"
                f"{profile.bert.port}/{profile.bert.channel}",
                "mode=PAM4;patterns=SSPRQ,SSPRQ",
            ),
        },
    )


@dataclass(frozen=True)
class _FieldValue:
    name: str
    channel: int
    address: int
    raw_value: int
    value: int


@dataclass(frozen=True)
class _WriteValue:
    name: str
    channel: int
    address: int
    before_raw: int
    before_value: int
    requested_value: int
    after_raw: int
    after_value: int
    verified: bool = True


class _Writer:
    def __init__(self, b0, *, fail_restore=False):
        self.b0 = dict(b0.i2c_fields)
        self.raw = dict(b0.i2c_raw_registers)
        self.writes = []
        self.fail_restore = fail_restore

    def read_field(self, field, *, channel):
        address = field.address_for_channel(channel)
        raw = self.raw[address]
        return _FieldValue(
            field.name,
            channel,
            address,
            raw,
            (raw >> field.bit_start) & field.encoded_max,
        )

    def write_field(
        self, field, *, channel, value, expected_current, expected_raw
    ):
        before = self.read_field(field, channel=channel)
        if self.fail_restore and self.writes and value == self.b0[field.name]:
            raise RuntimeError("injected I2C restore failure")
        assert before.value == expected_current
        assert before.raw_value == expected_raw
        mask = field.encoded_max << field.bit_start
        after_raw = (before.raw_value & ~mask) | (value << field.bit_start)
        self.raw[before.address] = after_raw
        self.writes.append((field.name, value))
        return _WriteValue(
            field.name,
            channel,
            before.address,
            before.raw_value,
            before.value,
            value,
            after_raw,
            value,
        )


class _Bert:
    def __init__(self, b0, *, fail_restore=False, mode="PAM4"):
        self.b0 = b0
        self.taps = b0.fir_taps
        self.upper_nlc = b0.upper_nlc
        self.lower_nlc = b0.lower_nlc
        self.set_calls = []
        self.fail_restore = fail_restore
        self.mode = mode

    def get_mode(self):
        return self.mode

    def get_configuration(self, port, channel):
        return BertConfiguration(
            port=port,
            channel=str(channel),
            taps=self.taps,
            upper_nlc=self.upper_nlc,
            lower_nlc=self.lower_nlc,
            raw_response="fake",
            pattern_fields=("SSPRQ", "SSPRQ"),
        )

    def set_fir(self, port, channel, *, taps, upper_nlc, lower_nlc):
        self.set_calls.append(tuple(taps))
        if self.fail_restore and tuple(taps) == self.b0.fir_taps:
            raise RuntimeError("injected BERT restore failure")
        self.taps = tuple(taps)
        self.upper_nlc = upper_nlc
        self.lower_nlc = lower_nlc


class _Dca:
    def __init__(self):
        self.contexts = []

    def set_operation_context(self, **context):
        self.contexts.append(context)

    def preflight(self):
        return {"identity_verified": True, "configuration_verified": True}

    def measure_single(self):
        return SimpleNamespace(
            tdecq_db=2.4,
            er_db=4.1,
            tdecq_status="CORR",
            er_status="CORR",
            tdecq_invalid_reason=None,
            acquisition_valid=True,
            elapsed_s=0.01,
            raw_responses={"isolated": True},
            artifacts=(),
        )


class _DatasetDca(_Dca):
    def __init__(self, b0, bert, *, non_b0_tdecq=2.0):
        super().__init__()
        self.b0 = b0
        self.bert = bert
        self.non_b0_tdecq = non_b0_tdecq
        self.acquisition_count = 0

    def measure_single(self):
        self.acquisition_count += 1
        acquisition_id = f"acquisition-{self.acquisition_count:04d}"
        density = (
            np.arange(24, dtype=np.uint32).reshape(4, 6)
            + np.uint32(self.acquisition_count)
        )
        return SimpleNamespace(
            tdecq_db=(
                2.4
                if self.bert.taps == self.b0.fir_taps
                else self.non_b0_tdecq
            ),
            er_db=4.1,
            tdecq_status="CORR",
            er_status="CORR",
            tdecq_invalid_reason=None,
            acquisition_valid=True,
            elapsed_s=0.01,
            raw_responses={
                "isolated": True,
                "tdecq": {
                    "equalizer_taps": "0,-0.03125,-0.0625,0.8125,-0.0625,-0.03125,0",
                    "equalizer_precursors": "3",
                },
            },
            artifacts=(
                FrozenDatasetCapture(
                    acquisition_id=acquisition_id,
                    same_acquisition_verified=True,
                    raw_eye_density=density,
                    density_axis_order=("amplitude", "time"),
                    density_source="CHAN1A",
                    raw_eye_png=encode_density_png(density),
                    tdecq_equalized_eye_png=encode_density_png(
                        np.square(density.astype(np.uint64) + 1)
                    ),
                    rlm=0.97,
                    oma=0.91,
                    average_power=0.50,
                    equalizer_taps=(0.0, -0.03125, -0.0625, 0.8125, -0.0625, -0.03125, 0.0),
                    equalizer_main_index=3,
                    equalizer_source="FUNC1",
                    raw_responses={"isolated": True},
                ),
            ),
        )


def _plant(
    *,
    alternating=False,
    fail_i2c_restore=False,
    fail_bert_restore=False,
    initial_fir=None,
):
    b0 = _b0()
    writer = _Writer(b0, fail_restore=fail_i2c_restore)
    bert = _Bert(b0, fail_restore=fail_bert_restore)
    if initial_fir is not None:
        bert.taps = tuple(initial_fir)
    dca = _Dca()
    bundle = LiveSweepDriverBundle(
        dca=dca,
        i2c_writer=writer,
        i2c_identity=SimpleNamespace(matched=True),
        bert=bert,
    )
    plant_type = (
        LiveAlternatingOptimizationPlant
        if alternating
        else LiveI2cOptimizationPlant
    )
    plant = plant_type(
        profile=_profile(),
        task_id="native-v12-live-ai",
        b0=b0,
        bundle=bundle,
        optimizer_api=load_v11_optimizer_api(),
        approved_recovery_values={
            "ibias_dac": frozenset(range(60, 81)),
            "res_gain": frozenset(range(0, 11)),
        },
    )
    return plant, writer, bert, dca


def test_live_ai_bert_guard_accepts_pam4_device_mode_details():
    plant, _writer, bert, _dca = _plant()
    bert.mode = "PAM4 53.125 1lanePerPort"

    evidence = plant.preflight()

    assert evidence["verified"] is True


def test_factory_construction_is_connection_and_artifact_free(tmp_path):
    calls = []

    def forbidden_drivers(*_args, **_kwargs):
        calls.append("opened")
        raise AssertionError("drivers opened during construction")

    factory = build_live_ai_execution_factory(
        _profile(),
        database_path=tmp_path / "tasks.sqlite3",
        artifact_root=tmp_path / "artifacts",
        driver_factory=forbidden_drivers,
    )

    assert callable(factory)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_live_ai_driver_composition_forwards_cleanup_grace(monkeypatch, tmp_path):
    seen = {}

    @contextmanager
    def fake_open(_profile, plan, *, artifact_root, cleanup_grace_s):
        seen.update(
            artifact_root=artifact_root,
            cleanup_grace_s=cleanup_grace_s,
            save_dataset=plan.save_dataset,
        )
        yield "bundle"

    monkeypatch.setattr(
        "vcsel_ai_tuning.v1_1.live_ai_execution.open_live_sweep_drivers",
        fake_open,
    )

    with open_live_ai_drivers(
        _profile(),
        _b0(),
        artifact_root=tmp_path,
        save_dataset=True,
        cleanup_grace_s=7.5,
    ) as bundle:
        assert bundle == "bundle"

    assert seen == {
        "artifact_root": tmp_path,
        "cleanup_grace_s": 7.5,
        "save_dataset": True,
    }


def _staged_payload():
    return {
        "task_id": "native-v12-staged",
        "task_type": "AI_OPTIMIZATION",
        "live_ai_profile": "PILOT_3_THEN_FAST24",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        "depth": "FAST",
        "candidate_budget": 27,
        "pilot_candidate_budget": 3,
        "fast_candidate_budget": 24,
        "coarse_candidate_budget": 6,
        "fine_candidate_budget": 16,
        "fir_first_candidate_budget": 1,
        "fir_iterative_candidate_budget": 1,
        "settle_time_s": 0.1,
        "save_dataset": True,
        "seed": 17,
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
    }


def _pilot_payload():
    return {
        "task_id": "native-v12-pilot",
        "task_type": "AI_OPTIMIZATION",
        "live_ai_profile": "PILOT_3",
        "optimization_scope": "I2C_ONLY",
        "depth": "STANDARD",
        "candidate_budget": 3,
        "coarse_candidate_budget": 3,
        "fine_candidate_budget": 0,
        "settle_time_s": 0.1,
        "restore_after_each_candidate": True,
        "save_dataset": True,
        "seed": 17,
        "parameters": [
            {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
            {"name": "res_gain", "min": 4, "max": 8, "step": 4},
        ],
    }


def test_staged_factory_returns_inert_combined_execution(tmp_path):
    calls = []

    def forbidden_drivers(*_args, **_kwargs):
        calls.append("opened")
        raise AssertionError("drivers opened during staged preparation")

    factory = build_live_ai_execution_factory(
        _profile(),
        database_path=tmp_path / "tasks.sqlite3",
        artifact_root=tmp_path / "artifacts",
        driver_factory=forbidden_drivers,
        measurement_timeout_s=60.0,
        cleanup_grace_s=5.0,
        live_ai_profile=LiveAIProfile.PILOT_3_THEN_FAST24,
    )

    execution = factory("native-v12-staged", _b0(), _staged_payload())

    assert isinstance(execution, LiveStagedAIExecution)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_standalone_pilot_persists_three_complete_datasets_and_restores_b0(
    tmp_path,
):
    b0 = _b0()
    payload = _pilot_payload()
    database = tmp_path / "tasks.sqlite3"
    artifact_root = tmp_path / "artifacts"
    with TaskStore(database) as store:
        store.create_task(
            task_id="native-v12-pilot",
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.LIVE,
            config_fingerprint="c" * 64,
            payload=payload,
        )
        store.save_b0("native-v12-pilot", b0)

    writer = _Writer(b0)
    bert = _Bert(b0)
    dca = _DatasetDca(b0, bert)
    bundle = LiveSweepDriverBundle(
        dca=dca,
        i2c_writer=writer,
        i2c_identity=SimpleNamespace(matched=True),
        bert=bert,
    )

    @contextmanager
    def drivers(*_args, **_kwargs):
        yield bundle

    factory = build_live_ai_execution_factory(
        _profile(),
        database_path=database,
        artifact_root=artifact_root,
        driver_factory=drivers,
        live_ai_profile=LiveAIProfile.PILOT_3,
    )

    execution = factory("native-v12-pilot", b0, payload)
    terminal = execution.run(Event())

    assert terminal in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }
    assert execution.terminal_b0_verified is True
    with TaskStore(database) as store:
        rows = store.observations("native-v12-pilot")
    assert len(rows) == 3
    assert all(row.phase == "PILOT_3" for row in rows)
    assert all(row.raw_response.get("dataset_status") == "COMPLETE" for row in rows)
    assert len(list(artifact_root.rglob("record.json"))) == 3
    assert bert.set_calls == []
    assert writer.read_field(
        _profile().characterization.parameters[0],
        channel=b0.i2c_channel,
    ).value == dict(b0.i2c_fields)["ibias_dac"]


def test_staged_live_execution_runs_exact_3_plus_24_with_real_capture_contract(
    tmp_path,
):
    b0 = _b0()
    payload = _staged_payload()
    database = tmp_path / "tasks.sqlite3"
    artifact_root = tmp_path / "artifacts"
    with TaskStore(database) as store:
        store.create_task(
            task_id="native-v12-staged",
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.LIVE,
            config_fingerprint="c" * 64,
            payload=payload,
        )
        store.save_b0("native-v12-staged", b0)

    writer = _Writer(b0)
    bert = _Bert(b0)
    dca = _DatasetDca(b0, bert)
    bundle = LiveSweepDriverBundle(
        dca=dca,
        i2c_writer=writer,
        i2c_identity=SimpleNamespace(matched=True),
        bert=bert,
    )

    @contextmanager
    def drivers(*_args, **_kwargs):
        yield bundle

    factory = build_live_ai_execution_factory(
        _profile(),
        database_path=database,
        artifact_root=artifact_root,
        driver_factory=drivers,
        live_ai_profile=LiveAIProfile.PILOT_3_THEN_FAST24,
    )
    execution = factory("native-v12-staged", b0, payload)

    terminal = execution.run(Event())

    assert terminal in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }
    with TaskStore(database) as store:
        rows = store.observations("native-v12-staged")
        events = store.events("native-v12-staged")
    candidate_phases = tuple(
        row.phase
        for row in rows
        if row.phase not in {"B0", "FINAL_VERIFICATION"}
    )
    assert candidate_phases == (
        ("PILOT_3",) * 3
        + ("I2C_COARSE",) * 6
        + ("I2C_FINE",) * 16
        + ("FIR_FIRST", "FIR_ITERATIVE")
    )
    assert all(
        row.raw_response.get("dataset_status") == "COMPLETE"
        for row in rows[:3]
    )
    assert all(
        isinstance(row.raw_response.get("dataset"), dict)
        and row.raw_response["dataset"].get("dataset_status") == "COMPLETE"
        for row in rows[3:]
    )
    assert any(
        event.event_type == "PILOT_SAFETY_GATE_PASSED" for event in events
    )
    assert writer.read_field(
        _profile().characterization.parameters[0],
        channel=b0.i2c_channel,
    ).value == dict(b0.i2c_fields)[
        _profile().characterization.parameters[0].name
    ]
    assert bert.taps == b0.fir_taps
    assert len(list(artifact_root.rglob("record.json"))) == len(rows)


def test_staged_live_execution_keeps_exact_schedule_when_first_fir_is_worse(
    tmp_path,
):
    b0 = _b0()
    payload = _staged_payload()
    database = tmp_path / "tasks.sqlite3"
    artifact_root = tmp_path / "artifacts"
    with TaskStore(database) as store:
        store.create_task(
            task_id="native-v12-staged-first-fir-miss",
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.LIVE,
            config_fingerprint="c" * 64,
            payload=payload,
        )
        store.save_b0("native-v12-staged-first-fir-miss", b0)

    writer = _Writer(b0)
    bert = _Bert(b0)
    dca = _DatasetDca(b0, bert, non_b0_tdecq=3.0)
    bundle = LiveSweepDriverBundle(
        dca=dca,
        i2c_writer=writer,
        i2c_identity=SimpleNamespace(matched=True),
        bert=bert,
    )

    @contextmanager
    def drivers(*_args, **_kwargs):
        yield bundle

    factory = build_live_ai_execution_factory(
        _profile(),
        database_path=database,
        artifact_root=artifact_root,
        driver_factory=drivers,
        live_ai_profile=LiveAIProfile.PILOT_3_THEN_FAST24,
    )

    terminal = factory(
        "native-v12-staged-first-fir-miss",
        b0,
        payload,
    ).run(Event())

    assert terminal is LifecycleState.COMPLETED_NO_IMPROVEMENT
    with TaskStore(database) as store:
        rows = store.observations("native-v12-staged-first-fir-miss")
        events = store.events("native-v12-staged-first-fir-miss")
    assert tuple(
        row.phase
        for row in rows
        if row.phase not in {"B0", "FINAL_VERIFICATION"}
    ) == (
        ("PILOT_3",) * 3
        + ("I2C_COARSE",) * 6
        + ("I2C_FINE",) * 16
        + ("FIR_FIRST", "FIR_ITERATIVE")
    )
    fir_rows = tuple(row for row in rows if row.phase.startswith("FIR_"))
    assert [row.raw_response["controller_status"] for row in fir_rows] == [
        "REJECTED_FIRST_PASS",
        "REJECTED_ITERATIVE",
    ]
    assert not any(
        event.event_type == "AI_FIR_INVALID_DCA_TAPS" for event in events
    )
    assert bert.taps == b0.fir_taps


def test_staged_live_execution_fails_closed_on_fast_dataset_publish_failure(
    tmp_path,
    monkeypatch,
):
    b0 = _b0()
    payload = _staged_payload()
    database = tmp_path / "tasks.sqlite3"
    artifact_root = tmp_path / "artifacts"
    with TaskStore(database) as store:
        store.create_task(
            task_id="native-v12-staged-dataset-failure",
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.LIVE,
            config_fingerprint="c" * 64,
            payload=payload,
        )
        store.save_b0("native-v12-staged-dataset-failure", b0)

    writer = _Writer(b0)
    bert = _Bert(b0)
    dca = _DatasetDca(b0, bert)
    bundle = LiveSweepDriverBundle(
        dca=dca,
        i2c_writer=writer,
        i2c_identity=SimpleNamespace(matched=True),
        bert=bert,
    )
    real_record = MeasurementArtifactBundleWriter.record

    def fail_first_fast_bundle(self, context, capture):
        if context.sweep_point_id == "iteration-00000004":
            return DatasetOutcome(
                status=DatasetStatus.FAILED,
                error="injected publish failure",
            )
        return real_record(self, context, capture)

    monkeypatch.setattr(MeasurementArtifactBundleWriter, "record", fail_first_fast_bundle)

    @contextmanager
    def drivers(*_args, **_kwargs):
        yield bundle

    factory = build_live_ai_execution_factory(
        _profile(),
        database_path=database,
        artifact_root=artifact_root,
        driver_factory=drivers,
        live_ai_profile=LiveAIProfile.PILOT_3_THEN_FAST24,
    )

    with pytest.raises(LiveAIExecutionError) as captured:
        factory(
            "native-v12-staged-dataset-failure",
            b0,
            payload,
        ).run(Event())

    fast_evidence = captured.value.evidence["primary_evidence"]
    dataset_evidence = fast_evidence["primary_evidence"]
    assert dataset_evidence["sequence"] == 4
    assert dataset_evidence["dataset"]["dataset_status"] == "FAILED"
    assert "injected publish failure" in dataset_evidence["dataset"][
        "dataset_error"
    ]
    assert captured.value.rollback_verified is True
    with TaskStore(database) as store:
        rows = store.observations("native-v12-staged-dataset-failure")
    assert [row.phase for row in rows] == ["PILOT_3"] * 3
    assert bert.taps == b0.fir_taps


def test_pilot_i2c_apply_requires_exact_direct_readback_without_bert_write():
    plant, writer, bert, _dca = _plant()

    receipt = plant.apply_i2c({"ibias_dac": 71, "res_gain": 8})

    assert receipt["requested"] == receipt["applied"] == receipt["readback"]
    assert writer.writes == [("ibias_dac", 71), ("res_gain", 8)]
    assert bert.set_calls == []


def test_fast24_accepts_formula_non_main_taps_beyond_b0_plus_minus_three():
    plant, _writer, bert, _dca = _plant(alternating=True)
    target = FirTarget(
        taps=(-8, 6, -18, 104, -12, 9, 3),
        upper_nlc=plant.b0.upper_nlc,
        lower_nlc=plant.b0.lower_nlc,
    )

    assert plant.apply_fir(target) == target.taps
    assert bert.set_calls == [target.taps]


def test_fast24_can_retune_i2c_while_formula_fir_remains_active():
    plant, writer, _bert, _dca = _plant(alternating=True)
    target = FirTarget(
        taps=(-8, 6, -18, 104, -12, 9, 3),
        upper_nlc=6,
        lower_nlc=6,
    )
    plant.apply_fir(target)

    receipt = plant.apply_i2c({"ibias_dac": 71, "res_gain": 8})

    assert receipt["readback"] == {"ibias_dac": 71, "res_gain": 8}
    assert writer.writes == [("ibias_dac", 71), ("res_gain", 8)]


def test_fast24_rejects_changed_main_without_setfir():
    plant, _writer, bert, _dca = _plant(alternating=True)
    changed_main = FirTarget(
        taps=(0, 0, 0, 103, 0, 0, 0),
        upper_nlc=plant.b0.upper_nlc,
        lower_nlc=plant.b0.lower_nlc,
    )

    with pytest.raises(ValueError, match="Main"):
        plant.apply_fir(changed_main)
    assert bert.set_calls == []


def test_fast24_rejects_formula_l1_above_170_before_setfir():
    _plant_instance, _writer, bert, _dca = _plant(alternating=True)

    with pytest.raises(ValueError, match="L1"):
        FirTarget(
            taps=(40, 40, 40, 104, 40, 40, 40),
            upper_nlc=6,
            lower_nlc=6,
        )
    assert bert.set_calls == []


def test_measurement_uses_isolated_dca_and_returns_optimizer_metric():
    plant, _writer, _bert, dca = _plant()
    plant.set_measurement_context("candidate-1", 1)

    result = plant.measure()

    assert isinstance(result, MetricResult)
    assert result.tdecq_db == 2.4
    assert dca.contexts[-1]["item_id"] == "candidate-1"


def test_terminal_restore_verifies_i2c_and_bert():
    plant, _writer, bert, _dca = _plant(alternating=True)
    plant.apply_i2c({"ibias_dac": 71, "res_gain": 8})
    plant.apply_fir(
        FirTarget(
            taps=(-8, 6, -18, 104, -12, 9, 3),
            upper_nlc=6,
            lower_nlc=6,
        )
    )

    restored = plant.restore_b0()

    assert restored == dict(plant.b0.i2c_fields)
    assert bert.taps == plant.b0.fir_taps
    assert plant.is_at_b0() is True


def test_failed_restore_raises_recovery_required_without_retry():
    plant, writer, _bert, _dca = _plant(fail_i2c_restore=True)
    plant.apply_i2c({"ibias_dac": 71, "res_gain": 8})

    with pytest.raises(RecoveryRequired):
        plant.restore_b0()

    assert writer.writes.count(("ibias_dac", 70)) == 0


def test_explicit_recovery_uses_frozen_b0_not_constructor_readback():
    drifted = (-8, 6, -18, 104, -12, 9, 3)
    plant, _writer, bert, _dca = _plant(
        alternating=True, initial_fir=drifted
    )
    bert.upper_nlc = 0
    bert.lower_nlc = 0

    plant.restore_b0()

    assert bert.taps == plant.b0.fir_taps
    assert (bert.upper_nlc, bert.lower_nlc) == (
        plant.b0.upper_nlc,
        plant.b0.lower_nlc,
    )
    assert bert.set_calls == [plant.b0.fir_taps]
