from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.live_dca_isolation import DcaIsolationTimeout
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.hardware_control import (
    BertCandidateTransaction,
    I2cCandidateTransaction,
    RecoveryRequired,
)
from vcsel_ai_tuning.v1_1.dataset import (
    FrozenDatasetCapture,
    MeasurementArtifactBundleWriter,
    encode_density_png,
)
from vcsel_ai_tuning.v1_1 import live_sweep as live_sweep_module
from vcsel_ai_tuning.v1_1.live_dataset_capture import FlexDcaDatasetCapture
from vcsel_ai_tuning.v1_1.live_sweep import (
    LiveHardwareSweepExecutor,
    LiveSweepDriverBundle,
    LiveSweepExecutionError,
    build_live_sweep_executor_factory,
)
from tests_py.ai_tuning.v1_1.test_live_dataset_capture import (
    FakeBinarySession,
    acquisition_guard,
)
from vcsel_ai_tuning.v1_1.regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from vcsel_ai_tuning.v1_1.sweep import (
    FIR_TAP_NAMES,
    SweepHardware,
    SweepMode,
    SweepPlan,
    SweepPoint,
)


def _b0() -> B0Snapshot:
    fingerprint = DcaFingerprint.from_mappings(
        identity="Keysight,N1010A,lab,A.08",
        measurement_sources={"outer_er": "CHAN1A", "tdecq": "FUNC1"},
        configuration={"function": "1", "tdecq_equalizer_function": "FUNC1"},
    )
    return B0Snapshot.capture(
        b0_id="live-b0",
        version=1,
        captured_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
        i2c_fields={"eq": 7, "vref": 0},
        i2c_raw_registers={0x2006: 7, 0x200B: 0xC0},
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
        dca_fingerprint=fingerprint,
        hardware_identities={
            "dca": HardwareIdentity("dca", "TCPIP::DCA", fingerprint.identity),
            "i2c": HardwareIdentity("i2c", "COM7/115200/0xA6", "chip_id=0x52;revision=0x00"),
            "bert": HardwareIdentity("bert", "192.0.2.1:9000/port1/0", "mode=PAM4;patterns=SSPRQ,SSPRQ"),
        },
    )


def _mapping():
    return scan_parameter_mapping(
        (ScriptEntry("eq", 7, 0), ScriptEntry("vref", 0, 1)),
        (
            RegMapField("eq", 0x2006, 0x100, 0, 5, 0, 31),
            RegMapField("vref", 0x200B, 0x100, 2, 4, 0, 15),
        ),
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


class _FieldWriter:
    def __init__(self, *, fail_restore: bool = False):
        self.raw = {0x2006: 7, 0x200B: 0xC0}
        self.writes: list[tuple[str, int]] = []
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
        self,
        field,
        *,
        channel,
        value,
        expected_current,
        expected_raw,
    ):
        before = self.read_field(field, channel=channel)
        if self.fail_restore and value == 7 and field.name == "eq":
            raise RuntimeError("injected restore failure")
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


class _Dca:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    def preflight(self):
        self.calls.append("preflight")
        return {"identity_verified": True, "configuration_verified": True}

    def measure_single(self):
        self.calls.append("measure_single")
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _measurement(*, tdecq=2.5, er=3.1, tdecq_status="CORR", er_status="CORR"):
    return SimpleNamespace(
        acquisition_valid=tdecq_status == "CORR" or er_status == "CORR",
        er_db=er,
        er_status=er_status,
        tdecq_db=tdecq,
        tdecq_status=tdecq_status,
        tdecq_invalid_reason=None if tdecq_status == "CORR" else "No Data",
        raw_responses={
            "measurements": {
                "outer_er": {"status": er_status, "value": er},
                "tdecq": {"status": tdecq_status, "value": tdecq},
            },
            "scpi": [":ACQuire:STOP", ":ACQuire:CDISplay", ":ACQuire:SINGle"],
        },
    )


def _i2c_executor(
    *,
    dca=None,
    writer=None,
    outcomes=None,
    waits=None,
    save_dataset=False,
    artifact_root=None,
    dataset_capture=None,
    dca_capture_scope=None,
    id_factory=lambda: "unused-id",
):
    b0 = _b0()
    writer = writer or _FieldWriter()
    transaction = I2cCandidateTransaction(
        writer=writer,
        mapping=_mapping(),
        channel=0,
        b0_values=dict(b0.i2c_fields),
        b0_raw_registers=dict(b0.i2c_raw_registers),
    )
    plan = SweepPlan(
        hardware=SweepHardware.I2C_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        seed=11,
        max_points=1,
        points=(SweepPoint(0, 0, {"eq": 8, "vref": 0}),),
        campaign_id="live-dataset-campaign",
        save_dataset=save_dataset,
    )
    dca = dca or _Dca(outcomes or [_measurement(), _measurement()])
    wait_calls = [] if waits is None else waits
    executor_kwargs = {}
    if dca_capture_scope is not None:
        executor_kwargs["dca_capture_scope"] = dca_capture_scope
    executor = LiveHardwareSweepExecutor(
        b0=b0,
        plan=plan,
        i2c_transaction=transaction,
        dca=dca,
        settle_time_s=2.0,
        wait=wait_calls.append,
        dataset_recorder=(
            MeasurementArtifactBundleWriter(artifact_root)
            if save_dataset
            else None
        ),
        dataset_capture=dataset_capture,
        id_factory=id_factory,
        **executor_kwargs,
    )
    return executor, plan, writer, dca, wait_calls


class _RecordingIsolatedDca:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.actions = []
        self.contexts = []

    def set_operation_context(self, *, task_id, item_id, round_index):
        self.contexts.append((task_id, item_id, round_index))

    def preflight(self):
        self.actions.append("preflight")
        return {"identity_verified": True, "configuration_verified": True}

    def measure_single(self):
        self.actions.append("measure_single")
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _isolated_dataset_measurement():
    result = _measurement()
    density = np.arange(24, dtype=np.uint16).reshape(4, 6)
    result.artifacts = (
        FrozenDatasetCapture(
            acquisition_id="worker-acquisition",
            same_acquisition_verified=True,
            raw_eye_density=density,
            density_axis_order=("amplitude", "time"),
            density_source="CHAN1A",
            tdecq_equalized_eye_png=encode_density_png(np.square(density + 1)),
            rlm=0.96,
            oma=0.90,
            average_power=0.51,
            equalizer_taps=_b0().fir_taps,
            equalizer_main_index=3,
            equalizer_source="FUNC1",
            raw_responses={"worker": "isolated"},
        ),
    )
    return result


def test_live_sweep_uses_isolated_dca_and_worker_dataset(tmp_path):
    isolated = _RecordingIsolatedDca([_isolated_dataset_measurement()])
    executor, plan, _writer, _dca, _waits = _i2c_executor(
        dca=isolated,
        save_dataset=True,
        artifact_root=tmp_path,
        id_factory=lambda: "measurement-1",
    )

    executor.preflight()
    executor.apply_point(plan.points[0])
    measurement = executor.measure_point()

    assert isolated.actions == ["preflight", "measure_single"]
    assert isolated.contexts == [
        (plan.campaign_id, "B0", 0),
        (plan.campaign_id, "point-00000000", 0),
    ]
    assert measurement["acquisition_id"] == "worker-acquisition"
    assert measurement["dataset_status"] == "COMPLETE"
    assert executor.audit_evidence()["final_b0_readback"] is None


def test_live_sweep_preserves_watchdog_evidence_and_restores_once():
    timeout = DcaIsolationTimeout(
        "deadline",
        evidence={
            "worker_alive_after_cleanup": False,
            "last_in_flight": {"operation": "query"},
            "tdecq_db": None,
            "er_db": None,
        },
    )
    isolated = _RecordingIsolatedDca([timeout])
    executor, plan, writer, _dca, _waits = _i2c_executor(dca=isolated)
    executor.preflight()
    executor.apply_point(plan.points[0])

    with pytest.raises(LiveSweepExecutionError) as caught:
        executor.measure_point()
    executor.restore_b0()

    assert caught.value.evidence["watchdog"]["worker_alive_after_cleanup"] is False
    assert writer.writes == [("eq", 8), ("eq", 7)]


def test_exclusive_dca_access_is_reused_per_normalized_resource():
    first = live_sweep_module._exclusive_dca_access_for_resource(
        " TCPIP0::DCA::INSTR "
    )
    equivalent = live_sweep_module._exclusive_dca_access_for_resource(
        "tcpip0::dca::instr"
    )
    different = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::OTHER::INSTR"
    )

    assert first is equivalent
    assert first is not different


def test_exclusive_dca_access_tracks_nested_thread_ownership_and_exception_release():
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::ownership-nesting-test::INSTR"
    )
    assert access.owned_by_current_thread() is False

    with pytest.raises(RuntimeError, match="injected claim failure"):
        with access.claim():
            assert access.owned_by_current_thread() is True
            with access.claim():
                assert access.owned_by_current_thread() is True
            assert access.owned_by_current_thread() is True
            raise RuntimeError("injected claim failure")

    assert access.owned_by_current_thread() is False


def test_exclusive_dca_access_never_reports_another_thread_as_owner():
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::cross-thread-ownership-test::INSTR"
    )
    owner_entered = threading.Event()
    release_owner = threading.Event()
    observations = []

    def own_access():
        with access.claim():
            owner_entered.set()
            assert release_owner.wait(timeout=2.0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(own_access)
        assert owner_entered.wait(timeout=2.0)
        observations.append(access.owned_by_current_thread())
        release_owner.set()
        owner.result(timeout=2.0)

    assert observations == [False]


def test_same_resource_preflight_waits_for_active_measurement_capture(tmp_path):
    measurement_access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::preflight-blocking-test::INSTR"
    )
    preflight_access = live_sweep_module._exclusive_dca_access_for_resource(
        "tcpip0::preflight-blocking-test::instr"
    )
    assert measurement_access is preflight_access
    capture_entered = threading.Event()
    release_capture = threading.Event()
    preflight_scope_attempted = threading.Event()
    preflight_dca_entered = threading.Event()

    def hold_capture(acquisition_id, _result):
        capture_entered.set()
        if not release_capture.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to release dataset capture")
        density = np.arange(24, dtype=np.uint16).reshape(4, 6)
        return FrozenDatasetCapture(
            acquisition_id=acquisition_id,
            raw_eye_density=density,
            density_axis_order=("amplitude", "time"),
            density_source="AUDITED_REPLAY_CHAN1A",
            tdecq_equalized_eye_png=encode_density_png(
                np.square(density + 1)
            ),
        )

    @contextmanager
    def observed_preflight_scope():
        preflight_scope_attempted.set()
        with preflight_access.claim():
            yield

    class ObservedPreflightDca(_Dca):
        def preflight(self):
            preflight_dca_entered.set()
            return super().preflight()

    measuring, measure_plan, _writer, _dca, _waits = _i2c_executor(
        save_dataset=True,
        artifact_root=tmp_path,
        dataset_capture=hold_capture,
        dca_capture_scope=measurement_access.claim,
        id_factory=iter(("measurement-blocking", "acquisition-blocking")).__next__,
    )
    preflighting, _plan, _writer, _dca, _waits = _i2c_executor(
        dca_capture_scope=observed_preflight_scope,
    )
    preflighting._dca = ObservedPreflightDca([])
    measuring.preflight()
    measuring.apply_point(measure_plan.points[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        measurement = pool.submit(measuring.measure_point)
        assert capture_entered.wait(timeout=2.0)
        preflight = pool.submit(preflighting.preflight)
        assert preflight_scope_attempted.wait(timeout=2.0)
        assert preflight_dca_entered.is_set() is False
        release_capture.set()
        measurement.result(timeout=2.0)
        preflight.result(timeout=2.0)

    assert preflight_dca_entered.is_set() is True


def test_dca_preflight_scope_excludes_additional_hardware_preflight():
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::preflight-boundary-test::INSTR"
    )
    observations = []

    class OwnershipObservedPreflightDca(_Dca):
        def preflight(self):
            observations.append(
                ("dca", access.owned_by_current_thread())
            )
            return super().preflight()

    executor, _plan, _writer, _dca, _waits = _i2c_executor(
        dca_capture_scope=access.claim,
    )
    executor._dca = OwnershipObservedPreflightDca([])
    executor._additional_preflight = lambda: (
        observations.append(
            ("additional", access.owned_by_current_thread())
        )
        or {"identity_verified": True}
    )

    executor.preflight()

    assert observations == [("dca", True), ("additional", False)]


def test_dca_preflight_exception_releases_exclusive_claim():
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::preflight-failure-test::INSTR"
    )

    class FailingPreflightDca(_Dca):
        def preflight(self):
            raise RuntimeError("injected DCA preflight failure")

    executor, _plan, _writer, _dca, _waits = _i2c_executor(
        dca_capture_scope=access.claim,
    )
    executor._dca = FailingPreflightDca([])

    with pytest.raises(RuntimeError, match="injected DCA preflight failure"):
        executor.preflight()

    assert access.owned_by_current_thread() is False


def test_different_resource_preflight_scope_does_not_block():
    first = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::preflight-resource-a::INSTR"
    )
    second = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::preflight-resource-b::INSTR"
    )

    def claim_second_resource():
        with second.claim():
            return second.owned_by_current_thread()

    with first.claim(), ThreadPoolExecutor(max_workers=1) as pool:
        ownership = pool.submit(claim_second_resource)
        assert ownership.result(timeout=2.0) is True


def test_open_live_sweep_drivers_constructs_connection_free_isolated_dca(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    profile = load_live_hardware_config(
        root / "configs" / "hardware" / "vcsel_lab_live.yaml"
    )
    _executor, base_plan, _writer, _dca, _waits = _i2c_executor()
    plan = replace(base_plan, save_dataset=True)
    observed = {}

    class Managed:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class Probe(Managed):
        def probe(self):
            return SimpleNamespace(matched=True)

        def close(self):
            pass

    class Isolated:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.flexdca.open_visa_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("controller must not open DCA VISA")
        ),
    )
    monkeypatch.setattr("vcsel_ai_tuning.hardware.vd1124.Vd1124Probe", Probe)
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.vd1124.Vd1124RegisterWriter", Managed
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.vit_bert.VitBertClient", Managed
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_dca_isolation.IsolatedLiveDcaClient",
        Isolated,
    )

    with live_sweep_module.open_live_sweep_drivers(profile, plan) as bundle:
        assert isinstance(bundle.dca, Isolated)
        assert bundle.dataset_capture is None
        assert observed["profile"] is profile
        assert observed["capture_dataset"] is True
        assert observed["timeout_s"] == pytest.approx(
            profile.characterization.measurement_timeout_s
        )
        assert bundle.dca_exclusive_verifier() is False
        with bundle.dca_capture_scope():
            assert bundle.dca_exclusive_verifier() is True
        assert bundle.dca_exclusive_verifier() is False


def test_i2c_point_uses_candidate_transaction_waits_measures_and_restores_b0():
    executor, plan, writer, dca, waits = _i2c_executor()

    executor.preflight()
    assert executor.is_at_b0()
    application = executor.apply_point(plan.points[0])
    measurement = executor.measure_point()

    assert application["requested"] == {"eq": 8, "vref": 0}
    assert application["readback"] == {"eq": 8, "vref": 0}
    assert application["write_readback_duration_s"] >= 0
    assert writer.writes == [("eq", 8)]
    assert waits == [2.0]
    assert measurement["tdecq"] == 2.5
    assert measurement["er"] == 3.1
    assert measurement["settle_duration_s"] >= 0
    assert measurement["dca_duration_s"] >= 0
    assert measurement["raw"]["point_id"] == "point-00000000"
    assert measurement["raw"]["execution_order"] == 0
    assert dca.calls == ["preflight", "measure_single"]

    executor.restore_b0()
    assert executor.is_at_b0()
    assert writer.writes == [("eq", 8), ("eq", 7)]
    assert executor.audit_evidence()["final_b0_readback"] == {"eq": 7, "vref": 0}


def test_live_dataset_disabled_does_not_call_optional_capture(tmp_path):
    capture_calls = []
    executor, plan, _writer, _dca, _waits = _i2c_executor(
        artifact_root=tmp_path,
        dataset_capture=lambda *_args: capture_calls.append("capture"),
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["dataset_status"] == "DISABLED"
    assert capture_calls == []
    assert not (tmp_path / plan.campaign_id).exists()


def test_live_dataset_uses_explicit_same_acquisition_capture_provider(tmp_path):
    issued_ids = iter(("measurement-live-1", "acquisition-live-1"))
    capture_calls = []

    def capture(acquisition_id, result):
        capture_calls.append((acquisition_id, result))
        density = np.arange(24, dtype=np.uint16).reshape(4, 6)
        return FrozenDatasetCapture(
            acquisition_id=acquisition_id,
            same_acquisition_verified=True,
            raw_eye_density=density,
            density_axis_order=("amplitude", "time"),
            density_source="AUDITED_REPLAY_CHAN1A",
            tdecq_equalized_eye_png=encode_density_png(
                np.square(density + 1)
            ),
            rlm=0.96,
            oma=0.90,
            average_power=0.51,
            equalizer_taps=_b0().fir_taps,
            equalizer_main_index=3,
            equalizer_source="FUNC1",
            raw_responses={"provider": "audited-test"},
        )

    executor, plan, _writer, _dca, _waits = _i2c_executor(
        save_dataset=True,
        artifact_root=tmp_path,
        dataset_capture=capture,
        id_factory=lambda: next(issued_ids),
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["measurement_id"] == "measurement-live-1"
    assert measurement["acquisition_id"] == "acquisition-live-1"
    assert measurement["dataset_status"] == "COMPLETE"
    assert capture_calls[0][0] == measurement["acquisition_id"]
    bundle = tmp_path / plan.campaign_id / "point-00000000"
    assert np.array_equal(
        np.load(bundle / "raw_eye_density.npy", allow_pickle=False),
        np.arange(24, dtype=np.uint16).reshape(4, 6),
    )


def test_real_capture_and_resource_scope_publish_complete_dataset(tmp_path):
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::real-capture-complete-test::INSTR"
    )
    session = FakeBinarySession()
    session.density["FUNC1"] = np.array(
        [[12, 7, 9], [8, 11, 10]],
        dtype=np.uint32,
    )
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=access.owned_by_current_thread,
    )
    primary = _measurement()
    primary.raw_responses["acquisition_guard"] = acquisition_guard()
    executor, plan, _writer, _dca, _waits = _i2c_executor(
        outcomes=[primary],
        save_dataset=True,
        artifact_root=tmp_path,
        dataset_capture=adapter.capture,
        dca_capture_scope=access.claim,
        id_factory=iter(("measurement-real", "acquisition-real")).__next__,
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["dataset_status"] == "COMPLETE"
    bundle = tmp_path / plan.campaign_id / "point-00000000"
    assert (bundle / "raw_eye_density.npy").is_file()
    assert (bundle / "raw_eye.png").read_bytes() != (
        bundle / "tdecq_equalized_eye.png"
    ).read_bytes()
    assert access.owned_by_current_thread() is False
    assert not any("ACQuire" in command for command in session.writes)


def test_live_dataset_capture_stays_inside_measurement_exclusive_claim(tmp_path):
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::executor-success-test::INSTR"
    )
    observations = []

    class OwnershipObservedMeasurement:
        acquisition_valid = True
        er_db = 3.1
        er_status = "CORR"
        tdecq_db = 2.5
        tdecq_status = "CORR"
        tdecq_invalid_reason = None

        @property
        def raw_responses(self):
            observations.append(("payload", access.owned_by_current_thread()))
            return _measurement().raw_responses

    class OwnershipObservedDca(_Dca):
        def measure_single(self):
            observations.append(
                ("measure_single", access.owned_by_current_thread())
            )
            return OwnershipObservedMeasurement()

    def capture(acquisition_id, _result):
        observations.append(("capture", access.owned_by_current_thread()))
        density = np.arange(24, dtype=np.uint16).reshape(4, 6)
        return FrozenDatasetCapture(
            acquisition_id=acquisition_id,
            raw_eye_density=density,
            density_axis_order=("amplitude", "time"),
            density_source="AUDITED_REPLAY_CHAN1A",
            tdecq_equalized_eye_png=encode_density_png(
                np.square(density + 1)
            ),
            rlm=0.96,
            oma=0.90,
            average_power=0.51,
            equalizer_taps=_b0().fir_taps,
            equalizer_main_index=3,
            equalizer_source="FUNC1",
            raw_responses={"provider": "exclusive-test"},
        )

    executor, plan, _writer, _unused_dca, _waits = _i2c_executor(
        save_dataset=True,
        artifact_root=tmp_path,
        dataset_capture=capture,
        dca_capture_scope=access.claim,
        id_factory=iter(("measurement-exclusive", "acquisition-exclusive")).__next__,
    )
    executor._dca = OwnershipObservedDca([])
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["dataset_status"] == "PARTIAL"
    assert observations == [
        ("measure_single", True),
        ("payload", True),
        ("capture", True),
    ]
    assert access.owned_by_current_thread() is False


def test_live_dca_exception_releases_exclusive_claim():
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::executor-dca-failure-test::INSTR"
    )
    executor, plan, _writer, _dca, _waits = _i2c_executor(
        outcomes=[RuntimeError("injected DCA failure")],
        dca_capture_scope=access.claim,
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    with pytest.raises(LiveSweepExecutionError, match="DCA measurement failed"):
        executor.measure_point()

    assert access.owned_by_current_thread() is False


def test_live_dataset_exception_releases_exclusive_claim(tmp_path):
    access = live_sweep_module._exclusive_dca_access_for_resource(
        "TCPIP0::executor-dataset-failure-test::INSTR"
    )

    def fail_capture(_acquisition_id, _result):
        assert access.owned_by_current_thread() is True
        raise RuntimeError("injected dataset failure")

    executor, plan, _writer, _dca, _waits = _i2c_executor(
        save_dataset=True,
        artifact_root=tmp_path,
        dataset_capture=fail_capture,
        dca_capture_scope=access.claim,
        id_factory=iter(("measurement-failure", "acquisition-failure")).__next__,
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["dataset_status"] == "FAILED"
    assert "injected dataset failure" in measurement["dataset_error"]
    assert access.owned_by_current_thread() is False


def test_live_missing_dataset_provider_does_not_fail_primary_measurement(tmp_path):
    ids = iter(("measurement-live-2", "acquisition-live-2"))
    executor, plan, writer, _dca, _waits = _i2c_executor(
        save_dataset=True,
        artifact_root=tmp_path,
        id_factory=lambda: next(ids),
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["tdecq"] == 2.5
    assert measurement["measurement_status"] == "SUCCESS"
    assert measurement["dataset_status"] == "FAILED"
    assert "provider is unavailable" in measurement["dataset_error"]
    assert writer.writes == [("eq", 8)]


def test_point_write_readback_duration_covers_the_full_apply_boundary():
    executor, plan, _writer, _dca, _waits = _i2c_executor()
    clock_values = iter((10.0, 10.7))
    executor._monotonic = lambda: next(clock_values)

    executor.preflight()
    application = executor.apply_point(plan.points[0])

    assert application["write_readback_duration_s"] == pytest.approx(0.7)


def test_failed_b0_verification_retains_the_actual_mismatched_readback():
    executor, plan, _writer, _dca, _waits = _i2c_executor()
    executor.preflight()
    executor.apply_point(plan.points[0])

    assert executor.is_at_b0() is False
    assert executor.audit_evidence()["final_b0_readback"] == {
        "eq": 8,
        "vref": 0,
    }


def test_invalid_tdecq_is_returned_for_persistence_without_becoming_valid():
    executor, plan, _writer, _dca, _waits = _i2c_executor(
        outcomes=[_measurement(tdecq=None, tdecq_status="INV")]
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    measurement = executor.measure_point()

    assert measurement["tdecq"] is None
    assert measurement["er"] == 3.1
    assert measurement["raw"]["tdecq_status"] == "INV"
    assert measurement["raw"]["invalid_reason"] == "No Data"


def test_source_drift_stops_the_sweep_with_auditable_evidence():
    executor, plan, _writer, _dca, _waits = _i2c_executor(
        outcomes=[_measurement(tdecq=None, tdecq_status="SOURCE_MISMATCH")]
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    with pytest.raises(LiveSweepExecutionError, match="source") as caught:
        executor.measure_point()

    assert caught.value.evidence["point_id"] == "point-00000000"
    assert caught.value.evidence["tdecq_status"] == "SOURCE_MISMATCH"


def test_restore_failure_enters_recovery_required_and_never_retries():
    writer = _FieldWriter(fail_restore=True)
    executor, plan, _writer, _dca, _waits = _i2c_executor(writer=writer)
    executor.preflight()
    executor.apply_point(plan.points[0])

    with pytest.raises(RecoveryRequired) as caught:
        executor.restore_b0()

    assert writer.writes == [("eq", 8)]
    assert caught.value.evidence["last_observed_readback"] == {
        "eq": 8,
        "vref": 0,
    }
    assert executor.audit_evidence()["final_b0_readback"] == {
        "eq": 8,
        "vref": 0,
    }


def test_unrelated_i2c_bit_drift_stops_before_candidate_write():
    writer = _FieldWriter()
    writer.raw[0x200B] = 0x80
    executor, _plan, _writer, _dca, _waits = _i2c_executor(writer=writer)
    executor.preflight()

    with pytest.raises(RuntimeError, match="unrelated I2C bits drifted"):
        executor.is_at_b0()

    assert writer.writes == []


class _Bert:
    def __init__(self, *, fail_restore=False):
        self.configuration = BertConfiguration(
            port="port1",
            channel="0",
            taps=_b0().fir_taps,
            upper_nlc=6,
            lower_nlc=6,
            raw_response="raw",
            pattern_fields=("SSPRQ", "SSPRQ"),
        )
        self.writes = []
        self.fail_restore = fail_restore

    def get_configuration(self, port, channel):
        return self.configuration

    def get_mode(self):
        return "PAM4"

    def set_fir(self, port, channel, *, taps, upper_nlc, lower_nlc):
        self.writes.append(tuple(taps))
        if self.fail_restore and len(self.writes) == 2:
            raise RuntimeError("injected BERT restore failure")
        self.configuration = replace(self.configuration, taps=tuple(taps))
        return "OK"


def test_bert_point_uses_single_setfir_and_full_b0_restore():
    b0 = _b0()
    client = _Bert()
    transaction = BertCandidateTransaction(
        client=client,
        port=b0.bert_port,
        channel=b0.bert_channel,
        pattern=b0.bert_pattern,
        b0=client.configuration,
    )
    requested = (-1, 8, -31, 110, -3, -5, -1)
    plan = SweepPlan(
        hardware=SweepHardware.BERT_FIR_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        seed=1,
        max_points=1,
        points=(
            SweepPoint(
                0,
                0,
                dict(zip(("pre3", "pre2", "pre1", "main", "post1", "post2", "post3"), requested, strict=True)),
            ),
        ),
    )
    executor = LiveHardwareSweepExecutor(
        b0=b0,
        plan=plan,
        bert_transaction=transaction,
        dca=_Dca([_measurement()]),
        settle_time_s=0.0,
        wait=lambda _seconds: None,
    )

    executor.preflight()
    application = executor.apply_point(plan.points[0])
    executor.measure_point()
    executor.restore_b0()

    assert application["readback"] == dict(plan.points[0].requested)
    assert client.writes == [requested, b0.fir_taps]
    assert executor.is_at_b0()


def test_bert_restore_failure_is_cached_without_second_setfir():
    b0 = _b0()
    client = _Bert(fail_restore=True)
    transaction = BertCandidateTransaction(
        client=client,
        port=b0.bert_port,
        channel=b0.bert_channel,
        pattern=b0.bert_pattern,
        b0=client.configuration,
    )
    requested = (-1, 8, -31, 110, -3, -5, -1)
    plan = SweepPlan(
        hardware=SweepHardware.BERT_FIR_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        seed=1,
        max_points=1,
        points=(
            SweepPoint(
                0,
                0,
                dict(zip(FIR_TAP_NAMES, requested, strict=True)),
            ),
        ),
    )
    executor = LiveHardwareSweepExecutor(
        b0=b0,
        plan=plan,
        bert_transaction=transaction,
        dca=_Dca([_measurement()]),
        settle_time_s=0.0,
        wait=lambda _seconds: None,
    )
    executor.preflight()
    executor.apply_point(plan.points[0])

    with pytest.raises(RecoveryRequired) as caught:
        executor.restore_b0()

    assert client.writes == [requested, b0.fir_taps]
    assert caught.value.evidence["normalized_readback"] == dict(
        zip(FIR_TAP_NAMES, requested, strict=True)
    )
    assert executor.audit_evidence()["final_b0_readback"] == dict(
        zip(FIR_TAP_NAMES, requested, strict=True)
    )


class _ProfileWriter:
    def __init__(self, profile, values):
        self.raw = {
            field.address_for_channel(profile.characterization.default_channel): (
                values[field.name] << field.bit_start
            )
            for field in profile.characterization.parameters
        }
        self.writes = []

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
        self,
        field,
        *,
        channel,
        value,
        expected_current,
        expected_raw,
    ):
        before = self.read_field(field, channel=channel)
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


def test_reviewed_profile_factory_is_lazy_and_builds_real_i2c_executor():
    root = Path(__file__).resolve().parents[3]
    profile = load_live_hardware_config(
        root / "configs" / "hardware" / "vcsel_lab_live.yaml"
    )
    values = {field.name: 0 for field in profile.characterization.parameters}
    fingerprint = DcaFingerprint.from_mappings(
        identity="Keysight Technologies,N1010A,lab,A.08",
        measurement_sources={"outer_er": "CHAN1A", "tdecq": "FUNC1"},
        configuration={"tdecq_equalizer_function": "FUNC1"},
    )
    b0 = B0Snapshot.capture(
        b0_id="profile-b0",
        version=1,
        captured_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
        i2c_fields=values,
        i2c_raw_registers={
            field.address_for_channel(profile.characterization.default_channel): 0
            for field in profile.characterization.parameters
        },
        fir_taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
        bert_pattern="SSPRQ",
        i2c_channel=profile.characterization.default_channel,
        bert_port=profile.bert.port,
        bert_channel=profile.bert.channel,
        dca_tdecq_source=profile.characterization.tdecq.source,
        dca_tdecq_equalizer_function=profile.dca.function,
        dca_outer_er_source=profile.characterization.er.source,
        dca_fingerprint=fingerprint,
        hardware_identities={
            "dca": HardwareIdentity("dca", profile.dca.resource, fingerprint.identity),
            "i2c": HardwareIdentity(
                "i2c",
                f"{profile.i2c.port}/{profile.i2c.baudrate}/0x{profile.i2c.device_address:02X}",
                "chip_id=0x52;revision=0x00",
            ),
            "bert": HardwareIdentity(
                "bert",
                f"{profile.bert.ip}:{profile.bert.udp_port}/{profile.bert.port}/{profile.bert.channel}",
                "mode=PAM4;patterns=SSPRQ,SSPRQ",
            ),
        },
    )
    requested = dict(values)
    requested["eq"] = 1
    plan = SweepPlan(
        hardware=SweepHardware.I2C_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        seed=2,
        max_points=1,
        points=(SweepPoint(0, 0, requested),),
        settle_time_s=0.25,
    )
    writer = _ProfileWriter(profile, values)
    bert = _Bert()
    dca = _Dca([_measurement()])
    opens = []
    closes = []

    @contextmanager
    def drivers(selected_profile, selected_plan):
        opens.append((selected_profile, selected_plan))
        try:
            yield LiveSweepDriverBundle(
                dca=dca,
                i2c_writer=writer,
                i2c_identity=SimpleNamespace(
                    matched=True,
                    chip_id=profile.i2c.expected_chip_id,
                    chip_revision=profile.i2c.expected_chip_revision,
                ),
                bert=bert,
            )
        finally:
            closes.append(True)

    factory = build_live_sweep_executor_factory(
        profile,
        driver_factory=drivers,
    )
    assert opens == []

    executor = factory(b0, plan)
    executor._wait = lambda _seconds: None
    executor.preflight()
    executor.apply_point(plan.points[0])
    executor.measure_point()
    executor.restore_b0()
    executor.close()

    assert len(opens) == 1
    assert writer.writes == [("eq", 1), ("eq", 0)]
    assert closes == [True]
