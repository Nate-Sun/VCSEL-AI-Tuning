import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.hardware_authority import (
    HardwareAuthority,
    HardwareAuthorityBusy,
)
from vcsel_ai_tuning.v1_1.authorization import (
    AuthorizationDriftError,
    OptimizationDepth,
)
import vcsel_ai_tuning.v1_1.service as service_module
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.errors import ReadOnlyPreflightError
from vcsel_ai_tuning.v1_1.live_ai_policy import LiveAIPolicyError
from vcsel_ai_tuning.v1_1.regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from vcsel_ai_tuning.v1_1.repository import (
    ActiveTaskConflict,
    ObservationRecord,
    TaskStore,
    TaskStoreError,
)
from vcsel_ai_tuning.v1_1.service import (
    GatewayAttestation,
    PreflightCapture,
    V11ApplicationService,
)
from vcsel_ai_tuning.v1_1.sweep import SweepHardware, SweepMode


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def snapshot():
    return B0Snapshot.capture(
        b0_id="b0-live",
        version=1,
        captured_at=NOW,
        i2c_fields={"eq": 20},
        i2c_raw_registers={0x2006: 20},
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
            identity="KEYSIGHT",
            measurement_sources={"tdecq": "CHAN1A", "outer_er": "CHAN1A"},
            configuration={
                "rate": "53.125 GBd",
                "tdecq_equalizer_function": "FUNC1",
            },
        ),
        hardware_identities={
            "i2c": HardwareIdentity("i2c", "COM7", "0x52:0x00"),
            "bert": HardwareIdentity("bert", "192.168.3.2:9000", "VIT"),
            "dca": HardwareIdentity("dca", "hislip0", "KEYSIGHT"),
        },
    )


def mapping():
    return scan_parameter_mapping(
        (ScriptEntry("eq", 20, 0),),
        (RegMapField("eq", 0x2006, 0x100, 0, 5, 0, 31),),
    )


class Gateway:
    def __init__(self):
        self.calls = []
        self.capture_value = PreflightCapture(
            snapshot_token="snap-1",
            b0=snapshot(),
            mapping=mapping(),
        )
        self.attestation_value = GatewayAttestation.from_capture(self.capture_value)

    @property
    def runtime_mode(self):
        return RuntimeMode.FAKE

    def health(self):
        return {"status": "ok"}

    def capabilities(self):
        return {"runtime_mode": "FAKE"}

    def preflight_capture(self):
        self.calls.append("preflight_capture")
        return self.capture_value

    def current_attestation(self):
        self.calls.append("current_attestation")
        return self.attestation_value


class Runner:
    def __init__(self):
        self.calls = []

    def start(self, task_id, task_type):
        self.calls.append(("start", task_id, task_type))

    def pause_and_restore_b0(self, task_id):
        self.calls.append(("pause", task_id))

    def resume_from_b0(self, task_id):
        self.calls.append(("resume", task_id))

    def abort_and_restore_b0(self, task_id):
        self.calls.append(("abort", task_id))

    def recover_b0(self, task_id):
        self.calls.append(("recover", task_id))


class Request(SimpleNamespace):
    def model_dump(self, *, mode):
        assert mode == "json"
        return self.payload


def create_request(**channel_mapping_updates):
    channel_mapping = {
        "i2c_channel": 0,
        "bert_port": "port1",
        "bert_channel": "0",
        "dca_tdecq_source": "CHAN1A",
        "dca_tdecq_equalizer_function": 1,
        "dca_outer_er_source": "CHAN1A",
        "operator_confirmed": True,
    }
    channel_mapping.update(channel_mapping_updates)
    return Request(
        task_id="task-1",
        task_type=TaskType.AI_OPTIMIZATION,
        parameters=(SimpleNamespace(name="eq", min=1, max=31, step=2),),
        depth=OptimizationDepth.FAST,
        candidate_budget=10,
        advanced_iteration_limit=None,
        seed=7,
        max_points=None,
        payload={
            "task_id": "task-1",
            "task_type": "AI_OPTIMIZATION",
            "channel_mapping": channel_mapping,
            "parameters": [{"name": "eq", "min": 1, "max": 31, "step": 2}],
            "depth": "FAST",
            "candidate_budget": 10,
            "seed": 7,
        },
    )


def staged_request():
    request = create_request()
    request.depth = OptimizationDepth.FAST
    request.candidate_budget = 27
    request.parameters = (
        SimpleNamespace(name="ibias_dac", min=60, max=80, step=1),
        SimpleNamespace(name="res_gain", min=0, max=10, step=1),
    )
    request.payload = {
        "task_id": "task-1",
        "task_type": "AI_OPTIMIZATION",
        "channel_mapping": dict(request.payload["channel_mapping"]),
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
        "seed": 7,
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
    }
    return request


def pilot_request():
    request = staged_request()
    request.depth = OptimizationDepth.STANDARD
    request.candidate_budget = 3
    request.parameters = (
        SimpleNamespace(name="ibias_dac", min=69, max=73, step=2),
        SimpleNamespace(name="res_gain", min=4, max=8, step=4),
    )
    request.payload = {
        **request.payload,
        "live_ai_profile": "PILOT_3",
        "optimization_scope": "I2C_ONLY",
        "depth": "STANDARD",
        "candidate_budget": 3,
        "coarse_candidate_budget": 3,
        "fine_candidate_budget": 0,
        "restore_after_each_candidate": True,
        "save_dataset": True,
        "parameters": [
            {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
            {"name": "res_gain", "min": 4, "max": 8, "step": 4},
        ],
    }
    for field in (
        "pilot_candidate_budget",
        "fast_candidate_budget",
        "fir_first_candidate_budget",
        "fir_iterative_candidate_budget",
    ):
        request.payload.pop(field, None)
    return request


def staged_service(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    base = snapshot()
    staged_b0 = B0Snapshot.capture(
        b0_id="b0-staged",
        version=1,
        captured_at=NOW,
        i2c_fields={"ibias_dac": 71, "res_gain": 6},
        i2c_raw_registers={0x2006: 71, 0x2007: 6},
        fir_taps=base.fir_taps,
        upper_nlc=base.upper_nlc,
        lower_nlc=base.lower_nlc,
        bert_pattern=base.bert_pattern,
        i2c_channel=base.i2c_channel,
        bert_port=base.bert_port,
        bert_channel=base.bert_channel,
        dca_tdecq_source=base.dca_tdecq_source,
        dca_tdecq_equalizer_function=base.dca_tdecq_equalizer_function,
        dca_outer_er_source=base.dca_outer_er_source,
        dca_fingerprint=base.dca_fingerprint,
        hardware_identities={
            identity.kind: identity for identity in base.hardware_identities
        },
    )
    staged_mapping = scan_parameter_mapping(
        (ScriptEntry("ibias_dac", 71, 0), ScriptEntry("res_gain", 6, 0)),
        (
            RegMapField("ibias_dac", 0x2006, 0x100, 0, 7, 0, 127),
            RegMapField("res_gain", 0x2007, 0x100, 0, 4, 0, 15),
        ),
    )
    gateway.capture_value = PreflightCapture(
        snapshot_token="snap-1",
        b0=staged_b0,
        mapping=staged_mapping,
    )
    gateway.attestation_value = GatewayAttestation.from_capture(
        gateway.capture_value
    )
    return app, store, gateway, runner


def _mutate_persisted_task_payload(store, mutate):
    task = store.require_task("task-1")
    payload = dict(task.payload)
    mutate(payload)
    connection = store._require_connection()
    connection.execute(
        "UPDATE tasks SET payload_json = ? WHERE task_id = ?",
        (json.dumps(payload, separators=(",", ":")), "task-1"),
    )
    connection.commit()


def service(tmp_path):
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    gateway = Gateway()
    runner = Runner()
    return (
        V11ApplicationService(
            store=store,
            gateway=gateway,
            runner=runner,
            artifact_root=tmp_path / "artifacts",
        ),
        store,
        gateway,
        runner,
    )


def test_staged_policy_is_revalidated_at_create_before_persistence(tmp_path):
    app, store, _gateway, _runner = staged_service(tmp_path)
    request = staged_request()
    request.payload["fine_candidate_budget"] = 15
    try:
        with pytest.raises(LiveAIPolicyError, match="fine_candidate_budget"):
            app.create_task(request)
        assert store.get_task("task-1") is None
    finally:
        app.close()


def test_staged_policy_is_frozen_at_authorize_and_revalidated_before_io(tmp_path):
    app, store, gateway, _runner = staged_service(tmp_path)
    try:
        app.create_task(staged_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        _mutate_persisted_task_payload(
            store,
            lambda payload: payload.__setitem__("fine_candidate_budget", 15),
        )

        with pytest.raises(LiveAIPolicyError, match="fine_candidate_budget"):
            app.authorize_task("task-1", authorize_request())

        assert gateway.calls == ["preflight_capture"]
        with pytest.raises(TaskStoreError, match="no authorization"):
            store.authorization_payload("task-1")
    finally:
        app.close()


def test_staged_policy_drift_at_start_blocks_prepare_and_attestation(tmp_path):
    app, store, gateway, runner = staged_service(tmp_path)
    try:
        app.create_task(staged_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        app.authorize_task("task-1", authorize_request())
        frozen = store.load_authorization("task-1")
        assert frozen.payload()["execution_policy"]["candidate_budget"] == 27
        _mutate_persisted_task_payload(
            store,
            lambda payload: payload.__setitem__("candidate_budget", 26),
        )
        calls_before_start = list(gateway.calls)

        with pytest.raises(LiveAIPolicyError, match="candidate_budget"):
            app.start_task("task-1")

        assert gateway.calls == calls_before_start
        assert runner.calls == []
        assert store.require_task("task-1").state is LifecycleState.AUTHORIZED
    finally:
        app.close()


def test_pilot_authorization_freezes_restore_and_dataset_policy(tmp_path):
    app, store, _gateway, _runner = staged_service(tmp_path)
    try:
        app.create_task(pilot_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )

        app.authorize_task("task-1", authorize_request())

        policy = store.load_authorization("task-1").payload()["execution_policy"]
        assert policy == {
            "live_ai_profile": "PILOT_3",
            "optimization_scope": "I2C_ONLY",
            "candidate_budget": 3,
            "coarse_candidate_budget": 3,
            "fine_candidate_budget": 0,
            "settle_time_s": 0.1,
            "restore_after_each_candidate": True,
            "save_dataset": True,
            "restore_policy": "EACH_CANDIDATE_AND_TERMINAL_B0",
        }
    finally:
        app.close()


def test_create_task_persists_dataset_recording_option(tmp_path):
    app, store, _gateway, _runner = service(tmp_path)
    request = create_request()
    request.payload["save_dataset"] = True
    try:
        created = app.create_task(request)

        assert created["config"]["save_dataset"] is True
        assert store.require_task("task-1").payload["save_dataset"] is True
        assert store.events("task-1")[0].payload["save_dataset"] is True
    finally:
        app.close()


def test_task_payload_reports_optional_dataset_files_from_server_artifacts(tmp_path):
    app, store, _gateway, _runner = service(tmp_path)
    request = create_request()
    request.payload["save_dataset"] = True
    try:
        created = app.create_task(request)
        assert created["measurement_status"] == "PENDING"
        assert created["dataset_status"] == "PENDING"
        assert created["dataset_files"] == []

        bundle = app.artifact_root / "task-1" / "iteration-00000001"
        bundle.mkdir(parents=True)
        for name, content in (
            ("raw_eye_density.npy", b"npy"),
            ("raw_eye.png", b"png"),
            ("tdecq_equalized_eye.png", b"equalized"),
        ):
            (bundle / name).write_bytes(content)
        (bundle / "record.json").write_text(
            json.dumps(
                {
                    "dataset_status": "COMPLETE",
                    "dataset_error": None,
                    "measurement_status": "SUCCESS",
                    "sweep_point_id": "iteration-00000001",
                    "observation_id": "task-1:00000001:i2c_coarse",
                }
            ),
            encoding="utf-8",
        )
        store.append_observation(
            "task-1",
            ObservationRecord.measurement(
                observation_id="task-1:00000001:i2c_coarse",
                sequence=1,
                phase="I2C_COARSE",
                requested={"eq": 19},
                applied={"eq": 19},
                readback={"eq": 19},
                tdecq=2.7,
                er=5.5,
                raw_response={},
                created_at=NOW,
            ),
        )

        payload = app.get_task("task-1")
        assert payload["measurement_status"] == "SUCCESS"
        assert payload["dataset_status"] == "SUCCESS"
        assert payload["dataset_error"] is None
        assert {item["kind"] for item in payload["dataset_files"]} == {
            "EYE_DENSITY_NPY",
            "RAW_EYE_PNG",
            "TDECQ_EQUALIZED_EYE_PNG",
            "MEASUREMENT_RECORD_JSON",
        }
        raw = next(
            item
            for item in payload["dataset_files"]
            if item["kind"] == "RAW_EYE_PNG"
        )
        assert raw["observation_id"] == "task-1:00000001:i2c_coarse"
        assert raw["url"] == (
            "/api/v1.1/tasks/task-1/datasets/iteration-00000001/raw_eye.png"
        )
        assert app.dataset_file_path(
            "task-1", "iteration-00000001", "raw_eye.png"
        ) == bundle / "raw_eye.png"
    finally:
        app.close()


def test_terminal_dataset_summary_detects_missing_observation_bundle(tmp_path):
    app, store, _gateway, _runner = service(tmp_path)
    request = create_request()
    request.payload["save_dataset"] = True
    try:
        app.create_task(request)
        for sequence in (1, 2):
            store.append_observation(
                "task-1",
                ObservationRecord.measurement(
                    observation_id=f"task-1:{sequence:08d}:i2c_coarse",
                    sequence=sequence,
                    phase="I2C_COARSE",
                    requested={"eq": 18 + sequence},
                    applied={"eq": 18 + sequence},
                    readback={"eq": 18 + sequence},
                    tdecq=2.8 - sequence / 10,
                    er=5.5,
                    raw_response={},
                    created_at=NOW,
                ),
            )
        bundle = app.artifact_root / "task-1" / "iteration-00000001"
        bundle.mkdir(parents=True)
        (bundle / "record.json").write_text(
            json.dumps(
                {
                    "dataset_status": "COMPLETE",
                    "dataset_error": None,
                    "sweep_point_id": "iteration-00000001",
                    "observation_id": "task-1:00000001:i2c_coarse",
                }
            ),
            encoding="utf-8",
        )
        for expected, new in (
            (LifecycleState.DRAFT, LifecycleState.PREFLIGHTED),
            (LifecycleState.PREFLIGHTED, LifecycleState.B0_FROZEN),
            (LifecycleState.B0_FROZEN, LifecycleState.AUTHORIZED),
            (LifecycleState.AUTHORIZED, LifecycleState.RUNNING),
            (LifecycleState.RUNNING, LifecycleState.COMPLETED),
        ):
            store.transition(
                "task-1",
                expected_state=expected,
                new_state=new,
                event_type=f"TEST_{new.value}",
            )

        payload = app.final_report("task-1")

        assert payload["dataset_status"] == "PARTIAL"
        assert "task-1:00000002:i2c_coarse" in payload["dataset_error"]
    finally:
        app.close()


def test_task_payload_ignores_unpublished_temporary_dataset_bundle(tmp_path):
    app, _store, _gateway, _runner = service(tmp_path)
    request = create_request()
    request.payload["save_dataset"] = True
    try:
        created = app.create_task(request)
        assert created["dataset_status"] == "PENDING"

        temporary = (
            app.artifact_root
            / "task-1"
            / ".iteration-00000001.tmp-deadbeef"
        )
        temporary.mkdir(parents=True)
        (temporary / "record.json").write_text(
            json.dumps(
                {
                    "dataset_status": "COMPLETE",
                    "dataset_error": None,
                    "sweep_point_id": "iteration-00000001",
                }
            ),
            encoding="utf-8",
        )

        payload = app.get_task("task-1")

        assert payload["dataset_status"] == "PENDING"
        assert payload["dataset_error"] is None
        assert payload["dataset_files"] == []
    finally:
        app.close()


def test_dataset_file_path_rejects_unknown_or_traversal_names(tmp_path):
    app, _store, _gateway, _runner = service(tmp_path)
    try:
        app.create_task(create_request())
        with pytest.raises(FileNotFoundError):
            app.dataset_file_path("task-1", "..", "raw_eye.png")
        with pytest.raises(FileNotFoundError):
            app.dataset_file_path("task-1", "point-1", "unexpected.bin")
    finally:
        app.close()


def test_runtime_capabilities_publish_create_task_profile_budgets(tmp_path):
    app, store, _gateway, _runner = service(tmp_path)

    capabilities = app.runtime_capabilities()

    assert capabilities["profile_options"] == [
        {"name": "FAST", "candidate_budget": 24, "max_alternating_rounds": 1},
        {"name": "STANDARD", "candidate_budget": 64, "max_alternating_rounds": 3},
        {"name": "DEEP", "candidate_budget": 128, "max_alternating_rounds": 5},
        {"name": "ADVANCED", "candidate_budget": 192, "max_alternating_rounds": 5},
    ]
    store.close()


def authorize_request():
    return SimpleNamespace(
        authorization_id="auth-1",
        expires_at=NOW + timedelta(days=365),
        operator_confirmed=True,
    )


def prepare_authorized_task(app):
    app.create_task(create_request())
    app.hardware_snapshot()
    app.save_b0(
        "task-1",
        SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
    )
    app.authorize_task("task-1", authorize_request())


def test_expired_authorization_is_rejected_before_hardware_access(tmp_path):
    app, store, gateway, _runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        gateway.calls.clear()
        task_before = store.require_task("task-1")
        events_before = store.events("task-1")
        active_task_before = store.active_task_id()
        request = SimpleNamespace(
            authorization_id="auth-expired",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            operator_confirmed=True,
        )

        with pytest.raises(TaskStoreError, match="expires_at must be after issued_at"):
            app.authorize_task("task-1", request)

        assert gateway.calls == []
        with pytest.raises(TaskStoreError, match="has no authorization"):
            store.authorization_payload("task-1")
        assert store.require_task("task-1") == task_before
        assert store.events("task-1") == events_before
        assert store.active_task_id() == active_task_before
    finally:
        app.close()
        store.close()


def test_authorization_expiring_during_attestation_is_not_saved(
    tmp_path, monkeypatch
):
    entry_time = NOW
    expires_at = entry_time + timedelta(minutes=1)

    class AuthorizationClock(datetime):
        current = entry_time

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(service_module, "datetime", AuthorizationClock)
    app, store, gateway, _runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        original_attestation = gateway.current_attestation

        def expire_during_attestation():
            attestation = original_attestation()
            AuthorizationClock.current = expires_at
            return attestation

        gateway.current_attestation = expire_during_attestation

        with pytest.raises(TaskStoreError, match="expires_at must be after issued_at"):
            app.authorize_task(
                "task-1",
                SimpleNamespace(
                    authorization_id="auth-expired-during-attestation",
                    expires_at=expires_at,
                    operator_confirmed=True,
                ),
            )

        assert gateway.calls[-1] == "current_attestation"
        with pytest.raises(TaskStoreError, match="has no authorization"):
            store.authorization_payload("task-1")
        assert store.require_task("task-1").state is LifecycleState.B0_FROZEN
    finally:
        app.close()
        store.close()


def test_authorization_uses_post_attestation_issued_at(tmp_path, monkeypatch):
    entry_time = NOW
    issued_at = entry_time + timedelta(minutes=1)

    class AuthorizationClock(datetime):
        current = entry_time

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(service_module, "datetime", AuthorizationClock)
    app, store, gateway, _runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        original_attestation = gateway.current_attestation

        def finish_attestation():
            attestation = original_attestation()
            AuthorizationClock.current = issued_at
            return attestation

        gateway.current_attestation = finish_attestation
        app.authorize_task(
            "task-1",
            SimpleNamespace(
                authorization_id="auth-post-attestation",
                expires_at=issued_at + timedelta(minutes=1),
                operator_confirmed=True,
            ),
        )

        assert store.load_authorization("task-1").issued_at == issued_at
    finally:
        app.close()
        store.close()


@pytest.mark.parametrize(
    "target_state",
    (
        LifecycleState.DRAFT,
        LifecycleState.PREFLIGHTED,
        LifecycleState.B0_FROZEN,
        LifecycleState.AUTHORIZED,
    ),
)
def test_abort_before_start_is_inert_terminal_and_releases_task_slot(
    tmp_path, target_state
):
    app, store, gateway, runner = service(tmp_path)
    if target_state is LifecycleState.AUTHORIZED:
        prepare_authorized_task(app)
    else:
        app.create_task(create_request())
        if target_state in {
            LifecycleState.PREFLIGHTED,
            LifecycleState.B0_FROZEN,
        }:
            store.transition(
                "task-1",
                expected_state=LifecycleState.DRAFT,
                new_state=LifecycleState.PREFLIGHTED,
                event_type="TEST_PREFLIGHTED",
            )
        if target_state is LifecycleState.B0_FROZEN:
            store.transition(
                "task-1",
                expected_state=LifecycleState.PREFLIGHTED,
                new_state=LifecycleState.B0_FROZEN,
                event_type="TEST_B0_FROZEN",
            )
    app._ensure_task_lease("task-1")
    gateway_calls_before_cancel = list(gateway.calls)

    result = app.abort_task("task-1", SimpleNamespace(reason="operator cancelled"))

    assert result["state"] == LifecycleState.ABORTED.value
    assert runner.calls == []
    assert gateway.calls == gateway_calls_before_cancel
    assert store.active_task_id() is None
    event = store.events("task-1")[-1]
    assert event.event_type == "TASK_CANCELLED_BEFORE_START"
    assert event.from_state is target_state
    assert event.payload == {"reason": "operator cancelled"}

    request = create_request()
    request.task_id = "task-2"
    request.payload = {**request.payload, "task_id": "task-2"}
    assert app.create_task(request)["task_id"] == "task-2"
    store.close()


def test_running_abort_keeps_existing_runner_rollback_sequence(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    prepare_authorized_task(app)
    app.start_task("task-1")

    result = app.abort_task("task-1", SimpleNamespace(reason="stop now"))

    assert result["state"] == LifecycleState.ABORTED.value
    assert [call[0] for call in runner.calls] == ["start", "abort"]
    assert [event.event_type for event in store.events("task-1")][-4:] == [
        "TASK_STARTED",
        "TASK_ABORT_REQUESTED",
        "B0_ROLLBACK_STARTED",
        "TASK_ABORTED_B0_RESTORED",
    ]
    store.close()


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("i2c_channel", 1),
        ("bert_port", "port2"),
        ("bert_channel", "1"),
        ("dca_tdecq_source", "CHAN2A"),
        ("dca_outer_er_source", "CHAN2A"),
    ),
)
def test_save_b0_rejects_any_operator_confirmed_channel_mapping_mismatch(
    tmp_path, field, changed
):
    app, store, gateway, runner = service(tmp_path)
    try:
        app.create_task(create_request(**{field: changed}))
        app.hardware_snapshot()

        with pytest.raises(TaskStoreError, match="channel mapping"):
            app.save_b0(
                "task-1",
                SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
            )

        assert store.require_task("task-1").state is LifecycleState.DRAFT
        with pytest.raises(TaskStoreError, match="no frozen B0"):
            store.b0_payload("task-1")
        assert runner.calls == []
    finally:
        store.close()


def test_service_is_inert_until_explicit_preflight_and_freezes_dynamic_b0(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    try:
        created = app.create_task(create_request())
        assert created["state"] == "DRAFT"
        assert gateway.calls == []
        assert runner.calls == []

        captured = app.hardware_snapshot()
        saved = app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )

        assert captured["snapshot_token"] == "snap-1"
        assert saved["state"] == "B0_FROZEN"
        assert saved["b0_hash"] == snapshot().content_hash
        assert store.require_task("task-1").state is LifecycleState.B0_FROZEN
        artifact = tmp_path / "artifacts" / "preflight" / "snap-1.json"
        document = json.loads(artifact.read_text(encoding="utf-8"))
        assert document["status"] == "SUCCESS"
        assert document["snapshot_token"] == "snap-1"
        assert document["evidence"]["mutation_counts"][
            "i2c_tunable_register_writes"
        ] == 0
        assert list(artifact.parent.glob("*.tmp")) == []
    finally:
        store.close()


def test_failed_preflight_persists_no_b0_or_snapshot_token(tmp_path):
    app, store, gateway, _runner = service(tmp_path)

    def fail():
        raise RuntimeError("capture failed")

    gateway.preflight_capture = fail
    try:
        app._captures = {"stale-token": gateway.capture_value}
        with pytest.raises(RuntimeError, match="capture failed"):
            app.hardware_snapshot()

        artifacts = list((tmp_path / "artifacts" / "preflight").glob("failed-*.json"))
        assert len(artifacts) == 1
        document = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert document["status"] == "FAILED"
        assert "b0" not in document
        assert "snapshot_token" not in document
        assert app._captures == {}
        with pytest.raises(TaskStoreError, match="unknown or expired"):
            app.scan_parameter_mapping(
                SimpleNamespace(snapshot_token="stale-token")
            )
        assert list(artifacts[0].parent.glob("*.tmp")) == []
    finally:
        store.close()


def test_failed_read_only_preflight_attaches_persisted_artifact_reference(tmp_path):
    app, store, gateway, _runner = service(tmp_path)
    error = ReadOnlyPreflightError(
        "TDECQ source mismatch",
        stage="dca",
        evidence={
            "raw_responses": {
                "dca": {":MEASure:EYE:TDEQ:SOURce?": "FUNC1\n"}
            },
            "mutation_counts": {
                "dca_writes": 0,
                "i2c_tunable_register_writes": 0,
                "bert_writes": 0,
            },
        },
    )

    def fail():
        raise error

    gateway.preflight_capture = fail
    try:
        with pytest.raises(ReadOnlyPreflightError) as caught:
            app.hardware_snapshot()

        assert caught.value.failure_artifact_id.startswith("failed-")
        assert caught.value.failure_artifact_path == (
            f"preflight/{caught.value.failure_artifact_id}.json"
        )
        artifact = app.artifact_root / caught.value.failure_artifact_path
        document = json.loads(artifact.read_text(encoding="utf-8"))
        assert document["evidence"]["raw_responses"]["dca"][
            ":MEASure:EYE:TDEQ:SOURce?"
        ] == "FUNC1\n"
        assert document["evidence"]["mutation_counts"]["dca_writes"] == 0
    finally:
        store.close()


def test_one_snapshot_token_reuses_capture_without_reopening_hardware(tmp_path):
    app, store, gateway, _runner = service(tmp_path)
    try:
        app.create_task(create_request())
        captured = app.hardware_snapshot()
        token_request = SimpleNamespace(snapshot_token=captured["snapshot_token"])

        scanned = app.scan_parameter_mapping(token_request)
        validated = app.validate_parameters(
            SimpleNamespace(
                snapshot_token=captured["snapshot_token"],
                parameters=create_request().parameters,
            )
        )
        counted = app.count_sweep(
            SimpleNamespace(
                snapshot_token=captured["snapshot_token"],
                hardware=SweepHardware.I2C_SWEEP,
                mode=SweepMode.SINGLE_PARAMETER,
                parameters=create_request().parameters,
                seed=0,
                max_points=16,
            )
        )
        frozen = app.save_b0(
            "task-1",
            SimpleNamespace(
                snapshot_token=captured["snapshot_token"],
                operator_confirmed=True,
            ),
        )

        assert scanned["snapshot_token"] == captured["snapshot_token"]
        assert validated["snapshot_token"] == captured["snapshot_token"]
        assert counted["snapshot_token"] == captured["snapshot_token"]
        assert frozen["state"] == "B0_FROZEN"
        assert gateway.calls == ["preflight_capture"]
    finally:
        store.close()


def test_new_snapshot_invalidates_previous_unfrozen_token(tmp_path):
    app, store, gateway, _runner = service(tmp_path)
    try:
        first = app.hardware_snapshot()
        gateway.capture_value = PreflightCapture(
            snapshot_token="snap-2",
            b0=snapshot(),
            mapping=mapping(),
        )
        second = app.hardware_snapshot()

        with pytest.raises(TaskStoreError, match="unknown or expired snapshot token"):
            app.scan_parameter_mapping(
                SimpleNamespace(snapshot_token=first["snapshot_token"])
            )

        result = app.scan_parameter_mapping(
            SimpleNamespace(snapshot_token=second["snapshot_token"])
        )
        assert result["snapshot_token"] == "snap-2"
        assert gateway.calls == ["preflight_capture", "preflight_capture"]
    finally:
        store.close()


def test_concurrent_preflights_publish_only_last_serialized_token(tmp_path):
    app, store, gateway, _runner = service(tmp_path)
    counter = 0

    def capture():
        nonlocal counter
        counter += 1
        return replace(gateway.capture_value, snapshot_token=f"snap-{counter}")

    gateway.preflight_capture = capture
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: app.hardware_snapshot(), range(2)))

        assert {result["snapshot_token"] for result in results} == {
            "snap-1",
            "snap-2",
        }
        assert set(app._captures) == {"snap-2"}
        with pytest.raises(TaskStoreError, match="unknown or expired"):
            app.scan_parameter_mapping(
                SimpleNamespace(snapshot_token="snap-1")
            )
    finally:
        store.close()


def test_save_b0_and_new_preflight_cannot_race_old_snapshot_token(
    tmp_path, monkeypatch
):
    app, store, gateway, _runner = service(tmp_path)
    freeze_entered = Event()
    release_freeze = Event()
    new_preflight_entered = Event()
    original_freeze_b0 = store.freeze_b0
    original_capture = gateway.preflight_capture

    def blocking_freeze_b0(*args, **kwargs):
        freeze_entered.set()
        assert release_freeze.wait(timeout=2.0)
        return original_freeze_b0(*args, **kwargs)

    monkeypatch.setattr(store, "freeze_b0", blocking_freeze_b0)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        gateway.capture_value = replace(
            gateway.capture_value, snapshot_token="snap-2"
        )

        def marked_capture():
            new_preflight_entered.set()
            return original_capture()

        gateway.preflight_capture = marked_capture
        request = SimpleNamespace(
            snapshot_token="snap-1", operator_confirmed=True
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            save_future = executor.submit(app.save_b0, "task-1", request)
            assert freeze_entered.wait(timeout=2.0)
            preflight_future = executor.submit(app.hardware_snapshot)

            assert new_preflight_entered.wait(timeout=0.2) is False
            release_freeze.set()
            assert save_future.result(timeout=2.0)["state"] == "B0_FROZEN"
            assert preflight_future.result(timeout=2.0)[
                "snapshot_token"
            ] == "snap-2"
            assert new_preflight_entered.is_set()

        assert app._task_capture["task-1"].snapshot_token == "snap-1"
        assert set(app._captures) == {"snap-2"}
    finally:
        release_freeze.set()
        store.close()


def test_authorization_revalidates_mapping_and_start_rejects_drift(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        authorized = app.authorize_task("task-1", authorize_request())
        assert authorized["state"] == "AUTHORIZED"
        assert store.authorization_payload("task-1")["payload"][
            "channel_mapping"
        ] == snapshot().channel_mapping_payload()

        gateway.attestation_value = GatewayAttestation(
            hardware_identities=gateway.attestation_value.hardware_identities,
            mapping_hash="f" * 64,
            channel_mapping=gateway.attestation_value.channel_mapping,
            dca_config_fingerprint=gateway.attestation_value.dca_config_fingerprint,
            hardware_state_hash=gateway.attestation_value.hardware_state_hash,
        )
        with pytest.raises(AuthorizationDriftError, match="mapping_hash"):
            app.start_task("task-1")

        assert runner.calls == []
        assert store.require_task("task-1").state is LifecycleState.AUTHORIZED
    finally:
        store.close()


def test_start_and_pause_only_transition_after_runner_restore_succeeds(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        app.authorize_task("task-1", authorize_request())

        started = app.start_task("task-1")
        paused = app.pause_task("task-1")

        assert started["state"] == "RUNNING"
        assert paused["state"] == "PAUSED"
        assert runner.calls == [
            ("start", "task-1", TaskType.AI_OPTIMIZATION),
            ("pause", "task-1"),
        ]
    finally:
        store.close()


def test_bert_sweep_count_uses_fir_taps_not_i2c_regmap_names(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    try:
        app.hardware_snapshot()
        result = app.count_sweep(
            SimpleNamespace(
                snapshot_token="snap-1",
                hardware=SweepHardware.BERT_FIR_SWEEP,
                mode=SweepMode.FULL_COMBINATION,
                parameters=(
                    SimpleNamespace(name="main", min=100, max=110, step=10),
                    SimpleNamespace(name="pre1", min=-30, max=-28, step=2),
                ),
                seed=3,
                max_points=4,
            )
        )

        assert result["hardware"] == "BERT_FIR_SWEEP"
        assert result["point_count"] == 4
    finally:
        store.close()


def test_start_rejects_live_hardware_state_that_no_longer_matches_frozen_b0(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        app.authorize_task("task-1", authorize_request())
        gateway.attestation_value = GatewayAttestation(
            hardware_identities=gateway.attestation_value.hardware_identities,
            mapping_hash=gateway.attestation_value.mapping_hash,
            channel_mapping=gateway.attestation_value.channel_mapping,
            dca_config_fingerprint=gateway.attestation_value.dca_config_fingerprint,
            hardware_state_hash="0" * 64,
        )

        with pytest.raises(AuthorizationDriftError, match="b0_hash"):
            app.start_task("task-1")

        assert runner.calls == []
    finally:
        store.close()


def test_start_rejects_live_channel_mapping_drift(tmp_path):
    app, store, gateway, runner = service(tmp_path)
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        app.authorize_task("task-1", authorize_request())
        gateway.attestation_value = GatewayAttestation(
            hardware_identities=gateway.attestation_value.hardware_identities,
            mapping_hash=gateway.attestation_value.mapping_hash,
            channel_mapping={
                **gateway.attestation_value.channel_mapping,
                "bert_channel": "1",
            },
            dca_config_fingerprint=gateway.attestation_value.dca_config_fingerprint,
            hardware_state_hash=gateway.attestation_value.hardware_state_hash,
        )

        with pytest.raises(AuthorizationDriftError, match="channel_mapping"):
            app.start_task("task-1")

        assert runner.calls == []
        assert store.require_task("task-1").state is LifecycleState.AUTHORIZED
    finally:
        store.close()


def test_external_v1_hardware_task_blocks_v11_task_creation(tmp_path):
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    app = V11ApplicationService(
        store=store,
        gateway=Gateway(),
        runner=Runner(),
        external_active_task_id=lambda: "v1-running",
    )
    try:
        with pytest.raises(ActiveTaskConflict) as caught:
            app.create_task(create_request())

        assert caught.value.active_task_id == "v1:v1-running"
        assert store.active_task_id() is None
    finally:
        store.close()


def test_persisted_recovery_required_reserves_authority_without_hardware_access(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    authority_path = tmp_path / "hardware-authority.lock"
    original = TaskStore(database)
    original.create_task(
        task_id="task-1",
        task_type=TaskType.AI_OPTIMIZATION,
        runtime_mode=RuntimeMode.FAKE,
        config_fingerprint="f" * 64,
        payload={"task_id": "task-1"},
    )
    original.transition(
        "task-1",
        expected_state=LifecycleState.DRAFT,
        new_state=LifecycleState.RUNNING,
        event_type="TEST_RUNNING",
    )
    original.close()

    restarted = TaskStore(database)
    gateway = Gateway()
    app = V11ApplicationService(
        store=restarted,
        gateway=gateway,
        runner=Runner(),
        hardware_authority=HardwareAuthority(authority_path),
    )
    contender = HardwareAuthority(authority_path)
    try:
        assert restarted.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
        assert gateway.calls == []
        with pytest.raises(HardwareAuthorityBusy) as caught:
            contender.acquire("v1:campaign-1:stage")
        assert caught.value.current_owner == "v1.1:task:task-1"
    finally:
        app.close()
        restarted.close()

    with contender.acquire("later-owner"):
        pass


def test_running_task_reopens_inert_then_explicit_resume_recovers_from_persisted_b0(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    authority_path = tmp_path / "hardware-authority.lock"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
        hardware_authority=HardwareAuthority(authority_path),
    )
    prepare_authorized_task(original)
    original.start_task("task-1")
    original.close()
    original_store.close()

    restarted_store = TaskStore(database)
    gateway = Gateway()
    runner = Runner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=gateway,
        runner=runner,
        hardware_authority=HardwareAuthority(authority_path),
    )
    try:
        assert restarted_store.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
        assert gateway.calls == []
        assert runner.calls == []

        recovered = restarted.recover_task(
            "task-1", SimpleNamespace(action="RESUME")
        )

        assert recovered["state"] == "RUNNING"
        assert gateway.calls == ["current_attestation"]
        assert runner.calls == [
            ("recover", "task-1"),
            ("resume", "task-1"),
        ]
    finally:
        restarted.close()
        restarted_store.close()


def test_authorized_task_reopens_inert_and_can_start_from_persisted_evidence(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    authority_path = tmp_path / "hardware-authority.lock"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
        hardware_authority=HardwareAuthority(authority_path),
    )
    prepare_authorized_task(original)
    original.close()
    original_store.close()

    restarted_store = TaskStore(database)
    gateway = Gateway()
    runner = Runner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=gateway,
        runner=runner,
        hardware_authority=HardwareAuthority(authority_path),
    )
    try:
        assert restarted_store.require_task("task-1").state is LifecycleState.AUTHORIZED
        assert gateway.calls == []
        assert runner.calls == []

        started = restarted.start_task("task-1")

        assert started["state"] == "RUNNING"
        assert gateway.calls == ["current_attestation"]
        assert runner.calls == [("start", "task-1", TaskType.AI_OPTIMIZATION)]
    finally:
        restarted.close()
        restarted_store.close()


def test_expired_authorization_cannot_resume_and_retains_recovery_authority(
    tmp_path, monkeypatch
):
    database = tmp_path / "campaigns.sqlite3"
    authority_path = tmp_path / "hardware-authority.lock"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
        hardware_authority=HardwareAuthority(authority_path),
    )
    prepare_authorized_task(original)
    original.start_task("task-1")
    original.close()
    original_store.close()

    class ExpiredClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW + timedelta(days=366)

    monkeypatch.setattr(service_module, "datetime", ExpiredClock)
    restarted_store = TaskStore(database)
    gateway = Gateway()
    runner = Runner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=gateway,
        runner=runner,
        hardware_authority=HardwareAuthority(authority_path),
    )
    contender = HardwareAuthority(authority_path)
    try:
        with pytest.raises(TaskStoreError, match="expired"):
            restarted.recover_task("task-1", SimpleNamespace(action="RESUME"))

        assert restarted_store.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
        assert gateway.calls == []
        assert runner.calls == [("recover", "task-1")]
        with pytest.raises(HardwareAuthorityBusy):
            contender.acquire("later-owner")
    finally:
        restarted.close()
        restarted_store.close()


def test_abort_recovery_releases_authority_for_later_operation(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    authority_path = tmp_path / "hardware-authority.lock"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
        hardware_authority=HardwareAuthority(authority_path),
    )
    prepare_authorized_task(original)
    original.start_task("task-1")
    original.close()
    original_store.close()

    restarted_store = TaskStore(database)
    gateway = Gateway()
    runner = Runner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=gateway,
        runner=runner,
        hardware_authority=HardwareAuthority(authority_path),
    )
    try:
        recovered = restarted.recover_task(
            "task-1", SimpleNamespace(action="ABORT_RECOVERED")
        )

        assert recovered["state"] == "ABORTED_RECOVERED"
        assert gateway.calls == []
        assert runner.calls == [("recover", "task-1")]
        with HardwareAuthority(authority_path).acquire("later-owner"):
            pass
    finally:
        restarted.close()
        restarted_store.close()


@pytest.mark.parametrize("tamper_target", ("b0_payload", "authorization_hash"))
def test_restart_rejects_tampered_persisted_evidence_without_hardware_access(
    tmp_path, tamper_target
):
    database = tmp_path / "campaigns.sqlite3"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
    )
    prepare_authorized_task(original)
    original.close()
    original_store.close()

    connection = sqlite3.connect(database)
    try:
        if tamper_target == "b0_payload":
            payload_json = connection.execute(
                "SELECT payload_json FROM b0_snapshots WHERE task_id = ?",
                ("task-1",),
            ).fetchone()[0]
            payload = json.loads(payload_json)
            payload["b0_id"] = "tampered-b0"
            connection.execute(
                "UPDATE b0_snapshots SET payload_json = ? WHERE task_id = ?",
                (json.dumps(payload), "task-1"),
            )
        else:
            connection.execute(
                "UPDATE task_authorizations SET content_hash = ? WHERE task_id = ?",
                ("0" * 64, "task-1"),
            )
        connection.commit()
    finally:
        connection.close()

    restarted_store = TaskStore(database)
    gateway = Gateway()
    runner = Runner()
    try:
        with pytest.raises(TaskStoreError, match="content hash mismatch"):
            V11ApplicationService(
                store=restarted_store,
                gateway=gateway,
                runner=runner,
            )
        assert gateway.calls == []
        assert runner.calls == []
        assert restarted_store.require_task("task-1").state is LifecycleState.AUTHORIZED
    finally:
        restarted_store.close()


def test_recovery_resume_does_not_start_runner_before_running_transition_commits(
    tmp_path, monkeypatch
):
    database = tmp_path / "campaigns.sqlite3"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
    )
    prepare_authorized_task(original)
    original.start_task("task-1")
    original.close()
    original_store.close()

    restarted_store = TaskStore(database)
    runner = Runner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=Gateway(),
        runner=runner,
    )
    original_transition = restarted_store.transition

    def fail_running_transition(task_id, **kwargs):
        if (
            kwargs["expected_state"] is LifecycleState.RECOVERY_REQUIRED
            and kwargs["new_state"] is LifecycleState.RUNNING
        ):
            raise TaskStoreError("injected transition failure")
        return original_transition(task_id, **kwargs)

    monkeypatch.setattr(restarted_store, "transition", fail_running_transition)
    try:
        with pytest.raises(TaskStoreError, match="injected transition failure"):
            restarted.recover_task("task-1", SimpleNamespace(action="RESUME"))

        assert runner.calls == [("recover", "task-1")]
        assert restarted_store.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
    finally:
        restarted.close()
        restarted_store.close()


def test_recovery_resume_failure_compensates_b0_and_returns_to_recovery_required(
    tmp_path,
):
    class FailingResumeRunner(Runner):
        def resume_from_b0(self, task_id):
            self.calls.append(("resume", task_id))
            raise RuntimeError("injected resume failure")

    database = tmp_path / "campaigns.sqlite3"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
    )
    prepare_authorized_task(original)
    original.start_task("task-1")
    original.close()
    original_store.close()

    restarted_store = TaskStore(database)
    runner = FailingResumeRunner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=Gateway(),
        runner=runner,
    )
    try:
        with pytest.raises(TaskStoreError, match="resume failed"):
            restarted.recover_task("task-1", SimpleNamespace(action="RESUME"))

        assert runner.calls == [
            ("recover", "task-1"),
            ("resume", "task-1"),
            ("recover", "task-1"),
        ]
        assert restarted_store.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
        assert restarted_store.events("task-1")[-1].event_type == "RECOVERY_RESUME_FAILED"
    finally:
        restarted.close()
        restarted_store.close()


def test_recovery_rejects_unknown_action_before_runner_access(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    original_store = TaskStore(database)
    original = V11ApplicationService(
        store=original_store,
        gateway=Gateway(),
        runner=Runner(),
    )
    prepare_authorized_task(original)
    original.start_task("task-1")
    original.close()
    original_store.close()

    restarted_store = TaskStore(database)
    runner = Runner()
    restarted = V11ApplicationService(
        store=restarted_store,
        gateway=Gateway(),
        runner=runner,
    )
    try:
        with pytest.raises(TaskStoreError, match="RESUME or ABORT"):
            restarted.recover_task("task-1", SimpleNamespace(action="RETRY"))

        assert runner.calls == []
        assert restarted_store.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
    finally:
        restarted.close()
        restarted_store.close()


def test_failed_start_attestation_releases_authority(tmp_path):
    authority_path = tmp_path / "hardware-authority.lock"
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    gateway = Gateway()
    app = V11ApplicationService(
        store=store,
        gateway=gateway,
        runner=Runner(),
        hardware_authority=HardwareAuthority(authority_path),
    )
    try:
        app.create_task(create_request())
        app.hardware_snapshot()
        app.save_b0(
            "task-1",
            SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
        )
        app.authorize_task("task-1", authorize_request())
        gateway.attestation_value = GatewayAttestation(
            hardware_identities=gateway.attestation_value.hardware_identities,
            mapping_hash="f" * 64,
            channel_mapping=gateway.attestation_value.channel_mapping,
            dca_config_fingerprint=gateway.attestation_value.dca_config_fingerprint,
            hardware_state_hash=gateway.attestation_value.hardware_state_hash,
        )

        with pytest.raises(AuthorizationDriftError, match="mapping_hash"):
            app.start_task("task-1")

        with HardwareAuthority(authority_path).acquire("later-owner"):
            pass
    finally:
        app.close()
        store.close()


def test_closed_service_cannot_reacquire_authority(tmp_path):
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    gateway = Gateway()
    app = V11ApplicationService(
        store=store,
        gateway=gateway,
        runner=Runner(),
        hardware_authority=authority,
    )
    try:
        app.close()

        with pytest.raises(TaskStoreError, match="service is closed"):
            app.create_task(create_request())
        with pytest.raises(TaskStoreError, match="service is closed"):
            app.hardware_snapshot()
        with pytest.raises(TaskStoreError, match="service is closed"):
            app.scan_parameter_mapping(SimpleNamespace(snapshot_token="snap-1"))
        assert gateway.calls == []
        assert authority.current_owner is None
    finally:
        app.close()
        store.close()


def test_preflight_and_authorization_reads_use_shared_hardware_authority(tmp_path):
    authority_path = tmp_path / "hardware-authority.lock"
    authority = HardwareAuthority(authority_path)
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    gateway = Gateway()
    app = V11ApplicationService(
        store=store,
        gateway=gateway,
        runner=Runner(),
        hardware_authority=authority,
    )
    app.create_task(create_request())

    with HardwareAuthority(authority_path).acquire("v1:busy"):
        with pytest.raises(ActiveTaskConflict):
            app.hardware_snapshot()
    assert gateway.calls == []

    app.hardware_snapshot()
    app.save_b0(
        "task-1",
        SimpleNamespace(snapshot_token="snap-1", operator_confirmed=True),
    )
    gateway.calls.clear()
    with HardwareAuthority(authority_path).acquire("v1:busy"):
        with pytest.raises(ActiveTaskConflict):
            app.authorize_task("task-1", authorize_request())
    assert gateway.calls == []
    store.close()
