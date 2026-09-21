from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.hardware_authority import HardwareAuthority
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.history_apply import (
    HistoryApplyController,
    HistoryHardwareState,
    HistoryResultTarget,
)
from vcsel_ai_tuning.v1_1.hardware_control import RecoveryRequired
from vcsel_ai_tuning.v1_1.repository import (
    ActiveTaskConflict,
    BestStateRecord,
    ObservationRecord,
    RecoveryReasonCode,
    RecoveryStatus,
    TaskStore,
    TaskStoreError,
)
from vcsel_ai_tuning.v1_1.service import V11ApplicationService
import vcsel_ai_tuning.v1_1.fake_runtime as fake_runtime


CURRENT = HistoryHardwareState(
    i2c={"eq": 20},
    fir=(0, 9, -28, 104, -8, 1, -3),
    upper_nlc=6,
    lower_nlc=6,
    pattern="SSPRQ",
    bert_port="port1",
    bert_channel="0",
    dca_fingerprint="dca-current",
)
TARGET = HistoryResultTarget(
    source_task_id="old-task",
    observation_id="obs-7",
    i2c={"eq": 22},
    fir=(-1, 8, -31, 110, -3, -5, -1),
    upper_nlc=6,
    lower_nlc=6,
    pattern="SSPRQ",
    bert_port="port1",
    bert_channel="0",
    dca_fingerprint="dca-current",
)


class Hardware:
    def __init__(self, *, fail_at=None, restore_fails=False):
        self.fail_at = fail_at
        self.restore_fails = restore_fails
        self.calls = []
        self.state = CURRENT

    def preflight_for_history(self, target):
        self.calls.append(("preflight", target.observation_id))
        if self.fail_at == "preflight":
            raise RuntimeError("offline")

    def capture_current_state(self):
        self.calls.append(("capture",))
        return self.state

    def apply_i2c(self, requested, *, rollback_values):
        self.calls.append(("i2c", dict(requested), dict(rollback_values)))
        if self.fail_at == "i2c":
            raise RuntimeError("i2c failed")
        self.state = replace(self.state, i2c=dict(requested))
        return dict(requested)

    def apply_fir(self, target, *, rollback_state):
        self.calls.append(("fir", tuple(target.fir), rollback_state.fir))
        if self.fail_at == "fir":
            raise RuntimeError("fir failed")
        self.state = replace(
            self.state,
            fir=tuple(target.fir),
            upper_nlc=target.upper_nlc,
            lower_nlc=target.lower_nlc,
        )
        return tuple(target.fir)

    def measure_once(self):
        self.calls.append(("measure",))
        if self.fail_at == "measure":
            raise RuntimeError("dca failed")
        return {"tdecq": 2.3, "er": 3.8}

    def restore_state_once(self, snapshot):
        self.calls.append(("restore", snapshot))
        if self.restore_fails:
            raise RuntimeError("restore failed")
        self.state = snapshot


class BlockingHardware(Hardware):
    def __init__(self, entered, release):
        super().__init__()
        self.entered = entered
        self.release = release

    def preflight_for_history(self, target):
        super().preflight_for_history(target)
        self.entered.set()
        assert self.release.wait(timeout=5)


class GuardGateway:
    def __init__(self):
        self.calls = []
        self.runtime_mode = RuntimeMode.FAKE

    def health(self):
        return {"status": "ok"}

    def capabilities(self):
        return {}

    def preflight_capture(self):
        self.calls.append("preflight_capture")
        raise AssertionError("gateway must remain untouched")

    def current_attestation(self):
        self.calls.append("current_attestation")
        raise AssertionError("gateway must remain untouched")


class CreateRequest(SimpleNamespace):
    def model_dump(self, *, mode):
        assert mode == "json"
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
        }


def test_active_task_blocks_history_apply_before_hardware_access():
    hardware = Hardware()
    controller = HistoryApplyController(
        active_task_id=lambda: "running-task",
        hardware=hardware,
    )

    with pytest.raises(ActiveTaskConflict) as caught:
        controller.apply(TARGET)

    assert caught.value.active_task_id == "running-task"
    assert hardware.calls == []


def test_history_apply_and_task_creation_cannot_race_past_each_other(tmp_path):
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    entered = Event()
    release = Event()
    hardware = BlockingHardware(entered, release)
    controller = HistoryApplyController(
        active_task_id=lambda: store.active_task_id(),
        hardware=hardware,
        hardware_authority=authority,
    )
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    service = V11ApplicationService(
        store=store,
        gateway=SimpleNamespace(runtime_mode=RuntimeMode.FAKE),
        runner=object(),
        hardware_authority=authority,
    )
    request = CreateRequest(
        task_id="task-1",
        task_type=TaskType.AI_OPTIMIZATION,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        applying = executor.submit(controller.apply, TARGET)
        assert entered.wait(timeout=5)
        with pytest.raises(ActiveTaskConflict) as caught:
            service.create_task(request)
        assert caught.value.active_task_id.startswith("history-apply:old-task:obs-7")
        assert store.active_task_id() is None
        release.set()
        assert applying.result(timeout=5).status == "HISTORY_RESULT_APPLIED"

    assert service.create_task(request)["state"] == "DRAFT"
    store.close()


def test_history_apply_uses_click_time_state_as_temporary_rollback_and_measures_once():
    hardware = Hardware()
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )

    receipt = controller.apply(TARGET)

    assert receipt.status == "HISTORY_RESULT_APPLIED"
    assert receipt.temporary_rollback == CURRENT
    assert receipt.i2c_readback == {"eq": 22}
    assert receipt.fir_readback == TARGET.fir
    assert receipt.measurement == {"tdecq": 2.3, "er": 3.8}
    assert [call[0] for call in hardware.calls] == [
        "preflight", "capture", "i2c", "fir", "measure"
    ]
    assert hardware.calls[2][2] == {"eq": 20}
    assert hardware.calls[3][2] == CURRENT.fir


@pytest.mark.parametrize("failure", ["i2c", "fir", "measure"])
def test_any_apply_failure_restores_click_time_state(failure):
    hardware = Hardware(fail_at=failure)
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )

    with pytest.raises(RuntimeError, match="failed"):
        controller.apply(TARGET)

    assert hardware.state == CURRENT
    assert [call[0] for call in hardware.calls].count("restore") == 1


def test_failed_temporary_restore_enters_recovery_required():
    hardware = Hardware(fail_at="fir", restore_fails=True)
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )

    with pytest.raises(RecoveryRequired) as caught:
        controller.apply(TARGET)

    assert caught.value.rollback_verified is False
    assert [call[0] for call in hardware.calls].count("restore") == 1


def _fake_history_target(b0):
    i2c = dict(b0.i2c_fields)
    i2c["eq"] += 2
    return HistoryResultTarget(
        source_task_id="formal-fake-optimization",
        observation_id="formal-fake-optimization:00000002:i2c_coarse",
        i2c=i2c,
        fir=b0.fir_taps,
        upper_nlc=b0.upper_nlc,
        lower_nlc=b0.lower_nlc,
        pattern=b0.bert_pattern,
        bert_port=b0.bert_port,
        bert_channel=b0.bert_channel,
        dca_fingerprint=b0.dca_fingerprint.content_hash,
    )


def test_fake_history_hardware_repeats_deterministically_without_real_io():
    hardware_type = getattr(fake_runtime, "FakeHistoryHardware", None)
    assert hardware_type is not None
    b0 = fake_runtime.FakeV11Gateway().preflight_capture().b0
    hardware = hardware_type(b0)
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )
    target = _fake_history_target(b0)

    first = controller.apply(target)
    second = controller.apply(target)

    assert first.fake is True
    assert second.fake is True
    assert first.i2c_readback == second.i2c_readback == target.i2c
    assert first.fir_readback == second.fir_readback == target.fir
    assert first.measurement == second.measurement
    assert first.measurement["fake"] is True
    assert hardware.state.i2c == target.i2c
    assert [call[0] for call in hardware.calls] == [
        "preflight", "capture", "i2c", "fir", "measure",
        "preflight", "capture", "i2c", "fir", "measure",
    ]


def test_fake_history_apply_failure_restores_click_time_state():
    hardware_type = getattr(fake_runtime, "FakeHistoryHardware", None)
    assert hardware_type is not None
    b0 = fake_runtime.FakeV11Gateway().preflight_capture().b0
    hardware = hardware_type(b0, fail_at="measure")
    initial = hardware.state
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )

    with pytest.raises(RuntimeError, match="fake measurement failed"):
        controller.apply(_fake_history_target(b0))

    assert hardware.state == initial
    assert hardware.calls[-1][0] == "restore"


def test_fake_history_rollback_failure_enters_recovery_required():
    hardware_type = getattr(fake_runtime, "FakeHistoryHardware", None)
    assert hardware_type is not None
    b0 = fake_runtime.FakeV11Gateway().preflight_capture().b0
    hardware = hardware_type(b0, fail_at="measure", restore_fails=True)
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )

    with pytest.raises(RecoveryRequired) as caught:
        controller.apply(_fake_history_target(b0))

    assert caught.value.rollback_verified is False
    assert hardware.calls[-1][0] == "restore"


def test_retained_history_lease_is_bound_to_one_recovery_id():
    hardware = Hardware(fail_at="fir", restore_fails=True)
    controller = HistoryApplyController(
        active_task_id=lambda: None,
        hardware=hardware,
    )
    with pytest.raises(RecoveryRequired):
        controller.apply(TARGET)

    controller.bind_recovery("recovery-1")
    controller.reserve_recovery("recovery-1")

    assert controller.retained_recovery_id == "recovery-1"
    with pytest.raises(RecoveryRequired, match="different recovery"):
        controller.reserve_recovery("recovery-2")
    controller.close()


def test_history_hardware_state_rejects_i2c_array_of_pairs():
    payload = CURRENT.payload()
    payload["i2c"] = [["eq", 20]]

    with pytest.raises((TypeError, ValueError), match="I2C.*object"):
        HistoryHardwareState.from_payload(payload)


def _persist_history_source(store):
    created_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    store.create_task(
        task_id=TARGET.source_task_id,
        task_type=TaskType.AI_OPTIMIZATION,
        runtime_mode=RuntimeMode.FAKE,
        config_fingerprint="c" * 64,
        payload={},
        created_at=created_at,
    )
    observation = ObservationRecord.measurement(
        observation_id=TARGET.observation_id,
        sequence=1,
        phase="FINAL",
        requested=dict(TARGET.i2c),
        applied=dict(TARGET.i2c),
        readback=dict(TARGET.i2c),
        tdecq=2.3,
        er=3.8,
        raw_response={"status": "CORR"},
        created_at=created_at,
    )
    store.append_observation(TARGET.source_task_id, observation)
    store.save_best_state(
        TARGET.source_task_id,
        BestStateRecord(
            observation_id=TARGET.observation_id,
            i2c=dict(TARGET.i2c),
            fir=TARGET.fir,
            upper_nlc=TARGET.upper_nlc,
            lower_nlc=TARGET.lower_nlc,
            pattern=TARGET.pattern,
            bert_port=TARGET.bert_port,
            bert_channel=TARGET.bert_channel,
            dca_fingerprint=TARGET.dca_fingerprint,
            result={"tdecq": 2.3, "er": 3.8},
            updated_at=created_at,
        ),
    )
    store.transition(
        TARGET.source_task_id,
        expected_state=store.require_task(TARGET.source_task_id).state,
        new_state=LifecycleState.ABORTED,
        event_type="TEST_SOURCE_TERMINAL",
    )


def test_history_rollback_failure_persists_open_recovery_and_retains_authority(
    tmp_path,
):
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    _persist_history_source(store)
    hardware = Hardware(fail_at="fir", restore_fails=True)
    controller = HistoryApplyController(
        active_task_id=store.active_task_id,
        hardware=hardware,
        hardware_authority=authority,
    )
    service = V11ApplicationService(
        store=store,
        gateway=SimpleNamespace(
            runtime_mode=RuntimeMode.FAKE,
            health=lambda: {"status": "ok"},
            capabilities=lambda: {},
        ),
        runner=object(),
        history_apply_controller=controller,
        hardware_authority=authority,
    )

    with pytest.raises(RecoveryRequired):
        service.apply_history_result(
            SimpleNamespace(
                source_task_id=TARGET.source_task_id,
                observation_id=TARGET.observation_id,
            )
        )

    records = service.store.list_recovery_records(status="OPEN")
    assert len(records) == 1
    record = records[0]
    assert record.reason_code == "HISTORY_APPLY_ROLLBACK_FAILED"
    assert record.task_id == TARGET.source_task_id
    assert record.payload["temporary_rollback"] == CURRENT.payload()
    assert record.payload["apply_error"] == "fir failed"
    assert record.payload["rollback_error"] == "restore failed"
    assert service.runtime_health()["status"] == "RECOVERY_REQUIRED"
    assert service.runtime_health()["open_recovery_id"] == record.recovery_id
    assert authority.current_owner == (
        f"history-apply:{TARGET.source_task_id}:{TARGET.observation_id}"
    )
    with pytest.raises(ActiveTaskConflict):
        service.create_task(
            CreateRequest(task_id="new-task", task_type=TaskType.AI_OPTIMIZATION)
        )
    store.close()


def test_history_recovery_persistence_failure_retains_authority_and_health(
    tmp_path, monkeypatch
):
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    _persist_history_source(store)
    hardware = Hardware(fail_at="fir", restore_fails=True)
    controller = HistoryApplyController(
        active_task_id=store.active_task_id,
        hardware=hardware,
        hardware_authority=authority,
    )
    service = V11ApplicationService(
        store=store,
        gateway=GuardGateway(),
        runner=object(),
        history_apply_controller=controller,
        hardware_authority=authority,
    )

    def fail_persistence(**kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "create_recovery_record", fail_persistence)
    with pytest.raises(RuntimeError, match="database unavailable"):
        service.apply_history_result(
            SimpleNamespace(
                source_task_id=TARGET.source_task_id,
                observation_id=TARGET.observation_id,
            )
        )

    assert controller.has_unpersisted_recovery is True
    assert authority.current_owner == (
        f"history-apply:{TARGET.source_task_id}:{TARGET.observation_id}"
    )
    assert store.list_recovery_records(status=RecoveryStatus.OPEN) == ()
    assert service.runtime_health()["status"] == "RECOVERY_REQUIRED"
    assert service.runtime_health()["recovery_persistence"] == "FAILED"
    assert hardware.calls[-1][0] == "restore"
    service.close()
    store.close()


def _persist_open_history_recovery(database):
    with TaskStore(database) as store:
        _persist_history_source(store)
        return store.create_recovery_record(
            recovery_id="recovery-1",
            task_id=TARGET.source_task_id,
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload={
                "source_task_id": TARGET.source_task_id,
                "observation_id": TARGET.observation_id,
                "temporary_rollback": CURRENT.payload(),
                "apply_error": "fir failed",
                "rollback_error": "restore failed",
            },
        )


def _restart_history_service(database, authority, hardware):
    store = TaskStore(database)
    controller = HistoryApplyController(
        active_task_id=store.active_task_id,
        hardware=hardware,
        hardware_authority=authority,
    )
    service = V11ApplicationService(
        store=store,
        gateway=GuardGateway(),
        runner=object(),
        history_apply_controller=controller,
        hardware_authority=authority,
    )
    return service, store


def _history_service(tmp_path, hardware):
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    _persist_history_source(store)
    controller = HistoryApplyController(
        active_task_id=store.active_task_id,
        hardware=hardware,
        hardware_authority=authority,
    )
    service = V11ApplicationService(
        store=store,
        gateway=SimpleNamespace(
            runtime_mode=RuntimeMode.FAKE,
            health=lambda: {"status": "ok"},
            capabilities=lambda: {},
        ),
        runner=object(),
        history_apply_controller=controller,
        hardware_authority=authority,
    )
    return service, store


def test_service_history_success_appends_complete_receipt_audit(tmp_path):
    service, store = _history_service(tmp_path, Hardware())
    request = SimpleNamespace(
        source_task_id=TARGET.source_task_id,
        observation_id=TARGET.observation_id,
    )

    receipt = service.apply_history_result(request)

    event = store.events(TARGET.source_task_id)[-1]
    assert event.event_type == "HISTORY_RESULT_APPLIED"
    assert event.from_state is LifecycleState.ABORTED
    assert event.to_state is LifecycleState.ABORTED
    assert event.payload == receipt
    assert service.final_report(TARGET.source_task_id)["events"][-1] == receipt
    store.close()


def test_service_history_failure_with_verified_rollback_audits_and_propagates(
    tmp_path,
):
    hardware = Hardware(fail_at="fir")
    service, store = _history_service(tmp_path, hardware)
    request = SimpleNamespace(
        source_task_id=TARGET.source_task_id,
        observation_id=TARGET.observation_id,
    )

    with pytest.raises(RuntimeError, match="fir failed"):
        service.apply_history_result(request)

    event = store.events(TARGET.source_task_id)[-1]
    assert event.event_type == "HISTORY_RESULT_APPLY_FAILED_ROLLED_BACK"
    assert event.payload == {
        "source_task_id": TARGET.source_task_id,
        "observation_id": TARGET.observation_id,
        "temporary_rollback": CURRENT.payload(),
        "apply_error": "fir failed",
        "rollback_verified": True,
    }
    assert store.list_recovery_records(status=RecoveryStatus.OPEN) == ()
    store.close()


def test_restart_reserves_open_history_recovery_without_hardware_then_recovers(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    _persist_open_history_recovery(database)
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    hardware = Hardware()

    service, store = _restart_history_service(database, authority, hardware)

    assert hardware.calls == []
    assert authority.current_owner == "history-recovery:recovery-1"
    with pytest.raises(ActiveTaskConflict):
        service.create_task(
            CreateRequest(task_id="blocked", task_type=TaskType.AI_OPTIMIZATION)
        )

    result = service.recover_history_result(
        "recovery-1", SimpleNamespace(operator_confirmed=True)
    )
    retried = service.recover_history_result(
        "recovery-1", SimpleNamespace(operator_confirmed=True)
    )

    assert result["status"] == RecoveryStatus.RESOLVED.value
    assert retried == result
    assert hardware.calls == [("restore", CURRENT)]
    assert store.require_recovery_record("recovery-1").status is RecoveryStatus.RESOLVED
    assert store.events(TARGET.source_task_id)[-1].event_type == "HISTORY_APPLY_RECOVERED"
    assert authority.current_owner is None
    assert service.create_task(
        CreateRequest(task_id="allowed", task_type=TaskType.AI_OPTIMIZATION)
    )["task_id"] == "allowed"
    store.close()


def test_history_recovery_restore_failure_stays_open_and_blocked(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    _persist_open_history_recovery(database)
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    hardware = Hardware(restore_fails=True)
    service, store = _restart_history_service(database, authority, hardware)

    with pytest.raises(RuntimeError, match="restore failed"):
        service.recover_history_result(
            "recovery-1", SimpleNamespace(operator_confirmed=True)
        )

    assert store.require_recovery_record("recovery-1").status is RecoveryStatus.OPEN
    assert service.runtime_health()["status"] == "RECOVERY_REQUIRED"
    assert authority.current_owner == "history-recovery:recovery-1"
    with pytest.raises(ActiveTaskConflict):
        service.create_task(
            CreateRequest(task_id="blocked", task_type=TaskType.AI_OPTIMIZATION)
        )
    store.close()


def test_open_history_recovery_blocks_all_other_hardware_entrypoints(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    _persist_open_history_recovery(database)
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    hardware = Hardware()
    service, store = _restart_history_service(database, authority, hardware)

    blocked_calls = (
        service.hardware_snapshot,
        service.scan_parameter_mapping,
        lambda: service.authorize_task(
            TARGET.source_task_id,
            SimpleNamespace(operator_confirmed=True),
        ),
        lambda: service.start_task(TARGET.source_task_id),
        lambda: service.pause_task(TARGET.source_task_id),
        lambda: service.resume_task(TARGET.source_task_id),
    )
    for operation in blocked_calls:
        with pytest.raises(ActiveTaskConflict, match="history-recovery"):
            operation()

    assert service.gateway.calls == []
    assert hardware.calls == []
    service.close()
    store.close()


def test_closed_history_service_and_controller_never_touch_hardware(tmp_path):
    service, store = _history_service(tmp_path, Hardware())
    hardware = service.history_apply_controller._hardware
    service.close()

    with pytest.raises(TaskStoreError, match="service is closed"):
        service.apply_history_result(
            SimpleNamespace(
                source_task_id=TARGET.source_task_id,
                observation_id=TARGET.observation_id,
            )
        )
    with pytest.raises(RuntimeError, match="closed"):
        service.history_apply_controller.apply(TARGET)
    assert hardware.calls == []
    store.close()


def test_closed_service_cannot_execute_open_history_recovery(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    _persist_open_history_recovery(database)
    authority = HardwareAuthority(tmp_path / "hardware-authority.lock")
    hardware = Hardware()
    service, store = _restart_history_service(database, authority, hardware)
    service.close()

    with pytest.raises(TaskStoreError, match="service is closed"):
        service.recover_history_result(
            "recovery-1", SimpleNamespace(operator_confirmed=True)
        )
    assert hardware.calls == []
    store.close()
