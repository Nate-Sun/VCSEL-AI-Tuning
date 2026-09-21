import json
import sqlite3
from datetime import datetime, timedelta, timezone
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
from vcsel_ai_tuning.v1_1.regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from vcsel_ai_tuning.v1_1.repository import ActiveTaskConflict, TaskStore, TaskStoreError
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
        dca_tdecq_source="FUNC1",
        dca_outer_er_source="CHAN1A",
        dca_fingerprint=DcaFingerprint.from_mappings(
            identity="KEYSIGHT",
            measurement_sources={"tdecq": "FUNC1", "outer_er": "CHAN1A"},
            configuration={"rate": "53.125 GBd"},
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
        "dca_tdecq_source": "FUNC1",
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


def service(tmp_path):
    store = TaskStore(tmp_path / "campaigns.sqlite3")
    gateway = Gateway()
    runner = Runner()
    return V11ApplicationService(store=store, gateway=gateway, runner=runner), store, gateway, runner


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
        ("dca_tdecq_source", "FUNC2"),
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
    finally:
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
            app.scan_parameter_mapping()
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
