from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import sqlite3

import pytest

from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.constants import SCHEMA_VERSION
from vcsel_ai_tuning.v1_1.fake_runtime import FakeV11Gateway
from vcsel_ai_tuning.v1_1.repository import (
    ActiveTaskConflict,
    RecoveryReasonCode,
    TaskConcurrencyError,
    TaskStore,
    TaskStoreError,
)


CREATED_AT = datetime(2026, 7, 27, 10, 0, tzinfo=timezone.utc)


def create_task(store: TaskStore, task_id: str):
    return store.create_task(
        task_id=task_id,
        task_type=TaskType.AI_OPTIMIZATION,
        runtime_mode=RuntimeMode.FAKE,
        config_fingerprint="c" * 64,
        payload={"operator": "tester"},
        created_at=CREATED_AT,
    )


def test_create_task_persists_v11_metadata_event_and_active_lock(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        task = create_task(store, "task-001")

        assert task.task_id == "task-001"
        assert task.task_type is TaskType.AI_OPTIMIZATION
        assert task.state is LifecycleState.DRAFT
        assert task.runtime_mode is RuntimeMode.FAKE
        assert task.product_version == "1.2.0"
        assert task.schema_version == SCHEMA_VERSION
        assert task.config_fingerprint == "c" * 64
        assert task.payload == {"operator": "tester"}
        assert store.active_task_id() == "task-001"
        events = store.events("task-001")
        assert [(event.sequence, event.event_type) for event in events] == [
            (1, "TASK_CREATED")
        ]
        assert events[0].to_state is LifecycleState.DRAFT


def test_second_nonterminal_hardware_task_fails_with_active_task_id(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with TaskStore(database) as store:
        create_task(store, "task-active")

        with pytest.raises(ActiveTaskConflict) as error:
            create_task(store, "task-rejected")

        assert error.value.active_task_id == "task-active"
        assert store.get_task("task-rejected") is None


def test_global_lock_is_transactional_across_two_store_instances(tmp_path):
    database = tmp_path / "campaigns.sqlite3"

    def contender(task_id: str) -> tuple[str, str]:
        with TaskStore(database) as store:
            try:
                create_task(store, task_id)
            except ActiveTaskConflict as exc:
                return "conflict", exc.active_task_id
            return "created", task_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(contender, ("task-a", "task-b")))

    created = [value for status, value in outcomes if status == "created"]
    conflicts = [value for status, value in outcomes if status == "conflict"]
    assert len(created) == 1
    assert conflicts == created


def test_terminal_transition_releases_lock_but_recovery_required_keeps_it(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")
        store.transition(
            "task-001",
            expected_state=LifecycleState.DRAFT,
            new_state=LifecycleState.RECOVERY_REQUIRED,
            event_type="RESTART_DETECTED",
        )
        assert store.active_task_id() == "task-001"

        store.transition(
            "task-001",
            expected_state=LifecycleState.RECOVERY_REQUIRED,
            new_state=LifecycleState.ABORTED_RECOVERED,
            event_type="B0_RESTORED",
        )
        assert store.active_task_id() is None
        create_task(store, "task-002")
        assert store.active_task_id() == "task-002"


def test_stale_transition_is_rejected_without_event_or_state_change(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")

        with pytest.raises(TaskConcurrencyError, match="expected"):
            store.transition(
                "task-001",
                expected_state=LifecycleState.AUTHORIZED,
                new_state=LifecycleState.RUNNING,
                event_type="TASK_STARTED",
            )

        assert store.require_task("task-001").state is LifecycleState.DRAFT
        assert len(store.events("task-001")) == 1


def test_nonfinite_payload_is_rejected_atomically(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        with pytest.raises(ValueError, match="finite JSON"):
            store.create_task(
                task_id="task-001",
                task_type=TaskType.SWEEP,
                runtime_mode=RuntimeMode.FAKE,
                config_fingerprint="d" * 64,
                payload={"bad": float("nan")},
                created_at=CREATED_AT,
            )

        assert store.active_task_id() is None
        assert store.get_task("task-001") is None


def test_identical_b0_hash_is_allowed_across_tasks_but_not_task_versions(
    tmp_path,
):
    database = tmp_path / "campaigns.sqlite3"
    snapshot = FakeV11Gateway().preflight_capture().b0

    with TaskStore(database) as store:
        create_task(store, "sweep-task")
        store.save_b0("sweep-task", snapshot)
        store.transition(
            "sweep-task",
            expected_state=LifecycleState.DRAFT,
            new_state=LifecycleState.COMPLETED,
            event_type="TEST_SWEEP_COMPLETED",
        )

        create_task(store, "optimization-task")
        store.save_b0("optimization-task", snapshot)

        sweep_b0 = store.b0_payload("sweep-task")
        optimization_b0 = store.b0_payload("optimization-task")
        rows = store._require_connection().execute(
            "SELECT task_id, b0_hash FROM b0_snapshots ORDER BY task_id"
        ).fetchall()

        assert sweep_b0["b0_hash"] == snapshot.content_hash
        assert optimization_b0["b0_hash"] == snapshot.content_hash
        assert [(row["task_id"], row["b0_hash"]) for row in rows] == [
            ("optimization-task", snapshot.content_hash),
            ("sweep-task", snapshot.content_hash),
        ]

        with pytest.raises(TaskStoreError, match="version"):
            store.save_b0("optimization-task", snapshot)


def test_freeze_b0_persists_snapshot_state_and_events_in_one_operation(tmp_path):
    snapshot = replace(FakeV11Gateway().b0, captured_at=CREATED_AT)
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-freeze")

        task = store.freeze_b0(
            "task-freeze", snapshot, snapshot_token="snapshot-1"
        )

        assert task.state is LifecycleState.B0_FROZEN
        assert task.b0_hash == snapshot.content_hash
        assert store.load_b0("task-freeze").content_hash == snapshot.content_hash
        events = store.events("task-freeze")
        assert [event.event_type for event in events] == [
            "TASK_CREATED",
            "PREFLIGHT_CONFIRMED",
            "B0_FROZEN",
        ]
        assert events[1].from_state is LifecycleState.DRAFT
        assert events[1].to_state is LifecycleState.PREFLIGHTED
        assert events[2].from_state is LifecycleState.PREFLIGHTED
        assert events[2].to_state is LifecycleState.B0_FROZEN
        assert events[1].created_at >= events[0].created_at
        assert events[2].created_at == events[1].created_at
        assert task.updated_at == events[2].created_at
        assert task.updated_at > snapshot.captured_at
        assert datetime.fromisoformat(
            store.b0_payload("task-freeze")["frozen_at"].replace("Z", "+00:00")
        ) == task.updated_at


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    (
        ("product_version", "9.9.9"),
        ("schema_version", 999),
        ("campaign_type", TaskType.SWEEP.value),
        ("runtime_mode", RuntimeMode.LIVE.value),
        ("b0_hash", "b" * 64),
        ("config_fingerprint", "d" * 64),
    ),
)
def test_recovery_record_rejects_corrupt_wrapper_metadata(
    tmp_path, column, corrupt_value
):
    database = tmp_path / "campaigns.sqlite3"
    with TaskStore(database) as store:
        create_task(store, "task-001")
        store.create_recovery_record(
            recovery_id="recovery-1",
            task_id="task-001",
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload={"temporary_rollback": {}},
        )

    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE recovery_records SET {column} = ? WHERE recovery_id = ?",
            (corrupt_value, "recovery-1"),
        )
        connection.commit()
    finally:
        connection.close()

    with TaskStore(database) as store:
        with pytest.raises(TaskStoreError, match="metadata"):
            store.get_recovery_record("recovery-1")


def test_recovery_create_and_required_audit_are_atomic_and_idempotent(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")
        payload = {
            "source_task_id": "task-001",
            "observation_id": "obs-1",
            "temporary_rollback": {"i2c": {"eq": 20}},
            "apply_error": "fir failed",
            "rollback_error": "restore failed",
        }

        created = store.create_recovery_record(
            recovery_id="recovery-authoritative",
            task_id="task-001",
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload=payload,
        )
        retried = store.create_recovery_record(
            recovery_id="recovery-response-retry",
            task_id="task-001",
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload=payload,
        )

        assert retried == created
        assert [item.recovery_id for item in store.list_recovery_records()] == [
            "recovery-authoritative"
        ]
        recovery_events = [
            event
            for event in store.events("task-001")
            if event.event_type == "HISTORY_APPLY_ROLLBACK_FAILED"
        ]
        assert len(recovery_events) == 1
        assert recovery_events[0].payload == {
            "recovery_id": "recovery-authoritative",
            "reason_code": "HISTORY_APPLY_ROLLBACK_FAILED",
            **payload,
        }


def test_recovery_create_rejects_conflicting_logical_retry(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")
        store.create_recovery_record(
            recovery_id="recovery-1",
            task_id="task-001",
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload={"observation_id": "obs-1"},
        )

        with pytest.raises(TaskStoreError, match="conflicting recovery"):
            store.create_recovery_record(
                recovery_id="recovery-2",
                task_id="task-001",
                reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
                payload={"observation_id": "obs-2"},
            )
        with pytest.raises(TaskStoreError, match="conflicting recovery"):
            store.create_recovery_record(
                recovery_id="recovery-1",
                task_id="task-001",
                reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
                payload={"observation_id": "obs-other"},
            )

        assert len(store.list_recovery_records()) == 1
        assert sum(
            event.event_type == "HISTORY_APPLY_ROLLBACK_FAILED"
            for event in store.events("task-001")
        ) == 1


def test_recovery_create_rolls_back_record_when_audit_insert_fails(
    tmp_path, monkeypatch
):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")

        def fail_event(*args, **kwargs):
            raise RuntimeError("audit unavailable")

        monkeypatch.setattr(store, "_insert_event", fail_event)
        with pytest.raises(RuntimeError, match="audit unavailable"):
            store.create_recovery_record(
                recovery_id="recovery-1",
                task_id="task-001",
                reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
                payload={"observation_id": "obs-1"},
            )

        assert store.list_recovery_records() == ()


def test_recovery_resolve_and_recovered_audit_are_atomic_and_idempotent(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")
        payload = {
            "source_task_id": "task-001",
            "observation_id": "obs-1",
            "temporary_rollback": {"i2c": {"eq": 20}},
            "apply_error": "fir failed",
            "rollback_error": "restore failed",
        }
        store.create_recovery_record(
            recovery_id="recovery-1",
            task_id="task-001",
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload=payload,
        )

        resolved = store.resolve_recovery_record("recovery-1")
        retried = store.resolve_recovery_record("recovery-1")

        assert retried == resolved
        recovered_events = [
            event
            for event in store.events("task-001")
            if event.event_type == "HISTORY_APPLY_RECOVERED"
        ]
        assert len(recovered_events) == 1
        assert recovered_events[0].payload == {
            "recovery_id": "recovery-1",
            "reason_code": "HISTORY_APPLY_ROLLBACK_FAILED",
            "source_task_id": "task-001",
            "observation_id": "obs-1",
            "restored_state": {"i2c": {"eq": 20}},
        }


def test_recovery_resolve_rolls_back_status_when_audit_insert_fails(
    tmp_path, monkeypatch
):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create_task(store, "task-001")
        store.create_recovery_record(
            recovery_id="recovery-1",
            task_id="task-001",
            reason_code=RecoveryReasonCode.HISTORY_APPLY_ROLLBACK_FAILED,
            payload={
                "source_task_id": "task-001",
                "observation_id": "obs-1",
                "temporary_rollback": {"i2c": {"eq": 20}},
            },
        )

        def fail_event(*args, **kwargs):
            raise RuntimeError("recovered audit unavailable")

        monkeypatch.setattr(store, "_insert_event", fail_event)
        with pytest.raises(RuntimeError, match="recovered audit unavailable"):
            store.resolve_recovery_record("recovery-1")

        assert store.require_recovery_record("recovery-1").status.value == "OPEN"
