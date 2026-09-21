import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from vcsel_ai_tuning.v1_1.authorization import (
    FullCampaignAuthorization,
    OptimizationDepth,
    RollbackPolicy,
)
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.repository import (
    BestStateRecord,
    ObservationRecord,
    TaskStore,
    TaskStoreError,
)


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def b0():
    return B0Snapshot.capture(
        b0_id="b0-1",
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


def authorization(snapshot):
    return FullCampaignAuthorization.freeze(
        authorization_id="auth-1",
        task_id="task-1",
        task_type=TaskType.AI_OPTIMIZATION,
        b0_hash=snapshot.content_hash,
        hardware_identities={"i2c": "0x52", "bert": "VIT", "dca": "KEYSIGHT"},
        mapping_hash="a" * 64,
        channel_mapping=snapshot.channel_mapping_payload(),
        dca_config_fingerprint=snapshot.dca_fingerprint.content_hash,
        parameters=(),
        depth=OptimizationDepth.FAST,
        candidate_budget=5,
        seed=7,
        max_points=None,
        rollback_policy=RollbackPolicy.RESTORE_B0,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )


def create(store):
    store.create_task(
        task_id="task-1",
        task_type=TaskType.AI_OPTIMIZATION,
        runtime_mode=RuntimeMode.FAKE,
        config_fingerprint="c" * 64,
        payload={},
        created_at=NOW,
    )


def test_b0_and_full_campaign_authorization_round_trip_with_metadata(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create(store)
        snapshot = b0()
        store.save_b0("task-1", snapshot)
        frozen = authorization(snapshot)
        store.save_authorization("task-1", frozen)

        stored_b0 = store.b0_payload("task-1")
        stored_auth = store.authorization_payload("task-1")
        task = store.require_task("task-1")

    assert task.b0_hash == snapshot.content_hash
    assert stored_b0["product_version"] == "1.2.0"
    assert stored_b0["payload"]["b0_id"] == "b0-1"
    assert stored_b0["payload"]["channel_mapping"] == snapshot.channel_mapping_payload()
    assert stored_auth["payload"]["mode"] == "FULL_CAMPAIGN"
    assert stored_auth["payload"]["channel_mapping"] == snapshot.channel_mapping_payload()
    assert stored_auth["content_hash"] == frozen.content_hash


def test_observation_normalizes_sentinel_to_invalid_before_persistence(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create(store)
        record = ObservationRecord.measurement(
            observation_id="obs-1",
            sequence=1,
            phase="I2C_COARSE",
            requested={"eq": 21},
            applied={"eq": 21},
            readback={"eq": 21},
            tdecq=9.91e37,
            er=3.7,
            raw_response={"tdecq": "9.91E+37"},
            created_at=NOW,
        )
        store.append_observation("task-1", record)
        restored = store.observations("task-1")[0]

    assert record.valid is False
    assert record.status == "INVALID_TDECQ"
    assert restored.tdecq is None
    assert restored.er == 3.7


def test_complete_global_best_state_round_trips_for_history_and_final_apply(tmp_path):
    with TaskStore(tmp_path / "campaigns.sqlite3") as store:
        create(store)
        observation = ObservationRecord.measurement(
            observation_id="obs-best",
            sequence=1,
            phase="I2C_FINE",
            requested={"eq": 22},
            applied={"eq": 22},
            readback={"eq": 22},
            tdecq=2.3,
            er=3.8,
            raw_response={"status": "CORR"},
            created_at=NOW,
        )
        store.append_observation("task-1", observation)
        best = BestStateRecord(
            observation_id="obs-best",
            i2c={"eq": 22},
            fir=(-1, 8, -31, 110, -3, -5, -1),
            upper_nlc=6,
            lower_nlc=6,
            pattern="SSPRQ",
            bert_port="port1",
            bert_channel="0",
            dca_fingerprint="d" * 64,
            result={"tdecq": 2.3, "er": 3.8},
            updated_at=NOW,
        )
        store.save_best_state("task-1", best)

        restored = store.best_state("task-1")

    assert restored == best


def test_restart_marks_running_task_recovery_required_without_hardware_callback(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with TaskStore(database) as store:
        create(store)
        store.transition(
            "task-1",
            expected_state=LifecycleState.DRAFT,
            new_state=LifecycleState.RUNNING,
            event_type="TEST_RUNNING",
        )

    hardware_calls = []
    with TaskStore(database) as restarted:
        changed = restarted.mark_interrupted_tasks_recovery_required()

        assert changed == ("task-1",)
        assert restarted.require_task("task-1").state is LifecycleState.RECOVERY_REQUIRED
        assert restarted.active_task_id() == "task-1"
        assert restarted.events("task-1")[-1].event_type == "PROCESS_RESTART_DETECTED"
    assert hardware_calls == []


def test_corrupt_persisted_timestamp_type_is_normalized_to_task_store_error(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with TaskStore(database) as store:
        create(store)
        store.save_b0("task-1", b0())

    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE b0_snapshots SET frozen_at = ? WHERE task_id = ?",
            (7, "task-1"),
        )
        connection.commit()
    finally:
        connection.close()

    with TaskStore(database) as store:
        with pytest.raises(TaskStoreError, match="timestamp"):
            store.load_b0("task-1")
