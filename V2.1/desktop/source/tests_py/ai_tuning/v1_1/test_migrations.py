import sqlite3

import pytest

from vcsel_ai_tuning.v1_1.constants import PRODUCT_VERSION, SCHEMA_VERSION
from vcsel_ai_tuning.v1_1.migrations import (
    DatabaseVersionError,
    UnversionedDatabaseError,
    initialize_database,
)


EXPECTED_TABLES = {
    "schema_meta",
    "tasks",
    "active_hardware_task",
    "task_events",
    "b0_snapshots",
    "task_authorizations",
    "observations",
    "sweep_points",
    "best_states",
    "recovery_records",
}


def test_fresh_v11_database_uses_explicit_versioned_schema(tmp_path):
    database = tmp_path / "state" / "v1.1" / "campaigns.sqlite3"

    with initialize_database(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        metadata = dict(
            connection.execute("SELECT key, value FROM schema_meta").fetchall()
        )

    assert EXPECTED_TABLES <= tables
    assert metadata == {
        "product_version": PRODUCT_VERSION,
        "schema_version": str(SCHEMA_VERSION),
    }


def test_explicit_v0_to_v1_migration_preserves_existing_test_copy(tmp_path):
    database = tmp_path / "v0-copy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta(key, value) VALUES
                ('product_version', '1.1.0'),
                ('schema_version', '0');
            CREATE TABLE legacy_v11_notes (note TEXT NOT NULL);
            INSERT INTO legacy_v11_notes(note) VALUES ('preserve-me');
            """
        )

    with initialize_database(database) as connection:
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
        note = connection.execute("SELECT note FROM legacy_v11_notes").fetchone()[0]
        task_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()

    assert version == str(SCHEMA_VERSION)
    assert note == "preserve-me"
    assert task_table == (1,)


def test_newer_schema_is_rejected_without_mutation(tmp_path):
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta(key, value) VALUES
                ('product_version', '1.2.0'),
                ('schema_version', '999');
            """
        )

    with pytest.raises(DatabaseVersionError, match="newer"):
        initialize_database(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == "999"


def test_unversioned_nonempty_database_is_not_silently_adopted(tmp_path):
    database = tmp_path / "v1-database-copy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE campaign_state (campaign_id TEXT PRIMARY KEY)")

    with pytest.raises(UnversionedDatabaseError, match="unversioned"):
        initialize_database(database)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert tables == {"campaign_state"}


def _downgrade_copy_to_schema_v1(connection):
    connection.execute(
        "DROP INDEX IF EXISTS one_open_history_apply_recovery_per_task"
    )
    connection.execute(
        "UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'"
    )


def test_schema_v1_copy_migrates_to_v2_without_losing_rows(tmp_path):
    database = tmp_path / "schema-v1-copy.sqlite3"
    with initialize_database(database) as connection:
        _downgrade_copy_to_schema_v1(connection)
        connection.execute(
            "INSERT INTO tasks(task_id, task_type, state, runtime_mode, "
            "product_version, schema_version, b0_hash, config_fingerprint, "
            "payload_json, created_at, updated_at) "
            "VALUES ('task-1', 'AI_OPTIMIZATION', 'ABORTED', 'FAKE', "
            "'1.1.0', 1, NULL, ?, '{}', ?, ?)",
            ("c" * 64, "2026-07-28T00:00:00Z", "2026-07-28T00:00:00Z"),
        )

    with initialize_database(database) as connection:
        metadata = dict(
            connection.execute("SELECT key, value FROM schema_meta").fetchall()
        )
        index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND "
            "name='one_open_history_apply_recovery_per_task'"
        ).fetchone()
        preserved = connection.execute(
            "SELECT task_id FROM tasks WHERE task_id='task-1'"
        ).fetchone()

    assert SCHEMA_VERSION == 2
    assert metadata["schema_version"] == "2"
    assert index is not None
    assert "WHERE status = 'OPEN'" in index[0]
    assert preserved == ("task-1",)


def test_schema_v2_migration_fails_closed_on_duplicate_legacy_open_recoveries(
    tmp_path,
):
    database = tmp_path / "duplicate-schema-v1.sqlite3"
    with initialize_database(database) as connection:
        _downgrade_copy_to_schema_v1(connection)
        connection.execute(
            "INSERT INTO tasks(task_id, task_type, state, runtime_mode, "
            "product_version, schema_version, b0_hash, config_fingerprint, "
            "payload_json, created_at, updated_at) "
            "VALUES ('task-1', 'AI_OPTIMIZATION', 'ABORTED', 'FAKE', "
            "'1.1.0', 1, NULL, ?, '{}', ?, ?)",
            ("c" * 64, "2026-07-28T00:00:00Z", "2026-07-28T00:00:00Z"),
        )
        for recovery_id in ("recovery-1", "recovery-2"):
            connection.execute(
                "INSERT INTO recovery_records(recovery_id, task_id, reason_code, "
                "status, payload_json, product_version, schema_version, "
                "campaign_type, runtime_mode, b0_hash, config_fingerprint, "
                "created_at, resolved_at) VALUES (?, 'task-1', "
                "'HISTORY_APPLY_ROLLBACK_FAILED', 'OPEN', '{}', '1.1.0', 1, "
                "'AI_OPTIMIZATION', 'FAKE', NULL, ?, ?, NULL)",
                (recovery_id, "c" * 64, "2026-07-28T00:00:00Z"),
            )

    with pytest.raises(DatabaseVersionError, match="duplicate OPEN"):
        initialize_database(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == "1"
        assert connection.execute(
            "SELECT COUNT(*) FROM recovery_records"
        ).fetchone()[0] == 2
