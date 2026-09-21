"""Explicit SQLite schema migrations for the isolated V1.1 database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .constants import PRODUCT_VERSION, SCHEMA_VERSION


class DatabaseVersionError(RuntimeError):
    """Raised when a database cannot be safely migrated by this runtime."""


class UnversionedDatabaseError(DatabaseVersionError):
    """Raised instead of silently adopting a nonempty legacy database."""


_MIGRATION_1 = """
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    state TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    b0_hash TEXT,
    config_fingerprint TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE active_hardware_task (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    task_id TEXT NOT NULL UNIQUE,
    acquired_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE task_events (
    task_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    b0_hash TEXT,
    config_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, sequence),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE b0_snapshots (
    task_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    b0_hash TEXT NOT NULL CHECK (length(b0_hash) = 64),
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    PRIMARY KEY (task_id, version),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE task_authorizations (
    authorization_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
    b0_hash TEXT NOT NULL CHECK (length(b0_hash) = 64),
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE observations (
    observation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    b0_hash TEXT,
    config_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (task_id, sequence),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE sweep_points (
    task_id TEXT NOT NULL,
    original_index INTEGER NOT NULL CHECK (original_index >= 0),
    execution_index INTEGER NOT NULL CHECK (execution_index >= 0),
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    b0_hash TEXT,
    config_fingerprint TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY (task_id, original_index),
    UNIQUE (task_id, execution_index),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE best_states (
    task_id TEXT PRIMARY KEY,
    observation_id TEXT,
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    b0_hash TEXT,
    config_fingerprint TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE,
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
);

CREATE TABLE recovery_records (
    recovery_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    b0_hash TEXT,
    config_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX observations_by_task ON observations(task_id, sequence);
CREATE INDEX sweep_points_by_execution ON sweep_points(task_id, execution_index);
CREATE INDEX recovery_records_by_task ON recovery_records(task_id, created_at);
"""

_MIGRATION_2 = """
CREATE UNIQUE INDEX one_open_history_apply_recovery_per_task
ON recovery_records(task_id, reason_code)
WHERE status = 'OPEN'
  AND reason_code = 'HISTORY_APPLY_ROLLBACK_FAILED';

UPDATE tasks SET schema_version = 2 WHERE schema_version = 1;
UPDATE task_events SET schema_version = 2 WHERE schema_version = 1;
UPDATE b0_snapshots SET schema_version = 2 WHERE schema_version = 1;
UPDATE task_authorizations SET schema_version = 2 WHERE schema_version = 1;
UPDATE observations SET schema_version = 2 WHERE schema_version = 1;
UPDATE sweep_points SET schema_version = 2 WHERE schema_version = 1;
UPDATE best_states SET schema_version = 2 WHERE schema_version = 1;
UPDATE recovery_records SET schema_version = 2 WHERE schema_version = 1;
"""

_MIGRATION_3 = """
CREATE TABLE b0_snapshots_v3 (
    task_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    b0_hash TEXT NOT NULL CHECK (length(b0_hash) = 64),
    payload_json TEXT NOT NULL,
    product_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    campaign_type TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    PRIMARY KEY (task_id, version),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

INSERT INTO b0_snapshots_v3(
    task_id, version, b0_hash, payload_json, product_version,
    schema_version, campaign_type, runtime_mode, config_fingerprint, frozen_at
)
SELECT
    task_id, version, b0_hash, payload_json, product_version,
    3, campaign_type, runtime_mode, config_fingerprint, frozen_at
FROM b0_snapshots;

DROP TABLE b0_snapshots;
ALTER TABLE b0_snapshots_v3 RENAME TO b0_snapshots;

UPDATE tasks SET schema_version = 3 WHERE schema_version = 2;
UPDATE task_events SET schema_version = 3 WHERE schema_version = 2;
UPDATE task_authorizations SET schema_version = 3 WHERE schema_version = 2;
UPDATE observations SET schema_version = 3 WHERE schema_version = 2;
UPDATE sweep_points SET schema_version = 3 WHERE schema_version = 2;
UPDATE best_states SET schema_version = 3 WHERE schema_version = 2;
UPDATE recovery_records SET schema_version = 3 WHERE schema_version = 2;
"""

_MIGRATIONS = {1: _MIGRATION_1, 2: _MIGRATION_2, 3: _MIGRATION_3}


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM schema_meta").fetchall())


def _create_metadata(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        tables = _user_tables(connection)
        if "schema_meta" in tables:
            connection.commit()
            return
        if tables:
            raise UnversionedDatabaseError(
                "refusing to adopt an unversioned nonempty database"
            )
        connection.execute(
            "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
            (("product_version", PRODUCT_VERSION), ("schema_version", "0")),
        )
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def _apply_migration(connection: sqlite3.Connection, target_version: int) -> None:
    migration = _MIGRATIONS[target_version]
    connection.execute("BEGIN IMMEDIATE")
    try:
        current_version = int(_read_metadata(connection)["schema_version"])
        if current_version >= target_version:
            connection.commit()
            return
        if current_version != target_version - 1:
            raise DatabaseVersionError(
                f"cannot migrate schema {current_version} directly to {target_version}"
            )
        if target_version == 2:
            duplicate = connection.execute(
                "SELECT task_id, reason_code FROM recovery_records "
                "WHERE status = 'OPEN' AND "
                "reason_code = 'HISTORY_APPLY_ROLLBACK_FAILED' "
                "GROUP BY task_id, reason_code HAVING COUNT(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicate is not None:
                raise DatabaseVersionError(
                    "schema v2 migration found duplicate OPEN History Apply "
                    f"recoveries for task {duplicate[0]}"
                )
        b0_row_count = None
        if target_version == 3:
            b0_row_count = connection.execute(
                "SELECT COUNT(*) FROM b0_snapshots"
            ).fetchone()[0]
        for statement in migration.split(";"):
            if statement.strip():
                connection.execute(statement)
        if target_version == 3:
            migrated_count = connection.execute(
                "SELECT COUNT(*) FROM b0_snapshots"
            ).fetchone()[0]
            if migrated_count != b0_row_count:
                raise DatabaseVersionError(
                    "schema v3 migration did not preserve every B0 snapshot"
                )
            table_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='b0_snapshots'"
            ).fetchone()
            if table_sql is None or "UNIQUE (b0_hash)" in table_sql[0]:
                raise DatabaseVersionError(
                    "schema v3 migration did not remove global B0 hash uniqueness"
                )
            foreign_key_error = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchone()
            if foreign_key_error is not None:
                raise DatabaseVersionError(
                    "schema v3 migration failed foreign-key validation"
                )
        connection.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            (str(target_version),),
        )
        connection.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'product_version'",
            (PRODUCT_VERSION,),
        )
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def initialize_database(path: str | Path) -> sqlite3.Connection:
    """Open and explicitly migrate a V1.1 database, refusing ambiguous inputs."""

    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        database,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        tables = _user_tables(connection)
        if "schema_meta" not in tables:
            if tables:
                raise UnversionedDatabaseError(
                    "refusing to adopt an unversioned nonempty database"
                )
            _create_metadata(connection)

        metadata = _read_metadata(connection)
        try:
            current_version = int(metadata["schema_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatabaseVersionError("invalid schema_version metadata") from exc
        if current_version > SCHEMA_VERSION:
            raise DatabaseVersionError(
                f"database schema {current_version} is newer than supported "
                f"schema {SCHEMA_VERSION}"
            )
        if metadata.get("product_version") != PRODUCT_VERSION and current_version > 0:
            raise DatabaseVersionError(
                "database product_version is not compatible with V1.1"
            )
        for target in range(current_version + 1, SCHEMA_VERSION + 1):
            _apply_migration(connection, target)
        return connection
    except Exception:
        connection.close()
        raise
