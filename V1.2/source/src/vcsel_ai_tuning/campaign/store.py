from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Self

from .authorization import authorization_id_for
from .models import AuthorizationMode, Campaign, CampaignState, StageAuthorization


class StoreError(RuntimeError):
    """Base error for campaign persistence failures."""


class StoreSerializationError(StoreError):
    """Raised when an event payload cannot be represented as strict JSON."""


class ConcurrencyError(StoreError):
    """Raised when persisted state does not match the caller's expected state."""


class UnknownCampaignError(StoreError):
    """Raised when an operation requires a campaign that has not been persisted."""


class StoreClosedError(StoreError):
    """Raised when a closed campaign store is used."""


@dataclass(frozen=True, slots=True)
class CampaignEvent:
    campaign_id: str
    sequence: int
    event_type: str
    from_state: CampaignState | None
    to_state: CampaignState
    payload: Any
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AuthorizationUsage:
    campaign_id: str
    authorization: StageAuthorization
    activated_at: datetime
    candidate_count: int
    elapsed_s: float


ACTIVE_RESTART_STATES = frozenset(
    {
        CampaignState.RESTORE_B0,
        CampaignState.BASELINE_MEASUREMENT,
        CampaignState.I2C_SCREENING,
        CampaignState.I2C_COARSE,
        CampaignState.I2C_FINE,
        CampaignState.FIR_APPLY,
        CampaignState.I2C_RETUNE,
        CampaignState.FINAL_VERIFICATION,
        CampaignState.ROLLING_BACK,
        CampaignState.ABORTING,
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign_state (
    campaign_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign_metadata (
    campaign_id TEXT PRIMARY KEY,
    baseline_id TEXT NOT NULL,
    baseline_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    authorization_mode TEXT NOT NULL,
    tuning_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaign_state(campaign_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS campaign_events (
    campaign_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (campaign_id, sequence),
    FOREIGN KEY (campaign_id) REFERENCES campaign_state(campaign_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS campaign_observation_sequences (
    campaign_id TEXT PRIMARY KEY,
    last_sequence INTEGER NOT NULL CHECK (last_sequence >= 1),
    FOREIGN KEY (campaign_id) REFERENCES campaign_state(campaign_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS campaign_authorizations (
    campaign_id TEXT NOT NULL,
    authorization_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    authorization_json TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0 CHECK (candidate_count >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    PRIMARY KEY (campaign_id, authorization_id),
    FOREIGN KEY (campaign_id) REFERENCES campaign_state(campaign_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_authorization_per_stage
    ON campaign_authorizations(campaign_id, stage)
    WHERE is_active = 1;

DROP INDEX IF EXISTS one_active_authorization_per_campaign;

CREATE TABLE IF NOT EXISTS campaign_execution_leases (
    campaign_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaign_state(campaign_id)
        ON DELETE CASCADE
);
"""


def _coerce_state(value: CampaignState | str) -> CampaignState:
    try:
        return CampaignState(value)
    except (TypeError, ValueError) as exc:
        raise StoreError(f"unknown campaign state: {value!r}") from exc


def _validate_campaign_id(campaign_id: str) -> str:
    if not isinstance(campaign_id, str) or not campaign_id:
        raise StoreError("campaign_id must be a non-empty string")
    return campaign_id


def _validate_event_type(event_type: str) -> str:
    if not isinstance(event_type, str) or not event_type:
        raise StoreError("event_type must be a non-empty string")
    return event_type


def _validate_owner_id(owner_id: str) -> str:
    if not isinstance(owner_id, str) or not owner_id:
        raise StoreError("owner_id must be a non-empty string")
    return owner_id


def _validate_authorization_id(authorization_id: str) -> str:
    if not isinstance(authorization_id, str) or not authorization_id:
        raise StoreError("authorization_id must be a non-empty string")
    return authorization_id


def _strict_json(payload: Any) -> tuple[str, Any]:
    normalized_payload = {} if payload is None else payload
    try:
        payload_json = json.dumps(
            normalized_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        restored_payload = json.loads(payload_json)
    except (TypeError, ValueError) as exc:
        raise StoreSerializationError(
            f"payload must be strict finite JSON: {exc}"
        ) from exc
    return payload_json, restored_payload


def _utc_now_or(value: datetime | None) -> datetime:
    timestamp = datetime.now(timezone.utc) if value is None else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise StoreError("timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def _encode_datetime(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _decode_datetime(value: str) -> datetime:
    try:
        decoded = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise StoreError(f"invalid persisted timestamp: {value!r}") from exc
    if decoded.tzinfo is None or decoded.utcoffset() is None:
        raise StoreError("persisted timestamp must be timezone-aware")
    return decoded.astimezone(timezone.utc)


def _authorization_json(
    authorization: StageAuthorization,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(authorization, StageAuthorization):
        raise StoreError("authorization must be a StageAuthorization")
    if authorization.authorization_id != authorization_id_for(authorization):
        raise StoreError("authorization_id does not match authorization payload")
    payload_json, payload = _strict_json(authorization.model_dump(mode="json"))
    return payload_json, payload


def _decode_authorization(value: str) -> StageAuthorization:
    try:
        authorization = StageAuthorization.model_validate_json(value)
    except (TypeError, ValueError) as exc:
        raise StoreError("invalid persisted authorization JSON") from exc
    if authorization.authorization_id != authorization_id_for(authorization):
        raise StoreError("persisted authorization_id does not match its payload")
    return authorization


def _elapsed_since(activated_at: datetime, now: datetime) -> float:
    elapsed_s = (now - activated_at).total_seconds()
    if elapsed_s < 0:
        raise StoreError("now must not precede authorization activation")
    return elapsed_s


def _serialize_public_connection_calls(cls: type[Any]) -> type[Any]:
    """Hold one reentrant lock for every complete public store operation."""

    for name, method in tuple(vars(cls).items()):
        if name.startswith("_") or not callable(method):
            continue

        @wraps(method)
        def guarded(self: Any, *args: Any, __method: Any = method, **kwargs: Any):
            with self._connection_lock:
                return __method(self, *args, **kwargs)

        setattr(cls, name, guarded)
    return cls


@_serialize_public_connection_calls
class CampaignStore:
    """SQLite persistence for campaign state and strictly ordered events."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._connection_lock = RLock()
        self._connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.executescript(_SCHEMA)
            metadata_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(campaign_metadata)"
                ).fetchall()
            }
            if "tuning_path" not in metadata_columns:
                connection.execute(
                    "ALTER TABLE campaign_metadata ADD COLUMN tuning_path TEXT"
                )
        except sqlite3.Error as exc:
            if "connection" in locals():
                connection.close()
            raise StoreError(f"failed to initialize campaign store: {exc}") from exc
        self._connection = connection

    def __enter__(self) -> Self:
        self._require_connection()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        self._connection = None
        connection.close()

    def create_campaign(
        self,
        campaign: Campaign,
        *,
        tuning_path: str | None = None,
    ) -> Campaign:
        """Atomically persist immutable campaign metadata, DRAFT state, and audit."""

        if not isinstance(campaign, Campaign):
            raise StoreError("campaign must be a Campaign")
        if campaign.state != CampaignState.DRAFT:
            raise StoreError("new campaign state must be DRAFT")
        campaign_id = _validate_campaign_id(campaign.campaign_id)
        if tuning_path is not None and (
            not isinstance(tuning_path, str) or not tuning_path
        ):
            raise StoreError("tuning_path must be a non-empty string")
        created_at = _utc_now_or(campaign.created_at)
        updated_at = _utc_now_or(campaign.updated_at)
        encoded_created_at = _encode_datetime(created_at)
        encoded_updated_at = _encode_datetime(updated_at)
        payload_json, restored_payload = _strict_json(
            {
                "authorization_mode": campaign.authorization_mode.value,
                "baseline_hash": campaign.baseline_hash,
                "baseline_id": campaign.baseline_id,
                "config_hash": campaign.config_hash,
            }
        )
        connection = self._begin_immediate("create campaign")
        try:
            existing = connection.execute(
                "SELECT m.campaign_id, m.baseline_id, m.baseline_hash, "
                "m.config_hash, m.authorization_mode, m.tuning_path, m.created_at, "
                "s.state, s.updated_at FROM campaign_metadata AS m "
                "JOIN campaign_state AS s USING (campaign_id) "
                "WHERE m.campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if existing is not None:
                persisted = self._campaign_from_row(existing)
                immutable_matches = (
                    persisted.baseline_id == campaign.baseline_id
                    and persisted.baseline_hash == campaign.baseline_hash
                    and persisted.config_hash == campaign.config_hash
                    and persisted.authorization_mode == campaign.authorization_mode
                    and persisted.created_at == created_at
                )
                if not immutable_matches:
                    raise ConcurrencyError(
                        f"campaign {campaign_id} metadata conflict"
                    )
                if (
                    tuning_path is not None
                    and existing["tuning_path"] != tuning_path
                ):
                    raise ConcurrencyError(
                        f"campaign {campaign_id} tuning path conflict"
                    )
                self._commit(connection, "create campaign")
                return persisted

            legacy_state = connection.execute(
                "SELECT 1 FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if legacy_state is not None:
                raise ConcurrencyError(
                    f"campaign {campaign_id} already exists without runtime metadata"
                )
            connection.execute(
                "INSERT INTO campaign_state (campaign_id, state, updated_at) "
                "VALUES (?, ?, ?)",
                (campaign_id, CampaignState.DRAFT.value, encoded_updated_at),
            )
            connection.execute(
                "INSERT INTO campaign_metadata "
                "(campaign_id, baseline_id, baseline_hash, config_hash, "
                "authorization_mode, tuning_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    campaign.baseline_id,
                    campaign.baseline_hash,
                    campaign.config_hash,
                    campaign.authorization_mode.value,
                    tuning_path,
                    encoded_created_at,
                ),
            )
            sequence = self._next_sequence(connection, campaign_id)
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    "CAMPAIGN_CREATED",
                    None,
                    CampaignState.DRAFT.value,
                    payload_json,
                    encoded_created_at,
                ),
            )
            if tuning_path is not None:
                path_payload_json, _ = _strict_json(
                    {"tuning_path": tuning_path}
                )
                path_sequence = self._next_sequence(connection, campaign_id)
                connection.execute(
                    "INSERT INTO campaign_events "
                    "(campaign_id, sequence, event_type, from_state, to_state, "
                    "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        campaign_id,
                        path_sequence,
                        "CAMPAIGN_PATH_SELECTED",
                        CampaignState.DRAFT.value,
                        CampaignState.DRAFT.value,
                        path_payload_json,
                        encoded_created_at,
                    ),
                )
            self._commit(connection, "create campaign")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("create campaign", exc)
        return Campaign(
            campaign_id=campaign.campaign_id,
            baseline_id=campaign.baseline_id,
            baseline_hash=campaign.baseline_hash,
            config_hash=campaign.config_hash,
            authorization_mode=campaign.authorization_mode,
            state=CampaignState.DRAFT,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_campaign(self, campaign_id: str) -> Campaign:
        campaign_id = _validate_campaign_id(campaign_id)
        connection = self._require_connection()
        try:
            row = connection.execute(
                "SELECT m.campaign_id, m.baseline_id, m.baseline_hash, "
                "m.config_hash, m.authorization_mode, m.created_at, "
                "s.state, s.updated_at FROM campaign_metadata AS m "
                "JOIN campaign_state AS s USING (campaign_id) "
                "WHERE m.campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to read campaign metadata: {exc}") from exc
        if row is None:
            raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
        return self._campaign_from_row(row)

    def get_campaign_tuning_path(self, campaign_id: str) -> str | None:
        campaign_id = _validate_campaign_id(campaign_id)
        connection = self._require_connection()
        try:
            row = connection.execute(
                "SELECT tuning_path FROM campaign_metadata WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to read campaign tuning path: {exc}") from exc
        if row is None:
            raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
        value = row["tuning_path"]
        if value is not None and (not isinstance(value, str) or not value):
            raise StoreError("persisted tuning path is invalid")
        return value

    def list_campaigns(self, *, limit: int = 50, offset: int = 0) -> list[Campaign]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise StoreError("limit must be an integer between 1 and 100")
        if type(offset) is not int or offset < 0:
            raise StoreError("offset must be a non-negative integer")
        connection = self._require_connection()
        try:
            rows = connection.execute(
                "SELECT m.campaign_id, m.baseline_id, m.baseline_hash, "
                "m.config_hash, m.authorization_mode, m.created_at, "
                "s.state, s.updated_at FROM campaign_metadata AS m "
                "JOIN campaign_state AS s USING (campaign_id) "
                "ORDER BY m.created_at DESC, m.campaign_id ASC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to list campaign metadata: {exc}") from exc
        return [self._campaign_from_row(row) for row in rows]

    def legacy_campaign_ids(self) -> tuple[str, ...]:
        """Return state rows that predate required immutable runtime metadata."""

        connection = self._require_connection()
        try:
            rows = connection.execute(
                "SELECT s.campaign_id FROM campaign_state AS s "
                "LEFT JOIN campaign_metadata AS m USING (campaign_id) "
                "WHERE m.campaign_id IS NULL ORDER BY s.campaign_id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to inspect legacy campaigns: {exc}") from exc
        return tuple(str(row["campaign_id"]) for row in rows)

    @staticmethod
    def _campaign_from_row(row: sqlite3.Row) -> Campaign:
        try:
            authorization_mode = AuthorizationMode(row["authorization_mode"])
        except (TypeError, ValueError) as exc:
            raise StoreError("persisted authorization mode is invalid") from exc
        return Campaign(
            campaign_id=row["campaign_id"],
            baseline_id=row["baseline_id"],
            baseline_hash=row["baseline_hash"],
            config_hash=row["config_hash"],
            authorization_mode=authorization_mode,
            state=_coerce_state(row["state"]),
            created_at=_decode_datetime(row["created_at"]),
            updated_at=_decode_datetime(row["updated_at"]),
        )

    def save_state(
        self,
        campaign_id: str,
        state: CampaignState | str,
        *,
        payload: Any = None,
        created_at: datetime | None = None,
    ) -> CampaignEvent:
        """Persist a state snapshot and its STATE_SAVED event atomically."""

        campaign_id = _validate_campaign_id(campaign_id)
        target = _coerce_state(state)
        payload_json, restored_payload = _strict_json(payload)
        timestamp = _utc_now_or(created_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("save state")
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is not None:
                raise ConcurrencyError(
                    f"campaign {campaign_id} already exists; "
                    "save_state is bootstrap-only"
                )
            from_state = None
            connection.execute(
                "INSERT INTO campaign_state (campaign_id, state, updated_at) "
                "VALUES (?, ?, ?)",
                (campaign_id, target.value, encoded_timestamp),
            )
            sequence = self._next_sequence(connection, campaign_id)
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    "STATE_SAVED",
                    None if from_state is None else from_state.value,
                    target.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "save state")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("save state", exc)
        return CampaignEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            event_type="STATE_SAVED",
            from_state=from_state,
            to_state=target,
            payload=restored_payload,
            created_at=timestamp,
        )

    def save_transition(
        self,
        campaign_id: str,
        from_state: CampaignState | str,
        to_state: CampaignState | str,
        *,
        event_type: str = "STATE_TRANSITION",
        payload: Any = None,
        created_at: datetime | None = None,
    ) -> CampaignEvent:
        """Atomically compare state, update it, and append its transition event."""

        campaign_id = _validate_campaign_id(campaign_id)
        expected = _coerce_state(from_state)
        target = _coerce_state(to_state)
        event_type = _validate_event_type(event_type)
        payload_json, restored_payload = _strict_json(payload)
        timestamp = _utc_now_or(created_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("save transition")
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            persisted = _coerce_state(row["state"])
            if persisted != expected:
                raise ConcurrencyError(
                    f"expected current state {expected.value}, found {persisted.value}"
                )
            sequence = self._next_sequence(connection, campaign_id)
            updated = connection.execute(
                "UPDATE campaign_state SET state = ?, updated_at = ? "
                "WHERE campaign_id = ? AND state = ?",
                (
                    target.value,
                    encoded_timestamp,
                    campaign_id,
                    expected.value,
                ),
            )
            if updated.rowcount != 1:
                raise ConcurrencyError(
                    f"campaign {campaign_id} changed during transition"
                )
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    event_type,
                    expected.value,
                    target.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "save transition")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("save transition", exc)
        return CampaignEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            event_type=event_type,
            from_state=expected,
            to_state=target,
            payload=restored_payload,
            created_at=timestamp,
        )

    def save_authorized_transition(
        self,
        campaign_id: str,
        from_state: CampaignState | str,
        to_state: CampaignState | str,
        authorization: StageAuthorization,
        *,
        event_type: str = "AUTHORIZED_STAGE_ENTERED",
        activated_at: datetime | None = None,
    ) -> CampaignEvent:
        """Atomically transition state, activate authorization, and audit it."""

        campaign_id = _validate_campaign_id(campaign_id)
        expected = _coerce_state(from_state)
        target = _coerce_state(to_state)
        event_type = _validate_event_type(event_type)
        authorization_json, authorization_payload = _authorization_json(
            authorization
        )
        if authorization.campaign_id != campaign_id:
            raise StoreError("authorization campaign_id does not match campaign")
        if authorization.stage != target:
            raise StoreError("authorization stage does not match transition target")
        payload_json, restored_payload = _strict_json(
            {"authorization": authorization_payload}
        )
        timestamp = _utc_now_or(activated_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("save authorized transition")
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            persisted = _coerce_state(row["state"])
            if persisted != expected:
                raise ConcurrencyError(
                    f"expected current state {expected.value}, "
                    f"found {persisted.value}"
                )

            sequence = self._next_sequence(connection, campaign_id)
            updated = connection.execute(
                "UPDATE campaign_state SET state = ?, updated_at = ? "
                "WHERE campaign_id = ? AND state = ?",
                (
                    target.value,
                    encoded_timestamp,
                    campaign_id,
                    expected.value,
                ),
            )
            if updated.rowcount != 1:
                raise ConcurrencyError(
                    f"campaign {campaign_id} changed during authorized transition"
                )
            connection.execute(
                "UPDATE campaign_authorizations SET is_active = 0 "
                "WHERE campaign_id = ? AND stage != ? AND is_active = 1",
                (campaign_id, target.value),
            )
            self._activate_authorization_row(
                connection,
                campaign_id,
                authorization,
                authorization_json,
                encoded_timestamp,
                allow_replacement=True,
            )
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    event_type,
                    expected.value,
                    target.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "save authorized transition")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("save authorized transition", exc)
        return CampaignEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            event_type=event_type,
            from_state=expected,
            to_state=target,
            payload=restored_payload,
            created_at=timestamp,
        )

    def activate_authorization(
        self,
        campaign_id: str,
        authorization: StageAuthorization,
        *,
        event_type: str = "STAGE_AUTHORIZATION_ACTIVATED",
        replace_existing: bool = False,
        activated_at: datetime | None = None,
    ) -> CampaignEvent:
        """Activate or reaffirm authorization for an already-active stage."""

        campaign_id = _validate_campaign_id(campaign_id)
        event_type = _validate_event_type(event_type)
        if type(replace_existing) is not bool:
            raise StoreError("replace_existing must be a boolean")
        authorization_json, authorization_payload = _authorization_json(
            authorization
        )
        if authorization.campaign_id != campaign_id:
            raise StoreError("authorization campaign_id does not match campaign")
        payload_json, restored_payload = _strict_json(
            {"authorization": authorization_payload}
        )
        timestamp = _utc_now_or(activated_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("activate authorization")
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            persisted = _coerce_state(row["state"])
            if persisted != authorization.stage:
                raise ConcurrencyError(
                    "authorization stage "
                    f"{authorization.stage.value} does not match current state "
                    f"{persisted.value}"
                )

            self._activate_authorization_row(
                connection,
                campaign_id,
                authorization,
                authorization_json,
                encoded_timestamp,
                allow_replacement=replace_existing,
            )
            sequence = self._next_sequence(connection, campaign_id)
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    event_type,
                    persisted.value,
                    persisted.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "activate authorization")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("activate authorization", exc)
        return CampaignEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            event_type=event_type,
            from_state=persisted,
            to_state=persisted,
            payload=restored_payload,
            created_at=timestamp,
        )

    def authorization_usage(
        self,
        campaign_id: str,
        authorization_id: str,
        *,
        now: datetime | None = None,
    ) -> AuthorizationUsage:
        """Return persisted authorization bounds and derived runtime usage."""

        campaign_id = _validate_campaign_id(campaign_id)
        authorization_id = _validate_authorization_id(authorization_id)
        timestamp = _utc_now_or(now)
        connection = self._require_connection()
        try:
            row = connection.execute(
                "SELECT authorization_json, activated_at, candidate_count "
                "FROM campaign_authorizations "
                "WHERE campaign_id = ? AND authorization_id = ?",
                (campaign_id, authorization_id),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to read authorization usage: {exc}") from exc
        if row is None:
            raise StoreError(
                f"unknown authorization {authorization_id} for campaign {campaign_id}"
            )
        authorization = _decode_authorization(row["authorization_json"])
        activated_at = _decode_datetime(row["activated_at"])
        return AuthorizationUsage(
            campaign_id=campaign_id,
            authorization=authorization,
            activated_at=activated_at,
            candidate_count=int(row["candidate_count"]),
            elapsed_s=_elapsed_since(activated_at, timestamp),
        )

    def activate_nested_authorization(
        self,
        campaign_id: str,
        outer_authorization_id: str,
        authorization: StageAuthorization,
        *,
        event_type: str = "NESTED_AUTHORIZATION_ACTIVATED",
        activated_at: datetime | None = None,
    ) -> CampaignEvent:
        """Activate I2C_RETUNE authorization beneath an active FIR envelope."""

        campaign_id = _validate_campaign_id(campaign_id)
        outer_authorization_id = _validate_authorization_id(
            outer_authorization_id
        )
        event_type = _validate_event_type(event_type)
        authorization_json, authorization_payload = _authorization_json(
            authorization
        )
        if authorization.campaign_id != campaign_id:
            raise StoreError("authorization campaign_id does not match campaign")
        if authorization.stage != CampaignState.I2C_RETUNE:
            raise StoreError("nested authorization stage must be I2C_RETUNE")
        payload_json, restored_payload = _strict_json(
            {
                "authorization": authorization_payload,
                "outer_authorization_id": outer_authorization_id,
            }
        )
        timestamp = _utc_now_or(activated_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("activate nested authorization")
        try:
            campaign = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            persisted = _coerce_state(campaign["state"])
            if persisted != CampaignState.FIR_APPLY:
                raise ConcurrencyError(
                    "nested I2C_RETUNE authorization requires current state "
                    "FIR_APPLY"
                )
            outer = connection.execute(
                "SELECT 1 FROM campaign_authorizations "
                "WHERE campaign_id = ? AND authorization_id = ? "
                "AND stage = ? AND is_active = 1",
                (
                    campaign_id,
                    outer_authorization_id,
                    CampaignState.FIR_APPLY.value,
                ),
            ).fetchone()
            if outer is None:
                raise ConcurrencyError(
                    "named outer FIR_APPLY authorization is not active"
                )

            self._activate_authorization_row(
                connection,
                campaign_id,
                authorization,
                authorization_json,
                encoded_timestamp,
                allow_replacement=True,
            )
            sequence = self._next_sequence(connection, campaign_id)
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    event_type,
                    persisted.value,
                    persisted.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "activate nested authorization")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("activate nested authorization", exc)
        return CampaignEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            event_type=event_type,
            from_state=persisted,
            to_state=persisted,
            payload=restored_payload,
            created_at=timestamp,
        )

    def acquire_execution_lease(
        self,
        campaign_id: str,
        owner_id: str,
        *,
        acquired_at: datetime | None = None,
    ) -> None:
        """Acquire the campaign's exclusive persisted execution lease."""

        campaign_id = _validate_campaign_id(campaign_id)
        owner_id = _validate_owner_id(owner_id)
        timestamp = _utc_now_or(acquired_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("acquire execution lease")
        try:
            campaign = connection.execute(
                "SELECT 1 FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            lease = connection.execute(
                "SELECT owner_id FROM campaign_execution_leases "
                "WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if lease is not None:
                raise ConcurrencyError(
                    f"campaign {campaign_id} execution lease is already held by "
                    f"{lease['owner_id']}"
                )
            connection.execute(
                "INSERT INTO campaign_execution_leases "
                "(campaign_id, owner_id, acquired_at) VALUES (?, ?, ?)",
                (campaign_id, owner_id, encoded_timestamp),
            )
            self._commit(connection, "acquire execution lease")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("acquire execution lease", exc)

    def reserve_candidate(
        self,
        campaign_id: str,
        authorization_id: str,
        owner_id: str,
        *,
        now: datetime | None = None,
    ) -> int:
        """Atomically reserve the next zero-based authorized candidate ordinal."""

        campaign_id = _validate_campaign_id(campaign_id)
        authorization_id = _validate_authorization_id(authorization_id)
        owner_id = _validate_owner_id(owner_id)
        timestamp = _utc_now_or(now)
        connection = self._begin_immediate("reserve candidate")
        try:
            campaign = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            active = connection.execute(
                "SELECT authorization_id, authorization_json, activated_at, "
                "candidate_count FROM campaign_authorizations "
                "WHERE campaign_id = ? AND authorization_id = ? "
                "AND is_active = 1",
                (campaign_id, authorization_id),
            ).fetchone()
            if active is None:
                raise StoreError(
                    f"authorization {authorization_id} is not the active "
                    f"authorization for campaign {campaign_id}"
                )

            authorization = _decode_authorization(active["authorization_json"])
            persisted_state = _coerce_state(campaign["state"])
            is_regular_stage = persisted_state == authorization.stage
            is_nested_retune = (
                persisted_state == CampaignState.FIR_APPLY
                and authorization.stage == CampaignState.I2C_RETUNE
            )
            if not is_regular_stage and not is_nested_retune:
                raise ConcurrencyError(
                    f"active authorization stage {authorization.stage.value} "
                    f"does not match current state {persisted_state.value}"
                )
            if is_nested_retune:
                outer = connection.execute(
                    "SELECT 1 FROM campaign_authorizations "
                    "WHERE campaign_id = ? AND stage = ? AND is_active = 1",
                    (campaign_id, CampaignState.FIR_APPLY.value),
                ).fetchone()
                if outer is None:
                    raise ConcurrencyError(
                        "nested I2C_RETUNE reservation requires an active outer "
                        "FIR_APPLY authorization"
                    )
            lease = connection.execute(
                "SELECT owner_id FROM campaign_execution_leases "
                "WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if lease is None or lease["owner_id"] != owner_id:
                raise ConcurrencyError(
                    f"campaign {campaign_id} execution lease is not owned by "
                    f"{owner_id}"
                )

            activated_at = _decode_datetime(active["activated_at"])
            elapsed_s = _elapsed_since(activated_at, timestamp)
            candidate_count = int(active["candidate_count"])
            if candidate_count >= authorization.max_candidates:
                raise StoreError("authorization max_candidates budget exhausted")
            if elapsed_s >= authorization.max_duration_s:
                raise StoreError("authorization max_duration budget exhausted")

            ordinal = candidate_count
            updated = connection.execute(
                "UPDATE campaign_authorizations SET candidate_count = ? "
                "WHERE campaign_id = ? AND authorization_id = ? "
                "AND is_active = 1 AND candidate_count = ?",
                (
                    candidate_count + 1,
                    campaign_id,
                    authorization_id,
                    candidate_count,
                ),
            )
            if updated.rowcount != 1:
                raise ConcurrencyError(
                    f"authorization usage changed while reserving candidate"
                )
            self._commit(connection, "reserve candidate")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("reserve candidate", exc)
        return ordinal

    def release_execution_lease(self, campaign_id: str, owner_id: str) -> None:
        """Release an execution lease only when the caller owns it."""

        campaign_id = _validate_campaign_id(campaign_id)
        owner_id = _validate_owner_id(owner_id)
        connection = self._begin_immediate("release execution lease")
        try:
            campaign = connection.execute(
                "SELECT 1 FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            lease = connection.execute(
                "SELECT owner_id FROM campaign_execution_leases "
                "WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if lease is None:
                raise ConcurrencyError(
                    f"campaign {campaign_id} has no execution lease to release"
                )
            if lease["owner_id"] != owner_id:
                raise ConcurrencyError(
                    f"campaign {campaign_id} execution lease is owned by "
                    f"{lease['owner_id']}, not {owner_id}"
                )
            deleted = connection.execute(
                "DELETE FROM campaign_execution_leases "
                "WHERE campaign_id = ? AND owner_id = ?",
                (campaign_id, owner_id),
            )
            if deleted.rowcount != 1:
                raise ConcurrencyError(
                    f"campaign {campaign_id} execution lease changed during release"
                )
            self._commit(connection, "release execution lease")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("release execution lease", exc)

    def current_state(self, campaign_id: str) -> CampaignState | None:
        campaign_id = _validate_campaign_id(campaign_id)
        connection = self._require_connection()
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to read current state: {exc}") from exc
        return None if row is None else _coerce_state(row["state"])

    def allocate_observation_sequence(self, campaign_id: str) -> int:
        """Atomically reserve the next persistent observation sequence."""

        campaign_id = _validate_campaign_id(campaign_id)
        connection = self._begin_immediate("allocate observation sequence")
        try:
            campaign = connection.execute(
                "SELECT 1 FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise UnknownCampaignError(
                    "cannot allocate observation sequence for unknown campaign "
                    f"{campaign_id}"
                )
            row = connection.execute(
                "SELECT last_sequence FROM campaign_observation_sequences "
                "WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            sequence = 1 if row is None else int(row["last_sequence"]) + 1
            if row is None:
                connection.execute(
                    "INSERT INTO campaign_observation_sequences "
                    "(campaign_id, last_sequence) VALUES (?, ?)",
                    (campaign_id, sequence),
                )
            else:
                connection.execute(
                    "UPDATE campaign_observation_sequences "
                    "SET last_sequence = ? WHERE campaign_id = ?",
                    (sequence, campaign_id),
                )
            self._commit(connection, "allocate observation sequence")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("allocate observation sequence", exc)
        return sequence

    def append_event(
        self,
        campaign_id: str,
        event_type: str,
        *,
        payload: Any = None,
        created_at: datetime | None = None,
        expected_state: CampaignState | str | None = None,
    ) -> CampaignEvent:
        """Append an audit event atomically without changing campaign state."""

        campaign_id = _validate_campaign_id(campaign_id)
        event_type = _validate_event_type(event_type)
        expected = (
            None if expected_state is None else _coerce_state(expected_state)
        )
        payload_json, restored_payload = _strict_json(payload)
        timestamp = _utc_now_or(created_at)
        encoded_timestamp = _encode_datetime(timestamp)
        connection = self._begin_immediate("append event")
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            persisted = _coerce_state(row["state"])
            if expected is not None and persisted != expected:
                raise ConcurrencyError(
                    f"expected current state {expected.value}, found {persisted.value}"
                )
            sequence = self._next_sequence(connection, campaign_id)
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    event_type,
                    persisted.value,
                    persisted.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "append event")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("append event", exc)
        return CampaignEvent(
            campaign_id=campaign_id,
            sequence=sequence,
            event_type=event_type,
            from_state=persisted,
            to_state=persisted,
            payload=restored_payload,
            created_at=timestamp,
        )

    def events(
        self, campaign_id: str, *, after_sequence: int = 0
    ) -> list[CampaignEvent]:
        campaign_id = _validate_campaign_id(campaign_id)
        if type(after_sequence) is not int or after_sequence < 0:
            raise StoreError("after_sequence must be a non-negative integer")
        connection = self._require_connection()
        try:
            rows = connection.execute(
                "SELECT campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at FROM campaign_events "
                "WHERE campaign_id = ? AND sequence > ? ORDER BY sequence ASC",
                (campaign_id, after_sequence),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"failed to read campaign events: {exc}") from exc
        return [self._event_from_row(row) for row in rows]

    def recover_state(self, campaign_id: str) -> CampaignState:
        """Fence interrupted active work into RECOVERY_REQUIRED exactly once."""

        campaign_id = _validate_campaign_id(campaign_id)
        timestamp = _utc_now_or(None)
        encoded_timestamp = _encode_datetime(timestamp)
        payload_json, restored_payload = _strict_json(
            {"reason": "process_restart_during_active_state"}
        )
        connection = self._begin_immediate("recover state")
        try:
            row = connection.execute(
                "SELECT state FROM campaign_state WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
            persisted = _coerce_state(row["state"])
            connection.execute(
                "DELETE FROM campaign_execution_leases WHERE campaign_id = ?",
                (campaign_id,),
            )
            if persisted not in ACTIVE_RESTART_STATES:
                self._commit(connection, "recover state")
                return persisted

            target = CampaignState.RECOVERY_REQUIRED
            sequence = self._next_sequence(connection, campaign_id)
            connection.execute(
                "UPDATE campaign_state SET state = ?, updated_at = ? "
                "WHERE campaign_id = ?",
                (target.value, encoded_timestamp, campaign_id),
            )
            connection.execute(
                "INSERT INTO campaign_events "
                "(campaign_id, sequence, event_type, from_state, to_state, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    sequence,
                    "RESTART_RECOVERY",
                    persisted.value,
                    target.value,
                    payload_json,
                    encoded_timestamp,
                ),
            )
            self._commit(connection, "recover state")
        except Exception as exc:
            self._rollback(connection)
            self._raise_transaction_error("recover state", exc)
        return target

    @staticmethod
    def _activate_authorization_row(
        connection: sqlite3.Connection,
        campaign_id: str,
        authorization: StageAuthorization,
        authorization_json: str,
        encoded_activated_at: str,
        *,
        allow_replacement: bool,
    ) -> None:
        active = connection.execute(
            "SELECT authorization_id, authorization_json "
            "FROM campaign_authorizations "
            "WHERE campaign_id = ? AND stage = ? AND is_active = 1",
            (campaign_id, authorization.stage.value),
        ).fetchone()
        if active is not None:
            if active["authorization_id"] == authorization.authorization_id:
                if active["authorization_json"] != authorization_json:
                    raise StoreError(
                        "active authorization payload conflicts with persisted payload"
                    )
                return
            if not allow_replacement:
                raise ConcurrencyError(
                    "conflicting active authorization "
                    f"{active['authorization_id']} for campaign {campaign_id}"
                )
            connection.execute(
                "UPDATE campaign_authorizations SET is_active = 0 "
                "WHERE campaign_id = ? AND stage = ? AND is_active = 1",
                (campaign_id, authorization.stage.value),
            )

        existing = connection.execute(
            "SELECT authorization_json FROM campaign_authorizations "
            "WHERE campaign_id = ? AND authorization_id = ?",
            (campaign_id, authorization.authorization_id),
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO campaign_authorizations "
                "(campaign_id, authorization_id, stage, authorization_json, "
                "activated_at, candidate_count, is_active) "
                "VALUES (?, ?, ?, ?, ?, 0, 1)",
                (
                    campaign_id,
                    authorization.authorization_id,
                    authorization.stage.value,
                    authorization_json,
                    encoded_activated_at,
                ),
            )
            return
        if existing["authorization_json"] != authorization_json:
            raise StoreError(
                "authorization payload conflicts with persisted payload"
            )
        connection.execute(
            "UPDATE campaign_authorizations SET is_active = 1 "
            "WHERE campaign_id = ? AND authorization_id = ?",
            (campaign_id, authorization.authorization_id),
        )

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, campaign_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
            "FROM campaign_events WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        return int(row["next_sequence"])

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> CampaignEvent:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError) as exc:
            raise StoreError("invalid persisted event payload JSON") from exc
        return CampaignEvent(
            campaign_id=row["campaign_id"],
            sequence=row["sequence"],
            event_type=row["event_type"],
            from_state=(
                None if row["from_state"] is None else _coerce_state(row["from_state"])
            ),
            to_state=_coerce_state(row["to_state"]),
            payload=payload,
            created_at=_decode_datetime(row["created_at"]),
        )

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StoreClosedError("campaign store is closed")
        return self._connection

    def _begin_immediate(self, operation: str) -> sqlite3.Connection:
        connection = self._require_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise StoreError(f"failed to {operation}: {exc}") from exc
        return connection

    @staticmethod
    def _commit(connection: sqlite3.Connection, operation: str) -> None:
        try:
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise StoreError(f"failed to {operation}: {exc}") from exc

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass

    @staticmethod
    def _raise_transaction_error(operation: str, exc: Exception) -> None:
        if isinstance(exc, StoreError):
            raise exc
        if isinstance(exc, sqlite3.Error):
            raise StoreError(f"failed to {operation}: {exc}") from exc
        raise exc


SQLiteCampaignStore = CampaignStore
