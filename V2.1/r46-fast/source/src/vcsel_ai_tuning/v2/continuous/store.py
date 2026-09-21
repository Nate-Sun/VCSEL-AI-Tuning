"""Atomic isolated persistence for continuous optimization campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


class ContinuousStoreError(RuntimeError):
    pass


_SECRET_KEYS = {
    "api_key",
    "authorization",
    "moonshot_api_key",
}


def _scrub(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).replace("-", "_").lower() in _SECRET_KEYS
                or str(key).lower().startswith("bearer")
                else _scrub(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _scrub(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


class ContinuousCampaignStore:
    def __init__(self, result_dir: Path | str) -> None:
        self._root = Path(result_dir).resolve()

    @staticmethod
    def _validate_campaign_id(campaign_id: str) -> str:
        if (
            not isinstance(campaign_id, str)
            or not campaign_id
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in campaign_id)
        ):
            raise ContinuousStoreError("campaign_id must be a safe identifier")
        return campaign_id

    def _campaign_root(self, campaign_id: str) -> Path:
        return self._root / self._validate_campaign_id(campaign_id)

    def initialize(self, campaign_id: str, metadata: Mapping[str, object]) -> None:
        root = self._campaign_root(campaign_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "campaign.json"
        desired = _scrub(metadata)
        if path.exists():
            existing = self._read_json(path, label="campaign metadata")
            if existing != desired:
                if (
                    "target_tdecq_db" not in existing
                    and isinstance(desired, Mapping)
                    and "target_tdecq_db" in desired
                ):
                    raise ContinuousStoreError(
                        "campaign metadata predates target_tdecq_db; "
                        "migrate the artifacts or start a new campaign"
                    )
                raise ContinuousStoreError("campaign has conflicting immutable metadata")
            return
        _atomic_write(path, desired)

    def load_state(self, campaign_id: str) -> dict[str, Any]:
        return self._read_json(
            self._campaign_root(campaign_id) / "state.json",
            label="state",
        )

    def save_state(self, campaign_id: str, state: Mapping[str, object]) -> None:
        _atomic_write(self._campaign_root(campaign_id) / "state.json", state)

    def append_event(
        self,
        campaign_id: str,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        if not isinstance(event_type, str) or not event_type:
            raise ContinuousStoreError("event_type must be nonblank")
        path = self._campaign_root(campaign_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(_json_bytes({"event_type": event_type, "payload": payload}))
            handle.flush()
            os.fsync(handle.fileno())

    def load_events(self, campaign_id: str) -> tuple[dict[str, Any], ...]:
        path = self._campaign_root(campaign_id) / "events.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            return tuple(json.loads(line) for line in lines if line)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContinuousStoreError("event log is corrupt") from error

    def save_round(
        self,
        campaign_id: str,
        round_index: int,
        record: Mapping[str, object],
    ) -> None:
        if type(round_index) is not int or round_index <= 0:
            raise ContinuousStoreError("round_index must be positive")
        path = (
            self._campaign_root(campaign_id)
            / "rounds"
            / f"{round_index:04d}"
            / "round.json"
        )
        _atomic_write(path, record)

    def save_acceptance(self, campaign_id: str, record: Mapping[str, object]) -> None:
        _atomic_write(
            self._campaign_root(campaign_id) / "acceptance" / "acceptance.json",
            record,
        )

    def save_recovery(self, campaign_id: str, record: Mapping[str, object]) -> None:
        _atomic_write(
            self._campaign_root(campaign_id) / "recovery" / "recovery.json",
            record,
        )

    @staticmethod
    def _read_json(path: Path, *, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContinuousStoreError(f"{label} is unavailable or corrupt") from error
        if not isinstance(value, dict):
            raise ContinuousStoreError(f"{label} must be an object")
        return value
