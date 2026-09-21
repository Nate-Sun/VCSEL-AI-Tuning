"""Strict, local-only campaign memory for Kimi optimization results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple
from weakref import WeakKeyDictionary

from pydantic import AwareDatetime, TypeAdapter, ValidationError, model_validator

from .contracts import (
    BrainResult,
    DecisionProposal,
    EyeSample,
    FrozenModel,
    Identifier,
    PerceptionResult,
    Sha256,
)


KIMI_PROVIDER_ID = "moonshot-kimi"


class KimiMemoryRevisionConflict(ValueError):
    """Raised when a Kimi result was produced from a stale memory snapshot."""


@dataclass(frozen=True, slots=True)
class KimiMemorySnapshot:
    """Immutable, revision-coupled view of one validated campaign journal."""

    campaign_id: str
    entries: tuple["KimiMemoryEntry", ...]
    best_entry: "KimiMemoryEntry | None"
    recent_summaries: tuple[Mapping[str, Any], ...]
    revision: Sha256
    memory_id: Sha256


class _KimiJournalLock:
    """Non-blocking OS lock for one Kimi JSONL journal."""

    def __init__(self, journal_path: Path) -> None:
        self._path = journal_path.with_suffix(journal_path.suffix + ".lock")
        self._handle = None

    def __enter__(self) -> "_KimiJournalLock":
        handle = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            handle = self._path.open("a+b")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as error:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            raise ValueError("could not acquire Kimi memory journal lock") from error
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        release_error: OSError | ImportError | None = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ImportError) as error:
            release_error = error
        try:
            handle.close()
        except OSError as error:
            release_error = release_error or error
        if release_error is not None:
            raise ValueError("could not release Kimi memory journal lock") from release_error


class KimiMemoryEntry(FrozenModel):
    """One immutable Kimi analysis associated with a complete eye sample."""

    sample: EyeSample
    perception: PerceptionResult
    decision: DecisionProposal
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def validate_provenance(self) -> "KimiMemoryEntry":
        measurement_id = self.sample.measurement_id
        if self.perception.measurement_id != measurement_id:
            raise ValueError("sample and perception measurement_id must match")
        if self.decision.based_on_measurement_id != measurement_id:
            raise ValueError("sample and decision measurement_id must match")
        if self.perception.input_hash != self.sample.raw_eye_density_ref.sha256:
            raise ValueError("perception.input_hash must equal the raw-eye hash")
        return self


class _KimiMemoryIdentity(NamedTuple):
    path: Path
    campaign_id: str
    memory_id: str


def _new_kimi_memory_identity_accessors():
    identities = WeakKeyDictionary()
    lock = RLock()

    def register(memory: object, identity: _KimiMemoryIdentity) -> None:
        with lock:
            if memory in identities:
                raise RuntimeError("Kimi campaign memory identity is already registered")
            identities[memory] = identity

    def get(memory: object) -> _KimiMemoryIdentity:
        with lock:
            try:
                return identities[memory]
            except (KeyError, TypeError):
                raise ValueError("Kimi campaign memory identity is not registered") from None

    return register, get


_register_kimi_memory_identity, _get_kimi_memory_identity = (
    _new_kimi_memory_identity_accessors()
)
del _new_kimi_memory_identity_accessors


class KimiCampaignMemory:
    """Append-only, campaign-scoped JSONL storage with bounded prompt summaries."""

    __slots__ = ("__weakref__",)

    def __init__(self, path: str | Path, *, campaign_id: str) -> None:
        canonical_path = Path(path).expanduser().resolve(strict=False)
        validated_campaign_id = TypeAdapter(Identifier).validate_python(
            campaign_id, strict=True
        )
        _register_kimi_memory_identity(
            self,
            _KimiMemoryIdentity(
                path=canonical_path,
                campaign_id=validated_campaign_id,
                memory_id=KimiCampaignMemory._derive_memory_id(
                    canonical_path, validated_campaign_id
                ),
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Kimi campaign memory identity is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Kimi campaign memory identity is immutable")

    @staticmethod
    def _derive_memory_id(path: Path, campaign_id: str) -> str:
        normalized_path = os.path.normcase(
            os.path.normpath(str(path))
        )
        identity_bytes = json.dumps(
            {
                "campaign_id": campaign_id,
                "journal_path": normalized_path,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(identity_bytes).hexdigest()

    @property
    def _path(self) -> Path:
        return _get_kimi_memory_identity(self).path

    @property
    def _campaign_id(self) -> str:
        return _get_kimi_memory_identity(self).campaign_id

    @property
    def _memory_id(self) -> str:
        return _get_kimi_memory_identity(self).memory_id

    def _require_intact_identity(self) -> None:
        try:
            identity = _get_kimi_memory_identity(self)
            path = identity.path
            campaign_id = TypeAdapter(Identifier).validate_python(
                identity.campaign_id, strict=True
            )
            stored_memory_id = TypeAdapter(Sha256).validate_python(
                identity.memory_id, strict=True
            )
            canonical_path = Path(path).expanduser().resolve(strict=False)
            derived_memory_id = KimiCampaignMemory._derive_memory_id(
                canonical_path, campaign_id
            )
        except Exception:
            raise ValueError("Kimi campaign memory identity is invalid") from None
        if path != canonical_path or stored_memory_id != derived_memory_id:
            raise ValueError("Kimi campaign memory identity mismatch")

    @property
    def campaign_id(self) -> str:
        """Campaign accepted by this read-only memory boundary."""
        self._require_intact_identity()
        return self._campaign_id

    @property
    def memory_id(self) -> str:
        """Stable identity binding this campaign to one resolved journal path."""
        self._require_intact_identity()
        return self._memory_id

    def load(self) -> tuple[KimiMemoryEntry, ...]:
        """Read and strictly validate every record in the campaign journal."""
        self._require_intact_identity()
        return KimiCampaignMemory.read_snapshot(self).entries

    def read_snapshot(self, *, limit: int = 5) -> KimiMemorySnapshot:
        """Read one lock-protected, revision-coupled campaign journal snapshot."""
        self._require_intact_identity()
        self._validate_limit(limit)
        with _KimiJournalLock(self._path):
            self._require_intact_identity()
            return self._read_snapshot_unlocked(limit=limit)

    def _read_snapshot_unlocked(self, *, limit: int) -> KimiMemorySnapshot:
        entries, journal_bytes = self._read_journal_unlocked()
        best = self._derive_best(entries)
        recent_entries = entries[-limit:] if limit else ()
        return KimiMemorySnapshot(
            campaign_id=self._campaign_id,
            entries=entries,
            best_entry=best,
            recent_summaries=tuple(
                self._freeze_summary(self._summary(entry)) for entry in recent_entries
            ),
            revision=sha256(journal_bytes).hexdigest(),
            memory_id=self._memory_id,
        )

    def _read_journal_unlocked(self) -> tuple[tuple[KimiMemoryEntry, ...], bytes]:
        if not self._path.exists():
            return (), b""
        if not self._path.is_file():
            raise ValueError("Kimi memory path must be a regular file")

        try:
            journal_bytes = self._path.read_bytes()
        except OSError as error:
            raise ValueError("could not read Kimi memory journal") from error
        if not journal_bytes:
            raise ValueError("Kimi memory journal must not be empty")
        if not journal_bytes.endswith(b"\n"):
            raise ValueError("Kimi memory journal must end each record with LF")
        try:
            journal_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Kimi memory journal must be UTF-8") from error

        parsed_lines: list[tuple[KimiMemoryEntry, bytes]] = []
        for line_number, raw_line in enumerate(
            journal_bytes.splitlines(keepends=True), start=1
        ):
            if not raw_line[:-1].strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            try:
                entry = KimiMemoryEntry.model_validate_json(raw_line, strict=True)
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL record at line {line_number}") from error
            parsed_lines.append((entry, raw_line))

        return self._validate_loaded_entries(parsed_lines), journal_bytes

    def best_entry(
        self, entries: tuple[KimiMemoryEntry, ...] | list[KimiMemoryEntry]
    ) -> KimiMemoryEntry | None:
        """Return the deterministic current best after validating its reference chain."""
        self._require_intact_identity()
        validated = self._validate_entries(entries, require_campaign=True)
        return self._derive_best(validated)

    def recent_summaries(
        self,
        entries: tuple[KimiMemoryEntry, ...] | list[KimiMemoryEntry],
        *,
        limit: int = 5,
    ) -> tuple[dict[str, Any], ...]:
        """Return a bounded, image-free prompt projection of recent memory."""
        self._require_intact_identity()
        self._validate_limit(limit)
        validated = self._validate_entries(entries, require_campaign=True)
        self._derive_best(validated)
        return tuple(self._summary(entry) for entry in validated[-limit:]) if limit else ()

    def append(self, sample: EyeSample, brain_result: BrainResult) -> KimiMemoryEntry:
        """Validate then durably append one Kimi analysis record without executing it."""
        self._require_intact_identity()
        validated_sample = self._strict_model(EyeSample, sample, "sample")
        validated_result = self._strict_model(BrainResult, brain_result, "brain_result")
        if validated_sample.campaign_id != self._campaign_id:
            raise ValueError("sample campaign_id does not match this Kimi memory")
        if validated_result.perception.provider_id != KIMI_PROVIDER_ID:
            raise ValueError("Kimi memory requires a Kimi perception")

        entry = KimiMemoryEntry(
            sample=validated_sample,
            perception=validated_result.perception,
            decision=validated_result.decision,
            recorded_at=datetime.now(timezone.utc),
        )
        with _KimiJournalLock(self._path):
            self._require_intact_identity()
            snapshot = self._read_snapshot_unlocked(limit=5)
            expected_revision = validated_result.perception.memory_revision
            if expected_revision is None:
                raise KimiMemoryRevisionConflict("Kimi memory revision is required for append")
            if expected_revision != snapshot.revision:
                raise KimiMemoryRevisionConflict("Kimi memory revision changed before append")
            expected_memory_id = validated_result.perception.memory_id
            if expected_memory_id is None:
                raise ValueError("Kimi memory_id is required for live append")
            if expected_memory_id != self._memory_id:
                raise ValueError("Kimi memory_id does not match this journal")
            if any(
                existing.sample.measurement_id == entry.sample.measurement_id
                for existing in snapshot.entries
            ):
                raise ValueError("duplicate measurement_id in Kimi memory")
            self._derive_best((*snapshot.entries, entry))
            encoded = json.dumps(
                entry.model_dump(mode="json", round_trip=True, warnings=False),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            self._require_intact_identity()
            try:
                with self._path.open("a", encoding="utf-8", newline="\n") as journal:
                    journal.write(encoded + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
            except OSError as error:
                raise ValueError("could not durably append Kimi memory record") from error
        return entry

    def _validate_entries(
        self,
        entries: tuple[KimiMemoryEntry, ...] | list[KimiMemoryEntry],
        *,
        require_campaign: bool,
    ) -> tuple[KimiMemoryEntry, ...]:
        if not isinstance(entries, (tuple, list)):
            raise ValueError("entries must be a tuple or list of KimiMemoryEntry values")
        validated: list[KimiMemoryEntry] = []
        measurement_ids: set[str] = set()
        for index, entry in enumerate(entries, start=1):
            strict_entry = self._strict_model(KimiMemoryEntry, entry, f"entry {index}")
            if require_campaign and strict_entry.sample.campaign_id != self._campaign_id:
                raise ValueError("Kimi memory contains an entry from another campaign")
            measurement_id = strict_entry.sample.measurement_id
            if measurement_id in measurement_ids:
                raise ValueError("duplicate measurement_id in Kimi memory")
            measurement_ids.add(measurement_id)
            validated.append(strict_entry)

        return tuple(validated)

    def _validate_loaded_entries(
        self, parsed_lines: list[tuple[KimiMemoryEntry, bytes]]
    ) -> tuple[KimiMemoryEntry, ...]:
        """Validate path identity and the exact byte prefix before each entry.

        This lineage chain cannot authenticate changes confined to the final line
        without an external anchor. The snapshot revision is the live
        optimistic-concurrency anchor for the full current journal bytes.
        """
        measurement_ids: set[str] = set()
        entries: list[KimiMemoryEntry] = []
        prefix_hasher = sha256()
        for entry, raw_line in parsed_lines:
            if entry.sample.campaign_id != self._campaign_id:
                raise ValueError("Kimi memory contains an entry from another campaign")
            measurement_id = entry.sample.measurement_id
            if measurement_id in measurement_ids:
                raise ValueError("duplicate measurement_id in Kimi memory")
            measurement_ids.add(measurement_id)
            perception = entry.perception
            if perception.memory_id is None or perception.memory_revision is None:
                raise ValueError("Kimi memory entry requires complete memory provenance")
            if perception.memory_id != self._memory_id:
                raise ValueError("Kimi memory entry memory_id does not match this journal")
            if perception.memory_revision != prefix_hasher.hexdigest():
                raise ValueError("Kimi memory entry revision does not match journal prefix")
            prefix_hasher.update(raw_line)
            entries.append(entry)
        validated = tuple(entries)
        self._derive_best(validated)
        return validated

    def _derive_best(
        self, entries: tuple[KimiMemoryEntry, ...]
    ) -> KimiMemoryEntry | None:
        best: KimiMemoryEntry | None = None
        for entry in entries:
            perception = entry.perception
            if best is None:
                if perception.reference_measurement_id is not None:
                    raise ValueError("entries before a best exists must not reference a measurement")
            elif perception.reference_measurement_id != best.sample.measurement_id:
                raise ValueError("entry reference_measurement_id must match the current best")

            if self._is_best_eligible(entry) and (
                best is None or perception.better_than_reference is True
            ):
                best = entry
        return best

    @staticmethod
    def _validate_limit(limit: object) -> int:
        if type(limit) is not int or not 0 <= limit <= 5:
            raise ValueError("limit must be an integer from 0 through 5")
        return limit

    @classmethod
    def _freeze_summary(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return MappingProxyType(
            {key: cls._freeze_summary_value(item) for key, item in value.items()}
        )

    @classmethod
    def _freeze_summary_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls._freeze_summary(value)
        if isinstance(value, tuple):
            return tuple(cls._freeze_summary_value(item) for item in value)
        return value

    @staticmethod
    def _strict_model(model_type, value: object, name: str):
        if not isinstance(value, model_type):
            raise ValueError(f"{name} must be a {model_type.__name__}")
        try:
            return model_type.model_validate(
                value.model_dump(mode="python", round_trip=True, warnings=False), strict=True
            )
        except (ValidationError, ValueError, TypeError) as error:
            raise ValueError(f"invalid {name}") from error

    @staticmethod
    def _is_best_eligible(entry: KimiMemoryEntry) -> bool:
        perception = entry.perception
        return (
            perception.provider_id == KIMI_PROVIDER_ID
            and entry.decision.provider_id == KIMI_PROVIDER_ID
            and perception.valid
            and not perception.ood
        )

    @staticmethod
    def _summary(entry: KimiMemoryEntry) -> dict[str, Any]:
        perception = entry.perception
        decision = entry.decision
        metrics = entry.sample.metrics
        return {
            "measurement_id": entry.sample.measurement_id,
            "quality_score": perception.quality_score,
            "confidence": perception.confidence,
            "valid": perception.valid,
            "ood": perception.ood,
            "reference_measurement_id": perception.reference_measurement_id,
            "better_than_reference": perception.better_than_reference,
            "action": decision.action.value,
            "candidate_id": decision.candidate_id,
            "reason_codes": decision.reason_codes,
            "readback_params": dict(entry.sample.readback_params),
            "metrics": {
                "tdecq_db": metrics.tdecq_db,
                "outer_er_chan1a_db": metrics.outer_er_chan1a_db,
                "rlm": metrics.rlm,
                "outer_oma": metrics.outer_oma,
                "other_metrics": dict(metrics.other_metrics),
            },
        }
