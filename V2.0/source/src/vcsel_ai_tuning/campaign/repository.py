from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Self
from uuid import uuid4


class RepositoryError(RuntimeError):
    """Base error for the campaign artifact repository."""


class RepositorySerializationError(RepositoryError):
    """Raised when data cannot be represented as strict finite JSON."""


class RepositoryIntegrityError(RepositoryError):
    """Raised when an indexed immutable artifact is missing or changed."""


class RepositoryClosedError(RepositoryError):
    """Raised when a closed repository is used."""


class ImmutableArtifactError(RepositoryError):
    """Raised when a caller attempts to replace immutable content."""


class DuplicateObservationError(RepositoryError):
    """Raised when an observation identifier has already been published."""


def _serialize_public_connection_calls(cls: type[Any]) -> type[Any]:
    """Hold one reentrant lock for complete repository operations."""

    for name, method in tuple(vars(cls).items()):
        if name.startswith("_") or not callable(method):
            continue

        @wraps(method)
        def guarded(self: Any, *args: Any, __method: Any = method, **kwargs: Any):
            with self._connection_lock:
                return __method(self, *args, **kwargs)

        setattr(cls, name, guarded)
    return cls


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    relative_path: str
    sha256: str
    size: int
    media_type: str
    category: str


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    category: str
    relative_path: str
    content: bytes
    media_type: str = "application/octet-stream"

    @classmethod
    def from_bytes(
        cls,
        category: str,
        relative_path: str,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> Self:
        if not isinstance(content, bytes):
            raise RepositoryError("artifact bytes content must be bytes")
        return cls(category, relative_path, content, media_type)

    @classmethod
    def from_text(
        cls,
        category: str,
        relative_path: str,
        content: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
    ) -> Self:
        if not isinstance(content, str):
            raise RepositoryError("artifact text content must be a string")
        return cls(category, relative_path, content.encode("utf-8"), media_type)

    @classmethod
    def from_json(
        cls, category: str, relative_path: str, content: Any
    ) -> Self:
        return cls(
            category,
            relative_path,
            _strict_json_bytes(content),
            "application/json",
        )


@dataclass(frozen=True, slots=True)
class PublishedObservation:
    observation_id: str
    payload: Any
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(slots=True)
class _StagedArtifact:
    record: ArtifactRecord
    target: Path
    temporary: Path | None
    promoted: bool = False


CATEGORY_NAMES = (
    "config",
    "plans",
    "receipts",
    "events",
    "observations",
    "raw_responses",
    "eye_artifacts",
    "optimizer_snapshots",
    "exports",
    "reports",
)
CATEGORIES = frozenset(CATEGORY_NAMES)
_SUPPORT_DIRECTORIES = ("staging", "quarantine")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_WINDOWS_REPARSE_ATTRIBUTE = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400
)
_DATABASE_NAME = "artifact_index.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    relative_path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    size INTEGER NOT NULL CHECK(size >= 0),
    media_type TEXT NOT NULL,
    category TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    observation_path TEXT NOT NULL UNIQUE,
    FOREIGN KEY (observation_path) REFERENCES artifacts(relative_path)
);

CREATE TABLE IF NOT EXISTS observation_artifacts (
    observation_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    position INTEGER NOT NULL CHECK(position >= 0),
    PRIMARY KEY (observation_id, relative_path),
    UNIQUE (observation_id, position),
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (relative_path) REFERENCES artifacts(relative_path)
);
"""


def _strict_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RepositorySerializationError(
            f"value must be strict finite JSON: {exc}"
        ) from exc
    return encoded.encode("utf-8")


def _validate_identifier(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise RepositoryError(
            f"{name} must contain only letters, digits, '.', '_', and '-'"
        )
    _validate_windows_segment(value, context=name)
    return value


def _validate_category(category: str) -> str:
    if category not in CATEGORIES:
        raise RepositoryError(f"unknown artifact category: {category!r}")
    return category


def _validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RepositoryError(
            "artifact relative path must be non-empty and use '/' separators"
        )
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(":" in part for part in raw_parts)
    ):
        raise RepositoryError(
            "artifact relative path must not be absolute or contain traversal"
        )
    for part in raw_parts:
        _validate_windows_segment(part, context="artifact relative path")
    return path


def _validate_windows_segment(segment: str, *, context: str) -> None:
    if segment.endswith((".", " ")):
        raise RepositoryError(
            f"{context} contains a Windows-invalid trailing dot or space"
        )
    reserved_base = segment.split(".", 1)[0].upper()
    if reserved_base in _WINDOWS_RESERVED_NAMES:
        raise RepositoryError(
            f"{context} contains a Windows reserved name: {segment!r}"
        )


def _path_is_reparse_point(path: Path) -> bool:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RepositoryIntegrityError(
            f"cannot inspect path for symlink or reparse point: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode):
        return True
    if getattr(path_stat, "st_file_attributes", 0) & _WINDOWS_REPARSE_ATTRIBUTE:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _reject_reparse_point(path: Path) -> None:
    if _path_is_reparse_point(path):
        raise RepositoryIntegrityError(
            f"symlink, junction, or reparse point is not allowed: {path}"
        )


def _record_for(
    category: str, relative_path: PurePosixPath, content: bytes, media_type: str
) -> ArtifactRecord:
    return ArtifactRecord(
        relative_path=f"{category}/{relative_path.as_posix()}",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        media_type=media_type,
        category=category,
    )


@_serialize_public_connection_calls
class CampaignRepository:
    """Immutable, hash-indexed artifacts for one campaign run.

    The SQLite index is the visibility boundary for observations. Files can be
    promoted shortly before their transaction commits, but readers only consult
    committed observation rows, and failed promotions are fenced into quarantine.
    A repository instance owns one SQLite connection. It may cross runtime
    worker threads only while the campaign execution lease serializes writes;
    separate instances still coordinate through SQLite write locks.
    """

    def __init__(self, root: str | Path, campaign_id: str) -> None:
        self.root = Path(root)
        self.campaign_id = _validate_identifier(campaign_id, name="campaign_id")
        self.run_dir = self.root / self.campaign_id
        self.database_path = self.run_dir / _DATABASE_NAME
        self._connection_lock = RLock()
        self._connection: sqlite3.Connection | None = None
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            _reject_reparse_point(self.run_dir)
            for directory in (*CATEGORY_NAMES, *_SUPPORT_DIRECTORIES):
                directory_path = self.run_dir / directory
                directory_path.mkdir(exist_ok=True)
                _reject_reparse_point(directory_path)
            _reject_reparse_point(self.database_path)
            connection = sqlite3.connect(
                self.database_path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(_SCHEMA)
            self._connection = connection
            self.recover_orphans()
        except Exception as exc:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if isinstance(exc, RepositoryError):
                raise
            raise RepositoryError(
                f"failed to initialize campaign repository at {self.run_dir}: {exc}"
            ) from exc

    def __enter__(self) -> Self:
        self._require_connection()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def write_json(
        self, category: str, relative_path: str, value: Any
    ) -> ArtifactRecord:
        return self._write_immutable(
            category,
            relative_path,
            _strict_json_bytes(value),
            "application/json",
        )

    def write_text(
        self,
        category: str,
        relative_path: str,
        value: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
    ) -> ArtifactRecord:
        if not isinstance(value, str):
            raise RepositoryError("artifact text value must be a string")
        return self._write_immutable(
            category, relative_path, value.encode("utf-8"), media_type
        )

    def write_bytes(
        self,
        category: str,
        relative_path: str,
        value: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRecord:
        if not isinstance(value, bytes):
            raise RepositoryError("artifact byte value must be bytes")
        return self._write_immutable(category, relative_path, value, media_type)

    def publish_observation(
        self,
        observation_id: str,
        payload: Any,
        artifacts: tuple[ArtifactSpec, ...] | list[ArtifactSpec],
    ) -> PublishedObservation:
        self._require_connection()
        observation_id = _validate_identifier(
            observation_id, name="observation_id"
        )
        if self._observation_exists(observation_id):
            raise DuplicateObservationError(
                f"observation {observation_id!r} is already published"
            )

        payload_bytes = _strict_json_bytes(payload)
        restored_payload = json.loads(payload_bytes)
        specifications = [
            ArtifactSpec(
                "observations",
                f"{observation_id}.json",
                payload_bytes,
                "application/json",
            )
        ]
        try:
            specifications.extend(artifacts)
        except TypeError as exc:
            raise RepositoryError("artifacts must be an iterable of ArtifactSpec") from exc

        prepared: list[tuple[ArtifactSpec, PurePosixPath, ArtifactRecord]] = []
        seen_paths: set[str] = set()
        for specification in specifications:
            if not isinstance(specification, ArtifactSpec):
                raise RepositoryError("all published artifacts must be ArtifactSpec")
            category = _validate_category(specification.category)
            relative = _validate_relative_path(specification.relative_path)
            if not isinstance(specification.content, bytes):
                raise RepositoryError("ArtifactSpec content must be bytes")
            self._validate_media_type(specification.media_type)
            record = _record_for(
                category,
                relative,
                specification.content,
                specification.media_type,
            )
            if record.relative_path in seen_paths:
                raise RepositoryError(
                    f"duplicate artifact path in publication: {record.relative_path}"
                )
            seen_paths.add(record.relative_path)
            prepared.append((specification, relative, record))

        connection = self._require_connection()
        staged: list[_StagedArtifact] = []
        outcome_unknown = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM observations WHERE observation_id = ?",
                (observation_id,),
            ).fetchone() is not None:
                raise DuplicateObservationError(
                    f"observation {observation_id!r} is already published"
                )

            for specification, relative, record in prepared:
                staged.append(
                    self._stage_artifact(relative, specification.content, record)
                )
            for item in staged:
                self._promote_staged(item, connection)
            for item in staged:
                self._insert_or_validate_artifact(connection, item.record)

            observation_path = staged[0].record.relative_path
            connection.execute(
                "INSERT INTO observations "
                "(observation_id, payload_json, observation_path) VALUES (?, ?, ?)",
                (observation_id, payload_bytes.decode("utf-8"), observation_path),
            )
            connection.executemany(
                "INSERT INTO observation_artifacts "
                "(observation_id, relative_path, position) VALUES (?, ?, ?)",
                [
                    (observation_id, item.record.relative_path, position)
                    for position, item in enumerate(staged)
                ],
            )
            try:
                self._commit_transaction(connection)
            except Exception as commit_error:
                try:
                    committed = self._observation_was_committed(
                        observation_id,
                        payload_bytes.decode("utf-8"),
                        tuple(item.record for item in staged),
                    )
                except Exception as verification_error:
                    outcome_unknown = True
                    raise RepositoryError(
                        f"commit outcome unknown for observation {observation_id!r}: "
                        f"{verification_error}"
                    ) from verification_error
                if not committed:
                    raise commit_error
        except Exception as exc:
            self._rollback_transaction(connection)
            quarantine_errors = self._quarantine_staged(
                staged, include_promoted=not outcome_unknown
            )
            if isinstance(exc, DuplicateObservationError):
                raise
            detail = f": {'; '.join(quarantine_errors)}" if quarantine_errors else ""
            raise RepositoryError(
                f"failed to publish observation {observation_id!r}: {exc}{detail}"
            ) from exc

        return PublishedObservation(
            observation_id=observation_id,
            payload=restored_payload,
            artifacts=tuple(item.record for item in staged),
        )

    def get_observation(self, observation_id: str) -> PublishedObservation | None:
        self._require_connection()
        observation_id = _validate_identifier(
            observation_id, name="observation_id"
        )
        try:
            with self._read_connection() as connection:
                row = connection.execute(
                    "SELECT payload_json, observation_path FROM observations "
                    "WHERE observation_id = ?",
                    (observation_id,),
                ).fetchone()
                if row is None:
                    return None
                artifact_rows = connection.execute(
                    "SELECT a.relative_path, a.sha256, a.size, a.media_type, "
                    "a.category FROM observation_artifacts AS oa "
                    "JOIN artifacts AS a ON a.relative_path = oa.relative_path "
                    "WHERE oa.observation_id = ? ORDER BY oa.position",
                    (observation_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to read observation {observation_id!r}: {exc}"
            ) from exc
        expected_observation_path = f"observations/{observation_id}.json"
        if row["observation_path"] != expected_observation_path:
            raise RepositoryIntegrityError(
                f"observation {observation_id!r} has an unexpected observation path"
            )
        records = tuple(self._record_from_row(item) for item in artifact_rows)
        observation_records = tuple(
            record
            for record in records
            if record.relative_path == expected_observation_path
            and record.category == "observations"
        )
        if len(observation_records) != 1:
            raise RepositoryIntegrityError(
                f"observation {observation_id!r} does not reference its observation artifact"
            )
        for record in records:
            self._verify_record(record)
        observation_path = self._path_for_record(observation_records[0])
        try:
            artifact_bytes = observation_path.read_bytes()
            database_payload = row["payload_json"]
            if not isinstance(database_payload, str):
                raise TypeError("payload_json is not text")
            decoded_payload = json.loads(database_payload)
            canonical_payload = _strict_json_bytes(decoded_payload)
            if database_payload.encode("utf-8") != canonical_payload:
                raise ValueError("payload_json is not canonical JSON")
            if canonical_payload != artifact_bytes:
                raise ValueError("payload_json disagrees with observation artifact")
            payload = json.loads(artifact_bytes)
        except (OSError, TypeError, ValueError, RepositorySerializationError) as exc:
            raise RepositoryIntegrityError(
                f"observation {observation_id!r} has invalid or inconsistent payload JSON"
            ) from exc
        return PublishedObservation(observation_id, payload, records)

    def list_observations(self) -> list[PublishedObservation]:
        self._require_connection()
        try:
            with self._read_connection() as connection:
                rows = connection.execute(
                    "SELECT observation_id FROM observations ORDER BY observation_id"
                ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"failed to list observations: {exc}") from exc
        observations: list[PublishedObservation] = []
        for row in rows:
            observation = self.get_observation(row["observation_id"])
            if observation is None:
                raise RepositoryIntegrityError(
                    f"observation {row['observation_id']!r} disappeared from its index"
                )
            observations.append(observation)
        return observations

    def get_artifact(self, relative_path: str) -> ArtifactRecord | None:
        self._require_connection()
        if not isinstance(relative_path, str):
            raise RepositoryError("indexed artifact path must be a string")
        try:
            with self._read_connection() as connection:
                row = connection.execute(
                    "SELECT relative_path, sha256, size, media_type, category "
                    "FROM artifacts WHERE relative_path = ?",
                    (relative_path,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"failed to read artifact index: {exc}") from exc
        if row is None:
            return None
        record = self._record_from_row(row)
        self._verify_record(record)
        return record

    def recover_orphans(self) -> tuple[Path, ...]:
        """Quarantine staging residue, unindexed files, and changed artifacts."""

        connection = self._require_connection()
        quarantined: list[Path] = []
        integrity_problems: list[str] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            indexed = {
                row["relative_path"]: self._record_from_row(row)
                for row in connection.execute(
                    "SELECT relative_path, sha256, size, media_type, category "
                    "FROM artifacts"
                ).fetchall()
            }
            seen_indexed: set[str] = set()
            staging = self.run_dir / "staging"
            for path in self._walk_regular_files_safely(staging):
                quarantined.append(self._quarantine(path))

            for category in CATEGORY_NAMES:
                category_dir = self.run_dir / category
                for path in self._walk_regular_files_safely(category_dir):
                    relative_path = path.relative_to(self.run_dir).as_posix()
                    record = indexed.get(relative_path)
                    if record is None:
                        quarantined.append(self._quarantine(path))
                        continue
                    seen_indexed.add(relative_path)
                    if record.category != category:
                        quarantined.append(self._quarantine(path))
                        integrity_problems.append(
                            f"category mismatch for {relative_path}"
                        )
                        continue
                    if not self._path_matches_record(path, record):
                        quarantined.append(self._quarantine(path))
                        integrity_problems.append(
                            f"hash or size mismatch for {relative_path}"
                        )

            for relative_path in sorted(indexed.keys() - seen_indexed):
                integrity_problems.append(
                    f"indexed artifact is missing: {relative_path}"
                )
            self._commit_transaction(connection)
        except Exception as exc:
            self._rollback_transaction(connection)
            if isinstance(exc, RepositoryError):
                raise
            if isinstance(exc, sqlite3.Error):
                raise RepositoryError(
                    f"failed to recover artifact repository: {exc}"
                ) from exc
            raise RepositoryError(
                f"failed to recover repository orphans: {exc}"
            ) from exc
        if integrity_problems:
            raise RepositoryIntegrityError(
                "repository index integrity failure: "
                + "; ".join(integrity_problems)
            )
        return tuple(quarantined)

    def _walk_regular_files_safely(self, root: Path) -> Iterator[Path]:
        """Walk without following symlinks, junctions, or other reparse points."""

        _reject_reparse_point(root)
        pending = [root]
        while pending:
            directory = pending.pop()
            _reject_reparse_point(directory)
            try:
                with os.scandir(directory) as scan:
                    entries = sorted(scan, key=lambda entry: entry.name)
            except OSError as exc:
                raise RepositoryIntegrityError(
                    f"cannot safely scan artifact directory {directory}: {exc}"
                ) from exc
            for entry in entries:
                path = Path(entry.path)
                _reject_reparse_point(path)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        yield path
                    else:
                        raise RepositoryIntegrityError(
                            f"unsupported filesystem entry in artifact tree: {path}"
                        )
                except OSError as exc:
                    raise RepositoryIntegrityError(
                        f"cannot inspect artifact entry {path}: {exc}"
                    ) from exc

    def _write_immutable(
        self,
        category: str,
        relative_path: str,
        content: bytes,
        media_type: str,
    ) -> ArtifactRecord:
        connection = self._require_connection()
        category = _validate_category(category)
        relative = _validate_relative_path(relative_path)
        self._validate_media_type(media_type)
        record = _record_for(category, relative, content, media_type)
        existing = self._existing_immutable(record)
        if existing is not None:
            return existing

        staged: _StagedArtifact | None = None
        outcome_unknown = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._existing_immutable(record, connection=connection)
            if existing is not None:
                result = existing
            else:
                staged = self._stage_artifact(relative, content, record)
                self._promote_staged(staged, connection)
                self._insert_or_validate_artifact(connection, record)
                result = record
            try:
                self._commit_transaction(connection)
            except Exception as commit_error:
                try:
                    committed = self._artifact_was_committed(record)
                except Exception as verification_error:
                    outcome_unknown = True
                    raise RepositoryError(
                        f"commit outcome unknown for artifact "
                        f"{record.relative_path}: {verification_error}"
                    ) from verification_error
                if not committed:
                    raise commit_error
            return result
        except Exception as exc:
            self._rollback_transaction(connection)
            quarantine_errors = self._quarantine_staged(
                [] if staged is None else [staged],
                include_promoted=not outcome_unknown,
            )
            if isinstance(exc, (ImmutableArtifactError, RepositoryIntegrityError)):
                raise
            detail = f": {'; '.join(quarantine_errors)}" if quarantine_errors else ""
            raise RepositoryError(
                f"failed to publish immutable artifact {record.relative_path}: {exc}{detail}"
            ) from exc

    def _stage_artifact(
        self,
        relative: PurePosixPath,
        content: bytes,
        record: ArtifactRecord,
    ) -> _StagedArtifact:
        target = self._contained_artifact_path(record.category, relative)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RepositoryError(
                f"failed to create artifact directory {target.parent}: {exc}"
            ) from exc
        target = self._contained_artifact_path(record.category, relative)

        existing = self._existing_immutable(record)
        if existing is not None:
            return _StagedArtifact(record, target, None)

        temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
        try:
            _reject_reparse_point(temporary)
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if not self._path_matches_record(temporary, record):
                raise RepositoryIntegrityError(
                    f"staged bytes failed hash verification for {record.relative_path}"
                )
        except Exception as exc:
            if temporary.exists():
                try:
                    self._quarantine(temporary)
                except RepositoryError:
                    pass
            if isinstance(exc, RepositoryError):
                raise
            raise RepositoryError(
                f"failed to stage artifact {record.relative_path}: {exc}"
            ) from exc
        return _StagedArtifact(record, target, temporary)

    def _promote_staged(
        self, item: _StagedArtifact, connection: sqlite3.Connection
    ) -> None:
        expected_target = self._path_for_record(item.record)
        if item.target != expected_target:
            raise RepositoryIntegrityError(
                f"staged artifact escaped its category root: "
                f"{item.record.relative_path}"
            )
        if item.temporary is not None:
            if item.temporary.parent != item.target.parent:
                raise RepositoryIntegrityError(
                    f"staged artifact temp is outside its target directory: "
                    f"{item.record.relative_path}"
                )
            _reject_reparse_point(item.temporary)
        row = connection.execute(
            "SELECT relative_path, sha256, size, media_type, category "
            "FROM artifacts WHERE relative_path = ?",
            (item.record.relative_path,),
        ).fetchone()
        if item.target.exists():
            if not self._path_matches_record(item.target, item.record):
                raise ImmutableArtifactError(
                    f"immutable artifact already has different content: "
                    f"{item.record.relative_path}"
                )
            if row is None or self._record_from_row(row) != item.record:
                raise RepositoryIntegrityError(
                    f"artifact file/index mismatch: {item.record.relative_path}"
                )
            self._discard_temporary(item)
            return
        if row is not None:
            raise RepositoryIntegrityError(
                f"indexed artifact is missing: {item.record.relative_path}"
            )
        if item.temporary is None:
            raise RepositoryIntegrityError(
                f"staged artifact is unavailable: {item.record.relative_path}"
            )
        try:
            item.temporary.replace(item.target)
        except OSError as exc:
            raise RepositoryError(
                f"failed to atomically promote {item.record.relative_path}: {exc}"
            ) from exc
        item.promoted = True
        item.temporary = None
        self._fsync_directory_best_effort(item.target.parent)
        if not self._path_matches_record(item.target, item.record):
            raise RepositoryIntegrityError(
                f"promoted artifact failed hash verification: {item.record.relative_path}"
            )

    def _existing_immutable(
        self,
        expected: ArtifactRecord,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ArtifactRecord | None:
        connection = self._require_connection() if connection is None else connection
        target = self._path_for_record(expected)
        row = connection.execute(
            "SELECT relative_path, sha256, size, media_type, category "
            "FROM artifacts WHERE relative_path = ?",
            (expected.relative_path,),
        ).fetchone()
        if not target.exists() and row is None:
            return None
        if target.exists() and not self._path_matches_record(target, expected):
            raise ImmutableArtifactError(
                f"immutable artifact already has different content: "
                f"{expected.relative_path}"
            )
        if row is None:
            raise RepositoryIntegrityError(
                f"artifact exists without index record: {expected.relative_path}"
            )
        persisted = self._record_from_row(row)
        if persisted != expected:
            raise ImmutableArtifactError(
                f"immutable artifact metadata differs: {expected.relative_path}"
            )
        if not target.exists():
            raise RepositoryIntegrityError(
                f"indexed artifact is missing: {expected.relative_path}"
            )
        return persisted

    @staticmethod
    def _insert_or_validate_artifact(
        connection: sqlite3.Connection, record: ArtifactRecord
    ) -> None:
        row = connection.execute(
            "SELECT relative_path, sha256, size, media_type, category "
            "FROM artifacts WHERE relative_path = ?",
            (record.relative_path,),
        ).fetchone()
        if row is not None:
            if CampaignRepository._record_from_row(row) != record:
                raise ImmutableArtifactError(
                    f"immutable artifact index differs: {record.relative_path}"
                )
            return
        connection.execute(
            "INSERT INTO artifacts "
            "(relative_path, sha256, size, media_type, category) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.relative_path,
                record.sha256,
                record.size,
                record.media_type,
                record.category,
            ),
        )

    def _observation_exists(self, observation_id: str) -> bool:
        try:
            with self._read_connection() as connection:
                return connection.execute(
                    "SELECT 1 FROM observations WHERE observation_id = ?",
                    (observation_id,),
                ).fetchone() is not None
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to check observation {observation_id!r}: {exc}"
            ) from exc

    def _observation_was_committed(
        self,
        observation_id: str,
        payload_json: str,
        expected_records: tuple[ArtifactRecord, ...],
    ) -> bool:
        """Resolve an ambiguous COMMIT using a fresh, independently closed read."""

        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT payload_json, observation_path FROM observations "
                "WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            if row is None:
                return False
            expected_observation_path = f"observations/{observation_id}.json"
            if (
                row["payload_json"] != payload_json
                or row["observation_path"] != expected_observation_path
            ):
                raise RepositoryIntegrityError(
                    f"ambiguous commit left unexpected observation metadata: "
                    f"{observation_id}"
                )
            artifact_rows = connection.execute(
                "SELECT a.relative_path, a.sha256, a.size, a.media_type, "
                "a.category FROM observation_artifacts AS oa "
                "JOIN artifacts AS a ON a.relative_path = oa.relative_path "
                "WHERE oa.observation_id = ? ORDER BY oa.position",
                (observation_id,),
            ).fetchall()
        persisted_records = tuple(
            self._record_from_row(item) for item in artifact_rows
        )
        if persisted_records != expected_records:
            raise RepositoryIntegrityError(
                f"ambiguous commit left an incomplete artifact index for "
                f"observation {observation_id}"
            )
        for record in persisted_records:
            self._verify_record(record)
        return True

    def _artifact_was_committed(self, expected: ArtifactRecord) -> bool:
        """Resolve an ambiguous artifact COMMIT using a fresh read connection."""

        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT relative_path, sha256, size, media_type, category "
                "FROM artifacts WHERE relative_path = ?",
                (expected.relative_path,),
            ).fetchone()
        if row is None:
            return False
        if self._record_from_row(row) != expected:
            raise RepositoryIntegrityError(
                f"ambiguous commit left unexpected artifact metadata: "
                f"{expected.relative_path}"
            )
        self._verify_record(expected)
        return True

    def _verify_record(self, record: ArtifactRecord) -> None:
        path = self._path_for_record(record)
        if not path.is_file():
            raise RepositoryIntegrityError(
                f"indexed artifact is missing: {record.relative_path}"
            )
        if not self._path_matches_record(path, record):
            raise RepositoryIntegrityError(
                f"indexed artifact hash or size mismatch: {record.relative_path}"
            )

    def _path_for_record(self, record: ArtifactRecord) -> Path:
        persisted = PurePosixPath(record.relative_path)
        parts = persisted.parts
        if len(parts) < 2 or parts[0] != record.category:
            raise RepositoryIntegrityError(
                f"invalid indexed artifact path: {record.relative_path}"
            )
        _validate_category(record.category)
        relative = _validate_relative_path("/".join(parts[1:]))
        return self._contained_artifact_path(record.category, relative)

    def _contained_artifact_path(
        self, category: str, relative: PurePosixPath
    ) -> Path:
        category = _validate_category(category)
        category_root = self.run_dir / category
        _reject_reparse_point(category_root)
        try:
            resolved_root = category_root.resolve(strict=True)
        except OSError as exc:
            raise RepositoryIntegrityError(
                f"cannot resolve artifact category root {category_root}: {exc}"
            ) from exc

        target = category_root
        parts = relative.parts
        for position, part in enumerate(parts):
            target = target / part
            if os.path.lexists(target):
                _reject_reparse_point(target)
                if position < len(parts) - 1 and not target.is_dir():
                    raise RepositoryIntegrityError(
                        f"artifact parent is not a directory: {target}"
                    )

        existing_parent = target.parent
        while not os.path.lexists(existing_parent):
            if existing_parent == category_root:
                break
            existing_parent = existing_parent.parent
        _reject_reparse_point(existing_parent)
        try:
            resolved_parent = existing_parent.resolve(strict=True)
        except OSError as exc:
            raise RepositoryIntegrityError(
                f"cannot resolve artifact parent {existing_parent}: {exc}"
            ) from exc
        if resolved_parent != resolved_root and not resolved_parent.is_relative_to(
            resolved_root
        ):
            raise RepositoryIntegrityError(
                f"artifact path escapes category root: {target}"
            )
        return target

    @staticmethod
    def _path_matches_record(path: Path, record: ArtifactRecord) -> bool:
        try:
            stat = path.stat()
            if stat.st_size != record.size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == record.sha256
        except OSError:
            return False

    def _quarantine_staged(
        self,
        staged: list[_StagedArtifact],
        *,
        include_promoted: bool = True,
    ) -> list[str]:
        errors: list[str] = []
        for item in staged:
            candidates: list[Path] = []
            if include_promoted and item.promoted:
                candidates.append(item.target)
            if item.temporary is not None:
                candidates.append(item.temporary)
            for path in candidates:
                if not os.path.lexists(path):
                    continue
                try:
                    self._quarantine(path)
                except RepositoryError as exc:
                    errors.append(str(exc))
        return errors

    def _quarantine(self, path: Path) -> Path:
        quarantine = self.run_dir / "quarantine"
        quarantine.mkdir(exist_ok=True)
        self._assert_path_within_run(path)
        self._assert_path_within_run(quarantine)
        try:
            relative = path.relative_to(self.run_dir).as_posix().replace("/", "__")
        except ValueError:
            raise RepositoryIntegrityError(
                f"refusing to quarantine path outside campaign run: {path}"
            ) from None
        target = quarantine / f"{uuid4().hex}-{relative}"
        try:
            os.replace(path, target)
        except OSError as exc:
            raise RepositoryError(
                f"failed to quarantine {path} as {target}: {exc}"
            ) from exc
        self._fsync_directory_best_effort(path.parent)
        self._fsync_directory_best_effort(quarantine)
        return target

    def _assert_path_within_run(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.run_dir)
        except ValueError as exc:
            raise RepositoryIntegrityError(
                f"artifact path is outside campaign run: {path}"
            ) from exc
        _reject_reparse_point(self.run_dir)
        current = self.run_dir
        for part in relative.parts:
            current = current / part
            if os.path.lexists(current):
                _reject_reparse_point(current)
        try:
            resolved_run = self.run_dir.resolve(strict=True)
            resolved_parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise RepositoryIntegrityError(
                f"cannot resolve artifact containment for {path}: {exc}"
            ) from exc
        if resolved_parent != resolved_run and not resolved_parent.is_relative_to(
            resolved_run
        ):
            raise RepositoryIntegrityError(
                f"artifact path escapes campaign run: {path}"
            )

    @staticmethod
    def _fsync_directory_best_effort(directory: Path) -> None:
        """Persist rename metadata on POSIX; Windows directory fsync is unavailable."""

        if os.name == "nt":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _discard_temporary(item: _StagedArtifact) -> None:
        if item.temporary is not None and item.temporary.exists():
            try:
                item.temporary.unlink()
            except OSError as exc:
                raise RepositoryError(
                    f"failed to remove redundant staged artifact "
                    f"{item.temporary}: {exc}"
                ) from exc
            item.temporary = None

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            relative_path=row["relative_path"],
            sha256=row["sha256"],
            size=row["size"],
            media_type=row["media_type"],
            category=row["category"],
        )

    @staticmethod
    def _validate_media_type(media_type: str) -> None:
        if not isinstance(media_type, str) or not media_type:
            raise RepositoryError("artifact media_type must be a non-empty string")

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            connection.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RepositoryClosedError("campaign repository is closed")
        return self._connection

    @staticmethod
    def _commit_transaction(connection: sqlite3.Connection) -> None:
        connection.execute("COMMIT")

    @staticmethod
    def _rollback_transaction(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
