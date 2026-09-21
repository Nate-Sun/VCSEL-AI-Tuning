from __future__ import annotations

import json
import os
import re
import sqlite3
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .domain import ArtifactRef, Candidate, Observation, RunEvent


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.staging_root = self.root / ".staging"
        self.quarantine_root = self.root / "quarantine"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root.mkdir(parents=True, exist_ok=True)

    def stage_run(self, campaign_id: str, run_id: str) -> Path:
        stage = self.staging_root / campaign_id / f"{run_id}-{uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        return stage

    @staticmethod
    def _child(root: Path, relative_name: str) -> Path:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {relative_name}")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(
        self,
        stage: Path,
        relative_name: str,
        data: bytes,
        media_type: str,
    ) -> ArtifactRef:
        path = self._child(stage, relative_name)
        path.write_bytes(data)
        return ArtifactRef(
            relative_path=Path(relative_name),
            sha256=sha256(data).hexdigest(),
            media_type=media_type,
        )

    def write_json(self, stage: Path, relative_name: str, value: object) -> None:
        path = self._child(stage, relative_name)
        payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        path.write_text(payload + "\n", encoding="utf-8")

    def finalize(self, stage: Path, campaign_id: str, run_id: str) -> Path:
        final = self.root / "campaigns" / campaign_id / "runs" / run_id
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(final)
        os.replace(stage, final)
        return final

    def quarantine(self, run_path: Path, reason: str) -> Path:
        safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "-", reason).strip("-") or "error"
        destination = self.quarantine_root / f"{run_path.name}-{safe_reason}-{uuid4().hex}"
        os.replace(run_path, destination)
        return destination


class SqliteStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    run_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    valid INTEGER NOT NULL,
                    feasible INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, run_id)
                )
                """
            )

    def save_observation(self, observation: Observation) -> None:
        payload = observation.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO observations
                    (run_id, campaign_id, candidate_id, valid, feasible, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.run_id,
                    observation.campaign_id,
                    observation.candidate_id,
                    int(observation.valid),
                    int(observation.feasible),
                    payload,
                ),
            )

    def list_observations(self, campaign_id: str) -> list[Observation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM observations
                WHERE campaign_id = ?
                ORDER BY run_id
                """,
                (campaign_id,),
            ).fetchall()
        return [Observation.model_validate_json(row[0]) for row in rows]


class RunRepository:
    def __init__(self, store: SqliteStore, artifacts: ArtifactStore):
        self.store = store
        self.artifacts = artifacts

    def commit_run(
        self,
        candidate: Candidate,
        observation: Observation,
        events: list[RunEvent],
        raw_artifacts: dict[str, tuple[bytes, str]],
    ) -> Observation:
        stage = self.artifacts.stage_run(observation.campaign_id, observation.run_id)
        refs = [
            self.artifacts.write_bytes(stage, name, data, media_type)
            for name, (data, media_type) in sorted(raw_artifacts.items())
        ]
        saved = observation.model_copy(update={"artifacts": refs})
        self.artifacts.write_json(
            stage, "candidate.json", candidate.model_dump(mode="json")
        )
        self.artifacts.write_json(
            stage, "observation.json", saved.model_dump(mode="json")
        )
        event_lines = "".join(event.model_dump_json() + "\n" for event in events)
        self.artifacts._child(stage, "events.jsonl").write_text(
            event_lines, encoding="utf-8"
        )
        final = self.artifacts.finalize(
            stage, observation.campaign_id, observation.run_id
        )
        try:
            self.store.save_observation(saved)
        except Exception:
            self.artifacts.quarantine(final, "database-commit-failed")
            raise
        return saved
