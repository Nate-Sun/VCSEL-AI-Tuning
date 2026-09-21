"""Strict read-only projection of persisted continuous campaign evidence."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, FiniteFloat, StrictInt, ValidationError

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.v2.continuous.models import (
    CampaignResult,
    MeasurementStatus,
    RoundRecord,
)

from .integrity import VALIDATED_COMMIT, VALIDATED_TAG
from .models import FrozenModel


class DesktopArtifactError(RuntimeError):
    """Persisted campaign evidence is incomplete, corrupt, or inconsistent."""


class MeasuredPoint(FrozenModel):
    row_type: Literal["B0", "ROUND"]
    round_index: StrictInt = Field(ge=0)
    proposal_attempt: StrictInt | None = Field(default=None, gt=0)
    observation_id: str
    requested_params: dict[str, StrictInt]
    applied_params: dict[str, StrictInt]
    readback_params: dict[str, StrictInt]
    tdecq_db: FiniteFloat | None
    outer_er_db: FiniteFloat | None
    measurement_status: MeasurementStatus
    eye_path: Path | None = None
    eye_sha256: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    kimi_events: tuple[dict[str, object], ...] = ()


class CampaignProjection(FrozenModel):
    campaign_id: str
    terminal_state: str
    execution_mode: str
    max_rounds: StrictInt = Field(gt=0)
    target_tdecq_db: FiniteFloat = Field(gt=0)
    target_reached: bool
    b0_restored: bool
    evidence_complete: bool
    stop_reason: str | None
    provider_id: str
    provider_version: str
    adapter_identities: dict[str, object]
    campaign_root: Path
    points: tuple[MeasuredPoint, ...]
    best_round_index: StrictInt | None = Field(default=None, ge=0)
    kimi_events: tuple[dict[str, object], ...] = ()
    validated_tag: Literal[VALIDATED_TAG] = VALIDATED_TAG
    validated_commit: Literal[VALIDATED_COMMIT] = VALIDATED_COMMIT

    @property
    def b0_point(self) -> MeasuredPoint:
        return self.points[0]

    @property
    def last_point(self) -> MeasuredPoint:
        return self.points[-1]

    @property
    def best_point(self) -> MeasuredPoint | None:
        if self.best_round_index is None:
            return None
        return next(
            point
            for point in self.points
            if point.round_index == self.best_round_index
        )


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DesktopArtifactError(f"{label} is unavailable or corrupt") from error
    if not isinstance(value, dict):
        raise DesktopArtifactError(f"{label} must be a JSON object")
    return value


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CampaignArtifactReader:
    def __init__(self, campaign_root: str | Path) -> None:
        self.root = Path(campaign_root).resolve()
        self.campaign_id = self.root.name
        self.controller_root = self.root / "controller" / self.campaign_id
        self.engine_root = self.root / "engine" / self.campaign_id

    def read(self) -> CampaignProjection:
        campaign = _read_object(
            self.controller_root / "campaign.json", "campaign metadata"
        )
        state_path = self.controller_root / "state.json"
        try:
            state = CampaignResult.model_validate_json(
                state_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError, ValueError) as error:
            raise DesktopArtifactError("campaign state is invalid") from error
        if state.campaign_id != self.campaign_id:
            raise DesktopArtifactError("campaign state identity does not match its path")
        if campaign.get("campaign_id") != self.campaign_id:
            raise DesktopArtifactError("campaign metadata identity does not match its path")

        rounds = self._read_rounds()
        if tuple(rounds) != state.round_records:
            raise DesktopArtifactError("round files do not match strict campaign state")
        if not rounds:
            raise DesktopArtifactError("campaign has no measured baseline evidence")

        all_kimi_events, events_by_measurement = self._read_kimi_events()
        baseline = load_b0(self.root / "config" / "b0.yaml")
        first = rounds[0]
        baseline_observation = self._read_observation(first.before_observation_id)
        points = [
            self._baseline_point(
                baseline.i2c.model_dump(), baseline_observation
            )
        ]
        for record in rounds:
            observation = self._read_observation(record.after_observation_id)
            points.append(
                self._round_point(
                    record,
                    observation,
                    events_by_measurement.get(record.before_observation_id, ()),
                )
            )

        valid_points = [
            point
            for point in points
            if point.measurement_status is MeasurementStatus.VALID
            and point.tdecq_db is not None
        ]
        best_index = (
            min(valid_points, key=lambda point: point.tdecq_db).round_index
            if valid_points
            else None
        )
        provider = campaign.get("provider")
        if not isinstance(provider, dict):
            raise DesktopArtifactError("campaign provider metadata is invalid")
        max_rounds = campaign.get("max_rounds")
        if type(max_rounds) is not int or max_rounds <= 0:
            raise DesktopArtifactError("campaign max_rounds metadata is invalid")
        return CampaignProjection(
            campaign_id=self.campaign_id,
            terminal_state=state.state.value,
            execution_mode=state.execution_mode.value,
            max_rounds=max_rounds,
            target_tdecq_db=state.target_tdecq_db,
            target_reached=state.target_reached,
            b0_restored=state.b0_restored,
            evidence_complete=state.evidence_complete,
            stop_reason=state.stop_reason,
            provider_id=str(provider.get("provider_id", "")),
            provider_version=str(provider.get("provider_version", "")),
            adapter_identities=dict(state.adapter_identities),
            campaign_root=self.root,
            points=tuple(points),
            best_round_index=best_index,
            kimi_events=all_kimi_events,
        )

    def _read_rounds(self) -> list[RoundRecord]:
        rounds_root = self.controller_root / "rounds"
        records: list[RoundRecord] = []
        for expected, path in enumerate(
            sorted(rounds_root.glob("*/round.json")), start=1
        ):
            try:
                record = RoundRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValidationError, ValueError) as error:
                raise DesktopArtifactError(f"round {expected} is invalid") from error
            if record.round_index != expected:
                raise DesktopArtifactError(
                    f"round {expected} has a non-contiguous index"
                )
            records.append(record)
        return records

    def _read_kimi_events(
        self,
    ) -> tuple[
        tuple[dict[str, object], ...],
        dict[str, tuple[dict[str, object], ...]],
    ]:
        path = self.controller_root / "kimi-response-attempts.jsonl"
        if not path.exists():
            return (), {}
        events: list[dict[str, object]] = []
        grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        line_number = 0
        try:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("event is not an object")
                measurement_id = value.get("measurement_id")
                if not isinstance(measurement_id, str) or not measurement_id:
                    raise ValueError("measurement_id is missing")
                event = {str(key): item for key, item in value.items()}
                events.append(event)
                grouped[measurement_id].append(event)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DesktopArtifactError(
                f"Kimi response log is corrupt near line {line_number}"
            ) from error
        return tuple(events), {
            key: tuple(value) for key, value in grouped.items()
        }

    def _read_observation(self, observation_id: str) -> dict[str, object]:
        path = self.engine_root / "observations" / f"{observation_id}.json"
        observation = _read_object(path, f"observation {observation_id}")
        if (
            observation.get("observation_id") != observation_id
            or observation.get("campaign_id") != self.campaign_id
        ):
            raise DesktopArtifactError(
                f"observation {observation_id} identity is inconsistent"
            )
        return observation

    def _eye_evidence(
        self, observation: dict[str, object]
    ) -> tuple[Path | None, str | None]:
        evidence = observation.get("evidence")
        if not isinstance(evidence, dict) or "png_path" not in evidence:
            return None, None
        raw_path = evidence.get("png_path")
        expected_hash = evidence.get("png_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise DesktopArtifactError("eye evidence path or hash is invalid")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DesktopArtifactError("eye evidence path is unsafe")
        resolved = (self.engine_root / relative).resolve()
        try:
            resolved.relative_to(self.engine_root.resolve())
        except ValueError as error:
            raise DesktopArtifactError("eye evidence path escapes campaign") from error
        if not resolved.is_file() or resolved.is_symlink():
            raise DesktopArtifactError("eye evidence file is missing or unsafe")
        if _file_digest(resolved) != expected_hash:
            raise DesktopArtifactError("eye evidence hash mismatch")
        return resolved, expected_hash

    @staticmethod
    def _observation_values(
        observation: dict[str, object],
    ) -> tuple[dict[str, int], float | None, float | None, MeasurementStatus]:
        settings = observation.get("settings")
        metrics = observation.get("metrics")
        if not isinstance(settings, dict) or not isinstance(metrics, dict):
            raise DesktopArtifactError("observation settings or metrics are invalid")
        readback = settings.get("readback_i2c")
        if not isinstance(readback, dict) or any(
            type(value) is not int for value in readback.values()
        ):
            raise DesktopArtifactError("observation readback is invalid")
        tdecq = metrics.get("tdecq_db")
        outer_er = metrics.get("er_db")
        if tdecq is not None and not isinstance(tdecq, (int, float)):
            raise DesktopArtifactError("observation TDECQ is invalid")
        if outer_er is not None and not isinstance(outer_er, (int, float)):
            raise DesktopArtifactError("observation ER is invalid")
        status = (
            MeasurementStatus.VALID
            if observation.get("valid") is True
            else MeasurementStatus.INVALID
        )
        return dict(readback), tdecq, outer_er, status

    def _baseline_point(
        self,
        baseline_i2c: dict[str, int],
        observation: dict[str, object],
    ) -> MeasuredPoint:
        readback, tdecq, outer_er, status = self._observation_values(observation)
        eye_path, eye_hash = self._eye_evidence(observation)
        return MeasuredPoint(
            row_type="B0",
            round_index=0,
            observation_id=str(observation["observation_id"]),
            requested_params=baseline_i2c,
            applied_params=baseline_i2c,
            readback_params=readback,
            tdecq_db=tdecq,
            outer_er_db=outer_er,
            measurement_status=status,
            eye_path=eye_path,
            eye_sha256=eye_hash,
        )

    def _round_point(
        self,
        record: RoundRecord,
        observation: dict[str, object],
        kimi_events: tuple[dict[str, object], ...],
    ) -> MeasuredPoint:
        readback, tdecq, outer_er, status = self._observation_values(observation)
        if (
            readback != dict(record.readback_params)
            or tdecq != record.after_metrics.get("tdecq_db")
            or outer_er != record.after_metrics.get("outer_er_db")
            or status is not record.measurement_status
        ):
            raise DesktopArtifactError(
                f"round {record.round_index} observation does not match round evidence"
            )
        eye_path, eye_hash = self._eye_evidence(observation)
        return MeasuredPoint(
            row_type="ROUND",
            round_index=record.round_index,
            proposal_attempt=record.proposal_attempt,
            observation_id=record.after_observation_id,
            requested_params=dict(record.requested_params),
            applied_params=dict(record.applied_params),
            readback_params=dict(record.readback_params),
            tdecq_db=tdecq,
            outer_er_db=outer_er,
            measurement_status=record.measurement_status,
            eye_path=eye_path,
            eye_sha256=eye_hash,
            started_at=record.started_at,
            finished_at=record.finished_at,
            kimi_events=kimi_events,
        )
