"""Excel-friendly atomic CSV export for a projected desktop campaign."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .artifacts import CampaignProjection, MeasuredPoint


I2C_FIELDS = (
    "ibias_dac",
    "vref",
    "vbias_ref",
    "vga_cur",
    "res_gain",
    "res_bw",
    "gain_ctrl",
    "efin_cur",
    "eq",
)

CSV_FIELDS = (
    "validated_tag",
    "validated_commit",
    "campaign_id",
    "terminal_state",
    "execution_mode",
    "target_tdecq_db",
    "target_reached",
    "b0_restored",
    "evidence_complete",
    "provider_id",
    "provider_version",
    "row_type",
    "round_index",
    "proposal_attempt",
    "observation_id",
    *(f"requested_{name}" for name in I2C_FIELDS),
    *(f"applied_{name}" for name in I2C_FIELDS),
    *(f"readback_{name}" for name in I2C_FIELDS),
    "tdecq_db",
    "outer_er_db",
    "measurement_status",
    "is_best",
    "eye_path",
    "eye_sha256",
    "started_at",
    "finished_at",
    "kimi_events_json",
)


def _text(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _row(result: CampaignProjection, point: MeasuredPoint) -> dict[str, object]:
    row: dict[str, object] = {
        "validated_tag": result.validated_tag,
        "validated_commit": result.validated_commit,
        "campaign_id": result.campaign_id,
        "terminal_state": result.terminal_state,
        "execution_mode": result.execution_mode,
        "target_tdecq_db": result.target_tdecq_db,
        "target_reached": _text(result.target_reached),
        "b0_restored": _text(result.b0_restored),
        "evidence_complete": _text(result.evidence_complete),
        "provider_id": result.provider_id,
        "provider_version": result.provider_version,
        "row_type": point.row_type,
        "round_index": point.round_index,
        "proposal_attempt": _text(point.proposal_attempt),
        "observation_id": point.observation_id,
        "tdecq_db": _text(point.tdecq_db),
        "outer_er_db": _text(point.outer_er_db),
        "measurement_status": point.measurement_status.value,
        "is_best": _text(point.round_index == result.best_round_index),
        "eye_path": _text(point.eye_path),
        "eye_sha256": _text(point.eye_sha256),
        "started_at": _text(point.started_at.isoformat() if point.started_at else None),
        "finished_at": _text(point.finished_at.isoformat() if point.finished_at else None),
        "kimi_events_json": json.dumps(
            point.kimi_events,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
    }
    for name in I2C_FIELDS:
        row[f"requested_{name}"] = _text(point.requested_params.get(name))
        row[f"applied_{name}"] = _text(point.applied_params.get(name))
        row[f"readback_{name}"] = _text(point.readback_params.get(name))
    return row


def export_campaign_csv(
    result: CampaignProjection, output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
            writer.writeheader()
            for point in result.points:
                writer.writerow(_row(result, point))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output
