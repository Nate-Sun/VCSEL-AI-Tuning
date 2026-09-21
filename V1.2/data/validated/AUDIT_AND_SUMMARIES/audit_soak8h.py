from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SUMMARY_PATH = Path(".scratch/v1.2-native-live-capabilities/soak-8h-summary.json")
ARTIFACT_ROOT = Path("artifacts/hw-v12-soak-8h/v1.2")
EXPECTED_PHASES = {
    "PILOT_3": 3,
    "B0": 1,
    "I2C_COARSE": 6,
    "I2C_FINE": 16,
    "FIR_FIRST": 1,
    "FIR_ITERATIVE": 1,
    "FINAL_VERIFICATION": 1,
}
EXPECTED_FILES = {
    "raw_eye_density.npy",
    "raw_eye.png",
    "tdecq_equalized_eye.png",
}


def main() -> int:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    if summary.get("status") != "PASSED":
        errors.append("summary_status")
    if float(summary.get("elapsed_seconds", 0)) < 28800:
        errors.append("elapsed_too_short")
    campaigns = summary.get("campaigns", [])
    if summary.get("campaign_count") != 80 or len(campaigns) != 80:
        errors.append("campaign_count")

    states: Counter[str] = Counter()
    summary_statuses: Counter[str] = Counter()
    phase_totals: Counter[str] = Counter()
    task_ids: list[str] = []
    for campaign in campaigns:
        task_id = campaign.get("task_id")
        task_ids.append(task_id)
        states[campaign.get("state")] += 1
        summary_statuses.update(campaign.get("dataset", {}).get("statuses", {}))
        phase_totals.update(campaign.get("phases", {}))
        if campaign.get("state") not in {"COMPLETED", "COMPLETED_NO_IMPROVEMENT"}:
            errors.append(f"campaign_state:{task_id}")
        if campaign.get("observations") != 29:
            errors.append(f"campaign_observations:{task_id}")
        if campaign.get("phases") != EXPECTED_PHASES:
            errors.append(f"campaign_phases:{task_id}")
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate_task_id")

    records = sorted(ARTIFACT_ROOT.rglob("record.json"))
    statuses: Counter[str] = Counter()
    campaign_records: Counter[str] = Counter()
    measurement_ids: list[str] = []
    acquisition_ids: list[str] = []
    observation_ids: list[str] = []
    same_acquisition: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    npy_shapes: Counter[tuple[tuple[int, ...], str]] = Counter()
    png_formats: Counter[str] = Counter()
    png_modes: Counter[str] = Counter()
    png_sizes: Counter[tuple[int, int]] = Counter()
    total_bytes = 0
    parameter_mismatches = 0
    missing_metric_keys = 0

    for record_path in records:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{record_path}:json:{error}")
            continue
        status = str(record.get("dataset_status"))
        statuses[status] += 1
        campaign_records[str(record.get("campaign_id"))] += 1
        measurement_ids.append(record.get("measurement_id"))
        acquisition_ids.append(record.get("acquisition_id"))
        observation_ids.append(record.get("observation_id"))
        same_acquisition[str(record.get("same_acquisition_verified"))] += 1

        requested = record.get("requested_params")
        applied = record.get("applied_params")
        readback = record.get("readback_params")
        if not all(isinstance(value, dict) and value for value in (requested, applied, readback)):
            errors.append(f"{record_path}:parameters_missing")
        elif not (requested == applied == readback):
            parameter_mismatches += 1
            errors.append(f"{record_path}:parameters_mismatch")

        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or "tdecq" not in metrics or "outer_er" not in metrics:
            missing_metric_keys += 1
            errors.append(f"{record_path}:metrics")

        files = record.get("files") or {}
        if set(files) != EXPECTED_FILES:
            errors.append(f"{record_path}:file_set:{sorted(files)}")
        for name in EXPECTED_FILES:
            metadata = files.get(name)
            if not isinstance(metadata, dict):
                continue
            file_path = record_path.parent / metadata.get("relative_path", "")
            if not file_path.is_file():
                errors.append(f"{record_path}:missing:{name}")
                continue
            digest = hashlib.sha256()
            size = 0
            with file_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            total_bytes += size
            file_counts[name] += 1
            if size != metadata.get("size"):
                errors.append(f"{record_path}:size:{name}")
            if digest.hexdigest() != metadata.get("sha256"):
                errors.append(f"{record_path}:sha256:{name}")

        npy_path = record_path.parent / "raw_eye_density.npy"
        if npy_path.is_file():
            try:
                density = np.load(npy_path, allow_pickle=False, mmap_mode="r")
                npy_shapes[(tuple(density.shape), str(density.dtype))] += 1
            except Exception as error:
                errors.append(f"{record_path}:npy:{error}")
        for name in ("raw_eye.png", "tdecq_equalized_eye.png"):
            png_path = record_path.parent / name
            if not png_path.is_file():
                continue
            try:
                with Image.open(png_path) as image:
                    image.verify()
                with Image.open(png_path) as image:
                    png_formats[str(image.format)] += 1
                    png_modes[str(image.mode)] += 1
                    png_sizes[tuple(image.size)] += 1
            except Exception as error:
                errors.append(f"{record_path}:png:{name}:{error}")

    for label, values in (
        ("measurement_id", measurement_ids),
        ("acquisition_id", acquisition_ids),
        ("observation_id", observation_ids),
    ):
        if any(not value for value in values):
            errors.append(f"empty:{label}")
        if len(values) != len(set(values)):
            errors.append(f"duplicate:{label}")
    if set(campaign_records) != set(task_ids):
        errors.append("campaign_directory_set")
    if any(count != 29 for count in campaign_records.values()):
        errors.append("campaign_record_count")
    if statuses != summary_statuses:
        errors.append(f"status_aggregate:{dict(statuses)}!={dict(summary_statuses)}")

    print("summary_status", summary.get("status"))
    print("elapsed_seconds", summary.get("elapsed_seconds"))
    print("campaigns", len(task_ids))
    print("states", dict(states))
    print("phase_totals", dict(phase_totals))
    print("records", len(records))
    print("dataset_statuses", dict(statuses))
    print("unique_ids", len(set(measurement_ids)), len(set(acquisition_ids)), len(set(observation_ids)))
    print("campaign_record_counts", dict(Counter(campaign_records.values())))
    print("parameter_mismatches", parameter_mismatches)
    print("missing_metric_keys", missing_metric_keys)
    print("same_acquisition", dict(same_acquisition))
    print("files", dict(file_counts))
    print("file_bytes", total_bytes)
    print("npy", dict(npy_shapes))
    print("png_formats", dict(png_formats))
    print("png_modes", dict(png_modes))
    print("png_sizes", dict(png_sizes))
    print("errors", len(errors))
    for error in errors[:50]:
        print("ERROR", error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
