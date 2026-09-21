from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


TERMINAL = {
    "COMPLETED",
    "COMPLETED_NO_IMPROVEMENT",
    "FAILED",
    "ABORTED",
    "RECOVERY_REQUIRED",
}
SUCCESS = {"COMPLETED", "COMPLETED_NO_IMPROVEMENT"}
EXPECTED_PHASES = Counter(
    {
        "PILOT_3": 3,
        "B0": 1,
        "I2C_COARSE": 6,
        "I2C_FINE": 16,
        "FIR_FIRST": 1,
        "FIR_ITERATIVE": 1,
        "FINAL_VERIFICATION": 1,
    }
)
EXPECTED_I2C = {
    "efin_cur": 15,
    "eq": 22,
    "gain_ctrl": 5,
    "ibias_dac": 70,
    "res_bw": 0,
    "res_gain": 6,
    "vbias_ref": 15,
    "vga_cur": 13,
    "vref": 15,
}
EXPECTED_FIR = [0, 9, -28, 104, -8, 1, -3]


def request_json(method: str, url: str, body: object | None = None) -> Any:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error
    return None if not payload else json.loads(payload)


def identities(snapshot: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (item["kind"], item["resource"], item["identity"])
            for item in snapshot["b0"]["hardware_identities"]
        )
    )


def verify_b0(snapshot: dict[str, Any], expected_identities: object | None) -> None:
    b0 = snapshot["b0"]
    fir = b0["fir"]
    if b0["i2c_fields"] != EXPECTED_I2C:
        raise RuntimeError(f"B0 I2C mismatch: {b0['i2c_fields']!r}")
    if fir["taps"] != EXPECTED_FIR:
        raise RuntimeError(f"B0 FIR mismatch: {fir['taps']!r}")
    if (fir["upper_nlc"], fir["lower_nlc"], fir["pattern"]) != (6, 6, "SSPRQ"):
        raise RuntimeError(f"B0 NLC/pattern mismatch: {fir!r}")
    current = identities(snapshot)
    if expected_identities is not None and current != expected_identities:
        raise RuntimeError(f"hardware identity drift: {current!r}")


def task_payload(task_id: str, seed: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": "AI_OPTIMIZATION",
        "channel_mapping": {
            "i2c_channel": 0,
            "bert_port": "port1",
            "bert_channel": "0",
            "dca_tdecq_source": "FUNC1",
            "dca_tdecq_equalizer_function": 1,
            "dca_outer_er_source": "CHAN1A",
            "operator_confirmed": True,
        },
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
        "depth": "FAST",
        "live_ai_profile": "PILOT_3_THEN_FAST24",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        "candidate_budget": 27,
        "pilot_candidate_budget": 3,
        "fast_candidate_budget": 24,
        "coarse_candidate_budget": 6,
        "fine_candidate_budget": 16,
        "fir_first_candidate_budget": 1,
        "fir_iterative_candidate_budget": 1,
        "settle_time_s": 0.1,
        "save_dataset": True,
        "seed": seed,
    }


def audit_dataset(task_root: Path) -> dict[str, Any]:
    records = sorted(task_root.rglob("record.json"))
    if len(records) != 29:
        raise RuntimeError(f"Dataset record count is {len(records)}, expected 29")
    statuses: Counter[str] = Counter()
    measurement_ids: set[str] = set()
    acquisition_ids: set[str] = set()
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        status = str(record.get("dataset_status"))
        statuses[status] += 1
        if status not in {"COMPLETE", "PARTIAL"}:
            raise RuntimeError(f"{record_path}: Dataset {status}")
        measurement_ids.add(str(record.get("measurement_id") or ""))
        acquisition_ids.add(str(record.get("acquisition_id") or ""))
        for key in ("requested_params", "applied_params", "readback_params"):
            if not isinstance(record.get(key), dict) or not record[key]:
                raise RuntimeError(f"{record_path}: missing {key}")
        if not (
            record["requested_params"]
            == record["applied_params"]
            == record["readback_params"]
        ):
            raise RuntimeError(f"{record_path}: requested/applied/readback mismatch")
        if status == "PARTIAL":
            continue
        files = record.get("files", {})
        expected = {
            "raw_eye_density.npy",
            "raw_eye.png",
            "tdecq_equalized_eye.png",
        }
        if set(files) != expected:
            raise RuntimeError(f"{record_path}: COMPLETE file set mismatch")
        for name, metadata in files.items():
            path = record_path.parent / metadata["relative_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != metadata["sha256"] or path.stat().st_size != metadata["size"]:
                raise RuntimeError(f"{record_path}: integrity mismatch for {name}")
        density = np.load(record_path.parent / "raw_eye_density.npy", allow_pickle=False)
        if density.shape != (521, 751) or density.dtype != np.uint32:
            raise RuntimeError(f"{record_path}: invalid NPY {density.shape}/{density.dtype}")
        for name in ("raw_eye.png", "tdecq_equalized_eye.png"):
            with Image.open(record_path.parent / name) as image:
                image.load()
                if image.format != "PNG" or image.size != (751, 521):
                    raise RuntimeError(f"{record_path}: invalid PNG {name}")
    if "" in measurement_ids or "" in acquisition_ids:
        raise RuntimeError("Dataset contains empty measurement/acquisition ID")
    if len(measurement_ids) != 29 or len(acquisition_ids) != 29:
        raise RuntimeError("Dataset measurement/acquisition IDs are not unique")
    return {"records": 29, "statuses": dict(statuses)}


def persist_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--task-prefix", required=True)
    arguments = parser.parse_args()
    if arguments.summary.exists():
        raise RuntimeError(f"summary already exists: {arguments.summary}")
    api = arguments.base_url.rstrip("/") + "/api/v1.1"
    start_monotonic = time.monotonic()
    deadline = start_monotonic + arguments.duration_seconds
    started_at = datetime.now(timezone.utc)
    summary: dict[str, Any] = {
        "status": "RUNNING",
        "started_at_utc": started_at.isoformat(),
        "requested_duration_seconds": arguments.duration_seconds,
        "campaigns": [],
    }
    initial = request_json("GET", f"{api}/hardware/snapshot")
    verify_b0(initial, None)
    expected_identities = identities(initial)
    iteration = 0
    try:
        while time.monotonic() < deadline:
            iteration += 1
            now = datetime.now(timezone.utc)
            task_id = f"{arguments.task_prefix}-{now.strftime('%Y%m%dT%H%M%SZ')}-{iteration:03d}"
            active = request_json("GET", f"{api}/tasks/active")
            if active not in (None, "null"):
                raise RuntimeError(f"unexpected active task: {active!r}")
            request_json("POST", f"{api}/tasks", task_payload(task_id, 16 + iteration))
            snapshot = request_json("GET", f"{api}/hardware/snapshot")
            verify_b0(snapshot, expected_identities)
            task_url = f"{api}/tasks/{task_id}"
            request_json(
                "POST",
                f"{task_url}/b0",
                {"snapshot_token": snapshot["snapshot_token"], "operator_confirmed": True},
            )
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
            request_json(
                "POST",
                f"{task_url}/authorization",
                {
                    "authorization_id": f"auth-{task_id}",
                    "expires_at": expires_at.isoformat(),
                    "operator_confirmed": True,
                },
            )
            request_json("POST", f"{task_url}/start")
            print(f"CAMPAIGN_STARTED {task_id}", flush=True)
            campaign_started = time.monotonic()
            last_progress: tuple[str, int] | None = None
            while True:
                time.sleep(5)
                task = request_json("GET", task_url)
                observations = request_json("GET", f"{task_url}/observations")
                progress = (task["state"], len(observations))
                if progress != last_progress:
                    print(
                        f"PROGRESS {task_id} state={progress[0]} observations={progress[1]}",
                        flush=True,
                    )
                    last_progress = progress
                if task["state"] in TERMINAL:
                    break
                if time.monotonic() - campaign_started > 25 * 60:
                    raise RuntimeError(f"{task_id}: campaign monitor timeout")
            post_snapshot = request_json("GET", f"{api}/hardware/snapshot")
            verify_b0(post_snapshot, expected_identities)
            if task["state"] not in SUCCESS:
                raise RuntimeError(f"{task_id}: terminal state {task['state']}")
            if len(observations) != 29:
                raise RuntimeError(f"{task_id}: {len(observations)} observations")
            phases = Counter(item["phase"] for item in observations)
            if phases != EXPECTED_PHASES:
                raise RuntimeError(f"{task_id}: phase allocation {dict(phases)}")
            report = request_json("GET", f"{task_url}/final-report")
            if report["dataset_status"] not in {"SUCCESS", "PARTIAL"}:
                raise RuntimeError(
                    f"{task_id}: final Dataset {report['dataset_status']}"
                )
            dataset = audit_dataset(arguments.artifact_root / task_id)
            entry = {
                "task_id": task_id,
                "state": task["state"],
                "observations": 29,
                "phases": dict(phases),
                "dataset": dataset,
                "final_dataset_status": report["dataset_status"],
                "elapsed_seconds": time.monotonic() - campaign_started,
                "best": request_json("GET", f"{task_url}/best"),
                "post_b0_hash": post_snapshot["b0_hash"],
            }
            summary["campaigns"].append(entry)
            persist_summary(arguments.summary, summary)
            print(
                f"CAMPAIGN_COMPLETED {task_id} elapsed={entry['elapsed_seconds']:.1f}s "
                f"dataset={dataset['statuses']}",
                flush=True,
            )
        final_snapshot = request_json("GET", f"{api}/hardware/snapshot")
        verify_b0(final_snapshot, expected_identities)
        summary.update(
            status="PASSED",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - start_monotonic,
            campaign_count=len(summary["campaigns"]),
            final_b0_hash=final_snapshot["b0_hash"],
        )
        persist_summary(arguments.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        summary.update(
            status="FAILED",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - start_monotonic,
            error=f"{type(error).__name__}: {error}",
        )
        persist_summary(arguments.summary, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
