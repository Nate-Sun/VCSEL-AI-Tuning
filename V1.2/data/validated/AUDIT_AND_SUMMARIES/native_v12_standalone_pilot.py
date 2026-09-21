from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Event

from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware_authority import HardwareAuthority
from vcsel_ai_tuning.v1_1.baseline import B0Snapshot
from vcsel_ai_tuning.v1_1.domain import RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.live_ai_execution import build_live_ai_execution_factory
from vcsel_ai_tuning.v1_1.live_ai_policy import LiveAIProfile
from vcsel_ai_tuning.v1_1.repository import TaskStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Native V1.2 standalone three-candidate PILOT."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.artifact_root.exists():
        raise RuntimeError("artifact root already exists; refusing to overwrite")
    document = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if document.get("status") != "SUCCESS":
        raise RuntimeError("standalone PILOT requires a successful Preflight artifact")
    b0 = B0Snapshot.from_payload(document["b0"])
    profile = load_live_hardware_config(args.profile)
    payload = {
        "task_id": args.task_id,
        "task_type": "AI_OPTIMIZATION",
        "live_ai_profile": "PILOT_3",
        "optimization_scope": "I2C_ONLY",
        "depth": "STANDARD",
        "candidate_budget": 3,
        "coarse_candidate_budget": 3,
        "fine_candidate_budget": 0,
        "settle_time_s": 0.1,
        "restore_after_each_candidate": True,
        "save_dataset": True,
        "seed": 17,
        "parameters": [
            {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
            {"name": "res_gain", "min": 4, "max": 8, "step": 4},
        ],
    }
    database = args.artifact_root / "tasks.sqlite3"
    with TaskStore(database) as store:
        store.create_task(
            task_id=args.task_id,
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.LIVE,
            config_fingerprint="c" * 64,
            payload=payload,
        )
        store.save_b0(args.task_id, b0)

    factory = build_live_ai_execution_factory(
        profile,
        database_path=database,
        artifact_root=args.artifact_root,
        measurement_timeout_s=60.0,
        cleanup_grace_s=5.0,
        live_ai_profile=LiveAIProfile.PILOT_3,
    )
    execution = factory(args.task_id, b0, payload)
    terminal = None
    run_error: BaseException | None = None
    recovery_error: BaseException | None = None
    with HardwareAuthority(args.lock_path).acquire(
        f"native-v12:standalone-pilot:{args.task_id}"
    ):
        try:
            terminal = execution.run(Event())
        except BaseException as error:
            run_error = error
            try:
                execution.restore_b0()
            except BaseException as error:
                recovery_error = error

    with TaskStore(database) as store:
        rows = store.observations(args.task_id)
    records = sorted(args.artifact_root.rglob("record.json"))
    result = {
        "status": "PILOT_3_PASS" if run_error is None else "PILOT_3_FAILED",
        "terminal_state": getattr(terminal, "value", terminal),
        "terminal_b0_verified": execution.terminal_b0_verified,
        "run_error": None if run_error is None else repr(run_error),
        "recovery_error": None if recovery_error is None else repr(recovery_error),
        "observation_count": len(rows),
        "observations": [
            {
                "sequence": row.sequence,
                "phase": row.phase,
                "status": row.status,
                "valid": row.valid,
                "requested": row.requested,
                "applied": row.applied,
                "readback": row.readback,
                "tdecq": row.tdecq,
                "er": row.er,
                "dataset_status": (
                    row.raw_response.get("dataset_status")
                    if isinstance(row.raw_response, dict)
                    else None
                ),
                "error_reason": row.error_reason,
            }
            for row in rows
        ],
        "record_json_count": len(records),
        "record_json": [str(path.resolve()) for path in records],
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if recovery_error is not None:
        raise RuntimeError("standalone PILOT B0 recovery failed") from recovery_error
    if run_error is not None:
        raise RuntimeError("standalone PILOT failed") from run_error
    if execution.terminal_b0_verified is not True:
        raise RuntimeError("standalone PILOT did not verify terminal B0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
