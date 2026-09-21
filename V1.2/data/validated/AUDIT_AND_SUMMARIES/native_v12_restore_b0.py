from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware_authority import HardwareAuthority
from vcsel_ai_tuning.v1_1.live_ai_execution import build_live_ai_execution_factory
from vcsel_ai_tuning.v1_1.live_ai_policy import LiveAIProfile
from vcsel_ai_tuning.v1_1.repository import TaskStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore and verify a persisted Native V1.2 B0 exactly once."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--source-task-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.artifact_root.exists():
        raise RuntimeError("artifact root already exists; refusing to overwrite")
    profile = load_live_hardware_config(args.profile)
    with TaskStore(args.database) as store:
        b0 = store.load_b0(args.source_task_id)
        payload = dict(store.require_task(args.source_task_id).payload)
    factory = build_live_ai_execution_factory(
        profile,
        database_path=args.database,
        artifact_root=args.artifact_root,
        live_ai_profile=LiveAIProfile.PILOT_3_THEN_FAST24,
    )
    execution = factory(args.source_task_id, b0, payload)
    try:
        with HardwareAuthority(args.lock_path).acquire(
            f"native-v12:b0-restore:{args.source_task_id}"
        ):
            execution.restore_b0()
    finally:
        execution.close()
    print(
        json.dumps(
            {
                "status": "B0_RESTORED_AND_VERIFIED",
                "source_task_id": args.source_task_id,
                "i2c": dict(b0.i2c_fields),
                "fir_taps": list(b0.fir_taps),
                "upper_nlc": b0.upper_nlc,
                "lower_nlc": b0.lower_nlc,
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
