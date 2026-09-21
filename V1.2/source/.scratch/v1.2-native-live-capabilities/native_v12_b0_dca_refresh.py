from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.live_dca_isolation import IsolatedLiveDcaClient
from vcsel_ai_tuning.hardware_authority import HardwareAuthority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the DCA result table without changing I2C or BERT."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("attempts must be positive")
    if args.artifact_root.exists():
        raise RuntimeError("artifact root already exists; refusing to overwrite")
    profile = load_live_hardware_config(args.profile)
    results = []
    with HardwareAuthority(args.lock_path).acquire(
        f"native-v12:b0-dca-refresh:{args.task_id}"
    ):
        client = IsolatedLiveDcaClient(
            profile=profile,
            artifact_root=args.artifact_root,
            timeout_s=profile.characterization.measurement_timeout_s,
            cleanup_grace_s=5.0,
            capture_dataset=False,
        )
        for attempt in range(1, args.attempts + 1):
            client.set_operation_context(
                task_id=args.task_id,
                item_id=f"b0-refresh-{attempt}",
                round_index=attempt,
            )
            measurement = client.measure_single()
            results.append(
                {
                    "attempt": attempt,
                    "acquisition_valid": measurement.acquisition_valid,
                    "tdecq_status": measurement.tdecq_status,
                    "tdecq_db": measurement.tdecq_db,
                    "tdecq_invalid_reason": measurement.tdecq_invalid_reason,
                    "er_status": measurement.er_status,
                    "er_db": measurement.er_db,
                }
            )
            if (
                measurement.acquisition_valid is True
                and measurement.tdecq_status == "CORR"
                and measurement.er_status == "CORR"
            ):
                break
    success = bool(results and results[-1]["acquisition_valid"] is True)
    print(
        json.dumps(
            {"status": "REFRESHED" if success else "STILL_INVALID", "results": results},
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
