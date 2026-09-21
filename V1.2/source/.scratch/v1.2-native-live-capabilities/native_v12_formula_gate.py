from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcsel_ai_tuning.hardware.fir import calculate_iterative_pass
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware_authority import HardwareAuthority
from vcsel_ai_tuning.v1_1.baseline import B0Snapshot
from vcsel_ai_tuning.v1_1.live_ai_execution import (
    LiveAlternatingOptimizationPlant,
    open_live_ai_drivers,
)
from vcsel_ai_tuning.v1_1.optimizer_bridge import load_v11_optimizer_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one formula FIR candidate and restore the captured B0."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if document.get("status") != "SUCCESS":
        raise RuntimeError("formula gate requires a successful Preflight artifact")
    b0 = B0Snapshot.from_payload(document["b0"])
    profile = load_live_hardware_config(args.profile)
    api = load_v11_optimizer_api()
    authority = HardwareAuthority(args.lock_path)
    result: dict[str, object] = {}

    with authority.acquire(f"native-v12:formula-fir-gate:{args.task_id}"):
        with open_live_ai_drivers(
            profile,
            b0,
            artifact_root=args.artifact_root,
            save_dataset=True,
            cleanup_grace_s=5.0,
        ) as bundle:
            plant = LiveAlternatingOptimizationPlant(
                profile=profile,
                task_id=args.task_id,
                b0=b0,
                bundle=bundle,
                optimizer_api=api,
                approved_recovery_values={
                    name: frozenset({value}) for name, value in b0.i2c_fields
                },
                settle_s=0.1,
            )
            try:
                preflight = plant.preflight()
                plant.set_measurement_context("formula-source", 0)
                dca_taps, dca_main_index = plant.read_dca_taps()
                proposal = calculate_iterative_pass(
                    b0.fir_taps,
                    dca_taps,
                    dca_main_index,
                    manual_main=b0.fir_taps[3],
                )
                if proposal.valid is not True:
                    raise RuntimeError(
                        f"formula proposal invalid: {proposal.reason}"
                    )
                if proposal.taps == b0.fir_taps:
                    raise RuntimeError(
                        "formula proposal equals B0; non-baseline gate not exercised"
                    )
                if proposal.taps[3] != b0.fir_taps[3]:
                    raise RuntimeError("formula changed fixed Main tap")
                target = api.fir_target(
                    taps=proposal.taps,
                    upper_nlc=b0.upper_nlc,
                    lower_nlc=b0.lower_nlc,
                    pattern=b0.bert_pattern,
                )
                applied = tuple(plant.apply_fir(target))
                if applied != proposal.taps:
                    raise RuntimeError(
                        f"FIR apply readback mismatch: {applied!r}"
                    )
                plant.set_measurement_context("formula-applied", 1)
                metric = plant.measure()
                if metric.has_valid_tdecq is not True:
                    raise RuntimeError(
                        f"DCA measurement invalid after FIR: {metric!r}"
                    )
                result.update(
                    {
                        "status": "FORMULA_FIR_GATE_PASS",
                        "b0_taps": list(b0.fir_taps),
                        "dca_taps": list(dca_taps),
                        "dca_main_index": dca_main_index,
                        "formula_taps": list(proposal.taps),
                        "formula_l1": proposal.l1,
                        "applied_readback": list(applied),
                        "tdecq_db": metric.tdecq_db,
                        "er_db": metric.er_db,
                        "preflight_verified": preflight["verified"],
                    }
                )
            finally:
                try:
                    plant.restore_b0()
                    restored = plant.is_at_b0()
                    result["terminal_b0_verified"] = restored
                    result["terminal_fir_readback"] = list(
                        plant.current_fir.taps
                    )
                    if restored is not True:
                        raise RuntimeError("terminal B0 verification failed")
                finally:
                    plant.close()

    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
