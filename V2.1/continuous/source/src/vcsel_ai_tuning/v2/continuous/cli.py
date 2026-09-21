"""Thin Testbench CLI; all loops and safety decisions stay in the controller."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import yaml

from .factory import (
    LiveContactAuthorizationError,
    build_continuous_dependencies,
)
from .models import ContinuousState, ExecutionMode, HumanDecision


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _result_exit_code(state: ContinuousState) -> int:
    """Return 0 only for COMPLETED; every non-success campaign result returns 1."""

    return 0 if state is ContinuousState.COMPLETED else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vcsel-continuous-eye")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=[item.value for item in ExecutionMode])
    parser.add_argument(
        "--scenario",
        default="happy",
        choices=(
            "happy",
            "readback-mismatch",
            "dca-failure",
            "b0-recovery-failure",
            "stop-acceptance",
        ),
    )
    parser.add_argument("--allow-device-contact", action="store_true")
    parser.add_argument(
        "--live-operation",
        choices=("PREFLIGHT", "COMMISSION"),
    )
    parser.add_argument("--max-rounds", type=_positive_int)
    parser.add_argument("--target-tdecq-db", type=_positive_finite_float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"configuration error: {type(error).__name__}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("configuration must be an object", file=sys.stderr)
        return 2
    try:
        mode = ExecutionMode(args.mode or payload.get("mode", "FAKE"))
        if mode is ExecutionMode.LIVE:
            missing = [
                option
                for option, value in (
                    ("--max-rounds", args.max_rounds),
                    ("--target-tdecq-db", args.target_tdecq_db),
                )
                if value is None
            ]
            if missing:
                raise LiveContactAuthorizationError(
                    "LIVE requires " + " and ".join(missing)
                )
        max_rounds = (
            args.max_rounds
            if args.max_rounds is not None
            else int(payload.get("max_rounds", 3))
        )
        target_tdecq_db = (
            args.target_tdecq_db
            if args.target_tdecq_db is not None
            else float(payload.get("target_tdecq_db", 2.0))
        )
        dependencies = build_continuous_dependencies(
            mode=mode,
            result_dir=payload.get("result_dir", "artifacts/continuous"),
            scenario=args.scenario,
            max_rounds=max_rounds,
            target_tdecq_db=target_tdecq_db,
            allow_device_contact=args.allow_device_contact,
            live_config=payload.get("live"),
        )
    except (ValueError, LiveContactAuthorizationError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if mode is ExecutionMode.LIVE:
        assert dependencies.live_activation is not None
        if args.live_operation is None:
            print("LIVE requires --live-operation", file=sys.stderr)
            return 2
        if args.live_operation == "PREFLIGHT":
            report = dependencies.live_activation.preflight()
            print(json.dumps(report, sort_keys=True))
            return 0
        activated = dependencies.live_activation.activate()
        assert activated.controller is not None
        assert activated.request is not None
        try:
            result = activated.controller.start(activated.request)
            if result.state is ContinuousState.AWAITING_HUMAN:
                result = activated.controller.resolve_human(
                    activated.request.campaign_id,
                    HumanDecision.RESTORE_B0,
                )
        finally:
            if callable(activated.shutdown):
                activated.shutdown()
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return _result_exit_code(result.state)
    assert dependencies.controller is not None
    assert dependencies.request is not None
    result = dependencies.controller.start(dependencies.request)
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return _result_exit_code(result.state)
