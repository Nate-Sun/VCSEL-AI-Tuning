"""Safe CLI for pre-tuning hardware characterization / 调优前硬件标定。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence

from vcsel_ai_tuning.hardware.characterization import (
    CharacterizationError,
    build_convergence_defaults,
    create_baseline_plan,
    create_same_value_plan,
    create_settle_plan,
    execute_baseline_plan,
    execute_same_value_plan,
    execute_settle_plan,
    load_characterization_plan,
    save_characterization_plan,
    summarize_measurements,
    validate_plan_against_config,
    write_run_artifacts,
)
from vcsel_ai_tuning.hardware.flexdca import open_visa_session
from vcsel_ai_tuning.hardware.live_config import (
    LiveHardwareConfig,
    load_live_hardware_config,
)
from vcsel_ai_tuning.hardware.vd1124 import (
    Vd1124Probe,
    Vd1124ProbeResult,
    Vd1124RegisterReader,
    Vd1124RegisterWriter,
)
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration, VitBertClient


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "hardware_characterization"
DEFAULT_BASELINE_PLAN = DEFAULT_OUTPUT_ROOT / "latest_baseline_plan.json"
DEFAULT_SAME_VALUE_PLAN = DEFAULT_OUTPUT_ROOT / "latest_same_value_plan.json"
DEFAULT_SETTLE_PLAN = DEFAULT_OUTPUT_ROOT / "latest_settle_plan.json"


def _settle_delays(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "settle delays must be comma-separated numbers"
        ) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "settle delays must be non-empty and non-negative"
        )
    return values


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcsel-hw-characterize",
        description="Pre-tuning hardware characterization / 调优前硬件标定",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    baseline_plan = commands.add_parser(
        "baseline-plan", help="Create offline read-only plan / 创建离线只读计划"
    )
    _add_config(baseline_plan)
    baseline_plan.add_argument("--repeats", type=int, default=5)
    baseline_plan.add_argument("--output", type=Path, default=DEFAULT_BASELINE_PLAN)

    baseline_run = commands.add_parser(
        "baseline-run", help="Run confirmed DCA Singles / 执行已确认的 DCA Single"
    )
    _add_config(baseline_run)
    baseline_run.add_argument("--plan", type=Path, required=True)
    baseline_run.add_argument("--execute", action="store_true", required=True)
    baseline_run.add_argument("--confirmation-id", required=True)
    baseline_run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    same_plan = commands.add_parser(
        "same-value-plan", help="Read I2C and plan same-value writes / 读取并规划原值写回"
    )
    _add_config(same_plan)
    same_plan.add_argument("--channel", type=int, default=0)
    same_plan.add_argument("--output", type=Path, default=DEFAULT_SAME_VALUE_PLAN)

    same_run = commands.add_parser(
        "same-value-run", help="Apply confirmed same-value writes / 执行已确认原值写回"
    )
    _add_config(same_run)
    same_run.add_argument("--plan", type=Path, required=True)
    same_run.add_argument("--apply", action="store_true", required=True)
    same_run.add_argument("--confirmation-id", required=True)
    same_run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    settle_plan = commands.add_parser(
        "settle-plan", help="Plan one explicit parameter step / 规划单个明确参数步进"
    )
    _add_config(settle_plan)
    settle_plan.add_argument("--channel", type=int, default=0)
    settle_plan.add_argument("--parameter", required=True)
    settle_plan.add_argument("--target-value", type=int, required=True)
    settle_plan.add_argument("--settle-delays", type=_settle_delays, required=True)
    settle_plan.add_argument("--output", type=Path, default=DEFAULT_SETTLE_PLAN)

    settle_run = commands.add_parser(
        "settle-run", help="Run and restore one parameter step / 执行并恢复单参数步进"
    )
    _add_config(settle_run)
    settle_run.add_argument("--plan", type=Path, required=True)
    settle_run.add_argument("--apply", action="store_true", required=True)
    settle_run.add_argument("--confirmation-id", required=True)
    settle_run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _probe_i2c(config: LiveHardwareConfig) -> Vd1124ProbeResult:
    with Vd1124Probe(
        port=config.i2c.port,
        baudrate=config.i2c.baudrate,
        address=config.i2c.device_address,
        chip_id_register=config.i2c.chip_id_register,
        expected_chip_id=config.i2c.expected_chip_id,
        chip_revision_register=config.i2c.chip_revision_register,
        expected_chip_revision=config.i2c.expected_chip_revision,
    ) as probe:
        return probe.probe()


def _reader(config: LiveHardwareConfig) -> Vd1124RegisterReader:
    return Vd1124RegisterReader(
        port=config.i2c.port,
        baudrate=config.i2c.baudrate,
        address=config.i2c.device_address,
    )


def _writer(config: LiveHardwareConfig) -> Vd1124RegisterWriter:
    return Vd1124RegisterWriter(
        port=config.i2c.port,
        baudrate=config.i2c.baudrate,
        address=config.i2c.device_address,
    )


def _bert(config: LiveHardwareConfig) -> VitBertClient:
    return VitBertClient(
        config.bert.ip,
        config.bert.udp_port,
        local_ip=config.bert.local_ip,
        timeout_s=config.bert.timeout_s,
    )


def _bert_document(value: BertConfiguration) -> dict[str, object]:
    return {
        "port": value.port,
        "channel": value.channel,
        "fir_taps": list(value.taps),
        "upper_nlc": value.upper_nlc,
        "lower_nlc": value.lower_nlc,
        "raw_response": value.raw_response,
    }


def _verify_bert_target(
    value: BertConfiguration, config: LiveHardwareConfig
) -> None:
    if (
        value.port != config.bert.port
        or value.channel != config.bert.channel
        or not value.raw_response.strip()
    ):
        raise CharacterizationError(
            "BERT configuration identity/target does not match live config"
        )


def _verify_dca_identity(dca: object, plan: dict[str, object]) -> str:
    hardware = plan.get("hardware")
    if not isinstance(hardware, dict) or not isinstance(hardware.get("dca"), dict):
        raise CharacterizationError("plan DCA target is missing")
    expected = hardware["dca"].get("expected_idn_prefix")
    if not isinstance(expected, str):
        raise CharacterizationError("plan expected DCA identity is missing")
    try:
        idn = dca.query("*IDN?")
    except Exception as error:
        raise CharacterizationError(f"DCA identity query failed: {error}") from error
    if not isinstance(idn, str) or not idn.strip().startswith(expected):
        raise CharacterizationError("DCA identity does not match settle plan")
    return idn.strip()


def _field_values_document(values: object) -> dict[str, object]:
    return {
        value.name: {
            "channel": value.channel,
            "address": value.address,
            "raw_value": value.raw_value,
            "value": value.value,
        }
        for value in values
    }


def _persist_baseline(
    result: dict[str, object], output_root: Path, started: datetime
) -> Path:
    return write_run_artifacts(
        output_root,
        run_id=str(result["run_id"]),
        plan=result["plan"],
        events=result["events"],
        measurements=result["measurements"],
        raw_responses=result["raw_responses"],
        initial_snapshot=result["initial_snapshot"],
        final_snapshot=result["final_snapshot"],
        summary=result["summary"],
        convergence_defaults=result["convergence_defaults"],
        report_markdown=str(result["report_markdown"]),
        started_at_utc=started,
    )


def _persist_write_run(
    *,
    plan: dict[str, object],
    result: dict[str, object],
    initial_snapshot: dict[str, object],
    final_snapshot: dict[str, object],
    output_root: Path,
    started: datetime,
) -> Path:
    measurements = result.get("measurements", [])
    assert isinstance(measurements, list)
    summary = summarize_measurements(measurements)
    summary["i2c_write_readback_consistent"] = result.get("status") in {
        "VERIFIED", "COMPLETED_RESTORED", "FAILED_RESTORED"
    }
    summary["restoration_succeeded"] = result.get("restoration_succeeded")
    plan_parameters = plan.get("parameters", {})
    assert isinstance(plan_parameters, dict)
    convergence = build_convergence_defaults(
        summary,
        er_floor_db=plan_parameters.get("meaningful_er_floor_db"),
        tdecq_floor_db=plan_parameters.get("meaningful_tdecq_floor_db"),
        repeats=max(1, len(measurements)),
    )
    status = str(result.get("status"))
    report = (
        "# Hardware Characterization / 硬件标定\n\n"
        f"- Stage / 阶段: {plan['kind']}\n"
        f"- Status / 状态: {status}\n"
        "- BERT FIR writes / BERT FIR 写入: none / 无\n"
    )
    event = {
        "event": "run_completed",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return write_run_artifacts(
        output_root,
        run_id=str(plan["plan_id"])[:12],
        plan=plan,
        events=[event],
        measurements=measurements,
        raw_responses={"execution": result},
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        summary=summary,
        convergence_defaults=convergence,
        report_markdown=report,
        started_at_utc=started,
    )


def _persist_failure_run(
    *,
    plan: dict[str, object],
    output_root: Path,
    started: datetime,
    error: BaseException,
    initial_snapshot: dict[str, object] | None = None,
) -> Path:
    message = f"{type(error).__name__}: {error}"
    measurements: list[dict[str, object]] = []
    summary = summarize_measurements(measurements)
    parameters = plan.get("parameters", {})
    assert isinstance(parameters, dict)
    convergence = build_convergence_defaults(
        summary,
        er_floor_db=parameters.get("meaningful_er_floor_db"),
        tdecq_floor_db=parameters.get("meaningful_tdecq_floor_db"),
        repeats=max(1, int(parameters.get("repeats", 1))),
    )
    report = (
        "# Failed Hardware Characterization / 硬件标定失败\n\n"
        f"- Status / 状态: FAILED_AUDITED\n"
        f"- Error / 错误: {message}\n"
        "- Further hardware actions / 后续硬件动作: stopped / 已停止\n"
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    unavailable = {"unavailable_reason": message}
    return write_run_artifacts(
        output_root,
        run_id=str(plan["plan_id"])[:12] + "-failed",
        plan=plan,
        events=[
            {
                "event": "run_failed",
                "status": "FAILED_AUDITED",
                "error": message,
                "timestamp_utc": timestamp,
            }
        ],
        measurements=measurements,
        raw_responses={"execution_error": message},
        initial_snapshot=initial_snapshot or unavailable,
        final_snapshot=unavailable,
        summary=summary,
        convergence_defaults=convergence,
        report_markdown=report,
        started_at_utc=started,
    )


def _load_checked_plan(path: Path, config: LiveHardwareConfig) -> dict[str, object]:
    plan = load_characterization_plan(path)
    validate_plan_against_config(plan, config)
    return plan


def _run_baseline_plan(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    plan = create_baseline_plan(config, repeats=args.repeats)
    save_characterization_plan(plan, args.output)
    print(f"Plan ID: {plan['plan_id']}")
    print(f"Plan / 计划: {args.output}")
    print("Hardware access / 硬件访问: none / 无")
    return 0


def _run_baseline(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    plan = _load_checked_plan(args.plan, config)
    identity = _probe_i2c(config)
    started = datetime.now(timezone.utc)
    bert = _bert(config)
    try:
        with open_visa_session(
            config.dca.resource, timeout_ms=config.dca.timeout_ms
        ) as dca, _reader(config) as reader:
            result = execute_baseline_plan(
                plan,
                dca,
                bert,
                reader,
                identity,
                allow_execute=args.execute,
                confirmation_id=args.confirmation_id,
            )
    except BaseException as error:
        run_dir = _persist_failure_run(
            plan=plan,
            output_root=args.output_root,
            started=started,
            error=error,
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise CharacterizationError(
            f"baseline failed; failure audit written to {run_dir}: {error}"
        ) from error
    run_dir = _persist_baseline(result, args.output_root, started)
    print(f"Status / 状态: {result['status']}")
    print(f"Artifacts / 产物: {run_dir}")
    return 0


def _run_same_plan(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    identity = _probe_i2c(config)
    with _reader(config) as reader:
        plan = create_same_value_plan(
            config, reader, identity, channel=args.channel
        )
    save_characterization_plan(plan, args.output)
    print(f"Plan ID: {plan['plan_id']}")
    print(f"Plan / 计划: {args.output}")
    print("I2C writes / I2C 写入: none / 无")
    return 0


def _run_same(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    plan = _load_checked_plan(args.plan, config)
    identity = _probe_i2c(config)
    bert = _bert(config)
    initial_bert = bert.get_configuration(config.bert.port, config.bert.channel)
    _verify_bert_target(initial_bert, config)
    started = datetime.now(timezone.utc)
    initial_snapshot = dict(plan.get("creation_snapshot", {}))
    initial_snapshot["bert"] = _bert_document(initial_bert)
    try:
        with _writer(config) as writer:
            result = execute_same_value_plan(
                plan,
                writer,
                identity,
                allow_apply=args.apply,
                confirmation_id=args.confirmation_id,
            )
            final_fields = writer.read_fields(
                config.characterization.parameters,
                channel=int(plan["parameters"]["channel"]),
            )
        final_bert = bert.get_configuration(config.bert.port, config.bert.channel)
    except BaseException as error:
        run_dir = _persist_failure_run(
            plan=plan, output_root=args.output_root, started=started,
            error=error, initial_snapshot=initial_snapshot,
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise CharacterizationError(
            f"same-value run failed; failure audit written to {run_dir}: {error}"
        ) from error
    if final_bert.taps != initial_bert.taps:
        result["status"] = "RECOVERY_REQUIRED"
        result["bert_error"] = "BERT FIR changed unexpectedly; no FIR write was issued"
    final_snapshot = {
        "bert": _bert_document(final_bert),
        "i2c_parameters": _field_values_document(final_fields),
    }
    run_dir = _persist_write_run(
        plan=plan, result=result, initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot, output_root=args.output_root,
        started=started,
    )
    print(f"Status / 状态: {result['status']}")
    print(f"Artifacts / 产物: {run_dir}")
    return 0 if result["status"] == "VERIFIED" else 4


def _run_settle_plan(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    identity = _probe_i2c(config)
    with _reader(config) as reader:
        plan = create_settle_plan(
            config,
            reader,
            identity,
            parameter=args.parameter,
            target_value=args.target_value,
            settle_delays=args.settle_delays,
            channel=args.channel,
        )
    save_characterization_plan(plan, args.output)
    print(f"Plan ID: {plan['plan_id']}")
    print(f"Plan / 计划: {args.output}")
    print("I2C writes / I2C 写入: none / 无")
    return 0


def _run_settle(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    plan = _load_checked_plan(args.plan, config)
    identity = _probe_i2c(config)
    bert = _bert(config)
    initial_bert = bert.get_configuration(config.bert.port, config.bert.channel)
    _verify_bert_target(initial_bert, config)
    started = datetime.now(timezone.utc)
    baseline_value = plan.get("measurement_plan")
    if not isinstance(baseline_value, dict):
        raise CharacterizationError("settle plan has no embedded measurement plan")
    baseline = dict(baseline_value)
    validate_plan_against_config(baseline, config)
    settle_raw: dict[str, object] = {}
    initial_snapshot = dict(plan.get("creation_snapshot", {}))
    initial_snapshot["bert"] = _bert_document(initial_bert)
    try:
        with _writer(config) as writer, open_visa_session(
            config.dca.resource, timeout_ms=config.dca.timeout_ms
        ) as dca:
            settle_dca_idn = _verify_dca_identity(dca, plan)

            def measure_once(_delay: float) -> dict[str, object]:
                measured = execute_baseline_plan(
                    baseline,
                    dca,
                    bert,
                    writer,
                    identity,
                    allow_execute=True,
                    confirmation_id=str(baseline["plan_id"]),
                )
                settle_raw[f"measurement_{len(settle_raw) + 1}"] = {
                    "raw_responses": measured["raw_responses"],
                    "initial_snapshot": measured["initial_snapshot"],
                    "final_snapshot": measured["final_snapshot"],
                }
                return dict(measured["measurements"][0])

            result = execute_settle_plan(
                plan,
                writer,
                identity,
                measure_once,
                allow_apply=args.apply,
                confirmation_id=args.confirmation_id,
            )
            result["dca_measurement_evidence"] = settle_raw
            result["prewrite_dca_idn"] = settle_dca_idn
            final_fields = writer.read_fields(
                config.characterization.parameters,
                channel=int(plan["parameters"]["channel"]),
            )
        final_bert = bert.get_configuration(config.bert.port, config.bert.channel)
    except BaseException as error:
        run_dir = _persist_failure_run(
            plan=plan, output_root=args.output_root, started=started,
            error=error, initial_snapshot=initial_snapshot,
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise CharacterizationError(
            f"settle run failed; failure audit written to {run_dir}: {error}"
        ) from error
    if final_bert.taps != initial_bert.taps:
        result["status"] = "RECOVERY_REQUIRED"
        result["bert_error"] = "BERT FIR changed unexpectedly; no FIR write was issued"
    final_snapshot = {
        "bert": _bert_document(final_bert),
        "i2c_parameters": _field_values_document(final_fields),
        "restoration_succeeded": result.get("restoration_succeeded"),
    }
    run_dir = _persist_write_run(
        plan=plan, result=result, initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot, output_root=args.output_root,
        started=started,
    )
    print(f"Status / 状态: {result['status']}")
    print(f"Artifacts / 产物: {run_dir}")
    return 0 if result["status"] == "COMPLETED_RESTORED" else 4


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_live_hardware_config(args.config)
        handlers = {
            "baseline-plan": _run_baseline_plan,
            "baseline-run": _run_baseline,
            "same-value-plan": _run_same_plan,
            "same-value-run": _run_same,
            "settle-plan": _run_settle_plan,
            "settle-run": _run_settle,
        }
        return handlers[args.command](args, config)
    except (CharacterizationError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR / 错误: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
