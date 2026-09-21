"""Command line entry point for the commissioned real-hardware FIR workflow."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Sequence

from vcsel_ai_tuning.hardware.dca_acquisition import (
    create_measurement_plan,
    execute_single_measurement,
    load_measurement_plan,
    save_measurement_plan,
)
from vcsel_ai_tuning.hardware.flexdca import open_visa_session
from vcsel_ai_tuning.hardware.live_config import (
    LiveHardwareConfig,
    load_live_hardware_config,
)
from vcsel_ai_tuning.hardware.live_workflow import (
    apply_live_proposal,
    collect_hardware_snapshot,
    create_live_proposal,
    hardware_snapshot_document,
    load_live_proposal,
    restore_live_receipt,
    save_live_proposal,
)
from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe, Vd1124ProbeResult
from vcsel_ai_tuning.hardware.vit_bert import VitBertClient


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
DEFAULT_PROBE_OUTPUT = REPO_ROOT / "artifacts" / "fir_runs" / "live_probe.json"
DEFAULT_PROPOSAL_OUTPUT = (
    REPO_ROOT / "artifacts" / "fir_runs" / "latest" / "proposal.json"
)
DEFAULT_MEASUREMENT_PLAN_OUTPUT = (
    REPO_ROOT / "artifacts" / "fir_runs" / "latest" / "measurement_plan.json"
)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"真实端口 YAML（默认 {DEFAULT_CONFIG}）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcsel-live-fir",
        description="读取 DCA TDECQ、检查 VD1124、生成并安全下发 BERT FIR。",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe", help="只读检查 DCA、BERT 和 I2C")
    _add_config_argument(probe)
    probe.add_argument("--output", type=Path, default=DEFAULT_PROBE_OUTPUT)

    propose = commands.add_parser(
        "propose", help="只读生成 FIR proposal，不写 BERT"
    )
    _add_config_argument(propose)
    propose.add_argument("--mode", choices=("first", "iterative"), required=True)
    propose.add_argument("--main", type=int, required=True, help="人工指定 Main")
    propose.add_argument("--output", type=Path, default=DEFAULT_PROPOSAL_OUTPUT)

    apply = commands.add_parser("apply", help="显式批准后下发已保存 proposal")
    _add_config_argument(apply)
    apply.add_argument("--proposal", type=Path, required=True)
    apply.add_argument(
        "--apply",
        action="store_true",
        required=True,
        help="明确允许一次真实 BERT 写入",
    )
    apply.add_argument(
        "--confirmation-id",
        required=True,
        help="必须与 proposal 中的完整 ID 完全一致",
    )
    apply.add_argument("--receipt", type=Path)

    restore = commands.add_parser(
        "restore", help="Safely restore the FIR saved before a verified write"
    )
    _add_config_argument(restore)
    restore.add_argument("--proposal", type=Path, required=True)
    restore.add_argument("--write-receipt", type=Path, required=True)
    restore.add_argument(
        "--restore",
        action="store_true",
        required=True,
        help="Explicitly allow one real BERT restore write",
    )
    restore.add_argument(
        "--confirmation-id",
        required=True,
        help="Must exactly match the complete proposal ID",
    )
    restore.add_argument("--receipt", type=Path)

    measure_plan = commands.add_parser(
        "measure-plan", help="Create a non-mutating DCA Single acquisition plan"
    )
    _add_config_argument(measure_plan)
    measure_plan.add_argument("--mode", choices=("single",), required=True)
    measure_plan.add_argument("--timeout", type=float, default=180.0)
    measure_plan.add_argument("--poll-interval", type=float, default=1.0)
    measure_plan.add_argument(
        "--output", type=Path, default=DEFAULT_MEASUREMENT_PLAN_OUTPUT
    )

    measure = commands.add_parser(
        "measure", help="Execute one confirmed DCA Single acquisition plan"
    )
    _add_config_argument(measure)
    measure.add_argument("--plan", type=Path, required=True)
    measure.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="Explicitly allow Stop, Clear, and Single on the real DCA",
    )
    measure.add_argument(
        "--confirmation-id",
        required=True,
        help="Must exactly match the complete measurement ID",
    )
    measure.add_argument("--receipt", type=Path)
    return parser


def _create_bert_client(config: LiveHardwareConfig) -> VitBertClient:
    return VitBertClient(
        config.bert.ip,
        config.bert.udp_port,
        local_ip=config.bert.local_ip,
        timeout_s=config.bert.timeout_s,
    )


def _create_i2c_probe(config: LiveHardwareConfig) -> Vd1124Probe:
    return Vd1124Probe(
        port=config.i2c.port,
        baudrate=config.i2c.baudrate,
        address=config.i2c.device_address,
        chip_id_register=config.i2c.chip_id_register,
        expected_chip_id=config.i2c.expected_chip_id,
        chip_revision_register=config.i2c.chip_revision_register,
        expected_chip_revision=config.i2c.expected_chip_revision,
    )


def _collect_live(config: LiveHardwareConfig):
    bert = _create_bert_client(config)
    i2c = _create_i2c_probe(config)
    with open_visa_session(
        config.dca.resource, timeout_ms=config.dca.timeout_ms
    ) as dca_session, i2c:
        return collect_hardware_snapshot(
            dca_session,
            bert,
            i2c,
            dca_function=config.dca.function,
            dca_measurement_source=config.characterization.tdecq.source,
            bert_port=config.bert.port,
            bert_channel=config.bert.channel,
        )


def _probe_i2c_only(config: LiveHardwareConfig) -> Vd1124ProbeResult:
    with _create_i2c_probe(config) as i2c:
        return i2c.probe()


def _write_json(document: object, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"JSON document must contain an object: {path}")
    return document


def _run_probe(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    snapshot = _collect_live(config)
    output = _write_json(hardware_snapshot_document(snapshot), args.output)
    if not snapshot.i2c.matched:
        raise RuntimeError("I2C 芯片身份与 VD1124 配置不匹配")
    print("三模块检查通过（DCA 仅选择并验证 TDECQ source）：")
    print(f"  DCA: {snapshot.dca.idn}")
    print(f"  TDECQ: {snapshot.dca.tdecq_db:.6g} dB ({snapshot.dca.status})")
    print(f"  BERT: {snapshot.bert_mode}, FIR={list(snapshot.bert.taps)}")
    print(
        "  I2C: "
        f"{snapshot.i2c.port}, ID=0x{snapshot.i2c.chip_id:02X}, "
        f"REV=0x{snapshot.i2c.chip_revision:02X}"
    )
    print(f"  记录: {output}")
    return 0


def _run_propose(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    snapshot = _collect_live(config)
    proposal = create_live_proposal(
        snapshot,
        mode=args.mode,
        manual_main=args.main,
        bert_host=config.bert.ip,
        bert_udp_port=config.bert.udp_port,
    )
    output = save_live_proposal(proposal, args.output)
    calculation = proposal["calculation"]
    assert isinstance(calculation, dict)
    print(f"Proposal ID: {proposal['proposal_id']}")
    print(f"TDECQ: {snapshot.dca.tdecq_db:.6g} dB")
    print(f"当前 BERT FIR: {list(snapshot.bert.taps)}")
    print(f"建议 FIR: {calculation['final_taps']}")
    print(f"L1: {calculation['l1']} / 170")
    print(f"命令预览: {proposal['command_preview']}")
    print(f"Proposal: {output}")
    print("本命令只读取并计算，没有写入 BERT。")
    return 0 if calculation.get("valid") is True else 2


def _run_apply(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    proposal = load_live_proposal(args.proposal)
    i2c = _probe_i2c_only(config)
    if not i2c.matched:
        raise RuntimeError(
            "I2C 芯片身份已变化，停止 BERT 写入："
            f"ID=0x{i2c.chip_id:02X}, REV=0x{i2c.chip_revision:02X}"
        )
    bert = _create_bert_client(config)
    receipt = apply_live_proposal(
        proposal,
        bert,
        allow_apply=args.apply,
        confirmation_id=args.confirmation_id,
    )
    receipt_path = args.receipt or args.proposal.with_name("write_receipt.json")
    _write_json(receipt, receipt_path)
    status = receipt["status"]
    print(f"BERT 写入结果: {status}")
    print(f"Receipt: {receipt_path}")
    if status == "VERIFIED":
        return 0
    if status == "ROLLED_BACK":
        return 3
    return 4


def _run_restore(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    proposal = load_live_proposal(args.proposal)
    write_receipt = _read_json_object(args.write_receipt)
    i2c = _probe_i2c_only(config)
    if not i2c.matched:
        raise RuntimeError(
            "I2C identity changed; refusing the BERT restore write: "
            f"ID=0x{i2c.chip_id:02X}, REV=0x{i2c.chip_revision:02X}"
        )
    bert = _create_bert_client(config)
    receipt = restore_live_receipt(
        proposal,
        write_receipt,
        bert,
        allow_restore=args.restore,
        confirmation_id=args.confirmation_id,
    )
    receipt_path = args.receipt or args.write_receipt.with_name(
        "restore_receipt.json"
    )
    _write_json(receipt, receipt_path)
    status = receipt["status"]
    print(f"BERT restore result: {status}")
    print(f"Receipt: {receipt_path}")
    if status == "RESTORED":
        return 0
    if status == "ROLLED_FORWARD":
        return 3
    return 4


def _run_measure_plan(
    args: argparse.Namespace, config: LiveHardwareConfig
) -> int:
    plan = create_measurement_plan(
        resource=config.dca.resource,
        function=config.dca.function,
        measurement_source=config.characterization.tdecq.source,
        mode=args.mode,
        timeout_s=args.timeout,
        poll_interval_s=args.poll_interval,
    )
    output = save_measurement_plan(plan, args.output)
    target = plan["target"]
    acquisition = plan["acquisition"]
    assert isinstance(target, Mapping)
    assert isinstance(acquisition, Mapping)
    print(f"Measurement ID: {plan['measurement_id']}")
    print(f"DCA target: {target['resource']}")
    print(f"TDECQ source: {target['measurement_source']}")
    print(
        f"Timeout / poll: {acquisition['timeout_s']} s / "
        f"{acquisition['poll_interval_s']} s"
    )
    print("DCA mutation preview:")
    commands = acquisition["commands"]
    assert isinstance(commands, list)
    for command in commands:
        print(f"  {command}")
    print(f"Measurement plan: {output}")
    print("No hardware was accessed by this command.")
    return 0


def _run_measure(args: argparse.Namespace, config: LiveHardwareConfig) -> int:
    plan = load_measurement_plan(args.plan)
    target = plan.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError("measurement plan target is invalid")
    expected_target = {
        "resource": config.dca.resource,
        "function": config.dca.function,
        "measurement_source": config.characterization.tdecq.source,
    }
    if dict(target) != expected_target:
        raise RuntimeError("measurement plan DCA target differs from live config")

    i2c = _probe_i2c_only(config)
    if not i2c.matched:
        raise RuntimeError(
            "I2C identity changed; refusing DCA acquisition: "
            f"ID=0x{i2c.chip_id:02X}, REV=0x{i2c.chip_revision:02X}"
        )
    bert = _create_bert_client(config)
    with open_visa_session(
        config.dca.resource, timeout_ms=config.dca.timeout_ms
    ) as dca_session:
        receipt = execute_single_measurement(
            plan,
            dca_session,
            bert,
            i2c,
            bert_port=config.bert.port,
            bert_channel=config.bert.channel,
            allow_execute=args.execute,
            confirmation_id=args.confirmation_id,
        )

    receipt_path = args.receipt or args.plan.with_name(
        "measurement_receipt.json"
    )
    _write_json(receipt, receipt_path)
    status = receipt["status"]
    print(f"DCA measurement result: {status}")
    final_tdecq = receipt.get("final_tdecq")
    if isinstance(final_tdecq, Mapping):
        print(
            f"TDECQ: {final_tdecq.get('tdecq_db')} dB "
            f"({final_tdecq.get('status')})"
        )
        print(f"Details: {final_tdecq.get('details')}")
    elif receipt.get("error"):
        print(f"Details: {receipt['error']}")
    print(f"Receipt: {receipt_path}")
    if status == "MEASURED":
        return 0
    if status in {"TIMED_OUT", "INVALIDATED"}:
        return 3
    return 4


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_live_hardware_config(args.config)
        if args.command == "probe":
            return _run_probe(args, config)
        if args.command == "propose":
            return _run_propose(args, config)
        if args.command == "apply":
            return _run_apply(args, config)
        if args.command == "restore":
            return _run_restore(args, config)
        if args.command == "measure-plan":
            return _run_measure_plan(args, config)
        return _run_measure(args, config)
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
