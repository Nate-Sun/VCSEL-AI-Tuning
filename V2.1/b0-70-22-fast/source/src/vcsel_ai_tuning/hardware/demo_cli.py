from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

from vcsel_ai_tuning.hardware.fir import (
    FirProposal,
    calculate_first_pass,
    calculate_iterative_pass,
    format_setfir_preview,
)
from vcsel_ai_tuning.hardware.flexdca import (
    DcaReadError,
    DcaSnapshot,
    open_visa_session,
    read_existing_tdecq,
)


SCHEMA_VERSION = "1.0"


class DemoInputError(ValueError):
    """Raised when a demo input cannot produce a safe FIR proposal."""


@dataclass(frozen=True)
class DemoRun:
    source: dict[str, object]
    snapshot: DcaSnapshot
    mode: str
    manual_main: int
    current_fir: tuple[int, ...] | None
    port: str
    channel: str | int
    upper_nlc: int
    lower_nlc: int


class _FixtureSession:
    def __init__(self, responses: Mapping[str, str]):
        self._responses = responses

    def query(self, command: str) -> str:
        return self._responses[command]

    def write(self, command: str) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcsel-fir-demo",
        description="只读 DCA TDECQ 并生成 7-tap FIR proposal；从不写入 BERT。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture", help="从离线 JSON fixture 计算")
    fixture.add_argument("--input", required=True, type=Path, help="fixture JSON")
    fixture.add_argument("--output", required=True, type=Path, help="输出目录")

    flexdca = subparsers.add_parser(
        "flexdca", help="只读 FlexDCA 当前已有的 TDECQ 结果"
    )
    flexdca.add_argument("--visa-resource", required=True)
    flexdca.add_argument("--mode", required=True, choices=("first", "iterative"))
    flexdca.add_argument("--main", required=True, type=int)
    flexdca.add_argument("--upper-nlc", required=True, type=int)
    flexdca.add_argument("--lower-nlc", required=True, type=int)
    flexdca.add_argument("--current-fir", help="迭代模式当前 FIR，7 个逗号分隔整数")
    flexdca.add_argument("--function", type=int, default=1)
    flexdca.add_argument("--port", default="port1")
    flexdca.add_argument("--channel", default="0")
    flexdca.add_argument("--output", required=True, type=Path, help="输出目录")

    return parser


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DemoInputError(f"{label} 必须是 JSON object")
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise DemoInputError(f"{label} 必须是整数")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoInputError(f"{label} 必须是非空字符串")
    return value


def _seven_integers(value: object, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or any(type(tap) is not int for tap in value)
    ):
        raise DemoInputError(f"{label} 必须包含恰好 7 个整数")
    return tuple(value)


def _current_fir_from_csv(mode: str, raw: str | None) -> tuple[int, ...] | None:
    if mode == "first":
        if raw is not None:
            raise DemoInputError("--current-fir 仅用于 iterative 模式")
        return None
    if raw is None:
        raise DemoInputError("iterative 模式的 --current-fir 必须包含恰好 7 个整数")
    fields = [field.strip() for field in raw.split(",")]
    if len(fields) != 7 or any(not field for field in fields):
        raise DemoInputError("--current-fir 必须包含恰好 7 个整数")
    try:
        return tuple(int(field, 10) for field in fields)
    except ValueError as error:
        raise DemoInputError("--current-fir 必须包含恰好 7 个整数") from error


def _fixture_snapshot(dca: Mapping[str, Any]) -> DcaSnapshot:
    required = (
        "idn",
        "tdecq_db",
        "status",
        "details",
        "function",
        "taps",
        "precursor_count",
    )
    missing = [name for name in required if name not in dca]
    if missing:
        raise DemoInputError(f"fixture.dca 缺少字段: {', '.join(missing)}")

    function = _require_int(dca["function"], "fixture.dca.function")
    idn = _require_text(dca["idn"], "fixture.dca.idn")
    status = _require_text(dca["status"], "fixture.dca.status")
    details = dca["details"]
    if not isinstance(details, str):
        raise DemoInputError("fixture.dca.details 必须是字符串")
    taps = dca["taps"]
    if not isinstance(taps, list) or not taps:
        raise DemoInputError("fixture.dca.taps 必须是非空数组")
    precursor_count = _require_int(
        dca["precursor_count"], "fixture.dca.precursor_count"
    )
    tdecq_db = dca["tdecq_db"]
    if isinstance(tdecq_db, bool) or not isinstance(tdecq_db, (int, float)):
        raise DemoInputError("fixture.dca.tdecq_db 必须是数字")

    responses = {
        ":MEASure:EYE:TDEQ:SOURce?": f"FUNC{function}",
        "*IDN?": idn,
        ":MEASure:EYE:TDEQ?": str(tdecq_db),
        ":MEASure:EYE:TDEQ:STATus?": status,
        ":MEASure:EYE:TDEQ:STATus:DETails?": details,
        f":SPRocess{function}:TEQualizer:TAPS?": ",".join(
            str(tap) for tap in taps
        ),
        f":SPRocess{function}:TEQualizer:NPRecursors?": str(precursor_count),
    }
    return read_existing_tdecq(
        _FixtureSession(responses),
        function=function,
        measurement_source=f"FUNC{function}",
    )


def _load_fixture(path: Path) -> DemoRun:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DemoInputError(f"无法读取 fixture {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DemoInputError(f"fixture 不是有效 JSON: {error}") from error

    root = _require_mapping(document, "fixture")
    missing = [name for name in ("mode", "manual_main", "dca", "bert") if name not in root]
    if missing:
        raise DemoInputError(f"fixture 缺少字段: {', '.join(missing)}")
    mode = root["mode"]
    if mode not in ("first", "iterative"):
        raise DemoInputError("fixture.mode 必须是 first 或 iterative")
    manual_main = _require_int(root["manual_main"], "fixture.manual_main")
    dca = _require_mapping(root["dca"], "fixture.dca")
    bert = _require_mapping(root["bert"], "fixture.bert")
    required_bert = (
        "port",
        "channel",
        "upper_nlc",
        "lower_nlc",
        "current_fir",
    )
    missing_bert = [name for name in required_bert if name not in bert]
    if missing_bert:
        raise DemoInputError(f"fixture.bert 缺少字段: {', '.join(missing_bert)}")

    port = _require_text(bert["port"], "fixture.bert.port")
    channel = bert["channel"]
    if isinstance(channel, bool) or not isinstance(channel, (str, int)):
        raise DemoInputError("fixture.bert.channel 必须是字符串或整数")
    upper_nlc = _require_int(bert["upper_nlc"], "fixture.bert.upper_nlc")
    lower_nlc = _require_int(bert["lower_nlc"], "fixture.bert.lower_nlc")
    if mode == "iterative":
        current_fir = _seven_integers(
            bert["current_fir"], "fixture.bert.current_fir"
        )
    else:
        if bert["current_fir"] is not None:
            raise DemoInputError("first 模式的 fixture.bert.current_fir 必须是 null")
        current_fir = None

    return DemoRun(
        source={"kind": "fixture", "input": str(path), "read_only": True},
        snapshot=_fixture_snapshot(dca),
        mode=mode,
        manual_main=manual_main,
        current_fir=current_fir,
        port=port,
        channel=channel,
        upper_nlc=upper_nlc,
        lower_nlc=lower_nlc,
    )


def _load_flexdca(args: argparse.Namespace) -> DemoRun:
    current_fir = _current_fir_from_csv(args.mode, args.current_fir)
    try:
        with open_visa_session(args.visa_resource) as session:
            snapshot = read_existing_tdecq(
                session,
                function=args.function,
                measurement_source=f"FUNC{args.function}",
            )
    except Exception as error:
        raise DemoInputError(f"FlexDCA 读取失败: {error}") from error

    return DemoRun(
        source={
            "kind": "flexdca",
            "visa_resource": args.visa_resource,
            "read_only": False,
            "dca_source_selection_only": True,
        },
        snapshot=snapshot,
        mode=args.mode,
        manual_main=args.main,
        current_fir=current_fir,
        port=args.port,
        channel=args.channel,
        upper_nlc=args.upper_nlc,
        lower_nlc=args.lower_nlc,
    )


def _calculate(run: DemoRun) -> FirProposal:
    if run.mode == "first":
        return calculate_first_pass(
            run.snapshot.taps,
            dca_main_index=run.snapshot.main_index,
            manual_main=run.manual_main,
        )
    if run.current_fir is None:
        raise DemoInputError("iterative 模式必须提供当前 FIR")
    return calculate_iterative_pass(
        run.current_fir,
        run.snapshot.taps,
        dca_main_index=run.snapshot.main_index,
        manual_main=run.manual_main,
    )


def _snapshot_document(snapshot: DcaSnapshot) -> dict[str, object]:
    return {
        "timestamp_utc": snapshot.timestamp_utc.isoformat(),
        "function": snapshot.function,
        "measurement_source": snapshot.measurement_source,
        "verified_measurement_source": snapshot.verified_measurement_source,
        "idn": snapshot.idn,
        "tdecq_db": snapshot.tdecq_db,
        "status": snapshot.status,
        "details": snapshot.details,
        "taps": list(snapshot.taps),
        "precursor_count": snapshot.precursor_count,
        "main_index": snapshot.main_index,
        "raw_responses": {
            str(command): str(response)
            for command, response in snapshot.raw_responses.items()
        },
    }


def _artifact_document(run: DemoRun, proposal: FirProposal) -> dict[str, object]:
    command_preview = None
    if proposal.valid:
        command_preview = format_setfir_preview(
            run.port,
            run.channel,
            proposal.final_taps,
            run.upper_nlc,
            run.lower_nlc,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": run.source,
        "dca": _snapshot_document(run.snapshot),
        "calculation": asdict(proposal),
        "bert": {
            "port": run.port,
            "channel": run.channel,
            "upper_nlc": run.upper_nlc,
            "lower_nlc": run.lower_nlc,
            "current_fir": run.current_fir,
            "command_preview": command_preview,
        },
    }


def _write_artifact(output_dir: Path, document: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "fir_proposal.json"
    artifact.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact


def _print_summary(
    run: DemoRun,
    proposal: FirProposal,
    command_preview: object,
    artifact: Path,
) -> None:
    print(f"TDECQ: {run.snapshot.tdecq_db:.6g} dB")
    print(f"模式: {proposal.mode}")
    print(f"7 taps: {list(proposal.final_taps)}")
    print(f"L1: {proposal.l1} / 170")
    print(f"有效性: {'有效' if proposal.valid else '无效'}")
    print(f"舍弃比例: {proposal.discarded_l1_ratio:.2%}")
    if proposal.reason:
        print(f"原因: {proposal.reason}")
    if proposal.max_feasible_main is not None:
        print(f"最大可行 Main: {proposal.max_feasible_main}")
    if command_preview is None:
        print("命令预览: 未生成（proposal 无效）")
    else:
        print(f"命令预览: {command_preview}")
    print(f"JSON: {artifact}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run = _load_fixture(args.input) if args.command == "fixture" else _load_flexdca(args)
        proposal = _calculate(run)
        document = _artifact_document(run, proposal)
        artifact = _write_artifact(args.output, document)
        _print_summary(
            run,
            proposal,
            document["bert"]["command_preview"],
            artifact,
        )
        return 0 if proposal.valid else 2
    except (DemoInputError, DcaReadError, OSError, TypeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
