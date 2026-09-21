from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.hardware.characterization import create_baseline_plan
from vcsel_ai_tuning.hardware.characterization_cli import (
    _persist_failure_run,
    _persist_write_run,
    _run_same,
    _run_settle,
    build_parser,
    main,
)
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_CONFIG = REPO_ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
ARTIFACT_FILES = {
    "plan.json",
    "events.jsonl",
    "measurements.csv",
    "measurements.json",
    "raw_responses.json",
    "initial_snapshot.json",
    "final_snapshot.json",
    "summary.json",
    "convergence_defaults.json",
    "report.md",
    "round_artifacts",
}


def _write_plan(kind: str = "same_value") -> dict[str, object]:
    return {
        "plan_id": "a" * 64,
        "kind": kind,
        "parameters": {
            "channel": 0,
            "meaningful_er_floor_db": 0.05,
            "meaningful_tdecq_floor_db": 0.05,
        },
        "creation_snapshot": {},
    }


def _write_result(status: str) -> dict[str, object]:
    restored = status in {"COMPLETED_RESTORED", "FAILED_RESTORED"}
    result: dict[str, object] = {
        "status": status,
        "measurements": [
            {
                "repeat_index": 1,
                "observation_valid": True,
                "er_db": 3.5,
                "er_status": "CORR",
                "tdecq_db": 2.4,
                "tdecq_status": "CORR",
                "total_duration_s": 0.25,
                "_round_started_monotonic_ns": time.monotonic_ns(),
            }
        ],
        "target_write_readback_duration_s": 0.01,
        "restore_write_readback_duration_s": 0.02,
        "candidate_round_time_s": 0.25,
        "restore_duration_s": 0.02,
        "screening_round_with_restore_s": 0.27,
        "restoration_succeeded": restored,
        "restore_error": None,
        "manual_recovery": None,
        "error": None,
        "prewrite_dca_idn": "FAKE,DCA",
        "dca_measurement_evidence": {"measurement_1": {"status": "CORR"}},
    }
    if status == "FAILED_RESTORED":
        result["error"] = "simulated write verification failure"
    if status == "RECOVERY_REQUIRED":
        result["restoration_succeeded"] = False
        result["restore_error"] = "simulated restore failure"
        result["manual_recovery"] = {"required": True, "action": "restore B0"}
        result["error"] = "simulated write verification failure"
    return result


@pytest.mark.parametrize(
    ("status", "expected_consistent", "expected_restored"),
    [
        ("VERIFIED", True, False),
        ("COMPLETED_RESTORED", True, True),
        ("FAILED_RESTORED", True, True),
        ("RECOVERY_REQUIRED", False, False),
    ],
)
def test_persist_write_run_returns_path_and_preserves_complete_execution_result(
    tmp_path, status, expected_consistent, expected_restored
):
    plan = _write_plan()
    result = _write_result(status)

    run_dir = _persist_write_run(
        plan=plan,
        result=result,
        initial_snapshot={"snapshot": "initial"},
        final_snapshot={"snapshot": "final"},
        output_root=tmp_path,
        started=datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc),
    )

    assert isinstance(run_dir, Path)
    assert run_dir.is_dir()
    assert {path.name for path in run_dir.iterdir()} == ARTIFACT_FILES
    raw = json.loads((run_dir / "raw_responses.json").read_text(encoding="utf-8"))
    assert raw == {"execution": result}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["i2c_write_readback_consistent"] is expected_consistent
    assert summary["restoration_succeeded"] is expected_restored
    assert summary["restore_duration_s"] == pytest.approx(0.02)
    assert summary["screening_round_with_restore_s"] == pytest.approx(0.27)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Candidate round" in report
    assert "Screening round with restore" in report
    event = json.loads(
        (run_dir / "events.jsonl").read_text(encoding="utf-8").strip()
    )
    assert event["event"] == "run_completed"
    assert event["status"] == status
    assert status in (run_dir / "report.md").read_text(encoding="utf-8")


class _FakeBert:
    def __init__(self) -> None:
        self.configuration = BertConfiguration(
            port="1",
            channel="0",
            taps=(0, 9, -28, 104, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
            raw_response="fake-readback",
        )

    def get_configuration(self, _port, _channel):
        return self.configuration


class _FakeWriter:
    def read_fields(self, _parameters, *, channel):
        assert channel == 0
        return []


@contextmanager
def _fake_context(value):
    yield value


def _cli_config():
    return SimpleNamespace(
        bert=SimpleNamespace(port="1", channel="0"),
        characterization=SimpleNamespace(parameters=()),
        dca=SimpleNamespace(resource="FAKE::DCA", timeout_ms=1),
    )


def _cli_args(tmp_path, plan_id):
    return SimpleNamespace(
        plan=tmp_path / "unused-plan.json",
        output_root=tmp_path,
        apply=True,
        confirmation_id=plan_id,
    )


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [("VERIFIED", 0), ("RECOVERY_REQUIRED", 4)],
)
def test_run_same_prints_existing_artifact_path_without_real_hardware(
    tmp_path, monkeypatch, capsys, status, expected_exit_code
):
    plan = _write_plan("same_value")
    result = _write_result(status)
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._load_checked_plan",
        lambda *_args: plan,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._probe_i2c",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._bert",
        lambda *_args: _FakeBert(),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._writer",
        lambda *_args: _fake_context(_FakeWriter()),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli.execute_same_value_plan",
        lambda *_args, **_kwargs: result,
    )

    exit_code = _run_same(
        _cli_args(tmp_path, str(plan["plan_id"])), _cli_config()
    )

    run_dirs = list(tmp_path.iterdir())
    output = capsys.readouterr().out
    assert exit_code == expected_exit_code
    assert len(run_dirs) == 1 and run_dirs[0].is_dir()
    assert str(run_dirs[0]) in output
    assert "Artifacts: None" not in output
    assert "产物: None" not in output


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [("COMPLETED_RESTORED", 0), ("FAILED_RESTORED", 4)],
)
def test_run_settle_prints_existing_artifact_path_without_real_hardware(
    tmp_path, monkeypatch, capsys, status, expected_exit_code
):
    plan = _write_plan("settle")
    plan["measurement_plan"] = {"plan_id": "b" * 64}
    result = _write_result(status)
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._load_checked_plan",
        lambda *_args: plan,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli.validate_plan_against_config",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._probe_i2c",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._bert",
        lambda *_args: _FakeBert(),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._writer",
        lambda *_args: _fake_context(_FakeWriter()),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli.open_visa_session",
        lambda *_args, **_kwargs: _fake_context(object()),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._verify_dca_identity",
        lambda *_args: "FAKE,DCA",
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli.execute_settle_plan",
        lambda *_args, **_kwargs: result,
    )

    exit_code = _run_settle(
        _cli_args(tmp_path, str(plan["plan_id"])), _cli_config()
    )

    run_dirs = list(tmp_path.iterdir())
    output = capsys.readouterr().out
    assert exit_code == expected_exit_code
    assert len(run_dirs) == 1 and run_dirs[0].is_dir()
    assert str(run_dirs[0]) in output
    assert "Artifacts: None" not in output
    assert "产物: None" not in output


def test_baseline_plan_is_offline_and_writes_hashed_plan(tmp_path, monkeypatch):
    def hardware_forbidden(*_args, **_kwargs):
        raise AssertionError("baseline-plan must not access hardware")

    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._probe_i2c",
        hardware_forbidden,
    )
    output = tmp_path / "baseline.json"

    exit_code = main(
        [
            "baseline-plan",
            "--config",
            str(LIVE_CONFIG),
            "--repeats",
            "5",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["kind"] == "baseline"
    assert document["parameters"]["repeats"] == 5
    assert len(document["plan_id"]) == 64


@pytest.mark.parametrize(
    "argv",
    [
        ["baseline-run", "--plan", "p.json", "--confirmation-id", "x"],
        ["same-value-run", "--plan", "p.json", "--confirmation-id", "x"],
        ["settle-run", "--plan", "p.json", "--confirmation-id", "x"],
    ],
)
def test_run_commands_require_explicit_mutation_flag(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_settle_plan_requires_parameter_target_and_delays():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["settle-plan", "--parameter", "eq"])
    parsed = parser.parse_args(
        [
            "settle-plan",
            "--parameter",
            "eq",
            "--target-value",
            "23",
            "--settle-delays",
            "0.1,0.5,1.0",
        ]
    )
    assert parsed.parameter == "eq"
    assert parsed.target_value == 23
    assert parsed.settle_delays == (0.1, 0.5, 1.0)


def test_tampered_baseline_plan_fails_before_any_hardware_access(
    tmp_path, monkeypatch
):
    config = load_live_hardware_config(LIVE_CONFIG)
    plan = create_baseline_plan(config, repeats=1)
    plan["parameters"]["repeats"] = 99
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.characterization_cli._probe_i2c",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardware must not be reached")
        ),
    )
    assert (
        main(
            [
                "baseline-run",
                "--config",
                str(LIVE_CONFIG),
                "--plan",
                str(path),
                "--execute",
                "--confirmation-id",
                plan["plan_id"],
            ]
        )
        == 2
    )


def test_failure_audit_contains_complete_artifact_schema(tmp_path):
    config = load_live_hardware_config(LIVE_CONFIG)
    plan = create_baseline_plan(config, repeats=1)
    run_dir = _persist_failure_run(
        plan=plan,
        output_root=tmp_path,
        started=datetime.now(timezone.utc),
        error=RuntimeError("simulated acquisition failure"),
    )
    assert {path.name for path in run_dir.iterdir()} == {
        "plan.json",
        "events.jsonl",
        "measurements.csv",
        "measurements.json",
        "raw_responses.json",
        "initial_snapshot.json",
        "final_snapshot.json",
        "summary.json",
        "convergence_defaults.json",
        "report.md",
    }
