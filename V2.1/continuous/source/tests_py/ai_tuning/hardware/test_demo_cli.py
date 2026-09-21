from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from vcsel_ai_tuning.hardware import demo_cli


REPO_ROOT = Path(__file__).resolve().parents[3]
FIRST_CONFIG = REPO_ROOT / "configs" / "fir_demo_first.json"
ITERATIVE_CONFIG = REPO_ROOT / "configs" / "fir_demo_iterative.json"

FIRST_DCA_TAPS = [
    0.008756,
    0.079768,
    -0.291888,
    1.336349,
    -0.084736,
    0.02357,
    -0.016326,
    -0.02607,
    -0.029423,
]


def _read_artifact(output_dir: Path) -> dict[str, object]:
    return json.loads((output_dir / "fir_proposal.json").read_text(encoding="utf-8"))


def _fixture_config(*, mode: str = "first", manual_main: int = 90) -> dict[str, object]:
    return {
        "mode": mode,
        "manual_main": manual_main,
        "dca": {
            "idn": "Keysight Technologies,N1000A,MY12345678,A.07.50",
            "tdecq_db": 2.75,
            "status": "CORR",
            "details": "No errors",
            "function": 1,
            "taps": FIRST_DCA_TAPS,
            "precursor_count": 3,
        },
        "bert": {
            "port": "port1",
            "channel": 0,
            "upper_nlc": 0,
            "lower_nlc": 0,
            "current_fir": None,
        },
    }


def test_fixture_first_real_config_writes_complete_proposal_json(tmp_path, capsys):
    output_dir = tmp_path / "first"

    exit_code = demo_cli.main(
        ["fixture", "--input", str(FIRST_CONFIG), "--output", str(output_dir)]
    )

    assert exit_code == 0
    document = _read_artifact(output_dir)
    assert document["schema_version"] == "1.0"
    assert document["source"]["kind"] == "fixture"
    assert document["dca"]["tdecq_db"] == pytest.approx(2.75)
    assert document["dca"]["raw_responses"]
    assert document["calculation"]["final_taps"] == [1, 5, -20, 90, -6, 2, -1]
    assert document["calculation"]["l1"] == 125
    assert document["calculation"]["valid"] is True
    assert set(document["calculation"]) == {
        "mode",
        "manual_main",
        "current_taps",
        "dca_taps",
        "dca_main_index",
        "combined_main_index",
        "normalization_denominator",
        "full_vector",
        "retained",
        "padded_left_count",
        "padded_right_count",
        "discarded_left",
        "discarded_right",
        "discarded_l1_ratio",
        "scaled",
        "final_taps",
        "l1",
        "valid",
        "reason",
        "max_feasible_main",
    }
    assert document["bert"]["command_preview"] == (
        "setFir port1 0 -20 5 1 90 -6 2 -1 0 0"
    )
    terminal = capsys.readouterr().out
    assert "TDECQ" in terminal
    assert "模式" in terminal
    assert "7 taps" in terminal
    assert "L1" in terminal
    assert "有效性" in terminal
    assert "舍弃比例" in terminal
    assert "命令预览" in terminal


def test_fixture_iterative_real_config_writes_expected_taps(tmp_path):
    output_dir = tmp_path / "iterative"

    exit_code = demo_cli.main(
        ["fixture", "--input", str(ITERATIVE_CONFIG), "--output", str(output_dir)]
    )

    assert exit_code == 0
    document = _read_artifact(output_dir)
    assert document["calculation"]["mode"] == "iterative"
    assert document["calculation"]["current_taps"] == [0, 6, -28, 112, -8, -10, 0]
    assert document["calculation"]["final_taps"] == [0, 6, -26, 104, -7, -8, -2]
    assert document["calculation"]["l1"] == 153
    assert document["bert"]["command_preview"] == (
        "setFir port1 0 -26 6 0 104 -7 -8 -2 0 0"
    )


def test_invalid_budget_returns_exit_2_keeps_artifact_and_has_no_command(
    tmp_path, capsys
):
    config_path = tmp_path / "over-limit.json"
    config_path.write_text(
        json.dumps(_fixture_config(manual_main=124)), encoding="utf-8"
    )
    output_dir = tmp_path / "invalid"

    exit_code = demo_cli.main(
        ["fixture", "--input", str(config_path), "--output", str(output_dir)]
    )

    assert exit_code == 2
    document = _read_artifact(output_dir)
    assert document["calculation"]["valid"] is False
    assert document["calculation"]["l1"] == 171
    assert document["calculation"]["max_feasible_main"] == 123
    assert document["bert"]["command_preview"] is None
    terminal = capsys.readouterr().out
    assert "最大可行 Main: 123" in terminal


def test_parser_has_no_apply_option():
    parser = demo_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fixture",
                "--input",
                str(FIRST_CONFIG),
                "--output",
                "out",
                "--apply",
            ]
        )

    assert "--apply" not in parser.format_help()


def test_flexdca_iterative_requires_exactly_seven_current_fir_values(
    tmp_path, capsys, monkeypatch
):
    opened = False

    @contextmanager
    def must_not_open(resource):
        nonlocal opened
        opened = True
        yield object()

    monkeypatch.setattr(demo_cli, "open_visa_session", must_not_open)

    exit_code = demo_cli.main(
        [
            "flexdca",
            "--visa-resource",
            "TCPIP0::192.0.2.1::inst0::INSTR",
            "--mode",
            "iterative",
            "--main",
            "104",
            "--upper-nlc",
            "0",
            "--lower-nlc",
            "0",
            "--current-fir",
            "0,6,-28,112,-8,-10",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert opened is False
    assert "恰好 7 个整数" in capsys.readouterr().err
    assert not (tmp_path / "fir_proposal.json").exists()


def test_flexdca_parser_requires_explicit_nlc_values(tmp_path):
    parser = demo_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "flexdca",
                "--visa-resource",
                "TCPIP0::192.0.2.1::inst0::INSTR",
                "--mode",
                "first",
                "--main",
                "90",
                "--output",
                str(tmp_path),
            ]
        )


def test_flexdca_uses_read_only_fake_visa_context_and_preserves_raw_audit(
    tmp_path, monkeypatch
):
    raw_responses = {
        ":MEASure:EYE:TDEQ:SOURce?": "FUNC1\n",
        "*IDN?": "Keysight Technologies,N1000A,MY12345678,A.07.50\n",
        ":MEASure:EYE:TDEQ?": "2.75\n",
        ":MEASure:EYE:TDEQ:STATus?": "CORR\n",
        ":MEASure:EYE:TDEQ:STATus:DETails?": "No errors\n",
        ":SPRocess1:TEQualizer:TAPS?": ",".join(str(tap) for tap in FIRST_DCA_TAPS),
        ":SPRocess1:TEQualizer:NPRecursors?": "3\n",
    }

    class FakeVisaSession:
        def __init__(self):
            self.commands = []
            self.writes = []

        def write(self, command):
            self.writes.append(command)

        def query(self, command):
            self.commands.append(command)
            return raw_responses[command]

    session = FakeVisaSession()
    opened_resources = []

    @contextmanager
    def fake_open_visa_session(resource):
        opened_resources.append(resource)
        yield session

    monkeypatch.setattr(demo_cli, "open_visa_session", fake_open_visa_session)
    output_dir = tmp_path / "live"

    exit_code = demo_cli.main(
        [
            "flexdca",
            "--visa-resource",
            "TCPIP0::192.0.2.1::inst0::INSTR",
            "--mode",
            "first",
            "--main",
            "90",
            "--upper-nlc",
            "7",
            "--lower-nlc",
            "-3",
            "--function",
            "1",
            "--port",
            "port1",
            "--channel",
            "0",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert opened_resources == ["TCPIP0::192.0.2.1::inst0::INSTR"]
    assert session.writes == [":MEASure:EYE:TDEQ:SOURce FUNC1"]
    assert session.commands == [
        ":MEASure:EYE:TDEQ:SOURce?",
        ":MEASure:EYE:TDEQ:STATus?",
        ":MEASure:EYE:TDEQ:STATus:DETails?",
        ":MEASure:EYE:TDEQ?",
        "*IDN?",
        ":SPRocess1:TEQualizer:TAPS?",
        ":SPRocess1:TEQualizer:NPRecursors?",
    ]
    document = _read_artifact(output_dir)
    assert document["source"] == {
        "kind": "flexdca",
        "visa_resource": "TCPIP0::192.0.2.1::inst0::INSTR",
        "read_only": False,
        "dca_source_selection_only": True,
    }
    assert document["dca"]["measurement_source"] == "FUNC1"
    assert document["dca"]["verified_measurement_source"] == "FUNC1"
    assert document["dca"]["raw_responses"] == raw_responses
    assert document["calculation"]["final_taps"] == [1, 5, -20, 90, -6, 2, -1]
    assert document["bert"]["command_preview"] == (
        "setFir port1 0 -20 5 1 90 -6 2 -1 7 -3"
    )


def test_flexdca_instrument_error_returns_1_without_artifact(
    tmp_path, capsys, monkeypatch
):
    @contextmanager
    def broken_open_visa_session(resource):
        raise OSError("instrument unavailable")
        yield

    monkeypatch.setattr(demo_cli, "open_visa_session", broken_open_visa_session)

    exit_code = demo_cli.main(
        [
            "flexdca",
            "--visa-resource",
            "BAD::RESOURCE",
            "--mode",
            "first",
            "--main",
            "90",
            "--upper-nlc",
            "0",
            "--lower-nlc",
            "0",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert "instrument unavailable" in capsys.readouterr().err
    assert not (tmp_path / "fir_proposal.json").exists()


def test_fixture_dca_read_error_returns_1_without_artifact(tmp_path, capsys):
    config = _fixture_config()
    config["dca"]["status"] = "UNCORR"
    config_path = tmp_path / "uncorrected.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = demo_cli.main(
        ["fixture", "--input", str(config_path), "--output", str(output_dir)]
    )

    assert exit_code == 1
    assert "UNCORR" in capsys.readouterr().err
    assert not (output_dir / "fir_proposal.json").exists()
