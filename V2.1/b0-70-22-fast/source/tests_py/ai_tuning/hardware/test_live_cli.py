from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.hardware import live_cli
from vcsel_ai_tuning.hardware.flexdca import DcaSnapshot
from vcsel_ai_tuning.hardware.live_workflow import HardwareSnapshot
from vcsel_ai_tuning.hardware.vd1124 import Vd1124ProbeResult
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


def _snapshot() -> HardwareSnapshot:
    dca = DcaSnapshot(
        timestamp_utc=datetime(2026, 7, 21, 4, 0, tzinfo=timezone.utc),
        function=1,
        measurement_source="FUNC1",
        verified_measurement_source="FUNC1",
        idn="Keysight Technologies,N1010A,Nate-Rog,A.08.32.13",
        tdecq_db=3.89,
        status="CORR",
        details="No errors",
        taps=(
            0.014569,
            -0.045870,
            1.089082,
            0.030131,
            0.011441,
            -0.004040,
            -0.037609,
            -0.029014,
            -0.028690,
        ),
        precursor_count=2,
        main_index=2,
        raw_responses={"*IDN?": "raw-idn"},
    )
    bert = BertConfiguration(
        port="port1",
        channel="0",
        taps=(0, 5, -26, 110, -6, -6, -1),
        upper_nlc=6,
        lower_nlc=6,
        raw_response="raw BERT configuration",
    )
    i2c = Vd1124ProbeResult(
        port="COM7", chip_id=0x0B, chip_revision=0x0D, matched=True
    )
    return HardwareSnapshot(
        dca=dca,
        bert_mode="PAM4 53.125 1lanePerPort",
        bert=bert,
        i2c=i2c,
    )


def _config():
    return SimpleNamespace(
        dca=SimpleNamespace(
            resource="TCPIP0::Nate-Rog::hislip0,4880::INSTR",
            function=1,
            timeout_ms=60_000,
        ),
        characterization=SimpleNamespace(
            tdecq=SimpleNamespace(source="FUNC1")
        ),
        bert=SimpleNamespace(
            ip="192.168.3.2",
            local_ip="192.168.3.1",
            udp_port=9000,
            port="port1",
            channel="0",
            timeout_s=2.0,
        ),
        i2c=SimpleNamespace(
            port="COM7",
            baudrate=115_200,
            device_address=0xA6,
            chip_id_register=0x0004,
            expected_chip_id=0x0B,
            chip_revision_register=0x0005,
            expected_chip_revision=0x0D,
        ),
    )


def test_parser_offers_all_live_workflow_commands():
    parser = live_cli.build_parser()

    assert {action.dest for action in parser._actions} >= {"command"}
    help_text = parser.format_help()
    assert "probe" in help_text
    assert "propose" in help_text
    assert "apply" in help_text
    assert "restore" in help_text
    assert "measure-plan" in help_text
    assert "measure" in help_text


def test_apply_parser_requires_both_explicit_arguments(tmp_path):
    parser = live_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["apply", "--proposal", str(tmp_path / "p.json")])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["apply", "--proposal", str(tmp_path / "p.json"), "--apply"]
        )


def test_probe_records_source_selection_only_three_module_snapshot(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: _config())
    monkeypatch.setattr(live_cli, "_collect_live", lambda config: _snapshot())
    output = tmp_path / "probe.json"

    exit_code = live_cli.main(["probe", "--output", str(output)])

    assert exit_code == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["read_only"] is False
    assert document["dca_source_selection_only"] is True
    assert document["dca"]["tdecq_db"] == pytest.approx(3.89)
    assert document["dca"]["measurement_source"] == "FUNC1"
    assert document["dca"]["verified_measurement_source"] == "FUNC1"
    assert document["bert"]["taps"] == [0, 5, -26, 110, -6, -6, -1]
    assert document["i2c"]["chip_id"] == 0x0B
    assert "DCA 仅选择并验证 TDECQ source" in capsys.readouterr().out


def test_propose_reads_live_state_and_saves_hashed_proposal(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: _config())
    monkeypatch.setattr(live_cli, "_collect_live", lambda config: _snapshot())
    output = tmp_path / "proposal.json"

    exit_code = live_cli.main(
        [
            "propose",
            "--mode",
            "first",
            "--main",
            "110",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    proposal = json.loads(output.read_text(encoding="utf-8"))
    assert len(proposal["proposal_id"]) == 64
    assert proposal["calculation"]["final_taps"] == [0, 1, -5, 110, 3, 1, 0]
    assert proposal["command_preview"] == "setFir port1 0 -5 1 0 110 3 1 0 6 6"
    terminal = capsys.readouterr().out
    assert proposal["proposal_id"] in terminal
    assert "没有写入 BERT" in terminal


def test_apply_passes_configured_bert_and_saves_receipt(
    tmp_path, monkeypatch, capsys
):
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(
        json.dumps({"proposal_id": "a" * 64}), encoding="utf-8"
    )
    receipt_path = tmp_path / "receipt.json"
    config = _config()
    fake_client = object()
    calls = []
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: config)
    monkeypatch.setattr(
        live_cli, "load_live_proposal", lambda path: {"proposal_id": "a" * 64}
    )
    monkeypatch.setattr(live_cli, "_create_bert_client", lambda cfg: fake_client)
    monkeypatch.setattr(
        live_cli,
        "_probe_i2c_only",
        lambda cfg: Vd1124ProbeResult("COM7", 0x0B, 0x0D, True),
    )

    def fake_apply(proposal, client, *, allow_apply, confirmation_id):
        calls.append((proposal, client, allow_apply, confirmation_id))
        return {
            "schema_version": "2.0",
            "proposal_id": "a" * 64,
            "status": "VERIFIED",
        }

    monkeypatch.setattr(live_cli, "apply_live_proposal", fake_apply)

    exit_code = live_cli.main(
        [
            "apply",
            "--proposal",
            str(proposal_path),
            "--apply",
            "--confirmation-id",
            "a" * 64,
            "--receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 0
    assert calls == [({"proposal_id": "a" * 64}, fake_client, True, "a" * 64)]
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "VERIFIED"
    assert "VERIFIED" in capsys.readouterr().out


def test_apply_blocks_before_bert_when_i2c_identity_changed(
    tmp_path, monkeypatch, capsys
):
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: _config())
    monkeypatch.setattr(live_cli, "load_live_proposal", lambda path: {"proposal_id": "x"})
    monkeypatch.setattr(
        live_cli,
        "_probe_i2c_only",
        lambda cfg: Vd1124ProbeResult("COM7", 0x00, 0x00, False),
    )
    created = False

    def must_not_create(_config):
        nonlocal created
        created = True

    monkeypatch.setattr(live_cli, "_create_bert_client", must_not_create)

    exit_code = live_cli.main(
        [
            "apply",
            "--proposal",
            str(proposal_path),
            "--apply",
            "--confirmation-id",
            "x",
        ]
    )

    assert exit_code == 1
    assert created is False
    assert "I2C" in capsys.readouterr().err


def test_restore_passes_verified_receipt_and_saves_restore_receipt(
    tmp_path, monkeypatch, capsys
):
    proposal_path = tmp_path / "proposal.json"
    write_receipt_path = tmp_path / "write_receipt.json"
    restore_receipt_path = tmp_path / "restore_receipt.json"
    proposal_path.write_text("{}", encoding="utf-8")
    write_receipt_path.write_text("{}", encoding="utf-8")
    proposal = {"proposal_id": "b" * 64}
    write_receipt = {"status": "VERIFIED", "proposal_id": "b" * 64}
    config = _config()
    fake_client = object()
    calls = []
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: config)
    monkeypatch.setattr(live_cli, "load_live_proposal", lambda path: proposal)
    monkeypatch.setattr(live_cli, "_read_json_object", lambda path: write_receipt)
    monkeypatch.setattr(live_cli, "_create_bert_client", lambda cfg: fake_client)
    monkeypatch.setattr(
        live_cli,
        "_probe_i2c_only",
        lambda cfg: Vd1124ProbeResult("COM7", 0x0B, 0x0D, True),
    )

    def fake_restore(proposal_arg, receipt_arg, client, **kwargs):
        calls.append((proposal_arg, receipt_arg, client, kwargs))
        return {
            "schema_version": "2.0",
            "proposal_id": "b" * 64,
            "status": "RESTORED",
        }

    monkeypatch.setattr(live_cli, "restore_live_receipt", fake_restore)

    exit_code = live_cli.main(
        [
            "restore",
            "--proposal",
            str(proposal_path),
            "--write-receipt",
            str(write_receipt_path),
            "--restore",
            "--confirmation-id",
            "b" * 64,
            "--receipt",
            str(restore_receipt_path),
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            proposal,
            write_receipt,
            fake_client,
            {"allow_restore": True, "confirmation_id": "b" * 64},
        )
    ]
    assert json.loads(restore_receipt_path.read_text(encoding="utf-8"))[
        "status"
    ] == "RESTORED"
    assert "RESTORED" in capsys.readouterr().out


def test_measure_plan_saves_hashed_non_mutating_plan(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "measurement_plan.json"
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: _config())

    def must_not_touch_hardware(_config):
        raise AssertionError("measure-plan must not create hardware clients")

    monkeypatch.setattr(live_cli, "_create_bert_client", must_not_touch_hardware)
    monkeypatch.setattr(live_cli, "_create_i2c_probe", must_not_touch_hardware)

    exit_code = live_cli.main(
        [
            "measure-plan",
            "--mode",
            "single",
            "--timeout",
            "180",
            "--poll-interval",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert len(plan["measurement_id"]) == 64
    assert plan["target"]["resource"] == _config().dca.resource
    assert plan["acquisition"]["commands"] == [
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
        ":ACQuire:STOP",
        ":ACQuire:CDISplay",
        ":ACQuire:SINGle",
    ]
    terminal = capsys.readouterr().out
    assert plan["measurement_id"] in terminal
    assert ":ACQuire:CDISplay" in terminal


def test_measure_checks_i2c_before_opening_dca_and_saves_receipt(
    tmp_path, monkeypatch, capsys
):
    plan_path = tmp_path / "measurement_plan.json"
    receipt_path = tmp_path / "measurement_receipt.json"
    plan_path.write_text("{}", encoding="utf-8")
    measurement_id = "c" * 64
    plan = {
        "measurement_id": measurement_id,
        "target": {
            "resource": _config().dca.resource,
            "function": 1,
            "measurement_source": "FUNC1",
        },
    }
    events = []
    fake_bert = object()
    fake_session = object()

    class FakeContext:
        def __enter__(self):
            events.append("dca_enter")
            return fake_session

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("dca_exit")

    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: _config())
    monkeypatch.setattr(live_cli, "load_measurement_plan", lambda path: plan)

    def fake_i2c(config):
        events.append("i2c")
        return Vd1124ProbeResult("COM7", 0x0B, 0x0D, True)

    def fake_bert_factory(config):
        events.append("bert")
        return fake_bert

    def fake_open(resource, *, timeout_ms):
        events.append(("open", resource, timeout_ms))
        return FakeContext()

    calls = []

    def fake_execute(plan_arg, dca, bert, i2c, **kwargs):
        calls.append((plan_arg, dca, bert, i2c, kwargs))
        return {
            "schema_version": "2.0",
            "measurement_id": measurement_id,
            "status": "MEASURED",
            "final_tdecq": {"tdecq_db": 4.84, "status": "CORR"},
        }

    monkeypatch.setattr(live_cli, "_probe_i2c_only", fake_i2c)
    monkeypatch.setattr(live_cli, "_create_bert_client", fake_bert_factory)
    monkeypatch.setattr(live_cli, "open_visa_session", fake_open)
    monkeypatch.setattr(live_cli, "execute_single_measurement", fake_execute)

    exit_code = live_cli.main(
        [
            "measure",
            "--plan",
            str(plan_path),
            "--execute",
            "--confirmation-id",
            measurement_id,
            "--receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 0
    assert events == [
        "i2c",
        "bert",
        ("open", _config().dca.resource, _config().dca.timeout_ms),
        "dca_enter",
        "dca_exit",
    ]
    assert calls[0][:3] == (plan, fake_session, fake_bert)
    assert calls[0][4] == {
        "bert_port": "port1",
        "bert_channel": "0",
        "allow_execute": True,
        "confirmation_id": measurement_id,
    }
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == (
        "MEASURED"
    )
    assert "4.84" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("MEASURED", 0),
        ("TIMED_OUT", 3),
        ("INVALIDATED", 3),
        ("FAILED", 4),
    ],
)
def test_measure_maps_receipt_status_to_exit_code(
    status, expected, tmp_path, monkeypatch
):
    measurement_id = "d" * 64
    plan = {
        "measurement_id": measurement_id,
        "target": {
            "resource": _config().dca.resource,
            "function": 1,
            "measurement_source": "FUNC1",
        },
    }

    class FakeContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda path: _config())
    monkeypatch.setattr(live_cli, "load_measurement_plan", lambda path: plan)
    monkeypatch.setattr(
        live_cli,
        "_probe_i2c_only",
        lambda config: Vd1124ProbeResult("COM7", 0x0B, 0x0D, True),
    )
    monkeypatch.setattr(live_cli, "_create_bert_client", lambda config: object())
    monkeypatch.setattr(live_cli, "open_visa_session", lambda *a, **k: FakeContext())
    monkeypatch.setattr(
        live_cli,
        "execute_single_measurement",
        lambda *a, **k: {
            "measurement_id": measurement_id,
            "status": status,
            "final_tdecq": None,
        },
    )

    exit_code = live_cli.main(
        [
            "measure",
            "--plan",
            str(tmp_path / "plan.json"),
            "--execute",
            "--confirmation-id",
            measurement_id,
        ]
    )

    assert exit_code == expected
