from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.campaign.models import FirSettings
from vcsel_ai_tuning.hardware.flexdca import DcaSnapshot
import vcsel_ai_tuning.hardware.live_workflow as live_workflow_module
from vcsel_ai_tuning.hardware.live_workflow import (
    LiveWorkflowError,
    apply_live_proposal,
    collect_hardware_snapshot,
    create_live_proposal,
    load_live_proposal,
    restore_live_receipt,
    save_live_proposal,
)
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


LIVE_DCA_TAPS = (
    0.014569,
    -0.045870,
    1.089082,
    0.030131,
    0.011441,
    -0.004040,
    -0.037609,
    -0.029014,
    -0.028690,
)
BEFORE_TAPS = (0, 5, -26, 110, -6, -6, -1)
TARGET_TAPS = (0, 1, -5, 110, 3, 1, 0)


def _dca_snapshot() -> DcaSnapshot:
    return DcaSnapshot(
        timestamp_utc=datetime(2026, 7, 21, 4, 0, tzinfo=timezone.utc),
        function=1,
        measurement_source="FUNC1",
        verified_measurement_source="FUNC1",
        idn="Keysight Technologies,N1010A,Nate-Rog,A.08.32.13",
        tdecq_db=3.89,
        status="CORR",
        details="No errors",
        taps=LIVE_DCA_TAPS,
        precursor_count=2,
        main_index=2,
        raw_responses={"*IDN?": "Keysight Technologies,N1010A,Nate-Rog,A.08.32.13"},
    )


def _bert_configuration(
    taps: tuple[int, ...] = BEFORE_TAPS,
    *,
    upper_nlc: int = 6,
    lower_nlc: int = 6,
) -> BertConfiguration:
    return BertConfiguration(
        port="port1",
        channel="0",
        taps=taps,
        upper_nlc=upper_nlc,
        lower_nlc=lower_nlc,
        raw_response="raw BERT configuration",
    )


def _campaign_bert_configuration(
    taps: tuple[int, ...] = BEFORE_TAPS,
    *,
    pattern_fields=("SSPRQ", "SSPRQ"),
):
    return SimpleNamespace(
        port="port1",
        channel="0",
        taps=taps,
        upper_nlc=6,
        lower_nlc=6,
        pattern_fields=pattern_fields,
        raw_response="raw BERT configuration",
    )


def _hardware_snapshot(*, i2c_matched: bool = True):
    return SimpleNamespace(
        dca=_dca_snapshot(),
        bert_mode="PAM4 53.125 1lanePerPort",
        bert=_bert_configuration(),
        i2c=SimpleNamespace(
            port="COM7", chip_id=0x0B, chip_revision=0x0D, matched=i2c_matched
        ),
    )


class ScriptedBertClient:
    def __init__(self, configurations):
        self.host = "192.168.3.2"
        self.port = 9000
        self.configurations = deque(configurations)
        self.get_calls = []
        self.set_calls = []
        self.responses = deque()

    def get_configuration(self, port, channel):
        self.get_calls.append((port, str(channel)))
        if not self.configurations:
            raise AssertionError("unexpected GetConfiguration")
        value = self.configurations.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def set_fir(self, port, channel, *, taps, upper_nlc, lower_nlc):
        self.set_calls.append(
            (port, str(channel), tuple(taps), upper_nlc, lower_nlc)
        )
        if self.responses:
            value = self.responses.popleft()
            if isinstance(value, Exception):
                raise value
            return value
        return "OK"


class CampaignBertClient:
    def __init__(self, configurations):
        self.host = "192.168.3.2"
        self.port = 9000
        self.configurations = deque(configurations)
        self.mode_calls = 0
        self.get_calls = []
        self.set_calls = []

    def get_mode(self):
        self.mode_calls += 1
        return "PAM4 53.125 1lanePerPort"

    def get_configuration(self, port, channel):
        self.get_calls.append((port, str(channel)))
        if not self.configurations:
            raise AssertionError("unexpected GetConfiguration")
        return self.configurations.popleft()

    def set_fir(self, port, channel, *, taps, upper_nlc, lower_nlc):
        self.set_calls.append(
            (port, str(channel), tuple(taps), upper_nlc, lower_nlc)
        )
        return "OK"


def _proposal():
    return create_live_proposal(
        _hardware_snapshot(),
        mode="first",
        manual_main=110,
        bert_host="192.168.3.2",
        bert_udp_port=9000,
    )


def _verified_receipt(proposal=None):
    proposal = _proposal() if proposal is None else proposal
    return {
        "schema_version": "2.0",
        "proposal_id": proposal["proposal_id"],
        "status": "VERIFIED",
        "command": proposal["command_preview"],
        "response": "SUCCESS",
        "before": proposal["bert_before"],
        "after": {
            "port": "port1",
            "channel": "0",
            "taps": list(TARGET_TAPS),
            "upper_nlc": 6,
            "lower_nlc": 6,
            "raw_response": "target raw",
        },
        "error": None,
        "rollback": None,
    }


def test_collect_hardware_snapshot_queries_each_module_once():
    class DcaSession:
        def __init__(self):
            self.responses = iter(_dca_snapshot().raw_responses.values())

        def query(self, command):
            values = {
                ":MEASure:EYE:TDEQ:SOURce?": "FUNC1",
                "*IDN?": "Keysight Technologies,N1010A,Nate-Rog,A.08.32.13",
                ":MEASure:EYE:TDEQ?": "3.89",
                ":MEASure:EYE:TDEQ:STATus?": "CORR",
                ":MEASure:EYE:TDEQ:STATus:DETails?": "No errors",
                ":SPRocess1:TEQualizer:TAPS?": ",".join(map(str, LIVE_DCA_TAPS)),
                ":SPRocess1:TEQualizer:NPRecursors?": "2",
            }
            return values[command]

        def write(self, command):
            assert command == ":MEASure:EYE:TDEQ:SOURce FUNC1"

    class Bert:
        def __init__(self):
            self.calls = []

        def get_mode(self):
            self.calls.append(("mode",))
            return "PAM4 53.125 1lanePerPort"

        def get_configuration(self, port, channel):
            self.calls.append(("configuration", port, channel))
            return _bert_configuration()

    class Probe:
        calls = 0

        def probe(self):
            self.calls += 1
            return _hardware_snapshot().i2c

    bert = Bert()
    probe = Probe()

    snapshot = collect_hardware_snapshot(
        DcaSession(),
        bert,
        probe,
        dca_function=1,
        dca_measurement_source="FUNC1",
        bert_port="port1",
        bert_channel="0",
    )

    assert snapshot.dca.tdecq_db == pytest.approx(3.89)
    assert snapshot.dca.measurement_source == "FUNC1"
    assert snapshot.dca.verified_measurement_source == "FUNC1"
    assert snapshot.bert_mode == "PAM4 53.125 1lanePerPort"
    assert snapshot.bert.taps == BEFORE_TAPS
    assert snapshot.i2c.matched is True
    assert bert.calls == [("mode",), ("configuration", "port1", "0")]
    assert probe.calls == 1


def test_create_live_proposal_uses_live_two_precursor_taps_and_preserves_nlc():
    proposal = _proposal()

    assert len(proposal["proposal_id"]) == 64
    assert proposal["schema_version"] == "2.0"
    assert proposal["dca"]["tdecq_db"] == pytest.approx(3.89)
    assert proposal["dca"]["measurement_source"] == "FUNC1"
    assert proposal["dca"]["verified_measurement_source"] == "FUNC1"
    assert proposal["i2c"] == {
        "port": "COM7",
        "chip_id": 0x0B,
        "chip_revision": 0x0D,
        "matched": True,
    }
    assert proposal["bert_before"]["taps"] == list(BEFORE_TAPS)
    assert proposal["calculation"]["final_taps"] == list(TARGET_TAPS)
    assert proposal["calculation"]["padded_left_count"] == 1
    assert proposal["command_preview"] == (
        "setFir port1 0 -5 1 0 110 3 1 0 6 6"
    )


def test_iterative_proposal_uses_current_bert_fir():
    proposal = create_live_proposal(
        _hardware_snapshot(),
        mode="iterative",
        manual_main=104,
        bert_host="192.168.3.2",
        bert_udp_port=9000,
    )

    assert proposal["calculation"]["mode"] == "iterative"
    assert proposal["calculation"]["current_taps"] == list(BEFORE_TAPS)


def test_unmatched_i2c_identity_blocks_proposal():
    with pytest.raises(LiveWorkflowError, match="I2C.*identity"):
        create_live_proposal(
            _hardware_snapshot(i2c_matched=False),
            mode="first",
            manual_main=110,
            bert_host="192.168.3.2",
            bert_udp_port=9000,
        )


def test_proposal_round_trip_and_hash_validation(tmp_path: Path):
    path = tmp_path / "proposal.json"
    proposal = _proposal()

    save_live_proposal(proposal, path)
    loaded = load_live_proposal(path)

    assert loaded == proposal
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["calculation"]["final_taps"][0] = 99
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(LiveWorkflowError, match="hash"):
        load_live_proposal(path)


@pytest.mark.parametrize(
    ("allow_apply", "confirmation_id", "reason"),
    [(False, None, "--apply"), (True, "wrong", "proposal ID")],
)
def test_apply_requires_both_explicit_gates(
    allow_apply, confirmation_id, reason
):
    proposal = _proposal()
    client = ScriptedBertClient([_bert_configuration()])

    with pytest.raises(LiveWorkflowError, match=reason):
        apply_live_proposal(
            proposal,
            client,
            allow_apply=allow_apply,
            confirmation_id=confirmation_id,
        )

    assert client.get_calls == []
    assert client.set_calls == []


def test_apply_rejects_stale_bert_without_writing():
    proposal = _proposal()
    stale = _bert_configuration(taps=(0, 5, -25, 110, -6, -6, -1))
    client = ScriptedBertClient([stale])

    with pytest.raises(LiveWorkflowError, match="stale"):
        apply_live_proposal(
            proposal,
            client,
            allow_apply=True,
            confirmation_id=proposal["proposal_id"],
        )

    assert client.set_calls == []


def test_apply_verifies_exact_readback():
    proposal = _proposal()
    client = ScriptedBertClient(
        [_bert_configuration(), _bert_configuration(taps=TARGET_TAPS)]
    )

    receipt = apply_live_proposal(
        proposal,
        client,
        allow_apply=True,
        confirmation_id=proposal["proposal_id"],
    )

    assert receipt["status"] == "VERIFIED"
    assert len(client.set_calls) == 1
    assert client.set_calls[0][2:] == (TARGET_TAPS, 6, 6)
    assert receipt["after"]["taps"] == list(TARGET_TAPS)


def test_readback_mismatch_rolls_back_once_and_verifies_old_state():
    proposal = _proposal()
    mismatch = _bert_configuration(taps=(0, 1, -5, 110, 3, 1, -1))
    client = ScriptedBertClient(
        [_bert_configuration(), mismatch, _bert_configuration()]
    )

    receipt = apply_live_proposal(
        proposal,
        client,
        allow_apply=True,
        confirmation_id=proposal["proposal_id"],
    )

    assert receipt["status"] == "ROLLED_BACK"
    assert len(client.set_calls) == 2
    assert client.set_calls[1][2:] == (BEFORE_TAPS, 6, 6)
    assert receipt["rollback"]["verified"] is True


def test_failed_rollback_returns_recovery_required_without_more_writes():
    proposal = _proposal()
    mismatch = _bert_configuration(taps=(0, 1, -5, 110, 3, 1, -1))
    still_wrong = _bert_configuration(taps=(0, 0, 0, 110, 0, 0, 0))
    client = ScriptedBertClient(
        [_bert_configuration(), mismatch, still_wrong]
    )

    receipt = apply_live_proposal(
        proposal,
        client,
        allow_apply=True,
        confirmation_id=proposal["proposal_id"],
    )

    assert receipt["status"] == "RECOVERY_REQUIRED"
    assert len(client.set_calls) == 2
    assert receipt["rollback"]["verified"] is False


@pytest.mark.parametrize(
    ("allow_restore", "confirmation_id", "reason"),
    [(False, None, "--restore"), (True, "wrong", "proposal ID")],
)
def test_restore_requires_both_explicit_gates(
    allow_restore, confirmation_id, reason
):
    proposal = _proposal()
    client = ScriptedBertClient([])

    with pytest.raises(LiveWorkflowError, match=reason):
        restore_live_receipt(
            proposal,
            _verified_receipt(proposal),
            client,
            allow_restore=allow_restore,
            confirmation_id=confirmation_id,
        )

    assert client.get_calls == []
    assert client.set_calls == []


def test_restore_rejects_receipt_or_current_state_mismatch_without_writing():
    proposal = _proposal()
    stale = _bert_configuration(taps=(0, 1, -5, 110, 3, 1, -1))
    client = ScriptedBertClient([stale])

    with pytest.raises(LiveWorkflowError, match="stale"):
        restore_live_receipt(
            proposal,
            _verified_receipt(proposal),
            client,
            allow_restore=True,
            confirmation_id=proposal["proposal_id"],
        )

    assert client.set_calls == []


def test_restore_verified_receipt_restores_before_state_exactly():
    proposal = _proposal()
    client = ScriptedBertClient(
        [_bert_configuration(taps=TARGET_TAPS), _bert_configuration()]
    )

    receipt = restore_live_receipt(
        proposal,
        _verified_receipt(proposal),
        client,
        allow_restore=True,
        confirmation_id=proposal["proposal_id"],
    )

    assert receipt["status"] == "RESTORED"
    assert client.set_calls == [("port1", "0", BEFORE_TAPS, 6, 6)]
    assert receipt["restored"]["taps"] == list(BEFORE_TAPS)


def test_restore_mismatch_rolls_forward_once_to_original_after_state():
    proposal = _proposal()
    mismatch = _bert_configuration(taps=(0, 5, -26, 110, -6, -6, 0))
    client = ScriptedBertClient(
        [
            _bert_configuration(taps=TARGET_TAPS),
            mismatch,
            _bert_configuration(taps=TARGET_TAPS),
        ]
    )

    receipt = restore_live_receipt(
        proposal,
        _verified_receipt(proposal),
        client,
        allow_restore=True,
        confirmation_id=proposal["proposal_id"],
    )

    assert receipt["status"] == "ROLLED_FORWARD"
    assert len(client.set_calls) == 2
    assert client.set_calls[1][2:] == (TARGET_TAPS, 6, 6)
    assert receipt["roll_forward"]["verified"] is True


def _campaign_fir(taps=BEFORE_TAPS):
    return FirSettings(taps=taps, upper_nlc=6, lower_nlc=6)


def test_campaign_bert_adapter_construction_is_inert_and_preflight_reads_exact_target():
    client = CampaignBertClient([_campaign_bert_configuration()])
    adapter = live_workflow_module.CampaignBertAdapter(
        client=client,
        port="port1",
        channel="0",
        fixed_pattern="SSPRQ",
        upper_nlc=6,
        lower_nlc=6,
        expected_host="192.168.3.2",
        expected_udp_port=9000,
    )

    assert client.mode_calls == 0
    assert client.get_calls == []
    assert client.set_calls == []

    report = adapter.preflight()

    assert report["identity_verified"] is True
    assert report["configuration_verified"] is True
    assert report["pattern"] == "SSPRQ"
    assert report["pattern_fields"] == ("SSPRQ", "SSPRQ")
    assert report["pattern_verified"] is True
    assert report["port"] == "port1"
    assert report["channel"] == "0"
    assert report["upper_nlc"] == 6
    assert report["lower_nlc"] == 6
    assert client.mode_calls == 1
    assert client.get_calls == [("port1", "0")]
    assert client.set_calls == []


def test_campaign_bert_adapter_readback_and_apply_use_canonical_taps_and_fixed_nlc():
    target = _campaign_fir(TARGET_TAPS)
    client = CampaignBertClient(
        [
            _campaign_bert_configuration(),
            _campaign_bert_configuration(),
            _campaign_bert_configuration(taps=TARGET_TAPS),
        ]
    )
    adapter = live_workflow_module.CampaignBertAdapter(
        client=client,
        port="port1",
        channel="0",
        fixed_pattern="SSPRQ",
        upper_nlc=6,
        lower_nlc=6,
        expected_host="192.168.3.2",
        expected_udp_port=9000,
    )

    assert adapter.readback() == {
        "pattern": "SSPRQ",
        "fir": _campaign_fir(),
    }
    assert adapter.apply("SSPRQ", target) == target
    assert client.set_calls == [("port1", "0", TARGET_TAPS, 6, 6)]
    assert client.get_calls == [
        ("port1", "0"),
        ("port1", "0"),
        ("port1", "0"),
    ]


def test_campaign_bert_adapter_rejects_pattern_or_nlc_changes_before_network_call():
    client = CampaignBertClient([])
    adapter = live_workflow_module.CampaignBertAdapter(
        client=client,
        port="port1",
        channel="0",
        fixed_pattern="SSPRQ",
        upper_nlc=6,
        lower_nlc=6,
        expected_host="192.168.3.2",
        expected_udp_port=9000,
    )

    with pytest.raises(LiveWorkflowError, match="pattern"):
        adapter.apply("PRBS31Q", _campaign_fir())
    with pytest.raises(LiveWorkflowError, match="NLC"):
        adapter.apply(
            "SSPRQ",
            FirSettings(taps=BEFORE_TAPS, upper_nlc=5, lower_nlc=6),
        )

    assert client.get_calls == []
    assert client.set_calls == []


@pytest.mark.parametrize(
    "pattern_fields",
    [("PRBS31Q", "SSPRQ"), ("SSPRQ", "PRBS31Q"), ("", "")],
)
def test_campaign_bert_adapter_rejects_unattested_pattern_before_write(
    pattern_fields,
):
    client = CampaignBertClient(
        [
            _campaign_bert_configuration(
                taps=TARGET_TAPS,
                pattern_fields=pattern_fields,
            )
        ]
    )
    adapter = live_workflow_module.CampaignBertAdapter(
        client=client,
        port="port1",
        channel="0",
        fixed_pattern="SSPRQ",
        upper_nlc=6,
        lower_nlc=6,
        expected_host="192.168.3.2",
        expected_udp_port=9000,
    )

    with pytest.raises(LiveWorkflowError, match="pattern|attest|SSPRQ"):
        adapter.apply("SSPRQ", _campaign_fir(TARGET_TAPS))

    assert client.get_calls == [("port1", "0")]
    assert client.set_calls == []
