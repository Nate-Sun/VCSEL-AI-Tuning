from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import json

import pytest

from vcsel_ai_tuning.hardware.dca_acquisition import (
    AcquisitionWorkflowError,
    SINGLE_COMMANDS,
    create_measurement_plan,
    execute_single_measurement,
    load_measurement_plan,
    save_measurement_plan,
)
from vcsel_ai_tuning.hardware.vd1124 import Vd1124ProbeResult
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


CREATED_AT = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)
RESOURCE = "TCPIP0::Nate-Rog::hislip0,4880::INSTR"
BASELINE_TAPS = (0, 1, -5, 110, 3, 1, 0)
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


def _plan(**overrides):
    arguments = {
        "resource": RESOURCE,
        "function": 1,
        "measurement_source": "FUNC1",
        "mode": "single",
        "timeout_s": 180,
        "poll_interval_s": 1,
        "created_at_utc": CREATED_AT,
    }
    arguments.update(overrides)
    return create_measurement_plan(**arguments)


def test_measurement_plan_round_trip_and_hash_rejects_tampering(tmp_path):
    plan = _plan()
    path = save_measurement_plan(plan, tmp_path / "measurement_plan.json")

    assert load_measurement_plan(path) == plan
    assert len(plan["measurement_id"]) == 64
    assert plan["created_at_utc"] == "2026-07-21T08:00:00+00:00"
    assert plan["target"] == {
        "resource": RESOURCE,
        "function": 1,
        "measurement_source": "FUNC1",
    }
    assert plan["acquisition"] == {
        "mode": "single",
        "timeout_s": 180.0,
        "poll_interval_s": 1.0,
        "commands": list(SINGLE_COMMANDS),
    }

    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["acquisition"]["timeout_s"] = 5
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(AcquisitionWorkflowError, match="hash"):
        load_measurement_plan(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_s", 0),
        ("timeout_s", -1),
        ("timeout_s", float("nan")),
        ("timeout_s", float("inf")),
        ("poll_interval_s", 0),
        ("poll_interval_s", -1),
        ("poll_interval_s", float("nan")),
        ("poll_interval_s", float("inf")),
        ("poll_interval_s", 181),
    ],
)
def test_measurement_plan_rejects_invalid_timing(field, value):
    with pytest.raises(ValueError, match=field):
        _plan(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"function": 2},
        {"measurement_source": "CHAN1A"},
        {"measurement_source": "CHAN9A"},
        {"measurement_source": "FUNC17"},
        {"mode": "run"},
    ],
)
def test_measurement_plan_rejects_unsupported_target_or_mode(overrides):
    with pytest.raises(ValueError):
        _plan(**overrides)


def test_measurement_plan_canonicalizes_func1_source():
    plan = _plan(measurement_source="FUNCtion1")

    assert plan["target"]["measurement_source"] == "FUNC1"
    assert plan["acquisition"]["commands"][0].endswith(" FUNC1")


def test_load_measurement_plan_rejects_non_object(tmp_path):
    path = tmp_path / "measurement_plan.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(AcquisitionWorkflowError, match="root"):
        load_measurement_plan(path)


def _bert_configuration(taps=BASELINE_TAPS):
    return BertConfiguration(
        port="port1",
        channel="0",
        taps=tuple(taps),
        upper_nlc=6,
        lower_nlc=6,
        raw_response="raw BERT configuration",
    )


def _matched_i2c():
    return Vd1124ProbeResult("COM7", 0x52, 0x00, True)


class ScriptedBert:
    def __init__(self, configurations):
        self.host = "192.168.3.2"
        self.port = 9000
        self.configurations = deque(configurations)
        self.get_calls = []

    def get_configuration(self, port, channel):
        self.get_calls.append((port, str(channel)))
        return self.configurations.popleft()


class FakeClock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def wait(self, seconds):
        self.seconds += seconds

    def now_utc(self):
        return CREATED_AT + timedelta(seconds=self.seconds)


class ScriptedDca:
    def __init__(
        self,
        *,
        poll_statuses=("INV", "INV", "CORR"),
        clear_status="INV",
        source="FUNC1",
        tdecq="4.84",
    ):
        self.poll_statuses = deque(poll_statuses)
        self.clear_status = clear_status
        self.source = source
        self.tdecq = tdecq
        self.stage = "initial"
        self.last_status = "INV"
        self.writes = []
        self.queries = []

    def write(self, command):
        self.writes.append(command)
        if command.startswith(":MEASure:EYE:TDEQ:SOURce "):
            self.stage = "source_selected"
        elif command == ":ACQuire:STOP":
            self.stage = "stopped"
        elif command == ":ACQuire:CDISplay":
            self.stage = "cleared"
        elif command == ":ACQuire:SINGle":
            self.stage = "single"

    def query(self, command):
        self.queries.append(command)
        if command == "*IDN?":
            return "Keysight Technologies,N1010A,Nate-Rog,A.08.32.13\n"
        if command == ":MEASure:EYE:TDEQ:SOURce?":
            return self.source + "\n"
        if command == ":MEASure:EYE:TDEQ:STATus?":
            if self.stage == "cleared":
                self.last_status = self.clear_status
            elif self.stage == "single" and self.poll_statuses:
                self.last_status = self.poll_statuses.popleft()
            return self.last_status + "\n"
        if command == ":MEASure:EYE:TDEQ:STATus:DETails?":
            if self.last_status == "CORR":
                return "Left sigma: 13.8 uW rms,Right sigma: 14.0 uW rms\n"
            return "Target SER is below actual SER\n"
        if command == ":MEASure:EYE:TDEQ?":
            return self.tdecq + "\n"
        if command == ":SPRocess1:TEQualizer:TAPS?":
            return ",".join(map(str, LIVE_DCA_TAPS)) + "\n"
        if command == ":SPRocess1:TEQualizer:NPRecursors?":
            return "2\n"
        raise AssertionError(f"unexpected query {command}")


def test_single_measurement_waits_through_inv_and_accepts_fresh_corr():
    plan = _plan()
    dca = ScriptedDca()
    bert = ScriptedBert([_bert_configuration(), _bert_configuration()])
    clock = FakeClock()

    receipt = execute_single_measurement(
        plan,
        dca,
        bert,
        _matched_i2c(),
        bert_port="port1",
        bert_channel="0",
        allow_execute=True,
        confirmation_id=plan["measurement_id"],
        monotonic=clock.monotonic,
        wait=clock.wait,
        now_utc=clock.now_utc,
    )

    assert receipt["status"] == "MEASURED"
    assert receipt["final_tdecq"]["status"] == "CORR"
    assert receipt["final_tdecq"]["tdecq_db"] == pytest.approx(4.84)
    assert receipt["measurement"] == {
        "requested_source": "FUNC1",
        "verified_source": "FUNC1",
        "status": "CORR",
        "value": pytest.approx(4.84),
        "details": "Left sigma: 13.8 uW rms,Right sigma: 14.0 uW rms",
    }
    assert dca.writes == [
        *SINGLE_COMMANDS,
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
    ]
    assert receipt["poll_count"] == 3
    assert [entry["status"] for entry in receipt["status_history"]] == [
        "INV",
        "CORR",
    ]
    assert receipt["bert_before"]["taps"] == list(BASELINE_TAPS)
    assert receipt["bert_after"]["taps"] == list(BASELINE_TAPS)


def _execute(
    dca,
    *,
    plan=None,
    bert_configurations=None,
    i2c=None,
    allow_execute=True,
    confirmation_id=None,
):
    selected_plan = _plan() if plan is None else plan
    if confirmation_id is None:
        confirmation_id = selected_plan["measurement_id"]
    bert = ScriptedBert(
        bert_configurations
        or [_bert_configuration(), _bert_configuration()]
    )
    clock = FakeClock()
    receipt = execute_single_measurement(
        selected_plan,
        dca,
        bert,
        _matched_i2c() if i2c is None else i2c,
        bert_port="port1",
        bert_channel="0",
        allow_execute=allow_execute,
        confirmation_id=confirmation_id,
        monotonic=clock.monotonic,
        wait=clock.wait,
        now_utc=clock.now_utc,
    )
    return receipt, bert, clock


@pytest.mark.parametrize(
    ("allow_execute", "confirmation_id", "reason"),
    [(False, "unused", "--execute"), (True, "wrong", "measurement ID")],
)
def test_execution_gates_prevent_all_dca_writes(
    allow_execute, confirmation_id, reason
):
    dca = ScriptedDca()

    with pytest.raises(AcquisitionWorkflowError, match=reason):
        _execute(
            dca,
            allow_execute=allow_execute,
            confirmation_id=confirmation_id,
        )

    assert dca.writes == []
    assert dca.queries == []


def test_i2c_mismatch_prevents_all_dca_access():
    dca = ScriptedDca()
    mismatched = Vd1124ProbeResult("COM7", 0x00, 0x00, False)

    with pytest.raises(AcquisitionWorkflowError, match="I2C"):
        _execute(dca, i2c=mismatched)

    assert dca.writes == []
    assert dca.queries == []


def test_source_mismatch_stops_before_stop_clear_single():
    dca = ScriptedDca(source="CHAN1A")

    receipt, _, _ = _execute(dca)

    assert receipt["status"] == "FAILED"
    assert receipt["failed_stage"] == "verify_source"
    assert receipt["measurement"] == {
        "requested_source": "FUNC1",
        "verified_source": "CHAN1A",
        "status": "SOURCE_MISMATCH",
        "value": None,
        "details": "TDECQ source 'CHAN1A' does not match 'FUNC1'",
    }
    assert dca.writes == [":MEASure:EYE:TDEQ:SOURce FUNC1"]


def test_corr_at_clear_boundary_fails_before_single():
    dca = ScriptedDca(clear_status="CORR")

    receipt, _, _ = _execute(dca)

    assert receipt["status"] == "FAILED"
    assert receipt["failed_stage"] == "clear_boundary"
    assert ":ACQuire:SINGle" not in dca.writes


def test_permanent_inv_times_out_and_stops_once_more():
    plan = _plan(timeout_s=2, poll_interval_s=1)
    dca = ScriptedDca(poll_statuses=("INV",))

    receipt, _, _ = _execute(dca, plan=plan)

    assert receipt["status"] == "TIMED_OUT"
    assert receipt["poll_count"] == 2
    assert receipt["timeout_stop"]["succeeded"] is True
    assert dca.writes[-1] == ":ACQuire:STOP"
    assert dca.writes.count(":ACQuire:STOP") == 2


def test_dca_exception_after_clear_returns_failed_receipt():
    class SingleFailureDca(ScriptedDca):
        def write(self, command):
            if command == ":ACQuire:SINGle":
                raise OSError("DCA link lost")
            super().write(command)

    receipt, _, _ = _execute(SingleFailureDca())

    assert receipt["status"] == "FAILED"
    assert receipt["failed_stage"] == "single"
    assert "link lost" in receipt["error"]


def test_nonfinite_final_tdecq_returns_failed_receipt():
    receipt, _, _ = _execute(ScriptedDca(tdecq="NaN"))

    assert receipt["status"] == "FAILED"
    assert receipt["failed_stage"] == "final_snapshot"
    assert "finite" in receipt["error"]


def test_changed_bert_invalidates_measurement():
    changed = _bert_configuration(taps=(0, 1, -4, 110, 3, 1, 0))

    receipt, _, _ = _execute(
        ScriptedDca(),
        bert_configurations=[_bert_configuration(), changed],
    )

    assert receipt["status"] == "INVALIDATED"
    assert receipt["bert_after"]["taps"] == list(changed.taps)
