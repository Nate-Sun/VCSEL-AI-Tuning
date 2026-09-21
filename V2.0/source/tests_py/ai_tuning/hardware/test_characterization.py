from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import vcsel_ai_tuning.hardware.characterization as characterization_module
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
    write_run_artifacts,
)
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.vd1124 import (
    Vd1124FieldValue,
    Vd1124ProbeResult,
    Vd1124WriteResult,
)
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_CONFIG = REPO_ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
CREATED_AT = datetime(2026, 7, 22, 1, 2, 3, tzinfo=timezone.utc)


def _config():
    return load_live_hardware_config(LIVE_CONFIG)


def _observation(index, er, tdecq, duration, *, valid=True, reason=None):
    return {
        "run_id": "run-1",
        "stage": "baseline",
        "repeat_index": index,
        "started_at_utc": f"2026-07-22T01:02:0{index}+00:00",
        "finished_at_utc": f"2026-07-22T01:02:1{index}+00:00",
        "acquisition_duration_s": 1.0,
        "wait_for_valid_duration_s": 2.0,
        "query_duration_s": 0.5,
        "total_duration_s": duration,
        "requested_settle_s": None,
        "actual_settle_s": None,
        "er_db": er,
        "er_status": "CORR" if er is not None else "INV",
        "er_details": "No errors" if er is not None else "invalid",
        "tdecq_db": tdecq,
        "tdecq_status": "CORR" if tdecq is not None else "INV",
        "tdecq_details": "No errors" if tdecq is not None else "invalid",
        "observation_valid": valid,
        "invalid_reason": reason,
        "dca_measurement_source": "FUNC1",
    }


def test_baseline_plan_round_trip_and_tamper_detection(tmp_path):
    plan = create_baseline_plan(_config(), repeats=5, created_at_utc=CREATED_AT)
    path = save_characterization_plan(plan, tmp_path / "plan.json")

    assert load_characterization_plan(path) == plan
    assert len(plan["plan_id"]) == 64
    assert plan["kind"] == "baseline"
    assert plan["parameters"]["repeats"] == 5
    assert len(plan["hardware"]["i2c_parameters"]) == 9
    assert plan["hardware"]["er"]["value_query"] == ":MEASure:EYE:OER?"
    assert plan["safety"]["dca_mutations"] == [
        ":MEASure:EYE:OER:SOURce CHAN1A",
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
        ":ACQuire:STOP",
        ":ACQuire:CDISplay",
        ":ACQuire:SINGle",
    ]

    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["parameters"]["repeats"] = 99
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(CharacterizationError, match="hash"):
        load_characterization_plan(path)


def test_summary_calculates_statistics_and_linear_p95():
    observations = [
        _observation(1, 3.0, 4.0, 1.0),
        _observation(2, 4.0, 5.0, 2.0),
        _observation(3, 5.0, 6.0, 3.0),
    ]

    summary = summarize_measurements(observations)

    assert summary["valid_count"] == 3
    assert summary["valid_ratio"] == pytest.approx(1.0)
    assert summary["er_db"] == {
        "count": 3,
        "mean": 4.0,
        "median": 4.0,
        "sample_std": 1.0,
        "min": 3.0,
        "max": 5.0,
        "range": 2.0,
        "insufficient_reason": None,
    }
    assert summary["tdecq_db"]["median"] == pytest.approx(5.0)
    assert summary["total_duration_s"]["p95"] == pytest.approx(2.9)


def test_summary_uses_null_for_insufficient_samples_and_excludes_invalid():
    observations = [
        _observation(1, 3.1, 4.2, 2.0),
        _observation(
            2, None, None, 4.0, valid=False, reason="timeout waiting for CORR"
        ),
    ]

    summary = summarize_measurements(observations)

    assert summary["valid_count"] == 1
    assert summary["valid_ratio"] == pytest.approx(0.5)
    assert summary["timeout_count"] == 1
    assert summary["er_db"]["sample_std"] is None
    assert "two" in summary["er_db"]["insufficient_reason"]
    assert summary["er_db"]["mean"] == pytest.approx(3.1)


def test_summary_excludes_instrument_sentinel_even_if_row_is_marked_valid():
    observation = _observation(1, 9.91e37, 9.91e37, 1.0)

    summary = summarize_measurements([observation])

    assert summary["er_db"]["count"] == 0
    assert summary["tdecq_db"]["count"] == 0
    assert summary["er_db"]["mean"] is None
    assert summary["tdecq_db"]["mean"] is None


def test_convergence_defaults_use_larger_of_floor_and_three_sigma():
    summary = summarize_measurements(
        [
            _observation(1, 3.0, 4.0, 1.0),
            _observation(2, 3.2, 4.4, 1.2),
            _observation(3, 3.1, 4.2, 1.1),
        ]
    )

    defaults = build_convergence_defaults(
        summary,
        er_floor_db=0.5,
        tdecq_floor_db=0.1,
        repeats=5,
    )

    assert defaults["meaningful_er_change_db"] == pytest.approx(0.5)
    assert defaults["meaningful_tdecq_change_db"] == pytest.approx(0.6)
    assert defaults["coarse_er_threshold_db"] == 2.5
    assert defaults["fine_er_threshold_db"] == 3.0
    assert defaults["status"] == "SUGGESTED_NOT_FINAL"


def test_write_run_artifacts_creates_complete_finite_audit_bundle(tmp_path):
    plan = create_baseline_plan(_config(), repeats=1, created_at_utc=CREATED_AT)
    observations = [_observation(1, 3.7, 4.8, 3.5)]
    summary = summarize_measurements(observations)
    convergence = build_convergence_defaults(
        summary, er_floor_db=None, tdecq_floor_db=None, repeats=1
    )

    run_dir = write_run_artifacts(
        tmp_path,
        run_id="run-1",
        plan=plan,
        events=[{"event": "started", "timestamp_utc": CREATED_AT.isoformat()}],
        measurements=observations,
        raw_responses={"repeat_1": {"OER?": "3.7"}},
        initial_snapshot={"bert": {"taps": [0, 5, -26, 110, -6, -6, -1]}},
        final_snapshot={"bert": {"taps": [0, 5, -26, 110, -6, -6, -1]}},
        summary=summary,
        convergence_defaults=convergence,
        report_markdown="# Report / 报告\n\nVerified / 已验证\n",
        started_at_utc=CREATED_AT,
    )

    expected = {
        "plan.json",
        "events.jsonl",
        "measurements.csv",
        "raw_responses.json",
        "initial_snapshot.json",
        "final_snapshot.json",
        "summary.json",
        "convergence_defaults.json",
        "report.md",
    }
    assert {path.name for path in run_dir.iterdir()} == expected
    for name in expected - {"events.jsonl", "measurements.csv", "report.md"}:
        json.loads((run_dir / name).read_text(encoding="utf-8"))
    assert "NaN" not in (run_dir / "summary.json").read_text(encoding="utf-8")
    assert "er_db" in (run_dir / "measurements.csv").read_text(encoding="utf-8")


class StepClock:
    def __init__(self):
        self.nanoseconds = 0

    def monotonic_ns(self):
        self.nanoseconds += 100_000_000
        return self.nanoseconds

    def wait(self, seconds):
        self.nanoseconds += int(seconds * 1_000_000_000)

    def now_utc(self):
        return CREATED_AT


class ReadOnlyBert:
    def __init__(self, taps=(0, 5, -26, 110, -6, -6, -1)):
        self.taps = taps
        self.calls = []

    def get_configuration(self, port, channel):
        self.calls.append((port, str(channel)))
        return BertConfiguration(
            port=str(port),
            channel=str(channel),
            taps=tuple(self.taps),
            upper_nlc=6,
            lower_nlc=6,
            raw_response="raw BERT",
        )


class ReadOnlyFields:
    VALUES = {
        "ibias_dac": 70,
        "efin_cur": 15,
        "vga_cur": 13,
        "eq": 22,
        "res_gain": 6,
        "res_bw": 0,
        "gain_ctrl": 5,
        "vbias_ref": 15,
        "vref": 15,
    }

    def __init__(self):
        self.calls = []

    def read_fields(self, fields, *, channel):
        self.calls.append((tuple(field.name for field in fields), channel))
        return tuple(
            Vd1124FieldValue(
                field.name,
                channel,
                field.address_for_channel(channel),
                self.VALUES[field.name],
                self.VALUES[field.name],
            )
            for field in fields
        )


class BaselineDca:
    def __init__(self, *, invalid_round=None, idn=None):
        self.invalid_round = invalid_round
        self.writes = []
        self.queries = []
        self.events = []
        self.current_source = "FUNC1"
        self.stage = "initial"
        self.round_index = 0
        self.poll_index = 0
        self.idn = idn or "Keysight Technologies,N1010A,Nate-Rog,A.08.32.13"

    def write(self, command):
        self.writes.append(command)
        self.events.append(("write", command))
        if command.startswith(":MEASure:EYE:TDEQ:SOURce "):
            self.current_source = command.rsplit(" ", 1)[-1]
        elif command.startswith(":MEASure:EYE:OER:SOURce "):
            self.current_source = command.rsplit(" ", 1)[-1]
        if command == ":ACQuire:CDISplay":
            self.stage = "cleared"
        elif command == ":ACQuire:SINGle":
            self.stage = "single"
            self.round_index += 1
            self.poll_index = 0

    def _status(self):
        if self.stage == "cleared":
            return "INV"
        if self.invalid_round == self.round_index:
            return "INV"
        return "INV" if self.poll_index == 0 else "CORR"

    def query(self, command):
        self.queries.append(command)
        self.events.append(("query", command))
        if command == "*IDN?":
            return self.idn
        if command == ":MEASure:EYE:OER:SOURce?":
            return self.current_source
        if command == ":MEASure:EYE:OER:UNITs?":
            return "DB"
        if command == ":MEASure:EYE:OER:STATus?":
            return self._status()
        if command == ":MEASure:EYE:OER:STATus:DETails?":
            return "No errors" if self._status() == "CORR" else "Acquiring"
        if command == ":MEASure:EYE:OER?":
            return str(3.7 + self.round_index / 100)
        if command == ":MEASure:EYE:TDEQ:SOURce?":
            return self.current_source
        if command == ":MEASure:EYE:TDEQ:STATus?":
            status = self._status()
            if self.stage == "single":
                self.poll_index += 1
            return status
        if command == ":MEASure:EYE:TDEQ:STATus:DETails?":
            return "No errors" if self._status() == "CORR" else "Acquiring"
        if command == ":MEASure:EYE:TDEQ?":
            return str(4.8 + self.round_index / 100)
        if command == ":SPRocess1:TEQualizer:TAPS?":
            return "0.01,-0.04,1.08,0.03,0.01,-0.004,-0.03,-0.02,-0.02"
        if command == ":SPRocess1:TEQualizer:NPRecursors?":
            return "2"
        raise AssertionError(f"unexpected query {command}")


class BaselineTdecqSourceMismatch(BaselineDca):
    def query(self, command):
        if command == ":MEASure:EYE:TDEQ:SOURce?" and self.stage == "single":
            self.queries.append(command)
            self.events.append(("query", command))
            return "CHAN1A"
        return super().query(command)


def _assert_consecutive_events(events, expected):
    width = len(expected)
    assert any(
        events[index : index + width] == expected
        for index in range(len(events) - width + 1)
    ), f"missing consecutive event sequence: {expected!r}"


def test_baseline_executes_five_fresh_single_cycles_without_write_capabilities():
    plan = create_baseline_plan(_config(), repeats=5, created_at_utc=CREATED_AT)
    dca = BaselineDca()
    bert = ReadOnlyBert()
    fields = ReadOnlyFields()
    clock = StepClock()

    result = execute_baseline_plan(
        plan,
        dca,
        bert,
        fields,
        Vd1124ProbeResult("COM7", 0x52, 0x00, True),
        allow_execute=True,
        confirmation_id=plan["plan_id"],
        monotonic_ns=clock.monotonic_ns,
        wait=clock.wait,
        now_utc=clock.now_utc,
    )

    assert result["status"] == "COMPLETED"
    assert len(result["measurements"]) == 5
    assert all(row["observation_valid"] for row in result["measurements"])
    assert all(row["total_duration_s"] > 0 for row in result["measurements"])
    assert all(
        row["acquisition_duration_s"] == pytest.approx(0.3)
        for row in result["measurements"]
    )
    acquisition_writes = [
        command for command in dca.writes if command.startswith(":ACQuire:")
    ]
    assert acquisition_writes == 5 * [
        ":ACQuire:STOP",
        ":ACQuire:CDISplay",
        ":ACQuire:SINGle",
    ]
    _assert_consecutive_events(
        dca.events,
        [
            ("write", ":MEASure:EYE:OER:SOURce CHAN1A"),
            ("query", ":MEASure:EYE:OER:SOURce?"),
            ("query", ":MEASure:EYE:OER:STATus?"),
        ],
    )
    _assert_consecutive_events(
        dca.events,
        [
            ("write", ":MEASure:EYE:TDEQ:SOURce FUNC1"),
            ("query", ":MEASure:EYE:TDEQ:SOURce?"),
            ("query", ":MEASure:EYE:TDEQ:STATus?"),
        ],
    )
    assert len(fields.calls) == 6
    assert not hasattr(fields, "write_field")
    assert not hasattr(bert, "set_fir")
    assert result["initial_snapshot"]["bert"] == result["final_snapshot"]["bert"]
    assert result["raw_responses"]["initial"]["tdecq_source"] == "FUNC1"
    assert result["raw_responses"]["repeat_1"]["status_polls"] == [
        {"er_status": "INV", "tdecq_status": "INV"},
        {"er_status": "CORR", "tdecq_status": "CORR"},
    ]
    records = result["raw_responses"]["repeat_1"]["measurements"]
    assert records["outer_er"] == {
        "requested_source": "CHAN1A",
        "verified_source": "CHAN1A",
        "status": "CORR",
        "value": pytest.approx(3.71),
        "details": "No errors",
    }
    assert records["tdecq"] == {
        "requested_source": "FUNC1",
        "verified_source": "FUNC1",
        "status": "CORR",
        "value": pytest.approx(4.81),
        "details": "No errors",
    }


def test_baseline_records_timeout_as_invalid_without_reusing_values():
    config = _config()
    plan = create_baseline_plan(config, repeats=2, created_at_utc=CREATED_AT)
    plan["parameters"]["measurement_timeout_s"] = 2.0
    plan.pop("plan_id")
    # Recreate through the public hash-preserving helper by saving a fresh short plan.
    from vcsel_ai_tuning.hardware.characterization import rehash_plan

    plan = rehash_plan(plan)
    result = execute_baseline_plan(
        plan,
        BaselineDca(invalid_round=1),
        ReadOnlyBert(),
        ReadOnlyFields(),
        Vd1124ProbeResult("COM7", 0x52, 0x00, True),
        allow_execute=True,
        confirmation_id=plan["plan_id"],
        monotonic_ns=StepClock().monotonic_ns,
        wait=lambda _seconds: None,
        now_utc=lambda: CREATED_AT,
    )

    first, second = result["measurements"]
    assert first["observation_valid"] is False
    assert first["er_db"] is None
    assert first["tdecq_db"] is None
    assert "timeout" in first["invalid_reason"].lower()
    assert second["observation_valid"] is True
    assert result["summary"]["valid_count"] == 1


def test_baseline_records_tdecq_source_mismatch_and_rejects_value():
    plan = create_baseline_plan(_config(), repeats=1, created_at_utc=CREATED_AT)
    result = execute_baseline_plan(
        plan,
        BaselineTdecqSourceMismatch(),
        ReadOnlyBert(),
        ReadOnlyFields(),
        Vd1124ProbeResult("COM7", 0x52, 0x00, True),
        allow_execute=True,
        confirmation_id=plan["plan_id"],
        monotonic_ns=StepClock().monotonic_ns,
        wait=lambda _seconds: None,
        now_utc=lambda: CREATED_AT,
    )

    row = result["measurements"][0]
    record = result["raw_responses"]["repeat_1"]["measurements"]["tdecq"]
    assert row["observation_valid"] is False
    assert row["tdecq_db"] is None
    assert record == {
        "requested_source": "FUNC1",
        "verified_source": "CHAN1A",
        "status": "SOURCE_MISMATCH",
        "value": None,
        "details": "TDECQ source readback CHAN1A does not match configured source FUNC1",
    }


def test_baseline_rejects_wrong_dca_identity_before_any_dca_mutation():
    plan = create_baseline_plan(_config(), repeats=1, created_at_utc=CREATED_AT)
    dca = BaselineDca(idn="Wrong Vendor,Wrong Instrument,0,0")

    with pytest.raises(CharacterizationError, match="DCA identity"):
        execute_baseline_plan(
            plan,
            dca,
            ReadOnlyBert(),
            ReadOnlyFields(),
            Vd1124ProbeResult("COM7", 0x52, 0x00, True),
            allow_execute=True,
            confirmation_id=plan["plan_id"],
        )

    assert dca.writes == []


class FakeFieldWriter(ReadOnlyFields):
    def __init__(self, *, fail_restore=False):
        super().__init__()
        self.values = dict(self.VALUES)
        self.writes = []
        self.fail_restore = fail_restore

    def read_fields(self, fields, *, channel):
        self.calls.append((tuple(field.name for field in fields), channel))
        return tuple(
            Vd1124FieldValue(
                field.name,
                channel,
                field.address_for_channel(channel),
                self.values[field.name],
                self.values[field.name],
            )
            for field in fields
        )

    def write_field(
        self, field, *, channel, value, expected_current, expected_raw=None
    ):
        before = self.values[field.name]
        self.writes.append((field.name, channel, value, expected_current))
        if before != expected_current:
            raise RuntimeError(
                f"stale {field.name}: expected {expected_current}, read {before}"
            )
        if expected_raw is not None and before != expected_raw:
            raise RuntimeError(
                f"stale {field.name} raw: expected {expected_raw}, read {before}"
            )
        if self.fail_restore and len(self.writes) > 1:
            raise RuntimeError("restore readback mismatch")
        self.values[field.name] = value
        return Vd1124WriteResult(
            field.name,
            channel,
            field.address_for_channel(channel),
            before,
            before,
            value,
            value,
            value,
            True,
        )


def _config_with_eq_limits():
    config = _config()
    fields = tuple(
        replace(field, safety_min=18, safety_max=24, max_step=2)
        if field.name == "eq"
        else field
        for field in config.characterization.parameters
    )
    return replace(
        config,
        characterization=replace(config.characterization, parameters=fields),
    )


def test_same_value_plan_lists_exact_readback_and_execute_writes_only_same_values():
    writer = FakeFieldWriter()
    config = _config()
    plan = create_same_value_plan(
        config,
        writer,
        Vd1124ProbeResult("COM7", 0x52, 0x00, True),
        channel=0,
        created_at_utc=CREATED_AT,
    )

    assert len(plan["operations"]) == 9
    assert all(op["old_value"] == op["new_value"] for op in plan["operations"])
    assert plan["operations"][3]["address"] == 0x2006
    assert plan["safety"].get("dca_mutations", []) == []

    result = execute_same_value_plan(
        plan,
        writer,
        Vd1124ProbeResult("COM7", 0x52, 0x00, True),
        allow_apply=True,
        confirmation_id=plan["plan_id"],
    )

    assert result["status"] == "VERIFIED"
    assert len(writer.writes) == 9
    assert all(value == expected for _, _, value, expected in writer.writes)


def test_same_value_rejects_inexact_confirmation_and_stale_state_before_write():
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_same_value_plan(
        _config(), writer, identity, channel=0, created_at_utc=CREATED_AT
    )
    with pytest.raises(CharacterizationError, match="confirmation"):
        execute_same_value_plan(
            plan,
            writer,
            identity,
            allow_apply=True,
            confirmation_id=plan["plan_id"][:-1],
        )
    writer.values["eq"] = 21
    result = execute_same_value_plan(
        plan,
        writer,
        identity,
        allow_apply=True,
        confirmation_id=plan["plan_id"],
    )
    assert result["status"] == "STALE_REJECTED"
    assert writer.writes == []


def test_settle_plan_requires_limits_and_exactly_one_explicit_target():
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    with pytest.raises(CharacterizationError, match="safety limits"):
        create_settle_plan(
            _config(),
            ReadOnlyFields(),
            identity,
            parameter="eq",
            target_value=23,
            settle_delays=(0.1, 0.5),
            channel=0,
            created_at_utc=CREATED_AT,
        )

    plan = create_settle_plan(
        _config_with_eq_limits(),
        ReadOnlyFields(),
        identity,
        parameter="eq",
        target_value=23,
        settle_delays=(0.1, 0.5),
        channel=0,
        created_at_utc=CREATED_AT,
    )
    assert len(plan["operations"]) == 1
    assert plan["operations"][0]["name"] == "eq"
    assert plan["operations"][0]["old_value"] == 22
    assert plan["operations"][0]["new_value"] == 23
    assert plan["parameters"]["repeats"] == 2
    assert plan["safety"]["dca_mutations"] == [
        ":MEASure:EYE:OER:SOURce CHAN1A",
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
        ":ACQuire:STOP",
        ":ACQuire:CDISplay",
        ":ACQuire:SINGle",
    ]
    assert plan["measurement_plan"]["kind"] == "baseline"
    assert plan["measurement_plan"]["parameters"]["repeats"] == 1


@pytest.mark.parametrize("raised", [RuntimeError("measurement failed"), KeyboardInterrupt()])
def test_settle_restores_original_value_after_exception_or_ctrl_c(raised):
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_settle_plan(
        _config_with_eq_limits(),
        writer,
        identity,
        parameter="eq",
        target_value=23,
        settle_delays=(0.1,),
        channel=0,
        created_at_utc=CREATED_AT,
    )

    def fail_measurement(_delay):
        raise raised

    result = execute_settle_plan(
        plan,
        writer,
        identity,
        fail_measurement,
        allow_apply=True,
        confirmation_id=plan["plan_id"],
    )

    assert result["status"] == "FAILED_RESTORED"
    assert result["restoration_succeeded"] is True
    assert writer.values["eq"] == 22
    assert writer.writes == [("eq", 0, 23, 22), ("eq", 0, 22, 23)]


def test_settle_restore_failure_is_recovery_required_with_manual_value():
    writer = FakeFieldWriter(fail_restore=True)
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_settle_plan(
        _config_with_eq_limits(),
        writer,
        identity,
        parameter="eq",
        target_value=23,
        settle_delays=(0.1,),
        channel=0,
        created_at_utc=CREATED_AT,
    )

    result = execute_settle_plan(
        plan,
        writer,
        identity,
        lambda _delay: (_ for _ in ()).throw(RuntimeError("measurement failed")),
        allow_apply=True,
        confirmation_id=plan["plan_id"],
    )

    assert result["status"] == "RECOVERY_REQUIRED"
    assert result["restoration_succeeded"] is False
    assert result["manual_recovery"] == {
        "parameter": "eq",
        "channel": 0,
        "address": 0x2006,
        "bit_start": 0,
        "bits": 5,
        "raw_value": 22,
        "value": 22,
    }
    assert len(writer.writes) == 2


def test_settle_rejects_stale_non_target_field_before_any_write():
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_settle_plan(
        _config_with_eq_limits(),
        writer,
        identity,
        parameter="eq",
        target_value=23,
        settle_delays=(0.1,),
        channel=0,
        created_at_utc=CREATED_AT,
    )
    writer.values["vga_cur"] = 12

    with pytest.raises(CharacterizationError, match="stale.*vga_cur"):
        execute_settle_plan(
            plan,
            writer,
            identity,
            lambda _delay: {},
            allow_apply=True,
            confirmation_id=plan["plan_id"],
        )

    assert writer.writes == []


def test_settle_revalidates_safety_even_for_rehashed_plan():
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_settle_plan(
        _config_with_eq_limits(),
        writer,
        identity,
        parameter="eq",
        target_value=23,
        settle_delays=(0.1,),
        channel=0,
        created_at_utc=CREATED_AT,
    )
    plan["operations"][0]["new_value"] = 30
    from vcsel_ai_tuning.hardware.characterization import rehash_plan

    plan = rehash_plan(plan)
    with pytest.raises(CharacterizationError, match="safety limits"):
        execute_settle_plan(
            plan,
            writer,
            identity,
            lambda _delay: {},
            allow_apply=True,
            confirmation_id=plan["plan_id"],
        )
    assert writer.writes == []


def test_same_value_rejects_duplicate_operation_set():
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_same_value_plan(
        _config(), writer, identity, channel=0, created_at_utc=CREATED_AT
    )
    plan["operations"] = [dict(plan["operations"][0]) for _ in range(9)]
    from vcsel_ai_tuning.hardware.characterization import rehash_plan

    plan = rehash_plan(plan)
    with pytest.raises(CharacterizationError, match="exactly once"):
        execute_same_value_plan(
            plan,
            writer,
            identity,
            allow_apply=True,
            confirmation_id=plan["plan_id"],
        )
    assert writer.writes == []


def test_same_value_rejects_invalid_later_operation_before_any_write():
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_same_value_plan(
        _config(), writer, identity, channel=0, created_at_utc=CREATED_AT
    )
    plan["operations"][-1]["new_value"] = 14
    from vcsel_ai_tuning.hardware.characterization import rehash_plan

    plan = rehash_plan(plan)
    with pytest.raises(CharacterizationError, match="inconsistent|changes"):
        execute_same_value_plan(
            plan,
            writer,
            identity,
            allow_apply=True,
            confirmation_id=plan["plan_id"],
        )
    assert writer.writes == []


def test_settle_rejects_invalid_delay_before_target_write():
    writer = FakeFieldWriter()
    identity = Vd1124ProbeResult("COM7", 0x52, 0x00, True)
    plan = create_settle_plan(
        _config_with_eq_limits(),
        writer,
        identity,
        parameter="eq",
        target_value=23,
        settle_delays=(0.1,),
        channel=0,
        created_at_utc=CREATED_AT,
    )
    plan["parameters"]["settle_delays_s"] = [-1.0]
    from vcsel_ai_tuning.hardware.characterization import rehash_plan

    plan = rehash_plan(plan)
    with pytest.raises(CharacterizationError, match="non-negative"):
        execute_settle_plan(
            plan,
            writer,
            identity,
            lambda _delay: {},
            allow_apply=True,
            confirmation_id=plan["plan_id"],
        )
    assert writer.writes == []


class CampaignDcaSession:
    def __init__(self, *, tdecq_status="INV"):
        self.tdecq_status = tdecq_status
        self.writes = []
        self.queries = []
        self.events = []
        self.current_source = "FUNC1"
        self.source_binding_active = False
        self.acquisition_started = False

    def write(self, command):
        self.writes.append(command)
        self.events.append(("write", command))
        if command.startswith(":MEASure:EYE:TDEQ:SOURce "):
            self.current_source = command.rsplit(" ", 1)[-1]
            self.source_binding_active = True
        elif command.startswith(":MEASure:EYE:OER:SOURce "):
            self.current_source = command.rsplit(" ", 1)[-1]
            self.source_binding_active = True
        if command == ":ACQuire:SINGle":
            self.acquisition_started = True

    def query(self, command):
        self.queries.append(command)
        self.events.append(("query", command))
        if command == "*IDN?":
            return "Keysight Technologies,N1010A,Nate-Rog,A.08.32.13"
        if command == ":MEASure:EYE:TDEQ:SOURce?":
            return self.current_source if self.source_binding_active else "FUNC1"
        if command == ":MEASure:EYE:OER:SOURce?":
            return self.current_source if self.source_binding_active else "CHAN1A"
        if command == ":MEASure:EYE:OER:UNITs?":
            return "DB"
        if command == ":MEASure:EYE:OER:STATus?":
            return "CORR" if self.acquisition_started else "INV"
        if command == ":MEASure:EYE:OER:STATus:DETails?":
            return "No errors" if self.acquisition_started else "Cleared"
        if command == ":MEASure:EYE:OER?":
            return "3.14"
        if command == ":MEASure:EYE:TDEQ:STATus?":
            return self.tdecq_status if self.acquisition_started else "INV"
        if command == ":MEASure:EYE:TDEQ:STATus:DETails?":
            return "Target SER lower than actual" if self.tdecq_status == "INV" else "No errors"
        if command == ":MEASure:EYE:TDEQ?":
            return "9.91E+37" if self.tdecq_status == "INV" else "4.21"
        if command == ":SPRocess1:TEQualizer:TAPS?":
            return "0.01,-0.04,1.08,0.03,0.01,-0.004,-0.03,-0.02,-0.02"
        if command == ":SPRocess1:TEQualizer:NPRecursors?":
            return "2"
        raise AssertionError(f"unexpected query {command}")


class SecondsClock:
    def __init__(self):
        self.value = 10.0
        self.wait_calls = []

    def monotonic(self):
        self.value += 0.125
        return self.value

    def wait(self, seconds):
        self.wait_calls.append(seconds)
        self.value += seconds


class DelayedTdecqSession(CampaignDcaSession):
    def __init__(self, invalid_details):
        super().__init__(tdecq_status="CORR")
        self.invalid_details = invalid_details
        self.post_single_tdecq_status_queries = 0

    def query(self, command):
        if command == ":MEASure:EYE:TDEQ:STATus?" and self.acquisition_started:
            self.queries.append(command)
            self.post_single_tdecq_status_queries += 1
            if self.post_single_tdecq_status_queries == 1:
                return "INV"
            return "CORR"
        if (
            command == ":MEASure:EYE:TDEQ:STATus:DETails?"
            and self.acquisition_started
            and self.post_single_tdecq_status_queries == 1
        ):
            self.queries.append(command)
            return self.invalid_details
        return super().query(command)


class NeverReadyTdecqSession(CampaignDcaSession):
    def query(self, command):
        if (
            command == ":MEASure:EYE:TDEQ:STATus:DETails?"
            and self.acquisition_started
        ):
            self.queries.append(command)
            return "Acquiring"
        return super().query(command)


class InvalidErReadSession(CampaignDcaSession):
    def __init__(self):
        super().__init__(tdecq_status="CORR")

    def query(self, command):
        if command == ":MEASure:EYE:OER:SOURce?" and self.acquisition_started:
            self.queries.append(command)
            return "CHAN2A"
        return super().query(command)


class InvalidVerifiedTdecqSession(CampaignDcaSession):
    def __init__(self):
        super().__init__(tdecq_status="CORR")

    def query(self, command):
        if command == ":MEASure:EYE:TDEQ?" and self.acquisition_started:
            self.queries.append(command)
            return "9.91E+37"
        return super().query(command)


class LateTdecqCorrSession(CampaignDcaSession):
    def __init__(self):
        super().__init__(tdecq_status="CORR")
        self.post_single_tdecq_status_queries = 0

    def query(self, command):
        if command == ":MEASure:EYE:TDEQ:STATus?" and self.acquisition_started:
            self.queries.append(command)
            self.post_single_tdecq_status_queries += 1
            if self.post_single_tdecq_status_queries <= 2:
                return "INV"
            return "CORR"
        if (
            command == ":MEASure:EYE:TDEQ:STATus:DETails?"
            and self.acquisition_started
            and self.post_single_tdecq_status_queries <= 2
        ):
            self.queries.append(command)
            return "Acquiring"
        return super().query(command)


class CampaignIdentityProbe:
    def __init__(self):
        self.calls = 0

    def probe(self):
        self.calls += 1
        return Vd1124ProbeResult("COM7", 0x52, 0x00, True)


def test_campaign_dca_adapter_construction_is_inert_and_preflight_is_read_only():
    session = CampaignDcaSession()
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=_config(),
        monotonic=SecondsClock().monotonic,
        wait=lambda _seconds: None,
    )

    assert session.writes == []
    assert session.queries == []

    report = adapter.preflight()

    assert report["identity_verified"] is True
    assert report["configuration_verified"] is True
    assert report["idn"].startswith("Keysight Technologies,N1010A,")
    assert report["tdecq_source"] == "FUNC1"
    assert report["er_source"] == "CHAN1A"
    assert report["er_units"] == "DB"
    assert session.writes == []


@pytest.mark.parametrize(
    "invalid_details",
    ["", "Acquiring", "Calculating", "Previous result stale"],
)
def test_campaign_dca_single_waits_for_delayed_corr_when_inv_is_not_terminal(
    invalid_details,
):
    session = DelayedTdecqSession(invalid_details)
    clock = SecondsClock()
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=_config(),
        monotonic=clock.monotonic,
        wait=clock.wait,
    )

    result = adapter.measure_single()

    assert result.acquisition_valid is True
    assert result.er_db == pytest.approx(3.14)
    assert result.er_status == "CORR"
    assert result.tdecq_db == pytest.approx(4.21)
    assert result.tdecq_status == "CORR"
    assert result.tdecq_invalid_reason is None
    assert result.raw_responses["status_polls"] == [
        {"er_status": "CORR", "tdecq_status": "INV"},
        {"er_status": "CORR", "tdecq_status": "CORR"},
    ]
    assert clock.wait_calls == [pytest.approx(1.0)]
    assert (
        session.queries.count(":MEASure:EYE:TDEQ:STATus:DETails?")
        == 2
    )


def test_campaign_dca_single_stops_promptly_on_terminal_tdecq_inv_details():
    session = CampaignDcaSession(tdecq_status="INV")
    clock = SecondsClock()
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=_config(),
        monotonic=clock.monotonic,
        wait=clock.wait,
    )

    result = adapter.measure_single()

    assert result.acquisition_valid is True
    assert result.er_status == "CORR"
    assert result.er_db == pytest.approx(3.14)
    assert result.tdecq_status == "INV"
    assert result.tdecq_db is None
    assert "Target SER lower than actual" in result.tdecq_invalid_reason
    assert result.elapsed_s < 1.0
    assert clock.wait_calls == []


def test_campaign_dca_single_preserves_valid_er_when_tdecq_times_out_inv():
    session = NeverReadyTdecqSession(tdecq_status="INV")
    clock = SecondsClock()
    config = _config()
    config = replace(
        config,
        characterization=replace(
            config.characterization,
            measurement_timeout_s=0.5,
            poll_interval_s=0.1,
        ),
    )
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=config,
        monotonic=clock.monotonic,
        wait=clock.wait,
    )

    result = adapter.measure_single()

    acquisition_writes = [
        command for command in session.writes if command.startswith(":ACQuire:")
    ]
    assert acquisition_writes == [
        ":ACQuire:STOP",
        ":ACQuire:CDISplay",
        ":ACQuire:SINGle",
    ]
    assert result.acquisition_valid is True
    assert result.er_db == pytest.approx(3.14)
    assert result.er_status == "CORR"
    assert result.tdecq_db is None
    assert result.tdecq_status == "INV"
    assert "Acquiring" in result.tdecq_invalid_reason
    assert result.elapsed_s >= 0.5
    assert len(result.raw_responses["status_polls"]) >= 2
    assert all(
        poll == {"er_status": "CORR", "tdecq_status": "INV"}
        for poll in result.raw_responses["status_polls"]
    )
    assert result.raw_responses["tdecq"][":MEASure:EYE:TDEQ?"] == "9.91E+37"


def test_campaign_dca_single_normalizes_unverified_corr_seen_after_deadline():
    session = LateTdecqCorrSession()
    clock = SecondsClock()
    config = _config()
    config = replace(
        config,
        characterization=replace(
            config.characterization,
            measurement_timeout_s=0.5,
            poll_interval_s=0.1,
        ),
    )
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=config,
        monotonic=clock.monotonic,
        wait=clock.wait,
    )

    result = adapter.measure_single()

    assert result.acquisition_valid is True
    assert result.er_status == "CORR"
    assert result.er_db == pytest.approx(3.14)
    assert result.tdecq_status == "INV"
    assert result.tdecq_db is None
    assert result.tdecq_invalid_reason
    assert (
        result.raw_responses["tdecq"]
        [":MEASure:EYE:TDEQ:STATus?"]
        == "CORR"
    )


def test_campaign_dca_single_returns_verified_corr_tdecq_value_and_raw_data():
    session = CampaignDcaSession(tdecq_status="CORR")
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=_config(),
        monotonic=SecondsClock().monotonic,
        wait=lambda _seconds: None,
    )

    result = adapter.measure_single()

    assert result.acquisition_valid is True
    assert result.tdecq_db == pytest.approx(4.21)
    assert result.tdecq_status == "CORR"
    assert result.tdecq_invalid_reason is None
    assert result.raw_responses["tdecq"][":MEASure:EYE:TDEQ:STATus?"] == "CORR"
    assert result.raw_responses["measurements"]["tdecq"] == {
        "requested_source": "FUNC1",
        "verified_source": "FUNC1",
        "status": "CORR",
        "value": pytest.approx(4.21),
        "details": "No errors",
    }
    _assert_consecutive_events(
        session.events,
        [
            ("write", ":MEASure:EYE:OER:SOURce CHAN1A"),
            ("query", ":MEASure:EYE:OER:SOURce?"),
            ("query", ":MEASure:EYE:OER:STATus?"),
        ],
    )
    _assert_consecutive_events(
        session.events,
        [
            ("write", ":MEASure:EYE:TDEQ:SOURce FUNC1"),
            ("query", ":MEASure:EYE:TDEQ:SOURce?"),
            ("query", ":MEASure:EYE:TDEQ:STATus?"),
        ],
    )


def test_campaign_dca_single_normalizes_verified_er_read_error_to_invalid_result():
    session = InvalidErReadSession()
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=_config(),
        monotonic=SecondsClock().monotonic,
        wait=lambda _seconds: None,
    )

    result = adapter.measure_single()

    assert result.acquisition_valid is False
    assert result.er_db is None
    assert result.er_status == "SOURCE_MISMATCH"
    assert result.raw_responses["source_validation_error"]["message"]
    assert result.raw_responses["measurements"]["outer_er"] == {
        "requested_source": "CHAN1A",
        "verified_source": "CHAN2A",
        "status": "SOURCE_MISMATCH",
        "value": None,
        "details": (
            "OER source readback CHAN2A does not match configured source CHAN1A"
        ),
    }
    assert ":MEASure:EYE:OER?" not in session.queries


def test_campaign_dca_single_never_returns_corr_tdecq_with_no_numeric_value():
    session = InvalidVerifiedTdecqSession()
    adapter = characterization_module.CampaignDcaAdapter(
        session=session,
        config=_config(),
        monotonic=SecondsClock().monotonic,
        wait=lambda _seconds: None,
    )

    result = adapter.measure_single()

    assert result.acquisition_valid is True
    assert result.er_status == "CORR"
    assert result.er_db == pytest.approx(3.14)
    assert result.tdecq_status == "INV"
    assert result.tdecq_db is None
    assert result.tdecq_invalid_reason
    assert "sentinel" in result.tdecq_invalid_reason


def test_campaign_i2c_adapter_is_inert_then_uses_exact_stale_and_raw_guards():
    writer = FakeFieldWriter()
    probe = CampaignIdentityProbe()
    adapter = characterization_module.CampaignI2cAdapter(
        writer=writer,
        identity_probe=probe,
        config=_config(),
    )

    assert writer.calls == []
    assert writer.writes == []
    assert probe.calls == 0

    report = adapter.preflight()
    assert report["identity_verified"] is True
    assert report["configuration_verified"] is True
    assert report["readback"]["eq"] == 22
    assert writer.writes == []

    applied = adapter.apply({"eq": 23})

    assert applied == {"eq": 23}
    assert writer.writes[-1] == ("eq", 0, 23, 22)
    assert adapter.readback()["eq"] == 23


def test_campaign_i2c_adapter_rejects_unknown_or_unencodable_fields_before_write():
    writer = FakeFieldWriter()
    adapter = characterization_module.CampaignI2cAdapter(
        writer=writer,
        identity_probe=CampaignIdentityProbe(),
        config=_config(),
    )

    with pytest.raises(CharacterizationError, match="unknown"):
        adapter.apply({"not_a_field": 1})
    with pytest.raises(CharacterizationError, match="encoding"):
        adapter.apply({"eq": 32})

    assert writer.writes == []
