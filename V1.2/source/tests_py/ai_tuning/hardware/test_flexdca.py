from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.hardware.flexdca import (
    DcaReadError,
    DcaSourceMismatchError,
    open_visa_session,
    read_existing_tdecq,
    read_verified_oer,
    read_verified_tdecq,
)
from vcsel_ai_tuning.hardware.live_config import ErMeasurementConfig


def _responses(
    *,
    function: int = 1,
    tdecq: str = "2.75\n",
    status: str = " CORR \n",
    details: str = "No errors\n",
    taps: str = "0.1, -0.2, 1.3, 0.4\n",
    precursors: str = "2\n",
) -> dict[str, str]:
    return {
        ":MEASure:EYE:TDEQ:SOURce?": f"FUNC{function}\n",
        "*IDN?": "Keysight Technologies,N1000A,MY12345678,A.07.50\n",
        ":MEASure:EYE:TDEQ?": tdecq,
        ":MEASure:EYE:TDEQ:STATus?": status,
        ":MEASure:EYE:TDEQ:STATus:DETails?": details,
        f":SPRocess{function}:TEQualizer:TAPS?": taps,
        f":SPRocess{function}:TEQualizer:NPRecursors?": precursors,
    }


def _expected_tdecq_commands(function: int = 1) -> list[str]:
    return [
        ":MEASure:EYE:TDEQ:SOURce?",
        ":MEASure:EYE:TDEQ:STATus?",
        ":MEASure:EYE:TDEQ:STATus:DETails?",
        ":MEASure:EYE:TDEQ?",
        "*IDN?",
        f":SPRocess{function}:TEQualizer:TAPS?",
        f":SPRocess{function}:TEQualizer:NPRecursors?",
    ]


class FakeQuerySession:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.commands: list[str] = []
        self.writes: list[str] = []

    def write(self, command: str) -> None:
        self.writes.append(command)

    def query(self, command: str) -> str:
        self.commands.append(command)
        return self.responses[command]


class SharedMeasurementContextSession(FakeQuerySession):
    """Model the field behavior where TDEQ selection also changes OER context."""

    def __init__(self, responses: dict[str, str]):
        super().__init__(responses)
        self.current_source = "FUNC1"
        self.events: list[tuple[str, str]] = []

    def write(self, command: str) -> None:
        self.events.append(("write", command))
        super().write(command)
        if command.startswith(":MEASure:EYE:TDEQ:SOURce "):
            self.current_source = command.rsplit(" ", 1)[-1]
        elif command.startswith(":MEASure:EYE:OER:SOURce "):
            self.current_source = command.rsplit(" ", 1)[-1]

    def query(self, command: str) -> str:
        self.events.append(("query", command))
        if command in {
            ":MEASure:EYE:TDEQ:SOURce?",
            ":MEASure:EYE:OER:SOURce?",
        }:
            self.commands.append(command)
            return self.current_source
        return super().query(command)


def _er_config() -> ErMeasurementConfig:
    return ErMeasurementConfig(
        source="CHAN1A",
        value_query=":MEASure:EYE:OER?",
        source_query=":MEASure:EYE:OER:SOURce?",
        status_query=":MEASure:EYE:OER:STATus?",
        details_query=":MEASure:EYE:OER:STATus:DETails?",
        units_query=":MEASure:EYE:OER:UNITs?",
    )


def test_read_existing_tdecq_binds_function_source_and_builds_snapshot():
    responses = _responses(function=3)
    session = FakeQuerySession(responses)
    before = datetime.now(timezone.utc)

    snapshot = read_existing_tdecq(
        session, function=3, measurement_source="FUNCtion3"
    )

    after = datetime.now(timezone.utc)
    assert session.writes == [":MEASure:EYE:TDEQ:SOURce FUNC3"]
    assert session.commands == _expected_tdecq_commands(3)
    assert before <= snapshot.timestamp_utc <= after
    assert snapshot.timestamp_utc.tzinfo is timezone.utc
    assert snapshot.function == 3
    assert snapshot.measurement_source == "FUNC3"
    assert snapshot.verified_measurement_source == "FUNC3"
    assert snapshot.idn == "Keysight Technologies,N1000A,MY12345678,A.07.50"
    assert snapshot.tdecq_db == pytest.approx(2.75)
    assert snapshot.status == "CORR"
    assert snapshot.details == "No errors"
    assert snapshot.taps == pytest.approx((0.1, -0.2, 1.3, 0.4))
    assert snapshot.precursor_count == 2
    assert snapshot.main_index == 2
    assert snapshot.raw_responses == responses
    with pytest.raises(FrozenInstanceError):
        snapshot.function = 4


def test_read_verified_tdecq_reselects_and_verifies_source():
    responses = _responses(function=1)
    session = FakeQuerySession(responses)

    snapshot = read_verified_tdecq(
        session, function=1, measurement_source="FUNC1"
    )

    assert snapshot.verified_measurement_source == "FUNC1"
    assert snapshot.status == "CORR"
    assert session.writes == [":MEASure:EYE:TDEQ:SOURce FUNC1"]
    assert session.commands == _expected_tdecq_commands()


def test_er_and_tdecq_reselect_and_verify_their_own_sources_after_context_drift():
    responses = _responses(function=1)
    responses.update(
        {
            ":MEASure:EYE:OER:SOURce?": "FUNC1",
            ":MEASure:EYE:OER:UNITs?": "DB",
            ":MEASure:EYE:OER:STATus?": "CORR",
            ":MEASure:EYE:OER:STATus:DETails?": "No errors",
            ":MEASure:EYE:OER?": "3.762",
        }
    )
    session = SharedMeasurementContextSession(responses)

    tdecq = read_verified_tdecq(
        session, function=1, measurement_source="FUNC1"
    )
    outer_er = read_verified_oer(session, _er_config())

    assert tdecq.tdecq_db == pytest.approx(2.75)
    assert outer_er.value_db == pytest.approx(3.762)
    assert session.events == [
        ("write", ":MEASure:EYE:TDEQ:SOURce FUNC1"),
        ("query", ":MEASure:EYE:TDEQ:SOURce?"),
        ("query", ":MEASure:EYE:TDEQ:STATus?"),
        ("query", ":MEASure:EYE:TDEQ:STATus:DETails?"),
        ("query", ":MEASure:EYE:TDEQ?"),
        ("query", "*IDN?"),
        ("query", ":SPRocess1:TEQualizer:TAPS?"),
        ("query", ":SPRocess1:TEQualizer:NPRecursors?"),
        ("write", ":MEASure:EYE:OER:SOURce CHAN1A"),
        ("query", ":MEASure:EYE:OER:SOURce?"),
        ("query", ":MEASure:EYE:OER:STATus?"),
        ("query", ":MEASure:EYE:OER:STATus:DETails?"),
        ("query", ":MEASure:EYE:OER?"),
        ("query", ":MEASure:EYE:OER:UNITs?"),
    ]


def test_read_existing_tdecq_accepts_channel_source_with_explicit_function():
    responses = _responses(function=1)
    responses[":MEASure:EYE:TDEQ:SOURce?"] = "CHAN1A\n"
    session = FakeQuerySession(responses)

    snapshot = read_existing_tdecq(
        session, function=1, measurement_source="CHANnel1A"
    )

    assert snapshot.measurement_source == "CHAN1A"
    assert snapshot.function == 1
    assert session.writes == [":MEASure:EYE:TDEQ:SOURce CHAN1A"]
    assert ":SPRocess1:TEQualizer:TAPS?" in session.commands


def test_read_verified_oer_validates_source_units_status_and_raw_responses():
    responses = {
        ":MEASure:EYE:OER:SOURce?": "CHAN1A\n",
        ":MEASure:EYE:OER:UNITs?": "DECibel\n",
        ":MEASure:EYE:OER:STATus?": "CORR\n",
        ":MEASure:EYE:OER:STATus:DETails?": "No errors\n",
        ":MEASure:EYE:OER?": "3.762\n",
    }
    session = FakeQuerySession(responses)

    snapshot = read_verified_oer(session, _er_config())

    assert snapshot.source == "CHAN1A"
    assert snapshot.units == "DECIBEL"
    assert snapshot.status == "CORR"
    assert snapshot.value_db == pytest.approx(3.762)
    assert snapshot.valid is True
    assert snapshot.invalid_reason is None
    assert snapshot.raw_responses == responses
    assert session.writes == [":MEASure:EYE:OER:SOURce CHAN1A"]
    assert session.commands == [
        ":MEASure:EYE:OER:SOURce?",
        ":MEASure:EYE:OER:STATus?",
        ":MEASure:EYE:OER:STATus:DETails?",
        ":MEASure:EYE:OER?",
        ":MEASure:EYE:OER:UNITs?",
    ]


def test_read_verified_oer_preserves_invalid_status_without_substitute_value():
    responses = {
        ":MEASure:EYE:OER:SOURce?": "CHAN1A",
        ":MEASure:EYE:OER:UNITs?": "DB",
        ":MEASure:EYE:OER:STATus?": "INV",
        ":MEASure:EYE:OER:STATus:DETails?": "Acquisition incomplete",
        ":MEASure:EYE:OER?": "9.91E37",
    }

    snapshot = read_verified_oer(FakeQuerySession(responses), _er_config())

    assert snapshot.value_db is None
    assert snapshot.valid is False
    assert "INV" in snapshot.invalid_reason
    assert snapshot.raw_responses[":MEASure:EYE:OER?"] == "9.91E37"


@pytest.mark.parametrize("value", ["NaN", "Inf", "9.91E37"])
def test_read_verified_oer_marks_nonfinite_or_sentinel_value_invalid(value):
    responses = {
        ":MEASure:EYE:OER:SOURce?": "CHAN1A",
        ":MEASure:EYE:OER:UNITs?": "DEC",
        ":MEASure:EYE:OER:STATus?": "CORR",
        ":MEASure:EYE:OER:STATus:DETails?": "No errors",
        ":MEASure:EYE:OER?": value,
    }

    snapshot = read_verified_oer(FakeQuerySession(responses), _er_config())

    assert snapshot.value_db is None
    assert snapshot.valid is False
    assert snapshot.invalid_reason


@pytest.mark.parametrize(
    ("field", "response", "reason"),
    [
        (":MEASure:EYE:OER:SOURce?", "FUNC1", "source"),
        (":MEASure:EYE:OER:UNITs?", "RATIO", "units"),
    ],
)
def test_read_verified_oer_rejects_source_or_units_mismatch(field, response, reason):
    responses = {
        ":MEASure:EYE:OER:SOURce?": "CHAN1A",
        ":MEASure:EYE:OER:UNITs?": "DB",
        ":MEASure:EYE:OER:STATus?": "CORR",
        ":MEASure:EYE:OER:STATus:DETails?": "No errors",
        ":MEASure:EYE:OER?": "3.7",
    }
    responses[field] = response

    with pytest.raises(DcaReadError, match=reason):
        read_verified_oer(FakeQuerySession(responses), _er_config())


def test_read_verified_oer_source_mismatch_stops_before_result_queries():
    session = FakeQuerySession(
        {
            ":MEASure:EYE:OER:SOURce?": "FUNC1",
            ":MEASure:EYE:OER:UNITs?": "DB",
            ":MEASure:EYE:OER:STATus?": "CORR",
            ":MEASure:EYE:OER:STATus:DETails?": "No errors",
            ":MEASure:EYE:OER?": "3.7",
        }
    )

    with pytest.raises(DcaSourceMismatchError) as caught:
        read_verified_oer(session, _er_config())

    assert caught.value.measurement == "outer_er"
    assert caught.value.requested_source == "CHAN1A"
    assert caught.value.current_source == "FUNC1"
    assert session.writes == [":MEASure:EYE:OER:SOURce CHAN1A"]
    assert session.commands == [":MEASure:EYE:OER:SOURce?"]


def test_snapshot_raw_responses_are_immutable():
    snapshot = read_existing_tdecq(
        FakeQuerySession(_responses()), measurement_source="FUNC1"
    )

    with pytest.raises(TypeError):
        snapshot.raw_responses["*IDN?"] = "tampered"


def test_read_existing_tdecq_rejects_non_corr_after_all_six_queries():
    responses = _responses(status="uncorr\n", details="Questionable eye\n")
    session = FakeQuerySession(responses)

    with pytest.raises(DcaReadError, match="UNCORR.*Questionable eye"):
        read_existing_tdecq(session, measurement_source="FUNC1")

    assert session.commands == _expected_tdecq_commands()


@pytest.mark.parametrize("value", ["NaN", "Inf", "-Inf"])
def test_read_existing_tdecq_rejects_nonfinite_tdecq(value):
    with pytest.raises(DcaReadError, match="TDECQ.*finite"):
        read_existing_tdecq(
            FakeQuerySession(_responses(tdecq=value)),
            measurement_source="FUNC1",
        )


@pytest.mark.parametrize("value", ["9.91E37", "-9.91E37"])
def test_read_existing_tdecq_rejects_instrument_sentinel(value):
    with pytest.raises(DcaReadError, match="TDECQ.*sentinel") as caught:
        read_existing_tdecq(
            FakeQuerySession(_responses(tdecq=value)),
            measurement_source="FUNC1",
        )
    assert caught.value.raw_responses[":MEASure:EYE:TDEQ?"] == value


@pytest.mark.parametrize("value", ["NaN", "Inf", "-Inf"])
def test_read_existing_tdecq_rejects_nonfinite_taps(value):
    with pytest.raises(DcaReadError, match="taps.*finite"):
        read_existing_tdecq(
            FakeQuerySession(_responses(taps=f"0.1,{value},1.0")),
            measurement_source="FUNC1",
        )


@pytest.mark.parametrize("value", ["", "0.1,,1.0", "0.1,nope,1.0"])
def test_read_existing_tdecq_rejects_bad_tap_csv(value):
    with pytest.raises(DcaReadError, match="taps"):
        read_existing_tdecq(
            FakeQuerySession(_responses(taps=value)),
            measurement_source="FUNC1",
        )


@pytest.mark.parametrize(
    ("value", "reason"),
    [("2.5", "integer"), ("-1", "non-negative"), ("4", "main index")],
)
def test_read_existing_tdecq_rejects_bad_precursor_count(value, reason):
    with pytest.raises(DcaReadError, match=reason):
        read_existing_tdecq(
            FakeQuerySession(_responses(precursors=value)),
            measurement_source="FUNC1",
        )


def test_read_existing_tdecq_rejects_zero_main_tap():
    with pytest.raises(DcaReadError, match="Main tap.*non-zero"):
        read_existing_tdecq(
            FakeQuerySession(_responses(taps="0.1,-0.2,0.0,0.4")),
            measurement_source="FUNC1",
        )


@pytest.mark.parametrize("function", [0, -1, 17, 1.5, True])
def test_read_existing_tdecq_requires_supported_integer_function(function):
    session = FakeQuerySession(_responses())

    with pytest.raises(ValueError, match="function must be between 1 and 16"):
        read_existing_tdecq(
            session, function=function, measurement_source="FUNC1"
        )

    assert session.commands == []


def test_read_existing_tdecq_wraps_query_failure_with_command():
    class BrokenSession:
        def write(self, command: str) -> None:
            pass

        def query(self, command: str) -> str:
            if command == ":MEASure:EYE:TDEQ:SOURce?":
                return "FUNC1"
            raise OSError("link lost")

    with pytest.raises(DcaReadError, match=r"TDEQ:STATus\?.*link lost"):
        read_existing_tdecq(BrokenSession(), measurement_source="FUNC1")


def test_read_existing_tdecq_rejects_source_mismatch_before_measurement_queries():
    responses = _responses()
    responses[":MEASure:EYE:TDEQ:SOURce?"] = "CHAN1A\n"
    session = FakeQuerySession(responses)

    with pytest.raises(
        DcaSourceMismatchError, match="source.*CHAN1A.*FUNC1"
    ) as caught:
        read_existing_tdecq(session, measurement_source="FUNC1")

    assert caught.value.measurement == "tdecq"
    assert caught.value.requested_source == "FUNC1"
    assert caught.value.current_source == "CHAN1A"
    assert session.writes == [":MEASure:EYE:TDEQ:SOURce FUNC1"]
    assert session.commands == [":MEASure:EYE:TDEQ:SOURce?"]


@pytest.mark.parametrize("source", ["CHAN9A", "CHAN1E", "FUNC0", "FUNC17"])
def test_read_existing_tdecq_rejects_invalid_source(source):
    session = FakeQuerySession(_responses())

    with pytest.raises(ValueError, match="measurement_source"):
        read_existing_tdecq(session, function=1, measurement_source=source)

    assert session.writes == []
    assert session.commands == []


def test_open_visa_session_sets_timeout_and_closes_both_objects(monkeypatch):
    events: list[object] = []

    class Instrument:
        timeout = None

        def close(self):
            events.append("instrument.close")

    instrument = Instrument()

    class Manager:
        def open_resource(self, resource):
            events.append(("open_resource", resource))
            return instrument

        def close(self):
            events.append("manager.close")

    manager = Manager()
    fake_pyvisa = SimpleNamespace(ResourceManager=lambda: manager)
    monkeypatch.setitem(sys.modules, "pyvisa", fake_pyvisa)

    with pytest.raises(RuntimeError, match="inside"):
        with open_visa_session("TCPIP0::192.0.2.1::inst0::INSTR", 1234) as opened:
            assert opened is instrument
            assert instrument.timeout == 1234
            raise RuntimeError("inside")

    assert events == [
        ("open_resource", "TCPIP0::192.0.2.1::inst0::INSTR"),
        "instrument.close",
        "manager.close",
    ]


def test_open_visa_session_closes_manager_if_open_resource_fails(monkeypatch):
    events: list[str] = []

    class Manager:
        def open_resource(self, resource):
            raise OSError("unavailable")

        def close(self):
            events.append("manager.close")

    manager = Manager()
    fake_pyvisa = SimpleNamespace(ResourceManager=lambda: manager)
    monkeypatch.setitem(sys.modules, "pyvisa", fake_pyvisa)

    with pytest.raises(OSError, match="unavailable"):
        with open_visa_session("BAD::RESOURCE"):
            pytest.fail("context body must not run")

    assert events == ["manager.close"]


def test_open_visa_session_can_bound_resource_open(monkeypatch):
    events: list[object] = []

    class Instrument:
        timeout = None

        def close(self):
            events.append("instrument.close")

    class Manager:
        def open_resource(self, resource, *, open_timeout):
            events.append(("open_resource", resource, open_timeout))
            return Instrument()

        def close(self):
            events.append("manager.close")

    manager = Manager()
    monkeypatch.setitem(
        sys.modules,
        "pyvisa",
        SimpleNamespace(ResourceManager=lambda: manager),
    )

    with open_visa_session(
        "TCPIP0::192.0.2.1::inst0::INSTR",
        timeout_ms=1200,
        open_timeout_ms=345,
    ):
        pass

    assert events == [
        ("open_resource", "TCPIP0::192.0.2.1::inst0::INSTR", 345),
        "instrument.close",
        "manager.close",
    ]
