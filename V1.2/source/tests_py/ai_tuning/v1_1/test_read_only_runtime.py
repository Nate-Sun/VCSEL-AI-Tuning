from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import vcsel_ai_tuning.v1_1.read_only_runtime as read_only_runtime_module

from vcsel_ai_tuning.hardware.live_config import (
    BertConfig,
    CharacterizationConfig,
    DcaConfig,
    ErMeasurementConfig,
    I2cConfig,
    LiveHardwareConfig,
    ParameterFieldConfig,
    TdecqMeasurementConfig,
)
from vcsel_ai_tuning.hardware.vd1124 import (
    Vd1124FieldValue,
    Vd1124PageSelectEvent,
    Vd1124ProbeResult,
    Vd1124ReadAudit,
    Vd1124ReadOnlyCapture,
)
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration
from vcsel_ai_tuning.v1_1.domain import RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.read_only_runtime import (
    ReadOnlyDriverBundle,
    ReadOnlyPreflightError,
    ReadOnlyV11Gateway,
    ReadOnlyV11TaskRunner,
    open_read_only_drivers,
)


_STANDARD_RESULTS = (
    "Name=Undershoot(3.0E-3),Source=1A,Current=1.443E+1,Status=Correct,"
    "Min=1.393E+1,Max=1.479E+1,Count=5.31E+2,"
    "Name=TDECQ,Source=2A,Current=9.99,Status=Invalid,"
    "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
    "Min=3.015,Max=3.035,Mean=3.024,StdDev=0.005,Count=5.31E+2,"
    "Name=TDECQ,Source=F1,Current=4.56,Status=Correct,"
    "Min=4.15,Max=4.67,Mean=4.55,StdDev=0.12,Count=4.1E+1"
)


def _profile() -> LiveHardwareConfig:
    names_and_widths = (
        ("ibias_dac", 7),
        ("efin_cur", 4),
        ("vga_cur", 4),
        ("eq", 5),
        ("res_gain", 4),
        ("res_bw", 4),
        ("gain_ctrl", 6),
        ("vbias_ref", 4),
        ("vref", 4),
    )
    fields = tuple(
        ParameterFieldConfig(name, 0x2002 + index, 0, bits)
        for index, (name, bits) in enumerate(names_and_widths)
    )
    return LiveHardwareConfig(
        dca=DcaConfig(
            resource="TCPIP0::example-dca::hislip0,4880::INSTR",
            expected_idn_prefix="Keysight Technologies,N1010A,",
            function=1,
            timeout_ms=60_000,
        ),
        bert=BertConfig(
            ip="192.0.2.10",
            local_ip="192.0.2.11",
            udp_port=9100,
            port="port2",
            channel="3",
            timeout_s=2.0,
        ),
        i2c=I2cConfig(
            port="COM42",
            baudrate=230_400,
            device_address=0xA4,
            chip_id_register=0x0004,
            expected_chip_id=0x52,
            chip_revision_register=0x0005,
            expected_chip_revision=0x00,
        ),
        characterization=CharacterizationConfig(
            default_channel=2,
            baseline_repeats=1,
            measurement_timeout_s=180.0,
            poll_interval_s=1.0,
            meaningful_er_floor_db=None,
            meaningful_tdecq_floor_db=None,
            tdecq=TdecqMeasurementConfig(
                source="FUNC1",
                input_channel="CHAN1A",
            ),
            er=ErMeasurementConfig(
                source="CHAN1A",
                value_query=":MEASure:EYE:OER?",
                source_query=":MEASure:EYE:OER:SOURce?",
                status_query=":MEASure:EYE:OER:STATus?",
                details_query=":MEASure:EYE:OER:STATus:DETails?",
                units_query=":MEASure:EYE:OER:UNITs?",
            ),
            parameters=fields,
        ),
    )


class _DcaSpy:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.responses = {
            "*IDN?": "Keysight Technologies,N1010A,MY123,A.08.32.13",
            ":MEASure:RESults? STANdard": _STANDARD_RESULTS,
            ":FUNCtion1:OPERand?": "CHAN1A",
            ":CHAN1A:WAVelength:VALue?": "8.500000E-7",
            ":TRIGger:SOURce?": "FPAN",
            ":TRIGger:DCDRatio?": "SUB4",
            ":TRIGger:SRATe?": "5.312500E10",
            ":TRIGger:PLENgth?": "65535",
        }

    def query(self, command: str) -> str:
        self.queries.append(command)
        return self.responses[command]

    def write(self, command: str) -> None:
        raise AssertionError(f"READ_ONLY DCA write attempted: {command}")

    def clear(self) -> None:
        raise AssertionError("READ_ONLY DCA Clear attempted")

    def single(self) -> None:
        raise AssertionError("READ_ONLY DCA Single attempted")


class _I2cSpy:
    port = "COM42"
    baudrate = 230_400
    address = 0xA4

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def capture_identity_and_fields(
        self,
        fields,
        *,
        channel: int,
        chip_id_register: int,
        expected_chip_id: int,
        chip_revision_register: int,
        expected_chip_revision: int,
    ):
        self.calls.append((tuple(field.name for field in fields), channel))
        values = tuple(
            Vd1124FieldValue(
                name=field.name,
                channel=channel,
                address=field.address_for_channel(channel),
                raw_value=index + 1,
                value=index + 1,
            )
            for index, field in enumerate(fields)
        )
        return Vd1124ReadOnlyCapture(
            identity=Vd1124ProbeResult(
                port=self.port,
                chip_id=expected_chip_id,
                chip_revision=expected_chip_revision,
                matched=(
                    chip_id_register == 0x0004
                    and chip_revision_register == 0x0005
                ),
            ),
            fields=values,
            audit=Vd1124ReadAudit(
                entry_page=0x20,
                page_selects=(
                    Vd1124PageSelectEvent(
                        page=0x20,
                        reason="field:eq",
                        frame=b"page",
                        acknowledgement=b"ack",
                    ),
                ),
                restore_target=0x20,
                restore_readback=0x20,
                restore_verified=True,
                tunable_register_writes=0,
            ),
        )


class _BertSpy:
    host = "192.0.2.10"
    port = 9100
    local_ip = "192.0.2.11"

    def __init__(self) -> None:
        self.mode_calls = 0
        self.configuration_calls: list[tuple[str, str]] = []
        self.mode = "PAM4 53.125 1lanePerPort"
        self.configuration = BertConfiguration(
            port="port2",
            channel="3",
            taps=(0, 5, -26, 110, -6, -6, -1),
            upper_nlc=7,
            lower_nlc=8,
            raw_response="-26 5 0 110 -6 -6 -1 off on off SSPRQ N/A SSPRQ 7 8",
            pattern_fields=("SSPRQ", "SSPRQ"),
        )

    def get_mode(self) -> str:
        self.mode_calls += 1
        return self.mode

    def get_configuration(self, port: str, channel: str):
        self.configuration_calls.append((port, channel))
        return self.configuration

    def set_fir(self, *_args, **_kwargs):
        raise AssertionError("READ_ONLY setFir/NLC write attempted")


class _Spies:
    def __init__(self) -> None:
        self.dca = _DcaSpy()
        self.i2c = _I2cSpy()
        self.bert = _BertSpy()
        self.factory_calls = 0
        self.lifecycle: list[str] = []

    @contextmanager
    def factory(self, _profile: LiveHardwareConfig, remaining_s):
        self.factory_calls += 1
        assert remaining_s() > 0
        self.lifecycle.append("open")
        try:
            yield ReadOnlyDriverBundle(
                dca=self.dca,
                i2c=self.i2c,
                bert=self.bert,
                connection_durations_s={
                    "dca": 0.01,
                    "i2c": 0.02,
                    "bert": 0.0,
                },
            )
        finally:
            self.lifecycle.append("close")


def test_gateway_construction_health_and_capabilities_do_not_open_hardware():
    spies = _Spies()
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    assert gateway.runtime_mode is RuntimeMode.READ_ONLY
    assert gateway.health() == {"status": "ok", "hardware_connected": False}
    assert gateway.capabilities() == {
        "campaign_types": ["AI_OPTIMIZATION", "SWEEP"],
        "dynamic_b0": True,
        "full_campaign_authorization": False,
        "hardware_connected": False,
        "hardware_mutation": False,
        "fake": False,
    }
    assert spies.factory_calls == 0
    assert spies.lifecycle == []

    with pytest.raises(ReadOnlyPreflightError, match="explicit Preflight"):
        gateway.current_attestation()
    assert spies.factory_calls == 0
    assert spies.lifecycle == []


@pytest.mark.parametrize("timeout", (0, -1, float("inf"), float("nan"), True))
def test_gateway_rejects_invalid_total_timeout_before_hardware(timeout):
    spies = _Spies()

    with pytest.raises(ValueError, match="total_timeout_s"):
        ReadOnlyV11Gateway(
            _profile(),
            total_timeout_s=timeout,
            driver_factory=spies.factory,
        )

    assert spies.factory_calls == 0

def test_gateway_rejects_unsafe_dca_query_and_invalid_function_before_io():
    spies = _Spies()
    profile = _profile()
    unsafe_er = replace(
        profile.characterization.er,
        source_query=":ACQuire:SINGle",
    )
    unsafe_profile = replace(
        profile,
        characterization=replace(profile.characterization, er=unsafe_er),
    )

    with pytest.raises(ReadOnlyPreflightError, match="unsafe READ_ONLY"):
        ReadOnlyV11Gateway(
            unsafe_profile, total_timeout_s=30.0, driver_factory=spies.factory
        )

    inconsistent_profile = replace(
        profile,
        dca=replace(profile.dca, function=17),
    )
    with pytest.raises(ReadOnlyPreflightError, match="function"):
        ReadOnlyV11Gateway(
            inconsistent_profile,
            total_timeout_s=30.0,
            driver_factory=spies.factory,
        )

    assert spies.factory_calls == 0
    assert spies.lifecycle == []


def test_preflight_captures_complete_dynamic_b0_without_any_mutation():
    profile = _profile()
    spies = _Spies()
    gateway = ReadOnlyV11Gateway(
        profile, total_timeout_s=30.0, driver_factory=spies.factory
    )

    capture = gateway.preflight_capture()
    b0 = capture.b0

    assert capture.snapshot_token.startswith("read-only-snapshot-")
    assert dict(b0.i2c_fields) == {
        field.name: index + 1
        for index, field in enumerate(profile.characterization.parameters)
    }
    assert dict(b0.i2c_raw_registers) == {
        field.address_for_channel(profile.characterization.default_channel): index + 1
        for index, field in enumerate(profile.characterization.parameters)
    }
    assert b0.fir_taps == (0, 5, -26, 110, -6, -6, -1)
    assert (b0.upper_nlc, b0.lower_nlc) == (7, 8)
    assert b0.bert_pattern == "SSPRQ"
    assert b0.channel_mapping_payload() == {
        "i2c_channel": 2,
        "bert_port": "port2",
        "bert_channel": "3",
        "dca_tdecq_source": "FUNC1",
        "dca_tdecq_equalizer_function": 1,
        "dca_outer_er_source": "CHAN1A",
    }

    identities = {item.kind: item for item in b0.hardware_identities}
    assert identities["dca"].resource == profile.dca.resource
    assert identities["dca"].identity == spies.dca.responses["*IDN?"]
    assert identities["i2c"].resource == "COM42/230400/0xA4"
    assert identities["i2c"].identity == "chip_id=0x52;revision=0x00"
    assert identities["bert"].resource == "192.0.2.10:9100/port2/3"
    assert identities["bert"].identity == (
        "mode=PAM4 53.125 1lanePerPort;patterns=SSPRQ,SSPRQ"
    )

    assert dict(b0.dca_fingerprint.measurement_sources) == {
        "outer_er": "CHAN1A",
        "tdecq": "FUNC1",
    }
    assert dict(b0.dca_fingerprint.configuration) == {
        "channel": "CHAN1A",
        "function": "1",
        "tdecq_equalizer_function": "FUNC1",
        "tdecq_input_channel": "CHAN1A",
        "outer_er_details": "Correct",
        "outer_er_value_db": "3.025",
        "outer_er_status": "CORR",
        "outer_er_units": "DECIBEL",
        "pattern_length": "65535",
        "symbol_rate_baud": "53125000000",
        "tdecq_details": "Correct",
        "tdecq_value_db": "4.56",
        "tdecq_status": "CORR",
        "trigger_ratio": "SUB4",
        "trigger_source": "FPAN",
        "wavelength_m": "8.5e-07",
    }
    assert tuple(binding.name for binding in capture.mapping.bindings) == tuple(
        field.name for field in profile.characterization.parameters
    )
    assert all(
        binding.status.value == "AVAILABLE"
        for binding in capture.mapping.bindings
    )
    assert spies.dca.queries == [
        "*IDN?",
        ":MEASure:RESults? STANdard",
        ":FUNCtion1:OPERand?",
        ":CHAN1A:WAVelength:VALue?",
        ":TRIGger:SOURce?",
        ":TRIGger:DCDRatio?",
        ":TRIGger:SRATe?",
        ":TRIGger:PLENgth?",
    ]
    assert spies.i2c.calls == [
        (tuple(field.name for field in profile.characterization.parameters), 2)
    ]
    assert spies.bert.mode_calls == 1
    assert spies.bert.configuration_calls == [("port2", "3")]
    assert spies.lifecycle == ["open", "close"]
    evidence = capture.evidence.payload()
    assert evidence["fake"] is False
    assert evidence["targets"]["dca_visa"] == profile.dca.resource
    assert evidence["raw_responses"]["dca"]["*IDN?"] == (
        spies.dca.responses["*IDN?"]
    )
    dca = evidence["normalized"]["dca"]
    assert dca["outer_er"]["value"] == pytest.approx(3.025)
    assert dca["outer_er"]["status"] == "CORR"
    assert dca["tdecq"] == {
        "displayed_source": "FUNC1",
        "function": "FUNC1",
        "input_channel": "CHAN1A",
        "value": 4.56,
        "status": "CORR",
        "details": "Correct",
        "unit": None,
        "statistics": {
            "min": 4.15,
            "max": 4.67,
            "mean": 4.55,
            "stddev": 0.12,
        },
    }
    assert evidence["raw_responses"]["i2c"]["restore_verified"] is True
    assert evidence["mutation_counts"] == {
        "bert_writes": 0,
        "dca_writes": 0,
        "i2c_page_select_writes": 1,
        "i2c_tunable_register_writes": 0,
    }
    assert evidence["durations_s"]["dca"]["connection_s"] == 0.01
    assert all("?" in command and ";" not in command for command in spies.dca.queries)
    assert not any("TDEQ:STATus" in command for command in spies.dca.queries)


def test_results_table_selects_tdecq_f1_independent_of_order_and_case():
    spies = _Spies()
    spies.dca.responses[":MEASure:RESults? STANdard"] = (
        "name=tdecq,source=f2,current=8.5,status=invalid,"
        "NAME=tdecq,SOURCE=f1,CURRENT=4.56,STATUS=correct,"
        "Name=Average Power,Source=1A,Current=0.5,Status=Questionable,"
        "Name=outer er,Source=1a,Current=3.025,Status=correct"
    )
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    capture = gateway.preflight_capture()

    normalized = capture.evidence.payload()["normalized"]["dca"]
    assert normalized["tdecq"]["displayed_source"] == "FUNC1"
    assert normalized["tdecq"]["value"] == pytest.approx(4.56)
    assert normalized["outer_er"]["value"] == pytest.approx(3.025)


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (
            "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
            "Name=TDECQ,Source=1A,Current=2.0,Status=Correct",
            "TDECQ.*FUNC1.*not found",
        ),
        (
            "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
            "Name=TDECQ,Source=F1,Current=4.56,Status=Correct,"
            "Name=TDECQ,Source=F1,Current=4.57,Status=Correct",
            "multiple.*TDECQ.*FUNC1",
        ),
        (
            "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
            "Name=TDECQ,Source=F1,Current=4.56,Status=Invalid",
            "TDECQ status is INV",
        ),
        (
            "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
            "Name=TDECQ,Source=F1,Current=4.56,Status=Questionable",
            "TDECQ status is QUES",
        ),
        (
            "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
            "Name=TDECQ,Source=F1,Current=9.91E+37,Status=Correct",
            "TDECQ Current.*placeholder",
        ),
        (
            "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
            "Name=TDECQ,Source=F1,Current=4.56,Status=Unexpected",
            "TDECQ status.*unknown",
        ),
    ],
)
def test_results_table_tdecq_failures_are_explicit_and_keep_raw_evidence(
    results, message
):
    spies = _Spies()
    spies.dca.responses[":MEASure:RESults? STANdard"] = results
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    with pytest.raises(ReadOnlyPreflightError, match=message) as caught:
        gateway.preflight_capture()

    assert caught.value.evidence["raw_responses"]["dca"][
        ":MEASure:RESults? STANdard"
    ] == results
    assert spies.i2c.calls == []
    assert spies.bert.mode_calls == 0


def test_results_table_invalid_tdecq_preserves_inv_reason():
    spies = _Spies()
    spies.dca.responses[":MEASure:RESults? STANdard"] = (
        "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
        "Name=TDECQ,Source=F1,Current=4.56,Status=Invalid"
    )
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    with pytest.raises(ReadOnlyPreflightError, match="TDECQ status is INV") as caught:
        gateway.preflight_capture()

    row = caught.value.evidence["failure_context"]["dca_measurement"]
    assert row["status"] == "INV"
    assert row["details"] == "Invalid"


def test_results_table_rejects_func1_operand_mismatch_without_mutation():
    spies = _Spies()
    spies.dca.responses[":FUNCtion1:OPERand?"] = "CHAN2A"
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    with pytest.raises(ReadOnlyPreflightError, match="FUNC1 operand mismatch"):
        gateway.preflight_capture()

    assert spies.dca.queries == [
        "*IDN?",
        ":MEASure:RESults? STANdard",
        ":FUNCtion1:OPERand?",
    ]
    assert spies.i2c.calls == []
    assert spies.bert.mode_calls == 0


@pytest.mark.parametrize(
    ("command", "value", "message"),
    [
        (":CHAN1A:WAVelength:VALue?", "9.91E+37", "placeholder"),
        (":TRIGger:SRATe?", "NaN", "placeholder"),
        (":TRIGger:PLENgth?", "1.5", "integer"),
    ],
)
def test_preflight_rejects_invalid_dca_fingerprint_numbers(
    command, value, message
):
    spies = _Spies()
    spies.dca.responses[command] = value
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    with pytest.raises(ReadOnlyPreflightError, match=message):
        gateway.preflight_capture()

    assert spies.i2c.calls == []
    assert spies.bert.mode_calls == 0
    assert spies.lifecycle == ["open", "close"]


def test_preflight_total_timeout_closes_bundle_and_discards_capture():
    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()
    spies = _Spies()
    original_query = spies.dca.query

    def timed_query(command: str) -> str:
        result = original_query(command)
        clock.value += 4.0
        return result

    spies.dca.query = timed_query
    gateway = ReadOnlyV11Gateway(
        _profile(),
        total_timeout_s=30.0,
        driver_factory=spies.factory,
        monotonic=clock,
    )

    with pytest.raises(ReadOnlyPreflightError, match="total timeout") as caught:
        gateway.preflight_capture()

    assert caught.value.evidence["total_duration_s"] >= 30.0
    assert caught.value.evidence["timeout_s"] == 30.0
    assert gateway.total_timeout_s == 30.0
    assert spies.i2c.calls == []
    assert spies.lifecycle == ["open", "close"]


def test_preflight_cleanup_crossing_deadline_discards_capture():
    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()
    spies = _Spies()

    @contextmanager
    def slow_cleanup_factory(profile, remaining_s):
        with spies.factory(profile, remaining_s) as drivers:
            try:
                yield drivers
            finally:
                clock.value = 31.0

    gateway = ReadOnlyV11Gateway(
        _profile(),
        total_timeout_s=30.0,
        driver_factory=slow_cleanup_factory,
        monotonic=clock,
    )

    with pytest.raises(ReadOnlyPreflightError, match="driver cleanup"):
        gateway.preflight_capture()

    assert spies.lifecycle == ["open", "close"]


def test_builtin_driver_factory_uses_remaining_total_timeout_for_each_open(
    monkeypatch
):
    import vcsel_ai_tuning.hardware.flexdca as flexdca_module
    import vcsel_ai_tuning.hardware.vd1124 as vd1124_module
    import vcsel_ai_tuning.hardware.vit_bert as bert_module

    class Clock:
        value = 0.0

        def remaining(self):
            return 10.0 - self.value

    clock = Clock()
    calls = []

    @contextmanager
    def fake_visa(resource, timeout_ms, *, open_timeout_ms):
        calls.append(("dca", resource, timeout_ms, open_timeout_ms))
        clock.value += 4.0
        yield object()

    @contextmanager
    def fake_i2c(**kwargs):
        calls.append(("i2c", kwargs["timeout_s"]))
        clock.value += 3.0
        yield object()

    def fake_bert(host, port, *, local_ip, timeout_s):
        calls.append(("bert", host, port, local_ip, timeout_s))
        return object()

    monkeypatch.setattr(flexdca_module, "open_visa_session", fake_visa)
    monkeypatch.setattr(vd1124_module, "Vd1124ReadOnlySession", fake_i2c)
    monkeypatch.setattr(bert_module, "VitBertClient", fake_bert)

    with open_read_only_drivers(_profile(), clock.remaining):
        pass

    assert calls[0][0] == "dca"
    assert calls[0][2:] == (10_000, 10_000)
    assert calls[1] == ("i2c", 1.0)
    assert calls[2][0] == "bert"
    assert calls[2][-1] == pytest.approx(2.0)


def test_isolated_driver_factory_never_opens_dca_in_parent(
    tmp_path, monkeypatch
):
    import vcsel_ai_tuning.hardware.vd1124 as vd1124_module
    import vcsel_ai_tuning.hardware.vit_bert as bert_module

    calls = []

    class FakeDcaClient:
        def __init__(self, **kwargs):
            calls.append(("dca_client", kwargs))

        def set_operation_context(self, **kwargs):
            calls.append(("dca_context", kwargs))

        def read_only_queries(self):
            calls.append(("dca_capture",))
            return SimpleNamespace(
                responses={"*IDN?": "Keysight Technologies,N1010A,MY123,1.0"},
                worker_pid=43210,
                journal_path=str(tmp_path / "dca_io_journal.jsonl"),
            )

    @contextmanager
    def fake_i2c(**kwargs):
        calls.append(("i2c", kwargs))
        yield object()

    def fake_bert(host, port, *, local_ip, timeout_s):
        calls.append(("bert", host, port, local_ip, timeout_s))
        return object()

    monkeypatch.setattr(vd1124_module, "Vd1124ReadOnlySession", fake_i2c)
    monkeypatch.setattr(bert_module, "VitBertClient", fake_bert)

    with read_only_runtime_module.open_isolated_read_only_drivers(
        _profile(),
        lambda: 10.0,
        artifact_root=tmp_path,
        cleanup_grace_s=0.2,
        dca_client_factory=FakeDcaClient,
    ) as drivers:
        assert drivers.dca.query("*IDN?").startswith(
            "Keysight Technologies,N1010A,"
        )
        assert drivers.dca.isolation_evidence["worker_pid"] == 43210

    assert [call[0] for call in calls] == [
        "dca_client",
        "dca_context",
        "dca_capture",
        "i2c",
        "bert",
    ]


def test_preflight_page_restore_failure_has_audit_and_no_b0():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124PageRestoreError

    spies = _Spies()
    audit = Vd1124ReadAudit(
        entry_page=0x20,
        page_selects=(
            Vd1124PageSelectEvent(0x00, "identity", b"frame", b"ack"),
            Vd1124PageSelectEvent(0x20, "restore", b"frame", b"ack"),
        ),
        restore_target=0x20,
        restore_readback=0x00,
        restore_verified=False,
        tunable_register_writes=0,
    )

    def fail_restore(*_args, **_kwargs):
        raise Vd1124PageRestoreError("restore mismatch", audit=audit)

    spies.i2c.capture_identity_and_fields = fail_restore
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    with pytest.raises(ReadOnlyPreflightError, match="restore mismatch") as caught:
        gateway.preflight_capture()

    evidence = caught.value.evidence
    assert evidence["failure_context"]["i2c_page_restore"][
        "restore_verified"
    ] is False
    assert evidence["mutation_counts"]["i2c_page_select_writes"] == 2
    assert evidence["mutation_counts"]["i2c_tunable_register_writes"] == 0
    assert spies.bert.mode_calls == 0
    assert spies.lifecycle == ["open", "close"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda spies: spies.dca.responses.__setitem__(
                "*IDN?", "OTHER,DCA"
            ),
            "identity",
        ),
        (
            lambda spies: spies.dca.responses.__setitem__(
                ":MEASure:RESults? STANdard",
                _STANDARD_RESULTS.replace(
                    "Name=Outer ER,Source=1A",
                    "Name=Outer ER,Source=F1",
                ),
            ),
            "Outer ER source",
        ),
        (
            lambda spies: spies.dca.responses.__setitem__(
                ":MEASure:RESults? STANdard",
                _STANDARD_RESULTS.replace(
                    "Name=Outer ER,Source=1A,Current=3.025",
                    "Name=Outer ER,Source=1A,Current=3.025,Unit=WATT",
                ),
            ),
            "Outer ER units",
        ),
        (
            lambda spies: spies.dca.responses.__setitem__(
                ":MEASure:RESults? STANdard",
                _STANDARD_RESULTS.replace(
                    "Name=Outer ER,Source=1A,Current=3.025,Status=Correct",
                    "Name=Outer ER,Source=1A,Current=3.025,Status=Invalid",
                ),
            ),
            "Outer ER status",
        ),
        (
            lambda spies: spies.dca.responses.__setitem__(
                ":MEASure:RESults? STANdard",
                _STANDARD_RESULTS.replace(
                    "Name=TDECQ,Source=F1",
                    "Name=TDECQ,Source=F2",
                ),
            ),
            "TDECQ source",
        ),
        (
            lambda spies: spies.dca.responses.__setitem__(
                ":MEASure:RESults? STANdard",
                _STANDARD_RESULTS.replace(
                    "Name=TDECQ,Source=F1,Current=4.56,Status=Correct",
                    "Name=TDECQ,Source=F1,Current=4.56,Status=Invalid",
                ),
            ),
            "TDECQ status",
        ),
        (
            lambda spies: setattr(
                spies.i2c,
                "capture_identity_and_fields",
                lambda *_args, **_kwargs: Vd1124ReadOnlyCapture(
                    identity=Vd1124ProbeResult("COM42", 0x51, 0x00, False),
                    fields=(),
                    audit=Vd1124ReadAudit(0, (), 0, 0, True, 0),
                ),
            ),
            "I2C identity",
        ),
        (
            lambda spies: setattr(
                spies.bert,
                "configuration",
                replace(spies.bert.configuration, pattern_fields=("PRBS31Q", "SSPRQ")),
            ),
            "SSPRQ",
        ),
        (
            lambda spies: setattr(
                spies.bert,
                "configuration",
                replace(
                    spies.bert.configuration,
                    pattern_fields=("SSPRQ", "PRBS31Q"),
                ),
            ),
            "SSPRQ",
        ),
        (
            lambda spies: setattr(spies.bert, "mode", "NRZ"),
            "BERT mode",
        ),
        (
            lambda spies: setattr(
                spies.bert,
                "configuration",
                replace(spies.bert.configuration, channel="4"),
            ),
            "BERT target",
        ),
    ],
)
def test_preflight_mismatch_fails_closed_without_mutation(mutation, message):
    spies = _Spies()
    mutation(spies)
    gateway = ReadOnlyV11Gateway(
        _profile(), total_timeout_s=30.0, driver_factory=spies.factory
    )

    with pytest.raises(ReadOnlyPreflightError, match=message) as caught:
        gateway.preflight_capture()

    if message == "TDECQ source":
        assert caught.value.stage == "dca"
    assert spies.lifecycle == ["open", "close"]


def test_read_only_runner_rejects_every_execution_control_without_hardware():
    from vcsel_ai_tuning.runtime import ReadOnlyModeError

    runner = ReadOnlyV11TaskRunner()
    operations = (
        lambda: runner.start("task-1", TaskType.SWEEP),
        lambda: runner.cancel_prepared("task-1"),
        lambda: runner.pause_and_restore_b0("task-1"),
        lambda: runner.resume_from_b0("task-1"),
        lambda: runner.abort_and_restore_b0("task-1"),
        lambda: runner.recover_b0("task-1"),
    )

    for operation in operations:
        with pytest.raises(ReadOnlyModeError, match="READ_ONLY"):
            operation()
