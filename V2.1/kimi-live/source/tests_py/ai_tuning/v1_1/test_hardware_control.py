from dataclasses import dataclass, replace

import pytest

from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration
from vcsel_ai_tuning.v1_1.hardware_control import (
    BertCandidateTransaction,
    DcaConfigurationDrift,
    DcaMeasurementController,
    DcaMeasurementTimeout,
    HardwareTransactionError,
    I2cCandidateTransaction,
    RecoveryRequired,
)
from vcsel_ai_tuning.v1_1.regmap import (
    RegMapField,
    ScriptEntry,
    scan_parameter_mapping,
)


def mapping():
    return scan_parameter_mapping(
        (
            ScriptEntry("eq", 7, 0),
            ScriptEntry("vref", 3, 1),
        ),
        (
            RegMapField("eq", 0x2006, 0x100, 0, 5, 0, 31),
            RegMapField("vref", 0x200B, 0x100, 2, 4, 0, 15),
        ),
    )


@dataclass(frozen=True)
class FieldValue:
    name: str
    channel: int
    address: int
    raw_value: int
    value: int


@dataclass(frozen=True)
class WriteValue:
    name: str
    channel: int
    address: int
    before_raw: int
    before_value: int
    requested_value: int
    after_raw: int
    after_value: int
    verified: bool


class FakeFieldWriter:
    def __init__(self, *, fail_on=None, fail_rollback=False):
        self.raw = {0x2006: 7, 0x200B: 0xC3}
        self.calls = []
        self.fail_on = fail_on
        self.fail_rollback = fail_rollback
        self.failed_once = False

    def read_field(self, field, *, channel):
        address = field.address_for_channel(channel)
        raw = self.raw[address]
        value = (raw >> field.bit_start) & field.encoded_max
        self.calls.append(("read", field.name, value, raw))
        return FieldValue(field.name, channel, address, raw, value)

    def write_field(
        self,
        field,
        *,
        channel,
        value,
        expected_current,
        expected_raw,
    ):
        before = self.read_field(field, channel=channel)
        self.calls.append(("write", field.name, value))
        if self.fail_on == field.name and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("injected candidate failure")
        if self.fail_rollback and self.failed_once:
            raise RuntimeError("injected rollback failure")
        assert before.value == expected_current
        assert before.raw_value == expected_raw
        mask = field.encoded_max << field.bit_start
        after_raw = (before.raw_value & ~mask) | (value << field.bit_start)
        self.raw[before.address] = after_raw
        after = self.read_field(field, channel=channel)
        return WriteValue(
            field.name,
            channel,
            before.address,
            before.raw_value,
            before.value,
            value,
            after.raw_value,
            after.value,
            True,
        )


def test_i2c_candidate_writes_only_changes_in_script_order_and_verifies_raw():
    writer = FakeFieldWriter()
    transaction = I2cCandidateTransaction(
        writer=writer,
        mapping=mapping(),
        channel=0,
        b0_values={"eq": 7, "vref": 0},
    )

    receipt = transaction.apply({"vref": 5, "eq": 8})

    assert tuple(item.name for item in receipt.steps) == ("eq", "vref")
    assert receipt.requested == {"eq": 8, "vref": 5}
    assert receipt.readback == {"eq": 8, "vref": 5}
    assert [call for call in writer.calls if call[0] == "write"] == [
        ("write", "eq", 8),
        ("write", "vref", 5),
    ]
    # vref occupies bits 2..5; unrelated high bits must survive RMW.
    assert writer.raw[0x200B] & 0xC0 == 0xC0


def test_i2c_failure_stops_immediately_and_restores_b0_once():
    writer = FakeFieldWriter(fail_on="vref")
    transaction = I2cCandidateTransaction(
        writer=writer,
        mapping=mapping(),
        channel=0,
        b0_values={"eq": 7, "vref": 0},
    )

    with pytest.raises(HardwareTransactionError) as caught:
        transaction.apply({"eq": 8, "vref": 5})

    assert caught.value.rollback_verified is True
    assert ((writer.raw[0x2006] >> 0) & 31) == 7
    assert ((writer.raw[0x200B] >> 2) & 15) == 0
    assert [call for call in writer.calls if call[0] == "write"] == [
        ("write", "eq", 8),
        ("write", "vref", 5),
        ("write", "eq", 7),
    ]


def test_i2c_rollback_failure_requires_recovery_and_does_not_retry():
    writer = FakeFieldWriter(fail_on="vref", fail_rollback=True)
    transaction = I2cCandidateTransaction(
        writer=writer,
        mapping=mapping(),
        channel=0,
        b0_values={"eq": 7, "vref": 0},
    )

    with pytest.raises(RecoveryRequired) as caught:
        transaction.apply({"eq": 8, "vref": 5})

    assert caught.value.rollback_verified is False
    assert [call for call in writer.calls if call[0] == "write"] == [
        ("write", "eq", 8),
        ("write", "vref", 5),
        ("write", "eq", 7),
    ]


def bert_config(taps=(0, 9, -28, 104, -8, 1, -3)):
    return BertConfiguration(
        port="port1",
        channel="0",
        taps=taps,
        upper_nlc=6,
        lower_nlc=6,
        raw_response="fake",
        pattern_fields=("SSPRQ", "SSPRQ"),
    )


class FakeBert:
    host = "192.168.3.2"
    port = 9000

    def __init__(self, *, mismatch=False, fail_rollback=False):
        self.configuration = bert_config()
        self.calls = []
        self.mismatch = mismatch
        self.fail_rollback = fail_rollback
        self.write_count = 0

    def get_configuration(self, port, channel):
        self.calls.append(("get", port, channel))
        return self.configuration

    def set_fir(self, port, channel, *, taps, upper_nlc, lower_nlc):
        self.write_count += 1
        self.calls.append(
            ("setFir", port, channel, tuple(taps), upper_nlc, lower_nlc)
        )
        if self.fail_rollback and self.write_count == 2:
            raise RuntimeError("rollback failed")
        actual = tuple(taps)
        if self.mismatch and self.write_count == 1:
            actual = actual[:-1] + (actual[-1] + 1,)
        self.configuration = replace(
            self.configuration,
            taps=actual,
            upper_nlc=upper_nlc,
            lower_nlc=lower_nlc,
        )
        return "OK"


def test_bert_candidate_uses_one_complete_setfir_and_exact_readback():
    client = FakeBert()
    transaction = BertCandidateTransaction(
        client=client,
        port="port1",
        channel="0",
        pattern="SSPRQ",
        b0=bert_config(),
    )
    candidate = (-1, 8, -31, 110, -3, -5, -1)

    receipt = transaction.apply(candidate)

    assert receipt.requested_taps == candidate
    assert receipt.readback_taps == candidate
    writes = [call for call in client.calls if call[0] == "setFir"]
    assert writes == [("setFir", "port1", "0", candidate, 6, 6)]


def test_bert_invalid_fir_is_rejected_before_any_network_write():
    client = FakeBert()
    transaction = BertCandidateTransaction(
        client=client,
        port="port1",
        channel="0",
        pattern="SSPRQ",
        b0=bert_config(),
    )

    with pytest.raises(ValueError, match="L1"):
        transaction.apply((-20, -20, -20, 120, -20, -20, -20))

    assert not [call for call in client.calls if call[0] == "setFir"]


def test_bert_readback_mismatch_restores_full_b0_once():
    client = FakeBert(mismatch=True)
    transaction = BertCandidateTransaction(
        client=client,
        port="port1",
        channel="0",
        pattern="SSPRQ",
        b0=bert_config(),
    )

    with pytest.raises(HardwareTransactionError) as caught:
        transaction.apply((-1, 8, -31, 110, -3, -5, -1))

    assert caught.value.rollback_verified is True
    assert client.configuration == bert_config()
    assert len([call for call in client.calls if call[0] == "setFir"]) == 2


def test_bert_failed_rollback_enters_recovery_required():
    client = FakeBert(mismatch=True, fail_rollback=True)
    transaction = BertCandidateTransaction(
        client=client,
        port="port1",
        channel="0",
        pattern="SSPRQ",
        b0=bert_config(),
    )

    with pytest.raises(RecoveryRequired):
        transaction.apply((-1, 8, -31, 110, -3, -5, -1))

    assert len([call for call in client.calls if call[0] == "setFir"]) == 2


class FakeDca:
    def __init__(self, fingerprint, outcomes):
        self.fingerprint_value = fingerprint
        self.outcomes = list(outcomes)
        self.measure_calls = 0

    def fingerprint(self):
        return self.fingerprint_value

    def measure_single(self):
        self.measure_calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_dca_configuration_drift_fails_before_acquisition():
    adapter = FakeDca("changed", [object()])
    controller = DcaMeasurementController(
        adapter=adapter,
        reconnect=lambda: adapter,
        expected_fingerprint="frozen",
    )

    with pytest.raises(DcaConfigurationDrift):
        controller.measure_current_candidate()

    assert adapter.measure_calls == 0


def test_dca_timeout_reconnects_revalidates_and_retries_same_candidate_once():
    first = FakeDca("frozen", [DcaMeasurementTimeout("first timeout")])
    second = FakeDca("frozen", [{"tdecq": 2.4, "er": 3.7}])
    reconnect_calls = []
    controller = DcaMeasurementController(
        adapter=first,
        reconnect=lambda: reconnect_calls.append(True) or second,
        expected_fingerprint="frozen",
    )

    result = controller.measure_current_candidate()

    assert result == {"tdecq": 2.4, "er": 3.7}
    assert first.measure_calls == 1
    assert second.measure_calls == 1
    assert reconnect_calls == [True]


def test_second_dca_timeout_is_terminal_without_another_reconnect():
    first = FakeDca("frozen", [DcaMeasurementTimeout("first timeout")])
    second = FakeDca("frozen", [DcaMeasurementTimeout("second timeout")])
    reconnect_calls = []
    controller = DcaMeasurementController(
        adapter=first,
        reconnect=lambda: reconnect_calls.append(True) or second,
        expected_fingerprint="frozen",
    )

    with pytest.raises(DcaMeasurementTimeout, match="second timeout"):
        controller.measure_current_candidate()

    assert reconnect_calls == [True]
    assert second.measure_calls == 1
