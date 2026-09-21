from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from vcsel_ai_tuning.v1_1.baseline import B0Snapshot, DcaFingerprint, HardwareIdentity


CAPTURED_AT = datetime(2026, 7, 27, 8, 30, tzinfo=timezone.utc)


def make_b0(**updates) -> B0Snapshot:
    values = {
        "b0_id": "b0-001",
        "version": 1,
        "captured_at": CAPTURED_AT,
        "i2c_fields": {"ibias_dac": 70, "eq": 22},
        "i2c_raw_registers": {0x10: 0x46, 0x11: 0x16},
        "fir_taps": (0, 9, -28, 104, -8, 1, -3),
        "upper_nlc": 6,
        "lower_nlc": 6,
        "bert_pattern": "SSPRQ",
        "i2c_channel": 0,
        "bert_port": "port1",
        "bert_channel": "0",
        "dca_tdecq_source": "FUNC1",
        "dca_outer_er_source": "CHAN1A",
        "dca_fingerprint": DcaFingerprint.from_mappings(
            identity="KEYSIGHT,N1000A,MY123,7.0",
            measurement_sources={"outer_er": "CHAN1A", "tdecq": "FUNC1"},
            configuration={"pattern_lock": "ON", "symbol_rate": "53.125e9"},
        ),
        "hardware_identities": {
            "i2c": HardwareIdentity("i2c", "COM7/115200/0xA6", "0x52:0x00"),
            "bert": HardwareIdentity("bert", "192.168.3.2:9000", "VIT-BERT"),
            "dca": HardwareIdentity(
                "dca", "TCPIP0::Nate-Rog::hislip0,4880::INSTR", "KEYSIGHT,N1000A"
            ),
        },
    }
    values.update(updates)
    return B0Snapshot.capture(**values)


def test_dynamic_b0_records_complete_readback_evidence_and_hash():
    b0 = make_b0()

    assert b0.i2c_fields == (("eq", 22), ("ibias_dac", 70))
    assert b0.i2c_raw_registers == ((0x10, 0x46), (0x11, 0x16))
    assert b0.fir_taps == (0, 9, -28, 104, -8, 1, -3)
    assert b0.upper_nlc == b0.lower_nlc == 6
    assert b0.bert_pattern == "SSPRQ"
    assert b0.i2c_channel == 0
    assert b0.bert_port == "port1"
    assert b0.bert_channel == "0"
    assert b0.channel_mapping_payload() == {
        "i2c_channel": 0,
        "bert_port": "port1",
        "bert_channel": "0",
        "dca_tdecq_source": "FUNC1",
        "dca_outer_er_source": "CHAN1A",
    }
    assert b0.dca_fingerprint.measurement_sources == (
        ("outer_er", "CHAN1A"),
        ("tdecq", "FUNC1"),
    )
    assert {identity.kind for identity in b0.hardware_identities} == {
        "i2c",
        "bert",
        "dca",
    }
    assert len(b0.content_hash) == 64
    int(b0.content_hash, 16)


def test_b0_hash_is_deterministic_for_equivalent_mapping_order():
    reordered = make_b0(
        i2c_fields={"eq": 22, "ibias_dac": 70},
        hardware_identities={
            "dca": HardwareIdentity(
                "dca", "TCPIP0::Nate-Rog::hislip0,4880::INSTR", "KEYSIGHT,N1000A"
            ),
            "i2c": HardwareIdentity("i2c", "COM7/115200/0xA6", "0x52:0x00"),
            "bert": HardwareIdentity("bert", "192.168.3.2:9000", "VIT-BERT"),
        },
    )

    assert reordered.content_hash == make_b0().content_hash


def test_b0_hash_changes_when_hardware_readback_changes():
    changed = make_b0(i2c_fields={"ibias_dac": 71, "eq": 22})

    assert changed.content_hash != make_b0().content_hash


def test_b0_is_immutable_and_requires_complete_shape():
    b0 = make_b0()
    with pytest.raises(FrozenInstanceError):
        b0.upper_nlc = 7
    with pytest.raises(ValueError, match="seven"):
        make_b0(fir_taps=(0, 9, -28, 104, -8, 1))
    with pytest.raises(ValueError, match="timezone"):
        make_b0(captured_at=datetime(2026, 7, 27, 8, 30))


@pytest.mark.parametrize("field", ("bert_port", "bert_channel"))
def test_b0_rejects_non_string_vendor_identifiers(field):
    with pytest.raises(TypeError, match=field):
        make_b0(**{field: 1})


def test_b0_rejects_dca_source_that_disagrees_with_fingerprint():
    with pytest.raises(ValueError, match="DCA measurement sources"):
        make_b0(dca_tdecq_source="CHAN2A")
