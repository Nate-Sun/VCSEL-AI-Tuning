from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import importlib

import pytest


class FakeSerial:
    def __init__(self, responses: list[bytes]):
        self.responses = list(responses)
        self.writes: list[bytes] = []
        self.read_sizes: list[int] = []
        self.closed = False

    def write(self, payload: bytes) -> int:
        self.writes.append(bytes(payload))
        return len(payload)

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class SafeOpenSerialSpy(FakeSerial):
    def __init__(self, responses: list[bytes]):
        super().__init__(responses)
        self.dtr = True
        self.rts = True
        self.port = None
        self.open_calls = 0
        self.open_state: list[tuple[object, bool, bool]] = []

    def open(self) -> None:
        self.open_calls += 1
        self.open_state.append((self.port, self.dtr, self.rts))


def _good_responses(chip_id: int = 0x0B, chip_revision: int = 0x0D) -> list[bytes]:
    return [
        b"\x03\x01\x00",
        bytes((6, 1, 0x91, 0x92, 0x93, chip_id)),
        b"\x03\x01\x00",
        bytes((6, 1, 0xA1, 0xA2, 0xA3, chip_revision)),
    ]


def _read_response(value: int) -> bytes:
    return bytes((6, 1, 0, 0, 0, value))


def test_read_only_session_uses_one_safe_open_and_verifies_page_restore():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124ReadOnlySession

    serial = SafeOpenSerialSpy(
        [
            _read_response(0x37),
            b"\x03\x01\x00",
            _read_response(0x52),
            _read_response(0x00),
            b"\x03\x01\x00",
            _read_response(0b1011_0110),
            b"\x03\x01\x00",
            _read_response(0x37),
        ]
    )
    calls: list[dict[str, object]] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return serial

    field = ParameterFieldConfig("eq", 0x2006, 1, 3)
    with Vd1124ReadOnlySession(
        port="COM42",
        baudrate=230_400,
        address=0xA4,
        serial_factory=factory,
    ) as reader:
        capture = reader.capture_identity_and_fields(
            (field,),
            channel=0,
            chip_id_register=0x0004,
            expected_chip_id=0x52,
            chip_revision_register=0x0005,
            expected_chip_revision=0x00,
        )

    assert calls == [
        {
            "port": None,
            "baudrate": 230_400,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 1.0,
            "write_timeout": 1.0,
            "rtscts": False,
            "dsrdtr": False,
        }
    ]
    assert serial.open_calls == 1
    assert serial.open_state == [("COM42", False, False)]
    assert serial.closed is True
    assert capture.identity.matched is True
    assert capture.fields[0].value == 3
    assert capture.audit.entry_page == 0x37
    assert capture.audit.restore_readback == 0x37
    assert capture.audit.restore_verified is True
    assert capture.audit.tunable_register_writes == 0
    assert tuple(event.page for event in capture.audit.page_selects) == (
        0x00,
        0x20,
        0x37,
    )
    assert tuple(event.reason for event in capture.audit.page_selects) == (
        "identity",
        "field:eq",
        "restore",
    )
    assert not hasattr(reader, "write_field")
    assert not hasattr(reader, "write_register")


@pytest.mark.parametrize(
    ("last_response", "message"),
    [
        (_read_response(0x20), "restore readback mismatch"),
        (b"\x06\x01", "restoration could not be verified"),
    ],
)
def test_read_only_session_fails_closed_when_page_restore_is_unverified(
    last_response, message
):
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124PageRestoreError,
        Vd1124ReadOnlySession,
    )

    serial = SafeOpenSerialSpy(
        [
            _read_response(0x37),
            b"\x03\x01\x00",
            _read_response(0x52),
            _read_response(0x00),
            b"\x03\x01\x00",
            _read_response(0x16),
            b"\x03\x01\x00",
            last_response,
        ]
    )
    field = ParameterFieldConfig("eq", 0x2006, 0, 5)

    with Vd1124ReadOnlySession(
        port="COM42",
        baudrate=230_400,
        address=0xA4,
        serial_factory=lambda **_kwargs: serial,
    ) as reader:
        with pytest.raises(Vd1124PageRestoreError, match=message) as caught:
            reader.capture_identity_and_fields(
                (field,),
                channel=0,
                chip_id_register=0x0004,
                expected_chip_id=0x52,
                chip_revision_register=0x0005,
                expected_chip_revision=0x00,
            )

    assert caught.value.audit.restore_verified is False
    assert caught.value.audit.tunable_register_writes == 0
    assert serial.closed is True


def test_read_only_session_preserves_verified_restore_audit_after_read_failure():
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124ReadOnlyCaptureError,
        Vd1124ReadOnlySession,
    )

    serial = SafeOpenSerialSpy(
        [
            _read_response(0x37),
            b"\x03\x01\x00",
            _read_response(0x52),
            b"\x06\x01",
            b"\x03\x01\x00",
            _read_response(0x37),
        ]
    )

    with Vd1124ReadOnlySession(
        port="COM42",
        baudrate=230_400,
        address=0xA4,
        serial_factory=lambda **_kwargs: serial,
    ) as reader:
        with pytest.raises(Vd1124ReadOnlyCaptureError, match="read response") as caught:
            reader.capture_identity_and_fields(
                (),
                channel=0,
                chip_id_register=0x0004,
                expected_chip_id=0x52,
                chip_revision_register=0x0005,
                expected_chip_revision=0x00,
            )

    assert caught.value.audit.restore_verified is True
    assert caught.value.audit.restore_readback == 0x37
    assert tuple(event.reason for event in caught.value.audit.page_selects) == (
        "identity",
        "restore",
    )
    assert serial.closed is True


def test_module_exposes_read_only_probe_api():
    module = importlib.import_module("vcsel_ai_tuning.hardware.vd1124")

    assert hasattr(module, "Vd1124Probe")
    assert not hasattr(module.Vd1124Probe, "write_register")


def test_probe_uses_live_defaults_sends_exact_frames_and_closes():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe

    serial = FakeSerial(_good_responses())
    factory_calls: list[dict[str, object]] = []

    def serial_factory(**kwargs):
        factory_calls.append(kwargs)
        return serial

    with Vd1124Probe(serial_factory=serial_factory) as device:
        result = device.probe()
        assert serial.closed is False

    assert factory_calls == [
        {
            "port": "COM7",
            "baudrate": 115_200,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 1.0,
            "write_timeout": 1.0,
        }
    ]
    assert serial.writes == [
        bytes((5, 2, 3, 0xA6, 0xFF, 0x00)),
        bytes((8, 2, 2, 0xA6, 0x04, 3, 2, 0xA6, 1)),
        bytes((5, 2, 3, 0xA6, 0xFF, 0x00)),
        bytes((8, 2, 2, 0xA6, 0x05, 3, 2, 0xA6, 1)),
    ]
    assert serial.read_sizes == [3, 6, 3, 6]
    assert is_dataclass(result)
    assert result.port == "COM7"
    assert result.chip_id == 0x0B
    assert result.chip_revision == 0x0D
    assert result.matched is True
    assert serial.closed is True
    with pytest.raises(FrozenInstanceError):
        result.matched = False


def test_probe_extracts_page_and_offset_from_16_bit_registers():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe

    serial = FakeSerial(_good_responses(0x5A, 0xC3))
    device = Vd1124Probe(
        port="COM9",
        address=0x44,
        chip_id_register=0x1234,
        expected_chip_id=0x5A,
        chip_revision_register=0xABCD,
        expected_chip_revision=0xC3,
        serial_factory=lambda **_kwargs: serial,
    )

    device.open()
    result = device.probe()
    device.close()

    assert serial.writes == [
        bytes((5, 2, 3, 0x44, 0xFF, 0x12)),
        bytes((8, 2, 2, 0x44, 0x34, 3, 2, 0x44, 1)),
        bytes((5, 2, 3, 0x44, 0xFF, 0xAB)),
        bytes((8, 2, 2, 0x44, 0xCD, 3, 2, 0x44, 1)),
    ]
    assert result.port == "COM9"
    assert result.chip_id == 0x5A
    assert result.chip_revision == 0xC3
    assert result.matched is True


def test_probe_preserving_page_restores_entry_page():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe

    current_page = bytes((6, 1, 0, 0, 0, 0x37))
    serial = FakeSerial(
        [current_page]
        + _good_responses()
        + [b"\x03\x01\x00"]
    )

    with Vd1124Probe(serial_factory=lambda **_kwargs: serial) as device:
        result = device.probe_preserving_page()

    assert result.matched is True
    assert serial.writes[0] == bytes(
        (8, 2, 2, 0xA6, 0xFF, 3, 2, 0xA6, 1)
    )
    assert serial.writes[-1] == bytes((5, 2, 3, 0xA6, 0xFF, 0x37))


def test_probe_reports_identity_mismatch_without_writing_registers():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe

    serial = FakeSerial(_good_responses(0xFF, 0xEE))

    with Vd1124Probe(serial_factory=lambda **_kwargs: serial) as device:
        result = device.probe()

    assert result.chip_id == 0xFF
    assert result.chip_revision == 0xEE
    assert result.matched is False
    assert not hasattr(device, "write_register")


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([b"\x03\x00\x00"], "page-select ACK rejected"),
        ([b"\x03\x01"], "page-select ACK must contain 3 bytes"),
        (
            [b"\x03\x01\x00", b"\x06\x00\x00\x00\x00\x0b"],
            "read response rejected",
        ),
        (
            [b"\x03\x01\x00", b"\x06\x01\x00\x00\x0b"],
            "read response must contain 6 bytes",
        ),
    ],
)
def test_probe_rejects_bad_bridge_responses(responses, message):
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124Probe,
        Vd1124ProtocolError,
    )

    serial = FakeSerial(responses)

    with Vd1124Probe(serial_factory=lambda **_kwargs: serial) as device:
        with pytest.raises(Vd1124ProtocolError, match=message):
            device.probe()

    assert serial.closed is True


def test_probe_requires_an_open_connection():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe

    device = Vd1124Probe(serial_factory=lambda **_kwargs: FakeSerial([]))

    with pytest.raises(RuntimeError, match="not open"):
        device.probe()


def test_context_manager_closes_serial_when_body_raises():
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124Probe

    serial = FakeSerial([])

    with pytest.raises(ValueError, match="inside"):
        with Vd1124Probe(serial_factory=lambda **_kwargs: serial):
            raise ValueError("inside")

    assert serial.closed is True


def _register_read_responses(value: int) -> list[bytes]:
    return [
        b"\x03\x01\x00",
        bytes((6, 1, 0x91, 0x92, 0x93, value)),
    ]


def test_register_reader_extracts_field_and_channel_offset_without_write_api():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124RegisterReader

    serial = FakeSerial(_register_read_responses(0b1011_0110))
    field = ParameterFieldConfig("eq", 0x2006, 1, 3)

    with Vd1124RegisterReader(
        serial_factory=lambda **_kwargs: serial
    ) as reader:
        result = reader.read_field(field, channel=2)

    assert result.name == "eq"
    assert result.channel == 2
    assert result.address == 0x2206
    assert result.raw_value == 0b1011_0110
    assert result.value == 3
    assert serial.writes == [
        bytes((5, 2, 3, 0xA6, 0xFF, 0x22)),
        bytes((8, 2, 2, 0xA6, 0x06, 3, 2, 0xA6, 1)),
    ]
    assert not hasattr(reader, "write_register")
    assert not hasattr(reader, "write_field")


def test_register_reader_preserving_page_restores_entry_page():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124RegisterReader

    current_page = bytes((6, 1, 0, 0, 0, 0x19))
    serial = FakeSerial(
        [current_page]
        + _register_read_responses(0b1011_0110)
        + [b"\x03\x01\x00"]
    )
    field = ParameterFieldConfig("eq", 0x2006, 1, 3)

    with Vd1124RegisterReader(
        serial_factory=lambda **_kwargs: serial
    ) as reader:
        result = reader.read_fields_preserving_page((field,), channel=2)

    assert result[0].value == 3
    assert serial.writes[0] == bytes(
        (8, 2, 2, 0xA6, 0xFF, 3, 2, 0xA6, 1)
    )
    assert serial.writes[-1] == bytes((5, 2, 3, 0xA6, 0xFF, 0x19))


def test_register_reader_preserving_page_restores_after_read_failure():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124ProtocolError,
        Vd1124RegisterReader,
    )

    serial = FakeSerial(
        [
            bytes((6, 1, 0, 0, 0, 0x19)),
            b"\x03\x01\x00",
            b"\x06\x01",
            b"\x03\x01\x00",
        ]
    )
    field = ParameterFieldConfig("eq", 0x2006, 1, 3)

    with Vd1124RegisterReader(
        serial_factory=lambda **_kwargs: serial
    ) as reader:
        with pytest.raises(Vd1124ProtocolError, match="read response"):
            reader.read_fields_preserving_page((field,), channel=2)

    assert serial.writes[-1] == bytes((5, 2, 3, 0xA6, 0xFF, 0x19))


def test_register_writer_read_modify_write_preserves_unrelated_bits():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124RegisterWriter

    before = 0b1010_0110
    after = 0b1010_0100
    serial = FakeSerial(
        _register_read_responses(before)
        + [b"\x03\x01\x00", b"\x03\x01\x00"]
        + _register_read_responses(after)
    )
    field = ParameterFieldConfig("eq", 0x2006, 1, 3)

    with Vd1124RegisterWriter(
        serial_factory=lambda **_kwargs: serial
    ) as writer:
        result = writer.write_field(
            field,
            channel=0,
            value=2,
            expected_current=3,
        )

    assert result.before_raw == before
    assert result.requested_value == 2
    assert result.after_raw == after
    assert result.after_value == 2
    assert result.verified is True
    assert serial.writes[2:4] == [
        bytes((5, 2, 3, 0xA6, 0xFF, 0x20)),
        bytes((5, 2, 3, 0xA6, 0x06, after)),
    ]
    assert (after & 0b1111_0001) == (before & 0b1111_0001)


def test_register_writer_rejects_stale_field_before_write():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124ProtocolError,
        Vd1124RegisterWriter,
    )

    serial = FakeSerial(_register_read_responses(0x16))
    field = ParameterFieldConfig("eq", 0x2006, 0, 5)

    with Vd1124RegisterWriter(
        serial_factory=lambda **_kwargs: serial
    ) as writer:
        with pytest.raises(Vd1124ProtocolError, match="stale"):
            writer.write_field(field, channel=0, value=22, expected_current=21)

    assert len(serial.writes) == 2


def test_register_writer_rejects_changed_unrelated_raw_bits_before_write():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124ProtocolError,
        Vd1124RegisterWriter,
    )

    serial = FakeSerial(_register_read_responses(0x96))
    field = ParameterFieldConfig("eq", 0x2006, 0, 5)

    with Vd1124RegisterWriter(
        serial_factory=lambda **_kwargs: serial
    ) as writer:
        with pytest.raises(Vd1124ProtocolError, match="raw"):
            writer.write_field(
                field,
                channel=0,
                value=22,
                expected_current=22,
                expected_raw=0x16,
            )

    assert len(serial.writes) == 2


def test_register_writer_reports_readback_mismatch():
    from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124ProtocolError,
        Vd1124RegisterWriter,
    )

    serial = FakeSerial(
        _register_read_responses(0x16)
        + [b"\x03\x01\x00", b"\x03\x01\x00"]
        + _register_read_responses(0x15)
    )
    field = ParameterFieldConfig("eq", 0x2006, 0, 5)

    with Vd1124RegisterWriter(
        serial_factory=lambda **_kwargs: serial
    ) as writer:
        with pytest.raises(Vd1124ProtocolError, match="readback"):
            writer.write_field(field, channel=0, value=22, expected_current=22)
