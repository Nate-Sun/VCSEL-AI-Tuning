from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
from typing import Protocol

from vcsel_ai_tuning.hardware.live_config import ParameterFieldConfig


class SerialConnection(Protocol):
    """Subset of a pyserial connection required by the probe."""

    def write(self, payload: bytes) -> int: ...

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


SerialFactory = Callable[..., SerialConnection]


class Vd1124ProtocolError(RuntimeError):
    """Raised when the STM32 bridge returns an invalid protocol response."""


@dataclass(frozen=True)
class Vd1124ProbeResult:
    """Identity values returned by a read-only VD1124 health probe."""

    port: str
    chip_id: int
    chip_revision: int
    matched: bool


@dataclass(frozen=True)
class Vd1124FieldValue:
    name: str
    channel: int
    address: int
    raw_value: int
    value: int


@dataclass(frozen=True)
class Vd1124WriteResult:
    name: str
    channel: int
    address: int
    before_raw: int
    before_value: int
    requested_value: int
    after_raw: int
    after_value: int
    verified: bool


class Vd1124Probe:
    """Read-only VD1124 identity probe over the STM32 serial-to-I2C bridge."""

    def __init__(
        self,
        *,
        port: str = "COM7",
        baudrate: int = 115_200,
        address: int = 0xA6,
        timeout_s: float = 1.0,
        page_register: int = 0xFF,
        chip_id_register: int = 0x0004,
        expected_chip_id: int = 0x0B,
        chip_revision_register: int = 0x0005,
        expected_chip_revision: int = 0x0D,
        serial_factory: SerialFactory | None = None,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a non-empty string")
        if type(baudrate) is not int or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._validate_byte(address, "address")
        self._validate_byte(page_register, "page_register")
        self._validate_register(chip_id_register, "chip_id_register")
        self._validate_byte(expected_chip_id, "expected_chip_id")
        self._validate_register(
            chip_revision_register,
            "chip_revision_register",
        )
        self._validate_byte(expected_chip_revision, "expected_chip_revision")

        self.port = port.strip()
        self.baudrate = baudrate
        self.address = address
        self.timeout_s = float(timeout_s)
        self.page_register = page_register
        self.chip_id_register = chip_id_register
        self.expected_chip_id = expected_chip_id
        self.chip_revision_register = chip_revision_register
        self.expected_chip_revision = expected_chip_revision
        self._serial_factory = serial_factory
        self._serial: SerialConnection | None = None

    @staticmethod
    def _validate_byte(value: int, label: str) -> None:
        if type(value) is not int or not 0 <= value <= 0xFF:
            raise ValueError(f"{label} must be an integer from 0x00 to 0xFF")

    @staticmethod
    def _validate_register(value: int, label: str) -> None:
        if type(value) is not int or not 0 <= value <= 0xFFFF:
            raise ValueError(f"{label} must be a 16-bit register address")

    def open(self) -> Vd1124Probe:
        """Open the configured serial bridge; repeated calls are harmless."""
        if self._serial is not None:
            return self

        factory = self._serial_factory
        if factory is None:
            serial_module = importlib.import_module("serial")
            factory = serial_module.Serial

        self._serial = factory(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
        )
        return self

    def close(self) -> None:
        """Close the serial bridge; repeated calls are harmless."""
        connection = self._serial
        self._serial = None
        if connection is not None:
            connection.close()

    def __enter__(self) -> Vd1124Probe:
        return self.open()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.close()
        return False

    def probe(self) -> Vd1124ProbeResult:
        """Read chip ID and revision without exposing any register-write API."""
        if self._serial is None:
            raise RuntimeError("VD1124 serial connection is not open")

        chip_id = self._read_register(self.chip_id_register)
        chip_revision = self._read_register(self.chip_revision_register)
        return Vd1124ProbeResult(
            port=self.port,
            chip_id=chip_id,
            chip_revision=chip_revision,
            matched=(
                chip_id == self.expected_chip_id
                and chip_revision == self.expected_chip_revision
            ),
        )

    def _read_register(self, register: int) -> int:
        connection = self._serial
        if connection is None:
            raise RuntimeError("VD1124 serial connection is not open")

        page = register >> 8
        offset = register & 0xFF
        page_frame = bytes(
            (5, 2, 3, self.address, self.page_register, page)
        )
        self._send_complete_frame(connection, page_frame)
        page_ack = connection.read(3)
        if len(page_ack) != 3:
            raise Vd1124ProtocolError(
                "page-select ACK must contain 3 bytes, "
                f"received {len(page_ack)}"
            )
        if page_ack[1] != 1:
            raise Vd1124ProtocolError(
                f"page-select ACK rejected with status 0x{page_ack[1]:02X}"
            )

        read_frame = bytes(
            (8, 2, 2, self.address, offset, 3, 2, self.address, 1)
        )
        self._send_complete_frame(connection, read_frame)
        response = connection.read(6)
        if len(response) != 6:
            raise Vd1124ProtocolError(
                "read response must contain 6 bytes, "
                f"received {len(response)}"
            )
        if response[1] != 1:
            raise Vd1124ProtocolError(
                f"read response rejected with status 0x{response[1]:02X}"
            )
        return response[5]

    @staticmethod
    def _send_complete_frame(
        connection: SerialConnection,
        frame: bytes,
    ) -> None:
        written = connection.write(frame)
        if written != len(frame):
            raise Vd1124ProtocolError(
                f"serial write accepted {written} of {len(frame)} bytes"
            )


class Vd1124RegisterReader:
    """Explicit read-only byte/bitfield access over the STM32 bridge."""

    def __init__(
        self,
        *,
        port: str = "COM7",
        baudrate: int = 115_200,
        address: int = 0xA6,
        timeout_s: float = 1.0,
        page_register: int = 0xFF,
        serial_factory: SerialFactory | None = None,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a non-empty string")
        if type(baudrate) is not int or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        Vd1124Probe._validate_byte(address, "address")
        Vd1124Probe._validate_byte(page_register, "page_register")
        self.port = port.strip()
        self.baudrate = baudrate
        self.address = address
        self.timeout_s = float(timeout_s)
        self.page_register = page_register
        self._serial_factory = serial_factory
        self._serial: SerialConnection | None = None

    def open(self) -> Vd1124RegisterReader:
        if self._serial is not None:
            return self
        factory = self._serial_factory
        if factory is None:
            factory = importlib.import_module("serial").Serial
        self._serial = factory(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
        )
        return self

    def close(self) -> None:
        connection = self._serial
        self._serial = None
        if connection is not None:
            connection.close()

    def __enter__(self) -> Vd1124RegisterReader:
        return self.open()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.close()
        return False

    def read_raw(self, register: int) -> int:
        Vd1124Probe._validate_register(register, "register")
        connection = self._require_connection()
        page = register >> 8
        offset = register & 0xFF
        self._write_frame(
            connection,
            bytes((5, 2, 3, self.address, self.page_register, page)),
        )
        self._read_ack(connection, "page-select")
        self._write_frame(
            connection,
            bytes((8, 2, 2, self.address, offset, 3, 2, self.address, 1)),
        )
        response = connection.read(6)
        if len(response) != 6:
            raise Vd1124ProtocolError(
                f"read response must contain 6 bytes, received {len(response)}"
            )
        if response[1] != 1:
            raise Vd1124ProtocolError(
                f"read response rejected with status 0x{response[1]:02X}"
            )
        return response[5]

    def read_field(
        self, field: ParameterFieldConfig, *, channel: int
    ) -> Vd1124FieldValue:
        address = field.address_for_channel(channel)
        raw = self.read_raw(address)
        value = (raw >> field.bit_start) & field.encoded_max
        return Vd1124FieldValue(field.name, channel, address, raw, value)

    def read_fields(
        self,
        fields: tuple[ParameterFieldConfig, ...],
        *,
        channel: int,
    ) -> tuple[Vd1124FieldValue, ...]:
        return tuple(self.read_field(field, channel=channel) for field in fields)

    def _require_connection(self) -> SerialConnection:
        if self._serial is None:
            raise RuntimeError("VD1124 serial connection is not open")
        return self._serial

    @staticmethod
    def _write_frame(connection: SerialConnection, frame: bytes) -> None:
        Vd1124Probe._send_complete_frame(connection, frame)

    @staticmethod
    def _read_ack(connection: SerialConnection, operation: str) -> None:
        ack = connection.read(3)
        if len(ack) != 3:
            raise Vd1124ProtocolError(
                f"{operation} ACK must contain 3 bytes, received {len(ack)}"
            )
        if ack[1] != 1:
            raise Vd1124ProtocolError(
                f"{operation} ACK rejected with status 0x{ack[1]:02X}"
            )


class Vd1124RegisterWriter(Vd1124RegisterReader):
    """Opt-in VD1124 writer with exact stale/readback protection."""

    def write_field(
        self,
        field: ParameterFieldConfig,
        *,
        channel: int,
        value: int,
        expected_current: int,
        expected_raw: int | None = None,
    ) -> Vd1124WriteResult:
        if type(value) is not int or not 0 <= value <= field.encoded_max:
            raise ValueError(
                f"{field.name} value must be between 0 and {field.encoded_max}"
            )
        if type(expected_current) is not int:
            raise TypeError("expected_current must be an integer")
        if expected_raw is not None:
            Vd1124Probe._validate_byte(expected_raw, "expected_raw")
        before = self.read_field(field, channel=channel)
        if before.value != expected_current:
            raise Vd1124ProtocolError(
                f"stale {field.name}: expected {expected_current}, read {before.value}"
            )
        if expected_raw is not None and before.raw_value != expected_raw:
            raise Vd1124ProtocolError(
                f"stale {field.name} raw byte: expected 0x{expected_raw:02X}, "
                f"read 0x{before.raw_value:02X}"
            )

        mask = field.encoded_max << field.bit_start
        requested_raw = (before.raw_value & ~mask) | (value << field.bit_start)
        self._write_raw(before.address, requested_raw)
        after = self.read_field(field, channel=channel)
        if after.raw_value != requested_raw or after.value != value:
            raise Vd1124ProtocolError(
                f"{field.name} readback mismatch: requested field {value}, "
                f"raw 0x{requested_raw:02X}; read field {after.value}, "
                f"raw 0x{after.raw_value:02X}"
            )
        return Vd1124WriteResult(
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

    def _write_raw(self, register: int, value: int) -> None:
        Vd1124Probe._validate_register(register, "register")
        Vd1124Probe._validate_byte(value, "value")
        connection = self._require_connection()
        page = register >> 8
        offset = register & 0xFF
        self._write_frame(
            connection,
            bytes((5, 2, 3, self.address, self.page_register, page)),
        )
        self._read_ack(connection, "page-select")
        self._write_frame(
            connection,
            bytes((5, 2, 3, self.address, offset, value)),
        )
        self._read_ack(connection, "register-write")
