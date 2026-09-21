from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import time
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


class Vd1124PageRestoreError(Vd1124ProtocolError):
    """Raised when a READ_ONLY capture cannot prove page restoration."""

    def __init__(self, message: str, *, audit: "Vd1124ReadAudit") -> None:
        super().__init__(message)
        self.audit = audit


class Vd1124ReadOnlyCaptureError(Vd1124ProtocolError):
    """Raised after a failed READ_ONLY read whose page restore was verified."""

    def __init__(self, message: str, *, audit: "Vd1124ReadAudit") -> None:
        super().__init__(message)
        self.audit = audit


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
class Vd1124PageSelectEvent:
    page: int
    reason: str
    frame: bytes
    acknowledgement: bytes


@dataclass(frozen=True)
class Vd1124ReadAudit:
    entry_page: int
    page_selects: tuple[Vd1124PageSelectEvent, ...]
    restore_target: int
    restore_readback: int | None
    restore_verified: bool
    tunable_register_writes: int = 0


@dataclass(frozen=True)
class Vd1124ReadOnlyCapture:
    identity: Vd1124ProbeResult
    fields: tuple[Vd1124FieldValue, ...]
    audit: Vd1124ReadAudit


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
    apply_duration_s: float | None = None
    readback_duration_s: float | None = None


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

    def probe_preserving_page(self) -> Vd1124ProbeResult:
        """Probe identity and restore the page selected on entry."""

        connection = self._require_connection()
        original_page = self._read_offset(connection, self.page_register)
        try:
            return self.probe()
        finally:
            self._select_page(connection, original_page)

    def _read_register(self, register: int) -> int:
        connection = self._serial
        if connection is None:
            raise RuntimeError("VD1124 serial connection is not open")

        page = register >> 8
        offset = register & 0xFF
        self._select_page(connection, page)
        return self._read_offset(connection, offset)

    def _require_connection(self) -> SerialConnection:
        connection = self._serial
        if connection is None:
            raise RuntimeError("VD1124 serial connection is not open")
        return connection

    def _select_page(self, connection: SerialConnection, page: int) -> None:
        page_frame = bytes((5, 2, 3, self.address, self.page_register, page))
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

    def _read_offset(self, connection: SerialConnection, offset: int) -> int:
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
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
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
        self._monotonic_ns = monotonic_ns
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
        self._select_page(connection, page)
        return self._read_offset(connection, offset)

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

    def read_fields_preserving_page(
        self,
        fields: tuple[ParameterFieldConfig, ...],
        *,
        channel: int,
    ) -> tuple[Vd1124FieldValue, ...]:
        """Read fields and restore the page selected on entry."""

        connection = self._require_connection()
        original_page = self._read_offset(connection, self.page_register)
        try:
            return self.read_fields(fields, channel=channel)
        finally:
            self._select_page(connection, original_page)

    def _require_connection(self) -> SerialConnection:
        if self._serial is None:
            raise RuntimeError("VD1124 serial connection is not open")
        return self._serial

    @staticmethod
    def _write_frame(connection: SerialConnection, frame: bytes) -> None:
        Vd1124Probe._send_complete_frame(connection, frame)

    def _select_page(self, connection: SerialConnection, page: int) -> None:
        self._write_frame(
            connection,
            bytes((5, 2, 3, self.address, self.page_register, page)),
        )
        self._read_ack(connection, "page-select")

    def _read_offset(self, connection: SerialConnection, offset: int) -> int:
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


class Vd1124ReadOnlySession:
    """One safely-opened serial session for an audited READ_ONLY capture.

    The only bridge writes this class can emit are page-select frames required
    to address paged registers. It deliberately exposes no tunable-register
    write method.
    """

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        address: int,
        timeout_s: float = 1.0,
        page_register: int = 0xFF,
        serial_factory: SerialFactory | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a non-empty string")
        if type(baudrate) is not int or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        Vd1124Probe._validate_byte(address, "address")
        Vd1124Probe._validate_byte(page_register, "page_register")
        self.port = port.strip()
        self.baudrate = baudrate
        self.address = address
        self.timeout_s = float(timeout_s)
        self.page_register = page_register
        self._serial_factory = serial_factory
        self._monotonic_ns = monotonic_ns
        self._serial: SerialConnection | None = None
        self._deadline_monotonic: float | None = None

    def open(self) -> "Vd1124ReadOnlySession":
        if self._serial is not None:
            return self
        factory = self._serial_factory
        if factory is None:
            factory = importlib.import_module("serial").Serial
        connection = factory(
            port=None,
            baudrate=self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
            rtscts=False,
            dsrdtr=False,
        )
        try:
            if not all(hasattr(connection, name) for name in ("dtr", "rts", "port", "open")):
                raise Vd1124ProtocolError(
                    "serial factory does not support connection-free DTR/RTS control"
                )
            connection.dtr = False
            connection.rts = False
            connection.port = self.port
            connection.open()
        except Exception:
            try:
                connection.close()
            finally:
                raise
        self._serial = connection
        return self

    def close(self) -> None:
        connection = self._serial
        self._serial = None
        if connection is not None:
            connection.close()

    def set_timeout_s(self, timeout_s: float) -> None:
        """Cap subsequent bridge reads/writes to the remaining Preflight time."""
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self._deadline_monotonic = time.monotonic() + self.timeout_s
        connection = self._require_connection()
        if hasattr(connection, "timeout"):
            connection.timeout = self.timeout_s
        if hasattr(connection, "write_timeout"):
            connection.write_timeout = self.timeout_s

    def _refresh_timeout(self, *, restore: bool = False) -> None:
        deadline = self._deadline_monotonic
        if deadline is None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not restore:
            raise Vd1124ProtocolError(
                "VD1124 READ_ONLY capture exceeded its total timeout"
            )
        # Restoration gets a small, bounded emergency window even if a prior
        # read consumed the deadline. Leaving the bridge on another page is a
        # greater safety failure than this final bounded cleanup attempt.
        timeout_s = max(0.001, remaining) if remaining > 0 else 0.25
        connection = self._require_connection()
        if hasattr(connection, "timeout"):
            connection.timeout = timeout_s
        if hasattr(connection, "write_timeout"):
            connection.write_timeout = timeout_s

    def __enter__(self) -> "Vd1124ReadOnlySession":
        return self.open()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.close()
        return False

    def capture_identity_and_fields(
        self,
        fields: tuple[ParameterFieldConfig, ...],
        *,
        channel: int,
        chip_id_register: int,
        expected_chip_id: int,
        chip_revision_register: int,
        expected_chip_revision: int,
    ) -> Vd1124ReadOnlyCapture:
        connection = self._require_connection()
        Vd1124Probe._validate_register(chip_id_register, "chip_id_register")
        Vd1124Probe._validate_register(
            chip_revision_register, "chip_revision_register"
        )
        entry_page = self._read_offset(connection, self.page_register)
        current_page = entry_page
        events: list[Vd1124PageSelectEvent] = []

        def select(page: int, reason: str, *, force: bool = False) -> None:
            nonlocal current_page
            if page == current_page and not force:
                return
            frame = bytes(
                (5, 2, 3, self.address, self.page_register, page)
            )
            acknowledgement = b""
            try:
                self._refresh_timeout(restore=reason == "restore")
                Vd1124Probe._send_complete_frame(connection, frame)
                self._refresh_timeout(restore=reason == "restore")
                acknowledgement = connection.read(3)
            finally:
                events.append(
                    Vd1124PageSelectEvent(
                        page=page,
                        reason=reason,
                        frame=frame,
                        acknowledgement=acknowledgement,
                    )
                )
            if len(acknowledgement) != 3:
                raise Vd1124ProtocolError(
                    "page-select ACK must contain 3 bytes, "
                    f"received {len(acknowledgement)}"
                )
            if acknowledgement[1] != 1:
                raise Vd1124ProtocolError(
                    "page-select ACK rejected with status "
                    f"0x{acknowledgement[1]:02X}"
                )
            current_page = page

        def read_register(register: int, reason: str) -> int:
            Vd1124Probe._validate_register(register, "register")
            select(register >> 8, reason)
            return self._read_offset(connection, register & 0xFF)

        fields_result: tuple[Vd1124FieldValue, ...] | None = None
        identity: Vd1124ProbeResult | None = None
        primary_error: BaseException | None = None
        try:
            chip_id = read_register(chip_id_register, "identity")
            chip_revision = read_register(
                chip_revision_register, "identity"
            )
            identity = Vd1124ProbeResult(
                port=self.port,
                chip_id=chip_id,
                chip_revision=chip_revision,
                matched=(
                    chip_id == expected_chip_id
                    and chip_revision == expected_chip_revision
                ),
            )
            values: list[Vd1124FieldValue] = []
            for field in fields:
                address = field.address_for_channel(channel)
                raw = read_register(address, f"field:{field.name}")
                values.append(
                    Vd1124FieldValue(
                        name=field.name,
                        channel=channel,
                        address=address,
                        raw_value=raw,
                        value=(raw >> field.bit_start) & field.encoded_max,
                    )
                )
            fields_result = tuple(values)
        except BaseException as error:
            primary_error = error

        restore_readback: int | None = None
        restore_error: BaseException | None = None
        try:
            select(entry_page, "restore", force=True)
            restore_readback = self._read_offset(
                connection, self.page_register, restore=True
            )
            if restore_readback != entry_page:
                raise Vd1124ProtocolError(
                    "page restore readback mismatch: "
                    f"expected 0x{entry_page:02X}, got 0x{restore_readback:02X}"
                )
        except BaseException as error:
            restore_error = error

        audit = Vd1124ReadAudit(
            entry_page=entry_page,
            page_selects=tuple(events),
            restore_target=entry_page,
            restore_readback=restore_readback,
            restore_verified=restore_error is None,
            tunable_register_writes=0,
        )
        if restore_error is not None:
            raise Vd1124PageRestoreError(
                f"VD1124 page restoration could not be verified: {restore_error}",
                audit=audit,
            ) from restore_error
        if primary_error is not None:
            raise Vd1124ReadOnlyCaptureError(
                f"VD1124 READ_ONLY capture failed: {primary_error}",
                audit=audit,
            ) from primary_error
        if identity is None or fields_result is None:
            raise Vd1124ProtocolError("READ_ONLY capture did not produce complete data")
        return Vd1124ReadOnlyCapture(
            identity=identity,
            fields=fields_result,
            audit=audit,
        )

    def _require_connection(self) -> SerialConnection:
        if self._serial is None:
            raise RuntimeError("VD1124 serial connection is not open")
        return self._serial

    def _read_offset(
        self,
        connection: SerialConnection,
        offset: int,
        *,
        restore: bool = False,
    ) -> int:
        frame = bytes(
            (8, 2, 2, self.address, offset, 3, 2, self.address, 1)
        )
        self._refresh_timeout(restore=restore)
        Vd1124Probe._send_complete_frame(connection, frame)
        self._refresh_timeout(restore=restore)
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
        before_read_start = self._monotonic_ns()
        before = self.read_field(field, channel=channel)
        readback_duration_ns = max(
            0, self._monotonic_ns() - before_read_start
        )
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
        apply_start = self._monotonic_ns()
        self._write_raw(before.address, requested_raw)
        apply_duration_s = max(
            0, self._monotonic_ns() - apply_start
        ) / 1e9
        after_read_start = self._monotonic_ns()
        after = self.read_field(field, channel=channel)
        readback_duration_ns += max(
            0, self._monotonic_ns() - after_read_start
        )
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
            apply_duration_s,
            readback_duration_ns / 1e9,
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
