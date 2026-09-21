from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from math import isfinite
import re
from types import MappingProxyType
from typing import Protocol

from vcsel_ai_tuning.hardware.live_config import ErMeasurementConfig


class QuerySession(Protocol):
    """The session interface needed by the source-bound DCA adapter."""

    def write(self, command: str) -> object: ...
    def query(self, command: str) -> str: ...


class DcaReadError(RuntimeError):
    """Raised when a FlexDCA response cannot produce a safe snapshot."""

    def __init__(
        self,
        message: str,
        *,
        raw_responses: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_responses = MappingProxyType(dict(raw_responses or {}))


class DcaSourceMismatchError(DcaReadError):
    """Raised before a value query when a measurement source readback differs."""

    def __init__(
        self,
        message: str,
        *,
        measurement: str,
        requested_source: str,
        current_source: str,
        raw_responses: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message, raw_responses=raw_responses)
        self.measurement = measurement
        self.requested_source = requested_source
        self.current_source = current_source


@dataclass(frozen=True)
class DcaSnapshot:
    timestamp_utc: datetime
    function: int
    measurement_source: str
    verified_measurement_source: str
    idn: str
    tdecq_db: float
    status: str
    details: str
    taps: tuple[float, ...]
    precursor_count: int
    main_index: int
    raw_responses: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_responses",
            MappingProxyType(dict(self.raw_responses)),
        )


@dataclass(frozen=True)
class DcaScalarSnapshot:
    timestamp_utc: datetime
    source: str
    units: str
    status: str
    details: str
    value_db: float | None
    valid: bool
    invalid_reason: str | None
    raw_responses: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "raw_responses", MappingProxyType(dict(self.raw_responses))
        )


def _query_all(
    session: QuerySession, commands: tuple[str, ...]
) -> dict[str, str]:
    responses: dict[str, str] = {}
    for command in commands:
        try:
            response = session.query(command)
        except Exception as error:
            raise DcaReadError(
                f"query {command!r} failed: {error}", raw_responses=responses
            ) from error
        if not isinstance(response, str):
            raise DcaReadError(
                f"query {command!r} returned {type(response).__name__}, expected str",
                raw_responses=responses,
            )
        responses[command] = response
    return responses


def _parse_finite_float(raw: str, label: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError as error:
        raise DcaReadError(f"{label} response is not a number: {raw!r}") from error
    if not isfinite(value):
        raise DcaReadError(f"{label} must be finite, got {raw!r}")
    return value


def _parse_taps(raw: str) -> tuple[float, ...]:
    fields = [field.strip() for field in raw.strip().split(",")]
    if not fields or any(not field for field in fields):
        raise DcaReadError(f"equalizer taps response is not valid CSV: {raw!r}")
    try:
        taps = tuple(float(field) for field in fields)
    except ValueError as error:
        raise DcaReadError(
            f"equalizer taps response contains a non-numeric value: {raw!r}"
        ) from error
    if any(not isfinite(tap) for tap in taps):
        raise DcaReadError(f"equalizer taps must be finite, got {raw!r}")
    return taps


def _parse_precursor_count(raw: str) -> int:
    text = raw.strip()
    try:
        count = int(text, 10)
    except ValueError as error:
        raise DcaReadError(
            f"precursor count must be an integer, got {raw!r}"
        ) from error
    if count < 0:
        raise DcaReadError(f"precursor count must be non-negative, got {count}")
    return count


def read_verified_oer(
    session: QuerySession, config: ErMeasurementConfig
) -> DcaScalarSnapshot:
    """Select and verify the configured PAM Outer ER source, then read it."""
    source, source_raw = select_oer_source(session, config)

    commands = (
        config.status_query,
        config.details_query,
        config.value_query,
        config.units_query,
    )
    raw = dict(source_raw)
    raw.update(_query_all(session, commands))
    raw_units = raw[config.units_query].strip().upper()
    if raw_units not in {"DB", "DEC", "DECIBEL"}:
        raise DcaReadError(
            f"OER units {raw_units or '<empty>'} are not decibels",
            raw_responses=raw,
        )
    status = raw[config.status_query].strip().upper()
    details = raw[config.details_query].strip()
    value: float | None = None
    invalid_reason: str | None = None
    if status != "CORR":
        invalid_reason = f"OER status is {status or '<empty>'}: {details}"
    else:
        try:
            candidate = float(raw[config.value_query].strip())
        except ValueError:
            invalid_reason = "OER response is not numeric"
        else:
            if not isfinite(candidate):
                invalid_reason = "OER response is not finite"
            elif abs(candidate) >= 9.0e37:
                invalid_reason = "OER response is an instrument sentinel"
            else:
                value = candidate
    return DcaScalarSnapshot(
        timestamp_utc=datetime.now(timezone.utc),
        source=source,
        units="DECIBEL",
        status=status,
        details=details,
        value_db=value,
        valid=invalid_reason is None,
        invalid_reason=invalid_reason,
        raw_responses=raw,
    )


def select_oer_source(
    session: QuerySession, config: ErMeasurementConfig
) -> tuple[str, Mapping[str, str]]:
    """Select and immediately verify the configured Outer ER source."""
    requested_source = _normalize_measurement_source(config.source)
    select_command = f":MEASure:EYE:OER:SOURce {requested_source}"
    try:
        session.write(select_command)
    except Exception as error:
        raise DcaReadError(
            f"OER source selection {select_command!r} failed: {error}"
        ) from error

    source_raw = _query_all(session, (config.source_query,))
    raw_source = source_raw[config.source_query].strip()
    try:
        source = _normalize_measurement_source(raw_source)
    except ValueError as error:
        raise DcaReadError(
            f"OER source readback is not a supported waveform: "
            f"{raw_source or '<empty>'}",
            raw_responses=source_raw,
        ) from error
    if source != requested_source:
        raise DcaSourceMismatchError(
            f"OER source readback {source} does not match configured source "
            f"{requested_source}",
            measurement="outer_er",
            requested_source=requested_source,
            current_source=source,
            raw_responses=source_raw,
        )
    return source, source_raw


def _normalize_measurement_source(source: str) -> str:
    if not isinstance(source, str):
        raise ValueError("measurement_source must be a string")
    text = source.strip()
    patterns = (
        (r"CHAN(?:NEL)?([1-8])([A-D])", lambda m: f"CHAN{m.group(1)}{m.group(2).upper()}"),
        (r"FUNC(?:TION)?([0-9]+)", lambda m: f"FUNC{int(m.group(1), 10)}"),
        (r"DIFF([1-4])", lambda m: f"DIFF{m.group(1)}"),
        (r"CGM(?:EMORY)?([1-8])", lambda m: f"CGMEMORY{m.group(1)}"),
    )
    for pattern, formatter in patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        canonical = formatter(match)
        if canonical.startswith("FUNC") and not 1 <= int(canonical[4:]) <= 16:
            break
        return canonical
    raise ValueError("measurement_source must identify a supported DCA waveform")


def _normalize_tdecq_source(source: str) -> str:
    canonical = _normalize_measurement_source(source)
    if not canonical.startswith("FUNC"):
        raise ValueError("TDECQ measurement_source must identify a Function")
    return canonical


def _require_empty_scpi_error_queue(session: QuerySession) -> str:
    command = ":SYSTem:ERRor:COUNt?"
    raw = _query_all(session, (command,))[command]
    try:
        count = int(raw.strip(), 10)
    except ValueError as error:
        raise DcaReadError(
            f"SCPI error queue count is not an integer: {raw!r}",
            raw_responses={command: raw},
        ) from error
    if count != 0:
        raise DcaReadError(
            f"SCPI error queue contains {count} error(s)",
            raw_responses={command: raw},
        )
    return raw


def _write_without_scpi_error(
    session: QuerySession,
    command: str,
    raw_responses: dict[str, str],
) -> None:
    try:
        session.write(command)
    except Exception as error:
        raise DcaReadError(
            f"DCA command {command!r} failed: {error}",
            raw_responses=raw_responses,
        ) from error
    raw_responses[":SYSTem:ERRor:COUNt?"] = _require_empty_scpi_error_queue(
        session
    )


def configure_and_verify_tdecq_function1(
    session: QuerySession,
) -> Mapping[str, str]:
    """Define FUNC1 as a CHAN1A TDECQ equalizer and verify its readback."""

    raw_responses: dict[str, str] = {
        ":SYSTem:ERRor:COUNt?": _require_empty_scpi_error_queue(session)
    }
    for command in (
        ":FUNCtion1:FOPerator TEQualizer",
        ":FUNCtion1:OPERand1 CHAN1A",
        ":FUNCtion1:DISPlay ON",
    ):
        _write_without_scpi_error(session, command, raw_responses)

    queries = (
        ":FUNCtion1:STATus?",
        ":FUNCtion1:FOPerator?",
        ":FUNCtion1:OPERand1?",
        ":FUNCtion1:DISPlay?",
        ":FUNCtion1:SIGNal:TYPE?",
    )
    raw_responses.update(_query_all(session, queries))
    status = raw_responses[queries[0]].strip().upper()
    operator = raw_responses[queries[1]].strip().upper()
    operand = raw_responses[queries[2]].strip()
    display = raw_responses[queries[3]].strip().upper()
    signal_type = raw_responses[queries[4]].strip().upper()
    if status != "AVA":
        raise DcaReadError("FUNC1 is not available", raw_responses=raw_responses)
    if operator not in {"TEQ", "TEQUALIZER"}:
        raise DcaReadError(
            "FUNC1 is not a TDECQ equalizer", raw_responses=raw_responses
        )
    try:
        canonical_operand = _normalize_measurement_source(operand)
    except ValueError as error:
        raise DcaReadError(
            "FUNC1 operand is not a supported waveform",
            raw_responses=raw_responses,
        ) from error
    if canonical_operand != "CHAN1A":
        raise DcaReadError(
            f"FUNC1 operand is {canonical_operand}, not CHAN1A",
            raw_responses=raw_responses,
        )
    if display != "1":
        raise DcaReadError("FUNC1 is not displayed", raw_responses=raw_responses)
    if signal_type != "PAM4":
        raise DcaReadError(
            f"FUNC1 signal type is {signal_type or '<empty>'}, not PAM4",
            raw_responses=raw_responses,
        )

    _write_without_scpi_error(
        session, ":MEASure:EYE:TDEQ:SOURce FUNC1", raw_responses
    )
    _, source_responses = _verify_measurement_source(session, "FUNC1")
    raw_responses.update(source_responses)
    return MappingProxyType(raw_responses)


def _bind_measurement_source(
    session: QuerySession, measurement_source: str
) -> tuple[str, dict[str, str]]:
    select_command = f":MEASure:EYE:TDEQ:SOURce {measurement_source}"
    try:
        session.write(select_command)
    except Exception as error:
        raise DcaReadError(
            f"measurement source selection {select_command!r} failed: {error}"
        ) from error

    return _verify_measurement_source(session, measurement_source)


def _verify_measurement_source(
    session: QuerySession, measurement_source: str
) -> tuple[str, dict[str, str]]:
    """Read back a previously selected source without changing DCA state."""

    source_command = ":MEASure:EYE:TDEQ:SOURce?"
    raw_responses = _query_all(session, (source_command,))
    raw_source = raw_responses[source_command].strip()
    try:
        verified_source = _normalize_measurement_source(raw_source)
    except ValueError as error:
        raise DcaReadError(
            "TDECQ source readback is not a supported waveform: "
            f"{raw_source or '<empty>'}",
            raw_responses=raw_responses,
        ) from error
    if verified_source != measurement_source:
        raise DcaSourceMismatchError(
            f"TDECQ source readback {verified_source} does not match "
            f"configured source {measurement_source}",
            measurement="tdecq",
            requested_source=measurement_source,
            current_source=verified_source,
            raw_responses=raw_responses,
        )
    return verified_source, raw_responses


def verify_tdecq_source(
    session: QuerySession, *, measurement_source: str
) -> tuple[str, Mapping[str, str]]:
    """Verify the configured TDECQ source without reading a result value."""
    canonical_source = _normalize_tdecq_source(measurement_source)
    return _verify_measurement_source(session, canonical_source)


def select_tdecq_source(
    session: QuerySession, *, measurement_source: str
) -> tuple[str, Mapping[str, str]]:
    """Select and immediately verify the configured TDECQ source."""
    canonical_source = _normalize_tdecq_source(measurement_source)
    return _bind_measurement_source(session, canonical_source)


def _read_tdecq_snapshot(
    session: QuerySession,
    *,
    function: int,
    canonical_source: str,
    verified_source: str,
    source_responses: Mapping[str, str],
) -> DcaSnapshot:
    idn_command = "*IDN?"
    tdecq_command = ":MEASure:EYE:TDEQ?"
    status_command = ":MEASure:EYE:TDEQ:STATus?"
    details_command = ":MEASure:EYE:TDEQ:STATus:DETails?"
    taps_command = f":SPRocess{function}:TEQualizer:TAPS?"
    precursor_command = f":SPRocess{function}:TEQualizer:NPRecursors?"
    commands = (
        status_command,
        details_command,
        tdecq_command,
        idn_command,
        taps_command,
        precursor_command,
    )
    raw_responses = dict(source_responses)
    raw_responses.update(_query_all(session, commands))

    idn = raw_responses[idn_command].strip()
    try:
        tdecq_db = _parse_finite_float(raw_responses[tdecq_command], "TDECQ")
    except DcaReadError as error:
        raise DcaReadError(str(error), raw_responses=raw_responses) from error
    if abs(tdecq_db) >= 9.0e37:
        raise DcaReadError(
            f"TDECQ response is an instrument sentinel: "
            f"{raw_responses[tdecq_command]!r}",
            raw_responses=raw_responses,
        )
    status = raw_responses[status_command].strip().upper()
    details = raw_responses[details_command].strip()
    if status != "CORR":
        raise DcaReadError(
            f"TDECQ status is {status or '<empty>'}, not CORR; details: {details}",
            raw_responses=raw_responses,
        )

    try:
        taps = _parse_taps(raw_responses[taps_command])
        precursor_count = _parse_precursor_count(raw_responses[precursor_command])
    except DcaReadError as error:
        raise DcaReadError(str(error), raw_responses=raw_responses) from error
    main_index = precursor_count
    if main_index >= len(taps):
        raise DcaReadError(
            f"precursor count {precursor_count} gives main index {main_index}, "
            f"but only {len(taps)} taps were returned",
            raw_responses=raw_responses,
        )
    if taps[main_index] == 0:
        raise DcaReadError(
            f"Main tap at index {main_index} must be non-zero",
            raw_responses=raw_responses,
        )

    return DcaSnapshot(
        timestamp_utc=datetime.now(timezone.utc),
        function=function,
        measurement_source=canonical_source,
        verified_measurement_source=verified_source,
        idn=idn,
        tdecq_db=tdecq_db,
        status=status,
        details=details,
        taps=taps,
        precursor_count=precursor_count,
        main_index=main_index,
        raw_responses=raw_responses,
    )


def read_verified_tdecq(
    session: QuerySession,
    function: int = 1,
    *,
    measurement_source: str,
) -> DcaSnapshot:
    """Select and verify the Function source immediately before reading TDECQ."""
    if type(function) is not int or function <= 0:
        raise ValueError("function must be a positive integer")
    canonical_source = _normalize_tdecq_source(measurement_source)
    if canonical_source != f"FUNC{function}":
        raise ValueError(
            f"TDECQ measurement_source must be FUNC{function} for function {function}"
        )
    verified_source, source_responses = select_tdecq_source(
        session, measurement_source=canonical_source
    )
    return _read_tdecq_snapshot(
        session,
        function=function,
        canonical_source=canonical_source,
        verified_source=verified_source,
        source_responses=source_responses,
    )


def read_existing_tdecq(
    session: QuerySession,
    function: int = 1,
    *,
    measurement_source: str,
) -> DcaSnapshot:
    """Bind a TDECQ source, then read its already-completed result."""
    if type(function) is not int or function <= 0:
        raise ValueError("function must be a positive integer")
    canonical_source = _normalize_tdecq_source(measurement_source)
    if canonical_source != f"FUNC{function}":
        raise ValueError(
            f"TDECQ measurement_source must be FUNC{function} for function {function}"
        )
    verified_source, source_responses = select_tdecq_source(
        session, measurement_source=canonical_source
    )
    return _read_tdecq_snapshot(
        session,
        function=function,
        canonical_source=canonical_source,
        verified_source=verified_source,
        source_responses=source_responses,
    )


@contextmanager
def open_visa_session(
    resource: str, timeout_ms: int = 60_000
) -> Iterator[QuerySession]:
    """Open and reliably close a VISA query session."""
    pyvisa = importlib.import_module("pyvisa")
    manager = pyvisa.ResourceManager()
    instrument = None
    try:
        instrument = manager.open_resource(resource)
        instrument.timeout = timeout_ms
        instrument.chunk_size = 1024
        yield instrument
    finally:
        try:
            if instrument is not None:
                instrument.close()
        finally:
            manager.close()
