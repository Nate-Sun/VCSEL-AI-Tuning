from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib
from math import isfinite
import re
import time
from types import MappingProxyType
from typing import Callable, Protocol

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


@dataclass(frozen=True)
class DcaMetricResult:
    """One independently verified result from the shared DCA source selector."""

    configured_source: str
    verified_source: str | None
    value_db: float | None
    status: str
    details: str
    valid: bool
    invalid_reason: str | None
    raw_responses: Mapping[str, str]
    status_polls: tuple[Mapping[str, str], ...]
    wait_for_valid_duration_s: float = 0.0
    query_duration_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "raw_responses", MappingProxyType(dict(self.raw_responses))
        )
        object.__setattr__(
            self,
            "status_polls",
            tuple(MappingProxyType(dict(poll)) for poll in self.status_polls),
        )


@dataclass(frozen=True)
class DcaSequentialResult:
    """OER then TDECQ results read from one completed acquisition."""

    er: DcaMetricResult
    tdecq: DcaMetricResult


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


def _read_oer_snapshot(
    session: QuerySession,
    config: ErMeasurementConfig,
    *,
    source: str,
    source_raw: Mapping[str, str],
) -> DcaScalarSnapshot:
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


def read_verified_oer(
    session: QuerySession, config: ErMeasurementConfig
) -> DcaScalarSnapshot:
    """Select and verify the configured PAM Outer ER source, then read it."""
    source, source_raw = select_oer_source(session, config)
    return _read_oer_snapshot(
        session,
        config,
        source=source,
        source_raw=source_raw,
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

    return verify_oer_source(session, config)


def verify_oer_source(
    session: QuerySession, config: ErMeasurementConfig
) -> tuple[str, Mapping[str, str]]:
    """Verify the configured Outer ER source without changing DCA state."""
    requested_source = _normalize_measurement_source(config.source)

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
    return _normalize_measurement_source(source)


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
    """Select and verify the waveform source before reading function-specific taps."""
    if type(function) is not int or not 1 <= function <= 16:
        raise ValueError("function must be between 1 and 16")
    canonical_source = _normalize_tdecq_source(measurement_source)
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
    if type(function) is not int or not 1 <= function <= 16:
        raise ValueError("function must be between 1 and 16")
    canonical_source = _normalize_tdecq_source(measurement_source)
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


def _capture_metric_evidence(
    session: QuerySession,
    commands: tuple[str, ...],
    raw_responses: dict[str, str],
) -> None:
    """Best-effort evidence capture that never hides the primary invalid reason."""
    for command in commands:
        try:
            raw_responses.update(_query_all(session, (command,)))
        except DcaReadError as error:
            raw_responses.update(error.raw_responses)
            raw_responses[f"{command}#error"] = str(error)


def _source_from_raw(
    raw_responses: Mapping[str, str],
    *,
    source_query: str,
    fallback: str | None,
) -> str | None:
    if f"{source_query}#error" in raw_responses:
        return None
    raw_source = raw_responses.get(source_query)
    if raw_source is None:
        return fallback
    try:
        return _normalize_measurement_source(raw_source)
    except ValueError:
        return None


def _source_error_metric(
    error: DcaReadError,
    *,
    label: str,
    configured_source: str,
    source_query: str,
) -> DcaMetricResult:
    raw = dict(error.raw_responses)
    if isinstance(error, DcaSourceMismatchError):
        verified_source: str | None = error.current_source
        status = "SOURCE_MISMATCH"
    else:
        verified_source = _source_from_raw(
            raw, source_query=source_query, fallback=None
        )
        status = "INV"
    return DcaMetricResult(
        configured_source=configured_source,
        verified_source=verified_source,
        value_db=None,
        status=status,
        details=str(error),
        valid=False,
        invalid_reason=f"{label} source selection failed: {error}",
        raw_responses=raw,
        status_polls=(),
    )


def _poll_selected_metric(
    session: QuerySession,
    *,
    label: str,
    status_query: str,
    details_query: str,
    timeout_s: float,
    poll_interval_s: float,
    monotonic: Callable[[], float],
    wait: Callable[[float], None],
    inspect_invalid_details: bool,
    terminal_details: frozenset[str] = frozenset(),
    terminal_detail_prefixes: tuple[str, ...] = (),
) -> tuple[
    str,
    str,
    tuple[Mapping[str, str], ...],
    dict[str, str],
    str | None,
]:
    """Poll one selected metric without changing the shared source selector."""
    deadline = monotonic() + timeout_s
    status = "INV"
    details = ""
    polls: list[dict[str, str]] = []
    raw: dict[str, str] = {}
    while monotonic() < deadline:
        try:
            response = _query_all(session, (status_query,))
        except DcaReadError as error:
            raw.update(error.raw_responses)
            raw[f"{status_query}#error"] = str(error)
            return (
                "INV",
                details,
                tuple(polls),
                raw,
                f"{label} status query failed: {error}",
            )
        raw.update(response)
        status = response[status_query].strip().upper()
        poll = {"status": status}
        if status == "CORR":
            polls.append(poll)
            return status, details, tuple(polls), raw, None
        if inspect_invalid_details:
            try:
                response = _query_all(session, (details_query,))
            except DcaReadError as error:
                raw.update(error.raw_responses)
                raw[f"{details_query}#error"] = str(error)
                polls.append(poll)
                return (
                    "INV",
                    details,
                    tuple(polls),
                    raw,
                    f"{label} details query failed: {error}",
                )
            raw.update(response)
            details = response[details_query].strip()
            poll["details"] = details
            comparable_details = details
            if (
                len(comparable_details) >= 2
                and comparable_details[0] == comparable_details[-1] == '"'
            ):
                comparable_details = comparable_details[1:-1]
            normalized_details = comparable_details.casefold()
            if (
                normalized_details in terminal_details
                or any(
                    normalized_details.startswith(prefix)
                    for prefix in terminal_detail_prefixes
                )
            ):
                polls.append(poll)
                return (
                    status,
                    details,
                    tuple(polls),
                    raw,
                    f"{label} status is {status}: {details}",
                )
        polls.append(poll)
        wait(poll_interval_s)

    if not details:
        try:
            response = _query_all(session, (details_query,))
        except DcaReadError as error:
            raw.update(error.raw_responses)
            raw[f"{details_query}#error"] = str(error)
            details = f"details query failed: {error}"
        else:
            raw.update(response)
            details = response[details_query].strip()
    return (
        "INV",
        details,
        tuple(polls),
        raw,
        f"timeout waiting for {label} CORR after {timeout_s:g}s: {details}",
    )


def _invalid_after_poll(
    *,
    label: str,
    configured_source: str,
    initially_verified_source: str,
    source_query: str,
    details: str,
    invalid_reason: str,
    raw: dict[str, str],
    polls: tuple[Mapping[str, str], ...],
    status: str = "INV",
) -> DcaMetricResult:
    verified_source = _source_from_raw(
        raw,
        source_query=source_query,
        fallback=initially_verified_source,
    )
    final_details = details
    final_reason = invalid_reason
    if verified_source is not None and verified_source != configured_source:
        status = "SOURCE_MISMATCH"
        final_details = (
            f"{label} source readback {verified_source} does not match configured "
            f"source {configured_source}"
        )
        final_reason = f"{label} source validation failed: {final_details}"
    return DcaMetricResult(
        configured_source=configured_source,
        verified_source=verified_source,
        value_db=None,
        status=status,
        details=final_details,
        valid=False,
        invalid_reason=final_reason,
        raw_responses=raw,
        status_polls=polls,
    )


def _read_oer_after_acquisition(
    session: QuerySession,
    config: ErMeasurementConfig,
    *,
    timeout_s: float,
    poll_interval_s: float,
    monotonic: Callable[[], float],
    wait: Callable[[float], None],
) -> DcaMetricResult:
    wait_started = monotonic()
    wait_duration_s = 0.0
    query_started: float | None = None

    def finish(result: DcaMetricResult) -> DcaMetricResult:
        query_duration_s = (
            0.0
            if query_started is None
            else max(0.0, monotonic() - query_started)
        )
        return replace(
            result,
            wait_for_valid_duration_s=wait_duration_s,
            query_duration_s=query_duration_s,
        )

    configured_source = _normalize_measurement_source(config.source)
    raw: dict[str, str] = {}
    try:
        verified_source, selected_raw = select_oer_source(session, config)
    except DcaReadError as error:
        wait_duration_s = max(0.0, monotonic() - wait_started)
        return finish(
            _source_error_metric(
                error,
                label="OER",
                configured_source=configured_source,
                source_query=config.source_query,
            )
        )
    raw.update(selected_raw)
    _status, details, polls, polled_raw, poll_error = _poll_selected_metric(
        session,
        label="OER",
        status_query=config.status_query,
        details_query=config.details_query,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        monotonic=monotonic,
        wait=wait,
        inspect_invalid_details=False,
    )
    wait_duration_s = max(0.0, monotonic() - wait_started)
    raw.update(polled_raw)
    if poll_error is not None:
        query_started = monotonic()
        _capture_metric_evidence(
            session,
            (
                config.source_query,
                config.units_query,
                config.status_query,
                config.details_query,
            ),
            raw,
        )
        return finish(
            _invalid_after_poll(
                label="OER",
                configured_source=configured_source,
                initially_verified_source=verified_source,
                source_query=config.source_query,
                details=details,
                invalid_reason=poll_error,
                raw=raw,
                polls=polls,
            )
        )
    query_started = monotonic()
    try:
        final_source, final_source_raw = verify_oer_source(session, config)
    except DcaReadError as error:
        raw.update(error.raw_responses)
        return finish(DcaMetricResult(
            configured_source=configured_source,
            verified_source=(
                error.current_source
                if isinstance(error, DcaSourceMismatchError)
                else _source_from_raw(
                    error.raw_responses,
                    source_query=config.source_query,
                    fallback=None,
                )
            ),
            value_db=None,
            status=(
                "SOURCE_MISMATCH"
                if isinstance(error, DcaSourceMismatchError)
                else "INV"
            ),
            details=str(error),
            valid=False,
            invalid_reason=f"OER source validation failed: {error}",
            raw_responses=raw,
            status_polls=polls,
        ))
    raw.update(final_source_raw)
    try:
        snapshot = _read_oer_snapshot(
            session,
            config,
            source=final_source,
            source_raw=final_source_raw,
        )
    except DcaReadError as error:
        raw.update(error.raw_responses)
        return finish(DcaMetricResult(
            configured_source=configured_source,
            verified_source=final_source,
            value_db=None,
            status="INV",
            details=raw.get(config.details_query, details).strip(),
            valid=False,
            invalid_reason=f"OER validation failed: {error}",
            raw_responses=raw,
            status_polls=polls,
        ))
    raw.update(snapshot.raw_responses)
    try:
        post_read_source, post_read_raw = verify_oer_source(session, config)
    except DcaReadError as error:
        raw.update(error.raw_responses)
        raw_source = error.raw_responses.get(config.source_query)
        if raw_source is not None:
            raw[f"{config.source_query}#post_read"] = raw_source
        else:
            raw[f"{config.source_query}#post_read#error"] = str(error)
        return finish(DcaMetricResult(
            configured_source=configured_source,
            verified_source=(
                error.current_source
                if isinstance(error, DcaSourceMismatchError)
                else _source_from_raw(
                    error.raw_responses,
                    source_query=config.source_query,
                    fallback=None,
                )
            ),
            value_db=None,
            status=(
                "SOURCE_MISMATCH"
                if isinstance(error, DcaSourceMismatchError)
                else "INV"
            ),
            details=str(error),
            valid=False,
            invalid_reason=f"OER post-read source validation failed: {error}",
            raw_responses=raw,
            status_polls=polls,
        ))
    raw.update(post_read_raw)
    raw[f"{config.source_query}#post_read"] = post_read_raw[config.source_query]
    return finish(DcaMetricResult(
        configured_source=configured_source,
        verified_source=post_read_source,
        value_db=snapshot.value_db,
        status=snapshot.status if snapshot.valid else "INV",
        details=snapshot.details,
        valid=bool(snapshot.valid and snapshot.value_db is not None),
        invalid_reason=snapshot.invalid_reason,
        raw_responses=raw,
        status_polls=polls,
    ))


def _read_tdecq_after_acquisition(
    session: QuerySession,
    *,
    function: int,
    source: str,
    timeout_s: float,
    poll_interval_s: float,
    monotonic: Callable[[], float],
    wait: Callable[[float], None],
) -> DcaMetricResult:
    wait_started = monotonic()
    wait_duration_s = 0.0
    query_started: float | None = None

    def finish(result: DcaMetricResult) -> DcaMetricResult:
        query_duration_s = (
            0.0
            if query_started is None
            else max(0.0, monotonic() - query_started)
        )
        return replace(
            result,
            wait_for_valid_duration_s=wait_duration_s,
            query_duration_s=query_duration_s,
        )

    configured_source = _normalize_tdecq_source(source)
    source_query = ":MEASure:EYE:TDEQ:SOURce?"
    status_query = ":MEASure:EYE:TDEQ:STATus?"
    details_query = ":MEASure:EYE:TDEQ:STATus:DETails?"
    value_query = ":MEASure:EYE:TDEQ?"
    taps_query = f":SPRocess{function}:TEQualizer:TAPS?"
    precursors_query = f":SPRocess{function}:TEQualizer:NPRecursors?"
    raw: dict[str, str] = {}
    try:
        verified_source, selected_raw = select_tdecq_source(
            session, measurement_source=configured_source
        )
    except DcaReadError as error:
        wait_duration_s = max(0.0, monotonic() - wait_started)
        return finish(
            _source_error_metric(
                error,
                label="TDECQ",
                configured_source=configured_source,
                source_query=source_query,
            )
        )
    raw.update(selected_raw)
    polled_status, details, polls, polled_raw, poll_error = _poll_selected_metric(
        session,
        label="TDECQ",
        status_query=status_query,
        details_query=details_query,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        monotonic=monotonic,
        wait=wait,
        inspect_invalid_details=True,
        terminal_details=frozenset(
            {
                "target ser lower than actual",
                "the target ser is lower than actual ser of the signal.",
            }
        ),
        terminal_detail_prefixes=(
            "the measured noise margin is much smaller than the intrinsic "
            "channel noise. increase the signal's oma to improve the accuracy "
            "of the measurement result.",
        ),
    )
    wait_duration_s = max(0.0, monotonic() - wait_started)
    raw.update(polled_raw)
    if poll_error is not None:
        query_started = monotonic()
        _capture_metric_evidence(
            session,
            (
                source_query,
                status_query,
                details_query,
                taps_query,
                precursors_query,
            ),
            raw,
        )
        return finish(
            _invalid_after_poll(
                label="TDECQ",
                configured_source=configured_source,
                initially_verified_source=verified_source,
                source_query=source_query,
                details=details,
                invalid_reason=poll_error,
                raw=raw,
                polls=polls,
                status=polled_status,
            )
        )
    query_started = monotonic()
    try:
        final_source, final_source_raw = verify_tdecq_source(
            session, measurement_source=configured_source
        )
    except DcaReadError as error:
        raw.update(error.raw_responses)
        return finish(DcaMetricResult(
            configured_source=configured_source,
            verified_source=(
                error.current_source
                if isinstance(error, DcaSourceMismatchError)
                else _source_from_raw(
                    error.raw_responses,
                    source_query=source_query,
                    fallback=None,
                )
            ),
            value_db=None,
            status=(
                "SOURCE_MISMATCH"
                if isinstance(error, DcaSourceMismatchError)
                else "INV"
            ),
            details=str(error),
            valid=False,
            invalid_reason=f"TDECQ source validation failed: {error}",
            raw_responses=raw,
            status_polls=polls,
        ))
    raw.update(final_source_raw)
    try:
        snapshot = _read_tdecq_snapshot(
            session,
            function=function,
            canonical_source=configured_source,
            verified_source=final_source,
            source_responses=final_source_raw,
        )
    except DcaReadError as error:
        raw.update(error.raw_responses)
        return finish(DcaMetricResult(
            configured_source=configured_source,
            verified_source=final_source,
            value_db=None,
            status="INV",
            details=raw.get(details_query, details).strip(),
            valid=False,
            invalid_reason=f"TDECQ validation failed: {error}",
            raw_responses=raw,
            status_polls=polls,
        ))
    raw.update(snapshot.raw_responses)
    try:
        post_read_source, post_read_raw = verify_tdecq_source(
            session, measurement_source=configured_source
        )
    except DcaReadError as error:
        raw.update(error.raw_responses)
        raw_source = error.raw_responses.get(source_query)
        if raw_source is not None:
            raw[f"{source_query}#post_read"] = raw_source
        else:
            raw[f"{source_query}#post_read#error"] = str(error)
        return finish(DcaMetricResult(
            configured_source=configured_source,
            verified_source=(
                error.current_source
                if isinstance(error, DcaSourceMismatchError)
                else _source_from_raw(
                    error.raw_responses,
                    source_query=source_query,
                    fallback=None,
                )
            ),
            value_db=None,
            status=(
                "SOURCE_MISMATCH"
                if isinstance(error, DcaSourceMismatchError)
                else "INV"
            ),
            details=str(error),
            valid=False,
            invalid_reason=f"TDECQ post-read source validation failed: {error}",
            raw_responses=raw,
            status_polls=polls,
        ))
    raw.update(post_read_raw)
    raw[f"{source_query}#post_read"] = post_read_raw[source_query]
    return finish(DcaMetricResult(
        configured_source=configured_source,
        verified_source=post_read_source,
        value_db=snapshot.tdecq_db,
        status=snapshot.status,
        details=snapshot.details,
        valid=True,
        invalid_reason=None,
        raw_responses=raw,
        status_polls=polls,
    ))


def read_oer_then_tdecq(
    session: QuerySession,
    er_config: ErMeasurementConfig,
    *,
    function: int,
    tdecq_source: str,
    timeout_s: float,
    poll_interval_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
) -> DcaSequentialResult:
    """Complete OER, then TDECQ, without interleaving the shared selector."""
    if type(function) is not int or not 1 <= function <= 16:
        raise ValueError("function must be between 1 and 16")
    _normalize_tdecq_source(tdecq_source)
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be finite and positive")
    if (
        isinstance(poll_interval_s, bool)
        or not isinstance(poll_interval_s, (int, float))
        or not isfinite(float(poll_interval_s))
        or poll_interval_s <= 0
    ):
        raise ValueError("poll_interval_s must be finite and positive")
    er = _read_oer_after_acquisition(
        session,
        er_config,
        timeout_s=float(timeout_s),
        poll_interval_s=float(poll_interval_s),
        monotonic=monotonic,
        wait=wait,
    )
    tdecq = _read_tdecq_after_acquisition(
        session,
        function=function,
        source=tdecq_source,
        timeout_s=float(timeout_s),
        poll_interval_s=float(poll_interval_s),
        monotonic=monotonic,
        wait=wait,
    )
    return DcaSequentialResult(er=er, tdecq=tdecq)


@contextmanager
def open_visa_session(
    resource: str,
    timeout_ms: int = 60_000,
    *,
    open_timeout_ms: int | None = None,
    journal: object | None = None,
    operation_context: Callable[[], Mapping[str, object]] | None = None,
) -> Iterator[QuerySession]:
    """Open and reliably close a VISA query session."""
    pyvisa = importlib.import_module("pyvisa")
    from vcsel_ai_tuning.hardware.scpi_journal import (
        JournaledScpiSession,
        ScpiOperationRecorder,
    )

    context = operation_context or (lambda: {})
    recorder = (
        ScpiOperationRecorder(journal, context) if journal is not None else None
    )
    manager = (
        recorder.execute("resource_manager_open", None, pyvisa.ResourceManager)
        if recorder is not None
        else pyvisa.ResourceManager()
    )
    instrument = None
    try:
        if open_timeout_ms is None:
            instrument = (
                recorder.execute(
                    "open_resource", resource, manager.open_resource, resource
                )
                if recorder is not None
                else manager.open_resource(resource)
            )
        else:
            instrument = (
                recorder.execute(
                    "open_resource",
                    resource,
                    manager.open_resource,
                    resource,
                    open_timeout=open_timeout_ms,
                )
                if recorder is not None
                else manager.open_resource(resource, open_timeout=open_timeout_ms)
            )
        instrument.timeout = timeout_ms
        yield (
            JournaledScpiSession(instrument, journal, context)
            if journal is not None
            else instrument
        )
    finally:
        try:
            if instrument is not None:
                if recorder is not None:
                    recorder.execute("close", None, instrument.close)
                else:
                    instrument.close()
        finally:
            if recorder is not None:
                recorder.execute("resource_manager_close", None, manager.close)
            else:
                manager.close()
