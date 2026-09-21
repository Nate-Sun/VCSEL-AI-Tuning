"""Fail-closed, lazily connected V1.1 READ_ONLY hardware assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, ContextManager, NoReturn
from uuid import uuid4

from vcsel_ai_tuning.hardware.live_config import LiveHardwareConfig
from vcsel_ai_tuning.hardware.vd1124 import (
    Vd1124FieldValue,
    Vd1124ProbeResult,
    Vd1124ReadOnlyCapture,
)
if TYPE_CHECKING:
    from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration

from .baseline import B0Snapshot, DcaFingerprint, HardwareIdentity
from .domain import RuntimeMode, TaskType
from .errors import ReadOnlyPreflightError
from .regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from .service import PreflightCapture, PreflightEvidence


_DCA_QUERY_ALLOWLIST = frozenset(
    {
        "*IDN?",
        ":MEASure:RESults? STANdard",
        ":TRIGger:SOURce?",
        ":TRIGger:DCDRatio?",
        ":TRIGger:SRATe?",
        ":TRIGger:PLENgth?",
    }
)


@dataclass(frozen=True, slots=True)
class ReadOnlyDriverBundle:
    """Drivers opened for one bounded Preflight capture only."""

    dca: Any
    i2c: Any
    bert: Any
    connection_durations_s: Mapping[str, float] | None = None


ReadOnlyDriverFactory = Callable[
    [LiveHardwareConfig, Callable[[], float]],
    ContextManager[ReadOnlyDriverBundle],
]


def _canonical_source(value: str) -> str:
    if not isinstance(value, str):
        raise ReadOnlyPreflightError("DCA source response is not text")
    text = value.strip()
    patterns = (
        (
            r"([1-8])([A-D])",
            lambda match: f"CHAN{match.group(1)}{match.group(2).upper()}",
        ),
        (
            r"CHAN(?:NEL)?([1-8])([A-D])",
            lambda match: f"CHAN{match.group(1)}{match.group(2).upper()}",
        ),
        (
            r"FUNC(?:TION)?([0-9]+)",
            lambda match: f"FUNC{int(match.group(1), 10)}",
        ),
        (
            r"F([0-9]+)",
            lambda match: f"FUNC{int(match.group(1), 10)}",
        ),
    )
    for pattern, formatter in patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match is not None:
            return formatter(match)
    raise ReadOnlyPreflightError(
        f"DCA source response is not an approved waveform: {text or '<empty>'}"
    )


@dataclass(frozen=True, slots=True)
class DcaResultsRow:
    """One immutable row returned by ``MEASure:RESults? STANdard``."""

    name: str
    displayed_source: str
    current: str
    status: str
    unit: str | None
    minimum: str | None
    maximum: str | None
    mean: str | None
    stddev: str | None
    raw_fields: tuple[tuple[str, str], ...]

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "displayed_source": self.displayed_source,
            "current": self.current,
            "status": self.status,
            "unit": self.unit,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "stddev": self.stddev,
            "raw_fields": dict(self.raw_fields),
        }


def _parse_standard_results(response: str) -> tuple[DcaResultsRow, ...]:
    """Parse every FlexDCA standard Results row without selector state."""

    if not isinstance(response, str) or not response.strip():
        raise ReadOnlyPreflightError("DCA Results table is empty")
    chunks = re.split(r"(?i)(?=Name\s*=)", response.strip())
    rows: list[DcaResultsRow] = []
    for chunk in chunks:
        chunk = chunk.strip(" ,\r\n\t")
        if not chunk:
            continue
        fields: list[tuple[str, str]] = []
        keyed: dict[str, str] = {}
        for token in chunk.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ReadOnlyPreflightError(
                    f"DCA Results field is malformed: {token!r}"
                )
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            folded = key.casefold()
            if not key or folded in keyed:
                raise ReadOnlyPreflightError(
                    f"DCA Results row contains an invalid or duplicate field: {key!r}"
                )
            keyed[folded] = value
            fields.append((key, value))
        missing = [
            name
            for name in ("name", "source", "current", "status")
            if name not in keyed
        ]
        if missing:
            raise ReadOnlyPreflightError(
                "DCA Results row is missing required fields: " + ", ".join(missing)
            )
        unit = keyed.get("unit", keyed.get("units"))
        rows.append(
            DcaResultsRow(
                name=keyed["name"],
                displayed_source=keyed["source"],
                current=keyed["current"],
                status=keyed["status"],
                unit=unit,
                minimum=keyed.get("min"),
                maximum=keyed.get("max"),
                mean=keyed.get("mean"),
                stddev=keyed.get("stddev"),
                raw_fields=tuple(fields),
            )
        )
    if not rows:
        raise ReadOnlyPreflightError("DCA Results table contains no measurement rows")
    return tuple(rows)


def _normalize_results_status(value: str, label: str) -> str:
    statuses = {
        "correct": "CORR",
        "corr": "CORR",
        "invalid": "INV",
        "inv": "INV",
        "questionable": "QUES",
        "ques": "QUES",
    }
    normalized = statuses.get(value.strip().casefold())
    if normalized is None:
        raise ReadOnlyPreflightError(
            f"{label} status is unknown: {value or '<empty>'}"
        )
    return normalized


def _select_results_row(
    rows: Sequence[DcaResultsRow],
    *,
    name: str,
    source: str,
) -> DcaResultsRow:
    expected_source = _canonical_source(source)
    matches = []
    for row in rows:
        if row.name.strip().casefold() != name.casefold():
            continue
        try:
            row_source = _canonical_source(row.displayed_source)
        except ReadOnlyPreflightError:
            continue
        if row_source == expected_source:
            matches.append(row)
    if not matches:
        raise ReadOnlyPreflightError(
            f"DCA Results {name} source {expected_source} was not found"
        )
    if len(matches) != 1:
        raise ReadOnlyPreflightError(
            f"DCA Results contains multiple {name} rows with source {expected_source}"
        )
    return matches[0]


def _optional_result_number(value: str | None, label: str) -> float | None:
    if value is None or not value.strip():
        return None
    return _finite_result_number(value, label)


def _finite_result_number(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ReadOnlyPreflightError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed) or parsed >= 9.9e37:
        raise ReadOnlyPreflightError(
            f"{label} is invalid or a FlexDCA placeholder: {value!r}"
        )
    return parsed


def _query(
    dca: Any,
    command: str,
    label: str,
    *,
    allowlist: frozenset[str],
    raw_responses: dict[str, str],
    allow_blank: bool = False,
) -> str:
    is_query = command.endswith("?") or command == ":MEASure:RESults? STANdard"
    if not is_query or command not in allowlist:
        raise ReadOnlyPreflightError(
            f"{label} command is not an approved READ_ONLY DCA query"
        )
    try:
        response = dca.query(command)
    except Exception as error:
        raise ReadOnlyPreflightError(f"{label} query failed: {error}") from error
    if not isinstance(response, str):
        raise ReadOnlyPreflightError(f"{label} query did not return text")
    value = response.strip()
    raw_responses[command] = response
    if not value and not allow_blank:
        raise ReadOnlyPreflightError(f"{label} query returned an empty response")
    return value


def _finite_positive(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ReadOnlyPreflightError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed) or parsed >= 9.9e37 or parsed <= 0:
        raise ReadOnlyPreflightError(
            f"{label} is invalid or a FlexDCA placeholder: {value!r}"
        )
    return parsed


def _require_endpoint(target: Any, field: str, expected: object, label: str) -> None:
    missing = object()
    actual = getattr(target, field, missing)
    if actual != expected:
        rendered = "<missing>" if actual is missing else repr(actual)
        raise ReadOnlyPreflightError(
            f"{label} endpoint mismatch: expected {expected!r}, got {rendered}"
        )


def validate_profile_endpoints(
    profile: LiveHardwareConfig,
    *,
    dca_visa_resource: str,
    i2c_com_port: str,
    i2c_address: int,
    bert_host: str,
    bert_port: int,
) -> None:
    """Cross-check every endpoint duplicated in runtime and driver profile YAML."""

    comparisons = (
        ("DCA VISA resource", dca_visa_resource, profile.dca.resource),
        ("I2C COM port", i2c_com_port.upper(), profile.i2c.port.upper()),
        ("I2C address", i2c_address, profile.i2c.device_address),
        ("BERT host", bert_host, profile.bert.ip),
        ("BERT UDP port", bert_port, profile.bert.udp_port),
    )
    mismatches = [
        f"{label}: runtime={runtime_value!r}, profile={profile_value!r}"
        for label, runtime_value, profile_value in comparisons
        if runtime_value != profile_value
    ]
    if mismatches:
        raise ReadOnlyPreflightError(
            "runtime/profile endpoint mismatch: " + "; ".join(mismatches)
        )


def validate_read_only_profile(profile: LiveHardwareConfig) -> None:
    """Reject unsafe or internally inconsistent profile commands before I/O."""

    er = profile.characterization.er
    expected_queries = {
        "value_query": ":MEASure:EYE:OER?",
        "source_query": ":MEASure:EYE:OER:SOURce?",
        "status_query": ":MEASure:EYE:OER:STATus?",
        "details_query": ":MEASure:EYE:OER:STATus:DETails?",
        "units_query": ":MEASure:EYE:OER:UNITs?",
    }
    mismatches = [
        f"characterization.er.{field} must be {expected!r}"
        for field, expected in expected_queries.items()
        if getattr(er, field) != expected
    ]
    try:
        tdecq_source = _canonical_source(profile.characterization.tdecq.source)
    except ReadOnlyPreflightError as error:
        mismatches.append(
            f"characterization.tdecq.source is invalid: {error}"
        )
    else:
        if tdecq_source != profile.characterization.tdecq.source:
            mismatches.append(
                "characterization.tdecq.source must use canonical waveform syntax"
            )
        if not tdecq_source.startswith("FUNC"):
            mismatches.append(
                "characterization.tdecq.source must identify the displayed function"
            )
        elif tdecq_source != f"FUNC{profile.dca.function}":
            mismatches.append(
                "characterization.tdecq.source must match the configured "
                f"DCA equalizer function FUNC{profile.dca.function}"
            )
    try:
        tdecq_input_channel = _canonical_source(
            profile.characterization.tdecq.input_channel
        )
    except ReadOnlyPreflightError as error:
        mismatches.append(
            f"characterization.tdecq.input_channel is invalid: {error}"
        )
    else:
        if tdecq_input_channel != profile.characterization.tdecq.input_channel:
            mismatches.append(
                "characterization.tdecq.input_channel must use canonical waveform syntax"
            )
        if not tdecq_input_channel.startswith("CHAN"):
            mismatches.append(
                "characterization.tdecq.input_channel must identify an input channel"
            )
    if not 1 <= profile.dca.function <= 16:
        mismatches.append("DCA equalizer function must be between 1 and 16")
    if mismatches:
        raise ReadOnlyPreflightError(
            "unsafe READ_ONLY hardware profile: " + "; ".join(mismatches)
        )


@contextmanager
def open_read_only_drivers(
    profile: LiveHardwareConfig,
    remaining_s: Callable[[], float],
) -> Any:
    """Open real query/readback drivers only for an explicit Preflight call."""

    # These imports are deliberately inside the authorized operation. Importing
    # this module and constructing the server must not import optional live I/O
    # packages or construct VISA/serial/socket resources.
    from vcsel_ai_tuning.hardware.flexdca import open_visa_session
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124ReadOnlySession
    from vcsel_ai_tuning.hardware.vit_bert import VitBertClient

    i2c = profile.i2c
    durations: dict[str, float] = {}

    def remaining(label: str) -> float:
        value = float(remaining_s())
        if not math.isfinite(value) or value <= 0:
            raise ReadOnlyPreflightError(
                f"Preflight total timeout exceeded before {label}"
            )
        return value

    with ExitStack() as stack:
        timeout = remaining("DCA connection")
        timeout_ms = max(1, int(timeout * 1000.0))
        started = time.monotonic()
        dca_session = stack.enter_context(
            open_visa_session(
                profile.dca.resource,
                timeout_ms=min(
                    profile.dca.timeout_ms,
                    timeout_ms,
                ),
                open_timeout_ms=min(
                    profile.dca.timeout_ms,
                    timeout_ms,
                ),
            )
        )
        durations["dca"] = time.monotonic() - started
        timeout = remaining("I2C connection")
        started = time.monotonic()
        i2c_session = stack.enter_context(
            Vd1124ReadOnlySession(
                port=i2c.port,
                baudrate=i2c.baudrate,
                address=i2c.device_address,
                timeout_s=min(1.0, timeout),
            )
        )
        durations["i2c"] = time.monotonic() - started
        timeout = remaining("BERT connection")
        started = time.monotonic()
        bert = VitBertClient(
            profile.bert.ip,
            profile.bert.udp_port,
            local_ip=profile.bert.local_ip,
            timeout_s=min(profile.bert.timeout_s, timeout),
        )
        durations["bert"] = time.monotonic() - started
        remaining("Preflight queries")
        yield ReadOnlyDriverBundle(
            dca=dca_session,
            i2c=i2c_session,
            bert=bert,
            connection_durations_s=durations,
        )


class _CapturedDcaQuerySession:
    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        worker_pid: int,
        journal_path: str,
        timeout_ms: float,
    ) -> None:
        self._responses = dict(responses)
        self.timeout = float(timeout_ms)
        self.isolation_evidence = {
            "method": "spawn_watchdog",
            "parent_pid": os.getpid(),
            "worker_pid": worker_pid,
            "worker_is_separate": worker_pid != os.getpid(),
            "journal_path": journal_path,
        }

    def query(self, command: str) -> str:
        try:
            return self._responses[command]
        except KeyError as error:
            raise ReadOnlyPreflightError(
                f"isolated DCA capture did not contain approved query {command!r}"
            ) from error


@contextmanager
def open_isolated_read_only_drivers(
    profile: LiveHardwareConfig,
    remaining_s: Callable[[], float],
    *,
    artifact_root: str | Path,
    cleanup_grace_s: float,
    dca_client_factory: Any = None,
) -> Any:
    """Open I2C/BERT read-only ports while DCA I/O stays in a spawned worker."""

    from vcsel_ai_tuning.hardware.live_dca_isolation import (
        DcaIsolationError,
        IsolatedLiveDcaClient,
    )
    from vcsel_ai_tuning.hardware.vd1124 import Vd1124ReadOnlySession
    from vcsel_ai_tuning.hardware.vit_bert import VitBertClient

    def remaining(label: str) -> float:
        value = float(remaining_s())
        if not math.isfinite(value) or value <= 0:
            raise ReadOnlyPreflightError(
                f"Preflight total timeout exceeded before {label}"
            )
        return value

    durations: dict[str, float] = {}
    timeout = min(remaining("DCA connection"), profile.dca.timeout_ms / 1000.0)
    selected_factory = dca_client_factory or IsolatedLiveDcaClient
    started = time.monotonic()
    client = selected_factory(
        profile=profile,
        artifact_root=artifact_root,
        timeout_s=timeout,
        cleanup_grace_s=cleanup_grace_s,
    )
    client.set_operation_context(
        task_id="read-only-preflight",
        item_id=f"snapshot-{uuid4().hex}",
        round_index=0,
    )
    try:
        capture = client.read_only_queries()
    except DcaIsolationError as error:
        raise ReadOnlyPreflightError(
            "isolated READ_ONLY DCA capture failed",
            evidence={"dca_isolation": dict(error.evidence)},
        ) from error
    durations["dca"] = time.monotonic() - started
    dca_session = _CapturedDcaQuerySession(
        capture.responses,
        worker_pid=capture.worker_pid,
        journal_path=capture.journal_path,
        timeout_ms=timeout * 1000.0,
    )

    with ExitStack() as stack:
        i2c = profile.i2c
        timeout = remaining("I2C connection")
        started = time.monotonic()
        i2c_session = stack.enter_context(
            Vd1124ReadOnlySession(
                port=i2c.port,
                baudrate=i2c.baudrate,
                address=i2c.device_address,
                timeout_s=min(1.0, timeout),
            )
        )
        durations["i2c"] = time.monotonic() - started
        timeout = remaining("BERT connection")
        started = time.monotonic()
        bert = VitBertClient(
            profile.bert.ip,
            profile.bert.udp_port,
            local_ip=profile.bert.local_ip,
            timeout_s=min(profile.bert.timeout_s, timeout),
        )
        durations["bert"] = time.monotonic() - started
        remaining("Preflight queries")
        yield ReadOnlyDriverBundle(
            dca=dca_session,
            i2c=i2c_session,
            bert=bert,
            connection_durations_s=durations,
        )


class ReadOnlyV11Gateway:
    """Capture one attested B0 without exposing any hardware mutation API."""

    def __init__(
        self,
        profile: LiveHardwareConfig,
        *,
        total_timeout_s: float,
        driver_factory: ReadOnlyDriverFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(profile, LiveHardwareConfig):
            raise TypeError("profile must be LiveHardwareConfig")
        validate_read_only_profile(profile)
        if (
            isinstance(total_timeout_s, bool)
            or not isinstance(total_timeout_s, (int, float))
            or not math.isfinite(float(total_timeout_s))
            or float(total_timeout_s) <= 0
        ):
            raise ValueError("total_timeout_s must be a finite positive number")
        self._profile = profile
        self._total_timeout_s = min(
            float(total_timeout_s),
            float(profile.characterization.measurement_timeout_s),
        )
        self._driver_factory = driver_factory or open_read_only_drivers
        self._monotonic = monotonic
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._capture_count = 0

    @property
    def runtime_mode(self) -> RuntimeMode:
        return RuntimeMode.READ_ONLY

    @property
    def total_timeout_s(self) -> float:
        return self._total_timeout_s

    def health(self) -> Mapping[str, object]:
        return {"status": "ok", "hardware_connected": False}

    def capabilities(self) -> Mapping[str, object]:
        return {
            "campaign_types": ["AI_OPTIMIZATION", "SWEEP"],
            "dynamic_b0": True,
            "full_campaign_authorization": False,
            "hardware_connected": False,
            "hardware_mutation": False,
            "fake": False,
        }

    def _capture_dca(
        self,
        dca: Any,
        raw_responses: dict[str, str],
        remaining_s: Callable[[], float],
    ) -> tuple[DcaFingerprint, str, str, dict[str, Any]]:
        profile = self._profile
        allowlist = _DCA_QUERY_ALLOWLIST
        def query(
            command: str,
            label: str,
            *,
            current_allowlist: frozenset[str] | None = None,
            allow_blank: bool = False,
        ) -> str:
            remaining = remaining_s()
            if remaining <= 0:
                raise ReadOnlyPreflightError(
                    f"Preflight total timeout exceeded before {label}"
                )
            if hasattr(dca, "timeout"):
                current_timeout = float(getattr(dca, "timeout"))
                dca.timeout = min(current_timeout, remaining * 1000.0)
            result = _query(
                dca,
                command,
                label,
                allowlist=current_allowlist or allowlist,
                raw_responses=raw_responses,
                allow_blank=allow_blank,
            )
            if remaining_s() <= 0:
                raise ReadOnlyPreflightError(
                    f"Preflight total timeout exceeded during {label}"
                )
            return result

        idn = query(
            "*IDN?",
            "DCA identity",
        )
        if not idn.startswith(profile.dca.expected_idn_prefix):
            raise ReadOnlyPreflightError(
                "DCA identity does not match the reviewed expected IDN prefix"
            )

        results_response = query(
            ":MEASure:RESults? STANdard",
            "DCA standard Results table",
        )
        rows = _parse_standard_results(results_response)
        rows_payload = [row.payload() for row in rows]
        try:
            outer_er_row = _select_results_row(
                rows,
                name="Outer ER",
                source=profile.characterization.er.source,
            )
            tdecq_row = _select_results_row(
                rows,
                name="TDECQ",
                source=profile.characterization.tdecq.source,
            )
        except ReadOnlyPreflightError as error:
            error.evidence = {"dca_results_rows": rows_payload}
            raise

        def normalize_measurement(
            row: DcaResultsRow,
            label: str,
        ) -> tuple[str, float, str, dict[str, float]]:
            try:
                source = _canonical_source(row.displayed_source)
                status = _normalize_results_status(row.status, label)
                value = _finite_result_number(row.current, f"{label} Current")
                if row.unit is not None and row.unit.strip().upper() not in {
                    "DB",
                    "DEC",
                    "DECIBEL",
                }:
                    raise ReadOnlyPreflightError(
                        f"{label} units must be decibels, got {row.unit}"
                    )
                statistics = {
                    name: parsed
                    for name, parsed in (
                        ("min", _optional_result_number(row.minimum, f"{label} Min")),
                        ("max", _optional_result_number(row.maximum, f"{label} Max")),
                        ("mean", _optional_result_number(row.mean, f"{label} Mean")),
                        (
                            "stddev",
                            _optional_result_number(row.stddev, f"{label} StdDev"),
                        ),
                    )
                    if parsed is not None
                }
            except ReadOnlyPreflightError as error:
                error.evidence = {
                    "dca_measurement": row.payload(),
                    "dca_results_rows": rows_payload,
                }
                raise
            measurement = {
                "name": row.name,
                "displayed_source": source,
                "value": value if status == "CORR" else None,
                "raw_current": row.current,
                "status": status,
                "details": row.status,
                "unit": row.unit,
                "statistics": statistics,
                "raw_fields": dict(row.raw_fields),
            }
            if status != "CORR":
                raise ReadOnlyPreflightError(
                    f"{label} status is {status}, not CORR: {row.status}",
                    evidence={
                        "dca_measurement": measurement,
                        "dca_results_rows": rows_payload,
                    },
                )
            return source, value, status, statistics

        outer_er_source, outer_er_value, outer_er_status, outer_er_statistics = (
            normalize_measurement(outer_er_row, "Outer ER")
        )
        tdecq_source, tdecq_value, tdecq_status, tdecq_statistics = (
            normalize_measurement(tdecq_row, "TDECQ")
        )

        if not re.fullmatch(r"CHAN[1-8][A-D]", outer_er_source):
            raise ReadOnlyPreflightError("Outer ER source must identify an optical channel")
        function_source = f"FUNC{profile.dca.function}"
        operand_query = f":FUNCtion{profile.dca.function}:OPERand?"
        allowlist = frozenset((*allowlist, operand_query))
        tdecq_input_channel = _canonical_source(
            query(
                operand_query,
                f"{function_source} operand",
                current_allowlist=allowlist,
            )
        )
        expected_input_channel = profile.characterization.tdecq.input_channel
        if tdecq_input_channel != expected_input_channel:
            raise ReadOnlyPreflightError(
                f"{function_source} operand mismatch: expected "
                f"{expected_input_channel}, got {tdecq_input_channel}"
            )

        wavelength_query = f":{outer_er_source}:WAVelength:VALue?"
        allowlist = frozenset((*allowlist, wavelength_query))
        wavelength_m = _finite_positive(
            query(
                wavelength_query,
                "DCA wavelength",
                current_allowlist=allowlist,
            ),
            "DCA wavelength",
        )
        outer_er_details = outer_er_row.status
        tdecq_details = tdecq_row.status

        trigger_source = query(
            ":TRIGger:SOURce?",
            "DCA trigger source",
            current_allowlist=allowlist,
        )
        trigger_ratio = query(
            ":TRIGger:DCDRatio?",
            "DCA trigger ratio",
            current_allowlist=allowlist,
        )
        symbol_rate_baud = _finite_positive(
            query(
                ":TRIGger:SRATe?",
                "DCA symbol rate",
                current_allowlist=allowlist,
            ),
            "DCA symbol rate",
        )
        pattern_length_value = _finite_positive(
            query(
                ":TRIGger:PLENgth?",
                "DCA pattern length",
                current_allowlist=allowlist,
            ),
            "DCA pattern length",
        )
        if not pattern_length_value.is_integer():
            raise ReadOnlyPreflightError("DCA pattern length must be an integer")
        pattern_length = int(pattern_length_value)

        fingerprint = DcaFingerprint.from_mappings(
            identity=idn,
            measurement_sources={
                "outer_er": outer_er_source,
                "tdecq": tdecq_source,
            },
            configuration={
                "function": str(profile.dca.function),
                "tdecq_equalizer_function": function_source,
                "tdecq_input_channel": tdecq_input_channel,
                "channel": outer_er_source,
                "wavelength_m": format(wavelength_m, ".12g"),
                "trigger_source": trigger_source,
                "trigger_ratio": trigger_ratio,
                "symbol_rate_baud": format(symbol_rate_baud, ".12g"),
                "pattern_length": str(pattern_length),
                "outer_er_details": outer_er_details,
                "outer_er_status": outer_er_status,
                "outer_er_units": "DECIBEL",
                "outer_er_value_db": format(outer_er_value, ".12g"),
                "tdecq_details": tdecq_details,
                "tdecq_status": tdecq_status,
                "tdecq_value_db": format(tdecq_value, ".12g"),
            },
        )
        normalized = {
            "identity": idn,
            "function": profile.dca.function,
            "channel": outer_er_source,
            "wavelength_m": wavelength_m,
            "trigger_source": trigger_source,
            "trigger_ratio": trigger_ratio,
            "symbol_rate_baud": symbol_rate_baud,
            "pattern_length": pattern_length,
            "results_rows": rows_payload,
            "outer_er": {
                "source": outer_er_source,
                "units": "DECIBEL",
                "status": outer_er_status,
                "details": outer_er_details,
                "value": outer_er_value,
                "reported_unit": outer_er_row.unit,
                "statistics": outer_er_statistics,
            },
            "tdecq": {
                "displayed_source": tdecq_source,
                "function": function_source,
                "input_channel": tdecq_input_channel,
                "value": tdecq_value,
                "status": tdecq_status,
                "details": tdecq_details,
                "unit": tdecq_row.unit,
                "statistics": tdecq_statistics,
            },
        }
        return fingerprint, idn, outer_er_source, normalized

    def _capture_i2c(
        self,
        reader: Any,
        on_capture: Callable[[Vd1124ReadOnlyCapture], None] | None = None,
    ) -> Vd1124ReadOnlyCapture:
        i2c = self._profile.i2c
        _require_endpoint(reader, "port", i2c.port, "I2C reader")
        _require_endpoint(reader, "baudrate", i2c.baudrate, "I2C reader")
        _require_endpoint(reader, "address", i2c.device_address, "I2C reader")
        fields = self._profile.characterization.parameters
        channel = self._profile.characterization.default_channel
        try:
            capture = reader.capture_identity_and_fields(
                fields,
                channel=channel,
                chip_id_register=i2c.chip_id_register,
                expected_chip_id=i2c.expected_chip_id,
                chip_revision_register=i2c.chip_revision_register,
                expected_chip_revision=i2c.expected_chip_revision,
            )
        except Exception as error:
            audit = getattr(error, "audit", None)
            audit_payload = {}
            if audit is not None:
                audit_payload = {
                    "entry_page": getattr(audit, "entry_page", None),
                    "restore_target": getattr(audit, "restore_target", None),
                    "restore_readback": getattr(audit, "restore_readback", None),
                    "restore_verified": getattr(audit, "restore_verified", False),
                    "page_select_writes": len(
                        getattr(audit, "page_selects", ())
                    ),
                    "page_selects": [
                        {
                            "page": getattr(event, "page", None),
                            "reason": getattr(event, "reason", None),
                            "frame_hex": getattr(event, "frame", b"").hex(),
                            "ack_hex": getattr(
                                event, "acknowledgement", b""
                            ).hex(),
                        }
                        for event in getattr(audit, "page_selects", ())
                    ],
                    "tunable_register_writes": getattr(
                        audit, "tunable_register_writes", None
                    ),
                }
            raise ReadOnlyPreflightError(
                f"I2C identity/field read failed: {error}",
                evidence={"i2c_page_restore": audit_payload},
            ) from error
        if not isinstance(capture, Vd1124ReadOnlyCapture):
            raise ReadOnlyPreflightError("I2C capture has an unexpected type")
        if on_capture is not None:
            on_capture(capture)
        identity = capture.identity
        if (
            not isinstance(identity, Vd1124ProbeResult)
            or identity.matched is not True
            or identity.port.upper() != i2c.port.upper()
            or identity.chip_id != i2c.expected_chip_id
            or identity.chip_revision != i2c.expected_chip_revision
        ):
            raise ReadOnlyPreflightError(
                "I2C identity does not match the reviewed profile"
            )
        if capture.audit.restore_verified is not True:
            raise ReadOnlyPreflightError(
                "I2C page restoration was not verified"
            )
        if capture.audit.tunable_register_writes != 0:
            raise ReadOnlyPreflightError(
                "READ_ONLY I2C capture reported a tunable-register write"
            )
        values = capture.fields
        if not isinstance(values, Sequence) or len(values) != len(fields):
            raise ReadOnlyPreflightError("I2C field readback is incomplete")
        result = tuple(values)
        for field, value in zip(fields, result, strict=True):
            expected_address = field.address_for_channel(channel)
            if (
                not isinstance(value, Vd1124FieldValue)
                or value.name != field.name
                or value.channel != channel
                or value.address != expected_address
                or not 0 <= value.raw_value <= 0xFF
                or not 0 <= value.value <= field.encoded_max
                or ((value.raw_value >> field.bit_start) & field.encoded_max)
                != value.value
            ):
                raise ReadOnlyPreflightError(
                    f"I2C readback for {field.name} is malformed"
                )
        return Vd1124ReadOnlyCapture(
            identity=identity,
            fields=result,
            audit=capture.audit,
        )

    def _capture_bert(
        self, bert: Any, remaining_s: Callable[[], float]
    ) -> tuple[str, BertConfiguration]:
        # Importing the UDP implementation is deliberately deferred until an
        # explicit Preflight has acquired hardware authority. Runtime assembly,
        # health and capabilities remain connection- and driver-import-free.
        from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration
        config = self._profile.bert
        _require_endpoint(bert, "host", config.ip, "BERT")
        _require_endpoint(bert, "port", config.udp_port, "BERT")
        _require_endpoint(bert, "local_ip", config.local_ip, "BERT")
        try:
            remaining = remaining_s()
            if remaining <= 0:
                raise ReadOnlyPreflightError(
                    "Preflight total timeout exceeded before BERT mode"
                )
            if hasattr(bert, "timeout_s"):
                bert.timeout_s = min(float(bert.timeout_s), remaining)
            mode = bert.get_mode()
            remaining = remaining_s()
            if remaining <= 0:
                raise ReadOnlyPreflightError(
                    "Preflight total timeout exceeded during BERT mode"
                )
            if hasattr(bert, "timeout_s"):
                bert.timeout_s = min(float(bert.timeout_s), remaining)
            state = bert.get_configuration(config.port, config.channel)
            if remaining_s() <= 0:
                raise ReadOnlyPreflightError(
                    "Preflight total timeout exceeded during BERT configuration"
                )
        except Exception as error:
            if isinstance(error, ReadOnlyPreflightError):
                raise
            raise ReadOnlyPreflightError(f"BERT readback failed: {error}") from error
        raw_mode = mode.strip() if isinstance(mode, str) else ""
        mode_token = raw_mode.split(maxsplit=1)[0].upper() if raw_mode else ""
        if mode_token != "PAM4":
            raise ReadOnlyPreflightError(
                f"BERT mode must be PAM4, got {mode!r}"
            )
        if (
            not isinstance(state, BertConfiguration)
            or state.port != config.port
            or state.channel != config.channel
            or not state.raw_response.strip()
        ):
            raise ReadOnlyPreflightError(
                "BERT target/readback does not match the profile"
            )
        if state.pattern_fields != ("SSPRQ", "SSPRQ"):
            raise ReadOnlyPreflightError(
                "BERT must attest SSPRQ in both pattern fields"
            )
        return raw_mode, state

    def preflight_capture(self) -> PreflightCapture:
        profile = self._profile
        started_at = self._utcnow()
        started = self._monotonic()
        deadline = started + self._total_timeout_s
        subsystem_durations: dict[str, dict[str, float]] = {
            name: {"connection_s": 0.0, "query_s": 0.0}
            for name in ("dca", "i2c", "bert")
        }
        raw_responses: dict[str, Any] = {
            "dca": {},
            "i2c": {},
            "bert": {},
        }
        normalized: dict[str, Any] = {}
        mutation_counts = {
            "dca_writes": 0,
            "i2c_page_select_writes": 0,
            "i2c_tunable_register_writes": 0,
            "bert_writes": 0,
        }

        def check_deadline(label: str) -> None:
            if self._monotonic() >= deadline:
                raise ReadOnlyPreflightError(
                    f"Preflight total timeout exceeded during {label}"
                )

        def evidence_payload(*, finished_at: datetime) -> dict[str, Any]:
            return {
                "fake": False,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "total_duration_s": max(0.0, self._monotonic() - started),
                "timeout_s": self._total_timeout_s,
                "targets": {
                    "dca_visa": profile.dca.resource,
                    "i2c": {
                        "com": profile.i2c.port,
                        "baudrate": profile.i2c.baudrate,
                        "address": f"0x{profile.i2c.device_address:02X}",
                    },
                    "bert": {
                        "host": profile.bert.ip,
                        "udp_port": profile.bert.udp_port,
                        "local_ip": profile.bert.local_ip,
                        "mode": "PAM4",
                        "port": profile.bert.port,
                        "channel": profile.bert.channel,
                    },
                },
                "durations_s": subsystem_durations,
                "raw_responses": raw_responses,
                "normalized": normalized,
                "mutation_counts": mutation_counts,
            }

        def record_i2c_capture(capture: Vd1124ReadOnlyCapture) -> None:
            audit = capture.audit
            mutation_counts["i2c_page_select_writes"] = len(
                audit.page_selects
            )
            mutation_counts["i2c_tunable_register_writes"] = (
                audit.tunable_register_writes
            )
            register_bytes = {
                f"0x{profile.i2c.chip_id_register:04X}": (
                    f"0x{capture.identity.chip_id:02X}"
                ),
                f"0x{profile.i2c.chip_revision_register:04X}": (
                    f"0x{capture.identity.chip_revision:02X}"
                ),
            }
            register_bytes.update(
                {
                    f"0x{value.address:04X}": f"0x{value.raw_value:02X}"
                    for value in capture.fields
                    if isinstance(value, Vd1124FieldValue)
                }
            )
            raw_responses["i2c"] = {
                "register_bytes": register_bytes,
                "page_selects": [
                    {
                        "page": f"0x{event.page:02X}",
                        "reason": event.reason,
                        "frame_hex": event.frame.hex(),
                        "ack_hex": event.acknowledgement.hex(),
                    }
                    for event in audit.page_selects
                ],
                "entry_page": f"0x{audit.entry_page:02X}",
                "restore_target": f"0x{audit.restore_target:02X}",
                "restore_readback": (
                    None
                    if audit.restore_readback is None
                    else f"0x{audit.restore_readback:02X}"
                ),
                "restore_verified": audit.restore_verified,
            }

        try:
            check_deadline("driver connection")
            with self._driver_factory(
                profile, lambda: deadline - self._monotonic()
            ) as drivers:
                for name, duration in dict(
                    drivers.connection_durations_s or {}
                ).items():
                    if name in subsystem_durations:
                        subsystem_durations[name]["connection_s"] = max(
                            0.0, float(duration)
                        )
                check_deadline("driver connection")

                query_started = self._monotonic()
                fingerprint, dca_idn, outer_er_source, dca_normalized = (
                    self._capture_dca(
                        drivers.dca,
                        raw_responses["dca"],
                        lambda: deadline - self._monotonic(),
                    )
                )
                subsystem_durations["dca"]["query_s"] = max(
                    0.0, self._monotonic() - query_started
                )
                normalized["dca"] = dca_normalized
                isolation_evidence = getattr(
                    drivers.dca, "isolation_evidence", None
                )
                if isinstance(isolation_evidence, Mapping):
                    raw_responses["dca_isolation"] = dict(isolation_evidence)
                    if isolation_evidence.get("worker_is_separate") is not True:
                        raise ReadOnlyPreflightError(
                            "READ_ONLY DCA worker was not isolated from the parent"
                        )
                check_deadline("DCA queries")

                query_started = self._monotonic()
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise ReadOnlyPreflightError(
                        "Preflight total timeout exceeded before I2C queries"
                    )
                timeout_setter = getattr(drivers.i2c, "set_timeout_s", None)
                if callable(timeout_setter):
                    timeout_setter(remaining)
                i2c_capture = self._capture_i2c(
                    drivers.i2c, on_capture=record_i2c_capture
                )
                subsystem_durations["i2c"]["query_s"] = max(
                    0.0, self._monotonic() - query_started
                )
                i2c_identity = i2c_capture.identity
                i2c_values = i2c_capture.fields
                normalized["i2c"] = {
                    "chip_id": i2c_identity.chip_id,
                    "chip_revision": i2c_identity.chip_revision,
                    "fields": {
                        value.name: value.value for value in i2c_values
                    },
                }
                check_deadline("I2C queries")

                query_started = self._monotonic()
                bert_mode, bert_state = self._capture_bert(
                    drivers.bert,
                    lambda: deadline - self._monotonic(),
                )
                subsystem_durations["bert"]["query_s"] = max(
                    0.0, self._monotonic() - query_started
                )
                raw_responses["bert"] = {
                    "mode": bert_mode,
                    "configuration": bert_state.raw_response,
                }
                normalized["bert"] = {
                    "mode": bert_mode,
                    "pattern_fields": list(bert_state.pattern_fields),
                    "fir_taps": list(bert_state.taps),
                    "upper_nlc": bert_state.upper_nlc,
                    "lower_nlc": bert_state.lower_nlc,
                    "port": bert_state.port,
                    "channel": bert_state.channel,
                }
                check_deadline("BERT queries")
            check_deadline("driver cleanup")
        except Exception as error:
            finished_at = self._utcnow()
            if isinstance(error, ReadOnlyPreflightError):
                restore = error.evidence.get("i2c_page_restore")
                if isinstance(restore, Mapping):
                    mutation_counts["i2c_page_select_writes"] = int(
                        restore.get("page_select_writes") or 0
                    )
                    mutation_counts["i2c_tunable_register_writes"] = int(
                        restore.get("tunable_register_writes") or 0
                    )
            partial = evidence_payload(finished_at=finished_at)
            if isinstance(error, ReadOnlyPreflightError):
                if error.evidence:
                    partial["failure_context"] = error.evidence
                error.evidence = partial
                raise
            raise ReadOnlyPreflightError(
                f"READ_ONLY Preflight failed: {error}", evidence=partial
            ) from error

        try:
            check_deadline("Preflight finalization")
            fields = profile.characterization.parameters
            script_entries = tuple(
                ScriptEntry(value.name, value.value, order)
                for order, value in enumerate(i2c_values)
            )
            regmap_fields = tuple(
                RegMapField(
                    name=field.name,
                    channel0_address=field.channel0_address,
                    channel_stride=0x100,
                    bit_start=field.bit_start,
                    bits=field.bits,
                    encoding_min=0,
                    encoding_max=field.encoded_max,
                )
                for field in fields
            )
            mapping = scan_parameter_mapping(script_entries, regmap_fields)
            check_deadline("parameter mapping finalization")
            self._capture_count += 1
            capture_number = self._capture_count
            snapshot = B0Snapshot.capture(
                b0_id=f"read-only-b0-{capture_number:08d}",
                version=1,
                captured_at=self._utcnow(),
                i2c_fields={value.name: value.value for value in i2c_values},
                i2c_raw_registers={
                    value.address: value.raw_value for value in i2c_values
                },
                fir_taps=bert_state.taps,
                upper_nlc=bert_state.upper_nlc,
                lower_nlc=bert_state.lower_nlc,
                bert_pattern="SSPRQ",
                i2c_channel=profile.characterization.default_channel,
                bert_port=bert_state.port,
                bert_channel=bert_state.channel,
                dca_tdecq_source=profile.characterization.tdecq.source,
                dca_tdecq_equalizer_function=profile.dca.function,
                dca_outer_er_source=outer_er_source,
                dca_fingerprint=fingerprint,
                hardware_identities={
                    "dca": HardwareIdentity(
                        "dca", profile.dca.resource, dca_idn
                    ),
                    "i2c": HardwareIdentity(
                        "i2c",
                        f"{profile.i2c.port}/{profile.i2c.baudrate}/"
                        f"0x{profile.i2c.device_address:02X}",
                        f"chip_id=0x{i2c_identity.chip_id:02X};"
                        f"revision=0x{i2c_identity.chip_revision:02X}",
                    ),
                    "bert": HardwareIdentity(
                        "bert",
                        f"{profile.bert.ip}:{profile.bert.udp_port}/"
                        f"{bert_state.port}/{bert_state.channel}",
                        f"mode={bert_mode};patterns="
                        f"{bert_state.pattern_fields[0]},"
                        f"{bert_state.pattern_fields[1]}",
                    ),
                },
            )
            check_deadline("B0 finalization")
            finished_at = self._utcnow()
            evidence = PreflightEvidence.from_payload(
                evidence_payload(finished_at=finished_at)
            )
            check_deadline("evidence finalization")
            return PreflightCapture(
                snapshot_token=(
                    f"read-only-snapshot-{capture_number:08d}-"
                    f"{snapshot.content_hash[:12]}"
                ),
                b0=snapshot,
                mapping=mapping,
                evidence=evidence,
            )
        except Exception as error:
            finished_at = self._utcnow()
            partial = evidence_payload(finished_at=finished_at)
            if isinstance(error, ReadOnlyPreflightError):
                error.evidence = partial
                raise
            raise ReadOnlyPreflightError(
                f"READ_ONLY Preflight finalization failed: {error}",
                evidence=partial,
            ) from error

    def current_attestation(self) -> NoReturn:
        raise ReadOnlyPreflightError(
            "READ_ONLY hardware queries are available only through explicit "
            "Preflight"
        )


class ReadOnlyV11TaskRunner:
    """Runner that rejects every execution path before hardware access."""

    @staticmethod
    def _reject() -> None:
        # Deferred to avoid importing the broader runtime module at server import
        # solely for an exception that can only be raised on an explicit action.
        from vcsel_ai_tuning.runtime import ReadOnlyModeError

        raise ReadOnlyModeError("READ_ONLY mode rejects V1.1 hardware execution")

    def start(self, _task_id: str, _task_type: TaskType) -> None:
        self._reject()

    def cancel_prepared(self, _task_id: str) -> None:
        self._reject()

    def pause_and_restore_b0(self, _task_id: str) -> None:
        self._reject()

    def resume_from_b0(self, _task_id: str) -> None:
        self._reject()

    def abort_and_restore_b0(self, _task_id: str) -> None:
        self._reject()

    def recover_b0(self, _task_id: str) -> None:
        self._reject()

    def close(self) -> None:
        return None
