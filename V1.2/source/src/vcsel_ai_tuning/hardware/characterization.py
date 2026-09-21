"""Auditable pre-tuning hardware characterization workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
from math import floor, isclose, isfinite
import os
from pathlib import Path
from statistics import fmean, median, stdev
import time
from typing import Any, Callable

from vcsel_ai_tuning.hardware.flexdca import (
    DcaMetricResult,
    DcaSequentialResult,
    QuerySession,
    _normalize_measurement_source,
    read_oer_then_tdecq,
)
from vcsel_ai_tuning.hardware.live_config import (
    ErMeasurementConfig,
    LiveHardwareConfig,
    ParameterFieldConfig,
)
from vcsel_ai_tuning.hardware.vd1124 import (
    Vd1124FieldValue,
    Vd1124ProbeResult,
)
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


SCHEMA_VERSION = "2.0"


class CharacterizationError(RuntimeError):
    """Raised when a characterization safety or audit invariant fails."""


RoundArtifactPersist = Callable[
    [Mapping[str, object], Mapping[str, object] | None, int],
    dict[str, object],
]


_ROUND_PHASE_FIELDS = (
    "apply_duration_s",
    "readback_duration_s",
    "actual_settle_s",
    "acquisition_duration_s",
    "wait_for_valid_duration_s",
    "query_duration_s",
)


def _phase_sum_duration(row: Mapping[str, object]) -> float:
    total = 0.0
    for field in _ROUND_PHASE_FIELDS:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidate = float(value)
            if isfinite(candidate) and candidate >= 0.0:
                total += candidate
    return total


@dataclass(frozen=True, slots=True)
class CampaignMeasurementResult:
    """Duck-compatible campaign result without importing the orchestrator."""

    acquisition_valid: bool
    er_db: float | None
    er_status: str
    tdecq_db: float | None
    tdecq_status: str
    tdecq_invalid_reason: str | None
    elapsed_s: float
    raw_responses: Mapping[str, object]
    artifacts: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class _DcaCycleResult:
    measurements: DcaSequentialResult
    acquisition_duration_s: float
    wait_for_valid_duration_s: float
    query_duration_s: float
    cleared_er_status: str
    cleared_tdecq_status: str


class CampaignDcaAdapter:
    """Lazy bridge from an already-open FlexDCA session to campaign calls."""

    def __init__(
        self,
        *,
        session: QuerySession,
        config: LiveHardwareConfig,
        monotonic: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session = session
        self._config = config
        self._monotonic = monotonic
        self._wait = wait

    def preflight(self) -> Mapping[str, object]:
        """Read identity and the current measurement context without writes."""
        dca = self._config.dca
        tdecq = self._config.characterization.tdecq
        er = self._config.characterization.er
        idn = _safe_query(self._session, "*IDN?").strip()
        tdecq_source = _safe_query(
            self._session, ":MEASure:EYE:TDEQ:SOURce?"
        ).strip().upper()
        er_source = _safe_query(self._session, er.source_query).strip().upper()
        er_units = _safe_query(self._session, er.units_query).strip().upper()
        expected_tdecq_source = tdecq.source.strip().upper()
        expected_er_source = er.source.strip().upper()
        independent_sources_verified = (
            tdecq_source == expected_tdecq_source
            and er_source == expected_er_source
        )
        shared_source_verified = (
            tdecq_source == er_source
            and tdecq_source in {expected_tdecq_source, expected_er_source}
        )
        return {
            "identity_verified": idn.startswith(dca.expected_idn_prefix),
            "configuration_verified": (
                (independent_sources_verified or shared_source_verified)
                and er_units in {"DB", "DEC", "DECIBEL"}
            ),
            "idn": idn,
            "tdecq_source": tdecq_source,
            "er_source": er_source,
            "er_units": er_units,
            "raw_responses": {
                "*IDN?": idn,
                ":MEASure:EYE:TDEQ:SOURce?": tdecq_source,
                er.source_query: er_source,
                er.units_query: er_units,
            },
        }

    def measure_single(self) -> CampaignMeasurementResult:
        """Run exactly one fresh Stop/Clear/Single acquisition cycle."""
        started = self._monotonic()
        tdecq = self._config.characterization.tdecq
        er = self._config.characterization.er
        cycle = _measure_dca_once(
            self._session,
            er,
            function=self._config.dca.function,
            tdecq_source=tdecq.source,
            timeout_s=self._config.characterization.measurement_timeout_s,
            poll_interval_s=self._config.characterization.poll_interval_s,
            monotonic=self._monotonic,
            wait=self._wait,
        )
        er_result = cycle.measurements.er
        tdecq_result = cycle.measurements.tdecq
        er_document = _metric_raw_document(er_result)
        tdecq_document = _metric_raw_document(tdecq_result)
        status_polls = [
            {
                "er_status": er_result.status,
                "tdecq_status": str(poll.get("status", "INV")),
            }
            for poll in tdecq_result.status_polls
        ]
        raw: dict[str, object] = {
            "cleared_er_status": cycle.cleared_er_status,
            "cleared_tdecq_status": cycle.cleared_tdecq_status,
            "status_polls": status_polls,
            "tdecq_details_polls": [
                str(poll["details"])
                for poll in tdecq_result.status_polls
                if "details" in poll
            ],
            "tdecq_source_selection": {
                "requested_source": tdecq_result.configured_source,
                "verified_source": tdecq_result.verified_source,
                "raw_responses": dict(tdecq_result.raw_responses),
            },
            "er": er_document,
            "tdecq": tdecq_document,
            "measurements": {
                "outer_er": er_document,
                "tdecq": tdecq_document,
            },
            "timing": {
                "apply_duration_s": 0.0,
                "readback_duration_s": 0.0,
                "acquisition_duration_s": cycle.acquisition_duration_s,
                "wait_for_valid_duration_s": cycle.wait_for_valid_duration_s,
                "query_duration_s": cycle.query_duration_s,
                "artifact_duration_s": 0.0,
                "total_duration_s": max(0.0, self._monotonic() - started),
            },
        }
        raw["acquisition_guard"] = _acquisition_guard(
            self._session,
            er_result,
            tdecq_result,
        )
        mismatch = next(
            (
                result
                for result in (er_result, tdecq_result)
                if result.status == "SOURCE_MISMATCH"
            ),
            None,
        )
        if mismatch is not None:
            raw["source_validation_error"] = {
                "message": mismatch.details,
                "raw_responses": dict(mismatch.raw_responses),
            }
        return CampaignMeasurementResult(
            acquisition_valid=er_result.valid or tdecq_result.valid,
            er_db=er_result.value_db,
            er_status=er_result.status,
            tdecq_db=tdecq_result.value_db,
            tdecq_status=tdecq_result.status,
            tdecq_invalid_reason=tdecq_result.invalid_reason,
            elapsed_s=max(0.0, self._monotonic() - started),
            raw_responses=raw,
        )


class CampaignI2cAdapter:
    """Lazy VD1124 campaign bridge with exact stale/readback guards."""

    def __init__(
        self,
        *,
        writer: object,
        identity_probe: object,
        config: LiveHardwareConfig,
    ) -> None:
        self._writer = writer
        self._identity_probe = identity_probe
        self._config = config
        self._channel = config.characterization.default_channel
        self._fields = config.characterization.parameters
        self._by_name = {field.name: field for field in self._fields}

    @staticmethod
    def _optional_attribute_matches(
        target: object, name: str, expected: object
    ) -> bool:
        actual = getattr(target, name, expected)
        return actual == expected

    def preflight(self) -> Mapping[str, object]:
        identity = self._identity_probe.probe()
        readback = self.readback()
        i2c = self._config.i2c
        identity_verified = bool(
            identity.matched
            and identity.port.upper() == i2c.port.upper()
            and identity.chip_id == i2c.expected_chip_id
            and identity.chip_revision == i2c.expected_chip_revision
        )
        configuration_verified = bool(
            len(readback) == len(self._fields)
            and self._optional_attribute_matches(
                self._writer, "port", i2c.port
            )
            and self._optional_attribute_matches(
                self._writer, "baudrate", i2c.baudrate
            )
            and self._optional_attribute_matches(
                self._writer, "address", i2c.device_address
            )
        )
        return {
            "identity_verified": identity_verified,
            "configuration_verified": configuration_verified,
            "port": identity.port,
            "chip_id": identity.chip_id,
            "chip_revision": identity.chip_revision,
            "channel": self._channel,
            "readback": readback,
        }

    def _read_fields(
        self, fields: tuple[ParameterFieldConfig, ...]
    ) -> tuple[Vd1124FieldValue, ...]:
        try:
            values = self._writer.read_fields(fields, channel=self._channel)
        except Exception as error:
            raise CharacterizationError(
                f"I2C parameter readback failed: {error}"
            ) from error
        if not isinstance(values, Sequence) or len(values) != len(fields):
            raise CharacterizationError("I2C parameter readback is incomplete")
        result = tuple(values)
        for field, value in zip(fields, result, strict=True):
            if (
                not isinstance(value, Vd1124FieldValue)
                or value.name != field.name
                or value.channel != self._channel
                or value.address != field.address_for_channel(self._channel)
                or not 0 <= value.raw_value <= 0xFF
                or not 0 <= value.value <= field.encoded_max
                or ((value.raw_value >> field.bit_start) & field.encoded_max)
                != value.value
            ):
                raise CharacterizationError(
                    f"I2C readback for {field.name} is malformed"
                )
        return result

    def readback(self) -> Mapping[str, int]:
        return {
            value.name: value.value
            for value in self._read_fields(self._fields)
        }

    def apply(self, requested: Mapping[str, int]) -> Mapping[str, int]:
        if not isinstance(requested, Mapping):
            raise CharacterizationError("I2C requested settings must be an object")
        unknown = sorted(set(requested) - set(self._by_name))
        if unknown:
            raise CharacterizationError(
                "unknown I2C parameter(s): " + ", ".join(unknown)
            )
        for name, value in requested.items():
            field = self._by_name[name]
            if type(value) is not int or not 0 <= value <= field.encoded_max:
                raise CharacterizationError(
                    f"{name} value is outside its encoding range"
                )

        applied: dict[str, int] = {}
        for field in self._fields:
            if field.name not in requested:
                continue
            before = self._read_fields((field,))[0]
            target = requested[field.name]
            try:
                result = self._writer.write_field(
                    field,
                    channel=self._channel,
                    value=target,
                    expected_current=before.value,
                    expected_raw=before.raw_value,
                )
            except Exception as error:
                raise CharacterizationError(
                    f"I2C write/readback failed for {field.name}: {error}"
                ) from error
            mask = field.encoded_max << field.bit_start
            expected_after_raw = (
                before.raw_value & ~mask
            ) | (target << field.bit_start)
            if (
                result.name != field.name
                or result.channel != self._channel
                or result.address != before.address
                or result.before_raw != before.raw_value
                or result.before_value != before.value
                or result.requested_value != target
                or result.after_raw != expected_after_raw
                or result.after_value != target
                or result.verified is not True
            ):
                raise CharacterizationError(
                    f"I2C write result for {field.name} is inconsistent"
                )
            applied[field.name] = result.after_value
        return applied


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _plan_hash(document: Mapping[str, object]) -> str:
    payload = {key: value for key, value in document.items() if key != "plan_id"}
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def rehash_plan(document: Mapping[str, object]) -> dict[str, object]:
    """Return a detached plan document with a newly calculated plan ID."""
    try:
        result = json.loads(_canonical_json(document))
    except (TypeError, ValueError) as error:
        raise CharacterizationError(f"plan cannot be hashed: {error}") from error
    if not isinstance(result, dict):
        raise CharacterizationError("characterization plan root must be an object")
    result.pop("plan_id", None)
    result["plan_id"] = _plan_hash(result)
    return result


def _parameter_document(config: LiveHardwareConfig) -> list[dict[str, object]]:
    return [
        {
            "name": field.name,
            "channel0_address": field.channel0_address,
            "bit_start": field.bit_start,
            "bits": field.bits,
            "safety_min": field.safety_min,
            "safety_max": field.safety_max,
            "max_step": field.max_step,
        }
        for field in config.characterization.parameters
    ]


def create_baseline_plan(
    config: LiveHardwareConfig,
    *,
    repeats: int = 5,
    created_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Create an offline immutable baseline acquisition plan."""
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    created = created_at_utc or datetime.now(timezone.utc)
    if created.tzinfo is None:
        raise ValueError("created_at_utc must be timezone-aware")
    er = config.characterization.er
    tdecq = config.characterization.tdecq
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "baseline",
        "created_at_utc": created.astimezone(timezone.utc).isoformat(),
        "hardware": {
            "dca": {
                "resource": config.dca.resource,
                "expected_idn_prefix": config.dca.expected_idn_prefix,
                "function": config.dca.function,
                "tdecq_source": tdecq.source,
                "timeout_ms": config.dca.timeout_ms,
            },
            "bert": {
                "ip": config.bert.ip,
                "udp_port": config.bert.udp_port,
                "port": config.bert.port,
                "channel": config.bert.channel,
            },
            "i2c": {
                "port": config.i2c.port,
                "baudrate": config.i2c.baudrate,
                "device_address": config.i2c.device_address,
                "expected_chip_id": config.i2c.expected_chip_id,
                "expected_chip_revision": config.i2c.expected_chip_revision,
            },
            "er": {
                "source": er.source,
                "value_query": er.value_query,
                "source_query": er.source_query,
                "status_query": er.status_query,
                "details_query": er.details_query,
                "units_query": er.units_query,
            },
            "i2c_parameters": _parameter_document(config),
        },
        "parameters": {
            "channel": config.characterization.default_channel,
            "repeats": repeats,
            "measurement_timeout_s": config.characterization.measurement_timeout_s,
            "poll_interval_s": config.characterization.poll_interval_s,
            "meaningful_er_floor_db": (
                config.characterization.meaningful_er_floor_db
            ),
            "meaningful_tdecq_floor_db": (
                config.characterization.meaningful_tdecq_floor_db
            ),
        },
        "safety": {
            "dca_mutations": [
                ":ACQuire:STOP",
                ":ACQuire:CDISplay",
                ":ACQuire:SINGle",
                f":MEASure:EYE:OER:SOURce {er.source}",
                f":MEASure:EYE:TDEQ:SOURce {tdecq.source}",
            ],
            "i2c_write_allowed": False,
            "bert_write_allowed": False,
        },
    }
    plan["plan_id"] = _plan_hash(plan)
    return plan


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CharacterizationError(f"{label} must be an object")
    return value


def _require_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CharacterizationError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0):
        raise CharacterizationError(f"{label} must be finite and positive")
    return result


def _utc_text(now_utc: Callable[[], datetime]) -> str:
    value = now_utc()
    if value.tzinfo is None:
        raise CharacterizationError("now_utc must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat()


def _bert_document(configuration: BertConfiguration) -> dict[str, object]:
    return {
        "port": configuration.port,
        "channel": configuration.channel,
        "fir_taps": list(configuration.taps),
        "upper_nlc": configuration.upper_nlc,
        "lower_nlc": configuration.lower_nlc,
        "raw_response": configuration.raw_response,
    }


def _field_document(values: Sequence[Vd1124FieldValue]) -> dict[str, object]:
    return {
        value.name: {
            "channel": value.channel,
            "address": value.address,
            "raw_value": value.raw_value,
            "value": value.value,
        }
        for value in values
    }


def _parameter_configs(plan: Mapping[str, object]) -> tuple[ParameterFieldConfig, ...]:
    hardware = _require_mapping(plan.get("hardware"), "plan hardware")
    raw_parameters = hardware.get("i2c_parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise CharacterizationError("plan i2c_parameters must be a non-empty list")
    fields: list[ParameterFieldConfig] = []
    try:
        for raw in raw_parameters:
            item = _require_mapping(raw, "I2C parameter")
            fields.append(
                ParameterFieldConfig(
                    name=str(item["name"]),
                    channel0_address=int(item["channel0_address"]),
                    bit_start=int(item["bit_start"]),
                    bits=int(item["bits"]),
                    safety_min=(
                        None if item.get("safety_min") is None else int(item["safety_min"])
                    ),
                    safety_max=(
                        None if item.get("safety_max") is None else int(item["safety_max"])
                    ),
                    max_step=(
                        None if item.get("max_step") is None else int(item["max_step"])
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise CharacterizationError(f"invalid I2C parameter definition: {error}") from error
    return tuple(fields)


def _verify_identity(
    identity: Vd1124ProbeResult, i2c_plan: Mapping[str, object]
) -> None:
    if not identity.matched:
        raise CharacterizationError("I2C identity probe did not match")
    expected_port = str(i2c_plan.get("port"))
    expected_id = i2c_plan.get("expected_chip_id")
    expected_revision = i2c_plan.get("expected_chip_revision")
    if (
        identity.port.upper() != expected_port.upper()
        or identity.chip_id != expected_id
        or identity.chip_revision != expected_revision
    ):
        raise CharacterizationError("I2C identity does not match the approved plan")


def _safe_query(session: QuerySession, command: str) -> str:
    try:
        value = session.query(command)
    except Exception as error:
        raise CharacterizationError(f"DCA query {command!r} failed: {error}") from error
    if not isinstance(value, str):
        raise CharacterizationError(f"DCA query {command!r} did not return text")
    return value


def _safe_write(session: QuerySession, command: str) -> None:
    try:
        session.write(command)
    except Exception as error:
        raise CharacterizationError(f"DCA command {command!r} failed: {error}") from error


def _acquisition_guard(
    session: QuerySession,
    er_result: DcaMetricResult,
    tdecq_result: DcaMetricResult,
) -> dict[str, object]:
    """Best-effort source-bound snapshot used to bind later eye-data queries."""

    if not er_result.valid or not tdecq_result.valid:
        return {
            "commands": {},
            "errors": {
                "primary": (
                    "both primary TDECQ and Outer ER must be valid before "
                    "acquisition consistency queries"
                )
            },
            "original_source": None,
            "original_source_raw": None,
            "source_restore": {
                "readback": None,
                "verified": False,
            },
            "primary_match": False,
        }

    values: dict[str, str] = {}
    errors: dict[str, str] = {}
    original_source: str | None = None
    original_source_raw: str | None = None
    restored_source: str | None = None
    restore_write_succeeded = False

    original_source_query = ":MEASure:EYE:TDEQ:SOURce?"
    try:
        value = session.query(original_source_query)
        if not isinstance(value, str):
            raise TypeError("query did not return text")
        original_source_raw = value
        original_source = _normalize_measurement_source(value)
    except Exception as error:
        errors["original_source"] = str(error)

    try:
        if original_source is not None:
            for stem, source in (
                (":MEASure:EYE:OER", er_result.configured_source),
                (":MEASure:EYE:TDEQ", tdecq_result.configured_source),
            ):
                select_command = f"{stem}:SOURce {source}"
                try:
                    session.write(select_command)
                except Exception as error:
                    errors[select_command] = str(error)
                for suffix in (":SOURce?", ":STATus?", ":COUNt?", "?"):
                    command = f"{stem}{suffix}"
                    try:
                        value = session.query(command)
                        if not isinstance(value, str):
                            raise TypeError("query did not return text")
                        values[command] = value.strip()
                    except Exception as error:
                        errors[command] = str(error)
    finally:
        if original_source is not None:
            restore_command = (
                f":MEASure:EYE:TDEQ:SOURce {original_source}"
            )
            try:
                session.write(restore_command)
                restore_write_succeeded = True
            except Exception as error:
                errors[f"restore:{restore_command}"] = str(error)
            try:
                value = session.query(original_source_query)
                if not isinstance(value, str):
                    raise TypeError("query did not return text")
                restored_source = _normalize_measurement_source(value)
            except Exception as error:
                errors[f"restore:{original_source_query}"] = str(error)

    restoration_verified = bool(
        restore_write_succeeded
        and original_source is not None
        and restored_source is not None
        and restored_source == original_source
    )
    if original_source is not None and not restoration_verified:
        errors.setdefault(
            "source_restore",
            (
                f"source restore readback {restored_source!r} does not match "
                f"original source {original_source!r}"
            ),
        )

    def numeric_matches(command: str, expected: float | None) -> bool:
        if expected is None:
            return False
        try:
            actual = float(values[command])
        except (KeyError, TypeError, ValueError):
            return False
        return isfinite(actual) and isclose(
            actual,
            float(expected),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

    def positive_count(command: str) -> bool:
        try:
            count = float(values[command])
        except (KeyError, TypeError, ValueError):
            return False
        return isfinite(count) and count >= 1

    primary_match = bool(
        not errors
        and restoration_verified
        and values.get(":MEASure:EYE:TDEQ:SOURce?", "").upper()
        == str(tdecq_result.verified_source).upper()
        and values.get(":MEASure:EYE:TDEQ:STATus?", "").upper()
        == str(tdecq_result.status).upper()
        and numeric_matches(":MEASure:EYE:TDEQ?", tdecq_result.value_db)
        and positive_count(":MEASure:EYE:TDEQ:COUNt?")
        and values.get(":MEASure:EYE:OER:SOURce?", "").upper()
        == str(er_result.verified_source).upper()
        and values.get(":MEASure:EYE:OER:STATus?", "").upper()
        == str(er_result.status).upper()
        and numeric_matches(":MEASure:EYE:OER?", er_result.value_db)
        and positive_count(":MEASure:EYE:OER:COUNt?")
    )
    return {
        "commands": values,
        "errors": errors,
        "original_source": original_source,
        "original_source_raw": original_source_raw,
        "source_restore": {
            "readback": restored_source,
            "verified": restoration_verified,
        },
        "primary_match": primary_match,
    }


def _metric_raw_document(metric: DcaMetricResult) -> dict[str, object]:
    raw_responses = dict(metric.raw_responses)
    taps = next(
        (
            value
            for command, value in raw_responses.items()
            if command.endswith(":TEQualizer:TAPS?")
        ),
        None,
    )
    precursors = next(
        (
            value
            for command, value in raw_responses.items()
            if command.endswith(":TEQualizer:NPRecursors?")
        ),
        None,
    )
    document: dict[str, object] = dict(raw_responses)
    document.update(
        {
            "requested_source": metric.configured_source,
            "verified_source": metric.verified_source,
            "status": metric.status,
            "value": metric.value_db,
            "details": metric.details,
            "invalid_reason": metric.invalid_reason,
            "valid": metric.valid,
            "raw_responses": raw_responses,
            "status_polls": [dict(poll) for poll in metric.status_polls],
            "equalizer_taps": taps,
            "equalizer_precursors": precursors,
        }
    )
    return document


def _measure_dca_once(
    session: QuerySession,
    er_config: ErMeasurementConfig,
    *,
    function: int,
    tdecq_source: str,
    timeout_s: float,
    poll_interval_s: float,
    monotonic: Callable[[], float],
    wait: Callable[[float], None],
) -> _DcaCycleResult:
    """Run one acquisition, then complete OER before selecting TDECQ."""
    acquisition_duration_s = 0.0
    for command in (":ACQuire:STOP", ":ACQuire:CDISplay"):
        command_start = monotonic()
        _safe_write(session, command)
        acquisition_duration_s += max(0.0, monotonic() - command_start)
    cleared_er_status = "NOT_QUERIED"
    cleared_tdecq_status = "NOT_QUERIED"
    command_start = monotonic()
    _safe_write(session, ":ACQuire:SINGle")
    acquisition_duration_s += max(0.0, monotonic() - command_start)
    measurements = read_oer_then_tdecq(
        session,
        er_config,
        function=function,
        tdecq_source=tdecq_source,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        monotonic=monotonic,
        wait=wait,
    )
    return _DcaCycleResult(
        measurements=measurements,
        acquisition_duration_s=acquisition_duration_s,
        wait_for_valid_duration_s=sum(
            metric.wait_for_valid_duration_s
            for metric in (measurements.er, measurements.tdecq)
        ),
        query_duration_s=sum(
            metric.query_duration_s
            for metric in (measurements.er, measurements.tdecq)
        ),
        cleared_er_status=cleared_er_status,
        cleared_tdecq_status=cleared_tdecq_status,
    )


def execute_baseline_plan(
    plan: Mapping[str, object],
    dca_session: QuerySession,
    bert_client: object,
    i2c_reader: object,
    i2c_identity: Vd1124ProbeResult,
    *,
    allow_execute: bool,
    confirmation_id: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wait: Callable[[float], None] = time.sleep,
    now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    persist_round: RoundArtifactPersist | None = None,
) -> dict[str, object]:
    """Execute a source-bound read-only baseline acquisition plan."""
    validate_characterization_plan(plan, expected_kind="baseline")
    if allow_execute is not True:
        raise CharacterizationError("baseline execution requires allow_execute=True")
    if not isinstance(confirmation_id, str) or not hmac.compare_digest(
        confirmation_id, str(plan["plan_id"])
    ):
        raise CharacterizationError("confirmation ID does not match the baseline plan")

    hardware = _require_mapping(plan["hardware"], "plan hardware")
    parameters = _require_mapping(plan["parameters"], "plan parameters")
    dca_plan = _require_mapping(hardware.get("dca"), "plan DCA")
    bert_plan = _require_mapping(hardware.get("bert"), "plan BERT")
    i2c_plan = _require_mapping(hardware.get("i2c"), "plan I2C")
    er_plan = _require_mapping(hardware.get("er"), "plan ER")
    _verify_identity(i2c_identity, i2c_plan)

    repeats = parameters.get("repeats")
    channel = parameters.get("channel")
    if type(repeats) is not int or repeats <= 0:
        raise CharacterizationError("baseline repeats must be a positive integer")
    if type(channel) is not int or not 0 <= channel <= 3:
        raise CharacterizationError("baseline channel must be from 0 through 3")
    timeout_s = _require_number(
        parameters.get("measurement_timeout_s"),
        "measurement_timeout_s",
        positive=True,
    )
    poll_interval_s = _require_number(
        parameters.get("poll_interval_s"), "poll_interval_s", positive=True
    )
    function = dca_plan.get("function")
    source = dca_plan.get("tdecq_source")
    if type(function) is not int or not isinstance(source, str):
        raise CharacterizationError("baseline DCA function/source are invalid")
    if not 1 <= function <= 16 or not source.strip():
        raise CharacterizationError(
            "baseline TDECQ function/source must be explicitly valid"
        )
    er_config = ErMeasurementConfig(
        source=str(er_plan.get("source")),
        value_query=str(er_plan.get("value_query")),
        source_query=str(er_plan.get("source_query")),
        status_query=str(er_plan.get("status_query")),
        details_query=str(er_plan.get("details_query")),
        units_query=str(er_plan.get("units_query")),
    )
    if er_config.source.strip().upper() != "CHAN1A":
        raise CharacterizationError("baseline Outer ER source must be CHAN1A")
    fields = _parameter_configs(plan)

    bert_port = str(bert_plan.get("port"))
    bert_channel = str(bert_plan.get("channel"))
    try:
        initial_bert = bert_client.get_configuration(bert_port, bert_channel)
        initial_fields = i2c_reader.read_fields(fields, channel=channel)
    except Exception as error:
        raise CharacterizationError(f"initial read-only snapshot failed: {error}") from error
    initial_snapshot = {
        "timestamp_utc": _utc_text(now_utc),
        "dca_idn": _safe_query(dca_session, "*IDN?").strip(),
        "i2c_identity": {
            "port": i2c_identity.port,
            "chip_id": i2c_identity.chip_id,
            "chip_revision": i2c_identity.chip_revision,
            "matched": i2c_identity.matched,
        },
        "bert": _bert_document(initial_bert),
        "i2c_parameters": _field_document(initial_fields),
    }
    expected_idn_prefix = dca_plan.get("expected_idn_prefix")
    if (
        not isinstance(expected_idn_prefix, str)
        or not str(initial_snapshot["dca_idn"]).startswith(expected_idn_prefix)
    ):
        raise CharacterizationError(
            "DCA identity does not match the approved expected IDN prefix"
        )
    if (
        initial_bert.port != bert_port
        or initial_bert.channel != bert_channel
        or not initial_bert.raw_response.strip()
    ):
        raise CharacterizationError(
            "BERT configuration identity/target does not match the approved plan"
        )
    initial_bert_state = _bert_document(initial_bert)
    initial_field_state = _field_document(initial_fields)

    verified_source = _safe_query(
        dca_session, ":MEASure:EYE:TDEQ:SOURce?"
    ).strip().upper()

    run_id = str(plan["plan_id"])[:12]
    events: list[dict[str, object]] = [
        {"event": "baseline_started", "timestamp_utc": _utc_text(now_utc)}
    ]
    measurements: list[dict[str, object]] = []
    raw_responses: dict[str, object] = {
        "initial": {
            "dca_idn": initial_snapshot["dca_idn"],
            "tdecq_source": verified_source,
        }
    }

    for repeat_index in range(1, repeats + 1):
        total_start = monotonic_ns()
        started_at = _utc_text(now_utc)
        cycle = _measure_dca_once(
            dca_session,
            er_config,
            function=function,
            tdecq_source=source,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            monotonic=lambda: monotonic_ns() / 1e9,
            wait=wait,
        )
        acquisition_duration_s = cycle.acquisition_duration_s
        wait_duration_s = cycle.wait_for_valid_duration_s
        query_duration_s = cycle.query_duration_s
        er_result = cycle.measurements.er
        tdecq_result = cycle.measurements.tdecq
        er_document = _metric_raw_document(er_result)
        tdecq_document = _metric_raw_document(tdecq_result)
        if repeat_index == 1:
            raw_responses["initial"].update(
                {
                    "er_source": er_result.verified_source,
                    "tdecq_source": tdecq_result.verified_source,
                }
            )
        er_db = er_result.value_db
        tdecq_db = tdecq_result.value_db
        er_status = er_result.status
        tdecq_status = tdecq_result.status
        er_details = er_result.details
        tdecq_details = tdecq_result.details
        invalid_reasons = [
            f"{label}: {result.invalid_reason}"
            for label, result in (("OER", er_result), ("TDECQ", tdecq_result))
            if not result.valid
        ]
        invalid_reason = "; ".join(invalid_reasons) or None
        repeat_raw: dict[str, object] = {
            "cleared_er_status": cycle.cleared_er_status,
            "cleared_tdecq_status": cycle.cleared_tdecq_status,
            "status_polls": [
                {
                    "er_status": er_result.status,
                    "tdecq_status": str(poll.get("status", "INV")),
                }
                for poll in tdecq_result.status_polls
            ],
            "er": er_document,
            "tdecq": tdecq_document,
            "measurements": {
                "outer_er": er_document,
                "tdecq": tdecq_document,
            },
        }
        readback_start = monotonic_ns()
        try:
            round_bert = bert_client.get_configuration(bert_port, bert_channel)
            round_fields = i2c_reader.read_fields(fields, channel=channel)
        except Exception as error:
            raise CharacterizationError(f"round readback failed: {error}") from error
        readback_duration_s = max(
            0.0, (monotonic_ns() - readback_start) / 1e9
        )
        round_bert_state = _bert_document(round_bert)
        round_field_state = _field_document(round_fields)
        stable = (
            round_bert_state == initial_bert_state
            and round_field_state == initial_field_state
        )
        if not stable:
            invalid_reason = "BERT or I2C state changed during read-only baseline"
            er_db = None
            tdecq_db = None
        finished_at = _utc_text(now_utc)
        total_duration_s = (monotonic_ns() - total_start) / 1e9
        field_values = {value.name: value.value for value in round_fields}
        observation_valid = invalid_reason is None and er_db is not None and tdecq_db is not None
        row: dict[str, object] = {
            "run_id": run_id,
            "stage": "baseline",
            "repeat_index": repeat_index,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "apply_duration_s": 0.0,
            "readback_duration_s": readback_duration_s,
            "acquisition_duration_s": acquisition_duration_s,
            "wait_for_valid_duration_s": wait_duration_s,
            "query_duration_s": query_duration_s,
            "artifact_duration_s": 0.0,
            "total_duration_s": total_duration_s,
            "requested_settle_s": None,
            "actual_settle_s": None,
            "er_db": er_db,
            "er_status": er_status,
            "er_details": er_details,
            "er_source": er_result.verified_source,
            "er_invalid_reason": er_result.invalid_reason,
            "tdecq_db": tdecq_db,
            "tdecq_status": tdecq_status,
            "tdecq_details": tdecq_details,
            "tdecq_source": tdecq_result.verified_source,
            "tdecq_invalid_reason": tdecq_result.invalid_reason,
            "observation_valid": observation_valid,
            "invalid_reason": invalid_reason,
            "bert_fir_taps": list(round_bert.taps),
            "dca_measurement_source": source,
            **field_values,
        }
        repeat_raw["bert"] = round_bert_state
        repeat_raw["i2c_parameters"] = round_field_state
        raw_responses[f"repeat_{repeat_index}"] = repeat_raw
        row["phase_sum_duration_s"] = _phase_sum_duration(row)
        row["candidate_round_time_s"] = total_duration_s
        if persist_round is not None:
            row = persist_round(row, repeat_raw, total_start)
        else:
            row["_round_started_monotonic_ns"] = total_start
        measurements.append(row)
        events.append(
            {
                "event": "baseline_repeat_completed",
                "repeat_index": repeat_index,
                "valid": observation_valid,
                "timestamp_utc": finished_at,
            }
        )

    final_bert = bert_client.get_configuration(bert_port, bert_channel)
    # The final field state is the state already read after the final round. This
    # preserves exactly one parameter readback per completed acquisition.
    final_snapshot = {
        "timestamp_utc": _utc_text(now_utc),
        "dca_idn": initial_snapshot["dca_idn"],
        "i2c_identity": initial_snapshot["i2c_identity"],
        "bert": _bert_document(final_bert),
        "i2c_parameters": round_field_state,
    }
    if final_snapshot["bert"] != initial_bert_state:
        raise CharacterizationError("BERT state changed during read-only baseline")
    summary = summarize_measurements(measurements)
    convergence = build_convergence_defaults(
        summary,
        er_floor_db=parameters.get("meaningful_er_floor_db"),
        tdecq_floor_db=parameters.get("meaningful_tdecq_floor_db"),
        repeats=repeats,
    )
    events.append({"event": "baseline_completed", "timestamp_utc": _utc_text(now_utc)})
    valid_count = summary["valid_count"]
    report = (
        "# Baseline Characterization / 基线表征\n\n"
        f"- Status / 状态: COMPLETED\n"
        f"- Valid observations / 有效观测: {valid_count}/{repeats}\n"
        "- Hardware writes / 硬件写入: DCA acquisition control only; "
        "no BERT or I2C writes / 仅 DCA 采集控制；无 BERT 或 I2C 写入\n"
    )
    return {
        "status": "COMPLETED",
        "run_id": run_id,
        "plan": dict(plan),
        "events": events,
        "measurements": measurements,
        "raw_responses": raw_responses,
        "initial_snapshot": initial_snapshot,
        "final_snapshot": final_snapshot,
        "summary": summary,
        "convergence_defaults": convergence,
        "report_markdown": report,
    }


def _read_parameter_snapshot(
    reader: object,
    fields: tuple[ParameterFieldConfig, ...],
    channel: int,
) -> tuple[Vd1124FieldValue, ...]:
    try:
        values = reader.read_fields(fields, channel=channel)
    except Exception as error:
        raise CharacterizationError(f"I2C parameter readback failed: {error}") from error
    if not isinstance(values, Sequence) or len(values) != len(fields):
        raise CharacterizationError("I2C parameter readback is incomplete")
    return tuple(values)


def _operation_document(
    field: ParameterFieldConfig,
    value: Vd1124FieldValue,
    *,
    new_value: int,
) -> dict[str, object]:
    return {
        "name": field.name,
        "channel": value.channel,
        "address": value.address,
        "bit_start": field.bit_start,
        "bits": field.bits,
        "old_raw": value.raw_value,
        "old_value": value.value,
        "new_value": new_value,
        "safety_min": field.safety_min,
        "safety_max": field.safety_max,
        "max_step": field.max_step,
    }


def _config_plan_base(
    config: LiveHardwareConfig,
    *,
    kind: str,
    channel: int,
    created_at_utc: datetime | None,
) -> dict[str, object]:
    baseline = create_baseline_plan(
        config, repeats=config.characterization.baseline_repeats,
        created_at_utc=created_at_utc,
    )
    baseline.pop("plan_id", None)
    baseline["kind"] = kind
    parameters = _require_mapping(baseline["parameters"], "plan parameters")
    # This is a plain dict produced locally by create_baseline_plan.
    parameters["channel"] = channel  # type: ignore[index]
    return baseline


def create_same_value_plan(
    config: LiveHardwareConfig,
    i2c_reader: object,
    i2c_identity: Vd1124ProbeResult,
    *,
    channel: int,
    created_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Read the nine fields and create an immutable same-value write plan."""
    if type(channel) is not int or not 0 <= channel <= 3:
        raise ValueError("channel must be an integer from 0 through 3")
    base = _config_plan_base(
        config, kind="same-value", channel=channel,
        created_at_utc=created_at_utc,
    )
    hardware = _require_mapping(base["hardware"], "plan hardware")
    _verify_identity(
        i2c_identity, _require_mapping(hardware["i2c"], "plan I2C")
    )
    fields = config.characterization.parameters
    values = _read_parameter_snapshot(i2c_reader, fields, channel)
    base["operations"] = [
        _operation_document(field, value, new_value=value.value)
        for field, value in zip(fields, values, strict=True)
    ]
    base["creation_snapshot"] = {
        "i2c_identity": {
            "port": i2c_identity.port,
            "chip_id": i2c_identity.chip_id,
            "chip_revision": i2c_identity.chip_revision,
        },
        "i2c_parameters": _field_document(values),
    }
    base["safety"] = {
        "i2c_write_allowed": "same-value-only",
        "bert_write_allowed": False,
        "requires_apply": True,
        "prohibited_register_classes": [
            "disable", "mute", "power", "OTP", "fault-mask"
        ],
    }
    return rehash_plan(base)


def _validate_apply(
    plan: Mapping[str, object],
    *,
    expected_kind: str,
    allow_apply: bool,
    confirmation_id: str,
) -> None:
    validate_characterization_plan(plan, expected_kind=expected_kind)
    if allow_apply is not True:
        raise CharacterizationError(f"{expected_kind} execution requires --apply")
    if not isinstance(confirmation_id, str) or not hmac.compare_digest(
        confirmation_id, str(plan["plan_id"])
    ):
        raise CharacterizationError("confirmation ID does not match the plan")


def _preflight_operations(
    plan: Mapping[str, object], writer: object, identity: Vd1124ProbeResult
) -> tuple[tuple[ParameterFieldConfig, ...], int, tuple[Vd1124FieldValue, ...]]:
    hardware = _require_mapping(plan["hardware"], "plan hardware")
    _verify_identity(identity, _require_mapping(hardware["i2c"], "plan I2C"))
    parameters = _require_mapping(plan["parameters"], "plan parameters")
    channel = parameters.get("channel")
    if type(channel) is not int or not 0 <= channel <= 3:
        raise CharacterizationError("plan channel is invalid")
    fields = _parameter_configs(plan)
    current = _read_parameter_snapshot(writer, fields, channel)
    creation = _require_mapping(
        plan.get("creation_snapshot"), "plan creation_snapshot"
    )
    created_fields = _require_mapping(
        creation.get("i2c_parameters"), "creation I2C parameters"
    )
    for value in current:
        created = _require_mapping(
            created_fields.get(value.name), f"creation field {value.name}"
        )
        if (
            created.get("value") != value.value
            or created.get("raw_value") != value.raw_value
            or created.get("address") != value.address
            or created.get("channel") != value.channel
        ):
            raise CharacterizationError(
                f"stale I2C state for {value.name}: creation snapshot no longer matches"
            )
    operations = plan.get("operations")
    if not isinstance(operations, list):
        raise CharacterizationError("plan operations must be a list")
    expected = {str(item.get("name")): item for item in operations if isinstance(item, Mapping)}
    for value in current:
        operation = expected.get(value.name)
        if operation is not None and (
            operation.get("old_value") != value.value
            or operation.get("old_raw") != value.raw_value
            or operation.get("address") != value.address
        ):
            raise CharacterizationError(
                f"stale I2C state for {value.name}: plan no longer matches readback"
            )
    return fields, channel, current


def execute_same_value_plan(
    plan: Mapping[str, object],
    i2c_writer: object,
    i2c_identity: Vd1124ProbeResult,
    *,
    allow_apply: bool,
    confirmation_id: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, object]:
    """Execute only exact old-value-to-same-value field writes."""
    _validate_apply(
        plan, expected_kind="same-value", allow_apply=allow_apply,
        confirmation_id=confirmation_id,
    )
    try:
        fields, channel, _ = _preflight_operations(plan, i2c_writer, i2c_identity)
    except CharacterizationError as error:
        return {
            "status": "STALE_REJECTED",
            "error": str(error),
            "write_results": [],
        }
    operations = plan.get("operations")
    assert isinstance(operations, list)
    if len(operations) != len(fields):
        raise CharacterizationError("same-value plan must contain all nine fields")
    by_name = {field.name: field for field in fields}
    operation_names = [
        str(operation.get("name"))
        for operation in operations
        if isinstance(operation, Mapping)
    ]
    if len(operation_names) != len(operations) or set(operation_names) != set(by_name):
        raise CharacterizationError(
            "same-value plan must contain every configured field exactly once"
        )
    validated_operations: list[tuple[Mapping[str, object], ParameterFieldConfig]] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise CharacterizationError("same-value operation must be an object")
        name = str(operation.get("name"))
        field = by_name[name]
        old_value = operation.get("old_value")
        old_raw = operation.get("old_raw")
        if (
            type(old_value) is not int
            or operation.get("new_value") != old_value
            or type(old_raw) is not int
            or not 0 <= old_raw <= 0xFF
            or operation.get("channel") != channel
            or operation.get("address") != field.address_for_channel(channel)
            or operation.get("bit_start") != field.bit_start
            or operation.get("bits") != field.bits
            or ((old_raw >> field.bit_start) & field.encoded_max) != old_value
        ):
            raise CharacterizationError(
                f"same-value operation for {name} is inconsistent or changes value"
            )
        validated_operations.append((operation, field))
    results: list[dict[str, object]] = []
    for operation, field in validated_operations:
        name = str(operation.get("name"))
        old_value = operation.get("old_value")
        new_value = operation.get("new_value")
        old_raw = operation.get("old_raw")
        assert type(old_value) is int and type(new_value) is int
        assert type(old_raw) is int
        try:
            write_start = monotonic_ns()
            written = i2c_writer.write_field(
                field, channel=channel, value=new_value,
                expected_current=old_value, expected_raw=old_raw,
            )
            write_readback_duration_s = (monotonic_ns() - write_start) / 1e9
        except Exception as error:
            return {
                "status": "RECOVERY_REQUIRED",
                "error": str(error),
                "failed_parameter": name,
                "write_results": results,
            }
        apply_duration_s = getattr(written, "apply_duration_s", None)
        readback_duration_s = getattr(written, "readback_duration_s", None)
        if not isinstance(apply_duration_s, (int, float)):
            apply_duration_s = write_readback_duration_s
        if not isinstance(readback_duration_s, (int, float)):
            readback_duration_s = max(
                0.0, write_readback_duration_s - float(apply_duration_s)
            )
        write_readback_duration_s = float(apply_duration_s) + float(
            readback_duration_s
        )
        results.append({
            "name": written.name,
            "address": written.address,
            "before_raw": written.before_raw,
            "before_value": written.before_value,
            "requested_value": written.requested_value,
            "after_raw": written.after_raw,
            "after_value": written.after_value,
            "verified": written.verified,
            "apply_duration_s": float(apply_duration_s),
            "readback_duration_s": float(readback_duration_s),
            "write_readback_duration_s": write_readback_duration_s,
        })
    return {"status": "VERIFIED", "write_results": results}


def create_settle_plan(
    config: LiveHardwareConfig,
    i2c_reader: object,
    i2c_identity: Vd1124ProbeResult,
    *,
    parameter: str,
    target_value: int,
    settle_delays: Sequence[float],
    channel: int,
    created_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Create a one-parameter settling plan with explicit human limits."""
    if type(channel) is not int or not 0 <= channel <= 3:
        raise ValueError("channel must be an integer from 0 through 3")
    if not isinstance(parameter, str) or not parameter.strip():
        raise CharacterizationError("parameter must be explicitly provided")
    if type(target_value) is not int:
        raise CharacterizationError("target value must be an integer")
    delays = tuple(_require_number(value, "settle delay") for value in settle_delays)
    if not delays or any(value < 0 for value in delays):
        raise CharacterizationError("settle delays must be non-empty and non-negative")
    field = next(
        (item for item in config.characterization.parameters if item.name == parameter),
        None,
    )
    if field is None:
        raise CharacterizationError(f"unknown parameter {parameter}")
    if field.safety_min is None or field.safety_max is None or field.max_step is None:
        raise CharacterizationError(f"{parameter} has no complete human safety limits")
    if not 0 <= target_value <= field.encoded_max:
        raise CharacterizationError(f"target value exceeds {parameter} encoding width")
    if not field.safety_min <= target_value <= field.safety_max:
        raise CharacterizationError(f"target value is outside {parameter} safety limits")

    base = _config_plan_base(
        config, kind="settle", channel=channel, created_at_utc=created_at_utc
    )
    hardware = _require_mapping(base["hardware"], "plan hardware")
    _verify_identity(identity=i2c_identity, i2c_plan=_require_mapping(hardware["i2c"], "plan I2C"))
    values = _read_parameter_snapshot(
        i2c_reader, config.characterization.parameters, channel
    )
    current = next(value for value in values if value.name == parameter)
    if target_value == current.value:
        raise CharacterizationError("settle target must differ from the original value")
    if abs(target_value - current.value) > field.max_step:
        raise CharacterizationError(f"target step exceeds {parameter} max_step")
    base["operations"] = [
        _operation_document(field, current, new_value=target_value)
    ]
    parameters = _require_mapping(base["parameters"], "plan parameters")
    parameters["parameter"] = parameter  # type: ignore[index]
    parameters["target_value"] = target_value  # type: ignore[index]
    parameters["settle_delays_s"] = list(delays)  # type: ignore[index]
    parameters["repeats"] = len(delays)  # type: ignore[index]
    base["creation_snapshot"] = {
        "i2c_identity": {
            "port": i2c_identity.port,
            "chip_id": i2c_identity.chip_id,
            "chip_revision": i2c_identity.chip_revision,
        },
        "i2c_parameters": _field_document(values),
    }
    base_safety = _require_mapping(base.get("safety"), "plan safety")
    base["safety"] = {
        "i2c_write_allowed": "one-parameter-and-one-restore",
        "bert_write_allowed": False,
        "requires_apply": True,
        "automatic_safe_off_allowed": False,
        "restore_attempts": 1,
        "dca_mutations": list(base_safety.get("dca_mutations", [])),
    }
    base["measurement_plan"] = create_baseline_plan(
        config, repeats=1, created_at_utc=created_at_utc
    )
    return rehash_plan(base)


def execute_settle_plan(
    plan: Mapping[str, object],
    i2c_writer: object,
    i2c_identity: Vd1124ProbeResult,
    measure_once: Callable[[float], Mapping[str, object] | object],
    *,
    allow_apply: bool,
    confirmation_id: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wait: Callable[[float], None] = time.sleep,
    now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    persist_round: RoundArtifactPersist | None = None,
) -> dict[str, object]:
    """Apply one field, measure at delays, and restore exactly once in finally."""
    _validate_apply(
        plan, expected_kind="settle", allow_apply=allow_apply,
        confirmation_id=confirmation_id,
    )
    fields, channel, _ = _preflight_operations(plan, i2c_writer, i2c_identity)
    operations = plan.get("operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise CharacterizationError("settle plan must contain exactly one operation")
    operation = _require_mapping(operations[0], "settle operation")
    name = str(operation.get("name"))
    old_value = operation.get("old_value")
    target_value = operation.get("new_value")
    if type(old_value) is not int or type(target_value) is not int:
        raise CharacterizationError("settle old and target values must be integers")
    field = next((item for item in fields if item.name == name), None)
    if field is None:
        raise CharacterizationError(f"unknown settle field {name}")
    if field.safety_min is None or field.safety_max is None or field.max_step is None:
        raise CharacterizationError(f"{name} has no complete human safety limits")
    if not field.safety_min <= target_value <= field.safety_max:
        raise CharacterizationError(f"target value is outside {name} safety limits")
    if not 0 <= target_value <= field.encoded_max:
        raise CharacterizationError(f"target value exceeds {name} encoding width")
    if target_value == old_value:
        raise CharacterizationError("settle target must differ from the original value")
    if abs(target_value - old_value) > field.max_step:
        raise CharacterizationError(f"target step exceeds {name} max_step")
    parameters = _require_mapping(plan["parameters"], "plan parameters")
    raw_delays = parameters.get("settle_delays_s")
    if not isinstance(raw_delays, list) or not raw_delays:
        raise CharacterizationError("settle plan has no delays")
    delays = tuple(_require_number(value, "settle delay") for value in raw_delays)
    if any(value < 0 for value in delays):
        raise CharacterizationError("settle delays must be non-negative")
    if parameters.get("repeats") != len(delays):
        raise CharacterizationError("settle measurement count does not match delays")
    old_raw = operation.get("old_raw")
    if type(old_raw) is not int or not 0 <= old_raw <= 0xFF:
        raise CharacterizationError("settle old raw byte is invalid")
    if operation.get("address") != field.address_for_channel(channel):
        raise CharacterizationError("settle address does not match parameter/channel")
    mask = field.encoded_max << field.bit_start
    target_raw = (old_raw & ~mask) | (target_value << field.bit_start)

    measurements: list[dict[str, object]] = []
    failure: BaseException | None = None
    restoration_succeeded = False
    restore_error: str | None = None
    restore_required = False
    target_write_readback_duration_s: float | None = None
    restore_write_readback_duration_s: float | None = None
    apply_duration_s: float | None = None
    readback_duration_s: float | None = None
    restore_apply_duration_s: float | None = None
    restore_readback_duration_s: float | None = None
    restore_duration_s: float | None = None
    started_at_utc = _utc_text(now_utc)
    manual_recovery = {
        "parameter": name,
        "channel": channel,
        "address": field.address_for_channel(channel),
        "bit_start": field.bit_start,
        "bits": field.bits,
        "raw_value": old_raw,
        "value": old_value,
    }
    total_start = monotonic_ns()
    try:
        restore_required = True
        target_write_start = monotonic_ns()
        target_write = i2c_writer.write_field(
            field, channel=channel, value=target_value,
            expected_current=old_value, expected_raw=old_raw,
        )
        target_write_readback_duration_s = (
            monotonic_ns() - target_write_start
        ) / 1e9
        apply_duration_s = getattr(target_write, "apply_duration_s", None)
        readback_duration_s = getattr(target_write, "readback_duration_s", None)
        if not isinstance(apply_duration_s, (int, float)):
            apply_duration_s = target_write_readback_duration_s
        if not isinstance(readback_duration_s, (int, float)):
            readback_duration_s = max(
                0.0, target_write_readback_duration_s - float(apply_duration_s)
            )
        target_write_readback_duration_s = float(apply_duration_s) + float(
            readback_duration_s
        )
        for index, delay in enumerate(delays, start=1):
            wait_start = monotonic_ns()
            wait(delay)
            actual_wait_s = (monotonic_ns() - wait_start) / 1e9
            measured = measure_once(delay)
            row = dict(measured) if isinstance(measured, Mapping) else {}
            round_raw_responses = row.pop("_round_raw_responses", None)
            row.update({
                "stage": "settle",
                "repeat_index": index,
                "requested_settle_s": delay,
                "actual_settle_s": actual_wait_s,
            })
            row["apply_duration_s"] = float(apply_duration_s)
            row["readback_duration_s"] = float(readback_duration_s)
            row["phase_sum_duration_s"] = _phase_sum_duration(row)
            candidate_round_time_s = max(
                0.0, (monotonic_ns() - total_start) / 1e9
            )
            row["candidate_round_time_s"] = candidate_round_time_s
            row["total_duration_s"] = candidate_round_time_s
            if persist_round is not None:
                row = persist_round(
                    row,
                    (
                        round_raw_responses
                        if isinstance(round_raw_responses, Mapping)
                        else None
                    ),
                    total_start,
                )
            else:
                row["_round_started_monotonic_ns"] = total_start
            measurements.append(row)
    except BaseException as error:
        failure = error
    finally:
        if restore_required:
            restore_phase_start = monotonic_ns()
            try:
                restore_start = monotonic_ns()
                restore_write = i2c_writer.write_field(
                    field, channel=channel, value=old_value,
                    expected_current=target_value, expected_raw=target_raw,
                )
                restore_write_readback_duration_s = (
                    monotonic_ns() - restore_start
                ) / 1e9
                restore_apply_duration_s = getattr(
                    restore_write, "apply_duration_s", None
                )
                restore_readback_duration_s = getattr(
                    restore_write, "readback_duration_s", None
                )
                if not isinstance(restore_apply_duration_s, (int, float)):
                    restore_apply_duration_s = restore_write_readback_duration_s
                if not isinstance(restore_readback_duration_s, (int, float)):
                    restore_readback_duration_s = max(
                        0.0,
                        restore_write_readback_duration_s
                        - float(restore_apply_duration_s),
                    )
                restore_write_readback_duration_s = float(
                    restore_apply_duration_s
                ) + float(restore_readback_duration_s)
                restored_fields = _read_parameter_snapshot(
                    i2c_writer, fields, channel
                )
                creation = _require_mapping(
                    plan.get("creation_snapshot"), "plan creation_snapshot"
                )
                created_fields = _require_mapping(
                    creation.get("i2c_parameters"),
                    "creation I2C parameters",
                )
                mismatches = []
                for value in restored_fields:
                    created = _require_mapping(
                        created_fields.get(value.name),
                        f"creation field {value.name}",
                    )
                    if (
                        created.get("value") != value.value
                        or created.get("raw_value") != value.raw_value
                    ):
                        mismatches.append(value.name)
                if mismatches:
                    raise CharacterizationError(
                        "post-restore I2C mismatch: " + ", ".join(mismatches)
                    )
                restoration_succeeded = True
            except BaseException as error:
                restore_error = str(error)
            finally:
                restore_duration_s = max(
                    0.0, (monotonic_ns() - restore_phase_start) / 1e9
                )

    candidate_round_time_s = (
        measurements[-1].get("candidate_round_time_s")
        if measurements
        else None
    )
    screening_round_with_restore_s = max(
        0.0, (monotonic_ns() - total_start) / 1e9
    )

    if not restoration_succeeded:
        return {
            "status": "RECOVERY_REQUIRED",
            "measurements": measurements,
            "restoration_succeeded": False,
            "error": str(failure) if failure is not None else restore_error,
            "restore_error": restore_error,
            "manual_recovery": manual_recovery,
            "target_write_readback_duration_s": target_write_readback_duration_s,
            "restore_write_readback_duration_s": restore_write_readback_duration_s,
            "apply_duration_s": apply_duration_s,
            "readback_duration_s": readback_duration_s,
            "restore_apply_duration_s": restore_apply_duration_s,
            "restore_readback_duration_s": restore_readback_duration_s,
            "candidate_round_time_s": candidate_round_time_s,
            "restore_duration_s": restore_duration_s,
            "screening_round_with_restore_s": screening_round_with_restore_s,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_text(now_utc),
            "total_duration_s": screening_round_with_restore_s,
        }
    if failure is not None:
        return {
            "status": "FAILED_RESTORED",
            "measurements": measurements,
            "restoration_succeeded": True,
            "error": f"{type(failure).__name__}: {failure}",
            "manual_recovery": None,
            "target_write_readback_duration_s": target_write_readback_duration_s,
            "restore_write_readback_duration_s": restore_write_readback_duration_s,
            "apply_duration_s": apply_duration_s,
            "readback_duration_s": readback_duration_s,
            "restore_apply_duration_s": restore_apply_duration_s,
            "restore_readback_duration_s": restore_readback_duration_s,
            "candidate_round_time_s": candidate_round_time_s,
            "restore_duration_s": restore_duration_s,
            "screening_round_with_restore_s": screening_round_with_restore_s,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_text(now_utc),
            "total_duration_s": screening_round_with_restore_s,
        }
    return {
        "status": "COMPLETED_RESTORED",
        "measurements": measurements,
        "restoration_succeeded": True,
        "manual_recovery": None,
        "target_write_readback_duration_s": target_write_readback_duration_s,
        "restore_write_readback_duration_s": restore_write_readback_duration_s,
        "apply_duration_s": apply_duration_s,
        "readback_duration_s": readback_duration_s,
        "restore_apply_duration_s": restore_apply_duration_s,
        "restore_readback_duration_s": restore_readback_duration_s,
        "candidate_round_time_s": candidate_round_time_s,
        "restore_duration_s": restore_duration_s,
        "screening_round_with_restore_s": screening_round_with_restore_s,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_text(now_utc),
        "total_duration_s": screening_round_with_restore_s,
    }


def validate_characterization_plan(
    document: Mapping[str, object], *, expected_kind: str | None = None
) -> Mapping[str, object]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CharacterizationError("characterization plan schema does not match")
    plan_id = document.get("plan_id")
    if not isinstance(plan_id, str) or len(plan_id) != 64:
        raise CharacterizationError("characterization plan ID is invalid")
    try:
        expected_id = _plan_hash(document)
    except (TypeError, ValueError) as error:
        raise CharacterizationError(f"plan cannot be hashed: {error}") from error
    if not hmac.compare_digest(plan_id, expected_id):
        raise CharacterizationError("characterization plan hash does not match")
    if expected_kind is not None and document.get("kind") != expected_kind:
        raise CharacterizationError(f"plan kind must be {expected_kind}")
    for section in ("hardware", "parameters", "safety"):
        if not isinstance(document.get(section), Mapping):
            raise CharacterizationError(f"plan {section} must be an object")
    return document


def validate_plan_against_config(
    document: Mapping[str, object], config: LiveHardwareConfig
) -> None:
    """Fail closed if a saved plan targets a different live configuration."""
    validate_characterization_plan(document)
    expected = create_baseline_plan(config, repeats=1)
    if document.get("hardware") != expected.get("hardware"):
        raise CharacterizationError("plan hardware target does not match live config")
    parameters = _require_mapping(document.get("parameters"), "plan parameters")
    expected_parameters = _require_mapping(
        expected.get("parameters"), "expected parameters"
    )
    for key in (
        "measurement_timeout_s",
        "poll_interval_s",
        "meaningful_er_floor_db",
        "meaningful_tdecq_floor_db",
    ):
        if parameters.get(key) != expected_parameters.get(key):
            raise CharacterizationError(
                f"plan {key} does not match the current live config"
            )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    document = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(document)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class DurableRoundArtifactWriter:
    """Persist one round synchronously before execution can start the next."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        run_id: str,
        started_at_utc: datetime,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        atomic_json: Callable[[Path, object], None] = _atomic_json,
    ) -> None:
        timestamp = started_at_utc.astimezone(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        self.run_dir = Path(output_root) / f"{timestamp}-{run_id}"
        self._monotonic_ns = monotonic_ns
        self._now_utc = now_utc
        self._atomic_json = atomic_json

    def persist(
        self,
        measurement: Mapping[str, object],
        raw_responses: Mapping[str, object] | None,
        round_started_monotonic_ns: int,
    ) -> dict[str, object]:
        if type(round_started_monotonic_ns) is not int:
            raise TypeError("round monotonic start must be an integer")
        row = dict(measurement)
        row.pop("_round_started_monotonic_ns", None)
        row["phase_sum_duration_s"] = _phase_sum_duration(row)
        row["artifact_started_at_utc"] = _utc_text(self._now_utc)
        artifact_start = self._monotonic_ns()
        repeat_index = row.get("repeat_index")
        if type(repeat_index) is not int or repeat_index <= 0:
            raise CharacterizationError("round repeat_index must be positive")
        self._atomic_json(
            self.run_dir / "round_artifacts" / f"repeat_{repeat_index}.json",
            {
                "measurement": row,
                "raw_responses": raw_responses,
            },
        )
        artifact_finished = self._monotonic_ns()
        row["artifact_duration_s"] = max(
            0.0, (artifact_finished - artifact_start) / 1e9
        )
        row["artifact_finished_at_utc"] = _utc_text(self._now_utc)
        candidate_round_time_s = max(
            0.0,
            (artifact_finished - round_started_monotonic_ns) / 1e9,
        )
        row["candidate_round_time_s"] = candidate_round_time_s
        row["total_duration_s"] = candidate_round_time_s
        return row


def save_characterization_plan(
    document: Mapping[str, object], path: str | Path
) -> Path:
    validate_characterization_plan(document)
    output = Path(path)
    _atomic_json(output, document)
    return output


def load_characterization_plan(path: str | Path) -> dict[str, object]:
    input_path = Path(path)
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CharacterizationError(f"cannot load plan {input_path}: {error}") from error
    if not isinstance(value, dict):
        raise CharacterizationError("characterization plan root must be an object")
    validate_characterization_plan(value)
    return value


def _finite_values(values: Sequence[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if isfinite(numeric) and abs(numeric) < 9.0e37:
            result.append(numeric)
    return result


def _metric_statistics(values: Sequence[object]) -> dict[str, object]:
    numbers = _finite_values(values)
    count = len(numbers)
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_std": None,
            "min": None,
            "max": None,
            "range": None,
            "insufficient_reason": "no valid samples",
        }
    minimum = min(numbers)
    maximum = max(numbers)
    return {
        "count": count,
        "mean": fmean(numbers),
        "median": median(numbers),
        "sample_std": stdev(numbers) if count >= 2 else None,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "insufficient_reason": (
            None if count >= 2 else "at least two valid samples are required"
        ),
    }


def _linear_percentile(numbers: Sequence[float], probability: float) -> float:
    ordered = sorted(numbers)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = floor(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def _duration_statistics(values: Sequence[object]) -> dict[str, object]:
    basic = _metric_statistics(values)
    numbers = _finite_values(values)
    return {
        "count": basic["count"],
        "mean": basic["mean"],
        "median": basic["median"],
        "p95": _linear_percentile(numbers, 0.95) if numbers else None,
        "min": basic["min"],
        "max": basic["max"],
        "insufficient_reason": (
            None if numbers else "no measured round durations"
        ),
    }


def summarize_measurements(
    measurements: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    valid = [row for row in measurements if row.get("observation_valid") is True]
    er_timeout_count = sum(
        1
        for row in measurements
        if "timeout" in str(row.get("er_invalid_reason") or "").lower()
    )
    tdecq_timeout_count = sum(
        1
        for row in measurements
        if "timeout" in str(row.get("tdecq_invalid_reason") or "").lower()
    )
    timeout_count = sum(
        1
        for row in measurements
        if any(
            "timeout" in str(row.get(field) or "").lower()
            for field in (
                "invalid_reason",
                "er_invalid_reason",
                "tdecq_invalid_reason",
            )
        )
    )
    status_changes: list[dict[str, object]] = []
    previous_status: tuple[object, object] | None = None
    for row in measurements:
        current_status = (row.get("er_status"), row.get("tdecq_status"))
        if current_status != previous_status:
            status_changes.append(
                {
                    "repeat_index": row.get("repeat_index"),
                    "er_status": current_status[0],
                    "tdecq_status": current_status[1],
                }
            )
            previous_status = current_status
    return {
        "measurement_count": len(measurements),
        "valid_count": len(valid),
        "valid_ratio": (
            len(valid) / len(measurements) if measurements else None
        ),
        "er_db": _metric_statistics([row.get("er_db") for row in valid]),
        "tdecq_db": _metric_statistics(
            [row.get("tdecq_db") for row in valid]
        ),
        "apply_duration_s": _duration_statistics(
            [row.get("apply_duration_s") for row in measurements]
        ),
        "readback_duration_s": _duration_statistics(
            [row.get("readback_duration_s") for row in measurements]
        ),
        "acquisition_duration_s": _duration_statistics(
            [row.get("acquisition_duration_s") for row in measurements]
        ),
        "wait_for_valid_duration_s": _duration_statistics(
            [row.get("wait_for_valid_duration_s") for row in measurements]
        ),
        "query_duration_s": _duration_statistics(
            [row.get("query_duration_s") for row in measurements]
        ),
        "artifact_duration_s": _duration_statistics(
            [row.get("artifact_duration_s") for row in measurements]
        ),
        "phase_sum_duration_s": _duration_statistics(
            [row.get("phase_sum_duration_s") for row in measurements]
        ),
        "candidate_round_time_s": _duration_statistics(
            [row.get("candidate_round_time_s") for row in measurements]
        ),
        "total_duration_s": _duration_statistics(
            [row.get("total_duration_s") for row in measurements]
        ),
        "timeout_count": timeout_count,
        "er_timeout_count": er_timeout_count,
        "tdecq_timeout_count": tdecq_timeout_count,
        "status_changes": status_changes,
        "i2c_write_readback_consistent": None,
        "restoration_succeeded": None,
    }


def build_convergence_defaults(
    summary: Mapping[str, object],
    *,
    er_floor_db: float | None,
    tdecq_floor_db: float | None,
    repeats: int,
) -> dict[str, object]:
    er_stats = summary.get("er_db")
    tdecq_stats = summary.get("tdecq_db")
    if not isinstance(er_stats, Mapping) or not isinstance(tdecq_stats, Mapping):
        raise ValueError("summary metric statistics are missing")
    er_sigma = er_stats.get("sample_std")
    tdecq_sigma = tdecq_stats.get("sample_std")

    def meaningful(sigma: object, floor_value: float | None) -> float | None:
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
            return None
        noise = 3.0 * float(sigma)
        return max(noise, floor_value) if floor_value is not None else noise

    return {
        "status": "SUGGESTED_NOT_FINAL",
        "method": "meaningful change = max(configured floor, 3 * sample sigma)",
        "er_noise_sigma_db": er_sigma,
        "tdecq_noise_sigma_db": tdecq_sigma,
        "meaningful_er_change_db": meaningful(er_sigma, er_floor_db),
        "meaningful_tdecq_change_db": meaningful(
            tdecq_sigma, tdecq_floor_db
        ),
        "suggested_repeats": max(5, repeats),
        "suggested_settle_time_s": None,
        "settle_time_status": "INDETERMINATE",
        "suggested_patience": 3,
        "coarse_er_threshold_db": 2.5,
        "fine_er_threshold_db": 3.0,
    }


_MEASUREMENT_COLUMNS = (
    "run_id",
    "stage",
    "repeat_index",
    "started_at_utc",
    "finished_at_utc",
    "artifact_started_at_utc",
    "artifact_finished_at_utc",
    "apply_duration_s",
    "readback_duration_s",
    "acquisition_duration_s",
    "wait_for_valid_duration_s",
    "query_duration_s",
    "artifact_duration_s",
    "phase_sum_duration_s",
    "candidate_round_time_s",
    "total_duration_s",
    "requested_settle_s",
    "actual_settle_s",
    "er_db",
    "er_status",
    "er_details",
    "er_source",
    "er_invalid_reason",
    "tdecq_db",
    "tdecq_status",
    "tdecq_details",
    "tdecq_source",
    "tdecq_invalid_reason",
    "observation_valid",
    "invalid_reason",
    "ibias_dac",
    "efin_cur",
    "vga_cur",
    "eq",
    "res_gain",
    "res_bw",
    "gain_ctrl",
    "vbias_ref",
    "vref",
    "bert_fir_taps",
    "dca_measurement_source",
)


def write_run_artifacts(
    output_root: str | Path,
    *,
    run_id: str,
    plan: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    measurements: Sequence[Mapping[str, object]],
    raw_responses: Mapping[str, object],
    initial_snapshot: Mapping[str, object],
    final_snapshot: Mapping[str, object],
    summary: Mapping[str, object],
    convergence_defaults: Mapping[str, object],
    report_markdown: str,
    started_at_utc: datetime,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    existing_run_dir: str | Path | None = None,
    round_atomic_json: Callable[[Path, object], None] = _atomic_json,
) -> Path:
    timestamp = started_at_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    expected_run_dir = Path(output_root) / f"{timestamp}-{run_id}"
    run_dir = (
        Path(existing_run_dir)
        if existing_run_dir is not None
        else expected_run_dir
    )
    if run_dir != expected_run_dir:
        raise CharacterizationError("existing run directory does not match run identity")
    try:
        run_dir.mkdir(parents=True, exist_ok=existing_run_dir is not None)
    except OSError as error:
        raise CharacterizationError(f"cannot create run directory: {error}") from error

    persisted_measurements: list[dict[str, object]] = []
    try:
        round_writer = DurableRoundArtifactWriter(
            output_root,
            run_id=run_id,
            started_at_utc=started_at_utc,
            monotonic_ns=monotonic_ns,
            now_utc=now_utc,
            atomic_json=round_atomic_json,
        )
        for ordinal, original in enumerate(measurements, start=1):
            row = dict(original)
            for field in (
                "apply_duration_s",
                "readback_duration_s",
                "acquisition_duration_s",
                "wait_for_valid_duration_s",
                "query_duration_s",
                "artifact_duration_s",
            ):
                row.setdefault(field, 0.0)
            repeat_index = row.get("repeat_index", ordinal)
            raw_key = f"repeat_{repeat_index}"
            round_path = (
                run_dir / "round_artifacts" / f"repeat_{repeat_index}.json"
            )
            if (
                round_path.is_file()
                and isinstance(row.get("artifact_finished_at_utc"), str)
                and isinstance(row.get("artifact_duration_s"), (int, float))
            ):
                row.pop("_round_started_monotonic_ns", None)
            else:
                round_started = row.pop(
                    "_round_started_monotonic_ns", None
                )
                if type(round_started) is not int:
                    raise CharacterizationError(
                        "round is missing its continuous monotonic start"
                    )
                row = round_writer.persist(
                    row,
                    raw_responses.get(raw_key),
                    round_started,
                )
            persisted_measurements.append(row)

        persisted_summary = summarize_measurements(persisted_measurements)
        for key, value in summary.items():
            if key not in persisted_summary or (
                persisted_summary[key] is None and value is not None
            ):
                persisted_summary[key] = value
        persisted_convergence = dict(convergence_defaults)
        timing_lines = ["", "## Round timing / 每轮计时", ""]
        timing_labels = (
            ("apply_duration_s", "Apply"),
            ("readback_duration_s", "Readback"),
            ("acquisition_duration_s", "Acquisition"),
            ("wait_for_valid_duration_s", "Wait for valid"),
            ("query_duration_s", "Final SCPI queries"),
            ("artifact_duration_s", "Artifact persistence"),
            ("phase_sum_duration_s", "Phase sum before artifact"),
            ("candidate_round_time_s", "Candidate round"),
            ("total_duration_s", "Total"),
        )
        for field, label in timing_labels:
            statistics = persisted_summary.get(field)
            mean = statistics.get("mean") if isinstance(statistics, Mapping) else None
            timing_lines.append(f"- {label}: mean={mean!r} s")
        persisted_report = report_markdown.rstrip() + "\n" + "\n".join(timing_lines) + "\n"

        _atomic_json(run_dir / "plan.json", plan)
        events_path = run_dir / "events.jsonl"
        events_text = "".join(
            json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            for event in events
        )
        events_path.write_text(events_text, encoding="utf-8")
        csv_path = run_dir / "measurements.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=_MEASUREMENT_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in persisted_measurements:
                writer.writerow(row)
        _atomic_json(run_dir / "measurements.json", persisted_measurements)
        _atomic_json(run_dir / "raw_responses.json", raw_responses)
        _atomic_json(run_dir / "initial_snapshot.json", initial_snapshot)
        _atomic_json(run_dir / "final_snapshot.json", final_snapshot)
        _atomic_json(run_dir / "summary.json", persisted_summary)
        _atomic_json(
            run_dir / "convergence_defaults.json", persisted_convergence
        )
        (run_dir / "report.md").write_text(persisted_report, encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise CharacterizationError(f"cannot persist audit artifacts: {error}") from error
    return run_dir
