"""Confirmation-gated FlexDCA acquisition planning and execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
from math import isfinite
from pathlib import Path
import re
import time
from typing import Protocol

from vcsel_ai_tuning.hardware.flexdca import (
    DcaSnapshot,
    QuerySession,
    read_verified_tdecq,
)
from vcsel_ai_tuning.hardware.vd1124 import Vd1124ProbeResult
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


SCHEMA_VERSION = "2.0"
SINGLE_COMMANDS = (
    ":MEASure:EYE:TDEQ:SOURce FUNC1",
    ":ACQuire:STOP",
    ":ACQuire:CDISplay",
    ":ACQuire:SINGle",
)


class AcquisitionWorkflowError(RuntimeError):
    """Raised when an acquisition cannot pass a deterministic safety gate."""


class BertConfigurationReader(Protocol):
    def get_configuration(
        self, port: str, channel: str | int
    ) -> BertConfiguration: ...


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _measurement_hash(document: Mapping[str, object]) -> str:
    payload = {
        key: value
        for key, value in document.items()
        if key != "measurement_id"
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_measurement_source(source: str) -> str:
    if not isinstance(source, str):
        raise ValueError("measurement_source must be a string")
    text = source.strip()
    patterns = (
        (
            r"CHAN(?:NEL)?([1-8])([A-D])",
            lambda match: f"CHAN{match.group(1)}{match.group(2).upper()}",
        ),
        (
            r"FUNC(?:TION)?([0-9]+)",
            lambda match: f"FUNC{int(match.group(1), 10)}",
        ),
        (r"DIFF([1-4])", lambda match: f"DIFF{match.group(1)}"),
        (
            r"CGM(?:EMORY)?([1-8])",
            lambda match: f"CGMEMORY{match.group(1)}",
        ),
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


def _positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def create_measurement_plan(
    *,
    resource: str,
    function: int,
    measurement_source: str,
    mode: str,
    timeout_s: float,
    poll_interval_s: float,
    created_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Build one immutable plan for a Function 1 Single acquisition."""
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("resource must be a non-empty string")
    if type(function) is not int or function != 1:
        raise ValueError("Single acquisition currently requires function 1")
    source = _canonical_measurement_source(measurement_source)
    if source != "FUNC1":
        raise ValueError("TDECQ measurement_source must be FUNC1")
    if mode != "single":
        raise ValueError("mode must be 'single'")
    timeout = _positive_finite(timeout_s, "timeout_s")
    poll_interval = _positive_finite(poll_interval_s, "poll_interval_s")
    if poll_interval > timeout:
        raise ValueError("poll_interval_s must not exceed timeout_s")
    created = created_at_utc or datetime.now(timezone.utc)
    if not isinstance(created, datetime) or created.tzinfo is None:
        raise ValueError("created_at_utc must be a timezone-aware datetime")

    commands = (
        f":MEASure:EYE:TDEQ:SOURce {source}",
        *SINGLE_COMMANDS[1:],
    )
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created.astimezone(timezone.utc).isoformat(),
        "target": {
            "resource": resource.strip(),
            "function": function,
            "measurement_source": source,
        },
        "acquisition": {
            "mode": mode,
            "timeout_s": timeout,
            "poll_interval_s": poll_interval,
            "commands": list(commands),
        },
    }
    plan["measurement_id"] = _measurement_hash(plan)
    return plan


def _validated_measurement_plan(
    document: Mapping[str, object],
) -> Mapping[str, object]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise AcquisitionWorkflowError(
            f"measurement plan schema must be {SCHEMA_VERSION!r}"
        )
    measurement_id = document.get("measurement_id")
    if not isinstance(measurement_id, str) or len(measurement_id) != 64:
        raise AcquisitionWorkflowError("measurement ID is missing or invalid")
    try:
        expected_id = _measurement_hash(document)
    except (TypeError, ValueError) as error:
        raise AcquisitionWorkflowError(
            f"measurement plan cannot be hashed: {error}"
        ) from error
    if not hmac.compare_digest(measurement_id, expected_id):
        raise AcquisitionWorkflowError("measurement plan hash does not match")

    target = document.get("target")
    acquisition = document.get("acquisition")
    if not isinstance(target, Mapping) or not isinstance(acquisition, Mapping):
        raise AcquisitionWorkflowError("measurement plan sections are invalid")
    try:
        expected = create_measurement_plan(
            resource=target.get("resource"),
            function=target.get("function"),
            measurement_source=target.get("measurement_source"),
            mode=acquisition.get("mode"),
            timeout_s=acquisition.get("timeout_s"),
            poll_interval_s=acquisition.get("poll_interval_s"),
            created_at_utc=datetime.fromisoformat(str(document.get("created_at_utc"))),
        )
    except (TypeError, ValueError) as error:
        raise AcquisitionWorkflowError(
            f"measurement plan contents are invalid: {error}"
        ) from error
    if acquisition.get("commands") != expected["acquisition"]["commands"]:
        raise AcquisitionWorkflowError("measurement acquisition commands are invalid")
    return document


def save_measurement_plan(
    document: Mapping[str, object], path: str | Path
) -> Path:
    """Validate and atomically replace a measurement plan JSON file."""
    _validated_measurement_plan(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_measurement_plan(path: str | Path) -> dict[str, object]:
    """Load a measurement plan and reject malformed or edited content."""
    plan_path = Path(path)
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionWorkflowError(
            f"cannot load measurement plan {plan_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AcquisitionWorkflowError("measurement plan root must be an object")
    _validated_measurement_plan(value)
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AcquisitionWorkflowError(f"measurement plan {label} must be an object")
    return value


def _bert_document(configuration: BertConfiguration) -> dict[str, object]:
    return {
        "port": configuration.port,
        "channel": configuration.channel,
        "taps": list(configuration.taps),
        "upper_nlc": configuration.upper_nlc,
        "lower_nlc": configuration.lower_nlc,
        "raw_response": configuration.raw_response,
    }


def _i2c_document(result: Vd1124ProbeResult) -> dict[str, object]:
    return {
        "port": result.port,
        "chip_id": result.chip_id,
        "chip_revision": result.chip_revision,
        "matched": result.matched,
    }


def _dca_document(snapshot: DcaSnapshot) -> dict[str, object]:
    return {
        "function": snapshot.function,
        "measurement_source": snapshot.measurement_source,
        "verified_measurement_source": snapshot.verified_measurement_source,
        "idn": snapshot.idn,
        "tdecq_db": snapshot.tdecq_db,
        "status": snapshot.status,
        "details": snapshot.details,
        "taps": list(snapshot.taps),
        "precursor_count": snapshot.precursor_count,
        "main_index": snapshot.main_index,
        "raw_responses": dict(snapshot.raw_responses),
    }


def _same_bert(left: BertConfiguration, right: BertConfiguration) -> bool:
    return (
        left.port == right.port
        and left.channel == right.channel
        and left.taps == right.taps
        and left.upper_nlc == right.upper_nlc
        and left.lower_nlc == right.lower_nlc
    )


def execute_single_measurement(
    plan: Mapping[str, object],
    dca_session: QuerySession,
    bert_client: BertConfigurationReader,
    i2c: Vd1124ProbeResult,
    *,
    bert_port: str,
    bert_channel: str,
    allow_execute: bool,
    confirmation_id: str | None,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    now_utc: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Execute one authorized Stop/Clear/Single acquisition and wait for CORR."""
    if allow_execute is not True:
        raise AcquisitionWorkflowError(
            "live acquisition requires the explicit --execute flag"
        )
    measurement_id = plan.get("measurement_id")
    if not isinstance(confirmation_id, str) or confirmation_id != measurement_id:
        raise AcquisitionWorkflowError(
            "confirmation must contain the exact measurement ID"
        )
    _validated_measurement_plan(plan)
    if not i2c.matched:
        raise AcquisitionWorkflowError("I2C device identity does not match")

    target = _require_mapping(plan.get("target"), "target")
    acquisition = _require_mapping(plan.get("acquisition"), "acquisition")
    function = int(target["function"])
    expected_source = str(target["measurement_source"])
    timeout_s = float(acquisition["timeout_s"])
    poll_interval_s = float(acquisition["poll_interval_s"])
    commands = tuple(str(command) for command in acquisition["commands"])
    timestamp = now_utc or (lambda: datetime.now(timezone.utc))

    bert_before = bert_client.get_configuration(bert_port, bert_channel)
    started_at = timestamp()
    idn = dca_session.query("*IDN?").strip()
    command_log: list[dict[str, object]] = []
    measurement_record: dict[str, object] = {
        "requested_source": expected_source,
        "verified_source": None,
        "status": "PENDING",
        "value": None,
        "details": "",
    }
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_id": measurement_id,
        "status": None,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "dca_idn": idn,
        "target": dict(target),
        "verified_measurement_source": None,
        "command_log": command_log,
        "clear_boundary": None,
        "poll_count": 0,
        "status_history": [],
        "final_tdecq": None,
        "measurement": measurement_record,
        "bert_before": _bert_document(bert_before),
        "bert_after": None,
        "i2c": _i2c_document(i2c),
        "failed_stage": None,
        "error": None,
        "timeout_stop": None,
    }

    def finish(
        status: str,
        *,
        failed_stage: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        receipt["status"] = status
        receipt["finished_at_utc"] = timestamp().isoformat()
        receipt["failed_stage"] = failed_stage
        receipt["error"] = error
        return receipt

    def write(stage: str, command: str) -> None:
        entry = {
            "timestamp_utc": timestamp().isoformat(),
            "stage": stage,
            "command": command,
            "outcome": "OK",
            "error": None,
        }
        try:
            dca_session.write(command)
        except Exception as error:
            entry["outcome"] = "FAILED"
            entry["error"] = str(error)
            command_log.append(entry)
            raise
        command_log.append(entry)

    stage = "select_source"
    try:
        write(stage, commands[0])
        stage = "verify_source"
        raw_verified_source = dca_session.query(
            ":MEASure:EYE:TDEQ:SOURce?"
        ).strip().upper()
        try:
            verified_source = _canonical_measurement_source(raw_verified_source)
        except ValueError:
            verified_source = raw_verified_source
        receipt["verified_measurement_source"] = verified_source
        measurement_record["verified_source"] = verified_source
        if verified_source != expected_source:
            measurement_record["status"] = "SOURCE_MISMATCH"
            measurement_record["details"] = (
                f"TDECQ source {verified_source!r} does not match "
                f"{expected_source!r}"
            )
            raise AcquisitionWorkflowError(
                f"TDECQ source {verified_source!r} does not match {expected_source!r}"
            )

        stage = "stop_before_clear"
        write(stage, commands[1])
        stage = "clear"
        write(stage, commands[2])
        stage = "clear_boundary"
        clear_status = dca_session.query(
            ":MEASure:EYE:TDEQ:STATus?"
        ).strip().upper()
        clear_details = dca_session.query(
            ":MEASure:EYE:TDEQ:STATus:DETails?"
        ).strip()
        receipt["clear_boundary"] = {
            "timestamp_utc": timestamp().isoformat(),
            "status": clear_status,
            "details": clear_details,
        }
        if clear_status == "CORR":
            raise AcquisitionWorkflowError("Clear did not invalidate TDECQ")

        stage = "single"
        write(stage, commands[3])
        deadline = monotonic() + timeout_s
        poll_count = 0
        status_history: list[dict[str, object]] = []
        receipt["status_history"] = status_history
        last_status: str | None = None

        while monotonic() < deadline:
            stage = "poll"
            raw_source = dca_session.query(
                ":MEASure:EYE:TDEQ:SOURce?"
            ).strip().upper()
            try:
                source = _canonical_measurement_source(raw_source)
            except ValueError:
                source = raw_source
            status = dca_session.query(
                ":MEASure:EYE:TDEQ:STATus?"
            ).strip().upper()
            details = dca_session.query(
                ":MEASure:EYE:TDEQ:STATus:DETails?"
            ).strip()
            poll_count += 1
            receipt["poll_count"] = poll_count
            measurement_record["verified_source"] = source
            measurement_record["status"] = status
            measurement_record["details"] = details
            if source != expected_source:
                measurement_record["status"] = "SOURCE_MISMATCH"
                raise AcquisitionWorkflowError(
                    f"TDECQ source changed from {expected_source} to {source}"
                )
            if status != last_status:
                status_history.append(
                    {
                        "timestamp_utc": timestamp().isoformat(),
                        "source": source,
                        "status": status,
                        "details": details,
                    }
                )
                last_status = status
            if status == "CORR":
                stage = "final_snapshot"
                snapshot = read_verified_tdecq(
                    dca_session,
                    function=function,
                    measurement_source=expected_source,
                )
                receipt["final_tdecq"] = _dca_document(snapshot)
                receipt[
                    "verified_measurement_source"
                ] = snapshot.verified_measurement_source
                measurement_record.update(
                    {
                        "verified_source": snapshot.verified_measurement_source,
                        "status": snapshot.status,
                        "value": snapshot.tdecq_db,
                        "details": snapshot.details,
                    }
                )
                stage = "bert_after"
                bert_after = bert_client.get_configuration(
                    bert_port, bert_channel
                )
                receipt["bert_after"] = _bert_document(bert_after)
                return finish(
                    "MEASURED"
                    if _same_bert(bert_before, bert_after)
                    else "INVALIDATED"
                )
            stage = "wait"
            wait(poll_interval_s)
    except Exception as error:
        if measurement_record["status"] not in {"SOURCE_MISMATCH", "INV"}:
            measurement_record["status"] = "INVALID"
            measurement_record["value"] = None
            measurement_record["details"] = str(error)
        return finish("FAILED", failed_stage=stage, error=str(error))

    timeout_stop: dict[str, object] = {
        "timestamp_utc": timestamp().isoformat(),
        "command": ":ACQuire:STOP",
        "succeeded": False,
        "error": None,
    }
    try:
        write("timeout_stop", ":ACQuire:STOP")
        timeout_stop["succeeded"] = True
    except Exception as error:
        timeout_stop["error"] = str(error)
    receipt["timeout_stop"] = timeout_stop
    last_details = None
    history = receipt.get("status_history")
    if isinstance(history, list) and history:
        last_entry = history[-1]
        if isinstance(last_entry, Mapping):
            last_details = last_entry.get("details")
    return finish(
        "TIMED_OUT",
        failed_stage="wait_for_corr",
        error=(
            f"TDECQ did not become CORR before {timeout_s:g} s timeout; "
            f"last details: {last_details or '<none>'}"
        ),
    )
