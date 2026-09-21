"""Safe orchestration for the commissioned DCA, VD1124, and VIT BERT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path
from typing import Any, Protocol

from vcsel_ai_tuning.campaign.models import FirSettings
from vcsel_ai_tuning.hardware.fir import (
    calculate_first_pass,
    calculate_iterative_pass,
    format_setfir_preview,
)
from vcsel_ai_tuning.hardware.flexdca import (
    DcaSnapshot,
    QuerySession,
    read_existing_tdecq,
)
from vcsel_ai_tuning.hardware.vd1124 import Vd1124ProbeResult
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration


SCHEMA_VERSION = "2.0"


class LiveWorkflowError(RuntimeError):
    """Raised when a live operation cannot pass a deterministic safety gate."""


class BertReader(Protocol):
    host: str
    port: int

    def get_mode(self) -> str: ...

    def get_configuration(
        self, port: str, channel: str | int
    ) -> BertConfiguration: ...


class BertWriter(BertReader, Protocol):
    def set_fir(
        self,
        port: str,
        channel: str | int,
        *,
        taps: tuple[int, ...],
        upper_nlc: int,
        lower_nlc: int,
    ) -> str: ...


class I2cIdentityProbe(Protocol):
    def probe(self) -> Vd1124ProbeResult: ...


class CampaignBertAdapter:
    """Lazy fixed-pattern bridge from VitBertClient to campaign calls."""

    def __init__(
        self,
        *,
        client: BertWriter,
        port: str,
        channel: str,
        fixed_pattern: str,
        upper_nlc: int,
        lower_nlc: int,
        expected_host: str,
        expected_udp_port: int,
    ) -> None:
        if fixed_pattern != "SSPRQ":
            raise ValueError("campaign BERT pattern must be fixed to SSPRQ")
        if not isinstance(port, str) or not port.strip():
            raise ValueError("BERT port must be a non-empty string")
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("BERT channel must be a non-empty string")
        if type(upper_nlc) is not int or type(lower_nlc) is not int:
            raise TypeError("BERT NLC values must be integers")
        if type(expected_udp_port) is not int or not 1 <= expected_udp_port <= 65535:
            raise ValueError("expected_udp_port must be from 1 through 65535")
        self._client = client
        self._port = port.strip()
        self._channel = channel.strip()
        self._fixed_pattern = fixed_pattern
        self._upper_nlc = upper_nlc
        self._lower_nlc = lower_nlc
        self._expected_host = expected_host
        self._expected_udp_port = expected_udp_port

    def _endpoint_matches(self) -> bool:
        return (
            getattr(self._client, "host", None) == self._expected_host
            and getattr(self._client, "port", None) == self._expected_udp_port
        )

    def _configuration_fir(self, configuration: BertConfiguration) -> FirSettings:
        if (
            configuration.port != self._port
            or configuration.channel != self._channel
            or configuration.upper_nlc != self._upper_nlc
            or configuration.lower_nlc != self._lower_nlc
        ):
            raise LiveWorkflowError(
                "BERT configuration does not match fixed port/channel/NLC"
            )
        try:
            return FirSettings(
                taps=configuration.taps,
                upper_nlc=configuration.upper_nlc,
                lower_nlc=configuration.lower_nlc,
            )
        except (TypeError, ValueError) as error:
            raise LiveWorkflowError(f"BERT FIR readback is invalid: {error}") from error

    def _configuration_pattern_fields(
        self, configuration: BertConfiguration
    ) -> tuple[str, str]:
        raw_fields = getattr(configuration, "pattern_fields", None)
        if (
            not isinstance(raw_fields, tuple)
            or len(raw_fields) != 2
            or any(not isinstance(value, str) for value in raw_fields)
        ):
            raise LiveWorkflowError("BERT pattern attestation is unavailable")
        fields = tuple(value.strip() for value in raw_fields)
        if any(value != self._fixed_pattern for value in fields):
            raise LiveWorkflowError(
                "BERT pattern fields do not both attest fixed SSPRQ"
            )
        return fields

    def preflight(self) -> Mapping[str, object]:
        mode = self._client.get_mode()
        configuration = self._client.get_configuration(
            self._port, self._channel
        )
        configuration_verified = True
        pattern_verified = True
        try:
            fir = self._configuration_fir(configuration)
            pattern_fields = self._configuration_pattern_fields(configuration)
        except LiveWorkflowError:
            configuration_verified = False
            pattern_verified = False
            fir = None
            raw_pattern_fields = getattr(configuration, "pattern_fields", ())
            pattern_fields = (
                tuple(raw_pattern_fields)
                if isinstance(raw_pattern_fields, tuple)
                else ()
            )
        return {
            "identity_verified": self._endpoint_matches() and bool(mode.strip()),
            "configuration_verified": configuration_verified,
            "host": getattr(self._client, "host", None),
            "udp_port": getattr(self._client, "port", None),
            "mode": mode,
            "pattern": self._fixed_pattern,
            "pattern_fields": pattern_fields,
            "pattern_verified": pattern_verified,
            "port": configuration.port,
            "channel": configuration.channel,
            "upper_nlc": configuration.upper_nlc,
            "lower_nlc": configuration.lower_nlc,
            "fir": fir,
            "raw_response": configuration.raw_response,
        }

    def readback(self) -> Mapping[str, object]:
        if not self._endpoint_matches():
            raise LiveWorkflowError("BERT endpoint differs from configured target")
        configuration = self._client.get_configuration(
            self._port, self._channel
        )
        self._configuration_pattern_fields(configuration)
        return {
            "pattern": self._fixed_pattern,
            "fir": self._configuration_fir(configuration),
        }

    def apply(self, pattern: str, fir: FirSettings) -> FirSettings:
        if pattern != self._fixed_pattern:
            raise LiveWorkflowError(
                f"BERT pattern is fixed to {self._fixed_pattern}"
            )
        if not isinstance(fir, FirSettings):
            raise TypeError("fir must be FirSettings")
        if fir.upper_nlc != self._upper_nlc or fir.lower_nlc != self._lower_nlc:
            raise LiveWorkflowError("BERT NLC values are fixed for this campaign")
        if not self._endpoint_matches():
            raise LiveWorkflowError("BERT endpoint differs from configured target")
        before = self._client.get_configuration(self._port, self._channel)
        self._configuration_fir(before)
        self._configuration_pattern_fields(before)
        self._client.set_fir(
            self._port,
            self._channel,
            taps=fir.taps,
            upper_nlc=self._upper_nlc,
            lower_nlc=self._lower_nlc,
        )
        configuration = self._client.get_configuration(
            self._port, self._channel
        )
        self._configuration_pattern_fields(configuration)
        readback = self._configuration_fir(configuration)
        if readback != fir:
            raise LiveWorkflowError("BERT FIR readback does not match requested FIR")
        return readback


@dataclass(frozen=True)
class HardwareSnapshot:
    dca: DcaSnapshot
    bert_mode: str
    bert: BertConfiguration
    i2c: Vd1124ProbeResult


def collect_hardware_snapshot(
    dca_session: QuerySession,
    bert_client: BertReader,
    i2c_probe: I2cIdentityProbe,
    *,
    dca_function: int,
    dca_measurement_source: str,
    bert_port: str,
    bert_channel: str,
) -> HardwareSnapshot:
    """Read each connected module once without changing hardware state."""
    dca = read_existing_tdecq(
        dca_session,
        function=dca_function,
        measurement_source=dca_measurement_source,
    )
    bert_mode = bert_client.get_mode()
    bert = bert_client.get_configuration(bert_port, bert_channel)
    i2c = i2c_probe.probe()
    return HardwareSnapshot(dca=dca, bert_mode=bert_mode, bert=bert, i2c=i2c)


def _dca_document(snapshot: DcaSnapshot) -> dict[str, object]:
    return {
        "timestamp_utc": snapshot.timestamp_utc.isoformat(),
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


def hardware_snapshot_document(snapshot: HardwareSnapshot) -> dict[str, object]:
    """Convert a three-module snapshot to a JSON-safe read-only record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": False,
        "dca_source_selection_only": True,
        "dca": _dca_document(snapshot.dca),
        "bert_mode": snapshot.bert_mode,
        "bert": _bert_document(snapshot.bert),
        "i2c": _i2c_document(snapshot.i2c),
    }


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _proposal_hash(document: Mapping[str, object]) -> str:
    payload = {key: value for key, value in document.items() if key != "proposal_id"}
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_normalize(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def create_live_proposal(
    snapshot: HardwareSnapshot,
    *,
    mode: str,
    manual_main: int,
    bert_host: str,
    bert_udp_port: int,
) -> dict[str, object]:
    """Create an immutable proposal from current read-only hardware snapshots."""
    if not snapshot.i2c.matched:
        raise LiveWorkflowError(
            "I2C device identity does not match the commissioned VD1124"
        )
    if mode == "first":
        calculation = calculate_first_pass(
            snapshot.dca.taps,
            dca_main_index=snapshot.dca.main_index,
            manual_main=manual_main,
        )
    elif mode == "iterative":
        calculation = calculate_iterative_pass(
            snapshot.bert.taps,
            snapshot.dca.taps,
            dca_main_index=snapshot.dca.main_index,
            manual_main=manual_main,
        )
    else:
        raise ValueError("mode must be 'first' or 'iterative'")

    command_preview = None
    if calculation.valid:
        command_preview = format_setfir_preview(
            snapshot.bert.port,
            snapshot.bert.channel,
            calculation.final_taps,
            snapshot.bert.upper_nlc,
            snapshot.bert.lower_nlc,
        )

    proposal: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "host": bert_host,
            "udp_port": bert_udp_port,
            "port": snapshot.bert.port,
            "channel": snapshot.bert.channel,
        },
        "dca": _dca_document(snapshot.dca),
        "i2c": _i2c_document(snapshot.i2c),
        "bert_mode": snapshot.bert_mode,
        "bert_before": _bert_document(snapshot.bert),
        "calculation": _json_normalize(asdict(calculation)),
        "command_preview": command_preview,
    }
    proposal["proposal_id"] = _proposal_hash(proposal)
    return proposal


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LiveWorkflowError(f"proposal {label} must be an object")
    return value


def _validated_proposal(document: Mapping[str, object]) -> Mapping[str, object]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise LiveWorkflowError(
            f"proposal schema must be {SCHEMA_VERSION!r}"
        )
    proposal_id = document.get("proposal_id")
    if not isinstance(proposal_id, str) or len(proposal_id) != 64:
        raise LiveWorkflowError("proposal ID is missing or invalid")
    expected = _proposal_hash(document)
    if not hmac.compare_digest(proposal_id, expected):
        raise LiveWorkflowError("proposal hash does not match its contents")
    _require_mapping(document.get("target"), "target")
    _require_mapping(document.get("bert_before"), "bert_before")
    _require_mapping(document.get("calculation"), "calculation")
    return document


def save_live_proposal(document: Mapping[str, object], path: str | Path) -> Path:
    """Validate and atomically replace a proposal JSON file."""
    _validated_proposal(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_live_proposal(path: str | Path) -> dict[str, object]:
    """Load a proposal and reject any edited or malformed content."""
    proposal_path = Path(path)
    try:
        value = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LiveWorkflowError(f"cannot load proposal {proposal_path}: {error}") from error
    if not isinstance(value, dict):
        raise LiveWorkflowError("proposal root must be an object")
    _validated_proposal(value)
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or any(type(item) is not int for item in value)
    ):
        raise LiveWorkflowError(f"proposal {label} must contain seven integers")
    return tuple(value)


def _state_values(document: Mapping[str, object]) -> tuple[tuple[int, ...], int, int]:
    taps = _integer_tuple(document.get("taps"), "BERT taps")
    upper_nlc = document.get("upper_nlc")
    lower_nlc = document.get("lower_nlc")
    if type(upper_nlc) is not int or type(lower_nlc) is not int:
        raise LiveWorkflowError("proposal NLC values must be integers")
    return taps, upper_nlc, lower_nlc


def _configuration_matches(
    configuration: BertConfiguration,
    expected: Mapping[str, object],
) -> bool:
    taps, upper_nlc, lower_nlc = _state_values(expected)
    return (
        configuration.port == expected.get("port")
        and configuration.channel == str(expected.get("channel"))
        and configuration.taps == taps
        and configuration.upper_nlc == upper_nlc
        and configuration.lower_nlc == lower_nlc
    )


def _target_state(proposal: Mapping[str, object]) -> dict[str, object]:
    calculation = _require_mapping(proposal.get("calculation"), "calculation")
    before = _require_mapping(proposal.get("bert_before"), "bert_before")
    return {
        "port": before.get("port"),
        "channel": before.get("channel"),
        "taps": list(_integer_tuple(calculation.get("final_taps"), "final taps")),
        "upper_nlc": before.get("upper_nlc"),
        "lower_nlc": before.get("lower_nlc"),
    }


def apply_live_proposal(
    proposal: Mapping[str, object],
    bert_client: BertWriter,
    *,
    allow_apply: bool,
    confirmation_id: str | None,
) -> dict[str, object]:
    """Apply once, verify exact readback, and attempt at most one rollback."""
    if allow_apply is not True:
        raise LiveWorkflowError("live write requires the explicit --apply flag")
    proposal_id = proposal.get("proposal_id")
    if not isinstance(confirmation_id, str) or confirmation_id != proposal_id:
        raise LiveWorkflowError("confirmation must contain the exact proposal ID")
    _validated_proposal(proposal)

    calculation = _require_mapping(proposal.get("calculation"), "calculation")
    if calculation.get("valid") is not True:
        raise LiveWorkflowError("an invalid FIR proposal cannot be applied")
    before_document = _require_mapping(proposal.get("bert_before"), "bert_before")
    target_document = _target_state(proposal)
    target = _require_mapping(proposal.get("target"), "target")
    if getattr(bert_client, "host", None) != target.get("host") or getattr(
        bert_client, "port", None
    ) != target.get("udp_port"):
        raise LiveWorkflowError("BERT endpoint differs from the proposal target")

    port = str(target.get("port"))
    channel = str(target.get("channel"))
    current = bert_client.get_configuration(port, channel)
    if not _configuration_matches(current, before_document):
        raise LiveWorkflowError("proposal is stale because the BERT state changed")

    target_taps, target_upper, target_lower = _state_values(target_document)
    response: str | None = None
    after: BertConfiguration | None = None
    failure: str | None = None
    try:
        response = bert_client.set_fir(
            port,
            channel,
            taps=target_taps,
            upper_nlc=target_upper,
            lower_nlc=target_lower,
        )
        after = bert_client.get_configuration(port, channel)
        if _configuration_matches(after, target_document):
            return {
                "schema_version": SCHEMA_VERSION,
                "proposal_id": proposal_id,
                "status": "VERIFIED",
                "command": proposal.get("command_preview"),
                "response": response,
                "before": _bert_document(current),
                "after": _bert_document(after),
                "error": None,
                "rollback": None,
            }
        failure = "BERT readback does not match the proposed FIR and NLC"
    except Exception as error:  # a timeout can occur after the UDP datagram arrived
        failure = f"target write or readback failed: {error}"

    before_taps, before_upper, before_lower = _state_values(before_document)
    rollback_response: str | None = None
    rollback_after: BertConfiguration | None = None
    rollback_error: str | None = None
    rollback_verified = False
    try:
        rollback_response = bert_client.set_fir(
            port,
            channel,
            taps=before_taps,
            upper_nlc=before_upper,
            lower_nlc=before_lower,
        )
        rollback_after = bert_client.get_configuration(port, channel)
        rollback_verified = _configuration_matches(
            rollback_after, before_document
        )
        if not rollback_verified:
            rollback_error = "rollback readback does not match the saved state"
    except Exception as error:
        rollback_error = f"rollback failed: {error}"

    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "status": "ROLLED_BACK" if rollback_verified else "RECOVERY_REQUIRED",
        "command": proposal.get("command_preview"),
        "response": response,
        "before": _bert_document(current),
        "after": None if after is None else _bert_document(after),
        "error": failure,
        "rollback": {
            "response": rollback_response,
            "after": (
                None if rollback_after is None else _bert_document(rollback_after)
            ),
            "verified": rollback_verified,
            "error": rollback_error,
        },
    }


def _state_documents_match(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    return (
        left.get("port") == right.get("port")
        and str(left.get("channel")) == str(right.get("channel"))
        and _state_values(left) == _state_values(right)
    )


def restore_live_receipt(
    proposal: Mapping[str, object],
    write_receipt: Mapping[str, object],
    bert_client: BertWriter,
    *,
    allow_restore: bool,
    confirmation_id: str | None,
) -> dict[str, object]:
    """Restore a verified write's before state with one roll-forward attempt."""
    if allow_restore is not True:
        raise LiveWorkflowError("live restore requires the explicit --restore flag")
    proposal_id = proposal.get("proposal_id")
    if not isinstance(confirmation_id, str) or confirmation_id != proposal_id:
        raise LiveWorkflowError("confirmation must contain the exact proposal ID")
    _validated_proposal(proposal)

    if write_receipt.get("schema_version") != SCHEMA_VERSION:
        raise LiveWorkflowError("write receipt schema does not match")
    if write_receipt.get("proposal_id") != proposal_id:
        raise LiveWorkflowError("write receipt proposal ID does not match")
    if write_receipt.get("status") != "VERIFIED":
        raise LiveWorkflowError("only a VERIFIED write receipt can be restored")

    proposal_before = _require_mapping(proposal.get("bert_before"), "bert_before")
    proposal_after = _target_state(proposal)
    receipt_before = _require_mapping(write_receipt.get("before"), "receipt before")
    receipt_after = _require_mapping(write_receipt.get("after"), "receipt after")
    if not _state_documents_match(receipt_before, proposal_before):
        raise LiveWorkflowError("write receipt before state differs from proposal")
    if not _state_documents_match(receipt_after, proposal_after):
        raise LiveWorkflowError("write receipt after state differs from proposal")

    target = _require_mapping(proposal.get("target"), "target")
    if getattr(bert_client, "host", None) != target.get("host") or getattr(
        bert_client, "port", None
    ) != target.get("udp_port"):
        raise LiveWorkflowError("BERT endpoint differs from the proposal target")

    port = str(target.get("port"))
    channel = str(target.get("channel"))
    current = bert_client.get_configuration(port, channel)
    if not _configuration_matches(current, receipt_after):
        raise LiveWorkflowError(
            "restore is stale because the BERT state changed after the write receipt"
        )

    restore_taps, restore_upper, restore_lower = _state_values(receipt_before)
    restore_l1 = sum(abs(tap) for tap in restore_taps)
    if restore_l1 > 170:
        raise LiveWorkflowError(
            f"restore FIR L1 sum {restore_l1} exceeds limit 170"
        )
    command = format_setfir_preview(
        port, channel, restore_taps, restore_upper, restore_lower
    )

    response: str | None = None
    restored: BertConfiguration | None = None
    failure: str | None = None
    try:
        response = bert_client.set_fir(
            port,
            channel,
            taps=restore_taps,
            upper_nlc=restore_upper,
            lower_nlc=restore_lower,
        )
        restored = bert_client.get_configuration(port, channel)
        if _configuration_matches(restored, receipt_before):
            return {
                "schema_version": SCHEMA_VERSION,
                "proposal_id": proposal_id,
                "status": "RESTORED",
                "command": command,
                "response": response,
                "from": _bert_document(current),
                "restored": _bert_document(restored),
                "error": None,
                "roll_forward": None,
            }
        failure = "restore readback does not match the saved before state"
    except Exception as error:
        failure = f"restore write or readback failed: {error}"

    after_taps, after_upper, after_lower = _state_values(receipt_after)
    roll_response: str | None = None
    roll_after: BertConfiguration | None = None
    roll_error: str | None = None
    roll_verified = False
    try:
        roll_response = bert_client.set_fir(
            port,
            channel,
            taps=after_taps,
            upper_nlc=after_upper,
            lower_nlc=after_lower,
        )
        roll_after = bert_client.get_configuration(port, channel)
        roll_verified = _configuration_matches(roll_after, receipt_after)
        if not roll_verified:
            roll_error = "roll-forward readback does not match receipt after state"
    except Exception as error:
        roll_error = f"roll-forward failed: {error}"

    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "status": "ROLLED_FORWARD" if roll_verified else "RECOVERY_REQUIRED",
        "command": command,
        "response": response,
        "from": _bert_document(current),
        "restored": None if restored is None else _bert_document(restored),
        "error": failure,
        "roll_forward": {
            "response": roll_response,
            "after": None if roll_after is None else _bert_document(roll_after),
            "verified": roll_verified,
            "error": roll_error,
        },
    }
