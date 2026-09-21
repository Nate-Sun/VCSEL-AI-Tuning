"""Dynamic B0 snapshots captured from current hardware readbacks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


LEGACY_CHANNEL_MAPPING_FIELDS = (
    "i2c_channel",
    "bert_port",
    "bert_channel",
    "dca_tdecq_source",
    "dca_outer_er_source",
)
CHANNEL_MAPPING_FIELDS = (
    "i2c_channel",
    "bert_port",
    "bert_channel",
    "dca_tdecq_source",
    "dca_tdecq_equalizer_function",
    "dca_outer_er_source",
)


def _require_nonblank(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank")


def freeze_channel_mapping(
    mapping: Mapping[str, object],
) -> tuple[tuple[str, int | str], ...]:
    if not isinstance(mapping, Mapping):
        raise TypeError("channel mapping must be an object")
    fields = (
        CHANNEL_MAPPING_FIELDS
        if set(mapping) == set(CHANNEL_MAPPING_FIELDS)
        else LEGACY_CHANNEL_MAPPING_FIELDS
    )
    if set(mapping) != set(fields):
        raise ValueError("channel mapping must contain every required field exactly once")
    i2c_channel = mapping["i2c_channel"]
    if isinstance(i2c_channel, bool) or not isinstance(i2c_channel, int):
        raise TypeError("i2c_channel must be an integer")
    if not 0 <= i2c_channel <= 3:
        raise ValueError("i2c_channel must be between 0 and 3")
    for field_name in ("bert_port", "bert_channel", "dca_tdecq_source", "dca_outer_er_source"):
        value = mapping[field_name]
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        _require_nonblank(field_name, value)
    if fields is CHANNEL_MAPPING_FIELDS:
        function = mapping["dca_tdecq_equalizer_function"]
        if isinstance(function, bool) or not isinstance(function, int):
            raise TypeError("dca_tdecq_equalizer_function must be an integer")
        if not 1 <= function <= 16:
            raise ValueError("dca_tdecq_equalizer_function must be between 1 and 16")
    return tuple((field_name, mapping[field_name]) for field_name in fields)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_object(
    name: str, payload: object, fields: tuple[str, ...]
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError(f"{name} must be an object")
    if set(payload) != set(fields):
        raise ValueError(f"{name} fields do not match the persisted schema")
    return payload


def _decode_time(name: str, value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO datetime string")
    try:
        decoded = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO datetime string") from error
    if decoded.tzinfo is None or decoded.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return decoded


def _decode_string_map(name: str, value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be nonblank strings")
        if not isinstance(item, str):
            raise TypeError(f"{name} values must be strings")
        _require_nonblank(name, item)
        result[key] = item
    return result


def _decode_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True, order=True)
class HardwareIdentity:
    kind: str
    resource: str
    identity: str

    def __post_init__(self) -> None:
        _require_nonblank("kind", self.kind)
        _require_nonblank("resource", self.resource)
        _require_nonblank("identity", self.identity)

    def payload(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "resource": self.resource,
            "identity": self.identity,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "HardwareIdentity":
        value = _require_exact_object(
            "hardware identity", payload, ("kind", "resource", "identity")
        )
        return cls(
            kind=value["kind"],
            resource=value["resource"],
            identity=value["identity"],
        )


@dataclass(frozen=True, slots=True)
class DcaFingerprint:
    identity: str
    measurement_sources: tuple[tuple[str, str], ...]
    configuration: tuple[tuple[str, str], ...]
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_nonblank("DCA identity", self.identity)
        object.__setattr__(self, "content_hash", _sha256(self.payload()))

    @classmethod
    def from_mappings(
        cls,
        *,
        identity: str,
        measurement_sources: Mapping[str, str],
        configuration: Mapping[str, str],
    ) -> "DcaFingerprint":
        return cls(
            identity=identity,
            measurement_sources=tuple(sorted(measurement_sources.items())),
            configuration=tuple(sorted(configuration.items())),
        )

    def payload(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "measurement_sources": dict(self.measurement_sources),
            "configuration": dict(self.configuration),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "DcaFingerprint":
        value = _require_exact_object(
            "DCA fingerprint",
            payload,
            ("identity", "measurement_sources", "configuration"),
        )
        identity = value["identity"]
        if not isinstance(identity, str):
            raise TypeError("DCA identity must be a string")
        return cls.from_mappings(
            identity=identity,
            measurement_sources=_decode_string_map(
                "DCA measurement_sources", value["measurement_sources"]
            ),
            configuration=_decode_string_map(
                "DCA configuration", value["configuration"]
            ),
        )


@dataclass(frozen=True, slots=True)
class B0Snapshot:
    b0_id: str
    version: int
    captured_at: datetime
    i2c_fields: tuple[tuple[str, int], ...]
    i2c_raw_registers: tuple[tuple[int, int], ...]
    fir_taps: tuple[int, ...]
    upper_nlc: int
    lower_nlc: int
    bert_pattern: str
    i2c_channel: int
    bert_port: str
    bert_channel: str
    dca_tdecq_source: str
    dca_outer_er_source: str
    dca_fingerprint: DcaFingerprint
    hardware_identities: tuple[HardwareIdentity, ...]
    dca_tdecq_equalizer_function: int | None = None
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_nonblank("b0_id", self.b0_id)
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version < 1:
            raise ValueError("version must be positive")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if len(self.fir_taps) != 7:
            raise ValueError("B0 FIR must contain exactly seven taps")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.fir_taps):
            raise TypeError("B0 FIR taps must be integers")
        _require_nonblank("bert_pattern", self.bert_pattern)
        freeze_channel_mapping(self.channel_mapping_payload())
        measurement_sources = dict(self.dca_fingerprint.measurement_sources)
        if (
            measurement_sources.get("tdecq") != self.dca_tdecq_source
            or measurement_sources.get("outer_er") != self.dca_outer_er_source
        ):
            raise ValueError(
                "B0 channel mapping must match captured DCA measurement sources"
            )
        if self.dca_tdecq_equalizer_function is not None:
            expected_function = f"FUNC{self.dca_tdecq_equalizer_function}"
            configuration = dict(self.dca_fingerprint.configuration)
            if configuration.get("tdecq_equalizer_function") != expected_function:
                raise ValueError(
                    "B0 equalizer function must match captured DCA configuration"
                )
        identity_kinds = {identity.kind for identity in self.hardware_identities}
        if identity_kinds != {"i2c", "bert", "dca"}:
            raise ValueError("B0 requires i2c, bert, and dca hardware identities")
        object.__setattr__(self, "content_hash", _sha256(self.payload()))

    @classmethod
    def capture(
        cls,
        *,
        b0_id: str,
        version: int,
        captured_at: datetime,
        i2c_fields: Mapping[str, int],
        i2c_raw_registers: Mapping[int, int],
        fir_taps: tuple[int, ...],
        upper_nlc: int,
        lower_nlc: int,
        bert_pattern: str,
        i2c_channel: int,
        bert_port: str,
        bert_channel: str,
        dca_tdecq_source: str,
        dca_tdecq_equalizer_function: int | None = None,
        dca_outer_er_source: str,
        dca_fingerprint: DcaFingerprint,
        hardware_identities: Mapping[str, HardwareIdentity],
    ) -> "B0Snapshot":
        for key, identity in hardware_identities.items():
            if key != identity.kind:
                raise ValueError(
                    f"hardware identity key {key!r} does not match kind {identity.kind!r}"
                )
        return cls(
            b0_id=b0_id,
            version=version,
            captured_at=captured_at,
            i2c_fields=tuple(sorted(i2c_fields.items())),
            i2c_raw_registers=tuple(sorted(i2c_raw_registers.items())),
            fir_taps=tuple(fir_taps),
            upper_nlc=upper_nlc,
            lower_nlc=lower_nlc,
            bert_pattern=bert_pattern,
            i2c_channel=i2c_channel,
            bert_port=bert_port,
            bert_channel=bert_channel,
            dca_tdecq_source=dca_tdecq_source,
            dca_tdecq_equalizer_function=dca_tdecq_equalizer_function,
            dca_outer_er_source=dca_outer_er_source,
            dca_fingerprint=dca_fingerprint,
            hardware_identities=tuple(
                sorted(hardware_identities.values(), key=lambda item: item.kind)
            ),
        )

    def channel_mapping_payload(self) -> dict[str, int | str]:
        mapping: dict[str, int | str] = {
            "i2c_channel": self.i2c_channel,
            "bert_port": self.bert_port,
            "bert_channel": self.bert_channel,
            "dca_tdecq_source": self.dca_tdecq_source,
            "dca_outer_er_source": self.dca_outer_er_source,
        }
        if self.dca_tdecq_equalizer_function is not None:
            mapping["dca_tdecq_equalizer_function"] = (
                self.dca_tdecq_equalizer_function
            )
        return mapping

    def payload(self) -> dict[str, object]:
        return {
            "b0_id": self.b0_id,
            "version": self.version,
            "captured_at": self.captured_at.isoformat(),
            "i2c_fields": dict(self.i2c_fields),
            "i2c_raw_registers": [list(item) for item in self.i2c_raw_registers],
            "channel_mapping": self.channel_mapping_payload(),
            "fir": {
                "taps": list(self.fir_taps),
                "upper_nlc": self.upper_nlc,
                "lower_nlc": self.lower_nlc,
                "pattern": self.bert_pattern,
                "port": self.bert_port,
                "channel": self.bert_channel,
            },
            "dca_fingerprint": self.dca_fingerprint.payload(),
            "hardware_identities": [
                identity.payload() for identity in self.hardware_identities
            ],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "B0Snapshot":
        value = _require_exact_object(
            "B0 payload",
            payload,
            (
                "b0_id",
                "version",
                "captured_at",
                "i2c_fields",
                "i2c_raw_registers",
                "channel_mapping",
                "fir",
                "dca_fingerprint",
                "hardware_identities",
            ),
        )
        i2c_fields_payload = value["i2c_fields"]
        if type(i2c_fields_payload) is not dict:
            raise TypeError("B0 i2c_fields must be an object")
        i2c_fields: dict[str, int] = {}
        for name, item in i2c_fields_payload.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("B0 i2c_fields keys must be nonblank strings")
            i2c_fields[name] = _decode_int(f"B0 i2c field {name}", item)

        raw_registers_payload = value["i2c_raw_registers"]
        if type(raw_registers_payload) is not list:
            raise TypeError("B0 i2c_raw_registers must be an array")
        raw_registers: dict[int, int] = {}
        for item in raw_registers_payload:
            if type(item) is not list or len(item) != 2:
                raise ValueError("each B0 raw register must be a two-item array")
            address = _decode_int("B0 raw register address", item[0])
            register_value = _decode_int("B0 raw register value", item[1])
            if address in raw_registers:
                raise ValueError("B0 raw register addresses must be unique")
            raw_registers[address] = register_value

        channel_mapping_payload = value["channel_mapping"]
        if type(channel_mapping_payload) is not dict:
            raise TypeError("B0 channel mapping must be an object")
        mapping_fields = (
            CHANNEL_MAPPING_FIELDS
            if set(channel_mapping_payload) == set(CHANNEL_MAPPING_FIELDS)
            else LEGACY_CHANNEL_MAPPING_FIELDS
        )
        channel_mapping = _require_exact_object(
            "B0 channel mapping", channel_mapping_payload, mapping_fields
        )
        freeze_channel_mapping(channel_mapping)
        fir = _require_exact_object(
            "B0 FIR",
            value["fir"],
            ("taps", "upper_nlc", "lower_nlc", "pattern", "port", "channel"),
        )
        taps_payload = fir["taps"]
        if type(taps_payload) is not list:
            raise TypeError("B0 FIR taps must be an array")
        taps = tuple(_decode_int("B0 FIR tap", item) for item in taps_payload)
        if fir["port"] != channel_mapping["bert_port"]:
            raise ValueError("B0 FIR port does not match channel mapping")
        if fir["channel"] != channel_mapping["bert_channel"]:
            raise ValueError("B0 FIR channel does not match channel mapping")

        identities_payload = value["hardware_identities"]
        if type(identities_payload) is not list:
            raise TypeError("B0 hardware_identities must be an array")
        identities = tuple(
            HardwareIdentity.from_payload(item) for item in identities_payload
        )
        if len(identities) != 3 or len({item.kind for item in identities}) != 3:
            raise ValueError("B0 requires exactly one identity per hardware kind")

        b0_id = value["b0_id"]
        if not isinstance(b0_id, str):
            raise TypeError("B0 b0_id must be a string")
        pattern = fir["pattern"]
        if not isinstance(pattern, str):
            raise TypeError("B0 FIR pattern must be a string")
        return cls.capture(
            b0_id=b0_id,
            version=_decode_int("B0 version", value["version"]),
            captured_at=_decode_time("B0 captured_at", value["captured_at"]),
            i2c_fields=i2c_fields,
            i2c_raw_registers=raw_registers,
            fir_taps=taps,
            upper_nlc=_decode_int("B0 upper_nlc", fir["upper_nlc"]),
            lower_nlc=_decode_int("B0 lower_nlc", fir["lower_nlc"]),
            bert_pattern=pattern,
            i2c_channel=channel_mapping["i2c_channel"],
            bert_port=channel_mapping["bert_port"],
            bert_channel=channel_mapping["bert_channel"],
            dca_tdecq_source=channel_mapping["dca_tdecq_source"],
            dca_tdecq_equalizer_function=(
                None
                if "dca_tdecq_equalizer_function" not in channel_mapping
                else _decode_int(
                    "B0 TDECQ equalizer function",
                    channel_mapping["dca_tdecq_equalizer_function"],
                )
            ),
            dca_outer_er_source=channel_mapping["dca_outer_er_source"],
            dca_fingerprint=DcaFingerprint.from_payload(value["dca_fingerprint"]),
            hardware_identities={item.kind: item for item in identities},
        )
