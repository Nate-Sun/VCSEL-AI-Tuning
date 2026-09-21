"""Validated configuration for the currently connected lab hardware."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, AddressValueError
from math import isfinite
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DcaConfig:
    resource: str
    expected_idn_prefix: str
    function: int
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class BertConfig:
    ip: str
    local_ip: str
    udp_port: int
    port: str
    channel: str
    timeout_s: float


@dataclass(frozen=True, slots=True)
class I2cConfig:
    port: str
    baudrate: int
    device_address: int
    chip_id_register: int
    expected_chip_id: int
    chip_revision_register: int
    expected_chip_revision: int


@dataclass(frozen=True, slots=True)
class ErMeasurementConfig:
    source: str
    value_query: str
    source_query: str
    status_query: str
    details_query: str
    units_query: str


@dataclass(frozen=True, slots=True)
class TdecqMeasurementConfig:
    source: str


@dataclass(frozen=True, slots=True)
class ParameterFieldConfig:
    name: str
    channel0_address: int
    bit_start: int
    bits: int
    safety_min: int | None = None
    safety_max: int | None = None
    max_step: int | None = None

    @property
    def encoded_max(self) -> int:
        return (1 << self.bits) - 1

    def address_for_channel(self, channel: int) -> int:
        if type(channel) is not int or not 0 <= channel <= 3:
            raise ValueError("channel must be an integer from 0 through 3")
        return self.channel0_address + (channel << 8)


@dataclass(frozen=True, slots=True)
class CharacterizationConfig:
    default_channel: int
    baseline_repeats: int
    measurement_timeout_s: float
    poll_interval_s: float
    meaningful_er_floor_db: float | None
    meaningful_tdecq_floor_db: float | None
    tdecq: TdecqMeasurementConfig
    er: ErMeasurementConfig
    parameters: tuple[ParameterFieldConfig, ...]


@dataclass(frozen=True, slots=True)
class LiveHardwareConfig:
    dca: DcaConfig
    bert: BertConfig
    i2c: I2cConfig
    characterization: CharacterizationConfig


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required(mapping: dict[str, Any], field: str, path: str) -> Any:
    if field not in mapping:
        raise ValueError(f"{path}.{field} is required")
    return mapping[field]


def _required_section(document: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in document:
        raise ValueError(f"required section '{section}' is missing")
    return _mapping(document[section], section)


def _string(mapping: dict[str, Any], field: str, path: str) -> str:
    value = _required(mapping, field, path)
    if not isinstance(value, str):
        raise ValueError(f"{path}.{field} has invalid type; expected string")
    value = value.strip()
    if not value:
        raise ValueError(f"{path}.{field} must not be empty")
    return value


def _integer(mapping: dict[str, Any], field: str, path: str) -> int:
    value = _required(mapping, field, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{field} has invalid type; expected integer")
    return value


def _integer_in_range(
    mapping: dict[str, Any],
    field: str,
    path: str,
    minimum: int,
    maximum: int,
) -> int:
    value = _integer(mapping, field, path)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{path}.{field} must be between {minimum} and {maximum}"
        )
    return value


def _positive_integer(mapping: dict[str, Any], field: str, path: str) -> int:
    value = _integer(mapping, field, path)
    if value <= 0:
        raise ValueError(f"{path}.{field} must be greater than zero")
    return value


def _positive_number(mapping: dict[str, Any], field: str, path: str) -> float:
    value = _required(mapping, field, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{field} has invalid type; expected number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{path}.{field} must be greater than zero")
    return result


def _nullable_nonnegative_number(
    mapping: dict[str, Any], field: str, path: str
) -> float | None:
    value = _required(mapping, field, path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{field} has invalid type; expected number or null")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"{path}.{field} must be non-negative or null")
    return result


def _ipv4(mapping: dict[str, Any], field: str, path: str) -> str:
    value = _string(mapping, field, path)
    try:
        return str(IPv4Address(value))
    except AddressValueError as exc:
        raise ValueError(f"{path}.{field} must be a valid IPv4 address") from exc


def _channel(mapping: dict[str, Any]) -> str:
    value = _required(mapping, "channel", "bert")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("bert.channel has invalid type; expected string or integer")
    result = str(value).strip()
    if not result:
        raise ValueError("bert.channel must not be empty")
    return result


def _measurement_source(
    mapping: dict[str, Any], field: str, path: str
) -> str:
    value = _string(mapping, field, path)
    patterns = (
        (r"CHAN(?:NEL)?([1-8])([A-D])", lambda m: f"CHAN{m.group(1)}{m.group(2).upper()}"),
        (r"FUNC(?:TION)?([0-9]+)", lambda m: f"FUNC{int(m.group(1), 10)}"),
        (r"DIFF([1-4])", lambda m: f"DIFF{m.group(1)}"),
        (r"CGM(?:EMORY)?([1-8])", lambda m: f"CGMEMORY{m.group(1)}"),
    )
    for pattern, formatter in patterns:
        match = re.fullmatch(pattern, value, re.IGNORECASE)
        if match is None:
            continue
        canonical = formatter(match)
        if canonical.startswith("FUNC") and not 1 <= int(canonical[4:]) <= 16:
            break
        return canonical
    raise ValueError(
        f"{path}.{field} must identify a supported waveform"
    )


_PARAMETER_NAMES = (
    "ibias_dac",
    "efin_cur",
    "vga_cur",
    "eq",
    "res_gain",
    "res_bw",
    "gain_ctrl",
    "vbias_ref",
    "vref",
)


def _characterization_config(document: dict[str, Any]) -> CharacterizationConfig:
    section = _required_section(document, "characterization")
    tdecq = _required_section(section, "tdecq")
    er = _required_section(section, "er")
    parameters_document = _required_section(section, "parameters")
    safety_document = _required_section(section, "safety_limits")
    if set(parameters_document) != set(_PARAMETER_NAMES):
        raise ValueError(
            "characterization.parameters must define exactly the nine VD1124 fields"
        )
    unknown_safety = set(safety_document) - set(_PARAMETER_NAMES)
    if unknown_safety:
        raise ValueError(
            "characterization.safety_limits contains unknown parameters: "
            + ", ".join(sorted(unknown_safety))
        )

    parameters: list[ParameterFieldConfig] = []
    for name in _PARAMETER_NAMES:
        path = f"characterization.parameters.{name}"
        field = _mapping(parameters_document[name], path)
        address = _integer_in_range(field, "address", path, 0, 0xFFFF)
        bit_start = _integer_in_range(field, "bit_start", path, 0, 7)
        bits = _integer_in_range(field, "bits", path, 1, 8)
        if bit_start + bits > 8:
            raise ValueError(f"{path} bit field must fit within one byte")

        safety_min = safety_max = max_step = None
        if name in safety_document:
            safety_path = f"characterization.safety_limits.{name}"
            safety = _mapping(safety_document[name], safety_path)
            safety_min = _integer(safety, "minimum", safety_path)
            safety_max = _integer(safety, "maximum", safety_path)
            max_step = _positive_integer(safety, "max_step", safety_path)
            encoded_max = (1 << bits) - 1
            if not 0 <= safety_min <= safety_max <= encoded_max:
                raise ValueError(
                    f"{safety_path} must fit the field encoding and minimum <= maximum"
                )
        parameters.append(
            ParameterFieldConfig(
                name,
                address,
                bit_start,
                bits,
                safety_min,
                safety_max,
                max_step,
            )
        )

    default_channel = _integer_in_range(
        section, "default_channel", "characterization", 0, 3
    )
    timeout = _positive_number(
        section, "measurement_timeout_s", "characterization"
    )
    poll_interval = _positive_number(
        section, "poll_interval_s", "characterization"
    )
    if poll_interval > timeout:
        raise ValueError(
            "characterization.poll_interval_s must not exceed measurement_timeout_s"
        )
    tdecq_source = _measurement_source(
        tdecq, "source", "characterization.tdecq"
    )
    if tdecq_source != "FUNC1":
        raise ValueError("characterization.tdecq.source must be FUNC1")
    er_source = _measurement_source(er, "source", "characterization.er")
    if er_source != "CHAN1A":
        raise ValueError("characterization.er.source must be CHAN1A")
    return CharacterizationConfig(
        default_channel=default_channel,
        baseline_repeats=_positive_integer(
            section, "baseline_repeats", "characterization"
        ),
        measurement_timeout_s=timeout,
        poll_interval_s=poll_interval,
        meaningful_er_floor_db=_nullable_nonnegative_number(
            section, "meaningful_er_floor_db", "characterization"
        ),
        meaningful_tdecq_floor_db=_nullable_nonnegative_number(
            section, "meaningful_tdecq_floor_db", "characterization"
        ),
        tdecq=TdecqMeasurementConfig(source=tdecq_source),
        er=ErMeasurementConfig(
            source=er_source,
            value_query=_string(er, "value_query", "characterization.er"),
            source_query=_string(er, "source_query", "characterization.er"),
            status_query=_string(er, "status_query", "characterization.er"),
            details_query=_string(er, "details_query", "characterization.er"),
            units_query=_string(er, "units_query", "characterization.er"),
        ),
        parameters=tuple(parameters),
    )


def load_live_hardware_config(path: str | Path) -> LiveHardwareConfig:
    """Load and validate a live-hardware YAML configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "root")

    dca = _required_section(document, "dca")
    bert = _required_section(document, "bert")
    i2c = _required_section(document, "i2c")
    dca_function = _positive_integer(dca, "function", "dca")
    if dca_function != 1:
        raise ValueError("dca.function must be 1")

    return LiveHardwareConfig(
        dca=DcaConfig(
            resource=_string(dca, "resource", "dca"),
            expected_idn_prefix=_string(dca, "expected_idn_prefix", "dca"),
            function=dca_function,
            timeout_ms=_positive_integer(dca, "timeout_ms", "dca"),
        ),
        bert=BertConfig(
            ip=_ipv4(bert, "ip", "bert"),
            local_ip=_ipv4(bert, "local_ip", "bert"),
            udp_port=_integer_in_range(bert, "udp_port", "bert", 1, 65535),
            port=_string(bert, "port", "bert"),
            channel=_channel(bert),
            timeout_s=_positive_number(bert, "timeout_s", "bert"),
        ),
        i2c=I2cConfig(
            port=_string(i2c, "port", "i2c"),
            baudrate=_positive_integer(i2c, "baudrate", "i2c"),
            device_address=_integer_in_range(
                i2c, "device_address", "i2c", 0, 0xFF
            ),
            chip_id_register=_integer_in_range(
                i2c, "chip_id_register", "i2c", 0, 0xFFFF
            ),
            expected_chip_id=_integer_in_range(
                i2c, "expected_chip_id", "i2c", 0, 0xFF
            ),
            chip_revision_register=_integer_in_range(
                i2c, "chip_revision_register", "i2c", 0, 0xFFFF
            ),
            expected_chip_revision=_integer_in_range(
                i2c, "expected_chip_revision", "i2c", 0, 0xFF
            ),
        ),
        characterization=_characterization_config(document),
    )
