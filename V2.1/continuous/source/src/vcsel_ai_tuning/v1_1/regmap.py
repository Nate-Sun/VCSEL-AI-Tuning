"""Safe Script/RegMap discovery and frontend parameter validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .domain import DiscreteRange


SAFE_PARAMETER_NAMES = (
    "ibias_dac",
    "vref",
    "vbias_ref",
    "vga_cur",
    "res_gain",
    "res_bw",
    "gain_ctrl",
    "efin_cur",
    "eq",
)
_SAFE_PARAMETER_SET = frozenset(SAFE_PARAMETER_NAMES)


class MappingStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    EXCLUDED = "EXCLUDED"


class ParameterSelectionError(ValueError):
    pass


def _normalize_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("parameter name must be nonblank")
    return value.strip().lower()


def _strict_integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ScriptEntry:
    name: str
    current_value: int
    order: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name))
        _strict_integer("current_value", self.current_value)
        _strict_integer("order", self.order, minimum=0)


@dataclass(frozen=True, slots=True)
class RegMapField:
    name: str
    channel0_address: int
    channel_stride: int
    bit_start: int
    bits: int
    encoding_min: int
    encoding_max: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name))
        _strict_integer("channel0_address", self.channel0_address, minimum=0)
        _strict_integer("channel_stride", self.channel_stride, minimum=0)
        _strict_integer("bit_start", self.bit_start, minimum=0)
        _strict_integer("bits", self.bits, minimum=1)
        _strict_integer("encoding_min", self.encoding_min, minimum=0)
        _strict_integer("encoding_max", self.encoding_max, minimum=0)
        if self.bit_start + self.bits > 32:
            raise ValueError("bitfield must fit within 32 bits")
        if self.encoding_min > self.encoding_max:
            raise ValueError("encoding_min must not exceed encoding_max")
        if self.encoding_max > (1 << self.bits) - 1:
            raise ValueError("hard-coded encoding range exceeds bitfield width")


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    name: str
    status: MappingStatus
    reason: str | None
    script_order: int | None = None
    current_value: int | None = None
    channel0_address: int | None = None
    channel_stride: int | None = None
    bit_start: int | None = None
    bits: int | None = None
    encoding_min: int | None = None
    encoding_max: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name))
        if self.status is MappingStatus.AVAILABLE:
            required = (
                self.script_order,
                self.current_value,
                self.channel0_address,
                self.channel_stride,
                self.bit_start,
                self.bits,
                self.encoding_min,
                self.encoding_max,
            )
            if any(value is None for value in required):
                raise ValueError("available binding is missing Script/RegMap evidence")
            if self.reason is not None:
                raise ValueError("available binding must not contain an unavailable reason")
        elif not self.reason:
            raise ValueError("unavailable/excluded binding requires a reason")

    @property
    def encoding_range(self) -> tuple[int, int]:
        if self.encoding_min is None or self.encoding_max is None:
            raise ParameterSelectionError(f"parameter {self.name} has no encoding range")
        return self.encoding_min, self.encoding_max

    def address_for_channel(self, channel: int) -> int:
        if self.status is not MappingStatus.AVAILABLE:
            raise ParameterSelectionError(
                f"parameter {self.name} is {self.status.value}: {self.reason}"
            )
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 3:
            raise ValueError("channel must be an integer from 0 through 3")
        assert self.channel0_address is not None
        assert self.channel_stride is not None
        return self.channel0_address + channel * self.channel_stride

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "script_order": self.script_order,
            "current_value": self.current_value,
            "channel0_address": self.channel0_address,
            "channel_stride": self.channel_stride,
            "bit_start": self.bit_start,
            "bits": self.bits,
            "encoding_min": self.encoding_min,
            "encoding_max": self.encoding_max,
        }


@dataclass(frozen=True, slots=True)
class ParameterMapping:
    bindings: tuple[ParameterBinding, ...]
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = [binding.name for binding in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("parameter mapping contains duplicate names")
        object.__setattr__(
            self,
            "content_hash",
            _hash_payload([binding.payload() for binding in self.bindings]),
        )

    def binding(self, name: str) -> ParameterBinding:
        normalized = _normalize_name(name)
        for binding in self.bindings:
            if binding.name == normalized:
                return binding
        raise ParameterSelectionError(f"unknown parameter {normalized}")

    def require_available(self, name: str) -> ParameterBinding:
        binding = self.binding(name)
        if binding.status is not MappingStatus.AVAILABLE:
            raise ParameterSelectionError(
                f"parameter {binding.name} is {binding.status.value}: {binding.reason}"
            )
        return binding


def _available_binding(entry: ScriptEntry, regmap: RegMapField) -> ParameterBinding:
    return ParameterBinding(
        name=entry.name,
        status=MappingStatus.AVAILABLE,
        reason=None,
        script_order=entry.order,
        current_value=entry.current_value,
        channel0_address=regmap.channel0_address,
        channel_stride=regmap.channel_stride,
        bit_start=regmap.bit_start,
        bits=regmap.bits,
        encoding_min=regmap.encoding_min,
        encoding_max=regmap.encoding_max,
    )


def scan_parameter_mapping(
    script_entries: Sequence[ScriptEntry],
    regmap_fields: Sequence[RegMapField],
) -> ParameterMapping:
    """Join normalized Script and RegMap evidence without guessing ambiguities."""

    script_by_name: dict[str, list[ScriptEntry]] = defaultdict(list)
    for entry in script_entries:
        if not isinstance(entry, ScriptEntry):
            raise TypeError("script_entries must contain ScriptEntry values")
        script_by_name[entry.name].append(entry)
    regmap_by_name: dict[str, list[RegMapField]] = defaultdict(list)
    for regmap in regmap_fields:
        if not isinstance(regmap, RegMapField):
            raise TypeError("regmap_fields must contain RegMapField values")
        regmap_by_name[regmap.name].append(regmap)

    bindings: list[ParameterBinding] = []
    ordered_names = sorted(
        script_by_name,
        key=lambda name: min(entry.order for entry in script_by_name[name]),
    )
    for name in ordered_names:
        entries = script_by_name[name]
        script_order = min(entry.order for entry in entries)
        if name not in _SAFE_PARAMETER_SET:
            bindings.append(
                ParameterBinding(
                    name=name,
                    status=MappingStatus.EXCLUDED,
                    reason="parameter is not in the V1.1 safe list",
                    script_order=script_order,
                    current_value=entries[0].current_value,
                )
            )
            continue
        if len(entries) != 1:
            bindings.append(
                ParameterBinding(
                    name=name,
                    status=MappingStatus.UNAVAILABLE,
                    reason=f"multiple Script entries found ({len(entries)})",
                    script_order=script_order,
                )
            )
            continue
        matching_fields = regmap_by_name.get(name, [])
        if not matching_fields:
            bindings.append(
                ParameterBinding(
                    name=name,
                    status=MappingStatus.UNAVAILABLE,
                    reason="missing from RegMap",
                    script_order=script_order,
                    current_value=entries[0].current_value,
                )
            )
            continue
        if len(matching_fields) != 1:
            bindings.append(
                ParameterBinding(
                    name=name,
                    status=MappingStatus.UNAVAILABLE,
                    reason=f"multiple RegMap fields found ({len(matching_fields)})",
                    script_order=script_order,
                    current_value=entries[0].current_value,
                )
            )
            continue
        bindings.append(_available_binding(entries[0], matching_fields[0]))

    present = set(script_by_name)
    for name in SAFE_PARAMETER_NAMES:
        if name not in present:
            bindings.append(
                ParameterBinding(
                    name=name,
                    status=MappingStatus.UNAVAILABLE,
                    reason="missing from Script",
                )
            )
    return ParameterMapping(tuple(bindings))


@dataclass(frozen=True, slots=True)
class ParameterSelection:
    name: str
    minimum: int
    maximum: int
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name))
        try:
            DiscreteRange(self.minimum, self.maximum, self.step)
        except (TypeError, ValueError) as exc:
            raise ParameterSelectionError(str(exc)) from exc

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ParameterSelection":
        allowed = {"name", "min", "max", "step"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ParameterSelectionError(
                "unknown parameter selection fields: " + ", ".join(unknown)
            )
        missing = sorted(allowed - set(payload))
        if missing:
            raise ParameterSelectionError(
                "missing parameter selection fields: " + ", ".join(missing)
            )
        return cls(
            name=payload["name"],  # type: ignore[arg-type]
            minimum=payload["min"],  # type: ignore[arg-type]
            maximum=payload["max"],  # type: ignore[arg-type]
            step=payload["step"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ValidatedParameter:
    name: str
    binding: ParameterBinding
    value_range: DiscreteRange

    @property
    def values(self) -> tuple[int, ...]:
        return self.value_range.values()


def validate_selected_parameters(
    selections: Sequence[ParameterSelection],
    mapping: ParameterMapping,
) -> tuple[ValidatedParameter, ...]:
    selected: list[ValidatedParameter] = []
    seen: set[str] = set()
    for selection in selections:
        if not isinstance(selection, ParameterSelection):
            raise ParameterSelectionError(
                "selected parameters must be ParameterSelection values"
            )
        if selection.name in seen:
            raise ParameterSelectionError(
                f"duplicate selected parameter {selection.name}"
            )
        seen.add(selection.name)
        binding = mapping.require_available(selection.name)
        encoding_min, encoding_max = binding.encoding_range
        if selection.minimum < encoding_min or selection.maximum > encoding_max:
            raise ParameterSelectionError(
                f"parameter {selection.name} exceeds hard-coded encoding range "
                f"[{encoding_min}, {encoding_max}]"
            )
        selected.append(
            ValidatedParameter(
                name=selection.name,
                binding=binding,
                value_range=DiscreteRange(
                    selection.minimum, selection.maximum, selection.step
                ),
            )
        )
    selected.sort(key=lambda item: item.binding.script_order or 0)
    return tuple(selected)
