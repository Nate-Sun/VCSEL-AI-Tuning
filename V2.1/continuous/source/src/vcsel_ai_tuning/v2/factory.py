"""Explicit, local registry for selecting a V2 optimization brain."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self
from unicodedata import category

import yaml
from pydantic import BeforeValidator, Field, FiniteFloat, StrictBool, model_validator
from yaml.constructor import ConstructorError

from .brain import OptimizationBrain, _claim_factory_gate_creator
from .contracts import BrainMode, FrozenModel


def _wire_brain_mode(value: object) -> BrainMode:
    """Accept only an explicit BrainMode or its YAML string representation."""
    if isinstance(value, BrainMode):
        return value
    if type(value) is str:
        return BrainMode(value)
    raise ValueError("brain mode must be a BrainMode or its string value")


def _wire_brain_modes(value: object) -> tuple[object, ...]:
    """Accept YAML lists and Python tuples, preserving strict item validation."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise ValueError("enabled_brain_modes must be a list or tuple")


def _strict_confidence(value: object) -> float:
    if type(value) is not float:
        raise ValueError("minimum_confidence must be a finite float")
    return value


WireBrainMode = Annotated[BrainMode, BeforeValidator(_wire_brain_mode)]
WireBrainModes = Annotated[
    tuple[WireBrainMode, ...],
    BeforeValidator(_wire_brain_modes),
]
MinimumConfidence = Annotated[
    FiniteFloat,
    BeforeValidator(_strict_confidence),
    Field(ge=0.0, le=1.0),
]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys before construction."""

    def construct_mapping(self, node, deep: bool = False):
        self.flatten_mapping(node)
        keys: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in keys
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


class BrainFactoryConfig(FrozenModel):
    """Strict V2 brain selection, independent of the legacy runtime config."""

    brain_mode: WireBrainMode
    enabled_brain_modes: WireBrainModes
    minimum_confidence: MinimumConfidence = 0.7

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.enabled_brain_modes:
            raise ValueError("enabled_brain_modes must not be empty")
        if len(set(self.enabled_brain_modes)) != len(self.enabled_brain_modes):
            raise ValueError("enabled_brain_modes must be unique")
        if self.brain_mode not in self.enabled_brain_modes:
            raise ValueError("brain_mode must be enabled")
        return self


CapabilityReason = Literal["NOT_INSTALLED", "NOT_ENABLED"]


class BrainCapability(FrozenModel):
    """A self-consistent availability report for one supported mode."""

    brain_mode: BrainMode
    installed: StrictBool
    enabled: StrictBool
    available: StrictBool
    reason: CapabilityReason | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        expected_available = self.installed and self.enabled
        if self.available != expected_available:
            raise ValueError("available must equal installed and enabled")
        expected_reason: CapabilityReason | None
        if not self.installed:
            expected_reason = "NOT_INSTALLED"
        elif not self.enabled:
            expected_reason = "NOT_ENABLED"
        else:
            expected_reason = None
        if self.reason != expected_reason:
            raise ValueError("reason must match installed and enabled")
        return self


class V2ConfigDocument(FrozenModel):
    """The intentionally isolated V2 configuration document."""

    v2: BrainFactoryConfig


class BrainUnavailableError(RuntimeError):
    """Raised when the explicitly selected brain cannot be created."""


BrainBuilder = Callable[[BrainFactoryConfig], OptimizationBrain]


def _safe_identity(value: object) -> str | None:
    if type(value) is not str or not value or value != value.strip():
        return None
    components = value.split("|")
    if any(
        not component
        or component != component.strip()
        or any(category(character) in {"Cc", "Cf", "Cs"} for character in component)
        for component in components
    ):
        return None
    return value


def _validated_brain(result: object) -> tuple[OptimizationBrain, str, str, object]:
    """Close runtime Protocol gaps without exposing provider-side exceptions."""
    try:
        provider_key = result.provider_key
        provider_version = result.provider_version
        analyze = result.analyze
    except Exception:
        raise TypeError("builder must return a valid OptimizationBrain") from None
    if (
        _safe_identity(provider_key) is None
        or _safe_identity(provider_version) is None
        or not callable(analyze)
    ):
        raise TypeError("builder must return a valid OptimizationBrain")
    return result, provider_key, provider_version, analyze


def _new_factory_create_method(create_factory_gated_brain):
    def create(self, config: BrainFactoryConfig) -> OptimizationBrain:
        config = self._require_config(config)
        selected_mode = config.brain_mode
        minimum_confidence = config.minimum_confidence
        if selected_mode not in config.enabled_brain_modes:
            raise BrainUnavailableError(f"brain mode {selected_mode.value} is not enabled")
        builder = self._builders.get(selected_mode)
        if builder is None:
            raise BrainUnavailableError(
                f"brain mode {selected_mode.value} is not installed"
            )
        builder_config = BrainFactoryConfig.model_validate(
            config.model_dump(round_trip=True, warnings=False), strict=True
        )
        brain, provider_key, provider_version, analyze = _validated_brain(builder(builder_config))
        # Generic builders are single-provider: both output identities must equal
        # this frozen top-level identity. Multi-provider builders use native
        # ComposedBrain or MultimodalApiBrain so component provenance is explicit.
        return create_factory_gated_brain(
            brain,
            selected_mode,
            minimum_confidence,
            provider_key,
            provider_version,
            analyze,
        )

    return create


class BrainFactory:
    """Per-instance explicit registry; registration never enables a mode."""

    def __init__(self) -> None:
        self._builders: dict[BrainMode, BrainBuilder] = {}

    def register(self, mode: BrainMode, builder: BrainBuilder) -> None:
        if not isinstance(mode, BrainMode):
            raise TypeError("mode must be a BrainMode")
        if not callable(builder):
            raise TypeError("builder must be callable")
        if mode in self._builders:
            raise ValueError(f"builder already registered for {mode.value}")
        self._builders[mode] = builder

    def capabilities(
        self,
        config: BrainFactoryConfig,
    ) -> tuple[BrainCapability, ...]:
        config = self._require_config(config)
        return tuple(
            BrainCapability(
                brain_mode=mode,
                installed=mode in self._builders,
                enabled=mode in config.enabled_brain_modes,
                available=(mode in self._builders and mode in config.enabled_brain_modes),
                reason=(
                    "NOT_INSTALLED"
                    if mode not in self._builders
                    else "NOT_ENABLED"
                    if mode not in config.enabled_brain_modes
                    else None
                ),
            )
            for mode in BrainMode
        )

    create = _new_factory_create_method(_claim_factory_gate_creator())

    @staticmethod
    def _require_config(config: object) -> BrainFactoryConfig:
        if not isinstance(config, BrainFactoryConfig):
            raise TypeError("config must be a BrainFactoryConfig")
        return BrainFactoryConfig.model_validate(
            config.model_dump(round_trip=True, warnings=False), strict=True
        )


del _claim_factory_gate_creator
del _new_factory_create_method


def load_v2_config(path: Path | str) -> BrainFactoryConfig:
    """Load the standalone V2 YAML document without environment access."""
    raw = yaml.load(
        Path(path).read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("V2 configuration root must be a mapping")
    return V2ConfigDocument.model_validate(raw).v2
