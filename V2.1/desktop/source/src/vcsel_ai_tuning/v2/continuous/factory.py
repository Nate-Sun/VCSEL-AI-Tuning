"""Explicit continuous dependency assembly with a hard LIVE contact gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .controller import ContinuousOptimizationController
from .fake import SystemClock, fake_scenario
from .models import ContinuousRequest, ExecutionMode
from .store import ContinuousCampaignStore


class LiveContactAuthorizationError(PermissionError):
    pass


_CHECKED_IN_CAMPAIGN_PLACEHOLDER = "REPLACE_WITH_FRESH_CAMPAIGN_ID"
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_LIVE_OPERATIONS = frozenset({"PREFLIGHT", "COMMISSION"})


def _freeze_live_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_live_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_live_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_live_value(item) for item in value)
    return value


def _live_value_is_frozen(value: object) -> bool:
    if isinstance(value, _MAPPING_PROXY_TYPE):
        return all(_live_value_is_frozen(item) for item in value.values())
    if isinstance(value, (tuple, frozenset)):
        return all(_live_value_is_frozen(item) for item in value)
    return not isinstance(value, (Mapping, list, set))


def _validated_operator_controls(
    max_rounds: object,
    target_tdecq_db: object,
) -> tuple[int, float]:
    if type(max_rounds) is not int or max_rounds <= 0:
        raise ValueError("continuous commission rounds must be a positive integer")
    if isinstance(target_tdecq_db, bool) or not isinstance(
        target_tdecq_db, (int, float)
    ):
        raise ValueError("continuous commission target must be a positive finite number")
    try:
        target = float(target_tdecq_db)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "continuous commission target must be a positive finite number"
        ) from error
    if not math.isfinite(target) or target <= 0:
        raise ValueError("continuous commission target must be a positive finite number")
    return max_rounds, target


@dataclass(frozen=True, slots=True)
class ContinuousDependencies:
    mode: ExecutionMode
    controller: ContinuousOptimizationController | None
    request: ContinuousRequest | None
    contact_permitted: bool
    live_activation: "LazyLiveActivation | None" = None
    shutdown: Any | None = None
    preflight_metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _FrozenLiveScope:
    execution_mode: ExecutionMode
    allow_device_contact: bool
    config: Mapping[str, object]
    result_dir: Path


def _validated_live_scope(
    scope: _FrozenLiveScope,
    *,
    operation: str,
) -> Mapping[str, object]:
    if (
        type(scope) is not _FrozenLiveScope
        or scope.execution_mode is not ExecutionMode.LIVE
    ):
        raise LiveContactAuthorizationError("LIVE activation requires execution_mode LIVE")
    if operation not in _LIVE_OPERATIONS:
        raise LiveContactAuthorizationError("LIVE operation must be PREFLIGHT or COMMISSION")
    if scope.allow_device_contact is not True:
        raise LiveContactAuthorizationError("LIVE requires --allow-device-contact")
    if not isinstance(scope.config, _MAPPING_PROXY_TYPE):
        raise LiveContactAuthorizationError("LIVE config snapshot must remain immutable")

    runtime_config = scope.config.get("runtime_config")
    campaign_id = scope.config.get("campaign_id")
    if not isinstance(runtime_config, str) or not runtime_config.strip():
        raise LiveContactAuthorizationError("live_config requires runtime_config")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise LiveContactAuthorizationError("live_config requires campaign_id")
    if campaign_id.strip() == _CHECKED_IN_CAMPAIGN_PLACEHOLDER:
        raise LiveContactAuthorizationError(
            "live_config campaign_id is the checked-in placeholder; "
            "replace it with a fresh campaign ID"
        )
    _validated_operator_controls(
        scope.config.get("max_rounds"),
        scope.config.get("target_tdecq_db"),
    )
    if not _live_value_is_frozen(scope.config):
        raise LiveContactAuthorizationError("LIVE config snapshot must remain immutable")
    return scope.config


@dataclass(slots=True)
class LazyLiveActivation:
    """Store validated paths only; imports and device contact happen on activate()."""

    _frozen_scope: _FrozenLiveScope
    activated: bool = False

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_frozen_scope":
            try:
                object.__getattribute__(self, name)
            except AttributeError:
                pass
            else:
                raise AttributeError("LIVE activation scope cannot be replaced")
        object.__setattr__(self, name, value)

    @property
    def config(self) -> Mapping[str, object]:
        return self._frozen_scope.config

    @property
    def result_dir(self) -> Path:
        return self._frozen_scope.result_dir

    def activate(self) -> ContinuousDependencies:
        if self.activated:
            raise RuntimeError("LIVE activation is single-use")
        config = _validated_live_scope(
            self._frozen_scope,
            operation="COMMISSION",
        )
        self.activated = True
        # Import by name only after the explicit contact gates. This keeps the
        # provider-neutral V2 package statically isolated from upper hardware code.
        module = import_module("vcsel_ai_tuning.continuous_live")
        return module.activate_continuous_live(
            config,
            self.result_dir,
        )

    def preflight(self) -> Mapping[str, object]:
        if self.activated:
            raise RuntimeError("LIVE activation is single-use")
        config = _validated_live_scope(
            self._frozen_scope,
            operation="PREFLIGHT",
        )
        self.activated = True
        module = import_module("vcsel_ai_tuning.continuous_live")
        return module.preflight_continuous_live(config, self.result_dir)


def build_continuous_dependencies(
    *,
    mode: ExecutionMode,
    result_dir: Path | str,
    scenario: str = "happy",
    max_rounds: int = 3,
    target_tdecq_db: float = 2.0,
    allow_device_contact: bool = False,
    live_config: Mapping[str, object] | None = None,
) -> ContinuousDependencies:
    root = Path(result_dir)
    max_rounds, target_tdecq_db = _validated_operator_controls(
        max_rounds,
        target_tdecq_db,
    )
    if mode is ExecutionMode.LIVE:
        if allow_device_contact is not True:
            raise LiveContactAuthorizationError(
                "LIVE requires --allow-device-contact"
            )
        if not isinstance(live_config, Mapping):
            raise LiveContactAuthorizationError(
                "LIVE requires explicit live_config"
            )
        frozen_scope = _FrozenLiveScope(
            execution_mode=mode,
            allow_device_contact=allow_device_contact,
            config=MappingProxyType(
                {
                    key: _freeze_live_value(value)
                    for key, value in (
                        dict(live_config)
                        | {
                            "max_rounds": max_rounds,
                            "target_tdecq_db": target_tdecq_db,
                        }
                    ).items()
                }
            ),
            result_dir=root,
        )
        _validated_live_scope(frozen_scope, operation="PREFLIGHT")
        # The live assembler is deliberately lazy. No provider/hardware import,
        # environment secret lookup, discovery, or connection occurs here.
        return ContinuousDependencies(
            mode=mode,
            controller=None,
            request=None,
            contact_permitted=True,
            live_activation=LazyLiveActivation(
                _frozen_scope=frozen_scope,
            ),
        )
    provider, execution, request = fake_scenario(
        root,
        scenario,
        max_rounds=max_rounds,
        target_tdecq_db=target_tdecq_db,
    )
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(root),
        clock=SystemClock(),
    )
    return ContinuousDependencies(
        mode=mode,
        controller=controller,
        request=request,
        contact_permitted=False,
        live_activation=None,
        preflight_metadata=None,
    )
