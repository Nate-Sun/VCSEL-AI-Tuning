"""Immutable, single-confirmation authorization for an entire V1.1 task."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from .baseline import freeze_channel_mapping
from .domain import DiscreteRange, TaskType


class AuthorizationMode(StrEnum):
    FULL_CAMPAIGN = "FULL_CAMPAIGN"


class OptimizationDepth(StrEnum):
    FAST = "FAST"
    STANDARD = "STANDARD"
    DEEP = "DEEP"
    ADVANCED = "ADVANCED"


class RollbackPolicy(StrEnum):
    RESTORE_B0 = "RESTORE_B0"


class AuthorizationDriftError(RuntimeError):
    def __init__(self, mismatches: tuple[str, ...]) -> None:
        self.mismatches = mismatches
        super().__init__(
            "authorization inputs changed: " + ", ".join(self.mismatches)
        )


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hash") from exc


def _content_hash(payload: object) -> str:
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


def _decode_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _freeze_execution_policy(
    value: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("execution_policy must be an object")
    frozen: list[tuple[str, object]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("execution_policy names must be nonblank strings")
        if item is not None and type(item) not in {str, int, float, bool}:
            raise TypeError("execution_policy values must be JSON scalars")
        if isinstance(item, float):
            try:
                json.dumps(item, allow_nan=False)
            except ValueError as error:
                raise ValueError(
                    "execution_policy numbers must be finite"
                ) from error
        frozen.append((key, item))
    if len({key for key, _item in frozen}) != len(frozen):
        raise ValueError("execution_policy names must be unique")
    return tuple(sorted(frozen))


@dataclass(frozen=True, slots=True)
class AuthorizedParameter:
    name: str
    value_range: DiscreteRange

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must be nonblank")

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "minimum": self.value_range.minimum,
            "maximum": self.value_range.maximum,
            "step": self.value_range.step,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "AuthorizedParameter":
        value = _require_exact_object(
            "authorized parameter",
            payload,
            ("name", "minimum", "maximum", "step"),
        )
        name = value["name"]
        if not isinstance(name, str):
            raise TypeError("parameter name must be a string")
        return cls(
            name=name,
            value_range=DiscreteRange(
                minimum=_decode_int("parameter minimum", value["minimum"]),
                maximum=_decode_int("parameter maximum", value["maximum"]),
                step=_decode_int("parameter step", value["step"]),
            ),
        )


@dataclass(frozen=True, slots=True)
class FullCampaignAuthorization:
    authorization_id: str
    task_id: str
    task_type: TaskType
    b0_hash: str
    hardware_identities: tuple[tuple[str, str], ...]
    mapping_hash: str
    channel_mapping: tuple[tuple[str, int | str], ...]
    dca_config_fingerprint: str
    parameters: tuple[AuthorizedParameter, ...]
    depth: OptimizationDepth
    candidate_budget: int
    seed: int
    max_points: int | None
    rollback_policy: RollbackPolicy
    issued_at: datetime
    expires_at: datetime
    advanced_iteration_limit: int | None = None
    execution_policy: tuple[tuple[str, object], ...] | None = None
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.authorization_id.strip():
            raise ValueError("authorization_id must be nonblank")
        if not self.task_id.strip():
            raise ValueError("task_id must be nonblank")
        _require_sha256("b0_hash", self.b0_hash)
        _require_sha256("mapping_hash", self.mapping_hash)
        object.__setattr__(
            self,
            "channel_mapping",
            freeze_channel_mapping(dict(self.channel_mapping)),
        )
        _require_sha256("dca_config_fingerprint", self.dca_config_fingerprint)
        if any(
            not isinstance(parameter, AuthorizedParameter)
            for parameter in self.parameters
        ):
            raise TypeError("parameters must contain AuthorizedParameter objects")
        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("authorization parameter names must be unique")
        if isinstance(self.candidate_budget, bool) or self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.max_points is not None and (
            isinstance(self.max_points, bool) or self.max_points <= 0
        ):
            raise ValueError("max_points must be positive when provided")
        if self.issued_at.tzinfo is None or self.issued_at.utcoffset() is None:
            raise ValueError("issued_at must be timezone-aware")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        if self.depth is OptimizationDepth.ADVANCED:
            if self.advanced_iteration_limit is None:
                raise ValueError("Advanced depth requires an explicit iteration limit")
            if self.advanced_iteration_limit <= 0:
                raise ValueError("Advanced iteration limit must be positive")
        elif self.advanced_iteration_limit is not None:
            raise ValueError("advanced_iteration_limit is only valid for Advanced depth")
        object.__setattr__(
            self,
            "execution_policy",
            _freeze_execution_policy(
                None
                if self.execution_policy is None
                else dict(self.execution_policy)
            ),
        )
        object.__setattr__(self, "content_hash", _content_hash(self.payload()))

    @classmethod
    def freeze(
        cls,
        *,
        authorization_id: str,
        task_id: str,
        task_type: TaskType,
        b0_hash: str,
        hardware_identities: Mapping[str, str],
        mapping_hash: str,
        channel_mapping: Mapping[str, object],
        dca_config_fingerprint: str,
        parameters: tuple[AuthorizedParameter, ...],
        depth: OptimizationDepth,
        candidate_budget: int,
        seed: int,
        max_points: int | None,
        rollback_policy: RollbackPolicy,
        issued_at: datetime,
        expires_at: datetime,
        advanced_iteration_limit: int | None = None,
        execution_policy: Mapping[str, object] | None = None,
    ) -> "FullCampaignAuthorization":
        return cls(
            authorization_id=authorization_id,
            task_id=task_id,
            task_type=task_type,
            b0_hash=b0_hash,
            hardware_identities=tuple(sorted(hardware_identities.items())),
            mapping_hash=mapping_hash,
            channel_mapping=freeze_channel_mapping(channel_mapping),
            dca_config_fingerprint=dca_config_fingerprint,
            parameters=tuple(parameters),
            depth=depth,
            candidate_budget=candidate_budget,
            seed=seed,
            max_points=max_points,
            rollback_policy=rollback_policy,
            issued_at=issued_at,
            expires_at=expires_at,
            advanced_iteration_limit=advanced_iteration_limit,
            execution_policy=_freeze_execution_policy(execution_policy),
        )

    @property
    def mode(self) -> AuthorizationMode:
        return AuthorizationMode.FULL_CAMPAIGN

    @property
    def iteration_limit(self) -> int:
        if self.depth is OptimizationDepth.FAST:
            return 1
        if self.depth is OptimizationDepth.STANDARD:
            return 3
        if self.depth is OptimizationDepth.DEEP:
            return 5
        assert self.advanced_iteration_limit is not None
        return self.advanced_iteration_limit

    def verify_current(
        self,
        *,
        b0_hash: str,
        hardware_identities: Mapping[str, str],
        mapping_hash: str,
        channel_mapping: Mapping[str, object],
        dca_config_fingerprint: str,
        execution_policy: Mapping[str, object] | None = None,
    ) -> None:
        mismatches: list[str] = []
        if b0_hash != self.b0_hash:
            mismatches.append("b0_hash")
        if tuple(sorted(hardware_identities.items())) != self.hardware_identities:
            mismatches.append("hardware_identities")
        if mapping_hash != self.mapping_hash:
            mismatches.append("mapping_hash")
        if freeze_channel_mapping(channel_mapping) != self.channel_mapping:
            mismatches.append("channel_mapping")
        if dca_config_fingerprint != self.dca_config_fingerprint:
            mismatches.append("dca_config_fingerprint")
        if _freeze_execution_policy(execution_policy) != self.execution_policy:
            mismatches.append("execution_policy")
        if mismatches:
            raise AuthorizationDriftError(tuple(mismatches))

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "authorization_id": self.authorization_id,
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "b0_hash": self.b0_hash,
            "hardware_identities": dict(self.hardware_identities),
            "mapping_hash": self.mapping_hash,
            "channel_mapping": dict(self.channel_mapping),
            "dca_config_fingerprint": self.dca_config_fingerprint,
            "parameters": [parameter.payload() for parameter in self.parameters],
            "depth": self.depth.value,
            "candidate_budget": self.candidate_budget,
            "seed": self.seed,
            "max_points": self.max_points,
            "rollback_policy": self.rollback_policy.value,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "advanced_iteration_limit": self.advanced_iteration_limit,
            "execution_policy": (
                None
                if self.execution_policy is None
                else dict(self.execution_policy)
            ),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "FullCampaignAuthorization":
        fields = (
                "mode",
                "authorization_id",
                "task_id",
                "task_type",
                "b0_hash",
                "hardware_identities",
                "mapping_hash",
                "channel_mapping",
                "dca_config_fingerprint",
                "parameters",
                "depth",
                "candidate_budget",
                "seed",
                "max_points",
                "rollback_policy",
                "issued_at",
                "expires_at",
                "advanced_iteration_limit",
            )
        if type(payload) is not dict:
            raise TypeError("FULL_CAMPAIGN authorization must be an object")
        if set(payload) == set(fields):
            value = dict(payload)
            value["execution_policy"] = None
        else:
            value = _require_exact_object(
                "FULL_CAMPAIGN authorization",
                payload,
                fields + ("execution_policy",),
            )
        if value["mode"] != AuthorizationMode.FULL_CAMPAIGN.value:
            raise ValueError("authorization mode must be FULL_CAMPAIGN")

        hardware_payload = value["hardware_identities"]
        if type(hardware_payload) is not dict:
            raise TypeError("authorization hardware_identities must be an object")
        if set(hardware_payload) != {"i2c", "bert", "dca"}:
            raise ValueError("authorization requires i2c, bert, and dca identities")
        hardware_identities: dict[str, str] = {}
        for kind, identity in hardware_payload.items():
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("authorization hardware identities must be nonblank strings")
            hardware_identities[kind] = identity

        channel_mapping = value["channel_mapping"]
        if type(channel_mapping) is not dict:
            raise TypeError("authorization channel_mapping must be an object")
        freeze_channel_mapping(channel_mapping)

        parameters_payload = value["parameters"]
        if type(parameters_payload) is not list:
            raise TypeError("authorization parameters must be an array")
        parameters = tuple(
            AuthorizedParameter.from_payload(item) for item in parameters_payload
        )

        authorization_id = value["authorization_id"]
        task_id = value["task_id"]
        b0_hash = value["b0_hash"]
        mapping_hash = value["mapping_hash"]
        dca_fingerprint = value["dca_config_fingerprint"]
        for name, item in (
            ("authorization_id", authorization_id),
            ("task_id", task_id),
            ("b0_hash", b0_hash),
            ("mapping_hash", mapping_hash),
            ("dca_config_fingerprint", dca_fingerprint),
        ):
            if not isinstance(item, str):
                raise TypeError(f"{name} must be a string")

        max_points_payload = value["max_points"]
        max_points = (
            None
            if max_points_payload is None
            else _decode_int("max_points", max_points_payload)
        )
        iteration_limit_payload = value["advanced_iteration_limit"]
        advanced_iteration_limit = (
            None
            if iteration_limit_payload is None
            else _decode_int("advanced_iteration_limit", iteration_limit_payload)
        )
        execution_policy_payload = value["execution_policy"]
        if execution_policy_payload is not None and type(execution_policy_payload) is not dict:
            raise TypeError("authorization execution_policy must be an object")
        return cls.freeze(
            authorization_id=authorization_id,
            task_id=task_id,
            task_type=TaskType(value["task_type"]),
            b0_hash=b0_hash,
            hardware_identities=hardware_identities,
            mapping_hash=mapping_hash,
            channel_mapping=channel_mapping,
            dca_config_fingerprint=dca_fingerprint,
            parameters=parameters,
            depth=OptimizationDepth(value["depth"]),
            candidate_budget=_decode_int(
                "candidate_budget", value["candidate_budget"]
            ),
            seed=_decode_int("seed", value["seed"]),
            max_points=max_points,
            rollback_policy=RollbackPolicy(value["rollback_policy"]),
            issued_at=_decode_time("issued_at", value["issued_at"]),
            expires_at=_decode_time("expires_at", value["expires_at"]),
            advanced_iteration_limit=advanced_iteration_limit,
            execution_policy=execution_policy_payload,
        )
