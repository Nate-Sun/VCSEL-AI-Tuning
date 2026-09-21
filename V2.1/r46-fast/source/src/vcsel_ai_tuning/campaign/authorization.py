from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from .models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    Candidate,
    CampaignState,
    FirSettings,
    StageAuthorization,
)


class AuthorizationError(ValueError):
    """Raised when an authorization or candidate is outside its safe envelope."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python", warnings=False))
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise AuthorizationError("canonical datetime must be timezone-aware")
        utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AuthorizationError("canonical JSON mapping keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise AuthorizationError("canonical JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AuthorizationError(
        f"unsupported canonical JSON value type: {type(value).__name__}"
    )


def canonical_json_sha256(value: Any) -> str:
    """Return a SHA-256 over stable, compact JSON with UTC-normalized datetimes."""

    canonical_json = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def rollback_hash_for(snapshot: AppliedSettings) -> str:
    """Return the deterministic identity of a rollback snapshot."""

    return canonical_json_sha256({"rollback_snapshot": snapshot})


def _sorted_unique_parameter_specs(
    authorization: StageAuthorization,
) -> tuple[Any, ...]:
    seen_names: set[str] = set()
    for spec in authorization.parameter_specs:
        if spec.name in seen_names:
            raise AuthorizationError(f"duplicate parameter spec name: {spec.name}")
        seen_names.add(spec.name)
    return tuple(sorted(authorization.parameter_specs, key=lambda spec: spec.name))


def _authorization_payload(authorization: StageAuthorization) -> dict[str, Any]:
    payload = {
        "schema": "stage-authorization-v1",
        "campaign_id": authorization.campaign_id,
        "mode": authorization.mode,
        "stage": authorization.stage,
        "baseline_hash": authorization.baseline_hash,
        "rollback_hash": authorization.rollback_hash,
        "parameter_specs": _sorted_unique_parameter_specs(authorization),
        "fir_spec": authorization.fir_spec,
        "max_candidates": authorization.max_candidates,
        "max_duration_s": authorization.max_duration_s,
        "issued_at": authorization.issued_at,
        "expires_at": authorization.expires_at,
    }
    if authorization.time_policy is AuthorizationTimePolicy.BOUNDED:
        return payload
    if authorization.time_policy is AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED:
        payload["schema"] = "stage-authorization-v2"
        payload["time_policy"] = authorization.time_policy
        return payload
    raise AuthorizationError("authorization time policy is unsupported")


def authorization_id_for(authorization: StageAuthorization) -> str:
    """Derive an authorization ID without recursively hashing its current ID."""

    return canonical_json_sha256(_authorization_payload(authorization))


def with_authorization_id(authorization: StageAuthorization) -> StageAuthorization:
    """Return an immutable copy carrying its deterministic authorization ID."""

    return authorization.model_copy(
        update={"authorization_id": authorization_id_for(authorization)}
    )


def _require_aware_datetime(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise AuthorizationError(f"{field_name} must be timezone-aware")
    return value


def _strict_authorization_copy(
    authorization: StageAuthorization,
) -> StageAuthorization:
    try:
        payload = authorization.model_dump_json(warnings=False)
        return StageAuthorization.model_validate_json(payload)
    except ValidationError as exc:
        for error in exc.errors():
            if error["type"] == "timezone_aware":
                location = ".".join(str(part) for part in error["loc"])
                raise AuthorizationError(
                    f"{location} must be timezone-aware"
                ) from exc
        raise AuthorizationError(f"invalid authorization: {exc}") from exc
    except (
        PydanticSerializationError,
        TypeError,
        ValueError,
    ) as exc:
        raise AuthorizationError(f"invalid authorization: {exc}") from exc


def validate_authorization(
    authorization: StageAuthorization,
    *,
    now: datetime | None = None,
    current_campaign_id: str | None = None,
    current_baseline_hash: str | None = None,
) -> StageAuthorization:
    """Validate a V1 stage authorization without performing any I/O."""

    authorization = _strict_authorization_copy(authorization)
    _sorted_unique_parameter_specs(authorization)

    if authorization.mode == AuthorizationMode.FULL_CAMPAIGN:
        raise AuthorizationError("FULL_CAMPAIGN authorization is reserved and disabled")
    if authorization.mode == AuthorizationMode.PER_CANDIDATE:
        raise AuthorizationError("PER_CANDIDATE authorization is disabled in V1")
    if authorization.mode != AuthorizationMode.PER_STAGE:
        raise AuthorizationError(
            f"authorization mode {authorization.mode!s} is unsupported; "
            "only PER_STAGE is enabled"
        )

    if now is None:
        raise AuthorizationError("now is required for PER_STAGE authorization")
    if current_campaign_id is None:
        raise AuthorizationError(
            "current_campaign_id is required for PER_STAGE authorization"
        )
    if current_baseline_hash is None:
        raise AuthorizationError(
            "current_baseline_hash is required for PER_STAGE authorization"
        )

    if (
        authorization.fir_spec is not None
        and authorization.stage != CampaignState.FIR_APPLY
    ):
        raise AuthorizationError("fir_spec is only valid for FIR_APPLY")

    issued_at = _require_aware_datetime(authorization.issued_at, "issued_at")
    checked_at = _require_aware_datetime(now, "now")
    if checked_at < issued_at:
        raise AuthorizationError("authorization is not yet valid")
    if authorization.time_policy is AuthorizationTimePolicy.BOUNDED:
        if authorization.expires_at is None:
            raise AuthorizationError("bounded authorization requires expires_at")
        expires_at = _require_aware_datetime(
            authorization.expires_at, "expires_at"
        )
        if expires_at <= issued_at:
            raise AuthorizationError("expires_at must be after issued_at")
        if checked_at >= expires_at:
            raise AuthorizationError("authorization has expired")
    elif authorization.time_policy is AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED:
        if authorization.expires_at is not None:
            raise AuthorizationError(
                "continuous unbounded authorization requires null expires_at"
            )
    else:
        raise AuthorizationError("authorization time policy is unsupported")

    rollback_snapshot = authorization.rollback_snapshot
    if not rollback_snapshot.readback_i2c and rollback_snapshot.readback_fir is None:
        raise AuthorizationError("rollback snapshot unusable: missing readback")
    for spec in authorization.parameter_specs:
        if spec.enabled and spec.name not in rollback_snapshot.readback_i2c:
            raise AuthorizationError(
                f"rollback readback for {spec.name} is missing"
            )
        if spec.enabled:
            rollback_value = rollback_snapshot.readback_i2c[spec.name]
            if not spec.encoding_min <= rollback_value <= spec.encoding_max:
                raise AuthorizationError(
                    f"rollback readback for {spec.name} is outside its encoding range"
                )

    if authorization.fir_spec is not None:
        rollback_fir = rollback_snapshot.readback_fir
        if rollback_fir is None:
            raise AuthorizationError(
                "FIR_APPLY rollback snapshot requires FIR readback"
            )
        if (
            rollback_fir.upper_nlc
            != authorization.fir_spec.fixed_upper_nlc
            or rollback_fir.lower_nlc
            != authorization.fir_spec.fixed_lower_nlc
        ):
            raise AuthorizationError(
                "FIR rollback NLC does not match the frozen fir_spec"
            )
        for index, value in enumerate(rollback_fir.taps):
            if not (
                authorization.fir_spec.tap_min[index]
                <= value
                <= authorization.fir_spec.tap_max[index]
            ):
                raise AuthorizationError(
                    f"FIR rollback tap {index} is outside the authorized range"
                )

    expected_rollback_hash = rollback_hash_for(rollback_snapshot)
    if authorization.rollback_hash != expected_rollback_hash:
        raise AuthorizationError("rollback snapshot/hash mismatch")

    expected_authorization_id = authorization_id_for(authorization)
    if authorization.authorization_id != expected_authorization_id:
        raise AuthorizationError("authorization_id/hash mismatch")

    if (
        current_campaign_id is not None
        and authorization.campaign_id != current_campaign_id
    ):
        raise AuthorizationError("campaign is outside authorization scope")
    if (
        current_baseline_hash is not None
        and authorization.baseline_hash != current_baseline_hash
    ):
        raise AuthorizationError("baseline is stale")

    return authorization


def _validate_budget(
    authorization: StageAuthorization,
    candidate_index: int,
    elapsed_s: float,
) -> None:
    if type(candidate_index) is not int or candidate_index < 0:
        raise AuthorizationError("candidate_index must be a non-negative integer")
    if candidate_index >= authorization.max_candidates:
        raise AuthorizationError(
            "candidate_index reaches or exceeds max_candidates "
            "(candidate_index is zero-based)"
        )

    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(elapsed_s)
        or elapsed_s < 0
    ):
        raise AuthorizationError("elapsed_s must be a non-negative finite number")
    if authorization.time_policy is AuthorizationTimePolicy.BOUNDED:
        if authorization.max_duration_s is None:
            raise AuthorizationError(
                "bounded authorization requires max_duration_s"
            )
        if elapsed_s >= authorization.max_duration_s:
            raise AuthorizationError("elapsed_s reaches or exceeds max_duration_s")
    elif authorization.time_policy is not AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED:
        raise AuthorizationError("authorization time policy is unsupported")


def validate_candidate(
    authorization: StageAuthorization,
    candidate: Candidate,
    previous_readback: Mapping[str, object],
    *,
    previous_fir: FirSettings | None = None,
    now: datetime | None = None,
    current_campaign_id: str | None = None,
    current_baseline_hash: str | None = None,
    candidate_index: int = 0,
    elapsed_s: float = 0.0,
) -> Candidate:
    """Validate a candidate against an authorization and actual prior readback."""

    validate_candidate_static(
        authorization,
        candidate,
        now=now,
        current_campaign_id=current_campaign_id,
        current_baseline_hash=current_baseline_hash,
        candidate_index=candidate_index,
        elapsed_s=elapsed_s,
    )
    if not isinstance(previous_readback, Mapping):
        raise AuthorizationError("previous_readback must be a mapping")

    if candidate.requested_fir is not None:
        _validate_fir_candidate_delta(
            authorization,
            candidate.requested_fir,
            previous_fir=previous_fir,
        )
        return candidate

    specs_by_name = {spec.name: spec for spec in authorization.parameter_specs}
    for name, requested_value in candidate.requested_i2c.items():
        spec = specs_by_name[name]

        if name not in previous_readback:
            raise AuthorizationError(f"parameter {name} previous readback is missing")
        readback_value = previous_readback[name]
        if type(readback_value) is not int:
            raise AuthorizationError(
                f"parameter {name} previous readback must be an integer"
            )
        if not spec.encoding_min <= readback_value <= spec.encoding_max:
            raise AuthorizationError(
                f"parameter {name} previous readback is outside its encoding range"
            )
        if abs(requested_value - readback_value) > spec.max_step:
            raise AuthorizationError(
                f"parameter {name} exceeds max_step from actual previous readback"
            )

    return candidate


def validate_candidate_static(
    authorization: StageAuthorization,
    candidate: Candidate,
    *,
    now: datetime | None = None,
    current_campaign_id: str | None = None,
    current_baseline_hash: str | None = None,
    candidate_index: int = 0,
    elapsed_s: float = 0.0,
) -> Candidate:
    """Validate every candidate property that does not require hardware readback."""

    authorization = validate_authorization(
        authorization,
        now=now,
        current_campaign_id=current_campaign_id,
        current_baseline_hash=current_baseline_hash,
    )
    _validate_budget(authorization, candidate_index, elapsed_s)
    if candidate.stage != authorization.stage:
        raise AuthorizationError("candidate stage is outside authorization stage scope")
    if not isinstance(candidate.requested_i2c, Mapping):
        raise AuthorizationError("requested_i2c must be a mapping")

    if candidate.requested_fir is not None:
        if authorization.stage != CampaignState.FIR_APPLY:
            raise AuthorizationError("FIR changes are not authorized for this stage")
        if candidate.requested_i2c:
            raise AuthorizationError("FIR_APPLY candidates cannot request I2C changes")
        _validate_fir_candidate_static(authorization, candidate.requested_fir)
        return candidate
    if authorization.stage == CampaignState.FIR_APPLY:
        if candidate.requested_i2c:
            raise AuthorizationError("I2C changes are not authorized during FIR_APPLY")
        raise AuthorizationError("FIR_APPLY candidate requires requested_fir")

    specs_by_name = {spec.name: spec for spec in authorization.parameter_specs}
    for name, requested_value in candidate.requested_i2c.items():
        spec = specs_by_name.get(name)
        if spec is None:
            raise AuthorizationError(f"parameter {name} is unknown")
        if spec.enabled is not True:
            raise AuthorizationError(f"parameter {name} is disabled")
        if type(requested_value) is not int:
            raise AuthorizationError(f"parameter {name} value must be an integer")
        if not spec.encoding_min <= requested_value <= spec.encoding_max:
            raise AuthorizationError(
                f"parameter {name} is outside its encoding range"
            )
        if not spec.tuning_min <= requested_value <= spec.tuning_max:
            raise AuthorizationError(f"parameter {name} is outside its tuning range")
    return candidate


def _validate_fir_candidate_static(
    authorization: StageAuthorization,
    requested: FirSettings,
) -> None:
    spec = authorization.fir_spec
    if spec is None:
        raise AuthorizationError("FIR_APPLY candidate requires an explicit fir_spec")
    rollback_fir = authorization.rollback_snapshot.readback_fir
    if rollback_fir is None:
        raise AuthorizationError("FIR_APPLY rollback snapshot requires FIR readback")
    if not isinstance(requested, FirSettings):
        raise AuthorizationError("requested FIR must be strict FirSettings")
    fixed_nlc = (spec.fixed_upper_nlc, spec.fixed_lower_nlc)
    if (
        (rollback_fir.upper_nlc, rollback_fir.lower_nlc) != fixed_nlc
        or (requested.upper_nlc, requested.lower_nlc) != fixed_nlc
    ):
        raise AuthorizationError("FIR NLC values are frozen by authorization")
    for index, value in enumerate(requested.taps):
        if not spec.tap_min[index] <= value <= spec.tap_max[index]:
            raise AuthorizationError(
                f"FIR tap {index} is outside its authorized range"
            )


def _validate_fir_candidate_delta(
    authorization: StageAuthorization,
    requested: FirSettings,
    *,
    previous_fir: FirSettings | None,
) -> None:
    _validate_fir_candidate_static(authorization, requested)
    spec = authorization.fir_spec
    assert spec is not None
    if previous_fir is None:
        raise AuthorizationError("previous_fir readback is required for FIR_APPLY")
    if not isinstance(previous_fir, FirSettings):
        raise AuthorizationError("previous_fir must be strict FirSettings readback")

    fixed_nlc = (spec.fixed_upper_nlc, spec.fixed_lower_nlc)
    if (
        (previous_fir.upper_nlc, previous_fir.lower_nlc) != fixed_nlc
        or (requested.upper_nlc, requested.lower_nlc) != fixed_nlc
    ):
        raise AuthorizationError("FIR NLC values are frozen by authorization")

    total_change = 0
    for index, (value, previous) in enumerate(
        zip(requested.taps, previous_fir.taps, strict=True)
    ):
        if not spec.tap_min[index] <= value <= spec.tap_max[index]:
            raise AuthorizationError(
                f"FIR tap {index} is outside its authorized range"
            )
        delta = abs(value - previous)
        if delta > spec.max_step[index]:
            raise AuthorizationError(
                f"FIR tap {index} exceeds max_step from actual previous readback"
            )
        total_change += delta

    if total_change > spec.max_total_change_l1:
        raise AuthorizationError(
            "FIR total L1 change exceeds max_total_change_l1"
        )
