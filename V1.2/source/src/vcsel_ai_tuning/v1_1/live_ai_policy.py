"""Pure, immutable safety contracts for reviewed LIVE AI profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any


PILOT_3_PARAMETERS = {
    "ibias_dac": (69, 73, 2),
    "res_gain": (4, 8, 4),
}
FAST24_PARAMETERS = {
    "ibias_dac": (60, 80, 1),
    "res_gain": (0, 10, 1),
}
STAGED_BUDGETS = {
    "candidate_budget": 27,
    "pilot_candidate_budget": 3,
    "fast_candidate_budget": 24,
    "coarse_candidate_budget": 6,
    "fine_candidate_budget": 16,
    "fir_first_candidate_budget": 1,
    "fir_iterative_candidate_budget": 1,
}
STAGED_RESTORE_POLICY = "PILOT_EACH_CANDIDATE_AND_TERMINAL_B0"
PILOT_RESTORE_POLICY = "EACH_CANDIDATE_AND_TERMINAL_B0"
_PILOT_AUTHORIZATION_FIELDS = (
    "live_ai_profile",
    "optimization_scope",
    "candidate_budget",
    "coarse_candidate_budget",
    "fine_candidate_budget",
    "settle_time_s",
    "restore_after_each_candidate",
    "save_dataset",
)
_STAGED_AUTHORIZATION_FIELDS = (
    "live_ai_profile",
    "optimization_scope",
    "candidate_budget",
    "pilot_candidate_budget",
    "fast_candidate_budget",
    "coarse_candidate_budget",
    "fine_candidate_budget",
    "fir_first_candidate_budget",
    "fir_iterative_candidate_budget",
    "settle_time_s",
)

_FORBIDDEN_CALLER_FIELDS = frozenset(
    {
        "fir",
        "fir_taps",
        "upper_nlc",
        "lower_nlc",
        "pattern",
        "bert_pattern",
        "setfir",
        "apply_fir",
        "restore_after_each_candidate",
    }
)


class LiveAIPolicyError(ValueError):
    """A requested or persisted LIVE AI task exceeds reviewed authority."""


class LiveAIProfile(StrEnum):
    """Reviewed LIVE profiles; the combined profile is the product entrypoint."""

    PILOT_3 = "PILOT_3"
    FAST24 = "FAST24"
    PILOT_3_THEN_FAST24 = "PILOT_3_THEN_FAST24"


def _wire_value(value: object) -> object:
    return getattr(value, "value", value)


def _decoded_parameters(
    payload: Mapping[str, Any],
    *,
    profile_name: str,
) -> dict[str, tuple[int, int, int]]:
    parameters = payload.get("parameters")
    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
        raise LiveAIPolicyError(
            f"LIVE {profile_name} parameters must be a sequence"
        )
    decoded: dict[str, tuple[int, int, int]] = {}
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            raise LiveAIPolicyError(
                f"LIVE {profile_name} parameters must contain mappings"
            )
        name = parameter.get("name")
        if not isinstance(name, str) or not name.strip():
            raise LiveAIPolicyError(
                f"LIVE {profile_name} parameter name must be nonblank"
            )
        canonical_name = name.strip()
        if canonical_name in decoded:
            raise LiveAIPolicyError(
                f"LIVE {profile_name} parameter names must be unique"
            )
        try:
            values = (parameter["min"], parameter["max"], parameter["step"])
        except KeyError as error:
            raise LiveAIPolicyError(
                f"LIVE {profile_name} parameter range is incomplete"
            ) from error
        if any(type(value) is not int for value in values):
            raise LiveAIPolicyError(
                f"LIVE {profile_name} parameter ranges must be integers"
            )
        decoded[canonical_name] = values
    return decoded


def _require_task_type(payload: Mapping[str, Any], profile_name: str) -> None:
    if _wire_value(payload.get("task_type")) != "AI_OPTIMIZATION":
        raise LiveAIPolicyError(
            f"LIVE {profile_name} permits only AI_OPTIMIZATION"
        )


def _require_settle(payload: Mapping[str, Any], profile_name: str) -> None:
    settle = payload.get("settle_time_s")
    if (
        isinstance(settle, bool)
        or not isinstance(settle, (int, float))
        or float(settle) != 0.1
    ):
        raise LiveAIPolicyError(
            f"LIVE {profile_name} requires settle_time_s=0.1"
        )


def _reject_forbidden_fields(
    payload: Mapping[str, Any],
    profile_name: str,
) -> None:
    forbidden = sorted(
        str(key)
        for key in payload
        if str(key).casefold() in _FORBIDDEN_CALLER_FIELDS
    )
    if forbidden:
        raise LiveAIPolicyError(
            f"LIVE {profile_name} formula policy has forbidden fields: "
            + ", ".join(forbidden)
        )


def _require_parameter_grid(
    payload: Mapping[str, Any],
    *,
    profile_name: str,
    expected: Mapping[str, tuple[int, int, int]],
) -> None:
    decoded = _decoded_parameters(payload, profile_name=profile_name)
    forbidden_parameters = sorted(set(decoded) - set(expected))
    if forbidden_parameters:
        raise LiveAIPolicyError(
            f"forbidden LIVE {profile_name} parameter: "
            + ", ".join(forbidden_parameters)
        )
    if decoded != dict(expected):
        raise LiveAIPolicyError(
            f"LIVE {profile_name} parameter grid does not match reviewed policy"
        )


def validate_live_i2c_pilot_payload(payload: Mapping[str, Any]) -> None:
    """Validate the standalone read-only-BERT three-candidate pilot."""

    if not isinstance(payload, Mapping):
        raise LiveAIPolicyError("LIVE PILOT_3 payload must be a mapping")
    _require_task_type(payload, "PILOT_3")
    if _wire_value(payload.get("optimization_scope")) != "I2C_ONLY":
        raise LiveAIPolicyError("LIVE PILOT_3 requires I2C_ONLY")
    if _wire_value(payload.get("depth")) != "STANDARD":
        raise LiveAIPolicyError("LIVE PILOT_3 depth must be STANDARD")
    for name, expected in (
        ("candidate_budget", 3),
        ("coarse_candidate_budget", 3),
        ("fine_candidate_budget", 0),
    ):
        if type(payload.get(name)) is not int or payload.get(name) != expected:
            raise LiveAIPolicyError(
                f"LIVE PILOT_3 requires {name}={expected}"
            )
    _require_settle(payload, "PILOT_3")
    if payload.get("restore_after_each_candidate") is not True:
        raise LiveAIPolicyError(
            "LIVE PILOT_3 requires B0 restore after every candidate"
        )
    if payload.get("save_dataset") is not True:
        raise LiveAIPolicyError("LIVE PILOT_3 requires save_dataset=true")
    _reject_forbidden_fields(
        {
            key: value
            for key, value in payload.items()
            if key != "restore_after_each_candidate"
        },
        "PILOT_3",
    )
    _require_parameter_grid(
        payload,
        profile_name="PILOT_3",
        expected=PILOT_3_PARAMETERS,
    )


def validate_live_fast24_payload(payload: Mapping[str, Any]) -> None:
    """Validate the standalone 24-round alternating profile."""

    if not isinstance(payload, Mapping):
        raise LiveAIPolicyError("LIVE FAST24 payload must be a mapping")
    _require_task_type(payload, "FAST24")
    if _wire_value(payload.get("optimization_scope")) != "I2C_FIR_ALTERNATING":
        raise LiveAIPolicyError("LIVE FAST24 requires I2C_FIR_ALTERNATING")
    if _wire_value(payload.get("depth")) != "FAST":
        raise LiveAIPolicyError("LIVE FAST24 depth must be FAST")
    if type(payload.get("candidate_budget")) is not int or payload.get(
        "candidate_budget"
    ) != 24:
        raise LiveAIPolicyError("LIVE FAST24 requires candidate_budget=24")
    _require_settle(payload, "FAST24")
    _reject_forbidden_fields(payload, "FAST24")
    _require_parameter_grid(
        payload,
        profile_name="FAST24",
        expected=FAST24_PARAMETERS,
    )


def validate_staged_live_ai_payload(payload: Mapping[str, Any]) -> None:
    """Validate the single pre-authorized Native V1.2 staged profile."""

    profile_name = "PILOT_3_THEN_FAST24"
    if not isinstance(payload, Mapping):
        raise LiveAIPolicyError(
            f"LIVE {profile_name} payload must be a mapping"
        )
    _require_task_type(payload, profile_name)
    if _wire_value(payload.get("live_ai_profile")) != profile_name:
        raise LiveAIPolicyError(
            f"LIVE {profile_name} requires its exact profile name"
        )
    if _wire_value(payload.get("optimization_scope")) != "I2C_FIR_ALTERNATING":
        raise LiveAIPolicyError(
            f"LIVE {profile_name} requires I2C_FIR_ALTERNATING"
        )
    if _wire_value(payload.get("depth")) != "FAST":
        raise LiveAIPolicyError(f"LIVE {profile_name} depth must be FAST")
    for name, expected in STAGED_BUDGETS.items():
        if type(payload.get(name)) is not int or payload.get(name) != expected:
            raise LiveAIPolicyError(
                f"LIVE {profile_name} requires {name}={expected}"
            )
    _require_settle(payload, profile_name)
    if payload.get("save_dataset") is not True:
        raise LiveAIPolicyError(
            f"LIVE {profile_name} requires save_dataset=true"
        )
    _reject_forbidden_fields(payload, profile_name)
    _require_parameter_grid(
        payload,
        profile_name=profile_name,
        expected=FAST24_PARAMETERS,
    )


def staged_authorization_policy(
    payload: Mapping[str, Any],
) -> dict[str, object]:
    """Return the exact scalar policy frozen into FULL_CAMPAIGN authority."""

    validate_staged_live_ai_payload(payload)
    return {
        **{name: _wire_value(payload[name]) for name in _STAGED_AUTHORIZATION_FIELDS},
        "restore_policy": STAGED_RESTORE_POLICY,
    }


def pilot_authorization_policy(
    payload: Mapping[str, Any],
) -> dict[str, object]:
    """Return the exact scalar policy frozen for standalone PILOT_3."""

    validate_live_i2c_pilot_payload(payload)
    if _wire_value(payload.get("live_ai_profile")) != LiveAIProfile.PILOT_3.value:
        raise LiveAIPolicyError("LIVE PILOT_3 requires its exact profile name")
    return {
        **{name: _wire_value(payload[name]) for name in _PILOT_AUTHORIZATION_FIELDS},
        "restore_policy": PILOT_RESTORE_POLICY,
    }


def validate_live_ai_payload(
    profile: LiveAIProfile | str,
    payload: Mapping[str, Any],
) -> None:
    """Dispatch to an exact reviewed profile without widening authority."""

    try:
        selected = LiveAIProfile(profile)
    except (TypeError, ValueError) as error:
        raise LiveAIPolicyError(
            f"unknown LIVE AI profile: {profile!r}"
        ) from error
    if selected is LiveAIProfile.PILOT_3:
        validate_live_i2c_pilot_payload(payload)
    elif selected is LiveAIProfile.FAST24:
        validate_live_fast24_payload(payload)
    else:
        validate_staged_live_ai_payload(payload)


LiveAIPilotPolicyError = LiveAIPolicyError
