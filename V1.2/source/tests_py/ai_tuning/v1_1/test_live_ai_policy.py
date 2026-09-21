from __future__ import annotations

from copy import deepcopy

import pytest

from vcsel_ai_tuning.v1_1.live_ai_policy import (
    LiveAIProfile,
    LiveAIPolicyError,
    validate_live_ai_payload,
    validate_live_fast24_payload,
    validate_live_i2c_pilot_payload,
)


def staged_payload(*, save_dataset: bool = True) -> dict[str, object]:
    return {
        "task_type": "AI_OPTIMIZATION",
        "live_ai_profile": "PILOT_3_THEN_FAST24",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        "depth": "FAST",
        "candidate_budget": 27,
        "pilot_candidate_budget": 3,
        "fast_candidate_budget": 24,
        "coarse_candidate_budget": 6,
        "fine_candidate_budget": 16,
        "fir_first_candidate_budget": 1,
        "fir_iterative_candidate_budget": 1,
        "settle_time_s": 0.1,
        "save_dataset": save_dataset,
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
    }


def pilot_payload() -> dict[str, object]:
    return {
        "task_type": "AI_OPTIMIZATION",
        "optimization_scope": "I2C_ONLY",
        "depth": "STANDARD",
        "candidate_budget": 3,
        "coarse_candidate_budget": 3,
        "fine_candidate_budget": 0,
        "settle_time_s": 0.1,
        "restore_after_each_candidate": True,
        "save_dataset": True,
        "parameters": [
            {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
            {"name": "res_gain", "min": 4, "max": 8, "step": 4},
        ],
    }


def fast24_payload() -> dict[str, object]:
    return {
        "task_type": "AI_OPTIMIZATION",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        "depth": "FAST",
        "candidate_budget": 24,
        "settle_time_s": 0.1,
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
    }


def test_standalone_pilot_and_fast24_policies_remain_available():
    validate_live_i2c_pilot_payload(pilot_payload())
    validate_live_fast24_payload(fast24_payload())
    validate_live_ai_payload(LiveAIProfile.PILOT_3, pilot_payload())
    validate_live_ai_payload(LiveAIProfile.FAST24, fast24_payload())


@pytest.mark.parametrize(
    ("profile", "payload", "field", "value"),
    [
        (LiveAIProfile.PILOT_3, pilot_payload, "candidate_budget", 4),
        (LiveAIProfile.PILOT_3, pilot_payload, "restore_after_each_candidate", False),
        (LiveAIProfile.FAST24, fast24_payload, "candidate_budget", 25),
        (LiveAIProfile.FAST24, fast24_payload, "settle_time_s", 0.0),
    ],
)
def test_standalone_profiles_reject_changed_authority(
    profile: LiveAIProfile,
    payload,
    field: str,
    value: object,
):
    changed = payload()
    changed[field] = value

    with pytest.raises(LiveAIPolicyError, match=profile.value):
        validate_live_ai_payload(profile, changed)


def test_staged_policy_accepts_only_the_frozen_contract():
    validate_live_ai_payload(
        LiveAIProfile.PILOT_3_THEN_FAST24,
        staged_payload(save_dataset=True),
    )


def test_staged_policy_rejects_disabled_required_dataset():
    with pytest.raises(LiveAIPolicyError, match="save_dataset=true"):
        validate_live_ai_payload(
            LiveAIProfile.PILOT_3_THEN_FAST24,
            staged_payload(save_dataset=False),
        )


def test_standalone_pilot_rejects_disabled_required_dataset():
    payload = pilot_payload()
    payload["save_dataset"] = False

    with pytest.raises(LiveAIPolicyError, match="save_dataset=true"):
        validate_live_ai_payload(LiveAIProfile.PILOT_3, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_type", "SWEEP"),
        ("live_ai_profile", "FAST24"),
        ("optimization_scope", "I2C_ONLY"),
        ("depth", "STANDARD"),
        ("candidate_budget", 26),
        ("pilot_candidate_budget", 4),
        ("fast_candidate_budget", 23),
        ("coarse_candidate_budget", 7),
        ("fine_candidate_budget", 15),
        ("fir_first_candidate_budget", 0),
        ("fir_iterative_candidate_budget", 2),
        ("settle_time_s", 0.2),
        ("save_dataset", 1),
    ],
)
def test_staged_policy_rejects_changed_authority(field: str, value: object):
    payload = staged_payload()
    payload[field] = value

    with pytest.raises(LiveAIPolicyError, match="PILOT_3_THEN_FAST24"):
        validate_live_ai_payload(LiveAIProfile.PILOT_3_THEN_FAST24, payload)


@pytest.mark.parametrize(
    ("parameter_index", "field", "value"),
    [
        (0, "min", 59),
        (0, "max", 81),
        (0, "step", 2),
        (1, "min", -1),
        (1, "max", 11),
        (1, "step", 2),
    ],
)
def test_staged_policy_rejects_changed_fast24_grid(
    parameter_index: int,
    field: str,
    value: int,
):
    payload = deepcopy(staged_payload())
    parameters = payload["parameters"]
    assert isinstance(parameters, list)
    parameter = parameters[parameter_index]
    assert isinstance(parameter, dict)
    parameter[field] = value

    with pytest.raises(LiveAIPolicyError, match="PILOT_3_THEN_FAST24"):
        validate_live_ai_payload(LiveAIProfile.PILOT_3_THEN_FAST24, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fir_taps", [0, 9, -28, 104, -8, 1, -3]),
        ("upper_nlc", 6),
        ("lower_nlc", 6),
        ("pattern", "SSPRQ"),
        ("apply_fir", True),
        ("restore_after_each_candidate", True),
    ],
)
def test_staged_policy_rejects_caller_supplied_fir_or_restore_fields(
    field: str,
    value: object,
):
    payload = staged_payload()
    payload[field] = value

    with pytest.raises(LiveAIPolicyError, match="forbidden"):
        validate_live_ai_payload(LiveAIProfile.PILOT_3_THEN_FAST24, payload)


def test_staged_policy_rejects_extra_i2c_parameter():
    payload = staged_payload()
    parameters = payload["parameters"]
    assert isinstance(parameters, list)
    parameters.append({"name": "eq", "min": 20, "max": 24, "step": 1})

    with pytest.raises(LiveAIPolicyError, match="forbidden.*eq"):
        validate_live_ai_payload(LiveAIProfile.PILOT_3_THEN_FAST24, payload)
