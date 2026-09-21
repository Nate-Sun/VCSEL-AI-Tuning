from __future__ import annotations

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.api.v1_1_schemas import CreateTaskRequest
from vcsel_ai_tuning.v1_1.domain import OptimizationScope
from vcsel_ai_tuning.v1_1.live_ai_policy import LiveAIProfile


def _base_payload() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "task_type": "AI_OPTIMIZATION",
        "channel_mapping": {
            "i2c_channel": 0,
            "bert_port": "port1",
            "bert_channel": "0",
            "dca_tdecq_source": "FUNC1",
            "dca_tdecq_equalizer_function": 1,
            "dca_outer_er_source": "CHAN1A",
            "operator_confirmed": True,
        },
        "parameters": [
            {"name": "ibias_dac", "min": 60, "max": 80, "step": 1},
            {"name": "res_gain", "min": 0, "max": 10, "step": 1},
        ],
        "depth": "FAST",
        "candidate_budget": 27,
        "seed": 17,
        "save_dataset": True,
    }


def _staged_payload() -> dict[str, object]:
    return {
        **_base_payload(),
        "live_ai_profile": "PILOT_3_THEN_FAST24",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        "pilot_candidate_budget": 3,
        "fast_candidate_budget": 24,
        "coarse_candidate_budget": 6,
        "fine_candidate_budget": 16,
        "fir_first_candidate_budget": 1,
        "fir_iterative_candidate_budget": 1,
        "settle_time_s": 0.1,
    }


def _pilot_payload() -> dict[str, object]:
    return {
        **_base_payload(),
        "parameters": [
            {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
            {"name": "res_gain", "min": 4, "max": 8, "step": 4},
        ],
        "depth": "STANDARD",
        "candidate_budget": 3,
        "live_ai_profile": "PILOT_3",
        "optimization_scope": "I2C_ONLY",
        "coarse_candidate_budget": 3,
        "fine_candidate_budget": 0,
        "restore_after_each_candidate": True,
        "settle_time_s": 0.1,
        "save_dataset": True,
    }


def test_staged_create_task_has_typed_immutable_authority():
    request = CreateTaskRequest.model_validate(_staged_payload())

    assert request.live_ai_profile is LiveAIProfile.PILOT_3_THEN_FAST24
    assert request.optimization_scope is OptimizationScope.I2C_FIR_ALTERNATING
    assert request.candidate_budget == 27
    assert request.pilot_candidate_budget == 3
    assert request.fast_candidate_budget == 24
    assert request.coarse_candidate_budget == 6
    assert request.fine_candidate_budget == 16
    assert request.fir_first_candidate_budget == 1
    assert request.fir_iterative_candidate_budget == 1
    assert request.settle_time_s == 0.1
    assert request.save_dataset is True


def test_staged_create_task_json_round_trip_preserves_exact_authority():
    request = CreateTaskRequest.model_validate(_staged_payload())

    restored = CreateTaskRequest.model_validate(request.model_dump(mode="json"))

    assert restored == request


def test_standalone_pilot_create_task_preserves_restore_and_dataset_authority():
    request = CreateTaskRequest.model_validate(_pilot_payload())

    assert request.live_ai_profile is LiveAIProfile.PILOT_3
    assert request.optimization_scope is OptimizationScope.I2C_ONLY
    assert request.restore_after_each_candidate is True
    assert request.save_dataset is True
    assert request.model_dump(mode="json")["restore_after_each_candidate"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("live_ai_profile", "UNKNOWN"),
        ("candidate_budget", 26),
        ("pilot_candidate_budget", 4),
        ("fast_candidate_budget", 23),
        ("coarse_candidate_budget", 7),
        ("fine_candidate_budget", 15),
        ("fir_first_candidate_budget", 0),
        ("fir_iterative_candidate_budget", 2),
        ("settle_time_s", float("inf")),
        ("save_dataset", 1),
    ],
)
def test_staged_create_task_rejects_changed_or_untyped_authority(
    field: str,
    value: object,
):
    payload = _staged_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        CreateTaskRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fir_taps", [0, 9, -28, 104, -8, 1, -3]),
        ("upper_nlc", 6),
        ("lower_nlc", 6),
        ("pattern", "SSPRQ"),
    ],
)
def test_staged_create_task_rejects_caller_fir_authority(
    field: str,
    value: object,
):
    payload = _staged_payload()
    payload[field] = value

    with pytest.raises(ValidationError, match=field):
        CreateTaskRequest.model_validate(payload)


def test_legacy_ai_request_keeps_its_existing_serialized_shape():
    payload = {
        **_base_payload(),
        "parameters": [{"name": "eq", "min": 18, "max": 22, "step": 2}],
        "depth": "STANDARD",
        "candidate_budget": 20,
        "save_dataset": False,
    }
    request = CreateTaskRequest.model_validate(payload)

    dumped = request.model_dump(mode="json")

    assert "live_ai_profile" not in dumped
    assert "optimization_scope" not in dumped
    assert "pilot_candidate_budget" not in dumped
    assert "fast_candidate_budget" not in dumped
    assert "coarse_candidate_budget" not in dumped
    assert "fine_candidate_budget" not in dumped
    assert "fir_first_candidate_budget" not in dumped
    assert "fir_iterative_candidate_budget" not in dumped


def test_non_staged_request_cannot_supply_staged_fields():
    payload = {
        **_base_payload(),
        "parameters": [{"name": "eq", "min": 18, "max": 22, "step": 2}],
        "depth": "STANDARD",
        "candidate_budget": 20,
        "save_dataset": False,
        "pilot_candidate_budget": 3,
    }

    with pytest.raises(ValidationError, match="pilot_candidate_budget"):
        CreateTaskRequest.model_validate(payload)
