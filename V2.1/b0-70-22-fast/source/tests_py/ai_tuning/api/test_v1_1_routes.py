from datetime import datetime, timezone
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vcsel_ai_tuning.api.app import create_app
from vcsel_ai_tuning.api.routes import CampaignService
from vcsel_ai_tuning.api.v1_1_schemas import ChannelMappingRequest
from vcsel_ai_tuning.v1_1.authorization import AuthorizationDriftError
from vcsel_ai_tuning.v1_1.domain import LifecycleState
from vcsel_ai_tuning.v1_1.hardware_control import DcaConfigurationDrift
from vcsel_ai_tuning.v1_1.repository import (
    ActiveTaskConflict,
    TaskStoreError,
    UnknownTaskError,
)


class Service:
    def __init__(self):
        self.calls = []
        self.failure = None

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.failure is not None:
            raise self.failure
        return {"operation": name, "product_version": "1.1.0"}

    def runtime_health(self): return self._call("runtime_health")
    def runtime_capabilities(self): return self._call("runtime_capabilities")
    def create_task(self, request): return self._call("create_task", request)
    def active_task(self): return self._call("active_task")
    def hardware_snapshot(self): return self._call("hardware_snapshot")
    def scan_parameter_mapping(self): return self._call("scan_parameter_mapping")
    def validate_parameters(self, parameters): return self._call("validate_parameters", parameters)
    def save_b0(self, task_id, request): return self._call("save_b0", task_id, request)
    def count_sweep(self, request): return self._call("count_sweep", request)
    def authorize_task(self, task_id, request): return self._call("authorize_task", task_id, request)
    def start_task(self, task_id): return self._call("start_task", task_id)
    def pause_task(self, task_id): return self._call("pause_task", task_id)
    def resume_task(self, task_id): return self._call("resume_task", task_id)
    def abort_task(self, task_id, request): return self._call("abort_task", task_id, request)
    def recover_task(self, task_id, request): return self._call("recover_task", task_id, request)
    def get_task(self, task_id): return self._call("get_task", task_id)
    def observations(self, task_id): return self._call("observations", task_id)
    def best_result(self, task_id): return self._call("best_result", task_id)
    def sweep_points(self, task_id, *, offset, limit): return self._call("sweep_points", task_id, offset, limit)
    def final_report(self, task_id): return self._call("final_report", task_id)
    def apply_history_result(self, request): return self._call("apply_history_result", request)
    def recover_history_result(self, recovery_id, request): return self._call("recover_history_result", recovery_id, request)


class Events:
    def events(self, task_id, *, after_sequence=0):
        return []

    def current_state(self, task_id):
        return LifecycleState.COMPLETED


def client_and_service():
    service = Service()
    app = create_app(
        cast(CampaignService, cast(Any, object())),
        v11_service=service,
        v11_event_source=Events(),
    )
    return TestClient(app), service


def ai_payload():
    return {
        "task_id": "task-1",
        "task_type": "AI_OPTIMIZATION",
        "channel_mapping": {
            "i2c_channel": 0,
            "bert_port": "port1",
            "bert_channel": "0",
            "dca_tdecq_source": "FUNC1",
            "dca_outer_er_source": "CHAN1A",
            "operator_confirmed": True,
        },
        "parameters": [{"name": "eq", "min": 1, "max": 5, "step": 2}],
        "depth": "STANDARD",
        "candidate_budget": 20,
        "seed": 7,
    }


def test_v11_routes_extend_same_app_and_publish_versioned_openapi():
    client, service = client_and_service()

    response = client.post("/api/v1.1/tasks", json=ai_payload())
    openapi = client.get("/openapi.json").json()

    assert response.status_code == 201
    assert response.json()["product_version"] == "1.1.0"
    assert "/api/campaigns" in openapi["paths"]
    assert "/api/v1.1/tasks" in openapi["paths"]
    assert openapi["info"]["version"] == "1.1.0"
    assert service.calls[0][0] == "create_task"


def test_frontend_cannot_supply_register_address_or_bit():
    client, service = client_and_service()
    payload = ai_payload()
    payload["parameters"][0]["address"] = 0x2006

    response = client.post("/api/v1.1/tasks", json=payload)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("field", ("dca_tdecq_source", "dca_outer_er_source"))
def test_operator_must_explicitly_confirm_both_dca_mapping_sources(field):
    mapping = ai_payload()["channel_mapping"]
    del mapping[field]

    with pytest.raises(ValidationError):
        ChannelMappingRequest.model_validate(mapping)


def test_sweep_requires_explicit_hardware_mode_and_max_points():
    client, service = client_and_service()
    payload = {**ai_payload(), "task_type": "SWEEP"}

    response = client.post("/api/v1.1/tasks", json=payload)

    assert response.status_code == 422
    assert service.calls == []


def test_sweep_count_accepts_json_parameter_array_as_internal_tuple():
    client, service = client_and_service()

    response = client.post(
        "/api/v1.1/sweeps/count",
        json={
            "hardware": "I2C_SWEEP",
            "mode": "SINGLE_PARAMETER",
            "parameters": [{"name": "eq", "min": 1, "max": 5, "step": 2}],
            "seed": 7,
            "max_points": 10,
        },
    )

    assert response.status_code == 200
    assert service.calls[-1][0] == "count_sweep"
    assert isinstance(service.calls[-1][1][0].parameters, tuple)


def test_parameter_validation_accepts_bare_json_array_as_internal_tuple():
    client, service = client_and_service()

    response = client.post(
        "/api/v1.1/parameters/validate",
        json=[{"name": "eq", "min": 1, "max": 5, "step": 2}],
    )

    assert response.status_code == 200
    assert service.calls[-1][0] == "validate_parameters"
    assert isinstance(service.calls[-1][1][0], tuple)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/v1.1/parameters/validate",
            [{"name": "eq", "min": "1", "max": 5, "step": 2}],
        ),
        (
            "/api/v1.1/sweeps/count",
            {
                "hardware": "I2C_SWEEP",
                "mode": "SINGLE_PARAMETER",
                "parameters": [
                    {
                        "name": "eq",
                        "min": 1,
                        "max": 5,
                        "step": 2,
                        "address": 0x2006,
                    }
                ],
                "seed": 7,
                "max_points": 10,
            },
        ),
    ],
)
def test_json_parameter_array_preserves_strict_elements(path, payload):
    client, service = client_and_service()

    response = client.post(path, json=payload)

    assert response.status_code == 422
    assert service.calls == []


def test_active_task_conflict_returns_409_and_current_task_id():
    client, service = client_and_service()
    service.failure = ActiveTaskConflict("running-task")

    response = client.post("/api/v1.1/tasks", json=ai_payload())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ACTIVE_HARDWARE_TASK"
    assert response.json()["detail"]["active_task_id"] == "running-task"


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (AuthorizationDriftError(("b0_hash",)), 409, "AUTHORIZATION_DRIFT"),
        (TaskStoreError("invalid lifecycle state"), 409, "TASK_STATE_CONFLICT"),
        (
            DcaConfigurationDrift(expected="dca-a", current="dca-b"),
            409,
            "DCA_CONFIGURATION_DRIFT",
        ),
        (UnknownTaskError("unknown task"), 404, "TASK_NOT_FOUND"),
    ],
)
def test_v11_domain_errors_have_stable_wire_codes(failure, status, code):
    client, service = client_and_service()
    service.failure = failure

    response = client.get("/api/v1.1/tasks/task-1")

    assert response.status_code == status
    assert response.json()["detail"] == {
        "code": code,
        "message": str(failure),
    }


def test_last_event_id_is_supported_by_v11_sse_route():
    client, _ = client_and_service()

    response = client.get(
        "/api/v1.1/tasks/task-1/events/stream",
        headers={"Last-Event-ID": "4"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_history_recovery_route_requires_literal_operator_confirmation():
    client, service = client_and_service()

    rejected = client.post(
        "/api/v1.1/recovery/history/recovery-1",
        json={"operator_confirmed": False},
    )
    accepted = client.post(
        "/api/v1.1/recovery/history/recovery-1",
        json={"operator_confirmed": True},
    )

    assert rejected.status_code == 422
    assert accepted.status_code == 200
    assert service.calls[-1][0] == "recover_history_result"
    assert service.calls[-1][1][0] == "recovery-1"


def test_task_recovery_route_preserves_abort_recovered_wire_action():
    client, service = client_and_service()

    accepted = client.post(
        "/api/v1.1/tasks/task-1/recovery",
        json={"action": "ABORT_RECOVERED", "operator_confirmed": True},
    )
    rejected_legacy = client.post(
        "/api/v1.1/tasks/task-1/recovery",
        json={"action": "ABORT", "operator_confirmed": True},
    )

    assert accepted.status_code == 200
    assert rejected_legacy.status_code == 422
    assert service.calls[-1][0] == "recover_task"
    assert service.calls[-1][1][1].action == "ABORT_RECOVERED"
