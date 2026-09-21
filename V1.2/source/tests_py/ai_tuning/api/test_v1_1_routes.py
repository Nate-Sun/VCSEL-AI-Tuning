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
from vcsel_ai_tuning.v1_1.read_only_runtime import ReadOnlyPreflightError


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
    def scan_parameter_mapping(self, request): return self._call("scan_parameter_mapping", request)
    def validate_parameters(self, request): return self._call("validate_parameters", request)
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
    def dataset_file_path(self, task_id, bundle_id, file_name):
        return self._call("dataset_file_path", task_id, bundle_id, file_name)
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
            "dca_tdecq_equalizer_function": 1,
            "dca_outer_er_source": "CHAN1A",
            "operator_confirmed": True,
        },
        "parameters": [{"name": "eq", "min": 1, "max": 5, "step": 2}],
        "depth": "STANDARD",
        "candidate_budget": 20,
        "seed": 7,
    }


def authorization_payload(expires_at: object) -> dict[str, object]:
    return {
        "authorization_id": "auth-001",
        "expires_at": expires_at,
        "operator_confirmed": True,
    }


@pytest.mark.parametrize(
    "expires_at",
    ["2099-07-29T10:30:00+00:00", "2099-07-29T10:30:00Z"],
)
def test_authorization_accepts_rfc3339_json_strings(expires_at):
    client, service = client_and_service()

    response = client.post(
        "/api/v1.1/tasks/task-1/authorization",
        json=authorization_payload(expires_at),
    )

    assert response.status_code == 200
    request = service.calls[-1][1][1]
    assert request.expires_at.tzinfo is not None
    assert request.expires_at.utcoffset() is not None


@pytest.mark.parametrize(
    "expires_at",
    ["2099-07-29T10:30:00", "not-a-time", 123, True, None, {}, []],
)
def test_authorization_rejects_non_rfc3339_wire_values(expires_at):
    client, service = client_and_service()

    response = client.post(
        "/api/v1.1/tasks/task-1/authorization",
        json=authorization_payload(expires_at),
    )

    assert response.status_code == 422
    assert service.calls == []


def test_authorization_still_requires_literal_operator_confirmation():
    client, service = client_and_service()
    payload = authorization_payload("2099-07-29T10:30:00Z")
    payload["operator_confirmed"] = False

    response = client.post(
        "/api/v1.1/tasks/task-1/authorization",
        json=payload,
    )

    assert response.status_code == 422
    assert service.calls == []


def test_v11_routes_extend_same_app_and_publish_versioned_openapi():
    client, service = client_and_service()

    response = client.post("/api/v1.1/tasks", json=ai_payload())
    openapi = client.get("/openapi.json").json()

    assert response.status_code == 201
    assert response.json()["product_version"] == "1.1.0"
    assert "/api/campaigns" in openapi["paths"]
    assert "/api/v1.1/tasks" in openapi["paths"]
    assert openapi["info"]["version"] == "1.2.0"
    assert service.calls[0][0] == "create_task"


def test_create_task_dataset_recording_defaults_disabled():
    client, service = client_and_service()

    response = client.post("/api/v1.1/tasks", json=ai_payload())

    assert response.status_code == 201
    request = service.calls[0][1][0]
    assert request.save_dataset is False


@pytest.mark.parametrize("invalid", [0, 1, "true", None, [], {}])
def test_create_task_dataset_recording_requires_strict_boolean(invalid):
    client, service = client_and_service()
    payload = ai_payload()
    payload["save_dataset"] = invalid

    response = client.post("/api/v1.1/tasks", json=payload)

    assert response.status_code == 422
    assert service.calls == []


def test_hardware_snapshot_is_never_cacheable_or_prefetch_reusable():
    client, service = client_and_service()

    response = client.get("/api/v1.1/hardware/snapshot")

    assert response.status_code == 200
    assert response.headers["cache-control"] == (
        "no-store, no-cache, must-revalidate"
    )
    assert response.headers["pragma"] == "no-cache"
    assert service.calls == [("hardware_snapshot", (), {})]


def test_dataset_file_route_uses_only_service_resolved_paths(tmp_path):
    client, service = client_and_service()
    artifact = tmp_path / "raw_eye.png"
    artifact.write_bytes(b"png")
    service.dataset_file_path = lambda *_args: artifact

    response = client.get(
        "/api/v1.1/tasks/task-1/datasets/point-00000001/raw_eye.png"
    )

    assert response.status_code == 200
    assert response.content == b"png"


def test_frontend_cannot_supply_register_address_or_bit():
    client, service = client_and_service()
    payload = ai_payload()
    payload["parameters"][0]["address"] = 0x2006

    response = client.post("/api/v1.1/tasks", json=payload)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    "field",
    (
        "dca_tdecq_source",
        "dca_tdecq_equalizer_function",
        "dca_outer_er_source",
    ),
)
def test_operator_must_explicitly_confirm_both_dca_mapping_sources(field):
    mapping = ai_payload()["channel_mapping"]
    del mapping[field]

    with pytest.raises(ValidationError):
        ChannelMappingRequest.model_validate(mapping)


@pytest.mark.parametrize(
    ("stage", "message"),
    (
        ("dca", "TDECQ source mismatch"),
        ("timeout", "Preflight total timeout exceeded"),
        ("i2c", "I2C page restore verification failed"),
    ),
)
def test_read_only_preflight_failures_have_stable_http_contract(stage, message):
    client, service = client_and_service()
    service.failure = ReadOnlyPreflightError(
        message,
        stage=stage,
        failure_artifact_id="failed-20260731T045949.255894Z",
        failure_artifact_path=(
            "preflight/failed-20260731T045949.255894Z.json"
        ),
    )

    response = client.get("/api/v1.1/hardware/snapshot")

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "READ_ONLY_PREFLIGHT_FAILED",
            "message": message,
            "stage": stage,
            "failure_artifact_id": "failed-20260731T045949.255894Z",
            "failure_artifact_path": (
                "preflight/failed-20260731T045949.255894Z.json"
            ),
        }
    }
    assert "Traceback" not in response.text


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
            "snapshot_token": "snapshot-1",
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


def test_parameter_validation_requires_snapshot_token_and_normalizes_json_array():
    client, service = client_and_service()

    response = client.post(
        "/api/v1.1/parameters/validate",
        json={
            "snapshot_token": "snapshot-1",
            "parameters": [{"name": "eq", "min": 1, "max": 5, "step": 2}],
        },
    )

    assert response.status_code == 200
    assert service.calls[-1][0] == "validate_parameters"
    request = service.calls[-1][1][0]
    assert request.snapshot_token == "snapshot-1"
    assert isinstance(request.parameters, tuple)


def test_mapping_scan_requires_snapshot_token():
    client, service = client_and_service()

    missing = client.post("/api/v1.1/mapping/scan", json={})
    accepted = client.post(
        "/api/v1.1/mapping/scan",
        json={"snapshot_token": "snapshot-1"},
    )

    assert missing.status_code == 422
    assert accepted.status_code == 200
    assert service.calls[-1][1][0].snapshot_token == "snapshot-1"


def test_history_apply_accepts_persisted_colon_separated_observation_id():
    client, service = client_and_service()
    observation_id = "formal-fake-optimization:00000002:i2c_coarse"

    response = client.post(
        "/api/v1.1/history/apply",
        json={
            "source_task_id": "formal-fake-optimization",
            "observation_id": observation_id,
        },
    )

    assert response.status_code == 200, response.text
    request = service.calls[-1][1][0]
    assert request.observation_id == observation_id


@pytest.mark.parametrize(
    "observation_id",
    [
        "task/observation",
        r"task\observation",
        "task observation",
        "task\tobservation",
        "task\nobservation",
        "../observation",
        r"..\observation",
        "task:..:observation",
        "a" * 257,
    ],
)
def test_history_apply_rejects_unsafe_observation_id(observation_id):
    client, service = client_and_service()

    response = client.post(
        "/api/v1.1/history/apply",
        json={
            "source_task_id": "formal-fake-optimization",
            "observation_id": observation_id,
        },
    )

    assert response.status_code == 422
    assert service.calls == []


def test_common_task_identifier_still_rejects_colons():
    client, service = client_and_service()
    payload = ai_payload()
    payload["task_id"] = "task:with:colons"

    response = client.post("/api/v1.1/tasks", json=payload)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/v1.1/parameters/validate",
            {
                "snapshot_token": "snapshot-1",
                "parameters": [
                    {"name": "eq", "min": "1", "max": 5, "step": 2}
                ],
            },
        ),
        (
            "/api/v1.1/sweeps/count",
            {
                "snapshot_token": "snapshot-1",
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
