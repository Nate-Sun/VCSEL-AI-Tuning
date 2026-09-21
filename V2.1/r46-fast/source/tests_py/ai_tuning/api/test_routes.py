from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

import vcsel_ai_tuning.api.app as api_app
from vcsel_ai_tuning.api.app import create_app
from vcsel_ai_tuning.api import routes
from vcsel_ai_tuning.campaign.authorization import (
    AuthorizationError,
    rollback_hash_for,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    CampaignState,
    ParameterSpec,
    StageAuthorization,
    load_b0,
)
from vcsel_ai_tuning.campaign.orchestrator import OrchestratorError
from vcsel_ai_tuning.campaign.store import CampaignEvent, UnknownCampaignError


CAMPAIGN_ID = "campaign-api-1"
NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


class FakeCampaignService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, Any]] = []
        self.failures: dict[str, Exception] = {}

    def _result(
        self,
        operation: str,
        campaign_id: str | None = None,
        payload: Any = None,
    ) -> dict[str, Any]:
        self.calls.append((operation, campaign_id, payload))
        failure = self.failures.get(operation)
        if failure is not None:
            raise failure
        return {
            "operation": operation,
            "campaign_id": campaign_id,
        }

    def create_campaign(self, request: Any) -> dict[str, Any]:
        return self._result("create_campaign", request.campaign_id, request)

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._result("get_campaign", campaign_id)

    def preflight(self, campaign_id: str) -> dict[str, Any]:
        return self._result("preflight", campaign_id)

    def plan_stage(self, campaign_id: str, request: Any) -> dict[str, Any]:
        return self._result("plan_stage", campaign_id, request)

    def approve_stage(self, campaign_id: str, authorization: Any) -> dict[str, Any]:
        return self._result("approve_stage", campaign_id, authorization)

    def pause_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._result("pause_campaign", campaign_id)

    def resume_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._result("resume_campaign", campaign_id)

    def abort_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._result("abort_campaign", campaign_id)

    def rollback_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._result("rollback_campaign", campaign_id)

    def list_observations(self, campaign_id: str) -> dict[str, Any]:
        return self._result("list_observations", campaign_id)

    def get_best_observation(self, campaign_id: str) -> dict[str, Any]:
        return self._result("get_best_observation", campaign_id)

    def get_report(self, campaign_id: str) -> dict[str, Any]:
        return self._result("get_report", campaign_id)


class FakeEventSource:
    def __init__(self, events: list[CampaignEvent]) -> None:
        self.scripted_events = events
        self.calls: list[tuple[str, int]] = []

    def current_state(self, campaign_id: str) -> CampaignState | None:
        matching = [
            event for event in self.scripted_events if event.campaign_id == campaign_id
        ]
        return None if not matching else matching[-1].to_state

    def events(
        self,
        campaign_id: str,
        *,
        after_sequence: int = 0,
    ) -> list[CampaignEvent]:
        self.calls.append((campaign_id, after_sequence))
        return list(self.scripted_events)


class UnknownCampaignEventSource:
    def __init__(self) -> None:
        self.state_calls: list[str] = []
        self.event_calls: list[tuple[str, int]] = []

    def current_state(self, campaign_id: str) -> CampaignState | None:
        self.state_calls.append(campaign_id)
        return None

    def events(
        self,
        campaign_id: str,
        *,
        after_sequence: int = 0,
    ) -> list[CampaignEvent]:
        self.event_calls.append((campaign_id, after_sequence))
        raise AssertionError("unknown campaign must be rejected before event polling")


@pytest.fixture
def service() -> FakeCampaignService:
    return FakeCampaignService()


@pytest.fixture
def client(service: FakeCampaignService) -> TestClient:
    return TestClient(create_app(service))


def _create_payload() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "baseline_id": "B0",
        "baseline_hash": "b" * 64,
        "config_hash": "c" * 64,
        "authorization_mode": "PER_STAGE",
    }


def _authorization_payload(
    mode: AuthorizationMode = AuthorizationMode.PER_STAGE,
) -> dict[str, Any]:
    baseline = load_b0()
    rollback = AppliedSettings(
        readback_i2c=baseline.i2c.model_dump(),
        readback_fir=baseline.fir,
    )
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=mode,
        stage=CampaignState.I2C_COARSE,
        baseline_hash="b" * 64,
        rollback_hash=rollback_hash_for(rollback),
        parameter_specs=(
            ParameterSpec(
                name="ibias_dac",
                encoding_min=0,
                encoding_max=127,
                tuning_min=68,
                tuning_max=74,
                max_step=2,
            ),
        ),
        max_candidates=3,
        max_duration_s=30.0,
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned).model_dump(mode="json")


def _unbounded_authorization_payload(stage: CampaignState) -> dict[str, Any]:
    baseline = load_b0()
    rollback = AppliedSettings(
        readback_i2c=baseline.i2c.model_dump(),
        readback_fir=baseline.fir,
    )
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=AuthorizationMode.PER_STAGE,
        time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
        stage=stage,
        baseline_hash="b" * 64,
        rollback_hash=rollback_hash_for(rollback),
        max_candidates=1,
        max_duration_s=None,
        issued_at=NOW,
        expires_at=None,
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned).model_dump(mode="json")


def test_create_campaign_and_get_state_delegate_once_to_injected_service(
    client: TestClient,
    service: FakeCampaignService,
):
    created = client.post("/api/campaigns", json=_create_payload())
    state = client.get(f"/api/campaigns/{CAMPAIGN_ID}")

    assert created.status_code == 201
    assert created.json() == {
        "operation": "create_campaign",
        "campaign_id": CAMPAIGN_ID,
    }
    assert state.status_code == 200
    assert state.json()["operation"] == "get_campaign"
    assert [call[0] for call in service.calls] == [
        "create_campaign",
        "get_campaign",
    ]
    create_request = service.calls[0][2]
    assert create_request.model_dump(mode="json") == _create_payload()


@pytest.mark.parametrize(
    "authorization_mode",
    ["PER_CANDIDATE", "FULL_CAMPAIGN"],
)
def test_v1_rejects_non_stage_campaign_mode_before_service_invocation(
    client: TestClient,
    service: FakeCampaignService,
    authorization_mode: str,
):
    response = client.post(
        "/api/campaigns",
        json={
            **_create_payload(),
            "authorization_mode": authorization_mode,
        },
    )

    assert response.status_code == 422
    assert authorization_mode in response.text
    assert service.calls == []


@pytest.mark.parametrize(
    ("method", "path", "json_body", "service_method"),
    [
        ("post", "preflight", None, "preflight"),
        (
            "post",
            "stages/plan",
            {
                "stage": "I2C_COARSE",
                "max_candidates": 3,
                "max_duration_s": 30.0,
            },
            "plan_stage",
        ),
        ("post", "pause", None, "pause_campaign"),
        ("post", "resume", None, "resume_campaign"),
        ("post", "abort", None, "abort_campaign"),
        ("post", "rollback", None, "rollback_campaign"),
        ("get", "observations", None, "list_observations"),
        ("get", "best", None, "get_best_observation"),
        ("get", "report", None, "get_report"),
    ],
)
def test_semantic_route_delegates_synchronously_without_hidden_execution(
    client: TestClient,
    service: FakeCampaignService,
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    service_method: str,
):
    response = client.request(
        method,
        f"/api/campaigns/{CAMPAIGN_ID}/{path}",
        json=json_body,
    )

    assert response.status_code == 200
    assert response.json() == {
        "operation": service_method,
        "campaign_id": CAMPAIGN_ID,
    }
    assert [call[0] for call in service.calls] == [service_method]


def test_stage_approval_passes_strict_authorization_to_service(
    client: TestClient,
    service: FakeCampaignService,
):
    authorization_payload = _authorization_payload()

    response = client.post(
        f"/api/campaigns/{CAMPAIGN_ID}/stages/approve",
        json={"authorization": authorization_payload},
    )

    assert response.status_code == 200
    assert response.json()["operation"] == "approve_stage"
    assert [call[0] for call in service.calls] == ["approve_stage"]
    authorization = service.calls[0][2]
    assert isinstance(authorization, StageAuthorization)
    assert authorization.model_dump(mode="json") == authorization_payload


@pytest.mark.parametrize(
    "stage",
    [CampaignState.RESTORE_B0, CampaignState.I2C_SCREENING],
)
def test_stage_approval_route_rejects_unbounded_before_service_invocation(
    client: TestClient,
    service: FakeCampaignService,
    stage: CampaignState,
) -> None:
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_ID}/stages/approve",
        json={"authorization": _unbounded_authorization_payload(stage)},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "authorization_error"
    assert "bounded" in response.json()["detail"]["message"]
    assert "ACCEPTED" not in response.text
    assert service.calls == []


@pytest.mark.parametrize(
    "authorization_mode",
    [AuthorizationMode.PER_CANDIDATE, AuthorizationMode.FULL_CAMPAIGN],
)
def test_v1_rejects_non_stage_approval_before_service_invocation(
    client: TestClient,
    service: FakeCampaignService,
    authorization_mode: AuthorizationMode,
):
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_ID}/stages/approve",
        json={
            "authorization": _authorization_payload(authorization_mode)
        },
    )

    assert 400 <= response.status_code < 500
    assert authorization_mode.value in response.text
    assert service.calls == []


@pytest.mark.parametrize(
    ("payload"),
    [
        pytest.param(
            {**_create_payload(), "unexpected": "field"},
            id="extra-field",
        ),
        pytest.param(
            {**_create_payload(), "campaign_id": 123},
            id="strict-string",
        ),
    ],
)
def test_create_request_is_strict_and_forbids_extra_fields(
    client: TestClient,
    service: FakeCampaignService,
    payload: dict[str, Any],
):
    response = client.post("/api/campaigns", json=payload)

    assert response.status_code == 422
    assert service.calls == []


def test_stage_plan_rejects_bool_as_candidate_budget(
    client: TestClient,
    service: FakeCampaignService,
):
    response = client.post(
        f"/api/campaigns/{CAMPAIGN_ID}/stages/plan",
        json={
            "stage": "I2C_COARSE",
            "max_candidates": True,
            "max_duration_s": 30.0,
        },
    )

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    ("failure", "status_code", "error_code"),
    [
        (
            UnknownCampaignError("campaign missing"),
            404,
            "campaign_not_found",
        ),
        (
            AuthorizationError("authorization expired"),
            422,
            "authorization_error",
        ),
        (
            OrchestratorError("illegal campaign state"),
            409,
            "campaign_conflict",
        ),
    ],
)
def test_domain_errors_have_consistent_http_mapping(
    client: TestClient,
    service: FakeCampaignService,
    failure: Exception,
    status_code: int,
    error_code: str,
):
    service.failures["get_campaign"] = failure

    response = client.get(f"/api/campaigns/{CAMPAIGN_ID}")

    assert response.status_code == status_code
    assert response.json() == {
        "detail": {
            "code": error_code,
            "message": str(failure),
        }
    }


def test_route_module_is_thin_and_has_no_hardware_or_orchestrator_dependency():
    source = inspect.getsource(routes)

    assert "vcsel_ai_tuning.hardware" not in source
    assert "CampaignOrchestrator" not in source
    assert "BackgroundTasks" not in source


def test_sse_route_replays_after_last_event_id_without_starting_campaign_work(
    service: FakeCampaignService,
):
    event_source = FakeEventSource(
        [
            CampaignEvent(
                campaign_id=CAMPAIGN_ID,
                sequence=5,
                event_type="CAMPAIGN_COMPLETED",
                from_state=CampaignState.FINAL_VERIFICATION,
                to_state=CampaignState.COMPLETED,
                payload={"result": "SUCCESS"},
                created_at=NOW,
            )
        ]
    )
    client = TestClient(create_app(service, event_source=event_source))

    response = client.get(
        f"/api/campaigns/{CAMPAIGN_ID}/events/stream",
        headers={"Last-Event-ID": "4"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("id: 5\nevent: CAMPAIGN_COMPLETED\n")
    assert event_source.calls == [(CAMPAIGN_ID, 4)]
    assert service.calls == []


def test_sse_unknown_campaign_returns_404_before_stream_or_event_poll(
    service: FakeCampaignService,
):
    event_source = UnknownCampaignEventSource()
    client = TestClient(create_app(service, event_source=event_source))

    response = client.get(f"/api/campaigns/{CAMPAIGN_ID}/events/stream")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "campaign_not_found",
            "message": f"unknown campaign: {CAMPAIGN_ID}",
        }
    }
    assert event_source.state_calls == [CAMPAIGN_ID]
    assert event_source.event_calls == []
    assert service.calls == []


def test_api_runtime_factory_wires_service_and_event_source_into_app(
    service: FakeCampaignService,
):
    event_source = FakeEventSource(
        [
            CampaignEvent(
                campaign_id=CAMPAIGN_ID,
                sequence=1,
                event_type="CAMPAIGN_COMPLETED",
                from_state=CampaignState.FINAL_VERIFICATION,
                to_state=CampaignState.COMPLETED,
                payload={"result": "SUCCESS"},
                created_at=NOW,
            )
        ]
    )
    runtime = api_app.ApiRuntime(
        service=service,
        event_source=event_source,
    )

    app = api_app.create_app_from_runtime_factory(lambda: runtime)
    client = TestClient(app)
    state = client.get(f"/api/campaigns/{CAMPAIGN_ID}")
    stream = client.get(f"/api/campaigns/{CAMPAIGN_ID}/events/stream")

    assert state.status_code == 200
    assert stream.status_code == 200
    assert stream.text.startswith("id: 1\nevent: CAMPAIGN_COMPLETED\n")
    assert event_source.calls == [(CAMPAIGN_ID, 0)]
    assert [call[0] for call in service.calls] == ["get_campaign"]


@pytest.mark.parametrize("last_event_id", ["-1", "+1", "01", "1.0"])
def test_sse_route_rejects_invalid_last_event_id_as_422(
    service: FakeCampaignService,
    last_event_id: str,
):
    client = TestClient(create_app(service, event_source=FakeEventSource([])))

    response = client.get(
        f"/api/campaigns/{CAMPAIGN_ID}/events/stream",
        headers={"Last-Event-ID": last_event_id},
    )

    assert response.status_code == 422
    assert "Last-Event-ID" in response.text
    assert service.calls == []
