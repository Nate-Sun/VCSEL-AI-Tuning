from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient
import pytest

from vcsel_ai_tuning.api.app import ApiRuntime, create_app, main
from vcsel_ai_tuning.campaign.authorization import (
    rollback_hash_for,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
    load_b0,
)
from vcsel_ai_tuning.runtime import ReadOnlyModeError
from datetime import datetime, timedelta, timezone


class RuntimeRouteService:
    def __init__(self) -> None:
        self.list_calls: list[tuple[int, int]] = []

    def list_campaigns(self, *, limit: int, offset: int):
        self.list_calls.append((limit, offset))
        return [
            {
                "campaign_id": "campaign-1",
                "state": CampaignState.DRAFT.value,
            }
        ]

    def create_campaign(self, request):
        return request

    def get_campaign(self, campaign_id):
        return {"campaign_id": campaign_id}

    def preflight(self, campaign_id):
        return {"campaign_id": campaign_id}

    def plan_stage(self, campaign_id, request):
        return request

    def approve_stage(self, campaign_id, authorization):
        return {"campaign_id": campaign_id}

    def pause_campaign(self, campaign_id):
        return {"campaign_id": campaign_id}

    def resume_campaign(self, campaign_id):
        return {"campaign_id": campaign_id}

    def abort_campaign(self, campaign_id):
        return {"campaign_id": campaign_id}

    def rollback_campaign(self, campaign_id):
        return {"campaign_id": campaign_id}

    def list_observations(self, campaign_id):
        return []

    def get_best_observation(self, campaign_id):
        return None

    def get_report(self, campaign_id):
        return {"campaign_id": campaign_id}


@dataclass
class FakeStatusProvider:
    closed: int = 0

    def health(self):
        return {
            "status": "ok",
            "runtime_mode": "FAKE",
            "store": {"ok": True},
            "repository": {"ok": True},
        }

    def capabilities_payload(self):
        return {
            "runtime_mode": "FAKE",
            "hardware_connected": False,
            "read_only": False,
            "exhaustive_ready": True,
            "tdecq_bayesian_ready": True,
            "hierarchical_ready": False,
            "fir_proposer_ready": True,
            "live_tuning_enabled": False,
            "authorization_modes": ["PER_STAGE"],
            "FULL_CAMPAIGN": False,
        }

    def close(self) -> None:
        self.closed += 1


def test_health_and_capabilities_are_read_only_and_do_not_leak_endpoints():
    service = RuntimeRouteService()
    status_provider = FakeStatusProvider()
    app = create_app(
        service,
        status_provider=status_provider,
        shutdown=status_provider.close,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        capabilities = client.get("/capabilities")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "runtime_mode": "FAKE",
        "store": {"ok": True},
        "repository": {"ok": True},
    }
    assert capabilities.status_code == 200
    assert capabilities.json()["hierarchical_ready"] is False
    assert capabilities.json()["fir_proposer_ready"] is True
    assert capabilities.json()["FULL_CAMPAIGN"] is False
    serialized = health.text + capabilities.text
    for secret in ("COM7", "TCPIP", "192.0.2.10", "password", "sqlite_path"):
        assert secret not in serialized
    assert status_provider.closed == 1


def test_campaign_list_route_validates_and_forwards_pagination():
    service = RuntimeRouteService()
    app = create_app(service, status_provider=FakeStatusProvider())
    client = TestClient(app)

    response = client.get("/api/campaigns?limit=25&offset=50")

    assert response.status_code == 200
    assert response.json() == [
        {"campaign_id": "campaign-1", "state": "DRAFT"}
    ]
    assert service.list_calls == [(25, 50)]
    assert client.get("/api/campaigns?limit=0").status_code == 422
    assert client.get("/api/campaigns?limit=101").status_code == 422
    assert client.get("/api/campaigns?offset=-1").status_code == 422


def test_runtime_mode_error_is_returned_as_stable_http_code():
    service = RuntimeRouteService()

    def reject(_campaign_id, _authorization):
        raise ReadOnlyModeError("READ_ONLY mode rejects writes")

    service.approve_stage = reject
    baseline = load_b0()
    snapshot = AppliedSettings(
        readback_i2c=baseline.i2c.model_dump(),
        readback_fir=baseline.fir,
    )
    now = datetime.now(timezone.utc)
    authorization = with_authorization_id(
        StageAuthorization(
            authorization_id="pending",
            campaign_id="campaign-1",
            mode=AuthorizationMode.PER_STAGE,
            stage=CampaignState.RESTORE_B0,
            baseline_hash="a" * 64,
            rollback_hash=rollback_hash_for(snapshot),
            max_candidates=1,
            max_duration_s=30.0,
            issued_at=now,
            expires_at=now + timedelta(minutes=1),
            rollback_snapshot=snapshot,
        )
    )
    client = TestClient(create_app(service))

    response = client.post(
        "/api/campaigns/campaign-1/stages/approve",
        json={"authorization": authorization.model_dump(mode="json")},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "READ_ONLY_MODE",
            "message": "READ_ONLY mode rejects writes",
        }
    }


def test_main_defaults_to_builtin_fake_runtime(monkeypatch):
    service = RuntimeRouteService()
    provider = FakeStatusProvider()
    api_runtime = ApiRuntime(
        service=service,
        event_source=SimpleEventSource(),
        status_provider=provider,
        shutdown=provider.close,
    )
    captured: dict[str, object] = {}
    monkeypatch.delenv("VCSEL_AI_CAMPAIGN_SERVICE_FACTORY", raising=False)
    monkeypatch.delenv("VCSEL_AI_RUNTIME_CONFIG", raising=False)
    monkeypatch.setattr(
        "vcsel_ai_tuning.runtime.create_fake_runtime",
        lambda: api_runtime,
    )
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, host, port: captured.update(
            {"app": app, "host": host, "port": port}
        ),
    )

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


class SimpleEventSource:
    def events(self, campaign_id: str, *, after_sequence: int = 0):
        return []

    def current_state(self, campaign_id: str):
        return CampaignState.DRAFT
