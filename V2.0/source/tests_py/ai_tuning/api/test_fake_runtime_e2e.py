from __future__ import annotations

import builtins
import socket
from threading import Thread
import time
from pathlib import Path

import httpx
import uvicorn

from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
from vcsel_ai_tuning.campaign.authorization import canonical_json_sha256
from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.runtime import create_fake_runtime
from vcsel_ai_tuning.runtime_config import RuntimeConfig


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_builtin_fake_runtime_completes_real_http_campaign_without_hardware(
    tmp_path: Path,
    monkeypatch,
):
    original_import = builtins.__import__

    def reject_hardware_import(name, *args, **kwargs):
        if name in {"pyvisa", "serial", "vcsel_ai_tuning.hardware"} or (
            name.startswith("vcsel_ai_tuning.hardware.")
        ):
            raise AssertionError(f"FAKE runtime imported hardware module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_hardware_import)
    port = _unused_local_port()
    config = RuntimeConfig(
        api={"host": "127.0.0.1", "port": port},
        storage={
            "sqlite_path": tmp_path / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=2,
    )
    api_runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: api_runtime)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level="error",
        )
    )
    thread = Thread(target=server.run, name="fake-runtime-http", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    baseline_hash = canonical_json_sha256(load_b0(config.b0_path))
    base_url = f"http://{config.api.host}:{config.api.port}"
    try:
        with httpx.Client(
            base_url=base_url,
            timeout=5.0,
            trust_env=False,
        ) as client:
            health = client.get("/health")
            capabilities = client.get("/capabilities")
            assert health.json()["runtime_mode"] == "FAKE"
            assert capabilities.json()["hardware_connected"] is False
            assert capabilities.json()["hierarchical_ready"] is False
            assert capabilities.json()["fir_proposer_ready"] is False

            created = client.post(
                "/api/campaigns",
                json={
                    "campaign_id": "fake-e2e",
                    "baseline_id": "B0",
                    "baseline_hash": baseline_hash,
                    "config_hash": "b" * 64,
                    "authorization_mode": "PER_STAGE",
                    "tuning_path": "PATH_EXHAUSTIVE",
                },
            )
            assert created.status_code == 201, created.text
            assert created.json()["state"] == "DRAFT"

            preflight = client.post("/api/campaigns/fake-e2e/preflight")
            assert preflight.status_code == 200, preflight.text
            assert preflight.json()["verified"] is True

            planned = client.post(
                "/api/campaigns/fake-e2e/stages/plan",
                json={
                    "stage": "RESTORE_B0",
                    "max_candidates": 1,
                    "max_duration_s": 30.0,
                },
            )
            assert planned.status_code == 200, planned.text
            authorization = planned.json()
            assert authorization["rollback_snapshot"]["readback_fir"]["taps"] == [
                0,
                9,
                -28,
                104,
                -8,
                1,
                -3,
            ]

            approved = client.post(
                "/api/campaigns/fake-e2e/stages/approve",
                json={"authorization": authorization},
            )
            assert approved.status_code == 200, approved.text
            assert approved.json()["status"] == "ACCEPTED"

            state = None
            for _ in range(100):
                campaign = client.get("/api/campaigns/fake-e2e")
                assert campaign.status_code == 200
                state = campaign.json()["state"]
                if state == "AWAIT_COARSE_APPROVAL":
                    break
                time.sleep(0.02)
            assert state == "AWAIT_COARSE_APPROVAL"

            observations = client.get("/api/campaigns/fake-e2e/observations")
            assert observations.status_code == 200
            assert len(observations.json()) == 5
            assert all(
                item["payload"]["metrics"]["tdecq_db"] is not None
                for item in observations.json()
            )

            diagnostic_plan = client.post(
                "/api/campaigns/fake-e2e/stages/plan",
                json={
                    "stage": "I2C_SCREENING",
                    "max_candidates": 2,
                    "max_duration_s": 30.0,
                },
            )
            assert diagnostic_plan.status_code == 200, diagnostic_plan.text
            diagnostic_approval = client.post(
                "/api/campaigns/fake-e2e/stages/approve",
                json={"authorization": diagnostic_plan.json()},
            )
            assert diagnostic_approval.status_code == 200

            diagnostic_observations = []
            for _ in range(100):
                response = client.get(
                    "/api/campaigns/fake-e2e/observations"
                )
                assert response.status_code == 200
                diagnostic_observations = response.json()
                if len(diagnostic_observations) == 7:
                    break
                time.sleep(0.02)
            assert len(diagnostic_observations) == 7, [
                (event.event_type, event.payload)
                for event in api_runtime.event_source.events("fake-e2e")
                if event.event_type.startswith("WORKER_")
                or "FAILED" in event.event_type
                or "MISMATCH" in event.event_type
                or event.event_type in {"STAGE_COMPLETED", "OBSERVATION_INVALID"}
            ]
            assert {
                item["payload"]["settings"]["requested_i2c"].get("eq")
                for item in diagnostic_observations[-2:]
            } == {21, 22}
            completed = client.get("/api/campaigns/fake-e2e")
            assert completed.json()["state"] == "COMPLETED"

            report = client.get("/api/campaigns/fake-e2e/report")
            assert report.status_code == 200
            assert report.json()["observation_count"] == 7

            with client.stream(
                "GET",
                "/api/campaigns/fake-e2e/events/stream",
                headers={"Last-Event-ID": "2"},
            ) as stream:
                assert stream.status_code == 200
                first_resumed_id = next(
                    line
                    for line in stream.iter_lines()
                    if line.startswith("id: ")
                )
            assert int(first_resumed_id.removeprefix("id: ")) > 2
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()

    restarted_api_runtime = create_fake_runtime(config)
    try:
        restarted = restarted_api_runtime.status_provider
        assert restarted.store.current_state("fake-e2e").value == "COMPLETED"
    finally:
        restarted_api_runtime.shutdown()
