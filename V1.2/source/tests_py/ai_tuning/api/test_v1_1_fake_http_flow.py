from __future__ import annotations

import builtins
from datetime import datetime, timedelta, timezone
from pathlib import Path
import socket
from threading import Thread
import time

import httpx
import uvicorn

from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
from vcsel_ai_tuning.runtime import create_fake_runtime
from vcsel_ai_tuning.runtime_config import RuntimeConfig


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_v11_fake_runtime_authorizes_and_starts_over_real_http(
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
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
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
    thread = Thread(target=server.run, name="v11-fake-runtime-http", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    parameters = [{"name": "eq", "min": 18, "max": 22, "step": 2}]
    sweep_count = {
        "hardware": "I2C_SWEEP",
        "mode": "SINGLE_PARAMETER",
        "parameters": parameters,
        "seed": 1042,
        "max_points": 3,
    }
    create_task = {
        "task_id": "task-contract-e2e",
        "task_type": "SWEEP",
        "channel_mapping": {
            "i2c_channel": 0,
            "bert_port": "port1",
            "bert_channel": "0",
            "dca_tdecq_source": "FUNC1",
            "dca_tdecq_equalizer_function": 1,
            "dca_outer_er_source": "CHAN1A",
            "operator_confirmed": True,
        },
        "parameters": parameters,
        "depth": "STANDARD",
        "candidate_budget": 3,
        "sweep_hardware": "I2C_SWEEP",
        "sweep_mode": "SINGLE_PARAMETER",
        "seed": 1042,
        "max_points": 3,
    }
    base_url = f"http://{config.api.host}:{config.api.port}"
    try:
        with httpx.Client(base_url=base_url, timeout=5.0, trust_env=False) as client:
            capabilities = client.get("/api/v1.1/runtime/capabilities")
            snapshot = client.get("/api/v1.1/hardware/snapshot")
            snapshot_token = snapshot.json()["snapshot_token"]
            mapping = client.post(
                "/api/v1.1/mapping/scan",
                json={"snapshot_token": snapshot_token},
            )
            validated = client.post(
                "/api/v1.1/parameters/validate",
                json={
                    "snapshot_token": snapshot_token,
                    "parameters": parameters,
                },
            )
            sweep_count["snapshot_token"] = snapshot_token
            counted = client.post("/api/v1.1/sweeps/count", json=sweep_count)
            created = client.post("/api/v1.1/tasks", json=create_task)
            frozen = client.post(
                "/api/v1.1/tasks/task-contract-e2e/b0",
                json={
                    "snapshot_token": snapshot.json()["snapshot_token"],
                    "operator_confirmed": True,
                },
            )
            authorized = client.post(
                "/api/v1.1/tasks/task-contract-e2e/authorization",
                json={
                    "authorization_id": "auth-contract-e2e",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=30)
                    ).isoformat(),
                    "operator_confirmed": True,
                },
            )
            started = client.post("/api/v1.1/tasks/task-contract-e2e/start")

            assert capabilities.status_code == 200, capabilities.text
            assert capabilities.json()["runtime_mode"] == "FAKE"
            assert capabilities.json()["hardware_mutation"] is False
            assert snapshot.status_code == 200, snapshot.text
            assert mapping.status_code == 200, mapping.text
            assert validated.status_code == 200, validated.text
            assert counted.status_code == 200, counted.text
            assert created.status_code == 201, created.text
            assert created.json()["state"] == "DRAFT"
            assert frozen.status_code == 200, frozen.text
            assert frozen.json()["state"] == "B0_FROZEN"
            assert authorized.status_code == 200, authorized.text
            assert authorized.json()["state"] == "AUTHORIZED"
            assert started.status_code == 200, started.text
            assert started.json()["state"] == "RUNNING"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()
