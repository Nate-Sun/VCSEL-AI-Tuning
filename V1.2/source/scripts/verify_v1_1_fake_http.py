"""Exercise the production V1.1 FastAPI contract over localhost in FAKE mode."""

from __future__ import annotations

import builtins
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Thread
import time

import httpx
import uvicorn

from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
from vcsel_ai_tuning.runtime import create_fake_runtime
from vcsel_ai_tuning.runtime_config import RuntimeConfig


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _print_exchange(method: str, path: str, request: object, response) -> None:
    print(f"REQUEST {method} {path} {json.dumps(request, sort_keys=True)}")
    print(
        f"RESPONSE {response.status_code} "
        f"{json.dumps(response.json(), sort_keys=True)}"
    )


def main() -> int:
    original_import = builtins.__import__

    def reject_hardware_import(name, *args, **kwargs):
        if name in {"pyvisa", "serial", "vcsel_ai_tuning.hardware"} or name.startswith(
            "vcsel_ai_tuning.hardware."
        ):
            raise RuntimeError(f"FAKE verification blocked hardware import: {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = reject_hardware_import
    with TemporaryDirectory(prefix="vcsel-v11-fake-http-") as temporary:
        root = Path(temporary)
        config = RuntimeConfig(
            api={"host": "127.0.0.1", "port": _port()},
            storage={
                "sqlite_path": root / "state" / "campaigns.sqlite3",
                "artifact_root": root / "artifacts",
            },
            b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
            worker_count=2,
        )
        api_runtime = create_fake_runtime(config)
        server = uvicorn.Server(
            uvicorn.Config(
                create_app_from_runtime_factory(lambda: api_runtime),
                host=config.api.host,
                port=config.api.port,
                log_level="error",
            )
        )
        thread = Thread(target=server.run, name="v11-fake-http-proof", daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("localhost Uvicorn did not start")

        parameters = [{"name": "eq", "min": 18, "max": 22, "step": 2}]
        task = {
            "task_id": "task-http-proof",
            "task_type": "SWEEP",
            "channel_mapping": {
                "i2c_channel": 0,
                "bert_port": "port1",
                "bert_channel": "0",
                "dca_tdecq_source": "CHAN1A",
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
        try:
            with httpx.Client(
                base_url=f"http://{config.api.host}:{config.api.port}",
                timeout=5,
                trust_env=False,
            ) as client:
                capabilities = client.get("/api/v1.1/runtime/capabilities")
                _print_exchange("GET", "/api/v1.1/runtime/capabilities", None, capabilities)
                if capabilities.json().get("runtime_mode") != "FAKE":
                    raise RuntimeError("verification runtime is not FAKE")
                if capabilities.json().get("hardware_mutation") is not False:
                    raise RuntimeError("FAKE runtime unexpectedly permits hardware mutation")

                snapshot = client.get("/api/v1.1/hardware/snapshot")
                snapshot.raise_for_status()
                client.post("/api/v1.1/mapping/scan").raise_for_status()
                client.post(
                    "/api/v1.1/parameters/validate", json=parameters
                ).raise_for_status()
                client.post(
                    "/api/v1.1/sweeps/count",
                    json={
                        "hardware": "I2C_SWEEP",
                        "mode": "SINGLE_PARAMETER",
                        "parameters": parameters,
                        "seed": 1042,
                        "max_points": 3,
                    },
                ).raise_for_status()

                created = client.post("/api/v1.1/tasks", json=task)
                _print_exchange("POST", "/api/v1.1/tasks", task, created)
                b0_request = {
                    "snapshot_token": snapshot.json()["snapshot_token"],
                    "operator_confirmed": True,
                }
                frozen = client.post(
                    "/api/v1.1/tasks/task-http-proof/b0", json=b0_request
                )
                _print_exchange(
                    "POST", "/api/v1.1/tasks/task-http-proof/b0", b0_request, frozen
                )
                authorization = {
                    "authorization_id": "auth-http-proof",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=30)
                    ).isoformat(),
                    "operator_confirmed": True,
                }
                authorized = client.post(
                    "/api/v1.1/tasks/task-http-proof/authorization",
                    json=authorization,
                )
                _print_exchange(
                    "POST",
                    "/api/v1.1/tasks/task-http-proof/authorization",
                    authorization,
                    authorized,
                )
                started = client.post("/api/v1.1/tasks/task-http-proof/start")
                _print_exchange(
                    "POST", "/api/v1.1/tasks/task-http-proof/start", None, started
                )

                expected = [
                    (created, 201, "DRAFT"),
                    (frozen, 200, "B0_FROZEN"),
                    (authorized, 200, "AUTHORIZED"),
                    (started, 200, "RUNNING"),
                ]
                for response, status, state in expected:
                    if response.status_code != status or response.json().get("state") != state:
                        raise RuntimeError(f"expected HTTP {status} {state}: {response.text}")
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("localhost Uvicorn did not stop")
            builtins.__import__ = original_import
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
