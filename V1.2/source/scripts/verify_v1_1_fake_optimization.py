"""Prove the V1.2 product's v1.1 FAKE API over loopback HTTP only."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import socket
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Any


BLOCKED_IMPORT_PREFIXES = (
    "pyvisa",
    "serial",
    "vcsel_ai_tuning.hardware.characterization",
    "vcsel_ai_tuning.hardware.dca_acquisition",
    "vcsel_ai_tuning.hardware.flexdca",
    "vcsel_ai_tuning.hardware.live_cli",
    "vcsel_ai_tuning.hardware.live_config",
    "vcsel_ai_tuning.hardware.live_workflow",
    "vcsel_ai_tuning.hardware.vd1124",
)

EXPECTED_FAKE_CAPABILITIES = {
    "product_version": "1.2.0",
    "schema_version": 3,
    "runtime_mode": "FAKE",
    "campaign_types": ["AI_OPTIMIZATION", "SWEEP"],
    "dynamic_b0": True,
    "full_campaign_authorization": True,
    "hardware_connected": False,
    "hardware_mutation": False,
    "fake": True,
    "sweep_execution_ready": True,
    "ai_execution_ready": True,
    "profile_options": [
        {
            "name": "FAST",
            "candidate_budget": 24,
            "max_alternating_rounds": 1,
        },
        {
            "name": "STANDARD",
            "candidate_budget": 64,
            "max_alternating_rounds": 3,
        },
        {
            "name": "DEEP",
            "candidate_budget": 128,
            "max_alternating_rounds": 5,
        },
        {
            "name": "ADVANCED",
            "candidate_budget": 192,
            "max_alternating_rounds": 5,
        },
    ],
}


class _BlockedHardwareImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in BLOCKED_IMPORT_PREFIXES
        ):
            raise RuntimeError(f"blocked real hardware import: {fullname}")
        return None


def _require_loopback(address: object, operation: str) -> None:
    if not isinstance(address, tuple) or not address:
        raise RuntimeError(f"blocked non-IP socket {operation}: {address!r}")
    host = address[0]
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise RuntimeError(f"blocked non-loopback socket {operation}: {address!r}")


_ORIGINAL_SOCKET = socket.socket


class _LoopbackOnlySocket(_ORIGINAL_SOCKET):
    def bind(self, address):
        _require_loopback(address, "bind")
        return super().bind(address)

    def connect(self, address):
        _require_loopback(address, "connect")
        return super().connect(address)

    def connect_ex(self, address):
        _require_loopback(address, "connect_ex")
        return super().connect_ex(address)

    def sendto(self, data, *args):
        address = args[-1] if args else None
        _require_loopback(address, "sendto")
        return super().sendto(data, *args)


def _emit(kind: str, **payload: object) -> None:
    print(
        json.dumps(
            {"kind": kind, **payload},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_exact_fake_capabilities(capabilities: object) -> None:
    assert capabilities == EXPECTED_FAKE_CAPABILITIES, (
        "unexpected V1.2 FAKE capabilities document: "
        f"expected {EXPECTED_FAKE_CAPABILITIES!r}, got {capabilities!r}"
    )


def _task_request(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
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
            {"name": "res_gain", "min": 10, "max": 14, "step": 2},
            {"name": "eq", "min": 20, "max": 24, "step": 2},
        ],
        "depth": "STANDARD",
        "candidate_budget": 6,
        "seed": 1042,
    }


def _response(client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    _emit(
        "http",
        method=method,
        path=path,
        status_code=response.status_code,
        response=body,
    )
    response.raise_for_status()
    return body


def _fresh_output_directory(argument: str | None, stack: ExitStack) -> Path:
    if argument is None:
        temporary = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="vcsel-v1.1-fake-proof-")
        )
        return Path(temporary).resolve()
    output = Path(argument).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"--output-dir must be empty: {output}")
    return output


def run_proof(output_dir: Path) -> dict[str, object]:
    sys.meta_path.insert(0, _BlockedHardwareImportFinder())
    socket.socket = _LoopbackOnlySocket

    import httpx
    import uvicorn

    from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
    from vcsel_ai_tuning.hardware import vit_bert
    from vcsel_ai_tuning.runtime import create_fake_runtime
    from vcsel_ai_tuning.runtime_config import RuntimeConfig

    def block_vit_operation(*_args, **_kwargs):
        raise RuntimeError("blocked real VIT BERT request/setFir")

    vit_bert.VitBertClient._request = block_vit_operation
    vit_bert.VitBertClient.set_fir = block_vit_operation

    project_root = Path(__file__).resolve().parents[1]
    config = RuntimeConfig(
        storage={
            "sqlite_path": output_dir / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": output_dir / "artifacts" / "ai_tuning",
        },
        b0_path=(project_root / "configs" / "ai_tuning" / "b0.yaml").resolve(),
        worker_count=1,
    )
    runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: runtime)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    server_errors: list[BaseException] = []

    def serve() -> None:
        try:
            server.run(sockets=[listener])
        except BaseException as error:
            server_errors.append(error)

    server_thread = Thread(target=serve, name="v1.1-fake-proof-http", daemon=False)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    task_id = "proof-fake-ai-optimization"
    task_path = f"/api/v1.1/tasks/{task_id}"
    try:
        with httpx.Client(base_url=base_url, timeout=5.0, trust_env=False) as client:
            deadline = time.monotonic() + 10
            while True:
                if server_errors:
                    raise RuntimeError("Uvicorn failed to start") from server_errors[0]
                try:
                    health = client.get("/health")
                    if health.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("loopback HTTP health did not become ready")
                time.sleep(0.02)

            _emit(
                "http",
                method="GET",
                path="/health",
                status_code=health.status_code,
                response=health.json(),
            )
            runtime_health = _response(client, "GET", "/api/v1.1/runtime/health")
            capabilities = _response(
                client, "GET", "/api/v1.1/runtime/capabilities"
            )
            _assert_exact_fake_capabilities(capabilities)
            assert runtime_health["runtime_mode"] == "FAKE"

            created = _response(
                client, "POST", "/api/v1.1/tasks", json=_task_request(task_id)
            )
            assert created["state"] == "DRAFT"
            snapshot = _response(client, "GET", "/api/v1.1/hardware/snapshot")
            frozen = _response(
                client,
                "POST",
                f"{task_path}/b0",
                json={
                    "snapshot_token": snapshot["snapshot_token"],
                    "operator_confirmed": True,
                },
            )
            assert frozen["state"] == "B0_FROZEN"
            authorized = _response(
                client,
                "POST",
                f"{task_path}/authorization",
                json={
                    "authorization_id": "auth-proof-fake-ai-optimization",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=30)
                    ).isoformat(),
                    "operator_confirmed": True,
                },
            )
            assert authorized["state"] == "AUTHORIZED"
            started = _response(client, "POST", f"{task_path}/start")
            assert started["state"] == "RUNNING"

            deadline = time.monotonic() + 15
            final = started
            while final["state"] not in {
                "COMPLETED",
                "COMPLETED_NO_IMPROVEMENT",
            }:
                if time.monotonic() >= deadline:
                    raise TimeoutError("FAKE optimization did not reach terminal state")
                time.sleep(0.02)
                response = client.get(task_path)
                response.raise_for_status()
                final = response.json()
            _emit(
                "http",
                method="GET",
                path=task_path,
                status_code=200,
                response=final,
            )

            observations = _response(
                client, "GET", f"{task_path}/observations"
            )
            best = _response(client, "GET", f"{task_path}/best")
            events = _response(client, "GET", f"{task_path}/events")
            final_report = _response(
                client, "GET", f"{task_path}/final-report"
            )

        candidate_rows = [
            row
            for row in observations
            if row["phase"] not in {"B0", "FINAL_VERIFICATION"}
        ]
        assert candidate_rows
        assert len(candidate_rows) <= 6
        assert best is not None
        assert best["observation_id"] in {
            row["observation_id"] for row in observations
        }
        assert final_report["state"] == final["state"]
        assert final_report["observations"] == observations
        assert any(event["event_type"] == "AI_EXECUTION_TERMINAL" for event in events)

        artifact_path = (
            config.storage.artifact_root.parent
            / "v1.2"
            / task_id
            / "optimization_result.json"
        ).resolve()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["ordered_observation_ids"] == [
            row["observation_id"] for row in observations
        ]
        assert artifact["observation_count"] == len(observations)
        assert artifact["candidate_budget"] == 6
        assert artifact["candidates_consumed"] == len(candidate_rows)
        assert artifact["best_state"] == best
        assert artifact["terminal_lifecycle"] == final["state"]
        assert artifact["best_applied_to_hardware"] is False
        assert artifact["restored_to_b0"] is True
        artifact_summary = {
            "schema": artifact["schema"],
            "schema_version": artifact["schema_version"],
            "task_id": artifact["task_id"],
            "observation_count": artifact["observation_count"],
            "candidates_consumed": artifact["candidates_consumed"],
            "best_observation_id": artifact["best_state"]["observation_id"],
            "terminal_lifecycle": artifact["terminal_lifecycle"],
            "best_applied_to_hardware": artifact["best_applied_to_hardware"],
            "restored_to_b0": artifact["restored_to_b0"],
        }
        _emit(
            "artifact",
            absolute_path=str(artifact_path),
            content_summary=artifact_summary,
        )
        return {
            "terminal_lifecycle": final["state"],
            "observation_count": len(observations),
            "candidate_count": len(candidate_rows),
            "artifact_path": str(artifact_path),
            "artifact": artifact_summary,
            "safety": {
                "runtime_mode": capabilities["runtime_mode"],
                "hardware_connected": capabilities["hardware_connected"],
                "hardware_mutation": capabilities["hardware_mutation"],
                "blocked_import_prefixes": list(BLOCKED_IMPORT_PREFIXES),
                "vit_bert_request_blocked": True,
                "network_scope": "127.0.0.1 only",
            },
        }
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        listener.close()
        if server_thread.is_alive():
            raise RuntimeError("Uvicorn proof thread did not stop")
        if server_errors:
            raise RuntimeError("Uvicorn proof server failed") from server_errors[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="Empty directory in which to retain the fresh DB and artifacts",
    )
    arguments = parser.parse_args()
    with ExitStack() as stack:
        output_dir = _fresh_output_directory(arguments.output_dir, stack)
        summary = run_proof(output_dir)
        _emit("proof_summary", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
