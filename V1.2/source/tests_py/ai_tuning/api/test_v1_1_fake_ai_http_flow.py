from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import builtins
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient
import scripts.verify_v1_1_fake_optimization as proof_script

from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
from vcsel_ai_tuning.runtime import create_fake_runtime, create_runtime
from vcsel_ai_tuning.runtime_config import RuntimeConfig
from vcsel_ai_tuning.v1_1.domain import (
    LifecycleState,
    RuntimeMode,
    TaskType,
)
from vcsel_ai_tuning.v1_1.fake_ai_execution import (
    FakeOptimizationPlant,
    FakeV11AIExecution,
)
from vcsel_ai_tuning.v1_1.fake_runtime import FakeV11Gateway
from vcsel_ai_tuning.v1_1.persistent_runner import PersistentV11TaskRunner
from vcsel_ai_tuning.v1_1.repository import TaskStore, TaskStoreError


class _HttpCandidateGatePlant(FakeOptimizationPlant):
    def __init__(self, b0, *, block_first_candidate: bool = True) -> None:
        super().__init__(b0)
        self.block_first_candidate = block_first_candidate
        self.measure_calls = 0
        self.candidate_measurement_started = Event()
        self.allow_candidate_measurement = Event()

    def measure(self):
        self.measure_calls += 1
        if self.block_first_candidate and self.measure_calls == 2:
            self.candidate_measurement_started.set()
            if not self.allow_candidate_measurement.wait(timeout=5):
                raise TimeoutError("HTTP candidate gate was not released")
        return super().measure()


class _StopObservedEvent(Event):
    def __init__(self) -> None:
        super().__init__()
        self.set_called = Event()

    def set(self) -> None:
        super().set()
        self.set_called.set()


def _ai_task_request(task_id: str) -> dict[str, object]:
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


def _six_point_sweep_request(task_id: str) -> dict[str, object]:
    request = _ai_task_request(task_id)
    return {
        **request,
        "task_type": "SWEEP",
        "parameters": [
            {"name": "eq", "min": 18, "max": 28, "step": 2},
        ],
        "sweep_hardware": "I2C_SWEEP",
        "sweep_mode": "SINGLE_PARAMETER",
        "max_points": 6,
    }


def _freeze_authorize_start(client, task_id: str):
    task_path = f"/api/v1.1/tasks/{task_id}"
    snapshot = client.get("/api/v1.1/hardware/snapshot")
    assert snapshot.status_code == 200, snapshot.text
    frozen = client.post(
        f"{task_path}/b0",
        json={
            "snapshot_token": snapshot.json()["snapshot_token"],
            "operator_confirmed": True,
        },
    )
    assert frozen.status_code == 200, frozen.text
    assert frozen.json()["state"] == "B0_FROZEN"
    authorized = client.post(
        f"{task_path}/authorization",
        json={
            "authorization_id": f"auth-{task_id}",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=30)
            ).isoformat(),
            "operator_confirmed": True,
        },
    )
    assert authorized.status_code == 200, authorized.text
    assert authorized.json()["state"] == "AUTHORIZED"
    started = client.post(f"{task_path}/start")
    assert started.status_code == 200, started.text
    assert started.json()["state"] == "RUNNING"
    return frozen, started


def test_expired_authorization_returns_stable_conflict_and_stays_frozen(
    tmp_path: Path,
):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: runtime)
    task_id = "expired-authorization"
    task_path = f"/api/v1.1/tasks/{task_id}"

    with TestClient(app) as client:
        created = client.post("/api/v1.1/tasks", json=_ai_task_request(task_id))
        snapshot = client.get("/api/v1.1/hardware/snapshot")
        frozen = client.post(
            f"{task_path}/b0",
            json={
                "snapshot_token": snapshot.json()["snapshot_token"],
                "operator_confirmed": True,
            },
        )
        rejected = client.post(
            f"{task_path}/authorization",
            json={
                "authorization_id": "auth-expired",
                "expires_at": (
                    datetime.now(timezone.utc) - timedelta(minutes=1)
                ).isoformat(),
                "operator_confirmed": True,
            },
        )

        assert created.status_code == 201, created.text
        assert snapshot.status_code == 200, snapshot.text
        assert frozen.status_code == 200, frozen.text
        assert rejected.status_code == 409, rejected.text
        assert rejected.json()["detail"] == {
            "code": "TASK_STATE_CONFLICT",
            "message": "expires_at must be after issued_at",
        }
        assert client.get(task_path).json()["state"] == "B0_FROZEN"
        with pytest.raises(TaskStoreError, match="has no authorization"):
            runtime.v11_service.store.authorization_payload(task_id)


def test_completed_sweep_and_optimization_can_freeze_identical_b0(tmp_path: Path):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: runtime)

    with TestClient(app) as client:
        sweep_id = "six-point-sweep"
        created = client.post(
            "/api/v1.1/tasks", json=_six_point_sweep_request(sweep_id)
        )
        assert created.status_code == 201, created.text
        _freeze_authorize_start(client, sweep_id)
        assert runtime.v11_service.runner.wait_for_idle(timeout=10)
        sweep = client.get(f"/api/v1.1/tasks/{sweep_id}")
        assert sweep.status_code == 200, sweep.text
        assert sweep.json()["state"] == "COMPLETED"
        points = client.get(f"/api/v1.1/tasks/{sweep_id}/sweep-points")
        assert points.status_code == 200, points.text
        assert len(points.json()["items"]) == 6
        point_payload = points.json()["items"][0]
        assert {
            "write_readback_duration_s",
            "settle_duration_s",
            "dca_duration_s",
            "result_persistence_duration_s",
            "restore_duration_s",
            "candidate_round_time_s",
            "screening_round_with_restore_s",
            "restore_readback_result",
            "restore_status",
            "evidence_persistence_error",
        }.issubset(point_payload)

        optimization_id = "same-b0-optimization"
        created = client.post(
            "/api/v1.1/tasks", json=_ai_task_request(optimization_id)
        )
        assert created.status_code == 201, created.text
        frozen, started = _freeze_authorize_start(client, optimization_id)
        assert frozen.json()["b0_hash"] == sweep.json()["b0_hash"]
        assert started.json()["state"] == "RUNNING"


def test_fake_http_sweep_optionally_publishes_point_dataset_bundles(
    tmp_path: Path,
):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: runtime)

    with TestClient(app) as client:
        task_id = "fake-dataset-http"
        request = _six_point_sweep_request(task_id)
        request["save_dataset"] = True
        created = client.post("/api/v1.1/tasks", json=request)
        assert created.status_code == 201, created.text
        assert created.json()["config"]["save_dataset"] is True
        _freeze_authorize_start(client, task_id)
        assert runtime.v11_service.runner.wait_for_idle(timeout=10)

        response = client.get(f"/api/v1.1/tasks/{task_id}/sweep-points")
        assert response.status_code == 200, response.text
        points = response.json()["items"]
        assert len(points) == 6
        assert all(item["measurement_status"] == "SUCCESS" for item in points)
        assert all(item["dataset_status"] == "COMPLETE" for item in points)
        dataset_root = config.storage.artifact_root.parent / "v1.2" / task_id
        for point in points:
            bundle = dataset_root / point["point_id"]
            assert set(path.name for path in bundle.iterdir()) == {
                "record.json",
                "raw_eye_density.npy",
                "raw_eye.png",
                "tdecq_equalized_eye.png",
            }
            record = json.loads(
                (bundle / "record.json").read_text(encoding="utf-8")
            )
            assert record["measurement_id"] == point["measurement_id"]
            assert record["acquisition_id"] == point["acquisition_id"]

        sweep_json = json.loads(
            (dataset_root / "sweep_results.json").read_text(encoding="utf-8")
        )
        assert sweep_json["points"] == points


def test_fake_http_optimization_publishes_each_iteration_dataset(tmp_path: Path):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: runtime)

    with TestClient(app) as client:
        task_id = "fake-ai-dataset-http"
        request = _ai_task_request(task_id)
        request["save_dataset"] = True
        created = client.post("/api/v1.1/tasks", json=request)
        assert created.status_code == 201, created.text
        assert created.json()["dataset_status"] == "PENDING"
        _freeze_authorize_start(client, task_id)
        assert runtime.v11_service.runner.wait_for_idle(timeout=10)

        response = client.get(f"/api/v1.1/tasks/{task_id}")
        assert response.status_code == 200, response.text
        document = response.json()
        assert document["measurement_status"] == "SUCCESS"
        assert document["dataset_status"] == "SUCCESS"
        observations = client.get(
            f"/api/v1.1/tasks/{task_id}/observations"
        ).json()
        record_files = [
            item
            for item in document["dataset_files"]
            if item["kind"] == "MEASUREMENT_RECORD_JSON"
        ]
        assert len(record_files) == len(observations)
        assert all(item["observation_id"] for item in record_files)
        downloaded = client.get(record_files[0]["url"])
        assert downloaded.status_code == 200
        record = downloaded.json()
        assert record["observation_id"] == record_files[0]["observation_id"]


class _StopAfterFirstCandidatePlant(FakeOptimizationPlant):
    def __init__(self, b0, *, stop_event: Event) -> None:
        super().__init__(b0)
        self._stop_event = stop_event
        self._measure_calls = 0

    def measure(self):
        result = super().measure()
        self._measure_calls += 1
        if self._measure_calls == 2:
            self._stop_event.set()
        return result


def test_first_committed_candidate_exports_database_backed_artifact(tmp_path: Path):
    task_id = "first-candidate-artifact"
    database_path = tmp_path / "state" / "campaigns.sqlite3"
    artifact_root = tmp_path / "artifacts" / "v1.1"
    request = _ai_task_request(task_id)
    task_config = {
        key: request[key]
        for key in ("parameters", "depth", "candidate_budget", "seed")
    }
    b0 = FakeV11Gateway().preflight_capture().b0
    with TaskStore(database_path) as store:
        store.create_task(
            task_id=task_id,
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.FAKE,
            config_fingerprint="c" * 64,
            payload=task_config,
        )
        store.save_b0(task_id, b0)
    stop = Event()
    plant = _StopAfterFirstCandidatePlant(b0, stop_event=stop)
    execution = FakeV11AIExecution(
        task_id,
        b0,
        task_config,
        database_path,
        artifact_root,
        plant_factory=lambda _b0: plant,
    )

    assert execution.run(stop) is None

    artifact_path = artifact_root / task_id / "optimization_result.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    with TaskStore(database_path) as store:
        task = store.require_task(task_id)
        observations = store.observations(task_id)
        best = store.best_state(task_id)
    candidate_rows = [row for row in observations if row.phase != "B0"]
    assert len(candidate_rows) == 1
    assert artifact["schema"] == "vcsel_ai_tuning.v1_1.optimization_result"
    assert artifact["schema_version"] == task.schema_version
    assert artifact["task_id"] == task_id
    assert artifact["configuration_fingerprint"] == task.config_fingerprint
    assert artifact["b0_hash"] == b0.content_hash == task.b0_hash
    assert artifact["ordered_observation_ids"] == [
        row.observation_id for row in observations
    ]
    assert artifact["observation_count"] == len(observations)
    assert artifact["counts_by_phase"] == dict(
        Counter(row.phase for row in observations)
    )
    assert artifact["candidate_budget"] == task_config["candidate_budget"]
    assert artifact["candidates_consumed"] == 1
    assert artifact["best_state"]["observation_id"] == best.observation_id
    assert artifact["terminal_lifecycle"] is None
    assert artifact["best_applied_to_hardware"] is False
    assert artifact["restored_to_b0"] is False


def test_real_http_fake_optimization_proof_script(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / "proof"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str((project_root / "src").resolve()),
                environment.get("PYTHONPATH"),
            ),
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_v1_1_fake_optimization.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    messages = [json.loads(line) for line in completed.stdout.splitlines()]
    summary = messages[-1]
    assert summary["kind"] == "proof_summary"
    assert summary["terminal_lifecycle"] in {
        "COMPLETED",
        "COMPLETED_NO_IMPROVEMENT",
    }
    assert summary["artifact"]["restored_to_b0"] is True
    assert summary["artifact"]["best_applied_to_hardware"] is False
    assert Path(summary["artifact_path"]).is_file()


def test_proof_requires_the_complete_exact_capabilities_document(tmp_path: Path):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    runtime = create_fake_runtime(config)
    try:
        capabilities = runtime.v11_service.runtime_capabilities()
        proof_script._assert_exact_fake_capabilities(capabilities)
        with pytest.raises(AssertionError):
            proof_script._assert_exact_fake_capabilities(
                {**capabilities, "unexpected_capability": True}
            )
        missing = dict(capabilities)
        missing.pop("fake")
        with pytest.raises(AssertionError):
            proof_script._assert_exact_fake_capabilities(missing)
    finally:
        runtime.shutdown()


def test_artifact_replace_failure_is_nonfatal_and_leaves_no_partial_or_temp_file(
    tmp_path: Path,
    monkeypatch,
):
    task_id = "atomic-artifact-failure"
    database_path = tmp_path / "state" / "campaigns.sqlite3"
    artifact_root = tmp_path / "artifacts" / "v1.1"
    request = _ai_task_request(task_id)
    task_config = {
        key: request[key]
        for key in ("parameters", "depth", "candidate_budget", "seed")
    }
    b0 = FakeV11Gateway().preflight_capture().b0
    with TaskStore(database_path) as store:
        store.create_task(
            task_id=task_id,
            task_type=TaskType.AI_OPTIMIZATION,
            runtime_mode=RuntimeMode.FAKE,
            config_fingerprint="d" * 64,
            payload=task_config,
        )
        store.save_b0(task_id, b0)
    execution = FakeV11AIExecution(
        task_id,
        b0,
        task_config,
        database_path,
        artifact_root,
    )

    def fail_replace(_source, _target):
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.v1_1.fake_ai_execution.os.replace",
        fail_replace,
    )

    terminal = execution.run(Event())

    task_directory = artifact_root / task_id
    assert terminal in {
        LifecycleState.COMPLETED,
        LifecycleState.COMPLETED_NO_IMPROVEMENT,
    }
    assert not (task_directory / "optimization_result.json").exists()
    assert not (task_directory / ".optimization_result.json.tmp").exists()
    with TaskStore(database_path) as store:
        failures = [
            event
            for event in store.events(task_id)
            if event.event_type == "AI_ARTIFACT_EXPORT_FAILED"
        ]
    assert failures
    assert all("stage" in event.payload for event in failures)


def test_default_fake_runtime_completes_ai_optimization_over_http(
    tmp_path: Path,
    monkeypatch,
):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    api_runtime = create_fake_runtime(config)
    app = create_app_from_runtime_factory(lambda: api_runtime)
    task_id = "fake-ai-http-flow"
    task_path = f"/api/v1.1/tasks/{task_id}"

    with TestClient(app) as client:
        created = client.post(
            "/api/v1.1/tasks",
            json=_ai_task_request(task_id),
        )
        assert created.status_code == 201, created.text
        assert created.json()["state"] == "DRAFT"

        snapshot = client.get("/api/v1.1/hardware/snapshot")
        assert snapshot.status_code == 200, snapshot.text
        frozen = client.post(
            f"{task_path}/b0",
            json={
                "snapshot_token": snapshot.json()["snapshot_token"],
                "operator_confirmed": True,
            },
        )
        assert frozen.status_code == 200, frozen.text
        assert frozen.json()["state"] == "B0_FROZEN"

        authorized = client.post(
            f"{task_path}/authorization",
            json={
                "authorization_id": "auth-fake-ai-http-flow",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=30)
                ).isoformat(),
                "operator_confirmed": True,
            },
        )
        assert authorized.status_code == 200, authorized.text
        assert authorized.json()["state"] == "AUTHORIZED"

        started = client.post(f"{task_path}/start")
        assert started.status_code == 200, started.text
        assert started.json()["state"] == "RUNNING"

        terminal_states = {
            state.value for state in LifecycleState if not state.is_nonterminal
        }
        deadline = time.monotonic() + 10
        task = started.json()
        while task["state"] not in terminal_states and time.monotonic() < deadline:
            time.sleep(0.01)
            response = client.get(task_path)
            assert response.status_code == 200, response.text
            task = response.json()

        assert task["state"] in {"COMPLETED", "COMPLETED_NO_IMPROVEMENT"}

        observations = client.get(f"{task_path}/observations")
        best = client.get(f"{task_path}/best")
        events = client.get(f"{task_path}/events")
        final_report = client.get(f"{task_path}/final-report")

        assert observations.status_code == 200, observations.text
        assert len(observations.json()) > 0
        assert best.status_code == 200, best.text
        assert best.json() is not None
        assert events.status_code == 200, events.text
        event_types = [event["event_type"] for event in events.json()]
        assert "TASK_STARTED" in event_types
        assert "AI_EXECUTION_TERMINAL" in event_types
        assert final_report.status_code == 200, final_report.text
        assert final_report.json()["task_id"] == task_id
        assert final_report.json()["state"] == task["state"]
        assert final_report.json()["observations"] == observations.json()

        artifact_path = (
            config.storage.artifact_root.parent
            / "v1.2"
            / task_id
            / "optimization_result.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        observation_rows = observations.json()
        candidate_rows = [
            row
            for row in observation_rows
            if row["phase"] not in {"B0", "FINAL_VERIFICATION"}
        ]
        assert artifact["ordered_observation_ids"] == [
            row["observation_id"] for row in observation_rows
        ]
        assert artifact["observation_count"] == len(observation_rows)
        assert artifact["counts_by_phase"] == dict(
            Counter(row["phase"] for row in observation_rows)
        )
        assert artifact["candidate_budget"] == 6
        assert artifact["candidates_consumed"] == len(candidate_rows)
        assert 0 < artifact["candidates_consumed"] <= artifact["candidate_budget"]
        assert artifact["best_state"] == best.json()
        assert artifact["best_state"]["observation_id"] in {
            row["observation_id"] for row in observation_rows
        }
        assert artifact["terminal_lifecycle"] == task["state"]
        assert artifact["best_applied_to_hardware"] is False
        assert artifact["restored_to_b0"] is True

        original_import = builtins.__import__

        def reject_real_hardware(name, *args, **kwargs):
            if name in {"pyvisa", "serial", "vcsel_ai_tuning.hardware"} or name.startswith(
                "vcsel_ai_tuning.hardware."
            ):
                raise AssertionError(
                    f"FAKE History Apply imported hardware module {name}"
                )
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", reject_real_hardware)
        observation_id = best.json()["observation_id"]
        assert observation_id.startswith(f"{task_id}:")
        history_request = {
            "source_task_id": task_id,
            "observation_id": observation_id,
        }
        first_apply = client.post(
            "/api/v1.1/history/apply", json=history_request
        )
        assert first_apply.status_code == 200, first_apply.text
        first_receipt = first_apply.json()
        assert first_receipt["status"] == "HISTORY_RESULT_APPLIED"
        assert first_receipt["fake"] is True
        assert first_receipt["observation_id"] == observation_id
        assert first_receipt["i2c_readback"] == best.json()["i2c"]
        assert first_receipt["fir_readback"] == best.json()["fir"]
        assert first_receipt["measurement"]["fake"] is True

        second_apply = client.post(
            "/api/v1.1/history/apply", json=history_request
        )
        assert second_apply.status_code == 200, second_apply.text
        second_receipt = second_apply.json()
        assert second_receipt["fake"] is True
        assert second_receipt["i2c_readback"] == first_receipt["i2c_readback"]
        assert second_receipt["fir_readback"] == first_receipt["fir_readback"]
        assert second_receipt["measurement"] == first_receipt["measurement"]

        history_events = client.get(f"{task_path}/events")
        assert history_events.status_code == 200, history_events.text
        applied_events = [
            event
            for event in history_events.json()
            if event["event_type"] == "HISTORY_RESULT_APPLIED"
        ]
        assert len(applied_events) == 2
        assert all(event["payload"]["fake"] is True for event in applied_events)
        assert applied_events[-1]["payload"] == second_receipt

        persisted_task = task
        persisted_observations = observations.json()
        persisted_best = best.json()
        persisted_events = history_events.json()

    restarted_runtime = create_fake_runtime(config)
    restarted_app = create_app_from_runtime_factory(lambda: restarted_runtime)
    with TestClient(restarted_app) as restarted_client:
        restarted_task = restarted_client.get(task_path)
        restarted_observations = restarted_client.get(
            f"{task_path}/observations"
        )
        restarted_best = restarted_client.get(f"{task_path}/best")
        restarted_events = restarted_client.get(f"{task_path}/events")

        assert restarted_task.status_code == 200, restarted_task.text
        assert restarted_task.json() == persisted_task
        assert restarted_observations.status_code == 200
        assert restarted_observations.json() == persisted_observations
        assert restarted_best.status_code == 200
        assert restarted_best.json() == persisted_best
        assert restarted_events.status_code == 200
        assert restarted_events.json() == persisted_events
        assert "applied" not in restarted_best.json()
        assert "hardware_applied" not in restarted_best.json()


def test_second_http_task_is_rejected_while_real_fake_ai_owns_execution(
    tmp_path: Path,
):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "ai_tuning" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "ai_tuning",
        },
        b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
        worker_count=1,
    )
    database_path = (
        config.storage.sqlite_path.parent.parent
        / "v1.1"
        / "campaigns.sqlite3"
    )
    artifact_root = config.storage.artifact_root.parent / "v1.1"
    plants: list[_HttpCandidateGatePlant] = []
    task_execution_counts: dict[str, int] = {}

    def ai_factory(task_id, b0, payload):
        execution_count = task_execution_counts.get(task_id, 0)
        task_execution_counts[task_id] = execution_count + 1
        plant = _HttpCandidateGatePlant(
            b0,
            block_first_candidate=execution_count == 0,
        )
        plants.append(plant)
        return FakeV11AIExecution(
            task_id,
            b0,
            payload,
            database_path,
            artifact_root,
            plant_factory=lambda _b0: plant,
        )

    runner = PersistentV11TaskRunner(
        database_path=database_path,
        artifact_root=artifact_root,
        ai_execution_factory=ai_factory,
    )
    stop_observed = _StopObservedEvent()
    runner._stop_requested = stop_observed
    runtime = create_runtime(
        config,
        v11_gateway=FakeV11Gateway(),
        v11_runner=runner,
    )
    app = create_app_from_runtime_factory(lambda: runtime)
    first_task_id = "first-http-owner"
    first_path = f"/api/v1.1/tasks/{first_task_id}"

    with TestClient(app) as client:
        try:
            created = client.post(
                "/api/v1.1/tasks",
                json=_ai_task_request(first_task_id),
            )
            assert created.status_code == 201, created.text
            snapshot = client.get("/api/v1.1/hardware/snapshot")
            assert snapshot.status_code == 200, snapshot.text
            frozen = client.post(
                f"{first_path}/b0",
                json={
                    "snapshot_token": snapshot.json()["snapshot_token"],
                    "operator_confirmed": True,
                },
            )
            assert frozen.status_code == 200, frozen.text
            authorized = client.post(
                f"{first_path}/authorization",
                json={
                    "authorization_id": "auth-first-http-owner",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=30)
                    ).isoformat(),
                    "operator_confirmed": True,
                },
            )
            assert authorized.status_code == 200, authorized.text
            started = client.post(f"{first_path}/start")
            assert started.status_code == 200, started.text
            assert started.json()["state"] == "RUNNING"
            assert plants[0].candidate_measurement_started.wait(timeout=5)

            conflict = client.post(
                "/api/v1.1/tasks",
                json=_ai_task_request("second-http-owner"),
            )

            assert conflict.status_code == 409, conflict.text
            detail = conflict.json()["detail"]
            assert detail["code"] == "ACTIVE_HARDWARE_TASK"
            assert first_task_id in detail["active_task_id"]
            missing = client.get("/api/v1.1/tasks/second-http-owner")
            assert missing.status_code == 404
            assert runtime.v11_service.store.get_task("second-http-owner") is None
            assert runtime.v11_service.store.active_task_id() == first_task_id

            pause_response: dict[str, object] = {}

            def pause() -> None:
                pause_response["response"] = client.post(f"{first_path}/pause")

            pause_thread = Thread(target=pause)
            pause_thread.start()
            assert stop_observed.set_called.wait(timeout=5)
            assert pause_thread.is_alive()
            plants[0].allow_candidate_measurement.set()
            pause_thread.join(timeout=5)
            assert not pause_thread.is_alive()
            paused = pause_response["response"]
            assert paused.status_code == 200, paused.text
            assert paused.json()["state"] == "PAUSED"

            resumed = client.post(f"{first_path}/resume")
            assert resumed.status_code == 200, resumed.text
            assert resumed.json()["state"] == "RUNNING"
            assert runner.wait_for_idle(timeout=10)
            terminal = client.get(first_path)
            assert terminal.status_code == 200
            assert terminal.json()["state"] in {
                "COMPLETED",
                "COMPLETED_NO_IMPROVEMENT",
            }
        finally:
            for plant in plants:
                plant.allow_candidate_measurement.set()
