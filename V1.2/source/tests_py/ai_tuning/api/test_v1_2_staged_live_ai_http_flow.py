from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.live_dca_isolation import IsolatedLiveDcaClient
from vcsel_ai_tuning.hardware.scpi_journal import ScpiOperationJournal
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration
from vcsel_ai_tuning.runtime import create_runtime
from vcsel_ai_tuning.runtime_config import RuntimeConfig
from vcsel_ai_tuning.v1_1.baseline import (
    B0Snapshot,
    DcaFingerprint,
    HardwareIdentity,
)
from vcsel_ai_tuning.v1_1.dataset import FrozenDatasetCapture, encode_density_png
from vcsel_ai_tuning.v1_1.domain import LifecycleState, RuntimeMode
from vcsel_ai_tuning.v1_1.fake_runtime import FakeV11Gateway
from vcsel_ai_tuning.v1_1.live_ai_execution import build_live_ai_execution_factory
from vcsel_ai_tuning.v1_1.live_ai_policy import (
    LiveAIProfile,
    STAGED_BUDGETS,
    validate_live_ai_payload,
)
from vcsel_ai_tuning.v1_1.live_sweep import (
    LiveSweepDriverBundle,
    _parameter_mapping,
)
from vcsel_ai_tuning.v1_1.persistent_runner import PersistentV11TaskRunner


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PHASES = (
    ("PILOT_3",) * 3
    + ("I2C_COARSE",) * 6
    + ("I2C_FINE",) * 16
    + ("FIR_FIRST", "FIR_ITERATIVE")
)


def _profile():
    return load_live_hardware_config(
        ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
    )


def _b0() -> B0Snapshot:
    profile = _profile()
    values = {field.name: 0 for field in profile.characterization.parameters}
    values.update(ibias_dac=70, res_gain=6)
    raw = {
        field.address_for_channel(profile.characterization.default_channel): (
            values[field.name] << field.bit_start
        )
        for field in profile.characterization.parameters
    }
    identity = "Keysight Technologies,N1010A,software-test,A.08"
    return B0Snapshot.capture(
        b0_id="native-v12-spawned-fake-b0",
        version=1,
        captured_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        i2c_fields=values,
        i2c_raw_registers=raw,
        fir_taps=(0, -4, -8, 104, -8, -4, 0),
        upper_nlc=6,
        lower_nlc=6,
        bert_pattern="SSPRQ",
        i2c_channel=profile.characterization.default_channel,
        bert_port=profile.bert.port,
        bert_channel=profile.bert.channel,
        dca_tdecq_source="FUNC1",
        dca_tdecq_equalizer_function=1,
        dca_outer_er_source="CHAN1A",
        dca_fingerprint=DcaFingerprint.from_mappings(
            identity=identity,
            measurement_sources={"outer_er": "CHAN1A", "tdecq": "FUNC1"},
            configuration={"tdecq_equalizer_function": "FUNC1"},
        ),
        hardware_identities={
            "dca": HardwareIdentity("dca", profile.dca.resource, identity),
            "i2c": HardwareIdentity(
                "i2c",
                f"{profile.i2c.port}/{profile.i2c.baudrate}/"
                f"0x{profile.i2c.device_address:02X}",
                "chip_id=0x52;revision=0x00",
            ),
            "bert": HardwareIdentity(
                "bert",
                f"{profile.bert.ip}:{profile.bert.udp_port}/"
                f"{profile.bert.port}/{profile.bert.channel}",
                "mode=PAM4;patterns=SSPRQ,SSPRQ",
            ),
        },
    )


def spawned_fake_dca_worker(send, request) -> None:
    """Picklable fake instrument worker using the production IPC protocol."""

    context = dict(request.operation_context)
    ScpiOperationJournal(request.journal_path).append(
        {
            "schema_version": "1.0",
            "operation_id": f"fake-{uuid4().hex}",
            "event": "COMPLETED",
            "operation": request.action,
            "command": "SPAWNED_FAKE_ONLY",
            "worker_pid": os.getpid(),
            **context,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    send.send({"event": "ROUND_STARTED", "repeat_index": request.round_index})
    if request.action == "preflight":
        result = {
            "kind": "preflight",
            "value": {
                "identity_verified": True,
                "configuration_verified": True,
                "spawned_fake": True,
            },
        }
    else:
        acquisition_id = f"acquisition-{uuid4().hex}"
        item_id = str(context.get("item_id", ""))
        tdecq = 2.0 if item_id.startswith("fast-fir-") else 2.4
        density = np.arange(24, dtype=np.uint32).reshape(4, 6)
        density = density + np.uint32(request.round_index + 1)
        capture = FrozenDatasetCapture(
            acquisition_id=acquisition_id,
            same_acquisition_verified=True,
            raw_eye_density=density,
            density_axis_order=("amplitude", "time"),
            density_source="CHAN1A",
            raw_eye_png=encode_density_png(density),
            tdecq_equalized_eye_png=encode_density_png(
                np.square(density.astype(np.uint64) + 1)
            ),
            rlm=0.97,
            oma=0.91,
            average_power=0.50,
            equalizer_taps=(
                0.0,
                -0.03125,
                -0.0625,
                0.8125,
                -0.0625,
                -0.03125,
                0.0,
            ),
            equalizer_main_index=3,
            equalizer_source="FUNC1",
            raw_responses={"spawned_fake": True, "item_id": item_id},
        )
        result = {
            "kind": "measurement",
            "acquisition_valid": True,
            "er_db": 4.1,
            "er_status": "CORR",
            "tdecq_db": tdecq,
            "tdecq_status": "CORR",
            "tdecq_invalid_reason": None,
            "elapsed_s": 0.01,
            "raw_responses": {
                "spawned_fake": True,
                "tdecq": {
                    "equalizer_taps": (
                        "0,-0.03125,-0.0625,0.8125,-0.0625,-0.03125,0"
                    ),
                    "equalizer_precursors": "3",
                },
            },
            "artifacts": (capture,),
        }
    send.send({"event": "ROUND_COMPLETED", "repeat_index": request.round_index})
    send.send({"event": "CLEANUP_STARTED"})
    send.send({"event": "WORKER_COMPLETED", "result": result})


@dataclass(frozen=True)
class _FieldValue:
    name: str
    channel: int
    address: int
    raw_value: int
    value: int


@dataclass(frozen=True)
class _WriteValue:
    name: str
    channel: int
    address: int
    before_raw: int
    before_value: int
    requested_value: int
    after_raw: int
    after_value: int
    verified: bool = True


class _Writer:
    def __init__(self, b0: B0Snapshot) -> None:
        self.b0 = dict(b0.i2c_fields)
        self.raw = dict(b0.i2c_raw_registers)

    def read_field(self, field, *, channel):
        address = field.address_for_channel(channel)
        raw = self.raw[address]
        return _FieldValue(
            field.name,
            channel,
            address,
            raw,
            (raw >> field.bit_start) & field.encoded_max,
        )

    def write_field(
        self, field, *, channel, value, expected_current, expected_raw
    ):
        before = self.read_field(field, channel=channel)
        assert before.value == expected_current
        assert before.raw_value == expected_raw
        mask = field.encoded_max << field.bit_start
        after_raw = (before.raw_value & ~mask) | (value << field.bit_start)
        self.raw[before.address] = after_raw
        return _WriteValue(
            field.name,
            channel,
            before.address,
            before.raw_value,
            before.value,
            value,
            after_raw,
            value,
        )


class _Bert:
    def __init__(self, b0: B0Snapshot) -> None:
        self.b0 = b0
        self.taps = b0.fir_taps

    def get_mode(self):
        return "PAM4"

    def get_configuration(self, port, channel):
        return BertConfiguration(
            port=port,
            channel=str(channel),
            taps=self.taps,
            upper_nlc=self.b0.upper_nlc,
            lower_nlc=self.b0.lower_nlc,
            raw_response="spawned fake",
            pattern_fields=("SSPRQ", "SSPRQ"),
        )

    def set_fir(self, _port, _channel, *, taps, upper_nlc, lower_nlc):
        assert upper_nlc == self.b0.upper_nlc
        assert lower_nlc == self.b0.lower_nlc
        self.taps = tuple(taps)


class _StagedFakeGateway(FakeV11Gateway):
    def __init__(self, profile, b0) -> None:
        super().__init__()
        self._snapshot = b0
        self._mapping = _parameter_mapping(profile, b0)

    @property
    def runtime_mode(self) -> RuntimeMode:
        return RuntimeMode.LIVE

    def capabilities(self):
        return {
            "campaign_types": ["AI_OPTIMIZATION"],
            "dynamic_b0": True,
            "full_campaign_authorization": True,
            "hardware_connected": False,
            "hardware_mutation": True,
            "fake": False,
            "live_ai_profile": "PILOT_3_THEN_FAST24",
            **STAGED_BUDGETS,
            "measurement_timeout_s": 60.0,
            "cleanup_grace_s": 5.0,
            "dataset_required": True,
            "fir_authority": "FORMULA_ONLY_MAIN_NLC_PATTERN_FROZEN_L1_170",
            "final_restore": "B0_VERIFIED",
            "profile_options": [
                {
                    "name": "PILOT_3_THEN_FAST24",
                    "candidate_budget": 27,
                    "max_alternating_rounds": 2,
                }
            ],
        }

    def validate_ai_payload(self, payload):
        validate_live_ai_payload(LiveAIProfile.PILOT_3_THEN_FAST24, payload)


def _request(task_id: str, b0: B0Snapshot) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": "AI_OPTIMIZATION",
        "channel_mapping": {
            "i2c_channel": b0.i2c_channel,
            "bert_port": b0.bert_port,
            "bert_channel": b0.bert_channel,
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
        "live_ai_profile": "PILOT_3_THEN_FAST24",
        "optimization_scope": "I2C_FIR_ALTERNATING",
        **STAGED_BUDGETS,
        "settle_time_s": 0.1,
        "save_dataset": True,
        "seed": 17,
    }


def _dataset(row: dict[str, object]) -> dict[str, object]:
    raw = row["raw_response"]
    assert isinstance(raw, dict)
    dataset = raw if row["phase"] == "PILOT_3" else raw.get("dataset")
    assert isinstance(dataset, dict)
    return dataset


def test_native_v12_staged_live_http_flow_with_spawned_fake_dca(tmp_path):
    profile = _profile()
    b0 = _b0()
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "state" / "platform" / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts" / "platform",
        },
        b0_path=(ROOT / "configs" / "ai_tuning" / "b0.yaml"),
        worker_count=1,
    )
    database = tmp_path / "state" / "v1.2" / "campaigns.sqlite3"
    artifact_root = tmp_path / "artifacts" / "v1.2"
    writer = _Writer(b0)
    bert = _Bert(b0)

    @contextmanager
    def drivers(*_args, **_kwargs):
        dca = IsolatedLiveDcaClient(
            profile=profile,
            artifact_root=artifact_root,
            timeout_s=5.0,
            cleanup_grace_s=0.5,
            capture_dataset=True,
            worker_target=spawned_fake_dca_worker,
        )
        yield LiveSweepDriverBundle(
            dca=dca,
            i2c_writer=writer,
            i2c_identity=SimpleNamespace(matched=True),
            bert=bert,
        )

    ai_factory = build_live_ai_execution_factory(
        profile,
        database_path=database,
        artifact_root=artifact_root,
        driver_factory=drivers,
        measurement_timeout_s=5.0,
        cleanup_grace_s=0.5,
        live_ai_profile=LiveAIProfile.PILOT_3_THEN_FAST24,
    )
    runner = PersistentV11TaskRunner(
        database_path=database,
        artifact_root=artifact_root,
        ai_execution_factory=ai_factory,
    )
    runtime = create_runtime(
        config,
        v11_gateway=_StagedFakeGateway(profile, b0),
        v11_runner=runner,
    )
    app = create_app_from_runtime_factory(lambda: runtime)
    task_id = "native-v12-spawned-http"
    task_path = f"/api/v1.1/tasks/{task_id}"

    with TestClient(app) as client:
        created = client.post("/api/v1.1/tasks", json=_request(task_id, b0))
        assert created.status_code == 201, created.text
        snapshot = client.get("/api/v1.1/hardware/snapshot")
        frozen = client.post(
            f"{task_path}/b0",
            json={
                "snapshot_token": snapshot.json()["snapshot_token"],
                "operator_confirmed": True,
            },
        )
        assert frozen.status_code == 200, frozen.text
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
        started = client.post(f"{task_path}/start")
        assert started.status_code == 200, started.text
        assert runner.wait_for_idle(timeout=120)

        task = client.get(task_path)
        observations_response = client.get(f"{task_path}/observations")
        events_response = client.get(f"{task_path}/events")
        final_report = client.get(f"{task_path}/final-report")
        assert task.json()["state"] in {
            LifecycleState.COMPLETED.value,
            LifecycleState.COMPLETED_NO_IMPROVEMENT.value,
        }
        assert observations_response.status_code == 200
        observations = observations_response.json()
        candidate_rows = [
            row
            for row in observations
            if row["phase"] not in {"B0", "FINAL_VERIFICATION"}
        ]
        assert tuple(row["phase"] for row in candidate_rows) == EXPECTED_PHASES
        assert len(candidate_rows) == 27
        assert len(observations) == 29
        assert all(
            row["requested"] == row["applied"] == row["readback"]
            for row in observations
        )

        acquisition_ids = set()
        for row in observations:
            dataset = _dataset(row)
            assert dataset["dataset_status"] == "COMPLETE"
            assert dataset["measurement_id"]
            assert dataset["acquisition_id"] not in acquisition_ids
            acquisition_ids.add(dataset["acquisition_id"])
            bundle = artifact_root / str(dataset["dataset_bundle_path"])
            record_path = bundle / "record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            assert record["measurement_id"] == dataset["measurement_id"]
            assert record["acquisition_id"] == dataset["acquisition_id"]
            assert record["observation_id"] == row["observation_id"]
            assert record["requested_params"] == row["requested"]
            assert record["applied_params"] == row["applied"]
            assert record["readback_params"] == row["readback"]
            assert record["metrics"]["tdecq"] is not None
            assert record["metrics"]["outer_er"] == 4.1
            assert record["metrics"]["oma"] == 0.91
            density = np.load(bundle / "raw_eye_density.npy", allow_pickle=False)
            assert density.shape == (4, 6)
            for png_name in ("raw_eye.png", "tdecq_equalized_eye.png"):
                with Image.open(bundle / png_name) as image:
                    image.verify()
            for evidence in record["files"].values():
                content = (bundle / evidence["relative_path"]).read_bytes()
                assert hashlib.sha256(content).hexdigest() == evidence["sha256"]

        events = events_response.json()
        event_types = [event["event_type"] for event in events]
        assert event_types.index("PILOT_SAFETY_GATE_PASSED") < event_types.index(
            "AI_EXECUTION_TERMINAL"
        )
        assert final_report.status_code == 200
        assert final_report.json()["state"] == task.json()["state"]
        assert final_report.json()["observations"] == observations

    assert bert.taps == b0.fir_taps
    assert all(
        writer.read_field(
            field, channel=profile.characterization.default_channel
        ).value
        == dict(b0.i2c_fields)[field.name]
        for field in profile.characterization.parameters
    )
    journal = artifact_root / task_id / "dca_io_journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert entries
    assert all(entry["command"] == "SPAWNED_FAKE_ONLY" for entry in entries)
    assert all(entry["worker_pid"] != os.getpid() for entry in entries)
