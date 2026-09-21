from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    CampaignState,
    Observation,
)
from vcsel_ai_tuning.v2.continuous.execution import OrchestratorExecutionPort
from vcsel_ai_tuning.v2.continuous.execution import PersistedObservationAdapter
from vcsel_ai_tuning.v2.continuous.models import ExecutionOutcome


class RecordingOrchestrator:
    def __init__(self, observations=()) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.observations = list(observations)

    def run_continuous_i2c_candidate(self, authorization, candidate):
        self.calls.append(("run", authorization, candidate))
        if self.observations:
            return (self.observations.pop(0),)
        return (SimpleNamespace(observation_id="after-1"),)

    def measure_current_i2c_state(self):
        self.calls.append(("measure",))
        if self.observations:
            return self.observations.pop(0)
        return SimpleNamespace(observation_id="acceptance-1")

    def restore_static_i2c_b0(self):
        self.calls.append(("restore",))
        return SimpleNamespace(readback_i2c={"eq": 22})

    def recover_continuous_commission_failure(self):
        self.calls.append(("failure_recovery",))
        return SimpleNamespace(readback_i2c={"eq": 22})


def test_execution_port_exposes_only_narrow_orchestrator_operations() -> None:
    orchestrator = RecordingOrchestrator()
    port = OrchestratorExecutionPort(orchestrator)

    observations = port.execute_candidate("auth", "candidate")
    acceptance = port.measure_current_state()
    restored = port.restore_b0({"eq": 22})
    recovered = port.recover_b0_after_failure({"eq": 22})

    assert observations[0].observation_id == "after-1"
    assert acceptance.observation_id == "acceptance-1"
    assert restored == {"eq": 22}
    assert recovered == {"eq": 22}
    assert orchestrator.calls == [
        ("run", "auth", "candidate"),
        ("measure",),
        ("restore",),
        ("failure_recovery",),
    ]
    assert not hasattr(port, "i2c")
    assert not hasattr(port, "dca")
    assert not hasattr(port, "bert")


class FakeObservationAdapter:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def to_eye_sample(self, observation):
        from vcsel_ai_tuning.v2.continuous.fake import fake_eye

        return fake_eye(
            self.artifact_root,
            observation.observation_id,
            readback=dict(observation.settings.readback_i2c),
            tdecq_db=observation.metrics["tdecq_db"],
            outer_er_db=observation.metrics["er_db"],
        )


def _observation(observation_id: str, *, requested=None) -> Observation:
    requested = requested or {}
    readback = {
        "ibias_dac": 70,
        "gain_ctrl": 5,
        "eq": 23,
        "efin_cur": 15,
        "vga_cur": 13,
        "res_gain": 6,
        "res_bw": 0,
        "vbias_ref": 15,
        "vref": 15,
    }
    return Observation(
        observation_id=observation_id,
        campaign_id="campaign-1",
        candidate_id="candidate-1" if requested else None,
        stage=CampaignState.I2C_FINE,
        settings=AppliedSettings(
            requested_i2c=requested,
            quantized_i2c=requested,
            applied_i2c=requested,
            readback_i2c=readback,
        ),
        metrics={"tdecq_db": 3.0, "er_db": 4.5},
        valid=True,
        feasible=True,
        elapsed_s=0.1,
        measured_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )


def test_execution_port_converts_persisted_observation_to_strict_outcome(
    tmp_path,
) -> None:
    observation = _observation("after-1", requested={"eq": 23})
    orchestrator = RecordingOrchestrator([observation])
    port = OrchestratorExecutionPort(
        orchestrator,
        observation_adapter=FakeObservationAdapter(tmp_path),
        adapter_identities={"i2c": "fake-i2c", "dca": "fake-dca"},
    )
    candidate = SimpleNamespace(requested_i2c={"eq": 23})

    outcome = port.execute_candidate("auth", candidate)

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.sample.measurement_id == "after-1"
    assert outcome.requested_params == {"eq": 23}
    assert outcome.applied_params == {"eq": 23}
    assert outcome.readback_params["eq"] == 23
    assert outcome.adapter_identities == {
        "i2c": "fake-i2c",
        "dca": "fake-dca",
    }


def test_execution_port_measurement_is_write_free_outcome(tmp_path) -> None:
    observation = _observation("acceptance-1")
    port = OrchestratorExecutionPort(
        RecordingOrchestrator([observation]),
        observation_adapter=FakeObservationAdapter(tmp_path),
    )

    outcome = port.measure_current_state()

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.requested_params == {}
    assert outcome.applied_params == {}


def test_persisted_observation_adapter_builds_eye_sample_from_indexed_evidence(
    tmp_path,
) -> None:
    from io import BytesIO
    from PIL import Image
    from vcsel_ai_tuning.campaign.repository import ArtifactSpec, CampaignRepository
    from vcsel_ai_tuning.eye_image import derive_kimi_input_png

    stream = BytesIO()
    Image.new("RGB", (4, 3), (1, 2, 3)).save(stream, format="PNG")
    png = stream.getvalue()
    npy_stream = BytesIO()

    np.save(
        npy_stream,
        np.full((3, 4, 3), (1, 2, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    npy = npy_stream.getvalue()
    png_hash = sha256(png).hexdigest()
    npy_hash = sha256(npy).hexdigest()
    with Image.open(BytesIO(png)) as image:
        kimi_png, _, _ = derive_kimi_input_png(image)
    kimi_hash = sha256(kimi_png).hexdigest()
    observation = _observation("after-1", requested={"eq": 23}).model_copy(
        update={
            "evidence": {
                "provenance": "SCREEN_IMAGE_DERIVED",
                "dca_fingerprint": "d" * 64,
                "png_path": "eye_artifacts/eye.png",
                "npy_path": "eye_artifacts/model_input_eye.npy",
                "png_sha256": png_hash,
                "npy_sha256": npy_hash,
                "kimi_png_path": "eye_artifacts/kimi_input_eye.png",
                "kimi_png_sha256": kimi_hash,
                "kimi_source_png_sha256": png_hash,
                "kimi_resize_width": 4,
                "kimi_resize_height": 3,
                "kimi_resize_applied": False,
                "kimi_max_width": 1280,
                "kimi_max_height": 528,
                "kimi_resample": "LANCZOS",
                "kimi_colors": 256,
                "kimi_quantization": "MEDIANCUT",
                "kimi_dither": "NONE",
                "kimi_output_mode": "P",
                "kimi_provenance": "DCA_SCREEN_DERIVED",
            }
        }
    )
    with CampaignRepository(tmp_path, "campaign-1") as repository:
        payload = observation.model_dump(mode="json")
        payload["metric_status"] = {
            "er_status": "CORR",
            "tdecq_status": "INV",
            "tdecq_invalid_reason": "The target SER is lower than actual SER.",
        }
        repository.publish_observation(
            observation.observation_id,
            payload,
            [
                ArtifactSpec("eye_artifacts", "eye.png", png, "image/png"),
                ArtifactSpec(
                    "eye_artifacts",
                    "model_input_eye.npy",
                    npy,
                    "application/x-npy",
                ),
                ArtifactSpec(
                    "eye_artifacts",
                    "kimi_input_eye.png",
                    kimi_png,
                    "image/png",
                ),
            ],
        )
        adapter = PersistedObservationAdapter(repository)

        sample = adapter.to_eye_sample(observation)

        assert sample.measurement_id == observation.observation_id
        assert sample.raw_eye_density_ref.media_type == "image/png"
        assert sample.raw_eye_density_ref.sha256 == kimi_hash
        assert sample.raw_eye_density_ref.relative_path == "eye_artifacts/kimi_input_eye.png"
        assert sample.waveform_ref.sha256 == npy_hash
        assert sample.artifact_hashes["eye_artifacts/eye.png"] == png_hash
        assert sample.dca_config_fingerprint == "d" * 64
        assert sample.metrics.tdecq_db == 3.0
        assert sample.er_status == "CORR"
        assert sample.tdecq_status == "INV"
        assert sample.tdecq_invalid_reason == "The target SER is lower than actual SER."


def test_persisted_observation_adapter_rejects_kimi_png_not_derived_from_source(
    tmp_path,
) -> None:
    from io import BytesIO
    from PIL import Image
    from vcsel_ai_tuning.campaign.repository import ArtifactSpec, CampaignRepository

    source_stream = BytesIO()
    Image.new("RGB", (8, 4), (1, 2, 3)).save(source_stream, format="PNG")
    source = source_stream.getvalue()
    kimi_stream = BytesIO()
    Image.new("RGB", (8, 4), (3, 2, 1)).save(kimi_stream, format="PNG")
    kimi = kimi_stream.getvalue()
    npy_stream = BytesIO()
    np.save(npy_stream, np.zeros((4, 8, 3), dtype=np.uint8), allow_pickle=False)
    npy = npy_stream.getvalue()
    source_hash = sha256(source).hexdigest()
    kimi_hash = sha256(kimi).hexdigest()
    npy_hash = sha256(npy).hexdigest()
    observation = _observation("after-invalid-kimi").model_copy(
        update={
            "evidence": {
                "provenance": "SCREEN_IMAGE_DERIVED",
                "dca_fingerprint": "d" * 64,
                "png_path": "eye_artifacts/source.png",
                "npy_path": "eye_artifacts/source.npy",
                "png_sha256": source_hash,
                "npy_sha256": npy_hash,
                "kimi_png_path": "eye_artifacts/kimi.png",
                "kimi_png_sha256": kimi_hash,
                "kimi_source_png_sha256": source_hash,
                "kimi_resize_width": 8,
                "kimi_resize_height": 4,
                "kimi_resize_applied": False,
                "kimi_max_width": 1280,
                "kimi_max_height": 528,
                "kimi_resample": "LANCZOS",
                "kimi_colors": 256,
                "kimi_quantization": "MEDIANCUT",
                "kimi_dither": "NONE",
                "kimi_output_mode": "P",
                "kimi_provenance": "DCA_SCREEN_DERIVED",
            }
        }
    )
    with CampaignRepository(tmp_path, "campaign-1") as repository:
        repository.publish_observation(
            observation.observation_id,
            observation.model_dump(mode="json"),
            [
                ArtifactSpec("eye_artifacts", "source.png", source, "image/png"),
                ArtifactSpec("eye_artifacts", "source.npy", npy, "application/x-npy"),
                ArtifactSpec("eye_artifacts", "kimi.png", kimi, "image/png"),
            ],
        )

        with pytest.raises(RuntimeError, match="derived from DCA screen"):
            PersistedObservationAdapter(repository).to_eye_sample(observation)


def test_persisted_observation_adapter_rejects_npy_not_derived_from_source(
    tmp_path,
) -> None:
    from io import BytesIO
    from PIL import Image
    from vcsel_ai_tuning.campaign.repository import ArtifactSpec, CampaignRepository
    from vcsel_ai_tuning.eye_image import derive_kimi_input_png

    source_stream = BytesIO()
    Image.new("RGB", (8, 4), (1, 2, 3)).save(source_stream, format="PNG")
    source = source_stream.getvalue()
    with Image.open(BytesIO(source)) as image:
        kimi, width, height = derive_kimi_input_png(image)
    npy_stream = BytesIO()
    np.save(npy_stream, np.zeros((4, 8, 3), dtype=np.uint8), allow_pickle=False)
    npy = npy_stream.getvalue()
    source_hash = sha256(source).hexdigest()
    kimi_hash = sha256(kimi).hexdigest()
    npy_hash = sha256(npy).hexdigest()
    observation = _observation("after-invalid-npy").model_copy(
        update={
            "evidence": {
                "provenance": "SCREEN_IMAGE_DERIVED",
                "dca_fingerprint": "d" * 64,
                "png_path": "eye_artifacts/source.png",
                "npy_path": "eye_artifacts/source.npy",
                "png_sha256": source_hash,
                "npy_sha256": npy_hash,
                "kimi_png_path": "eye_artifacts/kimi.png",
                "kimi_png_sha256": kimi_hash,
                "kimi_source_png_sha256": source_hash,
                "kimi_resize_width": width,
                "kimi_resize_height": height,
                "kimi_resize_applied": False,
                "kimi_max_width": 1280,
                "kimi_max_height": 528,
                "kimi_resample": "LANCZOS",
                "kimi_colors": 256,
                "kimi_quantization": "MEDIANCUT",
                "kimi_dither": "NONE",
                "kimi_output_mode": "P",
                "kimi_provenance": "DCA_SCREEN_DERIVED",
            }
        }
    )
    with CampaignRepository(tmp_path, "campaign-1") as repository:
        repository.publish_observation(
            observation.observation_id,
            observation.model_dump(mode="json"),
            [
                ArtifactSpec("eye_artifacts", "source.png", source, "image/png"),
                ArtifactSpec("eye_artifacts", "source.npy", npy, "application/x-npy"),
                ArtifactSpec("eye_artifacts", "kimi.png", kimi, "image/png"),
            ],
        )

        with pytest.raises(RuntimeError, match="NPY is not derived from DCA screen"):
            PersistedObservationAdapter(repository).to_eye_sample(observation)
