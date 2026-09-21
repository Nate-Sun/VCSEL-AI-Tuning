"""Narrow continuous execution port backed by the existing orchestrator."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from vcsel_ai_tuning.campaign.models import Candidate, StageAuthorization
from vcsel_ai_tuning.eye_image import derive_kimi_input_png
from vcsel_ai_tuning.v2.contracts import EyeMetrics, EyeSample, MeasurementStatus

from .artifacts import EyeArtifactBundle
from .models import ExecutionOutcome


class ContinuousExecution(Protocol):
    """Narrow execution contract with distinct success and failure recovery."""

    def execute_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
    ) -> ExecutionOutcome | tuple[object, ...]: ...

    def measure_current_state(self) -> ExecutionOutcome | object: ...

    def restore_b0(self, expected: Mapping[str, int] | None = None) -> object: ...

    def recover_b0_after_failure(
        self,
        expected: Mapping[str, int] | None = None,
    ) -> object: ...


class PersistedObservationAdapter:
    """Project one indexed campaign observation into strict continuous evidence."""

    __slots__ = ("_repository",)

    def __init__(self, repository: Any) -> None:
        if not callable(getattr(repository, "get_observation", None)) or not callable(
            getattr(repository, "get_artifact", None)
        ):
            raise TypeError("repository lacks indexed observation access")
        self._repository = repository

    def to_eye_sample(self, observation: Any) -> EyeSample:
        published = self._repository.get_observation(observation.observation_id)
        if published is None:
            raise RuntimeError("continuous observation is not durably indexed")
        evidence = dict(observation.evidence)
        required = {
            "provenance",
            "dca_fingerprint",
            "png_path",
            "npy_path",
            "png_sha256",
            "npy_sha256",
            "kimi_png_path",
            "kimi_png_sha256",
            "kimi_source_png_sha256",
            "kimi_resize_width",
            "kimi_resize_height",
            "kimi_resize_applied",
            "kimi_max_width",
            "kimi_max_height",
            "kimi_resample",
            "kimi_colors",
            "kimi_quantization",
            "kimi_dither",
            "kimi_output_mode",
            "kimi_provenance",
        }
        if set(evidence) < required:
            raise RuntimeError("continuous observation evidence is incomplete")
        png_path = str(evidence["png_path"])
        npy_path = str(evidence["npy_path"])
        kimi_png_path = str(evidence["kimi_png_path"])
        png_record = self._repository.get_artifact(png_path)
        npy_record = self._repository.get_artifact(npy_path)
        kimi_png_record = self._repository.get_artifact(kimi_png_path)
        if png_record is None or npy_record is None or kimi_png_record is None:
            raise RuntimeError("continuous eye artifacts are not indexed")
        if str(evidence["kimi_source_png_sha256"]) != str(evidence["png_sha256"]):
            raise RuntimeError("Kimi input source hash does not match DCA screen")
        if evidence["kimi_resample"] != "LANCZOS":
            raise RuntimeError("Kimi input resample method is invalid")
        if evidence["kimi_max_width"] != 1280:
            raise RuntimeError("Kimi input maximum width is invalid")
        if evidence["kimi_max_height"] != 528:
            raise RuntimeError("Kimi input maximum height is invalid")
        if evidence["kimi_colors"] != 256:
            raise RuntimeError("Kimi input color count is invalid")
        if evidence["kimi_quantization"] != "MEDIANCUT":
            raise RuntimeError("Kimi input quantization is invalid")
        if evidence["kimi_dither"] != "NONE":
            raise RuntimeError("Kimi input dither mode is invalid")
        if evidence["kimi_output_mode"] != "P":
            raise RuntimeError("Kimi input output mode is invalid")
        if evidence["kimi_provenance"] != "DCA_SCREEN_DERIVED":
            raise RuntimeError("Kimi input provenance is invalid")
        bundle = EyeArtifactBundle(
            png_ref={
                "relative_path": kimi_png_path,
                "sha256": str(evidence["kimi_png_sha256"]),
                "media_type": "image/png",
            },
            npy_ref={
                "relative_path": npy_path,
                "sha256": str(evidence["npy_sha256"]),
                "media_type": "application/x-npy",
            },
            source_png_ref={
                "relative_path": png_path,
                "sha256": str(evidence["png_sha256"]),
                "media_type": "image/png",
            },
            artifact_hashes={
                png_path: str(evidence["png_sha256"]),
                npy_path: str(evidence["npy_sha256"]),
                kimi_png_path: str(evidence["kimi_png_sha256"]),
            },
            dca_fingerprint=str(evidence["dca_fingerprint"]),
            provenance=str(evidence["provenance"]),
        )
        bundle.verify(Path(self._repository.run_dir))
        source_path = Path(self._repository.run_dir) / png_path
        kimi_path = Path(self._repository.run_dir) / kimi_png_path
        npy_path_on_disk = Path(self._repository.run_dir) / npy_path
        try:
            with Image.open(BytesIO(source_path.read_bytes())) as source_image:
                source_pixels = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
                expected_png, expected_width, expected_height = derive_kimi_input_png(
                    source_image
                )
            derived_pixels = np.load(npy_path_on_disk, allow_pickle=False)
            if not np.array_equal(source_pixels, derived_pixels):
                raise RuntimeError("NPY is not derived from DCA screen")
            if (
                int(evidence["kimi_resize_width"]) != expected_width
                or int(evidence["kimi_resize_height"]) != expected_height
                or evidence["kimi_resize_applied"] is not (
                    source_pixels.shape[1] > 1280 or source_pixels.shape[0] > 528
                )
                or kimi_path.read_bytes() != expected_png
            ):
                raise RuntimeError("Kimi input is not derived from DCA screen")
        except RuntimeError:
            raise
        except (OSError, TypeError, ValueError, UnidentifiedImageError) as error:
            raise RuntimeError("Kimi input is not derived from DCA screen") from error
        settings = observation.settings
        payload = published.payload
        metric_status = (
            payload.get("metric_status", {}) if isinstance(payload, Mapping) else {}
        )
        if not isinstance(metric_status, Mapping):
            metric_status = {}
        return EyeSample(
            measurement_id=observation.observation_id,
            campaign_id=observation.campaign_id,
            sweep_point_id=observation.candidate_id or observation.observation_id,
            raw_eye_density_ref=bundle.png_ref,
            waveform_ref=bundle.npy_ref,
            metrics=EyeMetrics(
                tdecq_db=observation.metrics.get("tdecq_db"),
                outer_er_chan1a_db=observation.metrics.get("er_db"),
            ),
            requested_params=dict(settings.requested_i2c),
            applied_params=dict(settings.applied_i2c),
            readback_params=dict(settings.readback_i2c),
            dca_config_fingerprint=bundle.dca_fingerprint,
            measurement_status=(
                MeasurementStatus.VALID
                if observation.valid and observation.metrics.get("tdecq_db") is not None
                else MeasurementStatus.INVALID
            ),
            er_status=metric_status.get("er_status"),
            tdecq_status=metric_status.get("tdecq_status"),
            tdecq_invalid_reason=metric_status.get("tdecq_invalid_reason"),
            captured_at=observation.measured_at,
            artifact_hashes=dict(bundle.artifact_hashes),
        )


class OrchestratorExecutionPort:
    __slots__ = ("_adapter_identities", "_observation_adapter", "_orchestrator")

    def __init__(
        self,
        orchestrator: Any,
        *,
        observation_adapter: Any | None = None,
        adapter_identities: Mapping[str, object] | None = None,
    ) -> None:
        required = (
            "run_continuous_i2c_candidate",
            "measure_current_i2c_state",
            "restore_static_i2c_b0",
            "recover_continuous_commission_failure",
        )
        if any(not callable(getattr(orchestrator, name, None)) for name in required):
            raise TypeError("orchestrator lacks continuous execution operations")
        if observation_adapter is not None and not callable(
            getattr(observation_adapter, "to_eye_sample", None)
        ):
            raise TypeError("observation_adapter must implement to_eye_sample")
        self._orchestrator = orchestrator
        self._observation_adapter = observation_adapter
        self._adapter_identities = dict(adapter_identities or {})

    def execute_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
    ) -> ExecutionOutcome | tuple[object, ...]:
        observations = tuple(
            self._orchestrator.run_continuous_i2c_candidate(
                authorization,
                candidate,
            )
        )
        if self._observation_adapter is None:
            return observations
        if len(observations) != 1:
            raise RuntimeError("continuous execution requires exactly one observation")
        return self._outcome(observations[0])

    def measure_current_state(self) -> ExecutionOutcome | object:
        observation = self._orchestrator.measure_current_i2c_state()
        if self._observation_adapter is None:
            return observation
        return self._outcome(observation)

    def restore_b0(self, expected: Mapping[str, int] | None = None) -> object:
        restored = self._orchestrator.restore_static_i2c_b0()
        return self._verified_readback(restored, expected)

    def recover_b0_after_failure(
        self,
        expected: Mapping[str, int] | None = None,
    ) -> object:
        restored = self._orchestrator.recover_continuous_commission_failure()
        return self._verified_readback(restored, expected)

    @staticmethod
    def _verified_readback(
        restored: object,
        expected: Mapping[str, int] | None,
    ) -> dict[str, int]:
        readback = getattr(restored, "readback_i2c", restored)
        result = dict(readback)
        if expected is not None and result != dict(expected):
            raise RuntimeError("static B0 restore readback mismatch")
        return result

    def _outcome(self, observation: Any) -> ExecutionOutcome:
        settings = observation.settings
        return ExecutionOutcome(
            sample=self._observation_adapter.to_eye_sample(observation),
            requested_params=dict(settings.requested_i2c),
            applied_params=dict(settings.applied_i2c),
            readback_params=dict(settings.readback_i2c),
            adapter_identities=self._adapter_identities,
            evidence_complete=True,
        )
