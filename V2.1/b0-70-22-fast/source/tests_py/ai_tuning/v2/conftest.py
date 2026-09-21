from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    BrainCandidate,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
    PerceptionResult,
)


_SHA256 = "a" * 64


@pytest.fixture
def eye_sample() -> EyeSample:
    raw_eye_density_ref = ArtifactReference(
        relative_path="point-001/raw_eye_density.npy",
        sha256=_SHA256,
        media_type="application/x-npy",
    )
    return EyeSample(
        measurement_id="measurement-001",
        campaign_id="campaign-001",
        sweep_point_id="point-001",
        raw_eye_density_ref=raw_eye_density_ref,
        waveform_ref=None,
        metrics=EyeMetrics(
            tdecq_db=1.25,
            outer_er_chan1a_db=4.1,
            rlm=0.98,
            outer_oma=0.72,
            other_metrics={"eye_height": 0.31},
        ),
        requested_params={"bias_ma": 6.0},
        applied_params={"bias_ma": 6.0},
        readback_params={"bias_ma": 6.01},
        dca_config_fingerprint="b" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc),
        artifact_hashes={raw_eye_density_ref.relative_path: raw_eye_density_ref.sha256},
    )


@pytest.fixture
def brain_candidates() -> tuple[BrainCandidate, ...]:
    return (
        BrainCandidate(
            candidate_id="candidate-001",
            requested_params={"bias_ma": 6.0, "equalizer": {"tap_1": 1.0}},
        ),
        BrainCandidate(
            candidate_id="candidate-002",
            requested_params={"bias_ma": 6.5},
        ),
    )


@pytest.fixture
def perception_result() -> PerceptionResult:
    return PerceptionResult(
        measurement_id="measurement-001",
        quality_score=0.92,
        confidence=0.88,
        valid=True,
        ood=False,
        defects={"closure": 0.04, "ringing": 0.12},
        feature_vector=(0.2, 0.4),
        provider_id="vision-provider",
        provider_version="1.0.0",
        input_hash="a" * 64,
    )
