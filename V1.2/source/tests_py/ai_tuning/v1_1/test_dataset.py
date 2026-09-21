from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.v1_1 import dataset as dataset_module
from vcsel_ai_tuning.v1_1.dataset import (
    DatasetStatus,
    FrozenDatasetCapture,
    MeasurementArtifactContext,
    MeasurementArtifactBundleWriter,
    MeasurementStatus,
    NoOpDatasetRecorder,
    decode_png_dimensions,
    encode_density_png,
)
from vcsel_ai_tuning.v1_1.live_dataset_capture import FlexDcaDatasetCapture
from tests_py.ai_tuning.v1_1.test_live_dataset_capture import (
    FakeBinarySession,
    acquisition_guard,
)


NOW = datetime(2026, 8, 4, 8, 30, tzinfo=timezone.utc)


def test_dataset_capture_error_is_runtime_compatible_and_detaches_evidence():
    error_type = getattr(dataset_module, "DatasetCaptureError", None)
    assert error_type is not None
    nested = {"verified": False}
    source = MappingProxyType(
        {"source_restore": MappingProxyType(nested), "commands": ("query",)}
    )

    error = error_type("capture restore failed", raw_responses=source)
    nested["verified"] = True

    assert isinstance(error, RuntimeError)
    assert error.raw_responses == {
        "source_restore": {"verified": False},
        "commands": ["query"],
    }
    leaked = error.raw_responses
    leaked["source_restore"]["verified"] = True
    assert error.raw_responses["source_restore"]["verified"] is False
    json.dumps(error.raw_responses, allow_nan=False)


def test_writer_ignores_post_construction_capture_error_evidence_pollution(tmp_path):
    class PollutingCaptureError(dataset_module.DatasetCaptureError):
        def __str__(self):
            exposed = self.raw_responses
            exposed["nan"] = float("nan")
            exposed["cycle"] = exposed
            exposed["unsupported"] = object()
            return super().__str__()

    error = PollutingCaptureError(
        "capture restore failed",
        raw_responses={
            "source_restore": {"verified": False},
            "commands": ("query",),
        },
    )

    outcome = MeasurementArtifactBundleWriter(tmp_path).record(
        measurement_context(),
        lambda: (_ for _ in ()).throw(error),
    )

    assert outcome.status is DatasetStatus.FAILED
    assert outcome.bundle_path == "campaign-dataset/point-00000000"
    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    assert sorted(path.name for path in bundle.iterdir()) == ["record.json"]
    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert record["raw_responses"] == {
        "source_restore": {"verified": False},
        "commands": ["query"],
    }
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "replacement",
    (b'{"forged":true}', b'{not-valid-json'),
    ids=("forged-valid-json", "invalid-json"),
)
def test_writer_snapshots_capture_evidence_before_error_string_side_effects(
    tmp_path,
    replacement,
):
    class ReplacingCaptureError(dataset_module.DatasetCaptureError):
        def __str__(self):
            self._raw_responses_json = replacement
            return super().__str__()

    error = ReplacingCaptureError(
        "capture restore failed",
        raw_responses={"source_restore": {"verified": False}},
    )

    outcome = MeasurementArtifactBundleWriter(tmp_path).record(
        measurement_context(),
        lambda: (_ for _ in ()).throw(error),
    )

    assert outcome.status is DatasetStatus.FAILED
    assert outcome.bundle_path == "campaign-dataset/point-00000000"
    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    assert sorted(path.name for path in bundle.iterdir()) == ["record.json"]
    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert record["raw_responses"] == {
        "source_restore": {"verified": False}
    }


def measurement_context():
    return MeasurementArtifactContext(
        campaign_id="campaign-dataset",
        sweep_point_id="point-00000000",
        measurement_id="measurement-0001",
        acquisition_id="acquisition-0001",
        captured_at_utc=NOW,
        requested_params={"eq": 21},
        applied_params={"eq": 21},
        readback_params={"eq": 21},
        tdecq=2.41,
        outer_er=5.72,
        outer_er_source="CHAN1A",
        dca_fingerprint={"rate": "53.125 GBd"},
        dca_identity="Keysight,N1010A",
        bert_port="port1",
        bert_channel="0",
        bert_pattern="SSPRQ",
        measurement_status=MeasurementStatus.SUCCESS,
        dca_acquisition_duration_s=1.25,
    )


def test_dataset_status_contract_is_independent_from_measurement_status():
    assert [item.value for item in MeasurementStatus] == [
        "SUCCESS",
        "FAILED",
        "REJECTED",
    ]
    assert [item.value for item in DatasetStatus] == [
        "DISABLED",
        "COMPLETE",
        "PARTIAL",
        "FAILED",
    ]


def test_noop_recorder_does_not_evaluate_capture_or_create_files(tmp_path):
    called = False

    def forbidden_capture():
        nonlocal called
        called = True
        raise AssertionError("disabled recorder accessed dataset capture")

    outcome = NoOpDatasetRecorder().record(
        measurement_context(),
        forbidden_capture,
    )

    assert outcome.status is DatasetStatus.DISABLED
    assert outcome.bundle_path is None
    assert outcome.error is None
    assert outcome.missing_items == ()
    assert called is False
    assert list(tmp_path.iterdir()) == []


def test_context_rejects_naive_time_and_nonboolean_shape_inputs():
    context = measurement_context()
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(
            context,
            captured_at_utc=datetime(2026, 8, 4, 8, 30),
        )


def test_frozen_capture_rejects_object_or_non_2d_density():
    base = {
        "acquisition_id": "acquisition-0001",
        "density_axis_order": ("time", "amplitude"),
    }
    with pytest.raises(ValueError, match="two-dimensional"):
        FrozenDatasetCapture(
            **base,
            raw_eye_density=np.array([1, 2, 3], dtype=np.uint16),
        )
    with pytest.raises(TypeError, match="object dtype"):
        FrozenDatasetCapture(
            **base,
            raw_eye_density=np.array([[object()]], dtype=object),
        )


def complete_capture(*, acquisition_id="acquisition-0001"):
    return FrozenDatasetCapture(
        acquisition_id=acquisition_id,
        same_acquisition_verified=True,
        raw_eye_density=np.array(
            [[0, 1, 3], [2, 5, 9]],
            dtype=np.uint16,
        ),
        density_axis_order=("amplitude", "time"),
        density_source="CHAN1A",
        tdecq_equalized_eye_png=encode_density_png(
            np.array([[9, 1], [4, 7]], dtype=np.uint16)
        ),
        rlm=0.97,
        oma=913.4,
        average_power=0.51,
        equalizer_taps=(0.0, 0.1, 1.0, -0.1),
        equalizer_main_index=2,
        equalizer_source="FUNC1",
        raw_responses={"results": "fixed replay"},
    )


def test_unverified_acquisition_publishes_files_only_as_partial(tmp_path):
    capture = replace(complete_capture(), same_acquisition_verified=False)

    outcome = MeasurementArtifactBundleWriter(tmp_path).record(
        measurement_context(),
        lambda: capture,
    )

    assert outcome.status is DatasetStatus.PARTIAL
    assert "same_acquisition_proof" in outcome.missing_items
    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    assert (bundle / "raw_eye_density.npy").is_file()
    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert record["same_acquisition_verified"] is False
    assert record["dataset_status"] == "PARTIAL"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_writer_publishes_complete_same_acquisition_bundle(tmp_path):
    density = complete_capture().raw_eye_density
    assert density is not None
    writer = MeasurementArtifactBundleWriter(tmp_path)

    outcome = writer.record(measurement_context(), complete_capture)

    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    assert outcome.status is DatasetStatus.COMPLETE
    assert outcome.bundle_path == "campaign-dataset/point-00000000"
    assert sorted(path.name for path in bundle.iterdir()) == [
        "raw_eye.png",
        "raw_eye_density.npy",
        "record.json",
        "tdecq_equalized_eye.png",
    ]
    loaded = np.load(bundle / "raw_eye_density.npy", allow_pickle=False)
    np.testing.assert_array_equal(loaded, density)
    assert loaded.shape == density.shape
    assert loaded.dtype == density.dtype
    assert decode_png_dimensions(bundle / "raw_eye.png") == (3, 2)
    assert decode_png_dimensions(bundle / "tdecq_equalized_eye.png") == (2, 2)
    assert (bundle / "raw_eye.png").read_bytes() != (
        bundle / "tdecq_equalized_eye.png"
    ).read_bytes()

    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert record["measurement_id"] == "measurement-0001"
    assert record["acquisition_id"] == "acquisition-0001"
    assert record["measurement_status"] == "SUCCESS"
    assert record["dataset_status"] == "COMPLETE"
    assert record["density"] == {
        "shape": [2, 3],
        "dtype": "uint16",
        "axis_order": ["amplitude", "time"],
        "source": "CHAN1A",
    }
    assert record["metrics"] == {
        "tdecq": 2.41,
        "outer_er": 5.72,
        "rlm": 0.97,
        "oma": 913.4,
        "average_power": 0.51,
    }
    assert record["equalizer"]["main_index"] == 2
    assert record["equalizer"]["source"] == "FUNC1"
    for name in (
        "raw_eye_density.npy",
        "raw_eye.png",
        "tdecq_equalized_eye.png",
    ):
        evidence = record["files"][name]
        assert evidence["relative_path"] == name
        assert evidence["sha256"] == _sha256(bundle / name)
        assert evidence["size"] == (bundle / name).stat().st_size


def test_missing_equalized_eye_publishes_real_partial_evidence(tmp_path):
    capture = replace(complete_capture(), tdecq_equalized_eye_png=None)
    writer = MeasurementArtifactBundleWriter(tmp_path)

    outcome = writer.record(measurement_context(), lambda: capture)

    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    assert outcome.status is DatasetStatus.PARTIAL
    assert "tdecq_equalized_eye.png" in outcome.missing_items
    assert (bundle / "raw_eye_density.npy").is_file()
    assert (bundle / "raw_eye.png").is_file()
    assert not (bundle / "tdecq_equalized_eye.png").exists()
    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert record["dataset_status"] == "PARTIAL"


def test_acquisition_mismatch_never_publishes_capture_as_complete(tmp_path):
    writer = MeasurementArtifactBundleWriter(tmp_path)

    outcome = writer.record(
        measurement_context(),
        lambda: complete_capture(acquisition_id="another-acquisition"),
    )

    assert outcome.status is DatasetStatus.FAILED
    assert "acquisition_id" in outcome.missing_items
    record = json.loads(
        (
            tmp_path
            / "campaign-dataset"
            / "point-00000000"
            / "record.json"
        ).read_text(encoding="utf-8")
    )
    assert record["dataset_status"] == "FAILED"
    assert record["files"] == {}
    assert record["metrics"]["rlm"] is None
    assert record["density"] is None
    assert record["equalizer"]["taps"] is None


def test_writer_never_overwrites_completed_point_directory(tmp_path):
    writer = MeasurementArtifactBundleWriter(tmp_path)
    first = writer.record(measurement_context(), complete_capture)
    original = (
        tmp_path
        / "campaign-dataset"
        / "point-00000000"
        / "record.json"
    ).read_bytes()

    second = writer.record(measurement_context(), complete_capture)

    assert first.status is DatasetStatus.COMPLETE
    assert second.status is DatasetStatus.FAILED
    assert "already exists" in (second.error or "")
    assert (
        tmp_path
        / "campaign-dataset"
        / "point-00000000"
        / "record.json"
    ).read_bytes() == original


def test_writer_retries_transient_permission_error_while_publishing(
    tmp_path,
    monkeypatch,
):
    original_rename = Path.rename
    attempts = 0
    sleeps = []

    def flaky_rename(source, target):
        nonlocal attempts
        if source.name.startswith(".point-00000000.tmp-"):
            attempts += 1
            if attempts == 1:
                raise PermissionError(13, "transient scanner lock", str(source))
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    writer = MeasurementArtifactBundleWriter(tmp_path, sleep=sleeps.append)

    outcome = writer.record(measurement_context(), complete_capture)

    assert outcome.status is DatasetStatus.COMPLETE
    assert attempts == 2
    assert sleeps == [0.1]
    assert (
        tmp_path
        / "campaign-dataset"
        / "point-00000000"
        / "record.json"
    ).is_file()


def test_npy_failure_preserves_other_real_files_as_partial(tmp_path, monkeypatch):
    writer = MeasurementArtifactBundleWriter(tmp_path)

    def fail_npy(*_args, **_kwargs):
        raise OSError("injected NPY write failure")

    monkeypatch.setattr(np, "save", fail_npy)

    outcome = writer.record(measurement_context(), complete_capture)

    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    assert outcome.status is DatasetStatus.PARTIAL
    assert "raw_eye_density.npy" in outcome.missing_items
    assert not (bundle / "raw_eye_density.npy").exists()
    assert (bundle / "raw_eye.png").is_file()
    assert (bundle / "tdecq_equalized_eye.png").is_file()


def test_capture_failure_is_dataset_failed_without_raising(tmp_path):
    writer = MeasurementArtifactBundleWriter(tmp_path)

    def fail_capture():
        raise RuntimeError("replay capture unavailable")

    outcome = writer.record(measurement_context(), fail_capture)

    assert outcome.status is DatasetStatus.FAILED
    assert "replay capture unavailable" in (outcome.error or "")
    record = json.loads(
        (
            tmp_path
            / "campaign-dataset"
            / "point-00000000"
            / "record.json"
        ).read_text(encoding="utf-8")
    )
    assert record["measurement_status"] == "SUCCESS"
    assert record["dataset_status"] == "FAILED"
    assert record["raw_responses"] is None


def _record_live_capture_failure(tmp_path, session):
    adapter = FlexDcaDatasetCapture(
        session=session,
        raw_source="CHAN1A",
        equalized_source="FUNC1",
        equalizer_function=1,
        exclusive_access=lambda: True,
    )
    outcome = MeasurementArtifactBundleWriter(tmp_path).record(
        measurement_context(),
        lambda: adapter.capture(
            "acquisition-0001",
            SimpleNamespace(
                raw_responses={"acquisition_guard": acquisition_guard()}
            ),
        ),
    )
    bundle = tmp_path / "campaign-dataset" / "point-00000000"
    record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    return outcome, bundle, record


def _assert_evidence_only_failed_bundle(outcome, bundle, record):
    assert outcome.status is DatasetStatus.FAILED
    assert sorted(path.name for path in bundle.iterdir()) == ["record.json"]
    assert record["dataset_status"] == "FAILED"
    assert record["files"] == {}
    assert record["raw_responses"] is not None
    json.dumps(record["raw_responses"], allow_nan=False)


def test_optional_metric_restore_failure_persists_capture_evidence_only(tmp_path):
    session = FakeBinarySession(
        refused_measurement_writes={(":MEASure:EYE:OOMA", "FUNC1")}
    )

    outcome, bundle, record = _record_live_capture_failure(tmp_path, session)

    _assert_evidence_only_failed_bundle(outcome, bundle, record)
    evidence = record["raw_responses"][":MEASure:EYE:OOMA"]
    assert evidence["source_restore"] == {
        "expected": "FUNC1",
        "readback": "CHAN1A",
        "verified": False,
    }


def test_guard_restore_failure_persists_capture_evidence_only(tmp_path):
    session = FakeBinarySession(
        measurement_source="FUNC2",
        failed_measurement_writes={(":MEASure:EYE:TDEQ", "FUNC2")},
    )

    outcome, bundle, record = _record_live_capture_failure(tmp_path, session)

    _assert_evidence_only_failed_bundle(outcome, bundle, record)
    evidence = record["raw_responses"]["acquisition_guard"]
    assert evidence["source_restore"] == {
        "expected": "FUNC2",
        "readback": "FUNC1",
        "verified": False,
    }
    assert "source_restore" in evidence["errors"]


def test_waveform_restore_failure_persists_capture_evidence_only(tmp_path):
    session = FakeBinarySession(refuse_restore=True)

    outcome, bundle, record = _record_live_capture_failure(tmp_path, session)

    _assert_evidence_only_failed_bundle(outcome, bundle, record)
    assert record["raw_responses"]["source_restore"] == {
        "expected": "WMEM1",
        "original_source_raw": "WMEM1",
        "readback": "FUNC1",
        "verified": False,
    }


def test_equalized_eye_cannot_copy_raw_eye(tmp_path):
    capture = complete_capture()
    assert capture.raw_eye_density is not None
    copied_raw = encode_density_png(capture.raw_eye_density)
    capture = replace(capture, tdecq_equalized_eye_png=copied_raw)

    outcome = MeasurementArtifactBundleWriter(tmp_path).record(
        measurement_context(),
        lambda: capture,
    )

    assert outcome.status is DatasetStatus.PARTIAL
    assert "tdecq_equalized_eye.png" in outcome.missing_items
    assert "must differ" in (outcome.error or "")
