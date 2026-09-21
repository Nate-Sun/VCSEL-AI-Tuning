from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from fractions import Fraction
import ast
import hashlib
import inspect
import io
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)
from vcsel_ai_tuning.v2.preprocessing import (
    EyeDensityError,
    EyeDensityPreprocessConfig,
    preprocess_eye_density,
)


def _npy_bytes(value: object, *, allow_pickle: bool = False) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=allow_pickle)
    return stream.getvalue()


def _npy_header_bytes(shape: tuple[int, int], *, descr: str = "<f8") -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {"descr": descr, "fortran_order": False, "shape": shape},
    )
    return stream.getvalue()


def _sample_for_payload(
    artifact_root: Path,
    payload: bytes,
    *,
    relative_path: str = "point-001/raw_eye_density.npy",
    media_type: str = "application/x-npy",
) -> EyeSample:
    path = artifact_root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    ref = ArtifactReference(
        relative_path=relative_path,
        sha256=digest,
        media_type=media_type,
    )
    return EyeSample(
        measurement_id="measurement-001",
        campaign_id="campaign-001",
        sweep_point_id="point-001",
        raw_eye_density_ref=ref,
        waveform_ref=None,
        metrics=EyeMetrics(),
        requested_params={},
        applied_params={},
        readback_params={},
        dca_config_fingerprint="b" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        artifact_hashes={ref.relative_path: ref.sha256},
    )


def _sample_for_array(
    artifact_root: Path,
    array: object,
    **kwargs: object,
) -> EyeSample:
    return _sample_for_payload(artifact_root, _npy_bytes(array, allow_pickle=True), **kwargs)


def test_preprocess_returns_deterministic_contiguous_rgb_canvas(tmp_path: Path) -> None:
    sample = _sample_for_array(tmp_path, np.arange(35, dtype=np.uint16).reshape(5, 7))

    first = preprocess_eye_density(sample, tmp_path)
    second = preprocess_eye_density(sample, tmp_path)

    assert first.shape == (280, 392, 3)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert np.array_equal(first[..., 0], first[..., 1])
    assert np.array_equal(first[..., 1], first[..., 2])
    assert np.array_equal(first, second)


def test_preprocess_preserves_top_and_left_orientation(tmp_path: Path) -> None:
    values = np.zeros((272, 392), dtype=np.float64)
    values[0, 0] = 4095.0
    sample = _sample_for_array(tmp_path, values)

    image = preprocess_eye_density(sample, tmp_path)[..., 0]

    assert image[4:140, :196].mean() > image[140:276, :196].mean()
    assert image[4:140, :196].mean() > image[4:140, 196:].mean()


def test_preprocess_preserves_aspect_ratio_and_centers_content(tmp_path: Path) -> None:
    sample = _sample_for_array(tmp_path, np.full((272, 392), 4095.0, dtype=np.float64))

    image = preprocess_eye_density(sample, tmp_path)[..., 0]

    assert np.all(image[:4] == 0.0)
    assert np.all(image[-4:] == 0.0)
    assert np.all(image[4:-4] == 1.0)


def test_preprocess_uses_fixed_log_normalization_and_clipping(tmp_path: Path) -> None:
    shared = 32.0
    baseline = np.zeros((272, 392), dtype=np.float64)
    baseline[136, 1] = shared
    values = baseline.copy()
    values[136, :4] = [0.0, shared, 4095.0, 8190.0]
    baseline_sample = _sample_for_array(tmp_path / "baseline", baseline)
    sample = _sample_for_array(tmp_path / "with-larger-count", values)

    image = preprocess_eye_density(sample, tmp_path / "with-larger-count")[..., 0]
    baseline_image = preprocess_eye_density(baseline_sample, tmp_path / "baseline")[..., 0]
    expected = np.log1p(shared) / np.log1p(4095.0)

    assert image[140, 1] == pytest.approx(expected, abs=1e-6)
    assert image[140, 1] == baseline_image[140, 1]
    assert image[140, 0] == 0.0
    assert image[140, 2] == 1.0
    assert image[140, 3] == 1.0


@pytest.mark.parametrize(
    "payload",
    [
        _npy_bytes(np.ones((2, 2, 1), dtype=np.float64)),
        _npy_bytes(np.empty((0, 2), dtype=np.float64)),
        _npy_bytes(np.array([[1.0, -1.0]], dtype=np.float64)),
        _npy_bytes(np.array([[np.nan]], dtype=np.float64)),
        _npy_bytes(np.array([[np.inf]], dtype=np.float64)),
        _npy_bytes(np.zeros((2, 2), dtype=np.float64)),
        _npy_bytes(np.ones((2, 2), dtype=np.complex128)),
        _npy_bytes(np.ones((2, 2), dtype=bool)),
        _npy_bytes(np.array([["not-a-count"]], dtype="U11")),
        _npy_bytes(np.array([[{"not": "safe"}]], dtype=object), allow_pickle=True),
        b"not an npy file",
    ],
    ids=[
        "three-dimensional",
        "empty-dimension",
        "negative",
        "nan",
        "infinite",
        "all-zero",
        "complex",
        "boolean",
        "string",
        "object",
        "corrupt",
    ],
)
def test_preprocess_rejects_invalid_density_payloads(tmp_path: Path, payload: bytes) -> None:
    sample = _sample_for_payload(tmp_path, payload)

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_preprocess_rejects_npz_payload(tmp_path: Path) -> None:
    stream = io.BytesIO()
    np.savez(stream, density=np.ones((2, 2), dtype=np.float64))
    sample = _sample_for_payload(tmp_path, stream.getvalue())

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_preprocess_rejects_truncated_zip_payload(tmp_path: Path) -> None:
    sample = _sample_for_payload(tmp_path, b"PK\x03\x04")

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_preprocess_rejects_wrong_media_type_with_stable_message(tmp_path: Path) -> None:
    payload = _npy_bytes(np.ones((2, 2), dtype=np.float64))
    wrong_type = _sample_for_payload(tmp_path / "wrong", payload, media_type="image/png")

    with pytest.raises(EyeDensityError, match="application/x-npy"):
        preprocess_eye_density(wrong_type, tmp_path / "wrong")


def test_preprocess_rejects_missing_file(tmp_path: Path) -> None:
    payload = _npy_bytes(np.ones((2, 2), dtype=np.float64))
    missing = _sample_for_payload(tmp_path / "missing", payload)
    (tmp_path / "missing/point-001/raw_eye_density.npy").unlink()

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(missing, tmp_path / "missing")


def test_preprocess_rejects_hash_mismatch_with_stable_message(tmp_path: Path) -> None:
    payload = _npy_bytes(np.ones((2, 2), dtype=np.float64))
    mismatch = _sample_for_payload(tmp_path / "mismatch", payload)
    (tmp_path / "mismatch/point-001/raw_eye_density.npy").write_bytes(b"replacement")

    with pytest.raises(EyeDensityError, match="hash mismatch"):
        preprocess_eye_density(mismatch, tmp_path / "mismatch")


def test_preprocess_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    payload = _npy_bytes(np.ones((2, 2), dtype=np.float64))
    (outside / "raw_eye_density.npy").write_bytes(payload)
    link = tmp_path / "point-001"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if sys.platform == "win32" and (
            error.winerror == 1314 or isinstance(error, PermissionError)
        ):
            pytest.skip("symlink creation unavailable: permission denied")
        raise
    sample = _sample_for_payload(
        tmp_path,
        payload,
        relative_path="placeholder/raw_eye_density.npy",
    )
    ref = ArtifactReference(
        relative_path="point-001/raw_eye_density.npy",
        sha256=hashlib.sha256(payload).hexdigest(),
        media_type="application/x-npy",
    )
    sample = EyeSample(
        measurement_id=sample.measurement_id,
        campaign_id=sample.campaign_id,
        sweep_point_id=sample.sweep_point_id,
        raw_eye_density_ref=ref,
        waveform_ref=None,
        metrics=sample.metrics,
        requested_params=sample.requested_params,
        applied_params=sample.applied_params,
        readback_params=sample.readback_params,
        dca_config_fingerprint=sample.dca_config_fingerprint,
        measurement_status=sample.measurement_status,
        captured_at=sample.captured_at,
        artifact_hashes={ref.relative_path: ref.sha256},
    )

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_preprocess_loads_the_same_verified_bytes_after_hash_time_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    original_payload = _npy_bytes(np.ones((2, 2), dtype=np.float64))
    replacement_payload = _npy_bytes(np.full((2, 2), 4095.0, dtype=np.float64))
    sample = _sample_for_payload(tmp_path, original_payload)
    baseline = preprocess_eye_density(sample, tmp_path)
    target = tmp_path / "point-001/raw_eye_density.npy"
    original_sha256 = preprocessing.hashlib.sha256

    class ReplacingHasher:
        def __init__(self, payload: bytes) -> None:
            self._delegate = original_sha256(payload)

        def hexdigest(self) -> str:
            target.write_bytes(replacement_payload)
            return self._delegate.hexdigest()

    monkeypatch.setattr(
        preprocessing.hashlib,
        "sha256",
        lambda payload: ReplacingHasher(payload),
    )
    result = preprocess_eye_density(sample, tmp_path)

    assert target.read_bytes() == replacement_payload
    assert np.array_equal(result, baseline)


def test_preprocess_rejects_file_larger_than_configured_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    monkeypatch.setattr(preprocessing, "MAX_DENSITY_FILE_BYTES", 8)
    sample = _sample_for_payload(tmp_path, b"x" * 9)

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_preprocess_rejects_array_with_too_many_elements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    monkeypatch.setattr(preprocessing, "MAX_DENSITY_ELEMENTS", 3)
    sample = _sample_for_array(tmp_path, np.ones((2, 2), dtype=np.uint8))

    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_preflight_rejects_header_over_element_limit_without_np_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    sample = _sample_for_payload(tmp_path, _npy_header_bytes((2, 2)))
    monkeypatch.setattr(preprocessing, "MAX_DENSITY_ELEMENTS", 3)
    load_called = False

    def never_load(*args: object, **kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("np.load must not receive an over-limit header")

    monkeypatch.setattr(preprocessing.np, "load", never_load)
    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)
    assert not load_called


def test_preflight_rejects_bodyless_giant_header_without_np_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    sample = _sample_for_payload(tmp_path, _npy_header_bytes((MAX_TEST_ELEMENTS := 16_777_217, 1)))
    load_called = False

    def never_load(*args: object, **kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("np.load must not receive a giant header")

    monkeypatch.setattr(preprocessing.np, "load", never_load)
    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)
    assert not load_called


def test_preprocess_rejects_replace_race_even_when_payload_is_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    payload = _npy_bytes(np.ones((2, 2), dtype=np.float64))
    sample = _sample_for_payload(tmp_path, payload)
    target = tmp_path / "point-001/raw_eye_density.npy"
    replacement = tmp_path / "replacement.npy"
    replacement.write_bytes(payload)
    original_open = preprocessing.os.open
    replaced = False

    def replacing_open(path: str | bytes | os.PathLike[str], flags: int) -> int:
        nonlocal replaced
        os.replace(replacement, target)
        replaced = True
        return original_open(path, flags)

    monkeypatch.setattr(preprocessing.os, "open", replacing_open)
    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)
    assert replaced


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_width", True),
        ("content_height", 0),
        ("canvas_width", 1.5),
        ("canvas_height", "280"),
        ("count_clip", True),
        ("count_clip", "4095"),
        ("count_clip", 0.0),
        ("count_clip", float("nan")),
        ("count_clip", float("inf")),
        ("count_clip", 1 + 2j),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError)):
        EyeDensityPreprocessConfig(**kwargs)


def test_config_requires_content_to_fit_and_is_frozen(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EyeDensityPreprocessConfig(content_width=393)
    config = EyeDensityPreprocessConfig()
    with pytest.raises(FrozenInstanceError):
        config.canvas_width = 1  # type: ignore[misc]
    sample = _sample_for_array(tmp_path, np.ones((2, 2), dtype=np.float64))
    with pytest.raises(TypeError):
        preprocess_eye_density(sample, tmp_path, config=object())  # type: ignore[arg-type]


def test_config_normalizes_fraction_and_numpy_real_scalars(tmp_path: Path) -> None:
    fraction_config = EyeDensityPreprocessConfig(count_clip=Fraction(1, 2))
    numpy_config = EyeDensityPreprocessConfig(count_clip=np.float64(2.5))
    sample = _sample_for_array(tmp_path, np.ones((2, 2), dtype=np.float64))

    assert type(fraction_config.count_clip) is float
    assert fraction_config.count_clip == 0.5
    assert type(numpy_config.count_clip) is float
    assert np.isfinite(preprocess_eye_density(sample, tmp_path, fraction_config)).all()


def test_config_rejects_fraction_too_large_to_normalize() -> None:
    with pytest.raises(ValueError):
        EyeDensityPreprocessConfig(count_clip=Fraction(10**10000, 1))


def test_preprocessing_source_excludes_orientation_randomness_and_hardware() -> None:
    import vcsel_ai_tuning.v2.preprocessing as preprocessing

    source = inspect.getsource(preprocessing)
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}

    assert not names.intersection({"flip", "fliplr", "flipud", "rot90", "random"})
    assert "hardware" not in source.lower()
    assert "max()" not in source
