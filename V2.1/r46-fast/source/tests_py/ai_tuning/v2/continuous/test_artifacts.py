from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vcsel_ai_tuning.v2.continuous.artifacts import (
    EyeArtifactBundle,
    EyeArtifactError,
    fingerprints_compatible,
)


def _png_bytes() -> bytes:
    pixels = np.asarray(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_bundle_preserves_exact_png_and_derives_deterministic_npy(tmp_path) -> None:
    png = _png_bytes()
    bundle = EyeArtifactBundle.from_png(
        png_bytes=png,
        artifact_root=tmp_path,
        relative_dir="campaign-1/rounds/0001",
        stem="eye_before",
        dca_fingerprint="a" * 64,
    )

    assert (tmp_path / bundle.png_ref.relative_path).read_bytes() == png
    pixels = np.load(tmp_path / bundle.npy_ref.relative_path, allow_pickle=False)
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [
        [[0, 10, 20], [30, 40, 50]],
        [[60, 70, 80], [90, 100, 110]],
    ]
    assert bundle.provenance == "SCREEN_IMAGE_DERIVED"
    assert bundle.verify(tmp_path) is True
    assert bundle.artifact_hashes == {
        bundle.png_ref.relative_path: bundle.png_ref.sha256,
        bundle.npy_ref.relative_path: bundle.npy_ref.sha256,
    }


def test_bundle_rejects_path_escape_corrupt_png_and_hash_mismatch(tmp_path) -> None:
    with pytest.raises(EyeArtifactError, match="relative"):
        EyeArtifactBundle.from_png(
            png_bytes=_png_bytes(),
            artifact_root=tmp_path,
            relative_dir="../outside",
            stem="eye",
            dca_fingerprint="a" * 64,
        )
    with pytest.raises(EyeArtifactError, match="PNG"):
        EyeArtifactBundle.from_png(
            png_bytes=b"not-png",
            artifact_root=tmp_path,
            relative_dir="campaign-1",
            stem="eye",
            dca_fingerprint="a" * 64,
        )

    bundle = EyeArtifactBundle.from_png(
        png_bytes=_png_bytes(),
        artifact_root=tmp_path,
        relative_dir="campaign-1",
        stem="eye",
        dca_fingerprint="a" * 64,
    )
    (tmp_path / bundle.png_ref.relative_path).write_bytes(b"changed")
    with pytest.raises(EyeArtifactError, match="hash"):
        bundle.verify(tmp_path)


def test_fingerprint_compatibility_is_exact_and_strict() -> None:
    assert fingerprints_compatible("a" * 64, "a" * 64) is True
    assert fingerprints_compatible("a" * 64, "b" * 64) is False
    with pytest.raises(ValueError):
        fingerprints_compatible("not-a-hash", "a" * 64)
