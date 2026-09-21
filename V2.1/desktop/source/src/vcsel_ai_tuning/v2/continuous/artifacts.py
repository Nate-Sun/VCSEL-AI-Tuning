"""Exact DCA screen-image evidence and deterministic derived model arrays."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import Field

from vcsel_ai_tuning.v2.contracts import ArtifactReference

from .models import FrozenModel, ImmutableMapping


class EyeArtifactError(ValueError):
    pass


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise EyeArtifactError("artifact path must be a safe relative path")
    return path


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


class EyeArtifactBundle(FrozenModel):
    png_ref: ArtifactReference
    npy_ref: ArtifactReference
    source_png_ref: ArtifactReference | None = None
    artifact_hashes: ImmutableMapping
    dca_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Literal["SCREEN_IMAGE_DERIVED"] = "SCREEN_IMAGE_DERIVED"

    @classmethod
    def from_png(
        cls,
        *,
        png_bytes: bytes,
        artifact_root: Path | str,
        relative_dir: str,
        stem: str,
        dca_fingerprint: str,
    ) -> "EyeArtifactBundle":
        if type(png_bytes) is not bytes or not png_bytes:
            raise EyeArtifactError("PNG must be non-empty bytes")
        directory = _safe_relative(relative_dir)
        if not stem or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in stem):
            raise EyeArtifactError("artifact stem is invalid")
        try:
            with Image.open(BytesIO(png_bytes)) as image:
                if image.format != "PNG":
                    raise EyeArtifactError("payload is not a PNG")
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as error:
            raise EyeArtifactError("payload is not a valid PNG") from error
        png_relative = (directory / f"{stem}.png").as_posix()
        npy_relative = (directory / f"model_input_{stem}.npy").as_posix()
        array_buffer = BytesIO()
        np.save(array_buffer, np.ascontiguousarray(pixels), allow_pickle=False)
        npy_bytes = array_buffer.getvalue()
        root = Path(artifact_root).resolve()
        png_path = root.joinpath(*PurePosixPath(png_relative).parts)
        npy_path = root.joinpath(*PurePosixPath(npy_relative).parts)
        _atomic_write(png_path, png_bytes)
        _atomic_write(npy_path, npy_bytes)
        png_hash = sha256(png_bytes).hexdigest()
        npy_hash = sha256(npy_bytes).hexdigest()
        bundle = cls(
            png_ref=ArtifactReference(
                relative_path=png_relative,
                sha256=png_hash,
                media_type="image/png",
            ),
            npy_ref=ArtifactReference(
                relative_path=npy_relative,
                sha256=npy_hash,
                media_type="application/x-npy",
            ),
            artifact_hashes={png_relative: png_hash, npy_relative: npy_hash},
            dca_fingerprint=dca_fingerprint,
        )
        bundle.verify(root)
        return bundle

    def verify(self, artifact_root: Path | str) -> bool:
        root = Path(artifact_root).resolve()
        for reference in (self.png_ref, self.npy_ref, self.source_png_ref):
            if reference is None:
                continue
            relative = _safe_relative(reference.relative_path)
            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise EyeArtifactError("artifact is missing")
            if sha256(path.read_bytes()).hexdigest() != reference.sha256:
                raise EyeArtifactError("artifact hash mismatch")
        try:
            pixels = np.load(
                root.joinpath(*PurePosixPath(self.npy_ref.relative_path).parts),
                allow_pickle=False,
            )
        except Exception as error:
            raise EyeArtifactError("derived NPY is invalid") from error
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise EyeArtifactError("derived NPY has invalid pixel shape")
        return True


def fingerprints_compatible(left: str, right: str) -> bool:
    for value in (left, right):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("DCA fingerprint must be a lowercase SHA-256 value")
    return left == right
