"""Deterministic preprocessing for stored eye-density count matrices."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from scipy import ndimage

from .contracts import EyeSample


MAX_DENSITY_FILE_BYTES = 64 * 1024 * 1024
MAX_DENSITY_ELEMENTS = 16_777_216


class EyeDensityError(ValueError):
    """Raised when an eye-density artifact is unsuitable for preprocessing."""


@dataclass(frozen=True, slots=True)
class EyeDensityPreprocessConfig:
    content_width: int = 392
    content_height: int = 272
    canvas_width: int = 392
    canvas_height: int = 280
    count_clip: float = 4095.0

    def __post_init__(self) -> None:
        for field_name in (
            "content_width",
            "content_height",
            "canvas_width",
            "canvas_height",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.content_width > self.canvas_width or self.content_height > self.canvas_height:
            raise ValueError("content dimensions must fit within the canvas")
        if isinstance(self.count_clip, bool) or not isinstance(
            self.count_clip, (Real, np.floating)
        ):
            raise ValueError("count_clip must be a positive finite real number")
        try:
            count_clip = float(self.count_clip)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("count_clip must be a positive finite real number") from None
        if not math.isfinite(count_clip) or count_clip <= 0:
            raise ValueError("count_clip must be a positive finite real number")
        object.__setattr__(self, "count_clip", count_clip)


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int]:
    device = getattr(stat_result, "st_dev", None)
    inode = getattr(stat_result, "st_ino", None)
    if type(device) is not int or type(inode) is not int or device < 0 or inode <= 0:
        raise EyeDensityError("invalid eye density artifact")
    return device, inode


def _artifact_path(sample: EyeSample, artifact_root: str | Path) -> tuple[Path, tuple[int, int]]:
    reference = sample.raw_eye_density_ref
    if reference.media_type != "application/x-npy":
        raise EyeDensityError("raw eye density media_type must be application/x-npy")
    try:
        root = Path(artifact_root).resolve(strict=True)
        relative = PurePosixPath(str(reference.relative_path))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise EyeDensityError("invalid eye density artifact")
        target = root.joinpath(*relative.parts).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise EyeDensityError("invalid eye density artifact")
        identity = _stat_identity(target.stat(follow_symlinks=False))
    except EyeDensityError:
        raise
    except (OSError, ValueError, RuntimeError):
        raise EyeDensityError("invalid eye density artifact") from None
    return target, identity


def _read_verified_payload(sample: EyeSample, artifact_root: str | Path) -> bytes:
    target, expected_identity = _artifact_path(sample, artifact_root)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            os.fspath(target),
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        if _stat_identity(os.fstat(descriptor)) != expected_identity:
            raise EyeDensityError("invalid eye density artifact")
        payload = bytearray()
        while len(payload) <= MAX_DENSITY_FILE_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_DENSITY_FILE_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
    except EyeDensityError:
        raise
    except (OSError, ValueError):
        raise EyeDensityError("invalid eye density artifact") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise EyeDensityError("invalid eye density artifact") from None
    if len(payload) > MAX_DENSITY_FILE_BYTES:
        raise EyeDensityError("invalid eye density artifact")
    verified_payload = bytes(payload)
    if hashlib.sha256(verified_payload).hexdigest() != sample.raw_eye_density_ref.sha256:
        raise EyeDensityError("eye density hash mismatch")
    return verified_payload


def _preflight_npy_stream(stream: io.BytesIO, payload_length: int) -> None:
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY version")
        dtype = np.dtype(dtype)
    except (OSError, ValueError, TypeError, EOFError):
        raise EyeDensityError("invalid eye density artifact") from None
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or dtype.fields is not None
        or dtype.kind not in {"i", "u", "f"}
    ):
        raise EyeDensityError("invalid eye density artifact")
    element_count = shape[0] * shape[1]
    if element_count > MAX_DENSITY_ELEMENTS:
        raise EyeDensityError("invalid eye density artifact")
    expected_data_bytes = element_count * dtype.itemsize
    if stream.tell() + expected_data_bytes != payload_length:
        raise EyeDensityError("invalid eye density artifact")


def _load_counts(payload: bytes) -> np.ndarray:
    stream = io.BytesIO(payload)
    _preflight_npy_stream(stream, len(payload))
    stream.seek(0)
    try:
        loaded = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        raise EyeDensityError("invalid eye density artifact") from None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        raise EyeDensityError("invalid eye density artifact")
    if not isinstance(loaded, np.ndarray):
        raise EyeDensityError("invalid eye density artifact")
    if loaded.ndim != 2 or 0 in loaded.shape or loaded.size > MAX_DENSITY_ELEMENTS:
        raise EyeDensityError("invalid eye density artifact")
    if not (
        np.issubdtype(loaded.dtype, np.integer)
        or np.issubdtype(loaded.dtype, np.floating)
    ):
        raise EyeDensityError("invalid eye density artifact")
    try:
        counts = loaded.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise EyeDensityError("invalid eye density artifact") from None
    if not np.isfinite(counts).all() or np.any(counts < 0) or not np.any(counts):
        raise EyeDensityError("invalid eye density artifact")
    return counts


def _scaled_content(counts: np.ndarray, config: EyeDensityPreprocessConfig) -> np.ndarray:
    normalized = np.log1p(np.clip(counts, 0.0, config.count_clip)) / math.log1p(
        float(config.count_clip)
    )
    source_height, source_width = normalized.shape
    scale = min(
        config.content_width / source_width,
        config.content_height / source_height,
    )
    target_width = int(round(source_width * scale))
    target_height = int(round(source_height * scale))
    if target_width < 1:
        target_width = 1
    if target_height < 1:
        target_height = 1
    try:
        resized = ndimage.zoom(
            normalized,
            (target_height / source_height, target_width / source_width),
            order=1,
            prefilter=False,
            mode="nearest",
            grid_mode=True,
        )
    except (OSError, ValueError):
        raise EyeDensityError("invalid eye density artifact") from None
    resized_target = np.zeros((target_height, target_width), dtype=np.float64)
    copy_height = min(target_height, resized.shape[0])
    copy_width = min(target_width, resized.shape[1])
    source_y = (resized.shape[0] - copy_height) // 2
    source_x = (resized.shape[1] - copy_width) // 2
    target_y = (target_height - copy_height) // 2
    target_x = (target_width - copy_width) // 2
    resized_target[target_y : target_y + copy_height, target_x : target_x + copy_width] = resized[
        source_y : source_y + copy_height,
        source_x : source_x + copy_width,
    ]
    content = np.zeros((config.content_height, config.content_width), dtype=np.float64)
    content_y = (config.content_height - target_height) // 2
    content_x = (config.content_width - target_width) // 2
    content[
        content_y : content_y + target_height,
        content_x : content_x + target_width,
    ] = resized_target
    return np.clip(content, 0.0, 1.0)


def preprocess_eye_density(
    sample: EyeSample,
    artifact_root: str | Path,
    config: EyeDensityPreprocessConfig = EyeDensityPreprocessConfig(),
) -> np.ndarray:
    """Load one verified count matrix and return a fixed RGB model input."""
    if type(config) is not EyeDensityPreprocessConfig:
        raise TypeError("config must be an EyeDensityPreprocessConfig")
    counts = _load_counts(_read_verified_payload(sample, artifact_root))
    content = _scaled_content(counts, config)
    canvas = np.zeros((config.canvas_height, config.canvas_width), dtype=np.float64)
    y_offset = (config.canvas_height - config.content_height) // 2
    x_offset = (config.canvas_width - config.content_width) // 2
    canvas[
        y_offset : y_offset + config.content_height,
        x_offset : x_offset + config.content_width,
    ] = content
    return np.ascontiguousarray(np.repeat(canvas[..., np.newaxis], 3, axis=2), dtype=np.float32)
