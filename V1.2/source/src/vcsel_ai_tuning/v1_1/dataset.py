"""Optional standardized measurement datasets for V1.1 Sweep points.

This module owns filesystem evidence only.  It has no DCA, I2C, or BERT
connection factory and cannot perform hardware mutation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Protocol
from uuid import uuid4
import zlib

import numpy as np

from .constants import SCHEMA_VERSION


class MeasurementStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class DatasetStatus(StrEnum):
    DISABLED = "DISABLED"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


def _detached_json_value(
    value: object,
    *,
    path: str,
    active: set[int] | None = None,
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must not contain nonfinite numbers")
        return value
    if active is None:
        active = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain recursive mappings")
        active.add(identity)
        try:
            detached: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} mapping keys must be text")
                detached[key] = _detached_json_value(
                    item,
                    path=f"{path}.{key}",
                    active=active,
                )
            return detached
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain recursive sequences")
        active.add(identity)
        try:
            return [
                _detached_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


class DatasetCaptureError(RuntimeError):
    """Dataset capture failure carrying detached JSON-safe audit evidence."""

    def __init__(
        self,
        message: str,
        *,
        raw_responses: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        detached = _detached_json_value(
            raw_responses,
            path="raw_responses",
        )
        if not isinstance(detached, dict):  # pragma: no cover
            raise TypeError("raw_responses must be a mapping")
        self._raw_responses_json = json.dumps(
            detached,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _fresh_raw_responses(self) -> dict[str, object]:
        restored = json.loads(self._raw_responses_json.decode("utf-8"))
        if not isinstance(restored, dict):  # pragma: no cover
            raise RuntimeError("stored capture evidence is not a mapping")
        return restored

    @property
    def raw_responses(self) -> dict[str, object]:
        return self._fresh_raw_responses()


def _nonblank(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _integer_mapping(
    label: str,
    value: Mapping[str, int],
) -> dict[str, int]:
    copied = dict(value)
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(item, bool)
        or not isinstance(item, int)
        for name, item in copied.items()
    ):
        raise TypeError(f"{label} must map nonblank names to integers")
    return copied


@dataclass(frozen=True, slots=True)
class MeasurementArtifactContext:
    campaign_id: str
    sweep_point_id: str
    measurement_id: str
    acquisition_id: str
    captured_at_utc: datetime
    requested_params: Mapping[str, int]
    applied_params: Mapping[str, int]
    readback_params: Mapping[str, int]
    tdecq: float | None
    outer_er: float | None
    outer_er_source: str
    dca_fingerprint: Mapping[str, object]
    dca_identity: str
    bert_port: str
    bert_channel: str
    bert_pattern: str
    measurement_status: MeasurementStatus
    dca_acquisition_duration_s: float
    observation_id: str | None = None

    def __post_init__(self) -> None:
        for label in (
            "campaign_id",
            "sweep_point_id",
            "measurement_id",
            "acquisition_id",
            "outer_er_source",
            "dca_identity",
            "bert_port",
            "bert_channel",
            "bert_pattern",
        ):
            _nonblank(label, getattr(self, label))
        if (
            self.captured_at_utc.tzinfo is None
            or self.captured_at_utc.utcoffset() is None
        ):
            raise ValueError("captured_at_utc must be timezone-aware")
        if not isinstance(self.measurement_status, MeasurementStatus):
            raise TypeError("measurement_status must be MeasurementStatus")
        if self.observation_id is not None:
            _nonblank("observation_id", self.observation_id)
        duration = float(self.dca_acquisition_duration_s)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("dca_acquisition_duration_s must be nonnegative")
        object.__setattr__(
            self,
            "requested_params",
            _integer_mapping("requested_params", self.requested_params),
        )
        object.__setattr__(
            self,
            "applied_params",
            _integer_mapping("applied_params", self.applied_params),
        )
        object.__setattr__(
            self,
            "readback_params",
            _integer_mapping("readback_params", self.readback_params),
        )
        object.__setattr__(self, "dca_fingerprint", dict(self.dca_fingerprint))
        object.__setattr__(self, "dca_acquisition_duration_s", duration)


@dataclass(frozen=True, slots=True)
class FrozenDatasetCapture:
    acquisition_id: str
    same_acquisition_verified: bool = False
    raw_eye_density: np.ndarray | None = None
    density_axis_order: tuple[str, str] = ("time", "amplitude")
    density_source: str = "CHAN1A"
    tdecq_equalized_eye_png: bytes | None = None
    raw_eye_png: bytes | None = None
    rlm: float | None = None
    oma: float | None = None
    average_power: float | None = None
    equalizer_taps: tuple[float, ...] | None = None
    equalizer_main_index: int | None = None
    equalizer_source: str | None = None
    raw_responses: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _nonblank("acquisition_id", self.acquisition_id)
        if type(self.same_acquisition_verified) is not bool:
            raise TypeError("same_acquisition_verified must be a boolean")
        if (
            not isinstance(self.density_axis_order, tuple)
            or len(self.density_axis_order) != 2
            or any(not isinstance(item, str) or not item for item in self.density_axis_order)
        ):
            raise ValueError("density_axis_order must contain two nonblank names")
        _nonblank("density_source", self.density_source)
        density = self.raw_eye_density
        if density is not None:
            if not isinstance(density, np.ndarray):
                raise TypeError("raw_eye_density must be a numpy array")
            if density.ndim != 2:
                raise ValueError("raw_eye_density must be two-dimensional")
            if density.dtype.hasobject:
                raise TypeError("raw_eye_density must not use object dtype")
            frozen = density.copy()
            frozen.flags.writeable = False
            object.__setattr__(self, "raw_eye_density", frozen)
        if self.raw_responses is not None:
            object.__setattr__(self, "raw_responses", dict(self.raw_responses))


@dataclass(frozen=True, slots=True)
class DatasetFileEvidence:
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class DatasetOutcome:
    status: DatasetStatus
    bundle_path: str | None = None
    error: str | None = None
    missing_items: tuple[str, ...] = ()
    files: Mapping[str, DatasetFileEvidence] | None = None
    persistence_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.status, DatasetStatus):
            raise TypeError("status must be DatasetStatus")
        object.__setattr__(self, "missing_items", tuple(self.missing_items))
        object.__setattr__(self, "files", dict(self.files or {}))

    def payload(self) -> dict[str, object]:
        return {
            "dataset_status": self.status.value,
            "dataset_bundle_path": self.bundle_path,
            "dataset_error": self.error,
            "dataset_missing_items": list(self.missing_items),
            "dataset_files": {
                name: {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "size": item.size,
                }
                for name, item in self.files.items()
            },
            "dataset_persistence_duration_s": self.persistence_duration_s,
        }


class DatasetRecorder(Protocol):
    def record(
        self,
        context: MeasurementArtifactContext,
        capture: Callable[[], FrozenDatasetCapture],
    ) -> DatasetOutcome: ...


class NoOpDatasetRecorder:
    def record(
        self,
        context: MeasurementArtifactContext,
        capture: Callable[[], FrozenDatasetCapture],
    ) -> DatasetOutcome:
        del context, capture
        return DatasetOutcome(status=DatasetStatus.DISABLED)


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def encode_density_png(density: np.ndarray) -> bytes:
    """Render a 2D density matrix without changing its geometric axes."""

    if not isinstance(density, np.ndarray) or density.ndim != 2:
        raise ValueError("PNG density must be a two-dimensional numpy array")
    if density.dtype.hasobject or not np.issubdtype(density.dtype, np.number):
        raise TypeError("PNG density must use a numeric non-object dtype")
    if np.issubdtype(density.dtype, np.complexfloating):
        raise TypeError("PNG density must not use a complex dtype")
    values = density.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("PNG density must contain only finite values")
    minimum = float(values.min()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0
    if maximum == minimum:
        pixels = np.zeros(values.shape, dtype=np.uint8)
    else:
        pixels = np.rint(
            (values - minimum) * (255.0 / (maximum - minimum))
        ).astype(np.uint8)
    height, width = pixels.shape
    if width <= 0 or height <= 0:
        raise ValueError("PNG density dimensions must be positive")
    scanlines = b"".join(
        b"\x00" + pixels[row].tobytes(order="C")
        for row in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def decode_png_dimensions(source: str | Path | bytes) -> tuple[int, int]:
    """Validate the grayscale PNG produced/accepted by dataset recording."""

    content = source if isinstance(source, bytes) else Path(source).read_bytes()
    if not content.startswith(_PNG_SIGNATURE):
        raise ValueError("image is not a PNG")
    offset = len(_PNG_SIGNATURE)
    width = height = None
    idat = bytearray()
    saw_end = False
    while offset < len(content):
        if offset + 12 > len(content):
            raise ValueError("PNG chunk is truncated")
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        kind = content[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(content):
            raise ValueError("PNG chunk payload is truncated")
        payload = content[payload_start:payload_end]
        expected_crc = struct.unpack(">I", content[payload_end:crc_end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if kind == b"IHDR":
            if length != 13:
                raise ValueError("PNG IHDR has an invalid length")
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if (
                width <= 0
                or height <= 0
                or depth != 8
                or color != 0
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ValueError("PNG must be non-interlaced 8-bit grayscale")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            offset = crc_end
            break
        offset = crc_end
    if width is None or height is None or not idat or not saw_end:
        raise ValueError("PNG is missing required chunks")
    scanlines = zlib.decompress(bytes(idat))
    expected_size = height * (width + 1)
    if len(scanlines) != expected_size:
        raise ValueError("PNG scanline size does not match IHDR")
    for row in range(height):
        if scanlines[row * (width + 1)] != 0:
            raise ValueError("PNG uses an unsupported row filter")
    return width, height


def _safe_component(label: str, value: str) -> str:
    _nonblank(label, value)
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError(f"{label} contains an unsafe path character")
    if value in {".", ".."}:
        raise ValueError(f"{label} must not be a path traversal segment")
    return value


def _metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) and abs(result) < 9.0e37 else None


class MeasurementArtifactBundleWriter:
    """Validate and atomically publish one optional Sweep dataset bundle."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._artifact_root = Path(artifact_root)
        self._monotonic = monotonic
        self._sleep = sleep

    @staticmethod
    def _write_bytes(path: Path, content: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _write_npy(path: Path, density: np.ndarray) -> None:
        with path.open("xb") as stream:
            np.save(stream, density, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _file_evidence(path: Path) -> DatasetFileEvidence:
        content = path.read_bytes()
        return DatasetFileEvidence(
            relative_path=path.name,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )

    @staticmethod
    def _record_payload(
        context: MeasurementArtifactContext,
        capture: FrozenDatasetCapture | None,
        *,
        status: DatasetStatus,
        error: str | None,
        missing: tuple[str, ...],
        files: Mapping[str, DatasetFileEvidence],
        file_duration_s: float,
        failed_capture_raw_responses: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        density = None if capture is None else capture.raw_eye_density
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": context.campaign_id,
            "sweep_point_id": context.sweep_point_id,
            "observation_id": context.observation_id,
            "measurement_id": context.measurement_id,
            "acquisition_id": context.acquisition_id,
            "same_acquisition_verified": (
                False
                if capture is None
                else capture.same_acquisition_verified
            ),
            "captured_at_utc": context.captured_at_utc.isoformat(),
            "requested_params": dict(context.requested_params),
            "applied_params": dict(context.applied_params),
            "readback_params": dict(context.readback_params),
            "metrics": {
                "tdecq": context.tdecq,
                "outer_er": context.outer_er,
                "rlm": None if capture is None else _metric(capture.rlm),
                "oma": None if capture is None else _metric(capture.oma),
                "average_power": (
                    None if capture is None else _metric(capture.average_power)
                ),
            },
            "outer_er_source": context.outer_er_source,
            "equalizer": {
                "taps": (
                    None
                    if capture is None or capture.equalizer_taps is None
                    else list(capture.equalizer_taps)
                ),
                "main_index": (
                    None if capture is None else capture.equalizer_main_index
                ),
                "source": None if capture is None else capture.equalizer_source,
            },
            "dca_fingerprint": dict(context.dca_fingerprint),
            "dca_identity": context.dca_identity,
            "bert": {
                "port": context.bert_port,
                "channel": context.bert_channel,
                "pattern": context.bert_pattern,
            },
            "measurement_status": context.measurement_status.value,
            "dataset_status": status.value,
            "dataset_error": error,
            "missing_items": list(missing),
            "density": (
                None
                if density is None
                else {
                    "shape": list(density.shape),
                    "dtype": str(density.dtype),
                    "axis_order": list(capture.density_axis_order),
                    "source": capture.density_source,
                }
            ),
            "files": {
                name: {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "size": item.size,
                }
                for name, item in files.items()
            },
            "dca_acquisition_duration_s": (
                context.dca_acquisition_duration_s
            ),
            "file_persistence_duration_s": file_duration_s,
            "raw_responses": (
                dict(failed_capture_raw_responses)
                if failed_capture_raw_responses is not None
                else (
                    None
                    if capture is None or capture.raw_responses is None
                    else dict(capture.raw_responses)
                )
            ),
        }

    def record(
        self,
        context: MeasurementArtifactContext,
        capture: Callable[[], FrozenDatasetCapture],
    ) -> DatasetOutcome:
        started = self._monotonic()
        campaign = _safe_component("campaign_id", context.campaign_id)
        point = _safe_component("sweep_point_id", context.sweep_point_id)
        parent = self._artifact_root / campaign
        final = parent / point
        bundle_path = Path(campaign, point).as_posix()
        if final.exists():
            return DatasetOutcome(
                status=DatasetStatus.FAILED,
                error=f"dataset destination already exists: {bundle_path}",
                persistence_duration_s=max(0.0, self._monotonic() - started),
            )
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{point}.tmp-{uuid4().hex}"
        temporary.mkdir()
        captured: FrozenDatasetCapture | None = None
        errors: list[str] = []
        missing: set[str] = set()
        files: dict[str, DatasetFileEvidence] = {}
        allow_capture_files = True
        failed_capture_raw_responses: Mapping[str, object] | None = None
        try:
            try:
                captured = capture()
                if not isinstance(captured, FrozenDatasetCapture):
                    raise TypeError(
                        "dataset capture provider returned an invalid object"
                    )
            except Exception as capture_error:
                if isinstance(capture_error, DatasetCaptureError):
                    original_snapshot = DatasetCaptureError._fresh_raw_responses(
                        capture_error
                    )
                    detached_snapshot = _detached_json_value(
                        original_snapshot,
                        path="failed_capture_raw_responses",
                    )
                    if not isinstance(detached_snapshot, dict):  # pragma: no cover
                        raise RuntimeError(
                            "failed capture evidence is not a mapping"
                        )
                    failed_capture_raw_responses = detached_snapshot
                errors.append(f"capture failed: {capture_error}")
                missing.add("capture")
                allow_capture_files = False
            if (
                captured is not None
                and captured.acquisition_id != context.acquisition_id
            ):
                errors.append(
                    "capture acquisition_id does not match primary measurement"
                )
                missing.add("acquisition_id")
                allow_capture_files = False
            if (
                captured is not None
                and captured.same_acquisition_verified is not True
            ):
                errors.append(
                    "same DCA acquisition could not be independently verified"
                )
                missing.add("same_acquisition_proof")

            if captured is not None:
                for name, value in (
                    ("rlm", _metric(captured.rlm)),
                    ("oma", _metric(captured.oma)),
                    ("average_power", _metric(captured.average_power)),
                    ("equalizer_taps", captured.equalizer_taps),
                    ("equalizer_main_index", captured.equalizer_main_index),
                    ("equalizer_source", captured.equalizer_source),
                ):
                    if value is None:
                        missing.add(name)

            raw_png: bytes | None = None
            if allow_capture_files and captured is not None:
                density = captured.raw_eye_density
                if density is None:
                    missing.update(("raw_eye_density.npy", "raw_eye.png"))
                else:
                    try:
                        npy_path = temporary / "raw_eye_density.npy"
                        self._write_npy(npy_path, density)
                        files[npy_path.name] = self._file_evidence(npy_path)
                    except Exception as file_error:
                        npy_path.unlink(missing_ok=True)
                        errors.append(f"raw_eye_density.npy: {file_error}")
                        missing.add("raw_eye_density.npy")
                    try:
                        raw_png = encode_density_png(density)
                        raw_path = temporary / "raw_eye.png"
                        self._write_bytes(raw_path, raw_png)
                        files[raw_path.name] = self._file_evidence(raw_path)
                    except Exception as file_error:
                        errors.append(f"raw_eye.png: {file_error}")
                        missing.add("raw_eye.png")

                equalized = captured.tdecq_equalized_eye_png
                if equalized is None:
                    missing.add("tdecq_equalized_eye.png")
                else:
                    try:
                        decode_png_dimensions(equalized)
                        if raw_png is not None and equalized == raw_png:
                            raise ValueError(
                                "equalized eye must differ from raw eye"
                            )
                        equalized_path = temporary / "tdecq_equalized_eye.png"
                        self._write_bytes(equalized_path, equalized)
                        files[equalized_path.name] = self._file_evidence(
                            equalized_path
                        )
                    except Exception as file_error:
                        errors.append(
                            f"tdecq_equalized_eye.png: {file_error}"
                        )
                        missing.add("tdecq_equalized_eye.png")
            else:
                missing.update(
                    (
                        "raw_eye_density.npy",
                        "raw_eye.png",
                        "tdecq_equalized_eye.png",
                    )
                )

            if not missing and len(files) == 3:
                status = DatasetStatus.COMPLETE
            elif files:
                status = DatasetStatus.PARTIAL
            else:
                status = DatasetStatus.FAILED
            error_text = "; ".join(errors) or (
                None
                if not missing
                else "missing required dataset items: "
                + ", ".join(sorted(missing))
            )
            file_duration_s = max(0.0, self._monotonic() - started)
            record_capture = captured if allow_capture_files else None
            record = self._record_payload(
                context,
                record_capture,
                status=status,
                error=error_text,
                missing=tuple(sorted(missing)),
                files=files,
                file_duration_s=file_duration_s,
                failed_capture_raw_responses=failed_capture_raw_responses,
            )
            record_path = temporary / "record.json"
            self._write_bytes(
                record_path,
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            if final.exists():
                raise FileExistsError(
                    f"dataset destination already exists: {bundle_path}"
                )
            for attempt in range(10):
                try:
                    temporary.rename(final)
                    break
                except OSError as publish_error:
                    transient_access_denied = (
                        isinstance(publish_error, PermissionError)
                        or getattr(publish_error, "winerror", None) == 5
                    )
                    if not transient_access_denied or attempt == 9:
                        raise
                    if final.exists():
                        raise FileExistsError(
                            f"dataset destination already exists: {bundle_path}"
                        ) from publish_error
                    self._sleep(0.1)
            return DatasetOutcome(
                status=status,
                bundle_path=bundle_path,
                error=error_text,
                missing_items=tuple(sorted(missing)),
                files=files,
                persistence_duration_s=max(
                    0.0, self._monotonic() - started
                ),
            )
        except Exception as persistence_error:
            shutil.rmtree(temporary, ignore_errors=True)
            return DatasetOutcome(
                status=DatasetStatus.FAILED,
                error=f"dataset persistence failed: {persistence_error}",
                missing_items=tuple(sorted(missing)),
                persistence_duration_s=max(
                    0.0, self._monotonic() - started
                ),
            )
