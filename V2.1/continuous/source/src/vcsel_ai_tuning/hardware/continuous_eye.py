"""FlexDCA screen-evidence wrapper for continuous eye optimization."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from vcsel_ai_tuning.campaign.repository import ArtifactSpec
from vcsel_ai_tuning.eye_image import derive_kimi_input_png


_SOURCE = re.compile(r"^(?:CHAN(?:NEL)?|FUNC(?:TION)?|DIFF|CMOD)[A-Z0-9]+$", re.I)
_PREFIX = re.compile(r"^[A-Za-z0-9_.-]+$")


class DcaEvidenceCaptureError(RuntimeError):
    """A VISA transfer failed while collecting DCA screen evidence."""

    def __init__(self, stage: str, command: str) -> None:
        self.stage = stage
        self.command = command
        super().__init__(f"DCA evidence {stage} failed for command {command}")


class ScreenEvidenceDcaAdapter:
    """Add exact screen PNG, RGB NPY, and a display fingerprint to a DCA read."""

    def __init__(
        self,
        adapter: Any,
        *,
        session: Any,
        source: str,
        artifact_prefix: str,
        artifact_directory: str | Path | None = None,
    ) -> None:
        if not callable(getattr(adapter, "measure_single", None)):
            raise TypeError("adapter must implement measure_single")
        if not callable(getattr(session, "write", None)) or not callable(
            getattr(session, "query", None)
        ):
            raise TypeError("session must implement write and query")
        if not callable(getattr(session, "query_binary_values", None)):
            raise TypeError("session must implement query_binary_values")
        if not isinstance(source, str) or _SOURCE.fullmatch(source) is None:
            raise ValueError("source must be an explicit FlexDCA waveform identifier")
        if not isinstance(artifact_prefix, str) or _PREFIX.fullmatch(artifact_prefix) is None:
            raise ValueError("artifact_prefix is invalid")
        self._adapter = adapter
        self._session = session
        self._source = source.upper().replace("FUNCTION", "FUNC").replace("CHANNEL", "CHAN")
        self._artifact_prefix = artifact_prefix
        self._sequence = self._existing_sequence(artifact_directory)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def measure_single(self) -> Any:
        measurement = self._adapter.measure_single()
        self._sequence += 1
        stem = f"{self._artifact_prefix}_{self._sequence:06d}"
        filename = f'%TEMP_DIR%\\{stem}.png'
        self._evidence_write(":DISK:SIMage:WINDow TIME1", "SCREEN_WINDOW_SELECT")
        self._evidence_write(
            f':DISK:SIMage:FNAMe "{filename}"',
            "SCREEN_FILENAME",
        )
        save_command = ":DISK:SIMage:SAVE;*OPC?"
        completed = str(self._query_text(save_command, "SCREEN_SAVE")).strip()
        if completed != "1":
            raise RuntimeError("FlexDCA screen capture did not complete")
        png_command = f':DISK:FILE:READ? "{filename}"'
        png = self._query_binary(
            png_command,
            "PNG_BINARY_READ",
            datatype="s",
            header_fmt="ieee",
            container=bytes,
        )
        if type(png) is not bytes or not png:
            raise ValueError("FlexDCA screen capture did not return bytes")
        try:
            with Image.open(BytesIO(png)) as image:
                if image.format != "PNG":
                    raise ValueError("FlexDCA screen capture is not a valid PNG")
                pixels = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
                kimi_png, kimi_width, kimi_height = derive_kimi_input_png(image)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("FlexDCA screen capture is not a valid PNG") from error
        npy_stream = BytesIO()
        np.save(npy_stream, pixels, allow_pickle=False)
        npy = npy_stream.getvalue()
        fingerprint_payload = self._fingerprint_payload()
        fingerprint = sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        png_name = f"{stem}.png"
        npy_name = f"model_input_{stem}.npy"
        kimi_png_name = f"kimi_input_{stem}.png"
        png_hash = sha256(png).hexdigest()
        npy_hash = sha256(npy).hexdigest()
        kimi_png_hash = sha256(kimi_png).hexdigest()
        evidence = {
            "provenance": "SCREEN_IMAGE_DERIVED",
            "dca_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "png_path": f"eye_artifacts/{png_name}",
            "npy_path": f"eye_artifacts/{npy_name}",
            "png_sha256": png_hash,
            "npy_sha256": npy_hash,
            "kimi_png_path": f"eye_artifacts/{kimi_png_name}",
            "kimi_png_sha256": kimi_png_hash,
            "kimi_source_png_sha256": png_hash,
            "kimi_resize_width": kimi_width,
            "kimi_resize_height": kimi_height,
            "kimi_resize_applied": pixels.shape[1] > 1280 or pixels.shape[0] > 528,
            "kimi_max_width": 1280,
            "kimi_max_height": 528,
            "kimi_resample": "LANCZOS",
            "kimi_colors": 256,
            "kimi_quantization": "MEDIANCUT",
            "kimi_dither": "NONE",
            "kimi_output_mode": "P",
            "kimi_provenance": "DCA_SCREEN_DERIVED",
        }
        raw = dict(measurement.raw_responses)
        raw["continuous_eye_evidence"] = evidence
        artifacts = tuple(measurement.artifacts) + (
            ArtifactSpec("eye_artifacts", png_name, png, "image/png"),
            ArtifactSpec("eye_artifacts", npy_name, npy, "application/x-npy"),
            ArtifactSpec("eye_artifacts", kimi_png_name, kimi_png, "image/png"),
        )
        return replace(measurement, raw_responses=raw, artifacts=artifacts)

    def _fingerprint_payload(self) -> dict[str, str]:
        queries = {
            "window": ":DISK:SIMage:WINDow?",
            "timebase_units": ":TIMebase:UNITs?",
            "ui_range": ":TIMebase:UIRange?",
            "ui_position": ":TIMebase:UIRPosition?",
            "vertical_scale": f":{self._source}:YSCale?",
            "vertical_offset": f":{self._source}:YOFFset?",
        }
        payload = {
            name: str(self._query_text(command, "FINGERPRINT_QUERY")).strip()
            for name, command in queries.items()
        }
        if payload["window"].upper() != "TIME1":
            raise ValueError("FlexDCA screen window is not TIME1")
        for name in ("ui_range", "ui_position", "vertical_scale", "vertical_offset"):
            try:
                value = float(payload[name])
            except ValueError as error:
                raise ValueError(f"FlexDCA fingerprint {name} is not numeric") from error
            if not isfinite(value):
                raise ValueError(f"FlexDCA fingerprint {name} is not finite")
        payload["source"] = self._source
        return payload

    def _query_text(self, command: str, stage: str) -> Any:
        try:
            return self._session.query(command)
        except Exception as error:
            raise DcaEvidenceCaptureError(stage, command) from error

    def _evidence_write(self, command: str, stage: str) -> Any:
        try:
            return self._session.write(command)
        except Exception as error:
            raise DcaEvidenceCaptureError(stage, command) from error

    def _query_binary(self, command: str, stage: str, **kwargs: Any) -> Any:
        try:
            return self._session.query_binary_values(command, **kwargs)
        except Exception as error:
            raise DcaEvidenceCaptureError(stage, command) from error

    def _existing_sequence(self, artifact_directory: str | Path | None) -> int:
        if artifact_directory is None:
            return 0
        directory = Path(artifact_directory)
        pattern = re.compile(
            rf"^{re.escape(self._artifact_prefix)}_(\d{{6}})\.png$"
        )
        return max(
            (
                int(match.group(1))
                for path in directory.glob(f"{self._artifact_prefix}_*.png")
                if (match := pattern.fullmatch(path.name)) is not None
            ),
            default=0,
        )


__all__ = ["DcaEvidenceCaptureError", "ScreenEvidenceDcaAdapter"]
