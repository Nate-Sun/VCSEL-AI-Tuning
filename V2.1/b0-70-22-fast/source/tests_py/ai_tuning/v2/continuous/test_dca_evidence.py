from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vcsel_ai_tuning.hardware.characterization import CampaignMeasurementResult
from vcsel_ai_tuning.hardware.continuous_eye import (
    DcaEvidenceCaptureError,
    ScreenEvidenceDcaAdapter,
)


def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (6, 4), (10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


class RecordingSession:
    def __init__(self, png: bytes) -> None:
        self.png = png
        self.writes = []
        self.queries = []
        self.binary_queries = []

    def write(self, command):
        self.writes.append(command)

    def query(self, command):
        self.queries.append(command)
        responses = {
            ":DISK:SIMage:SAVE;*OPC?": "1",
            ":DISK:SIMage:WINDow?": "TIME1",
            ":TIMebase:UNITs?": "UINTerval",
            ":TIMebase:UIRange?": "2.0",
            ":TIMebase:UIRPosition?": "0.0",
            ":FUNC1:YSCale?": "0.01",
            ":FUNC1:YOFFset?": "0.02",
        }
        return responses[command]

    def query_binary_values(self, command, **kwargs):
        self.binary_queries.append((command, kwargs))
        return self.png


class FailingSession(RecordingSession):
    def __init__(self, png: bytes, *, failing_command: str) -> None:
        super().__init__(png)
        self.failing_command = failing_command
        self._failure = TimeoutError("stored VISA timeout")

    def write(self, command):
        if command == self.failing_command and self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure
        return super().write(command)

    def query(self, command):
        if command == self.failing_command and self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure
        return super().query(command)

    def query_binary_values(self, command, **kwargs):
        if command == self.failing_command and self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure
        return super().query_binary_values(command, **kwargs)


class MeasurementAdapter:
    def measure_single(self):
        return CampaignMeasurementResult(
            acquisition_valid=True,
            er_db=4.5,
            er_status="CORR",
            tdecq_db=3.0,
            tdecq_status="CORR",
            tdecq_invalid_reason=None,
            elapsed_s=0.1,
            raw_responses={"measurement": "fake"},
            artifacts=(),
        )


def test_screen_evidence_adapter_captures_exact_png_and_deterministic_rgb_npy() -> None:
    png = _png()
    session = RecordingSession(png)
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=session,
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    result = adapter.measure_single()

    assert result.artifacts[0].media_type == "image/png"
    assert result.artifacts[0].content == png
    assert result.artifacts[0].category == "eye_artifacts"
    array = np.load(BytesIO(result.artifacts[1].content), allow_pickle=False)
    assert array.shape == (4, 6, 3)
    assert array.dtype == np.uint8
    evidence = result.raw_responses["continuous_eye_evidence"]
    assert evidence["provenance"] == "SCREEN_IMAGE_DERIVED"
    assert len(evidence["dca_fingerprint"]) == 64
    assert evidence["png_sha256"] != evidence["npy_sha256"]
    assert result.artifacts[2].media_type == "image/png"
    assert result.artifacts[2].category == "eye_artifacts"
    with Image.open(BytesIO(result.artifacts[2].content)) as kimi_input:
        assert kimi_input.size == (6, 4)
        assert kimi_input.convert("RGB").getpixel((0, 0)) == (10, 20, 30)
    assert evidence["kimi_png_path"] == "eye_artifacts/kimi_input_campaign-1_000001.png"
    assert len(evidence["kimi_png_sha256"]) == 64
    assert evidence["kimi_source_png_sha256"] == evidence["png_sha256"]
    assert evidence["kimi_resize_width"] == 6
    assert evidence["kimi_resize_height"] == 4
    assert evidence["kimi_resize_applied"] is False
    assert evidence["kimi_max_width"] == 1280
    assert evidence["kimi_max_height"] == 528
    assert evidence["kimi_resample"] == "LANCZOS"
    assert evidence["kimi_colors"] == 256
    assert evidence["kimi_quantization"] == "MEDIANCUT"
    assert evidence["kimi_dither"] == "NONE"
    assert evidence["kimi_output_mode"] == "P"
    assert evidence["kimi_provenance"] == "DCA_SCREEN_DERIVED"
    assert session.writes == [
        ":DISK:SIMage:WINDow TIME1",
        ':DISK:SIMage:FNAMe "%TEMP_DIR%\\campaign-1_000001.png"',
    ]
    assert session.queries[0] == ":DISK:SIMage:SAVE;*OPC?"
    assert session.binary_queries[0][0] == (
        ':DISK:FILE:READ? "%TEMP_DIR%\\campaign-1_000001.png"'
    )


def test_screen_evidence_adapter_continues_sequence_after_process_restart(
    tmp_path,
) -> None:
    png = _png()
    first = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=RecordingSession(png),
        source="FUNC1",
        artifact_prefix="campaign-1",
        artifact_directory=tmp_path,
    ).measure_single()
    (tmp_path / first.artifacts[0].relative_path).write_bytes(
        first.artifacts[0].content
    )

    restarted_session = RecordingSession(png)
    restarted = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=restarted_session,
        source="FUNC1",
        artifact_prefix="campaign-1",
        artifact_directory=tmp_path,
    ).measure_single()

    assert restarted.artifacts[0].relative_path == "campaign-1_000002.png"
    assert restarted_session.writes[1] == (
        ':DISK:SIMage:FNAMe "%TEMP_DIR%\\campaign-1_000002.png"'
    )


def test_screen_evidence_adapter_fails_closed_when_png_is_invalid() -> None:
    session = RecordingSession(b"not-a-png")
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=session,
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    import pytest

    with pytest.raises(ValueError, match="valid PNG"):
        adapter.measure_single()


def test_screen_evidence_adapter_downsamples_kimi_input_to_1280_width() -> None:
    pixels = np.zeros((970, 2353, 3), dtype=np.uint8)
    pixels[:, ::2, 1] = 255
    stream = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    session = RecordingSession(stream.getvalue())
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=session,
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    result = adapter.measure_single()

    original = result.artifacts[0]
    kimi_input = result.artifacts[2]
    assert original.content == stream.getvalue()
    assert result.raw_responses["continuous_eye_evidence"]["kimi_resize_applied"] is True
    with Image.open(BytesIO(kimi_input.content)) as image:
        assert image.size == (1280, 528)


def test_screen_evidence_adapter_keeps_high_entropy_kimi_png_under_byte_limit() -> None:
    random = np.random.default_rng(1)
    pixels = random.integers(0, 256, (970, 2353, 3), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=RecordingSession(stream.getvalue()),
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    result = adapter.measure_single()

    assert len(result.artifacts[2].content) <= 1024 * 1024


def test_screen_evidence_adapter_fits_kimi_input_inside_1280_by_528() -> None:
    pixels = np.zeros((1200, 640, 3), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=RecordingSession(stream.getvalue()),
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    result = adapter.measure_single()

    with Image.open(BytesIO(result.artifacts[2].content)) as image:
        assert image.size == (282, 528)


@pytest.mark.parametrize(
    ("failing_command", "stage"),
    [
        (":DISK:SIMage:SAVE;*OPC?", "SCREEN_SAVE"),
        (':DISK:FILE:READ? "%TEMP_DIR%\\campaign-1_000001.png"', "PNG_BINARY_READ"),
        (":TIMebase:UIRange?", "FINGERPRINT_QUERY"),
    ],
)
def test_dca_visa_timeout_identifies_evidence_stage(failing_command, stage) -> None:
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=FailingSession(_png(), failing_command=failing_command),
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    with pytest.raises(DcaEvidenceCaptureError) as caught:
        adapter.measure_single()

    error = caught.value
    assert error.stage == stage
    assert error.command == failing_command
    assert stage in str(error)
    assert failing_command in str(error)
    assert isinstance(error.__cause__, TimeoutError)


@pytest.mark.parametrize(
    ("failing_command", "stage"),
    [
        (":DISK:SIMage:WINDow TIME1", "SCREEN_WINDOW_SELECT"),
        (':DISK:SIMage:FNAMe "%TEMP_DIR%\\campaign-1_000001.png"', "SCREEN_FILENAME"),
    ],
)
def test_dca_visa_timeout_identifies_evidence_setup_write_stage(
    failing_command, stage
) -> None:
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=FailingSession(_png(), failing_command=failing_command),
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    with pytest.raises(DcaEvidenceCaptureError) as caught:
        adapter.measure_single()

    error = caught.value
    assert error.stage == stage
    assert error.command == failing_command
    assert stage in str(error)
    assert failing_command in str(error)
    assert isinstance(error.__cause__, TimeoutError)
