"""Deterministic provider-image derivation shared by capture and verification."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


KIMI_INPUT_MAX_WIDTH = 1280
KIMI_INPUT_MAX_HEIGHT = 528
KIMI_INPUT_COLORS = 256


def derive_kimi_input_png(image: Image.Image) -> tuple[bytes, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(KIMI_INPUT_MAX_WIDTH / width, KIMI_INPUT_MAX_HEIGHT / height, 1.0)
    if scale < 1.0:
        target_width = round(width * scale)
        target_height = round(height * scale)
        rgb = rgb.resize((target_width, target_height), Image.Resampling.LANCZOS)
    indexed = rgb.quantize(
        colors=KIMI_INPUT_COLORS,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    stream = BytesIO()
    indexed.save(stream, format="PNG", compress_level=6)
    return stream.getvalue(), indexed.width, indexed.height


__all__ = [
    "KIMI_INPUT_COLORS",
    "KIMI_INPUT_MAX_HEIGHT",
    "KIMI_INPUT_MAX_WIDTH",
    "derive_kimi_input_png",
]
