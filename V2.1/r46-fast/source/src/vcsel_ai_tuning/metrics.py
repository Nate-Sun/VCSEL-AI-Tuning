from __future__ import annotations

from math import atan2, pi

import numpy as np
from scipy.stats import beta

from .domain import Fidelity, MetricValue


def _validate_waveform(time_s: np.ndarray, amplitude: np.ndarray, ui_s: float) -> None:
    if time_s.ndim != 1 or amplitude.ndim != 1 or len(time_s) != len(amplitude):
        raise ValueError("time and amplitude must be equal-length one-dimensional arrays")
    if len(time_s) < 4:
        raise ValueError("waveform has too few samples")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(amplitude)):
        raise ValueError("waveform values must be finite")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time values must be strictly increasing")
    if not np.isfinite(ui_s) or ui_s <= 0:
        raise ValueError("ui_s must be finite and positive")


def _rising_crossing_phases(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    threshold: float,
    ui_s: float,
) -> np.ndarray:
    lower = amplitude[:-1]
    upper = amplitude[1:]
    mask = (lower < threshold) & (upper >= threshold) & (upper > lower)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"no rising crossings for threshold {threshold}")
    fraction = (threshold - amplitude[indices]) / (
        amplitude[indices + 1] - amplitude[indices]
    )
    crossing_time = time_s[indices] + fraction * (
        time_s[indices + 1] - time_s[indices]
    )
    return ((crossing_time / ui_s + 0.5) % 1.0) - 0.5


def _circular_center(phases_ui: np.ndarray) -> float:
    angles = 2.0 * pi * phases_ui
    center = atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    return float(((center / (2.0 * pi) + 0.5) % 1.0) - 0.5)


def _relative_phase(value: float, reference: float) -> float:
    return float(((value - reference + 0.5) % 1.0) - 0.5)


def level_metrics(levels: list[float]) -> dict[str, MetricValue]:
    values = np.asarray(levels, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("exactly four finite PAM4 levels are required")
    spacings = np.diff(values)
    if np.any(spacings <= 0):
        raise ValueError("PAM4 levels must be strictly increasing")
    outer = float(values[-1] - values[0])
    minimum = float(spacings.min())
    metrics = {
        "rlm_proxy": MetricValue(
            value=3.0 * minimum / outer,
            unit="1",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="rlm-proxy-v1",
        ),
        "worst_eye_height_norm": MetricValue(
            value=minimum / outer,
            unit="1",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="level-spacing-v1",
        ),
    }
    for index, spacing in enumerate(spacings):
        label = ("01", "12", "23")[index]
        metrics[f"eye_height_{label}_norm"] = MetricValue(
            value=float(spacing / outer),
            unit="1",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="level-spacing-v1",
        )
    return metrics


def analyze_pam4(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    ui_s: float,
    levels: list[float],
    transition_guard_ui: float = 0.10,
) -> dict[str, MetricValue]:
    time_s = np.asarray(time_s, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    _validate_waveform(time_s, amplitude, ui_s)
    values = np.asarray(levels, dtype=float)
    if values.shape != (4,) or np.any(np.diff(values) <= 0):
        raise ValueError("four strictly increasing PAM4 levels are required")
    thresholds = (values[:-1] + values[1:]) / 2.0
    centers = [
        _circular_center(_rising_crossing_phases(time_s, amplitude, value, ui_s))
        for value in thresholds
    ]
    reference = centers[1]
    offsets = [_relative_phase(center, reference) for center in centers]
    skew = max(offsets) - min(offsets)
    metrics = {
        "eye_center_01_ui": MetricValue(
            value=offsets[0],
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="crossing-center-v1",
        ),
        "eye_center_12_ui": MetricValue(
            value=offsets[1],
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="crossing-center-v1",
        ),
        "eye_center_23_ui": MetricValue(
            value=offsets[2],
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="crossing-center-v1",
        ),
        "skew_pp_ui": MetricValue(
            value=skew,
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="skew-v1",
        ),
        "common_eye_width_proxy_ui": MetricValue(
            value=max(0.0, 1.0 - skew - 2.0 * transition_guard_ui),
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="common-window-proxy-v1",
        ),
        "worst_eye_width_proxy_ui": MetricValue(
            value=max(0.0, 1.0 - skew - 2.0 * transition_guard_ui),
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="common-window-proxy-v1",
        ),
    }
    metrics.update(level_metrics(levels))
    return metrics


def ber_upper_confidence_bound(
    error_count: int,
    bit_count: int,
    confidence: float = 0.95,
) -> float:
    if bit_count <= 0:
        raise ValueError("bit_count must be positive")
    if error_count < 0 or error_count > bit_count:
        raise ValueError("error_count must be between zero and bit_count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if error_count == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / bit_count))
    if error_count == bit_count:
        return 1.0
    return float(beta.ppf(confidence, error_count + 1, bit_count - error_count))
