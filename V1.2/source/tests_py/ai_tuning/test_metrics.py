import numpy as np
import pytest

from vcsel_ai_tuning.metrics import (
    analyze_pam4,
    ber_upper_confidence_bound,
    level_metrics,
)


def synthetic_pam4_ramp(
    skew_pp_ui: float,
    global_delay_ui: float = 0.0,
    periods: int = 64,
    samples_per_ui: int = 512,
):
    ui_s = 1.0e-12
    time_s = np.arange(periods * samples_per_ui, dtype=float)
    time_s *= ui_s / samples_per_ui
    local_ui = ((time_s / ui_s - global_delay_ui + 0.5) % 1.0) - 0.5
    offsets = (-skew_pp_ui / 2.0, 0.0, skew_pp_ui / 2.0)
    amplitude = np.interp(
        local_ui,
        [-0.25, offsets[0], offsets[1], offsets[2], 0.25],
        [0.0, 0.5, 1.5, 2.5, 3.0],
    )
    return time_s, amplitude, ui_s


@pytest.mark.parametrize("injected", [0.0, 0.03, 0.06, 0.10])
def test_skew_metric_recovers_known_injection(injected):
    time_s, amplitude, ui_s = synthetic_pam4_ramp(injected)

    metrics = analyze_pam4(time_s, amplitude, ui_s, [0.0, 1.0, 2.0, 3.0])

    tolerance = max(0.005, injected * 0.10)
    assert metrics["skew_pp_ui"].value == pytest.approx(injected, abs=tolerance)


def test_global_delay_does_not_change_skew():
    base_t, base_y, ui_s = synthetic_pam4_ramp(0.06, global_delay_ui=0.0)
    shift_t, shift_y, _ = synthetic_pam4_ramp(0.06, global_delay_ui=0.21)

    base = analyze_pam4(base_t, base_y, ui_s, [0.0, 1.0, 2.0, 3.0])
    shifted = analyze_pam4(shift_t, shift_y, ui_s, [0.0, 1.0, 2.0, 3.0])

    assert abs(base["skew_pp_ui"].value - shifted["skew_pp_ui"].value) <= 0.005


def test_level_metrics_are_one_for_ideal_rlm_proxy():
    metrics = level_metrics([0.0, 1.0, 2.0, 3.0])

    assert metrics["rlm_proxy"].value == pytest.approx(1.0)
    assert metrics["worst_eye_height_norm"].value == pytest.approx(1.0 / 3.0)
    assert metrics["eye_height_01_norm"].value == pytest.approx(1.0 / 3.0)
    assert metrics["eye_height_12_norm"].value == pytest.approx(1.0 / 3.0)
    assert metrics["eye_height_23_norm"].value == pytest.approx(1.0 / 3.0)


def test_zero_error_upper_bound_is_not_zero():
    upper = ber_upper_confidence_bound(0, 1_000_000, confidence=0.95)

    assert upper == pytest.approx(2.995728e-6, rel=1e-5)


def test_nan_waveform_is_rejected():
    time_s, amplitude, ui_s = synthetic_pam4_ramp(0.03)
    amplitude[3] = np.nan

    with pytest.raises(ValueError, match="finite"):
        analyze_pam4(time_s, amplitude, ui_s, [0.0, 1.0, 2.0, 3.0])
