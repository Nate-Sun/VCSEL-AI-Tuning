from math import inf, nextafter

import pytest

from vcsel_ai_tuning.hardware.fir import (
    calculate_first_pass,
    calculate_iterative_pass,
    format_setfir_preview,
    round_half_away_from_zero,
)


FIRST_DCA_TAPS = [
    0.008756,
    0.079768,
    -0.291888,
    1.336349,
    -0.084736,
    0.02357,
    -0.016326,
    -0.02607,
    -0.029423,
]

SECOND_DCA_TAPS = [
    0,
    0.080305,
    -0.278433,
    1.557635,
    -0.255666,
    -0.202142,
    0.034391,
    0.063909,
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.49, 0),
        (0.5, 1),
        (1.5, 2),
        (-0.49, 0),
        (-0.5, -1),
        (-1.5, -2),
    ],
)
def test_round_half_away_from_zero(value, expected):
    assert round_half_away_from_zero(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (nextafter(0.5, 0.0), 0),
        (nextafter(-0.5, 0.0), 0),
        (nextafter(0.5, inf), 1),
        (nextafter(-0.5, -inf), -1),
    ],
)
def test_round_half_away_from_zero_respects_adjacent_half_boundaries(
    value, expected
):
    assert round_half_away_from_zero(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_round_half_away_from_zero_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="finite"):
        round_half_away_from_zero(value)


def test_first_pass_matches_workbook_example():
    proposal = calculate_first_pass(FIRST_DCA_TAPS, dca_main_index=3, manual_main=90)

    assert proposal.mode == "first"
    assert proposal.current_taps is None
    assert proposal.dca_taps == tuple(FIRST_DCA_TAPS)
    assert proposal.dca_main_index == 3
    assert proposal.combined_main_index == 3
    assert proposal.normalization_denominator == pytest.approx(1.336349)
    assert proposal.full_vector == tuple(FIRST_DCA_TAPS)
    assert proposal.retained == tuple(FIRST_DCA_TAPS[:7])
    assert proposal.discarded_left == ()
    assert proposal.discarded_right == tuple(FIRST_DCA_TAPS[7:])
    assert proposal.discarded_l1_ratio == pytest.approx(0.02925478916497881)
    assert proposal.scaled == pytest.approx(
        (
            0.5896962544963927,
            5.372189450510309,
            -19.65797856697614,
            90.0,
            -5.706772706830327,
            1.5873847325810848,
            -1.0995181647907846,
        )
    )
    assert proposal.taps == (1, 5, -20, 90, -6, 2, -1)
    assert proposal.final_taps == proposal.taps
    assert proposal.l1 == 125
    assert proposal.valid is True
    assert proposal.reason is None
    assert proposal.max_feasible_main is None


def test_second_first_pass_matches_workbook_main_110_example():
    proposal = calculate_first_pass(SECOND_DCA_TAPS, dca_main_index=3, manual_main=110)

    assert proposal.taps == (0, 6, -20, 110, -18, -14, 2)
    assert proposal.l1 == 170
    assert proposal.valid is True


def test_first_pass_uses_dynamic_dca_main_index():
    prefixed_taps = [0.125, -0.25, *FIRST_DCA_TAPS]

    proposal = calculate_first_pass(
        prefixed_taps,
        dca_main_index=5,
        manual_main=90,
    )

    assert proposal.dca_main_index == 5
    assert proposal.combined_main_index == 5
    assert proposal.normalization_denominator == pytest.approx(1.336349)
    assert proposal.retained == tuple(FIRST_DCA_TAPS[:7])
    assert proposal.taps == (1, 5, -20, 90, -6, 2, -1)
    assert proposal.valid is True


def test_first_pass_rejects_more_than_170_without_scaling():
    proposal = calculate_first_pass(FIRST_DCA_TAPS, dca_main_index=3, manual_main=124)

    assert proposal.taps == (1, 7, -27, 124, -8, 2, -2)
    assert proposal.taps[3] == 124
    assert proposal.scaled[3] == 124.0
    assert proposal.l1 == 171
    assert proposal.valid is False
    assert proposal.reason == "tap L1 sum 171 exceeds limit 170"
    assert proposal.max_feasible_main == 123


def test_first_pass_accepts_l1_sum_equal_to_170():
    proposal = calculate_first_pass(FIRST_DCA_TAPS, dca_main_index=3, manual_main=123)

    assert proposal.taps == (1, 7, -27, 123, -8, 2, -2)
    assert proposal.l1 == 170
    assert proposal.valid is True


def test_first_pass_rejects_zero_normalization_main():
    proposal = calculate_first_pass(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dca_main_index=3,
        manual_main=90,
    )

    assert proposal.valid is False
    assert proposal.reason == "normalization Main must be finite and non-zero"
    assert proposal.taps == ()
    assert proposal.current_taps is None
    assert proposal.dca_taps == (0.0,) * 7
    assert proposal.dca_main_index == 3
    assert proposal.combined_main_index == 3
    assert proposal.normalization_denominator == 0.0


@pytest.mark.parametrize("manual_main", [0, -1, 90.5, True])
def test_first_pass_rejects_invalid_manual_main(manual_main):
    with pytest.raises(ValueError, match="manual_main must be a positive integer"):
        calculate_first_pass(FIRST_DCA_TAPS, dca_main_index=3, manual_main=manual_main)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_first_pass_rejects_nonfinite_dca_tap(bad_value):
    dca_taps = list(FIRST_DCA_TAPS)
    dca_taps[0] = bad_value

    with pytest.raises(ValueError, match="DCA taps must be finite"):
        calculate_first_pass(dca_taps, dca_main_index=3, manual_main=90)


@pytest.mark.parametrize("dca_main_index", [-1, len(FIRST_DCA_TAPS)])
def test_first_pass_rejects_out_of_range_main_index(dca_main_index):
    with pytest.raises(ValueError, match="dca_main_index"):
        calculate_first_pass(FIRST_DCA_TAPS, dca_main_index, manual_main=90)


def test_first_pass_zero_pads_missing_precursor_for_live_dca_shape():
    live_taps = [
        0.014569,
        -0.045870,
        1.089082,
        0.030131,
        0.011441,
        -0.004040,
        -0.037609,
        -0.029014,
        -0.028690,
    ]

    proposal = calculate_first_pass(live_taps, dca_main_index=2, manual_main=110)

    assert proposal.retained == pytest.approx(
        (0.0, 0.014569, -0.045870, 1.089082, 0.030131, 0.011441, -0.004040)
    )
    assert proposal.padded_left_count == 1
    assert proposal.padded_right_count == 0
    assert proposal.discarded_left == ()
    assert proposal.discarded_right == pytest.approx((-0.037609, -0.029014, -0.028690))
    assert proposal.taps == (0, 1, -5, 110, 3, 1, 0)
    assert proposal.l1 == 120
    assert proposal.valid is True


def test_first_pass_zero_pads_missing_postcursor_positions():
    taps = FIRST_DCA_TAPS[:7]

    proposal = calculate_first_pass(taps, dca_main_index=6, manual_main=90)

    assert proposal.retained == pytest.approx(
        (1.336349, -0.084736, 0.02357, -0.016326, 0.0, 0.0, 0.0)
    )
    assert proposal.padded_left_count == 0
    assert proposal.padded_right_count == 3
    assert proposal.discarded_left == pytest.approx(taps[:3])
    assert proposal.discarded_right == ()
    assert len(proposal.taps) == 7


def test_iterative_pass_convolves_then_crops_seven_taps():
    current = [0, 6, -28, 112, -8, -10, 0]
    dca = [0, 0, 0.0012, 1.188325, 0.009687, 0.011, -0.028617, -0.04]

    proposal = calculate_iterative_pass(
        current,
        dca,
        dca_main_index=3,
        manual_main=104,
    )

    assert proposal.mode == "iterative"
    assert proposal.current_taps == tuple(current)
    assert proposal.dca_taps == tuple(dca)
    assert proposal.dca_main_index == 3
    assert proposal.combined_main_index == 6
    assert proposal.normalization_denominator == pytest.approx(132.877564)
    assert proposal.full_vector == pytest.approx(
        (
            0,
            0,
            0,
            0.0072,
            7.096350000000001,
            -33.080578,
            132.877564,
            -8.913358,
            -10.16747,
            -2.269974,
            -4.361064000000001,
            0.60617,
            0.4,
            0,
        )
    )
    assert proposal.retained == pytest.approx(
        (
            0.0072,
            7.096350000000001,
            -33.080578,
            132.877564,
            -8.913358,
            -10.16747,
            -2.269974,
        )
    )
    assert proposal.discarded_left == pytest.approx((0, 0, 0))
    assert proposal.discarded_right == pytest.approx(
        (-4.361064000000001, 0.60617, 0.4, 0)
    )
    assert proposal.scaled == pytest.approx(
        (
            0.005635262849942071,
            5.554138545164781,
            -25.891354480279304,
            104.0,
            -6.976266000782495,
            -7.957828606791738,
            -1.7766527989631116,
        )
    )
    assert proposal.taps == (0, 6, -26, 104, -7, -8, -2)
    assert proposal.l1 == 153
    assert proposal.valid is True
    assert proposal.reason is None


def test_discarded_l1_ratio_is_reported():
    proposal = calculate_iterative_pass(
        [0, 6, -28, 112, -8, -10, 0],
        [0, 0, 0.0012, 1.188325, 0.009687, 0.011, -0.028617, -0.04],
        dca_main_index=3,
        manual_main=104,
    )

    assert proposal.discarded_l1_ratio == pytest.approx(0.02686575887219148)


def test_iterative_pass_rejects_over_limit_without_changing_main():
    proposal = calculate_iterative_pass(
        [0, 6, -28, 112, -8, -10, 0],
        [0, 0, 0.0012, 1.188325, 0.009687, 0.011, -0.028617, -0.04],
        dca_main_index=3,
        manual_main=117,
    )

    assert proposal.taps == (0, 6, -29, 117, -8, -9, -2)
    assert proposal.l1 == 171
    assert proposal.valid is False
    assert proposal.reason == "tap L1 sum 171 exceeds limit 170"
    assert proposal.max_feasible_main == 116


def test_iterative_pass_rejects_zero_combined_main_without_dividing():
    proposal = calculate_iterative_pass(
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        dca_main_index=3,
        manual_main=104,
    )

    assert proposal.valid is False
    assert proposal.reason == "normalization Main must be finite and non-zero"
    assert proposal.taps == ()


@pytest.mark.parametrize(
    "current",
    [
        [0, 6, -28, 112, -8, -10],
        [0, 6, -28, 112.0, -8, -10, 0],
        [0, 6, -28, True, -8, -10, 0],
    ],
)
def test_iterative_pass_rejects_current_fir_that_is_not_seven_integers(current):
    with pytest.raises(ValueError, match="current_taps must contain exactly seven integers"):
        calculate_iterative_pass(
            current,
            [0, 0, 0.0012, 1.188325, 0.009687, 0.011, -0.028617, -0.04],
            dca_main_index=3,
            manual_main=104,
        )


def test_setfir_preview_uses_vendor_order_and_preserves_nlc():
    preview = format_setfir_preview(
        port="port1",
        channel="0",
        taps=(1, 5, -20, 90, -6, 2, -1),
        upper_nlc=7,
        lower_nlc=-3,
    )

    assert preview == "setFir port1 0 -20 5 1 90 -6 2 -1 7 -3"


def test_setfir_preview_requires_nlc_values():
    with pytest.raises(TypeError):
        format_setfir_preview("port1", "0", (1, 5, -20, 90, -6, 2, -1))


def test_setfir_preview_rejects_non_seven_tap_input():
    with pytest.raises(ValueError, match="exactly seven"):
        format_setfir_preview(
            "port1",
            "0",
            (1, 5, -20, 90, -6, 2),
            upper_nlc=7,
            lower_nlc=-3,
        )


def test_setfir_preview_rejects_non_integer_taps():
    with pytest.raises(TypeError, match="integers"):
        format_setfir_preview(
            "port1",
            "0",
            (1, 5, -20, 90.0, -6, 2, -1),
            upper_nlc=7,
            lower_nlc=-3,
        )


@pytest.mark.parametrize(
    ("upper_nlc", "lower_nlc"),
    [
        (True, 0),
        (1.0, 0),
        (None, 0),
        (0, False),
        (0, 1.0),
        (0, None),
    ],
)
def test_setfir_preview_rejects_non_integer_nlc(upper_nlc, lower_nlc):
    with pytest.raises(TypeError, match="NLC values must be integers"):
        format_setfir_preview(
            "port1",
            "0",
            (1, 5, -20, 90, -6, 2, -1),
            upper_nlc=upper_nlc,
            lower_nlc=lower_nlc,
        )
