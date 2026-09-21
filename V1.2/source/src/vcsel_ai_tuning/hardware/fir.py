from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Sequence


FIR_TAP_COUNT = 7
FIR_MAIN_INDEX = 3
FIR_L1_LIMIT = 170


@dataclass(frozen=True)
class FirProposal:
    mode: str
    manual_main: int
    current_taps: tuple[int, ...] | None
    dca_taps: tuple[float, ...]
    dca_main_index: int
    combined_main_index: int
    normalization_denominator: float
    full_vector: tuple[float, ...]
    retained: tuple[float, ...]
    padded_left_count: int
    padded_right_count: int
    discarded_left: tuple[float, ...]
    discarded_right: tuple[float, ...]
    discarded_l1_ratio: float
    scaled: tuple[float, ...]
    final_taps: tuple[int, ...]
    l1: int
    valid: bool
    reason: str | None
    max_feasible_main: int | None

    @property
    def taps(self) -> tuple[int, ...]:
        return self.final_taps


def round_half_away_from_zero(value: float) -> int:
    """Round a finite value to the nearest integer, with halves away from zero."""
    if not isfinite(value):
        raise ValueError("value must be finite")
    magnitude = abs(value)
    integer_part = floor(magnitude)
    rounded_magnitude = integer_part + (magnitude - integer_part >= 0.5)
    return rounded_magnitude if value >= 0 else -rounded_magnitude


def _validate_manual_main(manual_main: int) -> None:
    if type(manual_main) is not int or manual_main <= 0:
        raise ValueError("manual_main must be a positive integer")


def _dca_vector(dca_taps: Sequence[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in dca_taps)
    except (TypeError, ValueError) as error:
        raise ValueError("DCA taps must contain only numbers") from error
    if any(not isfinite(value) for value in vector):
        raise ValueError("DCA taps must be finite")
    return vector


def _validate_dca_main_index(
    dca_main_index: int, dca_taps: tuple[float, ...]
) -> None:
    if (
        type(dca_main_index) is not int
        or dca_main_index < 0
        or dca_main_index >= len(dca_taps)
    ):
        raise ValueError("dca_main_index must identify a DCA tap")


def _discarded_l1_ratio(
    full_vector: tuple[float, ...],
    discarded_left: tuple[float, ...],
    discarded_right: tuple[float, ...],
) -> float:
    full_l1 = sum(abs(value) for value in full_vector)
    if full_l1 == 0:
        return 0.0
    discarded_l1 = sum(abs(value) for value in discarded_left + discarded_right)
    return discarded_l1 / full_l1


def _crop_around_main(
    full_vector: tuple[float, ...], main_index: int
) -> tuple[
    tuple[float, ...],
    int,
    int,
    tuple[float, ...],
    tuple[float, ...],
]:
    desired_start = main_index - FIR_MAIN_INDEX
    desired_stop = main_index + FIR_MAIN_INDEX + 1
    source_start = max(0, desired_start)
    source_stop = min(len(full_vector), desired_stop)
    padded_left_count = max(0, -desired_start)
    padded_right_count = max(0, desired_stop - len(full_vector))
    retained = (
        (0.0,) * padded_left_count
        + full_vector[source_start:source_stop]
        + (0.0,) * padded_right_count
    )
    discarded_left = full_vector[:source_start]
    discarded_right = full_vector[source_stop:]
    return (
        retained,
        padded_left_count,
        padded_right_count,
        discarded_left,
        discarded_right,
    )


def _scaled_and_rounded(
    retained: tuple[float, ...], source_main: float, manual_main: int
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    scaled = tuple(value * manual_main / source_main for value in retained)
    taps = tuple(round_half_away_from_zero(value) for value in scaled)
    taps = taps[:FIR_MAIN_INDEX] + (manual_main,) + taps[FIR_MAIN_INDEX + 1 :]
    return scaled, taps


def _max_feasible_main(
    retained: tuple[float, ...], source_main: float
) -> int | None:
    feasible: int | None = None
    for candidate in range(1, FIR_L1_LIMIT + 1):
        _, taps = _scaled_and_rounded(retained, source_main, candidate)
        if sum(abs(tap) for tap in taps) <= FIR_L1_LIMIT:
            feasible = candidate
    return feasible


def _proposal_from_full_vector(
    *,
    mode: str,
    manual_main: int,
    current_taps: tuple[int, ...] | None,
    dca_taps: tuple[float, ...],
    dca_main_index: int,
    full_vector: tuple[float, ...],
    main_index: int,
) -> FirProposal:
    (
        retained,
        padded_left_count,
        padded_right_count,
        discarded_left,
        discarded_right,
    ) = _crop_around_main(full_vector, main_index)
    discarded_l1_ratio = _discarded_l1_ratio(
        full_vector, discarded_left, discarded_right
    )
    source_main = full_vector[main_index]

    if not isfinite(source_main) or source_main == 0:
        return FirProposal(
            mode=mode,
            manual_main=manual_main,
            current_taps=current_taps,
            dca_taps=dca_taps,
            dca_main_index=dca_main_index,
            combined_main_index=main_index,
            normalization_denominator=source_main,
            full_vector=full_vector,
            retained=retained,
            padded_left_count=padded_left_count,
            padded_right_count=padded_right_count,
            discarded_left=discarded_left,
            discarded_right=discarded_right,
            discarded_l1_ratio=discarded_l1_ratio,
            scaled=(),
            final_taps=(),
            l1=0,
            valid=False,
            reason="normalization Main must be finite and non-zero",
            max_feasible_main=None,
        )

    scaled, taps = _scaled_and_rounded(retained, source_main, manual_main)
    l1 = sum(abs(tap) for tap in taps)
    valid = l1 <= FIR_L1_LIMIT
    return FirProposal(
        mode=mode,
        manual_main=manual_main,
        current_taps=current_taps,
        dca_taps=dca_taps,
        dca_main_index=dca_main_index,
        combined_main_index=main_index,
        normalization_denominator=source_main,
        full_vector=full_vector,
        retained=retained,
        padded_left_count=padded_left_count,
        padded_right_count=padded_right_count,
        discarded_left=discarded_left,
        discarded_right=discarded_right,
        discarded_l1_ratio=discarded_l1_ratio,
        scaled=scaled,
        final_taps=taps,
        l1=l1,
        valid=valid,
        reason=None if valid else f"tap L1 sum {l1} exceeds limit {FIR_L1_LIMIT}",
        max_feasible_main=(
            None if valid else _max_feasible_main(retained, source_main)
        ),
    )


def calculate_first_pass(
    dca_taps: Sequence[float], dca_main_index: int, manual_main: int
) -> FirProposal:
    """Create a seven-tap FIR proposal from a complete DCA equalizer vector."""
    _validate_manual_main(manual_main)
    full_vector = _dca_vector(dca_taps)
    _validate_dca_main_index(dca_main_index, full_vector)
    return _proposal_from_full_vector(
        mode="first",
        manual_main=manual_main,
        current_taps=None,
        dca_taps=full_vector,
        dca_main_index=dca_main_index,
        full_vector=full_vector,
        main_index=dca_main_index,
    )


def _convolve(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, ...]:
    result = [0.0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return tuple(result)


def calculate_iterative_pass(
    current_taps: Sequence[int],
    dca_taps: Sequence[float],
    dca_main_index: int,
    manual_main: int,
) -> FirProposal:
    """Convolve the current FIR with all DCA taps, then crop around their Main."""
    _validate_manual_main(manual_main)
    current = tuple(current_taps)
    if len(current) != FIR_TAP_COUNT or any(type(tap) is not int for tap in current):
        raise ValueError("current_taps must contain exactly seven integers")
    dca_vector = _dca_vector(dca_taps)
    _validate_dca_main_index(dca_main_index, dca_vector)
    full_vector = _convolve(tuple(float(tap) for tap in current), dca_vector)
    return _proposal_from_full_vector(
        mode="iterative",
        manual_main=manual_main,
        current_taps=current,
        dca_taps=dca_vector,
        dca_main_index=dca_main_index,
        full_vector=full_vector,
        main_index=FIR_MAIN_INDEX + dca_main_index,
    )


def format_setfir_preview(
    port: str,
    channel: str | int,
    taps: Sequence[int],
    upper_nlc: int,
    lower_nlc: int,
) -> str:
    """Format, but never send, the vendor ``setFir`` command."""
    internal_taps = tuple(taps)
    if len(internal_taps) != FIR_TAP_COUNT:
        raise ValueError("setFir preview requires exactly seven taps")
    if any(type(tap) is not int for tap in internal_taps):
        raise TypeError("setFir taps must be integers")
    if type(upper_nlc) is not int or type(lower_nlc) is not int:
        raise TypeError("NLC values must be integers")
    pre3, pre2, pre1, main, post1, post2, post3 = internal_taps
    vendor_values = (
        pre1,
        pre2,
        pre3,
        main,
        post1,
        post2,
        post3,
        upper_nlc,
        lower_nlc,
    )
    values = " ".join(str(value) for value in vendor_values)
    return f"setFir {port} {channel} {values}"
