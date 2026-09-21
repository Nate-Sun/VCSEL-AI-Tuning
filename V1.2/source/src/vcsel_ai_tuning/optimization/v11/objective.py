from __future__ import annotations

from enum import StrEnum
from statistics import median

from .models import (
    AggregatedObservation,
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
)


class Comparison(StrEnum):
    BETTER = "BETTER"
    EQUAL = "EQUAL"
    WORSE = "WORSE"
    INVALID = "INVALID"


def compare_results(
    candidate: MetricResult,
    reference: MetricResult,
) -> Comparison:
    """Compare using raw TDECQ only; ER remains validation evidence."""
    if not isinstance(candidate, MetricResult) or not isinstance(
        reference,
        MetricResult,
    ):
        raise TypeError("candidate and reference must be MetricResult values")
    if not candidate.has_valid_tdecq:
        return Comparison.INVALID
    if not reference.has_valid_tdecq:
        return Comparison.BETTER

    assert candidate.tdecq_db is not None
    assert reference.tdecq_db is not None
    if candidate.tdecq_db < reference.tdecq_db:
        return Comparison.BETTER
    if candidate.tdecq_db > reference.tdecq_db:
        return Comparison.WORSE

    return Comparison.EQUAL


def best_observation(
    observations: tuple[I2CObservation, ...],
) -> I2CObservation | None:
    """Return the strict-objective best with ID as a deterministic final tie."""
    best: I2CObservation | None = None
    for observation in sorted(observations, key=lambda item: item.observation_id):
        if not isinstance(observation, I2CObservation):
            raise TypeError("observations must contain I2CObservation values")
        if not observation.result.has_valid_tdecq:
            continue
        if best is None:
            best = observation
            continue
        if compare_results(observation.result, best.result) is Comparison.BETTER:
            best = observation
    return best


def _group_key(
    observation: I2CObservation,
) -> tuple[str, tuple[tuple[str, int], ...]]:
    return (
        observation.fingerprint.model_dump_json(),
        tuple(sorted(observation.readback_i2c.items())),
    )


def aggregate_repeated_observations(
    observations: tuple[I2CObservation, ...],
) -> tuple[AggregatedObservation, ...]:
    """Median repeated observations keyed by exact fingerprint and readback."""
    groups: dict[
        tuple[str, tuple[tuple[str, int], ...]],
        list[I2CObservation],
    ] = {}
    for observation in observations:
        if not isinstance(observation, I2CObservation):
            raise TypeError("observations must contain I2CObservation values")
        groups.setdefault(_group_key(observation), []).append(observation)

    aggregated: list[AggregatedObservation] = []
    for key in sorted(groups):
        raw = tuple(sorted(groups[key], key=lambda item: item.observation_id))
        valid = tuple(item for item in raw if item.result.has_valid_tdecq)
        if valid:
            tdecq_values = tuple(
                float(item.result.tdecq_db)
                for item in valid
                if item.result.tdecq_db is not None
            )
            er_values = tuple(
                float(item.result.er_db)
                for item in valid
                if item.result.er_db is not None
            )
            result = MetricResult.valid(
                tdecq_db=float(median(tdecq_values)),
                er_db=float(median(er_values)) if er_values else None,
            )
        else:
            result = MetricResult.invalid(
                status="INVALID_AGGREGATE",
                reason="no valid TDECQ in repeated observations",
            )
        aggregated.append(
            AggregatedObservation(
                readback_i2c=dict(raw[0].readback_i2c),
                fingerprint=raw[0].fingerprint,
                result=result,
                raw_observations=raw,
            )
        )
    return tuple(aggregated)
