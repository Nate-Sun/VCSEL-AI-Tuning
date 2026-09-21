"""Pure, offline provider evaluation over validated V2 replay records."""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite

from pydantic import Field, FiniteFloat, StrictInt, model_validator

from .contracts import (
    BrainMode,
    DecisionAction,
    FrozenModel,
    Identifier,
    NonBlankText,
    NonnegativeFiniteFloat,
    Probability,
)
from .replay import ReplayRecord


class ProviderSummary(FrozenModel):
    """Offline aggregate for one provider identity and one brain mode."""

    provider_key: NonBlankText
    # Kept plural for compatibility with the approved V2 evaluation contract;
    # each summary nevertheless represents exactly one provider version.
    provider_versions: NonBlankText
    brain_mode: BrainMode
    sample_count: StrictInt = Field(ge=0)
    propose_count: StrictInt = Field(ge=0)
    hold_count: StrictInt = Field(ge=0)
    hold_rate: Probability
    p50_latency_ms: NonnegativeFiniteFloat
    p95_latency_ms: NonnegativeFiniteFloat
    repeated_output_agreement: Probability
    total_estimated_cost_usd: NonnegativeFiniteFloat | None = None

    @model_validator(mode="after")
    def validate_counts_and_percentiles(self) -> "ProviderSummary":
        if self.propose_count + self.hold_count > self.sample_count:
            raise ValueError("propose_count + hold_count must not exceed sample_count")
        if self.sample_count == 0:
            if any((self.propose_count, self.hold_count, self.hold_rate)):
                raise ValueError("empty summaries must have zero counts and hold_rate")
        elif self.hold_rate != self.hold_count / self.sample_count:
            raise ValueError("hold_rate must equal hold_count / sample_count")
        if self.p95_latency_ms < self.p50_latency_ms:
            raise ValueError("p95_latency_ms must be at least p50_latency_ms")
        return self


class ProviderComparison(FrozenModel):
    """Order-independent quality and decision comparison between two providers."""

    provider_a_key: NonBlankText
    provider_b_key: NonBlankText
    matched_sample_count: StrictInt = Field(ge=1)
    quality_score_mae: FiniteFloat = Field(ge=0.0, le=1.0)
    decision_agreement: Probability

    @model_validator(mode="after")
    def validate_different_providers(self) -> "ProviderComparison":
        if self.provider_a_key == self.provider_b_key:
            raise ValueError("provider keys must differ")
        return self


class RankingLabel(FrozenModel):
    """A human preference between two measurements."""

    better_measurement_id: Identifier
    worse_measurement_id: Identifier

    @model_validator(mode="after")
    def validate_distinct_measurements(self) -> "RankingLabel":
        if self.better_measurement_id == self.worse_measurement_id:
            raise ValueError("ranking labels must compare distinct measurements")
        return self


def _validated_model(model_type, value: object):
    """Revalidate dumped data to close Pydantic construct/copy escape hatches."""
    return model_type.model_validate(value.model_dump(round_trip=True), strict=True)


def _validated_records(
    records: tuple[ReplayRecord, ...], *, unique_measurement_ids: bool
) -> tuple[ReplayRecord, ...]:
    if type(records) is not tuple:
        raise TypeError("records must be an exact tuple")
    if not records:
        raise ValueError("records must not be empty")
    validated: list[ReplayRecord] = []
    for record in records:
        if type(record) is not ReplayRecord:
            raise TypeError("records must contain ReplayRecord instances")
        validated.append(_validated_model(ReplayRecord, record))
    result = tuple(validated)
    if unique_measurement_ids:
        measurement_ids = tuple(record.measurement_id for record in result)
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("records must have unique measurement IDs")
    return result


def _provider_identity(records: tuple[ReplayRecord, ...]) -> tuple[str, str, BrainMode]:
    first = records[0]
    identity = (
        first.brain_provider_key,
        first.brain_provider_version,
        first.result.brain_mode,
    )
    if any(
        (record.brain_provider_key, record.brain_provider_version, record.result.brain_mode)
        != identity
        for record in records[1:]
    ):
        raise ValueError("records must have one provider identity and brain mode")
    return identity


def _linear_percentile(sorted_values: tuple[float, ...], percentile: float) -> float:
    """Return the explicitly specified `(n - 1) * q` linear percentile."""
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _total_estimated_cost(costs: tuple[float | None, ...]) -> float | None:
    if any(cost is None for cost in costs):
        return None
    try:
        total = fsum(cost for cost in costs if cost is not None)
    except OverflowError as error:
        raise ValueError("total estimated cost must be finite") from error
    if not isfinite(total):
        raise ValueError("total estimated cost must be finite")
    return total


def summarize_provider(records: tuple[ReplayRecord, ...]) -> ProviderSummary:
    """Summarize one offline provider run, allowing repeated measurement reruns."""
    validated = _validated_records(records, unique_measurement_ids=False)
    provider_key, provider_version, brain_mode = _provider_identity(validated)
    actions = tuple(record.result.decision.action for record in validated)
    latencies = tuple(sorted(record.result.latency_ms for record in validated))
    by_measurement: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in validated:
        by_measurement[record.measurement_id].append(record)
    agreements: list[float] = []
    for grouped in by_measurement.values():
        if len(grouped) >= 2:
            first_output = (grouped[0].result.perception, grouped[0].result.decision)
            agreements.append(
                1.0
                if all(
                    (record.result.perception, record.result.decision) == first_output
                    for record in grouped[1:]
                )
                else 0.0
            )
    costs = tuple(record.estimated_cost_usd for record in validated)
    return ProviderSummary(
        provider_key=provider_key,
        provider_versions=provider_version,
        brain_mode=brain_mode,
        sample_count=len(validated),
        propose_count=sum(action is DecisionAction.PROPOSE for action in actions),
        hold_count=sum(action is DecisionAction.HOLD for action in actions),
        hold_rate=sum(action is DecisionAction.HOLD for action in actions) / len(validated),
        p50_latency_ms=_linear_percentile(latencies, 0.5),
        p95_latency_ms=_linear_percentile(latencies, 0.95),
        repeated_output_agreement=fsum(agreements) / len(agreements) if agreements else 1.0,
        total_estimated_cost_usd=_total_estimated_cost(costs),
    )


def compare_providers(
    provider_a_records: tuple[ReplayRecord, ...], provider_b_records: tuple[ReplayRecord, ...]
) -> ProviderComparison:
    """Compare two distinct providers against their same measurement-ID set."""
    left = _validated_records(provider_a_records, unique_measurement_ids=True)
    right = _validated_records(provider_b_records, unique_measurement_ids=True)
    left_key, _, _ = _provider_identity(left)
    right_key, _, _ = _provider_identity(right)
    if left_key == right_key:
        raise ValueError("comparison requires different provider keys")
    left_by_id = {record.measurement_id: record for record in left}
    right_by_id = {record.measurement_id: record for record in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("providers must have identical measurement-ID sets")
    ordered_ids = tuple(sorted(left_by_id))
    quality_errors = tuple(
        abs(
            left_by_id[measurement_id].result.perception.quality_score
            - right_by_id[measurement_id].result.perception.quality_score
        )
        for measurement_id in ordered_ids
    )
    decisions_match = tuple(
        (
            left_by_id[measurement_id].result.decision.action,
            left_by_id[measurement_id].result.decision.candidate_id,
        )
        == (
            right_by_id[measurement_id].result.decision.action,
            right_by_id[measurement_id].result.decision.candidate_id,
        )
        for measurement_id in ordered_ids
    )
    return ProviderComparison(
        provider_a_key=left_key,
        provider_b_key=right_key,
        matched_sample_count=len(ordered_ids),
        quality_score_mae=fsum(quality_errors) / len(ordered_ids),
        decision_agreement=fsum(decisions_match) / len(ordered_ids),
    )


def _validated_labels(labels: tuple[RankingLabel, ...]) -> tuple[RankingLabel, ...]:
    if type(labels) is not tuple:
        raise TypeError("labels must be an exact tuple")
    if not labels:
        raise ValueError("labels must not be empty")
    validated: list[RankingLabel] = []
    seen_pairs: set[frozenset[str]] = set()
    for label in labels:
        if type(label) is not RankingLabel:
            raise TypeError("labels must contain RankingLabel instances")
        verified = _validated_model(RankingLabel, label)
        pair = frozenset((verified.better_measurement_id, verified.worse_measurement_id))
        if pair in seen_pairs:
            raise ValueError("ranking label pairs must be unique regardless of direction")
        seen_pairs.add(pair)
        validated.append(verified)
    return tuple(validated)


def ranking_accuracy(
    records: tuple[ReplayRecord, ...], labels: tuple[RankingLabel, ...]
) -> float:
    """Return the fraction of human rankings matched by explicit quality scores."""
    validated_records = _validated_records(records, unique_measurement_ids=True)
    _provider_identity(validated_records)
    validated_labels = _validated_labels(labels)
    scores = {
        record.measurement_id: record.result.perception.quality_score
        for record in validated_records
    }
    correct = 0
    for label in validated_labels:
        try:
            is_correct = scores[label.better_measurement_id] > scores[label.worse_measurement_id]
        except KeyError as error:
            raise ValueError("ranking labels must reference known measurement IDs") from error
        correct += is_correct
    return correct / len(validated_labels)
