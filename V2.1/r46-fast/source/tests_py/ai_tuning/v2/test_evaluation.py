from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.v2.contracts import BrainMode, BrainResult, DecisionAction, DecisionProposal, PerceptionResult
from vcsel_ai_tuning.v2.evaluation import (
    ProviderComparison,
    ProviderSummary,
    RankingLabel,
    compare_providers,
    ranking_accuracy,
    summarize_provider,
)
from vcsel_ai_tuning.v2.replay import ReplayRecord


def _record(
    measurement_id: str,
    *,
    provider: str = "provider-a",
    version: str = "1",
    mode: BrainMode = BrainMode.PURE_API,
    action: DecisionAction = DecisionAction.PROPOSE,
    quality: float = 0.5,
    candidate_id: str | None = "candidate-a",
    latency: float = 1.0,
    cost: float | None = 0.01,
    request_hash: str = "a" * 64,
    response_hash: str = "b" * 64,
) -> ReplayRecord:
    if action is not DecisionAction.PROPOSE:
        candidate_id = None
    perception = PerceptionResult(
        measurement_id=measurement_id,
        quality_score=quality,
        confidence=0.9,
        valid=True,
        ood=False,
        defects={},
        provider_id=provider,
        provider_version=version,
        input_hash="c" * 64,
    )
    decision = DecisionProposal(
        based_on_measurement_id=measurement_id,
        action=action,
        candidate_id=candidate_id,
        requested_params={"bias": 1.0} if action is DecisionAction.PROPOSE else None,
        reason_codes=("TEST",),
        provider_id=provider,
        provider_version=version,
    )
    return ReplayRecord(
        sequence=97,
        campaign_id="campaign-a",
        sweep_point_id=f"point-{measurement_id}",
        measurement_id=measurement_id,
        brain_provider_key=provider,
        brain_provider_version=version,
        result=BrainResult(
            perception=perception,
            decision=decision,
            brain_mode=mode,
            latency_ms=latency,
            request_hash=request_hash,
            response_hash=response_hash,
        ),
        estimated_cost_usd=cost,
    )


def _records(*items: ReplayRecord) -> tuple[ReplayRecord, ...]:
    return tuple(items)


def test_provider_summary_uses_the_planned_latency_field_names():
    summary = ProviderSummary(
        provider_key="provider-a",
        provider_versions="1",
        brain_mode=BrainMode.PURE_API,
        sample_count=1,
        propose_count=1,
        hold_count=0,
        hold_rate=0.0,
        p50_latency_ms=1.0,
        p95_latency_ms=1.0,
        repeated_output_agreement=1.0,
    )

    assert set(summary.model_dump()) >= {"p50_latency_ms", "p95_latency_ms"}
    assert "latency_ms_p50" not in summary.model_dump()
    assert "latency_ms_p95" not in summary.model_dump()


def test_summarize_provider_uses_linear_percentiles_cost_and_action_counts():
    records = _records(
        _record("m-1", action=DecisionAction.PROPOSE, latency=1.0),
        _record("m-2", action=DecisionAction.HOLD, latency=2.0),
        _record("m-3", action=DecisionAction.STOP, latency=3.0),
        _record("m-4", action=DecisionAction.REQUEST_HUMAN, latency=4.0),
    )

    summary = summarize_provider(records)

    assert summary.provider_key == "provider-a"
    assert summary.provider_versions == "1"
    assert summary.sample_count == 4
    assert summary.propose_count == summary.hold_count == 1
    assert summary.hold_rate == 0.25
    assert summary.p50_latency_ms == 2.5
    assert summary.p95_latency_ms == pytest.approx(3.85)
    assert summary.total_estimated_cost_usd == pytest.approx(0.04)


def test_summarize_provider_handles_none_cost_and_repeated_output_semantics():
    same_again = _record("m-1", latency=77, request_hash="d" * 64, response_hash="e" * 64)
    different = _record("m-2", quality=0.8)
    records = _records(_record("m-1", cost=None), same_again, _record("m-2"), different)

    summary = summarize_provider(records)

    assert summary.total_estimated_cost_usd is None
    assert summary.repeated_output_agreement == 0.5
    assert summarize_provider(_records(_record("unique"))).repeated_output_agreement == 1.0


def test_summarize_provider_rejects_an_unrepresentable_total_cost():
    records = _records(_record("m-1", cost=1e308), _record("m-2", cost=1e308))

    with pytest.raises(ValueError, match=r"total estimated cost|finite"):
        summarize_provider(records)


@pytest.mark.parametrize(
    "records",
    [
        (),
        [_record("m-1")],
        _records(_record("m-1"), _record("m-2", provider="provider-b")),
        _records(_record("m-1"), _record("m-2", version="2")),
        _records(_record("m-1"), _record("m-2", mode=BrainMode.LOCAL_VISION_API)),
    ],
)
def test_summarize_provider_rejects_wrong_container_or_mixed_identity(records):
    with pytest.raises((TypeError, ValueError)):
        summarize_provider(records)


def test_compare_providers_aligns_by_measurement_id_and_ignores_input_order():
    left = _records(_record("m-b", quality=0.4, candidate_id="candidate-2"), _record("m-a", quality=0.2))
    right = _records(
        _record("m-a", provider="provider-b", quality=0.3),
        _record("m-b", provider="provider-b", quality=0.6, candidate_id="candidate-3"),
    )

    comparison = compare_providers(left, right)
    reversed_comparison = compare_providers(tuple(reversed(left)), tuple(reversed(right)))

    assert comparison == reversed_comparison
    assert comparison.matched_sample_count == 2
    assert comparison.quality_score_mae == pytest.approx(0.15)
    assert comparison.decision_agreement == 0.5
    assert (comparison.provider_a_key, comparison.provider_b_key) == ("provider-a", "provider-b")


def test_compare_providers_allows_different_brain_modes_between_sides():
    comparison = compare_providers(
        _records(_record("m-1", mode=BrainMode.PURE_API)),
        _records(
            _record(
                "m-1",
                provider="provider-b",
                mode=BrainMode.LOCAL_VISION_API,
            )
        ),
    )

    assert comparison.matched_sample_count == 1
    assert (comparison.provider_a_key, comparison.provider_b_key) == (
        "provider-a",
        "provider-b",
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ((), _records(_record("m-1", provider="provider-b"))),
        ([_record("m-1")], _records(_record("m-1", provider="provider-b"))),
        (_records(_record("m-1"), _record("m-1")), _records(_record("m-1", provider="provider-b"), _record("m-2", provider="provider-b"))),
        (_records(_record("m-1")), _records(_record("m-2", provider="provider-b"))),
        (_records(_record("m-1")), _records(_record("m-1", version="2"))),
        (_records(_record("m-1"), _record("m-2", provider="provider-c")), _records(_record("m-1", provider="provider-b"), _record("m-2", provider="provider-b"))),
    ],
)
def test_compare_providers_rejects_invalid_pairs(left, right):
    with pytest.raises((TypeError, ValueError)):
        compare_providers(left, right)


def test_ranking_accuracy_uses_only_explicit_quality_scores():
    records = _records(
        _record("a", quality=0.9),
        _record("b", quality=0.4),
        _record("c", quality=0.1),
    )
    labels = (
        RankingLabel(better_measurement_id="a", worse_measurement_id="b"),
        RankingLabel(better_measurement_id="b", worse_measurement_id="c"),
        RankingLabel(better_measurement_id="c", worse_measurement_id="a"),
    )

    assert ranking_accuracy(records, labels) == pytest.approx(2 / 3)
    tied = _records(_record("a", quality=0.5), _record("b", quality=0.5))
    assert ranking_accuracy(tied, (RankingLabel(better_measurement_id="a", worse_measurement_id="b"),)) == 0.0


@pytest.mark.parametrize(
    "labels",
    [
        (),
        [RankingLabel(better_measurement_id="a", worse_measurement_id="b")],
        (RankingLabel(better_measurement_id="a", worse_measurement_id="missing"),),
        (RankingLabel(better_measurement_id="a", worse_measurement_id="b"), RankingLabel(better_measurement_id="a", worse_measurement_id="b")),
        (RankingLabel(better_measurement_id="a", worse_measurement_id="b"), RankingLabel(better_measurement_id="b", worse_measurement_id="a")),
    ],
)
def test_ranking_accuracy_rejects_invalid_or_duplicate_labels(labels):
    records = _records(_record("a"), _record("b"))
    with pytest.raises((TypeError, ValueError)):
        ranking_accuracy(records, labels)


def test_evaluation_models_are_strict_frozen_and_cross_validate():
    summary = ProviderSummary(
        provider_key="provider-a",
        provider_versions="1",
        brain_mode=BrainMode.PURE_API,
        sample_count=0,
        propose_count=0,
        hold_count=0,
        hold_rate=0.0,
        p50_latency_ms=0.0,
        p95_latency_ms=0.0,
        repeated_output_agreement=1.0,
    )
    assert summary.sample_count == 0
    with pytest.raises(ValidationError):
        ProviderSummary.model_validate({**summary.model_dump(), "extra": 1}, strict=True)
    with pytest.raises(ValidationError):
        ProviderSummary.model_validate({**summary.model_dump(), "sample_count": True}, strict=True)
    with pytest.raises(ValidationError):
        ProviderSummary.model_validate({**summary.model_dump(), "p50_latency_ms": nan}, strict=True)
    with pytest.raises(ValidationError):
        ProviderSummary.model_validate({**summary.model_dump(), "p50_latency_ms": inf}, strict=True)
    with pytest.raises(ValidationError):
        ProviderSummary.model_validate({**summary.model_dump(), "sample_count": 1, "hold_rate": 0.5}, strict=True)
    with pytest.raises(ValidationError):
        ProviderSummary.model_validate({**summary.model_dump(), "p50_latency_ms": 2.0, "p95_latency_ms": 1.0}, strict=True)
    with pytest.raises(ValidationError):
        RankingLabel(better_measurement_id="same", worse_measurement_id="same")
    with pytest.raises(ValidationError):
        ProviderComparison(provider_a_key="same", provider_b_key="same", matched_sample_count=1, quality_score_mae=0.0, decision_agreement=1.0)
    with pytest.raises(ValidationError):
        summary.sample_count = 1


def test_public_functions_revalidate_copy_construct_and_nested_corruption():
    corrupt_result = BrainResult.model_construct(
        perception=_record("m-1").result.perception,
        decision=_record("m-1").result.decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=nan,
        request_hash="a" * 64,
        response_hash="b" * 64,
    )
    corrupt_record = _record("m-1").model_copy(update={"result": corrupt_result})
    corrupt_label = RankingLabel.model_construct(better_measurement_id="bad id", worse_measurement_id="m-1")

    with pytest.raises(ValidationError):
        summarize_provider((corrupt_record,))
    with pytest.raises(ValidationError):
        compare_providers((_record("m-1"),), (corrupt_record.model_copy(update={"brain_provider_key": "provider-b"}),))
    with pytest.raises(ValidationError):
        ranking_accuracy((_record("m-1"), _record("m-2")), (corrupt_label,))
