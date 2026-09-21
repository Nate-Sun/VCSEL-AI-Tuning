from __future__ import annotations

from collections.abc import Iterable
from math import inf, nan

import pytest
from pydantic import ValidationError

import vcsel_ai_tuning.v2.replay as replay_module
from vcsel_ai_tuning.optimization.protocol import OptimizationContext, TuningObservation
from vcsel_ai_tuning.v2 import (
    BrainCandidate,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    PerceptionResult,
    V2Observation,
)
from vcsel_ai_tuning.v2.brain import ComposedBrain, OptimizationBrain, _request_hash, _response_hash
from vcsel_ai_tuning.v2.providers import (
    FakeEyeScorer,
    FirstCandidateDecisionProvider,
    ScriptedFakeBrain,
)
from vcsel_ai_tuning.v2.replay import (
    MAX_REPLAY_CANDIDATES,
    MAX_REPLAY_CASES,
    MAX_REPLAY_HISTORY,
    ReplayCase,
    ReplayRecord,
    replay,
    write_replay_jsonl,
)


def _result(
    sample,
    candidate: BrainCandidate | None = None,
    *,
    history=(),
    safe_candidates=None,
) -> BrainResult:
    perception = PerceptionResult(
        measurement_id=sample.measurement_id,
        quality_score=0.9,
        confidence=0.9,
        valid=True,
        ood=False,
        defects={},
        provider_id="scripted-provider",
        provider_version="1",
        input_hash=sample.raw_eye_density_ref.sha256,
    )
    decision = DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction.HOLD if candidate is None else DecisionAction.PROPOSE,
        candidate_id=None if candidate is None else candidate.candidate_id,
        requested_params=None if candidate is None else candidate.requested_params,
        reason_codes=("SCRIPTED_HOLD",) if candidate is None else ("SCRIPTED_SELECTION",),
        provider_id="scripted-provider",
        provider_version="1",
    )
    mode = BrainMode.LOCAL_VISION_API
    candidates = (candidate,) if safe_candidates is None and candidate is not None else (
        () if safe_candidates is None else safe_candidates
    )
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=mode,
        latency_ms=0.0,
        request_hash=_request_hash(sample, history, candidates),
        response_hash=_response_hash(perception, decision, mode),
    )


def _case(eye_sample, **changes: object) -> ReplayCase:
    payload = {"sample": eye_sample, "history": (), "safe_candidates": (), "estimated_cost_usd": None}
    payload.update(changes)
    return ReplayCase(**payload)


def _next_sample(eye_sample, suffix: str):
    return eye_sample.model_copy(
        update={
            "measurement_id": f"measurement-{suffix}",
            "sweep_point_id": f"point-{suffix}",
        }
    )


def _records(eye_sample) -> tuple[ReplayRecord, ...]:
    return (
        ReplayRecord(
            sequence=0,
            campaign_id=eye_sample.campaign_id,
            sweep_point_id=eye_sample.sweep_point_id,
            measurement_id=eye_sample.measurement_id,
            brain_provider_key="test-brain",
            brain_provider_version="1",
            result=_result(eye_sample),
        ),
    )


class CountingBrain:
    provider_key = "counting-brain"
    provider_version = "foundation-1"

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    def analyze(self, sample, history, safe_candidates):
        self.calls.append(sample.measurement_id)
        return self.result


class ChangingIdentityBrain:
    provider_version = "foundation-1"

    def __init__(self, results: Iterable[BrainResult]) -> None:
        self._results = iter(results)
        self.identity_reads = 0
        self.calls: list[str] = []

    @property
    def provider_key(self) -> str:
        self.identity_reads += 1
        return "changing-brain" if self.identity_reads == 1 else "other-brain"

    def analyze(self, sample, history, safe_candidates):
        self.calls.append(sample.measurement_id)
        return next(self._results)


class ExplodingIdentityBrain:
    provider_version = "foundation-1"

    @property
    def provider_key(self) -> str:
        raise RuntimeError("identity unavailable")

    def analyze(self, sample, history, safe_candidates):
        raise AssertionError("analyze must not run")


class IdentityStringSubclass(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class DescriptorAnalyzeBrain:
    provider_key = "descriptor-brain"
    provider_version = "foundation-1"

    def __init__(self) -> None:
        self.events: list[str] = []

    @property
    def analyze(self):
        self.events.append("analyze descriptor read")

        def analyze(sample, history, safe_candidates):
            self.events.append(f"analyze call {sample.measurement_id}")
            return _result(sample)

        return analyze


class RecordingBrain:
    provider_key = "recording-brain"
    provider_version = "foundation-1"

    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def analyze(self, sample, history, safe_candidates):
        self.calls.append((sample, history, safe_candidates))
        return _result(
            sample,
            safe_candidates[0] if safe_candidates else None,
            history=history,
            safe_candidates=safe_candidates,
        )


class MutatingReplayBrain:
    provider_key = "mutating-replay"
    provider_version = "1"

    def analyze(self, sample, history, safe_candidates):
        object.__setattr__(safe_candidates[0], "candidate_id", "outside")
        object.__setattr__(safe_candidates[0], "requested_params", {"bias_ma": 999.0})
        return _result(sample, safe_candidates[0], history=history, safe_candidates=safe_candidates)


def _history_observation(eye_sample) -> V2Observation:
    history_id = "history-measurement-001"
    return V2Observation(
        measurement_id=history_id,
        tuning_observation=TuningObservation(
            observation_id="history-observation-001",
            candidate_id="candidate-001",
            context=OptimizationContext(
                driver_profile_hash="d" * 64,
                vcsel_model="model-001",
                device_id="device-001",
                test_context_id="context-001",
            ),
            settings_requested={"bias_ma": 6.0},
            settings_readback={"bias_ma": 6.0},
            tdecq_db=1.0,
            tdecq_status="CORR",
            er_db=4.0,
            er_valid=True,
        ),
        perception=PerceptionResult(
            measurement_id=history_id,
            quality_score=0.8,
            confidence=0.8,
            valid=True,
            ood=False,
            defects={},
            provider_id="history-provider",
            provider_version="1",
            input_hash=eye_sample.raw_eye_density_ref.sha256,
        ),
    )


def test_fake_eye_scorer_is_deterministic_and_binds_the_raw_density_hash(eye_sample):
    scorer = FakeEyeScorer(quality_score=0.75, confidence=0.95, valid=True, ood=False)

    result = scorer.score(eye_sample)

    assert scorer.calls == [eye_sample.measurement_id]
    assert result.measurement_id == eye_sample.measurement_id
    assert result.input_hash == eye_sample.raw_eye_density_ref.sha256
    assert result.provider_id == "fake-eye-scorer"
    assert result.provider_version == "foundation-1"
    assert dict(result.defects) == {}


@pytest.mark.parametrize("keyword,value", [("quality_score", True), ("confidence", False), ("quality_score", nan), ("confidence", inf), ("quality_score", -0.1), ("confidence", 1.1)])
def test_fake_eye_scorer_rejects_non_boolean_or_non_probability_configuration(keyword, value):
    kwargs = {keyword: value}
    with pytest.raises((TypeError, ValueError)):
        FakeEyeScorer(**kwargs)


@pytest.mark.parametrize("keyword,value", [("valid", 1), ("ood", "false")])
def test_fake_eye_scorer_requires_strict_boolean_flags(keyword, value):
    with pytest.raises(TypeError):
        FakeEyeScorer(**{keyword: value})


def test_first_candidate_provider_selects_the_first_candidate_and_holds_when_empty(
    eye_sample, brain_candidates
):
    provider = FirstCandidateDecisionProvider()
    perception = FakeEyeScorer().score(eye_sample)

    selected = provider.propose(eye_sample, perception, (), brain_candidates)
    held = provider.propose(eye_sample, perception, (), ())

    assert provider.calls == [eye_sample.measurement_id, eye_sample.measurement_id]
    assert selected.action is DecisionAction.PROPOSE
    assert selected.candidate_id == brain_candidates[0].candidate_id
    assert selected.requested_params == brain_candidates[0].requested_params
    assert selected.reason_codes == ("DETERMINISTIC_FAKE_SELECTION",)
    assert held.action is DecisionAction.HOLD
    assert held.reason_codes == ("NO_SAFE_CANDIDATE",)
    assert held.candidate_id is held.requested_params is None


def test_scripted_fake_brain_consumes_its_script_and_never_falls_back(eye_sample):
    brain = ScriptedFakeBrain((_result(eye_sample),))

    assert brain.analyze(eye_sample, (), ()) == _result(eye_sample)
    with pytest.raises(AssertionError, match="exhausted"):
        brain.analyze(eye_sample, (), ())
    assert brain.calls == [eye_sample.measurement_id, eye_sample.measurement_id]


def test_composed_and_scripted_brains_share_replay_contract(eye_sample, brain_candidates):
    second = _next_sample(eye_sample, "002")
    cases = (_case(eye_sample, safe_candidates=brain_candidates), _case(second, safe_candidates=brain_candidates))
    composed = ComposedBrain(
        BrainMode.LOCAL_VISION_API,
        FakeEyeScorer(),
        FirstCandidateDecisionProvider(),
        0.7,
    )
    scripted = ScriptedFakeBrain((
        _result(eye_sample, brain_candidates[0], safe_candidates=brain_candidates),
        _result(second, brain_candidates[0], safe_candidates=brain_candidates),
    ))

    composed_records = replay(composed, cases)
    scripted_records = replay(scripted, cases)

    assert isinstance(composed, OptimizationBrain)
    assert isinstance(scripted, OptimizationBrain)
    assert [item.measurement_id for item in composed_records] == [item.measurement_id for item in scripted_records]
    assert all(item.result.brain_mode is BrainMode.LOCAL_VISION_API for item in composed_records)
    assert all(item.result.decision.based_on_measurement_id == item.measurement_id for item in scripted_records)


def test_replay_preserves_order_calls_each_case_once_and_snapshots_provider_identity(eye_sample):
    second = _next_sample(eye_sample, "002")
    brain = ChangingIdentityBrain((_result(eye_sample), _result(second)))

    records = replay(brain, (_case(eye_sample), _case(second)))

    assert brain.calls == [eye_sample.measurement_id, second.measurement_id]
    assert brain.identity_reads == 1
    assert [record.sequence for record in records] == [0, 1]
    assert {record.brain_provider_key for record in records} == {"changing-brain"}
    assert {record.brain_provider_version for record in records} == {"foundation-1"}


def test_replay_preserves_valid_history_and_nested_candidate_order_for_the_brain(
    eye_sample, brain_candidates
):
    history = (_history_observation(eye_sample),)
    case = _case(eye_sample, history=history, safe_candidates=brain_candidates)
    brain = RecordingBrain()

    replay(brain, (case,))

    received_sample, received_history, received_candidates = brain.calls[0]
    assert received_sample == eye_sample
    assert received_history == history
    assert received_candidates == brain_candidates
    assert [candidate.candidate_id for candidate in received_candidates] == [
        "candidate-001",
        "candidate-002",
    ]
    assert received_candidates[0].requested_params == {
        "bias_ma": 6.0,
        "equalizer": {"tap_1": 1.0},
    }


def test_replay_rejects_bad_cases_before_calling_the_brain(eye_sample):
    brain = CountingBrain(_result(eye_sample))

    with pytest.raises(TypeError, match="ReplayCase"):
        replay(brain, (_case(eye_sample), object()))

    assert brain.calls == []


def test_replay_prevalidates_every_case_before_reading_analyze_descriptor(eye_sample):
    brain = DescriptorAnalyzeBrain()

    with pytest.raises(TypeError, match="ReplayCase"):
        replay(brain, (_case(eye_sample), object()))

    assert brain.events == []


@pytest.mark.parametrize("cases", [(), [], (object(),)])
def test_replay_requires_a_nonempty_tuple_of_cases(eye_sample, cases):
    brain = CountingBrain(_result(eye_sample))
    with pytest.raises((TypeError, ValueError)):
        replay(brain, cases)
    assert brain.calls == []


def test_replay_rejects_too_many_cases_before_calling_the_brain(eye_sample):
    brain = CountingBrain(_result(eye_sample))
    case = _case(eye_sample)

    with pytest.raises(ValueError, match="1.."):
        replay(brain, (case,) * (MAX_REPLAY_CASES + 1))

    assert brain.calls == []


def test_replay_rejects_duplicate_measurement_ids_before_calling_the_brain(eye_sample):
    brain = CountingBrain(_result(eye_sample))

    with pytest.raises(ValueError, match="measurement IDs must be unique"):
        replay(brain, (_case(eye_sample), _case(eye_sample)))

    assert brain.calls == []


def test_replay_revalidates_model_copy_cases_before_calling_the_brain(eye_sample, brain_candidates):
    brain = CountingBrain(_result(eye_sample))
    bypassed = _case(eye_sample).model_copy(
        update={"safe_candidates": (brain_candidates[0], brain_candidates[0])}
    )

    with pytest.raises(ValidationError, match="candidate IDs must be unique"):
        replay(brain, (bypassed,))

    assert brain.calls == []


def test_replay_rejects_model_construct_case_before_calling_the_brain(eye_sample, brain_candidates):
    brain = CountingBrain(_result(eye_sample))
    bypassed = ReplayCase.model_construct(
        sample=eye_sample,
        history=(),
        safe_candidates=(brain_candidates[0], brain_candidates[0]),
        estimated_cost_usd=0.0,
    )

    with pytest.raises(ValidationError, match="candidate IDs must be unique"):
        replay(brain, (bypassed,))

    assert brain.calls == []


@pytest.mark.parametrize("cost", [nan, inf, -0.01])
def test_replay_models_reject_invalid_costs(cost, eye_sample):
    with pytest.raises(ValidationError):
        _case(eye_sample, estimated_cost_usd=cost)
    with pytest.raises(ValidationError):
        ReplayRecord(
            sequence=0,
            campaign_id=eye_sample.campaign_id,
            sweep_point_id=eye_sample.sweep_point_id,
            measurement_id=eye_sample.measurement_id,
            brain_provider_key="brain",
            brain_provider_version="1",
            result=_result(eye_sample),
            estimated_cost_usd=cost,
        )


def test_replay_case_is_strict_forbids_extra_and_rejects_duplicate_candidate_ids(eye_sample, brain_candidates):
    with pytest.raises(ValidationError):
        ReplayCase(sample=eye_sample, unexpected=True)
    with pytest.raises(ValidationError):
        ReplayCase(sample=eye_sample, history=[])
    with pytest.raises(ValidationError, match="candidate IDs must be unique"):
        _case(eye_sample, safe_candidates=(brain_candidates[0], brain_candidates[0]))


def test_replay_case_bounds_are_enforced_without_large_object_creation(eye_sample, brain_candidates):
    with pytest.raises(ValidationError):
        _case(eye_sample, history=(object(),) * (MAX_REPLAY_HISTORY + 1))
    with pytest.raises(ValidationError):
        _case(eye_sample, safe_candidates=(brain_candidates[0],) * (MAX_REPLAY_CANDIDATES + 1))


def test_replay_rejects_invalid_or_unreadable_provider_identity_before_analyze(eye_sample):
    invalid = CountingBrain(_result(eye_sample))
    invalid.provider_key = " "
    with pytest.raises(TypeError, match="provider_key"):
        replay(invalid, (_case(eye_sample),))
    assert invalid.calls == []

    with pytest.raises(TypeError, match="provider identity"):
        replay(ExplodingIdentityBrain(), (_case(eye_sample),))

    invalid.provider_key = "one|two"
    assert replay(invalid, (_case(eye_sample),))[0].brain_provider_key == "one|two"

    invalid.provider_key = IdentityStringSubclass("claimed")
    with pytest.raises(TypeError, match="provider_key"):
        replay(invalid, (_case(eye_sample),))
    invalid.provider_key = "bad\u200fidentity"
    with pytest.raises(TypeError, match="provider_key"):
        replay(invalid, (_case(eye_sample),))


def test_replay_rejects_non_brain_and_wrong_result_type(eye_sample):
    with pytest.raises(TypeError, match="OptimizationBrain"):
        replay(object(), (_case(eye_sample),))
    brain = CountingBrain(object())
    with pytest.raises(TypeError, match="BrainResult"):
        replay(brain, (_case(eye_sample),))
    assert brain.calls == [eye_sample.measurement_id]


def test_replay_rejects_result_with_foreign_measurement_id(eye_sample):
    foreign = _next_sample(eye_sample, "foreign")
    brain = CountingBrain(_result(foreign))

    with pytest.raises(ValueError, match="measurement ID mismatch"):
        replay(brain, (_case(eye_sample),))


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("outside", "candidate not in input set"),
        ("params", "candidate not in input set"),
        ("input", "input hash mismatch"),
        ("request", "request hash mismatch"),
        ("response", "response hash mismatch"),
    ],
)
def test_replay_rejects_adversarial_results_without_creating_records(
    attack, message, eye_sample, brain_candidates
):
    good = _result(eye_sample, brain_candidates[0], safe_candidates=brain_candidates)
    perception, decision = good.perception, good.decision
    if attack == "outside":
        decision = decision.model_copy(update={"candidate_id": "foreign"})
    elif attack == "params":
        decision = decision.model_copy(update={"requested_params": {"bias_ma": 99.0}})
    elif attack == "input":
        perception = perception.model_copy(update={"input_hash": "b" * 64})
    result = good.model_copy(update={"perception": perception, "decision": decision})
    result = result.model_copy(update={
        "response_hash": _response_hash(perception, decision, result.brain_mode)
    })
    if attack == "request":
        result = result.model_copy(update={"request_hash": "b" * 64})
    if attack == "response":
        result = result.model_copy(update={"response_hash": "c" * 64})
    brain = CountingBrain(result)

    with pytest.raises(ValueError, match=message):
        replay(brain, (_case(eye_sample, safe_candidates=brain_candidates),))
    assert brain.calls == [eye_sample.measurement_id]


def test_replay_keeps_low_confidence_scripted_fact_when_hashes_and_candidates_match(
    eye_sample, brain_candidates
):
    good = _result(eye_sample, brain_candidates[0], safe_candidates=brain_candidates)
    perception = good.perception.model_copy(update={"confidence": 0.1})
    result = good.model_copy(update={
        "perception": perception,
        "response_hash": _response_hash(perception, good.decision, good.brain_mode),
    })

    records = replay(
        ScriptedFakeBrain((result,)),
        (_case(eye_sample, safe_candidates=brain_candidates),),
    )

    assert records[0].result.perception.confidence == 0.1
    assert records[0].result.decision.action is DecisionAction.PROPOSE


def test_replay_uses_unexposed_case_snapshot_against_brain_mutation(eye_sample, brain_candidates):
    original_candidate = brain_candidates[0].model_dump(round_trip=True)

    with pytest.raises(ValueError):
        replay(
            MutatingReplayBrain(),
            (_case(eye_sample, safe_candidates=brain_candidates),),
        )

    assert brain_candidates[0].model_dump(round_trip=True) == original_candidate


@pytest.mark.parametrize(("candidate_value", "decision_value"), [(1, 1.0), (1.0, 1)])
def test_replay_rejects_json_distinct_numeric_candidate_parameters(
    candidate_value, decision_value, eye_sample
):
    candidate = BrainCandidate(candidate_id="numeric", requested_params={"bias_ma": candidate_value})
    good = _result(eye_sample, candidate, safe_candidates=(candidate,))
    decision = good.decision.model_copy(update={"requested_params": {"bias_ma": decision_value}})
    result = good.model_copy(update={
        "decision": decision,
        "response_hash": _response_hash(good.perception, decision, good.brain_mode),
    })

    with pytest.raises(ValueError, match="candidate not in input set"):
        replay(CountingBrain(result), (_case(eye_sample, safe_candidates=(candidate,)),))


def test_replay_revalidates_model_copy_result_before_recording(eye_sample):
    result = _result(eye_sample)
    foreign = _next_sample(eye_sample, "foreign")
    bypassed = result.model_copy(
        update={"perception": result.perception.model_copy(update={"measurement_id": foreign.measurement_id})}
    )
    brain = CountingBrain(bypassed)

    with pytest.raises(ValidationError, match="perception.measurement_id"):
        replay(brain, (_case(eye_sample),))

    assert brain.calls == [eye_sample.measurement_id]


def test_replay_rejects_model_construct_result_before_recording(eye_sample):
    result = _result(eye_sample)
    foreign = _next_sample(eye_sample, "foreign")
    bypassed = BrainResult.model_construct(
        perception=result.perception.model_copy(
            update={"measurement_id": foreign.measurement_id}
        ),
        decision=result.decision,
        brain_mode=result.brain_mode,
        latency_ms=result.latency_ms,
        request_hash=result.request_hash,
        response_hash=result.response_hash,
    )
    brain = CountingBrain(bypassed)

    with pytest.raises(ValidationError, match="perception.measurement_id"):
        replay(brain, (_case(eye_sample),))

    assert brain.calls == [eye_sample.measurement_id]


def test_replay_propagates_script_exhaustion_without_fabricating_records(eye_sample):
    second = _next_sample(eye_sample, "002")
    brain = ScriptedFakeBrain((_result(eye_sample),))

    with pytest.raises(AssertionError, match="exhausted"):
        replay(brain, (_case(eye_sample), _case(second)))
    assert brain.calls == [eye_sample.measurement_id, second.measurement_id]


def test_replay_record_is_strict_and_binds_result_measurement(eye_sample):
    foreign = _next_sample(eye_sample, "foreign")
    with pytest.raises(ValidationError, match="measurement ID mismatch"):
        ReplayRecord(
            sequence=0,
            campaign_id=eye_sample.campaign_id,
            sweep_point_id=eye_sample.sweep_point_id,
            measurement_id=eye_sample.measurement_id,
            brain_provider_key="brain",
            brain_provider_version="1",
            result=_result(foreign),
        )
    with pytest.raises(ValidationError):
        ReplayRecord(**{**_records(eye_sample)[0].model_dump(), "extra": "forbidden"})


def test_write_replay_jsonl_round_trips_strict_models(eye_sample, tmp_path):
    path = write_replay_jsonl(_records(eye_sample), tmp_path / "v2-replay.jsonl")

    restored = tuple(
        ReplayRecord.model_validate_json(line, strict=True)
        for line in path.read_text(encoding="utf-8").splitlines()
    )

    assert path.exists()
    assert restored == _records(eye_sample)


def test_write_replay_jsonl_round_trips_nested_result_requested_params(
    eye_sample, brain_candidates, tmp_path
):
    record = _records(eye_sample)[0].model_copy(
        update={"result": _result(eye_sample, brain_candidates[0])}
    )
    path = write_replay_jsonl((record,), tmp_path / "nested.jsonl")

    restored = ReplayRecord.model_validate_json(path.read_text(encoding="utf-8"), strict=True)

    assert restored.result.decision.requested_params == brain_candidates[0].requested_params


def test_write_replay_jsonl_requires_a_nonempty_tuple_of_contiguous_unique_records(eye_sample, tmp_path):
    record = _records(eye_sample)[0]
    output = tmp_path / "protected.jsonl"
    output.write_text("original\n", encoding="utf-8")

    for records in ((), [record], (record.model_copy(update={"sequence": 1}),)):
        with pytest.raises((TypeError, ValueError)):
            write_replay_jsonl(records, output)
    duplicate = record.model_copy(update={"sequence": 1})
    with pytest.raises(ValueError, match="measurement IDs must be unique"):
        write_replay_jsonl((record, duplicate), output)
    with pytest.raises(TypeError, match="ReplayRecord"):
        write_replay_jsonl((record, object()), output)
    with pytest.raises(ValueError, match="1.."):
        write_replay_jsonl((record,) * (MAX_REPLAY_CASES + 1), output)

    assert output.read_text(encoding="utf-8") == "original\n"


def test_write_replay_jsonl_rejects_model_construct_record_without_replacing_output(
    eye_sample, tmp_path
):
    output = tmp_path / "protected.jsonl"
    output.write_text("original\n", encoding="utf-8")
    foreign = _next_sample(eye_sample, "foreign")
    bypassed = ReplayRecord.model_construct(
        **{
            **_records(eye_sample)[0].model_dump(),
            "result": _result(foreign),
        }
    )

    with pytest.raises(ValidationError, match="measurement ID mismatch"):
        write_replay_jsonl((bypassed,), output)

    assert output.read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize("failure", ["replace", "fsync"])
def test_write_replay_jsonl_write_failures_preserve_output_and_clean_temporary_files(
    eye_sample, tmp_path, monkeypatch, failure
):
    output = tmp_path / "protected.jsonl"
    output.write_text("original\n", encoding="utf-8")

    def raise_os_error(*_args, **_kwargs):
        raise OSError(f"{failure} failed")

    monkeypatch.setattr(replay_module.os, failure, raise_os_error)

    with pytest.raises(OSError, match=failure):
        write_replay_jsonl(_records(eye_sample), output)

    assert output.read_text(encoding="utf-8") == "original\n"
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))
