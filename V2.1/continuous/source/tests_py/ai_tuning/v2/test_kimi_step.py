from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    Candidate,
    CampaignState,
    FirSettings,
    StageAuthorization,
)
import vcsel_ai_tuning.v2.brain as brain_module
import vcsel_ai_tuning.v2.kimi_memory as kimi_memory_module
import vcsel_ai_tuning.v2.kimi_step as kimi_step_module
from vcsel_ai_tuning.v2.brain import MultimodalApiBrain
from vcsel_ai_tuning.v2.contracts import (
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    PerceptionResult,
)
from vcsel_ai_tuning.v2.kimi_memory import (
    KIMI_PROVIDER_ID,
    KimiCampaignMemory,
    KimiMemoryRevisionConflict,
)
from vcsel_ai_tuning.v2.kimi_step import execute_kimi_step, submit_kimi_step
from vcsel_ai_tuning.v2.factory import BrainFactory, BrainFactoryConfig
from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient
from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider


EMPTY_MEMORY_REVISION = sha256(b"").hexdigest()


def brain_result(
    eye_sample,
    *,
    action,
    candidate=None,
    perception_provider=KIMI_PROVIDER_ID,
    decision_provider=KIMI_PROVIDER_ID,
    memory_revision=EMPTY_MEMORY_REVISION,
    memory_id="d" * 64,
    reference_measurement_id=None,
    better_than_reference=None,
    perception_version="kimi-k3",
    decision_version="kimi-k3",
    proposed_params=None,
    memory=None,
    confidence=0.90,
    valid=True,
    ood=False,
):
    if memory is not None:
        memory_revision = memory.read_snapshot().revision
        memory_id = memory.memory_id
    perception = PerceptionResult(
        measurement_id=eye_sample.measurement_id,
        quality_score=0.75,
        confidence=confidence,
        valid=valid,
        ood=ood,
        defects={},
        provider_id=perception_provider,
        provider_version=perception_version,
        input_hash=eye_sample.raw_eye_density_ref.sha256,
        memory_revision=memory_revision,
        memory_id=memory_id,
        reference_measurement_id=reference_measurement_id,
        better_than_reference=better_than_reference,
    )
    decision = DecisionProposal(
        based_on_measurement_id=eye_sample.measurement_id,
        action=action,
        candidate_id=None if candidate is None else candidate.candidate_id,
        requested_params=(
            None
            if candidate is None
            else dict(candidate.requested_i2c)
            if proposed_params is None
            else proposed_params
        ),
        reason_codes=("TEST",),
        provider_id=decision_provider,
        provider_version=decision_version,
    )
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=1.0,
        request_hash="b" * 64,
        response_hash="c" * 64,
    )


class StaticBrain:
    def __init__(
        self,
        result,
        events=None,
        *,
        provider_key=KIMI_PROVIDER_ID,
        provider_version="kimi-k3",
    ):
        self.result = result
        self.events = events
        self.provider_key = provider_key
        self.provider_version = provider_version
        self.calls = 0
        self.safe_candidates = None

    def analyze(self, sample, history, safe_candidates):
        self.calls += 1
        self.safe_candidates = safe_candidates
        if self.events is not None:
            self.events.append("brain")
        return self.result


class ControlledFusedProvider:
    def __init__(
        self,
        result,
        events=None,
        *,
        provider_id=KIMI_PROVIDER_ID,
        provider_version="kimi-k3",
    ):
        self.result = result
        self.events = events
        self.provider_id = provider_id
        self.provider_version = provider_version

    def analyze_once(self, sample, history, safe_candidates):
        if self.events is not None:
            self.events.append("brain")
        return self.result.perception, self.result.decision


def factory_brain(
    result,
    events=None,
    *,
    provider_key=KIMI_PROVIDER_ID,
    provider_version="kimi-k3",
    minimum_confidence=0.0,
):
    provider = ControlledFusedProvider(
        result,
        events,
        provider_id=provider_key,
        provider_version=provider_version,
    )
    factory = BrainFactory()
    factory.register(
        BrainMode.PURE_API,
        lambda config: MultimodalApiBrain(provider, config.minimum_confidence),
    )
    return factory.create(
        BrainFactoryConfig(
            brain_mode=BrainMode.PURE_API,
            enabled_brain_modes=(BrainMode.PURE_API,),
            minimum_confidence=float(minimum_confidence),
        )
    )


def concrete_memory(
    tmp_path,
    eye_sample,
    *,
    name="kimi_memory.jsonl",
    campaign_id=None,
):
    return KimiCampaignMemory(
        tmp_path / name,
        campaign_id=campaign_id or eye_sample.campaign_id,
    )


class RecordingOrchestrator:
    def __init__(self, events=None, error=None, observations=(), memory=None):
        self.events = events
        self.error = error
        self.observations = observations
        self.memory = memory
        self.calls = 0
        self.candidate = None

    def run_one_candidate(self, authorization, candidate):
        if self.memory is not None:
            snapshot = KimiCampaignMemory.read_snapshot(self.memory)
            assert snapshot.entries
            assert snapshot.entries[-1].sample.campaign_id == authorization.campaign_id
            if self.events is not None:
                self.events.append("memory")
        if self.events is not None:
            self.events.append("runner")
        self.calls += 1
        self.candidate = candidate
        if self.error is not None:
            raise self.error
        return self.observations


class ImmediateRuntime:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.submissions = []

    def submit_stage(self, campaign_id, operation, work):
        self.submissions.append((campaign_id, operation))
        future = Future()
        try:
            future.set_result(work(SimpleNamespace(orchestrator=self.orchestrator)))
        except Exception as error:
            future.set_exception(error)
        return future


class DuckMemory:
    campaign_id = "campaign-001"
    memory_id = "d" * 64

    def __init__(self):
        self.calls = 0

    def append(self, sample, result):
        self.calls += 1


class SecretFailingSdk:
    def __init__(self, secret):
        def create(**request):
            raise RuntimeError(f"offline provider failure: {secret}")

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )


def two_candidates():
    return (
        Candidate(
            candidate_id="candidate-001",
            stage=CampaignState.I2C_COARSE,
            requested_i2c={"eq": 21},
            model_version="kimi-k3",
            reason="candidate menu",
        ),
        Candidate(
            candidate_id="candidate-002",
            stage=CampaignState.I2C_COARSE,
            requested_i2c={"eq": 22},
            model_version="kimi-k3",
            reason="candidate menu",
        ),
    )


@pytest.fixture
def stage_authorization(eye_sample) -> StageAuthorization:
    issued_at = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    return StageAuthorization(
        authorization_id="authorization-001",
        campaign_id=eye_sample.campaign_id,
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_COARSE,
        baseline_hash="e" * 64,
        rollback_hash="f" * 64,
        max_candidates=2,
        max_duration_s=60.0,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=10),
        rollback_snapshot=AppliedSettings(),
    )


def execute_args(
    tmp_path,
    eye_sample,
    *,
    brain,
    orchestrator,
    memory,
    authorization,
    candidates=None,
    campaign_id=None,
    output_path=None,
):
    return {
        "campaign_id": eye_sample.campaign_id if campaign_id is None else campaign_id,
        "brain": brain,
        "orchestrator": orchestrator,
        "memory": memory,
        "authorization": authorization,
        "sample": eye_sample,
        "candidates": two_candidates() if candidates is None else candidates,
        "instruction": "Improve the three-eye balance using I2C only",
        "output_path": output_path or tmp_path / "v21_result.json",
    }


def test_kimi_proposal_is_persisted_before_exact_local_candidate_execution(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    events = []
    campaign_candidates = two_candidates()
    memory = concrete_memory(
        tmp_path,
        eye_sample,
    )
    result_from_brain = brain_result(
        eye_sample,
        action=DecisionAction.PROPOSE,
        candidate=campaign_candidates[1],
        memory=memory,
    )
    brain = factory_brain(result_from_brain, events)
    orchestrator = RecordingOrchestrator(events, memory=memory)

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=brain,
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
            candidates=campaign_candidates,
        )
    )

    assert events == ["brain", "memory", "runner"]
    assert orchestrator.candidate is campaign_candidates[1]
    assert memory.load()[0].decision == result_from_brain.decision
    assert result["execution_status"] == "EXECUTED"
    assert result["reference_measurement_id"] is None
    assert result["better_than_reference"] is None
    assert result["memory_revision"] == EMPTY_MEMORY_REVISION
    assert result["memory_id"] == memory.memory_id
    assert result["model"] == "kimi-k3"
    assert result["perception_provider_id"] == KIMI_PROVIDER_ID
    assert result["perception_provider_version"] == "kimi-k3"
    assert result["decision_provider_id"] == KIMI_PROVIDER_ID
    assert result["decision_provider_version"] == "kimi-k3"
    assert result["requested_params"] == {"eq": 22}
    assert result["selected_local_params"] == {"eq": 22}
    assert result["proposed_params"] == {"eq": 22}
    assert json.loads((tmp_path / "v21_result.json").read_text(encoding="utf-8")) == result


def test_runner_is_blocked_when_persisted_assessment_cannot_be_reread_exactly(
    tmp_path,
    eye_sample,
    stage_authorization,
    monkeypatch,
):
    campaign_candidates = two_candidates()
    memory = concrete_memory(tmp_path, eye_sample)
    result_from_brain = brain_result(
        eye_sample,
        action=DecisionAction.PROPOSE,
        candidate=campaign_candidates[0],
        memory=memory,
    )
    orchestrator = RecordingOrchestrator()
    path_type = type(memory._path)
    original_read_bytes = path_type.read_bytes

    def corrupted_read_bytes(path):
        journal_bytes = original_read_bytes(path)
        if path == memory._path:
            corrupted = journal_bytes.replace(
                b'"quality_score":0.75',
                b'"quality_score":0.74',
                1,
            )
            assert corrupted != journal_bytes
            return corrupted
        return journal_bytes

    monkeypatch.setattr(path_type, "read_bytes", corrupted_read_bytes)

    with pytest.raises(ValueError, match="could not be verified"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(result_from_brain),
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
                candidates=campaign_candidates,
            )
        )

    assert orchestrator.calls == 0


@pytest.mark.parametrize(
    "action",
    [DecisionAction.HOLD, DecisionAction.STOP, DecisionAction.REQUEST_HUMAN],
)
def test_genuine_kimi_non_proposal_is_persisted_without_execution(
    tmp_path,
    eye_sample,
    action,
    stage_authorization,
):
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(eye_sample, action=action, memory=memory)
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == 1
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


def test_brain_guard_hold_is_not_persisted_or_executed(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.HOLD,
                    perception_provider="brain-guard",
                    decision_provider="brain-guard",
                    perception_version="1",
                    decision_version="1",
                    memory_revision=None,
                    memory_id=None,
                )
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert memory.load() == ()
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


@pytest.mark.parametrize(
    "perception_provider,decision_provider,expected_records",
    [
        ("not-kimi", KIMI_PROVIDER_ID, 0),
        (KIMI_PROVIDER_ID, "not-kimi", 1),
    ],
)
def test_factory_gate_records_only_kimi_perception_when_holding_non_kimi_proposal(
    tmp_path,
    eye_sample,
    perception_provider,
    decision_provider,
    expected_records,
    stage_authorization,
):
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.PROPOSE,
                    candidate=candidate,
                    perception_provider=perception_provider,
                    decision_provider=decision_provider,
                    memory=memory,
                )
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == expected_records
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


def test_factory_gate_records_kimi_perception_when_holding_unknown_candidate(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    unknown = Candidate(
        candidate_id="candidate-unknown",
        stage=CampaignState.I2C_COARSE,
        requested_i2c={"eq": 99},
        model_version="kimi-k3",
        reason="not in local menu",
    )
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.PROPOSE,
                    candidate=unknown,
                    memory=memory,
                )
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == 1
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


def test_memory_append_failure_blocks_execution(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    candidate = two_candidates()[0]
    orchestrator = RecordingOrchestrator()
    memory_path = tmp_path / "kimi_memory.jsonl"
    memory_path.mkdir()
    memory = KimiCampaignMemory(memory_path, campaign_id=eye_sample.campaign_id)

    with pytest.raises(ValueError, match="regular file"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory=memory,
                    )
                ),
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
            )
        )

    assert orchestrator.calls == 0


def test_stale_memory_revision_blocks_execution(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    candidate = two_candidates()[0]
    orchestrator = RecordingOrchestrator()
    memory = KimiCampaignMemory(
        tmp_path / "kimi_memory.jsonl",
        campaign_id=eye_sample.campaign_id,
    )

    with pytest.raises(KimiMemoryRevisionConflict, match="revision changed"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory_revision="d" * 64,
                        memory_id=memory.memory_id,
                    )
                ),
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
            )
        )

    assert orchestrator.calls == 0


def test_runner_failure_is_recorded_after_kimi_assessment_and_reraised(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    events = []
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator(
        events,
        error=RuntimeError("hardware failed"),
        memory=memory,
    )

    with pytest.raises(RuntimeError, match="hardware failed"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory=memory,
                    ),
                    events,
                ),
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
            )
        )

    document = json.loads((tmp_path / "v21_result.json").read_text(encoding="utf-8"))
    assert events == ["brain", "memory", "runner"]
    assert document["execution_status"] == "FAILED"
    assert document["observation_ids"] == []
    assert document["error"] == "CANDIDATE_EXECUTION_FAILED"
    assert "hardware failed" not in json.dumps(document)


def test_kimi_step_rejects_fir_candidates_before_brain_memory_or_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    events = []
    candidate = Candidate(
        candidate_id="fir-candidate-001",
        stage=CampaignState.FIR_APPLY,
        requested_fir=FirSettings(
            taps=(0, 9, -28, 104, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
        ),
        model_version="kimi-k3",
        reason="must not enter V2.1 I2C step",
    )
    brain = factory_brain(
        brain_result(
            eye_sample,
            action=DecisionAction.PROPOSE,
            candidate=candidate,
        ),
        events,
    )
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator(events)

    with pytest.raises(ValueError, match="I2C-only"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=brain,
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
                candidates=(candidate,),
            )
        )

    assert events == []
    assert memory.load() == ()
    assert orchestrator.calls == 0


def test_submit_kimi_step_forwards_exact_memory_through_runtime(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    events = []
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator(events, memory=memory)
    runtime = ImmediateRuntime(orchestrator)

    future = submit_kimi_step(
        runtime,
        campaign_id=eye_sample.campaign_id,
        brain=factory_brain(
            brain_result(
                eye_sample,
                action=DecisionAction.PROPOSE,
                candidate=candidate,
                memory=memory,
            ),
            events,
        ),
        memory=memory,
        authorization=stage_authorization,
        sample=eye_sample,
        candidates=(candidate,),
        instruction="I2C only",
        output_path=tmp_path / "v21_result.json",
    )

    assert future.result()["execution_status"] == "EXECUTED"
    assert runtime.submissions == [
        (eye_sample.campaign_id, f"KIMI_STEP:{eye_sample.measurement_id}")
    ]
    assert memory.load()[0].sample is not None
    assert events == ["brain", "memory", "runner"]


@pytest.mark.parametrize(
    "mismatch_target",
    ["campaign_id", "sample", "authorization", "memory"],
)
def test_execute_rejects_campaign_binding_mismatch_before_any_side_effect(
    tmp_path,
    eye_sample,
    stage_authorization,
    mismatch_target,
):
    events = []
    campaign_id = eye_sample.campaign_id
    sample = eye_sample
    authorization = stage_authorization
    memory_campaign_id = eye_sample.campaign_id
    if mismatch_target == "campaign_id":
        campaign_id = "campaign-other"
    elif mismatch_target == "sample":
        sample = eye_sample.model_copy(update={"campaign_id": "campaign-other"})
    elif mismatch_target == "authorization":
        authorization = stage_authorization.model_copy(
            update={"campaign_id": "campaign-other"}
        )
    else:
        memory_campaign_id = "campaign-other"
    candidate = two_candidates()[0]
    brain = factory_brain(
        brain_result(sample, action=DecisionAction.PROPOSE, candidate=candidate),
        events,
    )
    memory = concrete_memory(
        tmp_path,
        eye_sample,
        campaign_id=memory_campaign_id,
    )
    orchestrator = RecordingOrchestrator(events)

    with pytest.raises(ValueError, match="campaign_id"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                sample,
                campaign_id=campaign_id,
                brain=brain,
                orchestrator=orchestrator,
                memory=memory,
                authorization=authorization,
            )
        )

    assert events == []
    assert memory.load() == ()
    assert orchestrator.calls == 0


def test_submit_rejects_campaign_mismatch_before_runtime_submission(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    runtime = ImmediateRuntime(RecordingOrchestrator())
    mismatched_authorization = stage_authorization.model_copy(
        update={"campaign_id": "campaign-other"}
    )
    memory = concrete_memory(tmp_path, eye_sample)

    with pytest.raises(ValueError, match="campaign_id"):
        submit_kimi_step(
            runtime,
            campaign_id=eye_sample.campaign_id,
            brain=factory_brain(brain_result(eye_sample, action=DecisionAction.HOLD)),
            memory=memory,
            authorization=mismatched_authorization,
            sample=eye_sample,
            candidates=two_candidates(),
            instruction="I2C only",
            output_path=tmp_path / "v21_result.json",
        )

    assert runtime.submissions == []


def test_provider_memory_binding_blocks_different_empty_writer_before_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    candidate = two_candidates()[0]
    reader_memory = KimiCampaignMemory(
        tmp_path / "reader.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    writer_path = tmp_path / "writer.jsonl"
    writer_memory = KimiCampaignMemory(
        writer_path,
        campaign_id=eye_sample.campaign_id,
    )
    orchestrator = RecordingOrchestrator()

    with pytest.raises(ValueError, match="memory_id"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory=reader_memory,
                    )
                ),
                orchestrator=orchestrator,
                memory=writer_memory,
                authorization=stage_authorization,
            )
        )

    assert orchestrator.calls == 0
    assert not writer_path.exists()


def test_factory_gate_holds_mismatched_brain_identity_before_persistence(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.PROPOSE,
                    candidate=candidate,
                ),
                provider_key="not-kimi",
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert memory.load() == ()
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


@pytest.mark.parametrize(
    "perception_version,decision_version,brain_version,expected_records",
    [
        ("kimi-other", "kimi-k3", "kimi-k3", 0),
        ("kimi-k3", "kimi-other", "kimi-k3", 1),
        ("kimi-k3", "kimi-k3", "kimi-other", 0),
    ],
)
def test_factory_gate_records_only_kimi_perception_for_provider_version_hold(
    tmp_path,
    eye_sample,
    stage_authorization,
    perception_version,
    decision_version,
    brain_version,
    expected_records,
):
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.PROPOSE,
                    candidate=candidate,
                    perception_version=perception_version,
                    decision_version=decision_version,
                    memory=memory,
                ),
                provider_version=brain_version,
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == expected_records
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


def test_factory_gate_holds_proposal_param_mismatch(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.PROPOSE,
                    candidate=candidate,
                    proposed_params={"eq": 99},
                    memory=memory,
                )
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == 1
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


def test_low_confidence_kimi_perception_accepts_only_canonical_guard_hold(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()
    candidate = two_candidates()[0]
    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.PROPOSE,
                    candidate=candidate,
                    memory=memory,
                    confidence=0.10,
                ),
                minimum_confidence=0.80,
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert result["model"] == "kimi-k3"
    assert result["perception_provider_id"] == KIMI_PROVIDER_ID
    assert result["perception_provider_version"] == "kimi-k3"
    assert result["decision_provider_id"] == "brain-guard"
    assert result["decision_provider_version"] == "1"
    assert result["action"] == "HOLD"
    assert "LOW_CONFIDENCE" in result["reason_codes"]
    entries = memory.load()
    assert len(entries) == 1
    assert entries[0].perception.provider_id == KIMI_PROVIDER_ID
    assert entries[0].decision.provider_id == "brain-guard"
    assert memory.read_snapshot().best_entry is None
    assert orchestrator.calls == 0


def test_execute_rejects_duck_memory_before_brain_or_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    events = []
    memory = DuckMemory()
    brain = factory_brain(
        brain_result(
            eye_sample,
            action=DecisionAction.HOLD,
            perception_provider="brain-guard",
            decision_provider="brain-guard",
            memory_revision=None,
            memory_id=None,
        ),
        events,
    )
    orchestrator = RecordingOrchestrator(events)

    with pytest.raises(TypeError, match="KimiCampaignMemory"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=brain,
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
            )
        )

    assert events == []
    assert memory.calls == 0
    assert orchestrator.calls == 0


def test_submit_rejects_duck_memory_before_runtime_submission(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    runtime = ImmediateRuntime(RecordingOrchestrator())

    with pytest.raises(TypeError, match="KimiCampaignMemory"):
        submit_kimi_step(
            runtime,
            campaign_id=eye_sample.campaign_id,
            brain=factory_brain(brain_result(eye_sample, action=DecisionAction.HOLD)),
            memory=DuckMemory(),
            authorization=stage_authorization,
            sample=eye_sample,
            candidates=two_candidates(),
            instruction="I2C only",
            output_path=tmp_path / "v21_result.json",
        )

    assert runtime.submissions == []


def test_execute_rejects_direct_brain_before_memory_or_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    events = []
    memory = concrete_memory(tmp_path, eye_sample)
    direct_brain = StaticBrain(
        brain_result(
            eye_sample,
            action=DecisionAction.HOLD,
            memory=memory,
        ),
        events,
    )
    orchestrator = RecordingOrchestrator(events)

    with pytest.raises(TypeError, match="BrainFactory"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=direct_brain,
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
            )
        )

    assert events == []
    assert memory.load() == ()
    assert orchestrator.calls == 0


def test_submit_rejects_direct_brain_before_runtime_submission(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    memory = concrete_memory(tmp_path, eye_sample)
    runtime = ImmediateRuntime(RecordingOrchestrator())

    with pytest.raises(TypeError, match="BrainFactory"):
        submit_kimi_step(
            runtime,
            campaign_id=eye_sample.campaign_id,
            brain=StaticBrain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.HOLD,
                    memory=memory,
                )
            ),
            memory=memory,
            authorization=stage_authorization,
            sample=eye_sample,
            candidates=two_candidates(),
            instruction="I2C only",
            output_path=tmp_path / "v21_result.json",
        )

    assert runtime.submissions == []
    assert memory.load() == ()


def test_execute_rejects_memory_subclass_before_brain_or_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    class MemorySubclass(KimiCampaignMemory):
        pass

    events = []
    memory = MemorySubclass(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    brain = factory_brain(
        brain_result(eye_sample, action=DecisionAction.HOLD),
        events,
    )
    orchestrator = RecordingOrchestrator(events)

    with pytest.raises(TypeError, match="KimiCampaignMemory"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=brain,
                orchestrator=orchestrator,
                memory=memory,
                authorization=stage_authorization,
            )
        )

    assert events == []
    assert orchestrator.calls == 0


def test_low_level_memory_redirect_attempt_uses_original_journal_for_execution(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    original_path = tmp_path / "kimi_memory.jsonl"
    redirected_path = tmp_path / "redirected_memory.jsonl"
    memory = concrete_memory(tmp_path, eye_sample)
    candidate = two_candidates()[0]
    events = []
    brain = factory_brain(
        brain_result(
            eye_sample,
            action=DecisionAction.PROPOSE,
            candidate=candidate,
            memory=memory,
        ),
        events,
    )
    redirected_memory_id = KimiCampaignMemory(
        redirected_path, campaign_id=eye_sample.campaign_id
    ).memory_id
    identity = kimi_memory_module._get_kimi_memory_identity(memory)
    for attribute, replacement in (
        ("path", redirected_path.resolve()),
        ("campaign_id", "campaign-redirected"),
        ("memory_id", redirected_memory_id),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(identity, attribute, replacement)
    for attribute, replacement in (
        ("_path", redirected_path.resolve()),
        ("_memory_id", redirected_memory_id),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(memory, attribute, replacement)
    orchestrator = RecordingOrchestrator(events, memory=memory)

    document = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=brain,
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
            candidates=(candidate,),
        )
    )

    assert document["execution_status"] == "EXECUTED"
    assert events == ["brain", "memory", "runner"]
    assert orchestrator.calls == 1
    assert original_path.is_file()
    assert len(memory.load()) == 1
    assert not redirected_path.exists()


@pytest.mark.parametrize(
    "perception_updates,reason_code",
    [
        ({"valid": False}, "INVALID_EYE"),
        ({"ood": True}, "OOD"),
        ({"confidence": 0.10}, "LOW_CONFIDENCE"),
    ],
)
def test_real_factory_gate_persists_parsed_unsafe_perception_without_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
    perception_updates,
    reason_code,
):
    memory = concrete_memory(tmp_path, eye_sample)
    candidate = two_candidates()[0]
    orchestrator = RecordingOrchestrator()
    result = brain_result(
        eye_sample,
        action=DecisionAction.PROPOSE,
        candidate=candidate,
        memory=memory,
        **perception_updates,
    )

    brain = factory_brain(result, minimum_confidence=0.80)
    state = brain_module._get_factory_gate_state(brain)
    assert isinstance(state, tuple)
    assert state.built_in_profile == (
        KIMI_PROVIDER_ID,
        "kimi-k3",
        KIMI_PROVIDER_ID,
        "kimi-k3",
    )
    assert isinstance(state.built_in_profile, tuple)
    malicious_calls = 0

    def malicious(*args):
        nonlocal malicious_calls
        malicious_calls += 1
        return result

    for attribute, replacement in (
        ("analyze", malicious),
        ("_analyze_callable", malicious),
        ("_minimum_confidence", 0.0),
        ("_provider_key", "spoofed-provider"),
        ("_provider_version", "spoofed-version"),
        ("_mode", BrainMode.PURE_API),
        ("_built_in_profile", None),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(brain, attribute, replacement)
    for attribute, replacement in (
        ("mode", BrainMode.PURE_API),
        ("minimum_confidence", 0.0),
        ("provider_key", "spoofed-provider"),
        ("provider_version", "spoofed-version"),
        ("analyze_callable", malicious),
        ("native_builtin", False),
        ("built_in_profile", None),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(state, attribute, replacement)

    document = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=brain,
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
            candidates=(candidate,),
        )
    )

    assert document["action"] == "HOLD"
    assert reason_code in document["reason_codes"]
    entries = memory.load()
    assert len(entries) == 1
    assert entries[0].perception.provider_id == KIMI_PROVIDER_ID
    assert entries[0].decision.provider_id == "brain-guard"
    assert memory.read_snapshot().best_entry is None
    assert malicious_calls == 0
    assert orchestrator.calls == 0


def test_offline_kimi_sdk_failure_is_sanitized_through_real_factory_gate(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    secret = "moonshot-secret-api-key"
    relative_path = "eyes/offline-current.npy"
    density_path = tmp_path / relative_path
    density_path.parent.mkdir(parents=True)
    np.save(density_path, np.ones((16, 24), dtype=np.uint32))
    digest = sha256(density_path.read_bytes()).hexdigest()
    artifact = eye_sample.raw_eye_density_ref.model_copy(
        update={"relative_path": relative_path, "sha256": digest}
    )
    sample = eye_sample.model_copy(
        update={
            "raw_eye_density_ref": artifact,
            "artifact_hashes": {relative_path: digest},
        }
    )
    memory = concrete_memory(tmp_path, sample, name="offline_memory.jsonl")
    provider = KimiProvider(
        KimiApiClient(
            api_key=secret,
            sdk_client=SecretFailingSdk(secret),
        ),
        artifact_root=tmp_path,
        instruction="I2C only",
        memory=memory,
    )
    factory = BrainFactory()
    factory.register(
        BrainMode.PURE_API,
        lambda config: MultimodalApiBrain(provider, config.minimum_confidence),
    )
    brain = factory.create(
        BrainFactoryConfig(
            brain_mode=BrainMode.PURE_API,
            enabled_brain_modes=(BrainMode.PURE_API,),
            minimum_confidence=0.80,
        )
    )
    orchestrator = RecordingOrchestrator()

    document = execute_kimi_step(
        **execute_args(
            tmp_path,
            sample,
            brain=brain,
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
            candidates=two_candidates(),
            output_path=tmp_path / "offline_result.json",
        )
    )

    serialized = json.dumps(document)
    assert document["action"] == "HOLD"
    assert "PROVIDER_ERROR" in document["reason_codes"]
    assert document["error"] is None
    assert secret not in serialized
    assert memory.load() == ()
    assert orchestrator.calls == 0


@pytest.mark.parametrize(
    "brain_key,brain_version,perception_version,decision_version,expected_records",
    [
        ("not-kimi", "kimi-k3", "kimi-k3", "kimi-k3", 0),
        (KIMI_PROVIDER_ID, "kimi-other", "kimi-k3", "kimi-other", 0),
        (KIMI_PROVIDER_ID, "kimi-k3", "kimi-k3", "kimi-other", 1),
    ],
)
def test_factory_gate_sanitizes_mismatched_hold_identity(
    tmp_path,
    eye_sample,
    stage_authorization,
    brain_key,
    brain_version,
    perception_version,
    decision_version,
    expected_records,
):
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=DecisionAction.HOLD,
                    perception_version=perception_version,
                    decision_version=decision_version,
                    memory=memory,
                ),
                provider_key=brain_key,
                provider_version=brain_version,
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == expected_records
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


@pytest.mark.parametrize(
    "action,decision_provider,decision_version",
    [
        (DecisionAction.HOLD, "foreign-provider", "1"),
        (DecisionAction.STOP, "foreign-provider", "1"),
        (DecisionAction.STOP, "brain-guard", "1"),
        (DecisionAction.REQUEST_HUMAN, "brain-guard", "1"),
        (DecisionAction.HOLD, "brain-guard", "not-guard-1"),
    ],
)
def test_factory_gate_sanitizes_foreign_or_noncanonical_guard_decision(
    tmp_path,
    eye_sample,
    stage_authorization,
    action,
    decision_provider,
    decision_version,
):
    memory = concrete_memory(tmp_path, eye_sample)
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        **execute_args(
            tmp_path,
            eye_sample,
            brain=factory_brain(
                brain_result(
                    eye_sample,
                    action=action,
                    decision_provider=decision_provider,
                    decision_version=decision_version,
                    memory=memory,
                )
            ),
            orchestrator=orchestrator,
            memory=memory,
            authorization=stage_authorization,
        )
    )

    assert len(memory.load()) == 1
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"


def test_output_preflight_failure_blocks_memory_and_runner(
    tmp_path,
    eye_sample,
    stage_authorization,
):
    candidate = two_candidates()[0]
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")
    events = []
    memory = concrete_memory(tmp_path, eye_sample)

    with pytest.raises(OSError):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory=memory,
                    ),
                    events,
                ),
                orchestrator=RecordingOrchestrator(events),
                memory=memory,
                authorization=stage_authorization,
                output_path=blocked_parent / "v21_result.json",
            )
        )

    assert events == ["brain"]


def test_failed_result_publication_does_not_mask_runner_exception(
    tmp_path,
    eye_sample,
    stage_authorization,
    monkeypatch,
):
    candidate = two_candidates()[0]
    memory = concrete_memory(tmp_path, eye_sample)

    def fail_publication(*args, **kwargs):
        raise OSError("private publication detail")

    monkeypatch.setattr(kimi_step_module, "_write_result", fail_publication)

    with pytest.raises(RuntimeError, match="private runner detail"):
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory=memory,
                    )
                ),
                orchestrator=RecordingOrchestrator(
                    error=RuntimeError("private runner detail")
                ),
                memory=memory,
                authorization=stage_authorization,
            )
        )


def test_post_success_publication_failure_reports_hardware_executed(
    tmp_path,
    eye_sample,
    stage_authorization,
    monkeypatch,
):
    candidate = two_candidates()[0]
    observation = SimpleNamespace(observation_id="observation-001")
    memory = concrete_memory(tmp_path, eye_sample)

    def fail_publication(*args, **kwargs):
        raise OSError("private publication detail")

    monkeypatch.setattr(kimi_step_module, "_write_result", fail_publication)

    with pytest.raises(kimi_step_module.KimiResultPublicationError) as captured:
        execute_kimi_step(
            **execute_args(
                tmp_path,
                eye_sample,
                brain=factory_brain(
                    brain_result(
                        eye_sample,
                        action=DecisionAction.PROPOSE,
                        candidate=candidate,
                        memory=memory,
                    )
                ),
                orchestrator=RecordingOrchestrator(observations=(observation,)),
                memory=memory,
                authorization=stage_authorization,
            )
        )

    assert captured.value.hardware_executed is True
    assert captured.value.observation_ids == ("observation-001",)
    assert "private" not in str(captured.value)
