from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import AwareDatetime

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
)
from vcsel_ai_tuning.v2.continuous.controller import (
    ContinuousControllerError,
    ContinuousOptimizationController,
)
from vcsel_ai_tuning.v2.continuous.models import (
    ContinuousPreflightRequest,
    ContinuousRequest,
    ContinuousState,
    ExecutionMode,
    ExecutionOutcome,
    HumanDecision,
    ProviderAction,
    ProviderActionKind,
    ReasonCode,
    StoppingPolicy,
)
from vcsel_ai_tuning.v2.continuous.policy import DEFAULT_I2C_POLICY
from vcsel_ai_tuning.v2.continuous.policy import ProposalErrorCode
from vcsel_ai_tuning.v2.continuous.store import ContinuousCampaignStore
from vcsel_ai_tuning.v2.continuous.store import ContinuousStoreError
from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)
from vcsel_ai_tuning.v2.kimi_memory import KimiCampaignMemory
from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient
from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc)
B0 = {
    "ibias_dac": 70,
    "gain_ctrl": 5,
    "eq": 22,
    "efin_cur": 15,
    "vga_cur": 13,
    "res_gain": 6,
    "res_bw": 0,
    "vbias_ref": 15,
    "vref": 15,
}


class LegacyDeadlineContinuousRequest(ContinuousRequest):
    """Legacy test contract used only to prove controller independence from it."""

    commission_deadline: AwareDatetime


def _eye(
    tmp_path: Path,
    measurement_id: str,
    *,
    tdecq: float | None = 3.1,
    er: float = 4.5,
    readback: dict[str, int] | None = None,
    fingerprint: str = "b" * 64,
    status: MeasurementStatus = MeasurementStatus.VALID,
    er_status: str | None = None,
    tdecq_status: str | None = None,
    tdecq_invalid_reason: str | None = None,
) -> EyeSample:
    relative = f"eyes/{measurement_id}.png"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"fake-png-{measurement_id}".encode()
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    reference = ArtifactReference(
        relative_path=relative,
        sha256=digest,
        media_type="image/png",
    )
    values = readback or B0
    return EyeSample(
        measurement_id=measurement_id,
        campaign_id="campaign-1",
        sweep_point_id=measurement_id,
        raw_eye_density_ref=reference,
        metrics=EyeMetrics(tdecq_db=tdecq, outer_er_chan1a_db=er),
        requested_params={},
        applied_params={name: float(value) for name, value in values.items()},
        readback_params={name: float(value) for name, value in values.items()},
        dca_config_fingerprint=fingerprint,
        measurement_status=status,
        er_status=er_status,
        tdecq_status=tdecq_status,
        tdecq_invalid_reason=tdecq_invalid_reason,
        captured_at=NOW,
        artifact_hashes={relative: digest},
    )


def _authorization() -> StageAuthorization:
    return StageAuthorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_FINE,
        baseline_hash="c" * 64,
        rollback_hash="d" * 64,
        parameter_specs=DEFAULT_I2C_POLICY.authorization_specs(),
        max_candidates=3,
        max_duration_s=300.0,
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        rollback_snapshot=AppliedSettings(readback_i2c=B0),
    )


class FakeClock:
    def __init__(self) -> None:
        self.wall = NOW
        self.elapsed = 0.0

    def now(self):
        self.wall += timedelta(milliseconds=10)
        return self.wall

    def monotonic(self):
        self.elapsed += 0.01
        return self.elapsed


class ScriptedProvider:
    provider_id = "fake-provider"
    provider_version = "1"

    def __init__(self, *actions: ProviderAction | Exception) -> None:
        self.actions = list(actions)
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        item = self.actions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeExecution:
    def __init__(self, outcomes: list[ExecutionOutcome], *, restore_ok: bool = True) -> None:
        self.outcomes = list(outcomes)
        self.restore_ok = restore_ok
        self.calls = []

    def execute_candidate(self, authorization, candidate):
        self.calls.append(("execute", dict(candidate.requested_i2c)))
        return self.outcomes.pop(0)

    def measure_current_state(self):
        self.calls.append(("measure",))
        return self.outcomes.pop(0)

    def restore_b0(self, expected):
        self.calls.append(("restore", dict(expected)))
        if not self.restore_ok:
            raise RuntimeError("restore failed")
        return dict(expected)

    def recover_b0_after_failure(self, expected):
        self.calls.append(("failure_recovery", dict(expected)))
        if not self.restore_ok:
            raise RuntimeError("restore failed")
        return dict(expected)


class FakeLease:
    def __init__(self) -> None:
        self.calls = []

    def acquire(self, **kwargs):
        self.calls.append(("acquire", kwargs))
        return type("Lease", (), {"continuity_token": "f" * 64})()

    def release(self, campaign_id, continuity_token):
        self.calls.append(("release", campaign_id, continuity_token))


class FailingReleaseLease(FakeLease):
    def release(self, campaign_id, continuity_token):
        super().release(campaign_id, continuity_token)
        raise RuntimeError("injected lease release failure")


class StaleLease(FakeLease):
    def load(self):
        return type("Lease", (), {"campaign_id": "campaign-1"})()


class FailingInitializeStore:
    def initialize(self, campaign_id, metadata):
        del campaign_id, metadata
        raise RuntimeError("injected initialize failure")


class FailingTerminalStore(ContinuousCampaignStore):
    def save_state(self, campaign_id, payload):
        if payload.get("state") == ContinuousState.COMPLETED.value:
            raise RuntimeError("injected terminal state failure")
        return super().save_state(campaign_id, payload)


class FailingRunningStore(ContinuousCampaignStore):
    def save_state(self, campaign_id, payload):
        if payload.get("state") == ContinuousState.RUNNING.value:
            raise RuntimeError("injected running state failure")
        return super().save_state(campaign_id, payload)


class FailingSafeStore(ContinuousCampaignStore):
    def save_state(self, campaign_id, payload):
        if payload.get("state") == ContinuousState.FAILED_SAFE.value:
            raise RuntimeError("injected terminal state failure")
        return super().save_state(campaign_id, payload)


class FailingAwaitingStore(ContinuousCampaignStore):
    def save_state(self, campaign_id, payload):
        if payload.get("state") == ContinuousState.AWAITING_HUMAN.value:
            raise RuntimeError("injected awaiting state failure")
        return super().save_state(campaign_id, payload)


def _propose(**params: int) -> ProviderAction:
    return ProviderAction(
        action=ProviderActionKind.PROPOSE,
        requested_params=params,
        reason_codes=(ReasonCode.EXPLORATION,),
        confidence=0.9,
        summary="Try bounded values.",
    )


def _stop() -> ProviderAction:
    return ProviderAction(
        action=ProviderActionKind.STOP,
        reason_codes=(ReasonCode.EXHAUSTED_SEARCH,),
        confidence=0.9,
        summary="Stop and verify.",
    )


def _hold(*, confidence: float = 0.9) -> ProviderAction:
    return ProviderAction(
        action=ProviderActionKind.HOLD,
        reason_codes=(ReasonCode.EXPLORATION,),
        confidence=confidence,
        summary="Reconsider the same observation.",
    )


def _outcome(sample: EyeSample, requested: dict[str, int]) -> ExecutionOutcome:
    readback = {name: int(value) for name, value in sample.readback_params.items()}
    return ExecutionOutcome(
        sample=sample,
        requested_params=requested,
        applied_params=requested,
        readback_params=readback,
        adapter_identities={"i2c": "fake-i2c", "dca": "fake-dca"},
        evidence_complete=True,
    )


def _request(
    tmp_path: Path,
    initial: EyeSample,
    *,
    max_rounds: int = 3,
    target_tdecq_db: float = 2.0,
) -> ContinuousRequest:
    return ContinuousRequest(
        campaign_id="campaign-1",
        initial_eye=initial,
        authorization=_authorization(),
        policy=DEFAULT_I2C_POLICY.model_dump(mode="json"),
        max_rounds=max_rounds,
        provider_config={"provider_id": "fake-provider"},
        stopping_policy=StoppingPolicy(target_tdecq_db=target_tdecq_db),
        b0_path=Path("configs/ai_tuning/b0.yaml"),
        b0_i2c=B0,
        result_dir=tmp_path / "results",
        execution_mode=ExecutionMode.FAKE,
    )


def test_controller_completes_max_rounds_only_after_durable_valid_evidence(tmp_path) -> None:
    initial = _eye(tmp_path, "before")
    readback1 = B0 | {"eq": 23}
    readback2 = readback1 | {"gain_ctrl": 6}
    execution = FakeExecution(
        [
            _outcome(_eye(tmp_path, "after-1", readback=readback1), {"eq": 23}),
            _outcome(_eye(tmp_path, "after-2", readback=readback2), {"gain_ctrl": 6}),
        ]
    )
    provider = ScriptedProvider(_propose(eq=23), _propose(gain_ctrl=6))
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    policy_payload = DEFAULT_I2C_POLICY.model_dump(mode="json")
    policy_payload["version"] = "i2c-restricted-v1"
    ibias_rule = next(
        rule for rule in policy_payload["rules"] if rule["name"] == "ibias_dac"
    )
    ibias_rule.update(tuning_min=60, tuning_max=80, max_step=10)
    request = _request(tmp_path, initial, max_rounds=2).model_copy(
        update={"policy": policy_payload}
    )

    result = controller.start(request)

    campaign = json.loads(
        (tmp_path / "results" / "campaign-1" / "campaign.json").read_text(
            encoding="utf-8"
        )
    )
    assert campaign["policy"] == policy_payload
    assert campaign["max_rounds"] == 2
    assert campaign["target_tdecq_db"] == 2.0

    assert result.state is ContinuousState.COMPLETED
    assert result.completed_rounds == 2
    assert result.stop_reason == "MAX_ROUNDS_REACHED"
    assert result.closed_loop_proven is False
    assert len(result.round_records) == 2
    assert [item.hardware_round for item in provider.contexts] == [1, 2]
    assert provider.contexts[0].best_reference is None
    assert provider.contexts[1].best_reference == initial
    assert execution.calls[-1][0] == "restore"
    assert (tmp_path / "results" / "campaign-1" / "rounds" / "0002" / "round.json").is_file()


def test_controller_keeps_best_prior_eye_as_next_round_reference(tmp_path) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.1)
    after_1 = _eye(tmp_path, "after-1", tdecq=2.9, readback=B0 | {"eq": 23})
    after_2 = _eye(
        tmp_path,
        "after-2",
        tdecq=3.3,
        readback=B0 | {"eq": 23, "gain_ctrl": 6},
    )
    after_3 = _eye(
        tmp_path,
        "after-3",
        tdecq=3.0,
        readback=B0 | {"eq": 23, "gain_ctrl": 6, "res_bw": 1},
    )
    execution = FakeExecution(
        [
            _outcome(after_1, {"eq": 23}),
            _outcome(after_2, {"gain_ctrl": 6}),
            _outcome(after_3, {"res_bw": 1}),
        ]
    )
    provider = ScriptedProvider(
        _propose(eq=23),
        _propose(gain_ctrl=6),
        _propose(res_bw=1),
    )
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=3))

    assert result.completed_rounds == 3
    assert [context.best_reference for context in provider.contexts] == [
        None,
        initial,
        after_1,
    ]


def test_controller_continues_after_an_exploratory_first_round_is_worse(tmp_path) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.1)
    worse = _eye(tmp_path, "worse", tdecq=3.4, readback=B0 | {"eq": 23})
    recovered = _eye(
        tmp_path,
        "recovered",
        tdecq=3.0,
        readback=B0 | {"eq": 23, "gain_ctrl": 6},
    )
    provider = ScriptedProvider(_propose(eq=23), _propose(gain_ctrl=6))
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=FakeExecution(
            [
                _outcome(worse, {"eq": 23}),
                _outcome(recovered, {"gain_ctrl": 6}),
            ]
        ),
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=2))

    assert result.state is ContinuousState.COMPLETED
    assert result.completed_rounds == 2
    assert [context.sample for context in provider.contexts] == [initial, worse]
    assert [context.best_reference for context in provider.contexts] == [None, initial]


def test_controller_continues_from_invalid_candidate_without_intermediate_b0_restore(
    tmp_path,
) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.1)
    invalid_readback = B0 | {"eq": 23}
    invalid = _eye(
        tmp_path,
        "invalid",
        tdecq=None,
        er=1.982,
        readback=invalid_readback,
        status=MeasurementStatus.INVALID,
        er_status="CORR",
        tdecq_status="INV",
        tdecq_invalid_reason="The target SER is lower than actual SER of the signal.",
    )
    recovered_readback = invalid_readback | {"gain_ctrl": 6}
    recovered = _eye(
        tmp_path,
        "recovered",
        tdecq=2.9,
        readback=recovered_readback,
    )
    execution = FakeExecution(
        [
            _outcome(invalid, {"eq": 23}),
            _outcome(recovered, {"gain_ctrl": 6}),
        ]
    )
    provider = ScriptedProvider(_propose(eq=23), _propose(gain_ctrl=6))
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=2))

    assert result.state is ContinuousState.COMPLETED
    assert result.completed_rounds == 2
    assert [record.measurement_status for record in result.round_records] == [
        MeasurementStatus.INVALID,
        MeasurementStatus.VALID,
    ]
    assert provider.contexts[1].sample == invalid
    assert provider.contexts[1].best_reference == initial
    assert provider.contexts[1].verified_readback == invalid_readback
    assert provider.contexts[1].recent_valid_summaries == ()
    assert [call[0] for call in execution.calls] == ["execute", "execute", "restore"]


def test_invalid_er_candidate_with_finite_tdecq_is_not_target_or_best_reference(
    tmp_path,
) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.1)
    questionable = _eye(
        tmp_path,
        "questionable-er",
        tdecq=2.5,
        er=1.9,
        readback=B0 | {"eq": 23},
        status=MeasurementStatus.INVALID,
        er_status="QUES",
        tdecq_status="CORR",
    )
    recovered = _eye(
        tmp_path,
        "recovered",
        tdecq=2.9,
        readback=B0 | {"eq": 23, "gain_ctrl": 6},
    )
    provider = ScriptedProvider(_propose(eq=23), _propose(gain_ctrl=6))
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=FakeExecution(
            [
                _outcome(questionable, {"eq": 23}),
                _outcome(recovered, {"gain_ctrl": 6}),
            ]
        ),
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=2))

    assert result.completed_rounds == 2
    assert result.target_reached is False
    assert provider.contexts[1].sample == questionable
    assert provider.contexts[1].best_reference == initial


def test_controller_reports_no_valid_candidates_after_invalid_round_budget(
    tmp_path,
) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.1)
    invalid_1 = _eye(
        tmp_path,
        "invalid-1",
        tdecq=None,
        readback=B0 | {"eq": 23},
        status=MeasurementStatus.INVALID,
        er_status="CORR",
        tdecq_status="INV",
    )
    invalid_2 = _eye(
        tmp_path,
        "invalid-2",
        tdecq=None,
        readback=B0 | {"eq": 23, "gain_ctrl": 6},
        status=MeasurementStatus.INVALID,
        er_status="CORR",
        tdecq_status="INV",
    )
    execution = FakeExecution(
        [
            _outcome(invalid_1, {"eq": 23}),
            _outcome(invalid_2, {"gain_ctrl": 6}),
        ]
    )
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_propose(eq=23), _propose(gain_ctrl=6)),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=2))

    assert result.state is ContinuousState.COMPLETED
    assert result.stop_reason == "NO_VALID_CANDIDATES"
    assert result.completed_rounds == 2
    assert result.b0_restored is True
    assert [call[0] for call in execution.calls] == ["execute", "execute", "restore"]


def test_candidate_after_former_aggregate_deadline_completes_round_budget(
    tmp_path,
) -> None:
    clock = FakeClock()
    initial = _eye(tmp_path, "before", tdecq=3.2)
    after = _eye(tmp_path, "after", tdecq=3.0, readback=B0 | {"eq": 23})

    class SlowProvider(ScriptedProvider):
        def decide(self, context):
            clock.wall += timedelta(seconds=901)
            return super().decide(context)

    class SlowExecution(FakeExecution):
        def execute_candidate(self, authorization, candidate):
            clock.wall += timedelta(seconds=901)
            return super().execute_candidate(authorization, candidate)

    controller = ContinuousOptimizationController(
        provider=SlowProvider(_propose(eq=23)),
        execution=SlowExecution([_outcome(after, {"eq": 23})]),
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=clock,
    )

    base_request = _request(tmp_path, initial, max_rounds=1)
    request = LegacyDeadlineContinuousRequest(
        **base_request.model_dump(),
        commission_deadline=NOW + timedelta(seconds=1_500),
    )

    result = controller.start(request)

    assert result.state is ContinuousState.COMPLETED
    assert result.stop_reason == "MAX_ROUNDS_REACHED"
    assert result.stop_reason != "COMMISSION_TIMEOUT"
    assert result.b0_restored is True


@pytest.mark.parametrize(
    ("measurement_id", "status", "tdecq", "expected_stop_reason", "target_reached"),
    [
        ("below-target", MeasurementStatus.VALID, 2.79, "TARGET_ACCEPTED", True),
        ("at-target", MeasurementStatus.VALID, 2.80, "MAX_ROUNDS_REACHED", False),
        ("invalid-below-target", MeasurementStatus.INVALID, 2.79, "NO_VALID_CANDIDATES", False),
    ],
)
def test_operator_target_requires_valid_tdecq_strictly_below_boundary(
    tmp_path,
    measurement_id,
    status,
    tdecq,
    expected_stop_reason,
    target_reached,
) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.1)
    candidate = _eye(
        tmp_path,
        measurement_id,
        tdecq=tdecq,
        readback=B0 | {"eq": 23},
        status=status,
        er_status="CORR" if status is MeasurementStatus.INVALID else None,
        tdecq_status="CORR" if status is MeasurementStatus.INVALID else None,
    )
    outcomes = [_outcome(candidate, {"eq": 23})]
    if target_reached:
        outcomes.append(
            _outcome(
                _eye(
                    tmp_path,
                    "acceptance-confirmation",
                    tdecq=tdecq,
                    readback=B0 | {"eq": 23},
                ),
                {},
            )
        )
    execution = FakeExecution(outcomes)
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_propose(eq=23)),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )
    request = _request(tmp_path, initial, max_rounds=1).model_copy(
        update={"stopping_policy": StoppingPolicy(target_tdecq_db=2.8)}
    )

    result = controller.start(request)

    assert result.state is ContinuousState.COMPLETED
    assert result.stop_reason == expected_stop_reason
    assert result.target_reached is target_reached


def test_initial_valid_eye_below_target_accepts_without_provider_or_candidate_write(
    tmp_path,
) -> None:
    initial = _eye(tmp_path, "initial-below-target", tdecq=2.79)
    confirmation = _eye(tmp_path, "initial-target-confirmation", tdecq=2.79)
    provider = ScriptedProvider(_propose(eq=23))
    execution = FakeExecution([_outcome(confirmation, {})])
    result_root = tmp_path / "results"
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(result_root),
        clock=FakeClock(),
    )

    result = controller.start(
        _request(tmp_path, initial, target_tdecq_db=2.8)
    )

    campaign = json.loads(
        (result_root / "campaign-1" / "campaign.json").read_text(encoding="utf-8")
    )
    final_state = json.loads(
        (result_root / "campaign-1" / "state.json").read_text(encoding="utf-8")
    )
    assert result.state is ContinuousState.COMPLETED
    assert result.completed_rounds == 0
    assert result.target_reached is True
    assert result.acceptance_passed is True
    assert result.stop_reason == "TARGET_ACCEPTED"
    assert result.target_tdecq_db == 2.8
    assert campaign["target_tdecq_db"] == 2.8
    assert final_state["target_tdecq_db"] == 2.8
    assert provider.contexts == []
    assert [call[0] for call in execution.calls] == ["measure", "restore"]
    assert not any(call[0] == "execute" for call in execution.calls)


@pytest.mark.parametrize(
    ("status", "tdecq"),
    [
        pytest.param(MeasurementStatus.VALID, 2.8, id="equal-target"),
        pytest.param(MeasurementStatus.INVALID, 2.79, id="invalid-below-target"),
    ],
)
def test_initial_eye_not_strictly_valid_below_target_continues_provider_flow(
    tmp_path,
    status: MeasurementStatus,
    tdecq: float,
) -> None:
    initial = _eye(
        tmp_path,
        "initial-boundary",
        tdecq=tdecq,
        status=status,
        er_status="CORR" if status is MeasurementStatus.INVALID else None,
        tdecq_status="INV" if status is MeasurementStatus.INVALID else None,
        tdecq_invalid_reason=(
            "instrument marked the measurement invalid"
            if status is MeasurementStatus.INVALID
            else None
        ),
    )
    after = _eye(
        tmp_path,
        "after-normal-provider-flow",
        tdecq=3.0,
        readback=B0 | {"eq": 23},
    )
    provider = ScriptedProvider(_propose(eq=23))
    execution = FakeExecution([_outcome(after, {"eq": 23})])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(
        _request(tmp_path, initial, max_rounds=1, target_tdecq_db=2.8)
    )

    assert result.state is ContinuousState.COMPLETED
    assert result.target_reached is False
    assert result.stop_reason == "MAX_ROUNDS_REACHED"
    assert len(provider.contexts) == 1
    assert [call[0] for call in execution.calls].count("execute") == 1


def test_resume_with_persisted_eye_below_target_avoids_another_candidate(
    tmp_path,
) -> None:
    request_human = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Pause before another candidate.",
    )
    confirmation = _eye(tmp_path, "resume-target-confirmation", tdecq=2.79)
    provider = ScriptedProvider(request_human, _propose(eq=23))
    execution = FakeExecution([_outcome(confirmation, {})])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=FakeLease(),
    )
    request = _request(
        tmp_path,
        _eye(tmp_path, "before-human-wait", tdecq=3.1),
        target_tdecq_db=2.8,
    )

    awaiting = controller.start(request)
    persisted = list(controller._awaiting[request.campaign_id])
    persisted[1] = _eye(tmp_path, "persisted-below-target", tdecq=2.79)
    controller._awaiting[request.campaign_id] = tuple(persisted)
    result = controller.resolve_human(request.campaign_id, HumanDecision.RESUME)

    assert awaiting.state is ContinuousState.AWAITING_HUMAN
    assert result.state is ContinuousState.COMPLETED
    assert result.stop_reason == "TARGET_ACCEPTED"
    assert result.completed_rounds == 0
    assert len(provider.contexts) == 1
    assert [call[0] for call in execution.calls] == ["measure", "restore"]


@pytest.mark.parametrize("resumed", [False, True], ids=["reinitialize", "resume"])
def test_same_campaign_target_drift_fails_before_provider_or_execution(
    tmp_path,
    resumed: bool,
) -> None:
    result_root = tmp_path / "results"
    store = ContinuousCampaignStore(result_root)
    initial = _eye(tmp_path, "before-target-drift")
    original = _request(tmp_path, initial, target_tdecq_db=2.8)
    store.initialize(
        original.campaign_id,
        {
            "campaign_id": original.campaign_id,
            "execution_mode": original.execution_mode.value,
            "max_rounds": original.max_rounds,
            "target_tdecq_db": original.stopping_policy.target_tdecq_db,
            "policy": dict(original.policy),
            "provider": dict(original.provider_config),
        },
    )
    provider = ScriptedProvider(_propose(eq=23))
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=store,
        clock=FakeClock(),
    )
    if resumed:
        controller._resume_states[original.campaign_id] = (
            initial,
            None,
            [],
            {},
            NOW,
        )
    drifted = original.model_copy(
        update={"stopping_policy": StoppingPolicy(target_tdecq_db=2.9)}
    )

    with pytest.raises(ContinuousStoreError, match="immutable metadata"):
        controller.start(drifted)

    assert provider.contexts == []
    assert execution.calls == []


def test_campaign_metadata_without_target_requires_migration_before_execution(
    tmp_path,
) -> None:
    result_root = tmp_path / "results"
    store = ContinuousCampaignStore(result_root)
    initial = _eye(tmp_path, "before-historical-campaign")
    request = _request(tmp_path, initial, target_tdecq_db=2.8)
    store.initialize(
        request.campaign_id,
        {
            "campaign_id": request.campaign_id,
            "execution_mode": request.execution_mode.value,
            "max_rounds": request.max_rounds,
            "policy": dict(request.policy),
            "provider": dict(request.provider_config),
        },
    )
    provider = ScriptedProvider(_propose(eq=23))
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=store,
        clock=FakeClock(),
    )

    with pytest.raises(
        ContinuousStoreError,
        match="target_tdecq_db.*new campaign|migrat",
    ):
        controller.start(request)

    assert provider.contexts == []
    assert execution.calls == []


def test_controller_restores_b0_when_initial_persistence_fails(tmp_path) -> None:
    initial = _eye(tmp_path, "before")
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_propose(eq=23)),
        execution=execution,
        store=FailingInitializeStore(),
        clock=FakeClock(),
    )

    with pytest.raises(RuntimeError, match="initialize failure"):
        controller.start(_request(tmp_path, initial))

    assert execution.calls == [("failure_recovery", B0)]


def test_controller_recovers_b0_when_initial_running_state_persistence_fails(
    tmp_path,
) -> None:
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_propose(eq=23)),
        execution=execution,
        store=FailingRunningStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    with pytest.raises(RuntimeError, match="running state failure"):
        controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert execution.calls == [("failure_recovery", B0)]
    with pytest.raises(ContinuousControllerError, match="unknown campaign"):
        controller.get_status("campaign-1")


def test_stop_runs_acceptance_without_write_and_can_finish_without_target(tmp_path) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.0, er=4.5)
    confirmation = _eye(tmp_path, "confirm", tdecq=3.1, er=4.6)
    execution = FakeExecution([_outcome(confirmation, {})])
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_stop()),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial))

    assert result.state is ContinuousState.COMPLETED
    assert result.acceptance_passed is True
    assert result.target_reached is False
    assert result.stop_reason == "PROVIDER_STOP_ACCEPTED"
    assert [call[0] for call in execution.calls] == ["measure", "restore"]


@pytest.mark.parametrize(
    ("failure", "restore_ok", "state"),
    [
        (RuntimeError("provider unavailable"), True, ContinuousState.FAILED_SAFE),
        (RuntimeError("provider unavailable"), False, ContinuousState.RECOVERY_REQUIRED),
    ],
)
def test_controller_fails_closed_and_restores_b0(
    tmp_path,
    failure: Exception,
    restore_ok: bool,
    state: ContinuousState,
) -> None:
    execution = FakeExecution([], restore_ok=restore_ok)
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(failure),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert result.state is state
    assert result.completed_rounds == 0
    assert result.closed_loop_proven is False
    assert result.stop_reason in {"PROVIDER_FAILURE", "B0_RECOVERY_FAILED"}
    assert result.provider_error_type == "RuntimeError"
    assert result.provider_error_message == "provider unavailable"
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_exhausted_kimi_truncation_retries_fail_safe_and_restore_b0(tmp_path) -> None:
    requests: list[dict[str, object]] = []

    def create(**request: object) -> SimpleNamespace:
        requests.append(request)
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content='{"action":"PROPOSE"'),
                ),
            ),
            usage={"completion_tokens": request["max_completion_tokens"]},
        )

    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = KimiProvider(
        KimiApiClient(
            api_key="test-key",
            sdk_client=sdk,
            response_log_path=tmp_path / "kimi-response-attempts.jsonl",
        ),
        artifact_root=tmp_path,
        instruction="Improve TDECQ within the supplied policy.",
        memory=KimiCampaignMemory(
            tmp_path / "kimi-memory.jsonl",
            campaign_id="campaign-1",
        ),
    )
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert [request["max_completion_tokens"] for request in requests] == [
        4096,
        8192,
    ]
    assert result.state is ContinuousState.FAILED_SAFE
    assert result.stop_reason == "PROVIDER_FAILURE"
    assert result.b0_restored is True
    assert result.provider_error_type == "ValueError"
    assert "invalid response" in result.provider_error_message
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_one_invalid_proposal_is_corrected_against_same_observation(tmp_path) -> None:
    initial = _eye(tmp_path, "before")
    readback = B0 | {"eq": 23}
    provider = ScriptedProvider(_propose(eq=99), _propose(eq=23))
    execution = FakeExecution(
        [_outcome(_eye(tmp_path, "after", readback=readback), {"eq": 23})]
    )
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=1))

    assert result.completed_rounds == 1
    assert len(provider.contexts) == 2
    assert provider.contexts[0].sample == provider.contexts[1].sample
    assert provider.contexts[1].correction_error_code == ProposalErrorCode.OUTSIDE_ENCODING_RANGE.value


def test_second_invalid_proposal_ends_without_hardware_round(tmp_path) -> None:
    provider = ScriptedProvider(_propose(eq=99), _propose(eq=98))
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert result.state is ContinuousState.FAILED_SAFE
    assert result.completed_rounds == 0
    assert result.stop_reason == "INVALID_PROPOSAL_EXHAUSTED"
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_restricted_range_rejection_never_executes_hardware_round(tmp_path) -> None:
    policy_payload = DEFAULT_I2C_POLICY.model_dump(mode="json")
    policy_payload["version"] = "i2c-restricted-v1"
    eq_rule = next(rule for rule in policy_payload["rules"] if rule["name"] == "eq")
    eq_rule.update(tuning_min=16, tuning_max=25, max_step=3)
    request = _request(tmp_path, _eye(tmp_path, "before")).model_copy(
        update={"policy": policy_payload}
    )
    provider = ScriptedProvider(_propose(eq=26), _propose(eq=26))
    execution = FakeExecution([])
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(request)

    assert result.state is ContinuousState.FAILED_SAFE
    assert result.completed_rounds == 0
    assert provider.contexts[1].correction_error_code == (
        ProposalErrorCode.OUTSIDE_TUNING_RANGE.value
    )
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_one_eligible_hold_requeries_same_observation_then_executes(tmp_path) -> None:
    initial = _eye(tmp_path, "before")
    readback = B0 | {"eq": 23}
    provider = ScriptedProvider(_hold(), _propose(eq=23))
    execution = FakeExecution(
        [_outcome(_eye(tmp_path, "after", readback=readback), {"eq": 23})]
    )
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial, max_rounds=1))

    assert result.completed_rounds == 1
    assert len(provider.contexts) == 2
    assert provider.contexts[0].sample == provider.contexts[1].sample


def test_second_hold_and_low_confidence_hold_stop_without_execution(tmp_path) -> None:
    for actions, reason in (
        ((_hold(), _hold()), "HOLD_RETRY_EXHAUSTED"),
        ((_hold(confidence=0.1),), "LOW_CONFIDENCE"),
    ):
        run_root = tmp_path / reason
        controller = ContinuousOptimizationController(
            provider=ScriptedProvider(*actions),
            execution=FakeExecution([]),
            store=ContinuousCampaignStore(run_root / "results"),
            clock=FakeClock(),
        )
        result = controller.start(_request(run_root, _eye(run_root, "before")))
        assert result.state is ContinuousState.FAILED_SAFE
        assert result.stop_reason == reason
        assert result.completed_rounds == 0


@pytest.mark.parametrize(
    ("tdecq", "er", "fingerprint", "readback", "expected"),
    [
        (3.2, 4.6, "b" * 64, B0, True),
        (3.21, 4.6, "b" * 64, B0, False),
        (3.1, 4.71, "b" * 64, B0, False),
        (3.1, 4.6, "e" * 64, B0, False),
        (3.1, 4.6, "b" * 64, B0 | {"eq": 23}, False),
    ],
)
def test_acceptance_requires_metric_thresholds_fingerprint_and_same_readback(
    tmp_path,
    tdecq: float,
    er: float,
    fingerprint: str,
    readback: dict[str, int],
    expected: bool,
) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.0, er=4.5)
    confirmation = _eye(
        tmp_path,
        "confirm",
        tdecq=tdecq,
        er=er,
        readback=readback,
        fingerprint=fingerprint,
    )
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_stop()),
        execution=FakeExecution([_outcome(confirmation, {})]),
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, initial))

    assert result.acceptance_passed is expected
    assert result.state is (
        ContinuousState.COMPLETED if expected else ContinuousState.FAILED_SAFE
    )


def test_request_human_holds_lease_and_restore_decision_recovers_b0(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    execution = FakeExecution([])
    lease = FakeLease()
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=lease,
    )

    request = _request(tmp_path, _eye(tmp_path, "before")).model_copy(
        update={"human_timeout_s": 123.0}
    )
    awaiting = controller.start(request)
    result = controller.resolve_human("campaign-1", HumanDecision.RESTORE_B0)

    assert awaiting.state is ContinuousState.AWAITING_HUMAN
    assert result.state is ContinuousState.COMPLETED
    assert result.stop_reason == "HUMAN_RESTORE_B0"
    assert [call[0] for call in execution.calls] == ["restore"]
    assert [call[0] for call in lease.calls] == ["acquire", "release"]
    lease_request = lease.calls[0][1]
    assert lease_request["expires_at"] == lease_request["issued_at"] + timedelta(
        seconds=request.human_timeout_s
    )


def test_request_human_legacy_deadline_does_not_cap_lease_or_timeout(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    clock = FakeClock()
    execution = FakeExecution([])
    lease = FakeLease()
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=clock,
        bench_lease=lease,
    )
    base_request = _request(tmp_path, _eye(tmp_path, "before")).model_copy(
        update={"human_timeout_s": 123.0}
    )
    request = LegacyDeadlineContinuousRequest(
        **base_request.model_dump(),
        commission_deadline=NOW + timedelta(seconds=1),
    )

    awaiting = controller.start(request)
    lease_request = lease.calls[0][1]
    assert lease_request["expires_at"] == lease_request["issued_at"] + timedelta(
        seconds=request.human_timeout_s
    )

    clock.wall += timedelta(seconds=2)
    still_awaiting = controller.check_human_timeout("campaign-1")

    assert awaiting.state is ContinuousState.AWAITING_HUMAN
    assert still_awaiting.state is ContinuousState.AWAITING_HUMAN
    assert controller.get_status("campaign-1").state is ContinuousState.AWAITING_HUMAN

    clock.wall += timedelta(seconds=request.human_timeout_s)
    timed_out = controller.check_human_timeout("campaign-1")

    assert timed_out.state is ContinuousState.FAILED_SAFE
    assert timed_out.stop_reason == "HUMAN_TIMEOUT"
    assert timed_out.b0_restored is True
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_human_resolution_releases_lease_when_terminal_store_fails(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    lease = FakeLease()
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=FakeExecution([]),
        store=FailingTerminalStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=lease,
    )

    controller.start(_request(tmp_path, _eye(tmp_path, "before")))
    with pytest.raises(RuntimeError, match="terminal state failure"):
        controller.resolve_human("campaign-1", HumanDecision.RESTORE_B0)

    assert [call[0] for call in lease.calls] == ["acquire", "release"]


def test_request_human_accept_decision_uses_acceptance_measurement(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    initial = _eye(tmp_path, "before", tdecq=3.0, er=4.5)
    confirmation = _eye(tmp_path, "confirm", tdecq=3.1, er=4.6)
    execution = FakeExecution([_outcome(confirmation, {})])
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=FakeLease(),
    )

    controller.start(_request(tmp_path, initial))
    result = controller.resolve_human("campaign-1", HumanDecision.ACCEPT)

    assert result.state is ContinuousState.COMPLETED
    assert result.acceptance_passed is True
    assert result.stop_reason == "HUMAN_ACCEPTED"


def test_request_human_after_round_accepts_against_current_observation(tmp_path) -> None:
    request_human = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    initial = _eye(tmp_path, "before", tdecq=3.2, er=4.5)
    readback = B0 | {"eq": 23}
    after = _eye(tmp_path, "after", tdecq=2.8, er=4.6, readback=readback)
    confirmation = _eye(
        tmp_path,
        "confirm",
        tdecq=2.8,
        er=4.6,
        readback=readback,
    )
    execution = FakeExecution(
        [_outcome(after, {"eq": 23}), _outcome(confirmation, {})]
    )
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_propose(eq=23), request_human),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=FakeLease(),
    )

    awaiting = controller.start(_request(tmp_path, initial, max_rounds=2))
    result = controller.resolve_human("campaign-1", HumanDecision.ACCEPT)

    assert awaiting.state is ContinuousState.AWAITING_HUMAN
    assert result.state is ContinuousState.COMPLETED
    assert result.acceptance_passed is True


def test_human_timeout_attempts_b0_restore(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    execution = FakeExecution([])
    clock = FakeClock()
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=clock,
        bench_lease=FakeLease(),
    )
    request = _request(tmp_path, _eye(tmp_path, "before")).model_copy(
        update={"human_timeout_s": 900.0}
    )

    controller.start(request)
    clock.wall += timedelta(minutes=16)
    result = controller.check_human_timeout("campaign-1")

    assert result.state is ContinuousState.FAILED_SAFE
    assert result.stop_reason == "HUMAN_TIMEOUT"
    assert result.b0_restored is True
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_human_timeout_releases_lease_when_terminal_store_fails(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    clock = FakeClock()
    lease = FakeLease()
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=FakeExecution([]),
        store=FailingSafeStore(tmp_path / "results"),
        clock=clock,
        bench_lease=lease,
    )
    request = _request(tmp_path, _eye(tmp_path, "before")).model_copy(
        update={"human_timeout_s": 1.0}
    )
    controller.start(request)
    clock.wall += timedelta(seconds=2)

    with pytest.raises(RuntimeError, match="terminal state failure"):
        controller.check_human_timeout("campaign-1")

    assert [call[0] for call in lease.calls] == ["acquire", "release"]


def test_await_human_state_failure_releases_acquired_lease(tmp_path) -> None:
    action = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Need a human decision.",
    )
    lease = FakeLease()
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(action),
        execution=FakeExecution([]),
        store=FailingAwaitingStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=lease,
    )

    result = controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert result.state is ContinuousState.FAILED_SAFE
    assert [call[0] for call in lease.calls] == ["acquire", "release"]


def test_explicit_human_resume_continues_remaining_rounds(tmp_path) -> None:
    request_human = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Pause for approval.",
    )
    readback = B0 | {"eq": 23}
    after = _eye(tmp_path, "after-resume", readback=readback, tdecq=3.0)
    execution = FakeExecution([_outcome(after, {"eq": 23})])
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(request_human, _propose(eq=23)),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=FakeLease(),
    )

    awaiting = controller.start(
        _request(tmp_path, _eye(tmp_path, "before"), max_rounds=1)
    )
    result = controller.resolve_human("campaign-1", HumanDecision.RESUME)

    assert awaiting.state is ContinuousState.AWAITING_HUMAN
    assert result.state is ContinuousState.COMPLETED
    assert result.completed_rounds == 1
    assert result.stop_reason == "MAX_ROUNDS_REACHED"


def test_human_resume_after_completed_round_preserves_best_reference(tmp_path) -> None:
    request_human = ProviderAction(
        action=ProviderActionKind.REQUEST_HUMAN,
        reason_codes=(ReasonCode.NEED_HUMAN,),
        confidence=0.8,
        summary="Pause after the exploratory round.",
    )
    initial = _eye(tmp_path, "before", tdecq=3.1)
    worse = _eye(tmp_path, "worse", tdecq=3.4, readback=B0 | {"eq": 23})
    recovered = _eye(
        tmp_path,
        "recovered",
        tdecq=3.0,
        readback=B0 | {"eq": 23, "gain_ctrl": 6},
    )
    provider = ScriptedProvider(
        _propose(eq=23),
        request_human,
        _propose(gain_ctrl=6),
    )
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=FakeExecution(
            [
                _outcome(worse, {"eq": 23}),
                _outcome(recovered, {"gain_ctrl": 6}),
            ]
        ),
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=FakeLease(),
    )

    awaiting = controller.start(_request(tmp_path, initial, max_rounds=2))
    result = controller.resolve_human("campaign-1", HumanDecision.RESUME)

    assert awaiting.state is ContinuousState.AWAITING_HUMAN
    assert awaiting.completed_rounds == 1
    assert result.state is ContinuousState.COMPLETED
    assert result.completed_rounds == 2
    assert [context.best_reference for context in provider.contexts] == [
        None,
        initial,
        initial,
    ]


@pytest.mark.parametrize(
    "reason, expected",
    [
        (ReasonCode.INVALID_OBSERVATION, "INVALID_OBSERVATION"),
        (ReasonCode.OOD, "OOD"),
    ],
)
def test_invalid_or_ood_provider_assessment_stops_immediately(
    tmp_path,
    reason,
    expected,
) -> None:
    action = ProviderAction(
        action=ProviderActionKind.HOLD,
        reason_codes=(reason,),
        confidence=0.9,
        summary="Observation cannot be used safely.",
    )
    execution = FakeExecution([])
    provider = ScriptedProvider(action)
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert result.state is ContinuousState.FAILED_SAFE
    assert result.stop_reason == expected
    assert len(provider.contexts) == 1
    assert [call[0] for call in execution.calls] == ["failure_recovery"]


def test_acceptance_uses_configured_tolerances(tmp_path) -> None:
    initial = _eye(tmp_path, "before", tdecq=3.0, er=4.5)
    confirmation = _eye(tmp_path, "confirm", tdecq=3.3, er=4.8)
    execution = FakeExecution([_outcome(confirmation, {})])
    controller = ContinuousOptimizationController(
        provider=ScriptedProvider(_stop()),
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )
    request = _request(tmp_path, initial).model_copy(
        update={
            "stopping_policy": StoppingPolicy(
                acceptance_tdecq_delta_db=0.31,
                acceptance_outer_er_delta_db=0.31,
            )
        }
    )

    result = controller.start(request)

    assert result.state is ContinuousState.COMPLETED
    assert result.acceptance_passed is True


def test_restart_with_unprovable_lease_requires_recovery_without_execution(
    tmp_path,
) -> None:
    execution = FakeExecution([])
    provider = ScriptedProvider(_propose(eq=23))
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
        bench_lease=StaleLease(),
    )

    result = controller.start(_request(tmp_path, _eye(tmp_path, "before")))

    assert result.state is ContinuousState.RECOVERY_REQUIRED
    assert result.stop_reason == "LEASE_CONTINUITY_LOST"
    assert provider.contexts == []
    assert execution.calls == []


def test_provider_preview_never_calls_execution_port(tmp_path) -> None:
    action = _propose(eq=23)
    execution = FakeExecution([])
    provider = ScriptedProvider(action)
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )

    result = controller.preview_initial_decision(
        _request(tmp_path, _eye(tmp_path, "before"))
    )

    assert result == action
    assert len(provider.contexts) == 1
    assert execution.calls == []


def test_provider_preview_accepts_authorization_free_preflight_request(tmp_path) -> None:
    action = _propose(eq=23)
    execution = FakeExecution([])
    provider = ScriptedProvider(action)
    controller = ContinuousOptimizationController(
        provider=provider,
        execution=execution,
        store=ContinuousCampaignStore(tmp_path / "results"),
        clock=FakeClock(),
    )
    request = ContinuousPreflightRequest(
        campaign_id="campaign-1",
        initial_eye=_eye(tmp_path, "preflight-before"),
        policy=DEFAULT_I2C_POLICY.model_dump(mode="json"),
        max_rounds=3,
        stopping_policy=StoppingPolicy(),
    )

    result = controller.preview_initial_decision(request)

    assert result == action
    assert len(provider.contexts) == 1
    assert execution.calls == []
