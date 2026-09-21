from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.campaign.authorization import (
    AuthorizationError,
    canonical_json_sha256,
    rollback_hash_for,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    BaselineConfig,
    CampaignState,
    Candidate,
    FirAuthorizationSpec,
    FirSettings,
    ParameterSpec,
    StageAuthorization,
    load_b0,
)
from vcsel_ai_tuning.campaign.orchestrator import (
    CampaignOrchestrator,
    CampaignPausedError,
    HardwareSafetyError,
    MeasurementResult,
    OrchestratorError,
    PreflightError,
    RecoveryRequiredError,
)
from vcsel_ai_tuning.campaign.repository import (
    ArtifactSpec,
    CampaignRepository,
    RepositoryError,
)
from vcsel_ai_tuning.campaign.service import ConcreteCampaignService
from vcsel_ai_tuning.campaign.store import CampaignStore, ConcurrencyError, StoreError
from vcsel_ai_tuning.campaign.state_machine import IllegalTransition
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    OptimizationContext,
    TuningObservation,
    TuningPath,
)
from vcsel_ai_tuning.v2.continuous.controller import ContinuousOptimizationController
from vcsel_ai_tuning.v2.continuous.execution import OrchestratorExecutionPort
from vcsel_ai_tuning.v2.continuous.fake import SystemClock, fake_eye, fake_request
from vcsel_ai_tuning.v2.continuous.models import ContinuousState
from vcsel_ai_tuning.v2.continuous.store import ContinuousCampaignStore


CAMPAIGN_ID = "campaign-1"
NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.current = NOW
        self.elapsed_s = 0.0

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.elapsed_s

    def advance(self, seconds: float) -> None:
        self.elapsed_s += seconds
        self.current += timedelta(seconds=seconds)


class FakeDca:
    def __init__(
        self,
        measurements: Iterable[MeasurementResult] = (),
        *,
        identity_verified: bool = True,
        configuration_verified: bool = True,
        operation_log: list[str] | None = None,
    ) -> None:
        self.measurements = deque(measurements)
        self.identity_verified = identity_verified
        self.configuration_verified = configuration_verified
        self.calls: list[str] = []
        self.operation_log = operation_log

    def preflight(self) -> Mapping[str, object]:
        self.calls.append("preflight")
        return {
            "identity": "KEYSIGHT,N1092E,FAKE,1.0",
            "identity_verified": self.identity_verified,
            "configuration_verified": self.configuration_verified,
            "measurement_source": "CHAN1A",
        }

    def measure_single(self) -> MeasurementResult:
        self.calls.append("measure_single")
        if self.operation_log is not None:
            self.operation_log.append("dca.measure")
        if not self.measurements:
            raise AssertionError("fake DCA has no scripted measurement")
        return self.measurements.popleft()


class FakeI2c:
    def __init__(
        self,
        settings: Mapping[str, int],
        *,
        identity_verified: bool = True,
        configuration_verified: bool = True,
        operation_log: list[str] | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.identity_verified = identity_verified
        self.configuration_verified = configuration_verified
        self.calls: list[str] = []
        self.requests: list[dict[str, int]] = []
        self.applied: list[dict[str, int]] = []
        self.quantized_values: dict[tuple[str, int], int] = {}
        self.mismatch_after_writes: set[int] = set()
        self.operation_log = operation_log
        self.after_apply: Callable[[], None] | None = None

    def preflight(self) -> Mapping[str, object]:
        self.calls.append("preflight")
        return {
            "identity": "VD1124-FAKE",
            "identity_verified": self.identity_verified,
            "configuration_verified": self.configuration_verified,
            "device_address": 0xA6,
            "channel": 0,
        }

    def apply(self, requested: Mapping[str, int]) -> Mapping[str, int]:
        self.calls.append("apply")
        if self.operation_log is not None:
            self.operation_log.append("i2c.apply")
        request = dict(requested)
        applied = {
            name: self.quantized_values.get((name, value), value)
            for name, value in request.items()
        }
        self.requests.append(request)
        self.applied.append(applied)
        self.settings.update(applied)
        if self.after_apply is not None:
            self.after_apply()
        return applied

    def readback(self) -> Mapping[str, int]:
        self.calls.append("readback")
        result = dict(self.settings)
        if len(self.applied) in self.mismatch_after_writes:
            result["ibias_dac"] = result["ibias_dac"] + 1
        return result


class FakeBert:
    def __init__(
        self,
        pattern: str,
        fir: FirSettings,
        *,
        identity_verified: bool = True,
        configuration_verified: bool = True,
        operation_log: list[str] | None = None,
    ) -> None:
        self.pattern = pattern
        self.fir = fir
        self.identity_verified = identity_verified
        self.configuration_verified = configuration_verified
        self.calls: list[str] = []
        self.writes: list[tuple[str, FirSettings]] = []
        self.mismatch_after_writes: set[int] = set()
        self.operation_log = operation_log

    def preflight(self) -> Mapping[str, object]:
        self.calls.append("preflight")
        return {
            "identity": "VIT-BERT-FAKE",
            "identity_verified": self.identity_verified,
            "configuration_verified": self.configuration_verified,
            "port": "port1",
            "channel": "0",
        }

    def apply(self, pattern: str, fir: FirSettings) -> FirSettings:
        self.calls.append("apply")
        if self.operation_log is not None:
            self.operation_log.append("bert.apply")
        self.pattern = pattern
        self.fir = fir
        self.writes.append((pattern, fir))
        return fir

    def readback(self) -> Mapping[str, object]:
        self.calls.append("readback")
        fir = self.fir
        if len(self.writes) in self.mismatch_after_writes:
            changed = list(fir.taps)
            changed[0] += 1
            fir = fir.model_copy(update={"taps": tuple(changed)})
        return {"pattern": self.pattern, "fir": fir}


class ScriptedOptimizer:
    def __init__(self, *suggestions: object) -> None:
        self.suggestions = deque(suggestions)
        self.histories: list[tuple[TuningObservation, ...]] = []
        self.operation_log: list[str] | None = None

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[tuple[int, ...]] = frozenset(),
    ) -> object:
        del pending_keys
        if self.operation_log is not None:
            self.operation_log.append("optimizer.suggest")
        self.histories.append(observations)
        if not self.suggestions:
            raise AssertionError("fake optimizer has no scripted suggestion")
        return self.suggestions.popleft()

    @property
    def received_observations(self) -> tuple[TuningObservation, ...]:
        received: dict[str, TuningObservation] = {}
        for history in self.histories:
            for observation in history:
                received[observation.observation_id] = observation
        return tuple(received.values())


class FakeFirProposer:
    def __init__(self, full_target: FirSettings) -> None:
        self.full_target = full_target
        self.calls: list[FirSettings] = []

    def propose(self, current_fir: FirSettings) -> FirSettings:
        self.calls.append(current_fir)
        return self.full_target


class RejectingRepository:
    def __init__(self) -> None:
        self.publish_attempts = 0

    def publish_observation(
        self,
        observation_id: str,
        payload: Any,
        artifacts: tuple[ArtifactSpec, ...] | list[ArtifactSpec],
    ) -> None:
        del observation_id, payload, artifacts
        self.publish_attempts += 1
        raise RepositoryError("injected persistence failure")


@dataclass
class Harness:
    orchestrator: CampaignOrchestrator
    store: CampaignStore
    repository: CampaignRepository | RejectingRepository
    dca: FakeDca
    i2c: FakeI2c
    bert: FakeBert
    optimizer: ScriptedOptimizer
    clock: FakeClock
    baseline: BaselineConfig
    baseline_hash: str
    policy: object | None
    fir_proposer: FakeFirProposer | None
    operation_log: list[str]


@pytest.fixture
def harness_factory(tmp_path):
    resources: list[tuple[CampaignStore, CampaignRepository | None]] = []
    counter = 0

    def create(
        state: CampaignState,
        *,
        measurements: Iterable[MeasurementResult] = (),
        optimizer: ScriptedOptimizer | None = None,
        repository: CampaignRepository | RejectingRepository | None = None,
        i2c_settings: Mapping[str, int] | None = None,
        fir: FirSettings | None = None,
        pattern: str | None = None,
        policy: object | None = None,
        fir_proposer: FakeFirProposer | None = None,
        parameter_specs: tuple[ParameterSpec, ...] = (),
        tdecq_only: bool = False,
        continuous_i2c_only: bool = False,
    ) -> Harness:
        nonlocal counter
        counter += 1
        run_root = tmp_path / f"run-{counter}"
        run_root.mkdir()
        store = CampaignStore(run_root / "campaign.sqlite3")
        store.save_state(CAMPAIGN_ID, state)
        actual_repository: CampaignRepository | RejectingRepository
        owned_repository: CampaignRepository | None
        if repository is None:
            owned_repository = CampaignRepository(run_root / "artifacts", CAMPAIGN_ID)
            actual_repository = owned_repository
        else:
            owned_repository = None
            actual_repository = repository
        resources.append((store, owned_repository))

        baseline = load_b0()
        baseline_hash = canonical_json_sha256(baseline)
        clock = FakeClock()
        operation_log: list[str] = []
        dca = FakeDca(measurements, operation_log=operation_log)
        i2c = FakeI2c(
            i2c_settings or baseline.i2c.model_dump(),
            operation_log=operation_log,
        )
        bert = FakeBert(
            pattern or baseline.pattern,
            fir or baseline.fir,
            operation_log=operation_log,
        )
        actual_optimizer = optimizer or ScriptedOptimizer(
            NoSafeCandidate(
                path=TuningPath.TDECQ_BAYESIAN,
                reason="scripted stop",
            )
        )
        context = OptimizationContext(
            driver_profile_hash="b" * 64,
            vcsel_model="APA4701",
            device_id="vcsel-fake-1",
            test_context_id="fake-53g-ssprq-25c",
        )
        orchestrator_arguments = dict(
            campaign_id=CAMPAIGN_ID,
            baseline=baseline,
            baseline_hash=baseline_hash,
            optimization_context=context,
            store=store,
            repository=actual_repository,
            optimizer=actual_optimizer,
            dca=dca,
            i2c=i2c,
            bert=bert,
            clock=clock,
            parameter_specs=parameter_specs,
            tdecq_only=tdecq_only,
            continuous_i2c_only=continuous_i2c_only,
        )
        if policy is not None:
            orchestrator_arguments["policy"] = policy
        if fir_proposer is not None:
            orchestrator_arguments["fir_proposer"] = fir_proposer
        orchestrator = CampaignOrchestrator(**orchestrator_arguments)
        return Harness(
            orchestrator=orchestrator,
            store=store,
            repository=actual_repository,
            dca=dca,
            i2c=i2c,
            bert=bert,
            optimizer=actual_optimizer,
            clock=clock,
            baseline=baseline,
            baseline_hash=baseline_hash,
            policy=policy,
            fir_proposer=fir_proposer,
            operation_log=operation_log,
        )

    yield create

    for store, repository in reversed(resources):
        if repository is not None:
            repository.close()
        store.close()


def _measurement(
    label: str,
    *,
    acquisition_valid: bool = True,
    er_db: float | None = 3.2,
    er_status: str = "CORR",
    tdecq_db: float | None = 2.4,
    tdecq_status: str = "CORR",
    tdecq_invalid_reason: str | None = None,
    elapsed_s: float = 0.2,
    continuous_evidence: Mapping[str, object] | None = None,
) -> MeasurementResult:
    raw_responses: dict[str, object] = {
        "label": label,
        "er_status": er_status,
        "tdecq_status": tdecq_status,
    }
    if continuous_evidence is not None:
        raw_responses["continuous_eye_evidence"] = dict(continuous_evidence)
    return MeasurementResult(
        acquisition_valid=acquisition_valid,
        er_db=er_db,
        er_status=er_status,
        tdecq_db=tdecq_db,
        tdecq_status=tdecq_status,
        tdecq_invalid_reason=tdecq_invalid_reason,
        elapsed_s=elapsed_s,
        raw_responses=raw_responses,
        artifacts=(
            ArtifactSpec.from_json(
                "raw_responses",
                f"{label}.json",
                {"label": label, "er": er_db, "tdecq": tdecq_db},
            ),
        ),
    )


def _rollback_snapshot(
    baseline: BaselineConfig,
    *,
    i2c: Mapping[str, int] | None = None,
    fir: FirSettings | None = None,
) -> AppliedSettings:
    return AppliedSettings(
        readback_i2c=i2c or baseline.i2c.model_dump(),
        readback_fir=fir or baseline.fir,
    )


def _i2c_authorization(
    harness: Harness,
    stage: CampaignState,
    *,
    rollback_snapshot: AppliedSettings | None = None,
    max_candidates: int = 6,
    max_duration_s: float = 120.0,
) -> StageAuthorization:
    rollback = rollback_snapshot or _rollback_snapshot(harness.baseline)
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=AuthorizationMode.PER_STAGE,
        stage=stage,
        baseline_hash=harness.baseline_hash,
        rollback_hash=rollback_hash_for(rollback),
        parameter_specs=(
            ParameterSpec(
                name="ibias_dac",
                encoding_min=0,
                encoding_max=127,
                tuning_min=68,
                tuning_max=74,
                max_step=2,
            ),
        ),
        max_candidates=max_candidates,
        max_duration_s=max_duration_s,
        issued_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=30),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def _continuous_unbounded_authorization(
    harness: Harness,
    stage: CampaignState,
    *,
    max_candidates: int = 3,
) -> StageAuthorization:
    rollback = _rollback_snapshot(harness.baseline)
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=AuthorizationMode.PER_STAGE,
        time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
        stage=stage,
        baseline_hash=harness.baseline_hash,
        rollback_hash=rollback_hash_for(rollback),
        parameter_specs=(
            ParameterSpec(
                name="ibias_dac",
                encoding_min=0,
                encoding_max=127,
                tuning_min=68,
                tuning_max=74,
                max_step=2,
            ),
        ),
        max_candidates=max_candidates,
        max_duration_s=None,
        issued_at=NOW,
        expires_at=None,
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def _fir_authorization(harness: Harness) -> StageAuthorization:
    rollback = _rollback_snapshot(harness.baseline)
    fir_spec = FirAuthorizationSpec(
        tap_min=tuple(value - 4 for value in harness.baseline.fir.taps),
        tap_max=tuple(value + 4 for value in harness.baseline.fir.taps),
        max_step=(2, 2, 2, 2, 2, 2, 2),
        max_total_change_l1=4,
        fixed_upper_nlc=6,
        fixed_lower_nlc=6,
    )
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.FIR_APPLY,
        baseline_hash=harness.baseline_hash,
        rollback_hash=rollback_hash_for(rollback),
        fir_spec=fir_spec,
        max_candidates=3,
        max_duration_s=120.0,
        issued_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=30),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def _candidate(stage: CampaignState, *, value: int = 71) -> Candidate:
    return Candidate(
        candidate_id=f"{stage.value.lower()}-candidate-1",
        stage=stage,
        requested_i2c={"ibias_dac": value},
        model_version="optimizer-fake-1",
        reason="scripted candidate",
    )


def _suggestion(*, candidate_id: str = "candidate-1", value: int = 71):
    return CandidateSuggestion(
        path=TuningPath.TDECQ_BAYESIAN,
        candidate_id=candidate_id,
        requested={"ibias_dac": value},
        reason="scripted candidate",
    )


def _stop(path: TuningPath = TuningPath.TDECQ_BAYESIAN):
    return NoSafeCandidate(path=path, reason="scripted stage stop")


def _capture_optimizer_and_persistence_order(
    harness: Harness,
    monkeypatch: Any,
) -> None:
    assert isinstance(harness.repository, CampaignRepository)
    harness.optimizer.operation_log = harness.operation_log
    publish_observation = harness.repository.publish_observation

    def publish_and_log(
        observation_id: str,
        payload: Any,
        artifacts: tuple[ArtifactSpec, ...] | list[ArtifactSpec],
    ) -> Any:
        published = publish_observation(observation_id, payload, artifacts)
        harness.operation_log.append("observation.persisted")
        return published

    monkeypatch.setattr(harness.repository, "publish_observation", publish_and_log)


def _event_types(harness: Harness) -> list[str]:
    return [event.event_type for event in harness.store.events(CAMPAIGN_ID)]


def _transition_targets(harness: Harness) -> list[CampaignState]:
    return [
        event.to_state
        for event in harness.store.events(CAMPAIGN_ID)
        if event.from_state != event.to_state
    ]


def test_continuous_i2c_candidate_reuses_authorized_execution_without_bert(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_FINE,
        measurements=(_measurement("continuous"),),
        tdecq_only=True,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    observations = harness.orchestrator.run_continuous_i2c_candidate(
        authorization,
        _candidate(CampaignState.I2C_FINE, value=71),
    )

    assert len(observations) == 1
    assert observations[0].settings.requested_i2c == {"ibias_dac": 71}
    assert observations[0].settings.readback_i2c["ibias_dac"] == 71
    assert harness.i2c.requests == [{"ibias_dac": 71}]
    assert harness.dca.calls == ["measure_single"]
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_invalid_candidate_is_persisted_as_invalid_evidence(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_FINE,
        measurements=(
            _measurement(
                "continuous-invalid",
                tdecq_db=None,
                tdecq_status="INV",
                tdecq_invalid_reason="The target SER is lower than actual SER.",
            ),
        ),
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    observations = harness.orchestrator.run_continuous_i2c_candidate(
        authorization,
        _candidate(CampaignState.I2C_FINE, value=71),
    )

    assert len(observations) == 1
    assert observations[0].valid is False
    published = harness.repository.list_observations()
    assert published[-1].payload["valid"] is False
    assert published[-1].payload["metric_status"] == {
        "er_status": "CORR",
        "tdecq_status": "INV",
        "tdecq_invalid_reason": "The target SER is lower than actual SER.",
    }
    assert harness.i2c.requests == [{"ibias_dac": 71}]


def test_continuous_current_state_measurement_is_read_only(harness_factory) -> None:
    harness = harness_factory(
        CampaignState.I2C_FINE,
        measurements=(_measurement("acceptance"),),
        tdecq_only=True,
    )

    observation = harness.orchestrator.measure_current_i2c_state()

    assert observation.settings.readback_i2c == harness.i2c.settings
    assert harness.i2c.requests == []
    assert harness.dca.calls == ["measure_single"]
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_preflight_bootstraps_new_campaign_without_bert_or_i2c_write(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.DRAFT,
        measurements=(_measurement("continuous-preflight"),),
        tdecq_only=True,
    )

    result = harness.orchestrator.capture_continuous_preflight_observation()

    assert result.verified is True
    assert result.observation.settings.readback_i2c == harness.i2c.settings
    assert result.observation.stage is CampaignState.PREFLIGHT
    assert harness.store.current_state(CAMPAIGN_ID) is (
        CampaignState.AWAIT_BASELINE_APPROVAL
    )
    assert harness.dca.calls == ["preflight", "measure_single"]
    assert harness.i2c.calls == ["preflight", "readback"]
    assert harness.i2c.requests == []
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_preflight_records_dca_setup_failure_before_other_contact(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.DRAFT,
        measurements=(_measurement("must-not-run"),),
        tdecq_only=True,
    )

    def fail_dca_preflight() -> Mapping[str, object]:
        harness.dca.calls.append("preflight")
        raise RuntimeError("FUNC1 setup rejected")

    harness.dca.preflight = fail_dca_preflight

    with pytest.raises(PreflightError, match="DCA preflight failed"):
        harness.orchestrator.capture_continuous_preflight_observation()

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED
    assert _event_types(harness)[-1] == "PREFLIGHT_FAILED"
    assert harness.dca.calls == ["preflight"]
    assert harness.i2c.calls == []
    assert harness.i2c.requests == []
    assert harness.bert.calls == []
    assert harness.bert.writes == []
    assert harness.operation_log == []


def test_continuous_preflight_records_i2c_readback_failure_before_acquisition(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.DRAFT,
        measurements=(_measurement("must-not-run"),),
        tdecq_only=True,
    )

    def fail_i2c_readback() -> Mapping[str, int]:
        harness.i2c.calls.append("readback")
        raise RuntimeError("readback rejected")

    harness.i2c.readback = fail_i2c_readback

    with pytest.raises(PreflightError, match="I2C readback failed"):
        harness.orchestrator.capture_continuous_preflight_observation()

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED
    assert _event_types(harness)[-1] == "PREFLIGHT_FAILED"
    assert harness.dca.calls == ["preflight"]
    assert harness.i2c.calls == ["preflight", "readback"]
    assert harness.i2c.requests == []
    assert harness.bert.calls == []
    assert harness.bert.writes == []
    assert harness.operation_log == []


def test_continuous_preflight_reports_invalid_tdecq_raw_status(harness_factory) -> None:
    measurement = MeasurementResult(
        acquisition_valid=True,
        er_db=2.84,
        er_status="CORR",
        tdecq_db=None,
        tdecq_status="INV",
        tdecq_invalid_reason="TDECQ status is INV: Target SER lower than actual",
        elapsed_s=3.0,
        raw_responses={
            "status_polls": [
                {"er_status": "CORR", "tdecq_status": "INV"},
            ]
            * 3,
            "er_details_polls": [],
            "tdecq_details_polls": ["Target SER lower than actual"] * 3,
        },
    )
    harness = harness_factory(
        CampaignState.DRAFT,
        measurements=(measurement,),
        tdecq_only=True,
    )

    with pytest.raises(PreflightError) as error:
        harness.orchestrator.capture_continuous_preflight_observation()

    message = str(error.value)
    assert "er_status='CORR'" in message
    assert "tdecq_status='INV'" in message
    assert "Target SER lower than actual" in message
    assert "status_polls=" in message
    assert "er_details_polls=[]" in message
    assert "tdecq_details_polls=" in message


def test_continuous_preflight_records_persistence_failure_as_failed_campaign(
    harness_factory,
) -> None:
    repository = RejectingRepository()
    harness = harness_factory(
        CampaignState.DRAFT,
        measurements=(_measurement("persistence-failure"),),
        repository=repository,
        tdecq_only=True,
    )

    with pytest.raises(RepositoryError, match="persistence failure"):
        harness.orchestrator.capture_continuous_preflight_observation()

    assert repository.publish_attempts == 1
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED
    assert _event_types(harness)[-2:] == [
        "OBSERVATION_PERSISTENCE_FAILED",
        "PREFLIGHT_FAILED",
    ]
    assert harness.i2c.requests == []
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_preflight_can_repeat_without_advancing_or_writing(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        measurements=(_measurement("continuous-preflight-repeat"),),
        tdecq_only=True,
    )

    result = harness.orchestrator.capture_continuous_preflight_observation()

    assert result.verified is True
    assert harness.store.current_state(CAMPAIGN_ID) is (
        CampaignState.AWAIT_BASELINE_APPROVAL
    )
    assert harness.i2c.requests == []
    assert harness.bert.calls == []
    assert "CONTINUOUS_PREFLIGHT_COMPLETED" in _event_types(harness)


def test_continuous_static_b0_restore_is_i2c_only(harness_factory) -> None:
    harness = harness_factory(
        CampaignState.I2C_FINE,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
    )

    restored = harness.orchestrator.restore_static_i2c_b0()

    expected = harness.baseline.i2c.model_dump()
    assert restored.requested_i2c == expected
    assert restored.applied_i2c == expected
    assert restored.readback_i2c == expected
    assert harness.i2c.requests == [expected]
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_static_b0_restore_completes_active_screening_campaign(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
        continuous_i2c_only=True,
    )

    harness.orchestrator.restore_static_i2c_b0()

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.COMPLETED
    assert _event_types(harness)[-1] == "CONTINUOUS_B0_RESTORED"
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_static_b0_restore_failure_marks_recovery_required(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    harness.i2c.mismatch_after_writes.add(1)

    with pytest.raises(OrchestratorError, match="readback mismatch"):
        harness.orchestrator.restore_static_i2c_b0()

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.RECOVERY_REQUIRED
    assert _event_types(harness)[-1] == "CONTINUOUS_B0_RESTORE_FAILED"


def test_continuous_commission_preparation_is_i2c_only_and_enters_screening(
    harness_factory,
) -> None:
    baseline = load_b0()
    previous = baseline.i2c.model_copy(update={"eq": 21}).model_dump()
    specs = (
        ParameterSpec(
            name="eq",
            encoding_min=0,
            encoding_max=31,
            tuning_min=0,
            tuning_max=31,
            max_step=5,
            enabled=True,
        ),
    )
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        measurements=(
            _measurement(
                "continuous-commission-baseline",
                continuous_evidence={"measurement_id": "baseline-eye-1"},
            ),
        ),
        i2c_settings=previous,
        parameter_specs=specs,
        tdecq_only=True,
        continuous_i2c_only=True,
    )

    prepared = harness.orchestrator.prepare_continuous_commission(
        max_candidates=20,
    )

    assert prepared.observation.settings.readback_i2c == baseline.i2c.model_dump()
    assert prepared.observation.valid is True
    assert prepared.observation.evidence == {"measurement_id": "baseline-eye-1"}
    assert prepared.authorization.stage is CampaignState.I2C_SCREENING
    assert prepared.authorization.max_candidates == 20
    assert (
        prepared.authorization.time_policy
        is AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED
    )
    assert prepared.authorization.max_duration_s is None
    assert prepared.authorization.expires_at is None
    assert not hasattr(prepared, "deadline")
    assert prepared.authorization.rollback_snapshot.readback_i2c == (
        baseline.i2c.model_dump()
    )
    assert prepared.authorization.rollback_snapshot.readback_fir is None
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.I2C_SCREENING
    assert harness.i2c.requests == [baseline.i2c.model_dump()]
    assert harness.dca.calls == ["measure_single"]
    assert harness.bert.calls == []
    assert harness.bert.writes == []

    planned = [
        event.payload["authorization"]
        for event in harness.store.events(CAMPAIGN_ID)
        if event.event_type == "STAGE_PLANNED"
    ]
    assert [authorization["stage"] for authorization in planned] == [
        CampaignState.RESTORE_B0.value,
        CampaignState.I2C_SCREENING.value,
    ]
    assert all(
        authorization["time_policy"]
        == AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED.value
        for authorization in planned
    )
    assert all(authorization["max_duration_s"] is None for authorization in planned)
    assert all(authorization["expires_at"] is None for authorization in planned)
    assert planned[-1] == prepared.authorization.model_dump(mode="json")


@pytest.mark.parametrize("max_candidates", [1, 10_000])
def test_continuous_commission_accepts_any_positive_integer_candidate_budget(
    harness_factory,
    max_candidates,
) -> None:
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        measurements=(_measurement(f"continuous-budget-{max_candidates}"),),
        parameter_specs=(
            ParameterSpec(
                name="eq",
                encoding_min=0,
                encoding_max=31,
                tuning_min=0,
                tuning_max=31,
                max_step=5,
            ),
        ),
        tdecq_only=True,
        continuous_i2c_only=True,
    )

    prepared = harness.orchestrator.prepare_continuous_commission(
        max_candidates=max_candidates,
    )

    assert prepared.authorization.max_candidates == max_candidates


@pytest.mark.parametrize("max_candidates", [True, False, 0, -1, 1.0, 1.5])
def test_continuous_commission_rejects_non_positive_or_non_integer_candidate_budget(
    harness_factory,
    max_candidates,
) -> None:
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        continuous_i2c_only=True,
    )

    with pytest.raises(OrchestratorError, match="positive integer"):
        harness.orchestrator.prepare_continuous_commission(
            max_candidates=max_candidates,
        )

    assert harness.i2c.requests == []
    assert harness.dca.calls == []


def test_non_continuous_orchestrator_cannot_plan_unbounded_stage(
    harness_factory,
) -> None:
    harness = harness_factory(CampaignState.AWAIT_BASELINE_APPROVAL)

    with pytest.raises(OrchestratorError, match="I2C-only orchestrator"):
        harness.orchestrator._plan_continuous_unbounded_stage(
            CampaignState.RESTORE_B0,
            max_candidates=1,
        )

    assert harness.i2c.calls == []
    assert harness.bert.calls == []


@pytest.mark.parametrize(
    ("initial_state", "stage"),
    [
        (CampaignState.AWAIT_BASELINE_APPROVAL, CampaignState.RESTORE_B0),
        (CampaignState.AWAIT_COARSE_APPROVAL, CampaignState.I2C_SCREENING),
    ],
)
def test_api_stage_execution_rejects_persisted_unbounded_authorization_for_non_continuous_orchestrator(
    harness_factory,
    initial_state,
    stage,
) -> None:
    harness = harness_factory(initial_state)
    authorization = _continuous_unbounded_authorization(harness, stage)
    harness.store.append_event(
        CAMPAIGN_ID,
        "STAGE_PLANNED",
        payload={"authorization": authorization.model_dump(mode="json")},
        expected_state=initial_state,
        created_at=NOW,
    )

    with pytest.raises(OrchestratorError, match="unbounded authorization requires"):
        ConcreteCampaignService._execute_authorized_stage(
            harness.orchestrator,
            authorization,
        )

    assert harness.store.current_state(CAMPAIGN_ID) is initial_state
    assert harness.i2c.requests == []
    assert harness.bert.writes == []
    assert not any(
        event.event_type in {"AUTHORIZED_STAGE_ENTERED", "B0_RESTORE_STARTED"}
        for event in harness.store.events(CAMPAIGN_ID)
    )
    with pytest.raises(StoreError, match="authorization"):
        harness.store.authorization_usage(
            CAMPAIGN_ID,
            authorization.authorization_id,
            now=NOW,
        )


def test_active_unbounded_authorization_is_rejected_before_candidate_reservation(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        measurements=(_measurement("must-not-run"),),
    )
    authorization = _continuous_unbounded_authorization(
        harness,
        CampaignState.I2C_SCREENING,
    )
    harness.store.activate_authorization(
        CAMPAIGN_ID,
        authorization,
        activated_at=NOW,
    )

    with pytest.raises(OrchestratorError, match="unbounded authorization requires"):
        harness.orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_SCREENING),
        )

    usage = harness.store.authorization_usage(
        CAMPAIGN_ID,
        authorization.authorization_id,
        now=NOW,
    )
    assert usage.candidate_count == 0
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.I2C_SCREENING
    assert harness.i2c.calls == []
    assert harness.bert.calls == []
    assert harness.dca.calls == []


def test_public_continuous_failure_recovery_requires_continuous_orchestrator(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
    )

    with pytest.raises(OrchestratorError, match="I2C-only orchestrator"):
        harness.orchestrator.recover_continuous_commission_failure()

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.I2C_SCREENING
    assert harness.i2c.requests == []


def test_public_continuous_failure_recovery_restores_b0_and_marks_failed(
    harness_factory,
) -> None:
    baseline = load_b0()
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
        continuous_i2c_only=True,
    )

    restored = harness.orchestrator.recover_continuous_commission_failure()

    assert restored.readback_i2c == baseline.i2c.model_dump()
    assert harness.i2c.settings == baseline.i2c.model_dump()
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED
    assert _event_types(harness)[-1] == "CONTINUOUS_COMMISSION_FAILED_SAFE"


def test_public_continuous_failure_recovery_does_not_retry_recovery_required(
    harness_factory,
) -> None:
    harness = harness_factory(
        CampaignState.RECOVERY_REQUIRED,
        tdecq_only=True,
        continuous_i2c_only=True,
    )

    with pytest.raises(RecoveryRequiredError, match="RECOVERY_REQUIRED"):
        harness.orchestrator.recover_continuous_commission_failure()

    assert harness.i2c.requests == []
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.RECOVERY_REQUIRED


def test_continuous_failure_recovery_retries_after_transition_persistence_failure(
    harness_factory,
    monkeypatch,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    save_transition = harness.store.save_transition
    failed_once = False

    def fail_first_recovery_transition(
        campaign_id,
        from_state,
        to_state,
        **kwargs,
    ):
        nonlocal failed_once
        if not failed_once and to_state is CampaignState.ABORTING:
            failed_once = True
            raise StoreError("injected first recovery transition persistence failure")
        return save_transition(campaign_id, from_state, to_state, **kwargs)

    monkeypatch.setattr(
        harness.store,
        "save_transition",
        fail_first_recovery_transition,
    )

    with pytest.raises(StoreError, match="first recovery transition"):
        harness.orchestrator.recover_continuous_commission_failure()

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.I2C_SCREENING
    assert harness.store.current_state(CAMPAIGN_ID) is not CampaignState.COMPLETED

    restored = harness.orchestrator.recover_continuous_commission_failure()

    expected = harness.baseline.i2c.model_dump()
    assert restored.readback_i2c == expected
    assert harness.i2c.requests == [expected, expected]
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED


@pytest.mark.parametrize(
    ("failed_transition", "persisted_state"),
    [
        (CampaignState.ROLLING_BACK, CampaignState.ABORTING),
        (CampaignState.FAILED, CampaignState.ROLLING_BACK),
    ],
)
def test_continuous_failure_recovery_resumes_after_later_transition_failure(
    harness_factory,
    monkeypatch,
    failed_transition,
    persisted_state,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    save_transition = harness.store.save_transition
    failed_once = False

    def fail_selected_recovery_transition(
        campaign_id,
        from_state,
        to_state,
        **kwargs,
    ):
        nonlocal failed_once
        if not failed_once and to_state is failed_transition:
            failed_once = True
            raise StoreError(
                f"injected {failed_transition.value} transition persistence failure"
            )
        return save_transition(campaign_id, from_state, to_state, **kwargs)

    monkeypatch.setattr(
        harness.store,
        "save_transition",
        fail_selected_recovery_transition,
    )

    with pytest.raises(StoreError, match="transition persistence failure"):
        harness.orchestrator.recover_continuous_commission_failure()

    assert harness.store.current_state(CAMPAIGN_ID) is persisted_state
    assert harness.store.current_state(CAMPAIGN_ID) is not CampaignState.COMPLETED

    restored = harness.orchestrator.recover_continuous_commission_failure()

    expected = harness.baseline.i2c.model_dump()
    assert restored.readback_i2c == expected
    assert harness.i2c.requests == [expected, expected]
    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED


@pytest.mark.parametrize(
    ("recovery_mismatch", "continuous_state", "campaign_state", "b0_restored"),
    [
        (False, ContinuousState.FAILED_SAFE, CampaignState.FAILED, True),
        (
            True,
            ContinuousState.RECOVERY_REQUIRED,
            CampaignState.RECOVERY_REQUIRED,
            False,
        ),
    ],
)
def test_continuous_controller_and_campaign_store_agree_after_failure_recovery(
    harness_factory,
    tmp_path,
    recovery_mismatch,
    continuous_state,
    campaign_state,
    b0_restored,
) -> None:
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        i2c_settings={"ibias_dac": 72, "eq": 23},
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    if recovery_mismatch:
        harness.i2c.mismatch_after_writes.add(1)

    class FailingProvider:
        provider_id = "failing-provider"
        provider_version = "1"

        @staticmethod
        def decide(context):
            del context
            raise RuntimeError("injected provider failure")

    result_root = tmp_path / ("mismatch" if recovery_mismatch else "verified")
    controller_store = ContinuousCampaignStore(result_root / "continuous")
    request = fake_request(
        result_root,
        fake_eye(
            result_root,
            "baseline",
            readback=harness.baseline.i2c.model_dump(),
            tdecq_db=3.2,
        ),
        max_rounds=1,
    )
    controller = ContinuousOptimizationController(
        provider=FailingProvider(),
        execution=OrchestratorExecutionPort(harness.orchestrator),
        store=controller_store,
        clock=SystemClock(),
    )

    result = controller.start(request)

    assert result.state is continuous_state
    assert result.b0_restored is b0_restored
    assert controller_store.load_state(CAMPAIGN_ID)["state"] == continuous_state.value
    assert harness.store.current_state(CAMPAIGN_ID) is campaign_state
    assert harness.store.current_state(CAMPAIGN_ID) is not CampaignState.COMPLETED
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.i2c.calls[-2:] == ["apply", "readback"]


def test_continuous_commission_baseline_failure_restores_b0_without_bert(
    harness_factory,
) -> None:
    baseline = load_b0()
    previous = baseline.i2c.model_copy(update={"eq": 21}).model_dump()
    specs = (
        ParameterSpec(
            name="eq",
            encoding_min=0,
            encoding_max=31,
            tuning_min=0,
            tuning_max=31,
            max_step=5,
            enabled=True,
        ),
    )
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        i2c_settings=previous,
        parameter_specs=specs,
        tdecq_only=True,
        continuous_i2c_only=True,
    )

    with pytest.raises(Exception, match="measurement"):
        harness.orchestrator.prepare_continuous_commission(
            max_candidates=3,
        )

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.FAILED
    assert harness.i2c.settings == baseline.i2c.model_dump()
    assert harness.i2c.requests == [
        baseline.i2c.model_dump(),
        baseline.i2c.model_dump(),
    ]
    assert harness.bert.calls == []
    assert harness.bert.writes == []


def test_continuous_commission_marks_recovery_when_b0_recovery_cannot_be_verified(
    harness_factory,
) -> None:
    baseline = load_b0()
    previous = baseline.i2c.model_copy(update={"eq": 21}).model_dump()
    specs = (
        ParameterSpec(
            name="eq",
            encoding_min=0,
            encoding_max=31,
            tuning_min=0,
            tuning_max=31,
            max_step=5,
            enabled=True,
        ),
    )
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        i2c_settings=previous,
        parameter_specs=specs,
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    harness.i2c.mismatch_after_writes.add(2)

    with pytest.raises(RecoveryRequiredError, match="could not be restored"):
        harness.orchestrator.prepare_continuous_commission(
            max_candidates=3,
        )

    assert harness.store.current_state(CAMPAIGN_ID) is CampaignState.RECOVERY_REQUIRED


@pytest.mark.parametrize(
    ("failed_transition", "residual_state", "expected_final_state"),
    [
        (
            CampaignState.ABORTING,
            CampaignState.BASELINE_MEASUREMENT,
            CampaignState.FAILED,
        ),
        (
            CampaignState.ROLLING_BACK,
            CampaignState.ABORTING,
            CampaignState.FAILED,
        ),
        (
            CampaignState.FAILED,
            CampaignState.ROLLING_BACK,
            CampaignState.FAILED,
        ),
        (
            CampaignState.RECOVERY_REQUIRED,
            CampaignState.BASELINE_MEASUREMENT,
            CampaignState.RECOVERY_REQUIRED,
        ),
    ],
)
def test_continuous_commission_preparation_recovery_is_durably_resumable(
    harness_factory,
    monkeypatch,
    failed_transition,
    residual_state,
    expected_final_state,
) -> None:
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        tdecq_only=True,
        continuous_i2c_only=True,
    )
    if failed_transition is CampaignState.RECOVERY_REQUIRED:
        harness.i2c.mismatch_after_writes = {2, 3}
    save_transition = harness.store.save_transition
    failed_once = False

    def fail_selected_recovery_transition(
        campaign_id,
        from_state,
        to_state,
        **kwargs,
    ):
        nonlocal failed_once
        if not failed_once and to_state is failed_transition:
            failed_once = True
            raise StoreError(
                f"injected {failed_transition.value} transition persistence failure"
            )
        return save_transition(campaign_id, from_state, to_state, **kwargs)

    monkeypatch.setattr(
        harness.store,
        "save_transition",
        fail_selected_recovery_transition,
    )

    with pytest.raises(StoreError, match="transition persistence failure"):
        harness.orchestrator.prepare_continuous_commission(max_candidates=3)

    assert harness.store.current_state(CAMPAIGN_ID) is residual_state
    assert harness.store.current_state(CAMPAIGN_ID) is not CampaignState.COMPLETED

    if expected_final_state is CampaignState.RECOVERY_REQUIRED:
        with pytest.raises(RecoveryRequiredError, match="could not be restored"):
            harness.orchestrator.recover_continuous_commission_failure()
    else:
        harness.orchestrator.recover_continuous_commission_failure()

    expected = harness.baseline.i2c.model_dump()
    assert harness.i2c.requests == [expected, expected, expected]
    assert harness.store.current_state(CAMPAIGN_ID) is expected_final_state
    assert harness.store.current_state(CAMPAIGN_ID) is not CampaignState.COMPLETED


def _rebuild_orchestrator(harness: Harness) -> CampaignOrchestrator:
    arguments = dict(
        campaign_id=CAMPAIGN_ID,
        baseline=harness.baseline,
        baseline_hash=harness.baseline_hash,
        optimization_context=harness.orchestrator.optimization_context,
        store=harness.store,
        repository=harness.repository,
        optimizer=harness.optimizer,
        dca=harness.dca,
        i2c=harness.i2c,
        bert=harness.bert,
        clock=harness.clock,
    )
    if harness.policy is not None:
        arguments["policy"] = harness.policy
    if harness.fir_proposer is not None:
        arguments["fir_proposer"] = harness.fir_proposer
    return CampaignOrchestrator(**arguments)


def test_preflight_is_read_only_and_persists_verified_identity_configuration_and_readback(
    harness_factory,
):
    harness = harness_factory(CampaignState.PREFLIGHT)

    result = harness.orchestrator.preflight()

    assert result.verified is True
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.AWAIT_BASELINE_APPROVAL
    )
    assert harness.dca.calls == ["preflight"]
    assert harness.i2c.calls == ["preflight", "readback"]
    assert harness.bert.calls == ["preflight", "readback"]
    assert harness.i2c.requests == []
    assert harness.bert.writes == []
    assert "PREFLIGHT_COMPLETED" in _event_types(harness)


def test_preflight_rejects_identity_or_configuration_mismatch_without_any_write(
    harness_factory,
):
    harness = harness_factory(CampaignState.PREFLIGHT)
    harness.dca.configuration_verified = False

    with pytest.raises(PreflightError, match="DCA.*configuration"):
        harness.orchestrator.preflight()

    assert harness.i2c.requests == []
    assert harness.bert.writes == []
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED


def test_preflight_refuses_unresolved_recovery_before_contacting_hardware(
    harness_factory,
):
    harness = harness_factory(CampaignState.RECOVERY_REQUIRED)

    with pytest.raises(RecoveryRequiredError, match="RECOVERY_REQUIRED"):
        harness.orchestrator.preflight()

    assert harness.dca.calls == []
    assert harness.i2c.calls == []
    assert harness.bert.calls == []


def test_restore_b0_writes_exact_pattern_fir_nlc_and_i2c_then_verifies_readback(
    harness_factory,
):
    non_b0_i2c = load_b0().i2c.model_copy(update={"ibias_dac": 69}).model_dump()
    non_b0_fir = FirSettings(
        taps=(0, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.AWAIT_BASELINE_APPROVAL,
        i2c_settings=non_b0_i2c,
        fir=non_b0_fir,
    )
    rollback = _rollback_snapshot(
        harness.baseline,
        i2c=non_b0_i2c,
        fir=non_b0_fir,
    )
    authorization = _i2c_authorization(
        harness,
        CampaignState.RESTORE_B0,
        rollback_snapshot=rollback,
    )

    applied = harness.orchestrator.restore_b0(authorization)

    assert applied.readback_i2c == harness.baseline.i2c.model_dump()
    assert applied.readback_fir == harness.baseline.fir
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.bert.writes == [("SSPRQ", harness.baseline.fir)]
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.BASELINE_MEASUREMENT
    )
    assert "B0_READBACK_VERIFIED" in _event_types(harness)
    assert harness.store.authorization_usage(
        CAMPAIGN_ID,
        authorization.authorization_id,
        now=harness.clock.now(),
    ).candidate_count == 1


def test_baseline_runs_five_independent_singles_and_accepts_four_valid_er_results_when_tdecq_is_missing(
    harness_factory,
):
    measurements = [
        _measurement("baseline-1", er_db=3.0, tdecq_db=2.6),
        _measurement("baseline-2", er_db=3.1, tdecq_db=2.5),
        _measurement(
            "baseline-3",
            er_db=3.2,
            tdecq_db=None,
            tdecq_status="INV",
            tdecq_invalid_reason="not measurable",
        ),
        _measurement("baseline-4", er_db=3.3, tdecq_db=2.4),
        _measurement(
            "baseline-5",
            acquisition_valid=False,
            er_db=None,
            er_status="INV",
            tdecq_db=None,
            tdecq_status="INV",
            tdecq_invalid_reason="acquisition timeout",
        ),
    ]
    harness = harness_factory(
        CampaignState.BASELINE_MEASUREMENT,
        measurements=measurements,
    )

    report = harness.orchestrator.run_baseline_measurement()

    assert report.measurement_count == 5
    assert report.acquisition_valid_count == 4
    assert report.er_valid_count == 4
    assert report.tdecq_valid_count == 3
    assert report.valid is True
    assert report.er_median_db == pytest.approx(3.15)
    assert report.tdecq_median_db == pytest.approx(2.5)
    assert report.er_sample_std_db is not None
    assert report.raw_responses
    assert harness.dca.calls == ["measure_single"] * 5
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.AWAIT_COARSE_APPROVAL
    )


def test_baseline_does_not_pass_with_only_three_valid_acquisitions_and_er_results(
    harness_factory,
):
    measurements = [
        _measurement(f"baseline-valid-{index}") for index in range(3)
    ] + [
        _measurement(
            f"baseline-invalid-{index}",
            acquisition_valid=False,
            er_db=None,
            er_status="INV",
            tdecq_db=None,
            tdecq_status="INV",
            tdecq_invalid_reason="timeout",
        )
        for index in range(2)
    ]
    harness = harness_factory(
        CampaignState.BASELINE_MEASUREMENT,
        measurements=measurements,
    )

    report = harness.orchestrator.run_baseline_measurement()

    assert report.measurement_count == 5
    assert report.acquisition_valid_count == 3
    assert report.er_valid_count == 3
    assert report.valid is False
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.BASELINE_MEASUREMENT
    )
    assert _event_types(harness)[-1] == "BASELINE_INVALID"


def test_candidate_apply_records_requested_quantized_applied_readback_and_measurement_before_visibility(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement("candidate-quantized")],
    )
    harness.i2c.quantized_values[("ibias_dac", 71)] = 72
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    observations = harness.orchestrator.run_one_candidate(
        authorization,
        _candidate(CampaignState.I2C_COARSE),
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.settings.requested_i2c == {"ibias_dac": 71}
    assert observation.settings.quantized_i2c == {"ibias_dac": 72}
    assert observation.settings.applied_i2c == {"ibias_dac": 72}
    assert observation.settings.readback_i2c["ibias_dac"] == 72
    assert observation.metrics == {"er_db": 3.2, "tdecq_db": 2.4}
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.get_observation(observation.observation_id)
    assert published is not None
    assert published.payload["settings"]["readback_i2c"]["ibias_dac"] == 72
    assert _event_types(harness)[-1] == "OBSERVATION_PERSISTED"


def test_screening_runner_restores_stage_snapshot_before_another_direction(
    harness_factory,
):
    optimizer = ScriptedOptimizer(
        CandidateSuggestion(
            path=TuningPath.EXHAUSTIVE,
            candidate_id="screen-1",
            requested={"ibias_dac": 71},
            reason="positive direction",
        ),
        _stop(TuningPath.EXHAUSTIVE),
    )
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        measurements=[_measurement("screening-1")],
        optimizer=optimizer,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_SCREENING)

    summary = harness.orchestrator.run_screening_stage(authorization)

    assert summary.stage == CampaignState.I2C_SCREENING
    assert summary.candidates_used == 1
    assert harness.i2c.requests[0] == {"ibias_dac": 71}
    assert harness.i2c.requests[1] == harness.baseline.i2c.model_dump()
    assert harness.i2c.readback() == harness.baseline.i2c.model_dump()
    assert len(optimizer.received_observations) == 1
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_SCREENING


def test_tdecq_only_screening_persists_valid_tdecq_independently_of_er(
    harness_factory,
):
    optimizer = ScriptedOptimizer(
        CandidateSuggestion(
            path=TuningPath.EXHAUSTIVE,
            candidate_id="tdecq-only",
            requested={"ibias_dac": 71},
            reason="ER invariant diagnostic path",
        ),
        _stop(TuningPath.EXHAUSTIVE),
    )
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        measurements=[
            _measurement(
                "tdecq-only",
                er_db=None,
                er_status="INV",
                tdecq_db=2.1,
            )
        ],
        optimizer=optimizer,
        tdecq_only=True,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_SCREENING)

    summary = harness.orchestrator.run_screening_stage(authorization)

    assert summary.best_observation is not None
    assert summary.best_observation.metrics["tdecq_db"] == 2.1
    assert summary.best_observation.feasible is True
    assert optimizer.received_observations[0].er_db is None
    assert optimizer.received_observations[0].tdecq_db == 2.1


def test_screening_enters_coarse_only_with_a_separate_coarse_authorization(
    harness_factory,
):
    optimizer = ScriptedOptimizer(
        CandidateSuggestion(
            path=TuningPath.EXHAUSTIVE,
            candidate_id="screen-before-coarse",
            requested={"ibias_dac": 71},
            reason="screening direction",
        ),
        _stop(TuningPath.EXHAUSTIVE),
    )
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        measurements=[_measurement("screen-before-coarse")],
        optimizer=optimizer,
    )
    screening_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_SCREENING,
    )
    coarse_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_COARSE,
    )

    harness.orchestrator.run_screening_stage(
        screening_authorization,
        next_authorization=coarse_authorization,
    )

    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_COARSE


def test_screening_rejects_reusing_its_authorization_for_coarse_transition(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_stop(TuningPath.EXHAUSTIVE))
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        optimizer=optimizer,
    )
    screening_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_SCREENING,
    )

    with pytest.raises((AuthorizationError, OrchestratorError), match="I2C_COARSE|stage"):
        harness.orchestrator.run_screening_stage(
            screening_authorization,
            next_authorization=screening_authorization,
        )

    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_SCREENING
    assert harness.i2c.requests == []
    assert harness.bert.writes == []


def test_screening_restore_failure_traces_rollback_before_recovery_required(
    harness_factory,
):
    optimizer = ScriptedOptimizer(
        CandidateSuggestion(
            path=TuningPath.EXHAUSTIVE,
            candidate_id="screen-restore-failure",
            requested={"ibias_dac": 71},
            reason="screening restore failure",
        )
    )
    harness = harness_factory(
        CampaignState.I2C_SCREENING,
        optimizer=optimizer,
        measurements=[_measurement("screen-restore-failure")],
    )
    harness.i2c.mismatch_after_writes = {2}
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_SCREENING,
    )

    with pytest.raises(RecoveryRequiredError, match="restore|rollback"):
        harness.orchestrator.run_screening_stage(authorization)

    assert _transition_targets(harness)[-3:] == [
        CampaignState.ABORTING,
        CampaignState.ROLLING_BACK,
        CampaignState.RECOVERY_REQUIRED,
    ]
    assert len(harness.i2c.requests) == 2
    assert len(harness.bert.writes) == 1


def test_coarse_runner_repeats_each_candidate_three_times_and_passes_only_persisted_readbacks_to_optimizer(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_suggestion(candidate_id="coarse-1"), _stop())
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement(f"coarse-{index}") for index in range(3)],
        optimizer=optimizer,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    summary = harness.orchestrator.run_coarse_stage(authorization)

    assert summary.stage == CampaignState.I2C_COARSE
    assert summary.candidates_used == 1
    assert harness.dca.calls == ["measure_single"] * 3
    assert len(optimizer.received_observations) == 3
    assert all(
        observation.settings_readback["ibias_dac"] == 71
        for observation in optimizer.received_observations
    )
    assert isinstance(harness.repository, CampaignRepository)
    assert len(harness.repository.list_observations()) == 3


@pytest.mark.parametrize(
    ("state", "runner_name"),
    [
        (CampaignState.I2C_FINE, "run_fine_stage"),
        (CampaignState.I2C_RETUNE, "run_retune_stage"),
    ],
)
def test_fine_and_retune_runners_delegate_candidates_to_optimizer_without_optimizer_math(
    harness_factory, state, runner_name
):
    optimizer = ScriptedOptimizer(_suggestion(candidate_id=f"{state.value}-1"), _stop())
    harness = harness_factory(
        state,
        measurements=[_measurement(f"{state.value}-measurement")],
        optimizer=optimizer,
    )
    authorization = _i2c_authorization(harness, state)

    summary = getattr(harness.orchestrator, runner_name)(authorization)

    assert summary.stage == state
    assert summary.candidates_used == 1
    assert len(optimizer.received_observations) == 1
    assert optimizer.received_observations[0].settings_readback == {
        "ibias_dac": 71,
        **{
            key: value
            for key, value in harness.baseline.i2c.model_dump().items()
            if key != "ibias_dac"
        },
    }


def test_fir_runner_applies_one_direct_formula_target_then_measures_repeats(
    harness_factory,
    monkeypatch,
):
    import vcsel_ai_tuning.campaign.orchestrator as orchestrator_module
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(2, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    proposer = FakeFirProposer(full_target)
    captured_candidates: list[Candidate] = []
    original_validate_candidate_static = orchestrator_module.validate_candidate_static

    def capture_candidate(authorization, candidate, **kwargs):
        captured_candidates.append(candidate)
        return original_validate_candidate_static(authorization, candidate, **kwargs)

    monkeypatch.setattr(
        orchestrator_module,
        "validate_candidate_static",
        capture_candidate,
    )
    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        measurements=[
            _measurement(f"fir-{index}", er_db=3.2, tdecq_db=2.3)
            for index in range(3)
        ],
        policy=policy,
        fir_proposer=proposer,
    )
    authorization = _fir_authorization(harness)
    candidate = Candidate(
        candidate_id="fir-formula-direct",
        stage=CampaignState.FIR_APPLY,
        requested_fir=full_target,
        model_version="fir-formula-direct-v1.1",
        reason="direct FIR formula proposal",
    )

    retune_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_RETUNE,
    )
    summary = harness.orchestrator.run_fir_stage(
        authorization,
        retune_authorizations=(retune_authorization,),
    )

    assert summary.stage == CampaignState.FIR_APPLY
    assert summary.candidates_used == 1
    assert summary.convergence_reason == "FIR_ACCEPTED_UPDATE"
    assert captured_candidates
    assert all(validated == candidate for validated in captured_candidates)
    assert harness.bert.writes == [("SSPRQ", full_target)]
    assert harness.bert.readback() == {
        "pattern": "SSPRQ",
        "fir": full_target,
    }
    assert harness.dca.calls == ["measure_single"] * 3
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.list_observations()
    assert len(published) == 3
    assert {record.payload["candidate_id"] for record in published} == {
        "fir-formula-direct"
    }
    assert proposer.calls == [harness.baseline.fir]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_RETUNE


def test_final_verification_uses_five_measurements_and_medians_then_completes(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    measurements = [
        _measurement(f"final-{index}", er_db=er, tdecq_db=tdecq)
        for index, (er, tdecq) in enumerate(
            [(3.0, 2.7), (3.1, 2.5), (3.2, 2.4), (3.3, 2.3), (9.9, 0.1)]
        )
    ]
    harness = harness_factory(
        CampaignState.FINAL_VERIFICATION,
        measurements=measurements,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.05,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
    )

    report = harness.orchestrator.run_final_verification_stage()

    assert report.measurement_count == 5
    assert report.er_median_db == pytest.approx(3.2)
    assert report.tdecq_median_db == pytest.approx(2.4)
    assert report.er_median_db != 9.9
    assert report.tdecq_median_db != 0.1
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.COMPLETED
    assert _event_types(harness)[-1] == "FINAL_VERIFICATION_COMPLETED"


def test_pause_is_persisted_before_runner_stops_and_never_touches_hardware(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_suggestion(), _stop())
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement("must-not-run")],
        optimizer=optimizer,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    harness.orchestrator.pause()

    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.PAUSED
    with pytest.raises(CampaignPausedError, match="PAUSED"):
        harness.orchestrator.run_coarse_stage(authorization)
    assert harness.i2c.requests == []
    assert harness.bert.writes == []
    assert harness.dca.calls == []
    assert optimizer.histories == []


def test_abort_from_paused_uses_persisted_stage_authorization_and_rolls_back(
    harness_factory,
):
    nonbaseline_i2c = {**load_b0().i2c.model_dump(), "ibias_dac": 72}
    nonbaseline_fir = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.I2C_FINE,
        i2c_settings=nonbaseline_i2c,
        fir=nonbaseline_fir,
    )
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_FINE,
    )
    harness.orchestrator.pause()

    final_state = harness.orchestrator.abort(authorization)

    assert final_state == CampaignState.FAILED
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.bert.writes == [("SSPRQ", harness.baseline.fir)]
    assert _transition_targets(harness)[-3:] == [
        CampaignState.ABORTING,
        CampaignState.ROLLING_BACK,
        CampaignState.FAILED,
    ]


def test_abort_uses_stage_preauthorized_snapshot_once_and_finishes_failed_after_verified_rollback(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_FINE,
        i2c_settings={**load_b0().i2c.model_dump(), "ibias_dac": 72},
        fir=FirSettings(
            taps=(1, 9, -27, 103, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
        ),
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    final_state = harness.orchestrator.abort(authorization)

    assert final_state == CampaignState.FAILED
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.bert.writes == [("SSPRQ", harness.baseline.fir)]
    assert _event_types(harness)[-2:] == ["ROLLBACK_VERIFIED", "ABORTED"]


def test_valid_er_with_invalid_tdecq_is_persisted_and_delivered_to_coarse_optimizer(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_suggestion(), _stop())
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[
            _measurement(
                f"invalid-tdecq-{index}",
                tdecq_db=None,
                tdecq_status="INV",
                tdecq_invalid_reason="instrument returned INV",
            )
            for index in range(3)
        ],
        optimizer=optimizer,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    harness.orchestrator.run_coarse_stage(authorization)

    assert len(optimizer.received_observations) == 3
    assert all(
        observation.er_valid is True
        and observation.er_db == 3.2
        and observation.tdecq_db is None
        and observation.tdecq_status == "INV"
        and observation.tdecq_invalid_reason == "instrument returned INV"
        for observation in optimizer.received_observations
    )
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.list_observations()
    assert len(published) == 3
    assert all(observation.payload["feasible"] is False for observation in published)
    assert "OBSERVATION_INVALID" not in _event_types(harness)
    assert "ROLLBACK_STARTED" not in _event_types(harness)


def test_failed_acquisition_is_neither_persisted_nor_visible_to_optimizer(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_suggestion(), _stop())
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[
            _measurement(
                f"invalid-acquisition-{index}",
                acquisition_valid=False,
                er_db=None,
                er_status="INV",
                tdecq_db=None,
                tdecq_status="INV",
                tdecq_invalid_reason="acquisition timeout",
            )
            for index in range(3)
        ],
        optimizer=optimizer,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    harness.orchestrator.run_coarse_stage(authorization)

    assert optimizer.received_observations == ()
    assert isinstance(harness.repository, CampaignRepository)
    assert harness.repository.list_observations() == []
    assert _event_types(harness).count("OBSERVATION_INVALID") == 3


def test_persistence_failure_never_reaches_optimizer_and_triggers_preauthorized_rollback(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_suggestion())
    repository = RejectingRepository()
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement(f"persist-fail-{index}") for index in range(3)],
        optimizer=optimizer,
        repository=repository,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    with pytest.raises(RepositoryError, match="persistence failure"):
        harness.orchestrator.run_coarse_stage(authorization)

    assert repository.publish_attempts == 1
    assert optimizer.received_observations == ()
    assert "OBSERVATION_PERSISTENCE_FAILED" in _event_types(harness)
    assert harness.i2c.requests[-1] == harness.baseline.i2c.model_dump()
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED


def test_candidate_readback_mismatch_gets_exactly_one_preauthorized_rollback_attempt(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement("never-measured")],
    )
    harness.i2c.mismatch_after_writes = {1}
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    with pytest.raises(HardwareSafetyError, match="readback"):
        harness.orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE),
        )

    assert harness.i2c.requests == [
        {"ibias_dac": 71},
        harness.baseline.i2c.model_dump(),
    ]
    assert harness.bert.writes == [("SSPRQ", harness.baseline.fir)]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED
    assert harness.dca.calls == []
    assert _event_types(harness).count("ROLLBACK_STARTED") == 1
    assert _event_types(harness).count("ROLLBACK_VERIFIED") == 1


def test_i2c_measurement_exception_after_write_triggers_exactly_one_rollback(
    harness_factory,
):
    harness = harness_factory(CampaignState.I2C_COARSE)
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    with pytest.raises(HardwareSafetyError, match="measurement|DCA"):
        harness.orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE),
        )

    assert harness.i2c.requests == [
        {"ibias_dac": 71},
        harness.baseline.i2c.model_dump(),
    ]
    assert harness.bert.writes == [("SSPRQ", harness.baseline.fir)]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED
    assert _event_types(harness).count("ROLLBACK_STARTED") == 1
    assert harness.store.authorization_usage(
        CAMPAIGN_ID,
        authorization.authorization_id,
        now=harness.clock.now(),
    ).candidate_count == 1
    assert _event_types(harness).count("ROLLBACK_VERIFIED") == 1


def test_fir_measurement_exception_after_write_triggers_exactly_one_rollback(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
        optimizer=ScriptedOptimizer(_stop()),
        fir_proposer=FakeFirProposer(full_target),
    )
    authorization = _fir_authorization(harness)

    with pytest.raises(HardwareSafetyError, match="measurement|DCA"):
        harness.orchestrator.run_fir_stage(
            authorization,
            retune_authorizations=(
                _i2c_authorization(harness, CampaignState.I2C_RETUNE),
            ),
        )

    assert harness.bert.writes == [
        ("SSPRQ", full_target),
        ("SSPRQ", harness.baseline.fir),
    ]
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED
    assert _event_types(harness).count("ROLLBACK_STARTED") == 1
    assert _event_types(harness).count("ROLLBACK_VERIFIED") == 1


def test_rollback_readback_mismatch_stops_after_one_attempt_in_recovery_required(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement("never-measured")],
    )
    harness.i2c.mismatch_after_writes = {1, 2}
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    with pytest.raises(RecoveryRequiredError, match="rollback.*readback"):
        harness.orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE),
        )

    assert len(harness.i2c.requests) == 2
    assert len(harness.bert.writes) == 1
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.RECOVERY_REQUIRED
    )
    assert _event_types(harness).count("ROLLBACK_STARTED") == 1
    assert _event_types(harness).count("ROLLBACK_MISMATCH") == 1


def test_fir_readback_mismatch_restores_previous_fir_and_i2c_once(
    harness_factory,
):
    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        measurements=[_measurement("never-measured")],
        fir_proposer=FakeFirProposer(full_target),
    )
    harness.bert.mismatch_after_writes = {1}
    authorization = _fir_authorization(harness)
    candidate = Candidate(
        candidate_id="fir-formula-direct",
        stage=CampaignState.FIR_APPLY,
        requested_fir=full_target,
        model_version="fir-formula-direct-v1.1",
        reason="direct FIR formula proposal",
    )

    with pytest.raises(HardwareSafetyError, match="FIR.*readback"):
        harness.orchestrator.run_fir_stage(
            authorization,
            retune_authorizations=(
                _i2c_authorization(harness, CampaignState.I2C_RETUNE),
            ),
        )

    assert harness.bert.writes == [
        ("SSPRQ", candidate.requested_fir),
        ("SSPRQ", harness.baseline.fir),
    ]
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.bert.readback() == {
        "pattern": "SSPRQ",
        "fir": harness.baseline.fir,
    }
    assert harness.i2c.readback() == harness.baseline.i2c.model_dump()
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED
    assert harness.dca.calls == []


def test_fir_readback_and_rollback_mismatch_enters_recovery_required(
    harness_factory,
):
    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        measurements=[_measurement("never-measured")],
        fir_proposer=FakeFirProposer(full_target),
    )
    harness.bert.mismatch_after_writes = {1, 2}
    authorization = _fir_authorization(harness)

    with pytest.raises(
        RecoveryRequiredError,
        match="rollback readback mismatch after FIR candidate readback mismatch",
    ):
        harness.orchestrator.run_fir_stage(
            authorization,
            retune_authorizations=(
                _i2c_authorization(harness, CampaignState.I2C_RETUNE),
            ),
        )

    assert harness.bert.writes == [
        ("SSPRQ", full_target),
        ("SSPRQ", harness.baseline.fir),
    ]
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.dca.calls == []
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.RECOVERY_REQUIRED
    )


def test_coarse_gate_advances_only_after_three_repeats_meet_er_and_tdecq_counts(
    harness_factory,
):
    optimizer = ScriptedOptimizer(_suggestion(candidate_id="coarse-gate"), _stop())
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        optimizer=optimizer,
        measurements=[
            _measurement("coarse-gate-1", er_db=2.8, tdecq_db=3.2),
            _measurement("coarse-gate-2", er_db=3.0, tdecq_db=3.0),
            _measurement(
                "coarse-gate-3",
                er_db=3.1,
                tdecq_db=None,
                tdecq_status="INV",
                tdecq_invalid_reason="validity-negative sample",
            ),
        ],
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    summary = harness.orchestrator.run_coarse_stage(authorization)

    assert summary.convergence_reason == "COARSE_GATE_SATISFIED"
    assert harness.dca.calls == ["measure_single"] * 3
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.AWAIT_FINE_APPROVAL
    )


@pytest.mark.parametrize(
    "measurements",
    [
        pytest.param(
            [
                _measurement(f"coarse-low-er-{index}", er_db=value, tdecq_db=3.0)
                for index, value in enumerate((2.3, 2.4, 2.6))
            ],
            id="median-er-below-2.5",
        ),
        pytest.param(
            [
                _measurement("coarse-one-corr", er_db=3.0, tdecq_db=3.0),
                _measurement(
                    "coarse-inv-1",
                    er_db=3.1,
                    tdecq_db=None,
                    tdecq_status="INV",
                    tdecq_invalid_reason="not measurable",
                ),
                _measurement(
                    "coarse-inv-2",
                    er_db=3.2,
                    tdecq_db=None,
                    tdecq_status="INV",
                    tdecq_invalid_reason="not measurable",
                ),
            ],
            id="fewer-than-two-corr-tdecq",
        ),
    ],
)
def test_coarse_gate_stays_in_stage_with_deterministic_partial_reason(
    harness_factory, measurements
):
    optimizer = ScriptedOptimizer(_suggestion(candidate_id="coarse-partial"), _stop())
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        optimizer=optimizer,
        measurements=measurements,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    summary = harness.orchestrator.run_coarse_stage(authorization)

    assert summary.convergence_reason == "COARSE_GATE_NOT_SATISFIED"
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_COARSE


@pytest.mark.parametrize(
    ("state", "runner_name", "measurement", "expected_reason"),
    [
        pytest.param(
            CampaignState.I2C_FINE,
            "run_fine_stage",
            _measurement(
                "fine-invalid-tdecq",
                er_db=3.2,
                tdecq_db=None,
                tdecq_status="INV",
                tdecq_invalid_reason="not measurable",
            ),
            "FINE_STAGE_INFEASIBLE",
            id="fine-needs-corr-tdecq",
        ),
        pytest.param(
            CampaignState.I2C_FINE,
            "run_fine_stage",
            _measurement("fine-low-er", er_db=2.9, tdecq_db=2.3),
            "PARTIAL_CONVERGENCE",
            id="fine-needs-er-3.0",
        ),
        pytest.param(
            CampaignState.I2C_RETUNE,
            "run_retune_stage",
            _measurement(
                "retune-invalid-tdecq",
                er_db=3.2,
                tdecq_db=None,
                tdecq_status="INV",
                tdecq_invalid_reason="not measurable",
            ),
            "RETUNE_STAGE_INFEASIBLE",
            id="retune-needs-corr-tdecq",
        ),
        pytest.param(
            CampaignState.I2C_RETUNE,
            "run_retune_stage",
            _measurement("retune-low-er", er_db=2.9, tdecq_db=2.3),
            "RETUNE_STAGE_INFEASIBLE",
            id="retune-needs-er-3.0",
        ),
    ],
)
def test_fine_and_retune_persist_constraint_negative_observations_but_do_not_advance(
    harness_factory, state, runner_name, measurement, expected_reason
):
    optimizer = ScriptedOptimizer(_suggestion(candidate_id=f"{state.value}-negative"), _stop())
    harness = harness_factory(
        state,
        optimizer=optimizer,
        measurements=[measurement],
    )
    authorization = _i2c_authorization(harness, state)

    summary = getattr(harness.orchestrator, runner_name)(authorization)

    assert summary.convergence_reason == expected_reason
    assert harness.store.current_state(CAMPAIGN_ID) == state
    assert len(optimizer.received_observations) == 1
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.list_observations()
    assert len(published) == 1
    assert published[0].payload["feasible"] is False


@pytest.mark.parametrize(
    ("state", "runner_name", "next_state", "reason"),
    [
        (
            CampaignState.I2C_FINE,
            "run_fine_stage",
            CampaignState.AWAIT_FIR_APPROVAL,
            "FINE_STAGE_FEASIBLE",
        ),
        (
            CampaignState.I2C_RETUNE,
            "run_retune_stage",
            CampaignState.FINAL_VERIFICATION,
            "RETUNE_STAGE_FEASIBLE",
        ),
    ],
)
def test_feasible_fine_and_retune_runners_advance_to_next_nonmutating_state(
    harness_factory, state, runner_name, next_state, reason
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    optimizer = ScriptedOptimizer(_suggestion(candidate_id=f"{state.value}-valid"), _stop())
    policy = (
        OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        )
        if state == CampaignState.I2C_FINE
        else None
    )
    harness = harness_factory(
        state,
        optimizer=optimizer,
        policy=policy,
        measurements=[
            _measurement(
                f"{state.value}-valid-{index}",
                er_db=3.1,
                tdecq_db=2.3,
            )
            for index in range(4 if state == CampaignState.I2C_FINE else 1)
        ],
    )
    authorization = _i2c_authorization(harness, state)

    summary = getattr(harness.orchestrator, runner_name)(authorization)

    assert summary.convergence_reason == reason
    assert harness.store.current_state(CAMPAIGN_ID) == next_state


def test_orchestrator_policy_defaults_fine_patience_sigma_and_confirmation():
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy()

    assert policy.baseline_tdecq_sample_std_db == 0.0
    assert policy.fine_patience_candidates == 5
    assert policy.best_point_confirmation_repeats == 3
    with pytest.raises(ValidationError):
        OrchestratorPolicy(baseline_tdecq_sample_std_db=-0.1)
    with pytest.raises(ValidationError):
        OrchestratorPolicy(fine_patience_candidates=True)


def test_fine_selects_best_feasible_point_and_confirms_it_three_times(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    optimizer = ScriptedOptimizer(
        _suggestion(candidate_id="fine-first", value=71),
        _suggestion(candidate_id="fine-best", value=72),
        _stop(),
    )
    harness = harness_factory(
        CampaignState.I2C_FINE,
        policy=policy,
        optimizer=optimizer,
        measurements=[
            _measurement("fine-first", er_db=3.2, tdecq_db=2.2),
            _measurement("fine-best", er_db=3.2, tdecq_db=2.0),
            _measurement("fine-confirm-1", er_db=3.2, tdecq_db=2.05),
            _measurement("fine-confirm-2", er_db=3.2, tdecq_db=2.0),
            _measurement("fine-confirm-3", er_db=3.2, tdecq_db=1.95),
        ],
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    summary = harness.orchestrator.run_fine_stage(authorization)

    assert summary.convergence_reason == "FINE_STAGE_FEASIBLE"
    assert summary.best_observation is not None
    assert summary.best_observation.settings.readback_i2c["ibias_dac"] == 72
    assert harness.i2c.requests[-1] == {"ibias_dac": 72}
    assert harness.dca.calls == ["measure_single"] * 5
    assert isinstance(harness.repository, CampaignRepository)
    assert len(harness.repository.list_observations()) == 5
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.AWAIT_FIR_APPROVAL
    )
    assert harness.store.authorization_usage(
        CAMPAIGN_ID,
        authorization.authorization_id,
        now=harness.clock.now(),
    ).candidate_count == 3


def test_fine_stops_after_five_candidates_without_a_new_best_then_confirms_best(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    optimizer = ScriptedOptimizer(
        *(
            _suggestion(candidate_id=f"fine-patience-{index}", value=value)
            for index, value in enumerate((71, 72, 71, 72, 71, 72), start=1)
        )
    )
    harness = harness_factory(
        CampaignState.I2C_FINE,
        policy=policy,
        optimizer=optimizer,
        measurements=[
            _measurement("fine-patience-best", er_db=3.2, tdecq_db=2.0),
            *[
                _measurement(f"fine-patience-worse-{index}", er_db=3.2, tdecq_db=2.1)
                for index in range(5)
            ],
            *[
                _measurement(f"fine-patience-confirm-{index}", er_db=3.2, tdecq_db=2.0)
                for index in range(3)
            ],
        ],
    )
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_FINE,
        max_candidates=7,
    )

    summary = harness.orchestrator.run_fine_stage(authorization)

    assert summary.candidates_used == 6
    assert summary.convergence_reason == "FINE_STAGE_FEASIBLE"
    assert harness.i2c.requests[-1] == {"ibias_dac": 71}
    assert harness.dca.calls == ["measure_single"] * 9
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.AWAIT_FIR_APPROVAL
    )


def test_fine_meaningful_threshold_uses_three_times_baseline_sigma(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        baseline_tdecq_sample_std_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    optimizer = ScriptedOptimizer(
        _suggestion(candidate_id="fine-below-noise", value=71),
        _stop(),
    )
    harness = harness_factory(
        CampaignState.I2C_FINE,
        policy=policy,
        optimizer=optimizer,
        measurements=[
            _measurement("fine-below-noise", er_db=3.2, tdecq_db=2.25),
        ],
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    summary = harness.orchestrator.run_fine_stage(authorization)

    assert summary.convergence_reason == "FINE_NO_MEANINGFUL_IMPROVEMENT"
    assert harness.dca.calls == ["measure_single"]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_FINE


def test_fine_budget_end_reports_partial_convergence_for_er_between_2_5_and_3(
    harness_factory,
):
    optimizer = ScriptedOptimizer(
        _suggestion(candidate_id="fine-partial", value=71),
    )
    harness = harness_factory(
        CampaignState.I2C_FINE,
        optimizer=optimizer,
        measurements=[
            _measurement("fine-partial", er_db=2.8, tdecq_db=2.2),
        ],
    )
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_FINE,
        max_candidates=1,
    )

    summary = harness.orchestrator.run_fine_stage(authorization)

    assert summary.convergence_reason == "PARTIAL_CONVERGENCE"
    assert summary.best_observation is not None
    assert summary.best_observation.feasible is False
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_FINE


def test_invalid_final_verification_never_claims_completed(harness_factory):
    measurements = [
        _measurement(f"final-valid-{index}", er_db=3.2, tdecq_db=2.3)
        for index in range(4)
    ] + [
        _measurement(
            "final-invalid",
            acquisition_valid=False,
            er_db=None,
            er_status="INV",
            tdecq_db=None,
            tdecq_status="INV",
            tdecq_invalid_reason="timeout",
        )
    ]
    harness = harness_factory(
        CampaignState.FINAL_VERIFICATION,
        measurements=measurements,
    )

    result = harness.orchestrator.run_final_verification_stage()

    assert result.measurement_count == 5
    assert result.report.valid is False
    assert result.outcome is None
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.FINAL_VERIFICATION
    )
    assert _event_types(harness)[-1] == "FINAL_VERIFICATION_INVALID"


@pytest.mark.parametrize(
    ("er_db", "tdecq_db", "expected_outcome"),
    [
        pytest.param(3.2, 2.2, "SUCCESS", id="success"),
        pytest.param(2.8, 2.3, "PARTIAL_CONVERGENCE", id="partial"),
        pytest.param(3.2, 2.45, "NO_IMPROVEMENT", id="no-improvement"),
    ],
)
def test_final_verification_returns_and_persists_deterministic_outcome(
    harness_factory,
    er_db,
    tdecq_db,
    expected_outcome,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        baseline_tdecq_sample_std_db=0.05,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    harness = harness_factory(
        CampaignState.FINAL_VERIFICATION,
        policy=policy,
        measurements=[
            _measurement(
                f"final-{expected_outcome.lower()}-{index}",
                er_db=er_db,
                tdecq_db=tdecq_db,
            )
            for index in range(5)
        ],
    )

    result = harness.orchestrator.run_final_verification_stage()

    assert result.outcome == expected_outcome
    assert result.report.valid is True
    assert result.measurement_count == 5
    final_event = harness.store.events(CAMPAIGN_ID)[-1]
    assert final_event.event_type == "FINAL_VERIFICATION_COMPLETED"
    assert final_event.payload["outcome"] == expected_outcome
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.COMPLETED


def test_static_candidate_range_rejection_happens_before_any_adapter_call(
    harness_factory,
):
    harness = harness_factory(CampaignState.I2C_COARSE)
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_COARSE,
        max_candidates=1,
        max_duration_s=10.0,
    )
    candidate = _candidate(CampaignState.I2C_COARSE, value=80)

    with pytest.raises(AuthorizationError):
        harness.orchestrator.run_one_candidate(authorization, candidate)

    assert harness.dca.calls == []
    assert harness.i2c.calls == []
    assert harness.bert.calls == []


@pytest.mark.parametrize("rebuild", [False, True], ids=["same-instance", "rebuilt"])
def test_persisted_authorization_candidate_budget_cannot_be_reset(
    harness_factory, rebuild
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[
            _measurement("budget-first"),
            _measurement("budget-must-not-run"),
        ],
    )
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_COARSE,
        max_candidates=1,
    )

    harness.orchestrator.run_one_candidate(
        authorization,
        _candidate(CampaignState.I2C_COARSE, value=71).model_copy(
            update={"candidate_id": "budget-first"}
        ),
    )
    orchestrator = _rebuild_orchestrator(harness) if rebuild else harness.orchestrator
    calls_before_excess = (
        tuple(harness.i2c.calls),
        tuple(harness.bert.calls),
        tuple(harness.dca.calls),
        tuple(harness.i2c.requests),
        tuple(harness.bert.writes),
    )

    with pytest.raises(
        (AuthorizationError, OrchestratorError),
        match="authorization|max_candidates|candidate budget|exhausted",
    ):
        orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE, value=72).model_copy(
                update={"candidate_id": "budget-excess"}
            ),
        )

    assert (
        tuple(harness.i2c.calls),
        tuple(harness.bert.calls),
        tuple(harness.dca.calls),
        tuple(harness.i2c.requests),
        tuple(harness.bert.writes),
    ) == calls_before_excess


def test_persisted_authorization_duration_survives_rebuild_and_rejects_before_hardware(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[
            _measurement("duration-first"),
            _measurement("duration-must-not-run"),
        ],
    )
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_COARSE,
        max_candidates=2,
        max_duration_s=5.0,
    )
    harness.orchestrator.run_one_candidate(
        authorization,
        _candidate(CampaignState.I2C_COARSE, value=71).model_copy(
            update={"candidate_id": "duration-first"}
        ),
    )
    rebuilt = _rebuild_orchestrator(harness)
    harness.clock.advance(6.0)
    calls_before_expiry = (
        tuple(harness.i2c.calls),
        tuple(harness.bert.calls),
        tuple(harness.dca.calls),
        tuple(harness.i2c.requests),
        tuple(harness.bert.writes),
    )

    with pytest.raises(
        (AuthorizationError, OrchestratorError),
        match="authorization|max_duration|duration|elapsed",
    ):
        rebuilt.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE, value=72).model_copy(
                update={"candidate_id": "duration-expired"}
            ),
        )

    assert (
        tuple(harness.i2c.calls),
        tuple(harness.bert.calls),
        tuple(harness.dca.calls),
        tuple(harness.i2c.requests),
        tuple(harness.bert.writes),
    ) == calls_before_expiry


def test_shared_campaign_execution_lease_rejects_racing_orchestrator_before_hardware(
    tmp_path,
):
    store_path = tmp_path / "shared-campaign.sqlite3"
    repository_root = tmp_path / "shared-artifacts"
    with CampaignStore(store_path) as bootstrap_store:
        bootstrap_store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)

    baseline = load_b0()
    baseline_hash = canonical_json_sha256(baseline)
    authorization = _i2c_authorization(
        SimpleNamespace(baseline=baseline, baseline_hash=baseline_hash),
        CampaignState.I2C_COARSE,
        max_candidates=2,
    )
    context = OptimizationContext(
        driver_profile_hash="b" * 64,
        vcsel_model="APA4701",
        device_id="vcsel-fake-1",
        test_context_id="fake-53g-ssprq-25c",
    )
    winner_entered_hardware = Event()
    release_winner = Event()
    loser_done = Event()
    winner_errors: list[BaseException] = []
    loser_errors: list[BaseException] = []
    winner_results: list[tuple[object, ...]] = []
    loser_results: list[tuple[object, ...]] = []
    winner_log: list[str] = []
    loser_log: list[str] = []

    class BlockingI2c(FakeI2c):
        def apply(self, requested):
            winner_entered_hardware.set()
            if not release_winner.wait(timeout=3.0):
                raise AssertionError("timed out waiting to release winner hardware")
            return super().apply(requested)

    winner_i2c = BlockingI2c(baseline.i2c.model_dump(), operation_log=winner_log)
    winner_bert = FakeBert(
        baseline.pattern,
        baseline.fir,
        operation_log=winner_log,
    )
    winner_dca = FakeDca(
        [_measurement("lease-winner")],
        operation_log=winner_log,
    )
    loser_i2c = FakeI2c(baseline.i2c.model_dump(), operation_log=loser_log)
    loser_bert = FakeBert(
        baseline.pattern,
        baseline.fir,
        operation_log=loser_log,
    )
    loser_dca = FakeDca(
        [_measurement("lease-loser-must-not-run")],
        operation_log=loser_log,
    )

    def build_orchestrator(store, repository, *, dca, i2c, bert):
        return CampaignOrchestrator(
            campaign_id=CAMPAIGN_ID,
            baseline=baseline,
            baseline_hash=baseline_hash,
            optimization_context=context,
            store=store,
            repository=repository,
            optimizer=ScriptedOptimizer(_stop()),
            dca=dca,
            i2c=i2c,
            bert=bert,
            clock=FakeClock(),
        )

    def run_winner():
        try:
            with CampaignStore(store_path) as store:
                with CampaignRepository(repository_root, CAMPAIGN_ID) as repository:
                    orchestrator = build_orchestrator(
                        store,
                        repository,
                        dca=winner_dca,
                        i2c=winner_i2c,
                        bert=winner_bert,
                    )
                    winner_results.append(
                        orchestrator.run_one_candidate(
                            authorization,
                            _candidate(CampaignState.I2C_COARSE, value=71).model_copy(
                                update={"candidate_id": "lease-winner"}
                            ),
                        )
                    )
        except BaseException as error:
            winner_errors.append(error)

    def run_loser():
        try:
            with CampaignStore(store_path) as store:
                with CampaignRepository(repository_root, CAMPAIGN_ID) as repository:
                    orchestrator = build_orchestrator(
                        store,
                        repository,
                        dca=loser_dca,
                        i2c=loser_i2c,
                        bert=loser_bert,
                    )
                    loser_results.append(
                        orchestrator.run_one_candidate(
                            authorization,
                            _candidate(CampaignState.I2C_COARSE, value=72).model_copy(
                                update={"candidate_id": "lease-loser"}
                            ),
                        )
                    )
        except BaseException as error:
            loser_errors.append(error)
        finally:
            loser_done.set()

    winner_thread = Thread(target=run_winner, name="campaign-lease-winner")
    loser_thread = Thread(target=run_loser, name="campaign-lease-loser")
    winner_thread.start()
    try:
        assert winner_entered_hardware.wait(timeout=2.0)
        loser_thread.start()
        assert loser_done.wait(timeout=2.0), "racing orchestrator did not fail fast"
    finally:
        release_winner.set()
        winner_thread.join(timeout=3.0)
        if loser_thread.ident is not None:
            loser_thread.join(timeout=3.0)

    assert not winner_thread.is_alive()
    assert not loser_thread.is_alive()
    assert winner_errors == []
    assert len(winner_results) == 1
    assert loser_results == []
    assert len(loser_errors) == 1
    assert isinstance(loser_errors[0], (ConcurrencyError, OrchestratorError))
    assert loser_i2c.calls == []
    assert loser_i2c.requests == []
    assert loser_bert.calls == []
    assert loser_bert.writes == []
    assert loser_dca.calls == []
    assert loser_log == []
    assert winner_i2c.requests == [{"ibias_dac": 71}]
    assert winner_log == ["i2c.apply", "dca.measure"]


def test_same_orchestrator_rejects_concurrent_thread_without_sharing_lease_depth(
    tmp_path,
):
    store_path = tmp_path / "same-instance.sqlite3"
    repository_root = tmp_path / "same-instance-artifacts"
    with CampaignStore(store_path) as bootstrap_store:
        bootstrap_store.save_state(CAMPAIGN_ID, CampaignState.I2C_COARSE)

    baseline = load_b0()
    baseline_hash = canonical_json_sha256(baseline)
    authorization = _i2c_authorization(
        SimpleNamespace(baseline=baseline, baseline_hash=baseline_hash),
        CampaignState.I2C_COARSE,
        max_candidates=2,
    )
    context = OptimizationContext(
        driver_profile_hash="b" * 64,
        vcsel_model="APA4701",
        device_id="vcsel-fake-1",
        test_context_id="fake-53g-ssprq-25c",
    )
    orchestrator_ready = Event()
    winner_entered_hardware = Event()
    release_winner = Event()
    orchestrator_holder: list[CampaignOrchestrator] = []
    winner_errors: list[BaseException] = []
    winner_results: list[tuple[object, ...]] = []

    class BlockingI2c(FakeI2c):
        def apply(self, requested):
            winner_entered_hardware.set()
            if not release_winner.wait(timeout=3.0):
                raise AssertionError("timed out waiting to release same-instance winner")
            return super().apply(requested)

    i2c = BlockingI2c(baseline.i2c.model_dump())
    bert = FakeBert(baseline.pattern, baseline.fir)
    dca = FakeDca([_measurement("same-instance-winner")])

    def run_winner():
        try:
            with CampaignStore(store_path) as store:
                with CampaignRepository(repository_root, CAMPAIGN_ID) as repository:
                    orchestrator = CampaignOrchestrator(
                        campaign_id=CAMPAIGN_ID,
                        baseline=baseline,
                        baseline_hash=baseline_hash,
                        optimization_context=context,
                        store=store,
                        repository=repository,
                        optimizer=ScriptedOptimizer(_stop()),
                        dca=dca,
                        i2c=i2c,
                        bert=bert,
                        clock=FakeClock(),
                    )
                    orchestrator_holder.append(orchestrator)
                    orchestrator_ready.set()
                    winner_results.append(
                        orchestrator.run_one_candidate(
                            authorization,
                            _candidate(CampaignState.I2C_COARSE, value=71).model_copy(
                                update={"candidate_id": "same-instance-winner"}
                            ),
                        )
                    )
        except BaseException as error:
            winner_errors.append(error)
            orchestrator_ready.set()

    winner_thread = Thread(target=run_winner, name="same-instance-winner")
    winner_thread.start()
    try:
        assert orchestrator_ready.wait(timeout=2.0)
        assert winner_entered_hardware.wait(timeout=2.0)
        assert len(orchestrator_holder) == 1

        with pytest.raises(OrchestratorError, match="already executing"):
            orchestrator_holder[0].run_one_candidate(
                authorization,
                _candidate(CampaignState.I2C_COARSE, value=72).model_copy(
                    update={"candidate_id": "same-instance-loser"}
                ),
            )

        assert i2c.calls == ["readback"]
        assert i2c.requests == []
        assert bert.calls == []
        assert dca.calls == []
    finally:
        release_winner.set()
        winner_thread.join(timeout=3.0)

    assert not winner_thread.is_alive()
    assert winner_errors == []
    assert len(winner_results) == 1
    assert i2c.requests == [{"ibias_dac": 71}]
    assert dca.calls == ["measure_single"]


def test_authorized_transition_persists_full_authorization_envelope(
    harness_factory,
):
    harness = harness_factory(CampaignState.AWAIT_FINE_APPROVAL)
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_FINE,
        max_candidates=4,
        max_duration_s=45.0,
    )

    harness.orchestrator.enter_stage(authorization)

    event = harness.store.events(CAMPAIGN_ID)[-1]
    assert event.event_type == "AUTHORIZED_STAGE_ENTERED"
    assert event.payload != {}
    assert event.payload["authorization"] == authorization.model_dump(mode="json")


def test_actual_i2c_readback_is_used_only_after_static_checks_for_max_step(
    harness_factory,
):
    initial = {**load_b0().i2c.model_dump(), "ibias_dac": 68}
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        i2c_settings=initial,
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)

    with pytest.raises(AuthorizationError, match="max_step"):
        harness.orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE, value=71),
            candidate_index=0,
            stage_started_monotonic_s=harness.clock.monotonic(),
        )

    assert harness.i2c.calls == ["readback"]
    assert harness.i2c.requests == []
    assert harness.bert.calls == []
    assert harness.dca.calls == []


def test_elapsed_is_recomputed_after_i2c_write_before_applied_value_validation(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[_measurement("must-not-measure-after-duration")],
    )
    harness.i2c.after_apply = lambda: harness.clock.advance(10.0)
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_COARSE,
        max_duration_s=10.0,
    )

    with pytest.raises(HardwareSafetyError, match="duration|elapsed"):
        harness.orchestrator.run_one_candidate(
            authorization,
            _candidate(CampaignState.I2C_COARSE),
            candidate_index=0,
            stage_started_monotonic_s=0.0,
        )

    assert harness.dca.calls == []
    assert harness.i2c.requests == [
        {"ibias_dac": 71},
        harness.baseline.i2c.model_dump(),
    ]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED
    assert _event_types(harness).count("ROLLBACK_STARTED") == 1


def test_stage_runner_passes_actual_candidate_indices_and_elapsed_time_to_authorization(
    harness_factory,
):
    class AdvancingOptimizer(ScriptedOptimizer):
        def __init__(self, clock: FakeClock) -> None:
            super().__init__(
                _suggestion(candidate_id="timed-1", value=71),
                _suggestion(candidate_id="timed-2", value=72),
            )
            self.clock = clock

        def suggest(self, observations, pending_keys=frozenset()):
            result = super().suggest(observations, pending_keys)
            if len(self.histories) == 2:
                self.clock.advance(10.0)
            return result

    harness = harness_factory(
        CampaignState.I2C_FINE,
        measurements=[_measurement("timed-first", er_db=2.9, tdecq_db=2.3)],
    )
    optimizer = AdvancingOptimizer(harness.clock)
    harness.optimizer = optimizer
    harness.orchestrator.optimizer = optimizer
    authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_FINE,
        max_candidates=3,
        max_duration_s=10.0,
    )

    with pytest.raises(AuthorizationError, match="elapsed_s|max_duration"):
        harness.orchestrator.run_fine_stage(authorization)

    assert len(harness.i2c.requests) == 1
    assert harness.dca.calls == ["measure_single"]


def test_enter_stage_is_the_only_direct_authorized_mutating_transition(
    harness_factory,
):
    harness = harness_factory(CampaignState.AWAIT_FINE_APPROVAL)
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    entered = harness.orchestrator.enter_stage(authorization)

    assert entered == CampaignState.I2C_FINE
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_FINE
    assert harness.i2c.calls == []
    assert harness.bert.calls == []
    assert harness.dca.calls == []


def test_enter_stage_rejects_illegal_jump_even_with_valid_target_authorization(
    harness_factory,
):
    harness = harness_factory(CampaignState.DRAFT)
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    with pytest.raises(IllegalTransition, match="illegal transition|DRAFT.*I2C_FINE"):
        harness.orchestrator.enter_stage(authorization)

    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.DRAFT
    assert harness.i2c.calls == []
    assert harness.bert.calls == []


def test_resume_uses_persisted_pause_resume_state_after_orchestrator_rebuild(
    harness_factory,
):
    harness = harness_factory(CampaignState.I2C_FINE)
    authorization = _i2c_authorization(harness, CampaignState.I2C_FINE)

    harness.orchestrator.pause()
    pause_event = next(
        event
        for event in reversed(harness.store.events(CAMPAIGN_ID))
        if event.event_type == "CAMPAIGN_PAUSED"
    )
    assert pause_event.payload["resume_state"] == "I2C_FINE"

    rebuilt = _rebuild_orchestrator(harness)
    resumed = rebuilt.resume(authorization)

    assert resumed == CampaignState.I2C_FINE
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_FINE
    assert harness.i2c.calls == []
    assert harness.bert.calls == []


def test_rebuilt_orchestrator_allocates_unique_persistent_observation_ids(
    harness_factory,
):
    harness = harness_factory(
        CampaignState.I2C_COARSE,
        measurements=[
            _measurement("before-orchestrator-rebuild"),
            _measurement("after-orchestrator-rebuild"),
        ],
    )
    authorization = _i2c_authorization(harness, CampaignState.I2C_COARSE)
    candidate_id = "same-candidate-across-rebuild"

    first = harness.orchestrator.run_one_candidate(
        authorization,
        _candidate(CampaignState.I2C_COARSE, value=71).model_copy(
            update={"candidate_id": candidate_id}
        ),
    )[0]
    rebuilt = _rebuild_orchestrator(harness)
    second = rebuilt.run_one_candidate(
        authorization,
        _candidate(CampaignState.I2C_COARSE, value=72).model_copy(
            update={"candidate_id": candidate_id}
        ),
    )[0]

    assert first.observation_id != second.observation_id
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.list_observations()
    assert {record.observation_id for record in published} == {
        first.observation_id,
        second.observation_id,
    }
    artifact_names_by_observation = {
        record.observation_id: {
            Path(artifact.relative_path).name for artifact in record.artifacts
        }
        for record in published
    }
    assert "before-orchestrator-rebuild.json" in artifact_names_by_observation[
        first.observation_id
    ]
    assert "after-orchestrator-rebuild.json" in artifact_names_by_observation[
        second.observation_id
    ]
    assert "after-orchestrator-rebuild.json" not in artifact_names_by_observation[
        first.observation_id
    ]
    assert "before-orchestrator-rebuild.json" not in artifact_names_by_observation[
        second.observation_id
    ]
    assert all(
        harness.repository.get_artifact(artifact.relative_path) == artifact
        for record in published
        for artifact in record.artifacts
    )


def test_orchestrator_policy_is_frozen_strict_and_defaults_to_three_fir_accepts():
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )

    assert policy.coarse_er_gate_db == 2.5
    assert policy.constrained_er_min_db == 3.0
    assert policy.max_accepted_fir_updates == 3
    with pytest.raises(ValidationError, match="frozen"):
        policy.max_accepted_fir_updates = 4
    with pytest.raises(ValidationError):
        OrchestratorPolicy(
            meaningful_tdecq_improvement_db=-0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        )


def test_fir_outer_prevalidates_formula_target_before_hardware(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    outside_envelope = FirSettings(
        taps=(5, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=policy,
        fir_proposer=FakeFirProposer(outside_envelope),
    )
    authorization = _fir_authorization(harness)

    with pytest.raises(AuthorizationError, match="range"):
        harness.orchestrator.run_fir_stage(
            authorization,
            retune_authorizations=(
                _i2c_authorization(harness, CampaignState.I2C_RETUNE),
            ),
        )

    assert harness.i2c.calls == []
    assert harness.bert.calls == []
    assert harness.dca.calls == []


@pytest.mark.parametrize(
    "unsafe_argument",
    [
        pytest.param({"candidates": ()}, id="candidate-stream"),
        pytest.param({"full_target": load_b0().fir}, id="raw-full-target"),
    ],
)
def test_fir_stage_has_no_public_raw_candidate_or_target_boundary(
    harness_factory,
    unsafe_argument,
):
    harness = harness_factory(CampaignState.FIR_APPLY)
    authorization = _fir_authorization(harness)

    with pytest.raises(TypeError, match="candidates|full_target"):
        harness.orchestrator.run_fir_stage(
            authorization,
            **unsafe_argument,
        )

    assert harness.i2c.calls == []
    assert harness.bert.calls == []
    assert harness.dca.calls == []


def test_fir_stage_requires_separate_inner_retune_authorization_before_hardware(
    harness_factory,
):
    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        fir_proposer=FakeFirProposer(full_target),
    )
    authorization = _fir_authorization(harness)

    with pytest.raises((AuthorizationError, OrchestratorError), match="retune|authorization"):
        harness.orchestrator.run_fir_stage(authorization)

    assert harness.i2c.calls == []
    assert harness.bert.calls == []
    assert harness.dca.calls == []


def test_direct_fir_candidate_runs_write_then_bounded_retune_then_three_measurements(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
        optimizer=ScriptedOptimizer(
            _suggestion(candidate_id="fir-inner-retune-order", value=71),
            _stop(),
        ),
        fir_proposer=FakeFirProposer(full_target),
        measurements=[
            _measurement("fir-inner-retune-order", er_db=3.2, tdecq_db=2.4),
            *(
                _measurement(
                    f"fir-inner-retune-confirmation-{index}",
                    er_db=3.2,
                    tdecq_db=2.3,
                )
                for index in range(3)
            ),
        ],
    )
    authorization = _fir_authorization(harness)

    summary = harness.orchestrator.run_fir_stage(
        authorization,
        retune_authorizations=(
            _i2c_authorization(harness, CampaignState.I2C_RETUNE),
        ),
    )

    assert summary.convergence_reason == "FIR_ACCEPTED_UPDATE"
    assert harness.bert.writes == [("SSPRQ", full_target)]
    assert harness.operation_log == [
        "bert.apply",
        "i2c.apply",
        "dca.measure",
        "dca.measure",
        "dca.measure",
        "dca.measure",
    ]
    assert isinstance(harness.repository, CampaignRepository)
    assert all(
        record.payload["settings"]["readback_i2c"]["ibias_dac"] == 71
        for record in harness.repository.list_observations()
    )


def test_inner_fir_retune_measures_and_persists_before_each_next_suggestion(
    harness_factory,
    monkeypatch,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    optimizer = ScriptedOptimizer(
        _suggestion(candidate_id="fir-inner-retune-1", value=71),
        _suggestion(candidate_id="fir-inner-retune-2", value=72),
        _stop(),
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
        optimizer=optimizer,
        fir_proposer=FakeFirProposer(full_target),
        measurements=[
            _measurement("fir-inner-retune-1", er_db=3.2, tdecq_db=2.4),
            _measurement("fir-inner-retune-2", er_db=3.2, tdecq_db=2.35),
            *(
                _measurement(
                    f"fir-confirmation-{index}",
                    er_db=3.2,
                    tdecq_db=2.3,
                )
                for index in range(3)
            ),
        ],
    )
    _capture_optimizer_and_persistence_order(harness, monkeypatch)

    summary = harness.orchestrator.run_fir_stage(
        _fir_authorization(harness),
        retune_authorizations=(
            _i2c_authorization(
                harness,
                CampaignState.I2C_RETUNE,
                max_candidates=2,
            ),
        ),
    )

    assert summary.convergence_reason == "FIR_ACCEPTED_UPDATE"
    assert harness.operation_log == [
        "bert.apply",
        "optimizer.suggest",
        "i2c.apply",
        "dca.measure",
        "observation.persisted",
        "optimizer.suggest",
        "i2c.apply",
        "dca.measure",
        "observation.persisted",
        "dca.measure",
        "observation.persisted",
        "dca.measure",
        "observation.persisted",
        "dca.measure",
        "observation.persisted",
    ]
    assert len(optimizer.histories) == 2
    assert optimizer.histories[0] == ()
    assert len(optimizer.histories[1]) == 1
    first_retune_observation = optimizer.histories[1][0]
    assert first_retune_observation.candidate_id == "fir-inner-retune-1"
    assert first_retune_observation.settings_readback["ibias_dac"] == 71
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.list_observations()
    assert first_retune_observation.observation_id in {
        record.observation_id for record in published
    }
    assert sorted(record.payload["candidate_id"] for record in published) == sorted(
        [
            "fir-inner-retune-1",
            "fir-inner-retune-2",
            "fir-formula-direct",
            "fir-formula-direct",
            "fir-formula-direct",
        ]
    )


def test_invalid_inner_fir_retune_measurement_is_excluded_from_optimizer_history(
    harness_factory,
    monkeypatch,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    optimizer = ScriptedOptimizer(
        _suggestion(candidate_id="invalid-inner-retune", value=71),
        _suggestion(candidate_id="valid-inner-retune", value=72),
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
        optimizer=optimizer,
        fir_proposer=FakeFirProposer(full_target),
        measurements=[
            _measurement(
                "invalid-inner-retune",
                acquisition_valid=False,
                er_db=None,
                er_status="INV",
                tdecq_db=None,
                tdecq_status="INV",
            ),
            _measurement("valid-inner-retune", er_db=3.2, tdecq_db=2.35),
            *(
                _measurement(
                    f"invalid-history-confirmation-{index}",
                    er_db=3.2,
                    tdecq_db=2.3,
                )
                for index in range(3)
            ),
        ],
    )
    _capture_optimizer_and_persistence_order(harness, monkeypatch)

    harness.orchestrator.run_fir_stage(
        _fir_authorization(harness),
        retune_authorizations=(
            _i2c_authorization(
                harness,
                CampaignState.I2C_RETUNE,
                max_candidates=2,
            ),
        ),
    )

    assert optimizer.histories == [(), ()]
    assert harness.operation_log[:7] == [
        "bert.apply",
        "optimizer.suggest",
        "i2c.apply",
        "dca.measure",
        "optimizer.suggest",
        "i2c.apply",
        "dca.measure",
    ]
    assert isinstance(harness.repository, CampaignRepository)
    published = harness.repository.list_observations()
    assert sorted(record.payload["candidate_id"] for record in published) == sorted(
        [
            "valid-inner-retune",
            "fir-formula-direct",
            "fir-formula-direct",
            "fir-formula-direct",
        ]
    )
    assert _event_types(harness).count("OBSERVATION_INVALID") == 1


def test_unpersisted_inner_fir_retune_measurement_never_reaches_optimizer(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    optimizer = ScriptedOptimizer(
        _suggestion(candidate_id="unpersisted-inner-retune", value=71),
        _suggestion(candidate_id="must-not-be-requested", value=72),
    )
    repository = RejectingRepository()
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
        optimizer=optimizer,
        repository=repository,
        fir_proposer=FakeFirProposer(full_target),
        measurements=[_measurement("unpersisted-inner-retune")],
    )

    with pytest.raises(RepositoryError, match="persistence failure"):
        harness.orchestrator.run_fir_stage(
            _fir_authorization(harness),
            retune_authorizations=(
                _i2c_authorization(
                    harness,
                    CampaignState.I2C_RETUNE,
                    max_candidates=2,
                ),
            ),
        )

    assert optimizer.histories == [()]
    assert optimizer.received_observations == ()
    assert repository.publish_attempts == 1
    assert harness.dca.calls == ["measure_single"]
    assert harness.i2c.requests == [
        {"ibias_dac": 71},
        harness.baseline.i2c.model_dump(),
    ]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FAILED


def test_fir_acceptance_threshold_uses_three_times_baseline_sigma(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        baseline_tdecq_sample_std_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=policy,
        fir_proposer=FakeFirProposer(full_target),
        measurements=[
            _measurement(f"fir-below-noise-{index}", er_db=3.2, tdecq_db=2.25)
            for index in range(3)
        ],
    )
    authorization = _fir_authorization(harness)

    summary = harness.orchestrator.run_fir_stage(
        authorization,
        retune_authorizations=(
            _i2c_authorization(harness, CampaignState.I2C_RETUNE),
        ),
    )

    assert summary.convergence_reason == "FIR_NO_ACCEPTED_UPDATE"
    assert harness.bert.writes == [
        ("SSPRQ", full_target),
        ("SSPRQ", harness.baseline.fir),
    ]
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FIR_APPLY


def test_accepted_fir_enters_retune_only_with_separate_retune_authorization(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    accepted_fir = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=policy,
        fir_proposer=FakeFirProposer(accepted_fir),
        optimizer=ScriptedOptimizer(
            _suggestion(candidate_id="fir-inner-retune", value=71),
            _stop(),
        ),
        measurements=[
            _measurement("fir-inner-retune", er_db=3.2, tdecq_db=2.4),
            *(
                _measurement(
                    f"fir-before-authorized-retune-{index}",
                    er_db=3.2,
                    tdecq_db=2.3,
                )
                for index in range(3)
            ),
        ],
    )
    fir_authorization = _fir_authorization(harness)
    retune_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_RETUNE,
    )

    summary = harness.orchestrator.run_fir_stage(
        fir_authorization,
        retune_authorizations=(retune_authorization,),
    )

    assert summary.convergence_reason == "FIR_ACCEPTED_UPDATE"
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.I2C_RETUNE
    assert harness.bert.writes == [("SSPRQ", accepted_fir)]
    assert harness.i2c.requests == [{"ibias_dac": 71}]
    assert harness.store.authorization_usage(
        CAMPAIGN_ID,
        fir_authorization.authorization_id,
        now=harness.clock.now(),
    ).candidate_count == 1
    assert harness.store.authorization_usage(
        CAMPAIGN_ID,
        retune_authorization.authorization_id,
        now=harness.clock.now(),
    ).candidate_count == 1


@pytest.mark.parametrize(
    ("er_db", "tdecq_db", "tdecq_status", "invalid_reason", "case"),
    [
        pytest.param(
            3.1,
            2.4,
            "CORR",
            None,
            "threshold-equality",
            id="improvement-must-exceed",
        ),
        pytest.param(
            3.05,
            2.3,
            "CORR",
            None,
            "er-degraded",
            id="must-not-degrade-reference-er",
        ),
        pytest.param(
            3.2,
            None,
            "INV",
            "not measurable",
            "tdecq-invalid",
            id="fir-needs-corr-tdecq",
        ),
    ],
)
def test_rejected_fir_candidate_restores_exact_stage_entry_snapshot_once(
    harness_factory, er_db, tdecq_db, tdecq_status, invalid_reason, case
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=policy,
        fir_proposer=FakeFirProposer(full_target),
        measurements=[
            _measurement(
                f"fir-reject-{case}-{index}",
                er_db=er_db,
                tdecq_db=tdecq_db,
                tdecq_status=tdecq_status,
                tdecq_invalid_reason=invalid_reason,
            )
            for index in range(3)
        ],
    )
    authorization = _fir_authorization(harness)
    candidate = Candidate(
        candidate_id=f"fir-reject-{case}",
        stage=CampaignState.FIR_APPLY,
        requested_fir=full_target,
        model_version="fir-proposer-1",
        reason="rejection gate test",
    )

    summary = harness.orchestrator.run_fir_stage(
        authorization,
        retune_authorizations=(
            _i2c_authorization(harness, CampaignState.I2C_RETUNE),
        ),
    )

    assert summary.convergence_reason == "FIR_NO_ACCEPTED_UPDATE"
    assert harness.bert.writes == [
        ("SSPRQ", candidate.requested_fir),
        ("SSPRQ", harness.baseline.fir),
    ]
    assert harness.i2c.requests == [harness.baseline.i2c.model_dump()]
    assert harness.bert.readback() == {
        "pattern": "SSPRQ",
        "fir": harness.baseline.fir,
    }
    assert harness.i2c.readback() == harness.baseline.i2c.model_dump()
    assert harness.store.current_state(CAMPAIGN_ID) == CampaignState.FIR_APPLY


def test_rejected_fir_restore_failure_traces_rollback_before_recovery_required(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=OrchestratorPolicy(
            meaningful_tdecq_improvement_db=0.1,
            fir_reference_tdecq_db=2.5,
            fir_reference_er_db=3.1,
        ),
        optimizer=ScriptedOptimizer(_stop()),
        fir_proposer=FakeFirProposer(full_target),
        measurements=[
            _measurement(f"fir-restore-failure-{index}", er_db=3.2, tdecq_db=2.4)
            for index in range(3)
        ],
    )
    harness.bert.mismatch_after_writes = {2}
    authorization = _fir_authorization(harness)

    with pytest.raises(RecoveryRequiredError, match="restore|rollback"):
        harness.orchestrator.run_fir_stage(
            authorization,
            retune_authorizations=(
                _i2c_authorization(harness, CampaignState.I2C_RETUNE),
            ),
        )

    assert _transition_targets(harness)[-3:] == [
        CampaignState.ABORTING,
        CampaignState.ROLLING_BACK,
        CampaignState.RECOVERY_REQUIRED,
    ]
    assert len(harness.bert.writes) == 2


def test_fir_outer_executes_only_one_formula_target_when_budgets_allow_multiple(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    full_target = FirSettings(
        taps=(2, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=policy,
        fir_proposer=FakeFirProposer(full_target),
        optimizer=ScriptedOptimizer(_stop(), _stop()),
        measurements=[
            _measurement(
                f"fir-accepted-{index}",
                er_db=3.2,
                tdecq_db=(2.3, 2.1)[index // 3],
            )
            for index in range(6)
        ],
    )
    authorization = _fir_authorization(harness)
    retune_authorizations = (
        _i2c_authorization(harness, CampaignState.I2C_RETUNE),
        _i2c_authorization(
            harness,
            CampaignState.I2C_RETUNE,
            max_candidates=2,
        ),
    )

    summary = harness.orchestrator.run_fir_stage(
        authorization,
        retune_authorizations=retune_authorizations,
    )

    assert summary.convergence_reason == "FIR_ACCEPTED_UPDATE"
    assert summary.candidates_used == 1
    assert harness.bert.writes == [("SSPRQ", full_target)]
    assert harness.bert.fir == full_target
    assert harness.dca.calls == ["measure_single"] * 3


def test_retune_starts_from_last_accepted_fir_snapshot_under_its_own_authorization(
    harness_factory,
):
    from vcsel_ai_tuning.campaign.orchestrator import OrchestratorPolicy

    accepted_fir = FirSettings(
        taps=(1, 9, -27, 103, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    policy = OrchestratorPolicy(
        meaningful_tdecq_improvement_db=0.1,
        fir_reference_tdecq_db=2.5,
        fir_reference_er_db=3.1,
    )
    optimizer = ScriptedOptimizer(
        _stop(),
        _suggestion(candidate_id="retune-after-fir", value=71),
        _stop(),
    )
    harness = harness_factory(
        CampaignState.FIR_APPLY,
        policy=policy,
        optimizer=optimizer,
        fir_proposer=FakeFirProposer(accepted_fir),
        measurements=[
            *[
                _measurement(f"fir-before-retune-{index}", er_db=3.2, tdecq_db=2.3)
                for index in range(3)
            ],
            _measurement("retune-after-fir", er_db=3.2, tdecq_db=2.2),
        ],
    )
    fir_authorization = _fir_authorization(harness)
    inner_retune_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_RETUNE,
    )

    harness.orchestrator.run_fir_stage(
        fir_authorization,
        retune_authorizations=(inner_retune_authorization,),
    )
    accepted_snapshot = AppliedSettings(
        readback_i2c=harness.baseline.i2c.model_dump(),
        readback_fir=accepted_fir,
    )
    retune_authorization = _i2c_authorization(
        harness,
        CampaignState.I2C_RETUNE,
        rollback_snapshot=accepted_snapshot,
    )
    summary = harness.orchestrator.run_retune_stage(retune_authorization)

    assert summary.convergence_reason == "RETUNE_STAGE_FEASIBLE"
    assert harness.bert.fir == accepted_fir
    assert harness.bert.writes == [("SSPRQ", accepted_fir)]
    assert harness.store.current_state(CAMPAIGN_ID) == (
        CampaignState.FINAL_VERIFICATION
    )
