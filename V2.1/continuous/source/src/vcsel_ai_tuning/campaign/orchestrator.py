"""Deterministic, authorization-gated campaign hardware orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from math import isclose
from statistics import median, stdev
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    OptimizationContext,
    TuningObservation,
)

from .authorization import (
    rollback_hash_for,
    validate_authorization,
    validate_candidate,
    validate_candidate_static,
    with_authorization_id,
)
from .models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    BaselineConfig,
    CampaignState,
    Candidate,
    FinalOutcome,
    FirAuthorizationSpec,
    FirSettings,
    Observation,
    ParameterSpec,
    StageAuthorization,
    StageSummary,
)
from .repository import ArtifactSpec, RepositoryError
from .state_machine import CampaignStateMachine, MUTATING_STATES
from .store import AuthorizationUsage, ConcurrencyError, StoreError


class OrchestratorError(RuntimeError):
    """Base class for deterministic campaign orchestration failures."""


class PreflightError(OrchestratorError):
    """Raised when a read-only hardware preflight cannot be verified."""


class CampaignPausedError(OrchestratorError):
    """Raised when execution is requested while the campaign is paused."""


class HardwareSafetyError(OrchestratorError):
    """Raised after a hardware safety failure was rolled back exactly once."""


class RecoveryRequiredError(HardwareSafetyError):
    """Raised when the single pre-authorized rollback cannot be verified."""


class OrchestratorPolicy(BaseModel):
    """Frozen campaign-level decision thresholds owned by the orchestrator."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        validate_default=True,
    )

    coarse_er_gate_db: FiniteFloat = Field(default=2.5, ge=0)
    constrained_er_min_db: FiniteFloat = Field(default=3.0, ge=0)
    meaningful_tdecq_improvement_db: FiniteFloat = Field(default=0.0, ge=0)
    baseline_tdecq_sample_std_db: FiniteFloat = Field(default=0.0, ge=0)
    fine_patience_candidates: StrictInt = Field(default=5, gt=0)
    best_point_confirmation_repeats: StrictInt = Field(default=3, gt=0)
    fir_reference_tdecq_db: FiniteFloat | None = Field(default=None, ge=0)
    fir_reference_er_db: FiniteFloat | None = Field(default=None, ge=0)
    max_accepted_fir_updates: StrictInt = Field(default=3, gt=0)


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    acquisition_valid: bool
    er_db: float | None
    er_status: str
    tdecq_db: float | None
    tdecq_status: str
    tdecq_invalid_reason: str | None
    elapsed_s: float
    raw_responses: Mapping[str, object]
    artifacts: tuple[ArtifactSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class PreflightResult:
    verified: bool
    dca: Mapping[str, object]
    i2c: Mapping[str, object]
    bert: Mapping[str, object]
    i2c_readback: Mapping[str, int]
    bert_readback: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ContinuousPreflightResult:
    verified: bool
    dca: Mapping[str, object]
    i2c: Mapping[str, object]
    i2c_readback: Mapping[str, int]
    observation: Observation


@dataclass(frozen=True, slots=True)
class ContinuousCommissionPreparation:
    observation: Observation
    authorization: StageAuthorization


@dataclass(frozen=True, slots=True)
class RepeatedMeasurementReport:
    measurement_count: int
    acquisition_valid_count: int
    er_valid_count: int
    tdecq_valid_count: int
    valid: bool
    er_median_db: float | None
    er_sample_std_db: float | None
    tdecq_median_db: float | None
    tdecq_sample_std_db: float | None
    raw_responses: tuple[Mapping[str, object], ...]
    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class FinalVerificationResult:
    """Typed final result while retaining report-style attribute access."""

    report: RepeatedMeasurementReport
    outcome: FinalOutcome | None

    def __getattr__(self, name: str) -> object:
        return getattr(self.report, name)


class DcaAdapter(Protocol):
    def preflight(self) -> Mapping[str, object]: ...

    def measure_single(self) -> MeasurementResult: ...


class I2cAdapter(Protocol):
    def preflight(self) -> Mapping[str, object]: ...

    def apply(self, requested: Mapping[str, int]) -> Mapping[str, int]: ...

    def readback(self) -> Mapping[str, int]: ...


class BertAdapter(Protocol):
    def preflight(self) -> Mapping[str, object]: ...

    def apply(self, pattern: str, fir: FirSettings) -> FirSettings: ...

    def readback(self) -> Mapping[str, object]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class Repository(Protocol):
    def publish_observation(
        self,
        observation_id: str,
        payload: Any,
        artifacts: tuple[ArtifactSpec, ...] | list[ArtifactSpec],
    ) -> Any: ...


class Optimizer(Protocol):
    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[tuple[int, ...]] = frozenset(),
    ) -> object: ...


class FirProposer(Protocol):
    def propose(self, current_fir: FirSettings) -> FirSettings: ...


def _execution_guarded(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run one public orchestration entry under the campaign execution lease."""

    @wraps(method)
    def guarded(self: "CampaignOrchestrator", *args: Any, **kwargs: Any) -> Any:
        with self._execution_scope():
            return method(self, *args, **kwargs)

    return guarded


class CampaignOrchestrator:
    """The sole campaign component permitted to call injected hardware adapters."""

    def __init__(
        self,
        *,
        campaign_id: str,
        baseline: BaselineConfig,
        baseline_hash: str,
        optimization_context: OptimizationContext,
        store: Any,
        repository: Repository,
        optimizer: Optimizer,
        dca: DcaAdapter,
        i2c: I2cAdapter,
        bert: BertAdapter,
        clock: Clock,
        policy: OrchestratorPolicy | None = None,
        fir_proposer: FirProposer | None = None,
        parameter_specs: tuple[ParameterSpec, ...] = (),
        fir_authorization_spec: FirAuthorizationSpec | None = None,
        tdecq_only: bool = False,
        continuous_i2c_only: bool = False,
    ) -> None:
        self.campaign_id = campaign_id
        self.baseline = baseline
        self.baseline_hash = baseline_hash
        self.optimization_context = optimization_context
        self.store = store
        self.repository = repository
        self.optimizer = optimizer
        self.dca = dca
        self.i2c = i2c
        self.bert = bert
        self.clock = clock
        self.policy = policy or OrchestratorPolicy()
        self.fir_proposer = fir_proposer
        self.parameter_specs = parameter_specs
        self.fir_authorization_spec = fir_authorization_spec
        self.tdecq_only = tdecq_only
        self.continuous_i2c_only = continuous_i2c_only
        self._measurement_by_observation: dict[str, MeasurementResult] = {}
        self._execution_owner_id = f"orchestrator-{uuid4().hex}"
        self._execution_lock = RLock()
        self._execution_depth = 0

    @_execution_guarded
    def start_preflight(self) -> PreflightResult:
        """Enter PREFLIGHT through the state machine and run read-only checks."""

        self._require_state(CampaignState.DRAFT)
        self._transition(
            CampaignState.PREFLIGHT,
            event_type="PREFLIGHT_STARTED",
        )
        return self.preflight()

    @_execution_guarded
    def plan_stage(
        self,
        stage: CampaignState,
        *,
        max_candidates: int,
        max_duration_s: float,
    ) -> StageAuthorization:
        """Freeze a bounded stage authorization around an exact readback snapshot."""

        return self._construct_stage_authorization(
            stage,
            max_candidates=max_candidates,
            time_policy=AuthorizationTimePolicy.BOUNDED,
            max_duration_s=max_duration_s,
        )

    def _plan_continuous_unbounded_stage(
        self,
        stage: CampaignState,
        *,
        max_candidates: int,
    ) -> StageAuthorization:
        if self.continuous_i2c_only is not True:
            raise OrchestratorError(
                "continuous commission requires an I2C-only orchestrator"
            )
        if stage not in {
            CampaignState.RESTORE_B0,
            CampaignState.I2C_SCREENING,
        }:
            raise OrchestratorError(
                "continuous unbounded authorization permits only RESTORE_B0 "
                "and I2C_SCREENING"
            )
        return self._construct_stage_authorization(
            stage,
            max_candidates=max_candidates,
            time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
            max_duration_s=None,
        )

    def _construct_stage_authorization(
        self,
        stage: CampaignState,
        *,
        max_candidates: int,
        time_policy: AuthorizationTimePolicy,
        max_duration_s: float | None,
    ) -> StageAuthorization:
        """Freeze a complete stage authorization around an exact readback snapshot."""

        expected_states = {
            CampaignState.RESTORE_B0: CampaignState.AWAIT_BASELINE_APPROVAL,
            CampaignState.I2C_SCREENING: CampaignState.AWAIT_COARSE_APPROVAL,
            CampaignState.I2C_FINE: CampaignState.AWAIT_FINE_APPROVAL,
            CampaignState.FIR_APPLY: CampaignState.AWAIT_FIR_APPROVAL,
        }
        expected = expected_states.get(stage)
        if expected is not None:
            self._require_state(expected)
        if stage not in MUTATING_STATES:
            raise OrchestratorError(
                f"stage {stage.value} is not an authorizable mutating stage"
            )
        snapshot = self._read_current_settings()
        now = self.clock.now()
        parameter_specs = (
            self.parameter_specs
            if stage
            in {
                CampaignState.I2C_SCREENING,
                CampaignState.I2C_COARSE,
                CampaignState.I2C_FINE,
                CampaignState.I2C_RETUNE,
            }
            else ()
        )
        if stage in {
            CampaignState.I2C_SCREENING,
            CampaignState.I2C_COARSE,
            CampaignState.I2C_FINE,
            CampaignState.I2C_RETUNE,
        } and not parameter_specs:
            raise OrchestratorError(
                f"stage {stage.value} has no configured safe parameter specs"
            )
        fir_spec = (
            self.fir_authorization_spec
            if stage is CampaignState.FIR_APPLY
            else None
        )
        if stage is CampaignState.FIR_APPLY and fir_spec is None:
            raise OrchestratorError("FIR stage has no configured safety envelope")
        unsigned = StageAuthorization(
            authorization_id="pending",
            campaign_id=self.campaign_id,
            mode=AuthorizationMode.PER_STAGE,
            time_policy=time_policy,
            stage=stage,
            baseline_hash=self.baseline_hash,
            rollback_hash=rollback_hash_for(snapshot),
            parameter_specs=parameter_specs,
            fir_spec=fir_spec,
            max_candidates=max_candidates,
            max_duration_s=max_duration_s,
            issued_at=now,
            expires_at=(
                now + timedelta(seconds=max_duration_s + 300.0)
                if time_policy is AuthorizationTimePolicy.BOUNDED
                and max_duration_s is not None
                else None
            ),
            rollback_snapshot=snapshot,
        )
        authorization = with_authorization_id(unsigned)
        self._event(
            "STAGE_PLANNED",
            {"authorization": authorization.model_dump(mode="json")},
        )
        return authorization

    @_execution_guarded
    def preflight(self) -> PreflightResult:
        state = self._current_state()
        if state == CampaignState.RECOVERY_REQUIRED:
            raise RecoveryRequiredError(
                "campaign is RECOVERY_REQUIRED; preflight cannot contact hardware"
            )
        self._require_state(CampaignState.PREFLIGHT)

        dca_report = self.dca.preflight()
        i2c_report = self.i2c.preflight()
        i2c_readback = dict(self.i2c.readback())
        bert_report = self.bert.preflight()
        bert_readback = dict(self.bert.readback())
        reports = (
            ("DCA", dca_report),
            ("I2C", i2c_report),
            ("BERT", bert_report),
        )
        for label, report in reports:
            if report.get("identity_verified") is not True:
                self._preflight_failure(f"{label} identity is not verified")
            if report.get("configuration_verified") is not True:
                self._preflight_failure(f"{label} configuration is not verified")
        if not i2c_readback:
            self._preflight_failure("I2C readback is empty")
        if not self._bert_readback_is_well_formed(bert_readback):
            self._preflight_failure("BERT configuration readback is invalid")

        result = PreflightResult(
            verified=True,
            dca=dca_report,
            i2c=i2c_report,
            bert=bert_report,
            i2c_readback=i2c_readback,
            bert_readback=bert_readback,
        )
        self._event(
            "PREFLIGHT_COMPLETED",
            {
                "verified": True,
                "dca": dict(dca_report),
                "i2c": dict(i2c_report),
                "bert": dict(bert_report),
                "i2c_readback": i2c_readback,
                "bert_pattern": bert_readback["pattern"],
            },
        )
        self._transition(
            CampaignState.AWAIT_BASELINE_APPROVAL,
            event_type="STATE_TRANSITION",
        )
        return result

    @_execution_guarded
    def restore_b0(self, authorization: StageAuthorization) -> AppliedSettings:
        self._require_state(CampaignState.AWAIT_BASELINE_APPROVAL)
        self._transition(
            CampaignState.RESTORE_B0,
            event_type="B0_RESTORE_STARTED",
            authorization=authorization,
        )
        self._reserve_authorized_operation(authorization)
        requested_i2c = self.baseline.i2c.model_dump()
        try:
            applied_i2c = dict(self.i2c.apply(requested_i2c))
            readback_i2c = dict(self.i2c.readback())
            if self.continuous_i2c_only:
                bert_readback = None
            else:
                self.bert.apply(self.baseline.pattern, self.baseline.fir)
                bert_readback = dict(self.bert.readback())
        except Exception as exc:
            self._hardware_failure(
                authorization, f"B0 write or readback failed: {exc}"
            )
            raise AssertionError("unreachable")

        if readback_i2c != requested_i2c or (
            bert_readback is not None
            and not self._bert_matches(
                bert_readback,
                self.baseline.pattern,
                self.baseline.fir,
            )
        ):
            self._hardware_failure(authorization, "B0 readback mismatch")

        applied = AppliedSettings(
            requested_i2c=requested_i2c,
            quantized_i2c=applied_i2c,
            applied_i2c=applied_i2c,
            readback_i2c=readback_i2c,
            requested_fir=(
                None if self.continuous_i2c_only else self.baseline.fir
            ),
            applied_fir=(
                None if self.continuous_i2c_only else self.baseline.fir
            ),
            readback_fir=(
                None
                if bert_readback is None
                else self._fir_from_readback(bert_readback)
            ),
        )
        self._event(
            "B0_READBACK_VERIFIED",
            {"settings": applied.model_dump(mode="json")},
        )
        self._transition(
            CampaignState.BASELINE_MEASUREMENT,
            event_type="B0_RESTORE_COMPLETED",
        )
        return applied

    @_execution_guarded
    def prepare_continuous_commission(
        self,
        *,
        max_candidates: int,
    ) -> ContinuousCommissionPreparation:
        """Enter the existing I2C-only screening stage from verified preflight."""

        if self.continuous_i2c_only is not True:
            raise OrchestratorError(
                "continuous commission requires an I2C-only orchestrator"
            )
        if type(max_candidates) is not int or max_candidates <= 0:
            raise OrchestratorError(
                "continuous commission max_candidates must be a positive integer"
            )
        restore_authorization = self._plan_continuous_unbounded_stage(
            CampaignState.RESTORE_B0,
            max_candidates=1,
        )
        try:
            restored = self.restore_b0(restore_authorization)
            measurement = self.dca.measure_single()
            if not self._measurement_has_valid_tdecq(measurement):
                raise OrchestratorError(
                    "continuous baseline measurement has invalid TDECQ"
                )
            observation = self._persist_measurement(
                candidate=None,
                stage=CampaignState.BASELINE_MEASUREMENT,
                settings=restored,
                measurement=measurement,
                authorization=restore_authorization,
            )
            self._event(
                "CONTINUOUS_BASELINE_COMPLETED",
                {"observation_id": observation.observation_id},
            )
            self._transition(
                CampaignState.AWAIT_COARSE_APPROVAL,
                event_type="STATE_TRANSITION",
            )
            authorization = self._plan_continuous_unbounded_stage(
                CampaignState.I2C_SCREENING,
                max_candidates=max_candidates,
            )
            self.enter_stage(authorization)
        except Exception:
            self._recover_continuous_commission_to_b0()
            raise
        return ContinuousCommissionPreparation(
            observation=observation,
            authorization=authorization,
        )

    @_execution_guarded
    def run_baseline_measurement(self) -> RepeatedMeasurementReport:
        self._require_state(CampaignState.BASELINE_MEASUREMENT)
        report = self._run_repeated_measurements(
            stage=CampaignState.BASELINE_MEASUREMENT,
            repeat_count=5,
            candidate=None,
        )
        valid = (
            report.acquisition_valid_count >= 4 and report.er_valid_count >= 4
        )
        report = self._with_report_validity(report, valid)
        if valid:
            self._transition(
                CampaignState.AWAIT_COARSE_APPROVAL,
                event_type="BASELINE_COMPLETED",
                payload=self._report_payload(report),
            )
        else:
            self._event("BASELINE_INVALID", self._report_payload(report))
        return report

    @_execution_guarded
    def run_one_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
        *,
        repeat_count: int = 1,
        candidate_index: int = 0,
        stage_started_monotonic_s: float | None = None,
    ) -> tuple[Observation, ...]:
        self._ensure_not_paused()
        current_state = self._current_state()
        if current_state != candidate.stage:
            raise OrchestratorError(
                f"candidate stage {candidate.stage.value} does not match "
                f"current state {current_state.value}"
            )
        if type(repeat_count) is not int or repeat_count <= 0:
            raise ValueError("repeat_count must be a positive integer")
        del candidate_index, stage_started_monotonic_s

        reserved_index, elapsed_s = self._reserve_authorized_candidate(
            authorization,
            candidate,
        )

        previous_i2c = dict(self.i2c.readback())
        previous_fir: FirSettings | None = None
        if candidate.requested_fir is not None:
            previous_bert = dict(self.bert.readback())
            previous_fir = self._fir_from_readback(previous_bert)
        validate_candidate(
            authorization,
            candidate,
            previous_readback=previous_i2c,
            previous_fir=previous_fir,
            now=self.clock.now(),
            current_campaign_id=self.campaign_id,
            current_baseline_hash=self.baseline_hash,
            candidate_index=reserved_index,
            elapsed_s=elapsed_s,
        )

        if candidate.requested_fir is not None:
            settings = self._apply_fir_candidate(
                authorization,
                candidate,
                previous_i2c=previous_i2c,
            )
        else:
            settings = self._apply_i2c_candidate(
                authorization,
                candidate,
                previous_i2c=previous_i2c,
                candidate_index=reserved_index,
            )

        observations: list[Observation] = []
        for _ in range(repeat_count):
            try:
                measurement = self.dca.measure_single()
            except Exception as exc:
                self._hardware_failure(
                    authorization,
                    f"DCA measurement failed after candidate write: {exc}",
                )
                raise AssertionError("unreachable")
            if (
                not self.tdecq_only
                and not self._measurement_has_valid_er(measurement)
            ):
                self._event(
                    "OBSERVATION_INVALID",
                    {
                        "candidate_id": candidate.candidate_id,
                        "acquisition_valid": measurement.acquisition_valid,
                        "er_status": measurement.er_status,
                        "tdecq_status": measurement.tdecq_status,
                    },
                )
                continue
            observation = self._persist_measurement(
                candidate=candidate,
                stage=candidate.stage,
                settings=settings,
                measurement=measurement,
                authorization=authorization,
            )
            observations.append(observation)
        return tuple(observations)

    @_execution_guarded
    def run_continuous_i2c_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
    ) -> tuple[Observation, ...]:
        """Execute one I2C-only candidate through the existing authorization chain."""

        if candidate.requested_fir is not None:
            raise OrchestratorError("continuous optimization forbids FIR candidates")
        return self.run_one_candidate(
            authorization,
            candidate,
            repeat_count=1,
        )

    @_execution_guarded
    def measure_current_i2c_state(self) -> Observation:
        """Measure the current verified I2C state without any write or BERT access."""

        self._ensure_not_paused()
        stage = self._current_state()
        if stage not in {
            CampaignState.I2C_SCREENING,
            CampaignState.I2C_COARSE,
            CampaignState.I2C_FINE,
            CampaignState.I2C_RETUNE,
        }:
            raise OrchestratorError(
                f"continuous measurement is unavailable in {stage.value}"
            )
        readback = dict(self.i2c.readback())
        try:
            measurement = self.dca.measure_single()
        except Exception as exc:
            raise OrchestratorError(
                f"continuous DCA measurement failed: {type(exc).__name__}"
            ) from None
        if not self._measurement_has_valid_tdecq(measurement):
            raise OrchestratorError("continuous DCA measurement has invalid TDECQ")
        settings = AppliedSettings(readback_i2c=readback)
        return self._persist_measurement(
            candidate=None,
            stage=stage,
            settings=settings,
            measurement=measurement,
            authorization=None,
        )

    @_execution_guarded
    def capture_continuous_preflight_observation(
        self,
    ) -> ContinuousPreflightResult:
        """Verify DCA/I2C and persist one eye without DUT or BERT writes."""

        if not self.tdecq_only:
            raise OrchestratorError(
                "continuous preflight requires an I2C-only orchestrator"
            )
        state = self._current_state()
        if state is CampaignState.RECOVERY_REQUIRED:
            raise RecoveryRequiredError(
                "campaign is RECOVERY_REQUIRED; preflight cannot contact hardware"
            )
        if state is CampaignState.DRAFT:
            self._transition(
                CampaignState.PREFLIGHT,
                event_type="CONTINUOUS_PREFLIGHT_STARTED",
            )
        elif state not in {
            CampaignState.PREFLIGHT,
            CampaignState.AWAIT_BASELINE_APPROVAL,
        }:
            raise OrchestratorError(
                f"continuous preflight is unavailable in {state.value}"
            )

        try:
            dca_report = self.dca.preflight()
        except Exception as exc:
            self._preflight_failure(
                f"DCA preflight failed: {type(exc).__name__}"
            )
            raise AssertionError("unreachable")
        try:
            i2c_report = self.i2c.preflight()
        except Exception as exc:
            self._preflight_failure(
                f"I2C preflight failed: {type(exc).__name__}"
            )
            raise AssertionError("unreachable")
        for label, report in (("DCA", dca_report), ("I2C", i2c_report)):
            if report.get("identity_verified") is not True:
                self._preflight_failure(f"{label} identity is not verified")
            if report.get("configuration_verified") is not True:
                self._preflight_failure(f"{label} configuration is not verified")

        try:
            i2c_readback = dict(self.i2c.readback())
        except Exception as exc:
            self._preflight_failure(
                f"I2C readback failed: {type(exc).__name__}"
            )
            raise AssertionError("unreachable")
        if not i2c_readback:
            self._preflight_failure("I2C readback is empty")
        try:
            measurement = self.dca.measure_single()
        except Exception as exc:
            self._preflight_failure(
                f"continuous DCA measurement failed: {type(exc).__name__}"
            )
            raise AssertionError("unreachable")
        if not self._measurement_has_valid_tdecq(measurement):
            raw = measurement.raw_responses
            self._preflight_failure(
                "continuous DCA measurement has invalid TDECQ: "
                f"er_status={measurement.er_status!r}, "
                f"tdecq_status={measurement.tdecq_status!r}, "
                f"tdecq_invalid_reason={measurement.tdecq_invalid_reason!r}, "
                f"status_polls={raw.get('status_polls')!r}, "
                f"er_details_polls={raw.get('er_details_polls')!r}, "
                f"tdecq_details_polls={raw.get('tdecq_details_polls')!r}"
            )

        try:
            observation = self._persist_measurement(
                candidate=None,
                stage=CampaignState.PREFLIGHT,
                settings=AppliedSettings(readback_i2c=i2c_readback),
                measurement=measurement,
                authorization=None,
            )
        except RepositoryError:
            self._transition(
                CampaignState.FAILED,
                event_type="PREFLIGHT_FAILED",
                payload={
                    "reason": (
                        "continuous preflight observation persistence failed"
                    )
                },
            )
            raise
        self._event(
            "CONTINUOUS_PREFLIGHT_COMPLETED",
            {
                "verified": True,
                "dca": dict(dca_report),
                "i2c": dict(i2c_report),
                "i2c_readback": i2c_readback,
                "observation_id": observation.observation_id,
            },
        )
        if self._current_state() is CampaignState.PREFLIGHT:
            self._transition(
                CampaignState.AWAIT_BASELINE_APPROVAL,
                event_type="STATE_TRANSITION",
            )
        return ContinuousPreflightResult(
            verified=True,
            dca=dca_report,
            i2c=i2c_report,
            i2c_readback=i2c_readback,
            observation=observation,
        )

    @_execution_guarded
    def restore_static_i2c_b0(self) -> AppliedSettings:
        """Restore and verify only the static I2C portion of B0."""

        try:
            restored = self._apply_static_i2c_b0()
        except Exception:
            if (
                self.continuous_i2c_only
                and self._current_state() is CampaignState.I2C_SCREENING
            ):
                self._transition(
                    CampaignState.RECOVERY_REQUIRED,
                    event_type="CONTINUOUS_B0_RESTORE_FAILED",
                )
            raise
        if (
            self.continuous_i2c_only
            and self._current_state() is CampaignState.I2C_SCREENING
        ):
            self._transition(
                CampaignState.COMPLETED,
                event_type="CONTINUOUS_B0_RESTORED",
                continuous_completion=True,
            )
        return restored

    def _apply_static_i2c_b0(self) -> AppliedSettings:
        requested = self.baseline.i2c.model_dump()
        try:
            applied = dict(self.i2c.apply(requested))
            readback = dict(self.i2c.readback())
        except Exception as exc:
            raise OrchestratorError(
                f"static I2C B0 restore failed: {type(exc).__name__}"
            ) from None
        if applied != requested or any(
            readback.get(name) != value for name, value in requested.items()
        ):
            raise OrchestratorError("static I2C B0 restore readback mismatch")
        restored = AppliedSettings(
            requested_i2c=requested,
            quantized_i2c=applied,
            applied_i2c=applied,
            readback_i2c=readback,
        )
        return restored

    @_execution_guarded
    def recover_continuous_commission_failure(self) -> AppliedSettings:
        """Restore B0 and persist a failed-safe state after assembly failure."""

        if self.continuous_i2c_only is not True:
            raise OrchestratorError(
                "continuous failure recovery requires an I2C-only orchestrator"
            )
        current = self._current_state()
        if current is CampaignState.RECOVERY_REQUIRED:
            raise RecoveryRequiredError(
                "continuous campaign is already RECOVERY_REQUIRED"
            )
        if current is CampaignState.COMPLETED:
            raise OrchestratorError(
                "completed continuous campaign cannot enter failure recovery"
            )
        return self._recover_continuous_commission_to_b0()

    def _recover_continuous_commission_to_b0(self) -> AppliedSettings:
        """Best-effort B0 recovery for every failure after commission starts."""

        try:
            restored = self._apply_static_i2c_b0()
        except Exception as exc:
            current = self._current_state()
            if current is not CampaignState.RECOVERY_REQUIRED:
                self.store.save_transition(
                    self.campaign_id,
                    current,
                    CampaignState.RECOVERY_REQUIRED,
                    event_type="CONTINUOUS_B0_RECOVERY_FAILED",
                    payload={"reason": "static B0 restore could not be verified"},
                    created_at=self.clock.now(),
                )
            raise RecoveryRequiredError(
                "continuous commission failed and static B0 could not be restored"
            ) from exc
        current = self._current_state()
        if current not in {
            CampaignState.FAILED,
            CampaignState.RECOVERY_REQUIRED,
            CampaignState.COMPLETED,
        }:
            if current not in {
                CampaignState.ABORTING,
                CampaignState.ROLLING_BACK,
            }:
                current = self._transition(
                    CampaignState.ABORTING,
                    event_type="CONTINUOUS_COMMISSION_ABORTING",
                )
            if current is CampaignState.ABORTING:
                current = self._transition(
                    CampaignState.ROLLING_BACK,
                    event_type="STATE_TRANSITION",
                )
            if current is CampaignState.ROLLING_BACK:
                self._event("CONTINUOUS_B0_RECOVERY_VERIFIED")
                self._transition(
                    CampaignState.FAILED,
                    event_type="CONTINUOUS_COMMISSION_FAILED_SAFE",
                )
        return restored

    @_execution_guarded
    def run_screening_stage(
        self,
        authorization: StageAuthorization,
        *,
        next_authorization: StageAuthorization | None = None,
    ) -> StageSummary:
        if next_authorization is not None:
            self._validate_stage_authorization(
                next_authorization,
                CampaignState.I2C_COARSE,
            )
        summary = self._run_optimizer_stage(
            authorization,
            stage=CampaignState.I2C_SCREENING,
            repeats=1,
            restore_after_candidate=True,
        )
        if next_authorization is not None:
            self._transition(
                CampaignState.I2C_COARSE,
                event_type="AUTHORIZED_STAGE_ENTERED",
                authorization=next_authorization,
            )
        return summary

    @_execution_guarded
    def run_coarse_stage(self, authorization: StageAuthorization) -> StageSummary:
        return self._run_optimizer_stage(
            authorization,
            stage=CampaignState.I2C_COARSE,
            repeats=3,
            restore_after_candidate=False,
        )

    @_execution_guarded
    def run_fine_stage(self, authorization: StageAuthorization) -> StageSummary:
        self._ensure_active_stage(CampaignState.I2C_FINE)
        self._activate_stage_authorization(authorization, CampaignState.I2C_FINE)
        started = self.clock.monotonic()
        optimizer_history: list[TuningObservation] = []
        campaign_observations: list[Observation] = []
        best_observation: Observation | None = None
        best_candidate: Candidate | None = None
        best_candidate_index = 0
        candidates_used = 0
        candidates_without_new_best = 0

        while candidates_used < authorization.max_candidates:
            suggestion = self.optimizer.suggest(tuple(optimizer_history))
            if not isinstance(suggestion, CandidateSuggestion):
                break
            candidate = Candidate(
                candidate_id=suggestion.candidate_id,
                stage=CampaignState.I2C_FINE,
                requested_i2c=dict(suggestion.requested),
                model_version=suggestion.path.value,
                reason=suggestion.reason,
            )
            candidate_index = candidates_used
            observations = self.run_one_candidate(
                authorization,
                candidate,
                repeat_count=1,
                candidate_index=candidate_index,
                stage_started_monotonic_s=started,
            )
            candidates_used += 1
            campaign_observations.extend(observations)
            optimizer_history.extend(
                self._optimizer_observation(observation)
                for observation in observations
            )

            feasible = next(
                (observation for observation in observations if observation.feasible),
                None,
            )
            if feasible is not None and (
                best_observation is None
                or self._observation_tdecq(feasible)
                < self._observation_tdecq(best_observation)
            ):
                best_observation = feasible
                best_candidate = candidate
                best_candidate_index = candidate_index
                candidates_without_new_best = 0
            else:
                candidates_without_new_best += 1

            if candidates_without_new_best >= self.policy.fine_patience_candidates:
                break

        convergence_reason = "FINE_STAGE_INFEASIBLE"
        confirmed = False
        reference_tdecq = self.policy.fir_reference_tdecq_db
        if (
            best_observation is not None
            and best_candidate is not None
            and reference_tdecq is not None
            and self._is_meaningful_improvement(
                reference_tdecq - self._observation_tdecq(best_observation)
            )
        ):
            confirmation = self.run_one_candidate(
                authorization,
                best_candidate,
                repeat_count=self.policy.best_point_confirmation_repeats,
                candidate_index=best_candidate_index,
                stage_started_monotonic_s=started,
            )
            campaign_observations.extend(confirmation)
            confirmed = self._confirmed_fine_observations(
                confirmation,
                reference_tdecq=reference_tdecq,
            )
            convergence_reason = (
                "FINE_STAGE_FEASIBLE"
                if confirmed
                else "FINE_CONFIRMATION_FAILED"
            )
        elif best_observation is not None:
            convergence_reason = "FINE_NO_MEANINGFUL_IMPROVEMENT"
        elif any(
            self._observation_is_partial_convergence(observation)
            for observation in campaign_observations
        ):
            convergence_reason = "PARTIAL_CONVERGENCE"

        summary = self._stage_summary(
            CampaignState.I2C_FINE,
            convergence_reason=convergence_reason,
            candidates_used=candidates_used,
            observations=campaign_observations,
            started=started,
        )
        self._event("STAGE_COMPLETED", summary.model_dump(mode="json"))
        if confirmed:
            self._transition(
                CampaignState.AWAIT_FIR_APPROVAL,
                event_type="STATE_TRANSITION",
            )
        return summary

    @_execution_guarded
    def run_retune_stage(self, authorization: StageAuthorization) -> StageSummary:
        return self._run_optimizer_stage(
            authorization,
            stage=CampaignState.I2C_RETUNE,
            repeats=1,
            restore_after_candidate=False,
        )

    @_execution_guarded
    def run_fir_stage(
        self,
        authorization: StageAuthorization,
        *,
        retune_authorizations: Iterable[StageAuthorization] | None = None,
    ) -> StageSummary:
        self._ensure_active_stage(CampaignState.FIR_APPLY)
        self._activate_stage_authorization(authorization, CampaignState.FIR_APPLY)
        if retune_authorizations is None:
            raise OrchestratorError(
                "separate I2C_RETUNE authorization is required before FIR hardware"
            )
        retune_authorization_sequence = tuple(retune_authorizations)
        if not retune_authorization_sequence:
            raise OrchestratorError(
                "at least one I2C_RETUNE authorization is required"
            )
        for retune_authorization in retune_authorization_sequence:
            self._validate_stage_authorization(
                retune_authorization,
                CampaignState.I2C_RETUNE,
            )
        if self.fir_proposer is None:
            raise OrchestratorError("a pure FIR proposer is required")

        started = self.clock.monotonic()
        stage_entry_fir = authorization.rollback_snapshot.readback_fir
        if stage_entry_fir is None:
            raise OrchestratorError("FIR authorization has no stage-entry FIR")
        full_target = self.fir_proposer.propose(stage_entry_fir)
        if not isinstance(full_target, FirSettings):
            raise OrchestratorError("FIR proposer must return strict FirSettings")
        candidate_stream = (
            Candidate(
                candidate_id="fir-formula-direct",
                stage=CampaignState.FIR_APPLY,
                requested_fir=full_target,
                model_version="fir-formula-direct-v1.1",
                reason="direct FIR formula proposal",
            ),
        )

        statically_considered = candidate_stream[: authorization.max_candidates]
        prevalidation_elapsed_s = self.clock.monotonic() - started
        for candidate_index, candidate in enumerate(statically_considered):
            validate_candidate_static(
                authorization,
                candidate,
                now=self.clock.now(),
                current_campaign_id=self.campaign_id,
                current_baseline_hash=self.baseline_hash,
                candidate_index=candidate_index,
                elapsed_s=prevalidation_elapsed_s,
            )

        considered = statically_considered[: len(retune_authorization_sequence)]
        all_observations: list[Observation] = []
        optimizer_history: list[TuningObservation] = []
        candidates_used = 0
        accepted_updates = 0
        rejected = False
        last_accepted_snapshot = authorization.rollback_snapshot
        reference_tdecq = self.policy.fir_reference_tdecq_db
        reference_er = self.policy.fir_reference_er_db
        transition_authorization = retune_authorization_sequence[0]

        for candidate_index, candidate in enumerate(considered):
            retune_authorization = retune_authorization_sequence[candidate_index]
            if retune_authorization.rollback_snapshot != last_accepted_snapshot:
                raise OrchestratorError(
                    "I2C_RETUNE authorization rollback snapshot must equal the "
                    "last accepted FIR and I2C point"
                )

            fir_settings = self._apply_fir_for_outer_candidate(
                authorization,
                candidate,
                candidate_index=candidate_index,
                stage_started_monotonic_s=started,
            )
            settings, retune_observations = self._run_inner_fir_retune(
                retune_authorization,
                outer_authorization=authorization,
                fir_settings=fir_settings,
                optimizer_history=tuple(optimizer_history),
                stage_started_monotonic_s=started,
            )
            optimizer_history.extend(retune_observations)
            observations = self._measure_written_candidate(
                candidate=candidate,
                settings=settings,
                repeat_count=3,
                rollback_authorization=retune_authorization,
            )
            candidates_used += 1
            all_observations.extend(observations)
            optimizer_history.extend(
                self._optimizer_observation(observation)
                for observation in observations
            )
            if self._fir_candidate_accepted(
                observations,
                reference_tdecq=reference_tdecq,
                reference_er=reference_er,
            ):
                accepted_updates += 1
                accepted_fir = candidate.requested_fir
                assert accepted_fir is not None
                last_accepted_snapshot = AppliedSettings(
                    readback_i2c=settings.readback_i2c,
                    readback_fir=accepted_fir,
                )
                reference_tdecq = median(
                    self._observation_tdecq(observation)
                    for observation in observations
                )
                reference_er = median(
                    float(observation.metrics["er_db"])
                    for observation in observations
                    if observation.metrics.get("er_db") is not None
                )
                transition_authorization = retune_authorization
                if accepted_updates >= self.policy.max_accepted_fir_updates:
                    break
                continue

            rejected = True
            if not self._restore_snapshot_once(last_accepted_snapshot):
                self._failed_restore_to_recovery(
                    event_type="FIR_RESTORE_FAILED",
                    message="rejected FIR restore readback mismatch",
                )
            self._event("FIR_LAST_ACCEPTED_RESTORED")
            transition_authorization = retune_authorization
            break

        if accepted_updates >= self.policy.max_accepted_fir_updates:
            convergence_reason = "FIR_ACCEPTED_LIMIT"
        elif accepted_updates > 0:
            convergence_reason = "FIR_ACCEPTED_UPDATE"
        else:
            convergence_reason = "FIR_NO_ACCEPTED_UPDATE"
        summary = self._stage_summary(
            CampaignState.FIR_APPLY,
            convergence_reason=convergence_reason,
            candidates_used=candidates_used,
            observations=all_observations,
            started=started,
        )
        self._event("STAGE_COMPLETED", summary.model_dump(mode="json"))
        if accepted_updates > 0:
            self._transition(
                CampaignState.I2C_RETUNE,
                event_type="AUTHORIZED_STAGE_ENTERED",
                authorization=transition_authorization,
            )
        return summary

    def _fir_candidate_accepted(
        self,
        observations: tuple[Observation, ...],
        *,
        reference_tdecq: float | None,
        reference_er: float | None,
    ) -> bool:
        if len(observations) != 3 or not all(
            observation.feasible for observation in observations
        ):
            return False
        if reference_tdecq is None or reference_er is None:
            return False
        tdecq_values = [
            float(observation.metrics["tdecq_db"])
            for observation in observations
            if observation.metrics.get("tdecq_db") is not None
        ]
        er_values = [
            float(observation.metrics["er_db"])
            for observation in observations
            if observation.metrics.get("er_db") is not None
        ]
        if len(tdecq_values) != 3 or len(er_values) != 3:
            return False
        return (
            self._is_meaningful_improvement(
                reference_tdecq - median(tdecq_values)
            )
            and median(er_values) >= self.policy.constrained_er_min_db
            and median(er_values) >= reference_er
        )

    def _apply_fir_for_outer_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
        *,
        candidate_index: int,
        stage_started_monotonic_s: float,
    ) -> AppliedSettings:
        del candidate_index, stage_started_monotonic_s
        reserved_index, elapsed_s = self._reserve_authorized_candidate(
            authorization,
            candidate,
        )
        previous_i2c = dict(self.i2c.readback())
        previous_fir = self._fir_from_readback(dict(self.bert.readback()))
        validate_candidate(
            authorization,
            candidate,
            previous_readback=previous_i2c,
            previous_fir=previous_fir,
            now=self.clock.now(),
            current_campaign_id=self.campaign_id,
            current_baseline_hash=self.baseline_hash,
            candidate_index=reserved_index,
            elapsed_s=elapsed_s,
        )
        settings = self._apply_fir_candidate(
            authorization,
            candidate,
            previous_i2c=previous_i2c,
        )
        try:
            validate_candidate(
                authorization,
                candidate,
                previous_readback=previous_i2c,
                previous_fir=previous_fir,
                now=self.clock.now(),
                current_campaign_id=self.campaign_id,
                current_baseline_hash=self.baseline_hash,
                candidate_index=reserved_index,
                elapsed_s=self._authorization_usage(authorization).elapsed_s,
            )
        except Exception as exc:
            self._hardware_failure(
                authorization,
                f"FIR candidate exceeded authorization after write: {exc}",
            )
            raise AssertionError("unreachable")
        return settings

    def _run_inner_fir_retune(
        self,
        authorization: StageAuthorization,
        *,
        outer_authorization: StageAuthorization,
        fir_settings: AppliedSettings,
        optimizer_history: tuple[TuningObservation, ...],
        stage_started_monotonic_s: float,
    ) -> tuple[AppliedSettings, tuple[TuningObservation, ...]]:
        del stage_started_monotonic_s
        self._activate_nested_authorization(outer_authorization, authorization)
        history = list(optimizer_history)
        new_observations: list[TuningObservation] = []
        latest_settings: AppliedSettings | None = None
        candidate_index = 0
        while candidate_index < authorization.max_candidates:
            suggestion = self.optimizer.suggest(tuple(history))
            if not isinstance(suggestion, CandidateSuggestion):
                break
            candidate = Candidate(
                candidate_id=suggestion.candidate_id,
                stage=CampaignState.I2C_RETUNE,
                requested_i2c=dict(suggestion.requested),
                model_version=suggestion.path.value,
                reason=suggestion.reason,
            )
            reserved_index, elapsed_s = self._reserve_authorized_candidate(
                authorization,
                candidate,
                outer_authorization=outer_authorization,
            )
            previous_i2c = dict(self.i2c.readback())
            validate_candidate(
                authorization,
                candidate,
                previous_readback=previous_i2c,
                now=self.clock.now(),
                current_campaign_id=self.campaign_id,
                current_baseline_hash=self.baseline_hash,
                candidate_index=reserved_index,
                elapsed_s=elapsed_s,
            )
            i2c_settings = self._apply_i2c_candidate(
                authorization,
                candidate,
                previous_i2c=previous_i2c,
                candidate_index=reserved_index,
            )
            latest_settings = self._combine_retuned_fir_settings(
                i2c_settings,
                fir_settings,
            )
            observation = self._measure_inner_fir_retune_candidate(
                authorization,
                candidate=candidate,
                settings=latest_settings,
            )
            if observation is not None:
                optimizer_observation = self._optimizer_observation(observation)
                history.append(optimizer_observation)
                new_observations.append(optimizer_observation)
            candidate_index += 1

        if latest_settings is None:
            i2c_readback = dict(self.i2c.readback())
            latest_settings = self._combine_retuned_fir_settings(
                AppliedSettings(readback_i2c=i2c_readback),
                fir_settings,
            )
        return latest_settings, tuple(new_observations)

    @staticmethod
    def _combine_retuned_fir_settings(
        i2c_settings: AppliedSettings,
        fir_settings: AppliedSettings,
    ) -> AppliedSettings:
        return AppliedSettings(
            requested_i2c=i2c_settings.requested_i2c,
            quantized_i2c=i2c_settings.quantized_i2c,
            applied_i2c=i2c_settings.applied_i2c,
            readback_i2c=i2c_settings.readback_i2c,
            requested_fir=fir_settings.requested_fir,
            applied_fir=fir_settings.applied_fir,
            readback_fir=fir_settings.readback_fir,
        )

    def _measure_inner_fir_retune_candidate(
        self,
        authorization: StageAuthorization,
        *,
        candidate: Candidate,
        settings: AppliedSettings,
    ) -> Observation | None:
        try:
            measurement = self.dca.measure_single()
        except Exception as exc:
            self._hardware_failure(
                authorization,
                f"DCA measurement failed after inner FIR retune write: {exc}",
            )
            raise AssertionError("unreachable")
        if not self._measurement_has_valid_er(measurement):
            self._event(
                "OBSERVATION_INVALID",
                {
                    "candidate_id": candidate.candidate_id,
                    "acquisition_valid": measurement.acquisition_valid,
                    "er_status": measurement.er_status,
                    "tdecq_status": measurement.tdecq_status,
                },
            )
            return None
        return self._persist_measurement(
            candidate=candidate,
            stage=CampaignState.I2C_RETUNE,
            settings=settings,
            measurement=measurement,
            authorization=authorization,
        )

    def _measure_written_candidate(
        self,
        *,
        candidate: Candidate,
        settings: AppliedSettings,
        repeat_count: int,
        rollback_authorization: StageAuthorization,
    ) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        for _ in range(repeat_count):
            try:
                measurement = self.dca.measure_single()
            except Exception as exc:
                self._hardware_failure(
                    rollback_authorization,
                    f"DCA measurement failed after FIR write and retune: {exc}",
                )
                raise AssertionError("unreachable")
            if not self._measurement_has_valid_er(measurement):
                self._event(
                    "OBSERVATION_INVALID",
                    {
                        "candidate_id": candidate.candidate_id,
                        "acquisition_valid": measurement.acquisition_valid,
                        "er_status": measurement.er_status,
                        "tdecq_status": measurement.tdecq_status,
                    },
                )
                continue
            observations.append(
                self._persist_measurement(
                    candidate=candidate,
                    stage=CampaignState.FIR_APPLY,
                    settings=settings,
                    measurement=measurement,
                    authorization=rollback_authorization,
                )
            )
        return tuple(observations)

    @_execution_guarded
    def run_final_verification_stage(self) -> FinalVerificationResult:
        self._require_state(CampaignState.FINAL_VERIFICATION)
        report = self._run_repeated_measurements(
            stage=CampaignState.FINAL_VERIFICATION,
            repeat_count=5,
            candidate=None,
        )
        report = self._with_report_validity(
            report,
            report.acquisition_valid_count == 5
            and report.er_valid_count == 5
            and report.tdecq_valid_count == 5,
        )
        outcome: FinalOutcome | None = None
        if report.valid:
            outcome = self._final_outcome(report)
            payload = self._report_payload(report)
            payload["outcome"] = outcome
            self._transition(
                CampaignState.COMPLETED,
                event_type="FINAL_VERIFICATION_COMPLETED",
                payload=payload,
            )
        else:
            self._event("FINAL_VERIFICATION_INVALID", self._report_payload(report))
        return FinalVerificationResult(report=report, outcome=outcome)

    def _final_outcome(self, report: RepeatedMeasurementReport) -> FinalOutcome:
        er_median = report.er_median_db
        tdecq_median = report.tdecq_median_db
        reference_er = self.policy.fir_reference_er_db
        reference_tdecq = self.policy.fir_reference_tdecq_db
        if (
            er_median is None
            or tdecq_median is None
            or reference_er is None
            or reference_tdecq is None
        ):
            return "NO_IMPROVEMENT"
        improved = self._is_meaningful_improvement(
            reference_tdecq - tdecq_median
        )
        if (
            improved
            and er_median >= self.policy.constrained_er_min_db
            and er_median >= reference_er
        ):
            return "SUCCESS"
        if improved and er_median >= self.policy.coarse_er_gate_db:
            return "PARTIAL_CONVERGENCE"
        return "NO_IMPROVEMENT"

    @_execution_guarded
    def pause(self) -> CampaignState:
        self._ensure_not_paused()
        resume_state = self._current_state()
        return self._transition(
            CampaignState.PAUSED,
            event_type="CAMPAIGN_PAUSED",
            payload={"resume_state": resume_state.value},
        )

    @_execution_guarded
    def resume(
        self,
        authorization: StageAuthorization | None = None,
    ) -> CampaignState:
        self._require_state(CampaignState.PAUSED)
        resume_state = self._persisted_pause_resume_state()
        return self._transition(
            resume_state,
            event_type="CAMPAIGN_RESUMED",
            authorization=authorization,
            resume_state=resume_state,
        )

    def _persisted_pause_resume_state(self) -> CampaignState:
        pause_event = next(
            (
                event
                for event in reversed(self.store.events(self.campaign_id))
                if event.event_type == "CAMPAIGN_PAUSED"
            ),
            None,
        )
        if pause_event is None or not isinstance(pause_event.payload, Mapping):
            raise OrchestratorError("PAUSED campaign has no persisted resume state")
        raw_resume_state = pause_event.payload.get("resume_state")
        try:
            resume_state = CampaignState(raw_resume_state)
        except (TypeError, ValueError) as exc:
            raise OrchestratorError("persisted resume state is invalid") from exc
        return resume_state

    @_execution_guarded
    def enter_stage(self, authorization: StageAuthorization) -> CampaignState:
        """Enter the next authorized mutating stage without touching hardware."""

        target = authorization.stage
        if target not in MUTATING_STATES:
            raise OrchestratorError(
                f"authorization target {target.value} is not a mutating stage"
            )
        return self._transition(
            target,
            event_type="AUTHORIZED_STAGE_ENTERED",
            authorization=authorization,
        )

    @_execution_guarded
    def abort(self, authorization: StageAuthorization) -> CampaignState:
        current = self._current_state()
        resume_state = (
            self._persisted_pause_resume_state()
            if current == CampaignState.PAUSED
            else None
        )
        authorization_stage = current if resume_state is None else resume_state
        self._validate_stage_authorization(authorization, authorization_stage)
        return self._rollback_to_terminal(
            authorization,
            final_event="ABORTED",
            resume_state=resume_state,
        )

    def _run_optimizer_stage(
        self,
        authorization: StageAuthorization,
        *,
        stage: CampaignState,
        repeats: int,
        restore_after_candidate: bool,
    ) -> StageSummary:
        self._ensure_active_stage(stage)
        if self.tdecq_only and stage is not CampaignState.I2C_SCREENING:
            raise OrchestratorError(
                "TDECQ-only diagnostic paths execute only I2C_SCREENING"
            )
        self._activate_stage_authorization(authorization, stage)
        started = self.clock.monotonic()
        optimizer_history: list[TuningObservation] = []
        campaign_observations: list[Observation] = []
        candidates_used = 0
        convergence_reason = "authorization budget exhausted"
        gate_satisfied = False

        while candidates_used < authorization.max_candidates:
            suggestion = self.optimizer.suggest(tuple(optimizer_history))
            if not isinstance(suggestion, CandidateSuggestion):
                convergence_reason = str(
                    getattr(suggestion, "reason", getattr(suggestion, "kind", "complete"))
                )
                break
            candidate = Candidate(
                candidate_id=suggestion.candidate_id,
                stage=stage,
                requested_i2c=dict(suggestion.requested),
                model_version=suggestion.path.value,
                reason=suggestion.reason,
            )
            observations = self.run_one_candidate(
                authorization,
                candidate,
                repeat_count=repeats,
                candidate_index=candidates_used,
                stage_started_monotonic_s=started,
            )
            candidates_used += 1
            campaign_observations.extend(observations)
            optimizer_history.extend(
                self._optimizer_observation(observation)
                for observation in observations
            )
            if restore_after_candidate:
                if not self._restore_snapshot_once(authorization.rollback_snapshot):
                    self._failed_restore_to_recovery(
                        event_type="SCREENING_RESTORE_FAILED",
                        message=(
                        "screening stage baseline restore readback mismatch"
                        ),
                    )
                self._event("SCREENING_BASELINE_RESTORED")

            if self._candidate_gate_satisfied(stage, observations):
                gate_satisfied = True
                # The optimizer only receives observations through ``suggest``.
                # Deliver the persisted candidate group before ending the stage.
                self.optimizer.suggest(tuple(optimizer_history))
                break

        if not self.tdecq_only and stage == CampaignState.I2C_COARSE:
            convergence_reason = (
                "COARSE_GATE_SATISFIED"
                if gate_satisfied
                else "COARSE_GATE_NOT_SATISFIED"
            )
        elif not self.tdecq_only and stage == CampaignState.I2C_FINE:
            convergence_reason = (
                "FINE_STAGE_FEASIBLE" if gate_satisfied else "FINE_STAGE_INFEASIBLE"
            )
        elif not self.tdecq_only and stage == CampaignState.I2C_RETUNE:
            convergence_reason = (
                "RETUNE_STAGE_FEASIBLE"
                if gate_satisfied
                else "RETUNE_STAGE_INFEASIBLE"
            )

        summary = self._stage_summary(
            stage,
            convergence_reason=convergence_reason,
            candidates_used=candidates_used,
            observations=campaign_observations,
            started=started,
        )
        self._event("STAGE_COMPLETED", summary.model_dump(mode="json"))
        if self.tdecq_only:
            self._transition(
                CampaignState.COMPLETED,
                event_type="DIAGNOSTIC_COMPLETED",
                continuous_completion=(
                    stage is CampaignState.I2C_SCREENING
                    and self.continuous_i2c_only
                ),
            )
            return summary
        if gate_satisfied:
            next_state = {
                CampaignState.I2C_COARSE: CampaignState.AWAIT_FINE_APPROVAL,
                CampaignState.I2C_FINE: CampaignState.AWAIT_FIR_APPROVAL,
                CampaignState.I2C_RETUNE: CampaignState.FINAL_VERIFICATION,
            }.get(stage)
            if next_state is not None:
                self._transition(next_state, event_type="STATE_TRANSITION")
        return summary

    def _candidate_gate_satisfied(
        self,
        stage: CampaignState,
        observations: tuple[Observation, ...],
    ) -> bool:
        if self.tdecq_only:
            return False
        if stage == CampaignState.I2C_COARSE:
            if len(observations) != 3:
                return False
            er_values = [
                float(observation.metrics["er_db"])
                for observation in observations
                if observation.metrics.get("er_db") is not None
            ]
            tdecq_corr_count = sum(
                self._observation_has_valid_tdecq(observation)
                for observation in observations
            )
            return (
                len(er_values) == 3
                and median(er_values) >= self.policy.coarse_er_gate_db
                and tdecq_corr_count >= 2
            )
        if stage in {CampaignState.I2C_FINE, CampaignState.I2C_RETUNE}:
            return any(observation.feasible for observation in observations)
        return False

    @staticmethod
    def _observation_tdecq(observation: Observation) -> float:
        value = observation.metrics.get("tdecq_db")
        if value is None:
            raise OrchestratorError("observation has no valid TDECQ value")
        return float(value)

    def _observation_is_partial_convergence(
        self,
        observation: Observation,
    ) -> bool:
        measurement = self._measurement_by_observation[observation.observation_id]
        return (
            self._measurement_has_valid_tdecq(measurement)
            and measurement.er_db is not None
            and self.policy.coarse_er_gate_db
            <= measurement.er_db
            < self.policy.constrained_er_min_db
        )

    def _confirmed_fine_observations(
        self,
        observations: tuple[Observation, ...],
        *,
        reference_tdecq: float,
    ) -> bool:
        if len(observations) != self.policy.best_point_confirmation_repeats:
            return False
        if not all(observation.feasible for observation in observations):
            return False
        confirmed_tdecq = median(
            self._observation_tdecq(observation) for observation in observations
        )
        return self._is_meaningful_improvement(reference_tdecq - confirmed_tdecq)

    def _is_meaningful_improvement(self, improvement: float) -> bool:
        threshold = max(
            self.policy.meaningful_tdecq_improvement_db,
            3.0 * self.policy.baseline_tdecq_sample_std_db,
        )
        return improvement > threshold and not isclose(
            improvement,
            threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    def _apply_i2c_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
        *,
        previous_i2c: Mapping[str, int],
        candidate_index: int,
    ) -> AppliedSettings:
        try:
            applied = dict(self.i2c.apply(candidate.requested_i2c))
            applied_candidate = candidate.model_copy(
                update={"requested_i2c": applied}
            )
            elapsed_s = self._authorization_usage(authorization).elapsed_s
            validate_candidate(
                authorization,
                applied_candidate,
                previous_readback=previous_i2c,
                now=self.clock.now(),
                current_campaign_id=self.campaign_id,
                current_baseline_hash=self.baseline_hash,
                candidate_index=candidate_index,
                elapsed_s=elapsed_s,
            )
            readback = dict(self.i2c.readback())
        except Exception as exc:
            if isinstance(exc, (RecoveryRequiredError, HardwareSafetyError)):
                raise
            self._hardware_failure(
                authorization, f"I2C candidate apply/readback failed: {exc}"
            )
            raise AssertionError("unreachable")
        if any(readback.get(name) != value for name, value in applied.items()):
            self._hardware_failure(authorization, "I2C candidate readback mismatch")
        return AppliedSettings(
            requested_i2c=candidate.requested_i2c,
            quantized_i2c=applied,
            applied_i2c=applied,
            readback_i2c=readback,
        )

    def _apply_fir_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
        *,
        previous_i2c: Mapping[str, int],
    ) -> AppliedSettings:
        requested = candidate.requested_fir
        if requested is None:
            raise OrchestratorError("FIR_APPLY candidate is missing requested_fir")
        try:
            applied = self.bert.apply(self.baseline.pattern, requested)
            readback = dict(self.bert.readback())
        except Exception as exc:
            self._hardware_failure(
                authorization, f"FIR candidate apply/readback failed: {exc}"
            )
            raise AssertionError("unreachable")
        if not self._bert_matches(readback, self.baseline.pattern, requested):
            self._hardware_failure(authorization, "FIR candidate readback mismatch")
        return AppliedSettings(
            readback_i2c=previous_i2c,
            requested_fir=requested,
            applied_fir=applied,
            readback_fir=self._fir_from_readback(readback),
        )

    def _run_repeated_measurements(
        self,
        *,
        stage: CampaignState,
        repeat_count: int,
        candidate: Candidate | None,
    ) -> RepeatedMeasurementReport:
        measurements: list[MeasurementResult] = []
        observations: list[Observation] = []
        settings = self._read_current_settings()
        for _ in range(repeat_count):
            measurement = self.dca.measure_single()
            measurements.append(measurement)
            if not self._measurement_has_valid_er(measurement):
                self._event(
                    "OBSERVATION_INVALID",
                    {
                        "candidate_id": None if candidate is None else candidate.candidate_id,
                        "acquisition_valid": measurement.acquisition_valid,
                        "er_status": measurement.er_status,
                        "tdecq_status": measurement.tdecq_status,
                    },
                )
                continue
            observations.append(
                self._persist_measurement(
                    candidate=candidate,
                    stage=stage,
                    settings=settings,
                    measurement=measurement,
                    authorization=None,
                )
            )
        return self._measurement_report(measurements, observations)

    def _persist_measurement(
        self,
        *,
        candidate: Candidate | None,
        stage: CampaignState,
        settings: AppliedSettings,
        measurement: MeasurementResult,
        authorization: StageAuthorization | None,
    ) -> Observation:
        observation_sequence = self.store.allocate_observation_sequence(
            self.campaign_id
        )
        candidate_part = "baseline" if candidate is None else candidate.candidate_id
        observation_id = (
            f"{stage.value.lower()}-{candidate_part}-{observation_sequence}"
        )
        continuous_evidence = measurement.raw_responses.get(
            "continuous_eye_evidence",
            {},
        )
        if not isinstance(continuous_evidence, Mapping):
            raise OrchestratorError("continuous eye evidence must be a mapping")
        observation = Observation(
            observation_id=observation_id,
            campaign_id=self.campaign_id,
            candidate_id=None if candidate is None else candidate.candidate_id,
            stage=stage,
            settings=settings,
            metrics={"er_db": measurement.er_db, "tdecq_db": measurement.tdecq_db},
            valid=(
                self._measurement_has_valid_tdecq(measurement)
                if self.continuous_i2c_only
                else True
            ),
            feasible=self._measurement_is_feasible(stage, measurement),
            elapsed_s=measurement.elapsed_s,
            evidence=dict(continuous_evidence),
            measured_at=self.clock.now(),
        )
        payload = observation.model_dump(mode="json")
        payload["metric_status"] = {
            "er_status": measurement.er_status,
            "tdecq_status": measurement.tdecq_status,
            "tdecq_invalid_reason": measurement.tdecq_invalid_reason,
        }
        payload["raw_responses"] = dict(measurement.raw_responses)
        try:
            published = self.repository.publish_observation(
                observation_id,
                payload,
                measurement.artifacts,
            )
        except RepositoryError:
            self._event(
                "OBSERVATION_PERSISTENCE_FAILED",
                {
                    "observation_id": observation_id,
                    "candidate_id": observation.candidate_id,
                },
            )
            if authorization is not None:
                self._rollback_to_terminal(
                    authorization,
                    final_event="PERSISTENCE_FAILURE_ROLLED_BACK",
                )
            raise
        artifact_paths = tuple(
            record.relative_path for record in getattr(published, "artifacts", ())
        )
        observation = observation.model_copy(update={"artifacts": artifact_paths})
        self._measurement_by_observation[observation_id] = measurement
        self._event(
            "OBSERVATION_PERSISTED",
            {
                "observation_id": observation_id,
                "candidate_id": observation.candidate_id,
                "artifacts": list(artifact_paths),
            },
        )
        return observation

    def _optimizer_observation(
        self, observation: Observation
    ) -> TuningObservation:
        measurement = self._measurement_by_observation[observation.observation_id]
        optimizer_names = {
            spec.name for spec in self.parameter_specs if spec.enabled
        }
        requested = {
            name: value
            for name, value in observation.settings.requested_i2c.items()
            if not optimizer_names or name in optimizer_names
        }
        readback = {
            name: value
            for name, value in observation.settings.readback_i2c.items()
            if not optimizer_names or name in optimizer_names
        }
        return TuningObservation(
            observation_id=observation.observation_id,
            candidate_id=observation.candidate_id or "baseline",
            context=self.optimization_context,
            settings_requested=requested,
            settings_readback=readback,
            tdecq_db=measurement.tdecq_db,
            tdecq_status=measurement.tdecq_status,
            tdecq_invalid_reason=measurement.tdecq_invalid_reason,
            er_db=measurement.er_db,
            er_valid=self._measurement_has_valid_er(measurement),
            er_status=measurement.er_status,
            artifact_refs=observation.artifacts,
        )

    def _observation_has_valid_tdecq(self, observation: Observation) -> bool:
        measurement = self._measurement_by_observation[observation.observation_id]
        return self._measurement_has_valid_tdecq(measurement)

    def _measurement_is_feasible(
        self,
        stage: CampaignState,
        measurement: MeasurementResult,
    ) -> bool:
        if self.tdecq_only:
            return self._measurement_has_valid_tdecq(measurement)
        if not self._measurement_has_valid_er(measurement):
            return False
        minimum_er = (
            self.policy.coarse_er_gate_db
            if stage == CampaignState.I2C_COARSE
            else self.policy.constrained_er_min_db
        )
        if measurement.er_db is None or measurement.er_db < minimum_er:
            return False
        if stage in {
            CampaignState.I2C_COARSE,
            CampaignState.I2C_FINE,
            CampaignState.FIR_APPLY,
            CampaignState.I2C_RETUNE,
        }:
            return self._measurement_has_valid_tdecq(measurement)
        return True

    def _hardware_failure(
        self, authorization: StageAuthorization, message: str
    ) -> None:
        state = self._rollback_to_terminal(
            authorization,
            final_event="HARDWARE_SAFETY_FAILED",
        )
        if state == CampaignState.RECOVERY_REQUIRED:
            raise RecoveryRequiredError(
                f"rollback readback mismatch after {message}"
            )
        raise HardwareSafetyError(message)

    def _failed_restore_to_recovery(
        self,
        *,
        event_type: str,
        message: str,
    ) -> None:
        self._transition(CampaignState.ABORTING, event_type=event_type)
        self._transition(
            CampaignState.ROLLING_BACK,
            event_type="STATE_TRANSITION",
        )
        self._event("ROLLBACK_MISMATCH")
        self._transition(
            CampaignState.RECOVERY_REQUIRED,
            event_type="STATE_TRANSITION",
        )
        raise RecoveryRequiredError(message)

    def _rollback_to_terminal(
        self,
        authorization: StageAuthorization,
        *,
        final_event: str,
        resume_state: CampaignState | None = None,
        continuous_completion: bool = False,
    ) -> CampaignState:
        current = self._current_state()
        if current != CampaignState.ABORTING:
            self._transition(
                CampaignState.ABORTING,
                event_type="HARDWARE_SAFETY_ABORTING",
                resume_state=resume_state,
            )
        self._transition(
            CampaignState.ROLLING_BACK,
            event_type="STATE_TRANSITION",
        )
        self._event("ROLLBACK_STARTED")
        restored = self._restore_snapshot_once(authorization.rollback_snapshot)
        if not restored:
            self._event("ROLLBACK_MISMATCH")
            self._transition(
                CampaignState.RECOVERY_REQUIRED,
                event_type="STATE_TRANSITION",
            )
            return CampaignState.RECOVERY_REQUIRED
        self._event("ROLLBACK_VERIFIED")
        self._transition(CampaignState.FAILED, event_type=final_event)
        return CampaignState.FAILED

    def _restore_snapshot_once(self, snapshot: AppliedSettings) -> bool:
        try:
            if snapshot.readback_i2c:
                self.i2c.apply(snapshot.readback_i2c)
            if snapshot.readback_fir is not None:
                self.bert.apply(self.baseline.pattern, snapshot.readback_fir)
            i2c_readback = dict(self.i2c.readback())
            bert_readback = (
                None
                if snapshot.readback_fir is None
                else dict(self.bert.readback())
            )
        except Exception:
            return False
        i2c_matches = all(
            i2c_readback.get(name) == value
            for name, value in snapshot.readback_i2c.items()
        )
        fir_matches = (
            snapshot.readback_fir is None
            or (
                bert_readback is not None
                and self._bert_matches(
                    bert_readback,
                    self.baseline.pattern,
                    snapshot.readback_fir,
                )
            )
        )
        return i2c_matches and fir_matches

    def _read_current_settings(self) -> AppliedSettings:
        i2c_readback = dict(self.i2c.readback())
        if self.continuous_i2c_only:
            return AppliedSettings(readback_i2c=i2c_readback)
        bert_readback = dict(self.bert.readback())
        return AppliedSettings(
            readback_i2c=i2c_readback,
            readback_fir=self._fir_from_readback(bert_readback),
        )

    @contextmanager
    def _execution_scope(self) -> Iterator[None]:
        local_acquired = self._execution_lock.acquire(blocking=False)
        if not local_acquired:
            raise OrchestratorError(
                f"campaign {self.campaign_id} is already executing in this process"
            )
        outermost = self._execution_depth == 0
        try:
            if outermost:
                self.store.acquire_execution_lease(
                    self.campaign_id,
                    self._execution_owner_id,
                    acquired_at=self.clock.now(),
                )
            self._execution_depth += 1
            try:
                yield
            finally:
                self._execution_depth -= 1
                if outermost:
                    self.store.release_execution_lease(
                        self.campaign_id,
                        self._execution_owner_id,
                    )
        finally:
            self._execution_lock.release()

    def _activate_stage_authorization(
        self,
        authorization: StageAuthorization,
        stage: CampaignState,
    ) -> StageAuthorization:
        validated = self._validate_stage_authorization(authorization, stage)
        try:
            self.store.activate_authorization(
                self.campaign_id,
                validated,
                replace_existing=True,
                activated_at=self.clock.now(),
            )
        except ConcurrencyError:
            raise
        except StoreError as exc:
            raise OrchestratorError(str(exc)) from exc
        return validated

    def _activate_nested_authorization(
        self,
        outer_authorization: StageAuthorization,
        authorization: StageAuthorization,
    ) -> StageAuthorization:
        validated = self._validate_stage_authorization(
            authorization,
            CampaignState.I2C_RETUNE,
        )
        try:
            self.store.activate_nested_authorization(
                self.campaign_id,
                outer_authorization.authorization_id,
                validated,
                activated_at=self.clock.now(),
            )
        except ConcurrencyError:
            raise
        except StoreError as exc:
            raise OrchestratorError(str(exc)) from exc
        return validated

    def _authorization_usage(
        self,
        authorization: StageAuthorization,
    ) -> AuthorizationUsage:
        try:
            return self.store.authorization_usage(
                self.campaign_id,
                authorization.authorization_id,
                now=self.clock.now(),
            )
        except ConcurrencyError:
            raise
        except StoreError as exc:
            raise OrchestratorError(str(exc)) from exc

    def _reserve_authorized_operation(
        self,
        authorization: StageAuthorization,
    ) -> int:
        try:
            return self.store.reserve_candidate(
                self.campaign_id,
                authorization.authorization_id,
                self._execution_owner_id,
                now=self.clock.now(),
            )
        except ConcurrencyError:
            raise
        except StoreError as exc:
            raise OrchestratorError(str(exc)) from exc

    def _reserve_authorized_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
        *,
        outer_authorization: StageAuthorization | None = None,
    ) -> tuple[int, float]:
        if outer_authorization is None:
            validated = self._activate_stage_authorization(
                authorization,
                candidate.stage,
            )
        else:
            validated = self._activate_nested_authorization(
                outer_authorization,
                authorization,
            )
        usage = self._authorization_usage(validated)
        validate_candidate_static(
            validated,
            candidate,
            now=self.clock.now(),
            current_campaign_id=self.campaign_id,
            current_baseline_hash=self.baseline_hash,
            candidate_index=usage.candidate_count,
            elapsed_s=usage.elapsed_s,
        )
        ordinal = self._reserve_authorized_operation(validated)
        persisted = self._authorization_usage(validated)
        return ordinal, persisted.elapsed_s

    def _validate_stage_authorization(
        self,
        authorization: StageAuthorization,
        stage: CampaignState,
    ) -> StageAuthorization:
        validated = validate_authorization(
            authorization,
            now=self.clock.now(),
            current_campaign_id=self.campaign_id,
            current_baseline_hash=self.baseline_hash,
        )
        self._require_authorization_time_policy_scope(validated)
        if validated.stage != stage:
            raise OrchestratorError(
                f"authorization stage {validated.stage.value} does not match {stage.value}"
            )
        return validated

    def _require_authorization_time_policy_scope(
        self,
        authorization: StageAuthorization,
    ) -> None:
        if (
            authorization.time_policy
            is AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED
            and self.continuous_i2c_only is not True
        ):
            raise OrchestratorError(
                "unbounded authorization requires an I2C-only continuous orchestrator"
            )

    def _transition(
        self,
        target: CampaignState,
        *,
        event_type: str,
        payload: Any = None,
        authorization: StageAuthorization | None = None,
        resume_state: CampaignState | None = None,
        continuous_completion: bool = False,
    ) -> CampaignState:
        current = self._current_state()
        machine = CampaignStateMachine(current, resume_state=resume_state)
        context: dict[str, object] = {}
        if authorization is not None:
            context = {
                "authorization": authorization,
                "now": self.clock.now(),
                "current_campaign_id": self.campaign_id,
                "current_baseline_hash": self.baseline_hash,
            }
        if continuous_completion:
            if not self.continuous_i2c_only:
                raise OrchestratorError(
                    "continuous completion requires an I2C-only orchestrator"
                )
        transitioned = machine.transition(current, target, **context)
        if authorization is not None:
            self._require_authorization_time_policy_scope(authorization)
        if authorization is None:
            self.store.save_transition(
                self.campaign_id,
                current,
                transitioned,
                event_type=event_type,
                payload={} if payload is None else payload,
                created_at=self.clock.now(),
            )
        else:
            if payload not in (None, {}):
                raise OrchestratorError(
                    "authorized transition payload is owned by the authorization audit"
                )
            self.store.save_authorized_transition(
                self.campaign_id,
                current,
                transitioned,
                authorization,
                event_type=event_type,
                activated_at=self.clock.now(),
            )
        return transitioned

    def _event(self, event_type: str, payload: Any = None) -> None:
        state = self._current_state()
        self.store.append_event(
            self.campaign_id,
            event_type,
            payload={} if payload is None else payload,
            created_at=self.clock.now(),
            expected_state=state,
        )

    def _preflight_failure(self, message: str) -> None:
        self._transition(
            CampaignState.FAILED,
            event_type="PREFLIGHT_FAILED",
            payload={"reason": message},
        )
        raise PreflightError(message)

    def _ensure_active_stage(self, stage: CampaignState) -> None:
        self._ensure_not_paused()
        self._require_state(stage)

    def _ensure_not_paused(self) -> None:
        if self._current_state() == CampaignState.PAUSED:
            raise CampaignPausedError("campaign is PAUSED")

    def _require_state(self, expected: CampaignState) -> None:
        current = self._current_state()
        if current != expected:
            raise OrchestratorError(
                f"operation requires {expected.value}, current state is {current.value}"
            )

    def _current_state(self) -> CampaignState:
        state = self.store.current_state(self.campaign_id)
        if state is None:
            raise OrchestratorError(f"unknown campaign: {self.campaign_id}")
        return state

    @staticmethod
    def _measurement_has_valid_er(measurement: MeasurementResult) -> bool:
        return (
            measurement.acquisition_valid
            and measurement.er_status.strip().upper() == "CORR"
            and measurement.er_db is not None
        )

    @staticmethod
    def _measurement_has_valid_tdecq(measurement: MeasurementResult) -> bool:
        return (
            measurement.acquisition_valid
            and measurement.tdecq_status.strip().upper() == "CORR"
            and measurement.tdecq_db is not None
        )

    @staticmethod
    def _bert_readback_is_well_formed(readback: Mapping[str, object]) -> bool:
        return (
            isinstance(readback.get("pattern"), str)
            and isinstance(readback.get("fir"), FirSettings)
        )

    @staticmethod
    def _fir_from_readback(readback: Mapping[str, object]) -> FirSettings:
        fir = readback.get("fir")
        if not isinstance(fir, FirSettings):
            raise HardwareSafetyError("BERT FIR readback is missing or malformed")
        return fir

    @staticmethod
    def _bert_matches(
        readback: Mapping[str, object], pattern: str, fir: FirSettings
    ) -> bool:
        return readback.get("pattern") == pattern and readback.get("fir") == fir

    @staticmethod
    def _measurement_report(
        measurements: list[MeasurementResult],
        observations: list[Observation],
    ) -> RepeatedMeasurementReport:
        acquisition_valid = [item for item in measurements if item.acquisition_valid]
        er_values = [
            float(item.er_db)
            for item in measurements
            if CampaignOrchestrator._measurement_has_valid_er(item)
            and item.er_db is not None
        ]
        tdecq_values = [
            float(item.tdecq_db)
            for item in measurements
            if item.acquisition_valid
            and item.tdecq_status.strip().upper() == "CORR"
            and item.tdecq_db is not None
        ]
        return RepeatedMeasurementReport(
            measurement_count=len(measurements),
            acquisition_valid_count=len(acquisition_valid),
            er_valid_count=len(er_values),
            tdecq_valid_count=len(tdecq_values),
            valid=False,
            er_median_db=None if not er_values else median(er_values),
            er_sample_std_db=None if len(er_values) < 2 else stdev(er_values),
            tdecq_median_db=(None if not tdecq_values else median(tdecq_values)),
            tdecq_sample_std_db=(
                None if len(tdecq_values) < 2 else stdev(tdecq_values)
            ),
            raw_responses=tuple(item.raw_responses for item in measurements),
            observations=tuple(observations),
        )

    @staticmethod
    def _with_report_validity(
        report: RepeatedMeasurementReport, valid: bool
    ) -> RepeatedMeasurementReport:
        return RepeatedMeasurementReport(
            measurement_count=report.measurement_count,
            acquisition_valid_count=report.acquisition_valid_count,
            er_valid_count=report.er_valid_count,
            tdecq_valid_count=report.tdecq_valid_count,
            valid=valid,
            er_median_db=report.er_median_db,
            er_sample_std_db=report.er_sample_std_db,
            tdecq_median_db=report.tdecq_median_db,
            tdecq_sample_std_db=report.tdecq_sample_std_db,
            raw_responses=report.raw_responses,
            observations=report.observations,
        )

    @staticmethod
    def _report_payload(report: RepeatedMeasurementReport) -> dict[str, object]:
        return {
            "measurement_count": report.measurement_count,
            "acquisition_valid_count": report.acquisition_valid_count,
            "er_valid_count": report.er_valid_count,
            "tdecq_valid_count": report.tdecq_valid_count,
            "valid": report.valid,
            "er_median_db": report.er_median_db,
            "er_sample_std_db": report.er_sample_std_db,
            "tdecq_median_db": report.tdecq_median_db,
            "tdecq_sample_std_db": report.tdecq_sample_std_db,
            "raw_responses": [dict(item) for item in report.raw_responses],
        }

    def _stage_summary(
        self,
        stage: CampaignState,
        *,
        convergence_reason: str,
        candidates_used: int,
        observations: list[Observation],
        started: float,
    ) -> StageSummary:
        valid_tdecq = [
            observation
            for observation in observations
            if observation.metrics.get("tdecq_db") is not None
        ]
        best = (
            None
            if not valid_tdecq
            else min(
                valid_tdecq,
                key=lambda item: float(item.metrics["tdecq_db"]),
            )
        )
        return StageSummary(
            stage=stage,
            convergence_reason=convergence_reason,
            best_observation=best,
            candidates_used=candidates_used,
            elapsed_s=max(0.0, self.clock.monotonic() - started),
        )
