"""Deep continuous optimization controller with fail-closed evidence rules."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from vcsel_ai_tuning.v2.contracts import EyeSample, MeasurementStatus

from .artifacts import fingerprints_compatible
from .execution import ContinuousExecution
from .models import (
    CampaignResult,
    ContinuousPreflightRequest,
    ContinuousRequest,
    ContinuousState,
    ExecutionMode,
    ExecutionOutcome,
    HumanDecision,
    ProviderActionKind,
    ReasonCode,
    RoundRecord,
)
from .policy import (
    I2cParameterPolicy,
    ProposalValidationError,
    materialize_candidate,
)
from .provider import ProviderContext
from .store import ContinuousStoreError


class ContinuousControllerError(RuntimeError):
    pass


class ContinuousOptimizationController:
    """Own continuous state while delegating all execution through one narrow port."""

    def __init__(
        self,
        *,
        provider: Any,
        execution: ContinuousExecution,
        store: Any,
        clock: Any,
        bench_lease: Any | None = None,
    ) -> None:
        if not callable(getattr(provider, "decide", None)):
            raise TypeError("provider must implement decide")
        for name in (
            "execute_candidate",
            "measure_current_state",
            "restore_b0",
            "recover_b0_after_failure",
        ):
            if not callable(getattr(execution, name, None)):
                raise TypeError(f"execution must implement {name}")
        self._provider = provider
        self._execution = execution
        self._store = store
        self._clock = clock
        self._bench_lease = bench_lease
        self._statuses: dict[str, CampaignResult] = {}
        self._stop_requests: set[str] = set()
        self._resume_states: dict[
            str,
            tuple[
                EyeSample,
                EyeSample | None,
                list[RoundRecord],
                dict[str, object],
                datetime,
            ],
        ] = {}
        self._awaiting: dict[
            str,
            tuple[
                ContinuousRequest,
                EyeSample,
                EyeSample | None,
                list[RoundRecord],
                dict[str, object],
                datetime,
                datetime,
                str | None,
            ],
        ] = {}

    def start(self, request: ContinuousRequest) -> CampaignResult:
        if not isinstance(request, ContinuousRequest):
            raise TypeError("request must be a ContinuousRequest")
        if self._bench_lease is not None and callable(
            getattr(self._bench_lease, "load", None)
        ):
            try:
                stale_lease = self._bench_lease.load()
            except Exception:
                stale_lease = object()
            if stale_lease is not None and request.campaign_id not in self._awaiting:
                started_at = self._clock.now()
                result = self._terminal_result(
                    request,
                    [],
                    {},
                    started_at,
                    ContinuousState.RECOVERY_REQUIRED,
                    "LEASE_CONTINUITY_LOST",
                    acceptance_passed=False,
                    b0_restored=False,
                    evidence_complete=False,
                    target_reached=False,
                )
                return result
        try:
            resumed = self._resume_states.get(request.campaign_id)
            self._store.initialize(
                request.campaign_id,
                {
                    "campaign_id": request.campaign_id,
                    "execution_mode": request.execution_mode.value,
                    "max_rounds": request.max_rounds,
                    "target_tdecq_db": (
                        request.stopping_policy.target_tdecq_db
                    ),
                    "policy": dict(request.policy),
                    "provider": dict(request.provider_config),
                },
            )
            if resumed is None:
                started_at = self._clock.now()
                current = request.initial_eye
                best_reference = None
                rounds: list[RoundRecord] = []
                identities: dict[str, object] = {}
            else:
                self._resume_states.pop(request.campaign_id)
                current, best_reference, rounds, identities, started_at = resumed
            running = CampaignResult(
                campaign_id=request.campaign_id,
                state=ContinuousState.RUNNING,
                execution_mode=request.execution_mode,
                completed_rounds=len(rounds),
                target_tdecq_db=request.stopping_policy.target_tdecq_db,
                target_reached=self._target_reached(current, request),
                acceptance_passed=False,
                closed_loop_proven=False,
                started_at=started_at,
                adapter_identities=identities,
                evidence_complete=all(item.evidence_persisted for item in rounds),
                round_records=tuple(rounds),
            )
            self._set_status(running)
            policy_payload = dict(request.policy)
            rules = policy_payload.get("rules")
            if isinstance(rules, list):
                policy_payload["rules"] = tuple(rules)
            policy = I2cParameterPolicy.model_validate(policy_payload)
            evidence_complete = all(item.evidence_persisted for item in rounds)
        except ContinuousStoreError:
            raise
        except Exception as startup_error:
            try:
                restored = self._execution.recover_b0_after_failure(
                    dict(request.b0_i2c)
                )
                if dict(restored) != dict(request.b0_i2c):
                    raise ContinuousControllerError("B0 readback mismatch")
            except Exception as recovery_error:
                failure = ContinuousControllerError(
                    "RECOVERY_REQUIRED: continuous startup failed and B0 recovery failed"
                )
                failure.add_note(
                    "startup error: "
                    f"{type(startup_error).__name__}: {startup_error}"
                )
                raise failure from recovery_error
            raise

        if self._target_reached(current, request):
            return self._accept_and_finish(
                request,
                current,
                rounds,
                identities,
                started_at,
                stop_reason="TARGET_ACCEPTED",
            )

        try:
            for round_index in range(len(rounds) + 1, request.max_rounds + 1):
                if request.campaign_id in self._stop_requests:
                    return self._accept_and_finish(
                        request,
                        current,
                        rounds,
                        identities,
                        started_at,
                        stop_reason="USER_STOP_ACCEPTED",
                    )
                self._validate_current_eye(current, request.campaign_id)
                context = ProviderContext(
                    sample=current,
                    verified_readback=self._integer_readback(current),
                    policy=policy,
                    hardware_round=round_index,
                    max_rounds=request.max_rounds,
                    target_tdecq_db=request.stopping_policy.target_tdecq_db,
                    recent_valid_summaries=tuple(
                        self._round_summary(record)
                        for record in rounds
                        if record.measurement_status is MeasurementStatus.VALID
                    )[-5:],
                    best_reference=best_reference,
                    allowed_output_schema={"version": "provider-action-v1"},
                )
                action = None
                candidate = None
                for proposal_attempt in (1, 2):
                    try:
                        action = self._provider.decide(context)
                    except Exception as error:
                        return self._failed_result(
                            request,
                            rounds,
                            identities,
                            started_at,
                            "PROVIDER_FAILURE",
                            provider_error_type=type(error).__name__,
                            provider_error_message=str(error),
                        )
                    if action.action is ProviderActionKind.HOLD:
                        terminal_assessment = next(
                            (
                                reason
                                for reason in (
                                    ReasonCode.INVALID_OBSERVATION,
                                    ReasonCode.OOD,
                                )
                                if reason in action.reason_codes
                            ),
                            None,
                        )
                        if terminal_assessment is not None:
                            return self._failed_result(
                                request,
                                rounds,
                                identities,
                                started_at,
                                terminal_assessment.value,
                            )
                        if action.confidence < 0.5:
                            return self._failed_result(
                                request,
                                rounds,
                                identities,
                                started_at,
                                "LOW_CONFIDENCE",
                            )
                        if proposal_attempt == 2:
                            return self._failed_result(
                                request,
                                rounds,
                                identities,
                                started_at,
                                "HOLD_RETRY_EXHAUSTED",
                            )
                        context = context.model_copy(
                            update={"correction_error_code": "HOLD_RETRY"}
                        )
                        continue
                    if action.action is ProviderActionKind.PROPOSE:
                        try:
                            candidate = materialize_candidate(
                                campaign_id=request.campaign_id,
                                round_index=round_index,
                                attempt=proposal_attempt,
                                requested=action.requested_params or {},
                                current_readback=context.verified_readback,
                                authorization=request.authorization,
                                policy=policy,
                                provider_version=self._provider.provider_version,
                                reason=action.summary,
                            )
                        except ProposalValidationError as error:
                            if proposal_attempt == 2:
                                return self._failed_result(
                                    request,
                                    rounds,
                                    identities,
                                    started_at,
                                    "INVALID_PROPOSAL_EXHAUSTED",
                                )
                            context = context.model_copy(
                                update={"correction_error_code": error.code.value}
                            )
                            continue
                    break
                if action is None:
                    raise ContinuousControllerError("provider returned no action")
                if action.action is ProviderActionKind.STOP:
                    return self._accept_and_finish(
                        request,
                        current,
                        rounds,
                        identities,
                        started_at,
                        stop_reason="PROVIDER_STOP_ACCEPTED",
                    )
                if action.action is ProviderActionKind.REQUEST_HUMAN:
                    return self._await_human(
                        request,
                        current,
                        best_reference,
                        rounds,
                        started_at,
                        identities,
                    )
                if candidate is None:
                    raise ContinuousControllerError("PROPOSE did not materialize a candidate")
                round_started = self._clock.now()
                monotonic_started = self._clock.monotonic()
                outcome = self._strict_outcome(
                    self._execution.execute_candidate(
                        request.authorization,
                        candidate,
                    )
                )
                if outcome.sample.measurement_id == current.measurement_id:
                    raise ContinuousControllerError("execution reused observation ID")
                if outcome.requested_params != candidate.requested_i2c:
                    raise ContinuousControllerError("execution request differs from candidate")
                self._validate_current_eye(outcome.sample, request.campaign_id)
                round_finished = self._clock.now()
                total_duration = self._clock.monotonic() - monotonic_started
                record = RoundRecord(
                    campaign_id=request.campaign_id,
                    round_index=round_index,
                    proposal_attempt=1,
                    before_observation_id=current.measurement_id,
                    after_observation_id=outcome.sample.measurement_id,
                    requested_params=outcome.requested_params,
                    applied_params=outcome.applied_params,
                    readback_params=outcome.readback_params,
                    before_metrics=self._metrics(current),
                    after_metrics=self._metrics(outcome.sample),
                    measurement_status=outcome.sample.measurement_status,
                    started_at=round_started,
                    finished_at=round_finished,
                    evidence_persisted=outcome.evidence_complete,
                    stage_timings_s={"total": total_duration},
                )
                if not outcome.evidence_complete:
                    raise ContinuousControllerError("round evidence is incomplete")
                self._store.save_round(
                    request.campaign_id,
                    round_index,
                    record.model_dump(mode="json"),
                )
                rounds.append(record)
                identities.update(dict(outcome.adapter_identities))
                evidence_complete = evidence_complete and outcome.evidence_complete
                best_reference = (
                    current
                    if best_reference is None
                    else self._best_reference(best_reference, current)
                )
                current = outcome.sample
                self._set_status(
                    running.model_copy(
                        update={
                            "completed_rounds": len(rounds),
                            "round_records": tuple(rounds),
                            "target_reached": self._target_reached(current, request),
                        }
                    )
                )
                if self._target_reached(current, request):
                    return self._accept_and_finish(
                        request,
                        current,
                        rounds,
                        identities,
                        started_at,
                        stop_reason="TARGET_ACCEPTED",
                    )
        except Exception:
            return self._failed_result(
                request,
                rounds,
                identities,
                started_at,
                "ROUND_FAILURE",
            )
        return self._complete(
            request,
            current,
            rounds,
            identities,
            started_at,
            stop_reason=(
                "NO_VALID_CANDIDATES"
                if rounds
                and all(
                    record.measurement_status is MeasurementStatus.INVALID
                    for record in rounds
                )
                else "MAX_ROUNDS_REACHED"
            ),
            acceptance_passed=False,
            evidence_complete=evidence_complete,
        )

    def request_stop(self, campaign_id: str) -> CampaignResult:
        status = self.get_status(campaign_id)
        if status.state is not ContinuousState.RUNNING:
            raise ContinuousControllerError("only a running campaign can request stop")
        self._stop_requests.add(campaign_id)
        return status

    def preview_initial_decision(
        self,
        request: ContinuousPreflightRequest | ContinuousRequest,
    ) -> object:
        """Contact only the provider for a preflight assessment; never execute."""

        if not isinstance(request, (ContinuousPreflightRequest, ContinuousRequest)):
            raise TypeError(
                "request must be a ContinuousPreflightRequest or ContinuousRequest"
            )
        policy_payload = dict(request.policy)
        rules = policy_payload.get("rules")
        if isinstance(rules, list):
            policy_payload["rules"] = tuple(rules)
        policy = I2cParameterPolicy.model_validate(policy_payload)
        current = request.initial_eye
        self._validate_eye(current, request.campaign_id)
        return self._provider.decide(
            ProviderContext(
                sample=current,
                verified_readback=self._integer_readback(current),
                policy=policy,
                hardware_round=1,
                max_rounds=request.max_rounds,
                target_tdecq_db=request.stopping_policy.target_tdecq_db,
                recent_valid_summaries=(),
                best_reference=None,
                allowed_output_schema={"version": "provider-action-v1"},
            )
        )

    def resolve_human(
        self,
        campaign_id: str,
        decision: HumanDecision,
    ) -> CampaignResult:
        status = self.get_status(campaign_id)
        if status.state is not ContinuousState.AWAITING_HUMAN:
            raise ContinuousControllerError("campaign is not awaiting a human")
        (
            request,
            current,
            best_reference,
            rounds,
            identities,
            started_at,
            expires_at,
            token,
        ) = self._awaiting[campaign_id]
        if self._clock.now() >= expires_at:
            return self.check_human_timeout(campaign_id)
        if decision is HumanDecision.RESUME:
            self._release_lease(campaign_id, token)
            del self._awaiting[campaign_id]
            self._resume_states[campaign_id] = (
                current,
                best_reference,
                rounds,
                identities,
                started_at,
            )
            return self.start(request)
        try:
            if decision is HumanDecision.ACCEPT:
                return self._accept_and_finish(
                    request,
                    current,
                    rounds,
                    identities,
                    started_at,
                    stop_reason="HUMAN_ACCEPTED",
                )
            return self._complete(
                request,
                current,
                rounds,
                identities,
                started_at,
                stop_reason="HUMAN_RESTORE_B0",
                acceptance_passed=False,
                evidence_complete=all(item.evidence_persisted for item in rounds),
            )
        finally:
            try:
                self._release_lease(campaign_id, token)
            finally:
                del self._awaiting[campaign_id]

    def check_human_timeout(self, campaign_id: str) -> CampaignResult:
        """Restore static B0 when a persisted human wait expires."""

        status = self.get_status(campaign_id)
        if status.state is not ContinuousState.AWAITING_HUMAN:
            return status
        (
            request,
            _current,
            _best_reference,
            rounds,
            identities,
            started_at,
            expires_at,
            token,
        ) = self._awaiting[campaign_id]
        if self._clock.now() < expires_at:
            return status
        result = None
        release_failed = False
        try:
            result = self._failed_result(
                request,
                rounds,
                identities,
                started_at,
                "HUMAN_TIMEOUT",
            )
        finally:
            try:
                self._release_lease(campaign_id, token)
            except Exception:
                release_failed = True
            finally:
                self._awaiting.pop(campaign_id, None)
        assert result is not None
        if release_failed:
            return self._terminal_result(
                request,
                rounds,
                identities,
                started_at,
                ContinuousState.RECOVERY_REQUIRED,
                "LEASE_CONTINUITY_LOST",
                acceptance_passed=False,
                b0_restored=result.b0_restored,
                evidence_complete=result.evidence_complete,
                target_reached=False,
            )
        return result

    def get_status(self, campaign_id: str) -> CampaignResult:
        try:
            return self._statuses[campaign_id]
        except KeyError as error:
            raise ContinuousControllerError(f"unknown campaign: {campaign_id}") from error

    def _accept_and_finish(
        self,
        request: ContinuousRequest,
        current: EyeSample,
        rounds: list[RoundRecord],
        identities: dict[str, object],
        started_at: datetime,
        *,
        stop_reason: str,
    ) -> CampaignResult:
        try:
            confirmation = self._strict_outcome(self._execution.measure_current_state())
            accepted = self._acceptance_passed(
                current,
                confirmation.sample,
                tdecq_tolerance=request.stopping_policy.acceptance_tdecq_delta_db,
                outer_er_tolerance=(
                    request.stopping_policy.acceptance_outer_er_delta_db
                ),
            )
            self._store.save_acceptance(
                request.campaign_id,
                {
                    "passed": accepted,
                    "initial_measurement_id": current.measurement_id,
                    "confirmation_measurement_id": confirmation.sample.measurement_id,
                },
            )
            if not accepted:
                return self._failed_result(
                    request,
                    rounds,
                    identities,
                    started_at,
                    "ACCEPTANCE_FAILED",
                )
            return self._complete(
                request,
                current,
                rounds,
                identities,
                started_at,
                stop_reason=stop_reason,
                acceptance_passed=True,
                evidence_complete=all(item.evidence_persisted for item in rounds),
            )
        except Exception:
            return self._failed_result(
                request,
                rounds,
                identities,
                started_at,
                "ACCEPTANCE_FAILURE",
            )

    def _complete(
        self,
        request: ContinuousRequest,
        current: EyeSample,
        rounds: list[RoundRecord],
        identities: dict[str, object],
        started_at: datetime,
        *,
        stop_reason: str,
        acceptance_passed: bool,
        evidence_complete: bool,
    ) -> CampaignResult:
        try:
            restored = self._execution.restore_b0(dict(request.b0_i2c))
            b0_restored = dict(restored) == dict(request.b0_i2c)
        except Exception:
            return self._terminal_result(
                request,
                rounds,
                identities,
                started_at,
                ContinuousState.RECOVERY_REQUIRED,
                "B0_RECOVERY_FAILED",
                acceptance_passed=False,
                b0_restored=False,
                evidence_complete=evidence_complete,
                target_reached=self._target_reached(current, request),
            )
        if not b0_restored:
            return self._terminal_result(
                request,
                rounds,
                identities,
                started_at,
                ContinuousState.RECOVERY_REQUIRED,
                "B0_RECOVERY_FAILED",
                acceptance_passed=False,
                b0_restored=False,
                evidence_complete=evidence_complete,
                target_reached=self._target_reached(current, request),
            )
        return self._terminal_result(
            request,
            rounds,
            identities,
            started_at,
            ContinuousState.COMPLETED,
            stop_reason,
            acceptance_passed=acceptance_passed,
            b0_restored=True,
            evidence_complete=evidence_complete,
            target_reached=self._target_reached(current, request),
        )

    def _failed_result(
        self,
        request: ContinuousRequest,
        rounds: list[RoundRecord],
        identities: dict[str, object],
        started_at: datetime,
        reason: str,
        *,
        provider_error_type: str | None = None,
        provider_error_message: str | None = None,
    ) -> CampaignResult:
        try:
            restored = self._execution.recover_b0_after_failure(
                dict(request.b0_i2c)
            )
            if dict(restored) != dict(request.b0_i2c):
                raise ContinuousControllerError("B0 readback mismatch")
        except Exception:
            state = ContinuousState.RECOVERY_REQUIRED
            reason = "B0_RECOVERY_FAILED"
            b0_restored = False
        else:
            state = ContinuousState.FAILED_SAFE
            b0_restored = True
        return self._terminal_result(
            request,
            rounds,
            identities,
            started_at,
            state,
            reason,
            acceptance_passed=False,
            b0_restored=b0_restored,
            evidence_complete=all(item.evidence_persisted for item in rounds),
            target_reached=False,
            provider_error_type=provider_error_type,
            provider_error_message=provider_error_message,
        )

    def _terminal_result(
        self,
        request: ContinuousRequest,
        rounds: list[RoundRecord],
        identities: dict[str, object],
        started_at: datetime,
        state: ContinuousState,
        stop_reason: str,
        *,
        acceptance_passed: bool,
        b0_restored: bool,
        evidence_complete: bool,
        target_reached: bool,
        provider_error_type: str | None = None,
        provider_error_message: str | None = None,
    ) -> CampaignResult:
        closed_loop_proven = (
            request.execution_mode is ExecutionMode.LIVE
            and bool(rounds)
            and bool(identities)
            and b0_restored
            and evidence_complete
        )
        result = CampaignResult(
            campaign_id=request.campaign_id,
            state=state,
            execution_mode=request.execution_mode,
            completed_rounds=len(rounds),
            target_tdecq_db=request.stopping_policy.target_tdecq_db,
            target_reached=target_reached,
            acceptance_passed=acceptance_passed,
            closed_loop_proven=closed_loop_proven,
            stop_reason=stop_reason,
            provider_error_type=provider_error_type,
            provider_error_message=provider_error_message,
            started_at=started_at,
            finished_at=self._clock.now(),
            adapter_identities=identities,
            b0_restored=b0_restored,
            evidence_complete=evidence_complete,
            round_records=tuple(rounds),
        )
        self._set_status(result)
        return result

    def _await_human(
        self,
        request: ContinuousRequest,
        current: EyeSample,
        best_reference: EyeSample | None,
        rounds: list[RoundRecord],
        started_at: datetime,
        identities: dict[str, object],
    ) -> CampaignResult:
        token: str | None = None
        issued_at = self._clock.now()
        expires_at = issued_at + timedelta(seconds=request.human_timeout_s)
        if self._bench_lease is not None:
            lease = self._bench_lease.acquire(
                campaign_id=request.campaign_id,
                owner_id=f"continuous:{request.campaign_id}",
                resource_ids=("vcsel-bench",),
                issued_at=issued_at,
                expires_at=expires_at,
            )
            token = lease.continuity_token
        result = CampaignResult(
            campaign_id=request.campaign_id,
            state=ContinuousState.AWAITING_HUMAN,
            execution_mode=request.execution_mode,
            completed_rounds=len(rounds),
            target_tdecq_db=request.stopping_policy.target_tdecq_db,
            target_reached=False,
            acceptance_passed=False,
            closed_loop_proven=False,
            started_at=started_at,
            adapter_identities=identities,
            evidence_complete=all(item.evidence_persisted for item in rounds),
            round_records=tuple(rounds),
        )
        self._awaiting[request.campaign_id] = (
            request,
            current,
            best_reference,
            rounds,
            identities,
            started_at,
            expires_at,
            token,
        )
        try:
            self._set_status(result)
        except Exception:
            try:
                self._release_lease(request.campaign_id, token)
            finally:
                self._awaiting.pop(request.campaign_id, None)
            raise
        return result

    def _release_lease(self, campaign_id: str, token: str | None) -> None:
        if self._bench_lease is not None and token is not None:
            self._bench_lease.release(campaign_id, token)

    def _set_status(self, result: CampaignResult) -> None:
        self._store.save_state(
            result.campaign_id,
            result.model_dump(mode="json"),
        )
        self._statuses[result.campaign_id] = result

    @staticmethod
    def _strict_outcome(value: object) -> ExecutionOutcome:
        if isinstance(value, ExecutionOutcome):
            return ExecutionOutcome.model_validate(
                value.model_dump(round_trip=True),
                strict=True,
            )
        raise TypeError("execution must return ExecutionOutcome")

    @staticmethod
    def _integer_readback(sample: EyeSample) -> dict[str, int]:
        result: dict[str, int] = {}
        for name, value in sample.readback_params.items():
            number = float(value)
            if not number.is_integer():
                raise ContinuousControllerError("I2C readback must contain integers")
            result[name] = int(number)
        return result

    @staticmethod
    def _validate_eye(sample: EyeSample, campaign_id: str) -> None:
        if sample.campaign_id != campaign_id:
            raise ContinuousControllerError("eye sample campaign mismatch")
        if sample.measurement_status is not MeasurementStatus.VALID:
            raise ContinuousControllerError("eye sample is not valid")
        if sample.metrics.tdecq_db is None:
            raise ContinuousControllerError("eye sample has no TDECQ")

    @staticmethod
    def _validate_current_eye(sample: EyeSample, campaign_id: str) -> None:
        if sample.campaign_id != campaign_id:
            raise ContinuousControllerError("eye sample campaign mismatch")
        if sample.measurement_status not in {
            MeasurementStatus.VALID,
            MeasurementStatus.INVALID,
        }:
            raise ContinuousControllerError("eye sample is incomplete")
        if (
            sample.measurement_status is MeasurementStatus.VALID
            and sample.metrics.tdecq_db is None
        ):
            raise ContinuousControllerError("valid eye sample has no TDECQ")

    @staticmethod
    def _metrics(sample: EyeSample) -> dict[str, float | None]:
        return {
            "tdecq_db": sample.metrics.tdecq_db,
            "outer_er_db": sample.metrics.outer_er_chan1a_db,
        }

    @staticmethod
    def _round_summary(record: RoundRecord) -> dict[str, object]:
        return {
            "round_index": record.round_index,
            "readback_params": dict(record.readback_params),
            "metrics": dict(record.after_metrics),
        }

    @staticmethod
    def _best_reference(initial: EyeSample, current: EyeSample) -> EyeSample:
        if current.measurement_status is not MeasurementStatus.VALID:
            return initial
        initial_tdecq = initial.metrics.tdecq_db
        current_tdecq = current.metrics.tdecq_db
        best = (
            current
            if current_tdecq is not None
            and (initial_tdecq is None or current_tdecq < initial_tdecq)
            else initial
        )
        return best

    @staticmethod
    def _target_reached(sample: EyeSample, request: ContinuousRequest) -> bool:
        value = sample.metrics.tdecq_db
        return (
            sample.measurement_status is MeasurementStatus.VALID
            and value is not None
            and value < request.stopping_policy.target_tdecq_db
        )

    @staticmethod
    def _acceptance_passed(
        first: EyeSample,
        second: EyeSample,
        *,
        tdecq_tolerance: float,
        outer_er_tolerance: float,
    ) -> bool:
        if first.measurement_id == second.measurement_id:
            return False
        if first.measurement_status is not MeasurementStatus.VALID:
            return False
        if second.measurement_status is not MeasurementStatus.VALID:
            return False
        if not fingerprints_compatible(
            first.dca_config_fingerprint,
            second.dca_config_fingerprint,
        ):
            return False
        if dict(first.readback_params) != dict(second.readback_params):
            return False
        first_tdecq = first.metrics.tdecq_db
        second_tdecq = second.metrics.tdecq_db
        first_er = first.metrics.outer_er_chan1a_db
        second_er = second.metrics.outer_er_chan1a_db
        if None in (first_tdecq, second_tdecq, first_er, second_er):
            return False
        return (
            abs(first_tdecq - second_tdecq) <= tdecq_tolerance + 1e-12
            and abs(first_er - second_er) <= outer_er_tolerance + 1e-12
        )
