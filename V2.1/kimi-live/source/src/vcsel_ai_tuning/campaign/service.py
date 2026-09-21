"""Production application service over the campaign runtime and orchestrator."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from statistics import median
from threading import RLock
from typing import Any

from vcsel_ai_tuning.runtime import (
    CampaignRuntime,
    OptimizerCapability,
    OptimizerCapabilityError,
    ReadOnlyModeError,
)
from vcsel_ai_tuning.runtime_config import RuntimeMode

from .models import Campaign, CampaignState, StageAuthorization
from .store import ConcurrencyError


class ConcreteCampaignService:
    """Thin, idempotent API service; hardware remains orchestrator-owned."""

    def __init__(self, runtime: CampaignRuntime) -> None:
        if not isinstance(runtime, CampaignRuntime):
            raise TypeError("runtime must be CampaignRuntime")
        self.runtime = runtime
        self._approval_lock = RLock()
        self._approval_responses: dict[tuple[str, str], dict[str, object]] = {}

    def create_campaign(self, request: Any) -> Campaign:
        path = OptimizerCapability(request.tuning_path)
        if (
            path is OptimizerCapability.PATH_HIERARCHICAL
            and not self.runtime.capabilities.hierarchical_ready
        ):
            raise OptimizerCapabilityError(
                "PATH_HIERARCHICAL is not configured; explicitly select a "
                "diagnostic TDECQ-only path if that is the intended campaign"
            )
        expected_baseline_hash = self.runtime.baseline_hash
        if (
            expected_baseline_hash is not None
            and request.baseline_hash != expected_baseline_hash
        ):
            raise OptimizerCapabilityError(
                "campaign baseline hash does not match the configured B0"
            )
        existing_state = self.runtime.store.current_state(request.campaign_id)
        if existing_state is not None:
            persisted = self.runtime.store.get_campaign(request.campaign_id)
            immutable_matches = (
                persisted.baseline_id == request.baseline_id
                and persisted.baseline_hash == request.baseline_hash
                and persisted.config_hash == request.config_hash
                and persisted.authorization_mode == request.authorization_mode
            )
            if not immutable_matches:
                raise ConcurrencyError(
                    f"campaign {request.campaign_id} metadata conflict"
                )
            self._require_selected_path(persisted.campaign_id, path)
            return persisted

        now = datetime.now(timezone.utc)
        campaign = Campaign(
            campaign_id=request.campaign_id,
            baseline_id=request.baseline_id,
            baseline_hash=request.baseline_hash,
            config_hash=request.config_hash,
            authorization_mode=request.authorization_mode,
            created_at=now,
            updated_at=now,
        )
        persisted = self.runtime.store.create_campaign(
            campaign,
            tuning_path=path.value,
        )
        return persisted

    def _require_selected_path(
        self,
        campaign_id: str,
        path: OptimizerCapability,
    ) -> None:
        selected = self.runtime.store.get_campaign_tuning_path(campaign_id)
        if selected != path.value:
            raise ConcurrencyError(f"campaign {campaign_id} tuning path conflict")

    def get_campaign(self, campaign_id: str) -> Campaign:
        return self.runtime.store.get_campaign(campaign_id)

    def list_campaigns(self, *, limit: int = 50, offset: int = 0) -> list[Campaign]:
        return self.runtime.store.list_campaigns(limit=limit, offset=offset)

    def preflight(self, campaign_id: str) -> Any:
        state = self.runtime.store.current_state(campaign_id)
        entry = self.runtime.entry(campaign_id)
        if state is CampaignState.DRAFT:
            return entry.orchestrator.start_preflight()
        if state is CampaignState.PREFLIGHT:
            return entry.orchestrator.preflight()
        completed = next(
            (
                event.payload
                for event in reversed(self.runtime.store.events(campaign_id))
                if event.event_type == "PREFLIGHT_COMPLETED"
            ),
            None,
        )
        if completed is not None:
            return completed
        return {"state": None if state is None else state.value}

    def plan_stage(self, campaign_id: str, request: Any) -> StageAuthorization:
        selected_path = OptimizerCapability(
            self.runtime.store.get_campaign_tuning_path(campaign_id)
        )
        if (
            selected_path
            in {
                OptimizerCapability.PATH_EXHAUSTIVE,
                OptimizerCapability.PATH_TDECQ_BAYESIAN,
            }
            and request.stage
            not in {
                CampaignState.RESTORE_B0,
                CampaignState.I2C_SCREENING,
            }
        ):
            raise OptimizerCapabilityError(
                "TDECQ-only diagnostic paths authorize only RESTORE_B0 and "
                "I2C_SCREENING"
            )
        current_state = self.runtime.store.current_state(campaign_id)
        now = datetime.now(timezone.utc)
        for event in reversed(self.runtime.store.events(campaign_id)):
            if event.event_type != "STAGE_PLANNED":
                continue
            if event.to_state is not current_state:
                continue
            authorization = self._authorization_from_payload(event.payload)
            if authorization is None:
                continue
            if (
                authorization.campaign_id == campaign_id
                and authorization.stage is request.stage
                and authorization.max_candidates == request.max_candidates
                and authorization.max_duration_s == request.max_duration_s
                and authorization.expires_at > now
            ):
                return authorization
        entry = self.runtime.entry(campaign_id)
        return entry.orchestrator.plan_stage(
            request.stage,
            max_candidates=request.max_candidates,
            max_duration_s=request.max_duration_s,
        )

    def approve_stage(
        self,
        campaign_id: str,
        authorization: StageAuthorization,
    ) -> Mapping[str, object]:
        if self.runtime.config.mode is RuntimeMode.READ_ONLY:
            raise ReadOnlyModeError(
                "READ_ONLY mode rejects approve and every hardware write"
            )
        if (
            authorization.stage is CampaignState.FIR_APPLY
            and not self.runtime.capabilities.fir_proposer_ready
        ):
            raise OptimizerCapabilityError(
                "FIR proposer is not configured; FIR stage is unavailable"
            )
        key = (campaign_id, authorization.authorization_id)
        operation = f"APPROVE:{authorization.authorization_id}"
        with self._approval_lock:
            existing = self._approval_responses.get(key)
            if existing is not None:
                return existing
            persisted_status = self._persisted_worker_status(
                campaign_id,
                operation,
            )
            if persisted_status is not None:
                response = {
                    "authorization_id": authorization.authorization_id,
                    "campaign_id": campaign_id,
                    "status": persisted_status,
                }
                self._approval_responses[key] = response
                return response
            response: dict[str, object] = {
                "authorization_id": authorization.authorization_id,
                "campaign_id": campaign_id,
                "status": "ACCEPTED",
            }
            self._approval_responses[key] = response

        self.runtime.submit_stage(
            campaign_id,
            operation,
            lambda entry: self._execute_authorized_stage(
                entry.orchestrator,
                authorization,
            ),
        )
        return response

    def _persisted_worker_status(
        self,
        campaign_id: str,
        operation: str,
    ) -> str | None:
        for event in reversed(self.runtime.store.events(campaign_id)):
            if not isinstance(event.payload, Mapping):
                continue
            if event.payload.get("operation") != operation:
                continue
            if event.event_type == "WORKER_COMPLETED":
                return "COMPLETED"
            if event.event_type == "WORKER_FAILED":
                return "FAILED"
            if event.event_type == "WORKER_STARTED":
                state = self.runtime.store.current_state(campaign_id)
                return (
                    "RECOVERY_REQUIRED"
                    if state is CampaignState.RECOVERY_REQUIRED
                    else "IN_PROGRESS"
                )
        return None

    @staticmethod
    def _execute_authorized_stage(
        orchestrator: Any,
        authorization: StageAuthorization,
    ) -> Any:
        stage = authorization.stage
        if stage is CampaignState.RESTORE_B0:
            orchestrator.restore_b0(authorization)
            return orchestrator.run_baseline_measurement()
        orchestrator.enter_stage(authorization)
        runners = {
            CampaignState.I2C_SCREENING: orchestrator.run_screening_stage,
            CampaignState.I2C_COARSE: orchestrator.run_coarse_stage,
            CampaignState.I2C_FINE: orchestrator.run_fine_stage,
            CampaignState.I2C_RETUNE: orchestrator.run_retune_stage,
        }
        runner = runners.get(stage)
        if runner is None:
            raise OptimizerCapabilityError(
                f"stage {stage.value} has no configured production runner"
            )
        return runner(authorization)

    def pause_campaign(self, campaign_id: str) -> CampaignState:
        state = self.runtime.store.current_state(campaign_id)
        if state is CampaignState.PAUSED:
            return state
        return self.runtime.entry(campaign_id).orchestrator.pause()

    def resume_campaign(self, campaign_id: str) -> CampaignState:
        state = self.runtime.store.current_state(campaign_id)
        if state is not CampaignState.PAUSED:
            if state is None:
                return self.runtime.store.get_campaign(campaign_id).state
            return state
        return self.runtime.entry(campaign_id).orchestrator.resume()

    def abort_campaign(self, campaign_id: str) -> CampaignState:
        state = self.runtime.store.current_state(campaign_id)
        if state in {
            CampaignState.FAILED,
            CampaignState.COMPLETED,
            CampaignState.RECOVERY_REQUIRED,
        }:
            return state
        authorization = self._latest_authorization(campaign_id)
        return self.runtime.entry(campaign_id).orchestrator.abort(authorization)

    def rollback_campaign(self, campaign_id: str) -> CampaignState:
        return self.abort_campaign(campaign_id)

    def list_observations(self, campaign_id: str) -> list[Any]:
        return self.runtime.entry(campaign_id).repository.list_observations()

    def get_best_observation(self, campaign_id: str) -> Mapping[str, object] | None:
        published = self.list_observations(campaign_id)
        groups: dict[str, list[tuple[float, Any]]] = {}
        for item in published:
            payload = item.payload
            if payload.get("valid") is not True or payload.get("feasible") is not True:
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            value = metrics.get("tdecq_db")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            candidate_id = payload.get("candidate_id")
            if not isinstance(candidate_id, str):
                continue
            groups.setdefault(candidate_id, []).append((float(value), item))
        if not groups:
            return None
        candidate_id, measurements = min(
            groups.items(),
            key=lambda group: (median(value for value, _ in group[1]), group[0]),
        )
        return {
            "candidate_id": candidate_id,
            "measurement_count": len(measurements),
            "tdecq_median_db": median(value for value, _ in measurements),
            "observation_ids": [item.observation_id for _, item in measurements],
        }

    def get_report(self, campaign_id: str) -> Mapping[str, object]:
        campaign = self.get_campaign(campaign_id)
        observations = self.list_observations(campaign_id)
        return {
            "campaign": campaign.model_dump(mode="json"),
            "best_observation": self.get_best_observation(campaign_id),
            "event_count": len(self.runtime.store.events(campaign_id)),
            "observation_count": len(observations),
        }

    def _latest_authorization(self, campaign_id: str) -> StageAuthorization:
        for event in reversed(self.runtime.store.events(campaign_id)):
            if not isinstance(event.payload, Mapping):
                continue
            authorization = self._authorization_from_payload(event.payload)
            if authorization is not None:
                return authorization
        raise OptimizerCapabilityError(
            "no pre-authorized rollback snapshot is available"
        )

    @staticmethod
    def _authorization_from_payload(
        payload: Mapping[str, object],
    ) -> StageAuthorization | None:
        raw = payload.get("authorization")
        if not isinstance(raw, Mapping):
            return None
        return StageAuthorization.model_validate_json(
            json.dumps(
                raw,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
