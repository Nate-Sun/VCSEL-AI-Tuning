"""Upper-layer LIVE composition for the provider-neutral continuous controller."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from vcsel_ai_tuning.v2.continuous.models import ContinuousRequest


def _require_policy_config(config: Mapping[str, object]) -> Mapping[str, object]:
    policy = config.get("parameter_policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("continuous LIVE parameter_policy is required")
    return policy


def _build_live_policy(
    raw_policy: Mapping[str, object],
    *,
    baseline: Any,
    parameter_specs: tuple[Any, ...],
) -> Any:
    from vcsel_ai_tuning.v2.continuous.policy import build_restricted_policy

    return build_restricted_policy(
        raw_policy,
        authorization_specs=parameter_specs,
        baseline_i2c=baseline.i2c.model_dump(),
    )


def preflight_continuous_live(
    config: Mapping[str, object],
    result_dir: Path,
) -> Mapping[str, object]:
    """Run the operator-authorized DCA/I2C/Kimi preflight without DUT writes."""

    raw_policy = _require_policy_config(config)

    from vcsel_ai_tuning.api.schemas import (
        CreateCampaignRequest,
        TuningRequestPath,
    )
    from vcsel_ai_tuning.campaign.models import AuthorizationMode, load_b0
    from vcsel_ai_tuning.hardware.live_runtime_factory import (
        build_continuous_live_tuning_profile,
        create_live_api_runtime_from_path,
    )
    from vcsel_ai_tuning.runtime import DurableBenchLease
    from vcsel_ai_tuning.v2.continuous.controller import (
        ContinuousOptimizationController,
    )
    from vcsel_ai_tuning.v2.continuous.execution import (
        OrchestratorExecutionPort,
        PersistedObservationAdapter,
    )
    from vcsel_ai_tuning.v2.continuous.fake import SystemClock
    from vcsel_ai_tuning.v2.continuous.models import (
        ContinuousPreflightRequest,
        StoppingPolicy,
    )
    from vcsel_ai_tuning.v2.continuous.store import ContinuousCampaignStore
    from vcsel_ai_tuning.v2.kimi_memory import KimiCampaignMemory
    from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient
    from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider

    api_runtime = None
    try:
        runtime_path = Path(str(config["runtime_config"]))
        campaign_id = str(config["campaign_id"])
        api_runtime = create_live_api_runtime_from_path(
            runtime_path,
            continuous_i2c_only=True,
        )
        service = api_runtime.service
        runtime = service.runtime
        bench_lease = DurableBenchLease(runtime.config.storage.sqlite_path)
        bench_lease.assert_available_for_contact(campaign_id)
        # The durable continuity guard must pass before even constructing the
        # provider client; decide() below remains the network boundary.
        kimi_client = KimiApiClient(
            model=str(config.get("kimi_model", "kimi-k3")),
            response_log_path=(
                result_dir / campaign_id / "kimi-response-attempts.jsonl"
            ),
        )
        if runtime.config.hardware is None or runtime.baseline_hash is None:
            raise RuntimeError("continuous LIVE runtime metadata is incomplete")
        baseline = load_b0(runtime.config.b0_path)
        tuning_profile = build_continuous_live_tuning_profile(
            runtime.config.hardware.driver_profile,
            baseline,
        )
        policy = _build_live_policy(
            raw_policy,
            baseline=baseline,
            parameter_specs=tuning_profile.parameter_specs,
        )
        profile_hash = sha256(
            runtime.config.hardware.driver_profile.read_bytes()
        ).hexdigest()
        service.create_campaign(
            CreateCampaignRequest(
                campaign_id=campaign_id,
                baseline_id=baseline.baseline_id,
                baseline_hash=runtime.baseline_hash,
                config_hash=profile_hash,
                authorization_mode=AuthorizationMode.PER_STAGE,
                tuning_path=TuningRequestPath.PATH_TDECQ_BAYESIAN,
            )
        )

        # entry() is the first point that constructs live DCA/I2C adapters.
        entry = runtime.entry(campaign_id)
        captured = entry.orchestrator.capture_continuous_preflight_observation()
        observation_adapter = PersistedObservationAdapter(entry.repository)
        initial_eye = observation_adapter.to_eye_sample(captured.observation)
        provider = KimiProvider(
            kimi_client,
            artifact_root=entry.repository.run_dir,
            instruction=str(
                config.get(
                    "instruction",
                    "Improve TDECQ using sparse safe absolute I2C values.",
                )
            ),
            memory=KimiCampaignMemory(
                result_dir / campaign_id / "kimi-memory.jsonl",
                campaign_id=campaign_id,
            ),
        )
        execution = OrchestratorExecutionPort(
            entry.orchestrator,
            observation_adapter=observation_adapter,
            adapter_identities={"dca": captured.dca, "i2c": captured.i2c},
        )
        controller = ContinuousOptimizationController(
            provider=provider,
            execution=execution,
            store=ContinuousCampaignStore(result_dir),
            clock=SystemClock(),
        )
        request = ContinuousPreflightRequest(
            campaign_id=campaign_id,
            initial_eye=initial_eye,
            policy=policy.model_dump(mode="json"),
            max_rounds=config["max_rounds"],
            stopping_policy=StoppingPolicy(
                target_tdecq_db=config["target_tdecq_db"]
            ),
        )
        action = controller.preview_initial_decision(request)
        return {
            "operation": "PREFLIGHT",
            "execution_mode": "LIVE",
            "campaign_id": campaign_id,
            "dut_write_performed": False,
            "hardware_write_performed": False,
            "hardware_write_scope": "NO_DUT_REGISTER_OR_BERT_WRITE",
            "dca_acquisition_commands_performed": True,
            "bert_contacted": False,
            "kimi_connectivity_verified": True,
            "dca_observation_id": initial_eye.measurement_id,
            "i2c_readback": dict(initial_eye.readback_params),
            "adapter_identities": {
                "dca": dict(captured.dca),
                "i2c": dict(captured.i2c),
            },
            "provider_action": action.action.value,
            "provider_requested_params": (
                None
                if action.requested_params is None
                else dict(action.requested_params)
            ),
        }
    finally:
        if api_runtime is not None and callable(api_runtime.shutdown):
            api_runtime.shutdown()


def activate_continuous_live(
    config: Mapping[str, object],
    result_dir: Path,
) -> Any:
    """Perform the explicitly authorized provider and hardware activation."""

    raw_policy = _require_policy_config(config)

    from vcsel_ai_tuning.hardware.live_runtime_factory import (
        build_continuous_live_tuning_profile,
        create_live_api_runtime_from_path,
    )
    from vcsel_ai_tuning.runtime import DurableBenchLease
    from vcsel_ai_tuning.campaign.models import CampaignState, load_b0
    from vcsel_ai_tuning.v2.continuous.controller import (
        ContinuousOptimizationController,
    )
    from vcsel_ai_tuning.v2.continuous.execution import (
        OrchestratorExecutionPort,
        PersistedObservationAdapter,
    )
    from vcsel_ai_tuning.v2.continuous.factory import ContinuousDependencies
    from vcsel_ai_tuning.v2.continuous.fake import SystemClock
    from vcsel_ai_tuning.v2.continuous.models import (
        ContinuousState,
        ExecutionMode,
        HumanDecision,
        StoppingPolicy,
    )
    from vcsel_ai_tuning.v2.continuous.store import ContinuousCampaignStore
    from vcsel_ai_tuning.v2.kimi_memory import KimiCampaignMemory
    from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient
    from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider

    api_runtime = None
    commission_orchestrator = None
    try:
        runtime_path = Path(str(config["runtime_config"]))
        campaign_id = str(config["campaign_id"])
        max_rounds = config["max_rounds"]
        target_tdecq_db = config["target_tdecq_db"]
        api_runtime = create_live_api_runtime_from_path(
            runtime_path,
            continuous_i2c_only=True,
        )
        service = api_runtime.service
        runtime = service.runtime
        bench_lease = DurableBenchLease(runtime.config.storage.sqlite_path)
        bench_lease.assert_available_for_contact(campaign_id)
        kimi_client = KimiApiClient(
            model=str(config.get("kimi_model", "kimi-k3")),
            response_log_path=(
                result_dir / campaign_id / "kimi-response-attempts.jsonl"
            ),
        )
        if runtime.config.hardware is None:
            raise RuntimeError("continuous LIVE runtime hardware metadata is incomplete")
        baseline = load_b0(runtime.config.b0_path)
        tuning_profile = build_continuous_live_tuning_profile(
            runtime.config.hardware.driver_profile,
            baseline,
        )
        policy = _build_live_policy(
            raw_policy,
            baseline=baseline,
            parameter_specs=tuning_profile.parameter_specs,
        )
        entry = runtime.entry(campaign_id)
        observation_adapter = PersistedObservationAdapter(entry.repository)
        dca_identity = dict(entry.adapters.dca.preflight())
        i2c_identity = dict(entry.adapters.i2c.preflight())
        if dca_identity.get("identity_verified") is not True:
            raise RuntimeError("continuous DCA identity is not verified")
        if i2c_identity.get("identity_verified") is not True:
            raise RuntimeError("continuous I2C identity is not verified")
        commission_orchestrator = entry.orchestrator
        prepared = entry.orchestrator.prepare_continuous_commission(
            max_candidates=max_rounds,
        )
        initial_eye = observation_adapter.to_eye_sample(prepared.observation)
        authorization = prepared.authorization
        memory_path = result_dir / campaign_id / "kimi-memory.jsonl"
        provider = KimiProvider(
            kimi_client,
            artifact_root=entry.repository.run_dir,
            instruction=str(
                config.get(
                    "instruction",
                    "Improve TDECQ using sparse safe absolute I2C values.",
                )
            ),
            memory=KimiCampaignMemory(memory_path, campaign_id=campaign_id),
        )
        execution = OrchestratorExecutionPort(
            entry.orchestrator,
            observation_adapter=observation_adapter,
            adapter_identities={"dca": dca_identity, "i2c": i2c_identity},
        )
        controller = ContinuousOptimizationController(
            provider=provider,
            execution=execution,
            store=ContinuousCampaignStore(result_dir),
            clock=SystemClock(),
            bench_lease=bench_lease,
        )
        request = ContinuousRequest(
            campaign_id=campaign_id,
            initial_eye=initial_eye,
            authorization=authorization,
            policy=policy.model_dump(mode="json"),
            max_rounds=max_rounds,
            provider_config={
                "provider_id": provider.provider_id,
                "provider_version": provider.provider_version,
            },
            stopping_policy=StoppingPolicy(target_tdecq_db=target_tdecq_db),
            b0_path=runtime.config.b0_path,
            b0_i2c=entry.orchestrator.baseline.i2c.model_dump(),
            result_dir=result_dir,
            execution_mode=ExecutionMode.LIVE,
            human_timeout_s=float(config.get("human_timeout_s", 900.0)),
        )
        def shutdown_continuous() -> None:
            try:
                try:
                    controller_status = controller.get_status(campaign_id)
                except Exception:
                    controller_status = None
                if (
                    controller_status is not None
                    and controller_status.state is ContinuousState.AWAITING_HUMAN
                ):
                    controller.resolve_human(
                        campaign_id,
                        HumanDecision.RESTORE_B0,
                    )
                state = entry.orchestrator.store.current_state(campaign_id)
                if state in {
                    CampaignState.I2C_SCREENING,
                    CampaignState.ABORTING,
                    CampaignState.ROLLING_BACK,
                }:
                    entry.orchestrator.recover_continuous_commission_failure()
            finally:
                api_runtime.shutdown()
        return ContinuousDependencies(
            mode=ExecutionMode.LIVE,
            controller=controller,
            request=request,
            contact_permitted=True,
            shutdown=shutdown_continuous,
            preflight_metadata={
                "dca": dca_identity,
                "i2c": i2c_identity,
            },
        )
    except Exception:
        recovery_error = None
        if commission_orchestrator is not None:
            try:
                state = commission_orchestrator.store.current_state(campaign_id)
                if state in {
                    CampaignState.RESTORE_B0,
                    CampaignState.BASELINE_MEASUREMENT,
                    CampaignState.AWAIT_COARSE_APPROVAL,
                    CampaignState.I2C_SCREENING,
                    CampaignState.ABORTING,
                    CampaignState.ROLLING_BACK,
                }:
                    commission_orchestrator.recover_continuous_commission_failure()
            except Exception as error:
                recovery_error = error
        if api_runtime is not None and callable(api_runtime.shutdown):
            api_runtime.shutdown()
        if recovery_error is not None:
            raise RuntimeError(
                "continuous commission assembly failed and B0 recovery failed"
            ) from recovery_error
        raise


def _latest_authorization(runtime: Any, campaign_id: str) -> Any:
    from vcsel_ai_tuning.campaign.models import StageAuthorization

    for event in reversed(runtime.store.events(campaign_id)):
        if event.event_type != "STAGE_PLANNED":
            continue
        payload = event.payload.get("authorization")
        if isinstance(payload, Mapping):
            authorization = StageAuthorization.model_validate(payload)
            if authorization.campaign_id == campaign_id:
                return authorization
    raise RuntimeError("continuous campaign has no persisted stage authorization")


__all__ = ["activate_continuous_live", "preflight_continuous_live"]
