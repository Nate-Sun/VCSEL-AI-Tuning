from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from vcsel_ai_tuning.api.schemas import (
    CreateCampaignRequest,
    StagePlanRequest,
    TuningRequestPath,
)
from vcsel_ai_tuning.campaign.authorization import (
    AuthorizationError,
    rollback_hash_for,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    CampaignState,
    Observation,
    StageAuthorization,
    load_b0,
)
from vcsel_ai_tuning.campaign.repository import CampaignRepository
from vcsel_ai_tuning.campaign.service import ConcreteCampaignService
from vcsel_ai_tuning.campaign.store import CampaignStore
from vcsel_ai_tuning.runtime import (
    AdapterBundle,
    CampaignEntry,
    CampaignRuntime,
    OptimizerCapabilityError,
    ReadOnlyModeError,
)
from vcsel_ai_tuning.runtime_config import RuntimeConfig, RuntimeMode


HASH_A = "a" * 64
HASH_B = "b" * 64


class FakeOrchestrator:
    def __init__(self, store: CampaignStore, campaign_id: str) -> None:
        self.store = store
        self.campaign_id = campaign_id
        self.calls: list[str] = []

    def start_preflight(self) -> dict[str, object]:
        self.calls.append("start_preflight")
        self.store.save_transition(
            self.campaign_id,
            CampaignState.DRAFT,
            CampaignState.PREFLIGHT,
        )
        self.store.save_transition(
            self.campaign_id,
            CampaignState.PREFLIGHT,
            CampaignState.AWAIT_BASELINE_APPROVAL,
            event_type="PREFLIGHT_COMPLETED",
            payload={"verified": True},
        )
        return {"verified": True}

    def plan_stage(
        self,
        stage: CampaignState,
        *,
        max_candidates: int,
        max_duration_s: float,
    ) -> StageAuthorization:
        self.calls.append(f"plan:{stage.value}")
        now = datetime.now(timezone.utc)
        snapshot = AppliedSettings(
            readback_i2c=load_b0().i2c.model_dump(),
            readback_fir=load_b0().fir,
        )
        authorization = with_authorization_id(
            StageAuthorization(
                authorization_id="pending",
                campaign_id=self.campaign_id,
                mode=AuthorizationMode.PER_STAGE,
                stage=stage,
                baseline_hash=HASH_A,
                rollback_hash=rollback_hash_for(snapshot),
                max_candidates=max_candidates,
                max_duration_s=max_duration_s,
                issued_at=now,
                expires_at=now + timedelta(minutes=5),
                rollback_snapshot=snapshot,
            )
        )
        state = self.store.current_state(self.campaign_id)
        assert state is not None
        self.store.append_event(
            self.campaign_id,
            "STAGE_PLANNED",
            payload={"authorization": authorization.model_dump(mode="json")},
            expected_state=state,
        )
        return authorization

    def restore_b0(self, _authorization: StageAuthorization) -> dict[str, object]:
        self.calls.append("restore_b0")
        self.store.save_transition(
            self.campaign_id,
            CampaignState.AWAIT_BASELINE_APPROVAL,
            CampaignState.RESTORE_B0,
        )
        self.store.save_transition(
            self.campaign_id,
            CampaignState.RESTORE_B0,
            CampaignState.BASELINE_MEASUREMENT,
        )
        return {"restored": True}

    def run_baseline_measurement(self) -> dict[str, object]:
        self.calls.append("run_baseline_measurement")
        self.store.save_transition(
            self.campaign_id,
            CampaignState.BASELINE_MEASUREMENT,
            CampaignState.AWAIT_COARSE_APPROVAL,
        )
        return {"valid": True}

    def pause(self) -> CampaignState:
        self.calls.append("pause")
        current = self.store.current_state(self.campaign_id)
        assert current is not None
        self.store.save_transition(
            self.campaign_id,
            current,
            CampaignState.PAUSED,
            event_type="CAMPAIGN_PAUSED",
            payload={"resume_state": current.value},
        )
        return CampaignState.PAUSED

    def resume(self) -> CampaignState:
        self.calls.append("resume")
        self.store.save_transition(
            self.campaign_id,
            CampaignState.PAUSED,
            CampaignState.AWAIT_COARSE_APPROVAL,
        )
        return CampaignState.AWAIT_COARSE_APPROVAL

    def abort(self, _authorization: StageAuthorization) -> CampaignState:
        self.calls.append("abort")
        current = self.store.current_state(self.campaign_id)
        assert current is not None
        self.store.save_transition(
            self.campaign_id,
            current,
            CampaignState.ABORTING,
        )
        self.store.save_transition(
            self.campaign_id,
            CampaignState.ABORTING,
            CampaignState.ROLLING_BACK,
        )
        self.store.save_transition(
            self.campaign_id,
            CampaignState.ROLLING_BACK,
            CampaignState.FAILED,
        )
        return CampaignState.FAILED


def _runtime_config(tmp_path: Path, mode: RuntimeMode = RuntimeMode.FAKE) -> RuntimeConfig:
    values: dict[str, object] = {
        "mode": mode,
        "storage": {
            "sqlite_path": tmp_path / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts",
        },
        "worker_count": 1,
    }
    if mode is not RuntimeMode.FAKE:
        values["hardware"] = {
            "driver_profile": tmp_path / "driver.yaml",
            "dca_visa_resource": "TCPIP0::dca.example::inst0::INSTR",
            "i2c_com_port": "COM7",
            "i2c_address": 0xA6,
            "bert_host": "192.0.2.10",
            "bert_port": 9000,
            "resource_ids": ("bench-1",),
        }
    return RuntimeConfig(**values)


def _service(tmp_path: Path, mode: RuntimeMode = RuntimeMode.FAKE):
    store = CampaignStore(tmp_path / "campaigns.sqlite3")
    orchestrators: dict[str, FakeOrchestrator] = {}

    def entry_factory(campaign_id: str) -> CampaignEntry:
        orchestrator = FakeOrchestrator(store, campaign_id)
        orchestrators[campaign_id] = orchestrator
        repository = CampaignRepository(tmp_path / "artifacts", campaign_id)
        no_close = SimpleNamespace(close=lambda: None)
        return CampaignEntry(
            campaign_id=campaign_id,
            repository=repository,
            adapters=AdapterBundle(no_close, no_close, no_close),
            orchestrator=orchestrator,
            resource_ids=("bench-1",),
        )

    runtime = CampaignRuntime(
        _runtime_config(tmp_path, mode),
        store=store,
        entry_factory=entry_factory,
    )
    return ConcreteCampaignService(runtime), runtime, orchestrators


def _create_request(
    campaign_id: str = "campaign-1",
    path: TuningRequestPath = TuningRequestPath.PATH_EXHAUSTIVE,
) -> CreateCampaignRequest:
    return CreateCampaignRequest(
        campaign_id=campaign_id,
        baseline_id="B0",
        baseline_hash=HASH_A,
        config_hash=HASH_B,
        tuning_path=path,
    )


def test_service_rejects_default_hierarchical_campaign_when_capability_is_missing(
    tmp_path: Path,
):
    service, runtime, _ = _service(tmp_path)
    request = CreateCampaignRequest(
        campaign_id="campaign-1",
        baseline_id="B0",
        baseline_hash=HASH_A,
        config_hash=HASH_B,
    )

    with pytest.raises(OptimizerCapabilityError, match="PATH_HIERARCHICAL"):
        service.create_campaign(request)

    assert runtime.store.current_state("campaign-1") is None


def test_service_create_get_and_paginated_list_are_idempotent(tmp_path: Path):
    service, runtime, _ = _service(tmp_path)

    first = service.create_campaign(_create_request("campaign-1"))
    repeated = service.create_campaign(_create_request("campaign-1"))
    service.create_campaign(_create_request("campaign-2"))

    assert first == repeated
    assert service.get_campaign("campaign-1").campaign_id == "campaign-1"
    assert [item.campaign_id for item in service.list_campaigns(limit=1, offset=0)] == [
        "campaign-2"
    ]
    assert [item.campaign_id for item in service.list_campaigns(limit=1, offset=1)] == [
        "campaign-1"
    ]
    assert [event.event_type for event in runtime.store.events("campaign-1")].count(
        "CAMPAIGN_PATH_SELECTED"
    ) == 1


def test_service_preflight_and_plan_delegate_to_orchestrator_idempotently(tmp_path: Path):
    service, _runtime, orchestrators = _service(tmp_path)
    service.create_campaign(_create_request())

    first = service.preflight("campaign-1")
    repeated = service.preflight("campaign-1")
    authorization = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=CampaignState.RESTORE_B0,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )
    repeated_authorization = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=CampaignState.RESTORE_B0,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )

    assert first == repeated == {"verified": True}
    assert authorization.stage is CampaignState.RESTORE_B0
    assert authorization.time_policy is AuthorizationTimePolicy.BOUNDED
    assert authorization.max_duration_s == 30.0
    assert authorization.expires_at is not None
    assert repeated_authorization == authorization
    assert orchestrators["campaign-1"].calls == [
        "start_preflight",
        "plan:RESTORE_B0",
    ]


def test_service_never_reuses_unbounded_authorization_for_generic_stage_request(
    tmp_path: Path,
) -> None:
    service, _runtime, orchestrators = _service(tmp_path)
    service.create_campaign(_create_request())
    service.preflight("campaign-1")
    now = datetime.now(timezone.utc)
    service._authorization_from_payload = lambda payload: SimpleNamespace(
        campaign_id="campaign-1",
        stage=CampaignState.RESTORE_B0,
        max_candidates=1,
        max_duration_s=30.0,
        time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
        expires_at=None,
    )
    service.runtime.store.append_event(
        "campaign-1",
        "STAGE_PLANNED",
        payload={"authorization": {}},
        expected_state=CampaignState.AWAIT_BASELINE_APPROVAL,
        created_at=now,
    )

    authorization = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=CampaignState.RESTORE_B0,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )

    assert authorization.time_policy is AuthorizationTimePolicy.BOUNDED
    assert authorization.max_duration_s == 30.0
    assert authorization.expires_at is not None
    assert orchestrators["campaign-1"].calls[-1] == "plan:RESTORE_B0"


def test_duplicate_approve_uses_one_background_execution(tmp_path: Path):
    service, runtime, orchestrators = _service(tmp_path)
    service.create_campaign(_create_request())
    service.preflight("campaign-1")
    authorization = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=CampaignState.RESTORE_B0,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )

    first = service.approve_stage("campaign-1", authorization)
    repeated = service.approve_stage("campaign-1", authorization)
    runtime.close()

    assert first == repeated
    assert first["status"] == "ACCEPTED"
    assert orchestrators["campaign-1"].calls.count("restore_b0") == 1
    assert orchestrators["campaign-1"].calls.count("run_baseline_measurement") == 1


@pytest.mark.parametrize(
    "stage",
    [CampaignState.RESTORE_B0, CampaignState.I2C_SCREENING],
)
def test_service_rejects_unbounded_approval_before_cache_or_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: CampaignState,
) -> None:
    service, runtime, orchestrators = _service(tmp_path)
    service.create_campaign(_create_request())
    service.preflight("campaign-1")
    bounded = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=stage,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )
    unbounded = with_authorization_id(
        bounded.model_copy(
            update={
                "authorization_id": "pending",
                "time_policy": AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
                "max_duration_s": None,
                "expires_at": None,
            }
        )
    )
    unbounded = StageAuthorization.model_validate_json(
        unbounded.model_dump_json()
    )
    submit_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runtime,
        "submit_stage",
        lambda *args: submit_calls.append(args),
    )

    with pytest.raises(AuthorizationError, match="bounded"):
        service.approve_stage("campaign-1", unbounded)

    assert submit_calls == []
    assert service._approval_responses == {}
    assert orchestrators["campaign-1"].calls == [
        "start_preflight",
        f"plan:{stage.value}",
    ]
    runtime.close()


def test_duplicate_approve_after_service_restart_never_reexecutes_stage(tmp_path: Path):
    service, runtime, orchestrators = _service(tmp_path)
    service.create_campaign(_create_request())
    service.preflight("campaign-1")
    authorization = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=CampaignState.RESTORE_B0,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )
    service.approve_stage("campaign-1", authorization)
    operation = f"APPROVE:{authorization.authorization_id}"
    for _ in range(100):
        if any(
            event.event_type == "WORKER_COMPLETED"
            and event.payload.get("operation") == operation
            for event in runtime.store.events("campaign-1")
        ):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("background approval did not complete")

    restarted_service = ConcreteCampaignService(runtime)
    repeated = restarted_service.approve_stage("campaign-1", authorization)
    time.sleep(0.05)

    assert repeated["status"] == "COMPLETED"
    assert orchestrators["campaign-1"].calls.count("restore_b0") == 1
    assert orchestrators["campaign-1"].calls.count("run_baseline_measurement") == 1
    runtime.close()


def test_read_only_service_rejects_approve_before_worker_or_write(tmp_path: Path):
    service, runtime, orchestrators = _service(tmp_path, RuntimeMode.READ_ONLY)
    service.create_campaign(_create_request())
    service.preflight("campaign-1")
    authorization = service.plan_stage(
        "campaign-1",
        StagePlanRequest(
            stage=CampaignState.RESTORE_B0,
            max_candidates=1,
            max_duration_s=30.0,
        ),
    )

    with pytest.raises(ReadOnlyModeError, match="READ_ONLY"):
        service.approve_stage("campaign-1", authorization)

    assert "restore_b0" not in orchestrators["campaign-1"].calls
    runtime.close()


def test_missing_fir_proposer_fails_closed_before_fir_stage(tmp_path: Path):
    service, runtime, _ = _service(tmp_path)
    service.create_campaign(_create_request())
    service.preflight("campaign-1")

    with pytest.raises(
        OptimizerCapabilityError,
        match="TDECQ-only diagnostic paths",
    ):
        service.plan_stage(
            "campaign-1",
            StagePlanRequest(
                stage=CampaignState.FIR_APPLY,
                max_candidates=1,
                max_duration_s=30.0,
            ),
        )

    runtime.close()


def test_pause_resume_abort_and_rollback_are_idempotent_service_delegations(
    tmp_path: Path,
):
    service, runtime, orchestrators = _service(tmp_path)
    service.create_campaign(_create_request("pause-campaign"))
    service.preflight("pause-campaign")

    assert service.pause_campaign("pause-campaign") is CampaignState.PAUSED
    assert service.pause_campaign("pause-campaign") is CampaignState.PAUSED
    assert service.resume_campaign("pause-campaign") is (
        CampaignState.AWAIT_COARSE_APPROVAL
    )
    assert service.resume_campaign("pause-campaign") is (
        CampaignState.AWAIT_COARSE_APPROVAL
    )
    assert orchestrators["pause-campaign"].calls.count("pause") == 1
    assert orchestrators["pause-campaign"].calls.count("resume") == 1

    for campaign_id, operation in (
        ("abort-campaign", service.abort_campaign),
        ("rollback-campaign", service.rollback_campaign),
    ):
        service.create_campaign(_create_request(campaign_id))
        service.preflight(campaign_id)
        service.plan_stage(
            campaign_id,
            StagePlanRequest(
                stage=CampaignState.RESTORE_B0,
                max_candidates=1,
                max_duration_s=30.0,
            ),
        )
        assert operation(campaign_id) is CampaignState.FAILED
        assert operation(campaign_id) is CampaignState.FAILED
        assert orchestrators[campaign_id].calls.count("abort") == 1
    runtime.close()


def test_best_observation_uses_repeated_candidate_median_not_single_minimum(
    tmp_path: Path,
):
    service, runtime, _ = _service(tmp_path)
    service.create_campaign(_create_request())
    repository = runtime.entry("campaign-1").repository
    for observation_id, candidate_id, tdecq_db in (
        ("obs-a-1", "candidate-a", 1.0),
        ("obs-a-2", "candidate-a", 10.0),
        ("obs-b-1", "candidate-b", 3.0),
        ("obs-b-2", "candidate-b", 3.0),
    ):
        repository.publish_observation(
            observation_id,
            {
                "candidate_id": candidate_id,
                "feasible": True,
                "metrics": {"tdecq_db": tdecq_db},
                "valid": True,
            },
            (),
        )

    best = service.get_best_observation("campaign-1")
    report = service.get_report("campaign-1")

    assert best == {
        "candidate_id": "candidate-b",
        "measurement_count": 2,
        "tdecq_median_db": 3.0,
        "observation_ids": ["obs-b-1", "obs-b-2"],
    }
    assert len(service.list_observations("campaign-1")) == 4
    assert report["observation_count"] == 4
    assert report["best_observation"] == best
    runtime.close()
