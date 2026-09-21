from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from vcsel_ai_tuning.campaign.authorization import (
    canonical_json_sha256,
    rollback_hash_for,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    FirAuthorizationSpec,
    FirSettings,
    ParameterSpec,
    StageAuthorization,
    load_b0,
)
from vcsel_ai_tuning.campaign.orchestrator import (
    CampaignOrchestrator,
    FinalVerificationResult,
    MeasurementResult,
    OrchestratorPolicy,
)
from vcsel_ai_tuning.campaign.repository import (
    ArtifactSpec,
    CampaignRepository,
    DuplicateObservationError,
)
from vcsel_ai_tuning.campaign.state_machine import CampaignStateMachine
from vcsel_ai_tuning.campaign.store import CampaignStore
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    OptimizationContext,
    TuningObservation,
    TuningPath,
)


CAMPAIGN_ID = "hierarchical-replay-1"
NOW = datetime(2026, 7, 23, 8, 0, tzinfo=timezone.utc)
FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "hierarchical_campaign.jsonl"
)


class ReplayClock:
    def __init__(self) -> None:
        self.current = NOW
        self.elapsed_s = 0.0

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.elapsed_s


class ReplayDca:
    def __init__(self, measurements: Iterable[MeasurementResult]) -> None:
        self.measurements = deque(measurements)
        self.measured_labels: list[str] = []

    def preflight(self) -> Mapping[str, object]:
        return {
            "identity": "REPLAY-DCA",
            "identity_verified": True,
            "configuration_verified": True,
            "measurement_source": "REPLAY",
        }

    def measure_single(self) -> MeasurementResult:
        if not self.measurements:
            raise AssertionError("replay DCA has no scripted measurement")
        measurement = self.measurements.popleft()
        self.measured_labels.append(str(measurement.raw_responses["label"]))
        return measurement


class ReplayI2c:
    def __init__(self, settings: Mapping[str, int]) -> None:
        self.settings = dict(settings)
        self.requests: list[dict[str, int]] = []

    def preflight(self) -> Mapping[str, object]:
        return {
            "identity": "REPLAY-I2C",
            "identity_verified": True,
            "configuration_verified": True,
            "device_address": 0xA6,
            "channel": 0,
        }

    def apply(self, requested: Mapping[str, int]) -> Mapping[str, int]:
        applied = dict(requested)
        self.requests.append(applied)
        self.settings.update(applied)
        return applied

    def readback(self) -> Mapping[str, int]:
        return dict(self.settings)


class ReplayBert:
    def __init__(self, pattern: str, fir: FirSettings) -> None:
        self.pattern = pattern
        self.fir = fir
        self.writes: list[tuple[str, FirSettings]] = []

    def preflight(self) -> Mapping[str, object]:
        return {
            "identity": "REPLAY-BERT",
            "identity_verified": True,
            "configuration_verified": True,
            "port": "port1",
            "channel": "0",
        }

    def apply(self, pattern: str, fir: FirSettings) -> FirSettings:
        self.pattern = pattern
        self.fir = fir
        self.writes.append((pattern, fir))
        return fir

    def readback(self) -> Mapping[str, object]:
        return {"pattern": self.pattern, "fir": self.fir}


class ReplayOptimizer:
    def __init__(self, script: Iterable[CandidateSuggestion | NoSafeCandidate]) -> None:
        self.script = deque(script)
        self.histories: list[tuple[TuningObservation, ...]] = []

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[tuple[int, ...]] = frozenset(),
    ) -> CandidateSuggestion | NoSafeCandidate:
        assert pending_keys == frozenset()
        self.histories.append(observations)
        if not self.script:
            raise AssertionError("replay optimizer has no scripted result")
        return self.script.popleft()

    @property
    def received_observations(self) -> tuple[TuningObservation, ...]:
        by_id: dict[str, TuningObservation] = {}
        for history in self.histories:
            for observation in history:
                by_id[observation.observation_id] = observation
        return tuple(by_id.values())


class ReplayFirProposer:
    def __init__(self, target: FirSettings) -> None:
        self.target = target
        self.calls: list[FirSettings] = []

    def propose(self, current_fir: FirSettings) -> FirSettings:
        self.calls.append(current_fir)
        return self.target


def _load_fixture() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _measurement(record: Mapping[str, Any]) -> MeasurementResult:
    label = str(record["label"])
    return MeasurementResult(
        acquisition_valid=bool(record.get("acquisition_valid", True)),
        er_db=record.get("er_db"),
        er_status=str(record.get("er_status", "CORR")),
        tdecq_db=record.get("tdecq_db"),
        tdecq_status=str(record.get("tdecq_status", "CORR")),
        tdecq_invalid_reason=record.get("tdecq_invalid_reason"),
        elapsed_s=float(record.get("elapsed_s", 0.2)),
        raw_responses={
            "label": label,
            "stage": record["stage"],
            "tdecq_status": record.get("tdecq_status", "CORR"),
        },
        artifacts=(
            ArtifactSpec.from_json(
                "raw_responses",
                f"{label}.json",
                dict(record),
            ),
        ),
    )


def _optimizer_result(
    record: Mapping[str, Any],
) -> CandidateSuggestion | NoSafeCandidate:
    path = TuningPath(str(record.get("path", "PATH_TDECQ_BAYESIAN")))
    if record["kind"] == "stop":
        return NoSafeCandidate(path=path, reason=str(record["reason"]))
    return CandidateSuggestion(
        path=path,
        candidate_id=str(record["candidate_id"]),
        requested={
            str(name): int(value)
            for name, value in dict(record["requested"]).items()
        },
        reason=str(record["reason"]),
    )


def _snapshot(
    i2c: Mapping[str, int],
    fir: FirSettings,
) -> AppliedSettings:
    return AppliedSettings(readback_i2c=dict(i2c), readback_fir=fir)


def _i2c_authorization(
    baseline_hash: str,
    stage: CampaignState,
    rollback: AppliedSettings,
    *,
    max_candidates: int = 1,
) -> StageAuthorization:
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id=CAMPAIGN_ID,
        mode=AuthorizationMode.PER_STAGE,
        stage=stage,
        baseline_hash=baseline_hash,
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
        max_duration_s=120.0,
        issued_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(hours=1),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def _fir_authorization(
    baseline_hash: str,
    rollback: AppliedSettings,
) -> StageAuthorization:
    stage_entry_fir = rollback.readback_fir
    assert stage_entry_fir is not None
    fir_spec = FirAuthorizationSpec(
        tap_min=tuple(value - 4 for value in stage_entry_fir.taps),
        tap_max=tuple(value + 4 for value in stage_entry_fir.taps),
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
        baseline_hash=baseline_hash,
        rollback_hash=rollback_hash_for(rollback),
        fir_spec=fir_spec,
        max_candidates=3,
        max_duration_s=120.0,
        issued_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(hours=1),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def _enter_preflight(store: CampaignStore) -> None:
    store.save_state(CAMPAIGN_ID, CampaignState.DRAFT, created_at=NOW)
    machine = CampaignStateMachine(CampaignState.DRAFT)
    target = machine.transition(CampaignState.DRAFT, CampaignState.PREFLIGHT)
    store.save_transition(
        CAMPAIGN_ID,
        CampaignState.DRAFT,
        target,
        event_type="STATE_TRANSITION",
        created_at=NOW,
    )


def test_fixed_hierarchical_campaign_replay_end_to_end(tmp_path):
    records = _load_fixture()
    scenario = next(record for record in records if record["kind"] == "scenario")
    measurement_records = [
        record for record in records if record["kind"] == "measurement"
    ]
    optimizer_records = [
        record
        for record in records
        if record["kind"] in {"suggestion", "stop"}
    ]

    baseline = load_b0()
    baseline_hash = canonical_json_sha256(baseline)
    initial_i2c = {
        **baseline.i2c.model_dump(),
        "ibias_dac": int(scenario["initial_ibias_dac"]),
    }
    initial_fir = baseline.fir.model_copy(
        update={"taps": tuple(int(value) for value in scenario["initial_fir_taps"])}
    )
    full_fir_target = FirSettings(
        taps=tuple(int(value) for value in scenario["full_fir_target"]["taps"]),
        upper_nlc=int(scenario["full_fir_target"]["upper_nlc"]),
        lower_nlc=int(scenario["full_fir_target"]["lower_nlc"]),
    )
    dca = ReplayDca(_measurement(record) for record in measurement_records)
    i2c = ReplayI2c(initial_i2c)
    bert = ReplayBert(baseline.pattern, initial_fir)
    optimizer = ReplayOptimizer(
        _optimizer_result(record) for record in optimizer_records
    )
    fir_proposer = ReplayFirProposer(full_fir_target)
    clock = ReplayClock()
    database = tmp_path / "campaign.sqlite3"
    artifact_root = tmp_path / "artifacts"

    with (
        CampaignStore(database) as store,
        CampaignRepository(artifact_root, CAMPAIGN_ID) as repository,
    ):
        _enter_preflight(store)
        orchestrator = CampaignOrchestrator(
            campaign_id=CAMPAIGN_ID,
            baseline=baseline,
            baseline_hash=baseline_hash,
            optimization_context=OptimizationContext(
                driver_profile_hash="d" * 64,
                vcsel_model="APA4701",
                device_id="replay-vcsel-1",
                test_context_id="replay-53g-ssprq-25c",
            ),
            store=store,
            repository=repository,
            optimizer=optimizer,
            dca=dca,
            i2c=i2c,
            bert=bert,
            clock=clock,
            policy=OrchestratorPolicy(
                meaningful_tdecq_improvement_db=0.05,
                baseline_tdecq_sample_std_db=0.0,
                fine_patience_candidates=5,
                best_point_confirmation_repeats=3,
                fir_reference_tdecq_db=2.5,
                fir_reference_er_db=3.1,
                max_accepted_fir_updates=3,
            ),
            fir_proposer=fir_proposer,
        )

        preflight = orchestrator.preflight()
        assert preflight.verified is True

        initial_snapshot = _snapshot(initial_i2c, initial_fir)
        orchestrator.restore_b0(
            _i2c_authorization(
                baseline_hash,
                CampaignState.RESTORE_B0,
                initial_snapshot,
            )
        )
        baseline_report = orchestrator.run_baseline_measurement()
        assert baseline_report.valid is True

        b0_snapshot = _snapshot(baseline.i2c.model_dump(), baseline.fir)
        screening_authorization = _i2c_authorization(
            baseline_hash,
            CampaignState.I2C_SCREENING,
            b0_snapshot,
        )
        coarse_authorization = _i2c_authorization(
            baseline_hash,
            CampaignState.I2C_COARSE,
            b0_snapshot,
        )
        orchestrator.enter_stage(screening_authorization)
        orchestrator.run_screening_stage(
            screening_authorization,
            next_authorization=coarse_authorization,
        )
        coarse_summary = orchestrator.run_coarse_stage(coarse_authorization)
        assert coarse_summary.convergence_reason == "COARSE_GATE_SATISFIED"

        coarse_snapshot = _snapshot(i2c.readback(), baseline.fir)
        fine_authorization = _i2c_authorization(
            baseline_hash,
            CampaignState.I2C_FINE,
            coarse_snapshot,
            max_candidates=2,
        )
        orchestrator.enter_stage(fine_authorization)
        fine_summary = orchestrator.run_fine_stage(fine_authorization)
        assert fine_summary.convergence_reason == "FINE_STAGE_FEASIBLE"

        fine_snapshot = _snapshot(i2c.readback(), baseline.fir)
        fir_authorization = _fir_authorization(baseline_hash, fine_snapshot)
        orchestrator.enter_stage(fir_authorization)
        first_retune_authorization = _i2c_authorization(
            baseline_hash,
            CampaignState.I2C_RETUNE,
            fine_snapshot,
            max_candidates=2,
        )
        accepted_snapshot = _snapshot(
            {
                **dict(fine_snapshot.readback_i2c),
                "ibias_dac": int(scenario["direct_fir_ibias_dac"]),
            },
            full_fir_target,
        )
        second_retune_authorization = _i2c_authorization(
            baseline_hash,
            CampaignState.I2C_RETUNE,
            accepted_snapshot,
            max_candidates=2,
        )
        fir_summary = orchestrator.run_fir_stage(
            fir_authorization,
            retune_authorizations=(first_retune_authorization,),
        )
        assert fir_summary.convergence_reason == "FIR_ACCEPTED_UPDATE"
        assert i2c.readback() == dict(accepted_snapshot.readback_i2c)
        assert bert.fir == full_fir_target
        assert bert.writes[-1] == (baseline.pattern, full_fir_target)
        assert [
            fir for _, fir in bert.writes if fir != baseline.fir
        ] == [full_fir_target]
        assert i2c.requests[-1] == {
            "ibias_dac": int(scenario["direct_fir_ibias_dac"])
        }

        retune_summary = orchestrator.run_retune_stage(
            second_retune_authorization
        )
        assert retune_summary.convergence_reason == "RETUNE_STAGE_FEASIBLE"
        final_result = orchestrator.run_final_verification_stage()

        assert isinstance(final_result, FinalVerificationResult)
        assert final_result.outcome == scenario["expected_outcome"] == "SUCCESS"
        assert final_result.report.valid is True
        assert store.current_state(CAMPAIGN_ID) == CampaignState.COMPLETED

        events = store.events(CAMPAIGN_ID)
        transition_targets = [
            event.to_state
            for event in events
            if event.from_state != event.to_state
        ]
        assert transition_targets == [
            CampaignState.DRAFT,
            CampaignState.PREFLIGHT,
            CampaignState.AWAIT_BASELINE_APPROVAL,
            CampaignState.RESTORE_B0,
            CampaignState.BASELINE_MEASUREMENT,
            CampaignState.AWAIT_COARSE_APPROVAL,
            CampaignState.I2C_SCREENING,
            CampaignState.I2C_COARSE,
            CampaignState.AWAIT_FINE_APPROVAL,
            CampaignState.I2C_FINE,
            CampaignState.AWAIT_FIR_APPROVAL,
            CampaignState.FIR_APPLY,
            CampaignState.I2C_RETUNE,
            CampaignState.FINAL_VERIFICATION,
            CampaignState.COMPLETED,
        ]
        event_types = [event.event_type for event in events]
        assert "FIR_LAST_ACCEPTED_RESTORED" not in event_types

        published = repository.list_observations()
        assert len(published) == len(measurement_records) == 23
        invalid_tdecq = next(
            observation
            for observation in published
            if observation.payload["raw_responses"]["label"]
            == "coarse-tdecq-invalid"
        )
        assert invalid_tdecq.payload["metrics"]["tdecq_db"] is None
        assert invalid_tdecq.payload["metric_status"] == {
            "er_status": "CORR",
            "tdecq_status": "INV",
            "tdecq_invalid_reason": "instrument returned INV",
        }
        optimizer_invalid = next(
            observation
            for observation in optimizer.received_observations
            if observation.observation_id == invalid_tdecq.observation_id
        )
        assert optimizer_invalid.tdecq_db is None
        assert optimizer_invalid.tdecq_invalid_reason == "instrument returned INV"
        assert all(
            observation.payload["metrics"]["tdecq_db"] is None
            or observation.payload["metrics"]["tdecq_db"] > 0
            for observation in published
        )
        assert all(len(observation.artifacts) == 2 for observation in published)
        assert all(
            repository.get_artifact(artifact.relative_path) == artifact
            for observation in published
            for artifact in observation.artifacts
        )
        first_published = published[0]
        with pytest.raises(DuplicateObservationError):
            repository.publish_observation(
                first_published.observation_id,
                first_published.payload,
                (),
            )
        assert repository.get_observation(first_published.observation_id) == (
            first_published
        )

    assert dca.measurements == deque()
    assert optimizer.script == deque()
    assert fir_proposer.calls == [baseline.fir]
    assert dca.measured_labels == [
        str(record["label"]) for record in measurement_records
    ]
