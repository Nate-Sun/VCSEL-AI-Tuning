"""Bridge one Kimi decision into the existing campaign safety chain."""

from __future__ import annotations

from concurrent.futures import Future
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

from vcsel_ai_tuning.campaign.models import Candidate, StageAuthorization
from vcsel_ai_tuning.runtime import CampaignRuntime

from .brain import (
    OptimizationBrain,
    analyze_factory_gated,
    is_factory_gated_brain,
)
from .contracts import (
    BrainCandidate,
    BrainResult,
    DecisionAction,
    EyeSample,
)
from .kimi_memory import KIMI_PROVIDER_ID, KimiCampaignMemory


_GUARD_PROVIDER_ID = "brain-guard"
_GUARD_PROVIDER_VERSION = "1"


class CandidateRunner(Protocol):
    def run_one_candidate(
        self,
        authorization: StageAuthorization,
        candidate: Candidate,
    ) -> tuple[object, ...]: ...


class KimiResultPublicationError(RuntimeError):
    """Raised when execution succeeded but its summary artifact was not published."""

    __slots__ = ("_observation_ids",)
    hardware_executed = True

    def __init__(self, observation_ids: tuple[str, ...]) -> None:
        super().__init__("Kimi result publication failed after candidate execution")
        self._observation_ids = tuple(observation_ids)

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return self._observation_ids


def _require_concrete_memory(memory: object) -> KimiCampaignMemory:
    if type(memory) is not KimiCampaignMemory:
        raise TypeError("memory must be a KimiCampaignMemory")
    return memory


def _require_factory_gated_brain(brain: object) -> OptimizationBrain:
    if not is_factory_gated_brain(brain):
        raise TypeError("brain must be created by BrainFactory")
    return brain


def _validate_campaign_binding(
    campaign_id: str,
    sample: EyeSample,
    authorization: StageAuthorization,
    memory: KimiCampaignMemory,
) -> None:
    campaign_ids = (
        campaign_id,
        sample.campaign_id,
        authorization.campaign_id,
        memory.campaign_id,
    )
    if any(value != campaign_ids[0] for value in campaign_ids[1:]):
        raise ValueError(
            "campaign_id must match sample, authorization, and Kimi memory"
        )


def _preflight_result_publication(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".preflight",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(b"KIMI_RESULT_PREFLIGHT\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.unlink()
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _strict_brain_result(value: object) -> BrainResult:
    try:
        return BrainResult.model_validate(
            value.model_dump(round_trip=True, warnings=False),
            strict=True,
        )
    except Exception as error:
        raise ValueError("brain returned an invalid BrainResult") from error


def _validate_kimi_non_proposal_identity(
    brain: OptimizationBrain,
    brain_result: BrainResult,
) -> None:
    perception = brain_result.perception
    if perception.provider_id != KIMI_PROVIDER_ID:
        return
    brain_key = brain.provider_key
    brain_version = brain.provider_version
    if brain_key != KIMI_PROVIDER_ID:
        raise ValueError("Kimi non-PROPOSE requires the Kimi brain provider")
    if perception.provider_version != brain_version:
        raise ValueError("Kimi non-PROPOSE perception provider version mismatch")

    decision = brain_result.decision
    if decision.provider_id == KIMI_PROVIDER_ID:
        if decision.provider_version != brain_version:
            raise ValueError("Kimi non-PROPOSE decision provider version mismatch")
        return
    if not _is_canonical_guard_hold(decision):
        raise ValueError(
            "Kimi non-PROPOSE decision must be Kimi or a canonical brain-guard HOLD"
        )


def _is_canonical_guard_hold(decision) -> bool:
    return (
        decision.action is DecisionAction.HOLD
        and decision.provider_id == _GUARD_PROVIDER_ID
        and decision.provider_version == _GUARD_PROVIDER_VERSION
        and decision.candidate_id is None
        and decision.requested_params is None
    )


def _persist_and_verify(
    memory: KimiCampaignMemory,
    sample: EyeSample,
    brain_result: BrainResult,
) -> None:
    KimiCampaignMemory.append(memory, sample, brain_result)
    snapshot = KimiCampaignMemory.read_snapshot(memory)
    if not snapshot.entries:
        raise ValueError("persisted Kimi assessment could not be verified")
    persisted = snapshot.entries[-1]
    if (
        persisted.sample != sample
        or persisted.perception != brain_result.perception
        or persisted.decision != brain_result.decision
    ):
        raise ValueError("persisted Kimi assessment could not be verified")


def _canonical_params(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_result(
    output_path: Path,
    brain_result: BrainResult,
    execution_status: str,
    observations: tuple[object, ...],
    *,
    instruction: str,
    selected: Candidate | None = None,
    error: str | None = None,
) -> dict[str, object]:
    decision_json = brain_result.decision.model_dump(mode="json")
    proposed_params = decision_json["requested_params"]
    selected_local_params = (
        selected.model_dump(mode="json")["requested_i2c"]
        if selected is not None
        else None
    )
    document: dict[str, object] = {
        "measurement_id": brain_result.perception.measurement_id,
        "instruction": instruction,
        "brain_mode": brain_result.brain_mode.value,
        "model": brain_result.perception.provider_version,
        "perception_provider_id": brain_result.perception.provider_id,
        "perception_provider_version": brain_result.perception.provider_version,
        "decision_provider_id": brain_result.decision.provider_id,
        "decision_provider_version": brain_result.decision.provider_version,
        "quality_score": brain_result.perception.quality_score,
        "confidence": brain_result.perception.confidence,
        "reference_measurement_id": brain_result.perception.reference_measurement_id,
        "better_than_reference": brain_result.perception.better_than_reference,
        "memory_revision": brain_result.perception.memory_revision,
        "memory_id": brain_result.perception.memory_id,
        "action": brain_result.decision.action.value,
        "candidate_id": brain_result.decision.candidate_id,
        "requested_params": (
            selected_local_params if selected is not None else proposed_params
        ),
        "selected_local_params": selected_local_params,
        "proposed_params": proposed_params,
        "reason_codes": list(brain_result.decision.reason_codes),
        "api_latency_ms": brain_result.latency_ms,
        "execution_status": execution_status,
        "observation_ids": [
            getattr(observation, "observation_id") for observation in observations
        ],
        "error": error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return document


def execute_kimi_step(
    *,
    campaign_id: str,
    brain: OptimizationBrain,
    orchestrator: CandidateRunner,
    memory: KimiCampaignMemory,
    authorization: StageAuthorization,
    sample: EyeSample,
    candidates: tuple[Candidate, ...],
    instruction: str,
    output_path: Path,
) -> dict[str, object]:
    """Analyze once, then execute only the locally supplied selected candidate."""
    memory = _require_concrete_memory(memory)
    brain = _require_factory_gated_brain(brain)
    _validate_campaign_binding(campaign_id, sample, authorization, memory)
    if any(candidate.requested_fir is not None for candidate in candidates):
        raise ValueError("V2.1 Kimi step accepts I2C-only candidates")
    safe_candidates = tuple(
        BrainCandidate(
            candidate_id=candidate.candidate_id,
            requested_params=dict(candidate.requested_i2c),
        )
        for candidate in candidates
    )
    brain_result = _strict_brain_result(
        analyze_factory_gated(brain, sample, (), safe_candidates)
    )
    if brain_result.decision.action is not DecisionAction.PROPOSE:
        _validate_kimi_non_proposal_identity(brain, brain_result)
        if brain_result.perception.provider_id == KIMI_PROVIDER_ID:
            _persist_and_verify(memory, sample, brain_result)
        return _write_result(
            output_path,
            brain_result,
            "HOLD",
            (),
            instruction=instruction,
        )

    if (
        brain_result.perception.provider_id != KIMI_PROVIDER_ID
        or brain_result.decision.provider_id != KIMI_PROVIDER_ID
    ):
        raise ValueError("PROPOSE requires Kimi provider identity")
    if brain.provider_key != KIMI_PROVIDER_ID:
        raise ValueError("PROPOSE requires the Kimi brain provider")
    if not (
        brain_result.perception.provider_version
        == brain_result.decision.provider_version
        == brain.provider_version
    ):
        raise ValueError("PROPOSE provider versions must match")
    matches = tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id == brain_result.decision.candidate_id
    )
    if len(matches) != 1:
        raise ValueError("PROPOSE must resolve to exactly one supplied local candidate")
    selected = matches[0]
    local_params = selected.model_dump(mode="json")["requested_i2c"]
    proposed_params = brain_result.decision.model_dump(mode="json")[
        "requested_params"
    ]
    if _canonical_params(proposed_params) != _canonical_params(local_params):
        raise ValueError("PROPOSE requested_params must match the selected local candidate")
    _preflight_result_publication(output_path)
    _persist_and_verify(memory, sample, brain_result)
    try:
        observations = orchestrator.run_one_candidate(authorization, selected)
    except Exception as runner_error:
        try:
            _write_result(
                output_path,
                brain_result,
                "FAILED",
                (),
                instruction=instruction,
                selected=selected,
                error="CANDIDATE_EXECUTION_FAILED",
            )
        except Exception:
            runner_error.add_note("Kimi FAILED result publication also failed")
        raise
    observation_ids: tuple[str, ...] = ()
    try:
        observation_ids = tuple(
            getattr(observation, "observation_id") for observation in observations
        )
        return _write_result(
            output_path,
            brain_result,
            "EXECUTED",
            observations,
            instruction=instruction,
            selected=selected,
        )
    except Exception:
        raise KimiResultPublicationError(observation_ids) from None


def submit_kimi_step(
    runtime: CampaignRuntime,
    *,
    campaign_id: str,
    brain: OptimizationBrain,
    memory: KimiCampaignMemory,
    authorization: StageAuthorization,
    sample: EyeSample,
    candidates: tuple[Candidate, ...],
    instruction: str,
    output_path: Path,
) -> Future[dict[str, object]]:
    """Run the step under V2.0's global hardware and resource leases."""
    memory = _require_concrete_memory(memory)
    brain = _require_factory_gated_brain(brain)
    _validate_campaign_binding(campaign_id, sample, authorization, memory)
    return runtime.submit_stage(
        campaign_id,
        f"KIMI_STEP:{sample.measurement_id}",
        lambda entry: execute_kimi_step(
            campaign_id=campaign_id,
            brain=brain,
            orchestrator=entry.orchestrator,
            memory=memory,
            authorization=authorization,
            sample=sample,
            candidates=candidates,
            instruction=instruction,
            output_path=output_path,
        ),
    )
