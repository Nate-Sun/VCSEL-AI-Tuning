"""V2 fused-brain adapter for Kimi K3."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import (
    BrainCandidate,
    DecisionAction,
    DecisionProposal,
    EyeSample,
    PerceptionResult,
    V2Observation,
)
from ..kimi_memory import KIMI_PROVIDER_ID, KimiCampaignMemory
from ..preprocessing import preprocess_eye_density
from .kimi_client import KimiApiClient


def _png_bytes(model_input: np.ndarray) -> bytes:
    pixels = np.rint(np.clip(model_input, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _prompt_summary(value: Mapping[str, object]) -> dict[str, object]:
    """Copy immutable snapshot data into the strict client payload shape."""
    return {
        key: _prompt_value(item)
        for key, item in value.items()
    }


def _prompt_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _prompt_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prompt_value(item) for item in value)
    return value


class KimiProvider:
    """Give Kimi a verified eye image and map its choice to local candidates."""

    provider_id = KIMI_PROVIDER_ID

    def __init__(
        self,
        client: KimiApiClient,
        *,
        artifact_root: str | Path,
        instruction: str,
        memory: KimiCampaignMemory,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be blank")
        if type(memory) is not KimiCampaignMemory:
            raise TypeError("memory must be a KimiCampaignMemory")
        self._client = client
        self._artifact_root = Path(artifact_root)
        self._instruction = instruction
        self._memory = memory
        self.provider_version = client.model

    def analyze_once(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> tuple[PerceptionResult, DecisionProposal]:
        del history
        if sample.campaign_id != self._memory.campaign_id:
            raise ValueError("sample campaign_id does not match Kimi memory")
        snapshot = KimiCampaignMemory.read_snapshot(self._memory, limit=5)
        if any(
            entry.sample.measurement_id == sample.measurement_id
            for entry in snapshot.entries
        ):
            raise ValueError("duplicate measurement_id in Kimi memory")
        best = snapshot.best_entry

        png = _png_bytes(preprocess_eye_density(sample, self._artifact_root))
        reference_png = (
            _png_bytes(preprocess_eye_density(best.sample, self._artifact_root))
            if best is not None
            else None
        )
        reference_measurement_id = (
            best.sample.measurement_id if best is not None else None
        )
        payload = self._client.analyze(
            png_bytes=png,
            instruction=self._instruction,
            sample=sample,
            safe_candidates=safe_candidates,
            reference_png_bytes=reference_png,
            reference_measurement_id=reference_measurement_id,
            recent_summaries=tuple(
                _prompt_summary(summary) for summary in snapshot.recent_summaries
            ),
        )
        action = DecisionAction(payload["action"])
        raw_candidate_id = payload.get("candidate_id")
        selected = None
        if action is DecisionAction.PROPOSE:
            if not isinstance(raw_candidate_id, str):
                raise ValueError("Kimi PROPOSE response has no candidate_id")
            selected = next(
                (
                    candidate
                    for candidate in safe_candidates
                    if candidate.candidate_id == raw_candidate_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("Kimi selected a candidate outside the safe set")

        perception = PerceptionResult(
            measurement_id=sample.measurement_id,
            quality_score=payload["quality_score"],
            confidence=payload["confidence"],
            valid=payload["valid"],
            ood=payload["ood"],
            defects={},
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            input_hash=sample.raw_eye_density_ref.sha256,
            memory_revision=snapshot.revision,
            memory_id=snapshot.memory_id,
            reference_measurement_id=reference_measurement_id,
            better_than_reference=payload["better_than_reference"],
        )
        decision = DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=action,
            candidate_id=selected.candidate_id if selected is not None else None,
            requested_params=(
                selected.requested_params if selected is not None else None
            ),
            reason_codes=tuple(payload["reason_codes"]),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )
        return perception, decision
