"""V2 fused-brain adapter for Kimi K3."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import (
    BrainCandidate,
    DecisionAction,
    DecisionProposal,
    EyeSample,
    MeasurementStatus,
    PerceptionResult,
    V2Observation,
)
from ..kimi_memory import KIMI_PROVIDER_ID, KimiCampaignMemory
from ..preprocessing import preprocess_eye_density
from ..continuous.models import ProviderAction, ProviderActionKind, ReasonCode
from ..continuous.provider import ProviderContext
from .kimi_client import KimiApiClient


def _png_bytes(model_input: np.ndarray) -> bytes:
    pixels = np.rint(np.clip(model_input, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _provider_png(sample: EyeSample, artifact_root: Path) -> bytes:
    reference = sample.raw_eye_density_ref
    if reference.media_type == "image/png":
        root = artifact_root.resolve()
        path = (root / reference.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("screen PNG path escapes artifact root") from error
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != reference.sha256:
            raise ValueError("screen PNG hash mismatch")
        return payload
    return _png_bytes(preprocess_eye_density(sample, artifact_root))


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
        scripted_schedule: tuple[Mapping[str, int], ...] = (),
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be blank")
        if type(memory) is not KimiCampaignMemory:
            raise TypeError("memory must be a KimiCampaignMemory")
        self._client = client
        self._artifact_root = Path(artifact_root)
        self._instruction = instruction
        self._memory = memory
        self._scripted_schedule = tuple(dict(point) for point in scripted_schedule)
        self.provider_version = client.model

    def _scripted_required_params(
        self,
        context: ProviderContext,
    ) -> dict[str, int] | None:
        if not self._scripted_schedule:
            return None
        schedule_index = context.hardware_round - 1
        if schedule_index >= len(self._scripted_schedule):
            raise ValueError("scripted schedule is exhausted")
        target = self._scripted_schedule[schedule_index]
        required: dict[str, int] = {}
        for name, value in target.items():
            if isinstance(value, bool) or type(value) is not int:
                raise ValueError("scripted schedule values must be strict integers")
            if name not in context.verified_readback:
                raise ValueError(f"scripted parameter {name!r} has no verified readback")
            if context.verified_readback[name] != value:
                required[name] = value
        if not required:
            raise ValueError("scripted round must change at least one parameter")
        if len(required) > context.policy.max_changed_parameters:
            raise ValueError("scripted round changes too many parameters")
        return required

    def decide(self, context: ProviderContext) -> ProviderAction:
        """Return a read-only sparse proposal; local code assigns candidate identity."""

        if not isinstance(context, ProviderContext):
            raise TypeError("context must be a ProviderContext")
        if context.sample.campaign_id != self._memory.campaign_id:
            raise ValueError("sample campaign_id does not match Kimi memory")
        required_params = self._scripted_required_params(context)
        instruction = self._instruction
        if required_params is not None:
            encoded = json.dumps(
                required_params,
                sort_keys=True,
                separators=(",", ":"),
            )
            instruction = (
                f"{instruction.rstrip()} SCRIPTED_REQUIRED_PARAMS={encoded}. "
                "Return PROPOSE with requested_params exactly equal to "
                "SCRIPTED_REQUIRED_PARAMS while assessing the real eye truthfully."
            )
        current_png = _provider_png(context.sample, self._artifact_root)
        best = context.best_reference
        reference_png = (
            _provider_png(best, self._artifact_root)
            if best is not None
            else None
        )
        payload = self._client.analyze_continuous(
            png_bytes=current_png,
            instruction=instruction,
            context=context,
            reference_png_bytes=reference_png,
        )
        if payload["ood"] or (
            not payload["valid"]
            and context.sample.measurement_status is not MeasurementStatus.INVALID
        ):
            reason = (
                ReasonCode.OOD
                if payload["ood"]
                else ReasonCode.INVALID_OBSERVATION
            )
            return ProviderAction(
                action=ProviderActionKind.HOLD,
                requested_params=None,
                reason_codes=(reason,),
                confidence=payload["confidence"],
                summary=payload["summary"],
            )
        if required_params is not None:
            confirmed = (
                payload["action"] == ProviderActionKind.PROPOSE.value
                and payload["requested_params"] == required_params
            )
            disposition = "SCRIPT_CONFIRMED" if confirmed else "SCRIPT_ENFORCED"
            summary = (
                f"{disposition} R{context.hardware_round}: "
                f"{payload['summary']}"
            )[:512]
            return ProviderAction(
                action=ProviderActionKind.PROPOSE,
                requested_params=required_params,
                reason_codes=(ReasonCode.EXPLORATION,),
                confidence=payload["confidence"],
                summary=summary,
            )
        return ProviderAction(
            action=ProviderActionKind(payload["action"]),
            requested_params=payload["requested_params"],
            reason_codes=tuple(ReasonCode(code) for code in payload["reason_codes"]),
            confidence=payload["confidence"],
            summary=payload["summary"],
        )

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
