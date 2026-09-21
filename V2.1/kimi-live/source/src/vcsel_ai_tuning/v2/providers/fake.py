"""Deterministic fake providers with no hardware, network, or artifact side effects."""

from __future__ import annotations

from collections import deque
from math import isfinite
from typing import Iterable

from ..contracts import DecisionAction, DecisionProposal, PerceptionResult


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite numeric probability")
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite numeric probability")
    return float(value)


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


class FakeEyeScorer:
    """A deterministic scorer that binds every score to its supplied eye sample."""

    provider_id = "fake-eye-scorer"
    provider_version = "foundation-1"

    def __init__(
        self,
        *,
        quality_score: float = 0.75,
        confidence: float = 0.95,
        valid: bool = True,
        ood: bool = False,
    ) -> None:
        self._quality_score = _probability("quality_score", quality_score)
        self._confidence = _probability("confidence", confidence)
        self._valid = _strict_bool("valid", valid)
        self._ood = _strict_bool("ood", ood)
        self.calls: list[str] = []

    def score(self, sample) -> PerceptionResult:
        self.calls.append(sample.measurement_id)
        return PerceptionResult(
            measurement_id=sample.measurement_id,
            quality_score=self._quality_score,
            confidence=self._confidence,
            valid=self._valid,
            ood=self._ood,
            defects={},
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            input_hash=sample.raw_eye_density_ref.sha256,
        )


class FirstCandidateDecisionProvider:
    """A deterministic provider that proposes the caller's first safe candidate."""

    provider_id = "fake-first-candidate"
    provider_version = "foundation-1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def propose(self, sample, perception, history, safe_candidates) -> DecisionProposal:
        self.calls.append(sample.measurement_id)
        if not safe_candidates:
            return DecisionProposal(
                based_on_measurement_id=sample.measurement_id,
                action=DecisionAction.HOLD,
                reason_codes=("NO_SAFE_CANDIDATE",),
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        candidate = safe_candidates[0]
        return DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=candidate.candidate_id,
            requested_params=candidate.requested_params,
            reason_codes=("DETERMINISTIC_FAKE_SELECTION",),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


class ScriptedFakeBrain:
    """Independent test brain whose script exhaustion is always an explicit failure."""

    provider_key = "scripted-fake-brain"
    provider_version = "foundation-1"

    def __init__(self, results: Iterable[object]) -> None:
        self._results = deque(results)
        self.calls: list[str] = []

    def analyze(self, sample, history, safe_candidates):
        self.calls.append(sample.measurement_id)
        if not self._results:
            raise AssertionError("scripted fake brain result queue is exhausted")
        return self._results.popleft()
