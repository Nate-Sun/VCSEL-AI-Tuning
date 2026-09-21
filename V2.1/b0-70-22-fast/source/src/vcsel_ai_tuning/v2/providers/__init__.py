"""Hardware-neutral deterministic fake providers for V2 tests and replay."""

from .fake import (
    FakeEyeScorer,
    FirstCandidateDecisionProvider,
    ScriptedFakeBrain,
)

__all__ = [
    "FakeEyeScorer",
    "FirstCandidateDecisionProvider",
    "ScriptedFakeBrain",
]
