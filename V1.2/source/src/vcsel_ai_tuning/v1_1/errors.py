"""Lightweight V1.1 errors safe to import in every runtime mode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ReadOnlyPreflightError(RuntimeError):
    """Raised when read-only hardware evidence cannot be safely attested."""

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        stage: str | None = None,
        failure_artifact_id: str | None = None,
        failure_artifact_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})
        lowered = message.lower()
        self.stage = stage or (
            "timeout"
            if "timeout" in lowered
            else "i2c"
            if "i2c" in lowered or "restore" in lowered
            else "bert"
            if "bert" in lowered or "ssprq" in lowered
            else "dca"
            if "dca" in lowered or "tdecq" in lowered or "outer er" in lowered
            else "preflight"
        )
        self.failure_artifact_id = failure_artifact_id
        self.failure_artifact_path = failure_artifact_path

    def attach_failure_artifact(self, artifact_id: str, artifact_path: str) -> None:
        self.failure_artifact_id = artifact_id
        self.failure_artifact_path = artifact_path
