"""Lazy production V1.1 LIVE_TUNING gateway assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .domain import RuntimeMode
from .live_ai_policy import (
    LiveAIProfile,
    STAGED_BUDGETS,
    validate_live_ai_payload,
)
from .read_only_runtime import ReadOnlyV11Gateway
from .service import GatewayAttestation


class LiveV11Gateway(ReadOnlyV11Gateway):
    """Use the audited read-only capture for LIVE authorization attestation.

    Construction is intentionally inert. Hardware drivers are opened only by
    an explicit Preflight capture or a later authorization attestation.
    Mutation remains owned by the separately authorized task executor.
    """

    @property
    def runtime_mode(self) -> RuntimeMode:
        return RuntimeMode.LIVE

    def capabilities(self) -> Mapping[str, object]:
        if self._live_ai_profile is LiveAIProfile.PILOT_3_THEN_FAST24:
            return {
                "campaign_types": ["AI_OPTIMIZATION"],
                "dynamic_b0": True,
                "full_campaign_authorization": True,
                "hardware_connected": False,
                "hardware_mutation": True,
                "fake": False,
                "live_ai_profile": self._live_ai_profile.value,
                **STAGED_BUDGETS,
                "measurement_timeout_s": self._measurement_timeout_s,
                "cleanup_grace_s": self._cleanup_grace_s,
                "dataset_required": True,
                "fir_authority": "FORMULA_ONLY_MAIN_NLC_PATTERN_FROZEN_L1_170",
                "final_restore": "B0_VERIFIED",
                "profile_options": [
                    {
                        "name": self._live_ai_profile.value,
                        "candidate_budget": 27,
                        "max_alternating_rounds": 2,
                    }
                ],
            }
        return {
            "campaign_types": ["AI_OPTIMIZATION", "SWEEP"],
            "dynamic_b0": True,
            "full_campaign_authorization": True,
            "hardware_connected": False,
            "hardware_mutation": True,
            "fake": False,
        }

    def validate_ai_payload(self, payload: Mapping[str, object]) -> None:
        if self._live_ai_profile is None:
            raise ValueError("LIVE AI execution profile is unavailable")
        validate_live_ai_payload(self._live_ai_profile, payload)

    def current_attestation(self) -> GatewayAttestation:
        return GatewayAttestation.from_capture(self.preflight_capture())
    def __init__(
        self,
        *args: Any,
        live_ai_profile: LiveAIProfile | None = None,
        measurement_timeout_s: float = 60.0,
        cleanup_grace_s: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._live_ai_profile = live_ai_profile
        self._measurement_timeout_s = float(measurement_timeout_s)
        self._cleanup_grace_s = float(cleanup_grace_s)
