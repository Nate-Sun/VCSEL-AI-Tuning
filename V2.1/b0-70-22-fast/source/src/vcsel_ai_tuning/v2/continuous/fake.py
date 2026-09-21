"""Deterministic non-hardware dependencies for continuous controller tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
)
from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)

from .models import (
    ContinuousRequest,
    ExecutionMode,
    ExecutionOutcome,
    ProviderAction,
    ProviderActionKind,
    ReasonCode,
    StoppingPolicy,
)
from .policy import DEFAULT_I2C_POLICY


FAKE_B0 = {
    "ibias_dac": 70,
    "gain_ctrl": 5,
    "eq": 22,
    "efin_cur": 15,
    "vga_cur": 13,
    "res_gain": 6,
    "res_bw": 0,
    "vbias_ref": 15,
    "vref": 15,
}


class SystemClock:
    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def monotonic() -> float:
        import time

        return time.monotonic()


class ScriptedContinuousProvider:
    provider_id = "fake-provider"
    provider_version = "fake-v1"

    def __init__(self, actions: list[ProviderAction]) -> None:
        self._actions = list(actions)

    def decide(self, context: Any) -> ProviderAction:
        del context
        if not self._actions:
            raise RuntimeError("fake provider has no scripted action")
        return self._actions.pop(0)


class ScriptedContinuousExecution:
    def __init__(
        self,
        outcomes: list[ExecutionOutcome | Exception],
        *,
        restore_ok: bool = True,
    ) -> None:
        self._outcomes = list(outcomes)
        self._restore_ok = restore_ok

    def execute_candidate(self, authorization, candidate) -> ExecutionOutcome:
        del authorization, candidate
        return self._next()

    def measure_current_state(self) -> ExecutionOutcome:
        return self._next()

    def restore_b0(self, expected) -> dict[str, int]:
        if not self._restore_ok:
            raise RuntimeError("injected B0 recovery failure")
        return dict(expected)

    def recover_b0_after_failure(self, expected) -> dict[str, int]:
        if not self._restore_ok:
            raise RuntimeError("injected B0 recovery failure")
        return dict(expected)

    def _next(self) -> ExecutionOutcome:
        if not self._outcomes:
            raise RuntimeError("fake execution has no scripted outcome")
        value = self._outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def fake_eye(
    artifact_root: Path,
    measurement_id: str,
    *,
    readback: dict[str, int],
    tdecq_db: float,
    outer_er_db: float = 4.5,
) -> EyeSample:
    relative = f"fake_inputs/{measurement_id}.png"
    path = artifact_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"FAKE_EYE:{measurement_id}".encode("ascii")
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    reference = ArtifactReference(
        relative_path=relative,
        sha256=digest,
        media_type="image/png",
    )
    return EyeSample(
        measurement_id=measurement_id,
        campaign_id="campaign-1",
        sweep_point_id=measurement_id,
        raw_eye_density_ref=reference,
        metrics=EyeMetrics(
            tdecq_db=tdecq_db,
            outer_er_chan1a_db=outer_er_db,
        ),
        requested_params={},
        applied_params={name: float(value) for name, value in readback.items()},
        readback_params={name: float(value) for name, value in readback.items()},
        dca_config_fingerprint="a" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=datetime.now(timezone.utc),
        artifact_hashes={relative: digest},
    )


def fake_request(
    result_dir: Path,
    initial: EyeSample,
    *,
    max_rounds: int,
    target_tdecq_db: float = 2.0,
) -> ContinuousRequest:
    now = datetime.now(timezone.utc)
    authorization = StageAuthorization(
        authorization_id="fake-auth-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_FINE,
        baseline_hash="b" * 64,
        rollback_hash="c" * 64,
        parameter_specs=DEFAULT_I2C_POLICY.authorization_specs(),
        max_candidates=max_rounds,
        max_duration_s=300.0,
        issued_at=now,
        expires_at=now + timedelta(minutes=10),
        rollback_snapshot=AppliedSettings(readback_i2c=FAKE_B0),
    )
    return ContinuousRequest(
        campaign_id="campaign-1",
        initial_eye=initial,
        authorization=authorization,
        policy=DEFAULT_I2C_POLICY.model_dump(mode="json"),
        max_rounds=max_rounds,
        provider_config={"provider_id": "fake-provider"},
        stopping_policy=StoppingPolicy(target_tdecq_db=target_tdecq_db),
        b0_path=Path("configs/ai_tuning/b0.yaml"),
        b0_i2c=FAKE_B0,
        result_dir=result_dir,
        execution_mode=ExecutionMode.FAKE,
    )


def fake_scenario(
    result_dir: Path,
    scenario: str,
    *,
    max_rounds: int,
    target_tdecq_db: float = 2.0,
) -> tuple[ScriptedContinuousProvider, ScriptedContinuousExecution, ContinuousRequest]:
    initial = fake_eye(
        result_dir,
        "baseline",
        readback=FAKE_B0,
        tdecq_db=3.2,
    )
    if scenario == "happy":
        actions: list[ProviderAction] = []
        outcomes: list[ExecutionOutcome | Exception] = []
        readback = dict(FAKE_B0)
        proposals = (("eq", 23), ("gain_ctrl", 6), ("res_bw", 1))
        for index in range(max_rounds):
            name, value = proposals[index % len(proposals)]
            readback = readback | {name: value}
            requested = {name: value}
            actions.append(
                ProviderAction(
                    action=ProviderActionKind.PROPOSE,
                    requested_params=requested,
                    reason_codes=(ReasonCode.EXPLORATION,),
                    confidence=0.9,
                    summary="Deterministic FAKE proposal.",
                )
            )
            sample = fake_eye(
                result_dir,
                f"round-{index + 1}",
                readback=readback,
                tdecq_db=3.2 - 0.05 * (index + 1),
            )
            outcomes.append(
                ExecutionOutcome(
                    sample=sample,
                    requested_params=requested,
                    applied_params=requested,
                    readback_params=readback,
                    adapter_identities={"i2c": "fake-i2c", "dca": "fake-dca"},
                    evidence_complete=True,
                )
            )
        restore_ok = True
    elif scenario == "stop-acceptance":
        actions = [
            ProviderAction(
                action=ProviderActionKind.STOP,
                reason_codes=(ReasonCode.EXHAUSTED_SEARCH,),
                confidence=0.9,
                summary="Accept the current fake observation.",
            )
        ]
        confirmation = fake_eye(
            result_dir,
            "acceptance",
            readback=FAKE_B0,
            tdecq_db=3.1,
            outer_er_db=4.6,
        )
        outcomes = [
            ExecutionOutcome(
                sample=confirmation,
                requested_params={},
                applied_params={},
                readback_params=FAKE_B0,
                adapter_identities={"i2c": "fake-i2c", "dca": "fake-dca"},
                evidence_complete=True,
            )
        ]
        restore_ok = True
    elif scenario in {"readback-mismatch", "dca-failure"}:
        actions = [
            ProviderAction(
                action=ProviderActionKind.PROPOSE,
                requested_params={"eq": 23},
                reason_codes=(ReasonCode.EXPLORATION,),
                confidence=0.9,
                summary="Injected failure proposal.",
            )
        ]
        outcomes = [RuntimeError(f"injected {scenario}")]
        restore_ok = True
    elif scenario == "b0-recovery-failure":
        actions = []
        outcomes = []
        restore_ok = False
    else:
        raise ValueError(f"unknown FAKE scenario: {scenario}")
    return (
        ScriptedContinuousProvider(actions),
        ScriptedContinuousExecution(outcomes, restore_ok=restore_ok),
        fake_request(
            result_dir,
            initial,
            max_rounds=max_rounds,
            target_tdecq_db=target_tdecq_db,
        ),
    )
