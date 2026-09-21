from __future__ import annotations

from datetime import datetime
from itertools import pairwise

from .authorization import AuthorizationError, validate_authorization
from .models import CampaignState, StageAuthorization


class IllegalTransition(ValueError):
    """Raised when a requested campaign state transition is not permitted."""


MAIN_CHAIN = (
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
)

MUTATING_STATES = frozenset(
    {
        CampaignState.RESTORE_B0,
        CampaignState.I2C_SCREENING,
        CampaignState.I2C_COARSE,
        CampaignState.I2C_FINE,
        CampaignState.FIR_APPLY,
        CampaignState.I2C_RETUNE,
    }
)

HARDWARE_ACTIVE_STATES = frozenset(
    {
        CampaignState.RESTORE_B0,
        CampaignState.BASELINE_MEASUREMENT,
        CampaignState.I2C_SCREENING,
        CampaignState.I2C_COARSE,
        CampaignState.I2C_FINE,
        CampaignState.FIR_APPLY,
        CampaignState.I2C_RETUNE,
        CampaignState.FINAL_VERIFICATION,
    }
)

TERMINAL_STATES = frozenset(
    {
        CampaignState.COMPLETED,
        CampaignState.FAILED,
        CampaignState.RECOVERY_REQUIRED,
    }
)

_SAFE_OPERATION_TARGETS = frozenset(
    {
        CampaignState.PAUSED,
        CampaignState.ABORTING,
        CampaignState.RECOVERY_REQUIRED,
    }
)

_TRANSITIONS: dict[CampaignState, frozenset[CampaignState]] = {
    current: (
        frozenset({target})
        | _SAFE_OPERATION_TARGETS
        | (
            frozenset()
            if current in HARDWARE_ACTIVE_STATES
            else frozenset({CampaignState.FAILED})
        )
    )
    for current, target in pairwise(MAIN_CHAIN)
    if current not in TERMINAL_STATES
}
_TRANSITIONS.update(
    {
        CampaignState.ABORTING: frozenset(
            {
                CampaignState.ROLLING_BACK,
                CampaignState.RECOVERY_REQUIRED,
            }
        ),
        CampaignState.ROLLING_BACK: frozenset(
            {CampaignState.FAILED, CampaignState.RECOVERY_REQUIRED}
        ),
    }
)
_TRANSITIONS[CampaignState.I2C_SCREENING] = (
    _TRANSITIONS[CampaignState.I2C_SCREENING]
    | frozenset({CampaignState.COMPLETED})
)


def _coerce_state(value: CampaignState | str, field_name: str) -> CampaignState:
    try:
        return CampaignState(value)
    except (TypeError, ValueError) as exc:
        raise IllegalTransition(f"unknown {field_name} state: {value!r}") from exc


class CampaignStateMachine:
    """Pure, in-memory campaign transition policy with authorization gates."""

    def __init__(
        self,
        initial_state: CampaignState | str = CampaignState.DRAFT,
        *,
        resume_state: CampaignState | str | None = None,
    ) -> None:
        self._state = _coerce_state(initial_state, "initial")
        self._resume_state = (
            None if resume_state is None else _coerce_state(resume_state, "resume")
        )
        if self._state == CampaignState.PAUSED:
            if self._resume_state is None:
                raise IllegalTransition("PAUSED resume state is required")
            if CampaignState.PAUSED not in _TRANSITIONS.get(
                self._resume_state, frozenset()
            ):
                raise IllegalTransition(
                    f"resume state {self._resume_state.value} is not a pausable "
                    "main-chain state"
                )
        elif self._resume_state is not None:
            raise IllegalTransition("resume state is only valid for a PAUSED machine")

    @property
    def state(self) -> CampaignState:
        return self._state

    @property
    def resume_state(self) -> CampaignState | None:
        return self._resume_state

    def transition(
        self,
        current: CampaignState | str,
        target: CampaignState | str,
        *,
        authorization: StageAuthorization | None = None,
        now: datetime | None = None,
        current_campaign_id: str | None = None,
        current_baseline_hash: str | None = None,
    ) -> CampaignState:
        """Validate and apply one state transition without performing any I/O."""

        supplied_current = _coerce_state(current, "current")
        requested_target = _coerce_state(target, "target")
        if supplied_current != self._state:
            raise IllegalTransition(
                "supplied current state "
                f"{supplied_current.value} does not match machine current state "
                f"{self._state.value}"
            )
        if self._state in TERMINAL_STATES:
            raise IllegalTransition(
                f"terminal state {self._state.value} cannot be left"
            )

        self._check_transition_shape(supplied_current, requested_target)
        self._check_authorization(
            requested_target,
            authorization=authorization,
            now=now,
            current_campaign_id=current_campaign_id,
            current_baseline_hash=current_baseline_hash,
        )

        if requested_target == CampaignState.PAUSED:
            self._resume_state = supplied_current
        elif supplied_current == CampaignState.PAUSED:
            self._resume_state = None
        self._state = requested_target
        return self._state

    def _check_transition_shape(
        self, current: CampaignState, target: CampaignState
    ) -> None:
        if current == CampaignState.PAUSED:
            pause_exits = {
                CampaignState.ABORTING,
                CampaignState.RECOVERY_REQUIRED,
            }
            if self._resume_state is not None:
                pause_exits.add(self._resume_state)
            if (
                self._resume_state is not None
                and self._resume_state not in HARDWARE_ACTIVE_STATES
            ):
                pause_exits.add(CampaignState.FAILED)
            if target not in pause_exits:
                if (
                    target == CampaignState.FAILED
                    and self._resume_state in HARDWARE_ACTIVE_STATES
                ):
                    raise IllegalTransition(
                        f"PAUSED hardware-active state {self._resume_state.value} "
                        "cannot enter FAILED before rollback"
                    )
                expected_resume = (
                    "no saved state" if self._resume_state is None else self._resume_state.value
                )
                raise IllegalTransition(
                    f"PAUSED may only resume to {expected_resume} or enter a safe exit"
                )
            return

        allowed = _TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise IllegalTransition(
                f"illegal transition from {current.value} to {target.value}"
            )

    @staticmethod
    def _check_authorization(
        target: CampaignState,
        *,
        authorization: StageAuthorization | None,
        now: datetime | None,
        current_campaign_id: str | None,
        current_baseline_hash: str | None,
    ) -> None:
        if target not in MUTATING_STATES:
            return
        if authorization is None:
            raise AuthorizationError(
                f"authorization is required before entering {target.value}"
            )

        validated = validate_authorization(
            authorization,
            now=now,
            current_campaign_id=current_campaign_id,
            current_baseline_hash=current_baseline_hash,
        )
        if validated.stage != target:
            raise AuthorizationError(
                f"authorization stage {validated.stage.value} does not match "
                f"target {target.value}"
            )
