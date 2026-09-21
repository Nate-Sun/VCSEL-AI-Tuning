"""Hardware-neutral V1.1 task vocabulary and discrete parameter rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskType(StrEnum):
    AI_OPTIMIZATION = "AI_OPTIMIZATION"
    SWEEP = "SWEEP"


class OptimizationScope(StrEnum):
    I2C_FIR_ALTERNATING = "I2C_FIR_ALTERNATING"
    I2C_ONLY = "I2C_ONLY"


class RuntimeMode(StrEnum):
    FAKE = "FAKE"
    READ_ONLY = "READ_ONLY"
    REPLAY = "REPLAY"
    LIVE = "LIVE"


class LifecycleState(StrEnum):
    DRAFT = "DRAFT"
    PREFLIGHTED = "PREFLIGHTED"
    B0_FROZEN = "B0_FROZEN"
    AUTHORIZED = "AUTHORIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ABORTING = "ABORTING"
    ROLLING_BACK = "ROLLING_BACK"
    COMPLETED = "COMPLETED"
    COMPLETED_NO_IMPROVEMENT = "COMPLETED_NO_IMPROVEMENT"
    ABORTED = "ABORTED"
    ABORTED_RECOVERED = "ABORTED_RECOVERED"
    FAILED = "FAILED"
    FAILED_INVALID_DCA_TAPS = "FAILED_INVALID_DCA_TAPS"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

    @property
    def is_nonterminal(self) -> bool:
        return self not in {
            LifecycleState.COMPLETED,
            LifecycleState.COMPLETED_NO_IMPROVEMENT,
            LifecycleState.ABORTED,
            LifecycleState.ABORTED_RECOVERED,
            LifecycleState.FAILED,
            LifecycleState.FAILED_INVALID_DCA_TAPS,
        }


@dataclass(frozen=True, slots=True)
class DiscreteRange:
    """Inclusive raw-code grid without implicit clipping or max insertion."""

    minimum: int
    maximum: int
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise TypeError("minimum must be an integer")
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, int):
            raise TypeError("maximum must be an integer")
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError("step must be an integer")
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        if self.step <= 0:
            raise ValueError("step must be positive")

    def values(self) -> tuple[int, ...]:
        count = (self.maximum - self.minimum) // self.step
        return tuple(self.minimum + index * self.step for index in range(count + 1))

    def require_value(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("candidate value must be an integer")
        if value < self.minimum or value > self.maximum:
            raise ValueError("candidate value is outside the configured range")
        if (value - self.minimum) % self.step:
            raise ValueError("candidate value is not aligned to the configured grid")
        return value
