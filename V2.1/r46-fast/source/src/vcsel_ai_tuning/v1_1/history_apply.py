"""Atomic one-click application of a persisted historical result."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from vcsel_ai_tuning.hardware_authority import (
    HardwareAuthority,
    HardwareAuthorityBusy,
    HardwareAuthorityLease,
)

from .hardware_control import RecoveryRequired
from .repository import ActiveTaskConflict


def _validate_fir(taps: tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(taps)
    if len(result) != 7 or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in result
    ):
        raise ValueError("historical FIR must contain seven integers")
    if result[3] <= 0:
        raise ValueError("historical FIR Main must be positive")
    if sum(abs(value) for value in result) > 170:
        raise ValueError("historical FIR L1 exceeds 170")
    return result


@dataclass(frozen=True, slots=True)
class HistoryHardwareState:
    i2c: Mapping[str, int]
    fir: tuple[int, ...]
    upper_nlc: int
    lower_nlc: int
    pattern: str
    bert_port: str
    bert_channel: str
    dca_fingerprint: str

    def __post_init__(self) -> None:
        i2c = dict(self.i2c)
        if any(
            not isinstance(name, str)
            or not name.strip()
            or isinstance(value, bool)
            or not isinstance(value, int)
            for name, value in i2c.items()
        ):
            raise ValueError("history state I2C must map names to integers")
        object.__setattr__(self, "i2c", i2c)
        object.__setattr__(self, "fir", _validate_fir(self.fir))
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.upper_nlc, self.lower_nlc)
        ):
            raise ValueError("history state NLC values must be integers")
        if self.pattern != "SSPRQ":
            raise ValueError("history state pattern must be SSPRQ")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.bert_port,
                self.bert_channel,
                self.dca_fingerprint,
            )
        ):
            raise ValueError("history state hardware metadata must be nonblank")

    def payload(self) -> dict[str, object]:
        return {
            "i2c": dict(self.i2c),
            "fir": list(self.fir),
            "upper_nlc": self.upper_nlc,
            "lower_nlc": self.lower_nlc,
            "pattern": self.pattern,
            "bert_port": self.bert_port,
            "bert_channel": self.bert_channel,
            "dca_fingerprint": self.dca_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HistoryHardwareState":
        if not isinstance(payload, Mapping):
            raise TypeError("history state payload must be an object")
        expected = {
            "i2c",
            "fir",
            "upper_nlc",
            "lower_nlc",
            "pattern",
            "bert_port",
            "bert_channel",
            "dca_fingerprint",
        }
        if set(payload) != expected:
            raise ValueError("history state payload fields are invalid")
        if not isinstance(payload["i2c"], Mapping):
            raise TypeError("history state I2C must be a JSON object")
        try:
            return cls(
                i2c=dict(payload["i2c"]),
                fir=tuple(payload["fir"]),
                upper_nlc=payload["upper_nlc"],
                lower_nlc=payload["lower_nlc"],
                pattern=payload["pattern"],
                bert_port=payload["bert_port"],
                bert_channel=payload["bert_channel"],
                dca_fingerprint=payload["dca_fingerprint"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("history state payload is malformed") from error


@dataclass(frozen=True, slots=True)
class HistoryResultTarget:
    source_task_id: str
    observation_id: str
    i2c: Mapping[str, int]
    fir: tuple[int, ...]
    upper_nlc: int
    lower_nlc: int
    pattern: str
    bert_port: str
    bert_channel: str
    dca_fingerprint: str

    def __post_init__(self) -> None:
        if not self.source_task_id or not self.observation_id:
            raise ValueError("historical task and observation IDs must be nonblank")
        object.__setattr__(self, "i2c", dict(self.i2c))
        object.__setattr__(self, "fir", _validate_fir(self.fir))
        if self.pattern != "SSPRQ":
            raise ValueError("history target pattern must be SSPRQ")


class HistoryHardware(Protocol):
    def preflight_for_history(self, target: HistoryResultTarget) -> None: ...

    def capture_current_state(self) -> HistoryHardwareState: ...

    def apply_i2c(
        self,
        requested: Mapping[str, int],
        *,
        rollback_values: Mapping[str, int],
    ) -> Mapping[str, int]: ...

    def apply_fir(
        self,
        target: HistoryResultTarget,
        *,
        rollback_state: HistoryHardwareState,
    ) -> tuple[int, ...]: ...

    def measure_once(self) -> Mapping[str, Any]: ...

    def restore_state_once(self, snapshot: HistoryHardwareState) -> None: ...


@dataclass(frozen=True, slots=True)
class HistoryApplyReceipt:
    status: str
    source_task_id: str
    observation_id: str
    temporary_rollback: HistoryHardwareState
    i2c_readback: Mapping[str, int]
    fir_readback: tuple[int, ...]
    measurement: Mapping[str, Any]

    def payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source_task_id": self.source_task_id,
            "observation_id": self.observation_id,
            "temporary_rollback": self.temporary_rollback.payload(),
            "i2c_readback": dict(self.i2c_readback),
            "fir_readback": list(self.fir_readback),
            "measurement": dict(self.measurement),
        }


class HistoryApplyRollbackFailure(RecoveryRequired):
    def __init__(
        self,
        *,
        temporary_rollback: HistoryHardwareState,
        apply_error: Exception,
        rollback_error: Exception,
    ) -> None:
        self.temporary_rollback = temporary_rollback
        self.apply_error = apply_error
        self.rollback_error = rollback_error
        super().__init__(
            "history apply failed and click-time rollback could not be verified: "
            f"apply={apply_error}; rollback={rollback_error}",
            rollback_verified=False,
        )


class HistoryApplyRolledBackFailure(RuntimeError):
    def __init__(
        self,
        *,
        temporary_rollback: HistoryHardwareState,
        apply_error: Exception,
    ) -> None:
        self.temporary_rollback = temporary_rollback
        self.apply_error = apply_error
        super().__init__(str(apply_error))


class HistoryApplyController:
    def __init__(
        self,
        *,
        active_task_id: Callable[[], str | None],
        hardware: HistoryHardware,
        hardware_authority: HardwareAuthority | None = None,
    ) -> None:
        self._active_task_id = active_task_id
        self._hardware = hardware
        self._hardware_authority = hardware_authority or HardwareAuthority()
        self._retained_lease: HardwareAuthorityLease | None = None
        self._retained_recovery_id: str | None = None
        self._state_lock = RLock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("history apply controller is closed")

    @property
    def retained_recovery_id(self) -> str | None:
        with self._state_lock:
            return self._retained_recovery_id

    @property
    def has_unpersisted_recovery(self) -> bool:
        with self._state_lock:
            return (
                self._retained_lease is not None
                and self._retained_recovery_id is None
            )

    def apply(self, target: HistoryResultTarget) -> HistoryApplyReceipt:
        with self._state_lock:
            self._ensure_open()
            if not isinstance(target, HistoryResultTarget):
                raise TypeError("target must be a HistoryResultTarget")
            owner = f"history-apply:{target.source_task_id}:{target.observation_id}"
            try:
                lease = self._hardware_authority.acquire(owner)
            except HardwareAuthorityBusy as error:
                raise ActiveTaskConflict(error.current_owner) from error
            try:
                receipt = self._apply_owned(target)
            except HistoryApplyRollbackFailure:
                self._retained_lease = lease
                raise
            except BaseException:
                lease.release()
                raise
            lease.release()
            return receipt

    def bind_recovery(self, recovery_id: str) -> None:
        if not isinstance(recovery_id, str) or not recovery_id.strip():
            raise ValueError("recovery_id must be nonblank")
        with self._state_lock:
            self._ensure_open()
            if self._retained_lease is None:
                raise RecoveryRequired(
                    "history recovery authority is not retained",
                    rollback_verified=False,
                )
            if self._retained_recovery_id not in {None, recovery_id}:
                raise RecoveryRequired(
                    "retained authority belongs to a different recovery",
                    rollback_verified=False,
                )
            self._retained_recovery_id = recovery_id

    def _apply_owned(self, target: HistoryResultTarget) -> HistoryApplyReceipt:
        active = self._active_task_id()
        if active is not None:
            raise ActiveTaskConflict(active)
        self._hardware.preflight_for_history(target)
        temporary_rollback = self._hardware.capture_current_state()
        if not isinstance(temporary_rollback, HistoryHardwareState):
            raise TypeError("history hardware snapshot is malformed")
        try:
            i2c_readback = self._hardware.apply_i2c(
                target.i2c,
                rollback_values=temporary_rollback.i2c,
            )
            fir_readback = self._hardware.apply_fir(
                target,
                rollback_state=temporary_rollback,
            )
            measurement = self._hardware.measure_once()
        except Exception as apply_error:
            try:
                self._hardware.restore_state_once(temporary_rollback)
            except Exception as rollback_error:
                raise HistoryApplyRollbackFailure(
                    temporary_rollback=temporary_rollback,
                    apply_error=apply_error,
                    rollback_error=rollback_error,
                ) from rollback_error
            raise HistoryApplyRolledBackFailure(
                temporary_rollback=temporary_rollback,
                apply_error=apply_error,
            ) from apply_error
        return HistoryApplyReceipt(
            status="HISTORY_RESULT_APPLIED",
            source_task_id=target.source_task_id,
            observation_id=target.observation_id,
            temporary_rollback=temporary_rollback,
            i2c_readback=dict(i2c_readback),
            fir_readback=tuple(fir_readback),
            measurement=dict(measurement),
        )

    def reserve_recovery(self, recovery_id: str) -> None:
        with self._state_lock:
            self._ensure_open()
            if self._retained_lease is not None:
                if self._retained_recovery_id != recovery_id:
                    raise RecoveryRequired(
                        "retained authority belongs to a different recovery",
                        rollback_verified=False,
                    )
                return
            self._retained_lease = self._hardware_authority.acquire(
                f"history-recovery:{recovery_id}"
            )
            self._retained_recovery_id = recovery_id

    def restore_recovery(
        self, recovery_id: str, snapshot: HistoryHardwareState
    ) -> None:
        with self._state_lock:
            self._ensure_open()
            if self._retained_lease is None:
                raise RecoveryRequired(
                    "history recovery authority is not reserved",
                    rollback_verified=False,
                )
            if self._retained_recovery_id != recovery_id:
                raise RecoveryRequired(
                    "retained authority belongs to a different recovery",
                    rollback_verified=False,
                )
            self._hardware.restore_state_once(snapshot)

    def release_recovery(self, recovery_id: str | None = None) -> None:
        with self._state_lock:
            lease = self._retained_lease
            if lease is None:
                return
            if (
                recovery_id is not None
                and self._retained_recovery_id != recovery_id
            ):
                raise RecoveryRequired(
                    "retained authority belongs to a different recovery",
                    rollback_verified=False,
                )
            self._retained_lease = None
            self._retained_recovery_id = None
            lease.release()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self.release_recovery()
