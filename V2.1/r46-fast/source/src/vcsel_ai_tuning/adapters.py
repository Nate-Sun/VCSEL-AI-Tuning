from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class InstrumentRole(StrEnum):
    BERT = "bert"
    DRIVER = "driver_bias"
    SCOPE_RX = "scope_rx"


class InstrumentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    vendor: str
    model: str
    serial: str
    firmware: str
    role: InstrumentRole


class InstrumentAdapter(Protocol):
    def identify(self) -> InstrumentIdentity: ...

    def configure(self, settings: dict[str, object]) -> None: ...

    def apply(self, settings: dict[str, float]) -> None: ...

    def readback(self) -> dict[str, float]: ...

    def measure(self) -> dict[str, object]: ...

    def abort(self) -> None: ...

    def safe_off(self) -> None: ...


class DryRunInstrumentAdapter:
    def __init__(self, role: InstrumentRole, fixture: dict[str, object]):
        self.role = role
        self.fixture = dict(fixture)
        self._applied: dict[str, float] = {}
        self.is_safe = True
        self.call_order: list[str] = []

    def identify(self) -> InstrumentIdentity:
        self.call_order.append("identify")
        return InstrumentIdentity(
            vendor="DRY-RUN",
            model=f"{self.role.value}-fixture",
            serial="DRY-RUN",
            firmware="fixture-v1",
            role=self.role,
        )

    def configure(self, settings: dict[str, object]) -> None:
        self.call_order.append("configure")

    def apply(self, settings: dict[str, float]) -> None:
        self.call_order.append("apply")
        self._applied = dict(settings)
        self.is_safe = False

    def readback(self) -> dict[str, float]:
        self.call_order.append("readback")
        return dict(self._applied)

    def measure(self) -> dict[str, object]:
        self.call_order.append("measure")
        return dict(self.fixture)

    def abort(self) -> None:
        self.call_order.append("abort")

    def safe_off(self) -> None:
        self.call_order.append("safe_off")
        self.is_safe = True
