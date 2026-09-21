from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .domain import BackendResult, Candidate, Fidelity, MetricValue, Observation


class BackendFault(RuntimeError):
    pass


class BackendUnavailable(RuntimeError):
    pass


class ExperimentBackend(Protocol):
    def identify(self) -> dict[str, str]: ...

    def configure(self, settings: dict[str, object]) -> None: ...

    def apply(self, candidate: Candidate, applied: dict[str, float]) -> None: ...

    def readback(self) -> dict[str, float]: ...

    def wait_ready(self) -> None: ...

    def measure_fast(self) -> BackendResult: ...

    def measure_slow(self) -> BackendResult | None: ...

    def abort(self) -> None: ...

    def safe_off(self) -> None: ...


class FakeBackend:
    def __init__(
        self,
        fail_at: str | None = None,
        readback_offsets: dict[str, float] | None = None,
    ):
        self.fail_at = fail_at
        self.readback_offsets = readback_offsets or {}
        self.call_order: list[str] = []
        self._applied: dict[str, float] = {}
        self._candidate: Candidate | None = None
        self.is_safe = True

    def _record(self, operation: str) -> None:
        self.call_order.append(operation)
        if self.fail_at == operation:
            raise BackendFault(f"injected fault at {operation}")

    def identify(self) -> dict[str, str]:
        self._record("identify")
        return {"vendor": "Codex", "model": "FakeVCSELBench", "serial": "FAKE-1"}

    def configure(self, settings: dict[str, object]) -> None:
        self._record("configure")

    def apply(self, candidate: Candidate, applied: dict[str, float]) -> None:
        self._record("apply")
        self._candidate = candidate
        self._applied = dict(applied)
        self.is_safe = False

    def readback(self) -> dict[str, float]:
        self._record("readback")
        return {
            name: value + self.readback_offsets.get(name, 0.0)
            for name, value in self._applied.items()
        }

    def wait_ready(self) -> None:
        self._record("wait_ready")

    def measure_fast(self) -> BackendResult:
        self._record("measure_fast")
        if self._candidate is None:
            raise BackendFault("no candidate has been applied")
        skew = abs(self._applied.get("skew_control_ui", 0.0))
        bias = self._applied.get("bias_ma", 0.0)
        modulation = self._applied.get("modulation_ma", 0.0)
        return BackendResult(
            metrics={
                "skew_pp_ui": MetricValue(
                    value=skew,
                    unit="UI",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "rlm_proxy": MetricValue(
                    value=max(0.0, 1.0 - 0.5 * skew),
                    unit="1",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "worst_eye_height_norm": MetricValue(
                    value=max(0.0, 1.0 / 3.0 - 0.2 * skew),
                    unit="1",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "worst_eye_width_proxy_ui": MetricValue(
                    value=max(0.0, 0.8 - skew),
                    unit="UI",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "power_mw": MetricValue(
                    value=1.0 + 0.2 * bias + 0.1 * modulation,
                    unit="mW",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
            },
            raw_artifacts={
                "fake_measurement.txt": (
                    f"candidate={self._candidate.candidate_id}\n".encode("utf-8")
                )
            },
            status={"lock": True, "sync": True},
        )

    def measure_slow(self) -> BackendResult | None:
        self._record("measure_slow")
        return None

    def abort(self) -> None:
        self._record("abort")

    def safe_off(self) -> None:
        self._record("safe_off")
        self.is_safe = True


class ReplayBackend:
    def __init__(self, observations: dict[str, Observation]):
        self._observations = observations
        self._active: Observation | None = None
        self.call_order: list[str] = []
        self.is_safe = True

    @classmethod
    def from_jsonl(cls, path: Path) -> "ReplayBackend":
        observations: dict[str, Observation] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            observation = Observation.model_validate_json(line)
            observations[observation.candidate_id] = observation
        return cls(observations)

    def identify(self) -> dict[str, str]:
        self.call_order.append("identify")
        return {"vendor": "Replay", "model": "ObservationFixture", "serial": "N/A"}

    def configure(self, settings: dict[str, object]) -> None:
        self.call_order.append("configure")

    def apply(self, candidate: Candidate, applied: dict[str, float]) -> None:
        self.call_order.append("apply")
        if candidate.candidate_id not in self._observations:
            raise KeyError(candidate.candidate_id)
        self._active = self._observations[candidate.candidate_id]
        self.is_safe = False

    def readback(self) -> dict[str, float]:
        self.call_order.append("readback")
        if self._active is None:
            raise BackendFault("no replay observation selected")
        return dict(self._active.settings.readback)

    def wait_ready(self) -> None:
        self.call_order.append("wait_ready")

    def measure_fast(self) -> BackendResult:
        self.call_order.append("measure_fast")
        if self._active is None:
            raise BackendFault("no replay observation selected")
        metrics = {
            name: metric
            for name, metric in self._active.metrics.items()
            if metric.fidelity is not Fidelity.HARDWARE_STANDARD
        }
        return BackendResult(metrics=metrics)

    def measure_slow(self) -> BackendResult | None:
        self.call_order.append("measure_slow")
        if self._active is None:
            raise BackendFault("no replay observation selected")
        metrics = {
            name: metric
            for name, metric in self._active.metrics.items()
            if metric.fidelity is Fidelity.HARDWARE_STANDARD
        }
        return BackendResult(metrics=metrics) if metrics else None

    def abort(self) -> None:
        self.call_order.append("abort")

    def safe_off(self) -> None:
        self.call_order.append("safe_off")
        self.is_safe = True
