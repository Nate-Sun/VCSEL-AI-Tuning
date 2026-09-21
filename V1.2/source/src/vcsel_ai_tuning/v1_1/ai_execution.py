"""Hardware-independent sequential PILOT execution and persistence ordering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event
from typing import Any, Protocol, runtime_checkable

from vcsel_ai_tuning.optimization.v11 import (
    Comparison,
    DiscreteParameterSpace,
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
    OptimizerStatus,
    Pilot3Planner,
    TuningParameter,
    compare_results,
)

from .baseline import B0Snapshot
from .domain import LifecycleState
from .live_ai_policy import validate_live_i2c_pilot_payload
from .repository import BestStateRecord, ObservationRecord, TaskStore


@runtime_checkable
class OptimizationPlant(Protocol):
    @property
    def current_i2c(self) -> Mapping[str, int]: ...

    @property
    def current_fir(self) -> object: ...

    def apply_i2c(self, requested: Mapping[str, int]) -> Mapping[str, object]: ...

    def apply_fir(self, target: object) -> object: ...

    def read_dca_taps(self) -> tuple[object, int]: ...

    def measure(self) -> object: ...

    def restore_b0(self) -> Mapping[str, int] | None: ...

    def is_at_b0(self) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class FinalSafetyGuard(Protocol):
    def verify(self) -> Mapping[str, object]: ...


class DcaObservationInvalidError(RuntimeError):
    """A durable PILOT observation has no valid numeric TDECQ."""

    def __init__(self, observation_id: str, reason: str | None) -> None:
        self.observation_id = observation_id
        self.reason = reason
        super().__init__(
            f"PILOT_3 DCA observation is invalid: {reason or 'unknown reason'}"
        )


class V11AIExecution:
    """Run the three-candidate pilot through an injected hardware-neutral plant."""

    def __init__(
        self,
        task_id: str,
        b0: B0Snapshot,
        config: Mapping[str, object],
        database_path: str | Path,
        artifact_root: str | Path,
        *,
        plant: OptimizationPlant,
        final_guard: FinalSafetyGuard | None = None,
        persisted_config: Mapping[str, object] | None = None,
        observation_metadata_factory: Callable[..., Mapping[str, object]] | None = None,
        wait: Callable[[float], None],
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be nonblank")
        if not isinstance(b0, B0Snapshot):
            raise TypeError("b0 must be a B0Snapshot")
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        if not isinstance(plant, OptimizationPlant):
            raise TypeError("plant must implement OptimizationPlant")
        if final_guard is None or not isinstance(final_guard, FinalSafetyGuard):
            raise ValueError("I2C_ONLY requires an explicit final_guard")
        if not callable(wait) or not callable(now):
            raise TypeError("wait and now must be callable")

        validate_live_i2c_pilot_payload(config)
        parameters = config.get("parameters")
        assert isinstance(parameters, (tuple, list))
        space = DiscreteParameterSpace(
            tuple(
                TuningParameter(
                    name=parameter["name"],
                    tuning_min=parameter["min"],
                    tuning_max=parameter["max"],
                    tuning_step=parameter["step"],
                )
                for parameter in parameters
            )
        )
        seed = config.get("seed", 0)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self._task_id = task_id
        self._b0 = b0
        self._config = dict(config)
        self._database_path = Path(database_path)
        self._artifact_root = Path(artifact_root)
        self._plant = plant
        self._final_guard = final_guard
        self._observation_metadata_factory = observation_metadata_factory
        self._wait = wait
        self._now = now
        self._planner = Pilot3Planner(space, seed=seed)
        self._fingerprint = ExperimentFingerprint(
            fir_taps=b0.fir_taps,
            upper_nlc=b0.upper_nlc,
            lower_nlc=b0.lower_nlc,
            dca_config_fingerprint=b0.dca_fingerprint.content_hash,
        )
        self._history: list[I2CObservation] = []
        self._best_result: MetricResult | None = None
        self._closed = False

        with TaskStore(self._database_path) as store:
            task = store.require_task(task_id)
            persisted_b0 = store.load_b0(task_id)
            expected_persisted_config = (
                config if persisted_config is None else persisted_config
            )
            normalized_config = json.loads(
                json.dumps(
                    dict(expected_persisted_config),
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            if task.payload != normalized_config:
                raise ValueError("execution config differs from persisted task")
            if persisted_b0.content_hash != b0.content_hash:
                raise ValueError("execution B0 differs from persisted task")
            if store.observations(task_id):
                raise ValueError("fresh PILOT execution requires no observations")

    @staticmethod
    def _strict_int_mapping(value: object, label: str) -> dict[str, int]:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{label} must be a mapping")
        result: dict[str, int] = {}
        for key, item in value.items():
            if not isinstance(key, str) or type(item) is not int:
                raise RuntimeError(f"{label} must contain string/integer pairs")
            result[key] = item
        return result

    def _decision_payload(self, decision: object) -> dict[str, object]:
        dumper = getattr(decision, "model_dump", None)
        if not callable(dumper):
            raise RuntimeError("optimizer decision cannot be serialized")
        payload = dumper(mode="json")
        if not isinstance(payload, dict):
            raise RuntimeError("optimizer decision must serialize to an object")
        return payload

    def _restore_once(self) -> None:
        readback = self._plant.restore_b0()
        restored = self._strict_int_mapping(readback, "B0 restore readback")
        expected = dict(self._b0.i2c_fields)
        if restored != expected or self._plant.is_at_b0() is not True:
            raise RuntimeError(
                f"B0 restore readback mismatch: expected {expected}, got {restored}"
            )

    def _current_fir_payload(self) -> tuple[tuple[int, ...], int, int, str]:
        fir = self._plant.current_fir
        taps = tuple(getattr(fir, "taps", ()))
        upper_nlc = getattr(fir, "upper_nlc", None)
        lower_nlc = getattr(fir, "lower_nlc", None)
        pattern = getattr(fir, "pattern", None)
        if (
            len(taps) != 7
            or any(type(value) is not int for value in taps)
            or type(upper_nlc) is not int
            or type(lower_nlc) is not int
            or pattern != "SSPRQ"
        ):
            raise RuntimeError("plant FIR state is malformed")
        return taps, upper_nlc, lower_nlc, pattern

    def _persist_observation(
        self,
        *,
        sequence: int,
        candidate: object,
        decision_payload: Mapping[str, object],
        applied: Mapping[str, int],
        readback: Mapping[str, int],
        result: MetricResult,
    ) -> I2CObservation:
        requested = self._strict_int_mapping(
            getattr(candidate, "requested_i2c", None),
            "optimizer request",
        )
        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            raise RuntimeError("optimizer candidate ID is malformed")
        observation_id = f"{self._task_id}-pilot-{sequence:08d}"
        extra_raw: Mapping[str, object] = {}
        if self._observation_metadata_factory is not None:
            extra_raw = self._observation_metadata_factory(
                observation_id=observation_id,
                sequence=sequence,
                requested=requested,
                applied=applied,
                readback=readback,
                result=result,
            )
            if not isinstance(extra_raw, Mapping):
                raise RuntimeError(
                    "observation metadata factory must return a mapping"
                )
        record = ObservationRecord.measurement(
            observation_id=observation_id,
            sequence=sequence,
            phase="PILOT_3",
            requested=requested,
            applied=dict(applied),
            readback=dict(readback),
            tdecq=result.tdecq_db,
            er=result.er_db,
            raw_response={
                "candidate_id": candidate_id,
                "optimizer_decision": dict(decision_payload),
                "optimizer_readback": dict(readback),
                "metrics": result.model_dump(mode="json"),
                **dict(extra_raw),
            },
            error_reason=result.tdecq_invalid_reason,
            created_at=self._now(),
        )
        observation = I2CObservation(
            observation_id=observation_id,
            candidate_id=candidate_id,
            requested_i2c=requested,
            readback_i2c=dict(readback),
            result=result,
            fingerprint=self._fingerprint,
        )
        with TaskStore(self._database_path) as store:
            store.append_observation(self._task_id, record)
            if result.has_valid_tdecq and (
                self._best_result is None
                or compare_results(result, self._best_result) is Comparison.BETTER
            ):
                taps, upper_nlc, lower_nlc, pattern = self._current_fir_payload()
                store.save_best_state(
                    self._task_id,
                    BestStateRecord(
                        observation_id=observation_id,
                        i2c=dict(readback),
                        fir=taps,
                        upper_nlc=upper_nlc,
                        lower_nlc=lower_nlc,
                        pattern=pattern,
                        bert_port=self._b0.bert_port,
                        bert_channel=self._b0.bert_channel,
                        dca_fingerprint=self._fingerprint.model_dump_json(),
                        result=result.model_dump(mode="json"),
                        updated_at=self._now(),
                    ),
                )
                self._best_result = result
        self._history.append(observation)
        return observation

    def _run_candidate(self, sequence: int) -> I2CObservation:
        decision = self._planner.suggest(tuple(self._history))
        if decision.status is not OptimizerStatus.CANDIDATE:
            raise RuntimeError(f"PILOT_3 planner is not ready: {decision.reason}")
        candidate = decision.candidate
        assert candidate is not None
        decision_payload = self._decision_payload(decision)
        with TaskStore(self._database_path) as store:
            store.append_audit_event(
                self._task_id,
                event_type="AI_PILOT_CANDIDATE_SELECTED",
                payload=decision_payload,
                occurred_at=self._now(),
            )

        attempted = True
        try:
            context_setter = getattr(self._plant, "set_measurement_context", None)
            if callable(context_setter):
                context_setter(f"pilot-{sequence:08d}", sequence)
            receipt = self._plant.apply_i2c(candidate.requested_i2c)
            if not isinstance(receipt, Mapping):
                raise RuntimeError("I2C application receipt must be a mapping")
            applied = self._strict_int_mapping(receipt.get("applied"), "applied")
            readback = self._strict_int_mapping(receipt.get("readback"), "readback")
            requested = dict(candidate.requested_i2c)
            if requested != applied or requested != readback:
                raise RuntimeError(
                    "I2C requested/applied/readback mismatch"
                )
            self._wait(float(self._config["settle_time_s"]))
            measured = self._plant.measure()
            if not isinstance(measured, MetricResult):
                raise RuntimeError("DCA measurement must be a MetricResult")
            observation = self._persist_observation(
                sequence=sequence,
                candidate=candidate,
                decision_payload=decision_payload,
                applied=applied,
                readback=readback,
                result=measured,
            )
        finally:
            if attempted:
                self._restore_once()
        return observation

    def run(self, stop_requested: Event) -> LifecycleState | None:
        if self._closed:
            raise RuntimeError("execution is closed")
        if not isinstance(stop_requested, Event):
            raise TypeError("stop_requested must be an Event")
        for sequence in range(1, 4):
            if stop_requested.is_set():
                return None
            observation = self._run_candidate(sequence)
            if not observation.result.has_valid_tdecq:
                raise DcaObservationInvalidError(
                    observation.observation_id,
                    observation.result.tdecq_invalid_reason,
                )
            if stop_requested.is_set():
                return None

        evidence = self._final_guard.verify()
        if not isinstance(evidence, Mapping) or evidence.get("verified") is not True:
            raise RuntimeError("final safety guard did not verify B0")
        if self._plant.is_at_b0() is not True:
            raise RuntimeError("final safety guard completed away from B0")
        return LifecycleState.COMPLETED

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._plant.close()
