"""Deterministic in-memory plant for V1.1 optimizer execution tests.

This module contains no device, transport, command, or acquisition adapter.
The plant only applies values to local state and generates synthetic metrics.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

import numpy as np

from .baseline import B0Snapshot
from .constants import PRODUCT_VERSION, SCHEMA_VERSION
from .dataset import (
    FrozenDatasetCapture,
    MeasurementArtifactBundleWriter,
    MeasurementArtifactContext,
    MeasurementStatus,
    NoOpDatasetRecorder,
    encode_density_png,
)
from .domain import LifecycleState
from .optimizer_bridge import V11OptimizerApi, load_v11_optimizer_api


_OPTIMIZATION_PHASES = frozenset(
    {"I2C_COARSE", "I2C_FINE", "FIR_FIRST", "I2C_RETUNE", "FIR_ITERATIVE"}
)
_LOGGER = logging.getLogger(__name__)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _metric_payload(result: Any) -> dict[str, object]:
    return dict(result.model_dump(mode="json"))


def _repository_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Keep importing the standalone FAKE plant free of hardware import chains."""

    from .repository import BestStateRecord, ObservationRecord, TaskStore

    return BestStateRecord, ObservationRecord, TaskStore


def _write_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FakeOptimizationPlant:
    """Pure FAKE I2C/FIR state with a deterministic metric surface."""

    def __init__(self, b0: B0Snapshot) -> None:
        if not isinstance(b0, B0Snapshot):
            raise TypeError("b0 must be a B0Snapshot")
        optimizer_api = load_v11_optimizer_api()
        self._b0 = b0
        self._fir_target_type = optimizer_api.fir_target
        self._metric_result_type = optimizer_api.metric_result
        self._baseline_i2c = dict(b0.i2c_fields)
        self._baseline_fir = self._fir_target_type(
            taps=b0.fir_taps,
            upper_nlc=b0.upper_nlc,
            lower_nlc=b0.lower_nlc,
            pattern=b0.bert_pattern,
        )
        self._current_i2c = dict(self._baseline_i2c)
        self._current_fir = self._baseline_fir

    @property
    def current_i2c(self) -> dict[str, int]:
        return dict(self._current_i2c)

    @property
    def current_fir(self) -> Any:
        return self._current_fir

    def apply_i2c(
        self,
        requested: Mapping[str, int],
    ) -> dict[str, int]:
        if not isinstance(requested, Mapping):
            raise TypeError("requested I2C values must be a mapping")
        if set(requested) != set(self._baseline_i2c):
            raise ValueError("requested I2C field set must exactly match B0")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in requested.values()
        ):
            raise TypeError("requested I2C values must be integers")
        self._current_i2c = dict(requested)
        return self.current_i2c

    def apply_fir(self, target: Any) -> tuple[int, ...]:
        if not isinstance(target, self._fir_target_type):
            raise TypeError("target must be the bridge FirTarget")
        self._current_fir = target
        return self._current_fir.taps

    def read_dca_taps(self) -> tuple[tuple[float, ...], int]:
        taps = self._current_fir.taps
        norm = max(1, sum(abs(value) for value in taps))
        return tuple(value / norm for value in taps), 3

    def measure(self) -> Any:
        i2c_delta = sum(
            (self._current_i2c[name] - baseline) ** 2
            for name, baseline in self._baseline_i2c.items()
        )
        fir_delta = sum(
            abs(value - baseline)
            for value, baseline in zip(
                self._current_fir.taps,
                self._baseline_fir.taps,
                strict=True,
            )
        )
        return self._metric_result_type.valid(
            tdecq_db=round(2.8 - 0.02 * i2c_delta + 0.002 * fir_delta, 6),
            er_db=round(5.4 + 0.01 * i2c_delta - 0.001 * fir_delta, 6),
        )

    def restore_b0(self) -> None:
        self._current_i2c = dict(self._baseline_i2c)
        self._current_fir = self._baseline_fir

    def is_at_b0(self) -> bool:
        return (
            self._current_i2c == self._baseline_i2c
            and self._current_fir == self._baseline_fir
        )


class FakeV11AIExecution:
    """Persistent, adaptive FAKE execution of the pure V1.1 optimizer graph."""

    def __init__(
        self,
        task_id: str,
        b0: B0Snapshot,
        task_config: Mapping[str, Any],
        database_path: str | Path,
        artifact_root: str | Path,
        *,
        optimizer_api: V11OptimizerApi | None = None,
        optimizer_api_loader: Callable[[], V11OptimizerApi] = load_v11_optimizer_api,
        plant_factory: Callable[[B0Snapshot], FakeOptimizationPlant] = FakeOptimizationPlant,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be nonblank")
        if not isinstance(b0, B0Snapshot):
            raise TypeError("b0 must be a B0Snapshot")
        if not isinstance(task_config, Mapping):
            raise TypeError("task_config must be a mapping")
        self._task_id = task_id
        self._b0 = b0
        self._config = dict(task_config)
        self._database_path = Path(database_path)
        self._artifact_root = Path(artifact_root)
        save_dataset = self._config.get("save_dataset", False)
        if type(save_dataset) is not bool:
            raise ValueError("save_dataset must be a boolean")
        self._save_dataset = save_dataset
        self._dataset_recorder = (
            MeasurementArtifactBundleWriter(self._artifact_root)
            if save_dataset
            else NoOpDatasetRecorder()
        )
        self._api = optimizer_api if optimizer_api is not None else optimizer_api_loader()
        self._plant = plant_factory(b0)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._space = self._build_space()
        self._budget = self._positive_int("candidate_budget")
        self._seed = self._integer("seed", default=0)
        self._observations: list[ObservationRecord] = []
        self._i2c_history: list[Any] = []
        self._best_metric: Any | None = None
        self._best_observation_id: str | None = None
        self._best_i2c = dict(b0.i2c_fields)
        self._best_fir = self._api.fir_target(
            taps=b0.fir_taps,
            upper_nlc=b0.upper_nlc,
            lower_nlc=b0.lower_nlc,
            pattern=b0.bert_pattern,
        )
        self._baseline_metric: Any | None = None
        self._sequence = 0
        self._spent = 0
        self._alternating: Any | None = None
        self._fir_initial_fir: Any | None = None
        self._fir_initial_result: Any | None = None
        self._terminal_lifecycle: LifecycleState | None = None

    @property
    def plant(self) -> FakeOptimizationPlant:
        return self._plant

    def _positive_int(self, name: str) -> int:
        value = self._config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _integer(self, name: str, *, default: int) -> int:
        value = self._config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    def _build_space(self) -> Any:
        raw = self._config.get("parameters")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ValueError("parameters must be a non-empty sequence")
        parameters = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("parameters must contain mappings")
            try:
                parameters.append(
                    self._api.tuning_parameter(
                        name=item["name"],
                        tuning_min=item["min"],
                        tuning_max=item["max"],
                        tuning_step=item["step"],
                        enabled=True,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("persisted AI parameter configuration is invalid") from error
        return self._api.discrete_parameter_space(tuple(parameters))

    def _policy(self) -> Any:
        depth = self._config.get("depth", "STANDARD")
        if depth == "ADVANCED":
            return self._api.alternation_policy.for_mode(
                depth,
                advanced_rounds=self._config.get("advanced_iteration_limit"),
            )
        return self._api.alternation_policy.for_mode(depth)

    def _fingerprint(self, fir: Any | None = None) -> Any:
        current = self._plant.current_fir if fir is None else fir
        return self._api.experiment_fingerprint(
            fir_taps=current.taps,
            upper_nlc=current.upper_nlc,
            lower_nlc=current.lower_nlc,
            dca_config_fingerprint=self._b0.dca_fingerprint.content_hash,
        )

    def _complete_state_payload(
        self,
        *,
        i2c: Mapping[str, int],
        fir: Any,
    ) -> dict[str, object]:
        return {
            "i2c": dict(i2c),
            "fir": {
                "taps": list(fir.taps),
                "upper_nlc": fir.upper_nlc,
                "lower_nlc": fir.lower_nlc,
                "pattern": fir.pattern,
            },
        }

    def _fir_from_payload(self, payload: Mapping[str, object]) -> Any:
        taps = payload.get("taps")
        return self._api.fir_target(
            taps=tuple(taps) if isinstance(taps, list) else taps,
            upper_nlc=payload["upper_nlc"],
            lower_nlc=payload["lower_nlc"],
            pattern=payload.get("pattern", self._b0.bert_pattern),
        )

    def _apply_complete_state(self, payload: Mapping[str, object]) -> None:
        i2c = payload.get("i2c")
        fir = payload.get("fir")
        if not isinstance(i2c, Mapping) or not isinstance(fir, Mapping):
            raise ValueError("persisted complete replay state is malformed")
        target = self._fir_from_payload(fir)
        self._plant.apply_fir(target)
        self._plant.apply_i2c({name: int(value) for name, value in i2c.items()})

    def _metric_from_payload(self, payload: Mapping[str, object]) -> Any:
        return self._api.metric_result.model_validate(dict(payload))

    def _metric_from_record(self, record: ObservationRecord) -> Any:
        if record.valid and record.tdecq is not None:
            return self._api.metric_result.valid(
                tdecq_db=record.tdecq,
                er_db=record.er,
            )
        return self._api.metric_result.invalid(
            status=record.status,
            reason=record.error_reason or "persisted invalid TDECQ",
            er_db=record.er,
        )

    def _optimizer_observation(self, record: ObservationRecord) -> Any:
        raw = record.raw_response if isinstance(record.raw_response, Mapping) else {}
        selected_requested = raw.get("optimizer_requested", record.requested)
        selected_readback = raw.get("optimizer_readback", record.readback)
        fingerprint_payload = raw.get("experiment_fingerprint")
        if isinstance(fingerprint_payload, Mapping):
            canonical_fingerprint = dict(fingerprint_payload)
            serialized_taps = canonical_fingerprint.get("fir_taps")
            if isinstance(serialized_taps, list):
                canonical_fingerprint["fir_taps"] = tuple(serialized_taps)
            fingerprint = self._api.experiment_fingerprint.model_validate(
                canonical_fingerprint
            )
        else:
            fingerprint = self._fingerprint()
        return self._api.i2c_observation(
            observation_id=record.observation_id.replace(":", "."),
            candidate_id=str(raw.get("candidate_id", record.observation_id)).replace(":", "."),
            requested_i2c={name: int(selected_requested[name]) for name in self._space.names},
            readback_i2c={name: int(selected_readback[name]) for name in self._space.names},
            result=self._metric_from_record(record),
            fingerprint=fingerprint,
        )

    def _observed_complete_state(self, record: ObservationRecord) -> tuple[dict[str, int], Any]:
        raw = record.raw_response
        if not isinstance(raw, Mapping):
            raise ValueError("persisted observation replay metadata is missing")
        complete = raw.get("complete_state_observed")
        if not isinstance(complete, Mapping):
            raise ValueError("persisted observed complete state is missing")
        i2c = complete.get("i2c")
        fir = complete.get("fir")
        if not isinstance(i2c, Mapping) or not isinstance(fir, Mapping):
            raise ValueError("persisted observed complete state is malformed")
        return (
            {name: int(value) for name, value in i2c.items()},
            self._fir_from_payload(fir),
        )

    def _best_record(
        self,
        record: ObservationRecord,
        result: Any,
        *,
        i2c: Mapping[str, int],
        fir: Any,
        updated_at: datetime,
    ) -> Any:
        BestStateRecord, _, _ = _repository_types()
        return BestStateRecord(
            observation_id=record.observation_id,
            i2c=dict(i2c),
            fir=fir.taps,
            upper_nlc=fir.upper_nlc,
            lower_nlc=fir.lower_nlc,
            pattern=fir.pattern,
            bert_port=self._b0.bert_port,
            bert_channel=self._b0.bert_channel,
            dca_fingerprint=self._b0.dca_fingerprint.content_hash,
            result=self._raw_result(result),
            updated_at=updated_at,
        )

    @staticmethod
    def _same_best(left: Any, right: Any) -> bool:
        fields = (
            "observation_id",
            "i2c",
            "fir",
            "upper_nlc",
            "lower_nlc",
            "pattern",
            "bert_port",
            "bert_channel",
            "dca_fingerprint",
            "result",
        )
        return all(getattr(left, name) == getattr(right, name) for name in fields)

    def _recompute_and_repair_best(self, store: TaskStore) -> None:
        persisted = store.best_state(self._task_id)
        self._best_metric = None
        self._best_observation_id = None
        self._best_i2c = dict(self._b0.i2c_fields)
        self._best_fir = self._api.fir_target(
            taps=self._b0.fir_taps,
            upper_nlc=self._b0.upper_nlc,
            lower_nlc=self._b0.lower_nlc,
            pattern=self._b0.bert_pattern,
        )
        repaired = None
        for record in self._observations:
            result = self._metric_from_record(record)
            if not result.has_valid_tdecq:
                continue
            if self._best_metric is not None and _enum_value(
                self._api.compare_results(result, self._best_metric)
            ) != "BETTER":
                continue
            i2c, fir = self._observed_complete_state(record)
            self._best_metric = result
            self._best_observation_id = record.observation_id
            self._best_i2c = i2c
            self._best_fir = fir
            repaired = self._best_record(
                record,
                result,
                i2c=i2c,
                fir=fir,
                updated_at=record.created_at,
            )
        if repaired is not None and (
            persisted is None or not self._same_best(persisted, repaired)
        ):
            store.save_best_state(self._task_id, repaired)

    def _load_persisted(self, store: TaskStore) -> LifecycleState | None:
        self._i2c_history.clear()
        self._observations = list(store.observations(self._task_id))
        self._sequence = max((row.sequence for row in self._observations), default=0)
        self._spent = sum(
            row.phase in _OPTIMIZATION_PHASES for row in self._observations
        )
        for row in self._observations:
            if row.phase == "B0":
                self._baseline_metric = self._metric_from_record(row)
            if row.phase in {"I2C_COARSE", "I2C_FINE", "I2C_RETUNE"}:
                self._i2c_history.append(self._optimizer_observation(row))
        self._recompute_and_repair_best(store)
        if self._observations:
            last_raw = self._observations[-1].raw_response
            if isinstance(last_raw, Mapping):
                complete_after = last_raw.get("complete_state_after")
                if isinstance(complete_after, Mapping):
                    self._apply_complete_state(complete_after)
        final = next(
            (row for row in reversed(self._observations) if row.phase == "FINAL_VERIFICATION"),
            None,
        )
        if final is None or self._baseline_metric is None:
            return None
        comparison = _enum_value(
            self._api.compare_results(self._metric_from_record(final), self._baseline_metric)
        )
        return (
            LifecycleState.COMPLETED
            if comparison == "BETTER"
            else LifecycleState.COMPLETED_NO_IMPROVEMENT
        )

    def _raw_result(self, result: Any) -> dict[str, object]:
        return _metric_payload(result)

    def _refresh_artifact(
        self,
        terminal_lifecycle: LifecycleState | None = None,
    ) -> Path:
        """Atomically rebuild the optimization evidence from committed rows."""

        _, _, TaskStore = _repository_types()
        with TaskStore(self._database_path) as store:
            task = store.require_task(self._task_id)
            b0 = store.load_b0(self._task_id)
            observations = store.observations(self._task_id)
            best = store.best_state(self._task_id)
        terminal = terminal_lifecycle
        if terminal is None and task.state in {
            LifecycleState.COMPLETED,
            LifecycleState.COMPLETED_NO_IMPROVEMENT,
        }:
            terminal = task.state
        counts_by_phase = Counter(row.phase for row in observations)
        candidate_budget = task.payload.get("candidate_budget")
        payload: dict[str, object] = {
            "schema": "vcsel_ai_tuning.v1_1.optimization_result",
            "product_version": PRODUCT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "campaign_type": task.task_type.value,
            "runtime_mode": task.runtime_mode.value,
            "task_id": task.task_id,
            "b0_hash": b0.content_hash,
            "configuration_fingerprint": task.config_fingerprint,
            "ordered_observation_ids": [
                row.observation_id for row in observations
            ],
            "observation_count": len(observations),
            "counts_by_phase": dict(counts_by_phase),
            "candidate_budget": candidate_budget,
            "candidates_consumed": sum(
                row.phase in _OPTIMIZATION_PHASES for row in observations
            ),
            "best_state": None if best is None else best.payload(),
            "terminal_lifecycle": None if terminal is None else terminal.value,
            "best_applied_to_hardware": False,
            "restored_to_b0": self._plant.is_at_b0(),
        }
        path = self._artifact_root / self._task_id / "optimization_result.json"
        _write_atomic_json(path, payload)
        return path

    def _safe_refresh_artifact(
        self,
        *,
        stage: str,
        terminal_lifecycle: LifecycleState | None = None,
    ) -> Path | None:
        try:
            return self._refresh_artifact(terminal_lifecycle)
        except Exception as artifact_error:
            _LOGGER.warning(
                "FAKE optimization artifact export failed for task %s at %s: %s",
                self._task_id,
                stage,
                artifact_error,
                exc_info=True,
            )
            _, _, TaskStore = _repository_types()
            try:
                with TaskStore(self._database_path) as store:
                    store.append_audit_event(
                        self._task_id,
                        event_type="AI_ARTIFACT_EXPORT_FAILED",
                        payload={"stage": stage, "error": str(artifact_error)},
                    )
            except Exception:
                _LOGGER.exception(
                    "FAKE optimization artifact failure audit could not be persisted "
                    "for task %s at %s",
                    self._task_id,
                    stage,
                )
            return None

    def _record_combination(
        self,
        observation_id: str,
        result: Any,
        *,
        i2c: Mapping[str, int],
        fir: Any,
    ) -> None:
        assert self._alternating is not None
        self._alternating.record(
            self._api.complete_tuning_combination(
                combination_id=observation_id.replace(":", "."),
                readback_i2c={name: i2c[name] for name in self._space.names},
                fir=fir,
                result=result,
            )
        )

    def _consider_best(
        self,
        store: TaskStore,
        record: ObservationRecord,
        result: Any,
        *,
        i2c: Mapping[str, int],
        fir: Any,
    ) -> None:
        self._record_combination(record.observation_id, result, i2c=i2c, fir=fir)
        if not result.has_valid_tdecq:
            return
        if self._best_metric is not None and _enum_value(
            self._api.compare_results(result, self._best_metric)
        ) != "BETTER":
            return
        BestStateRecord, _, _ = _repository_types()
        best = BestStateRecord(
            observation_id=record.observation_id,
            i2c=dict(i2c),
            fir=fir.taps,
            upper_nlc=fir.upper_nlc,
            lower_nlc=fir.lower_nlc,
            pattern=fir.pattern,
            bert_port=self._b0.bert_port,
            bert_channel=self._b0.bert_channel,
            dca_fingerprint=self._b0.dca_fingerprint.content_hash,
            result=self._raw_result(result),
            updated_at=self._now(),
        )
        # The repository enforces that the referenced observation committed first.
        store.save_best_state(self._task_id, best)
        self._best_metric = result
        self._best_observation_id = record.observation_id
        self._best_i2c = dict(i2c)
        self._best_fir = fir

    def _append(
        self,
        store: TaskStore,
        *,
        phase: str,
        candidate_id: str,
        requested: Mapping[str, int],
        applied: Mapping[str, int],
        readback: Mapping[str, int],
        result: Any,
        raw_response: Mapping[str, object],
        i2c: Mapping[str, int],
        fir: Any,
        plant_state_before: tuple[Mapping[str, int], Any] | None = None,
        plant_state_after: tuple[Mapping[str, int], Any] | None = None,
        counts_budget: bool = True,
    ) -> ObservationRecord:
        _, ObservationRecord, _ = _repository_types()
        self._sequence += 1
        after_i2c, after_fir = (
            (i2c, fir) if plant_state_after is None else plant_state_after
        )
        before_i2c, before_fir = (
            (i2c, fir) if plant_state_before is None else plant_state_before
        )
        observation_id = (
            f"{self._task_id}:{self._sequence:08d}:{phase.lower()}"
        )
        dataset = self._record_dataset(
            observation_id=observation_id,
            sequence=self._sequence,
            requested=requested,
            applied=applied,
            readback=readback,
            result=result,
        )
        persisted_raw = {
            **dict(raw_response),
            "dataset": dataset,
            "complete_state_before": self._complete_state_payload(
                i2c=before_i2c,
                fir=before_fir,
            ),
            "complete_state_observed": self._complete_state_payload(
                i2c=i2c,
                fir=fir,
            ),
            "complete_state_after": self._complete_state_payload(
                i2c=after_i2c,
                fir=after_fir,
            ),
        }
        record = ObservationRecord.measurement(
            observation_id=observation_id,
            sequence=self._sequence,
            phase=phase,
            requested=dict(requested),
            applied=dict(applied),
            readback=dict(readback),
            tdecq=result.tdecq_db,
            er=result.er_db,
            raw_response=persisted_raw,
            error_reason=result.tdecq_invalid_reason,
            created_at=self._now(),
        )
        store.append_observation(self._task_id, record)
        self._observations.append(record)
        if counts_budget:
            self._spent += 1
        self._consider_best(store, record, result, i2c=i2c, fir=fir)
        self._safe_refresh_artifact(stage=phase)
        return record

    def _record_dataset(
        self,
        *,
        observation_id: str,
        sequence: int,
        requested: Mapping[str, int],
        applied: Mapping[str, int],
        readback: Mapping[str, int],
        result: Any,
    ) -> dict[str, object]:
        point_id = f"iteration-{sequence:08d}"
        identity_by_kind = {
            identity.kind: identity for identity in self._b0.hardware_identities
        }
        measurement_id = f"measurement-{uuid4().hex}"
        acquisition_id = f"acquisition-{uuid4().hex}"
        has_valid_tdecq = bool(getattr(result, "has_valid_tdecq", False))
        context = MeasurementArtifactContext(
            campaign_id=self._task_id,
            sweep_point_id=point_id,
            measurement_id=measurement_id,
            acquisition_id=acquisition_id,
            captured_at_utc=self._now(),
            requested_params=dict(requested),
            applied_params=dict(applied),
            readback_params=dict(readback),
            tdecq=getattr(result, "tdecq_db", None),
            outer_er=getattr(result, "er_db", None),
            outer_er_source=self._b0.dca_outer_er_source,
            dca_fingerprint=self._b0.dca_fingerprint.payload(),
            dca_identity=identity_by_kind["dca"].identity,
            bert_port=self._b0.bert_port,
            bert_channel=self._b0.bert_channel,
            bert_pattern=self._b0.bert_pattern,
            measurement_status=(
                MeasurementStatus.SUCCESS
                if has_valid_tdecq
                else MeasurementStatus.FAILED
            ),
            dca_acquisition_duration_s=0.0,
            observation_id=observation_id,
        )

        def capture() -> FrozenDatasetCapture:
            density = (
                np.arange(24, dtype=np.uint32).reshape(4, 6)
                + np.uint32(sequence)
            )
            equalized = np.square(density.astype(np.uint64) + 1)
            taps, main_index = self._plant.read_dca_taps()
            return FrozenDatasetCapture(
                acquisition_id=acquisition_id,
                same_acquisition_verified=True,
                raw_eye_density=density,
                density_axis_order=("amplitude", "time"),
                density_source="CHAN1A",
                tdecq_equalized_eye_png=encode_density_png(equalized),
                rlm=0.97,
                oma=0.91,
                average_power=0.50,
                equalizer_taps=tuple(float(value) for value in taps),
                equalizer_main_index=int(main_index),
                equalizer_source=(
                    f"FUNC{self._b0.dca_tdecq_equalizer_function}"
                ),
                raw_responses={
                    "fake": True,
                    "same_acquisition": True,
                    "result": self._raw_result(result),
                },
            )

        try:
            outcome = self._dataset_recorder.record(context, capture)
            return {
                "measurement_id": measurement_id,
                "acquisition_id": acquisition_id,
                "measurement_status": context.measurement_status.value,
                **outcome.payload(),
            }
        except Exception as error:
            return {
                "measurement_id": measurement_id,
                "acquisition_id": acquisition_id,
                "measurement_status": context.measurement_status.value,
                "dataset_status": "FAILED" if self._save_dataset else "DISABLED",
                "dataset_error": f"dataset recorder failed: {error}",
                "dataset_bundle_path": None,
            }

    def _measure_b0(self, store: TaskStore) -> None:
        if self._baseline_metric is not None:
            return
        dca_taps, dca_main_index = self._plant.read_dca_taps()
        result = self._plant.measure()
        full_i2c = self._plant.current_i2c
        fir = self._plant.current_fir
        self._baseline_metric = result
        self._append(
            store,
            phase="B0",
            candidate_id="b0",
            requested=full_i2c,
            applied=full_i2c,
            readback=full_i2c,
            result=result,
            raw_response={
                "candidate_id": "b0",
                "stage": "B0",
                "raw_measurements": [self._raw_result(result)],
                "aggregate": self._raw_result(result),
                "dca_tdecq_equalizer_taps": list(dca_taps),
                "dca_tdecq_equalizer_main_index": dca_main_index,
                "experiment_fingerprint": self._fingerprint(fir).model_dump(mode="json"),
            },
            i2c=full_i2c,
            fir=fir,
            counts_budget=False,
        )

    def _candidate_capacity(self) -> int:
        return max(0, self._budget - self._spent)

    def _execute_i2c_candidate(
        self,
        store: TaskStore,
        candidate: Any,
        *,
        phase: str,
        round_index: int,
    ) -> Any:
        selected_requested = dict(candidate.requested_i2c)
        before_i2c = self._plant.current_i2c
        before_fir = self._plant.current_fir
        full_requested = dict(before_i2c)
        full_requested.update(selected_requested)
        applied = self._plant.apply_i2c(full_requested)
        result = self._plant.measure()
        selected_readback = {name: applied[name] for name in self._space.names}
        fingerprint = self._fingerprint()
        raw = {
            "candidate_id": candidate.candidate_id,
            "stage": phase,
            "round_index": round_index,
            "model_version": candidate.model_version,
            "budget_index": self._spent + 1,
            "optimizer_requested": selected_requested,
            "optimizer_readback": selected_readback,
            "raw_measurements": [self._raw_result(result)],
            "aggregate": self._raw_result(result),
            "experiment_fingerprint": fingerprint.model_dump(mode="json"),
        }
        record = self._append(
            store,
            phase=phase,
            candidate_id=candidate.candidate_id,
            requested=full_requested,
            applied=applied,
            readback=applied,
            result=result,
            raw_response=raw,
            i2c=applied,
            fir=self._plant.current_fir,
            plant_state_before=(before_i2c, before_fir),
        )
        optimizer_observation = self._api.i2c_observation(
            observation_id=record.observation_id.replace(":", "."),
            candidate_id=candidate.candidate_id,
            requested_i2c=selected_requested,
            readback_i2c=selected_readback,
            result=result,
            fingerprint=fingerprint,
        )
        self._i2c_history.append(optimizer_observation)
        return optimizer_observation

    def _coarse_budget(self) -> int:
        desired = min(8, max(2, self._budget // 4))
        return min(desired, self._space.point_count, self._budget)

    def _run_coarse(self, store: TaskStore, stop_requested: Event) -> tuple[Any, ...]:
        budget = self._coarse_budget()
        planner = self._api.coarse_planner(self._space, budget=max(1, budget), seed=self._seed)
        existing = tuple(row for row in self._i2c_history if row.candidate_id.startswith("i2c-coarse-"))
        executed = {row.candidate_id for row in existing}
        rows = list(existing)
        for candidate in planner.candidates():
            if candidate.candidate_id in executed:
                continue
            if stop_requested.is_set() or len(rows) >= budget or self._candidate_capacity() <= 0:
                break
            rows.append(
                self._execute_i2c_candidate(
                    store, candidate, phase="I2C_COARSE", round_index=0
                )
            )
        return tuple(rows)

    def _fine_budget(self, policy: Any) -> int:
        reserved_alternation = 2 * policy.max_rounds
        return max(
            0,
            self._budget
            - self._coarse_budget()
            - reserved_alternation
        )

    def _run_fine(self, store: TaskStore, stop_requested: Event, *, budget: int) -> bool:
        optimizer = self._api.fine_optimizer(self._space, seed=self._seed)
        completed = sum(row.phase == "I2C_FINE" for row in self._observations)
        for _ in range(completed, budget):
            if stop_requested.is_set() or self._candidate_capacity() <= 0:
                break
            decision = optimizer.suggest(tuple(self._i2c_history))
            if _enum_value(decision.status) != "CANDIDATE" or decision.candidate is None:
                break
            self._execute_i2c_candidate(
                store,
                decision.candidate,
                phase="I2C_FINE",
                round_index=0,
            )
        return any(row.result.has_valid_tdecq for row in self._i2c_history)

    def _authorization_envelope(self) -> Any:
        taps = self._b0.fir_taps
        ranges = tuple(
            (tap, tap) if index == 3 else (-170, 170)
            for index, tap in enumerate(taps)
        )
        return self._api.fir_authorization_envelope(
            tap_ranges=ranges,
            max_step=(340, 340, 340, 1, 340, 340, 340),
            max_total_change_l1=2040,
        )

    def _build_fir_controller(
        self,
        *,
        baseline_fir: Any | None = None,
        baseline_result: Any | None = None,
        expected_dca_tap_count: int | None = None,
        allow_unaccepted_iterative: bool = False,
    ) -> Any:
        from vcsel_ai_tuning.hardware.fir import calculate_iterative_pass

        if expected_dca_tap_count is None:
            dca_taps, _ = self._plant.read_dca_taps()
            expected_dca_tap_count = len(tuple(dca_taps))
        initial_fir = self._plant.current_fir if baseline_fir is None else baseline_fir
        initial_result = self._best_metric if baseline_result is None else baseline_result
        assert initial_result is not None
        self._fir_initial_fir = initial_fir
        self._fir_initial_result = initial_result
        return self._api.fir_controller(
            baseline_fir=initial_fir,
            baseline_result=initial_result,
            authorization=self._authorization_envelope(),
            expected_dca_tap_count=expected_dca_tap_count,
            calculate_iterative_pass=calculate_iterative_pass,
            dca_config_fingerprint=self._b0.dca_fingerprint.content_hash,
            allow_unaccepted_iterative=allow_unaccepted_iterative,
        )

    def _aggregate_measurements(
        self,
        metrics: tuple[Any, ...],
        *,
        candidate_id: str,
        selected_i2c: Mapping[str, int],
        fingerprint: Any,
    ) -> Any:
        rows = tuple(
            self._api.i2c_observation(
                observation_id=(
                    f"{self._task_id}.{self._sequence + 1:08d}.repeat.{index}"
                ),
                candidate_id=candidate_id,
                requested_i2c=dict(selected_i2c),
                readback_i2c=dict(selected_i2c),
                result=result,
                fingerprint=fingerprint,
            )
            for index, result in enumerate(metrics, start=1)
        )
        aggregated = self._api.aggregate_repeated_observations(rows)
        if len(aggregated) != 1:
            raise RuntimeError("bridge aggregate did not return one exact-context group")
        return aggregated[0].result

    def _run_fir(
        self,
        store: TaskStore,
        stop_requested: Event,
        *,
        controller: Any,
        mode: str,
        phase: str,
        round_index: int,
    ) -> bool | None:
        while self._candidate_capacity() > 0:
            if stop_requested.is_set():
                return None
            dca_taps, main_index = self._plant.read_dca_taps()
            convolution_base = controller.best_fir
            decision = (
                controller.propose_first(dca_taps, dca_main_index=main_index)
                if mode == "FIRST"
                else controller.propose_iterative(dca_taps, dca_main_index=main_index)
            )
            status = _enum_value(decision.status)
            candidate = decision.candidate
            if status == "FAILED_INVALID_DCA_TAPS":
                store.append_audit_event(
                    self._task_id,
                    event_type="AI_FIR_INVALID_DCA_TAPS",
                    payload={
                        "stage": phase,
                        "round_index": round_index,
                        "reason": decision.reason,
                        "recommended_fir": list(
                            decision.recommended_fir.taps
                        )
                        if decision.recommended_fir is not None
                        else None,
                    },
                )
                return False
            if status != "CANDIDATE" or candidate is None:
                return False
            requested_fir = candidate.requested_fir
            before_i2c = self._plant.current_i2c
            before_fir = self._plant.current_fir
            readback_taps = self._plant.apply_fir(requested_fir)
            first = self._plant.measure()
            metrics = (first,)
            evaluated = controller.evaluate(candidate, metrics)
            if _enum_value(evaluated.status) == "NEEDS_CONFIRMATION":
                metrics = (first, self._plant.measure(), self._plant.measure())
                evaluated = controller.evaluate(candidate, metrics)
            fingerprint = self._fingerprint(requested_fir)
            selected_i2c = {
                name: self._plant.current_i2c[name] for name in self._space.names
            }
            aggregate = self._aggregate_measurements(
                metrics,
                candidate_id=candidate.candidate_id,
                selected_i2c=selected_i2c,
                fingerprint=fingerprint,
            )
            requested = {
                **{f"tap_{index}": tap for index, tap in enumerate(requested_fir.taps)},
                "upper_nlc": requested_fir.upper_nlc,
                "lower_nlc": requested_fir.lower_nlc,
            }
            raw = {
                "candidate_id": candidate.candidate_id,
                "stage": phase,
                "round_index": round_index,
                "model_version": candidate.model_version,
                "budget_index": self._spent + 1,
                "raw_measurements": [self._raw_result(item) for item in metrics],
                "aggregate": self._raw_result(aggregate),
                "controller_status": _enum_value(evaluated.status),
                "controller_no_improvement_count": evaluated.no_improvement_count,
                "controller_initial_fir": self._complete_state_payload(
                    i2c=self._plant.current_i2c,
                    fir=self._fir_initial_fir,
                )["fir"],
                "controller_initial_result": self._raw_result(
                    self._fir_initial_result
                ),
                "convolution_base_fir": list(convolution_base.taps),
                "dca_taps": list(dca_taps),
                "dca_main_index": main_index,
                "fir_mode": mode,
                "experiment_fingerprint": fingerprint.model_dump(mode="json"),
            }
            recommended = evaluated.recommended_fir
            boundary_fir = (
                requested_fir if evaluated.accepted or recommended is None else recommended
            )
            self._append(
                store,
                phase=phase,
                candidate_id=candidate.candidate_id,
                requested=requested,
                applied=requested,
                readback={
                    **{f"tap_{index}": tap for index, tap in enumerate(readback_taps)},
                    "upper_nlc": requested_fir.upper_nlc,
                    "lower_nlc": requested_fir.lower_nlc,
                },
                result=aggregate,
                raw_response=raw,
                i2c=self._plant.current_i2c,
                fir=requested_fir,
                plant_state_before=(before_i2c, before_fir),
                plant_state_after=(self._plant.current_i2c, boundary_fir),
            )
            if evaluated.accepted:
                return True
            if recommended is not None:
                self._plant.apply_fir(recommended)
            if _enum_value(evaluated.status) == "CONVERGED_NO_IMPROVEMENT":
                return False
            # All rejection retries are governed solely by the controller state.
        return False

    def _run_retune(
        self,
        store: TaskStore,
        stop_requested: Event,
        *,
        fir_controller: Any,
        round_index: int,
    ) -> bool:
        if stop_requested.is_set() or self._candidate_capacity() <= 0:
            return False
        planner = self._api.retune_planner(
            self._space,
            fingerprint=fir_controller.retune_fingerprint,
            current_readback={name: self._plant.current_i2c[name] for name in self._space.names},
            radius_steps=1,
            seed=self._seed + round_index,
        )
        result = planner.suggest(tuple(self._i2c_history))
        decision = result.decision
        if _enum_value(decision.status) != "CANDIDATE" or decision.candidate is None:
            return False
        observation = self._execute_i2c_candidate(
            store,
            decision.candidate,
            phase="I2C_RETUNE",
            round_index=round_index,
        )
        return bool(fir_controller.record_retune_result(observation))

    def _replay_fir_state(self, decision: Any) -> tuple[Any, Any]:
        fir_records = tuple(
            row
            for row in self._observations
            if row.phase in {"FIR_FIRST", "I2C_RETUNE", "FIR_ITERATIVE"}
        )
        first_fir = next(
            (row for row in fir_records if row.phase == "FIR_FIRST"),
            None,
        )
        if first_fir is None or not isinstance(first_fir.raw_response, Mapping):
            raise ValueError("persisted FIR replay metadata is missing")
        first_raw = first_fir.raw_response
        initial_fir_payload = first_raw.get("controller_initial_fir")
        initial_result_payload = first_raw.get("controller_initial_result")
        first_dca = first_raw.get("dca_taps")
        if (
            not isinstance(initial_fir_payload, Mapping)
            or not isinstance(initial_result_payload, Mapping)
            or not isinstance(first_dca, list)
        ):
            raise ValueError("persisted FIR controller checkpoint is malformed")
        controller = self._build_fir_controller(
            baseline_fir=self._fir_from_payload(initial_fir_payload),
            baseline_result=self._metric_from_payload(initial_result_payload),
            expected_dca_tap_count=len(first_dca),
        )
        i2c_by_observation = {
            row.observation_id.replace(":", "."): row
            for row in self._i2c_history
        }
        for record in fir_records:
            if record.phase == "I2C_RETUNE":
                observation = i2c_by_observation[
                    record.observation_id.replace(":", ".")
                ]
                improved = bool(controller.record_retune_result(observation))
                decision = self._alternating.after_retune(
                    round_index=record.raw_response["round_index"],
                    improved=improved,
                )
                continue
            raw = record.raw_response
            if not isinstance(raw, Mapping):
                raise ValueError("persisted FIR replay metadata is malformed")
            dca_taps = raw.get("dca_taps")
            main_index = raw.get("dca_main_index")
            mode = raw.get("fir_mode")
            raw_measurements = raw.get("raw_measurements")
            if (
                not isinstance(dca_taps, list)
                or not isinstance(main_index, int)
                or mode not in {"FIRST", "ITERATIVE"}
                or not isinstance(raw_measurements, list)
            ):
                raise ValueError("persisted FIR replay candidate metadata is malformed")
            proposed = (
                controller.propose_first(dca_taps, dca_main_index=main_index)
                if mode == "FIRST"
                else controller.propose_iterative(dca_taps, dca_main_index=main_index)
            )
            candidate = proposed.candidate
            if candidate is None:
                raise ValueError("persisted FIR replay candidate cannot be reproduced")
            measurements = tuple(
                self._metric_from_payload(item)
                for item in raw_measurements
                if isinstance(item, Mapping)
            )
            if len(measurements) != len(raw_measurements) or not measurements:
                raise ValueError("persisted FIR replay measurements are malformed")
            evaluated = controller.evaluate(candidate, (measurements[0],))
            if len(measurements) == 3:
                evaluated = controller.evaluate(candidate, measurements)
            expected_status = raw.get("controller_status")
            if _enum_value(evaluated.status) != expected_status:
                raise ValueError("persisted FIR replay decision diverged")
            round_index = raw["round_index"]
            if evaluated.accepted or _enum_value(
                evaluated.status
            ) == "CONVERGED_NO_IMPROVEMENT":
                decision = self._alternating.after_fir(
                    round_index=round_index,
                    improved=bool(evaluated.accepted),
                )
        return controller, decision

    def _final_verification(self, store: TaskStore, stop_requested: Event) -> LifecycleState | None:
        if stop_requested.is_set():
            return None
        if self._baseline_metric is None or self._best_metric is None:
            return LifecycleState.COMPLETED_NO_IMPROVEMENT
        previous_i2c = self._plant.current_i2c
        previous_fir = self._plant.current_fir
        self._plant.apply_fir(self._best_fir)
        applied_i2c = self._plant.apply_i2c(self._best_i2c)
        metrics = (
            self._plant.measure(),
            self._plant.measure(),
            self._plant.measure(),
        )
        fingerprint = self._fingerprint(self._best_fir)
        selected = {name: applied_i2c[name] for name in self._space.names}
        aggregate = self._aggregate_measurements(
            metrics,
            candidate_id="final-verification",
            selected_i2c=selected,
            fingerprint=fingerprint,
        )
        self._append(
            store,
            phase="FINAL_VERIFICATION",
            candidate_id="final-verification",
            requested=self._best_i2c,
            applied=applied_i2c,
            readback=applied_i2c,
            result=aggregate,
            raw_response={
                "candidate_id": "final-verification",
                "stage": "FINAL_VERIFICATION",
                "round_index": 0,
                "model_version": "final-verification-v1.1",
                "budget_index": None,
                "raw_measurements": [self._raw_result(item) for item in metrics],
                "aggregate": self._raw_result(aggregate),
                "experiment_fingerprint": fingerprint.model_dump(mode="json"),
            },
            i2c=applied_i2c,
            fir=self._best_fir,
            plant_state_before=(previous_i2c, previous_fir),
            plant_state_after=(previous_i2c, previous_fir),
            counts_budget=False,
        )
        # Verification is temporary. Ordinary run completion does not apply best or B0.
        self._plant.apply_fir(previous_fir)
        self._plant.apply_i2c(previous_i2c)
        comparison = _enum_value(
            self._api.compare_results(aggregate, self._baseline_metric)
        )
        return (
            LifecycleState.COMPLETED
            if comparison == "BETTER"
            else LifecycleState.COMPLETED_NO_IMPROVEMENT
        )

    def run(self, stop_requested: Event) -> LifecycleState | None:
        if not isinstance(stop_requested, Event):
            raise TypeError("stop_requested must be a threading.Event")
        _, _, TaskStore = _repository_types()
        with TaskStore(self._database_path) as store:
            persisted_terminal = self._load_persisted(store)
            if persisted_terminal is not None:
                self._terminal_lifecycle = persisted_terminal
                return persisted_terminal
            policy = self._policy()
            self._alternating = self._api.alternating_controller(policy)
            self._measure_b0(store)
            if stop_requested.is_set():
                return None
            decision = self._alternating.start()
            store.append_audit_event(
                self._task_id,
                event_type="AI_STAGE_ENTERED",
                payload={"stage": _enum_value(decision.next_stage), "round_index": 0},
            )
            coarse_rows = self._run_coarse(store, stop_requested)
            coarse = self._api.coarse_planner(
                self._space,
                budget=max(1, self._coarse_budget()),
                seed=self._seed,
            ).summarize(coarse_rows)
            decision = self._alternating.after_coarse(
                has_valid_tdecq=coarse.best_observation is not None
            )
            if _enum_value(decision.next_stage) == "I2C_FINE":
                fine_valid = self._run_fine(
                    store,
                    stop_requested,
                    budget=self._fine_budget(policy),
                )
                decision = self._alternating.after_fine(has_valid_tdecq=fine_valid)
            if stop_requested.is_set():
                return None
            if _enum_value(decision.next_stage) == "FIR_FIRST" and self._candidate_capacity() > 0:
                has_fir_history = any(
                    row.phase == "FIR_FIRST" for row in self._observations
                )
                if has_fir_history:
                    fir_controller, decision = self._replay_fir_state(decision)
                    if _enum_value(decision.next_stage) == "FIR_FIRST":
                        improved = self._run_fir(
                            store,
                            stop_requested,
                            controller=fir_controller,
                            mode="FIRST",
                            phase="FIR_FIRST",
                            round_index=decision.round_index,
                        )
                        if improved is None:
                            return None
                        decision = self._alternating.after_fir(
                            round_index=decision.round_index,
                            improved=improved,
                        )
                else:
                    self._plant.apply_i2c(self._best_i2c)
                    fir_controller = self._build_fir_controller()
                    improved = self._run_fir(
                        store,
                        stop_requested,
                        controller=fir_controller,
                        mode="FIRST",
                        phase="FIR_FIRST",
                        round_index=0,
                    )
                    if improved is None:
                        return None
                    decision = self._alternating.after_fir(
                        round_index=0,
                        improved=improved,
                    )
                while _enum_value(decision.next_stage) != "COMPLETE":
                    stage = _enum_value(decision.next_stage)
                    if stage == "I2C_RETUNE":
                        retune_improved = self._run_retune(
                            store,
                            stop_requested,
                            fir_controller=fir_controller,
                            round_index=decision.round_index,
                        )
                        decision = self._alternating.after_retune(
                            round_index=decision.round_index,
                            improved=retune_improved,
                        )
                    elif stage == "FIR_ITERATIVE":
                        iterative_improved = self._run_fir(
                            store,
                            stop_requested,
                            controller=fir_controller,
                            mode="ITERATIVE",
                            phase="FIR_ITERATIVE",
                            round_index=decision.round_index,
                        )
                        if iterative_improved is None:
                            return None
                        decision = self._alternating.after_fir(
                            round_index=decision.round_index,
                            improved=iterative_improved,
                        )
                    else:
                        raise RuntimeError(f"optimizer returned unknown stage {stage}")
                    if stop_requested.is_set() or self._candidate_capacity() <= 0:
                        break
            terminal = self._final_verification(store, stop_requested)
            self._terminal_lifecycle = terminal
            return terminal

    def restore_b0(self) -> None:
        self._plant.restore_b0()

    def refresh_artifact(
        self,
        terminal_lifecycle: LifecycleState | None = None,
    ) -> Path | None:
        stage = (
            "RESTORE_B0"
            if terminal_lifecycle is None
            else f"TERMINAL_{terminal_lifecycle.value}"
        )
        return self._safe_refresh_artifact(
            stage=stage,
            terminal_lifecycle=terminal_lifecycle,
        )


def build_fake_ai_execution_factory(
    database_path: str | Path,
    artifact_root: str | Path,
    *,
    optimizer_api_loader: Callable[[], V11OptimizerApi] = load_v11_optimizer_api,
    plant_factory: Callable[[B0Snapshot], FakeOptimizationPlant] = FakeOptimizationPlant,
    now: Callable[[], datetime] | None = None,
) -> Callable[[str, B0Snapshot, Mapping[str, Any]], FakeV11AIExecution]:
    """Validate the bridge now and return the runner-compatible FAKE factory."""

    optimizer_api = optimizer_api_loader()

    def factory(
        task_id: str,
        b0: B0Snapshot,
        task_config: Mapping[str, Any],
    ) -> FakeV11AIExecution:
        return FakeV11AIExecution(
            task_id,
            b0,
            task_config,
            database_path,
            artifact_root,
            optimizer_api=optimizer_api,
            plant_factory=plant_factory,
            now=now,
        )

    return factory
