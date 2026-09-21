"""Connection-lazy Native V1.2 LIVE AI plant composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import sys
from threading import Event
import time
from typing import Any, ContextManager, Protocol
from uuid import uuid4

from vcsel_ai_tuning.hardware.live_config import LiveHardwareConfig
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration

from .ai_execution import V11AIExecution
from .baseline import B0Snapshot
from .dataset import (
    DatasetCaptureError,
    FrozenDatasetCapture,
    MeasurementArtifactBundleWriter,
    MeasurementArtifactContext,
    MeasurementStatus,
)
from .domain import LifecycleState
from .fake_ai_execution import FakeV11AIExecution
from .hardware_control import (
    BertCandidateTransaction,
    I2cCandidateTransaction,
    RecoveryRequired,
)
from .live_ai_policy import (
    LiveAIProfile,
    validate_live_i2c_pilot_payload,
    validate_staged_live_ai_payload,
)
from .live_sweep import (
    LiveSweepDriverBundle,
    _parameter_mapping,
    _validate_profile_targets,
    open_live_sweep_drivers,
)
from .optimizer_bridge import V11OptimizerApi, load_v11_optimizer_api
from .repository import TaskStore
from .staged_ai_execution import PilotSafetyGateEvidence, StagedLiveAIExecution
from .sweep import SweepHardware, SweepMode, SweepPlan


class LiveAIExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, object] | None = None,
        rollback_verified: bool = False,
    ) -> None:
        self.evidence = dict(evidence or {})
        self.rollback_verified = rollback_verified
        super().__init__(message)


class LiveAIDriverFactory(Protocol):
    def __call__(
        self, profile: LiveHardwareConfig, b0: B0Snapshot
    ) -> ContextManager[LiveSweepDriverBundle]: ...


class ReadOnlyBertGuard:
    """Verify the complete frozen BERT state without exposing a write method."""

    def __init__(self, *, bert: Any, b0: B0Snapshot) -> None:
        self._bert = bert
        self._b0 = b0
        self._closed = False

    def _configuration(self) -> BertConfiguration:
        if self._closed:
            raise LiveAIExecutionError("LIVE AI plant is closed")
        return self._bert.get_configuration(
            self._b0.bert_port, self._b0.bert_channel
        )

    def verify_for(
        self, stage: str, *, allow_formula_fir: bool = False
    ) -> Mapping[str, object]:
        configuration = self._configuration()
        mode = self._bert.get_mode()
        raw_mode = mode.strip() if isinstance(mode, str) else ""
        mode_token = raw_mode.split(maxsplit=1)[0].upper() if raw_mode else ""
        patterns = tuple(getattr(configuration, "pattern_fields", ()))
        taps = tuple(configuration.taps)
        formula_ok = bool(
            allow_formula_fir
            and len(taps) == 7
            and all(type(value) is int for value in taps)
            and taps[3] == self._b0.fir_taps[3]
            and sum(abs(value) for value in taps) <= 170
        )
        verified = bool(
            configuration.port == self._b0.bert_port
            and configuration.channel == self._b0.bert_channel
            and (taps == self._b0.fir_taps or formula_ok)
            and configuration.upper_nlc == self._b0.upper_nlc
            and configuration.lower_nlc == self._b0.lower_nlc
            and patterns == (self._b0.bert_pattern, self._b0.bert_pattern)
            and mode_token == "PAM4"
        )
        return {
            "stage": stage,
            "verified": verified,
            "bert_operation": "READ_ONLY",
            "formula_fir_authorized": formula_ok,
            "readback": {
                "port": configuration.port,
                "channel": configuration.channel,
                "taps": list(taps),
                "upper_nlc": configuration.upper_nlc,
                "lower_nlc": configuration.lower_nlc,
                "pattern_fields": list(patterns),
                "mode": raw_mode,
                "mode_token": mode_token,
            },
        }

    def require(self, stage: str, *, allow_formula_fir: bool = False) -> None:
        evidence = self.verify_for(stage, allow_formula_fir=allow_formula_fir)
        if evidence["verified"] is not True:
            raise LiveAIExecutionError(
                f"BERT immutable-state drift detected during {stage}",
                evidence=evidence,
            )

    def close(self) -> None:
        self._closed = True


class LiveI2cOptimizationPlant:
    """PILOT plant with I2C writes and read-only BERT guarding."""

    def __init__(
        self,
        *,
        profile: LiveHardwareConfig,
        task_id: str,
        b0: B0Snapshot,
        bundle: LiveSweepDriverBundle,
        optimizer_api: V11OptimizerApi,
        approved_recovery_values: Mapping[str, frozenset[int]],
        settle_s: float = 0.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(profile, LiveHardwareConfig):
            raise TypeError("profile must be LiveHardwareConfig")
        if not isinstance(b0, B0Snapshot):
            raise TypeError("b0 must be B0Snapshot")
        if bundle.dca is None or bundle.i2c_writer is None or bundle.bert is None:
            raise LiveAIExecutionError("LIVE AI drivers are incomplete")
        _validate_profile_targets(profile, b0)
        self._profile = profile
        self._task_id = task_id
        self._b0 = b0
        self._bundle = bundle
        self._metric_type = optimizer_api.metric_result
        self._fir_type = optimizer_api.fir_target
        b0_i2c = dict(b0.i2c_fields)
        self._approved: dict[str, frozenset[int]] = {}
        for name, raw_values in approved_recovery_values.items():
            values = frozenset(raw_values)
            if (
                name not in b0_i2c
                or not values
                or any(type(value) is not int for value in values)
                or b0_i2c[name] not in values
            ):
                raise ValueError("approved I2C values must include the frozen B0")
            self._approved[name] = values
        self._transaction = I2cCandidateTransaction(
            writer=bundle.i2c_writer,
            mapping=_parameter_mapping(profile, b0),
            channel=b0.i2c_channel,
            b0_values=dict(b0.i2c_fields),
            b0_raw_registers=dict(b0.i2c_raw_registers),
        )
        self._guard = ReadOnlyBertGuard(bert=bundle.bert, b0=b0)
        self._current_fir = self._fir_type(
            taps=b0.fir_taps,
            upper_nlc=b0.upper_nlc,
            lower_nlc=b0.lower_nlc,
            pattern=b0.bert_pattern,
        )
        self._measurement_item_id = "B0"
        self._measurement_round_index = 0
        self._allow_formula_fir = False
        if (
            isinstance(settle_s, bool)
            or not isinstance(settle_s, (int, float))
            or not isfinite(float(settle_s))
            or settle_s < 0
        ):
            raise ValueError("settle_s must be finite and nonnegative")
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self._settle_s = float(settle_s)
        self._sleeper = sleeper
        self._settle_pending = False
        self._last_measurement: object | None = None
        self._restore_attempt_count = 0
        self._restore_verified_count = 0
        self._dca_fault_count = 0
        self._dataset_complete_count = 0
        self._closed = False

    @property
    def b0(self) -> B0Snapshot:
        return self._b0

    @property
    def current_i2c(self) -> Mapping[str, int]:
        return self._transaction.readback()

    @property
    def current_fir(self) -> object:
        return self._current_fir

    def set_measurement_context(self, item_id: str, round_index: int) -> None:
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("measurement item_id must be nonblank")
        if type(round_index) is not int or round_index < 0:
            raise ValueError("measurement round_index must be nonnegative")
        self._measurement_item_id = item_id
        self._measurement_round_index = round_index

    def _set_dca_context(self) -> None:
        setter = getattr(self._bundle.dca, "set_operation_context", None)
        if callable(setter):
            setter(
                task_id=self._task_id,
                item_id=self._measurement_item_id,
                round_index=self._measurement_round_index,
            )

    def preflight(self) -> Mapping[str, object]:
        self._set_dca_context()
        try:
            dca = self._bundle.dca.preflight()
        except Exception:
            self._dca_fault_count += 1
            raise
        if (
            dca.get("identity_verified") is not True
            or dca.get("configuration_verified") is not True
        ):
            raise LiveAIExecutionError(
                "isolated DCA preflight failed", evidence={"dca": dict(dca)}
            )
        self._guard.require("preflight")
        if getattr(self._bundle.i2c_identity, "matched", False) is not True:
            raise LiveAIExecutionError("I2C identity preflight failed")
        current_i2c = self._transaction.readback()
        if current_i2c != dict(self._b0.i2c_fields):
            raise LiveAIExecutionError(
                "LIVE AI preflight requires verified I2C B0",
                evidence={"i2c_readback": current_i2c},
            )
        return {"verified": True, "dca": dict(dca)}

    @staticmethod
    def _strict_request(requested: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(requested, Mapping) or not requested:
            raise ValueError("I2C candidate must be a nonempty mapping")
        result: dict[str, int] = {}
        for name, value in requested.items():
            if not isinstance(name, str) or type(value) is not int:
                raise TypeError("I2C candidate must contain string/integer pairs")
            result[name] = value
        return result

    def apply_i2c(self, requested: Mapping[str, int]) -> Mapping[str, object]:
        values = self._strict_request(requested)
        b0_values = dict(self._b0.i2c_fields)
        if not set(self._approved).issubset(values) or any(
            value not in self._approved[name] for name, value in values.items()
            if name in self._approved
        ) or any(
            name not in self._approved
            and (name not in b0_values or value != b0_values[name])
            for name, value in values.items()
        ) or any(
            name not in values for name in self._approved
        ):
            raise ValueError("I2C candidate is outside the frozen staged ranges")
        self._guard.require(
            "before_i2c_apply",
            allow_formula_fir=self._allow_formula_fir,
        )
        receipt = self._transaction.apply(values)
        full_readback = self._transaction.readback()
        readback = {name: full_readback[name] for name in values}
        if readback != values:
            raise LiveAIExecutionError(
                "I2C requested/applied/readback mismatch",
                evidence={"requested": values, "readback": readback},
            )
        self._settle_pending = True
        return {
            "requested": dict(values),
            "applied": dict(values),
            "readback": readback,
            "receipt": receipt.payload(),
        }

    def apply_fir(self, target: object) -> object:
        del target
        raise LiveAIExecutionError("PILOT_3 cannot write FIR")

    def read_dca_taps(self) -> tuple[object, int]:
        self._set_dca_context()
        try:
            result = self._bundle.dca.measure_single()
        except Exception:
            self._dca_fault_count += 1
            raise
        raw = getattr(result, "raw_responses", {})
        tdecq = raw.get("tdecq", {}) if isinstance(raw, Mapping) else {}
        taps_text = tdecq.get("equalizer_taps") if isinstance(tdecq, Mapping) else None
        main_text = (
            tdecq.get("equalizer_precursors") if isinstance(tdecq, Mapping) else None
        )
        if not isinstance(taps_text, str):
            raise LiveAIExecutionError("DCA Equalizer taps are unavailable")
        try:
            taps = tuple(float(value.strip()) for value in taps_text.split(","))
            main_index = int(str(main_text).strip())
        except (TypeError, ValueError) as error:
            raise LiveAIExecutionError("DCA Equalizer taps are malformed") from error
        return taps, main_index

    def measure(self) -> object:
        if self._settle_pending:
            self._sleeper(self._settle_s)
            self._settle_pending = False
        self._set_dca_context()
        try:
            result = self._bundle.dca.measure_single()
        except Exception:
            self._dca_fault_count += 1
            raise
        self._last_measurement = result
        status = str(getattr(result, "tdecq_status", "INV")).strip().upper()
        value = getattr(result, "tdecq_db", None)
        if (
            status != "CORR"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or abs(float(value)) >= 9.0e37
        ):
            return self._metric_type.invalid(
                status=status or "INV",
                reason=(
                    getattr(result, "tdecq_invalid_reason", None)
                    or "isolated DCA returned invalid TDECQ"
                ),
            )
        er = getattr(result, "er_db", None)
        er_value = (
            float(er)
            if isinstance(er, (int, float))
            and not isinstance(er, bool)
            and isfinite(float(er))
            and abs(float(er)) < 9.0e37
            else None
        )
        return self._metric_type.valid(tdecq_db=float(value), er_db=er_value)

    def take_last_measurement(self) -> object:
        result = self._last_measurement
        self._last_measurement = None
        if result is None:
            raise DatasetCaptureError(
                "no isolated DCA measurement is available for this observation",
                raw_responses={"measurement_available": False},
            )
        return result

    def note_dataset_status(self, status: object) -> None:
        if status == "COMPLETE":
            self._dataset_complete_count += 1

    def _restore_i2c(self) -> dict[str, int]:
        current = self._transaction.readback()
        if current != dict(self._b0.i2c_fields):
            restored = self._transaction.restore_b0()
        else:
            restored = current
        result = dict(restored)
        if result != dict(self._b0.i2c_fields):
            raise RecoveryRequired(
                "I2C B0 restore readback mismatch", rollback_verified=False
            )
        return result

    def restore_b0(self) -> Mapping[str, int]:
        self._restore_attempt_count += 1
        restored = self._restore_i2c()
        if self.is_at_b0() is not True:
            raise RecoveryRequired(
                "PILOT B0 restore verification failed", rollback_verified=False
            )
        self._restore_verified_count += 1
        return restored

    def pilot_gate_evidence(self) -> PilotSafetyGateEvidence:
        identities_unchanged = False
        final_b0_verified = False
        try:
            self._set_dca_context()
            dca = self._bundle.dca.preflight()
            identities_unchanged = bool(
                dca.get("identity_verified") is True
                and dca.get("configuration_verified") is True
                and getattr(self._bundle.i2c_identity, "matched", False) is True
                and self._guard.verify_for("pilot_gate")["verified"] is True
            )
            final_b0_verified = self.is_at_b0() is True
        except Exception:
            self._dca_fault_count += 1
        return PilotSafetyGateEvidence(
            valid_observation_count=3,
            expected_observation_count=3,
            every_restore_verified=(
                self._restore_attempt_count == 3
                and self._restore_verified_count == 3
            ),
            final_b0_verified=final_b0_verified,
            dataset_complete=self._dataset_complete_count == 3,
            identities_unchanged=identities_unchanged,
            dca_fault_count=self._dca_fault_count,
        )

    def is_at_b0(self) -> bool:
        if self._transaction.readback() != dict(self._b0.i2c_fields):
            return False
        return self._guard.verify_for("is_at_b0")["verified"] is True

    def verify(self) -> Mapping[str, object]:
        verified = self.is_at_b0()
        return {"verified": verified, "stage": "final_b0"}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._guard.close()


class LiveAlternatingOptimizationPlant(LiveI2cOptimizationPlant):
    """FAST24 plant adding formula-only, exact-readback FIR writes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        frozen_configuration = BertConfiguration(
            port=self._b0.bert_port,
            channel=self._b0.bert_channel,
            taps=self._b0.fir_taps,
            upper_nlc=self._b0.upper_nlc,
            lower_nlc=self._b0.lower_nlc,
            raw_response="persisted frozen B0",
            pattern_fields=(self._b0.bert_pattern, self._b0.bert_pattern),
        )
        self._bert_transaction = BertCandidateTransaction(
            client=self._bundle.bert,
            port=self._b0.bert_port,
            channel=self._b0.bert_channel,
            pattern=self._b0.bert_pattern,
            b0=frozen_configuration,
        )
        self._allow_formula_fir = True

    @property
    def current_fir(self) -> object:
        configuration = self._bert_transaction.readback()
        return self._fir_type(
            taps=configuration.taps,
            upper_nlc=configuration.upper_nlc,
            lower_nlc=configuration.lower_nlc,
            pattern=self._b0.bert_pattern,
        )

    def apply_fir(self, target: object) -> object:
        if not isinstance(target, self._fir_type):
            raise TypeError("target must be a FirTarget")
        taps = tuple(target.taps)
        if (
            len(taps) != 7
            or any(type(value) is not int for value in taps)
            or taps[3] != self._b0.fir_taps[3]
            or target.upper_nlc != self._b0.upper_nlc
            or target.lower_nlc != self._b0.lower_nlc
            or target.pattern != self._b0.bert_pattern
            or sum(abs(value) for value in taps) > 170
        ):
            if taps[3] != self._b0.fir_taps[3]:
                raise ValueError("FAST24 FIR Main must remain fixed at B0")
            raise ValueError("FAST24 formula FIR is outside the frozen safety contract")
        receipt = self._bert_transaction.apply(taps)
        configuration = self._bert_transaction.readback()
        if tuple(configuration.taps) != taps:
            raise LiveAIExecutionError("BERT FIR exact readback mismatch")
        self._current_fir = target
        self._settle_pending = True
        return receipt.readback_taps

    def restore_b0(self) -> Mapping[str, int]:
        i2c = self._restore_i2c()
        configuration = self._bert_transaction.readback_for_restore()
        if (
            tuple(configuration.taps) != self._b0.fir_taps
            or configuration.upper_nlc != self._b0.upper_nlc
            or configuration.lower_nlc != self._b0.lower_nlc
        ):
            configuration = self._bert_transaction.restore_b0()
        if (
            tuple(configuration.taps) != self._b0.fir_taps
            or configuration.upper_nlc != self._b0.upper_nlc
            or configuration.lower_nlc != self._b0.lower_nlc
        ):
            raise RecoveryRequired(
                "BERT B0 restore readback mismatch", rollback_verified=False
            )
        self._current_fir = self._fir_type(
            taps=self._b0.fir_taps,
            upper_nlc=self._b0.upper_nlc,
            lower_nlc=self._b0.lower_nlc,
            pattern=self._b0.bert_pattern,
        )
        return i2c


class _LiveDatasetWriter:
    """Persist only the frozen capture returned by the isolated DCA worker."""

    def __init__(
        self,
        *,
        task_id: str,
        b0: B0Snapshot,
        artifact_root: Path,
        plant: LiveI2cOptimizationPlant,
    ) -> None:
        self._task_id = task_id
        self._b0 = b0
        self._plant = plant
        self._writer = MeasurementArtifactBundleWriter(artifact_root)

    def record(
        self,
        *,
        observation_id: str,
        sequence: int,
        requested: Mapping[str, int],
        applied: Mapping[str, int],
        readback: Mapping[str, int],
        result: object,
    ) -> dict[str, object]:
        isolated = self._plant.take_last_measurement()
        artifacts = tuple(getattr(isolated, "artifacts", ()) or ())
        captures = tuple(
            item for item in artifacts if isinstance(item, FrozenDatasetCapture)
        )
        capture = captures[0] if len(captures) == 1 else None
        acquisition_id = (
            capture.acquisition_id
            if capture is not None
            else f"acquisition-missing-{uuid4().hex}"
        )
        measurement_id = f"measurement-{uuid4().hex}"
        identities = {
            identity.kind: identity for identity in self._b0.hardware_identities
        }
        isolated_tdecq = getattr(isolated, "tdecq_db", None)
        isolated_er = getattr(isolated, "er_db", None)
        valid = bool(
            getattr(isolated, "acquisition_valid", False) is True
            and str(getattr(isolated, "tdecq_status", "")).strip().upper()
            == "CORR"
            and isinstance(isolated_tdecq, (int, float))
            and not isinstance(isolated_tdecq, bool)
            and isfinite(float(isolated_tdecq))
        )
        context = MeasurementArtifactContext(
            campaign_id=self._task_id,
            sweep_point_id=f"iteration-{sequence:08d}",
            measurement_id=measurement_id,
            acquisition_id=acquisition_id,
            captured_at_utc=datetime.now(timezone.utc),
            requested_params=dict(requested),
            applied_params=dict(applied),
            readback_params=dict(readback),
            tdecq=float(isolated_tdecq) if valid else None,
            outer_er=(
                float(isolated_er)
                if isinstance(isolated_er, (int, float))
                and not isinstance(isolated_er, bool)
                and isfinite(float(isolated_er))
                else None
            ),
            outer_er_source=self._b0.dca_outer_er_source,
            dca_fingerprint=self._b0.dca_fingerprint.payload(),
            dca_identity=identities["dca"].identity,
            bert_port=self._b0.bert_port,
            bert_channel=self._b0.bert_channel,
            bert_pattern=self._b0.bert_pattern,
            measurement_status=(
                MeasurementStatus.SUCCESS if valid else MeasurementStatus.FAILED
            ),
            dca_acquisition_duration_s=float(
                getattr(isolated, "elapsed_s", 0.0)
            ),
            observation_id=observation_id,
        )

        def provide_capture() -> FrozenDatasetCapture:
            if capture is None:
                raise DatasetCaptureError(
                    "isolated DCA did not return exactly one frozen dataset capture",
                    raw_responses={
                        "artifact_count": len(artifacts),
                        "frozen_capture_count": len(captures),
                        "dca_raw_responses": dict(
                            getattr(isolated, "raw_responses", {}) or {}
                        ),
                    },
                )
            return capture

        outcome = self._writer.record(context, provide_capture)
        payload = {
            "measurement_id": measurement_id,
            "acquisition_id": acquisition_id,
            "measurement_status": context.measurement_status.value,
            **outcome.payload(),
        }
        self._plant.note_dataset_status(payload.get("dataset_status"))
        return payload


class _FastPlantView:
    """Adapt the audited LIVE receipt to the optimizer's direct readback API."""

    def __init__(self, plant: LiveAlternatingOptimizationPlant) -> None:
        self._plant = plant
        self._i2c_round = 0
        self._fir_round = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._plant, name)

    @property
    def current_i2c(self) -> Mapping[str, int]:
        return self._plant.current_i2c

    @property
    def current_fir(self) -> object:
        return self._plant.current_fir

    def apply_i2c(self, requested: Mapping[str, int]) -> Mapping[str, int]:
        self._i2c_round += 1
        self._plant.set_measurement_context(
            f"fast-i2c-{self._i2c_round:08d}", self._i2c_round
        )
        receipt = self._plant.apply_i2c(requested)
        readback = receipt.get("readback")
        if not isinstance(readback, Mapping):
            raise LiveAIExecutionError("LIVE I2C receipt has no direct readback")
        return {str(name): int(value) for name, value in readback.items()}

    def apply_fir(self, target: object) -> object:
        self._fir_round += 1
        self._plant.set_measurement_context(
            f"fast-fir-{self._fir_round:08d}", self._fir_round
        )
        return self._plant.apply_fir(target)


class _ExactLiveFast24Execution(FakeV11AIExecution):
    """Use the optimizer primitives while freezing the 6+16+1+1 schedule."""

    def __init__(
        self,
        *args: object,
        live_plant: LiveAlternatingOptimizationPlant,
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            plant_factory=lambda _b0: _FastPlantView(live_plant),
            **kwargs,
        )
        self._live_plant = live_plant
        self._live_dataset_writer = _LiveDatasetWriter(
            task_id=self._task_id,
            b0=self._b0,
            artifact_root=self._artifact_root,
            plant=live_plant,
        )
        self._closed = False

    def _record_dataset(self, **kwargs: object) -> dict[str, object]:
        payload = self._live_dataset_writer.record(**kwargs)
        if payload.get("dataset_status") == "FAILED":
            raise LiveAIExecutionError(
                "FAST24 dataset persistence failed",
                evidence={
                    "observation_id": kwargs.get("observation_id"),
                    "sequence": kwargs.get("sequence"),
                    "dataset": payload,
                },
            )
        return payload

    def run(self, stop_requested: Event) -> LifecycleState | None:
        if not isinstance(stop_requested, Event):
            raise TypeError("stop_requested must be a threading.Event")
        terminal: LifecycleState | None = None
        try:
            with TaskStore(self._database_path) as store:
                existing = tuple(
                    row
                    for row in store.observations(self._task_id)
                    if row.phase != "PILOT_3"
                )
                if existing:
                    raise RuntimeError("FAST24 requires a fresh post-gate stage")
                pilot_rows = tuple(store.observations(self._task_id))
                self._sequence = max(
                    (row.sequence for row in pilot_rows), default=0
                )
                self._alternating = self._api.alternating_controller(self._policy())
                self._measure_b0(store)
                if stop_requested.is_set():
                    return None

                coarse = self._run_coarse(store, stop_requested)
                if len(coarse) != 6:
                    raise RuntimeError("FAST24 did not produce exactly 6 coarse candidates")
                fine_start = sum(row.phase == "I2C_FINE" for row in self._observations)
                self._run_fine(store, stop_requested, budget=16)
                fine_count = sum(row.phase == "I2C_FINE" for row in self._observations)
                if fine_count - fine_start != 16:
                    raise RuntimeError("FAST24 did not produce exactly 16 fine candidates")
                if stop_requested.is_set():
                    return None
                if self._best_metric is None:
                    raise RuntimeError("FAST24 has no valid TDECQ for FIR optimization")

                self._plant.apply_i2c(self._best_i2c)
                controller = self._build_fir_controller(
                    allow_unaccepted_iterative=True,
                )
                self._budget = self._spent + 1
                before = len(self._observations)
                first = self._run_fir(
                    store,
                    stop_requested,
                    controller=controller,
                    mode="FIRST",
                    phase="FIR_FIRST",
                    round_index=0,
                )
                if first is None:
                    return None
                if len(self._observations) != before + 1:
                    raise RuntimeError("FAST24 did not produce exactly 1 first FIR candidate")

                self._budget = self._spent + 1
                before = len(self._observations)
                iterative = self._run_fir(
                    store,
                    stop_requested,
                    controller=controller,
                    mode="ITERATIVE",
                    phase="FIR_ITERATIVE",
                    round_index=1,
                )
                if iterative is None:
                    return None
                if len(self._observations) != before + 1:
                    raise RuntimeError(
                        "FAST24 did not produce exactly 1 iterative FIR candidate"
                    )
                terminal = self._final_verification(store, stop_requested)
                self._terminal_lifecycle = terminal
                return terminal
        finally:
            active_error = sys.exc_info()[1]
            if not isinstance(active_error, RecoveryRequired):
                self._live_plant.restore_b0()
                if self._live_plant.is_at_b0() is not True:
                    raise RecoveryRequired(
                        "FAST24 terminal B0 verification failed",
                        rollback_verified=False,
                    )
                if active_error is not None:
                    evidence: dict[str, object] = {
                        "primary_error": (
                            f"{type(active_error).__name__}: {active_error}"
                        ),
                        "terminal_b0_verified": True,
                    }
                    if (
                        isinstance(active_error, LiveAIExecutionError)
                        and active_error.evidence
                    ):
                        evidence["primary_evidence"] = dict(
                            active_error.evidence
                        )
                    raise LiveAIExecutionError(
                        "FAST24 failed after verified terminal B0 restoration",
                        evidence=evidence,
                        rollback_verified=True,
                    ) from active_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._live_plant.close()


@contextmanager
def open_live_ai_drivers(
    profile: LiveHardwareConfig,
    b0: B0Snapshot,
    *,
    artifact_root: str | Path,
    save_dataset: bool,
    cleanup_grace_s: float = 5.0,
):
    del b0
    plan = SweepPlan(
        hardware=SweepHardware.I2C_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        seed=0,
        max_points=0,
        points=(),
        campaign_id="native-v12-live-ai",
        save_dataset=save_dataset,
    )
    with open_live_sweep_drivers(
        profile,
        plan,
        artifact_root=artifact_root,
        cleanup_grace_s=cleanup_grace_s,
    ) as bundle:
        yield bundle


class LiveI2cAIExecution:
    """Inert wrapper that opens drivers only inside run or restore_b0."""

    def __init__(
        self,
        *,
        task_id: str,
        b0: B0Snapshot,
        payload: Mapping[str, Any],
        database_path: str | Path,
        artifact_root: str | Path,
        profile: LiveHardwareConfig,
        driver_factory: LiveAIDriverFactory,
    ) -> None:
        validate_live_i2c_pilot_payload(payload)
        self._task_id = task_id
        self._b0 = b0
        self._payload = dict(payload)
        self._database_path = Path(database_path)
        self._artifact_root = Path(artifact_root)
        self._profile = profile
        self._driver_factory = driver_factory
        self._terminal_b0_verified = False

    @property
    def terminal_b0_verified(self) -> bool:
        return self._terminal_b0_verified

    def _approved(self) -> dict[str, frozenset[int]]:
        approved = {}
        b0_i2c = dict(self._b0.i2c_fields)
        for parameter in self._payload["parameters"]:
            name = str(parameter["name"])
            values = set(
                range(
                    parameter["min"],
                    parameter["max"] + 1,
                    parameter["step"],
                )
            )
            values.add(b0_i2c[name])
            approved[name] = frozenset(values)
        return approved

    def _plant(self, bundle: LiveSweepDriverBundle) -> LiveI2cOptimizationPlant:
        return LiveI2cOptimizationPlant(
            profile=self._profile,
            task_id=self._task_id,
            b0=self._b0,
            bundle=bundle,
            optimizer_api=load_v11_optimizer_api(),
            approved_recovery_values=self._approved(),
        )

    def run(self, stop_requested: Event):
        with self._driver_factory(self._profile, self._b0) as bundle:
            plant = self._plant(bundle)
            try:
                plant.preflight()
                execution = V11AIExecution(
                    self._task_id,
                    self._b0,
                    self._payload,
                    self._database_path,
                    self._artifact_root,
                    plant=plant,
                    final_guard=plant,
                    observation_metadata_factory=_LiveDatasetWriter(
                        task_id=self._task_id,
                        b0=self._b0,
                        artifact_root=self._artifact_root,
                        plant=plant,
                    ).record,
                    wait=time.sleep,
                )
                result = execution.run(stop_requested)
                if (
                    isinstance(result, LifecycleState)
                    and plant.is_at_b0() is True
                ):
                    self._terminal_b0_verified = True
                return result
            finally:
                plant.close()

    def restore_b0(self) -> None:
        with self._driver_factory(self._profile, self._b0) as bundle:
            plant = self._plant(bundle)
            try:
                plant.restore_b0()
                if plant.is_at_b0() is not True:
                    raise RecoveryRequired(
                        "final B0 verification failed", rollback_verified=False
                    )
            finally:
                plant.close()


class LiveStagedAIExecution:
    """Connection-free prepared Native V1.2 PILOT_3 then FAST24 execution."""

    def __init__(
        self,
        *,
        task_id: str,
        b0: B0Snapshot,
        payload: Mapping[str, Any],
        database_path: str | Path,
        artifact_root: str | Path,
        profile: LiveHardwareConfig,
        driver_factory: LiveAIDriverFactory,
    ) -> None:
        validate_staged_live_ai_payload(payload)
        self._task_id = task_id
        self._b0 = b0
        self._payload = dict(payload)
        self._database_path = Path(database_path)
        self._artifact_root = Path(artifact_root)
        self._profile = profile
        self._driver_factory = driver_factory
        self._terminal_b0_verified = False
        self._closed = False

    @property
    def terminal_b0_verified(self) -> bool:
        return self._terminal_b0_verified

    def run(self, stop_requested: Event):
        if self._closed:
            raise LiveAIExecutionError("staged LIVE AI execution is closed")
        result = self._run_staged(stop_requested)
        if isinstance(result, LifecycleState):
            self._terminal_b0_verified = True
        return result

    def _run_staged(self, stop_requested: Event):
        pilot_payload = {
            "task_type": "AI_OPTIMIZATION",
            "optimization_scope": "I2C_ONLY",
            "depth": "STANDARD",
            "candidate_budget": 3,
            "coarse_candidate_budget": 3,
            "fine_candidate_budget": 0,
            "settle_time_s": 0.1,
            "restore_after_each_candidate": True,
            "save_dataset": True,
            "seed": self._payload.get("seed", 0),
            "parameters": [
                {"name": "ibias_dac", "min": 69, "max": 73, "step": 2},
                {"name": "res_gain", "min": 4, "max": 8, "step": 4},
            ],
        }
        fast_payload = {
            "task_type": "AI_OPTIMIZATION",
            "optimization_scope": "I2C_FIR_ALTERNATING",
            "depth": "FAST",
            "candidate_budget": 24,
            "settle_time_s": 0.1,
            "save_dataset": True,
            "seed": self._payload.get("seed", 0),
            "parameters": [dict(item) for item in self._payload["parameters"]],
        }
        api = load_v11_optimizer_api()
        with self._driver_factory(self._profile, self._b0) as bundle:
            pilot_plant = LiveI2cOptimizationPlant(
                profile=self._profile,
                task_id=self._task_id,
                b0=self._b0,
                bundle=bundle,
                optimizer_api=api,
                approved_recovery_values={
                    "ibias_dac": frozenset(
                        {69, 71, 73, dict(self._b0.i2c_fields)["ibias_dac"]}
                    ),
                    "res_gain": frozenset(
                        {4, 8, dict(self._b0.i2c_fields)["res_gain"]}
                    ),
                },
            )
            pilot_plant.preflight()
            pilot_dataset_writer = _LiveDatasetWriter(
                task_id=self._task_id,
                b0=self._b0,
                artifact_root=self._artifact_root,
                plant=pilot_plant,
            )
            pilot = V11AIExecution(
                self._task_id,
                self._b0,
                pilot_payload,
                self._database_path,
                self._artifact_root,
                plant=pilot_plant,
                final_guard=pilot_plant,
                persisted_config=self._payload,
                observation_metadata_factory=pilot_dataset_writer.record,
                wait=time.sleep,
            )

            def make_fast() -> _ExactLiveFast24Execution:
                plant = LiveAlternatingOptimizationPlant(
                    profile=self._profile,
                    task_id=self._task_id,
                    b0=self._b0,
                    bundle=bundle,
                    optimizer_api=api,
                    approved_recovery_values={
                        "ibias_dac": frozenset(
                            {
                                *range(60, 81),
                                dict(self._b0.i2c_fields)["ibias_dac"],
                            }
                        ),
                        "res_gain": frozenset(
                            {
                                *range(0, 11),
                                dict(self._b0.i2c_fields)["res_gain"],
                            }
                        ),
                    },
                    settle_s=0.1,
                )
                return _ExactLiveFast24Execution(
                    self._task_id,
                    self._b0,
                    fast_payload,
                    self._database_path,
                    self._artifact_root,
                    optimizer_api=api,
                    live_plant=plant,
                    now=lambda: datetime.now(timezone.utc),
                )

            staged = StagedLiveAIExecution(
                self._task_id,
                self._database_path,
                pilot_execution=pilot,
                fast_execution_factory=make_fast,
                gate_evidence_factory=pilot_plant.pilot_gate_evidence,
            )
            try:
                return staged.run(stop_requested)
            except RecoveryRequired:
                raise
            except Exception as error:
                if pilot_plant.is_at_b0() is True:
                    evidence: dict[str, object] = {
                        "primary_error": f"{type(error).__name__}: {error}",
                        "pilot_b0_verified": True,
                    }
                    if (
                        isinstance(error, LiveAIExecutionError)
                        and error.evidence
                    ):
                        evidence["primary_evidence"] = dict(error.evidence)
                    raise LiveAIExecutionError(
                        "staged LIVE AI failed after verified PILOT B0 restoration",
                        evidence=evidence,
                        rollback_verified=True,
                    ) from error
                raise
            finally:
                staged.close()

    def restore_b0(self) -> None:
        if self._closed:
            raise LiveAIExecutionError("staged LIVE AI execution is closed")
        with self._driver_factory(self._profile, self._b0) as bundle:
            plant = LiveAlternatingOptimizationPlant(
                profile=self._profile,
                task_id=self._task_id,
                b0=self._b0,
                bundle=bundle,
                optimizer_api=load_v11_optimizer_api(),
                approved_recovery_values={
                    name: frozenset({value})
                    for name, value in dict(self._b0.i2c_fields).items()
                },
            )
            try:
                plant.restore_b0()
                if plant.is_at_b0() is not True:
                    raise RecoveryRequired(
                        "final B0 verification failed", rollback_verified=False
                    )
            finally:
                plant.close()

    def close(self) -> None:
        self._closed = True


def build_live_ai_execution_factory(
    profile: LiveHardwareConfig,
    *,
    database_path: str | Path,
    artifact_root: str | Path,
    driver_factory: LiveAIDriverFactory | None = None,
    measurement_timeout_s: float | None = None,
    cleanup_grace_s: float = 5.0,
    live_ai_profile: LiveAIProfile | None = None,
    **_unused: object,
) -> Callable[
    [str, B0Snapshot, Mapping[str, Any]],
    LiveI2cAIExecution | LiveStagedAIExecution,
]:
    if not isinstance(profile, LiveHardwareConfig):
        raise TypeError("profile must be LiveHardwareConfig")
    if measurement_timeout_s is not None and (
        isinstance(measurement_timeout_s, bool)
        or not isinstance(measurement_timeout_s, (int, float))
        or not isfinite(float(measurement_timeout_s))
        or measurement_timeout_s <= 0
    ):
        raise ValueError("measurement_timeout_s must be positive and finite")
    if measurement_timeout_s is not None:
        timeout = float(measurement_timeout_s)
        profile = replace(
            profile,
            dca=replace(
                profile.dca,
                timeout_ms=min(profile.dca.timeout_ms, max(1, int(timeout * 1000))),
            ),
            characterization=replace(
                profile.characterization,
                measurement_timeout_s=min(
                    profile.characterization.measurement_timeout_s, timeout
                ),
            ),
        )
    if (
        isinstance(cleanup_grace_s, bool)
        or not isinstance(cleanup_grace_s, (int, float))
        or not isfinite(float(cleanup_grace_s))
        or cleanup_grace_s <= 0
    ):
        raise ValueError("cleanup_grace_s must be positive and finite")
    selected_driver_factory = driver_factory or (
        lambda selected_profile, selected_b0: open_live_ai_drivers(
            selected_profile,
            selected_b0,
            artifact_root=artifact_root,
            save_dataset=True,
            cleanup_grace_s=float(cleanup_grace_s),
        )
    )

    def create(
        task_id: str, b0: B0Snapshot, payload: Mapping[str, Any]
    ) -> LiveI2cAIExecution | LiveStagedAIExecution:
        _validate_profile_targets(profile, b0)
        if live_ai_profile is LiveAIProfile.PILOT_3_THEN_FAST24:
            return LiveStagedAIExecution(
                task_id=task_id,
                b0=b0,
                payload=payload,
                database_path=database_path,
                artifact_root=artifact_root,
                profile=profile,
                driver_factory=selected_driver_factory,
            )
        return LiveI2cAIExecution(
            task_id=task_id,
            b0=b0,
            payload=payload,
            database_path=database_path,
            artifact_root=artifact_root,
            profile=profile,
            driver_factory=selected_driver_factory,
        )

    return create
