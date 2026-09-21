"""Fail-closed V1.1 real-hardware SweepExecutor composition.

The module owns no connection at import or construction time beyond the
explicitly injected transactions/adapters.  Production driver opening is kept
in a separate lazy factory so server startup remains connection-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import time
from threading import Lock, RLock, get_ident
from typing import Any, ContextManager, Protocol
from uuid import uuid4

from vcsel_ai_tuning.hardware.live_config import LiveHardwareConfig
from vcsel_ai_tuning.hardware.vit_bert import BertConfiguration

from .baseline import B0Snapshot
from .dataset import (
    DatasetRecorder,
    FrozenDatasetCapture,
    MeasurementArtifactBundleWriter,
    MeasurementArtifactContext,
    MeasurementStatus,
    NoOpDatasetRecorder,
)
from .hardware_control import (
    BertCandidateTransaction,
    I2cCandidateTransaction,
    RecoveryRequired,
)
from .regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from .sweep import FIR_TAP_NAMES, SweepHardware, SweepPlan, SweepPoint


class LiveSweepExecutionError(RuntimeError):
    """Terminal point failure with JSON-safe evidence for persistence."""

    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        self.evidence = dict(evidence)
        super().__init__(message)


class LiveSweepDca(Protocol):
    def preflight(self) -> Mapping[str, object]: ...

    def measure_single(self) -> Any: ...


class _ExclusiveDcaAccess:
    """Reentrant acquisition ownership within this Python process only."""

    def __init__(self) -> None:
        self._claim_lock = RLock()
        self._owner_thread_id: int | None = None
        self._nesting_depth = 0

    @contextmanager
    def claim(self):
        """Claim exclusive in-process DCA acquisition access reentrantly."""

        thread_id = get_ident()
        self._claim_lock.acquire()
        try:
            if self._nesting_depth == 0:
                self._owner_thread_id = thread_id
            elif self._owner_thread_id != thread_id:  # pragma: no cover
                raise RuntimeError("exclusive DCA ownership invariant failed")
            self._nesting_depth += 1
            try:
                yield
            finally:
                self._nesting_depth -= 1
                if self._nesting_depth == 0:
                    self._owner_thread_id = None
        finally:
            self._claim_lock.release()

    def owned_by_current_thread(self) -> bool:
        """Return whether the calling thread currently owns this access."""

        return (
            self._nesting_depth > 0
            and self._owner_thread_id == get_ident()
        )


_EXCLUSIVE_DCA_ACCESS_REGISTRY: dict[str, _ExclusiveDcaAccess] = {}
_EXCLUSIVE_DCA_ACCESS_REGISTRY_LOCK = Lock()


def _exclusive_dca_access_for_resource(resource: str) -> _ExclusiveDcaAccess:
    """Return the in-process coordinator for one canonical DCA resource."""

    canonical_resource = resource.strip().casefold()
    with _EXCLUSIVE_DCA_ACCESS_REGISTRY_LOCK:
        access = _EXCLUSIVE_DCA_ACCESS_REGISTRY.get(canonical_resource)
        if access is None:
            access = _ExclusiveDcaAccess()
            _EXCLUSIVE_DCA_ACCESS_REGISTRY[canonical_resource] = access
        return access


@dataclass(frozen=True, slots=True)
class LiveSweepDriverBundle:
    """Drivers and identity evidence owned by one authorized Sweep run."""

    dca: LiveSweepDca
    i2c_writer: Any
    i2c_identity: Any
    bert: Any
    dataset_capture: (
        Callable[[str, Any], FrozenDatasetCapture] | None
    ) = None
    dca_capture_scope: Callable[[], ContextManager[None]] = nullcontext
    dca_exclusive_verifier: Callable[[], bool] = lambda: False


LiveSweepDriverFactory = Callable[
    [LiveHardwareConfig, SweepPlan],
    ContextManager[LiveSweepDriverBundle],
]


def _bert_state(configuration: Any) -> dict[str, int]:
    taps = tuple(configuration.taps)
    if len(taps) != len(FIR_TAP_NAMES):
        raise RuntimeError("BERT readback did not contain seven FIR taps")
    return dict(zip(FIR_TAP_NAMES, taps, strict=True))


class LiveHardwareSweepExecutor:
    """Execute one authorized sweep plan through guarded hardware transactions."""

    restore_after_each_point = True

    def __init__(
        self,
        *,
        b0: B0Snapshot,
        plan: SweepPlan,
        dca: LiveSweepDca,
        settle_time_s: float,
        i2c_transaction: I2cCandidateTransaction | None = None,
        bert_transaction: BertCandidateTransaction | None = None,
        wait: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        close_callback: Callable[[], None] = lambda: None,
        additional_preflight: Callable[[], Mapping[str, object]] | None = None,
        dataset_recorder: DatasetRecorder | None = None,
        dataset_capture: Callable[[str, Any], FrozenDatasetCapture] | None = None,
        dca_capture_scope: Callable[[], ContextManager[None]] = nullcontext,
        id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        if not isinstance(b0, B0Snapshot):
            raise TypeError("b0 must be a B0Snapshot")
        if not isinstance(plan, SweepPlan):
            raise TypeError("plan must be a SweepPlan")
        if (
            isinstance(settle_time_s, bool)
            or not isinstance(settle_time_s, (int, float))
            or not isfinite(float(settle_time_s))
            or settle_time_s < 0
        ):
            raise ValueError("settle_time_s must be a finite nonnegative number")
        if plan.hardware is SweepHardware.I2C_SWEEP:
            if i2c_transaction is None or bert_transaction is not None:
                raise ValueError("I2C sweep requires only an I2C transaction")
            baseline = dict(b0.i2c_fields)
        elif plan.hardware is SweepHardware.BERT_FIR_SWEEP:
            if bert_transaction is None or i2c_transaction is not None:
                raise ValueError("BERT FIR sweep requires only a BERT transaction")
            baseline = dict(zip(FIR_TAP_NAMES, b0.fir_taps, strict=True))
        else:  # pragma: no cover - enum construction already prevents this
            raise ValueError(f"unsupported sweep hardware {plan.hardware}")

        points_by_index: dict[int, SweepPoint] = {}
        authorized: set[str] = set()
        for point in plan.points:
            if point.original_index in points_by_index:
                raise ValueError("sweep point original indexes must be unique")
            requested = dict(point.requested)
            if set(requested) != set(baseline):
                raise ValueError("sweep point shape must match the frozen B0")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in requested.values()):
                raise TypeError("sweep point values must be integers")
            points_by_index[point.original_index] = point
            authorized.update(
                name for name, value in requested.items() if value != baseline[name]
            )

        self._b0 = b0
        self._plan = plan
        self._dca = dca
        self._settle_time_s = float(settle_time_s)
        self._i2c_transaction = i2c_transaction
        self._bert_transaction = bert_transaction
        self._wait = wait
        self._monotonic = monotonic
        self._now_utc = now_utc
        self._close_callback = close_callback
        self._additional_preflight = additional_preflight
        self._dataset_recorder = dataset_recorder or NoOpDatasetRecorder()
        self._dataset_capture = dataset_capture
        self._dca_capture_scope = dca_capture_scope
        self._id_factory = id_factory
        self._baseline = baseline
        self._authorized_parameters = frozenset(authorized)
        self._points_by_index = points_by_index
        self._pending_point: SweepPoint | None = None
        self._pending_application: dict[str, object] | None = None
        self._preflight_complete = False
        self._closed = False
        self._final_b0_readback: dict[str, int] | None = None
        self._execution_order: list[str] = []

    @staticmethod
    def _point_id(point: SweepPoint) -> str:
        return f"point-{point.original_index:08d}"

    def _readback(self) -> dict[str, int]:
        if self._i2c_transaction is not None:
            return self._i2c_transaction.readback()
        assert self._bert_transaction is not None
        return _bert_state(self._bert_transaction.readback())

    def _set_dca_context(self, item_id: str, round_index: int) -> None:
        setter = getattr(self._dca, "set_operation_context", None)
        if callable(setter):
            setter(
                task_id=self._plan.campaign_id,
                item_id=item_id,
                round_index=round_index,
            )

    def preflight(self) -> None:
        if self._closed:
            raise RuntimeError("live sweep executor is closed")
        self._set_dca_context("B0", 0)
        with self._dca_capture_scope():
            result = self._dca.preflight()
        if (
            result.get("identity_verified") is not True
            or result.get("idn", self._b0.dca_fingerprint.identity)
            != self._b0.dca_fingerprint.identity
        ):
            raise LiveSweepExecutionError(
                "DCA identity preflight failed",
                evidence={"stage": "preflight", "dca": dict(result)},
            )
        if result.get("configuration_verified") is not True:
            raise LiveSweepExecutionError(
                "DCA configuration/source preflight failed",
                evidence={"stage": "preflight", "dca": dict(result)},
            )
        if self._additional_preflight is not None:
            additional = self._additional_preflight()
            if additional.get("identity_verified") is not True:
                raise LiveSweepExecutionError(
                    "I2C/BERT identity or frozen configuration preflight failed",
                    evidence={
                        "stage": "preflight",
                        "dca": dict(result),
                        "additional": dict(additional),
                    },
                )
        self._preflight_complete = True
        self._execution_order.append("preflight")

    def is_at_b0(self) -> bool:
        current = self._readback()
        self._execution_order.append("verify_b0")
        self._final_b0_readback = dict(current)
        return current == self._baseline

    def restore_b0(self) -> None:
        self._execution_order.append("restore_b0")
        try:
            if self._readback() == self._baseline:
                self._final_b0_readback = dict(self._baseline)
                self._pending_point = None
                self._pending_application = None
                return
            if self._i2c_transaction is not None:
                restored = self._i2c_transaction.restore_b0()
            else:
                assert self._bert_transaction is not None
                restored = _bert_state(self._bert_transaction.restore_b0())
        except RecoveryRequired as error:
            evidence = getattr(error, "evidence", None)
            observed = (
                evidence.get("last_observed_readback")
                if isinstance(evidence, Mapping)
                else None
            )
            if isinstance(observed, Mapping):
                if self._i2c_transaction is not None:
                    self._final_b0_readback = {
                        str(name): int(value)
                        for name, value in observed.items()
                    }
                else:
                    taps = observed.get("taps")
                    if isinstance(taps, (list, tuple)) and len(taps) == len(
                        FIR_TAP_NAMES
                    ):
                        normalized = dict(
                            zip(FIR_TAP_NAMES, taps, strict=True)
                        )
                        self._final_b0_readback = {
                            name: int(value)
                            for name, value in normalized.items()
                        }
                        error.evidence = {
                            **dict(evidence),
                            "normalized_readback": dict(
                                self._final_b0_readback
                            ),
                        }
            raise
        except Exception as error:
            raise RecoveryRequired(
                f"sweep B0 restore failed: {error}",
                rollback_verified=False,
            ) from error
        restored_state = dict(restored)
        if restored_state != self._baseline:
            raise RecoveryRequired(
                "sweep B0 restore readback mismatch",
                rollback_verified=False,
            )
        self._final_b0_readback = restored_state
        self._pending_point = None
        self._pending_application = None

    def restore_parameter(self, name: str) -> None:
        if name not in self._authorized_parameters:
            raise ValueError(f"parameter {name} was not authorized for this sweep")
        if self.is_at_b0():
            return
        if self._i2c_transaction is not None:
            self._i2c_transaction.apply({name: self._baseline[name]})
        else:
            assert self._bert_transaction is not None
            current = _bert_state(self._bert_transaction.readback())
            current[name] = self._baseline[name]
            self._bert_transaction.apply(tuple(current[item] for item in FIR_TAP_NAMES))
        if not self.is_at_b0():
            raise RecoveryRequired(
                f"parameter {name} restore did not return to B0",
                rollback_verified=False,
            )

    def apply_point(self, point: SweepPoint) -> Mapping[str, object]:
        if not self._preflight_complete:
            raise RuntimeError("live sweep preflight must complete before apply")
        expected = self._points_by_index.get(point.original_index)
        if expected != point:
            raise ValueError("sweep point is not part of the authorized plan")
        if self._pending_point is not None:
            raise RuntimeError("previous sweep point has not been restored")
        if not self.is_at_b0():
            raise RecoveryRequired(
                "candidate apply requires verified B0",
                rollback_verified=False,
            )
        requested = dict(point.requested)
        changed = {
            name: value
            for name, value in requested.items()
            if value != self._baseline[name]
        }
        if not set(changed).issubset(self._authorized_parameters):
            raise ValueError("candidate attempts to change an unauthorized parameter")
        self._final_b0_readback = None
        write_readback_started = self._monotonic()
        try:
            if self._i2c_transaction is not None:
                receipt = self._i2c_transaction.apply(changed)
                readback = self._i2c_transaction.readback()
                receipt_payload = receipt.payload()
            else:
                assert self._bert_transaction is not None
                taps = tuple(requested[name] for name in FIR_TAP_NAMES)
                receipt = self._bert_transaction.apply(taps)
                readback = _bert_state(self._bert_transaction.readback())
                receipt_payload = receipt.payload()
        except Exception as error:
            evidence = getattr(error, "evidence", None)
            error.evidence = {
                **(dict(evidence) if isinstance(evidence, Mapping) else {}),
                "stage": "apply_readback",
                "point_id": self._point_id(point),
                "requested": requested,
                "write_readback_duration_s": max(
                    0.0, self._monotonic() - write_readback_started
                ),
            }
            raise
        write_readback_duration_s = max(
            0.0, self._monotonic() - write_readback_started
        )
        if readback != requested:
            raise LiveSweepExecutionError(
                "candidate full readback does not match the authorized point",
                evidence={
                    "stage": "readback",
                    "point_id": self._point_id(point),
                    "requested": requested,
                    "readback": readback,
                    "receipt": receipt_payload,
                    "write_readback_duration_s": (
                        write_readback_duration_s
                    ),
                },
            )
        application = {
            "requested": requested,
            "applied": dict(requested),
            "readback": dict(readback),
            "receipt": receipt_payload,
            "write_readback_duration_s": write_readback_duration_s,
        }
        self._pending_point = point
        self._pending_application = application
        self._execution_order.extend(
            (f"{self._point_id(point)}:apply", f"{self._point_id(point)}:readback")
        )
        return application

    def _measurement_payload(self, result: Any, *, point_id: str) -> dict[str, object]:
        raw = dict(getattr(result, "raw_responses", {}) or {})
        tdecq_status = str(getattr(result, "tdecq_status", "INV"))
        er_status = str(getattr(result, "er_status", "INV"))
        evidence: dict[str, object] = {
            "point_id": point_id,
            "tdecq_status": tdecq_status,
            "er_status": er_status,
            "invalid_reason": getattr(result, "tdecq_invalid_reason", None),
            "dca_raw_responses": raw,
        }
        if "SOURCE_MISMATCH" in {tdecq_status, er_status}:
            raise LiveSweepExecutionError(
                "DCA measurement source drift stopped the sweep",
                evidence=evidence,
            )
        return {
            "tdecq": getattr(result, "tdecq_db", None),
            "er": getattr(result, "er_db", None),
            "raw": evidence,
        }

    def _record_dataset(
        self,
        *,
        point: SweepPoint,
        application: Mapping[str, object],
        result: Any,
        measured: Mapping[str, object],
        dca_duration_s: float,
    ) -> dict[str, object]:
        point_id = self._point_id(point)
        measurement_id = self._id_factory()
        worker_captures = tuple(
            artifact
            for artifact in tuple(getattr(result, "artifacts", ()) or ())
            if isinstance(artifact, FrozenDatasetCapture)
        )
        if len(worker_captures) > 1:
            raise RuntimeError("isolated DCA returned multiple Dataset captures")
        worker_capture = worker_captures[0] if worker_captures else None
        acquisition_id = (
            worker_capture.acquisition_id
            if worker_capture is not None
            else self._id_factory()
        )
        tdecq = measured.get("tdecq")
        measurement_status = (
            MeasurementStatus.SUCCESS
            if isinstance(tdecq, (int, float))
            and not isinstance(tdecq, bool)
            and isfinite(float(tdecq))
            else MeasurementStatus.FAILED
        )
        identities = _identity_by_kind(self._b0)
        context = MeasurementArtifactContext(
            campaign_id=self._plan.campaign_id,
            sweep_point_id=point_id,
            measurement_id=measurement_id,
            acquisition_id=acquisition_id,
            captured_at_utc=self._now_utc(),
            requested_params=dict(application.get("requested", point.requested)),
            applied_params=dict(application.get("applied", {})),
            readback_params=dict(application.get("readback", {})),
            tdecq=(None if tdecq is None else float(tdecq)),
            outer_er=(
                None
                if measured.get("er") is None
                else float(measured["er"])
            ),
            outer_er_source=self._b0.dca_outer_er_source,
            dca_fingerprint=self._b0.dca_fingerprint.payload(),
            dca_identity=identities["dca"].identity,
            bert_port=self._b0.bert_port,
            bert_channel=self._b0.bert_channel,
            bert_pattern=self._b0.bert_pattern,
            measurement_status=measurement_status,
            dca_acquisition_duration_s=dca_duration_s,
        )

        def capture() -> FrozenDatasetCapture:
            if worker_capture is not None:
                return worker_capture
            if self._dataset_capture is None:
                raise RuntimeError(
                    "audited LIVE dataset capture provider is unavailable"
                )
            return self._dataset_capture(acquisition_id, result)

        try:
            outcome = self._dataset_recorder.record(context, capture)
        except Exception as error:
            return {
                "measurement_id": measurement_id,
                "acquisition_id": acquisition_id,
                "measurement_status": measurement_status.value,
                "dataset_status": "FAILED",
                "dataset_error": f"dataset recorder failed: {error}",
                "dataset_bundle_path": None,
            }
        return {
            "measurement_id": measurement_id,
            "acquisition_id": acquisition_id,
            "measurement_status": measurement_status.value,
            "dataset_status": outcome.status.value,
            "dataset_error": outcome.error,
            "dataset_bundle_path": outcome.bundle_path,
            "dataset": outcome.payload(),
        }

    def measure_b0(self) -> Mapping[str, object]:
        if not self._preflight_complete:
            raise RuntimeError("live sweep preflight must complete before B0 measurement")
        if not self.is_at_b0():
            raise RecoveryRequired(
                "B0 measurement requires verified B0",
                rollback_verified=False,
            )
        self._set_dca_context("B0", 0)
        with self._dca_capture_scope():
            result = self._dca.measure_single()
            measured = self._measurement_payload(result, point_id="B0")
        measured.update(
            requested=dict(self._baseline),
            applied=dict(self._baseline),
            readback=dict(self._baseline),
        )
        raw = dict(measured["raw"])
        raw["execution_order"] = "B0"
        measured["raw"] = raw
        self._execution_order.append("B0:measure")
        return measured

    def measure_point(self) -> Mapping[str, object]:
        point = self._pending_point
        application = self._pending_application
        if point is None or application is None:
            raise RuntimeError("no applied sweep point is awaiting measurement")
        settle_started = self._monotonic()
        self._wait(self._settle_time_s)
        actual_settle_s = max(0.0, self._monotonic() - settle_started)
        point_id = self._point_id(point)
        self._execution_order.append(f"{point_id}:settle")
        self._set_dca_context(point_id, point.execution_index)
        with self._dca_capture_scope():
            dca_started = self._monotonic()
            try:
                result = self._dca.measure_single()
            except Exception as error:
                dca_duration_s = max(0.0, self._monotonic() - dca_started)
                watchdog = getattr(error, "evidence", None)
                raise LiveSweepExecutionError(
                    f"DCA measurement failed: {error}",
                    evidence={
                        "stage": "measurement",
                        "point_id": point_id,
                        "application": application,
                        "settle_duration_s": actual_settle_s,
                        "dca_duration_s": dca_duration_s,
                        "error": str(error),
                        "watchdog": (
                            dict(watchdog)
                            if isinstance(watchdog, Mapping)
                            else None
                        ),
                    },
                ) from error
            dca_duration_s = max(0.0, self._monotonic() - dca_started)
            try:
                measured = self._measurement_payload(result, point_id=point_id)
            except LiveSweepExecutionError as error:
                error.evidence.update(
                    application=application,
                    settle_duration_s=actual_settle_s,
                    dca_duration_s=dca_duration_s,
                )
                raise
            measured.update(
                settle_duration_s=actual_settle_s,
                dca_duration_s=dca_duration_s,
            )
            dataset = self._record_dataset(
                point=point,
                application=application,
                result=result,
                measured=measured,
                dca_duration_s=dca_duration_s,
            )
        measured.update(
            (name, value)
            for name, value in dataset.items()
            if name != "dataset"
        )
        raw = dict(measured["raw"])
        raw.update(
            application=application,
            execution_order=point.execution_index,
            requested_settle_s=self._settle_time_s,
            actual_settle_s=actual_settle_s,
            dca_duration_s=dca_duration_s,
            dataset=dataset.get("dataset", dataset),
        )
        measured["raw"] = raw
        self._execution_order.append(f"{point_id}:measure")
        return measured

    def audit_evidence(self) -> dict[str, object]:
        return {
            "hardware": self._plan.hardware.value,
            "authorized_parameters": sorted(self._authorized_parameters),
            "execution_order": list(self._execution_order),
            "final_b0_readback": (
                None
                if self._final_b0_readback is None
                else dict(self._final_b0_readback)
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_callback()


def _identity_by_kind(b0: B0Snapshot) -> dict[str, Any]:
    return {identity.kind: identity for identity in b0.hardware_identities}


def _validate_profile_targets(
    profile: LiveHardwareConfig,
    b0: B0Snapshot,
) -> None:
    identities = _identity_by_kind(b0)
    expected = {
        "dca": profile.dca.resource,
        "i2c": (
            f"{profile.i2c.port}/{profile.i2c.baudrate}/"
            f"0x{profile.i2c.device_address:02X}"
        ),
        "bert": (
            f"{profile.bert.ip}:{profile.bert.udp_port}/"
            f"{profile.bert.port}/{profile.bert.channel}"
        ),
    }
    mismatches = [
        f"{kind}: expected {identity.resource}, configured {expected[kind]}"
        for kind, identity in identities.items()
        if kind in expected and identity.resource != expected[kind]
    ]
    if set(identities) != set(expected):
        mismatches.append("B0 does not contain exactly DCA, I2C, and BERT identities")
    if b0.i2c_channel != profile.characterization.default_channel:
        mismatches.append("I2C channel does not match the frozen B0")
    if (
        b0.bert_port != profile.bert.port
        or b0.bert_channel != profile.bert.channel
    ):
        mismatches.append("BERT port/channel does not match the frozen B0")
    if (
        b0.dca_outer_er_source != profile.characterization.er.source
        or b0.dca_tdecq_source != profile.characterization.tdecq.source
    ):
        mismatches.append("DCA measurement sources do not match the frozen B0")
    if mismatches:
        raise ValueError("live sweep profile/B0 mismatch: " + "; ".join(mismatches))


def _parameter_mapping(
    profile: LiveHardwareConfig,
    b0: B0Snapshot,
):
    fields = profile.characterization.parameters
    values = dict(b0.i2c_fields)
    if set(values) != {field.name for field in fields}:
        raise ValueError("B0 I2C fields do not match the reviewed driver profile")
    return scan_parameter_mapping(
        tuple(
            ScriptEntry(field.name, values[field.name], order)
            for order, field in enumerate(fields)
        ),
        tuple(
            RegMapField(
                name=field.name,
                channel0_address=field.channel0_address,
                channel_stride=0x100,
                bit_start=field.bit_start,
                bits=field.bits,
                encoding_min=(
                    0 if field.safety_min is None else field.safety_min
                ),
                encoding_max=(
                    field.encoded_max
                    if field.safety_max is None
                    else field.safety_max
                ),
            )
            for field in fields
        ),
    )


@contextmanager
def open_live_sweep_drivers(
    profile: LiveHardwareConfig,
    plan: SweepPlan,
    *,
    artifact_root: str | Path = Path("artifacts") / "v1_2",
    cleanup_grace_s: float = 5.0,
):
    """Open production drivers only when an authorized task requests an executor."""

    # Imports remain inside the authorized operation so application startup is
    # connection-free and does not import optional VISA/serial hardware stacks.
    from vcsel_ai_tuning.hardware.live_dca_isolation import IsolatedLiveDcaClient
    from vcsel_ai_tuning.hardware.vd1124 import (
        Vd1124Probe,
        Vd1124RegisterWriter,
    )
    from vcsel_ai_tuning.hardware.vit_bert import VitBertClient

    dca_access = _exclusive_dca_access_for_resource(profile.dca.resource)

    with ExitStack() as stack:
        isolated_dca = IsolatedLiveDcaClient(
            profile=profile,
            artifact_root=artifact_root,
            timeout_s=profile.characterization.measurement_timeout_s,
            cleanup_grace_s=cleanup_grace_s,
            capture_dataset=plan.save_dataset,
        )
        probe = stack.enter_context(
            Vd1124Probe(
                port=profile.i2c.port,
                baudrate=profile.i2c.baudrate,
                address=profile.i2c.device_address,
                chip_id_register=profile.i2c.chip_id_register,
                expected_chip_id=profile.i2c.expected_chip_id,
                chip_revision_register=profile.i2c.chip_revision_register,
                expected_chip_revision=profile.i2c.expected_chip_revision,
            )
        )
        i2c_identity = probe.probe()
        probe.close()
        i2c_writer = None
        if plan.hardware is SweepHardware.I2C_SWEEP:
            i2c_writer = stack.enter_context(
                Vd1124RegisterWriter(
                    port=profile.i2c.port,
                    baudrate=profile.i2c.baudrate,
                    address=profile.i2c.device_address,
                )
            )
        bert = VitBertClient(
            profile.bert.ip,
            profile.bert.udp_port,
            local_ip=profile.bert.local_ip,
            timeout_s=profile.bert.timeout_s,
        )
        yield LiveSweepDriverBundle(
            dca=isolated_dca,
            i2c_writer=i2c_writer,
            i2c_identity=i2c_identity,
            bert=bert,
            dataset_capture=None,
            dca_capture_scope=dca_access.claim,
            dca_exclusive_verifier=dca_access.owned_by_current_thread,
        )


def build_live_sweep_executor_factory(
    profile: LiveHardwareConfig,
    *,
    driver_factory: LiveSweepDriverFactory | None = None,
    measurement_timeout_s: float | None = None,
    artifact_root: str | Path | None = None,
) -> Callable[[B0Snapshot, SweepPlan], LiveHardwareSweepExecutor]:
    """Create a startup-safe factory for production V1.1 Sweep execution."""

    if not isinstance(profile, LiveHardwareConfig):
        raise TypeError("profile must be LiveHardwareConfig")
    if measurement_timeout_s is not None:
        if (
            isinstance(measurement_timeout_s, bool)
            or not isinstance(measurement_timeout_s, (int, float))
            or not isfinite(float(measurement_timeout_s))
            or measurement_timeout_s <= 0
        ):
            raise ValueError("measurement_timeout_s must be positive and finite")
        profile = replace(
            profile,
            dca=replace(
                profile.dca,
                timeout_ms=min(
                    profile.dca.timeout_ms,
                    max(1, int(float(measurement_timeout_s) * 1000.0)),
                ),
            ),
            characterization=replace(
                profile.characterization,
                measurement_timeout_s=min(
                    profile.characterization.measurement_timeout_s,
                    float(measurement_timeout_s),
                ),
            ),
        )
    selected_driver_factory = driver_factory or (
        lambda selected_profile, selected_plan: open_live_sweep_drivers(
            selected_profile,
            selected_plan,
            artifact_root=(
                Path("artifacts") / "v1_2"
                if artifact_root is None
                else artifact_root
            ),
        )
    )

    def create(
        b0: B0Snapshot,
        plan: SweepPlan,
    ) -> LiveHardwareSweepExecutor:
        _validate_profile_targets(profile, b0)
        if plan.save_dataset and artifact_root is None:
            raise RuntimeError(
                "save_dataset=true requires an explicit LIVE artifact root"
            )
        context = selected_driver_factory(profile, plan)
        bundle = context.__enter__()
        try:
            i2c_transaction = None
            bert_transaction = None
            if plan.hardware is SweepHardware.I2C_SWEEP:
                if bundle.i2c_writer is None:
                    raise RuntimeError("live I2C sweep driver was not opened")
                i2c_transaction = I2cCandidateTransaction(
                    writer=bundle.i2c_writer,
                    mapping=_parameter_mapping(profile, b0),
                    channel=b0.i2c_channel,
                    b0_values=dict(b0.i2c_fields),
                    b0_raw_registers=dict(b0.i2c_raw_registers),
                )
            else:
                frozen_bert = BertConfiguration(
                    port=b0.bert_port,
                    channel=b0.bert_channel,
                    taps=b0.fir_taps,
                    upper_nlc=b0.upper_nlc,
                    lower_nlc=b0.lower_nlc,
                    raw_response="B0Snapshot",
                    pattern_fields=(b0.bert_pattern, b0.bert_pattern),
                )
                bert_transaction = BertCandidateTransaction(
                    client=bundle.bert,
                    port=b0.bert_port,
                    channel=b0.bert_channel,
                    pattern=b0.bert_pattern,
                    b0=frozen_bert,
                )

            def verify_non_dca_hardware() -> Mapping[str, object]:
                configuration = bundle.bert.get_configuration(
                    b0.bert_port,
                    b0.bert_channel,
                )
                mode = bundle.bert.get_mode()
                identities = _identity_by_kind(b0)
                actual_i2c_identity = (
                    f"chip_id=0x{bundle.i2c_identity.chip_id:02X};"
                    f"revision=0x{bundle.i2c_identity.chip_revision:02X}"
                )
                actual_bert_identity = (
                    f"mode={mode};patterns={configuration.pattern_fields[0]},"
                    f"{configuration.pattern_fields[1]}"
                )
                identity_verified = bool(
                    bundle.i2c_identity.matched
                    and identities["i2c"].identity == actual_i2c_identity
                    and identities["bert"].identity == actual_bert_identity
                    and configuration.port == b0.bert_port
                    and configuration.channel == b0.bert_channel
                    and configuration.taps == b0.fir_taps
                    and configuration.upper_nlc == b0.upper_nlc
                    and configuration.lower_nlc == b0.lower_nlc
                    and tuple(configuration.pattern_fields)
                    == (b0.bert_pattern, b0.bert_pattern)
                )
                return {
                    "identity_verified": identity_verified,
                    "i2c": {
                        "chip_id": bundle.i2c_identity.chip_id,
                        "chip_revision": bundle.i2c_identity.chip_revision,
                        "matched": bundle.i2c_identity.matched,
                    },
                    "bert": {
                        "mode": mode,
                        "port": configuration.port,
                        "channel": configuration.channel,
                        "taps": list(configuration.taps),
                        "upper_nlc": configuration.upper_nlc,
                        "lower_nlc": configuration.lower_nlc,
                        "pattern_fields": list(configuration.pattern_fields),
                        "raw_response": configuration.raw_response,
                    },
                }

            return LiveHardwareSweepExecutor(
                b0=b0,
                plan=plan,
                dca=bundle.dca,
                settle_time_s=plan.settle_time_s,
                i2c_transaction=i2c_transaction,
                bert_transaction=bert_transaction,
                additional_preflight=verify_non_dca_hardware,
                dataset_recorder=(
                    MeasurementArtifactBundleWriter(artifact_root)
                    if plan.save_dataset
                    else NoOpDatasetRecorder()
                ),
                dataset_capture=bundle.dataset_capture,
                dca_capture_scope=bundle.dca_capture_scope,
                close_callback=lambda: context.__exit__(None, None, None),
            )
        except BaseException:
            context.__exit__(None, None, None)
            raise

    return create
