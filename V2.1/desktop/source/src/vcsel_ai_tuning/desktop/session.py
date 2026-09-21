"""Desktop PRE-FLIGHT, explicit approval, background run, and safe stop."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Iterator, Protocol
from uuid import uuid4

import yaml

from vcsel_ai_tuning.v2.continuous.factory import build_continuous_dependencies
from vcsel_ai_tuning.v2.continuous.models import (
    ContinuousState,
    ExecutionMode,
    HumanDecision,
    RoundRecord,
)

from .models import (
    DesktopPhase,
    DesktopResult,
    PreflightSummary,
    ProgressSnapshot,
    RunInputs,
)
from .session_config import SessionConfigBuilder, SessionPaths


class DesktopSessionError(RuntimeError):
    """The requested desktop state transition is not allowed."""


class EngineAdapter(Protocol):
    def preflight(self, inputs: RunInputs) -> PreflightSummary: ...

    def commission(self, inputs: RunInputs) -> DesktopResult: ...

    def request_stop(self) -> None: ...

    def completed_rounds(self) -> int: ...


@contextmanager
def _temporary_api_key(inputs: RunInputs) -> Iterator[None]:
    name = "MOONSHOT_API_KEY"
    previous = os.environ.get(name)
    os.environ[name] = inputs.api_key.get_secret_value()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class DesktopSession:
    def __init__(self, engine: EngineAdapter) -> None:
        self._engine = engine
        self._lock = Lock()
        self._phase = DesktopPhase.IDLE
        self._pending_inputs: RunInputs | None = None
        self._max_rounds = 1
        self._worker: Thread | None = None
        self._result: DesktopResult | None = None
        self._error: Exception | None = None

    @property
    def phase(self) -> DesktopPhase:
        with self._lock:
            return self._phase

    def run_preflight(self, inputs: RunInputs) -> PreflightSummary:
        with self._lock:
            if self._phase not in {DesktopPhase.IDLE, DesktopPhase.FAILED}:
                raise DesktopSessionError("PRE-FLIGHT requires an idle session")
            self._phase = DesktopPhase.PREFLIGHT
            self._pending_inputs = None
            self._worker = None
            self._result = None
            self._error = None
            self._max_rounds = inputs.max_rounds
        try:
            with _temporary_api_key(inputs):
                summary = self._engine.preflight(inputs)
        except Exception:
            with self._lock:
                self._phase = DesktopPhase.FAILED
                self._pending_inputs = None
            raise
        with self._lock:
            if summary.approvable:
                self._pending_inputs = inputs
                self._phase = DesktopPhase.AWAITING_APPROVAL
            else:
                self._pending_inputs = None
                self._phase = DesktopPhase.FAILED
        return summary

    def approve_and_start(self) -> None:
        with self._lock:
            if (
                self._phase is not DesktopPhase.AWAITING_APPROVAL
                or self._pending_inputs is None
            ):
                raise DesktopSessionError(
                    "commission requires successful PRE-FLIGHT approval"
                )
            inputs = self._pending_inputs
            self._phase = DesktopPhase.RUNNING
            self._worker = Thread(
                target=self._run_commission,
                args=(inputs,),
                name="vcsel-desktop-commission",
                daemon=False,
            )
            worker = self._worker
        worker.start()

    def cancel_approval(self) -> None:
        with self._lock:
            if self._phase is not DesktopPhase.AWAITING_APPROVAL:
                raise DesktopSessionError("only pending approval can be cancelled")
            self._pending_inputs = None
            self._phase = DesktopPhase.IDLE

    def _run_commission(self, inputs: RunInputs) -> None:
        try:
            with _temporary_api_key(inputs):
                result = self._engine.commission(inputs)
        except Exception as error:
            with self._lock:
                self._error = error
                self._phase = DesktopPhase.FAILED
                self._pending_inputs = None
            return
        with self._lock:
            self._result = result
            self._phase = result.phase
            self._pending_inputs = None

    def request_stop(self) -> None:
        with self._lock:
            if self._phase not in {DesktopPhase.RUNNING, DesktopPhase.STOPPING}:
                raise DesktopSessionError("safe stop requires a running session")
        self._engine.request_stop()
        with self._lock:
            if self._phase is DesktopPhase.RUNNING:
                self._phase = DesktopPhase.STOPPING

    def progress(self) -> ProgressSnapshot:
        current = self._engine.completed_rounds()
        with self._lock:
            phase = self._phase
            maximum = self._max_rounds
        return ProgressSnapshot(
            phase=phase,
            current_round=current,
            max_rounds=maximum,
        )

    def wait(self, timeout: float | None = None) -> DesktopResult:
        with self._lock:
            worker = self._worker
        if worker is None:
            raise DesktopSessionError("commission has not started")
        worker.join(timeout)
        if worker.is_alive():
            raise TimeoutError("desktop commission is still running")
        with self._lock:
            error = self._error
            result = self._result
        if error is not None:
            raise error
        if result is None:
            raise DesktopSessionError("commission ended without a result")
        return result

    def export_csv(self, output_path: str | Path) -> Path:
        with self._lock:
            result = self._result
        if result is None:
            raise DesktopSessionError("a terminal result is required before export")
        from .artifacts import CampaignArtifactReader
        from .csv_export import export_campaign_csv

        projection = CampaignArtifactReader(result.campaign_root).read()
        return export_campaign_csv(projection, output_path)


class ContinuousLiveEngineAdapter:
    """The sole desktop seam into the validated continuous LIVE engine."""

    def __init__(
        self,
        config_builder: SessionConfigBuilder,
        output_root: str | Path,
        campaign_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._builder = config_builder
        self._output_root = Path(output_root)
        self._campaign_id_factory = campaign_id_factory or self._new_campaign_id
        self._paths: SessionPaths | None = None
        self._input_snapshot: dict[str, object] | None = None
        self._active_lock = Lock()
        self._active_controller: object | None = None
        self._active_campaign_id: str | None = None
        self._deferred_stop = Event()
        self._stop_delivery_started = Event()

    @staticmethod
    def _new_campaign_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"v21-desktop-{stamp}-{uuid4().hex[:8]}"

    @staticmethod
    def _nonsecret_snapshot(inputs: RunInputs) -> dict[str, object]:
        return inputs.model_dump(mode="json", exclude={"api_key"})

    def preflight(self, inputs: RunInputs) -> PreflightSummary:
        campaign_id = self._campaign_id_factory()
        paths = self._builder.build(
            self._output_root, inputs, campaign_id
        )
        payload = self._live_payload(paths)
        dependencies = build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=paths.controller_root,
            max_rounds=inputs.max_rounds,
            target_tdecq_db=inputs.target_tdecq_db,
            allow_device_contact=True,
            live_config=payload,
        )
        if dependencies.live_activation is None:
            raise RuntimeError("LIVE PRE-FLIGHT activation is unavailable")
        report = dict(dependencies.live_activation.preflight())
        self._paths = paths
        self._input_snapshot = self._nonsecret_snapshot(inputs)
        return self._preflight_summary(campaign_id, inputs, report)

    @staticmethod
    def _live_payload(paths: SessionPaths) -> dict[str, object]:
        document = yaml.safe_load(paths.live.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("live"), dict):
            raise RuntimeError("generated LIVE configuration is invalid")
        return dict(document["live"])

    @staticmethod
    def _preflight_summary(
        campaign_id: str,
        inputs: RunInputs,
        report: dict[str, object],
    ) -> PreflightSummary:
        identities = report.get("adapter_identities")
        dca = identities.get("dca") if isinstance(identities, dict) else None
        i2c = identities.get("i2c") if isinstance(identities, dict) else None
        readback = report.get("i2c_readback")
        action = report.get("provider_action")
        dca_verified = isinstance(dca, dict) and (
            dca.get("identity_verified") is True
            and dca.get("configuration_verified") is True
        )
        i2c_verified = (
            isinstance(i2c, dict)
            and i2c.get("identity_verified") is True
            and i2c.get("configuration_verified") is True
            and isinstance(readback, dict)
            and readback == inputs.b0.model_dump()
        )
        kimi_valid = (
            report.get("kimi_connectivity_verified") is True
            and action in {"PROPOSE", "HOLD", "STOP", "REQUEST_HUMAN"}
        )
        return PreflightSummary(
            campaign_id=campaign_id,
            dca_verified=dca_verified,
            i2c_readback_verified=i2c_verified,
            kimi_proposal_valid=kimi_valid,
            dut_write_performed=(
                report.get("dut_write_performed") is not False
                or report.get("hardware_write_performed") is not False
            ),
            bert_contacted=report.get("bert_contacted") is not False,
            details=json.dumps(report, ensure_ascii=False, sort_keys=True),
        )

    def commission(self, inputs: RunInputs) -> DesktopResult:
        paths = self._paths
        if paths is None or self._input_snapshot != self._nonsecret_snapshot(inputs):
            raise DesktopSessionError(
                "commission inputs do not match the approved PRE-FLIGHT"
            )
        dependencies = build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=paths.controller_root,
            max_rounds=inputs.max_rounds,
            target_tdecq_db=inputs.target_tdecq_db,
            allow_device_contact=True,
            live_config=self._live_payload(paths),
        )
        if dependencies.live_activation is None:
            raise RuntimeError("LIVE commission activation is unavailable")
        activated = dependencies.live_activation.activate()
        controller = activated.controller
        request = activated.request
        if controller is None or request is None:
            raise RuntimeError("LIVE commission dependencies are incomplete")
        with self._active_lock:
            self._active_controller = controller
            self._active_campaign_id = request.campaign_id
        if self._deferred_stop.is_set():
            self._start_stop_delivery()
        try:
            result = controller.start(request)
            if result.state is ContinuousState.AWAITING_HUMAN:
                result = controller.resolve_human(
                    request.campaign_id, HumanDecision.RESTORE_B0
                )
        finally:
            try:
                if callable(activated.shutdown):
                    activated.shutdown()
            finally:
                with self._active_lock:
                    self._active_controller = None
                    self._active_campaign_id = None
        phase = {
            ContinuousState.COMPLETED: DesktopPhase.COMPLETED,
            ContinuousState.RECOVERY_REQUIRED: DesktopPhase.RECOVERY_REQUIRED,
        }.get(result.state, DesktopPhase.FAILED)
        return DesktopResult(
            campaign_id=result.campaign_id,
            phase=phase,
            campaign_root=paths.campaign_root,
            b0_restored=result.b0_restored,
            message=result.stop_reason or result.state.value,
        )

    def request_stop(self) -> None:
        self._deferred_stop.set()
        with self._active_lock:
            active = self._active_controller is not None
        if active:
            self._start_stop_delivery()

    def _start_stop_delivery(self) -> None:
        if self._stop_delivery_started.is_set():
            return
        self._stop_delivery_started.set()
        Thread(
            target=self._deliver_stop_when_running,
            name="vcsel-desktop-safe-stop",
            daemon=True,
        ).start()

    def _deliver_stop_when_running(self) -> None:
        deadline = monotonic() + 30.0
        poll = Event()
        while monotonic() < deadline:
            with self._active_lock:
                controller = self._active_controller
                campaign_id = self._active_campaign_id
            if controller is None or campaign_id is None:
                return
            try:
                status = controller.get_status(campaign_id)
                if status.state is ContinuousState.RUNNING:
                    controller.request_stop(campaign_id)
                    return
                if status.state in {
                    ContinuousState.COMPLETED,
                    ContinuousState.FAILED_SAFE,
                    ContinuousState.RECOVERY_REQUIRED,
                }:
                    return
            except Exception:
                pass
            poll.wait(0.05)

    def completed_rounds(self) -> int:
        paths = self._paths
        if paths is None:
            return 0
        rounds_root = (
            paths.controller_root / paths.campaign_root.name / "rounds"
        )
        count = 0
        for expected, path in enumerate(
            sorted(rounds_root.glob("*/round.json")), start=1
        ):
            try:
                record = RoundRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as error:
                raise RuntimeError(
                    f"persisted progress round {expected} is invalid"
                ) from error
            if record.round_index != expected:
                raise RuntimeError("persisted progress rounds are not contiguous")
            count += 1
        return count
