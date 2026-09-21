"""Request-scoped DCA execution in a killable spawned process.

The controller owns I2C/BERT and restoration. Only the worker imports PyVISA,
opens the DCA resource, and executes SCPI calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
import os
from pathlib import Path
import re
import time
from types import MappingProxyType
from uuid import uuid4

from .dca_watchdog import (
    DcaWatchdogError,
    DcaWatchdogTimeout,
    run_with_dca_watchdog,
)
from .live_config import LiveHardwareConfig
from .scpi_journal import ScpiOperationJournal


DEFAULT_DCA_CLEANUP_GRACE_S = 5.0
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


class DcaIsolationError(RuntimeError):
    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        self.evidence = MappingProxyType(dict(evidence))
        super().__init__(message)


class DcaIsolationTimeout(DcaIsolationError):
    pass


@dataclass(frozen=True, slots=True)
class IsolatedDcaMeasurementResult:
    acquisition_valid: bool
    er_db: float | None
    er_status: str
    tdecq_db: float | None
    tdecq_status: str
    tdecq_invalid_reason: str | None
    elapsed_s: float
    raw_responses: Mapping[str, object]
    artifacts: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class IsolatedDcaReadOnlyCapture:
    responses: Mapping[str, str]
    worker_pid: int
    journal_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "responses",
            MappingProxyType(dict(self.responses)),
        )


@dataclass(frozen=True, slots=True)
class LiveDcaWorkerRequest:
    profile: LiveHardwareConfig
    action: str
    journal_path: str
    timeout_s: float
    round_index: int
    operation_context: Mapping[str, object]
    capture_dataset: bool = False


def _finite_positive(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _metric_for_status(value: object, status: str) -> float | None:
    if status.strip().upper() != "CORR":
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or abs(float(value)) >= 9.0e37
    ):
        return None
    return float(value)


def _measurement_payload(result: object) -> dict[str, object]:
    raw = getattr(result, "raw_responses", None)
    if not isinstance(raw, Mapping):
        raise TypeError("DCA measurement raw_responses must be a mapping")
    return {
        "kind": "measurement",
        "acquisition_valid": getattr(result, "acquisition_valid"),
        "er_db": getattr(result, "er_db"),
        "er_status": getattr(result, "er_status"),
        "tdecq_db": getattr(result, "tdecq_db"),
        "tdecq_status": getattr(result, "tdecq_status"),
        "tdecq_invalid_reason": getattr(result, "tdecq_invalid_reason"),
        "elapsed_s": getattr(result, "elapsed_s"),
        "raw_responses": dict(raw),
        "artifacts": tuple(getattr(result, "artifacts", ()) or ()),
    }


def _read_only_query_commands(profile: LiveHardwareConfig) -> tuple[str, ...]:
    return (
        "*IDN?",
        ":MEASure:RESults? STANdard",
        f":FUNCtion{profile.dca.function}:OPERand?",
        f":{profile.characterization.er.source}:WAVelength:VALue?",
        ":TRIGger:SOURce?",
        ":TRIGger:DCDRatio?",
        ":TRIGger:SRATe?",
        ":TRIGger:PLENgth?",
    )


def _live_dca_worker(send: object, request: LiveDcaWorkerRequest) -> None:
    deadline = time.monotonic() + request.timeout_s

    def operation_context() -> dict[str, object]:
        return {
            **dict(request.operation_context),
            "round_index": request.round_index,
            "remaining_deadline_s": max(0.0, deadline - time.monotonic()),
        }

    send.send({"event": "WORKER_STARTED"})
    send.send(
        {
            "event": "ROUND_STARTED",
            "repeat_index": request.round_index,
            "started_at_utc": request.operation_context["started_at_utc"],
        }
    )
    try:
        from vcsel_ai_tuning.hardware.characterization import CampaignDcaAdapter
        from vcsel_ai_tuning.hardware.flexdca import open_visa_session

        timeout_ms = max(
            1,
            min(request.profile.dca.timeout_ms, int(request.timeout_s * 1_000)),
        )
        with open_visa_session(
            request.profile.dca.resource,
            timeout_ms=timeout_ms,
            open_timeout_ms=timeout_ms,
            journal=ScpiOperationJournal(request.journal_path),
            operation_context=operation_context,
        ) as session:
            adapter = CampaignDcaAdapter(session=session, config=request.profile)
            if request.action == "preflight":
                payload: dict[str, object] = {
                    "kind": "preflight",
                    "value": dict(adapter.preflight()),
                }
            elif request.action == "measure_single":
                from vcsel_ai_tuning.v1_1.live_sweep import (
                    _exclusive_dca_access_for_resource,
                )

                access = _exclusive_dca_access_for_resource(
                    request.profile.dca.resource
                )
                with access.claim():
                    measurement = adapter.measure_single()
                    payload = _measurement_payload(measurement)
                    if request.capture_dataset:
                        try:
                            from vcsel_ai_tuning.v1_1.live_dataset_capture import (
                                FlexDcaDatasetCapture,
                            )

                            capture = FlexDcaDatasetCapture(
                                session=session,
                                raw_source=request.profile.characterization.tdecq.input_channel,
                                equalized_source=f"FUNC{request.profile.dca.function}",
                                equalizer_function=request.profile.dca.function,
                                exclusive_access=access.owned_by_current_thread,
                            ).capture(f"acquisition-{uuid4().hex}", measurement)
                            payload["artifacts"] = (capture,)
                        except Exception as dataset_error:
                            payload["raw_responses"] = {
                                **dict(payload["raw_responses"]),
                                "dataset_capture_error": str(dataset_error),
                            }
            elif request.action == "read_only_queries":
                from vcsel_ai_tuning.v1_1.read_only_runtime import (
                    ReadOnlyV11Gateway,
                )

                raw_responses: dict[str, str] = {}
                gateway = ReadOnlyV11Gateway(
                    request.profile,
                    total_timeout_s=request.timeout_s,
                )
                gateway._capture_dca(
                    session,
                    raw_responses,
                    lambda: deadline - time.monotonic(),
                )
                payload = {
                    "kind": "read_only_queries",
                    "responses": raw_responses,
                    "worker_pid": os.getpid(),
                }
            else:
                raise ValueError(f"unsupported isolated DCA action {request.action}")
            send.send(
                {"event": "ROUND_COMPLETED", "repeat_index": request.round_index}
            )
            send.send({"event": "CLEANUP_STARTED"})
        send.send({"event": "WORKER_COMPLETED", "result": payload})
    except BaseException as error:
        try:
            send.send(
                {
                    "event": "WORKER_FAILED",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            raise


class IsolatedLiveDcaClient:
    """Synchronous facade backed by one worker process per request."""

    def __init__(
        self,
        *,
        profile: LiveHardwareConfig,
        artifact_root: str | Path,
        timeout_s: float,
        cleanup_grace_s: float = DEFAULT_DCA_CLEANUP_GRACE_S,
        capture_dataset: bool = False,
        worker_target: Callable[..., None] = _live_dca_worker,
        watchdog: Callable[..., dict[str, object]] = run_with_dca_watchdog,
    ) -> None:
        if not isinstance(profile, LiveHardwareConfig):
            raise TypeError("profile must be LiveHardwareConfig")
        self._profile = profile
        self._artifact_root = Path(artifact_root)
        self._timeout_s = _finite_positive(timeout_s, "timeout_s")
        self._cleanup_grace_s = _finite_positive(
            cleanup_grace_s, "cleanup_grace_s"
        )
        if type(capture_dataset) is not bool:
            raise TypeError("capture_dataset must be a boolean")
        self._capture_dataset = capture_dataset
        self._worker_target = worker_target
        self._watchdog = watchdog
        self._context: dict[str, object] | None = None
        self._active = False

    @property
    def worker_alive(self) -> bool:
        return self._active

    def set_operation_context(
        self, *, task_id: str, item_id: str, round_index: int
    ) -> None:
        if not isinstance(task_id, str) or not _IDENTIFIER.fullmatch(task_id):
            raise ValueError("task_id must be a path-safe identifier")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be nonblank")
        if any(character in item_id for character in ("/", "\\")):
            raise ValueError("item_id cannot contain path separators")
        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_index must be a nonnegative integer")
        self._context = {
            "task_id": task_id,
            "item_id": item_id,
            "round_index": round_index,
        }

    def _request(self, action: str) -> dict[str, object]:
        if self._context is None:
            raise RuntimeError("DCA operation context must be set before use")
        context = {
            **self._context,
            "action": action,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        journal_path = (
            self._artifact_root
            / str(context["task_id"])
            / "dca_io_journal.jsonl"
        )
        request = LiveDcaWorkerRequest(
            profile=self._profile,
            action=action,
            journal_path=str(journal_path),
            timeout_s=self._timeout_s,
            round_index=int(context["round_index"]),
            operation_context=context,
            capture_dataset=self._capture_dataset,
        )
        self._active = True
        try:
            result = self._watchdog(
                worker_target=self._worker_target,
                worker_args=(request,),
                journal_path=journal_path,
                timeout_s=self._timeout_s,
                cleanup_grace_s=self._cleanup_grace_s,
            )
        except DcaWatchdogError as error:
            evidence = {
                **context,
                **dict(error.evidence),
                "observation_valid": False,
                "er_db": None,
                "tdecq_db": None,
            }
            error_type = (
                DcaIsolationTimeout
                if isinstance(error, DcaWatchdogTimeout)
                else DcaIsolationError
            )
            raise error_type(str(error), evidence=evidence) from error
        finally:
            self._active = False
        return {**result, "_journal_path": str(journal_path)}

    def preflight(self) -> Mapping[str, object]:
        result = self._request("preflight")
        value = result.get("value")
        if result.get("kind") != "preflight" or not isinstance(value, Mapping):
            raise DcaIsolationError(
                "isolated DCA preflight returned a malformed result",
                evidence={
                    **(self._context or {}),
                    "worker_result": result,
                    "observation_valid": False,
                },
            )
        return dict(value)

    def read_only_queries(self) -> IsolatedDcaReadOnlyCapture:
        result = self._request("read_only_queries")
        responses = result.get("responses")
        worker_pid = result.get("worker_pid")
        expected_commands = _read_only_query_commands(self._profile)
        if (
            result.get("kind") != "read_only_queries"
            or not isinstance(responses, Mapping)
            or set(responses) != set(expected_commands)
            or any(
                not isinstance(responses[command], str)
                for command in expected_commands
            )
            or type(worker_pid) is not int
            or worker_pid <= 0
        ):
            raise DcaIsolationError(
                "isolated READ_ONLY DCA capture returned a malformed result",
                evidence={
                    **(self._context or {}),
                    "worker_result": result,
                    "observation_valid": False,
                },
            )
        return IsolatedDcaReadOnlyCapture(
            responses={
                command: responses[command] for command in expected_commands
            },
            worker_pid=worker_pid,
            journal_path=str(result["_journal_path"]),
        )

    def measure_single(self) -> IsolatedDcaMeasurementResult:
        result = self._request("measure_single")
        if result.get("kind") != "measurement":
            raise DcaIsolationError(
                "isolated DCA acquisition returned a malformed result",
                evidence={
                    **(self._context or {}),
                    "worker_result": result,
                    "observation_valid": False,
                    "er_db": None,
                    "tdecq_db": None,
                },
            )
        try:
            acquisition_valid = result["acquisition_valid"]
            if type(acquisition_valid) is not bool:
                raise TypeError("acquisition_valid must be a boolean")
            er_status = str(result["er_status"]).strip().upper()
            tdecq_status = str(result["tdecq_status"]).strip().upper()
            tdecq = _metric_for_status(result["tdecq_db"], tdecq_status)
            er = _metric_for_status(result["er_db"], er_status)
            invalid_reason = result["tdecq_invalid_reason"]
            if tdecq is None and not isinstance(invalid_reason, str):
                invalid_reason = (
                    f"terminal TDECQ status {tdecq_status or 'UNKNOWN'}"
                )
            return IsolatedDcaMeasurementResult(
                acquisition_valid=acquisition_valid,
                er_db=er,
                er_status=er_status,
                tdecq_db=tdecq,
                tdecq_status=tdecq_status,
                tdecq_invalid_reason=invalid_reason,
                elapsed_s=float(result["elapsed_s"]),
                raw_responses=dict(result["raw_responses"]),
                artifacts=tuple(result.get("artifacts", ()) or ()),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DcaIsolationError(
                "isolated DCA acquisition returned a malformed result",
                evidence={
                    **(self._context or {}),
                    "worker_result": result,
                    "observation_valid": False,
                    "er_db": None,
                    "tdecq_db": None,
                },
            ) from error
