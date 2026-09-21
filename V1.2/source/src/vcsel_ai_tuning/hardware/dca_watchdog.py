"""Killable process watchdog for DCA hardware execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite
import multiprocessing
from pathlib import Path
import time
from types import MappingProxyType
from uuid import uuid4

from .scpi_journal import ScpiOperationJournal


class DcaWatchdogError(RuntimeError):
    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        self.evidence = MappingProxyType(dict(evidence))
        super().__init__(message)


class DcaWatchdogTimeout(DcaWatchdogError):
    pass


class DcaWorkerFailure(DcaWatchdogError):
    pass


def _positive_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _completed_status(
    events: Sequence[Mapping[str, object]], command: str, round_index: int
) -> str:
    for event in reversed(events):
        if (
            event.get("event") == "COMPLETED"
            and event.get("operation") == "query"
            and event.get("command") == command
            and event.get("round_index") == round_index
            and isinstance(event.get("response"), str)
        ):
            return str(event["response"]).strip().upper()
    return "NOT_AVAILABLE"


def _connection_state(
    events: Sequence[Mapping[str, object]],
    *,
    worker_alive: bool,
    request_event: Mapping[str, object] | None,
) -> str:
    if worker_alive:
        return "WORKER_STILL_ALIVE"
    context_keys = ("task_id", "item_id", "round_index", "action")
    if request_event is not None:
        request_context = {
            key: request_event[key]
            for key in context_keys
            if key in request_event
        }
        events = tuple(
            event
            for event in events
            if all(event.get(key) == value for key, value in request_context.items())
        )
    if any(
        event.get("event") == "FAILED"
        and event.get("operation") in {"close", "resource_manager_close"}
        for event in events
    ):
        return "CLOSE_FAILED"
    completed = {
        str(event.get("operation"))
        for event in events
        if event.get("event") == "COMPLETED"
    }
    if {"close", "resource_manager_close"} <= completed:
        return "VERIFIED_CLOSED"
    return "UNVERIFIED_AFTER_FORCED_TERMINATION"


def _stop_worker(process: multiprocessing.Process, cleanup_grace_s: float) -> bool:
    half = cleanup_grace_s / 2.0
    if process.is_alive():
        process.terminate()
        process.join(half)
    if process.is_alive():
        process.kill()
        process.join(max(0.0, cleanup_grace_s - half))
    return process.is_alive()


def _failure_evidence(
    *,
    process: multiprocessing.Process,
    journal_path: str | Path,
    lifecycle: list[dict[str, object]],
    reason: str,
) -> dict[str, object]:
    journal = ScpiOperationJournal(journal_path)
    events = journal.events()
    last = journal.last_in_flight()
    request_event = last or next(
        (event for event in reversed(events) if event.get("event") == "STARTED"),
        None,
    )
    round_index = int(
        last.get("round_index", 0)
        if last is not None
        else next(
            (
                event.get("repeat_index", 0)
                for event in reversed(lifecycle)
                if event.get("event") == "ROUND_STARTED"
            ),
            0,
        )
    )
    return {
        "timeout_reason": reason,
        "exitcode": process.exitcode,
        "failed_repeat_index": round_index,
        "last_in_flight": last,
        "er_status": _completed_status(
            events, ":MEASure:EYE:OER:STATus?", round_index
        ),
        "tdecq_status": _completed_status(
            events, ":MEASure:EYE:TDEQ:STATus?", round_index
        ),
        "worker_terminated": not process.is_alive(),
        "worker_alive_after_cleanup": process.is_alive(),
        "connection_state": _connection_state(
            events,
            worker_alive=process.is_alive(),
            request_event=request_event,
        ),
        "lifecycle_events": list(lifecycle),
    }


def run_with_dca_watchdog(
    *,
    worker_target: Callable[..., None],
    worker_args: Sequence[object],
    journal_path: str | Path,
    timeout_s: float,
    cleanup_grace_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    process_context: multiprocessing.context.BaseContext | None = None,
) -> dict[str, object]:
    timeout = _positive_finite(timeout_s, "timeout_s")
    cleanup_grace = _positive_finite(cleanup_grace_s, "cleanup_grace_s")
    context = process_context or multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(send, *tuple(worker_args)),
        daemon=True,
    )
    started = monotonic()
    stage_started = started
    deadline = started + timeout
    stage = "startup"
    lifecycle: list[dict[str, object]] = []
    process.start()
    send.close()
    try:
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            if receive.poll(min(0.05, remaining)):
                try:
                    message = receive.recv()
                except EOFError:
                    message = None
                if not isinstance(message, dict):
                    if not process.is_alive():
                        evidence = _failure_evidence(
                            process=process,
                            journal_path=journal_path,
                            lifecycle=lifecycle,
                            reason="DCA_WORKER_EXITED",
                        )
                        raise DcaWorkerFailure(
                            "DCA worker exited without a terminal message",
                            evidence=evidence,
                        )
                    continue
                normalized = dict(message)
                lifecycle.append(normalized)
                event = normalized.get("event")
                if event == "ROUND_STARTED":
                    stage = "round"
                    stage_started = monotonic()
                    deadline = stage_started + timeout
                elif event == "ROUND_COMPLETED":
                    stage = "between_rounds"
                    stage_started = monotonic()
                    deadline = stage_started + timeout
                elif event == "CLEANUP_STARTED":
                    stage = "cleanup"
                    stage_started = monotonic()
                    deadline = stage_started + cleanup_grace
                elif event == "WORKER_COMPLETED":
                    process.join(cleanup_grace)
                    if process.is_alive():
                        break
                    result = normalized.get("result", {})
                    return dict(result) if isinstance(result, dict) else {}
                elif event == "WORKER_FAILED":
                    process.join(cleanup_grace)
                    if process.is_alive():
                        _stop_worker(process, cleanup_grace)
                    evidence = _failure_evidence(
                        process=process,
                        journal_path=journal_path,
                        lifecycle=lifecycle,
                        reason="DCA_WORKER_FAILED",
                    )
                    evidence["worker_error"] = normalized
                    raise DcaWorkerFailure(
                        str(normalized.get("error", "DCA worker failed")),
                        evidence=evidence,
                    )
            elif not process.is_alive():
                evidence = _failure_evidence(
                    process=process,
                    journal_path=journal_path,
                    lifecycle=lifecycle,
                    reason="DCA_WORKER_EXITED",
                )
                raise DcaWorkerFailure(
                    "DCA worker exited without a terminal message", evidence=evidence
                )

        worker_alive = _stop_worker(process, cleanup_grace)
        finished = monotonic()
        evidence = _failure_evidence(
            process=process,
            journal_path=journal_path,
            lifecycle=lifecycle,
            reason="DCA_IO_WATCHDOG_EXPIRED",
        )
        evidence.update(
            {
                "watchdog_stage": stage,
                "elapsed_s": max(0.0, finished - stage_started),
                "total_elapsed_s": max(0.0, finished - started),
                "timeout_s": timeout,
                "cleanup_grace_s": cleanup_grace,
                "worker_terminated": not worker_alive,
                "worker_alive_after_cleanup": worker_alive,
            }
        )
        last = evidence["last_in_flight"]
        last_mapping = last if isinstance(last, Mapping) else {}
        ScpiOperationJournal(journal_path).append(
            {
                **{
                    key: value
                    for key, value in last_mapping.items()
                    if key
                    not in {
                        "schema_version",
                        "operation_id",
                        "event",
                        "operation",
                        "command",
                        "started_at_utc",
                    }
                },
                "schema_version": "1.0",
                "operation_id": str(last_mapping.get("operation_id", uuid4().hex)),
                "event": "WATCHDOG_TIMEOUT",
                "operation": last_mapping.get("operation", stage),
                "command": last_mapping.get("command"),
                "round_index": evidence["failed_repeat_index"],
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "timeout_reason": evidence["timeout_reason"],
                "connection_state": evidence["connection_state"],
            }
        )
        raise DcaWatchdogTimeout(
            "DCA I/O watchdog deadline expired", evidence=evidence
        )
    finally:
        receive.close()
