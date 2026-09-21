from __future__ import annotations

from datetime import datetime, timezone
import time

import pytest

from vcsel_ai_tuning.hardware.dca_watchdog import (
    DcaWatchdogTimeout,
    run_with_dca_watchdog,
)
from vcsel_ai_tuning.hardware.scpi_journal import ScpiOperationJournal


def hang_in_query_worker(send, journal_path: str) -> None:
    send.send({"event": "WORKER_STARTED"})
    send.send({"event": "ROUND_STARTED", "repeat_index": 7})
    journal = ScpiOperationJournal(journal_path)
    for operation in ("close", "resource_manager_close"):
        journal.append(
            {
                "schema_version": "1.0",
                "operation_id": f"old-{operation}",
                "event": "STARTED",
                "operation": operation,
                "command": None,
                "task_id": "t1",
                "item_id": "old-candidate",
                "round_index": 6,
            }
        )
        journal.append(
            {
                "schema_version": "1.0",
                "operation_id": f"old-{operation}",
                "event": "COMPLETED",
                "operation": operation,
                "command": None,
                "task_id": "t1",
                "item_id": "old-candidate",
                "round_index": 6,
            }
        )
    journal.append(
        {
            "schema_version": "1.0",
            "operation_id": "blocked-query",
            "event": "STARTED",
            "operation": "query",
            "command": ":MEASure:EYE:TDEQ?",
            "task_id": "t1",
            "item_id": "c1",
            "round_index": 7,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    while True:
        time.sleep(1.0)


def test_watchdog_terminates_hung_spawned_worker_and_keeps_evidence(tmp_path):
    journal_path = tmp_path / "journal.jsonl"

    with pytest.raises(DcaWatchdogTimeout) as caught:
        run_with_dca_watchdog(
            worker_target=hang_in_query_worker,
            worker_args=(str(journal_path),),
            journal_path=journal_path,
            timeout_s=1.0,
            cleanup_grace_s=0.2,
        )

    evidence = caught.value.evidence
    assert evidence["worker_alive_after_cleanup"] is False
    assert evidence["worker_terminated"] is True
    assert evidence["failed_repeat_index"] == 7
    assert evidence["last_in_flight"]["operation"] == "query"
    assert evidence["connection_state"] == "UNVERIFIED_AFTER_FORCED_TERMINATION"
    assert ScpiOperationJournal(journal_path).events()[-1]["event"] == (
        "WATCHDOG_TIMEOUT"
    )
