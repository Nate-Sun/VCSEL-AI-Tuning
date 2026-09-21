from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.hardware.live_dca_isolation import (
    DcaIsolationTimeout,
    IsolatedLiveDcaClient,
    LiveDcaWorkerRequest,
)
import vcsel_ai_tuning.hardware.live_dca_isolation as isolation_module
from vcsel_ai_tuning.hardware.scpi_journal import ScpiOperationJournal
from vcsel_ai_tuning.v1_1.dataset import FrozenDatasetCapture, encode_density_png


ROOT = Path(__file__).resolve().parents[3]


def _profile():
    return load_live_hardware_config(
        ROOT / "configs" / "hardware" / "vcsel_lab_live.yaml"
    )


def hang_in_query_worker(send, request) -> None:
    send.send({"event": "ROUND_STARTED", "repeat_index": request.round_index})
    ScpiOperationJournal(request.journal_path).append(
        {
            "schema_version": "1.0",
            "operation_id": "blocked-query",
            "event": "STARTED",
            "operation": "query",
            "command": ":MEASure:EYE:TDEQ?",
            **dict(request.operation_context),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    while True:
        time.sleep(1.0)


def dataset_result_worker(send, request) -> None:
    raw = np.arange(6, dtype=np.uint32).reshape(2, 3)
    equalized = raw + 10
    capture = FrozenDatasetCapture(
        acquisition_id="isolated-acquisition",
        same_acquisition_verified=True,
        raw_eye_density=raw,
        density_axis_order=("amplitude", "time"),
        density_source="CHAN1A",
        tdecq_equalized_eye_png=encode_density_png(equalized),
        equalizer_source="FUNC1",
        raw_responses={"isolated": True},
    )
    send.send({"event": "ROUND_STARTED", "repeat_index": request.round_index})
    send.send({"event": "ROUND_COMPLETED", "repeat_index": request.round_index})
    send.send({"event": "CLEANUP_STARTED"})
    send.send(
        {
            "event": "WORKER_COMPLETED",
            "result": {
                "kind": "measurement",
                "acquisition_valid": True,
                "er_db": 5.2,
                "er_status": "CORR",
                "tdecq_db": 2.4,
                "tdecq_status": "CORR",
                "tdecq_invalid_reason": None,
                "elapsed_s": 0.1,
                "raw_responses": {"worker": "isolated"},
                "artifacts": (capture,),
            },
        }
    )


def terminal_invalid_worker(send, request) -> None:
    send.send({"event": "ROUND_STARTED", "repeat_index": request.round_index})
    send.send({"event": "ROUND_COMPLETED", "repeat_index": request.round_index})
    send.send({"event": "CLEANUP_STARTED"})
    send.send(
        {
            "event": "WORKER_COMPLETED",
            "result": {
                "kind": "measurement",
                "acquisition_valid": False,
                "er_db": 5.2,
                "er_status": "INV",
                "tdecq_db": 2.4,
                "tdecq_status": "INV",
                "tdecq_invalid_reason": "No Data",
                "elapsed_s": 0.1,
                "raw_responses": {"terminal_status": "INV"},
                "artifacts": (),
            },
        }
    )


def read_only_queries_worker(send, request) -> None:
    send.send({"event": "ROUND_STARTED", "repeat_index": request.round_index})
    send.send({"event": "ROUND_COMPLETED", "repeat_index": request.round_index})
    send.send({"event": "CLEANUP_STARTED"})
    send.send(
        {
            "event": "WORKER_COMPLETED",
            "result": {
                "kind": "read_only_queries",
                "responses": {
                    "*IDN?": "Keysight Technologies,N1010A,MY123,1.0",
                    ":MEASure:RESults? STANdard": "results",
                    ":FUNCtion1:OPERand?": "CHAN1A",
                    ":CHAN1A:WAVelength:VALue?": "8.5e-7",
                    ":TRIGger:SOURce?": "FPAN",
                    ":TRIGger:DCDRatio?": "SUB4",
                    ":TRIGger:SRATe?": "53125000000",
                    ":TRIGger:PLENgth?": "65535",
                },
                "worker_pid": os.getpid(),
            },
        }
    )


def test_construction_opens_no_connection_or_artifact(tmp_path):
    client = IsolatedLiveDcaClient(
        profile=_profile(), artifact_root=tmp_path, timeout_s=1.0
    )
    assert list(tmp_path.iterdir()) == []
    assert client.worker_alive is False


def test_timeout_nulls_metrics_and_confirms_worker_dead(tmp_path):
    client = IsolatedLiveDcaClient(
        profile=_profile(),
        artifact_root=tmp_path,
        timeout_s=1.0,
        cleanup_grace_s=0.2,
        worker_target=hang_in_query_worker,
    )
    client.set_operation_context(task_id="t1", item_id="c1", round_index=1)

    with pytest.raises(DcaIsolationTimeout) as caught:
        client.measure_single()

    evidence = caught.value.evidence
    assert evidence["worker_alive_after_cleanup"] is False
    assert evidence["tdecq_db"] is None
    assert evidence["er_db"] is None
    assert evidence["last_in_flight"]["operation"] == "query"


def test_worker_transports_same_acquisition_dataset(tmp_path):
    client = IsolatedLiveDcaClient(
        profile=_profile(),
        artifact_root=tmp_path,
        timeout_s=2.0,
        cleanup_grace_s=0.2,
        capture_dataset=True,
        worker_target=dataset_result_worker,
    )
    client.set_operation_context(task_id="t2", item_id="c2", round_index=2)

    result = client.measure_single()

    assert result.tdecq_db == 2.4
    assert len(result.artifacts) == 1
    capture = result.artifacts[0]
    assert isinstance(capture, FrozenDatasetCapture)
    assert capture.acquisition_id == "isolated-acquisition"
    assert capture.same_acquisition_verified is True


def test_production_worker_proves_dataset_capture_owns_its_dca_session(
    monkeypatch,
    tmp_path,
):
    import vcsel_ai_tuning.hardware.characterization as characterization_module
    import vcsel_ai_tuning.hardware.flexdca as flexdca_module
    import vcsel_ai_tuning.v1_1.live_dataset_capture as capture_module

    ownership_checks = []

    @contextmanager
    def open_session(*_args, **_kwargs):
        yield object()

    class Adapter:
        def __init__(self, *, session, config):
            assert session is not None
            assert config == _profile()

        def measure_single(self):
            return SimpleNamespace(
                acquisition_valid=True,
                er_db=3.1,
                er_status="CORR",
                tdecq_db=4.2,
                tdecq_status="CORR",
                tdecq_invalid_reason=None,
                elapsed_s=0.1,
                raw_responses={"worker": "production"},
                artifacts=(),
            )

    class Capture:
        def __init__(self, *, exclusive_access=None, **_kwargs):
            self._exclusive_access = exclusive_access

        def capture(self, acquisition_id, _measurement):
            before = (
                callable(self._exclusive_access)
                and self._exclusive_access() is True
            )
            ownership_checks.append(before)
            after = (
                callable(self._exclusive_access)
                and self._exclusive_access() is True
            )
            ownership_checks.append(after)
            raw = np.arange(6, dtype=np.uint32).reshape(2, 3)
            return FrozenDatasetCapture(
                acquisition_id=acquisition_id,
                same_acquisition_verified=before and after,
                raw_eye_density=raw,
                density_axis_order=("amplitude", "time"),
                density_source="CHAN1A",
                tdecq_equalized_eye_png=encode_density_png(raw + 10),
                equalizer_source="FUNC1",
                raw_responses={"isolated": True},
            )

    monkeypatch.setattr(flexdca_module, "open_visa_session", open_session)
    monkeypatch.setattr(characterization_module, "CampaignDcaAdapter", Adapter)
    monkeypatch.setattr(capture_module, "FlexDcaDatasetCapture", Capture)

    class Send:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

    send = Send()
    request = LiveDcaWorkerRequest(
        profile=_profile(),
        action="measure_single",
        journal_path=str(tmp_path / "journal.jsonl"),
        timeout_s=2.0,
        round_index=1,
        operation_context={
            "task_id": "worker-ownership",
            "item_id": "candidate-1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        capture_dataset=True,
    )

    isolation_module._live_dca_worker(send, request)

    completed = next(
        message for message in send.messages if message["event"] == "WORKER_COMPLETED"
    )
    capture = completed["result"]["artifacts"][0]
    assert capture.same_acquisition_verified is True
    assert ownership_checks == [True, True]


def test_terminal_invalid_status_never_reuses_numeric_metrics(tmp_path):
    client = IsolatedLiveDcaClient(
        profile=_profile(),
        artifact_root=tmp_path,
        timeout_s=2.0,
        cleanup_grace_s=0.2,
        worker_target=terminal_invalid_worker,
    )
    client.set_operation_context(task_id="t3", item_id="c3", round_index=3)

    result = client.measure_single()

    assert result.acquisition_valid is False
    assert result.tdecq_status == "INV"
    assert result.tdecq_db is None
    assert result.er_status == "INV"
    assert result.er_db is None
    assert result.tdecq_invalid_reason == "No Data"


def test_read_only_query_capture_runs_in_spawned_worker(tmp_path):
    client = IsolatedLiveDcaClient(
        profile=_profile(),
        artifact_root=tmp_path,
        timeout_s=2.0,
        cleanup_grace_s=0.2,
        worker_target=read_only_queries_worker,
    )
    client.set_operation_context(
        task_id="read-only-preflight",
        item_id="snapshot-1",
        round_index=0,
    )

    capture = client.read_only_queries()

    assert capture.worker_pid != os.getpid()
    assert capture.responses["*IDN?"].startswith("Keysight Technologies,N1010A,")
    assert capture.responses[":MEASure:RESults? STANdard"] == "results"


def test_production_worker_executes_full_read_only_dca_capture(
    tmp_path, monkeypatch
):
    import vcsel_ai_tuning.hardware.flexdca as flexdca_module

    standard_results = (
        "Name=Outer ER,Source=1A,Current=3.025,Status=Correct,"
        "Name=TDECQ,Source=F1,Current=4.56,Status=Correct"
    )
    responses = {
        "*IDN?": "Keysight Technologies,N1010A,MY123,1.0",
        ":MEASure:RESults? STANdard": standard_results,
        ":FUNCtion1:OPERand?": "CHAN1A",
        ":CHAN1A:WAVelength:VALue?": "8.5e-7",
        ":TRIGger:SOURce?": "FPAN",
        ":TRIGger:DCDRatio?": "SUB4",
        ":TRIGger:SRATe?": "53125000000",
        ":TRIGger:PLENgth?": "65535",
    }

    class Session:
        timeout = 60_000

        def query(self, command):
            return responses[command]

    @contextmanager
    def fake_visa(*_args, **_kwargs):
        yield Session()

    class Send:
        messages = []

        def send(self, message):
            self.messages.append(message)

    monkeypatch.setattr(flexdca_module, "open_visa_session", fake_visa)
    send = Send()
    request = LiveDcaWorkerRequest(
        profile=_profile(),
        action="read_only_queries",
        journal_path=str(tmp_path / "journal.jsonl"),
        timeout_s=2.0,
        round_index=0,
        operation_context={
            "task_id": "read-only-preflight",
            "item_id": "snapshot-1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    isolation_module._live_dca_worker(send, request)

    result = send.messages[-1]
    assert result["event"] == "WORKER_COMPLETED"
    assert result["result"]["kind"] == "read_only_queries"
    assert result["result"]["responses"] == responses
