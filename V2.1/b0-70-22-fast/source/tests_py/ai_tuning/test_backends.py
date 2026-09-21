from pathlib import Path

import pytest

from vcsel_ai_tuning.backends import BackendFault, FakeBackend, ReplayBackend
from vcsel_ai_tuning.domain import Candidate, Fidelity


def test_fake_backend_records_contract_order_and_is_deterministic():
    backend = FakeBackend()
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )

    assert backend.identify()["model"] == "FakeVCSELBench"
    backend.configure({"mode": "fast"})
    backend.apply(candidate, candidate.requested)
    readback = backend.readback()
    backend.wait_ready()
    result = backend.measure_fast()
    backend.safe_off()
    backend.safe_off()

    assert readback == candidate.requested
    assert result.metrics["skew_pp_ui"].value == pytest.approx(0.04)
    assert backend.call_order == [
        "identify",
        "configure",
        "apply",
        "readback",
        "wait_ready",
        "measure_fast",
        "safe_off",
        "safe_off",
    ]
    assert backend.is_safe is True


def test_fake_backend_injects_fault_at_named_operation():
    backend = FakeBackend(fail_at="measure_fast")
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )
    backend.apply(candidate, candidate.requested)

    with pytest.raises(BackendFault, match="measure_fast"):
        backend.measure_fast()


def test_replay_backend_preserves_metric_fidelity():
    fixture = Path(__file__).parent / "fixtures" / "replay_observations.jsonl"
    backend = ReplayBackend.from_jsonl(fixture)
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )

    backend.apply(candidate, candidate.requested)
    result = backend.measure_fast()

    assert backend.readback() == candidate.requested
    assert result.metrics["skew_pp_ui"].fidelity is Fidelity.SIMULATION_PROXY


def test_replay_backend_rejects_unknown_candidate():
    fixture = Path(__file__).parent / "fixtures" / "replay_observations.jsonl"
    backend = ReplayBackend.from_jsonl(fixture)
    candidate = Candidate(
        candidate_id="candidate-9999",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )

    with pytest.raises(KeyError, match="candidate-9999"):
        backend.apply(candidate, candidate.requested)
