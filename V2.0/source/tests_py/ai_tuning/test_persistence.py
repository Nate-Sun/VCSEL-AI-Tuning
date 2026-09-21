from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from vcsel_ai_tuning.domain import (
    AppliedSettings,
    Candidate,
    Fidelity,
    MetricValue,
    Observation,
    RunEvent,
    RunState,
)
from vcsel_ai_tuning.persistence import ArtifactStore, RunRepository, SqliteStore


def completed_observation():
    now = datetime.now(timezone.utc)
    return Observation(
        campaign_id="campaign-1",
        run_id="run-1",
        candidate_id="candidate-0000",
        state=RunState.COMPLETE,
        settings=AppliedSettings(
            requested={"bias_ma": 5.0},
            applied={"bias_ma": 5.0},
            readback={"bias_ma": 5.0},
        ),
        metrics={
            "skew_pp_ui": MetricValue(
                value=0.03,
                unit="UI",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="skew-v1",
            )
        },
        valid=True,
        started_at=now,
        finished_at=now,
    )


def test_commit_writes_artifacts_then_exposes_sqlite_observation(tmp_path: Path):
    store = SqliteStore(tmp_path / "state.sqlite")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    repository = RunRepository(store, artifacts)
    candidate = Candidate(candidate_id="candidate-0000", requested={"bias_ma": 5.0})
    waveform = b"time_s,amplitude_v\n0,0\n1e-12,1\n"

    saved = repository.commit_run(
        candidate=candidate,
        observation=completed_observation(),
        events=[RunEvent(state=RunState.COMPLETE)],
        raw_artifacts={"waveform.csv": (waveform, "text/csv")},
    )

    final = tmp_path / "artifacts" / "campaigns" / "campaign-1" / "runs" / "run-1"
    assert final.is_dir()
    assert (final / "observation.json").is_file()
    assert saved.artifacts[0].sha256 == sha256(waveform).hexdigest()
    assert store.list_observations("campaign-1") == [saved]


def test_database_failure_quarantines_artifacts(tmp_path: Path, monkeypatch):
    store = SqliteStore(tmp_path / "state.sqlite")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    repository = RunRepository(store, artifacts)

    def fail_save(_observation):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "save_observation", fail_save)

    with pytest.raises(RuntimeError, match="database unavailable"):
        repository.commit_run(
            candidate=Candidate(
                candidate_id="candidate-0000", requested={"bias_ma": 5.0}
            ),
            observation=completed_observation(),
            events=[RunEvent(state=RunState.COMPLETE)],
            raw_artifacts={},
        )

    assert list((tmp_path / "artifacts" / "quarantine").iterdir())
    assert store.list_observations("campaign-1") == []
