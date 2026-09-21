from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from pydantic import ValidationError

import vcsel_ai_tuning.v2.kimi_memory as kimi_memory

from vcsel_ai_tuning.v2.contracts import (
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    EyeSample,
    PerceptionResult,
)
from vcsel_ai_tuning.v2.kimi_memory import (
    KIMI_PROVIDER_ID,
    KimiCampaignMemory,
    KimiMemoryEntry,
    KimiMemoryRevisionConflict,
    KimiMemorySnapshot,
)


def _sample(eye_sample: EyeSample, measurement_id: str, campaign_id: str | None = None) -> EyeSample:
    return eye_sample.model_copy(
        update={
            "measurement_id": measurement_id,
            "campaign_id": campaign_id or eye_sample.campaign_id,
        }
    )


def _result(
    sample: EyeSample,
    *,
    provider_id: str = KIMI_PROVIDER_ID,
    decision_provider_id: str | None = None,
    valid: bool = True,
    ood: bool = False,
    memory_revision: str | None = None,
    memory_id: str | None = None,
    reference_measurement_id: str | None = None,
    better_than_reference: bool | None = None,
) -> BrainResult:
    perception = PerceptionResult(
        measurement_id=sample.measurement_id,
        quality_score=0.92,
        confidence=0.88,
        valid=valid,
        ood=ood,
        defects={"closure": 0.04},
        provider_id=provider_id,
        provider_version="test-1",
        input_hash=sample.raw_eye_density_ref.sha256,
        memory_revision=memory_revision,
        memory_id=memory_id,
        reference_measurement_id=reference_measurement_id,
        better_than_reference=better_than_reference,
    )

    decision = DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction.PROPOSE,
        candidate_id=f"candidate-{sample.measurement_id}",
        requested_params={"bias_ma": 6.5},
        reason_codes=("IMPROVE_MIDDLE_EYE",),
        provider_id=decision_provider_id or provider_id,
        provider_version="test-1",
    )
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=1.0,
        request_hash="b" * 64,
        response_hash="c" * 64,
    )


def _append(
    memory: KimiCampaignMemory,
    sample: EyeSample,
    **result_kwargs: object,
) -> KimiMemoryEntry:
    result = _result(
        sample,
        memory_revision=memory.read_snapshot().revision,
        memory_id=memory.memory_id,
        **result_kwargs,
    )
    return memory.append(sample, result)


def _entry(sample: EyeSample, **result_kwargs: object) -> KimiMemoryEntry:
    result = _result(sample, **result_kwargs)
    return KimiMemoryEntry(
        sample=sample,
        perception=result.perception,
        decision=result.decision,
        recorded_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
    )


def test_missing_campaign_memory_is_empty(tmp_path) -> None:
    memory = KimiCampaignMemory(tmp_path / "missing" / "kimi.jsonl", campaign_id="campaign-001")

    assert memory.load() == ()
    assert memory.best_entry(()) is None


def test_snapshot_revision_is_exact_and_changes_after_append(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)

    empty = memory.read_snapshot()

    assert isinstance(empty, KimiMemorySnapshot)
    assert empty.entries == ()
    assert empty.best_entry is None
    assert empty.recent_summaries == ()
    assert empty.revision == sha256(b"").hexdigest()
    assert empty.memory_id == memory.memory_id
    assert memory.read_snapshot().revision == empty.revision
    sample = _sample(eye_sample, "measurement-001")
    _append(memory, sample)
    recorded = memory.read_snapshot()

    assert recorded.revision == sha256(path.read_bytes()).hexdigest()
    assert recorded.revision != empty.revision
    assert memory.read_snapshot().revision == recorded.revision
    assert recorded.entries[0].sample.measurement_id == "measurement-001"


def test_live_append_rejects_kimi_result_without_memory_revision(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    sample = _sample(eye_sample, "measurement-001")

    with pytest.raises(KimiMemoryRevisionConflict, match="revision"):
        memory.append(sample, _result(sample))

    assert not path.exists()


def test_different_empty_journals_have_different_memory_ids(tmp_path, eye_sample) -> None:
    first = KimiCampaignMemory(
        tmp_path / "first" / "kimi.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    second = KimiCampaignMemory(
        tmp_path / "second" / "kimi.jsonl",
        campaign_id=eye_sample.campaign_id,
    )

    assert first.memory_id != second.memory_id
    assert first.read_snapshot().memory_id == first.memory_id
    assert second.read_snapshot().memory_id == second.memory_id


def test_sequential_appends_record_exact_prefix_revisions(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    first = _append(memory, _sample(eye_sample, "measurement-001"))
    first_raw_line = path.read_bytes()
    second = _append(
        memory,
        _sample(eye_sample, "measurement-002"),
        reference_measurement_id=first.sample.measurement_id,
        better_than_reference=False,
    )

    loaded = memory.load()

    assert first.perception.memory_revision == sha256(b"").hexdigest()
    assert second.perception.memory_revision == sha256(first_raw_line).hexdigest()
    assert loaded == (first, second)


def test_copying_valid_journal_to_another_memory_id_is_rejected(tmp_path, eye_sample) -> None:
    source_path = tmp_path / "source.jsonl"
    source = KimiCampaignMemory(source_path, campaign_id=eye_sample.campaign_id)
    _append(source, _sample(eye_sample, "measurement-001"))
    copied_path = tmp_path / "copied.jsonl"
    copied_path.write_bytes(source_path.read_bytes())
    copied = KimiCampaignMemory(copied_path, campaign_id=eye_sample.campaign_id)

    with pytest.raises(ValueError, match="memory_id"):
        copied.load()


def test_altering_earlier_record_breaks_later_prefix_revision(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    first = _append(memory, _sample(eye_sample, "measurement-001"))
    _append(
        memory,
        _sample(eye_sample, "measurement-002"),
        reference_measurement_id=first.sample.measurement_id,
        better_than_reference=False,
    )
    original = path.read_bytes()
    tampered = original.replace(b'"quality_score":0.92', b'"quality_score":0.91', 1)
    assert tampered != original
    path.write_bytes(tampered)

    with pytest.raises(ValueError, match="revision"):
        memory.read_snapshot()


@pytest.mark.parametrize(
    "missing_fields",
    [("memory_id",), ("memory_revision",), ("memory_id", "memory_revision")],
)
def test_live_load_rejects_entry_missing_memory_provenance(
    tmp_path,
    eye_sample,
    missing_fields,
) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    _append(memory, _sample(eye_sample, "measurement-001"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in missing_fields:
        del payload["perception"][field]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance|JSON"):
        memory.load()


def test_relative_journal_path_is_resolved_once_before_cwd_changes(
    tmp_path,
    eye_sample,
    monkeypatch,
) -> None:
    original_cwd = tmp_path / "original"
    changed_cwd = tmp_path / "changed"
    original_cwd.mkdir()
    changed_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    memory = KimiCampaignMemory(
        "relative/kimi.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    expected_path = (original_cwd / "relative" / "kimi.jsonl").resolve()

    assert memory._path == expected_path
    monkeypatch.chdir(changed_cwd)
    _append(memory, _sample(eye_sample, "measurement-relative"))

    assert expected_path.is_file()
    assert not (changed_cwd / "relative" / "kimi.jsonl").exists()
    assert memory.load()[0].sample.measurement_id == "measurement-relative"


def test_live_append_rejects_result_from_a_different_memory(
    tmp_path,
    eye_sample,
) -> None:
    first = KimiCampaignMemory(
        tmp_path / "first.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    second_path = tmp_path / "second.jsonl"
    second = KimiCampaignMemory(second_path, campaign_id=eye_sample.campaign_id)
    sample = _sample(eye_sample, "measurement-001")
    result = _result(
        sample,
        memory_revision=first.read_snapshot().revision,
        memory_id=first.memory_id,
    )

    with pytest.raises(ValueError, match="memory_id"):
        second.append(sample, result)

    assert not second_path.exists()


def test_result_contract_rejects_memory_revision_without_memory_id(
    tmp_path,
    eye_sample,
) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    sample = _sample(eye_sample, "measurement-001")
    with pytest.raises(ValidationError, match="memory_id.*memory_revision"):
        _result(
            sample,
            memory_revision=memory.read_snapshot().revision,
            memory_id=None,
        )

    assert not path.exists()


def test_read_snapshot_rejects_busy_lock_before_reading(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    path.write_bytes(b"{partial")

    with kimi_memory._KimiJournalLock(path):
        with pytest.raises(ValueError, match="lock"):
            memory.read_snapshot()


def test_campaign_id_is_a_read_only_public_boundary(tmp_path) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id="campaign-001")

    assert memory.campaign_id == "campaign-001"
    assert len(memory.memory_id) == 64


def test_memory_instance_cannot_override_durability_methods(tmp_path) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id="campaign-001")

    with pytest.raises(AttributeError):
        memory.append = lambda sample, result: None

    assert not hasattr(memory, "__dict__")


def test_memory_identity_fields_are_immutable_after_construction(tmp_path) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id="campaign-001")

    for attribute, replacement in (
        ("_path", tmp_path / "redirected.jsonl"),
        ("_campaign_id", "campaign-redirected"),
        ("_memory_id", "f" * 64),
    ):
        with pytest.raises(AttributeError):
            setattr(memory, attribute, replacement)


def test_low_level_coherent_identity_reassignment_is_rejected(tmp_path) -> None:
    original_path = tmp_path / "kimi.jsonl"
    redirected_path = tmp_path / "redirected.jsonl"
    redirected_path_identity = KimiCampaignMemory(
        redirected_path, campaign_id="campaign-001"
    ).memory_id
    redirected_campaign_identity = KimiCampaignMemory(
        original_path, campaign_id="campaign-redirected"
    ).memory_id

    for assignments in (
        (
            ("_path", redirected_path.resolve()),
            ("_memory_id", redirected_path_identity),
        ),
        (
            ("_campaign_id", "campaign-redirected"),
            ("_memory_id", redirected_campaign_identity),
        ),
        ):
        memory = KimiCampaignMemory(original_path, campaign_id="campaign-001")
        identity = kimi_memory._get_kimi_memory_identity(memory)
        assert isinstance(identity, tuple)
        for attribute, replacement in (
            ("path", redirected_path.resolve()),
            ("campaign_id", "campaign-redirected"),
            ("memory_id", "f" * 64),
        ):
            with pytest.raises(AttributeError):
                object.__setattr__(identity, attribute, replacement)
        for attribute, replacement in assignments:
            with pytest.raises(AttributeError):
                object.__setattr__(memory, attribute, replacement)
        assert memory.campaign_id == "campaign-001"
        assert memory.read_snapshot().entries == ()

    assert not original_path.exists()
    assert not redirected_path.exists()


def test_existing_empty_or_blank_journal_is_rejected(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        memory.load()

    path.write_bytes(b"\n")
    with pytest.raises(ValueError, match="blank"):
        memory.load()


def test_first_eligible_entry_becomes_best(tmp_path, eye_sample) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    first = _sample(eye_sample, "measurement-001")

    recorded = _append(memory, first)

    assert recorded.sample.measurement_id == first.measurement_id
    assert memory.best_entry(memory.load()).sample.measurement_id == first.measurement_id


def test_better_replaces_best_and_worse_keeps_it(tmp_path, eye_sample) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    first = _sample(eye_sample, "measurement-001")
    better = _sample(eye_sample, "measurement-002")
    worse = _sample(eye_sample, "measurement-003")

    _append(memory, first)
    _append(
        memory,
        better,
        reference_measurement_id=first.measurement_id,
        better_than_reference=True,
    )
    _append(
        memory,
        worse,
        reference_measurement_id=better.measurement_id,
        better_than_reference=False,
    )

    assert memory.best_entry(memory.load()).sample.measurement_id == better.measurement_id


@pytest.mark.parametrize(
    "ineligible_kwargs",
    [
        {"valid": False},
        {"ood": True},
        {"decision_provider_id": "other-provider"},
    ],
    ids=["invalid", "out-of-distribution", "foreign-decision"],
)
def test_ineligible_entry_cannot_replace_best_and_next_entry_references_current_best(
    tmp_path,
    eye_sample,
    ineligible_kwargs,
) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    first = _sample(eye_sample, "measurement-001")
    ineligible = _sample(eye_sample, "measurement-002")
    next_entry = _sample(eye_sample, "measurement-003")

    _append(memory, first)
    _append(
        memory,
        ineligible,
        reference_measurement_id=first.measurement_id,
        better_than_reference=True,
        **ineligible_kwargs,
    )
    _append(
        memory,
        next_entry,
        reference_measurement_id=first.measurement_id,
        better_than_reference=False,
    )

    assert memory.best_entry(memory.load()).sample.measurement_id == first.measurement_id


def test_public_best_and_summaries_reject_foreign_campaign_entries(tmp_path, eye_sample) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    foreign = _entry(_sample(eye_sample, "measurement-foreign", "campaign-foreign"))

    with pytest.raises(ValueError, match="campaign"):
        memory.best_entry((foreign,))
    with pytest.raises(ValueError, match="campaign"):
        memory.recent_summaries((foreign,))


def test_recent_summaries_are_bounded_and_do_not_include_samples(tmp_path, eye_sample) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    first = _sample(eye_sample, "measurement-001")
    _append(memory, first)
    for index in range(2, 7):
        sample = _sample(eye_sample, f"measurement-{index:03d}")
        _append(
            memory,
            sample,
            reference_measurement_id=first.measurement_id,
            better_than_reference=False,
        )

    summaries = memory.recent_summaries(memory.load())

    assert [summary["measurement_id"] for summary in summaries] == [
        "measurement-002",
        "measurement-003",
        "measurement-004",
        "measurement-005",
        "measurement-006",
    ]
    assert summaries[-1] == {
        "measurement_id": "measurement-006",
        "quality_score": 0.92,
        "confidence": 0.88,
        "valid": True,
        "ood": False,
        "reference_measurement_id": "measurement-001",
        "better_than_reference": False,
        "action": "PROPOSE",
        "candidate_id": "candidate-measurement-006",
        "reason_codes": ("IMPROVE_MIDDLE_EYE",),
        "readback_params": {"bias_ma": 6.01},
        "metrics": {
            "tdecq_db": 1.25,
            "outer_er_chan1a_db": 4.1,
            "rlm": 0.98,
            "outer_oma": 0.72,
            "other_metrics": {"eye_height": 0.31},
        },
    }
    assert "sample" not in summaries[-1]
    assert "raw_eye_density_ref" not in summaries[-1]


def test_recent_summaries_accept_zero_but_reject_boolean_limits(tmp_path, eye_sample) -> None:
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    foreign = _entry(_sample(eye_sample, "measurement-foreign", "campaign-foreign"))

    assert memory.recent_summaries((), limit=0) == ()
    with pytest.raises(ValueError, match="campaign"):
        memory.recent_summaries((foreign,), limit=0)
    with pytest.raises(ValueError, match="limit"):
        memory.recent_summaries((), limit=True)
    with pytest.raises(ValueError, match="limit"):
        memory.recent_summaries((), limit=6)


def test_load_rejects_corrupt_json_and_blank_interior_line(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)

    with pytest.raises(ValueError, match="JSON"):
        memory.load()

    path.write_text(json.dumps(_entry(eye_sample).model_dump(mode="json")) + "\n\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank"):
        memory.load()


def test_unterminated_journal_is_rejected_without_append_modification(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    first = _entry(_sample(eye_sample, "measurement-001"))
    path.write_bytes(json.dumps(first.model_dump(mode="json")).encode("utf-8"))
    original = path.read_bytes()
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    second = _sample(eye_sample, "measurement-002")

    with pytest.raises(ValueError, match="LF"):
        memory.load()
    with pytest.raises(ValueError, match="LF"):
        _append(
            memory,
            second,
            reference_measurement_id=first.sample.measurement_id,
            better_than_reference=False,
        )
    assert path.read_bytes() == original


def test_load_rejects_cross_campaign_and_duplicate_measurements(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    foreign = _entry(_sample(eye_sample, "measurement-foreign", "campaign-foreign"))
    path.write_text(json.dumps(foreign.model_dump(mode="json")) + "\n", encoding="utf-8")
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)

    with pytest.raises(ValueError, match="campaign"):
        memory.load()

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_memory = KimiCampaignMemory(
        duplicate_path,
        campaign_id=eye_sample.campaign_id,
    )
    _append(duplicate_memory, _sample(eye_sample, "measurement-001"))
    first_raw_line = duplicate_path.read_bytes()
    duplicate_path.write_bytes(first_raw_line + first_raw_line)
    with pytest.raises(ValueError, match="duplicate"):
        duplicate_memory.load()


def test_load_rejects_stale_reference_chain(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    first = _append(memory, _sample(eye_sample, "measurement-001"))
    _append(
        memory,
        _sample(eye_sample, "measurement-002"),
        reference_measurement_id=first.sample.measurement_id,
        better_than_reference=False,
    )
    journal_bytes = path.read_bytes().replace(
        b'"reference_measurement_id":"measurement-001"',
        b'"reference_measurement_id":"measurement-not-best"',
        1,
    )
    path.write_bytes(journal_bytes)

    with pytest.raises(ValueError, match="reference"):
        memory.load()


def test_entry_rejects_mismatched_ids_hash_and_naive_timestamp(eye_sample) -> None:
    result = _result(eye_sample)

    with pytest.raises(ValidationError, match="measurement_id"):
        KimiMemoryEntry(
            sample=eye_sample,
            perception=result.perception.model_copy(update={"measurement_id": "other"}),
            decision=result.decision,
            recorded_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
        )
    with pytest.raises(ValidationError, match="input_hash"):
        KimiMemoryEntry(
            sample=eye_sample,
            perception=result.perception.model_copy(update={"input_hash": "d" * 64}),
            decision=result.decision,
            recorded_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
        )
    with pytest.raises(ValidationError, match="decision"):
        KimiMemoryEntry(
            sample=eye_sample,
            perception=result.perception,
            decision=result.decision.model_copy(
                update={"based_on_measurement_id": "other"}
            ),
            recorded_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
        )
    with pytest.raises(ValidationError, match="timezone"):
        KimiMemoryEntry(
            sample=eye_sample,
            perception=result.perception,
            decision=result.decision,
            recorded_at=datetime(2026, 8, 10, 8, 0),
        )


def test_append_rejects_non_kimi_duplicate_and_stale_entries_before_writing(tmp_path, eye_sample) -> None:
    path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    first = _sample(eye_sample, "measurement-001")
    _append(memory, first)

    non_kimi = _sample(eye_sample, "measurement-non-kimi")
    with pytest.raises(ValueError, match="Kimi"):
        _append(memory, non_kimi, provider_id="other-provider")
    with pytest.raises(ValueError, match="duplicate"):
        _append(memory, first)

    stale = _sample(eye_sample, "measurement-stale")
    with pytest.raises(ValueError, match="reference"):
        _append(
            memory,
            stale,
            reference_measurement_id="measurement-not-best",
            better_than_reference=False,
        )
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_round_trips_through_strict_load(tmp_path, eye_sample) -> None:
    path = tmp_path / "nested" / "kimi.jsonl"
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    sample = _sample(eye_sample, "measurement-001")

    entry = _append(memory, sample)
    loaded = memory.load()

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert loaded == (entry,)


def test_concurrent_duplicate_append_serializes_before_journal_write(
    tmp_path,
    eye_sample,
    monkeypatch,
) -> None:
    path = tmp_path / "kimi.jsonl"
    first_memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    second_memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)
    sample = _sample(eye_sample, "measurement-001")
    entered_fsync = threading.Event()
    release_fsync = threading.Event()
    original_fsync = kimi_memory.os.fsync
    outcomes: list[object] = []

    def delayed_fsync(file_descriptor: int) -> None:
        entered_fsync.set()
        assert release_fsync.wait(timeout=2)
        original_fsync(file_descriptor)

    def append(memory: KimiCampaignMemory) -> None:
        try:
            outcomes.append(_append(memory, sample))
        except ValueError as error:
            outcomes.append(error)

    monkeypatch.setattr(kimi_memory.os, "fsync", delayed_fsync)
    first_thread = threading.Thread(target=append, args=(first_memory,))
    second_thread = threading.Thread(target=append, args=(second_memory,))
    first_thread.start()
    assert entered_fsync.wait(timeout=2)
    second_thread.start()
    second_thread.join(timeout=2)
    release_fsync.set()
    first_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert sum(isinstance(outcome, KimiMemoryEntry) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    assert (memory_error := next(outcome for outcome in outcomes if isinstance(outcome, ValueError)))
    assert "lock" in str(memory_error).lower()
    assert second_memory.load()[0].sample.measurement_id == sample.measurement_id
