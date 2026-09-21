from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from threading import Event

import pytest

from vcsel_ai_tuning.campaign.repository import (
    ArtifactSpec,
    CampaignRepository,
    DuplicateObservationError,
    ImmutableArtifactError,
    RepositoryClosedError,
    RepositoryError,
    RepositoryIntegrityError,
    RepositorySerializationError,
    _validate_identifier,
    _validate_relative_path,
)


CATEGORIES = {
    "config",
    "plans",
    "receipts",
    "events",
    "observations",
    "raw_responses",
    "eye_artifacts",
    "optimizer_snapshots",
    "exports",
    "reports",
}


def _artifacts() -> tuple[ArtifactSpec, ...]:
    return (
        ArtifactSpec.from_bytes(
            "raw_responses",
            "scope.bin",
            b"raw-response",
            media_type="application/octet-stream",
        ),
        ArtifactSpec.from_bytes(
            "eye_artifacts",
            "eye.png",
            b"fake-png",
            media_type="image/png",
        ),
    )


def _published_paths(repo: CampaignRepository) -> set[str]:
    with sqlite3.connect(repo.database_path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT relative_path FROM artifacts ORDER BY relative_path"
            )
        }


def _make_directory_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def test_repository_creates_explicit_run_layout_and_private_index(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        assert repo.run_dir == tmp_path / "campaign-1"
        assert {path.name for path in repo.run_dir.iterdir() if path.is_dir()} == (
            CATEGORIES | {"staging", "quarantine"}
        )
        assert repo.database_path.is_file()


def test_json_write_is_stable_utf8_hashed_and_manifested(tmp_path):
    expected = '{"a":"中文","z":[2,3]}'.encode()

    with CampaignRepository(tmp_path, "campaign-1") as repo:
        record = repo.write_json(
            "config", "immutable.json", {"z": [2, 3], "a": "中文"}
        )

        assert (repo.run_dir / Path(record.relative_path)).read_bytes() == expected
        assert record.relative_path == "config/immutable.json"
        assert record.sha256 == hashlib.sha256(expected).hexdigest()
        assert record.size == len(expected)
        assert record.category == "config"
        assert record.media_type == "application/json"
        assert _published_paths(repo) == {"config/immutable.json"}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_rejected_without_file_or_temp_residue(tmp_path, invalid):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        with pytest.raises(RepositorySerializationError, match="finite JSON"):
            repo.write_json("config", "bad.json", {"metric": invalid})

        assert list((repo.run_dir / "config").iterdir()) == []
        assert _published_paths(repo) == set()


def test_unserializable_json_is_rejected_without_residue(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        with pytest.raises(RepositorySerializationError, match="finite JSON"):
            repo.write_json("config", "bad.json", {"bad": object()})

        assert list((repo.run_dir / "config").iterdir()) == []


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.json", "/absolute.json", r"C:\escape.json", r"nested\escape.json", "a/../b.json", "", "."],
)
def test_artifact_paths_reject_absolute_traversal_and_windows_separators(
    tmp_path, relative_path
):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        with pytest.raises(RepositoryError, match="relative path"):
            repo.write_bytes("config", relative_path, b"unsafe")

        assert list((repo.run_dir / "config").iterdir()) == []


def test_unknown_category_is_rejected(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        with pytest.raises(RepositoryError, match="category"):
            repo.write_bytes("live_hardware", "command.bin", b"forbidden")


def test_immutable_write_is_idempotent_only_for_identical_content(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        first = repo.write_text("plans", "plan.txt", "same")
        repeated = repo.write_text("plans", "plan.txt", "same")

        assert repeated == first
        with pytest.raises(ImmutableArtifactError, match="immutable"):
            repo.write_text("plans", "plan.txt", "different")
        assert (repo.run_dir / "plans" / "plan.txt").read_text() == "same"


def test_immutable_write_promotes_a_unique_temp_from_target_directory(
    tmp_path, monkeypatch
):
    replacements: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def record_replace(source: Path, target: Path) -> Path:
        replacements.append((source, Path(target)))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", record_replace)
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        repo.write_bytes("receipts", "receipt.bin", b"receipt")

    source, target = replacements[0]
    assert source.parent == target.parent
    assert source.name != target.name
    assert re.fullmatch(r"\.stage-[0-9a-f]{32}\.tmp", source.name)
    assert target.name == "receipt.bin"


def test_publish_observation_becomes_queryable_only_after_commit_and_hashes_all_files(
    tmp_path, monkeypatch
):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        original_commit = repo._commit_transaction
        visibility_before_commit: list[object] = []

        def observe_then_commit(connection: sqlite3.Connection) -> None:
            visibility_before_commit.append(repo.get_observation("obs-1"))
            original_commit(connection)

        monkeypatch.setattr(repo, "_commit_transaction", observe_then_commit)
        published = repo.publish_observation(
            "obs-1", {"metric": 1.25, "valid": True}, _artifacts()
        )

        assert visibility_before_commit == [None]
        assert repo.get_observation("obs-1") == published
        assert repo.list_observations() == [published]
        assert published.payload == {"metric": 1.25, "valid": True}
        assert {record.category for record in published.artifacts} == {
            "observations",
            "raw_responses",
            "eye_artifacts",
        }
        for record in published.artifacts:
            data = (repo.run_dir / Path(record.relative_path)).read_bytes()
            assert record.sha256 == hashlib.sha256(data).hexdigest()
            assert record.size == len(data)
        assert _published_paths(repo) == {
            record.relative_path for record in published.artifacts
        }


def test_observation_commit_then_error_is_confirmed_as_success(tmp_path, monkeypatch):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        original_commit = repo._commit_transaction

        def commit_then_raise(connection: sqlite3.Connection) -> None:
            original_commit(connection)
            raise sqlite3.OperationalError("commit acknowledgement lost")

        monkeypatch.setattr(repo, "_commit_transaction", commit_then_raise)
        published = repo.publish_observation(
            "obs-1", {"metric": 1.0}, _artifacts()
        )

        assert repo.get_observation("obs-1") == published
        assert all(
            (repo.run_dir / Path(record.relative_path)).is_file()
            for record in published.artifacts
        )
        assert list((repo.run_dir / "quarantine").iterdir()) == []


def test_direct_artifact_commit_then_error_is_confirmed_as_success(
    tmp_path, monkeypatch
):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        original_commit = repo._commit_transaction

        def commit_then_raise(connection: sqlite3.Connection) -> None:
            original_commit(connection)
            raise sqlite3.OperationalError("commit acknowledgement lost")

        monkeypatch.setattr(repo, "_commit_transaction", commit_then_raise)
        record = repo.write_json("config", "settings.json", {"z": 1, "a": 2})

        assert repo.get_artifact(record.relative_path) == record
        assert (repo.run_dir / Path(record.relative_path)).is_file()
        assert list((repo.run_dir / "quarantine").iterdir()) == []


def test_unknown_observation_commit_outcome_keeps_formal_files_for_recovery(
    tmp_path, monkeypatch
):
    repo = CampaignRepository(tmp_path, "campaign-1")
    original_commit = repo._commit_transaction

    def commit_then_raise(connection: sqlite3.Connection) -> None:
        original_commit(connection)
        raise sqlite3.OperationalError("commit acknowledgement lost")

    def verification_unavailable(*_args: object) -> bool:
        raise sqlite3.OperationalError("cannot verify commit outcome")

    monkeypatch.setattr(repo, "_commit_transaction", commit_then_raise)
    monkeypatch.setattr(
        repo, "_observation_was_committed", verification_unavailable, raising=False
    )
    with pytest.raises(RepositoryError, match="outcome.*unknown|unknown.*outcome"):
        repo.publish_observation("obs-1", {"metric": 1.0}, _artifacts())

    expected_paths = {
        repo.run_dir / "observations" / "obs-1.json",
        repo.run_dir / "raw_responses" / "scope.bin",
        repo.run_dir / "eye_artifacts" / "eye.png",
    }
    assert all(path.is_file() for path in expected_paths)
    repo.close()

    with CampaignRepository(tmp_path, "campaign-1") as reopened:
        assert reopened.get_observation("obs-1") is not None


def test_query_rejects_database_payload_that_disagrees_with_observation_artifact(
    tmp_path,
):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        repo.publish_observation("obs-1", {"metric": 1.0}, _artifacts())
        tampered_payload = json.dumps(
            {"metric": 999.0}, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with sqlite3.connect(repo.database_path) as connection:
            connection.execute(
                "UPDATE observations SET payload_json = ? WHERE observation_id = ?",
                (tampered_payload, "obs-1"),
            )

        with pytest.raises(RepositoryIntegrityError, match="obs-1|payload"):
            repo.get_observation("obs-1")
        with pytest.raises(RepositoryIntegrityError, match="obs-1|payload"):
            repo.list_observations()


@pytest.mark.parametrize("replacement", ["associated", "unassociated"])
def test_query_rejects_observation_path_that_does_not_match_observation_id(
    tmp_path, replacement
):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        published = repo.publish_observation(
            "obs-1", {"metric": 1.0}, _artifacts()
        )
        if replacement == "associated":
            replacement_path = next(
                record.relative_path
                for record in published.artifacts
                if record.category == "raw_responses"
            )
        else:
            replacement_path = repo.write_json(
                "observations", "unrelated.json", {"metric": 2.0}
            ).relative_path
        with sqlite3.connect(repo.database_path) as connection:
            connection.execute(
                "UPDATE observations SET observation_path = ? "
                "WHERE observation_id = ?",
                (replacement_path, "obs-1"),
            )

        with pytest.raises(RepositoryIntegrityError, match="observation.*path|obs-1"):
            repo.get_observation("obs-1")


def test_promotion_failure_quarantines_staged_and_promoted_evidence(tmp_path, monkeypatch):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        original_replace = Path.replace

        def fail_eye_promotion(source: Path, target: Path) -> Path:
            if Path(target).name == "eye.png":
                raise PermissionError("forced promotion failure")
            return original_replace(source, target)

        monkeypatch.setattr(Path, "replace", fail_eye_promotion)
        with pytest.raises(RepositoryError, match="publish observation.*obs-1"):
            repo.publish_observation("obs-1", {"metric": 1.0}, _artifacts())

        assert repo.get_observation("obs-1") is None
        assert _published_paths(repo) == set()
        assert not (repo.run_dir / "observations" / "obs-1.json").exists()
        assert not (repo.run_dir / "raw_responses" / "scope.bin").exists()
        assert not (repo.run_dir / "eye_artifacts" / "eye.png").exists()
        assert any(path.is_file() for path in (repo.run_dir / "quarantine").iterdir())


def test_sqlite_insert_failure_rolls_back_index_and_quarantines_files(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        with sqlite3.connect(repo.database_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_eye_artifact
                BEFORE INSERT ON artifacts
                WHEN NEW.category = 'eye_artifacts'
                BEGIN
                    SELECT RAISE(ABORT, 'forced artifact insert failure');
                END
                """
            )

        with pytest.raises(RepositoryError, match="publish observation.*obs-1"):
            repo.publish_observation("obs-1", {"metric": 1.0}, _artifacts())

        assert repo.get_observation("obs-1") is None
        assert repo.list_observations() == []
        assert _published_paths(repo) == set()
        assert not (repo.run_dir / "observations" / "obs-1.json").exists()
        assert any(path.is_file() for path in (repo.run_dir / "quarantine").iterdir())


def test_commit_failure_rolls_back_and_quarantines_files(tmp_path, monkeypatch):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        def fail_commit(_connection: sqlite3.Connection) -> None:
            raise sqlite3.OperationalError("forced commit failure")

        monkeypatch.setattr(repo, "_commit_transaction", fail_commit)
        with pytest.raises(RepositoryError, match="publish observation.*obs-1"):
            repo.publish_observation("obs-1", {"metric": 1.0}, _artifacts())

        assert repo.get_observation("obs-1") is None
        assert _published_paths(repo) == set()
        assert not (repo.run_dir / "raw_responses" / "scope.bin").exists()
        assert any(path.is_file() for path in (repo.run_dir / "quarantine").iterdir())


def test_failed_publication_keeps_reused_shared_artifact_indexed_and_present(tmp_path):
    shared_content = b"shared-raw-response"
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        shared = repo.write_bytes(
            "raw_responses", "shared.bin", shared_content
        )
        with sqlite3.connect(repo.database_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_new_eye_artifact
                BEFORE INSERT ON artifacts
                WHEN NEW.category = 'eye_artifacts'
                BEGIN
                    SELECT RAISE(ABORT, 'forced new artifact failure');
                END
                """
            )

        with pytest.raises(RepositoryError, match="publish observation.*obs-1"):
            repo.publish_observation(
                "obs-1",
                {"metric": 1.0},
                (
                    ArtifactSpec.from_bytes(
                        "raw_responses", "shared.bin", shared_content
                    ),
                    ArtifactSpec.from_bytes(
                        "eye_artifacts",
                        "new-eye.png",
                        b"new-eye",
                        media_type="image/png",
                    ),
                ),
            )

        assert repo.get_observation("obs-1") is None
        assert repo.get_artifact(shared.relative_path) == shared
        assert (repo.run_dir / Path(shared.relative_path)).read_bytes() == shared_content
        assert not (repo.run_dir / "observations" / "obs-1.json").exists()
        assert not (repo.run_dir / "eye_artifacts" / "new-eye.png").exists()


def test_recovery_waits_for_inflight_publish_before_scanning(tmp_path):
    promoted = Event()
    allow_commit = Event()

    def publish() -> object:
        with CampaignRepository(tmp_path, "campaign-1") as repo:
            original_commit = repo._commit_transaction

            def pause_before_commit(connection: sqlite3.Connection) -> None:
                promoted.set()
                if not allow_commit.wait(timeout=10):
                    raise TimeoutError("test did not release publisher")
                original_commit(connection)

            repo._commit_transaction = pause_before_commit
            return repo.publish_observation(
                "obs-1", {"metric": 1.0}, _artifacts()
            )

    def recover() -> list[object]:
        with CampaignRepository(tmp_path, "campaign-1") as repo:
            return repo.list_observations()

    with ThreadPoolExecutor(max_workers=2) as executor:
        publish_future = executor.submit(publish)
        assert promoted.wait(timeout=10)
        recovery_future = executor.submit(recover)
        try:
            recovery_future.result(timeout=0.5)
        except FutureTimeoutError:
            recovery_completed_before_commit = False
        else:
            recovery_completed_before_commit = True
        finally:
            allow_commit.set()
        published = publish_future.result(timeout=10)
        recovered = recovery_future.result(timeout=10)

    assert not recovery_completed_before_commit
    assert recovered == [published]
    with CampaignRepository(tmp_path, "campaign-1") as reopened:
        assert reopened.get_observation("obs-1") == published


def test_publish_holds_recovery_lock_before_creating_staged_temp(tmp_path):
    first_stage_created = Event()
    allow_staging_to_continue = Event()

    def publish() -> object:
        with CampaignRepository(tmp_path, "campaign-1") as repo:
            original_stage = repo._stage_artifact

            def pause_after_first_stage(*args: object, **kwargs: object) -> object:
                staged = original_stage(*args, **kwargs)
                if not first_stage_created.is_set():
                    first_stage_created.set()
                    if not allow_staging_to_continue.wait(timeout=10):
                        raise TimeoutError("test did not release staging")
                return staged

            repo._stage_artifact = pause_after_first_stage
            return repo.publish_observation(
                "obs-1", {"metric": 1.0}, _artifacts()
            )

    def recover() -> list[object]:
        with CampaignRepository(tmp_path, "campaign-1") as repo:
            return repo.list_observations()

    with ThreadPoolExecutor(max_workers=2) as executor:
        publish_future = executor.submit(publish)
        assert first_stage_created.wait(timeout=10)
        recovery_future = executor.submit(recover)
        try:
            recovery_future.result(timeout=0.5)
        except FutureTimeoutError:
            recovery_completed_before_stage_release = False
        else:
            recovery_completed_before_stage_release = True
        finally:
            allow_staging_to_continue.set()
        published = publish_future.result(timeout=10)
        recovered = recovery_future.result(timeout=10)

    assert not recovery_completed_before_stage_release
    assert recovered == [published]
    with CampaignRepository(tmp_path, "campaign-1") as reopened:
        assert reopened.get_observation("obs-1") == published
        assert list((reopened.run_dir / "quarantine").iterdir()) == []


def test_duplicate_observation_id_with_different_content_is_rejected(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        original = repo.publish_observation("obs-1", {"metric": 1.0}, _artifacts())

        with pytest.raises(DuplicateObservationError, match="obs-1"):
            repo.publish_observation("obs-1", {"metric": 2.0}, ())

        assert repo.get_observation("obs-1") == original


def test_successful_observation_survives_reopen(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        published = repo.publish_observation(
            "obs-1", {"metric": 1.0}, _artifacts()
        )

    with CampaignRepository(tmp_path, "campaign-1") as reopened:
        assert reopened.get_observation("obs-1") == published
        assert reopened.list_observations() == [published]


def test_publish_uses_fixed_length_staging_name_for_long_continuous_observation(
    tmp_path,
):
    campaign_id = "v21-b-20260817T090138707Z-6a4329e1"
    candidate_id = f"{campaign_id}-r0001-a1-329c6930854c388c"
    observation_id = f"i2c_screening-{candidate_id}-3"
    temporary_names: list[str] = []

    preliminary_target = (
        tmp_path
        / campaign_id
        / "observations"
        / f"{observation_id}.json"
    )
    padding_length = max(
        0,
        240 - len(os.fspath(preliminary_target.resolve(strict=False))) - 1,
    )
    repository_root = (
        tmp_path / ("r" * padding_length) if padding_length else tmp_path
    )
    repository_root.mkdir(exist_ok=True)

    with CampaignRepository(repository_root, campaign_id) as repo:
        target = repo.run_dir / "observations" / f"{observation_id}.json"
        legacy_temporary = target.with_name(
            f".{target.name}.tmp-{'0' * 32}"
        )
        fixed_temporary = target.with_name(f".stage-{'0' * 32}.tmp")

        assert len(os.fspath(legacy_temporary.resolve(strict=False))) > 259
        assert len(os.fspath(fixed_temporary.resolve(strict=False))) <= 259

        original_stage = repo._stage_artifact

        def capture_stage(*args, **kwargs):
            staged = original_stage(*args, **kwargs)
            if staged.temporary is not None:
                temporary_names.append(staged.temporary.name)
            return staged

        repo._stage_artifact = capture_stage
        published = repo.publish_observation(
            observation_id,
            {"metric": 6.39},
            (),
        )

    assert published.observation_id == observation_id
    assert temporary_names
    assert all(
        re.fullmatch(r"\.stage-[0-9a-f]{32}\.tmp", name)
        for name in temporary_names
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH guard")
def test_publish_reports_windows_path_budget_before_staging(tmp_path):
    campaign_id = "campaign-1"
    observation_id = "obs-" + "x" * 120

    with CampaignRepository(tmp_path, campaign_id) as repo:
        target = repo.run_dir / "observations" / f"{observation_id}.json"
        resolved_length = len(str(target.resolve()))
        if resolved_length < 260:
            observation_id += "y" * (260 - resolved_length)
            target = repo.run_dir / "observations" / f"{observation_id}.json"

        assert len(str(target.resolve())) >= 260
        with pytest.raises(
            RepositoryError,
            match=r"artifact path length \d+ exceeds Windows limit 259",
        ):
            repo.publish_observation(observation_id, {"metric": 1.0}, ())


@pytest.mark.parametrize("damage", ["tamper", "missing"])
def test_query_rejects_tampered_or_missing_referenced_artifact(tmp_path, damage):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        published = repo.publish_observation(
            "obs-1", {"metric": 1.0}, _artifacts()
        )
        raw_record = next(
            record for record in published.artifacts if record.category == "raw_responses"
        )
        raw_path = repo.run_dir / Path(raw_record.relative_path)
        if damage == "tamper":
            raw_path.write_bytes(b"tampered")
        else:
            raw_path.unlink()

        with pytest.raises(RepositoryIntegrityError, match="scope.bin"):
            repo.get_observation("obs-1")


def test_stage_rejects_symlinked_parent_without_writing_outside_run(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        link = repo.run_dir / "plans" / "link"
        _make_directory_symlink_or_skip(link, outside)

        with pytest.raises(RepositoryIntegrityError, match="link|symlink|reparse"):
            repo.write_bytes("plans", "link/outside.bin", b"must-stay-contained")

    assert not (outside / "outside.bin").exists()


def test_startup_recovery_does_not_follow_or_ignore_reparse_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-evidence")
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        link = repo.run_dir / "reports" / "linked-outside"
        _make_directory_symlink_or_skip(link, outside)

    try:
        with CampaignRepository(tmp_path, "campaign-1"):
            pass
    except RepositoryIntegrityError:
        safely_handled = True
    else:
        safely_handled = not link.is_symlink()

    assert safely_handled
    assert sentinel.read_bytes() == b"outside-evidence"


def test_startup_recovery_quarantines_temp_and_unindexed_formal_artifact(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        temp = repo.run_dir / "plans" / ".plan.json.tmp-crash"
        orphan = repo.run_dir / "reports" / "orphan.md"
        staged = repo.run_dir / "staging" / "transaction-crash.bin"
        temp.write_bytes(b"temporary")
        orphan.write_bytes(b"orphan")
        staged.write_bytes(b"staged")

    with CampaignRepository(tmp_path, "campaign-1") as reopened:
        assert not temp.exists()
        assert not orphan.exists()
        assert not staged.exists()
        quarantine_contents = [
            path.read_bytes()
            for path in (reopened.run_dir / "quarantine").iterdir()
            if path.is_file()
        ]
        assert {b"temporary", b"orphan", b"staged"} <= set(quarantine_contents)


def test_startup_rejects_index_with_missing_artifact(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        published = repo.publish_observation(
            "obs-1", {"metric": 1.0}, _artifacts()
        )
        raw_record = next(
            record for record in published.artifacts if record.category == "raw_responses"
        )
        (repo.run_dir / Path(raw_record.relative_path)).unlink()

    with pytest.raises(RepositoryIntegrityError, match="missing|scope.bin"):
        with CampaignRepository(tmp_path, "campaign-1"):
            pass


def test_startup_quarantines_tampered_indexed_artifact_then_rejects_index(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        published = repo.publish_observation(
            "obs-1", {"metric": 1.0}, _artifacts()
        )
        raw_record = next(
            record for record in published.artifacts if record.category == "raw_responses"
        )
        raw_path = repo.run_dir / Path(raw_record.relative_path)
        raw_path.write_bytes(b"tampered-at-rest")

    with pytest.raises(RepositoryIntegrityError, match="mismatch|scope.bin"):
        with CampaignRepository(tmp_path, "campaign-1"):
            pass

    quarantine_contents = [
        path.read_bytes()
        for path in (tmp_path / "campaign-1" / "quarantine").iterdir()
        if path.is_file()
    ]
    assert b"tampered-at-rest" in quarantine_contents


def test_indexed_artifact_with_tmp_marker_in_name_survives_reopen(tmp_path):
    with CampaignRepository(tmp_path, "campaign-1") as repo:
        record = repo.write_bytes("plans", ".plan.json.tmp-user", b"user-content")
        artifact_path = repo.run_dir / Path(record.relative_path)

    with CampaignRepository(tmp_path, "campaign-1") as reopened:
        assert reopened.get_artifact(record.relative_path) == record
        assert artifact_path.read_bytes() == b"user-content"


@pytest.mark.parametrize(
    "path",
    ["CON", "nested/con.txt", "PRN.csv", "AUX", "COM1.bin", "lpt9", "plan.", "plan "],
)
def test_relative_path_rejects_windows_reserved_or_trimmed_segments(path):
    with pytest.raises(RepositoryError, match="Windows|reserved|trailing"):
        _validate_relative_path(path)


@pytest.mark.parametrize("identifier", ["CON", "com1.txt", "LPT9", "campaign."])
def test_identifier_rejects_windows_reserved_or_trailing_dot(identifier):
    with pytest.raises(RepositoryError, match="Windows|reserved|trailing"):
        _validate_identifier(identifier, name="campaign_id")


def test_close_is_idempotent_and_closed_repository_rejects_operations(tmp_path):
    repo = CampaignRepository(tmp_path, "campaign-1")
    repo.close()
    repo.close()

    with pytest.raises(RepositoryClosedError, match="closed"):
        repo.list_observations()
