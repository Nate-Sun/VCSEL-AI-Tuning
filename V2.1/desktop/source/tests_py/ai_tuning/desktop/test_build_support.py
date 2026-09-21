from pathlib import Path

import pytest

from vcsel_ai_tuning.desktop.build_support import stage_integrity_bundle
from vcsel_ai_tuning.desktop.integrity import (
    BUNDLED_PROTECTED_PATHS,
    load_manifest,
    verify_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_staging_copies_required_sources_and_builds_verified_manifest(
    tmp_path,
) -> None:
    stage = stage_integrity_bundle(
        REPOSITORY_ROOT,
        tmp_path / "desktop-stage",
        desktop_commit="abc123",
    )
    manifest = load_manifest(stage / "build-manifest.json")

    result = verify_manifest(
        stage,
        manifest,
        required_paths=BUNDLED_PROTECTED_PATHS,
    )

    assert result.verified is True
    assert manifest.desktop_commit == "abc123"
    assert (
        stage
        / "protected"
        / "src"
        / "vcsel_ai_tuning"
        / "v2"
        / "continuous"
        / "controller.py"
    ).is_file()
    assert (
        stage
        / "protected"
        / "src"
        / "vcsel_ai_tuning"
        / "desktop"
        / "session.py"
    ).is_file()


def test_staging_refuses_to_delete_unowned_existing_directory(tmp_path) -> None:
    stage = tmp_path / "desktop-stage"
    stage.mkdir()
    (stage / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not owned"):
        stage_integrity_bundle(REPOSITORY_ROOT, stage, desktop_commit="abc123")

    assert (stage / "user-file.txt").read_text(encoding="utf-8") == "keep"
