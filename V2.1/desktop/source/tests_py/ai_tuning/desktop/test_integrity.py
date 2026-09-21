import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.desktop.integrity import (
    VALIDATED_COMMIT,
    VALIDATED_TAG,
    BuildManifest,
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)


def test_manifest_accepts_exact_protected_files(tmp_path) -> None:
    (tmp_path / "engine.py").write_text("validated", encoding="utf-8")
    manifest = build_manifest(
        tmp_path, [Path("engine.py")], desktop_commit="abc"
    )

    result = verify_manifest(
        tmp_path, manifest, required_paths=[Path("engine.py")]
    )

    assert result.verified is True
    assert result.errors == ()
    assert manifest.validated_tag == VALIDATED_TAG
    assert manifest.validated_commit == VALIDATED_COMMIT


def test_manifest_rejects_modified_and_missing_protected_files(tmp_path) -> None:
    protected = tmp_path / "engine.py"
    protected.write_text("validated", encoding="utf-8")
    manifest = build_manifest(
        tmp_path, [Path("engine.py")], desktop_commit="abc"
    )

    protected.write_text("modified", encoding="utf-8")
    modified = verify_manifest(tmp_path, manifest)
    protected.unlink()
    missing = verify_manifest(tmp_path, manifest)

    assert modified.verified is False
    assert "engine.py" in modified.errors[0]
    assert "hash" in modified.errors[0]
    assert missing.verified is False
    assert "missing" in missing.errors[0]


def test_required_protected_path_set_must_match_manifest(tmp_path) -> None:
    (tmp_path / "engine.py").write_text("validated", encoding="utf-8")
    manifest = build_manifest(
        tmp_path, [Path("engine.py")], desktop_commit="abc"
    )

    result = verify_manifest(
        tmp_path,
        manifest,
        required_paths=[Path("engine.py"), Path("restore.py")],
    )

    assert result.verified is False
    assert "restore.py" in " ".join(result.errors)


@pytest.mark.parametrize(
    "unsafe",
    [Path("../engine.py"), Path("folder/../../engine.py"), Path("C:/engine.py")],
)
def test_manifest_builder_rejects_unsafe_paths(tmp_path, unsafe: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        build_manifest(tmp_path, [unsafe], desktop_commit="abc")


def test_manifest_round_trip_is_strict_and_atomic(tmp_path) -> None:
    (tmp_path / "engine.py").write_text("validated", encoding="utf-8")
    manifest = build_manifest(
        tmp_path, [Path("engine.py")], desktop_commit="abc"
    )
    output = tmp_path / "nested" / "build-manifest.json"

    write_manifest(output, manifest)

    assert load_manifest(output) == manifest
    assert not output.with_suffix(".tmp").exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["validated_commit"] = "wrong"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_manifest(output)


def test_manifest_schema_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(
            {
                "schema_version": 1,
                "validated_tag": VALIDATED_TAG,
                "validated_commit": VALIDATED_COMMIT,
                "desktop_commit": "abc",
                "files": {},
                "allow_hardware_anyway": True,
            }
        )
