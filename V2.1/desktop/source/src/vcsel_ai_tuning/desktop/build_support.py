"""Build-only staging for the desktop integrity bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence

from .integrity import (
    BUNDLED_PROTECTED_PATHS,
    PROTECTED_REPOSITORY_PATHS,
    build_manifest,
    write_manifest,
)


_OWNER_MARKER = ".vcsel-desktop-integrity-stage"
_OWNER_VALUE = "VCSEL_CONTINUOUS_EYE_DESKTOP_STAGE_V1\n"


def stage_integrity_bundle(
    repository_root: str | Path,
    stage_root: str | Path,
    *,
    desktop_commit: str,
) -> Path:
    repository = Path(repository_root).resolve()
    stage = Path(stage_root).resolve()
    if stage == repository or stage == stage.parent:
        raise RuntimeError("integrity stage target is unsafe")
    marker = stage / _OWNER_MARKER
    if stage.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8") != _OWNER_VALUE:
            raise RuntimeError(f"existing integrity stage is not owned: {stage}")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    marker.write_text(_OWNER_VALUE, encoding="utf-8", newline="\n")

    for relative in PROTECTED_REPOSITORY_PATHS:
        source = repository / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"protected source is missing or unsafe: {relative}")
        destination = stage / "protected" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest = build_manifest(
        stage,
        list(BUNDLED_PROTECTED_PATHS),
        desktop_commit=desktop_commit,
    )
    write_manifest(stage / "build-manifest.json", manifest)
    return stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--desktop-commit", required=True)
    args = parser.parse_args(argv)
    stage_integrity_bundle(
        args.repository_root,
        args.stage_root,
        desktop_commit=args.desktop_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
