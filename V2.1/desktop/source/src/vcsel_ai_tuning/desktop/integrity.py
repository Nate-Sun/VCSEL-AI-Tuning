"""Minimal lineage manifest for fail-closed desktop hardware enablement."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from .models import FrozenModel


VALIDATED_TAG = "v2.1-continuous-eye-b-validated-20260817"
VALIDATED_COMMIT = "4085b007546f820820dbdb7f7f0bba6ebc4b2049"

VALIDATED_PROTECTED_PATHS = tuple(
    Path(value)
    for value in (
        "configs/ai_tuning/b0.v2_1.continuous.yaml",
        "configs/ai_tuning/runtime.v2_1.continuous.live.yaml",
        "configs/ai_tuning/v2_1_continuous.live.yaml",
        "configs/hardware/vcsel_lab_v2_1_continuous.yaml",
        "src/vcsel_ai_tuning/continuous_live.py",
        "src/vcsel_ai_tuning/runtime.py",
        "src/vcsel_ai_tuning/runtime_config.py",
        "src/vcsel_ai_tuning/campaign/authorization.py",
        "src/vcsel_ai_tuning/campaign/models.py",
        "src/vcsel_ai_tuning/campaign/orchestrator.py",
        "src/vcsel_ai_tuning/campaign/repository.py",
        "src/vcsel_ai_tuning/campaign/service.py",
        "src/vcsel_ai_tuning/campaign/state_machine.py",
        "src/vcsel_ai_tuning/campaign/store.py",
        "src/vcsel_ai_tuning/hardware/continuous_eye.py",
        "src/vcsel_ai_tuning/hardware/dca_acquisition.py",
        "src/vcsel_ai_tuning/hardware/fir.py",
        "src/vcsel_ai_tuning/hardware/flexdca.py",
        "src/vcsel_ai_tuning/hardware/live_config.py",
        "src/vcsel_ai_tuning/hardware/live_runtime_factory.py",
        "src/vcsel_ai_tuning/hardware/live_workflow.py",
        "src/vcsel_ai_tuning/hardware/vd1124.py",
        "src/vcsel_ai_tuning/hardware/vit_bert.py",
        "src/vcsel_ai_tuning/v2/contracts.py",
        "src/vcsel_ai_tuning/v2/kimi_memory.py",
        "src/vcsel_ai_tuning/v2/continuous/artifacts.py",
        "src/vcsel_ai_tuning/v2/continuous/controller.py",
        "src/vcsel_ai_tuning/v2/continuous/execution.py",
        "src/vcsel_ai_tuning/v2/continuous/factory.py",
        "src/vcsel_ai_tuning/v2/continuous/models.py",
        "src/vcsel_ai_tuning/v2/continuous/policy.py",
        "src/vcsel_ai_tuning/v2/continuous/provider.py",
        "src/vcsel_ai_tuning/v2/continuous/store.py",
        "src/vcsel_ai_tuning/v2/providers/kimi_client.py",
        "src/vcsel_ai_tuning/v2/providers/kimi_provider.py",
    )
)

DESKTOP_SAFETY_PATHS = tuple(
    Path(value)
    for value in (
        "packaging/v2_1_desktop_launcher.py",
        "src/vcsel_ai_tuning/desktop/app.py",
        "src/vcsel_ai_tuning/desktop/artifacts.py",
        "src/vcsel_ai_tuning/desktop/build_support.py",
        "src/vcsel_ai_tuning/desktop/csv_export.py",
        "src/vcsel_ai_tuning/desktop/integrity.py",
        "src/vcsel_ai_tuning/desktop/main.py",
        "src/vcsel_ai_tuning/desktop/models.py",
        "src/vcsel_ai_tuning/desktop/session.py",
        "src/vcsel_ai_tuning/desktop/session_config.py",
        "src/vcsel_ai_tuning/desktop/settings.py",
    )
)

PROTECTED_REPOSITORY_PATHS = VALIDATED_PROTECTED_PATHS + DESKTOP_SAFETY_PATHS
BUNDLED_PROTECTED_PATHS = tuple(
    Path("protected") / path for path in PROTECTED_REPOSITORY_PATHS
)

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class BuildManifest(FrozenModel):
    schema_version: Literal[1] = 1
    validated_tag: Literal[VALIDATED_TAG] = VALIDATED_TAG
    validated_commit: Literal[VALIDATED_COMMIT] = VALIDATED_COMMIT
    desktop_commit: NonBlank
    files: dict[str, Sha256Hex]


class IntegrityResult(FrozenModel):
    verified: bool
    errors: tuple[str, ...] = ()


def _relative_name(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"protected path must be a safe relative path: {value}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"protected path must be a safe relative path: {value}")
    return normalized


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: str | Path,
    protected_paths: list[Path] | tuple[Path, ...],
    *,
    desktop_commit: str,
) -> BuildManifest:
    base = Path(root).resolve()
    files: dict[str, str] = {}
    for relative in protected_paths:
        name = _relative_name(relative)
        if name in files:
            raise ValueError(f"duplicate protected path: {name}")
        source = base / Path(name)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"protected file is missing or unsafe: {name}")
        files[name] = _digest(source)
    return BuildManifest(desktop_commit=desktop_commit, files=files)


def verify_manifest(
    root: str | Path,
    manifest: BuildManifest,
    *,
    required_paths: list[Path] | tuple[Path, ...] | None = None,
) -> IntegrityResult:
    base = Path(root).resolve()
    errors: list[str] = []
    if required_paths is not None:
        try:
            required = {_relative_name(path) for path in required_paths}
        except ValueError as error:
            errors.append(str(error))
        else:
            present = set(manifest.files)
            for missing in sorted(required - present):
                errors.append(f"protected manifest entry missing: {missing}")
            for unexpected in sorted(present - required):
                errors.append(f"unexpected protected manifest entry: {unexpected}")

    for name, expected in manifest.files.items():
        try:
            normalized = _relative_name(name)
        except ValueError as error:
            errors.append(str(error))
            continue
        source = base / Path(normalized)
        if not source.exists():
            errors.append(f"protected file missing: {normalized}")
        elif not source.is_file() or source.is_symlink():
            errors.append(f"protected file is unsafe: {normalized}")
        elif _digest(source) != expected:
            errors.append(f"protected file hash mismatch: {normalized}")
    return IntegrityResult(verified=not errors, errors=tuple(errors))


def write_manifest(path: str | Path, manifest: BuildManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(manifest.model_dump_json(indent=2))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def load_manifest(path: str | Path) -> BuildManifest:
    return BuildManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
