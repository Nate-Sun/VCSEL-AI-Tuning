"""Filesystem defaults owned exclusively by the V1.1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value


@dataclass(frozen=True, slots=True)
class V11StorageConfig:
    """Independent persistence roots; V1 paths are never fallback values."""

    sqlite_path: Path = Path("state/v1.1/campaigns.sqlite3")
    artifact_root: Path = Path("artifacts/v1.1")

    def resolved(self, base: Path) -> "V11StorageConfig":
        return V11StorageConfig(
            sqlite_path=_resolve(base, self.sqlite_path),
            artifact_root=_resolve(base, self.artifact_root),
        )
