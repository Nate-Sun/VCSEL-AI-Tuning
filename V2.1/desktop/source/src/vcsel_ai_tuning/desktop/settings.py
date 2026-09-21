"""Persist the three approved bench endpoints and nothing else."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

from .models import BenchSettings


class DesktopSettingsError(RuntimeError):
    """The local desktop settings cannot be trusted or persisted."""


class BenchSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> BenchSettings | None:
        if not self.path.exists():
            return None
        try:
            return BenchSettings.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError, ValueError) as error:
            raise DesktopSettingsError(
                f"desktop settings are invalid: {self.path}"
            ) from error

    def save(self, value: BenchSettings) -> None:
        if type(value) is not BenchSettings:
            raise TypeError("value must be BenchSettings")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(value.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise DesktopSettingsError(
                f"desktop settings could not be saved: {self.path}"
            ) from error
