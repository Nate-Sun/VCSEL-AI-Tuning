"""Audited FlexDCA dataset queries for an already completed acquisition.

The adapter never starts, clears, or resumes an acquisition.  It temporarily
selects waveform export sources, reads their color-grade hit matrices, and
restores the original export source before returning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
import re
from typing import Any, Protocol

import numpy as np

from .dataset import DatasetCaptureError, FrozenDatasetCapture, encode_density_png


class BinaryQuerySession(Protocol):
    def write(self, command: str) -> object: ...

    def query(self, command: str) -> str: ...

    def query_binary_values(self, command: str, **kwargs: object) -> object: ...


class _UnsafeSourceReadback(RuntimeError):
    def __init__(self, message: str, *, raw_source: str) -> None:
        super().__init__(message)
        self.raw_source = raw_source


class _GuardSourceRestoreError(DatasetCaptureError):
    def __init__(self, message: str, *, guard_snapshot: Mapping[str, object]) -> None:
        super().__init__(
            message,
            raw_responses={"acquisition_guard": guard_snapshot},
        )
        self.guard_snapshot = self.raw_responses["acquisition_guard"]


class _WaveformSourceRestoreError(DatasetCaptureError):
    def __init__(
        self,
        message: str,
        *,
        capture_evidence: Mapping[str, object],
    ) -> None:
        super().__init__(message, raw_responses=capture_evidence)
        self.capture_evidence = self.raw_responses


def _snapshot_evidence(
    value: object,
    *,
    path: str = "primary_measurement",
    active: set[int] | None = None,
) -> object:
    """Build detached JSON-like evidence without invoking ``deepcopy``."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must not contain nonfinite numbers")
        return value
    if active is None:
        active = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain recursive mappings")
        active.add(identity)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} mapping keys must be text")
                result[key] = _snapshot_evidence(
                    item,
                    path=f"{path}.{key}",
                    active=active,
                )
            return result
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain recursive sequences")
        active.add(identity)
        try:
            return [
                _snapshot_evidence(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise TypeError(
        f"{path} contains unsupported evidence type {type(value).__name__}"
    )


def _response(session: BinaryQuerySession, command: str) -> str:
    value = session.query(command)
    if not isinstance(value, str):
        raise RuntimeError(f"FlexDCA query {command!r} did not return text")
    return value.strip()


def _positive_int(value: str, label: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise RuntimeError(f"FlexDCA {label} is not an integer: {value!r}") from error
    if result <= 0:
        raise RuntimeError(f"FlexDCA {label} must be positive")
    return result


def _finite_metric(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(result) or abs(result) >= 9.0e37:
        return None
    return result


def _canonical_waveform_source(
    source: str,
    *,
    allow_waveform_memory: bool = False,
) -> str:
    """Return a safe SCPI token for a supported FlexDCA waveform source."""
    if not isinstance(source, str):
        raise ValueError("source readback must be text")
    text = source.strip()
    patterns = [
        (
            r"CHAN(?:NEL)?([1-8])([A-D])",
            lambda match: f"CHAN{match.group(1)}{match.group(2).upper()}",
        ),
        (
            r"FUNC(?:TION)?([0-9]+)",
            lambda match: f"FUNC{int(match.group(1), 10)}",
        ),
        (r"DIFF([1-4])", lambda match: f"DIFF{match.group(1)}"),
        (
            r"CGM(?:EMORY)?([1-8])",
            lambda match: f"CGMEMORY{match.group(1)}",
        ),
    ]
    if allow_waveform_memory:
        patterns.append(
            (
                r"WMEM(?:ORY)?([1-8])",
                lambda match: f"WMEM{match.group(1)}",
            )
        )
    for pattern, formatter in patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        canonical = formatter(match)
        if canonical.startswith("FUNC") and not 1 <= int(canonical[4:]) <= 16:
            break
        return canonical
    raise ValueError("source readback must identify a supported DCA waveform")


def _source_readback(
    session: BinaryQuerySession,
    command: str,
    *,
    allow_waveform_memory: bool = False,
) -> tuple[str, str]:
    raw = session.query(command)
    if not isinstance(raw, str):
        raise RuntimeError(f"FlexDCA query {command!r} did not return text")
    try:
        canonical = _canonical_waveform_source(
            raw,
            allow_waveform_memory=allow_waveform_memory,
        )
    except ValueError as error:
        source_error = _UnsafeSourceReadback(
            (
                f"FlexDCA {command!r} returned an unsupported waveform source: "
                f"{raw!r}"
            ),
            raw_source=raw,
        )
        raise source_error from error
    return canonical, raw


class FlexDcaDatasetCapture:
    """Capture raw CHAN1A and TDECQ-function eye evidence without reacquiring."""

    def __init__(
        self,
        *,
        session: BinaryQuerySession,
        raw_source: str,
        equalized_source: str,
        equalizer_function: int,
        exclusive_access: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise ValueError("raw_source must be nonblank")
        if not isinstance(equalized_source, str) or not equalized_source.strip():
            raise ValueError("equalized_source must be nonblank")
        if (
            isinstance(equalizer_function, bool)
            or not isinstance(equalizer_function, int)
            or not 1 <= equalizer_function <= 16
        ):
            raise ValueError("equalizer_function must be in [1, 16]")
        query_binary = getattr(session, "query_binary_values", None)
        if not callable(query_binary):
            raise TypeError("FlexDCA dataset capture requires binary VISA queries")
        if exclusive_access is not None and not callable(exclusive_access):
            raise TypeError("exclusive_access must be callable when provided")
        self._session = session
        self._raw_source = _canonical_waveform_source(raw_source)
        self._equalized_source = _canonical_waveform_source(equalized_source)
        self._function = equalizer_function
        self._exclusive_access = exclusive_access

    def _sample_exclusive_access(
        self,
        label: str,
        errors: dict[str, str],
    ) -> bool:
        if self._exclusive_access is None:
            return False
        try:
            return self._exclusive_access() is True
        except Exception as error:
            errors[label] = str(error)
            return False

    def _read_density(
        self,
        source: str,
        *,
        is_big_endian: bool,
    ) -> tuple[np.ndarray, dict[str, object]]:
        self._session.write(f":WAVeform:SOURce {source}")
        verified = _response(self._session, ":WAVeform:SOURce?").upper()
        if verified != source:
            raise RuntimeError(
                f"FlexDCA waveform source readback {verified!r} does not match {source}"
            )
        status = _response(self._session, ":WAVeform:EYE:STATus?").upper()
        reason = _response(self._session, ":WAVeform:EYE:STATus:REASon?")
        if status != "CORR":
            raise RuntimeError(
                f"FlexDCA {source} eye waveform status is {status or '<empty>'}: "
                f"{reason}"
            )
        rows = _positive_int(
            _response(self._session, ":WAVeform:EYE:ROWS?"), "eye row count"
        )
        columns = _positive_int(
            _response(self._session, ":WAVeform:EYE:COLumns?"),
            "eye column count",
        )
        points = _positive_int(
            _response(self._session, ":WAVeform:EYE:POINts?"),
            "eye point count",
        )
        if points != rows * columns:
            raise RuntimeError(
                "FlexDCA eye dimensions do not match the reported point count"
            )
        scaling = {
            command: _response(self._session, command)
            for command in (
                ":WAVeform:EYE:XINCrement?",
                ":WAVeform:EYE:XORigin?",
                ":WAVeform:EYE:YINCrement?",
                ":WAVeform:EYE:YORigin?",
            )
        }
        values = self._session.query_binary_values(
            ":WAVeform:EYE:INTeger:DATa?",
            datatype="L",
            container=list,
            is_big_endian=is_big_endian,
            header_fmt="ieee",
        )
        array = np.asarray(values, dtype=np.uint32)
        if array.ndim != 1 or array.size != points:
            raise RuntimeError(
                f"FlexDCA {source} returned {array.size} eye points, expected {points}"
            )
        # FlexDCA serializes complete amplitude columns from left to right.
        # order="F" decodes that transport order into [amplitude, time]
        # without display rotation, flipping, or synthetic augmentation.
        density = array.reshape((rows, columns), order="F")
        return density, {
            "source": source,
            "verified_source": verified,
            "status": status,
            "status_reason": reason,
            "rows": rows,
            "columns": columns,
            "points": points,
            "scaling": scaling,
        }

    def _optional_metric(
        self,
        stem: str,
        *,
        raw: dict[str, object],
        units: bool = False,
    ) -> float | None:
        evidence: dict[str, object] = {
            "original_source": None,
            "original_source_raw": None,
            "requested_source": self._raw_source,
            "verified_source": None,
            "status": None,
            "units": None,
            "value": None,
            "source_restore": {
                "expected": None,
                "readback": None,
                "verified": False,
            },
        }
        raw[stem] = evidence
        source_query = f"{stem}:SOURce?"
        try:
            original_source, original_source_raw = _source_readback(
                self._session,
                source_query,
            )
        except _UnsafeSourceReadback as error:
            evidence["original_source_raw"] = error.raw_source
            evidence["error"] = str(error)
            raise
        except Exception as error:
            evidence["error"] = str(error)
            return None

        evidence["original_source"] = original_source
        evidence["original_source_raw"] = original_source_raw
        evidence["source_restore"] = {
            "expected": original_source,
            "readback": None,
            "verified": False,
        }
        metric_error: Exception | None = None
        metric_value: float | None = None
        try:
            self._session.write(f"{stem}:SOURce {self._raw_source}")
            verified_source, _ = _source_readback(self._session, source_query)
            evidence["verified_source"] = verified_source
            if verified_source != self._raw_source:
                raise RuntimeError(
                    f"FlexDCA {stem} source readback {verified_source!r} does not "
                    f"match requested source {self._raw_source!r}"
                )
            status = _response(self._session, f"{stem}:STATus?").upper()
            unit = _response(self._session, f"{stem}:UNITs?") if units else None
            value_text = _response(self._session, f"{stem}?") if status == "CORR" else ""
            evidence["status"] = status
            evidence["units"] = unit
            evidence["value"] = value_text
            metric_value = _finite_metric(value_text)
        except Exception as error:
            metric_error = error
            evidence["error"] = str(error)
        finally:
            restore_readback: str | None = None
            restore_write_succeeded = False
            restore_error: Exception | None = None
            try:
                self._session.write(f"{stem}:SOURce {original_source}")
                restore_write_succeeded = True
            except Exception as error:
                restore_error = error
            try:
                restore_readback, _ = _source_readback(
                    self._session,
                    source_query,
                )
            except Exception as error:
                restore_error = restore_error or error
            restore_verified = bool(
                restore_write_succeeded
                and restore_readback == original_source
            )
            evidence["source_restore"] = {
                "expected": original_source,
                "readback": restore_readback,
                "verified": restore_verified,
            }
            if not restore_verified:
                detail = f": {restore_error}" if restore_error is not None else ""
                raise DatasetCaptureError(
                    f"FlexDCA {stem} source restore failed: expected "
                    f"{original_source!r}, read {restore_readback!r}{detail}",
                    raw_responses=raw,
                ) from restore_error
        if metric_error is not None:
            return None
        return metric_value

    def _equalizer(self, raw: dict[str, object]) -> tuple[tuple[float, ...] | None, int | None]:
        taps_query = f":SPRocess{self._function}:TEQualizer:TAPS?"
        precursors_query = (
            f":SPRocess{self._function}:TEQualizer:NPRecursors?"
        )
        try:
            taps_text = _response(self._session, taps_query)
            taps = tuple(float(field.strip()) for field in taps_text.split(","))
            if not taps or any(not isfinite(value) for value in taps):
                raise ValueError("equalizer taps are empty or nonfinite")
            main_index = int(_response(self._session, precursors_query), 10)
            if not 0 <= main_index < len(taps):
                raise ValueError("equalizer main index is outside the tap vector")
            raw["equalizer"] = {
                "taps_query": taps_query,
                "precursors_query": precursors_query,
                "taps": list(taps),
                "main_index": main_index,
            }
            return taps, main_index
        except Exception as error:
            raw["equalizer"] = {"error": str(error)}
            return None, None

    def _read_acquisition_guard(
        self,
        expected: Mapping[str, object],
    ) -> dict[str, object]:
        commands = expected.get("commands")
        if not isinstance(commands, Mapping):
            return {
                "commands": {},
                "errors": {"guard": "primary acquisition guard is unavailable"},
                "original_source": None,
                "original_source_raw": None,
                "source_restore": {
                    "expected": None,
                    "readback": None,
                    "verified": False,
                },
            }
        values: dict[str, str] = {}
        errors: dict[str, str] = {}
        original_source: str | None = None
        original_source_raw: str | None = None
        restored_source: str | None = None
        restored_source_raw: str | None = None
        restore_write_succeeded = False
        shared_source_query = ":MEASure:EYE:TDEQ:SOURce?"
        try:
            original_source, original_source_raw = _source_readback(
                self._session,
                shared_source_query,
            )
        except Exception as error:
            errors["original_source"] = str(error)
            value = getattr(error, "raw_source", None)
            if isinstance(value, str):
                original_source_raw = value

        try:
            if original_source is not None:
                for stem, expected_source in (
                    (":MEASure:EYE:OER", self._raw_source),
                    (":MEASure:EYE:TDEQ", self._equalized_source),
                ):
                    source_query = f"{stem}:SOURce?"
                    select_command = f"{stem}:SOURce {expected_source}"
                    try:
                        self._session.write(select_command)
                    except Exception as error:
                        errors[select_command] = str(error)
                    try:
                        verified_source, _ = _source_readback(
                            self._session,
                            source_query,
                        )
                        values[source_query] = verified_source
                        if verified_source != expected_source:
                            errors[source_query] = (
                                f"source readback {verified_source!r} does not "
                                f"match expected source {expected_source!r}"
                            )
                    except Exception as error:
                        errors[source_query] = str(error)
                    for suffix in (":STATus?", ":COUNt?", "?"):
                        command = f"{stem}{suffix}"
                        try:
                            values[command] = _response(self._session, command)
                        except Exception as error:
                            errors[command] = str(error)
        finally:
            if original_source is not None:
                restore_command = (
                    f":MEASure:EYE:TDEQ:SOURce {original_source}"
                )
                try:
                    self._session.write(restore_command)
                    restore_write_succeeded = True
                except Exception as error:
                    errors[f"restore:{restore_command}"] = str(error)
                try:
                    restored_source, restored_source_raw = _source_readback(
                        self._session,
                        shared_source_query,
                    )
                except Exception as error:
                    errors[f"restore:{shared_source_query}"] = str(error)

        restoration_verified = bool(
            restore_write_succeeded
            and original_source is not None
            and restored_source == original_source
        )
        if original_source is not None and not restoration_verified:
            errors.setdefault(
                "source_restore",
                (
                    f"source restore readback {restored_source!r} does not "
                    f"match original source {original_source!r}"
                ),
            )
        result: dict[str, object] = {
            "commands": values,
            "errors": errors,
            "original_source": original_source,
            "original_source_raw": original_source_raw,
            "source_restore": {
                "expected": original_source,
                "readback": restored_source,
                "verified": restoration_verified,
            },
        }
        if restored_source_raw is not None:
            result["source_restore_raw"] = restored_source_raw
        if original_source is not None and not restoration_verified:
            raise _GuardSourceRestoreError(
                (
                    "FlexDCA acquisition guard source restore failed: "
                    f"expected {original_source!r}, read {restored_source!r}"
                ),
                guard_snapshot=result,
            )
        return result

    def _acquisition_snapshot_consistent(
        self,
        primary: Mapping[str, object],
        before: Mapping[str, object],
        after: Mapping[str, object],
    ) -> bool:
        primary_commands = primary.get("commands")
        if (
            primary.get("primary_match") is not True
            or not isinstance(primary_commands, Mapping)
            or primary.get("errors") not in ({}, None)
            or before.get("errors") not in ({}, None)
            or after.get("errors") not in ({}, None)
            or before.get("commands") != primary_commands
            or after.get("commands") != primary_commands
            or not isinstance(primary.get("source_restore"), Mapping)
            or primary["source_restore"].get("verified") is not True
            or not isinstance(before.get("source_restore"), Mapping)
            or before["source_restore"].get("verified") is not True
            or not isinstance(after.get("source_restore"), Mapping)
            or after["source_restore"].get("verified") is not True
        ):
            return False
        restored_sources: list[str] = []
        for snapshot in (primary, before, after):
            original_source = snapshot.get("original_source")
            source_restore = snapshot.get("source_restore")
            if (
                not isinstance(original_source, str)
                or not original_source
                or not isinstance(source_restore, Mapping)
                or source_restore.get("readback") != original_source
                or source_restore.get("expected", original_source)
                != original_source
            ):
                return False
            restored_sources.append(original_source)
        if len(set(restored_sources)) != 1:
            return False
        required = {
            ":MEASure:EYE:TDEQ:SOURce?": self._equalized_source,
            ":MEASure:EYE:TDEQ:STATus?": "CORR",
            ":MEASure:EYE:OER:SOURce?": self._raw_source,
            ":MEASure:EYE:OER:STATus?": "CORR",
        }
        if any(
            str(primary_commands.get(command, "")).upper() != expected
            for command, expected in required.items()
        ):
            return False
        for command in (
            ":MEASure:EYE:TDEQ:COUNt?",
            ":MEASure:EYE:OER:COUNt?",
        ):
            count = _finite_metric(str(primary_commands.get(command, "")))
            if count is None or count < 1:
                return False
        for command in (
            ":MEASure:EYE:TDEQ?",
            ":MEASure:EYE:OER?",
        ):
            if _finite_metric(str(primary_commands.get(command, ""))) is None:
                return False
        return True

    def capture(
        self,
        acquisition_id: str,
        primary_result: Any,
    ) -> FrozenDatasetCapture:
        exclusive_errors: dict[str, str] = {}
        exclusive_before = self._sample_exclusive_access(
            "before",
            exclusive_errors,
        )
        primary_raw = getattr(primary_result, "raw_responses", {}) or {}
        if not isinstance(primary_raw, Mapping):
            raise RuntimeError("primary measurement evidence must be a mapping")
        try:
            primary_measurement_snapshot = _snapshot_evidence(primary_raw)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"primary measurement evidence is invalid: {error}"
            ) from error
        if not isinstance(primary_measurement_snapshot, dict):  # pragma: no cover
            raise RuntimeError("primary measurement evidence must be a mapping")
        primary_measurement = primary_measurement_snapshot
        primary_guard_raw = primary_measurement.get("acquisition_guard")
        if primary_guard_raw is not None and not isinstance(
            primary_guard_raw,
            Mapping,
        ):
            raise RuntimeError(
                "primary measurement acquisition_guard must be a mapping"
            )
        primary_guard = (
            primary_guard_raw
            if isinstance(primary_guard_raw, Mapping)
            else {}
        )
        original_source, original_source_raw = _source_readback(
            self._session,
            ":WAVeform:SOURce?",
            allow_waveform_memory=True,
        )
        border = _response(self._session, ":SYSTem:BORDer?").upper()
        if border.startswith(("BEND", "BIG", "MSB")):
            is_big_endian = True
        elif border.startswith(("LEND", "LIT", "LSB")):
            is_big_endian = False
        else:
            raise RuntimeError(f"unsupported FlexDCA byte order {border!r}")
        guard_before = self._read_acquisition_guard(primary_guard)
        raw: dict[str, object] = {
            "primary_measurement": primary_measurement,
            "byte_order": border,
        }
        exclusive_after = False
        try:
            raw_density, raw_metadata = self._read_density(
                self._raw_source,
                is_big_endian=is_big_endian,
            )
            equalized_density, equalized_metadata = self._read_density(
                self._equalized_source,
                is_big_endian=is_big_endian,
            )
            raw["raw_eye"] = raw_metadata
            raw["tdecq_equalized_eye"] = equalized_metadata
            rlm = self._optional_metric(
                ":MEASure:EYE:PAM:LINearity", raw=raw
            )
            oma = self._optional_metric(
                ":MEASure:EYE:OOMA", raw=raw, units=True
            )
            average_power = self._optional_metric(
                ":MEASure:EYE:APOWer", raw=raw, units=True
            )
            taps, main_index = self._equalizer(raw)
            guard_after = self._read_acquisition_guard(primary_guard)
            acquisition_snapshot_consistent = self._acquisition_snapshot_consistent(
                primary_guard,
                guard_before,
                guard_after,
            )
        finally:
            try:
                restore_evidence: dict[str, object] = {
                    "expected": original_source,
                    "original_source_raw": original_source_raw,
                    "readback": None,
                    "verified": False,
                }
                raw["source_restore"] = restore_evidence
                restore_error: Exception | None = None
                write_succeeded = False
                try:
                    self._session.write(f":WAVeform:SOURce {original_source}")
                    write_succeeded = True
                except Exception as error:
                    restore_error = error
                    restore_evidence["write_error"] = str(error)
                try:
                    restored_source = _response(
                        self._session, ":WAVeform:SOURce?"
                    ).upper()
                    restore_evidence["readback"] = restored_source
                except Exception as error:
                    restore_error = restore_error or error
                    restore_evidence["readback_error"] = str(error)
                restore_evidence["verified"] = bool(
                    write_succeeded
                    and restore_evidence["readback"] == original_source
                )
                if restore_evidence["verified"] is not True:
                    raise _WaveformSourceRestoreError(
                        "FlexDCA waveform source restore failed: "
                        f"expected {original_source!r}, read "
                        f"{restore_evidence['readback']!r}",
                        capture_evidence=raw,
                    ) from restore_error
            finally:
                exclusive_after = self._sample_exclusive_access(
                    "after",
                    exclusive_errors,
                )

        waveform_sources_verified = bool(
            raw_metadata.get("verified_source") == self._raw_source
            and equalized_metadata.get("verified_source") == self._equalized_source
        )
        waveform_restore_verified = bool(
            isinstance(raw.get("source_restore"), Mapping)
            and raw["source_restore"].get("verified") is True
        )
        no_acquisition_command_issued = True
        verified = bool(
            exclusive_before
            and exclusive_after
            and acquisition_snapshot_consistent
            and waveform_sources_verified
            and waveform_restore_verified
            and no_acquisition_command_issued
        )
        raw["same_acquisition"] = {
            "method": "source_bound_frozen_snapshot",
            "isolation": "in_process_resource_scope",
            "cross_process_limitation": (
                "does_not_prevent_other_processes_or_hosts_from_accessing_instrument"
            ),
            "primary": primary_guard,
            "before_dataset_queries": guard_before,
            "after_dataset_queries": guard_after,
            "consistent": acquisition_snapshot_consistent,
            "verified": verified,
            "exclusive_access": {
                "before": exclusive_before,
                "after": exclusive_after,
                "errors": exclusive_errors,
            },
            "no_acquisition_command_issued": no_acquisition_command_issued,
        }

        return FrozenDatasetCapture(
            acquisition_id=acquisition_id,
            same_acquisition_verified=verified,
            raw_eye_density=raw_density,
            density_axis_order=("amplitude", "time"),
            density_source=self._raw_source,
            tdecq_equalized_eye_png=encode_density_png(equalized_density),
            rlm=rlm,
            oma=oma,
            average_power=average_power,
            equalizer_taps=taps,
            equalizer_main_index=main_index,
            equalizer_source=self._equalized_source,
            raw_responses=raw,
        )
