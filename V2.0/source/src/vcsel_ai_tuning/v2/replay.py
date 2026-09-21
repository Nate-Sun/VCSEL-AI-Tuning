"""Pure, hardware-neutral V2 brain replay and JSONL audit output."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unicodedata import category

from pydantic import Field, FiniteFloat, StrictInt, model_validator

from .brain import (
    OptimizationBrain,
    _detached_inputs,
    _proposal_matches_input,
    _request_hash,
    _response_hash,
)
from .contracts import (
    BrainCandidate,
    BrainResult,
    EyeSample,
    FrozenModel,
    Identifier,
    NonBlankText,
    V2Observation,
)


MAX_REPLAY_CASES = 4096
MAX_REPLAY_CANDIDATES = 4096
MAX_REPLAY_HISTORY = 4096


def _safe_identity(value: object) -> str | None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(category(character) in {"Cc", "Cf", "Cs"} for character in value)
    ):
        return None
    if any(
        not component
        or component != component.strip()
        or any(category(character) in {"Cc", "Cf", "Cs"} for character in component)
        for component in value.split("|")
    ):
        return None
    return value


def _validated_model(model_type, value: object):
    """Re-validate a frozen model after model_copy/model_construct escape hatches."""
    return model_type.model_validate(value.model_dump(round_trip=True), strict=True)


class ReplayCase(FrozenModel):
    sample: EyeSample
    history: tuple[V2Observation, ...] = Field(default=(), max_length=MAX_REPLAY_HISTORY)
    safe_candidates: tuple[BrainCandidate, ...] = Field(
        default=(), max_length=MAX_REPLAY_CANDIDATES
    )
    estimated_cost_usd: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "ReplayCase":
        candidate_ids = tuple(candidate.candidate_id for candidate in self.safe_candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("replay candidate IDs must be unique")
        return self


class ReplayRecord(FrozenModel):
    sequence: StrictInt = Field(ge=0)
    campaign_id: Identifier
    sweep_point_id: Identifier
    measurement_id: Identifier
    brain_provider_key: NonBlankText
    brain_provider_version: NonBlankText
    result: BrainResult
    estimated_cost_usd: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_record(self) -> "ReplayRecord":
        if self.result.perception.measurement_id != self.measurement_id:
            raise ValueError("replay result measurement ID mismatch")
        return self


def _brain_identity(brain: object) -> tuple[str, str]:
    try:
        provider_key = brain.provider_key
        provider_version = brain.provider_version
    except Exception as error:
        raise TypeError("brain provider identity could not be read") from error
    safe_key = _safe_identity(provider_key)
    safe_version = _safe_identity(provider_version)
    if safe_key is None:
        raise TypeError("brain provider_key must be a safe nonblank identity")
    if safe_version is None:
        raise TypeError("brain provider_version must be a safe nonblank identity")
    return safe_key, safe_version


def _validated_cases(cases: tuple[ReplayCase, ...]) -> tuple[ReplayCase, ...]:
    if not isinstance(cases, tuple):
        raise TypeError("replay cases must be a tuple")
    if not cases or len(cases) > MAX_REPLAY_CASES:
        raise ValueError(f"replay requires 1..{MAX_REPLAY_CASES} cases")
    validated: list[ReplayCase] = []
    for case in cases:
        if not isinstance(case, ReplayCase):
            raise TypeError("replay cases must contain ReplayCase instances")
        validated.append(_validated_model(ReplayCase, case))
    measurement_ids = tuple(case.sample.measurement_id for case in validated)
    if len(measurement_ids) != len(set(measurement_ids)):
        raise ValueError("replay measurement IDs must be unique")
    return tuple(validated)


def replay(brain: OptimizationBrain, cases: tuple[ReplayCase, ...]) -> tuple[ReplayRecord, ...]:
    """Replay exactly one read-only analysis call per fully pre-validated case."""
    validated_cases = _validated_cases(cases)
    try:
        is_brain = isinstance(brain, OptimizationBrain)
        analyze = brain.analyze
    except Exception as error:
        raise TypeError("brain must implement OptimizationBrain") from error
    if not is_brain or not callable(analyze):
        raise TypeError("brain must implement OptimizationBrain")
    provider_key, provider_version = _brain_identity(brain)
    records: list[ReplayRecord] = []
    for sequence, case in enumerate(validated_cases):
        gate_sample, gate_history, gate_candidates = _detached_inputs(
            case.sample, case.history, case.safe_candidates
        )
        call_sample, call_history, call_candidates = _detached_inputs(
            gate_sample, gate_history, gate_candidates
        )
        result = analyze(call_sample, call_history, call_candidates)
        if not isinstance(result, BrainResult):
            raise TypeError("brain analyze must return BrainResult")
        validated_result = _validated_model(BrainResult, result)
        if validated_result.perception.measurement_id != gate_sample.measurement_id:
            raise ValueError("replay result measurement ID mismatch")
        if validated_result.perception.input_hash != gate_sample.raw_eye_density_ref.sha256:
            raise ValueError("replay result input hash mismatch")
        expected_request_hash = _request_hash(gate_sample, gate_history, gate_candidates)
        if validated_result.request_hash != expected_request_hash:
            raise ValueError("replay result request hash mismatch")
        if validated_result.response_hash != _response_hash(
            validated_result.perception,
            validated_result.decision,
            validated_result.brain_mode,
        ):
            raise ValueError("replay result response hash mismatch")
        if not _proposal_matches_input(validated_result.decision, gate_candidates):
            raise ValueError("replay result candidate not in input set")
        records.append(
            ReplayRecord(
                sequence=sequence,
                campaign_id=gate_sample.campaign_id,
                sweep_point_id=gate_sample.sweep_point_id,
                measurement_id=gate_sample.measurement_id,
                brain_provider_key=provider_key,
                brain_provider_version=provider_version,
                result=validated_result,
                estimated_cost_usd=case.estimated_cost_usd,
            )
        )
    return tuple(records)


def _validated_records(records: tuple[ReplayRecord, ...]) -> tuple[ReplayRecord, ...]:
    if not isinstance(records, tuple):
        raise TypeError("replay records must be a tuple")
    if not records or len(records) > MAX_REPLAY_CASES:
        raise ValueError(f"replay records require 1..{MAX_REPLAY_CASES} entries")
    validated: list[ReplayRecord] = []
    for expected_sequence, record in enumerate(records):
        if not isinstance(record, ReplayRecord):
            raise TypeError("replay records must contain ReplayRecord instances")
        verified = _validated_model(ReplayRecord, record)
        if verified.sequence != expected_sequence:
            raise ValueError("replay record sequence must be contiguous from zero")
        validated.append(verified)
    measurement_ids = tuple(record.measurement_id for record in validated)
    if len(measurement_ids) != len(set(measurement_ids)):
        raise ValueError("replay record measurement IDs must be unique")
    return tuple(validated)


def write_replay_jsonl(records: tuple[ReplayRecord, ...], output: str | Path) -> Path:
    """Write validated replay records atomically, never replacing output on validation failure."""
    validated_records = _validated_records(records)
    lines = tuple(record.model_dump_json() + "\n" for record in validated_records)
    for line, record in zip(lines, validated_records, strict=True):
        restored = ReplayRecord.model_validate_json(line, strict=True)
        if restored != record:
            raise ValueError("replay record JSON round trip changed the record")

    destination = Path(output)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.writelines(lines)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        return destination
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
