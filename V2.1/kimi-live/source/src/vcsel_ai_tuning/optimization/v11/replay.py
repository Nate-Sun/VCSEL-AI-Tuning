from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    Field,
    PrivateAttr,
    StrictInt,
    StringConstraints,
    model_validator,
)

from .coarse import CoarsePlanner
from .fine import FineDecision, build_fine_from_coarse
from .grid import DiscreteParameterSpace, TuningParameter
from .models import (
    AggregatedObservation,
    AlgorithmCandidate,
    ExperimentFingerprint,
    FrozenModel,
    I2CObservation,
    Identifier,
    ImmutableIntMapping,
    MetricResult,
    OptimizerStatus,
)
from .objective import best_observation
from .retune import RetunePlanner


MAX_REPLAY_GRID_POINTS = 4096
MAX_REPLAY_CANDIDATES = 128
MAX_REPLAY_ORACLE_ENTRIES = 128
MAX_REPLAY_OUTCOMES_PER_REQUEST = 8
MAX_REPLAY_INITIAL_HISTORY = 128
MAX_REPLAY_PARAMETERS = 32
MAX_REPLAY_RADIUS_STEPS = 4096
MAX_REPLAY_MODEL_READBACKS = 64
MAX_REPLAY_OBSERVATIONS = (
    MAX_REPLAY_CANDIDATES * MAX_REPLAY_OUTCOMES_PER_REQUEST
)


class ReplayStage(StrEnum):
    COARSE_FINE = "COARSE_FINE"
    RETUNE = "RETUNE"


class ReplayPhase(StrEnum):
    COARSE = "COARSE"
    FINE_BOOTSTRAP = "FINE_BOOTSTRAP"
    FINE = "FINE"
    RETUNE = "RETUNE"


class ReplayOracleOutcome(FrozenModel):
    """One simulated result returned for an exact requested I2C point."""

    readback_i2c: ImmutableIntMapping
    result: MetricResult


class ReplayOracleEntry(FrozenModel):
    requested_i2c: ImmutableIntMapping
    outcomes: tuple[ReplayOracleOutcome, ...] = Field(
        min_length=1,
        max_length=MAX_REPLAY_OUTCOMES_PER_REQUEST,
    )


def _point_key(point: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(point.items()))


class ReplayOracle(FrozenModel):
    """Serializable lookup table; never a callable execution backend."""

    entries: tuple[ReplayOracleEntry, ...] = Field(
        default=(),
        max_length=MAX_REPLAY_ORACLE_ENTRIES,
    )
    _lookup: dict[
        tuple[tuple[str, int], ...],
        tuple[ReplayOracleOutcome, ...],
    ] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        self._lookup.update(
            {
                _point_key(entry.requested_i2c): entry.outcomes
                for entry in self.entries
            }
        )

    @model_validator(mode="after")
    def validate_unique_requests(self) -> Self:
        keys = tuple(_point_key(entry.requested_i2c) for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ValueError("oracle requested_i2c entries must be unique")
        total_outcomes = sum(len(entry.outcomes) for entry in self.entries)
        if total_outcomes > MAX_REPLAY_OBSERVATIONS:
            raise ValueError("oracle total outcomes exceed replay observation bound")
        return self

    def outcomes_for(
        self,
        requested_i2c: Mapping[str, int],
    ) -> tuple[ReplayOracleOutcome, ...] | None:
        return self._lookup.get(_point_key(requested_i2c))


def _validate_readback(
    space: DiscreteParameterSpace,
    readback_i2c: Mapping[str, int],
    *,
    field_name: str,
) -> None:
    if (
        set(readback_i2c) != set(space.names)
        or len(readback_i2c) != len(space.names)
    ):
        raise ValueError(
            f"{field_name} must contain exactly the enabled parameters"
        )
    by_name = {parameter.name: parameter for parameter in space.parameters}
    for name in space.names:
        value = readback_i2c[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field_name} values must be integers")
        parameter = by_name[name]
        if value < parameter.tuning_min or value > parameter.tuning_max:
            raise ValueError(f"{field_name} is outside the tuning bounds")


class ReplaySnapshot(FrozenModel):
    """Complete pure-data input needed to replay a V1.1 algorithm run."""

    schema_version: str = Field(
        default="ai-tuning-algorithm-replay-snapshot-v1.1",
        pattern=r"^ai-tuning-algorithm-replay-snapshot-v1\.1$",
    )
    stage: ReplayStage
    parameters: tuple[TuningParameter, ...] = Field(
        min_length=1,
        max_length=MAX_REPLAY_PARAMETERS,
    )
    seed: StrictInt
    coarse_budget: StrictInt = Field(gt=0, le=MAX_REPLAY_CANDIDATES)
    adaptive_budget: StrictInt = Field(gt=0, le=MAX_REPLAY_CANDIDATES)
    radius_steps: StrictInt = Field(gt=0, le=MAX_REPLAY_RADIUS_STEPS)
    fingerprint: ExperimentFingerprint
    oracle: ReplayOracle
    initial_history: tuple[I2CObservation, ...] = Field(
        default=(),
        max_length=MAX_REPLAY_INITIAL_HISTORY,
    )
    current_readback: ImmutableIntMapping | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.coarse_budget + self.adaptive_budget > MAX_REPLAY_CANDIDATES:
            raise ValueError(
                "combined candidate budget exceeds replay bound"
            )
        space = DiscreteParameterSpace(self.parameters)
        if space.point_count > MAX_REPLAY_GRID_POINTS:
            raise ValueError("parameter space point_count exceeds replay bound")
        potential_model_readbacks: set[tuple[tuple[str, int], ...]] = set()
        for entry in self.oracle.entries:
            try:
                space.index(entry.requested_i2c)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "oracle requested_i2c must be on the parameter grid"
                ) from error
            for outcome in entry.outcomes:
                _validate_readback(
                    space,
                    outcome.readback_i2c,
                    field_name="oracle readback_i2c",
                )
                potential_model_readbacks.add(
                    _point_key(outcome.readback_i2c)
                )
        for observation in self.initial_history:
            try:
                space.index(observation.requested_i2c)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "initial_history requested_i2c must be on the parameter grid"
                ) from error
            _validate_readback(
                space,
                observation.readback_i2c,
                field_name="initial_history readback_i2c",
            )
            if (
                self.stage is ReplayStage.RETUNE
                and observation.fingerprint == self.fingerprint
            ):
                potential_model_readbacks.add(
                    _point_key(observation.readback_i2c)
                )
        if self.current_readback is not None:
            _validate_readback(
                space,
                self.current_readback,
                field_name="current_readback",
            )
        if self.stage is ReplayStage.RETUNE and self.current_readback is None:
            raise ValueError("RETUNE replay requires current_readback")
        if len(potential_model_readbacks) > MAX_REPLAY_MODEL_READBACKS:
            raise ValueError(
                "potential unique model readbacks exceed replay bound"
            )
        return self

    @property
    def parameter_space(self) -> DiscreteParameterSpace:
        return DiscreteParameterSpace(self.parameters)


class ReplayDecision(FrozenModel):
    """Serializable preservation of a planner or optimizer decision."""

    status: OptimizerStatus
    candidate: AlgorithmCandidate | None = None
    best_observation: AggregatedObservation | None = None
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    requested_point_count: StrictInt = Field(ge=0)
    training_point_count: StrictInt = Field(ge=0)
    all_history_count: StrictInt = Field(ge=0)
    selected_history_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.status not in {
            OptimizerStatus.CANDIDATE,
            OptimizerStatus.COMPLETE,
            OptimizerStatus.NO_VALID_TDECQ,
            OptimizerStatus.NOT_READY,
        }:
            raise ValueError("unsupported replay decision status")
        if self.selected_history_count > self.all_history_count:
            raise ValueError(
                "selected_history_count cannot exceed all_history_count"
            )
        if (
            self.status is not OptimizerStatus.CANDIDATE
            and self.candidate is not None
        ):
            raise ValueError("only CANDIDATE may carry a candidate")
        return self


class ReplayTraceStep(FrozenModel):
    execution_index: StrictInt = Field(ge=0)
    phase: ReplayPhase
    decision: ReplayDecision
    observations: tuple[I2CObservation, ...] = Field(
        default=(),
        max_length=MAX_REPLAY_OUTCOMES_PER_REQUEST,
    )

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        candidate = self.decision.candidate
        if (
            self.phase in {ReplayPhase.COARSE, ReplayPhase.FINE, ReplayPhase.RETUNE}
            and self.decision.status is OptimizerStatus.CANDIDATE
            and candidate is None
        ):
            raise ValueError("candidate decision must preserve its candidate")
        if candidate is None and self.observations:
            raise ValueError("observations require a preserved candidate")
        if candidate is not None:
            for observation in self.observations:
                if observation.candidate_id != candidate.candidate_id:
                    raise ValueError(
                        "trace observation candidate_id must match its candidate"
                    )
                if dict(observation.requested_i2c) != dict(
                    candidate.requested_i2c
                ):
                    raise ValueError(
                        "trace observation request must match its candidate"
                    )
        return self


def _canonical_digest(payload: Mapping[str, object]) -> str:
    document = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


@dataclass(frozen=True)
class _ReplaySimulation:
    status: OptimizerStatus
    reason: str
    trace: tuple[ReplayTraceStep, ...]


class ReplayRunReport(FrozenModel):
    """Ordered trace produced by :func:`run_replay`."""

    schema_version: str = Field(
        default="ai-tuning-algorithm-replay-v1.1",
        pattern=r"^ai-tuning-algorithm-replay-v1\.1$",
    )
    status: OptimizerStatus
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    snapshot: ReplaySnapshot
    trace: tuple[ReplayTraceStep, ...] = Field(
        max_length=MAX_REPLAY_CANDIDATES + 1,
    )
    candidate_order: tuple[Identifier, ...] = Field(
        max_length=MAX_REPLAY_CANDIDATES,
    )
    observations: tuple[I2CObservation, ...] = Field(
        max_length=MAX_REPLAY_OBSERVATIONS,
    )
    observation_count: StrictInt = Field(ge=0, le=MAX_REPLAY_OBSERVATIONS)
    invalid_count: StrictInt = Field(ge=0, le=MAX_REPLAY_OBSERVATIONS)
    final_history: tuple[I2CObservation, ...] = Field(
        max_length=MAX_REPLAY_INITIAL_HISTORY + MAX_REPLAY_OBSERVATIONS,
    )
    best_observation: I2CObservation | None
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if tuple(step.execution_index for step in self.trace) != tuple(
            range(len(self.trace))
        ):
            raise ValueError("trace execution_index values must be contiguous")
        expected_candidates = tuple(
            step.decision.candidate.candidate_id
            for step in self.trace
            if step.decision.candidate is not None
        )
        if self.candidate_order != expected_candidates:
            raise ValueError(
                "candidate_order must preserve every candidate in trace order"
            )
        expected_observations = tuple(
            observation
            for step in self.trace
            for observation in step.observations
        )
        if self.observations != expected_observations:
            raise ValueError(
                "observations must preserve every trace observation in order"
            )
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count must equal observations length")
        expected_invalid = sum(
            not row.result.has_valid_tdecq for row in self.observations
        )
        if self.invalid_count != expected_invalid:
            raise ValueError("invalid_count must match preserved observations")
        if self.final_history != self.snapshot.initial_history + self.observations:
            raise ValueError(
                "final_history must append replay observations to initial_history"
            )
        if self.status not in {
            OptimizerStatus.COMPLETE,
            OptimizerStatus.NO_VALID_TDECQ,
            OptimizerStatus.NOT_READY,
        }:
            raise ValueError("unsupported algorithm replay status")
        if self.snapshot.stage is ReplayStage.COARSE_FINE:
            objective_history = self.observations
        else:
            objective_history = tuple(
                observation
                for observation in self.final_history
                if observation.fingerprint == self.snapshot.fingerprint
            )
        expected_best = best_observation(objective_history)
        if self.best_observation != expected_best:
            raise ValueError(
                "best_observation must match the stage-specific replay history"
            )
        if not self.trace:
            raise ValueError("algorithm replay trace cannot be empty")
        terminal = self.trace[-1]
        terminal_missing_oracle = (
            terminal.decision.candidate is not None
            and not terminal.observations
        )
        if self.snapshot.stage is ReplayStage.COARSE_FINE:
            expected_coarse_count = min(
                self.snapshot.coarse_budget,
                self.snapshot.parameter_space.point_count,
            )
            coarse_count = 0
            for step in self.trace:
                if step.phase is not ReplayPhase.COARSE:
                    break
                coarse_count += 1
            if coarse_count < expected_coarse_count:
                if not (
                    coarse_count == len(self.trace)
                    and terminal_missing_oracle
                ):
                    raise ValueError("algorithm replay trace is incomplete")
            elif coarse_count == len(self.trace):
                if not terminal_missing_oracle:
                    raise ValueError("algorithm replay trace is incomplete")
            else:
                bootstrap = self.trace[coarse_count]
                if bootstrap.phase is not ReplayPhase.FINE_BOOTSTRAP:
                    raise ValueError("algorithm replay trace phase order is invalid")
                fine_steps = self.trace[coarse_count + 1 :]
                if bootstrap.decision.status is OptimizerStatus.CANDIDATE:
                    if not fine_steps:
                        raise ValueError("algorithm replay trace is incomplete")
                    if any(step.phase is not ReplayPhase.FINE for step in fine_steps):
                        raise ValueError(
                            "algorithm replay trace phase order is invalid"
                        )
                    if len(fine_steps) > self.snapshot.adaptive_budget:
                        raise ValueError("algorithm replay trace exceeds its budget")
                    if (
                        len(fine_steps) < self.snapshot.adaptive_budget
                        and terminal.decision.candidate is not None
                        and terminal.observations
                    ):
                        raise ValueError("algorithm replay trace is incomplete")
                elif fine_steps:
                    raise ValueError(
                        "terminal fine bootstrap cannot have later trace steps"
                    )
        else:
            if any(step.phase is not ReplayPhase.RETUNE for step in self.trace):
                raise ValueError("algorithm replay trace phase order is invalid")
            if len(self.trace) > self.snapshot.adaptive_budget:
                raise ValueError("algorithm replay trace exceeds its budget")
            if (
                len(self.trace) < self.snapshot.adaptive_budget
                and terminal.decision.candidate is not None
                and terminal.observations
            ):
                raise ValueError("algorithm replay trace is incomplete")
        if terminal.decision.candidate is not None and not terminal.observations:
            expected_status = OptimizerStatus.NOT_READY
        elif terminal.decision.candidate is None:
            expected_status = terminal.decision.status
        else:
            expected_status = (
                OptimizerStatus.COMPLETE
                if expected_best is not None
                else OptimizerStatus.NO_VALID_TDECQ
            )
        if self.status is not expected_status:
            raise ValueError(
                "terminal status must be derived from the final trace decision"
            )
        expected_simulation = _simulate_replay(self.snapshot)
        if self.trace != expected_simulation.trace:
            raise ValueError(
                "trace does not match deterministic replay simulation"
            )
        if (
            self.status is not expected_simulation.status
            or self.reason != expected_simulation.reason
        ):
            raise ValueError(
                "report outcome does not match deterministic replay simulation"
            )
        expected_digest = _canonical_digest(
            self.model_dump(mode="json", exclude={"digest"})
        )
        if self.digest != expected_digest:
            raise ValueError("digest must be SHA256 of the canonical replay report")
        return self


class ReplayReport(FrozenModel):
    schema_version: str = Field(default="ai-tuning-replay-v1.1", pattern=r"^ai-tuning-replay-v1\.1$")
    status: OptimizerStatus
    observations: tuple[I2CObservation, ...]
    observation_count: StrictInt = Field(ge=0)
    invalid_count: StrictInt = Field(ge=0)
    candidate_order: tuple[Identifier, ...]
    best_observation: I2CObservation | None
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_counts_and_status(self) -> Self:
        ordered = tuple(
            sorted(self.observations, key=lambda row: row.observation_id)
        )
        if self.observations != ordered:
            raise ValueError("observations must be in canonical observation_id order")
        observation_ids = tuple(row.observation_id for row in self.observations)
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation_id values must be unique for replay")
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count must equal observations length")
        expected_invalid = sum(
            not row.result.has_valid_tdecq for row in self.observations
        )
        if self.invalid_count != expected_invalid:
            raise ValueError("invalid_count must match preserved observations")
        expected_best = best_observation(self.observations)
        expected_status = (
            OptimizerStatus.NOT_READY
            if not self.observations
            else (
                OptimizerStatus.COMPLETE
                if expected_best is not None
                else OptimizerStatus.NO_VALID_TDECQ
            )
        )
        if self.status is not expected_status:
            raise ValueError("status must match the canonical replay observations")
        if self.best_observation != expected_best:
            raise ValueError(
                "best_observation must match the canonical replay observations"
            )
        expected_candidates = tuple(
            sorted({row.candidate_id for row in self.observations})
        )
        if self.candidate_order != expected_candidates:
            raise ValueError("candidate_order must be canonical and deduplicated")
        payload = {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "observations": [
                row.model_dump(mode="json") for row in self.observations
            ],
            "observation_count": self.observation_count,
            "invalid_count": self.invalid_count,
            "candidate_order": self.candidate_order,
            "best_observation_id": (
                None
                if self.best_observation is None
                else self.best_observation.observation_id
            ),
        }
        if self.digest != _canonical_digest(payload):
            raise ValueError("digest must match the canonical replay report")
        return self


def evaluate_replay(
    observations: tuple[I2CObservation, ...],
) -> ReplayReport:
    """Create a canonical offline report; this is not the algorithm runner."""
    if not isinstance(observations, tuple):
        raise TypeError("observations must be a tuple")
    if not all(isinstance(row, I2CObservation) for row in observations):
        raise TypeError("observations must contain I2CObservation values")

    ordered = tuple(sorted(observations, key=lambda row: row.observation_id))
    observation_ids = tuple(row.observation_id for row in ordered)
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("observation_id values must be unique for replay")

    best = best_observation(ordered)
    if not ordered:
        status = OptimizerStatus.NOT_READY
    elif best is None:
        status = OptimizerStatus.NO_VALID_TDECQ
    else:
        status = OptimizerStatus.COMPLETE
    invalid_count = sum(not row.result.has_valid_tdecq for row in ordered)
    candidate_order = tuple(sorted({row.candidate_id for row in ordered}))

    payload = {
        "schema_version": "ai-tuning-replay-v1.1",
        "status": status.value,
        "observations": [
            row.model_dump(mode="json") for row in ordered
        ],
        "observation_count": len(ordered),
        "invalid_count": invalid_count,
        "candidate_order": candidate_order,
        "best_observation_id": (
            None if best is None else best.observation_id
        ),
    }
    document = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(document).hexdigest()
    return ReplayReport(
        status=status,
        observations=ordered,
        observation_count=len(ordered),
        invalid_count=invalid_count,
        candidate_order=candidate_order,
        best_observation=best,
        digest=digest,
    )


def _fine_replay_decision(
    decision: FineDecision,
    *,
    all_history_count: int,
    selected_history_count: int,
) -> ReplayDecision:
    return ReplayDecision(
        status=decision.status,
        candidate=decision.candidate,
        best_observation=decision.best_observation,
        reason=decision.reason,
        requested_point_count=decision.requested_point_count,
        training_point_count=decision.training_point_count,
        all_history_count=all_history_count,
        selected_history_count=selected_history_count,
    )


def _unique_observation_id(
    candidate_index: int,
    outcome_index: int,
    used_ids: set[str],
) -> str:
    base = f"replay-{candidate_index:08d}-sample-{outcome_index:04d}"
    observation_id = base
    collision_index = 0
    while observation_id in used_ids:
        collision_index += 1
        observation_id = f"{base}-collision-{collision_index:04d}"
    used_ids.add(observation_id)
    return observation_id


def _candidate_trace_step(
    snapshot: ReplaySnapshot,
    *,
    phase: ReplayPhase,
    decision: ReplayDecision,
    execution_index: int,
    candidate_index: int,
    used_ids: set[str],
) -> tuple[ReplayTraceStep, bool]:
    candidate = decision.candidate
    if candidate is None:
        raise ValueError("candidate trace step requires a candidate")
    outcomes = snapshot.oracle.outcomes_for(candidate.requested_i2c)
    if outcomes is None:
        return (
            ReplayTraceStep(
                execution_index=execution_index,
                phase=phase,
                decision=decision,
                observations=(),
            ),
            False,
        )
    observations = tuple(
        I2CObservation(
            observation_id=_unique_observation_id(
                candidate_index,
                outcome_index,
                used_ids,
            ),
            candidate_id=candidate.candidate_id,
            requested_i2c=dict(candidate.requested_i2c),
            readback_i2c=dict(outcome.readback_i2c),
            result=outcome.result,
            fingerprint=snapshot.fingerprint,
        )
        for outcome_index, outcome in enumerate(outcomes)
    )
    return (
        ReplayTraceStep(
            execution_index=execution_index,
            phase=phase,
            decision=decision,
            observations=observations,
        ),
        True,
    )


def _missing_oracle_reason(candidate: AlgorithmCandidate) -> str:
    requested = json.dumps(
        dict(candidate.requested_i2c),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"simulated oracle has no outcome for requested I2C {requested}"


def _objective_history(
    snapshot: ReplaySnapshot,
    observations: tuple[I2CObservation, ...],
) -> tuple[I2CObservation, ...]:
    if snapshot.stage is ReplayStage.COARSE_FINE:
        return observations
    return tuple(
        observation
        for observation in snapshot.initial_history + observations
        if observation.fingerprint == snapshot.fingerprint
    )


def _build_run_report(
    snapshot: ReplaySnapshot,
    *,
    status: OptimizerStatus,
    reason: str,
    trace: tuple[ReplayTraceStep, ...],
) -> ReplayRunReport:
    observations = tuple(
        observation
        for step in trace
        for observation in step.observations
    )
    candidate_order = tuple(
        step.decision.candidate.candidate_id
        for step in trace
        if step.decision.candidate is not None
    )
    final_history = snapshot.initial_history + observations
    best = best_observation(_objective_history(snapshot, observations))
    data = {
        "schema_version": "ai-tuning-algorithm-replay-v1.1",
        "status": status,
        "reason": reason,
        "snapshot": snapshot,
        "trace": trace,
        "candidate_order": candidate_order,
        "observations": observations,
        "observation_count": len(observations),
        "invalid_count": sum(
            not row.result.has_valid_tdecq for row in observations
        ),
        "final_history": final_history,
        "best_observation": best,
    }
    provisional = ReplayRunReport.model_construct(
        **data,
        digest="0" * 64,
    )
    digest = _canonical_digest(
        provisional.model_dump(mode="json", exclude={"digest"})
    )
    return ReplayRunReport(**data, digest=digest)


def _budget_simulation(
    snapshot: ReplaySnapshot,
    trace: tuple[ReplayTraceStep, ...],
    *,
    reason: str,
) -> _ReplaySimulation:
    observations = tuple(
        observation
        for step in trace
        for observation in step.observations
    )
    status = (
        OptimizerStatus.COMPLETE
        if best_observation(_objective_history(snapshot, observations)) is not None
        else OptimizerStatus.NO_VALID_TDECQ
    )
    return _ReplaySimulation(
        status=status,
        reason=reason,
        trace=trace,
    )


def _simulate_coarse_fine(snapshot: ReplaySnapshot) -> _ReplaySimulation:
    space = snapshot.parameter_space
    planner = CoarsePlanner(
        space,
        budget=snapshot.coarse_budget,
        seed=snapshot.seed,
    )
    trace: list[ReplayTraceStep] = []
    coarse_observations: list[I2CObservation] = []
    used_ids = {row.observation_id for row in snapshot.initial_history}
    candidate_index = 0

    for candidate in planner.candidates():
        decision = ReplayDecision(
            status=OptimizerStatus.CANDIDATE,
            candidate=candidate,
            reason=candidate.reason,
            requested_point_count=candidate_index,
            training_point_count=sum(
                row.result.has_valid_tdecq for row in coarse_observations
            ),
            all_history_count=len(coarse_observations),
            selected_history_count=len(coarse_observations),
        )
        step, oracle_ready = _candidate_trace_step(
            snapshot,
            phase=ReplayPhase.COARSE,
            decision=decision,
            execution_index=len(trace),
            candidate_index=candidate_index,
            used_ids=used_ids,
        )
        trace.append(step)
        candidate_index += 1
        if not oracle_ready:
            return _ReplaySimulation(
                status=OptimizerStatus.NOT_READY,
                reason=_missing_oracle_reason(candidate),
                trace=tuple(trace),
            )
        coarse_observations.extend(step.observations)

    coarse_summary = planner.summarize(tuple(coarse_observations))
    bootstrap = build_fine_from_coarse(
        space,
        coarse_summary,
        radius_steps=snapshot.radius_steps,
        seed=snapshot.seed,
    )
    trace.append(
        ReplayTraceStep(
            execution_index=len(trace),
            phase=ReplayPhase.FINE_BOOTSTRAP,
            decision=ReplayDecision(
                status=bootstrap.status,
                reason=bootstrap.reason,
                requested_point_count=len(coarse_observations),
                training_point_count=sum(
                    row.result.has_valid_tdecq
                    for row in coarse_observations
                ),
                all_history_count=len(coarse_observations),
                selected_history_count=len(coarse_observations),
            ),
        )
    )
    if not bootstrap.ready:
        return _ReplaySimulation(
            status=bootstrap.status,
            reason=bootstrap.reason,
            trace=tuple(trace),
        )

    optimizer = bootstrap.optimizer
    if optimizer is None:  # Defensive; FineBootstrap validates this invariant.
        return _ReplaySimulation(
            status=OptimizerStatus.NOT_READY,
            reason="fine bootstrap did not provide an optimizer",
            trace=tuple(trace),
        )
    fine_history = list(coarse_observations)
    for _ in range(snapshot.adaptive_budget):
        fine_decision = optimizer.suggest(tuple(fine_history))
        decision = _fine_replay_decision(
            fine_decision,
            all_history_count=len(fine_history),
            selected_history_count=len(fine_history),
        )
        if decision.candidate is None:
            trace.append(
                ReplayTraceStep(
                    execution_index=len(trace),
                    phase=ReplayPhase.FINE,
                    decision=decision,
                )
            )
            return _ReplaySimulation(
                status=decision.status,
                reason=decision.reason,
                trace=tuple(trace),
            )
        step, oracle_ready = _candidate_trace_step(
            snapshot,
            phase=ReplayPhase.FINE,
            decision=decision,
            execution_index=len(trace),
            candidate_index=candidate_index,
            used_ids=used_ids,
        )
        trace.append(step)
        candidate_index += 1
        if not oracle_ready:
            return _ReplaySimulation(
                status=OptimizerStatus.NOT_READY,
                reason=_missing_oracle_reason(decision.candidate),
                trace=tuple(trace),
            )
        fine_history.extend(step.observations)

    return _budget_simulation(
        snapshot,
        tuple(trace),
        reason="adaptive fine replay budget exhausted",
    )


def _simulate_retune(snapshot: ReplaySnapshot) -> _ReplaySimulation:
    current_readback = snapshot.current_readback
    if current_readback is None:  # ReplaySnapshot validates this invariant.
        return _ReplaySimulation(
            status=OptimizerStatus.NOT_READY,
            reason="RETUNE replay requires current_readback",
            trace=(),
        )
    planner = RetunePlanner(
        snapshot.parameter_space,
        fingerprint=snapshot.fingerprint,
        current_readback=current_readback,
        radius_steps=snapshot.radius_steps,
        seed=snapshot.seed,
    )
    trace: list[ReplayTraceStep] = []
    history = list(snapshot.initial_history)
    used_ids = {row.observation_id for row in snapshot.initial_history}
    candidate_index = 0

    for _ in range(snapshot.adaptive_budget):
        result = planner.suggest(tuple(history))
        decision = _fine_replay_decision(
            result.decision,
            all_history_count=len(result.all_history),
            selected_history_count=len(result.selected_history),
        )
        if decision.candidate is None:
            trace.append(
                ReplayTraceStep(
                    execution_index=len(trace),
                    phase=ReplayPhase.RETUNE,
                    decision=decision,
                )
            )
            return _ReplaySimulation(
                status=decision.status,
                reason=decision.reason,
                trace=tuple(trace),
            )
        step, oracle_ready = _candidate_trace_step(
            snapshot,
            phase=ReplayPhase.RETUNE,
            decision=decision,
            execution_index=len(trace),
            candidate_index=candidate_index,
            used_ids=used_ids,
        )
        trace.append(step)
        candidate_index += 1
        if not oracle_ready:
            return _ReplaySimulation(
                status=OptimizerStatus.NOT_READY,
                reason=_missing_oracle_reason(decision.candidate),
                trace=tuple(trace),
            )
        history.extend(step.observations)

    return _budget_simulation(
        snapshot,
        tuple(trace),
        reason="adaptive retune replay budget exhausted",
    )


def _simulate_replay(snapshot: ReplaySnapshot) -> _ReplaySimulation:
    if snapshot.stage is ReplayStage.COARSE_FINE:
        return _simulate_coarse_fine(snapshot)
    if snapshot.stage is ReplayStage.RETUNE:
        return _simulate_retune(snapshot)
    raise ValueError("unsupported replay stage")


def run_replay(snapshot: ReplaySnapshot) -> ReplayRunReport:
    """Run planners against a bounded serialized oracle, never hardware."""
    if not isinstance(snapshot, ReplaySnapshot):
        raise TypeError("snapshot must be a ReplaySnapshot")
    simulation = _simulate_replay(snapshot)
    return _build_run_report(
        snapshot,
        status=simulation.status,
        reason=simulation.reason,
        trace=simulation.trace,
    )
