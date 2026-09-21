"""Sequential three-candidate planner for the reviewed LIVE pilot."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Self

from pydantic import StringConstraints, model_validator

from .fine import DiscreteBayesianFine
from .grid import DiscreteParameterSpace, seeded_permutation_indices
from .models import (
    AlgorithmCandidate,
    FrozenModel,
    I2CObservation,
    Identifier,
    OptimizerStatus,
)


PILOT3_PLANNER_VERSION = "i2c-pilot3-sequential-v1.2"


class PilotStrategy(StrEnum):
    SEED = "seed"
    DIVERSITY_BOOTSTRAP = "diversity_bootstrap"
    BAYESIAN_ACQUISITION = "bayesian_acquisition"


class PilotDecision(FrozenModel):
    status: OptimizerStatus
    candidate: AlgorithmCandidate | None = None
    strategy: PilotStrategy | None = None
    observation_ids_used: tuple[Identifier, ...] = ()
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    planner_version: str = PILOT3_PLANNER_VERSION

    @model_validator(mode="after")
    def validate_candidate_state(self) -> Self:
        if self.status is OptimizerStatus.CANDIDATE:
            if self.candidate is None or self.strategy is None:
                raise ValueError("CANDIDATE requires a candidate and strategy")
        elif self.candidate is not None:
            raise ValueError("only CANDIDATE may carry a candidate")
        if self.status not in {
            OptimizerStatus.CANDIDATE,
            OptimizerStatus.COMPLETE,
            OptimizerStatus.NO_VALID_TDECQ,
            OptimizerStatus.NOT_READY,
        }:
            raise ValueError("unsupported PILOT_3 decision status")
        return self


class Pilot3Planner:
    """Choose each PILOT_3 point from already persisted observations."""

    def __init__(self, space: DiscreteParameterSpace, *, seed: int) -> None:
        if not isinstance(space, DiscreteParameterSpace):
            raise TypeError("space must be a DiscreteParameterSpace")
        if space.point_count < 3:
            raise ValueError("PILOT_3 requires at least three candidate points")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.space = space
        self.seed = seed
        self._bayesian = DiscreteBayesianFine(space, seed=seed)

    def suggest(
        self,
        observations: tuple[I2CObservation, ...],
        pending_i2c: tuple[Mapping[str, int], ...] = (),
    ) -> PilotDecision:
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        if not isinstance(pending_i2c, tuple):
            raise TypeError("pending_i2c must be a tuple")
        observation_ids = self._validate_history(observations)
        excluded = {self.space.index(row.requested_i2c) for row in observations}
        for point in pending_i2c:
            excluded.add(self.space.index(point))

        if len(observations) >= 3:
            return PilotDecision(
                status=OptimizerStatus.COMPLETE,
                observation_ids_used=observation_ids,
                reason="the fixed three-candidate PILOT_3 budget is complete",
            )
        if len(excluded) >= self.space.point_count:
            return PilotDecision(
                status=OptimizerStatus.COMPLETE,
                observation_ids_used=observation_ids,
                reason="the discrete PILOT_3 candidate pool is exhausted",
            )
        if not observations:
            return self._seed_decision(excluded, observation_ids)
        if len(observations) == 1:
            return self._diversity_decision(
                observations[0], excluded, observation_ids
            )
        return self._bayesian_decision(observations, pending_i2c, observation_ids)

    def _validate_history(
        self, observations: tuple[I2CObservation, ...]
    ) -> tuple[str, ...]:
        ids: list[str] = []
        for observation in observations:
            if not isinstance(observation, I2CObservation):
                raise TypeError("observations must contain I2CObservation values")
            self.space.index(observation.requested_i2c)
            self._normalized_readback(observation.readback_i2c)
            ids.append(observation.observation_id)
        if len(ids) != len(set(ids)):
            raise ValueError("PILOT_3 observation IDs must be unique")
        return tuple(ids)

    def _seeded_rank(self) -> dict[int, int]:
        return {
            original_index: rank
            for rank, original_index in enumerate(
                seeded_permutation_indices(self.space.point_count, self.seed)
            )
        }

    def _candidate(
        self,
        *,
        original_index: int,
        sequence: int,
        strategy: PilotStrategy,
        reason: str,
        predicted_metrics: Mapping[str, float | None] | None = None,
    ) -> AlgorithmCandidate:
        return AlgorithmCandidate(
            candidate_id=(
                f"i2c-pilot-{sequence:08d}-{strategy.value}-"
                f"point-{original_index:08d}"
            ),
            requested_i2c=self.space.point_from_index(original_index),
            predicted_metrics=predicted_metrics or {},
            reason=reason,
            model_version=PILOT3_PLANNER_VERSION,
        )

    def _seed_decision(
        self, excluded: set[int], observation_ids: tuple[str, ...]
    ) -> PilotDecision:
        original_index = next(
            index
            for index in seeded_permutation_indices(self.space.point_count, self.seed)
            if index not in excluded
        )
        return PilotDecision(
            status=OptimizerStatus.CANDIDATE,
            candidate=self._candidate(
                original_index=original_index,
                sequence=0,
                strategy=PilotStrategy.SEED,
                reason="deterministic PILOT_3 seed selected at execution time",
            ),
            strategy=PilotStrategy.SEED,
            observation_ids_used=observation_ids,
            reason="the deterministic PILOT_3 seed is ready",
        )

    def _normalized_readback(self, point: Mapping[str, int]) -> tuple[float, ...]:
        if set(point) != set(self.space.names) or len(point) != len(self.space.names):
            raise ValueError("readback must contain exactly the enabled parameters")
        by_name = {parameter.name: parameter for parameter in self.space.parameters}
        normalized: list[float] = []
        for name in self.space.names:
            value = point[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("readback values must be integers")
            parameter = by_name[name]
            if value < parameter.tuning_min or value > parameter.tuning_max:
                raise ValueError(
                    f"readback value is outside the {name!r} grid bounds"
                )
            span = parameter.tuning_max - parameter.tuning_min
            normalized.append(
                0.0 if span == 0 else (value - parameter.tuning_min) / span
            )
        return tuple(normalized)

    def _diversity_decision(
        self,
        observation: I2CObservation,
        excluded: set[int],
        observation_ids: tuple[str, ...],
    ) -> PilotDecision:
        origin = self._normalized_readback(observation.readback_i2c)
        ranks = self._seeded_rank()
        best_index: int | None = None
        best_key: tuple[float, int] | None = None
        for original_index in range(self.space.point_count):
            if original_index in excluded:
                continue
            candidate_features = self._normalized_readback(
                self.space.point_from_index(original_index)
            )
            squared_distance = sum(
                (left - right) ** 2
                for left, right in zip(candidate_features, origin, strict=True)
            )
            key = (squared_distance, -ranks[original_index])
            if best_key is None or key > best_key:
                best_key = key
                best_index = original_index
        if best_index is None:
            return PilotDecision(
                status=OptimizerStatus.COMPLETE,
                strategy=PilotStrategy.DIVERSITY_BOOTSTRAP,
                observation_ids_used=observation_ids,
                reason="no unmeasured diversity bootstrap point remains",
            )
        return PilotDecision(
            status=OptimizerStatus.CANDIDATE,
            candidate=self._candidate(
                original_index=best_index,
                sequence=1,
                strategy=PilotStrategy.DIVERSITY_BOOTSTRAP,
                reason=(
                    "maximum normalized distance from the first persisted "
                    "PILOT_3 readback"
                ),
            ),
            strategy=PilotStrategy.DIVERSITY_BOOTSTRAP,
            observation_ids_used=observation_ids,
            reason="an observation-conditioned diversity bootstrap is ready",
        )

    def _bayesian_decision(
        self,
        observations: tuple[I2CObservation, ...],
        pending_i2c: tuple[Mapping[str, int], ...],
        observation_ids: tuple[str, ...],
    ) -> PilotDecision:
        fine = self._bayesian.suggest(observations, pending_i2c)
        if fine.status is not OptimizerStatus.CANDIDATE or fine.candidate is None:
            return PilotDecision(
                status=fine.status,
                strategy=PilotStrategy.BAYESIAN_ACQUISITION,
                observation_ids_used=observation_ids,
                reason=fine.reason,
            )
        original_index = self.space.index(fine.candidate.requested_i2c)
        return PilotDecision(
            status=OptimizerStatus.CANDIDATE,
            candidate=self._candidate(
                original_index=original_index,
                sequence=2,
                strategy=PilotStrategy.BAYESIAN_ACQUISITION,
                reason=fine.candidate.reason,
                predicted_metrics=fine.candidate.predicted_metrics,
            ),
            strategy=PilotStrategy.BAYESIAN_ACQUISITION,
            observation_ids_used=observation_ids,
            reason="a Bayesian acquisition conditioned on two observations is ready",
        )

