from __future__ import annotations

from pydantic import Field, StrictInt, model_validator

from .grid import DiscreteParameterSpace, seeded_permutation_indices
from .models import (
    AlgorithmCandidate,
    FrozenModel,
    I2CObservation,
    OptimizerStatus,
)
from .objective import best_observation


class CoarseSummary(FrozenModel):
    status: OptimizerStatus
    observations: tuple[I2CObservation, ...]
    best_observation: I2CObservation | None
    candidate_count: StrictInt = Field(ge=0)
    budget: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_completion(self) -> "CoarseSummary":
        allowed = {
            OptimizerStatus.COMPLETE,
            OptimizerStatus.NO_VALID_TDECQ,
        }
        if self.status not in allowed:
            raise ValueError("coarse status must describe a completed run")
        if (
            self.status is OptimizerStatus.NO_VALID_TDECQ
            and self.best_observation is not None
        ):
            raise ValueError("NO_VALID_TDECQ cannot have a best observation")
        if (
            self.status is OptimizerStatus.COMPLETE
            and self.best_observation is None
        ):
            raise ValueError("COMPLETE requires a best observation")
        return self


class CoarsePlanner:
    """Deterministic, bounded global coverage of a discrete I2C space."""

    def __init__(
        self,
        space: DiscreteParameterSpace,
        *,
        budget: int,
        seed: int,
    ) -> None:
        if not isinstance(space, DiscreteParameterSpace):
            raise TypeError("space must be a DiscreteParameterSpace")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("budget must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self._space = space
        self._budget = budget
        self._seed = seed

    @property
    def candidate_count(self) -> int:
        return min(self._budget, self._space.point_count)

    def candidates(self):
        permutation = seeded_permutation_indices(
            self._space.point_count,
            self._seed,
        )
        for execution_index, original_index in enumerate(permutation):
            if execution_index >= self.candidate_count:
                break
            yield AlgorithmCandidate(
                candidate_id=(
                    f"i2c-coarse-{execution_index:08d}-"
                    f"point-{original_index:08d}"
                ),
                requested_i2c=self._space.point_from_index(original_index),
                predicted_metrics={},
                reason="deterministic seeded sparse global coverage",
                model_version="i2c-coarse-v1.1",
            )

    def summarize(
        self,
        observations: tuple[I2CObservation, ...],
    ) -> CoarseSummary:
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        best = best_observation(observations)
        status = (
            OptimizerStatus.COMPLETE
            if best is not None
            else OptimizerStatus.NO_VALID_TDECQ
        )
        return CoarseSummary(
            status=status,
            observations=observations,
            best_observation=best,
            candidate_count=self.candidate_count,
            budget=self._budget,
        )
