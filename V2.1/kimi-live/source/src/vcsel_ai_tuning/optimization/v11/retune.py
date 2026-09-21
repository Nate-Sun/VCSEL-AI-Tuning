from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .fine import DiscreteBayesianFine, FineDecision
from .grid import (
    DiscreteParameterSpace,
    TuningParameter,
    seeded_permutation_indices,
)
from .models import (
    AlgorithmCandidate,
    ExperimentFingerprint,
    I2CObservation,
    OptimizerStatus,
)


def select_retune_history(
    history: tuple[I2CObservation, ...],
    fingerprint: ExperimentFingerprint,
) -> tuple[I2CObservation, ...]:
    if not isinstance(history, tuple):
        raise TypeError("history must be a tuple")
    if not isinstance(fingerprint, ExperimentFingerprint):
        raise TypeError("fingerprint must be an ExperimentFingerprint")
    selected: list[I2CObservation] = []
    for observation in history:
        if not isinstance(observation, I2CObservation):
            raise TypeError("history must contain I2CObservation values")
        if observation.fingerprint == fingerprint:
            selected.append(observation)
    return tuple(selected)


def build_local_space(
    space: DiscreteParameterSpace,
    *,
    current_readback: Mapping[str, int],
    radius_steps: int,
) -> DiscreteParameterSpace:
    if not isinstance(space, DiscreteParameterSpace):
        raise TypeError("space must be a DiscreteParameterSpace")
    if (
        not isinstance(radius_steps, int)
        or isinstance(radius_steps, bool)
        or radius_steps <= 0
    ):
        raise ValueError("radius_steps must be a positive integer")
    if (
        not isinstance(current_readback, Mapping)
        or set(current_readback) != set(space.names)
        or len(current_readback) != len(space.names)
    ):
        raise ValueError("current_readback must contain exactly the selected parameters")

    local_parameters: list[TuningParameter] = []
    for parameter in space.parameters:
        current = current_readback[parameter.name]
        if not isinstance(current, int) or isinstance(current, bool):
            raise ValueError("current readback values must be integers")
        if current < parameter.tuning_min or current > parameter.tuning_max:
            raise ValueError("current readback is outside the selected tuning range")

        radius = radius_steps * parameter.tuning_step
        lower = current - radius
        upper = current + radius
        first_position = max(
            0,
            (lower - parameter.tuning_min + parameter.tuning_step - 1)
            // parameter.tuning_step,
        )
        last_position = min(
            parameter.axis_length - 1,
            (upper - parameter.tuning_min) // parameter.tuning_step,
        )
        if first_position > last_position:
            raise ValueError("local radius contains no point from the selected grid")
        local_parameters.append(
            TuningParameter(
                name=parameter.name,
                tuning_min=parameter.value_at(first_position),
                tuning_max=parameter.value_at(last_position),
                tuning_step=parameter.tuning_step,
                enabled=True,
            )
        )
    return DiscreteParameterSpace(tuple(local_parameters))


@dataclass(frozen=True)
class RetuneResult:
    all_history: tuple[I2CObservation, ...]
    selected_history: tuple[I2CObservation, ...]
    decision: FineDecision


class RetunePlanner:
    """Run the fine model only on an exact FIR/NLC/DCA experiment context."""

    def __init__(
        self,
        space: DiscreteParameterSpace,
        *,
        fingerprint: ExperimentFingerprint,
        current_readback: Mapping[str, int],
        radius_steps: int,
        seed: int,
    ) -> None:
        if not isinstance(fingerprint, ExperimentFingerprint):
            raise TypeError("fingerprint must be an ExperimentFingerprint")
        self.fingerprint = fingerprint
        self.seed = seed
        self.local_space = build_local_space(
            space,
            current_readback=current_readback,
            radius_steps=radius_steps,
        )
        self._fine = DiscreteBayesianFine(
            self.local_space,
            seed=seed,
            feature_space=space,
        )

    def suggest(
        self,
        history: tuple[I2CObservation, ...],
        pending_i2c: tuple[Mapping[str, int], ...] = (),
    ) -> RetuneResult:
        selected = select_retune_history(history, self.fingerprint)
        decision = self._fine.suggest(selected, pending_i2c)
        if (
            decision.status is OptimizerStatus.NOT_READY
            and decision.training_point_count
            < self._fine.min_valid_observations
        ):
            decision = self._initial_design_decision(
                selected,
                pending_i2c,
                decision,
            )
        return RetuneResult(
            all_history=history,
            selected_history=selected,
            decision=decision,
        )

    def _initial_design_decision(
        self,
        history: tuple[I2CObservation, ...],
        pending_i2c: tuple[Mapping[str, int], ...],
        not_ready: FineDecision,
    ) -> FineDecision:
        excluded: set[int] = set()
        for observation in history:
            try:
                excluded.add(self.local_space.index(observation.requested_i2c))
            except ValueError:
                pass
        for point in pending_i2c:
            excluded.add(self.local_space.index(point))

        for original_index in seeded_permutation_indices(
            self.local_space.point_count,
            self.seed,
        ):
            if original_index in excluded:
                continue
            return FineDecision(
                status=OptimizerStatus.CANDIDATE,
                candidate=AlgorithmCandidate(
                    candidate_id=f"i2c-retune-init-point-{original_index:08d}",
                    requested_i2c=self.local_space.point_from_index(
                        original_index
                    ),
                    predicted_metrics={},
                    reason=(
                        "deterministic initial design for context-isolated "
                        "Bayesian retune"
                    ),
                    model_version="i2c-retune-initial-design-v1.1",
                ),
                best_observation=not_ready.best_observation,
                reason=(
                    "a deterministic Bayesian retune initial-design point "
                    "is ready"
                ),
                requested_point_count=len(excluded),
                training_point_count=not_ready.training_point_count,
            )
        return not_ready
