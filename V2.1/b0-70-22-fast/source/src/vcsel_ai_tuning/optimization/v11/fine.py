from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Self

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on runtimes without NumPy
    np = None  # type: ignore[assignment]

from pydantic import Field, StrictInt, StringConstraints, model_validator

from .coarse import CoarseSummary
from .grid import DiscreteParameterSpace
from .models import (
    AggregatedObservation,
    AlgorithmCandidate,
    FrozenModel,
    I2CObservation,
    OptimizerStatus,
)
from .objective import aggregate_repeated_observations, compare_results, Comparison


class FineDecision(FrozenModel):
    status: OptimizerStatus
    candidate: AlgorithmCandidate | None = None
    best_observation: AggregatedObservation | None = None
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    requested_point_count: StrictInt = Field(ge=0)
    training_point_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_candidate_state(self) -> Self:
        if self.status is OptimizerStatus.CANDIDATE:
            if self.candidate is None:
                raise ValueError("CANDIDATE requires a candidate")
        elif self.candidate is not None:
            raise ValueError("only CANDIDATE may carry a candidate")
        if self.status not in {
            OptimizerStatus.CANDIDATE,
            OptimizerStatus.COMPLETE,
            OptimizerStatus.NO_VALID_TDECQ,
            OptimizerStatus.NOT_READY,
        }:
            raise ValueError("unsupported fine decision status")
        return self


class DiscreteBayesianFine:
    """Finite-pool NumPy GP over exact integer candidates and actual readback."""

    def __init__(
        self,
        space: DiscreteParameterSpace,
        *,
        seed: int,
        feature_space: DiscreteParameterSpace | None = None,
        min_valid_observations: int = 2,
        length_scale: float = 0.35,
        xi: float = 0.01,
        model_jitter: float = 1e-8,
        candidate_chunk_size: int = 2048,
    ) -> None:
        if not isinstance(space, DiscreteParameterSpace):
            raise TypeError("space must be a DiscreteParameterSpace")
        if feature_space is not None and not isinstance(
            feature_space,
            DiscreteParameterSpace,
        ):
            raise TypeError("feature_space must be a DiscreteParameterSpace")
        model_space = feature_space or space
        if model_space.names != space.names:
            raise ValueError("feature_space must use the same selected parameters")
        feature_by_name = {
            parameter.name: parameter for parameter in model_space.parameters
        }
        for parameter in space.parameters:
            feature_parameter = feature_by_name[parameter.name]
            try:
                feature_parameter.position_of(parameter.tuning_min)
                feature_parameter.position_of(parameter.tuning_max)
            except ValueError as error:
                raise ValueError(
                    "candidate space must remain on the feature-space grid"
                ) from error
            if (
                parameter.axis_length > 1
                and parameter.tuning_step % feature_parameter.tuning_step != 0
            ):
                raise ValueError(
                    "candidate space must remain on the feature-space grid"
                )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(min_valid_observations, int)
            or isinstance(min_valid_observations, bool)
            or min_valid_observations < 2
        ):
            raise ValueError("min_valid_observations must be an integer of at least 2")
        for name, value, positive in (
            ("length_scale", length_scale, True),
            ("xi", xi, False),
            ("model_jitter", model_jitter, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (float(value) <= 0 if positive else float(value) < 0)
            ):
                qualifier = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be a finite {qualifier} number")
        if (
            not isinstance(candidate_chunk_size, int)
            or isinstance(candidate_chunk_size, bool)
            or candidate_chunk_size <= 0
        ):
            raise ValueError("candidate_chunk_size must be a positive integer")

        self.space = space
        self.feature_space = model_space
        self.seed = seed
        self.min_valid_observations = min_valid_observations
        self.length_scale = float(length_scale)
        self.xi = float(xi)
        self.model_jitter = float(model_jitter)
        self.candidate_chunk_size = candidate_chunk_size

    def aggregate(
        self,
        observations: tuple[I2CObservation, ...],
    ) -> tuple[AggregatedObservation, ...]:
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        return aggregate_repeated_observations(observations)

    def suggest(
        self,
        observations: tuple[I2CObservation, ...],
        pending_i2c: tuple[Mapping[str, int], ...] = (),
    ) -> FineDecision:
        if not isinstance(observations, tuple):
            raise TypeError("observations must be a tuple")
        if not isinstance(pending_i2c, tuple):
            raise TypeError("pending_i2c must be a tuple")

        aggregated = self.aggregate(observations)
        best = _best_aggregated(aggregated)
        requested_indices = self._requested_indices(observations, pending_i2c)
        common = {
            "best_observation": best,
            "requested_point_count": len(requested_indices),
            "training_point_count": sum(
                row.result.has_valid_tdecq for row in aggregated
            ),
        }

        fingerprints = {
            row.fingerprint.model_dump_json() for row in observations
        }
        if len(fingerprints) > 1:
            return FineDecision(
                status=OptimizerStatus.NOT_READY,
                reason="fine history contains more than one experiment fingerprint",
                **common,
            )

        available_count = self.space.point_count - len(requested_indices)
        if available_count == 0:
            return FineDecision(
                status=(
                    OptimizerStatus.COMPLETE
                    if best is not None
                    else OptimizerStatus.NO_VALID_TDECQ
                ),
                reason="the discrete fine candidate pool is exhausted",
                **common,
            )

        valid_rows = tuple(
            row for row in aggregated if row.result.has_valid_tdecq
        )
        if len(valid_rows) < self.min_valid_observations:
            return FineDecision(
                status=OptimizerStatus.NOT_READY,
                reason=(
                    "not enough distinct valid TDECQ readbacks to fit the "
                    "discrete Bayesian model"
                ),
                **common,
            )
        if np is None:
            return FineDecision(
                status=OptimizerStatus.NOT_READY,
                reason="NumPy is unavailable; the Bayesian model cannot be fitted",
                **common,
            )

        try:
            model = self._fit(valid_rows)
            choice = self._best_candidate(
                model,
                observations,
                requested_indices,
            )
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return FineDecision(
                status=OptimizerStatus.NOT_READY,
                reason="the discrete Bayesian model could not be fitted safely",
                **common,
            )

        if choice is None:
            return FineDecision(
                status=OptimizerStatus.NOT_READY,
                reason="the discrete Bayesian model produced no finite candidate",
                **common,
            )
        original_index, point, metrics = choice
        return FineDecision(
            status=OptimizerStatus.CANDIDATE,
            candidate=AlgorithmCandidate(
                candidate_id=f"i2c-fine-point-{original_index:08d}",
                requested_i2c=point,
                predicted_metrics=metrics,
                reason=(
                    "maximum discrete TDECQ expected improvement times "
                    "validity probability"
                ),
                model_version="i2c-discrete-gp-v1.1",
            ),
            reason="a discrete Bayesian fine candidate is ready",
            **common,
        )

    def _requested_indices(
        self,
        observations: tuple[I2CObservation, ...],
        pending_i2c: tuple[Mapping[str, int], ...],
    ) -> set[int]:
        requested_indices: set[int] = set()
        for observation in observations:
            if not isinstance(observation, I2CObservation):
                raise TypeError("observations must contain I2CObservation values")
            self.feature_space.index(observation.requested_i2c)
            try:
                requested_indices.add(self.space.index(observation.requested_i2c))
            except ValueError:
                pass
            self._feature_row(observation.readback_i2c)
        for point in pending_i2c:
            requested_indices.add(self.space.index(point))
        return requested_indices

    def _feature_row(self, point: Mapping[str, int]):
        if (
            set(point) != set(self.feature_space.names)
            or len(point) != len(self.feature_space.names)
        ):
            raise ValueError("readback must contain exactly the enabled parameters")
        row: list[float] = []
        by_name = {item.name: item for item in self.feature_space.parameters}
        for name in self.feature_space.names:
            value = point[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("readback values must be integers")
            parameter = by_name[name]
            if value < parameter.tuning_min or value > parameter.tuning_max:
                raise ValueError("readback value is outside the tuning bounds")
            span = parameter.tuning_max - parameter.tuning_min
            row.append(
                0.0
                if span == 0
                else (value - parameter.tuning_min) / span
            )
        return tuple(row)

    def _fit(self, valid_rows: tuple[AggregatedObservation, ...]):
        features = np.asarray(
            [self._feature_row(row.readback_i2c) for row in valid_rows],
            dtype=float,
        )
        targets = np.asarray(
            [float(row.result.tdecq_db) for row in valid_rows],
            dtype=float,
        )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("model training data must be finite")

        kernel = self._rbf(features, features)
        kernel.flat[:: len(valid_rows) + 1] += self.model_jitter
        cholesky = np.linalg.cholesky(kernel)
        target_mean = float(np.mean(targets))
        target_scale = max(float(np.std(targets)), 0.1)
        centered = (targets - target_mean) / target_scale
        alpha = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, centered),
        )
        return features, cholesky, alpha, target_mean, target_scale, float(np.min(targets))

    def _best_candidate(self, model, observations, excluded_indices):
        best_choice = None
        best_key: tuple[float, int] | None = None
        chunk_indices: list[int] = []
        chunk_points: list[dict[str, int]] = []

        def score_chunk():
            nonlocal best_choice, best_key
            if not chunk_indices:
                return
            candidate_features = np.asarray(
                [self._feature_row(point) for point in chunk_points],
                dtype=float,
            )
            predictions = self._predict(model, candidate_features)
            validity = self._valid_probability(observations, candidate_features)
            for offset, original_index in enumerate(chunk_indices):
                predicted_tdecq, sigma, improvement = predictions[offset]
                valid_probability = float(validity[offset])
                acquisition = float(improvement * valid_probability)
                values = (
                    predicted_tdecq,
                    sigma,
                    improvement,
                    valid_probability,
                    acquisition,
                )
                if not all(math.isfinite(float(value)) for value in values):
                    continue
                key = (acquisition, -original_index)
                if best_key is None or key > best_key:
                    best_key = key
                    best_choice = (
                        original_index,
                        chunk_points[offset],
                        {
                            "predicted_tdecq_db": float(predicted_tdecq),
                            "predicted_sigma_db": float(sigma),
                            "expected_improvement": float(improvement),
                            "valid_probability": valid_probability,
                            "acquisition": acquisition,
                        },
                    )
            chunk_indices.clear()
            chunk_points.clear()

        for original_index in range(self.space.point_count):
            if original_index in excluded_indices:
                continue
            chunk_indices.append(original_index)
            chunk_points.append(self.space.point_from_index(original_index))
            if len(chunk_indices) >= self.candidate_chunk_size:
                score_chunk()
        score_chunk()
        return best_choice

    def _rbf(self, left, right):
        delta = left[:, None, :] - right[None, :, :]
        squared_distance = np.sum(delta * delta, axis=2)
        return np.exp(-0.5 * squared_distance / (self.length_scale**2))

    def _predict(self, model, candidate_features):
        features, cholesky, alpha, mean, scale, best = model
        cross_kernel = self._rbf(features, candidate_features)
        predicted = mean + (scale * (cross_kernel.T @ alpha))
        projected = np.linalg.solve(cholesky, cross_kernel)
        variance = np.maximum(1.0 - np.sum(projected * projected, axis=0), 0.0)
        sigma = scale * np.sqrt(variance)
        improvement = best - predicted - self.xi
        expected = np.empty_like(improvement)
        for index, (delta, stddev) in enumerate(zip(improvement, sigma, strict=True)):
            if stddev <= 1e-12:
                expected[index] = max(float(delta), 0.0)
                continue
            z_value = float(delta / stddev)
            cdf = 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))
            density = math.exp(-0.5 * z_value * z_value) / math.sqrt(2.0 * math.pi)
            expected[index] = max(float(delta * cdf + stddev * density), 0.0)
        return tuple(
            (float(predicted[index]), float(sigma[index]), float(expected[index]))
            for index in range(len(predicted))
        )

    def _valid_probability(self, observations, candidate_features):
        if not observations:
            return np.full(len(candidate_features), 0.5, dtype=float)
        canonical_observations = tuple(
            sorted(
                observations,
                key=lambda row: (
                    row.observation_id,
                    row.candidate_id,
                    row.fingerprint.model_dump_json(),
                    tuple(sorted(row.requested_i2c.items())),
                    tuple(sorted(row.readback_i2c.items())),
                    row.result.model_dump_json(),
                ),
            )
        )
        training_features = np.asarray(
            [
                self._feature_row(row.readback_i2c)
                for row in canonical_observations
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                float(row.result.has_valid_tdecq)
                for row in canonical_observations
            ],
            dtype=float,
        )
        weights = self._rbf(training_features, candidate_features)
        weighted_valid = labels @ weights
        total_weight = np.sum(weights, axis=0)
        return (weighted_valid + 0.5) / (total_weight + 1.0)


def _best_aggregated(
    observations: tuple[AggregatedObservation, ...],
) -> AggregatedObservation | None:
    best: AggregatedObservation | None = None
    ordered = sorted(
        observations,
        key=lambda row: (
            row.fingerprint.model_dump_json(),
            tuple(sorted(row.readback_i2c.items())),
        ),
    )
    for observation in ordered:
        if not observation.result.has_valid_tdecq:
            continue
        if best is None or compare_results(
            observation.result,
            best.result,
        ) is Comparison.BETTER:
            best = observation
    return best


@dataclass(frozen=True)
class FineBootstrap:
    """Explicit outcome of bridging a completed coarse run into fine search."""

    status: OptimizerStatus
    optimizer: DiscreteBayesianFine | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, OptimizerStatus):
            raise TypeError("status must be an OptimizerStatus")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.status is OptimizerStatus.CANDIDATE:
            if not isinstance(self.optimizer, DiscreteBayesianFine):
                raise ValueError("CANDIDATE requires a fine optimizer")
        elif self.status in {
            OptimizerStatus.NO_VALID_TDECQ,
            OptimizerStatus.NOT_READY,
        }:
            if self.optimizer is not None:
                raise ValueError("non-ready bootstrap cannot carry an optimizer")
        else:
            raise ValueError("unsupported fine bootstrap status")

    @property
    def ready(self) -> bool:
        return self.status is OptimizerStatus.CANDIDATE


def build_fine_from_coarse(
    global_space: DiscreteParameterSpace,
    coarse_summary: CoarseSummary,
    *,
    radius_steps: int,
    seed: int,
    min_valid_observations: int = 2,
    length_scale: float = 0.35,
    xi: float = 0.01,
    model_jitter: float = 1e-8,
    candidate_chunk_size: int = 2048,
) -> FineBootstrap:
    """Create an exact-grid local fine optimizer from coarse readback."""

    if not isinstance(global_space, DiscreteParameterSpace):
        raise TypeError("global_space must be a DiscreteParameterSpace")
    if not isinstance(coarse_summary, CoarseSummary):
        raise TypeError("coarse_summary must be a CoarseSummary")

    if coarse_summary.status is OptimizerStatus.NO_VALID_TDECQ:
        return FineBootstrap(
            status=OptimizerStatus.NO_VALID_TDECQ,
            optimizer=None,
            reason="coarse search produced no valid TDECQ observation",
        )
    if coarse_summary.status is not OptimizerStatus.COMPLETE:
        return FineBootstrap(
            status=OptimizerStatus.NOT_READY,
            optimizer=None,
            reason="coarse search is not complete",
        )

    best = coarse_summary.best_observation
    if best is None:
        return FineBootstrap(
            status=OptimizerStatus.NOT_READY,
            optimizer=None,
            reason="completed coarse search has no best observation",
        )
    if not best.result.has_valid_tdecq:
        return FineBootstrap(
            status=OptimizerStatus.NO_VALID_TDECQ,
            optimizer=None,
            reason="coarse best observation has no valid TDECQ",
        )

    # Local import avoids a module cycle: retune depends on the fine optimizer.
    from .retune import build_local_space

    local_space = build_local_space(
        global_space,
        current_readback=best.readback_i2c,
        radius_steps=radius_steps,
    )
    optimizer = DiscreteBayesianFine(
        local_space,
        seed=seed,
        feature_space=global_space,
        min_valid_observations=min_valid_observations,
        length_scale=length_scale,
        xi=xi,
        model_jitter=model_jitter,
        candidate_chunk_size=candidate_chunk_size,
    )
    return FineBootstrap(
        status=OptimizerStatus.CANDIDATE,
        optimizer=optimizer,
        reason="a local discrete Bayesian fine optimizer is ready",
    )
