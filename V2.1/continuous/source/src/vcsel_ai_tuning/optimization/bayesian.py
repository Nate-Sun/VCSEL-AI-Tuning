from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
from math import isclose, isfinite
from numbers import Real
from types import MappingProxyType
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PlainSerializer,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)
from scipy.stats import norm
import sklearn
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)

from .parameter_space import (
    IntegerParameterSpace,
    Point,
    PointKey,
    validated_observation_map,
)
from .protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    OptimizationContext,
    PathComplete,
    TuningObservation,
    TuningPath,
)


_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_MAX_SKLEARN_SEED = 2**32 - 1
_SNAPSHOT_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    allow_inf_nan=False,
    validate_default=True,
)
_NonBlankText = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_ImmutablePointKey = tuple[StrictInt, ...]


def _freeze_point(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(values))


def _serialize_point(values: Mapping[str, int]) -> dict[str, int]:
    return dict(values)


_ImmutablePoint = Annotated[
    Mapping[str, StrictInt],
    AfterValidator(_freeze_point),
    PlainSerializer(_serialize_point, return_type=dict[str, int]),
]


def _observation_payload(
    observation: TuningObservation,
) -> dict[str, object]:
    return {
        name: getattr(observation, name)
        for name in TuningObservation.model_fields
    }


def _validate_strict_observation_payload(
    payload: Mapping[str, object],
) -> None:
    expected_fields = set(TuningObservation.model_fields)
    if set(payload) != expected_fields:
        raise ValueError(
            "snapshot observation must contain exactly every observation field"
        )

    for field_name in ("observation_id", "candidate_id", "tdecq_status"):
        if not isinstance(payload[field_name], str):
            raise ValueError(
                f"snapshot observation {field_name} must be an exact string"
            )
    for field_name in ("tdecq_invalid_reason", "er_status"):
        value = payload[field_name]
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"snapshot observation {field_name} must be a string or null"
            )

    context = payload["context"]
    if isinstance(context, OptimizationContext):
        pass
    elif isinstance(context, Mapping):
        expected_context_fields = set(OptimizationContext.model_fields)
        if set(context) != expected_context_fields or any(
            not isinstance(context[name], str)
            for name in expected_context_fields
        ):
            raise ValueError(
                "snapshot observation context must use exact string fields"
            )
    else:
        raise ValueError(
            "snapshot observation context must be an OptimizationContext"
        )

    for field_name in ("settings_requested", "settings_readback"):
        settings = payload[field_name]
        if not isinstance(settings, Mapping):
            raise ValueError(
                f"snapshot observation {field_name} must be a mapping"
            )
        if any(not isinstance(name, str) for name in settings):
            raise ValueError(
                f"snapshot observation {field_name} names must be strings"
            )
        if any(type(value) is not int for value in settings.values()):
            raise ValueError(
                f"snapshot observation {field_name} values must be exact integers"
            )

    for field_name in ("tdecq_db", "er_db"):
        value = payload[field_name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
        ):
            raise ValueError(
                f"snapshot observation {field_name} must be an exact finite number"
            )
    if type(payload["er_valid"]) is not bool:
        raise ValueError(
            "snapshot observation er_valid must be an exact boolean"
        )

    artifact_refs = payload["artifact_refs"]
    if not isinstance(artifact_refs, (list, tuple)) or any(
        not isinstance(value, str) for value in artifact_refs
    ):
        raise ValueError(
            "snapshot observation artifact_refs must be an ordered string array"
        )


def _canonical_settings_mapping(
    settings: Mapping[str, int],
    parameter_names: tuple[str, ...],
) -> dict[str, int]:
    known_names = tuple(name for name in parameter_names if name in settings)
    extra_names = tuple(sorted(set(settings) - set(parameter_names)))
    return {
        name: settings[name]
        for name in (*known_names, *extra_names)
    }


class CandidatePrediction(BaseModel):
    model_config = _MODEL_CONFIG

    key: PointKey
    predicted_tdecq_db: float
    predicted_sigma_db: float = Field(ge=0)
    expected_improvement: float = Field(ge=0)
    valid_probability: float = Field(ge=0, le=1)
    acquisition: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_acquisition_identity(self) -> "CandidatePrediction":
        expected = self.expected_improvement * self.valid_probability
        if not isclose(
            self.acquisition,
            expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "acquisition must equal expected improvement multiplied by "
                "valid probability"
            )
        return self


class BayesianSnapshot(BaseModel):
    """Portable optimizer inputs; fitted sklearn objects are deliberately absent."""

    model_config = _SNAPSHOT_CONFIG

    schema_version: Literal["tdecq-bayesian-v1"] = "tdecq-bayesian-v1"
    context: OptimizationContext
    space_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_names: tuple[_NonBlankText, ...]
    seed: StrictInt = Field(ge=0, le=_MAX_SKLEARN_SEED)
    min_valid_observations: StrictInt = Field(ge=1)
    xi: FiniteFloat = Field(ge=0, strict=True)
    observations: tuple[TuningObservation, ...]
    pending_keys: tuple[_ImmutablePointKey, ...]
    anchor_points: tuple[_ImmutablePoint, ...]
    sklearn_version: _NonBlankText = Field(default_factory=lambda: sklearn.__version__)

    @model_validator(mode="before")
    @classmethod
    def validate_raw_observations(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        observations = value.get("observations", ())
        if isinstance(observations, (str, bytes, bytearray)):
            return value
        try:
            prepared = tuple(observations)
        except TypeError:
            return value
        raw_parameter_names = value.get("parameter_names", ())
        parameter_names = (
            tuple(raw_parameter_names)
            if isinstance(raw_parameter_names, (list, tuple))
            and all(isinstance(name, str) for name in raw_parameter_names)
            else ()
        )
        rebuilt_observations: list[object] = []
        for observation in prepared:
            if isinstance(observation, TuningObservation):
                payload = _observation_payload(observation)
            elif isinstance(observation, Mapping):
                payload = dict(observation)
            else:
                rebuilt_observations.append(observation)
                continue
            _validate_strict_observation_payload(payload)
            rebuilt_payload = dict(payload)
            for field_name in ("settings_requested", "settings_readback"):
                rebuilt_payload[field_name] = _canonical_settings_mapping(
                    payload[field_name],
                    parameter_names,
                )
            rebuilt_observations.append(rebuilt_payload)

        rebuilt_value = dict(value)
        rebuilt_value["observations"] = tuple(rebuilt_observations)
        return rebuilt_value

    @model_validator(mode="after")
    def validate_snapshot_integrity(self) -> "BayesianSnapshot":
        names = self.parameter_names
        if not names:
            raise ValueError("parameter_names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("parameter_names must be unique")

        observed_keys: list[PointKey] = []
        expected_names = set(names)
        for observation in self.observations:
            if observation.context != self.context:
                raise ValueError(
                    "every snapshot observation context must equal snapshot context"
                )
            readback = observation.settings_readback
            if set(readback) != expected_names or len(readback) != len(names):
                raise ValueError(
                    "snapshot observation readback must contain exactly parameter_names"
                )
            if any(type(readback[name]) is not int for name in names):
                raise ValueError(
                    "snapshot observation readback values must be exact integers"
                )
            observed_keys.append(tuple(readback[name] for name in names))
        if len(set(observed_keys)) != len(observed_keys):
            raise ValueError("duplicate snapshot observation readback mapping")

        dimension = len(names)
        if any(len(key) != dimension for key in self.pending_keys):
            raise ValueError("every pending key must match parameter_names dimension")
        if len(set(self.pending_keys)) != len(self.pending_keys):
            raise ValueError("pending keys must be unique")
        if self.pending_keys != tuple(sorted(self.pending_keys)):
            raise ValueError("pending keys must use deterministic sorted order")
        if set(self.pending_keys) & set(observed_keys):
            raise ValueError("pending keys must not overlap measured observations")

        anchor_keys: list[PointKey] = []
        for point in self.anchor_points:
            if set(point) != expected_names or len(point) != len(names):
                raise ValueError(
                    "every anchor point must contain exactly parameter_names"
                )
            anchor_keys.append(tuple(point[name] for name in names))
        if len(set(anchor_keys)) != len(anchor_keys):
            raise ValueError("anchor points must be unique")
        return self


class TdecqBayesianOptimizer:
    """Deterministic finite-pool Bayesian optimization of numeric TDECQ."""

    def __init__(
        self,
        space: IntegerParameterSpace,
        context: OptimizationContext,
        seed: int,
        min_valid_observations: int | None = None,
        xi: float = 0.01,
        anchor_points: Iterable[Mapping[str, int]] = (),
    ) -> None:
        if type(seed) is not int:
            raise ValueError("Bayesian seed must be an integer, not bool")
        if seed < 0 or seed > _MAX_SKLEARN_SEED:
            raise ValueError(
                "Bayesian seed must be in the inclusive range "
                f"0..{_MAX_SKLEARN_SEED}"
            )
        if min_valid_observations is not None and (
            type(min_valid_observations) is not int
            or min_valid_observations < 1
        ):
            raise ValueError(
                "min_valid_observations must be an integer of at least 1"
            )
        if isinstance(xi, bool) or not isinstance(xi, Real):
            raise ValueError("xi must be a finite nonnegative numeric value")
        numeric_xi = float(xi)
        if not isfinite(numeric_xi) or numeric_xi < 0:
            raise ValueError("xi must be a finite nonnegative numeric value")

        self.space = space
        self.context = context
        self.seed = seed
        self.xi = numeric_xi
        self.min_valid_observations = (
            min_valid_observations
            if min_valid_observations is not None
            else max(3, len(space.names) + 1)
        )

        try:
            supplied_anchors = tuple(anchor_points)
        except TypeError as error:
            raise ValueError("anchor_points must be an iterable of point mappings") from error
        canonical_anchors: list[Mapping[str, int]] = []
        anchor_keys: list[PointKey] = []
        seen_anchor_keys: set[PointKey] = set()
        for index, point in enumerate(supplied_anchors):
            try:
                key = space.key(point)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"anchor point {index} is not a valid safe endpoint: {error}"
                ) from error
            if key in seen_anchor_keys:
                continue
            seen_anchor_keys.add(key)
            anchor_keys.append(key)
            canonical_anchors.append(
                MappingProxyType(dict(space.point(key)))
            )
        self.anchor_points = tuple(canonical_anchors)
        self._anchor_keys = tuple(anchor_keys)

        self._points = tuple(space.enumerate_points())
        self._keys = tuple(space.key(point) for point in self._points)
        self._index_by_key = {
            key: index for index, key in enumerate(self._keys)
        }
        self._feature_minima = {
            name: min(point[name] for point in self._points)
            for name in space.names
        }
        self._feature_spans = {
            name: (
                max(point[name] for point in self._points)
                - self._feature_minima[name]
            )
            for name in space.names
        }
        self._pool_features = self._features(self._points)

        self.tdecq_training_ids: tuple[str, ...] = ()
        self._regressor: GaussianProcessRegressor | None = None
        self._classifier: GaussianProcessClassifier | None = None
        self._validity_mode = "unfitted"
        self._fitted_history_signature: str | None = None

    def _features(self, points: list[Point] | tuple[Point, ...]) -> np.ndarray:
        rows = [
            [
                (
                    0.0
                    if self._feature_spans[name] == 0
                    else (
                        point[name] - self._feature_minima[name]
                    )
                    / self._feature_spans[name]
                )
                for name in self.space.names
            ]
            for point in points
        ]
        if not rows:
            return np.empty((0, len(self.space.names)), dtype=float)
        return np.asarray(rows, dtype=float)

    def fit(self, observations: tuple[TuningObservation, ...]) -> None:
        self._reset_fit_state()
        by_key = validated_observation_map(
            self.space,
            self.context,
            observations,
        )
        canonical_rows = tuple(sorted(by_key.items()))
        valid_rows = tuple(
            (key, observation)
            for key, observation in canonical_rows
            if self._has_numeric_tdecq(observation)
        )
        self.tdecq_training_ids = tuple(
            observation.observation_id for _, observation in valid_rows
        )

        regressor: GaussianProcessRegressor | None = None
        if len(valid_rows) >= self.min_valid_observations:
            x_valid = self._features(
                [
                    dict(observation.settings_readback)
                    for _, observation in valid_rows
                ]
            )
            y_valid = np.asarray(
                [observation.tdecq_db for _, observation in valid_rows],
                dtype=float,
            )
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(
                    noise_level=1e-5,
                    noise_level_bounds=(1e-8, 1e1),
                )
            )
            candidate_regressor = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=self.seed,
                n_restarts_optimizer=0,
            )
            candidate_regressor.fit(x_valid, y_valid)
            regressor = candidate_regressor

        labels = np.asarray(
            [
                int(self._has_numeric_tdecq(observation))
                for _, observation in canonical_rows
            ],
            dtype=int,
        )
        classes = frozenset(labels.tolist())
        classifier: GaussianProcessClassifier | None = None
        if classes == {1}:
            validity_mode = "all_valid"
        elif classes == {0} or not classes:
            validity_mode = "all_invalid"
        else:
            validity_mode = "classifier"
            candidate_classifier = GaussianProcessClassifier(
                kernel=1.0 * RBF(length_scale=1.0),
                random_state=self.seed,
                n_restarts_optimizer=0,
            )
            candidate_classifier.fit(
                self._features(
                    [
                        dict(observation.settings_readback)
                        for _, observation in canonical_rows
                    ]
                ),
                labels,
            )
            classifier = candidate_classifier

        self._regressor = regressor
        self._classifier = classifier
        self._validity_mode = validity_mode
        self._fitted_history_signature = self._history_signature(by_key)

    def snapshot(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: Iterable[PointKey],
    ) -> BayesianSnapshot:
        """Capture validated inputs needed for an exact same-runtime refit."""

        by_key = validated_observation_map(
            self.space,
            self.context,
            observations,
        )
        pending = self._validated_pending_keys(pending_keys)
        overlap = frozenset(by_key) & pending
        if overlap:
            raise ValueError(
                "pending keys must not overlap measured observations: "
                f"{tuple(sorted(overlap))!r}"
            )

        canonical_anchors: list[dict[str, int]] = []
        seen_anchor_keys: set[PointKey] = set()
        for index, point in enumerate(self.anchor_points):
            try:
                key = self.space.key(point)
            except ValueError as error:
                raise ValueError(
                    f"anchor point {index} is not a valid safe endpoint: {error}"
                ) from error
            if key in seen_anchor_keys:
                raise ValueError("anchor points must remain unique")
            seen_anchor_keys.add(key)
            canonical_anchors.append(self.space.point(key))

        return BayesianSnapshot(
            context=self.context,
            space_fingerprint=self.space.fingerprint,
            parameter_names=self.space.names,
            seed=self.seed,
            min_valid_observations=self.min_valid_observations,
            xi=self.xi,
            observations=tuple(
                self._canonical_snapshot_observation(observation)
                for _, observation in sorted(by_key.items())
            ),
            pending_keys=tuple(sorted(pending)),
            anchor_points=tuple(canonical_anchors),
            sklearn_version=sklearn.__version__,
        )

    @classmethod
    def from_snapshot(
        cls,
        space: IntegerParameterSpace,
        document: BayesianSnapshot | str | bytes | bytearray,
    ) -> "TdecqBayesianOptimizer":
        """Validate a portable snapshot and deterministically refit its models."""

        if not isinstance(space, IntegerParameterSpace):
            raise ValueError("Bayesian snapshot requires an IntegerParameterSpace")
        if isinstance(document, BayesianSnapshot):
            document = document.model_dump_json()
        try:
            loaded = BayesianSnapshot.model_validate_json(document)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid Bayesian snapshot: {error}") from error

        if loaded.space_fingerprint != space.fingerprint:
            raise ValueError("Bayesian snapshot space fingerprint does not match")
        if loaded.parameter_names != space.names:
            raise ValueError("Bayesian snapshot parameter names do not match the space")
        if loaded.sklearn_version != sklearn.__version__:
            raise ValueError(
                "Bayesian snapshot sklearn version does not match this runtime"
            )

        optimizer = cls(
            space,
            context=loaded.context,
            seed=loaded.seed,
            min_valid_observations=loaded.min_valid_observations,
            xi=loaded.xi,
            anchor_points=loaded.anchor_points,
        )
        by_key = validated_observation_map(
            space,
            loaded.context,
            loaded.observations,
        )
        pending = optimizer._validated_pending_keys(loaded.pending_keys)
        overlap = frozenset(by_key) & pending
        if overlap:
            raise ValueError(
                "Bayesian snapshot pending keys overlap measured observations"
            )
        if tuple(sorted(pending)) != loaded.pending_keys:
            raise ValueError("Bayesian snapshot pending keys are not canonical")
        if optimizer.anchor_points != loaded.anchor_points:
            raise ValueError("Bayesian snapshot anchor points are not canonical")

        optimizer.fit(loaded.observations)
        return optimizer

    def _canonical_snapshot_observation(
        self,
        observation: TuningObservation,
    ) -> TuningObservation:
        payload = _observation_payload(observation)
        _validate_strict_observation_payload(payload)
        payload["settings_requested"] = self._canonical_settings(
            observation.settings_requested
        )
        payload["settings_readback"] = self._canonical_settings(
            observation.settings_readback
        )
        return TuningObservation.model_validate(payload)

    def _canonical_settings(
        self,
        settings: Mapping[str, int],
    ) -> dict[str, int]:
        return _canonical_settings_mapping(settings, self.space.names)

    def score_unmeasured(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey],
    ) -> tuple[CandidatePrediction, ...]:
        by_key = validated_observation_map(
            self.space,
            self.context,
            observations,
        )
        pending = self._validated_pending_keys(pending_keys)
        history_signature = self._history_signature(by_key)
        if self._fitted_history_signature is None:
            raise RuntimeError("Bayesian model has not been fitted")
        if history_signature != self._fitted_history_signature:
            raise ValueError(
                "observations do not match the fitted history; refusing stale "
                "Bayesian scoring"
            )
        if self._regressor is None:
            raise RuntimeError("TDECQ regressor is not fitted")

        measured_keys = frozenset(by_key)
        available_indices = tuple(
            index
            for index, key in enumerate(self._keys)
            if key not in measured_keys and key not in pending
        )
        if not available_indices:
            return ()

        features = self._pool_features[np.asarray(available_indices)]
        raw_mean, raw_sigma = self._regressor.predict(
            features,
            return_std=True,
        )
        mean = self._model_vector(
            raw_mean,
            expected_length=len(available_indices),
            label="predicted TDECQ",
        )
        sigma = self._model_vector(
            raw_sigma,
            expected_length=len(available_indices),
            label="predicted sigma",
        )
        if np.any(sigma < 0):
            raise ValueError("predicted sigma contains a negative value")

        valid_probability = self._valid_probability(features)
        if valid_probability.shape != mean.shape:
            raise ValueError("validity probability has an unexpected shape")
        if not np.all(np.isfinite(valid_probability)):
            raise ValueError("validity probability contains nonfinite output")
        if np.any((valid_probability < 0) | (valid_probability > 1)):
            raise ValueError("validity probability is outside [0, 1]")

        current_best = min(
            float(observation.tdecq_db)
            for _, observation in sorted(by_key.items())
            if self._has_numeric_tdecq(observation)
        )
        improvement = current_best - mean - self.xi
        expected_improvement = np.zeros_like(mean)
        positive_sigma = sigma > np.finfo(float).eps
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = improvement[positive_sigma] / sigma[positive_sigma]
            expected_improvement[positive_sigma] = (
                improvement[positive_sigma] * norm.cdf(z)
                + sigma[positive_sigma] * norm.pdf(z)
            )
        expected_improvement[~positive_sigma] = np.maximum(
            improvement[~positive_sigma],
            0.0,
        )
        expected_improvement = np.maximum(expected_improvement, 0.0)
        acquisition = expected_improvement * valid_probability
        if not np.all(np.isfinite(expected_improvement)):
            raise ValueError("expected improvement contains nonfinite output")
        if not np.all(np.isfinite(acquisition)):
            raise ValueError("Bayesian acquisition contains nonfinite output")

        return tuple(
            CandidatePrediction(
                key=self._keys[pool_index],
                predicted_tdecq_db=float(mean[result_index]),
                predicted_sigma_db=float(sigma[result_index]),
                expected_improvement=float(expected_improvement[result_index]),
                valid_probability=float(valid_probability[result_index]),
                acquisition=float(acquisition[result_index]),
            )
            for result_index, pool_index in enumerate(available_indices)
        )

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey] = frozenset(),
    ) -> CandidateSuggestion | FallbackRequested | NoSafeCandidate | PathComplete:
        by_key = validated_observation_map(
            self.space,
            self.context,
            observations,
        )
        pending = self._validated_pending_keys(pending_keys)
        measured_keys = frozenset(by_key)
        remaining_keys = frozenset(self._keys) - measured_keys

        if not remaining_keys:
            return self._complete(by_key)

        available_keys = remaining_keys - pending
        if not available_keys:
            return NoSafeCandidate(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    f"All {len(remaining_keys)} unmeasured safe endpoints are "
                    "pending; no endpoint can be suggested until pending work "
                    "resolves."
                ),
            )

        try:
            self.fit(observations)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    "Bayesian model fit failed: "
                    f"{type(error).__name__}: {error}"
                ),
            )

        if self._regressor is None:
            for anchor_index, anchor_key in enumerate(self._anchor_keys):
                if anchor_key in measured_keys or anchor_key in pending:
                    continue
                return CandidateSuggestion(
                    path=TuningPath.TDECQ_BAYESIAN,
                    candidate_id=self._anchor_candidate_id(
                        anchor_key,
                        anchor_index,
                        by_key,
                    ),
                    requested=self.space.point(anchor_key),
                    reason="compatible effect-memory anchor",
                )
            return FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    "insufficient numeric TDECQ observations for Bayesian fit: "
                    f"have {len(self.tdecq_training_ids)}, require "
                    f"{self.min_valid_observations}"
                ),
            )

        try:
            predictions = self.score_unmeasured(observations, pending)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    "Bayesian model scoring failed: "
                    f"{type(error).__name__}: {error}"
                ),
            )

        if not predictions:
            raise RuntimeError(
                "unmeasured non-pending finite pool produced no predictions"
            )
        epsilon = np.finfo(float).eps
        if (
            max(item.acquisition for item in predictions) <= epsilon
            or all(item.predicted_sigma_db <= epsilon for item in predictions)
        ):
            return FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    "Bayesian acquisition is materially degenerate; coordinate "
                    "fallback is required"
                ),
            )

        chosen = max(
            predictions,
            key=lambda item: (
                item.acquisition,
                tuple(-value for value in item.key),
            ),
        )
        return CandidateSuggestion(
            path=TuningPath.TDECQ_BAYESIAN,
            candidate_id=self._candidate_id(chosen.key, by_key),
            requested=self.space.point(chosen.key),
            reason="maximum TDECQ EI multiplied by valid probability",
            acquisition=chosen.acquisition,
        )

    def _complete(
        self,
        by_key: dict[PointKey, TuningObservation],
    ) -> PathComplete:
        ranked = sorted(
            (
                (observation, key)
                for key, observation in by_key.items()
                if self._has_numeric_tdecq(observation)
            ),
            key=lambda item: (
                item[0].tdecq_db,
                item[0].observation_id,
                item[1],
            ),
        )
        if not ranked:
            return PathComplete(
                path=TuningPath.TDECQ_BAYESIAN,
                status="NO_VALID_TDECQ",
                measured_count=len(by_key),
                candidate_count=self.space.candidate_count,
            )
        best = ranked[0][0]
        return PathComplete(
            path=TuningPath.TDECQ_BAYESIAN,
            status="GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
            measured_count=len(by_key),
            candidate_count=self.space.candidate_count,
            best_observation_id=best.observation_id,
            best_tdecq_db=best.tdecq_db,
        )

    def _valid_probability(self, features: np.ndarray) -> np.ndarray:
        if self._validity_mode == "all_valid":
            return np.ones(features.shape[0], dtype=float)
        if self._validity_mode == "all_invalid":
            return np.zeros(features.shape[0], dtype=float)
        if self._classifier is None:
            raise RuntimeError("validity classifier is not fitted")

        classes = np.asarray(self._classifier.classes_)
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size != 1:
            raise ValueError("validity classifier has no unique class-1 column")
        probabilities = np.asarray(
            self._classifier.predict_proba(features),
            dtype=float,
        )
        if probabilities.ndim != 2 or probabilities.shape != (
            features.shape[0],
            classes.size,
        ):
            raise ValueError("validity classifier output has an unexpected shape")
        result = probabilities[:, int(positive_columns[0])]
        if not np.all(np.isfinite(result)):
            raise ValueError("validity classifier produced nonfinite output")
        if np.any((result < 0) | (result > 1)):
            raise ValueError("validity classifier probability is outside [0, 1]")
        return result

    def _validated_pending_keys(
        self,
        pending_keys: Iterable[PointKey],
    ) -> frozenset[PointKey]:
        validated: set[PointKey] = set()
        for key in pending_keys:
            try:
                point = self.space.point(key)
            except ValueError as error:
                raise ValueError(
                    f"pending key {key!r} is not a valid safe key: {error}"
                ) from error
            validated.add(self.space.key(point))
        return frozenset(validated)

    def _anchor_candidate_id(
        self,
        key: PointKey,
        anchor_index: int,
        by_key: dict[PointKey, TuningObservation],
    ) -> str:
        payload = {
            "anchor_configuration": [
                list(anchor_key) for anchor_key in self._anchor_keys
            ],
            "anchor_priority_index": anchor_index,
            "context": self.context.model_dump(mode="json"),
            "endpoint_index": self._index_by_key[key],
            "endpoint_key": list(key),
            "history_signature": self._history_signature(by_key),
            "min_valid_observations": self.min_valid_observations,
            "parameter_names": list(self.space.names),
            "seed": self.seed,
            "space_fingerprint": self.space.fingerprint,
            "xi": self.xi,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = sha256(canonical).hexdigest()[:16]
        return (
            f"bayesian-memory-anchor-{anchor_index:08d}-"
            f"endpoint-{self._index_by_key[key]:08d}-history-{digest}"
        )

    def _candidate_id(
        self,
        key: PointKey,
        by_key: dict[PointKey, TuningObservation],
    ) -> str:
        payload = {
            "context": self.context.model_dump(mode="json"),
            "endpoint_key": list(key),
            "history_signature": self._history_signature(by_key),
            "min_valid_observations": self.min_valid_observations,
            "seed": self.seed,
            "space_fingerprint": self.space.fingerprint,
            "xi": self.xi,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = sha256(canonical).hexdigest()[:16]
        return (
            f"bayesian-endpoint-{self._index_by_key[key]:08d}-"
            f"history-{digest}"
        )

    @staticmethod
    def _history_signature(
        by_key: dict[PointKey, TuningObservation],
    ) -> str:
        authoritative_history = [
            {
                "candidate_id": observation.candidate_id,
                "key": list(key),
                "observation_id": observation.observation_id,
                "tdecq_db": observation.tdecq_db,
                "tdecq_invalid_reason": observation.tdecq_invalid_reason,
                "tdecq_status": observation.tdecq_status,
            }
            for key, observation in sorted(by_key.items())
        ]
        canonical = json.dumps(
            authoritative_history,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    @staticmethod
    def _has_numeric_tdecq(observation: TuningObservation) -> bool:
        return (
            observation.tdecq_status == "CORR"
            and observation.tdecq_db is not None
        )

    @staticmethod
    def _model_vector(
        values: object,
        *,
        expected_length: int,
        label: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (expected_length,):
            raise ValueError(f"{label} output has an unexpected shape")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} contains nonfinite output")
        return vector

    def _reset_fit_state(self) -> None:
        self.tdecq_training_ids = ()
        self._regressor = None
        self._classifier = None
        self._validity_mode = "unfitted"
        self._fitted_history_signature = None
