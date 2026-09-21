from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .coordinate import CoordinateDescent
from .exhaustive import ExhaustiveOptimizer
from .parameter_space import (
    IntegerParameterSpace,
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


ReplayStatus = Literal[
    "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
    "EXHAUSTIVE_INCOMPLETE",
    "BEST_TDECQ_FOUND_WITHIN_BUDGET",
    "NO_VALID_TDECQ",
]
ReplayStopReason = Literal[
    "POOL_EXHAUSTED",
    "BUDGET_EXHAUSTED",
    "NO_SAFE_CANDIDATE",
    "PATH_COMPLETE",
]
_MAX_SKLEARN_SEED = 2**32 - 1


def _snapshot_parameter_space(
    space: IntegerParameterSpace,
) -> IntegerParameterSpace:
    """Clone parameters and the exact finite authorized endpoint membership."""

    if not isinstance(space, IntegerParameterSpace):
        raise ValueError("replay surface requires an IntegerParameterSpace")
    parameters = tuple(
        parameter.model_copy(deep=True) for parameter in space.parameters
    )
    names = tuple(
        parameter.name
        for parameter in sorted(parameters, key=lambda item: item.name)
    )
    try:
        endpoint_keys = frozenset(
            tuple(point[name] for name in names)
            for point in space.enumerate_points()
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot snapshot replay parameter space: {error}") from error
    if not endpoint_keys:
        raise ValueError("replay parameter space has no safe endpoints")

    return IntegerParameterSpace(
        parameters=parameters,
        allowed=lambda point: (
            tuple(point[name] for name in names) in endpoint_keys
        ),
    )


class ReplayReport(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    path: TuningPath
    status: ReplayStatus
    stop_reason: ReplayStopReason
    seed: int = Field(ge=0, le=_MAX_SKLEARN_SEED)
    budget: int = Field(gt=0)
    measurement_count: int = Field(ge=0)
    candidate_count: int = Field(gt=0)
    invalid_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    best_observation_id: str | None
    best_tdecq_db: float | None
    grid_best_tdecq_db: float | None
    regret_db: float | None = Field(default=None, ge=0)
    reached_tdecq_below_3: bool

    @model_validator(mode="after")
    def validate_truthful_counts_and_claims(self) -> "ReplayReport":
        if self.measurement_count > self.candidate_count:
            raise ValueError("measurement_count cannot exceed candidate_count")
        if self.measurement_count > self.budget:
            raise ValueError("measurement_count cannot exceed budget")
        if self.invalid_count > self.measurement_count:
            raise ValueError("invalid_count cannot exceed measurement_count")

        has_best_id = self.best_observation_id is not None
        has_best_value = self.best_tdecq_db is not None
        if has_best_id != has_best_value:
            raise ValueError("best observation identifier and TDECQ must form a pair")
        if has_best_id and not self.best_observation_id.strip():
            raise ValueError("best_observation_id must be nonblank")
        if has_best_value != (self.regret_db is not None):
            raise ValueError("regret requires, and is required by, a best TDECQ")
        if has_best_value and self.grid_best_tdecq_db is None:
            raise ValueError("a measured best TDECQ requires a numeric grid best")
        if has_best_value and self.invalid_count >= self.measurement_count:
            raise ValueError("a measured best requires at least one valid observation")
        if not has_best_value and self.invalid_count != self.measurement_count:
            raise ValueError("no measured best requires every observation to be invalid")
        if self.best_tdecq_db is not None and self.regret_db is not None:
            if self.grid_best_tdecq_db is None:
                raise ValueError("regret requires a numeric grid-best TDECQ")
            if self.grid_best_tdecq_db > self.best_tdecq_db:
                raise ValueError("grid-best TDECQ cannot exceed the measured best")
            expected_regret = self.best_tdecq_db - self.grid_best_tdecq_db
            if not isclose(
                self.regret_db,
                expected_regret,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "regret must equal best_tdecq_db minus grid_best_tdecq_db"
                )
        if self.reached_tdecq_below_3 != bool(
            self.best_tdecq_db is not None and self.best_tdecq_db < 3.0
        ):
            raise ValueError("TDECQ-below-3 milestone must match the best TDECQ")

        full_pool = self.measurement_count == self.candidate_count
        if self.path is TuningPath.EXHAUSTIVE and self.fallback_count != 0:
            raise ValueError("exhaustive replay cannot report Bayesian fallbacks")
        if self.stop_reason == "POOL_EXHAUSTED" and not full_pool:
            raise ValueError("POOL_EXHAUSTED requires every candidate to be measured")
        if full_pool and self.stop_reason != "POOL_EXHAUSTED":
            raise ValueError("every full-pool result must report POOL_EXHAUSTED")
        if self.stop_reason == "BUDGET_EXHAUSTED" and (
            self.measurement_count != self.budget or full_pool
        ):
            raise ValueError(
                "BUDGET_EXHAUSTED requires an incomplete pool at the exact budget"
            )
        if self.stop_reason == "PATH_COMPLETE" and full_pool:
            raise ValueError("a full-pool completion must report POOL_EXHAUSTED")
        if self.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID" and (
            not full_pool
            or self.best_tdecq_db is None
            or self.grid_best_tdecq_db is None
            or self.regret_db is None
            or not isclose(self.regret_db, 0.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "grid-optimum status requires full enumeration, a best TDECQ, "
                "and zero regret"
            )
        if self.status == "NO_VALID_TDECQ" and self.best_tdecq_db is not None:
            raise ValueError("NO_VALID_TDECQ cannot include a best observation")
        if self.status == "EXHAUSTIVE_INCOMPLETE" and full_pool:
            raise ValueError("EXHAUSTIVE_INCOMPLETE requires an incomplete pool")
        if self.status == "BEST_TDECQ_FOUND_WITHIN_BUDGET" and (
            full_pool or self.best_tdecq_db is None
        ):
            raise ValueError(
                "BEST_TDECQ_FOUND_WITHIN_BUDGET requires a best TDECQ and "
                "an incomplete pool"
            )
        if full_pool and self.best_tdecq_db is not None and (
            self.status != "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
        ):
            raise ValueError(
                "a full pool with valid TDECQ must report the enumerated-grid optimum"
            )
        if full_pool and self.grid_best_tdecq_db is not None and (
            self.best_tdecq_db is None
        ):
            raise ValueError("a full pool with a numeric grid best requires a best")
        if full_pool and self.grid_best_tdecq_db is None and (
            self.status != "NO_VALID_TDECQ"
        ):
            raise ValueError(
                "a full pool without numeric grid TDECQ must report NO_VALID_TDECQ"
            )
        if (
            self.status == "EXHAUSTIVE_INCOMPLETE"
            and self.path is not TuningPath.EXHAUSTIVE
        ):
            raise ValueError("EXHAUSTIVE_INCOMPLETE is exclusive to exhaustive replay")
        if (
            self.status == "BEST_TDECQ_FOUND_WITHIN_BUDGET"
            and self.path is not TuningPath.TDECQ_BAYESIAN
        ):
            raise ValueError(
                "BEST_TDECQ_FOUND_WITHIN_BUDGET is exclusive to Bayesian replay"
            )
        return self


class ReplaySurface:
    """Immutable, readback-keyed oracle for an exact finite safe grid."""

    __slots__ = ("_space", "_context", "_by_key", "_keys")

    def __init__(
        self,
        space: IntegerParameterSpace,
        observations: tuple[TuningObservation, ...],
    ) -> None:
        internal_space = _snapshot_parameter_space(space)
        try:
            supplied = tuple(observations)
        except TypeError as error:
            raise ValueError("replay surface observations must be iterable") from error
        if not supplied:
            raise ValueError("replay surface requires observations")

        canonical: list[TuningObservation] = []
        for index, observation in enumerate(supplied, start=1):
            if not isinstance(observation, TuningObservation):
                raise ValueError(
                    f"replay surface observation {index} must be a "
                    "TuningObservation"
                )
            try:
                canonical.append(
                    TuningObservation.model_validate(
                        observation.model_dump(mode="python"),
                        strict=True,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"replay surface observation {index} is invalid: {error}"
                ) from error

        context = canonical[0].context
        by_key = validated_observation_map(internal_space, context, canonical)
        if len(by_key) != internal_space.candidate_count:
            expected_keys = {
                internal_space.key(point)
                for point in internal_space.enumerate_points()
            }
            missing = tuple(sorted(expected_keys - set(by_key)))
            raise ValueError(
                "replay surface must cover every safe endpoint exactly once; "
                f"missing readback keys={missing!r}"
            )

        ordered = dict(sorted(by_key.items()))
        self._space = internal_space
        self._context = context
        self._by_key = MappingProxyType(ordered)
        self._keys = tuple(ordered)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        space: IntegerParameterSpace,
    ) -> "ReplaySurface":
        try:
            source = Path(path)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid replay surface path {path!r}: {error}") from error
        try:
            document = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(
                f"cannot read replay surface {source}: {error}"
            ) from error

        observations: list[TuningObservation] = []
        for line_number, line in enumerate(document.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                observations.append(
                    TuningObservation.model_validate_json(line, strict=True)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid replay surface {source} line "
                    f"{line_number}: {error}"
                ) from error

        if not observations:
            raise ValueError(
                f"replay surface {source} contains no observations"
            )
        try:
            return cls(space, tuple(observations))
        except ValueError as error:
            raise ValueError(
                f"invalid replay surface {source}: {error}"
            ) from error

    @property
    def space(self) -> IntegerParameterSpace:
        return _snapshot_parameter_space(self._space)

    @property
    def context(self) -> OptimizationContext:
        return self._context

    @property
    def keys(self) -> tuple[PointKey, ...]:
        return self._keys

    def observe(
        self,
        suggestion: CandidateSuggestion,
        sequence: int,
    ) -> TuningObservation:
        if not isinstance(suggestion, CandidateSuggestion):
            raise ValueError("replay observation requires a CandidateSuggestion")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("replay sequence must be a nonnegative integer")
        try:
            key = self._space.key(suggestion.requested)
        except ValueError as error:
            raise ValueError(
                f"replay suggestion is not a safe endpoint: {error}"
            ) from error

        source = self._by_key[key]
        payload = source.model_dump(mode="python")
        payload.update(
            {
                "observation_id": f"replay-{sequence:08d}",
                "candidate_id": suggestion.candidate_id,
                "settings_requested": dict(suggestion.requested),
            }
        )
        try:
            return TuningObservation.model_validate(payload, strict=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"replay suggestion cannot produce a valid observation: {error}"
            ) from error

    @property
    def grid_best_tdecq_db(self) -> float | None:
        values = tuple(
            float(observation.tdecq_db)
            for observation in self._by_key.values()
            if observation.tdecq_status == "CORR"
            and observation.tdecq_db is not None
        )
        return min(values) if values else None

    def with_er_values(
        self,
        values: Mapping[PointKey, float],
    ) -> "ReplaySurface":
        if not isinstance(values, Mapping):
            raise ValueError("ER replacements must be a mapping")

        supplied_keys = tuple(values.keys())
        keys_are_strict = all(
            isinstance(key, tuple)
            and len(key) == len(self._space.names)
            and all(type(item) is int for item in key)
            for key in supplied_keys
        )
        if not keys_are_strict or set(supplied_keys) != set(self._keys):
            raise ValueError(
                "ER replacements must contain exactly every safe endpoint key"
            )

        replacements: dict[PointKey, float] = {}
        for key in self._keys:
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"ER replacement for {key!r} must be a finite numeric value"
                )
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(
                    f"ER replacement for {key!r} must be a finite numeric value"
                )
            replacements[key] = numeric

        observations: list[TuningObservation] = []
        for key, source in self._by_key.items():
            payload = source.model_dump(mode="python")
            payload.update(
                {
                    "er_db": replacements[key],
                    "er_valid": True,
                    "er_status": "CORR",
                }
            )
            try:
                observations.append(
                    TuningObservation.model_validate(payload, strict=True)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"ER replacement for {key!r} violates observation validity: "
                    f"{error}"
                ) from error
        return ReplaySurface(self._space, tuple(observations))


def _best(
    observations: tuple[TuningObservation, ...],
) -> TuningObservation | None:
    valid = tuple(
        observation
        for observation in observations
        if observation.tdecq_status == "CORR"
        and observation.tdecq_db is not None
    )
    return (
        min(valid, key=lambda item: (item.tdecq_db, item.observation_id))
        if valid
        else None
    )


def _validate_terminal(
    terminal: PathComplete,
    *,
    path: TuningPath,
    observations: tuple[TuningObservation, ...],
    candidate_count: int,
) -> PathComplete:
    try:
        validated = PathComplete.model_validate(
            terminal.model_dump(mode="python"),
            strict=True,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"optimizer returned an invalid PathComplete: {error}") from error
    if validated.path is not path:
        raise ValueError("optimizer PathComplete does not match the replay path")
    if validated.measured_count != len(observations):
        raise ValueError("optimizer PathComplete measured_count does not match replay")
    if validated.candidate_count != candidate_count:
        raise ValueError("optimizer PathComplete candidate_count does not match replay")

    best = _best(observations)
    expected_id = best.observation_id if best is not None else None
    expected_tdecq = best.tdecq_db if best is not None else None
    if (
        validated.best_observation_id != expected_id
        or validated.best_tdecq_db != expected_tdecq
    ):
        raise ValueError("optimizer PathComplete best observation does not match replay")
    return validated


def _bayesian_partial_terminal(
    observations: tuple[TuningObservation, ...],
    candidate_count: int,
) -> PathComplete:
    best = _best(observations)
    return PathComplete(
        path=TuningPath.TDECQ_BAYESIAN,
        status=(
            "BEST_TDECQ_FOUND_WITHIN_BUDGET"
            if best is not None
            else "NO_VALID_TDECQ"
        ),
        measured_count=len(observations),
        candidate_count=candidate_count,
        best_observation_id=best.observation_id if best is not None else None,
        best_tdecq_db=best.tdecq_db if best is not None else None,
    )


def replay(
    surface: ReplaySurface,
    path: TuningPath,
    seed: int,
    budget: int,
) -> ReplayReport:
    if not isinstance(surface, ReplaySurface):
        raise ValueError("replay requires a ReplaySurface")
    if not isinstance(path, TuningPath):
        raise ValueError("replay path must be a TuningPath enum value")
    if type(seed) is not int:
        raise ValueError("replay seed must be an integer, not bool")
    if seed < 0 or seed > _MAX_SKLEARN_SEED:
        raise ValueError(
            "replay seed must be in the inclusive range "
            f"0..{_MAX_SKLEARN_SEED}"
        )
    if type(budget) is not int or budget <= 0:
        raise ValueError("replay budget must be a positive integer, not bool")

    grid_best = surface.grid_best_tdecq_db
    space = surface._space
    candidate_count = space.candidate_count
    coordinate: CoordinateDescent | None = None
    if path is TuningPath.EXHAUSTIVE:
        optimizer = ExhaustiveOptimizer(space, context=surface.context)
    else:
        from .bayesian import TdecqBayesianOptimizer

        optimizer = TdecqBayesianOptimizer(
            space,
            context=surface.context,
            seed=seed,
        )
        coordinate = CoordinateDescent(
            space,
            context=surface.context,
            seed=seed,
        )

    observations: tuple[TuningObservation, ...] = ()
    measured_keys: set[PointKey] = set()
    fallback_count = 0
    terminal: PathComplete | None = None
    stop_reason: ReplayStopReason | None = None
    no_pending_points: frozenset[PointKey] = frozenset()

    while len(observations) < budget:
        suggestion = optimizer.suggest(observations, no_pending_points)
        if isinstance(suggestion, FallbackRequested):
            if suggestion.path is not path:
                raise ValueError("optimizer fallback path does not match replay path")
            if path is not TuningPath.TDECQ_BAYESIAN or coordinate is None:
                raise ValueError("fallback is only valid for Bayesian replay")
            fallback_count += 1
            suggestion = coordinate.suggest(observations, no_pending_points)

        if isinstance(suggestion, CandidateSuggestion):
            if suggestion.path is not path:
                raise ValueError("optimizer candidate does not match the replay path")
            key = space.key(suggestion.requested)
            if key in measured_keys:
                raise ValueError(
                    f"optimizer repeated already measured replay endpoint {key!r}"
                )
            observation = surface.observe(suggestion, len(observations))
            if space.key(observation.settings_readback) != key:
                raise ValueError("replay oracle readback does not match its lookup key")
            observations = (*observations, observation)
            measured_keys.add(key)
            continue

        if isinstance(suggestion, PathComplete):
            terminal = _validate_terminal(
                suggestion,
                path=path,
                observations=observations,
                candidate_count=candidate_count,
            )
            stop_reason = (
                "POOL_EXHAUSTED"
                if len(observations) == candidate_count
                else "PATH_COMPLETE"
            )
            break

        if isinstance(suggestion, NoSafeCandidate):
            if suggestion.path is not path:
                raise ValueError("NoSafeCandidate does not match the replay path")
            stop_reason = "NO_SAFE_CANDIDATE"
            break

        raise TypeError(
            f"optimizer returned unsupported replay suggestion {type(suggestion).__name__}"
        )

    if stop_reason is None:
        if len(observations) == candidate_count:
            completion = optimizer.suggest(observations, no_pending_points)
            if not isinstance(completion, PathComplete):
                raise ValueError(
                    "optimizer did not return PathComplete after full-pool replay"
                )
            terminal = _validate_terminal(
                completion,
                path=path,
                observations=observations,
                candidate_count=candidate_count,
            )
            stop_reason = "POOL_EXHAUSTED"
        else:
            stop_reason = "BUDGET_EXHAUSTED"

    if terminal is None:
        if path is TuningPath.EXHAUSTIVE:
            terminal = optimizer.finish(observations, budget_exhausted=True)
        else:
            terminal = _bayesian_partial_terminal(observations, candidate_count)
        terminal = _validate_terminal(
            terminal,
            path=path,
            observations=observations,
            candidate_count=candidate_count,
        )

    best = _best(observations)
    regret: float | None = None
    if best is not None and best.tdecq_db is not None:
        if grid_best is None:
            raise ValueError("measured valid TDECQ requires a numeric grid best")
        regret = float(best.tdecq_db) - grid_best
        if regret < 0:
            raise ValueError("replay regret cannot be negative against the full grid")

    return ReplayReport(
        path=path,
        status=terminal.status,
        stop_reason=stop_reason,
        seed=seed,
        budget=budget,
        measurement_count=len(observations),
        candidate_count=candidate_count,
        invalid_count=sum(
            observation.tdecq_status != "CORR"
            or observation.tdecq_db is None
            for observation in observations
        ),
        fallback_count=fallback_count,
        best_observation_id=best.observation_id if best is not None else None,
        best_tdecq_db=best.tdecq_db if best is not None else None,
        grid_best_tdecq_db=grid_best,
        regret_db=regret,
        reached_tdecq_below_3=bool(
            best is not None
            and best.tdecq_db is not None
            and best.tdecq_db < 3.0
        ),
    )
