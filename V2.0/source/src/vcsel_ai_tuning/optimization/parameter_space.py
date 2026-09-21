from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from numbers import Integral, Real

from vcsel_ai_tuning.domain import ParameterSpec

from .protocol import OptimizationContext, TuningObservation


Point = dict[str, int]
PointKey = tuple[int, ...]


class NoTransitionPath(RuntimeError):
    """Raised when two safe endpoints are in disconnected transition regions."""


def _integer_value(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be exactly integer-valued")
    numeric = float(value)
    if not isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be exactly integer-valued")
    return int(value)


class IntegerParameterSpace:
    """Canonical finite set of safe integer tuning endpoints."""

    def __init__(
        self,
        parameters: tuple[ParameterSpec, ...],
        allowed: Callable[[Mapping[str, int]], bool] | None = None,
    ) -> None:
        if not parameters:
            raise ValueError("parameter space requires at least one parameter")

        names = tuple(parameter.name for parameter in parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")

        self.parameters = tuple(sorted(parameters, key=lambda parameter: parameter.name))
        self.names = tuple(parameter.name for parameter in self.parameters)

        axes: list[tuple[int, ...]] = []
        for parameter in self.parameters:
            if not parameter.grid_values:
                raise ValueError(
                    f"parameter {parameter.name!r} requires nonempty grid_values"
                )
            prepared_values = {
                _integer_value(
                    parameter.prepare(raw_value),
                    label=f"prepared grid value for {parameter.name}",
                )
                for raw_value in parameter.grid_values
            }
            axes.append(tuple(sorted(prepared_values)))

        self._axes = tuple(axes)
        endpoint_keys: list[PointKey] = []
        for key in product(*self._axes):
            point = self._point_from_key(key)
            if allowed is None or bool(allowed(point)):
                endpoint_keys.append(key)

        if not endpoint_keys:
            raise ValueError("parameter space has no safe endpoints")

        self._endpoint_keys = tuple(endpoint_keys)
        self._endpoint_key_set = frozenset(endpoint_keys)
        self.fingerprint = self._fingerprint()

    @property
    def candidate_count(self) -> int:
        return len(self._endpoint_keys)

    def enumerate_points(self) -> Iterator[Point]:
        for key in self._endpoint_keys:
            yield self._point_from_key(key)

    def key(self, point: Mapping[str, int]) -> PointKey:
        if not isinstance(point, Mapping):
            raise ValueError("point must be a mapping with exactly every parameter")

        supplied_names = set(point.keys())
        expected_names = set(self.names)
        if supplied_names != expected_names or len(point) != len(self.names):
            missing = tuple(sorted(expected_names - supplied_names))
            extra = tuple(sorted(supplied_names - expected_names, key=str))
            raise ValueError(
                "point must contain exactly every parameter; "
                f"missing={missing!r}, extra={extra!r}"
            )

        key = tuple(
            _integer_value(point[name], label=f"point value for {name}")
            for name in self.names
        )
        if key not in self._endpoint_key_set:
            raise ValueError(f"point {key!r} is not an allowed endpoint")
        return key

    def point(self, key: PointKey) -> Point:
        if not isinstance(key, tuple) or len(key) != len(self.names):
            raise ValueError(
                f"key must be a tuple of {len(self.names)} integer values"
            )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in key):
            raise ValueError("key must contain only integer values")

        canonical_key = tuple(int(value) for value in key)
        if canonical_key not in self._endpoint_key_set:
            raise ValueError(f"key {canonical_key!r} is not a valid safe key")
        return self._point_from_key(canonical_key)

    def local_neighbors(self, point: Mapping[str, int]) -> Iterator[Point]:
        source_key = self.key(point)
        for neighbor_key in self._local_neighbor_keys(source_key):
            yield self._point_from_key(neighbor_key)

    def adjacent_neighbors(self, point: Mapping[str, int]) -> Iterator[Point]:
        source_key = self.key(point)
        neighbor_keys: set[PointKey] = set()

        for index, (parameter, axis) in enumerate(zip(self.parameters, self._axes)):
            position = axis.index(source_key[index])
            for adjacent_position in (position - 1, position + 1):
                if adjacent_position < 0 or adjacent_position >= len(axis):
                    continue
                adjacent_value = axis[adjacent_position]
                if (
                    parameter.max_step is not None
                    and abs(adjacent_value - source_key[index]) > parameter.max_step
                ):
                    continue
                candidate = list(source_key)
                candidate[index] = adjacent_value
                candidate_key = tuple(candidate)
                if candidate_key in self._endpoint_key_set:
                    neighbor_keys.add(candidate_key)

        for neighbor_key in sorted(neighbor_keys):
            yield self._point_from_key(neighbor_key)

    def shortest_transition(
        self,
        source: Mapping[str, int],
        target: Mapping[str, int],
    ) -> tuple[Point, ...]:
        source_key = self.key(source)
        target_key = self.key(target)
        if source_key == target_key:
            return ()

        frontier = deque([source_key])
        predecessor: dict[PointKey, PointKey | None] = {source_key: None}

        while frontier:
            current = frontier.popleft()
            for neighbor in self._local_neighbor_keys(current):
                if neighbor in predecessor:
                    continue
                predecessor[neighbor] = current
                if neighbor == target_key:
                    return self._reconstruct_path(
                        source_key=source_key,
                        target_key=target_key,
                        predecessor=predecessor,
                    )
                frontier.append(neighbor)

        raise NoTransitionPath(
            f"no transition path from {source_key!r} to {target_key!r}"
        )

    def _fingerprint(self) -> str:
        payload = {
            "parameters": [
                {
                    "name": parameter.name,
                    "unit": parameter.unit,
                    "max_step": parameter.max_step,
                    "grid_values": list(axis),
                }
                for parameter, axis in zip(self.parameters, self._axes)
            ],
            "allowed_endpoint_keys": [list(key) for key in self._endpoint_keys],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def _local_neighbor_keys(self, source_key: PointKey) -> Iterator[PointKey]:
        neighbor_keys: set[PointKey] = set()
        for index, (parameter, axis) in enumerate(zip(self.parameters, self._axes)):
            source_value = source_key[index]
            for candidate_value in axis:
                if candidate_value == source_value:
                    continue
                if (
                    parameter.max_step is not None
                    and abs(candidate_value - source_value) > parameter.max_step
                ):
                    continue
                candidate = list(source_key)
                candidate[index] = candidate_value
                candidate_key = tuple(candidate)
                if candidate_key in self._endpoint_key_set:
                    neighbor_keys.add(candidate_key)

        yield from sorted(neighbor_keys)

    def _point_from_key(self, key: PointKey) -> Point:
        return dict(zip(self.names, key))

    def _reconstruct_path(
        self,
        *,
        source_key: PointKey,
        target_key: PointKey,
        predecessor: Mapping[PointKey, PointKey | None],
    ) -> tuple[Point, ...]:
        reversed_path: list[PointKey] = []
        current = target_key
        while current != source_key:
            reversed_path.append(current)
            previous = predecessor[current]
            if previous is None:
                raise RuntimeError("transition path reconstruction reached no predecessor")
            current = previous
        return tuple(
            self._point_from_key(key) for key in reversed(reversed_path)
        )


def validated_observation_map(
    space: IntegerParameterSpace,
    context: OptimizationContext,
    observations: Iterable[TuningObservation],
) -> dict[PointKey, TuningObservation]:
    validated: dict[PointKey, TuningObservation] = {}
    for observation in observations:
        if not isinstance(observation, TuningObservation):
            raise ValueError("optimizer observations must be TuningObservation values")
        if observation.context != context:
            raise ValueError(
                f"observation {observation.observation_id!r} context does not "
                "exactly match the campaign context"
            )
        try:
            key = space.key(observation.settings_readback)
        except ValueError as error:
            raise ValueError(
                f"observation {observation.observation_id!r} readback is not "
                f"an allowed endpoint: {error}"
            ) from error
        if key in validated:
            raise ValueError(
                f"duplicate/repeated observation for endpoint {key!r} is not supported"
            )
        validated[key] = observation
    return validated
