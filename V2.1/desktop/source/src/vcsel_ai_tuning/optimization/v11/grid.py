from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from typing import Self

from pydantic import StrictInt, model_validator

from .models import FrozenModel, Identifier


class TuningParameter(FrozenModel):
    """One enabled-or-disabled exact integer tuning axis."""

    name: Identifier
    tuning_min: StrictInt
    tuning_max: StrictInt
    tuning_step: StrictInt
    enabled: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.tuning_min > self.tuning_max:
            raise ValueError("tuning_min must not exceed tuning_max")
        if self.tuning_step <= 0:
            raise ValueError("tuning_step must be positive")
        return self

    @property
    def axis_length(self) -> int:
        return ((self.tuning_max - self.tuning_min) // self.tuning_step) + 1

    @property
    def grid_values(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.tuning_min,
                self.tuning_max + 1,
                self.tuning_step,
            )
        )

    def value_at(self, position: int) -> int:
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError("axis position must be an integer")
        if position < 0 or position >= self.axis_length:
            raise IndexError("axis position is out of range")
        return self.tuning_min + (position * self.tuning_step)

    def position_of(self, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("grid values must be integers")
        offset = value - self.tuning_min
        if (
            value < self.tuning_min
            or value > self.tuning_max
            or offset % self.tuning_step != 0
        ):
            raise ValueError(f"{value} is not on the {self.name!r} grid")
        return offset // self.tuning_step


class DiscreteParameterSpace:
    """A canonical Cartesian grid addressed by lazy mixed-radix indices."""

    def __init__(self, parameters: tuple[TuningParameter, ...]) -> None:
        if not isinstance(parameters, tuple) or not parameters:
            raise ValueError("parameters must be a non-empty tuple")
        if not all(isinstance(item, TuningParameter) for item in parameters):
            raise TypeError("parameters must contain TuningParameter values")

        all_names = tuple(item.name for item in parameters)
        if len(set(all_names)) != len(all_names):
            raise ValueError("parameter names must be unique")

        enabled = tuple(sorted(
            (item for item in parameters if item.enabled),
            key=lambda item: item.name,
        ))
        if not enabled:
            raise ValueError("at least one parameter must be enabled")

        self._parameters = enabled
        self._axis_lengths = tuple(item.axis_length for item in enabled)
        self._point_count = math.prod(self._axis_lengths)

    @property
    def parameters(self) -> tuple[TuningParameter, ...]:
        return self._parameters

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._parameters)

    @property
    def point_count(self) -> int:
        return self._point_count

    def point_from_index(self, original_index: int) -> dict[str, int]:
        if not isinstance(original_index, int) or isinstance(original_index, bool):
            raise TypeError("point index must be an integer")
        if original_index < 0 or original_index >= self.point_count:
            raise IndexError("point index is out of range")

        remainder = original_index
        positions: list[int] = []
        for axis_length in reversed(self._axis_lengths):
            remainder, position = divmod(remainder, axis_length)
            positions.append(position)

        return {
            parameter.name: parameter.value_at(position)
            for parameter, position in zip(
                self._parameters,
                reversed(positions),
                strict=True,
            )
        }

    def index(self, point: Mapping[str, int]) -> int:
        if not isinstance(point, Mapping):
            raise TypeError("point must be a mapping")
        if set(point) != set(self.names) or len(point) != len(self.names):
            raise ValueError("point must contain exactly the enabled parameters")

        original_index = 0
        for parameter in self._parameters:
            position = parameter.position_of(point[parameter.name])
            original_index = (
                original_index * parameter.axis_length
            ) + position
        return original_index

    def iter_points(self) -> Iterator[dict[str, int]]:
        for original_index in range(self.point_count):
            yield self.point_from_index(original_index)


def seeded_permutation_indices(count: int, seed: int) -> Iterator[int]:
    """Yield a deterministic affine permutation without storing ``count`` items."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if count == 1:
        yield 0
        return

    seed_bytes = str(seed).encode("ascii")
    offset_digest = hashlib.sha256(b"offset:" + seed_bytes).digest()
    stride_digest = hashlib.sha256(b"stride:" + seed_bytes).digest()
    offset = int.from_bytes(offset_digest, "big") % count
    stride = int.from_bytes(stride_digest, "big") % count
    if stride == 0:
        stride = 1
    while math.gcd(stride, count) != 1:
        stride += 1
        if stride == count:
            stride = 1

    for execution_index in range(count):
        yield (offset + (execution_index * stride)) % count
