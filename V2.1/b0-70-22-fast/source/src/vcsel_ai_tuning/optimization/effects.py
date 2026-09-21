from __future__ import annotations

from collections.abc import Iterable
from math import isclose
from numbers import Integral
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_serializer,
    model_validator,
)

from .parameter_space import (
    IntegerParameterSpace,
    PointKey,
    validated_observation_map,
)
from .protocol import OptimizationContext, TuningObservation


_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_NonBlankText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_TdecqStatus = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_upper=True, min_length=1),
]
_ImmutablePointKey = tuple[StrictInt, ...]


def _validated_anchor_limit(limit: object) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, Integral)
        or int(limit) < 0
    ):
        raise ValueError("limit must be an integer greater than or equal to 0")
    return int(limit)


class ResponseNode(BaseModel):
    """One measured response retained at its authoritative readback key."""

    model_config = _MODEL_CONFIG

    key: _ImmutablePointKey
    observation_id: _NonBlankText
    tdecq_db: float | None
    tdecq_status: _TdecqStatus
    tdecq_invalid_reason: str | None = None
    er_db: float | None = None
    er_valid: bool = False
    er_status: str | None = None
    artifact_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_metric_states(self) -> "ResponseNode":
        if self.tdecq_status == "CORR" and self.tdecq_db is None:
            raise ValueError("CORR TDECQ requires a numeric value")
        if self.tdecq_status == "CORR" and self.tdecq_invalid_reason is not None:
            raise ValueError("CORR TDECQ cannot have an invalid reason")
        if self.tdecq_status != "CORR" and self.tdecq_db is not None:
            raise ValueError("non-CORR TDECQ must keep value=None")
        if self.tdecq_status != "CORR" and (
            self.tdecq_invalid_reason is None
            or not self.tdecq_invalid_reason.strip()
        ):
            raise ValueError("non-CORR TDECQ requires an invalid reason")
        if self.er_valid != (self.er_db is not None):
            raise ValueError("ER validity must match the presence of a numeric value")
        return self


class EffectEdge(BaseModel):
    """A measured one-axis response change between adjacent safe endpoints."""

    model_config = _MODEL_CONFIG

    source_key: _ImmutablePointKey
    target_key: _ImmutablePointKey
    changed_parameter: _NonBlankText
    step: StrictInt
    delta_tdecq_db: float

    @model_validator(mode="after")
    def validate_endpoints(self) -> "EffectEdge":
        if len(self.source_key) != len(self.target_key):
            raise ValueError("edge endpoint keys must have equal dimensions")
        if self.source_key >= self.target_key:
            raise ValueError("edge endpoints must use increasing canonical orientation")
        if self.step == 0:
            raise ValueError("edge step must be nonzero")
        if sum(
            source != target
            for source, target in zip(self.source_key, self.target_key)
        ) != 1:
            raise ValueError("edge endpoints must differ in exactly one parameter")
        return self


class PairInteraction(BaseModel):
    """A focused two-parameter adjacent-rectangle interaction."""

    model_config = _MODEL_CONFIG

    base_key: _ImmutablePointKey
    parameter_a: _NonBlankText
    parameter_b: _NonBlankText
    delta_delta_tdecq_db: float

    @model_validator(mode="after")
    def validate_parameter_pair(self) -> "PairInteraction":
        if self.parameter_a == self.parameter_b:
            raise ValueError("interaction parameters must be distinct")
        return self


class EffectMemory(BaseModel):
    """Versioned, context-isolated effects derived from measured responses."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["effect-memory-v1"] = "effect-memory-v1"
    context: OptimizationContext
    space_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_names: tuple[_NonBlankText, ...]
    nodes: tuple[ResponseNode, ...]
    edges: tuple[EffectEdge, ...]
    interactions: tuple[PairInteraction, ...]
    invalid_keys: frozenset[_ImmutablePointKey]

    @field_serializer("invalid_keys", when_used="json")
    def serialize_invalid_keys(
        self,
        invalid_keys: frozenset[_ImmutablePointKey],
    ) -> tuple[_ImmutablePointKey, ...]:
        return tuple(sorted(invalid_keys))

    @model_validator(mode="after")
    def validate_semantic_integrity(self) -> "EffectMemory":
        return _validate_effect_memory(self)

    @classmethod
    def build(
        cls,
        space: IntegerParameterSpace,
        observations: Iterable[TuningObservation],
    ) -> "EffectMemory":
        prepared = tuple(observations)
        if not prepared:
            raise ValueError("effect memory requires at least one observation")
        if not isinstance(prepared[0], TuningObservation):
            raise ValueError("optimizer observations must be TuningObservation values")

        context = prepared[0].context
        observation_by_key = validated_observation_map(
            space,
            context,
            prepared,
        )
        nodes = tuple(
            ResponseNode(
                key=key,
                observation_id=observation.observation_id,
                tdecq_db=observation.tdecq_db,
                tdecq_status=observation.tdecq_status,
                tdecq_invalid_reason=observation.tdecq_invalid_reason,
                er_db=observation.er_db,
                er_valid=observation.er_valid,
                er_status=observation.er_status,
                artifact_refs=observation.artifact_refs,
            )
            for key, observation in sorted(observation_by_key.items())
        )
        valid_keys = frozenset(
            key
            for key, observation in observation_by_key.items()
            if observation.tdecq_db is not None
        )

        return cls(
            context=context,
            space_fingerprint=space.fingerprint,
            parameter_names=space.names,
            nodes=nodes,
            edges=_build_edges(space, observation_by_key, valid_keys),
            interactions=_build_interactions(
                space,
                observation_by_key,
                valid_keys,
            ),
            invalid_keys=frozenset(set(observation_by_key) - valid_keys),
        )

    @classmethod
    def from_json(
        cls,
        document: str | bytes | bytearray,
        space: IntegerParameterSpace,
    ) -> "EffectMemory":
        """Safely load an external artifact against its canonical parameter space."""

        loaded = cls.model_validate_json(document)
        if not isinstance(space, IntegerParameterSpace):
            raise ValueError("effect memory requires an IntegerParameterSpace")
        if loaded.space_fingerprint != space.fingerprint:
            raise ValueError("effect memory space fingerprint does not match")
        if loaded.parameter_names != space.names:
            raise ValueError("effect memory parameter names do not match the space")

        observations: list[TuningObservation] = []
        for index, node in enumerate(loaded.nodes):
            try:
                point = space.point(node.key)
            except ValueError as error:
                raise ValueError(
                    f"effect memory node {node.key!r} is not a safe space endpoint"
                ) from error
            observations.append(
                TuningObservation(
                    observation_id=node.observation_id,
                    candidate_id=f"effect-memory-load-{index:08d}",
                    context=loaded.context,
                    settings_requested=point,
                    settings_readback=point,
                    tdecq_db=node.tdecq_db,
                    tdecq_status=node.tdecq_status,
                    tdecq_invalid_reason=node.tdecq_invalid_reason,
                    er_db=node.er_db,
                    er_valid=node.er_valid,
                    er_status=node.er_status,
                    artifact_refs=node.artifact_refs,
                )
            )

        rebuilt = cls.build(space, observations)
        if rebuilt != loaded:
            raise ValueError(
                "effect memory derived fields do not match canonical reconstruction"
            )
        return loaded

    def anchor_keys(self, limit: int) -> tuple[PointKey, ...]:
        validated_limit = _validated_anchor_limit(limit)
        ordered = sorted(
            (
                node
                for node in self.nodes
                if node.tdecq_db is not None and node.key not in self.invalid_keys
            ),
            key=lambda node: (node.tdecq_db, node.key),
        )
        return tuple(node.key for node in ordered[:validated_limit])

    def compatible_anchor_points(
        self,
        context: OptimizationContext,
        space: IntegerParameterSpace,
        limit: int,
    ) -> tuple[dict[str, int], ...]:
        """Return ranked weak-start points for an exactly compatible setup."""

        validated_limit = _validated_anchor_limit(limit)
        if not isinstance(context, OptimizationContext):
            raise ValueError("anchor context must be an OptimizationContext")
        if not isinstance(space, IntegerParameterSpace):
            raise ValueError("anchor space must be an IntegerParameterSpace")
        if context.compatibility_key != self.context.compatibility_key:
            return ()
        if (
            space.fingerprint != self.space_fingerprint
            or space.names != self.parameter_names
        ):
            return ()
        if validated_limit == 0:
            return ()

        anchors: list[dict[str, int]] = []
        seen_keys: set[PointKey] = set()
        for key in self.anchor_keys(len(self.nodes)):
            try:
                point = space.point(key)
            except ValueError:
                continue
            canonical_key = space.key(point)
            if canonical_key in seen_keys:
                continue
            seen_keys.add(canonical_key)
            anchors.append(dict(point))
            if len(anchors) == validated_limit:
                break
        return tuple(anchors)


def _finite_values_match(actual: float, expected: float) -> bool:
    return isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _validate_effect_memory(memory: EffectMemory) -> EffectMemory:
    parameter_names = memory.parameter_names
    if not parameter_names:
        raise ValueError("parameter_names must be nonempty")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be unique")

    dimension = len(parameter_names)
    node_keys = tuple(node.key for node in memory.nodes)
    if not node_keys:
        raise ValueError("effect memory requires at least one node")
    if any(len(key) != dimension for key in node_keys):
        raise ValueError("every node key must match the parameter_names dimension")
    if len(set(node_keys)) != len(node_keys):
        raise ValueError("node keys must be unique; duplicate node key found")
    if node_keys != tuple(sorted(node_keys)):
        raise ValueError("nodes must be in canonical key order")

    node_by_key = {node.key: node for node in memory.nodes}
    expected_invalid_keys = frozenset(
        node.key for node in memory.nodes if node.tdecq_db is None
    )
    if memory.invalid_keys != expected_invalid_keys:
        raise ValueError(
            "invalid_keys must exactly match nodes without valid numeric TDECQ"
        )
    valid_node_by_key = {
        key: node
        for key, node in node_by_key.items()
        if key not in expected_invalid_keys
    }

    edge_pairs = tuple(
        (edge.source_key, edge.target_key) for edge in memory.edges
    )
    if len(set(edge_pairs)) != len(edge_pairs):
        raise ValueError("effect edge endpoint pairs must be unique")
    if edge_pairs != tuple(sorted(edge_pairs)):
        raise ValueError("effect edges must be in deterministic canonical order")

    parameter_index = {
        parameter_name: index
        for index, parameter_name in enumerate(parameter_names)
    }
    edge_by_pair: dict[tuple[PointKey, PointKey], EffectEdge] = {}
    positive_edges: dict[tuple[PointKey, str], list[EffectEdge]] = {}
    for edge in memory.edges:
        source = valid_node_by_key.get(edge.source_key)
        target = valid_node_by_key.get(edge.target_key)
        if source is None or target is None:
            raise ValueError(
                "effect edge endpoints must reference existing valid nodes"
            )
        if len(edge.source_key) != dimension or len(edge.target_key) != dimension:
            raise ValueError("effect edge key dimensions must match parameter_names")

        changed_index, expected_step = _changed_coordinate(
            edge.source_key,
            edge.target_key,
        )
        expected_parameter = parameter_names[changed_index]
        if edge.changed_parameter != expected_parameter:
            raise ValueError(
                "effect edge changed_parameter does not match its changed coordinate"
            )
        if edge.step != expected_step:
            raise ValueError("effect edge step does not match its endpoint difference")
        if source.tdecq_db is None or target.tdecq_db is None:
            raise ValueError("effect edge endpoints require numeric TDECQ")
        expected_delta = target.tdecq_db - source.tdecq_db
        if not _finite_values_match(edge.delta_tdecq_db, expected_delta):
            raise ValueError(
                "effect edge delta does not match target minus source TDECQ"
            )

        pair = (edge.source_key, edge.target_key)
        edge_by_pair[pair] = edge
        positive_edges.setdefault(
            (edge.source_key, edge.changed_parameter),
            [],
        ).append(edge)

    interaction_ids = tuple(
        (
            interaction.base_key,
            interaction.parameter_a,
            interaction.parameter_b,
        )
        for interaction in memory.interactions
    )
    if len(set(interaction_ids)) != len(interaction_ids):
        raise ValueError("pair interactions must be unique")
    if interaction_ids != tuple(sorted(interaction_ids)):
        raise ValueError("pair interactions must be in deterministic canonical order")

    for interaction in memory.interactions:
        _validate_interaction(
            interaction,
            parameter_index=parameter_index,
            valid_node_by_key=valid_node_by_key,
            edge_by_pair=edge_by_pair,
            positive_edges=positive_edges,
        )

    expected_interaction_ids = _infer_interaction_ids(
        parameter_names=parameter_names,
        valid_node_by_key=valid_node_by_key,
        edge_by_pair=edge_by_pair,
        positive_edges=positive_edges,
    )
    if frozenset(interaction_ids) != expected_interaction_ids:
        raise ValueError(
            "interactions must exactly summarize every inferable measured rectangle"
        )
    return memory


def _interaction_corners(
    *,
    base_key: PointKey,
    parameter_a: str,
    parameter_b: str,
    parameter_index: dict[str, int],
    positive_edges: dict[tuple[PointKey, str], list[EffectEdge]],
) -> tuple[PointKey, PointKey, PointKey] | None:
    edges_a = positive_edges.get((base_key, parameter_a), [])
    edges_b = positive_edges.get((base_key, parameter_b), [])
    if len(edges_a) != 1 or len(edges_b) != 1:
        return None

    key_a = edges_a[0].target_key
    key_b = edges_b[0].target_key
    key_ab_values = list(base_key)
    index_a = parameter_index[parameter_a]
    index_b = parameter_index[parameter_b]
    key_ab_values[index_a] = key_a[index_a]
    key_ab_values[index_b] = key_b[index_b]
    return key_a, key_b, tuple(key_ab_values)


def _rectangle_is_evidenced(
    *,
    key_a: PointKey,
    key_b: PointKey,
    key_ab: PointKey,
    parameter_a: str,
    parameter_b: str,
    valid_node_by_key: dict[PointKey, ResponseNode],
    edge_by_pair: dict[tuple[PointKey, PointKey], EffectEdge],
) -> bool:
    if key_ab not in valid_node_by_key:
        return False
    edge_a_to_ab = edge_by_pair.get((key_a, key_ab))
    edge_b_to_ab = edge_by_pair.get((key_b, key_ab))
    return bool(
        edge_a_to_ab is not None
        and edge_a_to_ab.changed_parameter == parameter_b
        and edge_b_to_ab is not None
        and edge_b_to_ab.changed_parameter == parameter_a
    )


def _validate_interaction(
    interaction: PairInteraction,
    *,
    parameter_index: dict[str, int],
    valid_node_by_key: dict[PointKey, ResponseNode],
    edge_by_pair: dict[tuple[PointKey, PointKey], EffectEdge],
    positive_edges: dict[tuple[PointKey, str], list[EffectEdge]],
) -> None:
    base = valid_node_by_key.get(interaction.base_key)
    if base is None:
        raise ValueError(
            "pair interaction base key must reference an existing valid node"
        )
    if (
        interaction.parameter_a not in parameter_index
        or interaction.parameter_b not in parameter_index
    ):
        raise ValueError("pair interaction parameters must exist in parameter_names")
    if parameter_index[interaction.parameter_a] >= parameter_index[
        interaction.parameter_b
    ]:
        raise ValueError("pair interaction parameters must use canonical pair order")

    corners = _interaction_corners(
        base_key=interaction.base_key,
        parameter_a=interaction.parameter_a,
        parameter_b=interaction.parameter_b,
        parameter_index=parameter_index,
        positive_edges=positive_edges,
    )
    if corners is None:
        raise ValueError(
            "pair interaction must have unique positive edges from its base"
        )
    key_a, key_b, key_ab = corners
    if not _rectangle_is_evidenced(
        key_a=key_a,
        key_b=key_b,
        key_ab=key_ab,
        parameter_a=interaction.parameter_a,
        parameter_b=interaction.parameter_b,
        valid_node_by_key=valid_node_by_key,
        edge_by_pair=edge_by_pair,
    ):
        raise ValueError(
            "pair interaction corners must form a measured valid edge rectangle"
        )

    node_a = valid_node_by_key[key_a]
    node_b = valid_node_by_key[key_b]
    node_ab = valid_node_by_key[key_ab]
    if (
        base.tdecq_db is None
        or node_a.tdecq_db is None
        or node_b.tdecq_db is None
        or node_ab.tdecq_db is None
    ):
        raise ValueError("pair interaction corners require numeric TDECQ")
    expected_delta_delta = (
        node_ab.tdecq_db
        - node_a.tdecq_db
        - node_b.tdecq_db
        + base.tdecq_db
    )
    if not _finite_values_match(
        interaction.delta_delta_tdecq_db,
        expected_delta_delta,
    ):
        raise ValueError(
            "pair interaction delta-delta does not match its rectangle responses"
        )


def _infer_interaction_ids(
    *,
    parameter_names: tuple[str, ...],
    valid_node_by_key: dict[PointKey, ResponseNode],
    edge_by_pair: dict[tuple[PointKey, PointKey], EffectEdge],
    positive_edges: dict[tuple[PointKey, str], list[EffectEdge]],
) -> frozenset[tuple[PointKey, str, str]]:
    parameter_index = {
        parameter_name: index
        for index, parameter_name in enumerate(parameter_names)
    }
    inferred: set[tuple[PointKey, str, str]] = set()
    for base_key in sorted(valid_node_by_key):
        for index_a, parameter_a in enumerate(parameter_names):
            for parameter_b in parameter_names[index_a + 1 :]:
                corners = _interaction_corners(
                    base_key=base_key,
                    parameter_a=parameter_a,
                    parameter_b=parameter_b,
                    parameter_index=parameter_index,
                    positive_edges=positive_edges,
                )
                if corners is None:
                    continue
                key_a, key_b, key_ab = corners
                if _rectangle_is_evidenced(
                    key_a=key_a,
                    key_b=key_b,
                    key_ab=key_ab,
                    parameter_a=parameter_a,
                    parameter_b=parameter_b,
                    valid_node_by_key=valid_node_by_key,
                    edge_by_pair=edge_by_pair,
                ):
                    inferred.add((base_key, parameter_a, parameter_b))
    return frozenset(inferred)


def _changed_coordinate(
    source_key: PointKey,
    target_key: PointKey,
) -> tuple[int, int]:
    changes = tuple(
        (index, target - source)
        for index, (source, target) in enumerate(zip(source_key, target_key))
        if source != target
    )
    if len(changes) != 1:
        raise ValueError("adjacent endpoints must differ in exactly one parameter")
    return changes[0]


def _build_edges(
    space: IntegerParameterSpace,
    observation_by_key: dict[PointKey, TuningObservation],
    valid_keys: frozenset[PointKey],
) -> tuple[EffectEdge, ...]:
    edges: list[EffectEdge] = []
    for source_key in sorted(valid_keys):
        for neighbor in space.adjacent_neighbors(space.point(source_key)):
            target_key = space.key(neighbor)
            if target_key not in valid_keys or target_key <= source_key:
                continue
            index, step = _changed_coordinate(source_key, target_key)
            source_tdecq = observation_by_key[source_key].tdecq_db
            target_tdecq = observation_by_key[target_key].tdecq_db
            if source_tdecq is None or target_tdecq is None:
                continue
            edges.append(
                EffectEdge(
                    source_key=source_key,
                    target_key=target_key,
                    changed_parameter=space.names[index],
                    step=step,
                    delta_tdecq_db=target_tdecq - source_tdecq,
                )
            )
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.source_key,
                edge.target_key,
                edge.changed_parameter,
            ),
        )
    )


def _positive_adjacent_keys(
    space: IntegerParameterSpace,
    base_key: PointKey,
) -> dict[int, PointKey]:
    positive: dict[int, PointKey] = {}
    for neighbor in space.adjacent_neighbors(space.point(base_key)):
        neighbor_key = space.key(neighbor)
        index, step = _changed_coordinate(base_key, neighbor_key)
        if step > 0:
            positive[index] = neighbor_key
    return positive


def _build_interactions(
    space: IntegerParameterSpace,
    observation_by_key: dict[PointKey, TuningObservation],
    valid_keys: frozenset[PointKey],
) -> tuple[PairInteraction, ...]:
    interactions: list[PairInteraction] = []
    for base_key in sorted(valid_keys):
        positive = _positive_adjacent_keys(space, base_key)
        for index_a in range(len(space.names)):
            key_a = positive.get(index_a)
            if key_a is None or key_a not in valid_keys:
                continue
            for index_b in range(index_a + 1, len(space.names)):
                key_b = positive.get(index_b)
                if key_b is None or key_b not in valid_keys:
                    continue

                combined = list(base_key)
                combined[index_a] = key_a[index_a]
                combined[index_b] = key_b[index_b]
                key_ab = tuple(combined)
                try:
                    space.point(key_ab)
                except ValueError:
                    continue
                if key_ab not in valid_keys:
                    continue

                y_base = observation_by_key[base_key].tdecq_db
                y_a = observation_by_key[key_a].tdecq_db
                y_b = observation_by_key[key_b].tdecq_db
                y_ab = observation_by_key[key_ab].tdecq_db
                if any(value is None for value in (y_base, y_a, y_b, y_ab)):
                    continue
                if y_base is None or y_a is None or y_b is None or y_ab is None:
                    continue
                interactions.append(
                    PairInteraction(
                        base_key=base_key,
                        parameter_a=space.names[index_a],
                        parameter_b=space.names[index_b],
                        delta_delta_tdecq_db=y_ab - y_a - y_b + y_base,
                    )
                )
    return tuple(
        sorted(
            interactions,
            key=lambda interaction: (
                interaction.base_key,
                interaction.parameter_a,
                interaction.parameter_b,
            ),
        )
    )
