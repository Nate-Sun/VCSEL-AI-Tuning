from __future__ import annotations

import json
import random

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.effects import (
    EffectEdge,
    EffectMemory,
    PairInteraction,
    ResponseNode,
)
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import OptimizationContext


def _parameter(
    name: str,
    values: tuple[int, ...],
    *,
    max_step: int = 1,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        unit="count",
        lower=min(values),
        upper=max(values),
        max_step=max_step,
        grid_values=values,
    )


def _memory(space, observations) -> EffectMemory:
    return EffectMemory.build(space, observations)


def _json_payload(memory: EffectMemory) -> dict:
    return json.loads(memory.model_dump_json())


def _json_document(payload: dict) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def test_build_rejects_empty_observations(two_by_two_space):
    with pytest.raises(ValueError, match="at least one observation|empty"):
        _memory(two_by_two_space, ())


def test_edges_are_unique_oriented_one_coordinate_adjacent_effects(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)

    edge = next(
        edge
        for edge in memory.edges
        if edge.source_key == (0, 0) and edge.target_key == (1, 0)
    )
    assert edge.changed_parameter == "eq"
    assert edge.step == 1
    assert edge.delta_tdecq_db == pytest.approx(-0.4)
    assert len(memory.edges) == 4
    assert len({(edge.source_key, edge.target_key) for edge in memory.edges}) == 4
    assert all(edge.source_key < edge.target_key for edge in memory.edges)
    assert all(
        sum(
            source != target
            for source, target in zip(edge.source_key, edge.target_key)
        )
        == 1
        for edge in memory.edges
    )


def test_pair_interaction_is_adjacent_rectangle_delta_delta(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)

    assert len(memory.interactions) == 1
    interaction = memory.interactions[0]
    assert interaction.base_key == (0, 0)
    assert (interaction.parameter_a, interaction.parameter_b) == (
        "eq",
        "ibias_dac",
    )
    assert interaction.delta_delta_tdecq_db == pytest.approx(
        2.3 - 2.8 - 3.0 + 3.2
    )


def test_invalid_point_is_remembered_but_excluded_from_derived_valid_effects(
    two_by_two_space,
    surface_with_invalid_best,
):
    memory = _memory(two_by_two_space, surface_with_invalid_best)

    assert memory.invalid_keys == frozenset({(1, 1)})
    assert (1, 1) not in memory.anchor_keys(10)
    assert all(
        (1, 1) not in (edge.source_key, edge.target_key)
        for edge in memory.edges
    )
    assert memory.interactions == ()


def test_nodes_preserve_artifacts_invalid_reason_and_report_only_er(
    two_by_two_space,
    surface_with_invalid_best,
):
    artifact = "measurements/obs-11.json#sha256=" + "b" * 64
    observations = surface_with_invalid_best[:-1] + (
        surface_with_invalid_best[-1].model_copy(
            update={"artifact_refs": (artifact,), "er_db": 1234.5}
        ),
    )

    memory = _memory(two_by_two_space, observations)
    node = next(node for node in memory.nodes if node.key == (1, 1))

    assert node.observation_id == "obs-11"
    assert node.tdecq_db is None
    assert node.tdecq_status == "INV"
    assert node.tdecq_invalid_reason == "timeout waiting for CORR"
    assert node.er_db == 1234.5
    assert node.er_valid is True
    assert node.er_status == "CORR"
    assert node.artifact_refs == (artifact,)


def test_every_public_model_and_memory_round_trip_exactly_as_json(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)
    values = (
        (ResponseNode, memory.nodes[0]),
        (EffectEdge, memory.edges[0]),
        (PairInteraction, memory.interactions[0]),
        (EffectMemory, memory),
    )

    for model, value in values:
        assert model.model_validate_json(value.model_dump_json()) == value


def test_json_validation_rejects_duplicate_node_key(
    two_by_two_space,
    complete_surface,
):
    payload = _json_payload(_memory(two_by_two_space, complete_surface))
    payload["nodes"].append(dict(payload["nodes"][0]))

    with pytest.raises(ValidationError, match="node.*unique|duplicate.*node"):
        EffectMemory.model_validate_json(_json_document(payload))


def test_json_validation_rejects_invalid_keys_mismatch(
    two_by_two_space,
    surface_with_invalid_best,
):
    payload = _json_payload(_memory(two_by_two_space, surface_with_invalid_best))
    payload["invalid_keys"] = []

    with pytest.raises(ValidationError, match="invalid_keys"):
        EffectMemory.model_validate_json(_json_document(payload))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("changed_parameter", "eq", id="changed-parameter"),
        pytest.param("step", 2, id="step"),
        pytest.param("delta_tdecq_db", 123.0, id="delta"),
    ],
)
def test_json_validation_rejects_corrupt_edge_semantics(
    two_by_two_space,
    complete_surface,
    field,
    replacement,
):
    payload = _json_payload(_memory(two_by_two_space, complete_surface))
    payload["edges"][0][field] = replacement

    with pytest.raises(ValidationError, match="edge|parameter|step|delta"):
        EffectMemory.model_validate_json(_json_document(payload))


def test_safe_loader_rejects_internally_plausible_nonadjacent_edge(
    context,
    tuning_observation,
):
    space = IntegerParameterSpace(
        parameters=(_parameter("eq", (0, 1, 2), max_step=2),)
    )
    observations = tuple(
        tuning_observation(
            context,
            point,
            observation_id=f"obs-{point['eq']}",
            tdecq_db=float(3 - point["eq"]),
        )
        for point in space.enumerate_points()
    )
    payload = _json_payload(_memory(space, observations))
    payload["edges"] = [
        {
            "source_key": [0],
            "target_key": [2],
            "changed_parameter": "eq",
            "step": 2,
            "delta_tdecq_db": -2.0,
        }
    ]
    document = _json_document(payload)

    assert EffectMemory.model_validate_json(document).edges[0].step == 2
    with pytest.raises(ValueError, match="canonical|recomputed|derived|space"):
        EffectMemory.from_json(document, space)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda payload: payload.__setitem__("interactions", []),
            id="missing",
        ),
        pytest.param(
            lambda payload: payload["interactions"].append(
                {
                    "base_key": [1, 1],
                    "parameter_a": "eq",
                    "parameter_b": "ibias_dac",
                    "delta_delta_tdecq_db": 0.0,
                }
            ),
            id="extra-impossible",
        ),
        pytest.param(
            lambda payload: payload["interactions"][0].__setitem__(
                "delta_delta_tdecq_db", 99.0
            ),
            id="wrong-formula",
        ),
    ],
)
def test_json_validation_rejects_missing_extra_or_impossible_interaction(
    two_by_two_space,
    complete_surface,
    mutation,
):
    payload = _json_payload(_memory(two_by_two_space, complete_surface))
    mutation(payload)

    with pytest.raises(ValidationError, match="interaction|rectangle|delta"):
        EffectMemory.model_validate_json(_json_document(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda payload: payload.__setitem__("space_fingerprint", "f" * 64),
            id="fingerprint",
        ),
        pytest.param(
            lambda payload: payload.__setitem__(
                "parameter_names", list(reversed(payload["parameter_names"]))
            ),
            id="parameter-names",
        ),
        pytest.param(
            lambda payload: payload["nodes"][0].__setitem__("key", [9, 9]),
            id="unsafe-node",
        ),
    ],
)
def test_safe_loader_rejects_space_identity_or_safe_endpoint_mismatch(
    two_by_two_space,
    complete_surface,
    mutation,
):
    memory = _memory(two_by_two_space, complete_surface[:1])
    payload = _json_payload(memory)
    mutation(payload)

    with pytest.raises(ValueError, match="fingerprint|parameter|safe|endpoint|space"):
        EffectMemory.from_json(_json_document(payload), two_by_two_space)


def test_safe_loader_round_trips_valid_full_and_partial_artifacts_exactly(
    two_by_two_space,
    complete_surface,
):
    for observations in (complete_surface, complete_surface[:2]):
        memory = _memory(two_by_two_space, observations)

        loaded = EffectMemory.from_json(memory.model_dump_json(), two_by_two_space)

        assert loaded == memory
        assert loaded.model_dump_json() == memory.model_dump_json()


def test_effect_memory_rejects_unsupported_schema_version(
    two_by_two_space,
    complete_surface,
):
    payload = _memory(two_by_two_space, complete_surface).model_dump()
    payload["schema_version"] = "effect-memory-v2"

    with pytest.raises(ValidationError, match="schema_version|effect-memory-v1"):
        EffectMemory.model_validate(payload)


def test_cross_constraint_missing_ab_corner_skips_interaction(
    context,
    tuning_observation,
):
    space = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1)),
            _parameter("ibias_dac", (0, 1)),
        ),
        allowed=lambda point: point != {"eq": 1, "ibias_dac": 1},
    )
    observations = tuple(
        tuning_observation(
            context,
            point,
            observation_id=f"obs-{space.key(point)}",
            tdecq_db=3.0,
        )
        for point in space.enumerate_points()
    )

    assert _memory(space, observations).interactions == ()


def test_sparse_surface_does_not_bridge_unmeasured_intermediate_axis_value(
    context,
    tuning_observation,
):
    space = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 2), max_step=2),
            _parameter("ibias_dac", (0, 1)),
        )
    )
    measured_keys = ((0, 0), (0, 1), (2, 0), (2, 1))
    observations = tuple(
        tuning_observation(
            context,
            space.point(key),
            observation_id=f"obs-{key}",
            tdecq_db=float(sum(key)),
        )
        for key in measured_keys
    )

    memory = _memory(space, observations)

    assert memory.interactions == ()
    assert all(
        abs(edge.target_key[0] - edge.source_key[0]) != 2
        for edge in memory.edges
    )


def test_invalid_rectangle_corner_skips_interaction(
    two_by_two_space,
    surface_with_invalid_best,
):
    assert _memory(two_by_two_space, surface_with_invalid_best).interactions == ()


def test_edges_use_adjacent_neighbors_even_when_max_step_spans_two_grid_cells(
    context,
    tuning_observation,
):
    space = IntegerParameterSpace(
        parameters=(_parameter("eq", (0, 1, 2), max_step=2),)
    )
    observations = tuple(
        tuning_observation(
            context,
            point,
            observation_id=f"obs-{point['eq']}",
            tdecq_db=float(3 - point["eq"]),
        )
        for point in space.enumerate_points()
    )

    memory = _memory(space, observations)

    assert tuple((edge.source_key, edge.target_key) for edge in memory.edges) == (
        ((0,), (1,)),
        ((1,), (2,)),
    )


def test_context_mismatch_is_rejected(
    two_by_two_space,
    complete_surface,
):
    other_context = complete_surface[-1].context.model_copy(
        update={"device_id": "vcsel-999"}
    )
    mismatched = complete_surface[:-1] + (
        complete_surface[-1].model_copy(update={"context": other_context}),
    )

    with pytest.raises(ValueError, match="context.*exactly match"):
        _memory(two_by_two_space, mismatched)


def test_duplicate_readback_endpoint_is_rejected(
    two_by_two_space,
    complete_surface,
):
    duplicate = complete_surface[0].model_copy(
        update={"observation_id": "obs-duplicate", "candidate_id": "duplicate"}
    )

    with pytest.raises(ValueError, match="duplicate|repeated"):
        _memory(two_by_two_space, complete_surface + (duplicate,))


def test_unsafe_readback_endpoint_is_rejected(
    two_by_two_space,
    complete_surface,
):
    unsafe = complete_surface[0].model_copy(
        update={"settings_readback": {"eq": 9, "ibias_dac": 9}}
    )

    with pytest.raises(ValueError, match="readback|allowed endpoint"):
        _memory(two_by_two_space, (unsafe,))


def test_artifact_is_deterministic_under_reversed_and_shuffled_input_order(
    two_by_two_space,
    complete_surface,
):
    shuffled = list(complete_surface)
    random.Random(20260722).shuffle(shuffled)

    canonical = _memory(two_by_two_space, complete_surface)
    reversed_memory = _memory(two_by_two_space, tuple(reversed(complete_surface)))
    shuffled_memory = _memory(two_by_two_space, shuffled)

    assert reversed_memory == canonical
    assert shuffled_memory == canonical
    assert reversed_memory.model_dump_json() == canonical.model_dump_json()
    assert shuffled_memory.model_dump_json() == canonical.model_dump_json()


@pytest.mark.parametrize("limit", [True, False, -1, 1.0, "1", None])
def test_anchor_limit_rejects_bool_negative_and_noninteger_values(
    two_by_two_space,
    complete_surface,
    limit,
):
    memory = _memory(two_by_two_space, complete_surface)

    with pytest.raises(ValueError, match="limit.*integer.*0|integer.*nonnegative"):
        memory.anchor_keys(limit)


def test_anchor_limit_zero_and_larger_than_valid_count(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)

    assert memory.anchor_keys(0) == ()
    assert memory.anchor_keys(99) == ((1, 1), (1, 0), (0, 1), (0, 0))


def test_anchors_sort_by_tdecq_then_key_and_ignore_er(
    two_by_two_space,
    complete_surface,
):
    tied = tuple(
        observation.model_copy(update={"tdecq_db": 2.5})
        for observation in complete_surface
    )
    er_mutated = tuple(
        observation.model_copy(update={"er_db": (-1.0 if index % 2 else 1.0) * 1e300})
        for index, observation in enumerate(tied)
    )

    expected = ((0, 0), (0, 1), (1, 0), (1, 1))
    assert _memory(two_by_two_space, tied).anchor_keys(4) == expected
    assert _memory(two_by_two_space, er_mutated).anchor_keys(4) == expected


def test_compatible_anchor_points_allow_a_new_device_and_keep_ranked_priority(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)
    replacement_device = memory.context.model_copy(
        update={"device_id": "vcsel-002"}
    )

    anchors = memory.compatible_anchor_points(
        replacement_device,
        two_by_two_space,
        2,
    )

    assert anchors == (
        {"eq": 1, "ibias_dac": 1},
        {"eq": 1, "ibias_dac": 0},
    )


@pytest.mark.parametrize(
    "context_update",
    [
        pytest.param({}, id="same-physical-context"),
        pytest.param({"device_id": "vcsel-002"}, id="compatible-new-device"),
    ],
)
def test_compatible_anchor_points_accept_same_compatibility_key(
    two_by_two_space,
    complete_surface,
    context_update,
):
    memory = _memory(two_by_two_space, complete_surface)
    target_context = memory.context.model_copy(update=context_update)

    assert memory.compatible_anchor_points(
        target_context,
        two_by_two_space,
        1,
    ) == ({"eq": 1, "ibias_dac": 1},)


@pytest.mark.parametrize(
    "context_update",
    [
        pytest.param({"driver_profile_hash": "b" * 64}, id="driver-profile"),
        pytest.param({"vcsel_model": "OTHER"}, id="vcsel-model"),
        pytest.param({"test_context_id": "other-test"}, id="test-context"),
    ],
)
def test_compatible_anchor_points_reject_incompatible_context_without_error(
    two_by_two_space,
    complete_surface,
    context_update,
):
    memory = _memory(two_by_two_space, complete_surface)
    target_context = memory.context.model_copy(update=context_update)

    assert memory.compatible_anchor_points(
        target_context,
        two_by_two_space,
        4,
    ) == ()


def test_compatible_anchor_points_reject_different_space_without_error(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)
    different_space = IntegerParameterSpace(
        parameters=(
            _parameter("eq", (0, 1, 2)),
            _parameter("ibias_dac", (0, 1)),
        )
    )

    assert memory.compatible_anchor_points(
        memory.context,
        different_space,
        4,
    ) == ()


def test_compatible_anchor_points_skip_invalid_or_impossible_nodes_and_ignore_er(
    two_by_two_space,
    complete_surface,
    surface_with_invalid_best,
):
    valid_memory = _memory(two_by_two_space, complete_surface)
    er_mutated = tuple(
        observation.model_copy(update={"er_db": 1e100 - index})
        for index, observation in enumerate(complete_surface)
    )
    invalid_memory = _memory(two_by_two_space, surface_with_invalid_best)
    impossible_node = valid_memory.nodes[-1].model_copy(update={"key": (9, 9)})
    defensively_corrupt = valid_memory.model_copy(
        update={"nodes": (impossible_node,) + valid_memory.nodes[:-1]}
    )

    expected = valid_memory.compatible_anchor_points(
        valid_memory.context,
        two_by_two_space,
        4,
    )
    assert _memory(two_by_two_space, er_mutated).compatible_anchor_points(
        valid_memory.context,
        two_by_two_space,
        4,
    ) == expected
    assert {"eq": 1, "ibias_dac": 1} not in invalid_memory.compatible_anchor_points(
        valid_memory.context,
        two_by_two_space,
        4,
    )
    assert {"eq": 9, "ibias_dac": 9} not in defensively_corrupt.compatible_anchor_points(
        valid_memory.context,
        two_by_two_space,
        4,
    )


def test_compatible_anchor_points_return_fresh_defensive_point_dicts(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)
    first = memory.compatible_anchor_points(
        memory.context,
        two_by_two_space,
        1,
    )
    first[0]["eq"] = 0

    second = memory.compatible_anchor_points(
        memory.context,
        two_by_two_space,
        1,
    )

    assert second == ({"eq": 1, "ibias_dac": 1},)
    assert first[0] is not second[0]


def test_compatible_anchor_points_filter_and_deduplicate_before_limit(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)
    best = next(node for node in memory.nodes if node.key == (1, 1))
    unsafe_best = best.model_copy(
        update={"key": (9, 9), "tdecq_db": 0.1}
    )
    duplicate_best = best.model_copy(update={"tdecq_db": 0.2})
    corrupt = memory.model_copy(
        update={"nodes": (unsafe_best, duplicate_best, *memory.nodes)}
    )

    assert corrupt.compatible_anchor_points(
        memory.context,
        two_by_two_space,
        2,
    ) == (
        {"eq": 1, "ibias_dac": 1},
        {"eq": 1, "ibias_dac": 0},
    )


@pytest.mark.parametrize("limit", [True, False, -1, 1.0, "1", None])
def test_compatible_anchor_points_reuse_strict_limit_validation(
    two_by_two_space,
    complete_surface,
    limit,
):
    memory = _memory(two_by_two_space, complete_surface)

    with pytest.raises(ValueError, match="limit.*integer.*0|integer.*nonnegative"):
        memory.compatible_anchor_points(
            memory.context,
            two_by_two_space,
            limit,
        )


def test_compatible_anchor_points_limit_zero_returns_no_points(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)

    assert memory.compatible_anchor_points(
        memory.context,
        two_by_two_space,
        0,
    ) == ()


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        pytest.param(
            ResponseNode,
            {
                "key": (0,),
                "observation_id": "obs",
                "tdecq_db": float("inf"),
                "tdecq_status": "CORR",
                "er_valid": False,
                "artifact_refs": (),
            },
            id="response-node",
        ),
        pytest.param(
            EffectEdge,
            {
                "source_key": (0,),
                "target_key": (1,),
                "changed_parameter": "eq",
                "step": 1,
                "delta_tdecq_db": float("nan"),
            },
            id="effect-edge",
        ),
        pytest.param(
            PairInteraction,
            {
                "base_key": (0, 0),
                "parameter_a": "eq",
                "parameter_b": "ibias_dac",
                "delta_delta_tdecq_db": float("-inf"),
            },
            id="pair-interaction",
        ),
    ],
)
def test_public_numeric_models_reject_nonfinite_values(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_public_models_are_frozen_and_forbid_extra_fields(
    two_by_two_space,
    complete_surface,
):
    memory = _memory(two_by_two_space, complete_surface)
    values = (
        (ResponseNode, memory.nodes[0]),
        (EffectEdge, memory.edges[0]),
        (PairInteraction, memory.interactions[0]),
        (EffectMemory, memory),
    )

    for model, value in values:
        with pytest.raises(ValidationError):
            model.model_validate({**value.model_dump(), "unexpected": True})
        with pytest.raises(ValidationError):
            value.__setattr__(next(iter(model.model_fields)), "changed")
