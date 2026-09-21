from importlib import import_module
from pathlib import Path

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    PathComplete,
    TuningObservation,
    TuningPath,
)
from vcsel_ai_tuning.optimization.replay import ReplayReport, ReplaySurface, replay


replay_module = import_module("vcsel_ai_tuning.optimization.replay")


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "dual_path_tdecq_surface.jsonl"
)


def _space() -> IntegerParameterSpace:
    return IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="eq",
                unit="code",
                lower=0,
                upper=2,
                grid_values=(0, 1, 2),
                max_step=1,
            ),
            ParameterSpec(
                name="ibias_dac",
                unit="code",
                lower=0,
                upper=2,
                grid_values=(0, 1, 2),
                max_step=1,
            ),
        )
    )


def _restricted_space() -> IntegerParameterSpace:
    return IntegerParameterSpace(
        parameters=_space().parameters,
        allowed=lambda point: point["eq"] + point["ibias_dac"] <= 1,
    )


def _fixture_observations() -> tuple[TuningObservation, ...]:
    return tuple(
        TuningObservation.model_validate_json(line, strict=True)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    )


def _all_invalid_observations() -> tuple[TuningObservation, ...]:
    return tuple(
        observation.model_copy(
            update={
                "tdecq_db": None,
                "tdecq_status": "INV",
                "tdecq_invalid_reason": "synthetic all-invalid replay surface",
            }
        )
        for observation in _fixture_observations()
    )


def _observations_for_space(
    space: IntegerParameterSpace,
) -> tuple[TuningObservation, ...]:
    allowed_keys = {
        tuple(point[name] for name in space.names)
        for point in space.enumerate_points()
    }
    return tuple(
        observation
        for observation in _fixture_observations()
        if tuple(
            observation.settings_readback[name] for name in space.names
        )
        in allowed_keys
    )


def _empty_candidate_pool(space: IntegerParameterSpace) -> None:
    space._endpoint_keys = ()
    space._endpoint_key_set = frozenset()


def _suggestion(
    point: dict[str, int],
    *,
    candidate_id: str = "current-candidate",
    path: TuningPath = TuningPath.TDECQ_BAYESIAN,
) -> CandidateSuggestion:
    return CandidateSuggestion(
        path=path,
        candidate_id=candidate_id,
        requested=point,
        reason="test replay lookup",
    )


def _report_payload(**updates):
    payload = {
        "path": TuningPath.EXHAUSTIVE,
        "status": "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
        "stop_reason": "POOL_EXHAUSTED",
        "seed": 1124,
        "budget": 9,
        "measurement_count": 9,
        "candidate_count": 9,
        "invalid_count": 1,
        "fallback_count": 0,
        "best_observation_id": "replay-00000007",
        "best_tdecq_db": 2.1,
        "grid_best_tdecq_db": 2.1,
        "regret_db": 0.0,
        "reached_tdecq_below_3": True,
    }
    payload.update(updates)
    return payload


def test_fixture_contains_exactly_nine_nonblank_jsonl_records():
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 9
    assert all(line and line == line.strip() for line in lines)
    assert tuple(item.observation_id for item in _fixture_observations()) == (
        "surface-00",
        "surface-01",
        "surface-02",
        "surface-10",
        "surface-11",
        "surface-12",
        "surface-20",
        "surface-21",
        "surface-22",
    )


def test_exhaustive_replay_returns_known_grid_optimum():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    report = replay(
        surface,
        path=TuningPath.EXHAUSTIVE,
        seed=1124,
        budget=9,
    )

    assert report.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert report.grid_best_tdecq_db == pytest.approx(2.1)
    assert report.best_tdecq_db == pytest.approx(2.1)
    assert report.measurement_count == 9
    assert report.candidate_count == 9
    assert report.invalid_count == 1
    assert report.fallback_count == 0
    assert report.regret_db == pytest.approx(0.0)
    assert report.reached_tdecq_below_3 is True
    assert report.stop_reason == "POOL_EXHAUSTED"


def test_bayesian_replay_is_seeded_and_reports_exact_regret():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    first = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )
    second = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )

    assert second == first
    assert 0 < first.measurement_count <= 7
    assert first.candidate_count == 9
    assert first.fallback_count > 0
    assert first.grid_best_tdecq_db == pytest.approx(2.1)
    assert first.best_tdecq_db is not None
    assert first.regret_db == pytest.approx(first.best_tdecq_db - 2.1)
    assert first.regret_db >= 0.0
    assert first.status == "BEST_TDECQ_FOUND_WITHIN_BUDGET"
    assert first.stop_reason == "BUDGET_EXHAUSTED"


def test_replay_result_is_invariant_to_er_only_mutation():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())
    mutated = surface.with_er_values(
        {
            key: 1000.0 - index * 100.0
            for index, key in enumerate(surface.keys)
        }
    )

    original_report = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )
    mutated_report = replay(
        mutated,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )

    assert mutated_report == original_report


def test_bayesian_full_pool_at_budget_boundary_is_not_reported_as_budget_limited():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    report = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=surface.space.candidate_count,
    )

    assert report.measurement_count == 9
    assert report.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert report.stop_reason == "POOL_EXHAUSTED"
    assert report.best_tdecq_db == pytest.approx(2.1)
    assert report.regret_db == pytest.approx(0.0)


def test_exhaustive_budget_before_full_pool_is_truthfully_incomplete():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    report = replay(
        surface,
        path=TuningPath.EXHAUSTIVE,
        seed=1124,
        budget=3,
    )

    assert report.measurement_count == 3
    assert report.status == "EXHAUSTIVE_INCOMPLETE"
    assert report.stop_reason == "BUDGET_EXHAUSTED"


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(TuningPath.EXHAUSTIVE, id="exhaustive"),
        pytest.param(TuningPath.TDECQ_BAYESIAN, id="bayesian"),
    ],
)
def test_all_invalid_full_pool_reports_no_valid_tdecq(path):
    surface = ReplaySurface(_space(), _all_invalid_observations())

    report = replay(
        surface,
        path=path,
        seed=1124,
        budget=surface.space.candidate_count,
    )

    assert surface.grid_best_tdecq_db is None
    assert report.status == "NO_VALID_TDECQ"
    assert report.stop_reason == "POOL_EXHAUSTED"
    assert report.measurement_count == 9
    assert report.candidate_count == 9
    assert report.invalid_count == 9
    assert report.best_observation_id is None
    assert report.best_tdecq_db is None
    assert report.grid_best_tdecq_db is None
    assert report.regret_db is None
    assert report.reached_tdecq_below_3 is False


def test_incomplete_bayesian_replay_can_find_no_valid_measurement_on_valid_grid():
    observations = list(_all_invalid_observations())
    observations[-1] = observations[-1].model_copy(
        update={
            "tdecq_db": 2.2,
            "tdecq_status": "CORR",
            "tdecq_invalid_reason": None,
        }
    )
    surface = ReplaySurface(_space(), tuple(observations))

    report = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=1,
    )

    assert report.status == "NO_VALID_TDECQ"
    assert report.stop_reason == "BUDGET_EXHAUSTED"
    assert report.measurement_count == 1
    assert report.invalid_count == 1
    assert report.best_observation_id is None
    assert report.best_tdecq_db is None
    assert report.grid_best_tdecq_db == pytest.approx(2.2)
    assert report.regret_db is None
    assert report.reached_tdecq_below_3 is False


def test_surface_coverage_is_keyed_by_actual_readback_not_requested_settings():
    observations = _fixture_observations()
    repeated_request = observations[-1].model_copy(
        update={"settings_requested": {"eq": 2, "ibias_dac": 1}}
    )

    surface = ReplaySurface(_space(), (*observations[:-1], repeated_request))
    observed = surface.observe(
        _suggestion({"eq": 2, "ibias_dac": 2}),
        sequence=0,
    )

    assert surface.keys == tuple((eq, ibias) for eq in range(3) for ibias in range(3))
    assert observed.tdecq_db == pytest.approx(2.3)
    assert observed.settings_readback == {"eq": 2, "ibias_dac": 2}


def test_surface_snapshots_original_restricted_space_before_caller_mutation():
    original = _restricted_space()
    fingerprint = original.fingerprint
    surface = ReplaySurface(original, _observations_for_space(original))

    _empty_candidate_pool(original)

    assert original.candidate_count == 0
    assert surface.keys == ((0, 0), (0, 1), (1, 0))
    assert surface.space is not original
    assert surface.space.candidate_count == 3
    assert surface.space.fingerprint == fingerprint
    report = replay(
        surface,
        path=TuningPath.EXHAUSTIVE,
        seed=1124,
        budget=3,
    )
    assert report.candidate_count == 3
    assert report.measurement_count == 3
    assert report.best_tdecq_db == pytest.approx(3.0)
    assert report.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"


def test_surface_returns_detached_restricted_spaces_that_cannot_mutate_oracle():
    original = _restricted_space()
    surface = ReplaySurface(original, _observations_for_space(original))
    exposed = surface.space

    _empty_candidate_pool(exposed)

    assert exposed.candidate_count == 0
    assert surface.space is not exposed
    assert surface.space.candidate_count == 3
    assert surface.keys == ((0, 0), (0, 1), (1, 0))
    report = replay(
        surface,
        path=TuningPath.EXHAUSTIVE,
        seed=1124,
        budget=3,
    )
    assert report.candidate_count == 3
    assert report.measurement_count == 3
    assert report.stop_reason == "POOL_EXHAUSTED"


def test_surface_rejects_missing_duplicate_and_foreign_context_readbacks():
    observations = _fixture_observations()

    with pytest.raises(ValueError, match="cover every safe endpoint"):
        ReplaySurface(_space(), observations[:-1])

    duplicate = observations[-1].model_copy(
        update={"settings_readback": {"eq": 2, "ibias_dac": 1}}
    )
    with pytest.raises(ValueError, match="duplicate|repeated.*endpoint"):
        ReplaySurface(_space(), (*observations[:-1], duplicate))

    foreign = observations[-1].model_copy(
        update={
            "context": observations[-1].context.model_copy(
                update={"device_id": "foreign-vcsel"}
            )
        }
    )
    with pytest.raises(ValueError, match="context"):
        ReplaySurface(_space(), (*observations[:-1], foreign))


def test_from_jsonl_reports_file_and_record_line_failures(tmp_path: Path):
    missing = tmp_path / "missing-surface.jsonl"
    with pytest.raises(ValueError, match="cannot read replay surface") as missing_error:
        ReplaySurface.from_jsonl(missing, _space())
    assert str(missing) in str(missing_error.value)

    malformed = tmp_path / "malformed-surface.jsonl"
    malformed.write_text(
        FIXTURE.read_text(encoding="utf-8").splitlines()[0]
        + "\n{not-json}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"line 2") as malformed_error:
        ReplaySurface.from_jsonl(malformed, _space())
    assert str(malformed) in str(malformed_error.value)

    empty = tmp_path / "empty-surface.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no observations") as empty_error:
        ReplaySurface.from_jsonl(empty, _space())
    assert str(empty) in str(empty_error.value)


def test_surface_copies_input_and_observe_returns_a_new_validated_linked_record():
    observations = _fixture_observations()
    external_readback = {"eq": 0, "ibias_dac": 0}
    externally_mutable = observations[0].model_copy(
        update={"settings_readback": external_readback}
    )
    surface = ReplaySurface(_space(), (externally_mutable, *observations[1:]))
    external_readback["eq"] = 2
    suggestion = _suggestion(
        {"eq": 0, "ibias_dac": 0},
        candidate_id="bayesian-current-0001",
    )

    observed = surface.observe(suggestion, sequence=4)

    assert observed is not externally_mutable
    assert observed.observation_id == "replay-00000004"
    assert observed.candidate_id == suggestion.candidate_id
    assert observed.settings_requested == suggestion.requested
    assert observed.settings_readback == {"eq": 0, "ibias_dac": 0}
    assert TuningObservation.model_validate(observed.model_dump(), strict=True) == observed
    with pytest.raises(TypeError):
        observed.settings_readback["eq"] = 1
    with pytest.raises(ValidationError):
        observed.candidate_id = "changed"


@pytest.mark.parametrize("sequence", [True, -1, 1.0])
def test_observe_requires_a_nonnegative_strict_integer_sequence(sequence):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    with pytest.raises(ValueError, match="sequence.*nonnegative integer"):
        surface.observe(_suggestion({"eq": 0, "ibias_dac": 0}), sequence)


def test_with_er_values_is_defensive_and_preserves_observation_validity():
    observations = _fixture_observations()
    invalid_er = observations[0].model_copy(
        update={"er_db": None, "er_valid": False, "er_status": "INV"}
    )
    surface = ReplaySurface(_space(), (invalid_er, *observations[1:]))
    replacements = {
        key: 100.0 + index for index, key in enumerate(surface.keys)
    }

    mutated = surface.with_er_values(replacements)
    replacements[(0, 0)] = -999.0
    original = surface.observe(_suggestion({"eq": 0, "ibias_dac": 0}), 0)
    changed = mutated.observe(_suggestion({"eq": 0, "ibias_dac": 0}), 0)

    assert original.er_db is None
    assert original.er_valid is False
    assert changed.er_db == pytest.approx(100.0)
    assert changed.er_valid is True
    assert changed.er_status == "CORR"
    assert changed.model_dump(
        exclude={"er_db", "er_valid", "er_status"}
    ) == original.model_dump(
        exclude={"er_db", "er_valid", "er_status"}
    )
    assert TuningObservation.model_validate(changed.model_dump(), strict=True) == changed


def test_with_er_values_requires_exact_safe_keys_and_finite_numeric_values():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())
    complete = {key: 4.0 for key in surface.keys}

    missing = dict(complete)
    missing.pop((2, 2))
    with pytest.raises(ValueError, match="exactly every safe endpoint"):
        surface.with_er_values(missing)

    extra = dict(complete)
    extra[(9, 9)] = 4.0
    with pytest.raises(ValueError, match="exactly every safe endpoint"):
        surface.with_er_values(extra)

    for invalid in (True, "4.0", float("inf"), float("nan")):
        invalid_values = dict(complete)
        invalid_values[(0, 0)] = invalid
        with pytest.raises(ValueError, match="finite numeric"):
            surface.with_er_values(invalid_values)


@pytest.mark.parametrize(
    ("path", "seed", "budget", "message"),
    [
        pytest.param("PATH_EXHAUSTIVE", 1124, 1, "TuningPath", id="string-path"),
        pytest.param(None, 1124, 1, "TuningPath", id="none-path"),
        pytest.param(TuningPath.EXHAUSTIVE, True, 1, "seed.*integer", id="bool-seed"),
        pytest.param(TuningPath.EXHAUSTIVE, 1.0, 1, "seed.*integer", id="float-seed"),
        pytest.param(
            TuningPath.EXHAUSTIVE,
            1124,
            True,
            "budget.*positive integer",
            id="bool-budget",
        ),
        pytest.param(
            TuningPath.EXHAUSTIVE,
            1124,
            0,
            "budget.*positive integer",
            id="zero-budget",
        ),
        pytest.param(
            TuningPath.EXHAUSTIVE,
            1124,
            -1,
            "budget.*positive integer",
            id="negative-budget",
        ),
        pytest.param(
            TuningPath.EXHAUSTIVE,
            1124,
            1.0,
            "budget.*positive integer",
            id="float-budget",
        ),
    ],
)
def test_replay_rejects_non_enum_path_and_non_strict_integer_controls(
    path,
    seed,
    budget,
    message,
):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    with pytest.raises(ValueError, match=message):
        replay(surface, path=path, seed=seed, budget=budget)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_replay_accepts_sklearn_seed_boundaries(seed):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    report = replay(
        surface,
        path=TuningPath.EXHAUSTIVE,
        seed=seed,
        budget=1,
    )

    assert report.seed == seed
    assert ReplayReport(**_report_payload(seed=seed)).seed == seed


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_replay_and_report_reject_seeds_outside_sklearn_range(seed):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    with pytest.raises(ValueError, match="seed.*0|seed.*range|seed.*32"):
        replay(
            surface,
            path=TuningPath.EXHAUSTIVE,
            seed=seed,
            budget=1,
        )
    with pytest.raises(ValidationError, match="seed|greater|less|range"):
        ReplayReport(**_report_payload(seed=seed))


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param(
            {"best_tdecq_db": 2.4, "regret_db": 0.2},
            id="regret-not-best-minus-grid",
        ),
        pytest.param(
            {"best_tdecq_db": 2.4, "regret_db": 0.3},
            id="grid-optimum-status-with-nonzero-regret",
        ),
        pytest.param(
            {
                "best_tdecq_db": 2.1,
                "grid_best_tdecq_db": 2.2,
                "regret_db": 0.0,
            },
            id="grid-best-worse-than-measured-best",
        ),
        pytest.param(
            {"status": "NO_VALID_TDECQ"},
            id="no-valid-with-best",
        ),
        pytest.param(
            {"fallback_count": 1},
            id="exhaustive-with-fallback",
        ),
        pytest.param(
            {"stop_reason": "NO_SAFE_CANDIDATE"},
            id="full-pool-with-non-pool-stop",
        ),
        pytest.param(
            {
                "path": TuningPath.TDECQ_BAYESIAN,
                "status": "NO_VALID_TDECQ",
                "invalid_count": 9,
                "best_observation_id": None,
                "best_tdecq_db": None,
                "regret_db": None,
                "reached_tdecq_below_3": False,
            },
            id="full-pool-numeric-grid-best-without-measured-best",
        ),
        pytest.param(
            {
                "best_tdecq_db": 2.1,
                "grid_best_tdecq_db": None,
                "regret_db": 0.0,
            },
            id="measured-best-without-grid-best",
        ),
        pytest.param(
            {
                "path": TuningPath.TDECQ_BAYESIAN,
                "status": "BEST_TDECQ_FOUND_WITHIN_BUDGET",
                "best_observation_id": None,
                "best_tdecq_db": None,
                "regret_db": None,
                "reached_tdecq_below_3": False,
            },
            id="best-status-without-best",
        ),
        pytest.param(
            {"status": "EXHAUSTIVE_INCOMPLETE"},
            id="incomplete-status-with-full-pool",
        ),
        pytest.param(
            {
                "path": TuningPath.TDECQ_BAYESIAN,
                "status": "BEST_TDECQ_FOUND_WITHIN_BUDGET",
            },
            id="budget-best-status-with-full-pool",
        ),
    ],
)
def test_replay_report_rejects_internally_untruthful_results(updates):
    with pytest.raises(ValidationError):
        ReplayReport(**_report_payload(**updates))


def test_replay_report_allows_truthful_partial_no_valid_result():
    report = ReplayReport(
        **_report_payload(
            path=TuningPath.TDECQ_BAYESIAN,
            status="NO_VALID_TDECQ",
            stop_reason="BUDGET_EXHAUSTED",
            budget=3,
            measurement_count=3,
            invalid_count=3,
            best_observation_id=None,
            best_tdecq_db=None,
            regret_db=None,
            reached_tdecq_below_3=False,
        )
    )

    assert report.status == "NO_VALID_TDECQ"


def test_replay_report_allows_truthful_full_pool_all_invalid_result():
    report = ReplayReport(
        **_report_payload(
            path=TuningPath.TDECQ_BAYESIAN,
            status="NO_VALID_TDECQ",
            invalid_count=9,
            best_observation_id=None,
            best_tdecq_db=None,
            grid_best_tdecq_db=None,
            regret_db=None,
            reached_tdecq_below_3=False,
        )
    )

    assert report.status == "NO_VALID_TDECQ"
    assert report.grid_best_tdecq_db is None


def test_replay_handles_no_safe_candidate_without_claiming_budget_exhaustion(
    monkeypatch,
):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    class NoCandidateOptimizer:
        def __init__(self, space, context):
            self.space = space

        def suggest(self, observations, pending_keys=frozenset()):
            assert observations == ()
            assert pending_keys == frozenset()
            return NoSafeCandidate(
                path=TuningPath.EXHAUSTIVE,
                reason="synthetic no-candidate terminal",
            )

        def finish(self, observations, budget_exhausted):
            assert budget_exhausted is True
            return PathComplete(
                path=TuningPath.EXHAUSTIVE,
                status="EXHAUSTIVE_INCOMPLETE",
                measured_count=0,
                candidate_count=self.space.candidate_count,
            )

    monkeypatch.setattr(replay_module, "ExhaustiveOptimizer", NoCandidateOptimizer)

    report = replay(surface, path=TuningPath.EXHAUSTIVE, seed=1124, budget=9)

    assert report.status == "EXHAUSTIVE_INCOMPLETE"
    assert report.stop_reason == "NO_SAFE_CANDIDATE"
    assert report.measurement_count == 0
    assert report.regret_db is None


def test_replay_rejects_foreign_path_fallback_before_counting(monkeypatch):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())
    bayesian_module = import_module("vcsel_ai_tuning.optimization.bayesian")

    class ForeignFallbackOptimizer:
        def __init__(self, space, context, seed):
            pass

        def suggest(self, observations, pending_keys=frozenset()):
            return FallbackRequested(
                path=TuningPath.EXHAUSTIVE,
                reason="synthetic foreign-path fallback",
            )

    monkeypatch.setattr(
        bayesian_module,
        "TdecqBayesianOptimizer",
        ForeignFallbackOptimizer,
    )

    with pytest.raises(ValueError, match="fallback.*path|does not match"):
        replay(
            surface,
            path=TuningPath.TDECQ_BAYESIAN,
            seed=1124,
            budget=1,
        )


def test_replay_handles_an_early_path_complete_without_claiming_pool_exhaustion(
    monkeypatch,
):
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    class EarlyCompleteOptimizer:
        def __init__(self, space, context):
            self.space = space

        def suggest(self, observations, pending_keys=frozenset()):
            assert pending_keys == frozenset()
            return PathComplete(
                path=TuningPath.EXHAUSTIVE,
                status="NO_VALID_TDECQ",
                measured_count=0,
                candidate_count=self.space.candidate_count,
            )

    monkeypatch.setattr(replay_module, "ExhaustiveOptimizer", EarlyCompleteOptimizer)

    report = replay(surface, path=TuningPath.EXHAUSTIVE, seed=1124, budget=9)

    assert report.status == "NO_VALID_TDECQ"
    assert report.stop_reason == "PATH_COMPLETE"
    assert report.measurement_count == 0
