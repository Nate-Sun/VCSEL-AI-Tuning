import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    OptimizationContext,
    PathComplete,
    TuningObservation,
    TuningPath,
)


def _context(**updates) -> OptimizationContext:
    values = {
        "driver_profile_hash": "a" * 64,
        "vcsel_model": "APA4701",
        "device_id": "vcsel-001",
        "test_context_id": "53g-ssprq-25c",
    }
    values.update(updates)
    return OptimizationContext(**values)


def _observation(**updates) -> TuningObservation:
    values = {
        "observation_id": "obs-1",
        "candidate_id": "candidate-1",
        "context": _context(),
        "settings_requested": {"ibias_dac": 70, "eq": 22},
        "settings_readback": {"ibias_dac": 72, "eq": 21},
        "tdecq_db": 2.4,
        "tdecq_status": "CORR",
        "er_db": 3.2,
        "er_valid": True,
        "er_status": "CORR",
    }
    values.update(updates)
    return TuningObservation(**values)


def _suggestion(**updates) -> CandidateSuggestion:
    values = {
        "path": TuningPath.TDECQ_BAYESIAN,
        "candidate_id": "candidate-1",
        "requested": {"ibias_dac": 70, "eq": 22},
        "reason": "highest finite-pool acquisition",
    }
    values.update(updates)
    return CandidateSuggestion(**values)


def test_protocol_results_round_trip_as_json():
    context = _context()
    observation = _observation()
    suggestion = _suggestion(candidate_id="candidate-0001")
    complete = PathComplete(
        path=TuningPath.EXHAUSTIVE,
        status="GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
        measured_count=4,
        candidate_count=4,
        best_observation_id="obs-3",
        best_tdecq_db=2.1,
    )

    assert OptimizationContext.model_validate_json(context.model_dump_json()) == context
    assert TuningObservation.model_validate_json(observation.model_dump_json()) == observation
    assert CandidateSuggestion.model_validate_json(suggestion.model_dump_json()) == suggestion
    assert PathComplete.model_validate_json(complete.model_dump_json()) == complete


@pytest.mark.parametrize(
    ("model", "field"),
    [
        pytest.param(_observation(), "settings_requested", id="requested-audit"),
        pytest.param(_observation(), "settings_readback", id="readback-model-input"),
        pytest.param(_suggestion(), "requested", id="candidate-request"),
    ],
)
def test_protocol_settings_maps_reject_direct_mutation(model, field):
    settings = getattr(model, field)

    with pytest.raises(TypeError):
        settings["eq"] = 99


def test_protocol_settings_maps_keep_mapping_behavior_and_json_shape():
    observation = _observation()
    suggestion = _suggestion()

    assert observation.settings_readback == {"ibias_dac": 72, "eq": 21}
    assert tuple(observation.settings_readback[name] for name in ("eq", "ibias_dac")) == (
        21,
        72,
    )
    assert suggestion.model_dump(mode="json")["requested"] == {
        "ibias_dac": 70,
        "eq": 22,
    }


def test_context_compatibility_excludes_physical_device_identity():
    first = _context(device_id="vcsel-001")
    second = _context(device_id="vcsel-999")

    assert first.compatibility_key == second.compatibility_key


@pytest.mark.parametrize(
    "driver_profile_hash",
    [
        pytest.param("a" * 63, id="too-short"),
        pytest.param("A" * 64, id="uppercase"),
        pytest.param("g" * 64, id="not-hex"),
    ],
)
def test_context_requires_exact_lowercase_sha256(driver_profile_hash):
    with pytest.raises(ValidationError):
        _context(driver_profile_hash=driver_profile_hash)


@pytest.mark.parametrize("field", ["vcsel_model", "device_id", "test_context_id"])
@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_context_identity_fields_must_be_nonblank(field, value):
    with pytest.raises(ValidationError):
        _context(**{field: value})


def test_context_identity_fields_are_trimmed():
    context = _context(
        vcsel_model="  APA4701 ",
        device_id=" vcsel-001\t",
        test_context_id=" 53g-ssprq-25c ",
    )

    assert context.vcsel_model == "APA4701"
    assert context.device_id == "vcsel-001"
    assert context.test_context_id == "53g-ssprq-25c"


def test_tdecq_status_is_trimmed_and_canonicalized_to_uppercase():
    valid = _observation(tdecq_status="  corr\t")
    invalid = _observation(
        tdecq_db=None,
        tdecq_status=" inv ",
        tdecq_invalid_reason="instrument timeout",
    )

    assert valid.tdecq_status == "CORR"
    assert invalid.tdecq_status == "INV"


def test_numeric_tdecq_rejects_explicit_non_corr_status():
    with pytest.raises(ValidationError, match="non-CORR TDECQ"):
        _observation(tdecq_status=" inv ")


@pytest.mark.parametrize("status", ["", "   ", "\t"])
def test_tdecq_status_must_be_nonblank(status):
    with pytest.raises(ValidationError):
        _observation(
            tdecq_db=None,
            tdecq_status=status,
            tdecq_invalid_reason="instrument timeout",
        )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: _observation(observation_id="   "),
            id="observation-id",
        ),
        pytest.param(
            lambda: _observation(candidate_id="\t"),
            id="observation-candidate-id",
        ),
        pytest.param(
            lambda: _suggestion(candidate_id=" "),
            id="suggestion-candidate-id",
        ),
        pytest.param(
            lambda: _suggestion(reason="   "),
            id="suggestion-reason",
        ),
        pytest.param(
            lambda: FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=" ",
            ),
            id="fallback-reason",
        ),
        pytest.param(
            lambda: NoSafeCandidate(
                path=TuningPath.EXHAUSTIVE,
                reason="\t",
            ),
            id="no-safe-candidate-reason",
        ),
    ],
)
def test_protocol_identifiers_and_explanations_must_be_nonblank(factory):
    with pytest.raises(ValidationError):
        factory()


def test_protocol_identifiers_and_explanations_are_trimmed():
    observation = _observation(observation_id=" obs-1 ", candidate_id=" candidate-1 ")
    suggestion = _suggestion(candidate_id=" candidate-2 ", reason=" explore ")

    assert observation.observation_id == "obs-1"
    assert observation.candidate_id == "candidate-1"
    assert suggestion.candidate_id == "candidate-2"
    assert suggestion.reason == "explore"


def test_corr_tdecq_requires_numeric_value():
    with pytest.raises(ValidationError, match="CORR TDECQ requires a numeric value"):
        _observation(tdecq_db=None)


def test_corr_tdecq_rejects_invalid_reason():
    with pytest.raises(ValidationError, match="cannot have an invalid reason"):
        _observation(tdecq_invalid_reason="instrument timeout")


def test_non_corr_tdecq_keeps_value_none():
    with pytest.raises(ValidationError, match="non-CORR TDECQ must keep value=None"):
        _observation(tdecq_status="INV", tdecq_invalid_reason="timeout")


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_non_corr_tdecq_requires_nonblank_invalid_reason(reason):
    with pytest.raises(ValidationError, match="requires an invalid reason"):
        _observation(
            tdecq_db=None,
            tdecq_status="INV",
            tdecq_invalid_reason=reason,
        )


@pytest.mark.parametrize(
    ("er_db", "er_valid"),
    [
        pytest.param(None, True, id="valid-without-value"),
        pytest.param(3.2, False, id="value-marked-invalid"),
    ],
)
def test_er_validity_exactly_matches_numeric_value_presence(er_db, er_valid):
    with pytest.raises(ValidationError, match="ER validity"):
        _observation(er_db=er_db, er_valid=er_valid)


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        pytest.param(_observation, "tdecq_db", float("inf"), id="observation-inf"),
        pytest.param(
            lambda **kwargs: CandidateSuggestion(
                path=TuningPath.TDECQ_BAYESIAN,
                candidate_id="candidate-1",
                requested={"eq": 1},
                reason="test",
                **kwargs,
            ),
            "acquisition",
            float("nan"),
            id="suggestion-nan",
        ),
        pytest.param(
            lambda **kwargs: PathComplete(
                path=TuningPath.EXHAUSTIVE,
                status="NO_VALID_TDECQ",
                measured_count=0,
                candidate_count=1,
                **kwargs,
            ),
            "best_tdecq_db",
            float("-inf"),
            id="completion-inf",
        ),
    ],
)
def test_protocol_numeric_models_reject_nonfinite_values(factory, field, value):
    with pytest.raises(ValidationError):
        factory(**{field: value})


@pytest.mark.parametrize(
    "status",
    ["GLOBAL_OPTIMUM", "BEST_TDECQ_FOUND_SO_FAR", "COMPLETE"],
)
def test_path_complete_accepts_only_defined_statuses(status):
    with pytest.raises(ValidationError):
        PathComplete(
            path=TuningPath.EXHAUSTIVE,
            status=status,
            measured_count=1,
            candidate_count=1,
        )


@pytest.mark.parametrize("field", ["measured_count", "candidate_count"])
def test_path_complete_counts_cannot_be_negative(field):
    values = {"measured_count": 1, "candidate_count": 1, field: -1}

    with pytest.raises(ValidationError):
        PathComplete(
            path=TuningPath.EXHAUSTIVE,
            status="EXHAUSTIVE_INCOMPLETE",
            **values,
        )


def _path_complete(**updates) -> PathComplete:
    values = {
        "path": TuningPath.EXHAUSTIVE,
        "status": "EXHAUSTIVE_INCOMPLETE",
        "measured_count": 1,
        "candidate_count": 2,
    }
    values.update(updates)
    return PathComplete(**values)


def test_path_complete_measured_count_cannot_exceed_candidate_count():
    with pytest.raises(ValidationError, match="measured_count.*candidate_count"):
        _path_complete(measured_count=3, candidate_count=2)


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param(
            {"best_observation_id": "obs-1"},
            id="identifier-without-value",
        ),
        pytest.param(
            {"best_tdecq_db": 2.1},
            id="value-without-identifier",
        ),
    ],
)
def test_path_complete_best_identifier_and_value_are_an_atomic_pair(updates):
    with pytest.raises(ValidationError, match="best observation.*pair"):
        _path_complete(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param(
            {"measured_count": 1, "candidate_count": 2},
            id="partial-enumeration",
        ),
        pytest.param(
            {"best_observation_id": None, "best_tdecq_db": None},
            id="no-valid-best",
        ),
    ],
)
def test_global_grid_optimum_requires_complete_enumeration_and_best(updates):
    values = {
        "status": "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
        "measured_count": 2,
        "candidate_count": 2,
        "best_observation_id": "obs-1",
        "best_tdecq_db": 2.1,
    }
    values.update(updates)

    with pytest.raises(ValidationError, match="GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"):
        _path_complete(**values)


def test_budget_best_requires_best_pair():
    with pytest.raises(ValidationError, match="BEST_TDECQ_FOUND_WITHIN_BUDGET"):
        _path_complete(
            path=TuningPath.TDECQ_BAYESIAN,
            status="BEST_TDECQ_FOUND_WITHIN_BUDGET",
        )


def test_no_valid_tdecq_forbids_best_pair():
    with pytest.raises(ValidationError, match="NO_VALID_TDECQ"):
        _path_complete(
            status="NO_VALID_TDECQ",
            best_observation_id="obs-1",
            best_tdecq_db=2.1,
        )


def test_exhaustive_incomplete_cannot_claim_full_enumeration():
    with pytest.raises(ValidationError, match="EXHAUSTIVE_INCOMPLETE"):
        _path_complete(measured_count=2, candidate_count=2)


def test_exhaustive_incomplete_may_preserve_valid_best_pair():
    complete = _path_complete(
        best_observation_id="obs-1",
        best_tdecq_db=2.1,
    )

    assert complete.best_observation_id == "obs-1"
    assert complete.best_tdecq_db == 2.1


@pytest.mark.parametrize(
    ("path", "status", "best"),
    [
        pytest.param(
            TuningPath.TDECQ_BAYESIAN,
            "EXHAUSTIVE_INCOMPLETE",
            {},
            id="exhaustive-status-on-bayesian-path",
        ),
        pytest.param(
            TuningPath.EXHAUSTIVE,
            "BEST_TDECQ_FOUND_WITHIN_BUDGET",
            {"best_observation_id": "obs-1", "best_tdecq_db": 2.1},
            id="bayesian-status-on-exhaustive-path",
        ),
    ],
)
def test_path_specific_completion_statuses_require_matching_path(path, status, best):
    with pytest.raises(ValidationError, match=status):
        _path_complete(path=path, status=status, **best)


def test_path_complete_best_pair_requires_at_least_one_measurement():
    with pytest.raises(ValidationError, match="best observation.*measured"):
        _path_complete(
            measured_count=0,
            best_observation_id="obs-1",
            best_tdecq_db=2.1,
        )


def test_path_complete_best_identifier_must_be_nonblank():
    with pytest.raises(ValidationError):
        _path_complete(best_observation_id="   ", best_tdecq_db=2.1)
