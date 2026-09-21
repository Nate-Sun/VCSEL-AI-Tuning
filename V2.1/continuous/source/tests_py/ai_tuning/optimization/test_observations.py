from pathlib import Path

import pytest

from vcsel_ai_tuning.domain import MetricValue, RunState
from vcsel_ai_tuning.optimization.observations import normalize_observation


def test_normalization_uses_readback_not_requested_or_applied(
    domain_observation,
    context,
):
    normalized = normalize_observation(domain_observation, context)

    assert normalized.settings_readback == {"ibias_dac": 72, "eq": 21}
    assert normalized.settings_requested == {"ibias_dac": 70, "eq": 22}
    assert normalized.settings_readback != {"ibias_dac": 71, "eq": 20}
    assert normalized.tdecq_db == 2.4
    assert normalized.er_db == 3.2


def test_invalid_tdecq_stays_none_while_er_survives(
    invalid_tdecq_observation,
    context,
):
    normalized = normalize_observation(invalid_tdecq_observation, context)

    assert normalized.tdecq_db is None
    assert normalized.tdecq_status == "INV"
    assert normalized.tdecq_invalid_reason == "timeout waiting for CORR"
    assert normalized.er_db == 3.4
    assert normalized.er_valid is True
    assert normalized.er_status == "CORR"


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            {
                "state": RunState.ABORTED,
                "valid": False,
                "failure_reason": "write/readback mismatch",
            },
            id="application-failure",
        ),
        pytest.param(
            {"valid": False, "failure_reason": "measurement invalid"},
            id="invalid-complete-observation",
        ),
    ],
)
def test_failed_or_invalid_run_is_not_normalized_as_a_measurement(
    domain_observation,
    context,
    update,
):
    failed = domain_observation.model_copy(update=update)

    with pytest.raises(ValueError, match="completed valid measurement"):
        normalize_observation(failed, context)


@pytest.mark.parametrize("setting_group", ["requested", "readback"])
def test_fractional_optimizer_setting_is_rejected(
    domain_observation,
    context,
    setting_group,
):
    settings = domain_observation.settings.model_copy(
        update={setting_group: {"ibias_dac": 70.5, "eq": 22.0}}
    )
    observation = domain_observation.model_copy(update={"settings": settings})

    with pytest.raises(ValueError, match="must be integer-valued"):
        normalize_observation(observation, context)


def test_missing_tdecq_gets_explicit_missing_state(domain_observation, context):
    metrics = dict(domain_observation.metrics)
    metrics.pop("tdecq_db")
    observation = domain_observation.model_copy(update={"metrics": metrics})

    normalized = normalize_observation(observation, context)

    assert normalized.tdecq_db is None
    assert normalized.tdecq_status == "MISSING"
    assert normalized.tdecq_invalid_reason == "tdecq_db missing"
    assert normalized.er_db == 3.2
    assert normalized.er_valid is True


def test_legacy_valid_tdecq_without_status_defaults_to_corr(
    domain_observation,
    context,
):
    metrics = dict(domain_observation.metrics)
    metrics["tdecq_db"] = metrics["tdecq_db"].model_copy(update={"status": None})
    observation = domain_observation.model_copy(update={"metrics": metrics})

    normalized = normalize_observation(observation, context)

    assert normalized.tdecq_db == 2.4
    assert normalized.tdecq_status == "CORR"
    assert normalized.tdecq_invalid_reason is None


def test_valid_tdecq_status_is_trimmed_and_canonicalized(
    domain_observation,
    context,
):
    metrics = dict(domain_observation.metrics)
    metrics["tdecq_db"] = metrics["tdecq_db"].model_copy(
        update={"status": "  corr\t"}
    )
    observation = domain_observation.model_copy(update={"metrics": metrics})

    normalized = normalize_observation(observation, context)

    assert normalized.tdecq_db == 2.4
    assert normalized.tdecq_status == "CORR"


def test_valid_numeric_tdecq_rejects_explicit_non_corr_status(
    domain_observation,
    context,
):
    metrics = dict(domain_observation.metrics)
    metrics["tdecq_db"] = metrics["tdecq_db"].model_copy(
        update={"status": " inv "}
    )
    observation = domain_observation.model_copy(update={"metrics": metrics})

    with pytest.raises(ValueError, match="valid numeric TDECQ.*CORR"):
        normalize_observation(observation, context)


def test_invalid_tdecq_status_is_canonicalized_without_losing_reason(
    invalid_tdecq_observation,
    context,
):
    metrics = dict(invalid_tdecq_observation.metrics)
    metrics["tdecq_db"] = metrics["tdecq_db"].model_copy(
        update={"status": " inv ", "invalid_reason": "instrument said INV"}
    )
    observation = invalid_tdecq_observation.model_copy(update={"metrics": metrics})

    normalized = normalize_observation(observation, context)

    assert normalized.tdecq_status == "INV"
    assert normalized.tdecq_invalid_reason == "instrument said INV"


def test_artifacts_are_serialized_as_posix_hash_references(
    domain_observation,
    context,
):
    artifact = domain_observation.artifacts[0].model_copy(
        update={"relative_path": Path("captures") / "eye" / "trace.json"}
    )
    observation = domain_observation.model_copy(update={"artifacts": [artifact]})

    normalized = normalize_observation(observation, context)

    assert normalized.artifact_refs == (
        f"captures/eye/trace.json#sha256={'b' * 64}",
    )


def test_invalid_er_is_kept_independent_of_valid_tdecq(
    domain_observation,
    context,
):
    metrics = dict(domain_observation.metrics)
    metrics["er_db"] = MetricValue(
        value=None,
        unit="dB",
        fidelity=metrics["er_db"].fidelity,
        method_version="flexdca-v1",
        valid=False,
        status="INV",
        invalid_reason="ER unavailable",
        measurement_definition_id="FUNC1:ER",
    )
    observation = domain_observation.model_copy(update={"metrics": metrics})

    normalized = normalize_observation(observation, context)

    assert normalized.tdecq_db == 2.4
    assert normalized.er_db is None
    assert normalized.er_valid is False
    assert normalized.er_status == "INV"
