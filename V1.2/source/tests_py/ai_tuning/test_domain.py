from datetime import datetime, timezone

import pytest

from vcsel_ai_tuning.domain import (
    AppliedSettings,
    BerEvidence,
    Candidate,
    ConstraintResult,
    Fidelity,
    MetricValue,
    Observation,
    ParameterSpec,
    RunState,
)


def _invalid_tdecq_metric(
    *,
    value: float | None = None,
    status: str | None = "INV",
    invalid_reason: str | None = "timeout waiting for CORR",
) -> MetricValue:
    return MetricValue(
        value=value,
        unit="dB",
        fidelity=Fidelity.HARDWARE_STANDARD,
        method_version="flexdca-tdecq-v1",
        valid=False,
        status=status,
        invalid_reason=invalid_reason,
        measurement_definition_id="tdecq-standard-v1",
    )


def _observation_with_tdecq(
    metric: MetricValue,
    *,
    observation_valid: bool = False,
) -> Observation:
    now = datetime.now(timezone.utc)
    return Observation(
        campaign_id="campaign-1",
        run_id="run-1",
        candidate_id="candidate-0000",
        state=RunState.COMPLETE if observation_valid else RunState.ABORTED,
        settings=AppliedSettings(
            requested={"bias_ma": 5.0},
            applied={"bias_ma": 5.0},
            readback={"bias_ma": 5.0},
        ),
        metrics={"tdecq_db": metric},
        valid=observation_valid,
        failure_reason=None if observation_valid else "TDECQ measurement failed",
        started_at=now,
        finished_at=now,
    )


def test_parameter_spec_quantizes_from_lower_bound():
    spec = ParameterSpec(
        name="bias_ma",
        unit="mA",
        lower=4.0,
        upper=7.0,
        quantization=0.1,
        max_step=1.0,
        readback_tolerance=0.02,
    )

    assert spec.prepare(4.34) == pytest.approx(4.3)


def test_parameter_spec_rejects_out_of_range_value():
    spec = ParameterSpec(name="bias_ma", unit="mA", lower=4.0, upper=7.0)

    with pytest.raises(ValueError, match="outside"):
        spec.prepare(7.1)


def test_parameter_spec_rejects_nan():
    spec = ParameterSpec(name="bias_ma", unit="mA", lower=4.0, upper=7.0)

    with pytest.raises(ValueError, match="finite"):
        spec.prepare(float("nan"))


def test_invalid_hardware_standard_metric_preserves_instrument_state():
    metric = _invalid_tdecq_metric()

    assert metric.value is None
    assert metric.valid is False
    assert metric.status == "INV"
    assert metric.invalid_reason == "timeout waiting for CORR"
    assert metric.measurement_definition_id == "tdecq-standard-v1"


def test_valid_metric_requires_numeric_value():
    with pytest.raises(ValueError, match="valid metric requires a numeric value"):
        MetricValue(
            value=None,
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="skew-v1",
        )


def test_legacy_valid_metric_without_status_remains_accepted():
    metric = MetricValue(
        value=0.03,
        unit="UI",
        fidelity=Fidelity.SIMULATION_PROXY,
        method_version="skew-v1",
    )

    assert metric.status is None


def test_non_tdecq_invalid_metric_may_omit_status_and_invalid_reason():
    metric = MetricValue(
        value=None,
        unit="UI",
        fidelity=Fidelity.SIMULATION_PROXY,
        method_version="skew-v1",
        valid=False,
    )

    assert metric.value is None
    assert metric.valid is False
    assert metric.status is None
    assert metric.invalid_reason is None


def test_observation_feasible_requires_every_constraint_to_pass():
    settings = AppliedSettings(
        requested={"bias_ma": 5.0},
        applied={"bias_ma": 5.0},
        readback={"bias_ma": 5.0},
    )
    observation = Observation(
        campaign_id="campaign-1",
        run_id="run-1",
        candidate_id="candidate-0000",
        state=RunState.COMPLETE,
        settings=settings,
        metrics={
            "skew_pp_ui": MetricValue(
                value=0.03,
                unit="UI",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="skew-v1",
            )
        },
        constraints=[
            ConstraintResult(name="bias_ma.bounds", passed=True),
            ConstraintResult(name="rlm_proxy", passed=False, reason="below 0.9"),
        ],
        valid=True,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )

    assert observation.feasible is False


def test_valid_observation_must_finish_complete():
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0},
    )

    with pytest.raises(ValueError, match="valid observation"):
        Observation(
            campaign_id="campaign-1",
            run_id="run-1",
            candidate_id=candidate.candidate_id,
            state=RunState.ABORTED,
            settings=AppliedSettings(
                requested=candidate.requested,
                applied=candidate.requested,
                readback=candidate.requested,
            ),
            valid=True,
            failure_reason="aborted",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )


def test_tdecq_db_requires_hardware_standard_fidelity():
    with pytest.raises(ValueError, match="tdecq_db"):
        _observation_with_tdecq(
            MetricValue(
                value=2.0,
                unit="dB",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="invalid-v1",
            ),
            observation_valid=True,
        )


def test_valid_observation_preserves_invalid_tdecq_state():
    observation = _observation_with_tdecq(
        _invalid_tdecq_metric(),
        observation_valid=True,
    )

    metric = observation.metrics["tdecq_db"]
    assert observation.valid is True
    assert observation.state is RunState.COMPLETE
    assert metric.value is None
    assert metric.valid is False
    assert metric.status == "INV"
    assert metric.invalid_reason == "timeout waiting for CORR"
    assert metric.fidelity is Fidelity.HARDWARE_STANDARD
    assert metric.measurement_definition_id == "tdecq-standard-v1"


def test_invalid_tdecq_db_rejects_numeric_substitute():
    with pytest.raises(ValueError, match="invalid tdecq_db must keep value=None"):
        _observation_with_tdecq(_invalid_tdecq_metric(value=0.0))


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
    ],
)
def test_invalid_tdecq_db_requires_nonblank_status(status):
    with pytest.raises(ValueError, match="invalid tdecq_db requires status"):
        _observation_with_tdecq(_invalid_tdecq_metric(status=status))


@pytest.mark.parametrize(
    "invalid_reason",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
    ],
)
def test_invalid_tdecq_db_requires_nonblank_invalid_reason(invalid_reason):
    with pytest.raises(ValueError, match="invalid tdecq_db requires invalid_reason"):
        _observation_with_tdecq(
            _invalid_tdecq_metric(invalid_reason=invalid_reason)
        )


def test_zero_error_ber_evidence_requires_nonzero_upper_bound():
    with pytest.raises(ValueError, match="upper confidence bound"):
        BerEvidence(
            error_count=0,
            bit_count=1_000_000,
            confidence=0.95,
            upper_bound=0.0,
        )
