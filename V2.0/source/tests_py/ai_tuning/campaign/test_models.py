from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.campaign import (
    AppliedSettings,
    AuthorizationMode,
    Campaign,
    CampaignState,
    Candidate,
    FirSettings,
    I2cSettings,
    Observation,
    ParameterSpec,
    StageAuthorization,
    StageSummary,
    load_b0 as load_b0_config,
)
from vcsel_ai_tuning.campaign.models import FirAuthorizationSpec


@pytest.fixture
def load_b0():
    return load_b0_config


def test_b0_uses_confirmed_fir_i2c_pattern_and_nlc(load_b0):
    b0 = load_b0()
    assert b0.pattern == "SSPRQ"
    assert b0.fir.taps == (0, 9, -28, 104, -8, 1, -3)
    assert b0.fir.upper_nlc == b0.fir.lower_nlc == 6
    assert b0.i2c.model_dump() == {
        "ibias_dac": 70,
        "vref": 15,
        "vbias_ref": 15,
        "vga_cur": 13,
        "res_gain": 6,
        "res_bw": 0,
        "gain_ctrl": 5,
        "efin_cur": 15,
        "eq": 22,
    }
    assert sum(abs(v) for v in b0.fir.taps) == 153


def test_campaign_enums_preserve_all_required_states_and_modes():
    assert {state.value for state in CampaignState} == {
        "DRAFT",
        "PREFLIGHT",
        "AWAIT_BASELINE_APPROVAL",
        "RESTORE_B0",
        "BASELINE_MEASUREMENT",
        "AWAIT_COARSE_APPROVAL",
        "I2C_SCREENING",
        "I2C_COARSE",
        "AWAIT_FINE_APPROVAL",
        "I2C_FINE",
        "AWAIT_FIR_APPROVAL",
        "FIR_APPLY",
        "I2C_RETUNE",
        "FINAL_VERIFICATION",
        "COMPLETED",
        "PAUSED",
        "ABORTING",
        "ROLLING_BACK",
        "FAILED",
        "RECOVERY_REQUIRED",
    }
    assert AuthorizationMode.PER_STAGE.value == "PER_STAGE"
    assert AuthorizationMode.FULL_CAMPAIGN.value == "FULL_CAMPAIGN"


@pytest.mark.parametrize(
    "taps, message",
    [
        ((0, 9, -28, 104, -8, 1), "7"),
        ((0, 9, -28, 0, -8, 1, -3), "main"),
        ((0, 9, -28, -104, -8, 1, -3), "main"),
        ((0, 9, -28, 122, -8, 1, -3), "L1"),
    ],
)
def test_fir_rejects_invalid_shape_main_and_l1(taps, message):
    with pytest.raises(ValidationError, match=message):
        FirSettings(taps=taps, upper_nlc=6, lower_nlc=6)


def _fir_authorization_spec(**updates) -> FirAuthorizationSpec:
    values = {
        "tap_min": (-2, 7, -30, 102, -10, -1, -5),
        "tap_max": (2, 11, -26, 106, -6, 3, -1),
        "max_step": (2, 2, 2, 2, 2, 2, 2),
        "max_total_change_l1": 4,
        "fixed_upper_nlc": 6,
        "fixed_lower_nlc": 6,
    }
    values.update(updates)
    return FirAuthorizationSpec(**values)


def test_fir_authorization_spec_is_frozen_strict_and_json_round_trippable():
    spec = _fir_authorization_spec()

    assert FirAuthorizationSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValidationError, match="frozen"):
        spec.max_total_change_l1 = 5
    with pytest.raises(ValidationError, match="extra"):
        FirAuthorizationSpec(**spec.model_dump(), undocumented_limit=1)


@pytest.mark.parametrize("field", ["tap_min", "tap_max", "max_step"])
def test_fir_authorization_spec_requires_exactly_seven_strict_integers(field):
    values = list(getattr(_fir_authorization_spec(), field))
    values.pop()

    with pytest.raises(ValidationError, match="7"):
        _fir_authorization_spec(**{field: tuple(values)})

    values.append(1.5)
    with pytest.raises(ValidationError):
        _fir_authorization_spec(**{field: tuple(values)})


def test_fir_authorization_spec_requires_ordered_tap_ranges():
    tap_min = list(_fir_authorization_spec().tap_min)
    tap_min[2] = -25

    with pytest.raises(ValidationError, match="tap_min.*tap_max|tap.*range"):
        _fir_authorization_spec(tap_min=tuple(tap_min))


def test_fir_authorization_spec_requires_positive_main_range_and_change_limits():
    tap_min = list(_fir_authorization_spec().tap_min)
    tap_min[3] = 0
    with pytest.raises(ValidationError, match="main"):
        _fir_authorization_spec(tap_min=tuple(tap_min))

    with pytest.raises(ValidationError, match="max_step"):
        _fir_authorization_spec(max_step=(2, 2, 0, 2, 2, 2, 2))
    with pytest.raises(ValidationError, match="max_total_change_l1"):
        _fir_authorization_spec(max_total_change_l1=0)


def test_stage_authorization_carries_optional_explicit_fir_envelope():
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    rollback = AppliedSettings(
        readback_fir=FirSettings(
            taps=(0, 9, -28, 104, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
        )
    )
    authorization = StageAuthorization(
        authorization_id="auth-fir-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.FIR_APPLY,
        baseline_hash="a" * 64,
        rollback_hash="b" * 64,
        fir_spec=_fir_authorization_spec(),
        max_candidates=3,
        max_duration_s=60.0,
        issued_at=now,
        expires_at=now + timedelta(minutes=10),
        rollback_snapshot=rollback,
    )

    assert authorization.fir_spec == _fir_authorization_spec()
    assert StageAuthorization.model_validate_json(
        authorization.model_dump_json()
    ) == authorization


def test_models_are_frozen_and_forbid_extra_fields():
    settings = I2cSettings(
        ibias_dac=70,
        vref=15,
        vbias_ref=15,
        vga_cur=13,
        res_gain=6,
        res_bw=0,
        gain_ctrl=5,
        efin_cur=15,
        eq=22,
    )

    with pytest.raises(ValidationError, match="frozen"):
        settings.eq = 23
    with pytest.raises(ValidationError, match="extra"):
        I2cSettings(**settings.model_dump(), undocumented_register=1)


def test_observation_allows_missing_tdecq():
    applied = AppliedSettings(
        requested_i2c={"ibias_dac": 70},
        quantized_i2c={"ibias_dac": 70},
        applied_i2c={"ibias_dac": 70},
        readback_i2c={"ibias_dac": 70},
    )
    observation = Observation(
        observation_id="observation-1",
        campaign_id="campaign-1",
        candidate_id=None,
        stage=CampaignState.BASELINE_MEASUREMENT,
        settings=applied,
        metrics={"er_db": 3.2, "tdecq_db": None},
        valid=True,
        feasible=True,
        elapsed_s=1.5,
    )
    assert observation.metrics["tdecq_db"] is None


def test_all_models_enable_global_strict_mode():
    assert I2cSettings.model_config["strict"] is True


def test_plain_bool_fields_reject_integer_and_string_coercion():
    parameter_fields = {
        "name": "ibias_dac",
        "encoding_min": 0,
        "encoding_max": 255,
        "tuning_min": 64,
        "tuning_max": 76,
        "max_step": 2,
    }
    with pytest.raises(ValidationError):
        ParameterSpec(**parameter_fields, enabled=0)

    applied = AppliedSettings(
        requested_i2c={"ibias_dac": 70},
        quantized_i2c={"ibias_dac": 70},
        applied_i2c={"ibias_dac": 70},
        readback_i2c={"ibias_dac": 70},
    )
    observation_fields = {
        "observation_id": "observation-1",
        "campaign_id": "campaign-1",
        "candidate_id": None,
        "stage": CampaignState.BASELINE_MEASUREMENT,
        "settings": applied,
        "metrics": {"er_db": 3.2, "tdecq_db": None},
        "elapsed_s": 1.5,
    }
    with pytest.raises(ValidationError):
        Observation(**observation_fields, valid=1, feasible=True)
    with pytest.raises(ValidationError):
        Observation(**observation_fields, valid=True, feasible="true")


@pytest.mark.parametrize("invalid_tap", [True, 1.0, "1"])
def test_fir_taps_reject_bool_float_and_string(invalid_tap):
    taps = (invalid_tap, 9, -28, 104, -8, 1, -3)
    with pytest.raises(ValidationError):
        FirSettings(taps=taps, upper_nlc=6, lower_nlc=6)


@pytest.mark.parametrize(
    "non_finite", [float("nan"), float("inf"), float("-inf")]
)
def test_candidate_and_observation_metrics_reject_non_finite_values(non_finite):
    with pytest.raises(ValidationError, match="finite"):
        Candidate(
            candidate_id="candidate-1",
            stage=CampaignState.I2C_COARSE,
            requested_i2c={"ibias_dac": 70},
            predicted_metrics={"er_db": non_finite},
            model_version="optimizer-1",
            reason="strict metric validation",
        )

    applied = AppliedSettings(
        requested_i2c={"ibias_dac": 70},
        quantized_i2c={"ibias_dac": 70},
        applied_i2c={"ibias_dac": 70},
        readback_i2c={"ibias_dac": 70},
    )
    with pytest.raises(ValidationError, match="finite"):
        Observation(
            observation_id="observation-1",
            campaign_id="campaign-1",
            candidate_id=None,
            stage=CampaignState.BASELINE_MEASUREMENT,
            settings=applied,
            metrics={"er_db": non_finite, "tdecq_db": None},
            valid=True,
            feasible=False,
            elapsed_s=1.5,
        )


def test_domain_mappings_are_deeply_immutable_and_serialize_as_json_objects():
    requested_source = {"ibias_dac": 70}
    candidate = Candidate(
        candidate_id="candidate-1",
        stage=CampaignState.I2C_COARSE,
        requested_i2c=requested_source,
        predicted_metrics={"er_db": 3.1, "tdecq_db": None},
        model_version="optimizer-1",
        reason="immutability contract",
    )
    applied = AppliedSettings(
        requested_i2c={"ibias_dac": 70},
        quantized_i2c={"ibias_dac": 70},
        applied_i2c={"ibias_dac": 70},
        readback_i2c={"ibias_dac": 70},
    )
    observation = Observation(
        observation_id="observation-1",
        campaign_id="campaign-1",
        candidate_id=candidate.candidate_id,
        stage=CampaignState.I2C_COARSE,
        settings=applied,
        metrics={"er_db": 3.0, "tdecq_db": None},
        valid=True,
        feasible=True,
        elapsed_s=1.25,
    )

    requested_source["ibias_dac"] = 71
    assert candidate.requested_i2c["ibias_dac"] == 70

    mappings = (
        candidate.requested_i2c,
        candidate.predicted_metrics,
        applied.requested_i2c,
        applied.quantized_i2c,
        applied.applied_i2c,
        applied.readback_i2c,
        observation.metrics,
    )
    for mapping in mappings:
        with pytest.raises(TypeError):
            mapping["new_key"] = 1

    candidate_payload = candidate.model_dump(mode="json")
    observation_payload = observation.model_dump(mode="json")
    assert isinstance(candidate_payload["requested_i2c"], dict)
    assert isinstance(candidate_payload["predicted_metrics"], dict)
    assert isinstance(observation_payload["metrics"], dict)
    assert json.loads(candidate.model_dump_json())["requested_i2c"] == {
        "ibias_dac": 70
    }
    assert json.loads(observation.model_dump_json())["metrics"] == {
        "er_db": 3.0,
        "tdecq_db": None,
    }


def test_default_mappings_are_also_deeply_immutable():
    candidate = Candidate(
        candidate_id="candidate-1",
        stage=CampaignState.I2C_COARSE,
        model_version="optimizer-1",
        reason="default mapping contract",
    )
    applied = AppliedSettings()
    observation = Observation(
        observation_id="observation-1",
        campaign_id="campaign-1",
        candidate_id=candidate.candidate_id,
        stage=CampaignState.I2C_COARSE,
        settings=applied,
        valid=True,
        feasible=True,
        elapsed_s=0.0,
    )

    mappings = (
        candidate.requested_i2c,
        candidate.predicted_metrics,
        applied.requested_i2c,
        applied.quantized_i2c,
        applied.applied_i2c,
        applied.readback_i2c,
        observation.metrics,
    )
    for mapping in mappings:
        with pytest.raises(TypeError):
            mapping["new_key"] = 1


def test_domain_contracts_construct_and_serialize_without_hardware_objects():
    now = datetime.now(timezone.utc)
    parameter = ParameterSpec(
        name="ibias_dac",
        encoding_min=0,
        encoding_max=255,
        tuning_min=64,
        tuning_max=76,
        max_step=2,
        enabled=True,
    )
    fir = FirSettings(
        taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    candidate = Candidate(
        candidate_id="candidate-1",
        stage=CampaignState.I2C_COARSE,
        requested_i2c={"ibias_dac": 72},
        requested_fir=fir,
        predicted_metrics={"er_db": 3.1, "tdecq_db": None},
        model_version="optimizer-1",
        reason="coarse exploration",
    )
    applied = AppliedSettings(
        requested_i2c=candidate.requested_i2c,
        quantized_i2c={"ibias_dac": 72},
        applied_i2c={"ibias_dac": 72},
        readback_i2c={"ibias_dac": 72},
        requested_fir=fir,
        applied_fir=fir,
        readback_fir=fir,
    )
    observation = Observation(
        observation_id="observation-1",
        campaign_id="campaign-1",
        candidate_id=candidate.candidate_id,
        stage=CampaignState.I2C_COARSE,
        settings=applied,
        metrics={"er_db": 3.0, "tdecq_db": None},
        valid=True,
        feasible=True,
        elapsed_s=1.25,
        artifacts=("observations/observation-1.json",),
        measured_at=now,
    )
    authorization = StageAuthorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_COARSE,
        baseline_hash="a" * 64,
        rollback_hash="b" * 64,
        parameter_specs=(parameter,),
        max_candidates=10,
        max_duration_s=300.0,
        issued_at=now,
        expires_at=now + timedelta(minutes=10),
        rollback_snapshot=applied,
    )
    summary = StageSummary(
        stage=CampaignState.I2C_COARSE,
        convergence_reason="budget exhausted",
        best_observation=observation,
        candidates_used=10,
        elapsed_s=120.0,
        warnings=("TDECQ unavailable",),
    )
    campaign = Campaign(
        campaign_id="campaign-1",
        baseline_id="B0",
        baseline_hash=authorization.baseline_hash,
        config_hash="c" * 64,
        authorization_mode=AuthorizationMode.PER_STAGE,
        state=CampaignState.I2C_COARSE,
        stage_summaries=(summary,),
        created_at=now,
        updated_at=now,
    )

    payload = campaign.model_dump(mode="json")
    assert payload["state"] == "I2C_COARSE"
    assert payload["stage_summaries"][0]["best_observation"]["metrics"] == {
        "er_db": 3.0,
        "tdecq_db": None,
    }


@pytest.mark.parametrize("naive_field", ["issued_at", "expires_at"])
def test_stage_authorization_rejects_naive_timestamps_before_comparison(naive_field):
    issued_at = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    fields = {
        "issued_at": issued_at,
        "expires_at": issued_at + timedelta(minutes=10),
    }
    fields[naive_field] = fields[naive_field].replace(tzinfo=None)

    with pytest.raises(ValidationError):
        StageAuthorization(
            authorization_id="auth-1",
            campaign_id="campaign-1",
            mode=AuthorizationMode.PER_STAGE,
            stage=CampaignState.I2C_COARSE,
            baseline_hash="a" * 64,
            rollback_hash="b" * 64,
            max_candidates=10,
            max_duration_s=300.0,
            rollback_snapshot=AppliedSettings(),
            **fields,
        )


def test_observation_rejects_naive_measured_at():
    with pytest.raises(ValidationError):
        Observation(
            observation_id="observation-1",
            campaign_id="campaign-1",
            candidate_id=None,
            stage=CampaignState.BASELINE_MEASUREMENT,
            settings=AppliedSettings(),
            valid=True,
            feasible=True,
            elapsed_s=1.0,
            measured_at=datetime(2026, 7, 22, 12, 0),
        )


@pytest.mark.parametrize("naive_field", ["created_at", "updated_at"])
def test_campaign_rejects_naive_timestamps_before_comparison(naive_field):
    created_at = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    fields = {
        "created_at": created_at,
        "updated_at": created_at + timedelta(minutes=10),
    }
    fields[naive_field] = fields[naive_field].replace(tzinfo=None)

    with pytest.raises(ValidationError):
        Campaign(
            campaign_id="campaign-1",
            baseline_id="B0",
            baseline_hash="a" * 64,
            config_hash="b" * 64,
            **fields,
        )


def test_canonical_json_round_trip_preserves_models_and_immutable_mappings():
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    candidate = Candidate(
        candidate_id="candidate-1",
        stage=CampaignState.I2C_COARSE,
        requested_i2c={"ibias_dac": 72},
        predicted_metrics={"er_db": 3.1, "tdecq_db": None},
        model_version="optimizer-1",
        reason="persistence contract",
    )
    applied = AppliedSettings(
        requested_i2c={"ibias_dac": 72},
        quantized_i2c={"ibias_dac": 72},
        applied_i2c={"ibias_dac": 72},
        readback_i2c={"ibias_dac": 72},
    )
    observation = Observation(
        observation_id="observation-1",
        campaign_id="campaign-1",
        candidate_id=candidate.candidate_id,
        stage=CampaignState.I2C_COARSE,
        settings=applied,
        metrics={"er_db": 3.0, "tdecq_db": None},
        valid=True,
        feasible=True,
        elapsed_s=1.25,
        measured_at=now,
    )
    authorization = StageAuthorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_COARSE,
        baseline_hash="a" * 64,
        rollback_hash="b" * 64,
        max_candidates=10,
        max_duration_s=300.0,
        issued_at=now,
        expires_at=now + timedelta(minutes=10),
        rollback_snapshot=applied,
    )
    campaign = Campaign(
        campaign_id="campaign-1",
        baseline_id="B0",
        baseline_hash="a" * 64,
        config_hash="c" * 64,
        state=CampaignState.I2C_COARSE,
        stage_summaries=(
            StageSummary(
                stage=CampaignState.I2C_COARSE,
                convergence_reason="budget exhausted",
                best_observation=observation,
                candidates_used=10,
                elapsed_s=120.0,
            ),
        ),
        created_at=now,
        updated_at=now,
    )

    restored_candidate = Candidate.model_validate_json(candidate.model_dump_json())
    restored_observation = Observation.model_validate_json(
        observation.model_dump_json()
    )
    restored_authorization = StageAuthorization.model_validate_json(
        authorization.model_dump_json()
    )
    restored_campaign = Campaign.model_validate_json(campaign.model_dump_json())

    for original, restored in (
        (candidate, restored_candidate),
        (observation, restored_observation),
        (authorization, restored_authorization),
        (campaign, restored_campaign),
    ):
        assert restored.model_dump(mode="json") == original.model_dump(mode="json")

    immutable_mappings = (
        restored_candidate.requested_i2c,
        restored_candidate.predicted_metrics,
        restored_observation.settings.requested_i2c,
        restored_observation.metrics,
        restored_authorization.rollback_snapshot.readback_i2c,
        restored_campaign.stage_summaries[0].best_observation.metrics,
    )
    for mapping in immutable_mappings:
        with pytest.raises(TypeError):
            mapping["new_key"] = 1
