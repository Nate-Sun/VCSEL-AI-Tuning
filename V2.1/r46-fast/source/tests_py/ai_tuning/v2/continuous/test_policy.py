from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
)
from vcsel_ai_tuning.v2.continuous.policy import (
    DEFAULT_I2C_POLICY,
    ProposalErrorCode,
    ProposalValidationError,
    build_restricted_policy,
    materialize_candidate,
)


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc)

APPROVED_B0 = {
    "ibias_dac": 60,
    "efin_cur": 15,
    "vga_cur": 13,
    "eq": 16,
    "res_gain": 6,
    "res_bw": 0,
    "gain_ctrl": 2,
    "vbias_ref": 15,
    "vref": 15,
}

APPROVED_POLICY_CONFIG = {
    "version": "i2c-restricted-v1",
    "max_changed_parameters": 3,
    "rules": {
        "ibias_dac": {"minimum": 60, "maximum": 80, "max_step": 10},
        "efin_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
        "vga_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
        "eq": {"minimum": 16, "maximum": 25, "max_step": 3},
        "res_gain": {"minimum": 0, "maximum": 10, "max_step": 3},
        "res_bw": {"minimum": 0, "maximum": 5, "max_step": 1},
        "gain_ctrl": {"minimum": 0, "maximum": 10, "max_step": 3},
        "vbias_ref": {"minimum": 10, "maximum": 15, "max_step": 3},
        "vref": {"minimum": 10, "maximum": 15, "max_step": 3},
    },
}


def _restricted_policy():
    return build_restricted_policy(
        APPROVED_POLICY_CONFIG,
        authorization_specs=DEFAULT_I2C_POLICY.authorization_specs(),
        baseline_i2c=APPROVED_B0,
    )


def test_build_restricted_policy_preserves_encoding_and_applies_tuning_bounds() -> None:
    policy = _restricted_policy()

    assert policy.version == "i2c-restricted-v1"
    assert policy.max_changed_parameters == 3
    assert {
        rule.name: (
            rule.encoding_min,
            rule.encoding_max,
            rule.tuning_min,
            rule.tuning_max,
            rule.max_step,
        )
        for rule in policy.rules
    } == {
        "ibias_dac": (0, 127, 60, 80, 10),
        "gain_ctrl": (0, 63, 0, 10, 3),
        "eq": (0, 31, 16, 25, 3),
        "efin_cur": (0, 15, 10, 15, 3),
        "vga_cur": (0, 15, 10, 15, 3),
        "res_gain": (0, 15, 0, 10, 3),
        "res_bw": (0, 15, 0, 5, 1),
        "vbias_ref": (0, 15, 10, 15, 3),
        "vref": (0, 15, 10, 15, 3),
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload["rules"].pop("eq"), "exactly"),
        (lambda payload: payload["rules"]["eq"].update(minimum=26), "minimum"),
        (lambda payload: payload["rules"]["eq"].update(maximum=32), "authorization"),
        (lambda payload: payload["rules"]["res_bw"].update(max_step=4), "max_step"),
    ],
)
def test_build_restricted_policy_rejects_invalid_campaign_limits(
    mutator,
    message: str,
) -> None:
    payload = deepcopy(APPROVED_POLICY_CONFIG)
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        build_restricted_policy(
            payload,
            authorization_specs=DEFAULT_I2C_POLICY.authorization_specs(),
            baseline_i2c=APPROVED_B0,
        )


def test_build_restricted_policy_rejects_b0_outside_tuning_range() -> None:
    baseline = dict(APPROVED_B0, eq=15)

    with pytest.raises(ValueError, match="B0.*eq"):
        build_restricted_policy(
            APPROVED_POLICY_CONFIG,
            authorization_specs=DEFAULT_I2C_POLICY.authorization_specs(),
            baseline_i2c=baseline,
        )


def _authorization() -> StageAuthorization:
    return StageAuthorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_FINE,
        baseline_hash="a" * 64,
        rollback_hash="b" * 64,
        parameter_specs=DEFAULT_I2C_POLICY.authorization_specs(),
        max_candidates=3,
        max_duration_s=300.0,
        issued_at=NOW,
        expires_at=datetime(2026, 8, 12, 10, 10, tzinfo=timezone.utc),
        rollback_snapshot=AppliedSettings(),
    )


def test_default_policy_has_complete_ranges_and_ceil_fifteen_percent_steps() -> None:
    assert {
        rule.name: (rule.encoding_min, rule.encoding_max, rule.max_step)
        for rule in DEFAULT_I2C_POLICY.rules
    } == {
        "ibias_dac": (0, 127, 20),
        "gain_ctrl": (0, 63, 10),
        "eq": (0, 31, 5),
        "efin_cur": (0, 15, 3),
        "vga_cur": (0, 15, 3),
        "res_gain": (0, 15, 3),
        "res_bw": (0, 15, 3),
        "vbias_ref": (0, 15, 3),
        "vref": (0, 15, 3),
    }
    assert DEFAULT_I2C_POLICY.version == "i2c-encoding-v1"
    assert DEFAULT_I2C_POLICY.max_changed_parameters == 3


def test_validate_proposal_removes_unchanged_values_without_densifying() -> None:
    current = {name: rule.encoding_min for name, rule in DEFAULT_I2C_POLICY.by_name.items()}
    current.update({"ibias_dac": 70, "gain_ctrl": 5, "eq": 22})

    sparse = DEFAULT_I2C_POLICY.validate_proposal(
        {"ibias_dac": 70, "gain_ctrl": 15, "eq": 27},
        current,
    )

    assert sparse == {"gain_ctrl": 15, "eq": 27}


@pytest.mark.parametrize(
    ("proposal", "code"),
    [
        ({"unknown": 1}, ProposalErrorCode.UNKNOWN_PARAMETER),
        ({"eq": 22.0}, ProposalErrorCode.NON_INTEGER_VALUE),
        ({"eq": True}, ProposalErrorCode.NON_INTEGER_VALUE),
        ({"eq": 32}, ProposalErrorCode.OUTSIDE_ENCODING_RANGE),
        ({"eq": 28}, ProposalErrorCode.EXCEEDS_MAX_STEP),
        (
            {"eq": 23, "vref": 14, "res_bw": 1, "res_gain": 7},
            ProposalErrorCode.TOO_MANY_CHANGED_PARAMETERS,
        ),
        ({"eq": 22}, ProposalErrorCode.NO_EFFECTIVE_CHANGE),
    ],
)
def test_validate_proposal_fails_with_deterministic_error_code(
    proposal: dict[str, object],
    code: ProposalErrorCode,
) -> None:
    current = {
        "ibias_dac": 70,
        "gain_ctrl": 5,
        "eq": 22,
        "efin_cur": 15,
        "vga_cur": 13,
        "res_gain": 6,
        "res_bw": 0,
        "vbias_ref": 15,
        "vref": 15,
    }
    with pytest.raises(ProposalValidationError) as captured:
        DEFAULT_I2C_POLICY.validate_proposal(proposal, current)
    assert captured.value.code is code


def test_restricted_policy_rejects_value_inside_encoding_but_outside_tuning() -> None:
    with pytest.raises(ProposalValidationError) as captured:
        _restricted_policy().validate_proposal(
            {"eq": 26},
            APPROVED_B0,
        )

    assert captured.value.code is ProposalErrorCode.OUTSIDE_TUNING_RANGE


def test_restricted_policy_treats_max_step_as_delta_not_grid_quantum() -> None:
    sparse = _restricted_policy().validate_proposal(
        {"ibias_dac": 67, "res_bw": 1},
        APPROVED_B0,
    )

    assert sparse == {"ibias_dac": 67, "res_bw": 1}


def test_materialize_candidate_is_sparse_local_and_deterministic() -> None:
    authorization = _authorization()
    first = materialize_candidate(
        campaign_id="campaign-1",
        round_index=2,
        attempt=1,
        requested={"eq": 23, "gain_ctrl": 6},
        current_readback={"eq": 22, "gain_ctrl": 5},
        authorization=authorization,
        policy=DEFAULT_I2C_POLICY,
        provider_version="fake-v1",
        reason="bounded exploration",
    )
    second = materialize_candidate(
        campaign_id="campaign-1",
        round_index=2,
        attempt=1,
        requested={"gain_ctrl": 6, "eq": 23},
        current_readback={"eq": 22, "gain_ctrl": 5},
        authorization=authorization,
        policy=DEFAULT_I2C_POLICY,
        provider_version="fake-v1",
        reason="bounded exploration",
    )

    assert first == second
    assert first.candidate_id.startswith("campaign-1-r0002-a1-")
    assert first.requested_i2c == {"eq": 23, "gain_ctrl": 6}
    assert first.requested_fir is None
    assert first.stage is CampaignState.I2C_FINE


def test_materialize_candidate_rejects_cross_campaign_authorization() -> None:
    authorization = _authorization().model_copy(update={"campaign_id": "other"})
    with pytest.raises(ValueError, match="campaign"):
        materialize_candidate(
            campaign_id="campaign-1",
            round_index=1,
            attempt=1,
            requested={"eq": 23},
            current_readback={"eq": 22},
            authorization=authorization,
            policy=DEFAULT_I2C_POLICY,
            provider_version="fake-v1",
            reason="bounded exploration",
        )
