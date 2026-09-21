from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from vcsel_ai_tuning.campaign.authorization import (
    AuthorizationError,
    authorization_id_for,
    canonical_json_sha256,
    rollback_hash_for,
    validate_authorization,
    validate_candidate,
    with_authorization_id,
)
from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    CampaignState,
    Candidate,
    FirAuthorizationSpec,
    FirSettings,
    ParameterSpec,
    StageAuthorization,
)


def _authorization(
    *,
    mode: AuthorizationMode = AuthorizationMode.PER_STAGE,
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
    rollback_snapshot: AppliedSettings | None = None,
) -> StageAuthorization:
    issued = issued_at or datetime.now(timezone.utc) - timedelta(minutes=1)
    snapshot = rollback_snapshot or AppliedSettings(
        requested_i2c={"ibias_dac": 70, "vref": 15},
        quantized_i2c={"ibias_dac": 70, "vref": 15},
        applied_i2c={"ibias_dac": 70, "vref": 15},
        readback_i2c={"ibias_dac": 70, "vref": 15},
    )
    unsigned = StageAuthorization(
        authorization_id="unsigned",
        campaign_id="campaign-1",
        mode=mode,
        stage=CampaignState.I2C_COARSE,
        baseline_hash="a" * 64,
        rollback_hash=rollback_hash_for(snapshot),
        parameter_specs=(
            ParameterSpec(
                name="ibias_dac",
                encoding_min=0,
                encoding_max=255,
                tuning_min=64,
                tuning_max=76,
                max_step=2,
                enabled=True,
            ),
            ParameterSpec(
                name="vref",
                encoding_min=0,
                encoding_max=255,
                tuning_min=10,
                tuning_max=20,
                max_step=1,
                enabled=False,
            ),
        ),
        max_candidates=3,
        max_duration_s=30.0,
        issued_at=issued,
        expires_at=expires_at or issued + timedelta(hours=1),
        rollback_snapshot=snapshot,
    )
    return with_authorization_id(unsigned)


def _continuous_unbounded_authorization(
    *, stage: CampaignState = CampaignState.I2C_SCREENING
) -> StageAuthorization:
    bounded = _authorization()
    unsigned = StageAuthorization.model_validate(
        {
            **bounded.model_dump(mode="python", warnings=False),
            "authorization_id": "unsigned",
            "time_policy": AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
            "stage": stage,
            "max_duration_s": None,
            "expires_at": None,
        }
    )
    return with_authorization_id(unsigned)


@pytest.fixture
def auth() -> StageAuthorization:
    return _authorization()


@pytest.fixture
def full_campaign_auth() -> StageAuthorization:
    return _authorization(mode=AuthorizationMode.FULL_CAMPAIGN)


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        candidate_id="candidate-1",
        stage=CampaignState.I2C_COARSE,
        requested_i2c={"ibias_dac": 72},
        model_version="optimizer-1",
        reason="coarse exploration",
    )


def _context(
    authorization: StageAuthorization, *, now: datetime | None = None
) -> dict[str, object]:
    return {
        "now": now or authorization.issued_at + timedelta(minutes=1),
        "current_campaign_id": authorization.campaign_id,
        "current_baseline_hash": authorization.baseline_hash,
    }


def _with_rollback_snapshot(
    authorization: StageAuthorization, snapshot: AppliedSettings
) -> StageAuthorization:
    changed = authorization.model_copy(update={"rollback_snapshot": snapshot})
    changed = changed.model_copy(update={"rollback_hash": rollback_hash_for(snapshot)})
    return with_authorization_id(changed)


def _rollback_missing_ibias(kind: str) -> AppliedSettings:
    if kind == "unrelated-i2c":
        return AppliedSettings(readback_i2c={"vref": 15})
    return AppliedSettings(
        readback_fir=FirSettings(
            taps=(0, 9, -28, 104, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
        )
    )


def _with_ibias_rollback(
    authorization: StageAuthorization, *, value: int, encoding_min: int
) -> StageAuthorization:
    ibias_spec = authorization.parameter_specs[0].model_copy(
        update={"encoding_min": encoding_min}
    )
    changed = authorization.model_copy(
        update={"parameter_specs": (ibias_spec, *authorization.parameter_specs[1:])}
    )
    return _with_rollback_snapshot(
        changed, AppliedSettings(readback_i2c={"ibias_dac": value})
    )


def test_canonical_json_sha256_ignores_mapping_insertion_order():
    left = {"outer": {"b": 2, "a": 1}, "items": [{"d": 4, "c": 3}]}
    right = {"items": [{"c": 3, "d": 4}], "outer": {"a": 1, "b": 2}}

    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_canonical_json_sha256_normalizes_aware_datetimes_to_utc():
    instant_utc = datetime(2026, 7, 22, 4, 0, tzinfo=timezone.utc)
    instant_cst = datetime(
        2026, 7, 22, 12, 0, tzinfo=timezone(timedelta(hours=8))
    )

    assert canonical_json_sha256({"at": instant_utc}) == canonical_json_sha256(
        {"at": instant_cst}
    )


def test_authorization_id_is_deterministic_for_equivalent_mapping_order():
    first_snapshot = AppliedSettings(
        requested_i2c={"ibias_dac": 70, "vref": 15},
        readback_i2c={"ibias_dac": 70, "vref": 15},
    )
    second_snapshot = AppliedSettings(
        requested_i2c={"vref": 15, "ibias_dac": 70},
        readback_i2c={"vref": 15, "ibias_dac": 70},
    )
    issued = datetime(2026, 7, 22, 4, 0, tzinfo=timezone.utc)
    first = _authorization(
        issued_at=issued,
        expires_at=issued + timedelta(hours=1),
        rollback_snapshot=first_snapshot,
    )
    second = _authorization(
        issued_at=issued,
        expires_at=issued + timedelta(hours=1),
        rollback_snapshot=second_snapshot,
    )

    assert authorization_id_for(first) == authorization_id_for(second)
    assert first.authorization_id == second.authorization_id


def test_authorization_id_uses_utc_instants_not_original_offsets():
    issued_utc = datetime(2026, 7, 22, 4, 0, tzinfo=timezone.utc)
    issued_cst = datetime(
        2026, 7, 22, 12, 0, tzinfo=timezone(timedelta(hours=8))
    )
    first = _authorization(
        issued_at=issued_utc,
        expires_at=issued_utc + timedelta(hours=1),
    )
    second = _authorization(
        issued_at=issued_cst,
        expires_at=issued_cst + timedelta(hours=1),
    )

    assert first.authorization_id == second.authorization_id


def test_authorization_id_does_not_hash_itself(auth):
    changed_id = auth.model_copy(update={"authorization_id": "different"})

    assert authorization_id_for(changed_id) == authorization_id_for(auth)


def test_bounded_authorization_preserves_legacy_v1_id_and_validates():
    issued_at = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    authorization = _authorization(
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    legacy_id = canonical_json_sha256(
        {
            "schema": "stage-authorization-v1",
            "campaign_id": authorization.campaign_id,
            "mode": authorization.mode,
            "stage": authorization.stage,
            "baseline_hash": authorization.baseline_hash,
            "rollback_hash": authorization.rollback_hash,
            "parameter_specs": tuple(
                sorted(
                    authorization.parameter_specs,
                    key=lambda spec: spec.name,
                )
            ),
            "fir_spec": authorization.fir_spec,
            "max_candidates": authorization.max_candidates,
            "max_duration_s": authorization.max_duration_s,
            "issued_at": authorization.issued_at,
            "expires_at": authorization.expires_at,
        }
    )
    persisted = authorization.model_dump(mode="json", warnings=False)
    persisted.pop("time_policy")
    persisted["authorization_id"] = legacy_id

    restored = StageAuthorization.model_validate_json(json.dumps(persisted))

    assert restored.time_policy is AuthorizationTimePolicy.BOUNDED
    assert authorization_id_for(restored) == legacy_id
    assert validate_authorization(restored, **_context(restored)) == restored


def test_continuous_unbounded_authorization_id_is_v2_and_policy_bound():
    authorization = _continuous_unbounded_authorization()
    expected_id = canonical_json_sha256(
        {
            "schema": "stage-authorization-v2",
            "campaign_id": authorization.campaign_id,
            "mode": authorization.mode,
            "time_policy": authorization.time_policy,
            "stage": authorization.stage,
            "baseline_hash": authorization.baseline_hash,
            "rollback_hash": authorization.rollback_hash,
            "parameter_specs": tuple(
                sorted(
                    authorization.parameter_specs,
                    key=lambda spec: spec.name,
                )
            ),
            "fir_spec": authorization.fir_spec,
            "max_candidates": authorization.max_candidates,
            "max_duration_s": authorization.max_duration_s,
            "issued_at": authorization.issued_at,
            "expires_at": authorization.expires_at,
        }
    )
    changed_policy = authorization.model_copy(
        update={"time_policy": AuthorizationTimePolicy.BOUNDED}
    )

    assert authorization.authorization_id == expected_id
    assert authorization_id_for(authorization) == expected_id
    assert authorization_id_for(changed_policy) != expected_id


def test_authorization_id_sorts_parameter_specs_by_name(auth):
    reversed_specs = auth.model_copy(
        update={"parameter_specs": tuple(reversed(auth.parameter_specs))}
    )

    assert authorization_id_for(reversed_specs) == authorization_id_for(auth)
    assert with_authorization_id(reversed_specs).authorization_id == auth.authorization_id


@pytest.mark.parametrize(
    "derive_id", [authorization_id_for, with_authorization_id]
)
def test_authorization_id_rejects_duplicate_parameter_name(auth, derive_id):
    duplicate = auth.model_copy(
        update={"parameter_specs": (*auth.parameter_specs, auth.parameter_specs[0])}
    )

    with pytest.raises(AuthorizationError, match="duplicate.*ibias_dac"):
        derive_id(duplicate)


def test_validate_authorization_rejects_duplicate_parameter_name(auth):
    duplicate = auth.model_copy(
        update={"parameter_specs": (*auth.parameter_specs, auth.parameter_specs[0])}
    )

    with pytest.raises(AuthorizationError, match="duplicate.*ibias_dac"):
        validate_authorization(duplicate, **_context(auth))


def test_valid_per_stage_authorization_is_returned(auth):
    validated = validate_authorization(auth, **_context(auth))

    assert validated == auth
    assert validated is not auth


def test_continuous_unbounded_authorization_validates_far_after_issued_at():
    authorization = _continuous_unbounded_authorization()

    validated = validate_authorization(
        authorization,
        **_context(
            authorization,
            now=authorization.issued_at + timedelta(days=36500),
        ),
    )

    assert validated == authorization
    assert validated is not authorization


def test_continuous_unbounded_authorization_keeps_not_yet_valid_check():
    authorization = _continuous_unbounded_authorization()

    with pytest.raises(AuthorizationError, match="not yet valid"):
        validate_authorization(
            authorization,
            **_context(
                authorization,
                now=authorization.issued_at - timedelta(microseconds=1),
            ),
        )


def test_continuous_unbounded_authorization_requires_aware_now():
    authorization = _continuous_unbounded_authorization()

    with pytest.raises(AuthorizationError, match="now.*timezone-aware"):
        validate_authorization(
            authorization,
            **_context(authorization, now=datetime(2026, 7, 22, 12, 0)),
        )


def test_continuous_unbounded_authorization_rejects_tampered_expiry():
    authorization = _continuous_unbounded_authorization()
    tampered = with_authorization_id(
        authorization.model_copy(
            update={"expires_at": authorization.issued_at + timedelta(hours=1)}
        )
    )

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_authorization(tampered, **_context(authorization))


@pytest.mark.parametrize(
    ("context_update", "message"),
    [
        ({"current_campaign_id": "campaign-2"}, "campaign.*scope"),
        ({"current_baseline_hash": "b" * 64}, "baseline.*stale"),
    ],
)
def test_continuous_unbounded_authorization_enforces_runtime_context(
    context_update, message
):
    authorization = _continuous_unbounded_authorization()
    context = {**_context(authorization), **context_update}

    with pytest.raises(AuthorizationError, match=message):
        validate_authorization(authorization, **context)


@pytest.mark.parametrize("tamper_kind", ["rollback-snapshot", "rollback-hash"])
def test_continuous_unbounded_authorization_rejects_rollback_tampering(tamper_kind):
    authorization = _continuous_unbounded_authorization()
    if tamper_kind == "rollback-snapshot":
        rollback_snapshot = authorization.rollback_snapshot.model_copy(
            update={"readback_i2c": {"ibias_dac": 71, "vref": 15}}
        )
        tampered = authorization.model_copy(
            update={"rollback_snapshot": rollback_snapshot}
        )
    else:
        tampered = with_authorization_id(
            authorization.model_copy(update={"rollback_hash": "f" * 64})
        )

    with pytest.raises(AuthorizationError, match="rollback.*hash"):
        validate_authorization(tampered, **_context(authorization))


@pytest.mark.parametrize(
    "missing", ["now", "current_campaign_id", "current_baseline_hash"]
)
def test_valid_per_stage_authorization_requires_complete_context(auth, missing):
    context = _context(auth)
    context.pop(missing)

    with pytest.raises(AuthorizationError, match=f"{missing}.*required"):
        validate_authorization(auth, **context)


def test_full_campaign_is_reserved_but_disabled(full_campaign_auth):
    with pytest.raises(AuthorizationError, match="FULL_CAMPAIGN.*disabled"):
        validate_authorization(full_campaign_auth)


def test_per_candidate_is_disabled_in_v1():
    per_candidate = _authorization(mode=AuthorizationMode.PER_CANDIDATE)

    with pytest.raises(AuthorizationError, match="PER_CANDIDATE.*disabled"):
        validate_authorization(per_candidate)


def test_authorization_rejects_tampered_id(auth):
    tampered = auth.model_copy(update={"authorization_id": "tampered"})

    with pytest.raises(AuthorizationError, match="authorization_id.*hash"):
        validate_authorization(tampered, **_context(auth))


def test_authorization_rejects_expiry_equality(auth):
    with pytest.raises(AuthorizationError, match="expired"):
        validate_authorization(auth, **_context(auth, now=auth.expires_at))


def test_authorization_rejects_time_before_issued_at(auth):
    with pytest.raises(AuthorizationError, match="not yet valid"):
        validate_authorization(
            auth,
            **_context(auth, now=auth.issued_at - timedelta(microseconds=1)),
        )


def test_authorization_rejects_unreasonable_timestamp_order(auth):
    invalid = auth.model_copy(update={"expires_at": auth.issued_at})

    with pytest.raises(AuthorizationError, match="expires_at.*issued_at"):
        validate_authorization(invalid, **_context(auth, now=auth.issued_at))


def test_authorization_rejects_naive_now(auth):
    with pytest.raises(AuthorizationError, match="now.*timezone-aware"):
        validate_authorization(
            auth, **_context(auth, now=datetime(2026, 7, 22, 12, 0))
        )


@pytest.mark.parametrize("field", ["issued_at", "expires_at"])
def test_authorization_rejects_tampered_naive_timestamps(auth, field):
    tampered = auth.model_copy(
        update={field: getattr(auth, field).replace(tzinfo=None)}
    )

    with pytest.raises(AuthorizationError, match=f"{field}.*timezone-aware"):
        validate_authorization(tampered, **_context(auth))


def test_authorization_rejects_stale_b0(auth):
    with pytest.raises(AuthorizationError, match="baseline.*stale"):
        validate_authorization(
            auth, **{**_context(auth), "current_baseline_hash": "b" * 64}
        )


def test_authorization_rejects_wrong_campaign_context(auth):
    with pytest.raises(AuthorizationError, match="campaign.*scope"):
        validate_authorization(
            auth, **{**_context(auth), "current_campaign_id": "campaign-2"}
        )


def test_authorization_rejects_rollback_snapshot_tamper(auth):
    tampered_snapshot = auth.rollback_snapshot.model_copy(
        update={"readback_i2c": {"ibias_dac": 71, "vref": 15}}
    )
    tampered = auth.model_copy(update={"rollback_snapshot": tampered_snapshot})

    with pytest.raises(AuthorizationError, match="rollback.*hash"):
        validate_authorization(tampered, **_context(auth))


def test_authorization_rejects_rollback_hash_tamper(auth):
    tampered = auth.model_copy(update={"rollback_hash": "f" * 64})

    with pytest.raises(AuthorizationError, match="rollback.*hash"):
        validate_authorization(tampered, **_context(auth))


def test_authorization_rejects_empty_rollback_snapshot_even_with_matching_hash(auth):
    tampered = auth.model_copy(update={"rollback_snapshot": {}})
    tampered = tampered.model_copy(
        update={"rollback_hash": rollback_hash_for(tampered.rollback_snapshot)}
    )
    tampered = with_authorization_id(tampered)

    with pytest.raises(
        AuthorizationError, match="rollback snapshot.*(unusable|missing readback)"
    ):
        validate_authorization(tampered, **_context(auth))


@pytest.mark.parametrize("rollback_kind", ["unrelated-i2c", "fir-only"])
def test_authorization_requires_rollback_for_every_enabled_i2c_parameter(
    auth, rollback_kind
):
    tampered = _with_rollback_snapshot(
        auth, _rollback_missing_ibias(rollback_kind)
    )

    with pytest.raises(
        AuthorizationError, match="rollback.*ibias_dac.*missing"
    ):
        validate_authorization(tampered, **_context(auth))


@pytest.mark.parametrize("rollback_kind", ["unrelated-i2c", "fir-only"])
def test_candidate_rejects_authorization_missing_enabled_parameter_rollback(
    auth, candidate, rollback_kind
):
    tampered = _with_rollback_snapshot(
        auth, _rollback_missing_ibias(rollback_kind)
    )

    with pytest.raises(
        AuthorizationError, match="rollback.*ibias_dac.*missing"
    ):
        validate_candidate(
            tampered,
            candidate,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


def test_disabled_i2c_parameter_does_not_require_rollback_coverage(auth):
    rollback = AppliedSettings(readback_i2c={"ibias_dac": 70})
    changed = _with_rollback_snapshot(auth, rollback)

    assert validate_authorization(changed, **_context(auth)) == changed


@pytest.mark.parametrize(
    ("encoding_min", "rollback_value"), [(10, 9), (0, 256)]
)
def test_authorization_rejects_enabled_rollback_outside_encoding_range(
    auth, encoding_min, rollback_value
):
    tampered = _with_ibias_rollback(
        auth, value=rollback_value, encoding_min=encoding_min
    )

    with pytest.raises(
        AuthorizationError, match="rollback.*ibias_dac.*encoding range"
    ):
        validate_authorization(tampered, **_context(auth))


@pytest.mark.parametrize(
    ("encoding_min", "rollback_value"), [(10, 9), (0, 256)]
)
def test_candidate_rejects_enabled_rollback_outside_encoding_range(
    auth, candidate, encoding_min, rollback_value
):
    tampered = _with_ibias_rollback(
        auth, value=rollback_value, encoding_min=encoding_min
    )

    with pytest.raises(
        AuthorizationError, match="rollback.*ibias_dac.*encoding range"
    ):
        validate_candidate(
            tampered,
            candidate,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


@pytest.mark.parametrize("rollback_value", [10, 255])
def test_authorization_accepts_rollback_at_encoding_range_boundaries(
    auth, rollback_value
):
    changed = _with_ibias_rollback(
        auth, value=rollback_value, encoding_min=10
    )

    assert validate_authorization(changed, **_context(auth)) == changed


def test_authorization_rejects_non_integer_max_candidates_after_model_copy(auth):
    tampered = with_authorization_id(
        auth.model_copy(update={"max_candidates": 3.5})
    )

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_authorization(tampered, **_context(auth))


def test_candidate_rejects_non_integer_authorization_budget(auth, candidate):
    tampered = with_authorization_id(
        auth.model_copy(update={"max_candidates": 3.5})
    )

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_candidate(
            tampered,
            candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=3,
            **_context(auth),
        )


def test_authorization_rejects_string_duration_after_model_copy(auth):
    tampered = with_authorization_id(
        auth.model_copy(update={"max_duration_s": "30"})
    )

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_authorization(tampered, **_context(auth))


def test_candidate_converts_string_duration_failure_to_authorization_error(
    auth, candidate
):
    tampered = with_authorization_id(
        auth.model_copy(update={"max_duration_s": "30"})
    )

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_candidate(
            tampered,
            candidate,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


@pytest.mark.parametrize("malformed_specs", [({"name": "ibias_dac"},), ("bad",)])
def test_authorization_rejects_malformed_parameter_spec_container(
    auth, malformed_specs
):
    tampered = auth.model_copy(update={"parameter_specs": malformed_specs})

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_authorization(tampered, **_context(auth))


def test_authorization_rejects_parameter_spec_with_invalid_nested_field(auth):
    invalid_spec = auth.parameter_specs[0].model_copy(
        update={"encoding_min": "zero"}
    )
    tampered = with_authorization_id(
        auth.model_copy(update={"parameter_specs": (invalid_spec,)})
    )

    with pytest.raises(AuthorizationError, match="invalid authorization"):
        validate_authorization(tampered, **_context(auth))


def test_stage_authorization_rejects_candidate_outside_envelope(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"ibias_dac": 80}})
    with pytest.raises(AuthorizationError, match="ibias_dac.*range"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


def test_valid_candidate_is_returned(auth, candidate):
    assert (
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=0,
            elapsed_s=0.0,
            **_context(auth),
        )
        is candidate
    )


@pytest.mark.parametrize(
    "missing", ["now", "current_campaign_id", "current_baseline_hash"]
)
def test_validate_candidate_requires_complete_authorization_context(
    auth, candidate, missing
):
    context = _context(auth)
    context.pop(missing)

    with pytest.raises(AuthorizationError, match=f"{missing}.*required"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            **context,
        )


def test_candidate_rejects_stage_mismatch(auth, candidate):
    changed = candidate.model_copy(update={"stage": CampaignState.I2C_FINE})

    with pytest.raises(AuthorizationError, match="stage.*scope"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


def test_candidate_rejects_unknown_parameter(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"unknown_dac": 1}})

    with pytest.raises(AuthorizationError, match="unknown_dac.*unknown"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"unknown_dac": 1},
            **_context(auth),
        )


def test_candidate_rejects_disabled_parameter(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"vref": 15}})

    with pytest.raises(AuthorizationError, match="vref.*disabled"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"vref": 15},
            **_context(auth),
        )


@pytest.mark.parametrize("value", [70.0, True, "70"])
def test_candidate_rejects_non_strict_integer_request(auth, candidate, value):
    changed = candidate.model_copy(update={"requested_i2c": {"ibias_dac": value}})

    with pytest.raises(AuthorizationError, match="ibias_dac.*integer"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


def test_candidate_rejects_value_outside_encoding_range(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"ibias_dac": -1}})

    with pytest.raises(AuthorizationError, match="ibias_dac.*encoding range"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 0},
            **_context(auth),
        )


def test_candidate_rejects_value_outside_tuning_range(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"ibias_dac": 77}})

    with pytest.raises(AuthorizationError, match="ibias_dac.*tuning range"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 76},
            **_context(auth),
        )


def test_candidate_max_step_uses_actual_previous_readback(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"ibias_dac": 72}})

    with pytest.raises(AuthorizationError, match="ibias_dac.*max_step"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 69},
            **_context(auth),
        )


def test_candidate_accepts_max_step_equality(auth, candidate):
    assert (
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )
        is candidate
    )


def test_candidate_rejects_missing_previous_readback(auth, candidate):
    with pytest.raises(AuthorizationError, match="ibias_dac.*readback.*missing"):
        validate_candidate(
            auth, candidate, previous_readback={}, **_context(auth)
        )


@pytest.mark.parametrize("value", [70.0, True, "70"])
def test_candidate_rejects_non_strict_integer_readback(auth, candidate, value):
    with pytest.raises(AuthorizationError, match="ibias_dac.*readback.*integer"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": value},
            **_context(auth),
        )


def test_candidate_rejects_readback_outside_encoding_range(auth, candidate):
    with pytest.raises(AuthorizationError, match="ibias_dac.*readback.*encoding range"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 256},
            **_context(auth),
        )


def test_candidate_rejects_fir_without_explicit_authorization(auth, candidate):
    fir = FirSettings(
        taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    changed = candidate.model_copy(update={"requested_fir": fir})

    with pytest.raises(AuthorizationError, match="FIR.*not authorized"):
        validate_candidate(
            auth,
            changed,
            previous_readback={"ibias_dac": 70},
            **_context(auth),
        )


def _fir_authorization(
    *,
    tap_max_step: int = 2,
    total_change_l1: int = 4,
    fir_spec: FirAuthorizationSpec | None = None,
) -> StageAuthorization:
    rollback_fir = FirSettings(
        taps=(0, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    rollback = AppliedSettings(readback_fir=rollback_fir)
    envelope = fir_spec or FirAuthorizationSpec(
        tap_min=tuple(value - 4 for value in rollback_fir.taps),
        tap_max=tuple(value + 4 for value in rollback_fir.taps),
        max_step=(tap_max_step,) * 7,
        max_total_change_l1=total_change_l1,
        fixed_upper_nlc=6,
        fixed_lower_nlc=6,
    )
    unsigned = StageAuthorization(
        authorization_id="pending",
        campaign_id="campaign-1",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.FIR_APPLY,
        baseline_hash="a" * 64,
        rollback_hash=rollback_hash_for(rollback),
        fir_spec=envelope,
        max_candidates=3,
        max_duration_s=60.0,
        issued_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=59),
        rollback_snapshot=rollback,
    )
    return with_authorization_id(unsigned)


def _fir_candidate(
    *,
    taps: tuple[int, ...] = (0, 9, -27, 103, -8, 1, -3),
    upper_nlc: int = 6,
    lower_nlc: int = 6,
    stage: CampaignState = CampaignState.FIR_APPLY,
) -> Candidate:
    return Candidate(
        candidate_id="fir-candidate-1",
        stage=stage,
        requested_fir=FirSettings(
            taps=taps,
            upper_nlc=upper_nlc,
            lower_nlc=lower_nlc,
        ),
        model_version="fir-proposer-1",
        reason="direct FIR formula proposal",
    )


def test_fir_apply_authorization_accepts_strict_fir_inside_explicit_envelope():
    authorization = _fir_authorization()
    candidate = _fir_candidate()

    assert (
        validate_candidate(
            authorization,
            candidate,
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )
        is candidate
    )


def test_authorization_id_hashes_explicit_fir_envelope():
    authorization = _fir_authorization(total_change_l1=4)
    changed_spec = authorization.fir_spec.model_copy(
        update={"max_total_change_l1": 5}
    )
    changed = authorization.model_copy(update={"fir_spec": changed_spec})

    assert authorization_id_for(changed) != authorization.authorization_id
    with pytest.raises(AuthorizationError, match="authorization_id.*hash"):
        validate_authorization(changed, **_context(authorization))


def test_non_fir_authorization_rejects_fir_envelope(auth):
    changed = auth.model_copy(update={"fir_spec": _fir_authorization().fir_spec})
    changed = with_authorization_id(changed)

    with pytest.raises(AuthorizationError, match="fir_spec.*FIR_APPLY"):
        validate_authorization(changed, **_context(auth))


def test_fir_apply_requires_explicit_fir_spec():
    authorization = _fir_authorization().model_copy(update={"fir_spec": None})
    authorization = with_authorization_id(authorization)

    with pytest.raises(AuthorizationError, match="fir_spec|FIR.*envelope"):
        validate_candidate(
            authorization,
            _fir_candidate(),
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_requires_fir_rollback_readback():
    authorization = _fir_authorization()
    rollback = AppliedSettings(readback_i2c={"ibias_dac": 70})
    changed = authorization.model_copy(
        update={
            "rollback_snapshot": rollback,
            "rollback_hash": rollback_hash_for(rollback),
        }
    )
    changed = with_authorization_id(changed)

    with pytest.raises(AuthorizationError, match="rollback.*FIR|FIR.*rollback"):
        validate_candidate(
            changed,
            _fir_candidate(),
            previous_readback={},
            previous_fir=None,
            **_context(changed),
        )


@pytest.mark.parametrize(
    ("upper_nlc", "lower_nlc"),
    [(7, 6), (6, 5)],
)
def test_fir_apply_keeps_both_nlc_values_frozen_to_rollback_snapshot(
    upper_nlc, lower_nlc
):
    authorization = _fir_authorization()

    with pytest.raises(AuthorizationError, match="NLC.*frozen"):
        validate_candidate(
            authorization,
            _fir_candidate(upper_nlc=upper_nlc, lower_nlc=lower_nlc),
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_rejects_candidate_from_another_stage():
    authorization = _fir_authorization()

    with pytest.raises(AuthorizationError, match="stage.*scope"):
        validate_candidate(
            authorization,
            _fir_candidate(stage=CampaignState.I2C_RETUNE),
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_rejects_i2c_changes_in_same_candidate():
    authorization = _fir_authorization()
    candidate = _fir_candidate().model_copy(
        update={"requested_i2c": {"ibias_dac": 71}}
    )

    with pytest.raises(AuthorizationError, match="FIR_APPLY.*I2C|I2C.*FIR_APPLY"):
        validate_candidate(
            authorization,
            candidate,
            previous_readback={"ibias_dac": 70},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_enforces_each_tap_absolute_range():
    authorization = _fir_authorization()

    with pytest.raises(AuthorizationError, match="tap 0.*range"):
        validate_candidate(
            authorization,
            _fir_candidate(taps=(5, 9, -28, 104, -8, 1, -3)),
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_enforces_each_tap_max_step_from_rollback_readback():
    authorization = _fir_authorization(tap_max_step=2, total_change_l1=8)

    with pytest.raises(AuthorizationError, match="tap 0.*max_step"):
        validate_candidate(
            authorization,
            _fir_candidate(taps=(3, 9, -28, 104, -8, 1, -3)),
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_enforces_total_l1_change_from_rollback_snapshot():
    authorization = _fir_authorization(tap_max_step=2, total_change_l1=4)

    with pytest.raises(AuthorizationError, match="total.*change"):
        validate_candidate(
            authorization,
            _fir_candidate(taps=(2, 9, -26, 102, -8, 1, -3)),
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )


def test_fir_apply_accepts_total_l1_change_at_authorized_boundary():
    authorization = _fir_authorization(tap_max_step=2, total_change_l1=4)
    candidate = _fir_candidate(taps=(2, 9, -26, 104, -8, 1, -3))

    assert (
        validate_candidate(
            authorization,
            candidate,
            previous_readback={},
            previous_fir=authorization.rollback_snapshot.readback_fir,
            **_context(authorization),
        )
        is candidate
    )


def test_fir_apply_validates_step_against_actual_previous_fir_not_only_rollback():
    authorization = _fir_authorization(tap_max_step=2, total_change_l1=8)
    actual_previous = FirSettings(
        taps=(-2, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )

    with pytest.raises(AuthorizationError, match="tap 0.*max_step"):
        validate_candidate(
            authorization,
            _fir_candidate(taps=(1, 9, -28, 104, -8, 1, -3)),
            previous_readback={},
            previous_fir=actual_previous,
            **_context(authorization),
        )


def test_fir_apply_requires_actual_previous_fir_readback():
    authorization = _fir_authorization()

    with pytest.raises(AuthorizationError, match="previous_fir.*required"):
        validate_candidate(
            authorization,
            _fir_candidate(),
            previous_readback={},
            previous_fir=None,
            **_context(authorization),
        )


def test_static_candidate_validation_checks_scope_range_and_budget_without_readback(
    auth, candidate
):
    from vcsel_ai_tuning.campaign.authorization import validate_candidate_static

    assert (
        validate_candidate_static(
            auth,
            candidate,
            candidate_index=0,
            elapsed_s=0.0,
            **_context(auth),
        )
        is candidate
    )
    with pytest.raises(AuthorizationError, match="range"):
        validate_candidate_static(
            auth,
            candidate.model_copy(update={"requested_i2c": {"ibias_dac": 80}}),
            candidate_index=0,
            elapsed_s=0.0,
            **_context(auth),
        )
    with pytest.raises(AuthorizationError, match="max_candidates"):
        validate_candidate_static(
            auth,
            candidate,
            candidate_index=auth.max_candidates,
            elapsed_s=0.0,
            **_context(auth),
        )
    with pytest.raises(AuthorizationError, match="max_duration"):
        validate_candidate_static(
            auth,
            candidate,
            candidate_index=0,
            elapsed_s=auth.max_duration_s,
            **_context(auth),
        )


def test_static_validation_defers_only_i2c_max_step_to_actual_readback(
    auth, candidate
):
    from vcsel_ai_tuning.campaign.authorization import validate_candidate_static

    assert validate_candidate_static(
        auth,
        candidate,
        candidate_index=0,
        elapsed_s=0.0,
        **_context(auth),
    ) is candidate
    with pytest.raises(AuthorizationError, match="max_step"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 68},
            candidate_index=0,
            elapsed_s=0.0,
            **_context(auth),
        )


def test_static_fir_validation_checks_absolute_envelope_but_defers_delta():
    from vcsel_ai_tuning.campaign.authorization import validate_candidate_static

    authorization = _fir_authorization(tap_max_step=1, total_change_l1=2)
    candidate = _fir_candidate(taps=(1, 9, -28, 104, -8, 1, -3))

    assert validate_candidate_static(
        authorization,
        candidate,
        candidate_index=0,
        elapsed_s=0.0,
        **_context(authorization),
    ) is candidate
    actual_previous = FirSettings(
        taps=(-1, 9, -28, 104, -8, 1, -3),
        upper_nlc=6,
        lower_nlc=6,
    )
    with pytest.raises(AuthorizationError, match="max_step"):
        validate_candidate(
            authorization,
            candidate,
            previous_readback={},
            previous_fir=actual_previous,
            candidate_index=0,
            elapsed_s=0.0,
            **_context(authorization),
        )


def test_candidate_budget_uses_zero_based_index_and_accepts_last_slot(
    auth, candidate
):
    assert (
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=auth.max_candidates - 1,
            **_context(auth),
        )
        is candidate
    )


def test_candidate_rejects_index_at_candidate_budget(auth, candidate):
    with pytest.raises(AuthorizationError, match="candidate_index.*max_candidates"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=auth.max_candidates,
            **_context(auth),
        )


def test_continuous_unbounded_candidate_accepts_large_finite_elapsed_time(candidate):
    authorization = _continuous_unbounded_authorization()
    screening_candidate = candidate.model_copy(
        update={"stage": CampaignState.I2C_SCREENING}
    )

    assert (
        validate_candidate(
            authorization,
            screening_candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=authorization.max_candidates - 1,
            elapsed_s=1.0e100,
            **_context(
                authorization,
                now=authorization.issued_at + timedelta(days=36500),
            ),
        )
        is screening_candidate
    )


def test_continuous_unbounded_candidate_still_enforces_candidate_budget(candidate):
    authorization = _continuous_unbounded_authorization()
    screening_candidate = candidate.model_copy(
        update={"stage": CampaignState.I2C_SCREENING}
    )

    with pytest.raises(AuthorizationError, match="candidate_index.*max_candidates"):
        validate_candidate(
            authorization,
            screening_candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=authorization.max_candidates,
            elapsed_s=1.0e100,
            **_context(
                authorization,
                now=authorization.issued_at + timedelta(days=36500),
            ),
        )


@pytest.mark.parametrize(
    ("requested_value", "elapsed_s", "message"),
    [
        pytest.param(
            72,
            float("inf"),
            "elapsed_s.*non-negative finite",
            id="elapsed",
        ),
        pytest.param(80, 0.0, "ibias_dac.*tuning range", id="parameter-envelope"),
    ],
)
def test_continuous_unbounded_candidate_fail_closed_parity(
    candidate, requested_value, elapsed_s, message
):
    authorization = _continuous_unbounded_authorization()
    screening_candidate = candidate.model_copy(
        update={
            "stage": CampaignState.I2C_SCREENING,
            "requested_i2c": {"ibias_dac": requested_value},
        }
    )

    with pytest.raises(AuthorizationError, match=message):
        validate_candidate(
            authorization,
            screening_candidate,
            previous_readback={"ibias_dac": 70},
            elapsed_s=elapsed_s,
            **_context(authorization),
        )


@pytest.mark.parametrize("candidate_index", [-1, 1.0, True])
def test_candidate_rejects_invalid_candidate_index(auth, candidate, candidate_index):
    with pytest.raises(AuthorizationError, match="candidate_index.*non-negative integer"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            candidate_index=candidate_index,
            **_context(auth),
        )


def test_candidate_accepts_elapsed_time_just_below_duration_budget(auth, candidate):
    assert (
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            elapsed_s=auth.max_duration_s - 0.001,
            **_context(auth),
        )
        is candidate
    )


def test_candidate_rejects_elapsed_time_at_duration_budget(auth, candidate):
    with pytest.raises(AuthorizationError, match="elapsed_s.*max_duration_s"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            elapsed_s=auth.max_duration_s,
            **_context(auth),
        )


@pytest.mark.parametrize("elapsed_s", [-0.001, float("inf"), True])
def test_candidate_rejects_invalid_elapsed_time(auth, candidate, elapsed_s):
    with pytest.raises(AuthorizationError, match="elapsed_s.*non-negative finite"):
        validate_candidate(
            auth,
            candidate,
            previous_readback={"ibias_dac": 70},
            elapsed_s=elapsed_s,
            **_context(auth),
        )
