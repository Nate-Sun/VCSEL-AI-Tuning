from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from vcsel_ai_tuning.v1_1.authorization import (
    AuthorizationDriftError,
    AuthorizedParameter,
    FullCampaignAuthorization,
    OptimizationDepth,
    RollbackPolicy,
)
from vcsel_ai_tuning.v1_1.domain import DiscreteRange, TaskType

from .test_baseline import make_b0


ISSUED_AT = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)


def make_authorization(**updates) -> FullCampaignAuthorization:
    b0 = make_b0()
    values = {
        "authorization_id": "auth-001",
        "task_id": "task-001",
        "task_type": TaskType.AI_OPTIMIZATION,
        "b0_hash": b0.content_hash,
        "hardware_identities": {
            identity.kind: identity.identity for identity in b0.hardware_identities
        },
        "mapping_hash": "a" * 64,
        "channel_mapping": b0.channel_mapping_payload(),
        "dca_config_fingerprint": b0.dca_fingerprint.content_hash,
        "parameters": (
            AuthorizedParameter("ibias_dac", DiscreteRange(60, 80, 2)),
            AuthorizedParameter("eq", DiscreteRange(18, 26, 2)),
        ),
        "depth": OptimizationDepth.STANDARD,
        "candidate_budget": 30,
        "seed": 1234,
        "max_points": None,
        "rollback_policy": RollbackPolicy.RESTORE_B0,
        "issued_at": ISSUED_AT,
        "expires_at": ISSUED_AT + timedelta(hours=1),
    }
    values.update(updates)
    return FullCampaignAuthorization.freeze(**values)


def test_full_campaign_authorization_freezes_every_safety_input():
    authorization = make_authorization()

    assert authorization.authorization_id == "auth-001"
    assert authorization.task_id == "task-001"
    assert authorization.task_type is TaskType.AI_OPTIMIZATION
    assert authorization.depth is OptimizationDepth.STANDARD
    assert authorization.candidate_budget == 30
    assert authorization.seed == 1234
    assert authorization.max_points is None
    assert authorization.rollback_policy is RollbackPolicy.RESTORE_B0
    assert authorization.payload()["channel_mapping"] == {
        "i2c_channel": 0,
        "bert_port": "port1",
        "bert_channel": "0",
        "dca_tdecq_source": "FUNC1",
        "dca_outer_er_source": "CHAN1A",
    }
    assert authorization.parameters[0].value_range.values() == tuple(range(60, 81, 2))
    assert len(authorization.content_hash) == 64
    with pytest.raises(FrozenInstanceError):
        authorization.candidate_budget = 31


def test_authorization_rejects_any_start_time_drift_and_reports_fields():
    authorization = make_authorization()

    authorization.verify_current(
        b0_hash=authorization.b0_hash,
        hardware_identities=dict(authorization.hardware_identities),
        mapping_hash=authorization.mapping_hash,
        channel_mapping=dict(authorization.channel_mapping),
        dca_config_fingerprint=authorization.dca_config_fingerprint,
    )

    with pytest.raises(AuthorizationDriftError) as error:
        authorization.verify_current(
            b0_hash="b" * 64,
            hardware_identities={"i2c": "changed"},
            mapping_hash="c" * 64,
            channel_mapping={
                **dict(authorization.channel_mapping),
                "bert_channel": "1",
            },
            dca_config_fingerprint="d" * 64,
        )

    assert error.value.mismatches == (
        "b0_hash",
        "hardware_identities",
        "mapping_hash",
        "channel_mapping",
        "dca_config_fingerprint",
    )


def test_authorization_hash_changes_when_explicit_channel_mapping_changes():
    original = make_authorization()
    changed = make_authorization(
        channel_mapping={
            **dict(original.channel_mapping),
            "dca_tdecq_source": "FUNC2",
        }
    )

    assert changed.content_hash != original.content_hash


def test_authorization_rejects_invalid_limits_and_expiry():
    with pytest.raises(ValueError, match="candidate_budget"):
        make_authorization(candidate_budget=0)
    with pytest.raises(ValueError, match="expires_at"):
        make_authorization(expires_at=ISSUED_AT)
    with pytest.raises(ValueError, match="Advanced"):
        make_authorization(depth=OptimizationDepth.ADVANCED)


def test_advanced_depth_requires_explicit_iteration_limit():
    authorization = make_authorization(
        depth=OptimizationDepth.ADVANCED,
        advanced_iteration_limit=7,
    )

    assert authorization.iteration_limit == 7
    assert make_authorization(depth=OptimizationDepth.FAST).iteration_limit == 1
    assert make_authorization(depth=OptimizationDepth.STANDARD).iteration_limit == 3
    assert make_authorization(depth=OptimizationDepth.DEEP).iteration_limit == 5


def test_fresh_authorization_rejects_duplicate_parameter_names():
    duplicate = AuthorizedParameter("eq", DiscreteRange(18, 26, 2))

    with pytest.raises(ValueError, match="unique"):
        make_authorization(parameters=(duplicate, duplicate))


def test_persisted_authorization_rejects_duplicate_parameter_names():
    payload = make_authorization().payload()
    payload["parameters"].append(dict(payload["parameters"][0]))

    with pytest.raises(ValueError, match="unique"):
        FullCampaignAuthorization.from_payload(payload)
