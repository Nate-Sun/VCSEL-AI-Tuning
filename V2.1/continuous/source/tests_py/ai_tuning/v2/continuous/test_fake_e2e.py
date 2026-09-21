from __future__ import annotations

import pytest

from vcsel_ai_tuning.v2.continuous.factory import build_continuous_dependencies
from vcsel_ai_tuning.v2.continuous.fake import fake_scenario
from vcsel_ai_tuning.v2.continuous.models import ContinuousState, ExecutionMode


def test_fake_scenario_propagates_explicit_tdecq_target_and_defaults_to_two_db(
    tmp_path,
) -> None:
    *_, explicit_request = fake_scenario(
        tmp_path / "explicit",
        "happy",
        max_rounds=1,
        target_tdecq_db=1.75,
    )
    *_, default_request = fake_scenario(
        tmp_path / "default",
        "happy",
        max_rounds=1,
    )

    assert explicit_request.stopping_policy.target_tdecq_db == pytest.approx(1.75)
    assert default_request.stopping_policy.target_tdecq_db == pytest.approx(2.0)


@pytest.mark.parametrize(
    "scenario, expected_state, expected_reason",
    [
        ("happy", ContinuousState.COMPLETED, "MAX_ROUNDS_REACHED"),
        (
            "stop-acceptance",
            ContinuousState.COMPLETED,
            "PROVIDER_STOP_ACCEPTED",
        ),
        ("readback-mismatch", ContinuousState.FAILED_SAFE, "ROUND_FAILURE"),
        ("dca-failure", ContinuousState.FAILED_SAFE, "ROUND_FAILURE"),
        (
            "b0-recovery-failure",
            ContinuousState.RECOVERY_REQUIRED,
            "B0_RECOVERY_FAILED",
        ),
    ],
)
def test_fake_scenarios_are_deterministic_and_never_prove_live_loop(
    tmp_path,
    scenario,
    expected_state,
    expected_reason,
) -> None:
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.FAKE,
        result_dir=tmp_path / scenario,
        scenario=scenario,
        max_rounds=3,
    )

    result = dependencies.controller.start(dependencies.request)

    assert result.state is expected_state
    assert result.stop_reason == expected_reason
    assert result.execution_mode is ExecutionMode.FAKE
    assert result.closed_loop_proven is False
