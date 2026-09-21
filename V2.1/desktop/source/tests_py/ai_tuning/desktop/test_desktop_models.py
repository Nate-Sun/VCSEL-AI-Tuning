from pathlib import Path

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.desktop.models import (
    DEFAULT_B0,
    B0Input,
    BenchSettings,
    DesktopPhase,
    DesktopResult,
    PreflightSummary,
    ProgressSnapshot,
    RunInputs,
)


BENCH = BenchSettings(
    dca_visa_resource="TCPIP0::scope::hislip0,4880::INSTR",
    i2c_com_port="COM7",
    i2c_address=0xA6,
)


def _inputs(**overrides: object) -> RunInputs:
    values: dict[str, object] = {
        "api_key": "secret",
        "max_rounds": 50,
        "target_tdecq_db": 2.8,
        "b0": DEFAULT_B0,
        "bench": BENCH,
    }
    values.update(overrides)
    return RunInputs.model_validate(values)


def test_run_inputs_accept_exact_operator_surface() -> None:
    value = _inputs()

    assert value.max_rounds == 50
    assert value.api_key.get_secret_value() == "secret"
    assert value.b0.eq == 16
    assert value.bench.i2c_address == 0xA6


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"max_rounds": True}, "max_rounds"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"target_tdecq_db": float("nan")}, "target_tdecq_db"),
        ({"target_tdecq_db": 0.0}, "target_tdecq_db"),
        ({"api_key": ""}, "api_key"),
    ],
)
def test_run_inputs_reject_invalid_operator_values(
    overrides: dict[str, object], expected_field: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        _inputs(**overrides)

    assert expected_field in str(caught.value)


def test_b0_requires_all_nine_nonnegative_strict_integers() -> None:
    raw = dict(DEFAULT_B0.model_dump())
    del raw["eq"]
    with pytest.raises(ValidationError):
        B0Input.model_validate(raw)

    with pytest.raises(ValidationError):
        B0Input.model_validate(DEFAULT_B0.model_dump() | {"eq": 16.0})

    with pytest.raises(ValidationError):
        B0Input.model_validate(DEFAULT_B0.model_dump() | {"eq": -1})


def test_models_are_immutable_and_forbid_extra_fields() -> None:
    value = _inputs()
    with pytest.raises(ValidationError):
        RunInputs.model_validate(value.model_dump() | {"fir": [1, 2, 3]})
    with pytest.raises(ValidationError):
        value.model_copy(update={"unknown": True}, deep=True).model_validate(
            value.model_dump() | {"unknown": True}
        )
    with pytest.raises(ValidationError):
        BenchSettings.model_validate(BENCH.model_dump() | {"api_key": "leak"})


def test_status_contracts_contain_no_hardware_objects() -> None:
    preflight = PreflightSummary(
        campaign_id="desktop-001",
        dca_verified=True,
        i2c_readback_verified=True,
        kimi_proposal_valid=True,
        dut_write_performed=False,
        bert_contacted=False,
        details="ready",
    )
    progress = ProgressSnapshot(
        phase=DesktopPhase.RUNNING,
        current_round=2,
        max_rounds=10,
    )
    result = DesktopResult(
        campaign_id="desktop-001",
        phase=DesktopPhase.COMPLETED,
        campaign_root=Path("campaign"),
        b0_restored=True,
        message="completed",
    )

    assert preflight.approvable is True
    assert progress.current_round == 2
    assert result.b0_restored is True


def test_preflight_is_not_approvable_if_any_required_gate_fails() -> None:
    summary = PreflightSummary(
        campaign_id="desktop-001",
        dca_verified=True,
        i2c_readback_verified=True,
        kimi_proposal_valid=True,
        dut_write_performed=True,
        bert_contacted=False,
        details="unexpected write",
    )

    assert summary.approvable is False
