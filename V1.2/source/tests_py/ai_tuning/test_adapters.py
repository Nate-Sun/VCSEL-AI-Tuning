import pytest

from vcsel_ai_tuning.adapters import DryRunInstrumentAdapter, InstrumentRole


@pytest.mark.parametrize(
    ("role", "fixture"),
    [
        (InstrumentRole.BERT, {"error_count": 0, "bit_count": 1_000_000}),
        (InstrumentRole.DRIVER, {"bias_ma": 5.0, "modulation_ma": 2.0}),
        (InstrumentRole.SCOPE_RX, {"skew_pp_ui": 0.04, "rlm": 0.96}),
    ],
)
def test_each_role_obeys_dry_run_contract(role, fixture):
    adapter = DryRunInstrumentAdapter(role=role, fixture=fixture)

    identity = adapter.identify()
    adapter.configure({"preset": "m1"})
    adapter.apply({"value": 1.0})
    readback = adapter.readback()
    measured = adapter.measure()
    adapter.abort()
    adapter.safe_off()
    adapter.safe_off()

    assert identity.vendor == "DRY-RUN"
    assert identity.role is role
    assert readback == {"value": 1.0}
    assert measured == fixture
    assert adapter.is_safe is True


def test_dry_run_adapter_has_no_live_mode_switch():
    with pytest.raises(TypeError):
        DryRunInstrumentAdapter(
            role=InstrumentRole.BERT,
            fixture={},
            live=True,
        )
