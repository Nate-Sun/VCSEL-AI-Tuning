from pathlib import Path

import pytest
import yaml

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.hardware.live_runtime_factory import (
    build_continuous_live_tuning_profile,
    build_live_tuning_profile,
)
from vcsel_ai_tuning.runtime import RuntimeAssemblyError
from vcsel_ai_tuning.runtime_config import RuntimeMode, load_runtime_config


B0_PATH = Path("configs/ai_tuning/b0.yaml")
LIVE_PROFILE = Path("configs/hardware/vcsel_lab_live.yaml")
V21_LIVE_PROFILE = Path("configs/hardware/vcsel_lab_v2_1_continuous.yaml")
V21_RUNTIME = Path("configs/ai_tuning/runtime.v2_1.continuous.live.yaml")
V21_CONTINUOUS = Path("configs/ai_tuning/v2_1_continuous.live.yaml")


def write_live_profile(tmp_path, *, safety_limits):
    document = yaml.safe_load(LIVE_PROFILE.read_text(encoding="utf-8"))
    document["characterization"]["safety_limits"] = safety_limits
    path = tmp_path / "live-profile.yaml"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_live_profile_requires_at_least_one_explicit_safety_limit(tmp_path):
    profile = write_live_profile(tmp_path, safety_limits={})
    with pytest.raises(RuntimeAssemblyError, match="safety limit"):
        build_live_tuning_profile(profile, load_b0(B0_PATH))


def test_live_profile_builds_campaign_limits_from_operator_values(tmp_path):
    profile = write_live_profile(
        tmp_path,
        safety_limits={"eq": {"minimum": 18, "maximum": 25, "max_step": 3}},
    )
    tuning = build_live_tuning_profile(profile, load_b0(B0_PATH))
    spec = tuning.parameter_specs[0]
    assert (spec.name, spec.tuning_min, spec.tuning_max, spec.max_step) == (
        "eq",
        18,
        25,
        3,
    )


def test_v21_continuous_live_configuration_is_isolated_and_run_parameterized():
    tuning = build_continuous_live_tuning_profile(
        V21_LIVE_PROFILE,
        load_b0(B0_PATH),
    )
    actual = {
        spec.name: (
            spec.encoding_min,
            spec.encoding_max,
            spec.tuning_min,
            spec.tuning_max,
            spec.max_step,
        )
        for spec in tuning.parameter_specs
    }
    assert actual == {
        "ibias_dac": (0, 127, 0, 127, 20),
        "efin_cur": (0, 15, 0, 15, 3),
        "vga_cur": (0, 15, 0, 15, 3),
        "eq": (0, 31, 0, 31, 5),
        "res_gain": (0, 15, 0, 15, 3),
        "res_bw": (0, 15, 0, 15, 3),
        "gain_ctrl": (0, 63, 0, 63, 10),
        "vbias_ref": (0, 15, 0, 15, 3),
        "vref": (0, 15, 0, 15, 3),
    }
    assert tuning.parameter_space.candidate_count == 1

    runtime = load_runtime_config(V21_RUNTIME)
    assert runtime.mode is RuntimeMode.LIVE_TUNING
    assert runtime.hardware is not None
    assert runtime.hardware.driver_profile.resolve() == V21_LIVE_PROFILE.resolve()
    assert runtime.hardware.resource_ids == (
        "DCA-N1010A-CHAN1A-FUNC1",
        "I2C-VD1124-COM7-0xA6",
    )
    assert "v2.1" in str(runtime.storage.sqlite_path)

    continuous = yaml.safe_load(V21_CONTINUOUS.read_text(encoding="utf-8"))
    assert continuous["mode"] == "LIVE"
    assert "max_rounds" not in continuous
    assert "target_tdecq_db" not in continuous
    live = continuous["live"]
    assert "commission_max_duration_s" not in live
    assert live["campaign_id"] == "REPLACE_WITH_FRESH_CAMPAIGN_ID"
    assert live["runtime_config"] == str(V21_RUNTIME).replace("\\", "/")
    instruction = live["instruction"]
    assert "left raw PAM4 eye" in instruction
    assert "right TDECQ-equalized view only as supporting evidence" in instruction
    assert "target.tdecq_db_lt" in instruction
    assert "round.maximum" in instruction
    assert "exploratory round may be worse" in instruction
