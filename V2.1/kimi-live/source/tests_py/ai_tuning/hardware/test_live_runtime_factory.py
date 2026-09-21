from pathlib import Path

import pytest
import yaml

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.hardware.live_runtime_factory import build_live_tuning_profile
from vcsel_ai_tuning.runtime import RuntimeAssemblyError


B0_PATH = Path("configs/ai_tuning/b0.yaml")
LIVE_PROFILE = Path("configs/hardware/vcsel_lab_live.yaml")


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
