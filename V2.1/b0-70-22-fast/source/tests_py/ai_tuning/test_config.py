from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vcsel_ai_tuning.config import (
    CalibrationManifest,
    CampaignConfig,
    load_yaml_model,
    validate_manifest_files,
)


def valid_campaign_dict():
    return {
        "campaign_id": "fake-skew-demo",
        "backend": {"kind": "fake", "dry_run": True},
        "optimizer": {"kind": "sobol", "budget": 8, "seed": 7},
        "parameters": [
            {
                "name": "bias_ma",
                "unit": "mA",
                "lower": 4.0,
                "upper": 7.0,
                "quantization": 0.1,
                "max_step": 1.0,
            },
            {
                "name": "skew_control_ui",
                "unit": "UI",
                "lower": -0.1,
                "upper": 0.1,
                "quantization": 0.001,
                "max_step": 0.05,
            },
        ],
        "safe_baseline": {"bias_ma": 5.0, "skew_control_ui": 0.05},
        "objective": {
            "primary_metric": "skew_pp_ui",
            "direction": "minimize",
            "metric_constraints": [
                {"metric": "rlm_proxy", "operator": ">=", "threshold": 0.9}
            ],
            "tie_breakers": [
                {"metric": "power_mw", "direction": "minimize"}
            ],
        },
    }


def dual_path_campaign_dict(kind="PATH_EXHAUSTIVE"):
    data = valid_campaign_dict()
    data["optimizer"] = {"kind": kind, "budget": 12, "seed": 1124}
    data["parameters"] = [
        {
            "name": "eq",
            "unit": "code",
            "lower": 0,
            "upper": 2,
            "grid_values": [0, 1, 2],
            "max_step": 1,
        },
        {
            "name": "ibias_dac",
            "unit": "code",
            "lower": 0,
            "upper": 2,
            "grid_values": [0, 1, 2],
            "max_step": 1,
        },
    ]
    data["safe_baseline"] = {"eq": 0, "ibias_dac": 0}
    data["objective"] = {
        "primary_metric": "tdecq_db",
        "direction": "minimize",
        "metric_constraints": [],
        "tie_breakers": [],
    }
    data["measurement_policy"] = "standard_every_candidate"
    data["tuning_identity"] = {
        "driver_profile_hash": "a" * 64,
        "vcsel_model": "APA4701",
        "device_id": "vcsel-001",
        "test_context_id": "53g-ssprq-25c",
    }
    return data


def test_campaign_accepts_exact_safe_baseline():
    campaign = CampaignConfig.model_validate(valid_campaign_dict())

    assert campaign.optimizer.budget == 8
    assert campaign.parameter_map["bias_ma"].unit == "mA"
    assert campaign.measurement_policy == "fast_proxy"
    assert campaign.tuning_identity is None


def test_campaign_rejects_live_hardware_in_milestone_one():
    data = valid_campaign_dict()
    data["backend"] = {"kind": "hardware", "dry_run": False}

    with pytest.raises(ValidationError, match="dry-run"):
        CampaignConfig.model_validate(data)


def test_campaign_rejects_missing_safe_baseline_parameter():
    data = valid_campaign_dict()
    del data["safe_baseline"]["bias_ma"]

    with pytest.raises(ValidationError, match="safe_baseline"):
        CampaignConfig.model_validate(data)


def test_yaml_loader_and_manifest_hash_validation(tmp_path: Path):
    source = tmp_path / "measurement.csv"
    source.write_text("bias_ma,power_mw\n5.0,1.0\n", encoding="utf-8")
    digest = sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "manifest_id": "manifest-1",
                "device_id": "vcsel-1",
                "temperature_c": 25.0,
                "sources": [{"path": source.name, "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_yaml_model(manifest_path, CalibrationManifest)

    validate_manifest_files(manifest, tmp_path)


@pytest.mark.parametrize(
    "objective",
    [
        pytest.param(
            {
                "primary_metric": "er_db",
                "direction": "minimize",
                "metric_constraints": [],
                "tie_breakers": [],
            },
            id="wrong-primary-metric",
        ),
        pytest.param(
            {
                "primary_metric": "tdecq_db",
                "direction": "maximize",
                "metric_constraints": [],
                "tie_breakers": [],
            },
            id="wrong-direction",
        ),
        pytest.param(
            {
                "primary_metric": "tdecq_db",
                "direction": "minimize",
                "metric_constraints": [
                    {"metric": "er_db", "operator": ">=", "threshold": 3.0}
                ],
                "tie_breakers": [],
            },
            id="metric-constraint",
        ),
        pytest.param(
            {
                "primary_metric": "tdecq_db",
                "direction": "minimize",
                "metric_constraints": [],
                "tie_breakers": [
                    {"metric": "er_db", "direction": "maximize"}
                ],
            },
            id="er-tie-breaker",
        ),
    ],
)
def test_dual_path_requires_tdecq_only_minimize_objective(objective):
    data = dual_path_campaign_dict("PATH_TDECQ_BAYESIAN")
    data["objective"] = objective

    with pytest.raises(ValidationError, match="TDECQ-only"):
        CampaignConfig.model_validate(data)


def test_dual_path_requires_standard_measurement_for_every_candidate():
    data = dual_path_campaign_dict()
    data["measurement_policy"] = "standard_best_only"

    with pytest.raises(ValidationError, match="standard_every_candidate"):
        CampaignConfig.model_validate(data)


def test_dual_path_requires_immutable_tuning_identity():
    data = dual_path_campaign_dict()
    data.pop("tuning_identity")

    with pytest.raises(ValidationError, match="tuning_identity"):
        CampaignConfig.model_validate(data)


@pytest.mark.parametrize("field", ["vcsel_model", "device_id", "test_context_id"])
@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_dual_path_tuning_identity_fields_must_be_nonblank(field, value):
    data = dual_path_campaign_dict()
    data["tuning_identity"][field] = value

    with pytest.raises(ValidationError):
        CampaignConfig.model_validate(data)


def test_dual_path_tuning_identity_fields_are_trimmed():
    data = dual_path_campaign_dict()
    data["tuning_identity"].update(
        {
            "vcsel_model": " APA4701 ",
            "device_id": " vcsel-001\t",
            "test_context_id": " 53g-ssprq-25c ",
        }
    )

    campaign = CampaignConfig.model_validate(data)

    assert campaign.tuning_identity is not None
    assert campaign.tuning_identity.vcsel_model == "APA4701"
    assert campaign.tuning_identity.device_id == "vcsel-001"
    assert campaign.tuning_identity.test_context_id == "53g-ssprq-25c"


@pytest.mark.parametrize(
    "grid_values",
    [
        pytest.param([], id="missing-grid"),
        pytest.param([0, 0.5, 2], id="fractional-grid"),
    ],
)
def test_dual_path_requires_nonempty_integer_prepared_grid(grid_values):
    data = dual_path_campaign_dict("PATH_TDECQ_BAYESIAN")
    data["parameters"][0]["grid_values"] = grid_values

    with pytest.raises(ValidationError, match="grid_values|integer grid values"):
        CampaignConfig.model_validate(data)


def test_dual_path_rejects_nonfinite_grid_values():
    data = dual_path_campaign_dict("PATH_TDECQ_BAYESIAN")
    data["parameters"][0]["grid_values"] = [0, float("nan"), 2]

    with pytest.raises(ValidationError):
        CampaignConfig.model_validate(data)


def test_exhaustive_budget_cannot_silently_truncate_the_grid():
    data = dual_path_campaign_dict("PATH_EXHAUSTIVE")
    data["optimizer"]["budget"] = 8

    with pytest.raises(ValidationError, match="exhaustive budget"):
        CampaignConfig.model_validate(data)


@pytest.mark.parametrize(
    ("kind", "budget"),
    [
        pytest.param("PATH_EXHAUSTIVE", 9, id="exhaustive-full-budget"),
        pytest.param("PATH_TDECQ_BAYESIAN", 4, id="bayesian-finite-grid"),
    ],
)
def test_dual_path_accepts_valid_finite_integer_grid(kind, budget):
    data = dual_path_campaign_dict(kind)
    data["optimizer"]["budget"] = budget

    campaign = CampaignConfig.model_validate(data)

    assert campaign.optimizer.kind == kind
    assert campaign.optimizer.budget == budget
    assert campaign.tuning_identity is not None
