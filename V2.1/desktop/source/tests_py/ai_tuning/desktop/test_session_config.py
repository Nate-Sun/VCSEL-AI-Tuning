from pathlib import Path

import yaml

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.desktop.models import (
    DEFAULT_B0,
    B0Input,
    BenchSettings,
    RunInputs,
)
from vcsel_ai_tuning.desktop.session_config import SessionConfigBuilder
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.runtime_config import load_runtime_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = REPOSITORY_ROOT / "configs" / "ai_tuning"
VALIDATED_RUNTIME = yaml.safe_load(
    (TEMPLATE_ROOT / "runtime.v2_1.continuous.live.yaml").read_text(
        encoding="utf-8"
    )
)
VALIDATED_HARDWARE = yaml.safe_load(
    (
        REPOSITORY_ROOT
        / "configs"
        / "hardware"
        / "vcsel_lab_v2_1_continuous.yaml"
    ).read_text(encoding="utf-8")
)

INPUTS = RunInputs(
    api_key="desktop-secret-must-not-be-written",
    max_rounds=12,
    target_tdecq_db=2.7,
    b0=B0Input(**(DEFAULT_B0.model_dump() | {"eq": 23, "res_bw": 2})),
    bench=BenchSettings(
        dca_visa_resource="TCPIP0::desktop-scope::hislip0,4880::INSTR",
        i2c_com_port="COM11",
        i2c_address=0xA8,
    ),
)


def _load(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_builder_changes_only_approved_b0_and_endpoint_fields(tmp_path) -> None:
    paths = SessionConfigBuilder(TEMPLATE_ROOT).build(
        tmp_path, INPUTS, "desktop-test"
    )
    b0 = _load(paths.b0)
    hardware_profile = _load(paths.hardware)
    runtime = _load(paths.runtime)

    assert b0["i2c"] == INPUTS.b0.model_dump()
    assert b0["fir"] == {
        "taps": [0, 9, -28, 104, -8, 1, -3],
        "upper_nlc": 6,
        "lower_nlc": 6,
    }
    assert b0["pattern"] == "SSPRQ"
    hardware = runtime["hardware"]
    assert isinstance(hardware, dict)
    assert hardware["dca_visa_resource"] == INPUTS.bench.dca_visa_resource
    assert hardware["i2c_com_port"] == INPUTS.bench.i2c_com_port
    assert hardware["i2c_address"] == INPUTS.bench.i2c_address
    assert hardware["bert_host"] == VALIDATED_RUNTIME["hardware"]["bert_host"]
    assert hardware["bert_port"] == VALIDATED_RUNTIME["hardware"]["bert_port"]
    assert hardware_profile["dca"]["resource"] == INPUTS.bench.dca_visa_resource
    assert hardware_profile["i2c"]["port"] == INPUTS.bench.i2c_com_port
    assert hardware_profile["i2c"]["device_address"] == INPUTS.bench.i2c_address
    assert hardware_profile["bert"] == VALIDATED_HARDWARE["bert"]
    assert hardware_profile["characterization"] == VALIDATED_HARDWARE["characterization"]


def test_builder_generates_isolated_absolute_paths_and_valid_documents(
    tmp_path,
) -> None:
    paths = SessionConfigBuilder(TEMPLATE_ROOT).build(
        tmp_path, INPUTS, "desktop-test"
    )
    runtime = load_runtime_config(paths.runtime)
    baseline = load_b0(paths.b0)

    assert paths.campaign_root == (tmp_path / "desktop-test").resolve()
    assert runtime.b0_path == paths.b0.resolve()
    assert runtime.storage.artifact_root == paths.engine_artifact_root.resolve()
    assert baseline.i2c.eq == 23
    assert runtime.hardware is not None
    assert runtime.hardware.driver_profile == paths.hardware.resolve()
    profile = load_live_hardware_config(runtime.hardware.driver_profile)
    assert profile.dca.resource == runtime.hardware.dca_visa_resource
    assert profile.i2c.port == runtime.hardware.i2c_com_port
    assert profile.i2c.device_address == runtime.hardware.i2c_address
    assert paths.controller_root.is_dir()
    assert paths.engine_artifact_root.is_dir()


def test_live_document_targets_generated_runtime_and_fresh_campaign(tmp_path) -> None:
    paths = SessionConfigBuilder(TEMPLATE_ROOT).build(
        tmp_path, INPUTS, "desktop-test"
    )
    live = _load(paths.live)

    assert live["result_dir"] == str(paths.controller_root)
    assert live["live"]["runtime_config"] == str(paths.runtime)
    assert live["live"]["campaign_id"] == "desktop-test"
    assert live["live"]["parameter_policy"]["version"] == "i2c-restricted-v1"


def test_generated_documents_never_contain_api_key(tmp_path) -> None:
    paths = SessionConfigBuilder(TEMPLATE_ROOT).build(
        tmp_path, INPUTS, "desktop-test"
    )
    combined = "".join(
        path.read_text(encoding="utf-8") for path in paths.config_files
    )

    assert "desktop-secret-must-not-be-written" not in combined
    assert "api_key" not in combined.lower()
    assert not list(paths.campaign_root.rglob("*.tmp"))


def test_campaign_id_rejects_paths_and_checked_in_placeholder(tmp_path) -> None:
    builder = SessionConfigBuilder(TEMPLATE_ROOT)

    for invalid in ("../escape", "nested/path", "REPLACE_WITH_FRESH_CAMPAIGN_ID"):
        try:
            builder.build(tmp_path, INPUTS, invalid)
        except ValueError as error:
            assert "campaign_id" in str(error)
        else:
            raise AssertionError(f"campaign_id {invalid!r} was accepted")
