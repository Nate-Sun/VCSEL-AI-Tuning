from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.runtime_config import (
    RuntimeConfig,
    RuntimeMode,
    load_runtime_config,
)


def _live_hardware() -> dict[str, object]:
    return {
        "driver_profile": "configs/ai_tuning/driver.yaml",
        "dca_visa_resource": "TCPIP0::dca.example::inst0::INSTR",
        "i2c_com_port": "COM7",
        "i2c_address": 0xA6,
        "bert_host": "192.0.2.10",
        "bert_port": 9000,
        "resource_ids": ["dca-1", "i2c-1", "bert-1"],
    }


def test_runtime_config_defaults_to_fake_without_hardware_endpoints(tmp_path: Path):
    config = RuntimeConfig(
        storage={
            "sqlite_path": tmp_path / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts",
        }
    )

    assert config.mode is RuntimeMode.FAKE
    assert config.api.host == "127.0.0.1"
    assert config.api.port == 8000
    assert config.worker_count == 2
    assert config.hardware is None


@pytest.mark.parametrize("mode", [RuntimeMode.READ_ONLY, RuntimeMode.LIVE_TUNING])
def test_hardware_modes_require_explicit_complete_endpoints(
    mode: RuntimeMode,
    tmp_path: Path,
):
    with pytest.raises(ValidationError, match="hardware configuration is required"):
        RuntimeConfig(
            mode=mode,
            storage={
                "sqlite_path": tmp_path / "campaigns.sqlite3",
                "artifact_root": tmp_path / "artifacts",
            },
        )


def test_hardware_modes_do_not_guess_missing_ports(tmp_path: Path):
    hardware = _live_hardware()
    del hardware["bert_port"]

    with pytest.raises(ValidationError, match="bert_port"):
        RuntimeConfig(
            mode=RuntimeMode.READ_ONLY,
            storage={
                "sqlite_path": tmp_path / "campaigns.sqlite3",
                "artifact_root": tmp_path / "artifacts",
            },
            hardware=hardware,
        )


def test_runtime_config_forbids_unknown_fields(tmp_path: Path):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig(
            storage={
                "sqlite_path": tmp_path / "campaigns.sqlite3",
                "artifact_root": tmp_path / "artifacts",
            },
            unsafe_autodetect=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_count", 0),
        ("worker_count", 33),
        ("measurement_timeout_s", 0.0),
    ],
)
def test_runtime_config_rejects_invalid_runtime_bounds(
    field: str,
    value: object,
    tmp_path: Path,
):
    values = {
        "storage": {
            "sqlite_path": tmp_path / "campaigns.sqlite3",
            "artifact_root": tmp_path / "artifacts",
        },
        field: value,
    }

    with pytest.raises(ValidationError):
        RuntimeConfig(**values)


def test_runtime_config_rejects_invalid_hardware_address(tmp_path: Path):
    hardware = _live_hardware()
    hardware["i2c_address"] = 256

    with pytest.raises(ValidationError, match="i2c_address"):
        RuntimeConfig(
            mode=RuntimeMode.LIVE_TUNING,
            storage={
                "sqlite_path": tmp_path / "campaigns.sqlite3",
                "artifact_root": tmp_path / "artifacts",
            },
            hardware=hardware,
        )


def test_load_runtime_config_is_strict_and_resolves_relative_paths(tmp_path: Path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: FAKE",
                "api:",
                "  host: 127.0.0.1",
                "  port: 8765",
                "storage:",
                "  sqlite_path: state/campaigns.sqlite3",
                "  artifact_root: artifacts/runs",
                "b0_path: configs/b0.yaml",
                "measurement_timeout_s: 15.0",
                "worker_count: 1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.mode is RuntimeMode.FAKE
    assert config.storage.sqlite_path == tmp_path / "state/campaigns.sqlite3"
    assert config.storage.artifact_root == tmp_path / "artifacts/runs"
    assert config.b0_path == tmp_path / "configs/b0.yaml"


def test_load_runtime_config_normalizes_parent_segments(tmp_path: Path):
    config_dir = tmp_path / "configs" / "ai_tuning"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: LIVE_TUNING",
                "storage:",
                "  sqlite_path: ../../state/campaigns.sqlite3",
                "  artifact_root: ../../artifacts/campaigns",
                "b0_path: b0.yaml",
                "hardware:",
                "  driver_profile: ../hardware/live.yaml",
                "  dca_visa_resource: TCPIP0::dca::INSTR",
                "  i2c_com_port: COM7",
                "  i2c_address: 166",
                "  bert_host: 192.0.2.10",
                "  bert_port: 9000",
                "  resource_ids: [dca-1, i2c-1]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.storage.sqlite_path == tmp_path / "state/campaigns.sqlite3"
    assert config.storage.artifact_root == tmp_path / "artifacts/campaigns"
    assert config.b0_path == config_dir / "b0.yaml"
    assert config.hardware is not None
    assert config.hardware.driver_profile == tmp_path / "configs/hardware/live.yaml"
    assert ".." not in config.storage.artifact_root.parts


def test_load_read_only_config_accepts_only_explicit_complete_hardware(tmp_path: Path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: READ_ONLY",
                "storage:",
                "  sqlite_path: state/campaigns.sqlite3",
                "  artifact_root: artifacts/runs",
                "hardware:",
                "  driver_profile: configs/driver.yaml",
                "  dca_visa_resource: TCPIP0::dca.example::inst0::INSTR",
                "  i2c_com_port: COM7",
                "  i2c_address: 166",
                "  bert_host: 192.0.2.10",
                "  bert_port: 9000",
                "  resource_ids: [dca-1, i2c-1, bert-1]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.mode is RuntimeMode.READ_ONLY
    assert config.hardware is not None
    assert config.hardware.resource_ids == ("dca-1", "i2c-1", "bert-1")
    assert config.hardware.driver_profile == tmp_path / "configs/driver.yaml"


def test_canonical_b0_values_are_unchanged():
    baseline = load_b0()

    assert baseline.pattern == "SSPRQ"
    assert baseline.fir.taps == (0, 9, -28, 104, -8, 1, -3)
    assert (baseline.fir.upper_nlc, baseline.fir.lower_nlc) == (6, 6)
    assert tuple(baseline.i2c.model_dump().values()) == (
        70,
        15,
        15,
        13,
        6,
        0,
        5,
        15,
        22,
    )
