"""Generate one isolated, validated configuration set per desktop campaign."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import yaml

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
from vcsel_ai_tuning.runtime_config import load_runtime_config

from .models import RunInputs


_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_CHECKED_IN_PLACEHOLDER = "REPLACE_WITH_FRESH_CAMPAIGN_ID"


@dataclass(frozen=True, slots=True)
class SessionPaths:
    campaign_root: Path
    controller_root: Path
    engine_artifact_root: Path
    b0: Path
    hardware: Path
    runtime: Path
    live: Path

    @property
    def config_files(self) -> tuple[Path, ...]:
        return (self.b0, self.hardware, self.runtime, self.live)


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"validated template must contain a mapping: {path}")
    return value


def _atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class SessionConfigBuilder:
    def __init__(self, template_root: str | Path) -> None:
        self.template_root = Path(template_root).resolve()

    def build(
        self,
        output_root: str | Path,
        inputs: RunInputs,
        campaign_id: str,
    ) -> SessionPaths:
        if (
            not _CAMPAIGN_ID.fullmatch(campaign_id)
            or campaign_id == _CHECKED_IN_PLACEHOLDER
        ):
            raise ValueError("campaign_id must be a fresh path-safe identifier")

        campaign_root = (Path(output_root) / campaign_id).resolve()
        controller_root = campaign_root / "controller"
        engine_artifact_root = campaign_root / "engine"
        config_root = campaign_root / "config"
        for directory in (controller_root, engine_artifact_root, config_root):
            directory.mkdir(parents=True, exist_ok=True)

        paths = SessionPaths(
            campaign_root=campaign_root,
            controller_root=controller_root,
            engine_artifact_root=engine_artifact_root,
            b0=config_root / "b0.yaml",
            hardware=config_root / "hardware.yaml",
            runtime=config_root / "runtime.yaml",
            live=config_root / "live.yaml",
        )

        b0 = _load_mapping(self.template_root / "b0.v2_1.continuous.yaml")
        hardware = _load_mapping(
            self.template_root.parent
            / "hardware"
            / "vcsel_lab_v2_1_continuous.yaml"
        )
        runtime = _load_mapping(
            self.template_root / "runtime.v2_1.continuous.live.yaml"
        )
        live = _load_mapping(self.template_root / "v2_1_continuous.live.yaml")

        b0["i2c"] = inputs.b0.model_dump()

        hardware["dca"]["resource"] = inputs.bench.dca_visa_resource
        hardware["i2c"]["port"] = inputs.bench.i2c_com_port
        hardware["i2c"]["device_address"] = inputs.bench.i2c_address

        runtime["storage"]["sqlite_path"] = str(campaign_root / "state.sqlite3")
        runtime["storage"]["artifact_root"] = str(engine_artifact_root)
        runtime["b0_path"] = str(paths.b0)
        runtime_hardware = runtime["hardware"]
        runtime_hardware["driver_profile"] = str(paths.hardware)
        runtime_hardware["dca_visa_resource"] = inputs.bench.dca_visa_resource
        runtime_hardware["i2c_com_port"] = inputs.bench.i2c_com_port
        runtime_hardware["i2c_address"] = inputs.bench.i2c_address
        resource_ids = list(runtime_hardware["resource_ids"])
        resource_ids[-1] = (
            f"I2C-VD1124-{inputs.bench.i2c_com_port}-"
            f"0x{inputs.bench.i2c_address:02X}"
        )
        runtime_hardware["resource_ids"] = resource_ids

        live["result_dir"] = str(controller_root)
        live["live"]["runtime_config"] = str(paths.runtime)
        live["live"]["campaign_id"] = campaign_id

        for path, document in (
            (paths.b0, b0),
            (paths.hardware, hardware),
            (paths.runtime, runtime),
            (paths.live, live),
        ):
            _atomic_yaml(path, document)

        baseline = load_b0(paths.b0)
        profile = load_live_hardware_config(paths.hardware)
        loaded_runtime = load_runtime_config(paths.runtime)
        loaded_hardware = loaded_runtime.hardware
        if loaded_hardware is None:
            raise ValueError("generated runtime is missing hardware settings")
        endpoint_pairs = (
            (loaded_hardware.dca_visa_resource, profile.dca.resource),
            (loaded_hardware.i2c_com_port, profile.i2c.port),
            (loaded_hardware.i2c_address, profile.i2c.device_address),
            (loaded_hardware.bert_host, profile.bert.ip),
            (loaded_hardware.bert_port, profile.bert.udp_port),
        )
        if any(runtime_value != profile_value for runtime_value, profile_value in endpoint_pairs):
            raise ValueError("generated runtime and hardware endpoints differ")
        if baseline.i2c.model_dump() != inputs.b0.model_dump():
            raise ValueError("generated B0 did not preserve operator input")
        return paths
