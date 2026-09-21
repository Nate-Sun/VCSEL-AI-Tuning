"""Strict LIVE_TUNING assembly using the repository's existing hardware adapters."""

from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import os
from pathlib import Path
from typing import Any

from vcsel_ai_tuning.campaign.models import (
    BaselineConfig,
    ParameterSpec as CampaignParameterSpec,
)
from vcsel_ai_tuning.domain import ParameterSpec as OptimizerParameterSpec
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.runtime import (
    AdapterBundle,
    RuntimeAssemblyError,
    RuntimeTuningProfile,
    create_runtime,
)
from vcsel_ai_tuning.runtime_config import (
    RuntimeConfig,
    RuntimeMode,
    load_runtime_config,
)

from .characterization import CampaignDcaAdapter, CampaignI2cAdapter
from .continuous_eye import ScreenEvidenceDcaAdapter
from .flexdca import open_visa_session
from .live_config import I2cConfig, LiveHardwareConfig, load_live_hardware_config
from .live_workflow import CampaignBertAdapter
from .vd1124 import Vd1124ProbeResult, Vd1124RegisterWriter
from .vit_bert import VitBertClient


class _WriterIdentityProbe:
    """Read identity through the writer's already-open serial connection."""

    def __init__(self, writer: Vd1124RegisterWriter, config: I2cConfig) -> None:
        self._writer = writer
        self._config = config

    def probe(self) -> Vd1124ProbeResult:
        chip_id = self._writer.read_raw(self._config.chip_id_register)
        chip_revision = self._writer.read_raw(
            self._config.chip_revision_register
        )
        return Vd1124ProbeResult(
            port=self._config.port,
            chip_id=chip_id,
            chip_revision=chip_revision,
            matched=(
                chip_id == self._config.expected_chip_id
                and chip_revision == self._config.expected_chip_revision
            ),
        )


class _OwnedDcaAdapter:
    """Delegate DCA calls and release every jointly owned live resource."""

    def __init__(self, adapter: CampaignDcaAdapter, resources: ExitStack) -> None:
        self._adapter = adapter
        self._resources = resources
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._resources.close()


class _ForbiddenContinuousBertAdapter:
    """Satisfy the legacy orchestrator shape while rejecting every BERT call."""

    def preflight(self) -> dict[str, object]:
        raise RuntimeAssemblyError("continuous I2C mode forbids BERT preflight")

    def apply(self, pattern: str, fir: object) -> object:
        del pattern, fir
        raise RuntimeAssemblyError("continuous I2C mode forbids BERT apply")

    def readback(self) -> dict[str, object]:
        raise RuntimeAssemblyError("continuous I2C mode forbids BERT readback")

    def close(self) -> None:
        return None


def build_live_tuning_profile(
    profile_path: str | Path,
    baseline: BaselineConfig,
    *,
    continuous_i2c_only: bool = False,
) -> RuntimeTuningProfile:
    """Build tuning bounds only from explicitly configured safety limits."""
    if not isinstance(baseline, BaselineConfig):
        raise TypeError("baseline must be BaselineConfig")
    path = Path(profile_path)
    try:
        raw_profile = path.read_bytes()
        config = load_live_hardware_config(path)
    except (OSError, ValueError) as error:
        raise RuntimeAssemblyError(f"invalid live driver profile: {error}") from error

    approved = tuple(
        field
        for field in config.characterization.parameters
        if field.safety_min is not None
        and field.safety_max is not None
        and field.max_step is not None
    )
    if not approved:
        raise RuntimeAssemblyError(
            "live driver profile requires at least one explicit safety limit"
        )

    campaign_specs: list[CampaignParameterSpec] = []
    optimizer_specs: list[OptimizerParameterSpec] = []
    for field in approved:
        assert field.safety_min is not None
        assert field.safety_max is not None
        assert field.max_step is not None
        if field.safety_min == field.safety_max:
            raise RuntimeAssemblyError(
                f"safety limit for {field.name} must contain at least two values"
            )
        campaign_specs.append(
            CampaignParameterSpec(
                name=field.name,
                encoding_min=0,
                encoding_max=field.encoded_max,
                tuning_min=field.safety_min,
                tuning_max=field.safety_max,
                max_step=field.max_step,
                enabled=True,
            )
        )
        optimizer_grid = (
            (float(getattr(baseline.i2c, field.name)),)
            if continuous_i2c_only
            else tuple(
                float(value)
                for value in range(field.safety_min, field.safety_max + 1)
            )
        )
        optimizer_specs.append(
            OptimizerParameterSpec(
                name=field.name,
                unit="code",
                lower=float(field.safety_min),
                upper=float(field.safety_max),
                quantization=1.0,
                max_step=float(field.max_step),
                readback_tolerance=0.0,
                grid_values=optimizer_grid,
            )
        )

    profile_hash = sha256(raw_profile).hexdigest()
    return RuntimeTuningProfile(
        driver_profile_hash=profile_hash,
        vcsel_model="VD1124",
        test_context_id=f"{baseline.pattern.lower()}-{profile_hash}",
        parameter_specs=tuple(campaign_specs),
        parameter_space=IntegerParameterSpace(tuple(optimizer_specs)),
    )


def build_continuous_live_tuning_profile(
    profile_path: str | Path,
    baseline: BaselineConfig,
) -> RuntimeTuningProfile:
    """Keep full authorization bounds but avoid an unused 9-D optimizer grid."""

    return build_live_tuning_profile(
        profile_path,
        baseline,
        continuous_i2c_only=True,
    )


def build_live_adapters_from_config(
    config: LiveHardwareConfig,
    baseline: BaselineConfig,
    runtime_timeout_s: float,
    *,
    campaign_id: str,
    continuous_i2c_only: bool = False,
    artifact_directory: str | Path | None = None,
) -> AdapterBundle:
    """Open one live resource set and hand ownership to the adapter bundle."""
    if not isinstance(config, LiveHardwareConfig):
        raise TypeError("config must be LiveHardwareConfig")
    if not isinstance(baseline, BaselineConfig):
        raise TypeError("baseline must be BaselineConfig")
    measurement_timeout = config.characterization.measurement_timeout_s
    if runtime_timeout_s < measurement_timeout:
        raise RuntimeAssemblyError(
            "runtime measurement timeout is shorter than the live driver profile"
        )

    resources = ExitStack()
    try:
        session = resources.enter_context(
            open_visa_session(config.dca.resource, config.dca.timeout_ms)
        )
        writer = resources.enter_context(
            Vd1124RegisterWriter(
                port=config.i2c.port,
                baudrate=config.i2c.baudrate,
                address=config.i2c.device_address,
            )
        )
        campaign_dca = CampaignDcaAdapter(session=session, config=config)
        dca = (
            ScreenEvidenceDcaAdapter(
                campaign_dca,
                session=session,
                source=config.characterization.tdecq.source,
                artifact_prefix=campaign_id,
                artifact_directory=artifact_directory,
            )
            if continuous_i2c_only
            else campaign_dca
        )
        i2c = CampaignI2cAdapter(
            writer=writer,
            identity_probe=_WriterIdentityProbe(writer, config.i2c),
            config=config,
        )
        bert = (
            _ForbiddenContinuousBertAdapter()
            if continuous_i2c_only
            else CampaignBertAdapter(
                client=VitBertClient(
                    config.bert.ip,
                    config.bert.udp_port,
                    local_ip=config.bert.local_ip,
                    timeout_s=config.bert.timeout_s,
                ),
                port=config.bert.port,
                channel=config.bert.channel,
                fixed_pattern=baseline.pattern,
                upper_nlc=baseline.fir.upper_nlc,
                lower_nlc=baseline.fir.lower_nlc,
                expected_host=config.bert.ip,
                expected_udp_port=config.bert.udp_port,
            )
        )
        return AdapterBundle(
            dca=_OwnedDcaAdapter(dca, resources),
            i2c=i2c,
            bert=bert,
            measurement_timeout_s=measurement_timeout,
        )
    except Exception:
        resources.close()
        raise


def _require_matching_endpoints(
    runtime_config: RuntimeConfig,
    live_config: LiveHardwareConfig,
) -> None:
    hardware = runtime_config.hardware
    if hardware is None:
        raise RuntimeAssemblyError("LIVE_TUNING requires hardware configuration")
    mismatches = []
    comparisons = (
        ("DCA", hardware.dca_visa_resource, live_config.dca.resource),
        ("I2C port", hardware.i2c_com_port, live_config.i2c.port),
        ("I2C address", hardware.i2c_address, live_config.i2c.device_address),
        ("BERT host", hardware.bert_host, live_config.bert.ip),
        ("BERT port", hardware.bert_port, live_config.bert.udp_port),
    )
    for label, runtime_value, driver_value in comparisons:
        if runtime_value != driver_value:
            mismatches.append(label)
    if mismatches:
        raise RuntimeAssemblyError(
            "runtime and live driver endpoints differ: " + ", ".join(mismatches)
        )


def create_live_api_runtime_from_path(
    runtime_path: str | Path,
    *,
    continuous_i2c_only: bool = False,
) -> Any:
    """Assemble the existing live runtime from one explicit config path."""

    runtime_config = load_runtime_config(runtime_path)
    if runtime_config.mode is not RuntimeMode.LIVE_TUNING:
        raise RuntimeAssemblyError(
            "live runtime factory requires mode LIVE_TUNING"
        )
    assert runtime_config.hardware is not None
    live_config = load_live_hardware_config(
        runtime_config.hardware.driver_profile
    )
    _require_matching_endpoints(runtime_config, live_config)
    if (
        runtime_config.measurement_timeout_s
        < live_config.characterization.measurement_timeout_s
    ):
        raise RuntimeAssemblyError(
            "runtime measurement timeout is shorter than the live driver profile"
        )
    return create_runtime(
        runtime_config,
        continuous_i2c_only=continuous_i2c_only,
        hardware_adapter_factory=(
            lambda campaign_id, baseline, timeout: build_live_adapters_from_config(
                live_config,
                baseline,
                timeout,
                campaign_id=campaign_id,
                continuous_i2c_only=continuous_i2c_only,
                artifact_directory=(
                    runtime_config.storage.artifact_root
                    / campaign_id
                    / "eye_artifacts"
                ),
            )
        ),
        tuning_profile_factory=(
            build_continuous_live_tuning_profile
            if continuous_i2c_only
            else build_live_tuning_profile
        ),
    )


def create_live_api_runtime() -> Any:
    """Factory hook used by ``VCSEL_AI_CAMPAIGN_SERVICE_FACTORY``."""
    runtime_path = os.environ.get("VCSEL_AI_RUNTIME_CONFIG")
    if not runtime_path:
        raise RuntimeAssemblyError("VCSEL_AI_RUNTIME_CONFIG is not set")
    return create_live_api_runtime_from_path(runtime_path)
