"""Lifecycle-controlled assembly for the VCSEL AI tuning API.

This module is intentionally hardware-neutral at import time. Real driver
builders are resolved only by an explicitly selected hardware runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from functools import partial
from importlib.util import find_spec
import logging
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vcsel_ai_tuning.campaign.authorization import canonical_json_sha256
from vcsel_ai_tuning.campaign.models import (
    BaselineConfig,
    CampaignState,
    FirAuthorizationSpec,
    FirSettings,
    ParameterSpec,
    load_b0,
)
from vcsel_ai_tuning.campaign.orchestrator import (
    CampaignOrchestrator,
    MeasurementResult,
)
from vcsel_ai_tuning.campaign.repository import CampaignRepository
from vcsel_ai_tuning.campaign.store import CampaignStore
from vcsel_ai_tuning.domain import ParameterSpec as SpaceParameterSpec
from vcsel_ai_tuning.hardware_authority import (
    HardwareAuthority,
    HardwareAuthorityBusy,
)
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import OptimizationContext

from .runtime_config import RuntimeMode


logger = logging.getLogger(__name__)


def _v11_sqlite_path(platform_sqlite_path: Path) -> Path:
    candidate = (
        platform_sqlite_path.parent.parent / "v1.2" / "campaigns.sqlite3"
    )
    if candidate.resolve(strict=False) == platform_sqlite_path.resolve(strict=False):
        return platform_sqlite_path.with_name("v1_1_tasks.sqlite3")
    return candidate


class RuntimeAssemblyError(RuntimeError):
    """Base error for runtime dependency assembly failures."""

    code = "RUNTIME_ASSEMBLY_ERROR"


class OptimizerCapabilityError(RuntimeAssemblyError):
    code = "CAPABILITY_UNAVAILABLE"


class ReadOnlyModeError(RuntimeAssemblyError):
    code = "READ_ONLY_MODE"


class RuntimeClosedError(RuntimeAssemblyError):
    code = "RUNTIME_CLOSED"


class HardwareResourceBusyError(RuntimeAssemblyError):
    code = "HARDWARE_RESOURCE_BUSY"


class RuntimeOwnerLease:
    """OS-backed exclusive ownership of one runtime SQLite authority."""

    def __init__(self, sqlite_path: Path) -> None:
        self.path = sqlite_path.with_suffix(sqlite_path.suffix + ".runtime.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        self._handle = handle
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise HardwareResourceBusyError(
                "runtime store is already owned by another live process"
            ) from exc

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class OptimizerCapability(StrEnum):
    PATH_EXHAUSTIVE = "PATH_EXHAUSTIVE"
    PATH_TDECQ_BAYESIAN = "PATH_TDECQ_BAYESIAN"
    PATH_HIERARCHICAL = "PATH_HIERARCHICAL"


class RuntimeCapabilities(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
        populate_by_name=True,
    )

    runtime_mode: RuntimeMode
    hardware_connected: bool
    read_only: bool
    exhaustive_ready: bool
    tdecq_bayesian_ready: bool
    hierarchical_ready: bool
    fir_proposer_ready: bool
    live_tuning_enabled: bool
    authorization_modes: tuple[str, ...] = ("PER_STAGE",)
    full_campaign: bool = Field(default=False, serialization_alias="FULL_CAMPAIGN")

    @classmethod
    def for_mode(
        cls,
        mode: RuntimeMode,
        *,
        hierarchical_ready: bool,
        fir_proposer_ready: bool,
        hardware_connected: bool,
    ) -> "RuntimeCapabilities":
        return cls(
            runtime_mode=mode,
            hardware_connected=hardware_connected,
            read_only=mode is RuntimeMode.READ_ONLY,
            exhaustive_ready=True,
            tdecq_bayesian_ready=find_spec("sklearn") is not None,
            hierarchical_ready=hierarchical_ready,
            fir_proposer_ready=fir_proposer_ready,
            live_tuning_enabled=mode is RuntimeMode.LIVE_TUNING,
        )


class OptimizerFactory:
    """Build only the explicitly requested optimization capability."""

    def __init__(
        self,
        hierarchical_builder: Callable[..., object] | None = None,
    ) -> None:
        self._hierarchical_builder = hierarchical_builder

    @property
    def hierarchical_ready(self) -> bool:
        return self._hierarchical_builder is not None

    def create(
        self,
        path: OptimizerCapability | str,
        *,
        space: IntegerParameterSpace,
        context: OptimizationContext,
        seed: int = 0,
    ) -> object:
        try:
            capability = OptimizerCapability(path)
        except (TypeError, ValueError) as exc:
            raise OptimizerCapabilityError(
                f"unknown optimizer capability: {path!r}"
            ) from exc
        if capability is OptimizerCapability.PATH_HIERARCHICAL:
            if self._hierarchical_builder is None:
                raise OptimizerCapabilityError(
                    "PATH_HIERARCHICAL is not configured; no TDECQ-only fallback "
                    "is permitted"
                )
            return self._hierarchical_builder(
                space=space,
                context=context,
                seed=seed,
            )
        if capability is OptimizerCapability.PATH_EXHAUSTIVE:
            from vcsel_ai_tuning.optimization.exhaustive import ExhaustiveOptimizer

            return ExhaustiveOptimizer(space, context)
        if find_spec("sklearn") is None:
            raise OptimizerCapabilityError(
                "PATH_TDECQ_BAYESIAN requires the optimizer dependency"
            )
        from vcsel_ai_tuning.optimization.bayesian import TdecqBayesianOptimizer

        return TdecqBayesianOptimizer(space, context, seed=seed)


@dataclass(slots=True)
class _FakeHardwareState:
    baseline: BaselineConfig
    i2c: dict[str, int]
    pattern: str
    fir: FirSettings
    lock: RLock
    closed: bool = False


class FakeDcaAdapter:
    def __init__(self, state: _FakeHardwareState) -> None:
        self._state = state

    def preflight(self) -> Mapping[str, object]:
        return {
            "identity_verified": True,
            "configuration_verified": True,
            "adapter": "FAKE_DCA",
        }

    def measure_single(self) -> MeasurementResult:
        with self._state.lock:
            if self._state.closed:
                raise RuntimeAssemblyError("fake hardware is closed")
            baseline_i2c = self._state.baseline.i2c.model_dump()
            i2c_distance = sum(
                abs(value - baseline_i2c[name])
                for name, value in self._state.i2c.items()
            )
            fir_distance = sum(
                abs(value - baseline)
                for value, baseline in zip(
                    self._state.fir.taps,
                    self._state.baseline.fir.taps,
                    strict=True,
                )
            )
            tdecq_db = 2.5 + 0.01 * i2c_distance + 0.005 * fir_distance
            er_db = max(0.0, 3.2 - 0.005 * i2c_distance)
            return MeasurementResult(
                acquisition_valid=True,
                er_db=er_db,
                er_status="CORR",
                tdecq_db=tdecq_db,
                tdecq_status="CORR",
                tdecq_invalid_reason=None,
                elapsed_s=0.01,
                raw_responses={
                    "source": "deterministic-fake",
                    "i2c_distance": i2c_distance,
                    "fir_distance": fir_distance,
                },
            )

    def close(self) -> None:
        return None


class FakeI2cAdapter:
    def __init__(self, state: _FakeHardwareState) -> None:
        self._state = state

    def preflight(self) -> Mapping[str, object]:
        return {
            "identity_verified": True,
            "configuration_verified": True,
            "adapter": "FAKE_I2C",
        }

    def apply(self, requested: Mapping[str, int]) -> Mapping[str, int]:
        with self._state.lock:
            if self._state.closed:
                raise RuntimeAssemblyError("fake hardware is closed")
            unknown = set(requested) - set(self._state.i2c)
            if unknown:
                raise RuntimeAssemblyError(
                    f"unknown fake I2C parameters: {sorted(unknown)}"
                )
            if any(type(value) is not int for value in requested.values()):
                raise RuntimeAssemblyError("fake I2C values must be integers")
            self._state.i2c.update(requested)
            return {
                name: self._state.i2c[name]
                for name in requested
            }

    def readback(self) -> Mapping[str, int]:
        with self._state.lock:
            if self._state.closed:
                raise RuntimeAssemblyError("fake hardware is closed")
            return dict(self._state.i2c)

    def close(self) -> None:
        self._state.closed = True


class FakeBertAdapter:
    def __init__(self, state: _FakeHardwareState) -> None:
        self._state = state

    def preflight(self) -> Mapping[str, object]:
        return {
            "identity_verified": True,
            "configuration_verified": True,
            "adapter": "FAKE_BERT",
        }

    def apply(self, pattern: str, fir: FirSettings) -> FirSettings:
        if pattern != "SSPRQ":
            raise RuntimeAssemblyError("fake BERT only permits SSPRQ")
        if not isinstance(fir, FirSettings):
            raise RuntimeAssemblyError("fake BERT FIR must be FirSettings")
        with self._state.lock:
            if self._state.closed:
                raise RuntimeAssemblyError("fake hardware is closed")
            self._state.pattern = pattern
            self._state.fir = fir
            return fir

    def readback(self) -> Mapping[str, object]:
        with self._state.lock:
            if self._state.closed:
                raise RuntimeAssemblyError("fake hardware is closed")
            return {"pattern": self._state.pattern, "fir": self._state.fir}

    def close(self) -> None:
        self._state.closed = True


@dataclass(frozen=True, slots=True)
class AdapterBundle:
    dca: Any
    i2c: Any
    bert: Any
    measurement_timeout_s: float | None = None

    def close(self) -> None:
        seen: set[int] = set()
        for adapter in (self.dca, self.i2c, self.bert):
            if id(adapter) in seen:
                continue
            seen.add(id(adapter))
            close = getattr(adapter, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True, slots=True)
class RuntimeTuningProfile:
    """Explicit human-approved bounds and identity for a non-FAKE runtime."""

    driver_profile_hash: str
    vcsel_model: str
    test_context_id: str
    parameter_specs: tuple[ParameterSpec, ...]
    parameter_space: IntegerParameterSpace
    fir_authorization_spec: FirAuthorizationSpec | None = None

    def __post_init__(self) -> None:
        if len(self.driver_profile_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.driver_profile_hash
        ):
            raise ValueError("driver_profile_hash must be lowercase SHA-256")
        if not self.vcsel_model or not self.test_context_id:
            raise ValueError("profile identity fields must be non-empty")
        authorization_names = tuple(
            sorted(spec.name for spec in self.parameter_specs)
        )
        if not authorization_names:
            raise ValueError("profile requires human-approved parameter specs")
        if authorization_names != self.parameter_space.names:
            raise ValueError(
                "authorization specs and optimizer parameter space must match"
            )


@dataclass(slots=True)
class CampaignEntry:
    campaign_id: str
    repository: Any
    adapters: AdapterBundle
    orchestrator: Any
    resource_ids: tuple[str, ...] = ()
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close_repository = getattr(self.repository, "close", None)
            if callable(close_repository):
                close_repository()
        finally:
            self.adapters.close()


class HardwareResourceRegistry:
    """Non-blocking, process-local ownership fence for named hardware."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._owners: dict[str, str] = {}

    @contextmanager
    def acquire(
        self,
        campaign_id: str,
        resource_ids: tuple[str, ...],
    ) -> Iterator[None]:
        if not isinstance(campaign_id, str) or not campaign_id:
            raise RuntimeAssemblyError("campaign_id must be a non-empty string")
        if len(resource_ids) != len(set(resource_ids)):
            raise RuntimeAssemblyError("hardware resource_ids must be unique")
        ordered = tuple(sorted(resource_ids))
        with self._lock:
            busy = [
                resource_id
                for resource_id in ordered
                if resource_id in self._owners
            ]
            if busy:
                owners = ", ".join(
                    f"{resource_id} ({self._owners[resource_id]})"
                    for resource_id in busy
                )
                raise HardwareResourceBusyError(
                    f"hardware resource is already leased: {owners}"
                )
            for resource_id in ordered:
                self._owners[resource_id] = campaign_id
        try:
            yield
        finally:
            with self._lock:
                for resource_id in ordered:
                    if self._owners.get(resource_id) == campaign_id:
                        del self._owners[resource_id]


class CampaignRuntime:
    """Own campaign dependency entries, bounded workers, recovery, and shutdown."""

    def __init__(
        self,
        config: Any,
        *,
        store: CampaignStore | None = None,
        entry_factory: Callable[[str], CampaignEntry] | None = None,
        resource_registry: HardwareResourceRegistry | None = None,
        capabilities: RuntimeCapabilities | None = None,
        owner_lease: RuntimeOwnerLease | None = None,
        external_active_task_id: Callable[[], str | None] | None = None,
        hardware_authority: HardwareAuthority | None = None,
    ) -> None:
        from .runtime_config import RuntimeConfig

        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be RuntimeConfig")
        self.config = config
        self._owner_lease = owner_lease or RuntimeOwnerLease(
            config.storage.sqlite_path
        )
        try:
            self.store = store or CampaignStore(config.storage.sqlite_path)
        except Exception:
            self._owner_lease.close()
            raise
        self._entry_factory = entry_factory
        self._resource_registry = resource_registry or HardwareResourceRegistry()
        self._hardware_authority = hardware_authority or HardwareAuthority()
        self.capabilities = capabilities or RuntimeCapabilities.for_mode(
            config.mode,
            hierarchical_ready=False,
            fir_proposer_ready=False,
            hardware_connected=False,
        )
        self._entries: dict[str, CampaignEntry] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=config.worker_count,
            thread_name_prefix="vcsel-campaign",
        )
        self._closed = False
        self._external_active_task_id = external_active_task_id or (lambda: None)
        self.baseline_hash: str | None = None

    @property
    def worker_count(self) -> int:
        return self.config.worker_count

    def entry(self, campaign_id: str) -> CampaignEntry:
        with self._lock:
            self._ensure_open()
            existing = self._entries.get(campaign_id)
            if existing is not None:
                return existing
            self.store.get_campaign(campaign_id)
            if self._entry_factory is None:
                raise RuntimeAssemblyError(
                    "campaign dependency entry factory is not configured"
                )
            created = self._entry_factory(campaign_id)
            if not isinstance(created, CampaignEntry):
                raise TypeError("entry_factory must return CampaignEntry")
            if created.campaign_id != campaign_id:
                created.close()
                raise RuntimeAssemblyError(
                    "entry_factory returned a different campaign_id"
                )
            self._entries[campaign_id] = created
            return created

    def recover(self) -> dict[str, CampaignState]:
        with self._lock:
            self._ensure_open()
        legacy = self.store.legacy_campaign_ids()
        if legacy:
            raise RuntimeAssemblyError(
                "legacy campaign state lacks immutable runtime metadata; "
                "manual migration or recovery is required before startup: "
                + ", ".join(legacy)
            )
        recovered: dict[str, CampaignState] = {}
        offset = 0
        while True:
            campaigns = self.store.list_campaigns(limit=100, offset=offset)
            if not campaigns:
                break
            for campaign in campaigns:
                recovered[campaign.campaign_id] = self.store.recover_state(
                    campaign.campaign_id
                )
            offset += len(campaigns)
        return dict(sorted(recovered.items()))

    def submit_stage(
        self,
        campaign_id: str,
        operation: str,
        work: Callable[[CampaignEntry], Any],
    ) -> Future[Any]:
        with self._lock:
            self._ensure_open()
        external = self._external_active_task_id()
        if external is not None:
            raise HardwareResourceBusyError(
                f"V1.1 hardware task is active: {external}"
            )
        if not isinstance(operation, str) or not operation:
            raise RuntimeAssemblyError("operation must be a non-empty string")
        if not callable(work):
            raise TypeError("work must be callable")
        entry = self.entry(campaign_id)

        def run() -> Any:
            self._append_worker_event(
                campaign_id,
                "WORKER_STARTED",
                {"operation": operation},
            )
            try:
                owner = ":".join(
                    (
                        "v1",
                        campaign_id,
                        operation,
                        ",".join(entry.resource_ids),
                    )
                )
                try:
                    authority_lease = self._hardware_authority.acquire(owner)
                except HardwareAuthorityBusy as error:
                    raise HardwareResourceBusyError(
                        f"hardware authority is already leased: {error.current_owner}"
                    ) from error
                with authority_lease:
                    with self._resource_registry.acquire(
                        campaign_id,
                        entry.resource_ids,
                    ):
                        result = work(entry)
            except Exception as exc:
                self._append_worker_event(
                    campaign_id,
                    "WORKER_FAILED",
                    {
                        "error_type": type(exc).__name__,
                        "operation": operation,
                    },
                )
                raise
            self._append_worker_event(
                campaign_id,
                "WORKER_COMPLETED",
                {"operation": operation},
            )
            return result

        return self._executor.submit(run)

    def health(self) -> Mapping[str, object]:
        with self._lock:
            self._ensure_open()
        store_ok = True
        try:
            self.store.list_campaigns(limit=1, offset=0)
        except Exception:
            store_ok = False
        return {
            "status": "ok" if store_ok else "degraded",
            "runtime_mode": self.config.mode.value,
            "store": {"ok": store_ok},
            "repository": {"ok": not self._closed},
        }

    def capabilities_payload(self) -> Mapping[str, object]:
        return self.capabilities.model_dump(mode="json", by_alias=True)

    def _append_worker_event(
        self,
        campaign_id: str,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        state = self.store.current_state(campaign_id)
        if state is None:
            raise RuntimeAssemblyError(f"unknown campaign: {campaign_id}")
        self.store.append_event(
            campaign_id,
            event_type,
            payload=dict(payload),
            expected_state=state,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            entries = tuple(self._entries.values())
            self._entries.clear()
        try:
            for entry in entries:
                entry.close()
        finally:
            try:
                self.store.close()
            finally:
                self._owner_lease.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("campaign runtime is closed")


def build_fake_adapters(
    baseline: BaselineConfig,
    *,
    measurement_timeout_s: float | None = None,
) -> AdapterBundle:
    state = _FakeHardwareState(
        baseline=baseline,
        i2c=baseline.i2c.model_dump(),
        pattern=baseline.pattern,
        fir=baseline.fir,
        lock=RLock(),
    )
    return AdapterBundle(
        dca=FakeDcaAdapter(state),
        i2c=FakeI2cAdapter(state),
        bert=FakeBertAdapter(state),
        measurement_timeout_s=measurement_timeout_s,
    )


class _ReadOnlyI2cAdapter:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def preflight(self) -> Mapping[str, object]:
        return self._adapter.preflight()

    def readback(self) -> Mapping[str, int]:
        return self._adapter.readback()

    def apply(self, _requested: Mapping[str, int]) -> Mapping[str, int]:
        raise ReadOnlyModeError("READ_ONLY mode rejects I2C writes")

    def close(self) -> None:
        close = getattr(self._adapter, "close", None)
        if callable(close):
            close()


class _ReadOnlyBertAdapter:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def preflight(self) -> Mapping[str, object]:
        return self._adapter.preflight()

    def readback(self) -> Mapping[str, object]:
        return self._adapter.readback()

    def apply(self, _pattern: str, _fir: FirSettings) -> FirSettings:
        raise ReadOnlyModeError("READ_ONLY mode rejects FIR writes")

    def close(self) -> None:
        close = getattr(self._adapter, "close", None)
        if callable(close):
            close()


def read_only_adapters(adapters: AdapterBundle) -> AdapterBundle:
    return AdapterBundle(
        dca=adapters.dca,
        i2c=_ReadOnlyI2cAdapter(adapters.i2c),
        bert=_ReadOnlyBertAdapter(adapters.bert),
        measurement_timeout_s=adapters.measurement_timeout_s,
    )


class FakeFirProposer:
    """Deterministic FIR proposer used only by the built-in FAKE runtime."""

    def propose(self, current_fir: FirSettings) -> FirSettings:
        if not isinstance(current_fir, FirSettings):
            raise TypeError("current_fir must be FirSettings")
        taps = list(current_fir.taps)
        if taps[3] <= 1:
            raise RuntimeAssemblyError("fake FIR proposer requires main tap above 1")
        taps[0] += 1
        taps[3] -= 1
        return FirSettings(
            taps=tuple(taps),
            upper_nlc=current_fir.upper_nlc,
            lower_nlc=current_fir.lower_nlc,
        )


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


def _selected_optimizer_path(store: CampaignStore, campaign_id: str) -> str:
    path = store.get_campaign_tuning_path(campaign_id)
    if path is None:
        raise RuntimeAssemblyError(
            f"campaign {campaign_id} has no persisted optimizer path"
        )
    return path


def _built_in_fake_tuning_profile(
    baseline: BaselineConfig,
) -> RuntimeTuningProfile:
    authorization_specs = (
        ParameterSpec(
            name="eq",
            encoding_min=0,
            encoding_max=31,
            tuning_min=21,
            tuning_max=22,
            max_step=1,
        ),
    )
    parameter_space = IntegerParameterSpace(
        (
            SpaceParameterSpec(
                name="eq",
                unit="code",
                lower=21,
                upper=22,
                quantization=1,
                max_step=1,
                grid_values=(21, 22),
            ),
        )
    )
    fir_min = tuple(value - 4 for value in baseline.fir.taps)
    fir_max = tuple(value + 4 for value in baseline.fir.taps)
    fir_min = (*fir_min[:3], max(1, fir_min[3]), *fir_min[4:])
    fir_authorization_spec = FirAuthorizationSpec(
        tap_min=fir_min,
        tap_max=fir_max,
        max_step=(2, 2, 2, 2, 2, 2, 2),
        max_total_change_l1=8,
        fixed_upper_nlc=baseline.fir.upper_nlc,
        fixed_lower_nlc=baseline.fir.lower_nlc,
    )
    return RuntimeTuningProfile(
        driver_profile_hash="0" * 64,
        vcsel_model="fake-vcsel",
        test_context_id="built-in-fake-context",
        parameter_specs=authorization_specs,
        parameter_space=parameter_space,
        fir_authorization_spec=fir_authorization_spec,
    )


def create_runtime(
    config: Any,
    *,
    hardware_adapter_factory: Callable[[str, BaselineConfig, float], AdapterBundle]
    | None = None,
    tuning_profile_factory: Callable[
        [Path, BaselineConfig], RuntimeTuningProfile
    ]
    | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    fir_proposer: Any = None,
    v11_gateway: Any = None,
    v11_runner: Any = None,
    v11_history_hardware: Any = None,
    v11_read_only_driver_factory: Any = None,
    v11_live_driver_factory: Any = None,
    v11_sweep_executor_factory: Any = None,
) -> Any:
    """Assemble an API runtime without connecting hardware during startup."""

    from vcsel_ai_tuning.api.app import ApiRuntime
    from vcsel_ai_tuning.campaign.service import ConcreteCampaignService
    from .runtime_config import RuntimeConfig

    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    if (
        config.mode is RuntimeMode.LIVE_TUNING
        and hardware_adapter_factory is None
        and tuning_profile_factory is not None
    ):
        raise RuntimeAssemblyError(
            "LIVE_TUNING requires an explicit hardware adapter factory when "
            "legacy V1 campaigns are enabled"
        )
    if (
        config.mode is RuntimeMode.LIVE_TUNING
        and hardware_adapter_factory is not None
        and tuning_profile_factory is None
    ):
        raise RuntimeAssemblyError(
            "LIVE_TUNING requires an explicit human-approved tuning profile "
            "factory when legacy V1 campaigns are enabled"
        )
    if config.mode is RuntimeMode.READ_ONLY and (
        (hardware_adapter_factory is None) != (tuning_profile_factory is None)
    ):
        raise RuntimeAssemblyError(
            "READ_ONLY legacy campaign hardware adapter and tuning profile "
            "factories must be configured together"
        )
    legacy_campaign_enabled = (
        hardware_adapter_factory is not None
        and tuning_profile_factory is not None
    )
    if (
        config.mode is not RuntimeMode.READ_ONLY
        and v11_read_only_driver_factory is not None
    ):
        raise RuntimeAssemblyError(
            "v11_read_only_driver_factory is only valid in READ_ONLY mode"
        )
    if (
        config.mode is not RuntimeMode.LIVE_TUNING
        and v11_sweep_executor_factory is not None
    ):
        raise RuntimeAssemblyError(
            "v11_sweep_executor_factory is only valid in LIVE_TUNING mode"
        )
    if (
        config.mode is not RuntimeMode.LIVE_TUNING
        and v11_live_driver_factory is not None
    ):
        raise RuntimeAssemblyError(
            "v11_live_driver_factory is only valid in LIVE_TUNING mode"
        )
    if v11_sweep_executor_factory is not None and v11_runner is not None:
        raise RuntimeAssemblyError(
            "v11_sweep_executor_factory cannot be combined with a custom "
            "V1.1 runner"
        )
    if config.mode is RuntimeMode.READ_ONLY and v11_history_hardware is not None:
        raise RuntimeAssemblyError(
            "READ_ONLY does not permit History Apply hardware assembly"
        )
    if config.mode is RuntimeMode.READ_ONLY and (
        v11_gateway is not None or v11_runner is not None
    ):
        raise RuntimeAssemblyError(
            "READ_ONLY requires the built-in V1.1 gateway and runner"
        )
    if v11_read_only_driver_factory is not None and (
        v11_gateway is not None or v11_runner is not None
    ):
        raise RuntimeAssemblyError(
            "v11_read_only_driver_factory cannot be combined with custom "
            "V1.1 components"
        )
    if v11_live_driver_factory is not None and v11_gateway is not None:
        raise RuntimeAssemblyError(
            "v11_live_driver_factory cannot be combined with a custom "
            "V1.1 gateway"
        )

    read_only_profile = None
    if config.mode is RuntimeMode.READ_ONLY:
        import yaml

        assert config.hardware is not None
        from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
        from vcsel_ai_tuning.v1_1.read_only_runtime import (
            ReadOnlyPreflightError,
            validate_read_only_profile,
            validate_profile_endpoints,
        )

        try:
            read_only_profile = load_live_hardware_config(
                config.hardware.driver_profile
            )
            validate_read_only_profile(read_only_profile)
            validate_profile_endpoints(
                read_only_profile,
                dca_visa_resource=config.hardware.dca_visa_resource,
                i2c_com_port=config.hardware.i2c_com_port,
                i2c_address=config.hardware.i2c_address,
                bert_host=config.hardware.bert_host,
                bert_port=config.hardware.bert_port,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
            ReadOnlyPreflightError,
        ) as error:
            raise RuntimeAssemblyError(
                f"READ_ONLY hardware profile validation failed: {error}"
            ) from error
    live_profile = None
    needs_builtin_live_gateway = (
        config.mode is RuntimeMode.LIVE_TUNING and v11_gateway is None
    )
    needs_builtin_live_sweep = (
        config.mode is RuntimeMode.LIVE_TUNING
        and v11_runner is None
        and v11_sweep_executor_factory is None
        and config.live_ai_profile is None
    )
    needs_builtin_live_ai = (
        config.mode is RuntimeMode.LIVE_TUNING
        and v11_runner is None
        and v11_sweep_executor_factory is None
        and config.live_ai_profile is not None
    )
    if needs_builtin_live_gateway or needs_builtin_live_sweep or needs_builtin_live_ai:
        import yaml

        assert config.hardware is not None
        from vcsel_ai_tuning.hardware.live_config import load_live_hardware_config
        from vcsel_ai_tuning.v1_1.read_only_runtime import (
            ReadOnlyPreflightError,
            validate_read_only_profile,
            validate_profile_endpoints,
        )

        try:
            live_profile = load_live_hardware_config(
                config.hardware.driver_profile
            )
            validate_read_only_profile(live_profile)
            validate_profile_endpoints(
                live_profile,
                dca_visa_resource=config.hardware.dca_visa_resource,
                i2c_com_port=config.hardware.i2c_com_port,
                i2c_address=config.hardware.i2c_address,
                bert_host=config.hardware.bert_host,
                bert_port=config.hardware.bert_port,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
            ReadOnlyPreflightError,
        ) as error:
            raise RuntimeAssemblyError(
                f"LIVE_TUNING hardware profile validation failed: {error}"
            ) from error
    if needs_builtin_live_sweep:
        assert live_profile is not None
        live_sweep_factory = None

        def create_live_sweep_executor(b0: Any, plan: Any) -> Any:
            nonlocal live_sweep_factory
            if live_sweep_factory is None:
                # Importing the production connection factory is deferred until
                # an authorized task actually requests its executor.
                from vcsel_ai_tuning.v1_1.live_sweep import (
                    build_live_sweep_executor_factory,
                )

                live_sweep_factory = build_live_sweep_executor_factory(
                    live_profile,
                    measurement_timeout_s=config.measurement_timeout_s,
                    artifact_root=(
                        config.storage.artifact_root.parent / "v1.2"
                    ),
                )
            return live_sweep_factory(b0, plan)

        v11_sweep_executor_factory = create_live_sweep_executor
    v11_ai_execution_factory = None
    if needs_builtin_live_ai:
        assert live_profile is not None
        live_ai_factory = None

        def create_live_ai_execution(task_id: str, b0: Any, payload: Any) -> Any:
            nonlocal live_ai_factory
            if live_ai_factory is None:
                from vcsel_ai_tuning.v1_1.live_ai_execution import (
                    build_live_ai_execution_factory,
                )

                live_ai_factory = build_live_ai_execution_factory(
                    live_profile,
                    database_path=_v11_sqlite_path(config.storage.sqlite_path),
                    artifact_root=config.storage.artifact_root.parent / "v1.2",
                    measurement_timeout_s=config.measurement_timeout_s,
                    cleanup_grace_s=config.cleanup_grace_s,
                    live_ai_profile=config.live_ai_profile,
                )
            return live_ai_factory(task_id, b0, payload)

        v11_ai_execution_factory = create_live_ai_execution
    config.storage.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    config.storage.artifact_root.mkdir(parents=True, exist_ok=True)
    hardware_authority = HardwareAuthority(
        config.storage.sqlite_path.parent.parent / "hardware-authority.lock"
    )
    baseline = load_b0(config.b0_path)
    baseline_hash = canonical_json_sha256(baseline)
    if config.mode is RuntimeMode.FAKE:
        tuning_profile = _built_in_fake_tuning_profile(baseline)
    elif tuning_profile_factory is not None:
        assert config.hardware is not None
        tuning_profile = tuning_profile_factory(
            config.hardware.driver_profile,
            baseline,
        )
        if not isinstance(tuning_profile, RuntimeTuningProfile):
            raise TypeError(
                "tuning profile factory must return RuntimeTuningProfile"
            )
    else:
        # The built-in READ_ONLY path intentionally exposes only V1.1
        # Preflight. It does not assemble the legacy mutation-capable campaign
        # adapters or require a tuning profile.
        tuning_profile = None
    owner_lease = RuntimeOwnerLease(config.storage.sqlite_path)
    try:
        store = CampaignStore(config.storage.sqlite_path)
    except Exception:
        owner_lease.close()
        raise
    selected_optimizer_factory = optimizer_factory or OptimizerFactory()
    selected_fir_proposer = (
        FakeFirProposer()
        if config.mode is RuntimeMode.FAKE and fir_proposer is None
        else fir_proposer
    )
    capabilities = RuntimeCapabilities.for_mode(
        config.mode,
        hierarchical_ready=selected_optimizer_factory.hierarchical_ready,
        # The proposer alone is insufficient: the API does not yet accept the
        # nested retune authorizations required by run_fir_stage.
        fir_proposer_ready=False,
        hardware_connected=False,
    )
    if config.mode is RuntimeMode.LIVE_TUNING and not legacy_campaign_enabled:
        # LIVE_TUNING can be a V1.1-only product runtime. The legacy V1 API is
        # retained for route compatibility but explicitly advertises that its
        # mutation-capable campaign path is unavailable.
        capabilities = capabilities.model_copy(
            update={"live_tuning_enabled": False}
        )

    def entry_factory(campaign_id: str) -> CampaignEntry:
        if not legacy_campaign_enabled:
            if config.mode is RuntimeMode.READ_ONLY:
                raise ReadOnlyModeError(
                    "built-in READ_ONLY runtime exposes V1.1 Preflight only"
                )
            if config.mode is RuntimeMode.LIVE_TUNING:
                raise RuntimeAssemblyError(
                    "legacy V1 campaigns are disabled in this V1.1-only "
                    "LIVE_TUNING runtime"
                )
        assert tuning_profile is not None
        campaign = store.get_campaign(campaign_id)
        if campaign.baseline_id != baseline.baseline_id:
            raise RuntimeAssemblyError(
                f"campaign baseline {campaign.baseline_id} is not configured"
            )
        if campaign.baseline_hash != baseline_hash:
            raise RuntimeAssemblyError("campaign baseline hash does not match B0")
        if config.mode is not RuntimeMode.FAKE and (
            campaign.config_hash != tuning_profile.driver_profile_hash
        ):
            raise RuntimeAssemblyError(
                "campaign config hash does not match the configured DriverProfile"
            )
        path = _selected_optimizer_path(store, campaign_id)
        context = OptimizationContext(
            driver_profile_hash=(
                campaign.config_hash
                if config.mode is RuntimeMode.FAKE
                else tuning_profile.driver_profile_hash
            ),
            vcsel_model=tuning_profile.vcsel_model,
            device_id=campaign_id,
            test_context_id=(
                campaign.config_hash
                if config.mode is RuntimeMode.FAKE
                else tuning_profile.test_context_id
            ),
        )
        optimizer = selected_optimizer_factory.create(
            path,
            space=tuning_profile.parameter_space,
            context=context,
            seed=0,
        )
        if config.mode is RuntimeMode.FAKE:
            adapters = build_fake_adapters(
                baseline,
                measurement_timeout_s=config.measurement_timeout_s,
            )
            resource_ids = ("fake-vcsel-bench",)
        else:
            assert hardware_adapter_factory is not None
            adapters = hardware_adapter_factory(
                campaign_id,
                baseline,
                config.measurement_timeout_s,
            )
            if not isinstance(adapters, AdapterBundle):
                raise TypeError("hardware adapter factory must return AdapterBundle")
            if (
                adapters.measurement_timeout_s is None
                or adapters.measurement_timeout_s <= 0
                or adapters.measurement_timeout_s
                > config.measurement_timeout_s
            ):
                adapters.close()
                raise RuntimeAssemblyError(
                    "hardware adapters must enforce and attest the configured "
                    "measurement timeout"
                )
            if config.mode is RuntimeMode.READ_ONLY:
                adapters = read_only_adapters(adapters)
            assert config.hardware is not None
            resource_ids = config.hardware.resource_ids
        repository: CampaignRepository | None = None
        try:
            repository = CampaignRepository(
                config.storage.artifact_root,
                campaign_id,
            )
            orchestrator = CampaignOrchestrator(
                campaign_id=campaign_id,
                baseline=baseline,
                baseline_hash=baseline_hash,
                optimization_context=context,
                store=store,
                repository=repository,
                optimizer=optimizer,
                dca=adapters.dca,
                i2c=adapters.i2c,
                bert=adapters.bert,
                clock=SystemClock(),
                fir_proposer=selected_fir_proposer,
                parameter_specs=tuning_profile.parameter_specs,
                fir_authorization_spec=tuning_profile.fir_authorization_spec,
                tdecq_only=path
                in {
                    OptimizerCapability.PATH_EXHAUSTIVE.value,
                    OptimizerCapability.PATH_TDECQ_BAYESIAN.value,
                },
            )
        except Exception:
            if repository is not None:
                repository.close()
            adapters.close()
            raise
        return CampaignEntry(
            campaign_id=campaign_id,
            repository=repository,
            adapters=adapters,
            orchestrator=orchestrator,
            resource_ids=resource_ids,
        )

    runtime: CampaignRuntime | None = None
    try:
        runtime = CampaignRuntime(
            config,
            store=store,
            entry_factory=entry_factory,
            capabilities=capabilities,
            owner_lease=owner_lease,
            hardware_authority=hardware_authority,
        )
        runtime.baseline_hash = baseline_hash
        runtime.recover()
    except Exception:
        if runtime is None:
            try:
                store.close()
            finally:
                owner_lease.close()
        else:
            runtime.close()
        raise
    service = ConcreteCampaignService(runtime)
    v11_store = None
    v11_service = None
    isolated_preflight_root = (
        config.storage.artifact_root.parent / "v1.2" / "preflight-dca"
    )
    if (
        config.mode is RuntimeMode.READ_ONLY
        and v11_read_only_driver_factory is None
    ):
        from vcsel_ai_tuning.v1_1.read_only_runtime import (
            open_isolated_read_only_drivers,
        )

        v11_read_only_driver_factory = partial(
            open_isolated_read_only_drivers,
            artifact_root=isolated_preflight_root,
            cleanup_grace_s=config.cleanup_grace_s,
        )
    if (
        config.mode is RuntimeMode.LIVE_TUNING
        and v11_live_driver_factory is None
    ):
        from vcsel_ai_tuning.v1_1.read_only_runtime import (
            open_isolated_read_only_drivers,
        )

        v11_live_driver_factory = partial(
            open_isolated_read_only_drivers,
            artifact_root=isolated_preflight_root,
            cleanup_grace_s=config.cleanup_grace_s,
        )
    if needs_builtin_live_gateway:
        from vcsel_ai_tuning.v1_1.live_runtime import LiveV11Gateway

        assert live_profile is not None
        v11_gateway = LiveV11Gateway(
            live_profile,
            total_timeout_s=config.measurement_timeout_s,
            driver_factory=v11_live_driver_factory,
            live_ai_profile=config.live_ai_profile,
            measurement_timeout_s=config.measurement_timeout_s,
            cleanup_grace_s=config.cleanup_grace_s,
        )
    default_fake_v11 = (
        config.mode is RuntimeMode.FAKE
        and v11_gateway is None
        and v11_runner is None
        and v11_sweep_executor_factory is None
    )
    default_read_only_v11 = (
        config.mode is RuntimeMode.READ_ONLY
        and v11_gateway is None
        and v11_runner is None
        and v11_sweep_executor_factory is None
    )
    has_v11_runner_path = (
        v11_runner is not None
        or v11_sweep_executor_factory is not None
        or v11_ai_execution_factory is not None
    )
    if (
        not default_fake_v11
        and not default_read_only_v11
        and (v11_gateway is None) != (not has_v11_runner_path)
    ):
        runtime.close()
        raise RuntimeAssemblyError(
            "v11_gateway and a V1.1 runner or sweep executor factory must be "
            "configured together"
        )
    if default_fake_v11:
        from vcsel_ai_tuning.v1_1.fake_runtime import (
            FakeHistoryHardware,
            FakeV11Gateway,
        )

        v11_gateway = FakeV11Gateway()
        if v11_history_hardware is None:
            v11_history_hardware = FakeHistoryHardware(v11_gateway.b0)
    elif default_read_only_v11:
        from vcsel_ai_tuning.v1_1.read_only_runtime import (
            ReadOnlyV11Gateway,
            ReadOnlyV11TaskRunner,
        )

        assert read_only_profile is not None
        v11_gateway = ReadOnlyV11Gateway(
            read_only_profile,
            total_timeout_s=config.measurement_timeout_s,
            driver_factory=v11_read_only_driver_factory,
        )
        v11_runner = ReadOnlyV11TaskRunner()
    elif (
        v11_sweep_executor_factory is not None
        or v11_ai_execution_factory is not None
    ):
        from vcsel_ai_tuning.v1_1.persistent_runner import (
            PersistentV11TaskRunner,
        )

        v11_runner = PersistentV11TaskRunner(
            database_path=_v11_sqlite_path(config.storage.sqlite_path),
            artifact_root=config.storage.artifact_root.parent / "v1.2",
            sweep_executor_factory=v11_sweep_executor_factory,
            ai_execution_factory=v11_ai_execution_factory,
        )
    if v11_gateway is not None:
        from vcsel_ai_tuning.v1_1.history_apply import HistoryApplyController
        from vcsel_ai_tuning.v1_1.repository import TaskStore
        from vcsel_ai_tuning.v1_1.service import V11ApplicationService

        v11_sqlite_path = _v11_sqlite_path(config.storage.sqlite_path)
        v11_artifact_root = config.storage.artifact_root.parent / "v1.2"
        runner_database_path = getattr(v11_runner, "database_path", None)
        runner_artifact_root = getattr(v11_runner, "artifact_root", None)
        if runner_database_path is not None:
            v11_sqlite_path = Path(runner_database_path)
        if runner_artifact_root is not None:
            v11_artifact_root = Path(runner_artifact_root)
        if default_fake_v11:
            try:
                from vcsel_ai_tuning.v1_1 import fake_ai_execution
                from vcsel_ai_tuning.v1_1.optimizer_bridge import (
                    OptimizerInterfaceUnavailable,
                )
                from vcsel_ai_tuning.v1_1.persistent_runner import (
                    FakeSweepExecutor,
                    PersistentV11TaskRunner,
                )

                ai_execution_factory = None
                try:
                    ai_execution_factory = (
                        fake_ai_execution.build_fake_ai_execution_factory(
                            database_path=v11_sqlite_path,
                            artifact_root=v11_artifact_root,
                        )
                    )
                except OptimizerInterfaceUnavailable:
                    logger.warning(
                        "V1.1 optimizer interface is unavailable; "
                        "FAKE AI execution remains disabled",
                        exc_info=True,
                    )

                def create_fake_sweep_executor(b0, plan):
                    return FakeSweepExecutor(
                        b0,
                        plan,
                        artifact_root=v11_artifact_root,
                    )

                v11_runner = PersistentV11TaskRunner(
                    database_path=v11_sqlite_path,
                    artifact_root=v11_artifact_root,
                    sweep_executor_factory=create_fake_sweep_executor,
                    ai_execution_factory=ai_execution_factory,
                )
            except Exception:
                runtime.close()
                raise
        def active_v1_campaign_id() -> str | None:
            if config.mode is RuntimeMode.FAKE:
                return None
            offset = 0
            inactive = {
                CampaignState.DRAFT,
                CampaignState.COMPLETED,
                CampaignState.FAILED,
            }
            while True:
                campaigns = store.list_campaigns(limit=100, offset=offset)
                if not campaigns:
                    return None
                for campaign in campaigns:
                    if campaign.state not in inactive:
                        return campaign.campaign_id
                offset += len(campaigns)

        try:
            v11_store = TaskStore(v11_sqlite_path)
            history_apply_controller = (
                None
                if v11_history_hardware is None
                else HistoryApplyController(
                    active_task_id=v11_store.active_task_id,
                    hardware=v11_history_hardware,
                    hardware_authority=hardware_authority,
                )
            )
            v11_service = V11ApplicationService(
                store=v11_store,
                gateway=v11_gateway,
                runner=v11_runner,
                artifact_root=v11_artifact_root,
                history_apply_controller=history_apply_controller,
                external_active_task_id=active_v1_campaign_id,
                hardware_authority=hardware_authority,
            )
            if config.mode is not RuntimeMode.FAKE:
                runtime._external_active_task_id = v11_store.active_task_id
        except Exception:
            if v11_service is not None:
                v11_service.close()
            else:
                close_v11_runner = getattr(v11_runner, "close", None)
                if callable(close_v11_runner):
                    close_v11_runner()
            if v11_store is not None:
                v11_store.close()
            runtime.close()
            raise

    def shutdown_all() -> None:
        try:
            try:
                if v11_service is not None:
                    v11_service.close()
            finally:
                if v11_store is not None:
                    v11_store.close()
        finally:
            runtime.close()

    return ApiRuntime(
        service=service,
        event_source=store,
        status_provider=runtime,
        v11_service=v11_service,
        v11_event_source=v11_service,
        shutdown=shutdown_all,
    )


def create_fake_runtime(config: Any = None) -> Any:
    """Return the built-in deterministic FAKE API runtime for CI and UI work."""

    from .runtime_config import RuntimeConfig

    selected = RuntimeConfig() if config is None else config
    if not isinstance(selected, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    if selected.mode is not RuntimeMode.FAKE:
        raise RuntimeAssemblyError("create_fake_runtime requires mode FAKE")
    return create_runtime(selected)
