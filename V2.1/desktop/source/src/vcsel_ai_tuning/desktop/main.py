"""Windows desktop entry point with an entirely offline smoke-test path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

from vcsel_ai_tuning.v2.continuous.models import MeasurementStatus

from .artifacts import CampaignProjection, MeasuredPoint
from .build_support import stage_integrity_bundle
from .csv_export import export_campaign_csv
from .integrity import (
    BUNDLED_PROTECTED_PATHS,
    IntegrityResult,
    load_manifest,
    verify_manifest,
)
from .models import DEFAULT_B0, BenchSettings, RunInputs
from .session import ContinuousLiveEngineAdapter, DesktopSession
from .session_config import SessionConfigBuilder
from .settings import BenchSettingsStore


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    resource_root: Path
    template_root: Path
    manifest_path: Path
    app_data_root: Path
    settings_path: Path
    campaigns_root: Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_application_paths(*, create: bool) -> ApplicationPaths:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is None:
        resource_root = _repository_root()
        template_root = resource_root / "configs" / "ai_tuning"
    else:
        resource_root = Path(str(frozen_root)).resolve()
        template_root = resource_root / "protected" / "configs" / "ai_tuning"
    local_app_data = Path(
        os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
    )
    app_data = local_app_data / "VCSELContinuousEye"
    paths = ApplicationPaths(
        resource_root=resource_root,
        template_root=template_root,
        manifest_path=resource_root / "build-manifest.json",
        app_data_root=app_data,
        settings_path=app_data / "settings.json",
        campaigns_root=app_data / "campaigns",
    )
    if create:
        paths.campaigns_root.mkdir(parents=True, exist_ok=True)
    return paths


def verify_distribution(paths: ApplicationPaths) -> IntegrityResult:
    try:
        manifest = load_manifest(paths.manifest_path)
        return verify_manifest(
            paths.resource_root,
            manifest,
            required_paths=BUNDLED_PROTECTED_PATHS,
        )
    except Exception as error:
        return IntegrityResult(
            verified=False,
            errors=(f"build manifest verification failed: {error}",),
        )


def build_live_session(paths: ApplicationPaths) -> DesktopSession:
    adapter = ContinuousLiveEngineAdapter(
        SessionConfigBuilder(paths.template_root),
        paths.campaigns_root,
    )
    return DesktopSession(adapter)


def launch_ui(
    paths: ApplicationPaths,
    session: DesktopSession,
    integrity: IntegrityResult,
) -> int:
    import tkinter as tk

    from .app import DesktopApp

    root = tk.Tk()
    DesktopApp(
        root,
        session,
        BenchSettingsStore(paths.settings_path),
        integrity,
    )
    root.mainloop()
    return 0


def _offline_projection(root: Path) -> CampaignProjection:
    baseline = DEFAULT_B0.model_dump()
    point = MeasuredPoint(
        row_type="B0",
        round_index=0,
        observation_id="offline-smoke-b0",
        requested_params=baseline,
        applied_params=baseline,
        readback_params=baseline,
        tdecq_db=3.2,
        outer_er_db=3.0,
        measurement_status=MeasurementStatus.VALID,
    )
    return CampaignProjection(
        campaign_id="offline-smoke",
        terminal_state="COMPLETED",
        execution_mode="FAKE",
        max_rounds=1,
        target_tdecq_db=2.8,
        target_reached=False,
        b0_restored=True,
        evidence_complete=True,
        stop_reason="OFFLINE_SMOKE_TEST",
        provider_id="offline",
        provider_version="offline",
        adapter_identities={},
        campaign_root=root,
        points=(point,),
        best_round_index=0,
    )


def offline_smoke_test(paths: ApplicationPaths) -> bool:
    if find_spec("sklearn") is None:
        return False
    with TemporaryDirectory(prefix="vcsel-desktop-smoke-") as temporary_name:
        temporary = Path(temporary_name)
        if getattr(sys, "_MEIPASS", None) is None:
            stage = stage_integrity_bundle(
                paths.resource_root,
                temporary / "integrity-stage",
                desktop_commit="source-offline-smoke",
            )
            manifest = load_manifest(stage / "build-manifest.json")
            integrity = verify_manifest(
                stage,
                manifest,
                required_paths=BUNDLED_PROTECTED_PATHS,
            )
        else:
            integrity = verify_distribution(paths)
        if not integrity.verified:
            return False

        bench = BenchSettings(
            dca_visa_resource="TCPIP0::offline::hislip0,4880::INSTR",
            i2c_com_port="COM99",
            i2c_address=0xA6,
        )
        store = BenchSettingsStore(temporary / "settings.json")
        store.save(bench)
        if store.load() != bench:
            return False
        inputs = RunInputs(
            api_key="offline-smoke-secret",
            max_rounds=1,
            target_tdecq_db=2.8,
            b0=DEFAULT_B0,
            bench=bench,
        )
        generated = SessionConfigBuilder(paths.template_root).build(
            temporary / "campaigns", inputs, "offline-smoke"
        )
        if any(
            "offline-smoke-secret" in path.read_text(encoding="utf-8")
            for path in generated.config_files
        ):
            return False
        output = export_campaign_csv(
            _offline_projection(generated.campaign_root),
            temporary / "offline-smoke.csv",
        )
        if not output.read_bytes().startswith(b"\xef\xbb\xbf"):
            return False
        import tkinter  # noqa: F401 - packaging smoke without creating a window
        from PIL import Image  # noqa: F401 - ensure preview dependency is bundled

        return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="VCSELContinuousEye")
    parser.add_argument("--offline-smoke-test", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = resolve_application_paths(create=not args.offline_smoke_test)
    if args.offline_smoke_test:
        return 0 if offline_smoke_test(paths) else 1
    integrity = verify_distribution(paths)
    session = build_live_session(paths)
    return launch_ui(paths, session, integrity)


def main() -> int:
    return run()
