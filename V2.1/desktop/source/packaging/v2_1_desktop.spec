# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path


repository_root = Path(SPECPATH).resolve().parent
stage_root = Path(os.environ["VCSEL_DESKTOP_STAGE"]).resolve()

analysis = Analysis(
    [str(repository_root / "packaging" / "v2_1_desktop_launcher.py")],
    pathex=[str(repository_root / "src")],
    binaries=[],
    datas=[
        (str(stage_root / "build-manifest.json"), "."),
        (str(stage_root / "protected"), "protected"),
    ],
    hiddenimports=[
        "vcsel_ai_tuning.continuous_live",
        "openai",
        "serial",
        "sklearn",
        "pyvisa",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="VCSELContinuousEye",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VCSELContinuousEye",
)
