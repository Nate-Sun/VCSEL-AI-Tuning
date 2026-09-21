"""Run a disposable localhost-only FAKE API for browser contract verification."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    original_import = builtins.__import__

    def reject_hardware_import(name, *args, **kwargs):
        if name in {"pyvisa", "serial"} or name.startswith(("pyvisa.", "serial.")):
            raise RuntimeError(f"FAKE E2E blocked hardware import: {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = reject_hardware_import
    try:
        import uvicorn

        from vcsel_ai_tuning.api.app import create_app_from_runtime_factory
        from vcsel_ai_tuning.runtime import create_fake_runtime
        from vcsel_ai_tuning.runtime_config import RuntimeConfig

        port = int(os.environ.get("VCSEL_V11_E2E_PORT", "8011"))
        with TemporaryDirectory(prefix="vcsel-v11-browser-e2e-") as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                api={"host": "127.0.0.1", "port": port},
                storage={
                    "sqlite_path": root / "state" / "campaigns.sqlite3",
                    "artifact_root": root / "artifacts",
                },
                b0_path=Path("configs/ai_tuning/b0.yaml").resolve(),
                worker_count=2,
            )
            runtime = create_fake_runtime(config)
            app = create_app_from_runtime_factory(lambda: runtime)
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        builtins.__import__ = original_import
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
