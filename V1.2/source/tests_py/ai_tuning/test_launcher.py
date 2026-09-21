from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "start-v1.2.ps1"
CMD_LAUNCHER = PROJECT_ROOT / "start-v1.2.cmd"
IMPLEMENTATION = PROJECT_ROOT / "start-v1.1.ps1"


def _fake_project(tmp_path: Path, *, frontend: bool = True) -> Path:
    root = tmp_path / "launcher project"
    root.mkdir()
    shutil.copy2(LAUNCHER, root / LAUNCHER.name)
    shutil.copy2(CMD_LAUNCHER, root / CMD_LAUNCHER.name)
    shutil.copy2(IMPLEMENTATION, root / IMPLEMENTATION.name)
    entrypoint = root / "src/vcsel_ai_tuning/api/app.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# test FastAPI entrypoint\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "vcsel-launcher-test"\nversion = "1.2.0"\n',
        encoding="utf-8",
    )
    if frontend:
        dist = root / "frontend/dist"
        dist.mkdir(parents=True)
        (dist / "index.html").write_text(
            "<!doctype html><title>VCSEL test</title>",
            encoding="utf-8",
        )
    return root


def _run_launcher(
    root: Path,
    *arguments: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["VCSEL_AI_PYTHON"] = sys.executable
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / LAUNCHER.name),
            *arguments,
        ],
        cwd=cwd or root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_fake_validation_uses_script_root_from_another_directory(
    tmp_path: Path,
) -> None:
    root = _fake_project(tmp_path)

    result = _run_launcher(
        root,
        "-Mode",
        "FAKE",
        "-Port",
        "18766",
        "-NoBrowser",
        "-ValidateOnly",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation passed" in result.stdout
    log = next((root / "logs").glob("v1.2-launcher-*.log"))
    contents = log.read_text(encoding="utf-8")
    assert "version=1.2.0" in contents
    assert "mode=FAKE" in contents
    assert "host=127.0.0.1" in contents
    assert "port=18766" in contents
    assert f"python={sys.executable}" in contents
    assert f"frontend_dist={root / 'frontend/dist'}" in contents


def test_missing_frontend_build_fails_with_exact_build_command(
    tmp_path: Path,
) -> None:
    root = _fake_project(tmp_path, frontend=False)

    result = _run_launcher(root, "-ValidateOnly")

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "frontend/dist/index.html" in output
    assert "pnpm --dir frontend build" in output


def test_hardware_modes_require_an_explicit_config(tmp_path: Path) -> None:
    root = _fake_project(tmp_path)

    result = _run_launcher(
        root,
        "-Mode",
        "READ_ONLY",
        "-NoBrowser",
        "-ValidateOnly",
    )

    assert result.returncode != 0
    assert "requires -ConfigPath" in result.stdout + result.stderr


def test_missing_config_path_fails_closed(tmp_path: Path) -> None:
    root = _fake_project(tmp_path)

    result = _run_launcher(
        root,
        "-Mode",
        "LIVE_TUNING",
        "-ConfigPath",
        str(root / "missing.yaml"),
        "-NoBrowser",
        "-ValidateOnly",
    )

    assert result.returncode != 0
    assert "Runtime config does not exist" in result.stdout + result.stderr


def test_occupied_port_fails_before_server_start(tmp_path: Path) -> None:
    root = _fake_project(tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        result = _run_launcher(
            root,
            "-Port",
            str(port),
            "-NoBrowser",
            "-ValidateOnly",
        )
    finally:
        listener.close()

    assert result.returncode != 0
    assert f"Port {port} is already in use" in result.stdout + result.stderr


def test_requested_mode_must_match_config_mode(tmp_path: Path) -> None:
    root = _fake_project(tmp_path)
    config = root / "runtime.yaml"
    config.write_text("mode: FAKE\n", encoding="utf-8")

    result = _run_launcher(
        root,
        "-Mode",
        "READ_ONLY",
        "-ConfigPath",
        str(config),
        "-NoBrowser",
        "-ValidateOnly",
    )

    assert result.returncode != 0
    assert "does not match runtime config mode FAKE" in result.stdout + result.stderr


def test_launcher_contains_no_direct_hardware_commands() -> None:
    contents = LAUNCHER.read_text(encoding="utf-8") + IMPLEMENTATION.read_text(
        encoding="utf-8"
    )

    for forbidden in ("setFir", ":MEASure", "pyvisa", "serial.Serial"):
        assert forbidden not in contents


def test_cmd_wrapper_bypasses_only_child_execution_policy(tmp_path: Path) -> None:
    root = _fake_project(tmp_path)
    environment = os.environ.copy()
    environment["VCSEL_AI_PYTHON"] = sys.executable

    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            str(root / CMD_LAUNCHER.name),
            "-Mode",
            "FAKE",
            "-Port",
            "18769",
            "-NoBrowser",
            "-ValidateOnly",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation passed" in result.stdout
