from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import vcsel_ai_tuning.optimization.v11 as v11


SOURCE_ROOT = Path(__file__).parents[4] / "src"
V11_ROOT = SOURCE_ROOT / "vcsel_ai_tuning" / "optimization" / "v11"


def test_v11_sources_do_not_import_hardware_adapters_or_campaign_execution():
    forbidden_roots = {
        "vcsel_ai_tuning.hardware",
        "vcsel_ai_tuning.adapters",
        "vcsel_ai_tuning.campaign",
    }
    violations = []
    for path in sorted(V11_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if any(
                    module == root or module.startswith(f"{root}.")
                    for root in forbidden_roots
                ):
                    violations.append((path.name, node.lineno, module))

    assert violations == []


def test_importing_v11_does_not_load_hardware_or_adapter_modules():
    script = f"""
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
import vcsel_ai_tuning.optimization.v11
for name in sys.modules:
    assert not name.startswith('vcsel_ai_tuning.hardware'), name
    assert not name.startswith('vcsel_ai_tuning.adapters'), name
    assert not name.startswith('vcsel_ai_tuning.campaign'), name
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=SOURCE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_v11_public_api_exports_only_pure_models_and_decisions():
    forbidden_names = {
        "setFir",
        "SCPI",
        "I2CFrame",
        "Connection",
        "RollbackCommand",
    }

    assert set(v11.__all__).isdisjoint(forbidden_names)
    assert all(hasattr(v11, name) for name in v11.__all__)


def test_v11_production_sources_do_not_emit_hardware_command_strings():
    forbidden_fragments = (
        "setFir",
        "SCPI",
        "raw I2C frame",
        "pyvisa",
        "serial.Serial",
        "socket.connect",
    )
    matches = []
    for path in sorted(V11_ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                matches.append((path.name, fragment))

    assert matches == []
