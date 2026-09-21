"""Static import and source-text isolation checks for the V2 foundation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
V2_ROOT = SOURCE_ROOT / "vcsel_ai_tuning" / "v2"

FORBIDDEN_APPLICATION_IMPORTS = (
    "vcsel_ai_tuning.hardware",
    "vcsel_ai_tuning.backends",
    "vcsel_ai_tuning.adapters",
    "vcsel_ai_tuning.hardware_authority",
    "vcsel_ai_tuning.persistence",
    "vcsel_ai_tuning.campaign.orchestrator",
    "vcsel_ai_tuning.v1_1.hardware_control",
)
FORBIDDEN_TRANSPORT_IMPORTS = ("pyvisa", "serial", "socket")
FORBIDDEN_FOUNDATION_TEXT = (
    "setFir",
    "pyvisa",
    "import serial",
    "import socket",
    "socket.",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_name(path: Path) -> str:
    module = _module_name(path)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _resolved_import_names(node: ast.Import | ast.ImportFrom, package: str) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    if node.level:
        reference = "." * node.level + (node.module or "")
        base = importlib.util.resolve_name(reference, package)
    else:
        base = node.module or ""
    names = [base] if base else []
    names.extend(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return tuple(names)


def _all_import_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_name(path)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.extend(_resolved_import_names(node, package))
    return tuple(names)


def _matches_forbidden(name: str, forbidden: str) -> bool:
    return name == forbidden or name.startswith(f"{forbidden}.")


def test_import_resolver_covers_from_package_import_hardware():
    node = ast.parse("from vcsel_ai_tuning import hardware").body[0]

    assert isinstance(node, ast.ImportFrom)
    assert _resolved_import_names(node, "vcsel_ai_tuning.v2") == (
        "vcsel_ai_tuning",
        "vcsel_ai_tuning.hardware",
    )


def test_import_resolver_covers_relative_package_import_hardware():
    node = ast.parse("from .. import hardware").body[0]

    assert isinstance(node, ast.ImportFrom)
    assert _resolved_import_names(node, "vcsel_ai_tuning.v2") == (
        "vcsel_ai_tuning",
        "vcsel_ai_tuning.hardware",
    )


def test_v2_source_has_no_forbidden_application_or_transport_imports():
    violations: list[str] = []
    for path in V2_ROOT.rglob("*.py"):
        for name in _all_import_names(path):
            if any(
                _matches_forbidden(name, forbidden)
                for forbidden in (*FORBIDDEN_APPLICATION_IMPORTS, *FORBIDDEN_TRANSPORT_IMPORTS)
            ):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {name}")

    assert violations == []


def test_v2_foundation_source_contains_no_hardware_control_text():
    violations: list[str] = []
    for path in V2_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_FOUNDATION_TEXT:
            if forbidden in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {forbidden}")

    assert violations == []
