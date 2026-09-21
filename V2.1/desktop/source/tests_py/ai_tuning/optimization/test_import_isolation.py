import ast
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys
import textwrap


OPTIMIZATION_ROOT = (
    Path(__file__).parents[3]
    / "src"
    / "vcsel_ai_tuning"
    / "optimization"
)
SOURCE_ROOT = OPTIMIZATION_ROOT.parents[1]
DOCUMENT_PATH = (
    OPTIMIZATION_ROOT.parents[2]
    / "docs"
    / "dual-path-tdecq-optimizer.md"
)
BANNED_ROOTS = {
    "fastapi",
    "pyvisa",
    "serial",
    "socket",
    "vcsel_ai_tuning.hardware",
    "vcsel_ai_tuning.persistence",
}
EXPECTED_PUBLIC_SYMBOLS = {
    "BayesianSnapshot",
    "CandidatePrediction",
    "CandidateSuggestion",
    "CoordinateDescent",
    "EffectEdge",
    "EffectMemory",
    "ExhaustiveOptimizer",
    "FallbackRequested",
    "IntegerParameterSpace",
    "NoSafeCandidate",
    "NoTransitionPath",
    "OptimizationContext",
    "OptimizerProtocol",
    "PairInteraction",
    "PathComplete",
    "ReplayReport",
    "ReplaySurface",
    "ResponseNode",
    "Suggestion",
    "TdecqBayesianOptimizer",
    "TuningObservation",
    "TuningPath",
    "replay",
}


def _resolved_import_names(
    path: Path,
    node: ast.Import | ast.ImportFrom,
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    if node.level:
        package = ".".join(path.relative_to(SOURCE_ROOT).parent.parts)
        reference = "." * node.level + (node.module or "")
        base = resolve_name(reference, package)
    else:
        base = node.module or ""

    names = [base] if base else []
    names.extend(
        f"{base}.{alias.name}" if base else alias.name
        for alias in node.names
        if alias.name != "*"
    )
    return tuple(names)


def test_optimizer_package_has_no_hardware_network_api_or_persistence_imports():
    violations = []
    for path in sorted(OPTIMIZATION_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for name in _resolved_import_names(path, node):
                if any(
                    name == root or name.startswith(f"{root}.")
                    for root in BANNED_ROOTS
                ):
                    relative_path = path.relative_to(OPTIMIZATION_ROOT)
                    violations.append(f"{relative_path}: {name}")

    assert violations == []


def test_import_resolver_handles_absolute_relative_and_aliased_hardware_imports():
    cases = (
        (
            "from vcsel_ai_tuning import hardware as hw",
            "vcsel_ai_tuning.hardware",
        ),
        (
            "from ..hardware import Instrument as Device",
            "vcsel_ai_tuning.hardware",
        ),
        (
            "from .. import hardware as hw",
            "vcsel_ai_tuning.hardware",
        ),
    )
    synthetic_path = OPTIMIZATION_ROOT / "synthetic.py"

    for source, expected in cases:
        tree = ast.parse(source)
        node = tree.body[0]
        assert isinstance(node, ast.ImportFrom)
        assert expected in _resolved_import_names(synthetic_path, node)


def test_optimizer_package_exports_completed_public_symbols():
    package = import_module("vcsel_ai_tuning.optimization")

    assert set(package.__all__) == EXPECTED_PUBLIC_SYMBOLS
    for name in EXPECTED_PUBLIC_SYMBOLS:
        exported = getattr(package, name)
        assert package.__dict__[name] is exported


def test_pure_optimizer_imports_do_not_require_sklearn():
    script = textwrap.dedent(
        f"""
        import builtins
        import sys

        sys.path.insert(0, {str(SOURCE_ROOT)!r})
        real_import = builtins.__import__

        def block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ModuleNotFoundError("blocked sklearn", name="sklearn")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_sklearn

        from vcsel_ai_tuning.optimization.protocol import TuningPath
        from vcsel_ai_tuning.optimization.exhaustive import ExhaustiveOptimizer
        from vcsel_ai_tuning.optimization import ReplayReport, ReplaySurface, replay
        import vcsel_ai_tuning.optimization as optimization

        assert TuningPath.EXHAUSTIVE.value == "PATH_EXHAUSTIVE"
        assert ExhaustiveOptimizer.__name__ == "ExhaustiveOptimizer"
        assert ReplayReport.__name__ == "ReplayReport"
        assert ReplaySurface.__name__ == "ReplaySurface"
        assert callable(replay)
        assert not any(
            name == "sklearn" or name.startswith("sklearn.")
            for name in sys.modules
        )

        try:
            optimization.BayesianSnapshot
        except ModuleNotFoundError as error:
            assert ".[optimizer]" in str(error)
        else:
            raise AssertionError("BayesianSnapshot unexpectedly loaded without sklearn")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=OPTIMIZATION_ROOT.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_replay_documentation_has_fresh_checkout_windows_prerequisites():
    document = DOCUMENT_PATH.read_text(encoding="utf-8")

    assert "Python 3.12 or newer" in document
    assert "Python 3.11 or newer" not in document
    assert "py -3.12" not in document
    assert r"$VcselPython = 'C:\path\to\python.exe'" in document
    assert "& $VcselPython -m venv .venv-ai-tuning" in document
    assert (
        "Replace the placeholder path with any installed Python 3.12 or newer "
        "executable."
    ) in document
    assert "already exists in this workspace" in document
    assert (
        r"& '.\.venv-ai-tuning\Scripts\python.exe' "
        "-m pip install -e '.[dev,optimizer]'"
    ) in document
    assert (
        r"& '.\.venv-ai-tuning\Scripts\python.exe' "
        "-m pytest tests_py/ai_tuning/optimization/test_replay.py -q"
    ) in document
