from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build_v2_1_desktop.ps1"
SPEC = REPOSITORY_ROOT / "packaging" / "v2_1_desktop.spec"
LAUNCHER = REPOSITORY_ROOT / "packaging" / "v2_1_desktop_launcher.py"


def test_build_script_requires_validated_base_and_protected_diff_check() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "4085b007546f820820dbdb7f7f0bba6ebc4b2049" in text
    assert "git status --porcelain" in text
    assert "git merge-base --is-ancestor" in text
    assert "git diff --quiet" in text
    assert "--offline-smoke-test" in text
    assert "Get-FileHash" in text
    assert "v2_1_desktop.spec" in text


def test_pyinstaller_spec_is_windowed_onedir_and_bundles_integrity_data() -> None:
    text = SPEC.read_text(encoding="utf-8")

    assert "console=False" in text
    assert "COLLECT(" in text
    assert "build-manifest.json" in text
    assert "protected" in text
    assert "vcsel_ai_tuning.continuous_live" in text
    assert '"sklearn"' in text
    assert "v2_1_desktop_launcher.py" in text


def test_pyinstaller_launcher_uses_absolute_package_import() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "from vcsel_ai_tuning.desktop.main import main" in text
    assert "from ." not in text


def test_project_declares_desktop_entry_and_build_dependencies() -> None:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = tomllib.loads(text)["project"]
    desktop_dependencies = project["optional-dependencies"]["desktop-build"]

    assert 'vcsel-continuous-eye-desktop = "vcsel_ai_tuning.desktop.main:main"' in text
    assert 'desktop-build = [' in text
    assert '"pyinstaller>=6.16,<7"' in text
    assert "scikit-learn>=1.9,<2" in desktop_dependencies
