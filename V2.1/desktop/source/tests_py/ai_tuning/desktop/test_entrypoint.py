from pathlib import Path

import vcsel_ai_tuning.desktop.main as main


def _forbidden(*args, **kwargs):
    raise AssertionError("offline smoke test crossed the live/UI boundary")


def test_entrypoint_offline_flag_never_builds_live_adapter_or_tk(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main, "build_live_session", _forbidden)
    monkeypatch.setattr(main, "launch_ui", _forbidden)

    assert main.run(["--offline-smoke-test"]) == 0


def test_offline_smoke_rejects_missing_optimizer_dependency(monkeypatch) -> None:
    monkeypatch.setattr(main, "find_spec", lambda _name: None, raising=False)

    assert main.offline_smoke_test(
        main.resolve_application_paths(create=False)
    ) is False


def test_development_paths_resolve_inside_current_worktree(monkeypatch) -> None:
    monkeypatch.delattr(main.sys, "_MEIPASS", raising=False)

    paths = main.resolve_application_paths(create=False)

    repository_root = Path(__file__).resolve().parents[3]
    assert paths.resource_root == repository_root
    assert paths.template_root == repository_root / "configs" / "ai_tuning"
    assert paths.manifest_path == repository_root / "build-manifest.json"


def test_frozen_paths_use_bundled_protected_templates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main.sys, "_MEIPASS", str(tmp_path), raising=False)

    paths = main.resolve_application_paths(create=False)

    assert paths.resource_root == tmp_path.resolve()
    assert paths.template_root == tmp_path.resolve() / "protected" / "configs" / "ai_tuning"
