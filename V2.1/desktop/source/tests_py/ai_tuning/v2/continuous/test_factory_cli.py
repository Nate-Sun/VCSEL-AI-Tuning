from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

from vcsel_ai_tuning.v2.continuous.cli import main
from vcsel_ai_tuning.v2.continuous.factory import (
    ContinuousDependencies,
    LiveContactAuthorizationError,
    build_continuous_dependencies,
)
from vcsel_ai_tuning.v2.continuous.models import ContinuousState, ExecutionMode
import vcsel_ai_tuning.v2.continuous.factory as continuous_factory


def test_fake_factory_does_not_read_api_key_or_import_live_builders(
    tmp_path,
    monkeypatch,
) -> None:
    def forbidden_getenv(name, default=None):
        if name == "MOONSHOT_API_KEY":
            raise AssertionError("FAKE must not read MOONSHOT_API_KEY")
        return default

    monkeypatch.setattr(os, "getenv", forbidden_getenv)
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.FAKE,
        result_dir=tmp_path,
        scenario="happy",
        live_config={"campaign_id": "REPLACE_WITH_FRESH_CAMPAIGN_ID"},
    )
    assert isinstance(dependencies, ContinuousDependencies)
    assert dependencies.mode is ExecutionMode.FAKE
    assert dependencies.contact_permitted is False


def test_fake_factory_preserves_defaults_and_allows_cli_control_values(tmp_path) -> None:
    default_dependencies = build_continuous_dependencies(
        mode=ExecutionMode.FAKE,
        result_dir=tmp_path / "default",
    )
    explicit_dependencies = build_continuous_dependencies(
        mode=ExecutionMode.FAKE,
        result_dir=tmp_path / "explicit",
        max_rounds=20,
        target_tdecq_db=2.8,
    )

    assert default_dependencies.request is not None
    assert default_dependencies.request.max_rounds == 3
    assert default_dependencies.request.stopping_policy.target_tdecq_db == 2.0
    assert explicit_dependencies.request is not None
    assert explicit_dependencies.request.max_rounds == 20
    assert explicit_dependencies.request.stopping_policy.target_tdecq_db == 2.8


@pytest.mark.parametrize(
    ("max_rounds", "target_tdecq_db"),
    [
        (True, 2.0),
        (1.0, 2.0),
        (0, 2.0),
        (-1, 2.0),
        (3, True),
        (3, 0),
        (3, -1),
        (3, float("nan")),
        (3, float("inf")),
        pytest.param(3, 10**10000, id="target-overflow"),
    ],
)
def test_factory_rejects_invalid_operator_controls_before_fake_construction(
    tmp_path,
    monkeypatch,
    max_rounds,
    target_tdecq_db,
) -> None:
    monkeypatch.setattr(
        continuous_factory,
        "fake_scenario",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not construct")),
    )

    with pytest.raises(ValueError):
        build_continuous_dependencies(
            mode=ExecutionMode.FAKE,
            result_dir=tmp_path,
            max_rounds=max_rounds,
            target_tdecq_db=target_tdecq_db,
        )


def test_live_factory_refuses_construction_without_device_contact_gate(tmp_path) -> None:
    with pytest.raises(LiveContactAuthorizationError, match="allow-device-contact"):
        build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=tmp_path,
            allow_device_contact=False,
        )


def test_live_factory_returns_lazy_activation_without_contact(tmp_path) -> None:
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        allow_device_contact=True,
        live_config={
            "runtime_config": "configs/ai_tuning/runtime_live.yaml",
            "campaign_id": "campaign-1",
        },
    )

    assert dependencies.mode is ExecutionMode.LIVE
    assert dependencies.controller is None
    assert dependencies.request is None
    assert dependencies.live_activation is not None
    assert dependencies.live_activation.activated is False
    assert dict(dependencies.live_activation.config)["target_tdecq_db"] == 2.0


def test_live_factory_rejects_checked_in_campaign_placeholder_before_import(
    tmp_path,
    monkeypatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(LiveContactAuthorizationError, match="placeholder"):
        build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=tmp_path,
            allow_device_contact=True,
            live_config={
                "runtime_config": "runtime-v2.1-live.yaml",
                "campaign_id": "REPLACE_WITH_FRESH_CAMPAIGN_ID",
            },
        )

    assert imports == []


def test_live_factory_binds_operator_controls_in_immutable_activation_scope(tmp_path) -> None:
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=20,
        target_tdecq_db=2.8,
        allow_device_contact=True,
        live_config={
            "runtime_config": "configs/ai_tuning/runtime_live.yaml",
            "campaign_id": "campaign-1",
            "max_rounds": 3,
            "target_tdecq_db": 9.9,
        },
    )

    assert dependencies.live_activation is not None
    assert dict(dependencies.live_activation.config) == {
        "runtime_config": "configs/ai_tuning/runtime_live.yaml",
        "campaign_id": "campaign-1",
        "max_rounds": 20,
        "target_tdecq_db": 2.8,
    }
    with pytest.raises(TypeError):
        dependencies.live_activation.config["max_rounds"] = 3


def test_lazy_live_activation_config_snapshot_cannot_be_replaced(
    tmp_path,
    monkeypatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: imports.append(name),
    )
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=3,
        target_tdecq_db=2.0,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime-v2.1-live.yaml",
            "campaign_id": "campaign-1",
        },
    )
    assert dependencies.live_activation is not None

    with pytest.raises(AttributeError):
        dependencies.live_activation.config = {
            "runtime_config": "other-runtime.yaml",
            "campaign_id": "other-campaign",
            "max_rounds": 999,
            "target_tdecq_db": 99.0,
        }
    replacement_scope = replace(
        dependencies.live_activation._frozen_scope,
        execution_mode=ExecutionMode.FAKE,
    )
    with pytest.raises(AttributeError):
        dependencies.live_activation._frozen_scope = replacement_scope

    assert dict(dependencies.live_activation.config) == {
        "runtime_config": "runtime-v2.1-live.yaml",
        "campaign_id": "campaign-1",
        "max_rounds": 3,
        "target_tdecq_db": 2.0,
    }
    assert dependencies.live_activation.activated is False
    assert imports == []


@pytest.mark.parametrize("entrypoint", ["preflight", "activate"])
@pytest.mark.parametrize(
    ("scope_field", "invalid_value", "error_type", "message"),
    [
        ("execution_mode", ExecutionMode.FAKE, LiveContactAuthorizationError, "LIVE"),
        (
            "allow_device_contact",
            False,
            LiveContactAuthorizationError,
            "allow-device-contact",
        ),
        (
            "config",
            {
                "runtime_config": "runtime-v2.1-live.yaml",
                "campaign_id": "campaign-1",
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            LiveContactAuthorizationError,
            "immutable",
        ),
    ],
)
def test_lazy_live_entrypoints_revalidate_complete_frozen_scope_before_import(
    tmp_path,
    monkeypatch,
    entrypoint,
    scope_field,
    invalid_value,
    error_type,
    message,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: imports.append(name),
    )
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=3,
        target_tdecq_db=2.0,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime-v2.1-live.yaml",
            "campaign_id": "campaign-1",
        },
    )
    assert dependencies.live_activation is not None
    activation = dependencies.live_activation
    object.__setattr__(
        activation,
        "_frozen_scope",
        replace(
            activation._frozen_scope,
            **{scope_field: invalid_value},
        ),
    )

    with pytest.raises(error_type, match=message):
        getattr(activation, entrypoint)()

    assert activation.activated is False
    assert imports == []


@pytest.mark.parametrize("entrypoint", ["preflight", "activate"])
@pytest.mark.parametrize(
    ("config_field", "invalid_value", "message"),
    [
        ("runtime_config", [], "runtime_config"),
        ("campaign_id", "REPLACE_WITH_FRESH_CAMPAIGN_ID", "placeholder"),
        ("max_rounds", True, "rounds"),
        ("target_tdecq_db", float("nan"), "target"),
        ("parameter_policy", {"rules": {}}, "immutable"),
    ],
)
def test_lazy_live_entrypoints_reject_tampered_internal_config_before_import(
    tmp_path,
    monkeypatch,
    entrypoint,
    config_field,
    invalid_value,
    message,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: imports.append(name),
    )
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=3,
        target_tdecq_db=2.0,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime-v2.1-live.yaml",
            "campaign_id": "campaign-1",
        },
    )
    assert dependencies.live_activation is not None
    activation = dependencies.live_activation
    malformed = dict(activation.config)
    malformed[config_field] = invalid_value
    object.__setattr__(
        activation,
        "_frozen_scope",
        replace(
            activation._frozen_scope,
            config=MappingProxyType(malformed),
        ),
    )

    with pytest.raises((ValueError, LiveContactAuthorizationError), match=message):
        getattr(activation, entrypoint)()

    assert activation.activated is False
    assert imports == []


@pytest.mark.parametrize(
    ("control", "invalid_value"),
    [("max_rounds", True), ("target_tdecq_db", float("nan"))],
)
def test_lazy_activation_revalidates_internal_controls_before_import(
    tmp_path,
    monkeypatch,
    control,
    invalid_value,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: imports.append(name),
    )
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=3,
        target_tdecq_db=2.0,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime-v2.1-live.yaml",
            "campaign_id": "campaign-1",
        },
    )
    assert dependencies.live_activation is not None
    malformed = dict(dependencies.live_activation.config)
    malformed[control] = invalid_value
    object.__setattr__(
        dependencies.live_activation,
        "_frozen_scope",
        replace(
            dependencies.live_activation._frozen_scope,
            config=MappingProxyType(malformed),
        ),
    )

    with pytest.raises(ValueError):
        dependencies.live_activation.activate()

    assert dependencies.live_activation.activated is False
    assert imports == []


def test_live_preflight_uses_dedicated_read_only_composition(tmp_path, monkeypatch) -> None:
    calls = []
    expected = {
        "operation": "PREFLIGHT",
        "hardware_write_performed": False,
        "kimi_connectivity_verified": True,
    }

    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: SimpleNamespace(
            preflight_continuous_live=lambda config, result_dir: (
                calls.append((name, dict(config), result_dir)) or expected
            ),
            activate_continuous_live=lambda *args: (_ for _ in ()).throw(
                AssertionError("preflight must not use commissioning activation")
            ),
        ),
    )
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=3,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime-v2.1-live.yaml",
            "campaign_id": "campaign-1",
        },
    )

    report = dependencies.live_activation.preflight()

    assert report == expected
    assert calls == [
        (
            "vcsel_ai_tuning.continuous_live",
            {
                "runtime_config": "runtime-v2.1-live.yaml",
                "campaign_id": "campaign-1",
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            tmp_path,
        )
    ]


def test_live_factory_requires_explicit_runtime_and_campaign_config(tmp_path) -> None:
    with pytest.raises(LiveContactAuthorizationError, match="live_config"):
        build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=tmp_path,
            allow_device_contact=True,
        )


def test_live_factory_rejects_removed_authorization_token_argument(tmp_path) -> None:
    with pytest.raises(TypeError, match="authorization_token"):
        build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=tmp_path,
            allow_device_contact=True,
            authorization_token="obsolete",
            live_config={
                "runtime_config": "runtime-v2.1-live.yaml",
                "campaign_id": "campaign-1",
            },
        )


def test_cli_defaults_to_fake_and_has_no_fir_bert_or_nlc_options(
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "FAKE",
                "max_rounds": 2,
                "result_dir": str(tmp_path / "results"),
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config), "--scenario", "happy"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["execution_mode"] == "FAKE"
    assert output["closed_loop_proven"] is False
    assert output["state"] == "COMPLETED"

    for forbidden in ("--set-fir", "--bert", "--nlc"):
        with pytest.raises(SystemExit):
            main(["--config", str(config), forbidden])


def test_cli_live_gate_fails_before_factory_contact(tmp_path, capsys) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump({"mode": "LIVE", "result_dir": str(tmp_path / "results")}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--config",
                str(config),
                "--max-rounds",
                "3",
                "--target-tdecq-db",
                "2.0",
            ]
        )
        == 2
    )
    assert "allow-device-contact" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("operator_args", "expected"),
    [
        ([], "--max-rounds and --target-tdecq-db"),
        (["--max-rounds", "3"], "--target-tdecq-db"),
        (["--target-tdecq-db", "2.8"], "--max-rounds"),
    ],
)
def test_cli_live_requires_operator_controls_before_factory(
    tmp_path,
    monkeypatch,
    capsys,
    operator_args,
    expected,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(yaml.safe_dump({"mode": "LIVE"}), encoding="utf-8")
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct")),
    )

    exit_code = main(["--config", str(config), *operator_args])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert expected in captured.err


def test_cli_live_operator_controls_override_stale_yaml_values(
    tmp_path,
    monkeypatch,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE",
                "max_rounds": 3,
                "target_tdecq_db": 9.9,
                "live": {
                    "runtime_config": "runtime-v2.1-live.yaml",
                    "campaign_id": "campaign-1",
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: captured.update(kwargs)
        or SimpleNamespace(
            live_activation=SimpleNamespace(preflight=lambda: {"operation": "PREFLIGHT"})
        ),
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "--allow-device-contact",
            "--max-rounds",
            "20",
            "--target-tdecq-db",
            "2.8",
            "--live-operation",
            "PREFLIGHT",
        ]
    )

    assert exit_code == 0
    assert captured["max_rounds"] == 20
    assert captured["target_tdecq_db"] == 2.8


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-rounds", "0"),
        ("--max-rounds", "-1"),
        ("--target-tdecq-db", "0"),
        ("--target-tdecq-db", "nan"),
        ("--target-tdecq-db", "inf"),
    ],
)
def test_cli_rejects_invalid_operator_control_values(
    tmp_path,
    monkeypatch,
    capsys,
    option,
    value,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(yaml.safe_dump({"mode": "FAKE"}), encoding="utf-8")
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct")),
    )

    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), option, value])
    captured = capsys.readouterr()

    assert error.value.code == 2
    assert option in captured.err
    assert "positive" in captured.err


def test_cli_live_with_device_contact_gate_invokes_lazy_activation(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE",
                "result_dir": str(tmp_path / "results"),
                "live": {
                    "runtime_config": "runtime-live.yaml",
                    "campaign_id": "campaign-1",
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class Activation:
        def activate(self):
            calls.append("activate")
            return type(
                "Activated",
                (),
                {
                    "controller": type(
                        "Controller",
                        (),
                            {"start": lambda self, request: type(
                                "Result",
                                (),
                                {
                                    "state": ContinuousState.COMPLETED,
                                    "model_dump": lambda self, mode: {
                                        "state": "COMPLETED",
                                        "execution_mode": "LIVE",
                                        "closed_loop_proven": True,
                                    },
                                },
                            )()},
                    )(),
                    "request": object(),
                    "shutdown": staticmethod(lambda: calls.append("shutdown")),
                },
            )()

    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: type(
            "Dependencies",
            (),
            {"live_activation": Activation(), "controller": None, "request": None},
        )(),
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "--allow-device-contact",
            "--max-rounds",
            "3",
            "--target-tdecq-db",
            "2.0",
            "--live-operation",
            "COMMISSION",
        ]
    )

    assert exit_code == 0
    assert calls == ["activate", "shutdown"]
    assert json.loads(capsys.readouterr().out)["execution_mode"] == "LIVE"


@pytest.mark.parametrize(
    "obsolete_option",
    ["--authorization-token", "--commission-authorization-token"],
)
def test_cli_rejects_removed_token_flags(tmp_path, obsolete_option) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(yaml.safe_dump({"mode": "FAKE"}), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), obsolete_option, "obsolete"])

    assert error.value.code == 2


def test_cli_commission_restores_b0_instead_of_exiting_awaiting_human(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE",
                "result_dir": str(tmp_path / "results"),
                "live": {
                    "runtime_config": "runtime-live.yaml",
                    "campaign_id": "campaign-1",
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class Result:
        def __init__(self, state):
            self.state = state

        def model_dump(self, mode):
            return {"state": self.state.value, "execution_mode": "LIVE"}

    class Controller:
        def start(self, request):
            calls.append("start")
            return Result(ContinuousState.AWAITING_HUMAN)

        def resolve_human(self, campaign_id, decision):
            calls.append(("resolve", campaign_id, decision.value))
            return Result(ContinuousState.COMPLETED)

    class Activation:
        def activate(self):
            calls.append("activate")
            return SimpleNamespace(
                controller=Controller(),
                request=SimpleNamespace(campaign_id="campaign-1"),
                shutdown=lambda: calls.append("shutdown"),
            )

    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: SimpleNamespace(live_activation=Activation()),
    )
    exit_code = main(
        [
            "--config",
            str(config),
            "--allow-device-contact",
            "--max-rounds",
            "3",
            "--target-tdecq-db",
            "2.0",
            "--live-operation",
            "COMMISSION",
        ]
    )

    assert exit_code == 0
    assert calls == [
        "activate",
        "start",
        ("resolve", "campaign-1", "RESTORE_B0"),
        "shutdown",
    ]
    assert json.loads(capsys.readouterr().out)["state"] == "COMPLETED"


@pytest.mark.parametrize(
    ("start_state", "resolved_state", "expected_state"),
    [
        (ContinuousState.FAILED_SAFE, None, ContinuousState.FAILED_SAFE),
        (
            ContinuousState.AWAITING_HUMAN,
            ContinuousState.RECOVERY_REQUIRED,
            ContinuousState.RECOVERY_REQUIRED,
        ),
    ],
)
def test_cli_commission_returns_nonzero_for_unsuccessful_terminal_result(
    tmp_path,
    monkeypatch,
    capsys,
    start_state,
    resolved_state,
    expected_state,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE",
                "result_dir": str(tmp_path / "results"),
                "live": {
                    "runtime_config": "runtime-live.yaml",
                    "campaign_id": "campaign-1",
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[object] = []

    class Result:
        def __init__(self, state):
            self.state = state

        def model_dump(self, mode):
            return {"state": self.state.value, "execution_mode": "LIVE"}

    class Controller:
        def start(self, request):
            calls.append("start")
            return Result(start_state)

        def resolve_human(self, campaign_id, decision):
            calls.append(("resolve", campaign_id, decision.value))
            return Result(resolved_state)

    class Activation:
        def activate(self):
            calls.append("activate")
            return SimpleNamespace(
                controller=Controller(),
                request=SimpleNamespace(campaign_id="campaign-1"),
                shutdown=lambda: calls.append("shutdown"),
            )

    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: SimpleNamespace(live_activation=Activation()),
    )
    exit_code = main(
        [
            "--config",
            str(config),
            "--allow-device-contact",
            "--max-rounds",
            "3",
            "--target-tdecq-db",
            "2.0",
            "--live-operation",
            "COMMISSION",
        ]
    )

    assert exit_code == 1
    assert calls[-1] == "shutdown"
    assert json.loads(capsys.readouterr().out)["state"] == expected_state.value


def test_cli_live_rejects_checked_in_campaign_placeholder_before_import(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE",
                "result_dir": str(tmp_path / "results"),
                "live": {
                    "runtime_config": "runtime-live.yaml",
                    "campaign_id": "REPLACE_WITH_FRESH_CAMPAIGN_ID",
                },
            }
        ),
        encoding="utf-8",
    )
    imports: list[str] = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: imports.append(name),
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "--allow-device-contact",
            "--max-rounds",
            "3",
            "--target-tdecq-db",
            "2.0",
            "--live-operation",
            "PREFLIGHT",
        ]
    )

    assert exit_code == 2
    assert "placeholder" in capsys.readouterr().err
    assert imports == []
