from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import yaml

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    AuthorizationTimePolicy,
    CampaignState,
    FirSettings,
    StageAuthorization,
    load_b0,
)
from vcsel_ai_tuning.campaign.orchestrator import (
    MeasurementResult,
    RecoveryRequiredError,
)
from vcsel_ai_tuning.campaign.repository import ArtifactSpec
from vcsel_ai_tuning.continuous_live import (
    _continuous_provider_config,
    _validated_scripted_schedule,
    activate_continuous_live,
    preflight_continuous_live,
)
from vcsel_ai_tuning.eye_image import derive_kimi_input_png
from vcsel_ai_tuning.hardware.live_runtime_factory import (
    build_continuous_live_tuning_profile,
)
from vcsel_ai_tuning.runtime import (
    AdapterBundle,
    DurableBenchLease,
    HardwareResourceBusyError,
)
from vcsel_ai_tuning.v2.continuous.models import (
    ContinuousRequest,
    ContinuousState,
    ExecutionMode,
    ProviderAction,
    ProviderActionKind,
    ReasonCode,
)
from vcsel_ai_tuning.v2.continuous.factory import (
    build_continuous_dependencies,
)
from vcsel_ai_tuning.v2.continuous.policy import (
    I2cParameterPolicy,
    build_restricted_policy,
)
from vcsel_ai_tuning.v2.continuous.store import ContinuousCampaignStore
from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)


def _policy_config() -> dict[str, object]:
    return {
        "version": "i2c-restricted-v1",
        "max_changed_parameters": 3,
        "rules": {
            "ibias_dac": {"minimum": 60, "maximum": 80, "max_step": 10},
            "efin_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
            "vga_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
            "eq": {"minimum": 16, "maximum": 25, "max_step": 3},
            "res_gain": {"minimum": 0, "maximum": 10, "max_step": 3},
            "res_bw": {"minimum": 0, "maximum": 5, "max_step": 1},
            "gain_ctrl": {"minimum": 0, "maximum": 10, "max_step": 3},
            "vbias_ref": {"minimum": 10, "maximum": 15, "max_step": 3},
            "vref": {"minimum": 10, "maximum": 15, "max_step": 3},
        },
    }


def _profile_path() -> Path:
    return Path("configs/hardware/vcsel_lab_v2_1_continuous.yaml").resolve()


def _continuous_b0_path() -> Path:
    return Path("configs/ai_tuning/b0.v2_1.continuous.yaml").resolve()


def _eye(tmp_path: Path) -> EyeSample:
    payload = b"fake-eye"
    digest = sha256(payload).hexdigest()
    relative = "eye_artifacts/preflight.png"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    reference = ArtifactReference(
        relative_path=relative,
        sha256=digest,
        media_type="image/png",
    )
    return EyeSample(
        measurement_id="preflight-baseline-1",
        campaign_id="campaign-1",
        sweep_point_id="preflight-baseline-1",
        raw_eye_density_ref=reference,
        metrics=EyeMetrics(tdecq_db=3.1, outer_er_chan1a_db=4.5),
        requested_params={},
        applied_params={},
        readback_params={"eq": 22},
        dca_config_fingerprint="a" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        artifact_hashes={relative: digest},
    )


def test_live_preflight_requires_explicit_parameter_policy_before_contact(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Kimi contacted")),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hardware runtime opened")
        ),
    )

    with pytest.raises(RuntimeError, match="parameter_policy"):
        preflight_continuous_live(
            {"runtime_config": "runtime.yaml", "campaign_id": "campaign-1"},
            tmp_path,
        )


def test_checked_in_v21_live_config_has_approved_b0_and_policy() -> None:
    document = yaml.safe_load(
        Path("configs/ai_tuning/v2_1_continuous.live.yaml").read_text(
            encoding="utf-8"
        )
    )
    live = document["live"]
    runtime_path = Path(live["runtime_config"])
    runtime_document = yaml.safe_load(
        runtime_path.read_text(encoding="utf-8")
    )
    baseline_path = (runtime_path.parent / runtime_document["b0_path"]).resolve()
    baseline = load_b0(baseline_path)
    profile_path = (
        runtime_path.parent / runtime_document["hardware"]["driver_profile"]
    ).resolve()
    profile = build_continuous_live_tuning_profile(profile_path, baseline)
    policy = build_restricted_policy(
        live["parameter_policy"],
        authorization_specs=profile.parameter_specs,
        baseline_i2c=baseline.i2c.model_dump(),
    )

    forbidden_runtime_keys = {
        "max_rounds",
        "target_tdecq_db",
        "commission_max_duration_s",
    }

    def assert_no_fixed_runtime_controls(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_runtime_keys.isdisjoint(value)
            for nested_value in value.values():
                assert_no_fixed_runtime_controls(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                assert_no_fixed_runtime_controls(nested_value)

    instruction = live["instruction"]
    numeric_tdecq_goal = re.compile(
        r"(?i)\b(?:seek|target|goal|tdecq)\b[^.\n]{0,50}"
        r"(?:below|under|less than|<|of|to)\s*\d+(?:\.\d+)?(?:\s*dB)?"
    )

    assert_no_fixed_runtime_controls(document)
    assert "scripted live demonstration" in instruction
    assert "real DCA measurements" in instruction
    assert "never invent TDECQ" in instruction
    assert "HISTORICAL_DATA_BEGIN" not in instruction
    assert numeric_tdecq_goal.search("seek TDECQ below 2.0 dB") is not None
    assert numeric_tdecq_goal.search(instruction) is None
    assert baseline.version == 4
    assert baseline_path.name == "b0.v2_1.continuous.yaml"
    assert document["mode"] == "LIVE"
    assert live["runtime_config"] == "configs/ai_tuning/runtime.v2_1.continuous.live.yaml"
    assert live["campaign_id"] == "REPLACE_WITH_FRESH_CAMPAIGN_ID"
    assert re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", live["campaign_id"])
    assert live["kimi_model"] == "kimi-k3"
    assert live["parameter_policy"] == _policy_config()
    assert baseline.i2c.model_dump() == {
        "ibias_dac": 62,
        "vref": 15,
        "vbias_ref": 15,
        "vga_cur": 13,
        "res_gain": 6,
        "res_bw": 0,
        "gain_ctrl": 2,
        "efin_cur": 15,
        "eq": 16,
    }
    assert policy.version == "i2c-restricted-v1"
    assert policy.max_changed_parameters == 3

    schedule = live["scripted_schedule"]
    assert len(schedule) == 20
    expected_keys = set(baseline.i2c.model_dump())
    previous = baseline.i2c.model_dump()
    rules = live["parameter_policy"]["rules"]
    for round_index, point in enumerate(schedule, start=1):
        assert set(point) == expected_keys
        assert "tdecq_db" not in point
        assert "er_db" not in point
        changed = [name for name in expected_keys if point[name] != previous[name]]
        assert 1 <= len(changed) <= 3, round_index
        for name in changed:
            assert abs(point[name] - previous[name]) <= rules[name]["max_step"]
        previous = point
    assert schedule[17] == {
        "ibias_dac": 70,
        "efin_cur": 15,
        "vga_cur": 12,
        "eq": 21,
        "res_gain": 9,
        "res_bw": 1,
        "gain_ctrl": 7,
        "vbias_ref": 15,
        "vref": 15,
    }


def _signed_live_launcher() -> str:
    return Path("scripts/run_v2_1_continuous_live.ps1").read_text(encoding="utf-8")


def test_scripted_schedule_length_must_match_operator_rounds() -> None:
    point = {
        "ibias_dac": 66,
        "efin_cur": 15,
        "vga_cur": 13,
        "eq": 18,
        "res_gain": 6,
        "res_bw": 0,
        "gain_ctrl": 3,
        "vbias_ref": 15,
        "vref": 15,
    }

    with pytest.raises(RuntimeError, match="must equal max_rounds"):
        _validated_scripted_schedule(
            {"max_rounds": 2, "scripted_schedule": [point]}
        )


def test_scripted_provider_provenance_is_explicit() -> None:
    provider = SimpleNamespace(provider_id="moonshot-kimi", provider_version="kimi-k3")

    assert _continuous_provider_config(provider, ({"eq": 19},) * 20) == {
        "provider_id": "moonshot-kimi",
        "provider_version": "kimi-k3",
        "decision_mode": "SCRIPTED_LIVE",
        "scripted_rounds": 20,
    }


def test_signed_live_launcher_has_only_approved_operator_inputs() -> None:
    launcher = _signed_live_launcher()
    prompts = re.findall(r"Read-Host\s+'([^']+)'", launcher)

    assert launcher.startswith("#requires -Version 7.0\n")
    assert len(prompts) == 4
    assert any("Max optimization rounds" in prompt for prompt in prompts)
    assert any("Target TDECQ" in prompt for prompt in prompts)
    assert any("MOONSHOT_API_KEY" in prompt for prompt in prompts)
    assert any("COMMISSION approval" in prompt for prompt in prompts)
    assert "Retype" not in launcher
    assert "Notepad" not in launcher
    assert "authorization-token" not in launcher
    assert "commission-authorization-token" not in launcher
    assert "REPLACE_WITH_FRESH_CAMPAIGN_ID" in launcher
    assert "[guid]::NewGuid()" in launcher
    assert "v21-b-{0}-{1}" in launcher


def test_signed_live_launcher_reuses_frozen_scope_for_both_operations() -> None:
    launcher = _signed_live_launcher()

    assert launcher.count("--max-rounds $SessionMaxRounds") == 2
    assert launcher.count("--target-tdecq-db $SessionTargetTdecqDb") == 2
    assert launcher.count("--config $SessionLiveConfigPath") == 2
    assert launcher.count("--allow-device-contact") == 2
    assert "Assert-ContinuousPreflightReport" in launcher
    assert "$PreflightExitCode -ne 0" in launcher
    assert "--live-operation PREFLIGHT" in launcher
    assert "--live-operation COMMISSION" in launcher
    assert "SessionHead" in launcher
    assert "SessionConfigManifest" in launcher
    assert "Assert-ContinuousSessionScope" in launcher
    assert "-Option ReadOnly" in launcher


def test_signed_live_launcher_materializes_only_a_fresh_campaign_id() -> None:
    launcher = _signed_live_launcher()

    assert "$PlaceholderCount -ne 1" in launcher
    assert "$TemplateText.Replace($CampaignPlaceholder, $SessionCampaignId)" in launcher
    assert "Copy-Item" not in launcher
    assert "Set-Content" in launcher
    assert launcher.count("$SessionConfigText =") == 1
    assert launcher.count("$TemplateText.Replace(") == 1
    assert "yaml.safe_dump" not in launcher


def test_signed_live_launcher_audits_preflight_before_commission_gate() -> None:
    launcher = _signed_live_launcher()

    preflight_assert = launcher.index("Assert-ContinuousPreflightReport -Report")
    commission_gate = launcher.index("PRE-FLIGHT passed; press Enter after COMMISSION approval")
    commission_call = launcher.index("--live-operation COMMISSION")
    assert preflight_assert < commission_gate < commission_call
    assert "operation -cne 'PREFLIGHT'" in launcher
    assert "dut_write_performed -ne $false" in launcher
    assert "bert_contacted -ne $false" in launcher
    assert "dca_acquisition_commands_performed -ne $true" in launcher
    assert "hardware_write_performed -ne $false" in launcher
    assert "NO_DUT_REGISTER_OR_BERT_WRITE" in launcher


def test_signed_live_launcher_cleans_only_session_secrets_and_temp_config() -> None:
    launcher = _signed_live_launcher()

    assert "} finally {" in launcher
    assert "Remove-Item -LiteralPath $SessionLiveConfigPath -Force" in launcher
    assert "Remove-Item Env:MOONSHOT_API_KEY -ErrorAction Stop" in launcher
    assert "Invoke-ContinuousSessionCleanup" in launcher
    assert "preserving the primary error" in launcher
    for evidence_root in ("artifacts", "outputs", "state"):
        assert f"Remove-Item {evidence_root}" not in launcher


def _run_signed_launcher_helper_smoke(source: str) -> subprocess.CompletedProcess[str]:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell 7 is not installed")
    return subprocess.run(
        [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", source],
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_signed_live_launcher_generates_unique_safe_campaign_ids() -> None:
    assert "param([switch]$LoadHelpersOnly)" in _signed_live_launcher()
    launcher_path = Path("scripts/run_v2_1_continuous_live.ps1").resolve()
    completed = _run_signed_launcher_helper_smoke(
        f"""
. '{str(launcher_path).replace("'", "''")}' -LoadHelpersOnly
$first = New-ContinuousCampaignId
$second = New-ContinuousCampaignId
$pattern = '^v21-b-[0-9]{{8}}T[0-9]{{9}}Z-[0-9a-f]{{8}}$'
if ($first -cnotmatch $pattern -or $second -cnotmatch $pattern) {{ throw 'unsafe campaign ID' }}
if ($first -ceq $second) {{ throw 'campaign IDs were reused' }}
[Console]::Out.Write("$first`n$second")
"""
    )

    assert completed.returncode == 0, completed.stderr
    campaign_ids = completed.stdout.splitlines()
    assert len(campaign_ids) == 2
    assert campaign_ids[0] != campaign_ids[1]
    assert all(
        re.fullmatch(r"v21-b-[0-9]{8}T[0-9]{9}Z-[0-9a-f]{8}", value)
        for value in campaign_ids
    )


def test_signed_live_launcher_cleanup_preserves_primary_error_and_clears_key() -> None:
    assert "param([switch]$LoadHelpersOnly)" in _signed_live_launcher()
    launcher_path = Path("scripts/run_v2_1_continuous_live.ps1").resolve()
    completed = _run_signed_launcher_helper_smoke(
        f"""
. '{str(launcher_path).replace("'", "''")}' -LoadHelpersOnly
$tempDirectory = Join-Path ([IO.Path]::GetTempPath()) ('vcsel-cleanup-test-' + [guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $tempDirectory)
Set-Content -LiteralPath (Join-Path $tempDirectory 'child.txt') -Value 'force non-recursive removal failure'
$env:MOONSHOT_API_KEY = 'test-only-secret'
$primaryError = $null
try {{ throw 'PRIMARY_SENTINEL' }} catch {{ $primaryError = $_ }}
try {{
    Invoke-ContinuousSessionCleanup -SessionLiveConfigPath $tempDirectory -PrimaryError $primaryError
    if (-not (Test-Path -LiteralPath $tempDirectory)) {{ throw 'cleanup fixture did not fail' }}
    throw $primaryError
}} catch {{
    if ($_.Exception.Message -cnotmatch 'PRIMARY_SENTINEL') {{ throw 'cleanup replaced primary error' }}
}} finally {{
    if (Test-Path Env:MOONSHOT_API_KEY) {{ throw 'API key survived cleanup' }}
    Remove-Item -LiteralPath $tempDirectory -Recurse -Force -ErrorAction SilentlyContinue
}}
[Console]::Out.Write('CLEANUP_ERROR_PATH_OK')
"""
    )

    assert completed.returncode == 0, completed.stderr
    assert "CLEANUP_ERROR_PATH_OK" in completed.stdout


def test_signed_live_launcher_validates_controls_before_campaign_materialization() -> None:
    launcher = _signed_live_launcher()

    round_validation = launcher.index(
        "$ValidatedMaxRoundsText = ConvertTo-PositiveDecimalInteger"
    )
    target_validation = launcher.index(
        "$ValidatedTargetTdecq = ConvertTo-PositiveInvariantDouble"
    )
    campaign_generation = launcher.index("$GeneratedCampaignId = New-ContinuousCampaignId")
    config_materialization = launcher.index("$SessionConfigText = $TemplateText.Replace")
    assert round_validation < target_validation < campaign_generation < config_materialization


def test_signed_live_launcher_checks_commission_exit_before_json_parse() -> None:
    launcher = _signed_live_launcher()

    exit_capture = launcher.index("$CommissionExitCode = $LASTEXITCODE")
    exit_guard = launcher.index("if ($CommissionExitCode -ne 0)", exit_capture)
    json_parse = launcher.index(
        "$CommissionResult = ConvertFrom-ContinuousJsonResult", exit_capture
    )
    assert exit_capture < exit_guard < json_parse


def test_signed_live_launcher_uses_isolated_python_safe_path() -> None:
    launcher = _signed_live_launcher()

    assert launcher.count("py -3.13") == 3
    assert launcher.count("py -3.13 -P") == 3
    assert "py -3.13 -c" not in launcher
    assert launcher.index("$env:PYTHONPATH = (Resolve-Path 'src').Path") < launcher.index(
        "$InitialConfigManifest = Get-ContinuousConfigManifest"
    )


def _initialize_scope_smoke_repo(repo: Path) -> Path:
    repo.mkdir()
    (repo / "live.yaml").write_text(
        "live:\n  runtime_config: runtime.yaml\n", encoding="utf-8"
    )
    (repo / "runtime.yaml").write_text(
        "hardware:\n  driver_profile: hardware.yaml\nb0_path: b0.yaml\n",
        encoding="utf-8",
    )
    (repo / "hardware.yaml").write_text("profile: smoke\n", encoding="utf-8")
    (repo / "b0.yaml").write_text("version: 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored_*.py\n", encoding="utf-8")
    for git_args in (
        ("init", "-b", "scope-smoke"),
        ("config", "user.email", "scope-smoke@example.invalid"),
        ("config", "user.name", "Scope Smoke"),
        ("add", "."),
        ("commit", "-m", "scope fixture"),
    ):
        completed = subprocess.run(
            ["git", *git_args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    session_config = repo.parent / f"{repo.name}-session.yaml"
    session_config.write_text(
        (repo / "live.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return session_config


@pytest.mark.parametrize(
    ("drift", "mutation", "expected_diagnostic"),
    [
        ("branch", "& git switch -c scope-drift *> $null", "Branch changed"),
        ("head", "& git commit --allow-empty -m scope-drift *> $null", "HEAD changed"),
        (
            "tracked",
            "Add-Content -LiteralPath hardware.yaml -Value 'drift: true'",
            "Tracked worktree changed",
        ),
        (
            "untracked",
            "Set-Content -LiteralPath rogue.py -Value 'print(1)'",
            "Unauthorized untracked path",
        ),
        (
            "ignored_untracked",
            "Set-Content -LiteralPath ignored_rogue.py -Value 'print(1)'",
            "Sensitive ignored path",
        ),
        (
            "manifest",
            "Add-Content -LiteralPath $SessionLiveConfigPath -Value '# drift'",
            "manifest changed",
        ),
    ],
)
def test_signed_live_scope_gate_dynamically_rejects_drift(
    tmp_path: Path,
    drift: str,
    mutation: str,
    expected_diagnostic: str,
) -> None:
    repo = tmp_path / f"scope-{drift}"
    session_config = _initialize_scope_smoke_repo(repo)
    launcher_path = Path("scripts/run_v2_1_continuous_live.ps1").resolve()
    ps_launcher = str(launcher_path).replace("'", "''")
    ps_repo = str(repo.resolve()).replace("'", "''")
    ps_session = str(session_config.resolve()).replace("'", "''")
    completed = _run_signed_launcher_helper_smoke(
        f"""
. '{ps_launcher}' -LoadHelpersOnly
Set-Location -LiteralPath '{ps_repo}'
$ExpectedBranch = 'scope-smoke'
$SessionHead = (& git rev-parse HEAD).Trim()
$SessionLiveConfigPath = '{ps_session}'
$SessionConfigManifest = Get-ContinuousConfigManifest -LiveConfigPath $SessionLiveConfigPath
{mutation}
if ($LASTEXITCODE -ne 0) {{ throw 'scope mutation setup failed' }}
$rejection = $null
try {{ Assert-ContinuousSessionScope }} catch {{ $rejection = $_.Exception.Message }}
if ($null -eq $rejection) {{ throw 'scope gate accepted {drift} drift' }}
if ($rejection -cnotmatch '{expected_diagnostic}') {{ throw "wrong scope rejection: $rejection" }}
[Console]::Out.Write('SCOPE_REJECTED')
"""
    )

    assert completed.returncode == 0, completed.stderr
    assert "SCOPE_REJECTED" in completed.stdout


def test_signed_live_scope_gate_allows_only_evidence_directories(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "scope-evidence"
    session_config = _initialize_scope_smoke_repo(repo)
    for evidence_root in ("artifacts", "outputs", "state"):
        path = repo / evidence_root / "nested" / "evidence.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("preserved: true\n", encoding="utf-8")
    launcher_path = Path("scripts/run_v2_1_continuous_live.ps1").resolve()
    ps_launcher = str(launcher_path).replace("'", "''")
    ps_repo = str(repo.resolve()).replace("'", "''")
    ps_session = str(session_config.resolve()).replace("'", "''")
    completed = _run_signed_launcher_helper_smoke(
        f"""
. '{ps_launcher}' -LoadHelpersOnly
Set-Location -LiteralPath '{ps_repo}'
$ExpectedBranch = 'scope-smoke'
$SessionHead = (& git rev-parse HEAD).Trim()
$SessionLiveConfigPath = '{ps_session}'
$SessionConfigManifest = Get-ContinuousConfigManifest -LiveConfigPath $SessionLiveConfigPath
Assert-ContinuousSessionScope
[Console]::Out.Write('EVIDENCE_SCOPE_OK')
"""
    )

    assert completed.returncode == 0, completed.stderr
    assert "EVIDENCE_SCOPE_OK" in completed.stdout


def test_continuous_live_runbook_uses_only_the_signed_launcher_entry() -> None:
    runbook = Path("docs/v2-1-continuous-eye-testbench.md").read_text(
        encoding="utf-8"
    )

    assert (
        "pwsh -NoLogo -NoProfile -File .\\scripts\\run_v2_1_continuous_live.ps1"
        in runbook
    )
    assert "Read-Host" not in runbook
    assert "scripts/run_continuous_eye.py" not in runbook
    assert "--authorization-token" not in runbook
    assert "--commission-authorization-token" not in runbook
    assert "edit the temporary" not in runbook.lower()
    assert "retype" not in runbook.lower()


def test_continuous_live_runbook_documents_inputs_safety_and_dca_diagnostics() -> None:
    runbook = Path("docs/v2-1-continuous-eye-testbench.md").read_text(
        encoding="utf-8"
    )

    assert "Campaign ID" in runbook and "automatically" in runbook
    assert "maximum optimization rounds" in runbook
    assert "target TDECQ" in runbook
    assert "secure" in runbook and "MOONSHOT_API_KEY" in runbook
    assert "COMMISSION Enter gate" in runbook
    assert "No authorization token is entered" in runbook
    for stage in (
        "SCREEN_WINDOW_SELECT",
        "SCREEN_FILENAME",
        "SCREEN_SAVE",
        "PNG_BINARY_READ",
        "FINGERPRINT_QUERY",
    ):
        assert stage in runbook
    assert "不得删除 `artifacts/`、`outputs/`、`state/` 或任何证据" in runbook
    assert "V1.1-derived V1.2" in runbook
    assert "FAST24/FIR-writing V1.2" in runbook


def test_lazy_live_activation_accepts_operator_scope_through_upper_layer(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[object] = []
    kimi_client_kwargs: list[dict[str, object]] = []

    class Orchestrator:
        baseline = load_b0(_continuous_b0_path())
        store = SimpleNamespace(current_state=lambda campaign_id: CampaignState.I2C_SCREENING)

        def prepare_continuous_commission(self, **kwargs):
            calls.append(("prepare", kwargs))
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            return SimpleNamespace(
                observation=object(),
                authorization=StageAuthorization(
                    authorization_id="auth-1",
                    campaign_id="campaign-1",
                    mode=AuthorizationMode.PER_STAGE,
                    time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
                    stage=CampaignState.I2C_SCREENING,
                    baseline_hash="b" * 64,
                    rollback_hash="c" * 64,
                    max_candidates=20,
                    max_duration_s=None,
                    issued_at=now,
                    expires_at=None,
                    rollback_snapshot=AppliedSettings(readback_i2c={"eq": 22}),
                ),
            )

        def restore_static_i2c_b0(self):
            calls.append("restore")

        def recover_continuous_commission_failure(self):
            calls.append("failure_recovery")
            return AppliedSettings(readback_i2c=self.baseline.i2c.model_dump())

        def run_continuous_i2c_candidate(self, *args):
            raise AssertionError("not executed during assembly")

        def measure_current_i2c_state(self):
            raise AssertionError("not executed during assembly")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
        adapters=SimpleNamespace(
            dca=SimpleNamespace(
                preflight=lambda: {
                    "identity_verified": True,
                    "configuration_verified": True,
                }
            ),
            i2c=SimpleNamespace(
                preflight=lambda: {
                    "identity_verified": True,
                    "configuration_verified": True,
                }
            ),
        ),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            b0_path=_continuous_b0_path(),
            hardware=SimpleNamespace(driver_profile=_profile_path()),
        ),
        entry=lambda campaign_id: entry,
    )
    api_runtime = SimpleNamespace(
        service=SimpleNamespace(runtime=runtime),
        shutdown=lambda: calls.append("shutdown"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: api_runtime,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: (
            kimi_client_kwargs.append(kwargs)
            or SimpleNamespace(model="fake-kimi")
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(to_eye_sample=lambda observation: _eye(tmp_path)),
    )
    lease_instances = []

    class GuardedLease:
        def __init__(self, sqlite_path):
            self.sqlite_path = sqlite_path
            self.guard_calls = 0
            lease_instances.append(self)

        def assert_available_for_contact(self, campaign_id):
            self.guard_calls += 1
            self.campaign_id = campaign_id

    monkeypatch.setattr("vcsel_ai_tuning.runtime.DurableBenchLease", GuardedLease)
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=20,
        target_tdecq_db=2.8,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime.yaml",
            "campaign_id": "campaign-1",
            "parameter_policy": _policy_config(),
        },
    )

    assert dependencies.live_activation is not None
    activated = dependencies.live_activation.activate()

    assert activated.request.max_rounds == 20
    assert activated.request.stopping_policy.target_tdecq_db == 2.8
    assert type(activated.request) is ContinuousRequest
    assert not hasattr(activated.request, "commission_deadline")
    assert len(lease_instances) == 1
    assert lease_instances[0].guard_calls == 1
    assert lease_instances[0].campaign_id == "campaign-1"
    assert activated.controller._bench_lease is lease_instances[0]
    assert kimi_client_kwargs == [
        {
            "model": "kimi-k3",
            "response_log_path": (
                tmp_path / "campaign-1" / "kimi-response-attempts.jsonl"
            ),
        }
    ]
    assert calls == [("prepare", {"max_candidates": 20})]
    assert callable(activated.shutdown)
    activated.shutdown()
    assert calls == [
        ("prepare", {"max_candidates": 20}),
        "failure_recovery",
        "shutdown",
    ]


@pytest.mark.parametrize("lease_state", ["stale", "unreadable"])
def test_live_commission_rejects_unprovable_lease_before_any_contact(
    tmp_path,
    monkeypatch,
    lease_state,
) -> None:
    calls: list[str] = []
    sqlite_path = tmp_path / "state" / "campaigns.sqlite3"
    lease = DurableBenchLease(sqlite_path)
    if lease_state == "stale":
        lease.acquire(
            campaign_id="previous-campaign",
            owner_id="previous-owner",
            resource_ids=("vcsel-bench",),
            issued_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
            expires_at=datetime(2026, 8, 12, 0, 15, tzinfo=timezone.utc),
        )
    else:
        lease.path.write_text("{unreadable", encoding="utf-8")

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=sqlite_path),
        ),
        entry=lambda campaign_id: (_ for _ in ()).throw(
            AssertionError("runtime.entry contacted adapters")
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=SimpleNamespace(runtime=runtime),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Kimi contacted")),
    )

    with pytest.raises(HardwareResourceBusyError):
        activate_continuous_live(
            {
                "runtime_config": "runtime.yaml",
                "campaign_id": "campaign-1",
                "parameter_policy": _policy_config(),
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            tmp_path,
        )

    assert calls == ["shutdown"]


@pytest.mark.parametrize("lease_state", ["stale", "unreadable"])
def test_live_preflight_rejects_unprovable_lease_before_any_contact(
    tmp_path,
    monkeypatch,
    lease_state,
) -> None:
    calls: list[str] = []
    sqlite_path = tmp_path / "state" / "campaigns.sqlite3"
    lease = DurableBenchLease(sqlite_path)
    if lease_state == "stale":
        lease.acquire(
            campaign_id="previous-campaign",
            owner_id="previous-owner",
            resource_ids=("vcsel-bench",),
            issued_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
            expires_at=datetime(2026, 8, 12, 0, 15, tzinfo=timezone.utc),
        )
    else:
        lease.path.write_text("{unreadable", encoding="utf-8")

    forbidden = lambda message: (_ for _ in ()).throw(AssertionError(message))
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=sqlite_path),
        ),
        entry=lambda campaign_id: forbidden("runtime.entry contacted adapters"),
    )
    service = SimpleNamespace(
        runtime=runtime,
        create_campaign=lambda request: forbidden("campaign creation ran"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=service,
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: forbidden("Kimi client constructed"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: forbidden("observation adapter constructed"),
    )

    with pytest.raises(HardwareResourceBusyError):
        preflight_continuous_live(
            {
                "runtime_config": "runtime.yaml",
                "campaign_id": "campaign-1",
                "parameter_policy": _policy_config(),
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            tmp_path,
        )

    assert calls == ["shutdown"]


@pytest.mark.parametrize(
    (
        "campaign_state",
        "run_controller",
        "transition_failures",
        "failure_state",
        "expected_calls",
        "expected_final_state",
        "shutdown_error",
    ),
    [
        (
            CampaignState.I2C_SCREENING,
            True,
            0,
            None,
            [("b0_apply", 1), ("b0_readback", 1), "shutdown"],
            CampaignState.FAILED,
            False,
        ),
        (
            CampaignState.I2C_SCREENING,
            True,
            1,
            CampaignState.ABORTING,
            [
                ("b0_apply", 1),
                ("b0_readback", 1),
                ("b0_apply", 2),
                ("b0_readback", 2),
                "shutdown",
            ],
            CampaignState.FAILED,
            False,
        ),
        (
            CampaignState.I2C_SCREENING,
            True,
            1,
            CampaignState.ROLLING_BACK,
            [
                ("b0_apply", 1),
                ("b0_readback", 1),
                ("b0_apply", 2),
                ("b0_readback", 2),
                "shutdown",
            ],
            CampaignState.FAILED,
            False,
        ),
        (
            CampaignState.I2C_SCREENING,
            True,
            2,
            CampaignState.ROLLING_BACK,
            [
                ("b0_apply", 1),
                ("b0_readback", 1),
                ("b0_apply", 2),
                ("b0_readback", 2),
                "shutdown",
            ],
            CampaignState.ROLLING_BACK,
            True,
        ),
        (
            CampaignState.FAILED,
            False,
            0,
            None,
            ["shutdown"],
            CampaignState.FAILED,
            False,
        ),
        (
            CampaignState.RECOVERY_REQUIRED,
            False,
            0,
            None,
            ["shutdown"],
            CampaignState.RECOVERY_REQUIRED,
            False,
        ),
    ],
)
def test_live_shutdown_resumes_incomplete_failure_recovery_after_persistence_failure(
    tmp_path,
    monkeypatch,
    campaign_state,
    run_controller,
    transition_failures,
    failure_state,
    expected_calls,
    expected_final_state,
    shutdown_error,
) -> None:
    calls: list[object] = []
    eye = _eye(tmp_path)
    state = {"value": campaign_state}

    class Orchestrator:
        baseline = load_b0(_continuous_b0_path())
        recovery_attempts = 0
        store = SimpleNamespace(
            current_state=lambda campaign_id: state["value"]
        )

        def prepare_continuous_commission(self, **kwargs):
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            return SimpleNamespace(
                observation=object(),
                authorization=StageAuthorization(
                    authorization_id="auth-1",
                    campaign_id="campaign-1",
                    mode=AuthorizationMode.PER_STAGE,
                    time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
                    stage=CampaignState.I2C_SCREENING,
                    baseline_hash="b" * 64,
                    rollback_hash="c" * 64,
                    max_candidates=3,
                    max_duration_s=None,
                    issued_at=now,
                    expires_at=None,
                    rollback_snapshot=AppliedSettings(readback_i2c={"eq": 22}),
                ),
            )

        def run_continuous_i2c_candidate(self, *args):
            raise AssertionError("not executed during assembly")

        def measure_current_i2c_state(self):
            raise AssertionError("not executed during assembly")

        def restore_static_i2c_b0(self):
            calls.append("normal_restore")

        def recover_continuous_commission_failure(self):
            self.recovery_attempts += 1
            calls.append(("b0_apply", self.recovery_attempts))
            calls.append(("b0_readback", self.recovery_attempts))
            if self.recovery_attempts <= transition_failures:
                if failure_state is not None:
                    state["value"] = failure_state
                raise RuntimeError("injected recovery transition persistence failure")
            state["value"] = CampaignState.FAILED
            return AppliedSettings(readback_i2c=self.baseline.i2c.model_dump())

    orchestrator = Orchestrator()
    entry = SimpleNamespace(
        orchestrator=orchestrator,
        repository=SimpleNamespace(run_dir=tmp_path),
        adapters=SimpleNamespace(
            dca=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
            i2c=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
        ),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            b0_path=_continuous_b0_path(),
            hardware=SimpleNamespace(driver_profile=_profile_path()),
        ),
        entry=lambda campaign_id: entry,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=SimpleNamespace(runtime=runtime),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(model="fake-kimi"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(to_eye_sample=lambda observation: eye),
    )
    dependencies = activate_continuous_live(
        {
            "runtime_config": "runtime.yaml",
            "campaign_id": "campaign-1",
            "parameter_policy": _policy_config(),
            "max_rounds": 3,
            "target_tdecq_db": 2.0,
        },
        tmp_path,
    )
    policy_payload = dict(dependencies.request.policy)
    policy_payload["rules"] = tuple(policy_payload["rules"])
    effective = I2cParameterPolicy.model_validate(policy_payload)
    assert type(dependencies.request) is ContinuousRequest
    assert not hasattr(dependencies.request, "commission_deadline")
    assert effective.version == "i2c-restricted-v1"
    assert effective.by_name["ibias_dac"].tuning_min == 60
    assert effective.by_name["ibias_dac"].tuning_max == 80
    assert effective.by_name["gain_ctrl"].max_step == 3
    if run_controller:
        class FailingRunningStateStore(ContinuousCampaignStore):
            def save_state(self, campaign_id, payload):
                if payload.get("state") == ContinuousState.RUNNING.value:
                    raise RuntimeError("injected running state failure")
                return super().save_state(campaign_id, payload)

        dependencies.controller._store = FailingRunningStateStore(
            tmp_path / "failing-controller-state"
        )
        expected_error = (
            "RECOVERY_REQUIRED"
            if transition_failures
            else "running state failure"
        )
        with pytest.raises(RuntimeError, match=expected_error):
            dependencies.controller.start(dependencies.request)
    if shutdown_error:
        with pytest.raises(RuntimeError, match="transition persistence failure"):
            dependencies.shutdown()
    else:
        dependencies.shutdown()

    assert calls == expected_calls
    assert state["value"] is expected_final_state
    assert state["value"] is not CampaignState.COMPLETED


def test_live_commission_shutdown_resolves_human_wait_before_runtime_close(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[object] = []
    eye = _eye(tmp_path)

    class Orchestrator:
        baseline = load_b0(_continuous_b0_path())
        store = SimpleNamespace(current_state=lambda campaign_id: CampaignState.COMPLETED)

        def prepare_continuous_commission(self, **kwargs):
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            return SimpleNamespace(
                observation=object(),
                authorization=StageAuthorization(
                    authorization_id="auth-1",
                    campaign_id="campaign-1",
                    mode=AuthorizationMode.PER_STAGE,
                    time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
                    stage=CampaignState.I2C_SCREENING,
                    baseline_hash="b" * 64,
                    rollback_hash="c" * 64,
                    max_candidates=3,
                    max_duration_s=None,
                    issued_at=now,
                    expires_at=None,
                    rollback_snapshot=AppliedSettings(readback_i2c={"eq": 22}),
                ),
            )

        def run_continuous_i2c_candidate(self, *args):
            raise AssertionError("not executed during assembly")

        def measure_current_i2c_state(self):
            raise AssertionError("not executed during assembly")

        def restore_static_i2c_b0(self):
            calls.append("restore")

        def recover_continuous_commission_failure(self):
            raise AssertionError("failure recovery used during successful assembly")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
        adapters=SimpleNamespace(
            dca=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
            i2c=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
        ),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            b0_path=_continuous_b0_path(),
            hardware=SimpleNamespace(driver_profile=_profile_path()),
        ),
        entry=lambda campaign_id: entry,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=SimpleNamespace(runtime=runtime),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(model="fake-kimi"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(to_eye_sample=lambda observation: eye),
    )
    dependencies = activate_continuous_live(
        {
            "runtime_config": "runtime.yaml",
            "campaign_id": "campaign-1",
            "parameter_policy": _policy_config(),
            "max_rounds": 3,
            "target_tdecq_db": 2.0,
        },
        tmp_path,
    )
    dependencies.controller.get_status = lambda campaign_id: SimpleNamespace(
        state=ContinuousState.AWAITING_HUMAN
    )
    dependencies.controller.resolve_human = lambda campaign_id, decision: calls.append(
        ("resolve", campaign_id, decision.value)
    )

    dependencies.shutdown()

    assert calls == [("resolve", "campaign-1", "RESTORE_B0"), "shutdown"]


@pytest.mark.parametrize(
    "campaign_state",
    [
        CampaignState.DRAFT,
        CampaignState.PREFLIGHT,
        CampaignState.AWAIT_BASELINE_APPROVAL,
        CampaignState.FAILED,
        CampaignState.RECOVERY_REQUIRED,
        CampaignState.COMPLETED,
    ],
)
def test_live_commission_preparation_failure_skips_inapplicable_recovery(
    tmp_path,
    monkeypatch,
    campaign_state,
) -> None:
    calls: list[str] = []

    class Orchestrator:
        baseline = load_b0(_continuous_b0_path())
        store = SimpleNamespace(
            current_state=lambda campaign_id: campaign_state
        )

        def prepare_continuous_commission(self, **kwargs):
            calls.append("prepare")
            raise RuntimeError("injected preparation failure")

        def recover_continuous_commission_failure(self):
            calls.append("failure_recovery")

        def restore_static_i2c_b0(self):
            calls.append("normal_restore")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
        adapters=SimpleNamespace(
            dca=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
            i2c=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
        ),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            b0_path=_continuous_b0_path(),
            hardware=SimpleNamespace(driver_profile=_profile_path()),
        ),
        entry=lambda campaign_id: entry,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=SimpleNamespace(runtime=runtime),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(model="fake-kimi"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="injected preparation failure"):
        activate_continuous_live(
            {
                "runtime_config": "runtime.yaml",
                "campaign_id": "campaign-1",
                "parameter_policy": _policy_config(),
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            tmp_path,
        )

    assert calls == ["prepare", "shutdown"]


@pytest.mark.parametrize(
    ("failed_transition", "residual_state", "expected_final_state"),
    [
        (
            CampaignState.ABORTING,
            CampaignState.BASELINE_MEASUREMENT,
            CampaignState.FAILED,
        ),
        (
            CampaignState.ROLLING_BACK,
            CampaignState.ABORTING,
            CampaignState.FAILED,
        ),
        (
            CampaignState.FAILED,
            CampaignState.ROLLING_BACK,
            CampaignState.FAILED,
        ),
        (
            CampaignState.RECOVERY_REQUIRED,
            CampaignState.RESTORE_B0,
            CampaignState.RECOVERY_REQUIRED,
        ),
    ],
)
def test_live_commission_retries_incomplete_preparation_recovery(
    tmp_path,
    monkeypatch,
    failed_transition,
    residual_state,
    expected_final_state,
) -> None:
    calls: list[object] = []
    campaign_state = {"value": CampaignState.AWAIT_BASELINE_APPROVAL}

    class Orchestrator:
        baseline = load_b0(_continuous_b0_path())
        store = SimpleNamespace(
            current_state=lambda campaign_id: campaign_state["value"]
        )

        def prepare_continuous_commission(self, **kwargs):
            calls.append(("prepare", kwargs))
            calls.extend([("b0_apply", 1), ("b0_readback", 1)])
            campaign_state["value"] = residual_state
            raise RuntimeError(
                f"injected {failed_transition.value} transition persistence failure"
            )

        def recover_continuous_commission_failure(self):
            calls.extend([("b0_apply", 2), ("b0_readback", 2)])
            campaign_state["value"] = expected_final_state
            if expected_final_state is CampaignState.RECOVERY_REQUIRED:
                raise RecoveryRequiredError("injected B0 recovery failure")
            return AppliedSettings(readback_i2c=self.baseline.i2c.model_dump())

        def restore_static_i2c_b0(self):
            raise AssertionError("normal completion recovery used after prepare failure")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
        adapters=SimpleNamespace(
            dca=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
            i2c=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
        ),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            b0_path=_continuous_b0_path(),
            hardware=SimpleNamespace(driver_profile=_profile_path()),
        ),
        entry=lambda campaign_id: entry,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=SimpleNamespace(runtime=runtime),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(model="fake-kimi"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(),
    )

    expected_error = (
        "B0 recovery failed"
        if expected_final_state is CampaignState.RECOVERY_REQUIRED
        else "transition persistence failure"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        activate_continuous_live(
            {
                "runtime_config": "runtime.yaml",
                "campaign_id": "campaign-1",
                "parameter_policy": _policy_config(),
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            tmp_path,
        )

    assert calls == [
        ("prepare", {"max_candidates": 3}),
        ("b0_apply", 1),
        ("b0_readback", 1),
        ("b0_apply", 2),
        ("b0_readback", 2),
        "shutdown",
    ]
    assert campaign_state["value"] is expected_final_state
    assert campaign_state["value"] is not CampaignState.COMPLETED


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_live_commission_later_assembly_failure_uses_failure_specific_recovery(
    tmp_path,
    monkeypatch,
    recovery_fails,
) -> None:
    calls: list[str] = []
    campaign_state = {"value": CampaignState.I2C_SCREENING}
    now = datetime(2026, 8, 12, tzinfo=timezone.utc)

    class Orchestrator:
        baseline = load_b0(_continuous_b0_path())
        store = SimpleNamespace(
            current_state=lambda campaign_id: campaign_state["value"]
        )

        def prepare_continuous_commission(self, **kwargs):
            calls.append("prepare")
            return SimpleNamespace(
                observation=object(),
                authorization=StageAuthorization(
                    authorization_id="auth-1",
                    campaign_id="campaign-1",
                    mode=AuthorizationMode.PER_STAGE,
                    time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
                    stage=CampaignState.I2C_SCREENING,
                    baseline_hash="b" * 64,
                    rollback_hash="c" * 64,
                    max_candidates=3,
                    max_duration_s=None,
                    issued_at=now,
                    expires_at=None,
                    rollback_snapshot=AppliedSettings(readback_i2c={"eq": 22}),
                ),
            )

        def recover_continuous_commission_failure(self):
            calls.append("failure_recovery")
            if recovery_fails:
                campaign_state["value"] = CampaignState.RECOVERY_REQUIRED
                raise RecoveryRequiredError("injected B0 recovery failure")
            campaign_state["value"] = CampaignState.FAILED

        def restore_static_i2c_b0(self):
            raise AssertionError("normal completion restore used for assembly failure")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
        adapters=SimpleNamespace(
            dca=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
            i2c=SimpleNamespace(preflight=lambda: {"identity_verified": True}),
        ),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            b0_path=_continuous_b0_path(),
            hardware=SimpleNamespace(driver_profile=_profile_path()),
        ),
        entry=lambda campaign_id: entry,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=SimpleNamespace(runtime=runtime),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(model="fake-kimi"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(
            to_eye_sample=lambda observation: (_ for _ in ()).throw(
                RuntimeError("injected later assembly failure")
            )
        ),
    )

    if recovery_fails:
        with pytest.raises(RuntimeError, match="B0 recovery failed") as caught:
            activate_continuous_live(
                {
                    "runtime_config": "runtime.yaml",
                    "campaign_id": "campaign-1",
                    "parameter_policy": _policy_config(),
                    "max_rounds": 3,
                    "target_tdecq_db": 2.0,
                },
                tmp_path,
            )
        assert isinstance(caught.value.__cause__, RecoveryRequiredError)
        assert campaign_state["value"] is CampaignState.RECOVERY_REQUIRED
    else:
        with pytest.raises(RuntimeError, match="injected later assembly failure"):
            activate_continuous_live(
                {
                    "runtime_config": "runtime.yaml",
                    "campaign_id": "campaign-1",
                    "parameter_policy": _policy_config(),
                    "max_rounds": 3,
                    "target_tdecq_db": 2.0,
                },
                tmp_path,
            )
        assert campaign_state["value"] is CampaignState.FAILED

    assert calls == ["prepare", "failure_recovery", "shutdown"]


def test_live_preflight_bootstraps_campaign_and_returns_nonexecuting_preview(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[object] = []
    captured_policies = []
    captured_requests = []
    kimi_client_kwargs: list[dict[str, object]] = []
    profile = _profile_path()
    baseline_path = tmp_path / "b0.yaml"
    baseline_path.write_text(
        _continuous_b0_path().read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    eye = _eye(tmp_path)
    observation = SimpleNamespace(
        observation_id=eye.measurement_id,
        campaign_id="campaign-1",
        stage=CampaignState.PREFLIGHT,
        settings=AppliedSettings(readback_i2c={"eq": 22}),
    )

    class Orchestrator:
        def capture_continuous_preflight_observation(self):
            calls.append("capture")
            return SimpleNamespace(
                verified=True,
                dca={"identity_verified": True},
                i2c={"identity_verified": True},
                observation=observation,
            )

        def run_continuous_i2c_candidate(self, *args):
            raise AssertionError("preflight must not execute a candidate")

        def measure_current_i2c_state(self):
            raise AssertionError("capture owns the preflight measurement")

        def restore_static_i2c_b0(self):
            raise AssertionError("preflight must not restore B0")

        def recover_continuous_commission_failure(self):
            raise AssertionError("preflight must not recover B0")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            hardware=SimpleNamespace(driver_profile=profile),
            b0_path=baseline_path,
        ),
        baseline_hash=sha256(
            load_b0(baseline_path).model_dump_json().encode()
        ).hexdigest(),
        entry=lambda campaign_id: calls.append(("entry", campaign_id)) or entry,
    )

    class Service:
        def __init__(self):
            self.runtime = runtime

        def create_campaign(self, request):
            calls.append(("create", request))

    fake_api_runtime = SimpleNamespace(
        service=Service(),
        shutdown=lambda: calls.append("shutdown"),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: fake_api_runtime,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(to_eye_sample=lambda item: eye),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: kimi_client_kwargs.append(kwargs) or SimpleNamespace(),
    )

    class Provider:
        provider_id = "fake-kimi"
        provider_version = "1"

        def decide(self, context):
            calls.append("kimi-preview")
            captured_policies.append(context.policy)
            return ProviderAction(
                action=ProviderActionKind.PROPOSE,
                requested_params={"eq": 23},
                reason_codes=(ReasonCode.EXPLORATION,),
                confidence=0.9,
                summary="Preview only.",
            )

    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_provider.KimiProvider",
        lambda *args, **kwargs: Provider(),
    )
    from vcsel_ai_tuning.v2.continuous.controller import ContinuousOptimizationController

    preview_initial_decision = ContinuousOptimizationController.preview_initial_decision

    def capture_preflight_request(controller, request):
        captured_requests.append(request)
        return preview_initial_decision(controller, request)

    monkeypatch.setattr(
        ContinuousOptimizationController,
        "preview_initial_decision",
        capture_preflight_request,
    )

    report = preflight_continuous_live(
        {
            "runtime_config": "runtime.yaml",
            "campaign_id": "campaign-1",
            "parameter_policy": _policy_config(),
            "max_rounds": 20,
            "target_tdecq_db": 2.8,
        },
        tmp_path,
    )

    created = next(item[1] for item in calls if isinstance(item, tuple) and item[0] == "create")
    assert created.tuning_path.value == "PATH_TDECQ_BAYESIAN"
    assert not hasattr(created, "authorization")
    assert calls.index("capture") < calls.index("kimi-preview") < calls.index("shutdown")
    assert report["dut_write_performed"] is False
    assert report["hardware_write_performed"] is False
    assert report["hardware_write_scope"] == "NO_DUT_REGISTER_OR_BERT_WRITE"
    assert report["dca_acquisition_commands_performed"] is True
    assert report["bert_contacted"] is False
    assert report["provider_action"] == "PROPOSE"
    assert report["provider_requested_params"] == {"eq": 23}
    assert captured_requests[0].max_rounds == 20
    assert captured_requests[0].stopping_policy.target_tdecq_db == 2.8
    assert captured_policies[0].by_name["eq"].tuning_min == 16
    assert captured_policies[0].by_name["eq"].tuning_max == 25
    assert captured_policies[0].by_name["res_bw"].max_step == 1
    assert kimi_client_kwargs == [
        {
            "model": "kimi-k3",
            "response_log_path": (
                tmp_path / "campaign-1" / "kimi-response-attempts.jsonl"
            ),
        }
    ]


@pytest.mark.parametrize("failure_phase", ["create", "entry", "capture", "kimi"])
def test_live_preflight_closes_runtime_at_every_failure_phase(
    tmp_path,
    monkeypatch,
    failure_phase,
) -> None:
    calls: list[str] = []
    profile = _profile_path()
    baseline_path = tmp_path / "b0.yaml"
    baseline_path.write_text(
        _continuous_b0_path().read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    eye = _eye(tmp_path)
    observation = SimpleNamespace(observation_id=eye.measurement_id)

    class Orchestrator:
        def capture_continuous_preflight_observation(self):
            calls.append("capture")
            if failure_phase == "capture":
                raise RuntimeError("injected capture failure")
            return SimpleNamespace(
                dca={"identity_verified": True},
                i2c={"identity_verified": True},
                observation=observation,
            )

        def run_continuous_i2c_candidate(self, *args):
            raise AssertionError("preflight must not execute a candidate")

        def measure_current_i2c_state(self):
            raise AssertionError("preflight must not measure through execution port")

        def restore_static_i2c_b0(self):
            raise AssertionError("preflight must not restore B0")

        def recover_continuous_commission_failure(self):
            raise AssertionError("preflight must not recover B0")

    entry = SimpleNamespace(
        orchestrator=Orchestrator(),
        repository=SimpleNamespace(run_dir=tmp_path),
    )

    def open_entry(campaign_id):
        calls.append("entry")
        if failure_phase == "entry":
            raise RuntimeError("injected entry failure")
        return entry

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            storage=SimpleNamespace(sqlite_path=tmp_path / "campaigns.sqlite3"),
            hardware=SimpleNamespace(driver_profile=profile),
            b0_path=baseline_path,
        ),
        baseline_hash="a" * 64,
        entry=open_entry,
    )

    class Service:
        def __init__(self):
            self.runtime = runtime

        def create_campaign(self, request):
            calls.append("create")
            if failure_phase == "create":
                raise RuntimeError("injected create failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: SimpleNamespace(
            service=Service(),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.execution.PersistedObservationAdapter",
        lambda repository: SimpleNamespace(to_eye_sample=lambda item: eye),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(),
    )

    class Provider:
        provider_id = "fake-kimi"
        provider_version = "1"

        def decide(self, context):
            calls.append("kimi")
            if failure_phase == "kimi":
                raise RuntimeError("injected kimi failure")
            raise AssertionError("provider should only be reached for kimi failure")

    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_provider.KimiProvider",
        lambda *args, **kwargs: Provider(),
    )

    with pytest.raises(RuntimeError, match="injected"):
        preflight_continuous_live(
            {
                "runtime_config": "runtime.yaml",
                "campaign_id": "campaign-1",
                "parameter_policy": _policy_config(),
                "max_rounds": 3,
                "target_tdecq_db": 2.0,
            },
            tmp_path,
        )

    assert calls[-1] == "shutdown"
    assert calls.count("shutdown") == 1


def test_live_preflight_uses_real_runtime_metadata_with_fake_adapters(
    tmp_path,
    monkeypatch,
) -> None:
    baseline = load_b0(_continuous_b0_path())
    baseline_path = tmp_path / "b0.yaml"
    baseline_path.write_text(
        _continuous_b0_path().read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    profile = Path("configs/hardware/vcsel_lab_v2_1_continuous.yaml").resolve()
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        yaml.safe_dump(
            {
                "mode": "LIVE_TUNING",
                "storage": {
                    "sqlite_path": str(tmp_path / "state" / "campaigns.sqlite3"),
                    "artifact_root": str(tmp_path / "artifacts"),
                },
                "b0_path": str(baseline_path),
                "measurement_timeout_s": 180.0,
                "worker_count": 1,
                "hardware": {
                    "driver_profile": str(profile),
                    "dca_visa_resource": "TCPIP0::Nate-Rog::hislip0,4880::INSTR",
                    "i2c_com_port": "COM7",
                    "i2c_address": 166,
                    "bert_host": "192.168.3.2",
                    "bert_port": 9000,
                    "resource_ids": ["fake-dca", "fake-i2c"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    image_stream = BytesIO()
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_stream, format="PNG")
    png = image_stream.getvalue()
    with Image.open(BytesIO(png)) as image:
        kimi_png, _, _ = derive_kimi_input_png(image)
    npy_stream = BytesIO()
    np.save(
        npy_stream,
        np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8),
        allow_pickle=False,
    )
    npy = npy_stream.getvalue()
    png_hash = sha256(png).hexdigest()
    kimi_hash = sha256(kimi_png).hexdigest()
    npy_hash = sha256(npy).hexdigest()
    closed: list[str] = []

    class Dca:
        def preflight(self):
            return {"identity_verified": True, "configuration_verified": True}

        def measure_single(self):
            return MeasurementResult(
                acquisition_valid=True,
                er_db=4.5,
                er_status="CORR",
                tdecq_db=3.1,
                tdecq_status="CORR",
                tdecq_invalid_reason=None,
                elapsed_s=0.1,
                raw_responses={
                    "continuous_eye_evidence": {
                        "provenance": "SCREEN_IMAGE_DERIVED",
                        "dca_fingerprint": "b" * 64,
                        "png_path": "eye_artifacts/preflight.png",
                        "npy_path": "eye_artifacts/preflight.npy",
                        "png_sha256": png_hash,
                        "npy_sha256": npy_hash,
                        "kimi_png_path": "eye_artifacts/kimi_input_preflight.png",
                        "kimi_png_sha256": kimi_hash,
                        "kimi_source_png_sha256": png_hash,
                        "kimi_resize_width": 2,
                        "kimi_resize_height": 2,
                        "kimi_resize_applied": False,
                        "kimi_max_width": 1280,
                        "kimi_max_height": 528,
                        "kimi_resample": "LANCZOS",
                        "kimi_colors": 256,
                        "kimi_quantization": "MEDIANCUT",
                        "kimi_dither": "NONE",
                        "kimi_output_mode": "P",
                        "kimi_provenance": "DCA_SCREEN_DERIVED",
                    }
                },
                artifacts=(
                    ArtifactSpec("eye_artifacts", "preflight.png", png, "image/png"),
                    ArtifactSpec(
                        "eye_artifacts",
                        "preflight.npy",
                        npy,
                        "application/x-npy",
                    ),
                    ArtifactSpec(
                        "eye_artifacts",
                        "kimi_input_preflight.png",
                        kimi_png,
                        "image/png",
                    ),
                ),
            )

        def close(self):
            closed.append("dca")

    class I2c:
        def preflight(self):
            return {"identity_verified": True, "configuration_verified": True}

        def readback(self):
            return baseline.i2c.model_dump()

        def apply(self, requested):
            raise AssertionError("real runtime preflight must not call i2c.apply")

        def close(self):
            closed.append("i2c")

    class Bert:
        def preflight(self):
            raise AssertionError("real runtime preflight must not contact BERT")

        def readback(self):
            raise AssertionError("real runtime preflight must not contact BERT")

        def apply(self, pattern: str, fir: FirSettings):
            raise AssertionError("real runtime preflight must not contact BERT")

        def close(self):
            closed.append("bert")

    adapter_factory_kwargs = []

    def build_adapters(*args, **kwargs):
        adapter_factory_kwargs.append(kwargs)
        return AdapterBundle(
            dca=Dca(),
            i2c=I2c(),
            bert=Bert(),
            measurement_timeout_s=180.0,
        )

    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.build_live_adapters_from_config",
        build_adapters,
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: SimpleNamespace(),
    )

    class Provider:
        provider_id = "fake-kimi"
        provider_version = "1"

        def decide(self, context):
            return ProviderAction(
                action=ProviderActionKind.HOLD,
                reason_codes=(ReasonCode.EXPLORATION,),
                confidence=0.8,
                summary="Integration preview.",
            )

    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_provider.KimiProvider",
        lambda *args, **kwargs: Provider(),
    )

    report = preflight_continuous_live(
        {
            "runtime_config": str(runtime_path),
            "campaign_id": "campaign-real-runtime",
            "parameter_policy": _policy_config(),
            "max_rounds": 3,
            "target_tdecq_db": 2.0,
        },
        tmp_path / "controller",
    )

    assert report["provider_action"] == "HOLD"
    assert report["dut_write_performed"] is False
    assert set(closed) == {"dca", "i2c", "bert"}
    assert adapter_factory_kwargs == [
        {
            "campaign_id": "campaign-real-runtime",
            "continuous_i2c_only": True,
            "artifact_directory": (
                tmp_path / "artifacts" / "campaign-real-runtime" / "eye_artifacts"
            ),
        }
    ]
    assert (tmp_path / "state" / "campaigns.sqlite3").is_file()
    assert list((tmp_path / "artifacts" / "campaign-real-runtime").rglob("*.png"))
