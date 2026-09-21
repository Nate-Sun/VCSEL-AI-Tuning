# V2.1 LIVE Simplified Operator and DCA Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the V2.1 continuous LIVE launcher ask only for rounds, target TDECQ, API key, and one COMMISSION approval Enter while removing V2.1 token checks and hardening FlexDCA PNG transfer against intermittent VISA timeouts.

**Architecture:** Keep Campaign ID as an internally generated audit key, but remove both user-facing and Python-level V2.1 authorization-token plumbing. Add one signed PowerShell 7 launcher that freezes the generated session scope and retains the PRE-FLIGHT-to-COMMISSION human gate. Configure the shared PyVISA session with Keysight's recommended 1024-byte binary chunk and convert raw screen-evidence VISA failures into command-labelled domain errors.

**Tech Stack:** Python 3.13, Pydantic, PyVISA, Pillow, pytest, PowerShell 7, YAML, Git.

---

## File map

- Modify `src/vcsel_ai_tuning/hardware/flexdca.py`: configure binary-transfer chunk size when opening VISA resources.
- Modify `src/vcsel_ai_tuning/hardware/continuous_eye.py`: label screenshot save, PNG transfer, and fingerprint query failures.
- Modify `tests_py/ai_tuning/hardware/test_flexdca.py`: prove VISA session configuration and closure behavior.
- Modify `tests_py/ai_tuning/v2/continuous/test_dca_evidence.py`: prove command-labelled evidence failures and preserved causes.
- Modify `src/vcsel_ai_tuning/v2/continuous/factory.py`: remove V2.1 token creation and activation validation while retaining the explicit device-contact gate.
- Modify `src/vcsel_ai_tuning/v2/continuous/cli.py`: remove token flags and activate COMMISSION without token arguments.
- Modify `src/vcsel_ai_tuning/continuous_live.py`: remove duplicate commission-token validation at upper hardware composition.
- Modify `tests_py/ai_tuning/v2/continuous/test_factory_cli.py`: replace token tests with token-free activation and rejection of obsolete flags.
- Modify `tests_py/ai_tuning/test_continuous_live.py`: update upper-layer activation and launcher/runbook contracts.
- Create `scripts/run_v2_1_continuous_live.ps1`: the only supported interactive V2.1 LIVE operator entry.
- Modify `docs/v2-1-continuous-eye-testbench.md`: call the signed script instead of embedding a large editable PowerShell block.

### Task 1: Configure reliable FlexDCA binary transfer

**Files:**
- Modify: `tests_py/ai_tuning/hardware/test_flexdca.py:515-548`
- Modify: `src/vcsel_ai_tuning/hardware/flexdca.py:552-569`

- [ ] **Step 1: Write the failing chunk-size assertion**

Change the fake instrument and assertions in `test_open_visa_session_sets_timeout_and_closes_both_objects` to:

```python
class Instrument:
    timeout = None
    chunk_size = None

    def close(self):
        events.append("instrument.close")

# inside the open context
assert opened is instrument
assert instrument.timeout == 1234
assert instrument.chunk_size == 1024
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
py -3.13 -m pytest tests_py/ai_tuning/hardware/test_flexdca.py::test_open_visa_session_sets_timeout_and_closes_both_objects -q
```

Expected: FAIL because `instrument.chunk_size` remains `None`.

- [ ] **Step 3: Set the Keysight-recommended chunk size**

Update `open_visa_session` immediately after setting timeout:

```python
instrument = manager.open_resource(resource)
instrument.timeout = timeout_ms
instrument.chunk_size = 1024
yield instrument
```

- [ ] **Step 4: Run both VISA lifecycle tests and verify GREEN**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/hardware/test_flexdca.py -q
```

Expected: all `test_flexdca.py` tests PASS; manager and instrument closure assertions remain unchanged.

- [ ] **Step 5: Commit the isolated transport change**

```powershell
git add src/vcsel_ai_tuning/hardware/flexdca.py tests_py/ai_tuning/hardware/test_flexdca.py
git commit -m "fix(v2.1): use reliable FlexDCA binary chunks"
```

### Task 2: Label every DCA screen-evidence VISA failure

**Files:**
- Modify: `tests_py/ai_tuning/v2/continuous/test_dca_evidence.py`
- Modify: `src/vcsel_ai_tuning/hardware/continuous_eye.py:15-173`

- [ ] **Step 1: Add failing fake-session tests for the three required stages**

Add `pytest` at module scope and use this failing session:

```python
import pytest


class FailingEvidenceSession(RecordingSession):
    def __init__(self, png: bytes, *, failing_command: str) -> None:
        super().__init__(png)
        self.failing_command = failing_command
        self.failure = TimeoutError("visa timeout")

    def query(self, command):
        if command == self.failing_command:
            raise self.failure
        return super().query(command)

    def query_binary_values(self, command, **kwargs):
        if command == self.failing_command:
            raise self.failure
        return super().query_binary_values(command, **kwargs)
```

Add three explicit tests:

```python
@pytest.mark.parametrize(
    ("command", "stage"),
    [
        (":DISK:SIMage:SAVE;*OPC?", "SCREEN_SAVE"),
        (
            ':DISK:FILE:READ? "%TEMP_DIR%\\campaign-1_000001.png"',
            "PNG_BINARY_READ",
        ),
        (":TIMebase:UIRange?", "FINGERPRINT_QUERY"),
    ],
)
def test_screen_evidence_reports_failing_stage_and_command(command, stage) -> None:
    session = FailingEvidenceSession(_png(), failing_command=command)
    adapter = ScreenEvidenceDcaAdapter(
        MeasurementAdapter(),
        session=session,
        source="FUNC1",
        artifact_prefix="campaign-1",
    )

    with pytest.raises(DcaEvidenceCaptureError) as captured:
        adapter.measure_single()

    assert captured.value.stage == stage
    assert captured.value.command == command
    assert stage in str(captured.value)
    assert command in str(captured.value)
    assert captured.value.__cause__ is session.failure
```

Import `DcaEvidenceCaptureError` beside `ScreenEvidenceDcaAdapter`.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/v2/continuous/test_dca_evidence.py::test_screen_evidence_reports_failing_stage_and_command -q
```

Expected: collection/import FAIL because `DcaEvidenceCaptureError` does not exist.

- [ ] **Step 3: Implement the minimal labelled wrappers**

Add this public domain error and two private helpers to `continuous_eye.py`:

```python
class DcaEvidenceCaptureError(RuntimeError):
    def __init__(self, *, stage: str, command: str) -> None:
        self.stage = stage
        self.command = command
        super().__init__(f"DCA evidence {stage} failed for SCPI {command!r}")


def _evidence_query(session: Any, *, stage: str, command: str) -> str:
    try:
        return str(session.query(command))
    except Exception as error:
        raise DcaEvidenceCaptureError(stage=stage, command=command) from error


def _evidence_binary_query(session: Any, *, stage: str, command: str) -> bytes:
    try:
        return session.query_binary_values(
            command,
            datatype="s",
            header_fmt="ieee",
            container=bytes,
        )
    except Exception as error:
        raise DcaEvidenceCaptureError(stage=stage, command=command) from error
```

Use `_evidence_query(..., stage="SCREEN_SAVE", ...)` for `:DISK:SIMage:SAVE;*OPC?`, `_evidence_binary_query(..., stage="PNG_BINARY_READ", ...)` for the PNG, and `_evidence_query(..., stage="FINGERPRINT_QUERY", ...)` for every fingerprint command. Export the new error in `__all__`.

- [ ] **Step 4: Run the complete screen-evidence test file and verify GREEN**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/v2/continuous/test_dca_evidence.py -q
```

Expected: all tests PASS, including exact PNG bytes, sequence persistence, Kimi resizing, and the three failure stages.

- [ ] **Step 5: Commit the evidence diagnostics**

```powershell
git add src/vcsel_ai_tuning/hardware/continuous_eye.py tests_py/ai_tuning/v2/continuous/test_dca_evidence.py
git commit -m "fix(v2.1): identify DCA evidence transfer failures"
```

### Task 3: Remove V2.1 token plumbing without removing stage authorization

**Files:**
- Modify: `tests_py/ai_tuning/v2/continuous/test_factory_cli.py`
- Modify: `tests_py/ai_tuning/test_continuous_live.py`
- Modify: `src/vcsel_ai_tuning/v2/continuous/factory.py:26-179`
- Modify: `src/vcsel_ai_tuning/v2/continuous/cli.py:14-170`
- Modify: `src/vcsel_ai_tuning/continuous_live.py:179-345`

- [ ] **Step 1: Replace token-binding tests with token-free activation tests**

In `test_factory_cli.py`, delete the five `test_commission_token_*` tests and the token-scope rejection tests. Remove `commission_authorization_token_for` imports. Update factory calls to omit `authorization_token`.

Replace `test_live_factory_refuses_construction_without_two_explicit_gates` with:

```python
def test_live_factory_requires_explicit_device_contact_gate(tmp_path) -> None:
    with pytest.raises(LiveContactAuthorizationError, match="allow-device-contact"):
        build_continuous_dependencies(
            mode=ExecutionMode.LIVE,
            result_dir=tmp_path,
            allow_device_contact=False,
        )
```

Add a token-free activation test:

```python
def test_lazy_live_activation_calls_upper_layer_without_token(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        continuous_factory,
        "import_module",
        lambda name: SimpleNamespace(
            activate_continuous_live=lambda config, result_dir: calls.append(
                (name, dict(config), result_dir)
            )
            or SimpleNamespace(),
        ),
    )
    dependencies = build_continuous_dependencies(
        mode=ExecutionMode.LIVE,
        result_dir=tmp_path,
        max_rounds=30,
        target_tdecq_db=2.8,
        allow_device_contact=True,
        live_config={
            "runtime_config": "runtime-v2.1-live.yaml",
            "campaign_id": "campaign-1",
        },
    )

    dependencies.live_activation.activate()

    assert calls[0][1]["max_rounds"] == 30
    assert calls[0][1]["target_tdecq_db"] == 2.8
```

- [ ] **Step 2: Add CLI tests that reject obsolete token flags and commission without them**

Update the existing successful COMMISSION fake so `Activation.activate(self)` takes no argument and records `"activate"`. Invoke CLI with only config, `--allow-device-contact`, rounds, target, and `--live-operation COMMISSION`.

Add:

```python
@pytest.mark.parametrize(
    "obsolete_flag",
    ["--authorization-token", "--commission-authorization-token"],
)
def test_cli_rejects_removed_token_flags(tmp_path, obsolete_flag) -> None:
    config = tmp_path / "continuous.yaml"
    config.write_text(yaml.safe_dump({"mode": "FAKE"}), encoding="utf-8")

    with pytest.raises(SystemExit) as captured:
        main(["--config", str(config), obsolete_flag, "removed"])

    assert captured.value.code == 2
```

Keep tests proving LIVE still requires rounds, target, `--live-operation`, `--allow-device-contact`, fresh Campaign ID, and explicit runtime configuration.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/v2/continuous/test_factory_cli.py -q
```

Expected: failures show that both CLI token flags still exist and `activate()` still requires a token.

- [ ] **Step 4: Remove token functions, parameters, flags, and duplicate validation**

Make these exact API changes:

```python
# factory.py
class LazyLiveActivation:
    def activate(self) -> ContinuousDependencies:
        if self.activated:
            raise RuntimeError("LIVE activation is single-use")
        _validated_operator_controls(
            self.config["max_rounds"],
            self.config["target_tdecq_db"],
        )
        self.activated = True
        module = import_module("vcsel_ai_tuning.continuous_live")
        return module.activate_continuous_live(self.config, self.result_dir)
```

Delete `commission_authorization_token_for`, remove `authorization_token` from `build_continuous_dependencies`, and remove only its nonempty-token gate. Do not remove `allow_device_contact`, `_validated_operator_controls`, fresh Campaign ID validation, or the `StageAuthorization` stored in `ContinuousRequest`.

In `cli.py`, remove both token arguments/imports and replace the COMMISSION branch activation with:

```python
activated = dependencies.live_activation.activate()
```

In `continuous_live.py`, change:

```python
def activate_continuous_live(
    config: Mapping[str, object],
    result_dir: Path,
) -> Any:
```

Delete the `commission_authorization_token_for` import and comparison, but retain `prepare_continuous_commission()`, its returned `StageAuthorization`, and `authorization=prepared.authorization` in `ContinuousRequest`.

- [ ] **Step 5: Update upper-layer tests without weakening campaign safety**

In `test_continuous_live.py`, remove only external token arguments and token mismatch tests. Keep assertions that changed rounds/target/config manifest, stale Campaign ID, lease uncertainty, DCA/I2C identity failures, and B0 recovery all fail closed. Update every `activate_continuous_live(...)` and `live_activation.activate(...)` call to the new two-argument/zero-argument signatures.

- [ ] **Step 6: Run both continuous test files and verify GREEN**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/v2/continuous/test_factory_cli.py tests_py/ai_tuning/test_continuous_live.py -q
```

Expected: PASS; obsolete token flags are rejected and internal stage authorization tests remain green.

- [ ] **Step 7: Commit token removal**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/factory.py src/vcsel_ai_tuning/v2/continuous/cli.py src/vcsel_ai_tuning/continuous_live.py tests_py/ai_tuning/v2/continuous/test_factory_cli.py tests_py/ai_tuning/test_continuous_live.py
git commit -m "refactor(v2.1): remove continuous operator tokens"
```

### Task 4: Add the immutable PowerShell 7 LIVE launcher

**Files:**
- Create: `scripts/run_v2_1_continuous_live.ps1`
- Modify: `tests_py/ai_tuning/test_continuous_live.py`
- Modify: `docs/v2-1-continuous-eye-testbench.md`

- [ ] **Step 1: Write failing launcher contract tests**

Add a helper that reads `scripts/run_v2_1_continuous_live.ps1`, then add:

```python
def test_signed_live_launcher_has_only_approved_operator_inputs() -> None:
    launcher = Path("scripts/run_v2_1_continuous_live.ps1").read_text(
        encoding="utf-8"
    )
    prompts = re.findall(r"Read-Host\s+'([^']+)'", launcher)

    assert len(prompts) == 4
    assert any("Max optimization rounds" in prompt for prompt in prompts)
    assert any("Target TDECQ" in prompt for prompt in prompts)
    assert any("MOONSHOT_API_KEY" in prompt for prompt in prompts)
    assert any("COMMISSION approval" in prompt for prompt in prompts)
    assert "Retype" not in launcher
    assert "authorization-token" not in launcher
    assert "commission-authorization-token" not in launcher
    assert "REPLACE_WITH_FRESH_CAMPAIGN_ID" in launcher
    assert "[guid]::NewGuid()" in launcher
```

Add a second static contract proving both operations share frozen controls:

```python
def test_signed_live_launcher_reuses_frozen_scope_for_both_operations() -> None:
    launcher = Path("scripts/run_v2_1_continuous_live.ps1").read_text(
        encoding="utf-8"
    )

    assert launcher.count("--max-rounds $SessionMaxRounds") == 2
    assert launcher.count("--target-tdecq-db $SessionTargetTdecqDb") == 2
    assert launcher.count("--config $SessionLiveConfigPath") == 2
    assert "Assert-ContinuousPreflightReport" in launcher
    assert "$PreflightExitCode -ne 0" in launcher
    assert "--live-operation PREFLIGHT" in launcher
    assert "--live-operation COMMISSION" in launcher
```

- [ ] **Step 2: Run the launcher tests and verify RED**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_continuous_live.py -k "signed_live_launcher" -q
```

Expected: FAIL because the signed launcher file does not exist.

- [ ] **Step 3: Create the PowerShell 7 launcher**

Create `scripts/run_v2_1_continuous_live.ps1` with:

```powershell
#requires -Version 7.0
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

$ExpectedBranch = 'codex/v2.1-continuous-eye-optimization'
$RequiredImplementationBaseline = 'ff5187be'
$TemplateConfigPath = Join-Path $PWD 'configs\ai_tuning\v2_1_continuous.live.yaml'
$SessionCampaignId = 'v21-b-{0}-{1}' -f `
    ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')), `
    ([guid]::NewGuid().ToString('N').Substring(0, 8))
$LiveConfigPath = Join-Path ([IO.Path]::GetTempPath()) `
    ("v2_1_continuous.live.$([guid]::NewGuid().ToString('N')).yaml")
```

Carry forward the existing positive-integer, JSON-result, config-manifest, scope-verifier, and PRE-FLIGHT-report helpers. Replace manual YAML editing with an exact single-placeholder replacement that throws unless the placeholder occurs once. Freeze `$SessionCampaignId`, `$SessionMaxRounds`, `$SessionTargetTdecqDb`, `$SessionLiveConfigPath`, manifest, branch, and HEAD before device contact.

Use exactly these operator prompts:

```powershell
$MaxRoundsText = Read-Host '请输入本次最大优化轮数 / Max optimization rounds'
$TargetTdecqDb = [double](Read-Host '请输入目标 TDECQ dB（严格小于） / Target TDECQ dB (strictly below)')
$secureMoonshotKey = Read-Host '请输入 MOONSHOT_API_KEY / Enter MOONSHOT_API_KEY' -AsSecureString
[void](Read-Host 'PRE-FLIGHT 已通过；取得 COMMISSION 授权后按 Enter / PRE-FLIGHT passed; press Enter after COMMISSION approval')
```

The PRE-FLIGHT and COMMISSION commands must omit token flags and use the same frozen config, rounds, and target. Keep `try/finally` cleanup for the temp YAML and API key. Do not catch a failed PRE-FLIGHT and continue.

- [ ] **Step 4: Replace the embedded runbook with the signed entry command**

Update the LIVE section in `docs/v2-1-continuous-eye-testbench.md` to instruct:

```powershell
pwsh -NoLogo -NoProfile -File .\scripts\run_v2_1_continuous_live.ps1
```

Document that Campaign ID is generated and displayed, rounds and target remain operator inputs, API key is secure input, and the only later interaction is the COMMISSION Enter gate. Remove instructions to edit a temporary YAML, retype Campaign ID, or create tokens. Retain provenance, evidence preservation, no-BERT/FIR/NLC statements, and recovery guidance.

- [ ] **Step 5: Run launcher and runbook tests and verify GREEN**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_continuous_live.py -k "launcher or runbook" -q
```

Expected: PASS under PowerShell 7 contracts; no test contacts hardware.

- [ ] **Step 6: Commit the operator entry**

```powershell
git add scripts/run_v2_1_continuous_live.ps1 docs/v2-1-continuous-eye-testbench.md tests_py/ai_tuning/test_continuous_live.py
git commit -m "feat(v2.1): add simplified signed live launcher"
```

### Task 5: Verify B, merge to main, and stop before hardware

**Files:**
- Verify all modified files above.
- Preserve untracked: `artifacts/`, `outputs/`, `state/`.

- [ ] **Step 1: Run focused V2.1 and hardware suites**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
py -3.13 -m pytest tests_py/ai_tuning/hardware/test_flexdca.py tests_py/ai_tuning/v2/continuous/test_dca_evidence.py tests_py/ai_tuning/v2/continuous/test_factory_cli.py tests_py/ai_tuning/test_continuous_live.py -q
```

Expected: all selected tests PASS with no warnings.

- [ ] **Step 2: Run the complete B suite**

```powershell
py -3.13 -m pytest -q
```

Expected: zero failures; Windows symlink-permission skips may remain. Record the exact pass/skip counts rather than copying an older count.

- [ ] **Step 3: Check diff and repository provenance**

```powershell
git diff --check
git status --short
git branch --show-current
git rev-parse HEAD
```

Expected: no tracked uncommitted changes; only preserved experiment directories may remain untracked; branch is `codex/v2.1-continuous-eye-optimization`.

- [ ] **Step 4: Review the complete implementation diff**

```powershell
git diff 5418d199..HEAD --stat
git diff 5418d199..HEAD -- src tests_py scripts docs/v2-1-continuous-eye-testbench.md
```

Expected: changes are limited to DCA transport/diagnostics, V2.1 token removal, the signed launcher, tests, and runbook.

- [ ] **Step 5: Merge B into main without deleting either worktree**

From `C:\Users\NateS\Documents\VCSEL\.worktrees\main-v1.1.1-merge`:

```powershell
git status --short
git merge --no-ff codex/v2.1-continuous-eye-optimization -m "Merge V2.1 simplified LIVE launcher and DCA transfer fix"
```

If main has unrelated tracked changes or conflicts, stop and report them; do not reset or overwrite user work.

- [ ] **Step 6: Verify the merged main checkout with its own source path**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path + ';' + $PWD.Path
py -3.13 -m pytest tests_py/ai_tuning/hardware/test_flexdca.py tests_py/ai_tuning/v2/continuous/test_dca_evidence.py tests_py/ai_tuning/v2/continuous/test_factory_cli.py tests_py/ai_tuning/test_continuous_live.py -q
git branch --show-current
git rev-parse HEAD
```

Expected: focused merged tests PASS and exact main merge HEAD is recorded.

- [ ] **Step 7: Stop before real hardware execution**

Report B branch/worktree/HEAD, main branch/worktree/merge HEAD, exact tests, preserved experiment directories, and the retained execution policy: manual rounds, manual target, secure API key, automatic Campaign ID, no V2.1 token, explicit `--allow-device-contact`, PRE-FLIGHT verification, one Enter before COMMISSION, I2C-only writes, no BERT/FIR/NLC. Do not launch the signed LIVE script until the user explicitly asks to start the real test.
