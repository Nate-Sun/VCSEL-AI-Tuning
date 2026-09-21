# V2.1 R46 Fast Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an isolated version-B-derived launcher that starts and recovers from the verified round-46 I2C snapshot without changing tuning ranges, Kimi behavior, controller logic, or hardware adapters.

**Architecture:** Keep the validated version B tag and worktree untouched. In the derived `codex/v2.1-continuous-eye-r46-fast` worktree, change only the checked-in B0 I2C values and the launcher's exact branch gate, with offline tests pinning both values while retaining the existing policy and safety assertions.

**Tech Stack:** YAML configuration, PowerShell 7 launcher, Python 3.13, pytest, Git worktrees.

---

### Task 1: Pin the R46 baseline and derived launcher branch with failing tests

**Files:**
- Modify: `tests_py/ai_tuning/test_continuous_live.py:144-226`

- [ ] **Step 1: Replace the checked-in B0 expectation with the complete verified R46 readback**

In `test_checked_in_v21_live_config_has_approved_b0_and_policy`, replace only the expected I2C dictionary with:

```python
    assert baseline.i2c.model_dump() == {
        "ibias_dac": 68,
        "vref": 15,
        "vbias_ref": 15,
        "vga_cur": 11,
        "res_gain": 9,
        "res_bw": 1,
        "gain_ctrl": 8,
        "efin_cur": 15,
        "eq": 23,
    }
```

Keep the existing `live["parameter_policy"] == _policy_config()` assertion and the `policy.max_changed_parameters == 3` assertion unchanged so this test simultaneously proves that only B0 moved.

- [ ] **Step 2: Add an exact derived-branch launcher test**

Immediately after `_signed_live_launcher`, add:

```python
def test_signed_live_launcher_requires_only_r46_derived_branch() -> None:
    launcher = _signed_live_launcher()

    assert (
        "$ExpectedBranch = 'codex/v2.1-continuous-eye-r46-fast'"
        in launcher
    )
    assert "codex/v2.1-continuous-eye-optimization" not in launcher
```

- [ ] **Step 3: Run the two exact tests and verify RED**

Run:

```powershell
py -3.13 -P -m pytest `
  tests_py/ai_tuning/test_continuous_live.py::test_checked_in_v21_live_config_has_approved_b0_and_policy `
  tests_py/ai_tuning/test_continuous_live.py::test_signed_live_launcher_requires_only_r46_derived_branch `
  -q
```

Expected: `2 failed`. The first failure must show the current B0 value `60` versus expected `68`; the second must show that the derived `$ExpectedBranch` assignment is absent. No LIVE entrypoint is invoked.

- [ ] **Step 4: Commit the RED tests**

```powershell
git add -- tests_py/ai_tuning/test_continuous_live.py
git commit -m "test(v2.1): pin R46 fast continuation scope"
```

### Task 2: Apply the minimal R46 configuration and branch-gate changes

**Files:**
- Modify: `configs/ai_tuning/b0.v2_1.continuous.yaml:9-17`
- Modify: `scripts/run_v2_1_continuous_live.ps1:248`

- [ ] **Step 1: Replace only the nine B0 I2C values**

Keep the `baseline_id`, `version`, `pattern`, FIR taps, `upper_nlc`, and `lower_nlc` unchanged. Replace the `i2c` mapping with:

```yaml
i2c:
  ibias_dac: 68
  vref: 15
  vbias_ref: 15
  vga_cur: 11
  res_gain: 9
  res_bw: 1
  gain_ctrl: 8
  efin_cur: 15
  eq: 23
```

- [ ] **Step 2: Change only the launcher's expected branch**

Replace:

```powershell
$ExpectedBranch = 'codex/v2.1-continuous-eye-optimization'
```

with:

```powershell
$ExpectedBranch = 'codex/v2.1-continuous-eye-r46-fast'
```

Do not change `$RequiredImplementationBaseline`, campaign-ID generation, session manifest checks, PRE-FLIGHT/COMMISSION commands, cleanup, or any hardware gate.

- [ ] **Step 3: Re-run the exact tests and verify GREEN**

Run:

```powershell
py -3.13 -P -m pytest `
  tests_py/ai_tuning/test_continuous_live.py::test_checked_in_v21_live_config_has_approved_b0_and_policy `
  tests_py/ai_tuning/test_continuous_live.py::test_signed_live_launcher_requires_only_r46_derived_branch `
  -q
```

Expected: `2 passed`.

- [ ] **Step 4: Inspect the exact production diff**

Run:

```powershell
git diff --check
git diff -- configs/ai_tuning/b0.v2_1.continuous.yaml scripts/run_v2_1_continuous_live.ps1
```

Expected: exactly six YAML value changes (the other three R46 fields were
already `15`) and one `$ExpectedBranch` value change, with all nine R46 fields
still pinned by the test and no whitespace errors.

- [ ] **Step 5: Commit the minimal implementation**

```powershell
git add -- configs/ai_tuning/b0.v2_1.continuous.yaml scripts/run_v2_1_continuous_live.ps1
git commit -m "feat(v2.1): start fast continuation from R46"
```

### Task 3: Prove safety and preserve the validated B lineage

**Files:**
- Verify: `tests_py/ai_tuning/test_continuous_live.py`
- Verify: `tests_py/ai_tuning/v2/continuous/test_controller.py`
- Verify: `tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py`
- Verify: `configs/ai_tuning/v2_1_continuous.live.yaml`

- [ ] **Step 1: Run the complete continuous LIVE configuration and launcher test file**

```powershell
py -3.13 -P -m pytest tests_py/ai_tuning/test_continuous_live.py -q
```

Expected: all tests pass, including PRE-FLIGHT non-write, branch/HEAD/session scope, immutable parameter policy, cleanup, and runbook checks. These are offline fakes/helper-only tests; the launcher main body is not run.

- [ ] **Step 2: Run controller and Kimi failed-safe regression tests**

```powershell
py -3.13 -P -m pytest `
  tests_py/ai_tuning/v2/continuous/test_controller.py `
  tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py `
  -q
```

Expected: all tests pass, including B0 restoration and exhausted truncation retry behavior. No network or hardware is contacted.

- [ ] **Step 3: Run the full offline Python suite**

```powershell
py -3.13 -P -m pytest -q
```

Expected: the suite passes with only its existing declared skips.

- [ ] **Step 4: Parse the launcher with PowerShell 7 without executing it**

```powershell
$pwshPath = 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
& $pwshPath -NoLogo -NoProfile -Command "`$errors = `$null; [void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'scripts/run_v2_1_continuous_live.ps1'), [ref]`$null, [ref]`$errors); if (`$errors.Count) { `$errors | ForEach-Object { Write-Error `$_ }; exit 1 }; 'POWERSHELL_PARSE_OK'"
```

Expected: `POWERSHELL_PARSE_OK`. This parses the file and does not enter the launcher.

- [ ] **Step 5: Verify exact diff scope and original B preservation**

```powershell
git diff --check v2.1-continuous-eye-b-validated-20260817..HEAD
git diff --name-only v2.1-continuous-eye-b-validated-20260817..HEAD
git -C ..\v2-1-continuous-eye-optimization status --short
git rev-parse "v2.1-continuous-eye-b-validated-20260817^{commit}"
```

Expected:

- the derived diff contains only the design document, implementation plan, one test file, the B0 YAML, and the launcher;
- the original B worktree still shows only its pre-existing untracked `artifacts/`, `outputs/`, and `state/` directories;
- the validated tag still resolves to `4085b007546f820820dbdb7f7f0bba6ebc4b2049`.

- [ ] **Step 6: Commit any plan checkbox updates only if they were recorded**

If this plan file was intentionally updated to mark completed checkboxes, commit only that documentation change:

```powershell
git add -- docs/superpowers/plans/2026-08-18-v2-1-r46-fast-continuation.md
git commit -m "docs(v2.1): record R46 continuation verification"
```

If no checkbox changes were recorded, do not create an empty commit.
