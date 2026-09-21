# Short Observation Staging Name Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent V2.1 candidate-observation persistence failures on Windows by using fixed-length sibling temporary names while preserving every durable identifier and artifact path.

**Architecture:** `CampaignRepository` continues to stage in the final artifact's parent directory and atomically promote the staged file. Only the unpublished temporary basename changes; a Windows path-budget guard reports deterministic length diagnostics before filesystem publication.

**Tech Stack:** Python 3.13, pathlib, pytest, SQLite-backed `CampaignRepository`.

---

### Task 1: Reproduce and fix the long temporary basename

**Files:**
- Modify: `tests_py/ai_tuning/campaign/test_repository.py`
- Modify: `src/vcsel_ai_tuning/campaign/repository.py:756-797`

- [ ] **Step 1: Write the failing regression test**

Add a repository test that publishes the exact incident-shaped observation ID, wraps `_stage_artifact`, and captures the temporary basename before promotion:

```python
import os
import re

campaign_id = "v21-b-20260817T090138707Z-6a4329e1"
candidate_id = f"{campaign_id}-r0001-a1-329c6930854c388c"
observation_id = f"i2c_screening-{candidate_id}-3"
temporary_names: list[str] = []

with CampaignRepository(tmp_path, campaign_id) as repo:
    original_stage = repo._stage_artifact

    def capture_stage(*args: object, **kwargs: object) -> object:
        staged = original_stage(*args, **kwargs)
        if staged.temporary is not None:
            temporary_names.append(staged.temporary.name)
        return staged

    repo._stage_artifact = capture_stage
    published = repo.publish_observation(observation_id, {"metric": 6.39}, ())

assert published.observation_id == observation_id
assert temporary_names
assert all(re.fullmatch(r"\.stage-[0-9a-f]{32}\.tmp", name) for name in temporary_names)
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -3.13 -m pytest tests_py/ai_tuning/campaign/test_repository.py -k "fixed_length_staging_name" -q
```

Expected: FAIL because the existing temporary basename repeats the complete final filename.

- [ ] **Step 3: Implement the fixed-length sibling name**

In `_stage_artifact`, replace the derived basename with:

```python
temporary = target.with_name(f".stage-{uuid4().hex}.tmp")
```

Keep the temporary in `target.parent`; do not change `_promote_staged`, database paths, observation IDs, or cleanup logic.

- [ ] **Step 4: Run the regression test and verify GREEN**

Run the Step 2 command.

Expected: PASS.

### Task 2: Add a deterministic Windows path-budget diagnostic

**Files:**
- Modify: `tests_py/ai_tuning/campaign/test_repository.py`
- Modify: `src/vcsel_ai_tuning/campaign/repository.py`

- [ ] **Step 1: Write the failing path-budget test**

On Windows, construct a repository root whose final observation path has at least 260 characters, then assert publication fails before staging with a diagnostic containing the actual length and limit:

```python
@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH guard")
def test_publish_reports_windows_path_budget_before_staging(tmp_path):
    # Pad only the repository root; keep the directory path itself creatable.
    campaign_id = "campaign-1"
    observation_id = "obs-" + "x" * 120
    with CampaignRepository(tmp_path, campaign_id) as repo:
        target = repo.run_dir / "observations" / f"{observation_id}.json"
        if len(str(target.resolve())) < 260:
            observation_id += "y" * (260 - len(str(target.resolve())))
        with pytest.raises(RepositoryError, match=r"path length \d+ exceeds Windows limit 259"):
            repo.publish_observation(observation_id, {"metric": 1.0}, ())
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -3.13 -m pytest tests_py/ai_tuning/campaign/test_repository.py -k "windows_path_budget" -q
```

Expected: FAIL because the current error is an unstructured filesystem failure.

- [ ] **Step 3: Implement the guard**

Add a focused helper and call it for both the final and temporary path before opening the temporary file:

```python
_WINDOWS_PATH_LIMIT = 259


def _validate_windows_path_budget(path: Path) -> None:
    if os.name != "nt":
        return
    resolved = path.resolve(strict=False)
    length = len(os.fspath(resolved))
    if length > _WINDOWS_PATH_LIMIT:
        raise RepositoryError(
            f"artifact path length {length} exceeds Windows limit "
            f"{_WINDOWS_PATH_LIMIT}: {resolved}"
        )
```

Call `_validate_windows_path_budget(target)` after the contained path is re-resolved and `_validate_windows_path_budget(temporary)` before `_reject_reparse_point(temporary)`.

- [ ] **Step 4: Run repository tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -3.13 -m pytest tests_py/ai_tuning/campaign/test_repository.py -q
```

Expected: all tests pass; only existing Windows symlink-permission cases may skip.

- [ ] **Step 5: Commit the implementation**

```powershell
git add src/vcsel_ai_tuning/campaign/repository.py tests_py/ai_tuning/campaign/test_repository.py
git commit -m "fix(v2.1): shorten observation staging names"
```

### Task 3: Verify integration and merge

**Files:**
- Verify only; no additional production files.

- [ ] **Step 1: Run focused continuous and repository regression tests**

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -3.13 -m pytest tests_py/ai_tuning/campaign/test_repository.py tests_py/ai_tuning/v2/continuous tests_py/ai_tuning/test_continuous_live.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the complete B suite**

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -3.13 -m pytest -q
```

Expected: zero failures; Windows symlink tests may skip for privilege reasons.

- [ ] **Step 3: Review and merge to main**

Run `git diff --check`, obtain independent specification and quality review, merge `codex/v2.1-continuous-eye-optimization` into `main`, resolve conflicts by preserving both intents, and rerun the focused tests from the committed main HEAD. Do not remove the B worktree or evidence directories.
