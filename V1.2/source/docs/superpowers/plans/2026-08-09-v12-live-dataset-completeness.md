# V1.2 LIVE Dataset Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow native V1.2 LIVE dataset bundles to become `COMPLETE` only when source-bound evidence proves a frozen acquisition under resource-scoped exclusive DCA ownership.

**Architecture:** Keep the existing single acquisition and strict dataset writer. Add a small re-entrant DCA access coordinator around `measure_single()` plus capture, make acquisition guards and optional metrics explicitly select/verify/restore their sources, and derive `same_acquisition_verified` from that evidence instead of a constant.

**Tech Stack:** Python 3.11, dataclasses/context managers/threading, NumPy, pytest, existing FastAPI/React verification commands.

---

### Task 1: Prove source-bound primary acquisition guards

**Files:**
- Modify: `tests_py/ai_tuning/hardware/test_characterization.py`
- Modify: `src/vcsel_ai_tuning/hardware/characterization.py`

- [ ] Add a test session whose OER and TDECQ source queries share one selector. Assert the primary guard selects `CHAN1A` before OER, selects `FUNC1` before TDECQ, returns both authoritative values/counts, and restores `FUNC1`.
- [ ] Run the new test and verify RED because `_acquisition_guard()` currently performs only plain queries.
- [ ] Implement a source-bound guard snapshot with this sequence:

```python
for stem, source in ((":MEASure:EYE:OER", er_source),
                     (":MEASure:EYE:TDEQ", tdecq_source)):
    session.write(f"{stem}:SOURce {source}")
    values[f"{stem}:SOURce?"] = session.query(f"{stem}:SOURce?").strip()
    for suffix in (":STATus?", ":COUNt?", "?"):
        values[f"{stem}{suffix}"] = session.query(f"{stem}{suffix}").strip()
```

- [ ] Restore the original TDECQ/shared source in `finally`, verify readback, and make `primary_match` false on any binding or restoration error.
- [ ] Run the characterization tests and verify GREEN.

### Task 2: Prove source-bound dataset metrics and guard snapshots

**Files:**
- Modify: `tests_py/ai_tuning/v1_1/test_live_dataset_capture.py`
- Modify: `src/vcsel_ai_tuning/v1_1/live_dataset_capture.py`

- [ ] Extend the fake binary session to model shared measurement-source state and complete real response structures.
- [ ] Add RED tests proving OMA/RLM initially bound to `FUNC1` are selected from `CHAN1A`, accepted, and restored; a refused restoration raises; source/count/value drift remains unverified.
- [ ] Replace `_optional_metric()` with a source-bound reader that records:

```python
{
    "original_source": original,
    "requested_source": self._raw_source,
    "verified_source": verified,
    "status": status,
    "units": unit,
    "value": value_text,
    "source_restore": {"readback": restored, "verified": restored == original},
}
```

- [ ] Make `_read_acquisition_guard()` source-bind OER then TDECQ and restore the original shared source.
- [ ] Run the focused capture tests and verify GREEN.

### Task 3: Add resource-scoped exclusive acquisition ownership

**Files:**
- Modify: `tests_py/ai_tuning/v1_1/test_live_sweep.py`
- Modify: `src/vcsel_ai_tuning/v1_1/live_sweep.py`

- [ ] Add a RED integration test whose dataset callback asserts that the same coordinator scope is active continuously from DCA measurement through capture.
- [ ] Implement `_ExclusiveDcaAccess` with an `RLock`, owner thread ID/depth, `claim()` context manager, and `owned_by_current_thread()` verifier.
- [ ] Store one coordinator per normalized DCA resource in a registry guarded by a module lock.
- [ ] Extend `LiveSweepDriverBundle` and `LiveHardwareSweepExecutor` with a scope factory. Wrap `measure_single()`, measurement validation, and `_record_dataset()` in one scope.
- [ ] Construct `FlexDcaDatasetCapture` with `exclusive_access=coordinator.owned_by_current_thread` and expose `coordinator.claim` through the bundle.
- [ ] Run the focused Sweep tests and verify GREEN.

### Task 4: Derive verification and COMPLETE status from evidence

**Files:**
- Modify: `tests_py/ai_tuning/v1_1/test_live_dataset_capture.py`
- Modify: `tests_py/ai_tuning/v1_1/test_dataset.py`
- Modify: `src/vcsel_ai_tuning/v1_1/live_dataset_capture.py`

- [ ] Change the existing consistent-capture test to expect RED on `same_acquisition_verified is True` while an exclusive verifier is active.
- [ ] Add a companion test proving identical evidence without exclusive ownership remains false.
- [ ] Set verification only from:

```python
verified = bool(
    self._exclusive_access()
    and acquisition_snapshot_consistent
    and guard_before["source_restore"]["verified"]
    and guard_after["source_restore"]["verified"]
)
```

- [ ] Record method `source_bound_frozen_snapshot` and the limitation `in_process_resource_scope` in raw evidence; pass the derived value into `FrozenDatasetCapture`.
- [ ] Add/adjust writer coverage proving a verified capture with OMA/RLM/files is `COMPLETE` and an unverified capture is `PARTIAL`.
- [ ] Run capture, dataset, characterization, and Sweep tests and verify GREEN.

### Task 5: Complete non-hardware verification and commit

**Files:** all intentional files above plus this plan.

- [ ] Run targeted tests:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& 'C:\Users\NateS\Documents\VCSEL\.venv-ai-tuning\Scripts\python.exe' -m pytest `
  tests_py\ai_tuning\hardware\test_characterization.py `
  tests_py\ai_tuning\v1_1\test_live_dataset_capture.py `
  tests_py\ai_tuning\v1_1\test_dataset.py `
  tests_py\ai_tuning\v1_1\test_live_sweep.py -q
```

- [ ] Run the full Python suite, frontend tests/build/lint, and `git diff --check`.
- [ ] Inspect the diff for acquisition commands, I2C/BERT/FIR changes, generated artifacts, and accidental lineage mixing.
- [ ] Commit the implementation and confirm `git status --short` is empty.
