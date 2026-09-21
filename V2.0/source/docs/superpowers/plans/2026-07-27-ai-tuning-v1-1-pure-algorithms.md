# AI Tuning V1.1 Pure Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver deterministic, hardware-independent AI Tuning V1.1 objective, discrete I2C coarse/fine/retune, FIR decision/convergence, alternating-stage decisions, sweep generation, replay, and offline evaluation.

**Architecture:** Add an isolated `vcsel_ai_tuning.optimization.v11` package whose inputs and outputs are immutable Pydantic models. The package emits candidates and decisions only; it never imports adapters, opens devices, or creates hardware commands. Existing `hardware.fir.calculate_first_pass()` and `calculate_iterative_pass()` remain the single source of FIR mathematics; the backend injects those pure callables through a formula protocol so the optimization package retains its hardware-import boundary.

**Tech Stack:** Python 3.12+, Pydantic 2, NumPy, pytest; no FastAPI, React, SCPI, I2C transport, BERT transport, or hardware client dependencies.

---

## File map

- `src/vcsel_ai_tuning/optimization/v11/models.py`: immutable V1.1 inputs, observations, fingerprints, candidates, statuses, and decisions.
- `src/vcsel_ai_tuning/optimization/v11/objective.py`: strict TDECQ-first comparison, ER exact-tie break, median aggregation, and best-result selection.
- `src/vcsel_ai_tuning/optimization/v11/grid.py`: exact integer grids and lazy mixed-radix parameter spaces.
- `src/vcsel_ai_tuning/optimization/v11/coarse.py`: deterministic sparse global I2C coverage and coarse completion.
- `src/vcsel_ai_tuning/optimization/v11/fine.py`: NumPy discrete Gaussian-process Bayesian suggestion with explicit `NOT_READY` and readback-keyed training.
- `src/vcsel_ai_tuning/optimization/v11/retune.py`: exact FIR/NLC/DCA history isolation and local Bayesian retune.
- `src/vcsel_ai_tuning/optimization/v11/fir.py`: first/iterative formula selection, direct-target validation, confirmation, and no-improvement convergence.
- `src/vcsel_ai_tuning/optimization/v11/alternating.py`: Fast/Standard/Deep/Advanced outer-round decisions and global-best tracking.
- `src/vcsel_ai_tuning/optimization/v11/sweep.py`: independent lazy I2C/FIR sweep point generation.
- `src/vcsel_ai_tuning/optimization/v11/replay.py`: deterministic offline trace evaluation and digesting.
- `src/vcsel_ai_tuning/optimization/v11/__init__.py`: public V1.1 algorithm API only.
- `tests_py/ai_tuning/optimization/v11/`: focused unit and replay tests matching the modules above.

### Task 1: Version and strict objective contract

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_objective.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/models.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/objective.py`
- Modify: `src/vcsel_ai_tuning/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests_py/ai_tuning/test_package_import.py`

- [ ] **Step 1: Write failing objective and version tests**

```python
def test_tdecq_always_beats_er():
    candidate = MetricResult.valid(tdecq_db=2.99, er_db=1.0)
    reference = MetricResult.valid(tdecq_db=3.00, er_db=100.0)
    assert compare_results(candidate, reference) is Comparison.BETTER

def test_er_breaks_only_an_exact_tdecq_tie():
    low_er = MetricResult.valid(tdecq_db=2.5, er_db=2.0)
    high_er = MetricResult.valid(tdecq_db=2.5, er_db=3.0)
    assert compare_results(high_er, low_er) is Comparison.BETTER

def test_invalid_tdecq_is_never_replaced_by_a_number():
    invalid = MetricResult.invalid("INV", "instrument returned INV")
    assert invalid.tdecq_db is None
    assert compare_results(invalid, MetricResult.valid(3.2, 4.0)) is Comparison.INVALID
```

- [ ] **Step 2: Run the tests and verify import/behavior failures**

Run: `python -m pytest tests_py/ai_tuning/optimization/v11/test_objective.py tests_py/ai_tuning/test_package_import.py -q`

Expected: FAIL because `optimization.v11` does not yet exist and package version is not `1.1.0`.

- [ ] **Step 3: Implement immutable result models and strict comparison**

```python
class Comparison(StrEnum):
    BETTER = "BETTER"
    EQUAL = "EQUAL"
    WORSE = "WORSE"
    INVALID = "INVALID"

def compare_results(candidate: MetricResult, reference: MetricResult) -> Comparison:
    if not candidate.has_valid_tdecq:
        return Comparison.INVALID
    if not reference.has_valid_tdecq:
        return Comparison.BETTER
    if candidate.tdecq_db < reference.tdecq_db:
        return Comparison.BETTER
    if candidate.tdecq_db > reference.tdecq_db:
        return Comparison.WORSE
    if candidate.er_db is not None and reference.er_db is not None:
        if candidate.er_db > reference.er_db:
            return Comparison.BETTER
        if candidate.er_db < reference.er_db:
            return Comparison.WORSE
    return Comparison.EQUAL
```

- [ ] **Step 4: Set package/product version to `1.1.0` and rerun tests**

Expected: all Task 1 tests PASS; no ER threshold or meaningful-improvement epsilon exists.

- [ ] **Step 5: Commit the objective contract**

```powershell
git add pyproject.toml src/vcsel_ai_tuning/__init__.py src/vcsel_ai_tuning/optimization/v11/models.py src/vcsel_ai_tuning/optimization/v11/objective.py tests_py/ai_tuning/test_package_import.py tests_py/ai_tuning/optimization/v11/test_objective.py
git commit -m "feat: define AI Tuning V1.1 objective"
```

### Task 2: Exact grids and deterministic coarse candidates

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_grid.py`
- Create: `tests_py/ai_tuning/optimization/v11/test_coarse.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/grid.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/coarse.py`

- [ ] **Step 1: Write failing grid and coarse tests**

```python
def test_grid_never_forces_unaligned_maximum():
    spec = TuningParameter(name="eq", tuning_min=0, tuning_max=10, tuning_step=3)
    assert spec.grid_values == (0, 3, 6, 9)

def test_baseline_outside_grid_is_not_inserted():
    spec = TuningParameter(name="eq", tuning_min=3, tuning_max=9, tuning_step=3)
    assert 1 not in spec.grid_values

def test_coarse_is_seeded_sparse_and_replayable(space):
    first = tuple(CoarsePlanner(space, budget=5, seed=17).candidates())
    second = tuple(CoarsePlanner(space, budget=5, seed=17).candidates())
    assert first == second
    assert len(first) == 5 < space.point_count
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests_py/ai_tuning/optimization/v11/test_grid.py tests_py/ai_tuning/optimization/v11/test_coarse.py -q`

Expected: FAIL on missing grid/coarse modules.

- [ ] **Step 3: Implement exact integer axes and lazy mixed-radix indexing**

```python
def integer_grid(tuning_min: int, tuning_max: int, tuning_step: int) -> tuple[int, ...]:
    return tuple(range(tuning_min, tuning_max + 1, tuning_step))

def point_from_index(self, original_index: int) -> dict[str, int]:
    digits = []
    remainder = original_index
    for axis in reversed(self.axes):
        remainder, digit = divmod(remainder, len(axis))
        digits.append(axis[digit])
    return dict(zip(self.names, reversed(digits), strict=True))
```

- [ ] **Step 4: Implement a seeded affine permutation for sparse coarse coverage**

Choose a seed-derived offset and a seed-derived stride coprime to `point_count`; emit at most `budget` unique decoded points without constructing the full Cartesian product. Complete with `NO_VALID_TDECQ` if the retained observations contain no valid TDECQ, otherwise retain the strict objective best. Preserve every supplied valid and invalid observation in the summary.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
git add src/vcsel_ai_tuning/optimization/v11/grid.py src/vcsel_ai_tuning/optimization/v11/coarse.py tests_py/ai_tuning/optimization/v11/test_grid.py tests_py/ai_tuning/optimization/v11/test_coarse.py
git commit -m "feat: add deterministic V1.1 I2C coarse planning"
```

### Task 3: Readback-keyed Bayesian fine and exact-context retune

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_fine.py`
- Create: `tests_py/ai_tuning/optimization/v11/test_retune.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/fine.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/retune.py`

- [ ] **Step 1: Write failing fine and retune tests**

```python
def test_fine_trains_on_actual_readback_not_requested(space, fingerprint):
    observation = observation_for(requested={"eq": 9}, readback={"eq": 3}, tdecq=2.4, fingerprint=fingerprint)
    optimizer = DiscreteBayesianFine(space, seed=5)
    assert optimizer.aggregate((observation,))[0].settings == {"eq": 3}

def test_repeated_readbacks_use_tdecq_median(space, fingerprint):
    history = repeats_at_eq_3((2.8, 2.0, 2.4), fingerprint)
    assert DiscreteBayesianFine(space, seed=5).aggregate(history)[0].result.tdecq_db == 2.4

def test_retune_excludes_history_from_other_fir(space, fingerprint):
    other = fingerprint.model_copy(update={"fir_taps": (0, 0, 0, 104, 0, 0, 0)})
    selected = select_retune_history(history_for(fingerprint, other), fingerprint)
    assert all(row.fingerprint == fingerprint for row in selected)
```

- [ ] **Step 2: Verify RED**

Expected: missing fine and retune interfaces.

- [ ] **Step 3: Implement a finite-pool Gaussian-process Bayesian optimizer**

Use a NumPy RBF kernel, Cholesky solve, expected improvement for minimizing raw TDECQ, and a separate kernel-smoothed validity probability. Train regression only on valid numeric TDECQ and never synthesize values for invalid measurements. Exclude already requested grid points while using actual readback coordinates as model features.

```python
acquisition = expected_improvement * valid_probability
chosen = max(predictions, key=lambda item: (item.acquisition, -item.original_index))
```

Return `OptimizerStatus.NOT_READY` with an explanatory reason when NumPy/model fitting is unavailable or materially degenerate. Do not instantiate random or coordinate fallbacks.

- [ ] **Step 4: Implement exact retune history filtering and local-space construction**

`ExperimentFingerprint` includes exactly seven FIR taps, Upper/Lower NLC, and a DCA configuration SHA-256. Retune must filter by complete fingerprint equality before aggregation/model fitting. It reuses the same selected I2C specs and exact min/max/step grid, restricted to a configured local radius around the current readback.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
git add src/vcsel_ai_tuning/optimization/v11/fine.py src/vcsel_ai_tuning/optimization/v11/retune.py tests_py/ai_tuning/optimization/v11/test_fine.py tests_py/ai_tuning/optimization/v11/test_retune.py
git commit -m "feat: add readback-keyed Bayesian fine and retune"
```

### Task 4: Direct FIR formula decisions and convergence

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_fir.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/fir.py`
- Modify: `src/vcsel_ai_tuning/campaign/orchestrator.py`
- Modify: `tests_py/ai_tuning/campaign/test_orchestrator.py`

- [ ] **Step 1: Write failing FIR decision tests**

Cover first-pass ignoring current/B0 convolution, iterative convolution after the first accepted FIR, fixed B0 Main and NLC, exact direct target, authorization envelope and total-change rejection, invalid/missing/nonfinite/wrong-count DCA taps returning `FAILED_INVALID_DCA_TAPS`, first-look confirmation, three-sample median, exact-TDECQ ER tie break, first-pass stop, iterative best-FIR restore recommendation, and convergence after three consecutive non-improvements.

```python
decision = controller.propose_first(dca_taps, dca_main_index=3)
assert decision.candidate.candidate_id == "fir-formula-direct"
assert decision.candidate.model_version == "fir-formula-direct-v1.1"
expected = calculate_first_pass(dca_taps, dca_main_index=3, manual_main=104)
assert decision.candidate.requested_fir.taps == expected.taps
```

- [ ] **Step 2: Verify RED**

Expected: missing V1.1 FIR controller and old orchestrator model version assertion.

- [ ] **Step 3: Implement the pure controller around unchanged formula functions**

Invoke the injected `calculate_first_pass()` callable exactly once for first-pass proposals and the injected `calculate_iterative_pass()` callable for all later proposals. Convert a valid proposal directly into one immutable `FirCandidate`; never interpolate. Validation failures become pure decisions and never retry. The optimization package itself must not import `vcsel_ai_tuning.hardware`.

- [ ] **Step 4: Implement confirmation and convergence**

One better first observation returns `NEEDS_CONFIRMATION` with two remaining measurements. Three observations are reduced with a TDECQ median and ER median, then compared strictly. Accepted candidates reset the no-improvement counter; rejected iterative candidates increment it; three consecutive misses return `CONVERGED_NO_IMPROVEMENT`.

- [ ] **Step 5: Update only the existing direct-FIR identifier to V1.1 and verify formula diff**

Run: `git diff --exit-code 64dc661 -- src/vcsel_ai_tuning/hardware/fir.py tests_py/ai_tuning/hardware/test_fir.py`

Expected: exit 0; formula implementation and tests are unchanged.

- [ ] **Step 6: Commit**

```powershell
git add src/vcsel_ai_tuning/optimization/v11/fir.py tests_py/ai_tuning/optimization/v11/test_fir.py src/vcsel_ai_tuning/campaign/orchestrator.py tests_py/ai_tuning/campaign/test_orchestrator.py
git commit -m "feat: add V1.1 FIR decision controller"
```

### Task 5: FIR/I2C alternating decisions

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_alternating.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/alternating.py`

- [ ] **Step 1: Write failing profile and stop-rule tests**

```python
@pytest.mark.parametrize(("mode", "rounds"), [("FAST", 1), ("STANDARD", 3), ("DEEP", 5)])
def test_profile_round_limits(mode, rounds):
    assert AlternationPolicy.for_mode(mode).max_rounds == rounds

def test_no_fir_improvement_stops_before_retune():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    assert controller.after_fir(round_index=0, improved=False).next_stage == "COMPLETE"

def test_no_retune_improvement_stops_before_next_fir():
    controller = AlternatingController(AlternationPolicy.for_mode("STANDARD"))
    assert controller.after_retune(round_index=0, improved=False).next_stage == "COMPLETE"

def test_both_improvements_advance_until_outer_limit():
    controller = AlternatingController(AlternationPolicy.for_mode("FAST"))
    assert controller.after_fir(round_index=0, improved=True).next_stage == "I2C_RETUNE"
    assert controller.after_retune(round_index=0, improved=True).next_stage == "COMPLETE"
```

- [ ] **Step 2: Verify RED, implement pure decisions, and verify GREEN**

The controller emits next-stage decisions only: `I2C_COARSE`, `I2C_FINE`, `FIR_FIRST`, `I2C_RETUNE`, `FIR_ITERATIVE`, or `COMPLETE`. Advanced requires an explicit positive round count. A strict-objective global best stores the complete I2C settings plus FIR target.

- [ ] **Step 3: Commit**

```powershell
git add src/vcsel_ai_tuning/optimization/v11/alternating.py tests_py/ai_tuning/optimization/v11/test_alternating.py
git commit -m "feat: add V1.1 alternating tuning decisions"
```

### Task 6: Independent lazy Sweep generation

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_sweep.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/sweep.py`

- [ ] **Step 1: Write failing Sweep tests**

Cover `I2C_SWEEP` and `BERT_FIR_SWEEP`, rejection of mixed task inputs, allowlisted I2C names, exact unaligned grids, Single Parameter holding all other values at B0, Full Combination original/execution indices, same-seed equality, different-seed order, lazy iteration, upfront `max_points` rejection, FIR Main `> 0`, FIR `L1 <= 170`, fixed NLC, and unmodified `SKIPPED_CONSTRAINT` points.

- [ ] **Step 2: Verify RED**

Expected: missing sweep API.

- [ ] **Step 3: Implement task/mode models and generators**

```python
class SweepTask(StrEnum):
    I2C_SWEEP = "I2C_SWEEP"
    BERT_FIR_SWEEP = "BERT_FIR_SWEEP"

class SweepMode(StrEnum):
    SINGLE_PARAMETER = "SINGLE_PARAMETER"
    FULL_COMBINATION = "FULL_COMBINATION"
```

Single mode emits one selected-axis assignment at a time over an unchanged B0 mapping. Full mode computes the raw point count, rejects if it exceeds `max_points`, then lazily walks a seed-derived affine permutation of mixed-radix indices. FIR constraint failures retain the exact requested taps and emit `SKIPPED_CONSTRAINT` plus a reason; never scale, clip, or substitute.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
git add src/vcsel_ai_tuning/optimization/v11/sweep.py tests_py/ai_tuning/optimization/v11/test_sweep.py
git commit -m "feat: add independent V1.1 sweep generation"
```

### Task 7: Replay, offline evaluation, and hardware isolation

**Files:**
- Create: `tests_py/ai_tuning/optimization/v11/test_replay.py`
- Create: `tests_py/ai_tuning/optimization/v11/test_import_isolation.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/replay.py`
- Create: `src/vcsel_ai_tuning/optimization/v11/__init__.py`
- Modify: `src/vcsel_ai_tuning/optimization/__init__.py`

- [ ] **Step 1: Write failing deterministic replay and import-boundary tests**

```python
def test_replay_digest_is_independent_of_input_history_order(history):
    assert evaluate_replay(history).digest == evaluate_replay(tuple(reversed(history))).digest

def test_replay_preserves_invalid_measurements_with_null_tdecq(invalid_history):
    report = evaluate_replay(invalid_history)
    assert report.invalid_count == 1
    assert report.observations[0].result.tdecq_db is None

def test_v11_public_api_does_not_import_adapters_or_hardware_clients():
    imported = imported_module_names("vcsel_ai_tuning.optimization.v11")
    assert not any(name.startswith("vcsel_ai_tuning.adapters") for name in imported)
    assert not any("hardware.flexdca" in name for name in imported)
```

- [ ] **Step 2: Verify RED, implement canonical offline reports, and verify GREEN**

Reports include all observations, invalid count, strict-objective best, candidate order, status, and a canonical SHA-256 digest. Export only models, pure planners/controllers, and replay helpers. Do not export command strings or connection objects.

- [ ] **Step 3: Commit**

```powershell
git add src/vcsel_ai_tuning/optimization/v11 src/vcsel_ai_tuning/optimization/__init__.py tests_py/ai_tuning/optimization/v11
git commit -m "feat: add V1.1 replay and pure API boundary"
```

### Task 8: Complete verification and handoff evidence

**Files:**
- Modify only if needed: `docs/superpowers/plans/2026-07-27-ai-tuning-v1-1-pure-algorithms.md`

- [ ] **Step 1: Run focused V1.1 tests**

Run: `python -m pytest tests_py/ai_tuning/optimization/v11 -q`

Expected: all V1.1 tests PASS.

- [ ] **Step 2: Run existing pure optimizer, FIR, and campaign regression tests**

Run: `python -m pytest tests_py/ai_tuning/optimization tests_py/ai_tuning/hardware/test_fir.py tests_py/ai_tuning/campaign/test_orchestrator.py tests_py/ai_tuning/campaign/test_campaign_replay.py -q`

Expected: PASS in a Python environment with compatible NumPy/SciPy/scikit-learn. If the local interpreter ABI cannot import the frozen V1 optional dependencies, separately report that environment limitation and run the maximal unblocked suite.

- [ ] **Step 3: Prove damping absence in production**

Search tracked production files for `generate_damped_fir_candidates`, `fir-damped`, `orchestrator-damping-v1`, `(0.25, 0.5, 1.0)`, and interpolation between current and target FIR. Expected: no production matches.

- [ ] **Step 4: Prove the pure boundary**

Search `src/vcsel_ai_tuning/optimization/v11` for adapter imports, `SCPI`, raw I2C frames, `setFir`, socket/serial/VISA use, and hardware command methods. Expected: no hardware-control path; the only permitted FIR dependency is the pure formula module.

- [ ] **Step 5: Verify repository state**

Run: `git diff --check`, `git status --short --branch`, `git log --oneline codex/ai-tuning-v1.1..HEAD`, and `git rev-parse HEAD`.

Expected: branch remains `codex/v1.1-ai-tuning`; no merge into `main`; tag `ai-tuning-v1.0.0` still points to `64dc661`; commits each have a single responsibility.
