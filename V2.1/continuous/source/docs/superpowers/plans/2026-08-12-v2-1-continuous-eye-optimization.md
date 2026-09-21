# V2.1 Continuous Eye Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the approved continuous eye-optimization controller on commit `1e648755477f7b848d2bd98498a4f8d6438e0e1c`, prove it through deterministic non-hardware tests, and stop for renewed user authorization before any real-device contact.

**Architecture:** Add a provider-neutral deep module under `vcsel_ai_tuning.v2.continuous` that owns policy, action handling, state, evidence, acceptance, and recovery. Reuse the existing V2.0 `StageAuthorization` and `CampaignOrchestrator` as the only execution authority; adapt Kimi from menu selection to sparse absolute I2C proposals without giving the provider hardware objects or execution authority. A single dependency-injected controller runs with fake or live adapters, while the implementation phase permits only fake adapters.

**Tech Stack:** Python 3.12+, Pydantic v2, NumPy, Pillow, PyYAML, pytest, existing `vcsel_ai_tuning` campaign/runtime modules.

---

## Fixed baseline and safety gate

- Repository: `C:\Users\NateS\Documents\VCSEL`
- Implementation worktree: `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization`
- Branch: `codex/v2.1-continuous-eye-optimization`
- Starting HEAD: `1e648755477f7b848d2bd98498a4f8d6438e0e1c`
- Lineage: `1e648755` is based on `codex/v2.1-kimi-live` at `5d04fc1724b0622873990055ac6be6e1765b535d`, itself descended from V2.0 commit `4a4655cace82dedd985abe6ed927c1ac750d331a`.
- Allowed during Tasks 1-13: source edits, unit tests, import tests, fake end-to-end runs, fixture and artifact inspection.
- Forbidden during Tasks 1-13: `RuntimeMode.LIVE_TUNING`, Kimi API/network calls, loading `MOONSHOT_API_KEY`, serial/VISA/socket device discovery, DCA acquisition, I2C read/write, BERT access, FIR/NLC changes, `setFir`, and any Kimi LIVE command.
- Task 14 is a hard authorization checkpoint. Do not run even the LIVE read-only preflight until the user explicitly authorizes real-device contact after reviewing the non-hardware evidence.

Before every implementation session, verify the exact checkout:

```powershell
git status --short --branch
git rev-parse HEAD
git branch --show-current
```

Expected initially: clean `codex/v2.1-continuous-eye-optimization` at `1e648755477f7b848d2bd98498a4f8d6438e0e1c`. If those values differ, stop and reconcile lineage before coding.

## File structure

New production files:

- `src/vcsel_ai_tuning/v2/continuous/models.py` — immutable request, action, round, result, timing, state, and human-decision contracts.
- `src/vcsel_ai_tuning/v2/continuous/policy.py` — the versioned nine-field I2C policy, 15% step calculation, sparse proposal validation, and candidate ID/materialization.
- `src/vcsel_ai_tuning/v2/continuous/provider.py` — provider-neutral decision protocol and bounded context.
- `src/vcsel_ai_tuning/v2/continuous/artifacts.py` — DCA PNG validation, exact PNG-to-NPY derivation, hashes, provenance, and fingerprint checks.
- `src/vcsel_ai_tuning/v2/continuous/store.py` — isolated campaign layout, atomic JSON replacement, append-only events, secret scrubbing, and reload validation.
- `src/vcsel_ai_tuning/v2/continuous/execution.py` — narrow execution port that delegates every write/readback/measurement/rollback to the existing orchestrator.
- `src/vcsel_ai_tuning/v2/continuous/controller.py` — the deep state machine and public `start`, `request_stop`, `resolve_human`, and `get_status` interface.
- `src/vcsel_ai_tuning/v2/continuous/fake.py` — deterministic fake provider, execution port, clock, and scenarios used by non-hardware E2E tests.
- `src/vcsel_ai_tuning/v2/continuous/factory.py` — explicit FAKE/LIVE dependency assembly; construction only, never implicit connection.
- `src/vcsel_ai_tuning/v2/continuous/cli.py` — thin Testbench command that defaults to FAKE and gates LIVE construction behind explicit flags.
- `src/vcsel_ai_tuning/v2/continuous/__init__.py` — deliberately small public surface.
- `scripts/run_continuous_eye.py` — Testbench-style entry point with `--mode FAKE`; LIVE requires an explicit flag and remains unrun before Task 14.
- `configs/ai_tuning/v2_1_continuous.example.yaml` — safe example with `mode: FAKE`, `max_rounds: 3`, and no secrets or device addresses.

Existing production files to modify:

- `src/vcsel_ai_tuning/v2/contracts.py` — allow provider-originated sparse proposals whose local candidate ID is assigned after schema validation.
- `src/vcsel_ai_tuning/v2/providers/kimi_client.py` — structured sparse absolute proposal schema and bounded policy/history payload.
- `src/vcsel_ai_tuning/v2/providers/kimi_provider.py` — implement the provider-neutral interface while retaining image and compact-memory behavior.
- `src/vcsel_ai_tuning/v2/kimi_memory.py` — retain only valid execution summaries as learning history and preserve a separate best reference.
- `src/vcsel_ai_tuning/v2/kimi_step.py` — factor reusable one-step assessment from continuous orchestration; preserve old public behavior for regression compatibility.
- `src/vcsel_ai_tuning/campaign/orchestrator.py` — add narrow I2C-only continuous execution, current-state measurement, and static I2C B0 restore methods; do not duplicate low-level adapter calls elsewhere.
- `src/vcsel_ai_tuning/runtime.py` — add an explicit durable bench lease used across `AWAITING_HUMAN` requests.
- `src/vcsel_ai_tuning/v2/__init__.py` and `pyproject.toml` — export the new API and register the Testbench command.

Tests mirror those responsibilities under `tests_py/ai_tuning/v2/continuous/`, plus focused regressions in existing Kimi, orchestrator, and runtime tests.

### Task 1: Freeze the continuous contracts and public state vocabulary

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/models.py`
- Create: `src/vcsel_ai_tuning/v2/continuous/__init__.py`
- Create: `tests_py/ai_tuning/v2/continuous/__init__.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_models.py`

- [x] **Step 1: Write failing strict-model tests**

Test that `ContinuousRequest`, `ProviderAction`, `RoundRecord`, `CampaignResult`, and `HumanDecision` reject extra fields, booleans masquerading as integers, blank IDs, naive timestamps, non-finite metrics, LIVE results that claim proof without evidence, and inconsistent terminal flags. Assert the exact public states `RUNNING`, `AWAITING_HUMAN`, `COMPLETED`, `FAILED_SAFE`, and `RECOVERY_REQUIRED`.

- [x] **Step 2: Run the model tests and verify collection fails**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_models.py -q
```

Expected: FAIL because `vcsel_ai_tuning.v2.continuous.models` does not exist.

- [x] **Step 3: Implement immutable contracts**

Use strict frozen Pydantic models and these exact enums/interfaces:

```python
class ExecutionMode(StrEnum):
    FAKE = "FAKE"
    LIVE = "LIVE"

class ContinuousState(StrEnum):
    RUNNING = "RUNNING"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED_SAFE = "FAILED_SAFE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

class ProviderActionKind(StrEnum):
    PROPOSE = "PROPOSE"
    HOLD = "HOLD"
    STOP = "STOP"
    REQUEST_HUMAN = "REQUEST_HUMAN"

class HumanDecision(StrEnum):
    RESUME = "RESUME"
    ACCEPT = "ACCEPT"
    RESTORE_B0 = "RESTORE_B0"
```

`ContinuousRequest` must carry `campaign_id`, `initial_eye`, `authorization`, `policy`, `max_rounds`, `provider`, stopping/acceptance policy, B0 path/reference, result directory, execution mode, and `human_timeout_s`. `CampaignResult` must separately record `state`, `target_reached`, `acceptance_passed`, `closed_loop_proven`, and `stop_reason`.

- [x] **Step 4: Run focused tests**

Expected: PASS, with no network or hardware imports during test collection.

- [x] **Step 5: Commit**

```powershell
git add src/vcsel_ai_tuning/v2/continuous tests_py/ai_tuning/v2/continuous
git commit -m "feat: define continuous optimization contracts"
```

### Task 2: Encode the complete I2C policy and deterministic candidate materialization

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/policy.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_policy.py`
- Modify: `src/vcsel_ai_tuning/v2/contracts.py`
- Modify: `tests_py/ai_tuning/v2/test_contracts.py`

- [x] **Step 1: Write failing policy tests**

Cover all nine names and ranges: `ibias_dac 0..127`, `gain_ctrl 0..63`, `eq 0..31`, and `efin_cur`, `vga_cur`, `res_gain`, `res_bw`, `vbias_ref`, `vref` at `0..15`. Assert `ceil(0.15 * (max-min))` produces steps `20, 10, 5, 3, 3, 3, 3, 3, 3`; values are exact integers; at most three fields change; unchanged proposed values are removed; unknown, clipped, rounded, over-step, out-of-range, and dense four-change proposals fail with deterministic codes.

- [x] **Step 2: Run tests and verify failure**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_policy.py tests_py/ai_tuning/v2/test_contracts.py -q
```

Expected: FAIL because the policy and provider-originated sparse proposal contract are absent.

- [x] **Step 3: Implement policy and IDs**

Define `I2cParameterPolicy`, `ParameterRule`, and `ProposalValidationError`. Materialize a `campaign.models.Candidate` only after validation, with:

```python
canonical = json.dumps(requested, sort_keys=True, separators=(",", ":"))
candidate_id = "-".join((campaign_id, f"r{round_index:04d}", f"a{attempt}", sha256(canonical.encode()).hexdigest()[:16]))
```

Set `stage=authorization.stage`, `requested_fir=None`, and preserve sparse `requested_i2c`. The local program owns `candidate_id`; provider output must not supply it.

- [x] **Step 4: Re-run focused tests**

Expected: PASS for exact boundaries and deterministic IDs.

- [x] **Step 5: Commit**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/policy.py src/vcsel_ai_tuning/v2/contracts.py tests_py/ai_tuning/v2
git commit -m "feat: validate sparse continuous I2C proposals"
```

### Task 3: Introduce the provider-neutral decision boundary

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/provider.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_provider.py`
- Modify: `src/vcsel_ai_tuning/v2/providers/fake.py`

- [x] **Step 1: Write protocol and context tests**

Assert a runtime-checkable `ContinuousDecisionProvider` receives only a frozen `ProviderContext`: current `EyeSample`, verified readback, full policy, round counts, target, at most five valid summaries, separate best reference, policy version, and allowed output schema. Assert no context field can contain an adapter, callable, API key, serial/VISA object, `apply`, `setFir`, FIR, NLC, or BERT command.

- [x] **Step 2: Verify tests fail, then implement**

The protocol is:

```python
@runtime_checkable
class ContinuousDecisionProvider(Protocol):
    @property
    def provider_id(self) -> str: ...
    @property
    def provider_version(self) -> str: ...
    def decide(self, context: ProviderContext) -> ProviderAction: ...
```

Use an explicit versioned `ReasonCode` enum containing target status, expected improvement, exploration, learned negative result, exhausted search, low confidence, OOD, invalid observation, and need-human codes. Unknown codes must fail validation.

- [x] **Step 3: Run tests**

Expected: PASS and import isolation succeeds without optional Kimi dependencies.

- [x] **Step 4: Commit**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/provider.py src/vcsel_ai_tuning/v2/providers/fake.py tests_py/ai_tuning/v2/continuous
git commit -m "feat: add continuous decision provider boundary"
```

### Task 4: Change Kimi from candidate-menu selection to sparse absolute proposals

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/providers/kimi_client.py`
- Modify: `src/vcsel_ai_tuning/v2/providers/kimi_provider.py`
- Modify: `src/vcsel_ai_tuning/v2/kimi_memory.py`
- Modify: `src/vcsel_ai_tuning/v2/kimi_step.py`
- Modify: `tests_py/ai_tuning/v2/test_kimi_provider.py`
- Modify: `tests_py/ai_tuning/v2/test_kimi_memory.py`
- Modify: `tests_py/ai_tuning/v2/test_kimi_step.py`

- [x] **Step 1: Replace menu-based test expectations**

Make the fake SDK return `requested_params`, `confidence`, `summary`, `reason_codes`, and one of the four actions. Assert `PROPOSE` requires 1-3 sparse integer fields and no `candidate_id`; non-PROPOSE forbids parameters. Assert the prompt contains the nine versioned rules, latest verified readback, round budget, target, valid history, and best-reference metadata, but no executable object or secret.

- [x] **Step 2: Run focused Kimi tests and verify failure**

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py tests_py/ai_tuning/v2/test_kimi_memory.py tests_py/ai_tuning/v2/test_kimi_step.py -q
```

Expected: FAIL because current code requires Kimi to select `candidate_id` from `safe_candidates`.

- [x] **Step 3: Implement the new strict JSON schema**

Change `KIMI_RESPONSE_FORMAT` so `requested_params` is an object with only the nine integer properties, `minProperties: 1`, `maxProperties: 3`, and `additionalProperties: false`. Add `summary` with a bounded nonblank length. Keep model `kimi-k3`, official Moonshot base URL, environment-only `MOONSHOT_API_KEY`, byte limits, strict response parsing, and hash/evidence behavior.

- [x] **Step 4: Preserve legacy one-step compatibility**

Factor the assessment portion of `execute_kimi_step` into a provider-neutral helper. Keep `submit_kimi_step` callable for existing tests, but make continuous orchestration materialize and authorize the returned sparse proposal locally. Record only valid executed results in normal learning history; failures remain audit events and never become positive or negative experience.

- [x] **Step 5: Run all V2 Kimi regressions**

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py tests_py/ai_tuning/v2/test_kimi_memory.py tests_py/ai_tuning/v2/test_kimi_step.py tests_py/ai_tuning/v2/test_factory.py tests_py/ai_tuning/v2/test_fail_closed.py -q
```

Expected: PASS using fake SDK clients only; no external request.

- [x] **Step 6: Commit**

```powershell
git add src/vcsel_ai_tuning/v2 tests_py/ai_tuning/v2
git commit -m "feat: accept sparse Kimi I2C proposals"
```

### Task 5: Build exact DCA screen-image evidence

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/artifacts.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_artifacts.py`
- Create: `tests_py/ai_tuning/v2/continuous/fixtures/eye.png`

- [x] **Step 1: Write failing artifact tests**

Assert the original DCA screen PNG is saved unchanged; a deterministic RGB provider PNG is saved at maximum width 1280 using aspect-preserving LANCZOS resize; the derived NPY remains a deterministic RGB/RGBA pixel array from the original PNG; all hashes and the provider-image source hash are verified after write; provenance is explicit; `EyeSample.artifact_hashes` links every reference; incompatible DCA fingerprints fail acceptance; corrupt PNG, NPY, hash, or traversal path fails closed.

- [x] **Step 2: Implement `EyeArtifactBundle.from_png`**

Use Pillow only to decode the already captured PNG, NumPy to save the pixel array, SHA-256 for each payload, and atomic same-directory replace. Never label the array as raw eye density.

- [x] **Step 3: Run tests**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_artifacts.py -q
```

Expected: PASS with fixture-only file I/O.

- [x] **Step 4: Commit**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/artifacts.py tests_py/ai_tuning/v2/continuous
git commit -m "feat: persist continuous eye evidence"
```

### Task 6: Add isolated, atomic campaign persistence

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/store.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_store.py`

- [x] **Step 1: Write persistence and recovery tests**

Cover `<result_dir>/<campaign_id>/campaign.json`, atomic `state.json`, append-only `events.jsonl`, zero-padded `rounds/0001`, `acceptance/`, and `recovery/`. Simulate replace failure and truncated JSONL; assert the previous state remains readable or reload returns a deterministic recovery error. Assert values named `MOONSHOT_API_KEY`, `api_key`, `authorization`, and bearer tokens are scrubbed recursively.

- [x] **Step 2: Implement `ContinuousCampaignStore`**

Expose only `initialize`, `load_state`, `save_state`, `append_event`, `save_round`, `save_acceptance`, and `save_recovery`. Reject path escape and campaign ID reuse with conflicting immutable metadata. Flush and `fsync` before `os.replace` or before completing an append.

- [x] **Step 3: Run tests and commit**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_store.py -q
git add src/vcsel_ai_tuning/v2/continuous/store.py tests_py/ai_tuning/v2/continuous/test_store.py
git commit -m "feat: persist continuous campaign state"
```

Expected: PASS; no files outside pytest temporary directories.

### Task 7: Expose a narrow I2C-only execution port through the existing orchestrator

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/execution.py`
- Modify: `src/vcsel_ai_tuning/campaign/orchestrator.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_execution.py`
- Modify: `tests_py/ai_tuning/campaign/test_orchestrator.py`

- [x] **Step 1: Write spy-adapter tests**

Assert continuous execution delegates to `CampaignOrchestrator`, validates against latest readback, applies only sparse I2C, requires exact applied/readback equality, captures one new DCA observation, and returns immutable evidence. Assert acceptance measurement performs no write. Assert static B0 restore writes only the nine `baseline.i2c` values and never calls `bert.apply`, `bert.readback`, FIR, NLC, or `setFir`.

- [x] **Step 2: Verify current orchestrator lacks the I2C-only operations**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_execution.py tests_py/ai_tuning/campaign/test_orchestrator.py -q
```

Expected: FAIL only in new continuous tests.

- [x] **Step 3: Implement the narrow port**

Add orchestrator public methods `run_continuous_i2c_candidate`, `measure_current_i2c_state`, and `restore_static_i2c_b0`. Reuse `_reserve_authorized_candidate`, `validate_candidate`, `_apply_i2c_candidate`, `_persist_measurement`, `_hardware_failure`, and rollback helpers. Do not expose raw adapters from the port and do not alter legacy `restore_b0`, whose BERT behavior belongs to the older campaign flow.

- [x] **Step 4: Run orchestrator and authorization regressions**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_execution.py tests_py/ai_tuning/campaign/test_orchestrator.py tests_py/ai_tuning/campaign/test_authorization.py -q
```

Expected: PASS, including explicit assertions that FIR/BERT spies received zero calls.

- [x] **Step 5: Commit**

```powershell
git add src/vcsel_ai_tuning/campaign/orchestrator.py src/vcsel_ai_tuning/v2/continuous/execution.py tests_py/ai_tuning
git commit -m "feat: expose I2C-only continuous execution port"
```

### Task 8: Implement the normal continuous controller loop

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/controller.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_controller_rounds.py`

- [x] **Step 1: Write failing happy-path and max-round tests**

With deterministic fakes, assert exact order: validate eye/readback, build context, decide, materialize, authorize, execute, exact readback, acquire new eye, save artifacts, atomically save round, append valid history, choose next state. Assert `completed_rounds` increments only when executor success, distinct bound observation IDs, exact proposal/readback, valid new DCA observation, and durable evidence all hold.

- [x] **Step 2: Implement the public controller surface**

```python
class ContinuousOptimizationController:
    def start(self, request: ContinuousRequest) -> CampaignResult: ...
    def request_stop(self, campaign_id: str) -> ContinuousStatus: ...
    def resolve_human(self, campaign_id: str, decision: HumanDecision) -> CampaignResult: ...
    def get_status(self, campaign_id: str) -> ContinuousStatus: ...
```

Dependencies are constructor-injected: provider, execution port, clock, store, and artifact service. The controller must never create a Kimi client, hardware adapter, or connection.

- [x] **Step 3: Record timing with two clocks**

Use monotonic values for provider, validation, I2C write, readback, DCA, persistence, round, active, wait, and elapsed durations. Use timezone-aware wall time for audit timestamps. Persist completed timings even on failure.

- [x] **Step 4: Run tests and commit**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_controller_rounds.py -q
git add src/vcsel_ai_tuning/v2/continuous/controller.py tests_py/ai_tuning/v2/continuous/test_controller_rounds.py
git commit -m "feat: run continuous optimization rounds"
```

Expected: PASS entirely in memory/temporary directories.

### Task 9: Add provider retries, STOP acceptance, and target handling

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/continuous/controller.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_controller_actions.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_acceptance.py`

- [x] **Step 1: Write action-matrix tests**

Cover: one invalid proposal correction against the same observation; second invalid ends safely; one eligible HOLD retry; second HOLD ends; low confidence, invalid eye, OOD, and provider exception end immediately; STOP and local `TDECQ < 2.7` share one acceptance path without a second provider call; provider STOP may complete with `target_reached=false`; baseline STOP records `BASELINE_ACCEPTED_WITHOUT_EXPLORATION` and never proves the loop.

- [x] **Step 2: Write acceptance tests**

Require the same verified I2C state, compatible fingerprint, two valid observations, `abs(TDECQ delta) <= 0.2`, and `abs(Outer ER delta) <= 0.2`. Acceptance remeasurement is stored under `acceptance/`, does not increment rounds, and performs no I2C write.

- [x] **Step 3: Implement and run**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_controller_actions.py tests_py/ai_tuning/v2/continuous/test_acceptance.py -q
```

Expected: PASS with exact stop reasons and provider call counts.

- [x] **Step 4: Commit**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/controller.py tests_py/ai_tuning/v2/continuous
git commit -m "feat: handle continuous stop and acceptance"
```

### Task 10: Add failure rollback, static I2C B0 recovery, and proof rules

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/continuous/controller.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_recovery.py`
- Modify: `tests_py/ai_tuning/campaign/test_orchestrator.py`

- [x] **Step 1: Write the failure matrix**

Inject readback mismatch, DCA failure, observation-ID reuse, artifact failure, persistence failure, rollback failure, and static B0 verification failure. Assert no later automatic round runs. Successful rollback plus verified static I2C B0 produces `FAILED_SAFE`; unverified rollback/B0 produces `RECOVERY_REQUIRED`.

- [x] **Step 2: Assert proof is evidence-derived**

`closed_loop_proven` is true only for LIVE mode with recorded adapter identities, at least one completed real proposal round, exact I2C readback, a new valid DCA PNG/NPY observation, durable evidence, and verified static I2C B0 restoration. FAKE mode must always serialize `closed_loop_proven: false`, even if all fake evidence is otherwise complete.

- [x] **Step 3: Implement, run, and commit**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_recovery.py tests_py/ai_tuning/campaign/test_orchestrator.py -q
git add src/vcsel_ai_tuning/v2/continuous/controller.py tests_py/ai_tuning
git commit -m "feat: recover continuous campaigns safely"
```

Expected: PASS; BERT/FIR spies remain untouched.

### Task 11: Hold a durable bench lease while awaiting a human

**Files:**
- Modify: `src/vcsel_ai_tuning/runtime.py`
- Modify: `src/vcsel_ai_tuning/v2/continuous/controller.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_human_wait.py`
- Modify: `tests_py/ai_tuning/test_runtime_registry.py`

- [x] **Step 1: Write lease and decision tests**

Assert `REQUEST_HUMAN` persists `AWAITING_HUMAN`, keeps the current non-B0 verified I2C state, performs no automatic action, blocks another bench owner, and accepts only `RESUME`, `ACCEPT`, or `RESTORE_B0`. Assert the configured 15-minute default timeout attempts I2C-only B0 restore. Lost lease, restart without provable continuity, or timeout recovery mismatch yields `RECOVERY_REQUIRED` and never auto-resumes.

- [x] **Step 2: Implement `DurableBenchLease`**

Back the lease with the existing runtime SQLite owner mechanism and persist owner, campaign, resource IDs, issued time, expiry, and continuity token. Acquisition and release must be explicit; process-local `HardwareResourceRegistry` remains the inner fence.

- [x] **Step 3: Run tests and commit**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_human_wait.py tests_py/ai_tuning/test_runtime_registry.py -q
git add src/vcsel_ai_tuning/runtime.py src/vcsel_ai_tuning/v2/continuous/controller.py tests_py/ai_tuning
git commit -m "feat: lease bench during human wait"
```

Expected: PASS with temporary SQLite files only.

### Task 12: Assemble explicit FAKE and LIVE dependency graphs without implicit contact

**Files:**
- Create: `src/vcsel_ai_tuning/v2/continuous/fake.py`
- Create: `src/vcsel_ai_tuning/v2/continuous/factory.py`
- Modify: `src/vcsel_ai_tuning/v2/continuous/__init__.py`
- Modify: `src/vcsel_ai_tuning/v2/__init__.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_factory.py`
- Modify: `tests_py/ai_tuning/v2/test_import_isolation.py`

- [x] **Step 1: Write assembly and import-isolation tests**

Assert importing the package opens no socket, serial port, VISA resource, file outside configuration, or environment secret. `build_continuous_dependencies(FAKE)` returns deterministic fake provider/execution/artifact/store dependencies. LIVE construction requires explicit config but still does not connect; contact occurs only inside an explicitly invoked preflight/start operation.

- [x] **Step 2: Implement factories**

The FAKE factory must support scripted actions and critical faults. The LIVE factory reuses `KimiApiClient`, `KimiProvider`, `CampaignRuntime`, production orchestrator entries, and existing hardware adapters; it must not introduce a second Kimi, I2C, DCA, or safety path.

- [x] **Step 3: Run tests and commit**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_factory.py tests_py/ai_tuning/v2/test_import_isolation.py -q
git add src/vcsel_ai_tuning/v2 src/vcsel_ai_tuning/runtime.py tests_py/ai_tuning/v2
git commit -m "feat: assemble continuous optimization dependencies"
```

Expected: PASS with network and hardware constructors replaced by tripwire fakes.

### Task 13: Add the Testbench command and complete non-hardware verification

**Files:**
- Create: `scripts/run_continuous_eye.py`
- Create: `src/vcsel_ai_tuning/v2/continuous/cli.py`
- Create: `configs/ai_tuning/v2_1_continuous.example.yaml`
- Modify: `pyproject.toml`
- Create: `tests_py/ai_tuning/v2/continuous/test_cli.py`
- Create: `tests_py/ai_tuning/v2/continuous/test_fake_e2e.py`
- Create: `docs/v2-1-continuous-eye-testbench.md`

- [x] **Step 1: Write CLI safety tests**

Assert default/example mode is FAKE; FAKE does not read `MOONSHOT_API_KEY`; `--mode LIVE` requires `--allow-device-contact` plus a separate nonempty authorization token supplied at invocation; missing either exits before factory construction. Assert no `--set-fir`, FIR, NLC, or BERT option exists.

- [x] **Step 2: Implement the thin Testbench**

It may load config, assemble dependencies, call the controller, display state/timing/result, and exit. It must not own loops, proposals, safety checks, I2C/DCA calls, or recovery. Register `vcsel-continuous-eye = "vcsel_ai_tuning.v2.continuous.cli:main"` in `pyproject.toml`; keep `scripts/run_continuous_eye.py` as a minimal wrapper, with the actual CLI module in `src/vcsel_ai_tuning/v2/continuous/cli.py`.

- [x] **Step 3: Run one FAKE happy path and four critical fake faults**

```powershell
python -m pytest tests_py/ai_tuning/v2/continuous/test_cli.py tests_py/ai_tuning/v2/continuous/test_fake_e2e.py -q
python scripts/run_continuous_eye.py --config configs/ai_tuning/v2_1_continuous.example.yaml --mode FAKE --scenario happy
python scripts/run_continuous_eye.py --config configs/ai_tuning/v2_1_continuous.example.yaml --mode FAKE --scenario readback-mismatch
python scripts/run_continuous_eye.py --config configs/ai_tuning/v2_1_continuous.example.yaml --mode FAKE --scenario dca-failure
python scripts/run_continuous_eye.py --config configs/ai_tuning/v2_1_continuous.example.yaml --mode FAKE --scenario b0-recovery-failure
python scripts/run_continuous_eye.py --config configs/ai_tuning/v2_1_continuous.example.yaml --mode FAKE --scenario stop-acceptance
```

Expected: happy and STOP scenarios finish normally; injected faults finish `FAILED_SAFE` or `RECOVERY_REQUIRED` exactly as designed; every output records `execution_mode: FAKE` and `closed_loop_proven: false`.

- [x] **Step 4: Run the complete non-hardware suite**

```powershell
python -m pytest tests_py/ai_tuning/v2 tests_py/ai_tuning/campaign tests_py/ai_tuning/test_runtime_registry.py tests_py/ai_tuning/test_hardware_authority.py -q
python -m pytest -q
```

Expected: all tests PASS. Hardware-named tests must use fakes/mocks only; do not run any marker or script that opens real devices.

- [x] **Step 5: Inspect evidence and scan safety boundaries**

```powershell
rg -n "setFir|requested_fir|bert\.apply|RuntimeMode\.LIVE_TUNING|MOONSHOT_API_KEY" src/vcsel_ai_tuning/v2/continuous scripts/run_continuous_eye.py configs/ai_tuning/v2_1_continuous.example.yaml
git diff 1e648755 --check
git status --short
```

Expected: no `setFir` call; no FIR/BERT mutation in the continuous package; LIVE and key references occur only in explicit guarded assembly/CLI code; `git diff --check` is clean.

- [x] **Step 6: Document evidence and commit**

Document exact branch, worktree, starting commit, final HEAD, execution policy, test commands/results, FAKE artifact paths, and explicit statement that neither Kimi LIVE nor real hardware was contacted.

```powershell
git add scripts/run_continuous_eye.py configs/ai_tuning/v2_1_continuous.example.yaml pyproject.toml src/vcsel_ai_tuning/v2/continuous/cli.py tests_py/ai_tuning/v2/continuous docs/v2-1-continuous-eye-testbench.md
git commit -m "feat: verify continuous eye controller in fake mode"
```

### Task 14: Stop and request authorization before real-device contact

**Files:**
- No source change is required before asking.
- After authorization only, evidence would be added under the configured campaign result directory and summarized in `docs/v2-1-continuous-eye-testbench.md`.

- [ ] **Step 1: Present the non-hardware evidence to the user**

Report exact branch/worktree/final HEAD, all test totals, FAKE scenario results, and safety scan. Explicitly state that Kimi LIVE, DCA, I2C, BERT, serial, VISA, and device network endpoints were not contacted.

- [ ] **Step 2: Ask for separate authorization**

Describe the proposed next operation precisely: first a LIVE read-only preflight that checks configured identities, Kimi connectivity, DCA PNG acquisition, and current I2C readback without writing; only after reviewing that preflight would a separately authorized `max_rounds: 3` commissioning run be proposed.

- [ ] **Step 3: Wait**

Do not run the preflight, load LIVE dependencies, connect to Kimi, inspect real device identities, acquire a DCA PNG, read I2C, or perform any write until the user explicitly grants authorization. Approval for implementation or FAKE verification does not count as approval for this task.

## Final self-review checklist

- Every approved design section maps to Tasks 1-14: provider schema/policy (2-4), controller/flow/timing (8-9), artifacts/evidence (5-6), existing execution authority and I2C-only B0 (7, 10), human wait/lease (11), modes/Testbench (12-13), and live authorization (14).
- FIR, NLC, BERT, and `setFir` remain outside the continuous execution path.
- `COMPLETED` remains distinct from target reached, acceptance passed, and closed-loop proof.
- A provider action never authorizes itself; local immutable candidate materialization and V2.0 deterministic validation remain mandatory.
- All automatic execution stops on invalid evidence, persistence failure, or uncertain recovery.
- Non-hardware implementation ends at Task 13. Task 14 is deliberately not an automatic continuation.
