# V2.1 LIVE Preflight Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a new V2.1 Campaign run an I2C-only LIVE preflight and Kimi provider preview without a mutating-stage authorization.

**Architecture:** Add a read-only preflight request and an orchestrator capture method at the existing continuous seam. Bootstrap Campaign metadata locally, then explicitly construct hardware adapters only inside the authorized preflight call. Keep `ContinuousRequest` and the execution port unchanged for commissioning.

**Tech Stack:** Python 3.13, Pydantic, SQLite, pytest, YAML, existing Campaign runtime and V2.1 continuous controller.

---

### Task 1: Add the authorization-free provider preflight request

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/continuous/models.py`
- Modify: `src/vcsel_ai_tuning/v2/continuous/controller.py`
- Test: `tests_py/ai_tuning/v2/continuous/test_controller.py`

- [x] Write a failing test that constructs a provider-preview request without
  `StageAuthorization`, receives the scripted provider action, and records zero
  execution-port calls.
- [x] Run the focused test and confirm it fails because the request model or
  accepted request type does not exist.
- [x] Add `ContinuousPreflightRequest` with campaign, initial eye, policy,
  `max_rounds`, and stopping policy only; update `preview_initial_decision` to
  accept it while leaving `start` restricted to `ContinuousRequest`.
- [x] Run the focused controller tests and commit.

### Task 2: Add I2C-only Campaign preflight capture

**Files:**
- Modify: `src/vcsel_ai_tuning/campaign/orchestrator.py`
- Test: `tests_py/ai_tuning/campaign/test_orchestrator.py`

- [x] Write failing tripwire tests for a new Campaign and a repeated preflight.
  Assert DCA/I2C identity, current readback, one observation, and zero BERT or
  I2C-write calls.
- [x] Run the focused tests and confirm the new method is missing.
- [x] Implement the smallest I2C-only capture operation, persist its evidence,
  and use existing state transitions only for the new-Campaign case.
- [x] Run the focused campaign tests and commit.

### Task 3: Bootstrap LIVE composition without pre-existing authorization

**Files:**
- Modify: `src/vcsel_ai_tuning/continuous_live.py`
- Modify: `src/vcsel_ai_tuning/v2/continuous/factory.py`
- Test: `tests_py/ai_tuning/v2/continuous/test_factory_cli.py`
- Test: `tests_py/ai_tuning/hardware/test_live_runtime_factory.py`

- [x] Write failing tests that prove Campaign creation is local-only, preflight
  accepts a new Campaign, Kimi preview uses the read-only request, and
  commissioning activation still requires persisted authorization.
- [x] Run the tests and confirm failure at the existing `_latest_authorization`
  dependency.
- [x] Add a dedicated preflight composition path. It creates Campaign metadata,
  opens DCA/I2C only after the explicit contact gate, captures the observation,
  calls controller preview, closes resources, and returns the report.
- [x] Run focused factory/runtime tests and commit.

### Task 4: Add isolated V2.1 LIVE configuration

**Files:**
- Create: `configs/hardware/vcsel_lab_v2_1_continuous.yaml`
- Create: `configs/ai_tuning/runtime.v2_1.continuous.live.yaml`
- Create: `configs/ai_tuning/v2_1_continuous.live.yaml`
- Test: `tests_py/ai_tuning/hardware/test_live_runtime_factory.py`

- [x] Write a failing configuration test requiring exactly the nine approved
  ranges and maximum steps.
- [x] Add the reviewed profile and isolated runtime/Testbench YAML files.
- [x] Load all three files through production parsers without constructing
  adapters; assert DCA/I2C resource identities and `max_rounds: 3`.
- [x] Run focused configuration tests and commit.

### Task 5: Non-hardware verification and authorization handoff

**Files:**
- Modify: `docs/v2-1-continuous-eye-testbench.md`

- [x] Run continuous, Campaign, hardware-factory, and import-isolation tests.
- [x] Run the complete pytest suite, compile check, `git diff --check`, and
  safety scan for BERT/FIR/`setFir` and I2C writes in the preflight path.
- [x] Record exact branch, worktree, HEAD, test results, and the statement that
  DCA, I2C, Kimi, BERT, serial, VISA, and device endpoints were not contacted.
- [x] Stop and give the operator the PowerShell API-Key input and explicit LIVE
  preflight command. Do not run it on the operator's behalf.
