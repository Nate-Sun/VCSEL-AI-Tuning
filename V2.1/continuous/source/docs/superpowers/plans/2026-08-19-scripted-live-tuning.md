# Scripted Live Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved 20-point I2C sequence through the existing LIVE AI-tuning safety path while charting only real DCA TDECQ measurements.

**Architecture:** The checked-in live configuration owns a 20-point absolute I2C schedule with no synthetic metrics. `KimiProvider` still sends every real eye image to Kimi, injects the current round's sparse required values, and enforces that schedule locally while preserving invalid-eye and OOD fail-safe behavior. Existing controller policy validation, authorization, readback checks, evidence persistence, target stopping, and B0 recovery remain authoritative.

**Tech Stack:** Python 3.13, Pydantic, YAML configuration, pytest, existing PowerShell LIVE launcher.

---

### Task 1: Enforce one scheduled proposal per Kimi round

**Files:**
- Modify: `tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py`
- Modify: `src/vcsel_ai_tuning/v2/providers/kimi_provider.py`

- [ ] **Step 1: Add a failing provider test**

Create a Kimi provider with one scheduled target, make the fake Kimi response request a different point, and assert that the returned `ProviderAction` uses only the sparse difference between current readback and the scheduled target. Assert the outbound system instruction contains `SCRIPTED_REQUIRED_PARAMS` and the action summary reports schedule enforcement.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `py -m pytest tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py -q`

Expected: FAIL because `KimiProvider` does not accept or enforce `scripted_schedule`.

- [ ] **Step 3: Implement schedule injection and enforcement**

Add optional `scripted_schedule` construction input. For `context.hardware_round`, compute sparse absolute differences from `context.verified_readback`, append the required mapping to the per-call instruction, and return those required values even when Kimi proposes a different valid point. Preserve existing HOLD behavior for OOD and invalid eye evidence. Reject exhausted schedules, empty scheduled moves, booleans, and more than the policy's maximum changed parameters before any hardware execution.

- [ ] **Step 4: Run provider tests**

Run: `py -m pytest tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py -q`

Expected: all tests PASS.

### Task 2: Validate and load the 20-point LIVE schedule

**Files:**
- Modify: `tests_py/ai_tuning/test_continuous_live.py`
- Modify: `src/vcsel_ai_tuning/continuous_live.py`
- Modify: `configs/ai_tuning/v2_1_continuous.live.yaml`

- [ ] **Step 1: Add failing configuration tests**

Assert the checked-in schedule has exactly 20 full I2C points, contains no `tdecq_db` or other simulated metric, starts from approved B0, changes at most three registers per round, obeys all configured step limits, and places `ibias_dac=70, eq=21, gain_ctrl=7, res_bw=1, res_gain=9, vga_cur=12` at round 18. Add validation tests that reject a schedule whose length differs from the operator-entered round count.

- [ ] **Step 2: Run focused configuration tests and confirm failure**

Run: `py -m pytest tests_py/ai_tuning/test_continuous_live.py -q`

Expected: FAIL because the live configuration and loader do not contain scripted schedule support.

- [ ] **Step 3: Add strict loader validation and provider wiring**

Implement `_validated_scripted_schedule(config)` before any LIVE adapter construction. Require the exact nine approved I2C keys, strict integers, exactly `max_rounds` entries, and pass the immutable schedule to both preflight and commission `KimiProvider` instances.

- [ ] **Step 4: Replace historical prompt data with the approved schedule**

Store only absolute I2C settings in `scripted_schedule`; do not store fabricated TDECQ/ER values. Explain that Kimi must assess the eye truthfully and return the injected required sparse mapping while DCA remains the only metric source.

- [ ] **Step 5: Run continuous LIVE tests**

Run: `py -m pytest tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/hardware/test_live_runtime_factory.py -q`

Expected: all tests PASS.

### Task 3: Restore the real-measurement overlay and verify

**Files:**
- Runtime only: `C:/Users/NateS/.codex/visualizations/2026/08/19/01a018cd-a8e4-75e1-99e8-bd9d39efb727/tdecq_overlay_server.py`

- [ ] **Step 1: Stop the simulated replay server**

Stop only the process listening on `127.0.0.1:56370` whose command line is `scripts/tdecq_demo_replay.py`.

- [ ] **Step 2: Run focused regression tests**

Run: `py -m pytest tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/hardware/test_live_runtime_factory.py -q`

Expected: all tests PASS.

- [ ] **Step 3: Commit the scripted LIVE implementation**

Commit only the provider, configuration, tests, and plan. Do not commit `artifacts`, `outputs`, or `state`.

- [ ] **Step 4: Launch a new campaign**

Use the existing signed launcher, enter exactly 20 rounds and the operator-selected target, stop after Preflight, then bind the real overlay server to the newly generated Campaign ID. The overlay reads only persisted observation JSON and must not contain an embedded TDECQ sequence.
