# Kimi Truncation Response Handling Implementation Plan

> **For Codex:** Execute this plan with the `executing-plans` skill and verify every step without contacting LIVE hardware or the network.

**Goal:** Prevent truncated or malformed Kimi responses from reaching hardware, retain raw attempt evidence before parsing, and preserve the existing FAILED_SAFE/B0 recovery path.

**Architecture:** Keep all response classification, retry sizing, and attempt journaling inside `KimiApiClient.analyze_continuous`. Pass a campaign-scoped journal path from the existing LIVE composition layer. The controller and hardware adapters remain unchanged, so exhausted provider failures continue through the established recovery path.

**Tech Stack:** Python 3.13, Pydantic, OpenAI-compatible SDK, pytest, append-only JSONL.

---

### Task 1: Lock the response contract with failing tests

**Files:**
- Modify: `tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py`
- Modify: `tests_py/ai_tuning/v2/continuous/test_controller.py` only if the existing generic provider-failure coverage cannot express the concrete Kimi exhaustion path

1. Add a valid-response test asserting a 4096-token request and two ordered journal events: raw response before parse result.
2. Add a truncation test asserting `finish_reason=length` retries with 8192 tokens and returns only the second valid response.
3. Add an exhausted-retry test asserting both attempts are logged, no partial response is accepted, and the controller remains FAILED_SAFE with B0 recovery and no candidate execution.
4. Run the focused tests and record the expected failures before production changes.

### Task 2: Implement minimal continuous-response handling

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/providers/kimi_client.py`

1. Define continuous-response constants: 4096 default tokens, 8192 truncation retry tokens, two maximum attempts.
2. Add an optional response journal path to `KimiApiClient` and an append-plus-fsync helper.
3. Before parsing each completion, append attempt number, token limit, finish reason, usage when available, and raw content.
4. Parse strictly with the existing Pydantic schema; append parse success/failure afterward.
5. Retry truncation with 8192 tokens; retry ordinary invalid JSON with 4096; never repair or partially consume invalid JSON.
6. Add the concise JSON-only instructions to the existing system prompt without changing policy, ranges, objectives, or hardware semantics.

### Task 3: Wire the campaign-scoped evidence path

**Files:**
- Modify: `src/vcsel_ai_tuning/continuous_live.py`
- Modify: `tests_py/ai_tuning/test_continuous_live.py` if needed for composition coverage

1. Pass `<result_dir>/<campaign_id>/kimi-response-attempts.jsonl` to both PRE-FLIGHT and COMMISSION Kimi clients.
2. Keep `kimi-memory.jsonl` unchanged and separate from raw response evidence.
3. Verify no DCA, I2C, BERT, FIR, policy, objective, or controller logic changed.

### Task 4: Verify and deliver

1. Run the focused Kimi tests.
2. Run the controller and LIVE composition regression tests.
3. Run the full offline Python test suite.
4. Run `git diff --check`, inspect the exact diff, and confirm only intended files changed plus this plan.
5. Commit the patch on the existing B branch and report constants, maximum attempts, journal path, tests, and confirmation that no LIVE/hardware test ran.
