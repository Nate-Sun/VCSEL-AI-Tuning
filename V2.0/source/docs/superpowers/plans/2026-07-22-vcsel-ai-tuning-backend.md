# VCSEL AI Tuning Backend and Hardware Loop Implementation Plan / VCSEL AI 调优后端与硬件闭环实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the campaign state machine, stage-scoped authorization, B0 restore/readback, deterministic hardware execution, persistence, FastAPI, and SSE needed by hierarchical VCSEL AI tuning. / 构建分层 VCSEL AI 调优所需的 Campaign 状态机、分阶段授权、B0 恢复与回读、确定性硬件执行、持久化、FastAPI 和 SSE。

**Architecture:** Reuse the existing FlexDCA, characterization, VD1124, VIT BERT, FIR, safety, and persistence code. The campaign orchestrator is the only component allowed to call hardware adapters; it exchanges pure candidates and observations with an injected optimizer protocol. / 复用现有 FlexDCA、标定、VD1124、VIT BERT、FIR、安全和持久化代码。Campaign Orchestrator 是唯一允许调用硬件适配器的组件，并通过注入的纯优化器协议交换候选点与观测。

**Tech Stack:** Python 3.12+, Pydantic 2, SQLite, FastAPI, Uvicorn, PyVISA, pyserial, NumPy, pytest. / Python 3.12+、Pydantic 2、SQLite、FastAPI、Uvicorn、PyVISA、pyserial、NumPy、pytest。

---

## Non-negotiable constraints / 不可违反的约束

- Read `docs/superpowers/specs/2026-07-22-vcsel-hierarchical-ai-tuning-design.md` and preserve its state names and B0 values exactly. / 完整读取设计文档并保持状态名和 B0 数值完全一致。
- Preserve unrelated dirty-worktree changes. / 保留工作区中的无关修改。
- Tests use fakes by default. Do not run live I²C or FIR writes during implementation. / 测试默认使用 fake；实施期间不得执行真实 I²C 或 FIR 写入。
- The frontend, API routes, and optimizer never call hardware adapters directly. / 前端、API 路由和优化器不得直接调用硬件适配器。
- `PER_STAGE` is enabled; `FULL_CAMPAIGN` is represented but rejected as disabled. / 启用 `PER_STAGE`；保留 `FULL_CAMPAIGN` 数据结构但拒绝执行。
- Pattern `SSPRQ` and NLC `6/6` remain fixed in V1. / 第一版固定 `SSPRQ` 和 NLC `6/6`。

## File map / 文件结构

- Create `src/vcsel_ai_tuning/campaign/__init__.py`: campaign package exports. / Campaign 包导出。
- Create `src/vcsel_ai_tuning/campaign/models.py`: B0, state, authorization, candidate, observation, and summary models. / 领域模型。
- Create `src/vcsel_ai_tuning/campaign/authorization.py`: immutable plan hashing and authorization validation. / 计划哈希与授权检查。
- Create `src/vcsel_ai_tuning/campaign/state_machine.py`: legal state transitions. / 合法状态转换。
- Create `src/vcsel_ai_tuning/campaign/store.py`: SQLite metadata and ordered events. / SQLite 元数据与有序事件。
- Create `src/vcsel_ai_tuning/campaign/orchestrator.py`: hardware-safe stage execution. / 硬件安全阶段执行。
- Create `src/vcsel_ai_tuning/campaign/repository.py`: atomic JSON/artifact persistence. / 原子 JSON 与制品存储。
- Create `src/vcsel_ai_tuning/api/app.py`, `schemas.py`, `routes.py`, `events.py`: FastAPI and SSE boundary. / FastAPI 与 SSE 边界。
- Create `configs/ai_tuning/b0.yaml`: frozen B0 source configuration. / B0 源配置。
- Modify `pyproject.toml`: API dependencies and `vcsel-ai-api` entry point. / API 依赖和入口。
- Create matching tests under `tests_py/ai_tuning/campaign/` and `tests_py/ai_tuning/api/`. / 对应测试。

### Task 1: Freeze B0 and campaign domain models / 冻结 B0 与 Campaign 领域模型

**Files:**
- Create: `configs/ai_tuning/b0.yaml`
- Create: `src/vcsel_ai_tuning/campaign/__init__.py`
- Create: `src/vcsel_ai_tuning/campaign/models.py`
- Test: `tests_py/ai_tuning/campaign/test_models.py`

- [ ] **Step 1: Write failing tests for the exact B0 and model invariants / 为精确 B0 与模型不变量编写失败测试**

```python
def test_b0_uses_confirmed_fir_i2c_pattern_and_nlc(load_b0):
    b0 = load_b0()
    assert b0.pattern == "SSPRQ"
    assert b0.fir.taps == (0, 9, -28, 104, -8, 1, -3)
    assert b0.fir.upper_nlc == b0.fir.lower_nlc == 6
    assert b0.i2c.model_dump() == {
        "ibias_dac": 70, "vref": 15, "vbias_ref": 15,
        "vga_cur": 13, "res_gain": 6, "res_bw": 0,
        "gain_ctrl": 5, "efin_cur": 15, "eq": 22,
    }
    assert sum(abs(v) for v in b0.fir.taps) == 153
```

- [ ] **Step 2: Run the focused test and confirm RED / 运行目标测试并确认失败**

Run: `python -m pytest tests_py/ai_tuning/campaign/test_models.py -q`
Expected: FAIL because the campaign package and B0 loader do not exist. / 预期：Campaign 包与 B0 loader 不存在。

- [ ] **Step 3: Implement strict Pydantic models and the exact YAML / 实现严格 Pydantic 模型和精确 YAML**

Define `AuthorizationMode`, `CampaignState`, `FirSettings`, `I2cSettings`, `BaselineConfig`, `ParameterSpec`, `Candidate`, `AppliedSettings`, `Observation`, `StageAuthorization`, `StageSummary`, and `Campaign`. Use frozen models, forbid extra fields, reject non-finite metrics, keep missing TDECQ as `None`, require seven integer FIR taps, positive Main, and L1 at most 170. / 使用冻结模型、禁止额外字段、拒绝非有限指标、以 `None` 表示缺失 TDECQ，并检查七个整数 tap、Main 为正和 L1 不超过 170。

- [ ] **Step 4: Re-run the focused tests and confirm GREEN / 重新运行目标测试并确认通过**

Run: `python -m pytest tests_py/ai_tuning/campaign/test_models.py -q`
Expected: PASS. / 预期：通过。

- [ ] **Step 5: Commit only Task 1 files / 仅提交 Task 1 文件**

```text
git add configs/ai_tuning/b0.yaml src/vcsel_ai_tuning/campaign tests_py/ai_tuning/campaign/test_models.py
git commit -m "feat: define AI tuning campaign domain"
```

### Task 2: Implement immutable stage authorization / 实现不可变阶段授权

**Files:**
- Create: `src/vcsel_ai_tuning/campaign/authorization.py`
- Test: `tests_py/ai_tuning/campaign/test_authorization.py`

- [ ] **Step 1: Write failing tests for canonical hashes, expiry, stale B0, scope, range, step, and budget rejection / 编写哈希、过期、陈旧 B0、范围、步长和预算拒绝测试**

```python
def test_stage_authorization_rejects_candidate_outside_envelope(auth, candidate):
    changed = candidate.model_copy(update={"requested_i2c": {"ibias_dac": 80}})
    with pytest.raises(AuthorizationError, match="ibias_dac.*range"):
        validate_candidate(auth, changed, previous_readback={"ibias_dac": 70})

def test_full_campaign_is_reserved_but_disabled(full_campaign_auth):
    with pytest.raises(AuthorizationError, match="FULL_CAMPAIGN.*disabled"):
        validate_authorization(full_campaign_auth)
```

- [ ] **Step 2: Run tests and confirm RED / 运行并确认失败**

Run: `python -m pytest tests_py/ai_tuning/campaign/test_authorization.py -q`
Expected: FAIL because authorization functions are missing. / 预期：授权函数缺失。

- [ ] **Step 3: Implement canonical JSON SHA-256 authorization IDs and deterministic envelope checks / 实现 canonical JSON SHA-256 ID 与确定性授权包络检查**

The signed payload contains campaign ID, stage, baseline hash, rollback hash, parameter specs, max candidates, max duration, issued time, and expiry. Validation consumes actual previous readback for max-step checks. / 哈希载荷包含 Campaign ID、阶段、基线哈希、回滚哈希、参数定义、最大候选数、最大时长、签发时间和过期时间；最大步长使用实际回读值检查。

- [ ] **Step 4: Re-run tests and confirm GREEN / 重新运行并确认通过**

Run: `python -m pytest tests_py/ai_tuning/campaign/test_authorization.py -q`
Expected: PASS. / 预期：通过。

- [ ] **Step 5: Commit / 提交**

```text
git add src/vcsel_ai_tuning/campaign/authorization.py tests_py/ai_tuning/campaign/test_authorization.py
git commit -m "feat: add stage-scoped tuning authorization"
```

### Task 3: Implement the persisted state machine / 实现持久化状态机

**Files:**
- Create: `src/vcsel_ai_tuning/campaign/state_machine.py`
- Create: `src/vcsel_ai_tuning/campaign/store.py`
- Test: `tests_py/ai_tuning/campaign/test_state_machine.py`
- Test: `tests_py/ai_tuning/campaign/test_store.py`

- [ ] **Step 1: Write failing transition and restart tests / 编写状态转换与重启测试**

```python
def test_fine_tuning_requires_fine_authorization(machine):
    machine.transition("I2C_COARSE", "AWAIT_FINE_APPROVAL")
    with pytest.raises(IllegalTransition):
        machine.transition("AWAIT_FINE_APPROVAL", "I2C_FINE")

def test_nonterminal_hardware_run_restarts_in_recovery_required(store):
    store.save_state("campaign-1", "FIR_APPLY")
    assert store.recover_state("campaign-1") == "RECOVERY_REQUIRED"
```

- [ ] **Step 2: Run tests and confirm RED / 运行并确认失败**

Run: `python -m pytest tests_py/ai_tuning/campaign/test_state_machine.py tests_py/ai_tuning/campaign/test_store.py -q`
Expected: FAIL because state and store modules are missing. / 预期：模块缺失。

- [ ] **Step 3: Implement explicit transition tables and SQLite transactions / 实现显式转换表与 SQLite 事务**

Persist campaign state and an increasing event sequence in one transaction. Require an approved, unexpired authorization before entering each mutating stage. Treat process restart in a nonterminal hardware state as `RECOVERY_REQUIRED`. / 在同一事务中保存 Campaign 状态与递增事件序号；进入每个写硬件阶段前要求有效授权；非终止硬件状态重启后进入 `RECOVERY_REQUIRED`。

- [ ] **Step 4: Re-run tests and confirm GREEN / 重新运行并确认通过**

Run the same command; expected PASS. / 使用相同命令，预期通过。

- [ ] **Step 5: Commit / 提交**

```text
git add src/vcsel_ai_tuning/campaign/state_machine.py src/vcsel_ai_tuning/campaign/store.py tests_py/ai_tuning/campaign/test_state_machine.py tests_py/ai_tuning/campaign/test_store.py
git commit -m "feat: persist tuning campaign state"
```

### Task 4: Add atomic campaign artifacts / 增加原子 Campaign 制品

**Files:**
- Create: `src/vcsel_ai_tuning/campaign/repository.py`
- Test: `tests_py/ai_tuning/campaign/test_repository.py`

- [ ] **Step 1: Write failing tests for temporary writes, hashes, atomic promotion, and quarantine / 编写临时写入、哈希、原子提升与隔离测试**
- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/campaign/test_repository.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement one run directory containing immutable config, plans, receipts, events, observations, raw responses, eye artifacts, optimizer snapshots, CSV, and bilingual report / 实现完整运行目录**

Use `allow_nan=False`, SHA-256 manifests, temporary paths, and `Path.replace`. An observation becomes queryable only after artifact promotion and the SQLite transaction both succeed. / 使用 `allow_nan=False`、SHA-256 manifest、临时路径和 `Path.replace`；只有制品提升与 SQLite 事务均成功后观测才可查询。

- [ ] **Step 4: Re-run tests and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 5: Commit with `git commit -m "feat: persist atomic tuning artifacts"` / 提交**

### Task 5: Build the hardware-safe orchestrator / 构建硬件安全 Orchestrator

**Files:**
- Create: `src/vcsel_ai_tuning/campaign/orchestrator.py`
- Modify: `src/vcsel_ai_tuning/hardware/characterization.py`
- Modify: `src/vcsel_ai_tuning/hardware/live_workflow.py`
- Test: `tests_py/ai_tuning/campaign/test_orchestrator.py`

- [ ] **Step 1: Write failing fake-hardware tests for preflight, B0 restore, baseline repeats, candidate apply/readback, measurement, pause, abort, and rollback / 编写 fake-hardware 全流程失败测试**

```python
def test_optimizer_never_receives_invalid_or_unpersisted_observation(orchestrator):
    orchestrator.hardware.next_tdecq_status = "INV"
    orchestrator.run_one_candidate()
    assert orchestrator.optimizer.observed == []
    assert orchestrator.store.last_event.type == "OBSERVATION_INVALID"

def test_fir_failure_restores_previous_fir_and_i2c(orchestrator):
    orchestrator.hardware.fail_fir_readback = True
    result = orchestrator.apply_fir_candidate()
    assert result.state == "ROLLING_BACK"
    assert orchestrator.hardware.readback() == orchestrator.stage_snapshot
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/campaign/test_orchestrator.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Define injected protocols for optimizer, DCA, I²C, BERT, clock, and repository / 定义可注入协议**
- [ ] **Step 4: Implement preflight and exact B0 restore with readback / 实现预检和精确 B0 恢复回读**
- [ ] **Step 5: Implement screening, coarse, fine, FIR, retune, and final-verification stage runners without embedding optimizer mathematics / 实现各阶段 runner，但不嵌入优化数学**
- [ ] **Step 6: Implement one pre-authorized rollback attempt and `RECOVERY_REQUIRED` on mismatch / 实现一次预授权回滚与失败状态**
- [ ] **Step 7: Re-run the focused tests and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 8: Commit with `git commit -m "feat: orchestrate hierarchical hardware tuning"` / 提交**

### Task 6: Expose FastAPI and ordered SSE / 暴露 FastAPI 与有序 SSE

**Files:**
- Create: `src/vcsel_ai_tuning/api/__init__.py`
- Create: `src/vcsel_ai_tuning/api/app.py`
- Create: `src/vcsel_ai_tuning/api/schemas.py`
- Create: `src/vcsel_ai_tuning/api/routes.py`
- Create: `src/vcsel_ai_tuning/api/events.py`
- Modify: `pyproject.toml`
- Test: `tests_py/ai_tuning/api/test_routes.py`
- Test: `tests_py/ai_tuning/api/test_events.py`

- [ ] **Step 1: Write failing API tests for create, plan, approve, pause, resume, abort, rollback, observations, best, report, and disabled full-campaign authorization / 编写 API 失败测试**
- [ ] **Step 2: Write failing SSE reconnection test using `Last-Event-ID` / 编写 SSE 重连失败测试**

```python
def test_sse_replays_after_last_event_id(client, campaign_with_events):
    response = client.get(
        f"/api/campaigns/{campaign_with_events}/events/stream",
        headers={"Last-Event-ID": "4"},
    )
    assert "id: 5" in response.text
    assert "id: 4" not in response.text
```

- [ ] **Step 3: Run `python -m pytest tests_py/ai_tuning/api -q` and confirm RED / 运行并确认失败**
- [ ] **Step 4: Add `fastapi`, `uvicorn`, and the `vcsel-ai-api` entry point / 增加依赖与入口**
- [ ] **Step 5: Implement thin routes that call an injected campaign service; never import hardware adapters in route modules / 实现薄路由并禁止导入硬件适配器**
- [ ] **Step 6: Implement ordered SSE with heartbeat, `Last-Event-ID`, and terminal-state close / 实现有序 SSE、心跳、重放与终止关闭**
- [ ] **Step 7: Re-run API tests and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 8: Commit with `git commit -m "feat: expose AI tuning campaign API"` / 提交**

### Task 7: End-to-end replay and verification / 端到端回放与验证

**Files:**
- Create: `tests_py/ai_tuning/campaign/test_campaign_replay.py`
- Create: `tests_py/ai_tuning/fixtures/hierarchical_campaign.jsonl`
- Modify: `README.md`

- [ ] **Step 1: Add a fixed replay covering invalid TDECQ, ER coarse success, constrained fine improvement, rejected FIR, accepted rollback, and final verification / 增加固定回放**
- [ ] **Step 2: Assert the complete state sequence and final `PARTIAL_CONVERGENCE` or `SUCCESS` from fixture data / 断言完整状态序列与最终状态**
- [ ] **Step 3: Run backend-focused tests / 运行后端目标测试**

Run: `python -m pytest tests_py/ai_tuning/campaign tests_py/ai_tuning/api tests_py/ai_tuning/hardware -q`
Expected: all collected tests pass; no live hardware access. / 预期：全部通过且不访问真实硬件。

- [ ] **Step 4: Run the complete Python suite / 运行完整 Python 测试**

Run: `python -m pytest tests_py/ai_tuning -q`
Expected: all collected tests pass. / 预期：全部通过。

- [ ] **Step 5: Run `git diff --check` and inspect that no secret, IP override, or generated artifact is staged / 检查差异与敏感内容**
- [ ] **Step 6: Update the bilingual README with API startup and fake/replay commands only / 更新中英双语 README，仅写 API 与 fake/replay 命令**
- [ ] **Step 7: Commit with `git commit -m "test: verify hierarchical tuning backend"` / 提交**

## Delivery report / 交付报告

Report changed files, exact test counts, API start command, state diagram, example stage authorization, example SSE events, replay result, and all remaining live-hardware prerequisites. Do not run live writes, push, or create a pull request unless explicitly requested. / 报告文件、测试数量、API 启动命令、状态图、授权示例、SSE 示例、回放结果和真实硬件前置条件；未经明确要求不得执行真实写入、push 或创建 PR。
