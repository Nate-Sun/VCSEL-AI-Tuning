# VCSEL Hierarchical AI Optimizer Implementation Plan / VCSEL 分层 AI 优化器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic sensitivity screening, coordinate-descent baseline, ER coarse optimization with TDECQ-validity modeling, noisy constrained Bayesian fine tuning, bounded FIR outer proposals, convergence, and replay evaluation without any hardware access. / 在完全不访问硬件的前提下，实现确定性敏感度筛选、坐标下降基线、带 TDECQ 有效性模型的 ER 粗调、含噪受约束贝叶斯精调、受限 FIR 外层候选、收敛和回放评估。

**Architecture:** The optimizer is a pure package behind an `initialize/suggest/observe` protocol. It consumes actual readback settings and immutable observations, persists serializable model snapshots, and emits candidates that still require backend authorization and safety validation. / 优化器是实现 `initialize/suggest/observe` 协议的纯包，使用实际回读值和不可变观测，保存可序列化模型快照，并输出仍需后端授权与安全检查的候选点。

**Tech Stack:** Python 3.12+, NumPy, SciPy, scikit-learn, Pydantic 2, pytest. / Python 3.12+、NumPy、SciPy、scikit-learn、Pydantic 2、pytest。

---

## Non-negotiable constraints / 不可违反的约束

- Read `docs/superpowers/specs/2026-07-22-vcsel-hierarchical-ai-tuning-design.md` completely. / 完整读取设计文档。
- No import from `hardware`, PyVISA, serial, socket, FastAPI, or frontend code. / 不得导入硬件、PyVISA、serial、socket、FastAPI 或前端代码。
- Missing TDECQ remains `None` plus validity status; never impute zero, a penalty, or the last value. / 缺失 TDECQ 保持 `None` 与状态，禁止填充为 0、惩罚值或上一值。
- Train on readback settings, not requested settings. / 使用回读参数训练。
- Use seeded deterministic replay. / 使用带 seed 的确定性回放。
- Coordinate descent is the required benchmark and fallback; direct joint 9-I²C-plus-7-FIR optimization is excluded. / 坐标下降是强制基线和降级方案；排除 9 个 I²C 与 7 个 FIR 的直接联合优化。

## File map / 文件结构

- Create `src/vcsel_ai_tuning/optimization/protocol.py`: optimizer protocol and result types. / 优化器协议与结果类型。
- Create `parameter_space.py`: integer normalization, quantization, and candidate enumeration. / 整数空间处理。
- Create `sensitivity.py`: one-parameter screening and ranking. / 单参数筛选与排名。
- Create `coordinate.py`: deterministic local baseline and fallback. / 确定性局部基线。
- Create `coarse.py`: ER regression plus validity probability. / ER 粗调。
- Create `bayesian.py`: constrained expected improvement for fine tuning. / 受约束期望改善。
- Create `convergence.py`: noise-aware stage decisions. / 噪声感知收敛。
- Create `fir_outer.py`: DCA-guided direct full-target FIR proposal. / DCA 引导的完整目标 FIR 提案。
- Create `replay.py`: offline campaign comparison and metrics. / 离线回放比较。
- Add tests under `tests_py/ai_tuning/optimization/`. / 增加对应测试。

### Task 1: Define the pure optimizer protocol and parameter space / 定义纯优化器协议与参数空间

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/__init__.py`
- Create: `src/vcsel_ai_tuning/optimization/protocol.py`
- Create: `src/vcsel_ai_tuning/optimization/parameter_space.py`
- Test: `tests_py/ai_tuning/optimization/test_protocol.py`
- Test: `tests_py/ai_tuning/optimization/test_parameter_space.py`

- [ ] **Step 1: Write failing tests for integer quantization, enabled parameters, max-step neighborhoods, and protocol results / 编写整数化、启用参数、最大步长邻域和协议结果测试**

```python
def test_parameter_space_emits_only_enabled_integer_readback_neighbors(space):
    points = list(space.local_neighbors({"ibias_dac": 70, "eq": 22}))
    assert {tuple(sorted(p.items())) for p in points} == {
        (("eq", 22), ("ibias_dac", 68)),
        (("eq", 22), ("ibias_dac", 72)),
        (("eq", 21), ("ibias_dac", 70)),
        (("eq", 23), ("ibias_dac", 70)),
    }
```

- [ ] **Step 2: Run tests and confirm RED / 运行并确认失败**

Run: `python -m pytest tests_py/ai_tuning/optimization/test_protocol.py tests_py/ai_tuning/optimization/test_parameter_space.py -q`
Expected: FAIL because the package does not exist. / 预期：包不存在。

- [ ] **Step 3: Implement `OptimizerProtocol`, `Suggestion`, `ConvergenceDecision`, and `IntegerParameterSpace` / 实现协议与整数参数空间**

`Suggestion` variants are `CandidateSuggestion`, `StageConverged`, `NoSafeCandidate`, and `FallbackRequested`. Candidate generation receives but never expands configured bounds. / Suggestion 类型为候选、阶段收敛、无安全候选和请求降级；候选生成不得扩大配置边界。

- [ ] **Step 4: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 5: Commit with `git commit -m "feat: define pure tuning optimizer protocol"` / 提交**

### Task 2: Implement deterministic sensitivity screening / 实现确定性敏感度筛选

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/sensitivity.py`
- Test: `tests_py/ai_tuning/optimization/test_sensitivity.py`

- [ ] **Step 1: Write failing tests for baseline-relative effects, noise floors, one-at-a-time enforcement, invalid-rate penalties, and stable ranking / 编写敏感度测试**

```python
def test_sensitivity_selects_only_effects_above_noise(screening_history):
    result = rank_parameters(
        screening_history,
        er_noise_db=0.03,
        tdecq_noise_db=0.04,
        max_selected=5,
    )
    assert result.selected == ("eq", "ibias_dac", "vga_cur")
    assert "res_gain" in result.frozen
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/optimization/test_sensitivity.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement normalized effects using repeated medians and `max(configured_floor, 3*sigma)` / 使用重复中位数和噪声阈值实现归一化影响**
- [ ] **Step 4: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 5: Commit with `git commit -m "feat: rank I2C tuning sensitivity"` / 提交**

### Task 3: Implement coordinate-descent benchmark and fallback / 实现坐标下降基线与降级

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/coordinate.py`
- Test: `tests_py/ai_tuning/optimization/test_coordinate.py`

- [ ] **Step 1: Write failing tests for seeded ordering, one-coordinate changes, tie-breaking, boundary behavior, patience, and replay determinism / 编写坐标搜索测试**

```python
def test_coordinate_candidate_changes_exactly_one_parameter(optimizer, history):
    suggestion = optimizer.suggest(history, stage="I2C_COARSE", budget_remaining=8)
    changed = [k for k, v in suggestion.requested_i2c.items() if v != history[-1].settings_readback[k]]
    assert len(changed) == 1
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/optimization/test_coordinate.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement deterministic cyclic coordinate descent using lexicographic ranking: validity, ER threshold class, TDECQ, ER, and distance from baseline / 实现确定性循环坐标下降与字典序排名**
- [ ] **Step 4: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 5: Commit with `git commit -m "feat: add coordinate tuning baseline"` / 提交**

### Task 4: Implement ER coarse tuning with validity modeling / 实现 ER 粗调与有效性建模

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/coarse.py`
- Modify: `pyproject.toml`
- Test: `tests_py/ai_tuning/optimization/test_coarse.py`

- [ ] **Step 1: Add `scikit-learn` to a dedicated optimizer optional dependency group and write failing model tests / 增加优化器可选依赖并编写失败测试**

```python
def test_coarse_acquisition_prefers_er_gain_and_validity_probability(model, pool):
    suggestion = model.suggest(pool)
    assert suggestion.predicted_er_db >= 2.5
    assert suggestion.predicted_tdecq_valid_probability >= 0.5

def test_coarse_falls_back_when_validity_has_one_class(model_with_all_invalid):
    assert model_with_all_invalid.suggest().kind == "FALLBACK_REQUESTED"
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/optimization/test_coarse.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Fit ER with `GaussianProcessRegressor`; fit validity only when both classes exist, otherwise request coordinate fallback / 使用 GPR 拟合 ER，仅在两类均存在时拟合有效性，否则请求降级**
- [ ] **Step 4: Score a seeded finite integer pool by ER expected improvement times TDECQ-validity probability; return prediction metadata / 使用 ER 期望改善乘有效概率评分候选池**
- [ ] **Step 5: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 6: Commit with `git commit -m "feat: model ER coarse tuning"` / 提交**

### Task 5: Implement noisy constrained Bayesian fine tuning / 实现含噪受约束贝叶斯精调

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/bayesian.py`
- Test: `tests_py/ai_tuning/optimization/test_bayesian.py`

- [ ] **Step 1: Write failing tests for TDECQ-only-valid fitting, ER feasibility, validity probability, duplicate avoidance, uncertainty, and seeded determinism / 编写贝叶斯优化测试**

```python
def test_constrained_ei_never_uses_missing_tdecq_as_numeric(history, optimizer):
    optimizer.fit(history)
    assert optimizer.tdecq_training_ids == tuple(
        o.observation_id for o in history if o.tdecq_db is not None and o.tdecq_status == "CORR"
    )

def test_fine_candidate_maximizes_ei_times_feasibility(optimizer, pool):
    chosen = optimizer.suggest_from_pool(pool)
    assert chosen.acquisition == pytest.approx(
        chosen.expected_improvement
        * chosen.er_feasibility_probability
        * chosen.tdecq_valid_probability
    )
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/optimization/test_bayesian.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement normalized GPR models for TDECQ and ER plus a guarded validity classifier / 实现标准化 TDECQ、ER GPR 与受保护的有效性分类器**
- [ ] **Step 4: Implement minimization expected improvement multiplied by `P(ER>=3.0)` and `P(valid TDECQ)` / 实现最小化 EI 乘两个可行概率**
- [ ] **Step 5: Exclude measured, pending, unauthorized, and max-step-invalid points before scoring / 评分前排除已测、待测、未授权和步长无效点**
- [ ] **Step 6: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 7: Commit with `git commit -m "feat: add constrained Bayesian tuner"` / 提交**

### Task 6: Implement noise-aware convergence / 实现噪声感知收敛

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/convergence.py`
- Test: `tests_py/ai_tuning/optimization/test_convergence.py`

- [ ] **Step 1: Write failing tests for coarse promotion, fine success, partial convergence, patience, budget, and repeated-best verification / 编写收敛测试**

```python
def test_partial_when_budget_ends_below_fine_er_target(best_observation):
    decision = decide_fine_convergence(
        history=[best_observation.model_copy(update={"er_db": 2.7, "tdecq_db": 2.2})],
        budget_remaining=0,
        er_coarse_db=2.5,
        er_fine_db=3.0,
    )
    assert decision.status == "PARTIAL_CONVERGENCE"
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/optimization/test_convergence.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement `meaningful_change=max(floor,3*sigma)`, default patience 5, three-repeat candidate confirmation, and five-repeat final verification / 实现噪声阈值与重复验证**
- [ ] **Step 4: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 5: Commit with `git commit -m "feat: add noise-aware tuning convergence"` / 提交**

### Task 7: Implement the bounded FIR outer loop / 实现受限 FIR 外层循环

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/fir_outer.py`
- Test: `tests_py/ai_tuning/optimization/test_fir_outer.py`

- [ ] **Step 1: Write failing tests for B0, exact full-target preservation, seven taps, positive Main, L1, fixed NLC, authorization change limits, and rollback recommendation / 编写完整目标 FIR 候选测试**

```python
def test_direct_candidate_preserves_formula_target_nlc_and_l1(generator):
    target = (0, 6, -26, 104, -7, -8, -2)
    candidate = generator.generate(
        current=(0, 9, -28, 104, -8, 1, -3),
        target=target,
    )
    assert candidate.taps == target
    assert sum(abs(value) for value in candidate.taps) <= 170
    assert candidate.upper_nlc == candidate.lower_nlc == 6
```

- [ ] **Step 2: Run `python -m pytest tests_py/ai_tuning/optimization/test_fir_outer.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Reuse pure calculations from `hardware/fir.py` without importing network clients; return the exact validated formula target and acceptance decision / 复用纯 FIR 计算，返回经过验证的完整公式目标与接受判断**
- [ ] **Step 4: Evaluate one full-target FIR candidate per stage and require repeated TDECQ improvement without ER-class degradation / 每个阶段只评估一个完整目标，并要求重复测量确认改善**
- [ ] **Step 5: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 6: Commit with `git commit -m "feat: propose bounded FIR outer updates"` / 提交**

### Task 8: Build replay evaluation and compare against coordinate descent / 构建回放评估并对比坐标下降

**Files:**
- Create: `src/vcsel_ai_tuning/optimization/replay.py`
- Create: `tests_py/ai_tuning/fixtures/hierarchical_optimizer.jsonl`
- Test: `tests_py/ai_tuning/optimization/test_replay.py`
- Modify: `README.md`

- [ ] **Step 1: Create a fixed fixture containing B0 noise, invalid TDECQ, coarse promotion, fine candidates, and FIR observations / 创建固定回放数据**
- [ ] **Step 2: Write failing tests that compare identical-budget coordinate and Bayesian campaigns / 编写同预算对比测试**

```python
def test_replay_is_seeded_and_reports_sample_efficiency(fixture):
    first = replay(fixture, optimizer="bayesian", seed=1124, budget=30)
    second = replay(fixture, optimizer="bayesian", seed=1124, budget=30)
    assert first == second
    assert first.best_tdecq_db <= first.baseline_tdecq_db
    assert first.measurement_count <= 30
```

- [ ] **Step 3: Run `python -m pytest tests_py/ai_tuning/optimization/test_replay.py -q` and confirm RED / 运行并确认失败**
- [ ] **Step 4: Implement replay metrics: best feasible TDECQ, ER class, measurements to threshold, invalid rate, regret proxy, fallback count, and convergence reason / 实现回放指标**
- [ ] **Step 5: Run all optimizer tests / 运行全部优化器测试**

Run: `python -m pytest tests_py/ai_tuning/optimization -q`
Expected: all collected tests pass without hardware or network access. / 预期：全部通过且不访问硬件或网络。

- [ ] **Step 6: Run complete AI tuning tests and `git diff --check` / 运行完整测试与差异检查**

Run: `python -m pytest tests_py/ai_tuning -q`
Expected: all collected tests pass. / 预期：全部通过。

- [ ] **Step 7: Document seeded replay and model limitations in the bilingual README / 在双语 README 记录回放与模型限制**
- [ ] **Step 8: Commit with `git commit -m "test: compare hierarchical tuning optimizers"` / 提交**

## Delivery report / 交付报告

Report changed files, dependency changes, exact test counts, replay commands, coordinate-versus-Bayesian results under identical budgets, model snapshots, fallback behavior, and limitations. Do not claim hardware improvement from replay results and do not access live hardware. / 报告文件、依赖、测试数量、回放命令、同预算算法对比、模型快照、降级行为和限制；不得用回放结果声称真实硬件改善，也不得访问真实硬件。
