# VCSEL Dual-Path TDECQ Tuning Design / VCSEL 双路径 TDECQ 调优设计

**Date / 日期:** 2026-07-22
**Status / 状态:** Approved design revision / 已批准的设计修订
**Scope / 范围:** Driver onboarding, exhaustive characterization, TDECQ-only Bayesian tuning, and reusable parameter-effect memory / Driver 接入、穷举表征、仅以 TDECQ 为目标的贝叶斯调优及可复用参数影响记忆

## 1. Decision Summary / 决策摘要

The tuning platform supports known drivers through a versioned `DriverProfile` and a standard adapter. Every new physical VCSEL receives an independent campaign and current-device model.

调优平台通过版本化 `DriverProfile` 和标准适配器支持已知 driver。每个新的 VCSEL 实体件都创建独立 Campaign 和当前器件模型。

The optimizer exposes two operating paths over the same finite, authorized, deterministic safe candidate space:

优化器在同一个有限、已授权且经过确定性安全检查的候选空间上提供两种运行路径：

1. `PATH_EXHAUSTIVE`: measure every unique safe parameter combination and return the lowest valid measured TDECQ on that enumerated grid.
2. `PATH_TDECQ_BAYESIAN`: use a finite-budget Bayesian optimizer whose only performance objective is lower TDECQ.

1. `PATH_EXHAUSTIVE`：测量每一个唯一的安全参数组合，返回枚举网格上有效实测 TDECQ 最低的组合。
2. `PATH_TDECQ_BAYESIAN`：使用有限预算的贝叶斯优化器，唯一性能目标是降低 TDECQ。

ER remains measured, persisted, and reported, but it does not constrain, score, rank, stop, or select candidates in either path. Hardware safety and TDECQ measurement validity remain separate from performance preference.

ER 仍然被测量、保存和报告，但在两条路径中均不用于约束、评分、排名、停止或选择候选。硬件安全和 TDECQ 测量有效性与性能偏好保持分离。

This document supersedes the ER feasibility constraint and the combined TDECQ/ER preference policy in `2026-07-22-vcsel-hierarchical-ai-tuning-design.md`. All unchanged authorization, readback, persistence, rollback, and hardware-isolation rules in that design remain in force.

本文取代 `2026-07-22-vcsel-hierarchical-ai-tuning-design.md` 中的 ER 可行性约束及 TDECQ/ER 综合偏好策略。原设计中未被修改的授权、回读、持久化、回滚和硬件隔离规则继续有效。

## 2. Supported-Driver Boundary / 已支持 Driver 的边界

The platform automates tuning only after a driver has a human-reviewed profile and an adapter. It does not discover unknown register maps or infer unknown electrical safety limits.

只有在 driver 具备人工审核的 Profile 和适配器之后，平台才允许自动调优。系统不自动发现未知 Reg map，也不推断未知电气安全范围。

### 2.1 DriverProfile / Driver 配置

Each versioned profile defines:

- driver identity and compatible revisions;
- parameter names, register mappings, encoding types, and quantization;
- human-approved safe ranges and enabled states;
- maximum per-write steps and cross-parameter constraints;
- baseline `B0`;
- write ordering and settling requirements;
- profile version and immutable content hash.

每个版本化 Profile 定义：

- driver 身份和兼容版本；
- 参数名称、寄存器映射、编码类型和量化规则；
- 人工批准的安全范围和启用状态；
- 单次最大写入步长和跨参数约束；
- 基线 `B0`；
- 写入顺序和稳定等待要求；
- Profile 版本和不可变内容哈希。

The three ranges remain distinct:

```text
AI authorization range
  subset of human-approved experimental safe range
    subset of register encoding range
```

### 2.2 DriverAdapter / Driver 适配器

Every adapter implements the same orchestration-facing operations:

```text
identify
health_check
snapshot
apply
readback
restore
```

The mathematical optimizer cannot import or call the adapter. It receives readback settings and immutable observations, then returns parameter candidates only.

数学优化器不得导入或调用适配器。它只接收实际回读参数和不可变观测，并只返回参数候选。

## 3. Minimal Driver Qualification / 新 Driver 最简验证

A new profile version completes one supervised smoke test before stage-level automatic tuning is enabled:

新 Profile 版本在开放分阶段自动调优之前，执行一次人工监督的冒烟测试：

1. validate schema, bounds, baseline, constraints, and unique register mappings;
2. identify the driver and snapshot current readback state;
3. apply one minimum-step change to one enabled parameter;
4. confirm actual readback equals the expected encoded value;
5. restore `B0` and confirm the restored readback;
6. acquire one DCA result to prove the control-to-measurement loop works.

1. 校验格式、范围、基线、组合约束及寄存器映射唯一性；
2. 识别 driver 并保存当前回读状态；
3. 对一个已启用参数执行一次最小步长修改；
4. 确认实际回读等于预期编码值；
5. 恢复 `B0` 并确认恢复后的回读；
6. 采集一次 DCA 结果，证明控制到测量的闭环可工作。

Failure leaves the profile unqualified and restores the entry snapshot. This smoke test verifies control, readback, measurement, and recovery plumbing; the human-provided safe range remains the engineering safety authority.

验证失败时，该 Profile 保持未认证状态并恢复入口快照。该冒烟测试验证控制、回读、测量和恢复链路；人工提供的安全范围仍是工程安全依据。

## 4. Shared Safe Candidate Space / 共享安全候选空间

Both paths consume the same `IntegerParameterSpace`. It is the Cartesian product of enabled, quantized parameter values after deterministic filtering by authorization ranges, encoding rules, and cross-parameter constraints.

两条路径使用同一个 `IntegerParameterSpace`。它由启用参数的量化值做笛卡尔积，再经过授权范围、编码规则和跨参数约束的确定性过滤得到。

Maximum step is a transition constraint, not an endpoint-membership constraint. A distant endpoint remains part of the safe candidate grid when it is inside all bounds and constraints, but the orchestrator must reach it through a sequence in which every write respects the configured maximum step.

最大步长属于过渡约束，而不是终点成员约束。只要远处终点满足所有范围和组合约束，它仍属于安全候选网格；但编排器必须通过一系列写入到达该点，并确保每次写入都满足配置的最大步长。

“Exhaustive” means every unique safe endpoint setting, not every possible ordering of writes. Repeated trajectories to the same endpoint are not separate candidates in version 1.

“穷举”指穷举每一个唯一安全终点设置，而不是穷举所有可能的写入顺序。第一版不把到达同一终点的重复轨迹视为不同候选。

The candidate count is computed before a campaign starts:

```text
candidate_count = product(value_count for each enabled parameter)
                  - deterministically rejected combinations
```

The plan and estimated duration are displayed before authorization. Exhaustive operation is never silently substituted when the operator requested Bayesian operation, or vice versa.

Campaign 开始前必须计算候选数量，并在授权前显示计划和预计时长。不得在未告知操作者的情况下用穷举替代贝叶斯路径，反之亦然。

## 5. Path 1: Exhaustive Characterization / 路径 1：穷举表征

`PATH_EXHAUSTIVE` deterministically enumerates every safe candidate exactly once. Candidate ordering may minimize register transitions but must not change membership or final ranking.

`PATH_EXHAUSTIVE` 以确定顺序将每个安全候选恰好枚举一次。候选顺序可以用于减少寄存器切换，但不得改变候选集合或最终排名。

For every candidate, the orchestrator:

1. generates a max-step-valid transition sequence;
2. validates every intermediate write against the frozen authorization envelope;
3. applies through the adapter;
4. records requested, encoded, applied, and readback settings;
5. acquires TDECQ, ER, status, and artifacts;
6. persists the immutable observation before advancing.

If at least one valid numeric TDECQ exists, the result is:

```python
best = min(valid_observations, key=lambda observation: observation.tdecq_db)
```

Only a fully completed enumeration may report:

```text
GLOBAL_OPTIMUM_ON_ENUMERATED_GRID
```

An interrupted, failed, or partially authorized enumeration reports `EXHAUSTIVE_INCOMPLETE` and the `BEST_TDECQ_FOUND_SO_FAR`; it never claims a grid optimum. If no valid TDECQ exists, it reports `NO_VALID_TDECQ`.

只有完整结束的穷举才能报告 `GLOBAL_OPTIMUM_ON_ENUMERATED_GRID`。中断、失败或授权不完整时报告 `EXHAUSTIVE_INCOMPLETE` 和 `BEST_TDECQ_FOUND_SO_FAR`，不得声称获得网格最优。如果没有任何有效 TDECQ，则报告 `NO_VALID_TDECQ`。

## 6. Parameter-Effect Memory / 参数影响记忆

Path 1 produces a reusable response surface and effect graph in addition to its best point.

路径 1 除了输出最佳点，还生成可复用的响应面和参数影响图。

### 6.1 Response nodes / 响应节点

Each node contains:

```text
driver_profile_hash
VCSEL model and device identity
test-context identity
settings_readback
TDECQ value and validity
ER value and validity
artifact references
```

### 6.2 One-step effect edges / 单步影响边

An edge connects two measured nodes that differ by exactly one legal parameter step:

```text
source settings
target settings
changed parameter
signed step
delta_tdecq = target_tdecq - source_tdecq
```

Edges are conditional effects. The effect of `eq + 1` at one `ibias` value is not assumed to equal its effect at another `ibias` value.

影响边表示条件影响。在一个 `ibias` 值下的 `eq + 1`，不假定与另一个 `ibias` 值下的效果相同。

### 6.3 Derived summaries / 派生摘要

Derived artifacts include:

- global and region-local parameter sensitivity;
- directional effect distributions;
- two-parameter interaction matrices;
- the best-N points;
- connected low-TDECQ regions and robust plateaus;
- TDECQ-invalid regions;
- suggested anchor points for later campaigns.

派生结果包括：

- 全局和局部参数敏感度；
- 有方向的参数影响分布；
- 双参数交互矩阵；
- 最佳 N 个点；
- 连续低 TDECQ 区域和鲁棒平台区；
- TDECQ 无效区域；
- 后续 Campaign 的建议锚点。

Raw observations are immutable. Effect graphs and summaries are versioned derived artifacts that can be recomputed when analysis code changes.

原始观测不可修改。影响图和摘要属于版本化派生制品，分析代码变化后可以重新计算。

### 6.4 Transfer policy / 迁移策略

- Same physical device and unchanged context: the measured lookup table is authoritative for that enumerated grid.
- Same VCSEL model and DriverProfile on a new physical device: effects, top regions, and anchors are weak priors and must be updated by current-device observations.
- Different VCSEL model or DriverProfile: do not transfer fitted posterior confidence or raw register values. Only explicitly mapped semantic parameter roles may contribute weak priors.

- 同一实体器件且测试条件不变：实测查找表是该枚举网格的权威结果。
- 同 VCSEL 型号和 DriverProfile 的新实体器件：影响、优良区域和锚点作为弱先验，必须由当前器件观测更新。
- 不同 VCSEL 型号或 DriverProfile：不得迁移已拟合 posterior 的置信度或原始寄存器值；只有显式映射的语义参数角色可以提供弱先验。

Version 1 assumes endpoint response is path-independent after configured settling. Hysteresis, thermal history, and write-order-dependent response are out of scope; evidence of such behavior requires a later transition-state design.

第一版假定在配置的稳定时间后，终点响应与到达路径无关。滞回、热历史和依赖写入顺序的响应不在第一版范围内；一旦观察到此类证据，需要另行设计带状态的转移模型。

## 7. Path 2: TDECQ-Only Bayesian Tuning / 路径 2：仅 TDECQ 贝叶斯调优

`PATH_TDECQ_BAYESIAN` minimizes valid numeric TDECQ under a finite candidate budget. It does not use ER as an objective, constraint, tie-breaker, acquisition term, convergence rule, or final-selection field.

`PATH_TDECQ_BAYESIAN` 在有限候选预算内最小化有效数值 TDECQ。ER 不作为目标、约束、平局裁决、采集函数项、收敛规则或最终选择字段。

The optimizer maintains:

1. a normalized TDECQ regression model fitted only to valid numeric TDECQ observations;
2. a guarded TDECQ-validity classifier when both valid and invalid classes exist;
3. a coordinate-descent fallback for insufficient or degenerate model data.

优化器维护：

1. 只使用有效数值 TDECQ 观测拟合的标准化 TDECQ 回归模型；
2. 在有效和无效两类都存在时启用的 TDECQ 有效性分类器；
3. 在模型数据不足或退化时使用的坐标下降 fallback。

Its finite-pool acquisition is:

```text
TDECQ minimization expected improvement
  multiplied by P(valid TDECQ)
```

TDECQ validity is a measurement-availability factor, not a performance preference. Missing TDECQ remains `None` plus status and reason and is never converted to zero, a penalty number, or a preceding value.

TDECQ 有效性是测量可用性因素，不是性能偏好。缺失 TDECQ 保持 `None`、状态和原因，禁止转换为零、惩罚数值或前一个值。

The best observation is always:

```python
best = min(valid_observations, key=lambda observation: observation.tdecq_db)
```

`TDECQ < 3.0` is reported as a milestone but does not change the objective or force termination. After crossing 3.0, the optimizer continues to seek lower TDECQ until another configured stop condition occurs.

`TDECQ < 3.0` 作为里程碑报告，但不改变目标，也不强制停止。低于 3.0 后，优化器继续寻找更低 TDECQ，直到其他已配置停止条件发生。

The terminal result is `BEST_TDECQ_FOUND_WITHIN_BUDGET`, never `GLOBAL_OPTIMUM`, unless every safe candidate was measured.

终态结果为 `BEST_TDECQ_FOUND_WITHIN_BUDGET`；除非所有安全候选均已测量，否则不得报告 `GLOBAL_OPTIMUM`。

## 8. Cross-Path Data Use / 双路径数据复用

Both paths write the same immutable observation schema and may reuse an already measured point instead of measuring it again when device identity, context identity, DriverProfile hash, readback settings, and artifact validation all match.

两条路径写入相同的不可变观测结构。当器件身份、测试条件身份、DriverProfile 哈希、回读设置和制品验证均一致时，可以复用已经测量的点而不重复测量。

Recommended usage is:

```text
R&D or a manageable grid:
  PATH_EXHAUSTIVE
  -> grid optimum + effect memory

Routine tuning of later devices:
  PATH_TDECQ_BAYESIAN
  -> use compatible effect memory as weak initialization
  -> update from current-device measurements
```

When both paths are evaluated on the same complete replay surface:

```text
regret = Bayesian best TDECQ - exhaustive grid-best TDECQ
```

The report includes Bayesian measurement count, exhaustive candidate count, regret, invalid count, fallback count, and stop reason.

## 9. Failure and Recovery / 失败与恢复

Deterministic orchestration remains responsible for hardware operations and recovery:

- reject candidates outside the frozen authorization envelope before any hardware call;
- stop on identity mismatch, stale authorization, unsafe combination, or unavailable rollback snapshot;
- treat write/readback mismatch as an application failure, not an observation;
- persist failure evidence before restoring the last authorized snapshot;
- never let optimizer exceptions bypass rollback;
- never train on requested settings when readback differs;
- keep valid ER data even when TDECQ is invalid, without allowing ER to affect selection.

确定性编排层继续负责硬件操作和恢复：

- 在任何硬件调用前拒绝超出冻结授权范围的候选；
- driver 身份不匹配、授权过期、组合不安全或缺少回滚快照时停止；
- 写入与回读不一致属于应用失败，不属于有效观测；
- 恢复最近授权快照前先保存失败证据；
- 优化器异常不得绕过回滚；
- 回读与请求不一致时禁止使用请求值训练；
- TDECQ 无效时仍保存有效 ER 数据，但 ER 不得影响选择。

## 10. Verification / 验证

Required deterministic tests include:

### Candidate space and Path 1

- enumerate every safe candidate exactly once;
- reject every out-of-range, cross-constraint-invalid, or max-step-invalid transition;
- preserve identical membership regardless of traversal optimization;
- return the minimum valid measured TDECQ;
- never claim `GLOBAL_OPTIMUM_ON_ENUMERATED_GRID` after partial enumeration;
- return `NO_VALID_TDECQ` when appropriate.

### Effect memory

- create an edge only for settings differing by exactly one legal parameter step;
- compute signed `delta_tdecq` correctly;
- keep effects conditional on the full source context;
- detect known synthetic parameter interactions;
- isolate artifacts by device, DriverProfile hash, and test context;
- recompute derived artifacts without mutating raw observations.

### Path 2

- fit TDECQ only from valid numeric TDECQ observations;
- changing ER alone cannot change candidate ranking, convergence, or best selection;
- exclude measured, pending, unauthorized, and unsafe candidates;
- request coordinate fallback for insufficient or degenerate model data;
- keep seeded replay deterministic;
- report `BEST_TDECQ_FOUND_WITHIN_BUDGET` when the pool is incomplete.

### Driver qualification and isolation

- validate one minimum-step write, readback, DCA acquisition, and `B0` restoration;
- reject unqualified profile versions;
- verify that no optimizer module imports hardware, network, API, or frontend modules;
- exercise rollback for adapter, acquisition, persistence, and optimizer failures with fake hardware.

## 11. Acceptance Criteria / 验收标准

The design is accepted when:

1. a small synthetic safe grid is exhaustively enumerated and its known lowest valid TDECQ is returned;
2. the same grid produces a correct parameter-effect graph and interaction summary;
3. a seeded TDECQ-only Bayesian replay returns the same result on repeated runs;
4. modifying only replay ER values does not change Bayesian suggestions or final ranking;
5. exhaustive and Bayesian results are compared by identical-grid regret and measurement count;
6. a new fake DriverProfile completes the minimal qualification flow and every injected failure restores its entry snapshot;
7. no optimizer path can contact hardware directly.

## 12. Explicit Non-Goals / 明确不做的内容

Version 1 does not:

- optimize or constrain ER;
- infer unknown register maps or safe ranges;
- claim a continuous-space physical global optimum;
- enumerate every possible write-order trajectory;
- model hysteresis, thermal history, aging, or path-dependent state;
- reuse one device's fitted posterior as confirmed truth for another device;
- claim real-hardware improvement from replay-only results.

第一版不做：

- 优化或约束 ER；
- 推断未知 Reg map 或安全范围；
- 声称获得连续物理空间的全局最优；
- 穷举所有可能的写入顺序轨迹；
- 建模滞回、热历史、老化或路径依赖状态；
- 将一个器件的已拟合 posterior 作为另一个器件的确认事实；
- 根据纯 replay 结果声称真实硬件改善。
