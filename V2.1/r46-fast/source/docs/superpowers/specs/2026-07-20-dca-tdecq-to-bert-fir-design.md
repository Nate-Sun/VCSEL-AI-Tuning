# DCA TDECQ 到 BERT TX FIR 半自动调参设计 / DCA TDECQ-to-BERT TX FIR Semi-Automatic Tuning Design

**日期 / Date:** 2026-07-20
**状态 / Status:** 用户已确认设计，等待实施计划 / User-approved design, awaiting implementation plan
**范围 / Scope:** 仅真实硬件数据路径，不包含数字孪生 / Real-hardware data path only; no digital twin

## 1. 目的 / Purpose

本阶段实现 AI tuning 项目的第一个可验证小闭环：Python 只读 FlexDCA 当前已经完成的 TDECQ 测量，提取 TDECQ 标量、测量状态和 TDECQ equalizer taps，按照已确认的首轮或后续迭代公式生成 BERT 可接受的 7-tap TX FIR，并在人工确认后通过厂商 UDP 命令写入 BERT。

This phase implements the first verifiable small loop of the AI-tuning project. Python reads an already-completed FlexDCA TDECQ measurement without changing the DCA setup, extracts the scalar TDECQ result, measurement status, and TDECQ equalizer taps, generates a BERT-compatible seven-tap TX FIR using the approved first-iteration or later-iteration formula, and writes it to the BERT only after explicit operator confirmation.

本阶段不是全自动 AI 优化。操作员仍然决定迭代类型和 Main 值，并决定是否执行真实写入。

This phase is not fully automatic AI optimization. The operator still selects the iteration type and Main value and decides whether a live write is allowed.

## 2. 已确认需求 / Confirmed Requirements

- BERT 可写 FIR 固定为 `Pre3, Pre2, Pre1, Main, Post1, Post2, Post3` 七项。
  The writable BERT FIR contains exactly seven values: `Pre3, Pre2, Pre1, Main, Post1, Post2, Post3`.
- Main 每轮由操作员人工指定，程序不得自动改变。
  Main is explicitly selected by the operator for every iteration and must not be changed automatically.
- 小数使用普通四舍五入，`1.5 -> 2`、`-1.5 -> -2`。
  Values use conventional half-away-from-zero rounding: `1.5 -> 2` and `-1.5 -> -2`.
- 所有轮次都要求七项绝对值总和不超过 170。
  Every iteration requires the sum of the absolute values of all seven taps to be at most 170.
- BERT 单个 tap 没有额外的软件范围限制。
  There is no additional software limit on an individual BERT tap.
- 第一版 DCA 适配器只读取当前结果，不发送 Autoscale、Clear、Single、Run、Recalculate 或 setup recall。
  The first DCA adapter is read-only and does not send Autoscale, Clear, Single, Run, Recalculate, or setup-recall commands.
- 第一版默认只生成 proposal；真实 BERT 写入要求显式 `--apply` 和第二次人工确认。
  The first version generates proposals by default; a live BERT write requires explicit `--apply` and a second operator confirmation.
- 第一轮和后续迭代必须由显式模式选择，程序不得从历史记录中猜测。
  First and later iterations are selected explicitly; the program must not infer the mode from history.

## 3. 选定方案 / Selected Approach

采用原生协议、模块化 CLI 方案：DCA 使用 PyVISA/SCPI，只读测量；FIR 计算为无仪器依赖的纯数学模块；BERT 使用 UDP 厂商命令；工作流层负责确认、持久化、回读和回退。

Use a native-protocol modular CLI: PyVISA/SCPI for read-only DCA measurements, a pure instrument-independent FIR calculation module, the vendor UDP protocol for the BERT, and a workflow layer for confirmation, persistence, readback, and rollback.

不使用 GUI 鼠标键盘自动化作为主控制路径。GUI 自动化易受窗口布局、分辨率和软件版本影响，只能作为人工诊断工具。

GUI mouse-and-keyboard automation is not the primary control path. It is sensitive to layout, resolution, and software revisions and is reserved for manual diagnosis.

## 4. 模块结构 / Module Structure

```text
src/vcsel_ai_tuning/hardware/
|-- __init__.py
|-- models.py
|-- flexdca_reader.py
|-- fir_calculator.py
|-- bert_fir_constraints.py
|-- vit_bert.py
`-- fir_workflow.py

configs/hardware/
`-- fir_loop.example.yaml

tests_py/ai_tuning/hardware/
|-- test_fir_calculator.py
|-- test_flexdca_reader.py
|-- test_bert_fir_constraints.py
|-- test_vit_bert.py
`-- test_fir_workflow.py
```

### 4.1 `models.py`

定义不可变、严格校验的数据模型：

Defines immutable, strictly validated models:

- `DcaTdecqSnapshot`：仪器身份、时间戳、原始响应、TDECQ dB、状态、状态详情、完整 taps、precursor 数量和 Main 索引。
  `DcaTdecqSnapshot`: instrument identity, timestamp, raw responses, TDECQ in dB, status, status details, full taps, precursor count, and Main index.
- `BertFir`：内部统一顺序的七个整数 tap，以及当前 upper/lower NLC。
  `BertFir`: seven integer taps in canonical order plus the current upper/lower NLC settings.
- `FirProposal`：模式、人工 Main、原始输入、计算中间量、完整卷积、保留/舍弃 taps、最终 FIR、绝对值总和、有效性和 proposal ID。
  `FirProposal`: mode, operator-selected Main, raw inputs, intermediate calculations, full convolution, retained/discarded taps, final FIR, absolute sum, validity, and proposal ID.
- `BertWriteReceipt`：写入前配置、命令、原始响应、写入后配置、验证状态和回退状态。
  `BertWriteReceipt`: before state, command, raw response, after state, verification state, and rollback state.

### 4.2 `flexdca_reader.py`

建立 VISA 会话并只执行查询。默认 Function 编号由配置提供，现场首次读取时验证该 Function 确实为 TDECQ equalizer。

Creates a VISA session and performs queries only. The Function number comes from configuration and is verified during first commissioning to be the active TDECQ equalizer.

### 4.3 `fir_calculator.py`

实现首轮公式、后续卷积公式、Main 对齐、七 tap 截取和普通四舍五入。该模块不导入 PyVISA、socket 或 GUI 库。

Implements the first-iteration formula, later-iteration convolution formula, Main alignment, seven-tap cropping, and conventional rounding. It imports no PyVISA, socket, or GUI library.

### 4.4 `bert_fir_constraints.py`

检查 tap 数量、整数类型、Main 一致性和七项绝对值总和。它不得复用现有通用 `SafetyPolicy` 的 FFE L1 归一化行为，因为该行为会改变人工指定的 Main。

Checks tap count, integer types, Main consistency, and the seven-value absolute sum. It must not reuse the existing generic `SafetyPolicy` FFE L1-normalization behavior because that would change the operator-selected Main.

### 4.5 `vit_bert.py`

封装 `GetMode`、`GetConfiguration` 和 `setFir` UDP 请求，保留厂商原始响应并提供严格解析。真实写入只有在 `GetConfiguration` 解析器通过现场只读 commissioning 后才启用。

Wraps the `GetMode`, `GetConfiguration`, and `setFir` UDP requests, preserves raw vendor responses, and provides strict parsing. Live writes remain disabled until the `GetConfiguration` parser passes read-only commissioning.

### 4.6 `fir_workflow.py`

组织读取、计算、proposal 持久化、apply 前复核、真实写入、读回和一次性回退。

Coordinates reading, calculation, proposal persistence, pre-apply revalidation, live writing, readback, and one rollback attempt.

## 5. DCA 只读接口 / Read-Only DCA Interface

第一版使用以下查询：

The first version uses these queries:

```text
*IDN?
:MEASure:EYE:TDEQ?
:MEASure:EYE:TDEQ:STATus?
:MEASure:EYE:TDEQ:STATus:DETails?
:SPRocess<function>:TEQualizer:TAPS?
:SPRocess<function>:TEQualizer:NPRecursors?
```

若 TDECQ 状态不是 `CORR`、结果无法解析、tap 中存在 NaN/Infinity、Main 缺失或 Main 为零，则本轮读取无效且不生成可应用 proposal。

If the TDECQ status is not `CORR`, the result cannot be parsed, a tap is NaN/Infinity, Main is missing, or Main is zero, the snapshot is invalid and no applicable proposal is generated.

原始 SCPI 字符串与解析后的数值同时保存，以便将来软件版本变化时复核解析器。

Raw SCPI strings and parsed values are both stored so parser behavior can be audited after software revisions.

## 6. FIR 数学定义 / FIR Mathematical Definition

### 6.1 统一顺序 / Canonical order

所有内部 FIR 向量统一采用：

All internal FIR vectors use:

```text
[Pre3, Pre2, Pre1, Main, Post1, Post2, Post3]
```

Main 的内部索引固定为 3。

The internal Main index is always 3.

### 6.2 普通四舍五入 / Conventional rounding

定义：

Define:

```text
R(x) = sign(x) * floor(abs(x) + 0.5)
```

不得使用 Python 内置 `round()`，因为其半值行为与要求不同。

Python's built-in `round()` is not used because its half-value behavior differs from the requirement.

### 6.3 Main 中心截取 / Main-centered crop

对向量 `x` 和 Main 索引 `j`：

For vector `x` and Main index `j`:

```text
Crop7(x, j) = x[j-3 : j+4]
```

如果 Main 任一侧不足三个 tap，proposal 标记为无效，不静默补零。多余的远端 taps 被记录为 `discarded_left` 和 `discarded_right`。

If fewer than three taps exist on either side of Main, the proposal is invalid; zeros are not inserted silently. Extra distant taps are recorded as `discarded_left` and `discarded_right`.

### 6.4 第一轮 / First iteration

设 DCA 完整 equalizer taps 为 `h1`，DCA precursor 数量为 `m1`，人工 Main 为 `M1`：

Let the full DCA equalizer taps be `h1`, the DCA precursor count be `m1`, and the operator-selected Main be `M1`:

```text
u1 = Crop7(h1, m1)
q1 = R(M1 * u1 / h1[m1])
q1[3] = M1
```

最终 `q1` 必须通过七 tap 约束。

The final `q1` must pass the seven-tap constraints.

### 6.5 后续迭代 / Later iterations

设当前 BERT 七 tap 为 `bk`，新 DCA taps 为 `h_next`，DCA Main 索引为 `m_next`，人工 Main 为 `M_next`：

Let the current BERT seven taps be `bk`, the new DCA taps be `h_next`, the DCA Main index be `m_next`, and the operator-selected Main be `M_next`:

```text
c = convolution(bk, h_next)
j = 3 + m_next
u = Crop7(c, j)
q_next = R(M_next * u / c[j])
q_next[3] = M_next
```

如果 `c[j]` 为零或非有限值，本轮无效。

If `c[j]` is zero or non-finite, the iteration is invalid.

卷积使用完整 DCA tap 向量，之后再围绕合成 Main 截取七项。不得在卷积前提前删除 DCA 的远端 taps。

The convolution uses the complete DCA tap vector and is cropped around the combined Main afterward. Distant DCA taps must not be removed before convolution.

### 6.6 舍弃量报告 / Discarded-tap report

报告舍弃 taps 及其 L1 占比：

Report discarded taps and their L1 ratio:

```text
discarded_l1_ratio = sum(abs(discarded)) / sum(abs(full_vector))
```

该指标只用于诊断 7-tap BERT 对 DCA 建议的逼近程度，不作为当前阶段自动优化目标。

This metric diagnoses how well the seven-tap BERT approximates the DCA recommendation; it is not an automatic optimization objective in this phase.

## 7. BERT 约束与命令映射 / BERT Constraints and Command Mapping

有效 proposal 必须满足：

A valid proposal must satisfy:

```text
len(FIR) == 7
all taps are integers
FIR[3] == operator-selected Main
sum(abs(tap) for tap in FIR) <= 170
```

如果总和超过 170，程序不得自动缩放或改变 Main。proposal 标记为无效，并通过对候选整数 Main 逐一复算，报告在相同 tap 比例与舍入规则下可满足约束的最大 Main，供操作员参考。

If the sum exceeds 170, the program must not scale the taps or change Main automatically. The proposal is invalid. For operator guidance, the program recomputes candidate integer Main values using the same ratios and rounding rule and reports the largest Main that satisfies the constraint.

内部顺序到厂商 UDP 命令顺序的映射为：

The canonical-to-vendor UDP mapping is:

```text
internal: Pre3 Pre2 Pre1 Main Post1 Post2 Post3
setFir:   Pre1 Pre2 Pre3 Main Post1 Post2 Post3
```

完整命令还包含 `upperNlc` 和 `lowerNlc`。第一版不得默认写零；必须从 `GetConfiguration` 读取并原样保留当前 NLC 值，除非未来增加单独、显式批准的 NLC 功能。

The complete command also includes `upperNlc` and `lowerNlc`. The first version must not default them to zero; it reads and preserves the current NLC values from `GetConfiguration` unless a future separately approved NLC feature is added.

## 8. Proposal 与真实写入 / Proposal and Live Apply

### 8.1 Proposal

`propose` 只读 DCA 和 BERT，生成不可变 proposal，不发送 `setFir`。proposal 包含当前 BERT FIR 指纹、目标设备、port/channel、所有数学中间量和最终命令预览。

`propose` reads the DCA and BERT and creates an immutable proposal without sending `setFir`. The proposal includes the current BERT FIR fingerprint, target device, port/channel, all mathematical intermediates, and the final command preview.

### 8.2 Apply 前复核 / Pre-apply revalidation

`apply` 前重新读取 BERT 配置。如果当前 FIR 或 NLC 与 proposal 中的 before state 不同，则 proposal 过期并拒绝写入。

Before `apply`, the BERT configuration is read again. If the current FIR or NLC differs from the proposal's before state, the proposal is stale and the write is rejected.

### 8.3 双重确认 / Dual confirmation

真实写入同时要求：

A live write requires both:

1. 命令行显式包含 `--apply`；
   the command line explicitly includes `--apply`; and
2. 操作员在提示中输入完整 proposal ID。
   the operator enters the complete proposal ID at the confirmation prompt.

### 8.4 写入验证 / Write verification

固定流程：

Fixed sequence:

```text
read and persist before state
-> send setFir once
-> capture raw UDP response
-> query GetConfiguration
-> compare all seven taps and preserved NLC values exactly
```

只有读回完全匹配才是 `VERIFIED`。

Only an exact readback match is `VERIFIED`.

## 9. 错误处理与回退 / Error Handling and Rollback

发生 UDP 超时、厂商错误响应、解析失败或读回不一致时：

On UDP timeout, vendor error response, parse failure, or readback mismatch:

1. 停止后续动作；
   stop further actions;
2. 使用已保存的 before state 尝试一次回退；
   attempt one rollback using the persisted before state;
3. 再次 `GetConfiguration` 验证；
   verify again with `GetConfiguration`;
4. 成功则状态为 `ROLLED_BACK`；
   mark `ROLLED_BACK` on success;
5. 无法确认则状态为 `RECOVERY_REQUIRED`，锁定后续真实写入，直到操作员显式解除。
   otherwise mark `RECOVERY_REQUIRED` and block subsequent live writes until explicit operator acknowledgement.

自动回退最多一次，不允许循环写入。DCA 在所有失败路径中仍保持只读。

Automatic rollback is attempted at most once; repeated write loops are forbidden. The DCA remains read-only on every failure path.

## 10. CLI 与配置 / CLI and Configuration

计划中的命令：

Planned commands:

```text
vcsel-tune fir read
vcsel-tune fir propose --mode first --main 110
vcsel-tune fir propose --mode iterative --main 104
vcsel-tune fir apply <proposal-id> --apply
vcsel-tune fir acknowledge-recovery
```

示例配置只提供结构，不提供可误用的真实硬件默认地址：

The example configuration provides structure but no live hardware address that could be used accidentally:

```yaml
dca:
  visa_resource: null
  function: 1
  timeout_ms: 60000

bert:
  ip: null
  udp_port: 9000
  port: port1
  channel: "0"

fir:
  tap_order: [pre3, pre2, pre1, main, post1, post2, post3]
  absolute_sum_limit: 170
```

当 `visa_resource` 或 BERT `ip` 为空时，只允许 fixture/replay 测试；不得 live connect 或 apply。

When `visa_resource` or BERT `ip` is null, only fixture/replay testing is allowed; live connect and apply are forbidden.

## 11. 制品与可追溯性 / Artifacts and Traceability

每次运行保存：

Every run stores:

```text
artifacts/fir_runs/<run-id>/
|-- dca_raw.json
|-- bert_before.json
|-- calculation.json
|-- proposal.json
|-- command_preview.txt
|-- bert_after.json
|-- write_receipt.json
`-- events.jsonl
```

`calculation.json` 保存原始 taps、Main 索引、归一化值、完整卷积、保留/舍弃 taps、舍弃占比、舍入前后结果和绝对值总和。

`calculation.json` stores raw taps, Main indices, normalized values, full convolution, retained/discarded taps, discarded ratio, pre/post-rounding results, and the absolute sum.

TDECQ 标量不参与 FIR 公式，只用于记录本轮写入前的链路质量，并与下一轮新测量比较。

The scalar TDECQ value is not an FIR formula input. It records link quality before the write and is compared with the next manually initiated measurement.

## 12. 测试设计 / Test Design

### 12.1 Excel 固定样例 / Fixed Excel examples

```text
First example, Main 90:
[1, 5, -20, 90, -6, 2, -1], absolute sum 125

Second first-iteration example, Main 110:
[0, 6, -20, 110, -18, -14, 2], absolute sum 170

Later-iteration example, Main 104:
[0, 6, -26, 104, -7, -8, -2], absolute sum 153
```

这些期望值来自提供的 `Calculate TX FIR.xlsx`，但按实际 7-tap BERT 约束修正了工作簿中包含 `Post4` 的 8-tap布局。

These expected values come from the supplied `Calculate TX FIR.xlsx`, with its eight-tap layout including `Post4` corrected to the actual seven-tap BERT constraint.

### 12.2 单元测试 / Unit tests

- 正负半值四舍五入；
  positive and negative half-value rounding;
- Main 位置由 precursor 数量确定；
  Main position derived from precursor count;
- 完整卷积后再截取；
  crop after full convolution;
- 总和 170 通过、171 拒绝；
  absolute sum 170 passes and 171 fails;
- 零 Main、NaN、Infinity、tap 数量不足拒绝；
  reject zero Main, NaN, Infinity, and insufficient taps;
- 内部顺序到 `setFir` 顺序映射；
  canonical-to-`setFir` order mapping;
- NLC 原样保留。
  exact NLC preservation.

### 12.3 假设备集成测试 / Fake-instrument integration tests

- DCA 命令记录中只出现查询；
  only queries appear in the DCA command log;
- 默认路径从不发送 `setFir`；
  the default path never sends `setFir`;
- 缺少 `--apply` 或确认 ID 时拒绝；
  reject missing `--apply` or confirmation ID;
- stale proposal 拒绝；
  reject stale proposals;
- 完整读回才成功；
  require complete readback for success;
- 不一致时只回退一次；
  rollback only once after mismatch;
- 未确认回退时进入恢复锁。
  enter recovery lock when rollback cannot be verified.

## 13. 真实硬件 Commissioning / Live-Hardware Commissioning

Commissioning 分三步并分别取得操作员授权：

Commissioning has three separately authorized steps:

1. **只读连接 / Read-only connection**：DCA `*IDN?`、TDECQ/status/taps/precursors；BERT `GetMode` 和 `GetConfiguration`。
   DCA `*IDN?`, TDECQ/status/taps/precursors; BERT `GetMode` and `GetConfiguration`.
2. **现场 dry-run / Live-data dry run**：用真实读取值生成 proposal，与 Excel 人工结果比较，只显示命令。
   Generate a proposal from live readings, compare it with the manual Excel result, and display the command only.
3. **受控写入 / Controlled write**：先将当前 FIR 原样写回并验证，再在单通道上应用一组人工批准的新 FIR。
   First write the current FIR back unchanged and verify it, then apply one operator-approved new FIR to one channel.

现场必须提供或只读发现 DCA VISA resource、BERT IP、port/channel，以及 `GetConfiguration` 原始响应。没有这些已经验证的现场信息，live apply 保持关闭。

Commissioning must provide or discover, through read-only operations, the DCA VISA resource, BERT IP, port/channel, and raw `GetConfiguration` response. Live apply remains disabled without these verified bench values.

## 14. 第一阶段验收标准 / Phase-One Acceptance Criteria

- Python 能读取当前 FlexDCA 已有的 TDECQ、状态、equalizer taps 和 precursor 数量。
  Python reads the current FlexDCA TDECQ, status, equalizer taps, and precursor count.
- 读取过程不改变 DCA 配置或采集状态。
  Reading does not change DCA configuration or acquisition state.
- 首轮与后续迭代公式分别实现并通过 Excel 固定样例。
  The first and later formulas are implemented separately and pass the fixed Excel examples.
- 输出严格为七个整数 taps，Main 等于人工值，总和不超过 170。
  Output is exactly seven integer taps, Main equals the operator value, and the sum is at most 170.
- 报告显示完整计算、舍弃 taps、约束结果和 BERT 命令预览。
  The report shows the full calculation, discarded taps, constraint results, and BERT command preview.
- 默认不写 BERT。
  BERT is not written by default.
- 在 BERT 配置解析器通过只读 commissioning 前，真实 apply 不可用。
  Live apply is unavailable until the BERT configuration parser passes read-only commissioning.
- 所有单元和 fake-instrument 集成测试通过。
  All unit and fake-instrument integration tests pass.

## 15. 明确不在本阶段范围 / Explicitly Out of Scope

- 自动触发或重新配置 DCA 测量；
  automatically triggering or reconfiguring DCA measurements;
- 自动选择 Main；
  automatically selecting Main;
- 自动连续多轮闭环；
  an automatic continuous multi-iteration loop;
- 贝叶斯优化、深度学习或强化学习；
  Bayesian optimization, deep learning, or reinforcement learning;
- 自动修改 NLC、bias、modulation 或其他硬件参数；
  automatically changing NLC, bias, modulation, or other hardware settings;
- 未经人工确认的硬件写入；
  hardware writes without operator confirmation;
- 数字孪生或 Cadence 后端。
  digital-twin or Cadence backends.

## 16. 参考输入 / Reference Inputs

- `Calculate TX FIR.xlsx`：首轮归一化和后续卷积公式的实验室参考。
  `Calculate TX FIR.xlsx`: laboratory reference for first-iteration normalization and later convolution.
- `Cronus_LDD_PAM4_100G-20250818.setx`：FlexDCA TDECQ Function1、9-tap equalizer 和测量设置参考。
  `Cronus_LDD_PAM4_100G-20250818.setx`: reference for FlexDCA TDECQ Function1, the nine-tap equalizer, and measurement settings.
- `Reslib.py`：旧版 `OSC_N1092C` SCPI 查询与 `BERT_VIT8X_TH5` UDP `setFir` 协议线索。
  `Reslib.py`: legacy `OSC_N1092C` SCPI queries and `BERT_VIT8X_TH5` UDP `setFir` protocol evidence.
- Keysight TDECQ SCPI：<https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/MEASure/EYE_TDEQ.htm>
  Keysight TDECQ SCPI: <https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/MEASure/EYE_TDEQ.htm>
- Keysight TDECQ Python 示例：<https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Python/TDECQ_parallel.htm>
  Keysight TDECQ Python example: <https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Python/TDECQ_parallel.htm>
