# V2.1 连续优化人工运行参数与无总时限设计

**日期：** 2026-08-16  
**目标代码：** `codex/v2.1-continuous-eye-optimization`  
**范围：** 暂不增加 Kimi 记忆；每次 LIVE 运行由操作员输入优化轮数和目标 TDECQ；取消 continuous campaign 的整体最大时间。

## 1. 已批准的行为

每次 PRE-FLIGHT/COMMISSION 必须显式提供：

- `max_rounds`：任意正整数，不保留 10 或 20 的代码上限；
- `target_tdecq_db`：有限正数；真实提前结束条件为有效测量 `TDECQ < target_tdecq_db`。

LIVE 不再从 YAML 静默取得这两个值。缺少任一输入时，程序在接触硬件前拒绝运行。FAKE 流程可以继续使用测试默认值。

推荐 CLI：

```text
--max-rounds <positive-int>
--target-tdecq-db <positive-float>
```

PowerShell 可以使用 `Read-Host` 收集一次输入，再把相同值传给 PRE-FLIGHT 和 COMMISSION。这样既满足人工输入，也避免 Python CLI 在自动化环境中等待标准输入而卡住。

## 2. 单一事实来源

CLI 解析后的两个值是本次运行的唯一事实来源，并同时进入：

1. PRE-FLIGHT 的 `ContinuousPreflightRequest`；
2. COMMISSION 的 `ContinuousRequest`；
3. Kimi 动态上下文中的 `round.maximum` 和 `target.tdecq_db_lt`；
4. controller 的循环上限和提前结束判断；
5. COMMISSION 人工授权 token。

YAML instruction 不再硬编码 `2.0 dB` 或 `2.8 dB`，而是要求 Kimi 使用结构化上下文中的 `target.tdecq_db_lt`。

授权 token 绑定：

```text
campaign_id + max_rounds + target_tdecq_db + provider=kimi + i2c_only + restore_b0
```

因此 PRE-FLIGHT 之后若改变轮数或目标，原 token 失效，必须重新执行对应范围的人工授权。

## 3. 取消 continuous campaign 整体最大时间

该需求采用“真实取消”，不使用超大秒数伪装无时限。仅 continuous I2C-only COMMISSION 可以创建无累计时限授权；其他 campaign 和 stage 继续保持原有 `max_duration_s` 与 `expires_at` 规则。

continuous 专用授权允许：

- `max_duration_s = null`；
- `expires_at = null`；
- `_validate_budget()` 在 duration 为 `null` 时只检查 `max_candidates`；
- 通用授权仍要求有限正数 duration 和未来 expiry；
- continuous controller 不再产生或检查 `commission_deadline`；
- 删除 `COMMISSION_TIMEOUT` 作为该路径的正常停止原因。

`max_rounds` 仍是不可越过的硬件写入次数预算，因此“无总时间”不等于无限写入。

## 4. 卡死与恢复边界

取消整体最大时间后仍保留：

- Kimi API 单次请求超时 60 秒；
- DCA 单次测量超时 180 秒；
- I2C 串口单次读写超时；
- controller 的异常处理、B0 恢复和 durable bench lease 释放；
- 操作员停止请求；停止请求在当前单次操作结束或超时后生效。

整体 deadline 原本也不能异步中断已经阻塞的 Kimi/DCA 调用；单次操作超时才是该问题的实际保护。因此本改动删除累计 wall-clock 截止条件，但不放宽任何单次 I/O 超时。

进程被操作系统强制终止时，当前进程无法执行 B0 恢复，这是现有路径同样存在的边界。下次运行仍必须经过已有 preflight、身份验证、B0 恢复和 lease/recovery 检查。不得通过取消 duration 绕过这些检查。

## 5. 不变的安全边界

- 每轮最多三个 I2C 参数；
- 参数范围、`max_step`、整数要求、读回一致性保持不变；
- Kimi 只提议，不能授权或执行硬件；
- 不接触 BERT、FIR、NLC 或 `setFir`；
- 完成、失败、人工停止时恢复 B0；
- DCA、I2C 身份和配置验证保持不变；
- 本次不实现 Kimi 长期记忆或跨 campaign 经验复用。

## 6. 失败处理

- 非正整数轮数、非有限/非正目标：接触硬件前退出；
- PRE-FLIGHT 和 COMMISSION 参数或 token 不一致：接触硬件前退出；
- Kimi/DCA/I2C 单次超时：本轮失败，进入既有失败与 B0 恢复路径；
- 达到 `max_rounds` 而未达到目标：`MAX_ROUNDS_REACHED`；
- 达到有效 `TDECQ < target_tdecq_db`：进入已有目标确认/验收路径；
- INVALID 测量不得被当作达到目标。

## 7. 测试范围

自动测试必须覆盖：

1. LIVE 缺少任一人工参数时在设备接触前失败；
2. `max_rounds=1`、`20` 和大于 20 的正整数通过，0/负数/布尔值失败；
3. target 为 2.8 时，2.79 达标、2.80 不达标、INVALID 2.79 不达标；
4. token 精确绑定 campaign、rounds 和 target；
5. continuous 无 duration 授权仍执行候选数量预算；
6. 通用非 continuous 授权仍拒绝空 duration/expiry；
7. Kimi context 收到人工输入的轮数和目标；
8. Kimi、DCA、I2C 单次超时保持；
9. 完成和失败路径继续验证 B0 恢复；
10. 不产生任何 BERT/FIR/NLC 调用。

## 8. 备选方案及取舍

- **保留超大 duration：** 改动最小，但不是真正取消整体时间，拒绝采用。
- **自动按轮数计算 duration：** 安全简单，但用户已明确拒绝，拒绝采用。
- **为所有 campaign 删除 duration：** 影响面过大，拒绝采用。
- **continuous-only 可空 duration：** 满足需求并保护其他工作流，采用。
