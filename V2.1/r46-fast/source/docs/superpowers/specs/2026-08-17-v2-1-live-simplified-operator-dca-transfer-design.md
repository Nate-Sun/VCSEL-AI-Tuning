# V2.1 LIVE 简化操作与 DCA 图像传输修复设计

**日期：** 2026-08-17

**目标代码：** `codex/v2.1-continuous-eye-optimization`

**目标工作树：** `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization`
**范围：** 简化 V2.1 continuous LIVE 操作输入；删除该路径的底层授权 token；修复 DCA 屏幕图像二进制传输的间歇性 VISA 超时；不改变 B0、参数范围、Kimi prompt 或其他硬件策略。

## 1. 已批准的操作界面

每次 LIVE 运行只允许操作员输入：

1. `max_rounds`：任意正十进制整数；
2. `target_tdecq_db`：有限正数，达标条件严格为有效测量 `TDECQ < target_tdecq_db`；
3. `MOONSHOT_API_KEY`：使用安全输入，不写入文件或日志；
4. PRE-FLIGHT 通过后按一次 Enter，明确批准 COMMISSION。

操作员不再输入或编辑：

- Campaign ID；
- PRE-FLIGHT token；
- COMMISSION token；
- Campaign ID 二次确认；
- 临时 session YAML；
- B0、参数范围、步长、Kimi model、prompt、硬件地址或 DCA 配置。

本设计取代 `2026-08-16-v2-1-operator-runtime-controls-design.md` 中关于人工 token 和 Campaign ID 确认的要求，但保留人工输入轮数、目标 TDECQ、无整体最大时间以及所有单次 I/O 超时要求。

## 2. 启动入口与数据流

新增一个签入仓库、要求 PowerShell 7 的正式启动脚本。文档只说明如何调用脚本，不再要求从 Markdown 复制或提取大型 PowerShell 代码块。

脚本执行顺序固定为：

1. 验证 branch、HEAD 基线和 tracked worktree 干净；
2. 验证签入的 LIVE 模板、runtime、hardware 和 B0 配置；
3. 读取并验证 `max_rounds` 与 `target_tdecq_db`；
4. 自动生成唯一 Campaign ID；
5. 复制签入模板到临时文件，并仅机械替换 `live.campaign_id` 占位符；
6. 计算并冻结 session 配置 manifest；
7. 安全读取 `MOONSHOT_API_KEY`；
8. 运行 PRE-FLIGHT；
9. 验证 PRE-FLIGHT 报告为只读操作，且未写 DUT、未接触 BERT；
10. 等待操作员按一次 Enter 批准 COMMISSION；
11. 使用完全相同的 Campaign ID、轮数、目标、HEAD 和配置 manifest 运行 COMMISSION；
12. 无论成功或失败，都删除临时配置并清除进程内 API key。

Campaign ID 继续作为数据库主键、artifact 隔离键和审计标识存在，但只由程序生成和显示，操作员不能修改。建议格式为 `v21-b-<UTC timestamp>-<random suffix>`，同时满足现有安全标识符约束和并发唯一性。

## 3. 删除 V2.1 continuous token

删除范围仅限 V2.1 continuous LIVE 路径：

- PRE-FLIGHT 和 COMMISSION CLI 不再接受或要求 authorization token；
- continuous request/factory 不再携带或校验 token；
- 删除 `commission_authorization_token_for` 及仅服务于该 token 的绑定逻辑；
- PowerShell 启动脚本不生成、不显示、不传递 token；
- 测试不再把 campaign、轮数和目标绑定到 token。

仍保留以下硬件边界：

- 必须显式选择 `PREFLIGHT` 或 `COMMISSION`；
- LIVE 设备接触仍要求 `--allow-device-contact`；
- 正式启动脚本只有在 PRE-FLIGHT 报告验证通过且操作员按 Enter 后才调用 COMMISSION；
- controller 的参数策略、每轮最多三个参数、I2C 读回、DCA/I2C 身份验证、B0 恢复和 durable lease 保持不变；
- 不允许 BERT、FIR、NLC 或 `setFir`。

删除 token 后，直接调用 CLI 并同时提供 `COMMISSION` 与 `--allow-device-contact` 的人员可以绕过 PowerShell 的 Enter 门控。这是用户明确选择方案 2 后接受的风险；runbook 必须突出说明正式实验只应通过签入的 PowerShell 入口启动。

## 4. DCA 图像传输修复

当前 PyVISA 会话只设置 `timeout`，没有设置 `chunk_size`。Keysight 的 FlexDCA 二进制文件传输示例建议将 `chunk_size` 设为 `1024` 字节，并说明较大的缓冲块可能造成间歇性传输失败。V2.1 使用同样的 `:DISK:FILE:READ?` 与 `query_binary_values()` 路径，因此打开 DCA VISA 会话时必须设置：

```text
chunk_size = 1024
```

屏幕证据采集继续采用 Keysight 支持的顺序：

1. 选择 `TIME1` 窗口；
2. 设置唯一 PNG 文件名；
3. 执行 `:DISK:SIMage:SAVE;*OPC?`；
4. 使用 `:DISK:FILE:READ?` 读取 PNG；
5. 验证 PNG；
6. 查询屏幕指纹设置；
7. 持久化原图、模型输入与哈希。

每个原始 VISA 操作必须包装成带阶段和 SCPI 命令的领域错误，至少区分：

- `SCREEN_SAVE`；
- `PNG_BINARY_READ`；
- `FINGERPRINT_QUERY`。

错误必须保留原始异常作为 cause，使日志能显示底层 `VisaIOError`，同时在 campaign 失败原因中给出具体步骤。不得在同一可能已污染输出队列的 VISA session 上进行盲目重试。

## 5. 失败与恢复

- 输入轮数或目标无效：接触硬件前退出；
- 自动 session 配置与模板除 Campaign ID 外存在差异：接触硬件前退出；
- PRE-FLIGHT DCA/I2C/Kimi 失败：不得显示 COMMISSION 批准门，也不得调用 COMMISSION；
- PRE-FLIGHT 报告未明确证明 `dut_write_performed=false`、`bert_contacted=false`：立即退出；
- COMMISSION DCA/Kimi/I2C 失败：进入既有 ABORTING、ROLLING_BACK 和 B0 恢复路径；
- API key 和临时 session 配置在所有退出路径中清除；
- 不恢复 continuous campaign 的整体最大时间；单次 Kimi、DCA、I2C 超时保持。

## 6. 测试策略

所有生产修改采用测试驱动开发，先观察测试因缺少新行为而失败，再做最小实现。

自动测试必须覆盖：

1. DCA VISA session 设置 `timeout` 和 `chunk_size=1024`；
2. 截图保存超时报告 `SCREEN_SAVE` 和具体 SCPI；
3. PNG 读取超时报告 `PNG_BINARY_READ` 和具体 SCPI；
4. 指纹查询超时报告 `FINGERPRINT_QUERY` 和具体 SCPI；
5. 原始异常保留为 cause；
6. V2.1 continuous CLI 不再接受或要求 token；
7. request/factory 不再包含 token；
8. 自动 Campaign ID 满足安全字符规则且连续运行不重复；
9. session YAML 只能机械改变 Campaign ID；
10. 启动脚本只读取轮数、目标、API key 和一次空 Enter；
11. PRE-FLIGHT 失败绝不调用 COMMISSION；
12. PRE-FLIGHT 与 COMMISSION 使用同一冻结范围；
13. 完成和失败继续验证 B0 恢复；
14. 不产生 BERT、FIR、NLC 或 `setFir` 调用。

先运行相关 focused tests，再运行 B 工作树完整测试。自动测试只使用 fake adapter，不接触真实硬件。验证通过后提交 B，并合并到 main；真实硬件测试必须由操作员另行明确开始。

## 7. 非目标

本次不实现：

- Kimi 长期记忆；
- 修改 B0 或九个参数的允许范围与步长；
- 修改 Kimi prompt 或 model；
- 自动保存 API key；
- 自动批准 COMMISSION；
- DCA 超时后的无限或盲目重试；
- 任何 BERT、FIR、NLC 或 `setFir` 功能；
- 修改 A 或其他非 V2.1 continuous 工作流。
