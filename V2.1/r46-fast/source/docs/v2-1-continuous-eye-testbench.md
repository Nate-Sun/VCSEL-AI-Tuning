# V2.1 Continuous Eye Optimization Testbench Evidence / V2.1 连续眼图优化 Testbench 证据

## Provenance / 来源与血缘

- Design start commit / 设计起始提交: `1e648755477f7b848d2bd98498a4f8d6438e0e1c`
- Implementation commit / 实现提交: `098bc08b8b5b1657690bd612092d30773723a65f`
- LIVE preflight bootstrap commits / LIVE preflight 引导提交: `e674c85`, `db0f4f8`, `d7e6445`
- Operator-runtime control commits / 操作员运行时控制提交: `c7bad406`, `8edeaf6c`, `8bb1d6cd`,
  `1c063570`, `f74e01b6`, `0ed27587`, `8a9babdf`, `ce6313d7`, `a4c426b3`,
  `54ddc8a6`, `347d2ec5`, `ff5187be`
- Reviewed runtime baseline / 已审查运行时基线: `ff5187be`
- Branch / 分支: `codex/v2.1-continuous-eye-optimization`
- Worktree / 工作树: `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization`
- Execution policy used for implementation and verification / 实现和验证使用的执行策略:
  `FAKE` and local mocks only / `FAKE` 和仅本地 mocks

The V2.1 continuous branch is based on the V1.1-derived V1.2 integration
lineage. It does not validate the separate FAST24/FIR-writing V1.2
implementation. / V2.1 continuous 分支源自 V1.1-derived V1.2 集成血缘，
不代表另一个 FAST24/FIR-writing V1.2 实现。

The implementation includes production bottom-layer contracts, controller,
policy, evidence persistence, existing-orchestrator execution port, FlexDCA
screen-evidence bridge, lazy LIVE composition, and a thin Testbench CLI. The
Testbench does not own optimization loops, proposal validation, I2C/DCA calls,
acceptance, recovery, or proof rules.

该实现包含 production 底层 contracts、controller、policy、证据持久化、现有
orchestrator execution port、FlexDCA 屏幕证据桥、延迟 LIVE composition 和轻量
Testbench CLI。Testbench 不负责优化循环、proposal 验证、I2C/DCA 调用、验收、
恢复或证明规则。

## Safety boundary / 安全边界

- The continuous path accepts only the nine approved integer I2C fields and at
  most three sparse absolute changes per proposal. / continuous path 只接受九个已批准的整数 I2C 字段，
  每个 proposal 最多包含三个稀疏绝对值变更。
- Local immutable materialization and the existing V2.0 authorization path own
  every execution decision; the provider has no hardware object or authority. /
  本地不可变 materialization 和现有 V2.0 authorization path 掌管每个执行决定；
  provider 不持有硬件对象或权限。
- FIR, NLC, BERT mutation, and `setFir` are excluded. The continuous LIVE
  adapter uses a BERT placeholder that rejects preflight, readback, and apply. /
  FIR、NLC、BERT mutation 和 `setFir` 均被排除；continuous LIVE adapter 使用
  会拒绝 preflight、readback 和 apply 的 BERT placeholder。
- Static terminal recovery restores and verifies only the nine I2C values from
  B0. / 静态终态恢复只恢复并验证 B0 中的九个 I2C 值。
- A stale durable bench lease, lost continuity, or unverified B0 restore yields
  `RECOVERY_REQUIRED` and never auto-resumes. / 过期的 durable bench lease、
  continuity 丢失或未经验证的 B0 restore 会产生 `RECOVERY_REQUIRED`，且绝不
  自动恢复运行。
- `PREFLIGHT` and `COMMISSION` remain distinct LIVE operations. The checked-in
  launcher preserves the human approval checkpoint between them; external
  operator token entry has been removed. / `PREFLIGHT` 与 `COMMISSION` 仍是两个
  独立 LIVE 操作；签入的 launcher 保留两者之间的人工批准关卡，但不再要求操作员
  输入外部 token。
- Continuous preflight constructs no BERT client and makes no `i2c.apply`, FIR,
  NLC, or `setFir` call. Its report must record `dut_write_performed: false`,
  `hardware_write_performed: false`, `hardware_write_scope:
  NO_DUT_REGISTER_OR_BERT_WRITE`, `bert_contacted: false`, and
  `dca_acquisition_commands_performed: true`. / Continuous preflight 不构造
  BERT client，也不调用 `i2c.apply`、FIR、NLC 或 `setFir`；其报告必须证明 DUT
  与 BERT 未写入、DCA 已采集。
- The V2.1 LIVE runtime uses an isolated Campaign database and artifact root;
  it does not reuse another V1.2/V2 Campaign merely because the displayed
  version or API route looks similar. / V2.1 LIVE runtime 使用隔离的 Campaign
  database 和 artifact root；不会仅因显示版本或 API route 相似就复用另一个
  V1.2/V2 Campaign。
- `closed_loop_proven` is always false in FAKE mode. / 在 FAKE mode 中，
  `closed_loop_proven` 始终为 false。

## Non-hardware verification / 非硬件验证

The implementation and automated verification use FAKE adapters and local
mocks only. They do not run the signed LIVE launcher, call Kimi, read an API
key, open VISA or serial resources, or contact DCA, I2C, or BERT hardware. /
实现和自动化验证仅使用 FAKE adapters 与本地 mocks；不会运行正式 LIVE
launcher，不会调用 Kimi、读取 API key、打开 VISA/串口，也不会接触 DCA、I2C
或 BERT 硬件。

Full-suite and focused verification commands are recorded here for repeatable
non-hardware checks / 可重复的非硬件完整与聚焦检查命令：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -3.13 -P -m pytest -q
py -3.13 -P -m pytest tests_py/ai_tuning/test_continuous_live.py -k "launcher or runbook" -q
```

Historical independently reviewed result at pre-documentation HEAD `2bf84f22`,
which contains runtime baseline `ff5187be`: `2442 passed, 3 skipped`. The three
skips were Windows symlink-creation privilege checks, and no warnings were
emitted. Current change verification must record its own exact result instead
of copying that historical count. / 历史独立复核结果为 `2442 passed, 3 skipped`；
三个 skip 是 Windows 创建符号链接权限检查，且没有 warning。本次修改必须记录
自己的实际结果，不能复制历史数字。

## FAKE Testbench scenarios / FAKE Testbench 场景

All scenarios used the example config and the same production controller. /
所有场景都使用示例配置和同一个 production controller：

| Scenario / 场景 | State / 状态 | Stop reason / 停止原因 | Rounds / 轮数 | Closed-loop proof / 闭环证明 |
| --- | --- | --- | ---: | --- |
| `happy` | `COMPLETED` | `MAX_ROUNDS_REACHED` | 3 | false |
| `readback-mismatch` | `FAILED_SAFE` | `ROUND_FAILURE` | 0 | false |
| `dca-failure` | `FAILED_SAFE` | `ROUND_FAILURE` | 0 | false |
| `b0-recovery-failure` | `RECOVERY_REQUIRED` | `B0_RECOVERY_FAILED` | 0 | false |
| `stop-acceptance` | `COMPLETED` | `PROVIDER_STOP_ACCEPTED` | 0 | false |

## LIVE entry and operator inputs / LIVE 入口与操作员输入

No LIVE PRE-FLIGHT or COMMISSION command was run while creating or testing this
launcher. When the laboratory is ready and separate human authorization has
been obtained, start only the checked-in PowerShell 7 launcher from the B
worktree root / 创建和测试 launcher 时没有运行 LIVE PRE-FLIGHT 或 COMMISSION。
实验室准备就绪且已取得独立人工批准后，只能从 B worktree 根目录启动签入的
PowerShell 7 launcher：

```powershell
pwsh -NoLogo -NoProfile -File .\scripts\run_v2_1_continuous_live.ps1
```

The launcher accepts exactly four types of operator interaction / launcher 只接受
以下四类人工交互：

1. maximum optimization rounds / 最大优化轮数；
2. target TDECQ, interpreted as a strict-below goal / 目标 TDECQ，按严格小于判断；
3. secure `MOONSHOT_API_KEY` input / 安全输入 `MOONSHOT_API_KEY`；
4. one empty confirmation at the COMMISSION Enter gate, after the PRE-FLIGHT
   report has passed the read-only audit / PRE-FLIGHT 报告通过只读审计后，在
   COMMISSION Enter gate 进行一次空确认。

The launcher validates maximum rounds and target TDECQ before it generates or
materializes a Campaign. Campaign ID is then generated and displayed
automatically in the form `v21-b-UTCtimestamp-random`. The launcher copies no
editable settings: it reads the checked-in template, requires exactly one
campaign placeholder, replaces that placeholder in a temporary YAML, and
freezes the campaign, maximum rounds, target TDECQ, Git HEAD, temporary config
path, and config manifest across both operations. B0, ranges, model, prompt,
runtime, and hardware configuration stay as checked in. No authorization token is entered or generated. / launcher 会先验证最大轮数与目标 TDECQ，然后才生成并
物化 Campaign。Campaign ID 以 `v21-b-UTCtimestamp-random` 形式自动生成并显示。
launcher 只替换临时 YAML 中唯一的 campaign placeholder，并在两次操作之间冻结
campaign、轮数、目标、Git HEAD、配置路径和 manifest；B0、范围、模型、prompt、
runtime 与硬件配置保持签入值。无需输入或生成 authorization token。

The PRE-FLIGHT report must have `operation=PREFLIGHT`,
`dut_write_performed=false`, `hardware_write_performed=false`,
`hardware_write_scope=NO_DUT_REGISTER_OR_BERT_WRITE`, `bert_contacted=false`,
and `dca_acquisition_commands_performed=true`. Any mismatch or nonzero exit
stops before the COMMISSION gate. COMMISSION retains `--allow-device-contact`,
I2C-only policy, internal `StageAuthorization`, leases, identity checks, B0
restore, round limit, and strict target handling. / PRE-FLIGHT 任一只读证明不符
或退出码非零都会在 COMMISSION 关卡前停止。COMMISSION 仍保留底层设备接触门、
I2C-only policy、内部 `StageAuthorization`、lease、身份检查、B0 恢复、轮数限制
和严格目标判断。

Every launcher-owned Python process uses `py -3.13 -P`; the launcher supplies
the checked-in `src` path through `PYTHONPATH` where project imports are needed.
The frozen scope rejects tracked drift, any ordinary untracked path outside
`artifacts/`, `outputs/`, and `state/`, and ignored executable or configuration
files outside those evidence roots. / launcher 发起的所有 Python 进程都使用
`py -3.13 -P`；需要项目 import 时仅通过签入的 `src` 设置 `PYTHONPATH`。冻结
范围会拒绝 tracked 漂移、三个证据目录之外的普通 untracked 路径，以及证据目录
之外被 ignore 的可执行或配置文件。

## DCA transfer diagnostics / DCA 传输诊断

FlexDCA evidence failures now report both a stage and the exact SCPI command.
Use the stage to distinguish setup, save, binary transfer, and fingerprint
failures / FlexDCA 证据失败现在同时报告 stage 与准确 SCPI command，可用下列
stage 区分设置、保存、二进制传输和 fingerprint 故障：

- `SCREEN_WINDOW_SELECT`
- `SCREEN_FILENAME`
- `SCREEN_SAVE`
- `PNG_BINARY_READ`
- `FINGERPRINT_QUERY`

The original VISA exception remains the cause. Do not blindly retry; preserve
the complete traceback and campaign evidence before diagnosis. / 原始 VISA
异常保留为 cause；不要盲目重试，诊断前须保存完整 traceback 与 campaign 证据。

## Evidence and rerun rules / 证据与重跑规则

The launcher's nested cleanup path always attempts API-key removal even if
temporary-YAML removal fails. A cleanup failure never replaces the primary LIVE
failure. Cleanup removes only the temporary session YAML and the process-local
API-key environment variable; it never removes persisted test or recovery
evidence. Do not delete `artifacts/`, `outputs/`, `state/`, or any evidence. /
launcher 的嵌套清理路径即使临时 YAML 删除失败，也仍会清理 API key；清理异常
不会覆盖原始 LIVE 异常。清理只涉及临时会话 YAML 与进程内 API-key 环境变量，
绝不会删除已持久化的测试或恢复证据。不得删除 `artifacts/`、`outputs/`、`state/` 或任何证据。

If PRE-FLIGHT or COMMISSION fails, keep the artifacts and traceback, resolve the
reported failure, and start a new launcher process. A new process creates a new
Campaign ID and a new frozen scope. Never resume a campaign in
`RECOVERY_REQUIRED`, never reuse another campaign's ID, and never place
`MOONSHOT_API_KEY` in YAML, artifacts, logs, source control, or chat. / 若
PRE-FLIGHT 或 COMMISSION 失败，应保留 artifacts 与 traceback，解决已报告故障，
再启动新的 launcher 进程。新进程会创建新的 Campaign ID 与冻结范围。不得恢复
`RECOVERY_REQUIRED` campaign，不得复用其他 campaign ID，也不得把
`MOONSHOT_API_KEY` 放进 YAML、artifacts、logs、源代码控制或聊天。
