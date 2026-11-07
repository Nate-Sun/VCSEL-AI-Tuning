# 接手指南 / Handoff guide

整理日期 / Prepared: 2026-11-06。给接手这个项目的同事：先读本页，再读 [结果总结 / Results](RESULTS.md)。
For the colleague taking over: read this page first, then [Results](RESULTS.md).

## 1. 这个项目做什么 / What the project does

它用程序自动调整 VCSEL 驱动芯片的寄存器和 BERT 的 FIR 抽头，让 PAM4 眼图的 TDECQ 变低。每一轮都经过安全门禁，结束后硬件必须回到已知基线 B0。
It automatically adjusts VCSEL driver registers and BERT FIR taps to lower PAM4 TDECQ. Every round passes safety gates, and hardware must return to the known baseline B0 at the end.

```text
                 ┌──────────── 安全门禁 / safety gates ────────────┐
优化器 / optimizer ──► I2C (COM7, 0xA6) ──► 驱动芯片 / driver chip ──► VCSEL ──► 光 / optical
       ▲          └► BERT (UDP 192.168.3.2:9000): SSPRQ 码型 + FIR 抽头 / pattern + FIR taps
       │                                                                   │
       └──── TDECQ (dB) ◄── Keysight N1010A (FlexDCA, CHAN1A / FUNC1) ◄────┘
             记录 → SQLite + dataset bundle；结束时回滚到 B0 / persisted; restored to B0
```

链路参数：53.125 GBd PAM4，850 nm，SSPRQ。地址和端口是原工位的值，换工位必须改（见第 5 节）。
Link: 53.125 GBd PAM4, 850 nm, SSPRQ. Addresses and ports are from the original bench and must be changed for a new bench (see section 5).

## 2. 选哪个版本 / Which version to use

| 想做的事 / Goal | 用这个 / Use | 说明 / Notes |
|---|---|---|
| 在硬件上跑经过验证的流程 / Run the validated flow on hardware | `V1.2/source` | 唯一做过 soak 测试的版本；策略 PILOT_3 → FAST24 / Only soak-tested version |
| 不连硬件，熟悉界面和流程 / Learn the UI and flow offline | `V1.2/source`，FAKE 模式 / FAKE mode | `start-v1.2.cmd -Mode FAKE` |
| 研究 AI（大模型）决策框架 / Study the AI decision framework | `V2.0/source/src/vcsel_ai_tuning/v2/` | 只有接口和测试用假 provider，没有接到 API 或硬件流程 / Interfaces and test fakes only; not wired to API or hardware |
| 继续 Kimi 连续优化实验 / Continue Kimi experiments | `V2.1/b0-70-22-fast` 或 `V2.1/continuous` | 两者都没达到目标；见下表 / Neither met its target; see below |

V2.1 五个分支的区别 / The five V2.1 variants:

| 分支 / Variant | 实际内容 / What it actually is |
|---|---|
| `kimi-live` | V2.0 + Kimi API 客户端，只有库代码，没有启动入口 / V2.0 + Kimi client; library only, no launcher |
| `continuous` | 连续眼图优化 + LIVE 启动脚本；**最后一次提交加入了演示用的脚本模式**（固定 20 点，覆盖 Kimi 建议）/ Continuous eye optimization + LIVE launcher; **last commit adds a demo scripted mode** (fixed 20 points, overrides Kimi) |
| `r46-fast` | continuous（无脚本模式），B0 换成 campaign b468337d 第 46 轮的回读值 / continuous without scripted mode; B0 = round-46 readback of campaign b468337d |
| `b0-70-22-fast` | 同 r46-fast，但 B0 为 `ibias_dac=70, eq=22`，并在每次 Kimi 请求里写入 150 轮历史摘要；250 轮报告中的最好结果 2.66 dB 来自这里 / Same as r46-fast but B0 `ibias_dac=70, eq=22`, with a 150-round history summary in every Kimi request; produced the 2.66 dB best |
| `desktop` | continuous + Tkinter 桌面界面和打包脚本；没有运行数据 / continuous + Tkinter GUI and packaging; no run data |

注意：`b0-70-22-fast` 没有自己的设计文档，沿用了 R46 的文档；那份文档写着“Kimi 不接收历史”，与该分支代码相反。
Note: `b0-70-22-fast` reuses the R46 design docs, which say Kimi receives no history — the opposite of this variant's code.

## 3. 调参参数 / Tuning parameters

驱动芯片寄存器（`V1.2/source/configs/hardware/vcsel_lab_live.yaml`）：
Driver registers (`V1.2/source/configs/hardware/vcsel_lab_live.yaml`):

| 寄存器 / Register | 地址 / Addr | 位宽 / Bits | B0 (V1.2) | V1.2 搜索范围 / V1.2 range | V2.1 允许范围 / V2.1 range |
|---|---|---|---|---|---|
| `ibias_dac` | 0x2002 | 7 | 70 | 60–80 | 60–80 |
| `efin_cur` | 0x2003 | 4 | 15 | 固定 / fixed | 10–15 |
| `vga_cur` | 0x2005 | 4 | 13 | 固定 / fixed | 10–15 |
| `eq` | 0x2006 | 5 | 22 | 固定 / fixed | 16–25 |
| `res_gain` | 0x2007 | 4 | 6 | 0–10 | 0–10 |
| `res_bw` | 0x2008 | 4 | 0 | 固定 / fixed | 0–5 |
| `gain_ctrl` | 0x2009 | 6 | 5 | 固定 / fixed | 0–10 |
| `vbias_ref` | 0x200A | 4 | 15 | 固定 / fixed | 10–15 |
| `vref` | 0x200B | 4 | 15 | 固定 / fixed | 10–15 |

- V1.2 的搜索范围来自前期扫描结论；扫描数据不在本仓库。V2.1 每步最多改 3 个参数（`V2.1/*/source/configs/ai_tuning/v2_1_continuous.live.yaml`）。
  V1.2 ranges come from an earlier sweep (data not in this repository). V2.1 changes at most 3 parameters per step.
- 各寄存器的物理含义请以芯片手册为准；本仓库没有手册。
  Refer to the chip datasheet for each register's physical meaning; the datasheet is not in this repository.
- 位宽只是编码范围，不是安全范围（配置里 `safety_limits: {}` 故意为空）。V1.2 的实际限制写在 `src/vcsel_ai_tuning/v1_1/live_ai_policy.py`。
  Bit width is encoding, not a safety limit (`safety_limits: {}` is intentionally empty). V1.2's real limits are in `src/vcsel_ai_tuning/v1_1/live_ai_policy.py`.
- FIR：7 个抽头，B0 为 `[0,9,-28,104,-8,1,-3]`，NLC 上下均为 6。
  FIR: 7 taps, B0 `[0,9,-28,104,-8,1,-3]`, upper/lower NLC 6.

## 4. 术语表 / Glossary

更完整的定义见 `V1.2/source/CONTEXT.md`。
Fuller definitions are in `V1.2/source/CONTEXT.md`.

| 术语 / Term | 含义 / Meaning |
|---|---|
| TDECQ | Transmitter and Dispersion Eye Closure Quaternary，PAM4 发射机眼图闭合代价（dB），越低越好；本项目唯一的优化目标 / PAM4 transmitter eye-closure penalty; lower is better; the only objective |
| SSPRQ | PAM4 测试用的短伪随机码型 / Short stress pattern random quaternary test pattern |
| NLC | BERT 发送端的上/下 NLC 设置（`upper_nlc` / `lower_nlc`），本项目中冻结为 6/6 / BERT transmitter upper/lower NLC settings, frozen at 6/6 here |
| B0 | 冻结的已知安全基线（寄存器、FIR、码型、通道、仪器身份），每次任务结束都必须回到这里 / Frozen known-good baseline every task must return to |
| PILOT_3 | 先试 3 个候选，每个之后恢复 B0；通过安全门禁才进入 FAST24 / 3 trial candidates, B0 restored after each; gate to FAST24 |
| FAST24 | 24 轮：6 粗调 + 16 细调（I2C）+ 1 公式 FIR + 1 迭代 FIR / 24 rounds: 6 coarse + 16 fine I2C, 1 formula FIR, 1 iterative FIR |
| Campaign | 一次完整任务，V1.2 中共 29 条记录 / One full task; 29 records in V1.2 |
| Soak | 连续跑多个 campaign（2h、8h）检验稳定性 / Back-to-back campaigns (2h, 8h) to test stability |
| FAKE / READ_ONLY / LIVE_TUNING | 纯软件模拟 / 只读连接仪器 / 可写硬件 / Software only / read-only instrument access / hardware writes allowed |
| PER_STAGE | 每个写入阶段都需操作员授权 / Operator authorization for every mutating stage |
| RECOVERY_REQUIRED | B0 恢复无法核实时的终止状态，必须专门恢复后才能继续 / Terminal state when B0 cannot be verified |
| COMPLETE / PARTIAL | dataset 是否完整；PARTIAL 记录实际是 TDECQ 测量失败 / Dataset completeness; PARTIAL records are in fact failed TDECQ readings |
| same_acquisition_proof | 证明图像、指标和回读来自同一次采集 / Proof that image, metrics and readback come from one acquisition |
| Native V1.2 vs V1.1-derived V1.2 | 两条不同谱系，界面都显示 1.2.0；本仓库 `V1.2/` 是 Native。注意 Native V1.2 的 LIVE 代码仍放在 `src/vcsel_ai_tuning/v1_1/` 目录下 / Two lineages that both show 1.2.0; `V1.2/` here is Native, though its LIVE code lives under `v1_1/` |
| R46 / 70-22 | V2.1 分支名：第 46 轮回读作为 B0 / `ibias_dac=70, eq=22` 作为 B0 / V2.1 variant names: round-46 readback as B0 / `ibias_dac=70, eq=22` as B0 |
| Scripted mode | V2.1 continuous 的演示模式，按固定点运行，不是 AI 决策 / Demo mode in V2.1 continuous; fixed points, not AI decisions |

## 5. 换工位需要改的地方 / Changing benches

配置里写的是原工位的主机名、COM 口和 IP。换工位时逐项修改，然后先跑 `-ValidateOnly`，再跑 READ_ONLY preflight。
Configs contain the original bench's host name, COM port and IPs. Update each item, run `-ValidateOnly`, then READ_ONLY preflight.

| 文件 / File | 要改的字段 / Fields |
|---|---|
| `V1.2/source/configs/hardware/vcsel_lab_live.yaml` | `dca.resource`、`dca.expected_idn_prefix`、`bert.ip/local_ip/udp_port`、`i2c.port/device_address/expected_chip_id` |
| `V1.2/source/configs/ai_tuning/runtime.v1.2.read-only.yaml`、`runtime.v1.2.live.yaml` | `hardware.dca_visa_resource`、`i2c_com_port`、`i2c_address`、`bert_host/bert_port`、`resource_ids`、`storage` 路径 / paths |
| `V1.2/source/configs/ai_tuning/b0.yaml` | 新器件必须用 READ_ONLY 快照重新确定 B0，不能沿用 / Re-establish B0 from a READ_ONLY snapshot on a new device |
| `V2.1/*/source/configs/ai_tuning/runtime.v2_1.continuous.live.yaml`、`configs/hardware/vcsel_lab_v2_1_continuous.yaml` | 同上 / Same fields |
| `V2.1/desktop/source/src/vcsel_ai_tuning/desktop/app.py`（第 44–45 行 / lines 44–45） | 代码里写死的默认 VISA 和 COM 口 / Hard-coded default VISA and COM port |

- 资源 ID 里的芯片名不一致：V1.2 写 `PHVD1124`，V2.1 写 `VD1124`。请按实物确认。
  The chip name in resource IDs differs: `PHVD1124` in V1.2, `VD1124` in V2.1. Confirm against the hardware.
- 测试代码里也有 `Nate-Rog` 等字符串，但只作为测试数据，不会连接设备，无需修改。
  Tests contain strings such as `Nate-Rog` as fixtures only; they do not connect to devices.
- Kimi API 密钥只从环境变量 `MOONSHOT_API_KEY` 读取，不要写进文件。
  The Kimi key is read only from `MOONSHOT_API_KEY`; never write it to a file.

## 6. 已知问题 / Known issues

快照源码按原样冻结（见 [VERSIONS](VERSIONS.md)），以下问题在本次交付中只记录、不修改。
Snapshot sources are frozen as delivered; these issues are recorded here, not fixed.

| 问题 / Issue | 位置 / Where | 影响与处理 / Impact and workaround |
|---|---|---|
| `vcsel-tune` 命令指向不存在的 `vcsel_ai_tuning.cli` / points to missing module | 所有快照的 `pyproject.toml` / every snapshot | 安装后运行会报错；改用 `vcsel-ai-api` 或 `start-v1.2.cmd` / Errors when run; use `vcsel-ai-api` or the launcher |
| 审计文件把空 TDECQ 计为 0 条缺失 / audit counts null TDECQ as present | `audit/dataset_record_audit.json` | 以 [RESULTS](RESULTS.md) 2.4 节为准 / Use RESULTS section 2.4 |
| Pilot 观测 ID 跨 campaign 重复 / pilot observation IDs repeat across campaigns | V1.2 LIVE 执行器 / executor | 曾导致第一次 2h soak 失败 / Caused the first 2h soak failure |
| V2.1 LIVE 启动脚本要求原始 git 分支和提交 / launcher requires original branch and commit | `V2.1/continuous/source/scripts/run_v2_1_continuous_live.ps1` 第 248–272 行 | 在本仓库中无法直接运行，需在原分支或去掉校验后使用 / Not runnable from this repository as is |
| V2.0 的 AI Brain 没有接入运行流程，默认没有 provider / not wired, no default provider | `V2.0/source/src/vcsel_ai_tuning/v2/` | `BrainFactory().create()` 会报 `BrainUnavailableError` / raises |
| 过时的 README 和文档 / stale READMEs and docs | `V1.2/source/README.md`、`V2.0/source/README.md`（写着“处于第 1、2 层之间”，目录结构也不符）；`V2.1/*/source/docs/v2-1-continuous-eye-testbench.md`（写着未运行 LIVE）/ (claim the project is between layers 1–2; wrong layout; testbench says no LIVE run) | 以本页和 RESULTS 为准 / Use this page and RESULTS |
| 文档中的个人路径 / personal paths in docs | `C:\Users\NateS\...` 等 / etc. | 只是历史记录，不影响运行 / Historical only |
| 只有 Windows PowerShell 安装说明 / Windows-only instructions | `docs/USAGE.md` | Linux/macOS 需自行改写命令 / Adapt commands yourself |
| 前端带 `pnpm-lock.yaml`，说明却用 `npm install` / lock vs instructions mismatch | `*/source/frontend/` | 用 `pnpm install` 才能按锁定版本安装 / Use `pnpm install` for locked versions |
| 2 个被跳过的测试是路径逃逸安全测试 / the 2 skipped tests are path-escape security tests | `tests_py/ai_tuning/campaign/test_repository.py:71` | Windows 无符号链接权限；可在 Linux 或管理员权限下补跑 / Run on Linux or with admin rights |
| V2.1 没有统一复测记录 / no V2.1 retest recorded | `docs/VERIFICATION.md` | 使用前先在各分支内跑测试 / Run each variant's tests before use |

## 7. 与 `VCSEL` 仓库的关系 / Relation to the `VCSEL` repository

`V1.2/source/README.md` 中的六层架构，第 1–4 层（L-I-V 测试、数据分析、Verilog-A 模型、Driver+VCSEL 仿真）不在本仓库，相关模型和仿真器在另一个私有仓库 `Nate-Sun/VCSEL` 中。本仓库只实现第 5–6 层（AI 调参和硬件闭环）。
Layers 1–4 of the six-layer architecture in `V1.2/source/README.md` (L-I-V testing, analysis, Verilog-A models, Driver+VCSEL simulation) are not in this repository; related models and simulators are in the private repository `Nate-Sun/VCSEL`. This repository implements layers 5–6 (AI tuning and the hardware loop).
