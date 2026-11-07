# 实验结果总结 / Results summary

整理日期 / Prepared: 2026-11-06。本文只汇总仓库内已有的结构化数据，没有新增测量。数字可用 `python tools/analyze_v12_soak.py` 复现。
This page summarizes structured evidence already in the repository; no new measurements were taken. V1.2 numbers are reproducible with `python tools/analyze_v12_soak.py`.

TDECQ 越低越好，单位 dB。所有结果只适用于当时那一个样品、那一套工位配置。
Lower TDECQ is better (dB). All results apply only to the sample and bench setup used at the time.

## 1. 一句话结论 / Bottom line

- 原生 V1.2 能稳定、安全地跑完整个 PILOT_3 → FAST24 流程（100 个 campaign 连续运行，B0 每次都恢复成功）。
  Native V1.2 runs the full PILOT_3 → FAST24 flow safely and repeatably: 100 back-to-back campaigns, B0 restored every time.
- 可重复的改善约 **0.13–0.15 dB**（B0 → 最终复测）。这个改善在统计上是真实的，但只比单次测量噪声（约 0.1 dB）大一点。单次运行的 0.34 dB 不能当作典型值。
  The repeatable gain is about **0.13–0.15 dB** (B0 → final verification). It is statistically real but only slightly larger than single-measurement noise (~0.1 dB). The single-run 0.34 dB is not typical.
- FIR 两轮在 101 个 campaign 中**从未胜出**；所有 TDECQ 测量失败都出现在 `ibias_dac` 搜索范围的两端。
  The two FIR rounds **never won** in 101 campaigns; every TDECQ measurement failure occurred at the edges of the `ibias_dac` search range.
- V2.1（Kimi 驱动）250 轮最好 2.66 dB，**所有 campaign 都没达到目标**；其最后的脚本模式是演示用途，不是 AI 优化结果。
  V2.1 (Kimi-driven) reached 2.66 dB at best over 250 rounds; **no campaign met its target**. Its final scripted mode is a demo, not an AI-optimization result.

## 2. 原生 V1.2 / Native V1.2

### 2.1 调了什么 / What was tuned

| 项目 / Item | 内容 / Value |
|---|---|
| 链路 / Link | 53.125 GBd PAM4，850 nm，SSPRQ 码型，Keysight N1010A 测 TDECQ / 53.125 GBd PAM4, 850 nm, SSPRQ, TDECQ from Keysight N1010A |
| I2C 搜索 / I2C search | 只调 `ibias_dac`（60–80）和 `res_gain`（0–10）；其余 7 个寄存器固定在 B0 / Only `ibias_dac` (60–80) and `res_gain` (0–10); the other 7 registers stay at B0 |
| 范围来源 / Range origin | 前期扫描结论（扫描数据不在本仓库）/ From an earlier sweep (sweep data not in this repository) |
| FIR | 7 个 tap 由公式推导，NLC 6/6 与码型冻结 / 7 taps derived by formula; NLC 6/6 and pattern frozen |
| 每个 campaign / Per campaign | 3 PILOT + 1 B0 + 6 coarse + 16 fine + 1 FIR_FIRST + 1 FIR_ITERATIVE + 1 final verification = 29 条记录 / 29 records |
| 目标 / Objective | 只优化 TDECQ；ER、OMA、RLM 只记录 / TDECQ only; ER, OMA, RLM recorded for audit |
| B0 | `ibias_dac=70, res_gain=6`, FIR `[0,9,-28,104,-8,1,-3]`，见 `V1.2/source/configs/ai_tuning/b0.yaml` |

### 2.2 结果 / Results

| 运行 / Run | Campaign 数 | B0 平均 / B0 mean | 最终复测平均 / Final verification mean | 搜索最好平均（范围）/ Best-in-search mean (range) | B0 − 最终（均值 ± SD）/ B0 − final (mean ± SD) | 变好 / 不变 / 变差 / Better / same / worse |
|---|---|---|---|---|---|---|
| FAST24 单次 / single run | 1 | 2.72 | 2.38 | 2.44 | 0.34 | – |
| Soak 2h-r2 | 20 | 2.74 | 2.59 | 2.50 (2.34–2.70) | 0.15 ± 0.18 | 16 / 0 / 4 |
| Soak 8h | 80 | 2.81 | 2.68 | 2.62 (2.36–2.91) | 0.13 ± 0.16 | 62 / 1 / 17 |

- 单次运行：最好设置 `ibias_dac=72, res_gain=2`，第 20 轮 2.44 dB，最终复测 2.38 dB；全程约 6 分钟；`best_applied_to_hardware=false`，结束后硬件回到 B0。
  Single run: best setting `ibias_dac=72, res_gain=2`, 2.44 dB in round 20 and 2.38 dB at final verification; about 6 minutes; `best_applied_to_hardware=false`, hardware returned to B0.
- Soak 中所有胜出设置都在 `ibias_dac` 67–75、`res_gain` 1–7 之内，最常见的是 `70/3`、`71/3`、`72/3`。
  Every soak winner lies within `ibias_dac` 67–75 and `res_gain` 1–7; the most frequent are `70/3`, `71/3`, `72/3`.
- Soak 脚本（`V1.2/data/validated/AUDIT_AND_SUMMARIES/native_v12_soak.py`）在每个 campaign 后回读并核对 B0，失败即中止；因此 PASSED 表示 100 次 B0 恢复都通过核对。但该脚本也接受 dataset 为 `PARTIAL`，所以测量失败不会让 soak 失败。
  The soak script verifies B0 after every campaign and aborts on mismatch, so PASSED means all 100 restorations were verified. It also accepts `PARTIAL` datasets, so measurement failures do not fail a soak.
- 状态为 `COMPLETED_NO_IMPROVEMENT` 的 campaign：2h-r2 有 3 个，8h 有 17 个。
  `COMPLETED_NO_IMPROVEMENT` campaigns: 3 in 2h-r2, 17 in 8h.

### 2.3 测量噪声与可信度 / Measurement noise and confidence

| 指标 / Metric | 2h-r2 | 8h | 含义 / Meaning |
|---|---|---|---|
| 同一 campaign 内相同设置重复测量的标准差 / Within-campaign repeat SD | 0.09 dB | 0.11 dB | 单次 TDECQ 读数的短期噪声 / Short-term noise of one reading |
| 改善的配对 t 值 / Paired t of the gain | 3.7 | 7.5 | 改善不是偶然 / The gain is not chance |
| 搜索最好值比复测乐观多少 / Best-in-search minus final, mean | 0.09 dB | 0.06 dB | “最好值”偏乐观，应以最终复测为准 / Best-in-search is optimistic; trust final verification |
| B0 漂移（按时间四等分）/ B0 drift by quarter | 2.77 → 2.75 | 2.72 → 2.86 | 8 小时内 B0 自身变差约 0.14 dB / B0 itself worsened ~0.14 dB over 8 h |

解读 / Interpretation:

1. 平均改善 0.13–0.15 dB 约为单次读数噪声的 1.2–1.5 倍。按 campaign 看，约 78–80% 变好，20% 左右持平或变差。
   The mean gain is about 1.2–1.5× single-reading noise. Per campaign, about 78–80% improved and about 20% did not.
2. 单次 campaign 中的“最好值”是 24 个带噪声读数里的最小值，天然偏低。对外报告请用最终复测值，并尽量报告多次 campaign 的平均值。
   The per-campaign best is the minimum of 24 noisy readings and is biased low. Report final verification, ideally averaged across campaigns.
3. B0 会随时间漂移，所以“改善”要和同一 campaign 里的 B0 比，不能和几小时前的 B0 比。
   B0 drifts, so compare against the same campaign's B0, not one measured hours earlier.
4. 代码里的 `meaningful_tdecq_floor_db`（`V1.2/source/configs/hardware/vcsel_lab_live.yaml`）目前为 `null`。建议用第 5 节的重复性测试结果填入。
   `meaningful_tdecq_floor_db` in `V1.2/source/configs/hardware/vcsel_lab_live.yaml` is still `null`; fill it from the repeatability test in section 5.

### 2.4 测量失败在哪里 / Where measurements fail

“PARTIAL”记录不只是缺少同次采集证明，它们的 `measurement_status` 都是 `FAILED`、`metrics.tdecq` 为空（2h-r2 共 63 条，8h 共 284 条）。
“PARTIAL” records are not just missing same-acquisition proof: all have `measurement_status: FAILED` and `metrics.tdecq: null` (63 in 2h-r2, 284 in 8h).

8h 数据中，**全部 284 次失败都发生在 `ibias_dac` ≤ 64 或 ≥ 77**，而这些边缘值占了 I2C 搜索轮次的 44%（777/1760）。`ibias_dac=60` 失败 116/202，`=80` 失败 70/210；65–76 之间零失败。`res_gain` 的 0–1 和 9–10 也明显偏高。
In the 8h data, **all 284 failures occurred at `ibias_dac` ≤ 64 or ≥ 77**, yet those edge values took 44% (777/1760) of I2C search rounds. `ibias_dac=60` failed 116/202 and `=80` failed 70/210; 65–76 never failed. `res_gain` 0–1 and 9–10 are also clearly worse.

建议（需同事结合前期扫描确认）：把 FAST24 范围收窄到约 `ibias_dac` 64–76、`res_gain` 1–8。这样既覆盖全部胜出设置，又能把原先浪费在边缘的近一半轮次用回有效区域。
Suggestion (to be confirmed against the earlier sweep): narrow FAST24 to roughly `ibias_dac` 64–76 and `res_gain` 1–8. This still covers every winner and returns almost half the budget from the failing edges.

注意：`audit/dataset_record_audit.json` 中 `missing_tdecq_or_er_records: 0` 是按“字段是否存在”统计的，没有计入值为空的情况，与上述数据不一致。审计文件按原样保留，以本页为准。
Note: `missing_tdecq_or_er_records: 0` in `audit/dataset_record_audit.json` counts missing keys, not null values, and is inconsistent with the data above. The audit file is preserved as delivered; use this page.

### 2.5 FIR 阶段 / FIR stages

101 个 campaign 中，FIR_FIRST 和 FIR_ITERATIVE 从未产生最好结果。Soak 中“迭代 FIR”给出的抽头与“第一轮 FIR”完全相同（`[-2,7,-27,104,-4,0,-6]`），实际没有迭代。是否保留这两轮，或改进公式，需要重新评估。
Across 101 campaigns FIR_FIRST and FIR_ITERATIVE never produced the best result. In the soaks the iterative FIR taps equal the first-pass taps (`[-2,7,-27,104,-4,0,-6]`), so no iteration happens. Whether to keep these rounds or improve the formula needs review.

### 2.6 失败与中止记录 / Failed and aborted runs

这些记录都保留在 `V1.2/data/validated/RUNTIME_STATE/`。注意文件夹编号比任务编号小 1（例如 `staged-8` 里是 task 9；`staged-1` 里有 task 1 和 2）。
All are kept under `V1.2/data/validated/RUNTIME_STATE/`. Folder numbers are one lower than task numbers (e.g. `staged-8` holds task 9; `staged-1` holds tasks 1 and 2).

| 任务 / Task | 结果 / Result | 记录到的原因 / Recorded reason |
|---|---|---|
| staged task 1 | ABORTED | 开始前取消 / Cancelled before start |
| staged task 2–3 | FAILED | PILOT_3 安全门禁拒绝进入 FAST24：dataset 不完整 / PILOT_3 gate rejected FAST24: incomplete dataset |
| staged task 4–6 | FAILED | FAST24 失败，B0 已恢复；具体根因未单独记录 / FAST24 failed after verified B0 restore; root cause not recorded separately |
| staged task 3, 5, 6, 7 的部分观测 / some observations | – | `dataset persistence failed: [WinError 5] 拒绝访问` —— Windows 写文件被拒，是 dataset 不完整的直接原因之一 / Windows denied a file write; a direct cause of incomplete datasets |
| staged task 7 | COMPLETED | 最好 2.79 dB / Best 2.79 dB |
| staged task 8 | ABORTED | 开始前发现 BERT FIR 基线偏离 B0 / BERT FIR baseline drifted from B0 before start |
| staged task 9 | COMPLETED | 即上面的 FAST24 单次运行 / The single FAST24 run above |
| soak 2h（第一次 / first） | FAILED | 第 2 个 campaign 报 `IntegrityError: UNIQUE constraint failed: observations.observation_id`：pilot 观测 ID 跨 campaign 重复，是**软件 bug**，与硬件无关；随后 2h-r2 通过 / Second campaign hit a duplicate pilot observation ID — a **software bug**, not hardware; 2h-r2 then passed |

## 3. V2.1（Kimi 连续优化）/ V2.1 (Kimi continuous optimization)

V2.1 搜索更多寄存器，每个 campaign 的 B0 也不同，所以数字不能直接和 V1.2 比。
V2.1 searches more registers and uses a different B0 per campaign, so its numbers are not directly comparable with V1.2.

`V2.1/continuous/data/V2_1_250_ROUNDS_20260819/` 里的 250 轮报告合并了**三个分支**的四个 campaign：
The 250-round report in `V2.1/continuous/data/V2_1_250_ROUNDS_20260819/` combines four campaigns from **three variants**:

| # | Campaign | 分支 / Variant | 轮数 / Rounds | 目标 / Target | 基线 / Baseline | 最好 / Best | 最终 / Final |
|---|---|---|---|---|---|---|---|
| 1 | b468337d | continuous | 50 | 2.8 | 5.21 | 3.16 | 3.25 |
| 2 | 6ee6ae55 | r46-fast | 50 | 2.8 | 3.66 | 3.24 | 3.49 |
| 3 | 74272a5f | b0-70-22-fast | 50 | 2.8 | 3.32 | 2.87 | 3.10 |
| 4 | 63ad6c41 | b0-70-22-fast | 100 | 2.0 | 3.01 | 2.66 | 3.06 |

- 合计 250 轮，248 次有效，最好 2.66 dB（第 191 轮），中位数 3.145 dB，14 轮低于 2.8 dB。
  250 rounds, 248 valid, best 2.66 dB (round 191), median 3.145 dB, 14 rounds below 2.8 dB.
- 所有 V2.1 campaign 的 `state.json` 都是 `acceptance_passed: false`，即没有达到目标。
  Every V2.1 campaign's `state.json` has `acceptance_passed: false`: the target was not met.
- 同一设置重复测量最大相差 0.83 dB（来自 b0-70-22-fast 代码中的历史摘要），远大于 V1.2 的噪声水平，原因未查明。
  Repeat spread at identical settings reached 0.83 dB (from the history summary embedded in b0-70-22-fast), far above V1.2's noise level; cause not investigated.
- continuous 分支最后的 campaign `c22f12a7` 使用**脚本模式**：按 `configs/ai_tuning/v2_1_continuous.live.yaml` 中固定的 20 个点设置参数，并覆盖 Kimi 的建议（`kimi_provider.py` 中的 `SCRIPT_ENFORCED`）。这是演示用途，最好 3.06 dB，不应计为 AI 优化结果。
  The last continuous campaign `c22f12a7` uses **scripted mode**: it applies a fixed 20-point schedule from `configs/ai_tuning/v2_1_continuous.live.yaml` and overrides Kimi (`SCRIPT_ENFORCED` in `kimi_provider.py`). It was a demo (best 3.06 dB) and should not be counted as AI optimization.

## 4. 下一步建议 / Suggested next steps

1. 先做第 5 节的重复性测试，确定“多大的改善才算数”。
   Run the repeatability test in section 5 to define a meaningful-improvement threshold.
2. 结合前期扫描，评估收窄 `ibias_dac` / `res_gain` 范围（见 2.4）。
   Re-evaluate the `ibias_dac` / `res_gain` ranges with the earlier sweep (see 2.4).
3. 决定 FIR 两轮是否保留或改进（见 2.5）。
   Decide whether to keep or improve the two FIR rounds (see 2.5).
4. 若要继续 V2.1，先查明 0.83 dB 重复性偏差的来源；否则 AI 无法区分真实改善和噪声。
   Before continuing V2.1, find the source of the 0.83 dB repeat spread; otherwise the AI cannot separate real gains from noise.
5. 修复 pilot 观测 ID 跨 campaign 重复的 bug，并确认 artifact 目录在 Windows 上的写权限（杀毒软件或文件占用都可能导致 WinError 5）。
   Fix the cross-campaign pilot observation ID bug and confirm Windows write access for artifact folders (antivirus or file locks can cause WinError 5).

## 5. 重复性测试步骤（给接手同事）/ Repeatability test (for the colleague taking over)

目的：测出 TDECQ 单次读数的噪声，作为判断改善是否有效的门槛。
Purpose: measure single-reading TDECQ noise and use it as the threshold for a real improvement.

1. 按 `docs/USAGE.md` 和原 SOP 完成 READ_ONLY preflight，确认仪器身份、通道映射和 B0 回读一致。
   Complete READ_ONLY preflight per `docs/USAGE.md` and the original SOP; confirm instrument identity, channel mapping and B0 readback.
2. **不改任何设置**，在 B0 上连续读取 TDECQ 30 次，每次间隔与 FAST24 相同（约 10 秒），记录时间、TDECQ、ER。READ_ONLY 模式允许测量，不会写硬件。
   **Without changing anything**, read TDECQ at B0 30 times at FAST24's pace (~10 s apart); log time, TDECQ and ER. READ_ONLY permits measurement without writes.
3. （需 LIVE 授权）在 `ibias_dac/res_gain` = 70/6、71/3、72/2 三个点之间轮流切换，每点 10 次，每次切换后等待相同的 settle 时间。这可以区分“读数噪声”和“切换后恢复不完全”。
   (Requires LIVE authorization) Alternate between 70/6, 71/3 and 72/2, 10 readings each, with the usual settle time. This separates reading noise from imperfect settling after a change.
4. 计算每个点的平均值和标准差。建议把 `meaningful_tdecq_floor_db` 设为约 2 倍标准差，并在本页记录日期、样品编号和结果。
   Compute mean and SD per point. Set `meaningful_tdecq_floor_db` to about 2× SD and record date, sample ID and results on this page.
5. 每 30 分钟复测一次 B0，持续 2 小时，确认漂移幅度。
   Re-measure B0 every 30 minutes for 2 hours to quantify drift.
