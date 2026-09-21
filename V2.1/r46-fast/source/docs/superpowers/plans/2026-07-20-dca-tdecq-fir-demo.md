# DCA TDECQ -> TX FIR Demo 实施计划

> **中文优先 / Chinese first.** 本计划用于直接实现一个可运行、默认只读硬件的 Demo。
>
> **Execution note:** Use `subagent-driven-development`, `test-driven-development`, and `verification-before-completion`. Do not send any BERT command in this milestone.

**目标 / Goal:** 从离线样例或 FlexDCA 当前已有测量结果读取 TDECQ 与均衡器 taps，计算 7-tap TX FIR，检查 `sum(abs(taps)) <= 170`，并生成 VIT BERT `setFir` 命令预览和 JSON 记录。

**架构 / Architecture:** 新增独立 `hardware` 子包，不修改现有通用优化框架。计算模块是纯函数；DCA 模块只依赖一个带 `query()` 的会话接口；CLI 将离线样例和真实 VISA 读取统一成同一个 `DcaSnapshot`。BERT 部分本阶段只格式化命令，不创建 UDP socket、不发送数据。

**技术栈 / Stack:** Python 3.13、标准库、可选 PyVISA、pytest。

---

## Task 1：FIR 计算模型与首轮算法 / FIR models and first-pass algorithm

**Files:**

- Create: `src/vcsel_ai_tuning/hardware/__init__.py`
- Create: `src/vcsel_ai_tuning/hardware/fir.py`
- Create: `tests_py/ai_tuning/hardware/test_fir.py`

**Step 1 — 写失败测试 / Write failing tests**

覆盖：

```python
def test_round_half_away_from_zero(): ...
def test_first_pass_matches_workbook_example(): ...
def test_first_pass_rejects_more_than_170_without_scaling(): ...
def test_first_pass_rejects_missing_three_side_taps(): ...
```

首轮样例固定为：

```python
dca = [0.008756, 0.079768, -0.291888, 1.336349,
       -0.084736, 0.02357, -0.016326, -0.02607, -0.029423]
precursors = 3
manual_main = 90
expected = [1, 5, -20, 90, -6, 2, -1]
```

**Step 2 — 确认失败 / Verify RED**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/hardware/test_fir.py -q
```

Expected: collection fails because `vcsel_ai_tuning.hardware.fir` does not exist.

**Step 3 — 最小实现 / Implement GREEN**

在 `fir.py` 实现：

```python
@dataclass(frozen=True)
class FirProposal: ...

def round_half_away_from_zero(value: float) -> int: ...
def calculate_first_pass(dca_taps, dca_main_index, manual_main) -> FirProposal: ...
```

算法：以 `dca_main_index` 为中心裁 3+1+3；除以 DCA main；乘以人工 Main；远离零的五入；最后强制 Main 等于人工值。只报告超限，不自动缩放。

**Step 4 — 验证 GREEN / Verify GREEN**

Run the same test command. Expected: all Task 1 tests pass.

---

## Task 2：迭代算法与 BERT 命令映射 / Iterative algorithm and BERT preview

**Files:**

- Modify: `src/vcsel_ai_tuning/hardware/fir.py`
- Modify: `tests_py/ai_tuning/hardware/test_fir.py`

**Step 1 — 写失败测试 / Write failing tests**

覆盖：

```python
def test_iterative_pass_convolves_then_crops_seven_taps(): ...
def test_setfir_preview_uses_vendor_order_and_preserves_nlc(): ...
def test_discarded_l1_ratio_is_reported(): ...
```

迭代样例：

```python
current = [0, 6, -28, 112, -8, -10, 0]
dca = [0, 0, 0.0012, 1.188325, 0.009687, 0.011, -0.028617, -0.04]
manual_main = 104
expected = [0, 6, -26, 104, -7, -8, -2]
```

内部顺序固定为 `[Pre3, Pre2, Pre1, Main, Post1, Post2, Post3]`；命令顺序固定为 `Pre1 Pre2 Pre3 Main Post1 Post2 Post3 upperNlc lowerNlc`。

**Step 2 — 确认失败 / Verify RED**

Run targeted tests; expected failure for missing iterative and formatter functions.

**Step 3 — 最小实现 / Implement GREEN**

实现纯 Python 全卷积、中心索引 `3 + dca_main_index`、中心 7 taps 裁剪、重新按人工 Main 归一化和取整，以及：

```python
def format_setfir_preview(port, channel, taps, upper_nlc, lower_nlc) -> str: ...
```

**Step 4 — 验证 GREEN / Verify GREEN**

Run targeted tests; expected all FIR tests pass.

---

## Task 3：FlexDCA 只读读取器 / Read-only FlexDCA reader

**Files:**

- Create: `src/vcsel_ai_tuning/hardware/flexdca.py`
- Create: `tests_py/ai_tuning/hardware/test_flexdca.py`

**Step 1 — 写失败测试 / Write failing tests**

用 `FakeQuerySession` 精确验证只发以下查询：

```text
*IDN?
:MEASure:EYE:TDEQ?
:MEASure:EYE:TDEQ:STATus?
:MEASure:EYE:TDEQ:STATus:DETails?
:SPRocess1:TEQualizer:TAPS?
:SPRocess1:TEQualizer:NPRecursors?
```

覆盖正确解析、非 `CORR` 状态拒绝、无任何 `write()` 调用。

**Step 2 — 确认失败 / Verify RED**

Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/hardware/test_flexdca.py -q
```

Expected: module import failure.

**Step 3 — 最小实现 / Implement GREEN**

实现冻结数据类 `DcaSnapshot`、协议 `QuerySession`、`read_existing_tdecq(session, function=1)`，以及可选的 `open_visa_session(resource)`。PyVISA 仅在实机模式中动态导入，离线模式不依赖它。

**Step 4 — 验证 GREEN / Verify GREEN**

Run targeted tests; expected all reader tests pass.

---

## Task 4：可运行 CLI 与 JSON 记录 / Runnable CLI and JSON artifact

**Files:**

- Create: `src/vcsel_ai_tuning/hardware/demo_cli.py`
- Create: `scripts/run_fir_demo.py`
- Create: `tests_py/ai_tuning/hardware/test_demo_cli.py`
- Create: `configs/fir_demo_first.json`
- Create: `configs/fir_demo_iterative.json`
- Modify: `pyproject.toml`

**Step 1 — 写失败测试 / Write failing tests**

覆盖：

```python
def test_fixture_first_writes_proposal_json(tmp_path): ...
def test_fixture_iterative_writes_expected_taps(tmp_path): ...
def test_invalid_budget_returns_exit_2_but_keeps_artifact(tmp_path): ...
def test_cli_has_no_apply_option(): ...
```

**Step 2 — 确认失败 / Verify RED**

Run targeted CLI tests; expected failure because CLI is absent.

**Step 3 — 最小实现 / Implement GREEN**

CLI interface:

```text
run_fir_demo.py fixture --input CONFIG.json --output DIR
run_fir_demo.py flexdca --visa-resource RESOURCE --mode first|iterative \
  --main N [--current-fir CSV] [--function 1] [--port 1] [--channel 0] --output DIR
```

输出：终端中文摘要 + `<output>/fir_proposal.json`。CLI 不提供 `--apply`，因此 Demo 不可能误写 BERT。

在 `pyproject.toml` 增加 `vcsel-fir-demo` 入口，以及可选 `hardware = ["pyvisa>=1.15"]`。

**Step 4 — 验证 GREEN / Verify GREEN**

Run targeted tests and both fixture commands. Expected output taps match workbook examples.

---

## Task 5：整体验证与安全审查 / Full verification and safety review

**Files:**

- Modify: `README.md` only if an existing tracked README section is appropriate; otherwise create `docs/fir-demo-quickstart.md`.

**Step 1 — 运行全量测试 / Run full tests**

```powershell
py -3.13 -m pytest tests_py/ai_tuning -q
```

Expected: baseline 35 tests plus all new tests pass.

**Step 2 — 实际运行两个离线 Demo / Run both fixture demos**

```powershell
py -3.13 scripts/run_fir_demo.py fixture --input configs/fir_demo_first.json --output artifacts/fir_demo_first
py -3.13 scripts/run_fir_demo.py fixture --input configs/fir_demo_iterative.json --output artifacts/fir_demo_iterative
```

Expected:

- First: `[1, 5, -20, 90, -6, 2, -1]`, L1 = 125, valid.
- Iterative: `[0, 6, -26, 104, -7, -8, -2]`, L1 = 153, valid.
- Both print a `setFir` preview and write JSON.

**Step 3 — 安全审查 / Safety audit**

Search production code for `send`, `sendto`, `write(` and confirm this Demo contains only file writes and VISA `query()` calls. Confirm there is no BERT network transmission path.

**Step 4 — 最终复核 / Final review**

Review against `docs/superpowers/specs/2026-07-20-dca-tdecq-to-bert-fir-design.md`. Report runnable commands and explicitly identify the remaining real-hardware inputs: VISA resource string and verified BERT readback/API schema.
