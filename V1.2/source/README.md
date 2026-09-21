# PhotonIC VCSEL AI Test & Tuning Platform / PhotonIC VCSEL AI 测试与调参平台

本项目把两个方向合并成一个统一平台：

V1.2 can be started as one FastAPI-hosted frontend/backend application with
[`start-v1.2.ps1`](start-v1.2.ps1). See the
[`V1.2 startup guide`](docs/v1.2-startup-guide.md) for FAKE, READ_ONLY, and
LIVE_TUNING usage and safety boundaries.

This project merges two directions into one unified platform:

1. **VCSEL 器件测试、数据分析、Verilog-A 模型校准和 Driver+VCSEL 仿真。**  
   **VCSEL device testing, data analysis, Verilog-A model calibration, and Driver+VCSEL simulation.**
2. **AI 辅助调参、眼图优化、安全门禁和实验室硬件闭环。**  
   **AI-assisted tuning, eye optimization, safety gating, and laboratory hardware closed loop.**

最简单地说，本项目要做的是：

In simple terms, this project aims to:

```text
先把 VCSEL 测清楚
-> 用数据校准模型
-> 用模型做 Driver+VCSEL 仿真
-> 用 AI 推荐更好的工作点
-> 最后接入真实硬件自动优化眼图
```

```text
Measure the VCSEL first
-> Calibrate the model with data
-> Run Driver+VCSEL simulation
-> Use AI to recommend better operating points
-> Finally connect real hardware for automatic eye optimization
```

## Documentation Rule / 文档规则

所有 Markdown 文档必须使用中英文双语。新增或更新 `.md` 文件时，应尽量保持中文和英文信息等价。

All Markdown documents must be bilingual in Chinese and English. When adding or updating `.md` files, keep the Chinese and English content as equivalent as practical.

## Current Position / 当前所处阶段

当前项目处在 **第 1 层和第 2 层之间**：准备做 PCB 级单 VCSEL DC L-I-V 测试，同时利用已有 Excel 数据先搭建分析、参数提取和 AI tuning 雏形。

The project is currently between **Layer 1 and Layer 2**: preparing for PCB-level single-VCSEL DC L-I-V testing while using existing Excel data to build the first analysis, parameter-extraction, and AI-tuning prototype.

目前不要急着做完整硬件闭环。正确顺序是先把数据路径跑通，再逐步进入模型、仿真、AI 和硬件自动化。

Do not rush into the full hardware loop yet. The right order is to make the data path work first, then gradually move into modeling, simulation, AI, and hardware automation.

## Unified Architecture / 融合后的统一架构

项目分为六层。前一层的输出，就是后一层的输入。

The project has six layers. The output of each layer becomes the input to the next layer.

| Layer / 层级 | Main Work / 主要工作 | Completion Target / 完成标准 | Output To Next Layer / 给下一层的输出 |
| --- | --- | --- | --- |
| 1. VCSEL 测试层 / VCSEL Testing | 单 VCSEL DC L-I-V 测试，后续扩展 RIN、S21、眼图 / Single-VCSEL DC L-I-V testing, later RIN, S21, and eye measurements | 能稳定采集数据，测试条件清楚 / Stable data capture with clear test conditions | 原始测量数据、温度、bias、样品编号 / Raw data, temperature, bias, device ID |
| 2. 数据分析层 / Data Analysis | 读取 Excel/CSV，绘图，提取 Ith、slope efficiency、Vop、Pop / Load Excel/CSV, plot curves, extract Ith, slope efficiency, Vop, Pop | 能自动生成图和参数表 / Automatically generate plots and parameter tables | 一阶参数表、数据质量判断 / First-order parameter table and data-quality checks |
| 3. 模型校准层 / Model Calibration | 将测量参数映射到 Verilog-A/rate-equation 参数 / Map measured parameters to Verilog-A or rate-equation parameters | 模型能复现实测 L-I-V 趋势 / Model reproduces measured L-I-V trends | 校准模型、参数 JSON/CSV、误差报告 / Calibrated model, parameter JSON/CSV, error report |
| 4. Driver+VCSEL 仿真层 / Driver+VCSEL Simulation | 联合 Driver 和 VCSEL，扫描 bias、swing、带宽和裕量 / Co-simulate Driver and VCSEL, sweep bias, swing, bandwidth, and margin | 能比较不同工作点好坏 / Can compare operating points | 候选工作点、仿真结果、约束条件 / Candidate operating points, simulation results, constraints |
| 5. AI Tuning 层 / AI Tuning | 用 random search 或 Bayesian optimization 推荐下一步参数 / Use random search or Bayesian optimization to suggest next parameters | 最小优化闭环可运行，可解释 / Minimal optimization loop runs and is explainable | 推荐 bias、swing、Driver 设置、下一次测试点 / Recommended bias, swing, Driver settings, next test point |
| 6. 硬件闭环层 / Hardware Closed Loop | 写入真实硬件，DCA/FlexDCA 测眼图，安全检查，回滚，报告 / Write hardware settings, measure eye with DCA/FlexDCA, safety checks, rollback, reports | 能安全自动优化眼图并记录历史 / Safely optimize eye quality and record history | 最佳配置、最佳眼图、完整实验记录、新训练数据 / Best config, best eye, experiment log, new training data |

## End-To-End Workflow / 端到端工作流

离线建模路径：

Offline modeling path:

```text
Existing Excel data / PCB measurement
-> L-I-V analysis
-> parameter extraction
-> Verilog-A model calibration
-> Driver+VCSEL simulation
-> AI tuning prototype
```

```text
已有 Excel 数据 / PCB 实测数据
-> L-I-V 分析
-> 参数提取
-> Verilog-A 模型校准
-> Driver+VCSEL 联合仿真
-> AI tuning 原型
```

未来在线硬件闭环路径：

Future online hardware-loop path:

```text
AI suggests configuration
-> safety gate
-> write bias / swing / EQ / bandwidth
-> wait for link settling
-> DCA/FlexDCA measures eye metrics
-> score result
-> record history
-> suggest next configuration
```

```text
AI 推荐配置
-> 安全门禁
-> 写入 bias / swing / EQ / bandwidth
-> 等待链路稳定
-> DCA/FlexDCA 测眼图指标
-> 计算评分
-> 记录历史
-> 推荐下一组配置
```

## AI Tuning Strategy / AI 调参策略

本项目的 AI tuning 不应从复杂神经网络开始。第一版应使用小数据友好的优化方法。

AI tuning in this project should not start with a complex neural network. The first version should use small-data-friendly optimization methods.

推荐顺序：

Recommended order:

1. 手动或半自动记录几轮测试结果。  
   Manually or semi-automatically record several test iterations.
2. 使用 grid search 或 random search 跑通最小闭环。  
   Use grid search or random search to run the minimal loop.
3. 数据增加后使用 Bayesian optimization 减少无效实验次数。  
   Use Bayesian optimization after more data is available to reduce wasted experiments.
4. Agent/LLM 只做外层编排、解释、报告和异常提醒，不直接裸写硬件。  
   Agent/LLM should only orchestrate, explain, report, and flag exceptions; it should not directly write hardware without deterministic tools and safety checks.

硬件眼图优化的目标函数可以从以下指标开始：

The hardware eye-optimization objective can start from these metrics:

- TDECQ / TDEQ 越低越好。  
  Lower TDECQ / TDEQ is better.
- eye height 越大越好。  
  Larger eye height is better.
- eye width 越大越好。  
  Larger eye width is better.
- jitter 越小越好。  
  Lower jitter is better.
- crossing balance 越接近目标越好。  
  Crossing balance should be close to target.
- 温度、电流、电压和配置变化幅度必须受安全约束限制。  
  Temperature, current, voltage, and configuration step size must be limited by safety constraints.

## Repository Structure / 仓库结构

```text
.
+-- docs/
+-- models/
+-- data/
+-- scripts/
+-- notebooks/
+-- results/
+-- tests/
+-- configs/
+-- state/
+-- artifacts/
```

### `docs/`

项目文档、测试计划、模型说明、器件 datasheet、架构图、周报、反馈记录和知识树。

Project documents, test plans, model notes, device datasheets, architecture diagrams, weekly reports, feedback logs, and knowledge trees.

### `models/`

VCSEL、Driver、Verilog-A、rate-equation 和 surrogate model 文件。

VCSEL, Driver, Verilog-A, rate-equation, and surrogate model files.

### `data/`

原始数据、处理后数据、样例 Excel/CSV 和测试元数据。

Raw data, processed data, sample Excel/CSV files, and measurement metadata.

### `scripts/`

确定性工具层：数据读取、L-I-V 分析、参数提取、模型拟合、仿真结果解析、AI 优化和未来硬件控制入口。

Deterministic tool layer: data loading, L-I-V analysis, parameter extraction, model fitting, simulation-result parsing, AI optimization, and future hardware-control entry points.

### `notebooks/`

探索性分析、拟合实验和 AI tuning demo。

Exploratory analysis, fitting experiments, and AI-tuning demos.

### `results/`

图表、参数表、拟合报告、仿真对比结果和优化日志。

Plots, parameter tables, fit reports, simulation comparisons, and optimization logs.

### `tests/`

数据解析、参数提取、拟合工具和优化循环的测试。

Tests for data parsing, parameter extraction, fitting utilities, and optimization loops.

### `configs/`

测试配置、仪器配置、优化参数范围、安全限制和目标函数权重。

Test configs, instrument configs, optimization parameter ranges, safety limits, and objective-function weights.

### `state/`

当前数据集、当前参数、当前模型版本、当前推荐配置和优化历史。

Current dataset, current parameters, current model version, current recommended configuration, and optimization history.

### `artifacts/`

自动生成的实验报告、最佳配置摘要、架构图导出文件和对外展示材料。

Generated experiment reports, best-configuration summaries, exported architecture diagrams, and presentation artifacts.

## Immediate Next Step / 你下一步应该做什么

下一步不要直接做完整 AI 或硬件闭环。最应该做的是建立 **最小数据闭环**。

The next step should not be the full AI or hardware loop. The most important next step is to build the **minimal data loop**.

建议按这个顺序执行：

Recommended order:

1. 把两份已有 VCSEL Excel 文件复制或登记到 `data/raw/`。  
   Copy or register the two existing VCSEL Excel files under `data/raw/`.
2. 定义统一 L-I-V 数据格式，例如 `current_ma`, `voltage_v`, `optical_power_mw`, `temperature_c`, `device_id`。  
   Define a standard L-I-V schema such as `current_ma`, `voltage_v`, `optical_power_mw`, `temperature_c`, and `device_id`.
3. 编写第一版 `scripts/analyze_liv.py`，能读取样例数据并输出曲线和参数表。  
   Build the first `scripts/analyze_liv.py` that reads sample data and outputs plots plus a parameter table.
4. 输出 `results/liv_summary.csv`，包含 Ith、slope efficiency、Vop、Pop。  
   Output `results/liv_summary.csv` with Ith, slope efficiency, Vop, and Pop.
5. 再写一个最小 `scripts/tune_operating_point.py`，先用 random search 或简单评分推荐 bias/swing。  
   Then build a minimal `scripts/tune_operating_point.py` that first uses random search or a simple score to recommend bias/swing.

完成这五步后，项目就从“概念架构”进入“可运行路径”。

After these five steps, the project moves from a concept architecture into a runnable path.

## Short-Term Deliverables / 短期交付物

短期目标是证明路径可运行，而不是证明 AI 已经最优。

The short-term goal is to prove that the path runs, not that the AI is already optimal.

优先交付物：

Priority deliverables:

- `data/raw/` 中有可追踪的样例数据或实测数据。  
  Trackable sample or measured data exists under `data/raw/`.
- `data/processed/` 中有标准化 L-I-V CSV。  
  Standardized L-I-V CSV exists under `data/processed/`.
- `results/` 中有 L-I-V 曲线和参数摘要。  
  L-I-V plots and parameter summaries exist under `results/`.
- `state/` 中有当前数据集和当前参数状态。  
  Current dataset and current parameter state exist under `state/`.
- 第一版 AI tuning demo 能推荐下一组 bias/swing。  
  The first AI-tuning demo can recommend the next bias/swing pair.

## Final Product Form / 最终产品形态

最终产品不是单个脚本，而是一个分层平台。

The final product is not a single script, but a layered platform.

可能形态：

Possible forms:

1. 研究型工具包：脚本、notebook、参数表、报告。  
   Research toolkit: scripts, notebooks, parameter tables, and reports.
2. 工程型 CLI：`vcsel-ai analyze`、`vcsel-ai fit`、`vcsel-ai tune`、`vcsel-ai report`。  
   Engineering CLI: `vcsel-ai analyze`, `vcsel-ai fit`, `vcsel-ai tune`, and `vcsel-ai report`.
3. 展示型 dashboard：上传数据、显示曲线、推荐工作点、导出报告。  
   Dashboard: upload data, show curves, recommend operating points, and export reports.
4. 实验室闭环平台：仪器控制、安全门禁、DCA 眼图测量、Bayesian optimization、最佳配置报告。  
   Laboratory closed-loop platform: instrument control, safety gating, DCA eye measurement, Bayesian optimization, and best-configuration reporting.

## VCSEL AI Tuning Backend / VCSEL AI 调优后端

### Windows 安装 / Windows installation

需要 Python 3.12 或更高版本。请在全新的 PowerShell 会话中创建虚拟环境，并安装开发与优化器依赖：

Python 3.12 or later is required. Create a fresh virtual environment in PowerShell and install the development and optimizer dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,optimizer]"
```

只有需要连接真实仪器时，才单独安装硬件依赖：

Install the hardware dependencies separately only when real instruments are needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hardware]"
```

### Fake 与 replay 测试 / Fake and replay tests

以下测试只使用 fake 或离线 replay 数据，不连接 DCA、I2C 或 BERT：

The following tests use only fakes or offline replay data and do not connect to the DCA, I2C, or BERT:

```powershell
.\.venv\Scripts\python.exe -m pytest tests_py/ai_tuning/test_backends.py -q
.\.venv\Scripts\python.exe -m pytest tests_py/ai_tuning/optimization/test_replay.py -q
.\.venv\Scripts\python.exe -m pytest tests_py/ai_tuning -q
```

### API 运行模式与启动 / API runtime modes and startup

不设置任何环境变量时，API 使用内置确定性 `FAKE` runtime，可供前端开发、CI 和本地 HTTP 测试使用。该模式不会构造 VISA、串口或 BERT UDP 连接：

With no environment variables, the API uses the built-in deterministic `FAKE` runtime for frontend development, CI, and local HTTP tests. This mode does not construct VISA, serial, or BERT UDP connections:

```powershell
.\.venv-ai-tuning\Scripts\vcsel-ai-api.exe
```

也可以显式使用仓库中的严格示例配置：

You can explicitly use the repository's strict example configuration:

```powershell
$env:VCSEL_AI_RUNTIME_CONFIG = (Resolve-Path .\configs\ai_tuning\runtime.example.yaml)
.\.venv-ai-tuning\Scripts\vcsel-ai-api.exe
```

支持的显式模式为 `FAKE`、`READ_ONLY` 和 `LIVE_TUNING`。`READ_ONLY` 允许只读 identify、health、snapshot、readback 和测量，但所有 approve、B0 restore、I²C/FIR apply 与 rollback 写入均返回 `READ_ONLY_MODE`。`LIVE_TUNING` 仍要求 `PER_STAGE` 授权，`FULL_CAMPAIGN` 保持禁用。两个硬件模式都要求完整端点配置和显式 hardware adapter factory；系统不会猜测 COM、VISA 或 UDP 端点。

The explicit modes are `FAKE`, `READ_ONLY`, and `LIVE_TUNING`. `READ_ONLY` permits read-only identify, health, snapshot, readback, and measurement operations, while approve, B0 restore, I²C/FIR apply, and rollback writes return `READ_ONLY_MODE`. `LIVE_TUNING` still requires `PER_STAGE` authorization, and `FULL_CAMPAIGN` remains disabled. Both hardware modes require complete endpoint configuration, an explicit hardware adapter factory, and a human-approved tuning-profile factory; the built-in FAKE bounds are never reused. COM, VISA, and UDP endpoints are never guessed.

自定义运行层或真实硬件组装继续通过 `VCSEL_AI_CAMPAIGN_SERVICE_FACTORY=module:callable` 注入；该 callable 不接收参数，并返回 `ApiRuntime(service, event_source, status_provider, shutdown)`。普通模块 import 和 API 启动本身不连接硬件：

Custom runtimes and real-hardware assembly remain injectable through `VCSEL_AI_CAMPAIGN_SERVICE_FACTORY=module:callable`; the callable takes no arguments and returns `ApiRuntime(service, event_source, status_provider, shutdown)`. Ordinary module import and API startup do not connect hardware:

```python
from vcsel_ai_tuning.api import ApiRuntime


def create_runtime() -> ApiRuntime:
    campaign_runtime = build_campaign_runtime()
    service = build_campaign_service(campaign_runtime)
    event_source = campaign_runtime.store
    return ApiRuntime(
        service,
        event_source,
        status_provider=campaign_runtime,
        shutdown=campaign_runtime.close,
    )
```

例如，如果上述函数位于 `my_vcsel_runtime.py`：

For example, if that function is in `my_vcsel_runtime.py`:

```powershell
$env:VCSEL_AI_CAMPAIGN_SERVICE_FACTORY = "my_vcsel_runtime:create_runtime"
.\.venv\Scripts\vcsel-ai-api.exe
```

默认监听 `http://127.0.0.1:8000`。运行状态接口与主要 Campaign 接口包括：

The default address is `http://127.0.0.1:8000`. Runtime-status and key campaign endpoints include:

- `GET /health`：安全的 Store/Repository 健康摘要。 / Safe Store/Repository health summary.
- `GET /capabilities`：运行模式与 fail-closed 能力报告。 / Runtime mode and fail-closed capability report.
- `GET /api/campaigns?limit=50&offset=0`：分页 Campaign 摘要。 / Paginated campaign summaries.
- `POST /api/campaigns`：创建 Campaign。 / Create a campaign.
- `POST /api/campaigns/{campaign_id}/preflight`：执行只读预检。 / Run the read-only preflight.
- `POST /api/campaigns/{campaign_id}/stages/plan` 与 `/stages/approve`：规划并逐阶段授权。 / Plan and authorize one stage at a time.
- `POST /api/campaigns/{campaign_id}/pause|resume|abort|rollback`：控制 Campaign。 / Control a campaign.
- `GET /api/campaigns/{campaign_id}/observations|best|report`：读取观测、最佳结果和报告。 / Read observations, the best result, and the report.
- `GET /api/campaigns/{campaign_id}/events/stream`：有序 SSE 事件流。 / Ordered SSE event stream.

SSE 重连时使用规范的非负整数 `Last-Event-ID`，服务端只重放该序号之后的事件：

For SSE reconnection, send a canonical nonnegative integer `Last-Event-ID`; the server replays only later events:

```powershell
curl.exe -N -H "Last-Event-ID: 4" http://127.0.0.1:8000/api/campaigns/campaign-1/events/stream
```

### 安全边界 / Safety boundary

测试和默认 `vcsel-ai-api` 启动入口不会构造或连接真实硬件；只有显式配置的 runtime factory 能提供真实依赖。在 V1 中，仅启用 `PER_STAGE` 人工授权，`FULL_CAMPAIGN` 会被拒绝。完整 `PATH_HIERARCHICAL` optimizer 和生产 FIR proposer 未配置时会在 `/capabilities` 中报告 `false` 并拒绝对应阶段，不会静默回退为 TDECQ-only。

Tests and the default `vcsel-ai-api` startup entry point do not construct or connect to real hardware; only an explicitly configured runtime factory can provide real dependencies. V1 enables operator authorization only in `PER_STAGE` mode and rejects `FULL_CAMPAIGN`. When the complete `PATH_HIERARCHICAL` optimizer or production FIR proposer is not configured, `/capabilities` reports `false` and the corresponding stage is rejected rather than silently falling back to TDECQ-only behavior.

进入真实闭环前，仍必须完成以下现场前置条件：

Before entering a live closed loop, all of these on-site prerequisites remain mandatory:

- 完成 DCA、I2C 与 BERT 的基础配置。 / Complete the DCA, I2C, and BERT base configuration.
- 确保 `COM7`、VISA 会话和 BERT UDP 端口由本 Campaign 独占。 / Ensure exclusive ownership of `COM7`, the VISA session, and the BERT UDP endpoint.
- 先通过只读 preflight，确认身份、配置与健康状态。 / Pass the read-only preflight for identity, configuration, and health.
- 每个写入阶段都取得操作员 `PER_STAGE` 授权。 / Obtain operator `PER_STAGE` authorization for every mutating stage.
- 在首次写入前确认可回读、可验证的 rollback snapshot。 / Confirm a readable and verified rollback snapshot before the first write.
