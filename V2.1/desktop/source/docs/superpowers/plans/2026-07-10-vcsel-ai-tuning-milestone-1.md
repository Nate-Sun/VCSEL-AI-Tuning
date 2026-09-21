# VCSEL AI 调参里程碑 1 实施计划 / VCSEL AI Tuning Milestone 1 Implementation Plan

> **面向智能体执行者：** 必需的子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐项任务实施本计划。各步骤使用 checkbox（`- [ ]`）语法跟踪进度。
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 构建一个经过测试、具备硬件安全性的基础平台，可运行确定性的 fake/replay VCSEL 调参活动、测量 PAM4 倾斜代理指标、持久化完整溯源信息，并为未来 Cadence 和仪器后端提供 dry-run 契约。
**Goal:** Build a tested, hardware-safe foundation that runs deterministic fake/replay VCSEL tuning campaigns, measures PAM4 skew proxies, persists complete provenance, and exposes dry-run contracts for future Cadence and instrument backends.

**架构：** 一个 Python 包统一管理不可变领域模型、经过校验的 YAML 配置、确定性安全策略、显式实验状态机、指标提取、后端契约、候选点生成、原子化制品、SQLite 元数据、CLI 命令和 Markdown 报告。fake 和 replay 后端可立即执行；Cadence 和真实 SCPI/VISA 实现仅保留接口，并且在里程碑 1 中不能写入硬件。
**Architecture:** A Python package owns immutable domain models, validated YAML configuration, deterministic safety policy, an explicit experiment state machine, metric extraction, backend contracts, candidate generation, atomic artifacts, SQLite metadata, CLI commands, and Markdown reports. Fake and replay backends execute now; Cadence and real SCPI/VISA implementations remain interface-only and cannot write hardware in Milestone 1.

**技术栈：** 采用 Python 3.12+ 包语义，通过可用的 `py -3.13` 启动器执行；使用 Pydantic 2、PyYAML、NumPy、SciPy `qmc.Sobol`、SQLite、`argparse`、pytest，以及 Windows 上的 PowerShell 命令。
**Tech Stack:** Python 3.12+ package semantics executed through the available `py -3.13` launcher, Pydantic 2, PyYAML, NumPy, SciPy `qmc.Sobol`, SQLite, `argparse`, pytest, and PowerShell commands on Windows.

---

## 权威设计与范围 / Authoritative Design and Scope

实施前请阅读：
Read before implementation:

- `docs/superpowers/specs/2026-07-10-vcsel-ai-tuning-platform-design.md`
- `models/vcsel_integrated_physical_100g.va`
- `models/pam4_voltage_source.va`
- `scripts/run_liv_interactive.py`
- `scripts/run_s11_smu_bias.py`

里程碑 1 包括共享内核、fake/replay 执行、倾斜指标、安全状态机、random/grid/Sobol 搜索、持久化、CLI 和报告。它明确排除实时启动 Cadence、厂商 SCPI 命令、真实硬件写入、符合标准的 TDECQ 计算、贝叶斯优化、Bi-LSTM 训练、强化学习和 Dashboard。
Milestone 1 includes the shared kernel, fake/replay execution, skew metrics, safety state machine, random/grid/Sobol search, persistence, CLI, and reporting. It explicitly excludes live Cadence launching, vendor SCPI commands, real hardware writes, standards-compliant TDECQ calculation, Bayesian optimization, Bi-LSTM training, reinforcement learning, and a dashboard.

## 计划文件映射 / Planned File Map

```text
pyproject.toml
src/vcsel_ai_tuning/
|-- __init__.py                 # package identity
|-- domain.py                   # immutable domain records and enums
|-- config.py                   # YAML loading, manifests, and schema validation
|-- safety.py                   # deterministic candidate/read-back gate
|-- persistence.py              # atomic artifacts, SQLite, quarantine
|-- metrics.py                  # PAM4 crossing, skew, level, and BER metrics
|-- backends.py                 # backend protocol, fake, replay, interface stubs
|-- adapters.py                 # BERT/driver/scope dry-run instrument contracts
|-- objectives.py               # metric constraints and lexicographic ranking
|-- recovery.py                 # crash marker and operator acknowledgement
|-- engine.py                   # experiment state machine and rollback
|-- search.py                   # grid, seeded random, and Sobol candidates
|-- runner.py                   # campaign orchestration and backend factory
|-- report.py                   # baseline/best Markdown report
`-- cli.py                      # validate-config, run, report, export-schema
configs/ai_tuning/
|-- fake_campaign.yaml
`-- fake_calibration_manifest.yaml
tests_py/ai_tuning/
|-- __init__.py
|-- conftest.py
|-- fixtures/replay_observations.json
|-- test_package_import.py
|-- test_domain.py
|-- test_config.py
|-- test_safety.py
|-- test_persistence.py
|-- test_metrics.py
|-- test_backends.py
|-- test_adapters.py
|-- test_objectives.py
|-- test_recovery.py
|-- test_engine.py
|-- test_search.py
|-- test_runner.py
|-- test_report.py
`-- test_cli.py
schemas/ai_tuning/
|-- campaign.schema.json
|-- calibration_manifest.schema.json
`-- observation.schema.json
```

## 任务 0：提交前恢复有效的本地 Git 仓库 / Task 0: Restore a Valid Local Git Repository Before Commits

当前工作区包含一个空的或无效的 `.git` 目录，且 `git status` 以状态码 128 退出。不要移除任何 Git 元数据。只有在向用户披露此前提、且用户选择执行方案后，才填充该空目录。
The workspace currently contains an empty or invalid `.git` directory and `git status` exits 128. Do not remove any Git metadata. Populate the empty directory only after the user selects an execution option with this prerequisite disclosed.

**文件：**
**Files:**
- 验证：`.git/`
- Verify: `.git/`
- 稍后提交：`docs/superpowers/specs/2026-07-10-vcsel-ai-tuning-platform-design.md`
- Commit later: `docs/superpowers/specs/2026-07-10-vcsel-ai-tuning-platform-design.md`
- 稍后提交：`docs/superpowers/plans/2026-07-10-vcsel-ai-tuning-milestone-1.md`
- Commit later: `docs/superpowers/plans/2026-07-10-vcsel-ai-tuning-milestone-1.md`

**步骤 1：证明现有 Git 元数据为空且无效**
- [ ] **Step 1: Prove the existing Git metadata is empty and invalid**

运行：
Run:

```powershell
$count = (Get-ChildItem -LiteralPath .git -Force | Measure-Object).Count
Write-Output "git_entries=$count"
git status --short
```

预期：`git_entries=0`，且 `git status` 报告该目录不是 Git 仓库。如果计数不为零，请停止并检查现有元数据，而不是在其上执行初始化。
Expected: `git_entries=0` and `git status` reports that the directory is not a Git repository. If the count is not zero, stop and inspect the existing metadata instead of initializing over it.

**步骤 2：初始化元数据，但不暂存无关的工作区文件**
- [ ] **Step 2: Initialize metadata without staging unrelated workspace files**

运行：
Run:

```powershell
git init -b main
git status --short
```

预期：Git 初始化分支 `main`；状态信息列出未跟踪文件，但不修改这些文件。
Expected: Git initializes branch `main`; status lists untracked files but does not modify them.

**步骤 3：验证作者身份信息已存在**
- [ ] **Step 3: Verify author identity exists**

运行：
Run:

```powershell
git config user.name
git config user.email
```

预期：两条命令均返回用户现有的已配置身份信息。如果任一项为空，请暂停并要求用户进行配置；绝不能虚构作者身份。
Expected: both commands return the user's existing configured identity. If either is empty, pause and ask the user to configure it; never invent an author identity.

**步骤 4：仅提交已批准的设计和实施计划**
- [ ] **Step 4: Commit only the approved design and implementation plan**

运行：
Run:

```powershell
git add -- docs/superpowers/specs/2026-07-10-vcsel-ai-tuning-platform-design.md docs/superpowers/plans/2026-07-10-vcsel-ai-tuning-milestone-1.md
git commit -m "docs: define VCSEL AI tuning milestone one"
```

预期：生成一个文档提交；不暂存测量数据、输出或无关文件。
Expected: one documentation commit; no measurement data, outputs, or unrelated files are staged.

## 任务 1：引导创建可安装的 Python 包 / Task 1: Bootstrap the Installable Python Package

**文件：**
**Files:**
- 创建：`pyproject.toml`
- Create: `pyproject.toml`
- 创建：`src/vcsel_ai_tuning/__init__.py`
- Create: `src/vcsel_ai_tuning/__init__.py`
- 创建：`tests_py/ai_tuning/__init__.py`
- Create: `tests_py/ai_tuning/__init__.py`
- 创建：`tests_py/ai_tuning/test_package_import.py`
- Create: `tests_py/ai_tuning/test_package_import.py`
- 修改：`.gitignore`
- Modify: `.gitignore`

**步骤 1：编写预期失败的导入测试**
- [ ] **Step 1: Write the failing import test**

创建空文件 `tests_py/ai_tuning/__init__.py`，并创建 `tests_py/ai_tuning/test_package_import.py`：
Create `tests_py/ai_tuning/__init__.py` as an empty file and create `tests_py/ai_tuning/test_package_import.py`:

```python
import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_exports_version(self):
        import vcsel_ai_tuning

        self.assertEqual(vcsel_ai_tuning.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行测试并验证该包尚不存在**
- [ ] **Step 2: Run the test and verify the package is missing**

运行：
Run:

```powershell
py -3.13 -m unittest tests_py.ai_tuning.test_package_import -v
```

预期：出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning'`。
Expected: ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning'`.

**步骤 3：添加包元数据和最小包结构**
- [ ] **Step 3: Add package metadata and the minimal package**

创建 `pyproject.toml`：
Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "vcsel-ai-tuning"
version = "0.1.0"
description = "Safe digital-twin and hardware-ready VCSEL PAM4 tuning framework"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0",
  "pydantic>=2.9",
  "PyYAML>=6.0",
  "scipy>=1.14",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "pytest-cov>=5.0",
]

[project.scripts]
vcsel-tune = "vcsel_ai_tuning.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests_py/ai_tuning"]
addopts = "-ra"
```

创建 `src/vcsel_ai_tuning/__init__.py`：
Create `src/vcsel_ai_tuning/__init__.py`:

```python
"""VCSEL PAM4 AI tuning experiment framework."""

__version__ = "0.1.0"
```

将以下生成路径追加到 `.gitignore`，不要移除现有条目：
Append these generated paths to `.gitignore` without removing existing entries:

```gitignore
.venv-ai-tuning/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
state/*.sqlite*
artifacts/campaigns/
artifacts/quarantine/
```

**步骤 4：安装可编辑包并验证 GREEN**
- [ ] **Step 4: Install the editable package and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pip install -e ".[dev]"
py -3.13 -m unittest tests_py.ai_tuning.test_package_import -v
```

预期：安装以状态码 0 退出，且导入测试报告 `OK`。
Expected: installation exits 0 and the import test reports `OK`.

**步骤 5：提交引导创建内容**
- [ ] **Step 5: Commit the bootstrap**

运行：
Run:

```powershell
git add -- pyproject.toml .gitignore src/vcsel_ai_tuning/__init__.py tests_py/ai_tuning/__init__.py tests_py/ai_tuning/test_package_import.py
git commit -m "build: bootstrap VCSEL AI tuning package"
```

## 任务 2：定义不可变领域记录 / Task 2: Define Immutable Domain Records

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/domain.py`
- Create: `src/vcsel_ai_tuning/domain.py`
- 创建：`tests_py/ai_tuning/test_domain.py`
- Create: `tests_py/ai_tuning/test_domain.py`

**步骤 1：编写量化和观测可行性的预期失败测试**
- [ ] **Step 1: Write failing tests for quantization and observation feasibility**

创建 `tests_py/ai_tuning/test_domain.py`：
Create `tests_py/ai_tuning/test_domain.py`:

```python
from datetime import datetime, timezone

import pytest

from vcsel_ai_tuning.domain import (
    AppliedSettings,
    BerEvidence,
    Candidate,
    ConstraintResult,
    Fidelity,
    MetricValue,
    Observation,
    ParameterSpec,
    RunState,
)


def test_parameter_spec_quantizes_from_lower_bound():
    spec = ParameterSpec(
        name="bias_ma",
        unit="mA",
        lower=4.0,
        upper=7.0,
        quantization=0.1,
        max_step=1.0,
        readback_tolerance=0.02,
    )

    assert spec.prepare(4.34) == pytest.approx(4.3)


def test_parameter_spec_rejects_out_of_range_value():
    spec = ParameterSpec(name="bias_ma", unit="mA", lower=4.0, upper=7.0)

    with pytest.raises(ValueError, match="outside"):
        spec.prepare(7.1)


def test_parameter_spec_rejects_nan():
    spec = ParameterSpec(name="bias_ma", unit="mA", lower=4.0, upper=7.0)

    with pytest.raises(ValueError, match="finite"):
        spec.prepare(float("nan"))


def test_observation_feasible_requires_every_constraint_to_pass():
    settings = AppliedSettings(
        requested={"bias_ma": 5.0},
        applied={"bias_ma": 5.0},
        readback={"bias_ma": 5.0},
    )
    observation = Observation(
        campaign_id="campaign-1",
        run_id="run-1",
        candidate_id="candidate-0000",
        state=RunState.COMPLETE,
        settings=settings,
        metrics={
            "skew_pp_ui": MetricValue(
                value=0.03,
                unit="UI",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="skew-v1",
            )
        },
        constraints=[
            ConstraintResult(name="bias_ma.bounds", passed=True),
            ConstraintResult(name="rlm_proxy", passed=False, reason="below 0.9"),
        ],
        valid=True,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )

    assert observation.feasible is False


def test_valid_observation_must_finish_complete():
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0},
    )

    with pytest.raises(ValueError, match="valid observation"):
        Observation(
            campaign_id="campaign-1",
            run_id="run-1",
            candidate_id=candidate.candidate_id,
            state=RunState.ABORTED,
            settings=AppliedSettings(
                requested=candidate.requested,
                applied=candidate.requested,
                readback=candidate.requested,
            ),
            valid=True,
            failure_reason="aborted",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )


def test_tdecq_db_requires_hardware_standard_fidelity():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="tdecq_db"):
        Observation(
            campaign_id="campaign-1",
            run_id="run-1",
            candidate_id="candidate-0000",
            state=RunState.COMPLETE,
            settings=AppliedSettings(
                requested={"bias_ma": 5.0},
                applied={"bias_ma": 5.0},
                readback={"bias_ma": 5.0},
            ),
            metrics={
                "tdecq_db": MetricValue(
                    value=2.0,
                    unit="dB",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="invalid-v1",
                )
            },
            valid=True,
            started_at=now,
            finished_at=now,
        )


def test_zero_error_ber_evidence_requires_nonzero_upper_bound():
    with pytest.raises(ValueError, match="upper confidence bound"):
        BerEvidence(
            error_count=0,
            bit_count=1_000_000,
            confidence=0.95,
            upper_bound=0.0,
        )
```

**步骤 2：运行测试并验证领域模块尚不存在**
- [ ] **Step 2: Run the tests and verify the domain module is missing**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_domain.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.domain'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.domain'`.

**步骤 3：实现领域记录**
- [ ] **Step 3: Implement the domain records**

创建 `src/vcsel_ai_tuning/domain.py`：
Create `src/vcsel_ai_tuning/domain.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Fidelity(StrEnum):
    SIMULATION_PROXY = "simulation_proxy"
    HARDWARE_FAST = "hardware_fast"
    HARDWARE_STANDARD = "hardware_standard"


class RunState(StrEnum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    ARMED = "ARMED"
    APPLY = "APPLY"
    READBACK = "READBACK"
    SETTLE = "SETTLE"
    MEASURE_FAST = "MEASURE_FAST"
    VALIDATE = "VALIDATE"
    MEASURE_SLOW = "MEASURE_SLOW"
    COMMIT = "COMMIT"
    COMPLETE = "COMPLETE"
    ROLLBACK = "ROLLBACK"
    SAFE_OFF = "SAFE_OFF"
    ABORTED = "ABORTED"


class ParameterSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str
    unit: str
    lower: float
    upper: float
    quantization: float | None = Field(default=None, gt=0)
    max_step: float | None = Field(default=None, gt=0)
    readback_tolerance: float = Field(default=0.0, ge=0)
    grid_values: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_interval_and_grid(self) -> "ParameterSpec":
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")
        if any(value < self.lower or value > self.upper for value in self.grid_values):
            raise ValueError("grid value outside parameter bounds")
        return self

    def prepare(self, value: float) -> float:
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{self.name} must be finite")
        if value < self.lower or value > self.upper:
            raise ValueError(
                f"{self.name}={value} is outside [{self.lower}, {self.upper}] {self.unit}"
            )
        if self.quantization is None:
            return value
        steps = round((value - self.lower) / self.quantization)
        prepared = self.lower + steps * self.quantization
        return float(min(self.upper, max(self.lower, prepared)))


ContextValue = str | int | float | bool


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    requested: dict[str, float]
    context: dict[str, ContextValue] = Field(default_factory=dict)


class AppliedSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    requested: dict[str, float]
    applied: dict[str, float]
    readback: dict[str, float]


class MetricValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    value: float
    unit: str
    fidelity: Fidelity
    method_version: str
    uncertainty: float | None = Field(default=None, ge=0)
    valid: bool = True
    measurement_definition_id: str | None = None

    @model_validator(mode="after")
    def require_standard_measurement_identity(self) -> "MetricValue":
        if (
            self.fidelity is Fidelity.HARDWARE_STANDARD
            and not self.measurement_definition_id
        ):
            raise ValueError(
                "hardware_standard metric requires measurement_definition_id"
            )
        return self


class ConstraintResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str
    passed: bool
    value: float | None = None
    limit: str | None = None
    reason: str | None = None


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    relative_path: Path
    sha256: str
    media_type: str


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    state: RunState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: dict[str, Any] = Field(default_factory=dict)


class BerEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    error_count: int = Field(ge=0)
    bit_count: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)
    upper_bound: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "BerEvidence":
        if self.error_count > self.bit_count:
            raise ValueError("error_count cannot exceed bit_count")
        if self.error_count == 0 and self.upper_bound <= 0:
            raise ValueError("zero-error evidence requires a positive upper confidence bound")
        return self


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    state: RunState
    settings: AppliedSettings
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    constraints: list[ConstraintResult] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    ber_evidence: BerEvidence | None = None
    valid: bool
    failure_reason: str | None = None
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def validate_terminal_status(self) -> "Observation":
        if self.valid and self.state is not RunState.COMPLETE:
            raise ValueError("valid observation must finish in COMPLETE")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid observation requires failure_reason")
        for name, metric in self.metrics.items():
            if name == "tdecq_db" and metric.fidelity is not Fidelity.HARDWARE_STANDARD:
                raise ValueError("tdecq_db requires hardware_standard fidelity")
            if name == "tdecq_proxy" and metric.fidelity is Fidelity.HARDWARE_STANDARD:
                raise ValueError("tdecq_proxy cannot use hardware_standard fidelity")
        return self

    @property
    def feasible(self) -> bool:
        return self.valid and all(result.passed for result in self.constraints)


class BackendResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metrics: dict[str, MetricValue]
    raw_artifacts: dict[str, bytes] = Field(default_factory=dict)
    status: dict[str, ContextValue] = Field(default_factory=dict)
```

**步骤 4：运行领域测试并验证 GREEN**
- [ ] **Step 4: Run the domain tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_domain.py -v
```

预期：7 个测试通过。
Expected: 7 tests pass.

**步骤 5：提交领域模型**
- [ ] **Step 5: Commit the domain model**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/domain.py tests_py/ai_tuning/test_domain.py
git commit -m "feat: add immutable tuning domain records"
```

## 任务 3：校验实验活动与校准清单 / Task 3: Validate Campaigns and Calibration Manifests

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/config.py`
- Create: `src/vcsel_ai_tuning/config.py`
- 创建：`tests_py/ai_tuning/test_config.py`
- Create: `tests_py/ai_tuning/test_config.py`

**步骤 1：编写预期失败的配置测试**
- [ ] **Step 1: Write failing configuration tests**

创建 `tests_py/ai_tuning/test_config.py`：
Create `tests_py/ai_tuning/test_config.py`:

```python
from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vcsel_ai_tuning.config import (
    CalibrationManifest,
    CampaignConfig,
    load_yaml_model,
    validate_manifest_files,
)


def valid_campaign_dict():
    return {
        "campaign_id": "fake-skew-demo",
        "backend": {"kind": "fake", "dry_run": True},
        "optimizer": {"kind": "sobol", "budget": 8, "seed": 7},
        "parameters": [
            {
                "name": "bias_ma",
                "unit": "mA",
                "lower": 4.0,
                "upper": 7.0,
                "quantization": 0.1,
                "max_step": 1.0,
            },
            {
                "name": "skew_control_ui",
                "unit": "UI",
                "lower": -0.1,
                "upper": 0.1,
                "quantization": 0.001,
                "max_step": 0.05,
            },
        ],
        "safe_baseline": {"bias_ma": 5.0, "skew_control_ui": 0.05},
        "objective": {
            "primary_metric": "skew_pp_ui",
            "direction": "minimize",
            "metric_constraints": [
                {"metric": "rlm_proxy", "operator": ">=", "threshold": 0.9}
            ],
            "tie_breakers": [
                {"metric": "power_mw", "direction": "minimize"}
            ],
        },
    }


def test_campaign_accepts_exact_safe_baseline():
    campaign = CampaignConfig.model_validate(valid_campaign_dict())

    assert campaign.optimizer.budget == 8
    assert campaign.parameter_map["bias_ma"].unit == "mA"


def test_campaign_rejects_live_hardware_in_milestone_one():
    data = valid_campaign_dict()
    data["backend"] = {"kind": "hardware", "dry_run": False}

    with pytest.raises(ValidationError, match="dry-run"):
        CampaignConfig.model_validate(data)


def test_campaign_rejects_missing_safe_baseline_parameter():
    data = valid_campaign_dict()
    del data["safe_baseline"]["bias_ma"]

    with pytest.raises(ValidationError, match="safe_baseline"):
        CampaignConfig.model_validate(data)


def test_yaml_loader_and_manifest_hash_validation(tmp_path: Path):
    source = tmp_path / "measurement.csv"
    source.write_text("bias_ma,power_mw\n5.0,1.0\n", encoding="utf-8")
    digest = sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "manifest_id": "manifest-1",
                "device_id": "vcsel-1",
                "temperature_c": 25.0,
                "sources": [{"path": source.name, "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_yaml_model(manifest_path, CalibrationManifest)

    validate_manifest_files(manifest, tmp_path)
```

**步骤 2：运行测试并验证配置模块尚不存在**
- [ ] **Step 2: Run the tests and verify the configuration module is missing**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_config.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.config'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.config'`.

**步骤 3：实现经校验的配置和溯源哈希**
- [ ] **Step 3: Implement validated configuration and provenance hashes**

创建 `src/vcsel_ai_tuning/config.py`：
Create `src/vcsel_ai_tuning/config.py`:

```python
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .domain import ParameterSpec


class BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["fake", "replay", "cadence", "hardware"]
    dry_run: bool = True
    options: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def forbid_live_hardware(self) -> "BackendConfig":
        if self.kind == "hardware" and not self.dry_run:
            raise ValueError("Milestone 1 hardware backend must remain dry-run")
        return self


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["grid", "random", "sobol"]
    budget: int = Field(gt=0)
    seed: int = 0


class MetricConstraintSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metric: str
    operator: Literal[">=", "<="]
    threshold: float


class TieBreakerSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metric: str
    direction: Literal["minimize", "maximize"]


class ObjectiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    primary_metric: str
    direction: Literal["minimize", "maximize"]
    metric_constraints: tuple[MetricConstraintSpec, ...] = ()
    tie_breakers: tuple[TieBreakerSpec, ...] = ()


class CampaignConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    backend: BackendConfig
    optimizer: OptimizerConfig
    parameters: tuple[ParameterSpec, ...] = Field(min_length=1)
    safe_baseline: dict[str, float]
    objective: ObjectiveConfig
    slow_measurement_for_best: bool = False

    @model_validator(mode="after")
    def validate_parameters_and_baseline(self) -> "CampaignConfig":
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if set(self.safe_baseline) != set(names):
            raise ValueError("safe_baseline must define every parameter exactly once")
        for parameter in self.parameters:
            parameter.prepare(self.safe_baseline[parameter.name])
        return self

    @property
    def parameter_map(self) -> dict[str, ParameterSpec]:
        return {parameter.name: parameter for parameter in self.parameters}


class SourceArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    manifest_id: str
    device_id: str
    temperature_c: float
    model_path: Path | None = None
    external_network_path: Path | None = None
    sources: tuple[SourceArtifact, ...] = ()
    model_parameters: dict[str, float] = Field(default_factory=dict)


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_model(path: Path, model_type: type[ModelT]) -> ModelT:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return model_type.model_validate(data)


def sha256_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_manifest_files(manifest: CalibrationManifest, base_dir: Path) -> None:
    for source in manifest.sources:
        resolved = (base_dir / source.path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual = sha256_file(resolved)
        if actual != source.sha256:
            raise ValueError(
                f"hash mismatch for {source.path}: expected {source.sha256}, got {actual}"
            )
```

**步骤 4：运行配置测试并验证 GREEN**
- [ ] **Step 4: Run configuration tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_config.py -v
```

预期：4 个测试通过。
Expected: 4 tests pass.

**步骤 5：提交配置校验实现**
- [ ] **Step 5: Commit configuration validation**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/config.py tests_py/ai_tuning/test_config.py
git commit -m "feat: validate tuning campaigns and calibration manifests"
```

## 任务 4：强制执行确定性的候选点与读回安全 / Task 4: Enforce Deterministic Candidate and Read-Back Safety

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/safety.py`
- Create: `src/vcsel_ai_tuning/safety.py`
- 创建：`tests_py/ai_tuning/test_safety.py`
- Create: `tests_py/ai_tuning/test_safety.py`

**步骤 1：编写预期失败的安全策略测试**
- [ ] **Step 1: Write failing safety-policy tests**

创建 `tests_py/ai_tuning/test_safety.py`：
Create `tests_py/ai_tuning/test_safety.py`:

```python
import pytest

from vcsel_ai_tuning.domain import Candidate, ParameterSpec
from vcsel_ai_tuning.safety import SafetyPolicy, SafetyViolation


def policy():
    return SafetyPolicy(
        (
            ParameterSpec(
                name="bias_ma",
                unit="mA",
                lower=4.0,
                upper=7.0,
                quantization=0.1,
                max_step=1.0,
                readback_tolerance=0.02,
            ),
            ParameterSpec(
                name="spacing_01_v",
                unit="V",
                lower=0.05,
                upper=0.4,
                quantization=0.01,
                max_step=0.05,
                readback_tolerance=0.005,
            ),
        )
    )


def test_candidate_is_quantized_before_it_passes():
    decision = policy().evaluate_candidate(
        Candidate(
            candidate_id="candidate-1",
            requested={"bias_ma": 5.04, "spacing_01_v": 0.123},
        ),
        reference={"bias_ma": 5.0, "spacing_01_v": 0.10},
    )

    assert decision.passed is True
    assert decision.applied == {"bias_ma": 5.0, "spacing_01_v": 0.12}


def test_candidate_step_is_measured_from_safe_reference():
    decision = policy().evaluate_candidate(
        Candidate(
            candidate_id="candidate-2",
            requested={"bias_ma": 6.2, "spacing_01_v": 0.10},
        ),
        reference={"bias_ma": 5.0, "spacing_01_v": 0.10},
    )

    assert decision.passed is False
    assert any(result.name == "bias_ma.max_step" for result in decision.constraints)


def test_missing_parameter_never_passes():
    decision = policy().evaluate_candidate(
        Candidate(candidate_id="candidate-3", requested={"bias_ma": 5.0}),
        reference={"bias_ma": 5.0, "spacing_01_v": 0.10},
    )

    assert decision.passed is False
    assert decision.applied == {}


def test_readback_mismatch_fails_and_require_raises():
    readback = policy().evaluate_readback(
        applied={"bias_ma": 5.0, "spacing_01_v": 0.10},
        readback={"bias_ma": 5.05, "spacing_01_v": 0.10},
    )

    assert readback.passed is False
    with pytest.raises(SafetyViolation, match="read-back"):
        readback.require("read-back")


def test_ffe_taps_are_l1_normalized_and_zero_norm_is_rejected():
    ffe_policy = SafetyPolicy(
        (
            ParameterSpec(name="ffe_tap_0", unit="1", lower=-1.0, upper=1.0),
            ParameterSpec(name="ffe_tap_1", unit="1", lower=-1.0, upper=1.0),
        )
    )
    normalized = ffe_policy.evaluate_candidate(
        Candidate(
            candidate_id="ffe-1",
            requested={"ffe_tap_0": 2.0 / 3.0, "ffe_tap_1": -1.0 / 3.0},
        ),
        reference={"ffe_tap_0": 1.0, "ffe_tap_1": 0.0},
    )
    zero = ffe_policy.evaluate_candidate(
        Candidate(
            candidate_id="ffe-2",
            requested={"ffe_tap_0": 0.0, "ffe_tap_1": 0.0},
        ),
        reference={"ffe_tap_0": 1.0, "ffe_tap_1": 0.0},
    )

    assert sum(abs(value) for value in normalized.applied.values()) == pytest.approx(1.0)
    assert zero.passed is False
    assert any(result.name == "ffe.l1_norm" for result in zero.constraints)
```

**步骤 2：运行安全测试并验证 RED**
- [ ] **Step 2: Run the safety tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_safety.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.safety'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.safety'`.

**步骤 3：实现安全策略**
- [ ] **Step 3: Implement the safety policy**

创建 `src/vcsel_ai_tuning/safety.py`：
Create `src/vcsel_ai_tuning/safety.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .domain import Candidate, ConstraintResult, ParameterSpec


class SafetyViolation(RuntimeError):
    pass


class SafetyDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    passed: bool
    applied: dict[str, float] = Field(default_factory=dict)
    constraints: tuple[ConstraintResult, ...] = ()

    def require(self, stage: str) -> None:
        if not self.passed:
            failed = ", ".join(
                result.name for result in self.constraints if not result.passed
            )
            raise SafetyViolation(f"{stage} safety check failed: {failed}")


class SafetyPolicy:
    def __init__(self, parameters: tuple[ParameterSpec, ...]):
        self._specs = {parameter.name: parameter for parameter in parameters}
        if len(self._specs) != len(parameters):
            raise ValueError("parameter names must be unique")

    def evaluate_candidate(
        self,
        candidate: Candidate,
        reference: dict[str, float],
    ) -> SafetyDecision:
        expected = set(self._specs)
        actual = set(candidate.requested)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            return SafetyDecision(
                passed=False,
                constraints=(
                    ConstraintResult(
                        name="candidate.parameter_set",
                        passed=False,
                        reason=f"missing={missing}, extra={extra}",
                    ),
                ),
            )

        constraints: list[ConstraintResult] = []
        applied: dict[str, float] = {}
        for name, spec in self._specs.items():
            try:
                prepared = spec.prepare(candidate.requested[name])
            except ValueError as error:
                constraints.append(
                    ConstraintResult(
                        name=f"{name}.bounds",
                        passed=False,
                        value=float(candidate.requested[name]),
                        limit=f"[{spec.lower}, {spec.upper}] {spec.unit}",
                        reason=str(error),
                    )
                )
                continue

            applied[name] = prepared
            constraints.append(
                ConstraintResult(
                    name=f"{name}.bounds",
                    passed=True,
                    value=prepared,
                    limit=f"[{spec.lower}, {spec.upper}] {spec.unit}",
                )
            )
            if spec.max_step is not None:
                if name not in reference:
                    constraints.append(
                        ConstraintResult(
                            name=f"{name}.max_step",
                            passed=False,
                            reason="reference value missing",
                        )
                    )
                else:
                    delta = abs(prepared - float(reference[name]))
                    constraints.append(
                        ConstraintResult(
                            name=f"{name}.max_step",
                            passed=delta <= spec.max_step,
                            value=delta,
                            limit=f"<= {spec.max_step} {spec.unit}",
                            reason=None if delta <= spec.max_step else "step too large",
                        )
                    )

        tap_names = sorted(name for name in applied if name.startswith("ffe_tap_"))
        if tap_names:
            norm = sum(abs(applied[name]) for name in tap_names)
            if norm == 0.0:
                constraints.append(
                    ConstraintResult(
                        name="ffe.l1_norm",
                        passed=False,
                        reason="FFE tap L1 norm is zero",
                    )
                )
            else:
                normalized = {name: applied[name] / norm for name in tap_names}
                normalized_in_bounds = all(
                    self._specs[name].lower <= value <= self._specs[name].upper
                    for name, value in normalized.items()
                )
                constraints.append(
                    ConstraintResult(
                        name="ffe.l1_norm",
                        passed=normalized_in_bounds,
                        value=sum(abs(value) for value in normalized.values()),
                        limit="L1 norm = 1 and every tap in bounds",
                        reason=None
                        if normalized_in_bounds
                        else "normalized FFE tap outside bounds",
                    )
                )
                if normalized_in_bounds:
                    applied.update(normalized)

        passed = len(applied) == len(self._specs) and all(
            result.passed for result in constraints
        )
        return SafetyDecision(
            passed=passed,
            applied=applied if passed else {},
            constraints=tuple(constraints),
        )

    def evaluate_readback(
        self,
        applied: dict[str, float],
        readback: dict[str, float],
    ) -> SafetyDecision:
        constraints: list[ConstraintResult] = []
        for name, spec in self._specs.items():
            if name not in applied or name not in readback:
                constraints.append(
                    ConstraintResult(
                        name=f"{name}.readback",
                        passed=False,
                        reason="applied or read-back value missing",
                    )
                )
                continue
            delta = abs(float(readback[name]) - float(applied[name]))
            constraints.append(
                ConstraintResult(
                    name=f"{name}.readback",
                    passed=delta <= spec.readback_tolerance,
                    value=delta,
                    limit=f"<= {spec.readback_tolerance} {spec.unit}",
                    reason=None
                    if delta <= spec.readback_tolerance
                    else "read-back mismatch",
                )
            )
        passed = len(constraints) == len(self._specs) and all(
            result.passed for result in constraints
        )
        return SafetyDecision(
            passed=passed,
            applied=dict(applied) if passed else {},
            constraints=tuple(constraints),
        )
```

**步骤 4：运行安全测试并验证 GREEN**
- [ ] **Step 4: Run the safety tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_safety.py -v
```

预期：5 个测试通过。
Expected: 5 tests pass.

**步骤 5：提交确定性安全门禁实现**
- [ ] **Step 5: Commit deterministic safety gating**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/safety.py tests_py/ai_tuning/test_safety.py
git commit -m "feat: gate candidates and instrument readback"
```

## 任务 5：使用 SQLite 和隔离区对运行进行原子持久化 / Task 5: Persist Runs Atomically with SQLite and Quarantine

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/persistence.py`
- Create: `src/vcsel_ai_tuning/persistence.py`
- 创建：`tests_py/ai_tuning/test_persistence.py`
- Create: `tests_py/ai_tuning/test_persistence.py`

**步骤 1：编写预期失败的原子持久化测试**
- [ ] **Step 1: Write failing atomic-persistence tests**

创建 `tests_py/ai_tuning/test_persistence.py`：
Create `tests_py/ai_tuning/test_persistence.py`:

```python
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from vcsel_ai_tuning.domain import (
    AppliedSettings,
    Candidate,
    Fidelity,
    MetricValue,
    Observation,
    RunEvent,
    RunState,
)
from vcsel_ai_tuning.persistence import ArtifactStore, RunRepository, SqliteStore


def completed_observation():
    now = datetime.now(timezone.utc)
    return Observation(
        campaign_id="campaign-1",
        run_id="run-1",
        candidate_id="candidate-0000",
        state=RunState.COMPLETE,
        settings=AppliedSettings(
            requested={"bias_ma": 5.0},
            applied={"bias_ma": 5.0},
            readback={"bias_ma": 5.0},
        ),
        metrics={
            "skew_pp_ui": MetricValue(
                value=0.03,
                unit="UI",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="skew-v1",
            )
        },
        valid=True,
        started_at=now,
        finished_at=now,
    )


def test_commit_writes_artifacts_then_exposes_sqlite_observation(tmp_path: Path):
    store = SqliteStore(tmp_path / "state.sqlite")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    repository = RunRepository(store, artifacts)
    candidate = Candidate(candidate_id="candidate-0000", requested={"bias_ma": 5.0})
    waveform = b"time_s,amplitude_v\n0,0\n1e-12,1\n"

    saved = repository.commit_run(
        candidate=candidate,
        observation=completed_observation(),
        events=[RunEvent(state=RunState.COMPLETE)],
        raw_artifacts={"waveform.csv": (waveform, "text/csv")},
    )

    final = tmp_path / "artifacts" / "campaigns" / "campaign-1" / "runs" / "run-1"
    assert final.is_dir()
    assert (final / "observation.json").is_file()
    assert saved.artifacts[0].sha256 == sha256(waveform).hexdigest()
    assert store.list_observations("campaign-1") == [saved]


def test_database_failure_quarantines_artifacts(tmp_path: Path, monkeypatch):
    store = SqliteStore(tmp_path / "state.sqlite")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    repository = RunRepository(store, artifacts)

    def fail_save(_observation):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "save_observation", fail_save)

    with pytest.raises(RuntimeError, match="database unavailable"):
        repository.commit_run(
            candidate=Candidate(
                candidate_id="candidate-0000", requested={"bias_ma": 5.0}
            ),
            observation=completed_observation(),
            events=[RunEvent(state=RunState.COMPLETE)],
            raw_artifacts={},
        )

    assert list((tmp_path / "artifacts" / "quarantine").iterdir())
    assert store.list_observations("campaign-1") == []
```

**步骤 2：运行持久化测试并验证 RED**
- [ ] **Step 2: Run the persistence tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_persistence.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.persistence'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.persistence'`.

**步骤 3：实现原子制品、SQLite 和隔离区**
- [ ] **Step 3: Implement atomic artifacts, SQLite, and quarantine**

创建 `src/vcsel_ai_tuning/persistence.py`：
Create `src/vcsel_ai_tuning/persistence.py`:

```python
from __future__ import annotations

import json
import os
import re
import sqlite3
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .domain import ArtifactRef, Candidate, Observation, RunEvent


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.staging_root = self.root / ".staging"
        self.quarantine_root = self.root / "quarantine"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root.mkdir(parents=True, exist_ok=True)

    def stage_run(self, campaign_id: str, run_id: str) -> Path:
        stage = self.staging_root / campaign_id / f"{run_id}-{uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        return stage

    @staticmethod
    def _child(root: Path, relative_name: str) -> Path:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {relative_name}")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(
        self,
        stage: Path,
        relative_name: str,
        data: bytes,
        media_type: str,
    ) -> ArtifactRef:
        path = self._child(stage, relative_name)
        path.write_bytes(data)
        return ArtifactRef(
            relative_path=Path(relative_name),
            sha256=sha256(data).hexdigest(),
            media_type=media_type,
        )

    def write_json(self, stage: Path, relative_name: str, value: object) -> None:
        path = self._child(stage, relative_name)
        payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        path.write_text(payload + "\n", encoding="utf-8")

    def finalize(self, stage: Path, campaign_id: str, run_id: str) -> Path:
        final = self.root / "campaigns" / campaign_id / "runs" / run_id
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(final)
        os.replace(stage, final)
        return final

    def quarantine(self, run_path: Path, reason: str) -> Path:
        safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "-", reason).strip("-") or "error"
        destination = self.quarantine_root / f"{run_path.name}-{safe_reason}-{uuid4().hex}"
        os.replace(run_path, destination)
        return destination


class SqliteStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    run_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    valid INTEGER NOT NULL,
                    feasible INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, run_id)
                )
                """
            )

    def save_observation(self, observation: Observation) -> None:
        payload = observation.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO observations
                    (run_id, campaign_id, candidate_id, valid, feasible, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.run_id,
                    observation.campaign_id,
                    observation.candidate_id,
                    int(observation.valid),
                    int(observation.feasible),
                    payload,
                ),
            )

    def list_observations(self, campaign_id: str) -> list[Observation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM observations
                WHERE campaign_id = ?
                ORDER BY run_id
                """,
                (campaign_id,),
            ).fetchall()
        return [Observation.model_validate_json(row[0]) for row in rows]


class RunRepository:
    def __init__(self, store: SqliteStore, artifacts: ArtifactStore):
        self.store = store
        self.artifacts = artifacts

    def commit_run(
        self,
        candidate: Candidate,
        observation: Observation,
        events: list[RunEvent],
        raw_artifacts: dict[str, tuple[bytes, str]],
    ) -> Observation:
        stage = self.artifacts.stage_run(observation.campaign_id, observation.run_id)
        refs = [
            self.artifacts.write_bytes(stage, name, data, media_type)
            for name, (data, media_type) in sorted(raw_artifacts.items())
        ]
        saved = observation.model_copy(update={"artifacts": refs})
        self.artifacts.write_json(
            stage, "candidate.json", candidate.model_dump(mode="json")
        )
        self.artifacts.write_json(
            stage, "observation.json", saved.model_dump(mode="json")
        )
        event_lines = "".join(event.model_dump_json() + "\n" for event in events)
        self.artifacts._child(stage, "events.jsonl").write_text(
            event_lines, encoding="utf-8"
        )
        final = self.artifacts.finalize(
            stage, observation.campaign_id, observation.run_id
        )
        try:
            self.store.save_observation(saved)
        except Exception:
            self.artifacts.quarantine(final, "database-commit-failed")
            raise
        return saved
```

**步骤 4：运行持久化测试并验证 GREEN**
- [ ] **Step 4: Run persistence tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_persistence.py -v
```

预期：2 个测试通过。
Expected: 2 tests pass.

**步骤 5：提交持久化与隔离行为**
- [ ] **Step 5: Commit persistence and quarantine behavior**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/persistence.py tests_py/ai_tuning/test_persistence.py
git commit -m "feat: persist tuning runs atomically"
```

## 任务 6：测量 PAM4 倾斜、电平质量和 BER 证据 / Task 6: Measure PAM4 Skew, Level Quality, and BER Evidence

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/metrics.py`
- Create: `src/vcsel_ai_tuning/metrics.py`
- 创建：`tests_py/ai_tuning/test_metrics.py`
- Create: `tests_py/ai_tuning/test_metrics.py`

**步骤 1：编写预期失败的合成波形指标测试**
- [ ] **Step 1: Write failing synthetic-waveform metric tests**

创建 `tests_py/ai_tuning/test_metrics.py`：
Create `tests_py/ai_tuning/test_metrics.py`:

```python
import numpy as np
import pytest

from vcsel_ai_tuning.metrics import (
    analyze_pam4,
    ber_upper_confidence_bound,
    level_metrics,
)


def synthetic_pam4_ramp(
    skew_pp_ui: float,
    global_delay_ui: float = 0.0,
    periods: int = 64,
    samples_per_ui: int = 512,
):
    ui_s = 1.0e-12
    time_s = np.arange(periods * samples_per_ui, dtype=float)
    time_s *= ui_s / samples_per_ui
    local_ui = ((time_s / ui_s - global_delay_ui + 0.5) % 1.0) - 0.5
    offsets = (-skew_pp_ui / 2.0, 0.0, skew_pp_ui / 2.0)
    amplitude = np.interp(
        local_ui,
        [-0.25, offsets[0], offsets[1], offsets[2], 0.25],
        [0.0, 0.5, 1.5, 2.5, 3.0],
    )
    return time_s, amplitude, ui_s


@pytest.mark.parametrize("injected", [0.0, 0.03, 0.06, 0.10])
def test_skew_metric_recovers_known_injection(injected):
    time_s, amplitude, ui_s = synthetic_pam4_ramp(injected)

    metrics = analyze_pam4(time_s, amplitude, ui_s, [0.0, 1.0, 2.0, 3.0])

    tolerance = max(0.005, injected * 0.10)
    assert metrics["skew_pp_ui"].value == pytest.approx(injected, abs=tolerance)


def test_global_delay_does_not_change_skew():
    base_t, base_y, ui_s = synthetic_pam4_ramp(0.06, global_delay_ui=0.0)
    shift_t, shift_y, _ = synthetic_pam4_ramp(0.06, global_delay_ui=0.21)

    base = analyze_pam4(base_t, base_y, ui_s, [0.0, 1.0, 2.0, 3.0])
    shifted = analyze_pam4(shift_t, shift_y, ui_s, [0.0, 1.0, 2.0, 3.0])

    assert abs(base["skew_pp_ui"].value - shifted["skew_pp_ui"].value) <= 0.005


def test_level_metrics_are_one_for_ideal_rlm_proxy():
    metrics = level_metrics([0.0, 1.0, 2.0, 3.0])

    assert metrics["rlm_proxy"].value == pytest.approx(1.0)
    assert metrics["worst_eye_height_norm"].value == pytest.approx(1.0 / 3.0)
    assert metrics["eye_height_01_norm"].value == pytest.approx(1.0 / 3.0)
    assert metrics["eye_height_12_norm"].value == pytest.approx(1.0 / 3.0)
    assert metrics["eye_height_23_norm"].value == pytest.approx(1.0 / 3.0)


def test_zero_error_upper_bound_is_not_zero():
    upper = ber_upper_confidence_bound(0, 1_000_000, confidence=0.95)

    assert upper == pytest.approx(2.995728e-6, rel=1e-5)


def test_nan_waveform_is_rejected():
    time_s, amplitude, ui_s = synthetic_pam4_ramp(0.03)
    amplitude[3] = np.nan

    with pytest.raises(ValueError, match="finite"):
        analyze_pam4(time_s, amplitude, ui_s, [0.0, 1.0, 2.0, 3.0])
```

**步骤 2：运行指标测试并验证 RED**
- [ ] **Step 2: Run metric tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_metrics.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.metrics'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.metrics'`.

**步骤 3：实现交叉点、倾斜、电平和 BER 指标**
- [ ] **Step 3: Implement crossing, skew, level, and BER metrics**

创建 `src/vcsel_ai_tuning/metrics.py`：
Create `src/vcsel_ai_tuning/metrics.py`:

```python
from __future__ import annotations

from math import atan2, pi

import numpy as np
from scipy.stats import beta

from .domain import Fidelity, MetricValue


def _validate_waveform(time_s: np.ndarray, amplitude: np.ndarray, ui_s: float) -> None:
    if time_s.ndim != 1 or amplitude.ndim != 1 or len(time_s) != len(amplitude):
        raise ValueError("time and amplitude must be equal-length one-dimensional arrays")
    if len(time_s) < 4:
        raise ValueError("waveform has too few samples")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(amplitude)):
        raise ValueError("waveform values must be finite")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time values must be strictly increasing")
    if not np.isfinite(ui_s) or ui_s <= 0:
        raise ValueError("ui_s must be finite and positive")


def _rising_crossing_phases(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    threshold: float,
    ui_s: float,
) -> np.ndarray:
    lower = amplitude[:-1]
    upper = amplitude[1:]
    mask = (lower < threshold) & (upper >= threshold) & (upper > lower)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"no rising crossings for threshold {threshold}")
    fraction = (threshold - amplitude[indices]) / (
        amplitude[indices + 1] - amplitude[indices]
    )
    crossing_time = time_s[indices] + fraction * (
        time_s[indices + 1] - time_s[indices]
    )
    return ((crossing_time / ui_s + 0.5) % 1.0) - 0.5


def _circular_center(phases_ui: np.ndarray) -> float:
    angles = 2.0 * pi * phases_ui
    center = atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    return float(((center / (2.0 * pi) + 0.5) % 1.0) - 0.5)


def _relative_phase(value: float, reference: float) -> float:
    return float(((value - reference + 0.5) % 1.0) - 0.5)


def level_metrics(levels: list[float]) -> dict[str, MetricValue]:
    values = np.asarray(levels, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("exactly four finite PAM4 levels are required")
    spacings = np.diff(values)
    if np.any(spacings <= 0):
        raise ValueError("PAM4 levels must be strictly increasing")
    outer = float(values[-1] - values[0])
    minimum = float(spacings.min())
    metrics = {
        "rlm_proxy": MetricValue(
            value=3.0 * minimum / outer,
            unit="1",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="rlm-proxy-v1",
        ),
        "worst_eye_height_norm": MetricValue(
            value=minimum / outer,
            unit="1",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="level-spacing-v1",
        ),
    }
    for index, spacing in enumerate(spacings):
        label = ("01", "12", "23")[index]
        metrics[f"eye_height_{label}_norm"] = MetricValue(
            value=float(spacing / outer),
            unit="1",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="level-spacing-v1",
        )
    return metrics


def analyze_pam4(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    ui_s: float,
    levels: list[float],
    transition_guard_ui: float = 0.10,
) -> dict[str, MetricValue]:
    time_s = np.asarray(time_s, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    _validate_waveform(time_s, amplitude, ui_s)
    values = np.asarray(levels, dtype=float)
    if values.shape != (4,) or np.any(np.diff(values) <= 0):
        raise ValueError("four strictly increasing PAM4 levels are required")
    thresholds = (values[:-1] + values[1:]) / 2.0
    centers = [
        _circular_center(_rising_crossing_phases(time_s, amplitude, value, ui_s))
        for value in thresholds
    ]
    reference = centers[1]
    offsets = [_relative_phase(center, reference) for center in centers]
    skew = max(offsets) - min(offsets)
    metrics = {
        "eye_center_01_ui": MetricValue(
            value=offsets[0],
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="crossing-center-v1",
        ),
        "eye_center_12_ui": MetricValue(
            value=offsets[1],
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="crossing-center-v1",
        ),
        "eye_center_23_ui": MetricValue(
            value=offsets[2],
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="crossing-center-v1",
        ),
        "skew_pp_ui": MetricValue(
            value=skew,
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="skew-v1",
        ),
        "common_eye_width_proxy_ui": MetricValue(
            value=max(0.0, 1.0 - skew - 2.0 * transition_guard_ui),
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="common-window-proxy-v1",
        ),
        "worst_eye_width_proxy_ui": MetricValue(
            value=max(0.0, 1.0 - skew - 2.0 * transition_guard_ui),
            unit="UI",
            fidelity=Fidelity.SIMULATION_PROXY,
            method_version="common-window-proxy-v1",
        ),
    }
    metrics.update(level_metrics(levels))
    return metrics


def ber_upper_confidence_bound(
    error_count: int,
    bit_count: int,
    confidence: float = 0.95,
) -> float:
    if bit_count <= 0:
        raise ValueError("bit_count must be positive")
    if error_count < 0 or error_count > bit_count:
        raise ValueError("error_count must be between zero and bit_count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if error_count == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / bit_count))
    if error_count == bit_count:
        return 1.0
    return float(beta.ppf(confidence, error_count + 1, bit_count - error_count))
```

**步骤 4：运行指标测试并验证 GREEN**
- [ ] **Step 4: Run metric tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_metrics.py -v
```

预期：8 个测试通过：四个注入倾斜用例，以及四个不变量/错误用例。
Expected: 8 tests pass: four injected-skew cases plus four invariant/error cases.

**步骤 5：提交科学指标实现**
- [ ] **Step 5: Commit the scientific metrics**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/metrics.py tests_py/ai_tuning/test_metrics.py
git commit -m "feat: measure PAM4 skew and BER evidence"
```

## 任务 7：实现后端契约、Fake 后端和 Replay 后端 / Task 7: Implement the Backend Contract, Fake Backend, and Replay Backend

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/backends.py`
- Create: `src/vcsel_ai_tuning/backends.py`
- 创建：`tests_py/ai_tuning/fixtures/replay_observations.jsonl`
- Create: `tests_py/ai_tuning/fixtures/replay_observations.jsonl`
- 创建：`tests_py/ai_tuning/test_backends.py`
- Create: `tests_py/ai_tuning/test_backends.py`

**步骤 1：编写预期失败的后端契约测试**
- [ ] **Step 1: Write failing backend contract tests**

将 `tests_py/ai_tuning/fixtures/replay_observations.jsonl` 创建为单行单个 JSON 对象：
Create `tests_py/ai_tuning/fixtures/replay_observations.jsonl` as one JSON object on one line:

```json
{"campaign_id":"replay-campaign","run_id":"replay-run-1","candidate_id":"candidate-0000","state":"COMPLETE","settings":{"requested":{"bias_ma":5.0,"skew_control_ui":0.04},"applied":{"bias_ma":5.0,"skew_control_ui":0.04},"readback":{"bias_ma":5.0,"skew_control_ui":0.04}},"metrics":{"skew_pp_ui":{"value":0.04,"unit":"UI","fidelity":"simulation_proxy","method_version":"replay-v1","uncertainty":null,"valid":true,"measurement_definition_id":null}},"constraints":[],"artifacts":[],"valid":true,"failure_reason":null,"started_at":"2026-07-10T00:00:00Z","finished_at":"2026-07-10T00:00:01Z"}
```

创建 `tests_py/ai_tuning/test_backends.py`：
Create `tests_py/ai_tuning/test_backends.py`:

```python
from pathlib import Path

import pytest

from vcsel_ai_tuning.backends import BackendFault, FakeBackend, ReplayBackend
from vcsel_ai_tuning.domain import Candidate, Fidelity


def test_fake_backend_records_contract_order_and_is_deterministic():
    backend = FakeBackend()
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )

    assert backend.identify()["model"] == "FakeVCSELBench"
    backend.configure({"mode": "fast"})
    backend.apply(candidate, candidate.requested)
    readback = backend.readback()
    backend.wait_ready()
    result = backend.measure_fast()
    backend.safe_off()
    backend.safe_off()

    assert readback == candidate.requested
    assert result.metrics["skew_pp_ui"].value == pytest.approx(0.04)
    assert backend.call_order == [
        "identify",
        "configure",
        "apply",
        "readback",
        "wait_ready",
        "measure_fast",
        "safe_off",
        "safe_off",
    ]
    assert backend.is_safe is True


def test_fake_backend_injects_fault_at_named_operation():
    backend = FakeBackend(fail_at="measure_fast")
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )
    backend.apply(candidate, candidate.requested)

    with pytest.raises(BackendFault, match="measure_fast"):
        backend.measure_fast()


def test_replay_backend_preserves_metric_fidelity():
    fixture = (
        Path(__file__).parent / "fixtures" / "replay_observations.jsonl"
    )
    backend = ReplayBackend.from_jsonl(fixture)
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )

    backend.apply(candidate, candidate.requested)
    result = backend.measure_fast()

    assert backend.readback() == candidate.requested
    assert result.metrics["skew_pp_ui"].fidelity is Fidelity.SIMULATION_PROXY


def test_replay_backend_rejects_unknown_candidate():
    fixture = (
        Path(__file__).parent / "fixtures" / "replay_observations.jsonl"
    )
    backend = ReplayBackend.from_jsonl(fixture)
    candidate = Candidate(
        candidate_id="candidate-9999",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )

    with pytest.raises(KeyError, match="candidate-9999"):
        backend.apply(candidate, candidate.requested)
```

**步骤 2：运行后端测试并验证 RED**
- [ ] **Step 2: Run backend tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_backends.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.backends'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.backends'`.

**步骤 3：实现后端协议、确定性 Fake 后端和 Replay 后端**
- [ ] **Step 3: Implement backend protocol, deterministic fake, and replay**

创建 `src/vcsel_ai_tuning/backends.py`：
Create `src/vcsel_ai_tuning/backends.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .domain import (
    BackendResult,
    Candidate,
    Fidelity,
    MetricValue,
    Observation,
)


class BackendFault(RuntimeError):
    pass


class BackendUnavailable(RuntimeError):
    pass


class ExperimentBackend(Protocol):
    def identify(self) -> dict[str, str]: ...

    def configure(self, settings: dict[str, object]) -> None: ...

    def apply(self, candidate: Candidate, applied: dict[str, float]) -> None: ...

    def readback(self) -> dict[str, float]: ...

    def wait_ready(self) -> None: ...

    def measure_fast(self) -> BackendResult: ...

    def measure_slow(self) -> BackendResult | None: ...

    def abort(self) -> None: ...

    def safe_off(self) -> None: ...


class FakeBackend:
    def __init__(
        self,
        fail_at: str | None = None,
        readback_offsets: dict[str, float] | None = None,
    ):
        self.fail_at = fail_at
        self.readback_offsets = readback_offsets or {}
        self.call_order: list[str] = []
        self._applied: dict[str, float] = {}
        self._candidate: Candidate | None = None
        self.is_safe = True

    def _record(self, operation: str) -> None:
        self.call_order.append(operation)
        if self.fail_at == operation:
            raise BackendFault(f"injected fault at {operation}")

    def identify(self) -> dict[str, str]:
        self._record("identify")
        return {"vendor": "Codex", "model": "FakeVCSELBench", "serial": "FAKE-1"}

    def configure(self, settings: dict[str, object]) -> None:
        self._record("configure")

    def apply(self, candidate: Candidate, applied: dict[str, float]) -> None:
        self._record("apply")
        self._candidate = candidate
        self._applied = dict(applied)
        self.is_safe = False

    def readback(self) -> dict[str, float]:
        self._record("readback")
        return {
            name: value + self.readback_offsets.get(name, 0.0)
            for name, value in self._applied.items()
        }

    def wait_ready(self) -> None:
        self._record("wait_ready")

    def measure_fast(self) -> BackendResult:
        self._record("measure_fast")
        if self._candidate is None:
            raise BackendFault("no candidate has been applied")
        skew = abs(self._applied.get("skew_control_ui", 0.0))
        bias = self._applied.get("bias_ma", 0.0)
        modulation = self._applied.get("modulation_ma", 0.0)
        return BackendResult(
            metrics={
                "skew_pp_ui": MetricValue(
                    value=skew,
                    unit="UI",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "rlm_proxy": MetricValue(
                    value=max(0.0, 1.0 - 0.5 * skew),
                    unit="1",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "worst_eye_height_norm": MetricValue(
                    value=max(0.0, 1.0 / 3.0 - 0.2 * skew),
                    unit="1",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "worst_eye_width_proxy_ui": MetricValue(
                    value=max(0.0, 0.8 - skew),
                    unit="UI",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
                "power_mw": MetricValue(
                    value=1.0 + 0.2 * bias + 0.1 * modulation,
                    unit="mW",
                    fidelity=Fidelity.SIMULATION_PROXY,
                    method_version="fake-v1",
                ),
            },
            raw_artifacts={
                "fake_measurement.txt": (
                    f"candidate={self._candidate.candidate_id}\n".encode("utf-8")
                )
            },
            status={"lock": True, "sync": True},
        )

    def measure_slow(self) -> BackendResult | None:
        self._record("measure_slow")
        return None

    def abort(self) -> None:
        self._record("abort")

    def safe_off(self) -> None:
        self._record("safe_off")
        self.is_safe = True


class ReplayBackend:
    def __init__(self, observations: dict[str, Observation]):
        self._observations = observations
        self._active: Observation | None = None
        self.call_order: list[str] = []
        self.is_safe = True

    @classmethod
    def from_jsonl(cls, path: Path) -> "ReplayBackend":
        observations: dict[str, Observation] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            observation = Observation.model_validate_json(line)
            observations[observation.candidate_id] = observation
        return cls(observations)

    def identify(self) -> dict[str, str]:
        self.call_order.append("identify")
        return {"vendor": "Replay", "model": "ObservationFixture", "serial": "N/A"}

    def configure(self, settings: dict[str, object]) -> None:
        self.call_order.append("configure")

    def apply(self, candidate: Candidate, applied: dict[str, float]) -> None:
        self.call_order.append("apply")
        if candidate.candidate_id not in self._observations:
            raise KeyError(candidate.candidate_id)
        self._active = self._observations[candidate.candidate_id]
        self.is_safe = False

    def readback(self) -> dict[str, float]:
        self.call_order.append("readback")
        if self._active is None:
            raise BackendFault("no replay observation selected")
        return dict(self._active.settings.readback)

    def wait_ready(self) -> None:
        self.call_order.append("wait_ready")

    def measure_fast(self) -> BackendResult:
        self.call_order.append("measure_fast")
        if self._active is None:
            raise BackendFault("no replay observation selected")
        metrics = {
            name: metric
            for name, metric in self._active.metrics.items()
            if metric.fidelity is not Fidelity.HARDWARE_STANDARD
        }
        return BackendResult(metrics=metrics)

    def measure_slow(self) -> BackendResult | None:
        self.call_order.append("measure_slow")
        if self._active is None:
            raise BackendFault("no replay observation selected")
        metrics = {
            name: metric
            for name, metric in self._active.metrics.items()
            if metric.fidelity is Fidelity.HARDWARE_STANDARD
        }
        return BackendResult(metrics=metrics) if metrics else None

    def abort(self) -> None:
        self.call_order.append("abort")

    def safe_off(self) -> None:
        self.call_order.append("safe_off")
        self.is_safe = True
```

**步骤 4：运行后端测试并验证 GREEN**
- [ ] **Step 4: Run backend tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_backends.py -v
```

预期：4 个测试通过。
Expected: 4 tests pass.

**步骤 5：提交后端契约**
- [ ] **Step 5: Commit backend contracts**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/backends.py tests_py/ai_tuning/test_backends.py tests_py/ai_tuning/fixtures/replay_observations.jsonl
git commit -m "feat: add fake and replay experiment backends"
```

## 任务 8：添加 BERT、Driver 和 Scope/RX Dry-Run 适配器 / Task 8: Add BERT, Driver, and Scope/RX Dry-Run Adapters

**文件：**
**Files:**
- 创建：`src/vcsel_ai_tuning/adapters.py`
- Create: `src/vcsel_ai_tuning/adapters.py`
- 创建：`tests_py/ai_tuning/test_adapters.py`
- Create: `tests_py/ai_tuning/test_adapters.py`

**步骤 1：编写预期失败的共享适配器契约测试**
- [ ] **Step 1: Write failing shared adapter-contract tests**

创建 `tests_py/ai_tuning/test_adapters.py`：
Create `tests_py/ai_tuning/test_adapters.py`:

```python
import pytest

from vcsel_ai_tuning.adapters import DryRunInstrumentAdapter, InstrumentRole


@pytest.mark.parametrize(
    ("role", "fixture"),
    [
        (InstrumentRole.BERT, {"error_count": 0, "bit_count": 1_000_000}),
        (InstrumentRole.DRIVER, {"bias_ma": 5.0, "modulation_ma": 2.0}),
        (InstrumentRole.SCOPE_RX, {"skew_pp_ui": 0.04, "rlm": 0.96}),
    ],
)
def test_each_role_obeys_dry_run_contract(role, fixture):
    adapter = DryRunInstrumentAdapter(role=role, fixture=fixture)

    identity = adapter.identify()
    adapter.configure({"preset": "m1"})
    adapter.apply({"value": 1.0})
    readback = adapter.readback()
    measured = adapter.measure()
    adapter.abort()
    adapter.safe_off()
    adapter.safe_off()

    assert identity.vendor == "DRY-RUN"
    assert identity.role is role
    assert readback == {"value": 1.0}
    assert measured == fixture
    assert adapter.is_safe is True


def test_dry_run_adapter_has_no_live_mode_switch():
    with pytest.raises(TypeError):
        DryRunInstrumentAdapter(
            role=InstrumentRole.BERT,
            fixture={},
            live=True,
        )
```

**步骤 2：运行适配器测试并验证 RED**
- [ ] **Step 2: Run adapter tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_adapters.py -v
```

预期：测试收集阶段出现 ERROR，内容包含 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.adapters'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.adapters'`.

**步骤 3：实现仅接口的 dry-run 适配器**
- [ ] **Step 3: Implement interface-only dry-run adapters**

创建 `src/vcsel_ai_tuning/adapters.py`：
Create `src/vcsel_ai_tuning/adapters.py`:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class InstrumentRole(StrEnum):
    BERT = "bert"
    DRIVER = "driver_bias"
    SCOPE_RX = "scope_rx"


class InstrumentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    vendor: str
    model: str
    serial: str
    firmware: str
    role: InstrumentRole


class InstrumentAdapter(Protocol):
    def identify(self) -> InstrumentIdentity: ...

    def configure(self, settings: dict[str, object]) -> None: ...

    def apply(self, settings: dict[str, float]) -> None: ...

    def readback(self) -> dict[str, float]: ...

    def measure(self) -> dict[str, object]: ...

    def abort(self) -> None: ...

    def safe_off(self) -> None: ...


class DryRunInstrumentAdapter:
    def __init__(self, role: InstrumentRole, fixture: dict[str, object]):
        self.role = role
        self.fixture = dict(fixture)
        self._applied: dict[str, float] = {}
        self.is_safe = True
        self.call_order: list[str] = []

    def identify(self) -> InstrumentIdentity:
        self.call_order.append("identify")
        return InstrumentIdentity(
            vendor="DRY-RUN",
            model=f"{self.role.value}-fixture",
            serial="DRY-RUN",
            firmware="fixture-v1",
            role=self.role,
        )

    def configure(self, settings: dict[str, object]) -> None:
        self.call_order.append("configure")

    def apply(self, settings: dict[str, float]) -> None:
        self.call_order.append("apply")
        self._applied = dict(settings)
        self.is_safe = False

    def readback(self) -> dict[str, float]:
        self.call_order.append("readback")
        return dict(self._applied)

    def measure(self) -> dict[str, object]:
        self.call_order.append("measure")
        return dict(self.fixture)

    def abort(self) -> None:
        self.call_order.append("abort")

    def safe_off(self) -> None:
        self.call_order.append("safe_off")
        self.is_safe = True
```

**步骤 4：运行 dry-run 适配器测试并验证 GREEN**
- [ ] **Step 4: Run dry-run adapter tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_adapters.py -v
```

预期：4 个测试通过：每个角色各一个，加上 live 模式拒绝测试。
Expected: 4 tests pass: one for each role plus the live-mode rejection.

**步骤 5：提交 dry-run 仪器契约**
- [ ] **Step 5: Commit dry-run instrument contracts**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/adapters.py tests_py/ai_tuning/test_adapters.py
git commit -m "feat: define dry-run instrument adapter contracts"
```

## 任务 9：评估结果约束并对有效观测进行排名 / Task 9: Evaluate Outcome Constraints and Rank Valid Observations

**文件 / Files:**
- 新建：`src/vcsel_ai_tuning/objectives.py`
- Create: `src/vcsel_ai_tuning/objectives.py`
- 新建：`tests_py/ai_tuning/test_objectives.py`
- Create: `tests_py/ai_tuning/test_objectives.py`

- [ ] **步骤 1：编写会失败的目标与词典序排名测试 / Step 1: Write failing objective and lexicographic-ranking tests**

新建 `tests_py/ai_tuning/test_objectives.py`：
Create `tests_py/ai_tuning/test_objectives.py`:

```python
from datetime import datetime, timezone

from vcsel_ai_tuning.config import ObjectiveConfig
from vcsel_ai_tuning.domain import (
    AppliedSettings,
    ConstraintResult,
    Fidelity,
    MetricValue,
    Observation,
    RunState,
)
from vcsel_ai_tuning.objectives import evaluate_metric_constraints, select_incumbent


def observation(candidate_id, skew, rlm, power, valid=True):
    now = datetime.now(timezone.utc)
    return Observation(
        campaign_id="campaign-1",
        run_id=f"run-{candidate_id}",
        candidate_id=candidate_id,
        state=RunState.COMPLETE if valid else RunState.ABORTED,
        settings=AppliedSettings(
            requested={"bias_ma": 5.0},
            applied={"bias_ma": 5.0},
            readback={"bias_ma": 5.0},
        ),
        metrics={
            "skew_pp_ui": MetricValue(
                value=skew,
                unit="UI",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="fake-v1",
            ),
            "rlm_proxy": MetricValue(
                value=rlm,
                unit="1",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="fake-v1",
            ),
            "power_mw": MetricValue(
                value=power,
                unit="mW",
                fidelity=Fidelity.SIMULATION_PROXY,
                method_version="fake-v1",
            ),
        },
        valid=valid,
        failure_reason=None if valid else "injected failure",
        started_at=now,
        finished_at=now,
    )


def objective():
    return ObjectiveConfig.model_validate(
        {
            "primary_metric": "skew_pp_ui",
            "direction": "minimize",
            "metric_constraints": [
                {"metric": "rlm_proxy", "operator": ">=", "threshold": 0.9}
            ],
            "tie_breakers": [
                {"metric": "power_mw", "direction": "minimize"}
            ],
        }
    )


def test_missing_or_failed_metric_constraint_is_infeasible():
    failed = evaluate_metric_constraints(
        observation("bad", skew=0.01, rlm=0.8, power=1.0).metrics,
        objective(),
    )

    assert failed[0].passed is False


def test_incumbent_filters_invalid_and_infeasible_before_ranking():
    safety_failed = observation("safety-failed", skew=0.0, rlm=1.0, power=0.4)
    safety_failed = safety_failed.model_copy(
        update={
            "constraints": [
                ConstraintResult(
                    name="bias_ma.bounds",
                    passed=False,
                    reason="unsafe",
                )
            ]
        }
    )
    candidates = [
        observation("invalid", skew=0.0, rlm=1.0, power=0.5, valid=False),
        safety_failed,
        observation("infeasible", skew=0.01, rlm=0.8, power=0.5),
        observation("winner", skew=0.02, rlm=0.95, power=1.2),
        observation("runner-up", skew=0.03, rlm=0.99, power=0.9),
    ]

    best = select_incumbent(candidates, objective())

    assert best.candidate_id == "winner"


def test_power_breaks_equal_skew_tie():
    candidates = [
        observation("higher-power", skew=0.02, rlm=0.95, power=1.4),
        observation("lower-power", skew=0.02, rlm=0.95, power=1.1),
    ]

    best = select_incumbent(candidates, objective())

    assert best.candidate_id == "lower-power"
```

- [ ] **步骤 2：运行目标测试并验证 RED / Step 2: Run objective tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_objectives.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.objectives'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.objectives'`.

- [ ] **步骤 3：实现指标约束与排名 / Step 3: Implement metric constraints and ranking**

新建 `src/vcsel_ai_tuning/objectives.py`：
Create `src/vcsel_ai_tuning/objectives.py`:

```python
from __future__ import annotations

from .config import ObjectiveConfig
from .domain import ConstraintResult, MetricValue, Observation


def evaluate_metric_constraints(
    metrics: dict[str, MetricValue],
    objective: ObjectiveConfig,
) -> list[ConstraintResult]:
    results: list[ConstraintResult] = []
    for spec in objective.metric_constraints:
        metric = metrics.get(spec.metric)
        if metric is None or not metric.valid:
            results.append(
                ConstraintResult(
                    name=f"metric.{spec.metric}",
                    passed=False,
                    limit=f"{spec.operator} {spec.threshold}",
                    reason="metric missing or invalid",
                )
            )
            continue
        passed = (
            metric.value >= spec.threshold
            if spec.operator == ">="
            else metric.value <= spec.threshold
        )
        results.append(
            ConstraintResult(
                name=f"metric.{spec.metric}",
                passed=passed,
                value=metric.value,
                limit=f"{spec.operator} {spec.threshold}",
                reason=None if passed else "outcome constraint failed",
            )
        )
    return results


def _directional(value: float, direction: str) -> float:
    return value if direction == "minimize" else -value


def _ranking_key(observation: Observation, objective: ObjectiveConfig) -> tuple[float, ...]:
    primary = observation.metrics[objective.primary_metric].value
    values = [_directional(primary, objective.direction)]
    for tie_breaker in objective.tie_breakers:
        metric = observation.metrics[tie_breaker.metric]
        values.append(_directional(metric.value, tie_breaker.direction))
    return tuple(values)


def select_incumbent(
    observations: list[Observation],
    objective: ObjectiveConfig,
) -> Observation:
    eligible: list[Observation] = []
    for observation in observations:
        if not observation.feasible:
            continue
        constraints = evaluate_metric_constraints(observation.metrics, objective)
        if not all(result.passed for result in constraints):
            continue
        required = [objective.primary_metric] + [
            tie_breaker.metric for tie_breaker in objective.tie_breakers
        ]
        if any(
            name not in observation.metrics or not observation.metrics[name].valid
            for name in required
        ):
            continue
        eligible.append(observation)
    if not eligible:
        raise ValueError("no valid and feasible observations")
    return min(eligible, key=lambda item: _ranking_key(item, objective))
```

- [ ] **步骤 4：运行目标测试并验证 GREEN / Step 4: Run objective tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_objectives.py -v
```

预期：3 个测试通过。
Expected: 3 tests pass.

- [ ] **步骤 5：提交目标约束与排名实现 / Step 5: Commit objective constraints and ranking**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/objectives.py tests_py/ai_tuning/test_objectives.py
git commit -m "feat: rank only valid feasible observations"
```

## 任务 10：添加崩溃标记与操作员确认的恢复机制 / Task 10: Add Crash Markers and Operator-Acknowledged Recovery

**文件 / Files:**
- 新建：`src/vcsel_ai_tuning/recovery.py`
- Create: `src/vcsel_ai_tuning/recovery.py`
- 新建：`tests_py/ai_tuning/test_recovery.py`
- Create: `tests_py/ai_tuning/test_recovery.py`

- [ ] **步骤 1：编写会失败的恢复注册表测试 / Step 1: Write failing recovery-registry tests**

新建 `tests_py/ai_tuning/test_recovery.py`：
Create `tests_py/ai_tuning/test_recovery.py`:

```python
from pathlib import Path

import pytest

from vcsel_ai_tuning.backends import FakeBackend
from vcsel_ai_tuning.domain import Candidate
from vcsel_ai_tuning.recovery import (
    RecoveryRegistry,
    RecoveryRequired,
    recover_backend,
)


def test_recovery_marker_survives_until_operator_acknowledges(tmp_path: Path):
    registry = RecoveryRegistry(tmp_path / "recovery")
    registry.arm("campaign-1", "run-1")

    assert registry.get("campaign-1").run_id == "run-1"

    registry.mark_recovered("campaign-1", {"model": "fake"}, {})
    registry.acknowledge("campaign-1")

    assert registry.get("campaign-1") is None


def test_recovery_forces_safe_off_and_stays_blocked(tmp_path: Path):
    registry = RecoveryRegistry(tmp_path / "recovery")
    registry.arm("campaign-1", "run-1")
    backend = FakeBackend()
    candidate = Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )
    backend.apply(candidate, candidate.requested)

    with pytest.raises(RecoveryRequired, match="acknowledgement"):
        recover_backend(backend, registry, "campaign-1")

    record = registry.get("campaign-1")
    assert record is not None and record.recovered_safe_off is True
    assert backend.is_safe is True
    assert registry.can_arm("campaign-1") is False
```

- [ ] **步骤 2：运行恢复测试并验证 RED / Step 2: Run recovery tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_recovery.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.recovery'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.recovery'`.

- [ ] **步骤 3：实现原子化恢复标记与安全恢复 / Step 3: Implement atomic recovery markers and safe recovery**

新建 `src/vcsel_ai_tuning/recovery.py`：
Create `src/vcsel_ai_tuning/recovery.py`:

```python
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .backends import ExperimentBackend


class RecoveryRequired(RuntimeError):
    pass


class RecoveryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    campaign_id: str
    run_id: str
    armed_at: datetime
    recovered_safe_off: bool = False
    identity: dict[str, str] | None = None
    readback: dict[str, float] | None = None


class RecoveryRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(campaign_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", campaign_id).strip("-")
        if not safe:
            raise ValueError("campaign_id has no safe filename characters")
        return safe

    def _path(self, campaign_id: str) -> Path:
        return self.root / f"{self._safe_name(campaign_id)}.json"

    def _write(self, record: RecoveryRecord) -> None:
        path = self._path(record.campaign_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def arm(self, campaign_id: str, run_id: str) -> None:
        if self.get(campaign_id) is not None:
            raise RecoveryRequired(
                f"campaign {campaign_id} has an unacknowledged recovery marker"
            )
        self._write(
            RecoveryRecord(
                campaign_id=campaign_id,
                run_id=run_id,
                armed_at=datetime.now(timezone.utc),
            )
        )

    def get(self, campaign_id: str) -> RecoveryRecord | None:
        path = self._path(campaign_id)
        if not path.is_file():
            return None
        return RecoveryRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def mark_recovered(
        self,
        campaign_id: str,
        identity: dict[str, str],
        readback: dict[str, float],
    ) -> None:
        record = self.get(campaign_id)
        if record is None:
            raise FileNotFoundError(campaign_id)
        self._write(
            record.model_copy(
                update={
                    "recovered_safe_off": True,
                    "identity": identity,
                    "readback": readback,
                }
            )
        )

    def acknowledge(self, campaign_id: str) -> None:
        record = self.get(campaign_id)
        if record is None:
            raise FileNotFoundError(campaign_id)
        if not record.recovered_safe_off:
            raise RecoveryRequired("safe-off recovery has not completed")
        self._path(campaign_id).unlink()

    def clear_completed(self, campaign_id: str, run_id: str) -> None:
        record = self.get(campaign_id)
        if record is None or record.run_id != run_id:
            raise RecoveryRequired("recovery marker does not match completed run")
        self._path(campaign_id).unlink()

    def can_arm(self, campaign_id: str) -> bool:
        return self.get(campaign_id) is None


def recover_backend(
    backend: ExperimentBackend,
    registry: RecoveryRegistry,
    campaign_id: str,
) -> None:
    record = registry.get(campaign_id)
    if record is None:
        return
    identity: dict[str, str] = {}
    readback: dict[str, float] = {}
    try:
        identity = backend.identify()
        readback = backend.readback()
    finally:
        try:
            backend.abort()
        finally:
            backend.safe_off()
    registry.mark_recovered(campaign_id, identity, readback)
    raise RecoveryRequired(
        f"campaign {campaign_id} requires operator acknowledgement after safe-off"
    )
```

- [ ] **步骤 4：运行恢复测试并验证 GREEN / Step 4: Run recovery tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_recovery.py -v
```

预期：2 个测试通过。
Expected: 2 tests pass.

- [ ] **步骤 5：提交崩溃恢复实现 / Step 5: Commit crash recovery**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/recovery.py tests_py/ai_tuning/test_recovery.py
git commit -m "feat: require acknowledged recovery after interrupted runs"
```

## 任务 11：执行实验状态机并回滚故障 / Task 11: Execute the Experiment State Machine and Roll Back Faults

**文件 / Files:**
- 新建：`src/vcsel_ai_tuning/engine.py`
- Create: `src/vcsel_ai_tuning/engine.py`
- 新建：`tests_py/ai_tuning/test_engine.py`
- Create: `tests_py/ai_tuning/test_engine.py`

- [ ] **步骤 1：编写会失败的正常路径与故障注入测试 / Step 1: Write failing normal-path and fault-injection tests**

新建 `tests_py/ai_tuning/test_engine.py`：
Create `tests_py/ai_tuning/test_engine.py`:

```python
import json
from pathlib import Path

import pytest

from vcsel_ai_tuning.backends import FakeBackend
from vcsel_ai_tuning.config import CampaignConfig
from vcsel_ai_tuning.domain import Candidate, RunState
from vcsel_ai_tuning.engine import ExperimentEngine
from vcsel_ai_tuning.persistence import ArtifactStore, RunRepository, SqliteStore
from vcsel_ai_tuning.recovery import RecoveryRegistry


def campaign():
    return CampaignConfig.model_validate(
        {
            "campaign_id": "engine-test",
            "backend": {"kind": "fake", "dry_run": True},
            "optimizer": {"kind": "random", "budget": 2, "seed": 1},
            "parameters": [
                {
                    "name": "bias_ma",
                    "unit": "mA",
                    "lower": 4.0,
                    "upper": 7.0,
                    "quantization": 0.1,
                    "max_step": 1.0,
                    "readback_tolerance": 0.02,
                },
                {
                    "name": "skew_control_ui",
                    "unit": "UI",
                    "lower": -0.1,
                    "upper": 0.1,
                    "quantization": 0.001,
                    "max_step": 0.05,
                    "readback_tolerance": 0.002,
                },
            ],
            "safe_baseline": {"bias_ma": 5.0, "skew_control_ui": 0.05},
            "objective": {
                "primary_metric": "skew_pp_ui",
                "direction": "minimize",
                "metric_constraints": [
                    {"metric": "rlm_proxy", "operator": ">=", "threshold": 0.9}
                ],
                "tie_breakers": [
                    {"metric": "power_mw", "direction": "minimize"}
                ],
            },
        }
    )


def make_engine(tmp_path: Path, backend: FakeBackend):
    repository = RunRepository(
        SqliteStore(tmp_path / "state.sqlite"),
        ArtifactStore(tmp_path / "artifacts"),
    )
    return ExperimentEngine(
        campaign=campaign(),
        backend=backend,
        repository=repository,
        recovery=RecoveryRegistry(tmp_path / "recovery"),
        run_id_factory=lambda: "run-1",
    )


def candidate():
    return Candidate(
        candidate_id="candidate-0000",
        requested={"bias_ma": 5.0, "skew_control_ui": 0.04},
    )


def test_normal_path_persists_exact_state_order(tmp_path: Path):
    engine = make_engine(tmp_path, FakeBackend())

    observation = engine.execute(candidate())

    events_path = (
        tmp_path
        / "artifacts"
        / "campaigns"
        / "engine-test"
        / "runs"
        / "run-1"
        / "events.jsonl"
    )
    states = [json.loads(line)["state"] for line in events_path.read_text().splitlines()]
    assert observation.valid is True
    assert observation.feasible is True
    assert states == [
        "IDLE",
        "PREFLIGHT",
        "ARMED",
        "APPLY",
        "READBACK",
        "SETTLE",
        "MEASURE_FAST",
        "VALIDATE",
        "COMMIT",
        "COMPLETE",
    ]


@pytest.mark.parametrize("fail_at", ["apply", "readback", "wait_ready", "measure_fast"])
def test_faults_roll_back_safe_off_and_return_invalid_observation(
    tmp_path: Path, fail_at: str
):
    backend = FakeBackend(fail_at=fail_at)
    engine = make_engine(tmp_path, backend)

    observation = engine.execute(candidate())

    assert observation.state is RunState.ABORTED
    assert observation.valid is False
    assert backend.is_safe is True
    assert "abort" in backend.call_order
    assert "safe_off" in backend.call_order


def test_readback_mismatch_is_never_measured(tmp_path: Path):
    backend = FakeBackend(readback_offsets={"bias_ma": 0.05})
    engine = make_engine(tmp_path, backend)

    observation = engine.execute(candidate())

    assert observation.valid is False
    assert "measure_fast" not in backend.call_order
    assert backend.is_safe is True


def test_persistence_failure_safe_off_and_propagates(tmp_path: Path, monkeypatch):
    backend = FakeBackend()
    engine = make_engine(tmp_path, backend)

    def fail_commit(*args, **kwargs):
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(engine.repository, "commit_run", fail_commit)

    with pytest.raises(RuntimeError, match="persistence failed"):
        engine.execute(candidate())

    assert backend.is_safe is True
```

- [ ] **步骤 2：运行引擎测试并验证 RED / Step 2: Run engine tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_engine.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.engine'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.engine'`.

- [ ] **步骤 3：实现状态转换、校验与回滚 / Step 3: Implement state transitions, validation, and rollback**

新建 `src/vcsel_ai_tuning/engine.py`：
Create `src/vcsel_ai_tuning/engine.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from .backends import ExperimentBackend
from .config import CampaignConfig
from .domain import (
    AppliedSettings,
    BackendResult,
    Candidate,
    Observation,
    RunEvent,
    RunState,
)
from .objectives import evaluate_metric_constraints
from .persistence import RunRepository
from .recovery import RecoveryRegistry, RecoveryRequired
from .safety import SafetyPolicy


class InvalidMeasurement(RuntimeError):
    pass


class ExperimentEngine:
    def __init__(
        self,
        campaign: CampaignConfig,
        backend: ExperimentBackend,
        repository: RunRepository,
        recovery: RecoveryRegistry,
        run_id_factory: Callable[[], str] | None = None,
    ):
        self.campaign = campaign
        self.backend = backend
        self.repository = repository
        self.recovery = recovery
        self.safety = SafetyPolicy(campaign.parameters)
        self.reference = dict(campaign.safe_baseline)
        self.run_id_factory = run_id_factory or (lambda: uuid4().hex)

    def execute(self, candidate: Candidate) -> Observation:
        run_id = self.run_id_factory()
        started = datetime.now(timezone.utc)
        events = [RunEvent(state=RunState.IDLE)]
        constraints = []
        metrics = {}
        raw_artifacts: dict[str, tuple[bytes, str]] = {}
        applied: dict[str, float] = {}
        readback: dict[str, float] = {}
        marker_armed = False

        def transition(state: RunState, **detail: object) -> None:
            events.append(RunEvent(state=state, detail=detail))

        if not self.recovery.can_arm(self.campaign.campaign_id):
            raise RecoveryRequired("campaign has an unacknowledged recovery marker")

        try:
            transition(RunState.PREFLIGHT)
            candidate_decision = self.safety.evaluate_candidate(candidate, self.reference)
            constraints.extend(candidate_decision.constraints)
            candidate_decision.require("candidate")
            applied = dict(candidate_decision.applied)

            self.recovery.arm(self.campaign.campaign_id, run_id)
            marker_armed = True
            transition(RunState.ARMED)
            transition(RunState.APPLY)
            self.backend.apply(candidate, applied)
            transition(RunState.READBACK)
            readback = self.backend.readback()
            readback_decision = self.safety.evaluate_readback(applied, readback)
            constraints.extend(readback_decision.constraints)
            readback_decision.require("read-back")

            transition(RunState.SETTLE)
            self.backend.wait_ready()
            transition(RunState.MEASURE_FAST)
            fast: BackendResult = self.backend.measure_fast()
            metrics.update(fast.metrics)
            raw_artifacts.update(
                {
                    name: (data, "application/octet-stream")
                    for name, data in fast.raw_artifacts.items()
                }
            )
            transition(RunState.VALIDATE)
            if not metrics or any(not metric.valid for metric in metrics.values()):
                raise InvalidMeasurement("fast measurement is empty or invalid")
            metric_constraints = evaluate_metric_constraints(
                metrics, self.campaign.objective
            )
            constraints.extend(metric_constraints)

            if self.campaign.slow_measurement_for_best:
                transition(RunState.MEASURE_SLOW)
                slow = self.backend.measure_slow()
                if slow is not None:
                    metrics.update(slow.metrics)
                    raw_artifacts.update(
                        {
                            name: (data, "application/octet-stream")
                            for name, data in slow.raw_artifacts.items()
                        }
                    )

            transition(RunState.COMMIT)
            transition(RunState.COMPLETE)
            observation = Observation(
                campaign_id=self.campaign.campaign_id,
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                state=RunState.COMPLETE,
                settings=AppliedSettings(
                    requested=candidate.requested,
                    applied=applied,
                    readback=readback,
                ),
                metrics=metrics,
                constraints=constraints,
                valid=True,
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )
        except Exception as error:
            transition(RunState.ROLLBACK, error=type(error).__name__)
            try:
                self.backend.abort()
            finally:
                self.backend.safe_off()
            transition(RunState.SAFE_OFF)
            transition(RunState.ABORTED)
            observation = Observation(
                campaign_id=self.campaign.campaign_id,
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                state=RunState.ABORTED,
                settings=AppliedSettings(
                    requested=candidate.requested,
                    applied=applied,
                    readback=readback,
                ),
                metrics=metrics,
                constraints=constraints,
                valid=False,
                failure_reason=f"{type(error).__name__}: {error}",
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )
            saved = self._persist_or_safe_off(
                candidate, observation, events, raw_artifacts
            )
            if marker_armed:
                self.recovery.clear_completed(self.campaign.campaign_id, run_id)
            return saved

        saved = self._persist_or_safe_off(candidate, observation, events, raw_artifacts)
        if marker_armed:
            self.recovery.clear_completed(self.campaign.campaign_id, run_id)
        if saved.feasible:
            self.reference = dict(saved.settings.readback)
        return saved

    def _persist_or_safe_off(
        self,
        candidate: Candidate,
        observation: Observation,
        events: list[RunEvent],
        raw_artifacts: dict[str, tuple[bytes, str]],
    ) -> Observation:
        try:
            return self.repository.commit_run(
                candidate=candidate,
                observation=observation,
                events=events,
                raw_artifacts=raw_artifacts,
            )
        except Exception:
            try:
                self.backend.abort()
            finally:
                self.backend.safe_off()
            raise
```

- [ ] **步骤 4：运行引擎测试并验证 GREEN / Step 4: Run engine tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_engine.py -v
```

预期：7 个测试通过，分别覆盖正常路径、四个注入故障的后端阶段、读回不匹配和持久化失败。
Expected: 7 tests pass: normal path, four injected backend stages, read-back mismatch, and persistence failure.

- [ ] **步骤 5：提交实验状态机 / Step 5: Commit the experiment state machine**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/engine.py tests_py/ai_tuning/test_engine.py
git commit -m "feat: execute safe tuning experiment state machine"
```

## 任务 12：生成确定性的网格、随机与 Sobol 候选点 / Task 12: Generate Deterministic Grid, Random, and Sobol Candidates

**文件 / Files:**
- 新建：`src/vcsel_ai_tuning/search.py`
- Create: `src/vcsel_ai_tuning/search.py`
- 新建：`tests_py/ai_tuning/test_search.py`
- Create: `tests_py/ai_tuning/test_search.py`

- [ ] **步骤 1：编写会失败的候选点生成测试 / Step 1: Write failing candidate-generation tests**

新建 `tests_py/ai_tuning/test_search.py`：
Create `tests_py/ai_tuning/test_search.py`:

```python
import pytest

from vcsel_ai_tuning.config import CampaignConfig
from vcsel_ai_tuning.search import SearchSpaceExhausted, generate_candidates


def campaign(kind="random", budget=8, seed=17):
    return CampaignConfig.model_validate(
        {
            "campaign_id": "search-test",
            "backend": {"kind": "fake", "dry_run": True},
            "optimizer": {"kind": kind, "budget": budget, "seed": seed},
            "parameters": [
                {
                    "name": "bias_ma",
                    "unit": "mA",
                    "lower": 4.0,
                    "upper": 6.0,
                    "quantization": 0.1,
                    "max_step": 1.0,
                    "grid_values": [4.0, 5.0, 6.0],
                },
                {
                    "name": "skew_control_ui",
                    "unit": "UI",
                    "lower": -0.1,
                    "upper": 0.1,
                    "quantization": 0.01,
                    "max_step": 0.1,
                    "grid_values": [-0.1, 0.0, 0.1],
                },
            ],
            "safe_baseline": {"bias_ma": 5.0, "skew_control_ui": 0.05},
            "objective": {
                "primary_metric": "skew_pp_ui",
                "direction": "minimize",
            },
        }
    )


@pytest.mark.parametrize("kind", ["random", "sobol"])
def test_seeded_generators_are_deterministic_unique_and_start_at_baseline(kind):
    first = generate_candidates(campaign(kind=kind))
    second = generate_candidates(campaign(kind=kind))

    assert first == second
    assert len(first) == 8
    assert first[0].requested == {"bias_ma": 5.0, "skew_control_ui": 0.05}
    assert len({tuple(sorted(item.requested.items())) for item in first}) == 8
    assert all(4.0 <= item.requested["bias_ma"] <= 6.0 for item in first)
    assert all(-0.1 <= item.requested["skew_control_ui"] <= 0.1 for item in first)


def test_grid_obeys_budget_and_stable_candidate_ids():
    candidates = generate_candidates(campaign(kind="grid", budget=5))

    assert len(candidates) == 5
    assert [item.candidate_id for item in candidates] == [
        "candidate-0000",
        "candidate-0001",
        "candidate-0002",
        "candidate-0003",
        "candidate-0004",
    ]


def test_grid_raises_when_budget_exceeds_unique_quantized_space():
    with pytest.raises(SearchSpaceExhausted, match="requested 20"):
        generate_candidates(campaign(kind="grid", budget=20))
```

- [ ] **步骤 2：运行搜索测试并验证 RED / Step 2: Run search tests and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_search.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.search'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.search'`.

- [ ] **步骤 3：实现有界且量化的候选点生成器 / Step 3: Implement bounded and quantized candidate generators**

新建 `src/vcsel_ai_tuning/search.py`：
Create `src/vcsel_ai_tuning/search.py`:

```python
from __future__ import annotations

from itertools import product
from math import ceil, log2

import numpy as np
from scipy.stats import qmc

from .config import CampaignConfig
from .domain import Candidate, ParameterSpec


class SearchSpaceExhausted(RuntimeError):
    pass


def _bounds(spec: ParameterSpec, baseline: float) -> tuple[float, float]:
    if spec.max_step is None:
        return spec.lower, spec.upper
    return (
        max(spec.lower, baseline - spec.max_step),
        min(spec.upper, baseline + spec.max_step),
    )


def _prepare_vector(
    campaign: CampaignConfig,
    values: list[float] | tuple[float, ...] | np.ndarray,
) -> dict[str, float]:
    return {
        parameter.name: parameter.prepare(float(value))
        for parameter, value in zip(campaign.parameters, values, strict=True)
    }


def _key(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(values.items()))


def _finalize(
    campaign: CampaignConfig,
    requested_values: list[dict[str, float]],
) -> list[Candidate]:
    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    baseline = {
        parameter.name: parameter.prepare(campaign.safe_baseline[parameter.name])
        for parameter in campaign.parameters
    }
    for values in [baseline, *requested_values]:
        item_key = _key(values)
        if item_key in seen:
            continue
        seen.add(item_key)
        unique.append(values)
        if len(unique) == campaign.optimizer.budget:
            break
    if len(unique) < campaign.optimizer.budget:
        raise SearchSpaceExhausted(
            f"requested {campaign.optimizer.budget} unique candidates, found {len(unique)}"
        )
    return [
        Candidate(candidate_id=f"candidate-{index:04d}", requested=values)
        for index, values in enumerate(unique)
    ]


def _grid(campaign: CampaignConfig) -> list[dict[str, float]]:
    axes: list[list[float]] = []
    for parameter in campaign.parameters:
        lower, upper = _bounds(
            parameter, campaign.safe_baseline[parameter.name]
        )
        values = [
            parameter.prepare(value)
            for value in parameter.grid_values
            if lower <= value <= upper
        ]
        if not values:
            raise SearchSpaceExhausted(
                f"parameter {parameter.name} has no grid values in the active bounds"
            )
        axes.append(values)
    return [_prepare_vector(campaign, values) for values in product(*axes)]


def _random(campaign: CampaignConfig) -> list[dict[str, float]]:
    rng = np.random.default_rng(campaign.optimizer.seed)
    lows, highs = zip(
        *[
            _bounds(parameter, campaign.safe_baseline[parameter.name])
            for parameter in campaign.parameters
        ],
        strict=True,
    )
    count = max(campaign.optimizer.budget * 100, 256)
    samples = rng.uniform(np.asarray(lows), np.asarray(highs), size=(count, len(lows)))
    return [_prepare_vector(campaign, row) for row in samples]


def _sobol(campaign: CampaignConfig) -> list[dict[str, float]]:
    dimensions = len(campaign.parameters)
    desired_pool = max(campaign.optimizer.budget * 4, 8)
    exponent = ceil(log2(desired_pool))
    engine = qmc.Sobol(d=dimensions, scramble=True, seed=campaign.optimizer.seed)
    unit_samples = engine.random_base2(exponent)
    bounds = [
        _bounds(parameter, campaign.safe_baseline[parameter.name])
        for parameter in campaign.parameters
    ]
    lows = np.asarray([bound[0] for bound in bounds])
    highs = np.asarray([bound[1] for bound in bounds])
    samples = qmc.scale(unit_samples, lows, highs)
    return [_prepare_vector(campaign, row) for row in samples]


def generate_candidates(campaign: CampaignConfig) -> list[Candidate]:
    if campaign.optimizer.kind == "grid":
        values = _grid(campaign)
    elif campaign.optimizer.kind == "random":
        values = _random(campaign)
    else:
        values = _sobol(campaign)
    return _finalize(campaign, values)
```

- [ ] **步骤 4：运行搜索测试并验证 GREEN / Step 4: Run search tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_search.py -v
```

预期：4 个测试通过，且不出现 SciPy 非 2 次幂 Sobol 警告。
Expected: 4 tests pass without SciPy non-power-of-two Sobol warnings.

- [ ] **步骤 5：提交确定性搜索实现 / Step 5: Commit deterministic search**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/search.py tests_py/ai_tuning/test_search.py
git commit -m "feat: generate deterministic tuning candidates"
```

## 任务 13：运行完整的 Fake 或 Replay 活动 / Task 13: Run a Complete Fake or Replay Campaign

**文件 / Files:**
- 修改：`src/vcsel_ai_tuning/persistence.py`
- Modify: `src/vcsel_ai_tuning/persistence.py`
- 新建：`src/vcsel_ai_tuning/runner.py`
- Create: `src/vcsel_ai_tuning/runner.py`
- 新建：`tests_py/ai_tuning/test_runner.py`
- Create: `tests_py/ai_tuning/test_runner.py`

- [ ] **步骤 1：编写会失败的端到端运行器测试 / Step 1: Write a failing end-to-end runner test**

新建 `tests_py/ai_tuning/test_runner.py`：
Create `tests_py/ai_tuning/test_runner.py`:

```python
from pathlib import Path

from vcsel_ai_tuning.config import CalibrationManifest, CampaignConfig
from vcsel_ai_tuning.runner import CampaignRunner


def test_fake_campaign_runs_budget_persists_provenance_and_safe_off(tmp_path: Path):
    campaign = CampaignConfig.model_validate(
        {
            "campaign_id": "runner-test",
            "backend": {"kind": "fake", "dry_run": True},
            "optimizer": {"kind": "sobol", "budget": 4, "seed": 11},
            "parameters": [
                {
                    "name": "bias_ma",
                    "unit": "mA",
                    "lower": 4.0,
                    "upper": 7.0,
                    "quantization": 0.1,
                    "max_step": 3.0,
                    "readback_tolerance": 0.02,
                },
                {
                    "name": "skew_control_ui",
                    "unit": "UI",
                    "lower": -0.1,
                    "upper": 0.1,
                    "quantization": 0.001,
                    "max_step": 0.2,
                    "readback_tolerance": 0.002,
                },
            ],
            "safe_baseline": {"bias_ma": 5.0, "skew_control_ui": 0.05},
            "objective": {
                "primary_metric": "skew_pp_ui",
                "direction": "minimize",
                "metric_constraints": [
                    {"metric": "rlm_proxy", "operator": ">=", "threshold": 0.9}
                ],
                "tie_breakers": [
                    {"metric": "power_mw", "direction": "minimize"}
                ],
            },
        }
    )
    manifest = CalibrationManifest(
        manifest_id="fake-manifest-v1",
        device_id="fake-vcsel",
        temperature_c=25.0,
    )
    runner = CampaignRunner(
        campaign=campaign,
        manifest=manifest,
        database_path=tmp_path / "state.sqlite",
        artifact_root=tmp_path / "artifacts",
        recovery_root=tmp_path / "recovery",
    )

    observations = runner.run()

    assert len(observations) == 4
    assert observations[0].settings.requested == campaign.safe_baseline
    assert runner.backend.is_safe is True
    assert (tmp_path / "artifacts" / "campaigns" / "runner-test" / "campaign.json").is_file()
    assert (tmp_path / "artifacts" / "campaigns" / "runner-test" / "calibration_manifest.json").is_file()
    assert len(runner.store.list_observations("runner-test")) == 4
```

- [ ] **步骤 2：运行运行器测试并验证 RED / Step 2: Run runner test and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_runner.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.runner'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.runner'`.

- [ ] **步骤 3：扩展持久化功能以保存活动溯源信息 / Step 3: Extend persistence with campaign provenance**

在 `src/vcsel_ai_tuning/persistence.py` 中添加以下导入：
In `src/vcsel_ai_tuning/persistence.py`, add these imports:

```python
from .config import CalibrationManifest, CampaignConfig
```

在现有 `observations` 表创建逻辑之后，将以下表创建逻辑添加到 `SqliteStore._initialize()` 中：
Add this table creation inside `SqliteStore._initialize()` after the existing `observations` table creation:

```python
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    campaign_json TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                )
                """
            )
```

将以下方法添加到 `SqliteStore`：
Add these methods to `SqliteStore`:

```python
    def save_campaign(
        self,
        campaign: CampaignConfig,
        manifest: CalibrationManifest,
    ) -> None:
        campaign_json = campaign.model_dump_json()
        manifest_json = manifest.model_dump_json()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT campaign_json, manifest_json
                FROM campaigns
                WHERE campaign_id = ?
                """,
                (campaign.campaign_id,),
            ).fetchone()
            if existing is not None:
                if existing != (campaign_json, manifest_json):
                    raise ValueError(
                        f"campaign {campaign.campaign_id} already exists with different provenance"
                    )
                return
            connection.execute(
                """
                INSERT INTO campaigns
                    (campaign_id, campaign_json, manifest_json)
                VALUES (?, ?, ?)
                """,
                (
                    campaign.campaign_id,
                    campaign_json,
                    manifest_json,
                ),
            )

    def load_campaign(
        self, campaign_id: str
    ) -> tuple[CampaignConfig, CalibrationManifest]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT campaign_json, manifest_json
                FROM campaigns
                WHERE campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return (
            CampaignConfig.model_validate_json(row[0]),
            CalibrationManifest.model_validate_json(row[1]),
        )
```

将以下方法添加到 `RunRepository`：
Add this method to `RunRepository`:

```python
    def initialize_campaign(
        self,
        campaign: CampaignConfig,
        manifest: CalibrationManifest,
    ) -> None:
        self.store.save_campaign(campaign, manifest)
        campaign_root = self.artifacts.root / "campaigns" / campaign.campaign_id
        campaign_root.mkdir(parents=True, exist_ok=True)
        files = {
            "campaign.json": campaign.model_dump(mode="json"),
            "calibration_manifest.json": manifest.model_dump(mode="json"),
        }
        for name, value in files.items():
            destination = campaign_root / name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
            temporary.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary, destination)
```

- [ ] **步骤 4：实现活动编排与后端选择 / Step 4: Implement campaign orchestration and backend selection**

新建 `src/vcsel_ai_tuning/runner.py`：
Create `src/vcsel_ai_tuning/runner.py`:

```python
from __future__ import annotations

from pathlib import Path

from .backends import (
    BackendUnavailable,
    ExperimentBackend,
    FakeBackend,
    ReplayBackend,
)
from .config import CalibrationManifest, CampaignConfig
from .domain import Observation
from .engine import ExperimentEngine
from .persistence import ArtifactStore, RunRepository, SqliteStore
from .recovery import RecoveryRegistry, recover_backend
from .search import generate_candidates


def build_backend(campaign: CampaignConfig) -> ExperimentBackend:
    if campaign.backend.kind == "fake":
        return FakeBackend()
    if campaign.backend.kind == "replay":
        fixture_path = campaign.backend.options.get("fixture_path")
        if not isinstance(fixture_path, str):
            raise ValueError("replay backend requires string option fixture_path")
        return ReplayBackend.from_jsonl(Path(fixture_path))
    raise BackendUnavailable(
        f"{campaign.backend.kind} backend is interface-only in Milestone 1"
    )


class CampaignRunner:
    def __init__(
        self,
        campaign: CampaignConfig,
        manifest: CalibrationManifest,
        database_path: Path,
        artifact_root: Path,
        recovery_root: Path,
        backend: ExperimentBackend | None = None,
    ):
        self.campaign = campaign
        self.manifest = manifest
        self.store = SqliteStore(database_path)
        self.repository = RunRepository(self.store, ArtifactStore(artifact_root))
        self.recovery = RecoveryRegistry(recovery_root)
        self.backend = backend or build_backend(campaign)

    def run(self) -> list[Observation]:
        if not self.recovery.can_arm(self.campaign.campaign_id):
            recover_backend(self.backend, self.recovery, self.campaign.campaign_id)
        self.repository.initialize_campaign(self.campaign, self.manifest)
        self.backend.identify()
        self.backend.configure(dict(self.campaign.backend.options))
        candidates = generate_candidates(self.campaign)
        run_ids = iter(f"run-{index:04d}" for index in range(len(candidates)))
        engine = ExperimentEngine(
            campaign=self.campaign,
            backend=self.backend,
            repository=self.repository,
            recovery=self.recovery,
            run_id_factory=lambda: next(run_ids),
        )
        observations: list[Observation] = []
        try:
            for candidate in candidates:
                observations.append(engine.execute(candidate))
        finally:
            self.backend.safe_off()
        return observations
```

- [ ] **步骤 5：运行运行器与持久化测试并验证 GREEN / Step 5: Run runner and persistence tests and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_persistence.py tests_py/ai_tuning/test_runner.py -v
```

预期：3 个测试通过，包括两个持久化测试和一个活动运行器测试。
Expected: 3 tests pass: two persistence tests and one campaign-runner test.

- [ ] **步骤 6：提交活动编排实现 / Step 6: Commit campaign orchestration**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/persistence.py src/vcsel_ai_tuning/runner.py tests_py/ai_tuning/test_runner.py
git commit -m "feat: run complete fake and replay campaigns"
```

## 任务 14：生成确定性的双语基线报告 / Task 14: Generate a Deterministic Bilingual Baseline Report

**文件 / Files:**
- 新建：`src/vcsel_ai_tuning/report.py`
- Create: `src/vcsel_ai_tuning/report.py`
- 新建：`tests_py/ai_tuning/test_report.py`
- Create: `tests_py/ai_tuning/test_report.py`

- [ ] **步骤 1：编写会失败的双语报告测试 / Step 1: Write a failing bilingual-report test**

新建 `tests_py/ai_tuning/test_report.py`：
Create `tests_py/ai_tuning/test_report.py`:

```python
from pathlib import Path

from vcsel_ai_tuning.config import CalibrationManifest, CampaignConfig
from vcsel_ai_tuning.report import build_baseline_report, write_report
from vcsel_ai_tuning.runner import CampaignRunner


def test_report_is_bilingual_deterministic_and_does_not_invent_tdecq(tmp_path: Path):
    campaign = CampaignConfig.model_validate(
        {
            "campaign_id": "report-test",
            "backend": {"kind": "fake", "dry_run": True},
            "optimizer": {"kind": "sobol", "budget": 4, "seed": 11},
            "parameters": [
                {
                    "name": "bias_ma",
                    "unit": "mA",
                    "lower": 4.0,
                    "upper": 7.0,
                    "quantization": 0.1,
                    "max_step": 3.0,
                },
                {
                    "name": "skew_control_ui",
                    "unit": "UI",
                    "lower": -0.1,
                    "upper": 0.1,
                    "quantization": 0.001,
                    "max_step": 0.2,
                },
            ],
            "safe_baseline": {"bias_ma": 5.0, "skew_control_ui": 0.05},
            "objective": {
                "primary_metric": "skew_pp_ui",
                "direction": "minimize",
                "metric_constraints": [
                    {"metric": "rlm_proxy", "operator": ">=", "threshold": 0.9}
                ],
                "tie_breakers": [
                    {"metric": "power_mw", "direction": "minimize"}
                ],
            },
        }
    )
    manifest = CalibrationManifest(
        manifest_id="fake-v1",
        device_id="fake-vcsel",
        temperature_c=25.0,
    )
    runner = CampaignRunner(
        campaign,
        manifest,
        tmp_path / "state.sqlite",
        tmp_path / "artifacts",
        tmp_path / "recovery",
    )
    observations = runner.run()

    first = build_baseline_report(campaign, manifest, observations)
    second = build_baseline_report(campaign, manifest, observations)
    output = tmp_path / "report.md"
    write_report(output, first)

    assert first == second
    assert "基线比较报告" in first
    assert "Baseline Comparison Report" in first
    assert "最佳合格候选点 / Best Feasible Candidate" in first
    assert "tdecq_db" not in first
    assert output.read_text(encoding="utf-8") == first
```

- [ ] **步骤 2：运行报告测试并验证 RED / Step 2: Run report test and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_report.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.report'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.report'`.

- [ ] **步骤 3：实现确定性的双语报告 / Step 3: Implement deterministic bilingual reporting**

新建 `src/vcsel_ai_tuning/report.py`：
Create `src/vcsel_ai_tuning/report.py`:

```python
from __future__ import annotations

from pathlib import Path

from .config import CalibrationManifest, CampaignConfig
from .domain import Observation
from .objectives import evaluate_metric_constraints, select_incumbent


def _format(value: float) -> str:
    return f"{value:.9g}"


def _metric_rows(baseline: Observation, best: Observation) -> list[str]:
    rows = [
        "| 指标 / Metric | 基线 / Baseline | 最佳值 / Best | 变化 / Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(set(baseline.metrics) & set(best.metrics)):
        base = baseline.metrics[name]
        current = best.metrics[name]
        if not base.valid or not current.valid or base.unit != current.unit:
            continue
        rows.append(
            f"| `{name}` ({base.unit}) | {_format(base.value)} | "
            f"{_format(current.value)} | {_format(current.value - base.value)} |"
        )
    return rows


def build_baseline_report(
    campaign: CampaignConfig,
    manifest: CalibrationManifest,
    observations: list[Observation],
) -> str:
    if not observations:
        raise ValueError("campaign has no observations")
    baseline = next(
        (
            observation
            for observation in observations
            if observation.candidate_id == "candidate-0000"
        ),
        observations[0],
    )
    best = select_incumbent(observations, campaign.objective)
    invalid_count = sum(not observation.valid for observation in observations)
    infeasible_count = sum(
        observation.valid
        and not all(
            result.passed
            for result in evaluate_metric_constraints(
                observation.metrics, campaign.objective
            )
        )
        for observation in observations
    )
    lines = [
        "# VCSEL AI 调参基线比较报告 / VCSEL AI Tuning Baseline Comparison Report",
        "",
        "## 运行身份 / Run Identity",
        "",
        f"- 活动 / Campaign: `{campaign.campaign_id}`",
        f"- 校准清单 / Calibration manifest: `{manifest.manifest_id}`",
        f"- 器件 / Device: `{manifest.device_id}`",
        f"- 温度 / Temperature: `{_format(manifest.temperature_c)} degC`",
        f"- 后端 / Backend: `{campaign.backend.kind}`",
        f"- 优化器 / Optimizer: `{campaign.optimizer.kind}`",
        f"- 随机种子 / Seed: `{campaign.optimizer.seed}`",
        "",
        "## 结果摘要 / Result Summary",
        "",
        f"- 总运行数 / Total runs: `{len(observations)}`",
        f"- 无效运行 / Invalid runs: `{invalid_count}`",
        f"- 指标不可行运行 / Metric-infeasible runs: `{infeasible_count}`",
        f"- 基线候选点 / Baseline candidate: `{baseline.candidate_id}`",
        f"- 最佳合格候选点 / Best Feasible Candidate: `{best.candidate_id}`",
        "",
        "## 指标变化 / Metric Changes",
        "",
        *_metric_rows(baseline, best),
        "",
        "## 解释边界 / Interpretation Boundary",
        "",
        "- 本报告只显示数据库中实际存在的指标，不会把 proxy 重命名为标准指标。",
        "  This report displays only metrics present in the database and never renames a proxy as a standards metric.",
        "- Milestone 1 结果不授权真实硬件写入，也不声明 TDECQ 合规或 BER 改善。",
        "  Milestone 1 results do not authorize live hardware writes or claim TDECQ compliance or BER improvement.",
        "",
    ]
    return "\n".join(lines)


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
```

- [ ] **步骤 4：运行报告测试并验证 GREEN / Step 4: Run report test and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_report.py -v
```

预期：1 个测试通过。
Expected: 1 test passes.

- [ ] **步骤 5：提交报告实现 / Step 5: Commit reporting**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/report.py tests_py/ai_tuning/test_report.py
git commit -m "feat: generate bilingual baseline reports"
```

## 任务 15：添加 CLI 命令、示例配置与 JSON Schema / Task 15: Add CLI Commands, Example Configs, and JSON Schemas

**文件 / Files:**
- 修改：`src/vcsel_ai_tuning/config.py`
- Modify: `src/vcsel_ai_tuning/config.py`
- 新建：`src/vcsel_ai_tuning/cli.py`
- Create: `src/vcsel_ai_tuning/cli.py`
- 新建：`src/vcsel_ai_tuning/__main__.py`
- Create: `src/vcsel_ai_tuning/__main__.py`
- 新建：`configs/ai_tuning/fake_campaign.yaml`
- Create: `configs/ai_tuning/fake_campaign.yaml`
- 新建：`configs/ai_tuning/fake_calibration_manifest.yaml`
- Create: `configs/ai_tuning/fake_calibration_manifest.yaml`
- 新建：`tests_py/ai_tuning/test_cli.py`
- Create: `tests_py/ai_tuning/test_cli.py`
- 生成：`schemas/ai_tuning/campaign.schema.json`
- Generate: `schemas/ai_tuning/campaign.schema.json`
- 生成：`schemas/ai_tuning/calibration_manifest.schema.json`
- Generate: `schemas/ai_tuning/calibration_manifest.schema.json`
- 生成：`schemas/ai_tuning/observation.schema.json`
- Generate: `schemas/ai_tuning/observation.schema.json`

- [ ] **步骤 1：编写会失败的 CLI 验收测试 / Step 1: Write failing CLI acceptance tests**

新建 `tests_py/ai_tuning/test_cli.py`：
Create `tests_py/ai_tuning/test_cli.py`:

```python
from pathlib import Path

from vcsel_ai_tuning.cli import main


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "configs" / "ai_tuning" / "fake_campaign.yaml"
MANIFEST = ROOT / "configs" / "ai_tuning" / "fake_calibration_manifest.yaml"


def test_validate_run_report_and_schema_commands(tmp_path: Path):
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"
    recovery = tmp_path / "recovery"
    report = tmp_path / "report.md"
    schemas = tmp_path / "schemas"

    assert main(
        ["validate-config", "--campaign", str(CAMPAIGN), "--calibration", str(MANIFEST)]
    ) == 0
    assert main(
        [
            "run",
            "--campaign",
            str(CAMPAIGN),
            "--calibration",
            str(MANIFEST),
            "--db",
            str(database),
            "--artifacts",
            str(artifacts),
            "--recovery",
            str(recovery),
        ]
    ) == 0
    assert main(
        [
            "report",
            "--campaign-id",
            "m1-fake-skew",
            "--db",
            str(database),
            "--output",
            str(report),
        ]
    ) == 0
    assert main(["export-schema", "--output", str(schemas)]) == 0

    text = report.read_text(encoding="utf-8")
    assert "基线比较报告" in text
    assert "Baseline Comparison Report" in text
    assert (schemas / "campaign.schema.json").is_file()
    assert (schemas / "calibration_manifest.schema.json").is_file()
    assert (schemas / "observation.schema.json").is_file()
```

- [ ] **步骤 2：运行 CLI 测试并验证 RED / Step 2: Run CLI test and verify RED**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_cli.py -v
```

预期：测试收集出现 ERROR，并报告 `ModuleNotFoundError: No module named 'vcsel_ai_tuning.cli'`。
Expected: collection ERROR with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.cli'`.

- [ ] **步骤 3：为配置添加确定性的 JSON Schema 导出 / Step 3: Add deterministic JSON Schema export to configuration**

在 `src/vcsel_ai_tuning/config.py` 中，将领域模型导入改为：
In `src/vcsel_ai_tuning/config.py`, change the domain import to:

```python
from .domain import Observation, ParameterSpec
```

添加以下函数：
Add this function:

```python
def write_json_schemas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "campaign.schema.json": CampaignConfig.model_json_schema(),
        "calibration_manifest.schema.json": CalibrationManifest.model_json_schema(),
        "observation.schema.json": Observation.model_json_schema(),
    }
    for name, schema in schemas.items():
        payload = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)
        (output_dir / name).write_text(payload + "\n", encoding="utf-8")
```

同时在 `config.py` 顶部添加以下标准库导入：
Also add this standard-library import at the top of `config.py`:

```python
import json
```

- [ ] **步骤 4：创建可执行的 fake 活动示例 / Step 4: Create executable fake campaign examples**

新建 `configs/ai_tuning/fake_campaign.yaml`：
Create `configs/ai_tuning/fake_campaign.yaml`:

```yaml
campaign_id: m1-fake-skew
backend:
  kind: fake
  dry_run: true
optimizer:
  kind: sobol
  budget: 8
  seed: 17
parameters:
  - name: bias_ma
    unit: mA
    lower: 4.0
    upper: 7.0
    quantization: 0.1
    max_step: 3.0
    readback_tolerance: 0.02
  - name: modulation_ma
    unit: mA
    lower: 0.5
    upper: 3.0
    quantization: 0.1
    max_step: 2.5
    readback_tolerance: 0.02
  - name: skew_control_ui
    unit: UI
    lower: -0.1
    upper: 0.1
    quantization: 0.001
    max_step: 0.2
    readback_tolerance: 0.002
safe_baseline:
  bias_ma: 5.0
  modulation_ma: 1.5
  skew_control_ui: 0.05
objective:
  primary_metric: skew_pp_ui
  direction: minimize
  metric_constraints:
    - metric: rlm_proxy
      operator: ">="
      threshold: 0.9
    - metric: worst_eye_height_norm
      operator: ">="
      threshold: 0.30
  tie_breakers:
    - metric: power_mw
      direction: minimize
slow_measurement_for_best: false
```

新建 `configs/ai_tuning/fake_calibration_manifest.yaml`：
Create `configs/ai_tuning/fake_calibration_manifest.yaml`:

```yaml
manifest_id: m1-fake-manifest-v1
device_id: fake-vcsel
temperature_c: 25.0
model_path: null
external_network_path: null
sources: []
model_parameters: {}
```

- [ ] **步骤 5：实现 CLI 命令和显式退出码 / Step 5: Implement CLI commands and explicit exit codes**

新建 `src/vcsel_ai_tuning/cli.py`：
Create `src/vcsel_ai_tuning/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import (
    CalibrationManifest,
    CampaignConfig,
    load_yaml_model,
    validate_manifest_files,
    write_json_schemas,
)
from .persistence import SqliteStore
from .recovery import RecoveryRegistry, RecoveryRequired
from .report import build_baseline_report, write_report
from .runner import CampaignRunner
from .safety import SafetyViolation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vcsel-tune")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--calibration", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--calibration", type=Path, required=True)
    run.add_argument("--db", type=Path, required=True)
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--recovery", type=Path, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--campaign-id", required=True)
    report.add_argument("--db", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    schema = subparsers.add_parser("export-schema")
    schema.add_argument("--output", type=Path, required=True)

    acknowledge = subparsers.add_parser("acknowledge-recovery")
    acknowledge.add_argument("--campaign-id", required=True)
    acknowledge.add_argument("--recovery", type=Path, required=True)
    return parser


def _load_pair(campaign_path: Path, calibration_path: Path):
    campaign = load_yaml_model(campaign_path, CampaignConfig)
    manifest = load_yaml_model(calibration_path, CalibrationManifest)
    validate_manifest_files(manifest, calibration_path.parent)
    return campaign, manifest


def _execute(arguments: argparse.Namespace) -> int:
    if arguments.command == "validate-config":
        campaign, manifest = _load_pair(arguments.campaign, arguments.calibration)
        print(f"valid campaign={campaign.campaign_id} manifest={manifest.manifest_id}")
        return 0
    if arguments.command == "run":
        campaign, manifest = _load_pair(arguments.campaign, arguments.calibration)
        observations = CampaignRunner(
            campaign,
            manifest,
            arguments.db,
            arguments.artifacts,
            arguments.recovery,
        ).run()
        print(f"completed campaign={campaign.campaign_id} runs={len(observations)}")
        return 0
    if arguments.command == "report":
        store = SqliteStore(arguments.db)
        campaign, manifest = store.load_campaign(arguments.campaign_id)
        observations = store.list_observations(arguments.campaign_id)
        write_report(
            arguments.output,
            build_baseline_report(campaign, manifest, observations),
        )
        print(arguments.output)
        return 0
    if arguments.command == "export-schema":
        write_json_schemas(arguments.output)
        print(arguments.output)
        return 0
    if arguments.command == "acknowledge-recovery":
        RecoveryRegistry(arguments.recovery).acknowledge(arguments.campaign_id)
        print(f"acknowledged campaign={arguments.campaign_id}")
        return 0
    raise AssertionError(arguments.command)


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        return _execute(arguments)
    except (ValidationError, ValueError, FileNotFoundError, KeyError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except (RecoveryRequired, SafetyViolation) as error:
        print(f"safety or recovery stop: {error}", file=sys.stderr)
        return 3
    except Exception as error:
        print(f"runtime error: {type(error).__name__}: {error}", file=sys.stderr)
        return 4
```

新建 `src/vcsel_ai_tuning/__main__.py`：
Create `src/vcsel_ai_tuning/__main__.py`:

```python
from .cli import main


raise SystemExit(main())
```

- [ ] **步骤 6：运行 CLI 验收测试并验证 GREEN / Step 6: Run CLI acceptance test and verify GREEN**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning/test_cli.py -v
```

预期：1 个端到端测试通过。
Expected: 1 end-to-end test passes.

- [ ] **步骤 7：生成并校验已提交的 JSON Schema / Step 7: Generate and validate committed JSON Schemas**

运行：
Run:

```powershell
py -3.13 -m vcsel_ai_tuning export-schema --output schemas/ai_tuning
Get-ChildItem -LiteralPath schemas/ai_tuning -Filter *.json | ForEach-Object {
  Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json | Out-Null
}
```

预期：存在三个 schema 文件，且每个文件都能解析为 JSON。
Expected: three schema files exist and every file parses as JSON.

- [ ] **步骤 8：提交 CLI、示例和 schema / Step 8: Commit CLI, examples, and schemas**

运行：
Run:

```powershell
git add -- src/vcsel_ai_tuning/config.py src/vcsel_ai_tuning/cli.py src/vcsel_ai_tuning/__main__.py configs/ai_tuning schemas/ai_tuning tests_py/ai_tuning/test_cli.py
git commit -m "feat: expose VCSEL tuning campaign CLI"
```

## 任务 16：运行完整验证并更新双语项目状态 / Task 16: Run Full Verification and Update Bilingual Project Status

**文件 / Files:**
- 修改：`README.md`
- Modify: `README.md`
- 修改：`docs/project-progress.md`
- Modify: `docs/project-progress.md`

- [ ] **步骤 1：运行完整的新里程碑 1 测试套件 / Step 1: Run the complete new Milestone 1 test suite**

运行：
Run:

```powershell
py -3.13 -m pytest tests_py/ai_tuning -v
```

预期：所有里程碑 1 测试均通过，失败、错误或警告数量均为零。
Expected: all Milestone 1 tests pass with zero failures, errors, or warnings.

- [ ] **步骤 2：运行全部 Python 回归测试，包括变更前的 23 项测试基线 / Step 2: Run all Python regression tests, including the 23-test pre-change baseline**

运行：
Run:

```powershell
cmd /c "py -3.13 -m unittest discover -s tests_py -v 2>&1"
```

预期：原有 23 个测试仍保持通过；包导入相关的 `unittest` 可能增加测试总数，但失败数和错误数均为零。
Expected: the original 23 tests remain passing; the package import `unittest` may increase the total, but there are zero failures and zero errors.

- [ ] **步骤 3：运行完整的仓库 JavaScript 结构测试套件 / Step 3: Run the complete repository JavaScript structural suite**

运行：
Run:

```powershell
$jsTests = Get-ChildItem -LiteralPath tests -Filter *.test.mjs | ForEach-Object { $_.FullName }
node --test $jsTests
```

预期：所有现有 JavaScript 测试均通过，失败数为零。
Expected: all existing JavaScript tests pass with zero failures.

- [ ] **步骤 4：在 pytest 之外执行一次全新的 CLI 活动 / Step 4: Execute a fresh CLI campaign outside pytest**

运行：
Run:

```powershell
$root = Join-Path $env:TEMP ("vcsel-ai-tuning-acceptance-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $root | Out-Null
py -3.13 -m vcsel_ai_tuning validate-config --campaign configs/ai_tuning/fake_campaign.yaml --calibration configs/ai_tuning/fake_calibration_manifest.yaml
py -3.13 -m vcsel_ai_tuning run --campaign configs/ai_tuning/fake_campaign.yaml --calibration configs/ai_tuning/fake_calibration_manifest.yaml --db "$root\state.sqlite" --artifacts "$root\artifacts" --recovery "$root\recovery"
py -3.13 -m vcsel_ai_tuning report --campaign-id m1-fake-skew --db "$root\state.sqlite" --output "$root\report.md"
Get-Content -LiteralPath "$root\report.md" -TotalCount 30
```

预期：校验和运行均以 0 退出，存在八个运行目录，且报告同时包含 `基线比较报告` 和 `Baseline Comparison Report`。
Expected: validation and run exit 0, eight run directories exist, and the report contains both `基线比较报告` and `Baseline Comparison Report`.

- [ ] **步骤 5：证明里程碑 1 不包含真实仪器传输实现 / Step 5: Prove Milestone 1 contains no live-instrument transport**

运行：
Run:

```powershell
$hits = rg -n -i "import pyvisa|hislip|vxi-11|socket\.connect|open_resource" src/vcsel_ai_tuning
if ($LASTEXITCODE -eq 1) { Write-Output "no live transport found" } else { $hits; exit 1 }
```

预期：`no live transport found`。
Expected: `no live transport found`.

- [ ] **步骤 6：将已验证的里程碑状态添加到 README / Step 6: Add the verified milestone status to the README**

验证成功后，将以下双语章节追加到 `README.md`：
Append this bilingual section to `README.md` after verification succeeds:

```markdown
## AI Tuning Milestone 1 / AI 调参里程碑 1

- 已建立统一的数字孪生与硬件就绪实验内核，包括配置校验、安全门禁、恢复标记、fake/replay 后端、PAM4 skew proxy、原子化实验记录、确定性候选生成和双语报告。
  A unified digital-twin and hardware-ready experiment kernel now provides configuration validation, safety gating, recovery markers, fake/replay backends, a PAM4 skew proxy, atomic experiment records, deterministic candidate generation, and bilingual reports.
- 真实 Cadence 执行、厂商 SCPI 命令和真实硬件写入仍保持禁用，等待后续里程碑和人工 commissioning。
  Live Cadence execution, vendor SCPI commands, and real hardware writes remain disabled pending later milestones and operator commissioning.
```

- [ ] **步骤 7：添加带日期的进度快照 / Step 7: Add a dated progress snapshot**

将以下双语章节追加到 `docs/project-progress.md`：
Append this bilingual section to `docs/project-progress.md`:

```markdown
## Progress Snapshot 2026-07-10 / 2026-07-10 进度快照

- AI tuning Milestone 1 已完成软件框架验证：fake/replay 双后端、PAM4 skew 指标、安全状态机、恢复与回滚、SQLite/artifact 溯源、grid/random/Sobol 搜索和 CLI 报告均已通过自动测试。
  AI tuning Milestone 1 has completed software-framework verification: fake/replay backends, PAM4 skew metrics, the safety state machine, recovery and rollback, SQLite/artifact provenance, grid/random/Sobol search, and CLI reporting all pass automated tests.
- 该状态只证明软件路径可运行，不代表真实仪器已接入，也不代表 TDECQ 或 BER 已得到硬件改善。
  This status proves only that the software path runs; it does not mean real instruments are connected or that hardware TDECQ or BER has improved.
```

- [ ] **步骤 8：在状态编辑后重新运行文档和测试验证 / Step 8: Re-run documentation and test verification after status edits**

运行：
Run:

```powershell
rg -n "AI Tuning Milestone 1|AI 调参里程碑 1|Progress Snapshot 2026-07-10|2026-07-10 进度快照" README.md docs/project-progress.md
py -3.13 -m pytest tests_py/ai_tuning -q
$jsTests = Get-ChildItem -LiteralPath tests -Filter *.test.mjs | ForEach-Object { $_.FullName }
node --test $jsTests
```

预期：找到全部四个双语标签，且两个测试命令均以 0 退出。
Expected: all four bilingual labels are found and both test commands exit 0.

- [ ] **步骤 9：审查最终差异并提交已验证的里程碑 1 / Step 9: Review the final diff and commit verified Milestone 1**

运行：
Run:

```powershell
git status --short
git diff --check
git diff --stat
git add -- README.md docs/project-progress.md
git commit -m "docs: record verified AI tuning milestone one"
```

预期：没有空白字符错误，仅暂存预期文件，并且文档提交成功。
Expected: no whitespace errors, only intentional files are staged, and the documentation commit succeeds.

## 最终验收清单 / Final Acceptance Checklist

- 每个生产函数都是在测试因预期缺失行为而失败之后才引入的。
- [ ] Every production function was introduced after a test failed for the expected missing behavior.
- 所有新测试均通过，且原有 23 个 Python 测试仍然通过。
- [ ] All new tests pass and the original 23 Python tests still pass.
- 现有 JavaScript 结构测试仍然通过。
- [ ] Existing JavaScript structural tests still pass.
- 对 0、0.03、0.06 和 0.10 UI 的注入倾斜，已知倾斜恢复满足 `max(0.005 UI, 注入倾斜的 10%)`。
- [ ] Known-skew recovery meets `max(0.005 UI, 10% of injected skew)` for 0, 0.03, 0.06, and 0.10 UI.
- 全局延迟造成的实测倾斜变化不超过 0.005 UI。
- [ ] Global delay changes measured skew by no more than 0.005 UI.
- 对固定种子和测试样本，fake 与 replay 活动具有确定性。
- [ ] Fake and replay campaigns are deterministic for a fixed seed and fixture.
- 只有有效且可行的观测才有资格参与最优点选择。
- [ ] Only valid and feasible observations are eligible for incumbent selection.
- FFE 抽头向量采用 L1 归一化，并拒绝零范数抽头向量。
- [ ] FFE tap vectors are L1-normalized and zero-norm tap vectors are rejected.
- 持久化失败会隔离制品并强制执行安全关断。
- [ ] Persistence failure quarantines artifacts and forces safe-off.
- 中断运行的恢复流程在操作员确认前保持未布防状态。
- [ ] Interrupted-run recovery remains disarmed until operator acknowledgement.
- 里程碑 1 中不存在真实仪器传输实现或真实硬件写入路径。
- [ ] No live instrument transport or real hardware-write path exists in Milestone 1.
- CLI 会生成 SQLite 状态、完整制品、已提交的 schema 和双语报告。
- [ ] The CLI generates SQLite state, complete artifacts, committed schemas, and a bilingual report.
- README 和进度快照陈述已验证结果，但不声称硬件或标准合规性。
- [ ] The README and progress snapshot state the verified result without claiming hardware or standards compliance.

## 执行交接 / Execution Handoff

在开始任一执行选项之前，Task 0 将使用 `git init -b main` 初始化当前为空的 `.git` 元数据；它不会暂存无关的工作区文件。如果全新检查发现元数据非空，则停止执行并进行检查。
Before either execution option begins, Task 0 will initialize the currently empty `.git` metadata with `git init -b main`; it will not stage unrelated workspace files. If the fresh check finds non-empty metadata, execution stops for inspection instead.

计划执行选项：
Plan execution options:

（1）**子代理驱动（推荐）**——为每项任务分派一个全新的实施代理，然后在任务之间执行规格审查和代码质量审查。
1. **Subagent-Driven (recommended)** — dispatch a fresh implementation agent per task, then perform specification and code-quality reviews between tasks.
（2）**内联执行**——在本会话中使用 `executing-plans` 执行任务，以经过审查的批次工作并设置检查点。
2. **Inline Execution** — execute tasks in this session with `executing-plans`, working in reviewed batches with checkpoints.
