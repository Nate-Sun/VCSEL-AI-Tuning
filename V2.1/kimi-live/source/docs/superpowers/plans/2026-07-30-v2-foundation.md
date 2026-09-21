# VCSEL AI Tuning V2 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不调用真实Kimi API、不加载DINOv2权重、不连接任何硬件的前提下，完成可插拔`OptimizationBrain`的统一合同、FAKE/replay、能力报告、确定性眼图预处理和失败关闭基础设施。

**Architecture:** V1继续负责采集、硬件执行、回读、回滚和可信Sweep产物；本计划只新增硬件无关的`vcsel_ai_tuning.v2`包。主循环未来只依赖一个`OptimizationBrain`协议；组合式Brain负责“评分器+Decision Provider”，融合式Brain负责一次API完成评分和决策。Factory只创建已安装且在当前配置中启用的模式，Provider异常、低置信度、OOD或越过候选集合时统一返回`HOLD`。

**Tech Stack:** Python 3.12、Pydantic 2、NumPy 2、SciPy、PyYAML、pytest

---

## 1. 本计划的边界

本计划包含：

- `EyeSample`、`PerceptionResult`、`DecisionProposal`、`BrainResult`和`V2Observation`严格合同；
- `OptimizationBrain`、`EyeScorer`、`DecisionProvider`和融合式Provider协议；
- `ComposedBrain`与`MultimodalApiBrain`硬件无关骨架；
- 显式注册、显式启用、无静默回退的`BrainFactory`；
- 一个模拟`LOCAL_VISION_API`组合的FAKE评分器和FAKE Decision Provider；
- 一个通用脚本Brain，用于证明调用方不依赖具体Provider；
- 只读取二维密度矩阵的固定对数归一化、缩放、填充和RGB转换；
- replay记录、跨Provider评估结构和失败关闭测试；
- V2包的硬件导入隔离测试。

本计划不包含：

- 真实Kimi网络请求、API Key读取、Prompt定稿和计费调用；
- DINOv2下载、权重加载、训练、校准或准确率声明；
- 把架构预留的每个`brain_mode`都当成本次交付内容；
- V1单次采集绑定、Outer ER来源修复或真实Sweep采集；
- 修改现有Campaign数据库结构、Dashboard或六步硬件主循环；
- 影子模式、真实硬件写入或自动调参。

这些内容分别进入Kimi Provider、本地视觉训练、本地Decision适配、V1样本绑定和线上闭环五个独立计划。V2 Foundation完成不代表这些能力已经可用。

### 仓库兼容性决定

- V2配置是独立YAML文档，不添加到现有`RuntimeConfig`；后者使用`extra="forbid"`，直接加入`v2:`会破坏V1启动配置。
- V2定义只读`BrainCandidate`投影，不复用仓库中三个同名但不兼容的候选合同。线上集成只能按ID找回原始`campaign.models.Candidate`并再次调用现有授权检查，绝不能把`BrainCandidate`或`DecisionProposal`直接交给执行器。
- `NO_SAFE_CANDIDATE`不是新的动作枚举，而是`action=HOLD`加固定`reason_codes=("NO_SAFE_CANDIDATE",)`。
- `V2Observation.measurement_id`只要求与`PerceptionResult.measurement_id`一致；现有`TuningObservation.observation_id`保留原语义，不强行假设两个ID相等。
- Foundation只能冻结一个进程内使用的Brain实例。跨重启的Campaign模式锁定需要修改Campaign存储和审计，属于Shadow Mode Integration计划。
- 当前`SweepPointRecord`还没有完整的measurement ID、密度artifact引用、DCA配置指纹和文件哈希，因此本计划不制作伪装成可用的V1→V2适配器。
- 本计划不修改项目版本号`1.1.0`，也不把V2导出到顶层`vcsel_ai_tuning.__init__`。
- replay JSONL是独立离线文件，不写入现有`CampaignRepository`；正式artifact类别和原子发布由线上集成计划定义。
- 传入Brain的候选顺序是请求合同的一部分：ID必须唯一、顺序原样保留并进入请求哈希，不由Provider自行重排。

## 2. 文件职责图

### 新建生产文件

- `src/vcsel_ai_tuning/v2/__init__.py`：只导出已经稳定且硬件无关的V2公共类型。
- `src/vcsel_ai_tuning/v2/contracts.py`：严格、可序列化、不可变的数据合同、`BrainCandidate`投影和跨字段验证。
- `src/vcsel_ai_tuning/v2/brain.py`：协议、组合式Brain、融合式Brain、统一失败关闭和候选集合核对。
- `src/vcsel_ai_tuning/v2/factory.py`：配置读取、Provider注册、能力报告和显式创建。
- `src/vcsel_ai_tuning/v2/preprocessing.py`：从受哈希保护的`.npy`密度矩阵生成固定尺寸RGB数组。
- `src/vcsel_ai_tuning/v2/providers/__init__.py`：只导出FAKE Provider；不导出未实现Provider。
- `src/vcsel_ai_tuning/v2/providers/fake.py`：确定性FAKE评分器、FAKE Decision和脚本Brain。
- `src/vcsel_ai_tuning/v2/replay.py`：在无硬件条件下逐样本调用Brain并保存JSONL审计记录。
- `src/vcsel_ai_tuning/v2/evaluation.py`：延迟、HOLD率、重复一致性和A/B排序评估。
- `configs/ai_tuning/v2.example.yaml`：只启用一个FAKE演示目标模式的示例配置。

### 新建测试文件

- `tests_py/ai_tuning/v2/__init__.py`：V2测试包标记。
- `tests_py/ai_tuning/v2/conftest.py`：统一`EyeSample`、候选集合和Provider输出夹具。
- `tests_py/ai_tuning/v2/test_contracts.py`：合同严格性、ID绑定和动作字段约束。
- `tests_py/ai_tuning/v2/test_brain.py`：组合、融合单次调用和失败关闭。
- `tests_py/ai_tuning/v2/test_factory.py`：安装/启用能力、拒绝未启用模式和禁止回退。
- `tests_py/ai_tuning/v2/test_preprocessing.py`：固定形状、方向、哈希、数值范围和确定性。
- `tests_py/ai_tuning/v2/test_replay.py`：两种FAKE Brain通过同一replay入口。
- `tests_py/ai_tuning/v2/test_evaluation.py`：跨Provider记录和评估统计。
- `tests_py/ai_tuning/v2/test_fail_closed.py`：异常、低置信度、OOD、非法候选均不产生`PROPOSE`。
- `tests_py/ai_tuning/v2/test_import_isolation.py`：V2不得导入硬件、执行器或持久化主循环。

### 保持不变

- `src/vcsel_ai_tuning/hardware/**`
- `src/vcsel_ai_tuning/campaign/orchestrator.py`
- `src/vcsel_ai_tuning/v1_1/**`
- `src/vcsel_ai_tuning/safety.py`
- `frontend/**`

## 3. 依赖与并行顺序

```text
Task 1 严格合同
    |
Task 2 Brain协议与失败关闭
    |
    +--> Task 3 Factory与配置 --------+
    +--> Task 4 密度矩阵预处理 -------+--> Task 7 集成与回归
    `--> Task 5 FAKE与replay --> Task 6 评估框架 --+
```

Task 1和Task 2由集成负责人串行完成并冻结接口。Task 3、Task 4和Task 5文件互不重叠，可以在同一个工作树中并行；Task 6只等待Task 5；Task 7由集成负责人统一完成。

执行本计划的PowerShell会话先设置：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
```

所有pytest命令使用仓库现有`.venv-ai-tuning`并禁用pytest缓存，避免并行任务生成共享缓存文件。

## Task 1: 定义严格V2合同并冻结字段名

**Files:**

- Create: `src/vcsel_ai_tuning/v2/__init__.py`
- Create: `src/vcsel_ai_tuning/v2/contracts.py`
- Create: `tests_py/ai_tuning/v2/__init__.py`
- Create: `tests_py/ai_tuning/v2/conftest.py`
- Create: `tests_py/ai_tuning/v2/test_contracts.py`

- [ ] **Step 1: 新建失败测试，固定公共枚举和成功样本合同**

先创建`tests_py/ai_tuning/v2/__init__.py`：

```python
"""Tests for the hardware-neutral V2 foundation package."""
```

在`tests_py/ai_tuning/v2/conftest.py`中建立共享夹具，字段名必须与设计稿一致：

```python
from datetime import datetime, timezone
import pytest

from vcsel_ai_tuning.v2.contracts import (
    ArtifactReference,
    BrainCandidate,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
)


@pytest.fixture
def eye_sample() -> EyeSample:
    density_sha = "a" * 64
    return EyeSample(
        measurement_id="measurement-001",
        campaign_id="campaign-001",
        sweep_point_id="point-001",
        raw_eye_density_ref=ArtifactReference(
            relative_path="point-001/raw_eye_density.npy",
            sha256=density_sha,
            media_type="application/x-npy",
        ),
        waveform_ref=None,
        metrics=EyeMetrics(
            tdecq_db=3.1,
            outer_er_chan1a_db=2.8,
            rlm=0.96,
            outer_oma=1.2,
            other_metrics={"average_power_dbm": -1.4},
        ),
        requested_params={"ibias_dac": 70.0, "eq": 21.0},
        applied_params={"ibias_dac": 70.0, "eq": 21.0},
        readback_params={"ibias_dac": 70.0, "eq": 21.0},
        dca_config_fingerprint="b" * 64,
        measurement_status=MeasurementStatus.VALID,
        captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        artifact_hashes={"point-001/raw_eye_density.npy": density_sha},
    )


@pytest.fixture
def safe_candidates() -> tuple[BrainCandidate, ...]:
    return (
        BrainCandidate(candidate_id="candidate-001", requested_params={"ibias_dac": 71.0, "eq": 21.0}),
        BrainCandidate(candidate_id="candidate-002", requested_params={"ibias_dac": 70.0, "eq": 22.0}),
    )
```

在`tests_py/ai_tuning/v2/test_contracts.py`中先固定最关键的跨字段规则：

```python
from pydantic import ValidationError
import pytest

from vcsel_ai_tuning.v2.contracts import (
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    PerceptionResult,
)


def test_contracts_accept_one_complete_brain_result(eye_sample):
    perception = PerceptionResult(
        measurement_id=eye_sample.measurement_id,
        quality_score=0.72,
        confidence=0.91,
        valid=True,
        ood=False,
        defects={"ringing": 0.1},
        feature_vector=(0.2, 0.3),
        provider_id="fake-vision",
        provider_version="1",
        input_hash="c" * 64,
    )
    decision = DecisionProposal(
        based_on_measurement_id=eye_sample.measurement_id,
        action=DecisionAction.PROPOSE,
        candidate_id="candidate-001",
        requested_params={"ibias_dac": 71.0, "eq": 21.0},
        reason_codes=("BEST_FAKE_SCORE",),
        expected_improvement=0.05,
        provider_id="fake-decision",
        provider_version="1",
    )
    result = BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=BrainMode.LOCAL_VISION_API,
        latency_ms=1.0,
        request_hash="d" * 64,
        response_hash="e" * 64,
    )
    assert result.decision.candidate_id == "candidate-001"


def test_propose_requires_candidate_and_requested_params(eye_sample):
    with pytest.raises(ValidationError, match="PROPOSE"):
        DecisionProposal(
            based_on_measurement_id=eye_sample.measurement_id,
            action=DecisionAction.PROPOSE,
            reason_codes=("BROKEN",),
            provider_id="fake-decision",
            provider_version="1",
        )


def test_hold_forbids_executable_fields(eye_sample):
    with pytest.raises(ValidationError, match="non-PROPOSE"):
        DecisionProposal(
            based_on_measurement_id=eye_sample.measurement_id,
            action=DecisionAction.HOLD,
            candidate_id="candidate-001",
            requested_params={"ibias_dac": 71.0},
            reason_codes=("LOW_CONFIDENCE",),
            provider_id="brain-guard",
            provider_version="1",
        )


def test_eye_sample_rejects_artifact_hash_mismatch(eye_sample):
    values = {
        name: getattr(eye_sample, name)
        for name in eye_sample.__class__.model_fields
    }
    values["artifact_hashes"] = {"point-001/raw_eye_density.npy": "f" * 64}
    with pytest.raises(ValidationError, match="artifact hash"):
        eye_sample.__class__(**values)
```

- [ ] **Step 2: 运行合同测试并确认因V2包不存在而失败**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_contracts.py -q -p no:cacheprovider
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'vcsel_ai_tuning.v2'`.

- [ ] **Step 3: 实现严格合同和跨字段验证**

在`src/vcsel_ai_tuning/v2/contracts.py`中实现以下结构；映射字段沿用现有`campaign.models`的`MappingProxyType + AfterValidator + PlainSerializer`模式，禁止在冻结模型内部继续原地修改：

```python
from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PlainSerializer,
    StrictBool,
    StringConstraints,
    model_validator,
)

from vcsel_ai_tuning.optimization.protocol import TuningObservation


Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+$")]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _serialize_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return dict(value)


ImmutableParams = Annotated[
    Mapping[str, FiniteFloat],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float]),
]
ImmutableMetrics = Annotated[
    Mapping[str, FiniteFloat | None],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float | None]),
]
ImmutableProbabilities = Annotated[
    Mapping[str, Probability],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, float]),
]
ImmutableHashes = Annotated[
    Mapping[str, Sha256],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict[str, str]),
]


class FrozenV2Model(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        validate_default=True,
    )


class BrainMode(StrEnum):
    PURE_API = "PURE_API"
    LOCAL_VISION_API = "LOCAL_VISION_API"
    LOCAL_VISION_LOCAL = "LOCAL_VISION_LOCAL"
    DIRECT_FEATURES_LOCAL = "DIRECT_FEATURES_LOCAL"


class MeasurementStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"


class DecisionAction(StrEnum):
    PROPOSE = "PROPOSE"
    HOLD = "HOLD"
    STOP = "STOP"
    REQUEST_HUMAN = "REQUEST_HUMAN"


class ArtifactReference(FrozenV2Model):
    relative_path: NonBlank
    sha256: Sha256
    media_type: NonBlank

    @model_validator(mode="after")
    def validate_relative_posix_path(self) -> "ArtifactReference":
        if "\\" in self.relative_path:
            raise ValueError("artifact path must use POSIX separators")
        parts = self.relative_path.split("/")
        if (
            self.relative_path.startswith("/")
            or ":" in parts[0]
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("artifact path must be a canonical relative POSIX path")
        return self


class BrainCandidate(FrozenV2Model):
    candidate_id: Identifier
    requested_params: ImmutableParams


class EyeMetrics(FrozenV2Model):
    tdecq_db: FiniteFloat | None = None
    outer_er_chan1a_db: FiniteFloat | None = None
    rlm: FiniteFloat | None = None
    outer_oma: FiniteFloat | None = None
    other_metrics: ImmutableMetrics = Field(default_factory=dict)


class EyeSample(FrozenV2Model):
    measurement_id: Identifier
    campaign_id: Identifier
    sweep_point_id: Identifier
    raw_eye_density_ref: ArtifactReference
    waveform_ref: ArtifactReference | None = None
    metrics: EyeMetrics
    requested_params: ImmutableParams
    applied_params: ImmutableParams
    readback_params: ImmutableParams
    dca_config_fingerprint: Sha256
    measurement_status: MeasurementStatus
    captured_at: AwareDatetime
    artifact_hashes: ImmutableHashes

    @model_validator(mode="after")
    def validate_artifact_hashes(self) -> "EyeSample":
        references = [self.raw_eye_density_ref]
        if self.waveform_ref is not None:
            references.append(self.waveform_ref)
        for reference in references:
            key = reference.relative_path
            if self.artifact_hashes.get(key) != reference.sha256:
                raise ValueError(f"artifact hash mismatch for {key}")
        return self


class PerceptionResult(FrozenV2Model):
    measurement_id: Identifier
    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    defects: ImmutableProbabilities = Field(default_factory=dict)
    feature_vector: tuple[FiniteFloat, ...] | None = None
    provider_id: NonBlank
    provider_version: NonBlank
    input_hash: Sha256


class DecisionProposal(FrozenV2Model):
    based_on_measurement_id: Identifier
    action: DecisionAction
    candidate_id: Identifier | None = None
    requested_params: ImmutableParams | None = None
    reason_codes: tuple[NonBlank, ...] = Field(min_length=1)
    expected_improvement: FiniteFloat | None = None
    provider_id: NonBlank
    provider_version: NonBlank

    @model_validator(mode="after")
    def validate_action_fields(self) -> "DecisionProposal":
        if self.action is DecisionAction.PROPOSE:
            if self.candidate_id is None or self.requested_params is None:
                raise ValueError("PROPOSE requires candidate_id and requested_params")
        elif self.candidate_id is not None or self.requested_params is not None:
            raise ValueError("non-PROPOSE action forbids executable candidate fields")
        return self


class BrainResult(FrozenV2Model):
    perception: PerceptionResult
    decision: DecisionProposal
    brain_mode: BrainMode
    latency_ms: FiniteFloat = Field(ge=0.0)
    request_hash: Sha256
    response_hash: Sha256

    @model_validator(mode="after")
    def validate_measurement_binding(self) -> "BrainResult":
        if self.perception.measurement_id != self.decision.based_on_measurement_id:
            raise ValueError("perception and decision measurement IDs differ")
        return self


class V2Observation(FrozenV2Model):
    measurement_id: Identifier
    tuning_observation: TuningObservation
    perception: PerceptionResult

    @model_validator(mode="after")
    def validate_measurement_binding(self) -> "V2Observation":
        if self.measurement_id != self.perception.measurement_id:
            raise ValueError("V2 observation and perception measurement IDs differ")
        return self
```

在`src/vcsel_ai_tuning/v2/__init__.py`中先只导出这些合同，避免导入任何Provider：

```python
from .contracts import (
    ArtifactReference,
    BrainCandidate,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    EyeMetrics,
    EyeSample,
    MeasurementStatus,
    PerceptionResult,
    V2Observation,
)

__all__ = [
    "BrainMode",
    "BrainCandidate",
    "BrainResult",
    "DecisionAction",
    "DecisionProposal",
    "EyeMetrics",
    "EyeSample",
    "MeasurementStatus",
    "PerceptionResult",
    "V2Observation",
    "ArtifactReference",
]
```

- [ ] **Step 4: 补充非法数值、额外字段、错误ID和映射不可变测试**

在`test_contracts.py`中增加以下测试：

```python
def test_nonfinite_score_is_rejected(eye_sample):
    with pytest.raises(ValidationError):
        PerceptionResult(
            measurement_id=eye_sample.measurement_id,
            quality_score=float("nan"),
            confidence=0.9,
            valid=True,
            ood=False,
            provider_id="fake-vision",
            provider_version="1",
            input_hash="c" * 64,
        )


def test_unknown_contract_field_is_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EyeMetrics(tdecq_db=3.0, leaked_answer=2.0)


def test_brain_result_rejects_mismatched_measurement_ids(valid_perception):
    decision = DecisionProposal(
        based_on_measurement_id="measurement-other",
        action=DecisionAction.HOLD,
        reason_codes=("NO_SAFE_CANDIDATE",),
        provider_id="fake-decision",
        provider_version="1",
    )
    with pytest.raises(ValidationError, match="measurement IDs differ"):
        BrainResult(
            perception=valid_perception,
            decision=decision,
            brain_mode=BrainMode.LOCAL_VISION_API,
            latency_ms=1.0,
            request_hash="d" * 64,
            response_hash="e" * 64,
        )


def test_nested_parameter_mapping_is_immutable(eye_sample):
    with pytest.raises(TypeError):
        eye_sample.requested_params["ibias_dac"] = 99.0


@pytest.mark.parametrize(
    "relative_path",
    ["/absolute/raw.npy", "C:/raw.npy", "../raw.npy", "point/../raw.npy", "point\\raw.npy"],
)
def test_artifact_reference_requires_canonical_relative_posix_path(relative_path):
    with pytest.raises(ValidationError, match="artifact path"):
        ArtifactReference(
            relative_path=relative_path,
            sha256="a" * 64,
            media_type="application/x-npy",
        )
```

在`conftest.py`增加：

```python
from vcsel_ai_tuning.v2.contracts import PerceptionResult


@pytest.fixture
def valid_perception(eye_sample) -> PerceptionResult:
    return PerceptionResult(
        measurement_id=eye_sample.measurement_id,
        quality_score=0.72,
        confidence=0.91,
        valid=True,
        ood=False,
        defects={"ringing": 0.1},
        feature_vector=(0.2, 0.3),
        provider_id="fake-vision",
        provider_version="1",
        input_hash="c" * 64,
    )
```

- [ ] **Step 5: 运行合同测试并确认通过**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_contracts.py -q -p no:cacheprovider
```

Expected: PASS，且没有warning。

- [ ] **Step 6: 提交冻结合同**

```powershell
git add src/vcsel_ai_tuning/v2/__init__.py src/vcsel_ai_tuning/v2/contracts.py tests_py/ai_tuning/v2/__init__.py tests_py/ai_tuning/v2/conftest.py tests_py/ai_tuning/v2/test_contracts.py
git commit -m "feat: define strict V2 brain contracts"
```

## Task 2: 实现Brain协议、组合方式和统一失败关闭

**Files:**

- Create: `src/vcsel_ai_tuning/v2/brain.py`
- Create: `tests_py/ai_tuning/v2/test_brain.py`
- Create: `tests_py/ai_tuning/v2/test_fail_closed.py`
- Modify: `src/vcsel_ai_tuning/v2/__init__.py`

- [ ] **Step 1: 写失败测试，固定组合式调用和融合式单次调用**

测试必须使用只记录调用次数的本地Spy，不导入API或硬件模块。关键断言如下：

```python
def test_composed_brain_calls_scorer_then_decision_once(eye_sample, safe_candidates, valid_scorer, selecting_decision):
    brain = ComposedBrain(
        mode=BrainMode.LOCAL_VISION_API,
        scorer=valid_scorer,
        decision_provider=selecting_decision,
        minimum_confidence=0.7,
    )
    result = brain.analyze(eye_sample, (), safe_candidates)
    assert valid_scorer.calls == [eye_sample.measurement_id]
    assert selecting_decision.calls == [eye_sample.measurement_id]
    assert result.decision.action is DecisionAction.PROPOSE


def test_multimodal_brain_uses_one_fused_provider_call(eye_sample, safe_candidates, fused_provider):
    brain = MultimodalApiBrain(provider=fused_provider, minimum_confidence=0.7)
    result = brain.analyze(eye_sample, (), safe_candidates)
    assert fused_provider.call_count == 1
    assert result.brain_mode is BrainMode.PURE_API
```

- [ ] **Step 2: 写失败关闭测试矩阵**

使用`pytest.mark.parametrize`覆盖以下输入：

```python
@pytest.mark.parametrize(
    ("valid", "ood", "confidence", "reason"),
    [
        (False, False, 0.9, "INVALID_EYE"),
        (True, True, 0.9, "OOD"),
        (True, False, 0.2, "LOW_CONFIDENCE"),
    ],
)
def test_perception_gate_returns_hold_without_calling_decision(
    eye_sample,
    safe_candidates,
    scorer_factory,
    selecting_decision,
    valid,
    ood,
    confidence,
    reason,
):
    brain = ComposedBrain(
        mode=BrainMode.LOCAL_VISION_API,
        scorer=scorer_factory(valid=valid, ood=ood, confidence=confidence),
        decision_provider=selecting_decision,
        minimum_confidence=0.7,
    )
    result = brain.analyze(eye_sample, (), safe_candidates)
    assert result.decision.action is DecisionAction.HOLD
    assert reason in result.decision.reason_codes
    assert selecting_decision.calls == []
```

另外增加下列测试；`RaisingScorer`、`RaisingDecision`和`FixedDecision`都放在测试文件内，只实现对应协议并记录调用：

```python
def assert_closed(result):
    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.candidate_id is None
    assert result.decision.requested_params is None


def test_scorer_exception_fails_closed(eye_sample, safe_candidates, raising_scorer, selecting_decision):
    brain = ComposedBrain(
        mode=BrainMode.LOCAL_VISION_API,
        scorer=raising_scorer(TimeoutError("secret provider detail")),
        decision_provider=selecting_decision,
        minimum_confidence=0.7,
    )
    result = brain.analyze(eye_sample, (), safe_candidates)
    assert_closed(result)
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert "secret" not in result.model_dump_json()


def test_decision_exception_fails_closed(eye_sample, safe_candidates, valid_scorer, raising_decision):
    brain = ComposedBrain(
        mode=BrainMode.LOCAL_VISION_API,
        scorer=valid_scorer,
        decision_provider=raising_decision(RuntimeError("provider stack")),
        minimum_confidence=0.7,
    )
    assert_closed(brain.analyze(eye_sample, (), safe_candidates))


def test_foreign_measurement_id_fails_closed(eye_sample, safe_candidates, foreign_id_scorer, selecting_decision):
    brain = ComposedBrain(
        mode=BrainMode.LOCAL_VISION_API,
        scorer=foreign_id_scorer,
        decision_provider=selecting_decision,
        minimum_confidence=0.7,
    )
    assert_closed(brain.analyze(eye_sample, (), safe_candidates))


def test_candidate_outside_input_set_fails_closed(eye_sample, safe_candidates, valid_scorer, fixed_decision):
    decision = DecisionProposal(
        based_on_measurement_id=eye_sample.measurement_id,
        action=DecisionAction.PROPOSE,
        candidate_id="candidate-foreign",
        requested_params={"ibias_dac": 100.0},
        reason_codes=("MODEL_SELECTED",),
        provider_id="fake-decision",
        provider_version="1",
    )
    brain = ComposedBrain(
        mode=BrainMode.LOCAL_VISION_API,
        scorer=valid_scorer,
        decision_provider=fixed_decision(decision),
        minimum_confidence=0.7,
    )
    result = brain.analyze(eye_sample, (), safe_candidates)
    assert_closed(result)
    assert result.decision.reason_codes == ("CANDIDATE_NOT_IN_INPUT_SET",)
```

测试Spy使用下面的单一实现族，避免每条测试产生不同语义：

```python
class FixedScorer:
    provider_id = "fake-vision"
    provider_version = "1"

    def __init__(self, *, valid=True, ood=False, confidence=0.9, measurement_id=None, error=None):
        self.valid = valid
        self.ood = ood
        self.confidence = confidence
        self.measurement_id = measurement_id
        self.error = error
        self.calls = []

    def score(self, sample):
        self.calls.append(sample.measurement_id)
        if self.error is not None:
            raise self.error
        return PerceptionResult(
            measurement_id=self.measurement_id or sample.measurement_id,
            quality_score=0.8,
            confidence=self.confidence,
            valid=self.valid,
            ood=self.ood,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            input_hash=sample.raw_eye_density_ref.sha256,
        )


class FixedDecisionProvider:
    provider_id = "fake-decision"
    provider_version = "1"

    def __init__(self, *, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.calls = []

    def propose(self, sample, perception, history, safe_candidates):
        self.calls.append(sample.measurement_id)
        if self.error is not None:
            raise self.error
        if self.decision is not None:
            return self.decision
        selected = safe_candidates[0]
        return DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=selected.candidate_id,
            requested_params=selected.requested_params,
            reason_codes=("TEST_SELECTION",),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )
```

`fused_provider`使用：

```python
class FixedFusedProvider:
    provider_id = "fake-fused"
    provider_version = "1"

    def __init__(self):
        self.call_count = 0

    def analyze_once(self, sample, history, safe_candidates):
        self.call_count += 1
        perception = FixedScorer().score(sample)
        decision = FixedDecisionProvider().propose(
            sample, perception, history, safe_candidates
        )
        return perception, decision


@pytest.fixture
def valid_scorer():
    return FixedScorer()


@pytest.fixture
def selecting_decision():
    return FixedDecisionProvider()


@pytest.fixture
def fused_provider():
    return FixedFusedProvider()


@pytest.fixture
def scorer_factory():
    return lambda **kwargs: FixedScorer(**kwargs)


@pytest.fixture
def raising_scorer():
    return lambda error: FixedScorer(error=error)


@pytest.fixture
def raising_decision():
    return lambda error: FixedDecisionProvider(error=error)


@pytest.fixture
def fixed_decision():
    return lambda decision: FixedDecisionProvider(decision=decision)
```

```python
@pytest.fixture
def foreign_id_scorer():
    return FixedScorer(measurement_id="measurement-foreign")
```

- [ ] **Step 3: 运行目标测试并确认失败**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_brain.py tests_py/ai_tuning/v2/test_fail_closed.py -q -p no:cacheprovider
```

Expected: FAIL because `vcsel_ai_tuning.v2.brain` does not exist.

- [ ] **Step 4: 实现协议和共享验证函数**

`src/vcsel_ai_tuning/v2/brain.py`必须包含以下协议和边界：

```python
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from time import perf_counter_ns
from typing import Protocol, runtime_checkable

from .contracts import (
    BrainCandidate,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    EyeSample,
    PerceptionResult,
    V2Observation,
)


@runtime_checkable
class OptimizationBrain(Protocol):
    @property
    def provider_key(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> BrainResult: ...


class EyeScorer(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def score(self, sample: EyeSample) -> PerceptionResult: ...


class DecisionProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def propose(
        self,
        sample: EyeSample,
        perception: PerceptionResult,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> DecisionProposal: ...


class FusedBrainProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze_once(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> tuple[PerceptionResult, DecisionProposal]: ...
```

共享帮助函数必须做到：

```python
def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _request_hash(sample, history, safe_candidates) -> str:
    return _canonical_hash(
        {
            "sample": sample.model_dump(mode="json"),
            "history": [item.model_dump(mode="json") for item in history],
            "safe_candidates": [item.model_dump(mode="json") for item in safe_candidates],
        }
    )


def _hold(sample: EyeSample, provider_id: str, provider_version: str, *reason_codes: str) -> DecisionProposal:
    return DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction.HOLD,
        reason_codes=tuple(reason_codes),
        provider_id=provider_id,
        provider_version=provider_version,
    )
```

最终核对函数必须逐项检查：

1. `perception.measurement_id == sample.measurement_id`；
2. `decision.based_on_measurement_id == sample.measurement_id`；
3. 无效眼图、OOD或低于阈值时覆盖为`HOLD`；
4. `PROPOSE`的`candidate_id`必须存在于传入候选集合；
5. `requested_params`必须与对应`BrainCandidate.requested_params`完全一致；
6. 任何失败都删除可执行字段；
7. `response_hash`不包含`latency_ms`，避免计时抖动破坏输出一致性比较。

- [ ] **Step 5: 实现`ComposedBrain`和`MultimodalApiBrain`**

两种实现都在最外层捕获普通`Exception`并转换为`HOLD`，但不得捕获`KeyboardInterrupt`、`SystemExit`等`BaseException`。共享实现如下；异常文本不进入reason code或哈希：

```python
def _failed_perception(sample: EyeSample, provider_id: str, provider_version: str, reason: str) -> PerceptionResult:
    return PerceptionResult(
        measurement_id=sample.measurement_id,
        quality_score=0.0,
        confidence=0.0,
        valid=False,
        ood=True,
        defects={reason.lower(): 1.0},
        feature_vector=None,
        provider_id=provider_id,
        provider_version=provider_version,
        input_hash=sample.raw_eye_density_ref.sha256,
    )


def _gate_reason(perception: PerceptionResult, minimum_confidence: float) -> str | None:
    if not perception.valid:
        return "INVALID_EYE"
    if perception.ood:
        return "OOD"
    if perception.confidence < minimum_confidence:
        return "LOW_CONFIDENCE"
    return None


def _normalize_outputs(
    sample: EyeSample,
    perception: object,
    decision: object,
    safe_candidates: tuple[BrainCandidate, ...],
    minimum_confidence: float,
) -> tuple[PerceptionResult, DecisionProposal]:
    if not isinstance(perception, PerceptionResult):
        failed = _failed_perception(sample, "brain-guard", "1", "INVALID_PERCEPTION_SCHEMA")
        return failed, _hold(sample, "brain-guard", "1", "INVALID_PERCEPTION_SCHEMA")
    if perception.measurement_id != sample.measurement_id:
        failed = _failed_perception(sample, perception.provider_id, perception.provider_version, "MEASUREMENT_ID_MISMATCH")
        return failed, _hold(sample, "brain-guard", "1", "MEASUREMENT_ID_MISMATCH")
    gate = _gate_reason(perception, minimum_confidence)
    if gate is not None:
        return perception, _hold(sample, "brain-guard", "1", gate)
    if not isinstance(decision, DecisionProposal):
        return perception, _hold(sample, "brain-guard", "1", "INVALID_DECISION_SCHEMA")
    if decision.based_on_measurement_id != sample.measurement_id:
        return perception, _hold(sample, "brain-guard", "1", "MEASUREMENT_ID_MISMATCH")
    if decision.action is not DecisionAction.PROPOSE:
        return perception, decision
    by_id = {candidate.candidate_id: candidate for candidate in safe_candidates}
    if len(by_id) != len(safe_candidates):
        return perception, _hold(sample, "brain-guard", "1", "DUPLICATE_CANDIDATE_ID")
    selected = by_id.get(decision.candidate_id)
    if selected is None or dict(selected.requested_params) != dict(decision.requested_params or {}):
        return perception, _hold(sample, "brain-guard", "1", "CANDIDATE_NOT_IN_INPUT_SET")
    return perception, decision


def _result(
    mode: BrainMode,
    request_hash: str,
    started_ns: int,
    perception: PerceptionResult,
    decision: DecisionProposal,
) -> BrainResult:
    response_hash = _canonical_hash(
        {
            "perception": perception.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "brain_mode": mode.value,
        }
    )
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=mode,
        latency_ms=(perf_counter_ns() - started_ns) / 1_000_000.0,
        request_hash=request_hash,
        response_hash=response_hash,
    )


class ComposedBrain:
    def __init__(self, *, mode: BrainMode, scorer: EyeScorer, decision_provider: DecisionProvider, minimum_confidence: float):
        if mode is BrainMode.PURE_API:
            raise ValueError("PURE_API must use MultimodalApiBrain")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be within 0..1")
        self._mode = mode
        self._scorer = scorer
        self._decision_provider = decision_provider
        self._minimum_confidence = minimum_confidence

    @property
    def provider_key(self) -> str:
        return f"{self._scorer.provider_id}|{self._decision_provider.provider_id}"

    @property
    def provider_version(self) -> str:
        return f"{self._scorer.provider_version}|{self._decision_provider.provider_version}"

    def analyze(self, sample, history, safe_candidates) -> BrainResult:
        started_ns = perf_counter_ns()
        request_hash = _request_hash(sample, history, safe_candidates)
        try:
            perception = self._scorer.score(sample)
            if not isinstance(perception, PerceptionResult):
                decision = _hold(sample, "brain-guard", "1", "INVALID_PERCEPTION_SCHEMA")
            elif perception.measurement_id != sample.measurement_id:
                decision = _hold(sample, "brain-guard", "1", "MEASUREMENT_ID_MISMATCH")
            else:
                gate = _gate_reason(perception, self._minimum_confidence)
                decision = (
                    _hold(sample, "brain-guard", "1", gate)
                    if gate is not None
                    else self._decision_provider.propose(sample, perception, history, safe_candidates)
                )
            perception, decision = _normalize_outputs(
                sample, perception, decision, safe_candidates, self._minimum_confidence
            )
        except Exception:
            perception = _failed_perception(
                sample, self._scorer.provider_id, self._scorer.provider_version, "PROVIDER_ERROR"
            )
            decision = _hold(sample, "brain-guard", "1", "PROVIDER_ERROR")
        return _result(self._mode, request_hash, started_ns, perception, decision)


class MultimodalApiBrain:
    def __init__(self, *, provider: FusedBrainProvider, minimum_confidence: float):
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be within 0..1")
        self._provider = provider
        self._minimum_confidence = minimum_confidence

    @property
    def provider_key(self) -> str:
        return self._provider.provider_id

    @property
    def provider_version(self) -> str:
        return self._provider.provider_version

    def analyze(self, sample, history, safe_candidates) -> BrainResult:
        started_ns = perf_counter_ns()
        request_hash = _request_hash(sample, history, safe_candidates)
        try:
            perception, decision = self._provider.analyze_once(sample, history, safe_candidates)
            perception, decision = _normalize_outputs(
                sample, perception, decision, safe_candidates, self._minimum_confidence
            )
        except Exception:
            perception = _failed_perception(
                sample, self._provider.provider_id, self._provider.provider_version, "PROVIDER_ERROR"
            )
            decision = _hold(sample, "brain-guard", "1", "PROVIDER_ERROR")
        return _result(BrainMode.PURE_API, request_hash, started_ns, perception, decision)
```

生产实现可增加结构化诊断日志，但日志内容不能改变稳定reason code。`ComposedBrain`先评分、通过感知门后才调用Decision；`MultimodalApiBrain`只调用一次`analyze_once()`。两者必须调用同一个最终核对函数。

同时在`src/vcsel_ai_tuning/v2/__init__.py`增加：

```python
from .brain import (
    ComposedBrain,
    DecisionProvider,
    EyeScorer,
    FusedBrainProvider,
    MultimodalApiBrain,
    OptimizationBrain,
)

__all__ += [
    "ComposedBrain",
    "DecisionProvider",
    "EyeScorer",
    "FusedBrainProvider",
    "MultimodalApiBrain",
    "OptimizationBrain",
]
```

- [ ] **Step 6: 运行Brain和失败关闭测试**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_brain.py tests_py/ai_tuning/v2/test_fail_closed.py -q -p no:cacheprovider
```

Expected: PASS；参数化测试中所有失败场景的动作均为`HOLD`。

- [ ] **Step 7: 提交Brain边界**

```powershell
git add src/vcsel_ai_tuning/v2/brain.py src/vcsel_ai_tuning/v2/__init__.py tests_py/ai_tuning/v2/test_brain.py tests_py/ai_tuning/v2/test_fail_closed.py
git commit -m "feat: add fail-closed V2 brain boundary"
```

## Task 3: 实现显式注册Factory和模式能力报告

**Files:**

- Create: `src/vcsel_ai_tuning/v2/factory.py`
- Create: `configs/ai_tuning/v2.example.yaml`
- Create: `tests_py/ai_tuning/v2/test_factory.py`

- [ ] **Step 1: 写失败测试，证明四种标识不等于四种交付**

测试只注册`LOCAL_VISION_API`，并断言其余三种仍出现在能力报告中但`installed=False`：

```python
def test_capabilities_distinguish_installed_enabled_and_available(factory, config):
    report = {item.brain_mode: item for item in factory.capabilities(config)}
    assert report[BrainMode.LOCAL_VISION_API].installed is True
    assert report[BrainMode.LOCAL_VISION_API].enabled is True
    assert report[BrainMode.LOCAL_VISION_API].available is True
    assert report[BrainMode.PURE_API].installed is False
    assert report[BrainMode.PURE_API].available is False
    assert len(report) == 4
```

其余Factory边界测试使用：

```python
def test_selected_mode_must_be_enabled():
    with pytest.raises(ValidationError, match="listed in enabled_brain_modes"):
        BrainFactoryConfig(
            brain_mode=BrainMode.PURE_API,
            enabled_brain_modes=(BrainMode.LOCAL_VISION_API,),
        )


def test_enabled_but_uninstalled_mode_is_rejected(factory):
    config = BrainFactoryConfig(
        brain_mode=BrainMode.PURE_API,
        enabled_brain_modes=(BrainMode.PURE_API,),
    )
    with pytest.raises(BrainUnavailableError, match="not installed"):
        factory.create(config)


def test_duplicate_registration_is_rejected(factory, composed_builder):
    with pytest.raises(ValueError, match="already registered"):
        factory.register(BrainMode.LOCAL_VISION_API, composed_builder)


def test_selected_builder_failure_does_not_call_another_builder(config):
    calls = []
    factory = BrainFactory()

    def selected(_config):
        calls.append("selected")
        raise RuntimeError("injected builder failure")

    def fallback(_config):
        calls.append("fallback")
        return object()

    factory.register(BrainMode.LOCAL_VISION_API, selected)
    factory.register(BrainMode.LOCAL_VISION_LOCAL, fallback)
    with pytest.raises(RuntimeError, match="injected builder failure"):
        factory.create(config)
    assert calls == ["selected"]


def test_example_yaml_selects_only_an_enabled_mode():
    config = load_v2_config(Path("configs/ai_tuning/v2.example.yaml"))
    assert config.brain_mode in config.enabled_brain_modes
    assert config.enabled_brain_modes == (BrainMode.LOCAL_VISION_API,)
```

- [ ] **Step 2: 运行Factory测试并确认失败**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_factory.py -q -p no:cacheprovider
```

Expected: FAIL because `vcsel_ai_tuning.v2.factory` does not exist.

- [ ] **Step 3: 实现严格配置和能力模型**

`factory.py`至少定义：

```python
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BeforeValidator, Field, model_validator

from .brain import OptimizationBrain
from .contracts import BrainMode, FrozenV2Model


class BrainUnavailableError(RuntimeError):
    pass


def _brain_mode_from_wire(value: object) -> object:
    if isinstance(value, BrainMode):
        return value
    if type(value) is str:
        return BrainMode(value)
    return value


def _brain_modes_from_wire(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_brain_mode_from_wire(item) for item in value)
    return value


BrainModeValue = Annotated[BrainMode, BeforeValidator(_brain_mode_from_wire)]
BrainModesValue = Annotated[
    tuple[BrainMode, ...],
    BeforeValidator(_brain_modes_from_wire),
]


class BrainFactoryConfig(FrozenV2Model):
    brain_mode: BrainModeValue
    enabled_brain_modes: BrainModesValue = Field(min_length=1)
    minimum_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_selection(self) -> "BrainFactoryConfig":
        if len(set(self.enabled_brain_modes)) != len(self.enabled_brain_modes):
            raise ValueError("enabled_brain_modes must be unique")
        if self.brain_mode not in self.enabled_brain_modes:
            raise ValueError("brain_mode must be listed in enabled_brain_modes")
        return self


class BrainCapability(FrozenV2Model):
    brain_mode: BrainMode
    installed: bool
    enabled: bool
    available: bool
    reason: str | None = None


BrainBuilder = Callable[[BrainFactoryConfig], OptimizationBrain]


class V2ConfigDocument(FrozenV2Model):
    v2: BrainFactoryConfig
```

`BrainFactory`使用私有Builder映射，`register()`拒绝重复，`create()`先检查enabled再检查installed，然后只调用所选Builder一次。`capabilities()`始终按`BrainMode`枚举顺序返回四条报告，不导入未注册Provider。实现代码固定为：

```python
class BrainFactory:
    def __init__(self) -> None:
        self._builders: dict[BrainMode, BrainBuilder] = {}

    def register(self, mode: BrainMode, builder: BrainBuilder) -> None:
        if not isinstance(mode, BrainMode):
            raise TypeError("mode must be BrainMode")
        if not callable(builder):
            raise TypeError("builder must be callable")
        if mode in self._builders:
            raise ValueError(f"brain mode {mode.value} is already registered")
        self._builders[mode] = builder

    def capabilities(self, config: BrainFactoryConfig) -> tuple[BrainCapability, ...]:
        enabled = frozenset(config.enabled_brain_modes)
        result = []
        for mode in BrainMode:
            installed = mode in self._builders
            is_enabled = mode in enabled
            available = installed and is_enabled
            reason = None
            if not installed:
                reason = "NOT_INSTALLED"
            elif not is_enabled:
                reason = "NOT_ENABLED"
            result.append(
                BrainCapability(
                    brain_mode=mode,
                    installed=installed,
                    enabled=is_enabled,
                    available=available,
                    reason=reason,
                )
            )
        return tuple(result)

    def create(self, config: BrainFactoryConfig) -> OptimizationBrain:
        mode = config.brain_mode
        if mode not in config.enabled_brain_modes:
            raise BrainUnavailableError(f"brain mode {mode.value} is not enabled")
        builder = self._builders.get(mode)
        if builder is None:
            raise BrainUnavailableError(f"brain mode {mode.value} is not installed")
        brain = builder(config)
        if not isinstance(brain, OptimizationBrain):
            raise TypeError("brain builder returned an incompatible object")
        return brain
```

- [ ] **Step 4: 实现YAML读取和首版示例**

`load_v2_config()`是独立加载器，不复用或修改`RuntimeConfig`：

```python
def load_v2_config(path: str | Path) -> BrainFactoryConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return V2ConfigDocument.model_validate(data).v2
```

`configs/ai_tuning/v2.example.yaml`内容固定为：

```yaml
v2:
  brain_mode: LOCAL_VISION_API
  enabled_brain_modes:
    - LOCAL_VISION_API
  minimum_confidence: 0.70
```

该文件不出现API Key，不把未实现模式标为启用，也不包含真实模型路径。

- [ ] **Step 5: 运行Factory测试并确认通过**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_factory.py -q -p no:cacheprovider
```

Expected: PASS；能力报告包含四个标识，但只有测试注册且配置启用的模式`available=True`。

- [ ] **Step 6: 提交Factory**

```powershell
git add src/vcsel_ai_tuning/v2/factory.py configs/ai_tuning/v2.example.yaml tests_py/ai_tuning/v2/test_factory.py
git commit -m "feat: add explicit V2 brain registry"
```

## Task 4: 实现二维眼图密度矩阵的确定性预处理

**Files:**

- Create: `src/vcsel_ai_tuning/v2/preprocessing.py`
- Create: `tests_py/ai_tuning/v2/test_preprocessing.py`

- [ ] **Step 1: 写失败测试，固定形状、方向和数据来源**

测试通过`tmp_path`创建`.npy`，更新`EyeSample`中的引用和SHA-256。至少包含：

```python
def test_preprocess_density_returns_fixed_rgb_tensor(sample_with_density, tmp_path):
    image = preprocess_eye_density(sample_with_density, tmp_path)
    assert image.shape == (280, 392, 3)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert np.array_equal(image[:, :, 0], image[:, :, 1])
    assert np.array_equal(image[:, :, 1], image[:, :, 2])


def test_preprocess_is_deterministic(sample_with_density, tmp_path):
    first = preprocess_eye_density(sample_with_density, tmp_path)
    second = preprocess_eye_density(sample_with_density, tmp_path)
    assert np.array_equal(first, second)


def test_preprocess_preserves_vertical_orientation(oriented_sample, tmp_path):
    image = preprocess_eye_density(oriented_sample, tmp_path)
    assert image[4:40].sum() > image[-40:-4].sum()
```

测试文件使用同一个样本构造帮助函数：

```python
from hashlib import sha256

import numpy as np
import pytest

from vcsel_ai_tuning.v2.contracts import ArtifactReference, EyeSample


def sample_for_array(eye_sample, tmp_path, array, *, media_type="application/x-npy"):
    relative_path = "point-001/raw_eye_density.npy"
    path = tmp_path.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True)
    np.save(path, array, allow_pickle=False)
    digest = sha256(path.read_bytes()).hexdigest()
    values = {
        name: getattr(eye_sample, name)
        for name in EyeSample.model_fields
    }
    values["raw_eye_density_ref"] = ArtifactReference(
        relative_path=relative_path,
        sha256=digest,
        media_type=media_type,
    )
    values["artifact_hashes"] = {relative_path: digest}
    return EyeSample(**values)


@pytest.mark.parametrize(
    "array",
    [
        np.ones((2, 2, 2), dtype=np.float32),
        np.asarray([[1.0, -1.0]], dtype=np.float32),
        np.asarray([[1.0, np.nan]], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ],
)
def test_invalid_density_matrices_are_rejected(eye_sample, tmp_path, array):
    sample = sample_for_array(eye_sample, tmp_path, array)
    with pytest.raises(EyeDensityError):
        preprocess_eye_density(sample, tmp_path)


def test_density_hash_mismatch_is_rejected(eye_sample, tmp_path):
    sample = sample_for_array(eye_sample, tmp_path, np.ones((2, 2)))
    path = tmp_path / "point-001" / "raw_eye_density.npy"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(EyeDensityError, match="hash mismatch"):
        preprocess_eye_density(sample, tmp_path)


def test_non_npy_media_type_is_rejected(eye_sample, tmp_path):
    sample = sample_for_array(
        eye_sample,
        tmp_path,
        np.ones((2, 2)),
        media_type="image/png",
    )
    with pytest.raises(EyeDensityError, match="application/x-npy"):
        preprocess_eye_density(sample, tmp_path)
```

路径越界在`ArtifactReference`合同测试中已经被拒绝；预处理仍保留resolve后的根目录检查作为第二道防线。完整DCA屏幕和PNG无法通过media type检查。

- [ ] **Step 2: 运行预处理测试并确认失败**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_preprocessing.py -q -p no:cacheprovider
```

Expected: FAIL because `vcsel_ai_tuning.v2.preprocessing` does not exist.

- [ ] **Step 3: 实现安全文件解析和哈希验证**

`preprocessing.py`先完成路径与文件验证：

```python
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from .contracts import EyeSample


class EyeDensityError(ValueError):
    pass


def _resolve_density_path(sample: EyeSample, artifact_root: Path) -> Path:
    reference = sample.raw_eye_density_ref
    if reference.media_type != "application/x-npy":
        raise EyeDensityError("raw eye density must be application/x-npy")
    root = artifact_root.resolve()
    path = root.joinpath(*reference.relative_path.split("/")).resolve()
    if not path.is_relative_to(root):
        raise EyeDensityError("raw eye density path escapes artifact root")
    if not path.is_file():
        raise EyeDensityError("raw eye density file is missing")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != reference.sha256:
        raise EyeDensityError("raw eye density hash mismatch")
    return path


def _load_density(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EyeDensityError("raw eye density cannot be loaded") from error
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
        raise EyeDensityError("raw eye density must be a numeric 2D matrix")
    density = np.asarray(value, dtype=np.float64)
    if 0 in density.shape:
        raise EyeDensityError("raw eye density dimensions must be nonzero")
    if not np.isfinite(density).all():
        raise EyeDensityError("raw eye density must contain only finite values")
    if (density < 0.0).any():
        raise EyeDensityError("raw eye density cannot contain negative counts")
    if not (density > 0.0).any():
        raise EyeDensityError("raw eye density cannot be all zero")
    return density
```

使用`np.load(path, allow_pickle=False)`；只接受二维数值数组，随后转换为`float64`做验证。负数、非有限数、空维度和全零均明确拒绝。

- [ ] **Step 4: 实现固定对数归一化、等比例缩放和填充**

配置与算法固定为：

```python
@dataclass(frozen=True, slots=True)
class EyeDensityPreprocessConfig:
    content_width: int = 392
    content_height: int = 272
    canvas_width: int = 392
    canvas_height: int = 280
    count_clip: float = 4095.0

    def __post_init__(self) -> None:
        dimensions = (
            self.content_width,
            self.content_height,
            self.canvas_width,
            self.canvas_height,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("preprocess dimensions must be positive integers")
        if self.content_width > self.canvas_width or self.content_height > self.canvas_height:
            raise ValueError("content dimensions must fit inside the canvas")
        if not np.isfinite(self.count_clip) or self.count_clip <= 0.0:
            raise ValueError("count_clip must be a positive finite value")


def _normalize_counts(density: np.ndarray, count_clip: float) -> np.ndarray:
    clipped = np.clip(density, 0.0, count_clip)
    return np.log1p(clipped) / np.log1p(count_clip)
```

缩放和公开入口使用以下实现：

```python
def _resize_into_content(normalized: np.ndarray, config: EyeDensityPreprocessConfig) -> np.ndarray:
    source_height, source_width = normalized.shape
    scale = min(
        config.content_width / source_width,
        config.content_height / source_height,
    )
    target_width = max(1, min(config.content_width, round(source_width * scale)))
    target_height = max(1, min(config.content_height, round(source_height * scale)))
    resized = zoom(
        normalized,
        (target_height / source_height, target_width / source_width),
        order=1,
        prefilter=False,
        mode="nearest",
        grid_mode=True,
    )
    exact = np.zeros((target_height, target_width), dtype=np.float64)
    copy_height = min(target_height, resized.shape[0])
    copy_width = min(target_width, resized.shape[1])
    exact[:copy_height, :copy_width] = resized[:copy_height, :copy_width]
    content = np.zeros(
        (config.content_height, config.content_width),
        dtype=np.float64,
    )
    top = (config.content_height - target_height) // 2
    left = (config.content_width - target_width) // 2
    content[top : top + target_height, left : left + target_width] = exact
    return content


def preprocess_eye_density(
    sample: EyeSample,
    artifact_root: str | Path,
    config: EyeDensityPreprocessConfig = EyeDensityPreprocessConfig(),
) -> np.ndarray:
    if config.content_width > config.canvas_width or config.content_height > config.canvas_height:
        raise EyeDensityError("content dimensions must fit inside the canvas")
    path = _resolve_density_path(sample, Path(artifact_root))
    density = _load_density(path)
    normalized = _normalize_counts(density, config.count_clip)
    content = _resize_into_content(normalized, config)
    canvas = np.zeros((config.canvas_height, config.canvas_width), dtype=np.float64)
    top = (config.canvas_height - config.content_height) // 2
    left = (config.canvas_width - config.content_width) // 2
    canvas[
        top : top + config.content_height,
        left : left + config.content_width,
    ] = content
    rgb = np.repeat(canvas[:, :, None], 3, axis=2)
    return np.ascontiguousarray(rgb, dtype=np.float32)
```

代码中不得出现翻转、旋转、随机数或基于单张图片最大值的归一化。

- [ ] **Step 5: 运行预处理测试并确认通过**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_preprocessing.py -q -p no:cacheprovider
```

Expected: PASS；同一输入逐元素完全一致，方向测试证明上方命中不会移动到下方。

- [ ] **Step 6: 提交预处理实现**

```powershell
git add src/vcsel_ai_tuning/v2/preprocessing.py tests_py/ai_tuning/v2/test_preprocessing.py
git commit -m "feat: add deterministic eye density preprocessing"
```

## Task 5: 实现FAKE Provider和统一replay入口

**Files:**

- Create: `src/vcsel_ai_tuning/v2/providers/__init__.py`
- Create: `src/vcsel_ai_tuning/v2/providers/fake.py`
- Create: `src/vcsel_ai_tuning/v2/replay.py`
- Create: `tests_py/ai_tuning/v2/test_replay.py`

- [ ] **Step 1: 写失败测试，要求两种Brain走同一replay接口**

一种Brain使用`FakeEyeScorer + FirstCandidateDecisionProvider + ComposedBrain`模拟首版`LOCAL_VISION_API`，另一种使用`ScriptedFakeBrain`模拟可替换实现。两者都调用：

```python
records = replay(brain, cases)
```

测试断言：

```python
def test_composed_and_scripted_brains_share_replay_contract(composed_brain, scripted_brain, replay_cases):
    composed = replay(composed_brain, replay_cases)
    scripted = replay(scripted_brain, replay_cases)
    assert [item.measurement_id for item in composed] == [item.measurement_id for item in scripted]
    assert all(item.result.brain_mode is BrainMode.LOCAL_VISION_API for item in composed)
    assert all(item.result.decision.based_on_measurement_id == item.measurement_id for item in scripted)
```

其余replay边界使用：

```python
def test_empty_candidate_set_returns_hold(composed_brain, replay_cases):
    case = replay_cases[0].model_copy(update={"safe_candidates": ()})
    record = replay(composed_brain, (case,))[0]
    assert record.result.decision.action is DecisionAction.HOLD
    assert record.result.decision.reason_codes == ("NO_SAFE_CANDIDATE",)


def test_duplicate_measurement_ids_are_rejected(composed_brain, replay_cases):
    duplicate = replay_cases[0].model_copy(
        update={"sample": replay_cases[0].sample.model_copy()}
    )
    with pytest.raises(ValueError, match="measurement IDs must be unique"):
        replay(composed_brain, (replay_cases[0], duplicate))


def test_foreign_result_measurement_id_is_rejected(foreign_result, replay_cases):
    brain = ScriptedFakeBrain((foreign_result,))
    with pytest.raises(ValidationError, match="measurement ID mismatch"):
        replay(brain, (replay_cases[0],))


def test_jsonl_round_trip(composed_brain, replay_cases, tmp_path):
    records = replay(composed_brain, replay_cases)
    path = write_replay_jsonl(records, tmp_path / "v2-replay.jsonl")
    restored = tuple(
        ReplayRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert restored == records
```

这里`model_copy(update=...)`只用于制造已经通过合同的测试输入组合，不用于验证Pydantic跨字段规则；跨字段规则仍通过重新构造模型测试。

- [ ] **Step 2: 运行replay测试并确认失败**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_replay.py -q -p no:cacheprovider
```

Expected: FAIL because FAKE Provider and replay module do not exist.

- [ ] **Step 3: 实现确定性FAKE Provider**

`providers/fake.py`使用显眼的FAKE身份，并且不导入Factory、API、硬件、持久化或网络库：

```python
from collections import deque

from ..contracts import DecisionAction, DecisionProposal, PerceptionResult


class FakeEyeScorer:
    provider_id = "fake-eye-scorer"
    provider_version = "foundation-1"

    def __init__(self, *, quality_score=0.75, confidence=0.95, valid=True, ood=False):
        self._quality_score = quality_score
        self._confidence = confidence
        self._valid = valid
        self._ood = ood
        self.calls: list[str] = []

    def score(self, sample):
        self.calls.append(sample.measurement_id)
        return PerceptionResult(
            measurement_id=sample.measurement_id,
            quality_score=self._quality_score,
            confidence=self._confidence,
            valid=self._valid,
            ood=self._ood,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            input_hash=sample.raw_eye_density_ref.sha256,
        )


class FirstCandidateDecisionProvider:
    provider_id = "fake-first-candidate"
    provider_version = "foundation-1"

    def __init__(self):
        self.calls: list[str] = []

    def propose(self, sample, perception, history, safe_candidates):
        self.calls.append(sample.measurement_id)
        if not safe_candidates:
            return DecisionProposal(
                based_on_measurement_id=sample.measurement_id,
                action=DecisionAction.HOLD,
                reason_codes=("NO_SAFE_CANDIDATE",),
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        selected = safe_candidates[0]
        return DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=selected.candidate_id,
            requested_params=selected.requested_params,
            reason_codes=("DETERMINISTIC_FAKE_SELECTION",),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


class ScriptedFakeBrain:
    """Test-only independent OptimizationBrain; script exhaustion is a test failure."""

    def __init__(self, results):
        self._results = deque(results)
        self.calls: list[str] = []

    @property
    def provider_key(self):
        return "scripted-fake-brain"

    @property
    def provider_version(self):
        return "foundation-1"

    def analyze(self, sample, history, safe_candidates):
        self.calls.append(sample.measurement_id)
        if not self._results:
            raise AssertionError("scripted fake brain result queue is exhausted")
        return self._results.popleft()
```

- [ ] **Step 4: 实现replay合同和JSONL输出**

`replay.py`定义：

```python
from pathlib import Path

from pydantic import Field, FiniteFloat, model_validator

from .brain import OptimizationBrain
from .contracts import (
    BrainCandidate,
    BrainResult,
    EyeSample,
    FrozenV2Model,
    Identifier,
    NonBlank,
    V2Observation,
)

MAX_REPLAY_CASES = 4096
MAX_REPLAY_CANDIDATES = 4096
MAX_REPLAY_HISTORY = 4096


class ReplayCase(FrozenV2Model):
    sample: EyeSample
    history: tuple[V2Observation, ...] = Field(default=(), max_length=MAX_REPLAY_HISTORY)
    safe_candidates: tuple[BrainCandidate, ...] = Field(default=(), max_length=MAX_REPLAY_CANDIDATES)
    estimated_cost_usd: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "ReplayCase":
        candidate_ids = tuple(item.candidate_id for item in self.safe_candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("replay candidate IDs must be unique")
        return self


class ReplayRecord(FrozenV2Model):
    sequence: int = Field(ge=0)
    campaign_id: Identifier
    sweep_point_id: Identifier
    measurement_id: Identifier
    brain_provider_key: NonBlank
    brain_provider_version: NonBlank
    result: BrainResult
    estimated_cost_usd: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_result_binding(self) -> "ReplayRecord":
        if self.result.perception.measurement_id != self.measurement_id:
            raise ValueError("replay result measurement ID mismatch")
        return self


def replay(
    brain: OptimizationBrain,
    cases: tuple[ReplayCase, ...],
) -> tuple[ReplayRecord, ...]:
    if not isinstance(brain, OptimizationBrain):
        raise TypeError("brain must implement OptimizationBrain")
    if not isinstance(cases, tuple):
        raise TypeError("replay cases must be a tuple")
    if not cases or len(cases) > MAX_REPLAY_CASES:
        raise ValueError(f"replay requires 1..{MAX_REPLAY_CASES} cases")
    measurement_ids = tuple(case.sample.measurement_id for case in cases)
    if len(measurement_ids) != len(set(measurement_ids)):
        raise ValueError("replay measurement IDs must be unique")
    records = []
    for sequence, case in enumerate(cases):
        result = brain.analyze(case.sample, case.history, case.safe_candidates)
        records.append(
            ReplayRecord(
                sequence=sequence,
                campaign_id=case.sample.campaign_id,
                sweep_point_id=case.sample.sweep_point_id,
                measurement_id=case.sample.measurement_id,
                brain_provider_key=brain.provider_key,
                brain_provider_version=brain.provider_version,
                result=result,
                estimated_cost_usd=case.estimated_cost_usd,
            )
        )
    return tuple(records)


def write_replay_jsonl(records: tuple[ReplayRecord, ...], path: str | Path) -> Path:
    if not records:
        raise ValueError("replay records cannot be empty")
    for expected_sequence, record in enumerate(records):
        if record.sequence != expected_sequence:
            raise ValueError("replay record sequence must be contiguous")
    output = Path(path)
    output.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    return output
```

该模块只记录输入对应的模型结果，不创建“调整后眼图”，也不调用`SafetyPolicy`、执行器、readback或rollback。正式上线时必须把选中的ID映射回原始Campaign候选并通过现有授权链；Foundation replay不声称完成这一步。

- [ ] **Step 5: 运行replay测试并确认通过**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_replay.py -q -p no:cacheprovider
```

Expected: PASS；两个Brain实现无需修改replay函数即可运行。

- [ ] **Step 6: 提交FAKE/replay**

```powershell
git add src/vcsel_ai_tuning/v2/providers/__init__.py src/vcsel_ai_tuning/v2/providers/fake.py src/vcsel_ai_tuning/v2/replay.py tests_py/ai_tuning/v2/test_replay.py
git commit -m "feat: add V2 fake providers and replay"
```

## Task 6: 实现跨Provider评估记录

**Files:**

- Create: `src/vcsel_ai_tuning/v2/evaluation.py`
- Create: `tests_py/ai_tuning/v2/test_evaluation.py`

- [ ] **Step 1: 写失败测试，固定单Provider统计和双Provider对照**

测试覆盖：

```python
def test_summary_reports_latency_hold_rate_and_cost(records):
    summary = summarize_provider(records)
    assert summary.sample_count == 4
    assert summary.hold_count == 1
    assert summary.hold_rate == pytest.approx(0.25)
    assert summary.p50_latency_ms == pytest.approx(2.5)
    assert summary.total_estimated_cost_usd == pytest.approx(0.04)


def test_compare_requires_matching_measurement_ids(provider_a_records, provider_b_records):
    comparison = compare_providers(provider_a_records, provider_b_records)
    assert comparison.matched_sample_count == len(provider_a_records)
    assert comparison.provider_a_key != comparison.provider_b_key
```

再增加重复运行一致性测试和三组人工A/B标签的排序准确率测试。只有一个Provider时可以生成单Provider摘要，不要求跨Provider比较。

- [ ] **Step 2: 运行评估测试并确认失败**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_evaluation.py -q -p no:cacheprovider
```

Expected: FAIL because `vcsel_ai_tuning.v2.evaluation` does not exist.

- [ ] **Step 3: 实现评估数据结构**

`evaluation.py`定义严格模型：

```python
class ProviderSummary(FrozenV2Model):
    provider_key: NonBlank
    provider_versions: NonBlank
    brain_mode: BrainMode
    sample_count: int = Field(ge=0)
    propose_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)
    hold_rate: float = Field(ge=0.0, le=1.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    repeated_output_agreement: float = Field(ge=0.0, le=1.0)
    total_estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class ProviderComparison(FrozenV2Model):
    provider_a_key: NonBlank
    provider_b_key: NonBlank
    matched_sample_count: int = Field(ge=1)
    quality_score_mae: float = Field(ge=0.0)
    decision_agreement: float = Field(ge=0.0, le=1.0)


class RankingLabel(FrozenV2Model):
    better_measurement_id: Identifier
    worse_measurement_id: Identifier
```

组合Brain由感知和Decision两个Provider组成，因此`ComposedBrain.provider_key`固定为`<perception provider_id>|<decision provider_id>`，并由replay显式写入记录；失败关闭生成的guard Decision不会改变Brain身份。统计函数实现为：

```python
from collections import defaultdict

import numpy as np

from .contracts import DecisionAction
from .replay import ReplayRecord


def _provider_key(record: ReplayRecord) -> tuple[str, str]:
    return (record.brain_provider_key, record.brain_provider_version)


def _semantic_output(record: ReplayRecord) -> tuple[object, object]:
    return (record.result.perception, record.result.decision)


def summarize_provider(records: tuple[ReplayRecord, ...]) -> ProviderSummary:
    if not records:
        raise ValueError("provider summary requires records")
    identities = {_provider_key(record) for record in records}
    modes = {record.result.brain_mode for record in records}
    if len(identities) != 1 or len(modes) != 1:
        raise ValueError("provider summary cannot mix providers or brain modes")
    provider_key, provider_versions = identities.pop()
    mode = modes.pop()
    latencies = np.asarray(
        [record.result.latency_ms for record in records],
        dtype=np.float64,
    )
    grouped = defaultdict(list)
    for record in records:
        grouped[record.measurement_id].append(_semantic_output(record))
    repeat_pairs = [
        outputs
        for outputs in grouped.values()
        if len(outputs) > 1
    ]
    repeated_output_agreement = (
        1.0
        if not repeat_pairs
        else sum(
            all(output == outputs[0] for output in outputs[1:])
            for outputs in repeat_pairs
        ) / len(repeat_pairs)
    )
    hold_count = sum(
        record.result.decision.action is DecisionAction.HOLD
        for record in records
    )
    costs = [record.estimated_cost_usd for record in records]
    total_cost = None if any(cost is None for cost in costs) else float(sum(costs))
    return ProviderSummary(
        provider_key=provider_key,
        provider_versions=provider_versions,
        brain_mode=mode,
        sample_count=len(records),
        propose_count=sum(
            record.result.decision.action is DecisionAction.PROPOSE
            for record in records
        ),
        hold_count=hold_count,
        hold_rate=hold_count / len(records),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        repeated_output_agreement=repeated_output_agreement,
        total_estimated_cost_usd=total_cost,
    )


def _unique_by_measurement(records: tuple[ReplayRecord, ...]) -> dict[str, ReplayRecord]:
    result = {record.measurement_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("comparison inputs require unique measurement IDs")
    return result


def compare_providers(
    provider_a: tuple[ReplayRecord, ...],
    provider_b: tuple[ReplayRecord, ...],
) -> ProviderComparison:
    by_a = _unique_by_measurement(provider_a)
    by_b = _unique_by_measurement(provider_b)
    if not by_a or set(by_a) != set(by_b):
        raise ValueError("provider comparison requires identical measurement IDs")
    key_a = summarize_provider(provider_a).provider_key
    key_b = summarize_provider(provider_b).provider_key
    if key_a == key_b:
        raise ValueError("provider comparison requires two distinct provider keys")
    ordered_ids = sorted(by_a)
    score_errors = [
        abs(
            by_a[item].result.perception.quality_score
            - by_b[item].result.perception.quality_score
        )
        for item in ordered_ids
    ]
    decision_agreement = sum(
        by_a[item].result.decision.action == by_b[item].result.decision.action
        and by_a[item].result.decision.candidate_id
        == by_b[item].result.decision.candidate_id
        for item in ordered_ids
    ) / len(ordered_ids)
    return ProviderComparison(
        provider_a_key=key_a,
        provider_b_key=key_b,
        matched_sample_count=len(ordered_ids),
        quality_score_mae=float(np.mean(score_errors)),
        decision_agreement=decision_agreement,
    )


def ranking_accuracy(
    records: tuple[ReplayRecord, ...],
    labels: tuple[RankingLabel, ...],
) -> float:
    if not labels:
        raise ValueError("ranking accuracy requires labels")
    by_measurement = _unique_by_measurement(records)
    correct = 0
    for label in labels:
        try:
            better = by_measurement[label.better_measurement_id]
            worse = by_measurement[label.worse_measurement_id]
        except KeyError as error:
            raise ValueError("ranking label references an unknown measurement") from error
        correct += (
            better.result.perception.quality_score
            > worse.result.perception.quality_score
        )
    return correct / len(labels)
```

重复一致性明确排除计时和Brain哈希。`compare_providers()`要求两边measurement ID集合完全一致。`ranking_accuracy()`只使用显式人工标签和`quality_score`，平分记为错误，不从TDECQ或ER推断标签。

- [ ] **Step 4: 运行评估测试并确认通过**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2/test_evaluation.py -q -p no:cacheprovider
```

Expected: PASS；一个Provider可独立汇总，只有传入两个Provider时才执行对照。

- [ ] **Step 5: 提交评估框架**

```powershell
git add src/vcsel_ai_tuning/v2/evaluation.py tests_py/ai_tuning/v2/test_evaluation.py
git commit -m "feat: add V2 provider evaluation records"
```

## Task 7: 集成Factory、验证硬件隔离并运行回归

**Files:**

- Create: `tests_py/ai_tuning/v2/test_import_isolation.py`
- Modify: `src/vcsel_ai_tuning/v2/__init__.py`
- Modify: `tests_py/ai_tuning/v2/test_factory.py`

- [ ] **Step 1: 写端到端FAKE测试，完成Factory接线**

在`test_factory.py`中注册一个Builder，Builder返回：

```python
ComposedBrain(
    mode=BrainMode.LOCAL_VISION_API,
    scorer=FakeEyeScorer(quality_score=0.8, confidence=0.95),
    decision_provider=FirstCandidateDecisionProvider(),
    minimum_confidence=config.minimum_confidence,
)
```

从`configs/ai_tuning/v2.example.yaml`读取配置，通过Factory创建Brain，然后对两个`ReplayCase`运行`replay()`。断言两条记录合法，并确认能力报告只有`LOCAL_VISION_API`为available。

```python
def test_example_config_runs_one_fake_target_mode(replay_cases):
    config = load_v2_config(Path("configs/ai_tuning/v2.example.yaml"))
    factory = BrainFactory()

    def build_composed(selected_config):
        return ComposedBrain(
            mode=BrainMode.LOCAL_VISION_API,
            scorer=FakeEyeScorer(quality_score=0.8, confidence=0.95),
            decision_provider=FirstCandidateDecisionProvider(),
            minimum_confidence=selected_config.minimum_confidence,
        )

    factory.register(BrainMode.LOCAL_VISION_API, build_composed)
    brain = factory.create(config)
    records = replay(brain, replay_cases[:2])
    assert len(records) == 2
    available = [
        item.brain_mode
        for item in factory.capabilities(config)
        if item.available
    ]
    assert available == [BrainMode.LOCAL_VISION_API]
```

在`src/vcsel_ai_tuning/v2/__init__.py`最后增加纯Factory导出；不要从这里导出FAKE、preprocessing、replay或任何可选Provider：

```python
from .factory import (
    BrainCapability,
    BrainFactory,
    BrainFactoryConfig,
    BrainUnavailableError,
    load_v2_config,
)

__all__ += [
    "BrainCapability",
    "BrainFactory",
    "BrainFactoryConfig",
    "BrainUnavailableError",
    "load_v2_config",
]
```

- [ ] **Step 2: 写AST硬件隔离测试**

`test_import_isolation.py`遍历`src/vcsel_ai_tuning/v2/**/*.py`并解析所有`Import`和`ImportFrom`。以下前缀一律视为违规：

```python
BANNED_IMPORT_ROOTS = {
    "vcsel_ai_tuning.hardware",
    "vcsel_ai_tuning.backends",
    "vcsel_ai_tuning.adapters",
    "vcsel_ai_tuning.hardware_authority",
    "vcsel_ai_tuning.persistence",
    "vcsel_ai_tuning.campaign.orchestrator",
    "vcsel_ai_tuning.v1_1.hardware_control",
}
```

完整扫描实现为：

```python
import ast
from importlib.util import resolve_name
from pathlib import Path


V2_ROOT = Path(__file__).parents[3] / "src" / "vcsel_ai_tuning" / "v2"
SOURCE_ROOT = V2_ROOT.parents[1]


def _resolved_imports(path: Path, node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if node.level:
        package = ".".join(path.relative_to(SOURCE_ROOT).parent.parts)
        base = resolve_name("." * node.level + (node.module or ""), package)
    else:
        base = node.module or ""
    return (base,) if base else ()


def test_v2_package_has_no_hardware_execution_or_persistence_imports():
    violations = []
    for path in sorted(V2_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for module_name in _resolved_imports(path, node):
                if any(
                    module_name == root or module_name.startswith(root + ".")
                    for root in BANNED_IMPORT_ROOTS
                ):
                    violations.append(
                        f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {module_name}"
                    )
    assert violations == []
```

再增加本阶段的源码文本扫描：

```python
FOUNDATION_BANNED_TEXT = ("setFir", "pyvisa", "import serial", "import socket", "socket.")


def test_v2_foundation_contains_no_hardware_command_or_network_text():
    violations = []
    for path in sorted(V2_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for token in FOUNDATION_BANNED_TEXT:
            if token in source:
                violations.append(f"{path.relative_to(SOURCE_ROOT)}: {token}")
    assert violations == []
```

网络库不在永久禁止列表中，因为独立Kimi Provider计划将需要HTTP客户端；但Foundation文件本身仍不得导入网络库。

- [ ] **Step 3: 运行完整V2测试**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/v2 -q -p no:cacheprovider
```

Expected: PASS；无跳过测试。

- [ ] **Step 4: 运行现有安全和优化器回归**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest `
  tests_py/ai_tuning/test_runtime_config.py `
  tests_py/ai_tuning/test_runtime_factories.py `
  tests_py/ai_tuning/test_runtime_registry.py `
  tests_py/ai_tuning/test_package_import.py `
  tests_py/ai_tuning/test_safety.py `
  tests_py/ai_tuning/campaign/test_authorization.py `
  tests_py/ai_tuning/campaign/test_repository.py `
  tests_py/ai_tuning/optimization `
  tests_py/ai_tuning/v1_1 `
  -q -p no:cacheprovider
```

Expected: PASS；现有V1.1和TDECQ优化路径没有回归。

- [ ] **Step 5: 运行项目测试入口和静态差异检查**

Run:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning -q -p no:cacheprovider
git diff --check
```

Expected: pytest PASS；`git diff --check`无错误。

- [ ] **Step 6: 检查验收证据**

逐项记录以下证据：

- 示例配置只启用`LOCAL_VISION_API`；
- 能力报告仍列出四个架构标识，但未安装模式不可用；
- 组合FAKE和脚本FAKE通过同一replay入口；
- Provider异常、低置信度、OOD和非法候选均返回无执行字段的`HOLD`；
- V2源码没有硬件或执行器导入；
- 预处理只读取哈希匹配的二维`.npy`；
- 当前V1.1测试保持通过。

- [ ] **Step 7: 提交V2 Foundation集成**

```powershell
git add src/vcsel_ai_tuning/v2 tests_py/ai_tuning/v2 configs/ai_tuning/v2.example.yaml
git commit -m "feat: integrate V2 foundation replay flow"
```

## 4. Foundation完成定义

只有同时满足以下条件才可宣布本计划完成：

1. 至少一个配置启用的FAKE目标模式可以通过Factory和replay运行；
2. 一个独立脚本FAKE也能走同一`OptimizationBrain`入口；
3. 未安装或未启用模式明确不可用，不发生静默回退；
4. 所有感知门和Provider失败都产生无可执行字段的`HOLD`；
5. 任何`PROPOSE`只能精确选择调用方传入的候选；
6. 眼图预处理结果固定为`280×392×3 float32`，方向不变且可重复；
7. replay和评估格式允许一个Provider独立运行，并在存在两个Provider时直接对照；
8. V2包没有硬件、执行器或现有主循环导入；
9. V1.1、安全层和现有优化器测试保持通过；
10. 没有对真实API、真实模型准确率或真实眼图改善作出声明。

## 5. Foundation之后的独立计划入口

Foundation合并后按实际产品选择创建，不要求四项同时开展：

1. `Kimi Decision Provider`：凭据注入、请求/响应Schema、超时/429、Prompt泄漏审查、成本记录和FAKE HTTP测试；
2. `DINOv2 Eye Scorer`：数据集切分、确定性预处理对接、冻结主干训练、置信度/OOD校准和Windows GPU基准；
3. `Local Decision Adapter`：把现有`OptimizerProtocol`结果映射为V2 Decision，明确处理`CandidateSuggestion`、`NoSafeCandidate`和终止结果，并拒绝整数设置到浮点候选之间的有损转换；
4. `V1 EyeSample Adapter`：单次采集绑定、Outer ER统一来源、原始/均衡眼图分离、manifest和哈希验证；
5. `Shadow Mode Integration`：Campaign冻结Brain配置、只读Dashboard审计、真实样本只评分和零硬件权限证明。

每个计划都必须以自己的数据或授权门槛为开始条件，不因Foundation存在而自动启用。
