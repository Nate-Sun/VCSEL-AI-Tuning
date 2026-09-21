# V2.1 Kimi Live I2C Minimal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 V2.0 `PURE_API` 底座中接入真实 Kimi K3，并把合法候选通过 V2.0 安全链真实写入 I2C、回读和重新测量。

**Architecture:** Kimi 只接收经过固定预处理的眼图、初始指示和本地候选表，并只返回候选编号。`kimi_step.py` 将编号映射回现有 `Candidate`，再通过 `CampaignRuntime.submit_stage()` 调用 `run_one_candidate()`；真实串口、DCA 和 BERT 由一个惰性 `LIVE_TUNING` 工厂装配。仓库当前没有真实安全范围，因此代码完成后仍必须先在 `vcsel_lab_live.yaml` 填入至少一个参数的 `minimum`、`maximum` 和 `max_step`，否则保持拒绝执行。

**Tech Stack:** Python 3.12、OpenAI-compatible SDK、Pillow、Pydantic、现有 V2 contracts、现有 CampaignOrchestrator、现有 VD1124/FlexDCA/VIT BERT adapters。

---

## 文件结构

- Create: `src/vcsel_ai_tuning/v2/providers/kimi_client.py` — Moonshot 请求和严格 JSON Schema。
- Create: `src/vcsel_ai_tuning/v2/providers/kimi_provider.py` — 眼图预处理、PNG 编码和 V2 contract 映射。
- Create: `src/vcsel_ai_tuning/hardware/live_runtime_factory.py` — 真实 `AdapterBundle` 与 `RuntimeTuningProfile`。
- Create: `src/vcsel_ai_tuning/v2/kimi_step.py` — 候选映射和受资源锁保护的单步执行。
- Modify: `pyproject.toml` — 增加 `kimi` 可选依赖。
- Test: `tests_py/ai_tuning/v2/test_kimi_provider.py`
- Test: `tests_py/ai_tuning/hardware/test_live_runtime_factory.py`
- Test: `tests_py/ai_tuning/v2/test_kimi_step.py`

不新增 Dashboard 页面、Fake Provider、Fake 硬件、replay Runner 或新的审核体系。

### Task 1: 接通真实 Kimi 多模态 API

**Files:**
- Modify: `pyproject.toml`
- Create: `src/vcsel_ai_tuning/v2/providers/kimi_client.py`
- Create: `src/vcsel_ai_tuning/v2/providers/kimi_provider.py`
- Test: `tests_py/ai_tuning/v2/test_kimi_provider.py`

- [ ] **Step 1: 写最小失败测试**

测试只注入一个记录请求的 SDK 对象，不建设产品级 Fake Provider。它必须证明图片作为真正的 `image_url` 数组发送、启用严格 JSON Schema，并且返回的 `candidate_id` 被映射成本地候选的原始参数。

```python
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient
from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider


class RecordingSdk:
    def __init__(self, payload):
        self.request = None

        def create(**request):
            self.request = request
            message = SimpleNamespace(content=json.dumps(payload))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )


@pytest.fixture
def valid_eye_sample(tmp_path, eye_sample):
    relative = eye_sample.raw_eye_density_ref.relative_path
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    np.save(path, np.ones((16, 24), dtype=np.uint32))
    digest = sha256(path.read_bytes()).hexdigest()
    reference = eye_sample.raw_eye_density_ref.model_copy(
        update={"sha256": digest}
    )
    return eye_sample.model_copy(
        update={
            "raw_eye_density_ref": reference,
            "artifact_hashes": {relative: digest},
        }
    )


def test_kimi_provider_sends_png_and_maps_only_local_candidate(
    tmp_path, valid_eye_sample, brain_candidates
):
    sdk = RecordingSdk(
        {
            "quality_score": 0.73,
            "confidence": 0.86,
            "valid": True,
            "ood": False,
            "action": "PROPOSE",
            "candidate_id": "candidate-002",
            "reason_codes": ["IMPROVE_MIDDLE_EYE"],
        }
    )
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="优先改善三只眼的均衡程度",
    )

    perception, decision = provider.analyze_once(
        valid_eye_sample, (), brain_candidates
    )

    content = sdk.request["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sdk.request["response_format"]["json_schema"]["strict"] is True
    assert perception.measurement_id == valid_eye_sample.measurement_id
    assert decision.candidate_id == "candidate-002"
    assert decision.requested_params == brain_candidates[1].requested_params
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py -q
```

Expected: FAIL，因为 `KimiApiClient` 和 `KimiProvider` 尚不存在。

- [ ] **Step 3: 实现最小客户端和 Provider**

`pyproject.toml` 增加：

```toml
kimi = [
  "openai>=1.0",
  "Pillow>=11.0",
]
```

`kimi_client.py` 的固定边界：

```python
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k3"

class KimiApiClient:
    def __init__(self, *, api_key=None, model=DEFAULT_MODEL, sdk_client=None):
        key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not key:
            raise RuntimeError("MOONSHOT_API_KEY is not set")
        if sdk_client is None:
            from openai import OpenAI
            sdk_client = OpenAI(api_key=key, base_url=MOONSHOT_BASE_URL, timeout=60.0)
        self.model = model
        self._sdk = sdk_client

    def analyze(self, *, png_bytes, instruction, sample, safe_candidates):
        image = base64.b64encode(png_bytes).decode("ascii")
        candidate_json = json.dumps(
            [item.model_dump(mode="json") for item in safe_candidates],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        completion = self._sdk.chat.completions.create(
            model=self.model,
            reasoning_effort="low",
            max_completion_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你只负责评价 PAM4 眼图并从给定候选中选一个 candidate_id。"
                        "禁止生成寄存器地址、I2C 命令或候选表之外的参数。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"初始指示：{instruction}\n"
                                f"measurement_id：{sample.measurement_id}\n"
                                f"候选：{candidate_json}"
                            ),
                        },
                    ],
                },
            ],
            response_format=KIMI_RESPONSE_FORMAT,
        )
        return json.loads(completion.choices[0].message.content or "{}")
```

`KIMI_RESPONSE_FORMAT` 必须要求这些字段，禁止额外字段：`quality_score`、`confidence`、`valid`、`ood`、`action`、`candidate_id`、`reason_codes`。`candidate_id` 使用 `string|null`，因此 `HOLD` 可以返回 `null`。

```python
KIMI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "vcsel_eye_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "valid": {"type": "boolean"},
                "ood": {"type": "boolean"},
                "action": {"type": "string", "enum": ["PROPOSE", "HOLD", "STOP"]},
                "candidate_id": {"type": ["string", "null"]},
                "reason_codes": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
            },
            "required": [
                "quality_score",
                "confidence",
                "valid",
                "ood",
                "action",
                "candidate_id",
                "reason_codes",
            ],
            "additionalProperties": False,
        },
    },
}
```

`kimi_provider.py` 实现：

```python
class KimiProvider:
    provider_id = "moonshot-kimi"

    def __init__(self, client, *, artifact_root, instruction):
        if not instruction.strip():
            raise ValueError("instruction must be non-empty")
        self._client = client
        self._artifact_root = Path(artifact_root)
        self._instruction = instruction.strip()
        self.provider_version = client.model

    def analyze_once(self, sample, history, safe_candidates):
        del history
        rgb = preprocess_eye_density(sample, self._artifact_root)
        image = Image.fromarray((rgb * 255.0).round().astype("uint8"), "RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        png = buffer.getvalue()
        payload = self._client.analyze(
            png_bytes=png,
            instruction=self._instruction,
            sample=sample,
            safe_candidates=safe_candidates,
        )
        action = DecisionAction(payload["action"])
        selected = next(
            (
                item for item in safe_candidates
                if item.candidate_id == payload["candidate_id"]
            ),
            None,
        )
        if action is DecisionAction.PROPOSE and selected is None:
            raise ValueError("Kimi selected a candidate outside the supplied menu")
        perception = PerceptionResult(
            measurement_id=sample.measurement_id,
            quality_score=payload["quality_score"],
            confidence=payload["confidence"],
            valid=payload["valid"],
            ood=payload["ood"],
            defects={},
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            input_hash=sha256(png).hexdigest(),
        )
        decision = DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=action,
            candidate_id=None if selected is None else selected.candidate_id,
            requested_params=None if selected is None else selected.requested_params,
            reason_codes=tuple(payload["reason_codes"]),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )
        return perception, decision
```

- [ ] **Step 4: 运行 Kimi Provider 测试和现有 V2 测试**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py tests_py/ai_tuning/v2 -q
```

Expected: PASS。测试期间不访问网络、不接触硬件。

- [ ] **Step 5: 提交 Task 1**

```powershell
git add pyproject.toml src/vcsel_ai_tuning/v2/providers/kimi_client.py src/vcsel_ai_tuning/v2/providers/kimi_provider.py tests_py/ai_tuning/v2/test_kimi_provider.py
git commit -m "feat: add Kimi multimodal provider"
```

### Task 2: 装配真实 LIVE_TUNING 运行时

**Files:**
- Create: `src/vcsel_ai_tuning/hardware/live_runtime_factory.py`
- Test: `tests_py/ai_tuning/hardware/test_live_runtime_factory.py`

- [ ] **Step 1: 写安全范围和资源所有权测试**

```python
from pathlib import Path

import pytest
import yaml

from vcsel_ai_tuning.campaign.models import load_b0
from vcsel_ai_tuning.hardware.live_runtime_factory import (
    build_live_tuning_profile,
)
from vcsel_ai_tuning.runtime import RuntimeAssemblyError


B0_PATH = Path("configs/ai_tuning/b0.yaml")
LIVE_PROFILE = Path("configs/hardware/vcsel_lab_live.yaml")


def write_live_profile(tmp_path, *, safety_limits):
    document = yaml.safe_load(LIVE_PROFILE.read_text(encoding="utf-8"))
    document["characterization"]["safety_limits"] = safety_limits
    path = tmp_path / "live-profile.yaml"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_live_profile_requires_at_least_one_explicit_safety_limit(tmp_path):
    profile = write_live_profile(tmp_path, safety_limits={})
    with pytest.raises(RuntimeAssemblyError, match="safety limit"):
        build_live_tuning_profile(profile, load_b0(B0_PATH))


def test_live_profile_builds_campaign_limits_from_operator_values(tmp_path):
    profile = write_live_profile(
        tmp_path,
        safety_limits={"eq": {"minimum": 18, "maximum": 25, "max_step": 3}},
    )
    tuning = build_live_tuning_profile(profile, load_b0(B0_PATH))
    spec = tuning.parameter_specs[0]
    assert (spec.name, spec.tuning_min, spec.tuning_max, spec.max_step) == (
        "eq", 18, 25, 3
    )


```

真实资源关闭不建立 Fake 硬件测试；由 `ExitStack` 的标准上下文语义和 Task 3 的真实单步运行共同验证。

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
python -m pytest tests_py/ai_tuning/hardware/test_live_runtime_factory.py -q
```

Expected: FAIL，因为工厂尚不存在。

- [ ] **Step 3: 实现惰性真实硬件工厂**

`live_runtime_factory.py` 公开三个具体函数：`build_live_tuning_profile(profile_path, baseline)` 返回 `RuntimeTuningProfile`；`build_live_adapters_from_config(config, baseline, runtime_timeout_s)` 返回 `AdapterBundle`；`create_live_api_runtime()` 返回 `ApiRuntime`。三个函数都必须在本任务完整实现，不保留空函数或延后实现分支。

`build_live_tuning_profile()` 必须只使用 `characterization.safety_limits` 中明确列出的参数。每个参数同时生成 Campaign `ParameterSpec` 和优化空间 `ParameterSpec`；配置为空时直接抛出 `RuntimeAssemblyError`。`driver_profile_hash` 使用配置文件原始字节的 SHA-256，不猜默认安全范围。

`build_live_adapters_from_config()` 使用一个 `ExitStack` 持有：

```python
session = stack.enter_context(
    open_visa_session(config.dca.resource, config.dca.timeout_ms)
)
writer = stack.enter_context(
    Vd1124RegisterWriter(
        port=config.i2c.port,
        baudrate=config.i2c.baudrate,
        address=config.i2c.device_address,
    )
)
bert_client = VitBertClient(
    config.bert.ip,
    config.bert.udp_port,
    local_ip=config.bert.local_ip,
    timeout_s=config.bert.timeout_s,
)
```

同一个 COM 口不能同时打开 `Vd1124Probe` 和 `Vd1124RegisterWriter`。因此使用只读 `_WriterIdentityProbe`，通过已经打开的 writer 读取 chip ID/revision并构造 `Vd1124ProbeResult`。随后复用现有：

```python
dca = CampaignDcaAdapter(session=session, config=config)
i2c = CampaignI2cAdapter(
    writer=writer,
    identity_probe=_WriterIdentityProbe(writer, config.i2c),
    config=config,
)
bert = CampaignBertAdapter(
    client=bert_client,
    port=config.bert.port,
    channel=config.bert.channel,
    fixed_pattern=baseline.pattern,
    upper_nlc=baseline.fir.upper_nlc,
    lower_nlc=baseline.fir.lower_nlc,
    expected_host=config.bert.ip,
    expected_udp_port=config.bert.udp_port,
)
```

返回的 `AdapterBundle` 中，DCA 外层包装器负责在 `close()` 时关闭 `ExitStack`。若实时配置的测量超时大于 runtime 超时，构造立即失败并关闭资源。

`create_live_api_runtime()` 从 `VCSEL_AI_RUNTIME_CONFIG` 加载 `RuntimeConfig`，要求模式为 `LIVE_TUNING`，加载 `hardware.driver_profile` 指向的真实配置，并核对 runtime 中的 DCA、I2C 和 BERT 端点与 driver profile 完全一致。随后将闭包传给现有 `create_runtime()`：

```python
return create_runtime(
    runtime_config,
    hardware_adapter_factory=lambda _campaign_id, baseline, timeout: (
        build_live_adapters_from_config(live_config, baseline, timeout)
    ),
    tuning_profile_factory=build_live_tuning_profile,
)
```

- [ ] **Step 4: 运行硬件工厂测试和现有硬件测试**

Run:

```powershell
python -m pytest tests_py/ai_tuning/hardware/test_live_runtime_factory.py tests_py/ai_tuning/hardware/test_characterization.py tests_py/ai_tuning/hardware/test_vd1124.py -q
```

Expected: PASS。该步骤不打开 COM7、VISA 或 UDP。

- [ ] **Step 5: 提交 Task 2**

```powershell
git add src/vcsel_ai_tuning/hardware/live_runtime_factory.py tests_py/ai_tuning/hardware/test_live_runtime_factory.py
git commit -m "feat: wire live campaign hardware adapters"
```

### Task 3: Kimi 单步建议接入真实安全执行链

**Files:**
- Create: `src/vcsel_ai_tuning/v2/kimi_step.py`
- Test: `tests_py/ai_tuning/v2/test_kimi_step.py`

- [ ] **Step 1: 写候选映射和执行测试**

```python
from vcsel_ai_tuning.campaign.models import Candidate, CampaignState
from vcsel_ai_tuning.v2.contracts import (
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    PerceptionResult,
)
from vcsel_ai_tuning.v2.kimi_step import execute_kimi_step


def brain_result(eye_sample, *, action, candidate=None):
    perception = PerceptionResult(
        measurement_id=eye_sample.measurement_id,
        quality_score=0.75,
        confidence=0.90,
        valid=True,
        ood=False,
        defects={},
        provider_id="test-provider",
        provider_version="1",
        input_hash="a" * 64,
    )
    decision = DecisionProposal(
        based_on_measurement_id=eye_sample.measurement_id,
        action=action,
        candidate_id=None if candidate is None else candidate.candidate_id,
        requested_params=(
            None if candidate is None else dict(candidate.requested_i2c)
        ),
        reason_codes=("TEST",),
        provider_id="test-provider",
        provider_version="1",
    )
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=1.0,
        request_hash="b" * 64,
        response_hash="c" * 64,
    )


class StaticBrain:
    def __init__(self, result):
        self.result = result

    def analyze(self, sample, history, safe_candidates):
        return self.result


class RecordingOrchestrator:
    def __init__(self):
        self.calls = 0
        self.candidate = None

    def run_one_candidate(self, authorization, candidate):
        self.calls += 1
        self.candidate = candidate
        return ()


def two_candidates():
    return (
        Candidate(
            candidate_id="candidate-001",
            stage=CampaignState.I2C_COARSE,
            requested_i2c={"eq": 21},
            model_version="kimi-k3",
            reason="candidate menu",
        ),
        Candidate(
            candidate_id="candidate-002",
            stage=CampaignState.I2C_COARSE,
            requested_i2c={"eq": 22},
            model_version="kimi-k3",
            reason="candidate menu",
        ),
    )


def test_kimi_step_executes_only_the_selected_campaign_candidate(
    tmp_path, eye_sample
):
    campaign_candidates = two_candidates()
    authorization = object()
    brain = StaticBrain(
        brain_result(
            eye_sample,
            action=DecisionAction.PROPOSE,
            candidate=campaign_candidates[1],
        )
    )
    orchestrator = RecordingOrchestrator()

    result = execute_kimi_step(
        brain=brain,
        orchestrator=orchestrator,
        authorization=authorization,
        sample=eye_sample,
        candidates=campaign_candidates,
        instruction="优先改善三只眼的均衡程度",
        output_path=tmp_path / "v21_result.json",
    )

    assert orchestrator.candidate is campaign_candidates[1]
    assert result["execution_status"] == "EXECUTED"


def test_kimi_hold_never_calls_orchestrator(
    tmp_path, eye_sample
):
    campaign_candidates = two_candidates()
    authorization = object()
    brain = StaticBrain(
        brain_result(eye_sample, action=DecisionAction.HOLD)
    )
    orchestrator = RecordingOrchestrator()
    result = execute_kimi_step(
        brain=brain,
        orchestrator=orchestrator,
        authorization=authorization,
        sample=eye_sample,
        candidates=campaign_candidates,
        instruction="优先改善三只眼的均衡程度",
        output_path=tmp_path / "v21_result.json",
    )
    assert orchestrator.calls == 0
    assert result["execution_status"] == "HOLD"
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_step.py -q
```

Expected: FAIL，因为 `execute_kimi_step()` 尚不存在。

- [ ] **Step 3: 实现单步桥接**

```python
def _write_result(
    output_path,
    brain_result,
    execution_status,
    observations,
    *,
    instruction,
    error=None,
):
    document = {
        "measurement_id": brain_result.perception.measurement_id,
        "instruction": instruction,
        "brain_mode": brain_result.brain_mode.value,
        "model": brain_result.decision.provider_version,
        "quality_score": brain_result.perception.quality_score,
        "confidence": brain_result.perception.confidence,
        "action": brain_result.decision.action.value,
        "candidate_id": brain_result.decision.candidate_id,
        "requested_params": brain_result.decision.requested_params,
        "reason_codes": list(brain_result.decision.reason_codes),
        "api_latency_ms": brain_result.latency_ms,
        "execution_status": execution_status,
        "observation_ids": [item.observation_id for item in observations],
        "error": error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return document


def execute_kimi_step(
    *,
    brain: OptimizationBrain,
    orchestrator: CampaignOrchestrator,
    authorization: StageAuthorization,
    sample: EyeSample,
    candidates: tuple[Candidate, ...],
    instruction: str,
    output_path: Path,
) -> dict[str, object]:
    safe_candidates = tuple(
        BrainCandidate(
            candidate_id=item.candidate_id,
            requested_params=dict(item.requested_i2c),
        )
        for item in candidates
    )
    brain_result = brain.analyze(sample, (), safe_candidates)
    selected = next(
        (
            item for item in candidates
            if item.candidate_id == brain_result.decision.candidate_id
        ),
        None,
    )
    if brain_result.decision.action is not DecisionAction.PROPOSE:
        return _write_result(
            output_path,
            brain_result,
            "HOLD",
            (),
            instruction=instruction,
        )
    if selected is None:
        return _write_result(
            output_path,
            brain_result,
            "HOLD",
            (),
            instruction=instruction,
        )
    try:
        observations = orchestrator.run_one_candidate(authorization, selected)
    except Exception as error:
        _write_result(
            output_path,
            brain_result,
            "FAILED",
            (),
            instruction=instruction,
            error=str(error),
        )
        raise
    return _write_result(
        output_path,
        brain_result,
        "EXECUTED",
        observations,
        instruction=instruction,
    )
```

同时提供：

```python
def submit_kimi_step(
    runtime: CampaignRuntime,
    *,
    campaign_id: str,
    brain: OptimizationBrain,
    authorization: StageAuthorization,
    sample: EyeSample,
    candidates: tuple[Candidate, ...],
    instruction: str,
    output_path: Path,
) -> Future[dict[str, object]]:
    return runtime.submit_stage(
        campaign_id,
        f"KIMI_STEP:{sample.measurement_id}",
        lambda entry: execute_kimi_step(
            brain=brain,
            orchestrator=entry.orchestrator,
            authorization=authorization,
            sample=sample,
            candidates=candidates,
            instruction=instruction,
            output_path=output_path,
        ),
    )
```

这样真实执行始终经过 V2.0 的全局硬件租约、资源锁、`validate_candidate()`、写后回读、DCA 重测、回滚和 `RECOVERY_REQUIRED`。Kimi 没有任何硬件对象引用。

- [ ] **Step 4: 运行全部相关测试**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2 tests_py/ai_tuning/campaign tests_py/ai_tuning/hardware/test_live_runtime_factory.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交 Task 3**

```powershell
git add src/vcsel_ai_tuning/v2/kimi_step.py tests_py/ai_tuning/v2/test_kimi_step.py
git commit -m "feat: execute Kimi candidate through V2 safety chain"
```

## 真实运行顺序

代码完成后只做以下操作，不建设 Fake 流程：

1. 在 `configs/hardware/vcsel_lab_live.yaml` 的 `safety_limits` 填入第一批真实范围；第一轮只启用一个 I2C 参数。
2. 准备 `LIVE_TUNING` runtime YAML，将 `hardware.driver_profile` 指向该文件，并把 `measurement_timeout_s` 设为不小于 DCA 的 180 秒。
3. PowerShell 设置 `MOONSHOT_API_KEY`。
4. 使用已有 V2.0 流程创建 Campaign、完成 preflight、进入 I2C stage 并得到 `StageAuthorization`。
5. 创建 `KimiProvider` 和 `MultimodalApiBrain`，将当前 Campaign ID、授权、样本和候选传给 `submit_kimi_step`，并等待返回结果。
6. 检查 `v21_result.json`、硬件回读和 DCA 新测量；若失败，遵从 V2.0 的 FAILED 或 `RECOVERY_REQUIRED` 状态。

真实写入前唯一仍需用户提供的数据是：首个参数名、`minimum`、`maximum`、`max_step`。这些值不能从寄存器位宽或测试示例推断。
