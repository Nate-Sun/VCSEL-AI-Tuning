# V2.1 Kimi Multimodal Integration Implementation Plan

> **已废止：** 本计划对应已取消的复杂方案，请勿执行。后续以精简设计和新的短实施计划为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 V2.0 `PURE_API`底座上接入 Kimi K3，使其读取 baseline/current 眼图和 Campaign 初始指示，从本地安全候选中选择一步 I2C 调整，并支持 replay、影子模式和注入式单步执行。

**Architecture:** 保持现有 `OptimizationBrain`、`MultimodalApiBrain`和硬件 Orchestrator 边界不变。新增 Campaign 指示契约、安全候选生成器、确定性 PNG 编码、Kimi API 客户端、`FusedBrainProvider`实现、影子 CLI 和单步 Runner；Kimi 永远只返回 `candidate_id`，本地程序负责复制真实参数并执行现有安全链路。

**Tech Stack:** Python 3.12、Pydantic 2、NumPy/SciPy、Pillow、OpenAI-compatible Python SDK、PyYAML、pytest

---

## Scope and file map

本计划依次完成网络前的纯本地组件、网络 Provider、影子运行入口和单步执行桥。实时 DCA 到 `EyeSample`的生产仍由 V1.1 数据采集链负责；本计划不伪造该输入，也不把 Dashboard 变成硬件控制入口。

**Create:**

- `src/vcsel_ai_tuning/v2/candidates.py`：从当前回读和 `StageAuthorization`生成一步安全候选菜单。
- `src/vcsel_ai_tuning/v2/providers/kimi_client.py`：Moonshot网络客户端和非敏感调用结果。
- `src/vcsel_ai_tuning/v2/providers/kimi_multimodal.py`：Kimi多模态Payload、JSON Schema解析、本地契约映射和调用审计。
- `src/vcsel_ai_tuning/v2/runner.py`：Brain公共单轮运行、影子模式和注入式单步执行。
- `src/vcsel_ai_tuning/v2/kimi_cli.py`：无硬件的Kimi影子模式命令行入口。
- `tests_py/ai_tuning/v2/test_candidates.py`
- `tests_py/ai_tuning/v2/test_kimi_client.py`
- `tests_py/ai_tuning/v2/test_kimi_multimodal.py`
- `tests_py/ai_tuning/v2/test_runner.py`
- `tests_py/ai_tuning/v2/test_kimi_cli.py`
- `configs/ai_tuning/v2.kimi.example.yaml`
- `docs/v2.1-kimi-quickstart.md`

**Modify:**

- `src/vcsel_ai_tuning/v2/contracts.py`：增加 `OptimizationDirective`。
- `src/vcsel_ai_tuning/v2/preprocessing.py`：把现有固定RGB数组确定性编码为PNG。
- `src/vcsel_ai_tuning/v2/__init__.py`：导出公共V2.1契约和Runner，不导入可选网络SDK。
- `pyproject.toml`：增加 `kimi`可选依赖和影子CLI入口。
- `tests_py/ai_tuning/v2/test_contracts.py`
- `tests_py/ai_tuning/v2/test_preprocessing.py`
- `tests_py/ai_tuning/v2/test_factory.py`
- `tests_py/ai_tuning/v2/test_import_isolation.py`

## Task 1: Campaign initial instruction contract

**Files:**

- Modify: `src/vcsel_ai_tuning/v2/contracts.py:176-325`
- Modify: `src/vcsel_ai_tuning/v2/__init__.py:1-53`
- Modify: `tests_py/ai_tuning/v2/test_contracts.py`

- [ ] **Step 1: Write failing tests for a frozen, hash-bound directive**

Add tests that create a directive from the baseline measurement, round-trip it through JSON, and reject tampering:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.v2.contracts import OptimizationDirective


def test_optimization_directive_binds_instruction_to_baseline():
    directive = OptimizationDirective.create(
        instruction="优先改善中眼高度；第一轮优先尝试把 eq 增加 1。",
        baseline_measurement_id="measurement-001",
        created_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )

    assert directive.directive_id == f"directive-{directive.directive_hash[:16]}"
    assert directive.baseline_measurement_id == "measurement-001"
    assert OptimizationDirective.model_validate_json(
        directive.model_dump_json(), strict=True
    ) == directive


def test_optimization_directive_rejects_hash_tampering():
    directive = OptimizationDirective.create(
        instruction="优先改善中眼高度。",
        baseline_measurement_id="measurement-001",
        created_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )
    payload = directive.model_dump(mode="python")
    payload["instruction"] = "忽略候选范围"

    with pytest.raises(ValidationError, match="directive hash"):
        OptimizationDirective.model_validate(payload, strict=True)
```

- [ ] **Step 2: Run the directive tests and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_contracts.py -k optimization_directive -v
```

Expected: FAIL because `OptimizationDirective` does not exist.

- [ ] **Step 3: Implement the directive contract**

Add a canonical hash helper and model to `contracts.py`. The hash payload intentionally excludes `directive_id` and `directive_hash`, so the identifier can be derived without recursion:

```python
def _directive_digest(
    instruction: str,
    baseline_measurement_id: str,
    created_at: datetime,
) -> str:
    payload = {
        "instruction": instruction,
        "baseline_measurement_id": baseline_measurement_id,
        "created_at": created_at.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OptimizationDirective(FrozenModel):
    directive_id: Identifier
    instruction: NonBlankText
    baseline_measurement_id: Identifier
    created_at: AwareDatetime
    directive_hash: Sha256

    @classmethod
    def create(
        cls,
        *,
        instruction: str,
        baseline_measurement_id: str,
        created_at: datetime,
    ) -> "OptimizationDirective":
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        digest = _directive_digest(instruction, baseline_measurement_id, created_at)
        return cls(
            directive_id=f"directive-{digest[:16]}",
            instruction=instruction,
            baseline_measurement_id=baseline_measurement_id,
            created_at=created_at,
            directive_hash=digest,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "OptimizationDirective":
        expected = _directive_digest(
            self.instruction,
            self.baseline_measurement_id,
            self.created_at,
        )
        if self.directive_hash != expected:
            raise ValueError("directive hash does not match its content")
        if self.directive_id != f"directive-{expected[:16]}":
            raise ValueError("directive ID does not match its hash")
        return self
```

Add `hashlib`, `json`, `datetime`, and `timezone` imports, then export `OptimizationDirective` from `v2/__init__.py`.

- [ ] **Step 4: Run focused and full contract tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_contracts.py -v
```

Expected: all contract tests PASS.

- [ ] **Step 5: Commit the directive contract**

```powershell
git add src/vcsel_ai_tuning/v2/contracts.py src/vcsel_ai_tuning/v2/__init__.py tests_py/ai_tuning/v2/test_contracts.py
git commit -m "feat: add V2 campaign optimization directive"
```

## Task 2: Deterministic one-parameter I2C candidate menu

**Files:**

- Create: `src/vcsel_ai_tuning/v2/candidates.py`
- Create: `tests_py/ai_tuning/v2/test_candidates.py`
- Modify: `src/vcsel_ai_tuning/v2/__init__.py`

- [ ] **Step 1: Write failing tests for ±1 through ±max_step choices**

Use a real `StageAuthorization` with `max_step=3` and a current readback of 10:

```python
from datetime import datetime, timedelta, timezone

from vcsel_ai_tuning.campaign.models import (
    AppliedSettings,
    AuthorizationMode,
    CampaignState,
    ParameterSpec,
    StageAuthorization,
)
from vcsel_ai_tuning.v2.candidates import generate_i2c_brain_candidates


def _authorization() -> StageAuthorization:
    now = datetime(2026, 8, 9, tzinfo=timezone.utc)
    return StageAuthorization(
        authorization_id="authorization-001",
        campaign_id="campaign-001",
        mode=AuthorizationMode.PER_STAGE,
        stage=CampaignState.I2C_COARSE,
        baseline_hash="a" * 64,
        rollback_hash="b" * 64,
        parameter_specs=(
            ParameterSpec(
                name="eq",
                encoding_min=0,
                encoding_max=31,
                tuning_min=7,
                tuning_max=13,
                max_step=3,
            ),
        ),
        max_candidates=8,
        max_duration_s=60.0,
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        rollback_snapshot=AppliedSettings(readback_i2c={"eq": 10}),
    )


def test_max_step_three_offers_every_smaller_step():
    candidates = generate_i2c_brain_candidates(_authorization(), {"eq": 10})

    assert [candidate.requested_params for candidate in candidates] == [
        {"eq": 9},
        {"eq": 11},
        {"eq": 8},
        {"eq": 12},
        {"eq": 7},
        {"eq": 13},
    ]


def test_candidate_generation_stops_at_tuning_bounds():
    candidates = generate_i2c_brain_candidates(_authorization(), {"eq": 7})

    assert [candidate.requested_params for candidate in candidates] == [
        {"eq": 8},
        {"eq": 9},
        {"eq": 10},
    ]
```

Add these explicit edge tests:

```python
def test_disabled_parameter_produces_no_candidates():
    authorization = _authorization()
    disabled = authorization.parameter_specs[0].model_copy(
        update={"enabled": False}
    )
    authorization = authorization.model_copy(
        update={"parameter_specs": (disabled,)}
    )

    assert generate_i2c_brain_candidates(authorization, {}) == ()


@pytest.mark.parametrize("readback", [{}, {"eq": 10.5}, {"eq": True}])
def test_invalid_readback_is_rejected(readback):
    with pytest.raises(ValueError, match="readback for eq"):
        generate_i2c_brain_candidates(_authorization(), readback)


def test_duplicate_enabled_parameter_names_are_rejected():
    authorization = _authorization().model_copy(
        update={"parameter_specs": _authorization().parameter_specs * 2}
    )

    with pytest.raises(ValueError, match="duplicate parameter spec"):
        generate_i2c_brain_candidates(authorization, {"eq": 10})


def test_non_i2c_stage_is_rejected():
    authorization = _authorization().model_copy(
        update={"stage": CampaignState.FIR_APPLY}
    )

    with pytest.raises(ValueError, match="I2C stage"):
        generate_i2c_brain_candidates(authorization, {"eq": 10})
```

- [ ] **Step 2: Run candidate tests and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_candidates.py -v
```

Expected: FAIL because `v2.candidates` does not exist.

- [ ] **Step 3: Implement deterministic candidate generation**

Create `candidates.py` with one changed parameter per candidate and unit register steps from 1 through `max_step`:

```python
from __future__ import annotations

from collections.abc import Mapping

from vcsel_ai_tuning.campaign.models import CampaignState, StageAuthorization

from .contracts import BrainCandidate


_I2C_STAGES = frozenset(
    {
        CampaignState.I2C_SCREENING,
        CampaignState.I2C_COARSE,
        CampaignState.I2C_FINE,
        CampaignState.I2C_RETUNE,
    }
)


def generate_i2c_brain_candidates(
    authorization: StageAuthorization,
    readback: Mapping[str, int | float],
) -> tuple[BrainCandidate, ...]:
    if not isinstance(authorization, StageAuthorization):
        raise TypeError("authorization must be a StageAuthorization")
    authorization = StageAuthorization.model_validate(
        authorization.model_dump(round_trip=True), strict=True
    )
    if authorization.stage not in _I2C_STAGES:
        raise ValueError("authorization stage must be an I2C stage")

    enabled_names = [spec.name for spec in authorization.parameter_specs if spec.enabled]
    if len(enabled_names) != len(set(enabled_names)):
        raise ValueError("duplicate parameter spec name")

    values: dict[str, int] = {}
    for spec in authorization.parameter_specs:
        if not spec.enabled:
            continue
        value = readback.get(spec.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"readback for {spec.name} must be an integer code")
        integer = int(value)
        if float(value) != float(integer):
            raise ValueError(f"readback for {spec.name} must be an integer code")
        values[spec.name] = integer

    candidates: list[BrainCandidate] = []
    for spec in sorted(authorization.parameter_specs, key=lambda item: item.name):
        if not spec.enabled:
            continue
        current = values[spec.name]
        for delta in range(1, spec.max_step + 1):
            for direction, target in (("down", current - delta), ("up", current + delta)):
                if not spec.tuning_min <= target <= spec.tuning_max:
                    continue
                candidates.append(
                    BrainCandidate(
                        candidate_id=f"i2c-{spec.name}-{direction}-{delta}-{target}",
                        requested_params={spec.name: target},
                    )
                )
    return tuple(candidates)
```

Export `generate_i2c_brain_candidates` from `v2/__init__.py`.

- [ ] **Step 4: Run candidate and authorization tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_candidates.py tests_py/ai_tuning/campaign/test_authorization.py -v
```

Expected: all selected tests PASS; existing authorization behavior is unchanged.

- [ ] **Step 5: Commit candidate generation**

```powershell
git add src/vcsel_ai_tuning/v2/candidates.py src/vcsel_ai_tuning/v2/__init__.py tests_py/ai_tuning/v2/test_candidates.py
git commit -m "feat: generate bounded V2 I2C candidates"
```

## Task 3: Deterministic PNG bytes for Kimi vision input

**Files:**

- Modify: `src/vcsel_ai_tuning/v2/preprocessing.py:1-210`
- Modify: `tests_py/ai_tuning/v2/test_preprocessing.py`
- Modify: `tests_py/ai_tuning/v2/test_import_isolation.py`
- Modify: `pyproject.toml:16-36`

- [ ] **Step 1: Add the optional Kimi dependencies**

Add a separate optional group so importing V2 foundations still requires no network or image SDK:

```toml
kimi = [
  "openai>=1.0,<3",
  "Pillow>=11,<13",
]
```

Install the development and Kimi extras:

```powershell
python -m pip install -e ".[dev,kimi]"
```

Expected: installation succeeds without changing the default dependency set.

Add `PIL` to the forbidden module prefixes in `test_import_isolation.py`; optional image encoding may load Pillow only when `encode_eye_density_png()` is called.

- [ ] **Step 2: Write failing deterministic PNG tests**

Extend `test_preprocessing.py`:

```python
from vcsel_ai_tuning.v2.preprocessing import encode_eye_density_png


def test_encode_eye_density_png_is_deterministic(tmp_path: Path) -> None:
    sample = _sample_for_array(
        tmp_path,
        np.arange(35, dtype=np.uint16).reshape(5, 7),
    )

    first = encode_eye_density_png(sample, tmp_path)
    second = encode_eye_density_png(sample, tmp_path)

    assert first.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert first.media_type == "image/png"
    assert first.width == 392
    assert first.height == 280
    assert first.sha256 == hashlib.sha256(first.data).hexdigest()
    assert first == second
```

- [ ] **Step 3: Run the PNG test and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_preprocessing.py -k encode_eye_density_png -v
```

Expected: FAIL because `encode_eye_density_png` does not exist.

- [ ] **Step 4: Implement lazy Pillow encoding**

Add this immutable result and encoder. Import Pillow inside the function so `import vcsel_ai_tuning.v2` remains dependency-isolated:

```python
@dataclass(frozen=True, slots=True)
class EncodedEyePng:
    data: bytes
    sha256: str
    media_type: str
    width: int
    height: int


def encode_eye_density_png(
    sample: EyeSample,
    artifact_root: str | Path,
    config: EyeDensityPreprocessConfig = EyeDensityPreprocessConfig(),
) -> EncodedEyePng:
    try:
        from PIL import Image
    except ImportError:
        raise EyeDensityError("Pillow is required for PNG encoding") from None

    rgb = preprocess_eye_density(sample, artifact_root, config)
    pixels = np.rint(rgb * 255.0).astype(np.uint8, copy=False)
    stream = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(
        stream,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    payload = stream.getvalue()
    return EncodedEyePng(
        data=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        media_type="image/png",
        width=config.canvas_width,
        height=config.canvas_height,
    )
```

- [ ] **Step 5: Verify preprocessing and import isolation**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_preprocessing.py tests_py/ai_tuning/v2/test_import_isolation.py -v
```

Expected: all selected tests PASS; no test imports `PIL` or `openai` merely by importing V2 foundations.

- [ ] **Step 6: Commit PNG support and optional dependencies**

```powershell
git add pyproject.toml src/vcsel_ai_tuning/v2/preprocessing.py tests_py/ai_tuning/v2/test_preprocessing.py
git commit -m "feat: encode deterministic Kimi eye PNG"
```

## Task 4: Kimi API client with dependency injection and sanitized errors

**Files:**

- Create: `src/vcsel_ai_tuning/v2/providers/kimi_client.py`
- Create: `tests_py/ai_tuning/v2/test_kimi_client.py`

- [ ] **Step 1: Write failing client tests using a fake completion callable**

The tests must never contact the network:

```python
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.v2.providers.kimi_client import (
    KimiApiError,
    KimiClient,
    KimiClientConfig,
)


def test_client_sends_bounded_k3_request():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="kimi-k3",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"HOLD"}'))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

    client = KimiClient(KimiClientConfig(), create)
    result = client.complete(
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "json_object"},
    )

    assert calls[0]["model"] == "kimi-k3"
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_completion_tokens"] == 1024
    assert calls[0]["stream"] is False
    assert result.content == '{"action":"HOLD"}'
    assert result.total_tokens == 120


def test_client_hides_provider_exception_details():
    def create(**kwargs):
        raise RuntimeError("Authorization: Bearer secret-value")

    client = KimiClient(KimiClientConfig(), create)
    with pytest.raises(KimiApiError, match="Kimi API request failed") as error:
        client.complete(messages=[], response_format={"type": "json_object"})

    assert "secret-value" not in str(error.value)
```

Add the remaining client boundary tests:

```python
def test_from_environment_requires_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(KimiApiError, match="MOONSHOT_API_KEY is not set"):
        KimiClient.from_environment(KimiClientConfig())


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(model="kimi-k3", choices=[], usage=None),
        SimpleNamespace(
            model="kimi-k3",
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        ),
    ],
)
def test_client_rejects_missing_completion_content(response):
    client = KimiClient(KimiClientConfig(), lambda **kwargs: response)

    with pytest.raises(KimiApiError, match="Kimi API request failed"):
        client.complete(messages=[], response_format={"type": "json_object"})


def test_client_allows_absent_usage_object():
    response = SimpleNamespace(
        model="kimi-k3",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"HOLD"}'))],
        usage=None,
    )
    client = KimiClient(KimiClientConfig(), lambda **kwargs: response)

    result = client.complete(messages=[], response_format={"type": "json_object"})

    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
```

- [ ] **Step 2: Run client tests and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_client.py -v
```

Expected: FAIL because `kimi_client.py` does not exist.

- [ ] **Step 3: Implement config and transport-neutral client**

Create strict models and a callable-injected client:

```python
from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Callable, Literal

from pydantic import Field, StrictInt

from ..contracts import FrozenModel, NonBlankText


class KimiApiError(RuntimeError):
    pass


class KimiClientConfig(FrozenModel):
    base_url: NonBlankText = "https://api.moonshot.cn/v1"
    model: NonBlankText = "kimi-k3"
    timeout_s: float = Field(default=60.0, gt=0.0, le=300.0)
    max_retries: StrictInt = Field(default=1, ge=0, le=3)
    reasoning_effort: Literal["low", "high", "max"] = "low"
    max_completion_tokens: StrictInt = Field(default=1024, ge=128, le=8192)


class KimiCompletion(FrozenModel):
    content: NonBlankText
    model: NonBlankText
    prompt_tokens: StrictInt | None = Field(default=None, ge=0)
    completion_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)


class KimiClient:
    def __init__(
        self,
        config: KimiClientConfig,
        completion_create: Callable[..., Any],
    ) -> None:
        self.config = KimiClientConfig.model_validate(
            config.model_dump(round_trip=True), strict=True
        )
        if not callable(completion_create):
            raise TypeError("completion_create must be callable")
        self._completion_create = completion_create

    @classmethod
    def from_environment(cls, config: KimiClientConfig) -> "KimiClient":
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key or not api_key.strip():
            raise KimiApiError("MOONSHOT_API_KEY is not set")
        try:
            from openai import OpenAI
            sdk = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout_s,
                max_retries=config.max_retries,
            )
        except Exception:
            raise KimiApiError("Kimi API client initialization failed") from None
        return cls(config, sdk.chat.completions.create)

    def complete(self, *, messages: list[dict[str, Any]], response_format: dict[str, Any]) -> KimiCompletion:
        started = perf_counter()
        try:
            response = self._completion_create(
                model=self.config.model,
                messages=messages,
                response_format=response_format,
                reasoning_effort=self.config.reasoning_effort,
                max_completion_tokens=self.config.max_completion_tokens,
                stream=False,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty completion")
            usage = getattr(response, "usage", None)
            return KimiCompletion(
                content=content,
                model=str(response.model),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                latency_ms=max(0.0, (perf_counter() - started) * 1000.0),
            )
        except KimiApiError:
            raise
        except Exception:
            raise KimiApiError("Kimi API request failed") from None
```

- [ ] **Step 4: Run client and import-isolation tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_client.py tests_py/ai_tuning/v2/test_import_isolation.py -v
```

Expected: all selected tests PASS and no real HTTP request occurs.

- [ ] **Step 5: Commit the Kimi client**

```powershell
git add src/vcsel_ai_tuning/v2/providers/kimi_client.py tests_py/ai_tuning/v2/test_kimi_client.py
git commit -m "feat: add injectable Kimi API client"
```

## Task 5: Kimi multimodal fused provider

**Files:**

- Create: `src/vcsel_ai_tuning/v2/providers/kimi_multimodal.py`
- Create: `tests_py/ai_tuning/v2/test_kimi_multimodal.py`

- [ ] **Step 1: Write failing provider tests with real NPY artifacts and fake API output**

Create a valid density file, baseline/current samples and safe candidates. Assert the first round contains one image, later rounds contain baseline then current, and a selected candidate receives the exact local parameter mapping:

```python
def test_provider_uses_directive_and_copies_local_candidate_params(
    tmp_path, baseline_sample, current_sample, directive, brain_candidates
):
    fake_client = FakeKimiClient(
        {
            "quality_score": 0.78,
            "confidence": 0.91,
            "valid": True,
            "ood": False,
            "defects": {"eye_closure": 0.22},
            "action": "PROPOSE",
            "candidate_id": brain_candidates[1].candidate_id,
            "reason_codes": ["IMPROVE_MIDDLE_EYE"],
        }
    )
    audits = []
    provider = KimiMultimodalProvider(
        client=fake_client,
        artifact_root=tmp_path,
        baseline=baseline_sample,
        directive=directive,
        audit_sink=audits.append,
    )

    perception, decision = provider.analyze_once(
        current_sample, (), brain_candidates
    )

    assert directive.instruction in fake_client.last_messages[-1]["content"][-1]["text"]
    assert decision.candidate_id == brain_candidates[1].candidate_id
    assert decision.requested_params == brain_candidates[1].requested_params
    assert perception.input_hash == current_sample.raw_eye_density_ref.sha256
    assert audits[0].directive_hash == directive.directive_hash
    assert "base64" not in audits[0].model_dump_json()
```

Add explicit provider boundary tests:

```python
@pytest.mark.parametrize(
    ("wire_action", "expected_action"),
    [("HOLD", DecisionAction.HOLD), ("STOP", DecisionAction.STOP)],
)
def test_provider_maps_non_proposal_actions(provider_factory, wire_action, expected_action):
    provider = provider_factory(
        response={
            "quality_score": 0.7,
            "confidence": 0.9,
            "valid": True,
            "ood": False,
            "defects": {},
            "action": wire_action,
            "candidate_id": None,
            "reason_codes": [wire_action],
        }
    )

    _, decision = provider.analyze_once(
        provider.current_sample, (), provider.safe_candidates
    )

    assert decision.action is expected_action
    assert decision.candidate_id is None


def test_provider_turns_unknown_candidate_into_hold(provider_factory):
    provider = provider_factory(candidate_id="candidate-not-supplied")

    _, decision = provider.analyze_once(
        provider.current_sample, (), provider.safe_candidates
    )

    assert decision.action is DecisionAction.HOLD
    assert decision.reason_codes == ("CANDIDATE_NOT_IN_INPUT_SET",)


def test_multimodal_brain_holds_on_malformed_kimi_json(provider_factory):
    provider = provider_factory(raw_content="not-json")
    brain = MultimodalApiBrain(provider, minimum_confidence=0.7)

    result = brain.analyze(
        provider.current_sample, (), provider.safe_candidates
    )

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)


def test_provider_rejects_cross_campaign_baseline(provider_factory):
    with pytest.raises(ValueError, match="Campaign mismatch"):
        provider_factory(baseline_campaign_id="other-campaign")


def test_provider_rejects_directive_for_another_baseline(provider_factory):
    with pytest.raises(ValueError, match="baseline measurement"):
        provider_factory(directive_baseline_id="other-measurement")


def test_provider_sends_only_latest_five_history_items(provider_factory, v2_history):
    provider = provider_factory()

    provider.analyze_once(
        provider.current_sample,
        v2_history[:7],
        provider.safe_candidates,
    )

    text = provider.client.last_messages[-1]["content"][-1]["text"]
    payload = json.loads(text)
    assert len(payload["recent_history"]) == 5
    assert payload["recent_history"][0]["measurement_id"] == v2_history[2].measurement_id


def test_audit_sink_failure_becomes_brain_hold(provider_factory):
    def fail_audit(record):
        raise OSError("audit unavailable")

    provider = provider_factory(audit_sink=fail_audit)
    brain = MultimodalApiBrain(provider, minimum_confidence=0.7)

    result = brain.analyze(
        provider.current_sample, (), provider.safe_candidates
    )

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
```

Define `provider_factory` and `v2_history` as local fixtures in this test file. The fixture creates real hashed NPY files and returns a provider test harness exposing `current_sample`, `safe_candidates` and its fake client; it never contacts the network.

- [ ] **Step 2: Run provider tests and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_multimodal.py -v
```

Expected: FAIL because `kimi_multimodal.py` does not exist.

- [ ] **Step 3: Implement the strict Kimi output model**

Use only semantic fields from Kimi:

```python
class KimiDecisionOutput(FrozenModel):
    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    defects: ImmutableProbabilityMapping = Field(default_factory=dict)
    action: Literal["PROPOSE", "HOLD", "STOP"]
    candidate_id: Identifier | None = None
    reason_codes: tuple[NonBlankText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> "KimiDecisionOutput":
        if self.action == "PROPOSE" and self.candidate_id is None:
            raise ValueError("PROPOSE requires candidate_id")
        if self.action != "PROPOSE" and self.candidate_id is not None:
            raise ValueError("non-PROPOSE action forbids candidate_id")
        return self
```

Define `KimiCallAudit` with `status` (`SUCCESS` or `ERROR`), baseline/current measurement IDs, directive ID/hash, input PNG hashes, request hash, optional raw structured response, optional sanitized error code, model, usage and latency. It must not contain API Key or Base64 image data:

```python
class KimiCallAudit(FrozenModel):
    status: Literal["SUCCESS", "ERROR"]
    baseline_measurement_id: Identifier
    current_measurement_id: Identifier
    directive_id: Identifier
    directive_hash: Sha256
    baseline_png_sha256: Sha256
    current_png_sha256: Sha256
    request_hash: Sha256
    model: NonBlankText
    response_json: str | None = None
    error_code: Literal["KIMI_API_ERROR", "INVALID_RESPONSE"] | None = None
    prompt_tokens: StrictInt | None = Field(default=None, ge=0)
    completion_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "KimiCallAudit":
        if self.status == "SUCCESS":
            if self.response_json is None or self.error_code is not None:
                raise ValueError("SUCCESS audit requires response and forbids error")
        elif self.response_json is not None or self.error_code is None:
            raise ValueError("ERROR audit requires error and forbids response")
        return self
```

Also add an atomic audit writer in the same module:

```python
def write_kimi_audit_jsonl(
    records: tuple[KimiCallAudit, ...],
    output: str | Path,
) -> Path:
    if not records:
        raise ValueError("Kimi audit output requires at least one record")
    lines = tuple(record.model_dump_json() + "\n" for record in records)
    destination = Path(output)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.writelines(lines)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        return destination
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
```

- [ ] **Step 4: Implement message construction and local decision mapping**

The provider must capture baseline and directive at construction and verify they belong to the same Campaign. Build image parts before the final text part, deduplicating the baseline image when it is also current:

```python
def _image_part(encoded: EncodedEyePng) -> dict[str, object]:
    payload = base64.b64encode(encoded.data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{payload}"},
    }


def _decision_from_output(
    output: KimiDecisionOutput,
    sample: EyeSample,
    candidates: tuple[BrainCandidate, ...],
    provider_id: str,
    provider_version: str,
) -> DecisionProposal:
    if output.action == "PROPOSE":
        selected = next(
            (item for item in candidates if item.candidate_id == output.candidate_id),
            None,
        )
        if selected is None:
            return DecisionProposal(
                based_on_measurement_id=sample.measurement_id,
                action=DecisionAction.HOLD,
                reason_codes=("CANDIDATE_NOT_IN_INPUT_SET",),
                provider_id=provider_id,
                provider_version=provider_version,
            )
        return DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=selected.candidate_id,
            requested_params=selected.requested_params,
            reason_codes=output.reason_codes,
            provider_id=provider_id,
            provider_version=provider_version,
        )
    return DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction(output.action),
        reason_codes=output.reason_codes,
        provider_id=provider_id,
        provider_version=provider_version,
    )
```

`analyze_once()` must:

1. Encode baseline and current density using `encode_eye_density_png()`.
2. Serialize only the latest five `V2Observation` summaries.
3. Send directive, trusted current readback, unified metrics and candidate menu.
   The text payload states the fixed priority: local candidate limits first, user directive second, eye evidence and recent outcomes third.
4. Request `response_format={"type":"json_schema", "json_schema":{"name":"vcsel_v21_decision", "strict":True, "schema":KimiDecisionOutput.model_json_schema()}}`.
5. Parse with `KimiDecisionOutput.model_validate_json(..., strict=True)`.
6. Fill `PerceptionResult.measurement_id`, `input_hash`, `provider_id` and `provider_version` locally.
7. Copy `requested_params` from the selected local `BrainCandidate`.
8. Emit a `SUCCESS` audit before returning. On `KimiApiError` or schema failure, emit an `ERROR` audit with only a stable category and then re-raise; propagate sink failure so `MultimodalApiBrain` converts it to `HOLD`.

- [ ] **Step 5: Run provider and Brain fail-closed tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_multimodal.py tests_py/ai_tuning/v2/test_brain.py tests_py/ai_tuning/v2/test_fail_closed.py -v
```

Expected: all selected tests PASS. API errors and malformed model output produce a Brain-level `HOLD`, never an exception escaping to hardware code.

- [ ] **Step 6: Commit the fused provider**

```powershell
git add src/vcsel_ai_tuning/v2/providers/kimi_multimodal.py tests_py/ai_tuning/v2/test_kimi_multimodal.py
git commit -m "feat: add Kimi multimodal V2 provider"
```

## Task 6: Register `PURE_API` without changing V2.0 factory semantics

**Files:**

- Create: `configs/ai_tuning/v2.kimi.example.yaml`
- Modify: `tests_py/ai_tuning/v2/test_factory.py`

- [ ] **Step 1: Add the strict example config**

Create:

```yaml
v2:
  brain_mode: PURE_API
  enabled_brain_modes:
    - PURE_API
  minimum_confidence: 0.70
```

Do not put API Key, model path, Base64 image data or hardware configuration in this file.

- [ ] **Step 2: Write a failing factory integration test**

Use a fake Kimi client and real `KimiMultimodalProvider`:

```python
def test_kimi_example_registers_only_pure_api(
    tmp_path, eye_sample_with_density, brain_candidates, directive, fake_kimi_client
):
    configured = load_v2_config(
        REPOSITORY_ROOT / "configs" / "ai_tuning" / "v2.kimi.example.yaml"
    )
    provider = KimiMultimodalProvider(
        client=fake_kimi_client,
        artifact_root=tmp_path,
        baseline=eye_sample_with_density,
        directive=directive,
        audit_sink=lambda record: None,
    )
    factory = BrainFactory()
    factory.register(
        BrainMode.PURE_API,
        lambda selected: MultimodalApiBrain(
            provider,
            selected.minimum_confidence,
        ),
    )

    result = factory.create(configured).analyze(
        eye_sample_with_density, (), brain_candidates
    )

    assert result.brain_mode is BrainMode.PURE_API
    assert factory.capabilities(configured)[0].available is True
    assert all(
        capability.available is False
        for capability in factory.capabilities(configured)[1:]
    )
```

- [ ] **Step 3: Run the test and verify failure, then make fixtures local and explicit**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_factory.py -k kimi_example -v
```

Expected first run: FAIL because the example config and test fixtures do not exist. Add only the small fixture builders needed by this test inside `test_factory.py`; do not import network SDKs from `factory.py`.

- [ ] **Step 4: Verify factory and import isolation**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_factory.py tests_py/ai_tuning/v2/test_import_isolation.py -v
```

Expected: all selected tests PASS. `BrainFactory` remains an explicit registry; registering `PURE_API` does not enable other modes.

- [ ] **Step 5: Commit the wiring example**

```powershell
git add configs/ai_tuning/v2.kimi.example.yaml tests_py/ai_tuning/v2/test_factory.py
git commit -m "test: wire Kimi through PURE_API factory"
```

## Task 7: Hardware-neutral V2.1 runner and execution receipt

**Files:**

- Create: `src/vcsel_ai_tuning/v2/runner.py`
- Create: `tests_py/ai_tuning/v2/test_runner.py`
- Modify: `src/vcsel_ai_tuning/v2/__init__.py`

- [ ] **Step 1: Write failing shadow and one-step tests**

Use `ScriptedFakeBrain` and a recording executor:

```python
def test_shadow_mode_never_calls_executor(eye_sample, brain_candidates, proposed_result):
    executor = RecordingExecutor()
    runner = V21Runner(
        brain=ScriptedFakeBrain((proposed_result,)),
        execution_mode=V21ExecutionMode.SHADOW,
        executor=executor,
    )

    record = runner.run_once(eye_sample, (), brain_candidates)

    assert record.executed is False
    assert record.selected_candidate_id == brain_candidates[0].candidate_id
    assert executor.calls == []


def test_one_step_executes_exactly_the_selected_local_candidate(
    eye_sample, brain_candidates, proposed_result
):
    executor = RecordingExecutor()
    runner = V21Runner(
        brain=ScriptedFakeBrain((proposed_result,)),
        execution_mode=V21ExecutionMode.ONE_STEP,
        executor=executor,
    )

    record = runner.run_once(eye_sample, (), brain_candidates)

    assert record.executed is True
    assert len(executor.calls) == 1
    assert executor.calls[0] == brain_candidates[0]
```

Add the complete runner boundary cases:

```python
@pytest.mark.parametrize("action", [DecisionAction.HOLD, DecisionAction.STOP])
def test_non_proposal_never_calls_executor(
    eye_sample, brain_candidates, result_for, action
):
    executor = RecordingExecutor()
    runner = V21Runner(
        brain=ScriptedFakeBrain((result_for(action=action),)),
        execution_mode=V21ExecutionMode.ONE_STEP,
        executor=executor,
    )

    record = runner.run_once(eye_sample, (), brain_candidates)

    assert record.executed is False
    assert executor.calls == []


def test_one_step_requires_executor():
    with pytest.raises(ValueError, match="ONE_STEP requires an executor"):
        V21Runner(
            brain=ScriptedFakeBrain(()),
            execution_mode=V21ExecutionMode.ONE_STEP,
        )


def test_runner_rejects_candidate_mismatch(
    eye_sample, brain_candidates, result_for
):
    mismatch = result_for(
        action=DecisionAction.PROPOSE,
        candidate_id=brain_candidates[0].candidate_id,
        requested_params={"eq": 999},
    )
    runner = V21Runner(
        brain=ScriptedFakeBrain((mismatch,)),
        execution_mode=V21ExecutionMode.SHADOW,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        runner.run_once(eye_sample, (), brain_candidates)


def test_one_step_rejects_second_analysis(
    eye_sample, brain_candidates, proposed_result
):
    runner = V21Runner(
        brain=ScriptedFakeBrain((proposed_result, proposed_result)),
        execution_mode=V21ExecutionMode.ONE_STEP,
        executor=RecordingExecutor(),
    )
    runner.run_once(eye_sample, (), brain_candidates)

    with pytest.raises(RuntimeError, match="already executed"):
        runner.run_once(eye_sample, (), brain_candidates)


def test_executor_failure_propagates_without_second_call(
    eye_sample, brain_candidates, proposed_result
):
    executor = RecordingExecutor(error=RuntimeError("RECOVERY_REQUIRED"))
    runner = V21Runner(
        brain=ScriptedFakeBrain((proposed_result,)),
        execution_mode=V21ExecutionMode.ONE_STEP,
        executor=executor,
    )

    with pytest.raises(RuntimeError, match="RECOVERY_REQUIRED"):
        runner.run_once(eye_sample, (), brain_candidates)

    assert len(executor.calls) == 1
```

Define `result_for`, `proposed_result` and `RecordingExecutor` as local test helpers at the top of `test_runner.py`; `result_for` must build a fully valid `BrainResult` with matching request/response hashes so the runner test does not bypass the Brain contract.

- [ ] **Step 2: Run runner tests and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_runner.py -v
```

Expected: FAIL because `runner.py` does not exist.

- [ ] **Step 3: Implement the generic runner without campaign imports**

The common Runner accepts a small executor protocol so it can be tested without hardware:

```python
class V21ExecutionMode(StrEnum):
    SHADOW = "SHADOW"
    ONE_STEP = "ONE_STEP"


class CandidateExecutor(Protocol):
    def execute(self, candidate: BrainCandidate) -> tuple[str, ...]: ...


class V21RunRecord(FrozenModel):
    measurement_id: Identifier
    execution_mode: V21ExecutionMode
    brain_result: BrainResult
    selected_candidate_id: Identifier | None = None
    executed: StrictBool
    observation_ids: tuple[Identifier, ...] = ()


class V21Runner:
    def __init__(self, *, brain, execution_mode, executor=None) -> None:
        self._brain = brain
        self._mode = V21ExecutionMode(execution_mode)
        self._executor = executor
        self._used = False
        if self._mode is V21ExecutionMode.ONE_STEP and executor is None:
            raise ValueError("ONE_STEP requires an executor")

    def run_once(self, sample, history, safe_candidates) -> V21RunRecord:
        if self._mode is V21ExecutionMode.ONE_STEP and self._used:
            raise RuntimeError("ONE_STEP runner has already executed its one analysis")
        result = self._brain.analyze(sample, history, safe_candidates)
        self._used = True
        selected = result.decision.candidate_id
        if result.decision.action is not DecisionAction.PROPOSE:
            return V21RunRecord(
                measurement_id=sample.measurement_id,
                execution_mode=self._mode,
                brain_result=result,
                selected_candidate_id=None,
                executed=False,
            )
        candidate = next(
            (item for item in safe_candidates if item.candidate_id == selected),
            None,
        )
        if candidate is None or candidate.requested_params != result.decision.requested_params:
            raise RuntimeError("Brain proposal does not match the supplied candidate menu")
        if self._mode is V21ExecutionMode.SHADOW:
            return V21RunRecord(
                measurement_id=sample.measurement_id,
                execution_mode=self._mode,
                brain_result=result,
                selected_candidate_id=candidate.candidate_id,
                executed=False,
            )
        observation_ids = self._executor.execute(candidate)
        return V21RunRecord(
            measurement_id=sample.measurement_id,
            execution_mode=self._mode,
            brain_result=result,
            selected_candidate_id=candidate.candidate_id,
            executed=True,
            observation_ids=tuple(observation_ids),
        )
```

Keep `CandidateExecutor` hardware-neutral. Export the runner contracts from `v2/__init__.py`.

- [ ] **Step 4: Run runner and Brain tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_runner.py tests_py/ai_tuning/v2/test_brain.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the common runner**

```powershell
git add src/vcsel_ai_tuning/v2/runner.py src/vcsel_ai_tuning/v2/__init__.py tests_py/ai_tuning/v2/test_runner.py
git commit -m "feat: add V2.1 shadow and one-step runner"
```

## Task 8: Existing CampaignOrchestrator execution adapter

**Files:**

- Modify: `src/vcsel_ai_tuning/v2/runner.py`
- Modify: `tests_py/ai_tuning/v2/test_runner.py`

- [ ] **Step 1: Write a failing adapter test with a fake orchestrator**

The adapter must translate only flat integer I2C parameters and call `run_one_candidate()` once:

```python
def test_campaign_executor_maps_brain_candidate_to_authorized_i2c_candidate(
    authorization
):
    orchestrator = RecordingOrchestrator()
    executor = CampaignOrchestratorExecutor(
        orchestrator=orchestrator,
        authorization=authorization,
        model_version="moonshot-kimi-k3|v2.1-1",
    )
    candidate = BrainCandidate(
        candidate_id="i2c-eq-up-1-11",
        requested_params={"eq": 11},
    )

    observation_ids = executor.execute(candidate)

    assert len(orchestrator.calls) == 1
    used_authorization, campaign_candidate = orchestrator.calls[0]
    assert used_authorization == authorization
    assert campaign_candidate.stage == authorization.stage
    assert campaign_candidate.requested_i2c == {"eq": 11}
    assert campaign_candidate.requested_fir is None
    assert observation_ids == ("observation-001",)
```

Add these adapter rejection tests:

```python
@pytest.mark.parametrize(
    "requested_params",
    [
        {},
        {"eq": {"nested": 11}},
        {"eq": 11.0},
        {"eq": True},
        {"unknown": 11},
    ],
)
def test_campaign_executor_rejects_non_i2c_candidate_values(
    authorization, requested_params
):
    orchestrator = RecordingOrchestrator()
    executor = CampaignOrchestratorExecutor(
        orchestrator=orchestrator,
        authorization=authorization,
        model_version="moonshot-kimi-k3|v2.1-1",
    )
    candidate = BrainCandidate(
        candidate_id="candidate-invalid",
        requested_params=requested_params,
    )

    with pytest.raises(ValueError):
        executor.execute(candidate)

    assert orchestrator.calls == []


def test_campaign_executor_rejects_campaign_mismatch(authorization):
    orchestrator = RecordingOrchestrator(campaign_id="other-campaign")

    with pytest.raises(ValueError, match="Campaign mismatch"):
        CampaignOrchestratorExecutor(
            orchestrator=orchestrator,
            authorization=authorization,
            model_version="moonshot-kimi-k3|v2.1-1",
        )
```

- [ ] **Step 2: Run the adapter test and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_runner.py -k campaign_executor -v
```

Expected: FAIL because `CampaignOrchestratorExecutor` does not exist.

- [ ] **Step 3: Implement the narrow adapter**

Add imports only from campaign models, not hardware drivers:

```python
class CampaignOrchestratorExecutor:
    def __init__(self, *, orchestrator, authorization, model_version: str) -> None:
        if not isinstance(authorization, StageAuthorization):
            raise TypeError("authorization must be a StageAuthorization")
        if getattr(orchestrator, "campaign_id", None) != authorization.campaign_id:
            raise ValueError("orchestrator and authorization Campaign mismatch")
        self._orchestrator = orchestrator
        self._authorization = authorization
        self._model_version = model_version

    def execute(self, candidate: BrainCandidate) -> tuple[str, ...]:
        allowed_names = {
            spec.name for spec in self._authorization.parameter_specs if spec.enabled
        }
        requested: dict[str, int] = {}
        for name, value in candidate.requested_params.items():
            if name not in allowed_names:
                raise ValueError(f"candidate parameter {name} is not authorized")
            if type(value) is not int:
                raise ValueError("V2.1 I2C candidate values must be integer codes")
            requested[name] = value
        if not requested:
            raise ValueError("V2.1 I2C candidate must change one authorized parameter")
        campaign_candidate = Candidate(
            candidate_id=candidate.candidate_id,
            stage=self._authorization.stage,
            requested_i2c=requested,
            requested_fir=None,
            predicted_metrics={},
            model_version=self._model_version,
            reason="Kimi selected a locally generated safe candidate",
        )
        observations = self._orchestrator.run_one_candidate(
            self._authorization,
            campaign_candidate,
            repeat_count=1,
        )
        return tuple(observation.observation_id for observation in observations)
```

The existing Orchestrator remains responsible for authorization validation, write, readback, DCA remeasurement, rollback and `RECOVERY_REQUIRED`.

- [ ] **Step 4: Run runner plus campaign authorization/orchestrator tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_runner.py tests_py/ai_tuning/campaign/test_authorization.py tests_py/ai_tuning/campaign/test_orchestrator.py -v
```

Expected: all selected tests PASS. No live adapters are constructed.

- [ ] **Step 5: Commit the execution adapter**

```powershell
git add src/vcsel_ai_tuning/v2/runner.py tests_py/ai_tuning/v2/test_runner.py
git commit -m "feat: bridge V2 proposal to campaign executor"
```

## Task 9: Kimi shadow CLI and audit files

**Files:**

- Create: `src/vcsel_ai_tuning/v2/kimi_cli.py`
- Create: `tests_py/ai_tuning/v2/test_kimi_cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Define the command entry point**

Add:

```toml
vcsel-v2-kimi-shadow = "vcsel_ai_tuning.v2.kimi_cli:main"
```

The command accepts:

```text
--v2-config PATH
--artifact-root PATH
--baseline-json PATH
--sample-json PATH
--candidates-json PATH
--directive TEXT
--output-dir PATH
--base-url URL
--model NAME
--reasoning-effort low|high|max
```

It has no real-execution flag and imports no DCA/I2C/BERT adapter.

- [ ] **Step 2: Write a failing CLI test with an injected fake Kimi client**

```python
def test_shadow_command_writes_directive_replay_and_sanitized_audit(
    tmp_path, sample_files, candidate_file, fake_kimi_client
):
    result = run_shadow(
        v2_config=REPOSITORY_ROOT / "configs/ai_tuning/v2.kimi.example.yaml",
        artifact_root=sample_files.artifact_root,
        baseline_json=sample_files.sample_json,
        sample_json=sample_files.sample_json,
        candidates_json=candidate_file,
        directive_text="优先改善中眼高度。",
        output_dir=tmp_path / "output",
        client=fake_kimi_client,
    )

    assert result.brain_result.brain_mode is BrainMode.PURE_API
    assert (tmp_path / "output/directive.json").is_file()
    assert (tmp_path / "output/replay.jsonl").is_file()
    assert (tmp_path / "output/kimi_api.jsonl").is_file()
    combined = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "output").iterdir()
    )
    assert "MOONSHOT_API_KEY" not in combined
    assert "data:image/png;base64" not in combined
```

Add explicit CLI boundary tests:

```python
def test_shadow_refuses_nonempty_output_directory(tmp_path, valid_shadow_inputs):
    output = tmp_path / "output"
    output.mkdir()
    (output / "owned.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory"):
        run_shadow(**valid_shadow_inputs, output_dir=output)

    assert (output / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_shadow_rejects_invalid_sample_json(tmp_path, valid_shadow_inputs):
    broken = tmp_path / "sample.json"
    broken.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        run_shadow(**valid_shadow_inputs, sample_json=broken)


def test_shadow_rejects_cross_campaign_baseline(valid_shadow_inputs, other_campaign_sample):
    with pytest.raises(ValueError, match="Campaign mismatch"):
        run_shadow(
            **valid_shadow_inputs,
            baseline_json=other_campaign_sample,
        )


def test_main_requires_api_key(monkeypatch, valid_cli_argv):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(KimiApiError, match="MOONSHOT_API_KEY is not set"):
        main(valid_cli_argv)
```

Create `valid_shadow_inputs`, `other_campaign_sample` and `valid_cli_argv` as local fixtures that point only to temporary files.

- [ ] **Step 3: Run CLI tests and verify failure**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_cli.py -v
```

Expected: FAIL because `kimi_cli.py` does not exist.

- [ ] **Step 4: Implement testable `run_shadow()` and a thin `main()`**

`run_shadow()` must:

1. Load `EyeSample` objects with `model_validate_json(..., strict=True)`.
2. Load a JSON array with `TypeAdapter(list[BrainCandidate])` and convert it to a tuple.
3. Create `OptimizationDirective` with the baseline measurement ID and current UTC time.
4. Create `KimiMultimodalProvider`, register only `PURE_API`, and call the existing `replay()` for one case.
5. Atomically write `directive.json`, `replay.jsonl` and `kimi_api.jsonl` into a newly created output directory.
6. Refuse to overwrite an existing non-empty output directory.

Use the existing `write_replay_jsonl()` for the replay file and `write_kimi_audit_jsonl()` for API audit. Write the directive atomically with this local helper:

```python
def _write_model_json_atomic(model: FrozenModel, destination: Path) -> Path:
    payload = model.model_dump_json(indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        return destination
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
```

`main()` parses arguments, builds `KimiClientConfig` from `--base-url`, `--model` and `--reasoning-effort`, constructs `KimiClient.from_environment(...)`, calls `run_shadow()` and prints only measurement ID, action, candidate ID, latency and output directory. Defaults remain the official Moonshot URL, `kimi-k3` and `low`; the Key remains environment-only.

- [ ] **Step 5: Run CLI and replay tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_cli.py tests_py/ai_tuning/v2/test_replay.py -v
```

Expected: all selected tests PASS with no network and no hardware calls.

- [ ] **Step 6: Commit the shadow entry point**

```powershell
git add pyproject.toml src/vcsel_ai_tuning/v2/kimi_cli.py tests_py/ai_tuning/v2/test_kimi_cli.py
git commit -m "feat: add Kimi V2.1 shadow command"
```

## Task 10: Quickstart and end-to-end FAKE verification

**Files:**

- Create: `docs/v2.1-kimi-quickstart.md`
- Modify: `tests_py/ai_tuning/v2/test_kimi_cli.py`

- [ ] **Step 1: Add one complete fake end-to-end test**

The test must exercise actual NPY verification, PNG encoding, directive injection, candidate selection, `MultimodalApiBrain`, factory outer gate, shadow Runner and all three audit outputs in one flow. The fake Kimi response chooses a real input candidate; assert no executor and no network module are used.

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_cli.py -k end_to_end -v
```

Expected first run: FAIL until the full fixture is connected; after wiring, PASS.

- [ ] **Step 2: Write the Windows quickstart**

Document these exact commands without a real Key value:

```powershell
python -m pip install -e ".[dev,kimi]"
$env:MOONSHOT_API_KEY = "<set-in-this-PowerShell-session>"
vcsel-v2-kimi-shadow `
  --v2-config configs/ai_tuning/v2.kimi.example.yaml `
  --artifact-root C:\path\to\campaign `
  --baseline-json C:\path\to\baseline\eye_sample.json `
  --sample-json C:\path\to\current\eye_sample.json `
  --candidates-json C:\path\to\safe_candidates.json `
  --directive "优先改善中眼高度，同时保持整体对称。" `
  --output-dir C:\path\to\output\v21-shadow-001
```

Explain the expected output files, how to recognize `PROPOSE/HOLD/STOP`, why V2.1 does not permanently train Kimi, and that `ONE_STEP` is activated only by the backend supplying an existing `CampaignOrchestrator` plus `StageAuthorization`.

- [ ] **Step 3: Run all V2 tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/v2 -v
```

Expected: all V2 tests PASS.

- [ ] **Step 4: Run the full Python suite**

Run:

```powershell
python -m pytest
```

Expected: the full suite PASS with only previously documented skips. Do not call the real Kimi API or hardware from pytest.

- [ ] **Step 5: Perform a real API smoke test only after unit tests pass**

Use one verified historical sample and a small safe candidate list. Run the shadow command once with `reasoning_effort=low`. Expected:

- exit code 0;
- one `BrainResult` with `PURE_API`;
- one sanitized Kimi audit record;
- no I2C/BERT/FIR call;
- no API Key or Base64 string in saved files.

If the API returns an error, preserve the output directory, report the sanitized failure category, and do not proceed to `ONE_STEP`.

- [ ] **Step 6: Commit documentation and final verification coverage**

```powershell
git add docs/v2.1-kimi-quickstart.md tests_py/ai_tuning/v2/test_kimi_cli.py
git commit -m "docs: add V2.1 Kimi shadow quickstart"
```

## Completion checkpoint

V2.1 is ready for a controlled real one-step test only when all of the following are true:

- the V1.1 producer supplies a valid current `EyeSample` and artifact root;
- the backend supplies an active `StageAuthorization` and existing `CampaignOrchestrator`;
- the Kimi shadow command succeeds on the same sample format;
- the selected `candidate_id` maps exactly to a locally generated candidate;
- full tests pass and no secret appears in artifacts;
- the run is explicitly started in `ONE_STEP`, never through the read-only Dashboard.
