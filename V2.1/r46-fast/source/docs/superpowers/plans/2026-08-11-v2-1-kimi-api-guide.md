# V2.1 Kimi API 中文说明书 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 创建一份简洁、准确、小白可读的中文说明书，解释现有 V2.1 如何调用 Kimi API、保存本地记忆，并在条件具备后通过 V2.0 安全链请求一次真实 I2C 调整。

**Architecture:** 只新增一份 Markdown 文档，不修改任何 Python、配置、测试、Dashboard 或硬件代码。说明书中的代码直接使用当前已有接口，并明确区分“真实 API 评分示例”与“依赖现成 Campaign 上下文的真实单步示例”。

**Tech Stack:** 中文 Markdown、PowerShell、Python 3.12、现有 vcsel_ai_tuning.v2 接口。

---

## 文件范围

- Create: docs/v2.1-kimi-api-control-and-collaboration-guide.md
- Reference only: src/vcsel_ai_tuning/v2/providers/kimi_client.py
- Reference only: src/vcsel_ai_tuning/v2/providers/kimi_provider.py
- Reference only: src/vcsel_ai_tuning/v2/kimi_memory.py
- Reference only: src/vcsel_ai_tuning/v2/kimi_step.py
- Reference only: src/vcsel_ai_tuning/v2/brain.py
- Reference only: src/vcsel_ai_tuning/v2/factory.py
- Reference only: src/vcsel_ai_tuning/v2/contracts.py
- Reference only: src/vcsel_ai_tuning/campaign/models.py

任何生产代码、测试代码和配置文件都不得修改。

### Task 1: 固定说明书使用的真实接口

**Files:**
- Read: src/vcsel_ai_tuning/v2/providers/kimi_client.py
- Read: src/vcsel_ai_tuning/v2/providers/kimi_provider.py
- Read: src/vcsel_ai_tuning/v2/kimi_memory.py
- Read: src/vcsel_ai_tuning/v2/kimi_step.py
- Read: src/vcsel_ai_tuning/v2/brain.py
- Read: src/vcsel_ai_tuning/v2/factory.py

- [ ] **Step 1: 验证评分示例需要的导入**

Run:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& "C:\Users\NateS\Documents\VCSEL\.venv-ai-tuning\Scripts\python.exe" -c "from vcsel_ai_tuning.v2.brain import MultimodalApiBrain, analyze_factory_gated; from vcsel_ai_tuning.v2.contracts import BrainCandidate, BrainMode; from vcsel_ai_tuning.v2.factory import BrainFactory, BrainFactoryConfig; from vcsel_ai_tuning.v2.kimi_memory import KIMI_PROVIDER_ID, KimiCampaignMemory; from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient; from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider; print('SCORE_IMPORTS_OK')"
~~~

Expected: SCORE_IMPORTS_OK。

- [ ] **Step 2: 验证真实单步入口的签名**

Run:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& "C:\Users\NateS\Documents\VCSEL\.venv-ai-tuning\Scripts\python.exe" -c "import inspect; from vcsel_ai_tuning.v2.kimi_step import submit_kimi_step; print(inspect.signature(submit_kimi_step))"
~~~

Expected parameters:

~~~text
runtime, campaign_id, brain, memory, authorization, sample, candidates,
instruction, output_path
~~~

- [ ] **Step 3: 核对说明书中的安全事实**

逐项从当前源码确认：

~~~text
KimiApiClient 读取 MOONSHOT_API_KEY。
KimiProvider 发送当前眼图、可选最佳眼图、最多五条摘要、回读、指标、指示和本地候选。
execute_kimi_step 拒绝 FIR 候选。
submit_kimi_step 通过 CampaignRuntime.submit_stage 进入现有执行链。
Orchestrator 负责范围和步长检查、I2C执行、回读、重新测量、回滚和 RECOVERY_REQUIRED。
~~~

### Task 2: 编写中文小白说明书

**Files:**
- Create: docs/v2.1-kimi-api-control-and-collaboration-guide.md

- [ ] **Step 1: 写一分钟总览**

说明书开头必须包含：

~~~text
DCA或历史眼图
→ 本地程序准备眼图和安全候选
→ Kimi评分并选择最多一个候选
→ 本地程序再次检查
→ V2.0安全链执行I2C、回读并重新测量
→ 本地保存结果，供下一次Kimi参考
~~~

紧接着用一句话说明：Kimi只返回结构化建议，没有I2C、串口、VISA、DCA、BERT、FIR或setFir对象。

- [ ] **Step 2: 写四方分工与准备清单**

使用以下四方分工：

~~~text
Kimi：看眼图、打分、从本地候选中选择。
V2.1：准备请求、保存记忆、验证Kimi回答。
V2.0：检查范围和步长、执行、回读、重新测量、回滚。
使用者：提供Key、样本、安全范围、Campaign和授权。
~~~

准备清单必须分成“只评分需要”和“真实I2C额外需要”。

- [ ] **Step 3: 写API Key和发送内容**

API Key只展示：

~~~powershell
$env:MOONSHOT_API_KEY = "在这里粘贴真实Key"
~~~

解释它只供当前PowerShell窗口使用，Python自动读取。不得把真实Key写入说明书、源码、JSON、截图或日志。

发送内容必须列出：当前眼图、可选最佳眼图、最多五条摘要、回读、指标、指示和安全候选。明确不发送：完整DCA屏幕、API Key、全部历史图片、NPY文本和硬件对象。

- [ ] **Step 4: 写真实API只评分示例**

代码前必须先写：

~~~text
下面代码假设现有数据流程已经提供 sample、candidates 和 artifact_root。
它是真实API评分代码，但不是独立的样本文件读取程序。
它不会创建LIVE_TUNING runtime，也不会执行I2C。
~~~

使用以下当前接口形状：

~~~python
from pathlib import Path

from vcsel_ai_tuning.v2.brain import (
    MultimodalApiBrain,
    analyze_factory_gated,
)
from vcsel_ai_tuning.v2.contracts import BrainCandidate, BrainMode
from vcsel_ai_tuning.v2.factory import BrainFactory, BrainFactoryConfig
from vcsel_ai_tuning.v2.kimi_memory import (
    KIMI_PROVIDER_ID,
    KimiCampaignMemory,
)
from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient
from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider

# sample、candidates、artifact_root 由现有数据流程提供。
shadow_memory = KimiCampaignMemory(
    Path("artifacts") / sample.campaign_id / "kimi_shadow_memory.jsonl",
    campaign_id=sample.campaign_id,
)
client = KimiApiClient()
provider = KimiProvider(
    client,
    artifact_root=artifact_root,
    instruction="观察当前PAM4眼图，只能从给定候选中选择。",
    memory=shadow_memory,
)

factory = BrainFactory()
factory.register(
    BrainMode.PURE_API,
    lambda config: MultimodalApiBrain(provider, config.minimum_confidence),
)
brain = factory.create(
    BrainFactoryConfig(
        brain_mode=BrainMode.PURE_API,
        enabled_brain_modes=(BrainMode.PURE_API,),
        minimum_confidence=0.70,
    )
)

safe_candidates = tuple(
    BrainCandidate(
        candidate_id=candidate.candidate_id,
        requested_params=dict(candidate.requested_i2c),
    )
    for candidate in candidates
)
result = analyze_factory_gated(brain, sample, (), safe_candidates)

if result.perception.provider_id == KIMI_PROVIDER_ID:
    shadow_memory.append(sample, result)

print("动作：", result.decision.action.value)
print("质量分数：", result.perception.quality_score)
print("置信度：", result.perception.confidence)
print("原因：", list(result.decision.reason_codes))
~~~

代码后逐项解释client、provider、memory、factory、candidate和result。明确影子记忆不得当作真实执行记录。

- [ ] **Step 5: 写真实I2C单步示例**

代码前必须先写：

~~~text
下面代码不能单独运行。runtime、campaign_id、brain、memory、sample、
authorization和candidates必须由已经完成预检并进入匹配I2C阶段的V2.0流程提供。
~~~

唯一展示的执行入口：

~~~python
from pathlib import Path

from vcsel_ai_tuning.v2.kimi_step import submit_kimi_step

future = submit_kimi_step(
    runtime,
    campaign_id=campaign_id,
    brain=brain,
    memory=memory,
    authorization=authorization,
    sample=sample,
    candidates=candidates,
    instruction="观察当前眼图，只能选择给定的安全I2C候选。",
    output_path=Path("artifacts") / campaign_id / "v21_result.json",
)
result_document = future.result()
print("执行状态：", result_document["execution_status"])
~~~

逐项解释参数和Future。不得展示任何直接I2C、DCA、BERT、FIR或setFir调用。

- [ ] **Step 6: 写结果、记忆、状态和测试顺序**

解释v21_result.json、Campaign observation、kimi_shadow_memory.jsonl和kimi_memory.jsonl。明确“有建议”不等于“已执行”；只有EXECUTED、匹配回读和新DCA观察共同成立才算真实执行。

状态表包含：HOLD、PROVIDER_ERROR、LOW_CONFIDENCE、INVALID_EYE、OOD、FAILED、RECOVERY_REQUIRED。

结尾使用：

~~~text
1. 一张历史真实眼图调用真实Kimi，不碰硬件。
2. DCA回来后做5～10次只评分测试。
3. 只开放一个I2C参数，执行一次live单步。
4. 单步稳定后，才讨论连续优化。
~~~

同时说明：同一次采集绑定、原始眼图与均衡眼图分离完成前，只能验证流程，不能宣称优化有效。

### Task 3: 验证并提交说明书

**Files:**
- Verify: docs/v2.1-kimi-api-control-and-collaboration-guide.md

- [ ] **Step 1: 再次验证文档里的导入**

Run:

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& "C:\Users\NateS\Documents\VCSEL\.venv-ai-tuning\Scripts\python.exe" -c "from vcsel_ai_tuning.v2.brain import MultimodalApiBrain, analyze_factory_gated; from vcsel_ai_tuning.v2.factory import BrainFactory, BrainFactoryConfig; from vcsel_ai_tuning.v2.kimi_memory import KIMI_PROVIDER_ID, KimiCampaignMemory; from vcsel_ai_tuning.v2.kimi_step import submit_kimi_step; from vcsel_ai_tuning.v2.providers.kimi_client import KimiApiClient; from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider; print('GUIDE_IMPORTS_OK')"
~~~

Expected: GUIDE_IMPORTS_OK。

- [ ] **Step 2: 扫描误导或危险内容**

Run:

~~~powershell
$guide = "docs/v2.1-kimi-api-control-and-collaboration-guide.md"
rg -n "TBD|TODO|vcsel-v21-kimi|sk-[A-Za-z0-9_-]{20,}|\.apply\(" $guide
~~~

Expected: no output。说明书不能出现不存在的命令、类似真实Key的内容或直接adapter调用。

- [ ] **Step 3: 检查纯文档范围**

Run:

~~~powershell
git status --short
git diff --check
~~~

Expected: 只有docs/v2.1-kimi-api-control-and-collaboration-guide.md新增或修改，git diff --check无输出。

- [ ] **Step 4: 提交说明书**

~~~powershell
git add -- "docs/v2.1-kimi-api-control-and-collaboration-guide.md"
git commit -m "docs: add V2.1 Kimi API beginner guide"
~~~
