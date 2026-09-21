# VCSEL AI Tuning V2 可插拔优化大脑设计

**日期：** 2026-07-30

**状态：** 设计已确认；进入V2 Foundation实施计划阶段

**分支：** `codex/V2.0`

**产品阶段：** V2 Foundation / V2基础架构

## 1. 目标

V2在不复制硬件闭环的前提下，为同一套六步优化流程提供可替换的AI实现：

1. `PURE_API`：多模态API直接完成眼图评分和Decision；
2. `LOCAL_VISION_API`：本地开源视觉模型评分，API完成Decision；
3. `LOCAL_VISION_LOCAL`：本地开源视觉模型评分，本地Decision完成决策；
4. `DIRECT_FEATURES_LOCAL`：直接从DCA数字眼图提取物理特征，本地Decision完成决策。

这四项是架构允许注册的Provider组合上限，不是首版必须全部实现或启用的交付清单。每个版本只公布并验收本次实际启用的模式，其他模式可以保留为未安装能力。

除第二、三步的AI实现外，DCA采集、样本验证、安全检查、硬件执行、参数回读、回滚和重新测量全部共用。

V2的核心承诺是：

> 更换模型不需要更改六步主循环，更不能绕过确定性安全门。

## 2. 必须保持不变的六步闭环

```text
1. DCA测量当前眼图
2. 计算眼图质量
3. 提出下一组候选参数
4. 执行确定性安全检查
5. 写入硬件并回读验证
6. DCA重新测量、比较、更新优化器或停止
```

状态流为：

```text
MEASURED
-> SCORED
-> PROPOSED
-> APPROVED
-> APPLIED
-> VERIFIED
-> MEASURED
```

任何AI Provider失败都只能返回`HOLD`或`NO_SAFE_CANDIDATE`。模型不能直接调用I2C、FIR、NLC、BERT、DCA写命令、`setFir`或回滚函数。

## 3. V1与V2的边界

### 3.1 V1负责客观事实

V1负责：

- 安全地执行一次参数调整；
- 硬件参数回读；
- DCA严格单次采集；
- 分离`CHAN1A`原始数据和`FUNC1` TDECQ数据；
- 保存可信Sweep样本；
- 失败回滚和`RECOVERY_REQUIRED`；
- 提供FAKE/replay数据源。

V1输出统一的`EyeSample`，不导入V2模型，也不知道当前使用哪种Brain模式。

### 3.2 V2负责主观评价和候选建议

V2负责：

- 将`EyeSample`转换成模型所需Payload；
- 计算视觉质量、缺陷、置信度和OOD状态；
- 基于历史Observation提出候选参数；
- 对不同Brain模式进行replay和效果比较；
- 将统一`DecisionProposal`交给现有安全层。

V2 Provider不得持有或构造真实硬件连接。

## 4. 统一输入：EyeSample

所有模式从同一个客观输入开始：

```text
EyeSample
|- measurement_id
|- campaign_id
|- sweep_point_id
|- raw_eye_density_ref
|- waveform_ref（可选在线输入，保留为产物）
|- metrics
|  |- tdecq_db
|  |- outer_er_chan1a_db
|  |- rlm
|  |- outer_oma
|  `- other_metrics
|- requested_params
|- applied_params
|- readback_params
|- dca_config_fingerprint
|- measurement_status
|- captured_at
`- artifact_hashes
```

`EyeSample`只包含DCA和硬件客观事实，不包含任何模型评分。

## 5. 模型专用Payload在哪里生成

模型专用Payload不在DCA采集步骤生成，而在`OptimizationBrain`内部生成。

### 5.1 PURE_API

```text
EyeSample
-> PureApiPayloadBuilder
-> 由密度矩阵确定性渲染的干净眼图
-> 多模态API
-> PerceptionResult + DecisionProposal
```

本地Payload Builder只负责固定对数归一化、等比例缩放、填充、RGB转换和JSON打包，不判断眼图质量。视觉特征提取发生在多模态API内部。

禁止向API发送包含TDECQ、ER、菜单、结果表或其他答案泄漏信息的完整DCA屏幕。

### 5.2 LOCAL_VISION_API

```text
EyeSample
-> LocalVisionPreprocessor
-> LocalVisionScorer
-> PerceptionResult
-> DecisionPayloadBuilder
-> API DecisionProvider
-> DecisionProposal
```

本地视觉模型负责看懂眼图；API只读取结构化视觉结果、工程指标、历史和安全候选。

### 5.3 LOCAL_VISION_LOCAL

```text
EyeSample
-> LocalVisionScorer
-> PerceptionResult
-> LocalDecisionProvider
-> DecisionProposal
```

第一版本地Decision复用安全贝叶斯优化。积累足够状态、动作和结果数据后，才允许替换为训练后的参数效果预测模型或受限策略模型。

### 5.4 DIRECT_FEATURES_LOCAL

```text
EyeSample
-> DirectFeatureScorer
-> 眼高、眼宽、skew、噪声、振铃等物理特征
-> PerceptionResult
-> LocalDecisionProvider
-> DecisionProposal
```

该模式用于验证是否可以完全去掉通用视觉模型。

## 6. 架构选择

### 6.1 不采用：主循环中的模式if/else

不在六步主循环中散布`if mode == ...`。这种方式会让每次增加模型都修改采集、安全和执行代码。

### 6.2 不采用：强制两段模型接口

如果所有模式都强制先调用`EyeScorer`、再调用`DecisionProvider`，`PURE_API`会被迫调用API两次，增加延迟和成本。

### 6.3 采用：统一Brain外观加内部组合

对主循环只暴露：

```python
class OptimizationBrain(Protocol):
    def analyze(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[Candidate, ...],
    ) -> BrainResult: ...
```

实现分为：

```text
MultimodalApiBrain
`- PURE_API，一次API调用完成评分和Decision

ComposedBrain
|- EyeScorer
`- DecisionProvider
```

`ComposedBrain`支持：

```text
LocalVisionScorer + ApiDecisionProvider
LocalVisionScorer + LocalDecisionProvider
DirectFeatureScorer + LocalDecisionProvider
```

## 7. 统一合同

### 7.1 PerceptionResult

```text
measurement_id
quality_score          0.0至1.0
confidence             0.0至1.0
valid                  bool
ood                    bool
defects                多标签概率
feature_vector         可选
provider_id
provider_version
input_hash
```

### 7.2 DecisionProposal

```text
based_on_measurement_id
action                 PROPOSE / HOLD / STOP / REQUEST_HUMAN
candidate_id           可选
requested_params       可选
reason_codes
expected_improvement   可选
provider_id
provider_version
```

### 7.3 BrainResult

```text
perception
decision
brain_mode
latency_ms
request_hash
response_hash
```

模型输出格式正确不代表候选安全。`DecisionProposal`必须经过现有确定性安全层，才能进入硬件执行器。

## 8. 复用现有V1.1能力

V2不重写以下现有能力：

- `OptimizerProtocol.suggest()`；
- `CandidateSuggestion`及相关终止结果；
- `SafetyPolicy.evaluate_candidate()`；
- `SafetyPolicy.evaluate_readback()`；
- Campaign状态机；
- B0、rollback和`RECOVERY_REQUIRED`；
- 硬件执行和回读适配器；
- artifact repository和Dashboard只读展示边界。

现有`TuningObservation`以TDECQ为主并有较大调用影响范围。V2初期不直接破坏性修改它，而是组合：

```text
V2Observation
|- tuning_observation: TuningObservation
`- perception: PerceptionResult
```

由适配器将`V2Observation`交给V2优化器。V1.1现有路径保持可运行。

## 9. 模式配置与Campaign锁定

用户使用预设模式，避免暴露大量无效组合：

```yaml
v2:
  brain_mode: LOCAL_VISION_API

  enabled_brain_modes:
    - LOCAL_VISION_API

  local_vision:
    provider: dinov2_vits14
    checkpoint: models/eye_quality_v1.pt

  decision_api:
    provider: kimi
    model: kimi-k3
    reasoning_effort: low
    timeout_s: 30

  local_decision:
    provider: safe_bayesian
```

架构预留的`brain_mode`标识：

```text
PURE_API
LOCAL_VISION_API
LOCAL_VISION_LOCAL
DIRECT_FEATURES_LOCAL
```

具体版本可以只实现并启用其中一部分。`BrainFactory`只允许创建`enabled_brain_modes`中的模式；未安装或未启用的模式必须在能力报告中显示为不可用，并在被请求时明确拒绝，不能静默回退到另一种模式。

Brain模式在Campaign创建时写入Campaign配置和审计记录，并在整个Campaign内锁定。

禁止API故障后静默切换到另一个Provider，因为不同Provider的评分尺度可能不同。需要切换时必须结束当前Stage或Campaign，记录切换，并用新Provider重新评分历史样本后启动新优化阶段。

API Key不写入YAML、数据库、Prompt记录或artifact，必须从环境变量或专用凭据提供者读取。

## 10. 失败行为

以下情况统一返回`HOLD`，不能产生可执行候选：

- API超时、429或服务错误；
- API返回无法解析或不符合Schema的内容；
- 本地模型或权重缺失；
- 输入measurement ID与输出不一致；
- 输入样本无效；
- 低置信度；
- OOD；
- 特征或评分不是有限数值；
- Provider版本或配置与Campaign冻结值不一致。

安全层拒绝的候选只保存拒绝原因，不生成“调整后眼图”。硬件回读失败进入现有rollback流程；rollback失败进入`RECOVERY_REQUIRED`。

## 11. 最小代码目录树

```text
src/vcsel_ai_tuning/v2/
|- __init__.py
|- contracts.py
|- brain.py
|- factory.py
|- preprocessing.py
|- providers/
|  |- __init__.py
|  |- multimodal_api.py
|  |- local_vision.py
|  |- api_decision.py
|  `- local_decision.py
|- replay.py
`- evaluation.py

configs/ai_tuning/
`- v2.example.yaml

tests_py/ai_tuning/v2/
|- test_contracts.py
|- test_brain.py
|- test_factory.py
|- test_preprocessing.py
|- test_multimodal_api.py
|- test_local_vision.py
|- test_api_decision.py
|- test_local_decision.py
|- test_replay.py
|- test_evaluation.py
`- test_fail_closed.py
```

该目录树刻意保持紧凑。只有当单个文件职责明显过大时才继续拆分。

## 12. 最小在线记录

主循环每点至少记录：

```text
raw_eye_density.npy
record.json
```

`record.json`统一保存：

- measurement、Campaign和Sweep Point ID；
- requested、applied和readback参数；
- TDECQ、ER和必要工程指标；
- Brain模式和Provider版本；
- PerceptionResult；
- DecisionProposal；
- 安全检查结果；
- DCA配置指纹；
- 文件哈希。

PNG、完整DCA屏幕、详细报告和API解释不阻塞每一个Sweep点，可以作为派生产物或按策略保存。V1格式验证阶段仍可保留更完整证据，待证明冗余后再缩减。

## 13. V1完成前可以制作的部分

以下工作只依赖合同和FAKE/replay，可立即进行：

1. 定义`EyeSample`、`PerceptionResult`、`DecisionProposal`、`BrainResult`和`V2Observation`；
2. 建立`OptimizationBrain`、`EyeScorer`和`DecisionProvider` Protocol；
3. 建立`MultimodalApiBrain`和`ComposedBrain`骨架；
4. 建立`BrainFactory`、Provider注册机制和已启用模式的能力验证；
5. 为首版计划启用的模式建立FAKE Provider，并用一个通用合同FAKE证明Provider可以替换而不修改六步主循环；
6. 建立replay runner；
7. 定义Kimi请求、响应JSON Schema和失败转`HOLD`规则；
8. 完成密度矩阵的确定性预处理；
9. 为现有`OptimizerProtocol`建立`LocalDecisionProvider`适配器；
10. 建立Provider合同测试和fail-closed测试；
11. 建立延迟、输出一致性、成本、排序准确率和安全拒绝率评估框架；
12. 使用当前能力探测样本做格式和smoke test，但不用于宣称模型准确率。

真实Kimi网络调用需要API Key和外部数据发送授权。API客户端可以先用FAKE响应完成，不阻塞架构开发。

## 14. 必须等待V1数据的部分

以下工作必须等待可信Sweep数据：

- 正式训练DINOv2评分头或小型CNN；
- 校准视觉质量分数、置信度和OOD阈值；
- 证明物理特征是否可以替代视觉模型；
- 训练参数效果预测模型或本地策略模型；
- 比较实验计划中实际实现并启用的Brain模式；四种模式不要求同时参加；
- 确定最终生产Provider；
- 校准视觉目标与TDECQ、ER护栏；
- 接入真实硬件影子模式和受限闭环；
- 宣称任何模型改善了真实眼图。

推荐数据门槛：

```text
30至50点：验证格式和采集一致性
200至500个状态：比较模型路线
300至800个人工A/B标签：训练和评估视觉排序
```

## 15. 单一Git工作树下的多任务并行

一个工作树可以并行，但必须按文件所有权而不是按“大家一起改主循环”拆任务。

### 15.1 先串行冻结共享合同

第一阶段由一个集成负责人独占以下文件：

```text
contracts.py
brain.py
factory.py
v2.example.yaml
```

先完成并冻结接口，再开始并行Provider开发。未冻结合同前并行会造成反复改接口和大量冲突。

### 15.2 冻结后并行四条任务线

```text
任务线A：PURE_API
|- providers/multimodal_api.py
`- test_multimodal_api.py

任务线B：本地视觉
|- preprocessing.py
|- providers/local_vision.py
|- test_preprocessing.py
`- test_local_vision.py

任务线C：Decision
|- providers/api_decision.py
|- providers/local_decision.py
|- test_api_decision.py
`- test_local_decision.py

任务线D：replay与评估
|- replay.py
|- evaluation.py
|- test_replay.py
`- test_evaluation.py
```

### 15.3 单一工作树协作规则

- 每条任务线只能修改自己的文件；
- 共享合同、Factory和配置只由集成负责人修改；
- 并行工作者不运行全仓库自动格式化；
- 并行工作者不执行`git add .`；
- 并行工作者不切换分支；
- 并行工作者不独立提交共享工作树；
- 每条任务线先运行自己的目标测试；
- 集成负责人统一检查diff、运行全套测试并提交；
- 需要修改共享合同的请求先汇总，由集成负责人一次性处理。

这种方式使多个任务共享同一工作树时仍能并行，同时避免彼此覆盖。

## 16. 并行任务依赖顺序

```text
合同和Brain接口
        |
        +--> PURE_API Provider
        +--> Local Vision Provider
        +--> API/Local Decision Provider
        `--> Replay/Evaluation
                     |
                     v
              Factory统一集成
                     |
                     v
              合同与fail-closed测试
```

合同阶段不能并行；Provider和replay阶段可以并行；Factory最终接线和全套验证由集成负责人串行完成。

## 17. 测试策略

每个已实现并启用的Brain模式必须通过同一组合同测试：

- 接收相同`EyeSample`；
- 返回合法`BrainResult`；
- 保留并核对measurement ID；
- 输出只包含有限数值；
- 低置信度或OOD返回`HOLD`；
- API超时和无效JSON返回`HOLD`；
- 不导入或构造硬件模块；
- 候选仍经过现有`SafetyPolicy`；
- readback不一致触发现有rollback路径；
- 模式不能在Campaign内静默改变。

replay比较至少记录：

- 眼图A/B排序准确率；
- 单点p50和p95延迟；
- 相同输入重复运行一致性；
- API失败率和无效输出率；
- 每点估算成本；
- 安全门拒绝率；
- 固定预算内的最佳安全质量分数。

## 18. 分阶段上线

```text
阶段0：FAKE/replay合同验证
阶段1：30至50点真实格式验证
阶段2：对本阶段已实现的Brain模式进行离线对照
阶段3：将离线结果合格的一个或多个模式放入影子模式
阶段4：本地评分为主，API只选择安全候选
阶段5：本地Decision替代API Decision
阶段6：视觉分数成为主目标，TDECQ/ER保持护栏
```

任何阶段都不允许模型获得硬件执行能力。

## 19. 当前设计决策

以下决策已经固定：

1. 所有模式使用相同`EyeSample`；
2. 模型Payload在`OptimizationBrain`内部生成；
3. 纯API使用一次融合调用；
4. 混合模式使用`ComposedBrain`；
5. 外部六步闭环和安全层不因模型变化而变化；
6. 模式在Campaign开始时选择并锁定；
7. Provider失败时fail-closed为`HOLD`；
8. 不静默跨Provider回退；
9. V2复用现有Optimizer、安全、回读、rollback和Campaign状态机；
10. 单一工作树并行采用合同先冻结、文件所有权隔离、单一集成负责人提交。

## 20. 验收标准

V2基础架构完成的最低标准是：

1. 至少一个首版目标模式和一个通用合同FAKE可以通过FAKE/replay运行，从而证明Brain可替换；
2. 六步主循环不包含Provider专用分支；
3. 在已安装并启用的模式之间切换只需要改变Campaign配置；
4. 所有已实现并启用的模式返回同一`BrainResult`合同；
5. API或本地模型失败时没有候选进入执行器；
6. Provider包没有硬件导入；
7. 现有V1.1 TDECQ路径和安全测试不回归；
8. 评估记录格式支持跨Provider比较；当存在两个及以上已启用或测试Provider时，同一Replay样本可以直接对照；
9. 未安装或未启用的模式明确报告不可用，且不会静默回退；
10. Dashboard继续只读展示候选、安全结果、回读和历史，不成为硬件控制入口。
