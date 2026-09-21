# V2.1 Kimi 多模态最小接入设计

日期：2026-08-09
状态：用户已确认真实 I2C 方向，等待书面复核

## 目标

在 V2.0 现有 `PURE_API` 入口上接入 Kimi K3，让它读取眼图和用户指示，从本地提供的候选中选择下一步参数。

Kimi 给出合法建议后，不等待人工逐步确认，直接调用 V2.0 已有 `run_one_candidate()`：先执行现有范围、步长和授权检查，通过后真实写入 I2C，硬件回读，再由 DCA 重新测量。

Kimi 不能直接调用寄存器写入器。它只能返回候选编号，真实参数由本地候选表提供，写入权只属于 V2.0 执行链。

## 最小链路

```text
EyeSample + 初始指示 + safe_candidates
-> 现有 NPY 眼图预处理
-> PNG/Base64
-> Kimi K3
-> 结构化 JSON
-> 本地检查 candidate_id
-> V2.0 validate_candidate()
-> V2.0 run_one_candidate()
-> 真实 I2C 写入与硬件回读
-> DCA 重新测量
-> v21_result.json
```

## 输入

第一次运行需要：

- 一份有效 `EyeSample`；
- 对应的 `raw_eye_density.npy`；
- 用户初始调参指示；
- V1.1 或人工准备的安全候选列表。

第一轮 Kimi 读取初始眼图和初始指示。后续轮次发送最新眼图、相同指示和上一轮结果，不重复发送长历史。

完整 DCA 屏幕和带 TDECQ/ER 数字的截图不作为模型输入。

## 最少新增四个文件

```text
src/vcsel_ai_tuning/v2/providers/kimi_client.py
src/vcsel_ai_tuning/v2/providers/kimi_provider.py
src/vcsel_ai_tuning/v2/kimi_step.py
src/vcsel_ai_tuning/hardware/live_runtime_factory.py
```

- `kimi_client.py`：读取环境变量中的 Key，调用 Moonshot API。
- `kimi_provider.py`：制作多模态 Payload，把 Kimi 回答转换成现有 `PerceptionResult`和 `DecisionProposal`。
- `kimi_step.py`：接收样本、指示和候选；Kimi 选择合法候选后，调用现有 `run_one_candidate()`，保存一个结果文件。
- `live_runtime_factory.py`：把已有 `CampaignI2cAdapter`、`Vd1124RegisterWriter`、DCA 和 BERT 适配器装成 V2.0 需要的真实 `AdapterBundle`。

继续复用现有 `preprocessing.py`、`MultimodalApiBrain`、`BrainFactory`和 `PURE_API`，不创建新的 Runner 体系。

## 当前代码的真实情况

V2.0 已有候选范围、单次步长、授权、硬件回读、回滚和 `RECOVERY_REQUIRED` 逻辑；仓库里也已有可写的 `CampaignI2cAdapter` 与 `Vd1124RegisterWriter`。

但默认 API 启动代码目前调用 `create_runtime(config)` 时没有提供真实 `hardware_adapter_factory`。所以现在直接切到 `LIVE_TUNING` 并不会自动接通 I2C，而是启动失败。V2.1 必须补上 `live_runtime_factory.py` 这根接线，之后才能真正复用 V2.0 的安全执行链。

## API 配置

```text
MOONSHOT_API_KEY=<环境变量>
base_url=https://api.moonshot.cn/v1
model=kimi-k3
reasoning_effort=low
```

API Key 不写入源码、配置或结果文件。使用 OpenAI-compatible Python SDK 只是为了少写 HTTP 代码，实际请求仍发往 Moonshot。

## Kimi 最小输出

```json
{
  "quality_score": 0.72,
  "confidence": 0.84,
  "valid": true,
  "ood": false,
  "action": "PROPOSE",
  "candidate_id": "candidate-002",
  "reason_codes": ["IMPROVE_MIDDLE_EYE"]
}
```

Kimi 不返回寄存器地址，也不直接执行 I2C。本地程序根据 `candidate_id` 复制候选中的真实参数，再交给 V2.0 执行。

## 只保留两个必要检查

1. Kimi 返回的 `candidate_id` 必须存在于输入候选表。
2. API 失败、JSON 无法解析或候选不存在时返回 `HOLD`，不执行硬件。

候选进入 V2.0 后，继续使用已有的范围、步长、授权、回读、回滚和 `RECOVERY_REQUIRED`，不在 V2.1 重复实现。

## 保存结果

每次只保存一个 `v21_result.json`，包含：

- `measurement_id`；
- 用户指示；
- 模型名称；
- 质量分数和置信度；
- `action`和 `candidate_id`；
- 本地映射后的参数；
- 简短原因；
- API耗时。

不保存 API Key 和 Base64 图片正文。

## 操作顺序

1. 安装 OpenAI-compatible SDK。
2. 在 PowerShell 设置 `MOONSHOT_API_KEY`。
3. 用一份真实历史眼图调用 Kimi。
4. 检查 `v21_result.json`是否得到合法候选。
5. 装配真实 `AdapterBundle`，以 `LIVE_TUNING` 启动。
6. 把合法候选交给现有 `run_one_candidate()`，真实写入 I2C。
7. 执行后回读硬件、DCA 重新测量，再把最新眼图交给 Kimi。

## 验收

- 使用真实 Kimi API 和一份真实眼图完成一次调用；
- Kimi 能同时收到眼图、初始指示和候选列表；
- 合法回答生成 `v21_result.json`；
- 非法回答变成 `HOLD`；
- 合法候选通过 V2.0 后能够真实写入 I2C，并完成回读与 DCA 重测；
- Kimi 本身没有 I2C、FIR、NLC 或 `setFir` 调用权限。

## 不做

- 不建设 Fake Provider、Fake 硬件或大规模 replay 测试；
- 不增加复杂哈希、审计对象和多份 JSONL；
- 第一版只执行一次“看眼图—选候选—真实写入—回读—重测”，暂不实现无人值守多轮循环；
- 不修改 Dashboard；
- 不训练视觉模型或 Decision 模型。
