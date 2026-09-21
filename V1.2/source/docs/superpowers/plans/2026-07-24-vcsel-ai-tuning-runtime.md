# VCSEL AI Tuning Runtime Implementation Plan / VCSEL AI 调优运行层实施计划

> **For agentic workers / 面向执行者：** REQUIRED SUB-SKILL: use `test-driven-development` while implementing every behavior. Each production change must follow a witnessed RED → GREEN cycle. / 必须使用 `test-driven-development`；每个生产行为都要先观察测试失败，再以最小实现转绿。

**Goal / 目标：** Assemble the existing FastAPI, campaign persistence/state machine, optimizers, and hardware adapter contracts into a safe, restartable production runtime with FAKE, READ_ONLY, and LIVE_TUNING modes. / 将现有 FastAPI、Campaign 持久化与状态机、优化器及硬件 Adapter 契约组装为安全、可重启、支持 FAKE、READ_ONLY 与 LIVE_TUNING 的生产运行层。

**Architecture / 架构：** `RuntimeConfig` strictly validates startup inputs without importing hardware. `CampaignRuntime` owns the store, repositories, dependency factories, workers, campaign registry, campaign and hardware leases, and shutdown. `ConcreteCampaignService` remains a thin application layer over `CampaignOrchestrator`; only factories construct adapters, and only the orchestrator invokes them. / `RuntimeConfig` 严格校验启动配置且不导入硬件；`CampaignRuntime` 管理 Store、Repository、依赖工厂、worker、Campaign registry、Campaign/硬件租约和关闭；`ConcreteCampaignService` 仅作为 `CampaignOrchestrator` 的薄应用层，只有工厂构造 Adapter，只有 Orchestrator 调用 Adapter。

**Tech Stack / 技术栈：** Python 3.12, Pydantic 2, FastAPI, SQLite, `concurrent.futures`, pytest, fake/replay adapters. / Python 3.12、Pydantic 2、FastAPI、SQLite、`concurrent.futures`、pytest、fake/replay Adapter。

---

### Task 1: Strict runtime configuration / 严格运行配置

**Files / 文件：**

- Create / 新建: `src/vcsel_ai_tuning/runtime_config.py`
- Create / 新建: `tests_py/ai_tuning/test_runtime_config.py`
- Create / 新建: `configs/ai_tuning/runtime.example.yaml`

- [ ] Write tests for default `FAKE`, forbidden unknown fields, bounded host/port/timeouts/workers, required explicit DCA/I2C/BERT endpoints in READ_ONLY and LIVE_TUNING, and stable B0 loading. / 为默认 `FAKE`、禁止未知字段、host/port/timeout/worker 边界、READ_ONLY/LIVE_TUNING 必须显式提供 DCA/I2C/BERT 端点及稳定 B0 加载编写测试。
- [ ] Run `pytest tests_py/ai_tuning/test_runtime_config.py -q` and confirm collection/import failure because `runtime_config` does not exist. / 运行测试并确认因模块不存在而按预期失败。
- [ ] Implement strict frozen Pydantic models: `RuntimeMode`, `ApiConfig`, `StorageConfig`, `HardwareConfig`, and `RuntimeConfig`, plus `load_runtime_config(path)`. Hardware endpoints stay optional only in FAKE. / 实现严格冻结的 Pydantic 模型及加载函数；硬件端点只允许在 FAKE 中为空。
- [ ] Run the focused test and `tests_py/ai_tuning/test_config.py`; both must pass. / 运行聚焦测试及现有配置测试并确保通过。

### Task 2: Persistent campaign metadata and pagination / Campaign 元数据持久化与分页

**Files / 文件：**

- Modify / 修改: `src/vcsel_ai_tuning/campaign/store.py`
- Create / 新建: `tests_py/ai_tuning/campaign/test_runtime_store.py`

- [ ] Write tests that bootstrap immutable campaign metadata atomically with DRAFT state, return one campaign, list stable summaries by `created_at/campaign_id`, paginate with validated `limit/offset`, and reject duplicate conflicting metadata. / 测试原子创建不可变 Campaign 元数据与 DRAFT 状态、单项读取、稳定排序分页及冲突重复创建拒绝。
- [ ] Run the focused tests and confirm the new methods are absent. / 运行并确认新方法缺失导致预期失败。
- [ ] Extend the SQLite schema with `campaign_metadata`; add typed `CampaignRecord`, `create_campaign`, `get_campaign`, and `list_campaigns` without changing existing state/event semantics. / 增加元数据表和强类型读取/分页方法，不改变现有状态与事件语义。
- [ ] Run focused store tests plus `campaign/test_store.py`. / 运行新旧 Store 测试。

### Task 3: Optimizer and adapter factories with capability fencing / 优化器与 Adapter 工厂及能力隔离

**Files / 文件：**

- Create / 新建: `src/vcsel_ai_tuning/runtime.py`
- Create / 新建: `tests_py/ai_tuning/test_runtime_factories.py`

- [ ] Write tests for explicit `PATH_EXHAUSTIVE`, `PATH_TDECQ_BAYESIAN`, and rejected `PATH_HIERARCHICAL`; deterministic fake adapters and fake FIR proposer; absent production FIR proposer; READ_ONLY write rejection with code `READ_ONLY_MODE`; and module import causing no socket/VISA/serial construction. / 测试优化路径显式区分、fake 适配器、生产 FIR 缺失、READ_ONLY 写入拒绝及 import 不连接硬件。
- [ ] Confirm RED because the runtime factories do not exist. / 确认因工厂不存在而失败。
- [ ] Implement dependency-injected factories and `RuntimeCapabilities`. Do not import `vcsel_ai_tuning.hardware` at module import time; LIVE_TUNING resolves real builders only inside the explicitly invoked factory. / 实现注入式工厂与能力模型；模块加载时不得导入硬件，LIVE_TUNING 只在显式调用工厂时延迟解析真实构造器。
- [ ] Run focused tests and optimizer import-isolation tests. / 运行聚焦测试及 optimizer import isolation 测试。

### Task 4: Campaign runtime registry, leases, recovery, and shutdown / Campaign registry、租约、恢复与关闭

**Files / 文件：**

- Modify / 修改: `src/vcsel_ai_tuning/runtime.py`
- Create / 新建: `tests_py/ai_tuning/test_runtime_registry.py`

- [ ] Write tests for one cached orchestrator dependency bundle per campaign, persisted execution leases, a process-wide hardware-resource lease across campaigns, restart fencing of active states to RECOVERY_REQUIRED, no auto-resume, bounded worker count, worker exception event persistence, and idempotent resource shutdown. / 测试每个 Campaign 单一依赖包、执行租约、跨 Campaign 硬件资源独占、重启恢复隔离、受控 worker、异常事件及幂等关闭。
- [ ] Confirm RED on missing registry behavior. / 确认 registry 行为缺失导致失败。
- [ ] Implement `CampaignRuntime`, `CampaignEntry`, `HardwareResourceRegistry`, `submit_stage`, `recover`, and `close`. Release hardware leases in `finally`; close repositories/adapters/store during shutdown. / 实现运行时、资源 registry、后台提交、恢复和关闭；始终在 `finally` 释放硬件租约。
- [ ] Run focused tests and existing Store/Repository recovery tests. / 运行聚焦测试及现有恢复测试。

### Task 5: ConcreteCampaignService / 生产 CampaignService

**Files / 文件：**

- Create / 新建: `src/vcsel_ai_tuning/campaign/service.py`
- Modify / 修改: `src/vcsel_ai_tuning/campaign/__init__.py`
- Create / 新建: `tests_py/ai_tuning/campaign/test_service.py`

- [ ] Write tests for every `CampaignService` method, including idempotent create/preflight/approve/control mutations, stale authorization rejection, FULL_CAMPAIGN rejection, READ_ONLY approval rejection, FIR/hierarchical capability failure, paginated summaries, best-observation median policy, and reports. / 覆盖 Protocol 全部方法、mutation 幂等、过期授权、FULL_CAMPAIGN、READ_ONLY、能力 fail-closed、分页、最佳观测与报告。
- [ ] Confirm RED because `ConcreteCampaignService` is absent. / 确认实现缺失导致失败。
- [ ] Implement the thin service: delegate state changes and hardware behavior to the orchestrator/runtime; persist idempotency evidence as events; dispatch long stages to bounded workers; never call an adapter directly. / 实现薄 Service；状态与硬件行为委托给 Orchestrator/runtime；长阶段交给受控 worker；绝不直接调用 Adapter。
- [ ] Run focused service, authorization, state-machine, and orchestrator tests. / 运行新 Service 与现有授权、状态机、Orchestrator 测试。

### Task 6: Health, capabilities, and campaign list API / 健康、能力与 Campaign 列表 API

**Files / 文件：**

- Create / 新建: `src/vcsel_ai_tuning/api/health.py`
- Modify / 修改: `src/vcsel_ai_tuning/api/routes.py`
- Modify / 修改: `src/vcsel_ai_tuning/api/app.py`
- Modify / 修改: `src/vcsel_ai_tuning/api/__init__.py`
- Create / 新建: `tests_py/ai_tuning/api/test_runtime_routes.py`

- [ ] Write HTTP tests for `GET /health`, `GET /capabilities`, and paginated `GET /api/campaigns`; assert responses omit endpoints, ports, passwords, and resource strings. / 编写三个只读接口的 HTTP 测试，并断言不泄露端点、端口、密码或资源串。
- [ ] Confirm RED with 404 or missing runtime integration. / 确认因接口缺失而失败。
- [ ] Add a runtime-status protocol and read-only routes. Keep campaign routes thin and change app lifespan to call runtime shutdown without constructing hardware on import. / 增加运行状态协议和只读接口；使用 lifespan 关闭 runtime，导入时不构造硬件。
- [ ] Run focused API tests and all existing route/SSE tests. / 运行新接口及现有 route/SSE 测试。

### Task 7: Built-in FAKE runtime and real HTTP end-to-end / 内置 FAKE runtime 与真实 HTTP 端到端

**Files / 文件：**

- Modify / 修改: `src/vcsel_ai_tuning/runtime.py`
- Modify / 修改: `src/vcsel_ai_tuning/api/app.py`
- Modify / 修改: `README.md`
- Create / 新建: `tests_py/ai_tuning/api/test_fake_runtime_e2e.py`

- [ ] Write an HTTP-level test using a real localhost uvicorn socket and the built-in FAKE runtime: create → preflight → plan → approve → background fake execution → observations → SSE with Last-Event-ID → report. Patch real hardware constructors to fail if referenced. / 使用真实 localhost uvicorn socket 测试完整 fake 流程，并让任何真实硬件构造引用立即失败。
- [ ] Confirm RED because built-in startup is not available. / 确认内置启动缺失导致失败。
- [ ] Implement `create_fake_runtime`, config-driven `create_runtime`, default FAKE startup, and bilingual README PowerShell commands. Keep environment factory injection compatible. / 实现内置 fake/runtime 工厂、默认 FAKE 启动及双语 README 命令，并保持现有环境工厂兼容。
- [ ] Run the focused E2E and full API suite. / 运行聚焦 E2E 与完整 API 测试。

### Task 8: Full verification and commit / 完整验收与提交

**Files / 文件：** all task-related files only / 仅本计划相关文件。

- [ ] Run `./.venv-ai-tuning/Scripts/python.exe -m pytest tests_py/ai_tuning -q` and record passed, skipped, and warning counts. / 运行完整测试并记录通过、跳过和 warning 数。
- [ ] Run `git diff --check`, inspect `git diff --stat`, and scan optimizer imports plus route-to-adapter calls. / 运行 diff 检查并复核安全边界。
- [ ] Start the FAKE API on an ephemeral localhost port, call `/health` and `/capabilities`, then shut it down; no hardware endpoint may be contacted. / 在临时本地端口启动 FAKE API，访问健康与能力接口后关闭，禁止访问硬件。
- [ ] Stage only exact task files, inspect `git diff --cached`, and commit once on `codex/vcsel-ai-tuning-runtime`. / 仅精确暂存本任务文件，复核后在功能分支提交一次。
- [ ] Report the commit hash, PowerShell launch command, example config path, verification counts, and remaining hierarchical optimizer, production FIR proposer, and real-hardware acceptance work. / 报告提交哈希、启动命令、示例配置、验收计数与三类剩余工作。
