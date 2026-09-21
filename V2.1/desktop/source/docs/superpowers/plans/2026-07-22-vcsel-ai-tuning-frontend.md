# VCSEL AI Tuning Frontend Console Implementation Plan / VCSEL AI 调优前端控制台实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade React console for creating B0-based campaigns, approving stages, following ordered SSE events, displaying candidates/readback/ER/TDECQ, controlling pause/abort/rollback, and comparing final results. / 构建生产级 React 控制台，用于创建基于 B0 的 Campaign、批准阶段、跟踪有序 SSE 事件、展示候选与回读及 ER/TDECQ、控制暂停/中止/回滚并对比最终结果。

**Architecture:** A typed API client and resumable EventSource feed a feature-local campaign store. Pages render backend-owned state; mutation buttons call semantic campaign endpoints only. The browser never derives safety decisions or sends raw I²C/BERT/SCPI commands. / 类型化 API 客户端与可恢复 EventSource 驱动功能内 Campaign store；页面只渲染后端状态，操作按钮仅调用语义化 Campaign API。浏览器不推导安全判断，也不发送原始 I²C/BERT/SCPI 命令。

**Tech Stack:** React 18+, TypeScript strict mode, Vite, TanStack Query, Zod, Recharts, Vitest, React Testing Library, MSW, Playwright. If the target project already uses equivalent libraries, preserve its stack and map these feature boundaries one-to-one. / React 18+、TypeScript 严格模式、Vite、TanStack Query、Zod、Recharts、Vitest、React Testing Library、MSW、Playwright；若目标项目已有等价技术栈，保持其技术栈并逐一映射这些功能边界。

---

## Non-negotiable constraints / 不可违反的约束

- Read `docs/superpowers/specs/2026-07-22-vcsel-hierarchical-ai-tuning-design.md` or the supplied copy before implementation. / 实施前完整读取设计文档或其副本。
- V1 exposes `PER_STAGE`; `FULL_CAMPAIGN` is shown as unavailable, not silently enabled. / 第一版开放 `PER_STAGE`，`FULL_CAMPAIGN` 显示为不可用。
- Render B0 exactly: SSPRQ, FIR `(0,9,-28,104,-8,1,-3)`, NLC `6/6`, and nine I²C values. / 精确展示 B0。
- Never infer missing TDECQ as zero and never rank on a single best measurement. / 禁止把缺失 TDECQ 当作 0，禁止以单次最好值排名。
- No free-form raw hardware command input. / 不提供原始硬件命令输入。
- Emergency controls remain accessible while a mutating stage runs. / 写硬件阶段运行时紧急控制始终可用。

Use this exact immutable frontend fixture; never infer tap order from array position without labels. / 使用以下精确不可变前端 fixture；不得在没有标签时仅凭数组位置推断 tap 顺序。

```ts
export const B0 = {
  baselineId: "B0",
  pattern: "SSPRQ",
  fir: {
    pre3: 0, pre2: 9, pre1: -28, main: 104,
    post1: -8, post2: 1, post3: -3,
    upperNlc: 6, lowerNlc: 6,
  },
  i2c: {
    ibiasDac: 70, vref: 15, vbiasRef: 15,
    vgaCur: 13, resGain: 6, resBw: 0,
    gainCtrl: 5, efinCur: 15, eq: 22,
  },
} as const;
```

## File map / 文件结构

- Create `src/features/ai-tuning/types.ts`: UI domain types. / UI 领域类型。
- Create `schemas.ts`: Zod validation for every API payload and SSE event. / API 与 SSE 校验。
- Create `api.ts`: semantic REST client. / 语义 REST 客户端。
- Create `event-stream.ts`: ordered resumable SSE client. / 有序可恢复 SSE。
- Create `campaign-store.ts`: reducer/selectors, no hardware logic. / reducer 与 selector。
- Create `routes.tsx`: campaign feature routes. / 功能路由。
- Create pages `CampaignSetupPage.tsx`, `CampaignRunPage.tsx`, `CampaignResultPage.tsx`. / 三个页面。
- Create components `BaselineCard.tsx`, `StageTimeline.tsx`, `StageApprovalDialog.tsx`, `MetricTrendChart.tsx`, `ParameterDiffTable.tsx`, `CandidatePanel.tsx`, `HardwareStatus.tsx`, `EmergencyControls.tsx`, `OutcomeBanner.tsx`. / 组件。
- Add tests beside feature files plus `e2e/ai-tuning.spec.ts`. / 单元与端到端测试。

### Task 1: Define and validate the frontend contracts / 定义并验证前端契约

**Files:**
- Create: `src/features/ai-tuning/types.ts`
- Create: `src/features/ai-tuning/schemas.ts`
- Test: `src/features/ai-tuning/schemas.test.ts`

- [ ] **Step 1: Write failing schema tests for B0, campaign state, authorization, candidate, applied settings, observation, event, and result / 编写契约失败测试**

```ts
it("preserves missing TDECQ as null", () => {
  const value = observationSchema.parse({
    observationId: "o-1",
    stage: "I2C_COARSE",
    erDb: 2.7,
    erStatus: "CORR",
    tdecqDb: null,
    tdecqStatus: "INVALID",
    valid: true,
    feasible: false,
    settingsReadback: {},
  });
  expect(value.tdecqDb).toBeNull();
});
```

- [ ] **Step 2: Run `npm test -- schemas.test.ts` and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement discriminated unions for all normal and exceptional states and reject non-finite metrics / 实现状态联合类型并拒绝非有限指标**
- [ ] **Step 4: Add the exact B0 fixture and assert FIR order labels explicitly / 增加精确 B0 fixture 并显式断言 FIR 标签顺序**
- [ ] **Step 5: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 6: Commit with `git commit -m "feat: define AI tuning frontend contracts"` / 提交**

### Task 2: Build the REST client and resumable SSE stream / 构建 REST 客户端与可恢复 SSE

**Files:**
- Create: `src/features/ai-tuning/api.ts`
- Create: `src/features/ai-tuning/event-stream.ts`
- Test: `src/features/ai-tuning/api.test.ts`
- Test: `src/features/ai-tuning/event-stream.test.ts`

- [ ] **Step 1: Write failing MSW tests for campaign CRUD, preflight, plan, approve, pause, resume, abort, rollback, observations, best, and report / 编写 REST 失败测试**
- [ ] **Step 2: Write failing SSE tests for ordered event IDs, duplicate suppression, gaps, reconnect, terminal close, and invalid payload / 编写 SSE 失败测试**

```ts
it("ignores duplicate events and reconnects after the last sequence", async () => {
  const stream = createCampaignEventStream("c-1", { lastEventId: 7 });
  stream.feed({ id: 7, type: "STATE_CHANGED", payload: {} });
  stream.feed({ id: 8, type: "OBSERVATION_RECORDED", payload: {} });
  expect(stream.events.map((event) => event.id)).toEqual([8]);
  expect(stream.reconnectCursor).toBe(8);
});
```

- [ ] **Step 3: Run focused tests and confirm RED / 运行并确认失败**
- [ ] **Step 4: Implement one typed method per semantic API endpoint; validate every response with Zod / 每个语义 API 实现一个类型化方法并用 Zod 校验**
- [ ] **Step 5: Implement SSE with sequence checking, exponential reconnect capped at 10 seconds, and REST resync on a gap / 实现序号检查、指数重连与缺口 REST 重同步**
- [ ] **Step 6: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 7: Commit with `git commit -m "feat: connect AI tuning API and events"` / 提交**

### Task 3: Implement the campaign store and selectors / 实现 Campaign store 与 selectors

**Files:**
- Create: `src/features/ai-tuning/campaign-store.ts`
- Test: `src/features/ai-tuning/campaign-store.test.ts`

- [ ] **Step 1: Write failing reducer tests for state updates, readback, best metrics, authorization wait, pause, rollback, invalid observation, and reconnect replay / 编写 reducer 失败测试**

```ts
it("never replaces a valid TDECQ with an invalid null observation", () => {
  const before = stateWithLatestTdecq(2.3);
  const after = reduceCampaignEvent(before, invalidTdecqEvent());
  expect(after.latestObservation?.tdecqDb).toBeNull();
  expect(after.lastValidTdecqDb).toBe(2.3);
});
```

- [ ] **Step 2: Run focused tests and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement a pure reducer and selectors for current stage, progress, best feasible point, metric series, pending authorization, emergency-action availability, and outcome / 实现纯 reducer 与 selectors**
- [ ] **Step 4: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 5: Commit with `git commit -m "feat: track AI tuning campaign state"` / 提交**

### Task 4: Build campaign setup with immutable B0 preview / 构建带不可变 B0 预览的任务创建页

**Files:**
- Create: `src/features/ai-tuning/pages/CampaignSetupPage.tsx`
- Create: `src/features/ai-tuning/components/BaselineCard.tsx`
- Create: `src/features/ai-tuning/components/ParameterRangeEditor.tsx`
- Test: `src/features/ai-tuning/pages/CampaignSetupPage.test.tsx`

- [ ] **Step 1: Write failing UI tests for exact B0, parameter encoding versus tuning ranges, validation, budget, ER thresholds, and disabled full-campaign mode / 编写创建页失败测试**
- [ ] **Step 2: Run focused tests and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement a read-only B0 card and editable tuning envelopes; visually separate encoding range from AI authorization range / 实现 B0 卡片与调优包络编辑，并区分编码与授权范围**
- [ ] **Step 4: Require every enabled parameter to have min, max, max step, and a value containing B0; submit semantic JSON only / 要求启用参数完整配置且包含 B0**
- [ ] **Step 5: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 6: Commit with `git commit -m "feat: create B0 tuning campaigns"` / 提交**

### Task 5: Build live campaign monitoring / 构建实时 Campaign 监控页

**Files:**
- Create: `src/features/ai-tuning/pages/CampaignRunPage.tsx`
- Create: `src/features/ai-tuning/components/StageTimeline.tsx`
- Create: `src/features/ai-tuning/components/MetricTrendChart.tsx`
- Create: `src/features/ai-tuning/components/ParameterDiffTable.tsx`
- Create: `src/features/ai-tuning/components/CandidatePanel.tsx`
- Create: `src/features/ai-tuning/components/HardwareStatus.tsx`
- Test: `src/features/ai-tuning/pages/CampaignRunPage.test.tsx`

- [ ] **Step 1: Write failing UI tests for every normal/exception state, ER 2.5/3.0 reference lines, null TDECQ gaps, requested-versus-readback differences, budget, AI reason, and DCA progress / 编写运行页失败测试**

```tsx
it("renders invalid TDECQ as unavailable rather than zero", () => {
  render(<MetricTrendChart observations={[invalidTdecqObservation()]} />);
  expect(screen.getByText("TDECQ unavailable")).toBeInTheDocument();
  expect(screen.queryByText("0 dB")).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run focused tests and confirm RED / 运行并确认失败**
- [ ] **Step 3: Implement a sticky stage/status header, horizontal timeline, responsive charts, candidate explanation, readback diff, and ordered event log / 实现阶段头部、时间线、趋势图、候选解释、回读差异和事件日志**
- [ ] **Step 4: Use median-confirmed best metrics and label provisional single measurements explicitly / 使用中位数确认最佳值并标记单次临时值**
- [ ] **Step 5: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 6: Commit with `git commit -m "feat: monitor live AI tuning campaigns"` / 提交**

### Task 6: Implement exact stage authorization and emergency controls / 实现精确阶段授权与紧急控制

**Files:**
- Create: `src/features/ai-tuning/components/StageApprovalDialog.tsx`
- Create: `src/features/ai-tuning/components/EmergencyControls.tsx`
- Test: `src/features/ai-tuning/components/StageApprovalDialog.test.tsx`
- Test: `src/features/ai-tuning/components/EmergencyControls.test.tsx`

- [ ] **Step 1: Write failing tests that require stage, parameters, ranges, max steps, candidate/time budget, snapshot hash, rollback target, expiry, and complete authorization ID / 编写授权弹窗失败测试**
- [ ] **Step 2: Write failing tests for pause, abort, rollback confirmation, double-submit prevention, API failure, and `RECOVERY_REQUIRED` lockout / 编写紧急控制失败测试**
- [ ] **Step 3: Run focused tests and confirm RED / 运行并确认失败**
- [ ] **Step 4: Implement an explicit typed confirmation field matching the full authorization ID; do not use a generic OK dialog / 实现完整 ID 类型确认，禁止通用 OK 弹窗**
- [ ] **Step 5: Keep pause/abort visible during mutating states; rollback displays the exact target snapshot and never fabricates success / 保持紧急按钮可见并精确展示回滚目标**
- [ ] **Step 6: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 7: Commit with `git commit -m "feat: authorize and control tuning stages"` / 提交**

### Task 7: Build result comparison and export / 构建结果对比与导出

**Files:**
- Create: `src/features/ai-tuning/pages/CampaignResultPage.tsx`
- Create: `src/features/ai-tuning/components/OutcomeBanner.tsx`
- Create: `src/features/ai-tuning/components/ConfigurationComparison.tsx`
- Test: `src/features/ai-tuning/pages/CampaignResultPage.test.tsx`

- [ ] **Step 1: Write failing tests for `SUCCESS`, `PARTIAL_CONVERGENCE`, `NO_IMPROVEMENT`, `FAILED`, and `RECOVERY_REQUIRED` / 编写五种结果状态测试**
- [ ] **Step 2: Require B0-versus-final FIR/I²C diff, median/std/range, validity, convergence reason, unmet targets, writes, rollback history, and JSON/CSV/report links / 要求完整结果字段**
- [ ] **Step 3: Run focused tests and confirm RED / 运行并确认失败**
- [ ] **Step 4: Implement honest outcome banners: partial convergence must state that ER 3 dB was not reached / 实现真实状态横幅，部分收敛必须说明未达到 ER 3 dB**
- [ ] **Step 5: Re-run and confirm GREEN / 重新运行并确认通过**
- [ ] **Step 6: Commit with `git commit -m "feat: compare AI tuning outcomes"` / 提交**

### Task 8: Accessibility, responsive layout, and end-to-end verification / 无障碍、响应式与端到端验证

**Files:**
- Create: `e2e/ai-tuning.spec.ts`
- Modify: feature styles and application routes
- Modify: project README documentation

- [ ] **Step 1: Add an MSW/fixture campaign covering invalid TDECQ, stage approval, coarse promotion, fine tuning, rejected FIR, rollback, and final partial/success result / 增加完整 fixture**
- [ ] **Step 2: Write Playwright flow for create -> preflight -> approve stages -> pause/resume -> result/export / 编写 Playwright 流程**
- [ ] **Step 3: Verify keyboard navigation, focus trap, chart text alternatives, color-independent status, 360px width, desktop width, and reconnect banner / 验证无障碍与响应式**
- [ ] **Step 4: Run unit tests / 运行单元测试**

Run: `npm test -- --run`
Expected: all collected frontend tests pass. / 预期：全部通过。

- [ ] **Step 5: Run typecheck and production build / 运行类型检查与生产构建**

Run: `npm run typecheck && npm run build`
Expected: both commands exit 0. / 预期：均成功退出。

- [ ] **Step 6: Run Playwright against the mocked API / 使用 mock API 运行 Playwright**

Run: `npx playwright test e2e/ai-tuning.spec.ts`
Expected: the full flow passes without contacting real hardware. / 预期：全流程通过且不接触真实硬件。

- [ ] **Step 7: Run lint and diff checks, update bilingual usage documentation, and commit / 运行 lint、差异检查、更新双语文档并提交**

```text
npm run lint
git diff --check
git add src/features/ai-tuning e2e/ai-tuning.spec.ts README.md
git commit -m "test: verify AI tuning console"
```

## Delivery report / 交付报告

Report framework assumptions, changed files, screenshots at desktop and 360px, exact test/build results, mocked campaign walkthrough, API contract mismatches, and accessibility findings. Do not connect to live hardware, expose raw command fields, push, or create a pull request unless explicitly requested. / 报告框架假设、文件、桌面与 360px 截图、测试/构建结果、mock 流程、API 契约差异和无障碍结果；未经明确要求不得连接真实硬件、暴露原始命令、push 或创建 PR。
