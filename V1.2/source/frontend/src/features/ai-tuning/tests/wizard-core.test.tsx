import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { App } from "../../../app/App";
import { server } from "../../../mocks/server";
import {
  canContinueMapping,
  isValidSettleTime,
  initialWizardState,
  supportedWizardSweepHardware,
  wizardReducer,
} from "../store/wizardReducer";
import { deriveV11WizardGate } from "../hooks/useV11RuntimeGate";
import { StagedLiveAuthority } from "../components/CapabilityPanel";
import {
  buildV11AuthorizationRequest,
  buildV11CreateTaskRequest,
  isCurrentSnapshotApproved,
} from "../store/wizardWorkflow";

describe("V1.1 wizard core", () => {
  beforeEach(() => window.history.pushState({}, "", "/"));

  it("shows the complete staged LIVE authority before authorization", () => {
    render(<StagedLiveAuthority capability={{
      product_version: "1.2.0",
      schema_version: 2,
      runtime_mode: "LIVE",
      campaign_types: ["AI_OPTIMIZATION"],
      dynamic_b0: true,
      full_campaign_authorization: true,
      hardware_mutation: true,
      fake: false,
      live_ai_profile: "PILOT_3_THEN_FAST24",
      candidate_budget: 27,
      pilot_candidate_budget: 3,
      fast_candidate_budget: 24,
      coarse_candidate_budget: 6,
      fine_candidate_budget: 16,
      fir_first_candidate_budget: 1,
      fir_iterative_candidate_budget: 1,
      measurement_timeout_s: 60,
      cleanup_grace_s: 5,
      dataset_required: true,
      fir_authority: "FORMULA_ONLY_MAIN_NLC_PATTERN_FROZEN_L1_170",
      final_restore: "B0_VERIFIED",
      profile_options: [{ name: "PILOT_3_THEN_FAST24", candidate_budget: 27, max_alternating_rounds: 2 }],
    }} />);

    expect(screen.getByText("PILOT_3 → FAST24")).toBeVisible();
    expect(screen.getByText(/69, 71, 73/)).toBeVisible();
    expect(screen.getByText(/6 coarse · 16 fine · 1 first FIR · 1 iterative FIR/)).toBeVisible();
    expect(screen.getByText(/Main, NLC and pattern remain frozen/)).toBeVisible();
    expect(screen.getByText(/Final hardware state: B0/)).toBeVisible();
  });

  it("selects Sweep and clears Sweep-only state when returning to Optimization", () => {
    const sweep = wizardReducer(initialWizardState, { type: "SELECT_TASK_TYPE", value: "SWEEP" });
    expect(sweep.taskType).toBe("SWEEP");
    const configured = { ...sweep, sweepMode: "FULL" as const, sweepSeed: 42 };
    const optimization = wizardReducer(configured, { type: "SELECT_TASK_TYPE", value: "AI_OPTIMIZATION" });
    expect(optimization).toMatchObject({ taskType: "AI_OPTIMIZATION", sweepMode: null, sweepSeed: null });
  });

  it("uses an explicit 0.1 second Sweep settle default and rejects invalid values", () => {
    expect(initialWizardState.settleTimeS).toBe(0.1);
    expect(initialWizardState.saveDataset).toBe(false);
    expect(isValidSettleTime(0)).toBe(true);
    expect(isValidSettleTime(0.1)).toBe(true);
    expect(isValidSettleTime(-0.1)).toBe(false);
    expect(isValidSettleTime(Number.NaN)).toBe(false);
    expect(isValidSettleTime(Number.POSITIVE_INFINITY)).toBe(false);
  });

  it("requires explicit same-link confirmation", () => {
    const mapping = {
      i2c_channel: 0,
      bert_port: "port1",
      bert_channel: "0",
      dca_tdecq_source: "FUNC1" as const,
      dca_tdecq_equalizer_function: 1 as const,
      dca_outer_er_source: "CHAN1A" as const,
      operator_confirmed: false,
    };
    expect(canContinueMapping(mapping)).toBe(false);
    expect(canContinueMapping({ ...mapping, operator_confirmed: true })).toBe(true);
  });

  it("replaces selected parameters with the server-available scan subset", () => {
    const scanned = wizardReducer(initialWizardState, { type: "SET_PARAMETERS", value: ["eq"] });
    expect(scanned.selectedParameters).toEqual(["eq"]);
    expect(scanned.selectedParameters).not.toContain("ibias_dac");
  });

  it("builds the complete SWEEP request from pre-create wizard evidence", () => {
    const request = buildV11CreateTaskRequest({
      taskId: "task-011",
      taskType: "SWEEP",
      channelMapping: {
        i2c_channel: 0,
        bert_port: "port1",
        bert_channel: "0",
        dca_tdecq_source: "FUNC1",
        dca_tdecq_equalizer_function: 1,
        dca_outer_er_source: "CHAN1A",
      },
      mappingConfirmed: true,
      parameters: [{ name: "eq", min: 18, max: 22, step: 2 }],
      depth: "STANDARD",
      candidateBudget: 3,
      sweepHardware: "I2C",
      sweepMode: "SINGLE",
      sweepSeed: 1042,
      maxPoints: 3,
      settleTimeS: 0.1,
      saveDataset: true,
    });

    expect(request).toEqual({
      task_id: "task-011",
      task_type: "SWEEP",
      channel_mapping: {
        i2c_channel: 0,
        bert_port: "port1",
        bert_channel: "0",
        dca_tdecq_source: "FUNC1",
        dca_tdecq_equalizer_function: 1,
        dca_outer_er_source: "CHAN1A",
        operator_confirmed: true,
      },
      parameters: [{ name: "eq", min: 18, max: 22, step: 2 }],
      depth: "STANDARD",
      candidate_budget: 3,
      sweep_hardware: "I2C_SWEEP",
      sweep_mode: "SINGLE_PARAMETER",
      seed: 1042,
      max_points: 3,
      settle_time_s: 0.1,
      save_dataset: true,
    });
  });

  it("sends an explicit disabled dataset choice for AI Optimization", () => {
    const request = buildV11CreateTaskRequest({
      taskId: "task-ai-no-dataset",
      taskType: "AI_OPTIMIZATION",
      channelMapping: {
        i2c_channel: 0,
        bert_port: "port1",
        bert_channel: "0",
        dca_tdecq_source: "FUNC1",
        dca_tdecq_equalizer_function: 1,
        dca_outer_er_source: "CHAN1A",
      },
      mappingConfirmed: true,
      parameters: [{ name: "eq", min: 18, max: 22, step: 2 }],
      depth: "STANDARD",
      candidateBudget: 3,
      sweepHardware: "I2C",
      sweepMode: null,
      sweepSeed: null,
      maxPoints: 3,
      settleTimeS: 0.1,
      saveDataset: false,
    });

    expect(request.save_dataset).toBe(false);
  });

  it("builds the frozen 27-round staged LIVE request with Dataset enabled", () => {
    const request = buildV11CreateTaskRequest({
      taskId: "native-v12-staged",
      taskType: "AI_OPTIMIZATION",
      channelMapping: {
        i2c_channel: 0,
        bert_port: "port1",
        bert_channel: "0",
        dca_tdecq_source: "FUNC1",
        dca_tdecq_equalizer_function: 1,
        dca_outer_er_source: "CHAN1A",
      },
      mappingConfirmed: true,
      parameters: [
        { name: "ibias_dac", min: 60, max: 80, step: 1 },
        { name: "res_gain", min: 0, max: 10, step: 1 },
      ],
      depth: "PILOT_3_THEN_FAST24",
      candidateBudget: 27,
      sweepHardware: "I2C",
      sweepMode: null,
      sweepSeed: 17,
      maxPoints: 27,
      settleTimeS: 0.1,
      saveDataset: true,
    });

    expect(request).toMatchObject({
      live_ai_profile: "PILOT_3_THEN_FAST24",
      optimization_scope: "I2C_FIR_ALTERNATING",
      depth: "FAST",
      candidate_budget: 27,
      pilot_candidate_budget: 3,
      fast_candidate_budget: 24,
      coarse_candidate_budget: 6,
      fine_candidate_budget: 16,
      fir_first_candidate_budget: 1,
      fir_iterative_candidate_budget: 1,
      settle_time_s: 0.1,
      save_dataset: true,
    });
  });

  it.each([-0.1, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects an invalid settle time before creating a task: %s",
    (settleTimeS) => {
      expect(() => buildV11CreateTaskRequest({
        taskId: "task-invalid-settle",
        taskType: "SWEEP",
        channelMapping: {
          i2c_channel: 0,
          bert_port: "port1",
          bert_channel: "0",
          dca_tdecq_source: "FUNC1",
          dca_tdecq_equalizer_function: 1,
          dca_outer_er_source: "CHAN1A",
        },
        mappingConfirmed: true,
        parameters: [{ name: "eq", min: 18, max: 22, step: 2 }],
        depth: "STANDARD",
        candidateBudget: 3,
        sweepHardware: "I2C",
        sweepMode: "SINGLE",
        sweepSeed: 1042,
        maxPoints: 3,
        settleTimeS,
        saveDataset: false,
      })).toThrow("settle_time_s must be a finite nonnegative number");
    },
  );

  it("creates a strict authorization request with a 30 minute TTL", () => {
    const issuedAt = new Date("2026-07-29T10:00:00.000Z");
    expect(buildV11AuthorizationRequest("task-011", issuedAt)).toEqual({
      authorization_id: "auth-task-011-1785319200000",
      expires_at: "2026-07-29T10:30:00.000Z",
      operator_confirmed: true,
    });
  });

  it("shows and locks the selected dataset choice in the Sweep review payload", async () => {
    const user = userEvent.setup();
    let createdRequest: Record<string, unknown> | null = null;
    let receivedSettleTime: unknown;
    const taskResponse = (state: "DRAFT" | "B0_FROZEN") => ({
      task_id: String(createdRequest?.task_id),
      task_type: "SWEEP",
      state,
      runtime_mode: "FAKE",
      product_version: "1.1.0",
      schema_version: 2,
      b0_hash: state === "DRAFT" ? null : "b".repeat(64),
      configuration_fingerprint: "c".repeat(64),
      config: createdRequest,
      measurement_status: "PENDING",
      dataset_status: createdRequest?.save_dataset ? "PENDING" : "DISABLED",
      dataset_error: null,
      dataset_files: [],
      created_at: "2026-08-04T10:00:00Z",
      updated_at: "2026-08-04T10:00:00Z",
    });
    server.use(
      http.post("/api/v1.1/tasks", async ({ request }) => {
        createdRequest = await request.json() as Record<string, unknown>;
        receivedSettleTime = createdRequest.settle_time_s;
        return HttpResponse.json(taskResponse("DRAFT"), { status: 201 });
      }),
      http.post("/api/v1.1/tasks/:taskId/b0", () => HttpResponse.json(taskResponse("B0_FROZEN"))),
    );
    window.history.pushState({}, "", "/tasks/new?type=SWEEP");
    render(<App />);

    await user.click(await screen.findByRole("button", { name: /Continue/i }));
    await user.click(await screen.findByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: /Continue/i }));
    await user.click(await screen.findByRole("button", { name: /Approve current snapshot as B0/i }));
    const readback = screen.getByLabelText("Current FIR readback");
    expect(within(readback).getByText("-28")).toBeVisible();
    expect(screen.getByText(/Reference only \(not B0\)/i)).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Continue/i }));
    await user.click(await screen.findByRole("button", { name: /Scan server mapping/i }));
    await user.click(screen.getByRole("button", { name: /Continue/i }));
    await user.click(await screen.findByRole("button", { name: /Continue/i }));

    const datasetSwitch = await screen.findByRole("checkbox", { name: /Save dataset/i });
    expect(datasetSwitch).not.toBeChecked();
    expect(screen.getByText(/不保存眼图密度矩阵和 PNG/)).toBeVisible();
    await user.click(datasetSwitch);
    expect(screen.getByText(/TDECQ 均衡眼图 PNG/)).toBeVisible();
    await user.click(await screen.findByRole("button", { name: /Continue/i }));

    const settleInput = await screen.findByRole("spinbutton", { name: "Settle time (seconds)" });
    expect(settleInput).toHaveValue(0.1);
    expect(screen.getByText(/Default: 0.1 s/)).toBeVisible();
    await user.clear(settleInput);
    await user.type(settleInput, "0.1");
    await user.click(screen.getByRole("button", { name: /Continue/i }));

    expect(await screen.findByText(/settle 0.1 s/)).toBeVisible();
    expect(screen.getByText(/Dataset saving/i).nextElementSibling).toHaveTextContent("ENABLED · LOCKED");
    expect(receivedSettleTime).toBe(0.1);
    expect(createdRequest).toMatchObject({ save_dataset: true });
  });

  it("invalidates B0 approval when a different snapshot is captured", () => {
    expect(isCurrentSnapshotApproved("snapshot-1", "snapshot-1")).toBe(true);
    expect(isCurrentSnapshotApproved("snapshot-1", "snapshot-2")).toBe(false);
    expect(isCurrentSnapshotApproved(null, "snapshot-2")).toBe(false);
  });

  it("uses V1.1 capabilities and permits start only for the owned task", () => {
    const capability = {
      product_version: "1.1.0",
      schema_version: 2,
      runtime_mode: "FAKE" as const,
      campaign_types: ["AI_OPTIMIZATION" as const, "SWEEP" as const],
      dynamic_b0: true,
      full_campaign_authorization: true,
      hardware_mutation: false,
      fake: true,
      profile_options: [
        { name: "STANDARD" as const, candidate_budget: 64, max_alternating_rounds: 3 },
      ],
    };
    const owned = {
      task_id: "task-011",
      state: "B0_FROZEN" as const,
    };

    expect(deriveV11WizardGate(capability, null, null)).toMatchObject({
      canMutate: true,
      canCreateDraft: true,
      canStart: false,
    });
    expect(deriveV11WizardGate(capability, owned, "task-011")).toMatchObject({
      canMutate: true,
      canCreateDraft: false,
      canStart: true,
    });
    expect(deriveV11WizardGate(capability, owned, "different-task")).toMatchObject({
      canCreateDraft: false,
      canStart: false,
    });
  });

  it("fails closed to I2C Sweep until FIR tap parameters are supported", () => {
    expect(supportedWizardSweepHardware).toEqual(["I2C"]);
    expect(supportedWizardSweepHardware).not.toContain("BERT_FIR");
  });

  it("presents V1.1 and interlocks task creation while a campaign is active", async () => {
    render(<App />);
    expect(await screen.findByText("AI Tuning V1.2")).toBeVisible();
    expect(screen.getByRole("link", { name: /View current task/i })).toBeVisible();
    expect(screen.queryByRole("link", { name: /Create task/i })).not.toBeInTheDocument();
  });
});
