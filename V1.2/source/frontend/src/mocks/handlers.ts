import { http, HttpResponse } from "msw";
import {
  campaignsFixture,
  capabilityFixture,
  eventFixtures,
  liveReportFixture,
  observationsFixture,
  resultObservationsFixture,
  resultReportFixture,
  runtimeFixture,
} from "../features/ai-tuning/fixtures/data";
import { v11SweepPointsResponseFixture } from "../features/ai-tuning/fixtures/v11SweepPoints";
import { v11DatasetFilesFixture } from "../features/ai-tuning/fixtures/v11Dataset";

function campaignFor(id: string) {
  if (id === "campaign-recovery") return { ...campaignsFixture[0], campaign_id: id, state: "RECOVERY_REQUIRED" };
  return campaignsFixture.find((campaign) => campaign.campaign_id === id) ?? {
    ...campaignsFixture[0],
    campaign_id: id,
    authorization_mode: "FULL_CAMPAIGN",
    state: "PREFLIGHT",
  };
}

function observationsFor(id: string) {
  return id === "campaign-result" ? resultObservationsFixture : observationsFixture;
}

const defaultV11Config = {
  task_id: "task-fake-011",
  task_type: "AI_OPTIMIZATION",
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
  candidate_budget: 32,
  seed: 0,
  save_dataset: false,
};
const v11Configs = new Map<string, typeof defaultV11Config & Record<string, unknown>>();
const fakeDatasetPreviewPng = Uint8Array.from(
  atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="),
  (character) => character.charCodeAt(0),
);

function v11Task(taskId: string, state: "DRAFT" | "B0_FROZEN" | "AUTHORIZED" | "RUNNING" | "PAUSED" | "ABORTED" | "ABORTED_RECOVERED" | "RECOVERY_REQUIRED") {
  const config = v11Configs.get(taskId) ?? { ...defaultV11Config, task_id: taskId };
  const datasetEnabled = config.save_dataset === true;
  const taskHasStarted = state !== "DRAFT" && state !== "B0_FROZEN" && state !== "AUTHORIZED";
  return {
    task_id: taskId,
    task_type: config.task_type,
    state,
    runtime_mode: "FAKE",
    product_version: "1.1.0",
    schema_version: 2,
    b0_hash: state === "DRAFT" ? null : "b".repeat(64),
    configuration_fingerprint: "c".repeat(64),
    config,
    measurement_status: taskHasStarted ? "SUCCESS" : "PENDING",
    dataset_status: datasetEnabled ? (taskHasStarted ? "SUCCESS" : "PENDING") : "DISABLED",
    dataset_error: null,
    dataset_files: datasetEnabled && taskHasStarted
      ? v11DatasetFilesFixture.map((file) => ({
          ...file,
          url: file.name === "record.json"
            ? `/api/v1.1/tasks/${taskId}/dataset/record.json`
            : file.url,
        }))
      : [],
    created_at: "2026-07-29T10:00:00Z",
    updated_at: "2026-07-29T10:00:00Z",
  };
}

const mappingParameters = ["ibias_dac", "vref", "vbias_ref", "vga_cur", "res_gain", "res_bw", "gain_ctrl", "efin_cur", "eq"].map((name, index) => ({
  name,
  status: "AVAILABLE",
  reason: null,
  script_order: index,
  current_value: name === "ibias_dac" ? 64 : name === "eq" ? 20 : 8,
  channel0_address: 8192 + index,
  channel_stride: 256,
  bit_start: 0,
  bits: 8,
  encoding_min: 0,
  encoding_max: 255,
}));

export const handlers = [
  http.get("/health", () => HttpResponse.json(runtimeFixture)),
  http.get("/capabilities", () => HttpResponse.json(capabilityFixture)),
  http.get("/artifacts/mock/v1.1/:fileName", ({ params }) => {
    const fileName = String(params.fileName);
    if (fileName.endsWith(".png")) {
      return new HttpResponse(fakeDatasetPreviewPng, {
        headers: { "Content-Type": "image/png", "X-VCSEL-Fixture": "FAKE" },
      });
    }
    return new HttpResponse("FAKE DATASET FIXTURE - NOT HARDWARE DATA\n", {
      headers: { "Content-Type": "application/octet-stream", "X-VCSEL-Fixture": "FAKE" },
    });
  }),
  http.get("/api/v1.1/tasks/:taskId/dataset/record.json", ({ params }) => HttpResponse.json({
    fixture: true,
    runtime_mode: "FAKE",
    task_id: String(params.taskId),
    notice: "Mock contract record; not hardware data.",
  }, { headers: { "X-VCSEL-Fixture": "FAKE" } })),
  http.get("/api/campaigns", ({ request }) => {
    const scenario = new URL(request.url).searchParams.get("scenario");
    if (scenario === "idle") return HttpResponse.json(campaignsFixture.filter((campaign) => campaign.state === "COMPLETED"));
    if (scenario === "recovery") return HttpResponse.json([{ ...campaignsFixture[0], campaign_id: "campaign-recovery", state: "RECOVERY_REQUIRED" }, ...campaignsFixture.slice(1)]);
    return HttpResponse.json(campaignsFixture);
  }),
  http.get("/api/campaigns/:campaignId", ({ params }) => {
    const campaign = campaignFor(String(params.campaignId));
    return campaign
      ? HttpResponse.json(campaign)
      : HttpResponse.json({ detail: { code: "campaign_not_found" } }, { status: 404 });
  }),
  http.get("/api/campaigns/:campaignId/observations", ({ params }) =>
    HttpResponse.json(observationsFor(String(params.campaignId))),
  ),
  http.get("/api/campaigns/:campaignId/best", ({ params }) => {
    const observations = observationsFor(String(params.campaignId));
    return HttpResponse.json(observations.at(-1) ?? null);
  }),
  http.get("/api/campaigns/:campaignId/report", ({ params }) =>
    HttpResponse.json(String(params.campaignId) === "campaign-result" ? resultReportFixture : liveReportFixture),
  ),
  http.get("/api/campaigns/:campaignId/events/stream", () => {
    const body = eventFixtures.map((event) => `id: ${event.sequence}\nevent: ${event.event_type}\ndata: ${JSON.stringify(event)}\n\n`).join("");
    return new HttpResponse(body, { headers: { "Content-Type": "text/event-stream" } });
  }),
  http.post("/api/campaigns/:campaignId/pause", ({ params }) =>
    HttpResponse.json({ campaign_id: String(params.campaignId), state: "PAUSED", simulated: true }),
  ),
  http.post("/api/campaigns/:campaignId/resume", ({ params }) =>
    HttpResponse.json({ campaign_id: String(params.campaignId), state: "I2C_COARSE", simulated: true }),
  ),
  http.post("/api/campaigns/:campaignId/abort", ({ params }) =>
    HttpResponse.json({ campaign_id: String(params.campaignId), state: "ROLLING_BACK", simulated: true }),
  ),

  http.get("/api/v1.1/runtime/capabilities", () => HttpResponse.json({
    product_version: "1.1.0",
    schema_version: 2,
    runtime_mode: "FAKE",
    campaign_types: ["AI_OPTIMIZATION", "SWEEP"],
    dynamic_b0: true,
    full_campaign_authorization: true,
    hardware_mutation: false,
    fake: true,
    sweep_execution_ready: true,
    ai_execution_ready: true,
    profile_options: [
      { name: "FAST", candidate_budget: 24, max_alternating_rounds: 1 },
      { name: "STANDARD", candidate_budget: 64, max_alternating_rounds: 3 },
      { name: "DEEP", candidate_budget: 128, max_alternating_rounds: 5 },
      { name: "ADVANCED", candidate_budget: 192, max_alternating_rounds: 5 },
    ],
  })),
  http.get("/api/v1.1/tasks/active", () => HttpResponse.json(null)),
  http.get("/api/v1.1/hardware/snapshot", () => HttpResponse.json({
    snapshot_token: "snapshot-fake-011",
    b0_hash: "b".repeat(64),
    b0: {
      b0_id: "fake-live-b0",
      version: 1,
      captured_at: "2026-07-29T10:00:00Z",
      i2c_fields: { ibias_dac: 64, vref: 8, vbias_ref: 8, vga_cur: 16, res_gain: 12, res_bw: 10, gain_ctrl: 16, efin_cur: 12, eq: 20 },
      i2c_raw_registers: [[8192, 64], [8200, 20]],
      channel_mapping: { i2c_channel: 0, bert_port: "port1", bert_channel: "0", dca_tdecq_source: "FUNC1", dca_tdecq_equalizer_function: 1, dca_outer_er_source: "CHAN1A" },
      fir: { taps: [0, 9, -28, 104, -8, 1, -3], upper_nlc: 6, lower_nlc: 6, pattern: "SSPRQ", port: "port1", channel: "0" },
      dca_fingerprint: { identity: "FAKE,FlexDCA,1.1" },
      hardware_identities: [{ kind: "dca", transport: "FAKE-VISA", identity: "FAKE-FlexDCA" }],
    },
    mapping_hash: "m".repeat(64),
    mapping: mappingParameters,
    evidence: {
      fake: true,
      mutation_counts: { bert_writes: 0, dca_writes: 0, i2c_page_select_writes: 0, i2c_tunable_register_writes: 0 },
    },
  })),
  http.post("/api/v1.1/mapping/scan", async ({ request }) => {
    const body = await request.json() as { snapshot_token?: string };
    if (body.snapshot_token !== "snapshot-fake-011") {
      return HttpResponse.json({ detail: "snapshot token mismatch" }, { status: 409 });
    }
    return HttpResponse.json({
    snapshot_token: "snapshot-fake-011",
    mapping_hash: "m".repeat(64),
    parameters: mappingParameters,
    });
  }),
  http.post("/api/v1.1/parameters/validate", async ({ request }) => {
    const body = await request.json() as { snapshot_token?: string; parameters?: Array<{ name: string; min: number; max: number; step: number }> };
    if (body.snapshot_token !== "snapshot-fake-011" || !body.parameters) {
      return HttpResponse.json({ detail: "snapshot token mismatch" }, { status: 409 });
    }
    const parameters = body.parameters.map((range, scriptOrder) => {
      const values: number[] = [];
      for (let value = range.min; value <= range.max && values.length < 512; value += range.step) values.push(value);
      return { ...range, values, script_order: scriptOrder };
    });
    return HttpResponse.json({ snapshot_token: body.snapshot_token, mapping_hash: "m".repeat(64), parameters });
  }),
  http.post("/api/v1.1/sweeps/count", async ({ request }) => {
    const body = await request.json() as { snapshot_token?: string; hardware: string; mode: string; seed: number; max_points: number };
    if (body.snapshot_token !== "snapshot-fake-011") {
      return HttpResponse.json({ detail: "snapshot token mismatch" }, { status: 409 });
    }
    return HttpResponse.json({ snapshot_token: body.snapshot_token, mapping_hash: "m".repeat(64), hardware: body.hardware, mode: body.mode, seed: body.seed, point_count: body.max_points, skipped_constraint_count: 0 });
  }),
  http.post("/api/v1.1/tasks", async ({ request }) => {
    const body = await request.json() as typeof defaultV11Config & Record<string, unknown>;
    v11Configs.set(body.task_id, body);
    return HttpResponse.json(v11Task(body.task_id, "DRAFT"), { status: 201 });
  }),
  http.post("/api/v1.1/tasks/:taskId/b0", ({ params }) =>
    HttpResponse.json(v11Task(String(params.taskId), "B0_FROZEN")),
  ),
  http.post("/api/v1.1/tasks/:taskId/authorization", ({ params }) =>
    HttpResponse.json(v11Task(String(params.taskId), "AUTHORIZED")),
  ),
  http.post("/api/v1.1/tasks/:taskId/start", ({ params }) =>
    HttpResponse.json(v11Task(String(params.taskId), "RUNNING")),
  ),
  http.get("/api/v1.1/tasks/:taskId", ({ params }) =>
    HttpResponse.json(v11Task(String(params.taskId), String(params.taskId) === "campaign-recovery" ? "RECOVERY_REQUIRED" : "RUNNING")),
  ),
  http.get("/api/v1.1/tasks/:taskId/events", ({ params }) => HttpResponse.json([{
    task_id: String(params.taskId),
    sequence: 1,
    event_type: "TASK_STARTED",
    from_state: "AUTHORIZED",
    to_state: "RUNNING",
    payload: {},
    product_version: "1.1.0",
    schema_version: 2,
    campaign_type: (v11Configs.get(String(params.taskId)) ?? defaultV11Config).task_type,
    runtime_mode: "FAKE",
    b0_hash: "b".repeat(64),
    config_fingerprint: "c".repeat(64),
    created_at: "2026-07-29T10:01:00Z",
  }])),
  http.get("/api/v1.1/tasks/:taskId/observations", () => HttpResponse.json([])),
  http.get("/api/v1.1/tasks/:taskId/final-report", ({ params }) => HttpResponse.json({
    ...v11Task(String(params.taskId), "RUNNING"),
    observations: [],
    events: [],
  })),
  http.post("/api/v1.1/tasks/:taskId/pause", ({ params }) => HttpResponse.json(v11Task(String(params.taskId), "PAUSED"))),
  http.post("/api/v1.1/tasks/:taskId/resume", ({ params }) => HttpResponse.json(v11Task(String(params.taskId), "RUNNING"))),
  http.post("/api/v1.1/tasks/:taskId/abort", ({ params }) => HttpResponse.json(v11Task(String(params.taskId), "ABORTED"))),
  http.post("/api/v1.1/tasks/:taskId/recovery", async ({ request, params }) => {
    const body = await request.json() as { action: "RESUME" | "ABORT_RECOVERED" };
    return HttpResponse.json(v11Task(String(params.taskId), body.action === "RESUME" ? "RUNNING" : "ABORTED_RECOVERED"));
  }),
  http.get("/api/v1.1/tasks/:taskId/sweep-points", () => HttpResponse.json(v11SweepPointsResponseFixture)),
  http.get("/api/v1.1/tasks/:taskId/best", () => HttpResponse.json({ observation_id: "observation-best-1", i2c: { eq: 18 }, fir: [0, 9, -28, 104, -8, 1, -3], upper_nlc: 6, lower_nlc: 6, pattern: "SSPRQ", bert_port: "port1", bert_channel: "0", dca_fingerprint: "fake", result: { tdecq: 3.2 }, updated_at: "2026-07-29T10:09:00Z" })),
  http.post("/api/v1.1/history/apply", async ({ request }) => {
    const body = await request.json() as { source_task_id: string; observation_id: string };
    return HttpResponse.json({ status: "HISTORY_RESULT_APPLIED", fake: true, source_task_id: body.source_task_id, observation_id: body.observation_id, temporary_rollback: { i2c: { eq: 20 }, fir: [0, 9, -28, 104, -8, 1, -3], upper_nlc: 6, lower_nlc: 6, pattern: "SSPRQ", bert_port: "port1", bert_channel: "0", dca_fingerprint: "fake" }, i2c_readback: { eq: 18 }, fir_readback: [0, 9, -28, 104, -8, 1, -3], measurement: { tdecq: 3.2, er: 3.0, fake: true } });
  }),
  http.post("/api/v1.1/recovery/history/:recoveryId", ({ params }) => HttpResponse.json({ recovery_id: String(params.recoveryId), task_id: "campaign-result", status: "RESOLVED", resolved_at: "2026-07-29T10:10:00Z" })),
];
