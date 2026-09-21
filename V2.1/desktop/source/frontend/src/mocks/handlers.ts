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

export const handlers = [
  http.get("/health", () => HttpResponse.json(runtimeFixture)),
  http.get("/capabilities", () => HttpResponse.json(capabilityFixture)),
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
    HttpResponse.json(
      String(params.campaignId) === "campaign-result"
        ? resultReportFixture
        : liveReportFixture,
    ),
  ),
  http.get("/api/campaigns/:campaignId/events/stream", () => {
    const body = eventFixtures
      .map(
        (event) =>
          `id: ${event.sequence}\nevent: ${event.event_type}\ndata: ${JSON.stringify(event)}\n\n`,
      )
      .join("");
    return new HttpResponse(body, {
      headers: { "Content-Type": "text/event-stream" },
    });
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
  http.post("/api/v1.1/tasks", async ({ request }) => {
    const body = await request.json() as { task_type?: string };
    return HttpResponse.json({ task_id: "task-fake-011", task_type: body.task_type ?? "AI_OPTIMIZATION", state: "DRAFT", simulated: true });
  }),
  http.post("/api/v1.1/tasks/:taskId/snapshot", () =>
    HttpResponse.json({
      snapshot_id: "snapshot-fake-011",
      simulated: true,
      dca: { identity: "Keysight N1092A", source: "CH1A", wavelength: "1310 nm", pattern: "PAM4", trigger: "CLOCK/2", tdecq_source: "CH1A", er_source: "CH1A", equalizer_taps: [0, 3, -1] },
      bert: { identity: "Keysight M8040A", port: "1", pattern: "PRBS31Q", fir: [0, 8, -27, 105, -7, 1, -3], upper_nlc: 6, lower_nlc: 6 },
      i2c: { identity: "VCSEL TX", bus: "2", values: { ibias_dac: 72, vref: 15, eq: 22 } },
    }),
  ),
  http.post("/api/v1.1/tasks/:taskId/scan-configuration", () =>
    HttpResponse.json({
      simulated: true,
      parameters: ["ibias_dac", "vref", "vbias_ref", "vga_cur", "res_gain", "res_bw", "gain_ctrl", "efin_cur", "eq"].map((name, index) => ({
        name,
        current: name === "ibias_dac" ? 72 : name === "eq" ? 22 : 15,
        address_summary: `field ${index + 1}:0`,
        encoding_range: "0…255",
        available: true,
        unavailable_reason: null,
      })),
      rejected_names: ["testch", "disable", "power_up_done", "spare"],
    }),
  ),
  http.post("/api/v1.1/tasks/:taskId/baseline", () =>
    HttpResponse.json({ baseline_id: "b0-v11-fake", status: "VERIFIED", simulated: true }),
  ),
  http.post("/api/v1.1/tasks/:taskId/validate-ranges", async ({ request }) => {
    const body = await request.json() as { ranges?: Array<{ name: string; min: number; max: number; step: number }> };
    const grids = Object.fromEntries((body.ranges ?? []).map((range) => {
      const values: number[] = [];
      for (let value = range.min; value <= range.max && values.length < 512; value += range.step) values.push(value);
      const b0 = range.name === "ibias_dac" ? 72 : range.name === "eq" ? 22 : 15;
      return [range.name, { values, b0_outside: !values.includes(b0) }];
    }));
    return HttpResponse.json({ grids, simulated: true });
  }),
  http.post("/api/v1.1/tasks/:taskId/sweep-count", () =>
    HttpResponse.json({ original_count: 81, legal_count: 64, skipped_count: 17, limit: 128, over_limit: false, simulated: true }),
  ),
  http.post("/api/v1.1/tasks/:taskId/authorize-start", async ({ request, params }) => {
    const body = await request.json() as { sweep?: unknown };
    return HttpResponse.json({ task_id: String(params.taskId), campaign_id: "campaign-fake-011", task_type: body.sweep ? "SWEEP" : "AI_OPTIMIZATION", state: "PREFLIGHT", authorization_id: "auth-fake-117", simulated: true });
  }),
  http.get("/api/v1.1/campaigns/:campaignId/sweep-points", () =>
    HttpResponse.json({ items: [
      { point_id: "point-16", original_index: 15, execution_index: 16, status: "VALID", parameters: { eq: 18 }, requested: { eq: 18 }, applied: { eq: 18 }, readback: { eq: 18 }, tdecq_db: 3.62, er_db: 2.82, is_best: false },
      { point_id: "point-17", original_index: 16, execution_index: 17, status: "INVALID_TDECQ", parameters: { eq: 21 }, requested: { eq: 21 }, applied: { eq: 21 }, readback: { eq: 21 }, tdecq_db: null, er_db: 2.91, is_best: false, error_reason: "DCA returned no valid TDECQ" },
      { point_id: "point-18", original_index: 17, execution_index: 18, status: "VALID", parameters: { eq: 22 }, requested: { eq: 22 }, applied: { eq: 22 }, readback: { eq: 22 }, tdecq_db: 3.36, er_db: 3.08, is_best: true },
      { point_id: "point-19", original_index: 18, execution_index: 19, status: "SKIPPED_CONSTRAINT", parameters: { eq: 25 }, requested: { eq: 25 }, applied: { eq: 25 }, readback: { eq: 25 }, tdecq_db: null, er_db: null, is_best: false, error_reason: "constraint rejected before write" },
    ], page: 1, page_size: 25, total: 64, summary: {
      hardware: "I2C",
      mode: "FULL",
      seed: 1042,
      completed_count: 18,
      legal_count: 64,
      skipped_count: 17,
      total_count: 81,
      remaining_count: 46,
      current_original_index: 17,
      current_execution_index: 18,
      preflight_allowed: true,
      b0_restore_status: "VERIFIED",
      hardware_restored: false,
      devices: { dca: "SIMULATED · READY", i2c: "SIMULATED · READY", bert: "SIMULATED · READY" },
      dca_drift: { detected: false, original: { source: "CH1A", wavelength: "1310 nm", pattern: "PAM4", trigger: "CLOCK/2" }, current: { source: "CH1A", wavelength: "1310 nm", pattern: "PAM4", trigger: "CLOCK/2" } },
    } }),
  ),
  http.get("/api/v1.1/campaigns/:campaignId/recovery", ({ params }) =>
    HttpResponse.json({ campaign_id: String(params.campaignId), task_type: "AI_OPTIMIZATION", interrupted_at: "2026-07-27T10:00:00Z", last_completed_point: "candidate-8", last_known_i2c: { eq: 22 }, last_known_fir: null, baseline: { i2c: { eq: 20 }, fir: null }, error_reason: "rollback interrupted", rollback_error: "BERT readback mismatch", devices: { dca: true, i2c: true, bert: true }, restore_verified: false }),
  ),
  http.post("/api/v1.1/campaigns/:campaignId/recovery/restore-baseline", ({ params }) =>
    HttpResponse.json({ campaign_id: String(params.campaignId), task_type: "AI_OPTIMIZATION", interrupted_at: "2026-07-27T10:00:00Z", last_completed_point: "candidate-8", last_known_i2c: { eq: 20 }, last_known_fir: null, baseline: { i2c: { eq: 20 }, fir: null }, error_reason: "rollback interrupted", rollback_error: null, devices: { dca: true, i2c: true, bert: true }, restore_verified: true }),
  ),
  http.post("/api/v1.1/campaigns/:campaignId/recovery/continue", ({ params }) => HttpResponse.json({ campaign_id: String(params.campaignId), state: "PREFLIGHT" })),
  http.post("/api/v1.1/campaigns/:campaignId/recovery/end", ({ params }) => HttpResponse.json({ campaign_id: String(params.campaignId), state: "FAILED" })),
  http.post("/api/v1.1/campaigns/:campaignId/apply", () => HttpResponse.json({ operation_id: "apply-fake-1", status: "HISTORY_RESULT_APPLIED", stages: ["DEVICE_VERIFICATION", "SNAPSHOT_CURRENT_STATE", "APPLY_I2C", "I2C_READBACK", "APPLY_FIR", "BERT_READBACK", "DCA_MEASUREMENT", "COMPLETE_OR_ROLLBACK"].map((name) => ({ name, status: "COMPLETE" })), rollback_status: null })),
  http.get("/api/v1.1/operations/:operationId", ({ params }) => HttpResponse.json({ operation_id: String(params.operationId), status: "HISTORY_RESULT_APPLIED", stages: ["DEVICE_VERIFICATION", "SNAPSHOT_CURRENT_STATE", "APPLY_I2C", "I2C_READBACK", "APPLY_FIR", "BERT_READBACK", "DCA_MEASUREMENT", "COMPLETE_OR_ROLLBACK"].map((name) => ({ name, status: "COMPLETE" })), rollback_status: null })),
];
