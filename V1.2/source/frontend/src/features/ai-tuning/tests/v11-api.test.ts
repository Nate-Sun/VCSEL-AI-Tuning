import { http, HttpResponse } from "msw";
import { server } from "../../../mocks/server";
import * as api from "../api/client";
import { v11SweepPointsResponseFixture } from "../fixtures/v11SweepPoints";

const parameters = [{ name: "eq", min: 18, max: 22, step: 2 }];

const createRequest = {
  task_id: "task-1",
  task_type: "SWEEP" as const,
  channel_mapping: {
    i2c_channel: 0,
    bert_port: "port1",
    bert_channel: "0",
    dca_tdecq_source: "FUNC1" as const,
    dca_tdecq_equalizer_function: 1 as const,
    dca_outer_er_source: "CHAN1A" as const,
    operator_confirmed: true as const,
  },
  parameters,
  depth: "STANDARD" as const,
  candidate_budget: 3,
  sweep_hardware: "I2C_SWEEP" as const,
  sweep_mode: "SINGLE_PARAMETER" as const,
  seed: 1042,
  max_points: 3,
  settle_time_s: 0.1,
  save_dataset: false,
};

function taskDocument(state: "DRAFT" | "B0_FROZEN" | "AUTHORIZED" | "RUNNING" | "PAUSED" | "ABORTED_RECOVERED") {
  return {
    task_id: "task-1",
    task_type: "SWEEP",
    state,
    runtime_mode: "FAKE",
    product_version: "1.1.0",
    schema_version: 2,
    b0_hash: state === "DRAFT" ? null : "b".repeat(64),
    configuration_fingerprint: "c".repeat(64),
    config: createRequest,
    measurement_status: "SUCCESS",
    dataset_status: "DISABLED",
    dataset_error: null,
    dataset_files: [],
    created_at: "2026-07-29T10:00:00Z",
    updated_at: "2026-07-29T10:00:00Z",
  };
}

describe("AI Tuning V1.1 API transport", () => {
  const requests: Array<{ method: string; url: string; body: string }> = [];

  beforeEach(() => {
    requests.length = 0;
    server.use(
      http.post("/api/campaigns/:campaignId/pause", async ({ request }) => {
        requests.push({ method: request.method, url: new URL(request.url).pathname, body: await request.text() });
        return HttpResponse.json({ campaign_id: "campaign-1", state: "PAUSED" });
      }),
      http.post("/api/v1.1/tasks", async ({ request }) => {
        requests.push({ method: request.method, url: new URL(request.url).pathname, body: await request.text() });
        return HttpResponse.json(taskDocument("DRAFT"));
      }),
    );
  });

  it("reads Wizard gates and profiles from V1.1 runtime endpoints", async () => {
    const paths: string[] = [];
    server.use(
      http.get("/api/v1.1/runtime/capabilities", ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json({
          product_version: "1.1.0",
          schema_version: 2,
          runtime_mode: "FAKE",
          campaign_types: ["AI_OPTIMIZATION", "SWEEP"],
          dynamic_b0: true,
          full_campaign_authorization: true,
          hardware_mutation: false,
          fake: true,
          profile_options: [
            { name: "STANDARD", candidate_budget: 64, max_alternating_rounds: 3 },
          ],
        });
      }),
      http.get("/api/v1.1/tasks/active", ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json(null);
      }),
    );

    const [capabilities, activeTask] = await Promise.all([
      api.getV11RuntimeCapabilities(),
      api.getV11ActiveTask(),
    ]);

    expect(capabilities).toMatchObject({
      runtime_mode: "FAKE",
      full_campaign_authorization: true,
      profile_options: [{ name: "STANDARD", candidate_budget: 64 }],
    });
    expect(activeTask).toBeNull();
    expect(paths.sort()).toEqual([
      "/api/v1.1/runtime/capabilities",
      "/api/v1.1/tasks/active",
    ]);
  });

  it("accepts the V1.1 REPLAY runtime mode for capabilities and tasks", async () => {
    server.use(
      http.get("/api/v1.1/runtime/capabilities", () =>
        HttpResponse.json({
          product_version: "1.1.0",
          schema_version: 2,
          runtime_mode: "REPLAY",
          campaign_types: ["AI_OPTIMIZATION", "SWEEP"],
          dynamic_b0: true,
          full_campaign_authorization: true,
          hardware_mutation: false,
          fake: false,
          profile_options: [
            { name: "FAST", candidate_budget: 24, max_alternating_rounds: 1 },
          ],
        }),
      ),
      http.get("/api/v1.1/tasks/active", () =>
        HttpResponse.json({ ...taskDocument("RUNNING"), runtime_mode: "REPLAY" }),
      ),
    );

    await expect(api.getV11RuntimeCapabilities()).resolves.toMatchObject({ runtime_mode: "REPLAY" });
    await expect(api.getV11ActiveTask()).resolves.toMatchObject({ runtime_mode: "REPLAY" });
  });

  it("matches the production OpenAPI methods, paths, and request bodies", async () => {
    const canonicalRequests: Array<{ method: string; path: string; body: string; cache: RequestCache }> = [];
    const capture = async (request: Request) => {
      canonicalRequests.push({
        method: request.method,
        path: new URL(request.url).pathname,
        body: await request.text(),
        cache: request.cache,
      });
    };
    server.use(
      http.get("/api/v1.1/hardware/snapshot", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({
          snapshot_token: "snapshot-1",
          b0_hash: "b".repeat(64),
          b0: {
            b0_id: "fake-live-b0",
            version: 1,
            captured_at: "2026-07-29T10:00:00Z",
            i2c_fields: { eq: 20 },
            i2c_raw_registers: [[8192, 20]],
            channel_mapping: {
              i2c_channel: 0,
              bert_port: "port1",
              bert_channel: "0",
              dca_tdecq_source: "FUNC1",
              dca_tdecq_equalizer_function: 1,
              dca_outer_er_source: "CHAN1A",
            },
            fir: { taps: [0, 9, -28, 104, -8, 1, -3], upper_nlc: 6, lower_nlc: 6, pattern: "SSPRQ", port: "port1", channel: "0" },
            dca_fingerprint: { identity: "FAKE-FlexDCA" },
            hardware_identities: [],
          },
          mapping_hash: "m".repeat(64),
          mapping: [],
          evidence: {
            fake: true,
            mutation_counts: { bert_writes: 0, dca_writes: 0, i2c_page_select_writes: 0, i2c_tunable_register_writes: 0 },
          },
        });
      }),
      http.post("/api/v1.1/mapping/scan", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({ snapshot_token: "snapshot-1", mapping_hash: "m".repeat(64), parameters: [] });
      }),
      http.post("/api/v1.1/parameters/validate", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({ snapshot_token: "snapshot-1", mapping_hash: "m".repeat(64), parameters: [{ ...parameters[0], values: [18, 20, 22], script_order: 8 }] });
      }),
      http.post("/api/v1.1/sweeps/count", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({ snapshot_token: "snapshot-1", mapping_hash: "m".repeat(64), hardware: "I2C_SWEEP", mode: "SINGLE_PARAMETER", seed: 1042, point_count: 3, skipped_constraint_count: 0 });
      }),
      http.post("/api/v1.1/tasks", async ({ request }) => {
        await capture(request);
        return HttpResponse.json(taskDocument("DRAFT"), { status: 201 });
      }),
      http.post("/api/v1.1/tasks/:taskId/b0", async ({ request }) => {
        await capture(request);
        return HttpResponse.json(taskDocument("B0_FROZEN"));
      }),
      http.post("/api/v1.1/tasks/:taskId/authorization", async ({ request }) => {
        await capture(request);
        return HttpResponse.json(taskDocument("AUTHORIZED"));
      }),
      http.post("/api/v1.1/tasks/:taskId/start", async ({ request }) => {
        await capture(request);
        return HttpResponse.json(taskDocument("RUNNING"));
      }),
    );

    await api.getV11HardwareSnapshot();
    await api.scanV11Mapping("snapshot-1");
    await api.validateV11Parameters("snapshot-1", parameters);
    await api.countV11Sweep({ snapshot_token: "snapshot-1", hardware: "I2C_SWEEP", mode: "SINGLE_PARAMETER", parameters, seed: 1042, max_points: 3 });
    await api.createV11Task(createRequest);
    await api.freezeV11B0("task-1", { snapshot_token: "snapshot-1", operator_confirmed: true });
    await api.authorizeV11Task("task-1", { authorization_id: "auth-1", expires_at: "2099-07-29T10:30:00.000Z", operator_confirmed: true });
    await api.startV11Task("task-1");

    expect(canonicalRequests.map(({ method, path }) => [method, path])).toEqual([
      ["GET", "/api/v1.1/hardware/snapshot"],
      ["POST", "/api/v1.1/mapping/scan"],
      ["POST", "/api/v1.1/parameters/validate"],
      ["POST", "/api/v1.1/sweeps/count"],
      ["POST", "/api/v1.1/tasks"],
      ["POST", "/api/v1.1/tasks/task-1/b0"],
      ["POST", "/api/v1.1/tasks/task-1/authorization"],
      ["POST", "/api/v1.1/tasks/task-1/start"],
    ]);
    expect(canonicalRequests[0]!.body).toBe("");
    expect(canonicalRequests[0]!.cache).toBe("no-store");
    expect(JSON.parse(canonicalRequests[1]!.body)).toEqual({ snapshot_token: "snapshot-1" });
    expect(JSON.parse(canonicalRequests[2]!.body)).toEqual({ snapshot_token: "snapshot-1", parameters });
    expect(JSON.parse(canonicalRequests[3]!.body)).toMatchObject({ snapshot_token: "snapshot-1" });
    expect(JSON.parse(canonicalRequests[4]!.body)).toEqual(createRequest);
    expect(JSON.parse(canonicalRequests[4]!.body) as unknown).toMatchObject({ settle_time_s: 0.1 });
    expect(canonicalRequests[7]!.body).toBe("");
    expect("TEMPORARY_V11_ENDPOINTS" in api).toBe(false);
  });

  it("sends the dataset choice and preserves dataset result metadata", async () => {
    server.use(
      http.post("/api/v1.1/tasks", async ({ request }) => {
        const body = await request.json() as typeof createRequest;
        requests.push({ method: request.method, url: new URL(request.url).pathname, body: JSON.stringify(body) });
        return HttpResponse.json({
          ...taskDocument("DRAFT"),
          config: body,
          dataset_status: "PENDING",
          dataset_files: [{
            file_id: "task-record",
            kind: "MEASUREMENT_RECORD_JSON",
            name: "record.json",
            url: "/api/v1.1/tasks/task-1/datasets/record.json",
            size_bytes: 512,
            point_id: null,
            observation_id: null,
          }],
        });
      }),
    );

    const result = await api.createV11Task({ ...createRequest, save_dataset: true });

    expect(JSON.parse(requests.at(-1)!.body)).toMatchObject({ save_dataset: true });
    expect(result.config.save_dataset).toBe(true);
    expect(result.dataset_status).toBe("PENDING");
    expect(result.dataset_files[0]?.name).toBe("record.json");
  });

  it("accepts unavailable scan bindings whose script order is null", async () => {
    server.use(
      http.post("/api/v1.1/mapping/scan", () =>
        HttpResponse.json({
          snapshot_token: "snapshot-unavailable",
          mapping_hash: "m".repeat(64),
          parameters: [
            {
              name: "reserved_field",
              status: "UNAVAILABLE",
              reason: "not present in this RegMap",
              script_order: null,
              current_value: null,
              channel0_address: null,
              channel_stride: null,
              bit_start: null,
              bits: null,
              encoding_min: null,
              encoding_max: null,
            },
          ],
        }),
      ),
    );

    await expect(api.scanV11Mapping("snapshot-unavailable")).resolves.toMatchObject({
      parameters: [{ status: "UNAVAILABLE", script_order: null }],
    });
  });

  it("uses task-scoped production paths for controls, recovery, Sweep results, and history", async () => {
    const secondaryRequests: Array<{ method: string; path: string; body: string; query: string }> = [];
    const capture = async (request: Request) => {
      const url = new URL(request.url);
      secondaryRequests.push({ method: request.method, path: url.pathname, body: await request.text(), query: url.search });
    };
    server.use(
      http.get("/api/v1.1/tasks/:taskId", async ({ request }) => { await capture(request); return HttpResponse.json(taskDocument("RUNNING")); }),
      http.post("/api/v1.1/tasks/:taskId/pause", async ({ request }) => { await capture(request); return HttpResponse.json(taskDocument("PAUSED")); }),
      http.post("/api/v1.1/tasks/:taskId/resume", async ({ request }) => { await capture(request); return HttpResponse.json(taskDocument("RUNNING")); }),
      http.post("/api/v1.1/tasks/:taskId/abort", async ({ request }) => { await capture(request); return HttpResponse.json(taskDocument("ABORTED_RECOVERED")); }),
      http.post("/api/v1.1/tasks/:taskId/recovery", async ({ request }) => { await capture(request); return HttpResponse.json(taskDocument("RUNNING")); }),
      http.get("/api/v1.1/tasks/:taskId/sweep-points", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({ items: [], total: 0, offset: 25, limit: 25, best_point: null });
      }),
      http.post("/api/v1.1/history/apply", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({
          status: "HISTORY_RESULT_APPLIED",
          fake: true,
          source_task_id: "task-1",
          observation_id: "observation-1",
          temporary_rollback: { i2c: { eq: 20 }, fir: [0, 9, -28, 104, -8, 1, -3], upper_nlc: 6, lower_nlc: 6, pattern: "SSPRQ", bert_port: "port1", bert_channel: "0", dca_fingerprint: "fake" },
          i2c_readback: { eq: 18 },
          fir_readback: [0, 9, -28, 104, -8, 1, -3],
          measurement: { tdecq: 3.2, er: 3.0, fake: true },
        });
      }),
      http.post("/api/v1.1/recovery/history/:recoveryId", async ({ request }) => {
        await capture(request);
        return HttpResponse.json({ recovery_id: "recovery-1", task_id: "task-1", status: "RESOLVED", resolved_at: "2026-07-29T10:10:00Z" });
      }),
    );

    await api.getV11Task("task-1");
    await api.pauseV11Task("task-1");
    await api.resumeV11Task("task-1");
    await api.abortV11Task("task-1", "operator requested stop");
    await api.recoverV11Task("task-1", "RESUME");
    await api.getV11SweepPoints("task-1", 25, 25);
    await api.applyV11HistoryResult({ source_task_id: "task-1", observation_id: "observation-1" });
    await api.recoverV11History("recovery-1");

    expect(secondaryRequests.map(({ method, path }) => [method, path])).toEqual([
      ["GET", "/api/v1.1/tasks/task-1"],
      ["POST", "/api/v1.1/tasks/task-1/pause"],
      ["POST", "/api/v1.1/tasks/task-1/resume"],
      ["POST", "/api/v1.1/tasks/task-1/abort"],
      ["POST", "/api/v1.1/tasks/task-1/recovery"],
      ["GET", "/api/v1.1/tasks/task-1/sweep-points"],
      ["POST", "/api/v1.1/history/apply"],
      ["POST", "/api/v1.1/recovery/history/recovery-1"],
    ]);
    expect(JSON.parse(secondaryRequests[3]!.body)).toEqual({ reason: "operator requested stop" });
    expect(JSON.parse(secondaryRequests[4]!.body)).toEqual({ action: "RESUME", operator_confirmed: true });
    expect(secondaryRequests[5]!.query).toBe("?offset=25&limit=25");
    expect(api.v11SweepExportUrl("task-1", "csv")).toContain("/api/v1.1/tasks/task-1/sweep-results.csv");
    expect(api.v11SweepExportUrl("task-1", "json")).toContain("/api/v1.1/tasks/task-1/sweep-results.json");
  });

  it("returns point_id, global best, and timing evidence from the canonical Sweep endpoint", async () => {
    server.use(
      http.get("/api/v1.1/tasks/:taskId/sweep-points", () => HttpResponse.json(v11SweepPointsResponseFixture)),
    );

    const page = await api.getV11SweepPoints("task-1", 0, 25);

    expect(page.items[0]?.point_id).toBe("point-00000004");
    expect(page.best_point?.point_id).toBe("point-00000001");
    expect(page.items[0]?.raw_response?.requested_settle_s).toBe(0.1);
    expect(page.items[0]?.candidate_round_time_s).toBe(8.2);
    expect(page.items[0]?.restore_status).toBe("VERIFIED");
  });

  it("reads V1.1 task events, observations, and final report without legacy campaign paths", async () => {
    const paths: string[] = [];
    const observation = {
      observation_id: "observation-v11-1",
      sequence: 1,
      phase: "I2C_COARSE",
      status: "COMPLETED",
      valid: true,
      requested: { eq: 18 },
      applied: { eq: 18 },
      readback: { eq: 18 },
      tdecq: 3.2,
      er: 3.0,
      raw_response: {},
      error_reason: null,
      created_at: "2026-07-29T10:01:00Z",
    };
    const event = {
      task_id: "task-1",
      sequence: 1,
      event_type: "OBSERVATION_RECORDED",
      from_state: "AUTHORIZED",
      to_state: "RUNNING",
      payload: {},
      product_version: "1.1.0",
      schema_version: 2,
      campaign_type: "SWEEP",
      runtime_mode: "REPLAY",
      b0_hash: "b".repeat(64),
      config_fingerprint: "c".repeat(64),
      created_at: "2026-07-29T10:01:00Z",
    };
    server.use(
      http.get("/api/v1.1/tasks/:taskId/events", ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json([event]);
      }),
      http.get("/api/v1.1/tasks/:taskId/observations", ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json([observation]);
      }),
      http.get("/api/v1.1/tasks/:taskId/final-report", ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json({
          ...taskDocument("RUNNING"),
          runtime_mode: "REPLAY",
          observations: [observation],
          events: [{}],
        });
      }),
    );

    await expect(api.getV11TaskEvents("task-1")).resolves.toMatchObject([
      { task_id: "task-1", runtime_mode: "REPLAY" },
    ]);
    await expect(api.getV11Observations("task-1")).resolves.toMatchObject([
      { observation_id: "observation-v11-1", tdecq: 3.2 },
    ]);
    await expect(api.getV11FinalReport("task-1")).resolves.toMatchObject({
      task_id: "task-1",
      runtime_mode: "REPLAY",
    });
    expect(paths).toEqual([
      "/api/v1.1/tasks/task-1/events",
      "/api/v1.1/tasks/task-1/observations",
      "/api/v1.1/tasks/task-1/final-report",
    ]);
  });

  it("does not call start when authorization fails", async () => {
    let startCalls = 0;
    server.use(
      http.post("/api/v1.1/tasks/:taskId/authorization", () =>
        HttpResponse.json(
          { detail: { code: "authorization_expired", message: "authorization has expired" } },
          { status: 409 },
        ),
      ),
      http.post("/api/v1.1/tasks/:taskId/start", () => {
        startCalls += 1;
        return HttpResponse.json(taskDocument("RUNNING"));
      }),
    );

    await expect(
      api.authorizeThenStartV11Task("task-1", {
        authorization_id: "auth-1",
        expires_at: "2099-07-29T10:30:00.000Z",
        operator_confirmed: true,
      }),
    ).rejects.toMatchObject({ status: 409, code: "authorization_expired" });
    expect(startCalls).toBe(0);
  });

  it("requires an AUTHORIZED response before calling start", async () => {
    let startCalls = 0;
    server.use(
      http.post("/api/v1.1/tasks/:taskId/authorization", () =>
        HttpResponse.json(taskDocument("B0_FROZEN")),
      ),
      http.post("/api/v1.1/tasks/:taskId/start", () => {
        startCalls += 1;
        return HttpResponse.json(taskDocument("RUNNING"));
      }),
    );

    await expect(
      api.authorizeThenStartV11Task("task-1", {
        authorization_id: "auth-1",
        expires_at: "2099-07-29T10:30:00.000Z",
        operator_confirmed: true,
      }),
    ).rejects.toMatchObject({ code: "authorization_state_mismatch" });
    expect(startCalls).toBe(0);
  });

  it("uses registered POST endpoints without low-level hardware payloads", async () => {
    await api.pauseCampaign("campaign-1");
    await expect(api.createV11Task(createRequest)).resolves.toMatchObject({ task_type: "SWEEP" });

    expect(requests[0]).toMatchObject({ method: "POST", url: "/api/campaigns/campaign-1/pause" });
    expect(requests[1]).toMatchObject({ method: "POST", url: "/api/v1.1/tasks" });
    expect(JSON.stringify(requests)).not.toMatch(/SCPI|register_address|setFir|raw_command/i);
  });

  it.each([409, 422, 503])("preserves structured FastAPI errors for HTTP %s", async (status) => {
    server.use(
      http.post("/api/v1.1/tasks", () =>
        HttpResponse.json(
          { detail: { code: "campaign_conflict", message: "another campaign is active", fields: { campaign_id: "active" } } },
          { status },
        ),
      ),
    );

    const error = await api.createV11Task(createRequest).catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(api.ApiResponseError);
    expect(error).toMatchObject({
      status,
      code: "campaign_conflict",
      message: "another campaign is active",
      fields: { campaign_id: "active" },
    });
  });

  it("preserves FastAPI's default 422 detail array", async () => {
    server.use(
      http.post("/api/v1.1/tasks", () =>
        HttpResponse.json(
          { detail: [{ loc: ["body", "task_type"], msg: "Field required", type: "missing" }] },
          { status: 422 },
        ),
      ),
    );

    const error = await api.createV11Task(createRequest).catch((reason: unknown) => reason);
    expect(error).toMatchObject({ status: 422, code: "validation_error", message: "Field required" });
  });

  it("does not accept an arbitrary 2xx body as a verified B0", async () => {
    server.use(
      http.post("/api/v1.1/tasks/:taskId/b0", () => HttpResponse.json({ status: "saved" })),
    );
    await expect(api.freezeV11B0("task-1", { snapshot_token: "snapshot-1", operator_confirmed: true })).rejects.toMatchObject({
      message: "Invalid response from POST /api/v1.1/tasks/task-1/b0",
    });
  });
});
