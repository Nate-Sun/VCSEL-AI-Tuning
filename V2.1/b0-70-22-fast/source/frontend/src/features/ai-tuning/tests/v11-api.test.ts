import { http, HttpResponse } from "msw";
import { server } from "../../../mocks/server";
import * as api from "../api/client";

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
        return HttpResponse.json({ task_id: "task-1", task_type: "SWEEP", state: "DRAFT" });
      }),
    );
  });

  it("uses registered POST endpoints without low-level hardware payloads", async () => {
    await api.pauseCampaign("campaign-1");
    await expect(api.createTask({ task_type: "SWEEP" })).resolves.toMatchObject({ task_type: "SWEEP" });

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

    const error = await api.createTask({ task_type: "SWEEP" }).catch((reason: unknown) => reason);
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

    const error = await api.createTask({ task_type: "SWEEP" }).catch((reason: unknown) => reason);
    expect(error).toMatchObject({ status: 422, code: "validation_error", message: "Field required" });
  });

  it("publishes the temporary V1.1 endpoint registry", () => {
    expect(api.TEMPORARY_V11_ENDPOINTS.createTask).toBe("/api/v1.1/tasks");
    expect(Object.values(api.TEMPORARY_V11_ENDPOINTS)).not.toContain("/hardware");
  });

  it("parses the centralized snapshot, scan, range, count, and authorization contracts", async () => {
    await expect(api.captureSnapshot("task-1")).resolves.toMatchObject({ snapshot_id: "snapshot-fake-011" });
    await expect(api.scanConfiguration("task-1")).resolves.toMatchObject({ rejected_names: ["testch", "disable", "power_up_done", "spare"] });
    await expect(api.validateParameterRanges("task-1", { ranges: [{ name: "eq", min: 0, max: 10, step: 3 }] })).resolves.toMatchObject({ grids: { eq: { values: [0, 3, 6, 9] } } });
    await expect(api.getSweepCount("task-1", {})).resolves.toMatchObject({ legal_count: 64, over_limit: false });
    await expect(api.authorizeAndStart("task-1", { sweep: { mode: "FULL" } })).resolves.toMatchObject({ campaign_id: "campaign-fake-011", authorization_id: "auth-fake-117" });
  });

  it("does not accept an arbitrary 2xx body as a verified B0", async () => {
    server.use(
      http.post("/api/v1.1/tasks/:taskId/baseline", () => HttpResponse.json({ status: "saved" })),
    );
    await expect(api.saveBaseline("task-1", {})).rejects.toMatchObject({
      message: "Invalid response from POST /api/v1.1/tasks/task-1/baseline",
    });
  });
});
