import { http, HttpResponse } from "msw";
import { server } from "../../../mocks/server";
import * as api from "../api/client";

const sha = "a".repeat(64);
const campaign = {
  campaign_id: "campaign-1",
  baseline_id: "b0-v1",
  baseline_hash: sha,
  config_hash: sha,
  authorization_mode: "PER_STAGE",
  state: "DRAFT",
  stage_summaries: [],
  result: null,
  created_at: "2026-07-24T03:00:00Z",
  updated_at: "2026-07-24T03:00:00Z",
};

describe("allow-listed API client", () => {
  it("exposes only approved GET endpoint templates", () => {
    expect(api.ALLOWED_GET_ENDPOINTS).toEqual([
      "/health",
      "/capabilities",
      "/api/campaigns",
      "/api/campaigns/{id}",
      "/api/campaigns/{id}/observations",
      "/api/campaigns/{id}/best",
      "/api/campaigns/{id}/report",
      "/api/campaigns/{id}/events/stream",
    ]);
    expect(api).not.toHaveProperty("approveStage");
    expect(api).not.toHaveProperty("setFir");
    expect(api).not.toHaveProperty("sendRawCommand");
  });

  it("parses the approved REST responses", async () => {
    server.use(
      http.get("/health", () =>
        HttpResponse.json({ status: "ok", runtime_mode: "FAKE" }),
      ),
      http.get("/capabilities", () =>
        HttpResponse.json({
          runtime_mode: "FAKE",
          hardware_connected: false,
          read_only: false,
          hierarchical_ready: false,
          fir_proposer_ready: true,
          live_tuning_enabled: false,
          authorization_modes: ["PER_STAGE"],
          FULL_CAMPAIGN: false,
        }),
      ),
      http.get("/api/campaigns", () => HttpResponse.json([campaign])),
      http.get("/api/campaigns/campaign-1", () => HttpResponse.json(campaign)),
      http.get("/api/campaigns/campaign-1/observations", () =>
        HttpResponse.json([]),
      ),
      http.get("/api/campaigns/campaign-1/best", () =>
        HttpResponse.json(null),
      ),
      http.get("/api/campaigns/campaign-1/report", () =>
        HttpResponse.json({
          campaign_id: "campaign-1",
          outcome: null,
          rollback_history: [],
          artifacts: [],
        }),
      ),
    );

    await expect(api.getHealth()).resolves.toMatchObject({ runtime_mode: "FAKE" });
    await expect(api.getCapabilities()).resolves.toMatchObject({
      hardware_connected: false,
    });
    await expect(api.getCampaigns()).resolves.toHaveLength(1);
    await expect(api.getCampaign("campaign-1")).resolves.toMatchObject({
      state: "DRAFT",
    });
    await expect(api.getObservations("campaign-1")).resolves.toEqual([]);
    await expect(api.getBestObservation("campaign-1")).resolves.toBeNull();
    await expect(api.getCampaignReport("campaign-1")).resolves.toMatchObject({
      campaign_id: "campaign-1",
    });
  });

  it("rejects an invalid response instead of polluting UI state", async () => {
    server.use(
      http.get("/health", () =>
        HttpResponse.json({ status: "ok", runtime_mode: "HARDWARE_IS_FINE" }),
      ),
    );

    await expect(api.getHealth()).rejects.toThrow(/invalid response/i);
  });
});
