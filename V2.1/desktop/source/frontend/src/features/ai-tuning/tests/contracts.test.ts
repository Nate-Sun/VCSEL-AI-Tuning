import {
  campaignSchema,
  observationSchema,
  runtimeHealthSchema,
  safeArtifactHref,
} from "../schemas/contracts";

const sha = "a".repeat(64);

describe("read-only API contracts", () => {
  it("preserves unavailable TDECQ as null", () => {
    const observation = observationSchema.parse({
      observation_id: "obs-1",
      campaign_id: "campaign-1",
      candidate_id: null,
      stage: "BASELINE_MEASUREMENT",
      settings: {
        requested_i2c: {},
        quantized_i2c: {},
        applied_i2c: {},
        readback_i2c: {},
        requested_fir: null,
        applied_fir: null,
        readback_fir: null,
      },
      metrics: { er_db: 2.7, tdecq_db: null },
      valid: false,
      feasible: true,
      elapsed_s: 1.2,
      artifacts: ["artifacts/campaign-1/observation.json"],
      measured_at: "2026-07-24T03:00:00Z",
    });

    expect(observation.metrics.tdecq_db).toBeNull();
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects non-finite metric %s",
    (metric) => {
      expect(() =>
        observationSchema.parse({
          observation_id: "obs-1",
          campaign_id: "campaign-1",
          candidate_id: null,
          stage: "I2C_COARSE",
          settings: {
            requested_i2c: {},
            quantized_i2c: {},
            applied_i2c: {},
            readback_i2c: {},
            requested_fir: null,
            applied_fir: null,
            readback_fir: null,
          },
          metrics: { tdecq_db: metric },
          valid: true,
          feasible: true,
          elapsed_s: 1,
          artifacts: [],
          measured_at: "2026-07-24T03:00:00Z",
        }),
      ).toThrow();
    },
  );

  it("accepts only server-defined campaign outcome values", () => {
    expect(
      campaignSchema.parse({
        campaign_id: "campaign-1",
        baseline_id: "b0-v1",
        baseline_hash: sha,
        config_hash: sha,
        authorization_mode: "PER_STAGE",
        state: "COMPLETED",
        stage_summaries: [],
        result: "PARTIAL_CONVERGENCE",
        created_at: "2026-07-24T03:00:00Z",
        updated_at: "2026-07-24T03:10:00Z",
      }).result,
    ).toBe("PARTIAL_CONVERGENCE");

    expect(() =>
      campaignSchema.parse({
        campaign_id: "campaign-1",
        baseline_id: "b0-v1",
        baseline_hash: sha,
        config_hash: sha,
        authorization_mode: "PER_STAGE",
        state: "COMPLETED",
        stage_summaries: [],
        result: "FRONTEND_GUESSED_SUCCESS",
        created_at: "2026-07-24T03:00:00Z",
        updated_at: "2026-07-24T03:10:00Z",
      }),
    ).toThrow();
  });

  it("keeps FAKE runtime separate from hardware connectivity", () => {
    const health = runtimeHealthSchema.parse({
      status: "ok",
      runtime_mode: "FAKE",
      store: { ok: true },
      repository: { ok: true },
    });

    expect(health.runtime_mode).toBe("FAKE");
    expect(health.hardware).toBeUndefined();
  });

  it("allows only safe artifact links", () => {
    expect(safeArtifactHref("/api/campaigns/campaign-1/report")).toBe(
      "/api/campaigns/campaign-1/report",
    );
    expect(safeArtifactHref("https://lab.example/report.json")).toBe(
      "https://lab.example/report.json",
    );
    expect(safeArtifactHref("javascript:alert(1)")).toBeNull();
    expect(safeArtifactHref("../../secrets.txt")).toBeNull();
  });
});
