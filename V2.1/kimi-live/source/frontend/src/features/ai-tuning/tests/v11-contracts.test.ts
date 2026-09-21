import {
  apiErrorSchema,
  discreteGridSchema,
  historyApplySchema,
  parameterNameSchema,
  recoverySchema,
  runtimeModeSchema,
  sweepPageSchema,
  taskTypeSchema,
  versionMetadataSchema,
} from "../schemas/contracts";

describe("AI Tuning V1.1 contracts", () => {
  it("normalizes current backend LIVE_TUNING to the V1.1 LIVE presentation", () => {
    expect(runtimeModeSchema.parse("LIVE_TUNING")).toBe("LIVE");
    expect(runtimeModeSchema.parse("FAKE")).toBe("FAKE");
    expect(runtimeModeSchema.parse("READ_ONLY")).toBe("READ_ONLY");
  });

  it("accepts only the two V1.1 task types", () => {
    expect(taskTypeSchema.parse("AI_OPTIMIZATION")).toBe("AI_OPTIMIZATION");
    expect(taskTypeSchema.parse("SWEEP")).toBe("SWEEP");
    expect(taskTypeSchema.safeParse("RAW_COMMAND").success).toBe(false);
  });

  it("accepts only the nine safe Script/RegMap parameters", () => {
    expect(parameterNameSchema.parse("eq")).toBe("eq");
    for (const forbidden of ["testch", "disable", "power_up_done", "spare"]) {
      expect(parameterNameSchema.safeParse(forbidden).success).toBe(false);
    }
  });

  it("keeps a server-provided discrete grid without appending the max", () => {
    const grid = discreteGridSchema.parse({ values: [0, 3, 6, 9], b0_outside: true });
    expect(grid.values).toEqual([0, 3, 6, 9]);
    expect(grid.values).not.toContain(10);
    expect(grid.b0_outside).toBe(true);
  });

  it("requires version metadata on V1.1 task documents", () => {
    expect(
      versionMetadataSchema.parse({
        product_version: "1.1.0",
        schema_version: "v1.1",
        runtime_mode: "FAKE",
        campaign_type: "SWEEP",
      }),
    ).toMatchObject({ product_version: "1.1.0", campaign_type: "SWEEP" });
  });

  it("rejects non-finite Sweep metrics", () => {
    expect(() =>
      sweepPageSchema.parse({
        items: [{
          point_id: "point-1",
          original_index: 0,
          execution_index: 0,
          status: "VALID",
          parameters: { eq: 22 },
          requested: { eq: 22 },
          applied: { eq: 22 },
          readback: { eq: 22 },
          tdecq_db: Number.NaN,
          er_db: 2.9,
          is_best: true,
        }],
        page: 1,
        page_size: 25,
        total: 1,
      }),
    ).toThrow();
  });

  it("parses Recovery and history Apply server stages", () => {
    const recovery = recoverySchema.parse({
      campaign_id: "campaign-recovery",
      task_type: "AI_OPTIMIZATION",
      interrupted_at: "2026-07-27T10:00:00Z",
      last_completed_point: "candidate-8",
      last_known_i2c: { eq: 22 },
      last_known_fir: null,
      baseline: { i2c: { eq: 20 }, fir: null },
      error_reason: "rollback failed",
      rollback_error: "BERT readback mismatch",
      devices: { dca: false, i2c: true, bert: true },
      restore_verified: false,
    });
    expect(recovery.restore_verified).toBe(false);

    const apply = historyApplySchema.parse({
      operation_id: "apply-1",
      status: "RUNNING",
      stages: [{ name: "DEVICE_VERIFICATION", status: "COMPLETE" }],
      rollback_status: null,
    });
    expect(apply.stages[0]?.name).toBe("DEVICE_VERIFICATION");
  });

  it("preserves structured FastAPI errors", () => {
    expect(apiErrorSchema.parse({
      detail: {
        code: "campaign_conflict",
        message: "another campaign is active",
        fields: { campaign_id: "active" },
      },
    }).detail.code).toBe("campaign_conflict");
  });
});
