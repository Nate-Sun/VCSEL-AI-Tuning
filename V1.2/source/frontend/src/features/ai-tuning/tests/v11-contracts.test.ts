import {
  apiErrorSchema,
  discreteGridSchema,
  historyApplySchema,
  v11HistoryApplyReceiptSchema,
  v11SweepPointPageSchema,
  parameterNameSchema,
  recoverySchema,
  runtimeModeSchema,
  sweepPageSchema,
  taskTypeSchema,
  v11RuntimeModeSchema,
  v11RuntimeCapabilitiesSchema,
  v11TaskDocumentSchema,
  versionMetadataSchema,
} from "../schemas/contracts";
import { v11SweepPointsResponseFixture } from "../fixtures/v11SweepPoints";

describe("AI Tuning V1.1 contracts", () => {
  it("accepts the complete Native V1.2 staged LIVE capability", () => {
    const parsed = v11RuntimeCapabilitiesSchema.parse({
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
      profile_options: [{
        name: "PILOT_3_THEN_FAST24",
        candidate_budget: 27,
        max_alternating_rounds: 2,
      }],
    });

    expect(parsed.live_ai_profile).toBe("PILOT_3_THEN_FAST24");
    expect(parsed.campaign_types).toEqual(["AI_OPTIMIZATION"]);
  });
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

  it("accepts the built-in V1.1 READ_ONLY runtime mode", () => {
    expect(v11RuntimeModeSchema.parse("READ_ONLY")).toBe("READ_ONLY");
  });

  it("preserves the FAKE attestation in a V1.1 History Apply receipt", () => {
    const receipt = v11HistoryApplyReceiptSchema.parse({
      status: "HISTORY_RESULT_APPLIED",
      fake: true,
      source_task_id: "formal-fake-optimization",
      observation_id: "formal-fake-optimization:00000002:i2c_coarse",
      temporary_rollback: {
        i2c: { eq: 20 },
        fir: [0, 9, -28, 104, -8, 1, -3],
        upper_nlc: 6,
        lower_nlc: 6,
        pattern: "SSPRQ",
        bert_port: "port1",
        bert_channel: "0",
        dca_fingerprint: "fake",
      },
      i2c_readback: { eq: 22 },
      fir_readback: [0, 9, -28, 104, -8, 1, -3],
      measurement: { tdecq: 2.7, er: 5.3, fake: true },
    });

    expect(receipt.fake).toBe(true);
  });

  it("preserves point identity, global best, and round timing from /sweep-points", () => {
    const page = v11SweepPointPageSchema.parse(v11SweepPointsResponseFixture);

    expect(page.items[0]?.point_id).toBe("point-00000004");
    expect(page.best_point?.point_id).toBe("point-00000001");
    expect(page.items[0]?.raw_response?.requested_settle_s).toBe(0.1);
    expect(page.items[0]?.raw_response?.application?.receipt?.total_duration_s).toBe(0.0132585);
    expect(page.items[0]?.raw_response?.dca_raw_responses?.timing?.total_duration_s).toBe(8.0866074);
    expect(page.items[0]?.candidate_round_time_s).toBe(8.2);
    expect(page.items[0]?.screening_round_with_restore_s).toBe(8.212);
    expect(page.items[0]?.restore_status).toBe("VERIFIED");
  });

  it("preserves dataset status and safe task-scoped artifact metadata", () => {
    const document = {
      task_id: "task-dataset-1",
      task_type: "SWEEP",
      state: "RUNNING",
      runtime_mode: "FAKE",
      product_version: "1.1.0",
      schema_version: 2,
      b0_hash: "b".repeat(64),
      configuration_fingerprint: "c".repeat(64),
      config: {
        task_id: "task-dataset-1",
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
        seed: 1042,
        save_dataset: true,
      },
      measurement_status: "SUCCESS",
      dataset_status: "PARTIAL",
      dataset_error: "equalized eye unavailable",
      dataset_files: [{
        file_id: "point-1-density",
        kind: "EYE_DENSITY_NPY",
        name: "density.npy",
        url: "/api/v1.1/tasks/task-dataset-1/datasets/point-1/density.npy",
        size_bytes: 2048,
        point_id: "point-1",
        observation_id: null,
      }],
      created_at: "2026-08-04T08:00:00Z",
      updated_at: "2026-08-04T08:01:00Z",
    };

    const parsed = v11TaskDocumentSchema.parse(document);

    expect(parsed.config.save_dataset).toBe(true);
    expect(parsed.measurement_status).toBe("SUCCESS");
    expect(parsed.dataset_status).toBe("PARTIAL");
    expect(parsed.dataset_error).toBe("equalized eye unavailable");
    expect(parsed.dataset_files[0]).toMatchObject({
      kind: "EYE_DENSITY_NPY",
      point_id: "point-1",
      size_bytes: 2048,
    });
    expect(() => v11TaskDocumentSchema.parse({
      ...document,
      dataset_files: [{ ...document.dataset_files[0], url: "https://untrusted.example/eye.png" }],
    })).toThrow();
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
