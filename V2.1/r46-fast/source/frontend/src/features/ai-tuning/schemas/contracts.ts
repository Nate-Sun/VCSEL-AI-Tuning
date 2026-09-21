import { z } from "zod";

export const runtimeModeSchema = z
  .enum(["FAKE", "READ_ONLY", "LIVE_TUNING", "LIVE"])
  .transform((mode) => (mode === "LIVE_TUNING" ? "LIVE" : mode));

export const taskTypeSchema = z.enum(["AI_OPTIMIZATION", "SWEEP"]);

export const safeParameterNames = [
  "ibias_dac",
  "vref",
  "vbias_ref",
  "vga_cur",
  "res_gain",
  "res_bw",
  "gain_ctrl",
  "efin_cur",
  "eq",
] as const;

export const parameterNameSchema = z.enum(safeParameterNames);

export const versionMetadataSchema = z.object({
  product_version: z.string().min(1),
  schema_version: z.string().min(1),
  runtime_mode: runtimeModeSchema,
  campaign_type: taskTypeSchema,
});

export const discreteGridSchema = z
  .object({
    values: z.array(z.number().int()),
    b0_outside: z.boolean(),
  })
  .strict();

export const hardwareSnapshotSchema = z.object({
  snapshot_id: z.string().min(1),
  dca: z.object({
    identity: z.string().min(1),
    source: z.string().min(1),
    wavelength: z.string().min(1),
    pattern: z.string().min(1),
    trigger: z.string().min(1),
    tdecq_source: z.string().min(1),
    er_source: z.string().min(1),
    equalizer_taps: z.array(z.number().finite()),
  }),
  bert: z.object({
    identity: z.string().min(1),
    port: z.string().min(1),
    pattern: z.string().min(1),
    fir: z.tuple([
      z.number().int(), z.number().int(), z.number().int(), z.number().int(),
      z.number().int(), z.number().int(), z.number().int(),
    ]),
    upper_nlc: z.number().int().nonnegative(),
    lower_nlc: z.number().int().nonnegative(),
  }),
  i2c: z.object({
    identity: z.string().min(1),
    bus: z.string().min(1),
    values: z.record(z.number().int()),
  }),
  simulated: z.boolean().optional(),
});

export const scannedParameterSchema = z.object({
  name: parameterNameSchema,
  current: z.number().int(),
  address_summary: z.string().min(1),
  encoding_range: z.string().min(1),
  available: z.boolean(),
  unavailable_reason: z.string().nullable(),
});

export const configurationScanSchema = z.object({
  parameters: z.array(scannedParameterSchema),
  rejected_names: z.array(z.string()).default([]),
  simulated: z.boolean().optional(),
});

export const rangeValidationSchema = z.object({
  grids: z.record(discreteGridSchema),
  simulated: z.boolean().optional(),
});

export const baselineResultSchema = z.object({
  baseline_id: z.string().min(1),
  status: z.literal("VERIFIED"),
  simulated: z.boolean().optional(),
});

export const sweepCountSchema = z.object({
  original_count: z.number().int().nonnegative(),
  legal_count: z.number().int().nonnegative(),
  skipped_count: z.number().int().nonnegative(),
  limit: z.number().int().positive(),
  over_limit: z.boolean(),
  simulated: z.boolean().optional(),
});

export const apiErrorSchema = z.object({
  detail: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
    fields: z.record(z.unknown()).optional(),
  }),
});

export const campaignStateSchema = z.enum([
  "DRAFT",
  "PREFLIGHT",
  "AWAIT_BASELINE_APPROVAL",
  "RESTORE_B0",
  "BASELINE_MEASUREMENT",
  "AWAIT_COARSE_APPROVAL",
  "I2C_SCREENING",
  "I2C_COARSE",
  "AWAIT_FINE_APPROVAL",
  "I2C_FINE",
  "AWAIT_FIR_APPROVAL",
  "FIR_APPLY",
  "I2C_RETUNE",
  "FINAL_VERIFICATION",
  "COMPLETED",
  "PAUSED",
  "ABORTING",
  "ROLLING_BACK",
  "FAILED",
  "RECOVERY_REQUIRED",
]);

export const authorizationResultSchema = z.object({
  task_id: z.string().min(1),
  campaign_id: z.string().min(1),
  task_type: taskTypeSchema,
  state: campaignStateSchema,
  authorization_id: z.string().min(1),
  simulated: z.boolean().optional(),
});

export const finalOutcomeSchema = z.enum([
  "SUCCESS",
  "PARTIAL_CONVERGENCE",
  "NO_IMPROVEMENT",
  "COMPLETED_NO_IMPROVEMENT",
  "FAILED_INVALID_DCA_TAPS",
  "FAILED",
  "RECOVERY_REQUIRED",
]);

const finiteMetricSchema = z.number().finite().nullable();
const metricMapSchema = z.record(finiteMetricSchema);
const registerMapSchema = z.record(z.number().int().nonnegative());

export const sweepPointSchema = z.object({
  point_id: z.string().min(1),
  original_index: z.number().int().nonnegative(),
  execution_index: z.number().int().nonnegative(),
  status: z.enum(["VALID", "INVALID_TDECQ", "SKIPPED_CONSTRAINT", "ERROR"]),
  parameters: registerMapSchema,
  requested: registerMapSchema,
  applied: registerMapSchema,
  readback: registerMapSchema,
  tdecq_db: finiteMetricSchema,
  er_db: finiteMetricSchema,
  is_best: z.boolean(),
  error_reason: z.string().optional(),
});

export const sweepPageSchema = z.object({
  items: z.array(sweepPointSchema),
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
  total: z.number().int().nonnegative(),
  summary: z.object({
    hardware: z.enum(["I2C", "BERT_FIR"]),
    mode: z.enum(["SINGLE", "FULL"]),
    seed: z.number().int().nullable(),
    completed_count: z.number().int().nonnegative(),
    legal_count: z.number().int().nonnegative(),
    skipped_count: z.number().int().nonnegative(),
    total_count: z.number().int().nonnegative(),
    remaining_count: z.number().int().nonnegative(),
    current_original_index: z.number().int().nonnegative().nullable(),
    current_execution_index: z.number().int().nonnegative().nullable(),
    preflight_allowed: z.boolean(),
    b0_restore_status: z.string().min(1),
    hardware_restored: z.boolean(),
    devices: z.object({ dca: z.string(), i2c: z.string(), bert: z.string() }),
    dca_drift: z.object({
      detected: z.boolean(),
      original: z.record(z.string()),
      current: z.record(z.string()),
    }),
  }).optional(),
});

export const firSettingsSchema = z.object({
  taps: z.tuple([
    z.number().int(),
    z.number().int(),
    z.number().int(),
    z.number().int().positive(),
    z.number().int(),
    z.number().int(),
    z.number().int(),
  ]),
  upper_nlc: z.number().int().nonnegative(),
  lower_nlc: z.number().int().nonnegative(),
});

export const recoverySchema = z.object({
  campaign_id: z.string().min(1),
  task_type: taskTypeSchema,
  interrupted_at: z.string().datetime({ offset: true }),
  last_completed_point: z.string().nullable(),
  last_known_i2c: registerMapSchema,
  last_known_fir: firSettingsSchema.nullable(),
  baseline: z.object({
    i2c: registerMapSchema,
    fir: firSettingsSchema.nullable(),
  }),
  error_reason: z.string().min(1),
  rollback_error: z.string().nullable(),
  devices: z.object({
    dca: z.boolean(),
    i2c: z.boolean(),
    bert: z.boolean(),
  }),
  restore_verified: z.boolean(),
});

export const historyApplyStageSchema = z.object({
  name: z.enum([
    "DEVICE_VERIFICATION",
    "SNAPSHOT_CURRENT_STATE",
    "APPLY_I2C",
    "I2C_READBACK",
    "APPLY_FIR",
    "BERT_READBACK",
    "DCA_MEASUREMENT",
    "COMPLETE_OR_ROLLBACK",
  ]),
  status: z.enum(["PENDING", "RUNNING", "COMPLETE", "FAILED", "ROLLED_BACK"]),
});

export const historyApplySchema = z.object({
  operation_id: z.string().min(1),
  status: z.enum([
    "RUNNING",
    "HISTORY_RESULT_APPLIED",
    "FAILED",
    "RECOVERY_REQUIRED",
  ]),
  stages: z.array(historyApplyStageSchema),
  rollback_status: z.string().nullable(),
});

export const i2cSettingsSchema = z.object({
  ibias_dac: z.number().int().nonnegative(),
  vref: z.number().int().nonnegative(),
  vbias_ref: z.number().int().nonnegative(),
  vga_cur: z.number().int().nonnegative(),
  res_gain: z.number().int().nonnegative(),
  res_bw: z.number().int().nonnegative(),
  gain_ctrl: z.number().int().nonnegative(),
  efin_cur: z.number().int().nonnegative(),
  eq: z.number().int().nonnegative(),
});

export const appliedSettingsSchema = z.object({
  requested_i2c: registerMapSchema.default({}),
  quantized_i2c: registerMapSchema.default({}),
  applied_i2c: registerMapSchema.default({}),
  readback_i2c: registerMapSchema.default({}),
  requested_fir: firSettingsSchema.nullable().default(null),
  applied_fir: firSettingsSchema.nullable().default(null),
  readback_fir: firSettingsSchema.nullable().default(null),
});

export const observationSchema = z.object({
  observation_id: z.string().min(1),
  campaign_id: z.string().min(1),
  candidate_id: z.string().min(1).nullable(),
  stage: campaignStateSchema,
  settings: appliedSettingsSchema,
  metrics: metricMapSchema.default({}),
  valid: z.boolean(),
  feasible: z.boolean(),
  elapsed_s: z.number().finite().nonnegative(),
  artifacts: z.array(z.string()).default([]),
  measured_at: z.string().datetime({ offset: true }),
});

export const stageSummarySchema = z.object({
  stage: campaignStateSchema,
  convergence_reason: z.string(),
  best_observation: observationSchema.nullable().default(null),
  candidates_used: z.number().int().nonnegative(),
  elapsed_s: z.number().finite().nonnegative(),
  warnings: z.array(z.string()).default([]),
});

export const campaignSchema = z.object({
  campaign_id: z.string().min(1),
  product_version: z.string().min(1).default("1.0.0"),
  schema_version: z.string().min(1).default("v1"),
  runtime_mode: runtimeModeSchema.optional(),
  campaign_type: taskTypeSchema.default("AI_OPTIMIZATION"),
  baseline_id: z.string().min(1),
  baseline_hash: z.string().regex(/^[0-9a-f]{64}$/),
  config_hash: z.string().regex(/^[0-9a-f]{64}$/),
  authorization_mode: z.enum(["PER_STAGE", "PER_CANDIDATE", "FULL_CAMPAIGN"]),
  state: campaignStateSchema,
  stage_summaries: z.array(stageSummarySchema).default([]),
  result: finalOutcomeSchema.nullable(),
  created_at: z.string().datetime({ offset: true }),
  updated_at: z.string().datetime({ offset: true }),
  optimization_summary: z.object({
    current_round: z.number().int().positive(),
    max_rounds: z.number().int().positive(),
    pass_kind: z.enum(["FIRST_PASS", "ITERATIVE"]),
    candidates_used: z.number().int().nonnegative(),
    fir_unchanged_rounds: z.number().int().nonnegative(),
    dca_equalizer_valid: z.boolean(),
    dca_equalizer_taps: z.array(z.number().finite()),
    convergence_reason: z.string(),
    rollback_count: z.number().int().nonnegative(),
    completed_phase_count: z.number().int().nonnegative(),
    active_phase_index: z.number().int().nonnegative(),
    b0_fir_main: z.number().int(),
    upper_nlc: z.number().int().nonnegative(),
    lower_nlc: z.number().int().nonnegative(),
    rollback_target_status: z.string().min(1),
  }).optional(),
});

const componentHealthSchema = z
  .object({
    ok: z.boolean(),
    status: z.string().optional(),
    message: z.string().optional(),
  })
  .passthrough();

export const runtimeHealthSchema = z
  .object({
    status: z.string(),
    runtime_mode: runtimeModeSchema,
    store: componentHealthSchema.optional(),
    repository: componentHealthSchema.optional(),
    hardware: z
      .object({
        dca: componentHealthSchema.optional(),
        i2c: componentHealthSchema.optional(),
        bert: componentHealthSchema.optional(),
      })
      .optional(),
  })
  .passthrough();

export const capabilitySchema = z
  .object({
    runtime_mode: runtimeModeSchema,
    hardware_connected: z.boolean(),
    read_only: z.boolean(),
    exhaustive_ready: z.boolean().optional(),
    tdecq_bayesian_ready: z.boolean().optional(),
    hierarchical_ready: z.boolean().optional(),
    fir_proposer_ready: z.boolean().optional(),
    live_tuning_enabled: z.boolean(),
    authorization_modes: z.array(z.string()).default([]),
    FULL_CAMPAIGN: z.boolean(),
    profile_options: z.array(z.object({
      name: z.enum(["FAST", "STANDARD", "DEEP", "ADVANCED"]),
      candidate_budget: z.number().int().positive(),
      max_alternating_rounds: z.number().int().positive(),
    })).optional(),
  })
  .passthrough();

export const candidateSchema = z.object({
  candidate_id: z.string().min(1),
  stage: campaignStateSchema,
  requested_i2c: registerMapSchema.default({}),
  requested_fir: firSettingsSchema.nullable().default(null),
  predicted_metrics: metricMapSchema.default({}),
  model_version: z.string(),
  reason: z.string(),
});

export const campaignEventSchema = z.object({
  campaign_id: z.string().min(1),
  created_at: z.string().datetime({ offset: true }),
  event_type: z.string().min(1),
  from_state: campaignStateSchema.nullable(),
  payload: z.record(z.unknown()),
  sequence: z.number().int().positive(),
  to_state: campaignStateSchema.nullable(),
});

const metricSummarySchema = z.object({
  median: finiteMetricSchema.optional(),
  std: finiteMetricSchema.optional(),
  min: finiteMetricSchema.optional(),
  max: finiteMetricSchema.optional(),
});

export const campaignReportSchema = z
  .object({
    campaign_id: z.string().min(1),
    product_version: z.string().min(1).default("1.0.0"),
    schema_version: z.string().min(1).default("v1"),
    runtime_mode: runtimeModeSchema.optional(),
    campaign_type: taskTypeSchema.default("AI_OPTIMIZATION"),
    outcome: finalOutcomeSchema.nullable().optional(),
    convergence_reason: z.string().optional(),
    measurement_validity: z.string().optional(),
    er_db: metricSummarySchema.optional(),
    tdecq_db: metricSummarySchema.optional(),
    baseline: appliedSettingsSchema.optional(),
    final: appliedSettingsSchema.optional(),
    rollback_history: z.array(z.record(z.unknown())).default([]),
    artifacts: z.array(z.string()).default([]),
    final_measurements: z.array(z.object({ tdecq_db: finiteMetricSchema, er_db: finiteMetricSchema })).max(3).optional(),
    alternating_rounds: z.number().int().nonnegative().optional(),
  })
  .passthrough();

export function safeArtifactHref(value: string): string | null {
  if (/^\/(?:api|artifacts)\//.test(value) && !value.includes("..")) {
    return value;
  }
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? value : null;
  } catch {
    return null;
  }
}

export type RuntimeMode = z.infer<typeof runtimeModeSchema>;
export type TaskType = z.infer<typeof taskTypeSchema>;
export type ParameterName = z.infer<typeof parameterNameSchema>;
export type VersionMetadata = z.infer<typeof versionMetadataSchema>;
export type DiscreteGrid = z.infer<typeof discreteGridSchema>;
export type HardwareSnapshot = z.infer<typeof hardwareSnapshotSchema>;
export type ConfigurationScan = z.infer<typeof configurationScanSchema>;
export type RangeValidation = z.infer<typeof rangeValidationSchema>;
export type BaselineResult = z.infer<typeof baselineResultSchema>;
export type SweepCount = z.infer<typeof sweepCountSchema>;
export type SweepPoint = z.infer<typeof sweepPointSchema>;
export type SweepPage = z.infer<typeof sweepPageSchema>;
export type Recovery = z.infer<typeof recoverySchema>;
export type HistoryApply = z.infer<typeof historyApplySchema>;
export type RuntimeHealth = z.infer<typeof runtimeHealthSchema>;
export type Capability = z.infer<typeof capabilitySchema>;
export type Campaign = z.infer<typeof campaignSchema>;
export type CampaignState = z.infer<typeof campaignStateSchema>;
export type Observation = z.infer<typeof observationSchema>;
export type CampaignEvent = z.infer<typeof campaignEventSchema>;
export type CampaignReport = z.infer<typeof campaignReportSchema>;
export type AppliedSettings = z.infer<typeof appliedSettingsSchema>;
export type FirSettings = z.infer<typeof firSettingsSchema>;
