import { z } from "zod";

export const runtimeModeSchema = z
  .enum(["FAKE", "READ_ONLY", "LIVE_TUNING", "LIVE"])
  .transform((mode) => (mode === "LIVE_TUNING" ? "LIVE" : mode));

export const v11RuntimeModeSchema = z.enum([
  "FAKE",
  "READ_ONLY",
  "REPLAY",
  "LIVE",
]);

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

export const v11TaskStateSchema = z.enum([
  "DRAFT",
  "PREFLIGHTED",
  "B0_FROZEN",
  "AUTHORIZED",
  "RUNNING",
  "PAUSED",
  "ABORTING",
  "ROLLING_BACK",
  "COMPLETED",
  "COMPLETED_NO_IMPROVEMENT",
  "ABORTED",
  "ABORTED_RECOVERED",
  "FAILED",
  "FAILED_INVALID_DCA_TAPS",
  "RECOVERY_REQUIRED",
]);

export const v11ChannelMappingSchema = z.object({
  i2c_channel: z.number().int().min(0).max(3),
  bert_port: z.string().min(1),
  bert_channel: z.string().min(1),
  dca_tdecq_source: z.literal("FUNC1"),
  dca_tdecq_equalizer_function: z.literal(1),
  dca_outer_er_source: z.literal("CHAN1A"),
  operator_confirmed: z.literal(true),
});

export const v11RuntimeCapabilitiesSchema = z.object({
  product_version: z.string().min(1),
  schema_version: z.number().int().positive(),
  runtime_mode: v11RuntimeModeSchema,
  campaign_types: z.array(taskTypeSchema),
  dynamic_b0: z.boolean(),
  full_campaign_authorization: z.boolean(),
  hardware_mutation: z.boolean(),
  fake: z.boolean(),
  live_ai_profile: z.literal("PILOT_3_THEN_FAST24").optional(),
  candidate_budget: z.number().int().positive().optional(),
  pilot_candidate_budget: z.number().int().positive().optional(),
  fast_candidate_budget: z.number().int().positive().optional(),
  coarse_candidate_budget: z.number().int().positive().optional(),
  fine_candidate_budget: z.number().int().positive().optional(),
  fir_first_candidate_budget: z.number().int().positive().optional(),
  fir_iterative_candidate_budget: z.number().int().positive().optional(),
  measurement_timeout_s: z.number().finite().positive().optional(),
  cleanup_grace_s: z.number().finite().positive().optional(),
  dataset_required: z.boolean().optional(),
  fir_authority: z.string().min(1).optional(),
  final_restore: z.string().min(1).optional(),
  profile_options: z.array(z.object({
    name: z.enum(["FAST", "STANDARD", "DEEP", "ADVANCED", "PILOT_3_THEN_FAST24"]),
    candidate_budget: z.number().int().positive(),
    max_alternating_rounds: z.number().int().positive(),
  })).min(1),
}).passthrough();

export const v11ParameterSelectionSchema = z.object({
  name: z.string().min(1),
  min: z.number().int(),
  max: z.number().int(),
  step: z.number().int().positive(),
});

const v11DatasetFileUrlSchema = z.string().refine(
  (value) => /^\/(?:api|artifacts)\//.test(value) && !value.includes(".."),
  "dataset file URL must be a safe same-origin API or artifact path",
);

export const v11MeasurementStatusSchema = z.enum([
  "PENDING",
  "RUNNING",
  "SUCCESS",
  "FAILED",
]);

export const v11DatasetStatusSchema = z.enum([
  "DISABLED",
  "PENDING",
  "RUNNING",
  "SUCCESS",
  "PARTIAL",
  "FAILED",
]);

export const v11DatasetFileSchema = z.object({
  file_id: z.string().min(1),
  kind: z.enum([
    "EYE_DENSITY_NPY",
    "RAW_EYE_PNG",
    "TDECQ_EQUALIZED_EYE_PNG",
    "MEASUREMENT_RECORD_JSON",
  ]),
  name: z.string().min(1),
  url: v11DatasetFileUrlSchema,
  size_bytes: z.number().int().nonnegative().nullable().optional(),
  point_id: z.string().min(1).nullable().optional(),
  observation_id: z.string().min(1).nullable().optional(),
});

const v11TaskConfigSchema = z.object({
  task_id: z.string().min(1),
  task_type: taskTypeSchema,
  channel_mapping: v11ChannelMappingSchema,
  parameters: z.array(v11ParameterSelectionSchema),
  depth: z.enum(["FAST", "STANDARD", "DEEP", "ADVANCED"]),
  candidate_budget: z.number().int().positive(),
  advanced_iteration_limit: z.number().int().positive().nullable().optional(),
  sweep_hardware: z.enum(["I2C_SWEEP", "BERT_FIR_SWEEP"]).nullable().optional(),
  sweep_mode: z.enum(["SINGLE_PARAMETER", "FULL_COMBINATION"]).nullable().optional(),
  seed: z.number().int(),
  max_points: z.number().int().positive().nullable().optional(),
  settle_time_s: z.number().finite().nonnegative().optional(),
  save_dataset: z.boolean(),
  live_ai_profile: z.literal("PILOT_3_THEN_FAST24").optional(),
  optimization_scope: z.literal("I2C_FIR_ALTERNATING").optional(),
  pilot_candidate_budget: z.number().int().positive().optional(),
  fast_candidate_budget: z.number().int().positive().optional(),
  coarse_candidate_budget: z.number().int().positive().optional(),
  fine_candidate_budget: z.number().int().positive().optional(),
  fir_first_candidate_budget: z.number().int().positive().optional(),
  fir_iterative_candidate_budget: z.number().int().positive().optional(),
}).passthrough();

export const v11TaskDocumentSchema = z.object({
  task_id: z.string().min(1),
  task_type: taskTypeSchema,
  state: v11TaskStateSchema,
  runtime_mode: v11RuntimeModeSchema,
  product_version: z.string().min(1),
  schema_version: z.number().int().positive(),
  b0_hash: z.string().length(64).nullable(),
  configuration_fingerprint: z.string().length(64),
  config: v11TaskConfigSchema,
  measurement_status: v11MeasurementStatusSchema,
  dataset_status: v11DatasetStatusSchema,
  dataset_error: z.string().min(1).nullable(),
  dataset_files: z.array(v11DatasetFileSchema),
  created_at: z.string().datetime({ offset: true }),
  updated_at: z.string().datetime({ offset: true }),
});

const v11ParameterBindingSchema = z.object({
  name: z.string().min(1),
  status: z.enum(["AVAILABLE", "UNAVAILABLE", "EXCLUDED"]),
  reason: z.string().nullable(),
  script_order: z.number().int().nullable(),
  current_value: z.number().int().nullable(),
  channel0_address: z.number().int().nullable(),
  channel_stride: z.number().int().nullable(),
  bit_start: z.number().int().nullable(),
  bits: z.number().int().nullable(),
  encoding_min: z.number().int().nullable(),
  encoding_max: z.number().int().nullable(),
});

const v11B0Schema = z.object({
  b0_id: z.string().min(1),
  version: z.number().int().positive(),
  captured_at: z.string().datetime({ offset: true }),
  i2c_fields: z.record(z.number().int()),
  i2c_raw_registers: z.array(z.tuple([z.number().int(), z.number().int()])),
  channel_mapping: v11ChannelMappingSchema.omit({ operator_confirmed: true }),
  fir: z.object({
    taps: z.array(z.number().int()).length(7),
    upper_nlc: z.number().int(),
    lower_nlc: z.number().int(),
    pattern: z.string().min(1),
    port: z.string().min(1),
    channel: z.string().min(1),
  }),
  dca_fingerprint: z.record(z.unknown()),
  hardware_identities: z.array(z.record(z.unknown())),
});

const v11PreflightEvidenceSchema = z.object({
  fake: z.boolean(),
  mutation_counts: z.object({
    bert_writes: z.literal(0),
    dca_writes: z.literal(0),
    i2c_page_select_writes: z.number().int().nonnegative(),
    i2c_tunable_register_writes: z.literal(0),
  }),
  targets: z.record(z.unknown()).optional(),
  raw_responses: z.record(z.unknown()).optional(),
}).superRefine((evidence, context) => {
  if (evidence.fake) return;
  const i2c = evidence.raw_responses?.i2c;
  if (!i2c || typeof i2c !== "object" || !("restore_verified" in i2c) || i2c.restore_verified !== true) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "READ_ONLY evidence must prove VD1124 page restoration",
      path: ["raw_responses", "i2c", "restore_verified"],
    });
  }
});

export const v11HardwareSnapshotSchema = z.object({
  snapshot_token: z.string().min(1),
  b0_hash: z.string().length(64),
  b0: v11B0Schema,
  mapping_hash: z.string().length(64),
  mapping: z.array(v11ParameterBindingSchema),
  evidence: v11PreflightEvidenceSchema,
});

export const v11MappingScanSchema = z.object({
  snapshot_token: z.string().min(1),
  mapping_hash: z.string().length(64),
  parameters: z.array(v11ParameterBindingSchema),
});

export const v11ParameterValidationSchema = z.object({
  snapshot_token: z.string().min(1),
  mapping_hash: z.string().length(64),
  parameters: z.array(v11ParameterSelectionSchema.extend({
    values: z.array(z.number().int()),
    script_order: z.number().int(),
  })),
});

export const v11SweepCountSchema = z.object({
  snapshot_token: z.string().min(1),
  mapping_hash: z.string().length(64),
  hardware: z.enum(["I2C_SWEEP", "BERT_FIR_SWEEP"]),
  mode: z.enum(["SINGLE_PARAMETER", "FULL_COMBINATION"]),
  seed: z.number().int(),
  point_count: z.number().int().nonnegative(),
  skipped_constraint_count: z.number().int().nonnegative(),
});

const v11SweepDurationSchema = z.number().finite().nonnegative();

export const v11SweepDcaTimingSchema = z.object({
  acquisition_duration_s: v11SweepDurationSchema.optional(),
  apply_duration_s: v11SweepDurationSchema.optional(),
  artifact_duration_s: v11SweepDurationSchema.optional(),
  query_duration_s: v11SweepDurationSchema.optional(),
  readback_duration_s: v11SweepDurationSchema.optional(),
  total_duration_s: v11SweepDurationSchema.optional(),
  wait_for_valid_duration_s: v11SweepDurationSchema.optional(),
}).passthrough();

export const v11SweepPointRawResponseSchema = z.object({
  point_id: z.string().min(1).optional(),
  execution_order: z.number().int().nonnegative().optional(),
  requested_settle_s: v11SweepDurationSchema.optional(),
  actual_settle_s: v11SweepDurationSchema.optional(),
  application: z.object({
    receipt: z.object({
      apply_duration_s: v11SweepDurationSchema.optional(),
      readback_duration_s: v11SweepDurationSchema.optional(),
      total_duration_s: v11SweepDurationSchema.optional(),
    }).passthrough().optional(),
  }).passthrough().optional(),
  dca_raw_responses: z.object({
    timing: v11SweepDcaTimingSchema.optional(),
  }).passthrough().optional(),
}).passthrough();

export const v11SweepPointSchema = z.object({
  point_id: z.string().min(1),
  original_index: z.number().int().nonnegative(),
  execution_index: z.number().int().nonnegative(),
  status: z.enum(["PLANNED", "COMPLETED", "INVALID_TDECQ", "SKIPPED_CONSTRAINT", "FAILED"]),
  requested: z.record(z.number().int()),
  applied: z.record(z.number().int()),
  readback: z.record(z.number().int()),
  tdecq: z.number().finite().nullable(),
  er: z.number().finite().nullable(),
  raw_response: v11SweepPointRawResponseSchema.nullable().catch(null),
  error_reason: z.string().nullable(),
  seed: z.number().int(),
  hardware: z.enum(["I2C_SWEEP", "BERT_FIR_SWEEP"]),
  mode: z.enum(["SINGLE_PARAMETER", "FULL_COMBINATION"]),
  completed_at: z.string().datetime({ offset: true }),
  write_readback_duration_s: v11SweepDurationSchema.nullable(),
  settle_duration_s: v11SweepDurationSchema.nullable(),
  dca_duration_s: v11SweepDurationSchema.nullable(),
  result_persistence_duration_s: v11SweepDurationSchema.nullable(),
  restore_duration_s: v11SweepDurationSchema.nullable(),
  candidate_round_time_s: v11SweepDurationSchema.nullable(),
  screening_round_with_restore_s: v11SweepDurationSchema.nullable(),
  restore_readback_result: z.record(z.number().int()).nullable(),
  restore_status: z.string().min(1),
});

export const v11SweepPointPageSchema = z.object({
  items: z.array(v11SweepPointSchema),
  total: z.number().int().nonnegative(),
  offset: z.number().int().nonnegative(),
  limit: z.number().int().positive(),
  best_point: v11SweepPointSchema.nullable(),
});

export const v11ObservationSchema = z.object({
  observation_id: z.string().min(1),
  sequence: z.number().int().positive(),
  phase: z.string().min(1),
  status: z.string().min(1),
  valid: z.boolean(),
  requested: z.record(z.number().int()),
  applied: z.record(z.number().int()),
  readback: z.record(z.number().int()),
  tdecq: z.number().finite().nullable(),
  er: z.number().finite().nullable(),
  raw_response: z.unknown(),
  error_reason: z.string().nullable(),
  created_at: z.string().datetime({ offset: true }),
});

export const v11TaskEventSchema = z.object({
  task_id: z.string().min(1),
  sequence: z.number().int().positive(),
  event_type: z.string().min(1),
  from_state: v11TaskStateSchema.nullable(),
  to_state: v11TaskStateSchema,
  payload: z.record(z.unknown()),
  product_version: z.string().min(1),
  schema_version: z.number().int().positive(),
  campaign_type: taskTypeSchema,
  runtime_mode: v11RuntimeModeSchema,
  b0_hash: z.string().length(64).nullable(),
  config_fingerprint: z.string().length(64),
  created_at: z.string().datetime({ offset: true }),
});

export const v11FinalReportSchema = v11TaskDocumentSchema.extend({
  observations: z.array(v11ObservationSchema),
  events: z.array(z.record(z.unknown())),
});

const v11HistoryHardwareStateSchema = z.object({
  i2c: z.record(z.number().int()),
  fir: z.array(z.number().int()).length(7),
  upper_nlc: z.number().int(),
  lower_nlc: z.number().int(),
  pattern: z.literal("SSPRQ"),
  bert_port: z.string().min(1),
  bert_channel: z.string().min(1),
  dca_fingerprint: z.string().min(1),
});

export const v11HistoryApplyReceiptSchema = z.object({
  status: z.literal("HISTORY_RESULT_APPLIED"),
  fake: z.boolean(),
  source_task_id: z.string().min(1),
  observation_id: z.string().min(1),
  temporary_rollback: v11HistoryHardwareStateSchema,
  i2c_readback: z.record(z.number().int()),
  fir_readback: z.array(z.number().int()).length(7),
  measurement: z.record(z.unknown()),
});

export const v11HistoryRecoveryResultSchema = z.object({
  recovery_id: z.string().min(1),
  task_id: z.string().min(1),
  status: z.literal("RESOLVED"),
  resolved_at: z.string().datetime({ offset: true }).nullable(),
});

export const v11BestStateSchema = z.object({
  observation_id: z.string().min(1),
  i2c: z.record(z.number().int()),
  fir: z.array(z.number().int()).length(7),
  upper_nlc: z.number().int(),
  lower_nlc: z.number().int(),
  pattern: z.literal("SSPRQ"),
  bert_port: z.string().min(1),
  bert_channel: z.string().min(1),
  dca_fingerprint: z.string().min(1),
  result: z.unknown(),
  updated_at: z.string().datetime({ offset: true }),
});

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
export type V11RuntimeMode = z.infer<typeof v11RuntimeModeSchema>;
export type TaskType = z.infer<typeof taskTypeSchema>;
export type ParameterName = z.infer<typeof parameterNameSchema>;
export type V11TaskState = z.infer<typeof v11TaskStateSchema>;
export type V11ChannelMapping = z.infer<typeof v11ChannelMappingSchema>;
export type V11RuntimeCapabilities = z.infer<typeof v11RuntimeCapabilitiesSchema>;
export type V11ParameterSelection = z.infer<typeof v11ParameterSelectionSchema>;
export type V11MeasurementStatus = z.infer<typeof v11MeasurementStatusSchema>;
export type V11DatasetStatus = z.infer<typeof v11DatasetStatusSchema>;
export type V11DatasetFile = z.infer<typeof v11DatasetFileSchema>;
export type V11TaskDocument = z.infer<typeof v11TaskDocumentSchema>;
export type V11HardwareSnapshot = z.infer<typeof v11HardwareSnapshotSchema>;
export type V11MappingScan = z.infer<typeof v11MappingScanSchema>;
export type V11ParameterValidation = z.infer<typeof v11ParameterValidationSchema>;
export type V11SweepCount = z.infer<typeof v11SweepCountSchema>;
export type V11SweepPointPage = z.infer<typeof v11SweepPointPageSchema>;
export type V11Observation = z.infer<typeof v11ObservationSchema>;
export type V11TaskEvent = z.infer<typeof v11TaskEventSchema>;
export type V11FinalReport = z.infer<typeof v11FinalReportSchema>;
export type V11HistoryApplyReceipt = z.infer<typeof v11HistoryApplyReceiptSchema>;
export type V11HistoryRecoveryResult = z.infer<typeof v11HistoryRecoveryResultSchema>;
export type V11OptimizationDepth = "FAST" | "STANDARD" | "DEEP" | "ADVANCED";
export type V11ProfileName = V11OptimizationDepth | "PILOT_3_THEN_FAST24";
export type V11SweepHardware = "I2C_SWEEP" | "BERT_FIR_SWEEP";
export type V11SweepMode = "SINGLE_PARAMETER" | "FULL_COMBINATION";

export interface V11CreateTaskRequest {
  task_id: string;
  task_type: TaskType;
  channel_mapping: V11ChannelMapping;
  parameters: V11ParameterSelection[];
  depth: V11OptimizationDepth;
  candidate_budget: number;
  advanced_iteration_limit?: number | null;
  sweep_hardware?: V11SweepHardware;
  sweep_mode?: V11SweepMode;
  seed: number;
  max_points?: number;
  settle_time_s?: number;
  save_dataset: boolean;
  live_ai_profile?: "PILOT_3_THEN_FAST24";
  optimization_scope?: "I2C_FIR_ALTERNATING";
  pilot_candidate_budget?: number;
  fast_candidate_budget?: number;
  coarse_candidate_budget?: number;
  fine_candidate_budget?: number;
  fir_first_candidate_budget?: number;
  fir_iterative_candidate_budget?: number;
}

export interface V11SweepCountRequest {
  snapshot_token: string;
  hardware: V11SweepHardware;
  mode: V11SweepMode;
  parameters: V11ParameterSelection[];
  seed: number;
  max_points: number;
}

export interface V11SaveB0Request {
  snapshot_token: string;
  operator_confirmed: true;
}

export interface V11AuthorizeTaskRequest {
  authorization_id: string;
  expires_at: string;
  operator_confirmed: true;
}

export interface V11ApplyHistoryResultRequest {
  source_task_id: string;
  observation_id: string;
}
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
