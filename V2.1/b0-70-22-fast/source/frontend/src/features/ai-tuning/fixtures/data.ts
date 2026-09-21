import type {
  Campaign,
  CampaignReport,
  Capability,
  Observation,
  RuntimeHealth,
} from "../schemas/contracts";

const sha = "a".repeat(64);
const requestedI2c = {
  ibias_dac: 72,
  vref: 15,
  vbias_ref: 15,
  vga_cur: 14,
  res_gain: 6,
  res_bw: 0,
  gain_ctrl: 5,
  efin_cur: 15,
  eq: 23,
};
const readbackI2c = { ...requestedI2c, ibias_dac: 71, eq: 22 };

export const runtimeFixture: RuntimeHealth = {
  status: "ok",
  runtime_mode: "FAKE",
  store: { ok: true },
  repository: { ok: true },
  hardware: {
    dca: { ok: true, status: "SIMULATED", message: "FAKE DCA session" },
    i2c: { ok: true, status: "SIMULATED", message: "FAKE I2C adapter" },
    bert: { ok: true, status: "SIMULATED", message: "FAKE BERT lane" },
  },
};

export const capabilityFixture: Capability = {
  runtime_mode: "FAKE",
  hardware_connected: false,
  read_only: false,
  exhaustive_ready: true,
  tdecq_bayesian_ready: true,
  hierarchical_ready: false,
  fir_proposer_ready: true,
  live_tuning_enabled: false,
  authorization_modes: ["PER_STAGE", "FULL_CAMPAIGN"],
  FULL_CAMPAIGN: true,
  profile_options: [
    { name: "FAST", candidate_budget: 24, max_alternating_rounds: 1 },
    { name: "STANDARD", candidate_budget: 64, max_alternating_rounds: 3 },
    { name: "DEEP", candidate_budget: 128, max_alternating_rounds: 5 },
    { name: "ADVANCED", candidate_budget: 192, max_alternating_rounds: 5 },
  ],
};

export const campaignsFixture: Campaign[] = [
  {
    campaign_id: "campaign-live",
    product_version: "1.1.0",
    schema_version: "v1.1",
    runtime_mode: "FAKE",
    campaign_type: "AI_OPTIMIZATION",
    baseline_id: "b0-v1",
    baseline_hash: sha,
    config_hash: sha,
    authorization_mode: "PER_STAGE",
    state: "I2C_COARSE",
    stage_summaries: [],
    result: null,
    created_at: "2026-07-24T02:30:00Z",
    updated_at: "2026-07-24T03:18:00Z",
    optimization_summary: {
      current_round: 2,
      max_rounds: 4,
      pass_kind: "ITERATIVE",
      candidates_used: 18,
      fir_unchanged_rounds: 0,
      dca_equalizer_valid: true,
      dca_equalizer_taps: [0, 3, -1],
      convergence_reason: "coarse search in progress",
      rollback_count: 0,
      completed_phase_count: 2,
      active_phase_index: 2,
      b0_fir_main: 104,
      upper_nlc: 6,
      lower_nlc: 6,
      rollback_target_status: "VERIFIED SNAPSHOT",
    },
  },
  {
    campaign_id: "campaign-result",
    product_version: "1.0.0",
    schema_version: "v1",
    runtime_mode: "FAKE",
    campaign_type: "AI_OPTIMIZATION",
    baseline_id: "b0-v1",
    baseline_hash: sha,
    config_hash: sha,
    authorization_mode: "PER_STAGE",
    state: "COMPLETED",
    stage_summaries: [],
    result: "PARTIAL_CONVERGENCE",
    created_at: "2026-07-23T09:00:00Z",
    updated_at: "2026-07-23T09:48:00Z",
  },
];

const settings = {
  requested_i2c: requestedI2c,
  quantized_i2c: requestedI2c,
  applied_i2c: requestedI2c,
  readback_i2c: readbackI2c,
  requested_fir: {
    taps: [0, 8, -27, 105, -7, 1, -3] as [number, number, number, number, number, number, number],
    upper_nlc: 6,
    lower_nlc: 6,
  },
  applied_fir: {
    taps: [0, 8, -27, 105, -7, 1, -3] as [number, number, number, number, number, number, number],
    upper_nlc: 6,
    lower_nlc: 6,
  },
  readback_fir: {
    taps: [0, 8, -27, 104, -7, 1, -3] as [number, number, number, number, number, number, number],
    upper_nlc: 6,
    lower_nlc: 6,
  },
};

export const observationsFixture: Observation[] = [
  {
    observation_id: "obs-b0",
    campaign_id: "campaign-live",
    candidate_id: null,
    stage: "BASELINE_MEASUREMENT",
    settings,
    metrics: { er_db: 2.42, tdecq_db: 4.18 },
    valid: true,
    feasible: true,
    elapsed_s: 8.2,
    artifacts: ["/artifacts/campaign-live/obs-b0.json"],
    measured_at: "2026-07-24T02:42:00Z",
  },
  {
    observation_id: "obs-02",
    campaign_id: "campaign-live",
    candidate_id: "candidate-02",
    stage: "I2C_COARSE",
    settings,
    metrics: { er_db: 2.74, tdecq_db: null },
    valid: false,
    feasible: true,
    elapsed_s: 11.4,
    artifacts: ["/artifacts/campaign-live/obs-02.json"],
    measured_at: "2026-07-24T03:12:00Z",
  },
  {
    observation_id: "obs-03",
    campaign_id: "campaign-live",
    candidate_id: "candidate-03",
    stage: "I2C_COARSE",
    settings,
    metrics: { er_db: 3.08, tdecq_db: 3.36 },
    valid: true,
    feasible: true,
    elapsed_s: 10.1,
    artifacts: ["/artifacts/campaign-live/obs-03.json"],
    measured_at: "2026-07-24T03:18:00Z",
  },
];

export const resultObservationsFixture: Observation[] = observationsFixture.map(
  (observation) => ({ ...observation, campaign_id: "campaign-result" }),
);

export const liveReportFixture: CampaignReport = {
  campaign_id: "campaign-live",
  product_version: "1.1.0",
  schema_version: "v1.1",
  campaign_type: "AI_OPTIMIZATION",
  runtime_mode: "FAKE",
  outcome: null,
  convergence_reason: "coarse search in progress",
  measurement_validity: "PROVISIONAL",
  rollback_history: [],
  artifacts: ["/api/campaigns/campaign-live/report"],
};

export const resultReportFixture: CampaignReport = {
  campaign_id: "campaign-result",
  product_version: "1.0.0",
  schema_version: "v1",
  campaign_type: "AI_OPTIMIZATION",
  runtime_mode: "FAKE",
  outcome: "PARTIAL_CONVERGENCE",
  convergence_reason: "candidate budget exhausted before constrained ER target",
  measurement_validity: "MEDIAN_CONFIRMED",
  er_db: { median: 2.86, std: 0.09, min: 2.72, max: 2.96 },
  tdecq_db: { median: 3.42, std: 0.08, min: 3.34, max: 3.51 },
  final_measurements: [
    { tdecq_db: 3.34, er_db: 2.72 },
    { tdecq_db: 3.42, er_db: 2.86 },
    { tdecq_db: 3.51, er_db: 2.96 },
  ],
  alternating_rounds: 4,
  baseline: settings,
  final: settings,
  rollback_history: [
    { status: "VERIFIED", reason: "readback mismatch recovered", sequence: 18 },
  ],
  artifacts: [
    "/api/campaigns/campaign-result/report",
    "/artifacts/campaign-result/observations.csv",
  ],
};

export const eventFixtures = [
  {
    campaign_id: "campaign-live",
    created_at: "2026-07-24T03:12:00Z",
    event_type: "OBSERVATION_RECORDED",
    from_state: "I2C_COARSE",
    payload: { observation_id: "obs-02" },
    sequence: 1,
    to_state: "I2C_COARSE",
  },
  {
    campaign_id: "campaign-live",
    created_at: "2026-07-24T03:18:00Z",
    event_type: "CANDIDATE_PROPOSED",
    from_state: "I2C_COARSE",
    payload: { candidate_id: "candidate-03", reason: "higher predicted ER" },
    sequence: 2,
    to_state: "I2C_COARSE",
  },
] as const;
