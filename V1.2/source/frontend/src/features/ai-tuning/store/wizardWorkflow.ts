import type {
  TaskType,
  V11CreateTaskRequest,
  V11OptimizationDepth,
  V11ProfileName,
  V11ParameterSelection,
} from "../schemas/contracts";

interface SnapshotChannelMapping {
  i2c_channel: number;
  bert_port: string;
  bert_channel: string;
  dca_tdecq_source: "FUNC1";
  dca_tdecq_equalizer_function: 1;
  dca_outer_er_source: "CHAN1A";
}

interface BuildTaskOptions {
  taskId: string;
  taskType: TaskType;
  channelMapping: SnapshotChannelMapping;
  mappingConfirmed: boolean;
  parameters: V11ParameterSelection[];
  depth: V11ProfileName;
  candidateBudget: number;
  advancedIterationLimit?: number | null;
  sweepHardware: "I2C";
  sweepMode: "SINGLE" | "FULL" | null;
  sweepSeed: number | null;
  maxPoints: number;
  settleTimeS: number;
  saveDataset: boolean;
}

export function buildV11CreateTaskRequest(options: BuildTaskOptions): V11CreateTaskRequest {
  if (!options.mappingConfirmed) {
    throw new Error("Operator channel-mapping confirmation is required");
  }
  const staged = options.depth === "PILOT_3_THEN_FAST24";
  const normalizedDepth: V11OptimizationDepth =
    options.depth === "PILOT_3_THEN_FAST24" ? "FAST" : options.depth;
  const common = {
    task_id: options.taskId,
    task_type: options.taskType,
    channel_mapping: {
      ...options.channelMapping,
      operator_confirmed: true as const,
    },
    parameters: options.parameters,
    depth: normalizedDepth,
    candidate_budget: options.candidateBudget,
    ...(options.depth === "ADVANCED"
      ? { advanced_iteration_limit: options.advancedIterationLimit }
      : {}),
    seed: options.sweepSeed ?? 0,
    save_dataset: staged ? true : options.saveDataset,
  } satisfies V11CreateTaskRequest;

  if (options.taskType === "AI_OPTIMIZATION") {
    return staged
      ? {
          ...common,
          live_ai_profile: "PILOT_3_THEN_FAST24",
          optimization_scope: "I2C_FIR_ALTERNATING",
          pilot_candidate_budget: 3,
          fast_candidate_budget: 24,
          coarse_candidate_budget: 6,
          fine_candidate_budget: 16,
          fir_first_candidate_budget: 1,
          fir_iterative_candidate_budget: 1,
          settle_time_s: 0.1,
        }
      : common;
  }
  if (options.sweepMode === null || options.sweepSeed === null) {
    throw new Error("Sweep mode and seed are required");
  }
  if (!Number.isFinite(options.settleTimeS) || options.settleTimeS < 0) {
    throw new Error("settle_time_s must be a finite nonnegative number");
  }
  return {
    ...common,
    sweep_hardware: "I2C_SWEEP",
    sweep_mode: options.sweepMode === "SINGLE" ? "SINGLE_PARAMETER" : "FULL_COMBINATION",
    seed: options.sweepSeed,
    max_points: options.maxPoints,
    settle_time_s: options.settleTimeS,
  };
}

export function buildV11AuthorizationRequest(taskId: string, issuedAt = new Date()) {
  const expiresAt = new Date(issuedAt.getTime() + 30 * 60 * 1000);
  return {
    authorization_id: `auth-${taskId}-${issuedAt.getTime()}`,
    expires_at: expiresAt.toISOString(),
    operator_confirmed: true as const,
  };
}

export function isCurrentSnapshotApproved(
  approvedSnapshotToken: string | null,
  currentSnapshotToken: string | null,
): boolean {
  return currentSnapshotToken !== null && approvedSnapshotToken === currentSnapshotToken;
}

export function createV11TaskId(issuedAt = new Date()): string {
  return `task-${issuedAt.getTime()}`;
}
