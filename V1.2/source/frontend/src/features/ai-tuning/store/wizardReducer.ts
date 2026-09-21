import type { ParameterName, TaskType } from "../schemas/contracts";

export interface HardwareMapping {
  i2c_channel: number;
  bert_port: string;
  bert_channel: string;
  dca_tdecq_source: "FUNC1";
  dca_tdecq_equalizer_function: 1;
  dca_outer_er_source: "CHAN1A";
  operator_confirmed: boolean;
}

export const supportedWizardSweepHardware = ["I2C"] as const;
type SupportedWizardSweepHardware = (typeof supportedWizardSweepHardware)[number];
export const DEFAULT_SWEEP_SETTLE_TIME_S = 0.1;

export function isValidSettleTime(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

export interface WizardState {
  step: number;
  taskType: TaskType;
  taskId: string | null;
  mapping: HardwareMapping;
  selectedParameters: ParameterName[];
  ranges: Record<ParameterName, { min: number; max: number; step: number }>;
  profile: "FAST" | "STANDARD" | "DEEP" | "ADVANCED" | "PILOT_3_THEN_FAST24";
  sweepHardware: SupportedWizardSweepHardware;
  sweepMode: "SINGLE" | "FULL" | null;
  sweepSeed: number | null;
  maxPoints: number;
  settleTimeS: number;
  saveDataset: boolean;
}

export const initialWizardState: WizardState = {
  step: 1,
  taskType: "AI_OPTIMIZATION",
  taskId: null,
  mapping: {
    i2c_channel: 0,
    bert_port: "port1",
    bert_channel: "0",
    dca_tdecq_source: "FUNC1",
    dca_tdecq_equalizer_function: 1,
    dca_outer_er_source: "CHAN1A",
    operator_confirmed: false,
  },
  selectedParameters: ["ibias_dac", "vref", "eq"],
  ranges: {
    ibias_dac: { min: 60, max: 84, step: 3 },
    vref: { min: 0, max: 30, step: 3 },
    vbias_ref: { min: 0, max: 30, step: 3 },
    vga_cur: { min: 0, max: 30, step: 3 },
    res_gain: { min: 0, max: 30, step: 3 },
    res_bw: { min: 0, max: 30, step: 3 },
    gain_ctrl: { min: 0, max: 30, step: 3 },
    efin_cur: { min: 0, max: 30, step: 3 },
    eq: { min: 0, max: 30, step: 3 },
  },
  profile: "STANDARD",
  sweepHardware: "I2C",
  sweepMode: null,
  sweepSeed: null,
  maxPoints: 64,
  settleTimeS: DEFAULT_SWEEP_SETTLE_TIME_S,
  saveDataset: false,
};

type WizardAction =
  | { type: "SELECT_TASK_TYPE"; value: TaskType }
  | { type: "SET_TASK_ID"; value: string }
  | { type: "SET_STEP"; value: number }
  | { type: "SET_MAPPING"; value: HardwareMapping }
  | { type: "TOGGLE_PARAMETER"; value: ParameterName }
  | { type: "SET_PARAMETERS"; value: ParameterName[] }
  | { type: "SET_RANGE"; name: ParameterName; field: "min" | "max" | "step"; value: number }
  | { type: "SET_PROFILE"; value: WizardState["profile"] }
  | { type: "SET_SWEEP_HARDWARE"; value: SupportedWizardSweepHardware }
  | { type: "SET_SWEEP_MODE"; value: "SINGLE" | "FULL" }
  | { type: "SET_SWEEP_SEED"; value: number }
  | { type: "SET_MAX_POINTS"; value: number }
  | { type: "SET_SETTLE_TIME"; value: number }
  | { type: "SET_SAVE_DATASET"; value: boolean };

export function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case "SELECT_TASK_TYPE":
      return action.value === "AI_OPTIMIZATION"
        ? { ...state, taskType: action.value, sweepMode: null, sweepSeed: null }
        : { ...state, taskType: action.value, sweepMode: state.sweepMode ?? "SINGLE", sweepSeed: state.sweepSeed ?? 1042 };
    case "SET_TASK_ID": return { ...state, taskId: action.value };
    case "SET_STEP": return { ...state, step: Math.min(8, Math.max(1, action.value)) };
    case "SET_MAPPING": return { ...state, mapping: action.value };
    case "TOGGLE_PARAMETER":
      return { ...state, selectedParameters: state.selectedParameters.includes(action.value) ? state.selectedParameters.filter((name) => name !== action.value) : [...state.selectedParameters, action.value] };
    case "SET_PARAMETERS": return { ...state, selectedParameters: action.value };
    case "SET_RANGE":
      return { ...state, ranges: { ...state.ranges, [action.name]: { ...state.ranges[action.name], [action.field]: action.value } } };
    case "SET_PROFILE":
      return action.value === "PILOT_3_THEN_FAST24"
        ? {
            ...state,
            profile: action.value,
            saveDataset: true,
            selectedParameters: ["ibias_dac", "res_gain"],
            ranges: {
              ...state.ranges,
              ibias_dac: { min: 60, max: 80, step: 1 },
              res_gain: { min: 0, max: 10, step: 1 },
            },
          }
        : { ...state, profile: action.value };
    case "SET_SWEEP_HARDWARE": return { ...state, sweepHardware: action.value };
    case "SET_SWEEP_MODE": return { ...state, sweepMode: action.value };
    case "SET_SWEEP_SEED": return { ...state, sweepSeed: action.value };
    case "SET_MAX_POINTS": return { ...state, maxPoints: action.value };
    case "SET_SETTLE_TIME": return { ...state, settleTimeS: action.value };
    case "SET_SAVE_DATASET": return { ...state, saveDataset: action.value };
  }
}

export function canContinueMapping(mapping: HardwareMapping): boolean {
  return Boolean(
    Number.isInteger(mapping.i2c_channel)
      && mapping.bert_port
      && mapping.bert_channel
      && mapping.dca_tdecq_source === "FUNC1"
      && mapping.dca_tdecq_equalizer_function === 1
      && mapping.dca_outer_er_source === "CHAN1A"
      && mapping.operator_confirmed,
  );
}
