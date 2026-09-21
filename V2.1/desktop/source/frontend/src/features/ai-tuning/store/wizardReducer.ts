import type { ParameterName, TaskType } from "../schemas/contracts";

export interface HardwareMapping {
  i2c_channel: string;
  bert_channel: string;
  dca_source: string;
  same_link_confirmed: boolean;
}

export interface WizardState {
  step: number;
  taskType: TaskType;
  taskId: string | null;
  mapping: HardwareMapping;
  selectedParameters: ParameterName[];
  ranges: Record<ParameterName, { min: number; max: number; step: number }>;
  profile: "FAST" | "STANDARD" | "DEEP" | "ADVANCED";
  sweepHardware: "I2C" | "BERT_FIR";
  sweepMode: "SINGLE" | "FULL" | null;
  sweepSeed: number | null;
  maxPoints: number;
}

export const initialWizardState: WizardState = {
  step: 1,
  taskType: "AI_OPTIMIZATION",
  taskId: null,
  mapping: { i2c_channel: "2", bert_channel: "1", dca_source: "CH1A", same_link_confirmed: false },
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
  | { type: "SET_SWEEP_HARDWARE"; value: WizardState["sweepHardware"] }
  | { type: "SET_SWEEP_MODE"; value: "SINGLE" | "FULL" }
  | { type: "SET_SWEEP_SEED"; value: number }
  | { type: "SET_MAX_POINTS"; value: number };

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
    case "SET_PROFILE": return { ...state, profile: action.value };
    case "SET_SWEEP_HARDWARE": return { ...state, sweepHardware: action.value };
    case "SET_SWEEP_MODE": return { ...state, sweepMode: action.value };
    case "SET_SWEEP_SEED": return { ...state, sweepSeed: action.value };
    case "SET_MAX_POINTS": return { ...state, maxPoints: action.value };
  }
}

export function canContinueMapping(mapping: HardwareMapping): boolean {
  return Boolean(mapping.i2c_channel && mapping.bert_channel && mapping.dca_source && mapping.same_link_confirmed);
}
