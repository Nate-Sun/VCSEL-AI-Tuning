import { render, screen } from "@testing-library/react";
import { App } from "../../../app/App";
import {
  canContinueMapping,
  initialWizardState,
  wizardReducer,
} from "../store/wizardReducer";

describe("V1.1 wizard core", () => {
  beforeEach(() => window.history.pushState({}, "", "/"));

  it("selects Sweep and clears Sweep-only state when returning to Optimization", () => {
    const sweep = wizardReducer(initialWizardState, { type: "SELECT_TASK_TYPE", value: "SWEEP" });
    expect(sweep.taskType).toBe("SWEEP");
    const configured = { ...sweep, sweepMode: "FULL" as const, sweepSeed: 42 };
    const optimization = wizardReducer(configured, { type: "SELECT_TASK_TYPE", value: "AI_OPTIMIZATION" });
    expect(optimization).toMatchObject({ taskType: "AI_OPTIMIZATION", sweepMode: null, sweepSeed: null });
  });

  it("requires explicit same-link confirmation", () => {
    expect(canContinueMapping({ i2c_channel: "2", bert_channel: "1", dca_source: "CH1A", same_link_confirmed: false })).toBe(false);
    expect(canContinueMapping({ i2c_channel: "2", bert_channel: "1", dca_source: "CH1A", same_link_confirmed: true })).toBe(true);
  });

  it("replaces selected parameters with the server-available scan subset", () => {
    const scanned = wizardReducer(initialWizardState, { type: "SET_PARAMETERS", value: ["eq"] });
    expect(scanned.selectedParameters).toEqual(["eq"]);
    expect(scanned.selectedParameters).not.toContain("ibias_dac");
  });

  it("presents V1.1 and interlocks task creation while a campaign is active", async () => {
    render(<App />);
    expect(await screen.findByText("AI Tuning V1.1")).toBeVisible();
    expect(screen.getByRole("link", { name: /View current task/i })).toBeVisible();
    expect(screen.queryByRole("link", { name: /Create task/i })).not.toBeInTheDocument();
  });
});
