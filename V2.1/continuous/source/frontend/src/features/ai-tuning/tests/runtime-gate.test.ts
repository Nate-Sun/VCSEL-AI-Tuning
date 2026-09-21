import type { Campaign, Capability } from "../schemas/contracts";
import { deriveRuntimeGate } from "../hooks/useRuntimeGate";

const capability = (overrides: Partial<Capability> = {}): Capability => ({
  runtime_mode: "FAKE",
  hardware_connected: false,
  read_only: false,
  live_tuning_enabled: false,
  authorization_modes: ["FULL_CAMPAIGN"],
  FULL_CAMPAIGN: true,
  ...overrides,
});

const active = (state: Campaign["state"]): Campaign => ({
  campaign_id: "campaign-1",
  product_version: "1.1.0",
  schema_version: "v1.1",
  runtime_mode: "FAKE",
  campaign_type: "AI_OPTIMIZATION",
  baseline_id: "b0",
  baseline_hash: "a".repeat(64),
  config_hash: "b".repeat(64),
  authorization_mode: "FULL_CAMPAIGN",
  state,
  stage_summaries: [],
  result: null,
  created_at: "2026-07-27T10:00:00Z",
  updated_at: "2026-07-27T10:00:00Z",
});

describe("runtime mutation gate", () => {
  it("allows simulated writes in FAKE but never presents them as hardware", () => {
    expect(deriveRuntimeGate(capability(), null)).toMatchObject({ canMutate: true, simulated: true, canStart: true });
  });

  it("locks every mutation in READ_ONLY", () => {
    expect(deriveRuntimeGate(capability({ runtime_mode: "READ_ONLY", read_only: true }), null)).toMatchObject({ canMutate: false, canStart: false });
  });

  it("requires FULL_CAMPAIGN before LIVE start", () => {
    expect(deriveRuntimeGate(capability({ runtime_mode: "LIVE", hardware_connected: true, live_tuning_enabled: true, FULL_CAMPAIGN: false }), null).canStart).toBe(false);
  });

  it("locks LIVE mutations when hardware is disconnected", () => {
    expect(deriveRuntimeGate(capability({ runtime_mode: "LIVE", hardware_connected: false, live_tuning_enabled: true }), null)).toMatchObject({ canMutate: false, canStart: false });
  });

  it("locks LIVE mutations when live tuning is disabled", () => {
    expect(deriveRuntimeGate(capability({ runtime_mode: "LIVE", hardware_connected: true, live_tuning_enabled: false }), null)).toMatchObject({ canMutate: false, canStart: false });
  });

  it("makes Recovery the only allowed workflow", () => {
    expect(deriveRuntimeGate(capability(), active("RECOVERY_REQUIRED"))).toMatchObject({ recoveryOnly: true, canCreateDraft: false, canStart: false });
  });

  it("prevents another draft while a campaign is active", () => {
    expect(deriveRuntimeGate(capability(), active("I2C_COARSE")).canCreateDraft).toBe(false);
  });
});
