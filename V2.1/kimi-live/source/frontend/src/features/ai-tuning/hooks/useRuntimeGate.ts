import type { Campaign, Capability } from "../schemas/contracts";

const TERMINAL_STATES = new Set<Campaign["state"]>(["COMPLETED", "FAILED"]);

export interface RuntimeGate {
  canMutate: boolean;
  canCreateDraft: boolean;
  canStart: boolean;
  recoveryOnly: boolean;
  simulated: boolean;
  reason: string | null;
}

export function runtimeAllowsMutations(capability: Capability): boolean {
  if (capability.runtime_mode === "FAKE") return !capability.read_only;
  if (capability.runtime_mode === "READ_ONLY" || capability.read_only) return false;
  return capability.hardware_connected && capability.live_tuning_enabled;
}

export function deriveRuntimeGate(capability: Capability, active: Campaign | null): RuntimeGate {
  const recoveryOnly = active?.state === "RECOVERY_REQUIRED";
  const readOnly = capability.runtime_mode === "READ_ONLY" || capability.read_only;
  const liveUnavailable = capability.runtime_mode === "LIVE" && (!capability.hardware_connected || !capability.live_tuning_enabled);
  const canMutate = runtimeAllowsMutations(capability) && !recoveryOnly;
  const hasActiveCampaign = active ? !TERMINAL_STATES.has(active.state) : false;
  const fullCampaignAllowed = capability.FULL_CAMPAIGN === true;

  let reason: string | null = null;
  if (recoveryOnly) reason = "Recovery must be completed before any other write operation.";
  else if (readOnly) reason = "READ_ONLY runtime blocks all mutation requests.";
  else if (liveUnavailable) reason = "LIVE mutations require connected hardware and live_tuning_enabled capability.";
  else if (!fullCampaignAllowed) reason = "FULL_CAMPAIGN authorization is unavailable.";

  return {
    canMutate,
    canCreateDraft: canMutate && !hasActiveCampaign,
    canStart: canMutate && fullCampaignAllowed && (!hasActiveCampaign || active?.state === "DRAFT"),
    recoveryOnly,
    simulated: capability.runtime_mode === "FAKE",
    reason,
  };
}
