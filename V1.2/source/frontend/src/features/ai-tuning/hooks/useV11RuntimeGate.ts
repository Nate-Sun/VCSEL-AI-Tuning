import type {
  V11RuntimeCapabilities,
  V11TaskDocument,
} from "../schemas/contracts";

type ActiveTaskIdentity = Pick<V11TaskDocument, "task_id" | "state">;

export interface V11WizardGate {
  canMutate: boolean;
  canCreateDraft: boolean;
  canStart: boolean;
  simulated: boolean;
  reason: string | null;
}

export function v11RuntimeAllowsMutations(capability: V11RuntimeCapabilities): boolean {
  return capability.runtime_mode === "FAKE"
    || capability.runtime_mode === "REPLAY"
    || (capability.runtime_mode === "LIVE" && capability.hardware_mutation);
}

export function deriveV11WizardGate(
  capability: V11RuntimeCapabilities,
  activeTask: ActiveTaskIdentity | null,
  ownedTaskId: string | null,
): V11WizardGate {
  const simulated = capability.runtime_mode === "FAKE" || capability.runtime_mode === "REPLAY";
  const canMutate = v11RuntimeAllowsMutations(capability);
  const ownsActiveTask = activeTask !== null && activeTask.task_id === ownedTaskId;
  const conflictingTask = activeTask !== null && !ownsActiveTask;
  const activeStateAllowsStart = activeTask === null
    || (ownsActiveTask && ["B0_FROZEN", "AUTHORIZED"].includes(activeTask.state));

  let reason: string | null = null;
  if (!canMutate) reason = "V1.1 runtime does not permit task mutations.";
  else if (!capability.full_campaign_authorization) {
    reason = "FULL_CAMPAIGN authorization is unavailable.";
  } else if (conflictingTask) {
    reason = `Another V1.1 task is active: ${activeTask.task_id}`;
  }

  return {
    canMutate,
    canCreateDraft: canMutate && activeTask === null,
    canStart: canMutate
      && capability.full_campaign_authorization
      && ownedTaskId !== null
      && !conflictingTask
      && activeStateAllowsStart,
    simulated,
    reason,
  };
}
