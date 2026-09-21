import { MutationButton } from "../MutationButton";
import { useAbortCampaign, usePauseCampaign, useResumeCampaign } from "../../hooks/useV11Queries";

export function RunControls({ campaignId, state, disabledReason, resumeAllowed = true }: { campaignId: string; state: string; disabledReason?: string | null | undefined; resumeAllowed?: boolean }) {
  const pause = usePauseCampaign(campaignId);
  const resume = useResumeCampaign(campaignId);
  const abort = useAbortCampaign(campaignId);
  const anyPending = pause.isPending || resume.isPending || abort.isPending;
  const pendingReason = anyPending ? "Another campaign action is awaiting the server response." : null;
  const error = pause.error ?? resume.error ?? abort.error;
  return <div><div className="run-controls"><MutationButton pending={pause.isPending} disabled={state === "PAUSED" || (anyPending && !pause.isPending)} disabledReason={disabledReason ?? pendingReason} className="secondary-button" onClick={() => pause.mutate()}>Pause / 暂停</MutationButton><MutationButton pending={resume.isPending} disabled={state !== "PAUSED" || !resumeAllowed || (anyPending && !resume.isPending)} disabledReason={!resumeAllowed ? "Resume blocked until server Preflight is ALLOWED." : disabledReason ?? pendingReason} className="secondary-button" onClick={() => resume.mutate()}>Resume / 继续</MutationButton><MutationButton pending={abort.isPending} disabled={anyPending && !abort.isPending} disabledReason={disabledReason ?? pendingReason} className="danger-button" onClick={() => abort.mutate()}>Abort & restore B0 / 中止并恢复</MutationButton></div>{error ? <div className="alert alert-fault" role="alert">{error instanceof Error ? error.message : "Campaign action failed"}</div> : null}</div>;
}
