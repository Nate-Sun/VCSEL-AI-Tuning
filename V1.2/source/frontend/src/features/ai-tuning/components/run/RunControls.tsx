import { MutationButton } from "../MutationButton";
import {
  useAbortCampaign,
  usePauseCampaign,
  useResumeCampaign,
} from "../../hooks/useV11Queries";

interface RunControlsProps {
  taskId: string;
  state: string;
  disabledReason?: string | null;
  resumeAllowed?: boolean;
}

export function RunControls({
  taskId,
  state,
  disabledReason,
  resumeAllowed = true,
}: RunControlsProps) {
  const pause = usePauseCampaign(taskId);
  const resume = useResumeCampaign(taskId);
  const abort = useAbortCampaign(taskId);
  const anyPending = pause.isPending || resume.isPending || abort.isPending;
  const pendingReason = anyPending
    ? "Another task action is awaiting the server response."
    : null;
  const error = pause.error ?? resume.error ?? abort.error;
  const pauseAllowed = state === "RUNNING";
  const resumeStateAllowed = state === "PAUSED";
  const abortAllowed = state === "RUNNING" || state === "PAUSED";

  return (
    <div>
      <div className="run-controls">
        <MutationButton
          pending={pause.isPending}
          disabled={!pauseAllowed || (anyPending && !pause.isPending)}
          disabledReason={disabledReason ?? pendingReason}
          className="secondary-button"
          onClick={() => pause.mutate()}
        >
          Pause / 暂停
        </MutationButton>
        <MutationButton
          pending={resume.isPending}
          disabled={!resumeStateAllowed || !resumeAllowed || (anyPending && !resume.isPending)}
          disabledReason={!resumeAllowed
            ? "Resume blocked until server Preflight is allowed."
            : disabledReason ?? pendingReason}
          className="secondary-button"
          onClick={() => resume.mutate()}
        >
          Resume / 继续
        </MutationButton>
        <MutationButton
          pending={abort.isPending}
          disabled={!abortAllowed || (anyPending && !abort.isPending)}
          disabledReason={disabledReason ?? pendingReason}
          className="danger-button"
          onClick={() => abort.mutate()}
        >
          Abort & restore B0 / 中止并恢复
        </MutationButton>
      </div>
      {error ? (
        <div className="alert alert-fault" role="alert">
          {error instanceof Error ? error.message : "Task action failed"}
        </div>
      ) : null}
    </div>
  );
}
