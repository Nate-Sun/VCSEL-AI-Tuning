import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { MutationButton } from "./MutationButton";
import { useApplyHistory, useHistoryApplyStatus } from "../hooks/useV11Queries";
import { ApiResponseError } from "../api/client";

export function HistoryResultApplyButton({ campaignId, disabledReason }: { campaignId: string; disabledReason?: string | null }) {
  const apply = useApplyHistory(campaignId);
  const status = useHistoryApplyStatus(apply.data?.operation_id, apply.data?.status === "RUNNING");
  const result = status.data ?? apply.data;
  const navigate = useNavigate();
  const recoveryError = apply.error instanceof ApiResponseError && apply.error.code === "RECOVERY_REQUIRED" || status.error instanceof ApiResponseError && status.error.code === "RECOVERY_REQUIRED";
  useEffect(() => {
    if (result?.status === "RECOVERY_REQUIRED" || recoveryError) navigate(`/recovery/${campaignId}`);
  }, [result?.status, recoveryError, campaignId, navigate]);
  const running = result?.status === "RUNNING";
  return <section className="panel apply-panel"><div><div className="panel-kicker">HISTORY APPLY · SERVER-VERIFIED</div><h2>Apply this result / 应用此历史结果</h2><p>FastAPI snapshots the current configuration, performs readbacks and a DCA measurement, then either confirms the result or rolls back.</p></div><MutationButton pending={apply.isPending || running} disabledReason={disabledReason} pendingLabel="Applying and verifying…" onClick={() => apply.mutate()}>Apply this result / 应用此结果</MutationButton>{apply.isError || status.isError ? <div className="alert alert-fault">{apply.error instanceof Error ? apply.error.message : status.error instanceof Error ? status.error.message : "Apply status unavailable"}</div> : null}{result ? <div className={`apply-result ${result.status === "HISTORY_RESULT_APPLIED" ? "success" : "fault"}`}><strong>{result.status}</strong><ol>{result.stages.map((stage) => <li key={stage.name}><span>{stage.name}</span><b>{stage.status}</b></li>)}</ol>{result.rollback_status ? <p>Rollback: {result.rollback_status}</p> : null}</div> : null}</section>;
}
