import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { MutationButton } from "../components/MutationButton";
import { useContinueRecoveredCampaign, useEndRecoveredCampaign, useRecovery, useRestoreBaseline } from "../hooks/useV11Queries";
import { useRuntimeQueries } from "../hooks/useRuntimeQueries";
import { runtimeAllowsMutations } from "../hooks/useRuntimeGate";

export function RecoveryPage() {
  const { campaignId = "" } = useParams();
  const recovery = useRecovery(campaignId);
  const restore = useRestoreBaseline(campaignId);
  const continueTask = useContinueRecoveredCampaign(campaignId);
  const endTask = useEndRecoveredCampaign(campaignId);
  const navigate = useNavigate();
  const runtime = useRuntimeQueries();
  useEffect(() => {
    if (continueTask.isSuccess && recovery.data) navigate(`/runs/${recovery.data.task_type === "SWEEP" ? "sweep" : "optimization"}/${campaignId}`);
    if (endTask.isSuccess) navigate(`/campaigns/${campaignId}/result`);
  }, [campaignId, continueTask.isSuccess, endTask.isSuccess, navigate, recovery.data]);
  if (recovery.isError) return <div className="alert alert-fault">Recovery evidence unavailable. All recovery mutations remain locked.</div>;
  if (!recovery.data) return <div className="panel skeleton">Loading Recovery evidence…</div>;
  const data = restore.data && "restore_verified" in restore.data ? restore.data : recovery.data;
  const matchingRecovery = runtime.campaigns.data?.find((item) => item.campaign_id === campaignId && item.state === "RECOVERY_REQUIRED");
  const runtimeReady = runtime.capabilities.isSuccess && runtime.campaigns.isSuccess;
  const canRecover = runtimeReady && Boolean(matchingRecovery) && Boolean(runtime.capabilities.data && runtimeAllowsMutations(runtime.capabilities.data));
  const lockReason = !canRecover ? runtime.capabilities.isError || runtime.campaigns.isError ? "Runtime or campaign state unavailable." : !matchingRecovery ? "This campaign is not in RECOVERY_REQUIRED state." : "Runtime capabilities do not authorize recovery mutations." : null;
  const actionPending = restore.isPending || continueTask.isPending || endTask.isPending;
  const error = restore.error ?? continueTask.error ?? endTask.error;
  return <div className="page-stack recovery-page"><header className="recovery-hero"><span>GLOBAL SAFETY LOCK</span><h1>RECOVERY_REQUIRED</h1><p>基线恢复验证完成前，整个应用禁止其他写操作。</p></header><section className="recovery-grid"><article className="panel"><div className="panel-kicker">INTERRUPTION EVIDENCE</div><h2>{data.campaign_id}</h2><dl className="definition-list"><div><dt>Task type</dt><dd>{data.task_type}</dd></div><div><dt>Interrupted</dt><dd>{new Date(data.interrupted_at).toLocaleString()}</dd></div><div><dt>Last completed point</dt><dd>{data.last_completed_point ?? "None"}</dd></div><div><dt>Error</dt><dd>{data.error_reason}</dd></div><div><dt>Rollback error</dt><dd>{data.rollback_error ?? "None"}</dd></div></dl></article><article className="panel"><div className="panel-kicker">DEVICE STATE</div><h2>Recovery readiness</h2>{Object.entries(data.devices).map(([name,ok]) => <div className="device-check" key={name}><span>{name.toUpperCase()}</span><b>{ok ? "READY" : "UNAVAILABLE"}</b></div>)}<div className={data.restore_verified ? "notice-success" : "notice-warning"}>{data.restore_verified ? "BASELINE RESTORE VERIFIED" : "Restore has not been verified"}</div></article></section><section className="panel"><div className="panel-kicker">INTERRUPTED HARDWARE STATE</div><h2>Last known I2C / FIR</h2><pre className="data-block">{JSON.stringify({ i2c: data.last_known_i2c, fir: data.last_known_fir }, null, 2)}</pre></section><section className="panel"><div className="panel-kicker">CANONICAL BASELINE</div><h2>B0 restore target</h2><pre className="data-block">{JSON.stringify(data.baseline, null, 2)}</pre><div className="recovery-actions"><MutationButton pending={restore.isPending} disabled={actionPending && !restore.isPending} disabledReason={lockReason} className="danger-button" onClick={() => restore.mutate()}>Restore B0 and verify / 恢复并验证</MutationButton><MutationButton pending={continueTask.isPending} disabled={!data.restore_verified || (actionPending && !continueTask.isPending)} disabledReason={lockReason ?? (!data.restore_verified ? "Restore B0 must be verified first." : null)} onClick={() => continueTask.mutate()}>Continue Original Task</MutationButton><MutationButton pending={endTask.isPending} disabled={!data.restore_verified || (actionPending && !endTask.isPending)} disabledReason={lockReason ?? (!data.restore_verified ? "Restore B0 must be verified first." : null)} className="secondary-button" onClick={() => endTask.mutate()}>End as ABORTED_RECOVERED</MutationButton></div>{error ? <div className="alert alert-fault" role="alert">{error instanceof Error ? error.message : "Recovery action failed"}</div> : null}</section></div>;
}
