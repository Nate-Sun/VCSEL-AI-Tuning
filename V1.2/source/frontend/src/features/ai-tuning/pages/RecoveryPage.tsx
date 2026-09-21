import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { MutationButton } from "../components/MutationButton";
import { useContinueRecoveredCampaign, useEndRecoveredCampaign, useRecovery } from "../hooks/useV11Queries";
import { useRuntimeQueries } from "../hooks/useRuntimeQueries";
import { runtimeAllowsMutations } from "../hooks/useRuntimeGate";

export function RecoveryPage() {
  const { taskId = "" } = useParams();
  const recovery = useRecovery(taskId);
  const continueTask = useContinueRecoveredCampaign(taskId);
  const endTask = useEndRecoveredCampaign(taskId);
  const navigate = useNavigate();
  const runtime = useRuntimeQueries();

  useEffect(() => {
    if (continueTask.data?.state === "RUNNING") {
      navigate(`/runs/${continueTask.data.task_type === "SWEEP" ? "sweep" : "optimization"}/${taskId}`);
    }
    if (endTask.data?.state === "ABORTED_RECOVERED") navigate(`/tasks/${taskId}/result`);
  }, [continueTask.data, endTask.data, navigate, taskId]);

  if (recovery.isError) return <div className="alert alert-fault">Recovery evidence unavailable. All recovery mutations remain locked.</div>;
  if (!recovery.data) return <div className="panel skeleton">Loading Recovery evidence…</div>;

  const runtimeReady = runtime.capabilities.isSuccess && Boolean(runtime.capabilities.data);
  const mutable = runtimeReady && runtime.capabilities.data ? runtimeAllowsMutations(runtime.capabilities.data) : false;
  const isRecoveryRequired = recovery.data.state === "RECOVERY_REQUIRED";
  const lockReason = !runtimeReady
    ? "Runtime capabilities are unavailable."
    : !mutable
      ? "Runtime capabilities do not authorize recovery mutations."
      : !isRecoveryRequired
        ? `Task state is ${recovery.data.state}, not RECOVERY_REQUIRED.`
        : null;
  const actionPending = continueTask.isPending || endTask.isPending;
  const error = continueTask.error ?? endTask.error;

  return (
    <div className="page-stack recovery-page">
      <header className="recovery-hero">
        <span>GLOBAL SAFETY LOCK</span>
        <h1>RECOVERY_REQUIRED</h1>
        <p>恢复请求由 FastAPI 原子地恢复并验证 B0，然后继续或安全结束任务。</p>
      </header>
      <section className="recovery-grid">
        <article className="panel">
          <div className="panel-kicker">PERSISTED TASK EVIDENCE</div>
          <h2>{recovery.data.task_id}</h2>
          <dl className="definition-list">
            <div><dt>Task type</dt><dd>{recovery.data.task_type}</dd></div>
            <div><dt>State</dt><dd>{recovery.data.state}</dd></div>
            <div><dt>Runtime</dt><dd>{recovery.data.runtime_mode}</dd></div>
            <div><dt>Updated</dt><dd>{new Date(recovery.data.updated_at).toLocaleString()}</dd></div>
          </dl>
        </article>
        <article className="panel">
          <div className="panel-kicker">CANONICAL B0</div>
          <h2>Frozen rollback target</h2>
          <p className="data-block">{recovery.data.b0_hash ?? "B0 hash unavailable"}</p>
          <div className={isRecoveryRequired ? "notice-warning" : "notice-success"}>{recovery.data.state}</div>
        </article>
      </section>
      <section className="panel">
        <div className="panel-kicker">SERVER-OWNED RECOVERY</div>
        <h2>Restore B0 and choose the terminal action</h2>
        <p>The browser does not perform a separate restore or assume it succeeded. Each canonical recovery request asks FastAPI to restore and verify B0 before returning the new persisted state.</p>
        <div className="recovery-actions">
          <MutationButton pending={continueTask.isPending} disabled={actionPending && !continueTask.isPending} disabledReason={lockReason} onClick={() => continueTask.mutate()}>Restore B0 and Continue</MutationButton>
          <MutationButton pending={endTask.isPending} disabled={actionPending && !endTask.isPending} disabledReason={lockReason} className="secondary-button" onClick={() => endTask.mutate()}>Restore B0 and End as ABORTED_RECOVERED</MutationButton>
        </div>
        {error ? <div className="alert alert-fault" role="alert">{error instanceof Error ? error.message : "Recovery action failed"}</div> : null}
      </section>
    </div>
  );
}
