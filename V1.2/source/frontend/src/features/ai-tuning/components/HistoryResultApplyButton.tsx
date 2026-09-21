import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ApiResponseError } from "../api/client";
import { useApplyHistory } from "../hooks/useV11Queries";
import { MutationButton } from "./MutationButton";

export function HistoryResultApplyButton({ campaignId: taskId, disabledReason }: { campaignId: string; disabledReason?: string | null }) {
  const apply = useApplyHistory(taskId);
  const navigate = useNavigate();
  const recoveryError = apply.error instanceof ApiResponseError && apply.error.code === "RECOVERY_REQUIRED";

  useEffect(() => {
    if (recoveryError) navigate(`/recovery/${taskId}`);
  }, [recoveryError, navigate, taskId]);

  return (
    <section className="panel apply-panel">
      <div>
        <div className="panel-kicker">HISTORY APPLY · SERVER-VERIFIED</div>
        <h2>Apply this result / 应用此历史结果</h2>
        <p>FastAPI resolves the persisted best observation, snapshots the click-time configuration, performs readbacks and one DCA measurement, then confirms or rolls back.</p>
      </div>
      <MutationButton pending={apply.isPending} disabledReason={disabledReason} pendingLabel="Applying and verifying…" onClick={() => apply.mutate()}>Apply this result / 应用此结果</MutationButton>
      {apply.isError ? <div className="alert alert-fault">{apply.error instanceof Error ? apply.error.message : "Apply failed"}</div> : null}
      {apply.data ? (
        <div className="apply-result success">
          <strong>{apply.data.status}</strong>
          <dl className="definition-list">
            <div><dt>Observation</dt><dd>{apply.data.observation_id}</dd></div>
            <div><dt>I2C readback</dt><dd>{JSON.stringify(apply.data.i2c_readback)}</dd></div>
            <div><dt>FIR readback</dt><dd>{apply.data.fir_readback.join(" / ")}</dd></div>
            <div><dt>Measurement</dt><dd>{JSON.stringify(apply.data.measurement)}</dd></div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}
