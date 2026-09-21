import { Link, useParams } from "react-router-dom";
import { DatasetStatusPanel } from "../components/DatasetStatusPanel";
import { EventLog } from "../components/EventLog";
import { RunControls } from "../components/run/RunControls";
import {
  useV11Observations,
  useV11BestState,
  useV11Task,
  useV11TaskEvents,
} from "../hooks/useV11Queries";
import { useV11RuntimeQueries } from "../hooks/useV11RuntimeQueries";
import { v11RuntimeAllowsMutations } from "../hooks/useV11RuntimeGate";

const TERMINAL_STATES = new Set([
  "COMPLETED",
  "COMPLETED_NO_IMPROVEMENT",
  "ABORTED",
  "ABORTED_RECOVERED",
  "FAILED",
  "FAILED_INVALID_DCA_TAPS",
]);

export function OptimizationRunPage() {
  const { taskId = "" } = useParams();
  const task = useV11Task(taskId);
  const observations = useV11Observations(taskId);
  const best = useV11BestState(taskId);
  const events = useV11TaskEvents(taskId);
  const runtime = useV11RuntimeQueries();

  if (task.isError || observations.isError || best.isError) {
    return <div className="alert alert-fault">V1.1 Optimization data unavailable. Controls remain locked.</div>;
  }
  if (!task.data || !observations.data) {
    return <div className="panel skeleton">Loading V1.1 Optimization task…</div>;
  }

  const capability = runtime.capabilities.data;
  const activeTask = runtime.activeTask.data ?? null;
  const mutable = capability ? v11RuntimeAllowsMutations(capability) : false;
  const conflictingTask = activeTask !== null && activeTask.task_id !== taskId;
  const controlReason = !runtime.capabilities.isSuccess || !runtime.activeTask.isSuccess
    ? "V1.1 runtime capabilities or active-task state is unavailable."
    : !mutable
      ? "V1.1 runtime does not permit task mutations."
      : conflictingTask
        ? `Another V1.1 task is active: ${activeTask.task_id}`
        : null;
  const latest = observations.data.at(-1);
  const eventStatus = events.isError ? "stale" : events.data ? "connected" : "connecting";

  return (
    <div className="page-stack run-page">
      <header className="run-hero">
        <div>
          <span className="eyebrow">AI OPTIMIZATION · V1.1 TASK</span>
          <h1>{task.data.task_id}</h1>
          <div className="run-status">
            <b>{task.data.state}</b>
            <span>{task.data.runtime_mode}</span>
            <span>{task.data.config.depth}</span>
            <span>{task.data.config.candidate_budget} candidates</span>
          </div>
        </div>
        <div className="run-score">
          <span>SERVER-OWNED BEST STATE</span>
          <strong className="text-size-medium">{best.data?.observation_id ?? "—"}</strong>
          <small>from /api/v1.1/tasks/{taskId}/best</small>
        </div>
      </header>

      <RunControls taskId={taskId} state={task.data.state} disabledReason={controlReason} />

      <DatasetStatusPanel
        saveDataset={task.data.config.save_dataset}
        measurementStatus={task.data.measurement_status}
        datasetStatus={task.data.dataset_status}
        datasetError={task.data.dataset_error}
        files={task.data.dataset_files}
      />

      <section className="telemetry-grid">
        <article className="panel">
          <div className="panel-kicker">LATEST PERSISTED OBSERVATION</div>
          <h2>{latest?.observation_id ?? "No observation yet"}</h2>
          {latest ? (
            <dl className="definition-list">
              <div><dt>Phase / status</dt><dd>{latest.phase} · {latest.status}</dd></div>
              <div><dt>Requested</dt><dd>{JSON.stringify(latest.requested)}</dd></div>
              <div><dt>Applied</dt><dd>{JSON.stringify(latest.applied)}</dd></div>
              <div><dt>Readback</dt><dd>{JSON.stringify(latest.readback)}</dd></div>
              <div><dt>TDECQ / ER</dt><dd>{latest.tdecq ?? "UNAVAILABLE"} / {latest.er ?? "UNAVAILABLE"}</dd></div>
            </dl>
          ) : <p className="empty-state">The server has not persisted an observation.</p>}
        </article>
        <article className="panel">
          <div className="panel-kicker">BEST COMPLETE STATE</div>
          <h2>{best.data?.observation_id ?? "No persisted best state"}</h2>
          {best.data ? <dl className="definition-list"><div><dt>I2C</dt><dd>{JSON.stringify(best.data.i2c)}</dd></div><div><dt>FIR</dt><dd>{best.data.fir.join(" / ")}</dd></div><div><dt>Result</dt><dd>{JSON.stringify(best.data.result)}</dd></div></dl> : null}
        </article>
        <article className="panel">
          <div className="panel-kicker">FROZEN AUTHORIZATION ENVELOPE</div>
          <h2>B0 and request bounds</h2>
          <dl className="definition-list">
            <div><dt>B0 hash</dt><dd>{task.data.b0_hash ?? "UNAVAILABLE"}</dd></div>
            <div><dt>Parameters</dt><dd>{task.data.config.parameters.map((item) => `${item.name} ${item.min}:${item.step}:${item.max}`).join(" · ")}</dd></div>
            <div><dt>Configuration</dt><dd>{task.data.configuration_fingerprint}</dd></div>
          </dl>
        </article>
      </section>

      <EventLog events={events.data ?? []} status={eventStatus} />
      {TERMINAL_STATES.has(task.data.state) ? (
        <Link className="primary-button" to={`/tasks/${taskId}/result`}>Open V1.1 final report</Link>
      ) : null}
    </div>
  );
}
