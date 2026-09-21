import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { v11SweepExportUrl } from "../api/client";
import { DatasetStatusPanel } from "../components/DatasetStatusPanel";
import { EventLog } from "../components/EventLog";
import { RunControls } from "../components/run/RunControls";
import {
  useV11SweepPoints,
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

function formatSeconds(value: number | undefined): string {
  return value === undefined ? "Unavailable" : `${value.toFixed(6)} s`;
}

export function SweepRunPage() {
  const { taskId = "" } = useParams();
  const [page, setPage] = useState(1);
  const pageSize = 25;
  const task = useV11Task(taskId);
  const points = useV11SweepPoints(taskId, page, pageSize);
  const events = useV11TaskEvents(taskId);
  const runtime = useV11RuntimeQueries();

  if (task.isError || points.isError) {
    return <div className="alert alert-fault">V1.1 Sweep data unavailable. Controls remain locked.</div>;
  }
  if (!task.data || !points.data) {
    return <div className="panel skeleton">Loading V1.1 Sweep task…</div>;
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
  const config = task.data.config;
  const bestPoint = points.data.best_point;
  const eventStatus = events.isError ? "stale" : events.data ? "connected" : "connecting";

  return (
    <div className="page-stack run-page">
      <header className="run-hero">
        <div>
          <span className="eyebrow">SWEEP · {config.sweep_mode ?? "UNAVAILABLE"}</span>
          <h1>{task.data.task_id}</h1>
          <div className="run-status">
            <b>{task.data.state}</b>
            <span>{config.sweep_hardware ?? "UNAVAILABLE"}</span>
            <span>SEED {config.seed}</span>
            <span>SETTLE {config.settle_time_s ?? "UNAVAILABLE"} s</span>
            <span>TOTAL {points.data.total}</span>
          </div>
        </div>
        <div className="run-score" aria-label="Server global best point">
          <span>GLOBAL BEST / SERVER</span>
          <strong>{bestPoint?.tdecq ?? "—"}</strong>
          <small>{bestPoint ? <><code>{bestPoint.point_id}</code> · ER {bestPoint.er ?? "—"}</> : "Unavailable from server"}</small>
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

      <section className="panel">
        <div className="table-tools">
          <div>
            <span className="panel-kicker">SWEEP RESULTS · SERVER ORDER</span>
            <h2>Persisted points</h2>
          </div>
          <div>
            <a className="secondary-button" href={v11SweepExportUrl(taskId, "csv")}>CSV</a>
            <a className="secondary-button" href={v11SweepExportUrl(taskId, "json")}>JSON</a>
          </div>
        </div>
        <p className="helper-copy">Global best is displayed exactly as returned by FastAPI, even when that point is outside this page. No page-local best is calculated.</p>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Point ID</th><th>Original</th><th>Execution</th><th>Status</th><th>Requested</th><th>Applied</th><th>Readback</th><th>TDECQ</th><th>ER</th><th>Round timing</th><th>Error</th></tr></thead>
            <tbody>
              {points.data.items.map((point) => {
                const raw = point.raw_response;
                return (
                  <tr key={point.point_id} className={point.point_id === bestPoint?.point_id ? "is-best-point" : undefined}>
                    <td><code>{point.point_id}</code></td>
                    <td>{point.original_index}</td>
                    <td>{point.execution_index}</td>
                    <td>{point.status}</td>
                    <td>{JSON.stringify(point.requested)}</td>
                    <td>{JSON.stringify(point.applied)}</td>
                    <td>{JSON.stringify(point.readback)}</td>
                    <td>{point.tdecq ?? "—"}</td>
                    <td>{point.er ?? "—"}</td>
                    <td>
                      <details className="timing-details">
                        <summary><span>{formatSeconds(point.candidate_round_time_s ?? undefined)}</span> candidate · <span>{formatSeconds(point.screening_round_with_restore_s ?? undefined)}</span> with restore</summary>
                        <dl>
                          <div><dt>Requested settle</dt><dd>{formatSeconds(raw?.requested_settle_s)}</dd></div>
                          <div><dt>Write + readback</dt><dd>{formatSeconds(point.write_readback_duration_s ?? undefined)}</dd></div>
                          <div><dt>Actual settle</dt><dd>{formatSeconds(point.settle_duration_s ?? undefined)}</dd></div>
                          <div><dt>DCA measurement</dt><dd>{formatSeconds(point.dca_duration_s ?? undefined)}</dd></div>
                          <div><dt>Result persistence</dt><dd>{formatSeconds(point.result_persistence_duration_s ?? undefined)}</dd></div>
                          <div><dt>Candidate round</dt><dd>{formatSeconds(point.candidate_round_time_s ?? undefined)}</dd></div>
                          <div><dt>B0 restore</dt><dd>{formatSeconds(point.restore_duration_s ?? undefined)}</dd></div>
                          <div><dt>Round with restore</dt><dd>{formatSeconds(point.screening_round_with_restore_s ?? undefined)}</dd></div>
                          <div><dt>Restore status</dt><dd>{point.restore_status}</dd></div>
                          <div><dt>Restore readback</dt><dd>{point.restore_readback_result ? JSON.stringify(point.restore_readback_result) : "Unavailable"}</dd></div>
                        </dl>
                      </details>
                    </td>
                    <td>{point.error_reason ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="pagination">
          <button disabled={page === 1} onClick={() => setPage((value) => value - 1)}>← Previous</button>
          <span>Page {page} · {points.data.total} points</span>
          <button disabled={points.data.offset + points.data.items.length >= points.data.total} onClick={() => setPage((value) => value + 1)}>Next →</button>
        </div>
      </section>

      <EventLog events={events.data ?? []} status={eventStatus} />
      {TERMINAL_STATES.has(task.data.state) ? (
        <Link className="primary-button" to={`/tasks/${taskId}/result`}>Open V1.1 final report</Link>
      ) : null}
    </div>
  );
}
