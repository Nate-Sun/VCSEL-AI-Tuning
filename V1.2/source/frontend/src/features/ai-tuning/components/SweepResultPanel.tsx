import { useState } from "react";
import { v11SweepExportUrl } from "../api/client";
import { useV11SweepPoints } from "../hooks/useV11Queries";

export function SweepResultPanel({ campaignId: taskId }: { campaignId: string }) {
  const [page, setPage] = useState(1);
  const pageSize = 25;
  const points = useV11SweepPoints(taskId, page, pageSize);

  return (
    <section className="panel">
      <div className="table-tools">
        <div><div className="panel-kicker">V1.1 SWEEP FINAL RESULT</div><h2>Server-paginated points</h2></div>
        <div><a className="secondary-button" href={v11SweepExportUrl(taskId, "csv")}>CSV</a><a className="secondary-button" href={v11SweepExportUrl(taskId, "json")}>JSON</a></div>
      </div>
      {points.isError ? <div className="alert alert-fault">Sweep results unavailable.</div> : !points.data ? <div className="skeleton">Loading Sweep results…</div> : (
        <div className="table-scroll"><table><thead><tr><th>Original</th><th>Execution</th><th>Status</th><th>Readback</th><th>TDECQ</th><th>ER</th><th>Reason</th></tr></thead><tbody>{points.data.items.map((point) => <tr key={`${point.original_index}:${point.execution_index}`}><td>{point.original_index}</td><td>{point.execution_index}</td><td>{point.status}</td><td>{JSON.stringify(point.readback)}</td><td>{point.tdecq ?? "—"}</td><td>{point.er ?? "—"}</td><td>{point.error_reason ?? "—"}</td></tr>)}</tbody></table></div>
      )}
      <div className="pagination"><button disabled={page === 1} onClick={() => setPage((value) => value - 1)}>← Previous</button><span>Page {page} · {points.data?.total ?? 0} points</span><button disabled={!points.data || points.data.offset + points.data.items.length >= points.data.total} onClick={() => setPage((value) => value + 1)}>Next →</button></div>
      <div className="notice-warning">B0 restore confirmation is unavailable from this endpoint.</div>
    </section>
  );
}
