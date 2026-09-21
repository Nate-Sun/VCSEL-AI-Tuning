import { useState } from "react";
import { sweepExportUrl } from "../api/client";
import { useSweepPage } from "../hooks/useV11Queries";

export function SweepResultPanel({ campaignId }: { campaignId: string }) {
  const [page, setPage] = useState(1);
  const points = useSweepPage(campaignId, `page=${page}&page_size=25&sort=tdecq_asc`);
  const summary = points.data?.summary;

  return <section className="panel"><div className="table-tools"><div><div className="panel-kicker">SWEEP FINAL RESULT</div><h2>All parameter points / 全部参数点</h2></div><div><a className="secondary-button" href={sweepExportUrl(campaignId, "csv")}>CSV</a><a className="secondary-button" href={sweepExportUrl(campaignId, "json")}>JSON</a></div></div>{points.isError ? <div className="alert alert-fault">Sweep results unavailable.</div> : !points.data ? <div className="skeleton">Loading Sweep results…</div> : <div className="table-scroll"><table><thead><tr><th>Original</th><th>Execution</th><th>Status</th><th>Parameters</th><th>TDECQ</th><th>ER</th><th>Reason</th><th>Best</th></tr></thead><tbody>{points.data.items.map((point) => <tr className={point.is_best ? "best-row" : ""} key={point.point_id}><td>{point.original_index}</td><td>{point.execution_index}</td><td>{point.status}</td><td>{JSON.stringify(point.readback)}</td><td>{point.tdecq_db ?? "—"}</td><td>{point.er_db ?? "—"}</td><td>{point.error_reason ?? "—"}</td><td>{point.is_best ? "BEST · TDECQ, then ER tie-break" : ""}</td></tr>)}</tbody></table></div>}<div className="pagination"><button disabled={page === 1} onClick={() => setPage((value) => value - 1)}>← Previous</button><span>Page {page} · {points.data?.total ?? 0} points</span><button disabled={!points.data || page * points.data.page_size >= points.data.total} onClick={() => setPage((value) => value + 1)}>Next →</button></div><div className={summary?.hardware_restored ? "notice-success" : "notice-warning"}>{summary?.hardware_restored ? "Hardware restored to B0 / 硬件已恢复 B0" : "B0 restore confirmation unavailable or pending."}</div></section>;
}
