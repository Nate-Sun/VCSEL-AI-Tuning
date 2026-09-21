import { Link, useParams } from "react-router-dom";
import { v11SweepExportUrl } from "../api/client";
import { DatasetStatusPanel } from "../components/DatasetStatusPanel";
import { useV11FinalReport } from "../hooks/useV11Queries";

export function V11TaskResultPage() {
  const { taskId = "" } = useParams();
  const report = useV11FinalReport(taskId);

  if (report.isError) return <div className="alert alert-fault">V1.1 final report unavailable.</div>;
  if (!report.data) return <div className="panel skeleton">Loading V1.1 final report…</div>;

  return (
    <div className="page-stack result-page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">{report.data.task_type} · V1.1 FINAL REPORT</div>
          <h1>{report.data.task_id}</h1>
          <p>Product {report.data.product_version} · Schema {report.data.schema_version} · Runtime {report.data.runtime_mode}</p>
        </div>
        <Link className="secondary-button" to={`/runs/${report.data.task_type === "SWEEP" ? "sweep" : "optimization"}/${taskId}`}>← Task</Link>
      </header>

      <section className="panel">
        <div className="panel-kicker">SERVER-PERSISTED OUTCOME</div>
        <h2>{report.data.state}</h2>
        <dl className="definition-list">
          <div><dt>B0 hash</dt><dd>{report.data.b0_hash ?? "UNAVAILABLE"}</dd></div>
          <div><dt>Configuration</dt><dd>{report.data.configuration_fingerprint}</dd></div>
          <div><dt>Observations</dt><dd>{report.data.observations.length}</dd></div>
          <div><dt>Event payloads</dt><dd>{report.data.events.length}</dd></div>
        </dl>
      </section>

      <DatasetStatusPanel
        saveDataset={report.data.config.save_dataset}
        measurementStatus={report.data.measurement_status}
        datasetStatus={report.data.dataset_status}
        datasetError={report.data.dataset_error}
        files={report.data.dataset_files}
      />

      <section className="panel">
        <div className="panel-kicker">OBSERVATIONS</div>
        <h2>Persisted measurements</h2>
        {report.data.observations.length === 0 ? <p className="empty-state">No observations persisted.</p> : (
          <div className="table-scroll"><table><thead><tr><th>ID</th><th>Phase</th><th>Status</th><th>Readback</th><th>TDECQ</th><th>ER</th></tr></thead><tbody>{report.data.observations.map((item) => <tr key={item.observation_id}><td>{item.observation_id}</td><td>{item.phase}</td><td>{item.status}</td><td>{JSON.stringify(item.readback)}</td><td>{item.tdecq ?? "—"}</td><td>{item.er ?? "—"}</td></tr>)}</tbody></table></div>
        )}
      </section>

      {report.data.task_type === "SWEEP" ? (
        <section className="panel"><div className="panel-kicker">SWEEP ARTIFACTS</div><h2>Server exports</h2><div className="run-controls"><a className="secondary-button" href={v11SweepExportUrl(taskId, "csv")}>CSV</a><a className="secondary-button" href={v11SweepExportUrl(taskId, "json")}>JSON</a></div></section>
      ) : null}
    </div>
  );
}
