import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Observation } from "../schemas/contracts";

export function MetricTrendChart({ observations }: { observations: Observation[] }) {
  const data = observations.map((observation, index) => ({
    name: `#${index + 1}`,
    er: observation.metrics.er_db,
    tdecq: observation.metrics.tdecq_db,
  }));
  return (
    <section className="panel chart-panel" aria-labelledby="trend-title">
      <div className="panel-kicker">MEASUREMENT TREND · PROVISIONAL UNTIL SERVER CONFIRMS</div>
      <h2 id="trend-title">ER / TDECQ trend</h2>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 18, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="#18303a" strokeDasharray="2 5" />
            <XAxis dataKey="name" stroke="#6f8790" />
            <YAxis stroke="#6f8790" domain={[0, "auto"]} />
            <Tooltip contentStyle={{ background: "#0a171d", border: "1px solid #29444f" }} />
            <Legend />
            <ReferenceLine y={2.5} stroke="#e5a84f" strokeDasharray="4 4" />
            <ReferenceLine y={3} stroke="#60d9c0" strokeDasharray="4 4" />
            <Line name="ER dB" dataKey="er" stroke="#60d9c0" strokeWidth={2} connectNulls={false} />
            <Line name="TDECQ dB" dataKey="tdecq" stroke="#70b8de" strokeWidth={2} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="threshold-legend"><span>◆ coarse threshold · 2.5 dB</span><span>◇ constrained target · 3.0 dB</span></div>
      <div role="img" aria-label="ER and TDECQ trend summary" className="sr-chart-summary">
        {observations.map((observation) => <span key={observation.observation_id}>{observation.observation_id}: ER {observation.metrics.er_db ?? "unavailable"}; {observation.metrics.tdecq_db === null || observation.metrics.tdecq_db === undefined ? "TDECQ unavailable / TDECQ 不可用" : `TDECQ ${observation.metrics.tdecq_db}`}; {observation.valid ? "valid" : "invalid provisional"}. </span>)}
      </div>
    </section>
  );
}
