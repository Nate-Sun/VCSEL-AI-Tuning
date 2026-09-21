import type { RuntimeHealth } from "../schemas/contracts";
import { StatusBadge } from "./StatusBadge";

export function RuntimeHealthCard({ health }: { health: RuntimeHealth }) {
  const fake = health.runtime_mode === "FAKE";
  return (
    <section className="panel runtime-card" aria-labelledby="runtime-health-title">
      <div className="panel-kicker">API / RUNTIME</div>
      <div className="panel-heading-row">
        <h2 id="runtime-health-title">运行状态 / Runtime health</h2>
        <StatusBadge label={health.status.toUpperCase()} tone="healthy" />
      </div>
      <div className="runtime-mode-row">
        <span className="metric-value compact">{health.runtime_mode}</span>
        {fake ? (
          <span className="mode-disclosure">模拟运行 / Simulated runtime</span>
        ) : null}
      </div>
      <div className="mini-status-grid">
        <span>Store <b>{health.store?.ok ? "OK" : "UNAVAILABLE"}</b></span>
        <span>Repository <b>{health.repository?.ok ? "OK" : "UNAVAILABLE"}</b></span>
      </div>
    </section>
  );
}
