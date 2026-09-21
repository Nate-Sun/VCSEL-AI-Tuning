import { Link } from "react-router-dom";
import { useRuntimeQueries } from "../hooks/useRuntimeQueries";
import { useBestObservation } from "../hooks/useV11Queries";

const terminal = new Set(["COMPLETED", "FAILED"]);

function metric(value: number | null | undefined, unit: string) {
  return value == null ? <span className="metric-unavailable">Unavailable / 暂不可用</span> : <><strong>{value.toFixed(2)}</strong><small>{unit}</small></>;
}
export function OverviewPage() {
  const { health, capabilities, campaigns } = useRuntimeQueries();
  const active = campaigns.data?.find((campaign) => !terminal.has(campaign.state));
  const snapshot = useBestObservation(active?.campaign_id);
  const lastUpdatedAt = Math.max(health.dataUpdatedAt, capabilities.dataUpdatedAt, campaigns.dataUpdatedAt, snapshot.dataUpdatedAt);

  const refresh = async () => Promise.all([health.refetch(), capabilities.refetch(), campaigns.refetch(), snapshot.refetch()]);
  const devices = health.data?.hardware;

  return (
    <div className="page-stack overview-page">
      <header className="page-heading v11-heading">
        <div><div className="eyebrow">VCSEL · CONTROL SURFACE · 01</div><h1>AI Tuning V1.1</h1><p>光链路优化任务中心 / Guided optical-link operations</p></div>
        <div className="refresh-cluster"><span>Last sync / 最后同步<br /><b>{lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleTimeString() : "Waiting for API"}</b></span><button className="secondary-button" onClick={() => void refresh()}>Refresh / 刷新</button></div>
      </header>

      {health.isError || capabilities.isError ? <div className="alert alert-fault">API unavailable / API 不可用。所有控制保持锁定。</div> : null}

      <section className="instrument-rack" aria-label="Instrument connections">
        {(["dca", "bert", "i2c"] as const).map((name, index) => {
          const state = devices?.[name];
          return <article className="instrument-tile" key={name}><span className="instrument-index">0{index + 1}</span><div><small>{name.toUpperCase()} CONNECTION</small><strong>{state?.status ?? (capabilities.data?.runtime_mode === "FAKE" ? "SIMULATED" : "UNAVAILABLE")}</strong></div><i className={state?.ok ? "signal on" : "signal"} aria-hidden="true" /></article>;
        })}
      </section>

      <section className="metric-stage">
        <div className="metric-copy"><span className="section-number">LIVE / 02</span><h2>Current optical state<br /><em>当前光学状态</em></h2><p>Only validated FastAPI observations appear here. FAKE values are explicitly simulated.</p></div>
        <article className="hero-metric"><span>TDECQ</span>{metric(snapshot.data?.metrics.tdecq_db, "dB")}</article>
        <article className="hero-metric"><span>EXTINCTION RATIO</span>{metric(snapshot.data?.metrics.er_db, "dB")}</article>
      </section>

      {active ? (
        <section className="active-interlock">
          <div><span className="status-pip" /><small>ACTIVE TASK · {active.state}</small><h2>{active.campaign_id}</h2><p>一次只允许运行一个任务 / A second task cannot be created while this campaign is active.</p></div>
          <Link className="primary-button" to={active.state === "RECOVERY_REQUIRED" ? `/recovery/${active.campaign_id}` : `/runs/${active.campaign_type === "SWEEP" ? "sweep" : "optimization"}/${active.campaign_id}`}>View current task / 查看当前任务</Link>
        </section>
      ) : (
        <section className="launch-grid" aria-label="Create task">
          <Link className="launch-card" to="/tasks/new?type=AI_OPTIMIZATION"><span>01 / OPTIMIZE</span><h2>AI Optimization</h2><p>分阶段搜索 I2C 与 FIR，在最终验证前始终保留 B0 回滚目标。</p><b>Create task / 创建任务 →</b></Link>
          <Link className="launch-card" to="/tasks/new?type=SWEEP"><span>02 / CHARACTERIZE</span><h2>Sweep</h2><p>扫描安全参数网格，以服务端判定的最优点和完整读回为准。</p><b>Create task / 创建任务 →</b></Link>
        </section>
      )}

      <section className="recent-section" id="recent"><div className="section-title"><span>HISTORY / 03</span><h2>Recent tasks / 最近任务</h2></div><div className="recent-list">{campaigns.data?.map((campaign) => <Link to={`/campaigns/${campaign.campaign_id}${campaign.state === "COMPLETED" ? "/result" : ""}`} key={campaign.campaign_id}><span>{campaign.campaign_id}<small>{campaign.campaign_type} · Product {campaign.product_version} · Schema {campaign.schema_version} · {campaign.runtime_mode ?? "runtime unavailable"}</small></span><small>{campaign.state}</small><time>{new Date(campaign.updated_at).toLocaleString()}</time></Link>) ?? <p>Loading…</p>}</div></section>
    </div>
  );
}
