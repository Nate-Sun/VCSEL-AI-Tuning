import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import {
  getBestObservation,
  getCampaign,
  getCampaignReport,
  getCapabilities,
  getHealth,
  getObservations,
} from "../api/client";
import { MetricTrendChart } from "../charts/MetricTrendChart";
import { ArtifactList } from "../components/ArtifactList";
import { BaselineCard } from "../components/BaselineCard";
import { EventLog } from "../components/EventLog";
import { ParameterDiffTable } from "../components/ParameterDiffTable";
import { StageTimeline } from "../components/StageTimeline";
import { StatusBadge } from "../components/StatusBadge";
import { useCampaignEvents } from "../hooks/useCampaignEvents";
import type { CampaignState } from "../schemas/contracts";

function safetyStatus(state: CampaignState): {
  label: string;
  tone: "healthy" | "warning" | "fault";
} {
  if (state === "RECOVERY_REQUIRED") return { label: "RECOVERY_REQUIRED", tone: "fault" };
  if (state === "FAILED") return { label: "FAULT", tone: "fault" };
  if (state === "PAUSED") return { label: "PAUSED", tone: "warning" };
  return { label: "SAFE · READ-ONLY", tone: "healthy" };
}

export function CampaignDetailPage() {
  const { campaignId = "" } = useParams();
  const campaign = useQuery({
    queryKey: ["campaign", campaignId],
    queryFn: () => getCampaign(campaignId),
  });
  const observations = useQuery({
    queryKey: ["campaign", campaignId, "observations"],
    queryFn: () => getObservations(campaignId),
  });
  const best = useQuery({
    queryKey: ["campaign", campaignId, "best"],
    queryFn: () => getBestObservation(campaignId),
  });
  const report = useQuery({
    queryKey: ["campaign", campaignId, "report"],
    queryFn: () => getCampaignReport(campaignId),
  });
  const health = useQuery({ queryKey: ["runtime", "health"], queryFn: getHealth });
  const capabilities = useQuery({
    queryKey: ["runtime", "capabilities"],
    queryFn: getCapabilities,
  });
  const stream = useCampaignEvents(campaignId);

  if (campaign.isError || observations.isError) {
    return <div className="alert alert-fault">Campaign unavailable / Campaign 不可用</div>;
  }
  if (!campaign.data || !observations.data) {
    return <div className="panel skeleton">Loading Campaign / 正在载入…</div>;
  }

  const latest = observations.data.at(-1);
  const er = latest?.metrics.er_db;
  const tdecq = latest?.metrics.tdecq_db;
  const safety = safetyStatus(campaign.data.state);
  const artifacts = [
    ...observations.data.flatMap((item) => item.artifacts),
    ...(report.data?.artifacts ?? []),
  ];

  return (
    <div className="page-stack">
      <header className="campaign-hero">
        <div>
          <div className="eyebrow">Campaign ID</div>
          <h1>{campaign.data.campaign_id}</h1>
          <div className="hero-badges">
            <StatusBadge label={campaign.data.state} tone="warning" />
            <StatusBadge label={safety.label} tone={safety.tone} />
            <StatusBadge
              label={health.data?.runtime_mode ?? "RUNTIME UNAVAILABLE"}
              tone={health.data ? "info" : "neutral"}
            />
            <StatusBadge
              label={latest?.valid ? "VALID" : "PROVISIONAL"}
              tone={latest?.valid ? "healthy" : "warning"}
            />
            <StatusBadge label={`Product ${campaign.data.product_version}`} tone="neutral" />
            <StatusBadge label={`Schema ${campaign.data.schema_version}`} tone="neutral" />
            <StatusBadge label={campaign.data.campaign_type} tone="info" />
          </div>
        </div>
        <div className="hero-metrics">
          <div>
            <span>ER / 当前 ER</span>
            <strong>{er === null || er === undefined ? "UNAVAILABLE" : `${er.toFixed(2)} dB`}</strong>
          </div>
          <div>
            <span>TDECQ / 当前 TDECQ</span>
            <strong>{tdecq === null || tdecq === undefined ? "UNAVAILABLE" : `${tdecq.toFixed(2)} dB`}</strong>
          </div>
          <div>
            <span>Candidate / Budget</span>
            <strong>UNAVAILABLE</strong>
          </div>
        </div>
      </header>

      <div className="detail-nav">
        <Link to="/">← Overview</Link>
        {campaign.data.result ? (
          <Link to={`/campaigns/${campaignId}/result`}>View result / 查看结果 →</Link>
        ) : null}
      </div>

      <StageTimeline current={campaign.data.state} />
      <MetricTrendChart observations={observations.data} />
      <div className="two-column">
        <BaselineCard />
        {latest ? <ParameterDiffTable settings={latest.settings} /> : <div className="panel">No settings</div>}
      </div>
      <div className="two-column">
        <section className="panel">
          <div className="panel-kicker">AI / OPTIMIZATION</div>
          <h2>Candidate context / 候选上下文</h2>
          <dl className="definition-list">
            <div><dt>Best source</dt><dd>{best.data ? "median-confirmed best / 服务端确认" : "UNAVAILABLE"}</dd></div>
            <div><dt>AI reason</dt><dd>{report.data?.convergence_reason ?? "UNAVAILABLE"}</dd></div>
            <div><dt>Optimization path</dt><dd>{campaign.data.state.includes("FIR") ? "FIR" : "I2C"}</dd></div>
            <div><dt>Model version</dt><dd>UNAVAILABLE</dd></div>
            <div><dt>hierarchical_ready</dt><dd>{capabilities.data ? String(capabilities.data.hierarchical_ready ?? false).toUpperCase() : "UNAVAILABLE"}</dd></div>
            <div><dt>fir_proposer_ready</dt><dd>{capabilities.data ? String(capabilities.data.fir_proposer_ready ?? false).toUpperCase() : "UNAVAILABLE"}</dd></div>
            <div><dt>Authorization</dt><dd>{campaign.data.authorization_mode}</dd></div>
          </dl>
        </section>
        <EventLog events={stream.events} status={stream.status} />
      </div>
      <ArtifactList artifacts={artifacts} />
    </div>
  );
}
