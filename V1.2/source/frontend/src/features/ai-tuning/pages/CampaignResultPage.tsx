import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { getCampaign, getCampaignReport } from "../api/client";
import { ArtifactList } from "../components/ArtifactList";
import { ConfigurationComparison } from "../components/ConfigurationComparison";
import { HistoryResultApplyButton } from "../components/HistoryResultApplyButton";
import { OutcomeBanner } from "../components/OutcomeBanner";
import { SweepResultPanel } from "../components/SweepResultPanel";
import { useRuntimeQueries } from "../hooks/useRuntimeQueries";
import { deriveRuntimeGate } from "../hooks/useRuntimeGate";
import type { CampaignReport } from "../schemas/contracts";

function MetricSummary({ title, metric }: { title: string; metric: CampaignReport["er_db"] }) {
  return <article className="metric-summary"><span>{title}</span><strong>{metric?.median ?? "UNAVAILABLE"}</strong><small>median · std {metric?.std ?? "—"} · range {metric?.min ?? "—"}–{metric?.max ?? "—"}</small></article>;
}
function textField(report: CampaignReport, key: string, fallback: string) {
  const value = report[key];
  return typeof value === "string" ? value : fallback;
}

function displayRecordValue(value: unknown, fallback: string): string {
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

export function CampaignResultPage() {
  const { campaignId = "" } = useParams();
  const campaign = useQuery({ queryKey: ["campaign", campaignId], queryFn: () => getCampaign(campaignId) });
  const report = useQuery({ queryKey: ["campaign", campaignId, "report"], queryFn: () => getCampaignReport(campaignId) });
  const runtime = useRuntimeQueries();
  if (campaign.isError || report.isError) return <div className="alert alert-fault">Result unavailable / 结果不可用</div>;
  if (!campaign.data || !report.data) return <div className="panel skeleton">Loading result… / 正在载入结果…</div>;
  const productVersion = report.data.product_version;
  const schemaVersion = report.data.schema_version;
  const campaignType = report.data.campaign_type;
  const active = runtime.campaigns.data?.find((item) => !["COMPLETED", "FAILED"].includes(item.state)) ?? null;
  const gate = runtime.capabilities.isSuccess && runtime.campaigns.isSuccess && runtime.capabilities.data ? deriveRuntimeGate(runtime.capabilities.data, active) : null;
  const outcomeBlocksApply = report.data.outcome === "FAILED" || report.data.outcome === "FAILED_INVALID_DCA_TAPS" || report.data.outcome === "RECOVERY_REQUIRED" || textField(report.data, "outcome_code", "") === "FAILED_INVALID_DCA_TAPS";
  const applyDisabledReason = outcomeBlocksApply ? "This result is not eligible for Apply." : !gate ? "Runtime or active campaign state unavailable." : !gate.canCreateDraft ? gate.reason ?? "Another task is active." : null;
  if (campaignType === "SWEEP") return <div className="page-stack result-page"><header className="page-heading"><div><div className="eyebrow">SWEEP · FINAL RESULT</div><h1>{campaignId}</h1><p>Product {productVersion} · Schema {schemaVersion} · Runtime {report.data.runtime_mode ?? "unavailable"}</p></div><Link className="secondary-button" to={`/campaigns/${campaignId}`}>← Campaign detail</Link></header><OutcomeBanner report={report.data} /><SweepResultPanel campaignId={campaignId} /><ArtifactList artifacts={report.data.artifacts} /></div>;
  return <div className="page-stack result-page"><header className="page-heading"><div><div className="eyebrow">{campaignType} · FINAL RESULT</div><h1>{campaignId}</h1><p>Product {productVersion} · Schema {schemaVersion} · Runtime {report.data.runtime_mode ?? "unavailable"} · historical versions are never relabeled</p></div><Link className="secondary-button" to={`/campaigns/${campaignId}`}>← Campaign detail</Link></header><OutcomeBanner report={report.data} /><section className="result-metrics"><MetricSummary title="ER dB" metric={report.data.er_db} /><MetricSummary title="TDECQ dB" metric={report.data.tdecq_db} /><article className="metric-summary"><span>Measurement validity</span><strong className="text-size-medium">{report.data.measurement_validity ?? "UNAVAILABLE"}</strong><small>raw-three → median → final · rounds {report.data.alternating_rounds ?? "—"}</small></article></section>{report.data.final_measurements ? <section className="panel"><div className="panel-kicker">FINAL VERIFICATION</div><h2>Three raw measurements / 最终三次原始测量</h2><div className="large-readback">{report.data.final_measurements.map((measurement,index) => <span key={index}>RUN {index+1}<strong>{measurement.tdecq_db ?? "—"} dB</strong><small>ER {measurement.er_db ?? "—"} dB</small></span>)}</div></section> : null}<ConfigurationComparison finalSettings={report.data.final} /><section className="panel"><div className="panel-kicker">SAFETY EVIDENCE</div><h2>Rollback history / 回滚记录</h2>{report.data.rollback_history.length === 0 ? <p className="empty-state">No rollback recorded / 无回滚记录</p> : <ol className="rollback-list">{report.data.rollback_history.map((item, index) => <li key={index}><span>#{displayRecordValue(item.sequence, String(index + 1))}</span><strong>{displayRecordValue(item.status, "RECORDED")}</strong><p>{displayRecordValue(item.reason, "No reason supplied")}</p></li>)}</ol>}</section>{campaignType === "AI_OPTIMIZATION" ? <HistoryResultApplyButton campaignId={campaignId} disabledReason={applyDisabledReason} /> : null}<ArtifactList artifacts={report.data.artifacts} /></div>;
}
