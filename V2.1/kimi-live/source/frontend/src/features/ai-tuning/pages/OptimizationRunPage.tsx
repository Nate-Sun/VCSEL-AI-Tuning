import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { getCampaign, getObservations } from "../api/client";
import { MetricTrendChart } from "../charts/MetricTrendChart";
import { EventLog } from "../components/EventLog";
import { ParameterDiffTable } from "../components/ParameterDiffTable";
import { RunControls } from "../components/run/RunControls";
import { useCampaignEvents } from "../hooks/useCampaignEvents";
import { useRuntimeQueries } from "../hooks/useRuntimeQueries";
import { deriveRuntimeGate } from "../hooks/useRuntimeGate";
import { useBestObservation } from "../hooks/useV11Queries";

const phases = ["I2C coarse", "I2C fine", "FIR 0 / 3", "I2C retune", "Final verification"];

export function OptimizationRunPage() {
  const { campaignId = "" } = useParams();
  const campaign = useQuery({ queryKey: ["campaign", campaignId], queryFn: () => getCampaign(campaignId) });
  const observations = useQuery({ queryKey: ["campaign", campaignId, "observations"], queryFn: () => getObservations(campaignId) });
  const bestObservation = useBestObservation(campaignId);
  const stream = useCampaignEvents(campaignId);
  const runtime = useRuntimeQueries();
  if (campaign.isError || observations.isError || bestObservation.isError) return <div className="alert alert-fault">Optimization data unavailable. Controls remain locked.</div>;
  if (!campaign.data || !observations.data) return <div className="panel skeleton">Loading Optimization run…</div>;
  const latest = observations.data.at(-1);
  const best = bestObservation.data;
  const summary = campaign.data.optimization_summary;
  const active = runtime.campaigns.data?.find((item) => !["COMPLETED", "FAILED"].includes(item.state)) ?? campaign.data;
  const gate = runtime.capabilities.isSuccess && runtime.campaigns.isSuccess && runtime.capabilities.data ? deriveRuntimeGate(runtime.capabilities.data, active) : null;
  const controlReason = !gate ? "Runtime capabilities or active campaign state unavailable." : !gate.canMutate ? gate.reason ?? "Mutations are locked." : null;
  return <div className="page-stack run-page"><header className="run-hero"><div><span className="eyebrow">AI OPTIMIZATION · LIVE OPERATION</span><h1>{campaignId}</h1><div className="run-status"><b>{campaign.data.state}</b><span>ROUND {summary?.current_round ?? "—"} / MAX {summary?.max_rounds ?? "—"}</span><span>{summary?.pass_kind ?? "PASS UNAVAILABLE"}</span></div></div><div className="run-score"><span>BEST TDECQ</span><strong>{best?.metrics.tdecq_db?.toFixed(2) ?? "—"}</strong><small>dB · /best endpoint</small></div></header><section className="phase-track">{phases.map((phase,index) => <div className={summary && index < summary.completed_phase_count ? "done" : summary && index === summary.active_phase_index ? "active" : ""} key={phase}><span>{String(index+1).padStart(2,"0")}</span><b>{phase}</b></div>)}</section><RunControls campaignId={campaignId} state={campaign.data.state} disabledReason={controlReason} /><section className="telemetry-grid"><article className="panel"><div className="panel-kicker">CURRENT CANDIDATE · {summary?.candidates_used ?? "—"} USED · MEDIAN OF 3</div><h2>Current I2C + FIR readback</h2>{latest ? <ParameterDiffTable settings={latest.settings} /> : null}</article><article className="panel"><div className="panel-kicker">DCA EQUALIZER · {summary?.dca_equalizer_valid ? "VALID" : "UNAVAILABLE"}</div><h2>Equalizer taps → FIR</h2><div className="tap-strip">{summary?.dca_equalizer_taps.map((tap, index) => <span key={index}>TAP {index + 1} <b>{tap}</b></span>) ?? <span>UNAVAILABLE</span>}</div><dl className="definition-list"><div><dt>FIR proposal</dt><dd>FIR = f(DCA equalizer taps)</dd></div><div><dt>FIR unchanged</dt><dd>{summary?.fir_unchanged_rounds ?? "—"} / 3 consecutive rounds</dd></div><div><dt>Bayesian FIR search</dt><dd>NOT USED</dd></div><div><dt>25% / 50% / 100% damping</dt><dd>NOT USED</dd></div><div><dt>NLC</dt><dd>{summary ? `${summary.upper_nlc} / ${summary.lower_nlc}` : "UNAVAILABLE"} · LOCKED TO B0</dd></div></dl></article></section><MetricTrendChart observations={observations.data} /><div className="two-column"><section className="panel"><div className="panel-kicker">B0 / SAFETY</div><h2>Main rollback target · FIR Main {summary?.b0_fir_main ?? "UNAVAILABLE"}</h2><p>B0 remains immutable throughout the campaign. Abort always requests server-side restore and readback verification.</p><div className="status-line"><span>Rollback target</span><b>{summary?.rollback_target_status ?? "UNAVAILABLE"}</b></div><div className="status-line"><span>Rollback records</span><b>{summary?.rollback_count ?? "UNAVAILABLE"}</b></div><div className="status-line"><span>Convergence reason</span><b>{summary?.convergence_reason ?? "UNAVAILABLE"}</b></div><div className="status-line"><span>Final verification</span><b>PENDING · raw 3 → median</b></div></section><EventLog events={stream.events} status={stream.status} /></div></div>;
}
