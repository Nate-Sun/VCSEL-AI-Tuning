import type { CampaignReport } from "../schemas/contracts";

export function OutcomeBanner({ report }: { report: CampaignReport }) {
  const outcome = report.outcome ?? "UNAVAILABLE";
  const fault = outcome === "FAILED" || outcome === "RECOVERY_REQUIRED";
  return (
    <section className={`outcome-banner ${fault ? "outcome-fault" : outcome === "SUCCESS" ? "outcome-success" : "outcome-warning"}`}>
      <div><span>SERVER OUTCOME / 服务端结论</span><h2>{outcome}</h2></div>
      <div><p>{report.convergence_reason ?? "Convergence reason unavailable / 收敛原因不可用"}</p>{outcome === "PARTIAL_CONVERGENCE" ? <strong>ER 未达到 3.0 dB · ER did not reach 3.0 dB</strong> : null}</div>
    </section>
  );
}
