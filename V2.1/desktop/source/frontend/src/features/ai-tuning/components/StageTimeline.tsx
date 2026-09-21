import type { CampaignState } from "../schemas/contracts";

const stages: CampaignState[] = [
  "PREFLIGHT",
  "RESTORE_B0",
  "BASELINE_MEASUREMENT",
  "I2C_SCREENING",
  "I2C_COARSE",
  "I2C_FINE",
  "FIR_APPLY",
  "I2C_RETUNE",
  "FINAL_VERIFICATION",
  "COMPLETED",
];

export function StageTimeline({ current }: { current: CampaignState }) {
  const currentIndex = stages.indexOf(current);
  return (
    <section className="panel" aria-labelledby="timeline-title">
      <div className="panel-kicker">CAMPAIGN FLOW</div>
      <h2 id="timeline-title">Stage Timeline / 阶段时间线</h2>
      <ol className="stage-timeline">
        {stages.map((stage, index) => (
          <li className={index === currentIndex ? "current" : index < currentIndex ? "done" : "future"} key={stage}>
            <span aria-hidden="true">{index < currentIndex ? "●" : index === currentIndex ? "◆" : "○"}</span>
            <small>{stage}</small>
          </li>
        ))}
      </ol>
    </section>
  );
}
