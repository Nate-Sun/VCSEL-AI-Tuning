import { Link } from "react-router-dom";
import type { Campaign } from "../schemas/contracts";
import { StatusBadge } from "./StatusBadge";

const terminal = new Set(["COMPLETED", "FAILED", "RECOVERY_REQUIRED"]);

export function CampaignList({ campaigns }: { campaigns: Campaign[] }) {
  return (
    <section className="panel campaign-panel" aria-labelledby="campaign-list-title">
      <div className="panel-kicker">EXPERIMENT QUEUE</div>
      <h2 id="campaign-list-title">Campaigns / 实验任务</h2>
      <div className="campaign-list">
        {campaigns.map((campaign) => (
          <Link
            className="campaign-row"
            key={campaign.campaign_id}
            to={`/campaigns/${campaign.campaign_id}`}
          >
            <span>
              <strong>{campaign.campaign_id}</strong>
              <small>{terminal.has(campaign.state) ? "Recent / 最近" : "Active / 进行中"}</small>
            </span>
            <span>
              <StatusBadge
                label={campaign.state}
                tone={
                  campaign.state === "RECOVERY_REQUIRED" || campaign.state === "FAILED"
                    ? "fault"
                    : campaign.state === "COMPLETED"
                      ? "healthy"
                      : "warning"
                }
              />
              <time dateTime={campaign.updated_at}>{new Date(campaign.updated_at).toLocaleString()}</time>
            </span>
          </Link>
        ))}
      </div>
    </section>
  );
}
