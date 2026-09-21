import { render, screen } from "@testing-library/react";
import { EventLog } from "../components/EventLog";
import { OutcomeBanner } from "../components/OutcomeBanner";

describe("server-owned safety states", () => {
  it("renders a stale SSE state as explicit text", () => {
    render(<EventLog events={[]} status="stale" />);

    expect(screen.getByText("STALE")).toBeVisible();
  });

  it("renders RECOVERY_REQUIRED as a fault outcome", () => {
    const { container } = render(
      <OutcomeBanner
        report={{
          campaign_id: "campaign-recovery",
          product_version: "1.1.0",
          schema_version: "v1.1",
          campaign_type: "AI_OPTIMIZATION",
          runtime_mode: "FAKE",
          outcome: "RECOVERY_REQUIRED",
          convergence_reason: "rollback readback could not be verified",
          rollback_history: [],
          artifacts: [],
        }}
      />,
    );

    expect(screen.getByText("RECOVERY_REQUIRED")).toBeVisible();
    expect(container.querySelector(".outcome-fault")).toBeInTheDocument();
  });
});
