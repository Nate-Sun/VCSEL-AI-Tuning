import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../../../app/App";
import { http, HttpResponse } from "msw";
import { server } from "../../../mocks/server";

describe("Recovery and history result operations", () => {
  it("restores B0 before enabling Recovery continuation", async () => {
    window.history.pushState({}, "", "/recovery/campaign-recovery?scenario=recovery");
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("RECOVERY_REQUIRED")).toBeVisible();
    const continueButton = await screen.findByRole("button", { name: /Continue Original Task/i });
    expect(continueButton).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /Restore B0 and verify/i }));
    expect(await screen.findByText("BASELINE RESTORE VERIFIED")).toBeVisible();
    expect(continueButton).toBeEnabled();
  });

  it("applies a historical result once and displays all server stages", async () => {
    window.history.pushState({}, "", "/campaigns/campaign-result/result?scenario=idle");
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("PARTIAL_CONVERGENCE")).toBeVisible();
    expect(screen.getByText(/Product 1\.0\.0/i)).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Apply this result/i }));
    expect(await screen.findByText("HISTORY_RESULT_APPLIED")).toBeVisible();
    expect(screen.getByText("DEVICE_VERIFICATION")).toBeVisible();
    expect(screen.getByText("COMPLETE_OR_ROLLBACK")).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("routes structured Apply RECOVERY_REQUIRED errors to Recovery", async () => {
    server.use(
      http.post("/api/v1.1/campaigns/:campaignId/apply", () => HttpResponse.json({ detail: { code: "RECOVERY_REQUIRED", message: "rollback verification failed" } }, { status: 409 })),
    );
    window.history.pushState({}, "", "/campaigns/campaign-result/result?scenario=idle");
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /Apply this result/i }));
    expect(await screen.findByRole("heading", { name: "RECOVERY_REQUIRED" })).toBeVisible();
  });
});
