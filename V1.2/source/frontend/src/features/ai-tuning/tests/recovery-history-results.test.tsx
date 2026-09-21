import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../../../app/App";
import { http, HttpResponse } from "msw";
import { server } from "../../../mocks/server";

describe("Recovery and history result operations", () => {
  it("uses one canonical recovery request to restore B0 and resume", async () => {
    window.history.pushState({}, "", "/recovery/campaign-recovery?scenario=recovery");
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("RECOVERY_REQUIRED")).toBeVisible();
    await user.click(await screen.findByRole("button", { name: /Restore B0 and Continue/i }));
    expect(await screen.findByRole("heading", { name: "campaign-recovery" })).toBeVisible();
    expect(window.location.pathname).toBe("/runs/optimization/campaign-recovery");
  });

  it("applies a persisted best result and displays server readbacks", async () => {
    window.history.pushState({}, "", "/campaigns/campaign-result/result?scenario=idle");
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("PARTIAL_CONVERGENCE")).toBeVisible();
    expect(screen.getByText(/Product 1\.0\.0/i)).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Apply this result/i }));
    expect(await screen.findByText("HISTORY_RESULT_APPLIED")).toBeVisible();
    expect(screen.getByText("observation-best-1")).toBeVisible();
    expect(screen.getByText("0 / 9 / -28 / 104 / -8 / 1 / -3")).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("routes structured Apply RECOVERY_REQUIRED errors to Recovery", async () => {
    server.use(
      http.post("/api/v1.1/history/apply", () => HttpResponse.json({ detail: { code: "RECOVERY_REQUIRED", message: "rollback verification failed" } }, { status: 409 })),
    );
    window.history.pushState({}, "", "/campaigns/campaign-result/result?scenario=idle");
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /Apply this result/i }));
    expect(await screen.findByRole("heading", { name: "RECOVERY_REQUIRED" })).toBeVisible();
  });
});
