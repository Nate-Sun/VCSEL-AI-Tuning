import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { App } from "../../../app/App";
import { server } from "../../../mocks/server";
import { RUNTIME_POLL_INTERVAL_MS } from "../hooks/useRuntimeQueries";

describe("VCSEL V1.2 dashboard", () => {
  beforeEach(() => window.history.pushState({}, "", "/"));

  it("renders the FAKE overview without claiming real hardware", async () => {
    render(<App />);
    expect(await screen.findByText("FAKE · SIMULATION")).toBeVisible();
    expect(await screen.findByRole("heading", { name: "AI Tuning V1.2" })).toBeVisible();
    expect(screen.getAllByText("SIMULATED")).toHaveLength(3);
    expect(screen.getByText("TDECQ")).toBeVisible();
    expect(screen.getByRole("link", { name: /View current task/i })).toBeVisible();
    expect(screen.queryByText(/Real hardware online/i)).not.toBeInTheDocument();
  });

  it("uses five-second polling and supports manual refresh", async () => {
    expect(RUNTIME_POLL_INTERVAL_MS).toBe(5000);
    let healthRequests = 0;
    server.use(http.get("/health", () => { healthRequests += 1; return HttpResponse.json({ status: "ok", runtime_mode: "FAKE" }); }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("FAKE · SIMULATION");
    const initialRequests = healthRequests;
    await user.click(await screen.findByRole("button", { name: /Refresh/i }));
    await waitFor(() => expect(healthRequests).toBeGreaterThan(initialRequests));
  });

  it("locks the surface when runtime endpoints fail", async () => {
    server.use(http.get("/health", () => HttpResponse.json({}, { status: 503 })), http.get("/capabilities", () => HttpResponse.json({}, { status: 503 })));
    render(<App />);
    expect(await screen.findByText(/API unavailable/i)).toBeVisible();
    expect(screen.getByText(/Waiting for API/i)).toBeVisible();
  });

  it("keeps the original campaign monitoring detail available", async () => {
    window.history.pushState({}, "", "/campaigns/campaign-live");
    render(<App />);
    expect(
      await screen.findByRole(
        "heading",
        { name: /campaign-live/i },
        { timeout: 3_000 },
      ),
    ).toBeVisible();
    expect(screen.getAllByText("I2C_COARSE").length).toBeGreaterThan(0);
    expect(screen.getByText("Pre-3")).toBeVisible();
    expect(screen.getByText("Main")).toBeVisible();
    expect(screen.getAllByText("ibias_dac").length).toBeGreaterThan(0);
    expect(screen.getByRole("img", { name: /ER and TDECQ trend summary/i })).toBeVisible();
  });

  it("preserves server-owned outcomes and recovery evidence", async () => {
    window.history.pushState({}, "", "/campaigns/campaign-result/result");
    render(<App />);
    expect(await screen.findByText("PARTIAL_CONVERGENCE")).toBeVisible();
    expect(screen.getByText("MEDIAN_CONFIRMED")).toBeVisible();
    expect(screen.getByText(/Rollback history/i)).toBeVisible();
    expect(screen.getByRole("link", { name: /observations\.csv/i })).toBeVisible();
  });

  it("never exposes raw hardware command input", async () => {
    render(<App />);
    await screen.findByText("AI Tuning V1.2");
    expect(screen.queryByRole("textbox", { name: /raw|SCPI|command|register address/i })).not.toBeInTheDocument();
  });

  it("provides visible keyboard focus order", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("AI Tuning V1.2");
    await user.tab();
    expect(screen.getByRole("link", { name: /Skip to content/i })).toHaveFocus();
  });
});
