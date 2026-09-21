import { render, screen } from "@testing-library/react";
import { App } from "./App";

describe("App shell", () => {
  it("discloses the runtime and exposes no raw hardware surface", async () => {
    render(<App />);
    expect(await screen.findByText("FAKE · SIMULATION")).toBeVisible();
    expect(screen.getByText(/FASTAPI CONTROL ONLY/i)).toBeVisible();
    expect(screen.queryByRole("textbox", { name: /raw|command|SCPI|register address/i })).not.toBeInTheDocument();
  });
});
