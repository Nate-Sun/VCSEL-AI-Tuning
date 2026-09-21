import { expect, test } from "@playwright/test";

test("FAKE overview, monitoring result, and history Apply", async ({ page }) => {
  const browserErrors: string[] = [];
  page.on("console", (message) => { if (message.type() === "error") browserErrors.push(message.text()); });
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await page.goto("/");
  await expect(page.getByText("FAKE · SIMULATION")).toBeVisible();
  await expect(page.getByRole("heading", { name: "AI Tuning V1.1" })).toBeVisible();
  await expect(page.locator(".instrument-tile").getByText("SIMULATED")).toHaveCount(3);
  await expect(page.getByRole("link", { name: /View current task/i })).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/dashboard-desktop.png", fullPage: true });

  await page.goto("/campaigns/campaign-result/result?scenario=idle");
  await expect(page.getByText("PARTIAL_CONVERGENCE")).toBeVisible();
  await expect(page.getByText(/Product 1\.0\.0/)).toBeVisible();
  await page.getByRole("button", { name: /Apply this result/i }).click();
  await expect(page.getByText("HISTORY_RESULT_APPLIED")).toBeVisible();
  await expect(page.getByText("COMPLETE_OR_ROLLBACK")).toBeVisible();
  await page.addStyleTag({ content: ".topbar { position: static !important; } .skip-link { display: none !important; }" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/result-desktop.png", fullPage: true });
  await expect(page.locator("vite-error-overlay, .vite-error-overlay")).toHaveCount(0);
  expect(browserErrors).toEqual([]);
});

test("360px layout remains readable and keyboard reachable", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 800 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "AI Tuning V1.1" })).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/dashboard-360px.png", fullPage: true });
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: /Skip to content/i })).toBeFocused();
});
