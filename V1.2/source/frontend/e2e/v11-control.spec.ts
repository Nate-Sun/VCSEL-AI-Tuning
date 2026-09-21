import { expect, test } from "@playwright/test";

const allowedPosts = [
  /\/api\/v1\.1\/mapping\/scan$/,
  /\/api\/v1\.1\/parameters\/validate$/,
  /\/api\/v1\.1\/sweeps\/count$/,
  /\/api\/v1\.1\/tasks$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/b0$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/authorization$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/start$/,
];

test("FAKE Sweep wizard authorizes once and opens operations", async ({ page }) => {
  const posts: string[] = [];
  let createPayload: Record<string, unknown> | null = null;
  page.on("request", (request) => {
    if (request.method() === "POST") {
      const path = new URL(request.url()).pathname;
      posts.push(path);
      if (/\/api\/v1\.1\/tasks$/.test(path)) createPayload = request.postDataJSON() as Record<string, unknown>;
      expect(allowedPosts.some((pattern) => pattern.test(path))).toBeTruthy();
      expect(request.postData() ?? "").not.toMatch(/SCPI|register_address|setFir|raw_command/i);
    }
  });

  await page.goto("/tasks/new?type=SWEEP&scenario=idle");
  await expect(page.getByRole("heading", { name: "Create tuning task" })).toBeVisible();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("checkbox", { name: /same physical link/i }).check();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Approve current snapshot as B0/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Scan server mapping/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  const saveDataset = page.getByRole("checkbox", { name: /Save dataset/i });
  await expect(saveDataset).not.toBeChecked();
  await saveDataset.check();
  await expect(page.getByText(/Round Time 和磁盘占用/)).toBeVisible();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByText("Full-combination", { exact: true }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await expect(page.getByText("ENABLED · LOCKED")).toBeVisible();
  await page.getByRole("button", { name: /Confirm and Start/i }).click();
  await expect(page).toHaveURL(/\/runs\/sweep\/task-/);
  await expect(page.getByText(/SWEEP · FULL/)).toBeVisible();
  await page.addStyleTag({ content: ".topbar { position: static !important; } .skip-link { display: none !important; }" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/sweep-run-desktop.png", fullPage: true });

  await expect(page.getByText("Dataset saving: SUCCESS")).toBeVisible();
  await expect(page.getByRole("link", { name: "Download eye_density.npy" })).toBeVisible();
  expect(createPayload).toMatchObject({ save_dataset: true });

  const authorizationIndex = posts.findIndex((path) => path.endsWith("/authorization"));
  const startIndex = posts.findIndex((path) => path.endsWith("/start"));
  expect(authorizationIndex).toBeGreaterThan(-1);
  expect(startIndex).toBeGreaterThan(authorizationIndex);
  expect(posts.filter((path) => path.endsWith("/authorization"))).toHaveLength(1);
  expect(posts.filter((path) => path.endsWith("/start"))).toHaveLength(1);
  expect(posts.some((path) => path.endsWith("/authorize-start"))).toBe(false);
});

test("FAKE Recovery delegates B0 restore and resume to one canonical request", async ({ page }) => {
  await page.goto("/recovery/campaign-recovery?scenario=recovery");
  await expect(page.getByRole("heading", { name: "RECOVERY_REQUIRED" })).toBeVisible();
  await page.addStyleTag({ content: ".topbar { position: static !important; } .skip-link { display: none !important; }" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/recovery-desktop.png", fullPage: true });
  await page.getByRole("button", { name: /Restore B0 and Continue/i }).click();
  await expect(page).toHaveURL(/\/runs\/optimization\/campaign-recovery/);
});
