import { expect, test } from "@playwright/test";

const allowedPosts = [
  /\/api\/v1\.1\/tasks$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/baseline$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/snapshot$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/scan-configuration$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/validate-ranges$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/sweep-count$/,
  /\/api\/v1\.1\/tasks\/[^/]+\/authorize-start$/,
];

test("FAKE Sweep wizard authorizes once and opens operations", async ({ page }) => {
  const posts: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") {
      const path = new URL(request.url()).pathname;
      posts.push(path);
      expect(allowedPosts.some((pattern) => pattern.test(path))).toBeTruthy();
      expect(request.postData() ?? "").not.toMatch(/SCPI|register_address|setFir|raw_command/i);
    }
  });

  await page.goto("/tasks/new?type=SWEEP&scenario=idle");
  await expect(page.getByRole("heading", { name: "Create tuning task" })).toBeVisible();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("checkbox", { name: /same physical link/i }).check();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Save current state as B0/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Scan selected sources/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByText("Full-combination", { exact: true }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Authorize and Start/i }).click();
  await expect(page).toHaveURL(/\/runs\/sweep\/campaign-fake-011/);
  await expect(page.getByText(/SWEEP · FULL/)).toBeVisible();
  await page.addStyleTag({ content: ".topbar { position: static !important; } .skip-link { display: none !important; }" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/sweep-run-desktop.png", fullPage: true });
  expect(posts.filter((path) => path.endsWith("/authorize-start"))).toHaveLength(1);
});

test("FAKE Recovery keeps continuation locked until verified", async ({ page }) => {
  await page.goto("/recovery/campaign-recovery?scenario=recovery");
  await expect(page.getByRole("heading", { name: "RECOVERY_REQUIRED" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Continue Original Task/i })).toBeDisabled();
  await page.getByRole("button", { name: /Restore B0 and verify/i }).click();
  await expect(page.getByText("BASELINE RESTORE VERIFIED")).toBeVisible();
  await expect(page.getByRole("button", { name: /Continue Original Task/i })).toBeEnabled();
  await page.addStyleTag({ content: ".topbar { position: static !important; } .skip-link { display: none !important; }" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "artifacts/screenshots/recovery-desktop.png", fullPage: true });
});
