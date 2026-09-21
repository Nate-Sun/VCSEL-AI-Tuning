import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";

let backend: ChildProcessWithoutNullStreams;
let backendOutput = "";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function repositoryRoot(): string {
  return path.resolve(process.cwd(), "..");
}

function pythonExecutable(): string {
  const configured = process.env.VCSEL_TEST_PYTHON;
  if (configured) return configured;
  const candidates = [
    path.resolve(repositoryRoot(), ".venv-ai-tuning", "Scripts", "python.exe"),
    path.resolve(repositoryRoot(), "..", "..", ".venv-ai-tuning", "Scripts", "python.exe"),
  ];
  return candidates.find(existsSync) ?? "python";
}

async function waitForBackend(): Promise<void> {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (backend.exitCode !== null) throw new Error(`FAKE backend exited early:\n${backendOutput}`);
    try {
      const response = await fetch("http://127.0.0.1:8011/api/v1.1/runtime/capabilities");
      const payload: unknown = await response.json();
      if (response.ok && isRecord(payload) && payload.runtime_mode === "FAKE") return;
    } catch {
      // The subprocess is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for FAKE backend:\n${backendOutput}`);
}

test.beforeAll(async () => {
  const root = repositoryRoot();
  backend = spawn(
    pythonExecutable(),
    [path.join(root, "scripts", "run_v1_1_fake_e2e_server.py")],
    {
      cwd: root,
      env: {
        ...process.env,
        PYTHONPATH: path.join(root, "src"),
        VCSEL_V11_E2E_PORT: "8011",
      },
      stdio: "pipe",
    },
  );
  backend.stdout.on("data", (chunk) => { backendOutput += String(chunk); });
  backend.stderr.on("data", (chunk) => { backendOutput += String(chunk); });
  await waitForBackend();
});

test.afterAll(async () => {
  if (!backend || backend.exitCode !== null) return;
  backend.kill();
  await Promise.race([
    new Promise<void>((resolve) => backend.once("exit", () => resolve())),
    new Promise<void>((resolve) => setTimeout(resolve, 5_000)),
  ]);
});

test("real FAKE FastAPI completes Wizard authorization and renders the V1.1 run page", async ({ page }) => {
  const paths: string[] = [];
  page.on("request", (request) => paths.push(new URL(request.url()).pathname));

  await page.goto("/tasks/new?type=SWEEP");
  await expect(page.getByRole("heading", { name: "Create tuning task" })).toBeVisible();
  expect(await page.evaluate(() => navigator.serviceWorker.controller)).toBeNull();

  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("checkbox", { name: /same physical link/i }).check();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Approve current snapshot as B0/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Scan server mapping/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();
  await page.getByRole("button", { name: /Continue/i }).click();

  const authorizationResponse = page.waitForResponse((response) => response.url().endsWith("/authorization"));
  const startResponse = page.waitForResponse((response) => response.url().endsWith("/start"));
  await page.getByRole("button", { name: /Confirm and Start/i }).click();
  const [authorized, started] = await Promise.all([authorizationResponse, startResponse]);
  expect(authorized.status()).toBe(200);
  const authorizedPayload: unknown = await authorized.json();
  expect(isRecord(authorizedPayload) && authorizedPayload.state).toBe("AUTHORIZED");
  expect(started.status()).toBe(200);
  const startedPayload: unknown = await started.json();
  expect(isRecord(startedPayload) && startedPayload.state).toBe("RUNNING");

  const authorizationRequest: unknown = authorized.request().postDataJSON();
  if (!isRecord(authorizationRequest) || typeof authorizationRequest.expires_at !== "string") {
    throw new Error("Authorization request did not contain an RFC3339 expires_at string");
  }
  expect(authorizationRequest.operator_confirmed).toBe(true);
  const ttlMs = Date.parse(authorizationRequest.expires_at) - Date.now();
  expect(ttlMs).toBeGreaterThan(29 * 60 * 1000);
  expect(ttlMs).toBeLessThanOrEqual(30 * 60 * 1000);

  await expect(page).toHaveURL(/\/runs\/sweep\/task-/);
  const taskId = page.url().split("/").at(-1)!;
  await expect(page.getByRole("heading", { name: taskId })).toBeVisible();
  await expect(page.getByText(/I2C_SWEEP/)).toBeVisible();
  await expect(page.getByText("TASK_STARTED")).toBeVisible();

  expect(paths).toContain("/api/v1.1/runtime/capabilities");
  expect(paths).toContain("/api/v1.1/tasks/active");
  expect(paths).toContain(`/api/v1.1/tasks/${taskId}/events`);
  expect(paths.some((value) => value === `/api/campaigns/${taskId}`)).toBe(false);
  expect(paths.some((value) => value.endsWith("/authorize-start"))).toBe(false);
});
