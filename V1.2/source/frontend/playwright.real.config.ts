import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-real",
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4174",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"], channel: "chrome" } },
  ],
  webServer: {
    command: "pnpm exec vite --host 127.0.0.1 --port 4174 --mode real",
    url: "http://127.0.0.1:4174",
    reuseExistingServer: false,
    env: {
      VITE_API_PROXY_TARGET: "http://127.0.0.1:8011",
      VITE_ENABLE_MOCKS: "false",
    },
  },
});
