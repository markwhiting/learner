import { defineConfig, devices } from "@playwright/test"

const PORT = 4173

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `http://localhost:${PORT}`, trace: "retain-on-failure" },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    // The cross-engine determinism probe (tests/e2e/detmath.spec.ts) only, and only when named with --project:
    // `npx playwright test detmath --project=chromium --project=webkit --project=firefox`. `npx playwright test` (CI,
    // npm run e2e) runs Chromium alone. WebKit is Playwright's build (`npx playwright install webkit`, binaries only).
    // Firefox is the installed stock Firefox driven over WebDriver BiDi (channel moz-firefox): Playwright's own Firefox
    // build (firefox-1555) exits with "Could not find profile folder" on macOS 27.2 (plan 05b, Task 6 probe note).
    { name: "webkit", use: { ...devices["Desktop Safari"] }, grep: /determinism/, default: false },
    { name: "firefox", use: { ...devices["Desktop Firefox"], channel: "moz-firefox" }, grep: /determinism/, default: false },
  ],
  // The e2e build honours ?testLifetime; the production (deploy) build does not.
  webServer: {
    command: `npm run build:e2e -w app && npm run preview:e2e -w app -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/learner/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
