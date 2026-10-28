import { configDefaults, defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    include: ["tools/**/*.test.ts", "packages/*/src/**/*.test.ts", "packages/*/test/**/*.test.ts", "app/test/**/*.test.ts"],
    // GPU suites need a real browser with WebGPU; they run only via `npm run test:gpu` (vitest.gpu.config.ts).
    exclude: [...configDefaults.exclude, "**/*.gpu.test.ts", "**/*.gpu.bench.ts"],
    // The suites are deterministic compute, and dozens of tests take 3–5 s on a CI runner (2026-10-28: a draw test took
    // 5.1 s and failed main's CI on vitest's 5 s default). A timeout here only has to catch a hang, not a slow runner;
    // tests that need longer still set their own.
    testTimeout: 30_000,
  },
})
