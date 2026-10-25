// Final review (Plan 05a): detmath in the page computes what Node computes. The e2e build's probe page computes the
// golden fixture's values in the browser, on the main thread and in a module worker (where births are drawn), and both
// must equal packages/detmath/test/fixtures/golden.json (written by Node) bit for bit.
//
// Plan 05b, Task 6: the same for the GP pipeline (packages/meta/test/gp-probe-inputs.ts: fitGp, conditionGp/predict,
// incumbentOf, and birthDraw's ts and confirm draws), against packages/meta/test/fixtures/gp-probe-golden.json. Both
// tests carry "determinism" in their names: `npx playwright test detmath --project=chromium --project=webkit
// --project=firefox` runs them in every engine; CI runs Chromium only.
import { test, expect } from "@playwright/test"
import { readFileSync } from "node:fs"

const GOLDEN = JSON.parse(readFileSync(new URL("../../packages/detmath/test/fixtures/golden.json", import.meta.url), "utf8")) as unknown
const GP_GOLDEN = JSON.parse(readFileSync(new URL("../../packages/meta/test/fixtures/gp-probe-golden.json", import.meta.url), "utf8")) as unknown

type Probe = { main: unknown; worker: unknown } | { error: string }

test("detmath determinism: the page (main thread and worker) equals the Node golden file bit for bit", async ({ page }) => {
  await page.goto("/learner/detmath-probe.html")
  await page.waitForFunction(() => (window as { __detmath?: unknown }).__detmath !== undefined)
  const got = await page.evaluate(() => (window as { __detmath?: unknown }).__detmath) as Probe
  expect("error" in got ? got.error : null).toBeNull()
  const { main, worker } = got as { main: unknown; worker: unknown }
  expect(main).toEqual(GOLDEN)
  expect(worker).toEqual(GOLDEN)
})

test("GP determinism: fit, ts picks and incumbent in the page (main thread and worker) equal the Node golden file bit for bit", async ({ page }, info) => {
  test.setTimeout(240_000)
  await page.goto("/learner/detmath-probe.html")
  await page.waitForFunction(() => (window as { __gpProbe?: unknown }).__gpProbe !== undefined, undefined, { timeout: 200_000 })
  const got = await page.evaluate(() => (window as { __gpProbe?: unknown }).__gpProbe) as Probe & { tsDrawMs300?: number }
  expect("error" in got ? got.error : null).toBeNull()
  const { main, worker, tsDrawMs300 } = got as { main: unknown; worker: unknown; tsDrawMs300: number }
  // One ts birthDraw at 300 model rows, timed in the worker (where births are drawn): recorded, not asserted.
  info.annotations.push({ type: "tsDrawMs300", description: String(tsDrawMs300) })
  console.log(`[${info.project.name}] one ts birthDraw at 300 rows in the worker: ${tsDrawMs300.toFixed(0)} ms`)
  expect(main).toEqual(GP_GOLDEN)
  expect(worker).toEqual(GP_GOLDEN)
})
