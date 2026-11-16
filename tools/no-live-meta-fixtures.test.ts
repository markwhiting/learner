// The gate rewrites data/meta/** on every run (decision 0019), so a test that pins its contents breaks main as soon as
// the gate moves on (05b's merge: the gate's v2 posterior failed a test that read the committed file as 05a's v1).
// Tests freeze the version they mean as a fixture; only the check that the live file parses may read it.
import { describe, it, expect } from "vitest"
import { readTracked, trackedFiles } from "./tracked"

// A path from a test file up to this checkout's data/meta (fixture repos built in a temp dir are fine).
const READS_LIVE_META = /\.\.\/data\/meta\/|\bREPO\b.*["'`]data\/meta\//
const ALLOWED = new Set(["packages/meta/test/posterior.test.ts"])

describe("no test pins the gate's live data/meta files", () => {
  it("only the parse check reads the repo's data/meta from a test", () => {
    // Unit tests and the browser tests alike (a tests/e2e spec read the live posterior and failed when the gate went flat).
    const tests = trackedFiles().filter(f => /\.(test|spec)\.ts$/.test(f) && f !== "tools/no-live-meta-fixtures.test.ts")
    expect(tests.some(f => f.startsWith("tests/e2e/"))).toBe(true)
    const offenders = tests.filter(f => !ALLOWED.has(f) && (readTracked(f) ?? "").split("\n").some(l => READS_LIVE_META.test(l)))
    expect(offenders).toEqual([])
  })
})
