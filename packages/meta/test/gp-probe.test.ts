// The GP determinism probe's Node side (plan 05b, Task 6): computeGp (gp-probe-inputs.ts) in Node, twice, gives the same
// bits, and they equal fixtures/gp-probe-golden.json, which tests/e2e/detmath.spec.ts compares each browser engine
// with. GP_PROBE_WRITE=1 regenerates the file (never in CI).
import { describe, it, expect } from "vitest"
import { readFileSync, writeFileSync } from "node:fs"
import { computeGp, toBits, tsDrawInputAt } from "./gp-probe-inputs"
import { birthDraw } from "../src/index"

const GOLDEN = new URL("./fixtures/gp-probe-golden.json", import.meta.url)

describe("the GP determinism probe (Node)", () => {
  it("computeGp twice in Node gives the same bytes, and they equal fixtures/gp-probe-golden.json", () => {
    const a = JSON.stringify(toBits(computeGp()), null, 1) + "\n"
    const b = JSON.stringify(toBits(computeGp()), null, 1) + "\n"
    expect(a).toBe(b)
    if (process.env.GP_PROBE_WRITE === "1") {
      expect(process.env.CI, "never regenerate the GP probe golden in CI").toBeUndefined()
      writeFileSync(GOLDEN, a)
    }
    expect(a).toBe(readFileSync(GOLDEN, "utf8"))
  }, 120_000)

  it("the probe covers what it claims: a 4-start fit, 20 ts picks and 10 confirm draws over all three arms", () => {
    const g = computeGp()
    expect(g.fit.iterations).toHaveLength(4)
    expect(g.picks).toHaveLength(20)
    expect(g.picks.every(d => d.method === "ts")).toBe(true)
    expect(g.confirms).toHaveLength(10)
    expect(g.confirms.every(d => d.method === "confirm")).toBe(true)
    expect(new Set(g.confirms.map(d => d.confirmation!.arm))).toEqual(new Set(["candidate", "centre", "newborn"]))
    expect(g.predict.cov).toHaveLength(36)
    expect(g.incumbent.pBeatsCentre).toBeGreaterThanOrEqual(0)
  }, 120_000)

  it("the timing input at 300 rows parses and draws ts", () => {
    const i = tsDrawInputAt(300)
    expect(i.posterior!.models.gpu!.rows).toHaveLength(300)
    expect(birthDraw(i).method).toBe("ts")
  }, 60_000)
})
