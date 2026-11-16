import { describe, it, expect } from "vitest"
import { icc1 } from "../../packages/evaluators/src/stats.ts"
import {
  anova1, betaInc, chiSquareQuantile, covComponents, fCdf, fQuantile, holmAdjust, iccCi, powerAnovaRandom, psd2, repeatsFor, sampler, spearman, tCrit,
} from "./flat-stats.ts"

describe("flat-stats distributions", () => {
  it("betaInc matches closed forms", () => {
    expect(betaInc(0.5, 2, 3)).toBeCloseTo(11 / 16, 12) // I_½(2,3) = 11/16
    expect(betaInc(0.3, 1, 1)).toBeCloseTo(0.3, 12)
    expect(betaInc(0.2, 1, 4)).toBeCloseTo(1 - 0.8 ** 4, 12)
    expect(betaInc(0.9, 0.5, 0.5)).toBeCloseTo((2 / Math.PI) * Math.asin(Math.sqrt(0.9)), 10)
  })
  it("F, t and χ² quantiles match tables", () => {
    expect(tCrit(0.05, 10)).toBeCloseTo(2.228139, 5)
    expect(fQuantile(0.95, 9, 50)).toBeCloseTo(2.0734, 3)
    expect(fCdf(fQuantile(0.3, 4, 7), 4, 7)).toBeCloseTo(0.3, 9)
    expect(chiSquareQuantile(0.95, 1)).toBeCloseTo(3.841459, 4)
    expect(chiSquareQuantile(0.025, 40)).toBeCloseTo(24.433, 2)
  })
})

describe("flat-stats ANOVA and power", () => {
  const groups = [[1, 2, 3], [2, 3, 4, 5], [6, 7, 8], [4, 4.5, 5]]
  it("anova1's ICC is the evaluators' icc1 where positive, and its F p is the F tail", () => {
    const a = anova1(groups)
    expect(a.icc).toBeCloseTo(icc1(groups), 12)
    expect(a.p).toBeCloseTo(1 - fCdf(a.F, a.k - 1, a.N - a.k), 12)
    const [lo, hi] = iccCi(a)
    expect(lo).toBeLessThan(a.icc)
    expect(hi).toBeGreaterThan(a.icc)
  })
  it("power is alpha at ICC 0, grows with n, and repeatsFor finds the 80% point", () => {
    expect(powerAnovaRandom(0, 10, 6, 1 / 60)).toBe(1 / 60)
    expect(powerAnovaRandom(0.2, 10, 12, 1 / 60)).toBeGreaterThan(powerAnovaRandom(0.2, 10, 6, 1 / 60))
    const n = repeatsFor(0.2, 10, 1 / 60)!
    expect(powerAnovaRandom(0.2, 10, n, 1 / 60)).toBeGreaterThanOrEqual(0.8)
    expect(powerAnovaRandom(0.2, 10, n - 1, 1 / 60)).toBeLessThan(0.8)
    expect(repeatsFor(0, 10, 1 / 60)).toBeNull()
  })
  it("the power formula agrees with a seeded simulation of the F test", () => {
    const s = sampler(7), k = 10, n = 6, icc = 0.3, crit = fQuantile(1 - 0.05, k - 1, k * (n - 1))
    let hits = 0
    const sims = 4000
    for (let i = 0; i < sims; i++) {
      const gs = Array.from({ length: k }, () => { const b = Math.sqrt(icc) * s.normal(); return Array.from({ length: n }, () => b + Math.sqrt(1 - icc) * s.normal()) })
      if (anova1(gs).F > crit) hits++
    }
    expect(Math.abs(hits / sims - powerAnovaRandom(icc, k, n, 0.05))).toBeLessThan(0.03)
  })
})

describe("flat-stats helpers", () => {
  it("holmAdjust is monotone and capped", () => {
    expect(holmAdjust([0.01, 0.04, 0.03, 0.5])).toEqual([0.04, 0.09, 0.09, 0.5])
    expect(holmAdjust([0.6, 0.7])).toEqual([1, 1])
  })
  it("spearman is rank-based", () => {
    expect(spearman([1, 2, 3, 4], [10, 100, 1000, 1e4])).toBeCloseTo(1, 12)
    expect(spearman([1, 2, 3, 4], [4, 3, 2, 1])).toBeCloseTo(-1, 12)
  })
  it("psd2 keeps a PSD matrix and clips a negative eigenvalue", () => {
    expect(psd2([[2, 1], [1, 2]]).flat().map(v => +v.toFixed(12))).toEqual([2, 1, 1, 2])
    const m = psd2([[1, 0], [0, -1]])
    expect(m[0]![0]).toBeCloseTo(1, 12)
    expect(m[1]![1]).toBeCloseTo(0, 12)
  })
  it("covComponents' diagonal is anova1's variance components", () => {
    const gs = [[[1, 0], [2, 1], [3, 0]], [[5, 1], [6, 2], [7, 2]], [[2, 5], [2, 4], [3, 6]]]
    const c = covComponents(gs)
    const a0 = anova1(gs.map(g => g.map(r => r[0]!))), a1 = anova1(gs.map(g => g.map(r => r[1]!)))
    expect(c.between[0]![0]).toBeCloseTo(a0.sb2, 12)
    expect(c.within[1]![1]).toBeCloseTo(a1.sw2, 12)
  })
  it("sampler is deterministic per seed", () => {
    const a = sampler(42), b = sampler(42)
    expect([a.normal(), a.beta(2, 3), a.index(10)]).toEqual([b.normal(), b.beta(2, 3), b.index(10)])
  })
})
