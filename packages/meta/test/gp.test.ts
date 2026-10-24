import { describe, it, expect } from "vitest"
import { readFileSync, writeFileSync } from "node:fs"
import { createRng } from "@blanktab/detmath"
import { cholesky, cholInverse, solveLower, solveUpper } from "../src/linalg"
import {
  conditionGp, hyperOf, matern52, objective, predict, predictMean, priorOf, thetaOf, FIT_JITTER_STEPS, NOISE_FLOOR, type GpData,
} from "../src/gp"
import type { GpHyper } from "../src/posterior"

const FIXTURE = new URL("./fixtures/gp-golden.json", import.meta.url)

/** A seeded dataset in the unit cube: N rows of d genes, smooth targets plus a little seeded noise. */
function dataset(seed: number, d: number, n: number): GpData {
  const rng = createRng(seed)
  const x: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: d }, () => rng())
    x.push(row)
    let s = 0
    for (let j = 0; j < d; j++) s += (j % 2 === 0 ? 1 : -1) * (row[j]! - 0.5) * (row[j]! - 0.5) * (1 + j * 0.1)
    y.push(s + 0.1 * (rng() - 0.5))
  }
  return { x, y }
}

/** A θ near the prior mode: each coordinate moved by a seeded offset within ±1.5 sd. */
function thetaNear(seed: number, d: number): Float64Array {
  const rng = createRng(seed)
  const { mode, sd } = priorOf(d)
  return mode.map((m, i) => m + (rng() * 3 - 1.5) * sd[i]!)
}

/** The float64 bits of a number as 16 hex digits: a bit-for-bit encoding (JSON numbers would lose −0). */
function bits(v: number): string {
  const buf = new DataView(new ArrayBuffer(8))
  buf.setFloat64(0, v)
  return buf.getBigUint64(0).toString(16).padStart(16, "0")
}
const hex = (a: ArrayLike<number>) => Array.from(a, bits)

describe("linalg", () => {
  it("cholesky of a known 3×3 SPD matrix equals the hand factor to 1e-15; a rank-1 4×4 matrix needs jitter > 0 and records it; a matrix with a NaN → null", () => {
    // A = L Lᵀ with L = [[2,0,0],[1,3,0],[0.5,-1,1.5]].
    const L = [2, 0, 0, 1, 3, 0, 0.5, -1, 1.5]
    const A = new Float64Array(9)
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += L[i * 3 + k]! * L[j * 3 + k]!
      A[i * 3 + j] = s
    }
    const c = cholesky(A, 3, 0, 1)
    expect(c).not.toBeNull()
    expect(c!.jitter).toBe(0)
    expect(c!.n).toBe(3)
    for (let k = 0; k < 9; k++) expect(Math.abs(c!.L[k]! - L[k]!)).toBeLessThanOrEqual(1e-15)

    // Solves and the inverse agree with A.
    const b = new Float64Array([1, -2, 3])
    const x = solveUpper(c!, solveLower(c!, b))
    for (let i = 0; i < 3; i++) {
      let s = 0
      for (let j = 0; j < 3; j++) s += A[i * 3 + j]! * x[j]!
      expect(s).toBeCloseTo(b[i]!, 13)
    }
    const inv = cholInverse(c!)
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      expect(inv[i * 3 + j]).toBe(inv[j * 3 + i]) // symmetric bit for bit
      let s = 0
      for (let k = 0; k < 3; k++) s += A[i * 3 + k]! * inv[k * 3 + j]!
      expect(s).toBeCloseTo(i === j ? 1 : 0, 13)
    }

    // Rank 1: v vᵀ with v = (1, 2, 3, 4). Without jitter it does not factor; with the schedule it does, and says so.
    const v = [1, 2, 3, 4]
    const R = new Float64Array(16)
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) R[i * 4 + j] = v[i]! * v[j]!
    expect(cholesky(R, 4, 0, 1)).toBeNull()
    expect(cholesky(R, 4, 0, 7)).toBeNull() // 0·10^k is still 0
    const r = cholesky(R, 4, 1e-8, 7)
    expect(r).not.toBeNull()
    expect(r!.jitter).toBeGreaterThan(0)
    expect([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2].map(j => j.toPrecision(12))).toContain(r!.jitter.toPrecision(12))
    for (const e of r!.L) expect(Number.isFinite(e)).toBe(true)

    // A NaN anywhere (even in the unread upper triangle) → null; so does a non-finite jitter.
    for (const at of [0, 1, 5]) {
      const bad = A.slice()
      bad[at] = Number.NaN
      expect(cholesky(bad, 3, 1e-8, 7)).toBeNull()
    }
    const inf = A.slice()
    inf[3] = Infinity
    expect(cholesky(inf, 3, 1e-8, 7)).toBeNull()
    expect(cholesky(A, 3, Number.NaN, 7)).toBeNull()
  })
})

describe("the GP", () => {
  it("matern52(0) = 1; matern52(1) = (1 + √5 + 5/3)·e^(−√5) to 1e-15; decreasing on [0, 10]", () => {
    expect(matern52(0)).toBe(1)
    expect(Math.abs(matern52(1) - (1 + Math.sqrt(5) + 5 / 3) * Math.exp(-Math.sqrt(5)))).toBeLessThanOrEqual(1e-15)
    let prev = matern52(0)
    for (let i = 1; i <= 1000; i++) {
      const k = matern52(i / 100)
      expect(k).toBeLessThan(prev)
      expect(k).toBeGreaterThan(0)
      prev = k
    }
  })

  it("thetaOf and hyperOf are inverse; the prior's mode and sd follow spec §4.3", () => {
    const h: GpHyper = { lengthscales: [0.3, 1.2, 4], outputscale: 0.8, noise: 0.25, mean: -0.1 }
    const back = hyperOf(thetaOf(h), 3)
    expect(back.lengthscales.map(v => v.toPrecision(14))).toEqual(h.lengthscales.map(v => v.toPrecision(14)))
    expect(back.outputscale).toBeCloseTo(h.outputscale, 14)
    expect(back.noise).toBeCloseTo(h.noise, 14)
    expect(back.mean).toBe(h.mean)
    expect(hyperOf(new Float64Array([0, 0, -50, 0]), 1).noise).toBeGreaterThanOrEqual(NOISE_FLOOR)
    const { mode, sd } = priorOf(12)
    expect(mode.length).toBe(15)
    expect(mode[0]).toBeCloseTo(Math.SQRT2 + 0.5 * Math.log(12), 14)
    expect(sd[0]).toBeCloseTo(Math.sqrt(3), 15)
    expect([mode[12], sd[12], mode[13], sd[13], mode[14], sd[14]]).toEqual([0, 1, expect.closeTo(Math.log(0.5), 14), 1, 0, 1])
  })

  it("objective's gradient matches central finite differences (h 1e-6) to 1e-5 relative on 3 seeded datasets (d 3, 12; N 13, 40)", () => {
    const cases: [number, number][] = [[3, 13], [12, 40], [12, 13]]
    let worst = 0
    for (const [ci, [d, n]] of cases.entries()) {
      const data = dataset(100 + ci, d, n)
      const theta = thetaNear(200 + ci, d)
      const o = objective(theta, data)
      expect(o).not.toBeNull()
      expect(Number.isFinite(o!.value)).toBe(true)
      expect(o!.grad.length).toBe(d + 3)
      for (let i = 0; i < d + 3; i++) {
        const plus = theta.slice(); plus[i] = plus[i]! + 1e-6
        const minus = theta.slice(); minus[i] = minus[i]! - 1e-6
        const fd = (objective(plus, data)!.value - objective(minus, data)!.value) / 2e-6
        const g = o!.grad[i]!
        const rel = Math.abs(g - fd) / Math.max(Math.abs(g), Math.abs(fd), 1)
        worst = Math.max(worst, rel)
        expect(rel, `case ${ci} θ[${i}]: analytic ${g}, finite difference ${fd}`).toBeLessThanOrEqual(1e-5)
      }
    }
    expect(worst).toBeLessThan(1e-5)
  })

  it("objective at a θ 7 prior sd from the mode → null", () => {
    const d = 3
    const data = dataset(7, d, 13)
    const { mode, sd } = priorOf(d)
    expect(objective(mode, data)).not.toBeNull()
    for (let i = 0; i < d + 3; i++) {
      for (const sign of [1, -1]) {
        const far = mode.slice(); far[i] = far[i]! + sign * 7 * sd[i]!
        expect(objective(far, data)).toBeNull()
        const near = mode.slice(); near[i] = near[i]! + sign * 5.9 * sd[i]!
        expect(objective(near, data)).not.toBeNull()
      }
    }
    const nan = mode.slice(); nan[0] = Number.NaN
    expect(objective(nan, data)).toBeNull()
  })

  it("predict at a training point with noise 1e-3 is within 0.05 of its y on a smooth fixture; joint cov is symmetric and its diagonal equals variance", () => {
    const x: number[][] = []
    const y: number[] = []
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
      const a = i / 4, b = j / 4
      x.push([a, b])
      y.push(Math.sin(3 * a) - (b - 0.3) * (b - 0.3))
    }
    const hyper: GpHyper = { lengthscales: [0.6, 0.6], outputscale: 1, noise: NOISE_FLOOR, mean: 0 }
    const post = conditionGp({ x, y }, hyper)
    expect(post).not.toBeNull()
    const at = predict(post!, x, false)!
    expect(at.cov).toBeNull()
    for (let i = 0; i < y.length; i++) expect(Math.abs(at.mean[i]! - y[i]!)).toBeLessThan(0.05)
    for (const v of at.variance) expect(v).toBeGreaterThanOrEqual(0)

    const xs = [[0.1, 0.2], [0.9, 0.9], [0.5, 0.5], [0.1, 0.2], [0.33, 0.71]]
    const joint = predict(post!, xs, true)!
    const solo = predict(post!, xs, false)!
    expect(joint.cov).not.toBeNull()
    const K = xs.length
    for (let a = 0; a < K; a++) {
      expect(joint.cov![a * K + a]).toBe(joint.variance[a])
      expect(joint.variance[a]).toBe(solo.variance[a])
      expect(joint.mean[a]).toBe(solo.mean[a])
      for (let b = 0; b < K; b++) expect(joint.cov![a * K + b]).toBe(joint.cov![b * K + a])
    }
    // Far from the data the latent variance returns to the prior's σ_f² and the mean to m.
    const far = predict(post!, [[40, -40]], false)!
    expect(far.variance[0]).toBeCloseTo(1, 10)
    expect(far.mean[0]).toBeCloseTo(0, 10)
  })

  it("predict validates xs: a row of the wrong length or with a non-finite entry → null; a NaN variance surfaces as NaN, never clamped to 0", () => {
    const post = conditionGp({ x: [[0.2, 0.3], [0.7, 0.6]], y: [0.1, -0.2] }, { lengthscales: [0.5, 0.5], outputscale: 1, noise: 0.01, mean: 0 })!
    for (const xs of [[[0.1]], [[0.1, 0.2, 0.3]], [[0.1, Number.NaN]], [[0.1, Infinity]], [[0.5, 0.5], [0.2]]])
      for (const joint of [false, true]) expect(predict(post, xs, joint), JSON.stringify(xs)).toBeNull()
    expect(predictMean(post, [[0.1, Number.NaN]])).toBeNull()
    expect(predict(post, [], true)).toEqual({ mean: new Float64Array(0), cov: new Float64Array(0), variance: new Float64Array(0) })
    // predictMean is predict's mean bit for bit.
    const xs = [[0.1, 0.9], [0.5, 0.5], [0.33, 0.71]]
    expect(Array.from(predictMean(post, xs)!)).toEqual(Array.from(predict(post, xs, false)!.mean))
    // A corrupted factor gives a NaN variance and covariance: surfaced, so callers can refuse it.
    const bad = { ...post, chol: { ...post.chol, L: new Float64Array([Number.NaN, 0, 0, 1]) } }
    const p = predict(bad, xs, true)!
    for (const v of p.variance) expect(Number.isNaN(v)).toBe(true)
    expect(Number.isNaN(p.cov![1])).toBe(true)
  })

  it("two identical x rows and all-equal y: objective and conditionGp return finite values", () => {
    for (const d of [1, 3]) {
      const row = Array.from({ length: d }, (_, j) => 0.25 + 0.1 * j)
      const data: GpData = { x: [row, row.slice(), row.map(v => v + 0.5)], y: [0.7, 0.7, 0.7] }
      const { mode } = priorOf(d)
      const o = objective(mode, data)
      expect(o).not.toBeNull()
      expect(Number.isFinite(o!.value)).toBe(true)
      for (const g of o!.grad) expect(Number.isFinite(g)).toBe(true)
      const post = conditionGp(data, hyperOf(mode, d))
      expect(post).not.toBeNull()
      for (const a of post!.alpha) expect(Number.isFinite(a)).toBe(true)
      const p = predict(post!, [row, [0.9, 0.1, 0.4].slice(0, d)], true)!
      for (const v of [...p.mean, ...p.variance, ...p.cov!]) expect(Number.isFinite(v)).toBe(true)
      // Exactly identical rows only (the kernel matrix is then singular but for the noise): still finite.
      const twin: GpData = { x: [row, row.slice()], y: [0.7, 0.7] }
      expect(objective(mode, twin)).not.toBeNull()
      expect(conditionGp(twin, hyperOf(mode, d))).not.toBeNull()
    }
  })

  it("every output on the fixture inputs equals fixtures/gp-golden.json bit for bit", () => {
    const out: Record<string, unknown> = { fitJitterSteps: FIT_JITTER_STEPS, noiseFloor: bits(NOISE_FLOOR) }
    out.matern52 = hex([0, 0.001, 0.5, 1, 2.5, 7.3].map(matern52))
    const A = new Float64Array([4, 2, 1, 2, 10, -1, 1, -1, 3.25])
    const c = cholesky(A, 3, 1e-8, FIT_JITTER_STEPS)!
    out.cholesky = { L: hex(c.L), jitter: bits(c.jitter), inverse: hex(cholInverse(c)), solve: hex(solveUpper(c, solveLower(c, new Float64Array([1, 2, 3])))) }
    const cases: [number, number][] = [[3, 13], [12, 40]]
    out.cases = cases.map(([d, n], ci) => {
      const data = dataset(300 + ci, d, n)
      const theta = thetaNear(400 + ci, d)
      const prior = priorOf(d)
      const o = objective(theta, data)!
      const hyper = hyperOf(theta, d)
      const post = conditionGp(data, hyper)!
      const xs = dataset(500 + ci, d, 6).x
      const p = predict(post, xs, true)!
      return {
        d, n, theta: hex(theta), priorMode: hex(prior.mode), priorSd: hex(prior.sd), thetaRoundTrip: hex(thetaOf(hyper)),
        objective: bits(o.value), grad: hex(o.grad), jitter: bits(post.chol.jitter), alpha: hex(post.alpha),
        mean: hex(p.mean), variance: hex(p.variance), cov: hex(p.cov!),
      }
    })
    if (process.env.GP_WRITE === "1") {
      expect(process.env.CI, "never regenerate the golden GP outputs in CI").toBeUndefined()
      writeFileSync(FIXTURE, JSON.stringify(out, null, 2) + "\n")
    }
    expect(out).toEqual(JSON.parse(readFileSync(FIXTURE, "utf8")))
  })
})
