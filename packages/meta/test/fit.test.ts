import { describe, it, expect } from "vitest"
import { readFileSync, writeFileSync } from "node:fs"
import { createRng, normInv } from "@blanktab/detmath"
import { minimise } from "../src/lbfgs"
import { fitGp, FIT_STARTS } from "../src/fit"
import { matern52, NOISE_FLOOR, type GpData } from "../src/gp"
import { cholesky } from "../src/linalg"

const FIXTURE = new URL("./fixtures/fit-golden.json", import.meta.url)

/** A standard normal from a seeded uniform (rejecting an exact 0). */
function normal(rng: () => number): number {
  let u = rng()
  while (u === 0) u = rng()
  return normInv(u)
}

/**
 * N rows drawn from a known GP: x uniform in the unit cube, f ~ GP(0, σ_f²·Matérn-5/2 with ARD length scales ell),
 * y = f + N(0, noise). Seeded, so the same rows every run.
 */
function gpDraw(seed: number, ell: number[], n: number, outputscale: number, noise: number): GpData {
  const rng = createRng(seed)
  const d = ell.length
  const x: number[][] = []
  for (let i = 0; i < n; i++) x.push(Array.from({ length: d }, () => rng()))
  const K = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0
    for (let l = 0; l < d; l++) { const u = (x[i]![l]! - x[j]![l]!) / ell[l]!; s += u * u }
    K[i * n + j] = outputscale * matern52(Math.sqrt(s))
  }
  const c = cholesky(K, n, 1e-10, 7)!
  const z = Array.from({ length: n }, () => normal(rng))
  const y: number[] = []
  const sn = Math.sqrt(noise)
  for (let i = 0; i < n; i++) {
    let f = 0
    for (let k = 0; k <= i; k++) f += c.L[i * n + k]! * z[k]!
    y.push(f + sn * normal(rng))
  }
  return { x, y }
}

/** A seeded smooth dataset in d genes (the shape the gate fits: d up to 12). */
function smooth(seed: number, d: number, n: number): GpData {
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

const KNOWN_ELL = [0.2, 1, 5]
const known = () => gpDraw(20261024, KNOWN_ELL, 60, 1, 0.1)

describe("minimise (L-BFGS)", () => {
  it("minimise finds the Rosenbrock minimum (1, 1) to 1e-4 from (−1.2, 1)", () => {
    const rosen = (x: Float64Array) => {
      const a = x[0]!, b = x[1]!
      const value = (1 - a) * (1 - a) + 100 * (b - a * a) * (b - a * a)
      const grad = new Float64Array([-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)])
      return { value, grad }
    }
    const r = minimise(rosen, new Float64Array([-1.2, 1]), { memory: 10, maxIters: 1000, tol: 1e-14 })
    expect(Math.abs(r.x[0]! - 1)).toBeLessThan(1e-4)
    expect(Math.abs(r.x[1]! - 1)).toBeLessThan(1e-4)
    expect(r.converged).toBe(true)
    expect(r.iterations).toBeGreaterThan(0)
  })

  it("a null objective counts as no decrease: a minimum at the edge of a null region is found from inside", () => {
    // f = (x − 2)², undefined for x > 1: the constrained minimum is at the edge, approached from below.
    const f = (x: Float64Array) => x[0]! > 1 ? null : { value: (x[0]! - 2) * (x[0]! - 2), grad: new Float64Array([2 * (x[0]! - 2)]) }
    const r = minimise(f, new Float64Array([0]), { memory: 10, maxIters: 100, tol: 1e-12 })
    expect(r.x[0]!).toBeLessThanOrEqual(1)
    expect(r.x[0]!).toBeGreaterThan(0.99)
    // Null at x0: nothing to minimise.
    const g = minimise(() => null, new Float64Array([0]), { memory: 10, maxIters: 100, tol: 1e-6 })
    expect(g).toEqual({ x: new Float64Array([0]), value: Infinity, iterations: 0, converged: false })
  })
})

describe("fitGp", () => {
  it("fitGp on data drawn from a known GP (d 3, N 60, ℓ [0.2, 1, 5], noise 0.1, seeded) recovers ℓ_1 < ℓ_2 < ℓ_3 and noise within ×3", () => {
    const fit = fitGp(known())
    expect(fit).not.toBeNull()
    const ell = fit!.hyper.lengthscales
    expect(ell[0]!).toBeLessThan(ell[1]!)
    expect(ell[1]!).toBeLessThan(ell[2]!)
    expect(fit!.hyper.noise).toBeGreaterThan(0.1 / 3)
    expect(fit!.hyper.noise).toBeLessThan(0.1 * 3)
    expect(fit!.iterations).toHaveLength(FIT_STARTS)
    expect(Number.isFinite(fit!.objective)).toBe(true)
  })

  it("fitGp twice gives identical bytes (JSON.stringify of the FitResult); the result equals fixtures/fit-golden.json", () => {
    const cases = { known: known(), smooth12: smooth(77, 12, 40) }
    const out: Record<string, string> = {}
    for (const [name, data] of Object.entries(cases)) {
      const a = JSON.stringify(fitGp(data))
      const b = JSON.stringify(fitGp(data))
      expect(a).toBe(b)
      out[name] = a
    }
    if (process.env.FIT_WRITE === "1") {
      expect(process.env.CI, "never regenerate the golden fit in CI").toBeUndefined()
      writeFileSync(FIXTURE, JSON.stringify(out, null, 2) + "\n")
    }
    expect(out).toEqual(JSON.parse(readFileSync(FIXTURE, "utf8")))
  })

  it("a start whose objective is null at θ0 is skipped; all four null → null", () => {
    // Targets so large that (y − m)ᵀK⁻¹(y − m) or its gradient overflows at start 3 (whose long length scales and low noise make
    // the two rows nearly collinear) but not at starts 0–2. Start 3 is skipped; a finite start wins.
    const big = 3e153
    const data: GpData = { x: [[0.1, 0.2], [0.9, 0.8]], y: [big, -big] }
    const fit = fitGp(data)
    expect(fit).not.toBeNull()
    expect(fit!.iterations).toHaveLength(FIT_STARTS)
    expect(fit!.iterations[3]).toBe(0)
    expect(fit!.start).not.toBe(3)
    expect(Number.isFinite(fit!.objective)).toBe(true)
    expect(fit!.hyper.noise).toBeGreaterThan(NOISE_FLOOR)

    // Every start null: a non-finite target, a ragged row, no rows.
    expect(fitGp({ x: [[0.1, 0.2], [0.9, 0.8]], y: [1, Number.NaN] })).toBeNull()
    expect(fitGp({ x: [[0.1, 0.2], [0.9]], y: [1, 2] })).toBeNull()
    expect(fitGp({ x: [], y: [] })).toBeNull()
  })

  it("d = 14 → throws (more searched genes than Sobol dimensions)", () => {
    const data = smooth(5, 14, 20)
    expect(() => fitGp(data)).toThrow(/Sobol/)
    // d = 13 is the most the direction numbers allow (d + 3 = 16).
    expect(() => fitGp(smooth(5, 13, 20))).not.toThrow()
  })
})
