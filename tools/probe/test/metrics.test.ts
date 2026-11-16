import { describe, it, expect } from "vitest"
import { createRng, hashSeed } from "@blanktab/detmath"
import type { Gram } from "../filters/layer.ts"
import { UNITS, cka, hungarianMax, matchedMedian, unitCorrelation } from "../filters/metrics.ts"

/** Seeded standard normals (Box–Muller). */
function gauss(seed: number): () => number {
  const r = createRng(seed)
  return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r())
}
/** A Gram built naively from raw sample columns (centred, float64), 16 columns per state. */
function gramOf(cols: Float64Array[]): Gram {
  const n = cols.length, N = cols[0]!.length
  const means = Float64Array.from(cols, c => c.reduce((a, v) => a + v, 0) / N)
  const varZero = Uint8Array.from(cols, c => (c.every(v => v === c[0]) ? 1 : 0))
  const g = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (varZero[i] || varZero[j]) continue
    let s = 0
    for (let p = 0; p < N; p++) s += (cols[i]![p]! - means[i]!) * (cols[j]![p]! - means[j]!)
    g[i * n + j] = s
  }
  return { cols: n, g, means, varZero, samples: N }
}
const block = (seed: number, N: number): Float64Array[] => { const z = gauss(seed); return Array.from({ length: UNITS }, () => Float64Array.from({ length: N }, z)) }
/** A seeded orthogonal 16×16 matrix (Gram–Schmidt on Gaussian rows). */
function orthogonal(seed: number): number[][] {
  const z = gauss(seed), q: number[][] = []
  for (let i = 0; i < UNITS; i++) {
    let v = Array.from({ length: UNITS }, z)
    for (const u of q) { const d = v.reduce((a, x, k) => a + x * u[k]!, 0); v = v.map((x, k) => x - d * u[k]!) }
    const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0))
    q.push(v.map(x => x / n))
  }
  return q
}
const mix = (cols: Float64Array[], m: number[][]) => m.map(row => Float64Array.from(cols[0]!, (_, p) => row.reduce((a, w, k) => a + w * cols[k]![p]!, 0)))
const perm = (seed: number, n: number) => { const r = createRng(seed), a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j]!, a[i]!] } return a }

describe("filter probe metrics", () => {
  it("cka(X, X) = 1; cka is unchanged when one state's units are permuted, rotated by a seeded orthogonal 16×16 matrix, or scaled by 3", () => {
    const X = block(1, 2000).map(c => c.map((v, p) => Math.max(0, v + 0.3 * Math.sin(p))))
    const Y = X.map((c, i) => c.map((v, p) => v + 0.8 * Math.cos(p * (i + 1))))
    const pi = perm(5, UNITS)
    const variants = [Y, pi.map(i => Y[i]!), mix(Y, orthogonal(7)), Y.map(c => c.map(v => 3 * v))]
    const G0 = gramOf([...X, ...X])
    expect(cka(G0, 0, 1)).toBeCloseTo(1, 12)
    expect(cka(G0, 0, 0)).toBeCloseTo(1, 12)
    const base = cka(gramOf([...X, ...Y]), 0, 1)!
    expect(base).toBeGreaterThan(0.05)
    expect(base).toBeLessThan(0.99)
    for (const V of variants) expect(cka(gramOf([...X, ...V]), 0, 1)).toBeCloseTo(base, 10)
  })

  it("cka of two independent seeded Gaussian blocks on 10,000 samples is below 0.05", () => {
    expect(cka(gramOf([...block(11, 10_000), ...block(12, 10_000)]), 0, 1)).toBeLessThan(0.05)
  })

  it("hungarianMax equals brute force over all permutations on 200 seeded matrices, n = 2..7, to 1e-12 in the matched sum", () => {
    const perms = (n: number): number[][] => (n === 0 ? [[]] : perms(n - 1).flatMap(p => Array.from({ length: n }, (_, k) => [...p.slice(0, k), n - 1, ...p.slice(k)])))
    const r = createRng(hashSeed(0, "hungarian"))
    for (let t = 0; t < 200; t++) {
      const n = 2 + (t % 6)
      // Some matrices with ties: values on a coarse grid.
      const c = Float64Array.from({ length: n * n }, () => (t % 3 === 0 ? Math.round(r() * 4) / 4 : 2 * r() - 1))
      const col = hungarianMax(c, n)
      expect(new Set(col).size).toBe(n)
      const got = Array.from(col).reduce((a, j, i) => a + c[i * n + j]!, 0)
      const best = Math.max(...perms(n).map(p => p.reduce((a, j, i) => a + c[i * n + j]!, 0)))
      expect(Math.abs(got - best), `t ${t}`).toBeLessThanOrEqual(1e-12)
    }
  })

  it("hungarianMax on an all-equal matrix returns the identity; on a permuted identity it returns that permutation", () => {
    for (const n of [1, 2, 5, 16]) {
      expect(Array.from(hungarianMax(new Float64Array(n * n).fill(0.25), n))).toEqual(Array.from({ length: n }, (_, i) => i))
      expect(Array.from(hungarianMax(new Float64Array(n * n), n))).toEqual(Array.from({ length: n }, (_, i) => i))
      const pi = perm(n, n)
      const c = new Float64Array(n * n)
      pi.forEach((j, i) => { c[i * n + j] = 1 })
      expect(Array.from(hungarianMax(c, n))).toEqual(pi)
    }
  })

  it("matchedMedian of a state against a unit-permuted copy of itself is 1", () => {
    const X = block(21, 3000).map(c => c.map(v => Math.max(0, v)))
    expect(matchedMedian(gramOf([...X, ...perm(3, UNITS).map(i => X[i]!)]), 0, 1)).toBeCloseTo(1, 12)
  })

  it("a dead unit's correlations are 0 and it is matched like any other unit", () => {
    const X = block(31, 2000).map(c => c.map(v => Math.max(0, v)))
    X[4] = new Float64Array(2000) // dead
    const Y = X.map(c => Float64Array.from(c))
    const G = gramOf([...X, ...Y])
    expect(G.varZero[4]).toBe(1)
    const c = unitCorrelation(G, 0, 1)
    for (let j = 0; j < UNITS; j++) { expect(c[4 * UNITS + j]).toBe(0); expect(c[j * UNITS + 4]).toBe(0) }
    const col = hungarianMax(c, UNITS)
    expect(new Set(col).size).toBe(UNITS)
    expect(col[4]).toBe(4) // the only column left once every live unit has its own twin
    expect(matchedMedian(G, 0, 1)).toBeCloseTo(1, 12) // 15 ones and one 0: the median is 1
  })
})
