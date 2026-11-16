// Pure statistics for tools/eval/flat-diagnosis.ts (the 2026-11-16 calibration-flat diagnosis): one-way random-effects
// ANOVA with an F-based ICC interval, the F and t quantiles behind it, analytic power of the random-effects F test,
// Spearman correlation and seeded samplers. Analysis only: nothing here scores a life (the evaluators do that) and
// nothing here is read by the gate. Uses Math.log/exp freely: these numbers are a research note's, not a protocol's.
import { chiSquareSf, createRng, lnGamma } from "@blanktab/detmath"
import { midRanks } from "../../packages/evaluators/src/stats.ts"

/**
 * The regularised incomplete beta I_x(a, b), by the continued fraction (modified Lentz), using the symmetry
 * I_x(a, b) = 1 − I_{1−x}(b, a) where it converges faster. Accurate to ~1e-12 for the a, b used here.
 */
export function betaInc(x: number, a: number, b: number): number {
  if (!(a > 0 && b > 0)) throw new Error("betaInc needs a, b > 0")
  if (x <= 0) return 0
  if (x >= 1) return 1
  const lnFront = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x)
  if (x > (a + 1) / (a + b + 2)) return 1 - betaInc(1 - x, b, a)
  const TINY = 1e-300
  let c = 1, d = 1 - ((a + b) * x) / (a + 1)
  if (Math.abs(d) < TINY) d = TINY
  d = 1 / d
  let h = d
  for (let m = 1; m <= 1000; m++) {
    const m2 = 2 * m
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2))
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY
    c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY
    d = 1 / d; h *= d * c
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1))
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY
    c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY
    d = 1 / d
    const del = d * c
    h *= del
    if (Math.abs(del - 1) < 1e-14) break
  }
  return (Math.exp(lnFront) * h) / a
}

/** P(F ≤ x) for F with (d1, d2) degrees of freedom. */
export const fCdf = (x: number, d1: number, d2: number): number => (x <= 0 ? 0 : betaInc((d1 * x) / (d1 * x + d2), d1 / 2, d2 / 2))

/** The p-quantile of F(d1, d2), by bisection on fCdf (monotone), to a relative 1e-10. */
export function fQuantile(p: number, d1: number, d2: number): number {
  if (!(p > 0 && p < 1)) throw new Error("fQuantile needs 0 < p < 1")
  let lo = 0, hi = 1
  while (fCdf(hi, d1, d2) < p) hi *= 2
  for (let i = 0; i < 200 && hi - lo > 1e-10 * hi; i++) {
    const mid = (lo + hi) / 2
    if (fCdf(mid, d1, d2) < p) lo = mid; else hi = mid
  }
  return (lo + hi) / 2
}

/** The two-sided critical t at level `alpha` with `df` degrees of freedom (t² ~ F(1, df)). */
export const tCrit = (alpha: number, df: number): number => Math.sqrt(fQuantile(1 - alpha, 1, df))

/** The p-quantile of χ²(df), by bisection on detmath's chiSquareSf. */
export function chiSquareQuantile(p: number, df: number): number {
  let lo = 0, hi = df + 10
  while (1 - chiSquareSf(hi, df) < p) hi *= 2
  for (let i = 0; i < 200 && hi - lo > 1e-10 * hi; i++) {
    const mid = (lo + hi) / 2
    if (1 - chiSquareSf(mid, df) < p) lo = mid; else hi = mid
  }
  return (lo + hi) / 2
}

export interface Anova1 {
  k: number; N: number; n0: number; msb: number; msw: number; F: number; p: number
  /** Method-of-moments variance components: between = (MSB − MSW)/n0 (may be negative), within = MSW. */
  sb2: number; sw2: number
  /** ICC(1) = (MSB − MSW)/(MSB + (n0 − 1)MSW), not truncated (the evaluator's icc1 truncates at 0). */
  icc: number
  grand: number
}

/** One-way ANOVA over `groups` (unequal sizes), with n0 = (N − Σn_g²/N)/(k − 1). Needs k ≥ 2 and N > k. */
export function anova1(groups: readonly (readonly number[])[]): Anova1 {
  const k = groups.length
  const N = groups.reduce((s, g) => s + g.length, 0)
  if (k < 2 || N <= k || groups.some(g => g.length === 0)) throw new Error("anova1 needs ≥ 2 non-empty groups and N > k")
  let total = 0
  for (const g of groups) for (const x of g) total += x
  const grand = total / N
  let ssb = 0, ssw = 0, sumN2 = 0
  for (const g of groups) {
    const m = g.reduce((s, x) => s + x, 0) / g.length
    ssb += g.length * (m - grand) * (m - grand)
    for (const x of g) ssw += (x - m) * (x - m)
    sumN2 += g.length * g.length
  }
  const msb = ssb / (k - 1), msw = ssw / (N - k)
  const n0 = (N - sumN2 / N) / (k - 1)
  const F = msw > 0 ? msb / msw : (msb > 0 ? Infinity : 0)
  const p = Number.isFinite(F) ? 1 - fCdf(F, k - 1, N - k) : 0
  const den = msb + (n0 - 1) * msw
  return { k, N, n0, msb, msw, F, p, sb2: (msb - msw) / n0, sw2: msw, icc: den > 0 ? (msb - msw) / den : 0, grand }
}

/**
 * The exact (balanced) / approximate (unbalanced, n0) two-sided CI for ICC(1) from the F ratio (Shrout & Fleiss 1979;
 * Donner 1986): F_L = F / F_{1−α/2}(k−1, N−k), F_U = F / F_{α/2}(k−1, N−k), ICC = (F* − 1)/(F* + n0 − 1), clipped to [0, 1].
 */
export function iccCi(a: Anova1, alpha = 0.05): [number, number] {
  if (!Number.isFinite(a.F)) return [1, 1]
  const d1 = a.k - 1, d2 = a.N - a.k
  const fl = a.F / fQuantile(1 - alpha / 2, d1, d2), fu = a.F / fQuantile(alpha / 2, d1, d2)
  const c = (f: number) => Math.min(1, Math.max(0, (f - 1) / (f + a.n0 - 1)))
  return [c(fl), c(fu)]
}

/**
 * Power of the one-way random-effects F test with k groups of n at level alpha, under normal effects and errors with
 * intraclass correlation `icc`: F/(1 + n·icc/(1 − icc)) ~ F(k − 1, k(n − 1)), so power = P(F > crit/λ).
 */
export function powerAnovaRandom(icc: number, k: number, n: number, alpha: number): number {
  if (icc <= 0) return alpha
  if (icc >= 1) return 1
  const d1 = k - 1, d2 = k * (n - 1)
  const lambda = 1 + (n * icc) / (1 - icc)
  return 1 - fCdf(fQuantile(1 - alpha, d1, d2) / lambda, d1, d2)
}

/** The least n (2…maxN) at which powerAnovaRandom reaches `target`, or null if it never does. */
export function repeatsFor(icc: number, k: number, alpha: number, target = 0.8, maxN = 5000): number | null {
  if (icc <= 0) return null
  for (let n = 2; n <= maxN; n++) if (powerAnovaRandom(icc, k, n, alpha) >= target) return n
  return null
}

/** Pearson correlation; 0 when either side is constant. */
export function pearson(x: readonly number[], y: readonly number[]): number {
  const n = x.length
  if (n !== y.length || n < 2) throw new Error("pearson needs two equal-length samples of ≥ 2")
  const mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) { const dx = x[i]! - mx, dy = y[i]! - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0
}

/** Spearman's rho: Pearson on mid-ranks (the evaluators' midRanks). */
export const spearman = (x: readonly number[], y: readonly number[]): number => pearson(midRanks(x), midRanks(y))

/** A seeded sampler: uniform, standard normal (Box–Muller), gamma (Marsaglia–Tsang), beta, an index, a shuffle. */
export function sampler(seed: number) {
  const u = createRng(seed)
  let spare: number | null = null
  const normal = (): number => {
    if (spare !== null) { const s = spare; spare = null; return s }
    let a = 0
    while (a === 0) a = u()
    const r = Math.sqrt(-2 * Math.log(a)), t = 2 * Math.PI * u()
    spare = r * Math.sin(t)
    return r * Math.cos(t)
  }
  const gamma = (shape: number): number => {
    if (shape < 1) return gamma(shape + 1) * Math.pow(u() || Number.MIN_VALUE, 1 / shape)
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d)
    for (;;) {
      let x: number, v: number
      do { x = normal(); v = 1 + c * x } while (v <= 0)
      v = v * v * v
      const w = u()
      if (w < 1 - 0.0331 * x * x * x * x || Math.log(w) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
    }
  }
  const beta = (a: number, b: number): number => { const x = gamma(a), y = gamma(b); return x / (x + y) }
  const index = (n: number): number => Math.floor(u() * n)
  const shuffle = <T>(xs: readonly T[]): T[] => {
    const out = [...xs]
    for (let i = out.length - 1; i > 0; i--) { const j = index(i + 1); [out[i], out[j]] = [out[j]!, out[i]!] }
    return out
  }
  return { u, normal, gamma, beta, index, shuffle }
}

/**
 * Between- and within-group covariance matrices of vector observations by MANOVA method of moments (groups of rows,
 * each row a vector of the same length): Σ_w = SSCP_w/(N − k), Σ_b = (SSCP_b/(k − 1) − Σ_w)/n0.
 */
export function covComponents(groups: readonly (readonly (readonly number[])[])[]): { between: number[][]; within: number[][]; mean: number[] } {
  const k = groups.length
  const rows = groups.flat()
  const N = rows.length, d = rows[0]!.length
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[j]!, 0) / N)
  const zero = () => Array.from({ length: d }, () => new Array<number>(d).fill(0))
  const sb = zero(), sw = zero()
  let sumN2 = 0
  for (const g of groups) {
    const m = Array.from({ length: d }, (_, j) => g.reduce((s, r) => s + r[j]!, 0) / g.length)
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
      sb[i]![j]! += g.length * (m[i]! - mean[i]!) * (m[j]! - mean[j]!)
      for (const r of g) sw[i]![j]! += (r[i]! - m[i]!) * (r[j]! - m[j]!)
    }
    sumN2 += g.length * g.length
  }
  const n0 = (N - sumN2 / N) / (k - 1)
  const within = sw.map(r => r.map(v => v / (N - k)))
  const between = sb.map((r, i) => r.map((v, j) => (v / (k - 1) - within[i]![j]!) / n0))
  return { between, within, mean }
}

/** A symmetric 2×2 matrix with negative eigenvalues set to 0 (the nearest PSD matrix in Frobenius norm). */
export function psd2(m: readonly (readonly number[])[]): number[][] {
  const a = m[0]![0]!, b = (m[0]![1]! + m[1]![0]!) / 2, c = m[1]![1]!
  const tr = a + c, det = a * c - b * b
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det))
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc
  // eigenvector of l1: (b, l1 − a), or an axis when b = 0
  let v1x = b, v1y = l1 - a
  if (Math.abs(b) < 1e-300) { v1x = a >= c ? 1 : 0; v1y = a >= c ? 0 : 1 }
  const nn = Math.hypot(v1x, v1y) || 1
  v1x /= nn; v1y /= nn
  const p1 = Math.max(0, l1), p2 = Math.max(0, l2)
  // v2 = (−v1y, v1x)
  return [
    [p1 * v1x * v1x + p2 * v1y * v1y, p1 * v1x * v1y - p2 * v1x * v1y],
    [p1 * v1x * v1y - p2 * v1x * v1y, p1 * v1y * v1y + p2 * v1x * v1x],
  ]
}

/** Holm-adjusted p-values (monotone, capped at 1), in input order. */
export function holmAdjust(ps: readonly number[]): number[] {
  const m = ps.length
  const order = ps.map((_, i) => i).sort((x, y) => ps[x]! - ps[y]! || x - y)
  const out = new Array<number>(m)
  let run = 0
  order.forEach((j, i) => { run = Math.max(run, Math.min(1, (m - i) * ps[j]!)); out[j] = run })
  return out
}
