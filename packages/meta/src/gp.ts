// The GP the gate fits and every tab samples from (spec §4.3; plan 05b, Task 2). A constant mean m, the kernel
// σ_f²·Matérn-5/2 with ARD length scales on the unit-cube genes, and homoscedastic noise σ_n² = NOISE_FLOOR + e^θ_n
// (a smooth floor; agent's call, decision 0024). No categoricals: none is active in Phase A, and the genome-space
// parser rejects a categorical gene owned by the GP. All arithmetic is + − × ÷, Math.sqrt and @blanktab/detmath in
// fixed loop orders, so the gate and every browser get the same bits.
import { dexp, dlog } from "@blanktab/detmath"
import { cholesky, cholInverse, solveLower, solveUpper, type Chol } from "./linalg"
import type { GpHyper } from "./posterior"

/** The noise variance never falls below this. */
export const NOISE_FLOOR = 1e-3
/** Cholesky jitter schedule: 1e-8·mean(diag), ×10 per step, at most 1e-2·mean(diag) (spec §4.3). */
export const FIT_JITTER_STEPS = 7
const JITTER_BASE = 1e-8
/** θ more than this many prior sd from the prior mode is outside the search box: objective → null. */
const PRIOR_BOX_SD = 6

const SQRT5 = Math.sqrt(5)
const SQRT3 = Math.sqrt(3)
const LN_2PI = dlog(2 * Math.PI)

/** Training rows (unit-cube x, one row per life) and their targets y, same order. */
export interface GpData { x: readonly (readonly number[])[]; y: readonly number[] }

/** Matérn-5/2 correlation at scaled distance r ≥ 0: (1 + √5 r + 5r²/3)·e^(−√5 r). */
export function matern52(r: number): number {
  const s = SQRT5 * r
  return (1 + s + 5 * r * r / 3) * dexp(-s)
}

/** θ = [ln ℓ_1..ℓ_d, ln σ_f², ln(σ_n² − NOISE_FLOOR), m]. */
export function thetaOf(h: GpHyper): Float64Array {
  const d = h.lengthscales.length
  const theta = new Float64Array(d + 3)
  for (let j = 0; j < d; j++) theta[j] = dlog(h.lengthscales[j]!)
  theta[d] = dlog(h.outputscale)
  theta[d + 1] = dlog(h.noise - NOISE_FLOOR)
  theta[d + 2] = h.mean
  return theta
}

/** The hyperparameters of θ (see thetaOf) for d genes. */
export function hyperOf(theta: Float64Array, d: number): GpHyper {
  const lengthscales: number[] = []
  for (let j = 0; j < d; j++) lengthscales.push(dexp(theta[j]!))
  return { lengthscales, outputscale: dexp(theta[d]!), noise: NOISE_FLOOR + dexp(theta[d + 1]!), mean: theta[d + 2]! }
}

/**
 * The MAP prior, independent normals on θ: ln ℓ_j ~ N(√2 + ½ ln d, √3) (Hvarfner et al. 2024's dimension-scaled
 * prior), ln σ_f² ~ N(0, 1), θ_n ~ N(ln 0.5, 1), m ~ N(0, 1).
 */
export function priorOf(d: number): { mode: Float64Array; sd: Float64Array } {
  const mode = new Float64Array(d + 3)
  const sd = new Float64Array(d + 3)
  const ell = Math.SQRT2 + 0.5 * dlog(d)
  for (let j = 0; j < d; j++) { mode[j] = ell; sd[j] = SQRT3 }
  mode[d] = 0; sd[d] = 1
  mode[d + 1] = dlog(0.5); sd[d + 1] = 1
  mode[d + 2] = 0; sd[d + 2] = 1
  return { mode, sd }
}

/** True when data has ≥ 1 row, every x row has d finite entries, and every y is finite. */
function wellFormed(data: GpData, d: number): boolean {
  const n = data.x.length
  if (n < 1 || data.y.length !== n) return false
  for (let i = 0; i < n; i++) {
    const row = data.x[i]!
    if (row.length !== d || !Number.isFinite(data.y[i]!)) return false
    for (let j = 0; j < d; j++) if (!Number.isFinite(row[j]!)) return false
  }
  return true
}

/** Σ_j ((a_j − b_j)/ℓ_j)². */
function scaledSq(a: readonly number[], b: readonly number[], ell: readonly number[]): number {
  let s = 0
  for (let j = 0; j < ell.length; j++) {
    const u = (a[j]! - b[j]!) / ell[j]!
    s += u * u
  }
  return s
}

/** K = σ_f²·Matérn(X, X) + σ_n² I (no jitter), its mean diagonal, and the Cholesky of K + jitter·I on the schedule. */
function factorKernel(data: GpData, h: GpHyper): { K: Float64Array; meanDiag: number; chol: Chol } | null {
  const n = data.x.length
  const K = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    K[i * n + i] = h.outputscale + h.noise
    for (let j = 0; j < i; j++) {
      const k = h.outputscale * matern52(Math.sqrt(scaledSq(data.x[i]!, data.x[j]!, h.lengthscales)))
      K[i * n + j] = k
      K[j * n + i] = k
    }
  }
  let diag = 0
  for (let i = 0; i < n; i++) diag += K[i * n + i]!
  const meanDiag = diag / n
  const chol = cholesky(K, n, JITTER_BASE * meanDiag, FIT_JITTER_STEPS)
  return chol === null ? null : { K, meanDiag, chol }
}

/**
 * Log marginal likelihood + log prior at θ, and its analytic gradient:
 * ∂/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ) + ∂ log prior/∂θ, with α = K⁻¹(y − m). K includes the jitter that factored it,
 * and the gradient includes the jitter's own dependence on σ_f² and σ_n² (jitter ∝ mean diag = σ_f² + σ_n²).
 * Null if θ is non-finite or outside mode ± 6 sd, if data is malformed, or if no jitter step factors K.
 */
export function objective(theta: Float64Array, data: GpData): { value: number; grad: Float64Array } | null {
  const d = theta.length - 3
  if (d < 1 || !wellFormed(data, d)) return null
  const { mode, sd } = priorOf(d)
  for (let i = 0; i < theta.length; i++) {
    const t = theta[i]!
    if (!Number.isFinite(t)) return null
    const z = t - mode[i]!
    if (z > PRIOR_BOX_SD * sd[i]! || z < -PRIOR_BOX_SD * sd[i]!) return null
  }
  const h = hyperOf(theta, d)
  const f = factorKernel(data, h)
  if (f === null) return null
  const { K, chol, meanDiag } = f
  const n = data.x.length
  const L = chol.L

  const yc = new Float64Array(n)
  for (let i = 0; i < n; i++) yc[i] = data.y[i]! - h.mean
  const alpha = solveUpper(chol, solveLower(chol, yc))
  let fit = 0
  for (let i = 0; i < n; i++) fit += yc[i]! * alpha[i]!
  let logDet = 0
  for (let i = 0; i < n; i++) logDet += dlog(L[i * n + i]!)
  let value = -0.5 * fit - logDet - 0.5 * n * LN_2PI

  const grad = new Float64Array(d + 3)
  const Kinv = cholInverse(chol)
  const sf2 = h.outputscale
  const jitterShare = chol.jitter / meanDiag
  // Diagonal: ∂K_ii/∂ln σ_f² = σ_f²(1 + jitterShare), ∂K_ii/∂θ_n = e^θ_n (1 + jitterShare); length scales: 0.
  let wDiag = 0
  for (let i = 0; i < n; i++) wDiag += alpha[i]! * alpha[i]! - Kinv[i * n + i]!
  const noiseVar = h.noise - NOISE_FLOOR
  // Off-diagonal pairs i < j, each counted once with weight W_ij (the ½ and the symmetric pair cancel).
  const u2 = new Float64Array(d)
  let gSf = 0
  for (let i = 0; i < n; i++) {
    const xi = data.x[i]!
    for (let j = i + 1; j < n; j++) {
      const xj = data.x[j]!
      const w = alpha[i]! * alpha[j]! - Kinv[i * n + j]!
      let r2 = 0
      for (let l = 0; l < d; l++) {
        const u = (xi[l]! - xj[l]!) / h.lengthscales[l]!
        u2[l] = u * u
        r2 += u2[l]!
      }
      const r = Math.sqrt(r2)
      const s = SQRT5 * r
      const e = dexp(-s)
      gSf += w * K[i * n + j]! // ∂K_ij/∂ln σ_f² = K_ij
      // ∂k/∂ln ℓ_l = σ_f²·(5/3)(1 + √5 r)e^(−√5 r)·u_l²
      const dl = w * (sf2 * (5 / 3) * (1 + s) * e)
      for (let l = 0; l < d; l++) grad[l] += dl * u2[l]!
    }
  }
  grad[d] = gSf + 0.5 * wDiag * sf2 * (1 + jitterShare)
  grad[d + 1] = 0.5 * wDiag * noiseVar * (1 + jitterShare)
  let gMean = 0
  for (let i = 0; i < n; i++) gMean += alpha[i]!
  grad[d + 2] = gMean

  // The log prior: Σ −½((θ − μ)/sd)² − ln sd − ½ ln 2π.
  for (let i = 0; i < theta.length; i++) {
    const s = sd[i]!
    const z = (theta[i]! - mode[i]!) / s
    value += -0.5 * z * z - dlog(s) - 0.5 * LN_2PI
    grad[i] += -z / s
  }
  if (!Number.isFinite(value)) return null
  for (let i = 0; i < grad.length; i++) if (!Number.isFinite(grad[i]!)) return null
  return { value, grad }
}

/** A GP conditioned on data: the factor of K + jitter·I and α = (K + jitter·I)⁻¹(y − m). */
export interface GpPosterior { data: GpData; hyper: GpHyper; chol: Chol; alpha: Float64Array }

/** Condition the GP with these hyperparameters on data. Null if data is malformed, a hyperparameter is not a
 * positive finite number (the mean: finite), or no jitter step factors K. */
export function conditionGp(data: GpData, hyper: GpHyper): GpPosterior | null {
  const d = hyper.lengthscales.length
  if (d < 1 || !wellFormed(data, d)) return null
  for (const v of [...hyper.lengthscales, hyper.outputscale, hyper.noise]) if (!(v > 0) || v === Infinity) return null
  if (!Number.isFinite(hyper.mean)) return null
  const f = factorKernel(data, hyper)
  if (f === null) return null
  const n = data.x.length
  const yc = new Float64Array(n)
  for (let i = 0; i < n; i++) yc[i] = data.y[i]! - hyper.mean
  const alpha = solveUpper(f.chol, solveLower(f.chol, yc))
  for (let i = 0; i < n; i++) if (!Number.isFinite(alpha[i]!)) return null
  return { data, hyper, chol: f.chol, alpha }
}

/**
 * The posterior of the latent f (no noise) at xs: mean, variance and, when joint, the K×K covariance (row-major;
 * symmetric bit for bit, its diagonal equal to variance). Variance is clamped at 0 against rounding.
 */
export function predict(post: GpPosterior, xs: readonly (readonly number[])[], joint: boolean): { mean: Float64Array; cov: Float64Array | null; variance: Float64Array } {
  const { data, hyper, chol, alpha } = post
  const n = data.x.length
  const K = xs.length
  const sf2 = hyper.outputscale
  const mean = new Float64Array(K)
  const variance = new Float64Array(K)
  const V: Float64Array[] = []
  const kv = new Float64Array(n)
  for (let a = 0; a < K; a++) {
    const xa = xs[a]!
    let m = 0
    for (let i = 0; i < n; i++) {
      kv[i] = sf2 * matern52(Math.sqrt(scaledSq(xa, data.x[i]!, hyper.lengthscales)))
      m += kv[i]! * alpha[i]!
    }
    mean[a] = hyper.mean + m
    const v = solveLower(chol, kv)
    let q = 0
    for (let i = 0; i < n; i++) q += v[i]! * v[i]!
    const s = sf2 - q
    variance[a] = s > 0 ? s : 0
    if (joint) V.push(v)
  }
  if (!joint) return { mean, cov: null, variance }
  const cov = new Float64Array(K * K)
  for (let a = 0; a < K; a++) {
    cov[a * K + a] = variance[a]!
    const va = V[a]!
    for (let b = a + 1; b < K; b++) {
      const vb = V[b]!
      let q = 0
      for (let i = 0; i < n; i++) q += va[i]! * vb[i]!
      const c = sf2 * matern52(Math.sqrt(scaledSq(xs[a]!, xs[b]!, hyper.lengthscales))) - q
      cov[a * K + b] = c
      cov[b * K + a] = c
    }
  }
  return { mean, cov, variance }
}
