// Thompson sampling on a seeded candidate set, and the incumbent (spec §4.3–§4.4; plan 05b, Task 4). A tab rebuilds
// the exact GP posterior from posterior.json's model (modelPosterior), builds 512 candidates from its draw rng
// (candidateSet), draws one joint sample of f over them and takes the argmax (thompsonPick). The gate finds the
// incumbent the same way, by the posterior mean, on a candidate set seeded by the model's sha (incumbentOf).
//
// Failure is never a NaN: a malformed posterior or candidate, a NaN prediction, or a covariance that no jitter step
// factors gives null, and the caller draws the centre. All arithmetic is + − × ÷, Math.sqrt and @blanktab/detmath in
// fixed loop orders, so the gate and every browser get the same bits.
import { createRng, normCdf, normInv } from "@blanktab/detmath"
import { conditionGp, predict, predictMean, type GpPosterior } from "./gp"
import { cholesky } from "./linalg"
import type { GpModel } from "./posterior"

export const TS_UNIFORM = 256, TS_PARENTS = 16, TS_PER_PARENT = 16, TS_LOCAL_SD = 0.1, TS_JITTER = 1e-6, TS_JITTER_STEPS = 5
/** 2^-33: half the rng's resolution, so u + 2^-33 is in (0, 1) for every rng() value in [0, 1). */
const HALF_ULP_U = 1 / 8589934592
/** The variance floor of the incumbent's difference f(x*) − f(centre) (spec §4.4). */
const DIFF_VAR_FLOOR = 1e-12

/** A standard normal from one rng draw: normInv(u + 2^-33), finite for every u = rng() ∈ [0, 1). */
export function normalOf(u: number): number {
  return normInv(u + HALF_ULP_U)
}

/** v rounded to 6 decimals, as the gate writes unit-cube coordinates. */
export function round6(v: number): number {
  return Math.round(v * 1e6) / 1e6
}

const clip01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)

/** Training-row indices by posterior mean, highest first, ties to the lowest index; a non-finite mean ranks last. */
function rankedRows(post: GpPosterior): number[] {
  const n = post.data.x.length
  const mean = predictMean(post, post.data.x)
  const key = (i: number) => (mean !== null && Number.isFinite(mean[i]!) ? mean[i]! : -Infinity)
  const order = Array.from({ length: n }, (_, i) => i)
  return order.sort((a, b) => key(b) - key(a) || a - b)
}

/**
 * The candidate set, drawing from rng in this order: TS_UNIFORM uniform points (d draws each, row by row); then, for
 * each of the TS_PARENTS training rows with the highest posterior mean (ties to the lowest index), TS_PER_PARENT
 * children x_j + σ_j·normalOf(rng()) per dimension, clipped to [0, 1], with σ_j = min(TS_LOCAL_SD·ℓ_j, TS_LOCAL_SD)
 * (ruling 4: the spec's 0.1·ℓ_j alone would put most children on the cube's faces at the prior's ℓ ≈ 14). With fewer
 * than TS_PARENTS rows the parents cycle in rank order, so there are always 512 candidates [agent's call].
 */
export function candidateSet(post: GpPosterior, rng: () => number): number[][] {
  const ell = post.hyper.lengthscales
  const d = ell.length
  const out: number[][] = []
  for (let i = 0; i < TS_UNIFORM; i++) {
    const row: number[] = []
    for (let j = 0; j < d; j++) row.push(rng())
    out.push(row)
  }
  const n = post.data.x.length
  if (n === 0) return out
  const sd = ell.map(l => Math.min(TS_LOCAL_SD * l, TS_LOCAL_SD))
  const ranked = rankedRows(post)
  for (let k = 0; k < TS_PARENTS; k++) {
    const parent = post.data.x[ranked[k % n]!]!
    for (let c = 0; c < TS_PER_PARENT; c++) {
      const row: number[] = []
      for (let j = 0; j < d; j++) row.push(clip01(parent[j]! + sd[j]! * normalOf(rng())))
      out.push(row)
    }
  }
  return out
}

/**
 * One Thompson sample: f = μ + L·z over the candidates, with L the Cholesky of Σ + jitter·I (TS_JITTER, ×10 for up to
 * TS_JITTER_STEPS steps) and z_i = normalOf(rng()) drawn only once L exists; the pick is the argmax, ties to the
 * lowest index. Null — never a NaN or out-of-cube x — when there are no candidates, a candidate is not d numbers in
 * [0, 1], the prediction is null or not finite, or no jitter step factors Σ.
 */
export function thompsonPick(post: GpPosterior, cands: readonly (readonly number[])[], rng: () => number): { index: number; x: number[] } | null {
  const K = cands.length
  const d = post.hyper.lengthscales.length
  if (K === 0) return null
  for (const c of cands) {
    if (c.length !== d) return null
    for (let j = 0; j < d; j++) if (!(c[j]! >= 0 && c[j]! <= 1)) return null
  }
  const p = predict(post, cands, true)
  if (p === null) return null
  for (let i = 0; i < K; i++) if (!Number.isFinite(p.mean[i]!)) return null
  const chol = cholesky(p.cov!, K, TS_JITTER, TS_JITTER_STEPS) // null on any non-finite entry, too
  if (chol === null) return null
  const z = new Float64Array(K)
  for (let i = 0; i < K; i++) z[i] = normalOf(rng())
  const L = chol.L
  let best = -1
  let bestF = -Infinity
  for (let i = 0; i < K; i++) {
    let f = p.mean[i]!
    for (let k = 0; k <= i; k++) f += L[i * K + k]! * z[k]!
    if (!Number.isFinite(f)) return null
    if (f > bestF) { bestF = f; best = i }
  }
  return best < 0 ? null : { index: best, x: cands[best]!.slice() }
}

/**
 * The incumbent (spec §4.4): x* = the argmax of the posterior mean over candidateSet(post, createRng(seed)), ties to
 * the lowest index, rounded to 6 decimals as the gate writes it. Its mean and pBeatsCentre = Φ((μ* − μ_c)/√max(v,
 * 1e-12)), v = σ*² + σ_c² − 2·cov, come from one joint predict at [x*, centreX], so they describe the written x*.
 * The gate seeds it with hashSeed(0, "incumbent/" + modelSha) (ruling 4). Null on a bad centreX or a non-finite result.
 */
export function incumbentOf(post: GpPosterior, centreX: readonly number[], seed: number): { x: number[]; mean: number; pBeatsCentre: number } | null {
  const cands = candidateSet(post, createRng(seed))
  const means = predictMean(post, cands)
  if (means === null) return null
  let best = -1
  let bestM = -Infinity
  for (let i = 0; i < cands.length; i++) {
    const m = means[i]!
    if (!Number.isFinite(m)) return null
    if (m > bestM) { bestM = m; best = i }
  }
  if (best < 0) return null
  const x = cands[best]!.map(round6)
  const p = predict(post, [x, centreX], true)
  if (p === null) return null
  const mean = p.mean[0]!
  const v = p.variance[0]! + p.variance[1]! - 2 * p.cov![1]!
  const pBeatsCentre = normCdf((mean - p.mean[1]!) / Math.sqrt(Math.max(v, DIFF_VAR_FLOOR)))
  if (!Number.isFinite(mean) || !Number.isFinite(pBeatsCentre)) return null
  return { x, mean, pBeatsCentre }
}

/** The GP posterior a model block ships: conditionGp of its rows' x, its y and its hyperparameters. Null if its genes
 * and length scales disagree in number, or conditionGp refuses it. */
export function modelPosterior(m: GpModel): GpPosterior | null {
  if (m.genes.length !== m.hyper.lengthscales.length) return null
  return conditionGp({ x: m.rows.map(r => r.x), y: m.y }, m.hyper)
}
