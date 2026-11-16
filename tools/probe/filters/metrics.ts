// The probe's similarity metrics, all from the centred Gram (plan 05c Task 4), in float64: linear CKA between two
// states' whole layers [KUH K27], each unit's signed correlation with each unit of another state, and the median
// correlation after matching units one to one by Hungarian matching [KUH K26].
import type { Gram } from "./layer.ts"

/** Units per state: GPU widths' conv1. */
export const UNITS = 16

/** ‖K‖²_F of the 16×16 block (a, b) of G: K = the centred cross-products of state a's and state b's units. */
function blockNorm2(G: Gram, a: number, b: number): number {
  let s = 0
  for (let i = 0; i < UNITS; i++) {
    const r = (a * UNITS + i) * G.cols + b * UNITS
    for (let j = 0; j < UNITS; j++) { const v = G.g[r + j]!; s += v * v }
  }
  return s
}

/** Linear CKA of states a and b: ‖K_ab‖²_F / (‖K_aa‖_F · ‖K_bb‖_F); null if either state's block is all zero. */
export function cka(G: Gram, a: number, b: number): number | null {
  const aa = blockNorm2(G, a, a), bb = blockNorm2(G, b, b)
  if (aa === 0 || bb === 0) return null
  return blockNorm2(G, a, b) / (Math.sqrt(aa) * Math.sqrt(bb))
}

/** c[i·16 + j]: the correlation of state a's unit i with state b's unit j over the probe samples; 0 when either is varZero. */
export function unitCorrelation(G: Gram, a: number, b: number): Float64Array {
  const c = new Float64Array(UNITS * UNITS)
  for (let i = 0; i < UNITS; i++) {
    const ci = a * UNITS + i
    if (G.varZero[ci] === 1) continue
    for (let j = 0; j < UNITS; j++) {
      const cj = b * UNITS + j
      if (G.varZero[cj] === 1) continue
      c[i * UNITS + j] = G.g[ci * G.cols + cj]! / Math.sqrt(G.g[ci * G.cols + ci]! * G.g[cj * G.cols + cj]!)
    }
  }
  return c
}

/**
 * The assignment row i → column that maximises Σ c[i, col(i)] over an n×n matrix (Kuhn–Munkres with potentials, on
 * −c, O(n³), float64). Deterministic on ties: rows are added in order, and when two columns tie in the shortest-path
 * step, a free column beats an assigned one, then the lower column index wins. So an all-equal matrix gives the identity.
 */
export function hungarianMax(c: Float64Array, n: number): Int32Array {
  const INF = Infinity
  const u = new Float64Array(n + 1), v = new Float64Array(n + 1)
  const p = new Int32Array(n + 1), way = new Int32Array(n + 1)
  const minv = new Float64Array(n + 1), used = new Uint8Array(n + 1)
  for (let i = 1; i <= n; i++) {
    p[0] = i
    let j0 = 0
    minv.fill(INF); used.fill(0)
    do {
      used[j0] = 1
      const i0 = p[j0]!
      let delta = INF, j1 = 0
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue
        const cur = -c[(i0 - 1) * n + (j - 1)]! - u[i0]! - v[j]!
        if (cur < minv[j]!) { minv[j] = cur; way[j] = j0 }
        if (minv[j]! < delta || (minv[j] === delta && p[j] === 0 && p[j1] !== 0)) { delta = minv[j]!; j1 = j }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]!]! += delta; v[j]! -= delta } else minv[j]! -= delta
      }
      j0 = j1
    } while (p[j0] !== 0)
    do { const j1 = way[j0]!; p[j0] = p[j1]!; j0 = j1 } while (j0 !== 0)
  }
  const col = new Int32Array(n)
  for (let j = 1; j <= n; j++) col[p[j]! - 1] = j - 1
  return col
}

/** The median of 16 values: the mean of the 8th and 9th sorted. */
export function median(xs: ArrayLike<number>): number {
  const s = Float64Array.from(xs).sort()
  const m = s.length >> 1
  return s.length % 2 === 1 ? s[m]! : (s[m - 1]! + s[m]!) / 2
}

/** The median, over state a's 16 units, of each unit's correlation with its Hungarian-matched unit of state b (signed). */
export function matchedMedian(G: Gram, a: number, b: number): number {
  const c = unitCorrelation(G, a, b)
  const col = hungarianMax(c, UNITS)
  return median(Array.from(col, (j, i) => c[i * UNITS + j]!))
}
