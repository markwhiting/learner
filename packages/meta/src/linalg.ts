// Dense linear algebra for the GP (plan 05b, Task 2). Matrices are row-major Float64Array, n×n. Every step is
// + − × ÷ or Math.sqrt in a fixed loop order, so the gate and every browser get the same bits (spec §4.3).

/** A lower-triangular Cholesky factor L of A + jitter·I (upper triangle zero), and the jitter that made it factor. */
export interface Chol { L: Float64Array; n: number; jitter: number }

/** One Cholesky–Banachiewicz attempt on the lower triangle of a + jitter·I; null at a pivot that is not > 0. */
function factor(a: Float64Array, n: number, jitter: number): Float64Array | null {
  const L = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j]!
      if (i === j) s += jitter
      for (let k = 0; k < j; k++) s -= L[i * n + k]! * L[j * n + k]!
      if (i === j) {
        if (!(s > 0) || s === Infinity) return null
        L[i * n + i] = Math.sqrt(s)
      } else {
        L[i * n + j] = s / L[j * n + j]!
      }
    }
  }
  return L
}

/**
 * The Cholesky factor of the symmetric n×n matrix a (its lower triangle is read). Tries jitter = baseJitter·10^k
 * for k = 0..steps−1, added to the diagonal; the first that factors wins and is recorded. Null if none does, or if
 * any entry of a (either triangle) or the jitter is not finite. baseJitter 0 means "no jitter" at every step.
 */
export function cholesky(a: Float64Array, n: number, baseJitter: number, steps: number): Chol | null {
  if (a.length !== n * n || !Number.isFinite(baseJitter) || baseJitter < 0) return null
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) return null
  let tens = 1
  for (let k = 0; k < steps; k++) {
    const jitter = baseJitter * tens
    const L = factor(a, n, jitter)
    if (L !== null) return { L, n, jitter }
    tens *= 10
  }
  return null
}

/** Forward substitution: z with L z = b. */
export function solveLower(c: Chol, b: Float64Array): Float64Array {
  const { L, n } = c
  const z = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]!
    for (let k = 0; k < i; k++) s -= L[i * n + k]! * z[k]!
    z[i] = s / L[i * n + i]!
  }
  return z
}

/** Back substitution: x with Lᵀ x = z. */
export function solveUpper(c: Chol, z: Float64Array): Float64Array {
  const { L, n } = c
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]!
    for (let k = i + 1; k < n; k++) s -= L[k * n + i]! * x[k]!
    x[i] = s / L[i * n + i]!
  }
  return x
}

/** (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹, symmetric bit for bit (each i ≤ j entry is computed once and mirrored). */
export function cholInverse(c: Chol): Float64Array {
  const { L, n } = c
  // M = L⁻¹, lower triangular, column by column (forward substitution on e_j).
  const M = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    M[j * n + j] = 1 / L[j * n + j]!
    for (let i = j + 1; i < n; i++) {
      let s = 0
      for (let k = j; k < i; k++) s -= L[i * n + k]! * M[k * n + j]!
      M[i * n + j] = s / L[i * n + i]!
    }
  }
  // (Mᵀ M)_ij = Σ_{k ≥ max(i, j)} M_ki M_kj.
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0
      for (let k = j; k < n; k++) s += M[k * n + i]! * M[k * n + j]!
      out[i * n + j] = s
      out[j * n + i] = s
    }
  }
  return out
}
