// A deterministic L-BFGS minimiser (plan 05b, Task 3; spec §4.3). No randomness and fixed constants: the two-loop
// recursion, then a backtracking Armijo line search that halves a fixed number of times. Only + − × ÷ in fixed
// loop orders, so every engine takes the same steps.

/** The minimiser's result: the best point reached, its value, the accepted steps, and whether a convergence test
 * (|Δvalue| < tol or ‖g‖∞ < GRAD_TOL) stopped it, rather than a failed line search or the iteration cap. */
export interface LbfgsResult { x: Float64Array; value: number; iterations: number; converged: boolean }

/** Armijo's sufficient-decrease constant c₁. */
const ARMIJO_C1 = 1e-4
/** The line search halves the step at most this many times. */
const MAX_HALVINGS = 30
/** A curvature pair with sᵀy at or below this is skipped. */
const MIN_CURVATURE = 1e-10
/** Stop when ‖g‖∞ falls below this. */
const GRAD_TOL = 1e-8

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!
  return s
}

function normInf(a: Float64Array): number {
  let m = 0
  for (let i = 0; i < a.length; i++) {
    const v = a[i]! < 0 ? -a[i]! : a[i]!
    if (v > m) m = v
  }
  return m
}

/** −H g by the two-loop recursion over the stored pairs (oldest first), with H₀ = γI, γ = sᵀy/yᵀy of the newest. */
function direction(g: Float64Array, S: Float64Array[], Y: Float64Array[], rho: number[]): Float64Array {
  const k = S.length
  const q = Float64Array.from(g)
  const a = new Float64Array(k)
  for (let i = k - 1; i >= 0; i--) {
    const ai = rho[i]! * dot(S[i]!, q)
    a[i] = ai
    const y = Y[i]!
    for (let j = 0; j < q.length; j++) q[j] = q[j]! - ai * y[j]!
  }
  if (k > 0) {
    const y = Y[k - 1]!
    const gamma = dot(S[k - 1]!, y) / dot(y, y)
    for (let j = 0; j < q.length; j++) q[j] = q[j]! * gamma
  }
  for (let i = 0; i < k; i++) {
    const b = rho[i]! * dot(Y[i]!, q)
    const s = S[i]!
    for (let j = 0; j < q.length; j++) q[j] = q[j]! + s[j]! * (a[i]! - b)
  }
  for (let j = 0; j < q.length; j++) q[j] = -q[j]!
  return q
}

/**
 * Minimise f from x0. f returns null where it is undefined; a null counts as no decrease in the line search, and a
 * null at x0 returns x0 with value Infinity and no iterations. Stops when |Δvalue| < tol, when ‖g‖∞ < 1e-8, when
 * the line search fails, or after maxIters accepted steps. The first step length is 1/max(1, ‖g‖∞) on the first
 * iteration and 1 after that. If the two-loop direction is not a descent direction (rounding), the memory is
 * cleared and the step is along −g.
 */
export function minimise(f: (x: Float64Array) => { value: number; grad: Float64Array } | null, x0: Float64Array,
  opts: { memory: number; maxIters: number; tol: number }): LbfgsResult {
  const n = x0.length
  let x = Float64Array.from(x0)
  let cur = f(x)
  if (cur === null) return { x, value: Infinity, iterations: 0, converged: false }
  const S: Float64Array[] = []
  const Y: Float64Array[] = []
  const rho: number[] = []
  let iterations = 0
  let converged = false
  while (iterations < opts.maxIters) {
    const g = cur.grad
    const gInf = normInf(g)
    if (gInf < GRAD_TOL) { converged = true; break }
    let p = direction(g, S, Y, rho)
    let slope = dot(g, p)
    if (!(slope < 0)) {
      S.length = 0; Y.length = 0; rho.length = 0
      p = new Float64Array(n)
      for (let j = 0; j < n; j++) p[j] = -g[j]!
      slope = dot(g, p)
    }
    let step = iterations === 0 ? 1 / (gInf > 1 ? gInf : 1) : 1
    let next: { value: number; grad: Float64Array } | null = null
    const xt = new Float64Array(n)
    for (let h = 0; h <= MAX_HALVINGS; h++) {
      for (let j = 0; j < n; j++) xt[j] = x[j]! + step * p[j]!
      const ft = f(xt)
      if (ft !== null && ft.value <= cur.value + ARMIJO_C1 * step * slope) { next = ft; break }
      step *= 0.5
    }
    if (next === null) break
    const s = new Float64Array(n)
    const y = new Float64Array(n)
    for (let j = 0; j < n; j++) { s[j] = xt[j]! - x[j]!; y[j] = next.grad[j]! - g[j]! }
    const sy = dot(s, y)
    if (sy > MIN_CURVATURE) {
      if (S.length === opts.memory) { S.shift(); Y.shift(); rho.shift() }
      S.push(s); Y.push(y); rho.push(1 / sy)
    }
    const delta = cur.value - next.value
    x = xt
    cur = next
    iterations++
    if ((delta < 0 ? -delta : delta) < opts.tol) { converged = true; break }
  }
  return { x, value: cur.value, iterations, converged }
}
