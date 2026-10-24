// The gate's deterministic fit of the GP's hyperparameters (plan 05b, Task 3; spec §4.3): maximise the log
// marginal likelihood + log prior by L-BFGS from 4 fixed starts — the prior mode and 3 scrambled-Sobol points
// within ±1.5 prior sd, seeded "gp-fit/1". The best start wins; ties go to the earliest.
import { sobolPoints, SOBOL_MAX_DIMS } from "@blanktab/detmath"
import { hyperOf, objective, priorOf, type GpData } from "./gp"
import { minimise } from "./lbfgs"
import type { GpHyper } from "./posterior"

export const FIT_SEED = "gp-fit/1", FIT_STARTS = 4, FIT_MAX_ITERS = 100, FIT_TOL = 1e-6, LBFGS_MEMORY = 10, START_SPREAD_SD = 1.5

/** The winning hyperparameters, their objective (log marginal likelihood + log prior), the winning start's index,
 * and the accepted L-BFGS steps of each start (0 for a start whose objective was null at θ0). */
export interface FitResult { hyper: GpHyper; objective: number; start: number; iterations: number[] }

/** The 4 starting θ: the prior mode, then mode + START_SPREAD_SD·sd·(2u − 1) for u in sobolPoints(d + 3, 3, FIT_SEED). */
function startsOf(d: number): Float64Array[] {
  const { mode, sd } = priorOf(d)
  const starts = [Float64Array.from(mode)]
  for (const u of sobolPoints(d + 3, FIT_STARTS - 1, FIT_SEED)) {
    const theta = new Float64Array(d + 3)
    for (let i = 0; i < d + 3; i++) theta[i] = mode[i]! + START_SPREAD_SD * sd[i]! * (2 * u[i]! - 1)
    starts.push(theta)
  }
  return starts
}

/**
 * Fit the GP to data. Null when every start's objective is null at θ0 (malformed data, or a kernel no jitter step
 * factors). Throws when d + 3 exceeds the Sobol direction numbers: searching more than 13 genes needs a recorded
 * decision that also adds direction numbers.
 */
export function fitGp(data: GpData): FitResult | null {
  const d = data.x.length > 0 ? data.x[0]!.length : 0
  if (d + 3 > SOBOL_MAX_DIMS)
    throw new RangeError(`fitGp: ${d} searched genes need ${d + 3} Sobol dimensions; at most ${SOBOL_MAX_DIMS} exist (more needs a recorded decision)`)
  if (d < 1) return null
  const negLogPosterior = (theta: Float64Array) => {
    const o = objective(theta, data)
    if (o === null) return null
    const grad = new Float64Array(o.grad.length)
    for (let i = 0; i < grad.length; i++) grad[i] = -o.grad[i]!
    return { value: -o.value, grad }
  }
  const iterations: number[] = []
  let best: { x: Float64Array; value: number; start: number } | null = null
  const starts = startsOf(d)
  for (let s = 0; s < starts.length; s++) {
    const r = minimise(negLogPosterior, starts[s]!, { memory: LBFGS_MEMORY, maxIters: FIT_MAX_ITERS, tol: FIT_TOL })
    iterations.push(r.iterations)
    if (r.value === Infinity) continue
    if (best === null || r.value < best.value) best = { x: r.x, value: r.value, start: s }
  }
  if (best === null) return null
  return { hyper: hyperOf(best.x, d), objective: -best.value, start: best.start, iterations }
}
