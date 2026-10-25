// The gate's search (spec §4.3–§4.4; plan 05b Task 8): the rows' subset, the GP model the tabs sample from, the
// incumbent and the trigger. buildMeta (meta.ts) calls these only in mode "search": the fit is never run otherwise.
//
// Every result is a pure function of its input. The fit is deterministic (packages/meta fit.ts), so a model whose rows,
// x and y equal the previous one's is reused as it was, without refitting: refitting gives the same bytes.
import { hashSeed } from "@blanktab/detmath"
import {
  fitGp, fromUnit, genomeSha, incumbentOf, modelPosterior, round6, searched, sha256Hex, toUnit,
  type FitResult, type Genome, type GenomeSpace, type GpModel, type Incumbent,
} from "@blanktab/meta"
import { serialise } from "./serialise.ts"

/** The most rows a model carries (spec §4.3); above it, rowSubset keeps a deterministic random subset. */
export const MAX_ROWS = 1000
/** The fewest rows a model is fitted on: one more than the cube's dimension. */
export const MIN_ROWS = (d: number): number => d + 1
/** The trigger (spec §4.4): P(f(x*) − f(centre) > 0) at least this [guess, SYN-M §5.5]. */
export const TRIGGER_P = 0.95
/** The trigger (spec §4.4): at least this many eligible rows since the last declaration [guess]. */
export const TRIGGER_NEW_ROWS = 100
/** The methods whose eligible, verified lives are rows (spec §4.3). */
export const ROW_METHODS = ["calibrate", "uniform", "ts", "centre"] as const
/**
 * The most rows the hyperparameters are fitted on [agent's call, decision 0024 "Rows and the fit"]: the fit costs
 * about n³ per objective, 19.7 s at 500 rows and 177 s at 1,000 on the laptop (CI runners 2–3× slower), and a gate run
 * may rebuild 3 times (commit-meta.ts). Above it, the hyperparameters come from the FIT_ROWS rows of the model with the
 * largest subset hash, and the model conditions on all its rows.
 */
export const FIT_ROWS = 500

const subsetKey = (id: string): number => hashSeed(0, "subset/" + id)
const byId = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

/** The `limit` ids with the largest subset hash (ties to the lower id), sorted by id; all of them, sorted, when ≤ limit. */
function largestBySubsetHash(ids: readonly string[], limit: number): string[] {
  if (ids.length <= limit) return [...ids].sort(byId)
  const keyed = ids.map(id => ({ id, k: subsetKey(id) }))
  keyed.sort((a, b) => b.k - a.k || byId(a.id, b.id))
  return keyed.slice(0, limit).map(e => e.id).sort(byId)
}

/** The rows a model keeps (spec §4.3): all when ≤ MAX_ROWS, else the MAX_ROWS with the largest hashSeed(0, "subset/" + id); sorted by id. */
export function rowSubset(ids: readonly string[]): string[] {
  return largestBySubsetHash(ids, MAX_ROWS)
}

const sameNumbers = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => v === b[i])

/** Whether `m` was built on exactly these genes, rows (learnerId and x) and targets. */
function sameData(m: GpModel, genes: readonly string[], rows: readonly { learnerId: string; x: number[] }[], y: readonly number[]): boolean {
  return m.genes.length === genes.length && m.genes.every((g, j) => g === genes[j])
    && m.rows.length === rows.length && m.rows.every((r, i) => r.learnerId === rows[i]!.learnerId && sameNumbers(r.x, rows[i]!.x))
    && sameNumbers(m.y, y)
}

/** What a build of the model did, for diagnostics.json (never the posterior, whose bytes must not move with it). */
export interface ModelRun {
  /** Rows the model conditions on, and the rows its hyperparameters were fitted on. */
  rows: number; fitRows: number
  /** True when fitGp ran in this build; false when the previous model's fit was reused, or nothing was fitted. */
  refit: boolean
  /** Fit starts that took no step (their objective was null at their start, or they began at a stationary point). */
  startsWithNoStep: number[]
}

/**
 * The model the tabs sample from (spec §4.3), on rows already subset (rowSubset) and their targets y. x is the logged
 * genome in the unit cube, rounded to 6 decimals (and kept in [0, 1]). No model, with the reason, below MIN_ROWS(d),
 * when fitGp throws (malformed data) or finds nothing, or when the fitted model cannot be conditioned. `previous`
 * (the last posterior's model) is reused, hyper and fit, when its genes, rows, x and y are identical.
 */
export function buildModel(
  rows: { learnerId: string; genome: Genome }[], y: number[], space: GenomeSpace, previous: GpModel | null,
): { model: GpModel | null; reason: string | null; run: ModelRun } {
  const genes = searched(space).map(g => g.name)
  const d = genes.length
  const run: ModelRun = { rows: rows.length, fitRows: 0, refit: false, startsWithNoStep: [] }
  const none = (reason: string) => ({ model: null, reason, run })
  if (rows.length !== y.length) throw new Error(`buildModel: ${rows.length} rows but ${y.length} targets`)
  if (rows.length > MAX_ROWS) throw new Error(`buildModel: ${rows.length} rows, more than MAX_ROWS ${MAX_ROWS} (rowSubset first)`)
  if (rows.length < MIN_ROWS(d)) return none(`fewer than ${MIN_ROWS(d)} rows`)
  if (!y.every(Number.isFinite)) return none("a target is not finite")
  const modelRows = rows.map(r => ({ learnerId: r.learnerId, x: toUnit(space, r.genome).map(u => Math.min(1, Math.max(0, round6(u)))) }))

  let fit: FitResult
  if (previous !== null && sameData(previous, genes, modelRows, y)) {
    fit = { hyper: previous.hyper, objective: previous.fit.objective, start: previous.fit.start, iterations: previous.fit.iterations }
    run.fitRows = Math.min(rows.length, FIT_ROWS)
  } else {
    const fitIds = new Set(largestBySubsetHash(modelRows.map(r => r.learnerId), FIT_ROWS))
    const fx: number[][] = [], fy: number[] = []
    modelRows.forEach((r, i) => { if (fitIds.has(r.learnerId)) { fx.push(r.x); fy.push(y[i]!) } })
    run.fitRows = fx.length
    let result: FitResult | null
    try { result = fitGp({ x: fx, y: fy }) } catch (e) { return none(`the fit threw: ${e instanceof Error ? e.message : String(e)}`) }
    run.refit = true
    if (result === null) return none("no fit start had a finite objective")
    fit = result
  }
  run.startsWithNoStep = fit.iterations.flatMap((n, s) => (n === 0 ? [s] : []))
  const model: GpModel = {
    genes, rows: modelRows, y: [...y],
    hyper: fit.hyper, fit: { objective: fit.objective, start: fit.start, iterations: [...fit.iterations] },
  }
  // A model no tab can condition would make every ts birth fall back to centre: serve none, and say why.
  if (modelPosterior(model) === null) return none("the fitted model cannot be conditioned on its rows")
  return { model, reason: null, run }
}

/**
 * The incumbent (spec §4.4): the posterior-mean argmax over the gate's candidate set, seeded with hashSeed(0,
 * "incumbent/" + sha256(serialise(model))) (ruling 4: the spec's posteriorSha is circular), against the centre. Its
 * genomeSha is that of fromUnit(space, x*, centre). Null when the model has no posterior or incumbentOf finds none.
 */
export async function incumbentFor(model: GpModel, space: GenomeSpace, centre: Genome): Promise<Incumbent | null> {
  const post = modelPosterior(model)
  if (post === null) return null
  const modelSha = await sha256Hex(new TextEncoder().encode(serialise(model)))
  const inc = incumbentOf(post, toUnit(space, centre), hashSeed(0, "incumbent/" + modelSha))
  if (inc === null) return null
  return { x: inc.x, genomeSha: await genomeSha(space.schemaVersion, fromUnit(space, inc.x, centre)), mean: inc.mean, pBeatsCentre: inc.pBeatsCentre }
}

/**
 * Whether the gate declares a confirmation (spec §4.4): an incumbent with pBeatsCentre ≥ TRIGGER_P, at least
 * TRIGGER_NEW_ROWS rows since the last declaration (or the signal look), no confirmation in flight, and an incumbent
 * that is not the centre. Task 9 acts on it.
 */
export function triggered(inc: Incumbent | null, centreSha: string, rowsSinceDeclaration: number, inFlight: boolean): boolean {
  return inc !== null && inc.pBeatsCentre >= TRIGGER_P && rowsSinceDeclaration >= TRIGGER_NEW_ROWS && !inFlight && inc.genomeSha !== centreSha
}
