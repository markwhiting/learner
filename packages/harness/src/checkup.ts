// Checkups (spec §8.3, decision 0016): the milestone suite, each milestone in its own fresh variant world, driven by
// the learner's frozen run. A frozen run changes nothing in the learner but stats.evalCU, so there is nothing to
// restore afterwards. Everything a milestone reads (state, trace) is privileged and harness only.
import type { TrainableLearner } from "@blanktab/learner"
import { hashSeed, type CanvasLike } from "@blanktab/world"
import { ERROR_REASON_RE, type CheckupResult, type MetaBlock } from "./log"
import { MILESTONES_V1, recordRun, type Milestone } from "./milestones"

/** hashSeed(0, `${learnerId}/${level}/${milestoneId}`): a variant this learner has never seen. */
export function checkupSeed(learnerId: string, level: number, milestoneId: string): number {
  return hashSeed(0, `${learnerId}/${level}/${milestoneId}`)
}

/**
 * hashSeed(0, `panel/${panel}/${level}/${milestoneId}`): the shared variant every life on panel `panel` meets (spec
 * §4.1 step 4; decision 0022 §Shared panels). Lives on one panel are compared on the same worlds (blocking).
 */
export function panelSeed(panel: number, level: number, milestoneId: string): number {
  return hashSeed(0, `panel/${panel}/${level}/${milestoneId}`)
}

/**
 * hashSeed(0, `confirm/${evalSeed}/${level}/${milestoneId}`): the variant a confirmation life with seed `evalSeed` meets
 * (spec §4.5; decision 0024 "Confirmation seeds"). Every arm's life with that seed meets the same worlds, and the
 * "confirm/" prefix keeps them apart from every shared panel's (panelSeed), which search lives meet.
 */
export function confirmSeed(evalSeed: number, level: number, milestoneId: string): number {
  return hashSeed(0, `confirm/${evalSeed}/${level}/${milestoneId}`)
}

/**
 * The checkup variant seeds a life with meta block `m` meets: confirmSeed(evalSeed, …) for a confirmation life,
 * panelSeed(evalPanel, …) for a life on a shared panel, else undefined (runCheckup's default, checkupSeed: a life
 * without meta, or a legacy one). The one rule, shared by the session and the newborn runner.
 */
export function variantSeedOf(m: Pick<MetaBlock, "evalPanel" | "evalSeed"> | undefined): ((level: number, milestoneId: string) => number) | undefined {
  if (m === undefined) return undefined
  const { evalSeed, evalPanel } = m
  if (evalSeed !== null) return (l, id) => confirmSeed(evalSeed, l, id)
  if (evalPanel !== null) return (l, id) => panelSeed(evalPanel, l, id)
  return undefined
}

/**
 * `${name}: ${message}` (name: e.name for an Error, else "thrown"), whitespace runs become " ", anything outside printable
 * ASCII "?", at most 80 characters, never empty ("unknown"). Short and bounded so a log stays small: no stack, and at most
 * an 80-character fragment of the message (decision 0017). An empty message leaves just the name.
 */
export function errorReason(e: unknown): string {
  const clean = (t: unknown) => String(t).replace(/\s+/g, " ").replace(/[^\x20-\x7E]/g, "?").trim()
  let name = ""
  let message = ""
  try {
    if (e instanceof Error) { name = clean(e.name); message = clean(e.message) } else { name = "thrown"; message = clean(e) }
  } catch {
    // an object whose toString or name throws: leave what we have
  }
  const text = (e instanceof Error ? (message ? (name ? `${name}: ${message}` : message) : name) : message ? `${name}: ${message}` : "").slice(0, 80).trimEnd()
  return ERROR_REASON_RE.test(text) ? text : "unknown"
}

export interface CheckupDeps {
  learnerId: string
  /** The log-schedule CU threshold this checkup is for. */
  level: number
  /** The learner's steps when it runs. */
  steps: number
  learner: Pick<TrainableLearner, "frozenRun" | "stats">
  /** A fresh canvas for each variant world (and for M5's reference). */
  canvas(): CanvasLike
  suite?: readonly Milestone[] // default MILESTONES_V1
  /** Errors the session handles itself (e.g. a lost device): rethrown after cleanup, not recorded. */
  recoverable?: (e: unknown) => boolean
  /** Checked before each milestone and after the last: true abandons the checkup. */
  shouldStop?: () => boolean
  /** Awaited after every frozen step (the session's pacer); without it, recordRun yields (setTimeout 0) every YIELD_EVERY steps. */
  yieldNow?: () => Promise<void>
  /**
   * Each milestone's variant seed. Default checkupSeed(learnerId, level, id): a life without meta, or a legacy life.
   * A life with a shared panel (meta.evalPanel) passes (l, m) => panelSeed(panel, l, m) (decision 0022); a confirmation
   * life (meta.evalSeed) passes (l, m) => confirmSeed(evalSeed, l, m) (decision 0024).
   */
  variantSeed?: (level: number, milestoneId: string) => number
}

export type CheckupOutcome = { status: "done"; result: CheckupResult } | { status: "abandoned" }

/**
 * Runs every milestone of the suite in order. Milestone m acts in m.makeVariant(seed), with seed variantSeed(level, m.id)
 * (default checkupSeed(learnerId, level, m.id)), through a frozen run seeded hashSeed(that seed, "policy"), for m.budgetSteps steps. A milestone that throws (not
 * recoverable), or whose run's dispose() throws, is not passed, its measure is null and the result has `error: true`;
 * the rest still run. A recoverable error is rethrown once the open run is disposed. A halt (shouldStop) abandons the
 * checkup: no result. `evalCU` is the learner's stats().evalCU delta over the whole checkup.
 */
export async function runCheckup(d: CheckupDeps): Promise<CheckupOutcome> {
  const suite = d.suite ?? MILESTONES_V1
  const evalCU0 = d.learner.stats().evalCU
  const passed: string[] = []
  const measures: Record<string, number | null> = {}
  let error = false
  const errorReasons: Record<string, string> = {}
  for (const m of suite) {
    if (d.shouldStop?.()) return { status: "abandoned" }
    const seed = d.variantSeed ? d.variantSeed(d.level, m.id) : checkupSeed(d.learnerId, d.level, m.id)
    let run: ReturnType<typeof d.learner.frozenRun> | null = null
    let outcome: { value: number; pass: boolean } | null = null // null: the milestone threw
    let thrown: { e: unknown } | null = null // what a non-recoverable milestone threw
    let rethrow: { e: unknown } | null = null
    try {
      const world = m.makeVariant(seed, d.canvas())
      const reference = m.reference?.(seed, d.canvas())
      const r = (run = d.learner.frozenRun({ seed: hashSeed(seed, "policy"), measureLoss: m.measuresLoss }))
      const trace = await recordRun(world, o => r.step(o), m.budgetSteps, d.yieldNow)
      const stats = r.result()
      outcome = { value: m.measure(trace, stats, reference), pass: m.passes(trace, stats, reference) }
    } catch (e) {
      if (d.recoverable?.(e)) rethrow = { e }
      else thrown = { e }
    }
    // A dispose that throws counts as the milestone throwing; it never masks a recoverable error in flight nor stops
    // the remaining milestones.
    try {
      run?.dispose()
    } catch (e) {
      outcome = null
      thrown = { e }
    }
    if (rethrow) throw rethrow.e
    if (outcome) {
      measures[m.id] = Number.isFinite(outcome.value) ? outcome.value : null
      if (outcome.pass) passed.push(m.id)
    } else {
      measures[m.id] = null
      error = true
      const e = thrown?.e
      console.warn("blanktab: milestone", m.id, "threw", e)
      errorReasons[m.id] = errorReason(e)
    }
  }
  if (d.shouldStop?.()) return { status: "abandoned" }
  const result: CheckupResult = { level: d.level, steps: d.steps, passed, measures, evalCU: d.learner.stats().evalCU - evalCU0 }
  if (error) { result.error = true; result.errorReasons = errorReasons }
  return { status: "done", result }
}
