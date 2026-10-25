// confirm-fisher@1 (spec §2.5): the confirmation test. A candidate genome is promoted only if (1) its lives sustain a
// milestone more often than newborns of the same genome do by luck, and (2) it beats the centre on the search score's
// rank average. Changing anything here means a new version (decision 0020).
import { lifeBudget, sameBudget, type LogFileV2 } from "@blanktab/harness/log"
import { budgetWindow, isGpuWidths, isV2, searchOf } from "./life.ts"
import { fisherOneSided, holm, mannWhitneyOneSided } from "./stats.ts"
import type { CohortContext, EvalOutput, Evaluator, LifeInput, LifeStanding, MilestoneKey, References } from "./types.ts"

const ID = "confirm-fisher"
const VERSION = 1
/** The budget and levels whose checkups decide a sustained pass (config/evaluation.json must agree). */
const LEVELS: readonly number[] = Object.freeze([512, 1024, 2048])
const BUDGET = Object.freeze({ budgetCU: 2048, levels: LEVELS })
/** A sustained pass of m: passing m at this many of the LEVELS or more (V2 §6.3). */
const SUSTAINED = 2
/** Test 1's family: M2–M6 (M1 is prediction; V2 §6.3 leaves it out). */
const FAMILY: readonly MilestoneKey[] = ["M2", "M3", "M4", "M5", "M6"]
/** Both tests' α (test 1 family-wise by Holm); a p-value at most α rejects. */
const ALPHA = 0.05
/** Abandonment rates differing between arms by more than this factor flag the result (spec §2.5). */
const ABANDON_RATIO = 2
/** The milestone suite this version reads. */
const SUITE = 1
const ARMS = ["candidate", "centre", "newborn"] as const
type Arm = typeof ARMS[number]

/** How a life ends up in its arm's denominator. */
type Status = "complete" | "failed" | "abandoned" | "pending" | "ignored"
interface Classified { status: Status; sustained: Set<MilestoneKey>; genomeSha: string | null; confirmation: number | null }

/** An arm's denominators: counted = complete + failed; abandoned and pending lives are excluded, ignored ones are not its arm's. */
interface ArmCounts { counted: number; complete: number; failed: number; abandoned: number; pending: number; ignored: number; duplicates: number }

const IGNORED: Classified = { status: "ignored", sustained: new Set(), genomeSha: null, confirmation: null }

/**
 * A life of `arm`, read up to its budget checkup:
 * - **ignored** unless every file has a meta block of role confirm, this arm, genome schema 1, inheritance none and
 *   this version's budget, on GPU widths, with suite-1 checkups;
 * - **complete** if it holds its budget checkup: it sustains m if it passed m at ≥ SUSTAINED of the levels;
 * - **failed** if its last file is final with end "failed" (a non-pass for every m);
 * - **abandoned** if its last file is otherwise final (excluded, and counted);
 * - **pending** if it may still be running (excluded, and counted).
 */
function classify(life: LifeInput, arm: Arm): Classified {
  const window = budgetWindow(life.files, BUDGET.budgetCU)
  if (window.length === 0 || !window.every(isV2)) return IGNORED
  const files = window as LogFileV2[]
  const ok = files.every(f => f.learnerId === life.learnerId && f.meta && f.meta.role === "confirm" && f.meta.confirmation?.arm === arm
    && f.meta.genomeSchema === 1 && f.meta.inheritance.filters === "none" && sameBudget(lifeBudget(f.meta), BUDGET) && isGpuWidths(f.widths)
    && ((f.checkups ?? []).length === 0 || f.milestoneSuiteVersion === SUITE))
  if (!ok) return IGNORED
  const m0 = files[0]!.meta!
  const shas = new Set(files.map(f => f.meta!.genomeSha)), ids = new Set(files.map(f => f.meta!.confirmation!.id))
  if (shas.size !== 1 || ids.size !== 1) return IGNORED
  const base = { genomeSha: m0.genomeSha, confirmation: m0.confirmation!.id }
  const passed = new Map<number, readonly string[]>()
  for (const f of files) for (const c of f.checkups ?? []) if (LEVELS.includes(c.level) && !passed.has(c.level)) passed.set(c.level, c.passed)
  if (passed.has(BUDGET.budgetCU)) {
    const sustained = new Set<MilestoneKey>()
    for (const m of FAMILY) {
      let n = 0
      for (const p of passed.values()) if (p.includes(m)) n++
      if (n >= SUSTAINED) sustained.add(m)
    }
    return { status: "complete", sustained, ...base }
  }
  const last = files[files.length - 1]!
  const status: Status = !last.final ? "pending" : last.meta!.end === "failed" ? "failed" : "abandoned"
  return { status, sustained: new Set(), ...base }
}

/** In an arm's denominator: complete, or failed (a non-pass); abandoned and pending lives are excluded from it. */
const isCounted = (s: Status) => s === "complete" || s === "failed"

/** Evaluator.standing: classify's verdict on `life` as a life of arm `group` (any other group: excluded). */
function standing(life: LifeInput, group: string): LifeStanding {
  if (!(ARMS as readonly string[]).includes(group)) return "excluded"
  const s = classify(life, group as Arm).status
  return isCounted(s) ? "counted" : s === "pending" ? "pending" : "excluded"
}

const byId = (a: LifeInput, b: LifeInput) => (a.learnerId < b.learnerId ? -1 : a.learnerId > b.learnerId ? 1 : 0)

/**
 * `groups.candidate`, `groups.centre` and `groups.newborn` (newborns of the candidate genome) of one confirmation.
 * Test 1: per milestone M2–M6, candidate sustained passes against newborn ones, one-sided Fisher exact, Holm over the
 * five at α 0.05; it passes if any survives. Test 2: the pinned search evaluator scores the candidate and centre lives
 * (as confirmation lives), its rankAverage over the pooled eligible ones gives a_i, and a one-sided Mann–Whitney U of
 * candidate over centre at α 0.05. The score is 1 if both pass, else 0.
 */
function evaluateCohort(groups: Record<string, LifeInput[]>, refs: References, ctx?: CohortContext): EvalOutput {
  const search = searchOf(ctx, `${ID}@${VERSION}`)
  const searchId = `${search.id}@${search.version}`
  if (!search.budget || !sameBudget(search.budget, BUDGET)) {
    throw new Error(`${ID}@${VERSION} reads a budget of ${BUDGET.budgetCU} CU; the search evaluator ${searchId} reads another`)
  }
  if (!(search.scoresRoles ?? ["search"]).includes("confirm")) {
    throw new Error(`${ID}@${VERSION} scores confirmation lives; the search evaluator ${searchId} does not declare scoresRoles "confirm"`)
  }
  const seen = new Set<string>()
  const arms = {} as Record<Arm, { lives: { life: LifeInput; c: Classified }[]; duplicates: number }>
  for (const arm of ARMS) {
    arms[arm] = { lives: [], duplicates: 0 }
    for (const life of [...(groups[arm] ?? [])].sort(byId)) {
      if (seen.has(life.learnerId)) { arms[arm].duplicates++; continue }
      seen.add(life.learnerId)
      arms[arm].lives.push({ life, c: classify(life, arm) })
    }
  }
  const count = (arm: Arm, s: Status) => arms[arm].lives.filter(x => x.c.status === s).length
  const counted = (arm: Arm) => arms[arm].lives.filter(x => isCounted(x.c.status))
  const counts = {} as Record<Arm, ArmCounts>
  for (const arm of ARMS) {
    counts[arm] = {
      counted: counted(arm).length, complete: count(arm, "complete"), failed: count(arm, "failed"), abandoned: count(arm, "abandoned"),
      pending: count(arm, "pending"), ignored: count(arm, "ignored"), duplicates: arms[arm].duplicates,
    }
  }
  const learnerIds = Object.fromEntries(ARMS.map(arm => [arm, counted(arm).map(x => x.life.learnerId)]))
  const rates = Object.fromEntries(ARMS.map(arm => {
    const ab = counts[arm].abandoned, all = ab + counts[arm].counted
    return [arm, all > 0 ? ab / all : null]
  })) as Record<Arm, number | null>
  const known = ARMS.flatMap(arm => (rates[arm] === null ? [] : [rates[arm]!]))
  const lo = known.length ? known.reduce((a, b) => (b < a ? b : a)) : 0, hi = known.length ? known.reduce((a, b) => (b > a ? b : a)) : 0
  const abandonmentFlag = hi > ABANDON_RATIO * lo
  const base = { search: searchId, counts, abandonmentRates: rates, abandonmentFlag, learnerIds }
  const no = (reason: string, extra: Record<string, unknown> = {}): EvalOutput => ({
    evaluator: ID, version: VERSION, eligible: false, reason, score: null,
    components: { behaviour: null, better: null, mannWhitneyP: null }, diagnostics: { ...base, ...extra },
  })
  const empty = ARMS.filter(arm => counted(arm).length === 0)
  if (empty.length) return no(`no counted lives in ${empty.join(", ")}`)
  // One confirmation; newborns of the candidate genome; centre of another (spec §2.5).
  const all = ARMS.flatMap(arm => counted(arm).map(x => x.c))
  const genome = (arm: Arm) => [...new Set(counted(arm).map(x => x.c.genomeSha))]
  const genomes = { candidate: genome("candidate"), centre: genome("centre"), newborn: genome("newborn") }
  if (new Set(all.map(c => c.confirmation)).size !== 1) return no("lives of more than one confirmation", { genomes })
  if (genomes.candidate.length !== 1 || genomes.centre.length !== 1 || genomes.newborn.length !== 1) return no("an arm mixes genomes", { genomes })
  if (genomes.newborn[0] !== genomes.candidate[0]) return no("newborns are not of the candidate genome", { genomes })
  if (genomes.centre[0] === genomes.candidate[0]) return no("the candidate is the centre genome", { genomes })

  // Test 1: behaviour against newborn luck.
  const cand = counted("candidate"), born = counted("newborn")
  const tables = FAMILY.map(m => {
    const a = cand.filter(x => x.c.sustained.has(m)).length, c = born.filter(x => x.c.sustained.has(m)).length
    return { milestone: m, a, b: cand.length - a, c, d: born.length - c, p: fisherOneSided(a, cand.length - a, c, born.length - c) }
  })
  const rejected = holm(tables.map(t => t.p), ALPHA)
  const fisher = tables.map((t, i) => ({ ...t, rejected: rejected[i]! }))
  const behaviour = rejected.some(Boolean)

  // Test 2: better than the centre, on the pinned search evaluator's rank average.
  const scored = (["candidate", "centre"] as const).flatMap(arm =>
    counted(arm).map(x => ({ arm, learnerId: x.life.learnerId, out: search.evaluateLife(x.life, refs, { role: "confirm" }) })))
  const rows = scored.filter(r => r.out.eligible)
  const a = search.rankAverage(rows.map(r => r.out))
  const x = rows.flatMap((r, i) => (r.arm === "candidate" ? [a[i]!] : []))
  const y = rows.flatMap((r, i) => (r.arm === "centre" ? [a[i]!] : []))
  const unscored = Object.fromEntries(scored.filter(r => !r.out.eligible).map(r => [r.learnerId, r.out.reason ?? null]))
  const unscoredCount = Object.keys(unscored).length
  const rankRows = { candidate: x.length, centre: y.length }
  // Test 2 cannot run without rows on both sides: that is no result, never a failed confirmation.
  const noRows = (["candidate", "centre"] as const).filter(arm => rankRows[arm] === 0)
  if (noRows.length) return no(`no scored lives in ${noRows.join(", ")}`, { genomes, fisher, rankRows, unscored, unscoredCount })
  const mannWhitneyP = mannWhitneyOneSided(x, y)
  const better = mannWhitneyP <= ALPHA

  return {
    evaluator: ID, version: VERSION, eligible: true, score: behaviour && better ? 1 : 0,
    components: { behaviour: behaviour ? 1 : 0, better: better ? 1 : 0, mannWhitneyP },
    diagnostics: { ...base, genomes, alpha: ALPHA, fisher, rankRows, unscored, unscoredCount, mannWhitneyP },
  }
}

export const confirmFisher: Evaluator = Object.freeze({
  // needs is empty: confirm-fisher reads no reference itself. It passes refs through to the pinned search evaluator,
  // whose own needs (pinned too) make config load them; if they are absent, its lives are unscored and test 2 has no rows.
  id: ID, version: VERSION, kind: "cohort" as const, needs: [], budget: BUDGET, evaluateCohort, standing,
})
