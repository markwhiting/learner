import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { parseLogFile, type LogFile } from "@blanktab/harness/log"
import {
  REGISTRY, cohortContext, confirmFisher, getEvaluator, parseNewbornReference, searchComposite,
  type Evaluator, type LifeInput, type NewbornReference,
} from "@blanktab/evaluators"
import { fisherOneSided } from "../src/stats"
import { checkup, confirmMeta, CPU, endedLife, file, lid, scoredLife, type Mid } from "./fixtures/build"

const REF = parseNewbornReference(JSON.parse(readFileSync(new URL("./fixtures/reference.json", import.meta.url), "utf8"))) as NewbornReference
const refs = { newborn: REF }
const ctx = cohortContext({ search: "search-composite@1" })
const CANDIDATE = "c1".repeat(32), CENTRE = "0e".repeat(32)
type Arm = "candidate" | "centre" | "newborn"

let next = 0
/** A fresh learnerId and confirmation seed. */
const fresh = () => next++

/** A complete confirm life of `arm`: M3 passed at `m3Levels` of the three levels; `quality` lifts M2 and the held-out score. */
function complete(arm: Exclude<Arm, "newborn">, sha: string, m3Levels: number, quality: number): LifeInput {
  const k = fresh(), learnerId = lid(0x5000 + k)
  const passed = [0, 1, 2].map(i => (i < m3Levels ? ["M3"] : [])) as [Mid[], Mid[], Mid[]]
  const files = scoredLife(learnerId, {
    ms: { M1: 0.9, M2: 0.5 + quality, M3: m3Levels ? 2 : 0.2, M4: 0.3, M5: 0.8, M6: 0 },
    ratio: 0.98 - 0.1 * quality, passed, meta: confirmMeta(arm, k, { genomeSha: sha }),
  })
  return { learnerId, files }
}
/** A confirm life of `arm` that ended at 1300 CU before its budget checkup. */
function ended(arm: Exclude<Arm, "newborn">, sha: string, end: "failed" | "orphan"): LifeInput {
  const k = fresh(), learnerId = lid(0x5000 + k)
  return { learnerId, files: endedLife(learnerId, end, confirmMeta(arm, k, { genomeSha: sha })) }
}
/**
 * A confirmation newborn: one final file at cu 0 with its three frozen checkups (as a reference newborn's); M3 passed
 * at `m3Levels` of them. The frozen confirmation-newborn file (decision 0024; plan 05b Task 7).
 */
function newborn(sha: string, m3Levels: number): LifeInput {
  const k = fresh(), learnerId = lid(0x5000 + k)
  const cks = [512, 1024, 2048].map((level, i) => ({ ...checkup(level, { M3: i < m3Levels ? 2 : 0 }, i < m3Levels ? ["M3"] : []), steps: 0 }))
  return { learnerId, files: [file({ learnerId, seq: 0, cu: 0, final: true, checkups: cks, meta: confirmMeta("newborn", k, { genomeSha: sha }) })] }
}
const times = <T>(n: number, f: (i: number) => T): T[] => Array.from({ length: n }, (_, i) => f(i))
const parses = (files: LogFile[]) => files.every(f => parseLogFile(JSON.parse(JSON.stringify(f))).ok)

/**
 * The plan's design point. Candidate: 20 lives sustaining M3 (passed at all three levels), 2 passing it at one level
 * only, 2 failed before the budget (non-passes), and `candidateOrphans` abandoned. Centre: 24 complete lives of lower
 * quality, and 1 abandoned. Newborns of the candidate genome: 6 of 60 sustain M3 (2 of 3 levels), 6 pass at one level.
 */
function design(candidateOrphans = 4, centreQuality = (i: number) => 0.01 * i, centreM3Levels = 0) {
  return {
    candidate: [
      ...times(20, i => complete("candidate", CANDIDATE, 3, 1 + 0.01 * i)),
      ...times(2, i => complete("candidate", CANDIDATE, 1, 1.5 + 0.01 * i)),
      ...times(2, () => ended("candidate", CANDIDATE, "failed")),
      ...times(candidateOrphans, () => ended("candidate", CANDIDATE, "orphan")),
    ],
    centre: [...times(24, i => complete("centre", CENTRE, centreM3Levels, centreQuality(i))), ended("centre", CENTRE, "orphan")],
    newborn: [...times(6, () => newborn(CANDIDATE, 2)), ...times(6, () => newborn(CANDIDATE, 1)), ...times(48, () => newborn(CANDIDATE, 0))],
  }
}
const run = (groups: Record<string, LifeInput[]>, c = ctx) => confirmFisher.evaluateCohort!(groups, refs, c)
interface Diag {
  counts: Record<Arm, Record<string, number>>; abandonmentRates: Record<Arm, number | null>; abandonmentFlag: boolean
  fisher: { milestone: string; a: number; b: number; c: number; d: number; p: number; rejected: boolean }[]
  rankRows: { candidate: number; centre: number }; mannWhitneyP: number; search: string; unscored: Record<string, string | null>
}

describe("confirm-fisher@1 (spec §2.5)", () => {
  it("is registered as a cohort evaluator with its own budget, and needs the pinned search evaluator in its context", () => {
    expect(REGISTRY["confirm-fisher@1"]).toBe(confirmFisher)
    expect(getEvaluator("confirm-fisher@1")).toBe(confirmFisher)
    expect(confirmFisher).toMatchObject({ id: "confirm-fisher", version: 1, kind: "cohort", needs: [] })
    expect(confirmFisher.budget).toEqual({ budgetCU: 2048, levels: [512, 1024, 2048] })
    expect(() => confirmFisher.evaluateCohort!(design(), refs)).toThrow(/pinned search evaluator/)
    const otherBudget: Evaluator = { ...searchComposite, budget: { budgetCU: 4096, levels: [1024, 2048, 4096] } }
    expect(() => run(design(), { search: otherBudget })).toThrow(/budget/)
    // A search evaluator that does not declare it can score confirmation lives is refused (absent reads as ["search"]).
    expect(searchComposite.scoresRoles).toEqual(["search", "confirm"])
    const noRoles: Evaluator = { ...searchComposite, scoresRoles: undefined }
    expect(() => run(design(), { search: noRoles })).toThrow(/scoresRoles "confirm"/)
    expect(() => run(design(), { search: { ...searchComposite, scoresRoles: ["search"] } })).toThrow(/scoresRoles "confirm"/)
  })

  it("a candidate that sustains M3 in 20/24 against 6/60 newborns and beats centre on a → 1; failed lives count as non-passes; abandoned excluded and counted; an abandonment ratio > 2× is flagged", () => {
    const groups = design()
    // Every arm's fixtures are logs the parser accepts, the confirmation newborns' frozen files (cu 0, frozen checkups;
    // decision 0024) included.
    for (const arm of ["candidate", "centre", "newborn"] as const) expect(groups[arm].every(l => parses(l.files)), arm).toBe(true)
    const out = run(groups)
    expect(out).toMatchObject({ evaluator: "confirm-fisher", version: 1, eligible: true, score: 1, components: { behaviour: 1, better: 1 } })
    const d = out.diagnostics as unknown as Diag
    expect(d.search).toBe("search-composite@1")
    // Failed lives are in the denominator as non-passes; abandoned ones are not, and are counted.
    expect(d.counts.candidate).toMatchObject({ counted: 24, complete: 22, failed: 2, abandoned: 4, pending: 0, ignored: 0 })
    expect(d.counts.centre).toMatchObject({ counted: 24, complete: 24, failed: 0, abandoned: 1 })
    expect(d.counts.newborn).toMatchObject({ counted: 60, complete: 60, abandoned: 0 })
    const m3 = d.fisher.find(t => t.milestone === "M3")!
    expect(m3).toMatchObject({ a: 20, b: 4, c: 6, d: 54, rejected: true })
    expect(m3.p).toBe(fisherOneSided(20, 4, 6, 54))
    expect(m3.p).toBeLessThan(1e-9)
    for (const t of d.fisher.filter(x => x.milestone !== "M3")) expect(t).toMatchObject({ a: 0, c: 0, p: 1, rejected: false })
    expect(d.fisher.map(t => t.milestone)).toEqual(["M2", "M3", "M4", "M5", "M6"])
    // Test 2 ranks the counted candidate and centre lives (failed ones as rows, ranked worst) through the search evaluator.
    expect(d.rankRows).toEqual({ candidate: 24, centre: 24 })
    expect(d.unscored).toEqual({})
    expect((d as unknown as { unscoredCount: number }).unscoredCount).toBe(0)
    expect(d.mannWhitneyP).toBeLessThan(0.001)
    // Abandonment: candidate 4/28, centre 1/25, newborn 0/60: more than 2× apart, so flagged (the score stands).
    expect(d.abandonmentRates).toEqual({ candidate: 4 / 28, centre: 1 / 25, newborn: 0 })
    expect(d.abandonmentFlag).toBe(true)
    // With one orphan in each search arm and two among newborns (rates 1/25, 1/25, 2/62), it is not flagged.
    const even = design(1)
    const unfinished = (final: boolean): LifeInput => {
      const k = fresh(), learnerId = lid(0x5000 + k)
      const m = confirmMeta("newborn", k, { genomeSha: CANDIDATE })
      return { learnerId, files: [file({ learnerId, seq: 0, cu: 0, final, meta: final ? { ...m, end: "orphan" } : m })] }
    }
    even.newborn.push(unfinished(false), unfinished(false)) // no budget checkup, not final: still running, so pending
    const evenD = run(even).diagnostics as unknown as Diag
    expect(evenD.counts.newborn).toMatchObject({ pending: 2, abandoned: 0, counted: 60 })
    even.newborn.push(unfinished(true), unfinished(true)) // final without it: abandoned
    const evenOut = run(even)
    expect((evenOut.diagnostics as unknown as Diag).abandonmentRates).toEqual({ candidate: 1 / 25, centre: 1 / 25, newborn: 2 / 62 })
    expect((evenOut.diagnostics as unknown as Diag).abandonmentFlag).toBe(false)
    expect(evenOut.score).toBe(1)
    // Pure: input order does not matter, and the output bytes repeat.
    const shuffled = { candidate: [...groups.candidate].reverse(), centre: [...groups.centre].reverse(), newborn: [...groups.newborn].reverse() }
    expect(JSON.stringify(run(shuffled))).toBe(JSON.stringify(out))
  })

  it("fails test 2 when the candidate does not beat the centre, and test 1 when newborns pass as often", () => {
    // The centre as good as the candidate on a (the same measures): behaviour passes, better does not → 0.
    const tie = run(design(4, i => 1 + 0.01 * i, 3))
    expect(tie).toMatchObject({ eligible: true, score: 0, components: { behaviour: 1, better: 0 } })
    // Newborns sustaining M3 as often as the candidate: better passes, behaviour does not → 0.
    const lucky = design()
    lucky.newborn = [...times(50, () => newborn(CANDIDATE, 3)), ...times(10, () => newborn(CANDIDATE, 0))]
    expect(run(lucky)).toMatchObject({ eligible: true, score: 0, components: { behaviour: 0, better: 1 } })
  })

  it("the search evaluator scores a confirmation life only when asked to (role confirm), by the same rule", () => {
    const life = complete("candidate", CANDIDATE, 3, 1)
    expect(searchComposite.evaluateLife!(life, refs)).toMatchObject({ eligible: false, reason: "not a search life" })
    const asConfirm = searchComposite.evaluateLife!(life, refs, { role: "confirm" })
    expect(asConfirm.eligible).toBe(true)
    // The same files as a search life score identically.
    const searchFiles = life.files.map(f => ({ ...f, meta: { ...(f as { meta: object }).meta, role: "search", method: "calibrate", evalPanel: 3, evalSeed: null, confirmation: null } })) as LogFile[]
    expect(searchComposite.evaluateLife!({ learnerId: life.learnerId, files: searchFiles }, refs)).toEqual(asConfirm)
    const searchLife = { learnerId: life.learnerId, files: searchFiles }
    expect(searchComposite.evaluateLife!(searchLife, refs, { role: "confirm" })).toMatchObject({ eligible: false, reason: "not a confirm life" })
  })

  it("test 2 without scored rows is no result, never a failed confirmation; unscored lives are reported", () => {
    // No newborn reference: the search evaluator scores no life, so test 2 cannot run.
    const out = confirmFisher.evaluateCohort!(design(), {}, ctx)
    expect(out).toMatchObject({ eligible: false, reason: "no scored lives in candidate, centre", score: null })
    const d = out.diagnostics as { unscored: Record<string, string>; unscoredCount: number; rankRows: object }
    expect(d.unscoredCount).toBe(48)
    expect(new Set(Object.values(d.unscored))).toEqual(new Set(["no newborn reference"]))
    expect(d.rankRows).toEqual({ candidate: 0, centre: 0 })
    // A counted candidate life the search evaluator cannot score (too few held-out points) is reported, and the rest rank.
    const g = design()
    const thin = complete("candidate", CANDIDATE, 3, 1)
    thin.files = thin.files.map(f => (f.seq === 2 ? { ...f, curve: [] } : f))
    g.candidate.push(thin)
    const partial = run(g).diagnostics as { unscored: Record<string, string>; unscoredCount: number; rankRows: object }
    expect(partial.unscored).toEqual({ [thin.learnerId]: "fewer than 3 held-out points in (1024, 2048]" })
    expect(partial.unscoredCount).toBe(1)
    expect(partial.rankRows).toEqual({ candidate: 24, centre: 24 })
  })

  it("obtains a_i through the pinned search evaluator it is given", () => {
    // A stand-in search evaluator whose rank average puts every row level: test 2 cannot pass through it.
    const flat: Evaluator = { ...searchComposite, id: "flat-search", rankAverage: outs => outs.filter(o => o.eligible).map(() => 0.5) }
    const out = run(design(), { search: flat })
    expect(out).toMatchObject({ score: 0, components: { behaviour: 1, better: 0, mannWhitneyP: 1 }, diagnostics: { search: "flat-search@1" } })
  })

  it("refuses arms that are not one confirmation's: wrong genomes, wrong arms, empty arms", () => {
    const g = design()
    expect(run({ ...g, newborn: times(60, () => newborn(CENTRE, 0)) })).toMatchObject({ eligible: false, reason: "newborns are not of the candidate genome" })
    expect(run({ ...g, centre: times(24, i => complete("centre", CANDIDATE, 0, 0.01 * i)) })).toMatchObject({ eligible: false, reason: "the candidate is the centre genome" })
    expect(run({ ...g, centre: [...g.centre, complete("centre", "77".repeat(32), 0, 0)] })).toMatchObject({ eligible: false, reason: "an arm mixes genomes" })
    // Lives filed under the wrong arm are ignored (and counted); an arm left with none is ineligible.
    const misfiled = run({ ...g, centre: times(3, i => complete("candidate", CENTRE, 0, 0.01 * i)) })
    expect(misfiled).toMatchObject({ eligible: false, reason: "no counted lives in centre" })
    expect((misfiled.diagnostics as unknown as Diag).counts.centre).toMatchObject({ ignored: 3, counted: 0 })
    expect(run({ candidate: g.candidate, centre: g.centre })).toMatchObject({ eligible: false, reason: "no counted lives in newborn", score: null })
    // A life listed in two arms counts once.
    const dup = run({ ...g, centre: [...g.centre, g.candidate[0]!] })
    expect((dup.diagnostics as unknown as Diag).counts.centre).toMatchObject({ duplicates: 1, counted: 24 })
  })

  it("standing() is the classification evaluateCohort counts by, on every fixture life (the gate decides full arms from it)", () => {
    const g = design()
    const k = fresh(), id = lid(0x5000 + k)
    // Lives the gate's own classifier once counted but this evaluator ignores or excludes: CPU widths, a logged arm
    // other than the group, and one still running (not final) before its budget checkup.
    const cpu: LifeInput = { learnerId: id, files: scoredLife(id, { ms: { M2: 1 }, ratio: 1, meta: confirmMeta("candidate", k, { genomeSha: CANDIDATE }) }).map(f => ({ ...f, widths: CPU })) }
    const k2 = fresh(), id2 = lid(0x5000 + k2)
    const running: LifeInput = { learnerId: id2, files: endedLife(id2, "orphan", confirmMeta("candidate", k2, { genomeSha: CANDIDATE })).slice(0, 1) }
    const lives: [Arm, LifeInput][] = [
      ...(["candidate", "centre", "newborn"] as const).flatMap(arm => g[arm].map(l => [arm, l] as [Arm, LifeInput])),
      ["candidate", cpu], ["candidate", running], ["centre", g.candidate[0]!], ["newborn", g.centre[0]!],
    ]
    const seen = new Set<string>()
    for (const [arm, life] of lives) {
      const s = confirmFisher.standing!(life, arm)
      const counts = (run({ [arm]: [life] }).diagnostics as unknown as Diag).counts[arm]!
      seen.add(s)
      expect({ learnerId: life.learnerId, arm, s }).toEqual({
        learnerId: life.learnerId, arm, s: counts.counted === 1 ? "counted" : counts.pending === 1 ? "pending" : "excluded",
      })
    }
    expect([...seen].sort()).toEqual(["counted", "excluded", "pending"])
    expect(confirmFisher.standing!(cpu, "candidate")).toBe("excluded")
    expect(confirmFisher.standing!(g.candidate[0]!, "elsewhere")).toBe("excluded")
  })
})
