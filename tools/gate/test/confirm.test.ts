import { describe, it, expect } from "vitest"
import { readFileSync, rmSync } from "node:fs"
import { join } from "node:path"
import { NEWBORN_LEVELS, type LogFileV2 } from "@blanktab/harness/log"
import { birthDraw, parseCentre, parsePosterior, toUnit, type ArmName, type Genome, type Posterior } from "@blanktab/meta"
import { checkup, confirmMeta, endedLife, lid, scoredLife } from "../../../packages/evaluators/test/fixtures/build.ts"
import {
  armGroups, CONFIRM_TARGET, confirmationPath, DEADLINE_DAYS, deadlineOf, FORWARD_SKEW_MINUTES, IN_FLIGHT_HOURS, parseConfirmation, type ConfirmationFile, type VerifiedConfirmLife,
} from "../confirm.ts"
import { buildMeta, type MetaFiles, type MetaInput } from "../meta.ts"
import { incumbentFor } from "../search.ts"
import { serialise } from "../serialise.ts"
import {
  bornLife, build, BUILT_AT, CENTRE, commit, CONFIG, grow, POSTERIOR, put, REFS, runsDir, shaOf, shaOfGenome, SPACE, useTempRuns, world,
  type Effect, type World,
} from "./meta-fixture.ts"

useTempRuns()

const C0 = confirmationPath(0)
const HOUR = 3_600_000
const at = (ms: number) => new Date(Date.parse(BUILT_AT) + ms).toISOString()
const posteriorOf = (out: MetaFiles): Posterior => {
  const p = parsePosterior(out[POSTERIOR])
  if (!p) throw new Error("posterior does not parse")
  return p
}
const diagOf = (out: MetaFiles) => JSON.parse(out["data/meta/diagnostics.json"]) as { confirmation?: { id: number; counted: Record<ArmName, number>; inFlight: Record<ArmName, number> } }

/** Scores that follow the first searched gene: the GP finds the incumbent far above the centre (pBeatsCentre ≈ 1). */
const byGenome: Effect = (k, _i, g) => {
  const u = toUnit(SPACE, g)[0]!
  return { m2: 0.3 + 3 * u + 0.01 * (k % 3), ratio: 0.98 - 0.08 * u + 0.001 * (k % 3) }
}
/** `n` more rows under main's posterior: a birth that draws a confirmation life is not merged (its files removed). */
async function growRows(w: World, n: number): Promise<void> {
  for (let i = 0; i < n;) {
    const b = await bornLife(w.previous.posteriorText, { effect: byGenome })
    if (b.meta.role === "confirm") rmSync(join(runsDir(), b.learnerId), { recursive: true })
    else i++
  }
}
/** Mode search through data (30 calibration rows with a genome effect), then 100 search-era rows: the build declares confirmation 0. */
async function declaredWorld(): Promise<{ w: World; out: MetaFiles; Q: string }> {
  const w = await world()
  await build(w)
  await grow(w, 30, byGenome)
  const signal = await build(w)
  expect(posteriorOf(signal).mode).toBe("search")
  await growRows(w, 100)
  const out = await build(w)
  expect(Object.keys(out.confirmations)).toEqual([C0])
  return { w, out, Q: out[POSTERIOR] }
}

/** How a confirmation life is laid out: complete (its budget checkup), failed, abandoned (orphan), or running. */
type Shape = "complete" | "failed" | "abandoned" | "running"
let nextK = 0x60000
/**
 * Confirmation lives born under the served posterior Q, as tabs would draw them: births are drawn in learnerId order
 * and kept while their arm wants more, one per evalSeed. `shape(arm, i)` lays out the arm's i-th. `advantage`: the
 * candidate's lives sustain M2 and score above the centre's; otherwise every arm scores alike and passes nothing.
 */
async function confirmLives(Q: string, want: Partial<Record<ArmName, number>>, opts: {
  advantage: boolean; shape?: (arm: ArmName, i: number) => Shape; writtenAt?: string
}): Promise<Record<ArmName, string[]>> {
  const q = parsePosterior(Q)!, qSha = await shaOf(Q), centreSha = await shaOfGenome(CENTRE)
  const made: Record<ArmName, string[]> = { candidate: [], centre: [], newborn: [] }
  const seeds = new Set<number>()
  const left = () => (["candidate", "centre", "newborn"] as const).some(a => made[a].length < (want[a] ?? 0))
  while (left()) {
    const k = nextK++
    const d = birthDraw({ learnerId: lid(0x10000 + k), space: SPACE, centre: CENTRE, centreSha, widthsClass: "gpu", posterior: q, posteriorSha: qSha })
    if (d.method !== "confirm") continue
    const arm = d.confirmation!.arm
    if (made[arm].length >= (want[arm] ?? 0) || seeds.has(d.evalSeed!)) continue
    seeds.add(d.evalSeed!)
    const shape = opts.shape?.(arm, made[arm].length) ?? "complete"
    const strong = opts.advantage && arm === "candidate"
    const b = await bornLife(Q, {
      k, effect: () => (strong ? { m2: 3, ratio: 0.9 } : { m2: 0.3, ratio: 0.98 }),
      ...(strong ? { passed: [["M2"], ["M2"], ["M2"]] as [["M2"], ["M2"], ["M2"]] } : {}),
    })
    made[arm].push(b.learnerId)
    if (arm === "newborn" && shape === "complete") {
      // A confirmation newborn, as a tab writes it: one frozen file (decision 0024 "Confirmation-newborn files").
      rmSync(join(runsDir(), b.learnerId), { recursive: true })
      const one = scoredLife(b.learnerId, { ms: {}, ratio: 1, meta: b.meta })[0]!
      put({
        ...one, seq: 0, cu: 0, stepsTotal: 0, final: true, curve: [],
        checkups: NEWBORN_LEVELS.map(level => ({ ...checkup(level, { M2: 1 }), steps: 0 })), milestoneSuiteVersion: 1,
        meta: { ...b.meta, heldOut: { loss: 1, baseline: 1 } },
      })
    } else if (shape === "failed" || shape === "abandoned") {
      rmSync(join(runsDir(), b.learnerId), { recursive: true })
      for (const f of endedLife(b.learnerId, shape === "failed" ? "failed" : "orphan", b.meta)) put(f)
    } else if (shape === "running") {
      rmSync(join(runsDir(), b.learnerId), { recursive: true })
      for (const f of b.files.slice(0, 2)) put({ ...f, writtenAt: opts.writtenAt ?? BUILT_AT })
    }
  }
  return made
}
const FULL = { candidate: 24, centre: 24, newborn: 60 }

describe("confirmations (spec §4.5; plan 05b Task 9)", () => {
  it("a trigger declares confirmation 0: seedBase 10000, targets 24/24/60, deadline +21 days; the posterior serves it with need = target", async () => {
    const { w, out } = await declaredWorld()
    const c = parseConfirmation(out.confirmations[C0]!)
    const p = posteriorOf(out)
    const centreSha = await shaOfGenome(CENTRE)
    expect(c).toMatchObject({
      schemaVersion: 1, id: 0, genomeSchema: SPACE.schemaVersion, confirm: "confirm-fisher@1", search: CONFIG.search,
      seedBase: 10_000, target: { candidate: 24, centre: 24, newborn: 60 }, declaredAt: BUILT_AT, deadline: "2026-11-13T12:00:00.000Z",
      sourceCommit: commit(w.built), centre: { genomeSha: centreSha, values: CENTRE }, result: null,
    })
    expect(DEADLINE_DAYS).toBe(21)
    expect(CONFIRM_TARGET).toEqual(FULL)
    // The candidate is the incumbent's genome; the trigger is recorded with its inputs.
    expect(c.candidate.genomeSha).toBe(p.incumbent!.genomeSha)
    expect(await shaOfGenome(c.candidate.values)).toBe(c.candidate.genomeSha)
    expect(c.trigger).toEqual({
      pBeatsCentre: p.incumbent!.pBeatsCentre, incumbentMean: p.incumbent!.mean, rowsAtDeclaration: 100,
      modelSha: await shaOf(serialise(p.models.gpu)),
    })
    expect(c.trigger.pBeatsCentre).toBeGreaterThanOrEqual(0.95)
    // Served by the same build, before any deploy.
    expect(p.confirmation).toEqual({
      id: 0, seedBase: 10_000, candidate: c.candidate, centre: c.centre,
      target: FULL, counted: { candidate: 0, centre: 0, newborn: 0 }, need: FULL,
    })
    expect(p.page.rowsSinceDeclaration).toBe(0)
    expect(out.promotionLine).toBeNull()
    expect(out["knowledge/genome.json"]).toBeNull()
    // Tabs served it draw confirmation lives of all three arms, with seeds from its block.
    const sha = await shaOf(out[POSTERIOR])
    const arms = new Set<string>()
    for (let k = 0; k < 400; k++) {
      const d = birthDraw({ learnerId: lid(0x90000 + k), space: SPACE, centre: CENTRE, centreSha, widthsClass: "gpu", posterior: p, posteriorSha: sha })
      if (d.method !== "confirm") continue
      arms.add(d.confirmation!.arm)
      expect(d.evalSeed! - 10_000).toBeGreaterThanOrEqual(0)
      expect(d.evalSeed! - 10_000).toBeLessThan(1000)
    }
    expect([...arms].sort()).toEqual(["candidate", "centre", "newborn"])
    // The file reads back, and the next build (nothing new) leaves it alone.
    const again = await build(w)
    expect(again.confirmations).toEqual({})
    expect(posteriorOf(again).confirmation).toEqual(p.confirmation)
  }, 240_000)

  it("armGroups: two lives with one (arm, evalSeed) → the lower learnerId; 30 candidate lives → the first 24 by learnerId", () => {
    const c = { id: 0, target: CONFIRM_TARGET } as unknown as ConfirmationFile
    const life = (k: number, arm: ArmName, seed: number): VerifiedConfirmLife => {
      const m = confirmMeta(arm, seed)
      return { life: { learnerId: lid(k), files: scoredLife(lid(k), { ms: { M2: 1 }, ratio: 1, meta: m }) }, meta: m }
    }
    // A duplicate draw of centre seed 3: the lower learnerId counts, whatever order the lives come in.
    const dup = [life(0x502, "centre", 3), life(0x501, "centre", 3), life(0x503, "centre", 4)]
    expect(armGroups(dup, c).centre.map(l => l.learnerId)).toEqual([lid(0x501), lid(0x503)])
    // 30 candidate lives on distinct seeds: the first 24 by learnerId; the extras are dropped by rule.
    const many = Array.from({ length: 30 }, (_, i) => life(0x700 + 29 - i, "candidate", i))
    const g = armGroups(many, c)
    expect(g.candidate.map(l => l.learnerId)).toEqual(Array.from({ length: 24 }, (_, i) => lid(0x700 + i)))
    expect(g.centre).toEqual([])
    expect(g.newborn).toEqual([])
    // Lives of another confirmation are not this one's.
    const other = { ...life(0x800, "candidate", 50), meta: { ...confirmMeta("candidate", 50), confirmation: { id: 1, arm: "candidate" as const }, evalSeed: 11_200 } }
    expect(armGroups([other], c).candidate).toEqual([])
  })

  it("need subtracts in-flight lives written within 48 h; a life silent for 49 h no longer holds a slot", async () => {
    const { w, Q } = await declaredWorld()
    expect(IN_FLIGHT_HOURS).toBe(48)
    await confirmLives(Q, { candidate: 1 }, { advantage: true })
    const running = await confirmLives(Q, { candidate: 1 }, { advantage: true, shape: () => "running", writtenAt: BUILT_AT })
    const within = await build(w, { builtAt: at(48 * HOUR) })
    expect(within.confirmations).toEqual({})
    expect(posteriorOf(within).confirmation).toMatchObject({ counted: { candidate: 1, centre: 0, newborn: 0 }, need: { candidate: 22, centre: 24, newborn: 60 } })
    expect(diagOf(within).confirmation).toEqual({ id: 0, counted: { candidate: 1, centre: 0, newborn: 0 }, inFlight: { candidate: 1, centre: 0, newborn: 0 } })
    const silent = await build(w, { builtAt: at(49 * HOUR) })
    expect(posteriorOf(silent).confirmation).toMatchObject({ counted: { candidate: 1, centre: 0, newborn: 0 }, need: { candidate: 23, centre: 24, newborn: 60 } })
    // It writes again: it holds its slot again.
    const f1 = JSON.parse(readFileSync(join(runsDir(), running.candidate[0]!, "1.json"), "utf8")) as LogFileV2
    put({ ...f1, writtenAt: at(48 * HOUR) })
    expect(posteriorOf(await build(w, { builtAt: at(49 * HOUR) })).confirmation!.need.candidate).toBe(22)
  }, 240_000)

  it("a running life written more than 10 min after builtAt holds no slot (a fast clock cannot keep one); 10 min ahead still does", async () => {
    const { w, Q } = await declaredWorld()
    expect(FORWARD_SKEW_MINUTES).toBe(10)
    const ahead = await confirmLives(Q, { candidate: 1 }, { advantage: true, shape: () => "running", writtenAt: at(10 * 60_000 + 1) })
    const out = await build(w)
    expect(diagOf(out).confirmation!.inFlight.candidate).toBe(0)
    expect(posteriorOf(out).confirmation!.need.candidate).toBe(24)
    const f1 = JSON.parse(readFileSync(join(runsDir(), ahead.candidate[0]!, "1.json"), "utf8")) as LogFileV2
    put({ ...f1, writtenAt: at(10 * 60_000) })
    const skewed = await build(w)
    expect(diagOf(skewed).confirmation!.inFlight.candidate).toBe(1)
    expect(posteriorOf(skewed).confirmation!.need.candidate).toBe(23)
  }, 240_000)

  it("the centre moved by hand (a gp gene) while confirmation 0 is open → decided at once: incomplete, no genome file, no confirmation served", async () => {
    const { w, Q } = await declaredWorld()
    await confirmLives(Q, FULL, { advantage: true }) // full arms that would pass under the declared centre
    const moved = { ...CENTRE, lr: 1e-3 }
    w.centres.set(await shaOfGenome(moved), moved)
    const out = await build(w, { centre: moved })
    const r = parseConfirmation(out.confirmations[C0]!).result!
    expect(r).toMatchObject({ outcome: "incomplete", decidedAt: BUILT_AT, output: { evaluator: "confirm-fisher", score: 1 } }) // it would have passed
    expect(out["knowledge/genome.json"]).toBeNull()
    expect(out.promotionLine).toBeNull()
    const p = posteriorOf(out)
    expect(p.confirmation).toBeNull()
    expect(p.centre).toEqual({ genomeSha: await shaOfGenome(moved), values: moved })
    expect(p.page.lastPromotion).toBeNull()
  }, 240_000)

  it("the gate refuses tampered confirmation files: a pin or genome-schema change, a genomeSha mismatch, an open file before a later one, an id out of place", async () => {
    const { w } = await declaredWorld()
    const c0 = w.previous.confirmations[0]!
    const tried = (confirmations: ConfirmationFile[]) => build(w, { previous: { ...w.previous, confirmations } })
    await expect(tried([{ ...c0, confirm: "confirm-fisher@2" }])).rejects.toThrow("data/meta/confirmations/0.json: was declared for confirm-fisher@2, but confirm-fisher@1 is pinned")
    await expect(tried([{ ...c0, search: "search-composite@2" }])).rejects.toThrow("data/meta/confirmations/0.json: was declared for search evaluator search-composite@2, but search-composite@1 is pinned")
    await expect(tried([{ ...c0, genomeSchema: 2 }])).rejects.toThrow("data/meta/confirmations/0.json: was declared for genome schema 2, but the space is 1")
    await expect(tried([{ ...c0, candidate: { ...c0.candidate, genomeSha: "9".repeat(64) } }])).rejects.toThrow("data/meta/confirmations/0.json: the candidate's genomeSha does not match its values")
    await expect(tried([{ ...c0, centre: { ...c0.centre, values: { ...c0.centre.values, lr: 1e-3 } } }])).rejects.toThrow("data/meta/confirmations/0.json: the centre's genomeSha does not match its values")
    const c1 = { ...c0, id: 1, seedBase: 11_000 }
    await expect(tried([c0, c1])).rejects.toThrow("data/meta/confirmations/0.json: open while a later confirmation exists")
    await expect(tried([c1])).rejects.toThrow("data/meta/confirmations/0.json: id 1, expected 0 (ids are 0..n-1 with no gaps)")
    // The untampered history still builds.
    expect((await tried([c0])).confirmations).toEqual({})
  }, 240_000)

  it("full arms with a planted candidate advantage → pass: genome.json is the candidate (parseCentre ok), promotions.jsonl has one line, the posterior's centre is the candidate and lastPromotion is set", async () => {
    const { w, out: declared, Q } = await declaredWorld()
    const c = parseConfirmation(declared.confirmations[C0]!)
    const made = await confirmLives(Q, FULL, { advantage: true })
    const out = await build(w)
    const done = parseConfirmation(out.confirmations[C0]!)
    expect(done.result).toMatchObject({ outcome: "pass", decidedAt: BUILT_AT, sourceCommit: commit(w.built), output: { evaluator: "confirm-fisher", version: 1, eligible: true, score: 1 } })
    expect(done.result!.lives).toEqual({ candidate: [...made.candidate].sort(), centre: [...made.centre].sort(), newborn: [...made.newborn].sort() })
    expect({ ...done, result: null }).toEqual(c) // only the result is appended
    // knowledge/genome.json: the candidate, with the confirmation's provenance.
    const genome = JSON.parse(out["knowledge/genome.json"]!) as { values: Genome; provenance: Record<string, unknown> }
    expect(parseCentre(genome, SPACE)).toEqual(c.candidate.values)
    expect(out["knowledge/genome.json"]).toBe(serialise(genome))
    expect(genome.provenance).toMatchObject({
      source: "promotion", confirmation: 0, decision: "docs/decisions/0018-one-meta-learner-over-a-learning-genome.md", promotedAt: BUILT_AT,
      mannWhitneyP: done.result!.output.components.mannWhitneyP, lives: done.result!.lives,
    })
    expect(genome.provenance.fisher).toEqual(done.result!.output.diagnostics.fisher)
    // promotions.jsonl: one line.
    expect(JSON.parse(out.promotionLine!)).toEqual({
      confirmation: 0, genomeSha: c.candidate.genomeSha, previous: await shaOfGenome(CENTRE), promotedAt: BUILT_AT, sourceCommit: commit(w.built),
    })
    expect(out.promotionLine).not.toMatch(/\n/)
    // The posterior: built for the new centre, its incumbent against it, no confirmation, lastPromotion set.
    const p = posteriorOf(out)
    expect(p.centre).toEqual(c.candidate)
    expect(p.confirmation).toBeNull()
    expect(p.page.lastPromotion).toBe(BUILT_AT)
    expect(p.incumbent).toEqual(await incumbentFor(p.models.gpu!, SPACE, c.candidate.values))
    // The next build, on the new genome.json: the result stays, no promotion again, and every row stays a row.
    w.centres.set(c.candidate.genomeSha, c.candidate.values)
    const next = await build(w, { centre: c.candidate.values })
    expect(next.confirmations).toEqual({})
    expect(next.promotionLine).toBeNull()
    expect(next["knowledge/genome.json"]).toBeNull()
    expect(posteriorOf(next).page).toMatchObject({ lastPromotion: BUILT_AT, eligibleLives: p.page.eligibleLives })
  }, 240_000)

  it("full arms, no advantage → fail; nothing else changes", async () => {
    const { w, Q } = await declaredWorld()
    await confirmLives(Q, FULL, { advantage: false })
    const out = await build(w)
    const done = parseConfirmation(out.confirmations[C0]!)
    expect(done.result).toMatchObject({ outcome: "fail", output: { eligible: true, score: 0 } })
    expect(out.promotionLine).toBeNull()
    expect(out["knowledge/genome.json"]).toBeNull()
    const p = posteriorOf(out)
    expect(p.centre).toEqual({ genomeSha: await shaOfGenome(CENTRE), values: CENTRE })
    expect(p.confirmation).toBeNull()
    expect(p.page.lastPromotion).toBeNull()
  }, 240_000)

  it("deadline passed with arms short → incomplete; no promotion; result appended once (a later build changes nothing)", async () => {
    const { w, Q } = await declaredWorld()
    const c = w.previous.confirmations[0]!
    await confirmLives(Q, { candidate: 3, centre: 2, newborn: 5 }, { advantage: true })
    const onTime = await build(w, { builtAt: c.deadline }) // at the deadline: still open
    expect(onTime.confirmations).toEqual({})
    expect(posteriorOf(onTime).confirmation!.counted).toEqual({ candidate: 3, centre: 2, newborn: 5 })
    const late = at(DEADLINE_DAYS * 24 * HOUR + 1)
    const out = await build(w, { builtAt: late })
    const done = parseConfirmation(out.confirmations[C0]!)
    expect(done.result).toMatchObject({ outcome: "incomplete", decidedAt: late, output: { evaluator: "confirm-fisher" } })
    expect(done.result!.lives).toMatchObject({ candidate: expect.any(Array) })
    expect(done.result!.lives.candidate).toHaveLength(3)
    expect(out.promotionLine).toBeNull()
    expect(out["knowledge/genome.json"]).toBeNull()
    expect(posteriorOf(out).confirmation).toBeNull()
    // More lives of it, a later build: the file is never changed again.
    await confirmLives(Q, { candidate: 2 }, { advantage: true })
    const later = await build(w, { builtAt: at(30 * 24 * HOUR) })
    expect(later.confirmations).toEqual({})
    expect(serialise(w.previous.confirmations[0])).toBe(out.confirmations[C0])
  }, 240_000)

  it("a failed candidate life counts as a non-pass; an abandoned one is excluded", async () => {
    const { w, Q } = await declaredWorld()
    const shape = (arm: ArmName, i: number): Shape => (arm !== "candidate" ? "complete" : i === 0 ? "abandoned" : i === 1 ? "failed" : "complete")
    const made = await confirmLives(Q, { ...FULL, candidate: 24 }, { advantage: true, shape })
    // 24 candidate lives: one abandoned, so 23 counted (the failed one among them) and one more wanted.
    const open = await build(w)
    expect(open.confirmations).toEqual({})
    expect(posteriorOf(open).confirmation).toMatchObject({ counted: { candidate: 23, centre: 24, newborn: 60 }, need: { candidate: 1, centre: 0, newborn: 0 } })
    const one = await confirmLives(Q, { candidate: 1 }, { advantage: true })
    const out = await build(w)
    const r = parseConfirmation(out.confirmations[C0]!).result!
    expect(r.outcome).toBe("pass")
    const counts = (r.output.diagnostics.counts as Record<ArmName, Record<string, number>>).candidate
    expect(counts).toMatchObject({ counted: 24, complete: 23, failed: 1, abandoned: 1 })
    expect(r.lives.candidate).toEqual([...made.candidate, ...one.candidate].sort())
    const fisher = r.output.diagnostics.fisher as { milestone: string; a: number; b: number }[]
    expect(fisher.find(t => t.milestone === "M2")).toMatchObject({ a: 23, b: 1 }) // the failed life: a non-pass
  }, 240_000)

  it("no declaration while one is in flight; the next trigger needs 100 new rows after rowsAtDeclaration", async () => {
    const { w } = await declaredWorld()
    expect(w.previous.confirmations[0]!.trigger.rowsAtDeclaration).toBe(100)
    // 99 new rows; confirmation 0 is decided at its deadline; the deciding build and the next declare nothing.
    await growRows(w, 99)
    const late = at(DEADLINE_DAYS * 24 * HOUR + 1)
    const decided = await build(w, { builtAt: late })
    expect(Object.keys(decided.confirmations)).toEqual([C0])
    expect(posteriorOf(decided).confirmation).toBeNull()
    const short = await build(w, { builtAt: late })
    expect(short.confirmations).toEqual({})
    expect(posteriorOf(short).page.rowsSinceDeclaration).toBe(99)
    expect(posteriorOf(short).incumbent!.pBeatsCentre).toBeGreaterThanOrEqual(0.95) // only the row count holds it back
    // The 100th: confirmation 1.
    await growRows(w, 1)
    const next = await build(w, { builtAt: late })
    expect(Object.keys(next.confirmations)).toEqual([confirmationPath(1)])
    const c1 = parseConfirmation(next.confirmations[confirmationPath(1)]!)
    expect(c1).toMatchObject({ id: 1, seedBase: 11_000, declaredAt: late, deadline: deadlineOf(late), trigger: { rowsAtDeclaration: 200 }, result: null })
    expect(posteriorOf(next).confirmation).toMatchObject({ id: 1, seedBase: 11_000, need: FULL })
    // 100 more rows while confirmation 1 is open: the trigger holds in every other respect, and nothing is declared.
    await growRows(w, 100)
    const blocked = await build(w, { builtAt: late })
    const p = posteriorOf(blocked)
    expect(blocked.confirmations).toEqual({})
    expect(p.page.rowsSinceDeclaration).toBe(100)
    expect(p.incumbent!.pBeatsCentre).toBeGreaterThanOrEqual(0.95)
    expect(p.incumbent!.genomeSha).not.toBe(p.centre.genomeSha)
    expect(p.confirmation!.id).toBe(1)
  }, 600_000)

  it("buildMeta twice on the same input (builtAt included) → identical bytes for every file", async () => {
    const { w, Q } = await declaredWorld()
    await confirmLives(Q, FULL, { advantage: true })
    const input: MetaInput = {
      runsDir: runsDir(), config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(90), builtAt: at(5 * HOUR),
      previous: w.previous, posteriorsBySha: w.history, centres: w.centres,
    }
    const a = await buildMeta(input)
    const b = await buildMeta(input)
    expect(a.promotionLine).not.toBeNull() // a pass: every kind of file is written
    expect(Object.keys(a.confirmations)).toEqual([C0])
    expect(b).toEqual(a)
    for (const k of Object.keys(a) as (keyof MetaFiles)[]) expect(serialise(b[k])).toBe(serialise(a[k]))
  }, 240_000)
})
