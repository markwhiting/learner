import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { sobolPoints } from "@blanktab/detmath"
import { NEWBORN_LEVELS, type LogFileV2 } from "@blanktab/harness/log"
import * as metaPkg from "@blanktab/meta"
import {
  birthDraw, decodeGenome, fromUnit, genomeSha, parsePosterior, searched, toUnit, round6,
  type Genome, type Posterior,
} from "@blanktab/meta"
import { SIGNAL_LOOKS } from "@blanktab/evaluators"
import { CPU, checkup, endedLife, lid, scoredLife, typicalLife } from "../../../packages/evaluators/test/fixtures/build.ts"
import { collectLives } from "../../eval/lives.ts"
import { FIXTURE_ENV, makeFixture, type Fixture } from "./git-fixture.ts"
import { serialise } from "../serialise.ts"
import {
  buildMeta, parseCalibration, parseVerified, posteriorHistory, wantedPosteriors, type CalibrationState, type MetaFiles, type MetaInput, type VerifiedCache,
} from "../meta.ts"
import {
  bornLife, build, BUILT_AT, resetK, CALIBRATION, CENTRE, commit, CONFIG, DIAGNOSTICS, grow, noise, planted, POSTERIOR, put, REFS, REPO, runsDir,
  searchWorld, shaOf, shaOfGenome, SPACE, useTempRuns, VERIFIED, world,
} from "./meta-fixture.ts"

// Spies on the GP fit and the draw, passing through: the inertness tests count fitGp calls (none outside search) and
// the cache test counts birthDraw calls (none for a cached life).
vi.mock("@blanktab/meta", async importOriginal => {
  const m = await importOriginal<typeof import("@blanktab/meta")>()
  return { ...m, fitGp: vi.fn(m.fitGp), birthDraw: vi.fn(m.birthDraw) }
})
const fitSpy = vi.mocked(metaPkg.fitGp)
const drawSpy = vi.mocked(metaPkg.birthDraw)

const GOLDEN = fileURLToPath(new URL("./fixtures/meta/", import.meta.url))
useTempRuns()
const posteriorOf = (out: MetaFiles): Posterior => {
  const p = parsePosterior(out[POSTERIOR])
  if (!p) throw new Error("posterior does not parse")
  return p
}
const calibrationOf = (out: MetaFiles): CalibrationState => parseCalibration(out[CALIBRATION])
interface Verify { checked: number; ineligible: { learnerId: string; reason: string }[]; genomeMismatch: { count: number; rate: number; learnerIds: string[] } }
interface Diagnostics {
  schemaVersion: 1; evaluator: string; verify: Verify; unscored: Record<string, number>
  perGenome: { genomeSha: string; born: number; rows: number; failed: number; abandoned: number }[]
  model?: { rows: number; fitRows: number; refit: boolean; startsWithNoStep: number[]; reason: string | null }
}
const diagOf = (out: MetaFiles) => JSON.parse(out[DIAGNOSTICS]) as Diagnostics
const verifyOf = (out: MetaFiles) => diagOf(out).verify

const regenerate = process.env.META_GOLDEN_WRITE === "1"
function golden(name: string, text: string) {
  const path = join(GOLDEN, name)
  if (regenerate) { mkdirSync(GOLDEN, { recursive: true }); writeFileSync(path, text) }
  expect(existsSync(path), `${name} is missing (META_GOLDEN_WRITE=1 writes it)`).toBe(true)
  expect(text).toBe(readFileSync(path, "utf8"))
}

/** The next float64 above `x` (> 0). */
function nextUp(x: number): number {
  const v = new DataView(new ArrayBuffer(8))
  v.setFloat64(0, x)
  v.setBigUint64(0, v.getBigUint64(0) + 1n)
  return v.getFloat64(0)
}

describe("buildMeta (spec §4.6, §7.2, §7.4)", () => {
  it("first build: mode calibrate, 10 genomes (centre first), need 3 each, no looks; the bytes are 05a's plus schemaVersion 2 and rowsSinceDeclaration 0", async () => {
    const w = await world()
    const out = await build(w)
    const p = posteriorOf(out)
    const centreSha = await genomeSha(1, CENTRE)
    expect(p).toMatchObject({ schemaVersion: 2, genomeSchema: 1, evaluator: CONFIG.search, sourceCommit: commit(1), mode: "calibrate" })
    expect(p.centre).toEqual({ genomeSha: centreSha, values: CENTRE })
    expect(p.calibration.genomes).toHaveLength(10)
    const design: Genome[] = [CENTRE, ...sobolPoints(searched(SPACE).length, 9, "calib/1").map(u => fromUnit(SPACE, u, CENTRE))]
    expect(p.calibration.genomes.map(g => g.values)).toEqual(design)
    expect(p.calibration.genomes.map(g => g.genomeSha)).toEqual(await Promise.all(design.map(g => genomeSha(1, g))))
    expect(p.calibration.genomes[0]!.genomeSha).toBe(centreSha)
    expect(p.calibration.need).toEqual(Array(10).fill(3))
    expect(p).toMatchObject({ models: {}, incumbent: null, confirmation: null })
    expect(p.page).toEqual({ eligibleLives: 0, lastPromotion: null, calibrationLives: 0, calibrationTarget: SIGNAL_LOOKS[0], looks: 0, rowsSinceDeclaration: 0 })
    const c = calibrationOf(out)
    expect(c).toMatchObject({ schemaVersion: 1, genomeSchema: 1, evaluator: CONFIG.search, signal: CONFIG.signal, seed: "calib/1", repeats: 3, looks: [], outcome: "pending" })
    expect(c.genomes).toEqual(p.calibration.genomes)
    expect(out.indexLine).not.toBeNull()
    expect(JSON.parse(out.indexLine!)).toEqual({ posteriorSha: await shaOf(out[POSTERIOR]), sourceCommit: commit(1), builtAt: BUILT_AT })
    expect(Object.keys(JSON.parse(out.indexLine!))).toEqual(["posteriorSha", "sourceCommit", "builtAt"])
    // The page's draw takes a calibration life from this file, weighted by need.
    const d = birthDraw({ learnerId: lid(1), space: SPACE, centre: CENTRE, centreSha, widthsClass: "gpu", posterior: p, posteriorSha: await shaOf(out[POSTERIOR]) })
    expect(d.method).toBe("calibrate")
    // 05a's posterior bytes for this build (never regenerated): 05b's are exactly those plus the two v2 fields.
    const old = JSON.parse(readFileSync(join(GOLDEN, "first-posterior-05a.json"), "utf8")) as { schemaVersion: number; page: Record<string, unknown> }
    expect(old.schemaVersion).toBe(1)
    expect(out[POSTERIOR]).toBe(serialise({ ...old, schemaVersion: 2, page: { ...old.page, rowsSinceDeclaration: 0 } }))
    golden("first-calibration.json", out[CALIBRATION])
  })

  it("buildMeta twice on the same input → identical bytes; a different Map insertion order → identical bytes", async () => {
    const w = await world()
    await build(w)
    await grow(w, 12, planted)
    await bornLife(null) // an offline centre birth
    await build(w)
    await grow(w, 5, planted) // born under the second posterior, so the history has two entries
    const input: MetaInput = {
      runsDir: runsDir(), config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(99), builtAt: BUILT_AT,
      previous: w.previous, posteriorsBySha: w.history, centres: w.centres,
    }
    const a = await buildMeta(input)
    const b = await buildMeta(input)
    expect(b).toEqual(a)
    expect(w.history.size).toBe(2)
    const reversed = new Map([...w.history].reverse())
    const c = await buildMeta({ ...input, posteriorsBySha: reversed })
    expect(c).toEqual(a)
    expect(posteriorOf(a).page.calibrationLives).toBe(17)
    expect(verifyOf(a).ineligible).toEqual([])
  })

  it("30 calibration rows with a planted effect → look 1 appended, outcome signal, mode search; a second build appends nothing", async () => {
    const w = await world()
    await build(w)
    await grow(w, 29, planted)
    let out = await build(w)
    expect(calibrationOf(out).looks).toEqual([]) // 29 < 30: no look yet
    expect(posteriorOf(out).mode).toBe("calibrate")
    await grow(w, 1, planted)
    out = await build(w)
    const c = calibrationOf(out)
    expect(c.looks).toHaveLength(1)
    const look = c.looks[0]!
    expect(look).toMatchObject({ look: 1, lives: 30, sourceCommit: commit(3) })
    expect(look.output).toMatchObject({ evaluator: "signal-check", version: 1, eligible: true, score: 1 })
    expect(look.learnerIds).toHaveLength(30)
    expect(look.learnerIds).toEqual([...look.learnerIds].sort())
    expect(c.outcome).toBe("signal")
    expect(c.repeats).toBe(3)
    const p = posteriorOf(out)
    expect(p.mode).toBe("search")
    expect(p.page).toMatchObject({ eligibleLives: 30, calibrationLives: 30, looks: 1 })
    const per = (p.diagnostics as { perGenome: { genomeSha: string; born: number; rows: number; failed: number; abandoned: number }[] }).perGenome
    expect(per.map(g => g.genomeSha)).toEqual(p.calibration.genomes.map(g => g.genomeSha))
    expect(per.reduce((s, g) => s + g.rows, 0)).toBe(30)
    expect(p.calibration.need).toEqual(per.map(g => Math.max(0, 3 - g.rows)))
    golden("signal-posterior.json", out[POSTERIOR])
    golden("signal-calibration.json", out[CALIBRATION])

    // A second build on the same rows: the recorded look is kept as it was, and nothing is appended.
    const again = await build(w)
    expect(again[CALIBRATION]).toBe(out[CALIBRATION])
    expect(again[POSTERIOR]).toBe(out[POSTERIOR])
    expect(again.indexLine).toBeNull()
    // Even with more calibration lives, a decided outcome takes no more looks.
    await grow(w, 16, planted)
    const later = await build(w)
    expect(calibrationOf(later).looks).toEqual(c.looks)
    expect(calibrationOf(later).outcome).toBe("signal")
  })

  it("05a's world under 05b code: lives born under 05a's v1 posteriors verify, calibration.json is 05a's bytes, and the search posterior is 05a's plus the model", async () => {
    // 05a's posterior for a 05b one: schemaVersion 1, no page.rowsSinceDeclaration, and none of the search's state.
    const to05a = (text: string): string => {
      const v = JSON.parse(text) as Record<string, unknown> & { page: Record<string, unknown>; diagnostics: Record<string, unknown> }
      const page = { ...v.page }, diagnostics = { ...v.diagnostics }
      delete page.rowsSinceDeclaration
      delete diagnostics.model
      return serialise({ ...v, schemaVersion: 1, models: {}, incumbent: null, confirmation: null, page, diagnostics })
    }
    const fixture = (name: string) => readFileSync(join(GOLDEN, name), "utf8")
    const w = await world()
    let out = await build(w)
    let v1 = to05a(out[POSTERIOR])
    expect(v1).toBe(fixture("first-posterior-05a.json"))
    w.history.set(await shaOf(v1), v1)
    // The lives of 05a's own test of this (its learnerIds, k = 19…48), born under 05a's bytes by 05b's birthDraw.
    for (let i = 0; i < 29; i++) await bornLife(v1, { effect: planted, k: 19 + i })
    out = await build(w)
    v1 = to05a(out[POSTERIOR])
    w.history.set(await shaOf(v1), v1)
    await bornLife(v1, { effect: planted, k: 48 })
    out = await build(w)
    expect(verifyOf(out)).toMatchObject({ checked: 30, ineligible: [] })
    expect(out[CALIBRATION]).toBe(fixture("signal-calibration-05a.json"))
    const p = posteriorOf(out)
    expect(p.mode).toBe("search")
    expect(p.models.gpu?.rows).toHaveLength(30)
    expect(to05a(out[POSTERIOR])).toBe(fixture("signal-posterior-05a.json"))
  })

  it("no signal at 30 → repeats 6, mode calibrate; no signal at 45 and 60 → flat", async () => {
    const w = await world()
    await build(w)
    await grow(w, 30, noise)
    let out = await build(w)
    let c = calibrationOf(out)
    expect(c.looks.map(l => [l.look, l.lives, l.output.eligible, l.output.score])).toEqual([[1, 30, true, 0]])
    expect(c).toMatchObject({ repeats: 6, outcome: "pending" })
    let p = posteriorOf(out)
    expect(p.mode).toBe("calibrate")
    const per = (q: Posterior) => (q.diagnostics as { perGenome: { rows: number }[] }).perGenome
    expect(p.calibration.need).toEqual(per(p).map(g => Math.max(0, 6 - g.rows)))
    expect(p.page).toMatchObject({ calibrationLives: 30, calibrationTarget: SIGNAL_LOOKS[1], looks: 1 })

    await grow(w, 14, noise)
    out = await build(w)
    expect(calibrationOf(out).looks).toHaveLength(1) // 44 < 45
    await grow(w, 1, noise)
    out = await build(w)
    c = calibrationOf(out)
    expect(c.looks.map(l => [l.look, l.lives, l.output.score])).toEqual([[1, 30, 0], [2, 45, 0]])
    expect(c).toMatchObject({ repeats: 6, outcome: "pending" })
    expect(posteriorOf(out).mode).toBe("calibrate")

    await grow(w, 15, noise)
    out = await build(w)
    c = calibrationOf(out)
    expect(c.looks.map(l => [l.look, l.lives, l.output.score])).toEqual([[1, 30, 0], [2, 45, 0], [3, 60, 0]])
    expect(c.outcome).toBe("flat")
    p = posteriorOf(out)
    expect(p.mode).toBe("flat")
    expect(p.page).toMatchObject({ calibrationLives: 60, calibrationTarget: SIGNAL_LOOKS[2], looks: 3 })
    // Looks 1 and 2 were never recomputed: their bytes are the ones first recorded.
    expect(JSON.stringify(c.looks.slice(0, 2))).toBe(JSON.stringify(calibrationOf(w.last!).looks.slice(0, 2)))
  })

  it("re-verification: a life whose logged method or panel was altered → ineligible and flagged; a calibrate genome not in the design → ineligible; a genome off by 1 ulp with a valid sha → kept, mismatch counted", async () => {
    const w = await world()
    const first = await build(w)
    const p0 = first[POSTERIOR]
    await grow(w, 10, noise)
    // A tab claiming it drew centre, running the centre: the posterior's draw was calibrate.
    const method = await bornLife(p0, { alter: async m => ({ ...m, method: "centre", genome: { ...CENTRE }, genomeSha: await shaOfGenome(CENTRE) }) })
    // A centre draw whose genome is no centre main ever had (here, a design genome) cannot be checked at all.
    const notCentre = await bornLife(p0, { alter: m => ({ ...m, method: "centre" }) })
    expect(notCentre.meta.genomeSha).not.toBe(await shaOfGenome(CENTRE))
    const panel = await bornLife(p0, { alter: m => ({ ...m, evalPanel: (m.evalPanel! % 16) + 1 }) })
    const offDesign = await bornLife(p0, {
      alter: async m => {
        const genome = { ...CENTRE, lr: 1e-3 }
        return { ...m, genome, genomeSha: await genomeSha(1, genome) }
      },
    })
    const unknown = await bornLife(p0, { alter: m => ({ ...m, posteriorSha: "9".repeat(64) }) })
    // A posterior in search mode that main once published (05a draws it like paused: 1 in 5 uniform, else centre).
    const searchText = p0.replace('"mode": "calibrate"', '"mode": "search"')
    expect(parsePosterior(searchText)?.mode).toBe("search")
    w.history.set(await shaOf(searchText), searchText)
    let k = 5000
    let uniform: Awaited<ReturnType<typeof bornLife>> | null = null
    while (!uniform) {
      const learnerId = lid(0x10000 + k)
      const d = birthDraw({ learnerId, space: SPACE, centre: CENTRE, centreSha: await genomeSha(1, CENTRE), widthsClass: "gpu", posterior: parsePosterior(searchText), posteriorSha: await shaOf(searchText) })
      if (d.method === "uniform") {
        uniform = await bornLife(searchText, {
          k,
          alter: async m => {
            const genome = { ...m.genome, lr: nextUp(m.genome.lr as number) }
            return { ...m, genome, genomeSha: await genomeSha(1, genome) }
          },
        })
      }
      k++
    }
    expect(uniform.meta.genome.lr).not.toBe(uniform.draw.genome.lr)
    const offline = await bornLife(null)
    expect(offline.meta).toMatchObject({ method: "centre", posteriorSha: null })

    const out = await build(w)
    const v = verifyOf(out)
    expect(v.ineligible).toEqual([
      { learnerId: method.learnerId, reason: "method centre, recomputed calibrate" },
      { learnerId: panel.learnerId, reason: `evalPanel ${panel.meta.evalPanel}, recomputed ${panel.draw.evalPanel}` },
      { learnerId: offDesign.learnerId, reason: "calibrate genome not in the calibration design" },
      { learnerId: unknown.learnerId, reason: "unknown posteriorSha" },
      { learnerId: notCentre.learnerId, reason: "unknown centre" },
    ].sort((a, b) => (a.learnerId < b.learnerId ? -1 : 1)))
    expect(v.genomeMismatch).toEqual({ count: 1, rate: 1 / v.checked, learnerIds: [uniform.learnerId] })
    expect(v.checked).toBe(10 + 5 + 2)
    const p = posteriorOf(out)
    // Rows: the 10 honest calibration lives, the uniform life (kept on its logged genome) and the offline centre life.
    expect(p.page).toMatchObject({ eligibleLives: 12, calibrationLives: 10 })
    // The posterior counts mismatches among its rows only; diagnostics.json among every life checked.
    expect((p.diagnostics as { genomeMismatch: unknown }).genomeMismatch).toEqual({ count: 1, rate: 1 / 12, learnerIds: [uniform.learnerId] })
  })

  it("re-verification: an unreadable posterior and a mismatched drawSeed are flagged; a life of another genome schema is skipped", async () => {
    const w = await world()
    const first = await build(w)
    const garbage = "not a posterior\n"
    const garbageSha = await shaOf(garbage)
    w.history.set(garbageSha, garbage)
    const unreadable = await bornLife(first[POSTERIOR], { alter: m => ({ ...m, posteriorSha: garbageSha }) })
    const seed = await bornLife(first[POSTERIOR], { alter: m => ({ ...m, drawSeed: (m.drawSeed! + 1) >>> 0 }) })
    const other = await bornLife(first[POSTERIOR], { alter: m => ({ ...m, genomeSchema: 2 }) })
    expect(collectLives(runsDir()).map(l => l.learnerId)).toContain(other.learnerId) // it parses, and is merged data
    const v = verifyOf(await build(w))
    expect(v.ineligible).toEqual([
      { learnerId: unreadable.learnerId, reason: "posterior unreadable" },
      { learnerId: seed.learnerId, reason: `drawSeed ${seed.meta.drawSeed}, recomputed ${seed.draw.drawSeed}` },
    ])
    expect(v.checked).toBe(2) // the other-schema life is neither checked nor a row
    expect(posteriorOf(w.last!).page.eligibleLives).toBe(0)
  })

  it("a burst of merges cannot bring looks close together: each look needs the schedule's spacing since the last", async () => {
    const w = await world()
    await build(w)
    await grow(w, 40, noise) // 40 at once: look 1 runs at 40, not 30
    let c = calibrationOf(await build(w))
    expect(c.looks.map(l => [l.look, l.lives, l.output.score])).toEqual([[1, 40, 0]])
    await grow(w, 14, noise) // 54 ≥ 45, but only 14 since look 1: it waits for 40 + 15
    expect(calibrationOf(await build(w)).looks).toHaveLength(1)
    await grow(w, 1, noise)
    c = calibrationOf(await build(w))
    expect(c.looks.map(l => [l.look, l.lives])).toEqual([[1, 40], [2, 55]])
    await grow(w, 14, noise)
    expect(calibrationOf(await build(w)).looks).toHaveLength(2) // 69 < 55 + 15
    await grow(w, 1, noise)
    c = calibrationOf(await build(w))
    expect(c.looks.map(l => [l.look, l.lives, l.output.score])).toEqual([[1, 40, 0], [2, 55, 0], [3, 70, 0]])
    expect(c.outcome).toBe("flat")
  })

  it("a newborn, legacy or cpu life is never a row", async () => {
    const w = await world()
    const first = await build(w)
    const newbornId = lid(0x77)
    const newborn: LogFileV2 = {
      ...scoredLife(newbornId, { ms: {}, ratio: 1 })[0]!,
      seq: 0, cu: 0, stepsTotal: 0, final: true, curve: [],
      checkups: NEWBORN_LEVELS.map(level => ({ ...checkup(level, { M2: 1 }), steps: 0 })), milestoneSuiteVersion: 1,
      meta: {
        v: 1, role: "newborn", method: "newborn", genomeSchema: 1, genome: { ...CENTRE }, genomeSha: await genomeSha(1, CENTRE),
        drawSeed: null, posteriorSha: null, evalPanel: 1, evalSeed: null, confirmation: null, inheritance: { filters: "none" }, heldOut: { loss: 1, baseline: 1 },
      },
    }
    put(newborn)
    const legacyId = lid(0x78)
    for (const f of typicalLife(legacyId, {
      meta: {
        v: 1, role: "legacy", method: "legacy", genomeSchema: 1, genome: { ...CENTRE }, genomeSha: await genomeSha(1, CENTRE),
        drawSeed: null, posteriorSha: null, evalPanel: null, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
      },
    })) put(f)
    const cpu = await bornLife(first[POSTERIOR], { widths: CPU })
    expect(collectLives(runsDir()).map(l => l.learnerId).sort()).toEqual([newbornId, legacyId, cpu.learnerId].sort())
    const out = await build(w)
    const p = posteriorOf(out)
    expect(p.page).toMatchObject({ eligibleLives: 0, calibrationLives: 0 })
    const per = (p.diagnostics as { perGenome: { rows: number }[] }).perGenome
    expect(per.every(g => g.rows === 0)).toBe(true)
    // The cpu birth is a draw, so it is re-checked (and passes); newborn and legacy lives are not draws.
    expect(verifyOf(out)).toMatchObject({ checked: 1, ineligible: [], genomeMismatch: { count: 0 } })
  })

  it("a failed calibration life is a row (ranked worst); an abandoned one is not; perGenome counts both", async () => {
    const w = await world()
    const first = await build(w)
    const ended = async (end: "failed" | "orphan") => {
      const l = await bornLife(first[POSTERIOR])
      rmSync(join(runsDir(), l.learnerId), { recursive: true })
      for (const f of endedLife(l.learnerId, end, l.meta)) put(f)
      return l
    }
    const failed = await ended("failed")
    const orphan = await ended("orphan")
    const out = await build(w)
    const p = posteriorOf(out)
    expect(p.page).toMatchObject({ eligibleLives: 1, calibrationLives: 1 })
    const per = diagOf(out).perGenome
    const of = (sha: string) => per.find(g => g.genomeSha === sha)!
    expect(of(failed.meta.genomeSha)).toMatchObject({ failed: 1 })
    expect(of(orphan.meta.genomeSha)).toMatchObject({ abandoned: 1 })
    expect(per.reduce((s, g) => s + g.born, 0)).toBe(2)
    expect(per.reduce((s, g) => s + g.rows, 0)).toBe(1)
    expect(diagOf(out).unscored).toEqual({ abandoned: 1 })
    // The posterior carries only what follows from the rows: the failed row, not the abandoned birth.
    expect((p.diagnostics as { perGenome: unknown[] }).perGenome).toContainEqual({ genomeSha: failed.meta.genomeSha, rows: 1, failed: 1 })
    // A short life that ended at its budget is not abandoned, even when nothing can score it (no reference pinned).
    const short = await bornLife(first[POSTERIOR])
    const end = short.files[short.files.length - 1]!
    put({ ...end, final: true, meta: { ...end.meta!, end: "budget" } })
    const unscored = await build(w, { refs: {} })
    const perNone = diagOf(unscored).perGenome
    expect(perNone.reduce((s, g) => s + g.abandoned, 0)).toBe(1)
    expect(perNone.reduce((s, g) => s + g.born, 0)).toBe(3)
    expect(posteriorOf(unscored).page.eligibleLives).toBe(0)
  })

  it("unchanged rows → indexLine null and posterior bytes unchanged", async () => {
    const w = await world()
    await build(w)
    await grow(w, 4, planted)
    const scored = await bornLife(w.previous.posteriorText, { effect: planted })
    const before = await build(w)
    expect(before.indexLine).not.toBeNull()
    // Main moved on (another merge, another commit), with no change to any row.
    const same = await build(w)
    expect(same[POSTERIOR]).toBe(before[POSTERIOR])
    expect(same[CALIBRATION]).toBe(before[CALIBRATION])
    expect(same.indexLine).toBeNull()
    expect(posteriorOf(same).sourceCommit).toBe(commit(2)) // the commit the unchanged rows first came from
    // A later file of a scored life lies past its budget, so it changes no row either.
    const last = scored.files[scored.files.length - 1]!
    put({ ...last, seq: last.seq + 1, cu: 4096, curve: [], checkups: undefined, milestoneSuiteVersion: undefined } as LogFileV2)
    const later = await build(w)
    expect(later.indexLine).toBeNull()
    expect(later[POSTERIOR]).toBe(before[POSTERIOR])
    // A new birth that is not yet a row (its first file only) moves the diagnostics, never the posterior.
    const fresh = await bornLife(w.previous.posteriorText, { effect: planted })
    rmSync(join(runsDir(), fresh.learnerId), { recursive: true })
    put(fresh.files[0]!)
    const born = await build(w)
    expect(born[POSTERIOR]).toBe(before[POSTERIOR])
    expect(born.indexLine).toBeNull()
    expect(diagOf(born).verify.checked).toBe(diagOf(later).verify.checked + 1)
    expect(diagOf(born).perGenome.reduce((s, g) => s + g.born, 0)).toBe(diagOf(later).perGenome.reduce((s, g) => s + g.born, 0) + 1)
    // A new row changes the posterior; its index line names the new commit.
    await bornLife(w.previous.posteriorText, { effect: planted })
    const changed = await build(w)
    expect(changed[POSTERIOR]).not.toBe(before[POSTERIOR])
    expect(JSON.parse(changed.indexLine!)).toEqual({ posteriorSha: await shaOf(changed[POSTERIOR]), sourceCommit: commit(w.built), builtAt: BUILT_AT })
  })

  it("output has no trailing whitespace and ends with a newline", async () => {
    const w = await world()
    await build(w)
    await grow(w, 30, planted)
    const out = await build(w)
    for (const text of [out[POSTERIOR], out[CALIBRATION], out[DIAGNOSTICS]]) {
      expect(text.endsWith("}\n")).toBe(true)
      expect(text.endsWith("\n\n")).toBe(false)
      expect(text.split("\n").filter(l => /[ \t]+$/.test(l))).toEqual([])
      expect(text).not.toMatch(/\d{4}-\d{2}-\d{2}T/) // no time in either file
    }
    expect(out.indexLine).not.toMatch(/\n/)
  })

  it("a calibration built for another evaluator, signal check or genome schema is refused, not silently restarted", async () => {
    const w = await world()
    await build(w)
    const c = w.previous.calibration!
    for (const bad of [{ ...c, evaluator: "search-composite@2" }, { ...c, signal: "signal-check@2" }, { ...c, genomeSchema: 2 }]) {
      await expect(build(await world(), { previous: { calibration: bad, posteriorText: null, verified: null } })).rejects.toThrow(/calibration/)
    }
    const tampered = { ...c, genomes: c.genomes.map((g, i) => (i === 1 ? { ...g, values: { ...g.values, lr: 1e-3 } } : g)) }
    await expect(build(await world(), { previous: { calibration: tampered, posteriorText: null, verified: null } })).rejects.toThrow(/genomeSha/)
    expect(() => parseCalibration("{}")).toThrow()
    expect(() => parseCalibration(JSON.stringify({ ...c, extra: 1 }))).toThrow()
  })

  it("the calibration design is kept from the previous state, not recomputed from the centre", async () => {
    const w = await world()
    await build(w)
    const design = w.previous.calibration!.genomes
    // A promoted centre (05b) moves the centre; the design stays the one first computed.
    const moved = { ...CENTRE, lr: 1e-3 }
    const out = await build(w, { centre: moved })
    expect(calibrationOf(out).genomes).toEqual(design)
    expect(posteriorOf(out).centre).toEqual({ genomeSha: await genomeSha(1, moved), values: moved })
    expect(decodeGenome(moved, SPACE).learner.lr).toBe(1e-3)
  })
})

describe("buildMeta in 05b: the search, only in mode search (plan 05b Task 8)", () => {
  beforeEach(() => { fitSpy.mockClear(); drawSpy.mockClear() })
  const D = searched(SPACE).length
  const unitGenome = (u: number): Genome => fromUnit(SPACE, Array(D).fill(u), CENTRE)

  it("calibrate mode with 200 eligible rows → posterior schemaVersion 2, models {}, incumbent null, confirmation null; no fit is run (a spy on fitGp is not called)", async () => {
    const w = await world()
    await build(w)
    for (let i = 0; i < 180; i++) await bornLife(null) // offline births: centre rows, never calibration rows
    await grow(w, 20, noise)
    const out = await build(w)
    const p = posteriorOf(out)
    expect(p).toMatchObject({ schemaVersion: 2, mode: "calibrate", models: {}, incumbent: null, confirmation: null })
    expect(p.page).toMatchObject({ eligibleLives: 200, calibrationLives: 20, looks: 0, rowsSinceDeclaration: 0 })
    expect(Object.keys(p.diagnostics).sort()).toEqual(["genomeMismatch", "perGenome"]) // 05a's keys, no model reason
    expect(diagOf(out).model).toBeUndefined()
    expect(fitSpy).not.toHaveBeenCalled()
  }, 60_000)

  it("flat after 3 looks: births keep uniform/centre; still no model", async () => {
    const w = await world()
    await build(w)
    await grow(w, 30, noise)
    await build(w)
    await grow(w, 15, noise)
    await build(w)
    await grow(w, 15, noise)
    const out = await build(w)
    expect(calibrationOf(out).outcome).toBe("flat")
    const p = posteriorOf(out)
    expect(p).toMatchObject({ schemaVersion: 2, mode: "flat", models: {}, incumbent: null, confirmation: null })
    const centreSha = await shaOfGenome(CENTRE)
    const sha = await shaOf(out[POSTERIOR])
    const methods = new Set<string>()
    for (let k = 0; k < 200; k++) methods.add(birthDraw({ learnerId: lid(0x40000 + k), space: SPACE, centre: CENTRE, centreSha, widthsClass: "gpu", posterior: p, posteriorSha: sha }).method)
    expect([...methods].sort()).toEqual(["centre", "uniform"])
    // More lives under flat: the gate still fits nothing.
    await grow(w, 20, noise)
    expect(posteriorOf(await build(w)).models).toEqual({})
    expect(fitSpy).not.toHaveBeenCalled()
  }, 60_000)

  it("a calibration.json with outcome signal but last look score 0 → buildMeta throws, writing nothing", async () => {
    const w = await world()
    await build(w)
    await grow(w, 30, noise)
    await build(w)
    const c = w.previous.calibration!
    expect(c.looks.map(l => l.output.score)).toEqual([0])
    const again = async (calibration: CalibrationState) => build(await world(), { previous: { calibration, posteriorText: null, verified: null } })
    await expect(again({ ...c, outcome: "signal" })).rejects.toThrow(/outcome signal needs a last look scored 1/)
    await expect(again({ ...c, outcome: "flat" })).rejects.toThrow(/outcome flat needs 3 looks/)
    const scored1 = { ...c.looks[0]!, output: { ...c.looks[0]!.output, score: 1 } }
    await expect(again({ ...c, looks: [scored1] })).rejects.toThrow(/outcome pending needs/)
    const alien = { ...c.looks[0]!, output: { ...c.looks[0]!.output, evaluator: "signal-check-2" } }
    await expect(again({ ...c, looks: [alien] })).rejects.toThrow(/is not an output of signal-check@1/)
    // Consistent: accepted. Mode search with no rows: no model, and the reason says why.
    const ok = await again({ ...c, looks: [scored1], outcome: "signal" })
    expect(posteriorOf(ok)).toMatchObject({ mode: "search", models: {}, incumbent: null, diagnostics: { model: { reason: "fewer than 13 rows" } } })
    expect(fitSpy).not.toHaveBeenCalled()
  }, 60_000)

  it("30 planted calibration rows → signal → the next build fits models.gpu on all rows; ts lives born from it verify and become rows", async () => {
    const { w, signal } = await searchWorld()
    expect(calibrationOf(signal).outcome).toBe("signal")
    const p = posteriorOf(signal)
    expect(p.mode).toBe("search")
    const model = p.models.gpu!
    const rows = collectLives(runsDir()).map(l => l.learnerId)
    expect(model.rows.map(r => r.learnerId)).toEqual(rows)
    expect(model.genes).toEqual(searched(SPACE).map(g => g.name))
    expect(fitSpy).toHaveBeenCalledTimes(1)
    expect(p.incumbent).not.toBeNull()
    expect(p.page).toMatchObject({ eligibleLives: 30, rowsSinceDeclaration: 0 })

    const born = await grow(w, 30, planted)
    const ts = born.filter(b => b.meta.method === "ts")
    expect(ts.length).toBeGreaterThan(15)
    expect(born.every(b => b.meta.method === "ts" || b.meta.method === "uniform")).toBe(true)
    const out = await build(w)
    expect(verifyOf(out)).toMatchObject({ checked: 60, ineligible: [], genomeMismatch: { count: 0 } })
    const q = posteriorOf(out)
    expect(q.models.gpu!.rows.map(r => r.learnerId)).toEqual(expect.arrayContaining(ts.map(b => b.learnerId)))
    expect(q.models.gpu!.rows).toHaveLength(60)
    expect(q.page).toMatchObject({ eligibleLives: 60, calibrationLives: 30, rowsSinceDeclaration: 30 })
    expect(fitSpy).toHaveBeenCalledTimes(2)
    expect(diagOf(out).model).toEqual({ rows: 60, fitRows: 60, refit: true, startsWithNoStep: [], reason: null })
  }, 120_000)

  it("a search posterior built twice → identical bytes; equals fixtures/meta/search-posterior.json", async () => {
    resetK(0x1000)
    const { w } = await searchWorld()
    await grow(w, 10, planted)
    const input: MetaInput = {
      runsDir: runsDir(), config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(77), builtAt: BUILT_AT,
      previous: w.previous, posteriorsBySha: w.history, centres: w.centres,
    }
    const a = await buildMeta(input)
    const b = await buildMeta(input)
    expect(b).toEqual(a)
    // Without the previous model, so the second build refits rather than reusing: the same bytes again.
    const prevNoModel = { ...input.previous, posteriorText: null }
    const c = await buildMeta({ ...input, previous: prevNoModel })
    expect(c[POSTERIOR]).toBe(a[POSTERIOR])
    const p = posteriorOf(a)
    expect(p.models.gpu!.rows).toHaveLength(40)
    // The gate writes x rounded to 6 decimals, in rows and the incumbent.
    for (const r of p.models.gpu!.rows) expect(r.x).toEqual(r.x.map(round6))
    expect(p.incumbent!.x).toEqual(p.incumbent!.x.map(round6))
    golden("search-posterior.json", a[POSTERIOR])
  }, 120_000)

  it("verification with a moved centre: a life born under posterior P (centre A) whose tab held centre B (in centres) verifies as centre/another-centre; one whose centre is not in centres → 'unknown centre'", async () => {
    const { w, signal } = await searchWorld()
    const P = signal[POSTERIOR]
    const B = unitGenome(0.3), C = unitGenome(0.7)
    w.centres.set(await shaOfGenome(B), B) // B was main's centre once (a promotion); C never was
    const heldB = await bornLife(P, { centre: B, effect: planted })
    const heldC = await bornLife(P, { centre: C, effect: planted })
    for (const l of [heldB, heldC]) expect(l.draw).toMatchObject({ method: "centre", reason: "posterior built for another centre" })
    expect(heldB.meta.genome).toEqual(B)
    const out = await build(w)
    expect(verifyOf(out).ineligible).toEqual([{ learnerId: heldC.learnerId, reason: "unknown centre" }])
    expect(verifyOf(out).genomeMismatch.count).toBe(0)
    const p = posteriorOf(out)
    expect(p.page.eligibleLives).toBe(31)
    const row = p.models.gpu!.rows.find(r => r.learnerId === heldB.learnerId)!
    expect(row.x).toEqual(toUnit(SPACE, B).map(round6)) // the row is on what the life ran
    // Born offline under a centre main once had: verified too (no posterior to try, only the centre itself).
    const offline = await bornLife(null, { centre: B })
    const later = await build(w)
    expect(verifyOf(later).ineligible.map(f => f.learnerId)).not.toContain(offline.learnerId)
  }, 120_000)

  /** P with confirmation 0 served (candidate unitGenome(0.25) against P's centre), as main would serve it (Task 9 declares). */
  async function withConfirmation(P: string, need = { candidate: 24, centre: 24, newborn: 60 }): Promise<string> {
    const v = JSON.parse(P) as Record<string, unknown> & { centre: { genomeSha: string; values: Genome } }
    const cand = unitGenome(0.25)
    const text = serialise({
      ...v, confirmation: {
        id: 0, seedBase: 10_000, candidate: { genomeSha: await shaOfGenome(cand), values: cand }, centre: v.centre,
        target: { candidate: 24, centre: 24, newborn: 60 }, counted: { candidate: 24 - need.candidate, centre: 24 - need.centre, newborn: 60 - need.newborn }, need,
      },
    })
    expect(parsePosterior(text)?.confirmation?.id).toBe(0)
    return text
  }

  /** The first k ≥ from whose birth under the served posterior is a confirm draw that `pick` accepts. */
  async function confirmK(Q: string, from: number, pick: (d: ReturnType<typeof birthDraw>) => boolean = () => true): Promise<number> {
    const q = parsePosterior(Q)!, qSha = await shaOf(Q), centreSha = await shaOfGenome(CENTRE)
    for (let k = from; ; k++) {
      const d = birthDraw({ learnerId: lid(0x10000 + k), space: SPACE, centre: CENTRE, centreSha, widthsClass: "gpu", posterior: q, posteriorSha: qSha })
      if (d.method === "confirm" && pick(d)) return k
    }
  }
  /** A confirmation newborn as a tab writes it: one frozen file (decision 0024 "Confirmation-newborn files"). */
  async function confirmNewborn(Q: string, k: number) {
    const b = await bornLife(Q, { k })
    rmSync(join(runsDir(), b.learnerId), { recursive: true })
    const one = scoredLife(b.learnerId, { ms: {}, ratio: 1, meta: b.meta })[0]!
    put({
      ...one, seq: 0, cu: 0, stepsTotal: 0, final: true, curve: [],
      checkups: NEWBORN_LEVELS.map(level => ({ ...checkup(level, { M2: 1 }), steps: 0 })), milestoneSuiteVersion: 1,
      meta: { ...b.meta, heldOut: { loss: 1, baseline: 1 } },
    })
    return b
  }
  const notNewborn = (d: ReturnType<typeof birthDraw>) => d.confirmation!.arm !== "newborn"

  it("a confirm life whose genome differs from the declaration's → ineligible; confirm lives verify (evalSeed and arm), duplicate evalSeeds included, and are never rows", async () => {
    const { w, signal } = await searchWorld()
    const Q = await withConfirmation(signal[POSTERIOR])
    w.history.set(await shaOf(Q), Q)
    // Two learnerIds that draw the same candidate or centre seed (the gate counts only the lower one, Task 9).
    const bySeed = new Map<number, number>()
    let pair: [number, number] | null = null
    for (let k = 0x50000; pair === null; k++) {
      k = await confirmK(Q, k, notNewborn)
      const d = (await bornLife(Q, { k })).meta
      const seen = bySeed.get(d.evalSeed!)
      if (seen !== undefined) pair = [seen, k]
      else bySeed.set(d.evalSeed!, k)
    }
    const born = collectLives(runsDir()).map(l => l.files[0] as LogFileV2).filter(f => f.meta?.role === "confirm").map(f => f.meta!)
    const arms = new Set(born.map(m => m.confirmation!.arm))
    expect([...arms].sort()).toEqual(["candidate", "centre"])
    for (const m of born) expect(m.genome).toEqual(m.confirmation!.arm === "centre" ? CENTRE : unitGenome(0.25))
    // And a confirmation newborn, in its frozen file.
    const newborn = await confirmNewborn(Q, await confirmK(Q, 0x58000, d => d.confirmation!.arm === "newborn"))
    expect(newborn.meta.genome).toEqual(unitGenome(0.25))
    // A candidate-arm life that ran another genome (valid, with its own sha): not the declaration's.
    const other = unitGenome(0.26)
    const bad = await bornLife(Q, {
      k: await confirmK(Q, 0x59000, d => d.confirmation!.arm === "candidate"),
      alter: async m => ({ ...m, genome: other, genomeSha: await shaOfGenome(other) }),
    })
    const out = await build(w)
    expect(verifyOf(out).ineligible).toEqual([{ learnerId: bad.learnerId, reason: "genome is not confirmation 0's candidate" }])
    expect(verifyOf(out).checked).toBe(30 + born.length + 2)
    expect(verifyOf(out).genomeMismatch.count).toBe(0)
    const p = posteriorOf(out)
    expect(p.page.eligibleLives).toBe(30) // confirmation lives are never rows
    expect(p.models.gpu!.rows.map(r => r.learnerId)).toEqual(collectLives(runsDir()).filter(l => (l.files[0] as LogFileV2).meta?.role === "search").map(l => l.learnerId))
  }, 120_000)

  it("a confirm life with a logged evalSeed or confirmation the draw does not reproduce → ineligible", async () => {
    const { w, signal } = await searchWorld()
    const Q = await withConfirmation(signal[POSTERIOR])
    w.history.set(await shaOf(Q), Q)
    const k0 = await confirmK(Q, 0x70000, notNewborn), k1 = await confirmK(Q, k0 + 1, notNewborn)
    // Another seed of the same arm (j + 4), and another confirmation's block: both blocks stay self-consistent.
    const seed = await bornLife(Q, { k: k0, alter: m => ({ ...m, evalSeed: m.evalSeed! + (m.evalSeed! + 4 <= 10_999 ? 4 : -4) }) })
    const id = await bornLife(Q, { k: k1, alter: m => ({ ...m, confirmation: { ...m.confirmation!, id: 1 }, evalSeed: m.evalSeed! + 1000 }) })
    expect(collectLives(runsDir()).map(l => l.learnerId)).toEqual(expect.arrayContaining([seed.learnerId, id.learnerId])) // they parse
    const out = await build(w)
    expect(verifyOf(out).ineligible).toEqual([
      { learnerId: seed.learnerId, reason: `evalSeed ${seed.meta.evalSeed}, recomputed ${seed.draw.evalSeed}` },
      { learnerId: id.learnerId, reason: `evalSeed ${id.meta.evalSeed}, recomputed ${id.draw.evalSeed}` },
    ])
  }, 120_000)

  it("verified.json: a second build verifies only new lives (a spy counts birthDraw calls); a changed verifier re-verifies all", async () => {
    const w = await world()
    await build(w)
    await grow(w, 12, noise)
    await bornLife(null)
    drawSpy.mockClear()
    const first = await build(w)
    expect(drawSpy).toHaveBeenCalledTimes(13)
    const cache = parseVerified(first[VERIFIED])!
    expect(cache).toMatchObject({ schemaVersion: 1, verifier: 2 })
    expect(Object.keys(cache.lives)).toHaveLength(13)
    expect(Object.values(cache.lives).every(v => v.reason === null && !v.genomeMismatch)).toBe(true)

    await grow(w, 3, noise)
    // The history walk is wanted only for the new lives' posterior.
    expect([...wantedPosteriors(runsDir(), cache)]).toEqual([await shaOf(w.previous.posteriorText!)])
    drawSpy.mockClear()
    const second = await build(w)
    expect(drawSpy).toHaveBeenCalledTimes(3)
    expect(Object.keys(parseVerified(second[VERIFIED])!.lives)).toHaveLength(16)
    // The cache saves work only: the same files as a build from no cache.
    const fresh = await buildMeta({
      runsDir: runsDir(), config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(w.built), builtAt: BUILT_AT,
      previous: { ...w.previous, verified: null }, posteriorsBySha: w.history, centres: w.centres,
    })
    expect(drawSpy).toHaveBeenCalledTimes(3 + 16)
    expect(fresh[DIAGNOSTICS]).toBe(second[DIAGNOSTICS])
    expect(fresh[VERIFIED]).toBe(second[VERIFIED])

    drawSpy.mockClear()
    const old = { ...w.previous.verified!, verifier: 1 } as unknown as VerifiedCache
    await build(w, { previous: { ...w.previous, verified: old } })
    expect(drawSpy).toHaveBeenCalledTimes(16)
    // A cached verdict holds for its posteriorSha only; a malformed cache parses as none.
    expect(parseVerified("{}")).toBeNull()
    expect(parseVerified(JSON.stringify({ schemaVersion: 1, verifier: 2, lives: { nope: { posteriorSha: null, reason: null, genomeMismatch: false } } }))).toBeNull()
  }, 60_000)

  it("a life born under the last v1 posterior verifies under 05b code", async () => {
    const text = readFileSync(join(GOLDEN, "last-v1-posterior.json"), "utf8") // main's data/meta/posterior.json under 05a
    const v1 = parsePosterior(text)!
    expect(v1).toMatchObject({ schemaVersion: 1, mode: "calibrate" })
    const w = await world()
    await build(w)
    w.history.set(await shaOf(text), text)
    const born = [await bornLife(text), await bornLife(text), await bornLife(text, { widths: CPU })]
    expect(born.map(b => b.meta.method).slice(0, 2)).toEqual(["calibrate", "calibrate"])
    expect(["uniform", "centre"]).toContain(born[2]!.meta.method) // a cpu tab never calibrates
    const out = await build(w)
    expect(verifyOf(out)).toMatchObject({ checked: 3, ineligible: [], genomeMismatch: { count: 0 } })
    expect(posteriorOf(out).page).toMatchObject({ eligibleLives: 2, calibrationLives: 2 })
  }, 60_000)
})

describe("posteriorHistory", () => {
  it("resolves only the wanted shas, newest first, stopping once all are found; skips commits that delete the file", async () => {
    const texts: Record<string, string> = { c1: '{"a":1}\n', c2: '{"a":2}\n', c3: '{"a":3}\n' }
    const sha = Object.fromEntries(await Promise.all(Object.entries(texts).map(async ([c, t]) => [c, await shaOf(t)] as const)))
    let calls: string[][] = []
    const git = (args: string[]) => {
      calls.push(args)
      if (args[0] === "log") return "c3\nc2\nc1\n"
      if (args[0] === "show") return texts[args[1]!.split(":")[0]!]!
      throw new Error(`unexpected git ${args.join(" ")}`)
    }
    const shows = () => calls.filter(a => a[0] === "show").map(a => a[1])
    const m = posteriorHistory(git, new Set([sha.c2!]))
    expect([...m]).toEqual([[sha.c2, texts.c2]])
    expect(calls[0]).toEqual(["log", "--format=%H", "--diff-filter=d", "--", POSTERIOR])
    expect(shows()).toEqual([`c3:${POSTERIOR}`, `c2:${POSTERIOR}`]) // c1 is never read
    calls = []
    const both = posteriorHistory(git, new Set([sha.c1!, sha.c3!]))
    expect([...both.keys()].sort()).toEqual([sha.c1, sha.c3].sort())
    calls = []
    expect(posteriorHistory(git, new Set(["9".repeat(64)])).size).toBe(0) // walks it all, finds nothing
    expect(shows()).toHaveLength(3)
    calls = []
    expect(posteriorHistory(git, new Set()).size).toBe(0)
    expect(calls).toEqual([]) // nothing wanted: no git at all
  })

  it("wantedPosteriors lists the posteriorShas lives were born with", async () => {
    const w = await world()
    const first = await build(w)
    await bornLife(first[POSTERIOR])
    await bornLife(null)
    expect([...wantedPosteriors(runsDir())]).toEqual([await shaOf(first[POSTERIOR])])
  })
})

describe("meta-cli (a checkout of main in a temp git repo)", () => {
  let fx: Fixture
  beforeEach(() => { fx = makeFixture() })
  afterEach(() => { fx.cleanup() })
  const cli = () => execFileSync(join(REPO, "node_modules/.bin/tsx"), [join(REPO, "tools/gate/meta-cli.ts")], {
    cwd: fx.work, encoding: "utf8", env: { ...process.env, ...FIXTURE_ENV }, stdio: ["ignore", "pipe", "pipe"],
  }).trim()
  const inWork = (rel: string) => readFileSync(join(fx.work, rel), "utf8")
  const runFiles = (learnerId: string) => Object.fromEntries(
    readdirSync(join(runsDir(), learnerId)).map(n => [`data/runs/${learnerId}/${n}`, readFileSync(join(runsDir(), learnerId, n), "utf8")]),
  )

  it("writes data/meta, resolves a life's posterior from history, and prints changed=true|false", async () => {
    fx.commitMain({
      "config/evaluation.json": readFileSync(join(REPO, "config/evaluation.json"), "utf8"),
      "knowledge/genome.json": readFileSync(join(REPO, "knowledge/genome.json"), "utf8"),
    })
    const offline = await bornLife(null)
    fx.commitMain(runFiles(offline.learnerId))
    expect(cli()).toBe("changed=true")
    const posterior = inWork(POSTERIOR)
    expect(parsePosterior(posterior)?.mode).toBe("calibrate")
    expect(JSON.parse(inWork(DIAGNOSTICS)).verify).toMatchObject({ checked: 1, ineligible: [] })
    const index = inWork("data/meta/posterior-index.jsonl").split("\n").filter(Boolean)
    expect(index).toHaveLength(1)
    expect(JSON.parse(index[0]!).posteriorSha).toBe(await shaOf(posterior))
    // Each life's birth verdict is recorded (the offline life verified against genome.json from main's history).
    expect(parseVerified(inWork(VERIFIED))?.lives).toEqual({ [offline.learnerId]: { posteriorSha: null, reason: null, genomeMismatch: false } })
    expect(cli()).toBe("changed=false")

    // The gate commits data/meta (Task 11); a tab is born under that posterior and its log merges.
    const meta = Object.fromEntries(["posterior.json", "calibration.json", "diagnostics.json", "verified.json", "posterior-index.jsonl"].map(n => [`data/meta/${n}`, inWork(`data/meta/${n}`)]))
    rmSync(join(fx.work, "data/meta"), { recursive: true })
    fx.commitMain(meta)
    const calib = await bornLife(posterior)
    expect(calib.meta.method).toBe("calibrate")
    fx.commitMain(runFiles(calib.learnerId))
    // No newborn reference in this checkout, so no row changes: the posterior keeps its bytes, the diagnostics move.
    expect(cli()).toBe("changed=true")
    expect(inWork(POSTERIOR)).toBe(posterior)
    expect(JSON.parse(inWork(DIAGNOSTICS)).verify).toMatchObject({ checked: 2, ineligible: [] }) // its posterior was found
    expect(Object.keys(parseVerified(inWork(VERIFIED))!.lives)).toEqual([offline.learnerId, calib.learnerId].sort())
    expect(inWork("data/meta/posterior-index.jsonl").split("\n").filter(Boolean)).toHaveLength(1)
    expect(cli()).toBe("changed=false")
  }, 60_000)
})
