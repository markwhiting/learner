import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { createRng, sobolPoints } from "@blanktab/detmath"
import { NEWBORN_LEVELS, type LogFile, type LogFileV2, type MetaBlock } from "@blanktab/harness/log"
import {
  birthDraw, decodeGenome, fromUnit, genomeSha, parseCentre, parseGenomeSpace, parsePosterior, searched, sha256Hex,
  type Genome, type Posterior,
} from "@blanktab/meta"
import { parseNewbornReference, SIGNAL_LOOKS, type NewbornReference } from "@blanktab/evaluators"
import { CPU, checkup, endedLife, lid, scoredLife, typicalLife } from "../../../packages/evaluators/test/fixtures/build.ts"
import { loadEvalConfig } from "../../eval/config.ts"
import { collectLives } from "../../eval/lives.ts"
import { FIXTURE_ENV, makeFixture, type Fixture } from "./git-fixture.ts"
import { buildMeta, parseCalibration, posteriorHistory, wantedPosteriors, type CalibrationState, type MetaFiles, type MetaInput } from "../meta.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
const GOLDEN = fileURLToPath(new URL("./fixtures/meta/", import.meta.url))
const SPACE = parseGenomeSpace(JSON.parse(readFileSync(join(REPO, "knowledge/genome-space.json"), "utf8")))
const CENTRE = parseCentre(JSON.parse(readFileSync(join(REPO, "knowledge/genome.json"), "utf8")), SPACE)
const CONFIG = loadEvalConfig(REPO)
const REF = parseNewbornReference(JSON.parse(readFileSync(join(REPO, "packages/evaluators/test/fixtures/reference.json"), "utf8"))) as NewbornReference
const REFS = { newborn: REF }
const commit = (n: number) => n.toString(16).padStart(40, "0")
const BUILT_AT = "2026-10-23T12:00:00.000Z"
const POSTERIOR = "data/meta/posterior.json"
const CALIBRATION = "data/meta/calibration.json"
const DIAGNOSTICS = "data/meta/diagnostics.json"

const shaOf = (text: string) => sha256Hex(new TextEncoder().encode(text))

let runs: string
beforeEach(() => { runs = mkdtempSync(join(tmpdir(), "meta-")) })
afterEach(() => { rmSync(runs, { recursive: true, force: true }) })
const put = (f: LogFile) => {
  mkdirSync(join(runs, f.learnerId), { recursive: true })
  writeFileSync(join(runs, f.learnerId, `${f.seq}.json`), JSON.stringify(f, null, 2))
}

/** The gate's state between builds, as main's data/meta and its history would hold it. */
interface World { previous: MetaInput["previous"]; history: Map<string, string>; last: MetaFiles | null; built: number }
const world = (): World => ({ previous: { calibration: null, posteriorText: null }, history: new Map(), last: null, built: 0 })

/** One gate run on `w`'s state at a new main commit; the result becomes main's data/meta. */
async function build(w: World, over: Partial<MetaInput> = {}): Promise<MetaFiles> {
  const out = await buildMeta({
    runsDir: runs, config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(++w.built), builtAt: BUILT_AT,
    previous: w.previous, posteriorsBySha: w.history, ...over,
  })
  w.previous = { calibration: parseCalibration(out[CALIBRATION]), posteriorText: out[POSTERIOR] }
  w.history.set(await shaOf(out[POSTERIOR]), out[POSTERIOR])
  w.last = out
  return out
}
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
}
const diagOf = (out: MetaFiles) => JSON.parse(out[DIAGNOSTICS]) as Diagnostics
const verifyOf = (out: MetaFiles) => diagOf(out).verify

/** How a test life scores: its M2 measure and held-out ratio. */
type Effect = (k: number, calibrationIndex: number | undefined) => { m2: number; ratio: number }
/** A genome effect far above the repeat noise (as signal-check's own test plants it). */
const planted: Effect = (k, i) => ({ m2: 0.3 + 0.4 * (i ?? 0) + 0.01 * (k % 3), ratio: 0.98 - 0.01 * (i ?? 0) + 0.001 * (k % 3) })
/** Scores that depend on the learner alone, never on its genome. */
const noise: Effect = k => {
  const r = createRng(0x5eed + k)
  return { m2: 0.3 + 2 * r(), ratio: 0.9 + 0.08 * r() }
}

let nextK = 1
/**
 * A life born as a tab would birth it under the posterior `p` (bytes `text`), on GPU widths unless `widths` says
 * otherwise: birthDraw's genome, method, seed and panel, logged in its meta block. `alter` edits the block it logs.
 */
async function bornLife(text: string | null, opts: { effect?: Effect; widths?: LogFileV2["widths"]; alter?: (m: MetaBlock) => MetaBlock | Promise<MetaBlock>; k?: number } = {}) {
  const k = opts.k ?? nextK++
  const learnerId = lid(0x10000 + k)
  const posterior = text === null ? null : parsePosterior(text)
  const posteriorSha = text === null ? null : await shaOf(text)
  const centreSha = await genomeSha(SPACE.schemaVersion, CENTRE)
  const d = birthDraw({ learnerId, space: SPACE, centre: CENTRE, centreSha, widthsClass: opts.widths ? "cpu" : "gpu", posterior, posteriorSha })
  let meta: MetaBlock = {
    v: 1, role: d.role, method: d.method, genomeSchema: SPACE.schemaVersion, genome: d.genome, genomeSha: await genomeSha(SPACE.schemaVersion, d.genome),
    drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
  }
  if (opts.alter) meta = await opts.alter(meta)
  const e = (opts.effect ?? noise)(k, d.calibrationIndex)
  const files = opts.widths
    ? typicalLife(learnerId, { meta, widths: opts.widths })
    : scoredLife(learnerId, { ms: { M1: 0.9, M2: e.m2, M3: 0.2, M4: 0.3, M5: 0.8, M6: 0 }, ratio: e.ratio, meta })
  for (const f of files) put(f)
  return { learnerId, draw: d, meta, files }
}
/** `n` more calibration lives under the current posterior. */
async function grow(w: World, n: number, effect: Effect) {
  for (let i = 0; i < n; i++) await bornLife(w.previous.posteriorText, { effect })
}

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
  it("first build: mode calibrate, 10 genomes (centre first), need 3 each, no looks", async () => {
    const w = world()
    const out = await build(w)
    const p = posteriorOf(out)
    const centreSha = await genomeSha(1, CENTRE)
    expect(p).toMatchObject({ schemaVersion: 1, genomeSchema: 1, evaluator: CONFIG.search, sourceCommit: commit(1), mode: "calibrate" })
    expect(p.centre).toEqual({ genomeSha: centreSha, values: CENTRE })
    expect(p.calibration.genomes).toHaveLength(10)
    const design: Genome[] = [CENTRE, ...sobolPoints(searched(SPACE).length, 9, "calib/1").map(u => fromUnit(SPACE, u, CENTRE))]
    expect(p.calibration.genomes.map(g => g.values)).toEqual(design)
    expect(p.calibration.genomes.map(g => g.genomeSha)).toEqual(await Promise.all(design.map(g => genomeSha(1, g))))
    expect(p.calibration.genomes[0]!.genomeSha).toBe(centreSha)
    expect(p.calibration.need).toEqual(Array(10).fill(3))
    expect(p).toMatchObject({ models: {}, incumbent: null, confirmation: null })
    // A v1 file reads with rowsSinceDeclaration 0 (packages/meta posterior.ts; plan 05b Task 1).
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
    golden("first-posterior.json", out[POSTERIOR])
    golden("first-calibration.json", out[CALIBRATION])
  })

  it("buildMeta twice on the same input → identical bytes; a different Map insertion order → identical bytes", async () => {
    const w = world()
    await build(w)
    await grow(w, 12, planted)
    await bornLife(null) // an offline centre birth
    await build(w)
    await grow(w, 5, planted) // born under the second posterior, so the history has two entries
    const input: MetaInput = {
      runsDir: runs, config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(99), builtAt: BUILT_AT,
      previous: w.previous, posteriorsBySha: w.history,
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
    const w = world()
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

  it("no signal at 30 → repeats 6, mode calibrate; no signal at 45 and 60 → flat", async () => {
    const w = world()
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
    const w = world()
    const first = await build(w)
    const p0 = first[POSTERIOR]
    await grow(w, 10, noise)
    const method = await bornLife(p0, { alter: m => ({ ...m, method: "centre" }) })
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
    ].sort((a, b) => (a.learnerId < b.learnerId ? -1 : 1)))
    expect(v.genomeMismatch).toEqual({ count: 1, rate: 1 / v.checked, learnerIds: [uniform.learnerId] })
    expect(v.checked).toBe(10 + 4 + 2)
    const p = posteriorOf(out)
    // Rows: the 10 honest calibration lives, the uniform life (kept on its logged genome) and the offline centre life.
    expect(p.page).toMatchObject({ eligibleLives: 12, calibrationLives: 10 })
    // The posterior counts mismatches among its rows only; diagnostics.json among every life checked.
    expect((p.diagnostics as { genomeMismatch: unknown }).genomeMismatch).toEqual({ count: 1, rate: 1 / 12, learnerIds: [uniform.learnerId] })
  })

  it("re-verification: an unreadable posterior and a mismatched drawSeed are flagged; a life of another genome schema is skipped", async () => {
    const w = world()
    const first = await build(w)
    const garbage = "not a posterior\n"
    const garbageSha = await shaOf(garbage)
    w.history.set(garbageSha, garbage)
    const unreadable = await bornLife(first[POSTERIOR], { alter: m => ({ ...m, posteriorSha: garbageSha }) })
    const seed = await bornLife(first[POSTERIOR], { alter: m => ({ ...m, drawSeed: (m.drawSeed! + 1) >>> 0 }) })
    const other = await bornLife(first[POSTERIOR], { alter: m => ({ ...m, genomeSchema: 2 }) })
    expect(collectLives(runs).map(l => l.learnerId)).toContain(other.learnerId) // it parses, and is merged data
    const v = verifyOf(await build(w))
    expect(v.ineligible).toEqual([
      { learnerId: unreadable.learnerId, reason: "posterior unreadable" },
      { learnerId: seed.learnerId, reason: `drawSeed ${seed.meta.drawSeed}, recomputed ${seed.draw.drawSeed}` },
    ])
    expect(v.checked).toBe(2) // the other-schema life is neither checked nor a row
    expect(posteriorOf(w.last!).page.eligibleLives).toBe(0)
  })

  it("a burst of merges cannot bring looks close together: each look needs the schedule's spacing since the last", async () => {
    const w = world()
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
    const w = world()
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
    expect(collectLives(runs).map(l => l.learnerId).sort()).toEqual([newbornId, legacyId, cpu.learnerId].sort())
    const out = await build(w)
    const p = posteriorOf(out)
    expect(p.page).toMatchObject({ eligibleLives: 0, calibrationLives: 0 })
    const per = (p.diagnostics as { perGenome: { rows: number }[] }).perGenome
    expect(per.every(g => g.rows === 0)).toBe(true)
    // The cpu birth is a draw, so it is re-checked (and passes); newborn and legacy lives are not draws.
    expect(verifyOf(out)).toMatchObject({ checked: 1, ineligible: [], genomeMismatch: { count: 0 } })
  })

  it("a failed calibration life is a row (ranked worst); an abandoned one is not; perGenome counts both", async () => {
    const w = world()
    const first = await build(w)
    const ended = async (end: "failed" | "orphan") => {
      const l = await bornLife(first[POSTERIOR])
      rmSync(join(runs, l.learnerId), { recursive: true })
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
    const w = world()
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
    rmSync(join(runs, fresh.learnerId), { recursive: true })
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
    const w = world()
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
    const w = world()
    await build(w)
    const c = w.previous.calibration!
    for (const bad of [{ ...c, evaluator: "search-composite@2" }, { ...c, signal: "signal-check@2" }, { ...c, genomeSchema: 2 }]) {
      await expect(build(world(), { previous: { calibration: bad, posteriorText: null } })).rejects.toThrow(/calibration/)
    }
    const tampered = { ...c, genomes: c.genomes.map((g, i) => (i === 1 ? { ...g, values: { ...g.values, lr: 1e-3 } } : g)) }
    await expect(build(world(), { previous: { calibration: tampered, posteriorText: null } })).rejects.toThrow(/genomeSha/)
    expect(() => parseCalibration("{}")).toThrow()
    expect(() => parseCalibration(JSON.stringify({ ...c, extra: 1 }))).toThrow()
  })

  it("the calibration design is kept from the previous state, not recomputed from the centre", async () => {
    const w = world()
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
    const w = world()
    const first = await build(w)
    await bornLife(first[POSTERIOR])
    await bornLife(null)
    expect([...wantedPosteriors(runs)]).toEqual([await shaOf(first[POSTERIOR])])
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
    readdirSync(join(runs, learnerId)).map(n => [`data/runs/${learnerId}/${n}`, readFileSync(join(runs, learnerId, n), "utf8")]),
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
    expect(cli()).toBe("changed=false")

    // The gate commits data/meta (Task 11); a tab is born under that posterior and its log merges.
    const meta = Object.fromEntries(["posterior.json", "calibration.json", "diagnostics.json", "posterior-index.jsonl"].map(n => [`data/meta/${n}`, inWork(`data/meta/${n}`)]))
    rmSync(join(fx.work, "data/meta"), { recursive: true })
    fx.commitMain(meta)
    const calib = await bornLife(posterior)
    expect(calib.meta.method).toBe("calibrate")
    fx.commitMain(runFiles(calib.learnerId))
    // No newborn reference in this checkout, so no row changes: the posterior keeps its bytes, the diagnostics move.
    expect(cli()).toBe("changed=true")
    expect(inWork(POSTERIOR)).toBe(posterior)
    expect(JSON.parse(inWork(DIAGNOSTICS)).verify).toMatchObject({ checked: 2, ineligible: [] }) // its posterior was found
    expect(inWork("data/meta/posterior-index.jsonl").split("\n").filter(Boolean)).toHaveLength(1)
    expect(cli()).toBe("changed=false")
  }, 60_000)
})
