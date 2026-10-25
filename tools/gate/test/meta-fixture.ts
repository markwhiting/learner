// Lives and gate runs for the meta tests (meta.test.ts, search.test.ts; plan 05b Tasks 8, 9 and 12): lives born as a
// tab would birth them (birthDraw under a posterior's bytes), written to a temp runs dir, and buildMeta runs on main's
// state between builds. Search mode is reached only through data: planted calibration lives through the real signal
// check (ruling 1), never a code switch.
import { afterEach, beforeEach } from "vitest"
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { createRng } from "@blanktab/detmath"
import type { LogFile, LogFileV2, MetaBlock } from "@blanktab/harness/log"
import { birthDraw, genomeSha, parseCentre, parseGenomeSpace, parsePosterior, sha256Hex, type Genome } from "@blanktab/meta"
import { parseNewbornReference, type NewbornReference } from "@blanktab/evaluators"
import { lid, scoredLife, typicalLife } from "../../../packages/evaluators/test/fixtures/build.ts"
import { loadEvalConfig } from "../../eval/config.ts"
import { buildMeta, parseCalibration, parseVerified, type MetaFiles, type MetaInput } from "../meta.ts"

export const REPO = fileURLToPath(new URL("../../..", import.meta.url))
export const SPACE = parseGenomeSpace(JSON.parse(readFileSync(join(REPO, "knowledge/genome-space.json"), "utf8")))
export const CENTRE = parseCentre(JSON.parse(readFileSync(join(REPO, "knowledge/genome.json"), "utf8")), SPACE)
export const CONFIG = loadEvalConfig(REPO)
const REF = parseNewbornReference(JSON.parse(readFileSync(join(REPO, "packages/evaluators/test/fixtures/reference.json"), "utf8"))) as NewbornReference
export const REFS = { newborn: REF }
export const commit = (n: number) => n.toString(16).padStart(40, "0")
export const BUILT_AT = "2026-10-23T12:00:00.000Z"
export const POSTERIOR = "data/meta/posterior.json"
export const CALIBRATION = "data/meta/calibration.json"
export const DIAGNOSTICS = "data/meta/diagnostics.json"
export const VERIFIED = "data/meta/verified.json"

export const shaOf = (text: string) => sha256Hex(new TextEncoder().encode(text))
/** genomeSha under the space's schema. */
export const shaOfGenome = (g: Genome) => genomeSha(SPACE.schemaVersion, g)

let runs = ""
/** A fresh runs dir for each test of the calling file (call once, at its top level). */
export function useTempRuns(): void {
  beforeEach(() => { runs = mkdtempSync(join(tmpdir(), "meta-")) })
  afterEach(() => { rmSync(runs, { recursive: true, force: true }) })
}
/** The current test's runs dir. */
export const runsDir = () => runs
export const put = (f: LogFile) => {
  mkdirSync(join(runs, f.learnerId), { recursive: true })
  writeFileSync(join(runs, f.learnerId, `${f.seq}.json`), JSON.stringify(f, null, 2))
}

/** The gate's state between builds, as main's data/meta and its history would hold it. */
export interface World { previous: MetaInput["previous"]; history: Map<string, string>; centres: Map<string, Genome>; last: MetaFiles | null; built: number }
export const world = async (): Promise<World> => ({
  previous: { calibration: null, posteriorText: null, verified: null }, history: new Map(),
  centres: new Map([[await shaOfGenome(CENTRE), CENTRE]]), last: null, built: 0,
})

/** One gate run on `w`'s state at a new main commit; the result becomes main's data/meta. */
export async function build(w: World, over: Partial<MetaInput> = {}): Promise<MetaFiles> {
  const out = await buildMeta({
    runsDir: runs, config: CONFIG, refs: REFS, space: SPACE, centre: CENTRE, sourceCommit: commit(++w.built), builtAt: BUILT_AT,
    previous: w.previous, posteriorsBySha: w.history, centres: w.centres, ...over,
  })
  w.previous = { calibration: parseCalibration(out[CALIBRATION]), posteriorText: out[POSTERIOR], verified: parseVerified(out[VERIFIED]) }
  w.history.set(await shaOf(out[POSTERIOR]), out[POSTERIOR])
  w.last = out
  return out
}

/** How a test life scores: its M2 measure and held-out ratio. */
export type Effect = (k: number, calibrationIndex: number | undefined) => { m2: number; ratio: number }
/** A genome effect far above the repeat noise (as signal-check's own test plants it). */
export const planted: Effect = (k, i) => ({ m2: 0.3 + 0.4 * (i ?? 0) + 0.01 * (k % 3), ratio: 0.98 - 0.01 * (i ?? 0) + 0.001 * (k % 3) })
/** Scores that depend on the learner alone, never on its genome. */
export const noise: Effect = k => {
  const r = createRng(0x5eed + k)
  return { m2: 0.3 + 2 * r(), ratio: 0.9 + 0.08 * r() }
}

let nextK = 1
/** Number the next lives from `k` (bornLife's default k counts up across a file): a golden then needs no test order. */
export const resetK = (k: number) => { nextK = k }
export interface Born { learnerId: string; draw: ReturnType<typeof birthDraw>; meta: MetaBlock; files: LogFileV2[] }
/**
 * A life born as a tab would birth it under the posterior bytes `text`, on GPU widths unless `widths` says otherwise,
 * with the tab's own centre `centre` (default: main's): birthDraw's genome, method, seeds, panel and confirmation,
 * logged in its meta block. `alter` edits the block it logs.
 */
export async function bornLife(text: string | null, opts: {
  effect?: Effect; widths?: LogFileV2["widths"]; alter?: (m: MetaBlock) => MetaBlock | Promise<MetaBlock>; k?: number; centre?: Genome
} = {}): Promise<Born> {
  const k = opts.k ?? nextK++
  const learnerId = lid(0x10000 + k)
  const posterior = text === null ? null : parsePosterior(text)
  const posteriorSha = text === null ? null : await shaOf(text)
  const centre = opts.centre ?? CENTRE
  const d = birthDraw({ learnerId, space: SPACE, centre, centreSha: await shaOfGenome(centre), widthsClass: opts.widths ? "cpu" : "gpu", posterior, posteriorSha })
  let meta: MetaBlock = {
    v: 1, role: d.role, method: d.method, genomeSchema: SPACE.schemaVersion, genome: d.genome, genomeSha: await shaOfGenome(d.genome),
    drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, evalSeed: d.evalSeed, confirmation: d.confirmation,
    inheritance: { filters: "none" },
  }
  if (opts.alter) meta = await opts.alter(meta)
  const e = (opts.effect ?? noise)(k, d.calibrationIndex)
  const files = opts.widths
    ? typicalLife(learnerId, { meta, widths: opts.widths })
    : scoredLife(learnerId, { ms: { M1: 0.9, M2: e.m2, M3: 0.2, M4: 0.3, M5: 0.8, M6: 0 }, ratio: e.ratio, meta })
  for (const f of files) put(f)
  return { learnerId, draw: d, meta, files }
}
/** `n` more lives under the current posterior. */
export async function grow(w: World, n: number, effect: Effect): Promise<Born[]> {
  const out: Born[] = []
  for (let i = 0; i < n; i++) out.push(await bornLife(w.previous.posteriorText, { effect }))
  return out
}
/** A world in mode search, reached through data: 30 planted calibration lives, and the build whose look 1 finds signal. */
export async function searchWorld(): Promise<{ w: World; signal: MetaFiles }> {
  const w = await world()
  await build(w)
  await grow(w, 30, planted)
  const signal = await build(w)
  return { w, signal }
}
