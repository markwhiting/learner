// The gate's meta-learner files, data/meta/** (spec §4.3, §4.6, §7.2, §7.4; decision 0019): computed from main's
// merged logs and pinned config only, never from PR content. 05a builds the calibration design, the noise check's
// looks and the re-verification of every draw; 05b adds the GP (models, incumbent, confirmation), in mode search only.
//
// Every byte is a pure function of the input: rows in learnerId order, fixed seeds, keys sorted at every level, and no
// clock. The one time, `builtAt`, is passed in and goes only into the index line, never into a file whose sha matters.
import { createHash } from "node:crypto"
import { sobolPoints } from "@blanktab/detmath"
import { lifeBudget, type LogFile, type LogFileV2, type MetaBlock } from "@blanktab/harness/log"
import {
  birthDraw, checkGenome, fromUnit, genomeSha, parsePosterior, POSTERIOR_SCHEMA, searched, sha256Hex,
  modelPosterior, type Draw, type Genome, type GenomeSpace, type GpModel, type GpPosterior, type Incumbent, type Mode, type Posterior,
} from "@blanktab/meta"
import {
  cohortContext, getEvaluator, GPU_WIDTHS_CLASS, SIGNAL_LOOKS, type EvalOutput, type LifeInput, type References,
} from "@blanktab/evaluators"
import type { EvalConfig } from "../eval/config.ts"
import { collectLives } from "../eval/lives.ts"
import { buildModel, incumbentFor, rowSubset, ROW_METHODS, type ModelRun } from "./search.ts"
import { serialise } from "./serialise.ts"

export const POSTERIOR_PATH = "data/meta/posterior.json"
export const CALIBRATION_PATH = "data/meta/calibration.json"
export const POSTERIOR_INDEX_PATH = "data/meta/posterior-index.jsonl"
/** Counts that move with every birth (born, checked, flags, reasons): kept out of posterior.json, so its bytes, and
 * the sha every tab seeds its draw with, change only when the rows, the mode, the looks or the design change. */
export const DIAGNOSTICS_PATH = "data/meta/diagnostics.json"
/** Each life's birth verdict, recorded once (plan 05b Task 8): the history walk then reads only new lives' posteriors. */
export const VERIFIED_PATH = "data/meta/verified.json"
/**
 * The verification rule's version: a cache of another verifier is ignored, and every life is verified again. Bump it
 * with any change between the verifier markers below; meta.test.ts pins their sha256 to this number.
 */
export const VERIFIER = 2

/** The calibration design's Sobol seed (spec §4.6). */
const CALIB_SEED = "calib/1"
/** The design: the centre plus this many Sobol points. */
const CALIB_POINTS = 9
/** Repeats per design genome before look 1, and after a no-signal look 1 or 2 (spec §4.6; plan conflict 2). */
const REPEATS_FIRST = 3
const REPEATS_MORE = 6

type ShaGenome = { genomeSha: string; values: Genome }
export interface CalibrationLook { look: 1 | 2 | 3; lives: number; learnerIds: string[]; output: EvalOutput; sourceCommit: string }
export interface CalibrationState {
  schemaVersion: 1; genomeSchema: number; evaluator: string; signal: string; seed: "calib/1"
  genomes: ShaGenome[] // centre first, then 9 Sobol points decoded
  repeats: 3 | 6
  looks: CalibrationLook[] // append-only
  outcome: "pending" | "signal" | "flat"
}
/** A life's birth verdict: the posteriorSha it was verified under and that posterior's mode (null: offline, or the
 * posterior is unknown or unreadable), why it is not a draw the gate can reproduce (or null), and whether its logged
 * genome differs from the recomputed one. Only facts of the birth: nothing that depends on the calibration state. */
export interface Verdict { posteriorSha: string | null; posteriorMode: Mode | null; reason: string | null; genomeMismatch: boolean }
export interface VerifiedCache { schemaVersion: 1; verifier: 2; lives: Record<string, Verdict> }
export interface MetaInput {
  runsDir: string; config: EvalConfig; refs: References; space: GenomeSpace; centre: Genome; sourceCommit: string
  /** When this build ran (ISO 8601): written only into the index line. The caller reads the clock; buildMeta never does. */
  builtAt: string
  previous: { calibration: CalibrationState | null; posteriorText: string | null; verified: VerifiedCache | null }
  /** posterior.json versions from main's history, by sha256 of their bytes: at least every one an unverified life names. */
  posteriorsBySha: ReadonlyMap<string, string>
  /** Every centre main has had (knowledge/genome.json's history, history.ts centresBySha), by genomeSha. */
  centres: ReadonlyMap<string, Genome>
}
export interface MetaFiles {
  "data/meta/posterior.json": string; "data/meta/calibration.json": string; "data/meta/diagnostics.json": string
  "data/meta/verified.json": string
  /** One JSON line (no newline) for posterior-index.jsonl when the posterior's bytes changed, else null. */
  indexLine: string | null
}

const isV2 = (f: LogFile): f is LogFileV2 => f.schemaVersion === 2
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const byId = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

const fail = (why: string): never => { throw new Error(`${CALIBRATION_PATH}: ${why}`) }
const CAL_KEYS = ["schemaVersion", "genomeSchema", "evaluator", "signal", "seed", "genomes", "repeats", "looks", "outcome"]
const LOOK_KEYS = ["look", "lives", "learnerIds", "output", "sourceCommit"]
const exactKeys = (v: Record<string, unknown>, keys: readonly string[]) => {
  const ks = Object.keys(v)
  return ks.length === keys.length && keys.every(k => Object.hasOwn(v, k))
}

/** data/meta/calibration.json's text → its state; throws on any shape error (the gate then stops, writing nothing). */
export function parseCalibration(text: string): CalibrationState {
  const v: unknown = JSON.parse(text)
  if (!isObj(v) || !exactKeys(v, CAL_KEYS)) return fail(`must have exactly the keys ${CAL_KEYS.join(", ")}`)
  if (v.schemaVersion !== 1) fail("schemaVersion must be 1")
  if (!Number.isSafeInteger(v.genomeSchema) || (v.genomeSchema as number) < 1) fail("genomeSchema must be an integer >= 1")
  if (typeof v.evaluator !== "string" || typeof v.signal !== "string") fail("evaluator and signal must be id@version strings")
  if (v.seed !== CALIB_SEED) fail(`seed must be ${CALIB_SEED}`)
  const genomes = v.genomes
  if (!Array.isArray(genomes) || genomes.length !== 1 + CALIB_POINTS
    || !genomes.every(g => isObj(g) && exactKeys(g, ["genomeSha", "values"]) && typeof g.genomeSha === "string" && isObj(g.values))) {
    fail(`genomes must be ${1 + CALIB_POINTS} { genomeSha, values }`)
  }
  if (v.repeats !== REPEATS_FIRST && v.repeats !== REPEATS_MORE) fail(`repeats must be ${REPEATS_FIRST} or ${REPEATS_MORE}`)
  const looks = v.looks
  if (!Array.isArray(looks) || looks.length > SIGNAL_LOOKS.length) fail(`looks must be an array of at most ${SIGNAL_LOOKS.length}`)
  ;(looks as unknown[]).forEach((l, i) => {
    if (!isObj(l) || !exactKeys(l, LOOK_KEYS) || l.look !== i + 1 || !Number.isSafeInteger(l.lives) || !Array.isArray(l.learnerIds)
      || !l.learnerIds.every(x => typeof x === "string") || !isObj(l.output) || typeof l.sourceCommit !== "string") {
      fail(`look ${i + 1} must be { ${LOOK_KEYS.join(", ")} } with look ${i + 1}`)
    }
  })
  if (v.outcome !== "pending" && v.outcome !== "signal" && v.outcome !== "flat") fail("outcome must be pending, signal or flat")
  return v as unknown as CalibrationState
}

/**
 * The posterior.json versions main committed whose sha256 is in `wanted`, by sha: `git log --format=%H
 * --diff-filter=d -- <path>` (commits that delete the file have no bytes), newest first, then `git show <c>:<path>`
 * until every wanted sha is found or the history ends. A sha never found is absent (its lives are flagged unknown).
 */
export function posteriorHistory(git: (args: string[]) => string, wanted: ReadonlySet<string>): Map<string, string> {
  const out = new Map<string, string>()
  if (wanted.size === 0) return out
  const commits = git(["log", "--format=%H", "--diff-filter=d", "--", POSTERIOR_PATH]).split("\n").map(s => s.trim()).filter(Boolean)
  for (const c of commits) {
    const text = git(["show", `${c}:${POSTERIOR_PATH}`])
    const sha = createHash("sha256").update(text, "utf8").digest("hex")
    if (wanted.has(sha)) out.set(sha, text)
    if (out.size === wanted.size) break
  }
  return out
}

const HEX64 = /^[0-9a-f]{64}$/
const LEARNER_ID = /^lrn_[0-9a-f]{16}$/
const MODES: readonly unknown[] = ["calibrate", "search", "paused", "flat"]
const isVerdict = (v: unknown): v is Verdict =>
  isObj(v) && exactKeys(v, ["posteriorSha", "posteriorMode", "reason", "genomeMismatch"])
  && (v.posteriorSha === null || (typeof v.posteriorSha === "string" && HEX64.test(v.posteriorSha)))
  && (v.posteriorMode === null || MODES.includes(v.posteriorMode)) && !(v.posteriorSha === null && v.posteriorMode !== null)
  && (v.reason === null || typeof v.reason === "string") && typeof v.genomeMismatch === "boolean" && !(v.reason !== null && v.genomeMismatch)

/**
 * data/meta/verified.json's text → the cache, or null on any shape error (the gate then verifies every life again: the
 * cache only saves work, it never decides). A cache of another verifier is null too.
 */
export function parseVerified(text: string): VerifiedCache | null {
  let v: unknown
  try { v = JSON.parse(text) } catch { return null }
  if (!isObj(v) || !exactKeys(v, ["schemaVersion", "verifier", "lives"]) || v.schemaVersion !== 1 || v.verifier !== VERIFIER) return null
  const lives = v.lives
  if (!isObj(lives) || !Object.entries(lives).every(([id, x]) => LEARNER_ID.test(id) && isVerdict(x))) return null
  return v as unknown as VerifiedCache
}

/** The cache's verdict on a life born under `posteriorSha`, or null when it has none (or is of another verifier). */
function cachedVerdict(cache: VerifiedCache | null, learnerId: string, posteriorSha: string | null): Verdict | null {
  if (cache === null || cache.verifier !== VERIFIER || !Object.hasOwn(cache.lives, learnerId)) return null
  const v = cache.lives[learnerId]!
  return v.posteriorSha === posteriorSha ? v : null
}

/** The lives buildMeta verifies: role search (the draws) and confirm (the confirmation draws), of the space's schema. */
const isChecked = (meta: MetaBlock, schemaVersion: number) => (meta.role === "search" || meta.role === "confirm") && meta.genomeSchema === schemaVersion

/**
 * The posteriorShas that the lives under `runsDir` buildMeta verifies (isChecked, for the space's `schemaVersion`) and
 * that are not yet in `verified` were born with (their first meta block's), for posteriorHistory: a cached life's
 * posterior is never read again, so the history walk stops at the recent posteriors.
 */
export function wantedPosteriors(runsDir: string, schemaVersion: number, verified: VerifiedCache | null = null): Set<string> {
  const out = new Set<string>()
  for (const life of collectLives(runsDir)) {
    const birth = birthOf(life.files)
    if (!birth || !isChecked(birth.meta, schemaVersion)) continue
    const sha = birth.meta.posteriorSha
    if (typeof sha === "string" && cachedVerdict(verified, life.learnerId, sha) === null) out.add(sha)
  }
  return out
}

/** The design (spec §4.6): the centre, then fromUnit of 9 Sobol points seeded "calib/1", each with its genomeSha. */
async function design(space: GenomeSpace, centre: Genome): Promise<ShaGenome[]> {
  const genomes = [centre, ...sobolPoints(searched(space).length, CALIB_POINTS, CALIB_SEED).map(u => fromUnit(space, u, centre))]
  return Promise.all(genomes.map(async values => ({ genomeSha: await genomeSha(space.schemaVersion, values), values: { ...values } })))
}

/** The previous calibration state, checked against the pins and the space it must have been built for. */
async function keptCalibration(c: CalibrationState, config: EvalConfig, space: GenomeSpace): Promise<CalibrationState> {
  // Rows never mix evaluators or genome schemas (spec §2.2, §4.6): a change of pin or schema restarts calibration
  // under a recorded decision, which removes this file. The gate refuses rather than restart on its own.
  if (c.evaluator !== config.search) fail(`was built for search evaluator ${c.evaluator}, but ${config.search} is pinned (a recorded decision restarts calibration)`)
  if (c.signal !== config.signal) fail(`was built for signal check ${c.signal}, but ${config.signal} is pinned (a recorded decision restarts calibration)`)
  if (c.genomeSchema !== space.schemaVersion) fail(`was built for genome schema ${c.genomeSchema}, but the space is ${space.schemaVersion} (a recorded decision restarts calibration)`)
  for (const [i, g] of c.genomes.entries()) {
    const bad = checkGenome(space, g.values)
    if (bad !== null) fail(`genome ${i} is not in the space: ${bad}`)
    if (g.genomeSha !== await genomeSha(space.schemaVersion, g.values)) fail(`genome ${i}'s genomeSha does not match its values`)
  }
  checkOutcome(c, config)
  return c
}

/**
 * The outcome must be the one the looks give (Review Focus 2): a hand-edited "signal" cannot switch the search on.
 * Every look is the pinned signal check's eligible output (nextLook records no other). "signal": the last look scored 1
 * and every earlier one 0. "flat": all SIGNAL_LOOKS looks, all 0. "pending": fewer looks, all 0.
 */
function checkOutcome(c: CalibrationState, config: EvalConfig): void {
  const at = config.signal.lastIndexOf("@")
  const id = config.signal.slice(0, at), version = Number(config.signal.slice(at + 1))
  for (const l of c.looks) {
    const o = l.output
    if (o.evaluator !== id || o.version !== version) fail(`look ${l.look} is not an output of ${config.signal}`)
    if (o.eligible !== true || (o.score !== 0 && o.score !== 1)) fail(`look ${l.look} must be an eligible output scored 0 or 1`)
  }
  const n = c.looks.length
  const zeroBefore = (k: number) => c.looks.slice(0, k).every(l => l.output.score === 0)
  if (c.outcome === "signal" && !(n > 0 && c.looks[n - 1]!.output.score === 1 && zeroBefore(n - 1)))
    fail("outcome signal needs a last look scored 1 after looks scored 0")
  if (c.outcome === "flat" && !(n === SIGNAL_LOOKS.length && zeroBefore(n))) fail(`outcome flat needs ${SIGNAL_LOOKS.length} looks, all scored 0`)
  if (c.outcome === "pending" && !(n < SIGNAL_LOOKS.length && zeroBefore(n))) fail(`outcome pending needs fewer than ${SIGNAL_LOOKS.length} looks, all scored 0`)
}

/** The first meta block a life logged (its birth), and the file it is in. */
function birthOf(files: readonly LogFile[]): { file: LogFileV2; meta: MetaBlock } | null {
  for (const f of files) if (isV2(f) && f.meta) return { file: f, meta: f.meta }
  return null
}
const widthsClassOf = (w: LogFileV2["widths"]): "gpu" | "cpu" =>
  w.conv1 === GPU_WIDTHS_CLASS.conv1 && w.conv2 === GPU_WIDTHS_CLASS.conv2 && w.enc === GPU_WIDTHS_CLASS.enc && w.hidden === GPU_WIDTHS_CLASS.hidden ? "gpu" : "cpu"

const sameGenome = (a: Genome, b: Genome): boolean => {
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort()
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && a[k] === b[k])
}

interface Checked { posteriorMode: Mode | null; reason: string | null; genomeMismatch: boolean }

// ---- verifier: begin (VERIFIER pins the sha256 of the text from here to "verifier: end") ----

/** Methods whose genome is a copy, never arithmetic: a recomputed genome that differs is no draw of the gate's. */
const EXACT_GENOME: ReadonlySet<string> = new Set(["centre", "confirm"])

/** The first field of the draw a life logged that its recomputed draw does not reproduce, or null. */
function fieldMismatch(meta: MetaBlock, d: Draw): string | null {
  if (meta.method !== d.method) return `method ${meta.method}, recomputed ${d.method}`
  if (meta.drawSeed !== d.drawSeed) return `drawSeed ${meta.drawSeed}, recomputed ${d.drawSeed}`
  if (meta.evalPanel !== d.evalPanel) return `evalPanel ${meta.evalPanel}, recomputed ${d.evalPanel}`
  if (meta.evalSeed !== d.evalSeed) return `evalSeed ${meta.evalSeed}, recomputed ${d.evalSeed}`
  const c = meta.confirmation, r = d.confirmation
  if ((c === null) !== (r === null) || (c !== null && r !== null && (c.id !== r.id || c.arm !== r.arm)))
    return `confirmation ${JSON.stringify(c)}, recomputed ${JSON.stringify(r)}`
  return null
}

/**
 * Re-verification (spec §7.4): the life's draw recomputed from its learnerId, the posterior bytes its posteriorSha
 * names, the space, its widths class and the centre in force at its birth. A row is lost for an unknown or unreadable
 * posterior, an unknown centre, a mismatched method, seed, panel, evalSeed or confirmation, a logged genome outside
 * the space or not matching its genomeSha (readLives adds, every build, a calibrate genome outside the design: the
 * design is the calibration state's, not the birth's, so it is never cached). A genome that differs from the
 * recomputed one alone keeps the row, on its logged genome, except for centre and confirm draws, whose genome the gate
 * fixed (spec §4.5 "Integrity").
 *
 * The centre in force (05a Task 10 note; Review Focus 3): the tab drew with its own knowledge/genome.json, which a
 * promotion may have moved away from main's. Two centres are tried, the first exact reproduction winning: the
 * posterior's centre (the one it was built for), then, for a centre draw only, the logged genome itself when it is a
 * centre main has had (`centres`), which reproduces the fallback "posterior built for another centre".
 */
async function verify(learnerId: string, meta: MetaBlock, widthsClass: "gpu" | "cpu", ctx: {
  space: GenomeSpace; centres: ReadonlyMap<string, Genome>; posteriors: ReadonlyMap<string, Posterior | null>
  /** Per posteriorSha, its model conditioned once (the ts branch's cost), shared by every life born under it. */
  conditioned: (sha: string, p: Posterior) => () => GpPosterior | null
}): Promise<Checked> {
  let posterior: Posterior | null = null
  let posteriorMode: Mode | null = null
  const no = (reason: string): Checked => ({ posteriorMode, reason, genomeMismatch: false })
  if (meta.posteriorSha !== null) {
    if (!ctx.posteriors.has(meta.posteriorSha)) return no("unknown posteriorSha")
    posterior = ctx.posteriors.get(meta.posteriorSha)!
    if (posterior === null) return no("posterior unreadable") // a tab that cannot parse it logs no sha (draws centre)
    posteriorMode = posterior.mode
  }
  // A centre draw runs the tab's centre: one main never had is no draw of the gate's.
  if (meta.method === "centre" && !ctx.centres.has(meta.genomeSha)) return no("unknown centre")
  const candidates: { values: Genome; sha: string }[] = []
  if (posterior !== null) candidates.push({ values: posterior.centre.values, sha: posterior.centre.genomeSha })
  if (meta.method === "centre" && meta.genomeSha !== posterior?.centre.genomeSha) candidates.push({ values: ctx.centres.get(meta.genomeSha)!, sha: meta.genomeSha })
  const tries = candidates.map(c => {
    const conditioned = posterior !== null && meta.posteriorSha !== null ? ctx.conditioned(meta.posteriorSha, posterior) : undefined
    const d = birthDraw({ learnerId, space: ctx.space, centre: c.values, centreSha: c.sha, widthsClass, posterior, posteriorSha: meta.posteriorSha, conditioned })
    return { d, why: fieldMismatch(meta, d) }
  })
  const exact = tries.find(t => t.why === null && sameGenome(meta.genome, t.d.genome))
  const loose = tries.find(t => t.why === null)
  if (!exact) {
    if (!loose) return no(tries[0]?.why ?? `method ${meta.method}, recomputed centre`)
    if (EXACT_GENOME.has(meta.method)) {
      return no(meta.method === "confirm"
        ? `genome is not confirmation ${meta.confirmation?.id}'s ${meta.confirmation?.arm === "centre" ? "centre" : "candidate"}`
        : "genome is not the centre in force")
    }
  }
  const bad = checkGenome(ctx.space, meta.genome)
  if (bad !== null) return no(`genome not in the space: ${bad}`)
  if (meta.genomeSha !== await genomeSha(meta.genomeSchema, meta.genome)) return no("genomeSha does not match the genome")
  return { posteriorMode, reason: null, genomeMismatch: !exact }
}
// ---- verifier: end ----

/** The lives' rows and counts under the pinned search evaluator, after re-verification. */
interface Lives {
  /** searchEra: born under a posterior of mode search (the rows rowsSinceDeclaration counts). */
  rows: { life: LifeInput; meta: MetaBlock; out: EvalOutput; searchEra: boolean }[]
  /** Verified confirmation lives (role confirm): never rows (spec §4.3); Task 9 groups them by arm. */
  confirmLives: { life: LifeInput; meta: MetaBlock }[]
  checked: number; flagged: { learnerId: string; reason: string }[]
  /** Verified lives whose logged genome differs from the recomputed one, and those of them that are rows. */
  mismatched: string[]; mismatchedRows: string[]
  perGenome: { genomeSha: string; born: number; rows: number; failed: number; abandoned: number; failedRows: number }[]
  unscored: Record<string, number>
  /** Every checked life's verdict: the next verified.json. */
  verdicts: Record<string, Verdict>
}

async function readLives(input: MetaInput, calibration: CalibrationState): Promise<Lives> {
  const { space, refs, config } = input
  const search = getEvaluator(config.search)
  if (search.kind !== "life" || !search.evaluateLife) throw new Error(`${config.search} is not a life evaluator`)
  // The posteriors by sha, each parsed once. The map's own keys are checked: a wrong key would verify wrong draws.
  const posteriors = new Map<string, Posterior | null>()
  for (const [sha, text] of input.posteriorsBySha) {
    if (await sha256Hex(new TextEncoder().encode(text)) !== sha) throw new Error(`posteriorsBySha: the key ${sha} is not the sha256 of its bytes`)
    posteriors.set(sha, parsePosterior(text))
  }
  // `centres` holds every schema's centres, each keyed by its own schema's sha. A wrong entry cannot pass a life: a
  // centre draw must reproduce the logged genome exactly, and the logged genome its own genomeSha.
  const designShas = calibration.genomes.map(g => g.genomeSha)
  const design = new Set(designShas)
  const memo = new Map<string, GpPosterior | null>()
  const conditioned = (sha: string, p: Posterior) => () => {
    if (!memo.has(sha)) memo.set(sha, p.models.gpu ? modelPosterior(p.models.gpu) : null)
    return memo.get(sha)!
  }
  const ctx = { space, centres: input.centres, posteriors, conditioned }
  const cache = input.previous.verified
  const out: Lives = {
    rows: [], confirmLives: [], checked: 0, flagged: [], mismatched: [], mismatchedRows: [], unscored: {}, verdicts: {},
    perGenome: designShas.map(genomeSha => ({ genomeSha, born: 0, rows: 0, failed: 0, abandoned: 0, failedRows: 0 })),
  }
  for (const life of collectLives(input.runsDir)) { // sorted by learnerId
    const birth = birthOf(life.files)
    // Search and confirmation draws are verified; newborn and legacy lives are not draws and never rows. Other genome
    // schemas cannot be recomputed against this space, and are never rows of this fit.
    if (!birth || !isChecked(birth.meta, space.schemaVersion)) continue
    const widthsClass = widthsClassOf(birth.file.widths)
    // A life's birth verdict is computed once (verified.json); scoring and rows below are recomputed every build.
    const checked: Checked = cachedVerdict(cache, life.learnerId, birth.meta.posteriorSha) ?? await verify(life.learnerId, birth.meta, widthsClass, ctx)
    out.verdicts[life.learnerId] = { posteriorSha: birth.meta.posteriorSha, ...checked }
    out.checked++
    // The design is the calibration state's, not the birth's: checked every build, after the cache, so a calibration
    // restart with another design flags the same lives with or without the cache.
    const reason = checked.reason ?? (birth.meta.method === "calibrate" && !design.has(birth.meta.genomeSha) ? "calibrate genome not in the calibration design" : null)
    if (reason !== null) { out.flagged.push({ learnerId: life.learnerId, reason }); continue }
    if (checked.genomeMismatch) out.mismatched.push(life.learnerId)
    // Confirmation lives are never rows, and search rows never confirmation lives (spec §4.3, §4.5).
    if (birth.meta.role === "confirm") { out.confirmLives.push({ life, meta: birth.meta }); continue }

    const scored: EvalOutput = search.evaluateLife(life, refs)
    const isRow = scored.eligible && typeof scored.score === "number" && widthsClass === "gpu"
      && (ROW_METHODS as readonly string[]).includes(birth.meta.method) && birth.meta.inheritance.filters === "none"
      && life.files.every(f => !isV2(f) || !f.meta || f.meta.genomeSchema === space.schemaVersion)
    if (isRow) {
      out.rows.push({ life, meta: birth.meta, out: scored, searchEra: checked.posteriorMode === "search" })
      if (checked.genomeMismatch) out.mismatchedRows.push(life.learnerId)
    } else { const why = scored.eligible ? "not a row of this fit" : scored.reason ?? "ineligible"; out.unscored[why] = (out.unscored[why] ?? 0) + 1 }

    const g = birth.meta.method === "calibrate" ? designShas.indexOf(birth.meta.genomeSha) : -1
    if (g < 0) continue
    // Failed: its final file says so. Abandoned: it ended, not failed, without its budget checkup (tab closed, orphan).
    const last = life.files[life.files.length - 1]!
    const end = isV2(last) ? last.meta?.end : undefined
    const budgetCU = lifeBudget(birth.meta).budgetCU
    const reachedBudget = life.files.some(f => isV2(f) && (f.checkups ?? []).some(c => c.level === budgetCU))
    const pg = out.perGenome[g]!
    pg.born++
    if (isRow) pg.rows++
    if (end === "failed") { pg.failed++; if (isRow) pg.failedRows++ }
    else if (last.final && !reachedBudget) pg.abandoned++
  }
  return out
}

/** Calibration rows (method calibrate) grouped by their design genome, for the signal check. */
function calibrationGroups(rows: Lives["rows"]): Record<string, LifeInput[]> {
  const groups: Record<string, LifeInput[]> = {}
  for (const r of rows) if (r.meta.method === "calibrate") (groups[r.meta.genomeSha] ??= []).push(r.life)
  return groups
}

/** The next look (spec §4.6), or none: at most one per build, once enough new calibration rows have merged. */
function nextLook(c: CalibrationState, rows: Lives["rows"], input: MetaInput): CalibrationState {
  const n = c.looks.length
  const calib = rows.filter(r => r.meta.method === "calibrate")
  if (c.outcome !== "pending" || n >= SIGNAL_LOOKS.length) return c
  // Each look needs its scheduled size, and as many new lives since the last look as the schedule spaces them, so a
  // burst of merges cannot bring two looks close together (controller ruling).
  const lastLives = n > 0 ? c.looks[n - 1]!.lives : 0
  const spacing = SIGNAL_LOOKS[n]! - (n > 0 ? SIGNAL_LOOKS[n - 1]! : 0)
  if (calib.length < Math.max(SIGNAL_LOOKS[n]!, lastLives + spacing)) return c
  const signal = getEvaluator(input.config.signal)
  if (signal.kind !== "cohort" || !signal.evaluateCohort) throw new Error(`${input.config.signal} is not a cohort evaluator`)
  const output = signal.evaluateCohort(calibrationGroups(rows), input.refs, cohortContext(input.config))
  if (!output.eligible) return c // not enough to judge yet: wait for more rows
  const look: CalibrationLook = {
    look: (n + 1) as 1 | 2 | 3, lives: calib.length, learnerIds: calib.map(r => r.life.learnerId).sort(byId), output, sourceCommit: input.sourceCommit,
  }
  const looks = [...c.looks, look]
  if (output.score === 1) return { ...c, looks, outcome: "signal" }
  if (looks.length === SIGNAL_LOOKS.length) return { ...c, looks, outcome: "flat" }
  return { ...c, looks, repeats: REPEATS_MORE } // paused, on the page; births keep calibrating (plan conflict 2)
}

const mismatch = (learnerIds: string[], of: number) => ({ count: learnerIds.length, rate: of > 0 ? learnerIds.length / of : 0, learnerIds })

const modeOf = (c: CalibrationState): Mode => (c.outcome === "signal" ? "search" : c.outcome === "flat" ? "flat" : "calibrate")

/** The search's state (spec §4.3–§4.4), built only in mode search. */
interface SearchState { model: GpModel | null; reason: string | null; run: ModelRun; incumbent: Incumbent | null; rowsSinceDeclaration: number }

/**
 * The model on the rows and the incumbent. y is the search evaluator's targets over every row, then rowSubset keeps at
 * most MAX_ROWS of them. The previous posterior's model is reused when its data are the same (search.ts buildModel).
 */
async function searchState(input: MetaInput, lives: Lives): Promise<SearchState> {
  const search = getEvaluator(input.config.search)
  if (!search.targets) throw new Error(`${input.config.search} has no targets transform`)
  const y = search.targets(lives.rows.map(r => r.out))
  const keep = new Set(rowSubset(lives.rows.map(r => r.life.learnerId)))
  const kept = lives.rows.flatMap((r, i) => (keep.has(r.life.learnerId) ? [{ learnerId: r.life.learnerId, genome: r.meta.genome, y: y[i]! }] : []))
  const prev = input.previous.posteriorText === null ? null : parsePosterior(input.previous.posteriorText)
  const built = buildModel(kept.map(({ learnerId, genome }) => ({ learnerId, genome })), kept.map(k => k.y), input.space, prev?.models.gpu ?? null)
  const incumbent = built.model === null ? null : await incumbentFor(built.model, input.space, input.centre)
  // Rows born under a search posterior: none before the signal look (plan conflict 5; controller ruling, Task 8
  // review). Task 9 counts from the last declaration's rowsAtDeclaration.
  return { ...built, incumbent, rowsSinceDeclaration: lives.rows.filter(r => r.searchEra).length }
}

/**
 * data/meta/posterior.json, calibration.json, diagnostics.json and verified.json from main's merged logs, and the
 * index line when the posterior changed. The posterior holds only what follows from the rows (and the mode, looks and
 * design). The model, the incumbent and the confirmation exist only in mode search: in every other mode no fit runs,
 * and the posterior is 05a's with schemaVersion 2 and page.rowsSinceDeclaration 0.
 */
export async function buildMeta(input: MetaInput): Promise<MetaFiles> {
  const { space, centre, config } = input
  const centreSha = await genomeSha(space.schemaVersion, centre)
  const prevCal = input.previous.calibration
  let calibration: CalibrationState = prevCal
    ? await keptCalibration(prevCal, config, space)
    : { schemaVersion: 1, genomeSchema: space.schemaVersion, evaluator: config.search, signal: config.signal, seed: CALIB_SEED, genomes: await design(space, centre), repeats: REPEATS_FIRST, looks: [], outcome: "pending" }

  const lives = await readLives(input, calibration)
  calibration = nextLook(calibration, lives.rows, input)
  const mode = modeOf(calibration)
  // The mode guard (the user's "inert until signal"): nothing of the search is computed in any other mode.
  const s = mode === "search" ? await searchState(input, lives) : null
  const calibRows = lives.rows.filter(r => r.meta.method === "calibrate").length
  const n = calibration.looks.length
  const posterior = {
    schemaVersion: POSTERIOR_SCHEMA, genomeSchema: space.schemaVersion, evaluator: config.search, sourceCommit: input.sourceCommit, mode,
    centre: { genomeSha: centreSha, values: centre },
    calibration: { genomes: calibration.genomes, need: lives.perGenome.map(g => Math.max(0, calibration.repeats - g.rows)) },
    models: s?.model ? { gpu: s.model } : {}, incumbent: s?.incumbent ?? null, confirmation: null,
    diagnostics: {
      perGenome: lives.perGenome.map(g => ({ genomeSha: g.genomeSha, rows: g.rows, failed: g.failedRows })),
      // Rows kept on a logged genome that differs from the recomputed draw (spec §7.4).
      genomeMismatch: mismatch(lives.mismatchedRows, lives.rows.length),
      // Why a search posterior serves no model (its tabs then draw centre).
      ...(s && s.reason !== null ? { model: { reason: s.reason } } : {}),
    },
    page: {
      eligibleLives: lives.rows.length, lastPromotion: null, calibrationLives: calibRows,
      // The next look's size while calibrating; the deciding look's once decided.
      calibrationTarget: SIGNAL_LOOKS[calibration.outcome === "pending" ? Math.min(n, SIGNAL_LOOKS.length - 1) : n - 1]!, looks: n,
      rowsSinceDeclaration: s?.rowsSinceDeclaration ?? 0,
    },
  }
  let posteriorText = serialise(posterior)
  if (parsePosterior(posteriorText) === null) throw new Error("buildMeta produced a posterior that parsePosterior refuses")
  // Unchanged rows keep the previous bytes, and their sourceCommit: those rows came from that commit too.
  const prevText = input.previous.posteriorText
  let changed = true
  if (prevText !== null) {
    let prevCommit: unknown
    try { prevCommit = (JSON.parse(prevText) as { sourceCommit?: unknown }).sourceCommit } catch { prevCommit = undefined }
    if (typeof prevCommit === "string" && serialise({ ...posterior, sourceCommit: prevCommit }) === prevText) {
      posteriorText = prevText
      changed = false
    }
  }
  const indexLine = changed
    ? JSON.stringify({ posteriorSha: await sha256Hex(new TextEncoder().encode(posteriorText)), sourceCommit: input.sourceCommit, builtAt: input.builtAt })
    : null
  const diagnostics = {
    schemaVersion: 1, evaluator: config.search,
    perGenome: lives.perGenome.map(g => ({ genomeSha: g.genomeSha, born: g.born, rows: g.rows, failed: g.failed, abandoned: g.abandoned })),
    verify: { checked: lives.checked, ineligible: lives.flagged, genomeMismatch: mismatch(lives.mismatched, lives.checked) },
    unscored: lives.unscored,
    // What the fit did this build (search only): it moves with every build, so it stays out of posterior.json.
    ...(s ? { model: { ...s.run, reason: s.reason } } : {}),
  }
  const verified: VerifiedCache = { schemaVersion: 1, verifier: VERIFIER, lives: lives.verdicts }
  return {
    [POSTERIOR_PATH]: posteriorText, [CALIBRATION_PATH]: serialise(calibration), [DIAGNOSTICS_PATH]: serialise(diagnostics),
    [VERIFIED_PATH]: serialise(verified), indexLine,
  } as MetaFiles
}
