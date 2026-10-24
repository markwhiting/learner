// The posterior file, data/meta/posterior.json (spec §7.2): what the gate tells every tab about the search. A tab that
// cannot read the file draws the centre (spec §4.1 step 1), so parsing never throws: any shape error is null.
//
// Two schema versions are read. schemaVersion 1 is 05a's file, checked exactly as 05a checks it; it carries no model,
// so it reads as `models: {}`, `incumbent: null`, `confirmation: null` and `page.rowsSinceDeclaration: 0`, and draws
// exactly as under 05a. schemaVersion 2 (05b) is strict everywhere, and carries the mode guard: a model, an
// incumbent or a confirmation outside mode "search" makes the whole file unreadable (Global Constraints, plan 05b).
// 05a code refuses every v2 file (its check is `schemaVersion !== 1`), so an old tab draws centre until it reloads.
import { MAX_GENES, SETTING_KEY_RE, type Genome } from "./genome"

/** The schemaVersion the 05b gate writes. */
export const POSTERIOR_SCHEMA = 2
/** The most rows a model may carry (spec §4.3: above 1,000 the gate keeps a deterministic subset). */
const MAX_ROWS = 1000
/** The GP's noise floor (spec §4.3). */
const NOISE_FLOOR = 1e-3
/** The fit's starts: the prior mode and 3 Sobol points (spec §4.3). */
const FIT_STARTS = 4

export type Mode = "calibrate" | "search" | "paused" | "flat"
export interface GpHyper { lengthscales: number[]; outputscale: number; noise: number; mean: number }
export interface GpModel {
  /** searched(space) names, in file order: the cube's axes. */
  genes: string[]
  /** x rounded to 6 decimals, in [0, 1]; learnerIds strictly increasing. */
  rows: { learnerId: string; x: number[] }[]
  /** The search evaluator's targets, in rows' order. */
  y: number[]
  hyper: GpHyper
  fit: { objective: number; start: number; iterations: number[] }
}
export interface Incumbent { x: number[]; genomeSha: string; mean: number; pBeatsCentre: number }
export type ArmName = "candidate" | "centre" | "newborn"
export interface ServedConfirmation {
  /** seedBase = 10_000 + 1_000·id (spec §4.5). */
  id: number; seedBase: number
  candidate: { genomeSha: string; values: Genome }; centre: { genomeSha: string; values: Genome }
  target: Record<ArmName, number>; counted: Record<ArmName, number>; need: Record<ArmName, number>
}
export interface PosteriorPage {
  eligibleLives: number; lastPromotion: string | null; calibrationLives: number; calibrationTarget: number; looks: number
  /** v2 only; a v1 file reads as 0. */
  rowsSinceDeclaration: number
}
export interface Posterior {
  schemaVersion: 1 | 2; genomeSchema: number; evaluator: string; sourceCommit: string; mode: Mode
  centre: { genomeSha: string; values: Genome }
  calibration: { genomes: { genomeSha: string; values: Genome }[]; need: number[] }
  models: { gpu?: GpModel }; incumbent: Incumbent | null; confirmation: ServedConfirmation | null
  diagnostics: Record<string, unknown>; page: PosteriorPage
}

const MODES: readonly Mode[] = ["calibrate", "search", "paused", "flat"]
const ARMS: readonly ArmName[] = ["candidate", "centre", "newborn"]
const TOP_KEYS = ["schemaVersion", "genomeSchema", "evaluator", "sourceCommit", "mode", "centre", "calibration", "models",
  "incumbent", "confirmation", "diagnostics", "page"]
const PAGE_KEYS_V1 = ["eligibleLives", "lastPromotion", "calibrationLives", "calibrationTarget", "looks"]
const PAGE_KEYS_V2 = [...PAGE_KEYS_V1, "rowsSinceDeclaration"]
const HEX64 = /^[0-9a-f]{64}$/
const COMMIT = /^([0-9a-f]{40}|[0-9a-f]{64})$/ // a sha-1 or sha-256 git object id
const LEARNER_ID = /^lrn_[0-9a-f]{16}$/

type Obj = Record<string, unknown>
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v)
const isCount = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0
const isReal = (v: unknown): v is number => Number.isFinite(v)
const isUnit = (v: unknown): v is number => isReal(v) && v >= 0 && v <= 1
const isPositive = (v: unknown): v is number => isReal(v) && v > 0
const exactKeys = (v: Obj, keys: readonly string[]) => {
  const ks = Object.keys(v)
  return ks.length === keys.length && keys.every(k => Object.hasOwn(v, k))
}
const isArrayOf = <T>(v: unknown, length: number, item: (x: unknown) => x is T): v is T[] =>
  Array.isArray(v) && v.length === length && v.every(item)

/** Genome-shaped: at most MAX_GENES SETTING_KEY_RE names, each a finite number or a string (bounds are the space's). */
function isGenome(v: unknown): v is Genome {
  if (!isObj(v)) return false
  const ks = Object.keys(v)
  return ks.length <= MAX_GENES && ks.every(k => SETTING_KEY_RE.test(k) && (typeof v[k] === "string" || Number.isFinite(v[k])))
}
const isShaGenome = (v: unknown): v is { genomeSha: string; values: Genome } =>
  isObj(v) && exactKeys(v, ["genomeSha", "values"]) && typeof v.genomeSha === "string" && HEX64.test(v.genomeSha) && isGenome(v.values)

/** The checks both versions share: everything but models, incumbent, confirmation and the page's key set. */
function commonOk(v: Obj): boolean {
  if (!Number.isSafeInteger(v.genomeSchema) || (v.genomeSchema as number) < 1) return false
  if (typeof v.evaluator !== "string" || v.evaluator.length === 0) return false
  if (typeof v.sourceCommit !== "string" || !COMMIT.test(v.sourceCommit)) return false
  if (!MODES.includes(v.mode as Mode)) return false
  if (!isShaGenome(v.centre)) return false

  const cal = v.calibration
  if (!isObj(cal) || !exactKeys(cal, ["genomes", "need"])) return false
  if (!Array.isArray(cal.genomes) || !cal.genomes.every(isShaGenome)) return false
  if (!Array.isArray(cal.need) || !cal.need.every(isCount) || cal.need.length !== cal.genomes.length) return false
  if (v.mode === "calibrate" && cal.genomes.length === 0) return false // nothing to calibrate on

  if (!isObj(v.diagnostics)) return false
  const page = v.page as Obj // its key set is checked by the caller
  if (!isCount(page.eligibleLives) || !isCount(page.calibrationLives) || !isCount(page.calibrationTarget) || !isCount(page.looks)) return false
  if (page.lastPromotion !== null && typeof page.lastPromotion !== "string") return false
  return true
}

/** 05a's file, checked as 05a checks it, read with nothing of 05b's: draws are exactly 05a's. */
function parseV1(v: Obj): Posterior | null {
  if (!isObj(v.page) || !exactKeys(v.page, PAGE_KEYS_V1) || !commonOk(v)) return null
  if (!isObj(v.models)) return null
  if (v.incumbent !== null && !isObj(v.incumbent)) return null
  if (v.confirmation !== null && !isObj(v.confirmation)) return null
  return { ...v, models: {}, incumbent: null, confirmation: null, page: { ...v.page, rowsSinceDeclaration: 0 } } as unknown as Posterior
}

function isGpModel(v: unknown): v is GpModel {
  if (!isObj(v) || !exactKeys(v, ["genes", "rows", "y", "hyper", "fit"])) return false
  const { genes, rows, y, hyper, fit } = v
  if (!Array.isArray(genes) || genes.length < 1 || genes.length > MAX_GENES) return false
  if (!genes.every(g => typeof g === "string" && SETTING_KEY_RE.test(g)) || new Set(genes).size !== genes.length) return false
  const d = genes.length

  if (!Array.isArray(rows) || !Array.isArray(y) || rows.length !== y.length || rows.length < 1 || rows.length > MAX_ROWS) return false
  let prev = ""
  for (const r of rows) {
    if (!isObj(r) || !exactKeys(r, ["learnerId", "x"])) return false
    if (typeof r.learnerId !== "string" || !LEARNER_ID.test(r.learnerId) || !(r.learnerId > prev)) return false
    if (!isArrayOf(r.x, d, isUnit)) return false
    prev = r.learnerId
  }
  if (!y.every(isReal)) return false

  if (!isObj(hyper) || !exactKeys(hyper, ["lengthscales", "outputscale", "noise", "mean"])) return false
  if (!isArrayOf(hyper.lengthscales, d, isPositive) || !isPositive(hyper.outputscale)) return false
  if (!isReal(hyper.noise) || hyper.noise < NOISE_FLOOR || !isReal(hyper.mean)) return false

  if (!isObj(fit) || !exactKeys(fit, ["objective", "start", "iterations"])) return false
  if (!isReal(fit.objective) || !Number.isSafeInteger(fit.start) || (fit.start as number) < 0 || (fit.start as number) >= FIT_STARTS) return false
  return isArrayOf(fit.iterations, FIT_STARTS, isCount)
}

const isIncumbent = (v: unknown, d: number): v is Incumbent =>
  isObj(v) && exactKeys(v, ["x", "genomeSha", "mean", "pBeatsCentre"]) && isArrayOf(v.x, d, isUnit)
  && typeof v.genomeSha === "string" && HEX64.test(v.genomeSha) && isReal(v.mean) && isUnit(v.pBeatsCentre)

const isArmCounts = (v: unknown): v is Record<ArmName, number> => isObj(v) && exactKeys(v, ARMS) && ARMS.every(a => isCount(v[a]))

function isServedConfirmation(v: unknown): v is ServedConfirmation {
  if (!isObj(v) || !exactKeys(v, ["id", "seedBase", "candidate", "centre", "target", "counted", "need"])) return false
  if (!isCount(v.id) || !Number.isSafeInteger(v.seedBase) || v.seedBase !== 10_000 + 1_000 * v.id) return false
  if (!isShaGenome(v.candidate) || !isShaGenome(v.centre) || v.candidate.genomeSha === v.centre.genomeSha) return false
  if (!isArmCounts(v.target) || !isArmCounts(v.counted) || !isArmCounts(v.need)) return false
  const { target, need } = v
  return ARMS.every(a => need[a] <= target[a])
}

/** 05b's file: strict in every part, and nothing of the search outside mode "search" (the mode guard). */
function parseV2(v: Obj): Posterior | null {
  if (!isObj(v.page) || !exactKeys(v.page, PAGE_KEYS_V2) || !commonOk(v)) return null
  if (!isCount(v.page.rowsSinceDeclaration)) return null

  const models = v.models
  if (!isObj(models) || !Object.keys(models).every(k => k === "gpu")) return null
  if (Object.hasOwn(models, "gpu") && !isGpModel(models.gpu)) return null
  const gpu = models.gpu as GpModel | undefined
  if (v.incumbent !== null && (gpu === undefined || !isIncumbent(v.incumbent, gpu.genes.length))) return null
  if (v.confirmation !== null && !isServedConfirmation(v.confirmation)) return null

  // The mode guard: the search's state exists only in mode search.
  if (v.mode !== "search" && (gpu !== undefined || v.incumbent !== null || v.confirmation !== null)) return null
  return v as unknown as Posterior
}

/** posterior.json's text → a Posterior, or null on any shape error or unknown schemaVersion (the tab then draws centre). */
export function parsePosterior(text: string): Posterior | null {
  let v: unknown
  try { v = JSON.parse(text) } catch { return null }
  if (!isObj(v) || !exactKeys(v, TOP_KEYS)) return null
  if (v.schemaVersion === 1) return parseV1(v)
  if (v.schemaVersion === POSTERIOR_SCHEMA) return parseV2(v)
  return null
}
