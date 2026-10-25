// The gate's confirmations (spec §4.5; plan 05b Task 9): declaration, serving, the decision through the pinned
// confirm evaluator, and promotion. buildMeta (meta.ts) calls confirmStep only in mode "search".
//
// A confirmation file is append-only: written once when declared, once more when its result is appended, never again.
// Its times (declaredAt, deadline, decidedAt) are the build's `builtAt`, passed in: no clock is read here, and no time
// reaches posterior.json. Promotion writes knowledge/genome.json, and only on a pass of both tests (decision 0018).
import type { MetaBlock } from "@blanktab/harness/log"
import {
  armOfSeed, checkGenome, fromUnit, genomeSha, parseCentre, sha256Hex,
  type ArmName, type Genome, type GenomeSpace, type GpModel, type Incumbent, type ServedConfirmation,
} from "@blanktab/meta"
import { cohortContext, getEvaluator, type EvalOutput, type LifeInput, type LifeStanding } from "@blanktab/evaluators"
import type { MetaInput } from "./meta.ts"
import { triggered } from "./search.ts"
import { serialise } from "./serialise.ts"

/** Counted lives per arm (spec §4.5; V2 §6.3 power): candidate 24, centre 24, newborns of the candidate 60. */
export const CONFIRM_TARGET = { candidate: 24, centre: 24, newborn: 60 } as const
/** Days from declaration to the deadline [guess, spec §4.5]; the hours a silent running life keeps its slot [guess]. */
export const DEADLINE_DAYS = 21, IN_FLIGHT_HOURS = 48
/**
 * The most a running life's newest file may be written after builtAt and still hold a slot [guess]: tolerated clock
 * skew. A file "written" further ahead (a fast or forged clock) holds none, so it cannot keep a slot until the deadline.
 */
export const FORWARD_SKEW_MINUTES = 10
/**
 * The most confirmations one candidate genomeSha gets, whatever their outcomes (decision 0024, agent's call; final
 * review ruling): a trigger for an incumbent already decided this many times declares nothing, so luck alone cannot
 * promote a recipe by asking again and again. Changing it is a recorded decision.
 */
export const MAX_CONFIRMATIONS_PER_CANDIDATE = 3
export const confirmationPath = (c: number): string => `data/meta/confirmations/${c}.json`
export const PROMOTIONS_PATH = "data/meta/promotions.jsonl", GENOME_PATH = "knowledge/genome.json"
/** The decision a promotion cites. */
const PROMOTION_DECISION = "docs/decisions/0018-one-meta-learner-over-a-learning-genome.md"

type ShaGenome = { genomeSha: string; values: Genome }
export type Outcome = "pass" | "fail" | "incomplete" | "ineligible"
export interface ConfirmationResult {
  outcome: Outcome; decidedAt: string; sourceCommit: string; output: EvalOutput; lives: Record<ArmName, string[]>
}
export interface ConfirmationFile {
  schemaVersion: 1; id: number; genomeSchema: number; confirm: string; search: string
  candidate: ShaGenome; centre: ShaGenome
  seedBase: number; target: typeof CONFIRM_TARGET; declaredAt: string; deadline: string; sourceCommit: string
  trigger: { pBeatsCentre: number; incumbentMean: number; rowsAtDeclaration: number; modelSha: string }
  result: null | ConfirmationResult
}
/** A verified confirmation life (meta.ts readLives): its files and its birth's meta block. */
export interface VerifiedConfirmLife { life: LifeInput; meta: MetaBlock }

const ARMS: readonly ArmName[] = ["candidate", "centre", "newborn"]
const OUTCOMES: readonly unknown[] = ["pass", "fail", "incomplete", "ineligible"]
const HEX64 = /^[0-9a-f]{64}$/
/** The form Date#toISOString writes (meta-cli's builtAt). */
const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/
const DAY_MS = 86_400_000, HOUR_MS = 3_600_000, MINUTE_MS = 60_000
const byId = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const exactKeys = (v: Record<string, unknown>, keys: readonly string[]) => {
  const ks = Object.keys(v)
  return ks.length === keys.length && keys.every(k => Object.hasOwn(v, k))
}
const isCount = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0
const isIso = (v: unknown): v is string => typeof v === "string" && ISO.test(v) && Number.isFinite(Date.parse(v))
const isShaGenome = (v: unknown): v is ShaGenome =>
  isObj(v) && exactKeys(v, ["genomeSha", "values"]) && typeof v.genomeSha === "string" && HEX64.test(v.genomeSha) && isObj(v.values)

export const seedBaseOf = (c: number): number => 10_000 + 1_000 * c
/** builtAt + DEADLINE_DAYS days, as an ISO string. */
export const deadlineOf = (declaredAt: string): string => new Date(Date.parse(declaredAt) + DEADLINE_DAYS * DAY_MS).toISOString()

const KEYS = ["schemaVersion", "id", "genomeSchema", "confirm", "search", "candidate", "centre", "seedBase", "target", "declaredAt", "deadline", "sourceCommit", "trigger", "result"]
const TRIGGER_KEYS = ["pBeatsCentre", "incumbentMean", "rowsAtDeclaration", "modelSha"]
const RESULT_KEYS = ["outcome", "decidedAt", "sourceCommit", "output", "lives"]

/** data/meta/confirmations/<c>.json's text → the file; throws on any shape error (the gate then stops, writing nothing). */
export function parseConfirmation(text: string): ConfirmationFile {
  const v: unknown = JSON.parse(text)
  const fail = (why: string): never => { throw new Error(`confirmation file: ${why}`) }
  if (!isObj(v) || !exactKeys(v, KEYS)) return fail(`must have exactly the keys ${KEYS.join(", ")}`)
  const where = `confirmation ${String(v.id)}`
  const bad = (why: string): never => fail(`${where}: ${why}`)
  if (v.schemaVersion !== 1) bad("schemaVersion must be 1")
  if (!isCount(v.id)) bad("id must be a non-negative integer")
  if (!Number.isSafeInteger(v.genomeSchema) || (v.genomeSchema as number) < 1) bad("genomeSchema must be an integer >= 1")
  if (typeof v.confirm !== "string" || typeof v.search !== "string") bad("confirm and search must be id@version strings")
  if (!isShaGenome(v.candidate) || !isShaGenome(v.centre)) bad("candidate and centre must be { genomeSha, values }")
  if ((v.candidate as ShaGenome).genomeSha === (v.centre as ShaGenome).genomeSha) bad("the candidate is the centre")
  if (v.seedBase !== seedBaseOf(v.id as number)) bad(`seedBase must be ${seedBaseOf(v.id as number)}`)
  const t = v.target
  if (!isObj(t) || !exactKeys(t, ARMS) || !ARMS.every(a => t[a] === CONFIRM_TARGET[a])) bad("target must be candidate 24, centre 24, newborn 60")
  if (!isIso(v.declaredAt) || !isIso(v.deadline) || v.deadline !== deadlineOf(v.declaredAt)) bad(`deadline must be declaredAt + ${DEADLINE_DAYS} days (ISO times)`)
  if (typeof v.sourceCommit !== "string") bad("sourceCommit must be a string")
  const tr = v.trigger
  if (!isObj(tr) || !exactKeys(tr, TRIGGER_KEYS) || typeof tr.pBeatsCentre !== "number" || !(tr.pBeatsCentre >= 0 && tr.pBeatsCentre <= 1)
    || typeof tr.incumbentMean !== "number" || !Number.isFinite(tr.incumbentMean) || !isCount(tr.rowsAtDeclaration)
    || typeof tr.modelSha !== "string" || !HEX64.test(tr.modelSha)) {
    bad(`trigger must be { ${TRIGGER_KEYS.join(", ")} }`)
  }
  const r = v.result
  if (r !== null) {
    if (!isObj(r) || !exactKeys(r, RESULT_KEYS) || !OUTCOMES.includes(r.outcome) || !isIso(r.decidedAt) || typeof r.sourceCommit !== "string"
      || !isObj(r.output) || !isObj(r.lives) || !exactKeys(r.lives, ARMS)
      || !ARMS.every(a => Array.isArray((r.lives as Record<string, unknown>)[a]) && ((r.lives as Record<string, unknown[]>)[a]!).every(x => typeof x === "string"))) {
      bad(`result must be null or { ${RESULT_KEYS.join(", ")} } with an outcome of ${OUTCOMES.join(", ")}`)
    }
  }
  return v as unknown as ConfirmationFile
}

/**
 * How the pinned confirm evaluator counts a life of an arm (its Evaluator.standing): the gate never decides an arm is
 * full on lives the evaluator would not count, and holds slots only for lives it calls pending (still running).
 */
export type Standing = (life: LifeInput, arm: ArmName) => LifeStanding

/** The pinned confirm evaluator's standing (it must have one). */
function standingOf(input: MetaInput): Standing {
  const ev = getEvaluator(input.config.confirm)
  if (!ev.standing) throw new Error(`${input.config.confirm} gives no standing for a life: the gate cannot count its arms`)
  return ev.standing.bind(ev)
}

/** Confirmation `c`'s verified lives by arm (the seed's: armOfSeed), the lowest learnerId per (arm, evalSeed), in learnerId order. */
function byArm(lives: readonly VerifiedConfirmLife[], c: ConfirmationFile): Record<ArmName, VerifiedConfirmLife[]> {
  const out: Record<ArmName, VerifiedConfirmLife[]> = { candidate: [], centre: [], newborn: [] }
  const seen = new Set<string>()
  for (const l of [...lives].sort((a, b) => byId(a.life.learnerId, b.life.learnerId))) {
    const conf = l.meta.confirmation, seed = l.meta.evalSeed
    if (l.meta.role !== "confirm" || conf === null || conf.id !== c.id || seed === null) continue
    const arm = armOfSeed(seed, c.id)
    const key = `${arm}/${seed}`
    if (seen.has(key)) continue // a duplicate draw of this seed: only the lowest learnerId counts
    seen.add(key)
    out[arm].push(l)
  }
  return out
}

/** Per arm, the lives that count: up to target[arm] counted ones, and the others (abandoned, running) among them. */
function keptByArm(lives: readonly VerifiedConfirmLife[], c: ConfirmationFile, standing: Standing): Record<ArmName, { kept: VerifiedConfirmLife[]; counted: number }> {
  const arms = byArm(lives, c)
  const out = {} as Record<ArmName, { kept: VerifiedConfirmLife[]; counted: number }>
  for (const arm of ARMS) {
    const kept: VerifiedConfirmLife[] = []
    let counted = 0
    for (const l of arms[arm]) {
      if (counted === c.target[arm]) break // extras beyond the target are dropped by rule
      kept.push(l)
      if (standing(l.life, arm) === "counted") counted++
    }
    out[arm] = { kept, counted }
  }
  return out
}

/**
 * The groups confirm-fisher@1 is run on (spec §4.5): per (arm, evalSeed) the lowest learnerId, then, per arm in
 * learnerId order, the first target[arm] counted lives. Lives before the last of those that are not counted
 * (abandoned, still running) are passed too: the evaluator excludes them, and reports their rates.
 */
export function armGroups(lives: readonly VerifiedConfirmLife[], c: ConfirmationFile, standing: Standing): Record<ArmName, LifeInput[]> {
  const k = keptByArm(lives, c, standing)
  return { candidate: k.candidate.kept.map(l => l.life), centre: k.centre.kept.map(l => l.life), newborn: k.newborn.kept.map(l => l.life) }
}

/**
 * Per arm, the counted lives armGroups keeps, and the running lives (never counted) that hold a slot: newest file
 * written at most IN_FLIGHT_HOURS before builtAt and at most FORWARD_SKEW_MINUTES after it.
 */
function armCounts(lives: readonly VerifiedConfirmLife[], c: ConfirmationFile, builtAt: string, standing: Standing): { counted: Record<ArmName, number>; inFlight: Record<ArmName, number> } {
  const k = keptByArm(lives, c, standing), arms = byArm(lives, c)
  const now = Date.parse(builtAt)
  const counted = { candidate: k.candidate.counted, centre: k.centre.counted, newborn: k.newborn.counted }
  const inFlight = { candidate: 0, centre: 0, newborn: 0 }
  for (const arm of ARMS) {
    for (const l of arms[arm]) {
      if (standing(l.life, arm) !== "pending") continue
      const newest = l.life.files[l.life.files.length - 1]!
      const age = now - Date.parse(newest.writtenAt)
      if (age >= -FORWARD_SKEW_MINUTES * MINUTE_MS && age <= IN_FLIGHT_HOURS * HOUR_MS) inFlight[arm]++
    }
  }
  return { counted, inFlight }
}

const lifeIds = (groups: Record<ArmName, LifeInput[]>): Record<ArmName, string[]> =>
  ({ candidate: groups.candidate.map(l => l.learnerId), centre: groups.centre.map(l => l.learnerId), newborn: groups.newborn.map(l => l.learnerId) })

/** The pinned confirm evaluator on `groups`, with the outcome given. */
function resultOf(c: ConfirmationFile, groups: Record<ArmName, LifeInput[]>, outcome: (o: EvalOutput) => Outcome, input: MetaInput): ConfirmationResult {
  const ev = getEvaluator(input.config.confirm)
  if (ev.kind !== "cohort" || !ev.evaluateCohort) throw new Error(`${input.config.confirm} is not a cohort evaluator`)
  const output = ev.evaluateCohort(groups, input.refs, cohortContext(input.config))
  return { outcome: outcome(output), decidedAt: input.builtAt, sourceCommit: input.sourceCommit, output, lives: lifeIds(groups) }
}

/**
 * The result to append, or null when it is not yet time (spec §4.5): when every arm is `full` (counted reaches its
 * target), or builtAt is past the deadline. Full: pass on score 1, fail on score 0, ineligible when the output is not
 * eligible. The deadline first: incomplete, with the output still recorded. A file with a result keeps it.
 */
export function decide(c: ConfirmationFile, groups: Record<ArmName, LifeInput[]>, full: boolean, input: MetaInput): ConfirmationFile["result"] {
  if (c.result !== null) return c.result
  if (full) return resultOf(c, groups, o => (!o.eligible ? "ineligible" : o.score === 1 ? "pass" : "fail"), input)
  if (Date.parse(input.builtAt) > Date.parse(c.deadline)) return resultOf(c, groups, () => "incomplete", input)
  return null
}

/** What confirmStep gives buildMeta. */
export interface ConfirmStep {
  /** Changed confirmation files only, by path. */
  files: Record<string, string>
  served: ServedConfirmation | null
  /** The centre in force after this build: the candidate after a pass, else the input's. */
  centre: ShaGenome
  promotionLine: string | null
  genomeFile: string | null
  rowsSinceDeclaration: number
  lastPromotion: string | null
  /** For diagnostics.json: the open confirmation's counts (inFlight moves with time, so never in the posterior). */
  diagnostics: { id: number; counted: Record<ArmName, number>; inFlight: Record<ArmName, number> } | null
  /** For diagnostics.json: why a trigger that fired in this build declared nothing, else null. */
  skipped: string | null
}

/** The previous confirmations, checked: ids 0..n−1 in order, and only the last may be open. */
function checkedHistory(cs: readonly ConfirmationFile[]): void {
  cs.forEach((c, i) => {
    if (c.id !== i) throw new Error(`${confirmationPath(i)}: id ${c.id}, expected ${i} (ids are 0..n-1 with no gaps)`)
    if (c.result === null && i !== cs.length - 1) throw new Error(`${confirmationPath(i)}: open while a later confirmation exists`)
  })
}

/** The open confirmation, checked against the pins and the space it must have been declared for. */
async function checkedOpen(c: ConfirmationFile, input: MetaInput): Promise<void> {
  const fail = (why: string): never => { throw new Error(`${confirmationPath(c.id)}: ${why}`) }
  // A pin or schema change is a recorded decision that restarts calibration; the gate refuses rather than guess.
  if (c.confirm !== input.config.confirm) fail(`was declared for ${c.confirm}, but ${input.config.confirm} is pinned`)
  if (c.search !== input.config.search) fail(`was declared for search evaluator ${c.search}, but ${input.config.search} is pinned`)
  if (c.genomeSchema !== input.space.schemaVersion) fail(`was declared for genome schema ${c.genomeSchema}, but the space is ${input.space.schemaVersion}`)
  for (const [name, g] of [["candidate", c.candidate], ["centre", c.centre]] as const) {
    const bad = checkGenome(input.space, g.values)
    if (bad !== null) fail(`the ${name} is not in the space: ${bad}`)
    if (g.genomeSha !== await genomeSha(c.genomeSchema, g.values)) fail(`the ${name}'s genomeSha does not match its values`)
  }
}

const served = (c: ConfirmationFile, counts: { counted: Record<ArmName, number>; inFlight: Record<ArmName, number> }): ServedConfirmation => ({
  id: c.id, seedBase: c.seedBase, candidate: c.candidate, centre: c.centre, target: { ...c.target }, counted: counts.counted,
  need: {
    candidate: Math.max(0, c.target.candidate - counts.counted.candidate - counts.inFlight.candidate),
    centre: Math.max(0, c.target.centre - counts.counted.centre - counts.inFlight.centre),
    newborn: Math.max(0, c.target.newborn - counts.counted.newborn - counts.inFlight.newborn),
  },
})

/** knowledge/genome.json for a pass, checked by parseCentre (it must read back as the candidate, or the build throws). */
function promotedGenome(c: ConfirmationFile, result: ConfirmationResult, space: GenomeSpace, builtAt: string): string {
  const d = result.output.diagnostics
  const text = serialise({
    schemaVersion: 1, genomeSchema: c.genomeSchema, values: c.candidate.values,
    provenance: {
      source: "promotion", confirmation: c.id, decision: PROMOTION_DECISION, promotedAt: builtAt,
      fisher: d.fisher ?? null, mannWhitneyP: result.output.components.mannWhitneyP ?? null, lives: result.lives,
    },
  })
  const back = parseCentre(JSON.parse(text), space)
  if (serialise(back) !== serialise(Object.fromEntries(space.genes.map(g => [g.name, c.candidate.values[g.name]])))) {
    throw new Error(`${GENOME_PATH}: the promoted genome does not read back as confirmation ${c.id}'s candidate`)
  }
  return text
}

/**
 * One build's confirmation step (spec §4.5), mode search only: the open confirmation is counted and served, decided
 * when due, and on a pass promoted (knowledge/genome.json, one promotions.jsonl line, the new centre). With none
 * open and none decided in this build, a trigger declares the next: id = n, seedBase 10,000 + 1,000·id, the
 * incumbent's genome against the current centre, need = target. `searchEraRows` are the rows born under a search
 * posterior; rowsSinceDeclaration counts from the last declaration's rowsAtDeclaration.
 *
 * A confirmation whose centre is no longer the centre in force (genome.json changed by hand while it was open) is
 * decided at once, incomplete, and is never promoted: its comparison is against a centre no longer in force.
 */
export async function confirmStep(input: MetaInput, confirmLives: readonly VerifiedConfirmLife[], searchEraRows: number, model: GpModel | null,
  incumbent: Incumbent | null, centre: ShaGenome): Promise<ConfirmStep> {
  const prev = input.previous.confirmations
  checkedHistory(prev)
  const last = prev.length > 0 ? prev[prev.length - 1]! : null
  const out: ConfirmStep = {
    files: {}, served: null, centre, promotionLine: null, genomeFile: null,
    rowsSinceDeclaration: Math.max(0, searchEraRows - (last?.trigger.rowsAtDeclaration ?? 0)), lastPromotion: null, diagnostics: null, skipped: null,
  }
  const all = [...prev]
  const standing = standingOf(input)
  let decided = false
  if (last !== null && last.result === null) {
    await checkedOpen(last, input)
    const groups = armGroups(confirmLives, last, standing)
    const counts = armCounts(confirmLives, last, input.builtAt, standing)
    const full = ARMS.every(a => counts.counted[a] >= last.target[a])
    const result = last.centre.genomeSha !== centre.genomeSha
      ? resultOf(last, groups, () => "incomplete", input)
      : decide(last, groups, full, input)
    if (result === null) {
      out.served = served(last, counts)
      out.diagnostics = { id: last.id, ...counts }
    } else {
      decided = true
      const done: ConfirmationFile = { ...last, result }
      all[all.length - 1] = done
      out.files[confirmationPath(done.id)] = serialise(done)
      if (result.outcome === "pass") {
        out.genomeFile = promotedGenome(done, result, input.space, input.builtAt)
        out.promotionLine = JSON.stringify({
          confirmation: done.id, genomeSha: done.candidate.genomeSha, previous: centre.genomeSha, promotedAt: input.builtAt, sourceCommit: input.sourceCommit,
        })
        out.centre = { genomeSha: done.candidate.genomeSha, values: { ...done.candidate.values } }
      }
    }
  }

  const open = all.length > 0 && all[all.length - 1]!.result === null
  if (!decided && !open && model !== null && incumbent !== null && triggered(incumbent, centre.genomeSha, out.rowsSinceDeclaration, false)) {
    const id = all.length
    const values = fromUnit(input.space, incumbent.x, centre.values)
    const candidateSha = await genomeSha(input.space.schemaVersion, values)
    if (candidateSha !== incumbent.genomeSha) throw new Error(`confirmation ${id}: the incumbent's genomeSha does not match fromUnit of its x`)
    const before = all.filter(p => p.result !== null && p.candidate.genomeSha === candidateSha).length
    if (before >= MAX_CONFIRMATIONS_PER_CANDIDATE) {
      out.skipped = `candidate ${candidateSha.slice(0, 8)} already confirmed ${before} times`
    } else {
      const c: ConfirmationFile = {
        schemaVersion: 1, id, genomeSchema: input.space.schemaVersion, confirm: input.config.confirm, search: input.config.search,
        candidate: { genomeSha: candidateSha, values }, centre: { genomeSha: centre.genomeSha, values: { ...centre.values } },
        seedBase: seedBaseOf(id), target: { ...CONFIRM_TARGET }, declaredAt: input.builtAt, deadline: deadlineOf(input.builtAt), sourceCommit: input.sourceCommit,
        trigger: {
          pBeatsCentre: incumbent.pBeatsCentre, incumbentMean: incumbent.mean, rowsAtDeclaration: searchEraRows,
          modelSha: await sha256Hex(new TextEncoder().encode(serialise(model))),
        },
        result: null,
      }
      const text = serialise(c)
      parseConfirmation(text) // what the gate writes, it can read back
      all.push(c)
      out.files[confirmationPath(id)] = text
      const counts = armCounts(confirmLives, c, input.builtAt, standing)
      out.served = served(c, counts)
      out.diagnostics = { id, ...counts }
      out.rowsSinceDeclaration = 0
    }
  }
  for (const c of all) if (c.result?.outcome === "pass") out.lastPromotion = c.result.decidedAt
  return out
}
