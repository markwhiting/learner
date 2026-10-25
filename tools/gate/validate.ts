// Gate validator for session PRs. Runs in Node (GitHub Action) on PR files treated purely as data.
import { CONFIG_BUDGET, EVAL_CU_TOLERANCE, levelAtOrBelow, lifeBudget, parseLogFile, sameBudget, type EvalBudget, type LogFile, type MetaBlock } from "@blanktab/harness/log"
import { checkGenome, genomeSha, settingsMismatch, type GenomeSpace } from "@blanktab/meta"

export interface PrFile { path: string; content: string }

const MAX_BYTES = 1_048_576
const PATH_RE = /^data\/runs\/(lrn_[0-9a-f]{16})\/(\d+\.json)$/

export interface ValidateOptions {
  /**
   * Every genome space main has had, by schemaVersion (history.ts spacesBySchema): a non-legacy meta.genome is checked
   * against the space of its own genomeSchema. Without it (or empty) such a file is rejected.
   */
  spaces?: ReadonlyMap<number, GenomeSpace>
  /**
   * The current config's evaluation budget (default CONFIG_BUDGET, config/evaluation.json): a new life's (seq 0) block
   * must be born under it. Every other check reads the file's own budget (lifeBudget), so a config change never
   * invalidates a life already running (decision 0020).
   */
  budget?: EvalBudget
}

/** A spaces map (ValidateOptions.spaces) of the given spaces, by schemaVersion (a later one wins a tie). */
export function spacesOf(...spaces: GenomeSpace[]): Map<number, GenomeSpace> {
  return new Map(spaces.map(s => [s.schemaVersion, s]))
}

/**
 * Checks a session PR's files against the learner's files on main (`previous`). Async: meta.genomeSha is recomputed
 * (sha256 through Web Crypto).
 */
export async function validateSessionPr(files: PrFile[], previous: LogFile[], opts: ValidateOptions = {}): Promise<{ ok: boolean; errors: string[]; kind: "log" | "needs-human" }> {
  const errors: string[] = []
  let kind: "log" | "needs-human" = "log"
  if (files.length === 0) return { ok: false, errors: ["PR contains no files"], kind }

  const dirIds = new Set<string>()
  const logs: { path: string; log: LogFile }[] = []

  for (const f of files) {
    const m = PATH_RE.exec(f.path)
    if (!m) {
      kind = "needs-human"
      errors.push(`${f.path}: path is not a session log path (data/runs/<learnerId>/<n>.json)`)
      continue
    }
    const dirId = m[1]!
    dirIds.add(dirId)
    if (new TextEncoder().encode(f.content).length > MAX_BYTES) {
      errors.push(`${f.path}: file exceeds 1 MB`)
      continue
    }
    let json: unknown
    try { json = JSON.parse(f.content) } catch { errors.push(`${f.path}: invalid JSON`); continue }

    const p = parseLogFile(json)
    if (!p.ok) { errors.push(`${f.path}: ${p.error}`); continue }
    if (p.value.learnerId !== dirId) errors.push(`${f.path}: path learnerId ${dirId} does not match learnerId ${p.value.learnerId} in file`)
    if (`${p.value.seq}.json` !== m[2]) errors.push(`${f.path}: filename does not match seq ${p.value.seq}`)
    logs.push({ path: f.path, log: p.value })
  }

  if (dirIds.size > 1) errors.push(`all files must be for the same learnerId (found ${[...dirIds].join(", ")})`)

  // previous state must belong to this PR's learner, and a finished session accepts no more logs
  for (const prev of previous) {
    for (const id of dirIds) {
      if (prev.learnerId !== id) { errors.push(`previous log (seq ${prev.seq}) belongs to ${prev.learnerId}, not ${id}`); break }
    }
  }
  if (logs.length > 0 && previous.some(p => p.final)) errors.push("session already has a final file; no further log files are allowed")

  // seq: unique within the PR and strictly above the highest seq already on main. Gaps are allowed (decision 0014): a
  // PR the gate closed takes its files with it, and the learner's next PR goes on from where the learner is.
  const seen = new Set<number>()
  for (const { path, log } of logs) {
    if (seen.has(log.seq)) errors.push(`${path}: duplicate seq ${log.seq}`)
    seen.add(log.seq)
  }
  const sorted = [...logs].sort((a, b) => a.log.seq - b.log.seq)
  const prevSorted = [...previous].sort((a, b) => a.seq - b.seq)
  const prevLast = prevSorted[prevSorted.length - 1]
  const mainMax = prevLast ? prevLast.seq : -1
  let lastSteps = prevLast ? prevLast.stepsTotal : -Infinity
  // From schema v2 on, these never decrease across seq (v1 files carry none and are skipped). restores, lostCU and
  // skippedUpdates keep restores on the record: a restore cannot hide the learning it threw away.
  const MONOTONIC = ["cu", "restores", "lostCU", "skippedUpdates"] as const
  const lastV2 = Object.fromEntries(MONOTONIC.map(k => [k, -Infinity])) as Record<(typeof MONOTONIC)[number], number>
  for (const p of prevSorted) if (p.schemaVersion === 2) for (const k of MONOTONIC) lastV2[k] = Math.max(lastV2[k], p[k])
  let lastEvalCU = -Infinity
  for (const p of prevSorted) if (p.schemaVersion === 2 && p.evalCU !== undefined) lastEvalCU = Math.max(lastEvalCU, p.evalCU)
  // The evalCU of the file just before (by seq; on main or earlier in this PR), undefined if it has none: a file's
  // checkups are billed to its own increase over that (decision 0016), not to evalCU spent before.
  let prevFileEvalCU = prevLast?.schemaVersion === 2 ? prevLast.evalCU : undefined
  // Checkup levels strictly increase across files (each file's own are checked by the parser): a level is checked once.
  let lastLevel = -Infinity
  for (const p of prevSorted) if (p.schemaVersion === 2) for (const c of p.checkups ?? []) lastLevel = Math.max(lastLevel, c.level)
  const reported = new Set<number>()
  for (const { path, log } of sorted) {
    if (reported.has(log.seq)) continue // duplicate already reported
    reported.add(log.seq)
    if (log.seq <= mainMax) errors.push(`${path}: seq ${log.seq} is not above seq ${mainMax}, the highest on main`)
    // A gap may be any size, but seq stays an exact integer (beyond 2^53 two seqs could name one file).
    if (!Number.isSafeInteger(log.seq)) errors.push(`${path}: seq ${log.seq} is beyond the largest exact integer`)
    if (log.stepsTotal < lastSteps) errors.push(`${path}: stepsTotal decreased (${log.stepsTotal} < ${lastSteps})`)
    lastSteps = Math.max(lastSteps, log.stepsTotal)
    if (log.schemaVersion === 2) {
      for (const k of MONOTONIC) {
        if (log[k] < lastV2[k]) errors.push(`${path}: ${k} decreased (${log[k]} < ${lastV2[k]})`)
        lastV2[k] = Math.max(lastV2[k], log[k])
      }
      // Optional (files before the held-out check have none): checked between files that both have it.
      // Once a learner's file has it, every later one must (a file never goes from a number back to absent).
      if (log.evalCU !== undefined) {
        if (log.evalCU < lastEvalCU) errors.push(`${path}: evalCU decreased (${log.evalCU} < ${lastEvalCU})`)
        lastEvalCU = Math.max(lastEvalCU, log.evalCU)
      } else if (lastEvalCU > -Infinity) {
        errors.push(`${path}: evalCU missing after a file that has it`)
      }
      for (const c of log.checkups ?? []) {
        if (c.level <= lastLevel) errors.push(`${path}: checkup level ${c.level} is not above ${lastLevel}`)
        lastLevel = Math.max(lastLevel, c.level)
      }
      const billed = (log.checkups ?? []).reduce((a, c) => a + c.evalCU, 0)
      if (billed > 0 && log.evalCU !== undefined && prevFileEvalCU !== undefined && Number.isFinite(prevFileEvalCU)) {
        const increase = log.evalCU - prevFileEvalCU
        if (billed - increase > EVAL_CU_TOLERANCE * Math.max(Math.abs(increase), billed)) {
          errors.push(`${path}: checkups' evalCU (${billed}) exceeds this file's evalCU increase (${log.evalCU} - ${prevFileEvalCU} = ${increase})`)
        }
      }
      prevFileEvalCU = log.evalCU
    } else {
      prevFileEvalCU = undefined
    }
  }

  // final: at most one, and it has the highest seq
  const finals = sorted.filter(l => l.log.final)
  if (finals.length > 1) errors.push(`at most one final file allowed (found ${finals.length})`)
  else if (finals.length === 1 && finals[0] !== sorted[sorted.length - 1]) errors.push(`${finals[0]!.path}: final file must have the highest seq`)

  errors.push(...await metaErrors(sorted, prevSorted, opts.spaces ?? new Map(), opts.budget ?? CONFIG_BUDGET))
  return { ok: errors.length === 0, errors, kind }
}

/** The meta fields that may differ between a learner's files; every other field is constant (spec §7.3). */
const META_VARIABLE = new Set<string>(["filters", "heldOut", "end"])

/** JSON with object keys sorted, so equal values compare equal whatever their key order. */
const canonical = (v: unknown): string => JSON.stringify(v, (_k, x: unknown) =>
  x !== null && typeof x === "object" && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) : x)

/**
 * The gate's meta checks (spec §7.3; plan 05a Task 5), over the PR's files (`pr`) after the learner's files on main
 * (`previous`), both in seq order:
 * - schema: never newer than the newest of `spaces`; a non-legacy block's schema must be one of `spaces`;
 * - bounds: a non-legacy genome lies in the space of its own schema (decision 0024 "Every schema checked against its
 *   own space": an older schema is no bypass); legacy lives keep their pre-05a values (plan gap 4);
 * - what ran: such a genome decodes (settingsMismatch) to the file's logged settings;
 * - seed: a confirmation life's learner seed (settings.seed) is its evalSeed (spec §4.5);
 * - integrity: genomeSha recomputes, for every role;
 * - constancy: every field but filters, heldOut and end equals the learner's first file with meta (absence allowed);
 * - budget: a new life's (seq 0) block is born under `budget`, the current config's (absent reads as V1_BUDGET);
 *   constancy keeps it for the rest of the life;
 * - filters: at most one file per level of the life's own budget carries them.
 */
async function metaErrors(pr: { path: string; log: LogFile }[], previous: LogFile[], spaces: ReadonlyMap<number, GenomeSpace>, budget: EvalBudget): Promise<string[]> {
  const errors: string[] = []
  const metaOf = (l: LogFile): MetaBlock | undefined => (l.schemaVersion === 2 ? l.meta : undefined)
  const newest = spaces.size > 0 ? Math.max(...spaces.keys()) : null
  for (const { path, log } of pr) {
    const m = metaOf(log)
    if (!m) continue
    // A tab is served from main, so its genome schema cannot be ahead of main's space (any role). Every other
    // non-legacy block is checked against the space of its own schema: bounds, what ran and (below) its sha.
    if (newest !== null && m.genomeSchema > newest) errors.push(`${path}: meta.genomeSchema ${m.genomeSchema} is newer than main's genome space (${newest})`)
    else if (m.role !== "legacy") {
      const space = spaces.get(m.genomeSchema)
      if (newest === null) errors.push(`${path}: no genome space to check meta.genome against`)
      else if (!space) errors.push(`${path}: unknown genome schema ${m.genomeSchema}`)
      else {
        const bad = checkGenome(space, m.genome)
        if (bad !== null) errors.push(`${path}: meta.genome is outside the genome space: ${bad}`)
        else {
          // The block describes the learner that ran: its genes decode to the file's own logged settings.
          const off = settingsMismatch(space, m.genome, log.schemaVersion === 2 ? log.settings : {})
          if (off !== null) errors.push(`${path}: meta.genome is not what ran: ${off}`)
        }
      }
    }
    // A confirmation life's learner seed is its evalSeed (spec §4.5); its world seed stays its learnerId's.
    if (m.role === "confirm" && log.schemaVersion === 2 && log.settings.seed !== m.evalSeed) {
      errors.push(`${path}: a confirmation life's settings.seed (${String(log.settings.seed)}) is not its meta.evalSeed (${m.evalSeed})`)
    }
    if (await genomeSha(m.genomeSchema, m.genome) !== m.genomeSha) errors.push(`${path}: meta.genomeSha does not match its genome`)
    if (log.seq === 0 && m.role !== "legacy" && !sameBudget(lifeBudget(m), budget)) {
      const b = lifeBudget(m)
      errors.push(`${path}: a new life's meta.budget (${b.budgetCU} CU at ${b.levels.join(", ")}) is not the config's (${budget.budgetCU} CU at ${budget.levels.join(", ")})`)
    }
  }
  const all = [...previous.map(log => ({ path: null as string | null, log })), ...pr]
  let first: MetaBlock | undefined
  const filterLevels = new Set<number>()
  for (const { path, log } of all) {
    const m = metaOf(log)
    if (!m) continue
    if (!first) first = m
    else if (path !== null) {
      const keys = new Set([...Object.keys(first), ...Object.keys(m)].filter(k => !META_VARIABLE.has(k)))
      for (const k of keys) {
        const a = (first as unknown as Record<string, unknown>)[k], b = (m as unknown as Record<string, unknown>)[k]
        if (canonical(a) !== canonical(b)) errors.push(`${path}: meta.${k} differs from the learner's earlier files`)
      }
    }
    if (m.filters !== undefined && log.schemaVersion === 2) {
      const level = levelAtOrBelow(log.cu)
      if (level !== null && lifeBudget(m).levels.includes(level)) {
        if (filterLevels.has(level) && path !== null) errors.push(`${path}: a second file carries meta.filters for level ${level}`)
        filterLevels.add(level)
      }
    }
  }
  return errors
}
