// Builds the shared summary the next tab reads: one row per learner, from its highest-seq log file.
import { readdirSync, readFileSync, existsSync } from "node:fs"
import { join } from "node:path"
import { parseLogFile, LEARNER_ID_RE, type LogFile, type SessionEval, type SessionRow, type Summary } from "@blanktab/harness/log"
import { getEvaluator, type Evaluator, type References } from "@blanktab/evaluators"
import { loadEvalConfig, loadReferences } from "../eval/config.ts"

export type { SessionRow, Summary }

/** Reads every parseable <n>.json in a learner dir (heartbeat and junk are skipped). */
export function readLearnerLogs(dir: string): LogFile[] {
  const out: LogFile[] = []
  for (const name of readdirSync(dir)) {
    if (!/^\d+\.json$/.test(name)) continue
    try {
      const p = parseLogFile(JSON.parse(readFileSync(join(dir, name), "utf8")))
      if (p.ok) out.push(p.value)
    } catch { /* unreadable file: skip */ }
  }
  return out.sort((a, b) => a.seq - b.seq)
}

/**
 * The learner's latest loss ratio: predLoss ÷ baselineLoss at the last curve point of its v2 logs (the highest-seq
 * file with a point), or null if there is none or the ratio is not finite (a zero baseline).
 */
export function lossRatio(logs: LogFile[]): number | null {
  for (let i = logs.length - 1; i >= 0; i--) {
    const l = logs[i]!
    if (l.schemaVersion !== 2) continue
    const p = l.curve[l.curve.length - 1]
    if (!p) continue
    const r = p.predLoss / p.baselineLoss
    return Number.isFinite(r) ? r : null
  }
  return null
}

/**
 * The learner's latest held-out ratio (decision 0015 §3): heldOut.loss ÷ heldOut.baseline at the newest curve point
 * of its v2 logs that has a held-out score, or null if none has or the ratio is not finite (a zero baseline).
 */
export function heldOutRatio(logs: LogFile[]): number | null {
  for (let i = logs.length - 1; i >= 0; i--) {
    const l = logs[i]!
    if (l.schemaVersion !== 2) continue
    for (let j = l.curve.length - 1; j >= 0; j--) {
      const h = l.curve[j]!.heldOut
      if (!h) continue
      const r = h.loss / h.baseline
      return Number.isFinite(r) ? r : null
    }
  }
  return null
}

/**
 * The learner's latest checkup: its level and score (how many milestones it passed), from the newest v2 file that
 * has one, or null if none does.
 */
export function latestCheckup(logs: LogFile[]): { level: number; score: number } | null {
  for (let i = logs.length - 1; i >= 0; i--) {
    const l = logs[i]!
    if (l.schemaVersion !== 2) continue
    const c = l.checkups?.[l.checkups.length - 1]
    if (c) return { level: c.level, score: c.passed.length }
  }
  return null
}

/**
 * How the life's genome was drawn: the role and method of its first meta block (lowest seq), or null when no file has
 * one (v1 lives, lives from before 05a). The block is constant across a life's files (the gate checks), except a
 * pre-05a life resumed later, whose first block is its legacy one.
 */
export function recipeOf(logs: LogFile[]): SessionRow["recipe"] | null {
  for (const l of logs) if (l.schemaVersion === 2 && l.meta) return { role: l.meta.role, method: l.meta.method }
  return null
}

/** What the summary scores each life with: the pinned search evaluator and the pinned references (summary-cli.ts). */
export interface SummaryScoring { evaluator: Evaluator; refs: References }

/**
 * The scoring every summary writer uses (deploy's summary-cli.ts and the dev server, app/vite.config.ts): the search
 * evaluator config/evaluation.json under `root` pins, with its pinned references (an absent reference: rows not
 * eligible). Throws on a malformed config, so no writer silently publishes an unscored summary.
 */
export function pinnedScoring(root: string): SummaryScoring {
  const config = loadEvalConfig(root)
  return { evaluator: getEvaluator(config.search), refs: loadReferences(root, config) }
}

/** `life` scored by `s.evaluator`, keyed by its id@version (spec §2.2: every score carries its evaluator). */
function evalsOf(learnerId: string, files: LogFile[], s: SummaryScoring): Record<string, SessionEval> {
  const e = s.evaluator
  if (e.kind !== "life" || !e.evaluateLife) throw new Error(`${e.id}@${e.version} is not a life evaluator: it scores no single life`)
  const out = e.evaluateLife({ learnerId, files }, s.refs)
  const num = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? v : null)
  const score = out.eligible ? num(out.score) : null
  return { [`${out.evaluator}@${out.version}`]: { eligible: out.eligible && score !== null, score, C: num(out.components.C), H: num(out.components.H) } }
}

/**
 * One row per learner, from its highest-seq file. With `scoring` (writers pass pinnedScoring(root)), each row also
 * carries `evals`: the life (all its files) scored by that evaluator, keyed by id@version; without a reference its rows
 * say not eligible. `scoring` is required: an explicit null builds rows without `evals` (summaries as before 05a; tests).
 */
export function buildSummary(runsDir: string, scoring: SummaryScoring | null): Summary {
  const sessions: SessionRow[] = []
  if (existsSync(runsDir)) {
    for (const id of readdirSync(runsDir).sort()) {
      if (!LEARNER_ID_RE.test(id)) continue
      const logs = readLearnerLogs(join(runsDir, id)).filter(l => l.learnerId === id)
      const last = logs[logs.length - 1]
      if (!last) continue
      const recipe = recipeOf(logs)
      sessions.push({
        learnerId: id, settings: last.settings, hostSig: last.hostSig, stepsTotal: last.stepsTotal,
        final: last.final, codeSha: last.codeSha, knowledgeSha: last.knowledgeSha, sessionsSeenAtBirth: last.sessionsSeenAtBirth,
        lastWrittenAt: last.writtenAt,
        cu: last.schemaVersion === 2 ? last.cu : null,
        lossRatio: lossRatio(logs),
        heldOutRatio: heldOutRatio(logs),
        checkupScore: latestCheckup(logs),
        ...(scoring ? { evals: evalsOf(id, logs, scoring) } : {}),
        ...(recipe ? { recipe } : {}),
      })
    }
  }
  return { schemaVersion: 1, sessions }
}
