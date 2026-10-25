// Pure helpers for the page's text: the history table, the links, the build label. No DOM here.
import type { MetaMethod, SessionRow, Summary } from "@blanktab/harness"
import { SUITE_SIZE, fmtLevel } from "./milestones"

export interface Repo { readonly owner: string; readonly name: string }

const gh = (repo: Repo) => `https://github.com/${repo.owner}/${repo.name}`

/**
 * A setting value as shown on the page: numbers to at most 3 decimals, except that a nonzero value below 1 keeps 3
 * significant digits (lr 5e-4 is "0.0005", not "0.001"; rounding to decimals would misstate small settings).
 */
export const fmt = (v: number | string) => {
  if (typeof v !== "number") return v
  return v !== 0 && Math.abs(v) < 1 ? String(Number(v.toPrecision(3))) : String(Math.round(v * 1000) / 1000)
}

export const fmtSettings = (settings: Record<string, number | string>) => {
  const entries = Object.entries(settings)
  return entries.length ? entries.map(([k, v]) => `${k} = ${fmt(v)}`).join(", ") : "none"
}

export interface HistoryRow {
  learnerId: string; href: string; steps: string; final: string; setting: string; host: string
  /** Compute lived (CU) and the last loss ratio (predLoss ÷ the lazy guess's), "—" for a v1 (flailer) learner. */
  cu: string; lossRatio: string
  /** The latest held-out ratio (decision 0015 §3), "—" for a learner from before the held-out check. */
  heldOutRatio: string
  /** The latest checkup's score, "2 of 6 at 512 CU", or "—" for a learner without one. */
  milestones: string
  /** How its genome was drawn (its row's recipe.method, in a word or two: RECIPE_WORDS), or "—" without a block. */
  recipe: string
  /** Its score under the pinned search evaluator ("0.41"), "not eligible", or "—" without one. */
  score: string
}

/**
 * The HISTORY table's word for each draw method (agent's call: short, plain words; the full line is the LEARNER tab's).
 * A centre life ran the centre of its own time, which a promotion may since have replaced: "centre", never "current".
 */
export const RECIPE_WORDS: Readonly<Record<MetaMethod, string>> = {
  centre: "centre", calibrate: "calibration", uniform: "random", ts: "meta-learner", confirm: "confirmation", newborn: "newborn", legacy: "legacy",
}

/** The HISTORY table's columns in order (the <th> texts of index.html); the numeric ones are right-aligned. Milestones is text. */
export const HISTORY_COLUMNS: readonly { label: string; numeric: boolean }[] = [
  { label: "Learner", numeric: false }, { label: "Steps", numeric: true }, { label: "CU", numeric: true },
  { label: "Held-out ratio", numeric: true }, { label: "Own-stream ratio", numeric: true }, { label: "Milestones", numeric: false },
  { label: "Recipe", numeric: false }, { label: "Score", numeric: true }, { label: "Final", numeric: false }, { label: "Setting", numeric: false }, { label: "Host", numeric: false },
]

const orDash = (v: number | null | undefined) => (typeof v === "number" ? fmt(v) : "—")

/** A row's score under `evaluator` (id@version), as the HISTORY table shows it. */
function scoreCell(s: SessionRow, evaluator: string | undefined): string {
  const e = evaluator === undefined ? undefined : s.evals?.[evaluator]
  if (!e) return "—"
  return e.score === null ? "not eligible" : e.score.toFixed(2)
}

/**
 * Past sessions, newest first by lastWrittenAt (ISO strings sort as times). Rows without it (older summaries)
 * go last, ordered by sessionsSeenAtBirth descending, then the later summary entry first. `evaluator`: the pinned search
 * evaluator (id@version) whose score the Score column shows (none: "—").
 */
export function historyRows(summary: Summary, repo: Repo, evaluator?: string): HistoryRow[] {
  return summary.sessions
    .map((s, i) => ({ s, i }))
    .sort((a, b) => {
      const ta = a.s.lastWrittenAt, tb = b.s.lastWrittenAt
      if (ta !== undefined || tb !== undefined) {
        if (ta === undefined) return 1
        if (tb === undefined) return -1
        if (ta !== tb) return ta < tb ? 1 : -1
      }
      return b.s.sessionsSeenAtBirth - a.s.sessionsSeenAtBirth || b.i - a.i
    })
    .map(({ s }: { s: SessionRow }) => ({
      learnerId: s.learnerId,
      href: `${gh(repo)}/tree/main/data/runs/${s.learnerId}`,
      steps: String(s.stepsTotal),
      final: s.final ? "yes" : "no",
      setting: fmtSettings(s.settings),
      host: `${s.hostSig.cores} cores · ${s.hostSig.webgpu ? "webgpu" : "no webgpu"}`,
      cu: orDash(s.cu),
      lossRatio: orDash(s.lossRatio),
      heldOutRatio: orDash(s.heldOutRatio),
      milestones: s.checkupScore ? `${s.checkupScore.score} of ${SUITE_SIZE} at ${fmtLevel(s.checkupScore.level)} CU` : "—",
      recipe: s.recipe ? RECIPE_WORDS[s.recipe.method] : "—",
      score: scoreCell(s, evaluator),
    }))
}

export interface Link { label: string; href: string }

export function siteLinks(repo: Repo): Link[] {
  const r = gh(repo)
  return [
    { label: "Source", href: r },
    { label: "Design", href: `${r}/blob/main/docs/specs/2026-10-19-blanktab-v1-design.md` },
    { label: "Decisions", href: `${r}/tree/main/docs/decisions` },
    { label: "Research notes", href: `${r}/blob/main/docs/research/2026-10-16-research-synthesis-v0.1.md` },
    { label: "How to check it works", href: `${r}/blob/main/docs/validate-plan-01.md` },
    { label: "Session pull requests", href: `${r}/pulls` },
    { label: "Actions", href: `${r}/actions` },
    { label: "Session data", href: `${r}/tree/main/data/runs` },
    { label: "Related: life", href: "https://markwhiting.github.io/life/" },
  ]
}

/** "<short sha> built <time>", linking to the commit (or to the repo for a dev build). */
export function buildLabel(codeSha: string, builtAt: string, repo: Repo): { text: string; href: string } {
  const when = `${builtAt.slice(0, 10)} ${builtAt.slice(11, 16)} UTC`
  const dev = codeSha === "dev"
  return { text: `${dev ? "dev" : codeSha.slice(0, 7)} built ${when}`, href: dev ? gh(repo) : `${gh(repo)}/commit/${codeSha}` }
}
