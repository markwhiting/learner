// Usage: tsx tools/gate/commit-meta.ts   (in the gate's checkout of main, after meta-cli.ts; prints
// meta=committed|verified-only|unchanged)
// The gate's one direct write to main (decision 0019; invariant 7): a commit of exactly the files meta-cli.ts computed
// from main's merged logs (data/meta/**, and knowledge/genome.json on a promotion), never PR content, never forced. On
// a push that main moved past, it resets to the new main and rebuilds from scratch (meta-cli.ts again), so it never
// pushes bytes built on an old main.
import { execFileSync } from "node:child_process"
import { createHash } from "node:crypto"
import { appendFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
import { isDeepStrictEqual } from "node:util"
import { parseCentre, parseGenomeSpace } from "@blanktab/meta"
import { confirmationPath, GENOME_PATH, parseConfirmation, PROMOTIONS_PATH, type ConfirmationFile } from "./confirm.ts"
import { POSTERIOR_INDEX_PATH, VERIFIED_PATH } from "./meta.ts"

/**
 * The files meta-cli.ts writes, and nothing else: meta.ts POSTERIOR_PATH, CALIBRATION_PATH, DIAGNOSTICS_PATH,
 * VERIFIED_PATH and POSTERIOR_INDEX_PATH; confirm.ts PROMOTIONS_PATH, confirmationPath(c) (c as meta-cli names them,
 * at most 6 digits) and GENOME_PATH (decision 0019 lets the gate promote it).
 */
export const META_PATH_RE = /^(data\/meta\/(posterior\.json|calibration\.json|diagnostics\.json|verified\.json|posterior-index\.jsonl|promotions\.jsonl|confirmations\/(0|[1-9][0-9]{0,5})\.json)|knowledge\/genome\.json)$/
const CONFIRMATION_RE = /^data\/meta\/confirmations\/(0|[1-9][0-9]{0,5})\.json$/
const GENOME_SPACE_PATH = "knowledge/genome-space.json"
/** The files the gate only ever appends lines to. */
const APPEND_ONLY = [PROMOTIONS_PATH, POSTERIOR_INDEX_PATH]

const BOT = "github-actions[bot]"
const BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
const POSTERIOR = "data/meta/posterior.json"

/** One message per staged path the gate may not commit; empty means every path is one of its files (or none is staged). */
export function metaPathErrors(staged: string[]): string[] {
  return staged.filter(p => !META_PATH_RE.test(p)).map(p => `${JSON.stringify(p)} is not a gate-computed data/meta file`)
}

const lines = (s: string, sep = "\n") => s.split(sep).filter(l => l !== "")
/** A file's text at a revision: `show(path)` is the index's (`git show :<path>`), `head(path)` HEAD's; null when absent. */
type Show = (path: string) => string | null
const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e))
/** Confirmation `path`'s text, parsed, or the error (one message) why not. */
function confirmationAt(path: string, text: string): ConfirmationFile | string {
  let c: ConfirmationFile
  try { c = parseConfirmation(text) } catch (e) { return `${path}: ${errorText(e)}` }
  if (confirmationPath(c.id) !== path) return `${path}: holds confirmation ${c.id}`
  return c
}
/** `text` as a JSON object, or null when it is not one. */
function jsonObject(text: string): Record<string, unknown> | null {
  try {
    const v: unknown = JSON.parse(text)
    return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null
  } catch { return null }
}
const withoutResult = (c: ConfirmationFile): ConfirmationFile => ({ ...c, result: null })

/**
 * One message per staged change that is not the gate's way of changing its record (spec §4.5; plan 05b Task 10):
 * - a confirmation file is written twice: declared (added, `result: null`, under its own id), then decided (only
 *   `result` changes, from null to a result); a file with a result never changes again;
 * - promotions.jsonl and posterior-index.jsonl only gain whole lines at their end.
 */
export function appendOnlyErrors(show: Show, staged: string[], head: Show): string[] {
  const errors: string[] = []
  for (const path of staged) {
    const now = show(path), before = head(path)
    if (now === null) continue // a deletion: the mode check refuses it
    if (CONFIRMATION_RE.test(path)) {
      const c = confirmationAt(path, now)
      if (typeof c === "string") { errors.push(c); continue }
      if (before === null) {
        if (c.result !== null) errors.push(`${path}: a new confirmation is declared with result null, not with a result`)
        continue
      }
      const old = confirmationAt(path, before)
      if (typeof old === "string") { errors.push(`main's ${old}`); continue }
      if (old.result !== null) errors.push(`${path}: already has a result (${old.result.outcome}): a decided confirmation never changes`)
      else if (!isDeepStrictEqual(withoutResult(old), withoutResult(c))) errors.push(`${path}: only its result may change once declared`)
      else if (c.result === null) errors.push(`${path}: staged without a change to its result`)
    } else if (APPEND_ONLY.includes(path)) {
      const base = before ?? ""
      if (!now.startsWith(base) || !(base === "" || base.endsWith("\n")) || !now.endsWith("\n")) {
        errors.push(`${path}: the gate may only append whole lines to it (an earlier line was rewritten or dropped, or a line is unfinished)`)
      }
    }
  }
  return errors
}

/**
 * Defence in depth for "knowledge/genome.json is gate-computed only" (decision 0019; spec §4.5): one message per way
 * the staged files are not exactly one promotion. When knowledge/genome.json is staged, all of these must hold:
 * - exactly one confirmation file is staged, and its result is a pass;
 * - genome.json reads (parseCentre, against main's genome space) as that confirmation's candidate, for its genome
 *   schema, with provenance `{ source: "promotion", confirmation: <its id> }`, and main's genome.json is its centre;
 * - promotions.jsonl gains exactly one line, naming that confirmation and its candidate's genomeSha.
 * Without genome.json, no staged confirmation may gain a pass (the pass would be lost) and promotions.jsonl may not
 * change. `show` gives the staged text (`git show :<path>`), `head` main's.
 */
export function promotionErrors(show: Show, staged: string[], head: Show): string[] {
  const confirmations = staged.filter(p => CONFIRMATION_RE.test(p))
  const parsed = confirmations.map(p => { const t = show(p); return t === null ? `${p}: deleted` : confirmationAt(p, t) })
  const passes = parsed.filter((c): c is ConfirmationFile => typeof c !== "string" && c.result?.outcome === "pass")
  const promotionsStaged = staged.includes(PROMOTIONS_PATH)
  if (!staged.includes(GENOME_PATH)) {
    const errors = passes.map(c => `${confirmationPath(c.id)}: a pass is staged without ${GENOME_PATH} (its promotion would be lost)`)
    if (promotionsStaged) errors.push(`${PROMOTIONS_PATH}: staged without ${GENOME_PATH} (a promotion line belongs to a promotion)`)
    return errors
  }
  const errors: string[] = []
  const bad = (why: string) => errors.push(`${GENOME_PATH}: ${why}`)
  if (confirmations.length !== 1) { bad(`a promotion stages exactly one confirmation file (the one it decides); staged: ${confirmations.length}`); return errors }
  const c = parsed[0]!
  if (typeof c === "string") { bad(c); return errors }
  if (c.result?.outcome !== "pass") { bad(`${confirmationPath(c.id)} is staged with result ${c.result?.outcome ?? "null"}, not a pass`); return errors }

  // The staged genome.json: the candidate, by parseCentre against main's genome space (the gate cannot stage that file).
  const text = show(GENOME_PATH), spaceText = head(GENOME_SPACE_PATH)
  if (text === null) { bad("deleted"); return errors }
  if (spaceText === null) { bad(`main has no ${GENOME_SPACE_PATH} to check it against`); return errors }
  let file: { genomeSchema?: unknown; provenance?: { source?: unknown; confirmation?: unknown } }, values: Record<string, unknown>
  try {
    file = JSON.parse(text) as typeof file
    const space = parseGenomeSpace(JSON.parse(spaceText))
    values = parseCentre(file, space)
  } catch (e) { bad(`does not read as a centre: ${errorText(e)}`); return errors }
  if (file.genomeSchema !== c.genomeSchema) bad(`genomeSchema ${String(file.genomeSchema)} is not confirmation ${c.id}'s ${c.genomeSchema}`)
  if (file.provenance?.source !== "promotion") bad(`provenance.source is ${JSON.stringify(file.provenance?.source)}, not "promotion"`)
  if (file.provenance?.confirmation !== c.id) bad(`provenance.confirmation is ${JSON.stringify(file.provenance?.confirmation)}, not ${c.id}`)
  if (!isDeepStrictEqual(values, c.candidate.values)) bad(`its values are not confirmation ${c.id}'s candidate's`)
  // What it replaces: main's genome.json must be the centre the confirmation compared against.
  const oldText = head(GENOME_PATH)
  let oldValues: unknown = null
  try { if (oldText !== null) oldValues = (JSON.parse(oldText) as { values?: unknown }).values } catch { /* compared as null below */ }
  if (!isDeepStrictEqual(oldValues, c.centre.values)) bad(`main's ${GENOME_PATH} is not confirmation ${c.id}'s centre: it promotes over a centre it did not test against`)

  // promotions.jsonl: exactly one new line (appendOnlyErrors checks the old ones are kept), naming c and its candidate.
  if (!promotionsStaged) { errors.push(`${PROMOTIONS_PATH}: not staged: a promotion appends exactly one line`); return errors }
  const added = lines(show(PROMOTIONS_PATH) ?? "").slice(lines(head(PROMOTIONS_PATH) ?? "").length)
  const line = added.length === 1 ? jsonObject(added[0]!) : null
  if (line === null) errors.push(`${PROMOTIONS_PATH}: a promotion appends exactly one JSON line; this appends ${added.length}`)
  else if (line.confirmation !== c.id || line.genomeSha !== c.candidate.genomeSha) {
    errors.push(`${PROMOTIONS_PATH}: the new line names confirmation ${JSON.stringify(line.confirmation)} and genome ${JSON.stringify(line.genomeSha)}, not ${c.id} and ${c.candidate.genomeSha}`)
  }
  return errors
}
/** The lockfile npm ci installed from: main's must still be it, or meta-cli ran main's new code on an old install. */
const LOCKFILE = "package-lock.json"
/** A push git refused because origin's main is ahead of it (not a protection rule, an auth failure or a network error). */
const isBehind = (e: unknown) => /\[rejected\][^\n]*\((fetch first|non-fast-forward)\)/.test(e instanceof Error ? e.message : String(e))

/**
 * What commitMeta did: pushed a commit that changes an output ("committed"), pushed one that changes verified.json
 * alone ("verified-only": a verification checkpoint, decision 0026, or a cache rebuilt with no output changed), or
 * pushed nothing ("unchanged"). gate.yml deploys on "committed" only: verified.json is the gate's cache, which
 * deploy.yml never copies, so a verified-only commit leaves every deployed byte as it was (commit-meta.test.ts checks
 * that deploy.yml does not name it) and a deploy for it would spend a runner and Pages API calls on nothing.
 */
export type MetaResult = "committed" | "verified-only" | "unchanged"

/**
 * Stages data/meta and knowledge/genome.json, refuses anything but the gate's own files (appended to as the gate
 * appends, and a genome.json only as one promotion), commits them as the Actions bot and pushes HEAD to main.
 * A push rejected because main moved: fetch, hard reset to origin/main, drop untracked data/meta files, rebuild()
 * (meta-cli.ts on the new main) and try again, at most `maxTries` pushes in all. `git` must throw with git's stderr
 * in the message when git fails. Throws, committing nothing, if the tree holds any other change, or if HEAD's
 * package-lock.json is not `installedLock` (the blob sha npm ci installed from) at any try.
 */
export async function commitMeta(
  deps: { git: (args: string[]) => string; installedLock: string; rebuild(): Promise<boolean>; log(m: string): void },
  maxTries = 3,
): Promise<MetaResult> {
  const { git, log } = deps
  if (!Number.isSafeInteger(maxTries) || maxTries < 1) throw new Error(`not a number of tries: ${maxTries}`)
  if (!/^[0-9a-f]{40}([0-9a-f]{24})?$/.test(deps.installedLock)) throw new Error(`not a lockfile blob sha: ${JSON.stringify(deps.installedLock.slice(0, 80))}`)
  // node_modules is ignored, so no status check sees an install older than the code: compare the lockfile itself.
  const checkLock = () => {
    const lock = git(["rev-parse", `HEAD:${LOCKFILE}`]).trim()
    if (lock !== deps.installedLock) {
      throw new Error(`main's ${LOCKFILE} (${lock.slice(0, 12)}) is not the one npm ci installed (${deps.installedLock.slice(0, 12)}): refusing to build or commit data/meta on an old install; the next run reinstalls`)
    }
  }
  for (let attempt = 1; ; attempt++) {
    // The commit goes on top of main as fetched, never on a local commit no one has checked.
    const head = git(["rev-parse", "HEAD"]).trim(), main = git(["rev-parse", "refs/remotes/origin/main"]).trim()
    if (head !== main) throw new Error(`HEAD ${head.slice(0, 12)} is not origin/main ${main.slice(0, 12)}: refusing to commit on top of it`)
    checkLock()

    // 1. Stage only data/meta and genome.json; every staged path must be one of the gate's files.
    git(["add", "--all", "--", ":(top,literal)data/meta", `:(top,literal)${GENOME_PATH}`])
    const staged = lines(git(["diff", "--cached", "--name-only", "--no-renames", "-z"]), "\0")
    const errors = metaPathErrors(staged)
    if (errors.length > 0) throw new Error(`refusing to commit: ${errors.join("; ")}`)
    // Each staged change adds or modifies a regular, non-executable file: no deletion, no symlink, no gitlink, no mode change.
    const raw = lines(git(["diff", "--cached", "--raw", "--no-renames", "--no-abbrev", "-z"]), "\0")
    for (let i = 0; i < raw.length; i += 2) {
      const [oldMode, newMode, , , status] = raw[i]!.slice(1).split(" ")
      // knowledge/genome.json always exists on main: the gate only ever replaces it.
      const added = status === "A" && oldMode === "000000" && raw[i + 1] !== GENOME_PATH
      const ok = newMode === "100644" && (added || (status === "M" && oldMode === "100644"))
      if (!ok) throw new Error(`refusing to commit: ${JSON.stringify(raw[i + 1])} is ${status} ${oldMode} → ${newMode}, not a regular file added or modified${raw[i + 1] === GENOME_PATH ? " (it may only be modified)" : ""}`)
    }

    // 2. Nothing else may differ from HEAD: no unstaged edit, no untracked file (npm ci must touch nothing).
    const other = lines(git(["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames"]), "\0")
      .filter(e => !(/^[AM] $/.test(e.slice(0, 2)) && META_PATH_RE.test(e.slice(3))))
    if (other.length > 0) throw new Error(`refusing to commit: the tree has other changes: ${other.map(e => JSON.stringify(e)).join(", ")}`)

    if (staged.length === 0) { log("data/meta unchanged: nothing to commit"); return "unchanged" }

    // 3. The gate's record only grows, and knowledge/genome.json changes only as one promotion it decided.
    const show: Show = p => showAt(git, `:${p}`), atHead: Show = p => showAt(git, `HEAD:${p}`)
    const recordErrors = [...appendOnlyErrors(show, staged, atHead), ...promotionErrors(show, staged, atHead)]
    if (recordErrors.length > 0) throw new Error(`refusing to commit: ${recordErrors.join("; ")}`)

    // 4. Commit as the Actions bot (author and committer).
    let message: string
    const verifiedOnly = staged.length === 1 && staged[0] === VERIFIED_PATH
    if (verifiedOnly) {
      // verified.json alone: a verification checkpoint (decision 0026: every output stays main's until verification
      // completes, and main may have no posterior.json yet), or a cache rebuilt with no output changed. Its message
      // names the verdicts, not a posterior this commit does not change.
      const lives = jsonObject(show(VERIFIED_PATH) ?? "")?.lives
      if (typeof lives !== "object" || lives === null) throw new Error(`${VERIFIED_PATH} has no lives`)
      message = `meta: verified.json only, ${Object.keys(lives).length} lives verified; outputs unchanged`
    } else {
      const posteriorText = git(["show", `:${POSTERIOR}`])
      const posterior = JSON.parse(posteriorText) as { mode?: unknown; page?: { eligibleLives?: unknown } }
      const n = posterior.page?.eligibleLives
      if (typeof posterior.mode !== "string" || !Number.isSafeInteger(n)) throw new Error(`${POSTERIOR} has no mode or page.eligibleLives`)
      const sha8 = createHash("sha256").update(posteriorText, "utf8").digest("hex").slice(0, 8)
      message = `meta: ${posterior.mode}, ${n} eligible lives, posterior ${sha8}`
    }
    for (const p of staged.filter(x => CONFIRMATION_RE.test(x))) {
      const c = parseConfirmation(show(p)!)
      message += atHead(p) === null ? `; declare confirmation ${c.id}` : `; confirmation ${c.id} ${c.result!.outcome}`
      if (staged.includes(GENOME_PATH) && c.result?.outcome === "pass") message += `; promote ${c.candidate.genomeSha.slice(0, 8)}`
    }
    git(["-c", `user.name=${BOT}`, "-c", `user.email=${BOT_EMAIL}`, "-c", "commit.gpgsign=false",
      "commit", "-q", "--no-verify", `--author=${BOT} <${BOT_EMAIL}>`, "-m", message])
    log(`committed ${message} (${staged.join(", ")})`)

    // 5. Push, never forced.
    try {
      git(["push", "origin", "HEAD:main"])
      log(`pushed to main (try ${attempt})`)
      return verifiedOnly ? "verified-only" : "committed"
    } catch (e) {
      if (!isBehind(e)) throw e
      if (attempt >= maxTries) throw new Error(`push rejected ${attempt} times because main kept moving (${maxTries} tries): giving up, the next run rebuilds`, { cause: e })
      log(`push rejected (main moved, try ${attempt}): resetting to the new main and rebuilding`)
    }
    git(["fetch", "-q", "origin", "main"])
    git(["reset", "-q", "--hard", "refs/remotes/origin/main"])
    git(["clean", "-q", "-f", "-d", "--", ":(top,literal)data/meta"]) // an untracked file of the old build is stale too (genome.json is tracked: reset restored it)
    checkLock() // before running the new main's meta-cli at all
    if (!(await deps.rebuild())) { log("the rebuild on the new main changed nothing"); return "unchanged" }
  }
}

/** `git show <spec>`'s text, or null when the path is not there (a file the commit adds, or one main never had). */
function showAt(git: (args: string[]) => string, spec: string): string | null {
  const [rev, path] = [spec.slice(0, spec.indexOf(":")), spec.slice(spec.indexOf(":") + 1)]
  const present = rev === "" ? git(["ls-files", "--cached", "-z", "--", `:(top,literal)${path}`]) : git(["ls-tree", "-z", "--name-only", rev, "--", `:(top,literal)${path}`])
  return present === "" ? null : git(["show", spec])
}

/** git in the current directory; a failure's message carries git's stderr (commitMeta reads it to tell a rejected push). */
function runGit(args: string[]): string {
  try {
    return execFileSync("git", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"] })
  } catch (e) {
    const err = e as { stderr?: string | Buffer; message: string }
    throw new Error(`git ${args.join(" ")} failed: ${String(err.stderr ?? "").trim() || err.message}`, { cause: e })
  }
}

/** meta-cli.ts, run afresh in this checkout (the new main's code, config and logs); its `changed=` line is the answer. */
async function rebuildMeta(): Promise<boolean> {
  const out = execFileSync("npx", ["tsx", "tools/gate/meta-cli.ts"], { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] })
  const m = /^changed=(true|false)$/m.exec(out)
  if (!m) throw new Error(`meta-cli.ts printed no changed= line: ${JSON.stringify(out.slice(-200))}`)
  return m[1] === "true"
}

async function main(): Promise<void> {
  // gate.yml records it right after npm ci (git hash-object package-lock.json).
  const installedLock = process.env.LOCK_BLOB
  if (!installedLock) throw new Error("LOCK_BLOB env var is required: the blob sha of the package-lock.json npm ci installed from")
  const result = await commitMeta({ git: runGit, installedLock, rebuild: rebuildMeta, log: m => console.log(m) })
  console.log(`meta=${result}`)
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `meta=${result}\n`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
