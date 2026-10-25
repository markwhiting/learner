// validate.yml entry point. Runs main's copy of this script; PR files are read with git (a partial fetch of
// refs/pull/N/head; github.ts loadPrViaGit) as data, never executed.
// Env: REPO (owner/name), PR_NUMBER, GH_TOKEN. cwd: a full-history checkout of main (for data/runs/<id>/ previous state).
import { join } from "node:path"
import { validateSessionPr } from "./validate.ts"
import { readLearnerLogs } from "./summary.ts"
import { git, isShallow, loadPrViaGit } from "./github.ts"
import { mainSpaces } from "./gate.ts"

const repo = process.env.REPO
const n = Number(process.env.PR_NUMBER)
if (!repo || !Number.isInteger(n) || n <= 0) { console.error("REPO and PR_NUMBER env vars are required"); process.exit(2) }
if (isShallow()) { console.error("::error::the checkout of main is shallow (validate.yml needs fetch-depth 0)"); process.exit(2) }
// Every genome space main has had (each genome is checked against its own schema's), before anything is read from the
// PR: without main's current one no meta genome can be checked.
const spaces = (() => {
  try { return mainSpaces(".", git) } catch (e) { console.error(`::error::${e instanceof Error ? e.message : e}`); return process.exit(2) }
})()

const loaded = loadPrViaGit(repo, n)
// Neutral: a newer push started its own run (concurrency cancels this one), and this check is advisory; the gate
// validates every head itself before merging, pinned to it. This run's commit is no longer the PR's head.
if (loaded.moved) { console.log(`::warning::PR #${n}'s head moved while this run read it: not validated here; the run for the new head validates it`); process.exit(0) }
// Fails: this is the PR's current head, and it could not be validated against the checkout (re-run the check).
if (loaded.stale) { console.error(`::error::PR #${n} is based on a main newer than this run's checkout; re-run this check`); process.exit(1) }
const ids = [...new Set(loaded.files.map(f => /^data\/runs\/(lrn_[0-9a-f]{16})\//.exec(f.path)?.[1]).filter((x): x is string => !!x))]
const previous = ids.flatMap(id => { try { return readLearnerLogs(join("data/runs", id)) } catch { return [] } })
const v = await validateSessionPr(loaded.files, previous, { spaces })
const errors = [...(loaded.baseRef === "main" ? [] : ["Session PRs must target main."]), ...loaded.loadErrors, ...v.errors]
if (errors.length > 0) {
  for (const e of errors) console.error(`::error::${e.replace(/[\r\n]+/g, " ").slice(0, 500)}`)
  process.exit(1)
}
console.log(`PR #${n}: ${loaded.files.length} file(s) valid`)
