// Usage: tsx tools/sim/meta-sim.ts <dir> [--quick] [--flat]   (dir: empty or absent; the fixture repo is kept there)
//
// The meta-learner end to end on a local fixture repo (plan 05b Task 12): a bare "origin" holding a copy of main's
// knowledge/, config/ and (full run) data/meta/ with an empty data/runs/, and the gate's checkout of it. Each stage
// merges synthetic lives into main, written as a session writes them (birthDraw under main's posterior, the decoded
// genome as settings, compact JSON; each checked with validateSessionPr against the fixture's genome spaces), then
// runs a gate pass: this checkout's real tools/gate/meta-cli.ts and tools/gate/commit-meta.ts, in the fixture.
//
// Search mode is reached by data only (ruling 1): planted calibration lives go through the real signal-check@1 in
// meta-cli. Nothing here, and no flag or environment variable, can set a mode. `--flat` plants lives whose scores
// do not depend on their genome: the signal check finds none, and the run must stay inert to the end.
//
// `--quick` (CI): a reduced test space, main's with only its first QUICK_D searched genes left to the GP (a recorded
// calibration restart in the real repo; here a fresh calibration). The full run uses main's space and data/meta.
import { execFileSync, spawnSync } from "node:child_process"
import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"
import { CONFIG_BUDGET, NEWBORN_LEVELS, type LogFileV2, type MetaBlock } from "@blanktab/harness/log"
import {
  birthDraw, genomeSha, modelPosterior, parseCentre, parseGenomeSpace, parsePosterior, sha256Hex, toUnit,
  type ArmName, type Genome, type GenomeSpace, type GpPosterior, type Mode, type Posterior,
} from "@blanktab/meta"
import { checkup, lid, scoredLife } from "../../packages/evaluators/test/fixtures/build.ts"
import { META_PATH_RE } from "../gate/commit-meta.ts"
import { CONFIRM_TARGET, parseConfirmation } from "../gate/confirm.ts"
import { mainSpaces } from "../gate/gate.ts"
import { computeKnowledgeSha } from "../gate/manifest.ts"
import { validateSessionPr } from "../gate/validate.ts"
import { makeFixture } from "../gate/test/git-fixture.ts"
import { bornLife, noise, planted, sessionSettings, useRunsDir, type Born, type Effect } from "../gate/test/meta-fixture.ts"

const REPO = fileURLToPath(new URL("../..", import.meta.url))
const TSX = join(REPO, "node_modules/.bin/tsx")
/** Searched genes in the quick run's reduced space. */
export const QUICK_D = 3
/** Rows per search batch (one merge, one gate run), and the most search rows before a declaration must have come. */
const SEARCH_BATCH = 50, SEARCH_CAP = 400
/** The most calibration lives before the signal check must have answered (its looks are at 30, 45 and 60). */
const CALIBRATION_CAP = 60
const BOT = "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"

export interface SimStage { name: string; mode: Mode; commit: string; paths: string[] }
export interface SimReport { stages: SimStage[]; promoted: boolean; knowledgeShaBefore: string; knowledgeShaAfter: string }
export interface SimOptions {
  /** Where the fixture repo is built (empty or absent): <dir>/origin.git is main, <dir>/work the gate's checkout. */
  dir: string
  quick: boolean
  log(m: string): void
  /** The calibration lives' scores: "planted" (a genome effect far above the noise; the default) or "noise" (none). */
  calibration?: "planted" | "noise"
}

/** Main's genome space with only its first QUICK_D searched genes left searched (the others fixed at their centres). */
export function quickSpaceText(text: string): string {
  const v = JSON.parse(text) as { genes: { owner: string }[] }
  let searched = 0
  for (const g of v.genes) if (g.owner === "gp" && ++searched > QUICK_D) g.owner = "fixed"
  return JSON.stringify(v, null, 1) + "\n"
}

/** git config isolated from the user's and the system's; no identity, so commit-meta's bot identity is the one used. */
function isolatedEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" }
  for (const k of ["GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GITHUB_OUTPUT"]) delete env[k]
  return env
}

const fail = (why: string): never => { throw new Error(`meta-sim: ${why}`) }
/** Every file under `rel` in this checkout, by repo path. */
function filesUnder(rel: string): Record<string, string> {
  const out: Record<string, string> = {}
  const abs = join(REPO, rel)
  if (!existsSync(abs)) return out
  for (const e of readdirSync(abs, { withFileTypes: true })) {
    const p = `${rel}/${e.name}`
    if (e.isDirectory()) Object.assign(out, filesUnder(p))
    else out[p] = readFileSync(join(REPO, p), "utf8")
  }
  return out
}

/** Scores that follow the first searched gene (a smooth optimum at its top end), far above the repeat noise. */
const byGenome = (space: GenomeSpace): Effect => (k, _i, g) => {
  const u = toUnit(space, g)[0]!
  return { m2: 0.3 + 3 * u + 0.01 * (k % 3), ratio: 0.98 - 0.08 * u + 0.001 * (k % 3) }
}

export async function runSim(opts: SimOptions): Promise<SimReport> {
  const { log } = opts
  const env = isolatedEnv()
  const fx = makeFixture(opts.dir)
  const work = fx.work
  const origin = (...args: string[]) => execFileSync("git", ["--git-dir", join(opts.dir, "origin.git"), ...args], { encoding: "utf8", env }).trim()
  const inWork = (args: string[]) => execFileSync("git", args, { cwd: work, encoding: "utf8", env, maxBuffer: 64 * 1024 * 1024 })
  const scratch = join(opts.dir, "scratch")
  mkdirSync(scratch, { recursive: true })
  useRunsDir(scratch) // bornLife's own copies; main gets the session's bytes below

  // main: knowledge/, config/ and (full run) data/meta/ as this checkout has them; an empty data/runs/.
  const knowledge = filesUnder("knowledge")
  if (opts.quick) knowledge["knowledge/genome-space.json"] = quickSpaceText(knowledge["knowledge/genome-space.json"]!)
  fx.commitMain({
    ...knowledge, ...filesUnder("config"), ...(opts.quick ? {} : filesUnder("data/meta")),
    "package-lock.json": "{\"lockfileVersion\":3}\n", // the fixture's lockfile: LOCK_BLOB below is its blob
  })
  const lockBlob = inWork(["rev-parse", "HEAD:package-lock.json"]).trim()
  const space = parseGenomeSpace(JSON.parse(readFileSync(join(work, "knowledge/genome-space.json"), "utf8")))
  const spaces = mainSpaces(work, inWork)
  const firstCentre = parseCentre(JSON.parse(readFileSync(join(work, "knowledge/genome.json"), "utf8")), space)
  const knowledgeShaBefore = computeKnowledgeSha(join(work, "knowledge"))
  log(`fixture at ${opts.dir}: ${opts.quick ? `quick (${QUICK_D} searched genes)` : "full"} space, main ${origin("rev-parse", "main").slice(0, 8)}`)

  /** origin's main moved only forward from `before` (never forced past). */
  const forwardOnly = (before: string, what: string) => {
    const after = origin("rev-parse", "main")
    if (after !== before && spawnSync("git", ["--git-dir", join(opts.dir, "origin.git"), "merge-base", "--is-ancestor", before, after], { env }).status !== 0) {
      fail(`${what}: main ${before.slice(0, 8)} is not an ancestor of ${after.slice(0, 8)} (a forced push)`)
    }
    return after
  }
  const tsx = (script: string, extra: NodeJS.ProcessEnv = {}) => {
    const r = spawnSync(TSX, [join(REPO, script)], { cwd: work, env: { ...env, ...extra }, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 })
    if (r.status !== 0) fail(`${script} exited ${r.status}: ${(r.stderr || r.stdout).trim().slice(-2000)}`)
    return { out: r.stdout, err: r.stderr }
  }
  const mainText = (path: string) => readFileSync(join(work, path), "utf8")
  const posteriorNow = (): { text: string; p: Posterior } => {
    const text = mainText("data/meta/posterior.json")
    const p = parsePosterior(text)
    return p === null ? fail("main's posterior does not parse") : { text, p }
  }

  // ---- Lives, as a session writes them, merged into main ----
  let nextK = 0x200000
  let checkedLives = 0 // every life here is a draw (role search or confirm): the gate verifies each one
  const withBudget = (m: MetaBlock): MetaBlock => ({ ...m, budget: { budgetCU: CONFIG_BUDGET.budgetCU, levels: [...CONFIG_BUDGET.levels] } })
  /** A life born under main's posterior bytes `text` by a tab holding centre `centre`. */
  const birth = (text: string, effect: Effect, more: { centre?: Genome; k?: number; passed?: [["M2"], ["M2"], ["M2"]] } = {}) =>
    bornLife(text, { k: more.k ?? nextK++, effect, space, centre: more.centre ?? centreNow(), session: true, alter: withBudget, ...(more.passed ? { passed: more.passed } : {}) })
  /** Merges `lives` (each its files) into main as one commit, after validateSessionPr accepts each life's files. */
  const merge = async (what: string, lives: LogFileV2[][]) => {
    const files: Record<string, string> = {}
    for (const life of lives) {
      const pr = life.map(f => ({ path: `data/runs/${f.learnerId}/${f.seq}.json`, content: JSON.stringify(f) }))
      const v = await validateSessionPr(pr, [], { spaces })
      if (!v.ok) fail(`${what}: ${life[0]!.learnerId} is refused by validateSessionPr: ${v.errors.join("; ")}`)
      for (const f of pr) files[f.path] = f.content
    }
    const before = origin("rev-parse", "main")
    fx.commitMain(files)
    forwardOnly(before, `${what}: merge`)
    checkedLives += lives.length
  }
  const centreNow = (): Genome => parseCentre(JSON.parse(mainText("knowledge/genome.json")), space)

  // ---- A gate pass: meta-cli, then commit-meta, then the checks every stage must pass ----
  const stages: SimStage[] = []
  const gate = async (name: string): Promise<SimStage> => {
    const t0 = Date.now()
    const before = origin("rev-parse", "main")
    const built = tsx("tools/gate/meta-cli.ts")
    const committed = tsx("tools/gate/commit-meta.ts", { LOCK_BLOB: lockBlob })
    const commit = forwardOnly(before, `${name}: gate`)
    if (inWork(["rev-parse", "HEAD"]).trim() !== commit) fail(`${name}: the gate's checkout is not main after commit-meta`)
    const moved = commit !== before
    if (moved !== /^meta=committed$/m.test(committed.out)) fail(`${name}: commit-meta said ${committed.out.trim()}, main ${moved ? "moved" : "did not move"}`)
    const paths = moved ? origin("diff-tree", "--no-commit-id", "--name-only", "-r", commit).split("\n").filter(Boolean) : []
    const stray = paths.filter(p => !META_PATH_RE.test(p))
    if (stray.length > 0) fail(`${name}: the gate committed ${stray.join(", ")}`)
    if (moved && origin("log", "-1", "--format=%an <%ae>|%cn <%ce>", commit) !== `${BOT}|${BOT}`) fail(`${name}: the gate's commit is not the bot's`)
    const { p } = posteriorNow()
    // Every life verifies: each draw recomputes (ts births through the served model) and none is flagged.
    const diag = JSON.parse(mainText("data/meta/diagnostics.json")) as { verify: { checked: number; ineligible: unknown[]; genomeMismatch: { count: number } }; model?: unknown }
    if (diag.verify.checked !== checkedLives) fail(`${name}: the gate verified ${diag.verify.checked} lives, main has ${checkedLives}`)
    if (diag.verify.ineligible.length > 0 || diag.verify.genomeMismatch.count > 0) fail(`${name}: lives did not verify: ${JSON.stringify(diag.verify).slice(0, 2000)}`)
    // The mode guard, from the outside: nothing of the search exists outside mode search.
    if (p.mode !== "search" && (Object.keys(p.models).length > 0 || p.incumbent !== null || p.confirmation !== null || diag.model !== undefined)) {
      fail(`${name}: mode ${p.mode} with a model, an incumbent, a confirmation or a fit`)
    }
    const fit = /buildMeta took ([0-9.]+) s/.exec(built.err)?.[1] ?? "?"
    log(`${name.padEnd(18)} ${p.mode.padEnd(9)} ${commit.slice(0, 8)} ${String(p.page.eligibleLives).padStart(4)} rows  buildMeta ${fit} s  stage ${((Date.now() - t0) / 1000).toFixed(1)} s  ${paths.join(" ")}`)
    const stage = { name, mode: p.mode, commit, paths }
    stages.push(stage)
    return stage
  }

  // 1. Start: the first gate run on main's state.
  await gate("start")

  // 2. Calibrate: lives under main's posterior until the signal check has answered (look 1 at 30 lives, then 45, 60).
  const calibrationEffect = opts.calibration === "noise" ? noise : planted
  for (let look = 1; posteriorNow().p.mode === "calibrate"; look++) {
    const { text, p } = posteriorNow()
    if (p.page.calibrationLives >= CALIBRATION_CAP) fail(`${CALIBRATION_CAP} calibration lives and the signal check has not answered`)
    const want = Math.max(1, p.page.calibrationTarget - p.page.calibrationLives)
    const lives: Born[] = []
    for (let i = 0; i < want; i++) lives.push(await birth(text, calibrationEffect))
    if (lives.some(b => b.meta.method !== "calibrate")) fail(`look ${look}: a birth under mode calibrate was not a calibrate draw`)
    await merge(`calibrate ${look}`, lives.map(b => b.files))
    await gate(`calibrate ${look}`)
  }

  if (posteriorNow().p.mode === "flat") {
    // Flat: births are uniform and centre; the gate fits nothing and declares nothing, to the end.
    const { text } = posteriorNow()
    const lives: Born[] = []
    for (let i = 0; i < 20; i++) lives.push(await birth(text, byGenome(space)))
    const methods = new Set(lives.map(b => b.meta.method))
    if ([...methods].some(m => m !== "uniform" && m !== "centre")) fail(`a birth under mode flat drew ${[...methods].join(", ")}`)
    await merge("after flat", lives.map(b => b.files))
    await gate("after flat")
    const knowledgeShaAfter = computeKnowledgeSha(join(work, "knowledge"))
    if (knowledgeShaAfter !== knowledgeShaBefore) fail("knowledge changed under mode flat")
    return { stages, promoted: false, knowledgeShaBefore, knowledgeShaAfter }
  }
  if (posteriorNow().p.mode !== "search") fail(`the signal check left mode ${posteriorNow().p.mode}`)

  // 3. Search: ts and uniform lives on a planted smooth optimum, until the trigger declares confirmation 0.
  const memo = new Map<string, GpPosterior | null>()
  const conditionedFor = (sha: string, p: Posterior) => () => {
    if (!memo.has(sha)) memo.set(sha, p.models.gpu ? modelPosterior(p.models.gpu) : null)
    return memo.get(sha)!
  }
  let searchRows = 0
  for (let batch = 1; posteriorNow().p.confirmation === null; batch++) {
    if (searchRows >= SEARCH_CAP) fail(`${SEARCH_CAP} search rows and no confirmation declared`)
    const { text, p } = posteriorNow()
    if (!p.models.gpu) fail(`search ${batch}: mode search serves no model (${JSON.stringify(p.diagnostics).slice(0, 300)})`)
    const lives: Born[] = []
    for (let i = 0; i < SEARCH_BATCH; i++) lives.push(await birth(text, byGenome(space)))
    const methods = lives.map(b => b.meta.method)
    if (!methods.includes("ts")) fail(`search ${batch}: no ts birth among ${SEARCH_BATCH}`)
    if (methods.some(m => m !== "ts" && m !== "uniform")) fail(`search ${batch}: a birth drew ${methods.find(m => m !== "ts" && m !== "uniform")}`)
    searchRows += lives.length
    await merge(`search ${batch}`, lives.map(b => b.files))
    await gate(`search ${batch}`)
  }

  // 4. Confirm: confirmation lives of all three arms, as tabs served the confirmation draw them (each birth in
  // learnerId order, kept while its arm wants more, one per evalSeed; other births are not merged). The candidate's
  // lives sustain M2 above the centre's (a planted advantage); a newborn-arm life is one frozen file.
  const c0 = parseConfirmation(mainText("data/meta/confirmations/0.json"))
  const centreBefore = centreNow()
  const seeds = new Set<number>()
  const confirmBatch = async (name: string, want: Record<ArmName, number>) => {
    const { text, p } = posteriorNow()
    const sha = await sha256Hex(new TextEncoder().encode(text))
    const centreSha = await genomeSha(space.schemaVersion, centreBefore)
    const made: Record<ArmName, LogFileV2[][]> = { candidate: [], centre: [], newborn: [] }
    const left = () => (["candidate", "centre", "newborn"] as const).some(a => made[a].length < want[a])
    for (let tries = 0; left(); tries++) {
      if (tries > 20_000) fail(`${name}: 20,000 births did not fill the arms`)
      const k = nextK++
      const d = birthDraw({ learnerId: lid(0x10000 + k), space, centre: centreBefore, centreSha, widthsClass: "gpu", posterior: p, posteriorSha: sha, conditioned: conditionedFor(sha, p) })
      if (d.method !== "confirm") continue
      const arm = d.confirmation!.arm
      if (made[arm].length >= want[arm] || seeds.has(d.evalSeed!)) continue
      seeds.add(d.evalSeed!)
      const strong = arm === "candidate"
      const b = await birth(text, () => (strong ? { m2: 3, ratio: 0.9 } : { m2: 0.3, ratio: 0.98 }), { k, centre: centreBefore, ...(strong ? { passed: [["M2"], ["M2"], ["M2"]] as [["M2"], ["M2"], ["M2"]] } : {}) })
      if (b.meta.method !== "confirm" || b.meta.confirmation?.arm !== arm) fail(`${name}: ${b.learnerId} drew ${b.meta.method} on its own`)
      if (arm !== "newborn") { made[arm].push(b.files); continue }
      // A confirmation newborn, as a tab writes it: one frozen file (decision 0024 "Confirmation-newborn files").
      const one = scoredLife(b.learnerId, { ms: {}, ratio: 1, meta: b.meta })[0]!
      const frozen: LogFileV2 = {
        ...one, seq: 0, cu: 0, stepsTotal: 0, final: true, curve: [], settings: sessionSettings(space, b.meta, k),
        checkups: NEWBORN_LEVELS.map(level => ({ ...checkup(level, { M2: 1 }), steps: 0 })), milestoneSuiteVersion: 1,
        meta: { ...b.meta, heldOut: { loss: 1, baseline: 1 } },
      }
      made.newborn.push([frozen])
    }
    await merge(name, [...made.candidate, ...made.centre, ...made.newborn])
    return gate(name)
  }
  const half = (n: number) => Math.floor(n / 2)
  const first = await confirmBatch("confirm 1", { candidate: half(CONFIRM_TARGET.candidate), centre: half(CONFIRM_TARGET.centre), newborn: half(CONFIRM_TARGET.newborn) })
  if (first.paths.includes("knowledge/genome.json")) fail("confirm 1: promoted on half the arms")
  const served = posteriorNow().p.confirmation
  if (served === null || served.id !== 0 || served.counted.candidate !== half(CONFIRM_TARGET.candidate)) fail(`confirm 1: served ${JSON.stringify(served)}`)
  const rest = served!.need
  const decided = await confirmBatch("confirm 2", rest)

  // 5. Promote: the pass commits knowledge/genome.json with its result, and the knowledge the page deploys changes.
  const result = parseConfirmation(mainText("data/meta/confirmations/0.json")).result
  if (result?.outcome !== "pass") fail(`confirm 2: confirmation 0 decided ${result?.outcome ?? "nothing"}`)
  for (const p of ["knowledge/genome.json", "data/meta/confirmations/0.json", "data/meta/promotions.jsonl"]) {
    if (!decided.paths.includes(p)) fail(`confirm 2: the promotion commit lacks ${p}`)
  }
  const knowledgeShaAfter = computeKnowledgeSha(join(work, "knowledge"))
  if (knowledgeShaAfter === knowledgeShaBefore) fail("the promotion left knowledgeSha unchanged")
  const newCentre = centreNow()
  if (await genomeSha(space.schemaVersion, newCentre) !== c0.candidate.genomeSha) fail("knowledge/genome.json is not confirmation 0's candidate")

  // 6. After the promotion: the posterior is built for the new centre. Tabs on the new genome.json draw from it; tabs
  // still holding the old one draw the old centre ("posterior built for another centre"). Every life still verifies.
  const { text, p } = posteriorNow()
  if (p.centre.genomeSha !== c0.candidate.genomeSha || p.incumbent === null || !p.models.gpu) fail("the posterior after the promotion is not built for the new centre")
  const rowsBefore = p.page.eligibleLives
  const lives: Born[] = []
  for (let i = 0; i < 20; i++) lives.push(await birth(text, byGenome(space)))
  for (let i = 0; i < 10; i++) lives.push(await birth(text, byGenome(space), { centre: firstCentre }))
  const old = lives.slice(20)
  if (!old.every(b => b.meta.method === "centre" && b.draw.reason === "posterior built for another centre")) fail("an old-centre tab did not draw its own centre")
  await merge("after promotion", lives.map(b => b.files))
  await gate("after promotion")
  const after = posteriorNow().p
  if (after.page.eligibleLives !== rowsBefore + lives.length) fail(`after promotion: ${after.page.eligibleLives} rows, expected ${rowsBefore + lives.length} (old-centre lives must stay rows)`)
  if (computeKnowledgeSha(join(work, "knowledge")) !== knowledgeShaAfter) fail("knowledge changed after the promotion")
  return { stages, promoted: true, knowledgeShaBefore, knowledgeShaAfter }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const dir = args.find(a => !a.startsWith("--"))
  if (!dir) { console.error("usage: meta-sim.ts <dir> [--quick] [--flat]"); process.exit(2) }
  const t0 = Date.now()
  const r = await runSim({ dir, quick: args.includes("--quick"), calibration: args.includes("--flat") ? "noise" : "planted", log: m => console.log(m) })
  console.log(`${r.stages.length} gate runs in ${((Date.now() - t0) / 1000).toFixed(1)} s; promoted ${r.promoted}; knowledgeSha ${r.knowledgeShaBefore.slice(0, 12)} → ${r.knowledgeShaAfter.slice(0, 12)}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
