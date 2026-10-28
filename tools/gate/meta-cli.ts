// Usage: tsx tools/gate/meta-cli.ts   (run from the root of a full-history checkout of main, after the gate's merges)
// Builds data/meta/** (posterior, calibration, diagnostics, verified, confirmations) from main's merged logs and pinned
// config (buildMeta), writes the files that changed, appends the index line when the posterior's bytes changed and the
// promotion line on a pass, and prints `changed=true|false`. It reads config/ and never writes it. knowledge/genome.json
// it writes only on a confirmation's pass, with the gate-computed content buildMeta returns (spec §4.5); committing is
// commit-meta.ts's. It prints the build's time on stderr, so a slow fit shows in the gate's log.
// A build whose verification is incomplete (VERIFY_BUDGET; decision 0026) writes verified.json only: every other file
// stays exactly as main has it until a later run completes verification.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { genomeSha, parseCentre } from "@blanktab/meta"
import { loadEvalConfig, loadReferences } from "../eval/config.ts"
import { loadGenomeSpace, mainSpaces } from "./gate.ts"
import { git, isShallow } from "./github.ts"
import { confirmationPath, GENOME_PATH, parseConfirmation, PROMOTIONS_PATH, type ConfirmationFile } from "./confirm.ts"
import { centresBySha, RUNS_DIR, runArrivals } from "./history.ts"
import {
  buildOrCheckpoint, CALIBRATION_PATH, DIAGNOSTICS_PATH, parseCalibration, parseVerified, POSTERIOR_INDEX_PATH, POSTERIOR_PATH, posteriorHistory,
  VERIFIED_PATH, VERIFY_BUDGET, wantedPosteriors,
} from "./meta.ts"

const root = git(["rev-parse", "--show-toplevel"]).trim()
// Without the full history, posteriorHistory misses old posteriors and every draw made under one would be flagged.
if (isShallow()) { console.error("::error::the checkout is shallow (meta-cli needs main's full history)"); process.exit(2) }
const read = (rel: string): string | null => (existsSync(join(root, rel)) ? readFileSync(join(root, rel), "utf8") : null)

const config = loadEvalConfig(root)
const space = loadGenomeSpace(root)
const centre = parseCentre(JSON.parse(readFileSync(join(root, "knowledge/genome.json"), "utf8")), space)
const calibrationText = read(CALIBRATION_PATH)
const posteriorText = read(POSTERIOR_PATH)
const verifiedText = read(VERIFIED_PATH)
// data/meta/confirmations/<c>.json: exactly 0.json .. (n−1).json, each its own id, or the gate stops (writing nothing).
// Nothing else belongs in that directory: a stray entry is named in the error, never skipped silently.
const confirmationsDir = dirname(confirmationPath(0))
const entries = existsSync(join(root, confirmationsDir)) ? readdirSync(join(root, confirmationsDir)).sort() : []
const stray = entries.filter(n => !/^(0|[1-9][0-9]*)\.json$/.test(n))
if (stray.length > 0) throw new Error(`${confirmationsDir}: stray entries ${stray.map(n => JSON.stringify(n)).join(", ")} (only <c>.json confirmation files belong there)`)
const confirmations: ConfirmationFile[] = entries.map((_, i) => {
  const rel = confirmationPath(i)
  const text = read(rel)
  if (text === null) throw new Error(`${confirmationsDir}: holds ${entries.join(", ")}; ids must be 0..${entries.length - 1} with no gaps (${rel} is missing)`)
  const c = parseConfirmation(text)
  if (c.id !== i) throw new Error(`${rel}: id ${c.id}, expected ${i}`)
  return c
})
// A malformed cache is no cache: every life is verified again (the cache saves work, it never decides).
const verified = verifiedText === null ? null : parseVerified(verifiedText)
if (verifiedText !== null && verified === null) console.error(`::warning::${VERIFIED_PATH} does not parse: verifying every life again`)
// Every centre main has had, for the centre in force at each birth; the checkout's own is HEAD's, but is added anyway.
const centres = centresBySha(git, mainSpaces(root, git))
centres.set(await genomeSha(space.schemaVersion, centre), centre)

const runsDir = join(root, RUNS_DIR)
// When each run file reached main, read only while a confirmation is open: past its deadline, the gate decides it on
// main as it was then (confirm.ts asOfDeadline; decision 0026), whenever this build runs.
const open = confirmations.length > 0 && confirmations[confirmations.length - 1]!.result === null
const arrivals = open ? runArrivals(git) : null
const started = Date.now() // timing for the log only, never for a file
// The build's time: the clock, except in a simulation (tools/sim/meta-sim.ts), which pins it with META_BUILT_AT so its
// main's history (the index line, a confirmation's times, and so the next posterior's sourceCommit and every draw
// seeded from its bytes) replays byte for byte. The gate's workflow never sets it (commit-meta.test.ts "workflows").
const pinned = process.env.META_BUILT_AT
if (pinned !== undefined && !(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(pinned) && new Date(pinned).toISOString() === pinned)) {
  throw new Error(`META_BUILT_AT must be an ISO time like 2026-01-01T00:00:00.000Z, got ${JSON.stringify(pinned)}`)
}
const builtAt = pinned ?? new Date().toISOString()
const built = await buildOrCheckpoint({
  runsDir, config, refs: loadReferences(root, config), space, centre,
  sourceCommit: git(["rev-parse", "HEAD"]).trim(),
  builtAt, // the one clock read that reaches a file: the index line and a confirmation's declaredAt, deadline, decidedAt
  previous: { calibration: calibrationText === null ? null : parseCalibration(calibrationText), posteriorText, verified, confirmations },
  // Only the versions some not-yet-verified life was born under.
  posteriorsBySha: posteriorHistory(git, wantedPosteriors(runsDir, space.schemaVersion, verified)),
  centres, arrivals,
}, VERIFY_BUDGET)
console.error(`buildMeta took ${((Date.now() - started) / 1000).toFixed(1)} s`)

let changed = false
const write = (rel: string, text: string) => {
  if (read(rel) === text) return
  mkdirSync(dirname(join(root, rel)), { recursive: true })
  writeFileSync(join(root, rel), text)
  changed = true
}
if ("checkpoint" in built) {
  // The checkpoint: the verdicts so far, and nothing else (no posterior, calibration, diagnostics, index line,
  // confirmation, promotion or genome.json), so main's outputs never reflect a part of the lives.
  const { fresh, verified, pending } = built.checkpoint
  console.error(`::notice::verification incomplete: ${fresh} lives verified fresh, ${verified} with a verdict, ${pending} left for the next run (VERIFY_BUDGET ${VERIFY_BUDGET}); only ${VERIFIED_PATH} is written`)
  write(VERIFIED_PATH, built[VERIFIED_PATH])
} else {
  for (const rel of [POSTERIOR_PATH, CALIBRATION_PATH, DIAGNOSTICS_PATH, VERIFIED_PATH] as const) write(rel, built[rel])
  for (const [rel, text] of Object.entries(built.confirmations)) write(rel, text)
  // Only a passing confirmation's result brings these, in the same build (and so the same gate commit) as that result.
  if (built[GENOME_PATH] !== null) write(GENOME_PATH, built[GENOME_PATH])
  if (built.promotionLine !== null) { mkdirSync(dirname(join(root, PROMOTIONS_PATH)), { recursive: true }); appendFileSync(join(root, PROMOTIONS_PATH), built.promotionLine + "\n"); changed = true }
  if (built.indexLine !== null) { appendFileSync(join(root, POSTERIOR_INDEX_PATH), built.indexLine + "\n"); changed = true }
}
console.log(`changed=${changed}`)
