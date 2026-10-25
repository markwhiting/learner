// Usage: tsx tools/gate/meta-cli.ts   (run from the root of a full-history checkout of main, after the gate's merges)
// Builds data/meta/** (posterior, calibration, diagnostics, verified) from main's merged logs and pinned config (buildMeta),
// writes the files that changed, appends the index line when the posterior's bytes changed, and prints
// `changed=true|false`. It reads config/ and knowledge/ (and their history) and never writes them; committing is
// commit-meta.ts's. It prints the build's time on stderr, so a slow fit shows in the gate's log.
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { genomeSha, parseCentre } from "@blanktab/meta"
import { loadEvalConfig, loadReferences } from "../eval/config.ts"
import { loadGenomeSpace, mainSpaces } from "./gate.ts"
import { git, isShallow } from "./github.ts"
import { centresBySha } from "./history.ts"
import {
  buildMeta, CALIBRATION_PATH, DIAGNOSTICS_PATH, parseCalibration, parseVerified, POSTERIOR_INDEX_PATH, POSTERIOR_PATH, posteriorHistory,
  VERIFIED_PATH, wantedPosteriors,
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
// A malformed cache is no cache: every life is verified again (the cache saves work, it never decides).
const verified = verifiedText === null ? null : parseVerified(verifiedText)
if (verifiedText !== null && verified === null) console.error(`::warning::${VERIFIED_PATH} does not parse: verifying every life again`)
// Every centre main has had, for the centre in force at each birth; the checkout's own is HEAD's, but is added anyway.
const centres = centresBySha(git, mainSpaces(root, git))
centres.set(await genomeSha(space.schemaVersion, centre), centre)

const runsDir = join(root, "data/runs")
const started = Date.now() // timing for the log only, never for a file
const files = await buildMeta({
  runsDir, config, refs: loadReferences(root, config), space, centre,
  sourceCommit: git(["rev-parse", "HEAD"]).trim(),
  builtAt: new Date().toISOString(), // the one clock read that reaches a file: it goes into the index line only
  previous: { calibration: calibrationText === null ? null : parseCalibration(calibrationText), posteriorText, verified },
  // Only the versions some not-yet-verified life was born under.
  posteriorsBySha: posteriorHistory(git, wantedPosteriors(runsDir, verified)),
  centres,
})
console.error(`buildMeta took ${((Date.now() - started) / 1000).toFixed(1)} s`)

let changed = false
for (const rel of [POSTERIOR_PATH, CALIBRATION_PATH, DIAGNOSTICS_PATH, VERIFIED_PATH] as const) {
  const text = files[rel]
  if (read(rel) === text) continue
  mkdirSync(dirname(join(root, rel)), { recursive: true })
  writeFileSync(join(root, rel), text)
  changed = true
}
if (files.indexLine !== null) { appendFileSync(join(root, POSTERIOR_INDEX_PATH), files.indexLine + "\n"); changed = true }
console.log(`changed=${changed}`)
