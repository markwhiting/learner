// Usage: tsx tools/gate/meta-cli.ts   (run from the root of a full-history checkout of main, after the gate's merges)
// Builds data/meta/** (posterior, calibration, diagnostics, verified, confirmations) from main's merged logs and pinned
// config (buildMeta), writes the files that changed, appends the index line when the posterior's bytes changed and the
// promotion line on a pass, and prints `changed=true|false`. It reads config/ and never writes it. knowledge/genome.json
// it writes only on a confirmation's pass, with the gate-computed content buildMeta returns (spec §4.5); committing is
// commit-meta.ts's. It prints the build's time on stderr, so a slow fit shows in the gate's log.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { genomeSha, parseCentre } from "@blanktab/meta"
import { loadEvalConfig, loadReferences } from "../eval/config.ts"
import { loadGenomeSpace, mainSpaces } from "./gate.ts"
import { git, isShallow } from "./github.ts"
import { confirmationPath, GENOME_PATH, parseConfirmation, PROMOTIONS_PATH, type ConfirmationFile } from "./confirm.ts"
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

const runsDir = join(root, "data/runs")
const started = Date.now() // timing for the log only, never for a file
const files = await buildMeta({
  runsDir, config, refs: loadReferences(root, config), space, centre,
  sourceCommit: git(["rev-parse", "HEAD"]).trim(),
  builtAt: new Date().toISOString(), // the one clock read that reaches a file: it goes into the index line only
  previous: { calibration: calibrationText === null ? null : parseCalibration(calibrationText), posteriorText, verified, confirmations },
  // Only the versions some not-yet-verified life was born under.
  posteriorsBySha: posteriorHistory(git, wantedPosteriors(runsDir, space.schemaVersion, verified)),
  centres,
})
console.error(`buildMeta took ${((Date.now() - started) / 1000).toFixed(1)} s`)

let changed = false
const write = (rel: string, text: string) => {
  if (read(rel) === text) return
  mkdirSync(dirname(join(root, rel)), { recursive: true })
  writeFileSync(join(root, rel), text)
  changed = true
}
for (const rel of [POSTERIOR_PATH, CALIBRATION_PATH, DIAGNOSTICS_PATH, VERIFIED_PATH] as const) write(rel, files[rel])
for (const [rel, text] of Object.entries(files.confirmations)) write(rel, text)
// Only a passing confirmation's result brings these, in the same build (and so the same gate commit) as that result.
if (files[GENOME_PATH] !== null) write(GENOME_PATH, files[GENOME_PATH])
if (files.promotionLine !== null) { mkdirSync(dirname(join(root, PROMOTIONS_PATH)), { recursive: true }); appendFileSync(join(root, PROMOTIONS_PATH), files.promotionLine + "\n"); changed = true }
if (files.indexLine !== null) { appendFileSync(join(root, POSTERIOR_INDEX_PATH), files.indexLine + "\n"); changed = true }
console.log(`changed=${changed}`)
