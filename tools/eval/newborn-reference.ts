// Usage: tsx tools/eval/newborn-reference.ts [runsDir] [--check] [--allow-missing] [--browser <text>] [--machine <text>]
//   (run from the repo root; default runsDir data/newborns, where the collected files are committed)
// Builds the newborn reference (spec §2.3) with the pinned baseline evaluator from newborn files collected from the
// tab's IndexedDB outbox (controller ruling B: newborns are run locally with ?newborns=a-b, not merged through 64 PRs),
// laid out like data/runs (<learnerId>/<seq>.json). It writes the file config/evaluation.json pins for "newborn"; a
// person reviews it and commits it under a decision. `--check` exits 1 when the committed file differs from the
// recomputation. The remeasure path: run ?newborns=0-63 again under new code, rebuild under a new file name, re-pin by
// decision.
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { pathToFileURL } from "node:url"
import { LEARNER_ID_RE, MAX_EVAL_PANEL, type LogFileV2 } from "@blanktab/harness/log"
import { NEWBORN_COUNT, NEWBORN_SEED_BASE } from "@blanktab/harness/newborn"
import { genomeSha, parseCentre } from "@blanktab/meta"
import { PROVENANCE_TEXT_RE, getEvaluator, parseNewbornReference, type NewbornProvenance, type NewbornReference } from "@blanktab/evaluators"
import { loadGenomeSpace } from "../gate/gate.ts"
import { spacesOf, validateSessionPr } from "../gate/validate.ts"
import { loadEvalConfig } from "./config.ts"
import { collectLives } from "./lives.ts"

/** Where the collected newborn files are committed (controller ruling); any other dir, e.g. .local/newborns, by argument. */
export const DEFAULT_NEWBORNS_DIR = "data/newborns"
const GENOME_PATH = "knowledge/genome.json"

export interface NewbornReferenceOptions {
  /** The checkout whose config/, knowledge/genome-space.json and knowledge/genome.json are used. */
  root: string
  /** The collected newborn files (<learnerId>/<seq>.json). */
  runsDir: string
  /** Compare with the committed file instead of writing it. */
  check: boolean
  /** Required to write (a log cannot say them); with --check, the committed file's are used unless given. */
  browser?: string
  machine?: string
  /** Write a reference although some k in 0…NEWBORN_COUNT − 1 has no newborn (refused without it). */
  allowMissing?: boolean
}
export interface NewbornReferenceResult { ok: boolean; path: string; messages: string[]; reference?: NewbornReference }

/** The reference file's bytes: 2-space JSON and a final newline (gate-written JSON's hygiene rule). */
export const referenceBytes = (r: NewbornReference): string => JSON.stringify(r, null, 2) + "\n"

/**
 * Every collected file, checked as the gate would check it as one learner's PR (validateSessionPr with main's genome
 * space; a heartbeat or other non-log name is skipped), and every life checked to be a newborn run as runNewborn
 * runs one: role newborn, the centre genome (knowledge/genome.json), seed NEWBORN_SEED_BASE + k for a k in
 * [0, NEWBORN_COUNT) that no other life has, and panel 1 + (k mod 16). The gate never sees these files, so these checks
 * stand in for its newborn checks. Errors, one per problem.
 */
async function checkCollected(root: string, runsDir: string): Promise<string[]> {
  const errors: string[] = []
  if (!existsSync(runsDir) || !statSync(runsDir).isDirectory()) return [`${runsDir}: no such directory`]
  const space = loadGenomeSpace(root)
  const centre = parseCentre(JSON.parse(readFileSync(join(root, GENOME_PATH), "utf8")), space)
  const centreSha = await genomeSha(space.schemaVersion, centre)
  for (const id of readdirSync(runsDir).sort()) {
    if (!LEARNER_ID_RE.test(id)) continue
    if (!statSync(join(runsDir, id)).isDirectory()) { errors.push(`${id}: not a directory (a learner's files go in <runsDir>/<learnerId>/<seq>.json)`); continue }
    const names = readdirSync(join(runsDir, id)).filter(n => /^\d+\.json$/.test(n)).sort()
    const files = names.map(n => ({ path: `data/runs/${id}/${n}`, content: readFileSync(join(runsDir, id, n), "utf8") }))
    if (files.length === 0) continue
    const v = await validateSessionPr(files, [], { spaces: spacesOf(space) })
    for (const e of v.errors) errors.push(`${id}: ${e}`)
  }
  if (errors.length > 0) return errors
  const byK = new Map<number, string>()
  for (const life of collectLives(runsDir)) {
    const at = (why: string) => errors.push(`${life.learnerId}: ${why}`)
    const f = life.files[0] as LogFileV2
    if (life.files.length !== 1 || f.schemaVersion !== 2 || f.meta?.role !== "newborn") { at("not a newborn reference file (one file, meta.role newborn)"); continue }
    if (f.meta.genomeSha !== centreSha) at(`its genome (${f.meta.genomeSha}) is not the centre's (${centreSha}, ${GENOME_PATH})`)
    const k = (f.settings.seed as number) - NEWBORN_SEED_BASE
    if (!(Number.isInteger(k) && k >= 0 && k < NEWBORN_COUNT)) { at(`its seed ${String(f.settings.seed)} is not a newborn's (${NEWBORN_SEED_BASE}…${NEWBORN_SEED_BASE + NEWBORN_COUNT - 1})`); continue }
    if (f.meta.evalPanel !== 1 + (k % MAX_EVAL_PANEL)) at(`its panel ${String(f.meta.evalPanel)} is not newborn ${k}'s (${1 + (k % MAX_EVAL_PANEL)})`)
    const other = byK.get(k)
    if (other !== undefined) at(`seed ${NEWBORN_SEED_BASE + k} is also ${other}'s: one life per newborn`)
    else byK.set(k, life.learnerId)
  }
  return errors
}

/** The reference `config.baseline` computes from the collected lives, with its provenance; or the errors. */
async function build(o: NewbornReferenceOptions, prov: { browser: string; machine: string }): Promise<{ reference?: NewbornReference; missing?: number[]; messages: string[]; errors: string[] }> {
  const errors = await checkCollected(o.root, o.runsDir)
  if (errors.length > 0) return { errors, messages: [] }
  const config = loadEvalConfig(o.root)
  const baseline = getEvaluator(config.baseline)
  if (!baseline.evaluateCohort) return { errors: [`${config.baseline} is not a cohort evaluator`], messages: [] }
  const lives = collectLives(o.runsDir)
  const out = baseline.evaluateCohort({ newborn: lives }, {})
  if (!out.eligible) return { errors: [`${config.baseline}: ${out.reason ?? "not eligible"}`], messages: [] }
  const body = parseNewbornReference((out.diagnostics as { reference?: unknown }).reference)
  if (!body) return { errors: [`${config.baseline} gave no newborn reference`], messages: [] }
  // Provenance: the one code they ran, the date of the latest, the browser and machine class as told.
  const used = lives.filter(l => body.learnerIds.includes(l.learnerId)).map(l => l.files[0] as LogFileV2)
  const shas = [...new Set(used.map(f => f.codeSha))].sort()
  if (shas.length !== 1) return { errors: [`the newborns ran ${shas.length} code versions (${shas.join(", ")}): build one reference per codeSha`], messages: [] }
  const messages: string[] = []
  if (shas[0] === "dev") messages.push("warning: the newborns ran a development build (codeSha dev): the code is not traceable")
  const date = used.map(f => f.writtenAt).sort().at(-1)!.slice(0, 10)
  const provenance: NewbornProvenance = { codeSha: shas[0]!, browser: prov.browser, machine: prov.machine, date }
  const reference = parseNewbornReference({ ...body, provenance })
  if (!reference) return { errors: ["the provenance is malformed (browser and machine: 1 to 120 printable ASCII characters)"], messages }
  const ignored = lives.length - body.learnerIds.length
  messages.push(`${body.learnerIds.length} newborns counted${ignored > 0 ? `, ${ignored} ignored by ${config.baseline} (not GPU widths, suite 1, genome schema 1, or the budget)` : ""}`)
  const ks = new Set(used.map(f => (f.settings.seed as number) - NEWBORN_SEED_BASE))
  const missing = Array.from({ length: NEWBORN_COUNT }, (_, k) => k).filter(k => !ks.has(k))
  if (missing.length > 0) messages.push(`missing newborns k = ${missing.join(", ")} (of 0–${NEWBORN_COUNT - 1})`)
  return { reference, missing, messages, errors: [] }
}

/**
 * Builds (or, with `check`, verifies) the pinned newborn reference. Never throws: any failure (bad input, an unreadable
 * file, a malformed config or genome file) is `ok: false` with an error message. Writing with newborns missing needs
 * `allowMissing`; checking a committed file does not (a person accepted it when it was written).
 */
export async function newbornReference(o: NewbornReferenceOptions): Promise<NewbornReferenceResult> {
  try {
    return await run(o)
  } catch (e) {
    return { ok: false, path: "", messages: [`error: ${e instanceof Error ? e.message : String(e)}`] }
  }
}

async function run(o: NewbornReferenceOptions): Promise<NewbornReferenceResult> {
  const config = loadEvalConfig(o.root)
  const rel = config.references.newborn
  if (rel === undefined) return { ok: false, path: "", messages: ["error: config/evaluation.json pins no newborn reference"] }
  const path = join(o.root, rel)
  const fail = (messages: string[]): NewbornReferenceResult => ({ ok: false, path, messages: messages.map(m => `error: ${m}`) })
  let committed: NewbornReference | null = null
  let committedBytes: string | null = null
  if (o.check) {
    if (!existsSync(path)) return fail([`${rel} does not exist: nothing to check`])
    committedBytes = readFileSync(path, "utf8")
    let json: unknown
    try { json = JSON.parse(committedBytes) } catch { return fail([`${rel} is not JSON`]) }
    committed = parseNewbornReference(json)
    if (!committed) return fail([`${rel} is not a valid newborn reference`])
  }
  const browser = o.browser ?? committed?.provenance?.browser
  const machine = o.machine ?? committed?.provenance?.machine
  if (browser === undefined || machine === undefined) return fail(["--browser and --machine are required (the provenance a log file cannot give)"])
  if (!PROVENANCE_TEXT_RE.test(browser) || !PROVENANCE_TEXT_RE.test(machine)) return fail(["--browser and --machine must be 1 to 120 printable ASCII characters"])
  const b = await build(o, { browser, machine })
  if (!b.reference) return fail([...b.errors, ...b.messages])
  const bytes = referenceBytes(b.reference)
  if (o.check) {
    if (bytes !== committedBytes) return { ok: false, path, reference: b.reference, messages: [...b.messages, `error: ${rel} differs from the recomputation from ${o.runsDir}`] }
    return { ok: true, path, reference: b.reference, messages: [...b.messages, `${rel} matches the recomputation`] }
  }
  if (b.missing!.length > 0 && !o.allowMissing) return fail([...b.messages, `${b.missing!.length} newborns are missing: pass --allow-missing to write the reference anyway`])
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, bytes)
  return { ok: true, path, reference: b.reference, messages: [...b.messages, `wrote ${rel}: review it, then commit it under a decision`] }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const opt = (name: string): string | undefined => {
    const i = args.indexOf(name)
    if (i < 0) return undefined
    const v = args[i + 1]
    args.splice(i, 2)
    if (v === undefined) { console.error(`${name} needs a value`); process.exit(2) }
    return v
  }
  const browser = opt("--browser"), machine = opt("--machine")
  const check = args.includes("--check"), allowMissing = args.includes("--allow-missing")
  const rest = args.filter(a => a !== "--check" && a !== "--allow-missing")
  if (rest.length > 1 || rest.some(a => a.startsWith("--"))) {
    console.error("usage: newborn-reference.ts [runsDir] [--check] [--allow-missing] [--browser <text>] [--machine <text>]")
    process.exit(2)
  }
  const r = await newbornReference({
    root: process.cwd(), runsDir: rest[0] ?? DEFAULT_NEWBORNS_DIR, check, allowMissing,
    ...(browser !== undefined ? { browser } : {}), ...(machine !== undefined ? { machine } : {}),
  })
  for (const m of r.messages) (m.startsWith("error") ? console.error : console.log)(m)
  process.exit(r.ok ? 0 : 1)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
