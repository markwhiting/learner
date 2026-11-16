// The filter-convergence probe (spec §5.1, §10 "05c"; plan 05c; decision 0027): do fresh lives' first-layer filters
// converge? Offline and harness-only (AGENTS.md invariant 2): it reads committed logs and writes only under --out.
//
//   npx tsx tools/probe/filter-convergence.ts [--runs data/runs] [--out .local/filter-probe] [--manifest tools/probe/filter-probe-set.json] [--genome knowledge/genome.json]
//   npx tsx tools/probe/filter-convergence.ts --write-manifest     (renders the probe set; writes the manifest)
//   npx tsx tools/probe/filter-convergence.ts --frames-png <dir>    (8 probe frames as PNGs, for looking at)
//   npx tsx tools/probe/filter-convergence.ts --filters-png <dir>   (init and 2,048 CU filter sheets of one life per genome)
//
// Exit codes: 0, the report was written with status "ok"; 2, written with status "identification failed" (every
// verdict withheld); 1, nothing written (a manifest mismatch, no eligible lives, or a bad flag).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import type { LifeInput } from "@blanktab/evaluators"
import type { Observation } from "@blanktab/interface"
import { obsToInputs } from "@blanktab/learner"
import { genomeSha } from "@blanktab/meta"
import { collectLives } from "../eval/lives.ts"
import { serialise } from "../gate/serialise.ts"
import { BAR, analyse, type Analysis, type ProbeReport } from "./filters/analyse.ts"
import { probeGrams } from "./filters/layer.ts"
import { ENCODERS, probeLives } from "./filters/lives.ts"
import { writeFilterPngs } from "./filters/png.ts"
import { buildManifest, renderProbeSet, writeFramePngs, type ProbeManifest } from "./filters/probe-set.ts"

const REPO = fileURLToPath(new URL("../..", import.meta.url))
export const MANIFEST_PATH = "tools/probe/filter-probe-set.json"

export interface ProbeInputs { frames: readonly Observation[]; manifestSha: string; lives: readonly LifeInput[]; centreSha: string }
export class ProbeRefusal extends Error {
  readonly why: "no eligible lives"
  constructor(why: "no eligible lives") { super(`filter probe: ${why}`); this.why = why; this.name = "ProbeRefusal" }
}

const fx = (v: number | null, d = 3) => (v === null ? "—" : v.toFixed(d))
const fp = (v: number | null) => (v === null ? "—" : v < 0.001 ? v.toExponential(2) : v.toFixed(4))
const verdictText = (c: boolean | null) => (c === null ? "not evaluable" : c ? "converged" : "not converged")
const short = (name: string) => (name.startsWith("genome:") ? `genome ${name.slice(7, 15)}` : name)

/** summary.md: the report's verdicts and controls as Markdown. No clock, no machine paths. */
export function summaryText(r: ProbeReport): string {
  const L: string[] = []
  if (r.status !== "ok") L.push("**Identification failed: every verdict is withheld.** The numbers below are descriptive only.", "")
  L.push("# Filter-convergence probe", "")
  L.push(`- Manifest sha256 \`${r.manifestSha}\`; centre genome \`${r.centreSha.slice(0, 8)}\`.`)
  L.push(`- ${r.lives} eligible lives; ${r.excluded.length} excluded. Status: ${r.status}.`)
  L.push(`- Bar (spec §5.1, a guess): n ≥ ${BAR.minLives}, ckaDelta ≥ ${BAR.ckaDelta}, matched ≥ ${BAR.matched}, p < ${BAR.alpha}.`)
  for (const e of ENCODERS) {
    L.push("", `## ${e}`, "", "| analysis | n | CKA learned | CKA init | ckaDelta | matched learned | matched init | p | Holm p | pFloor | verdict | verdict (Holm) |", "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (const a of r.convergence.filter((x: Analysis) => x.encoder === e)) {
      const v = a.verdict, s = a.stats
      const p = v.p ?? v.pDescriptive
      L.push(`| ${short(a.name)}${a.name === `genome:${r.centreSha}` ? " (centre)" : ""} | ${s.n} | ${fx(s.learned.cka)} | ${fx(s.init.cka)} | ${fx(v.ckaDelta)} | ${fx(s.learned.matched)} | ${fx(s.init.matched)} | ${fp(p)}${v.p === null ? " (descriptive)" : ""} | ${fp(v.pHolm)} | ${fp(v.pFloor)} | ${verdictText(v.converged)} | ${verdictText(v.convergedHolm)} |`)
    }
  }
  L.push("", "## Between-genome control", "")
  for (const c of r.recipe) L.push(`- ${c.encoder}: within ${fx(c.within)}, between ${fx(c.between)}, one-sided p ${fp(c.p)} over ${c.genomes} genomes.`)
  const margins = r.identification.filter(x => x.bestOther !== null).map(x => x.ownInit - x.bestOther!)
  L.push("", "## Identification", "", margins.length ? `- Smallest margin ownInit − bestOther: ${fx(Math.min(...margins))} (min ownInit ${fx(Math.min(...r.identification.map(x => x.ownInit)))}, max bestOther ${fx(Math.max(...r.identification.map(x => x.bestOther ?? -1)))}).` : "- Only one life: no other init to compare with.")
  L.push("", "## Flags", "", ...(r.flags.length ? r.flags.map(f => `- ${f.learnerId}: ${f.flag}`) : ["- none"]))
  L.push("", "## Excluded", "", ...(r.excluded.length ? r.excluded.map(x => `- ${x.learnerId}: ${x.reason}`) : ["- none"]))
  return L.join("\n") + "\n"
}

/**
 * The probe: probeLives → both encoders' centred Grams → analyse → summary.md text. Writes nothing. Throws ProbeRefusal
 * only when no life is eligible; a failed identification is a report with status "identification failed".
 */
export function runProbe(inp: ProbeInputs, onStage: (stage: string) => void = () => {}): { report: ProbeReport; summary: string } {
  const { lives, excluded } = probeLives(inp.lives)
  onStage(`lives: ${lives.length} eligible, ${excluded.length} excluded`)
  if (lives.length === 0) throw new ProbeRefusal("no eligible lives")
  const x = inp.frames.map(obsToInputs)
  const grams = probeGrams(lives, { periphery: x.map(i => i.periphery), fovea: x.map(i => i.fovea) })
  onStage("grams")
  const report = analyse(lives, grams, inp.centreSha, inp.manifestSha, excluded)
  onStage("analyse")
  return { report, summary: summaryText(report) }
}

/** report.json (serialise) and summary.md under `dir`. */
export function writeProbe(out: { report: ProbeReport; summary: string }, dir: string): void {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, "report.json"), serialise(out.report))
  writeFileSync(join(dir, "summary.md"), out.summary)
}

export const readManifest = (path: string): ProbeManifest => JSON.parse(readFileSync(path, "utf8")) as ProbeManifest

/** The centre genome's sha from knowledge/genome.json's shape ({ genomeSchema, values }). */
async function centreShaOf(path: string): Promise<string> {
  const g = JSON.parse(readFileSync(path, "utf8")) as { genomeSchema: number; values: Record<string, number | string> }
  return genomeSha(g.genomeSchema, g.values)
}

const FLAGS = ["--runs", "--out", "--manifest", "--genome"] as const
const USAGE = "usage: filter-convergence.ts [--runs <dir>] [--out <dir>] [--manifest <file>] [--genome <file>] | --write-manifest | --frames-png <dir> | --filters-png <dir>"

/** The CLI: returns the exit code (0 ok, 2 identification failed, 1 nothing written). */
export async function main(argv: readonly string[]): Promise<number> {
  const opt = new Map<string, string>()
  let mode: "run" | "--write-manifest" | "--frames-png" | "--filters-png" = "run"
  let pngDir = ""
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!
    if (a === "--write-manifest" && argv.length === 1) { mode = a; continue }
    if ((a === "--frames-png" || a === "--filters-png") && argv[i + 1] !== undefined && !argv[i + 1]!.startsWith("--")) { mode = a; pngDir = resolve(argv[++i]!); continue }
    if ((FLAGS as readonly string[]).includes(a) && argv[i + 1] !== undefined && !argv[i + 1]!.startsWith("--") && !opt.has(a)) { opt.set(a, resolve(argv[++i]!)); continue }
    console.error(`bad flag ${a}\n${USAGE}`)
    return 1
  }
  const runs = opt.get("--runs") ?? join(REPO, "data/runs"), out = opt.get("--out") ?? join(REPO, ".local/filter-probe")
  const manifestPath = opt.get("--manifest") ?? join(REPO, MANIFEST_PATH), genomePath = opt.get("--genome") ?? join(REPO, "knowledge/genome.json")

  if (mode === "--write-manifest") {
    writeFileSync(join(REPO, MANIFEST_PATH), serialise(await buildManifest()))
    console.log(`wrote ${MANIFEST_PATH}`)
    return 0
  }
  if (mode === "--filters-png") {
    const { lives } = probeLives(collectLives(runs))
    const seen = new Set<string>()
    for (const l of lives) {
      if (seen.has(l.genomeSha)) continue
      seen.add(l.genomeSha)
      for (const p of await writeFilterPngs(l, ["init", 2048], pngDir)) console.log(p)
    }
    return 0
  }

  const t0 = performance.now()
  const stage = (s: string) => console.log(`${((performance.now() - t0) / 1000).toFixed(1)} s  ${s}`)
  const manifest = readManifest(manifestPath)
  let frames: Observation[]
  try {
    frames = await renderProbeSet(manifest)
  } catch (e) {
    console.error((e as Error).message)
    return 1
  }
  stage(`probe set: ${frames.length} frames`)
  if (mode === "--frames-png") {
    for (const p of await writeFramePngs(frames, manifest, pngDir)) console.log(p)
    return 0
  }
  const lives = collectLives(runs)
  const centreSha = await centreShaOf(genomePath)
  stage(`read ${lives.length} lives`)
  let result: { report: ProbeReport; summary: string }
  try {
    result = runProbe({ frames, manifestSha: manifest.sha256, lives, centreSha }, stage)
  } catch (e) {
    if (e instanceof ProbeRefusal) { console.error(e.message); return 1 }
    throw e
  }
  writeProbe(result, out)
  stage(`wrote ${join(out, "report.json")} and summary.md: status ${result.report.status}`)
  return result.report.status === "ok" ? 0 : 2
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) process.exit(await main(process.argv.slice(2)))
