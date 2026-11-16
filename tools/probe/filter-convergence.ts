// The filter-convergence probe (spec §5.1, §10 "05c"; plan 05c). Offline and harness-only (AGENTS.md invariant 2).
//
//   npx tsx tools/probe/filter-convergence.ts --write-manifest     (renders the probe set; writes tools/probe/filter-probe-set.json)
//   npx tsx tools/probe/filter-convergence.ts --frames-png <dir>    (8 probe frames as PNGs, for looking at)
import { readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { serialise } from "../gate/serialise.ts"
import { buildManifest, renderProbeSet, writeFramePngs, type ProbeManifest } from "./filters/probe-set.ts"

const REPO = fileURLToPath(new URL("../..", import.meta.url))
export const MANIFEST_PATH = "tools/probe/filter-probe-set.json"

export const readManifest = (path: string): ProbeManifest => JSON.parse(readFileSync(path, "utf8")) as ProbeManifest

/** The CLI: returns the exit code. */
export async function main(argv: readonly string[]): Promise<number> {
  if (argv[0] === "--write-manifest" && argv.length === 1) {
    writeFileSync(join(REPO, MANIFEST_PATH), serialise(await buildManifest()))
    console.log(`wrote ${MANIFEST_PATH}`)
    return 0
  }
  if (argv[0] === "--frames-png" && argv.length === 2) {
    const m = readManifest(join(REPO, MANIFEST_PATH))
    for (const p of await writeFramePngs(await renderProbeSet(m), m, argv[1]!)) console.log(p)
    return 0
  }
  console.error("usage: filter-convergence.ts --write-manifest | --frames-png <dir>")
  return 1
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) process.exit(await main(process.argv.slice(2)))
