// Earlier versions of the knowledge files, read from main's git history (plan 05b Task 7). The gate and the validator
// check a genome against the space of its own schema (decision 0024 "Every schema checked against its own space"), and
// the gate verifies a life against the centre in force at its birth: both may be older than main's working tree.
// `git` runs in a full-history checkout of main (fetch-depth 0); HEAD is the main the run uses.
import { createHash } from "node:crypto"
import { parseCentre, parseGenomeSpace, type Genome, type GenomeSpace } from "@blanktab/meta"

export const GENOME_SPACE_FILE = "knowledge/genome-space.json"
export const GENOME_FILE = "knowledge/genome.json"

/**
 * The distinct texts `path` has had in the commits of HEAD's history that added or modified it, newest first (a text
 * that recurs counts once, at its newest place). Empty when the file never existed.
 */
export function fileHistory(git: (args: string[]) => string, path: string): string[] {
  const shas = git(["log", "--format=%H", "--diff-filter=AM", "HEAD", "--", path]).split("\n").map(x => x.trim()).filter(x => x !== "")
  const seen = new Set<string>()
  const out: string[] = []
  for (const sha of shas) {
    const text = git(["show", `${sha}:${path}`])
    if (seen.has(text)) continue
    seen.add(text)
    out.push(text)
  }
  return out
}

/**
 * Every genome space main has had, by schemaVersion: for each schemaVersion, the newest text of
 * knowledge/genome-space.json that parses (parseGenomeSpace). A text that does not parse is skipped: no genome can be
 * checked against it (the current space's validity is loadGenomeSpace's, which stops the gate).
 */
export function spacesBySchema(git: (args: string[]) => string): Map<number, GenomeSpace> {
  const out = new Map<number, GenomeSpace>()
  for (const text of fileHistory(git, GENOME_SPACE_FILE)) {
    let space: GenomeSpace
    try { space = parseGenomeSpace(JSON.parse(text)) } catch { continue }
    if (!out.has(space.schemaVersion)) out.set(space.schemaVersion, space) // newest first: the first seen is the newest
  }
  return out
}

/**
 * sha256 of the canonical `{ schemaVersion, values }` with sorted keys: @blanktab/meta genomeSha, synchronously through
 * node:crypto (history.test.ts pins the two equal).
 */
export function genomeShaSync(schemaVersion: number, g: Genome): string {
  const values: Genome = {}
  for (const k of Object.keys(g).sort()) values[k] = g[k]!
  return createHash("sha256").update(JSON.stringify({ schemaVersion, values })).digest("hex")
}

/**
 * Every centre main has had (knowledge/genome.json's history), by genomeSha: each text that parses (parseCentre) against
 * the space of its own genomeSchema in `spaces`. Others are skipped.
 */
export function centresBySha(git: (args: string[]) => string, spaces: ReadonlyMap<number, GenomeSpace>): Map<string, Genome> {
  const out = new Map<string, Genome>()
  for (const text of fileHistory(git, GENOME_FILE)) {
    let values: Genome, schema: number
    try {
      const v = JSON.parse(text) as { genomeSchema?: unknown }
      const space = typeof v.genomeSchema === "number" ? spaces.get(v.genomeSchema) : undefined
      if (!space) continue
      values = parseCentre(v, space)
      schema = space.schemaVersion
    } catch { continue }
    const sha = genomeShaSync(schema, values)
    if (!out.has(sha)) out.set(sha, values)
  }
  return out
}
