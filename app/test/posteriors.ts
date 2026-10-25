// Test posteriors built for the repo's knowledge (its genome space and centre), as the gate would write them: shared by
// app/test and tests/e2e/meta.spec.ts, so the page's draws can be compared with birthDraw in Node.
import { readFileSync } from "node:fs"
import { fromUnit, genomeSha, parseCentre, parseGenomeSpace, searched, type Genome, type Mode } from "@blanktab/meta"

export const readKnowledge = (name: string): unknown => JSON.parse(readFileSync(new URL(`../../knowledge/${name}`, import.meta.url), "utf8"))
export const SPACE_JSON = readKnowledge("genome-space.json")
export const GENOME_JSON = readKnowledge("genome.json")
export const SPACE = parseGenomeSpace(SPACE_JSON)
export const CENTRE = parseCentre(GENOME_JSON, SPACE)
export const CENTRE_SHA = await genomeSha(SPACE.schemaVersion, CENTRE)

/** 10 calibration genomes: the centre, then 9 fixed points of the cube (the gate's Sobol points stand in here). */
const n = searched(SPACE).length
export const CALIB: Genome[] = [CENTRE, ...Array.from({ length: 9 }, (_, k) => fromUnit(SPACE, Array.from({ length: n }, (_, j) => ((k + 1) * 0.1 + j * 0.037) % 1), CENTRE))]
const CALIB_SHAS = await Promise.all(CALIB.map(g => genomeSha(SPACE.schemaVersion, g)))

export interface PageCounts { eligibleLives?: number; calibrationLives?: number; calibrationTarget?: number; looks?: number }

/**
 * posterior.json's text in `mode` (2-space JSON and a final newline, as the gate writes it), built for `centre`
 * (default the repo's; the calibration genomes are the repo centre's design either way).
 */
export function posteriorText(mode: Mode, page: PageCounts = {}, need: number[] = CALIB.map(() => 3), centre = { genomeSha: CENTRE_SHA, values: CENTRE }): string {
  return JSON.stringify({
    schemaVersion: 1, genomeSchema: SPACE.schemaVersion, evaluator: "search-composite@1", sourceCommit: "a".repeat(40), mode,
    centre,
    calibration: { genomes: CALIB.map((values, i) => ({ genomeSha: CALIB_SHAS[i]!, values })), need },
    models: {}, incumbent: null, confirmation: null, diagnostics: {},
    page: { eligibleLives: 0, lastPromotion: null, calibrationLives: 0, calibrationTarget: 30, looks: 0, ...page },
  }, null, 2) + "\n"
}

/**
 * The v2 search fixture (packages/meta/test/fixtures/posterior-v2-search.json: a real fit on the repo's space and centre,
 * 40 rows, an incumbent, confirmation 3 in flight with need 7/9/22), as its bytes.
 */
export const V2_SEARCH_TEXT = readFileSync(new URL("../../packages/meta/test/fixtures/posterior-v2-search.json", import.meta.url), "utf8")
type Json = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
/** The v2 search fixture's text with `edit` applied to its parsed object (2-space JSON and a final newline). */
export function v2SearchText(edit: (f: Json) => void = () => {}): string {
  const f = JSON.parse(V2_SEARCH_TEXT) as Json
  edit(f)
  return JSON.stringify(f, null, 2) + "\n"
}
