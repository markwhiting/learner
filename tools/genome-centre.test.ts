// The genome space's centres are today's learner (spec §3.2): they must equal the code's defaults, and every gene's
// range must lie inside the clamps the learner and caregiver apply, so a decoded genome always runs as drawn.
// knowledge/genome.json is the centre in force: the space's centres while its provenance is "initial", and after a
// promotion the candidate of the passing confirmation it names (spec §4.5; plan 05b Task 10). @blanktab/meta cannot
// import the learner or harness (its wall), so these pins live here.
import { describe, it, expect } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import { isDeepStrictEqual } from "node:util"
import {
  DEFAULT_SETTINGS, EXPLORE_EPSILON_RANGE, GAMMA, GAMMA_RANGE, HEAD_LR_SCALE_RANGE, LR_RANGE, RESIDUAL_DECAY_RANGE, clampSettings,
} from "@blanktab/learner"
import { CAREGIVER_DEFAULTS, CAREGIVER_RANGES, clampCaregiverSettings } from "@blanktab/harness/caregiver"
import { CONFIRM_ARM_OF_J, MIN_EVAL_SEED, EVAL_SEED_BLOCK, SETTING_KEY_RE, confirmArmOfSeed } from "@blanktab/harness/log"
import { DEFAULT_MAX_STEPS_PER_SEC, DEFAULT_SLEEP_INTERVAL } from "@blanktab/harness/session"
import {
  ARM_OF_J, CONFIRM_SEEDS, PRE_05A_CONSTANTS, SETTING_KEY_RE as META_SETTING_KEY_RE, armOfSeed, decodeGenome, parseCentre, parseGenomeSpace,
  type Genome, type GenomeSpace,
} from "@blanktab/meta"
import { confirmationPath, parseConfirmation } from "./gate/confirm.ts"
import { serialise } from "./gate/serialise.ts"

const REPO = new URL("..", import.meta.url)
const readText = (rel: string): string | null => (existsSync(new URL(rel, REPO)) ? readFileSync(new URL(rel, REPO), "utf8") : null)
const readJson = (name: string): unknown => JSON.parse(readText(`knowledge/${name}`)!)
const space = parseGenomeSpace(readJson("genome-space.json"))
const spaceCentres: Genome = Object.fromEntries(space.genes.map(g => [g.name, g.centre]))

/**
 * Why `genomeText` (a knowledge/genome.json) is not a centre the repo may be in force with, or null when it is: it
 * passes parseCentre; with provenance "initial" it is the space's centres; with "promotion" it is the candidate of
 * data/meta/confirmations/<provenance.confirmation>.json (read by `read`), whose result is a pass.
 */
function centreProvenanceError(genomeText: string, s: GenomeSpace, read: (rel: string) => string | null): string | null {
  let values: Genome
  const file = JSON.parse(genomeText) as { genomeSchema?: unknown; provenance?: { source?: unknown; confirmation?: unknown } }
  try { values = parseCentre(file, s) } catch (e) { return `does not parse: ${e instanceof Error ? e.message : String(e)}` }
  const source = file.provenance?.source
  if (source === "initial") return isDeepStrictEqual(values, Object.fromEntries(s.genes.map(g => [g.name, g.centre]))) ? null : "an initial genome that is not the space's centres"
  if (source !== "promotion") return `provenance.source ${JSON.stringify(source)} is neither "initial" nor "promotion"`
  const id = file.provenance?.confirmation
  if (!Number.isSafeInteger(id) || (id as number) < 0) return `provenance.confirmation ${JSON.stringify(id)} is not a confirmation id`
  const text = read(confirmationPath(id as number))
  if (text === null) return `${confirmationPath(id as number)} does not exist`
  const c = parseConfirmation(text)
  if (c.id !== id) return `${confirmationPath(id as number)} holds confirmation ${c.id}`
  if (c.result?.outcome !== "pass") return `confirmation ${c.id}'s result is ${c.result?.outcome ?? "null"}, not a pass`
  if (c.genomeSchema !== file.genomeSchema || !isDeepStrictEqual(values, c.candidate.values)) return `its values are not confirmation ${c.id}'s candidate's`
  return null
}

describe("the genome centre is today's learner", () => {
  it("the genome-space centres = DEFAULT_SETTINGS, CAREGIVER_DEFAULTS, DEFAULT_MAX_STEPS_PER_SEC and DEFAULT_SLEEP_INTERVAL", () => {
    expect(decodeGenome(spaceCentres, space)).toEqual({
      learner: { ...DEFAULT_SETTINGS },
      caregiver: { ...CAREGIVER_DEFAULTS },
      harness: { maxStepsPerSec: DEFAULT_MAX_STEPS_PER_SEC, sleepInterval: DEFAULT_SLEEP_INTERVAL },
    })
  })
  it("knowledge/genome.json is the centre in force: the space's centres (initial), or a passing confirmation's candidate (promotion)", () => {
    expect(centreProvenanceError(readText("knowledge/genome.json")!, space, readText)).toBeNull()
  })
  it("a promoted genome.json fixture with its passing confirmation file is accepted; one naming a failed confirmation is rejected", () => {
    const candidate: Genome = { ...spaceCentres, lr: 1e-3 }
    const sha = "a".repeat(64), other = "b".repeat(64)
    const confirmation = (id: number, outcome: string | null) => serialise({
      schemaVersion: 1, id, genomeSchema: space.schemaVersion, confirm: "confirm-fisher@1", search: "search-composite@1",
      candidate: { genomeSha: sha, values: candidate }, centre: { genomeSha: other, values: spaceCentres },
      seedBase: 10_000 + 1_000 * id, target: { candidate: 24, centre: 24, newborn: 60 },
      declaredAt: "2026-11-01T10:00:00.000Z", deadline: "2026-11-22T10:00:00.000Z", sourceCommit: "c".repeat(40),
      trigger: { pBeatsCentre: 0.97, incumbentMean: 0.4, rowsAtDeclaration: 120, modelSha: "d".repeat(64) },
      result: outcome === null ? null : { outcome, decidedAt: "2026-11-10T10:00:00.000Z", sourceCommit: "c".repeat(40), output: {}, lives: { candidate: [], centre: [], newborn: [] } },
    })
    const genome = (values: Genome, id: number) => serialise({ schemaVersion: 1, genomeSchema: space.schemaVersion, values, provenance: { source: "promotion", confirmation: id } })
    const files: Record<string, string> = {
      [confirmationPath(0)]: confirmation(0, "fail"), [confirmationPath(1)]: confirmation(1, "pass"), [confirmationPath(2)]: confirmation(2, null),
    }
    const read = (rel: string) => files[rel] ?? null
    expect(centreProvenanceError(genome(candidate, 1), space, read)).toBeNull()
    expect(centreProvenanceError(genome(candidate, 0), space, read)).toMatch(/fail, not a pass/)
    expect(centreProvenanceError(genome(candidate, 2), space, read)).toMatch(/null, not a pass/)
    expect(centreProvenanceError(genome(candidate, 3), space, read)).toMatch(/does not exist/)
    expect(centreProvenanceError(genome({ ...candidate, lr: 1.5e-3 }, 1), space, read)).toMatch(/candidate/)
    expect(centreProvenanceError(genome({ ...candidate, lr: 1 }, 1), space, read)).toMatch(/does not parse/)
    // An initial genome must be the space's centres; a promoted one may differ from them (that is what promotion does).
    const initial = (values: Genome) => serialise({ schemaVersion: 1, genomeSchema: space.schemaVersion, values, provenance: { source: "initial" } })
    expect(centreProvenanceError(initial(spaceCentres), space, read)).toBeNull()
    expect(centreProvenanceError(initial(candidate), space, read)).toMatch(/not the space's centres/)
  })
  it("every gene range lies inside the learner's and caregiver's clamps (LR_RANGE, HEAD_LR_SCALE_RANGE, EXPLORE_EPSILON_RANGE, RESIDUAL_DECAY_RANGE, GAMMA_RANGE, CAREGIVER_RANGES)", () => {
    const ranges: Record<string, readonly [number, number]> = {
      lr: LR_RANGE, headLrScale: HEAD_LR_SCALE_RANGE, exploreEpsilon: EXPLORE_EPSILON_RANGE, residualDecay: RESIDUAL_DECAY_RANGE,
      gamma: GAMMA_RANGE, ...CAREGIVER_RANGES,
    }
    const live = space.genes.filter(g => g.implemented !== false && g.target !== "harness")
    for (const g of live) {
      for (const v of [g.min!, g.max!]) {
        const r = ranges[g.name]
        if (r) expect(v >= r[0] && v <= r[1], `${g.name} ${v} in [${r}]`).toBe(true)
        // Whatever the named ranges miss (entropyWeight ≥ 0, bufferSize a positive integer, …), the clamp itself
        // must pass the bound through unchanged.
        if (g.target === "learner") expect(clampSettings({ ...DEFAULT_SETTINGS, [g.name]: v })[g.name as keyof typeof DEFAULT_SETTINGS], g.name).toBe(v)
        else expect(clampCaregiverSettings({ [g.name]: v })[g.name as keyof typeof CAREGIVER_DEFAULTS], g.name).toBe(v)
      }
    }
    expect(live.filter(g => g.target === "learner").map(g => g.name).sort()).toEqual(Object.keys(DEFAULT_SETTINGS).sort())
    expect(live.filter(g => g.target === "caregiver").map(g => g.name).sort()).toEqual(Object.keys(CAREGIVER_DEFAULTS).sort())
  })
  it("legacyGenome's pre-05a gamma is the learner's GAMMA, the value a state saved without gamma resumes with", () => {
    expect(PRE_05A_CONSTANTS).toEqual({ gamma: GAMMA })
  })
  it("the log's copy of the confirmation arm table equals meta's ARM_OF_J, and the parser's arm rule is armOfSeed on every seed of two blocks", () => {
    expect([...CONFIRM_ARM_OF_J]).toEqual([...ARM_OF_J])
    expect(EVAL_SEED_BLOCK).toBe(CONFIRM_SEEDS)
    for (const id of [0, 5]) {
      for (let j = 0; j < CONFIRM_SEEDS; j++) {
        const evalSeed = MIN_EVAL_SEED + EVAL_SEED_BLOCK * id + j
        expect(confirmArmOfSeed(evalSeed, id)).toBe(armOfSeed(evalSeed, id))
      }
    }
  })
  it("meta's copy of SETTING_KEY_RE equals the log's", () => {
    expect(META_SETTING_KEY_RE.source).toBe(SETTING_KEY_RE.source)
    expect(META_SETTING_KEY_RE.flags).toBe(SETTING_KEY_RE.flags)
  })
})
