// The genome's centre is today's learner (spec §3.2): knowledge/genome.json and the genome-space centres must equal the
// code's defaults, and every gene's range must lie inside the clamps the learner and caregiver apply, so a decoded
// genome always runs as drawn. @blanktab/meta cannot import the learner or harness (its wall), so these pins live here.
import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import {
  DEFAULT_SETTINGS, EXPLORE_EPSILON_RANGE, GAMMA, GAMMA_RANGE, HEAD_LR_SCALE_RANGE, LR_RANGE, RESIDUAL_DECAY_RANGE, clampSettings,
} from "@blanktab/learner"
import { CAREGIVER_DEFAULTS, CAREGIVER_RANGES, clampCaregiverSettings } from "@blanktab/harness/caregiver"
import { CONFIRM_ARM_OF_J, MIN_EVAL_SEED, EVAL_SEED_BLOCK, SETTING_KEY_RE, confirmArmOfSeed } from "@blanktab/harness/log"
import { DEFAULT_MAX_STEPS_PER_SEC, DEFAULT_SLEEP_INTERVAL } from "@blanktab/harness/session"
import { ARM_OF_J, CONFIRM_SEEDS, PRE_05A_CONSTANTS, SETTING_KEY_RE as META_SETTING_KEY_RE, armOfSeed, decodeGenome, parseCentre, parseGenomeSpace } from "@blanktab/meta"

const readJson = (name: string): unknown => JSON.parse(readFileSync(new URL(`../knowledge/${name}`, import.meta.url), "utf8"))
const space = parseGenomeSpace(readJson("genome-space.json"))
const centre = parseCentre(readJson("genome.json"), space)
const spaceCentres = Object.fromEntries(space.genes.map(g => [g.name, g.centre]))

describe("the genome centre is today's learner", () => {
  it("genome.json values = genome-space centres = DEFAULT_SETTINGS, CAREGIVER_DEFAULTS, DEFAULT_MAX_STEPS_PER_SEC and DEFAULT_SLEEP_INTERVAL", () => {
    expect(centre).toEqual(spaceCentres)
    expect(decodeGenome(centre, space)).toEqual({
      learner: { ...DEFAULT_SETTINGS },
      caregiver: { ...CAREGIVER_DEFAULTS },
      harness: { maxStepsPerSec: DEFAULT_MAX_STEPS_PER_SEC, sleepInterval: DEFAULT_SLEEP_INTERVAL },
    })
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
