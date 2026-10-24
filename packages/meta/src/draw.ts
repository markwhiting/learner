// The birth draw (spec §4.1): which genome a new life runs. A pure function of the learnerId, the posterior's bytes
// (through its parsed form and sha), the genome space, the centre and the widths class, so the gate can recompute
// every pick. The order of rng() calls is fixed: u0 picks the method, then the method's own draws follow.
//
// The draw order (plan 05b, Task 5; decision 0024 "Draw order"), after the checks below and u0 = rng():
//   gpu + calibrate: one rng() picks a calibration genome by need (05a).
//   gpu + search:
//     1. u0 < UNIFORM_SHARE (0.2): uniform, d more rng() (05a).
//     2. a confirmation in flight and u0 < CONFIRM_SHARE_END (0.7): j = ⌊rng()·CONFIRM_SEEDS⌋, arm = ARM_OF_J[j mod 4].
//        need[arm] > 0: confirm, evalSeed = seedBase + j, no panel. need[arm] = 0: on to ts, the same rng continuing.
//     3. ts: candidateSet(post, rng), then thompsonPick(post, cands, rng); the genome is fromUnit of the pick. No gpu
//        model, a model of other genes, no GP posterior or no pick: centre (with a reason, except from a v1 file).
//   paused, flat, and every mode on cpu: uniform below UNIFORM_SHARE, else centre (05a). CPU tabs never draw ts or
//   confirm. Every branch tests the mode before it reads `models` or `confirmation`, so a hand-built Posterior with a
//   model outside mode search (which parsePosterior refuses) still draws exactly as 05a.
//
// Every 05a draw is unchanged: a v1 posterior reads as models {} and confirmation null, so its gpu search births with
// u0 >= UNIFORM_SHARE draw centre with no reason, as 05a did, and evalSeed and confirmation are null for every 05a method.
//
// Posterior-supplied genomes are checked against the tab's own knowledge (controller ruling, Task 4 review). A
// posterior for another genome schema, another centre, or with an invalid calibration or confirmation genome gives
// `centre` with a `reason`. drawSeed and posteriorSha are still recorded, so the gate recomputes the same fallback.
import { createRng, hashSeed } from "@blanktab/detmath"
import { checkGenome, fromUnit, searched, type Genome, type GenomeSpace } from "./genome"
import type { ArmName, Posterior } from "./posterior"
import { candidateSet, modelPosterior, thompsonPick } from "./thompson"

export type Method = "centre" | "calibrate" | "uniform" | "ts" | "confirm" | "newborn" | "legacy"
export type Role = "search" | "confirm" | "newborn" | "legacy"
export interface DrawInput {
  learnerId: string; space: GenomeSpace; centre: Genome; widthsClass: "gpu" | "cpu"
  /** genomeSha of `centre` (the tab's knowledge/genome.json): the posterior must have been built for this centre. */
  centreSha: string
  /** null together: offline, absent or malformed. */
  posterior: Posterior | null; posteriorSha: string | null
}
export interface Draw {
  role: Role; method: Method; drawSeed: number | null; posteriorSha: string | null
  /** The shared checkup panel; null exactly for role confirm (a confirmation life meets its own variants). */
  evalPanel: number | null
  /** Role confirm only: the confirmation's seedBase + j; null otherwise. */
  evalSeed: number | null
  confirmation: { id: number; arm: ArmName } | null
  genome: Genome
  calibrationIndex?: number
  /** Why a posterior was present but the draw fell back to centre. */
  reason?: string
}

/** The share of births that run a uniform point of the cube, in every mode (spec §4.1; OLS §10.1). */
export const UNIFORM_SHARE = 0.2
/** While a confirmation is in flight, gpu search births with u0 in [UNIFORM_SHARE, CONFIRM_SHARE_END) draw one of its lives (spec §4.5). */
export const CONFIRM_SHARE_END = 0.7
/** Seeds per confirmation: j in 0..CONFIRM_SEEDS − 1, evalSeed = seedBase + j (spec §4.5). */
export const CONFIRM_SEEDS = 1000
/** The arm of a confirmation seed's j, by j mod 4 (spec §4.5): candidate, centre, newborn, newborn. */
export const ARM_OF_J: readonly ArmName[] = ["candidate", "centre", "newborn", "newborn"]
/** The arm of confirmation `id`'s seed `evalSeed`: ARM_OF_J[(evalSeed − 10000 − 1000·id) mod 4]. */
export function armOfSeed(evalSeed: number, id: number): ArmName {
  const j = evalSeed - 10_000 - 1_000 * id
  return ARM_OF_J[((j % 4) + 4) % 4]!
}
/** The number of shared evaluation panels (spec §4.1 step 4). */
export const PANELS = 16

/** The learner's shared checkup panel, 1..PANELS. */
export function evalPanelOf(learnerId: string): number {
  return 1 + (hashSeed(0, "panel/" + learnerId) % PANELS)
}

/** The seed of the learner's draw under the posterior whose bytes hash to `posteriorSha`. */
export function drawSeedOf(learnerId: string, posteriorSha: string): number {
  return hashSeed(0, "draw/" + learnerId + "/" + posteriorSha)
}

/** Index i with probability need[i] / Σneed, or uniform when Σneed = 0, from one uniform u in [0, 1). */
function pickByNeed(need: readonly number[], u: number): number {
  let total = 0
  for (const w of need) total += w
  if (total === 0) return Math.min(need.length - 1, Math.floor(u * need.length))
  const target = u * total
  let cum = 0, last = 0
  for (let i = 0; i < need.length; i++) {
    if (need[i]! === 0) continue
    last = i
    cum += need[i]!
    if (target < cum) return i
  }
  return last // only reached through rounding at the top end
}

const HEX64 = /^[0-9a-f]{64}$/

/**
 * The genome this birth runs, and how it was chosen. Every draw has role "search" except a confirm draw (role
 * confirm). Throws on a programming error: `posterior` and `posteriorSha` not null together, or a sha that is not 64
 * lowercase hex.
 */
export function birthDraw(input: DrawInput): Draw {
  const { learnerId, space, centre, centreSha, widthsClass, posterior, posteriorSha } = input
  if ((posterior === null) !== (posteriorSha === null)) throw new TypeError("posterior and posteriorSha must be null together")
  if (posteriorSha !== null && !HEX64.test(posteriorSha)) throw new TypeError(`posteriorSha must be 64 lowercase hex, got ${JSON.stringify(posteriorSha)}`)
  if (!HEX64.test(centreSha)) throw new TypeError(`centreSha must be 64 lowercase hex, got ${JSON.stringify(centreSha)}`)
  const evalPanel = evalPanelOf(learnerId)
  const none = { evalSeed: null, confirmation: null }
  if (posterior === null || posteriorSha === null)
    return { role: "search", method: "centre", drawSeed: null, posteriorSha: null, evalPanel, ...none, genome: { ...centre } }

  const drawSeed = drawSeedOf(learnerId, posteriorSha)
  const base = { role: "search" as const, drawSeed, posteriorSha, evalPanel, ...none }
  const fallback = (reason: string): Draw => ({ ...base, method: "centre", genome: { ...centre }, reason })
  if (posterior.genomeSchema !== space.schemaVersion)
    return fallback(`posterior genomeSchema ${posterior.genomeSchema} is not the space's ${space.schemaVersion}`)
  if (posterior.centre.genomeSha !== centreSha) return fallback("posterior built for another centre")

  const rng = createRng(drawSeed)
  const u0 = rng()

  if (widthsClass === "gpu" && posterior.mode === "calibrate") {
    const { genomes, need } = posterior.calibration
    const i = pickByNeed(need, rng())
    const bad = checkGenome(space, genomes[i]!.values)
    if (bad !== null) return fallback(`calibration genome ${i} is invalid: ${bad}`)
    return { ...base, method: "calibrate", genome: { ...genomes[i]!.values }, calibrationIndex: i }
  }
  // search, paused and flat; and every mode on cpu.
  if (u0 < UNIFORM_SHARE) {
    const u = searched(space).map(() => rng())
    return { ...base, method: "uniform", genome: fromUnit(space, u, centre) }
  }
  // Only gpu search reads the search's state (the mode guard): paused, flat and cpu births draw centre, as in 05a.
  if (widthsClass !== "gpu" || posterior.mode !== "search") return { ...base, method: "centre", genome: { ...centre } }

  const conf = posterior.confirmation
  if (conf !== null && u0 < CONFIRM_SHARE_END) {
    const j = Math.floor(rng() * CONFIRM_SEEDS)
    const arm = ARM_OF_J[j % 4]!
    if (conf.need[arm] > 0) {
      const values = arm === "centre" ? conf.centre.values : conf.candidate.values
      const bad = checkGenome(space, values)
      if (bad !== null) return fallback(`confirmation ${conf.id}'s ${arm} genome is invalid: ${bad}`)
      return {
        role: "confirm", method: "confirm", drawSeed, posteriorSha, evalPanel: null, evalSeed: conf.seedBase + j,
        confirmation: { id: conf.id, arm }, genome: { ...values },
      }
    }
    // This arm is full: the birth is a ts draw, its rng continuing after j.
  }
  return tsDraw(space, centre, posterior, rng, base, fallback)
}

/** The ts branch: the gpu model's GP posterior, a Thompson pick on candidateSet(post, rng), and its genome. */
function tsDraw(
  space: GenomeSpace, centre: Genome, posterior: Posterior, rng: () => number,
  base: Omit<Draw, "method" | "genome">, fallback: (reason: string) => Draw,
): Draw {
  const model = posterior.models.gpu
  if (model === undefined) {
    // A v1 file has no models: its draw is 05a's centre, which carried no reason.
    if (posterior.schemaVersion === 1) return { ...base, method: "centre", genome: { ...centre } }
    return fallback("no gpu model")
  }
  const names = searched(space).map(g => g.name)
  if (model.genes.length !== names.length || model.genes.some((g, j) => g !== names[j]))
    return fallback("the model's genes are not the space's searched genes")
  const post = modelPosterior(model)
  if (post === null) return fallback("no GP posterior from the model")
  const pick = thompsonPick(post, candidateSet(post, rng), rng)
  if (pick === null) return fallback("no Thompson pick")
  return { ...base, method: "ts", genome: fromUnit(space, pick.x, centre) }
}
