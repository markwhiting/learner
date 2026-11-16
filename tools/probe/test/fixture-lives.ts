// Synthetic lives for the filter probe's tests: LogFileV2 files that parseLogFile accepts (a test checks it), so they
// take the same path through probeLives as committed lives. Nothing here reads knowledge/ or data/.
import { createRng, hashSeed } from "@blanktab/detmath"
import type { LifeInput } from "@blanktab/evaluators"
import type { LogFileV2, MetaBlock } from "@blanktab/harness/log"
import { GPU_WIDTHS } from "@blanktab/learner"
import { crossed, file, meta } from "../../../packages/evaluators/test/fixtures/build.ts"
import { LEVELS, encodeFilters, initFirstLayer, layerArrays, type FirstLayer, type Level } from "../filters/lives.ts"

/** Fixture genome shas: the first is the fixture's centre. */
export const FIXTURE_GENOMES = ["a1".repeat(32), "b2".repeat(32), "c3".repeat(32)] as const
export const FIXTURE_MANIFEST_SHA = "f0".repeat(32)

/** `f` with every array mapped by `g(value, arrayIndex, index)` (arrays in FIRST_LAYER_PARAMS order). */
export function mapLayer(f: FirstLayer, g: (v: number, a: number, i: number) => number): FirstLayer {
  const [pw, pb, fw, fb] = layerArrays(f).map((x, a) => Float32Array.from(x, (v, i) => g(v, a, i)))
  return { periphery: { w: pw!, b: pb! }, fovea: { w: fw!, b: fb! } }
}

export interface FixtureLifeSpec {
  learnerId: string; genomeSha: string; seed: number
  /** Uniform noise half-width at 2,048 CU, scaled by level/2,048 below it. */
  noise: number
  /** Pull toward a shared state: at level L the state is (1 − t)·init + t·toward.layer, t = toward.at2048·L/2,048. */
  toward?: { layer: FirstLayer; at2048: number }
  /** Overrides of the life's meta block (every file's). */
  meta?: Partial<MetaBlock>
  /** Replace a level's snapshot (e.g. to plant a swap or a frozen state). */
  snapshots?: Partial<Record<Level, FirstLayer>>
}

/** A fixture life's snapshot at `level`: init, pulled toward `toward`, plus seeded noise growing with the level. */
export function fixtureSnapshot(s: FixtureLifeSpec, level: Level, init = initFirstLayer(s.seed, GPU_WIDTHS)): FirstLayer {
  const rnd = createRng(hashSeed(s.seed, "filter-probe/fixture/" + level))
  const t = (s.toward?.at2048 ?? 0) * level / 2048
  const to = s.toward ? layerArrays(s.toward.layer) : null
  return mapLayer(init, (v, a, i) => (1 - t) * v + t * (to ? to[a]![i]! : 0) + s.noise * level / 2048 * (2 * rnd() - 1))
}

/** A life of role search, inheritance none, GPU widths, with snapshot files crossing 512, 1,024 and 2,048 CU. */
export function fixtureLife(s: FixtureLifeSpec): LifeInput {
  const init = initFirstLayer(s.seed, GPU_WIDTHS)
  const m = meta({ genomeSha: s.genomeSha, ...s.meta })
  const files: LogFileV2[] = LEVELS.map((level, seq) => {
    const snap = s.snapshots?.[level] ?? fixtureSnapshot(s, level, init)
    const f = file({ learnerId: s.learnerId, seq, cu: crossed(level), meta: { ...m, filters: encodeFilters(snap) }, final: level === 2048 })
    return { ...f, settings: { lr: 0.001, seed: s.seed } }
  })
  return { learnerId: s.learnerId, files }
}
