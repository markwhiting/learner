// The GP determinism probe's inputs and how its outputs are computed (plan 05b, Task 6), shared by gp-probe.test.ts
// (Node) and the e2e detmath probe page (app/detmath-probe.html: each engine's main thread and a module worker), so
// all of them compare the same values against fixtures/gp-probe-golden.json. Browser-safe: no Node API; the fixtures
// come in as JSON modules.
//
// What it exercises, end to end: fitGp (multi-start L-BFGS on the MAP objective) on a seeded 12-gene, 60-row dataset;
// conditionGp and a joint predict with the fitted hyperparameters; incumbentOf on that posterior; and birthDraw on the
// committed v2 search fixture (fixtures/posterior-v2-search.json): 20 ts draws (no confirmation) and 10 confirm draws
// (its confirmation in flight), which run modelPosterior, candidateSet and thompsonPick inside.
import { createRng, hashSeed } from "@blanktab/detmath"
import { birthDraw, drawSeedOf, parseCentre, parseGenomeSpace, parsePosterior, CONFIRM_SHARE_END, UNIFORM_SHARE } from "../src/index"
import type { Draw, DrawInput, Genome, GenomeSpace } from "../src/index"
import { fitGp, type FitResult } from "../src/fit"
import { conditionGp, predict, type GpData } from "../src/gp"
import { incumbentOf } from "../src/thompson"
import spaceJson from "../../../knowledge/genome-space.json"
import centreJson from "../../../knowledge/genome.json"
import posteriorJson from "./fixtures/posterior-v2-search.json"

/** The IEEE-754 bits of a double as 16 hex digits: exact for every value, including -0, NaN and ±Infinity. */
export function bits(x: number): string {
  const v = new DataView(new ArrayBuffer(8))
  v.setFloat64(0, x)
  return v.getBigUint64(0).toString(16).padStart(16, "0")
}

/** `v` with every number replaced by its bits (arrays, typed arrays and plain objects walked; strings, booleans and
 * null kept), so a deep equality of two such values is a bit-for-bit comparison of every number in them. */
export function toBits(v: unknown): unknown {
  if (typeof v === "number") return bits(v)
  if (Array.isArray(v)) return v.map(toBits)
  if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v as unknown as ArrayLike<number>, bits)
  if (v !== null && typeof v === "object") {
    const out: Record<string, unknown> = {}
    for (const k of Object.keys(v).sort()) out[k] = toBits((v as Record<string, unknown>)[k])
    return out
  }
  return v
}

export interface GpProbe {
  fit: FitResult
  /** conditionGp(data, fit.hyper) and predict(…, joint) at PREDICT_POINTS: the posterior's α, then mean, cov, variance. */
  predict: { alpha: number[]; mean: number[]; cov: number[]; variance: number[] }
  /** incumbentOf the fitted posterior, centre at the cube's middle, seeded by hashSeed(0, "incumbent/gp-probe"). */
  incumbent: { x: number[]; mean: number; pBeatsCentre: number }
  /** 20 birthDraw ts draws from the fixture posterior without its confirmation (every one method "ts"). */
  picks: Draw[]
  /** 10 birthDraw confirm draws from the fixture posterior with its confirmation in flight (every one method "confirm"). */
  confirms: Draw[]
}

const D = 12, N = 60, PREDICT_POINTS = 6, TS_DRAWS = 20, CONFIRM_DRAWS = 10
/** A stand-in posterior sha: the draws' seeds depend on it; any 64-hex string is a valid one. */
const PROBE_SHA = "9e3779b97f4a7c15f39cc0605cedc8341082276bf3a27251f86c6a11d0c18e95"

/** The seeded 12-gene, 60-row dataset: a smooth bowl-and-ridge in the unit cube plus small seeded noise. */
export function probeData(): GpData {
  const rng = createRng(hashSeed(0, "gp-probe/data"))
  const x: number[][] = []
  const y: number[] = []
  for (let i = 0; i < N; i++) {
    const row = Array.from({ length: D }, () => rng())
    let s = 0
    for (let j = 0; j < D; j++) s += (j % 3 === 0 ? -1.5 : 0.3) * (row[j]! - 0.6) * (row[j]! - 0.6) * (1 + j * 0.05)
    x.push(row)
    y.push(s + 0.2 * (rng() - 0.5))
  }
  return { x, y }
}

const space = (): GenomeSpace => parseGenomeSpace(spaceJson)
const centre = (s: GenomeSpace): Genome => parseCentre(centreJson, s)
type Json = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
const fixtureFile = (): Json => JSON.parse(JSON.stringify(posteriorJson)) as Json

/** A tab's input for a posterior file object: its text parsed (it must parse), with the probe's stand-in sha. */
function inputOf(learnerId: string, f: Json, s: GenomeSpace, c: Genome): DrawInput {
  const p = parsePosterior(JSON.stringify(f))
  if (p === null) throw new Error("gp probe: the posterior does not parse")
  return { learnerId, space: s, centre: c, centreSha: p.centre.genomeSha, widthsClass: "gpu", posterior: p, posteriorSha: PROBE_SHA }
}

const u0Of = (learnerId: string) => createRng(drawSeedOf(learnerId, PROBE_SHA))()
const idOf = (k: number) => "lrn_" + (0x9000 + k).toString(16).padStart(16, "0")

/** The first `count` learnerIds (lrn_…9000 upwards) whose u0 satisfies `keep`. */
function idsWhere(count: number, keep: (u0: number) => boolean): string[] {
  const out: string[] = []
  for (let k = 0; out.length < count; k++) if (keep(u0Of(idOf(k)))) out.push(idOf(k))
  return out
}

export function computeGp(): GpProbe {
  const data = probeData()
  const fit = fitGp(data)
  if (fit === null) throw new Error("gp probe: fitGp gave null")
  const post = conditionGp(data, fit.hyper)
  if (post === null) throw new Error("gp probe: conditionGp gave null")
  const prng = createRng(hashSeed(0, "gp-probe/predict"))
  const xs = Array.from({ length: PREDICT_POINTS }, () => Array.from({ length: D }, () => prng()))
  const p = predict(post, xs, true)
  if (p === null || p.cov === null) throw new Error("gp probe: predict gave null")
  const incumbent = incumbentOf(post, Array.from({ length: D }, () => 0.5), hashSeed(0, "incumbent/gp-probe"))
  if (incumbent === null) throw new Error("gp probe: incumbentOf gave null")

  const s = space(), c = centre(s)
  const plain = fixtureFile()
  plain.confirmation = null
  const confirming = fixtureFile()
  const picks = idsWhere(TS_DRAWS, u0 => u0 >= UNIFORM_SHARE).map(id => birthDraw(inputOf(id, plain, s, c)))
  const confirms: Draw[] = []
  for (const id of idsWhere(4 * CONFIRM_DRAWS, u0 => u0 >= UNIFORM_SHARE && u0 < CONFIRM_SHARE_END)) {
    const d = birthDraw(inputOf(id, confirming, s, c))
    if (d.method === "confirm") confirms.push(d)
    if (confirms.length === CONFIRM_DRAWS) break
  }
  return {
    fit,
    predict: { alpha: Array.from(post.alpha), mean: Array.from(p.mean), cov: Array.from(p.cov), variance: Array.from(p.variance) },
    incumbent, picks, confirms,
  }
}

/**
 * The timing probe's input (not part of the golden): the fixture posterior's model with `rows` seeded rows instead of
 * its 40 (same genes and hyperparameters, no confirmation), and the first learnerId whose draw is ts.
 */
export function tsDrawInputAt(rows: number): DrawInput {
  const f = fixtureFile()
  f.confirmation = null
  const rng = createRng(hashSeed(0, "gp-probe/timing"))
  const d = (f.models.gpu.genes as string[]).length
  f.models.gpu.rows = Array.from({ length: rows }, (_, i) => ({
    learnerId: "lrn_" + (0x1000 + i).toString(16).padStart(16, "0"),
    x: Array.from({ length: d }, () => Math.round(rng() * 1e6) / 1e6),
  }))
  f.models.gpu.y = Array.from({ length: rows }, () => rng() - 0.5)
  const s = space(), c = centre(s)
  return inputOf(idsWhere(1, u0 => u0 >= UNIFORM_SHARE)[0]!, f, s, c)
}
