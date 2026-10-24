import { describe, it, expect } from "vitest"
import { readFileSync, writeFileSync } from "node:fs"
import { createRng, hashSeed, normInv } from "@blanktab/detmath"
import { fromUnit, genomeSha, parseCentre, parseGenomeSpace, parsePosterior, searched, sha256Hex, toUnit, type GpModel } from "@blanktab/meta"
import { conditionGp, predict, type GpData, type GpPosterior } from "../src/gp"
import { fitGp } from "../src/fit"
import {
  candidateSet, incumbentOf, modelPosterior, normalOf, round6, thompsonPick,
  TS_JITTER, TS_JITTER_STEPS, TS_LOCAL_SD, TS_PARENTS, TS_PER_PARENT, TS_UNIFORM,
} from "../src/thompson"
import type { GpHyper } from "../src/posterior"

const PICKS = new URL("./fixtures/ts-picks.json", import.meta.url)
const POSTERIOR_FIXTURE = new URL("./fixtures/posterior-v2-search.json", import.meta.url)
const SPACE_FILE = new URL("../../../knowledge/genome-space.json", import.meta.url)
const CENTRE_FILE = new URL("../../../knowledge/genome.json", import.meta.url)
type Json = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any

/** The float64 bits of a number as 16 hex digits (bit for bit). */
function bits(v: number): string {
  const buf = new DataView(new ArrayBuffer(8))
  buf.setFloat64(0, v)
  return buf.getBigUint64(0).toString(16).padStart(16, "0")
}

/** The gate's serialise (tools/gate/meta.ts): sorted keys, one-space indent, final newline. */
function sortKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeys)
  if (typeof v === "object" && v !== null) return Object.fromEntries(Object.keys(v).sort().map(k => [k, sortKeys((v as Json)[k])]))
  return v
}
const serialise = (v: unknown): string => JSON.stringify(sortKeys(v), null, 1) + "\n"

/** An rng that counts its draws. */
function counting(seed: number): { rng: () => number; count: () => number } {
  const inner = createRng(seed)
  let n = 0
  return { rng: () => { n++; return inner() }, count: () => n }
}

/** Seeded rows in the unit cube with a smooth target. */
function dataset(seed: number, d: number, n: number, f: (x: number[]) => number): GpData {
  const rng = createRng(seed)
  const x: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: d }, () => rng())
    x.push(row)
    y.push(f(row))
  }
  return { x, y }
}

/** The 1-d model with a clear peak at 0.7: 8 rows, noisy enough to leave room to explore. */
function peaked1d(): GpPosterior {
  const x = [0.05, 0.2, 0.35, 0.5, 0.62, 0.7, 0.78, 0.95].map(v => [v])
  const y = x.map(([v]) => 1 - 30 * (v! - 0.7) * (v! - 0.7))
  const post = conditionGp({ x, y }, { lengthscales: [0.15], outputscale: 1, noise: 0.05, mean: 0 })
  expect(post).not.toBeNull()
  return post!
}

describe("thompson: constants and helpers", () => {
  it("the constants are the plan's; normalOf(u) = normInv(u + 2^-33) and is finite on [0, 1); round6 rounds to 6 decimals", () => {
    expect([TS_UNIFORM, TS_PARENTS, TS_PER_PARENT, TS_LOCAL_SD, TS_JITTER, TS_JITTER_STEPS]).toEqual([256, 16, 16, 0.1, 1e-6, 5])
    expect(normalOf(0.3)).toBe(normInv(0.3 + 1 / 8589934592))
    expect(Number.isFinite(normalOf(0))).toBe(true)
    expect(Number.isFinite(normalOf(4294967295 / 4294967296))).toBe(true) // the largest rng() value
    expect(round6(0.1234565)).toBe(Math.round(0.1234565 * 1e6) / 1e6)
    expect(round6(1)).toBe(1)
    expect(round6(0)).toBe(0)
  })
})

describe("candidateSet", () => {
  it("candidateSet has 512 points in [0,1]^d; the first 256 consume exactly 256·d rng draws; children stay within 4·σ_j of their parent unless clipped", () => {
    const d = 3
    const data = dataset(11, d, 40, r => r[0]! - (r[1]! - 0.4) * (r[1]! - 0.4) + 0.2 * r[2]!)
    // ℓ = 0.3 gives σ = 0.03; ℓ = 2 and 14 are capped at σ = TS_LOCAL_SD (ruling 4).
    const hyper: GpHyper = { lengthscales: [0.3, 2, 14], outputscale: 1, noise: 0.01, mean: 0 }
    const post = conditionGp(data, hyper)!
    const sd = [0.03, 0.1, 0.1].map((v, j) => Math.min(TS_LOCAL_SD * hyper.lengthscales[j]!, TS_LOCAL_SD))
    expect(sd[0]).toBeCloseTo(0.03, 15)
    expect(sd[1]).toBe(0.1)
    expect(sd[2]).toBe(0.1)

    const { rng, count } = counting(7)
    const cands = candidateSet(post, rng)
    expect(cands).toHaveLength(TS_UNIFORM + TS_PARENTS * TS_PER_PARENT)
    expect(cands).toHaveLength(512)
    expect(count()).toBe(512 * d) // 256·d uniform draws, then 256·d child draws, nothing else
    for (const c of cands) {
      expect(c).toHaveLength(d)
      for (const v of c) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1) }
    }
    // The first 256 are exactly the first 256·d draws, row by row.
    const fresh = createRng(7)
    for (let i = 0; i < TS_UNIFORM; i++) for (let j = 0; j < d; j++) expect(cands[i]![j]).toBe(fresh())
    // The parents are the 16 rows with the highest posterior mean, ties to the lowest index.
    const mean = predict(post, data.x, false)!.mean
    const order = Array.from({ length: data.x.length }, (_, i) => i).sort((a, b) => mean[b]! - mean[a]! || a - b)
    for (let k = 0; k < TS_PARENTS; k++) {
      const parent = data.x[order[k]!]!
      for (let c = 0; c < TS_PER_PARENT; c++) {
        const child = cands[TS_UNIFORM + k * TS_PER_PARENT + c]!
        for (let j = 0; j < d; j++) {
          const v = child[j]!
          if (v === 0 || v === 1) continue
          expect(Math.abs(v - parent[j]!)).toBeLessThanOrEqual(4 * sd[j]!)
        }
      }
    }
    // And each child is exactly its parent plus σ_j·normalOf(next draw), clipped: replay the child draws.
    const replay = createRng(7)
    for (let i = 0; i < TS_UNIFORM * d; i++) replay()
    for (let k = 0; k < TS_PARENTS; k++) for (let c = 0; c < TS_PER_PARENT; c++) for (let j = 0; j < d; j++) {
      const raw = data.x[order[k]!]![j]! + sd[j]! * normalOf(replay())
      expect(cands[TS_UNIFORM + k * TS_PER_PARENT + c]![j]).toBe(Math.min(1, Math.max(0, raw)))
    }
  })

  it("with fewer than 16 rows the parents cycle in rank order, so there are still 512 candidates", () => {
    const data: GpData = { x: [[0.2, 0.2], [0.8, 0.6], [0.5, 0.9]], y: [0, 1, 0.5] }
    const post = conditionGp(data, { lengthscales: [0.5, 0.5], outputscale: 1, noise: 0.01, mean: 0 })!
    const cands = candidateSet(post, createRng(3))
    expect(cands).toHaveLength(512)
    // Rank: row 1 (y 1), row 2 (0.5), row 0 (0); parent k is rank k mod 3, children within 4σ = 0.2.
    const rank = [1, 2, 0]
    for (let k = 0; k < TS_PARENTS; k++) for (let c = 0; c < TS_PER_PARENT; c++) {
      const child = cands[TS_UNIFORM + k * TS_PER_PARENT + c]!
      const parent = data.x[rank[k % 3]!]!
      for (let j = 0; j < 2; j++) if (child[j] !== 0 && child[j] !== 1) expect(Math.abs(child[j]! - parent[j]!)).toBeLessThanOrEqual(0.2)
    }
  })
})

describe("thompsonPick", () => {
  it("thompsonPick over 2,000 seeds on a 1-d model with a clear peak at 0.7: ≥ 80 % of picks within 0.1 of 0.7, and some outside (exploration)", () => {
    const post = peaked1d()
    let near = 0
    let outside = 0
    for (let s = 0; s < 2000; s++) {
      const rng = createRng(hashSeed(s, "ts-1d"))
      const pick = thompsonPick(post, candidateSet(post, rng), rng)
      expect(pick).not.toBeNull()
      if (Math.abs(pick!.x[0]! - 0.7) <= 0.1) near++
      else outside++
    }
    expect(near / 2000).toBeGreaterThanOrEqual(0.8)
    expect(outside).toBeGreaterThan(0)
  }, 120_000)

  it("thompsonPick is a pure function of (model, seed): same seed → same index; 20 picks equal fixtures/ts-picks.json", () => {
    const posterior = parsePosterior(readFileSync(POSTERIOR_FIXTURE, "utf8"))
    expect(posterior).not.toBeNull()
    const post = modelPosterior(posterior!.models.gpu!)
    expect(post).not.toBeNull()
    const run = () => Array.from({ length: 20 }, (_, i) => {
      const seed = hashSeed(i, "ts-picks")
      const rng = createRng(seed)
      const pick = thompsonPick(post!, candidateSet(post!, rng), rng)!
      return { seed, index: pick.index, x: pick.x.map(bits) }
    })
    const a = run()
    const b = run()
    expect(a).toEqual(b)
    const indices = new Set(a.map(p => p.index))
    expect(indices.size).toBeGreaterThan(1) // the seeds actually vary the pick
    const centreX = toUnit(SPACE(), CENTRE())
    const inc = incumbentOf(post!, centreX, 12345)!
    const out = { picks: a, incumbent: { x: inc.x.map(bits), mean: bits(inc.mean), pBeatsCentre: bits(inc.pBeatsCentre) } }
    if (process.env.TS_WRITE === "1") {
      expect(process.env.CI, "never regenerate the golden TS picks in CI").toBeUndefined()
      writeFileSync(PICKS, JSON.stringify(out, null, 2) + "\n")
    }
    expect(out).toEqual(JSON.parse(readFileSync(PICKS, "utf8")))
  })

  it("a Σ that fails every jitter step → thompsonPick null", () => {
    // A corrupted posterior: its factor is 0.001·I, so q = ‖L⁻¹k‖² is huge and the "covariance" kᵀ… − q is
    // strongly indefinite (variance clamps to 0, off-diagonals ≈ −1e6): no jitter up to 1e-2 makes it factor.
    const good = conditionGp({ x: [[0.3], [0.6]], y: [0, 1] }, { lengthscales: [0.4], outputscale: 1, noise: 0.01, mean: 0 })!
    const bad: GpPosterior = { ...good, chol: { L: new Float64Array([1e-3, 0, 0, 1e-3]), n: 2, jitter: 0 } }
    const cands = [[0.31], [0.32], [0.59], [0.61]]
    let draws = 0
    expect(thompsonPick(bad, cands, () => { draws++; return 0.5 })).toBeNull()
    expect(draws).toBe(0) // no z is drawn when no jitter step factors
    // The same candidates on the good posterior pick fine.
    expect(thompsonPick(good, cands, createRng(1))).not.toBeNull()
  })

  it("a NaN posterior, a bad candidate (wrong length, NaN, outside [0,1]) or no candidates → null, never a NaN or out-of-cube x", () => {
    const good = conditionGp({ x: [[0.3, 0.3], [0.6, 0.7]], y: [0, 1] }, { lengthscales: [0.4, 0.4], outputscale: 1, noise: 0.01, mean: 0 })!
    const nanPost: GpPosterior = { ...good, alpha: new Float64Array([Number.NaN, 0]) }
    expect(thompsonPick(nanPost, [[0.1, 0.1], [0.5, 0.5]], createRng(1))).toBeNull()
    const nanChol: GpPosterior = { ...good, chol: { ...good.chol, L: new Float64Array([Number.NaN, 0, 0, 1]) } }
    expect(thompsonPick(nanChol, [[0.1, 0.1], [0.5, 0.5]], createRng(1))).toBeNull()
    for (const cands of [[[0.1]], [[0.1, Number.NaN]], [[0.1, 1.5]], [[-0.1, 0.5]], []])
      expect(thompsonPick(good, cands, createRng(1)), JSON.stringify(cands)).toBeNull()
    // A NaN training row: candidateSet still runs, and the pick is null rather than a NaN genome.
    const nanRow: GpPosterior = { ...good, data: { x: [[Number.NaN, 0.3], [0.6, 0.7]], y: [0, 1] } }
    const rng = createRng(2)
    expect(thompsonPick(nanRow, candidateSet(nanRow, rng), rng)).toBeNull()
  })
})

describe("incumbentOf", () => {
  it("incumbentOf on a model peaked away from the centre gives pBeatsCentre > 0.95; on a flat model ≈ 0.5 ± 0.1", () => {
    const centreX = [0.5, 0.5]
    const peak = dataset(21, 2, 40, r => 1 - 4 * ((r[0]! - 0.85) * (r[0]! - 0.85) + (r[1]! - 0.2) * (r[1]! - 0.2)))
    const peaked = conditionGp(peak, { lengthscales: [0.4, 0.4], outputscale: 1, noise: 0.01, mean: 0 })!
    const inc = incumbentOf(peaked, centreX, hashSeed(0, "incumbent/peaked"))!
    expect(inc).not.toBeNull()
    expect(inc.pBeatsCentre).toBeGreaterThan(0.95)
    expect(Math.abs(inc.x[0]! - 0.85)).toBeLessThan(0.15)
    expect(Math.abs(inc.x[1]! - 0.2)).toBeLessThan(0.15)
    for (const v of inc.x) expect(v).toBe(round6(v)) // x is rounded as the gate writes it; mean and P are at that x
    expect(inc.mean).toBe(predict(peaked, [inc.x], false)!.mean[0])

    // Flat: targets are small noise around 0 with noise variance 0.5 — the mean barely moves anywhere.
    const noise = createRng(5)
    const flatData = dataset(22, 2, 40, () => 0.05 * (noise() - 0.5))
    const flat = conditionGp(flatData, { lengthscales: [0.4, 0.4], outputscale: 1, noise: 0.5, mean: 0 })!
    const f = incumbentOf(flat, centreX, hashSeed(0, "incumbent/flat"))!
    expect(f).not.toBeNull()
    expect(Math.abs(f.pBeatsCentre - 0.5)).toBeLessThanOrEqual(0.1)

    // Same seed → same incumbent; a centre of the wrong length or NaN → null.
    expect(incumbentOf(peaked, centreX, hashSeed(0, "incumbent/peaked"))).toEqual(inc)
    expect(incumbentOf(peaked, [0.5], 1)).toBeNull()
    expect(incumbentOf(peaked, [0.5, Number.NaN], 1)).toBeNull()
  })
})

const SPACE = () => parseGenomeSpace(JSON.parse(readFileSync(SPACE_FILE, "utf8")))
const CENTRE = () => parseCentre(JSON.parse(readFileSync(CENTRE_FILE, "utf8")), SPACE())

/**
 * fixtures/posterior-v2-search.json, built from a real fit: 40 seeded rows over the 12 searched genes of
 * knowledge/genome-space.json, fitGp's hyperparameters, and the incumbent from incumbentOf seeded by the model's sha
 * (ruling 4). The confirmation serves that incumbent against the committed centre.
 */
async function buildPosteriorFixture(): Promise<string> {
  const space = SPACE()
  const centre = CENTRE()
  const genes = searched(space).map(g => g.name)
  const d = genes.length
  const rng = createRng(hashSeed(0, "posterior-v2-fixture"))
  const rows: { learnerId: string; x: number[] }[] = []
  const y: number[] = []
  for (let i = 0; i < 40; i++) {
    const x = Array.from({ length: d }, () => round6(rng()))
    rows.push({ learnerId: "lrn_" + (0x1000 + 17 * i).toString(16).padStart(16, "0"), x })
    let s = 0
    for (let j = 0; j < d; j++) s += (j % 3 === 0 ? -1.5 : 0.3) * (x[j]! - 0.6) * (x[j]! - 0.6)
    y.push(round6(s + 0.2 * (rng() - 0.5)))
  }
  const fit = fitGp({ x: rows.map(r => r.x), y })!
  const model: GpModel = { genes, rows, y, hyper: fit.hyper, fit: { objective: fit.objective, start: fit.start, iterations: fit.iterations } }
  const modelSha = await sha256Hex(new TextEncoder().encode(serialise(model)))
  const post = modelPosterior(model)!
  const inc = incumbentOf(post, toUnit(space, centre), hashSeed(0, "incumbent/" + modelSha))!
  const centreBlock = { genomeSha: await genomeSha(space.schemaVersion, centre), values: centre }
  const candidateValues = fromUnit(space, inc.x, centre)
  const candidate = { genomeSha: await genomeSha(space.schemaVersion, candidateValues), values: candidateValues }
  const posterior = {
    schemaVersion: 2, genomeSchema: space.schemaVersion, evaluator: "search-composite@1",
    sourceCommit: "b".repeat(40), mode: "search", centre: centreBlock,
    calibration: { genomes: [centreBlock], need: [0] },
    models: { gpu: model },
    incumbent: { x: inc.x, genomeSha: candidate.genomeSha, mean: inc.mean, pBeatsCentre: inc.pBeatsCentre },
    confirmation: {
      id: 3, seedBase: 13000, candidate, centre: centreBlock,
      target: { candidate: 24, centre: 24, newborn: 60 }, counted: { candidate: 17, centre: 15, newborn: 38 },
      need: { candidate: 7, centre: 9, newborn: 22 },
    },
    diagnostics: {},
    page: { eligibleLives: 214, lastPromotion: null, calibrationLives: 30, calibrationTarget: 30, looks: 1, rowsSinceDeclaration: 40 },
  }
  return serialise(posterior)
}

describe("the v2 posterior fixture", () => {
  it("fixtures/posterior-v2-search.json is a genuine model: a real fitGp of its 40 rows over the 12 searched genes, and its incumbent from incumbentOf seeded by the model's sha", async () => {
    const text = await buildPosteriorFixture()
    if (process.env.POSTERIOR_FIXTURE_WRITE === "1") {
      expect(process.env.CI, "never regenerate the posterior fixture in CI").toBeUndefined()
      writeFileSync(POSTERIOR_FIXTURE, text)
    }
    expect(text).toBe(readFileSync(POSTERIOR_FIXTURE, "utf8"))
    const p = parsePosterior(text)
    expect(p).not.toBeNull()
    expect(p!.models.gpu!.genes).toHaveLength(12)
    expect(p!.models.gpu!.rows).toHaveLength(40)
    expect(p!.incumbent!.genomeSha).toBe(p!.confirmation!.candidate.genomeSha)
    expect(p!.incumbent!.genomeSha).not.toBe(p!.centre.genomeSha)
  }, 60_000)
})

describe("modelPosterior", () => {
  it("is conditionGp of the shipped rows and hyper; a model whose genes and length scales disagree → null", () => {
    const model: GpModel = {
      genes: ["lr", "gamma"], rows: [{ learnerId: "lrn_0000000000000001", x: [0.2, 0.4] }, { learnerId: "lrn_0000000000000002", x: [0.7, 0.1] }],
      y: [0.3, -0.2], hyper: { lengthscales: [0.5, 0.9], outputscale: 1.2, noise: 0.01, mean: 0.1 },
      fit: { objective: 0, start: 0, iterations: [0, 0, 0, 0] },
    }
    const a = modelPosterior(model)!
    const b = conditionGp({ x: model.rows.map(r => r.x), y: model.y }, model.hyper)!
    expect(Array.from(a.alpha)).toEqual(Array.from(b.alpha))
    expect(Array.from(a.chol.L)).toEqual(Array.from(b.chol.L))
    expect(modelPosterior({ ...model, genes: ["lr"] })).toBeNull()
    expect(modelPosterior({ ...model, y: [0.3] })).toBeNull()
  })
})
