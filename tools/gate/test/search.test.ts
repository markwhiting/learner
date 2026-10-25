import { describe, it, expect, vi, beforeEach } from "vitest"
import { createRng, hashSeed } from "@blanktab/detmath"
import * as metaPkg from "@blanktab/meta"
import {
  fromUnit, genomeSha, incumbentOf, modelPosterior, round6, searched, sha256Hex, toUnit,
  type GenomeSpace, type Incumbent,
} from "@blanktab/meta"
import { lid } from "../../../packages/evaluators/test/fixtures/build.ts"
import { buildModel, FIT_ROWS, incumbentFor, MAX_ROWS, MIN_ROWS, rowSubset, triggered, TRIGGER_NEW_ROWS, TRIGGER_P } from "../search.ts"
import { serialise } from "../serialise.ts"
import { CENTRE, SPACE } from "./meta-fixture.ts"

// A pass-through spy on the fit: it counts the fits, and one test sees the rows the fit is given.
vi.mock("@blanktab/meta", async importOriginal => {
  const m = await importOriginal<typeof import("@blanktab/meta")>()
  return { ...m, fitGp: vi.fn(m.fitGp) }
})
const fitSpy = vi.mocked(metaPkg.fitGp)
beforeEach(() => { fitSpy.mockClear() })

const D = searched(SPACE).length
/** n rows of genomes at seeded unit points, with a smooth target and a little noise. */
function rowsOf(n: number, seed = "search-test") {
  const rng = createRng(hashSeed(0, seed))
  const rows = [], y: number[] = []
  for (let i = 0; i < n; i++) {
    const u = Array.from({ length: D }, () => rng())
    rows.push({ learnerId: lid(0x2000 + 7 * i), genome: fromUnit(SPACE, u, CENTRE) })
    let s = 0
    for (let j = 0; j < D; j++) s += (j % 3 === 0 ? -1.5 : 0.3) * (u[j]! - 0.6) * (u[j]! - 0.6)
    y.push(s + 0.2 * (rng() - 0.5))
  }
  return { rows, y }
}

describe("rowSubset (spec §4.3)", () => {
  it("rowSubset keeps all ≤ 1000; of 1,200 ids keeps the 1,000 with the largest subset hash, sorted", () => {
    const ids = Array.from({ length: 1200 }, (_, i) => lid(0x9000 + 13 * ((i * 37) % 1200))) // unsorted
    const few = ids.slice(0, MAX_ROWS)
    expect(rowSubset(few)).toEqual([...few].sort())
    const kept = rowSubset(ids)
    expect(kept).toHaveLength(MAX_ROWS)
    expect(kept).toEqual([...kept].sort())
    expect(new Set(kept).size).toBe(MAX_ROWS)
    const key = (id: string) => hashSeed(0, "subset/" + id)
    const keptSet = new Set(kept)
    const dropped = ids.filter(id => !keptSet.has(id))
    expect(dropped).toHaveLength(200)
    expect(Math.min(...kept.map(key))).toBeGreaterThanOrEqual(Math.max(...dropped.map(key)))
    expect(rowSubset([...ids].reverse())).toEqual(kept) // the input order never matters
  })
})

describe("buildModel (spec §4.3)", () => {
  it("buildModel below d + 1 rows → no model, reason 'fewer than 13 rows'", () => {
    expect(MIN_ROWS(D)).toBe(13)
    const { rows, y } = rowsOf(12)
    const r = buildModel(rows, y, SPACE, null)
    expect(r.model).toBeNull()
    expect(r.reason).toBe("fewer than 13 rows")
    expect(fitSpy).not.toHaveBeenCalled()
  })

  it("buildModel with previous identical rows reuses hyper; refitting gives byte-identical hyper", () => {
    const { rows, y } = rowsOf(20)
    const first = buildModel(rows, y, SPACE, null)
    expect(first.reason).toBeNull()
    expect(first.run).toMatchObject({ rows: 20, fitRows: 20, refit: true })
    const model = first.model!
    expect(fitSpy).toHaveBeenCalledTimes(1)
    expect(model.genes).toEqual(searched(SPACE).map(g => g.name))
    expect(model.rows.map(r => r.learnerId)).toEqual(rows.map(r => r.learnerId))
    // x is the logged genome in the cube, rounded to 6 decimals: the file is the exact training set.
    model.rows.forEach((r, i) => expect(r.x).toEqual(toUnit(SPACE, rows[i]!.genome).map(round6)))
    expect(model.y).toEqual(y)

    const reused = buildModel(rows, y, SPACE, model)
    expect(fitSpy).toHaveBeenCalledTimes(1) // no refit
    expect(reused.run.refit).toBe(false)
    expect(serialise(reused.model)).toBe(serialise(model))

    const refit = buildModel(rows, y, SPACE, null)
    expect(fitSpy).toHaveBeenCalledTimes(2)
    expect(serialise(refit.model!.hyper)).toBe(serialise(model.hyper))
    expect(serialise(refit.model)).toBe(serialise(model))

    // A previous model on other data is not reused: one changed target refits.
    const moved = buildModel(rows, y.map((v, i) => (i === 3 ? v + 0.5 : v)), SPACE, model)
    expect(fitSpy).toHaveBeenCalledTimes(3)
    expect(moved.run.refit).toBe(true)
  }, 60_000)

  it("a fit that throws (here: 14 searched genes need more Sobol dimensions than exist) → no model, with the reason; a non-finite target → no model", () => {
    const wide: GenomeSpace = {
      schemaVersion: 1,
      genes: Array.from({ length: 14 }, (_, j) => ({ name: `g${j}`, type: "continuous", min: 0, max: 1, centre: 0.5, scale: "linear", owner: "gp", target: "learner" })),
    } as unknown as GenomeSpace
    const centre = Object.fromEntries(wide.genes.map(g => [g.name, 0.5]))
    const rows = Array.from({ length: 20 }, (_, i) => ({ learnerId: lid(0x3000 + i), genome: fromUnit(wide, Array.from({ length: 14 }, (_, j) => ((i + j) % 10) / 10), centre) }))
    const r = buildModel(rows, rows.map((_, i) => i / 20), wide, null)
    expect(r.model).toBeNull()
    expect(r.reason).toMatch(/^the fit threw: fitGp: 14 searched genes need 17 Sobol dimensions/)

    const { rows: ok, y } = rowsOf(20)
    expect(buildModel(ok, y.map((v, i) => (i === 0 ? NaN : v)), SPACE, null)).toMatchObject({ model: null, reason: "a target is not finite" })
  })

  it("an error of fitGp other than its own data refusal is a bug, and is thrown", () => {
    const { rows, y } = rowsOf(20)
    fitSpy.mockImplementationOnce(() => { throw new TypeError("boom") })
    expect(() => buildModel(rows, y, SPACE, null)).toThrow("boom")
  })

  it("above FIT_ROWS rows the hyperparameters are fitted on the FIT_ROWS rows with the largest subset hash; the model keeps every row", () => {
    const n = FIT_ROWS + 40
    const { rows, y } = rowsOf(n, "fit-rows")
    // The fit itself is faked here (its cost at FIT_ROWS is ~20 s); this test is about which rows it is given.
    let seen: { x: readonly (readonly number[])[]; y: readonly number[] } | null = null
    fitSpy.mockImplementationOnce(data => {
      seen = data
      return { hyper: { lengthscales: Array(D).fill(2), outputscale: 1, noise: 0.5, mean: 0 }, objective: -1, start: 0, iterations: [3, 0, 2, 1] }
    })
    const r = buildModel(rows, y, SPACE, null)
    expect(r.reason).toBeNull()
    expect(r.model!.rows).toHaveLength(n)
    expect(r.run).toEqual({ rows: n, fitRows: FIT_ROWS, refit: true, startsWithNoStep: [1] })
    const fitIds = [...rows.map(x => x.learnerId)].sort((a, b) => hashSeed(0, "subset/" + b) - hashSeed(0, "subset/" + a) || (a < b ? -1 : 1)).slice(0, FIT_ROWS)
    const want = new Set(fitIds)
    const idx = r.model!.rows.flatMap((row, i) => (want.has(row.learnerId) ? [i] : []))
    expect(seen!.x).toEqual(idx.map(i => r.model!.rows[i]!.x))
    expect(seen!.y).toEqual(idx.map(i => y[i]))
  }, 60_000)
})

describe("incumbentFor and triggered (spec §4.4)", () => {
  it("incumbentFor seeds incumbentOf with hashSeed(0, 'incumbent/' + sha256(serialise(model))); its genomeSha is fromUnit(x*)'s", async () => {
    const { rows, y } = rowsOf(20)
    const model = buildModel(rows, y, SPACE, null).model!
    const inc = (await incumbentFor(model, SPACE, CENTRE))!
    const modelSha = await sha256Hex(new TextEncoder().encode(serialise(model)))
    const want = incumbentOf(modelPosterior(model)!, toUnit(SPACE, CENTRE), hashSeed(0, "incumbent/" + modelSha))!
    expect(inc).toEqual({ ...want, genomeSha: await genomeSha(SPACE.schemaVersion, fromUnit(SPACE, want.x, CENTRE)) })
    expect(inc.x).toEqual(inc.x.map(round6))
    expect(await incumbentFor(model, SPACE, CENTRE)).toEqual(inc)
  }, 60_000)

  it("triggered: each of the four conditions false alone → false", () => {
    const centreSha = "c".repeat(64)
    const inc: Incumbent = { x: [0.5], genomeSha: "a".repeat(64), mean: 0.4, pBeatsCentre: TRIGGER_P }
    expect(triggered(inc, centreSha, TRIGGER_NEW_ROWS, false)).toBe(true)
    expect(triggered({ ...inc, pBeatsCentre: 0.9499 }, centreSha, TRIGGER_NEW_ROWS, false)).toBe(false)
    expect(triggered(inc, centreSha, TRIGGER_NEW_ROWS - 1, false)).toBe(false)
    expect(triggered(inc, centreSha, TRIGGER_NEW_ROWS, true)).toBe(false)
    expect(triggered({ ...inc, genomeSha: centreSha }, centreSha, TRIGGER_NEW_ROWS, false)).toBe(false)
    expect(triggered(null, centreSha, 1000, false)).toBe(false)
  })
})
