import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { parsePosterior, POSTERIOR_SCHEMA } from "@blanktab/meta"

const COMMITTED = readFileSync(new URL("../../../data/meta/posterior.json", import.meta.url), "utf8")
const V2_TEXT = readFileSync(new URL("./fixtures/posterior-v2-search.json", import.meta.url), "utf8")
type Json = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
const v1 = (): Json => JSON.parse(COMMITTED) as Json
const v2 = (): Json => JSON.parse(V2_TEXT) as Json
const text = (o: unknown) => JSON.stringify(o)

describe("parsePosterior: schemaVersion 1 (05a's files)", () => {
  it("the committed data/meta/posterior.json (v1) parses, as models {}, incumbent null, confirmation null, rowsSinceDeclaration 0", () => {
    const raw = v1()
    expect(raw.schemaVersion).toBe(1)
    const p = parsePosterior(COMMITTED)
    expect(p).not.toBeNull()
    expect(p!.models).toEqual({})
    expect(p!.incumbent).toBeNull()
    expect(p!.confirmation).toBeNull()
    expect(p!.page).toEqual({ ...raw.page, rowsSinceDeclaration: 0 })
    // Everything else is the file as written.
    expect(p).toEqual({ ...raw, page: { ...raw.page, rowsSinceDeclaration: 0 } })
  })

  it("a v1 file is checked exactly as 05a checks it: models and incumbent need only be objects, and read as {} and null", () => {
    const p = parsePosterior(text({ ...v1(), models: { gpu: { anything: 1 } }, incumbent: { x: [] }, confirmation: { id: 1 } }))
    expect(p).not.toBeNull()
    expect(p!.models).toEqual({})
    expect(p!.incumbent).toBeNull()
    expect(p!.confirmation).toBeNull()
  })

  it("v1: garbage, wrong schemaVersion, a need/genomes length mismatch → null (the 05a cases)", () => {
    const good = v1()
    const bad: [string, string][] = [
      ["garbage", "{not json"],
      ["empty", ""],
      ["an array", "[]"],
      ["null", "null"],
      ["schemaVersion 2 on a v1 body (no rowsSinceDeclaration)", text({ ...good, schemaVersion: 2 })],
      ["schemaVersion 0", text({ ...good, schemaVersion: 0 })],
      ["schemaVersion \"1\"", text({ ...good, schemaVersion: "1" })],
      ["need/genomes length mismatch", text({ ...good, calibration: { ...good.calibration, need: [1, 2] } })],
      ["unknown mode", text({ ...good, mode: "ts" })],
      ["negative need", text({ ...good, calibration: { ...good.calibration, need: good.calibration.need.map(() => -1) } })],
      ["calibrate with no genomes", text({ ...good, calibration: { genomes: [], need: [] } })],
      ["a bad genomeSha", text({ ...good, centre: { ...good.centre, genomeSha: "xyz" } })],
      ["an unknown top-level key", text({ ...good, extra: 1 })],
      ["a missing key", text({ ...good, page: undefined })],
      ["v1 with rowsSinceDeclaration", text({ ...good, page: { ...good.page, rowsSinceDeclaration: 0 } })],
      ["models not an object", text({ ...good, models: [] })],
    ]
    for (const [why, t] of bad) expect(parsePosterior(t), why).toBeNull()
  })
})

describe("parsePosterior: schemaVersion 2", () => {
  it("a v2 search posterior with a model, an incumbent and a confirmation parses (fixture fixtures/posterior-v2-search.json)", () => {
    expect(POSTERIOR_SCHEMA).toBe(2)
    const p = parsePosterior(V2_TEXT)
    expect(p).not.toBeNull()
    expect(p).toEqual(v2())
    expect(p!.schemaVersion).toBe(2)
    expect(p!.mode).toBe("search")
    expect(p!.models.gpu!.genes).toHaveLength(3)
    expect(p!.models.gpu!.rows).toHaveLength(4)
    expect(p!.incumbent).not.toBeNull()
    expect(p!.confirmation!.seedBase).toBe(10_000 + 1_000 * p!.confirmation!.id)
    // Its parts are each optional in search: no model, incumbent or confirmation also parses.
    expect(parsePosterior(text({ ...v2(), models: {}, incumbent: null, confirmation: null }))).not.toBeNull()
    expect(parsePosterior(text({ ...v2(), confirmation: null }))).not.toBeNull()
    expect(parsePosterior(text({ ...v2(), incumbent: null }))).not.toBeNull()
  })

  it("mode guard: the same v2 file with mode calibrate, paused or flat → null; with models {} but a confirmation in calibrate → null", () => {
    for (const mode of ["calibrate", "paused", "flat"]) {
      expect(parsePosterior(text({ ...v2(), mode })), mode).toBeNull()
      expect(parsePosterior(text({ ...v2(), mode, incumbent: null, confirmation: null })), `${mode}: a model alone`).toBeNull()
      expect(parsePosterior(text({ ...v2(), mode, models: {}, incumbent: null })), `${mode}: a confirmation alone`).toBeNull()
      // The same file with nothing of 05b's parses in that mode.
      expect(parsePosterior(text({ ...v2(), mode, models: {}, incumbent: null, confirmation: null })), `${mode}: empty`).not.toBeNull()
    }
    expect(parsePosterior(text({ ...v2(), mode: "calibrate", models: {}, incumbent: null }))).toBeNull()
  })

  it("v2 violations → null, one case each: rows/y length mismatch, x outside [0,1], 1001 rows, unsorted learnerIds, noise 5e-4, lengthscales of the wrong length, seedBase ≠ 10000 + 1000·id, need > target, candidate = centre, incumbent without a model", () => {
    const withGpu = (f: (g: Json) => void) => { const o = v2(); f(o.models.gpu); return text(o) }
    const withConf = (f: (c: Json) => void) => { const o = v2(); f(o.confirmation); return text(o) }
    const id = (k: number) => "lrn_" + k.toString(16).padStart(16, "0")
    const bad: [string, string][] = [
      ["rows/y length mismatch", withGpu(g => { g.y.pop() })],
      ["x outside [0,1]", withGpu(g => { g.rows[1].x[2] = 1.000001 })],
      ["x below 0", withGpu(g => { g.rows[0].x[0] = -0.1 })],
      ["1001 rows", withGpu(g => {
        g.rows = Array.from({ length: 1001 }, (_, k) => ({ learnerId: id(k + 1), x: [0.5, 0.5, 0.5] }))
        g.y = g.rows.map(() => 0)
      })],
      ["unsorted learnerIds", withGpu(g => { [g.rows[0], g.rows[1]] = [g.rows[1], g.rows[0]] })],
      ["duplicate learnerIds", withGpu(g => { g.rows[1].learnerId = g.rows[0].learnerId })],
      ["a bad learnerId", withGpu(g => { g.rows[0].learnerId = "lrn_short" })],
      ["noise 5e-4", withGpu(g => { g.hyper.noise = 5e-4 })],
      ["lengthscales of the wrong length", withGpu(g => { g.hyper.lengthscales.pop() })],
      ["a zero lengthscale", withGpu(g => { g.hyper.lengthscales[0] = 0 })],
      ["outputscale 0", withGpu(g => { g.hyper.outputscale = 0 })],
      ["x of the wrong length", withGpu(g => { g.rows[2].x.pop() })],
      ["duplicate genes", withGpu(g => { g.genes[1] = g.genes[0] })],
      ["no genes", withGpu(g => { g.genes = []; g.hyper.lengthscales = []; for (const r of g.rows) r.x = [] })],
      ["a bad gene name", withGpu(g => { g.genes[0] = "1lr" })],
      ["fit.start 4", withGpu(g => { g.fit.start = 4 })],
      ["fit.iterations of length 3", withGpu(g => { g.fit.iterations.pop() })],
      ["a fractional fit iteration", withGpu(g => { g.fit.iterations[0] = 1.5 })],
      ["a model under another key", (() => { const o = v2(); o.models = { cpu: o.models.gpu }; return text(o) })()],
      ["an extra model key", (() => { const o = v2(); o.models.cpu = o.models.gpu; return text(o) })()],
      ["an extra key in a model", withGpu(g => { g.extra = 1 })],
      ["seedBase ≠ 10000 + 1000·id", withConf(c => { c.seedBase = 14000 })],
      ["a negative confirmation id", withConf(c => { c.id = -1; c.seedBase = 9000 })],
      ["need > target", withConf(c => { c.need.newborn = 61 })],
      ["a missing arm", withConf(c => { delete c.counted.newborn })],
      ["candidate = centre", withConf(c => { c.candidate.genomeSha = c.centre.genomeSha })],
      ["incumbent without a model", (() => { const o = v2(); o.models = {}; return text(o) })()],
      ["incumbent x of the wrong length", (() => { const o = v2(); o.incumbent.x.pop(); return text(o) })()],
      ["incumbent pBeatsCentre > 1", (() => { const o = v2(); o.incumbent.pBeatsCentre = 1.5; return text(o) })()],
      ["a bad incumbent genomeSha", (() => { const o = v2(); o.incumbent.genomeSha = "x"; return text(o) })()],
      ["page without rowsSinceDeclaration", (() => { const o = v2(); delete o.page.rowsSinceDeclaration; return text(o) })()],
      ["a negative rowsSinceDeclaration", (() => { const o = v2(); o.page.rowsSinceDeclaration = -1; return text(o) })()],
      ["an unknown top-level key", text({ ...v2(), extra: 1 })],
    ]
    for (const [why, t] of bad) expect(parsePosterior(t), why).toBeNull()
  })

  it("schemaVersion 3 → null", () => {
    expect(parsePosterior(text({ ...v2(), schemaVersion: 3 }))).toBeNull()
    expect(parsePosterior(text({ ...v2(), schemaVersion: 3, models: {}, incumbent: null, confirmation: null }))).toBeNull()
  })

  it("05a's parser refuses every v2 file: its schemaVersion is 2", () => {
    // 05a's parsePosterior starts with `if (v.schemaVersion !== 1) return null`, so a 05a tab draws centre from any
    // v2 file until it reloads (Review Focus 1). Task 13 checks this against the real 05a code.
    expect((JSON.parse(V2_TEXT) as { schemaVersion: unknown }).schemaVersion).toBe(2)
    expect((JSON.parse(V2_TEXT) as { schemaVersion: unknown }).schemaVersion).not.toBe(1)
  })
})
