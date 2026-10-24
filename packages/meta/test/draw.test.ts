import { describe, it, expect } from "vitest"
import { createHash } from "node:crypto"
import { readFileSync, writeFileSync } from "node:fs"
import { chiSquareSf } from "@blanktab/detmath"
import {
  birthDraw, checkGenome, drawSeedOf, evalPanelOf, fromUnit, genomeSha, parseCentre, parseGenomeSpace, parsePosterior, searched,
  PANELS, UNIFORM_SHARE,
  type Draw, type DrawInput, type Genome, type Mode, type Posterior,
} from "@blanktab/meta"

const readJson = (name: string): unknown => JSON.parse(readFileSync(new URL(`../../../knowledge/${name}`, import.meta.url), "utf8"))
const space = parseGenomeSpace(readJson("genome-space.json"))
const centre = parseCentre(readJson("genome.json"), space)
const FIXTURE = new URL("./fixtures/draws.json", import.meta.url)
const HEX = (s: string) => createHash("sha256").update(s).digest("hex")

/** 10 calibration genomes: the centre, then 9 fixed points of the cube (the gate's Sobol points stand in here). */
const n = searched(space).length
const CALIB: Genome[] = [centre, ...Array.from({ length: 9 }, (_, k) => fromUnit(space, Array.from({ length: n }, (_, j) => ((k + 1) * 0.1 + j * 0.037) % 1), centre))]

const centreSha = await genomeSha(space.schemaVersion, centre)
const CALIB_SHAS = await Promise.all(CALIB.map(g => genomeSha(space.schemaVersion, g)))

/** A 05a (schemaVersion 1) posterior file: its page has no rowsSinceDeclaration. Its bytes pin fixtures/draws.json. */
type V1File = Omit<Posterior, "schemaVersion" | "page"> & { schemaVersion: 1; page: Omit<Posterior["page"], "rowsSinceDeclaration"> }
function posterior(mode: Mode, need: number[] = CALIB.map(() => 3), genomes: Genome[] = CALIB): V1File {
  return {
    schemaVersion: 1, genomeSchema: 1, evaluator: "search-composite@1", sourceCommit: "a".repeat(40), mode,
    centre: { genomeSha: centreSha, values: centre },
    calibration: { genomes: genomes.map((values, i) => ({ genomeSha: CALIB_SHAS[i] ?? HEX(`g${i}`), values })), need },
    models: {}, incumbent: null, confirmation: null, diagnostics: {},
    page: { eligibleLives: 0, lastPromotion: null, calibrationLives: 0, calibrationTarget: 30, looks: 0 },
  }
}

/** The input a tab would build: the posterior is parsed from its own bytes, and posteriorSha is their sha256. */
function input(learnerId: string, p: V1File | null, widthsClass: "gpu" | "cpu" = "gpu"): DrawInput {
  if (p === null) return { learnerId, space, centre, centreSha, widthsClass, posterior: null, posteriorSha: null }
  const text = JSON.stringify(p)
  const parsed = parsePosterior(text)
  if (parsed === null) throw new Error("test posterior does not parse")
  return { learnerId, space, centre, centreSha, widthsClass, posterior: parsed, posteriorSha: HEX(text) }
}

const UNIFORM_PAUSED_ID = "lrn_fixture_paused_5"
const CALIB_NONCENTRE_ID = "lrn_fixture_calib_0"
const ids = (k: number, tag: string) => Array.from({ length: k }, (_, i) => `lrn_${tag}_${i.toString(36)}`)

describe("birthDraw", () => {
  it("birthDraw is a pure function: the same input gives a deep-equal Draw; 22 draws equal fixtures/draws.json", () => {
    const cases: DrawInput[] = []
    const modes: (Mode | null)[] = [null, "calibrate", "search", "paused", "flat"]
    for (let i = 0; i < 20; i++) {
      const mode = modes[i % modes.length]!
      cases.push(input(`lrn_fixture_${i}`, mode === null ? null : posterior(mode), i % 3 === 2 ? "cpu" : "gpu"))
    }
    // Added cases (review round 1), ids chosen so the fixture pins a uniform draw in mode paused and a calibrate
    // draw that picks a non-centre genome.
    cases.push(input(UNIFORM_PAUSED_ID, posterior("paused")))
    cases.push(input(CALIB_NONCENTRE_ID, posterior("calibrate")))
    const draws = cases.map(c => birthDraw(c))
    expect(draws[20]!.method).toBe("uniform")
    expect(draws[21]!.method).toBe("calibrate")
    expect(draws[21]!.calibrationIndex).toBeGreaterThan(0)
    for (let i = 0; i < cases.length; i++) expect(birthDraw(cases[i]!)).toEqual(draws[i])
    // The 20 cases cover every method 05a has.
    expect(new Set(draws.map(d => d.method))).toEqual(new Set(["centre", "calibrate", "uniform"]))
    if (process.env.DRAW_WRITE === "1") {
      expect(process.env.CI, "never regenerate the golden draws in CI").toBeUndefined()
      writeFileSync(FIXTURE, JSON.stringify(draws, null, 2) + "\n")
    }
    const fixture = JSON.parse(readFileSync(FIXTURE, "utf8")) as Draw[]
    expect(draws).toEqual(fixture)
  })

  it("no posterior → centre, drawSeed null, genome = centre", () => {
    for (const id of ids(50, "none")) {
      for (const w of ["gpu", "cpu"] as const) {
        const d = birthDraw(input(id, null, w))
        expect(d).toEqual({ role: "search", method: "centre", drawSeed: null, posteriorSha: null, evalPanel: evalPanelOf(id), genome: centre })
      }
    }
  })

  it("programming errors throw: posterior and posteriorSha half-null, a sha that is not 64 lowercase hex", () => {
    const p = parsePosterior(JSON.stringify(posterior("flat")))!
    const at = { learnerId: "lrn_x", space, centre, centreSha, widthsClass: "gpu" as const }
    expect(() => birthDraw({ ...at, posterior: p, posteriorSha: null })).toThrow(/null together/)
    expect(() => birthDraw({ ...at, posterior: null, posteriorSha: HEX("x") })).toThrow(/null together/)
    for (const sha of ["", "abc", HEX("x").toUpperCase(), HEX("x") + "0", "g".repeat(64)])
      expect(() => birthDraw({ ...at, posterior: p, posteriorSha: sha }), sha).toThrow(/64 lowercase hex/)
    expect(() => birthDraw({ ...at, centreSha: "nope", posterior: null, posteriorSha: null })).toThrow(/centreSha/)
  })

  it("a posterior that does not fit the tab's knowledge → centre with a reason, drawSeed and posteriorSha kept", () => {
    const check = (p: V1File, why: RegExp, w: "gpu" | "cpu" = "gpu") => {
      const i = input("lrn_mismatch", p, w)
      const d = birthDraw(i)
      expect(d.method).toBe("centre")
      expect(d.genome).toEqual(centre)
      expect(d.reason).toMatch(why)
      expect(d.posteriorSha).toBe(i.posteriorSha)
      expect(d.drawSeed).toBe(drawSeedOf("lrn_mismatch", i.posteriorSha!))
    }
    for (const mode of ["calibrate", "flat"] as const) {
      check({ ...posterior(mode), genomeSchema: 2 }, /genomeSchema 2 is not the space's 1/)
      check({ ...posterior(mode), centre: { genomeSha: HEX("other"), values: centre } }, /posterior built for another centre/)
    }
    const { lr: _lr, ...noLr } = CALIB[1]!
    void _lr
    const only1 = CALIB.map((_, i) => i === 1 ? 1 : 0)
    check(posterior("calibrate", only1, CALIB.map((g, i) => i === 1 ? noLr : g)), /calibration genome 1 is invalid: missing gene lr/)
    check(posterior("calibrate", only1, CALIB.map((g, i) => i === 1 ? { ...g, lr: 1 } : g)), /calibration genome 1 is invalid: lr 1 is outside/)
    // A valid draw carries no reason.
    expect(birthDraw(input("lrn_ok", posterior("calibrate"))).reason).toBeUndefined()
  })

  it("calibrate on gpu: every draw is calibrate; over 2,000 learnerIds the genome frequencies follow need [3,0,…] (only index 0) and [1,1,…] (uniform, χ² p > 0.001)", () => {
    const only0 = posterior("calibrate", CALIB.map((_, i) => i === 0 ? 3 : 0))
    for (const id of ids(2000, "c0")) {
      const d = birthDraw(input(id, only0))
      expect(d.method).toBe("calibrate")
      expect(d.role).toBe("search")
      expect(d.calibrationIndex).toBe(0)
      expect(d.genome).toEqual(CALIB[0])
    }
    const counts = CALIB.map(() => 0)
    const even = posterior("calibrate", CALIB.map(() => 1))
    for (const id of ids(2000, "c1")) {
      const d = birthDraw(input(id, even))
      expect(d.method).toBe("calibrate")
      expect(d.genome).toEqual(CALIB[d.calibrationIndex!])
      counts[d.calibrationIndex!]!++
    }
    const e = 2000 / CALIB.length
    const chi2 = counts.reduce((s, c) => s + (c - e) * (c - e) / e, 0)
    expect(chiSquareSf(chi2, CALIB.length - 1)).toBeGreaterThan(0.001)
    // Σneed = 0: uniform over the genomes.
    const zero = posterior("calibrate", CALIB.map(() => 0))
    const seen = new Set(ids(2000, "cz").map(id => birthDraw(input(id, zero)).calibrationIndex))
    expect(seen.size).toBe(CALIB.length)
  })

  it("calibrate on cpu: only uniform or centre; uniform share 0.2 ± 0.03 over 2,000 ids", () => {
    let uniform = 0
    for (const id of ids(2000, "cpu")) {
      const d = birthDraw(input(id, posterior("calibrate"), "cpu"))
      expect(["uniform", "centre"]).toContain(d.method)
      expect(d.calibrationIndex).toBeUndefined()
      if (d.method === "uniform") uniform++
      else expect(d.genome).toEqual(centre)
    }
    expect(Math.abs(uniform / 2000 - UNIFORM_SHARE)).toBeLessThanOrEqual(0.03)
  })

  it("flat/paused/search: uniform share 0.2 ± 0.03; uniform genomes pass checkGenome and keep fixed genes at centre", () => {
    const fixed = space.genes.filter(g => g.owner !== "gp").map(g => g.name)
    for (const mode of ["flat", "paused", "search"] as const) {
      for (const w of ["gpu", "cpu"] as const) {
        let uniform = 0
        const p = posterior(mode)
        for (const id of ids(2000, `${mode}${w}`)) {
          const d = birthDraw(input(id, p, w))
          expect(d.role).toBe("search")
          expect(typeof d.drawSeed).toBe("number")
          if (d.method === "centre") { expect(d.genome).toEqual(centre); continue }
          expect(d.method).toBe("uniform")
          uniform++
          expect(checkGenome(space, d.genome)).toBeNull()
          for (const name of fixed) expect(d.genome[name]).toBe(centre[name])
        }
        expect(Math.abs(uniform / 2000 - UNIFORM_SHARE), `${mode} ${w}`).toBeLessThanOrEqual(0.03)
      }
    }
  })

  it("evalPanelOf is in 1..16 and roughly uniform over 1,600 ids", () => {
    const counts = Array.from({ length: PANELS }, () => 0)
    for (const id of ids(1600, "panel")) {
      const p = evalPanelOf(id)
      expect(Number.isInteger(p) && p >= 1 && p <= PANELS).toBe(true)
      counts[p - 1]!++
    }
    const e = 1600 / PANELS
    const chi2 = counts.reduce((s, c) => s + (c - e) * (c - e) / e, 0)
    expect(chiSquareSf(chi2, PANELS - 1)).toBeGreaterThan(0.001)
    expect(Math.min(...counts)).toBeGreaterThan(0)
  })
})
