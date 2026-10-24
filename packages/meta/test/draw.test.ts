import { describe, it, expect } from "vitest"
import { createHash } from "node:crypto"
import { readFileSync, writeFileSync } from "node:fs"
import { chiSquareSf, createRng } from "@blanktab/detmath"
import {
  armOfSeed, birthDraw, checkGenome, decodeGenome, drawSeedOf, evalPanelOf, fromUnit, genomeSha, parseCentre, parseGenomeSpace,
  parsePosterior, searched,
  ARM_OF_J, CONFIRM_SEEDS, CONFIRM_SHARE_END, PANELS, UNIFORM_SHARE,
  type ArmName, type Draw, type DrawInput, type Genome, type Mode, type Posterior,
} from "@blanktab/meta"
import { candidateSet, modelPosterior, thompsonPick } from "../src/thompson"

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

/** 05a's golden draws (fixtures/draws.json, frozen), and a 05a Draw as 05b code returns it. */
const fixture05a = () => JSON.parse(readFileSync(FIXTURE, "utf8")) as Omit<Draw, "evalSeed" | "confirmation">[]
const as05b = (d: Omit<Draw, "evalSeed" | "confirmation">): Draw => ({ ...d, evalSeed: null, confirmation: null })

const UNIFORM_PAUSED_ID = "lrn_fixture_paused_5"
const CALIB_NONCENTRE_ID = "lrn_fixture_calib_0"
const ids = (k: number, tag: string) => Array.from({ length: k }, (_, i) => `lrn_${tag}_${i.toString(36)}`)

/** The 22 inputs that pin fixtures/draws.json (05a): 20 by mode and widths class, and two added in 05a's review. */
function cases05a(): DrawInput[] {
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
  return cases
}

describe("birthDraw", () => {
  it("birthDraw is a pure function: the same input gives a deep-equal Draw; 22 draws equal fixtures/draws.json", () => {
    const cases = cases05a()
    const draws = cases.map(c => birthDraw(c))
    expect(draws[20]!.method).toBe("uniform")
    expect(draws[21]!.method).toBe("calibrate")
    expect(draws[21]!.calibrationIndex).toBeGreaterThan(0)
    for (let i = 0; i < cases.length; i++) expect(birthDraw(cases[i]!)).toEqual(draws[i])
    // The 20 cases cover every method 05a has.
    expect(new Set(draws.map(d => d.method))).toEqual(new Set(["centre", "calibrate", "uniform"]))
    // draws.json is 05a's, frozen: it is never regenerated. 05b's Draw adds evalSeed and confirmation (null for every
    // 05a method); every other field must equal 05a's bit for bit.
    expect(draws).toEqual(fixture05a().map(as05b))
  })

  it("no posterior → centre, drawSeed null, genome = centre", () => {
    for (const id of ids(50, "none")) {
      for (const w of ["gpu", "cpu"] as const) {
        const d = birthDraw(input(id, null, w))
        expect(d).toEqual({ role: "search", method: "centre", drawSeed: null, posteriorSha: null, evalPanel: evalPanelOf(id), evalSeed: null, confirmation: null, genome: centre })
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

// ---- 05b: the ts and confirm branches (plan 05b, Task 5) ----

const V2_FIXTURE = new URL("./fixtures/posterior-v2-search.json", import.meta.url)
const FIXTURE_05B = new URL("./fixtures/draws-05b.json", import.meta.url)
type Json = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any

/** The v2 search fixture (a real fit on the real space, with a confirmation in flight: id 3, need 7/9/22) as an object. */
const v2File = (): Json => JSON.parse(readFileSync(V2_FIXTURE, "utf8")) as Json

/** The v2 search fixture with `o` over it: the confirmation removed (null) or its need replaced, the model's block edited. */
function v2(o: { mode?: Mode; confirmation?: null; need?: Partial<Record<ArmName, number>>; models?: Json } = {}): Json {
  const f = v2File()
  if (o.mode !== undefined) f.mode = o.mode
  if (o.confirmation === null) f.confirmation = null
  if (o.need !== undefined) f.confirmation.need = { ...f.confirmation.need, ...o.need }
  if (o.models !== undefined) { f.models = o.models; if (!o.models.gpu) f.incumbent = null }
  return f
}

/** A tab's input for a file object: its JSON text parsed by parsePosterior (it must parse), and the text's sha256. */
function inputOfFile(learnerId: string, f: Json, widthsClass: "gpu" | "cpu" = "gpu"): DrawInput {
  const text = JSON.stringify(f)
  const parsed = parsePosterior(text)
  if (parsed === null) throw new Error("test posterior does not parse")
  return { learnerId, space, centre, centreSha, widthsClass, posterior: parsed, posteriorSha: HEX(text) }
}

/** An input whose Posterior bypasses the parser (a hand-built object), with a stand-in sha. */
const handBuilt = (learnerId: string, p: Posterior, widthsClass: "gpu" | "cpu" = "gpu", sha = HEX("hand-built")): DrawInput =>
  ({ learnerId, space, centre, centreSha, widthsClass, posterior: p, posteriorSha: sha })

/** The draw's own rng, as birthDraw seeds it, and its first two values: u0 and the confirmation's j (if it draws one). */
function streamOf(i: DrawInput): { rng: () => number; u0: number } {
  const rng = createRng(drawSeedOf(i.learnerId, i.posteriorSha!))
  return { rng, u0: rng() }
}

const fixedGenes = space.genes.filter(g => g.owner !== "gp").map(g => g.name)
/** A ts draw's genome is a point of the space: checkGenome passes, it decodes, and the fixed genes are the centre's. */
function expectRunnable(g: Genome): void {
  expect(checkGenome(space, g)).toBeNull()
  expect(() => decodeGenome(g, space)).not.toThrow()
  for (const name of fixedGenes) expect(g[name]).toBe(centre[name])
}

const share = (ds: readonly Draw[], m: string) => ds.filter(d => d.method === m).length / ds.length

/** The two 2,000-draw runs (the costly part: ~1,600 + ~600 Thompson picks), computed once. */
let runs: { plain: Draw[]; confirming: Draw[] } | undefined
function searchRuns() {
  if (runs === undefined) {
    const plainFile = v2({ confirmation: null }), confFile = v2()
    const plain = ids(2000, "ts").map(id => birthDraw(inputOfFile(id, plainFile)))
    const confirming = ids(2000, "tc").map(id => birthDraw(inputOfFile(id, confFile)))
    runs = { plain, confirming }
  }
  return runs
}

describe("05b branches", () => {
  it("the fixtures fit the tab: the v2 file's centre is knowledge/genome.json's, and its model searches the space's genes", () => {
    const f = v2File()
    expect(f.centre.genomeSha).toBe(centreSha)
    expect(f.models.gpu.genes).toEqual(searched(space).map(g => g.name))
    expect(f.confirmation.need).toEqual({ candidate: 7, centre: 9, newborn: 22 })
    expect([CONFIRM_SHARE_END, CONFIRM_SEEDS]).toEqual([0.7, 1000])
    expect(ARM_OF_J).toEqual(["candidate", "centre", "newborn", "newborn"])
  })

  it("05a's 20 fixture draws (draws.json) are unchanged under 05b code", () => {
    const fixture = fixture05a()
    expect(fixture).toHaveLength(22)
    for (const d of fixture) {
      expect(Object.hasOwn(d, "evalSeed")).toBe(false)
      expect(Object.hasOwn(d, "confirmation")).toBe(false)
    }
    const draws = cases05a().map(c => birthDraw(c))
    // toEqual compares numbers with Object.is, so every genome value, seed and panel is 05a's bit for bit.
    expect(draws).toEqual(fixture.map(as05b))
  })

  it("a v1 search posterior (models {}): gpu births with u0 ≥ 0.2 draw centre, as in 05a", () => {
    const p = posterior("search")
    let centres = 0
    for (const id of ids(2000, "v1s")) {
      const i = input(id, p)
      const d = birthDraw(i)
      const { u0 } = streamOf(i)
      if (u0 < UNIFORM_SHARE) { expect(d.method).toBe("uniform"); continue }
      centres++
      // Exactly 05a's draw: no reason, no seed, no confirmation.
      expect(d).toEqual({ role: "search", method: "centre", drawSeed: drawSeedOf(id, i.posteriorSha!), posteriorSha: i.posteriorSha, evalPanel: evalPanelOf(id), evalSeed: null, confirmation: null, genome: centre })
    }
    expect(centres).toBeGreaterThan(1500)
  })

  it("search with a model: over 2,000 ids, uniform 0.2 ± 0.03, ts 0.8 ± 0.03; with a confirmation: uniform 0.2, confirm ≈ 0.5, ts ≈ 0.3 (± 0.03)", () => {
    const { plain, confirming } = searchRuns()
    expect(new Set(plain.map(d => d.method))).toEqual(new Set(["uniform", "ts"]))
    expect(Math.abs(share(plain, "uniform") - 0.2)).toBeLessThanOrEqual(0.03)
    expect(Math.abs(share(plain, "ts") - 0.8)).toBeLessThanOrEqual(0.03)
    expect(new Set(confirming.map(d => d.method))).toEqual(new Set(["uniform", "confirm", "ts"]))
    expect(Math.abs(share(confirming, "uniform") - 0.2)).toBeLessThanOrEqual(0.03)
    expect(Math.abs(share(confirming, "confirm") - 0.5)).toBeLessThanOrEqual(0.03)
    expect(Math.abs(share(confirming, "ts") - 0.3)).toBeLessThanOrEqual(0.03)
    for (const d of [...plain, ...confirming]) {
      if (d.method === "confirm") continue
      expect(d.role).toBe("search")
      expect(d.evalSeed).toBeNull()
      expect(d.confirmation).toBeNull()
      expect(d.reason).toBeUndefined()
      expectRunnable(d.genome)
    }
    // The picks spread over the cube: ts genomes are not one point.
    expect(new Set(plain.filter(d => d.method === "ts").map(d => JSON.stringify(d.genome))).size).toBeGreaterThan(100)
  }, 600_000)

  it("confirm draws: evalSeed − seedBase = j, arm = ARM_OF_J[j mod 4]; candidate and newborn run candidate.values, centre runs centre.values; evalPanel null", () => {
    const f = v2File()
    const { confirming } = searchRuns()
    const tc = ids(2000, "tc")
    const file = v2()
    const arms = new Set<ArmName>()
    for (let k = 0; k < confirming.length; k++) {
      const d = confirming[k]!
      if (d.method !== "confirm") continue
      const { rng, u0 } = streamOf(inputOfFile(tc[k]!, file))
      expect(u0 >= UNIFORM_SHARE && u0 < CONFIRM_SHARE_END).toBe(true)
      const j = Math.floor(rng() * CONFIRM_SEEDS)
      const arm = ARM_OF_J[j % 4]!
      arms.add(arm)
      expect(d.role).toBe("confirm")
      expect(d.evalPanel).toBeNull()
      expect(d.evalSeed! - f.confirmation.seedBase).toBe(j)
      expect(d.evalSeed).toBe(13_000 + j)
      expect(d.confirmation).toEqual({ id: 3, arm })
      expect(armOfSeed(d.evalSeed!, 3)).toBe(arm)
      expect(d.genome).toEqual(arm === "centre" ? f.confirmation.centre.values : f.confirmation.candidate.values)
      expect(d.reason).toBeUndefined()
      expect(checkGenome(space, d.genome)).toBeNull()
      expect(() => decodeGenome(d.genome, space)).not.toThrow()
    }
    expect(arms).toEqual(new Set(["candidate", "centre", "newborn"]))
    // The seed's arm, from the table (spec §4.5).
    expect([13_004, 13_005, 13_006, 13_007, 13_999].map(e => armOfSeed(e, 3))).toEqual(["candidate", "centre", "newborn", "newborn", "newborn"])
    expect(armOfSeed(10_000, 0)).toBe("candidate")
  }, 600_000)

  it("an arm with need 0 falls through to ts with the same rng stream (the ts genome equals the one drawn when that arm's need is 0 by construction)", () => {
    const file = v2({ need: { candidate: 0 } })
    const post = modelPosterior(v2File().models.gpu)!
    let fell = 0, confirmed = 0
    for (const id of ids(600, "fall")) {
      const i = inputOfFile(id, file)
      const { rng, u0 } = streamOf(i)
      if (u0 < UNIFORM_SHARE || u0 >= CONFIRM_SHARE_END) continue
      const arm = ARM_OF_J[Math.floor(rng() * CONFIRM_SEEDS) % 4]!
      const d = birthDraw(i)
      if (arm !== "candidate") { expect(d.method).toBe("confirm"); confirmed++; continue }
      // The rng has given u0 and j; the ts pick continues the same stream.
      const pick = thompsonPick(post, candidateSet(post, rng), rng)!
      expect(d).toEqual({
        role: "search", method: "ts", drawSeed: drawSeedOf(id, i.posteriorSha!), posteriorSha: i.posteriorSha, evalPanel: evalPanelOf(id),
        evalSeed: null, confirmation: null, genome: fromUnit(space, pick.x, centre),
      })
      fell++
    }
    expect(fell).toBeGreaterThanOrEqual(20)
    expect(confirmed).toBeGreaterThanOrEqual(60)
  }, 600_000)

  it("mode guard: a hand-built Posterior with mode calibrate/paused/flat and models.gpu + confirmation set → no draw is ts or confirm over 2,000 ids", () => {
    const parsed = parsePosterior(JSON.stringify(v2File()))!
    expect(parsed.models.gpu).toBeDefined()
    expect(parsed.confirmation).not.toBeNull()
    for (const mode of ["calibrate", "paused", "flat"] as const) {
      const guarded: Posterior = { ...parsed, mode }
      const stripped: Posterior = { ...parsed, mode, models: {}, incumbent: null, confirmation: null }
      const seen = new Set<string>()
      for (const id of ids(2000, `g${mode}`)) {
        const d = birthDraw(handBuilt(id, guarded))
        seen.add(d.method)
        // Exactly the draw of the same posterior with nothing of the search in it (05a's draw).
        expect(d).toEqual(birthDraw(handBuilt(id, stripped)))
      }
      expect(seen.has("ts") || seen.has("confirm"), mode).toBe(false)
      expect([...seen].sort(), mode).toEqual(mode === "calibrate" ? ["calibrate"] : ["centre", "uniform"])
    }
  })

  it("cpu in search with a model and a confirmation → only uniform or centre", () => {
    const file = v2()
    const cpuIds = ids(2000, "cpu05b")
    const draws = cpuIds.map(id => birthDraw(inputOfFile(id, file, "cpu")))
    expect(new Set(draws.map(d => d.method))).toEqual(new Set(["uniform", "centre"]))
    expect(Math.abs(share(draws, "uniform") - UNIFORM_SHARE)).toBeLessThanOrEqual(0.03)
    draws.forEach((d, k) => {
      expect(d.role).toBe("search")
      expect(d.evalPanel).toBe(evalPanelOf(cpuIds[k]!))
      expect(d.evalSeed).toBeNull()
      expect(d.confirmation).toBeNull()
      expect(d.reason).toBeUndefined()
      if (d.method === "centre") expect(d.genome).toEqual(centre)
    })
  })

  it("a model whose genes differ from the space's searched genes → centre with reason", () => {
    const gpu = v2File().models.gpu
    const names = searched(space).map(g => g.name)
    const check = (i: DrawInput, why: RegExp) => {
      const d = birthDraw(i)
      const { u0 } = streamOf(i)
      if (u0 < UNIFORM_SHARE) { expect(d.method).toBe("uniform"); return 0 }
      expect(d.role).toBe("search")
      expect(d.method).toBe("centre")
      expect(d.genome).toEqual(centre)
      expect(d.reason).toMatch(why)
      expect(d.drawSeed).toBe(drawSeedOf(i.learnerId, i.posteriorSha!))
      return 1
    }
    const cases: [Json, RegExp][] = [
      [v2({ confirmation: null, models: { gpu: { ...gpu, genes: [...names].reverse() } } }), /model's genes are not the space's searched genes/],
      [v2({ confirmation: null, models: { gpu: { ...gpu, genes: names.map((n, j) => j === 3 ? "notAGene" : n) } } }), /model's genes are not the space's searched genes/],
      [v2({ confirmation: null, models: {} }), /no gpu model/],
    ]
    for (const [file, why] of cases) {
      let n = 0
      for (const id of ids(40, "genes")) n += check(inputOfFile(id, file), why)
      expect(n).toBeGreaterThan(20)
    }
    // A model the parser would refuse (length scales for 11 of 12 genes), hand-built: modelPosterior is null → centre.
    const parsed = parsePosterior(JSON.stringify(v2({ confirmation: null })))!
    const broken: Posterior = { ...parsed, models: { gpu: { ...parsed.models.gpu!, hyper: { ...parsed.models.gpu!.hyper, lengthscales: parsed.models.gpu!.hyper.lengthscales.slice(1) } } } }
    let n = 0
    for (const id of ids(40, "nopost")) n += check(handBuilt(id, broken), /no GP posterior from the model/)
    expect(n).toBeGreaterThan(20)
  })

  it("a confirmation genome outside the space → centre with a reason for that arm; the valid arm still confirms", () => {
    const parsed = parsePosterior(JSON.stringify(v2()))!
    const { lr: _lr, ...noLr } = parsed.confirmation!.candidate.values
    void _lr
    const bad: Posterior = { ...parsed, confirmation: { ...parsed.confirmation!, candidate: { ...parsed.confirmation!.candidate, values: noLr } } }
    const seen = new Set<string>()
    for (const id of ids(300, "badconf")) {
      const i = handBuilt(id, bad)
      const { rng, u0 } = streamOf(i)
      if (u0 < UNIFORM_SHARE || u0 >= CONFIRM_SHARE_END) continue
      const arm = ARM_OF_J[Math.floor(rng() * CONFIRM_SEEDS) % 4]!
      const d = birthDraw(i)
      seen.add(arm)
      if (arm === "centre") { expect(d.method).toBe("confirm"); continue }
      expect(d.method).toBe("centre")
      expect(d.role).toBe("search")
      expect(d.evalPanel).toBe(evalPanelOf(id))
      expect(d.reason).toMatch(new RegExp(`confirmation 3's ${arm} genome is invalid: missing gene lr`))
    }
    expect(seen).toEqual(new Set(["candidate", "centre", "newborn"]))
  })

  it("20 ts/confirm draws equal fixtures/draws-05b.json", () => {
    const plainFile = v2({ confirmation: null }), confFile = v2()
    const cases: DrawInput[] = []
    let ts = 0, tsConf = 0, conf = 0
    for (let k = 0; cases.length < 20 && k < 5000; k++) {
      const id = `lrn_05b_${k.toString(36)}`
      const p = inputOfFile(id, plainFile), c = inputOfFile(id, confFile)
      if (ts < 6 && streamOf(p).u0 >= UNIFORM_SHARE) { cases.push(p); ts++; continue }
      const { u0 } = streamOf(c)
      if (tsConf < 4 && u0 >= CONFIRM_SHARE_END) { cases.push(c); tsConf++; continue }
      if (conf < 10 && u0 >= UNIFORM_SHARE && u0 < CONFIRM_SHARE_END) { cases.push(c); conf++ }
    }
    expect(cases).toHaveLength(20)
    const draws = cases.map(c => birthDraw(c))
    for (let i = 0; i < cases.length; i++) expect(birthDraw(cases[i]!)).toEqual(draws[i])
    expect(draws.filter(d => d.method === "ts")).toHaveLength(10)
    expect(draws.filter(d => d.method === "confirm")).toHaveLength(10)
    expect(new Set(draws.flatMap(d => d.confirmation ? [d.confirmation.arm] : []))).toEqual(new Set(["candidate", "centre", "newborn"]))
    if (process.env.DRAW_WRITE === "1") {
      expect(process.env.CI, "never regenerate the golden draws in CI").toBeUndefined()
      writeFileSync(FIXTURE_05B, JSON.stringify(draws, null, 2) + "\n")
    }
    expect(draws).toEqual(JSON.parse(readFileSync(FIXTURE_05B, "utf8")) as Draw[])
  }, 120_000)
})
