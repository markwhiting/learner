import { describe, it, expect } from "vitest"
import type { MetaBlock, SessionRow } from "@blanktab/harness"
import { fromUnit, parsePosterior, searched, toUnit, type Posterior } from "@blanktab/meta"
import { GENE_LABELS, NO_GENOME_TEXT, currentRecipeText, genomeHighlights, metaLineOf, metaStateText, missingText, recipeText, scoreText } from "../src/recipe"
import { CALIB, CENTRE, GENOME_JSON, SPACE, SPACE_JSON, V2_SEARCH_TEXT, posteriorText, v2SearchText } from "./posteriors"

const P = (mode: Posterior["mode"], page: Parameters<typeof posteriorText>[1] = {}) => parsePosterior(posteriorText(mode, page))!
const meta = (o: Partial<MetaBlock>): MetaBlock => ({
  v: 1, role: "search", method: "centre", genomeSchema: 1, genome: { ...CENTRE }, genomeSha: "0".repeat(64), drawSeed: null, posteriorSha: null,
  evalPanel: 3, evalSeed: null, confirmation: null, inheritance: { filters: "none" }, ...o,
})

describe("recipeText", () => {
  it("says how the life's genome was drawn, in words", async () => {
    const p = P("calibrate")
    const fourth = p.calibration.genomes[3]!
    expect(recipeText(meta({ method: "calibrate", genome: fourth.values, genomeSha: fourth.genomeSha }), p)).toBe("a calibration life (recipe 4 of 10)")
    expect(recipeText(meta({ method: "calibrate" }), null)).toBe("a calibration life")
    expect(recipeText(meta({ method: "uniform" }), p)).toBe("one of the 1-in-5 random explorers")
    expect(recipeText(meta({ method: "centre" }), p)).toBe("the current recipe")
    expect(recipeText(meta({ method: "centre" }), null)).toBe("the current recipe")
    expect(recipeText(meta({ role: "legacy", method: "legacy", evalPanel: null }), p)).toBe("born before the meta-learner (its old settings)")
    expect(recipeText(meta({ role: "newborn", method: "newborn", evalPanel: 7 }), null)).toBe("a newborn reference (panel 7)")
  })
  it("a ts life: drawn by the meta-learner, with the number of lives in the model it was drawn from", () => {
    const p = parsePosterior(V2_SEARCH_TEXT)!
    expect(p.models.gpu!.rows).toHaveLength(40)
    expect(recipeText(meta({ method: "ts" }), p)).toBe("drawn by the meta-learner (a Thompson sample from 40 lives)")
    const many = parsePosterior(v2SearchText(f => {
      const r = f.models.gpu.rows, y = f.models.gpu.y
      while (r.length < 214) { r.push({ ...r[0], learnerId: `lrn_${(0xf000000 + r.length).toString(16).padStart(16, "0")}` }); y.push(y[0]) }
    }))!
    expect(recipeText(meta({ method: "ts" }), many)).toBe("drawn by the meta-learner (a Thompson sample from 214 lives)")
    // Without the posterior (or one with no model), the words alone.
    expect(recipeText(meta({ method: "ts" }), null)).toBe("drawn by the meta-learner")
    expect(recipeText(meta({ method: "ts" }), P("search"))).toBe("drawn by the meta-learner")
  })
  it("a confirmation life: its arm and its fresh world (its evalSeed)", () => {
    const c = (arm: string, evalSeed: number) => meta({ role: "confirm", method: "confirm", evalPanel: null, evalSeed, confirmation: { id: 0, arm } })
    expect(recipeText(c("candidate", 10_347), null)).toBe("a confirmation life (candidate arm, fresh world 10,347)")
    expect(recipeText(c("centre", 10_345), null)).toBe("a confirmation life (current-recipe arm, fresh world 10,345)")
    expect(recipeText(c("newborn", 10_346), null)).toBe("a confirmation life (newborn arm, fresh world 10,346)")
    expect(recipeText(meta({ role: "confirm", method: "confirm", evalPanel: null }), null)).toBe("a confirmation life")
  })
})

describe("metaStateText", () => {
  it("one line for the meta-learner's state, from the posterior's page counts", () => {
    expect(metaStateText(P("calibrate", { calibrationLives: 18, calibrationTarget: 30 })))
      .toBe("Calibrating: 18 of 30 lives measured, to learn whether the score can tell recipes apart.")
    expect(metaStateText(P("paused", { calibrationLives: 34, calibrationTarget: 45, looks: 1 })))
      .toBe("Paused: the score can't yet tell recipes apart; measuring more (34 of 45).")
    expect(metaStateText(P("flat", { calibrationLives: 61, calibrationTarget: 60, looks: 3 })))
      .toBe("Flat: after 60 lives the score could not tell recipes apart. Recipes are the current one, with 1 in 5 random.")
    expect(metaStateText(null)).toBe("Not reachable: this learner runs the current recipe.")
  })
  it("mode search: lives so far and the model's best guess, or the confirmation's progress; the last promotion's date", () => {
    const S = (edit: NonNullable<Parameters<typeof v2SearchText>[0]>) => {
      const p = parsePosterior(v2SearchText(f => { f.page.eligibleLives = 214; edit(f) }))
      if (p === null) throw new Error("test posterior does not parse")
      return p
    }
    expect(metaStateText(S(f => { f.models = {}; f.incumbent = null; f.confirmation = null }))).toBe("Searching: 214 lives, not yet enough for the model.")
    expect(metaStateText(S(f => { f.confirmation = null; f.incumbent.pBeatsCentre = 0.72 })))
      .toBe("Searching: 214 lives; its best guess is predicted to beat the current recipe with 72 % confidence.")
    // A model with no incumbent (no posterior from it): agent's call.
    expect(metaStateText(S(f => { f.confirmation = null; f.incumbent = null }))).toBe("Searching: 214 lives; the model has no best guess yet.")
    const confirming = S(f => {
      f.confirmation.candidate.genomeSha = "3f2a1c9e" + f.confirmation.candidate.genomeSha.slice(8)
      f.confirmation.counted = { candidate: 10, centre: 9, newborn: 12 }
    })
    expect(confirming.confirmation!.target).toEqual({ candidate: 24, centre: 24, newborn: 60 })
    expect(metaStateText(confirming)).toBe("Confirming recipe 3f2a1c9e on fresh worlds: 31 of 108 lives.")
    expect(metaStateText(S(f => { f.confirmation = null; f.incumbent.pBeatsCentre = 0.72; f.page.lastPromotion = "2026-11-03T09:00:00.000Z" })))
      .toBe("Searching: 214 lives; its best guess is predicted to beat the current recipe with 72 % confidence. Last promotion: 2026-11-03.")
    // A 05a (v1) search posterior read by 05b code: no model, and its lives as it counts them.
    expect(metaStateText(P("search"))).toBe("Searching: 0 lives, not yet enough for the model.")
  })
})

describe("currentRecipeText", () => {
  it("the centre in force, from knowledge/genome.json's provenance: the initial one, or its promotion's date and confirmation", () => {
    expect(currentRecipeText(GENOME_JSON)).toBe("Current recipe: the initial one")
    const promoted = { ...(GENOME_JSON as object), provenance: { source: "promotion", confirmation: 0, decision: "x", promotedAt: "2026-11-03T09:00:00.000Z", fisher: null, mannWhitneyP: null, lives: {} } }
    expect(currentRecipeText(promoted)).toBe("Current recipe: promoted 2026-11-03 (confirmation 0)")
    for (const bad of [null, "x", {}, { provenance: null }, { provenance: { source: "promotion", confirmation: "0", promotedAt: "2026-11-03T09:00:00.000Z" } }, { provenance: { source: "promotion", confirmation: 0 } }]) {
      expect(currentRecipeText(bad), JSON.stringify(bad)).toBeNull()
    }
  })
})

describe("metaLineOf", () => {
  it("the posterior's state with readable genome files; otherwise it says the learner runs the defaults, not counted", () => {
    const k = (posterior: string | null, space: unknown = SPACE_JSON) => ({ meta: { genomeSpace: space, genome: GENOME_JSON, posteriorText: posterior } })
    expect(metaLineOf(k(posteriorText("calibrate", { calibrationLives: 2 })))).toBe("Calibrating: 2 of 30 lives measured, to learn whether the score can tell recipes apart.")
    expect(metaLineOf(k(null))).toBe("Not reachable: this learner runs the current recipe.")
    expect(metaLineOf(k("garbage"))).toBe("Not reachable: this learner runs the current recipe.")
    expect(metaLineOf(k(null, "garbage"))).toBe(NO_GENOME_TEXT)
    expect(metaLineOf({ meta: null })).toBe(NO_GENOME_TEXT)
  })
})

describe("genomeHighlights", () => {
  it("the centre has none; GENE_LABELS name the searched genes", () => {
    expect(genomeHighlights(CENTRE, CENTRE, SPACE)).toEqual([])
    expect(GENE_LABELS.lr).toBe("learning rate")
    expect(GENE_LABELS.curiosityWeight).toBe("curiosity")
    for (const g of searched(SPACE)) expect(GENE_LABELS[g.name], g.name).toBeTruthy()
  })
  it("ranks by the unit-cube distance from the centre and shows ratios v/centre (the horizon ratio for gamma)", () => {
    const names = searched(SPACE).map(g => g.name)
    const u = toUnit(SPACE, CENTRE)
    // lr moved furthest, then gamma, then curiosity; bufferSize a little: it is not among the top 3.
    const at = (name: string, x: number) => { u[names.indexOf(name)] = x }
    at("lr", 0.95); at("gamma", 0.1); at("curiosityWeight", 0.25); at("bufferSize", 0.55)
    const g = fromUnit(SPACE, u, CENTRE)
    const ratio = (x: number) => String(Number(x.toPrecision(2)))
    const horizon = (v: number) => 1 / (1 - v)
    expect(genomeHighlights(g, CENTRE, SPACE)).toEqual([
      `learning rate ×${ratio((g.lr as number) / (CENTRE.lr as number))}`,
      `planning horizon ×${ratio(horizon(g.gamma as number) / horizon(CENTRE.gamma as number))}`,
      `curiosity ×${ratio((g.curiosityWeight as number) / (CENTRE.curiosityWeight as number))}`,
    ])
    expect(genomeHighlights(g, CENTRE, SPACE, 1)).toHaveLength(1)
    expect(genomeHighlights(g, CENTRE, SPACE, 10)).toHaveLength(4)
  })
  it("a calibration genome's three furthest genes", () => {
    expect(genomeHighlights(CALIB[5]!, CENTRE, SPACE)).toHaveLength(3)
  })
})

describe("missingText and scoreText", () => {
  it("a legacy life's unrecorded genes, named", () => {
    expect(missingText([])).toBeNull()
    expect(missingText(["caregiverPeriod", "sleepInterval"])).toBe("some settings not recorded: caregiver period, sleepInterval")
  })
  it("the life's own score once its summary row has the evaluator's entry", () => {
    const row = (evals?: SessionRow["evals"]) => ({ learnerId: "lrn_0123456789abcdef", ...(evals ? { evals } : {}) }) as SessionRow
    expect(scoreText(undefined, "search-composite@1")).toBeNull()
    expect(scoreText(row(), "search-composite@1")).toBeNull()
    expect(scoreText(row({ "search-composite@1": { eligible: true, score: 0.4123, C: 0.5, H: 0.3 } }), "search-composite@1")).toBe("0.41 (search-composite@1)")
    expect(scoreText(row({ "search-composite@1": { eligible: false, score: null, C: null, H: null } }), "search-composite@1")).toBe("not eligible (search-composite@1)")
    expect(scoreText(row({ "other@1": { eligible: true, score: 0.5, C: 0.5, H: 0.5 } }), "search-composite@1")).toBeNull()
  })
})
