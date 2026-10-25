import { describe, it, expect } from "vitest"
import { createHash } from "node:crypto"
import { DEFAULT_SETTINGS } from "@blanktab/learner"
import { CAREGIVER_DEFAULTS, CONFIG_BUDGET, DEFAULT_MAX_STEPS_PER_SEC, DEFAULT_SLEEP_INTERVAL, parseMeta } from "@blanktab/harness"
import { birthDraw, decodeGenome, genomeSha, legacyGenome, parsePosterior } from "@blanktab/meta"
import { birthFromDraw, drawContextOf, resumedHarnessOf, sessionMetaOf } from "../src/birth"
import { CENTRE, CENTRE_SHA, GENOME_JSON, SPACE, SPACE_JSON, V2_SEARCH_TEXT, posteriorText } from "./posteriors"

const sha = (s: string) => createHash("sha256").update(s).digest("hex")
const knowledge = (posterior: string | null, space: unknown = SPACE_JSON, genome: unknown = GENOME_JSON) =>
  ({ meta: { genomeSpace: space, genome, posteriorText: posterior } })
const ids = (k: number) => Array.from({ length: k }, (_, i) => `lrn_${(0x1000 + i * 7919).toString(16).padStart(16, "0")}`)

describe("drawContextOf", () => {
  it("parses the space and the centre, hashes the centre, and parses the posterior from its bytes (sha of the bytes)", async () => {
    const text = posteriorText("calibrate")
    const ctx = (await drawContextOf(knowledge(text)))!
    expect(ctx.space).toEqual(SPACE)
    expect(ctx.centre).toEqual(CENTRE)
    expect(ctx.centreSha).toBe(CENTRE_SHA)
    expect(ctx.posterior).toEqual(parsePosterior(text))
    expect(ctx.posteriorText).toBe(text)
    expect(ctx.posteriorSha).toBe(sha(text))
  })
  it("drawContextOf: malformed space or genome → null; malformed posterior → posterior null, sha null", async () => {
    expect(await drawContextOf({ meta: null })).toBeNull()
    for (const space of [null, "x", { schemaVersion: 1 }, { schemaVersion: 1, genes: [{ name: "lr" }] }]) expect(await drawContextOf(knowledge(null, space))).toBeNull()
    const outside = { ...(GENOME_JSON as object), values: { ...CENTRE, lr: 1 } }
    for (const genome of [null, 7, { schemaVersion: 1 }, outside]) expect(await drawContextOf(knowledge(null, SPACE_JSON, genome))).toBeNull()
    // An unknown schemaVersion, a wrong shape, not JSON, a BOM: no posterior (the draw is centre).
    const v2 = posteriorText("calibrate").replace('"schemaVersion": 1', '"schemaVersion": 2')
    for (const text of [null, "", "<html>", "{}", v2, "﻿" + posteriorText("calibrate")]) {
      const ctx = (await drawContextOf(knowledge(text)))!
      expect(ctx.centreSha, String(text)).toBe(CENTRE_SHA)
      expect([ctx.posterior, ctx.posteriorText, ctx.posteriorSha], String(text)).toEqual([null, null, null])
    }
  })
})

describe("birthFromDraw", () => {
  it("birthFromDraw at the committed knowledge with no posterior → method centre, DEFAULT_SETTINGS values, CAREGIVER_DEFAULTS", async () => {
    const ctx = (await drawContextOf(knowledge(null)))!
    for (const w of ["gpu", "cpu"] as const) {
      const b = await birthFromDraw(ctx, "lrn_0123456789abcdef", w)
      expect(b.learnerSettings).toEqual({ ...DEFAULT_SETTINGS })
      expect(b.caregiver).toEqual({ ...CAREGIVER_DEFAULTS })
      expect(b.maxStepsPerSec).toBe(DEFAULT_MAX_STEPS_PER_SEC)
      expect(b.sleepInterval).toBe(DEFAULT_SLEEP_INTERVAL)
      expect(b.meta).toMatchObject({ v: 1, role: "search", method: "centre", genomeSchema: 1, genome: CENTRE, genomeSha: CENTRE_SHA, drawSeed: null, posteriorSha: null, evalSeed: null, confirmation: null, inheritance: { filters: "none" } })
      expect(b.meta.budget).toEqual({ budgetCU: CONFIG_BUDGET.budgetCU, levels: [...CONFIG_BUDGET.levels] })
      expect(parseMeta(b.meta, { cu: 0, stepsTotal: 0, final: false })).toEqual(b.meta)
    }
  })
  it("birthFromDraw equals birthDraw + decodeGenome for 50 ids (the worker computes what the gate recomputes)", async () => {
    for (const mode of ["calibrate", "paused", "flat", "search"] as const) {
      const text = posteriorText(mode)
      const ctx = (await drawContextOf(knowledge(text)))!
      const methods = new Set<string>()
      for (const id of ids(50)) {
        for (const widthsClass of ["gpu", "cpu"] as const) {
          const b = await birthFromDraw(ctx, id, widthsClass)
          const d = birthDraw({ learnerId: id, space: SPACE, centre: CENTRE, centreSha: CENTRE_SHA, widthsClass, posterior: parsePosterior(text), posteriorSha: sha(text) })
          const dec = decodeGenome(d.genome, SPACE)
          expect(b.meta).toMatchObject({ role: d.role, method: d.method, drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, genome: d.genome })
          expect(b.meta.genomeSha).toBe(await genomeSha(SPACE.schemaVersion, d.genome))
          expect(b.learnerSettings).toEqual({ ...DEFAULT_SETTINGS, ...dec.learner })
          expect(b.caregiver).toEqual({ ...CAREGIVER_DEFAULTS, ...dec.caregiver })
          expect([b.maxStepsPerSec, b.sleepInterval]).toEqual([dec.harness.maxStepsPerSec, dec.harness.sleepInterval])
          methods.add(b.meta.method)
        }
      }
      expect(methods.has(mode === "calibrate" ? "calibrate" : "uniform"), mode).toBe(true)
    }
  })
  it("birthFromDraw of a confirm draw (candidate arm) → meta role confirm, evalSeed, confirmation, evalPanel null, seed = evalSeed; parseMeta accepts it", async () => {
    const ctx = (await drawContextOf(knowledge(V2_SEARCH_TEXT)))!
    const posterior = parsePosterior(V2_SEARCH_TEXT)!
    const drawOf = (id: string) => birthDraw({ learnerId: id, space: SPACE, centre: CENTRE, centreSha: CENTRE_SHA, widthsClass: "gpu", posterior, posteriorSha: sha(V2_SEARCH_TEXT) })
    const pool = ids(200)
    for (const arm of ["candidate", "centre", "newborn"] as const) {
      const id = pool.find(i => drawOf(i).confirmation?.arm === arm)!
      const d = drawOf(id)
      const b = await birthFromDraw(ctx, id, "gpu")
      expect(d.evalSeed).not.toBeNull()
      expect(b.meta).toMatchObject({ role: "confirm", method: "confirm", evalSeed: d.evalSeed, confirmation: { id: 3, arm }, evalPanel: null, drawSeed: d.drawSeed, posteriorSha: sha(V2_SEARCH_TEXT), genome: d.genome })
      expect(b.seed).toBe(d.evalSeed)
      expect(b.kind).toBe(arm === "newborn" ? "confirmNewborn" : "life")
      expect(b.meta.genome).toEqual(arm === "centre" ? posterior.confirmation!.centre.values : posterior.confirmation!.candidate.values)
      // The newborn arm's block is a frozen file's (cu 0, final); the others' are a life's first file.
      const at = arm === "newborn" ? { cu: 0, stepsTotal: 0, final: true } : { cu: 0, stepsTotal: 0, final: false }
      expect(parseMeta(b.meta, at)).toEqual(b.meta)
    }
    // Every other birth has the learnerId's seed (no override), and a ts or uniform one its panel.
    const other = await birthFromDraw(ctx, pool.find(i => drawOf(i).method === "ts")!, "gpu")
    expect(other).toMatchObject({ kind: "life", meta: { method: "ts", evalSeed: null, confirmation: null } })
    expect(other.seed).toBeUndefined()
    expect(typeof other.meta.evalPanel).toBe("number")
  })
  it("birthFromDraw equals birthDraw + decodeGenome for 50 ids under the v2 search fixture (ts and confirm included)", async () => {
    const ctx = (await drawContextOf(knowledge(V2_SEARCH_TEXT)))!
    const posterior = parsePosterior(V2_SEARCH_TEXT)!
    const methods = new Set<string>()
    for (const id of ids(50)) {
      for (const widthsClass of ["gpu", "cpu"] as const) {
        const b = await birthFromDraw(ctx, id, widthsClass)
        const d = birthDraw({ learnerId: id, space: SPACE, centre: CENTRE, centreSha: CENTRE_SHA, widthsClass, posterior, posteriorSha: sha(V2_SEARCH_TEXT) })
        const dec = decodeGenome(d.genome, SPACE)
        expect(b.meta).toMatchObject({ role: d.role, method: d.method, drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, evalSeed: d.evalSeed, confirmation: d.confirmation, genome: d.genome })
        expect(b.meta.genomeSha).toBe(await genomeSha(SPACE.schemaVersion, d.genome))
        expect(b.learnerSettings).toEqual({ ...DEFAULT_SETTINGS, ...dec.learner })
        expect(b.caregiver).toEqual({ ...CAREGIVER_DEFAULTS, ...dec.caregiver })
        expect([b.maxStepsPerSec, b.sleepInterval]).toEqual([dec.harness.maxStepsPerSec, dec.harness.sleepInterval])
        methods.add(`${widthsClass}:${b.meta.method}`)
      }
    }
    expect([...methods].sort()).toEqual(["cpu:centre", "cpu:uniform", "gpu:confirm", "gpu:ts", "gpu:uniform"])
  })
  it("a posterior built for another centre: centre, with the reason; drawSeed and posteriorSha still recorded", async () => {
    const text = posteriorText("calibrate").replace(CENTRE_SHA, "f".repeat(64))
    const b = await birthFromDraw((await drawContextOf(knowledge(text)))!, "lrn_0123456789abcdef", "gpu")
    expect(b.meta).toMatchObject({ method: "centre", posteriorSha: sha(text), genome: CENTRE })
    expect(typeof b.meta.drawSeed).toBe("number")
    expect(b.reason).toBe("posterior built for another centre")
  })
})

describe("sessionMetaOf", () => {
  it("carries the space, the birth block and the config's budget; legacy() is legacyGenome's block with its missing genes", async () => {
    const ctx = (await drawContextOf(knowledge(null)))!
    const b = await birthFromDraw(ctx, "lrn_0123456789abcdef", "gpu")
    const m = sessionMetaOf(ctx, b.meta)
    expect(m.space).toBe(ctx.space)
    expect(m.birth).toBe(b.meta)
    expect(m.budget).toEqual(CONFIG_BUDGET)
    expect(m.resumedHarness!(b.meta)).toEqual({ maxStepsPerSec: DEFAULT_MAX_STEPS_PER_SEC, sleepInterval: DEFAULT_SLEEP_INTERVAL })
    const flat = { lr: 1e-3, entropyWeight: 0.002, curiosityWeight: 1, bufferSize: 500, replayPerSleep: 16, headLrScale: 0.1, exploreEpsilon: 0.1, residualDecay: 1e-3, maxStepsPerSec: 30, seed: 5 }
    const block = await m.legacy(flat)
    const { genome, missing } = legacyGenome(flat, SPACE)
    expect(block).toEqual({
      v: 1, role: "legacy", method: "legacy", genomeSchema: 1, genome, genomeSha: await genomeSha(1, genome), drawSeed: null, posteriorSha: null,
      evalPanel: null, evalSeed: null, confirmation: null, inheritance: { filters: "none" }, missing,
    })
    expect(missing.length).toBeGreaterThan(0) // its caregiver and sleepInterval were not recorded
    expect(parseMeta(block, { cu: 1, stepsTotal: 10, final: false })).toEqual(block)
    // Every gene recorded: no `missing` key at all.
    const all = await m.legacy({ ...flat, ...CAREGIVER_DEFAULTS, sleepInterval: 250 })
    expect(all).not.toHaveProperty("missing")
  })
})

describe("resumedHarnessOf (fix round 1)", () => {
  it("a resumed block's genome gives the cap and sleep interval; a legacy block's unrecorded genes are left to the config; knobs win", async () => {
    const ctx = (await drawContextOf(knowledge(null)))!
    // A space where both harness genes are searched: a resume must run the block's values, not today's centre.
    const space = JSON.parse(JSON.stringify(SPACE_JSON)) as { genes: Record<string, unknown>[] }
    for (const g of space.genes) {
      if (g.name === "maxStepsPerSec") Object.assign(g, { min: 10, max: 60, owner: "gp" })
      if (g.name === "sleepInterval") Object.assign(g, { min: 100, max: 500, owner: "gp", scale: "log" })
    }
    const wide = { ...ctx, ...(await drawContextOf(knowledge(null, space)))! }
    const block = { ...(await birthFromDraw(ctx, "lrn_0123456789abcdef", "gpu")).meta, genome: { ...CENTRE, maxStepsPerSec: 45, sleepInterval: 400 } }
    expect(resumedHarnessOf(wide)(block)).toEqual({ maxStepsPerSec: 45, sleepInterval: 400 })
    expect(resumedHarnessOf(wide, { sleepInterval: 5 })(block)).toEqual({ maxStepsPerSec: 45, sleepInterval: 5 })
    expect(resumedHarnessOf(wide)({ ...block, role: "legacy", method: "legacy", missing: ["sleepInterval"] })).toEqual({ maxStepsPerSec: 45 })
    expect(resumedHarnessOf(wide, { maxStepsPerSec: 9 })({ ...block, genomeSchema: 2 })).toEqual({ maxStepsPerSec: 9 })
  })
})
