import { describe, it, expect } from "vitest"
import evaluationConfig from "../../../config/evaluation.json" with { type: "json" }
import { CONFIRM_ARM_OF_J, confirmArmOfSeed, BUDGET_CU, evalBudgetOf, lifeBudget, V1_BUDGET, ERROR_REASON_RE, FILTER_LEVELS, MAX_CHECKUPS_PER_FILE, MAX_EVAL_PANEL, MAX_GENOME_SCHEMA, NEWBORN_LEVELS, parseMeta, MAX_SUITE_SIZE, MILESTONE_IDS, MILESTONE_SUITE_VERSION, isLogLevel, levelAtOrBelow, logThresholds, nextCrossing, parseLogFile, parseSessionRow, MAX_SESSION_EVALS, type CheckupResult } from "../src/log"
import { mkFile } from "./helpers"
import { PANELS } from "@blanktab/meta"

const ID = "lrn_0123456789abcdef"
const IDS = MILESTONE_IDS[MILESTONE_SUITE_VERSION]!
const measures = (over: Record<string, number | null> = {}) => ({ ...Object.fromEntries(IDS.map(id => [id, 0.5])), ...over })
const checkup = (level: number, over: Partial<CheckupResult> = {}): CheckupResult =>
  ({ level, steps: level, passed: ["M1", "M3"], measures: measures({ M2: null }), evalCU: 0.01, ...over })
const v2 = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  ...mkFile(ID, 0), schemaVersion: 2, stepsTotal: 100_000, cu: 300, condition: "A",
  widths: { conv1: 16, conv2: 32, enc: 128, hidden: 256 }, curve: [], skippedUpdates: 0, restores: 0, lostCU: 0, evalCU: 0.5,
  milestoneSuiteVersion: 1, checkups: [checkup(64), checkup(128, { error: true })],
  caregiver: { carries: 5, to: { left: 2, right: 3 }, safetyNet: false }, ...over,
})
const ok = (o: unknown) => parseLogFile(o).ok

describe("log schedule helpers", () => {
  it("isLogLevel and levelAtOrBelow agree with logThresholds", () => {
    const levels = new Set<number>()
    for (const t of logThresholds()) { if (t > 20_000) break; levels.add(t) }
    for (let n = 0; n <= 5000; n++) expect(isLogLevel(n), String(n)).toBe(levels.has(n))
    expect(isLogLevel(12)).toBe(false)
    expect(isLogLevel(0.5)).toBe(false)
    const at = (cu: number) => { let best: number | null = null; for (const t of logThresholds()) { if (t > cu) break; best = t } return best }
    for (const cu of [0, 7.99, 8, 15.9, 16, 511, 512, 1000, 1023.9, 1024, 1500, 2048, 5000.5]) expect(levelAtOrBelow(cu), String(cu)).toBe(at(cu))
    expect(nextCrossing(8)).toBe(16)
  })
})

describe("parseLogFile: checkups, milestoneSuiteVersion, caregiver", () => {
  it("a v2 file with checkups, milestoneSuiteVersion and caregiver parses and round-trips; each object has strict keys", () => {
    const f = v2()
    const r = parseLogFile(f)
    expect(r.ok).toBe(true)
    if (r.ok) expect(JSON.parse(JSON.stringify(r.value))).toEqual(f)
    const withExtra = (mut: (o: any) => void) => { const o = JSON.parse(JSON.stringify(f)); mut(o); return o } // eslint-disable-line @typescript-eslint/no-explicit-any
    expect(ok(withExtra(o => { o.checkups[0].extra = 1 }))).toBe(false)
    expect(ok(withExtra(o => { o.caregiver.extra = 1 }))).toBe(false)
  })
  it("rejects bad checkups", () => {
    const bad = (checkups: unknown, over: Record<string, unknown> = {}) => parseLogFile(v2({ checkups, ...over }))
    const cases: [string, ReturnType<typeof bad>][] = [
      ["level off the schedule", bad([checkup(12)])],
      ["level above cu", bad([checkup(512)])],
      ["levels equal", bad([checkup(64), checkup(64)])],
      ["levels decreasing", bad([checkup(128), checkup(64)])],
      ["steps > stepsTotal", bad([checkup(64, { steps: 100_001 })])],
      ["unknown id", bad([checkup(64, { passed: ["M9"] })])],
      ["duplicate id", bad([checkup(64, { passed: ["M1", "M1"] })])],
      ["out of suite order", bad([checkup(64, { passed: ["M3", "M1"] })])],
      ["measures key outside suite", bad([checkup(64, { measures: measures({ M7: 1 }) })])],
      ["measures key missing", bad([checkup(64, { measures: { M1: 1 } })])],
      ["non-finite measure", bad([checkup(64, { measures: measures({ M1: NaN }) })])],
      ["string measure", bad([checkup(64, { measures: measures({ M1: "1" as unknown as number }) })])],
      ["error: false", bad([{ ...checkup(64), error: false }])],
      ["negative evalCU", bad([checkup(64, { evalCU: -1 })])],
      ["9 checkups", bad(Array.from({ length: MAX_CHECKUPS_PER_FILE + 1 }, (_, i) => checkup(8 * 2 ** (i % 6)))), ],
      ["not an array", bad({})],
      ["no milestoneSuiteVersion", parseLogFile((() => { const o = v2(); delete o.milestoneSuiteVersion; return o })())],
      ["unknown suite version 2", bad([checkup(64)], { milestoneSuiteVersion: 2 })],
    ]
    for (const [name, r] of cases) expect(r.ok, name).toBe(false)
    // 9 genuinely-increasing levels on a big file fails on count alone
    const many = [8, 16, 32, 64, 128, 256, 512, 1024, 2048].map(l => checkup(l))
    const r = bad(many, { cu: 5000 })
    expect(r.ok).toBe(false)
    expect(bad(many.slice(0, 8), { cu: 5000 }).ok).toBe(true)
  })
  it("errorReasons: accepted only with error true, ids in the suite with null measures, ERROR_REASON_RE values; an 81-char value, a newline, an unknown id, a non-null measure and {} are rejected; old error-true results without it parse", () => {
    const withReasons = (reasons: unknown, over: Partial<CheckupResult> = {}) =>
      parseLogFile(v2({ checkups: [{ ...checkup(64, { error: true, measures: measures({ M2: null, M4: null }) }), errorReasons: reasons, ...over }] }))
    const good = withReasons({ M2: "RangeError: bad", M4: "x".repeat(80) })
    expect(good.ok).toBe(true)
    if (good.ok && good.value.schemaVersion === 2) expect(good.value.checkups![0]!.errorReasons).toEqual({ M2: "RangeError: bad", M4: "x".repeat(80) })
    expect(ERROR_REASON_RE.test("x".repeat(80))).toBe(true)
    const cases: [string, ReturnType<typeof withReasons>][] = [
      ["81 chars", withReasons({ M2: "x".repeat(81) })],
      ["empty value", withReasons({ M2: "" })],
      ["newline", withReasons({ M2: "a\nb" })],
      ["non-ASCII", withReasons({ M2: "caf\u00e9" })],
      ["non-string", withReasons({ M2: 3 })],
      ["unknown id", withReasons({ M9: "x" })],
      ["non-null measure", withReasons({ M1: "x" })],
      ["empty object", withReasons({})],
      ["not an object", withReasons("x")],
      ["null", withReasons(null)],
      ["array", withReasons(["x"])],
      ["without error true", parseLogFile(v2({ checkups: [{ ...checkup(64, { measures: measures({ M2: null }) }), errorReasons: { M2: "x" } }] }))],
    ]
    for (const [name, r] of cases) expect(r.ok, name).toBe(false)
    expect(ok(v2({ checkups: [checkup(64, { error: true })] }))).toBe(true)
  })
  it("rejects a bad caregiver summary", () => {
    const cg = (c: unknown) => ok(v2({ caregiver: c }))
    expect(cg({ carries: 2, to: { a: 2 }, safetyNet: true })).toBe(true)
    expect(cg({ carries: 0, to: {}, safetyNet: false })).toBe(true)
    expect(cg({ carries: 1.5, to: { a: 1.5 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: -1, to: { a: -1 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: 1, to: { A: 1 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: 1, to: { abcdefghijklmnopq: 1 }, safetyNet: false })).toBe(false)
    const many = Object.fromEntries("abcdefghijklmnopq".split("").map(k => [k, 1]))
    expect(cg({ carries: 17, to: many, safetyNet: false })).toBe(false)
    expect(cg({ carries: 3, to: { a: 1, b: 1 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: 1, to: { a: 1 }, safetyNet: "no" })).toBe(false)
    expect(cg(null)).toBe(false)
  })
  it("files without the new fields (old v2, v1) still parse", () => {
    const o = v2(); delete o.checkups; delete o.milestoneSuiteVersion; delete o.caregiver; delete o.evalCU
    const r = parseLogFile(o)
    expect(r.ok).toBe(true)
    if (r.ok) expect(Object.keys(r.value)).not.toContain("checkups")
    expect(ok(mkFile(ID, 0))).toBe(true)
    expect(ok({ ...mkFile(ID, 0), checkups: [] })).toBe(false) // v1 has no such field
  })
  it("checkups are billed inside the file's evalCU: non-empty checkups need evalCU, and their sum fits in it (1e-9 relative)", () => {
    const noEval = v2(); delete noEval.evalCU
    expect(parseLogFile(noEval)).toEqual({ ok: false, error: "checkups need the file's evalCU" })
    const empty = v2({ checkups: [] }); delete empty.evalCU
    expect(ok(empty)).toBe(true) // no checkups: nothing to bill
    expect(ok(v2({ evalCU: 0.02 }))).toBe(true) // exactly the sum (0.01 + 0.01)
    expect(ok(v2({ evalCU: 0.019 }))).toBe(false)
    expect(ok(v2({ evalCU: 0, checkups: [checkup(64, { evalCU: 0 })] }))).toBe(true)
    // Float sums: 0.1 + 0.2 > 0.3 by one ulp, within the tolerance; a real excess of 1e-6 relative is not.
    const tenths = [checkup(64, { evalCU: 0.1 }), checkup(128, { evalCU: 0.2 })]
    expect(ok(v2({ evalCU: 0.3, checkups: tenths }))).toBe(true)
    expect(ok(v2({ evalCU: 0.3 * (1 - 1e-6), checkups: tenths }))).toBe(false)
  })
})

describe("parseSessionRow checkupScore", () => {
  const row = { learnerId: ID, settings: {}, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, stepsTotal: 1, sessionsSeenAtBirth: 0, final: false }
  it("keeps a good checkupScore and drops a malformed one to null", () => {
    expect(parseSessionRow({ ...row, checkupScore: { level: 64, score: 3 } })!.checkupScore).toEqual({ level: 64, score: 3 })
    for (const bad of [{ level: "64", score: 3 }, { level: 64 }, { level: -1, score: 1 }, { level: 64, score: 1.5 }, 3, [], "x"]) {
      expect(parseSessionRow({ ...row, checkupScore: bad })!.checkupScore, JSON.stringify(bad)).toBeNull()
    }
    expect(parseSessionRow(row)!.checkupScore).toBeNull()
  })
  it("parseSessionRow: checkupScore off the log schedule (level 12), score 7, a negative or fractional score → null; level 512 score 6 kept", () => {
    expect(MAX_SUITE_SIZE).toBe(6)
    for (const bad of [{ level: 12, score: 1 }, { level: 64, score: 7 }, { level: 64, score: -1 }, { level: 64, score: 0.5 }, { level: 0, score: 0 }]) {
      expect(parseSessionRow({ ...row, checkupScore: bad })!.checkupScore, JSON.stringify(bad)).toBeNull()
    }
    expect(parseSessionRow({ ...row, checkupScore: { level: 512, score: 6 } })!.checkupScore).toEqual({ level: 512, score: 6 })
    expect(parseSessionRow({ ...row, checkupScore: { level: 2048, score: 0 } })!.checkupScore).toEqual({ level: 2048, score: 0 })
  })
})

describe("parseSessionRow evals (Plan 05a Task 9)", () => {
  const row = { learnerId: ID, settings: {}, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, stepsTotal: 1, sessionsSeenAtBirth: 0, final: false }
  it("parseSessionRow drops a malformed evals entry, not the row", () => {
    const good = { eligible: true, score: 0.5, C: 0.2, H: 0.3 }
    const ineligible = { eligible: false, score: null, C: null, H: null }
    const r = parseSessionRow({
      ...row,
      evals: {
        "search-composite@1": good, "search-composite@2": ineligible,
        "no-version": good, "Bad@1": good, "x@0": good, // keys that are not id@version
        "a@1": { ...good, score: "0.5" }, "b@1": { ...good, eligible: "yes" }, "c@1": { ...good, C: Number.NaN }, "d@1": 3,
        "e@1": { ...good, score: null }, "f@1": { ...ineligible, score: 1 }, // score present exactly when eligible
        "g@1": { eligible: true, score: 1, C: 1 }, // H missing
      },
    })
    expect(r).not.toBeNull()
    expect(r!.evals).toEqual({ "search-composite@1": good, "search-composite@2": ineligible })
    // Extra keys inside an entry are not kept.
    expect(parseSessionRow({ ...row, evals: { "s@1": { ...good, extra: 1 } } })!.evals).toEqual({ "s@1": good })
    // Not an object, or nothing left: no evals at all (an old summary's row reads the same).
    for (const bad of [null, 3, "x", [], {}, { "a@1": 1 }]) expect(parseSessionRow({ ...row, evals: bad })!.evals, JSON.stringify(bad)).toBeUndefined()
    expect(parseSessionRow(row)!.evals).toBeUndefined()
    // At most MAX_SESSION_EVALS entries, in key order.
    const many = Object.fromEntries(Array.from({ length: MAX_SESSION_EVALS + 4 }, (_, i) => [`e${String(i).padStart(2, "0")}@1`, good]))
    expect(Object.keys(parseSessionRow({ ...row, evals: many })!.evals!)).toEqual(Object.keys(many).sort().slice(0, MAX_SESSION_EVALS))
  })
})

describe("caregiver.lostCarries", () => {
  it("caregiver.lostCarries: absent or an integer ≥ 1; 0, −1, 1.5 and '2' rejected; not in the to-sum", () => {
    const cg = (c: Record<string, unknown>) => parseLogFile(v2({ caregiver: { carries: 2, to: { a: 2 }, safetyNet: false, ...c } }))
    const r = cg({ lostCarries: 3 })
    expect(r.ok).toBe(true)
    if (r.ok && r.value.schemaVersion === 2) expect(r.value.caregiver).toEqual({ carries: 2, to: { a: 2 }, safetyNet: false, lostCarries: 3 })
    expect(cg({ lostCarries: 1 }).ok).toBe(true)
    const none = cg({})
    expect(none.ok && none.value.schemaVersion === 2 && "lostCarries" in none.value.caregiver!).toBe(false)
    for (const bad of [0, -1, 1.5, "2", null, Number.NaN]) expect(cg({ lostCarries: bad }).ok, String(bad)).toBe(false)
    // Not part of the to-sum: carries 2 with lost 3 needs to summing to 2, not 5.
    expect(parseLogFile(v2({ caregiver: { carries: 5, to: { a: 2 }, safetyNet: false, lostCarries: 3 } })).ok).toBe(false)
    expect(parseLogFile(v2({ caregiver: { carries: 0, to: {}, safetyNet: false, lostCarries: 3 } })).ok).toBe(true)
  })
})

describe("meta (spec §7.3; Plan 05a)", () => {
  const SHA = "ab".repeat(32), PSHA = "cd".repeat(32)
  const searchMeta = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
    v: 1, role: "search", method: "uniform", genomeSchema: 1, genome: { lr: 5e-4, bufferSize: 2000, creditMode: "off" }, genomeSha: SHA,
    drawSeed: 123_456, posteriorSha: PSHA, evalPanel: 3, evalSeed: null, confirmation: null, inheritance: { filters: "none" }, ...over,
  })
  const withMeta = (meta: unknown, over: Record<string, unknown> = {}) => parseLogFile(v2({ meta, ...over }))
  const err = (r: ReturnType<typeof parseLogFile>) => (r.ok ? "" : r.error)
  /** base64 of n float32 (little-endian), as the tab writes it. */
  const floats = (n: number) => btoa(String.fromCharCode(...new Uint8Array(Float32Array.from({ length: n }, (_, i) => i / n).buffer)))
  /** A confirmation life (spec §4.5): confirmation 3's seed block is [13,000, 13,999]. */
  const confirmMeta = (over: Record<string, unknown> = {}) =>
    searchMeta({ role: "confirm", method: "confirm", evalPanel: null, evalSeed: 13_004, confirmation: { id: 3, arm: "candidate" }, ...over })
  const legacyMeta = (over: Record<string, unknown> = {}) =>
    searchMeta({ role: "legacy", method: "legacy", drawSeed: null, posteriorSha: null, evalPanel: null, ...over })
  const newbornMeta = (over: Record<string, unknown> = {}) =>
    searchMeta({ role: "newborn", method: "newborn", drawSeed: null, posteriorSha: null, evalPanel: 5, heldOut: { loss: 0.03, baseline: 0.025 }, ...over })
  /** A confirmation newborn (decision 0024): confirmation 3's seed 13,006 (j = 6, arm newborn), no panel, a held-out score. */
  const confirmNewbornMeta = (over: Record<string, unknown> = {}) =>
    confirmMeta({ evalSeed: 13_006, confirmation: { id: 3, arm: "newborn" }, heldOut: { loss: 0.03, baseline: 0.025 }, ...over })
  /** A newborn reference's file; `meta` null leaves meta out. */
  const newbornFile = (meta: unknown = newbornMeta(), over: Record<string, unknown> = {}) => v2({
    seq: 0, cu: 0, stepsTotal: 0, final: true, curve: [], evalCU: 3, caregiver: { carries: 0, to: {}, safetyNet: false },
    checkups: NEWBORN_LEVELS.map(l => checkup(l, { steps: 0 })), ...(meta === null ? {} : { meta }), ...over,
  })

  it("MAX_EVAL_PANEL is @blanktab/meta's PANELS", () => {
    expect(MAX_EVAL_PANEL).toBe(PANELS)
  })

  it("meta: a full search block parses; each wrong key or value fails with a named error (one case per field)", () => {
    const r = withMeta(searchMeta())
    expect(err(r)).toBe("")
    if (r.ok && r.value.schemaVersion === 2) expect(r.value.meta).toEqual(searchMeta())
    // Round trip: what parses is what was written.
    if (r.ok) expect(JSON.parse(JSON.stringify(r.value))).toEqual(v2({ meta: searchMeta() }))
    const cases: [string, unknown, string][] = [
      ["not an object", "x", "meta must be an object"],
      ["unknown key", searchMeta({ extra: 1 }), 'unknown key "extra" in meta'],
      ["v", searchMeta({ v: 2 }), "meta.v"],
      ["role", searchMeta({ role: "elder" }), "meta.role"],
      ["method off the list", searchMeta({ method: "guess" }), "meta.method"],
      ["method of another role", searchMeta({ method: "newborn" }), "meta.method"],
      ["genomeSchema 0", searchMeta({ genomeSchema: 0 }), "meta.genomeSchema"],
      ["genomeSchema fractional", searchMeta({ genomeSchema: 1.5 }), "meta.genomeSchema"],
      ["genomeSchema above the cap", searchMeta({ genomeSchema: MAX_GENOME_SCHEMA + 1 }), "meta.genomeSchema"],
      ["genome not an object", searchMeta({ genome: [] }), "meta.genome must be an object"],
      ["genome 41 keys", searchMeta({ genome: Object.fromEntries(Array.from({ length: 41 }, (_, i) => [`g${i}`, i])) }), "meta.genome may have at most 40"],
      ["genome key", searchMeta({ genome: { "bad-key": 1 } }), "meta.genome key"],
      ["genome NaN", searchMeta({ genome: { lr: NaN } }), "meta.genome.lr"],
      ["genome boolean", searchMeta({ genome: { lr: true } }), "meta.genome.lr"],
      ["genome 65-char string", searchMeta({ genome: { mode: "x".repeat(65) } }), "meta.genome.mode"],
      ["genomeSha", searchMeta({ genomeSha: "AB".repeat(32) }), "meta.genomeSha"],
      ["drawSeed negative", searchMeta({ drawSeed: -1 }), "meta.drawSeed"],
      ["drawSeed above uint32", searchMeta({ drawSeed: 2 ** 32 }), "meta.drawSeed"],
      ["posteriorSha", searchMeta({ posteriorSha: "x" }), "meta.posteriorSha"],
      ["drawSeed without posteriorSha", searchMeta({ posteriorSha: null }), "meta.drawSeed and meta.posteriorSha must be null together"],
      ["evalPanel 0", searchMeta({ evalPanel: 0 }), "meta.evalPanel"],
      ["evalPanel 17", searchMeta({ evalPanel: 17 }), "meta.evalPanel"],
      ["evalSeed 9999", confirmMeta({ evalSeed: 9999 }), "meta.evalSeed"],
      ["evalSeed fractional", confirmMeta({ evalSeed: 13_000.5 }), "meta.evalSeed"],
      ["confirmation shape", confirmMeta({ confirmation: { id: 1 } }), "meta.confirmation.arm"],
      ["confirmation key", confirmMeta({ confirmation: { id: 3, arm: "centre", x: 1 } }), 'unknown key "x" in meta.confirmation'],
      ["confirmation id", confirmMeta({ confirmation: { id: -1, arm: "centre" } }), "meta.confirmation.id"],
      ["confirmation arm", confirmMeta({ confirmation: { id: 3, arm: "other" } }), "meta.confirmation.arm"],
      ["inheritance", searchMeta({ inheritance: { filters: "abc" } }), "meta.inheritance.filters"],
      ["inheritance key", searchMeta({ inheritance: { filters: "none", x: 1 } }), 'unknown key "x" in meta.inheritance'],
      ["missing on search", searchMeta({ missing: [] }), "meta.missing is only for role legacy"],
      ["filters off level", searchMeta({ filters: floats(1664) }), "meta.filters is only in a file at"],
      ["heldOut on search", searchMeta({ heldOut: { loss: 0.1, baseline: 0.1 } }), "meta.heldOut is only for role newborn"],
      ["end in a non-final file", searchMeta({ end: "budget" }), "meta.end is only in a final file"],
    ]
    for (const [name, meta, want] of cases) expect(err(withMeta(meta)), name).toContain(want)
    // Optional fields stay optional; nulls are accepted where the spec allows them.
    expect(err(withMeta(searchMeta({ drawSeed: null, posteriorSha: null, evalPanel: null })))).toBe("")
    expect(err(withMeta(confirmMeta()))).toBe("")
    expect(err(withMeta(searchMeta({ method: "ts", drawSeed: 2 ** 32 - 1 })))).toBe("")
  })

  it("confirmation and evalSeed are set iff role confirm; evalSeed lies in its confirmation's block [10,000 + 1,000·id, 10,999 + 1,000·id] (spec §4.5)", () => {
    expect(err(withMeta(confirmMeta({ evalSeed: 13_005, confirmation: { id: 3, arm: "centre" } })))).toBe("")
    expect(err(withMeta(confirmMeta({ evalSeed: 13_000 })))).toBe("")
    expect(err(withMeta(confirmMeta({ evalSeed: 13_997, confirmation: { id: 3, arm: "centre" } })))).toBe("")
    expect(err(parseLogFile(newbornFile(confirmNewbornMeta({ evalSeed: 13_999 }))))).toBe("")
    expect(err(withMeta(confirmMeta({ evalSeed: 10_000, confirmation: { id: 0, arm: "candidate" } })))).toBe("")
    expect(err(withMeta(confirmMeta({ evalSeed: 12_999 })))).toContain("meta.evalSeed must lie in confirmation 3's block [13000, 13999]")
    expect(err(withMeta(confirmMeta({ evalSeed: 14_000 })))).toContain("meta.evalSeed must lie in confirmation 3's block")
    // A confirm life needs both; it meets the confirmation's own variants, so it has no panel.
    expect(err(withMeta(confirmMeta({ confirmation: null })))).toContain("meta.confirmation must be set for role confirm")
    expect(err(withMeta(confirmMeta({ evalSeed: null })))).toContain("meta.evalSeed must be set for role confirm")
    expect(err(withMeta(confirmMeta({ evalPanel: 3 })))).toContain("meta.evalPanel must be null for role confirm")
    // Every other role has neither.
    const other = [searchMeta(), legacyMeta()]
    for (const m of other) {
      expect(err(withMeta({ ...m, confirmation: { id: 3, arm: "centre" } })), String(m.role)).toContain("meta.confirmation is only for role confirm")
      expect(err(withMeta({ ...m, evalSeed: 13_004 })), String(m.role)).toContain("meta.evalSeed is only for role confirm")
    }
    expect(err(parseLogFile(newbornFile(newbornMeta({ evalSeed: 13_004 }))))).toContain("meta.evalSeed is only for role confirm")
    expect(err(parseLogFile(newbornFile(newbornMeta({ confirmation: { id: 3, arm: "newborn" } }))))).toContain("meta.confirmation is only for role confirm")
  })

  it("a confirm-newborn file (cu 0, final, three checkups above cu, heldOut) parses; the same shape with arm candidate fails; arm newborn at cu 600 fails", () => {
    const r = parseLogFile(newbornFile(confirmNewbornMeta()))
    expect(err(r)).toBe("")
    if (r.ok && r.value.schemaVersion === 2) expect(r.value.meta).toEqual(confirmNewbornMeta())
    expect(err(parseLogFile(JSON.parse(JSON.stringify(newbornFile(confirmNewbornMeta({ heldOut: undefined }))))))).toBe("") // heldOut optional
    expect(err(parseLogFile(newbornFile(confirmNewbornMeta({ end: "orphan" }))))).toBe("")
    // The same frozen shape for arm candidate (seed 13,004): no frozen checkups, no heldOut.
    const cand = confirmNewbornMeta({ evalSeed: 13_004, confirmation: { id: 3, arm: "candidate" } })
    expect(err(parseLogFile(newbornFile(cand)))).toContain("meta.heldOut is only for role newborn or a confirmation newborn")
    expect(err(parseLogFile(JSON.parse(JSON.stringify(newbornFile({ ...cand, heldOut: undefined })))))).toContain("level is ahead")
    // A candidate life that ended at birth (a final file at cu 0, no checkups) is still a valid life.
    expect(err(parseLogFile(JSON.parse(JSON.stringify(newbornFile({ ...cand, heldOut: undefined, end: "orphan" }, { checkups: [] })))))).toBe("")
    // Arm newborn in any other shape: a running life's file, a checkpoint, a later seq, too few checkups, a panel.
    const cases: [string, Record<string, unknown>, string][] = [
      ["cu 600", newbornFile(confirmNewbornMeta(), { cu: 600 }), "a confirmation newborn (arm newborn) needs a final file at cu 0 and stepsTotal 0"],
      ["not final", newbornFile(confirmNewbornMeta(), { final: false }), "a confirmation newborn (arm newborn) needs a final file"],
      ["steps", newbornFile(confirmNewbornMeta(), { stepsTotal: 5 }), "a confirmation newborn (arm newborn) needs a final file"],
      ["seq 1", newbornFile(confirmNewbornMeta(), { seq: 1 }), "must have seq 0"],
      ["two checkups", newbornFile(confirmNewbornMeta(), { checkups: [checkup(512, { steps: 0 }), checkup(1024, { steps: 0 })] }), "exactly 3 checkups"],
      ["a panel", newbornFile(confirmNewbornMeta({ evalPanel: 3 })), "meta.evalPanel must be null for role confirm"],
    ]
    for (const [name, f, want] of cases) expect(err(parseLogFile(f)), name).toContain(want)
    expect(parseMeta(confirmNewbornMeta(), { cu: 600, stepsTotal: 1000, final: false })).toContain("needs a final file at cu 0")
    expect(parseMeta(JSON.parse(JSON.stringify(confirmNewbornMeta({ heldOut: undefined }))), { cu: 0, stepsTotal: 0, final: false })).toContain("needs a final file at cu 0")
  })

  it("confirmation.arm must equal ARM_OF_J[(evalSeed − 10000 − 1000·id) mod 4]: 13,004 candidate ok, 13,005 candidate fails, 13,006 newborn ok", () => {
    expect(CONFIRM_ARM_OF_J).toEqual(["candidate", "centre", "newborn", "newborn"])
    expect(err(withMeta(confirmMeta({ evalSeed: 13_004, confirmation: { id: 3, arm: "candidate" } })))).toBe("")
    expect(err(withMeta(confirmMeta({ evalSeed: 13_005, confirmation: { id: 3, arm: "candidate" } }))))
      .toContain("meta.confirmation.arm must be centre, the arm of evalSeed 13005 in confirmation 3")
    expect(err(parseLogFile(newbornFile(confirmNewbornMeta({ evalSeed: 13_006 }))))).toBe("")
    expect(err(parseLogFile(newbornFile(confirmNewbornMeta({ evalSeed: 13_007 }))))).toBe("") // j mod 4 = 3: newborn too
    expect(err(parseLogFile(newbornFile(confirmNewbornMeta({ evalSeed: 13_008 }))))).toContain("meta.confirmation.arm must be candidate")
    // Every seed of a block, for each arm: exactly the seed's arm parses (j counted from the block's own base).
    for (const id of [0, 7]) {
      for (let j = 0; j < 1000; j += 37) {
        const evalSeed = 10_000 + 1000 * id + j
        expect(confirmArmOfSeed(evalSeed, id)).toBe(CONFIRM_ARM_OF_J[j % 4])
        for (const arm of ["candidate", "centre"]) {
          const r = withMeta(confirmMeta({ evalSeed, confirmation: { id, arm } }))
          expect(err(r) === "", `${id}/${j}/${arm}`).toBe(arm === CONFIRM_ARM_OF_J[j % 4])
        }
      }
    }
  })

  it("parseMeta's widths are optional: without them a block without filters parses and one with filters is rejected", () => {
    const f = { cu: 512, stepsTotal: 1000, final: false }
    expect(typeof parseMeta(searchMeta(), f)).toBe("object")
    expect(parseMeta(searchMeta({ filters: floats(1664) }), f)).toContain("meta.filters needs the file's widths")
  })

  it("meta.missing (role legacy): sorted unique names of meta.genome; legacy accepts any genome values (bounds are the gate's, and not for legacy)", () => {
    expect(err(withMeta(legacyMeta({ missing: ["bufferSize", "creditMode"] })))).toBe("")
    expect(err(withMeta(legacyMeta({ genome: { entropyWeight: 0.001 } })))).toBe("")
    expect(err(withMeta(legacyMeta()))).toBe("")
    for (const missing of [["lr", "bufferSize"], ["lr", "lr"], ["sleepInterval"], [1], "lr"]) {
      expect(err(withMeta(legacyMeta({ missing }))), JSON.stringify(missing)).toContain("meta.missing")
    }
  })

  it("meta.budget: strict when present; filters and newborn checkups follow the file's own budget, never the current config", () => {
    const B4096 = { budgetCU: 4096, levels: [1024, 2048, 4096] }
    const r = withMeta(searchMeta({ budget: B4096 }))
    expect(err(r)).toBe("")
    if (r.ok && r.value.schemaVersion === 2) expect(r.value.meta!.budget).toEqual(B4096)
    // A file born under a hypothetical 4096 budget carries filters at 4096, not 512; one without budget (V1) the reverse.
    const gpu = floats(104 * 16)
    expect(err(withMeta(searchMeta({ budget: B4096, filters: gpu }), { cu: 4096 }))).toBe("")
    expect(err(withMeta(searchMeta({ budget: B4096, filters: gpu }), { cu: 512 }))).toContain("meta.filters is only in a file at 1024, 2048, 4096 CU")
    expect(err(withMeta(searchMeta({ filters: gpu }), { cu: 4096 }))).toContain("meta.filters is only in a file at 512, 1024, 2048 CU")
    expect(lifeBudget(undefined)).toEqual(V1_BUDGET)
    expect(lifeBudget({ budget: B4096 })).toEqual(B4096)
    // A newborn reference's checkups are its own budget's levels.
    const nb = newbornMeta({ budget: B4096 })
    expect(err(parseLogFile(newbornFile(nb, { checkups: [1024, 2048, 4096].map(l => checkup(l, { steps: 0 })) })))).toBe("")
    expect(err(parseLogFile(newbornFile(nb)))).toContain("a newborn file must hold exactly 3 checkups, at levels 1024, 2048, 4096")
    for (const [budget, want] of [
      [null, "meta.budget must be { budgetCU, levels }"], [{ budgetCU: 2048, levels: [512, 1024, 2048], x: 1 }, 'meta.budget has unknown key "x"'],
      [{ budgetCU: 2048, levels: [512, 1024] }, "meta.budget levels must be"], [{ budgetCU: 0, levels: [] }, "meta.budget budgetCU"],
      [{ budgetCU: 2048, levels: [1024, 512, 2048] }, "meta.budget levels must be"], [{ budgetCU: 2048, levels: [600, 2048] }, "meta.budget levels must be"],
    ] as const) expect(err(withMeta(searchMeta({ budget }))), JSON.stringify(budget)).toContain(want)
    expect(err(withMeta(legacyMeta({ budget: { budgetCU: 2048, levels: [512, 1024, 2048] } })))).toContain("meta.budget is not for role legacy")
  })

  it("meta.filters only in a file whose levelAtOrBelow(cu) is 512, 1024 or 2048, of exactly 104·conv1 floats", () => {
    const gpu = floats(104 * 16)
    for (const cu of [512, 700, 1024, 1500, 2048, 3071.9]) expect(err(withMeta(searchMeta({ filters: gpu }), { cu })), String(cu)).toBe("")
    for (const cu of [300, 511.9, 3072, 4096]) expect(err(withMeta(searchMeta({ filters: gpu }), { cu })), String(cu)).toContain("meta.filters is only in a file at")
    expect(err(withMeta(searchMeta({ filters: floats(1663) }), { cu: 512 }))).toContain("meta.filters must be base64 of 1664 float32")
    expect(err(withMeta(searchMeta({ filters: floats(1665) }), { cu: 512 }))).toContain("meta.filters must be base64")
    // CPU widths (conv1 8): 832 floats, and the GPU length is rejected there.
    const cpu = { conv1: 8, conv2: 16, enc: 64, hidden: 128 }
    expect(err(withMeta(searchMeta({ filters: floats(832) }), { cu: 1024, widths: cpu }))).toBe("")
    expect(err(withMeta(searchMeta({ filters: gpu }), { cu: 1024, widths: cpu }))).toContain("meta.filters must be base64 of 832 float32")
    // Not base64 at the right length; not a string.
    const bad = "!" + gpu.slice(1)
    expect(err(withMeta(searchMeta({ filters: bad }), { cu: 512 }))).toContain("meta.filters must be base64")
    expect(err(withMeta(searchMeta({ filters: 1 }), { cu: 512 }))).toContain("meta.filters must be base64")
  })

  it("newborn file: cu 0, three checkups at 512/1024/2048 above cu parse; the same checkups without role newborn fail 'level is ahead'", () => {
    expect(err(parseLogFile(newbornFile()))).toBe("")
    expect(err(parseLogFile(newbornFile(searchMeta(), { final: false })))).toContain("level is ahead")
    expect(err(parseLogFile(newbornFile(null, { final: false })))).toContain("level is ahead")
    expect(err(parseLogFile(newbornFile(legacyMeta(), { final: false })))).toContain("level is ahead")
    const cases: [string, Record<string, unknown>, string][] = [
      ["seq 1", newbornFile(newbornMeta(), { seq: 1 }), "seq 0"],
      ["not final", newbornFile(newbornMeta(), { final: false }), "meta.role newborn needs a final file at cu 0"],
      ["cu 1", newbornFile(newbornMeta(), { cu: 1 }), "meta.role newborn needs a final file at cu 0"],
      ["steps", newbornFile(newbornMeta(), { stepsTotal: 1 }), "meta.role newborn needs a final file at cu 0"],
      ["no panel", newbornFile(newbornMeta({ evalPanel: null })), "meta.evalPanel must be set for role newborn"],
      ["method centre", newbornFile(newbornMeta({ method: "centre" })), "meta.method"],
      ["two checkups", newbornFile(newbornMeta(), { checkups: [checkup(512, { steps: 0 }), checkup(1024, { steps: 0 })] }), "exactly 3 checkups"],
      ["no checkups", newbornFile(newbornMeta(), { checkups: [] }), "exactly 3 checkups"],
      ["wrong levels", newbornFile(newbornMeta(), { checkups: [256, 1024, 2048].map(l => checkup(l, { steps: 0 })) }), "exactly 3 checkups"],
      ["steps above 0", newbornFile(newbornMeta(), { checkups: NEWBORN_LEVELS.map(l => checkup(l, { steps: 1 })) }), "steps"],
    ]
    for (const [name, f, want] of cases) expect(err(parseLogFile(f)), name).toContain(want)
    const noCk = newbornFile(); delete noCk.checkups
    expect(err(parseLogFile(noCk))).toContain("exactly 3 checkups")
  })

  it("meta.end only in a final file; heldOut only for newborn", () => {
    for (const end of ["failed", "orphan", "budget"]) expect(err(withMeta(searchMeta({ end }), { final: true })), end).toBe("")
    expect(err(withMeta(searchMeta({ end: "budget" }), { final: false }))).toContain("meta.end is only in a final file")
    expect(err(withMeta(searchMeta({ end: "done" }), { final: true }))).toContain("meta.end must be one of")
    expect(err(parseLogFile(newbornFile(newbornMeta({ end: "budget" }))))).toBe("")
    expect(err(parseLogFile(JSON.parse(JSON.stringify(newbornFile(newbornMeta({ heldOut: undefined }))))))).toBe("") // heldOut is optional
    for (const role of ["search", "legacy", "confirm"]) {
      const h = { heldOut: { loss: 1, baseline: 1 } }
      const m = role === "search" ? searchMeta(h) : role === "legacy" ? legacyMeta(h) : confirmMeta(h)
      expect(err(withMeta(m)), role).toContain("meta.heldOut is only for role newborn")
    }
    for (const heldOut of [{ loss: -1, baseline: 1 }, { loss: 1 }, { loss: 1, baseline: 1, x: 1 }, null]) {
      expect(err(parseLogFile(newbornFile(newbornMeta({ heldOut })))), JSON.stringify(heldOut)).toContain("meta.heldOut")
    }
  })

  it("files without meta still parse, and BUDGET_CU / FILTER_LEVELS / NEWBORN_LEVELS are the spec's", () => {
    expect(ok(v2())).toBe(true)
    expect(BUDGET_CU).toBe(2048)
    expect(FILTER_LEVELS).toEqual([512, 1024, 2048])
    expect(NEWBORN_LEVELS).toEqual([512, 1024, 2048])
    expect(ok({ ...mkFile(ID, 0), meta: searchMeta() })).toBe(false) // v1 has no such field
  })

  it("BUDGET_CU and the levels come from config/evaluation.json; evalBudgetOf rejects a malformed budget", () => {
    expect(evalBudgetOf(evaluationConfig)).toEqual({ budgetCU: BUDGET_CU, levels: FILTER_LEVELS })
    expect(evalBudgetOf({ budgetCU: 3072, levels: [1024, 3072] })).toEqual({ budgetCU: 3072, levels: [1024, 3072] })
    for (const bad of [null, {}, { budgetCU: 2048 }, { budgetCU: 0, levels: [] }, { budgetCU: 2048, levels: [] },
      { budgetCU: 2048, levels: [512, 1024] }, { budgetCU: 2048, levels: [1024, 512, 2048] }, { budgetCU: 2048, levels: [500, 2048] },
      { budgetCU: 2048.5, levels: [2048.5] }, { budgetCU: 2048, levels: ["2048"] }]) {
      expect(() => evalBudgetOf(bad), JSON.stringify(bad)).toThrow(/config\/evaluation.json/)
    }
  })
})
