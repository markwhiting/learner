import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { buildSummary, pinnedScoring, readLearnerLogs } from "../summary.ts"
import { parseNewbornReference, searchComposite } from "@blanktab/evaluators"
import { file, meta, scoredLife } from "../../../packages/evaluators/test/fixtures/build.ts"
import { parseSessionRow, type LogFile, type LogFileV1, type LogFileV2 } from "@blanktab/harness/log"

const A = "lrn_aaaaaaaaaaaaaaaa"
const B = "lrn_bbbbbbbbbbbbbbbb"
const log = (id: string, seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId: id, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: "2026-10-19T12:00:00.000Z", stepsTotal: 100 * (seq + 1), settings: { moveProbability: 0.5 },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})

let dir: string
beforeEach(() => { dir = mkdtempSync(join(tmpdir(), "summary-")) })
afterEach(() => { rmSync(dir, { recursive: true, force: true }) })
const put = (l: LogFile) => {
  mkdirSync(join(dir, l.learnerId), { recursive: true })
  writeFileSync(join(dir, l.learnerId, `${l.seq}.json`), JSON.stringify(l))
}

describe("buildSummary", () => {
  it("one row per learner from its highest-seq file", () => {
    for (let s = 0; s <= 5; s++) put(log(A, s, { final: s === 5, sessionsSeenAtBirth: 3 }))
    put(log(B, 0)); put(log(B, 1, { settings: { moveProbability: 0.9 } }))
    writeFileSync(join(dir, B, "heartbeat.json"), JSON.stringify({ learnerId: B, writtenAt: "2026-10-19T12:00:00.000Z" }))
    const s = buildSummary(dir, null)
    expect(s.schemaVersion).toBe(1)
    expect(s.sessions).toHaveLength(2)
    const a = s.sessions.find(r => r.learnerId === A)!
    const b = s.sessions.find(r => r.learnerId === B)!
    expect(a).toMatchObject({ stepsTotal: 600, final: true, sessionsSeenAtBirth: 3, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64) })
    expect(b).toMatchObject({ stepsTotal: 200, final: false, settings: { moveProbability: 0.9 } })
  })
  it("records lastWrittenAt from the highest-seq file", () => {
    put(log(A, 0, { writtenAt: "2026-10-19T12:00:00.000Z" })); put(log(A, 1, { writtenAt: "2026-10-19T12:05:00.000Z" }))
    expect(buildSummary(dir, null).sessions[0]!.lastWrittenAt).toBe("2026-10-19T12:05:00.000Z")
  })
  it("sorts numerically (seq 10 beats seq 9) and rows by learnerId", () => {
    for (let s = 0; s <= 10; s++) put(log(B, s))
    put(log(A, 0))
    const s = buildSummary(dir, null)
    expect(s.sessions.map(r => r.learnerId)).toEqual([A, B])
    expect(s.sessions[1]!.stepsTotal).toBe(1100)
  })
  it("skips unparseable files and non-learner dirs; missing dir gives empty", () => {
    put(log(A, 0))
    mkdirSync(join(dir, "junk")); writeFileSync(join(dir, "junk", "0.json"), "{}")
    mkdirSync(join(dir, B)); writeFileSync(join(dir, B, "0.json"), "not json")
    expect(buildSummary(dir, null).sessions.map(r => r.learnerId)).toEqual([A])
    expect(buildSummary(join(dir, "nope"), null).sessions).toEqual([])
  })
  it("rows gain cu and lossRatio (the last curve point's predLoss / baselineLoss); null for v1 or without a point", () => {
    const v2 = (id: string, seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => ({
      ...log(id, seq), schemaVersion: 2, cu: 2 ** seq, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 },
      curve: [{ cu: 2 ** seq, steps: 100 * (seq + 1), predLoss: 0.02, baselineLoss: 0.025 }], skippedUpdates: 0, restores: 0, lostCU: 0, ...over,
    })
    put(v2(A, 0)); put(v2(A, 1, { final: true, curve: [{ cu: 1.5, steps: 150, predLoss: 0.03, baselineLoss: 0.04 }, { cu: 2, steps: 200, predLoss: 0.018, baselineLoss: 0.024 }] }))
    put(v2(B, 0, { curve: [{ cu: 1, steps: 100, predLoss: 0.05, baselineLoss: 0.04 }] })); put(v2(B, 1, { curve: [] })) // falls back to an earlier file's point
    const C = "lrn_cccccccccccccccc"
    put(log(C, 0))
    const D = "lrn_dddddddddddddddd"
    put(v2(D, 0, { curve: [] }))
    const rows = Object.fromEntries(buildSummary(dir, null).sessions.map(r => [r.learnerId, r]))
    expect(rows[A]).toMatchObject({ cu: 2, final: true })
    expect(rows[A]!.lossRatio).toBeCloseTo(0.75, 12)
    expect(rows[B]).toMatchObject({ cu: 2 })
    expect(rows[B]!.lossRatio).toBeCloseTo(1.25, 12)
    expect(rows[C]).toMatchObject({ cu: null, lossRatio: null })
    expect(rows[D]).toMatchObject({ cu: 1, lossRatio: null })
  })
  it("a zero baseline gives lossRatio null, not Infinity", () => {
    put({ ...log(A, 0), schemaVersion: 2, cu: 1, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 }, curve: [{ cu: 1, steps: 100, predLoss: 0.1, baselineLoss: 0 }], skippedUpdates: 0, restores: 0, lostCU: 0 })
    expect(buildSummary(dir, null).sessions[0]!.lossRatio).toBeNull()
    expect(JSON.parse(JSON.stringify(buildSummary(dir, null))).sessions[0].lossRatio).toBeNull()
  })
  it("rows gain heldOutRatio: the latest curve point with a held-out score, loss / baseline; null without one", () => {
    const v2 = (id: string, seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => ({
      ...log(id, seq), schemaVersion: 2, cu: 2 ** seq, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 },
      curve: [{ cu: 2 ** seq, steps: 100 * (seq + 1), predLoss: 0.02, baselineLoss: 0.025 }], skippedUpdates: 0, restores: 0, lostCU: 0, ...over,
    })
    const p = (cu: number, steps: number, heldOut?: { loss: number; baseline: number }) => ({ cu, steps, predLoss: 0.01, baselineLoss: 0.04, ...(heldOut ? { heldOut } : {}) })
    put(v2(A, 0, { curve: [p(1, 100, { loss: 0.5, baseline: 0.4 })] }))
    put(v2(A, 1, { curve: [p(1.5, 150, { loss: 0.3, baseline: 0.4 }), p(2, 200)] })) // the newest point has none: the latest one that has
    put(v2(B, 0, { curve: [p(1, 100, { loss: 0.2, baseline: 0.4 })] })); put(v2(B, 1, { curve: [] }))
    const C = "lrn_cccccccccccccccc"
    put(v2(C, 0)) // an older v2 learner: no held-out scores at all
    const D = "lrn_dddddddddddddddd"
    put(v2(D, 0, { curve: [p(1, 100, { loss: 0.2, baseline: 0 })] })) // a zero baseline
    const rows = Object.fromEntries(buildSummary(dir, null).sessions.map(r => [r.learnerId, r]))
    expect(rows[A]!.heldOutRatio).toBeCloseTo(0.75, 12)
    expect(rows[A]!.lossRatio).toBeCloseTo(0.25, 12) // the own-stream ratio stays as it was
    expect(rows[B]!.heldOutRatio).toBeCloseTo(0.5, 12)
    expect(rows[C]!.heldOutRatio).toBeNull()
    expect(rows[D]!.heldOutRatio).toBeNull()
  })
  it("checkupScore is the latest checkup's level and passed.length; null without checkups", () => {
    const v2 = (id: string, seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => ({
      ...log(id, seq), schemaVersion: 2, cu: 1000 * (seq + 1), condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 },
      curve: [], skippedUpdates: 0, restores: 0, lostCU: 0, ...over,
    })
    const ck = (level: number, passed: string[]) => ({ level, steps: 1, passed, measures: { M1: 1, M2: 1, M3: 0, M4: 0, M5: 0, M6: 0 }, evalCU: 0 })
    put(v2(A, 0, { milestoneSuiteVersion: 1, checkups: [ck(64, ["M1"]), ck(128, ["M1", "M2"])], evalCU: 0 })) // checkups are billed inside evalCU
    put(v2(A, 1)) // newest file has none: the latest checkup on record stands
    put(v2(B, 0)); put(log(B, 1))
    const rows = Object.fromEntries(buildSummary(dir, null).sessions.map(r => [r.learnerId, r]))
    expect(rows[A]!.checkupScore).toEqual({ level: 128, score: 2 })
    expect(rows[B]!.checkupScore).toBeNull()
    // What the gate writes, the page keeps (parseSessionRow's bounds: a log level, a score within the suite).
    for (const r of buildSummary(dir, null).sessions) expect(parseSessionRow(JSON.parse(JSON.stringify(r)))!.checkupScore).toEqual(rows[r.learnerId]!.checkupScore)
  })
  it("a full score at a large level survives the page's parser (score 6 of suite 1, level 3072)", () => {
    const ck = { level: 3072, steps: 1, passed: ["M1", "M2", "M3", "M4", "M5", "M6"], measures: { M1: 1, M2: 1, M3: 1, M4: 1, M5: 1, M6: 1 }, evalCU: 0 }
    put({ ...log(A, 0), schemaVersion: 2, cu: 4000, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 }, curve: [], skippedUpdates: 0, restores: 0, lostCU: 0, milestoneSuiteVersion: 1, checkups: [ck], evalCU: 0 })
    const row = buildSummary(dir, null).sessions[0]!
    expect(row.checkupScore).toEqual({ level: 3072, score: 6 })
    expect(parseSessionRow(JSON.parse(JSON.stringify(row)))!.checkupScore).toEqual({ level: 3072, score: 6 })
  })
})

describe("buildSummary scores (Plan 05a Task 9)", () => {
  const REF = parseNewbornReference(JSON.parse(readFileSync(new URL("../../../packages/evaluators/test/fixtures/reference.json", import.meta.url), "utf8")))!
  it("rows carry evals['search-composite@1'] when scoring is given; absent otherwise (old summaries)", () => {
    const S = "lrn_5555555555555555"
    for (const f of scoredLife(S, { ms: { M2: 1, M3: 2, M4: 0.25, M5: 0.7, M6: 0 }, ratio: 0.9 })) put(f)
    put(log(A, 0)) // a v1 life: not a search life
    const scored = buildSummary(dir, { evaluator: searchComposite, refs: { newborn: REF } })
    const rows = Object.fromEntries(scored.sessions.map(r => [r.learnerId, r]))
    const want = searchComposite.evaluateLife!({ learnerId: S, files: readLearnerLogs(join(dir, S)) }, { newborn: REF })
    expect(want.eligible).toBe(true)
    expect(rows[S]!.evals).toEqual({ "search-composite@1": { eligible: true, score: want.score, C: want.components.C, H: want.components.H } })
    expect(rows[A]!.evals).toEqual({ "search-composite@1": { eligible: false, score: null, C: null, H: null } })
    // Without a reference every row says ineligible; the page keeps what the gate writes.
    const noRef = buildSummary(dir, { evaluator: searchComposite, refs: {} })
    for (const r of noRef.sessions) expect(r.evals).toEqual({ "search-composite@1": { eligible: false, score: null, C: null, H: null } })
    for (const r of scored.sessions) expect(parseSessionRow(JSON.parse(JSON.stringify(r)))!.evals).toEqual(r.evals)
    // Without scoring (as before 05a): no evals key at all.
    for (const r of buildSummary(dir, null).sessions) expect("evals" in r).toBe(false)
  })
})

describe("buildSummary recipe (plan 05b, Task 11)", () => {
  it("rows carry recipe {role, method} from the first meta block; parseSessionRow drops a malformed recipe, not the row", () => {
    const T = "lrn_7777777777777777", L = "lrn_6666666666666666", C = "lrn_8888888888888888"
    const ts = meta({ method: "ts" })
    put(file({ learnerId: T, seq: 0, cu: 10, meta: ts }))
    put(file({ learnerId: T, seq: 1, cu: 20, meta: { ...ts, end: "failed" }, final: true }))
    // No block in its first file (older code), then one: the first block is the recipe.
    put(file({ learnerId: L, seq: 0, cu: 10, meta: null }))
    put(file({ learnerId: L, seq: 1, cu: 20, meta: meta({ role: "legacy", method: "legacy", evalPanel: null, drawSeed: null, posteriorSha: null }) }))
    put(file({ learnerId: C, seq: 0, cu: 10, meta: meta({ role: "confirm", method: "confirm", evalPanel: null, evalSeed: 10_000, confirmation: { id: 0, arm: "candidate" } }) }))
    put(log(A, 0)) // a v1 life: no block, no recipe
    const rows = Object.fromEntries(buildSummary(dir, null).sessions.map(r => [r.learnerId, r]))
    expect(rows[T]!.recipe).toEqual({ role: "search", method: "ts" })
    expect(rows[L]!.recipe).toEqual({ role: "legacy", method: "legacy" })
    expect(rows[C]!.recipe).toEqual({ role: "confirm", method: "confirm" })
    expect("recipe" in rows[A]!).toBe(false)
    for (const r of Object.values(rows)) expect(parseSessionRow(JSON.parse(JSON.stringify(r)))).toEqual(r)
    // Malformed: an unknown role or method, a method its role never has, extra keys, not an object. The row stays.
    for (const bad of [{ role: "x", method: "ts" }, { role: "search", method: "y" }, { role: "search", method: "confirm" }, { role: "search", method: "ts", extra: 1 }, "ts", null, 5]) {
      const r = parseSessionRow({ ...JSON.parse(JSON.stringify(rows[T]!)), recipe: bad })
      expect(r, JSON.stringify(bad)).not.toBeNull()
      expect(r!.learnerId).toBe(T)
      expect("recipe" in r!, JSON.stringify(bad)).toBe(false)
    }
  })
})

describe("pinnedScoring (every summary writer's scoring)", () => {
  it("is the search evaluator config/evaluation.json pins, with its pinned references; its rows carry that key", () => {
    const repo = fileURLToPath(new URL("../../..", import.meta.url))
    const s = pinnedScoring(repo)
    expect(`${s.evaluator.id}@${s.evaluator.version}`).toBe(JSON.parse(readFileSync(join(repo, "config/evaluation.json"), "utf8")).search)
    put(log(A, 0))
    expect(Object.keys(buildSummary(dir, s).sessions[0]!.evals!)).toEqual(["search-composite@1"])
  })
})
