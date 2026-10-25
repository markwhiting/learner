import { describe, it, expect, vi } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD, isValidAction, type Action } from "@blanktab/interface"
import { DEFAULT_SETTINGS, createLearner, type FrozenRun, type FrozenRunOptions, type LearnerStats, type Widths } from "@blanktab/learner"
import { createCpuBackend, createRng } from "@blanktab/tensor"
import { createWorld, hashSeed, type World } from "@blanktab/world"
import { uniformAction } from "../src/heldout"
import { ERROR_REASON_RE, MAX_EVAL_PANEL, MILESTONE_IDS, MILESTONE_SUITE_VERSION, NEWBORN_LEVELS } from "../src/log"
import { MILESTONES_V1, YIELD_EVERY, type Milestone } from "../src/milestones"
import { checkupSeed, confirmSeed, errorReason, panelSeed, runCheckup, variantSeedOf, type CheckupDeps } from "../src/checkup"

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const canvas = () => createCanvas(WORLD.width, WORLD.height)
const SUITE_IDS = MILESTONE_IDS[MILESTONE_SUITE_VERSION]!

/**
 * A stub learner: frozen runs act uniformly at random from their seed; each step bills 0.001 evalCU. The run opened
 * `disposeThrows`-th (0-based) closes but then throws from dispose().
 */
function stubLearner(disposeThrows = -1) {
  const stats = { evalCU: 0.5, cu: 3 } as LearnerStats
  const log = { openRuns: 0, opened: [] as FrozenRunOptions[], disposed: 0 }
  return {
    log,
    learner: {
      stats: () => ({ ...stats }),
      frozenRun(o: FrozenRunOptions): FrozenRun {
        if (log.openRuns > 0) throw new Error("stub: a frozen run is already open")
        log.openRuns++
        log.opened.push(o)
        const rnd = createRng(o.seed)
        const index = log.opened.length - 1
        let closed = false
        return {
          step: async () => { stats.evalCU += 0.001; return uniformAction(rnd) },
          result: () => ({ predLoss: NaN, baselineLoss: NaN, transitions: 0 }),
          dispose() {
            if (!closed) { closed = true; log.openRuns--; log.disposed++ }
            if (index === disposeThrows) throw new Error("stub: dispose failed")
          },
        }
      },
    },
  }
}

/** The v1 suite at small budgets (same worlds, measures and pass rules). */
const small = (budget = 20): Milestone[] => MILESTONES_V1.map(m => ({ ...m, budgetSteps: budget }))
const base = (over: Partial<CheckupDeps> & Pick<CheckupDeps, "learner">): CheckupDeps =>
  ({ learnerId: "abc", level: 8, steps: 1234, canvas, suite: small(), ...over })

describe("checkups (decision 0016)", { timeout: 60_000 }, () => {
  it("checkupSeed is deterministic and differs across learnerId, level and milestone", () => {
    const s = checkupSeed("abc", 8, "M1")
    expect(checkupSeed("abc", 8, "M1")).toBe(s)
    expect(s).toBe(hashSeed(0, "abc/8/M1"))
    const others = [checkupSeed("abd", 8, "M1"), checkupSeed("abc", 16, "M1"), checkupSeed("abc", 8, "M2")]
    expect(new Set([s, ...others]).size).toBe(4)
  })

  it("each milestone acts in a new world of its own on a new canvas, seeded checkupSeed; none is reused across milestones or checkups", async () => {
    type Canvas = ReturnType<typeof canvas>
    const issued: Canvas[] = []
    const made: { id: string; seed: number; canvas: unknown; world: World }[] = []
    const suite = small().map(m => ({
      ...m,
      makeVariant(seed: number, c: Parameters<Milestone["makeVariant"]>[1]) {
        const world = m.makeVariant(seed, c)
        made.push({ id: m.id, seed, canvas: c, world })
        return world
      },
    }))
    const deps = (learner: CheckupDeps["learner"]) => base({ learner, suite, canvas: () => { const c = canvas(); issued.push(c); return c } })
    const first = stubLearner(), second = stubLearner()
    expect((await runCheckup(deps(first.learner))).status).toBe("done")
    expect(made.map(x => [x.id, x.seed])).toEqual(SUITE_IDS.map(id => [id, checkupSeed("abc", 8, id)]))
    expect((await runCheckup(deps(second.learner))).status).toBe("done")
    expect(made).toHaveLength(12)
    // Every d.canvas() call gave a distinct canvas (M5 asks for a second one for its reference), and each variant drew
    // on one of them, never another's.
    expect(issued).toHaveLength(14)
    expect(new Set(issued).size).toBe(14)
    expect(made.every(x => issued.includes(x.canvas as Canvas))).toBe(true)
    expect(new Set(made.map(x => x.canvas)).size).toBe(12)
    // Twelve distinct World instances, each the one acted in (from tick 0, for exactly the budget).
    expect(new Set(made.map(x => x.world)).size).toBe(12)
    for (const x of made) expect(x.world.tick).toBe(20)
    // The policy: a frozen run per milestone, seeded hashSeed(variant seed, "policy"), measuring loss for M1 only.
    expect(first.log.opened.map(o => o.seed)).toEqual(SUITE_IDS.map(id => hashSeed(checkupSeed("abc", 8, id), "policy")))
    expect(first.log.opened.map(o => o.measureLoss)).toEqual(SUITE_IDS.map(id => id === "M1"))
  })

  it("with variantSeed, milestone m meets makeVariant(panelSeed(p, level, m.id)); two learners on panel p meet identical variants; without it, checkupSeed as before", async () => {
    expect(panelSeed(3, 512, "M1")).toBe(hashSeed(0, "panel/3/512/M1"))
    expect(new Set([panelSeed(3, 512, "M1"), panelSeed(4, 512, "M1"), panelSeed(3, 1024, "M1"), panelSeed(3, 512, "M2")]).size).toBe(4)
    const seen = (learnerId: string, variantSeed?: CheckupDeps["variantSeed"]) => {
      const seeds: [string, number][] = []
      const suite = small(5).map(m => ({ ...m, makeVariant(seed: number, c: Parameters<Milestone["makeVariant"]>[1]) { seeds.push([m.id, seed]); return m.makeVariant(seed, c) } }))
      const { learner, log } = stubLearner()
      return runCheckup(base({ learnerId, level: 512, learner, suite, ...(variantSeed ? { variantSeed } : {}) })).then(o => ({ o, seeds, policy: log.opened.map(r => r.seed) }))
    }
    const onPanel = (l: number, m: string) => panelSeed(3, l, m)
    const a = await seen("lrn_aaaaaaaaaaaaaaaa", onPanel), b = await seen("lrn_bbbbbbbbbbbbbbbb", onPanel)
    expect(a.o.status).toBe("done")
    expect(a.seeds).toEqual(SUITE_IDS.map(id => [id, panelSeed(3, 512, id)]))
    expect(b.seeds).toEqual(a.seeds)
    // The policy seed follows the variant seed, so two learners on one panel differ only by what they learned.
    expect(a.policy).toEqual(SUITE_IDS.map(id => hashSeed(panelSeed(3, 512, id), "policy")))
    expect(b.policy).toEqual(a.policy)
    const c = await seen("lrn_aaaaaaaaaaaaaaaa")
    expect(c.seeds).toEqual(SUITE_IDS.map(id => [id, checkupSeed("lrn_aaaaaaaaaaaaaaaa", 512, id)]))
    expect((await seen("lrn_bbbbbbbbbbbbbbbb")).seeds).not.toEqual(c.seeds)
  })

  it("with variantSeed confirmSeed(e, …), two learners with evalSeed e meet identical variants, disjoint from every panelSeed(p, …) on the fixture levels", async () => {
    const e = 13_006
    expect(confirmSeed(e, 512, "M1")).toBe(hashSeed(0, `confirm/${e}/512/M1`))
    const seen = async (learnerId: string, evalSeed: number) => {
      const seeds: number[] = []
      const suite = small(5).map(m => ({ ...m, makeVariant(seed: number, c: Parameters<Milestone["makeVariant"]>[1]) { seeds.push(seed); return m.makeVariant(seed, c) } }))
      const { learner } = stubLearner()
      // The session's and the newborn runner's rule for a confirmation life's block (variantSeedOf).
      const variantSeed = variantSeedOf({ evalSeed, evalPanel: null })!
      for (const level of NEWBORN_LEVELS) expect((await runCheckup(base({ learnerId, level, learner, suite, variantSeed }))).status).toBe("done")
      return seeds
    }
    const a = await seen("lrn_aaaaaaaaaaaaaaaa", e), b = await seen("lrn_bbbbbbbbbbbbbbbb", e)
    expect(a).toEqual(NEWBORN_LEVELS.flatMap(l => SUITE_IDS.map(id => confirmSeed(e, l, id))))
    expect(b).toEqual(a)
    expect(await seen("lrn_aaaaaaaaaaaaaaaa", e + 1)).not.toEqual(a)
    // Disjoint from every shared panel's variants at the fixture levels: a confirmation life never meets a search world.
    const panels = new Set<number>()
    for (let p = 1; p <= MAX_EVAL_PANEL; p++) for (const l of NEWBORN_LEVELS) for (const id of SUITE_IDS) panels.add(panelSeed(p, l, id))
    for (const s of a) expect(panels.has(s)).toBe(false)
    // variantSeedOf: confirmSeed for a confirm block, panelSeed for a panel, checkupSeed (undefined) otherwise.
    expect(variantSeedOf({ evalSeed: null, evalPanel: 3 })!(512, "M2")).toBe(panelSeed(3, 512, "M2"))
    expect(variantSeedOf({ evalSeed: null, evalPanel: null })).toBeUndefined()
    expect(variantSeedOf(undefined)).toBeUndefined()
  })

  it("a dispose() that throws marks that milestone not passed, measure null, error true; the rest still run", async () => {
    const { learner, log } = stubLearner(2) // M3's run
    const suite = small(10).map(m => ({ ...m, passes: () => true }))
    const out = await runCheckup(base({ learner, suite }))
    if (out.status !== "done") throw new Error("abandoned")
    expect(out.result.error).toBe(true)
    expect(out.result.passed).toEqual(SUITE_IDS.filter(id => id !== "M3"))
    expect(out.result.measures.M3).toBeNull()
    expect(log.opened).toHaveLength(6)
    expect(log.openRuns).toBe(0)
  })

  it("a dispose() that throws does not mask a recoverable error in flight", async () => {
    const { learner, log } = stubLearner(0)
    class Lost extends Error {}
    const suite = small(10).map(m => (m.id === "M1" ? { ...m, measure: () => { throw new Lost("device lost") } } : m))
    await expect(runCheckup(base({ learner, suite, recoverable: e => e instanceof Lost }))).rejects.toBeInstanceOf(Lost)
    expect(log.opened).toHaveLength(1)
    expect(log.openRuns).toBe(0)
  })

  it("result: level, steps, passed in suite order, a measure per id, evalCU = the learner's evalCU delta", async () => {
    const { learner } = stubLearner()
    // Force passes: M3 and M1 always pass, M2 never; the order of `passed` follows the suite.
    const suite = small(10).map(m => (m.id === "M3" || m.id === "M1" ? { ...m, passes: () => true } : m.id === "M2" ? { ...m, passes: () => false } : m))
    const out = await runCheckup(base({ learner, suite, level: 16, steps: 777 }))
    if (out.status !== "done") throw new Error("abandoned")
    const r = out.result
    expect(r.level).toBe(16)
    expect(r.steps).toBe(777)
    expect(r.passed.slice(0, 2)).toEqual(["M1", "M3"])
    expect(r.passed).toEqual(SUITE_IDS.filter(id => r.passed.includes(id)))
    expect(Object.keys(r.measures)).toEqual([...SUITE_IDS])
    for (const v of Object.values(r.measures)) expect(v === null || Number.isFinite(v)).toBe(true)
    expect(r.measures.M1).toBeNull() // the stub measures no loss: NaN → null
    expect(r.evalCU).toBeCloseTo(6 * 10 * 0.001, 12)
    expect(r.error).toBeUndefined()
  })

  it("a milestone that throws is not passed, measure null, error true; the others still run; every frozen run is disposed", async () => {
    const { learner, log } = stubLearner()
    const suite = small(10).map(m => (m.id === "M3" ? { ...m, measure: () => { throw new Error("boom") }, passes: () => true } : m.id === "M4" ? { ...m, passes: () => true } : m))
    const out = await runCheckup(base({ learner, suite }))
    if (out.status !== "done") throw new Error("abandoned")
    expect(out.result.error).toBe(true)
    expect(out.result.measures.M3).toBeNull()
    expect(out.result.passed).not.toContain("M3")
    expect(out.result.passed).toContain("M4") // a later milestone still ran
    expect(Object.keys(out.result.measures)).toEqual([...SUITE_IDS])
    expect(log.opened).toHaveLength(6)
    expect(log.openRuns).toBe(0)
    expect(log.disposed).toBe(6)
  })

  it("errorReason: Error → 'RangeError: bad'; a string throw → 'thrown: …'; control chars and non-ASCII → '?'; 200 chars → 80; '' → 'unknown'", () => {
    expect(errorReason(new RangeError("bad"))).toBe("RangeError: bad")
    expect(errorReason("oops")).toBe("thrown: oops")
    expect(errorReason(42)).toBe("thrown: 42")
    expect(errorReason(new Error("a\u0001b\u00e9c\u{1F600}"))).toBe("Error: a?b?c??")
    expect(errorReason(new Error("line one\n\n  line\ttwo"))).toBe("Error: line one line two")
    const long = errorReason(new Error("x".repeat(200)))
    expect(long).toHaveLength(80)
    expect(long.startsWith("Error: xxx")).toBe(true)
    expect(errorReason("")).toBe("unknown")
    expect(errorReason(Object.assign(new Error(""), { name: "" }))).toBe("unknown")
    expect(errorReason(new TypeError(""))).toBe("TypeError")
    expect(errorReason({ toString() { throw new Error("no") } })).toMatch(ERROR_REASON_RE)
    for (const e of [undefined, null, Symbol("s"), { a: 1 }, "é".repeat(500), new Error("\n")]) expect(errorReason(e)).toMatch(ERROR_REASON_RE)
  })

  it("a milestone that throws records its reason (≤ 80 chars) and warns once; the others have none", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    try {
      const { learner } = stubLearner()
      const suite = small(10).map(m => (m.id === "M3" ? { ...m, measure: () => { throw new RangeError("boom ".repeat(40)) } } : m))
      const out = await runCheckup(base({ learner, suite }))
      if (out.status !== "done") throw new Error("abandoned")
      expect(Object.keys(out.result.errorReasons ?? {})).toEqual(["M3"])
      const reason = out.result.errorReasons!.M3!
      expect(reason).toMatch(ERROR_REASON_RE)
      expect(reason.startsWith("RangeError: boom boom")).toBe(true)
      expect(warn).toHaveBeenCalledTimes(1)
      expect(warn.mock.calls[0]!.slice(0, 3)).toEqual(["blanktab: milestone", "M3", "threw"])
    } finally { warn.mockRestore() }
  })

  it("a dispose that throws records the dispose error's reason", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    try {
      const { learner } = stubLearner(2) // M3's run
      const out = await runCheckup(base({ learner }))
      if (out.status !== "done") throw new Error("abandoned")
      expect(out.result.errorReasons).toEqual({ M3: "Error: stub: dispose failed" })
      expect(warn).toHaveBeenCalledTimes(1)
    } finally { warn.mockRestore() }
  })

  it("no errorReasons without an error", async () => {
    const { learner } = stubLearner()
    const out = await runCheckup(base({ learner }))
    if (out.status !== "done") throw new Error("abandoned")
    expect("errorReasons" in out.result).toBe(false)
  })

  it("a recoverable error is rethrown after disposing the open run", async () => {
    const { learner, log } = stubLearner()
    class Lost extends Error {}
    const suite = small(10).map(m => (m.id === "M2" ? { ...m, makeVariant: (s: number, c: Parameters<Milestone["makeVariant"]>[1]) => { const w = m.makeVariant(s, c); let n = 0; return { ...w, tick: w.tick, observe: () => { if (++n === 5) throw new Lost("device lost"); return w.observe() } } as typeof w } } : m))
    await expect(runCheckup(base({ learner, suite, recoverable: e => e instanceof Lost }))).rejects.toBeInstanceOf(Lost)
    expect(log.opened).toHaveLength(2)
    expect(log.openRuns).toBe(0)
  })

  it("shouldStop() true before a milestone: abandoned, no result, nothing left open", async () => {
    const { learner, log } = stubLearner()
    let calls = 0
    const out = await runCheckup(base({ learner, shouldStop: () => ++calls > 3 }))
    expect(out).toEqual({ status: "abandoned" })
    expect(log.opened).toHaveLength(3)
    expect(log.openRuns).toBe(0)
    // And a halt during the last milestone abandons the checkup too.
    const s2 = stubLearner()
    let started = 0
    const suite = small(10).map(m => ({ ...m, makeVariant: (s: number, c: Parameters<Milestone["makeVariant"]>[1]) => { started++; return m.makeVariant(s, c) } }))
    const out2 = await runCheckup(base({ learner: s2.learner, suite, shouldStop: () => started === 6 }))
    expect(out2).toEqual({ status: "abandoned" })
    expect(s2.log.openRuns).toBe(0)
  })

  it("yields at least every YIELD_EVERY steps", async () => {
    const { learner } = stubLearner()
    const budget = 120
    let yields = 0
    const out = await runCheckup(base({ learner, suite: small(budget), yieldNow: async () => { yields++ } }))
    expect(out.status).toBe("done")
    expect(yields).toBeGreaterThanOrEqual(6 * Math.floor(budget / YIELD_EVERY))
    expect(yields).toBe(6 * budget) // passed straight through to recordRun: awaited every step
  })

  it("with a real CPU learner: save() equal before and after except stats.evalCU; a newborn passes nothing; cu unchanged", async () => {
    // A newborn is not a uniform-random policy, and newborns do pass M3–M6 by luck now and then (measured on 25:
    // see the task-4 report). This fixture (seed 21, id "nb-21") is one that passes nothing; M1 never passes (ratio 1).
    const l = await createLearner({ backend: createCpuBackend(), seed: 21, widths: TINY, settings: DEFAULT_SETTINGS })
    const before = await l.save()
    const out = await runCheckup({ learnerId: "nb-21", level: 8, steps: 0, learner: l, canvas })
    const after = await l.save()
    if (out.status !== "done") throw new Error("abandoned")
    expect({ ...after, stats: { ...after.stats, evalCU: 0 } }).toEqual({ ...before, stats: { ...before.stats, evalCU: 0 } })
    expect(after.stats.cu).toBe(before.stats.cu)
    expect(after.stats.evalCU).toBeGreaterThan(before.stats.evalCU)
    expect(out.result.evalCU).toBeCloseTo(after.stats.evalCU - before.stats.evalCU, 12)
    expect(out.result.passed).toEqual([])
    expect(out.result.error).toBeUndefined()
    expect(out.result.measures.M1).toBe(1) // a newborn predicts copy-last exactly
    expect(Object.keys(out.result.measures)).toEqual([...SUITE_IDS])
    // The learner is usable afterwards (no frozen run left open).
    const a: Action = await l.step(createWorld({ seed: 1, canvas: canvas() }).observe())
    expect(isValidAction(a)).toBe(true)
    l.dispose()
  })
})
