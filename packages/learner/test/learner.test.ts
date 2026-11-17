import { describe, it, expect } from "vitest"
import { BUTTON_OPTIONS, KEYS, MOVE_OPTIONS, SCROLL_OPTIONS, emptyObservation, isValidAction } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"
import { createCpuBackend, createRng } from "@blanktab/tensor"
import type { Backend, Storage } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, GAMMA, GAMMA_RANGE, LearnerStateVersionError, EXPLORE_EPSILON_RANGE, HEAD_LR_SCALE_RANGE, RESIDUAL_DECAY_RANGE, RESERVOIR_SHARE_RANGE, LEARNER_STATE_VERSION, clampSettings, createLearner, curiosityReward, LR_RANGE, sampleAction } from "../src/learner"
import type { HeldOutSet, LearnerSettings, LearnerState, UpdateInfo } from "../src/learner"
import { copyLastBaseline, targetsFromObs } from "../src/encode"
import type { Widths } from "../src/model"
import { createSyntheticStream } from "./synthetic"

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }

async function run(opts: {
  seed: number
  steps: number
  settings?: Partial<LearnerSettings>
  stream?: { observe(): Observation; act(a: Action): void }
  at?: (step: number, l: Awaited<ReturnType<typeof createLearner>>, be: ReturnType<typeof createCpuBackend>) => void | Promise<void>
  /** Sleep after every `sleepEvery`-th step (counting the learner's own steps, so a restored learner keeps the schedule). */
  sleepEvery?: number
  /** Resume from this checkpoint instead of starting fresh (seed and settings are then the state's). */
  state?: LearnerState
}) {
  const be = createCpuBackend()
  const updates: UpdateInfo[] = []
  const onUpdate = (u: UpdateInfo) => updates.push(u)
  const learner = opts.state
    ? await createLearner({ backend: be, state: opts.state, onUpdate })
    : await createLearner({ backend: be, seed: opts.seed, widths: TINY, settings: { ...DEFAULT_SETTINGS, ...opts.settings }, onUpdate })
  const world = opts.stream ?? createSyntheticStream()
  const actions: Action[] = []
  for (let i = 1; i <= opts.steps; i++) {
    const a = await learner.step(world.observe())
    actions.push(a)
    world.act(a)
    if (opts.sleepEvery && learner.stats().steps % opts.sleepEvery === 0) await learner.sleep()
    await opts.at?.(i, learner, be)
  }
  return { learner, be, actions, updates }
}

/**
 * Bitwise deep equality (typed arrays by bytes, numbers by Object.is), returning the first differing path or null.
 * vitest's toEqual walks typed arrays element by element and takes seconds on a full LearnerState.
 */
function firstDifference(a: unknown, b: unknown, path = "state"): string | null {
  if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) {
    if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b) || a.constructor !== b.constructor) return path
    const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength)
    const y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength)
    if (x.length !== y.length) return path
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return path
    return null
  }
  if (typeof a !== "object" || a === null || typeof b !== "object" || b === null) return Object.is(a, b) ? null : path
  if (Array.isArray(a) !== Array.isArray(b)) return path
  const ka = Object.keys(a)
  const kb = Object.keys(b)
  if (ka.length !== kb.length || ka.some((k) => !(k in b))) return `${path} (keys)`
  for (const k of ka) {
    const d = firstDifference((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`)
    if (d) return d
  }
  return null
}

const finiteParams = async (l: { readParams(): Promise<Float32Array[]> }) =>
  (await l.readParams()).every((p) => p.every(Number.isFinite))

describe("learner", { timeout: 30_000 }, () => { // every test builds a real learner: >= 30 s (controller ruling; ~1-8 s unloaded)
  // Several seeds must pass (task brief + controller rulings): after 3,000 steps the action-conditioned predictor
  // beats copy-last (predLoss < baselineLoss, both EMAs with the same alpha). Measured predLoss/baselineLoss at
  // 3,000 steps (the then-default lr 1e-3, headLrScale 0.1, widths 4,4,16,16): seed 1 0.883, seed 2 0.874, seed 3 0.786.
  // Still passing at the Plan 04a.2 defaults (lr 5e-4, exploreEpsilon 0.1, residualDecay 1e-3).
  for (const seed of [1, 2, 3]) {
    it(`learns to predict the next frame better than copy-last after 3000 steps (seed ${seed})`, async () => {
      const { learner } = await run({ seed, steps: 3000 })
      const s = learner.stats()
      expect(Number.isFinite(s.predLoss) && Number.isFinite(s.baselineLoss)).toBe(true)
      expect(s.predLoss).toBeLessThan(s.baselineLoss)
      expect(s.steps).toBe(3000)
      expect(s.skippedUpdates).toBe(0)
      learner.dispose()
    }, 90_000) // ~18-20 s each on an M-series CPU; generous for slower CI machines
  }

  it("is deterministic: same seed and stream give identical actions and losses", async () => {
    const a = await run({ seed: 9, steps: 120 })
    const b = await run({ seed: 9, steps: 120 })
    const c = await run({ seed: 10, steps: 120 })
    expect(b.actions).toEqual(a.actions)
    expect(b.updates).toEqual(a.updates)
    expect(b.learner.stats()).toEqual(a.learner.stats())
    expect(c.updates).not.toEqual(a.updates)
    for (const r of [a, b, c]) r.learner.dispose()
  }, 30_000) // ~2.7 s unloaded; timed out at the 5 s default under load (controller ruling: >= 2x, min 30 s)

  describe("ε-uniform exploration (exploreEpsilon)", () => {
    // Recorded from the learner before exploreEpsilon existed (commit 863adae): seed 5, TINY, DEFAULT_SETTINGS, the
    // synthetic stream, 40 steps with a sleep every 10. Actions as move.button.scroll.key; losses as an FNV-1a hash of
    // the 39 UpdateInfo.lossBefore values' float64 bytes.
    const GOLDEN_ACTIONS = "5.2.2.33 4.2.2.41 1.2.2.34 12.1.2.12 10.0.0.38 12.1.0.16 1.2.1.26 7.2.2.33 7.0.1.11 14.0.0.30 20.0.2.10 22.1.2.20 3.2.1.31 3.2.0.29 10.0.0.2 3.1.1.40 3.2.0.15 15.1.2.25 3.2.0.2 9.0.1.8 2.2.0.6 18.2.1.27 22.1.0.44 8.2.2.32 8.2.1.16 17.2.0.22 18.1.2.12 16.1.1.44 16.1.0.43 2.1.2.38 12.0.2.32 13.0.2.7 0.2.2.2 17.0.1.26 15.1.2.0 8.2.2.21 8.2.1.20 11.2.0.7 18.1.1.39 5.1.2.33"
    const GOLDEN_LOSS_HASH = "e40e135c"
    /** The settings of that recording, on top of today's DEFAULT_SETTINGS: no exploration, no residual decay. */
    const BEFORE = { lr: 1e-3, exploreEpsilon: 0, residualDecay: 0 }
    const fnv = (xs: number[]) => {
      let h = 0x811c9dc5
      for (const b of new Uint8Array(new Float64Array(xs).buffer)) { h ^= b; h = Math.imul(h, 0x01000193) >>> 0 }
      return h.toString(16)
    }
    const fmt = (as: Action[]) => as.map((a) => `${a.move}.${a.button}.${a.scroll}.${a.key}`).join(" ")

    it("defaults to 0.1 and is clamped into [0, 0.5]; a non-finite value is rejected", () => {
      expect(DEFAULT_SETTINGS.exploreEpsilon).toBe(0.1)
      expect(EXPLORE_EPSILON_RANGE).toEqual([0, 0.5])
      expect(clampSettings({ ...DEFAULT_SETTINGS, exploreEpsilon: 0.9 }).exploreEpsilon).toBe(0.5)
      expect(clampSettings({ ...DEFAULT_SETTINGS, exploreEpsilon: -1 }).exploreEpsilon).toBe(0)
      for (const v of [Number.NaN, Number.POSITIVE_INFINITY, undefined as unknown as number]) {
        expect(() => clampSettings({ ...DEFAULT_SETTINGS, exploreEpsilon: v })).toThrow(RangeError)
      }
    })

    it("ε = 0 and residualDecay = 0 (at the then-default lr 1e-3) reproduce the learner from before either existed, bit for bit", async () => {
      const r = await run({ seed: 5, steps: 40, sleepEvery: 10, settings: BEFORE })
      expect(fmt(r.actions)).toBe(GOLDEN_ACTIONS)
      expect(r.updates.length).toBe(39)
      expect(fnv(r.updates.map((u) => u.lossBefore))).toBe(GOLDEN_LOSS_HASH)
      r.learner.dispose()
    })

    it("ε > 0 stays deterministic per seed, keeps actions valid, and changes what is done", async () => {
      const a = await run({ seed: 5, steps: 80, sleepEvery: 10, settings: { exploreEpsilon: 0.3 } })
      const b = await run({ seed: 5, steps: 80, sleepEvery: 10, settings: { exploreEpsilon: 0.3 } })
      expect(b.actions).toEqual(a.actions)
      expect(b.updates).toEqual(a.updates)
      expect(a.actions.every(isValidAction)).toBe(true)
      expect(fmt(a.actions.slice(0, 40))).not.toBe(GOLDEN_ACTIONS)
      for (const r of [a, b]) r.learner.dispose()
    })

    it("ε > 0 survives save and load: the resumed learner explores exactly as one that never stopped", async () => {
      const whole = await run({ seed: 6, steps: 60, settings: { exploreEpsilon: 0.2 } })
      const first = await run({ seed: 6, steps: 30, settings: { exploreEpsilon: 0.2 } })
      const state = await first.learner.save()
      first.learner.dispose()
      expect(state.settings.exploreEpsilon).toBe(0.2)
      const be = createCpuBackend()
      const resumed = await createLearner({ backend: be, state })
      const world = createSyntheticStream()
      for (const a of first.actions) { world.observe(); world.act(a) }
      const rest: Action[] = []
      for (let i = 0; i < 30; i++) { const a = await resumed.step(world.observe()); rest.push(a); world.act(a) }
      expect(rest).toEqual(whole.actions.slice(30))
      resumed.dispose()
      whole.learner.dispose()
    })

    it("residualDecay: defaults to 1e-3 (with lr 5e-4), clamped into [0, 1e-2]; a non-finite value is rejected", () => {
      expect(DEFAULT_SETTINGS.residualDecay).toBe(1e-3)
      expect(DEFAULT_SETTINGS.lr).toBe(5e-4)
      expect(RESIDUAL_DECAY_RANGE).toEqual([0, 1e-2])
      expect(clampSettings({ ...DEFAULT_SETTINGS, residualDecay: 0.5 }).residualDecay).toBe(1e-2)
      expect(clampSettings({ ...DEFAULT_SETTINGS, residualDecay: -1 }).residualDecay).toBe(0)
      for (const v of [Number.NaN, undefined as unknown as number]) expect(() => clampSettings({ ...DEFAULT_SETTINGS, residualDecay: v })).toThrow(RangeError)
    })

    it("residualDecay > 0 pulls the residual layers (shift gate, output layers) toward zero, deterministically", async () => {
      const resid = async (l: { readParams(): Promise<Float32Array[]> }) => {
        const ps = await l.readParams()
        const n = ps.length
        return [ps[n - 9]!, ...ps.slice(-4)].reduce((a, p) => a + p.reduce((b, v) => b + v * v, 0), 0)
      }
      const off = await run({ seed: 5, steps: 60, sleepEvery: 10, settings: { exploreEpsilon: 0, residualDecay: 0 } })
      const on = await run({ seed: 5, steps: 60, sleepEvery: 10, settings: { exploreEpsilon: 0, residualDecay: 1e-2 } })
      const on2 = await run({ seed: 5, steps: 60, sleepEvery: 10, settings: { exploreEpsilon: 0, residualDecay: 1e-2 } })
      expect(await resid(on.learner)).toBeLessThan(await resid(off.learner))
      expect(on2.updates).toEqual(on.updates)
      for (const r of [off, on, on2]) r.learner.dispose()
    })

    it("sampleAction: with probability ε a factor is uniform, otherwise it follows the policy", () => {
      const sizes = [MOVE_OPTIONS, BUTTON_OPTIONS, SCROLL_OPTIONS, KEYS.length]
      const logits = new Float32Array(sizes.reduce((a, n) => a + n, 0)).fill(-100)
      let off = 0
      for (const n of sizes) { logits[off] = 100; off += n } // the policy always picks index 0 of every factor
      const draws = 20_000
      for (const eps of [0, 0.2]) {
        const rng = createRng(11)
        const off0 = [0, 0, 0, 0]
        for (let i = 0; i < draws; i++) {
          const a = sampleAction(logits, eps, rng)
          expect(isValidAction(a)).toBe(true)
          ;[a.move, a.button, a.scroll, a.key].forEach((v, f) => { if (v !== 0) off0[f]!++ })
        }
        sizes.forEach((n, f) => {
          const want = eps * (n - 1) / n // explored, and the uniform draw missed index 0
          expect(Math.abs(off0[f]! / draws - want)).toBeLessThan(eps === 0 ? 1e-12 : 0.012)
        })
      }
    })
  })

  it("returns only valid actions, counts steps and compute", async () => {
    const { learner, actions, updates } = await run({ seed: 4, steps: 200 })
    expect(actions.every(isValidAction)).toBe(true)
    // Not degenerate: the policy samples more than one move.
    expect(new Set(actions.map((a) => a.move)).size).toBeGreaterThan(1)
    expect(updates.length).toBe(199) // no transition to learn from on the first step
    const s = learner.stats()
    expect(s.steps).toBe(200)
    expect(s.cu).toBeGreaterThan(0)
    expect(Number.isFinite(s.predLoss) && Number.isFinite(s.baselineLoss)).toBe(true)
    expect(s.sleeps).toBe(0)
    expect(s.consecutiveSkips).toBe(0)
    learner.dispose()
  })

  it("keeps liveBuffers flat between step 100 and step 600", async () => {
    let at100 = -1
    const { learner, be } = await run({ seed: 5, steps: 600, at: (i, _l, b) => { if (i === 100) at100 = b.stats.liveBuffers } })
    expect(at100).toBeGreaterThan(0)
    expect(be.stats.liveBuffers).toBe(at100)
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  }, 30_000)

  it("Review Focus 4: an all-black stream with the cursor in a corner keeps losses finite and weights NaN-free", async () => {
    for (const corner of [{ x: 0, y: 0 }, { x: 1, y: 1 }]) {
      const black = { observe: () => ({ ...emptyObservation(), cursor: { ...corner } }), act: () => {} }
      const { learner, updates } = await run({ seed: 6, steps: 150, stream: black })
      expect(updates.every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.lossAfter) && Number.isFinite(u.reward))).toBe(true)
      expect(Number.isFinite(learner.stats().predLoss)).toBe(true)
      expect(learner.stats().skippedUpdates).toBe(0)
      expect(await finiteParams(learner)).toBe(true)
      learner.dispose()
    }
  })

  it("Review Focus 4: a corrupt cursor (NaN / Infinity) never reaches the weights", async () => {
    const s = createSyntheticStream()
    const bad = { observe: () => ({ ...s.observe(), cursor: { x: Number.NaN, y: Number.POSITIVE_INFINITY } }), act: s.act }
    const { learner } = await run({ seed: 6, steps: 50, stream: bad })
    expect(Number.isFinite(learner.stats().predLoss)).toBe(true)
    expect(await finiteParams(learner)).toBe(true)
    learner.dispose()
  })

  it("Review Focus 5: lr = 1 is clamped and the run stays finite", async () => {
    expect(clampSettings({ ...DEFAULT_SETTINGS, lr: 1 }).lr).toBe(LR_RANGE[1])
    expect(clampSettings({ ...DEFAULT_SETTINGS, lr: 1e-9 }).lr).toBe(LR_RANGE[0])
    expect(LR_RANGE).toEqual([1e-5, 1e-2])
    expect(DEFAULT_SETTINGS.headLrScale).toBe(0.1)
    expect(HEAD_LR_SCALE_RANGE).toEqual([0.01, 1])
    expect(clampSettings({ ...DEFAULT_SETTINGS, headLrScale: 5 }).headLrScale).toBe(1)
    expect(clampSettings({ ...DEFAULT_SETTINGS, headLrScale: 0 }).headLrScale).toBe(0.01)
    const { learner, updates } = await run({ seed: 7, steps: 200, settings: { lr: 1 } })
    expect(updates.every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.lossAfter))).toBe(true)
    expect(await finiteParams(learner)).toBe(true)
    learner.dispose()
  })

  it("Review Focus 5: the extremes of entropyWeight and curiosityWeight run without NaN", async () => {
    // Ranges from the plan (Task 7 writes them into knowledge/settings-space.json): entropyWeight [0.001, 0.05],
    // curiosityWeight [0.5, 2]. Also beyond them (0 and 10x), since a setting outside the space must not NaN either.
    for (const entropyWeight of [0, 0.001, 0.05, 0.5]) {
      for (const curiosityWeight of [0, 0.5, 2, 20]) {
        const { learner, updates } = await run({ seed: 8, steps: 60, settings: { entropyWeight, curiosityWeight } })
        expect(updates.every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.advantage))).toBe(true)
        expect(learner.stats().skippedUpdates).toBe(0)
        expect(await finiteParams(learner)).toBe(true)
        learner.dispose()
      }
    }
  }, 30_000)

  it("rejects settings that cannot be clamped", () => {
    for (const bad of [{ lr: Number.NaN }, { entropyWeight: -1 }, { curiosityWeight: Number.POSITIVE_INFINITY }, { bufferSize: 0 }, { bufferSize: 2.5 }, { replayPerSleep: -3 }, { headLrScale: Number.NaN }]) {
      expect(() => clampSettings({ ...DEFAULT_SETTINGS, ...bad })).toThrow(RangeError)
    }
  })

  it("the curiosity reward is relative learning progress, finite and clamped to [-1, 1]", async () => {
    expect(curiosityReward(0.5, 0.25, 1)).toBeCloseTo(0.25)
    expect(curiosityReward(0.011, 0.010, 0.01)).toBeCloseTo(0.1) // relative to the running loss scale
    expect(curiosityReward(5, 0, 1)).toBe(1)
    expect(curiosityReward(0, 5, 1)).toBe(-1)
    expect(curiosityReward(2e-6, 1e-6, 0)).toBeCloseTo(1) // scale floored at 1e-6
    for (const [b, a, sc] of [[Number.NaN, 0, 1], [0, Number.NaN, 1], [Number.POSITIVE_INFINITY, 0, 1], [1, 0, Number.NaN]] as const) {
      expect(curiosityReward(b, a, sc)).toBe(0)
    }
    const { learner, updates } = await run({ seed: 3, steps: 200 })
    expect(updates.length).toBeGreaterThan(0)
    for (const u of updates) {
      expect(Number.isFinite(u.reward)).toBe(true)
      expect(Math.abs(u.reward)).toBeLessThanOrEqual(1)
    }
    learner.dispose()
  })

  it("scales curiosity by the predLoss EMA from before this step's loss (no self-inclusion)", async () => {
    const { learner, updates } = await run({ seed: 5, steps: 60 })
    let ema = Number.NaN
    for (const u of updates) {
      expect(u.reward).toBe(curiosityReward(u.lossBefore, u.lossAfter, ema)) // the first update has no scale yet: 0
      ema = Number.isFinite(ema) ? ema + 0.01 * (u.lossBefore - ema) : u.lossBefore
    }
    expect(updates[0]!.reward).toBe(0)
    learner.dispose()
  })

  it("trains the residual output layers at lr x headLrScale and every other parameter at lr", async () => {
    // Adam's first step moves each parameter with a non-zero gradient by ~lr (m/sqrt(v) = sign(g)).
    const learner = await createLearner({ backend: createCpuBackend(), seed: 2, widths: TINY, settings: { ...DEFAULT_SETTINGS, lr: 1e-3, headLrScale: 0.1 } })
    const world = createSyntheticStream()
    world.act(await learner.step(world.observe()))
    const p0 = await learner.readParams()
    world.act(await learner.step(world.observe())) // first update
    const p1 = await learner.readParams()
    const maxStep = (i: number) => p1[i]!.reduce((m, v, k) => Math.max(m, Math.abs(v - p0[i]![k]!)), 0)
    const n = p0.length
    for (const i of [n - 4, n - 3, n - 2, n - 1]) expect(maxStep(i)).toBeLessThan(1.01e-4) // headP w/b, headF w/b
    // headP sees the block move; headF's target is unchanged on this step (block centred, plain fovea): zero gradient.
    for (const i of [n - 4, n - 3]) expect(maxStep(i)).toBeGreaterThan(0.9e-4)
    // On this first update the zero output layers block every gradient upstream of them (core, predictor h/action
    // layers get exactly 0), so compare with the policy and value heads, which are on the main optimizer.
    for (const i of [n - 15, n - 14, n - 13, n - 12]) { // policy w/b, value w/b (then shiftX, shiftY, shiftGate, predH, predA)
      expect(maxStep(i)).toBeGreaterThan(0.9e-3)
      expect(maxStep(i)).toBeLessThan(1.01e-3)
    }
    learner.dispose()
  })

  it("on a fixed stream, curiosity is on average positive early", async () => {
    // Fixed = independent of the learner's actions: the block ping-pongs 4 px per step (E for 40 steps, then W).
    const sweep = createSyntheticStream({ start: { x: 20, y: 100 } })
    let k = 0
    const fixed = { observe: () => { sweep.act({ move: Math.floor(k++ / 40) % 2 === 0 ? 2 : 14, button: 0, scroll: 0, key: 0 }); return sweep.observe() }, act: () => {} }
    const { learner, updates } = await run({ seed: 4, steps: 101, stream: fixed })
    const mean = updates.reduce((a, u) => a + u.reward, 0) / updates.length
    expect(updates.length).toBe(100)
    expect(mean).toBeGreaterThan(0)
    learner.dispose()
  })

  it("a non-finite loss skips the update: no throw, skippedUpdates + 1, every other parameter bitwise unchanged", async () => {
    // White-box: record CPU storages as they are allocated; the model's parameters come first, in Model.params order.
    const inner = createCpuBackend()
    const allocated: Storage[] = []
    const be: Backend = {
      ...inner,
      upload: (d) => { const st = inner.upload(d); allocated.push(st); return st },
      zeros: (n) => { const st = inner.zeros(n); allocated.push(st); return st },
    }
    const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    const world = createSyntheticStream()
    for (let i = 0; i < 5; i++) world.act(await learner.step(world.observe()))
    const before = await learner.readParams()
    // Poison the periphery output-layer bias (Model.params index 35: ..., predA w/b, headP w (34), headP b (35), headF w/b).
    const POISONED = 35
    expect(allocated[POISONED]!.length).toBe(3000)
    ;(allocated[POISONED] as unknown as { data: Float32Array }).data[7] = Number.NaN
    const skipped = learner.stats().skippedUpdates
    const a = await learner.step(world.observe())
    expect(isValidAction(a)).toBe(true)
    expect(learner.stats().skippedUpdates).toBe(skipped + 1)
    expect(learner.stats().consecutiveSkips).toBe(1)
    const after = await learner.readParams()
    after.forEach((p, i) => {
      if (i === POISONED) return
      expect(p).toEqual(before[i])
    })
    learner.dispose()
  })

  it("replay buffer holds the last bufferSize transitions as raw frames", async () => {
    const { learner } = await run({ seed: 2, steps: 30, settings: { bufferSize: 10 } })
    const buf = learner.replay
    expect(buf.size).toBe(10)
    const t = buf.at(buf.size - 1)
    expect(t.obs.periphery).toBeInstanceOf(Uint8Array)
    expect(t.next.fovea).toBeInstanceOf(Uint8Array)
    expect(t.hIn.length).toBe(TINY.hidden)
    // Consecutive transitions share the frame between them: next of one is obs of the following.
    expect(buf.at(buf.size - 2).next).toBe(t.obs)
    learner.dispose()
  })

  it("rejects a malformed frame without counting a step or leaking", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    await learner.step(emptyObservation())
    const live = be.stats.liveBuffers
    await expect(learner.step({ ...emptyObservation(), fovea: new Uint8Array(5) })).rejects.toThrow(RangeError)
    expect(learner.stats().steps).toBe(1)
    expect(be.stats.liveBuffers).toBe(live)
    await learner.step(emptyObservation())
    expect(learner.stats().steps).toBe(2)
    learner.dispose()
  })

  it("rejects overlapping step calls", async () => {
    const learner = await createLearner({ backend: createCpuBackend(), seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    const first = learner.step(emptyObservation())
    await expect(learner.step(emptyObservation())).rejects.toThrow(/in progress/)
    await first
    learner.dispose()
  })

  describe("sleep, save and load", { timeout: 30_000 }, () => {
    /** The stream is part of the run: a resumed learner continues against the same stream object. */
    it("save at step 300 (sleeps every 100), load into a fresh learner, 200 more steps = an uninterrupted 500-step run", async () => {
      const full = await run({ seed: 11, steps: 500, sleepEvery: 100 })
      expect(full.learner.stats().sleeps).toBe(5)

      const world = createSyntheticStream()
      const first = await run({ seed: 11, steps: 300, sleepEvery: 100, stream: world })
      const state = await first.learner.save()
      first.learner.dispose()
      expect(first.be.stats.liveBuffers).toBe(0)
      const second = await run({ seed: -1, steps: 200, sleepEvery: 100, stream: world, state })

      expect([...first.actions, ...second.actions]).toEqual(full.actions)
      expect([...first.updates, ...second.updates]).toEqual(full.updates)
      expect(second.learner.stats()).toEqual(full.learner.stats())
      expect(await second.learner.readParams()).toEqual(await full.learner.readParams())
      expect(firstDifference(await second.learner.save(), await full.learner.save())).toBeNull()
      full.learner.dispose()
      second.learner.dispose()
    }, 30_000) // ~7.6 s unloaded (controller ruling: >= 2x, min 30 s)

    it("restores a learner saved before its first step and one saved after a single step", async () => {
      for (const n of [0, 1]) {
        const full = await run({ seed: 12, steps: 20 })
        const world = createSyntheticStream()
        const first = await run({ seed: 12, steps: n, stream: world })
        const second = await run({ seed: -1, steps: 20 - n, stream: world, state: await first.learner.save() })
        expect([...first.actions, ...second.actions]).toEqual(full.actions)
        expect([...first.updates, ...second.updates]).toEqual(full.updates)
        for (const r of [full, first, second]) r.learner.dispose()
      }
    })

    it("LearnerState survives structuredClone, is a snapshot, and a cloned state resumes identically", async () => {
      const world = createSyntheticStream()
      const { learner } = await run({ seed: 13, steps: 40, sleepEvery: 20, stream: world, settings: { bufferSize: 16 } })
      const state = await learner.save()
      const clone = structuredClone(state)
      expect(firstDifference(clone, state)).toBeNull()
      const tweaked = structuredClone(state) // the comparison itself sees a one-float change
      tweaked.adam[1].slots[3]!.v[0]! += 1e-30
      expect(firstDifference(tweaked, state)).toBe("state.adam.1.slots.3.v")
      expect(state.params[0]).toBeInstanceOf(Float32Array)
      expect(state.replay.length).toBe(16)
      expect(state.version).toBe(LEARNER_STATE_VERSION)
      expect(state.seed).toBe(13)
      expect(state.widths).toEqual(TINY)
      // A snapshot: stepping on does not change it, and loading from it does not alias it.
      const frozen = structuredClone(state)
      world.act(await learner.step(world.observe()))
      await learner.sleep()
      expect(firstDifference(state, frozen)).toBeNull()
      const w2 = createSyntheticStream()
      const w3 = createSyntheticStream()
      const a = await run({ seed: -1, steps: 30, sleepEvery: 10, stream: w2, state })
      const b = await run({ seed: -1, steps: 30, sleepEvery: 10, stream: w3, state: clone })
      expect(firstDifference(state, frozen)).toBeNull()
      expect(b.actions).toEqual(a.actions)
      expect(b.updates).toEqual(a.updates)
      for (const r of [a, b]) r.learner.dispose()
      learner.dispose()
    }, 30_000) // ~2.8 s unloaded; timed out at the 5 s default under load (controller ruling)

    it("rejects a state that does not fit, before allocating anything", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      const bad: LearnerState[] = [
        { ...state, version: (LEARNER_STATE_VERSION + 1) as typeof LEARNER_STATE_VERSION },
        { ...state, params: state.params.slice(1) },
        { ...state, widths: { ...TINY, hidden: 8 } },
        { ...state, h: new Float32Array(3) },
        { ...state, adam: state.adam.slice(1) as unknown as LearnerState["adam"] },
        { ...state, settings: { ...state.settings, bufferSize: 2 } }, // fewer slots than the 4 stored transitions
      ]
      for (const s of bad) await expect(createLearner({ backend: be, state: s })).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("a state of another version is a LearnerStateVersionError (a RangeError), whatever else is in it, telling which version it is", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      for (const [s, found] of [[{ ...state, version: 1 }, 1], [{ version: 1 }, 1], [{ version: 3, seed: "x" }, 3], [{ version: "2" }, "2"]] as const) {
        const e = await createLearner({ backend: be, state: s as unknown as LearnerState }).catch((x: unknown) => x)
        expect(e).toBeInstanceOf(LearnerStateVersionError)
        expect(e).toBeInstanceOf(RangeError)
        expect((e as LearnerStateVersionError).found).toBe(found)
        expect((e as LearnerStateVersionError).older).toBe(found === 1)
      }
      const other = await createLearner({ backend: be, state: { ...state, seed: -1 } }).catch((x: unknown) => x)
      expect(other).toBeInstanceOf(RangeError)
      expect(other).not.toBeInstanceOf(LearnerStateVersionError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("rejects non-finite parameters, Adam moments and recurrent state with RangeError, allocating nothing", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      const poison = (mutate: (s: LearnerState) => void, v: number) => {
        const s = structuredClone(state)
        mutate(s)
        return { s, v }
      }
      const cases = [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY].flatMap(v => [
        poison(s => { s.params[3]![1] = v }, v),
        poison(s => { s.adam[0].slots[2]!.m[0] = v }, v),
        poison(s => { s.adam[1].slots[1]!.v[4] = v }, v),
        poison(s => { s.h[5] = v }, v),
        poison(s => { s.last!.hIn[0] = v }, v),
        poison(s => { s.replay[1]!.hIn[2] = v }, v),
      ])
      for (const { s } of cases) await expect(createLearner({ backend: be, state: s })).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("turns missing or malformed top-level fields into RangeError (not TypeError)", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      const keys = ["version", "seed", "widths", "settings", "params", "adam", "rng", "h", "last", "replay", "stats", "heldOut"] as const
      for (const k of keys) {
        const s = { ...state } as Partial<LearnerState>
        delete s[k]
        await expect(createLearner({ backend: be, state: s as LearnerState }), `without ${k}`).rejects.toThrow(RangeError)
      }
      const odd: unknown[] = [null, 7, "x", [], { ...state, adam: [null, null] }, { ...state, stats: null }, { ...state, widths: null }, { ...state, params: [null] }, { ...state, last: { obs: null } }, { ...state, replay: [null] }]
      for (const s of odd) await expect(createLearner({ backend: be, state: s as LearnerState })).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("a restore that fails part-way (the device refuses an allocation) frees everything it had allocated", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      // Count the allocations of a successful restore, then fail at each one in turn. The backend is patched in
      // place: a tensor's backend is its storage's, so a wrapper object would miss the model's and Adam's allocations.
      const refusing = (failAt: number) => {
        const be = createCpuBackend()
        const { upload, zeros } = be
        let k = 0
        const take = () => { if (++k === failAt) throw new Error("device lost") }
        Object.assign(be, { upload: (d: Float32Array) => { take(); return upload(d) }, zeros: (m: number) => { take(); return zeros(m) } })
        return { be, count: () => k }
      }
      const counting = refusing(0)
      ;(await createLearner({ backend: counting.be, state })).dispose()
      const n = counting.count()
      expect(n).toBeGreaterThan(50)
      for (let failAt = 1; failAt <= n; failAt++) {
        const { be: inner } = refusing(failAt)
        await expect(createLearner({ backend: inner, state }), `failing at allocation ${failAt}`).rejects.toThrow("device lost")
        expect(inner.stats.liveBuffers, `failing at allocation ${failAt}`).toBe(0)
      }
    })

    it("sleep() replays replayPerSleep uniformly sampled transitions and lowers the mean loss on the buffer", async () => {
      // Fixed stream, independent of the actions: the block sweeps E then W, 4 px per step.
      const sweep = createSyntheticStream({ start: { x: 20, y: 100 } })
      let k = 0
      const fixed = { observe: () => { sweep.act({ move: Math.floor(k++ / 40) % 2 === 0 ? 2 : 14, button: 0, scroll: 0, key: 0 }); return sweep.observe() }, act: () => {} }
      const { learner, updates } = await run({ seed: 15, steps: 120, stream: fixed, settings: { bufferSize: 100, replayPerSleep: 200 } })
      const s0 = learner.stats()
      const before = await learner.replayLoss()
      expect(learner.stats()).toEqual(s0) // measuring is not learning and not billed
      const p0 = await learner.readParams()
      await learner.sleep()
      const after = await learner.replayLoss()
      const s1 = learner.stats()
      expect(Number.isFinite(before) && Number.isFinite(after)).toBe(true)
      expect(after).toBeLessThan(before)
      expect(s1.sleeps).toBe(1)
      expect(s1.steps).toBe(s0.steps)
      expect(s1.cu).toBeGreaterThan(s0.cu)
      expect(s1.skippedUpdates).toBe(0)
      expect(updates.length).toBe(119) // sleep does not report online updates
      // Prediction only: the policy and value heads are untouched by replay.
      const p1 = await learner.readParams()
      const n = p0.length
      for (const i of [n - 12, n - 11, n - 10, n - 9]) expect(p1[i]).toEqual(p0[i])
      expect(p1[0]).not.toEqual(p0[0])
      learner.dispose()
    })

    it("sleep() on an empty buffer is a counted no-op, and sleep() cannot overlap a step", async () => {
      const be = createCpuBackend()
      const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
      const live = be.stats.liveBuffers
      await learner.sleep()
      expect(learner.stats().sleeps).toBe(1)
      expect(be.stats.liveBuffers).toBe(live)
      const step = learner.step(emptyObservation())
      const [sl, sv, rl] = [learner.sleep(), learner.save(), learner.replayLoss()] // all started while the step runs
      for (const p of [sl, sv, rl]) await expect(p).rejects.toThrow(/in progress/)
      await step
      expect(learner.stats().sleeps).toBe(1)
      learner.dispose()
      await expect(learner.sleep()).rejects.toThrow(/dispose/)
    })

    it("Review Focus 1: after 2000 steps with a sleep every 250, liveBuffers equals its value after the first sleep", async () => {
      let afterFirstSleep = -1
      const { learner, be } = await run({
        seed: 16, steps: 2000, sleepEvery: 250,
        at: (i, _l, b) => { if (i === 250) afterFirstSleep = b.stats.liveBuffers },
      })
      expect(learner.stats().sleeps).toBe(8)
      expect(afterFirstSleep).toBeGreaterThan(0)
      expect(be.stats.liveBuffers).toBe(afterFirstSleep)
      learner.dispose()
      expect(be.stats.liveBuffers).toBe(0)
    }, 60_000) // ~26-28 s on a loaded M-series CPU (controller ruling: 60 s)

    it("consecutiveSkips counts skipped updates in a row, and restoring the last checkpoint clears it", async () => {
      // Same white-box poisoning as the skip test above: a NaN in headP's bias makes every prediction loss non-finite.
      const inner = createCpuBackend()
      const allocated: Storage[] = []
      const be: Backend = {
        ...inner,
        upload: (d) => { const st = inner.upload(d); allocated.push(st); return st },
        zeros: (n) => { const st = inner.zeros(n); allocated.push(st); return st },
      }
      const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
      const world = createSyntheticStream()
      for (let i = 0; i < 5; i++) world.act(await learner.step(world.observe()))
      const checkpoint = await learner.save()
      ;(allocated[32] as unknown as { data: Float32Array }).data[7] = Number.NaN
      for (let i = 1; i <= 4; i++) {
        world.act(await learner.step(world.observe()))
        expect(learner.stats().consecutiveSkips).toBe(i)
      }
      await learner.sleep() // every replayed update is skipped too
      expect(learner.stats().consecutiveSkips).toBe(4 + DEFAULT_SETTINGS.replayPerSleep)
      expect(learner.stats().skippedUpdates).toBe(4 + DEFAULT_SETTINGS.replayPerSleep)
      learner.dispose()
      // The harness's response: restore from the last checkpoint and carry on.
      const restored = await createLearner({ backend: createCpuBackend(), state: checkpoint })
      expect(restored.stats().consecutiveSkips).toBe(0)
      world.act(await restored.step(world.observe()))
      expect(restored.stats().consecutiveSkips).toBe(0)
      expect(restored.stats().skippedUpdates).toBe(0)
      restored.dispose()
    })
  })
})

/** A held-out trajectory from the synthetic stream under uniformly random actions (the worker builds the real one from the world). */
function syntheticHeldOut(seed: number, n: number): HeldOutSet {
  const w = createSyntheticStream({ start: { x: 60, y: 60 } })
  const rnd = createRng(seed)
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const obs = [w.observe()]
  const actions: Action[] = []
  for (let i = 0; i < n; i++) {
    const a = { move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS), key: pick(KEYS.length) }
    w.act(a)
    actions.push(a)
    obs.push(w.observe())
  }
  return { obs, actions }
}
const copyLastOn = (set: HeldOutSet) =>
  set.actions.reduce((s, _, i) => s + copyLastBaseline(targetsFromObs(set.obs[i]!), targetsFromObs(set.obs[i + 1]!)), 0) / set.actions.length

describe("the held-out check (decision 0015 §3)", { timeout: 60_000 }, () => {
  const SET = syntheticHeldOut(7, 16)

  it("a newborn scores exactly copy-last on it; nothing is measured before the first sleep", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    const s0 = learner.stats()
    expect(Number.isNaN(s0.heldOutLoss) && Number.isNaN(s0.heldOutBaseline)).toBe(true)
    expect(s0.evalCU).toBe(0)
    await learner.sleep() // empty buffer: no training, but the held-out set is scored
    const s1 = learner.stats()
    expect(s1.heldOutBaseline).toBeCloseTo(copyLastOn(SET), 9)
    expect(s1.heldOutBaseline).toBeGreaterThan(0)
    expect(Math.abs(s1.heldOutLoss / s1.heldOutBaseline - 1)).toBeLessThan(1e-5) // newborn = copy-last
    expect(s1.evalCU).toBeGreaterThan(0)
    expect(s1.cu).toBe(0) // the scoring is not billed to training
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("never trains: actions, updates, weights, RNG and cu are bit-identical to a learner without it; evalCU grows per sleep", async () => {
    const plain = await run({ seed: 21, steps: 60, sleepEvery: 20 })
    const be = createCpuBackend()
    const updates: UpdateInfo[] = []
    const learner = await createLearner({ backend: be, seed: 21, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET, onUpdate: u => updates.push(u) })
    const world = createSyntheticStream()
    const actions: Action[] = []
    const evals: number[] = []
    const losses: number[] = []
    for (let i = 1; i <= 60; i++) {
      const a = await learner.step(world.observe())
      actions.push(a)
      world.act(a)
      if (i % 20 === 0) {
        await learner.sleep()
        evals.push(learner.stats().evalCU)
        losses.push(learner.stats().heldOutLoss)
      }
    }
    expect(actions).toEqual(plain.actions)
    expect(updates).toEqual(plain.updates)
    expect(await learner.readParams()).toEqual(await plain.learner.readParams())
    const { evalCU, heldOutLoss, heldOutBaseline, ...rest } = learner.stats()
    const { evalCU: e0, heldOutLoss: l0, heldOutBaseline: b0, ...plainRest } = plain.learner.stats()
    expect(rest).toEqual(plainRest) // incl. cu: bit-identical
    expect(e0).toBe(0)
    expect(Number.isNaN(l0) && Number.isNaN(b0)).toBe(true)
    // Each sleep scores the same set the same way: the same FLOPs each time.
    expect(evals[0]).toBeGreaterThan(0)
    expect(evals[1]! - evals[0]!).toBeCloseTo(evals[0]!, 12)
    expect(evals[2]! - evals[1]!).toBeCloseTo(evals[0]!, 12)
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(heldOutLoss).toBe(losses[2])
    expect(heldOutBaseline).toBeCloseTo(copyLastOn(SET), 9)
    expect(evalCU).toBe(evals[2])
    learner.dispose()
    plain.learner.dispose()
  })

  it("is fixed for the learner's life: saved, restored bit-exactly, and an own copy (the caller's arrays can change)", async () => {
    const mine = structuredClone(SET)
    const full = await createLearner({ backend: createCpuBackend(), seed: 22, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: mine })
    mine.obs[3]!.fovea.fill(0) // the learner holds its own copy
    mine.actions[0]!.move = 0
    const w1 = createSyntheticStream()
    for (let i = 1; i <= 30; i++) { w1.act(await full.step(w1.observe())); if (i % 10 === 0) await full.sleep() }

    const w2 = createSyntheticStream()
    const first = await createLearner({ backend: createCpuBackend(), seed: 22, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    for (let i = 1; i <= 20; i++) { w2.act(await first.step(w2.observe())); if (i % 10 === 0) await first.sleep() }
    const state = await first.save()
    first.dispose()
    expect(firstDifference(state.heldOut, SET)).toBeNull()
    const second = await createLearner({ backend: createCpuBackend(), state: structuredClone(state) })
    for (let i = 21; i <= 30; i++) { w2.act(await second.step(w2.observe())); if (i % 10 === 0) await second.sleep() }
    expect(second.stats()).toEqual(full.stats())
    expect(firstDifference(await second.save(), await full.save())).toBeNull()
    full.dispose()
    second.dispose()
  })

  it("a learner born without one saves heldOut null and measures nothing", async () => {
    const { learner } = await run({ seed: 23, steps: 10, sleepEvery: 5 })
    const s = learner.stats()
    expect(s.evalCU).toBe(0)
    expect(Number.isNaN(s.heldOutLoss)).toBe(true)
    expect((await learner.save()).heldOut).toBeNull()
    learner.dispose()
  })

  it("rejects a malformed held-out set (newborn or in a state) with RangeError, allocating nothing", async () => {
    const be = createCpuBackend()
    const badSets: unknown[] = [
      { obs: SET.obs, actions: SET.actions.slice(1) }, // obs must be one more than actions
      { obs: [SET.obs[0]], actions: [] }, // at least one transition
      { obs: SET.obs.map((o, i) => (i === 2 ? { ...o, fovea: new Uint8Array(5) } : o)), actions: SET.actions },
      { obs: SET.obs, actions: SET.actions.map((a, i) => (i === 1 ? { ...a, move: 999 } : a)) },
      { obs: SET.obs },
      null, 7,
    ]
    for (const h of badSets) {
      await expect(createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: h as HeldOutSet })).rejects.toThrow(RangeError)
    }
    const l = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    const state = await l.save()
    l.dispose()
    for (const h of badSets.filter(x => x !== null)) {
      await expect(createLearner({ backend: be, state: { ...state, heldOut: h as HeldOutSet } })).rejects.toThrow(RangeError)
    }
    const { heldOut: _drop, ...without } = state
    void _drop
    await expect(createLearner({ backend: be, state: without as LearnerState })).rejects.toThrow(RangeError)
    for (const k of ["evalCU", "heldOutLoss", "heldOutBaseline"] as const) {
      const { [k]: _gone, ...st } = state.stats
      void _gone
      await expect(createLearner({ backend: be, state: { ...state, stats: st as LearnerState["stats"] } }), `stats without ${k}`).rejects.toThrow(RangeError)
    }
    for (const bad of [{ evalCU: -1 }, { evalCU: Number.NaN }, { heldOutLoss: Number.POSITIVE_INFINITY }, { heldOutBaseline: -0.1 }, { heldOutLoss: "0.1" }]) {
      await expect(createLearner({ backend: be, state: { ...state, stats: { ...state.stats, ...bad } as LearnerState["stats"] } }), JSON.stringify(bad)).rejects.toThrow(RangeError)
    }
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("a sleep that fails part-way (the device refuses an allocation) leaks nothing: replay and scoring free what they made", async () => {
    const make = async (failAt: number) => {
      const be = createCpuBackend()
      const { upload, zeros } = be
      let k = 0, armed = false
      const take = () => { if (armed && ++k === failAt) throw new Error("device lost") }
      Object.assign(be, { upload: (d: Float32Array) => { take(); return upload(d) }, zeros: (m: number) => { take(); return zeros(m) } })
      const l = await createLearner({ backend: be, seed: 25, widths: TINY, settings: { ...DEFAULT_SETTINGS, replayPerSleep: 2 }, heldOut: syntheticHeldOut(3, 3) })
      const w = createSyntheticStream()
      for (let i = 0; i < 4; i++) w.act(await l.step(w.observe()))
      const before = be.stats.liveBuffers
      armed = true
      return { be, l, before, count: () => k }
    }
    const probe = await make(0)
    await probe.l.sleep()
    const n = probe.count()
    expect(n).toBeGreaterThan(20)
    probe.l.dispose()
    for (let failAt = 1; failAt <= n; failAt++) {
      const { be, l, before } = await make(failAt)
      await expect(l.sleep(), `failing at allocation ${failAt}`).rejects.toThrow("device lost")
      expect(be.stats.liveBuffers, `failing at allocation ${failAt}`).toBe(before)
      l.dispose()
      expect(be.stats.liveBuffers).toBe(0)
    }
  })

  it("keeps liveBuffers flat across sleeps that score it", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 24, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    const w = createSyntheticStream()
    let after = -1
    for (let i = 1; i <= 40; i++) {
      w.act(await learner.step(w.observe()))
      if (i % 10 === 0) await learner.sleep()
      if (i === 10) after = be.stats.liveBuffers
    }
    expect(be.stats.liveBuffers).toBe(after)
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })
})

/** A fixed observation stream (independent of anyone's actions): the synthetic world under seeded random actions. */
function stream(n: number, seed = 31): Observation[] {
  const w = createSyntheticStream()
  const rnd = createRng(seed)
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const out: Observation[] = []
  for (let i = 0; i < n; i++) {
    out.push(w.observe())
    w.act({ move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS), key: pick(KEYS.length) })
  }
  return out
}

describe("frozen runs (checkups, spec §8.3)", { timeout: 60_000 }, () => {
  const trained = async (seed: number, steps: number, be: Backend = createCpuBackend()) => {
    const updates: UpdateInfo[] = []
    const l = await createLearner({ backend: be, seed, widths: TINY, settings: DEFAULT_SETTINGS, onUpdate: (u) => updates.push(u) })
    const w = createSyntheticStream()
    const actions: Action[] = []
    for (let i = 0; i < steps; i++) { const a = await l.step(w.observe()); actions.push(a); w.act(a) }
    return { l, w, updates, actions, be }
  }

  it("a frozen run changes nothing but evalCU", async () => {
    const { l } = await trained(41, 50)
    await l.sleep()
    const before = await l.save()
    const run = l.frozenRun({ seed: 7, measureLoss: true })
    for (const o of stream(300)) await run.step(o)
    run.dispose()
    const after = await l.save()
    expect(firstDifference({ ...after, stats: { ...after.stats, evalCU: 0 } }, { ...before, stats: { ...before.stats, evalCU: 0 } })).toBeNull()
    expect({ ...after, stats: { ...after.stats, evalCU: 0 } }).toEqual({ ...before, stats: { ...before.stats, evalCU: 0 } })
    expect(after.stats.evalCU).toBeGreaterThan(before.stats.evalCU)
    expect(after.stats.cu).toBe(before.stats.cu)
    l.dispose()
  })

  it("after a frozen run, the learner continues bit-identically to a twin that had none", async () => {
    const a = await trained(42, 300)
    const b = await trained(42, 300)
    const run = a.l.frozenRun({ seed: 3, measureLoss: true })
    for (const o of stream(200)) await run.step(o)
    run.dispose()
    for (const t of [a, b]) for (let i = 0; i < 200; i++) { const x = await t.l.step(t.w.observe()); t.actions.push(x); t.w.act(x) }
    expect(a.updates.length).toBe(499)
    expect(a.updates).toEqual(b.updates)
    expect(a.actions).toEqual(b.actions)
    expect(await a.l.readParams()).toEqual(await b.l.readParams())
    a.l.dispose()
    b.l.dispose()
  })

  it("frozen runs are deterministic per seed and draw nothing from the learner's RNG", async () => {
    const { l } = await trained(43, 40)
    const rng0 = (await l.save()).rng
    const obs = stream(60)
    const actionsOf = async (seed: number) => {
      const run = l.frozenRun({ seed, measureLoss: false })
      const as: Action[] = []
      for (const o of obs) as.push(await run.step(o))
      run.dispose()
      return as
    }
    const a7 = await actionsOf(7)
    expect(a7.every(isValidAction)).toBe(true)
    expect(await actionsOf(7)).toEqual(a7)
    expect(await actionsOf(8)).not.toEqual(a7)
    expect((await l.save()).rng).toBe(rng0)
    l.dispose()
  })

  it("a newborn's frozen predLoss equals its baselineLoss exactly (copy-last), so M1 cannot pass at birth", async () => {
    const l = await createLearner({ backend: createCpuBackend(), seed: 44, widths: TINY, settings: DEFAULT_SETTINGS })
    const run = l.frozenRun({ seed: 1, measureLoss: true })
    for (const o of stream(50)) await run.step(o)
    const r = run.result()
    run.dispose()
    expect(r.transitions).toBe(49)
    expect(r.baselineLoss).toBeGreaterThan(0)
    expect(r.predLoss).toBe(r.baselineLoss)
    l.dispose()
  })

  it("measureLoss false: result() losses are NaN and transitions 0; evalCU spent is less than with measureLoss", async () => {
    const { l } = await trained(45, 30)
    const obs = stream(40)
    const spend = async (measureLoss: boolean) => {
      const e0 = l.stats().evalCU
      const run = l.frozenRun({ seed: 2, measureLoss })
      for (const o of obs) await run.step(o)
      const r = run.result()
      run.dispose()
      return { r, spent: l.stats().evalCU - e0 }
    }
    const off = await spend(false)
    const on = await spend(true)
    expect(Number.isNaN(off.r.predLoss) && Number.isNaN(off.r.baselineLoss)).toBe(true)
    expect(off.r.transitions).toBe(0)
    expect(on.r.transitions).toBe(39)
    expect(Number.isFinite(on.r.predLoss) && Number.isFinite(on.r.baselineLoss)).toBe(true)
    expect(off.spent).toBeGreaterThan(0)
    expect(off.spent).toBeLessThan(on.spent)
    // No transitions yet: NaN even with measureLoss.
    const fresh = l.frozenRun({ seed: 2, measureLoss: true })
    expect(fresh.result().transitions).toBe(0)
    expect(Number.isNaN(fresh.result().predLoss)).toBe(true)
    await fresh.step(obs[0]!)
    expect(fresh.result().transitions).toBe(0)
    fresh.dispose()
    l.dispose()
  })

  it("while a frozen run is open, step/sleep/save/replayLoss reject; after dispose they work", async () => {
    const { l, w } = await trained(46, 10)
    const run = l.frozenRun({ seed: 1, measureLoss: true })
    const open = /a frozen run is open \(dispose it first\)/
    await expect(l.step(w.observe())).rejects.toThrow(open)
    await expect(l.sleep()).rejects.toThrow(open)
    await expect(l.save()).rejects.toThrow(open)
    await expect(l.replayLoss()).rejects.toThrow(open)
    expect(() => l.frozenRun({ seed: 2, measureLoss: true })).toThrow(open)
    await run.step(w.observe())
    run.dispose()
    run.dispose() // idempotent
    await expect(run.step(w.observe())).rejects.toThrow(/dispose/)
    expect(l.stats().steps).toBe(10)
    w.act(await l.step(w.observe()))
    await l.sleep()
    await l.save()
    expect(Number.isFinite(await l.replayLoss())).toBe(true)
    expect(l.stats().steps).toBe(11)
    l.dispose()
  })

  it("a frozen run leaks nothing: liveBuffers after dispose equals before, also when a step throws part-way", async () => {
    const make = async (failAt: number) => {
      const be = createCpuBackend()
      const { upload, zeros } = be
      let k = 0, armed = false
      const take = () => { if (armed && ++k === failAt) throw new Error("device lost") }
      Object.assign(be, { upload: (d: Float32Array) => { take(); return upload(d) }, zeros: (m: number) => { take(); return zeros(m) } })
      const l = await createLearner({ backend: be, seed: 47, widths: TINY, settings: DEFAULT_SETTINGS })
      const w = createSyntheticStream()
      for (let i = 0; i < 4; i++) w.act(await l.step(w.observe()))
      const before = be.stats.liveBuffers
      armed = true
      return { be, l, before, count: () => k }
    }
    const obs = stream(3)
    const probe = await make(0)
    const pr = probe.l.frozenRun({ seed: 1, measureLoss: true })
    for (const o of obs) await pr.step(o)
    pr.dispose()
    expect(probe.be.stats.liveBuffers).toBe(probe.before)
    const n = probe.count()
    expect(n).toBeGreaterThan(10)
    probe.l.dispose()
    for (let failAt = 1; failAt <= n; failAt++) {
      const { be, l, before } = await make(failAt)
      let threw = false
      try {
        const run = l.frozenRun({ seed: 1, measureLoss: true })
        try {
          for (const o of obs) await run.step(o)
        } finally {
          run.dispose()
        }
      } catch (e) {
        expect((e as Error).message).toBe("device lost")
        threw = true
      }
      expect(threw, `failing at allocation ${failAt}`).toBe(true)
      expect(be.stats.liveBuffers, `failing at allocation ${failAt}`).toBe(before)
      await l.step(obs[0]!) // the learner itself is untouched and usable
      l.dispose()
      expect(be.stats.liveBuffers).toBe(0)
    }
  })

  it("lastLoss is this step's L_before (NaN on step 1); a state saved without lastLoss loads with NaN", async () => {
    const be = createCpuBackend()
    const updates: UpdateInfo[] = []
    const l = await createLearner({ backend: be, seed: 48, widths: TINY, settings: DEFAULT_SETTINGS, onUpdate: (u) => updates.push(u) })
    expect(Number.isNaN(l.stats().lastLoss)).toBe(true)
    const w = createSyntheticStream()
    w.act(await l.step(w.observe()))
    expect(Number.isNaN(l.stats().lastLoss)).toBe(true)
    for (let i = 0; i < 5; i++) {
      w.act(await l.step(w.observe()))
      expect(l.stats().lastLoss).toBe(updates.at(-1)!.lossBefore)
      expect(l.stats().lastLoss).toBeGreaterThan(0)
    }
    const state = await l.save()
    expect(state.stats.lastLoss).toBe(l.stats().lastLoss)
    l.dispose()
    const { lastLoss: _gone, ...older } = state.stats
    void _gone
    const restored = await createLearner({ backend: be, state: { ...state, stats: older as LearnerState["stats"] } })
    expect(Number.isNaN(restored.stats().lastLoss)).toBe(true)
    restored.dispose()
    for (const bad of [-1, Number.POSITIVE_INFINITY, "0.1", undefined]) {
      await expect(createLearner({ backend: be, state: { ...state, stats: { ...state.stats, lastLoss: bad } as unknown as LearnerState["stats"] } }), String(bad)).rejects.toThrow(RangeError)
    }
    expect(be.stats.liveBuffers).toBe(0)
  })
})

/**
 * A CPU backend whose reads can be delayed (each settles after 5 ms, reading the storage then) or made to fail (the
 * `failRead`-th read after arming rejects at once), and which counts uploads, any dispose made while a read is
 * still pending, and any dispose of a storage that itself has a read pending. Patched in place: a tensor's backend is its storage's, so a wrapper object would miss reads.
 */
function flakyBackend() {
  const be = createCpuBackend()
  const { read, dispose, upload } = be
  const ctl = { delay: false, failRead: 0, reads: 0, pending: 0, disposedWhilePending: 0, disposedUnderRead: 0, uploads: 0 }
  /** Storages with a read pending (a count per storage). */
  const underRead = new Map<Storage, number>()
  Object.assign(be, {
    read: (s: Storage) => {
      ctl.reads++
      if (ctl.failRead > 0 && ctl.reads === ctl.failRead) return Promise.reject(new Error("read failed"))
      if (!ctl.delay) return read(s)
      ctl.pending++
      underRead.set(s, (underRead.get(s) ?? 0) + 1)
      return new Promise((r) => setTimeout(r, 5)).then(() => read(s)).finally(() => {
        ctl.pending--
        const n = underRead.get(s)! - 1
        if (n === 0) underRead.delete(s)
        else underRead.set(s, n)
      })
    },
    dispose: (s: Storage) => {
      if (ctl.pending > 0) ctl.disposedWhilePending++
      if (underRead.has(s)) ctl.disposedUnderRead++
      dispose(s)
    },
    upload: (d: Float32Array) => { ctl.uploads++; return upload(d) },
  })
  /** From now on: delay every read, and fail the `failRead`-th one (0: none). */
  const arm = (failRead = 0) => { ctl.reads = 0; ctl.delay = true; ctl.failRead = failRead }
  const disarm = () => { ctl.delay = false; ctl.failRead = 0 }
  return { be, ctl, arm, disarm }
}

describe("robustness (Plan 04c)", { timeout: 60_000 }, () => {
  const warm = async (seed: number, steps: number) => {
    const f = flakyBackend()
    const l = await createLearner({ backend: f.be, seed, widths: TINY, settings: { ...DEFAULT_SETTINGS, replayPerSleep: 4 } })
    const w = createSyntheticStream()
    for (let i = 0; i < steps; i++) w.act(await l.step(w.observe()))
    return { ...f, l, w }
  }

  it("readAll: one read rejects while another is pending — nothing is disposed until both settle; the step rejects with the first error", async () => {
    // A newborn's first step reads two tensors at once (the acting heads and h_t): the first rejects at once.
    const { be, ctl, arm, l, w } = await warm(51, 0)
    arm(1)
    await expect(l.step(w.observe())).rejects.toThrow("read failed")
    expect(ctl.pending).toBe(0)
    expect(ctl.disposedWhilePending).toBe(0)
    l.dispose()
    expect(ctl.disposedWhilePending).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("a step whose read rejects frees its step-local tensors: after dispose, liveBuffers is 0", async () => {
    const probe = await warm(52, 3)
    probe.arm()
    probe.w.act(await probe.l.step(probe.w.observe()))
    const n = probe.ctl.reads
    expect(n).toBeGreaterThanOrEqual(6) // L_before, L_after, heads and h, actor-critic loss and advantage
    probe.l.dispose()
    for (let failAt = 1; failAt <= n; failAt++) {
      const { be, ctl, arm, disarm, l, w } = await warm(52, 3)
      const before = await l.save()
      arm(failAt)
      await expect(l.step(w.observe()), `failing read ${failAt}`).rejects.toThrow("read failed")
      expect(ctl.disposedWhilePending, `failing read ${failAt}`).toBe(0)
      disarm()
      // The carried state is as it was before the step.
      const after = await l.save()
      expect(firstDifference(after.h, before.h), `failing read ${failAt}`).toBeNull()
      expect(firstDifference(after.last, before.last), `failing read ${failAt}`).toBeNull()
      l.dispose()
      expect(be.stats.liveBuffers, `failing read ${failAt}`).toBe(0)
    }
  })

  it("dispose() during an in-flight step: the step settles, later calls reject 'used after dispose', no 'storage used after dispose' error, liveBuffers 0", async () => {
    const { be, ctl, arm, l, w } = await warm(53, 3)
    arm()
    const o = w.observe()
    const p = l.step(o)
    l.dispose()
    l.dispose() // idempotent
    const outcome = await p.catch((e: unknown) => e)
    expect(outcome instanceof Error ? outcome.message : "").not.toMatch(/storage used after dispose/)
    expect(isValidAction(outcome as Action)).toBe(true)
    for (const call of [() => l.step(o), () => l.sleep(), () => l.save(), () => l.replayLoss()]) {
      await expect(call()).rejects.toThrow("learner: used after dispose")
    }
    expect(() => l.frozenRun({ seed: 1, measureLoss: true })).toThrow("learner: used after dispose")
    expect(ctl.disposedWhilePending).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("dispose() during an in-flight sleep: the sleep settles, later calls reject 'used after dispose', liveBuffers 0", async () => {
    const { be, ctl, arm, l, w } = await warm(54, 6)
    arm()
    const p = l.sleep()
    l.dispose()
    const outcome = await p.catch((e: unknown) => e)
    expect(outcome).toBeUndefined()
    await expect(l.step(w.observe())).rejects.toThrow("learner: used after dispose")
    expect(ctl.disposedWhilePending).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("dispose() during an in-flight frozen step: the step settles, later calls reject, liveBuffers 0", async () => {
    const { be, ctl, arm, l, w } = await warm(55, 3)
    const run = l.frozenRun({ seed: 1, measureLoss: true })
    const obs = stream(3)
    await run.step(obs[0]!)
    await run.step(obs[1]!)
    arm()
    const p = run.step(obs[2]!)
    l.dispose()
    run.dispose()
    const outcome = await p.catch((e: unknown) => e)
    expect(outcome instanceof Error ? outcome.message : "").not.toMatch(/storage used after dispose/)
    expect(isValidAction(outcome as Action)).toBe(true)
    await expect(run.step(obs[0]!)).rejects.toThrow(/dispose/)
    await expect(l.step(w.observe())).rejects.toThrow("learner: used after dispose")
    expect(ctl.disposedWhilePending).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("dispose() during an in-flight readParams: it resolves to the values, nothing is freed under it, later calls reject, liveBuffers 0", async () => {
    const { be, ctl, arm, l } = await warm(58, 3)
    const expected = await l.readParams()
    arm()
    const p = l.readParams()
    l.dispose()
    expect(await p).toEqual(expected)
    await expect(l.readParams()).rejects.toThrow("learner: used after dispose")
    expect(ctl.disposedWhilePending).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("dispose() while an exclusive call and a readParams are in flight: the one that settles last tears down", async () => {
    // replayLoss on an empty buffer settles before readParams' delayed reads: teardown must still wait for them.
    const { be, ctl, arm, l } = await warm(59, 0)
    arm()
    const params = l.readParams()
    const replayLoss = l.replayLoss()
    l.dispose()
    expect(Number.isNaN(await replayLoss)).toBe(true)
    expect(ctl.pending).toBeGreaterThan(0) // readParams is still reading
    expect((await params).every((x) => x.every(Number.isFinite))).toBe(true)
    expect(ctl.disposedUnderRead).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("dispose() during a step and a readParams both in flight: nothing is freed under either's reads, liveBuffers 0", async () => {
    const { be, ctl, arm, l, w } = await warm(60, 3)
    arm()
    const step = l.step(w.observe())
    const params = l.readParams()
    l.dispose()
    const [a, ps] = await Promise.all([step, params])
    expect(isValidAction(a)).toBe(true)
    expect(ps.every((x) => x.every(Number.isFinite))).toBe(true)
    // The step frees its own scratch while readParams' reads are pending (other storages): only a storage freed under
    // its own pending read would be a fault.
    expect(ctl.disposedUnderRead).toBe(0)
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("frozenRun() while a step is in flight throws 'already in progress'", async () => {
    const { arm, l, w } = await warm(56, 2)
    arm()
    const p = l.step(w.observe())
    expect(() => l.frozenRun({ seed: 1, measureLoss: false })).toThrow(/already in progress/)
    await p
    l.dispose()
  })

  it("lastLoss is NaN after a step whose L_before is non-finite", async () => {
    const inner = createCpuBackend()
    const allocated: Storage[] = []
    const be: Backend = {
      ...inner,
      upload: (d) => { const st = inner.upload(d); allocated.push(st); return st },
      zeros: (n) => { const st = inner.zeros(n); allocated.push(st); return st },
    }
    const l = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    const w = createSyntheticStream()
    for (let i = 0; i < 5; i++) w.act(await l.step(w.observe()))
    expect(l.stats().lastLoss).toBeGreaterThan(0)
    ;(allocated[35] as unknown as { data: Float32Array }).data[7] = Number.NaN // Model.params[35], headP's bias
    w.act(await l.step(w.observe()))
    expect(l.stats().skippedUpdates).toBe(1)
    expect(Number.isNaN(l.stats().lastLoss)).toBe(true)
    l.dispose()
  })

  it("measureLoss false: no target upload per frozen step", async () => {
    const { ctl, l } = await warm(57, 5)
    const obs = stream(10)
    const uploads = async (measureLoss: boolean) => {
      const run = l.frozenRun({ seed: 3, measureLoss })
      const u0 = ctl.uploads
      for (const o of obs) await run.step(o)
      const n = ctl.uploads - u0
      run.dispose()
      return n
    }
    // Per step: 3 inputs, 2 targets and (from the second step on) 1 action when measuring; 3 inputs only when not.
    expect(await uploads(true)).toBe(10 * 5 + 9)
    expect(await uploads(false)).toBe(10 * 3)
    l.dispose()
  })
})

describe("step speed (Plan 04c Task 2)", { timeout: 60_000 }, () => {
  it("a learning step makes exactly 3 serial read rounds (no sleep)", async () => {
    // A round starts whenever a read is made while no earlier read is still pending (reads are delayed 5 ms, so reads
    // made together overlap and reads made one after another do not).
    const { be, ctl, arm } = flakyBackend()
    const { read } = be
    let rounds = 0
    Object.assign(be, { read: (s: Storage) => { if (ctl.pending === 0) rounds++; return read(s) } })
    arm()
    const updates: UpdateInfo[] = []
    const l = await createLearner({ backend: be, seed: 61, widths: TINY, settings: DEFAULT_SETTINGS, onUpdate: (u) => updates.push(u) })
    const w = createSyntheticStream()
    const perStep: number[] = []
    for (let i = 1; i <= 20; i++) {
      rounds = 0
      w.act(await l.step(w.observe()))
      if (i >= 3) perStep.push(rounds)
    }
    // Every measured step learned on finite losses (all four reads were needed).
    expect(updates.slice(1).every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.lossAfter) && !u.skipped)).toBe(true)
    expect(perStep).toEqual(Array(18).fill(3))
    l.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })
})

describe("gamma setting (Plan 05a)", { timeout: 60_000 }, () => {
  it("DEFAULT_SETTINGS.gamma is GAMMA (0.9)", () => {
    expect(GAMMA).toBe(0.9)
    expect(DEFAULT_SETTINGS.gamma).toBe(GAMMA)
    expect(GAMMA_RANGE).toEqual([0, 0.999])
    // Appended last, so the earlier keys keep their order.
    expect(Object.keys(DEFAULT_SETTINGS).at(-1)).toBe("gamma")
  })

  it("clampSettings: gamma absent → 0.9; 1.2 → 0.999; −1 → 0; NaN, null or a string → RangeError", () => {
    const { gamma: _g, ...older } = DEFAULT_SETTINGS
    void _g
    expect(clampSettings(older as LearnerSettings).gamma).toBe(0.9)
    expect(clampSettings({ ...DEFAULT_SETTINGS, gamma: 1.2 }).gamma).toBe(0.999)
    expect(clampSettings({ ...DEFAULT_SETTINGS, gamma: -1 }).gamma).toBe(0)
    expect(clampSettings({ ...DEFAULT_SETTINGS, gamma: 0.6 }).gamma).toBe(0.6)
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      expect(() => clampSettings({ ...DEFAULT_SETTINGS, gamma: bad }), String(bad)).toThrow(RangeError)
    }
    // Only undefined means absent (final review): null (e.g. a JSON round trip of NaN) is not the old constant.
    for (const bad of [null, "0.9"]) {
      expect(() => clampSettings({ ...DEFAULT_SETTINGS, gamma: bad as unknown as number }), String(bad)).toThrow(RangeError)
    }
  })

  it("a state saved without gamma resumes with γ 0.9, and its next 50 steps are bit-identical to a learner whose state had gamma 0.9", async () => {
    const first = await run({ seed: 51, steps: 40, sleepEvery: 20 })
    const state = await first.learner.save()
    first.learner.dispose()
    expect(state.settings.gamma).toBe(0.9)
    const { gamma: _g, ...older } = state.settings
    void _g
    const without = { ...state, settings: older as LearnerSettings } // as saved before Plan 05a
    const withGamma = { ...state, settings: { ...older, gamma: 0.9 } }
    const a = await run({ seed: -1, steps: 50, sleepEvery: 20, stream: createSyntheticStream(), state: without })
    const b = await run({ seed: -1, steps: 50, sleepEvery: 20, stream: createSyntheticStream(), state: withGamma })
    expect(a.updates.length).toBe(50)
    expect(a.actions).toEqual(b.actions)
    expect(a.updates).toEqual(b.updates)
    const [sa, sb] = [await a.learner.save(), await b.learner.save()]
    expect(sa.settings.gamma).toBe(0.9) // the next save writes gamma
    expect(Object.keys(sa.settings)).toEqual(Object.keys(DEFAULT_SETTINGS))
    expect(firstDifference(sa, sb)).toBeNull()
    for (const r of [a, b]) r.learner.dispose()
    expect(a.be.stats.liveBuffers).toBe(0)
    expect(b.be.stats.liveBuffers).toBe(0)
  })

  it("a stored gamma that is not a clamped setting is a bad checkpoint", async () => {
    const { learner } = await run({ seed: 52, steps: 3 })
    const state = await learner.save()
    learner.dispose()
    for (const bad of [1.2, -1, Number.NaN, "0.9"]) {
      const s = { ...state, settings: { ...state.settings, gamma: bad } } as unknown as LearnerState
      await expect(createLearner({ backend: createCpuBackend(), state: s }), String(bad)).rejects.toThrow(RangeError)
    }
  })

  it("γ 0.6 gives a different advantage trajectory from γ 0.9 on the same stream", async () => {
    const a = await run({ seed: 53, steps: 30, settings: { gamma: 0.6 } })
    const b = await run({ seed: 53, steps: 30, settings: { gamma: 0.9 } })
    // The first update's advantage uses the same untrained value head; from there the targets differ.
    const adv = (r: typeof a) => r.updates.map((u) => u.advantage)
    expect(adv(a)).toHaveLength(29)
    expect(adv(a)).not.toEqual(adv(b))
    expect((await a.learner.save()).settings.gamma).toBe(0.6)
    for (const r of [a, b]) r.learner.dispose()
  })
})

describe("whole-life replay (reservoirShare, Plan 06a)", { timeout: 60_000 }, () => {
  it("is not one of DEFAULT_SETTINGS: today's learner has no reservoir, and its settings keys are unchanged", () => {
    expect("reservoirShare" in DEFAULT_SETTINGS).toBe(false)
    expect(Object.keys(DEFAULT_SETTINGS).at(-1)).toBe("gamma")
    expect(RESERVOIR_SHARE_RANGE).toEqual([0, 1])
  })

  it("clampSettings: absent stays absent; 0.5 → 0.5; 1.2 → 1; −0.1 → 0; NaN, null or a string → RangeError", () => {
    expect("reservoirShare" in clampSettings(DEFAULT_SETTINGS)).toBe(false)
    expect(clampSettings({ ...DEFAULT_SETTINGS, reservoirShare: 0.5 }).reservoirShare).toBe(0.5)
    expect(clampSettings({ ...DEFAULT_SETTINGS, reservoirShare: 1.2 }).reservoirShare).toBe(1)
    expect(clampSettings({ ...DEFAULT_SETTINGS, reservoirShare: -0.1 }).reservoirShare).toBe(0)
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, null, "0.5"]) {
      expect(() => clampSettings({ ...DEFAULT_SETTINGS, reservoirShare: bad as unknown as number }), String(bad)).toThrow(RangeError)
    }
  })

  it("share 0 is bit-identical to a learner without the setting, and saves no reservoir", async () => {
    const a = await run({ seed: 61, steps: 120, sleepEvery: 20, settings: { bufferSize: 16 } })
    const b = await run({ seed: 61, steps: 120, sleepEvery: 20, settings: { bufferSize: 16, reservoirShare: 0 } })
    expect(b.actions).toEqual(a.actions)
    expect(b.updates).toEqual(a.updates)
    const [sa, sb] = [await a.learner.save(), await b.learner.save()]
    expect("reservoir" in sa).toBe(false)
    expect("reservoir" in sb).toBe(false)
    const { reservoirShare, ...rest } = sb.settings
    expect(reservoirShare).toBe(0)
    expect(firstDifference({ ...sb, settings: rest }, sa)).toBeNull()
    for (const r of [a, b]) r.learner.dispose()
  })

  it("share 0.5 of 16 slots: 8 recent transitions and 8 kept from the whole life, most of them older than the recent ones", async () => {
    const { learner, be } = await run({ seed: 62, steps: 100, sleepEvery: 20, settings: { bufferSize: 16, reservoirShare: 0.5 } })
    const state = await learner.save()
    expect(state.replay.length).toBe(8)
    expect(state.reservoir!.items.length).toBe(8)
    expect(state.reservoir!.seen).toBe(99) // every transition so far was offered: one per step after the first
    const recent = new Set(state.replay)
    expect(state.reservoir!.items.filter(tr => !recent.has(tr)).length).toBeGreaterThanOrEqual(6)
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("replaying the whole life changes what is learned at sleep (same seed, same stream, share 0 against 1)", async () => {
    const a = await run({ seed: 63, steps: 80, sleepEvery: 40, settings: { bufferSize: 8 } })
    const b = await run({ seed: 63, steps: 80, sleepEvery: 40, settings: { bufferSize: 8, reservoirShare: 1 } })
    const sb = await b.learner.save()
    expect(sb.replay.length).toBe(1) // one slot always stays with the most recent transition
    expect(sb.reservoir!.items.length).toBe(7)
    expect(firstDifference(await b.learner.readParams(), await a.learner.readParams())).not.toBeNull()
    for (const r of [a, b]) r.learner.dispose()
  })

  it("save at step 60 with a full reservoir, load, 60 more steps = an uninterrupted 120-step run", async () => {
    const settings = { bufferSize: 16, reservoirShare: 0.5 }
    const full = await run({ seed: 64, steps: 120, sleepEvery: 20, settings })
    const world = createSyntheticStream()
    const first = await run({ seed: 64, steps: 60, sleepEvery: 20, stream: world, settings })
    const state = await first.learner.save()
    expect(state.reservoir!.seen).toBe(59)
    first.learner.dispose()
    expect(first.be.stats.liveBuffers).toBe(0)
    const second = await run({ seed: -1, steps: 60, sleepEvery: 20, stream: world, state })
    expect([...first.actions, ...second.actions]).toEqual(full.actions)
    expect([...first.updates, ...second.updates]).toEqual(full.updates)
    expect(firstDifference(await second.learner.save(), await full.learner.save())).toBeNull()
    for (const r of [full, second]) r.learner.dispose()
    expect(second.be.stats.liveBuffers).toBe(0)
  })

  it("replayLoss covers the kept transitions too, each once", async () => {
    const { learner } = await run({ seed: 65, steps: 60, settings: { bufferSize: 8, reservoirShare: 0.5 } })
    const state = await learner.save()
    const distinct = new Set([...state.replay, ...state.reservoir!.items]).size
    expect(distinct).toBeGreaterThan(4)
    const flops = learner.stats().evalCU
    expect(Number.isFinite(await learner.replayLoss())).toBe(true)
    expect(learner.stats().evalCU).toBe(flops)
    learner.dispose()
  })

  it("a state whose reservoir does not fit its settings is a bad checkpoint", async () => {
    const { learner } = await run({ seed: 66, steps: 40, settings: { bufferSize: 8, reservoirShare: 0.5 } })
    const state = await learner.save()
    learner.dispose()
    const plain = await run({ seed: 66, steps: 5 })
    const plainState = await plain.learner.save()
    plain.learner.dispose()
    const r = state.reservoir!
    const bad: [string, unknown][] = [
      ["a reservoir on a learner without the setting", { ...plainState, reservoir: { items: [], seen: 0 } }],
      ["no reservoir on a learner with the setting", Object.fromEntries(Object.entries(state).filter(([k]) => k !== "reservoir"))],
      ["more kept than slots", { ...state, reservoir: { items: [...r.items, r.items[0]!], seen: r.seen + 1 } }],
      ["seen below what is kept", { ...state, reservoir: { items: r.items, seen: 2 } }],
      ["seen not an integer", { ...state, reservoir: { items: r.items, seen: 39.5 } }],
      ["not full, yet more seen than kept", { ...state, reservoir: { items: r.items.slice(0, 2), seen: 30 } }],
      ["a kept transition with a bad recurrent state", { ...state, reservoir: { items: [{ ...r.items[0]!, hIn: new Float32Array(3) }, ...r.items.slice(1)], seen: r.seen } }],
      ["a kept transition with a bad action", { ...state, reservoir: { items: [{ ...r.items[0]!, action: { move: 99, button: 0, scroll: 0, key: 0 } }, ...r.items.slice(1)], seen: r.seen } }],
      ["a kept transition with a bad frame", { ...state, reservoir: { items: [{ ...r.items[0]!, obs: { ...r.items[0]!.obs, fovea: new Uint8Array(4) } }, ...r.items.slice(1)], seen: r.seen } }],
      ["a malformed reservoir", { ...state, reservoir: { items: "none", seen: 0 } }],
      ["a share that is not a clamped setting", { ...state, settings: { ...state.settings, reservoirShare: 1.5 } }],
    ]
    for (const [what, s] of bad) {
      const be = createCpuBackend()
      await expect(createLearner({ backend: be, state: s as LearnerState }), what).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers, what).toBe(0)
    }
    // The untouched state is a good checkpoint.
    const ok = await createLearner({ backend: createCpuBackend(), state })
    ok.dispose()
  })
})
