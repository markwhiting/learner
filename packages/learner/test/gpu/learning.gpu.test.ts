import { describe, expect, it } from "vitest"
import { createCpuBackend, createWebGpuBackend, isSoftwareAdapter } from "@blanktab/tensor"
import type { Backend } from "@blanktab/tensor"
import { isValidAction, type Action } from "@blanktab/interface"
import { DEFAULT_SETTINGS, createLearner } from "../../src/learner"
import type { LearnerSettings } from "../../src/learner"
import type { Widths } from "../../src/model"
import { createSyntheticStream, heldOutSet } from "../synthetic"

// Does the learner learn on WebGPU as it does on the CPU? Same seed, same visible-motion stream, tiny widths.
// Runs only in real Chromium via `npm run test:gpu`.

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const STEPS = 300
const SEED = 1
const allowSoftware = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env.GPU_ALLOW_SOFTWARE === "1"

/** `start`: where the synthetic block starts (near the screen's edge the close-up changes as it moves, see synthetic.ts). */
async function live(backend: Backend, extra: Partial<LearnerSettings> = {}, start?: { x: number; y: number }) {
  const learner = await createLearner({ backend, seed: SEED, widths: TINY, settings: { ...DEFAULT_SETTINGS, ...extra }, heldOut: heldOutSet() })
  const world = createSyntheticStream(start ? { start } : {})
  const moves: number[] = []
  for (let i = 0; i < STEPS; i++) {
    const a = await learner.step(world.observe())
    moves.push(a.move)
    world.act(a)
  }
  // Training is identical with or without the held-out set; the sleep scores it (after its replay).
  const beforeSleep = learner.stats()
  await learner.sleep()
  const stats = { ...beforeSleep, heldOutLoss: learner.stats().heldOutLoss, heldOutBaseline: learner.stats().heldOutBaseline, evalCU: learner.stats().evalCU }
  const params = await learner.readParams()
  learner.dispose()
  return { stats, moves, outputs: params.slice(-4) }
}

const maxAbs = (a: Float32Array) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0)

describe("the learner on WebGPU", () => {
  it(`learns like the CPU learner: same seed and stream, ${STEPS} steps, predLoss within 2%; the residual output layers move`, async () => {
    const gpu = await createWebGpuBackend({ powerPreference: "high-performance" })
    if (!gpu) throw new Error("WebGPU backend unavailable")
    if (!allowSoftware && isSoftwareAdapter(gpu.adapterInfo)) throw new Error(`software adapter ${JSON.stringify(gpu.adapterInfo)}`)
    const cpu = await live(createCpuBackend())
    const onGpu = await live(gpu)
    const rel = Math.abs(onGpu.stats.predLoss - cpu.stats.predLoss) / cpu.stats.predLoss
    const sameMoves = onGpu.moves.filter((m, i) => m === cpu.moves[i]).length
    const outMax = onGpu.outputs.map(maxAbs)
    console.log(JSON.stringify({
      adapter: gpu.adapterInfo, cpu: { predLoss: cpu.stats.predLoss, baselineLoss: cpu.stats.baselineLoss, cu: cpu.stats.cu },
      gpu: { predLoss: onGpu.stats.predLoss, baselineLoss: onGpu.stats.baselineLoss, cu: onGpu.stats.cu, skipped: onGpu.stats.skippedUpdates },
      heldOut: { cpu: [cpu.stats.heldOutLoss, cpu.stats.heldOutBaseline, cpu.stats.evalCU], gpu: [onGpu.stats.heldOutLoss, onGpu.stats.heldOutBaseline, onGpu.stats.evalCU] },
      relPredLoss: rel, sameMoves: `${sameMoves}/${STEPS}`, gpuOutputMaxAbs: outMax, cpuOutputMaxAbs: cpu.outputs.map(maxAbs),
    }))
    expect(Number.isFinite(onGpu.stats.predLoss)).toBe(true)
    expect(onGpu.stats.skippedUpdates).toBe(0)
    expect(rel).toBeLessThanOrEqual(0.02)
    for (const m of outMax) expect(m).toBeGreaterThan(0)
    // The held-out score on WebGPU matches the CPU's (same weights up to float error), billed the same evalCU.
    expect(Math.abs(onGpu.stats.heldOutLoss - cpu.stats.heldOutLoss) / cpu.stats.heldOutLoss).toBeLessThanOrEqual(0.02)
    expect(onGpu.stats.heldOutBaseline).toBe(cpu.stats.heldOutBaseline)
    expect(onGpu.stats.evalCU).toBeCloseTo(cpu.stats.evalCU, 12)
    expect(onGpu.stats.evalCU).toBeGreaterThan(0)
  })

  it(`with the optional predictor mechanisms (Plan 06a1: the surround and a shift gain) it still learns like the CPU learner: ${STEPS} steps, predLoss within 2%, nothing leaks`, async () => {
    const gpu = await createWebGpuBackend({ powerPreference: "high-performance" })
    if (!gpu) throw new Error("WebGPU backend unavailable")
    if (!allowSoftware && isSoftwareAdapter(gpu.adapterInfo)) throw new Error(`software adapter ${JSON.stringify(gpu.adapterInfo)}`)
    const all = { foveaSurround: 1, shiftGain: 16 }
    // Started beside the screen's corner, so the close-up's padding and surround are in play from the first steps.
    const start = { x: 12, y: 12 }
    const before = gpu.stats.liveBuffers
    const cpu = await live(createCpuBackend(), all, start)
    const onGpu = await live(gpu, all, start)
    const rel = Math.abs(onGpu.stats.predLoss - cpu.stats.predLoss) / cpu.stats.predLoss
    console.log(JSON.stringify({ mechanisms: all, cpu: { predLoss: cpu.stats.predLoss, cu: cpu.stats.cu }, gpu: { predLoss: onGpu.stats.predLoss, cu: onGpu.stats.cu, skipped: onGpu.stats.skippedUpdates }, relPredLoss: rel,
      heldOut: { cpu: cpu.stats.heldOutLoss, gpu: onGpu.stats.heldOutLoss }, sameMoves: `${onGpu.moves.filter((m, i) => m === cpu.moves[i]).length}/${STEPS}` }))
    expect(Number.isFinite(onGpu.stats.predLoss)).toBe(true)
    expect(onGpu.stats.skippedUpdates).toBe(0)
    expect(rel).toBeLessThanOrEqual(0.02)
    expect(Math.abs(onGpu.stats.heldOutLoss - cpu.stats.heldOutLoss) / cpu.stats.heldOutLoss).toBeLessThanOrEqual(0.02)
    expect(onGpu.stats.cu).toBeCloseTo(cpu.stats.cu, 9) // the same operations, counted alike
    expect(gpu.stats.liveBuffers).toBe(before)
  })

  it("a 300-step frozen run on WebGPU: valid actions, liveBuffers back to its value before", async () => {
    const gpu = await createWebGpuBackend({ powerPreference: "high-performance" })
    if (!gpu) throw new Error("WebGPU backend unavailable")
    if (!allowSoftware && isSoftwareAdapter(gpu.adapterInfo)) throw new Error(`software adapter ${JSON.stringify(gpu.adapterInfo)}`)
    const learner = await createLearner({ backend: gpu, seed: SEED, widths: TINY, settings: { ...DEFAULT_SETTINGS } })
    const world = createSyntheticStream()
    for (let i = 0; i < 50; i++) world.act(await learner.step(world.observe()))
    const before = gpu.stats.liveBuffers
    const s0 = learner.stats()
    const run = learner.frozenRun({ seed: 7, measureLoss: true })
    const other = createSyntheticStream({ start: { x: 60, y: 60 } })
    const actions: Action[] = []
    for (let i = 0; i < STEPS; i++) {
      const a = await run.step(other.observe())
      actions.push(a)
      other.act(a)
    }
    const r = run.result()
    run.dispose()
    console.log(JSON.stringify({ frozen: r, evalCU: learner.stats().evalCU - s0.evalCU }))
    expect(actions.every(isValidAction)).toBe(true)
    expect(r.transitions).toBe(STEPS - 1)
    expect(Number.isFinite(r.predLoss) && Number.isFinite(r.baselineLoss)).toBe(true)
    expect(gpu.stats.liveBuffers).toBe(before)
    expect(learner.stats().cu).toBe(s0.cu)
    expect(learner.stats().steps).toBe(s0.steps)
    expect(learner.stats().evalCU).toBeGreaterThan(s0.evalCU)
    learner.dispose()
  })
})
