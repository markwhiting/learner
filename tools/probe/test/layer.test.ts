import { describe, it, expect } from "vitest"
import { createRng, hashSeed } from "@blanktab/detmath"
import { createModel, GPU_WIDTHS } from "@blanktab/learner"
import { conv2d, createCpuBackend, readSyncCpu, relu, tensor } from "@blanktab/tensor"
import { LAYER, centredGram, firstLayer, type LayerShape } from "../filters/layer.ts"
import type { Filters } from "../filters/lives.ts"

const rnd = (seed: number, n: number, lo: number, hi: number) => { const r = createRng(seed); return Float32Array.from({ length: n }, () => lo + (hi - lo) * r()) }
const input = (seed: number, s: LayerShape) => rnd(seed, 3 * s.inH * s.inW, 0, 1)
const filters = (seed: number, s: LayerShape, units = 16): Filters => ({ w: rnd(hashSeed(seed, "w"), units * 3 * s.k * s.k, -0.3, 0.3), b: rnd(hashSeed(seed, "b"), units, -0.1, 0.1) })

describe("the first layer in float64", () => {
  it("firstLayer equals relu(conv2d(...)) from the CPU backend on 3 seeded inputs and filters, for both encoders, to 1e-5 absolute", () => {
    const be = createCpuBackend()
    for (const e of ["periphery", "fovea"] as const) {
      const s = LAYER[e]
      for (let k = 0; k < 3; k++) {
        const x = input(hashSeed(k, e), s), f = filters(k, s)
        const ref = readSyncCpu(relu(conv2d(tensor(x, [3, s.inH, s.inW], { backend: be }), tensor(f.w, [16, 3 * s.k * s.k], { backend: be }), tensor(f.b, [16, 1], { backend: be }), { kh: s.k, kw: s.k, stride: s.stride, pad: s.pad })).storage)
        const mine = firstLayer(x, f, s)
        expect(mine.length).toBe(ref.length)
        let worst = 0
        for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(mine[i]! - ref[i]!))
        expect(worst, `${e} ${k}`).toBeLessThan(1e-5)
      }
    }
  })

  it("LAYER's output sizes equal model.ts's (25×40 and 16×16)", () => {
    expect([LAYER.periphery.outH, LAYER.periphery.outW]).toEqual([25, 40])
    expect([LAYER.fovea.outH, LAYER.fovea.outW]).toEqual([16, 16])
    // model.ts builds its layers from the same constants: a GPU model's first-layer weights have 3·k·k columns.
    const m = createModel(createCpuBackend(), 1, GPU_WIDTHS)
    expect(m.params[0]!.shape).toEqual([16, 3 * LAYER.periphery.k ** 2])
    expect(m.params[6]!.shape).toEqual([16, 3 * LAYER.fovea.k ** 2])
    m.dispose()
  })

  it("centredGram equals a naive two-pass float64 covariance on a 4-frame, 3-state fixture to 1e-12 relative", () => {
    for (const e of ["periphery", "fovea"] as const) {
      const s = LAYER[e]
      const frames = [0, 1, 2, 3].map(k => input(hashSeed(k, "frame"), s))
      const states = [0, 1, 2].map(k => filters(10 + k, s))
      const G = centredGram(frames, states, s)
      // Naive: every activation column in full, then mean, then centred dot products.
      const cols: number[][] = states.flatMap(() => Array.from({ length: 16 }, () => [] as number[]))
      for (const x of frames) states.forEach((f, si) => {
        const a = firstLayer(x, f, s), P = s.outH * s.outW
        for (let u = 0; u < 16; u++) for (let p = 0; p < P; p++) cols[si * 16 + u]!.push(a[u * P + p]!)
      })
      const mean = cols.map(c => c.reduce((a, v) => a + v, 0) / c.length)
      expect(G.cols).toBe(48)
      expect(G.samples).toBe(4 * s.outH * s.outW)
      for (let i = 0; i < 48; i++) for (let j = 0; j < 48; j++) {
        let want = 0
        for (let p = 0; p < cols[i]!.length; p++) want += (cols[i]![p]! - mean[i]!) * (cols[j]![p]! - mean[j]!)
        const got = G.g[i * 48 + j]!
        expect(Math.abs(got - want), `${e} ${i},${j}`).toBeLessThanOrEqual(1e-12 * Math.max(1e-300, Math.abs(want)) + 1e-14)
      }
    }
  })

  it("a state whose bias is −100 has every unit flagged varZero", () => {
    const s = LAYER.fovea
    const dead = { ...filters(3, s), b: new Float32Array(16).fill(-100) }
    const G = centredGram([input(1, s), input(2, s)], [filters(2, s), dead], s)
    expect(Array.from(G.varZero.slice(16))).toEqual(new Array(16).fill(1))
    for (let j = 0; j < 32; j++) expect(G.g[20 * 32 + j]).toBe(0)
  })

  it("centredGram run twice gives identical bytes", () => {
    const s = LAYER.periphery
    const frames = [0, 1].map(k => input(hashSeed(k, "twice"), s)), states = [0, 1, 2, 3, 4].map(k => filters(30 + k, s))
    const a = centredGram(frames, states, s), b = centredGram(frames, states, s)
    expect(Buffer.from(a.g.buffer).equals(Buffer.from(b.g.buffer))).toBe(true)
    expect(Buffer.from(a.means.buffer).equals(Buffer.from(b.means.buffer))).toBe(true)
  })
})
