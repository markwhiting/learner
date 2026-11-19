import { describe, it, expect } from "vitest"
import { PRED_PERIPHERY } from "@blanktab/interface"
import { createCpuBackend, createRng, getDefaultBackend, gradCheck, tensor } from "@blanktab/tensor"
import type { Backend } from "@blanktab/tensor"
import { ACTION_DIM } from "../src/encode"
import { WIDE_TAPS, createWideShift } from "../src/wide-shift"
import type { WideShiftParams } from "../src/wide-shift"

const PW = PRED_PERIPHERY.width, PH = PRED_PERIPHERY.height, PC = PRED_PERIPHERY.channels
const CELLS = PW * PH, PP = CELLS * PC
const R = (WIDE_TAPS - 1) / 2
const ENC = 4
const rng = createRng(44)
const signed = (n: number) => Float32Array.from({ length: n }, () => (rng() < 0.5 ? -1 : 1) * (0.2 + rng()))
const image = () => Float32Array.from({ length: PP }, (_, i) => (i + 1) / (PP + 1))
const pulse = (j: number) => {
  const k = new Float32Array(WIDE_TAPS)
  k[j] = 1
  return k
}
const action = () => {
  const a = new Float32Array(ACTION_DIM)
  for (const i of [3, 25, 29, 40]) a[i] = 1
  return a
}
const shapes: Record<keyof WideShiftParams, number[]> = {
  ax: [ACTION_DIM, WIDE_TAPS], ay: [ACTION_DIM, WIDE_TAPS], agate: [ACTION_DIM, 1], zx: [ENC, WIDE_TAPS], zy: [ENC, WIDE_TAPS], zgate: [ENC, 1], maskW: [ENC, CELLS], maskB: [CELLS],
}
const KEYS = Object.keys(shapes) as (keyof WideShiftParams)[]
const size = (k: keyof WideShiftParams) => shapes[k].reduce((a, d) => a * d, 1)
const params = (be: Backend, fill: (k: keyof WideShiftParams) => Float32Array): WideShiftParams =>
  Object.fromEntries(KEYS.map(k => [k, tensor(fill(k), shapes[k], { backend: be })])) as unknown as WideShiftParams

describe("wide-view shift (Plan 06a1)", () => {
  it("has 7 taps per axis: ±3 periphery-target cells", () => {
    expect(WIDE_TAPS).toBe(7)
  })

  it("its kernels are softmaxes of a·A + z·Z + the centre prior: with zero weights, e⁴ / (e⁴ + 6) stays put", async () => {
    const be = createCpuBackend()
    const wide = createWideShift(be)
    const T = (d: Float32Array, shape: number[]) => tensor(d, shape, { backend: be })
    const a = T(action(), [1, ACTION_DIM]), z = T(signed(ENC), [1, ENC])
    const k0 = await wide.kernel(a, T(new Float32Array(ACTION_DIM * WIDE_TAPS), shapes.ax), z, T(new Float32Array(ENC * WIDE_TAPS), shapes.zx)).read()
    expect(k0[R]).toBeCloseTo(Math.exp(4) / (Math.exp(4) + 6), 6)
    const k = await wide.kernel(a, T(signed(ACTION_DIM * WIDE_TAPS), shapes.ax), z, T(signed(ENC * WIDE_TAPS), shapes.zx)).read()
    expect(k.every(v => v >= 0)).toBe(true)
    expect(Math.abs(k.reduce((s, v) => s + v, 0) - 1)).toBeLessThan(1e-6)
    wide.dispose()
  })

  it("replays the oracle exactly: a one-hot kernel shifts a known frame by exactly k cells, the frame's edge repeated", async () => {
    const be = createCpuBackend()
    const wide = createWideShift(be)
    const src = image()
    const b = tensor(src, [1, PP], { backend: be })
    const clamp = (v: number, n: number) => Math.min(n - 1, Math.max(0, v))
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const got = await wide.translate(b, tensor(pulse(R + dx), [1, WIDE_TAPS], { backend: be }), tensor(pulse(R + dy), [1, WIDE_TAPS], { backend: be })).read()
      const want = new Float32Array(PP)
      for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) for (let c = 0; c < PC; c++) want[(y * PW + x) * PC + c] = src[(clamp(y + dy, PH) * PW + clamp(x + dx, PW)) * PC + c]!
      expect(got).toEqual(want) // bit-exact
    }
    wide.dispose()
  })

  it("its mask starts at sigmoid(2) in every cell and follows z·M + b", async () => {
    const be = createCpuBackend()
    const wide = createWideShift(be)
    const z = tensor(signed(ENC), [1, ENC], { backend: be })
    const m0 = await wide.mask(z, params(be, k => new Float32Array(size(k)))).read()
    expect(m0.length).toBe(CELLS)
    expect(m0.every(v => Math.abs(v - 1 / (1 + Math.exp(-2))) < 1e-6)).toBe(true)
    const b = signed(CELLS)
    const m = await wide.mask(z, params(be, k => (k === "maskB" ? b : new Float32Array(size(k))))).read()
    for (const i of [0, 17, CELLS - 1]) expect(m[i]).toBeCloseTo(1 / (1 + Math.exp(-(2 + b[i]!))), 5)
    wide.dispose()
  })

  it("a gain multiplies the learned kernel and mask logits before their priors are added", async () => {
    const be = createCpuBackend()
    const wide = createWideShift(be, { gain: 8 })
    const T = (d: Float32Array, shape: number[]) => tensor(d, shape, { backend: be })
    const z = T(new Float32Array(ENC), [1, ENC])
    const ax = new Float32Array(ACTION_DIM * WIDE_TAPS)
    ax[3 * WIDE_TAPS + R + 1] = 0.5 // one learned logit of 0.5 on the tap beside the centre: 8 × 0.5 = 4, level with the prior
    const k = await wide.kernel(T(action(), [1, ACTION_DIM]), T(ax, shapes.ax), z, T(new Float32Array(ENC * WIDE_TAPS), shapes.zx)).read()
    expect(k[R]).toBeCloseTo(k[R + 1]!, 6)
    expect(k[R]).toBeCloseTo(Math.exp(4) / (2 * Math.exp(4) + 5), 6)
    const b = new Float32Array(CELLS)
    b[5] = -0.25 // 8 × −0.25 = −2: cancels the mask prior
    const m = await wide.mask(z, params(be, key => (key === "maskB" ? b : new Float32Array(size(key))))).read()
    expect(m[5]).toBeCloseTo(0.5, 6)
    expect(m[6]).toBeCloseTo(1 / (1 + Math.exp(-2)), 6)
    for (const bad of [0, -1, Number.NaN]) expect(() => createWideShift(be, { gain: bad }), String(bad)).toThrow(RangeError)
    wide.dispose()
  })

  it("its term is exactly zero while both gates are zero (a newborn predicts copy-last), whatever the rest holds", async () => {
    const be = createCpuBackend()
    const wide = createWideShift(be)
    const p = params(be, k => (k === "agate" || k === "zgate" ? new Float32Array(size(k)) : signed(size(k))))
    const t = wide.term(tensor(action(), [1, ACTION_DIM], { backend: be }), tensor(signed(ENC), [1, ENC], { backend: be }), tensor(image(), [1, PP], { backend: be }), p)
    expect((await t.read()).every(v => v === 0)).toBe(true)
    wide.dispose()
  })

  it("with the gate at 1, the mask at 1 and a one-hot kernel, base + term is the shifted frame; where the mask is 0 nothing moves", async () => {
    const be = createCpuBackend()
    const wide = createWideShift(be)
    const src = image()
    const big = 40 // logits this large make the softmax one-hot and the sigmoid 0 or 1 in float32
    const a = action()
    const p = params(be, k => {
      const d = new Float32Array(size(k))
      if (k === "ay") d[3 * WIDE_TAPS + R + 2] = big // the move factor's row: two cells down
      if (k === "ax") d[3 * WIDE_TAPS + R] = big // and none across (without it the centre prior leaves a little blur)
      if (k === "agate") d[3] = 1
      if (k === "maskB") for (let i = 0; i < CELLS; i++) d[i] = i < CELLS / 2 ? big : -big // the top half moves
      return d
    })
    const t = await wide.term(tensor(a, [1, ACTION_DIM], { backend: be }), tensor(new Float32Array(ENC), [1, ENC], { backend: be }), tensor(src, [1, PP], { backend: be }), p).read()
    let worst = 0
    for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) for (let c = 0; c < PC; c++) {
      const i = (y * PW + x) * PC + c
      const moved = src[(Math.min(PH - 1, y + 2) * PW + x) * PC + c]!
      const want = y * PW + x < CELLS / 2 ? moved - src[i]! : 0
      worst = Math.max(worst, Math.abs(t[i]! - want))
    }
    expect(worst).toBeLessThan(1e-6)
    wide.dispose()
  })

  it("passes gradCheck on its term over every parameter, z, the action and the base", () => {
    const wide = createWideShift(getDefaultBackend())
    const err = gradCheck(
      (ax, ay, agate, zx, zy, zgate, maskW, maskB, z, a, b) => wide.term(a, z, b, { ax, ay, agate, zx, zy, zgate, maskW, maskB }),
      [
        ...KEYS.map(k => ({ data: signed(size(k)).map(v => (k === "maskW" || k === "maskB" ? 0.5 * v : v)), shape: shapes[k] })),
        { data: signed(ENC).map(v => 0.5 * v), shape: [1, ENC] },
        { data: signed(ACTION_DIM).map(v => 0.3 * v), shape: [1, ACTION_DIM] },
        { data: image(), shape: [1, PP] },
      ],
    )
    expect(err).toBeLessThan(5e-2)
    wide.dispose()
  }, 600_000)

  it("frees its constants on dispose and leaks nothing per call", () => {
    const be = createCpuBackend()
    const before = be.stats.liveBuffers
    const wide = createWideShift(be)
    const p = params(be, k => new Float32Array(size(k)))
    const held = [tensor(action(), [1, ACTION_DIM], { backend: be }), tensor(new Float32Array(ENC), [1, ENC], { backend: be }), tensor(image(), [1, PP], { backend: be })] as const
    const live = be.stats.liveBuffers
    const t = wide.term(held[0], held[1], held[2], p)
    expect(be.stats.liveBuffers).toBe(live + 1)
    t.dispose()
    for (const x of [...held, ...KEYS.map(k => p[k])]) x.dispose()
    wide.dispose()
    expect(be.stats.liveBuffers).toBe(before)
  })
})
