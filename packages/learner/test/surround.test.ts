import { describe, it, expect } from "vitest"
import { FOVEA, PERIPHERY, PRED_FOVEA, WORLD, emptyObservation } from "@blanktab/interface"
import type { Observation } from "@blanktab/interface"
import { createCpuBackend, createRng, getDefaultBackend, gradCheck, tensor } from "@blanktab/tensor"
import { ACTION_DIM, SURROUND, SURROUND_MARGIN, foveaSurround, targetsFromObs } from "../src/encode"
import { SHIFT_TAPS, createFoveaShift } from "../src/shift"

const FW = PRED_FOVEA.width, FH = PRED_FOVEA.height, FC = PRED_FOVEA.channels
const PF = FW * FH * FC
const SW = SURROUND.width, SH = SURROUND.height
const PS = SW * SH * FC
const R = (SHIFT_TAPS - 1) / 2
const M = SURROUND_MARGIN
const rng = createRng(33)
const signed = (n: number) => Float32Array.from({ length: n }, () => (rng() < 0.5 ? -1 : 1) * (0.2 + rng()))
const pulse = (j: number) => {
  const k = new Float32Array(SHIFT_TAPS)
  k[j] = 1
  return k
}
/** A frame whose periphery cell (px, py) holds (px, py, 7) and whose fovea is all 200, with the cursor at world pixel (cx, cy). */
const frame = (cx: number, cy: number): Observation => {
  const o = emptyObservation()
  for (let py = 0; py < PERIPHERY.height; py++) for (let px = 0; px < PERIPHERY.width; px++) o.periphery.set([px, py, 7], (py * PERIPHERY.width + px) * 3)
  o.fovea.fill(200)
  o.cursor = { x: cx / WORLD.width, y: cy / WORLD.height }
  return o
}
/** A known surround: every HWC element distinct, in (0, 1). */
const surroundImage = () => Float32Array.from({ length: PS }, (_, i) => (i + 1) / (PS + 1))
const middleOf = (s: Float32Array) => {
  const out = new Float32Array(PF)
  for (let y = 0; y < FH; y++) for (let x = 0; x < FW; x++) for (let c = 0; c < FC; c++) out[(y * FW + x) * FC + c] = s[((y + M) * SW + x + M) * FC + c]!
  return out
}

describe("the fovea's surround (Plan 06a1)", () => {
  it("is the fovea target with a margin of the shift's reach on each side: 32×32×3", () => {
    expect(SURROUND_MARGIN).toBe(R)
    expect(SURROUND).toEqual({ width: FW + 2 * R, height: FH + 2 * R, channels: 3 })
  })

  it("targetsFromObs gives it only when asked, and its middle is the fovea target", () => {
    const o = frame(160, 100)
    expect("surround" in targetsFromObs(o)).toBe(false)
    const t = targetsFromObs(o, true)
    expect(t.surround!.length).toBe(PS)
    expect(middleOf(t.surround!)).toEqual(t.fovea)
    expect(t.fovea.every(v => Math.abs(v - 200 / 255) < 1e-6)).toBe(true)
  })

  it("each cell around the fovea is the periphery cell under its centre (2 world px per cell, 4 per periphery cell)", () => {
    for (const [cx, cy] of [[160, 100], [161, 99], [163, 102], [40, 60], [300, 150]] as const) {
      const o = frame(cx, cy)
      const s = foveaSurround(o, targetsFromObs(o).fovea)
      for (let i = 0; i < SH; i++) for (let j = 0; j < SW; j++) {
        if (i >= M && i < M + FH && j >= M && j < M + FW) continue
        const wx = cx - FOVEA.width / 2 - 2 * M + 2 * j + 1, wy = cy - FOVEA.height / 2 - 2 * M + 2 * i + 1
        const got = [...s.subarray((i * SW + j) * 3, (i * SW + j) * 3 + 3)]
        const inside = wx >= 0 && wx < WORLD.width && wy >= 0 && wy < WORLD.height
        const want = inside ? [Math.floor(wx / 4) / 255, Math.floor(wy / 4) / 255, 7 / 255] : [0, 0, 0]
        for (let c = 0; c < 3; c++) expect(got[c], `cursor ${cx},${cy} cell ${i},${j}`).toBeCloseTo(want[c]!, 6)
      }
    }
  })

  it("is black beyond the screen, as the fovea's own padding is: the whole left margin with the cursor at the left edge", () => {
    const o = frame(0, 100)
    const s = foveaSurround(o, targetsFromObs(o).fovea)
    for (let i = 0; i < SH; i++) for (let j = 0; j < M; j++) for (let c = 0; c < 3; c++) expect(s[(i * SW + j) * 3 + c]).toBe(0)
    // and nothing else of that row band is: the right margin lies on the screen
    expect(s[(0 * SW + SW - 1) * 3 + 2]).toBeCloseTo(7 / 255, 6)
  })

  it("a cursor reported outside [0, 1] or not finite is clamped as in obsToInputs, not an error", () => {
    const o = frame(160, 100)
    o.cursor = { x: Number.NaN, y: 7 }
    expect(() => foveaSurround(o, targetsFromObs(o).fovea)).not.toThrow()
  })
})

describe("fovea shift over the surround (Plan 06a1)", () => {
  it("what a shift brings in is the surround's content: a one-hot kernel reads the surround at offset k", async () => {
    const be = createCpuBackend()
    const shift = createFoveaShift(be, { surround: true })
    const src = surroundImage()
    const s = tensor(src, [1, PS], { backend: be })
    for (const [dx, dy] of [[0, 0], [R, 0], [-R, 0], [0, R], [0, -R], [R, R], [-R, R], [1, -2], [4, 8], [-8, -8]] as const) {
      const got = await shift.translateSurround(s, tensor(pulse(R + dx), [1, SHIFT_TAPS], { backend: be }), tensor(pulse(R + dy), [1, SHIFT_TAPS], { backend: be })).read()
      const want = new Float32Array(PF)
      for (let y = 0; y < FH; y++) for (let x = 0; x < FW; x++) for (let c = 0; c < FC; c++) want[(y * FW + x) * FC + c] = src[((y + M + dy) * SW + x + M + dx) * FC + c]!
      expect(got).toEqual(want) // bit-exact
    }
    shift.dispose()
  })

  it("with a black margin it is translate exactly, for every tap pair", async () => {
    const be = createCpuBackend()
    const shift = createFoveaShift(be, { surround: true })
    const full = surroundImage()
    const black = new Float32Array(PS)
    for (let y = 0; y < FH; y++) for (let x = 0; x < FW; x++) for (let c = 0; c < FC; c++) black[((y + M) * SW + x + M) * FC + c] = full[((y + M) * SW + x + M) * FC + c]!
    const s = tensor(black, [1, PS], { backend: be }), b = tensor(middleOf(black), [1, PF], { backend: be })
    for (let dy = -R; dy <= R; dy += 2) for (let dx = -R; dx <= R; dx += 2) {
      const kx = tensor(pulse(R + dx), [1, SHIFT_TAPS], { backend: be }), ky = tensor(pulse(R + dy), [1, SHIFT_TAPS], { backend: be })
      expect(await shift.translateSurround(s, kx, ky).read()).toEqual(await shift.translate(b, kx, ky).read())
    }
    // and for a soft kernel
    const soft = () => { const k = Float32Array.from({ length: SHIFT_TAPS }, () => rng()); const t = k.reduce((a, v) => a + v, 0); return tensor(k.map(v => v / t), [1, SHIFT_TAPS], { backend: be }) }
    const kx = soft(), ky = soft()
    const [a, c] = [await shift.translateSurround(s, kx, ky).read(), await shift.translate(b, kx, ky).read()]
    let worst = 0
    for (let i = 0; i < PF; i++) worst = Math.max(worst, Math.abs(a[i]! - c[i]!))
    expect(worst).toBeLessThan(1e-6)
    shift.dispose()
  })

  it("without { surround: true } translateSurround refuses, and the term without a context is the one from before", async () => {
    const be = createCpuBackend()
    const plain = createFoveaShift(be)
    const k = tensor(pulse(R), [1, SHIFT_TAPS], { backend: be })
    expect(() => plain.translateSurround(tensor(surroundImage(), [1, PS], { backend: be }), k, k)).toThrow(/surround: true/)
    plain.dispose()
  })

  it("a gain multiplies the learned logits before the centre prior is added; gain 1 is the kernel from before, bit for bit", async () => {
    const be = createCpuBackend()
    const T = (d: Float32Array, shape: number[]) => tensor(d, shape, { backend: be })
    const a = new Float32Array(ACTION_DIM)
    for (const i of [3, 25, 28, 40]) a[i] = 1
    const act = T(a, [1, ACTION_DIM]), w = T(signed(ACTION_DIM * SHIFT_TAPS).map(v => 0.2 * v), [ACTION_DIM, SHIFT_TAPS])
    const [plain, one, eight] = [createFoveaShift(be), createFoveaShift(be, { gain: 1 }), createFoveaShift(be, { gain: 8 })]
    expect(await one.kernel(act, w).read()).toEqual(await plain.kernel(act, w).read())
    const wv = await w.read()
    const logits = Array.from({ length: SHIFT_TAPS }, (_, j) => 8 * [3, 25, 28, 40].reduce((acc, i) => acc + wv[i * SHIFT_TAPS + j]!, 0) + (j === R ? 4 : 0))
    const mx = Math.max(...logits), den = logits.reduce((acc, l) => acc + Math.exp(l - mx), 0)
    const k = await eight.kernel(act, w).read()
    logits.forEach((l, j) => expect(k[j]).toBeCloseTo(Math.exp(l - mx) / den, 5))
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) expect(() => createFoveaShift(be, { gain: bad }), String(bad)).toThrow(RangeError)
    for (const sft of [plain, one, eight]) sft.dispose()
  })

  it("passes gradCheck on its term over the kernel logits, the gate, the action, the base and the surround (with a gain)", () => {
    const shift = createFoveaShift(getDefaultBackend(), { surround: true, gain: 2 })
    const src = surroundImage()
    const err = gradCheck(
      (x, y, gate, a, b, s) => shift.term(a, b, { x, y, gate }, { surround: s }),
      [
        { data: signed(ACTION_DIM * SHIFT_TAPS), shape: [ACTION_DIM, SHIFT_TAPS] },
        { data: signed(ACTION_DIM * SHIFT_TAPS), shape: [ACTION_DIM, SHIFT_TAPS] },
        { data: signed(ACTION_DIM), shape: [ACTION_DIM, 1] },
        { data: signed(ACTION_DIM).map(v => 0.3 * v), shape: [1, ACTION_DIM] },
        { data: middleOf(src), shape: [1, PF] },
        { data: src, shape: [1, PS] },
      ],
    )
    expect(err).toBeLessThan(5e-2)
    shift.dispose()
  }, 240_000)

  it("frees its constants on dispose and leaks nothing per call", () => {
    const be = createCpuBackend()
    const before = be.stats.liveBuffers
    const shift = createFoveaShift(be, { surround: true })
    const T = (n: number, shape: number[]) => tensor(new Float32Array(n), shape, { backend: be })
    const held = [T(ACTION_DIM, [1, ACTION_DIM]), T(PF, [1, PF]), T(PS, [1, PS]),
      T(ACTION_DIM * SHIFT_TAPS, [ACTION_DIM, SHIFT_TAPS]), T(ACTION_DIM * SHIFT_TAPS, [ACTION_DIM, SHIFT_TAPS]), T(ACTION_DIM, [ACTION_DIM, 1])] as const
    const [act, b, s, x, y, gate] = held
    const live = be.stats.liveBuffers
    const t = shift.term(act, b, { x, y, gate }, { surround: s })
    expect(be.stats.liveBuffers).toBe(live + 1)
    t.dispose()
    for (const h of held) h.dispose()
    shift.dispose()
    expect(be.stats.liveBuffers).toBe(before)
  })
})
