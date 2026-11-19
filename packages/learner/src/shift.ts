import { PRED_FOVEA } from "@blanktab/interface"
import { add, composite, gatherRows, matmul, mul, reshape, scale, sigmoid, softmax, sub, tensor } from "@blanktab/tensor"
import type { Backend, Tensor } from "@blanktab/tensor"
import { SURROUND, SURROUND_MARGIN } from "./encode"

/*
 * The fovea-shift prior (decision 0015; docs/research/2026-10-21-why-not-learning.md). The fovea is a window centred
 * on the cursor, so a move translates its content, which a linear readout of the predictor's hidden layer cannot
 * express. This architectural prior (translation of the current image, as in CDNA, Finn et al. 2016) predicts the next
 * fovea as a convex, separable, action-chosen blend of shifted copies of the current one: kernel kx ⊗ ky, with
 * kx = softmax(a·Wx + c) and ky = softmax(a·Wy + c), applied as two banded matmuls with zero padding, then gated by
 * g = a·Wg. Nothing here knows the world or which action moves the cursor: the action → kernel map is learned.
 */

/** Kernel taps per axis: ±8 target pixels, the largest move (16 world px). */
export const SHIFT_TAPS = 17
const R = (SHIFT_TAPS - 1) / 2
/** Initial logit of the centre tap: e^4 / (e^4 + 16) ≈ 0.77 of each axis's mass stays put before learning. */
const CENTRE_LOGIT = 4
const FW = PRED_FOVEA.width, FH = PRED_FOVEA.height, FC = PRED_FOVEA.channels
const PF = FW * FH * FC
const ROWS = FC * FH // the fovea as a [FC*FH, FW] stack of planes
/** gather HWC -> CHW: the HWC index of each CHW element. */
const HWC_TO_CHW = Int32Array.from({ length: PF }, (_, i) => (i % (FW * FH)) * FC + Math.floor(i / (FW * FH)))
/** gather CHW -> HWC: the CHW index of each HWC element. */
const CHW_TO_HWC = Int32Array.from({ length: PF }, (_, i) => (i % FC) * FW * FH + Math.floor(i / FC))
/** [SHIFT_TAPS, FW*FW] 0/1: kx · this = Bx [FW, FW], Bx[x', x] = kx[x' − x + R]. */
const BAND_X = (() => {
  const m = new Float32Array(SHIFT_TAPS * FW * FW)
  for (let xp = 0; xp < FW; xp++) for (let x = 0; x < FW; x++) {
    const j = xp - x + R
    if (j >= 0 && j < SHIFT_TAPS) m[j * FW * FW + xp * FW + x] = 1
  }
  return m
})()
/** [SHIFT_TAPS, ROWS*ROWS] 0/1: ky · this = By [ROWS, ROWS], block-diagonal per channel, By[y, y'] = ky[y' − y + R]. */
const BAND_Y = (() => {
  const m = new Float32Array(SHIFT_TAPS * ROWS * ROWS)
  for (let c = 0; c < FC; c++) for (let y = 0; y < FH; y++) for (let yp = 0; yp < FH; yp++) {
    const j = yp - y + R
    if (j >= 0 && j < SHIFT_TAPS) m[j * ROWS * ROWS + (c * FH + y) * ROWS + c * FH + yp] = 1
  }
  return m
})()

if (SURROUND_MARGIN !== R) throw new Error(`shift: the surround's margin (${SURROUND_MARGIN}) must be the shift's reach (${R})`)
const SW = SURROUND.width, SH = SURROUND.height
const PS = SW * SH * FC
const SROWS = FC * SH // the surround as a [FC*SH, SW] stack of planes
/** gather HWC -> CHW for the surround. */
const S_HWC_TO_CHW = Int32Array.from({ length: PS }, (_, i) => (i % (SW * SH)) * FC + Math.floor(i / (SW * SH)))
/** [SHIFT_TAPS, SW*FW] 0/1: kx · this = Bx [SW, FW], Bx[x', x] = kx[x' − x] (x' a surround column, x a fovea column). */
const BAND_XS = (() => {
  const m = new Float32Array(SHIFT_TAPS * SW * FW)
  for (let xp = 0; xp < SW; xp++) for (let x = 0; x < FW; x++) {
    const j = xp - x
    if (j >= 0 && j < SHIFT_TAPS) m[j * SW * FW + xp * FW + x] = 1
  }
  return m
})()
/** [SHIFT_TAPS, ROWS*SROWS] 0/1: ky · this = By [ROWS, SROWS], block-diagonal per channel, By[y, y'] = ky[y' − y]. */
const BAND_YS = (() => {
  const m = new Float32Array(SHIFT_TAPS * ROWS * SROWS)
  for (let c = 0; c < FC; c++) for (let y = 0; y < FH; y++) for (let yp = 0; yp < SH; yp++) {
    const j = yp - y
    if (j >= 0 && j < SHIFT_TAPS) m[j * ROWS * SROWS + (c * FH + y) * SROWS + c * SH + yp] = 1
  }
  return m
})()

/** The shift's trainable parameters (owned by the model, in its params): kernel logits per axis and the gate. */
export interface ShiftParams {
  /** [ACTION_DIM, SHIFT_TAPS] */ x: Tensor
  /** [ACTION_DIM, SHIFT_TAPS] */ y: Tensor
  /** [ACTION_DIM, 1] */ gate: Tensor
  /**
   * The state gate (Plan 06a1, optional): the action's gate is multiplied by sigmoid(h·gate + bias), with h the core's
   * state [1, hidden], `gate` [hidden, 1] and `bias` [1, 1]. Whether an action shifts the view can depend on the
   * situation (a move against the screen's edge does not), and a product says "this action, here" directly.
   * Zero-initialised: the factor starts at one half everywhere.
   */
  state?: { gate: Tensor; bias: Tensor }
}

/** What else a prediction's shift may draw on (Plan 06a1). */
export interface ShiftContext {
  /** The fovea's surround [1, SURROUND] (HWC, encode.ts foveaSurround): what a move brings in comes from it, not black. */
  surround?: Tensor
  /** The core's state h [1, hidden], for ShiftParams.state. */
  h?: Tensor
}

/**
 * The fovea-shift operator on one backend. It owns only constants (the centre prior and the band matrices), which
 * `dispose` frees. Every method is a composite (packages/tensor/README.md rule 3): only its result survives.
 */
export interface FoveaShift {
  /**
   * One axis's kernel for `action` [1,ACTION_DIM]: softmax(action·w + centre prior), [1,SHIFT_TAPS], sums to 1. With
   * a gain (createFoveaShift), the learned logits are multiplied by it before the centre prior is added:
   * softmax(gain·(action·w) + centre prior).
   */
  kernel(action: Tensor, w: Tensor): Tensor
  /**
   * `base` [1,PRED_F] (HWC) translated by the separable kernel kx ⊗ ky [1,SHIFT_TAPS] each, zero-filled at the edges:
   * out[y,x,c] = Σ_i Σ_j ky[i]·kx[j]·base[y + i − R, x + j − R, c]. A one-hot kernel at tap R + k shifts by exactly k.
   */
  translate(base: Tensor, kx: Tensor, ky: Tensor): Tensor
  /**
   * `surround` [1,SURROUND] (HWC) translated by kx ⊗ ky and cropped to the fovea: as `translate` of its middle, except
   * that what comes in from beyond the fovea is the surround's content instead of zero:
   * out[y,x,c] = Σ_i Σ_j ky[i]·kx[j]·surround[y + i, x + j, c]. With a black margin it equals translate exactly.
   */
  translateSurround(surround: Tensor, kx: Tensor, ky: Tensor): Tensor
  /**
   * The prediction term g·(translate(base; kx, ky) − base), [1,PRED_F]; 0 while g = 0. The kernels are the action's
   * (kernel(a, p.x), kernel(a, p.y)) and the gate g is a·p.gate, times the state gate when `p.state` and `ctx.h` are
   * given. With `ctx.surround` the translation is translateSurround of it (its middle must be `base`).
   */
  term(action: Tensor, base: Tensor, p: ShiftParams, ctx?: ShiftContext): Tensor
  dispose(): void
}

/**
 * `surround`: also build translateSurround's constants (without it, translateSurround throws). `gain` (default 1, the
 * shift from before Plan 06a1, bit-exact): the kernels' learned logits count this many times over. An update moves a
 * weight by about the learning rate at most, and a kernel only sharpens once its logits outgrow the centre prior
 * (CENTRE_LOGIT), which at gain 1 takes thousands of updates of one action's row.
 */
export function createFoveaShift(backend: Backend, opts: { surround?: boolean; gain?: number } = {}): FoveaShift {
  const gain = opts.gain ?? 1
  if (!(Number.isFinite(gain) && gain > 0)) throw new RangeError(`shift: gain must be finite and > 0, got ${gain}`)
  const consts: Tensor[] = []
  const constant = (d: Float32Array, shape: number[]) => {
    const t = tensor(d, shape, { backend })
    consts.push(t)
    return t
  }
  let centre: Tensor, bandX: Tensor, bandY: Tensor
  let bandXS: Tensor | null = null, bandYS: Tensor | null = null
  try {
    const c = new Float32Array(SHIFT_TAPS)
    c[R] = CENTRE_LOGIT
    centre = constant(c, [1, SHIFT_TAPS])
    bandX = constant(BAND_X, [SHIFT_TAPS, FW * FW])
    bandY = constant(BAND_Y, [SHIFT_TAPS, ROWS * ROWS])
    if (opts.surround) {
      bandXS = constant(BAND_XS, [SHIFT_TAPS, SW * FW])
      bandYS = constant(BAND_YS, [SHIFT_TAPS, ROWS * SROWS])
    }
  } catch (e) {
    for (const t of consts) t.dispose()
    throw e
  }

  // At gain 1 the expression is exactly the one from before Plan 06a1 (bit-exact).
  const kernel = (action: Tensor, w: Tensor): Tensor => composite(() => {
    const learned = matmul(action, w)
    return softmax(add(gain === 1 ? learned : scale(learned, gain), centre))
  })
  // composite: the gathers of a constant base have no graph node and would otherwise leak (tensor README rule 2).
  const translate = (base: Tensor, kx: Tensor, ky: Tensor): Tensor => composite(() => {
    const bx = reshape(matmul(kx, bandX), [FW, FW])
    const by = reshape(matmul(ky, bandY), [ROWS, ROWS])
    const planes = reshape(gatherRows(reshape(base, [PF, 1]), HWC_TO_CHW), [ROWS, FW])
    const moved = matmul(by, matmul(planes, bx))
    return reshape(gatherRows(reshape(moved, [PF, 1]), CHW_TO_HWC), [1, PF])
  })

  const translateSurround = (surround: Tensor, kx: Tensor, ky: Tensor): Tensor => composite(() => {
    if (!bandXS || !bandYS) throw new Error("shift: translateSurround needs createFoveaShift(backend, { surround: true })")
    const bx = reshape(matmul(kx, bandXS), [SW, FW])
    const by = reshape(matmul(ky, bandYS), [ROWS, SROWS])
    const planes = reshape(gatherRows(reshape(surround, [PS, 1]), S_HWC_TO_CHW), [SROWS, SW])
    const moved = matmul(by, matmul(planes, bx))
    return reshape(gatherRows(reshape(moved, [PF, 1]), CHW_TO_HWC), [1, PF])
  })

  return {
    kernel,
    translate,
    translateSurround,
    term: (action, base, p, ctx) => composite(() => {
      const kx = kernel(action, p.x), ky = kernel(action, p.y)
      const moved = ctx?.surround ? translateSurround(ctx.surround, kx, ky) : translate(base, kx, ky)
      const here = p.state && ctx?.h ? sigmoid(add(matmul(ctx.h, p.state.gate), p.state.bias)) : null
      const gate = here ? mul(matmul(action, p.gate), here) : matmul(action, p.gate)
      return mul(sub(moved, base), gate)
    }),
    dispose() {
      for (const t of consts) t.dispose()
    },
  }
}
