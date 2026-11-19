import { PRED_PERIPHERY } from "@blanktab/interface"
import { add, composite, gatherRows, matmul, mul, reshape, scale, sigmoid, softmax, sub, tensor } from "@blanktab/tensor"
import type { Backend, Tensor } from "@blanktab/tensor"

/*
 * The wide-view shift (Plan 06a1): the fovea-shift prior's counterpart for the periphery. Part of the screen can move as
 * one piece (a page scrolls), which a linear readout of the predictor's hidden layer cannot express. Here the next
 * periphery may be the current one translated by a separable kernel kx ⊗ ky, in the cells a mask selects:
 *   term = g · m ⊙ (translate(base; kx, ky) − base)
 * with kx, ky = softmax(gain·(a·A + z·Z) + centre prior), the gate g = a·Ag + z·Zg and the mask
 * m = sigmoid(gain·(z·M + b) + prior) per cell, where a is the action one-hot, z the predictor's hidden layer and gain
 * a fixed number (createWideShift; 1 unless given: see createFoveaShift on why a gain). Whether anything moves, by how much and
 * where are all learned; nothing here knows the world. Rows and columns that would come in from beyond the frame
 * repeat the frame's edge (the mask can turn them off).
 */

/** Kernel taps per axis: ±3 periphery-target cells (24 world px). */
export const WIDE_TAPS = 7
const R = (WIDE_TAPS - 1) / 2
/** Initial logit of the centre tap, as the fovea shift's: most of each axis's mass stays put before learning. */
const CENTRE_LOGIT = 4
/** Initial mask logit: sigmoid(2) ≈ 0.88, so at first the whole frame would move together. */
const MASK_LOGIT = 2
const PW = PRED_PERIPHERY.width, PH = PRED_PERIPHERY.height, PC = PRED_PERIPHERY.channels
const CELLS = PW * PH
const PP = CELLS * PC
/** gather HWC -> rows (y, c), columns x: the HWC index of each element of that layout. */
const HWC_TO_YCX = Int32Array.from({ length: PP }, (_, i) => {
  const y = Math.floor(i / (PC * PW)), c = Math.floor(i / PW) % PC, x = i % PW
  return (y * PW + x) * PC + c
})
/** gather rows (y, c), columns x -> HWC: the (y, c, x) index of each HWC element. */
const YCX_TO_HWC = Int32Array.from({ length: PP }, (_, i) => {
  const c = i % PC, x = Math.floor(i / PC) % PW, y = Math.floor(i / (PC * PW))
  return y * PC * PW + c * PW + x
})
/** [WIDE_TAPS, PW*PW] counts: kx · this = Bx [PW, PW], Bx[x', x] = Σ kx[j] over the taps j whose source column, clamped to the frame, is x'. */
const BAND_X = (() => {
  const m = new Float32Array(WIDE_TAPS * PW * PW)
  for (let x = 0; x < PW; x++) for (let j = 0; j < WIDE_TAPS; j++) {
    const xp = Math.min(PW - 1, Math.max(0, x + j - R))
    m[j * PW * PW + xp * PW + x]! += 1
  }
  return m
})()
/** [WIDE_TAPS, PH*PH] counts: ky · this = By [PH, PH], By[y, y'] = Σ ky[j] over the taps j whose source row, clamped, is y'. */
const BAND_Y = (() => {
  const m = new Float32Array(WIDE_TAPS * PH * PH)
  for (let y = 0; y < PH; y++) for (let j = 0; j < WIDE_TAPS; j++) {
    const yp = Math.min(PH - 1, Math.max(0, y + j - R))
    m[j * PH * PH + y * PH + yp]! += 1
  }
  return m
})()

/** The wide shift's trainable parameters (owned by the model, in its params), all zero-initialised. */
export interface WideShiftParams {
  /** Action → kernel logits per axis [ACTION_DIM, WIDE_TAPS] and gate [ACTION_DIM, 1]. */
  ax: Tensor; ay: Tensor; agate: Tensor
  /** z → kernel logits per axis [enc, WIDE_TAPS] and gate [enc, 1]. */
  zx: Tensor; zy: Tensor; zgate: Tensor
  /** z → mask logits per cell: [enc, cells] and [cells]. */
  maskW: Tensor; maskB: Tensor
}

/** The operator on one backend. It owns only constants, which `dispose` frees. Every method is a composite. */
export interface WideShift {
  /** One axis's kernel: softmax(gain·(action·a + z·zw) + centre prior), [1, WIDE_TAPS], sums to 1. */
  kernel(action: Tensor, a: Tensor, z: Tensor, zw: Tensor): Tensor
  /**
   * `base` [1, PRED_P] (HWC) translated by kx ⊗ ky, the frame's edge repeated beyond it:
   * out[y,x,c] = Σ_i Σ_j ky[i]·kx[j]·base[clamp(y + i − R), clamp(x + j − R), c].
   */
  translate(base: Tensor, kx: Tensor, ky: Tensor): Tensor
  /** The mask [cells, 1] in (0, 1): sigmoid(gain·(z·maskW + maskB) + prior). */
  mask(z: Tensor, p: WideShiftParams): Tensor
  /** The prediction term g · m ⊙ (translate(base) − base), [1, PRED_P]; 0 while the gate g is 0. */
  term(action: Tensor, z: Tensor, base: Tensor, p: WideShiftParams): Tensor
  dispose(): void
}

export function createWideShift(backend: Backend, opts: { gain?: number } = {}): WideShift {
  const gain = opts.gain ?? 1
  if (!(Number.isFinite(gain) && gain > 0)) throw new RangeError(`wide shift: gain must be finite and > 0, got ${gain}`)
  const gained = (x: Tensor) => (gain === 1 ? x : scale(x, gain))
  const consts: Tensor[] = []
  const constant = (d: Float32Array, shape: number[]) => {
    const t = tensor(d, shape, { backend })
    consts.push(t)
    return t
  }
  let centre: Tensor, maskPrior: Tensor, bandX: Tensor, bandY: Tensor
  try {
    const c = new Float32Array(WIDE_TAPS)
    c[R] = CENTRE_LOGIT
    centre = constant(c, [1, WIDE_TAPS])
    maskPrior = constant(new Float32Array(CELLS).fill(MASK_LOGIT), [1, CELLS])
    bandX = constant(BAND_X, [WIDE_TAPS, PW * PW])
    bandY = constant(BAND_Y, [WIDE_TAPS, PH * PH])
  } catch (e) {
    for (const t of consts) t.dispose()
    throw e
  }

  const kernel = (action: Tensor, a: Tensor, z: Tensor, zw: Tensor): Tensor =>
    composite(() => softmax(add(gained(add(matmul(action, a), matmul(z, zw))), centre)))
  const translate = (base: Tensor, kx: Tensor, ky: Tensor): Tensor => composite(() => {
    const bx = reshape(matmul(kx, bandX), [PW, PW])
    const by = reshape(matmul(ky, bandY), [PH, PH])
    // Columns first, on rows (y, c); those rows regroup as [y][(c, x)] for the row shift with no copy.
    const rows = reshape(gatherRows(reshape(base, [PP, 1]), HWC_TO_YCX), [PH * PC, PW])
    const moved = matmul(by, reshape(matmul(rows, bx), [PH, PC * PW]))
    return reshape(gatherRows(reshape(moved, [PP, 1]), YCX_TO_HWC), [1, PP])
  })
  const mask = (z: Tensor, p: WideShiftParams): Tensor =>
    composite(() => reshape(sigmoid(add(gained(add(matmul(z, p.maskW), p.maskB)), maskPrior)), [CELLS, 1]))

  return {
    kernel,
    translate,
    mask,
    term: (action, z, base, p) => composite(() => {
      const moved = translate(base, kernel(action, p.ax, z, p.zx), kernel(action, p.ay, z, p.zy))
      const masked = reshape(mul(reshape(sub(moved, base), [CELLS, PC]), mask(z, p)), [1, PP])
      return mul(masked, add(matmul(action, p.agate), matmul(z, p.zgate)))
    }),
    dispose() {
      for (const t of consts) t.dispose()
    },
  }
}
