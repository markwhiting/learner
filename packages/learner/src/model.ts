import { FOVEA, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY } from "@blanktab/interface"
import { add, conv2d, createRng, gruCell, leakyRelu, linear, relu, reshape, tanh, tensor, zeros } from "@blanktab/tensor"
import type { Backend, GruParams, Tensor } from "@blanktab/tensor"
import { ACTION_DIM } from "./encode"
import { SHIFT_TAPS, createFoveaShift } from "./shift"
import { WIDE_TAPS, createWideShift } from "./wide-shift"

export { SHIFT_TAPS }

export interface Widths { conv1: number; conv2: number; enc: number; hidden: number }
/** WebGPU widths (spec §5.5). */
export const GPU_WIDTHS: Widths = { conv1: 16, conv2: 32, enc: 128, hidden: 256 }
/** CPU fallback: half width. */
export const CPU_WIDTHS: Widths = { conv1: 8, conv2: 16, enc: 64, hidden: 128 }
/**
 * Optional mechanisms of the predictor (Plan 06a1). Each is off unless asked for; a model with none of them is exactly
 * the model from before. Those that add parameters add them after every earlier parameter and before the output
 * layers, all zero-initialised (nothing is drawn from the seed for them, so the other weights start the same).
 */
export interface ModelOptions {
  /** The fovea shift draws on the fovea's surround: predict's base must carry `surround`. No parameters. */
  foveaSurround?: boolean
  /** The fovea shift's kernels and gate also depend on the predictor's hidden layer z (ShiftParams.state). */
  stateShift?: boolean
  /** The wide-view shift (./wide-shift.ts). */
  wideShift?: boolean
  /** The gain on both shifts' learned kernel (and mask) logits; absent or 1: none (createFoveaShift). No parameters. */
  shiftGain?: number
}

/**
 * Indices in Model.params of the first layer's filters (spec §5.1, the harness's meta.filters): periphery conv1 w
 * [conv1, 75] and b [conv1, 1], then fovea conv1 w [conv1, 27] and b [conv1, 1]. 104·conv1 values in all.
 */
export const FIRST_LAYER_PARAMS: readonly number[] = Object.freeze([0, 1, 6, 7])

/**
 * The learner's network. `core`, `heads` and `predict` build autograd graph nodes (every parameter
 * requires grad), so a training step frees them with `disposeGraph`; acting/inference must run inside
 * `noGrad(() => composite(...))` (or dispose every result), per packages/tensor/README.md.
 */
export interface Model {
  /**
   * Every trainable tensor, in creation order. This order is the checkpoint format: a saved state is
   * these tensors' data in this order, so never reorder, insert or remove parameters without a format bump.
   */
  readonly params: readonly Tensor[]
  /** The predictor's zero-initialised residual output layers (headP w/b, headF w/b): the last four of `params`. */
  readonly outputParams: readonly Tensor[]
  /**
   * Every parameter that starts at zero and expresses deviation from copy-last: the shift gate (and the optional
   * mechanisms' gates), then outputParams. The learner's residualDecay pulls these back toward zero.
   */
  readonly residualParams: readonly Tensor[]
  readonly paramCount: number
  readonly widths: Widths
  /**
   * Encode the observation and advance the GRU: returns h' [1,hidden].
   * Inputs: periphery [3,50,80] and fovea [3,32,32] (CHW, 0..1, see obsToInputs), proprio [1,3]; h [1,hidden].
   */
  core(inp: { periphery: Tensor; fovea: Tensor; proprio: Tensor }, h: Tensor): Tensor
  /** Policy logits [1,76] (the four action factors concatenated, see ACTION_DIM) and value [1,1]. */
  heads(h: Tensor): { logits: Tensor; value: Tensor }
  /**
   * Residual next-frame prediction conditioned on the chosen action one-hot [1,76]: for each output,
   * base + tanh(linear(z)) with z = leakyRelu(linear(h) + linear(action)), where `base` is the current frame's
   * targets (targetsFromObs of the observation the core just saw): periphery [1,3000], fovea [1,768] (HWC, 0..1). The fovea also gets an action-conditioned
   * translation of its base: + g(a)·(shift(base; kx(a) ⊗ ky(a)) − base), see ./shift.ts. The output layers and the gate
   * g start at zero, so a fresh model predicts exactly copy-last (no change) and learns only the change.
   * With options.foveaSurround, `base.surround` [1, SURROUND] (the fovea's surround, encode.ts) is required and the
   * translation draws on it. With options.wideShift the periphery also gets the wide-view shift's term.
   */
  predict(h: Tensor, action: Tensor, base: { periphery: Tensor; fovea: Tensor; surround?: Tensor }): { periphery: Tensor; fovea: Tensor }
  /** [1,hidden] zeros, caller-owned. */
  zeroState(): Tensor
  /** Frees every parameter (and any gradient on it). The model must not be used afterwards. */
  dispose(): void
}

const PROPRIO = 3
const outDim = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1
const PRED_P = PRED_PERIPHERY.width * PRED_PERIPHERY.height * PRED_PERIPHERY.channels
const PRED_F = PRED_FOVEA.width * PRED_FOVEA.height * PRED_FOVEA.channels

/**
 * Builds the model with every weight drawn from `seed`, or, given `initial` (a checkpoint's parameter values in
 * Model.params order), with copies of those values. A checkpoint that does not fit these widths throws RangeError
 * and leaks nothing.
 */
export function createModel(backend: Backend, seed: number, widths: Widths, initial?: readonly Float32Array[], options: ModelOptions = {}): Model {
  const params: Tensor[] = []
  let model: Model | undefined
  try {
    model = buildModel(backend, seed, widths, initial, params, options)
    if (initial && initial.length !== params.length) throw new RangeError(`createModel: ${initial.length} saved parameters, the model has ${params.length}`)
    return model
  } catch (e) {
    // A built model owns constants as well as params: dispose it whole. A partial build owns only params so far.
    if (model) model.dispose()
    else for (const t of params) t.dispose()
    throw e
  }
}

function buildModel(backend: Backend, seed: number, widths: Widths, initial: readonly Float32Array[] | undefined, params: Tensor[], options: ModelOptions): Model {
  const { conv1, conv2, enc, hidden } = widths
  const rng = createRng(seed)
  /** The saved values for the next parameter (a copy is uploaded), or null when drawing fresh weights. */
  const saved = (shape: number[]): Float32Array | null => {
    if (!initial) return null
    const i = params.length
    const n = shape.reduce((a, d) => a * d, 1)
    const v = initial[i]
    if (!v || v.length !== n) throw new RangeError(`createModel: saved parameter ${i} has ${v?.length ?? "no"} values, shape [${shape.join(",")}] needs ${n}`)
    return v
  }
  /** Trainable, uniform in ±1/sqrt(fanIn), drawn in creation order. */
  const p = (shape: number[], fanIn: number): Tensor => {
    const lim = 1 / Math.sqrt(fanIn)
    const n = shape.reduce((a, d) => a * d, 1)
    const data = saved(shape) ?? Float32Array.from({ length: n }, () => (2 * rng() - 1) * lim)
    const t = tensor(data, shape, { requiresGrad: true, backend })
    params.push(t)
    return t
  }
  /** Trainable, all zeros; draws nothing from the rng. */
  const p0 = (shape: number[]): Tensor => {
    const v = saved(shape)
    const t = v ? tensor(v, shape, { requiresGrad: true, backend }) : zeros(shape, { requiresGrad: true, backend })
    params.push(t)
    return t
  }
  const dense = (i: number, o: number) => ({ w: p([i, o], i), b: p([o], i) })
  const conv = (ci: number, co: number, k: number) => ({ w: p([co, ci * k * k], ci * k * k), b: p([co, 1], ci * k * k) })

  // Periphery: 50x80 -> 25x40 -> 13x20. Fovea: 32x32 -> 16x16 -> 8x8.
  const pH = outDim(outDim(PERIPHERY.height, 5, 2, 2), 3, 2, 1)
  const pW = outDim(outDim(PERIPHERY.width, 5, 2, 2), 3, 2, 1)
  const fH = outDim(outDim(FOVEA.height, 3, 2, 1), 3, 2, 1)
  const fW = outDim(outDim(FOVEA.width, 3, 2, 1), 3, 2, 1)
  const pc1 = conv(3, conv1, 5)
  const pc2 = conv(conv1, conv2, 3)
  const pLin = dense(conv2 * pH * pW, enc)
  const fc1 = conv(3, conv1, 3)
  const fc2 = conv(conv1, conv2, 3)
  const fLin = dense(conv2 * fH * fW, enc)
  const proprioLin = dense(PROPRIO, enc)
  const g = (i: number, o: number) => ({ w: p([i, o], o), u: p([o, o], o), b: p([o], o) })
  const [gz, gr, gh] = [g(enc, hidden), g(enc, hidden), g(enc, hidden)]
  const gru: GruParams = { wz: gz.w, uz: gz.u, bz: gz.b, wr: gr.w, ur: gr.u, br: gr.b, wh: gh.w, uh: gh.u, bh: gh.b }
  const headPolicy = dense(hidden, ACTION_DIM)
  const headValue = dense(hidden, 1)
  // The fovea-shift prior (./shift.ts): action -> kernel logits per axis and gate, all zero-initialised.
  const shiftX = p0([ACTION_DIM, SHIFT_TAPS])
  const shiftY = p0([ACTION_DIM, SHIFT_TAPS])
  const shiftGate = p0([ACTION_DIM, 1])
  const predH = dense(hidden, enc)
  const predA = dense(ACTION_DIM, enc)
  // Plan 06a1's optional mechanisms: created only when on, zero-initialised, before the output layers.
  const shiftState = options.stateShift ? { x: p0([enc, SHIFT_TAPS]), y: p0([enc, SHIFT_TAPS]), gate: p0([enc, 1]) } : undefined
  const wideParams = options.wideShift
    ? {
        ax: p0([ACTION_DIM, WIDE_TAPS]), ay: p0([ACTION_DIM, WIDE_TAPS]), agate: p0([ACTION_DIM, 1]),
        zx: p0([enc, WIDE_TAPS]), zy: p0([enc, WIDE_TAPS]), zgate: p0([enc, 1]),
        maskW: p0([enc, PRED_PERIPHERY.width * PRED_PERIPHERY.height]), maskB: p0([PRED_PERIPHERY.width * PRED_PERIPHERY.height]),
      }
    : null
  // Residual output layers, zero-initialised (weights and biases): the predictor starts at copy-last.
  const headP = { w: p0([enc, PRED_P]), b: p0([PRED_P]) }
  const headF = { w: p0([enc, PRED_F]), b: p0([PRED_F]) }

  // The fovea-shift operator's constants (caller-owned): freed in dispose, or by createModel if this build fails later.
  const gain = options.shiftGain ?? 1
  const shift = createFoveaShift(backend, { surround: options.foveaSurround === true, gain })
  const shiftParams = { x: shiftX, y: shiftY, gate: shiftGate, ...(shiftState ? { state: shiftState } : {}) }
  let wide: ReturnType<typeof createWideShift> | null = null
  try {
    if (wideParams) wide = createWideShift(backend, { gain })
  } catch (e) {
    shift.dispose()
    throw e
  }
  const optionalGates = [...(shiftState ? [shiftState.gate] : []), ...(wideParams ? [wideParams.agate, wideParams.zgate] : [])]

  return {
    params,
    outputParams: [headP.w, headP.b, headF.w, headF.b],
    residualParams: [shiftGate, ...optionalGates, headP.w, headP.b, headF.w, headF.b],
    paramCount: params.reduce((a, t) => a + t.size, 0),
    widths,

    core(inp, h) {
      const ep1 = relu(conv2d(inp.periphery, pc1.w, pc1.b, { kh: 5, kw: 5, stride: 2, pad: 2 }))
      const ep2 = relu(conv2d(ep1, pc2.w, pc2.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ep = linear(reshape(ep2, [1, -1]), pLin.w, pLin.b)
      const ef1 = relu(conv2d(inp.fovea, fc1.w, fc1.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ef2 = relu(conv2d(ef1, fc2.w, fc2.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ef = linear(reshape(ef2, [1, -1]), fLin.w, fLin.b)
      const coreIn = add(add(ep, ef), linear(inp.proprio, proprioLin.w, proprioLin.b))
      return gruCell(coreIn, h, gru)
    },

    heads(h) {
      return { logits: linear(h, headPolicy.w, headPolicy.b), value: linear(h, headValue.w, headValue.b) }
    },

    predict(h, action, base) {
      // Leaky (decision 0015 §2): with plain ReLU every z unit died once there was nothing for it to learn, and the
      // gradient into the output layers stayed exactly 0 for the rest of the life.
      const z = leakyRelu(add(linear(h, predH.w, predH.b), linear(action, predA.w, predA.b)))
      if (options.foveaSurround && !base.surround) throw new Error("model: predict needs base.surround (options.foveaSurround)")
      // No context at all without the options: the shift term is then exactly the one from before Plan 06a1.
      const ctx = options.foveaSurround || shiftState
        ? { ...(options.foveaSurround ? { surround: base.surround! } : {}), ...(shiftState ? { z } : {}) }
        : undefined
      const moved = wide && wideParams ? add(base.periphery, wide.term(action, z, base.periphery, wideParams)) : base.periphery
      return {
        periphery: add(moved, tanh(linear(z, headP.w, headP.b))),
        fovea: add(add(base.fovea, shift.term(action, base.fovea, shiftParams, ctx)), tanh(linear(z, headF.w, headF.b))),
      }
    },

    zeroState() {
      return zeros([1, hidden], { backend })
    },

    dispose() {
      shift.dispose()
      wide?.dispose()
      for (const t of params) {
        t.grad?.dispose()
        t.grad = null
        t.dispose()
      }
    },
  }
}
