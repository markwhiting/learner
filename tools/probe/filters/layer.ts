// The learner's first layer (conv1 + ReLU) recomputed in float64 on the probe frames, and the streamed centred Gram
// matrix over every (life, state) unit column (plan 05c Task 3). Only the Gram is kept, never the activations (about
// 2.4 GB at 37 lives): pass 1 takes every column's mean, pass 2 accumulates the centred cross-products.
import type { Encoder, Filters, ProbeLife } from "./lives.ts"

export interface LayerShape { k: number; stride: number; pad: number; inH: number; inW: number; outH: number; outW: number }
const out = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1
const shape = (k: number, stride: number, pad: number, inH: number, inW: number): LayerShape =>
  ({ k, stride, pad, inH, inW, outH: out(inH, k, stride, pad), outW: out(inW, k, stride, pad) })
/** packages/learner/src/model.ts: periphery 5×5 s2 p2 on 50×80 → 25×40; fovea 3×3 s2 p1 on 32×32 → 16×16. */
export const LAYER: Record<Encoder, LayerShape> = Object.freeze({ periphery: shape(5, 2, 2, 50, 80), fovea: shape(3, 2, 1, 32, 32) })
const CHANNELS = 3

/**
 * The input's patches, [positions, 3·k·k] row-major, float64, in the tensor package's im2col row order
 * ((c·k + ky)·k + kx), zero outside the frame. `x` is CHW in 0..1 (obsToInputs).
 */
export function patches(x: Float32Array, s: LayerShape): Float64Array {
  const K = CHANNELS * s.k * s.k, P = s.outH * s.outW
  const pt = new Float64Array(P * K)
  for (let oy = 0; oy < s.outH; oy++) for (let ox = 0; ox < s.outW; ox++) {
    const row = (oy * s.outW + ox) * K
    for (let c = 0; c < CHANNELS; c++) for (let ky = 0; ky < s.k; ky++) {
      const iy = oy * s.stride - s.pad + ky
      if (iy < 0 || iy >= s.inH) continue
      for (let kx = 0; kx < s.k; kx++) {
        const ix = ox * s.stride - s.pad + kx
        if (ix < 0 || ix >= s.inW) continue
        pt[row + (c * s.k + ky) * s.k + kx] = x[(c * s.inH + iy) * s.inW + ix]!
      }
    }
  }
  return pt
}

/** relu(w · patches + b) into `dst` at row offset `row0`: [units, positions], float64. */
function applyLayer(pt: Float64Array, f: Filters, K: number, P: number, dst: Float64Array, row0: number): void {
  const units = f.b.length
  for (let u = 0; u < units; u++) {
    const b = f.b[u]!, wo = u * K, d = (row0 + u) * P
    for (let p = 0; p < P; p++) {
      const po = p * K
      let acc = b
      for (let k = 0; k < K; k++) acc += f.w[wo + k]! * pt[po + k]!
      dst[d + p] = acc > 0 ? acc : 0
    }
  }
}

/** relu(conv1(x)) as [units, outH·outW], float64; the weight order is the learner's (the tensor package's im2col). */
export function firstLayer(x: Float32Array, f: Filters, s: LayerShape): Float64Array {
  const P = s.outH * s.outW, K = CHANNELS * s.k * s.k
  const a = new Float64Array(f.b.length * P)
  applyLayer(patches(x, s), f, K, P, a, 0)
  return a
}

/**
 * The centred Gram over `cols` = Σ units columns (state-major: state i's units are columns 16i … 16i + 15): g[i·cols + j]
 * = Σ over every (frame, position) sample of (a_i − mean_i)(a_j − mean_j). `varZero[i]` is 1 when column i is the same
 * value on every sample (a dead unit); its row and column of g are exactly 0.
 */
export interface Gram { cols: number; g: Float64Array; means: Float64Array; varZero: Uint8Array; samples: number }

/**
 * The streamed two-pass centred Gram of every state's first-layer activations on `frames` (CHW inputs). Each g entry is
 * summed in one fixed order (frames in order, positions in order), so the result is bit-identical run to run.
 */
export function centredGram(frames: readonly Float32Array[], states: readonly Filters[], s: LayerShape): Gram {
  const P = s.outH * s.outW, K = CHANNELS * s.k * s.k
  const units = states.map(f => f.b.length)
  const cols = units.reduce((a, u) => a + u, 0)
  const row0 = units.map((_, i) => units.slice(0, i).reduce((a, u) => a + u, 0))
  const A = new Float64Array(cols * P)
  const activations = (x: Float32Array) => {
    const pt = patches(x, s)
    states.forEach((f, i) => applyLayer(pt, f, K, P, A, row0[i]!))
  }
  // Pass 1: means, and which columns never change.
  const sums = new Float64Array(cols), lo = new Float64Array(cols).fill(Infinity), hi = new Float64Array(cols).fill(-Infinity)
  for (const x of frames) {
    activations(x)
    for (let c = 0; c < cols; c++) {
      let sum = 0, mn = lo[c]!, mx = hi[c]!
      for (let p = c * P, e = p + P; p < e; p++) { const v = A[p]!; sum += v; if (v < mn) mn = v; if (v > mx) mx = v }
      sums[c]! += sum; lo[c] = mn; hi[c] = mx
    }
  }
  const samples = frames.length * P
  const means = Float64Array.from(sums, v => v / samples)
  const varZero = Uint8Array.from(means, (_, c) => (lo[c] === hi[c] ? 1 : 0))
  // Pass 2: centred cross-products, upper triangle, then mirrored.
  const g = new Float64Array(cols * cols)
  for (const x of frames) {
    activations(x)
    for (let c = 0; c < cols; c++) {
      const m = means[c]!, z = varZero[c] === 1
      for (let p = c * P, e = p + P; p < e; p++) A[p] = z ? 0 : A[p]! - m
    }
    accumulateUpper(A, cols, P, g)
  }
  for (let i = 0; i < cols; i++) for (let j = 0; j < i; j++) g[i * cols + j] = g[j * cols + i]!
  return { cols, g, means, varZero, samples }
}

/**
 * g[i, j] += Σ_p A[i, p]·A[j, p] for j ≥ i, in 2×4 register blocks. Each entry is one sequential sum over p with its own
 * accumulator, so the blocking never changes a bit of the result.
 */
function accumulateUpper(A: Float64Array, cols: number, P: number, g: Float64Array): void {
  let i = 0
  for (; i + 1 < cols; i += 2) {
    const a0 = i * P, a1 = a0 + P
    // The pair's own corner: (i, i), (i, i+1), (i+1, i+1).
    let c00 = 0, c01 = 0, c11 = 0
    for (let p = 0; p < P; p++) { const u = A[a0 + p]!, v = A[a1 + p]!; c00 += u * u; c01 += u * v; c11 += v * v }
    g[i * cols + i]! += c00; g[i * cols + i + 1]! += c01; g[(i + 1) * cols + i + 1]! += c11
    let j = i + 2
    for (; j + 3 < cols; j += 4) {
      const b0 = j * P, b1 = b0 + P, b2 = b1 + P, b3 = b2 + P
      let s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0
      for (let p = 0; p < P; p++) {
        const u = A[a0 + p]!, v = A[a1 + p]!
        const w0 = A[b0 + p]!, w1 = A[b1 + p]!, w2 = A[b2 + p]!, w3 = A[b3 + p]!
        s00 += u * w0; s01 += u * w1; s02 += u * w2; s03 += u * w3
        s10 += v * w0; s11 += v * w1; s12 += v * w2; s13 += v * w3
      }
      const r0 = i * cols + j, r1 = r0 + cols
      g[r0]! += s00; g[r0 + 1]! += s01; g[r0 + 2]! += s02; g[r0 + 3]! += s03
      g[r1]! += s10; g[r1 + 1]! += s11; g[r1 + 2]! += s12; g[r1 + 3]! += s13
    }
    for (; j < cols; j++) {
      const b = j * P
      let s0 = 0, s1 = 0
      for (let p = 0; p < P; p++) { const w = A[b + p]!; s0 += A[a0 + p]! * w; s1 += A[a1 + p]! * w }
      g[i * cols + j]! += s0; g[(i + 1) * cols + j]! += s1
    }
  }
  for (; i < cols; i++) {
    const a0 = i * P
    let s0 = 0
    for (let p = 0; p < P; p++) s0 += A[a0 + p]! * A[a0 + p]!
    g[i * cols + i]! += s0
  }
}

/**
 * Each encoder's centred Gram over the lives' states in stateOf order (life i: init as state 2i, its 2,048 CU state as
 * 2i + 1), on that encoder's probe inputs (CHW, obsToInputs).
 */
export function probeGrams(lives: readonly ProbeLife[], inputs: Record<Encoder, readonly Float32Array[]>): Record<Encoder, Gram> {
  const states = (e: Encoder) => lives.flatMap(l => [l.init[e], l.at[2048][e]])
  return { periphery: centredGram(inputs.periphery, states("periphery"), LAYER.periphery), fovea: centredGram(inputs.fovea, states("fovea"), LAYER.fovea) }
}
