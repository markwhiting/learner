import { FOVEA, KEYS, MOVE_OPTIONS, BUTTON_OPTIONS, SCROLL_OPTIONS, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"

export const ACTION_DIM = MOVE_OPTIONS + BUTTON_OPTIONS + SCROLL_OPTIONS + KEYS.length // 76

export interface Inputs { periphery: Float32Array; fovea: Float32Array; proprio: Float32Array }
export interface Targets { periphery: Float32Array; fovea: Float32Array }

type Dims = { width: number; height: number; channels: number }

function checkLength(what: string, src: Uint8Array, d: Dims): void {
  const want = d.width * d.height * d.channels
  if (src.length !== want) throw new RangeError(`${what}: expected ${want} bytes (${d.width}x${d.height}x${d.channels}), got ${src.length}`)
}

/** Cursor coordinates are normalized; anything outside [0,1] is clamped and a non-finite value becomes 0. */
const unit = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0)

// uint8 HWC -> float32 CHW in 0..1
function toChw(what: string, src: Uint8Array, d: Dims): Float32Array {
  checkLength(what, src, d)
  const plane = d.width * d.height
  const out = new Float32Array(plane * d.channels)
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < d.channels; c++) out[c * plane + p] = src[p * d.channels + c]! / 255
  }
  return out
}

// uint8 HWC -> float32 HWC in 0..1, averaged over non-overlapping 2x2 boxes
function boxDown2(what: string, src: Uint8Array, d: Dims): Float32Array {
  checkLength(what, src, d)
  const ow = d.width >> 1
  const oh = d.height >> 1
  const out = new Float32Array(ow * oh * d.channels)
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      for (let c = 0; c < d.channels; c++) {
        const at = (yy: number, xx: number) => src[(yy * d.width + xx) * d.channels + c]!
        const s = at(2 * y, 2 * x) + at(2 * y, 2 * x + 1) + at(2 * y + 1, 2 * x) + at(2 * y + 1, 2 * x + 1)
        out[(y * ow + x) * d.channels + c] = s / (4 * 255)
      }
    }
  }
  return out
}

/** Frames must match PERIPHERY/FOVEA exactly (RangeError otherwise); the cursor is clamped to [0,1], non-finite -> 0. */
export function obsToInputs(o: Observation): Inputs {
  return {
    periphery: toChw("periphery", o.periphery, PERIPHERY),
    fovea: toChw("fovea", o.fovea, FOVEA),
    proprio: Float32Array.of(unit(o.cursor.x), unit(o.cursor.y), o.buttonHeld ? 1 : 0),
  }
}

export function targetsFromObs(o: Observation): Targets {
  const t = { periphery: boxDown2("periphery", o.periphery, PERIPHERY), fovea: boxDown2("fovea", o.fovea, FOVEA) }
  if (t.periphery.length !== PRED_PERIPHERY.width * PRED_PERIPHERY.height * PRED_PERIPHERY.channels
    || t.fovea.length !== PRED_FOVEA.width * PRED_FOVEA.height * PRED_FOVEA.channels) {
    throw new Error("target size does not match PRED_* constants")
  }
  return t
}

export function actionOneHot(a: Action): Float32Array {
  const out = new Float32Array(ACTION_DIM)
  const hot = (offset: number, n: number, v: number) => {
    if (!Number.isInteger(v) || v < 0 || v >= n) throw new RangeError(`invalid action component ${v}`)
    out[offset + v] = 1
  }
  hot(0, MOVE_OPTIONS, a.move)
  hot(MOVE_OPTIONS, BUTTON_OPTIONS, a.button)
  hot(MOVE_OPTIONS + BUTTON_OPTIONS, SCROLL_OPTIONS, a.scroll)
  hot(MOVE_OPTIONS + BUTTON_OPTIONS + SCROLL_OPTIONS, KEYS.length, a.key)
  return out
}

/** copyLastBaseline by view: each target's own mean squared change. Their size-weighted mean is copyLastBaseline. */
export function copyLastByView(prev: Targets, next: Targets): { periphery: number; fovea: number } {
  let p = 0
  let f = 0
  for (let i = 0; i < prev.periphery.length; i++) p += (prev.periphery[i]! - next.periphery[i]!) ** 2
  for (let i = 0; i < prev.fovea.length; i++) f += (prev.fovea[i]! - next.fovea[i]!) ** 2
  return { periphery: p / prev.periphery.length, fovea: f / prev.fovea.length }
}

// Mean squared error over the concatenation of both targets: the "nothing changes" guess.
export function copyLastBaseline(prev: Targets, next: Targets): number {
  let sum = 0
  for (let i = 0; i < prev.periphery.length; i++) sum += (prev.periphery[i]! - next.periphery[i]!) ** 2
  for (let i = 0; i < prev.fovea.length; i++) sum += (prev.fovea[i]! - next.fovea[i]!) ** 2
  return sum / (prev.periphery.length + prev.fovea.length)
}
