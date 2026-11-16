// Filter sheets as PNGs, for looking at (plan 05c Task 7 Step 3); never part of the analysis.
import { mkdirSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { createCanvas } from "@napi-rs/canvas"
import type { FirstLayer, Level, ProbeLife } from "./lives.ts"

const K = 5, SCALE = 8, GAP = 2, GRID = 4
type State = "init" | Level

/** A 4×4 sheet of the 16 periphery filters (5×5 RGB, each scaled to its own min..max), as PNG bytes. */
function sheet(f: FirstLayer): Buffer {
  const cell = K * SCALE, side = GRID * cell + (GRID + 1) * GAP
  const c = createCanvas(side, side)
  const ctx = c.getContext("2d")
  const img = ctx.createImageData(side, side)
  img.data.fill(255)
  for (let u = 0; u < 16; u++) {
    const w = f.periphery.w.subarray(u * 3 * K * K, (u + 1) * 3 * K * K)
    let lo = Infinity, hi = -Infinity
    for (const v of w) { if (v < lo) lo = v; if (v > hi) hi = v }
    const span = hi > lo ? hi - lo : 1
    const x0 = GAP + (u % GRID) * (cell + GAP), y0 = GAP + Math.floor(u / GRID) * (cell + GAP)
    for (let y = 0; y < cell; y++) for (let x = 0; x < cell; x++) {
      const ky = Math.floor(y / SCALE), kx = Math.floor(x / SCALE), d = ((y0 + y) * side + x0 + x) * 4
      for (let ch = 0; ch < 3; ch++) img.data[d + ch] = Math.round(255 * (w[(ch * K + ky) * K + kx]! - lo) / span)
      img.data[d + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return c.toBuffer("image/png")
}

/** One 4×4 sheet of `life`'s periphery filters per state, `<learnerId>-<state>.png` under `dir`; returns the paths. */
export async function writeFilterPngs(life: ProbeLife, states: readonly State[], dir: string): Promise<string[]> {
  mkdirSync(dir, { recursive: true })
  return states.map(s => {
    const p = join(dir, `${life.learnerId}-${s}.png`)
    writeFileSync(p, sheet(s === "init" ? life.init : life.at[s]))
    return p
  })
}
