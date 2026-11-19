// Offline diagnostic for docs/research/2026-11-19-what-the-predictor-can-learn.md §7 (not part of the product): can the
// room in front be read from the core's recurrent state h by a linear map? The cursor's position is read the same way
// as a control (it is an input, so a readout that works should find it).
//   npx tsx tools/probe/introspect-probe.ts --seed 1 --steps 25000 … --dumpParams params.json
//   npx tsx tools/probe/room-readout.ts params.json [--widths gpu]
// The fresh-world set (decision 0030) supplies the frames: the core runs along each trajectory from the zero state,
// a ridge regression is fitted on half the trajectories and scored on the other half (every room on both sides). The room is privileged (the
// harness's label of each trajectory's starting room; the few transitions after a room switch are dropped).
import { readFileSync } from "node:fs"
import { createCanvas } from "@napi-rs/canvas"
import { CPU_WIDTHS, GPU_WIDTHS, createModel, obsToInputs } from "@blanktab/learner"
import { composite, createCpuBackend, noGrad, tensor } from "@blanktab/tensor"
import { ROOM_KINDS } from "@blanktab/world"
import { buildFreshSet } from "@blanktab/harness/fresh"

const file = process.argv[2]
if (!file) throw new Error("usage: room-readout.ts params.json [--widths gpu]")
const widths = process.argv.includes("gpu") ? GPU_WIDTHS : CPU_WIDTHS
const params = (JSON.parse(readFileSync(file, "utf8")) as number[][]).map(a => Float32Array.from(a))
const be = createCpuBackend()
const model = createModel(be, 0, widths, params)
const H = widths.hidden

interface Row { h: Float32Array; room: string; x: number; y: number; segment: number }
const rows: Row[] = []
const segments = buildFreshSet(() => createCanvas(320, 200))
for (let k = 0; k < segments.length; k++) {
  const seg = segments[k]!
  let h = model.zeroState()
  let switched = false
  for (let i = 0; i < seg.events.length; i++) {
    const o = seg.set.obs[i]!
    const inp = obsToInputs(o)
    const its = [tensor(inp.periphery, [3, 50, 80], { backend: be }), tensor(inp.fovea, [3, 32, 32], { backend: be }), tensor(inp.proprio, [1, 3], { backend: be })] as const
    const hPrev = h
    h = noGrad(() => composite(() => model.core({ periphery: its[0], fovea: its[1], proprio: its[2] }, hPrev)))
    for (const t of [...its, hPrev]) t.dispose()
    if (!switched) rows.push({ h: Float32Array.from(await h.read()), room: seg.room, x: o.cursor.x, y: o.cursor.y, segment: k })
    if (seg.events[i] === "switch") switched = true // from here on another room is in front
  }
  h.dispose()
}
model.dispose()

const D = H + 1
const features = (r: Row) => [...r.h, 1]
// Trajectories come in fours, one per starting room: alternate fours go to fitting and to scoring, so both hold every room.
const fitted = (r: Row) => Math.floor(r.segment / ROOM_KINDS.length) % 2 === 0
const train = rows.filter(fitted), test = rows.filter(r => !fitted(r))
for (const room of ROOM_KINDS) if (!train.some(r => r.room === room) || !test.some(r => r.room === room)) throw new Error(`room-readout: no ${room} trajectory on one side of the split`)
/** Ridge regression of `target` on [h, 1] over the training rows: the weights. */
function fit(target: (r: Row) => number): number[] {
  const A = Array.from({ length: D }, () => new Float64Array(D + 1))
  for (const r of train) {
    const x = features(r), y = target(r)
    for (let i = 0; i < D; i++) { for (let j = 0; j < D; j++) A[i]![j]! += x[i]! * x[j]!; A[i]![D]! += x[i]! * y }
  }
  for (let i = 0; i < D; i++) A[i]![i]! += 1e-3 * train.length / D
  for (let c = 0; c < D; c++) {
    let p = c
    for (let r = c + 1; r < D; r++) if (Math.abs(A[r]![c]!) > Math.abs(A[p]![c]!)) p = r
    ;[A[c], A[p]] = [A[p]!, A[c]!]
    for (let r = 0; r < D; r++) if (r !== c) { const f = A[r]![c]! / A[c]![c]!; for (let j = c; j <= D; j++) A[r]![j]! -= f * A[c]![j]! }
  }
  return A.map((row, i) => row[D]! / row[i]!)
}
const read = (w: number[], r: Row) => features(r).reduce((a, v, i) => a + v * w[i]!, 0)
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)
const r2 = (target: (r: Row) => number) => {
  const w = fit(target), m = mean(test.map(target))
  return 1 - test.reduce((a, r) => a + (target(r) - read(w, r)) ** 2, 0) / test.reduce((a, r) => a + (target(r) - m) ** 2, 0)
}
const readers = ROOM_KINDS.map(room => fit(r => (r.room === room ? 1 : 0)))
const fourWay = mean(test.map(r => (ROOM_KINDS[readers.map(w => read(w, r)).reduce((best, v, i, all) => (v > all[best]! ? i : best), 0)] === r.room ? 1 : 0)))
/** One room against the rest, read at one half: how often right, and how often right on that room's own transitions. */
const oneRoom = (room: string) => {
  const w = readers[ROOM_KINDS.indexOf(room as (typeof ROOM_KINDS)[number])]!
  const own = test.filter(r => r.room === room)
  return { accuracy: +mean(test.map(r => ((read(w, r) > 0.5) === (r.room === room) ? 1 : 0))).toFixed(3), found: +mean(own.map(r => (read(w, r) > 0.5 ? 1 : 0))).toFixed(3) }
}
const out = {
  file, transitions: { train: train.length, test: test.length }, meanAbsH: +mean(rows.map(r => mean(Array.from(r.h, Math.abs)))).toFixed(3),
  /** Which of the four rooms is in front, from h, on held-out trajectories. Chance is 0.25. */
  roomAccuracy: +fourWay.toFixed(3),
  /** Variance of each room's 0/1 indicator explained on held-out trajectories (1 is perfect, 0 is no better than its mean). */
  roomR2: Object.fromEntries(ROOM_KINDS.map(room => [room, +r2(r => (r.room === room ? 1 : 0)).toFixed(3)])),
  /** Each room against the rest: accuracy (always answering "not this room" scores 0.75) and the share of its own transitions found. */
  oneRoom: Object.fromEntries(ROOM_KINDS.map(room => [room, oneRoom(room)])),
  /** The control: the cursor's position, which is an input. */
  cursorR2: { x: +r2(r => r.x).toFixed(3), y: +r2(r => r.y).toFixed(3) },
}
console.log(JSON.stringify(out))
