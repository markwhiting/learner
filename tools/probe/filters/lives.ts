// Which committed lives the filter probe compares, their logged first-layer filters (meta.filters at 512, 1,024 and
// 2,048 CU, plan 05a), and each life's first layer at birth, recomputed from its logged seed (plan 05c Task 2).
// Every rule here is an agent's call recorded in decision 0027, fixed before the probe first ran on real lives.
import { FILTER_FLOATS_PER_CONV1, levelAtOrBelow, type LogFileV2 } from "@blanktab/harness/log"
import type { LifeInput } from "@blanktab/evaluators"
import { FIRST_LAYER_PARAMS, GPU_WIDTHS, createModel, type Widths } from "@blanktab/learner"
import { createCpuBackend, readSyncCpu } from "@blanktab/tensor"

export type Encoder = "periphery" | "fovea"
export const ENCODERS: readonly Encoder[] = Object.freeze(["periphery", "fovea"])
export interface Filters { w: Float32Array; b: Float32Array }
/** One state of the first layer: w [conv1, 75] / [conv1, 27], b [conv1]. */
export interface FirstLayer { periphery: Filters; fovea: Filters }
/** The snapshot levels the probe reads (spec §5.1), in order. */
export const LEVELS = [512, 1024, 2048] as const
export type Level = typeof LEVELS[number]
export interface ProbeLife {
  learnerId: string; genomeSha: string; seed: number
  init: FirstLayer; at: Record<Level, FirstLayer>
}
export interface Excluded { learnerId: string; reason: string }

const PERIPHERY_FAN = 5 * 5 * 3
const FOVEA_FAN = 3 * 3 * 3

/** The first layer's four arrays in FIRST_LAYER_PARAMS order: periphery w, periphery b, fovea w, fovea b. */
export const layerArrays = (f: FirstLayer): Float32Array[] => [f.periphery.w, f.periphery.b, f.fovea.w, f.fovea.b]

/** Splits meta.filters (base64 of float32 LE, FIRST_LAYER_PARAMS order: pw, pb, fw, fb) into a FirstLayer. */
export function decodeFilters(base64: string, conv1: number): FirstLayer {
  const bytes = Buffer.from(base64, "base64")
  const n = FILTER_FLOATS_PER_CONV1 * conv1
  if (bytes.length !== 4 * n) throw new RangeError(`meta.filters: ${bytes.length} bytes, ${4 * n} expected for conv1 ${conv1}`)
  const all = new Float32Array(n)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (let i = 0; i < n; i++) all[i] = view.getFloat32(4 * i, true)
  let at = 0
  const take = (k: number) => all.slice(at, (at += k))
  const pw = take(conv1 * PERIPHERY_FAN), pb = take(conv1), fw = take(conv1 * FOVEA_FAN), fb = take(conv1)
  return { periphery: { w: pw, b: pb }, fovea: { w: fw, b: fb } }
}

/** The inverse of decodeFilters: base64 of the four arrays as float32 LE. */
export function encodeFilters(f: FirstLayer): string {
  const parts = layerArrays(f)
  const n = parts.reduce((a, p) => a + p.length, 0)
  const buf = Buffer.alloc(4 * n)
  let o = 0
  for (const p of parts) for (const v of p) { buf.writeFloatLE(v, o); o += 4 }
  return buf.toString("base64")
}

/** A newborn learner's first layer: createModel(cpu, seed, widths)'s FIRST_LAYER_PARAMS, before any step. */
export function initFirstLayer(seed: number, widths: Widths): FirstLayer {
  const model = createModel(createCpuBackend(), seed, widths)
  try {
    const [pw, pb, fw, fb] = FIRST_LAYER_PARAMS.map(i => Float32Array.from(readSyncCpu(model.params[i]!.storage)))
    return { periphery: { w: pw!, b: pb! }, fovea: { w: fw!, b: fb! } }
  } finally {
    model.dispose()
  }
}

/** ‖(w, b) − (w₀, b₀)‖ / ‖(w₀, b₀)‖ over one encoder's weights and biases together, in float64. */
export function weightMotion(at: FirstLayer, init: FirstLayer, e: Encoder): number {
  let d = 0, n = 0
  for (const k of ["w", "b"] as const) {
    const a = at[e][k], z = init[e][k]
    for (let i = 0; i < a.length; i++) { const x = a[i]! - z[i]!; d += x * x; n += z[i]! * z[i]! }
  }
  return Math.sqrt(d) / Math.sqrt(n)
}

/** Whether two first-layer states are bit-identical (all four arrays). */
function sameBits(a: FirstLayer, b: FirstLayer): boolean {
  const xs = layerArrays(a), ys = layerArrays(b)
  return xs.every((x, i) => Buffer.from(x.buffer, x.byteOffset, x.byteLength).equals(Buffer.from(ys[i]!.buffer, ys[i]!.byteOffset, ys[i]!.byteLength)))
}

const sameWidths = (w: Widths, v: Widths) => w.conv1 === v.conv1 && w.conv2 === v.conv2 && w.enc === v.enc && w.hidden === v.hidden

/** The first file (lowest seq) at each level that carries meta.filters, by levelAtOrBelow(cu). */
function snapshotsOf(files: readonly LogFileV2[]): Map<number, LogFileV2> {
  const by = new Map<number, LogFileV2>()
  for (const f of files) {
    const level = levelAtOrBelow(f.cu)
    if (f.meta?.filters !== undefined && level !== null && !by.has(level)) by.set(level, f)
  }
  return by
}

interface Candidate { learnerId: string; genomeSha: string; seed: number; snaps: Map<number, LogFileV2> }

/** The first reason a life fails the rules before the seed-sharing one (decision 0027), or its candidate record. */
function screen(life: LifeInput): Candidate | string {
  const files = life.files as LogFileV2[]
  if (files.length === 0 || files.some(f => f.schemaVersion !== 2 || f.meta === undefined)) return "no meta"
  const role = files.map(f => f.meta!.role).find(r => r !== "search")
  if (role !== undefined) return `role ${role}`
  const inh = files.map(f => f.meta!.inheritance.filters as string).find(x => x !== "none")
  if (inh !== undefined) return `inheritance ${inh}`
  const snaps = snapshotsOf(files)
  const off = [...snaps.values()].find(f => !sameWidths(f.widths, GPU_WIDTHS))
  if (off) return off.widths.conv1 !== GPU_WIDTHS.conv1 ? `widths conv1 ${off.widths.conv1}` : `widths ${off.widths.conv1}/${off.widths.conv2}/${off.widths.enc}/${off.widths.hidden}`
  for (const level of LEVELS) if (!snaps.has(level)) return `no snapshot at ${level}`
  const seed = files[0]!.settings.seed
  if (typeof seed !== "number" || !Number.isInteger(seed) || files.some(f => f.settings.seed !== seed)) return "no seed"
  return { learnerId: life.learnerId, genomeSha: files[0]!.meta!.genomeSha, seed, snaps }
}

/**
 * The lives the probe compares, sorted by learnerId, and every other life with the first rule it fails (decision 0027):
 * no meta, role, inheritance, widths, a missing snapshot, no seed, a seed shared with a lower learnerId, and a first
 * layer that did not change between two consecutive states (init, 512, 1,024, 2,048: bit-identical).
 */
export function probeLives(lives: readonly LifeInput[]): { lives: ProbeLife[]; excluded: Excluded[] } {
  const excluded: Excluded[] = []
  const candidates: Candidate[] = []
  for (const life of [...lives].sort((a, b) => (a.learnerId < b.learnerId ? -1 : a.learnerId > b.learnerId ? 1 : 0))) {
    const c = screen(life)
    if (typeof c === "string") excluded.push({ learnerId: life.learnerId, reason: c })
    else candidates.push(c)
  }
  const firstWithSeed = new Map<number, string>()
  const out: ProbeLife[] = []
  for (const c of candidates) {
    const first = firstWithSeed.get(c.seed)
    if (first !== undefined) { excluded.push({ learnerId: c.learnerId, reason: `seed shared with ${first}` }); continue }
    firstWithSeed.set(c.seed, c.learnerId)
    const init = initFirstLayer(c.seed, GPU_WIDTHS)
    const at = Object.fromEntries(LEVELS.map(l => [l, decodeFilters(c.snaps.get(l)!.meta!.filters!, GPU_WIDTHS.conv1)])) as Record<Level, FirstLayer>
    const states: [string, FirstLayer][] = [["init", init], ...LEVELS.map(l => [String(l), at[l]] as [string, FirstLayer])]
    const frozen = states.findIndex(([, s], i) => i > 0 && sameBits(states[i - 1]![1], s))
    if (frozen > 0) { excluded.push({ learnerId: c.learnerId, reason: `filters frozen after ${states[frozen - 1]![0]}` }); continue }
    out.push({ learnerId: c.learnerId, genomeSha: c.genomeSha, seed: c.seed, init, at })
  }
  excluded.sort((a, b) => (a.learnerId < b.learnerId ? -1 : a.learnerId > b.learnerId ? 1 : 0))
  return { lives: out, excluded }
}
