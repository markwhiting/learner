// How a learner's state is stored in a checkpoint: its replay frames (~15 kB each, ~2000 of them) gzip-compressed
// into one buffer, everything else as is. Lossless, so a resumed learner is bit-identical to one that never stopped.
import type { Action, Observation } from "@blanktab/interface"
import type { LearnerState } from "@blanktab/learner"

/** Every distinct observation of the state, frames concatenated (periphery then fovea, per observation). */
export interface PackedFrames {
  /** "raw" where CompressionStream is missing. */
  codec: "gzip" | "raw"
  count: number
  peripheryLen: number
  foveaLen: number
  data: Uint8Array
  /** count × (x, y). */
  cursors: Float64Array
  /** count × 0 | 1. */
  buttons: Uint8Array
}

interface PackedTransition { obs: number; hIn: Float32Array; action: Action; next: number }

/**
 * A LearnerState with its observations replaced by indices into `frames` (the replay's, then the held-out set's, then
 * the whole-life reservoir's that are not among them already).
 */
export type PackedLearnerState = Omit<LearnerState, "replay" | "last" | "heldOut" | "reservoir"> & {
  frames: PackedFrames
  replay: PackedTransition[]
  last: { obs: number; action: Action; hIn: Float32Array } | null
  heldOut: { obs: number[]; actions: Action[] } | null
  /** Present exactly when the state has a reservoir. */
  reservoir?: { items: PackedTransition[]; seen: number }
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isObs = (v: unknown): v is Observation =>
  isRecord(v) && v.periphery instanceof Uint8Array && v.fovea instanceof Uint8Array && isRecord(v.cursor) && typeof v.buttonHeld === "boolean"

/** Whether a stored learner state is packed (checkpoints written before packing, and test stubs, are not). */
export function isPacked(s: unknown): s is PackedLearnerState {
  return isRecord(s) && isRecord(s.frames) && (s.frames.codec === "gzip" || s.frames.codec === "raw")
}

async function pipe(data: Uint8Array, through: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(through as TransformStream<Uint8Array, Uint8Array>)
  return new Uint8Array(await new Response(out).arrayBuffer())
}

/**
 * Packs a learner state for storage. Anything that is not a learner state with well-formed observations of one size
 * (a stub, a state the learner will reject on load) is returned as is, for the learner to judge.
 * `compress: false` (or no CompressionStream) stores the frames raw.
 */
export async function packLearnerState(state: unknown, o: { compress?: boolean } = {}): Promise<unknown> {
  if (!isRecord(state) || isPacked(state) || !Array.isArray(state.replay) || !("last" in state)) return state
  const s = state as unknown as LearnerState
  const last = s.last as LearnerState["last"] | unknown
  if (last !== null && !(isRecord(last) && isObs(last.obs))) return state
  if (!s.replay.every(tr => isRecord(tr) && isObs(tr.obs) && isObs(tr.next))) return state
  // A state from before the held-out set (or without one) has none to pack.
  const heldOut = (s as { heldOut?: unknown }).heldOut ?? null
  if (heldOut !== null && !(isRecord(heldOut) && Array.isArray(heldOut.obs) && heldOut.obs.every(isObs) && Array.isArray(heldOut.actions))) return state
  // A state without a whole-life reservoir (every state before Plan 06a, and any learner without reservoirShare) has none to pack.
  const kept: unknown = (s as { reservoir?: unknown }).reservoir
  if (kept !== undefined && !(isRecord(kept) && Array.isArray(kept.items) && kept.items.every(tr => isRecord(tr) && isObs(tr.obs) && isObs(tr.next)))) return state
  // Distinct observations, by identity: consecutive transitions share one (`next` of one is `obs` of the next).
  const index = new Map<Observation, number>()
  const all: Observation[] = []
  const add = (ob: Observation): number => {
    let i = index.get(ob)
    if (i === undefined) {
      i = all.length
      index.set(ob, i)
      all.push(ob)
    }
    return i
  }
  const replay: PackedTransition[] = s.replay.map(tr => ({ obs: add(tr.obs), hIn: tr.hIn, action: tr.action, next: add(tr.next) }))
  const packedLast = s.last === null ? null : { obs: add(s.last.obs), action: s.last.action, hIn: s.last.hIn }
  const ho = heldOut as LearnerState["heldOut"]
  const packedHeldOut = ho === null ? null : { obs: ho.obs.map(add), actions: ho.actions }
  // After the others, so their indices are what they always were. A transition (or frame) the reservoir shares with
  // the replay is the same object, and is stored once.
  const res = kept as LearnerState["reservoir"]
  const packedReservoir = res === undefined ? undefined
    : { items: res.items.map((tr): PackedTransition => ({ obs: add(tr.obs), hIn: tr.hIn, action: tr.action, next: add(tr.next) })), seen: res.seen }
  const peripheryLen = all[0]?.periphery.length ?? 0
  const foveaLen = all[0]?.fovea.length ?? 0
  if (!all.every(ob => ob.periphery.length === peripheryLen && ob.fovea.length === foveaLen)) return state
  const frame = peripheryLen + foveaLen
  const raw = new Uint8Array(all.length * frame)
  const cursors = new Float64Array(all.length * 2)
  const buttons = new Uint8Array(all.length)
  all.forEach((ob, i) => {
    raw.set(ob.periphery, i * frame)
    raw.set(ob.fovea, i * frame + peripheryLen)
    cursors[2 * i] = ob.cursor.x
    cursors[2 * i + 1] = ob.cursor.y
    buttons[i] = ob.buttonHeld ? 1 : 0
  })
  const gzip = (o.compress ?? true) && typeof CompressionStream === "function"
  const { reservoir: _unpacked, ...other } = s
  void _unpacked
  const data = gzip ? await pipe(raw, new CompressionStream("gzip")) : raw
  return {
    ...other, replay, last: packedLast, heldOut: packedHeldOut, ...(packedReservoir ? { reservoir: packedReservoir } : {}),
    frames: { codec: gzip ? "gzip" : "raw", count: all.length, peripheryLen, foveaLen, data, cursors, buttons },
  } satisfies PackedLearnerState
}

/** The learner state a packed one stands for (shared observations shared again). Unpacked states are returned as is. */
export async function unpackLearnerState(stored: unknown): Promise<unknown> {
  if (!isPacked(stored)) return stored
  const { frames: f, replay, last, heldOut, reservoir, ...rest } = stored
  const fail = (what: string): never => { throw new RangeError(`packed learner state: ${what}`) }
  if (!(f.data instanceof Uint8Array && f.cursors instanceof Float64Array && f.buttons instanceof Uint8Array)) fail("malformed frames")
  if (![f.count, f.peripheryLen, f.foveaLen].every(n => Number.isInteger(n) && n >= 0)) fail("malformed frame sizes")
  if (f.codec === "gzip" && typeof DecompressionStream !== "function") fail("gzip frames, but this browser has no DecompressionStream")
  let raw: Uint8Array
  try {
    raw = f.codec === "gzip" ? await pipe(f.data, new DecompressionStream("gzip")) : f.data
  } catch (e) {
    return fail(`the frames do not decompress (${e instanceof Error ? e.message : String(e)})`)
  }
  const frame = f.peripheryLen + f.foveaLen
  if (raw.length !== f.count * frame || f.cursors.length !== 2 * f.count || f.buttons.length !== f.count) fail("frame data does not match its sizes")
  const all: Observation[] = Array.from({ length: f.count }, (_, i) => ({
    periphery: raw.slice(i * frame, i * frame + f.peripheryLen),
    fovea: raw.slice(i * frame + f.peripheryLen, (i + 1) * frame),
    cursor: { x: f.cursors[2 * i]!, y: f.cursors[2 * i + 1]! },
    buttonHeld: f.buttons[i] === 1,
  }))
  const at = (i: unknown): Observation => (Number.isInteger(i) && all[i as number] !== undefined ? all[i as number]! : fail(`no frame ${String(i)}`))
  if (!Array.isArray(replay)) fail("replay is not an array")
  if (reservoir !== undefined && !(isRecord(reservoir) && Array.isArray(reservoir.items))) fail("reservoir is malformed")
  // A transition held both as recent and in the reservoir was one object, and is one again: by its two frame indices,
  // recurrent state and action (all shared with the replay's entry it came from).
  const unpacked = new Map<string, { obs: Observation; hIn: Float32Array; action: Action; next: Observation }[]>()
  const transition = (tr: PackedTransition) => ({ obs: at(tr.obs), hIn: tr.hIn, action: tr.action, next: at(tr.next) })
  const recent = replay.map(tr => {
    const t = transition(tr)
    const key = `${tr.obs}/${tr.next}`
    unpacked.set(key, [...(unpacked.get(key) ?? []), t])
    return t
  })
  return {
    ...rest,
    replay: recent,
    ...(reservoir === undefined ? {} : {
      reservoir: {
        items: reservoir.items.map(tr => unpacked.get(`${tr.obs}/${tr.next}`)?.find(t => t.hIn === tr.hIn && t.action === tr.action) ?? transition(tr)),
        seen: reservoir.seen,
      },
    }),
    last: last === null ? null : { obs: at(last.obs), action: last.action, hIn: last.hIn },
    // A state packed before the held-out set has none (the learner then rejects it: it is not this version's).
    ...(heldOut === undefined ? {} : {
      heldOut: heldOut === null ? null : isRecord(heldOut) && Array.isArray(heldOut.obs)
        ? { obs: heldOut.obs.map(at), actions: heldOut.actions }
        : fail("heldOut is malformed"),
    }),
  } as LearnerState
}
