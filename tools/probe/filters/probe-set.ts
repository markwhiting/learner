// The filter probe's fixed pictures (spec §5.1; plan 05c Task 1): 256 frames rendered by the world in Node from 16
// world seeds no life ever lives in (20,000–20,015), under seeded uniformly random actions. A committed manifest
// (tools/probe/filter-probe-set.json) records every action and every frame's sha256, so later runs replay the actions
// and refuse to run on any frame that differs. Harness-only and offline (AGENTS.md invariant 2): nothing here reaches
// a learner, and nothing in packages/ or app/ imports tools/probe (tools/probe/test/isolation.test.ts).
import { createHash } from "node:crypto"
import { mkdirSync, writeFileSync } from "node:fs"
import { gunzipSync, gzipSync } from "node:zlib"
import { join } from "node:path"
import { createCanvas } from "@napi-rs/canvas"
import { createRng, hashSeed } from "@blanktab/detmath"
import { uniformAction } from "@blanktab/harness/heldout"
import { FOVEA, PERIPHERY, type Action, type Observation } from "@blanktab/interface"
import { ROOM_KINDS, createWorld } from "@blanktab/world"

/** The reserved probe world seeds (spec §5.1): no life lives in, or is checked in, a world with one of these seeds. */
export const PROBE_SEEDS: readonly number[] = Object.freeze(Array.from({ length: 16 }, (_, i) => 20_000 + i))
/** Frames per seed: the observation after each of the first 16 actions. */
export const PROBE_STEPS = 16

/** One probe frame: its world seed, its step (1-based), the action before it, and the sha256 hex of its pixels. */
export interface ProbeFrameEntry { seed: number; step: number; action: Action; periphery: string; fovea: string }
export interface ProbeManifest {
  schemaVersion: 1; seeds: number[]; steps: number; startRoom: "seed mod 4"; frames: ProbeFrameEntry[]
  /** sha256 hex over every frame's periphery then fovea bytes, in manifest order. */
  sha256: string
}

const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex")

/** The world a probe seed is rendered in: start room ROOM_KINDS[seed mod 4] (agent's call, plan 05c Task 1). */
function probeWorld(seed: number) {
  return createWorld({ seed, canvas: createCanvas(320, 200), start: ROOM_KINDS[seed % ROOM_KINDS.length]! })
}

/** The manifest's whole-set sha: every frame's periphery then fovea bytes, in order. */
function setSha(frames: readonly Observation[]): string {
  const h = createHash("sha256")
  for (const o of frames) h.update(o.periphery).update(o.fovea)
  return h.digest("hex")
}

/** Renders the probe set afresh, drawing each seed's actions from hashSeed(seed, "filter-probe/actions"). Only --write-manifest calls it. */
export async function buildManifest(): Promise<ProbeManifest> {
  const frames: ProbeFrameEntry[] = []
  const obs: Observation[] = []
  for (const seed of PROBE_SEEDS) {
    const world = probeWorld(seed)
    const rnd = createRng(hashSeed(seed, "filter-probe/actions"))
    for (let step = 1; step <= PROBE_STEPS; step++) {
      const action = uniformAction(rnd)
      world.act(action)
      world.drainTrace()
      const o = world.observe()
      obs.push(o)
      frames.push({ seed, step, action, periphery: sha(o.periphery), fovea: sha(o.fovea) })
    }
  }
  return { schemaVersion: 1, seeds: [...PROBE_SEEDS], steps: PROBE_STEPS, startRoom: "seed mod 4", frames, sha256: setSha(obs) }
}

/**
 * The probe frames, in manifest order, replayed from `m`'s recorded actions. Throws, naming the seed and step, on the
 * first frame whose sha differs from the manifest's (a canvas or world change: regenerating is a recorded decision).
 */
export async function renderProbeSet(m: ProbeManifest): Promise<Observation[]> {
  const out: Observation[] = []
  const bySeed = new Map<number, ProbeFrameEntry[]>()
  for (const f of m.frames) bySeed.set(f.seed, [...(bySeed.get(f.seed) ?? []), f])
  for (const [seed, entries] of bySeed) {
    const world = probeWorld(seed)
    entries.forEach((f, i) => {
      if (f.step !== i + 1) throw new Error(`filter probe manifest: seed ${seed} has step ${f.step} at position ${i + 1}`)
      world.act(f.action)
      world.drainTrace()
      const o = world.observe()
      if (sha(o.periphery) !== f.periphery || sha(o.fovea) !== f.fovea) {
        throw new Error(`filter probe set: the frame at seed ${seed} step ${f.step} does not match the manifest (a canvas or world change; regenerating the manifest is a recorded decision)`)
      }
      out.push(o)
    })
  }
  if (setSha(out) !== m.sha256) throw new Error("filter probe set: the frames' sha256 does not match the manifest's")
  return out
}

const PERIPHERY_BYTES = PERIPHERY.width * PERIPHERY.height * PERIPHERY.channels, FOVEA_BYTES = FOVEA.width * FOVEA.height * FOVEA.channels
/**
 * One frame's bytes in the committed frames file: its periphery, its fovea, the cursor's x and y (float64,
 * little-endian) and whether the button is held (1 byte). The whole observation, as the learner reads it.
 */
const FRAME_BYTES = PERIPHERY_BYTES + FOVEA_BYTES + 8 + 8 + 1

/**
 * The probe frames as committed (tools/probe/filter-probe-frames.bin.gz): every frame's whole observation (FRAME_BYTES), in
 * manifest order, gzipped. The frames are committed, not re-rendered, because the canvas (Skia) draws different pixels
 * on different platforms: frames rendered on macOS did not match on CI's Linux (2026-11-16). The spec's "fixed by a
 * committed manifest" therefore means fixed pixels; rendering is only for --write-manifest, a recorded decision.
 */
export function encodeFrames(frames: readonly Observation[]): Buffer {
  return gzipSync(Buffer.concat(frames.flatMap(o => {
    const tail = Buffer.alloc(17)
    tail.writeDoubleLE(o.cursor.x, 0); tail.writeDoubleLE(o.cursor.y, 8); tail.writeUInt8(o.buttonHeld ? 1 : 0, 16)
    return [Buffer.from(o.periphery), Buffer.from(o.fovea), tail]
  })), { level: 9 })
}

/**
 * The committed frames, checked against `m`: the right count and size, each frame's periphery and fovea sha, and the
 * whole set's. Throws, naming the seed and step, on the first frame that does not match (the file and the manifest
 * were not written together, or the file is damaged).
 */
export function loadProbeSet(m: ProbeManifest, gz: Uint8Array): Observation[] {
  const raw = gunzipSync(gz)
  if (raw.length !== m.frames.length * FRAME_BYTES) throw new Error(`filter probe frames: ${raw.length} bytes, not ${m.frames.length} frames of ${FRAME_BYTES}`)
  const out: Observation[] = m.frames.map((f, i) => {
    const at = i * FRAME_BYTES, tail = at + PERIPHERY_BYTES + FOVEA_BYTES
    const o: Observation = {
      periphery: new Uint8Array(raw.subarray(at, at + PERIPHERY_BYTES)), fovea: new Uint8Array(raw.subarray(at + PERIPHERY_BYTES, tail)),
      cursor: { x: raw.readDoubleLE(tail), y: raw.readDoubleLE(tail + 8) }, buttonHeld: raw.readUInt8(tail + 16) === 1,
    }
    if (sha(o.periphery) !== f.periphery || sha(o.fovea) !== f.fovea) {
      throw new Error(`filter probe frames: the frame at seed ${f.seed} step ${f.step} does not match the manifest`)
    }
    return o
  })
  if (setSha(out) !== m.sha256) throw new Error("filter probe frames: the frames' sha256 does not match the manifest's")
  return out
}

/** An RGB HWC image as PNG bytes, each pixel drawn as a `scale`×`scale` block. */
export function rgbPng(rgb: Uint8Array, w: number, h: number, scale: number): Buffer {
  const c = createCanvas(w * scale, h * scale)
  const ctx = c.getContext("2d")
  const img = ctx.createImageData(w * scale, h * scale)
  for (let y = 0; y < h * scale; y++) for (let x = 0; x < w * scale; x++) {
    const s = (Math.floor(y / scale) * w + Math.floor(x / scale)) * 3, d = (y * w * scale + x) * 4
    img.data[d] = rgb[s]!; img.data[d + 1] = rgb[s + 1]!; img.data[d + 2] = rgb[s + 2]!; img.data[d + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return c.toBuffer("image/png")
}

/**
 * For looking at, never for the analysis: the periphery and fovea of 8 frames (the first two frames of each start
 * room's first seed) as PNGs (×4) under `dir`. Returns the paths written.
 */
export async function writeFramePngs(frames: readonly Observation[], m: ProbeManifest, dir: string): Promise<string[]> {
  mkdirSync(dir, { recursive: true })
  const paths: string[] = []
  for (let r = 0; r < ROOM_KINDS.length; r++) {
    const seed = m.seeds.find(s => s % ROOM_KINDS.length === r)
    if (seed === undefined) continue
    for (let step = 1; step <= 2; step++) {
      const i = m.frames.findIndex(f => f.seed === seed && f.step === step)
      const o = frames[i]!
      for (const [name, bytes, d] of [["periphery", o.periphery, PERIPHERY], ["fovea", o.fovea, FOVEA]] as const) {
        const p = join(dir, `${ROOM_KINDS[r]}-${seed}-${step}-${name}.png`)
        writeFileSync(p, rgbPng(bytes, d.width, d.height, 4))
        paths.push(p)
      }
    }
  }
  return paths
}
