/// <reference types="node" />
import { createHash } from "node:crypto"
import { beforeAll, describe, expect, it } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { DEFAULT_SETTINGS, createLearner } from "@blanktab/learner"
import { createCpuBackend } from "@blanktab/tensor"
import { ROOM_KINDS, type TraceEvent } from "@blanktab/world"
import { emptyObservation } from "@blanktab/interface"
import {
  FRESH_EVENTS, FRESH_PARTS, FRESH_SEGMENTS, FRESH_SEGMENT_STEPS, FRESH_SET_VERSION, buildFreshSegment, buildFreshSet, freshEventOf, freshRatio, freshSeed,
  scoreFresh, type FreshEvent, type FreshSegment,
} from "../src/fresh"

const canvas = () => createCanvas(320, 200)
const TINY = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
const count = (segs: readonly FreshSegment[]) => {
  const c = Object.fromEntries(FRESH_EVENTS.map(e => [e, 0])) as Record<FreshEvent, number>
  for (const s of segs) for (const e of s.events) c[e]++
  return c
}

describe("the fresh-world set (Plan 06a)", { timeout: 120_000 }, () => {
  let set: FreshSegment[]
  beforeAll(() => { set = buildFreshSet(canvas) }, 120_000) // 5,120 world steps: seconds, more on a busy machine

  it("is 32 random and 96 probe trajectories of 40 transitions, every room in front equally often at the start", () => {
    expect(FRESH_SEGMENTS).toEqual({ random: 32, probe: 96 })
    expect(set.length).toBe(128)
    for (const part of FRESH_PARTS) {
      const segs = set.filter(s => s.part === part)
      expect(segs.length).toBe(FRESH_SEGMENTS[part])
      for (const room of ROOM_KINDS) expect(segs.filter(s => s.room === room).length).toBe(FRESH_SEGMENTS[part] / ROOM_KINDS.length)
    }
    for (const s of set) {
      expect(s.set.actions.length).toBe(FRESH_SEGMENT_STEPS)
      expect(s.set.obs.length).toBe(FRESH_SEGMENT_STEPS + 1)
      expect(s.events.length).toBe(FRESH_SEGMENT_STEPS)
    }
  })

  it("every trajectory is a world of its own seed, and building one twice gives the same frames, actions and labels", () => {
    const seeds = FRESH_PARTS.flatMap(p => Array.from({ length: FRESH_SEGMENTS[p] }, (_, i) => freshSeed(p, i)))
    expect(new Set(seeds).size).toBe(seeds.length)
    for (const [part, i] of [["random", 3], ["probe", 17]] as const) {
      const again = buildFreshSegment(part, i, canvas())
      const first = set.filter(s => s.part === part)[i]!
      expect(again.events).toEqual(first.events)
      expect(again.set.actions).toEqual(first.set.actions)
      expect(again.set.obs).toEqual(first.set.obs)
    }
  })

  it("the probe part holds the events random play rarely reaches", () => {
    const random = count(set.filter(s => s.part === "random"))
    const probe = count(set.filter(s => s.part === "probe"))
    expect(probe.respond).toBeGreaterThanOrEqual(200)
    expect(probe.widget).toBeGreaterThanOrEqual(50)
    expect(probe.reveal).toBeGreaterThanOrEqual(25)
    expect(probe.navigate).toBeGreaterThanOrEqual(20)
    expect(probe.switch).toBeGreaterThanOrEqual(20)
    // Per transition, the prober meets each of them more often than uniform actions do.
    for (const e of ["respond", "widget", "navigate", "switch"] as const) expect(probe[e] / 3840, e).toBeGreaterThan(2 * random[e] / 1280)
  })

  it("a static transition's frames are identical, and no other label's are", () => {
    for (const s of set) {
      s.events.forEach((e, i) => {
        const a = s.set.obs[i]!, b = s.set.obs[i + 1]!
        const same = a.buttonHeld === b.buttonHeld && a.cursor.x === b.cursor.x && a.cursor.y === b.cursor.y
          && Buffer.compare(a.periphery, b.periphery) === 0 && Buffer.compare(a.fovea, b.fovea) === 0
        if (e === "static") expect(same).toBe(true)
        if (e === "move") expect(same).toBe(false)
      })
    }
  })

  it(`is version ${FRESH_SET_VERSION}: its content is pinned (a change to the worlds, the prober or the labels must bump the version)`, () => {
    const h = createHash("sha256")
    for (const s of set) {
      h.update(`${s.part}/${s.room}/${s.events.join(",")}/${JSON.stringify(s.set.actions)}`)
      for (const o of s.set.obs) { h.update(o.periphery); h.update(o.fovea); h.update(`${o.cursor.x},${o.cursor.y},${o.buttonHeld}`) }
    }
    expect({ version: FRESH_SET_VERSION, sha256: h.digest("hex"), events: count(set) }).toEqual(PINNED)
  })
})

const PINNED = {
  version: 1,
  sha256: "f7e5c8cfd4bc8cb42b46e564f231e8b0863d594df98ca8dc76a9877d31ebc3bd",
  events: { switch: 23, navigate: 51, reveal: 61, widget: 111, respond: 216, scroll: 275, move: 4256, static: 127 },
}

describe("freshEventOf", () => {
  const ev = (kind: TraceEvent["kind"], extra: Partial<TraceEvent> = {}): TraceEvent => ({ tick: 1, room: "crib", kind, ...extra })
  const a = emptyObservation()
  const moved = { ...emptyObservation(), cursor: { x: 0.5, y: 0.5 } }
  it("is the first that applies of switch, navigate, reveal, widget, respond, scroll; hover and click are not events", () => {
    expect(freshEventOf([ev("hover", { interactive: true, responsive: true }), ev("click", { interactive: true, responsive: true })], a, moved)).toBe("move")
    expect(freshEventOf([ev("scroll"), ev("respond")], a, moved)).toBe("respond")
    expect(freshEventOf([ev("respond"), ev("widget"), ev("scroll")], a, moved)).toBe("widget")
    expect(freshEventOf([ev("reveal"), ev("widget")], a, moved)).toBe("reveal")
    expect(freshEventOf([ev("navigate"), ev("reveal")], a, moved)).toBe("navigate")
    expect(freshEventOf([ev("focus", { source: "learner" }), ev("navigate")], a, moved)).toBe("switch")
  })
  it("a caregiver's or the harness's focus is not the learner's switch; with no event, static if the frame is the same, else move", () => {
    expect(freshEventOf([ev("focus", { source: "caregiver" })], a, moved)).toBe("move")
    expect(freshEventOf([ev("focus", { source: "harness" })], a, emptyObservation())).toBe("static")
    expect(freshEventOf([], a, emptyObservation())).toBe("static")
    expect(freshEventOf([], a, { ...emptyObservation(), buttonHeld: true })).toBe("move")
  })
})

describe("scoreFresh", { timeout: 120_000 }, () => {
  const some = () => [0, 1, 2, 3].map(i => buildFreshSegment("random", i, canvas())).concat([0, 1, 2, 3, 4, 5, 6, 7].map(i => buildFreshSegment("probe", i, canvas())))

  it("a newborn scores copy-last in every group: each ratio is 1, a group copy-last gets exactly has none", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 5, widths: TINY, settings: DEFAULT_SETTINGS })
    const segs = some()
    const r = await scoreFresh(learner, segs)
    expect(r.version).toBe(FRESH_SET_VERSION)
    expect(r.parts.random.n).toBe(4 * FRESH_SEGMENT_STEPS)
    expect(r.parts.probe.n).toBe(8 * FRESH_SEGMENT_STEPS)
    const c = count(segs)
    for (const e of FRESH_EVENTS) expect(r.events[e].n, e).toBe(c[e])
    expect(FRESH_EVENTS.reduce((n, e) => n + r.events[e].n, 0)).toBe(12 * FRESH_SEGMENT_STEPS)
    for (const g of [r.parts.random, r.parts.probe, ...FRESH_EVENTS.filter(e => e !== "static" && c[e] > 0).map(e => r.events[e])]) {
      expect(Math.abs(freshRatio(g)! - 1)).toBeLessThan(1e-4)
    }
    expect(r.events.static.baseline).toBe(0)
    expect(freshRatio(r.events.static)).toBeNull()
    expect(freshRatio({ n: 0, loss: 0, baseline: 0 })).toBeNull()
    expect(learner.stats().evalCU).toBeGreaterThan(0)
    expect(learner.stats().cu).toBe(0)
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("refuses losses that do not match the transitions, or are not finite", async () => {
    const segs = some().slice(0, 1)
    await expect(scoreFresh({ scoreSet: () => Promise.resolve(new Float32Array(3)) }, segs)).rejects.toThrow(RangeError)
    await expect(scoreFresh({ scoreSet: () => Promise.resolve(new Float32Array(FRESH_SEGMENT_STEPS).fill(Number.NaN)) }, segs)).rejects.toThrow(RangeError)
  })
})
