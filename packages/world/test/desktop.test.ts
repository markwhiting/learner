import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD } from "@blanktab/interface"
import { createWorld, layoutCrib, ROOM_VIEW, type ElementInfo, type Room, type RoomEvent, type RoomFactory } from "../src/index"
import { hashSeed, mulberry32 } from "../src/rng"
import { WORDS } from "../src/words"
import { NOOP, PRESS, RELEASE, act, code, makeWorld, moveTo, rgbAt } from "./helpers"

/** A room that paints one flat colour and records every call it receives. */
function stub(color = "rgb(10,120,200)", onPress: RoomEvent[] = []) {
  const calls: string[] = []
  const ticks: number[] = []
  const made: { seed: number; words: readonly string[] }[] = []
  const factory: RoomFactory = (seed, words) => {
    made.push({ seed, words })
    const room: Room = {
      kind: "toys",
      draw(g) { g.rect(0, 0, ROOM_VIEW.width, ROOM_VIEW.height, color) },
      press(x, y) { calls.push(`press ${x},${y}`); return onPress },
      move(x, y, held) { calls.push(`move ${x},${y},${held}`); return [] },
      release(x, y) { calls.push(`release ${x},${y}`); return [] },
      scroll(dir) { calls.push(`scroll ${dir}`); return [] },
      key(k) { calls.push(`key ${k}`); return [] },
      step(tick) { ticks.push(tick); return tick === 3 ? [{ kind: "reveal", elementId: "timer", value: 3 }] : [] },
      inspect: () => null,
      areas: () => ({ interactive: 0, responsive: 0, unresponsive: 1 }),
      scrollState: () => null,
      page: () => null,
    }
    return room
  }
  return { factory, calls, ticks, made, of: (p: string) => calls.filter(c => c.startsWith(p)) }
}

const ICON_Y = WORLD.height - 16 + 2 // 186
const iconCentre = (i: number) => ({ x: 4 + i * 16 + 6, y: ICON_Y + 6 })
const two = (s = stub(), opts = {}) => ({ s, w: makeWorld(1, { rooms: ["crib", "toys"], roomFactories: { toys: s.factory }, ...opts }) })
const viewport = (screen: Uint8ClampedArray) => Array.from(screen.slice(0, WORLD.width * ROOM_VIEW.height * 4))

describe("desktop: privileged leakage (Review Focus 1)", () => {
  it("observe() returns exactly periphery, fovea, cursor, buttonHeld", () => {
    const { w } = two()
    expect(Object.keys(w.observe()).sort()).toEqual(["buttonHeld", "cursor", "fovea", "periphery"])
    expect(Object.keys(w.observe().cursor).sort()).toEqual(["x", "y"])
  })
})

describe("desktop: construction", () => {
  it("defaults to every implemented room, starting at the first", () => {
    const w = makeWorld(1)
    expect(w.rooms()).toEqual(["crib", "toys", "hideseek", "places"])
    expect(w.state().room).toBe("crib")
  })
  it("rejects duplicate kinds, unknown/unimplemented kinds, empty lists and a start outside the list", () => {
    const canvas = createCanvas(WORLD.width, WORLD.height)
    expect(() => createWorld({ seed: 1, canvas, rooms: ["crib", "crib"] })).toThrow(RangeError)
    expect(() => createWorld({ seed: 1, canvas, rooms: [] })).toThrow(RangeError)
    expect(() => createWorld({ seed: 1, canvas, rooms: ["nope" as "crib"] })).toThrow(RangeError)
    expect(() => createWorld({ seed: 1, canvas, rooms: ["crib"], start: "toys" })).toThrow(RangeError)
  })
  it("honours start and taskbar order", () => {
    const { w } = two(stub(), { start: "toys" })
    expect(w.rooms()).toEqual(["crib", "toys"])
    expect(w.state().room).toBe("toys")
  })
  it("seeds each room with hashSeed(seed, kind) and the word list", () => {
    const s = stub()
    makeWorld(7, { rooms: ["crib", "toys"], roomFactories: { toys: s.factory } })
    expect(s.made).toEqual([{ seed: hashSeed(7, "toys"), words: WORDS }])
  })
})

describe("desktop: cursor", () => {
  it("starts at the centre and moves by decoded amounts; act advances tick by 1", () => {
    const w = makeWorld(1)
    expect(w.observe().cursor).toEqual({ x: 0.5, y: 0.5 })
    expect(w.state().cursor).toEqual({ x: 160, y: 100 })
    w.act(act(code(0, 2)))
    expect(w.state().cursor).toEqual({ x: 176, y: 100 })
    expect(w.observe().cursor.x).toBeCloseTo(176 / 320, 10)
    expect(w.tick).toBe(1)
    expect(w.state().tick).toBe(1)
  })
  it("clamps to the screen", () => {
    const w = makeWorld(1)
    for (let i = 0; i < 40; i++) w.act(act(code(3, 2)))
    expect(w.state().cursor).toEqual({ x: 0, y: 0 })
    for (let i = 0; i < 60; i++) w.act(act(code(7, 2)))
    expect(w.state().cursor).toEqual({ x: WORLD.width - 1, y: WORLD.height - 1 })
  })
  it("is drawn last, on top of the taskbar icon", () => {
    const w = makeWorld(1)
    moveTo(w, iconCentre(0).x, iconCentre(0).y)
    expect(rgbAt(w.screen(), iconCentre(0).x, iconCentre(0).y + 2)).toEqual([0, 0, 0])
  })
})

describe("desktop: taskbar", () => {
  it("draws a 16px strip with one seeded 12x12 icon per room at x=4 with 4px gaps", () => {
    const { w } = two()
    moveTo(w, 100, 50) // keep the cursor off the taskbar
    const s = w.screen()
    const strip = rgbAt(s, 200, WORLD.height - 8)
    expect(rgbAt(s, 200, ROOM_VIEW.height)).toEqual(strip) // top row of the strip
    expect(rgbAt(s, 2, ICON_Y + 6)).toEqual(strip) // left margin
    const a = rgbAt(s, 4, ICON_Y), b = rgbAt(s, 20, ICON_Y)
    expect(a).not.toEqual(strip); expect(b).not.toEqual(strip); expect(a).not.toEqual(b)
    expect(rgbAt(s, 15, ICON_Y + 11)).toEqual(a) // icon 0 spans x 4..15, y 186..197
    expect(rgbAt(s, 17, ICON_Y + 6)).toEqual(strip) // gap between icons
    expect(rgbAt(s, 36, ICON_Y + 6)).toEqual(strip) // nothing after the last icon
    const other = makeWorld(2, { rooms: ["crib", "toys"], roomFactories: { toys: stub().factory } })
    moveTo(other, 100, 50)
    expect(rgbAt(other.screen(), 4, ICON_Y)).not.toEqual(a) // icon colours are seeded
  })
  it("outlines the active room's icon with a 1px light outline", () => {
    const { w } = two()
    moveTo(w, 100, 50)
    let s = w.screen()
    const light = rgbAt(s, 3, ICON_Y - 1)
    expect(Math.min(...light)).toBeGreaterThan(180)
    expect(rgbAt(s, 16, ICON_Y + 12)).toEqual(light)
    expect(rgbAt(s, 19, ICON_Y - 1)).not.toEqual(light)
    w.bringToFront("toys")
    s = w.screen()
    expect(rgbAt(s, 19, ICON_Y - 1)).toEqual(light)
    expect(rgbAt(s, 3, ICON_Y - 1)).not.toEqual(light)
  })
  it("state().hovered reports the icon under the cursor", () => {
    const { w } = two()
    moveTo(w, iconCentre(0).x, iconCentre(0).y)
    expect(w.state().hovered).toEqual({ id: "taskbar:crib", interactive: true, responsive: true })
    moveTo(w, iconCentre(1).x, iconCentre(1).y)
    expect(w.state().hovered).toEqual({ id: "taskbar:toys", interactive: true, responsive: true })
    moveTo(w, 2, iconCentre(0).y)
    expect(w.state().hovered).toBeNull()
  })
  it("emits hover traces when the element under the cursor changes", () => {
    const { w } = two()
    moveTo(w, iconCentre(0).x, ICON_Y - 4)
    w.drainTrace()
    w.act(act(code(6, 1))) // south 4 -> onto the icon's top row
    expect(w.drainTrace()).toEqual([{ tick: w.tick, room: "crib", kind: "hover", elementId: "taskbar:crib", interactive: true, responsive: true }])
    w.act(act(code(6, 0))) // still on the icon: no new hover
    expect(w.drainTrace()).toEqual([])
    w.act(act(code(4, 1))) // x 10 -> 6: still on the icon
    expect(w.drainTrace()).toEqual([])
    w.act(act(code(4, 1))) // x 6 -> 2: off the icon onto the bare strip, a hover to nothing
    expect(w.drainTrace()).toEqual([{ tick: w.tick, room: "crib", kind: "hover", interactive: false, responsive: false }])
  })
  it("a taskbar click carries the icon id and flags", () => {
    const { w } = two()
    moveTo(w, iconCentre(0).x, iconCentre(0).y)
    w.drainTrace()
    w.act(PRESS)
    expect(w.drainTrace()).toEqual([{ tick: w.tick, room: "crib", kind: "click", elementId: "taskbar:crib", interactive: true, responsive: true }])
  })
  it("a learner click on another icon switches room, emits focus/learner and changes the viewport", () => {
    const { w, s } = two()
    moveTo(w, iconCentre(1).x, iconCentre(1).y)
    const before = viewport(w.screen())
    w.drainTrace()
    w.act(PRESS)
    expect(w.state().room).toBe("toys")
    const t = w.drainTrace()
    expect(t).toContainEqual({ tick: w.tick, room: "toys", kind: "focus", source: "learner", elementId: "taskbar:toys" })
    expect(viewport(w.screen())).not.toEqual(before)
    w.act(RELEASE)
    expect(s.calls.filter(c => !c.startsWith("move"))).toEqual([]) // taskbar presses never reach rooms
  })
  it("clicking the already-active icon emits a click but no focus", () => {
    const { w } = two()
    moveTo(w, iconCentre(0).x, iconCentre(0).y)
    w.drainTrace()
    w.act(PRESS)
    const t = w.drainTrace()
    expect(t.map(e => e.kind)).toEqual(["click"])
    expect(w.state().room).toBe("crib")
  })
  it("bringToFront emits focus/caregiver; unknown kinds throw", () => {
    const { w } = two()
    moveTo(w, 1, 1) // empty space in both rooms, so no hover change on the switch
    w.drainTrace()
    w.bringToFront("toys")
    expect(w.state().room).toBe("toys")
    expect(w.drainTrace()).toEqual([{ tick: w.tick, room: "toys", kind: "focus", source: "caregiver" }])
    w.bringToFront("toys")
    expect(w.drainTrace()).toEqual([])
    expect(() => w.bringToFront("places")).toThrow(RangeError)
  })
  it("bringToFront(kind, 'harness') emits focus with source harness and no elementId; the default source stays caregiver; an M6 trace with it counts 0", () => {
    const { w } = two()
    moveTo(w, 1, 1)
    w.drainTrace()
    w.bringToFront("toys", "harness")
    expect(w.state().room).toBe("toys")
    const t = w.drainTrace()
    expect(t).toEqual([{ tick: w.tick, room: "toys", kind: "focus", source: "harness" }])
    // M6's rule (packages/harness/src/milestones/m6.ts): only focus events with source "learner" count.
    expect(t.filter(e => e.kind === "focus" && e.source === "learner")).toHaveLength(0)
    w.bringToFront("crib")
    expect(w.drainTrace()).toEqual([{ tick: w.tick, room: "crib", kind: "focus", source: "caregiver" }])
    // Only the learner's own taskbar click makes a "learner" focus.
    expect(() => w.bringToFront("toys", "learner" as never)).toThrow(RangeError)
    expect(w.state().room).toBe("crib")
  })
})

describe("desktop: input routing", () => {
  it("sends press/move/scroll/key to the active room in room-local coords, only while in the viewport", () => {
    const { w, s } = two(stub(), { start: "toys" })
    moveTo(w, 40, 30)
    w.act(PRESS)
    expect(s.of("press")).toEqual(["press 40,30"])
    w.act(act(code(0, 1))) // held move east 4
    expect(s.of("move").at(-1)).toBe("move 44,30,true")
    w.act(RELEASE)
    w.act(act(0, 0, 2))
    w.act(act(0, 0, 1))
    w.act(act(0, 0, 0, 1)) // key "a"
    expect(s.of("scroll")).toEqual(["scroll down", "scroll up"])
    expect(s.of("key")).toEqual(["key a"])
    s.calls.length = 0
    moveTo(w, 200, 190) // over the taskbar strip
    const n = s.of("move").length
    // exactly one move reaches the room from outside the viewport: the exit, so it can drop hover state
    expect(s.of("move").filter(m => Number(m.split(/[ ,]/)[2]) >= ROOM_VIEW.height)).toHaveLength(1)
    w.act(PRESS); w.act(RELEASE); w.act(act(0, 0, 2))
    expect(s.of("move").length).toBe(n)
    expect(s.of("press")).toEqual([])
    expect(s.of("release")).toEqual([])
    expect(s.of("scroll")).toEqual([])
  })
  it("a room switched away from mid-hover shows no stale highlight when it comes back", () => {
    const shape = layoutCrib(hashSeed(1, "crib")).shapes[0]!
    const { w } = two()
    moveTo(w, Math.round(shape.x + shape.size / 2), Math.round(shape.y + shape.size / 2))
    w.bringToFront("toys")
    moveTo(w, iconCentre(0).x, iconCentre(0).y)
    w.act(PRESS) // learner returns to the crib from the taskbar
    expect(w.state().room).toBe("crib")
    const fresh = two().w
    moveTo(fresh, iconCentre(0).x, iconCentre(0).y)
    expect(viewport(w.screen())).toEqual(viewport(fresh.screen()))
  })
  it("background rooms receive no input", () => {
    const { w, s } = two()
    moveTo(w, 40, 30)
    w.act(PRESS); w.act(RELEASE); w.act(act(0, 0, 2)); w.act(act(0, 0, 0, 1))
    expect(s.calls).toEqual([])
  })
  it("a drag that ends over the taskbar delivers the release to the pressed room", () => {
    const { w, s } = two(stub(), { start: "toys" })
    moveTo(w, 40, 170)
    w.act(PRESS)
    w.act(act(code(6, 2))) // south 16 -> y 186, over the taskbar
    expect(w.state().cursor.y).toBe(186)
    w.act(RELEASE)
    expect(s.of("release")).toEqual(["release 40,186"])
  })
  it("the release goes to the room that got the press even if the room changed mid-drag", () => {
    const { w, s } = two(stub(), { start: "toys" })
    moveTo(w, 40, 30)
    w.act(PRESS)
    w.bringToFront("crib")
    w.act(RELEASE)
    expect(s.of("release")).toEqual(["release 40,30"])
  })
  it("room events are traced with tick and room", () => {
    const s = stub(undefined, [{ kind: "widget", elementId: "w1", value: "press" }])
    const { w } = two(s, { start: "toys" })
    moveTo(w, 40, 30)
    w.drainTrace()
    w.act(PRESS)
    expect(w.drainTrace()).toEqual([
      { tick: w.tick, room: "toys", kind: "click", interactive: false, responsive: false }, // the stub's inspect finds nothing
      { tick: w.tick, room: "toys", kind: "widget", elementId: "w1", value: "press" },
    ])
  })
})

describe("desktop: ticks", () => {
  it("steps every room each tick, including background rooms, and traces their events", () => {
    const { w, s } = two() // crib active, stub in the background
    for (let i = 0; i < 5; i++) w.act(NOOP)
    expect(s.ticks).toEqual([1, 2, 3, 4, 5])
    expect(w.drainTrace()).toContainEqual({ tick: 3, room: "toys", kind: "reveal", elementId: "timer", value: 3 })
  })
})

describe("desktop: state()", () => {
  it("reports the active room's areas, scroll and page", () => {
    const w = makeWorld(1)
    const st = w.state()
    expect(st.room).toBe("crib")
    expect(st.page).toBeNull()
    expect(st.scroll).toBeNull()
    expect(st.areas.interactive).toBeGreaterThan(0)
    expect(st.areas.interactive).toBeLessThan(1)
  })
  it("is deterministic: same seed and actions give identical observations and traces", () => {
    const run = () => {
      const { w } = two()
      const seq = [code(6, 2), code(6, 2), code(6, 2), code(6, 2), code(6, 2), code(4, 2), code(4, 2), code(4, 2), code(4, 2), code(4, 2), code(4, 2), code(4, 2), code(4, 2), code(4, 2)]
      const out: unknown[] = []
      for (const m of seq) { w.act(act(m, m % 2 ? 1 : 2)); out.push(Array.from(w.observe().periphery)) }
      out.push(w.drainTrace(), w.state())
      return JSON.stringify(out)
    }
    expect(run()).toBe(run())
  })
})

describe("desktop: carried from review", () => {
  it("state() returns copies of the room's scrollState and inspect results", () => {
    const scroll = { offset: 8, max: 100, revealedFraction: 0.5 }
    const info = { id: "thing", interactive: true, responsive: true }
    const factory: RoomFactory = () => ({
      ...stub().factory(1, WORDS),
      scrollState: () => scroll,
      inspect: () => info,
    })
    const w = makeWorld(1, { rooms: ["toys"], roomFactories: { toys: factory } })
    moveTo(w, 40, 30)
    const st = w.state()
    st.scroll!.offset = 999
    st.hovered!.id = "mutated"
    expect(scroll.offset).toBe(8)
    expect(info.id).toBe("thing")
    expect(w.state().scroll).toEqual({ offset: 8, max: 100, revealedFraction: 0.5 })
    expect(w.state().hovered!.id).toBe("thing")
  })
  it("bringToFront emits the hover trace for the element now under the cursor in the same tick", () => {
    const shape = layoutCrib(hashSeed(1, "crib")).shapes[0]!
    const { w } = two(stub(), { start: "toys" })
    moveTo(w, Math.round(shape.x + shape.size / 2), Math.round(shape.y + shape.size / 2))
    w.drainTrace()
    w.bringToFront("crib")
    expect(w.drainTrace()).toEqual([
      { tick: w.tick, room: "crib", kind: "focus", source: "caregiver" },
      { tick: w.tick, room: "crib", kind: "hover", elementId: shape.id, interactive: true, responsive: true },
    ])
  })
  it("in every room, areas.unresponsive is the rest of the viewport: 1 - responsive (seeds 1-8)", () => {
    for (const kind of ["crib", "toys", "hideseek", "places"] as const) {
      for (let seed = 1; seed <= 8; seed++) {
        const { areas } = makeWorld(seed, { start: kind }).state()
        expect(areas.responsive).toBeGreaterThan(0)
        expect(areas.responsive).toBeLessThan(1)
        expect(areas.interactive).toBeCloseTo(areas.responsive, 12)
        expect(areas.unresponsive).toBeCloseTo(1 - areas.responsive, 12)
      }
    }
  })
  it("the hover trace alone reconstructs state().hovered at every tick (random walk, all rooms, carrying)", () => {
    const w = makeWorld(3)
    const rnd = mulberry32(99)
    let fromTrace: ElementInfo | null = null
    for (let i = 0; i < 1500; i++) {
      const r = rnd()
      const button = r < 0.08 ? 1 : r < 0.16 ? 2 : 0
      w.act(act(Math.floor(rnd() * 25), button, rnd() < 0.1 ? 1 + Math.floor(rnd() * 2) : 0))
      if (i % 61 === 0) w.bringToFront(w.rooms()[Math.floor(rnd() * 4)]!)
      for (const e of w.drainTrace()) {
        if (e.kind !== "hover") continue
        expect(typeof e.interactive).toBe("boolean")
        expect(typeof e.responsive).toBe("boolean")
        fromTrace = e.elementId === undefined ? null : { id: e.elementId, interactive: e.interactive!, responsive: e.responsive! }
        if (!fromTrace) expect([e.interactive, e.responsive]).toEqual([false, false])
      }
      expect(fromTrace).toEqual(w.state().hovered)
    }
  })
})

describe("desktop: inspect (privileged, Plan 06a)", () => {
  it("says what is at any screen point without moving the cursor, changing a pixel or leaving a trace", () => {
    const w = makeWorld(1, { rooms: ["crib", "toys"] })
    const shape = layoutCrib(hashSeed(1, "crib")).shapes[0]!
    const at = { x: Math.round(shape.x + shape.size / 2), y: Math.round(shape.y + shape.size / 2) }
    moveTo(w, 300, 100)
    w.drainTrace()
    const before = { obs: w.observe(), state: w.state(), tick: w.tick }
    expect(w.inspect(at.x, at.y)).toEqual({ id: shape.id, interactive: true, responsive: true })
    expect(w.inspect(iconCentre(0).x, iconCentre(0).y)).toEqual({ id: "taskbar:crib", interactive: true, responsive: true })
    expect(w.inspect(iconCentre(1).x, iconCentre(1).y)).toEqual({ id: "taskbar:toys", interactive: true, responsive: true })
    expect(w.inspect(300, ICON_Y + 6)).toBeNull() // bare taskbar
    for (const [x, y] of [[-1, 5], [5, -1], [WORLD.width, 5], [5, WORLD.height], [Number.NaN, 5]] as const) expect(w.inspect(x, y)).toBeNull()
    expect(w.drainTrace()).toEqual([])
    expect(w.tick).toBe(before.tick)
    expect(w.state()).toEqual(before.state)
    expect(w.observe()).toEqual(before.obs)
  })

  it("at the cursor it is state().hovered, in the room in front", () => {
    const w = makeWorld(1, { rooms: ["crib", "toys"] })
    const shape = layoutCrib(hashSeed(1, "crib")).shapes[0]!
    moveTo(w, Math.round(shape.x + shape.size / 2), Math.round(shape.y + shape.size / 2))
    const c = w.state().cursor
    expect(w.inspect(c.x, c.y)).toEqual(w.state().hovered)
    expect(w.state().hovered?.id).toBe(shape.id)
    w.bringToFront("toys")
    expect(w.inspect(c.x, c.y)).toEqual(w.state().hovered) // the crib's shape is no longer in front
    expect(w.inspect(c.x, c.y)?.id).not.toBe(shape.id)
  })
})
