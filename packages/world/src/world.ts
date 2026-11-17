import {
  FOVEA, KEYS, PERIPHERY, WORLD, isValidAction, decodeMove,
  type Action, type Observation, type WorldView,
} from "@blanktab/interface"
import { createGfx, type Context2DLike } from "./gfx"
import { hashSeed, mulberry32 } from "./rng"
import { WORDS } from "./words"
import {
  ROOM_KINDS, ROOM_VIEW, TASKBAR_HEIGHT, inView,
  type ElementInfo, type Room, type RoomEvent, type RoomFactory, type RoomKind,
} from "./room"
import { cribRoom } from "./rooms/crib"
import { toysRoom } from "./rooms/toys"
import { hideseekRoom } from "./rooms/hideseek"
import { placesRoom } from "./rooms/places"

/**
 * Privileged; harness only.
 *
 * `hover` fires whenever what is under the cursor changes: a different element, the same element with different
 * `interactive`/`responsive` flags (e.g. the hide-and-seek tab just made active), or nothing at all (no `elementId`,
 * both flags false). Replaying the hover events therefore reconstructs `state().hovered` at every tick, so time
 * over elements (M2) can be read from the trace alone. `click` fires for every learner press in the viewport or on
 * a taskbar icon, with `elementId` = the element under the press (absent on empty space). Both `hover` and `click`
 * always carry `interactive` and `responsive` as of that tick (false for empty space); no other kind has them.
 */
export interface TraceEvent {
  tick: number
  room: RoomKind
  kind: "hover" | "click" | "respond" | "scroll" | "reveal" | "navigate" | "widget" | "focus"
  elementId?: string
  value?: number | string
  source?: "learner" | "caregiver" | "harness" // only on "focus"
  interactive?: boolean // only, and always, on "hover" and "click"
  responsive?: boolean // only, and always, on "hover" and "click"
}

/** Privileged; harness only. */
export interface PrivilegedState {
  tick: number
  room: RoomKind
  page: string | null
  cursor: { x: number; y: number } // screen pixels
  hovered: ElementInfo | null // includes taskbar icons: id "taskbar:<kind>", interactive & responsive
  /**
   * Fractions of the room viewport (320x184 at (0,0)) covered by the active room's interactive and responsive
   * elements; `unresponsive` is everything else in the viewport (inert elements and bare background), so
   * `unresponsive === 1 - responsive`. The taskbar is outside the viewport and not in these areas: hover, click and
   * `hovered` entries with `taskbar:*` ids must be excluded from M2/M3, which compare time and clicks over the
   * viewport against these areas. Caveat: a hover to nothing (no id) does not say whether the cursor is on bare
   * viewport background or on the bare taskbar strip; `cursor.y >= 184` here tells them apart.
   */
  areas: { interactive: number; responsive: number; unresponsive: number }
  scroll: { offset: number; max: number; revealedFraction: number } | null
}

export interface World extends WorldView {
  readonly tick: number
  drainTrace(): TraceEvent[]
  state(): PrivilegedState
  /**
   * Privileged; harness only. What is at screen point (x, y) right now: the element of the room in front there, a
   * taskbar icon, or null (bare background, bare taskbar, or off the screen). It changes nothing and leaves no trace;
   * `state().hovered` is this at the cursor.
   */
  inspect(x: number, y: number): ElementInfo | null
  rooms(): readonly RoomKind[] // taskbar order
  /**
   * Brings a room to the front; emits focus with `source` (default "caregiver": the caregiver "carries" the learner;
   * "harness": the harness restores a checkpoint's room on resume). Nothing when the room is already in front.
   */
  bringToFront(kind: RoomKind, source?: "caregiver" | "harness"): void
}

export type { Context2DLike }

export interface CanvasLike {
  width: number
  height: number
  getContext(type: "2d"): Context2DLike | null
}

/** Built-in rooms. A kind is "implemented" once it has an entry here. */
const FACTORIES: Partial<Record<RoomKind, RoomFactory>> = { crib: cribRoom, toys: toysRoom, hideseek: hideseekRoom, places: placesRoom }

const DOWN = WORLD.width / PERIPHERY.width // 4
const ICON = { x0: 4, size: 12, gap: 4, y: WORLD.height - TASKBAR_HEIGHT + 2 } as const
const TASKBAR_COLOR = "rgb(60,60,70)"
const OUTLINE_COLOR = "rgb(240,240,240)"
const lighten = (c: readonly number[]) => c.map(v => Math.min(255, Math.round(v + (255 - v) * 0.4)))
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`
const taskbarId = (k: RoomKind) => `taskbar:${k}`

const copyInfo = (i: ElementInfo | null): ElementInfo | null => (i ? { ...i } : null)
const copyScroll = <T extends object>(s: T | null): T | null => (s ? { ...s } : null)

interface Slot {
  kind: RoomKind; room: Room; icon: number[]; iconX: number
  sawInside: boolean // the last move this room received was inside the viewport
}

export function createWorld(opts: {
  seed: number
  canvas: CanvasLike
  rooms?: readonly RoomKind[]
  start?: RoomKind
  roomFactories?: Partial<Record<RoomKind, RoomFactory>> // test-only override of a kind's factory
}): World {
  const { canvas, seed } = opts
  const kinds = opts.rooms ?? ROOM_KINDS.filter(k => FACTORIES[k])
  if (kinds.length === 0) throw new RangeError("a world needs at least one room")
  if (new Set(kinds).size !== kinds.length) throw new RangeError(`duplicate room kinds: ${kinds.join(",")}`)
  const slots: Slot[] = kinds.map((kind, i) => {
    if (!ROOM_KINDS.includes(kind)) throw new RangeError(`unknown room kind "${kind}"`)
    const factory = opts.roomFactories?.[kind] ?? FACTORIES[kind]
    if (!factory) throw new RangeError(`room kind "${kind}" is not implemented`)
    const room = factory(hashSeed(seed, kind), WORDS)
    if (room.kind !== kind) throw new RangeError(`factory for "${kind}" built a "${room.kind}" room`)
    const rnd = mulberry32(hashSeed(seed, taskbarId(kind)))
    const icon = [0, 0, 0].map(() => 70 + Math.floor(rnd() * 150))
    return { kind, room, icon, iconX: ICON.x0 + i * (ICON.size + ICON.gap), sawInside: false }
  })
  const startKind = opts.start ?? kinds[0]!
  let active = kinds.indexOf(startKind)
  if (active < 0) throw new RangeError(`start room "${startKind}" is not in rooms`)

  canvas.width = WORLD.width
  canvas.height = WORLD.height
  const ctx = canvas.getContext("2d")
  if (!ctx) throw new Error("2d context unavailable")
  const g = createGfx(ctx)

  let cx = Math.floor(WORLD.width / 2)
  let cy = Math.floor(WORLD.height / 2)
  let held = false
  let pressed: Slot | null = null // the room that received the current press; it gets the release
  let tick = 0
  let hoverKey: string | null = null
  let trace: TraceEvent[] = []

  const iconAt = (x: number, y: number) =>
    slots.findIndex(s => x >= s.iconX && x < s.iconX + ICON.size && y >= ICON.y && y < ICON.y + ICON.size)

  const iconInfo = (i: number): ElementInfo => ({ id: taskbarId(slots[i]!.kind), interactive: true, responsive: true })
  const infoAt = (x: number, y: number): ElementInfo | null => {
    if (inView(x, y)) return slots[active]!.room.inspect(x, y)
    const i = iconAt(x, y)
    return i >= 0 ? iconInfo(i) : null
  }
  const hoveredInfo = (): ElementInfo | null => infoAt(cx, cy)
  /** A hover/click event for `info` (null = empty space: no id, both flags false). */
  const pointerEvent = (kind: "hover" | "click", info: ElementInfo | null): TraceEvent => {
    const t: TraceEvent = {
      tick, room: slots[active]!.kind, kind, interactive: info?.interactive ?? false, responsive: info?.responsive ?? false,
    }
    if (info) t.elementId = info.id
    return t
  }

  const emit = (slot: Slot, events: RoomEvent[]) => {
    for (const e of events) {
      const t: TraceEvent = { tick, room: slot.kind, kind: e.kind }
      if (e.elementId !== undefined) t.elementId = e.elementId
      if (e.value !== undefined) t.value = e.value
      trace.push(t)
    }
  }

  /**
   * Emits `hover` when what is under the cursor changes: element (keyed by room), its flags, or to nothing.
   * Empty space has one key in every room, so a room switch over background emits nothing.
   */
  const checkHover = () => {
    const info = hoveredInfo()
    const key = info
      ? `${inView(cx, cy) ? `${slots[active]!.kind}|` : ""}${info.id}|${info.interactive}|${info.responsive}`
      : null
    if (key === hoverKey) return
    hoverKey = key
    trace.push(pointerEvent("hover", info))
  }

  /**
   * Moves reach a room while the cursor is in the viewport, plus once more after it leaves (with the
   * outside point) so the room can drop hover state. Tracked per room, so a room that lost the front
   * mid-hover is told the cursor left as soon as it is back in front.
   */
  const syncCursor = (slot: Slot) => {
    const here = inView(cx, cy)
    if (here || slot.sawInside) emit(slot, slot.room.move(cx, cy, held))
    slot.sawInside = here
  }

  const focus = (i: number, source: "learner" | "caregiver" | "harness") => {
    active = i
    const t: TraceEvent = { tick, room: slots[i]!.kind, kind: "focus", source }
    if (source === "learner") t.elementId = taskbarId(slots[i]!.kind)
    trace.push(t)
    syncCursor(slots[i]!)
  }

  function render() {
    const slot = slots[active]!
    g.clip(0, 0, ROOM_VIEW.width, ROOM_VIEW.height, () => slot.room.draw(g))
    g.rect(0, ROOM_VIEW.height, WORLD.width, TASKBAR_HEIGHT, TASKBAR_COLOR)
    const overIcon = inView(cx, cy) ? -1 : iconAt(cx, cy)
    slots.forEach((s, i) => {
      if (i === active) g.rect(s.iconX - 1, ICON.y - 1, ICON.size + 2, ICON.size + 2, OUTLINE_COLOR)
      g.rect(s.iconX, ICON.y, ICON.size, ICON.size, rgb(i === overIcon ? lighten(s.icon) : s.icon))
    })
    // cursor arrow, tip at (cx, cy); drawn last so it is on top of everything
    ctx!.fillStyle = "rgb(0,0,0)"
    ctx!.beginPath()
    ctx!.moveTo(cx, cy)
    ctx!.lineTo(cx, cy + 10)
    ctx!.lineTo(cx + 3, cy + 7)
    ctx!.lineTo(cx + 6, cy + 7)
    ctx!.closePath()
    ctx!.fill()
  }

  return {
    get tick() { return tick },

    act(a: Action) {
      if (!isValidAction(a)) throw new RangeError("invalid action")
      tick++
      const { dx, dy } = decodeMove(a.move)
      cx = Math.min(WORLD.width - 1, Math.max(0, cx + dx))
      cy = Math.min(WORLD.height - 1, Math.max(0, cy + dy))
      const here = inView(cx, cy)
      const slot = slots[active]!
      syncCursor(slot)
      checkHover()

      if (a.button === 1 && !held) {
        held = true
        if (here) {
          pressed = slot
          trace.push(pointerEvent("click", slot.room.inspect(cx, cy))) // before the press changes anything
          emit(slot, slot.room.press(cx, cy))
        } else {
          const i = iconAt(cx, cy)
          if (i >= 0) {
            trace.push(pointerEvent("click", iconInfo(i)))
            if (i !== active) focus(i, "learner")
          }
        }
      } else if (a.button === 2 && held) {
        held = false
        if (pressed) emit(pressed, pressed.room.release(cx, cy))
        pressed = null
      }

      const front = slots[active]!
      if (a.scroll !== 0 && here) emit(front, front.room.scroll(a.scroll === 1 ? "up" : "down"))
      if (a.key !== 0) emit(front, front.room.key(KEYS[a.key]!))

      for (const s of slots) emit(s, s.room.step(tick))
      checkHover() // content may have changed under a still cursor
    },

    observe(): Observation {
      render()
      const { data } = ctx.getImageData(0, 0, WORLD.width, WORLD.height)
      const periphery = new Uint8Array(PERIPHERY.width * PERIPHERY.height * 3)
      const n = DOWN * DOWN
      for (let py = 0; py < PERIPHERY.height; py++) {
        for (let px = 0; px < PERIPHERY.width; px++) {
          let r = 0, gr = 0, b = 0
          for (let yy = 0; yy < DOWN; yy++) {
            for (let xx = 0; xx < DOWN; xx++) {
              const i = ((py * DOWN + yy) * WORLD.width + px * DOWN + xx) * 4
              r += data[i]!; gr += data[i + 1]!; b += data[i + 2]!
            }
          }
          const o = (py * PERIPHERY.width + px) * 3
          periphery[o] = Math.round(r / n); periphery[o + 1] = Math.round(gr / n); periphery[o + 2] = Math.round(b / n)
        }
      }
      const fovea = new Uint8Array(FOVEA.width * FOVEA.height * 3) // zero = black padding
      const x0 = cx - FOVEA.width / 2, y0 = cy - FOVEA.height / 2
      for (let fy = 0; fy < FOVEA.height; fy++) {
        for (let fx = 0; fx < FOVEA.width; fx++) {
          const sx = x0 + fx, sy = y0 + fy
          if (sx < 0 || sy < 0 || sx >= WORLD.width || sy >= WORLD.height) continue
          const i = (sy * WORLD.width + sx) * 4, o = (fy * FOVEA.width + fx) * 3
          fovea[o] = data[i]!; fovea[o + 1] = data[i + 1]!; fovea[o + 2] = data[i + 2]!
        }
      }
      return { periphery, fovea, cursor: { x: cx / WORLD.width, y: cy / WORLD.height }, buttonHeld: held }
    },

    drainTrace() {
      const out = trace
      trace = []
      return out
    },

    state(): PrivilegedState {
      const room = slots[active]!.room
      return {
        tick, room: room.kind, page: room.page(),
        cursor: { x: cx, y: cy },
        hovered: copyInfo(hoveredInfo()),
        areas: { ...room.areas() },
        scroll: copyScroll(room.scrollState()),
      }
    },

    inspect: (x: number, y: number) => copyInfo(infoAt(x, y)),

    rooms: () => [...kinds],

    bringToFront(kind: RoomKind, source: "caregiver" | "harness" = "caregiver") {
      if (source !== "caregiver" && source !== "harness") throw new RangeError(`bringToFront: unknown source "${String(source)}"`)
      const i = kinds.indexOf(kind)
      if (i < 0) throw new RangeError(`room "${kind}" is not in this world`)
      if (i !== active) {
        focus(i, source)
        checkHover() // the element now under the cursor gets its hover trace this tick
      }
    },
  }
}
