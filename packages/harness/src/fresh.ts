// The fresh-world test (Plan 06a; decision 0029): how well a learner's predictor does in worlds it has never lived
// in. The harness builds a fixed set of short trajectories in worlds of its own seeds, the learner scores them forward
// only (TrainableLearner.scoreSet), and the harness groups the losses by what each action did. Everything here but the
// frames and actions handed to scoreSet is privileged and stays on the harness side: the labels, the prober that
// drives half of the set, and the scores never reach the learner.
//
// Two parts, every room in equal measure:
// - "random": actions drawn uniformly per factor, as a newborn would act and as the held-out set of 0015 is driven.
// - "probe": a privileged prober that walks to things and clicks them, so the set holds enough of the events random
//   play rarely reaches (a shape responding, a widget changing, a reveal, a navigation, a room switch).
import { KEYS, MOVE_OPTIONS, SCROLL_OPTIONS, WORLD, decodeMove, type Action, type Observation } from "@blanktab/interface"
import { copyLastBaseline, targetsFromObs, type HeldOutSet, type TrainableLearner } from "@blanktab/learner"
import { createRng } from "@blanktab/tensor"
import { ROOM_KINDS, ROOM_VIEW, createWorld, hashSeed, type CanvasLike, type RoomKind, type TraceEvent, type World } from "@blanktab/world"
import { uniformAction } from "./heldout"

/** Bumped by any change to what the set holds (its seeds, sizes, the prober or the labels): scores of two versions do not compare. */
export const FRESH_SET_VERSION = 1
/** Transitions per trajectory: each is scored from the zero recurrent state. */
export const FRESH_SEGMENT_STEPS = 40
/** Trajectories per part. */
export const FRESH_SEGMENTS = { random: 32, probe: 96 } as const
export type FreshPart = keyof typeof FRESH_SEGMENTS
export const FRESH_PARTS: readonly FreshPart[] = ["random", "probe"]

/**
 * What an action did, from the world's privileged trace of that tick; the first that applies, so each transition has
 * one label: `switch` the learner's click on a taskbar icon brought another room to the front; `navigate`, `reveal`,
 * `widget`, `respond`, `scroll` the room's events of those kinds; `move` none of those, but the frame changed (the
 * cursor moved, or something moved by itself); `static` the next frame is the same.
 */
export const FRESH_EVENTS = ["switch", "navigate", "reveal", "widget", "respond", "scroll", "move", "static"] as const
export type FreshEvent = (typeof FRESH_EVENTS)[number]

export interface FreshSegment {
  part: FreshPart
  /** The room in front when the trajectory starts. */
  room: RoomKind
  /** What the learner scores: FRESH_SEGMENT_STEPS transitions. */
  set: HeldOutSet
  /** One label per transition. */
  events: FreshEvent[]
}

/** The seed of trajectory `index` of `part`: a world no life lives in (lives' worlds are seeded from their own seeds). */
export function freshSeed(part: FreshPart, index: number): number {
  return hashSeed(0, `fresh/${FRESH_SET_VERSION}/${part}/${index}`)
}

const sameFrame = (a: Observation, b: Observation): boolean => {
  if (a.buttonHeld !== b.buttonHeld || a.cursor.x !== b.cursor.x || a.cursor.y !== b.cursor.y) return false
  for (let i = 0; i < a.periphery.length; i++) if (a.periphery[i] !== b.periphery[i]) return false
  for (let i = 0; i < a.fovea.length; i++) if (a.fovea[i] !== b.fovea[i]) return false
  return true
}

/** The label of one transition (see FRESH_EVENTS). */
export function freshEventOf(trace: readonly TraceEvent[], before: Observation, after: Observation): FreshEvent {
  const kinds = new Set<string>()
  for (const e of trace) kinds.add(e.kind === "focus" ? (e.source === "learner" ? "switch" : "") : e.kind)
  for (const k of FRESH_EVENTS) if (kinds.has(k)) return k
  return sameFrame(before, after) ? "static" : "move"
}

// The prober's habits (agent's calls, Plan 06a): how often a new target is a thing to click or a taskbar icon, how
// readily it presses and lets go, scrolls and types.
const P_TARGET_ELEMENT = 0.65
const P_TARGET_TASKBAR = 0.2
const P_RETARGET = 0.02
const P_PRESS = 0.6
const P_RELEASE = 0.7
const P_STAY = 0.5
const P_TOWARD = 0.7
const P_SCROLL = 0.25
const P_KEY = 0.1
const ARRIVED_PX = 3
const FIND_TRIES = 60
const MOVES = Array.from({ length: MOVE_OPTIONS }, (_, move) => ({ move, ...decodeMove(move) }))

/** The move whose step ends nearest (dx, dy) away; the lowest option on a tie. */
function moveToward(dx: number, dy: number): number {
  let best = 0
  let bestD = Infinity
  for (const m of MOVES) {
    const d = (dx - m.dx) ** 2 + (dy - m.dy) ** 2
    if (d < bestD) { bestD = d; best = m.move }
  }
  return best
}

/**
 * The prober: a privileged driver (it reads world.state() and world.inspect()) that picks a target, walks the cursor
 * to it and clicks what it finds there. A new target is an interactive element of the room in front (a random point
 * that lies on one), a taskbar icon, or any point of the room. Deterministic in `rnd`.
 */
export function createProber(world: World, rnd: () => number): { next(buttonHeld: boolean): Action; sawEvent(e: FreshEvent): void } {
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const anywhere = () => ({ x: pick(ROOM_VIEW.width), y: pick(ROOM_VIEW.height) })
  const find = (ok: (x: number, y: number) => boolean, gen: () => { x: number; y: number }) => {
    for (let k = 0; k < FIND_TRIES; k++) {
      const p = gen()
      if (ok(p.x, p.y)) return p
    }
    return null
  }
  let target: { x: number; y: number } | null = null
  return {
    next(buttonHeld) {
      const st = world.state()
      if (!target || Math.hypot(target.x - st.cursor.x, target.y - st.cursor.y) < ARRIVED_PX || rnd() < P_RETARGET) {
        const u = rnd()
        target = (u < P_TARGET_ELEMENT ? find((x, y) => world.inspect(x, y)?.interactive === true, anywhere)
          : u < P_TARGET_ELEMENT + P_TARGET_TASKBAR
            ? find((x, y) => world.inspect(x, y)?.id.startsWith("taskbar:") === true, () => ({ x: pick(WORLD.width), y: ROOM_VIEW.height + pick(WORLD.height - ROOM_VIEW.height) }))
            : null) ?? anywhere()
      }
      const over = st.hovered?.interactive === true
      const button = buttonHeld ? (rnd() < P_RELEASE ? 2 : 0) : over && rnd() < P_PRESS ? 1 : 0
      const move = over && (button === 1 || buttonHeld)
        ? (rnd() < P_STAY ? 0 : pick(MOVE_OPTIONS))
        : rnd() < P_TOWARD ? moveToward(target.x - st.cursor.x, target.y - st.cursor.y) : pick(MOVE_OPTIONS)
      const scroll = st.scroll !== null && rnd() < P_SCROLL ? 1 + pick(SCROLL_OPTIONS - 1) : 0
      const key = rnd() < P_KEY ? pick(KEYS.length) : 0
      return { move, button, scroll, key }
    },
    // What it was heading for may be gone: look again.
    sawEvent(e) { if (e === "switch" || e === "navigate" || e === "reveal") target = null },
  }
}

/** One trajectory: a world of freshSeed(part, index), room `index mod rooms` brought to the front, FRESH_SEGMENT_STEPS actions. */
export function buildFreshSegment(part: FreshPart, index: number, canvas: CanvasLike, steps = FRESH_SEGMENT_STEPS): FreshSegment {
  const seed = freshSeed(part, index)
  const world = createWorld({ seed, canvas })
  const room = ROOM_KINDS.filter(k => world.rooms().includes(k))[index % world.rooms().length]!
  world.bringToFront(room, "harness")
  world.drainTrace()
  const rnd = createRng(hashSeed(seed, "actions"))
  const prober = part === "probe" ? createProber(world, rnd) : null
  const obs: Observation[] = [world.observe()]
  const actions: Action[] = []
  const events: FreshEvent[] = []
  for (let i = 0; i < steps; i++) {
    const before = obs[i]!
    const a = prober ? prober.next(before.buttonHeld) : uniformAction(rnd)
    world.act(a)
    const after = world.observe()
    const e = freshEventOf(world.drainTrace(), before, after)
    prober?.sawEvent(e)
    actions.push(a)
    obs.push(after)
    events.push(e)
  }
  return { part, room, set: { obs, actions }, events }
}

/** The whole set, the random part then the probe part. `canvas()` gives each trajectory's world a canvas of its own. */
export function buildFreshSet(canvas: () => CanvasLike): FreshSegment[] {
  return FRESH_PARTS.flatMap(part => Array.from({ length: FRESH_SEGMENTS[part] }, (_, i) => buildFreshSegment(part, i, canvas())))
}

/** Sums over a group of transitions: the learner's loss, copy-last's, and how many. The group's ratio is loss / baseline. */
export interface FreshScore { n: number; loss: number; baseline: number }
export interface FreshResult {
  version: typeof FRESH_SET_VERSION
  parts: Record<FreshPart, FreshScore>
  /** By what the action did, over both parts. */
  events: Record<FreshEvent, FreshScore>
}

/** loss / baseline, or null when the group is empty or copy-last is exact on it (nothing to predict). */
export const freshRatio = (s: FreshScore): number | null => (s.n > 0 && s.baseline > 0 ? s.loss / s.baseline : null)

/**
 * Scores `segments` with the learner's current weights (scoreSet: forward only, billed to evalCU) and groups the
 * per-transition losses by part and by event. Copy-last on the same transitions is computed here.
 */
export async function scoreFresh(learner: Pick<TrainableLearner, "scoreSet">, segments: readonly FreshSegment[]): Promise<FreshResult> {
  const zero = (): FreshScore => ({ n: 0, loss: 0, baseline: 0 })
  const parts = Object.fromEntries(FRESH_PARTS.map(p => [p, zero()])) as Record<FreshPart, FreshScore>
  const events = Object.fromEntries(FRESH_EVENTS.map(e => [e, zero()])) as Record<FreshEvent, FreshScore>
  for (const seg of segments) {
    const losses = await learner.scoreSet(seg.set)
    if (losses.length !== seg.events.length) throw new RangeError(`scoreFresh: ${losses.length} losses for ${seg.events.length} transitions`)
    for (let i = 0; i < losses.length; i++) {
      const loss = losses[i]!
      if (!Number.isFinite(loss)) throw new RangeError(`scoreFresh: a non-finite loss (${seg.part} transition ${i})`)
      const baseline = copyLastBaseline(targetsFromObs(seg.set.obs[i]!), targetsFromObs(seg.set.obs[i + 1]!))
      for (const g of [parts[seg.part], events[seg.events[i]!]]) {
        g.n++
        g.loss += loss
        g.baseline += baseline
      }
    }
  }
  return { version: FRESH_SET_VERSION, parts, events }
}
