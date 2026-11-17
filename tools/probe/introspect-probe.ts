// Offline diagnostic for docs/research/2026-11-17-task-and-target-introspection.md (not part of the product): the real learner on the CPU backend in the real four-room world with
// the real caregiver and held-out set, instrumented from the harness side (privileged trace used for diagnosis only).
//   npx tsx tools/probe/introspect-probe.ts --seed 1 --steps 25000 [--curiosityWeight 0] [--residualDecay 0 --bufferSize 25000 --replayPerSleep 250] [--rooms crib] [--ext respond] [--out file.json]
// The environment switches X_LP, X_SHIFT_MOVE_ONLY and X_EXT_WEIGHT (with --ext respond) are read only by the experiment code on the branch diag/learner-introspect; on main they do nothing.
import { createCanvas } from "@napi-rs/canvas"
import { decodeMove, type Action, type Observation } from "@blanktab/interface"
import { CPU_WIDTHS, DEFAULT_SETTINGS, GPU_WIDTHS, actionOneHot, createLearner, createModel, obsToInputs, targetsFromObs, type LearnerSettings, type UpdateInfo, type Widths } from "@blanktab/learner"
import { composite, createCpuBackend, createRng, noGrad, tensor, type Tensor } from "@blanktab/tensor"
import { BUTTON_OPTIONS, KEYS, MOVE_OPTIONS, SCROLL_OPTIONS } from "@blanktab/interface"
import { createWorld, hashSeed } from "@blanktab/world"
import { buildHeldOut } from "@blanktab/harness/heldout"
import { CAREGIVER_DEFAULTS, createCaregiver } from "@blanktab/harness/caregiver"
import { writeFileSync } from "node:fs"

const args = new Map<string, string>()
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i]!.replace(/^--/, ""), process.argv[i + 1] ?? "")
const num = (k: string, d: number) => (args.has(k) ? Number(args.get(k)) : d)
const SEED = num("seed", 1), STEPS = num("steps", 5000), WIN = num("win", 2500), SLEEP = 250
const widths = args.get("widths") === "gpu" ? GPU_WIDTHS : CPU_WIDTHS
const settings: LearnerSettings = { ...DEFAULT_SETTINGS }
for (const k of Object.keys(settings) as (keyof LearnerSettings)[]) if (args.has(k)) settings[k] = Number(args.get(k))
const NO_CAREGIVER = args.get("caregiver") === "off"
const OUT = args.get("out")
const BROAD = num("broad", 800)
const ROOMS = args.get("rooms")?.split(",") as Parameters<typeof createWorld>[0]["rooms"] | undefined
const EXT = args.get("ext") // "respond": reward 1 when the action made an element respond

/** A broad held-out set: uniform random actions in a differently seeded world, carried to a random room every 40 steps, so all four rooms are covered. */
function broadSet(seed: number, n: number): { obs: Observation[]; actions: Action[]; rooms: string[] } {
  const w = createWorld({ seed: hashSeed(seed, "broad-heldout"), canvas: createCanvas(320, 200) })
  const rnd = createRng(hashSeed(seed, "broad-heldout-actions"))
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const obs = [w.observe()], actions: Action[] = [], rooms: string[] = []
  const kinds = w.rooms()
  for (let i = 0; i < n; i++) {
    if (i % 40 === 0) w.bringToFront(kinds[pick(kinds.length)]!)
    const a = { move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS), key: pick(KEYS.length) }
    rooms.push(w.state().room)
    w.act(a); w.drainTrace(); actions.push(a); obs.push(w.observe())
  }
  return { obs, actions, rooms }
}
const sqv = (a: Float32Array, b: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2; return s / a.length }
/** Scores `params` on the broad set (fresh model, core run along it, reset to the zero state at each room carry): ratios to copy-last. */
async function scoreBroad(params: Float32Array[], w: Widths, set: ReturnType<typeof broadSet>) {
  const be = createCpuBackend()
  const m = createModel(be, 0, w, params)
  const T = (d: Float32Array, sh: number[]) => tensor(d, sh, { backend: be })
  let h = m.zeroState()
  let lP = 0, lF = 0, cP = 0, cF = 0
  const byMag: Record<string, [number, number]> = {}
  for (let i = 0; i < set.actions.length; i++) {
    if (i % 40 === 0) { h.dispose(); h = m.zeroState() }
    const inp = obsToInputs(set.obs[i]!), base = targetsFromObs(set.obs[i]!), y = targetsFromObs(set.obs[i + 1]!)
    const its = [T(inp.periphery, [3, 50, 80]), T(inp.fovea, [3, 32, 32]), T(inp.proprio, [1, 3]), T(base.periphery, [1, base.periphery.length]), T(base.fovea, [1, base.fovea.length]), T(actionOneHot(set.actions[i]!), [1, 76])] as [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    const hNext = noGrad(() => composite(() => m.core({ periphery: its[0], fovea: its[1], proprio: its[2] }, h)))
    const pr = noGrad(() => m.predict(hNext, its[5], { periphery: its[3], fovea: its[4] }))
    const [vp, vf] = await Promise.all([pr.periphery.read(), pr.fovea.read()])
    const eP = sqv(vp, y.periphery), eF = sqv(vf, y.fovea), bP = sqv(base.periphery, y.periphery), bF = sqv(base.fovea, y.fovea)
    lP += eP; lF += eF; cP += bP; cF += bF
    const { dx, dy } = decodeMove(set.actions[i]!.move); const k = String(Math.max(Math.abs(dx), Math.abs(dy)))
    const e = (byMag[k] ??= [0, 0]); e[0] += eF; e[1] += bF
    for (const x of [...its, pr.periphery, pr.fovea, h]) x.dispose()
    h = hNext
  }
  h.dispose(); m.dispose()
  const nP = 3000, nF = 768
  return { broad: +((nP * lP + nF * lF) / (nP * cP + nF * cF)).toFixed(4), broadP: +(lP / cP).toFixed(4), broadF: +(lF / cF).toFixed(4), broadFByMag: Object.fromEntries(Object.entries(byMag).map(([k, [a, b]]) => [k, +(a / b).toFixed(3)])) }
}

const sq = (a: Float32Array, b: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2; return s / a.length }
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)
function corr(x: number[], y: number[]): number {
  const n = x.length; if (n < 3) return NaN
  const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) { sxy += (x[i]! - mx) * (y[i]! - my); sxx += (x[i]! - mx) ** 2; syy += (y[i]! - my) ** 2 }
  return sxy / Math.sqrt(sxx * syy)
}
function ranks(x: number[]): number[] {
  const idx = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]); const r = new Array<number>(x.length)
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1]![0] === idx[i]![0]) j++; for (let k = i; k <= j; k++) r[idx[k]![1]] = (i + j) / 2; i = j + 1 }
  return r
}
const spearman = (x: number[], y: number[]) => corr(ranks(x), ranks(y))
const entropy = (counts: number[]) => { const t = counts.reduce((a, b) => a + b, 0); return t ? -counts.reduce((s, c) => (c ? s + (c / t) * Math.log(c / t) : s), 0) : NaN }

interface Row {
  r: number; lb: number; la: number; adv: number; cl: number; clP: number; clF: number
  mag: number; move: number; button: number; scroll: number; key: number; moved: boolean; sameMove: boolean
  respond: boolean; focus: boolean; navigate: boolean; scrolled: boolean; reveal: boolean; widget: boolean
  taskbar: boolean; edge: boolean; overInteractive: boolean; room: string
}

async function main() {
  const backend = createCpuBackend()
  const world = createWorld({ seed: SEED, canvas: createCanvas(320, 200), ...(ROOMS ? { rooms: ROOMS } : {}) })
  const heldOut = buildHeldOut(SEED, createCanvas(320, 200))
  let info: UpdateInfo | null
  const learner = await createLearner({ backend, seed: SEED, widths, settings, heldOut, onUpdate: u => { info = u } })
  const caregiver = NO_CAREGIVER ? null : createCaregiver({ settings: CAREGIVER_DEFAULTS, seed: hashSeed(SEED, "caregiver") })
  const t0 = Date.now()
  let prevObs: Observation | null = null
  let prevAct: Action | null = null
  let prevPrevMove = -1
  let prevMeta: Omit<Row, "r" | "lb" | "la" | "adv" | "cl" | "clP" | "clF"> | null = null
  let prevRoom: string | undefined
  let win: Row[] = []
  const windows: Record<string, unknown>[] = []
  let carries = 0
  const heldSeries: number[] = []

  const broad = BROAD > 0 ? broadSet(SEED, BROAD) : null
  let broadNow: Record<string, unknown> = {}
  const flush = (step: number) => {
    const n = win.length; if (!n) return
    const f = (p: (r: Row) => boolean) => win.filter(p).length / n
    const mr = (p: (r: Row) => boolean) => { const xs = win.filter(p).map(r => r.r); return xs.length ? +mean(xs).toFixed(4) : null }
    const rs = win.map(r => r.r), cl = win.map(r => r.cl), lb = win.map(r => r.lb)
    const hist = (key: "move" | "button" | "scroll" | "key", k: number) => { const c = new Array<number>(k).fill(0); for (const r of win) c[r[key]]!++; return c }
    const st = learner.stats()
    const w = {
      step, cu: +st.cu.toFixed(1), n, ...broadNow,
      heldOut: +mean(heldSeries.splice(0)).toFixed(4), own: +(st.predLoss / st.baselineLoss).toFixed(4),
      meanCopyLast: +mean(cl).toExponential(3), meanCopyLastP: +mean(win.map(r => r.clP)).toExponential(3), meanCopyLastF: +mean(win.map(r => r.clF)).toExponential(3),
      meanReward: +mean(rs).toFixed(4), fracRewardPos: +f(r => r.r > 0).toFixed(3), fracRewardNeg: +f(r => r.r < 0).toFixed(3),
      corrRewardCopyLast: +corr(rs, cl).toFixed(3), spearRewardCopyLast: +spearman(rs, cl).toFixed(3), spearRewardLossBefore: +spearman(rs, lb).toFixed(3),
      // reward by what the action did
      rStatic: mr(r => r.cl === 0), rMag1: mr(r => r.mag === 1 && r.moved), rMag4: mr(r => r.mag === 4 && r.moved), rMag16: mr(r => r.mag === 16 && r.moved),
      rRespond: mr(r => r.respond), rFocus: mr(r => r.focus), rNavigate: mr(r => r.navigate), rScrolled: mr(r => r.scrolled),
      rSameMove: mr(r => r.sameMove && r.moved), rDiffMove: mr(r => !r.sameMove && r.moved),
      // behaviour
      fracStatic: +f(r => r.cl === 0).toFixed(3), fracMoved: +f(r => r.moved).toFixed(3),
      fracMag: [0, 1, 4, 16].map(m => +f(r => r.mag === m).toFixed(3)),
      fracTaskbar: +f(r => r.taskbar).toFixed(3), fracEdge: +f(r => r.edge).toFixed(3), fracOverInteractive: +f(r => r.overInteractive).toFixed(3),
      fracSameMove: +f(r => r.sameMove).toFixed(3),
      per1k: { respond: +(1000 * f(r => r.respond)).toFixed(1), focus: +(1000 * f(r => r.focus)).toFixed(1), navigate: +(1000 * f(r => r.navigate)).toFixed(1), scrolled: +(1000 * f(r => r.scrolled)).toFixed(1), reveal: +(1000 * f(r => r.reveal)).toFixed(1), widget: +(1000 * f(r => r.widget)).toFixed(1), press: +(1000 * f(r => r.button === 1)).toFixed(1), scrollAct: +(1000 * f(r => r.scroll !== 0)).toFixed(1) },
      entropy: { move: +entropy(hist("move", 25)).toFixed(3), button: +entropy(hist("button", 3)).toFixed(3), scroll: +entropy(hist("scroll", 3)).toFixed(3), key: +entropy(hist("key", 45)).toFixed(3) },
      maxEntropy: { move: +Math.log(25).toFixed(3), button: +Math.log(3).toFixed(3), scroll: +Math.log(3).toFixed(3), key: +Math.log(45).toFixed(3) },
      topMoves: hist("move", 25).map((c, i) => [i, c] as const).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([i, c]) => `${i}:${(c / n).toFixed(2)}`),
      rooms: Object.fromEntries(["crib", "toys", "hideseek", "places"].map(k => [k, +f(r => r.room === k).toFixed(2)])),
      carries, secs: Math.round((Date.now() - t0) / 1000),
    }
    windows.push(w)
    console.log(JSON.stringify(w))
    win = []
  }

  for (let step = 1; step <= STEPS; step++) {
    const room = world.state().room
    const obs = world.observe()
    info = null
    const action = await learner.step(obs)
    // the transition just learned from: (prevObs, prevAct) -> obs
    if (prevObs && prevAct && prevMeta && info) {
      const u = info as UpdateInfo
      const a = targetsFromObs(prevObs), b = targetsFromObs(obs)
      const clP = sq(a.periphery, b.periphery), clF = sq(a.fovea, b.fovea)
      const nP = a.periphery.length, nF = a.fovea.length
      win.push({ ...prevMeta, r: u.reward, lb: u.lossBefore, la: u.lossAfter, adv: u.advantage, cl: (nP * clP + nF * clF) / (nP + nF), clP, clF })
    }
    // act, and record what this action did (privileged)
    const before = world.state()
    world.act(action)
    const ev = world.drainTrace()
    const after = world.state()
    if (EXT === "respond") (globalThis as { __X_EXT_REWARD?: number }).__X_EXT_REWARD = ev.some(e => e.kind === "respond") ? 1 : 0
    const { dx, dy } = decodeMove(action.move)
    prevMeta = {
      mag: Math.max(Math.abs(dx), Math.abs(dy)), move: action.move, button: action.button, scroll: action.scroll, key: action.key,
      moved: after.cursor.x !== before.cursor.x || after.cursor.y !== before.cursor.y, sameMove: action.move === prevPrevMove && action.move !== 0,
      respond: ev.some(e => e.kind === "respond"), focus: ev.some(e => e.kind === "focus" && e.source === "learner"),
      navigate: ev.some(e => e.kind === "navigate"), scrolled: ev.some(e => e.kind === "scroll"), reveal: ev.some(e => e.kind === "reveal"), widget: ev.some(e => e.kind === "widget"),
      taskbar: after.cursor.y >= 184, edge: after.cursor.x === 0 || after.cursor.x === 319 || after.cursor.y === 0 || after.cursor.y === 199,
      overInteractive: !!after.hovered?.interactive && !String(after.hovered?.id).startsWith("taskbar:"), room: after.room,
    }
    prevPrevMove = action.move
    prevObs = obs
    prevAct = action
    if (step % SLEEP === 0) {
      await learner.sleep()
      const st = learner.stats()
      heldSeries.push(st.heldOutLoss / st.heldOutBaseline)
    }
    if (caregiver) {
      if (room === prevRoom) caregiver.observe(room, learner.stats().lastLoss)
      prevRoom = room
      const to = caregiver.decide(world.state().room, world.rooms())
      if (to !== null) { world.bringToFront(to as Parameters<typeof world.bringToFront>[0]); carries++ }
    }
    if (step % WIN === 0) { if (broad) broadNow = await scoreBroad(await learner.readParams(), widths, broad); flush(step) }
  }
  if (OUT) writeFileSync(OUT, JSON.stringify({ seed: SEED, steps: STEPS, settings, widths, windows }, null, 1))
  learner.dispose()
}
await main()
