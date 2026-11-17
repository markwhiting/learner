import { BUTTON_OPTIONS, FOVEA, KEYS, isValidAction, MOVE_OPTIONS, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY, SCROLL_OPTIONS } from "@blanktab/interface"
import type { Action, Learner, Observation } from "@blanktab/interface"
import {
  Adam, add, addScalar, backward, composite, createRng, disposeGraph, exp, gatherRows, logSoftmax, matmul, mse, mul,
  neg, noGrad, reshape, scale, settleAll, square, sum, tensor,
} from "@blanktab/tensor"
import type { AdamState, Backend, Tensor } from "@blanktab/tensor"
import { ACTION_DIM, actionOneHot, copyLastBaseline, obsToInputs, targetsFromObs } from "./encode"
import type { Inputs, Targets } from "./encode"
import { createModel } from "./model"
import type { Widths } from "./model"
import { ReplayBuffer, Reservoir, reservoirSlots, sampleStored } from "./replay"
import type { Transition } from "./replay"
import { LEARNER_STATE_VERSION } from "./version"
/** The checkpoint format version (in ./version, a leaf module, so plain-Node build tools can read it). */
export { LEARNER_STATE_VERSION }

export interface LearnerSettings {
  lr: number
  entropyWeight: number
  curiosityWeight: number
  bufferSize: number
  replayPerSleep: number
  /** The predictor's residual output layers train at lr × headLrScale (everything else at lr). */
  headLrScale: number
  /**
   * ε-uniform exploration: with this probability, each action factor is drawn uniformly (from the learner's RNG)
   * instead of from the policy. The actor-critic update uses the action actually taken, with its log-probability
   * under the policy as before and no importance weighting: at ε ≤ 0.5 the behaviour policy stays close to π and
   * the bias is accepted for v1 (controller ruling, Plan 04a.2 Task 1). Clamped into EXPLORE_EPSILON_RANGE.
   */
  exploreEpsilon: number
  /**
   * Decoupled weight decay toward zero (AdamW-style, on the parameter) on Model.residualParams only — the shift gate
   * and the residual output layers, which express "deviation from copy-last" and should return to it without
   * continuing evidence. Per update of those parameters: p ← (1 − residualDecay)·p − Adam step (not scaled by lr).
   * Ceiling: Adam moves a parameter by at most about its lr per update, so at equilibrium |p| ≲ lr_eff / residualDecay
   * — for the shift gate lr / λ (0.5 at the defaults), for the output layers lr·headLrScale / λ (0.05 at the
   * defaults). A residual must keep being re-earned by the gradient to stay near that ceiling.
   * 0 is off and bit-exact to the learner without it. Clamped into RESIDUAL_DECAY_RANGE.
   */
  residualDecay: number
  /**
   * TD discount of the critic's target: curiosityWeight · reward + gamma · V(next). Default GAMMA (0.9), the constant
   * every life before Plan 05a learned with. A state saved before 05a has no gamma and resumes at GAMMA. Clamped into
   * GAMMA_RANGE.
   */
  gamma: number
  /**
   * Whole-life replay (Plan 06a): this share of the bufferSize slots is a reservoir, a uniform sample of every
   * transition of the life so far (reservoirSlots: the nearest whole number of slots, always leaving one recent
   * slot); the rest stay the FIFO of recent transitions. A sleep replays uniformly over both. Each stored transition
   * is offered to the reservoir with the learner's RNG, which it draws from only once it is full. Absent or 0: no
   * reservoir, and bit-exact to the learner without it (it is not one of DEFAULT_SETTINGS, so today's settings and
   * saved states are unchanged). Clamped into RESERVOIR_SHARE_RANGE.
   */
  reservoirShare?: number
}
/** The default TD discount (DEFAULT_SETTINGS.gamma), and the one a state saved without gamma resumes with. */
export const GAMMA = 0.9
export const DEFAULT_SETTINGS: LearnerSettings = Object.freeze({
  // lr 5e-4 and residualDecay 1e-3: Plan 04a.2 Task 1 sweep (held-out ≤ 0.95 at 5k steps on 5 of 6 seeds; see the
  // task-1 report). exploreEpsilon 0.1: controller ruling.
  lr: 5e-4, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 2000, replayPerSleep: 64, headLrScale: 0.1, exploreEpsilon: 0.1,
  residualDecay: 1e-3, gamma: GAMMA, // gamma last (Plan 05a), so the earlier keys keep their order
})
export interface LearnerStats {
  steps: number
  cu: number
  predLoss: number
  baselineLoss: number
  /** Updates skipped for a non-finite loss: online (at most one per step) and replayed (during sleep). */
  skippedUpdates: number
  /** Skipped updates since the last applied one (reset by any applied update): the harness restores the last checkpoint after N. */
  consecutiveSkips: number
  sleeps: number
  /**
   * Compute units spent scoring the held-out set (decision 0015 §3): FLOP deltas on the learner's backend around the
   * scoring only, never included in `cu`.
   */
  evalCU: number
  /** Mean prediction loss on the held-out set at the last sleep (forward only, from the zero state); NaN until then or without a set. */
  heldOutLoss: number
  /** Copy-last's mean loss on the same transitions (the same size weighting); NaN until the first scoring or without a set. */
  heldOutBaseline: number
  /** L_before of this step's online update; NaN on the first step or when non-finite. Absent in older states: read as NaN. */
  lastLoss: number
}

export interface FrozenRunOptions {
  /** Seeds the run's own action-sampling RNG (the learner's is never drawn from). */
  seed: number
  /** Measure the prediction loss and copy-last on each transition (costs a forward pass of the predictor per step). */
  measureLoss: boolean
}
/** Means over the run's transitions (o_{t-1}, a_{t-1}) → o_t; NaN with none or without measureLoss. */
export interface FrozenResult { predLoss: number; baselineLoss: number; transitions: number }
/**
 * A frozen run (spec §8.3, checkups): the learner's current weights acting and predicting in some other stream, from
 * the zero recurrent state with its own RNG. It changes nothing in the learner but stats.evalCU (its FLOPs). While it
 * is open the learner's step, sleep, save and replayLoss reject; dispose it to free its tensors and reopen them. A run
 * whose step threw has freed its tensors and only rejects from then on: dispose it.
 */
export interface FrozenRun {
  step(o: Observation): Promise<Action>
  result(): FrozenResult
  dispose(): void
}

/**
 * A fixed held-out trajectory (decision 0015 §3): `obs.length === actions.length + 1`, transition i being
 * (obs[i], actions[i]) → obs[i+1]. Recorded at birth by the caller (the learner cannot see the world): in the tab, 64
 * transitions of a separate world seeded hashSeed(learnerSeed, "heldout") under uniformly random actions. It is only
 * ever scored, never trained on, and is kept for the learner's whole life (saved and restored with it).
 */
export interface HeldOutSet { obs: Observation[]; actions: Action[] }
/** Bounds on a held-out set's transitions (a hostile or corrupt checkpoint cannot make a sleep arbitrarily expensive). */
export const HELD_OUT_MAX = 1024

/**
 * Everything a learner carries between steps, as plain typed arrays and JSON (structured-clone friendly, for
 * IndexedDB). createLearner({ state }) resumes from it exactly: on CPU, the same observation stream then gives
 * bit-identical actions and losses to a learner that never stopped.
 */
export interface LearnerState {
  version: typeof LEARNER_STATE_VERSION
  seed: number
  widths: Widths
  /** Already clamped. */
  settings: LearnerSettings
  /** Model.params values, in that order. */
  params: Float32Array[]
  /** The two optimizers: [main parameters at lr, the residual output layers at lr × headLrScale]. */
  adam: [AdamState, AdamState]
  /** The action-sampling / replay-sampling RNG. */
  rng: number
  /** h_{t-1}, the recurrent state the next step starts from, [hidden]. */
  h: Float32Array
  /** The previous step (null before the first step): its observation, the action taken, and h_{t-2}. */
  last: { obs: Observation; action: Action; hIn: Float32Array } | null
  /** The replay buffer (the FIFO of recent transitions), oldest first. */
  replay: Transition[]
  /**
   * The whole-life reservoir (settings.reservoirShare): the kept transitions in slot order, and how many were ever
   * offered. Present exactly when the settings give it at least one slot.
   */
  reservoir?: { items: Transition[]; seen: number }
  stats: LearnerStats
  /** The learner's held-out set (fixed at birth), or null for a learner born without one. */
  heldOut: HeldOutSet | null
}

/** What one online update did, for instrumentation and tests (not part of the learner's inputs). */
export interface UpdateInfo {
  step: number
  /** Prediction loss on the last transition before the update (L_before). NaN if non-finite. */
  lossBefore: number
  /** The same loss with the updated weights (L_after); NaN if not measured. */
  lossAfter: number
  /** Curiosity: clamp((L_before − L_after) / max(predLoss EMA before this step, 1e-6), −1, 1), 0 when non-finite. */
  reward: number
  /** TD advantage curiosityWeight·reward + γ·V(h_t) − V(h_{t-1}); NaN if the actor-critic update did not run. */
  advantage: number
  skipped: boolean
}

export interface TrainableLearner extends Learner {
  stats(): LearnerStats
  /**
   * Consolidation condition A: replayPerSleep prediction updates, each on one transition sampled uniformly from
   * the stored ones (the replay buffer, then the reservoir if any: sampleStored) with the learner's RNG (core from the stored h_{t-2} and obs, predict with the stored action
   * against the next frame; size-weighted MSE, Adam). Counts one sleep, even on an empty buffer (no updates).
   * Then, if the learner has a held-out set, scores it (no update, no RNG draw): stats heldOutLoss and
   * heldOutBaseline, its FLOPs billed to evalCU, not cu.
   */
  sleep(): Promise<void>
  /** A snapshot of the whole learner (independent of it: later steps do not change it, nor it them). */
  save(): Promise<LearnerState>
  /**
   * Mean prediction loss over every stored transition (recent or in the reservoir, each once) with the current weights
   * (NaN if none). Measurement only:
   * no update, no RNG draw, not counted in stats.cu.
   */
  replayLoss(): Promise<number>
  /** Every parameter's values, in checkpoint order (Model.params). May run alongside a step; rejects after dispose. */
  readParams(): Promise<Float32Array[]>
  /** Opens a frozen run (see FrozenRun). Throws while one is open, while a step, sleep or save runs, or after dispose. */
  frozenRun(opts: FrozenRunOptions): FrozenRun
  /** The FIFO replay buffer of recent transitions (raw frames; do not mutate). A reservoir, if any, is not part of it. */
  readonly replay: ReplayBuffer
  /**
   * Frees the learner (idempotent). Later calls reject "learner: used after dispose" at once; a call in flight (step,
   * sleep, save, replayLoss, a frozen step, readParams) runs to completion first, and the learner and any open frozen
   * run are freed when the last of them settles.
   */
  dispose(): void
}

/**
 * A checkpoint's learner state is of another LEARNER_STATE_VERSION than this code's: this code cannot resume it.
 * `older`: it is from an older version (the learner was born on older code). A RangeError like any bad checkpoint.
 */
export class LearnerStateVersionError extends RangeError {
  readonly older: boolean
  constructor(readonly found: unknown) {
    super(`LearnerState: version ${String(found)}, expected ${LEARNER_STATE_VERSION}`)
    this.name = "LearnerStateVersionError"
    this.older = typeof found === "number" && found < LEARNER_STATE_VERSION
  }
}

/** The learning rate is clamped into this range (Review Focus 5). */
export const LR_RANGE = [1e-5, 1e-2] as const
/** headLrScale is clamped into this range. */
export const HEAD_LR_SCALE_RANGE = [0.01, 1] as const
/** exploreEpsilon is clamped into this range. */
export const EXPLORE_EPSILON_RANGE = [0, 0.5] as const
/** residualDecay is clamped into this range. */
export const RESIDUAL_DECAY_RANGE = [0, 1e-2] as const
/** gamma is clamped into this range. */
export const GAMMA_RANGE = [0, 0.999] as const
/** reservoirShare, when given, is clamped into this range. */
export const RESERVOIR_SHARE_RANGE = [0, 1] as const
/** Smoothing of the predLoss / baselineLoss moving averages. */
export const STATS_ALPHA = 0.01

const PRED_P = PRED_PERIPHERY.width * PRED_PERIPHERY.height * PRED_PERIPHERY.channels
const PRED_F = PRED_FOVEA.width * PRED_FOVEA.height * PRED_FOVEA.channels
/** Both outputs weighted by size: the MSE over their concatenation, comparable with copyLastBaseline. */
const W_P = PRED_P / (PRED_P + PRED_F)
const W_F = PRED_F / (PRED_P + PRED_F)

const FACTORS = [MOVE_OPTIONS, BUTTON_OPTIONS, SCROLL_OPTIONS, KEYS.length] as const
const OFFSETS = FACTORS.map((_, f) => FACTORS.slice(0, f).reduce((a, n) => a + n, 0))
const FACTOR_ROWS = FACTORS.map((n, f) => Int32Array.from({ length: n }, (_, k) => OFFSETS[f]! + k))

const isPosInt = (v: number) => Number.isInteger(v) && v >= 1
const isNonNeg = (v: number) => Number.isFinite(v) && v >= 0

/**
 * Validates settings and clamps lr, headLrScale, exploreEpsilon, residualDecay and gamma into their ranges. Throws
 * RangeError for values that cannot be clamped. An absent gamma (settings saved before Plan 05a) reads as GAMMA.
 */
export function clampSettings(s: LearnerSettings): LearnerSettings {
  if (!Number.isFinite(s.lr)) throw new RangeError(`lr must be finite, got ${s.lr}`)
  if (!isNonNeg(s.entropyWeight)) throw new RangeError(`entropyWeight must be finite and >= 0, got ${s.entropyWeight}`)
  if (!isNonNeg(s.curiosityWeight)) throw new RangeError(`curiosityWeight must be finite and >= 0, got ${s.curiosityWeight}`)
  if (!isPosInt(s.bufferSize)) throw new RangeError(`bufferSize must be an integer >= 1, got ${s.bufferSize}`)
  if (!isPosInt(s.replayPerSleep)) throw new RangeError(`replayPerSleep must be an integer >= 1, got ${s.replayPerSleep}`)
  if (!Number.isFinite(s.headLrScale)) throw new RangeError(`headLrScale must be finite, got ${s.headLrScale}`)
  if (!Number.isFinite(s.exploreEpsilon)) throw new RangeError(`exploreEpsilon must be finite, got ${s.exploreEpsilon}`)
  if (!Number.isFinite(s.residualDecay)) throw new RangeError(`residualDecay must be finite, got ${s.residualDecay}`)
  // Only an absent gamma (a state saved before Plan 05a) means the old constant; null or any non-number is refused.
  const given: unknown = (s as { gamma?: unknown }).gamma
  const gamma = given === undefined ? GAMMA : given
  if (typeof gamma !== "number" || !Number.isFinite(gamma)) throw new RangeError(`gamma must be a finite number, got ${String(gamma)}`)
  // Optional: only an absent reservoirShare means no reservoir; when given it must be a finite number.
  const share: unknown = (s as { reservoirShare?: unknown }).reservoirShare
  if (share !== undefined && !(typeof share === "number" && Number.isFinite(share))) throw new RangeError(`reservoirShare must be a finite number, got ${String(share)}`)
  const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v))
  return {
    ...s, lr: clamp(s.lr, LR_RANGE), headLrScale: clamp(s.headLrScale, HEAD_LR_SCALE_RANGE),
    exploreEpsilon: clamp(s.exploreEpsilon, EXPLORE_EPSILON_RANGE), residualDecay: clamp(s.residualDecay, RESIDUAL_DECAY_RANGE),
    gamma: clamp(gamma, GAMMA_RANGE),
    ...(share !== undefined ? { reservoirShare: clamp(share, RESERVOIR_SHARE_RANGE) } : {}),
  }
}

const FROZEN_OPEN = "learner: a frozen run is open (dispose it first)"
const IN_PROGRESS = "learner: a step, sleep or save is already in progress (await it first)"

/** Floor of the curiosity scale, so progress on a near-zero loss cannot blow up. */
export const CURIOSITY_SCALE_FLOOR = 1e-6

/**
 * Relative learning progress on one transition: clamp((L_before − L_after) / max(scale, 1e-6), −1, 1), where
 * `scale` is the running prediction loss (the predLoss EMA). Relative, so the reward keeps steering the policy as
 * the loss shrinks. 0 if anything is non-finite.
 */
export function curiosityReward(lossBefore: number, lossAfter: number, scale: number): number {
  const d = (lossBefore - lossAfter) / Math.max(scale, CURIOSITY_SCALE_FLOOR)
  return Number.isFinite(d) ? Math.min(1, Math.max(-1, d)) : 0
}

/** Samples one index from softmax(logits[off .. off+n)); a factor with any non-finite logit is sampled uniformly. */
function sampleFactor(logits: Float32Array, off: number, n: number, rng: () => number): number {
  let mx = -Infinity
  let finite = true
  for (let k = 0; k < n; k++) {
    const v = logits[off + k]!
    if (!Number.isFinite(v)) finite = false
    if (v > mx) mx = v
  }
  const u = rng()
  if (!finite) return Math.min(n - 1, Math.floor(u * n))
  let total = 0
  const w = new Float64Array(n)
  for (let k = 0; k < n; k++) total += (w[k] = Math.exp(logits[off + k]! - mx))
  let acc = 0
  const x = u * total
  for (let k = 0; k < n; k++) {
    acc += w[k]!
    if (x < acc) return k
  }
  return n - 1
}

/**
 * a_t from the policy logits [76], one factor at a time (move, button, scroll, key). With probability
 * `exploreEpsilon` a factor is drawn uniformly instead of from softmax(logits). At ε = 0 no extra RNG draw is made,
 * so the learner is bit-identical to one without exploration.
 */
export function sampleAction(logits: Float32Array, exploreEpsilon: number, rng: () => number): Action {
  const factor = (f: number) => {
    const n = FACTORS[f]!
    if (exploreEpsilon > 0 && rng() < exploreEpsilon) return Math.min(n - 1, Math.floor(rng() * n))
    return sampleFactor(logits, OFFSETS[f]!, n, rng)
  }
  return { move: factor(0), button: factor(1), scroll: factor(2), key: factor(3) }
}

/** Reads every tensor; waits for all to settle; then resolves to the values, or rejects with the first rejection. */
async function readAll(xs: readonly Tensor[]): Promise<Float32Array[]> {
  // The executor form turns a read that throws synchronously into a rejection, so every read is still awaited.
  return settleAll(xs.map((x) => new Promise<Float32Array>((resolve) => resolve(x.read()))))
}

const isUint32 = (v: unknown) => Number.isInteger(v) && (v as number) >= 0 && (v as number) < 2 ** 32
const PERIPHERY_LEN = PERIPHERY.width * PERIPHERY.height * PERIPHERY.channels
const FOVEA_LEN = FOVEA.width * FOVEA.height * FOVEA.channels
/** Number of trailing Model.params on the second optimizer (Model.outputParams = params.slice(-4)). */
const OUTPUT_PARAMS = 4

/**
 * Checks everything createLearner({ state }) relies on that createModel does not (createModel checks the parameter
 * count and sizes against the widths), so a bad checkpoint throws RangeError before anything is allocated.
 */
function validateState(s: LearnerState): void {
  try {
    checkState(s)
  } catch (e) {
    // A missing or mistyped field surfaces as a TypeError somewhere inside the checks: it is still a bad checkpoint.
    if (e instanceof RangeError) throw e
    throw new RangeError(`LearnerState: malformed (${e instanceof Error ? e.message : String(e)})`, { cause: e })
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const allFinite = (a: Float32Array) => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) return false
  return true
}
const STATE_KEYS = ["version", "seed", "widths", "settings", "params", "adam", "rng", "h", "last", "replay", "stats", "heldOut"] as const

/** Throws RangeError unless `h` is a well-formed held-out set (see HeldOutSet). */
function checkHeldOut(h: unknown, where: string): asserts h is HeldOutSet {
  const fail = (what: string) => { throw new RangeError(`${where}: ${what}`) }
  if (!isRecord(h) || !Array.isArray(h.obs) || !Array.isArray(h.actions)) return fail("must be { obs, actions } arrays")
  const n = h.actions.length
  if (!(n >= 1 && n <= HELD_OUT_MAX)) fail(`must have 1 to ${HELD_OUT_MAX} transitions, has ${n}`)
  if (h.obs.length !== n + 1) fail(`needs ${n + 1} observations for ${n} transitions, has ${h.obs.length}`)
  h.obs.forEach((o: unknown, i) => {
    if (!(isRecord(o) && o.periphery instanceof Uint8Array && o.fovea instanceof Uint8Array && isRecord(o.cursor)
      && typeof o.cursor.x === "number" && typeof o.cursor.y === "number" && typeof o.buttonHeld === "boolean")) fail(`obs[${i}] is malformed`)
    const ob = o as Observation
    if (ob.periphery.length !== PERIPHERY_LEN || ob.fovea.length !== FOVEA_LEN) fail(`obs[${i}]: malformed frame`)
  })
  h.actions.forEach((a: unknown, i) => { if (!isRecord(a) || !isValidAction(a as unknown as Action)) fail(`actions[${i}] is not a valid action`) })
}

function checkState(s: LearnerState): void {
  const fail = (what: string) => { throw new RangeError(`LearnerState: ${what}`) }
  if (!isRecord(s)) fail("not an object")
  // First: a state of another version is told apart from a corrupt one, whatever else it holds.
  if ("version" in s && s.version !== LEARNER_STATE_VERSION) throw new LearnerStateVersionError(s.version)
  for (const k of STATE_KEYS) if (!(k in s)) fail(`missing ${k}`)
  const isF32 = (v: unknown) => v instanceof Float32Array
  if (!isRecord(s.widths) || !isRecord(s.settings) || !isRecord(s.stats)) fail("widths, settings and stats must be objects")
  if (!Array.isArray(s.params) || !s.params.every(isF32)) fail("params must be Float32Arrays")
  if (!Array.isArray(s.adam) || !s.adam.every(a => isRecord(a) && Array.isArray(a.slots) && a.slots.every(sl => isRecord(sl) && isF32(sl.m) && isF32(sl.v)))) {
    fail("adam must be two optimizer states of Float32Array moments")
  }
  if (!isF32(s.h)) fail("h must be a Float32Array")
  if (s.last !== null && !(isRecord(s.last) && isRecord(s.last.obs) && isRecord(s.last.action) && isF32(s.last.hIn))) fail("last is malformed")
  if (!Array.isArray(s.replay) || !s.replay.every(tr => isRecord(tr) && isRecord(tr.obs) && isRecord(tr.next) && isRecord(tr.action) && isF32(tr.hIn))) {
    fail("replay is malformed")
  }
  if (!isUint32(s.seed)) fail(`seed must be a uint32, got ${s.seed}`)
  for (const k of ["conv1", "conv2", "enc", "hidden"] as const) if (!isPosInt(s.widths[k])) fail(`widths.${k} must be an integer >= 1`)
  const clamped = clampSettings(s.settings)
  for (const k of Object.keys(DEFAULT_SETTINGS) as (keyof LearnerSettings)[]) {
    // A state saved before Plan 05a has no gamma: it reads as GAMMA (createLearner), so there is nothing to compare.
    if (k === "gamma" && s.settings.gamma === undefined) continue
    if (clamped[k] !== s.settings[k]) fail(`settings.${k} = ${s.settings[k]} is not a clamped setting`)
  }
  if (clamped.reservoirShare !== s.settings.reservoirShare) fail(`settings.reservoirShare = ${s.settings.reservoirShare} is not a clamped setting`)
  const n = s.params.length
  if (n <= OUTPUT_PARAMS) fail(`${n} parameters`)
  const split = [s.params.slice(0, n - OUTPUT_PARAMS), s.params.slice(n - OUTPUT_PARAMS)]
  if (s.adam.length !== 2) fail(`${s.adam.length} optimizer states, expected 2`)
  split.forEach((ps, k) => {
    const slots = s.adam[k]!.slots
    if (slots.length !== ps.length) fail(`optimizer ${k} has ${slots.length} slots for ${ps.length} parameters`)
    slots.forEach((sl, i) => {
      const len = ps[i]!.length
      if (sl.m.length !== len || sl.v.length !== len) fail(`optimizer ${k} slot ${i} does not match its parameter's ${len} values`)
      if (!(Number.isInteger(sl.t) && sl.t >= 0)) fail(`optimizer ${k} slot ${i} step count ${sl.t}`)
      if (!allFinite(sl.m) || !allFinite(sl.v)) fail(`optimizer ${k} slot ${i} has a non-finite moment`)
    })
  })
  // A non-finite weight would poison every later update (Review Focus 4): such a checkpoint is no checkpoint.
  s.params.forEach((p, i) => { if (!allFinite(p)) fail(`params[${i}] has a non-finite value`) })
  if (!isUint32(s.rng)) fail(`rng state must be a uint32, got ${s.rng}`)
  const hidden = s.widths.hidden
  const checkObs = (o: Observation, where: string) => {
    if (o.periphery.length !== PERIPHERY_LEN || o.fovea.length !== FOVEA_LEN) fail(`${where}: malformed frame`)
  }
  if (s.h.length !== hidden) fail(`h has ${s.h.length} values, hidden is ${hidden}`)
  if (!allFinite(s.h)) fail("h has a non-finite value")
  if (s.last) {
    checkObs(s.last.obs, "last.obs")
    if (!isValidAction(s.last.action)) fail("last.action is not a valid action")
    if (s.last.hIn.length !== hidden) fail(`last.hIn has ${s.last.hIn.length} values, hidden is ${hidden}`)
    if (!allFinite(s.last.hIn)) fail("last.hIn has a non-finite value")
  }
  const slots = reservoirSlots(s.settings.bufferSize, s.settings.reservoirShare)
  if (s.replay.length > s.settings.bufferSize - slots) fail(`${s.replay.length} transitions exceed the ${s.settings.bufferSize - slots} recent slots of bufferSize ${s.settings.bufferSize}`)
  const checkTransitions = (trs: Transition[], where: string) => trs.forEach((tr, i) => {
    checkObs(tr.obs, `${where}[${i}].obs`)
    checkObs(tr.next, `${where}[${i}].next`)
    if (!isValidAction(tr.action)) fail(`${where}[${i}].action is not a valid action`)
    if (tr.hIn.length !== hidden) fail(`${where}[${i}].hIn has ${tr.hIn.length} values, hidden is ${hidden}`)
    if (!allFinite(tr.hIn)) fail(`${where}[${i}].hIn has a non-finite value`)
  })
  checkTransitions(s.replay, "replay")
  // The reservoir is there exactly when the settings give it a slot, and is one a Reservoir can continue from.
  if ((s.reservoir !== undefined) !== (slots > 0)) fail(slots > 0 ? "a learner with reservoirShare has no reservoir" : "a reservoir without reservoirShare")
  if (s.reservoir !== undefined) {
    const r: unknown = s.reservoir
    if (!isRecord(r) || !Array.isArray(r.items) || !r.items.every(tr => isRecord(tr) && isRecord(tr.obs) && isRecord(tr.next) && isRecord(tr.action) && tr.hIn instanceof Float32Array)) {
      fail("reservoir is malformed")
    }
    const { items, seen } = s.reservoir
    if (items.length > slots) fail(`reservoir: ${items.length} transitions exceed its ${slots} slots`)
    if (!(Number.isSafeInteger(seen) && (items.length < slots ? seen === items.length : seen >= items.length))) fail(`reservoir: seen ${String(seen)} does not fit ${items.length} kept of ${slots} slots`)
    checkTransitions(items, "reservoir.items")
  }
  for (const k of ["steps", "skippedUpdates", "consecutiveSkips", "sleeps"] as const) {
    if (!(Number.isInteger(s.stats[k]) && s.stats[k] >= 0)) fail(`stats.${k} = ${s.stats[k]}`)
  }
  if (!(Number.isFinite(s.stats.cu) && s.stats.cu >= 0)) fail(`stats.cu = ${s.stats.cu}`)
  if (!(Number.isFinite(s.stats.evalCU) && s.stats.evalCU >= 0)) fail(`stats.evalCU = ${s.stats.evalCU}`)
  for (const k of ["heldOutLoss", "heldOutBaseline"] as const) {
    const v = s.stats[k]
    // NaN (not scored yet, or no set) or a finite loss: a scoring never stores anything else.
    if (!(typeof v === "number" && (Number.isNaN(v) || (Number.isFinite(v) && v >= 0)))) fail(`stats.${k} = ${v}`)
  }
  // Optional (states from before it existed load with NaN); when present, NaN or a finite loss.
  if ("lastLoss" in s.stats) {
    const v: unknown = s.stats.lastLoss
    if (!(typeof v === "number" && (Number.isNaN(v) || (Number.isFinite(v) && v >= 0)))) fail(`stats.lastLoss = ${String(v)}`)
  }
  if (s.heldOut !== null) checkHeldOut(s.heldOut, "LearnerState: heldOut")
  if ((s.last === null) !== (s.stats.steps === 0)) fail("last must be null exactly when no step has been taken")
}

const copyObs = (o: Observation): Observation => ({
  periphery: o.periphery.slice(), fovea: o.fovea.slice(), cursor: { x: o.cursor.x, y: o.cursor.y }, buttonHeld: o.buttonHeld,
})

interface InputTensors { periphery: Tensor; fovea: Tensor; proprio: Tensor }

interface TargetTensors { periphery: Tensor; fovea: Tensor }

/** The previous step: what the next step learns from (all caller-owned). */
interface Last {
  obs: Observation
  inputs: InputTensors
  action: Action
  targets: Targets
  /** Its targets on the device: the residual predictor's base for the next transition. */
  targetTensors: TargetTensors
}

/**
 * The online learner, condition A (one set of weights). Each `step(obs_t)`:
 *  1. learns to predict y_t from the last transition: h_{t-1} = core(inputs_{t-1}, h_{t-2}) is recomputed with
 *     gradient from the stored, detached h_{t-2} (truncated BPTT of length 1, no graph lives across steps),
 *     L_before = mse(predict(h_{t-1}, a_{t-1}, y_{t-1}), y_t) over both outputs (residual on the current frame),
 *     then backward and one Adam step;
 *  2. measures curiosity r = clamp((L_before − L_after) / max(predLoss EMA before this step, 1e-6), −1, 1), L_after
 *     being the same loss with the new weights;
 *  3. acts: h_t = core(inputs_t, h_{t-1}) and the heads, under noGrad; one 77-float tensor (logits, V(h_t)), read in
 *     the same round as L_after and h_t (both need only the updated weights); a_t is sampled per factor with the
 *     learner's seeded RNG, uniformly with probability exploreEpsilon (sampleAction);
 *  4. updates the actor-critic on a_{t-1}: A = curiosityWeight·r + γ·V(h_t) − V(h_{t-1}),
 *     loss = −logπ(a_{t-1})·stopgrad(A) + A² − entropyWeight·H(π), one Adam step;
 *  5. stores (obs_{t-1}, h_{t-2}, a_{t-1}, obs_t) in the FIFO replay buffer, offers it to the whole-life reservoir
 *     if there is one (settings.reservoirShare), and carries h ← h_t (detached).
 *
 * Steps 1 and 4 are separate backward passes because r depends on the prediction update. They touch disjoint
 * parameters — the policy/value heads read the detached h_{t-1}, so the actor-critic gradient does not enter
 * the core — hence every parameter still gets exactly one Adam step per learner step, as for a single
 * backward(L_before + actorCritic). V(h_{t-1}) uses the carried h_{t-1}, which equals the recomputed one unless
 * a sleep() changed the core weights in between (both are then valid detached states; the carried one is used).
 *
 * sleep() replays buffered transitions (condition A); save() / createLearner({ state }) checkpoint and resume exactly.
 *
 * The predictor's residual output layers train at lr × headLrScale (a second Adam), all other parameters at lr.
 *
 * Guards: lr is clamped to LR_RANGE, headLrScale to HEAD_LR_SCALE_RANGE; a non-finite loss skips its update (counted in skippedUpdates, never
 * applied, never thrown). A rejected read() is fatal and propagates (packages/tensor/README.md rule 5); a step that
 * throws part-way frees its own tensors (the learner must then be disposed) and leaves h, hIn and last as they were.
 */
export async function createLearner(opts: {
  backend: Backend
  onUpdate?: (u: UpdateInfo) => void
} & (
  | { seed: number; widths: Widths; settings: LearnerSettings; heldOut?: HeldOutSet; state?: undefined }
  | { state: LearnerState }
)): Promise<TrainableLearner> {
  const { backend } = opts
  // The learner owns its copy. Any state given (even a falsy one) is a checkpoint and is validated.
  const saved = opts.state !== undefined ? structuredClone(opts.state) : null
  if (opts.state !== undefined) validateState(saved as LearnerState)
  const bornWith: unknown = opts.state === undefined ? opts.heldOut : undefined
  if (bornWith !== undefined) checkHeldOut(bornWith, "heldOut")
  /** The held-out set, the learner's own copy, fixed for its life. */
  const heldOut: HeldOutSet | null = saved ? saved.heldOut : bornWith !== undefined ? structuredClone(bornWith) : null
  /** Copy-last on the held-out set: a constant of the set, computed on the host once. */
  const heldOutCopyLast = heldOut
    ? heldOut.actions.reduce((acc, _, i) => acc + copyLastBaseline(targetsFromObs(heldOut.obs[i]!), targetsFromObs(heldOut.obs[i + 1]!)), 0) / heldOut.actions.length
    : Number.NaN
  const seed = saved ? saved.seed : (opts as { seed: number }).seed
  const widths: Widths = saved ? saved.widths : (opts as { widths: Widths }).widths
  // A saved state without gamma (from before Plan 05a) resumes at GAMMA, and its next save() writes it.
  const settings = saved ? { ...saved.settings, gamma: saved.settings.gamma ?? GAMMA } : clampSettings((opts as { settings: LearnerSettings }).settings)
  const model = createModel(backend, seed, widths, saved?.params)
  // Two optimizers over disjoint parameters: the residual output layers at lr × headLrScale (they start at zero and
  // see near-zero gradients over static background, where Adam's normalized step at full lr is pure jitter that
  // outweighs what they learn), everything else at lr. Each step runs both: one Adam step per parameter.
  // The checkpoint stores the two optimizers by position: [all but the last OUTPUT_PARAMS params, the last ones].
  if (model.outputParams.length !== OUTPUT_PARAMS || model.outputParams.some((p, i) => p !== model.params[model.params.length - OUTPUT_PARAMS + i])) {
    model.dispose()
    throw new Error("learner: Model.outputParams must be the last OUTPUT_PARAMS of Model.params (checkpoint format)")
  }
  const outputSet = new Set(model.outputParams)
  // Action sampling has its own stream, independent of the initializer's.
  const rng = createRng((seed ^ 0x9e3779b9) >>> 0)
  // The bufferSize slots: a whole-life reservoir of reservoirSlots of them (none without reservoirShare), the rest the
  // FIFO of recent transitions.
  const keptSlots = reservoirSlots(settings.bufferSize, settings.reservoirShare)
  const replay = new ReplayBuffer(settings.bufferSize - keptSlots)
  const reservoir = keptSlots > 0 ? new Reservoir(keptSlots, saved?.reservoir) : null
  const stats: LearnerStats = {
    steps: 0, cu: 0, predLoss: Number.NaN, baselineLoss: Number.NaN, skippedUpdates: 0, consecutiveSkips: 0, sleeps: 0,
    evalCU: 0, heldOutLoss: Number.NaN, heldOutBaseline: Number.NaN, lastLoss: Number.NaN,
  }

  // Every tensor made by `t` while the learner is being constructed is registered (see `owned` below).
  let constructing: ((x: Tensor) => void) | null = null
  const t = (data: Float32Array, shape: number[]) => {
    const x = tensor(data, shape, { backend })
    constructing?.(x)
    return x
  }
  const upload = (i: Inputs): InputTensors => ({
    periphery: t(i.periphery, [PERIPHERY.channels, PERIPHERY.height, PERIPHERY.width]),
    fovea: t(i.fovea, [FOVEA.channels, FOVEA.height, FOVEA.width]),
    proprio: t(i.proprio, [1, 3]),
  })
  const disposeLast = (l: Last) => {
    for (const x of [l.inputs.periphery, l.inputs.fovea, l.inputs.proprio, l.targetTensors.periphery, l.targetTensors.fovea]) x.dispose()
  }
  const scalar = async (x: Tensor) => (await x.read())[0]!
  const hTensor = (d: Float32Array) => t(d, [1, widths.hidden])
  const targetTensors = (y: Targets): TargetTensors => ({ periphery: t(y.periphery, [1, PRED_P]), fovea: t(y.fovea, [1, PRED_F]) })
  /** Every update outcome goes through here (online and replayed). */
  const countUpdate = (applied: boolean) => {
    if (applied) stats.consecutiveSkips = 0
    else {
      stats.skippedUpdates++
      stats.consecutiveSkips++
    }
  }

  // Everything construction allocates is registered here and freed if it fails part-way (e.g. the device refusing a
  // buffer while a checkpoint is restored), so a failed createLearner leaks nothing.
  const owned: { dispose(): void }[] = [model]
  const own = <T extends { dispose(): void }>(x: T): T => { owned.push(x); return x }
  constructing = own
  const opts_: Adam[] = []
  // Recurrent state: h = h_{t-1} (output of the last core call) and hIn = h_{t-2} (its input), detached.
  let h: Tensor
  let hData: Float32Array = new Float32Array(widths.hidden)
  let hIn: Tensor | null = null
  let hInData: Float32Array = new Float32Array(widths.hidden)
  let last: Last | null = null
  let busy = false
  /** readParams calls in flight (they may overlap each other and an exclusive call): teardown waits for them too. */
  let reading = 0
  let disposed = false
  /** A frozen run is open; closeFrozen disposes it (the learner's dispose does). */
  let frozenOpen = false
  let closeFrozen: (() => void) | null = null
  // concat(logits [1,76], value [1,1]) -> [1,77] as two exact 0/1 selection matmuls (no concat op in @blanktab/tensor),
  // so acting reads the heads once. Caller-owned constants, freed in dispose().
  let selL: Tensor
  let selV: Tensor
  try {
    // residualDecay acts on Model.residualParams only: the shift gate (main optimizer) and the output layers.
    const main = model.params.filter((p) => !outputSet.has(p))
    const decay = { weightDecay: settings.residualDecay }
    opts_.push(own(new Adam(main, { lr: settings.lr, ...decay, decayed: model.residualParams.filter((p) => !outputSet.has(p)) })))
    opts_.push(own(new Adam([...model.outputParams], { lr: settings.lr * settings.headLrScale, ...decay })))
    const selLogits = new Float32Array(ACTION_DIM * (ACTION_DIM + 1))
    for (let k = 0; k < ACTION_DIM; k++) selLogits[k * (ACTION_DIM + 1) + k] = 1
    const selValue = new Float32Array(ACTION_DIM + 1)
    selValue[ACTION_DIM] = 1
    selL = t(selLogits, [ACTION_DIM, ACTION_DIM + 1])
    selV = t(selValue, [1, ACTION_DIM + 1])
    if (saved) {
      // validateState checked every shape; the model already holds the parameters.
      opts_[0]!.setState(saved.adam[0])
      opts_[1]!.setState(saved.adam[1])
      rng.setState(saved.rng)
      for (const tr of saved.replay) replay.push(tr)
      Object.assign(stats, saved.stats)
      h = hTensor(saved.h)
      hData = saved.h
      if (saved.last) {
        const { obs, action } = saved.last
        const y = targetsFromObs(obs)
        hIn = hTensor(saved.last.hIn)
        hInData = saved.last.hIn
        last = { obs, inputs: upload(obsToInputs(obs)), action, targets: y, targetTensors: targetTensors(y) }
      }
    } else {
      h = own(model.zeroState())
    }
  } catch (e) {
    for (const x of owned.reverse()) x.dispose()
    throw e
  } finally {
    constructing = null
  }
  owned.length = 0
  const opt = {
    step: () => { for (const o of opts_) o.step() },
    zeroGrad: () => { for (const o of opts_) o.zeroGrad() },
    dispose: () => { for (const o of opts_) o.dispose() },
  }
  const ema = (prev: number, v: number) => (Number.isFinite(prev) ? prev + STATS_ALPHA * (v - prev) : v)

  /** Both outputs' MSE against y, weighted by size (the MSE over their concatenation, comparable with copyLastBaseline). */
  const sizeWeightedLoss = (p: TargetTensors, y: TargetTensors) =>
    add(scale(mse(p.periphery, y.periphery), W_P), scale(mse(p.fovea, y.fovea), W_F))

  /**
   * Size-weighted MSE of predict(core(inputs, hIn), action, base) against y. `base` is the targets of the frame
   * `inputs` came from (residual predictor). Builds a graph unless under noGrad.
   */
  const predictionLoss = (inputs: InputTensors, hIn: Tensor, action: Tensor, base: TargetTensors, y: TargetTensors) =>
    // A composite: if building it fails part-way, everything it made is freed (nothing unreachable is left).
    composite(() => sizeWeightedLoss(model.predict(model.core(inputs, hIn), action, base), y))

  /** The acting heads at h as one [1,77] tensor (76 logits then V(h)), no graph: acting reads them once. */
  const actingHeads = (hh: Tensor) => noGrad(() => composite(() => {
    const { logits, value } = model.heads(hh)
    return add(matmul(logits, selL), matmul(value, selV))
  }))

  /**
   * Holds the tensors of one pass (scoreHeldOut, a frozen run, a step): `keep` registers one, `drop` frees some now,
   * `release` hands some over to the caller, `freeAll` frees whatever is still held (at the end, or on any failure:
   * nothing leaks).
   */
  const tracker = () => {
    const held = new Set<Tensor>()
    const keep = <T extends Tensor>(x: T): T => { held.add(x); return x }
    const T = (d: Float32Array, shape: number[]) => keep(t(d, shape))
    return {
      keep,
      T,
      drop: (...xs: (Tensor | null)[]) => { for (const x of xs) if (x) { held.delete(x); x.dispose() } },
      /** Hands tensors over to the caller (no longer freed by freeAll). */
      release: (...xs: Tensor[]) => { for (const x of xs) held.delete(x) },
      freeAll: () => { for (const x of held) x.dispose(); held.clear() },
      inputsOf: (o: Observation): InputTensors => {
        const ins = obsToInputs(o)
        return {
          periphery: T(ins.periphery, [PERIPHERY.channels, PERIPHERY.height, PERIPHERY.width]),
          fovea: T(ins.fovea, [FOVEA.channels, FOVEA.height, FOVEA.width]),
          proprio: T(ins.proprio, [1, 3]),
        }
      },
      targetsOf: (y: Targets): TargetTensors => ({ periphery: T(y.periphery, [1, PRED_P]), fovea: T(y.fovea, [1, PRED_F]) }),
    }
  }

  /** −logπ(a)·A + (target − V)² − β·H(π) on the heads at the detached h. `adv` is stopgrad(target − V). */
  const actorCriticLoss = (hPrev: Tensor, action: Action, target: number) => {
    const { logits, value } = model.heads(hPrev)
    const adv = noGrad(() => composite(() => addScalar(neg(value), target)))
    const flat = reshape(logits, [ACTION_DIM, 1])
    const chosen = [action.move, action.button, action.scroll, action.key]
    let logp: Tensor | null = null
    let negEntropy: Tensor | null = null
    for (let f = 0; f < FACTORS.length; f++) {
      const n = FACTORS[f]!
      const lp = logSoftmax(reshape(gatherRows(flat, FACTOR_ROWS[f]!), [1, n]))
      const lpa = gatherRows(reshape(lp, [n, 1]), Int32Array.of(chosen[f]!))
      const ne = sum(mul(exp(lp), lp))
      logp = logp ? add(logp, lpa) : lpa
      negEntropy = negEntropy ? add(negEntropy, ne) : ne
    }
    const actor = neg(sum(mul(logp!, adv)))
    const critic = sum(square(addScalar(neg(value), target)))
    return { loss: add(add(actor, critic), scale(negEntropy!, settings.entropyWeight)), adv }
  }

  /** Builds the replay of one transition: its prediction loss (a graph unless under noGrad) and the constants it used. */
  const replayed = (tr: Transition) => {
    // Every constant is freed by `free`, and all of them if an upload fails part-way (nothing is left half-made).
    const made: Tensor[] = []
    const free = () => { for (const x of made.splice(0)) x.dispose() }
    const T = (d: Float32Array, shape: number[]) => { const x = t(d, shape); made.push(x); return x }
    try {
      const ins = obsToInputs(tr.obs), b = targetsFromObs(tr.obs), n = targetsFromObs(tr.next)
      const inputs: InputTensors = {
        periphery: T(ins.periphery, [PERIPHERY.channels, PERIPHERY.height, PERIPHERY.width]),
        fovea: T(ins.fovea, [FOVEA.channels, FOVEA.height, FOVEA.width]),
        proprio: T(ins.proprio, [1, 3]),
      }
      const base = { periphery: T(b.periphery, [1, PRED_P]), fovea: T(b.fovea, [1, PRED_F]) }
      const y = { periphery: T(n.periphery, [1, PRED_P]), fovea: T(n.fovea, [1, PRED_F]) }
      const hT = T(tr.hIn, [1, widths.hidden])
      const aT = T(actionOneHot(tr.action), [1, ACTION_DIM])
      return { loss: () => predictionLoss(inputs, hT, aT, base, y), free }
    } catch (e) {
      free()
      throw e
    }
  }

  async function sleepNow(): Promise<void> {
    const flops0 = backend.stats.flops
    for (let k = 0; k < settings.replayPerSleep; k++) {
      const tr = sampleStored(replay, reservoir, rng)
      if (!tr) break // nothing stored yet: no update, no draw
      const r = replayed(tr)
      let loss: Tensor | null = null
      try {
        loss = r.loss()
        const l = await scalar(loss)
        const ok = Number.isFinite(l)
        if (ok) {
          backward(loss)
          opt.step()
        }
        countUpdate(ok)
      } finally {
        opt.zeroGrad() // also after a failure part-way through backward: no parameter keeps a stray gradient
        if (loss) disposeGraph(loss, [])
        r.free()
      }
    }
    stats.sleeps++
    stats.cu += (backend.stats.flops - flops0) / 1e9
    await scoreHeldOut()
  }

  /**
   * The held-out check (decision 0015 §3): the current weights on the fixed held-out trajectory, forward only (noGrad,
   * no update, no RNG draw), the core run along it from the zero recurrent state. One read. Its FLOPs go to evalCU.
   */
  async function scoreHeldOut(): Promise<void> {
    if (!heldOut) return
    const flops0 = backend.stats.flops
    const n = heldOut.actions.length
    /** Every tensor this scoring holds right now (freed at the end, or on any failure: nothing leaks). */
    const { keep, drop, T, freeAll, inputsOf, targetsOf } = tracker()
    try {
      let hh = keep(model.zeroState())
      let total: Tensor | null = null
      let base = targetsOf(targetsFromObs(heldOut.obs[0]!))
      for (let i = 0; i < n; i++) {
        const inputs = inputsOf(heldOut.obs[i]!)
        const y = targetsOf(targetsFromObs(heldOut.obs[i + 1]!))
        const aT = T(actionOneHot(heldOut.actions[i]!), [1, ACTION_DIM])
        const h0 = hh
        const hNext = keep(noGrad(() => composite(() => model.core(inputs, h0))))
        drop(h0)
        hh = hNext
        const b = base
        const li = keep(noGrad(() => composite(() => sizeWeightedLoss(model.predict(hNext, aT, b), y))))
        const prev: Tensor | null = total
        total = prev ? keep(noGrad(() => composite(() => add(prev, li)))) : li
        if (prev) drop(prev, li) // li is held until it is merged into the running total
        // This step's constants are done with, except y: it is the next transition's base.
        drop(b.periphery, b.fovea, inputs.periphery, inputs.fovea, inputs.proprio, aT)
        base = y
      }
      const loss = (await scalar(total!)) / n
      stats.heldOutLoss = Number.isFinite(loss) ? loss : Number.NaN
      stats.heldOutBaseline = heldOutCopyLast
    } finally {
      freeAll()
      stats.evalCU += (backend.stats.flops - flops0) / 1e9
    }
  }

  async function replayLossNow(): Promise<number> {
    let total = 0
    // Each stored transition once: one held both as recent and in the reservoir is the same object.
    const stored = [...new Set([...replay.toArray(), ...(reservoir?.toArray() ?? [])])]
    for (const tr of stored) {
      const r = replayed(tr)
      try {
        const loss = noGrad(() => composite(r.loss))
        try {
          total += await scalar(loss)
        } finally {
          loss.dispose()
        }
      } finally {
        r.free()
      }
    }
    return stored.length > 0 ? total / stored.length : Number.NaN
  }

  async function saveNow(): Promise<LearnerState> {
    const [params, adam0, adam1] = (await settleAll<unknown>([readAll(model.params), opts_[0]!.state(), opts_[1]!.state()])) as [Float32Array[], AdamState, AdamState]
    // structuredClone: a snapshot that shares nothing with the learner (frames shared between consecutive
    // transitions stay shared within the snapshot).
    return structuredClone({
      version: LEARNER_STATE_VERSION,
      seed,
      widths: { ...widths },
      settings: { ...settings },
      params,
      adam: [adam0, adam1],
      rng: rng.getState(),
      h: hData,
      last: last && { obs: last.obs, action: last.action, hIn: hInData },
      replay: replay.toArray(),
      ...(reservoir ? { reservoir: { items: reservoir.toArray(), seen: reservoir.seen } } : {}),
      stats: { ...stats },
      heldOut,
    } satisfies LearnerState)
  }

  /**
   * Runs `fn` with the learner to itself: step, sleep, save, replayLoss and a frozen run's step never interleave.
   * Only a frozen run's own step may run while a frozen run is open. A dispose() made while `fn` runs takes effect
   * here, once it has settled (or when the last readParams in flight settles, whichever is later).
   */
  async function exclusive<T>(fn: () => Promise<T>, frozenStep = false): Promise<T> {
    if (disposed) throw new Error("learner: used after dispose")
    if (frozenOpen && !frozenStep) throw new Error(FROZEN_OPEN)
    if (busy) throw new Error(IN_PROGRESS)
    busy = true
    try {
      return await fn()
    } finally {
      busy = false
      if (disposed && reading === 0) teardown()
    }
  }

  /** readParams: not exclusive (it may run alongside a step), but a dispose() made meanwhile waits for it to settle. */
  async function readParamsNow(): Promise<Float32Array[]> {
    if (disposed) throw new Error("learner: used after dispose")
    reading++
    try {
      return await readAll(model.params)
    } finally {
      reading--
      if (disposed && !busy && reading === 0) teardown()
    }
  }

  /**
   * Frees everything the learner holds (once: dispose() calls it now, or else the last call in flight, exclusive or
   * readParams, when it settles).
   */
  function teardown(): void {
    closeFrozen?.()
    if (last) disposeLast(last)
    last = null
    selL.dispose()
    selV.dispose()
    h.dispose()
    hIn?.dispose()
    hIn = null
    opt.dispose()
    model.dispose()
  }

  async function learnAndAct(o: Observation): Promise<Action> {
    const flops0 = backend.stats.flops
    const obs = copyObs(o)
    const y = targetsFromObs(obs) // validates the frames (RangeError) before any state changes
    /**
     * This step's own tensors (inputs, yT, aT, after, hNext, headsT). If the step throws part-way they are all freed (the
     * graphs and the advantage are freed where they are made), and h, hIn and last are left as they were.
     */
    const tr = tracker()
    try {
      const inputs = tr.inputsOf(obs)
      const yT = tr.targetsOf(y)
      stats.steps++

      let info: UpdateInfo | null = null
      let aT: Tensor | null = null
      /** L_after's tensor (step 2), when L_before was finite: read together with the acting heads. */
      let after: Tensor | null = null
      stats.lastLoss = Number.NaN
      if (last && hIn) {
        const prev: Last = last
        const hPrev2: Tensor = hIn
        aT = tr.T(actionOneHot(prev.action), [1, ACTION_DIM])
        const a1 = aT
        info = { step: stats.steps, lossBefore: Number.NaN, lossAfter: Number.NaN, reward: 0, advantage: Number.NaN, skipped: true }

        // 1. Prediction update (TBPTT-1 through the core); the residual base is y_{t-1}.
        let lb = Number.NaN
        let lossBefore: Tensor | null = null
        try {
          lossBefore = predictionLoss(prev.inputs, hPrev2, a1, prev.targetTensors, yT)
          lb = await scalar(lossBefore)
          if (Number.isFinite(lb)) {
            stats.lastLoss = lb
            backward(lossBefore)
            opt.step()
          }
        } finally {
          opt.zeroGrad() // also after a failure part-way through backward: no parameter keeps a stray gradient
          if (lossBefore) disposeGraph(lossBefore, [])
        }
        // 2. Curiosity (built here, read with step 3's heads): the same loss with the updated weights.
        if (Number.isFinite(lb)) {
          info.lossBefore = lb
          after = tr.keep(noGrad(() => composite(() => predictionLoss(prev.inputs, hPrev2, a1, prev.targetTensors, yT))))
        }
      }

      // 3. Act: h_t and the heads (one [1,77] tensor: 76 logits then V(h_t)), no graph. L_after needs only the
      // updated weights too, so all three are read in one round.
      const hNext = tr.keep(noGrad(() => composite(() => model.core(inputs, h))))
      const headsT = tr.keep(actingHeads(hNext))
      const reads = await readAll(after ? [after, headsT, hNext] : [headsT, hNext])
      const [laOut, headsOut, hNextData] = (after ? reads : [null, ...reads]) as [Float32Array | null, Float32Array, Float32Array]

      // 2 (on the host). Curiosity relative to the running loss, then the moving averages.
      if (last && info) {
        if (laOut) {
          const la = laOut[0]!
          if (Number.isFinite(la)) {
            info.lossAfter = la
            // Scaled by the EMA from before this loss is folded in; the very first update has no scale yet (r = 0).
            info.reward = curiosityReward(info.lossBefore, la, stats.predLoss)
          }
          stats.predLoss = ema(stats.predLoss, info.lossBefore)
        }
        const base = copyLastBaseline(last.targets, y)
        if (Number.isFinite(base)) stats.baselineLoss = ema(stats.baselineLoss, base)
      }
      const logits = headsOut.subarray(0, ACTION_DIM)
      const valueNext = headsOut[ACTION_DIM]!
      const action = sampleAction(logits, settings.exploreEpsilon, rng)

      // 4. Actor-critic on a_{t-1}, only when the curiosity reward was measured.
      if (last && info && Number.isFinite(info.lossAfter)) {
        const target = settings.curiosityWeight * info.reward + settings.gamma * valueNext
        const { loss, adv } = actorCriticLoss(h, last.action, target)
        try {
          const [lac, advantage] = (await readAll([loss, adv])).map((v) => v[0]!) as [number, number]
          if (Number.isFinite(lac)) {
            backward(loss)
            opt.step()
            info.advantage = advantage
            info.skipped = false
          }
        } finally {
          opt.zeroGrad()
          disposeGraph(loss, [])
          adv.dispose()
        }
      }
      if (info) countUpdate(!info.skipped)

      // 5. Store the transition, carry the state, free the rest.
      if (last) {
        const stored: Transition = { obs: last.obs, hIn: hInData, action: last.action, next: obs }
        replay.push(stored)
        reservoir?.offer(stored, rng) // one draw once the reservoir is full; none without one
      }
      tr.drop(aT, after, headsT)
      tr.release(inputs.periphery, inputs.fovea, inputs.proprio, yT.periphery, yT.fovea, hNext)
      if (last) disposeLast(last)
      hIn?.dispose()
      hIn = h
      hInData = hData
      h = hNext
      hData = hNextData
      last = { obs, inputs, action, targets: y, targetTensors: yT }

      stats.cu += (backend.stats.flops - flops0) / 1e9
      if (info) opts.onUpdate?.(info)
      return action
    } finally {
      tr.freeAll() // empty after a step that completed
    }
  }

  /**
   * Opens a frozen run (spec §8.3): each step mirrors learnAndAct without its updates. From the zero state with its
   * own RNG: if measuring and there is a previous step, the size-weighted prediction loss of (o_{t-1}, a_{t-1}) → o_t
   * and copy-last's on the same transition (the same loss with the prediction replaced by the current frame, so a
   * newborn, whose prediction is exactly copy-last, scores exactly equal) are added to running sums; then the core and
   * heads, and a_t ~ sampleAction(…, the run's RNG). No graph, no update, no replay, nothing of the learner's state
   * touched; FLOPs to stats.evalCU. The prediction uses the carried h_{t-1} rather than recomputing it from h_{t-2}
   * as learnAndAct does: with the weights fixed the two are the same values.
   */
  function frozenRun(o: FrozenRunOptions): FrozenRun {
    if (disposed) throw new Error("learner: used after dispose")
    if (frozenOpen) throw new Error(FROZEN_OPEN)
    if (busy) throw new Error(IN_PROGRESS)
    const { measureLoss } = o
    const runRng = createRng(o.seed)
    const tr = tracker()
    /** h_{t-1}: the core's output on the previous observation (the zero state before the first). */
    let hh = tr.keep(model.zeroState())
    /** The previous step's action and, when measuring, its targets (the residual base of the next transition). */
    let prev: { action: Action; base: TargetTensors | null } | null = null
    let predSum = 0
    let baseSum = 0
    let transitions = 0
    let stepping = false
    let closed = false
    let failed = false
    const close = () => {
      tr.freeAll()
      frozenOpen = false
      closeFrozen = null
    }
    frozenOpen = true
    closeFrozen = () => run.dispose()

    async function frozenStep(ob: Observation): Promise<Action> {
      if (failed) throw new Error("learner: this frozen run failed part-way (dispose it)")
      const y = targetsFromObs(ob) // validates the frames (RangeError) before anything is allocated
      const flops0 = backend.stats.flops
      stepping = true
      try {
        const inputs = tr.inputsOf(ob)
        // The targets are only ever a loss's target or the next one's base: uploaded only when measuring.
        const yT = measureLoss ? tr.targetsOf(y) : null
        let aT: Tensor | null = null
        let lossT: Tensor | null = null
        let baseT: Tensor | null = null
        if (yT && prev?.base) {
          const pb = prev.base
          const h0 = hh
          const a = (aT = tr.T(actionOneHot(prev.action), [1, ACTION_DIM]))
          lossT = tr.keep(noGrad(() => composite(() => sizeWeightedLoss(model.predict(h0, a, pb), yT))))
          baseT = tr.keep(noGrad(() => composite(() => sizeWeightedLoss(pb, yT))))
        }
        const h0 = hh
        const hNext = tr.keep(noGrad(() => composite(() => model.core(inputs, h0))))
        const headsT = tr.keep(actingHeads(hNext))
        const [headsOut, l, b] = await readAll(lossT && baseT ? [headsT, lossT, baseT] : [headsT])
        const action = sampleAction(headsOut!.subarray(0, ACTION_DIM), settings.exploreEpsilon, runRng)
        if (l && b) {
          predSum += l[0]!
          baseSum += b[0]!
          transitions++
        }
        tr.drop(inputs.periphery, inputs.fovea, inputs.proprio, h0, headsT, aT, lossT, baseT, prev?.base?.periphery ?? null, prev?.base?.fovea ?? null)
        hh = hNext
        prev = { action, base: yT }
        return action
      } catch (e) {
        failed = true
        tr.freeAll()
        throw e
      } finally {
        stepping = false
        stats.evalCU += (backend.stats.flops - flops0) / 1e9
        if (closed) close() // disposed while this step ran
      }
    }

    const run: FrozenRun = {
      step: (ob) => (closed ? Promise.reject(new Error("learner: frozen run used after dispose")) : exclusive(() => frozenStep(ob), true)),
      result: () => (measureLoss && transitions > 0
        ? { predLoss: predSum / transitions, baselineLoss: baseSum / transitions, transitions }
        : { predLoss: Number.NaN, baselineLoss: Number.NaN, transitions: 0 }),
      dispose() {
        if (closed) return
        closed = true
        if (!stepping) close()
      },
    }
    return run
  }

  return {
    replay,

    frozenRun,

    step: (o: Observation) => exclusive(() => learnAndAct(o)),
    sleep: () => exclusive(sleepNow),
    save: () => exclusive(saveNow),
    replayLoss: () => exclusive(replayLossNow),

    stats: () => ({ ...stats }),

    readParams: readParamsNow,

    dispose() {
      if (disposed) return
      // From now on every call rejects. A call in flight (step, sleep, save, replayLoss, a frozen step, readParams) is
      // never freed under: the last of them to settle tears down.
      disposed = true
      if (!busy && reading === 0) teardown()
    },
  }
}
