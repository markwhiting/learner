// A learner's life: steps in the world, measured in compute units (CU), with sleeps, checkpoints and log files (v2).
import type { Observation, WorldView } from "@blanktab/interface"
import { FIRST_LAYER_PARAMS, clampSettings } from "@blanktab/learner"
import type { LearnerSettings, LearnerState, LearnerStats, TrainableLearner, Widths } from "@blanktab/learner"
import { hashSeed, type CanvasLike, type RoomKind, type World } from "@blanktab/world"
import { checkGenome, genomeSha, settingsMismatch, type GenomeSpace } from "@blanktab/meta"
import { clampCaregiverSettings, createCaregiver, type Caregiver, type CaregiverSettings } from "./caregiver"
import { runCheckup, variantSeedOf } from "./checkup"
import { DEFAULT_MAX_STEPS_PER_SEC, finalLogOf, thin, birthOf, type Birth, type Lived } from "./final"
import { deleteCheckpoint, fromRecord, loadCheckpoint, saveCheckpoint, tailEvalCUError, type Checkpoint } from "./checkpoint"
import { OFFLINE_KNOWLEDGE_SHA, type Manifest } from "./knowledge"
import {
  CONFIG_BUDGET, lifeBudget, parseMeta, sameBudget, FILTER_FLOATS_PER_CONV1, MAX_CHECKUPS_PER_FILE, MAX_RESTORES, MILESTONE_SUITE_VERSION, isCurveLoss, levelAtOrBelow, nextCrossing,
  type CheckupResult, type CurvePoint, type EvalBudget, type LogFileV2, type MetaBlock, type MetaEnd,
} from "./log"
import type { Milestone } from "./milestones"
import { openQueue } from "./queue"
// The log schedule lives in log.ts (browser-safe, shared with the gate); re-exported for existing callers.
export { FIRST_LOG_CU, LAST_DOUBLING_CU, LOG_STEP_CU, logThresholds, nextCrossing } from "./log"
// The final-log builder lives in final.ts (shared with the orphan sweep); its constant is re-exported for existing callers.
export { DEFAULT_MAX_STEPS_PER_SEC } from "./final"

export const DEFAULT_SLEEP_INTERVAL = 250
/** This many skipped updates in a row (a learner gone non-finite) restore the last checkpoint. */
export const RESTORE_AFTER_SKIPS = 20
/** Restores from one checkpoint before the life is given up as failing (no new checkpoint in between). */
export const MAX_RESTORES_PER_CHECKPOINT = 3
/** Five minutes: a GPU checkpoint is tens of MB, too much to write at every sleep. */
export const DEFAULT_CHECKPOINT_EVERY_MS = 300_000

/**
 * Where a life's meta block comes from (spec §7.3; Plan 05a). The harness never reads a genome: it stores the block in
 * every checkpoint and repeats it into every log file (with `filters` and `end` where they belong).
 */
export interface SessionMeta {
  /** The genome space the page decoded `birth` with: a newborn's block is checked against it (newbornBlock). */
  space: GenomeSpace
  /**
   * A newborn's draw (role search, or confirm). Its genome must decode to cfg.learnerSettings, cfg.caregiver and the
   * speed cap, as logged, and it must carry `budget` equal to the current config's (below); BadMetaBlockError otherwise.
   */
  birth: MetaBlock
  /**
   * The current config's evaluation budget, which a newborn's block must carry (default CONFIG_BUDGET, from
   * config/evaluation.json). A resumed life keeps the budget its checkpoint's block was born with (decision 0020).
   */
  budget?: EvalBudget
  /**
   * The block of a resumed life whose checkpoint has none (born before 05a): role and method "legacy", evalPanel null
   * (decision 0022: it keeps checkupSeed). `flat` is the life's settings as its log files record them.
   */
  legacy(flat: Record<string, number | string>): Promise<MetaBlock>
  /**
   * The speed cap and sleep interval a life resumed from a checkpoint with block `block` runs: its genome's (the page
   * decodes it, test knobs over it), so a resumed life never runs harness genes other than its own. A key left out
   * keeps the config's (e.g. a legacy block's unrecorded gene). Absent: the config's. A result that throws or is out
   * of range is said on console.warn and the config's are kept (a page bug never ends a forever life); if they are not
   * what the block's genome decodes to, that run's log files go without the block (its checkpoints keep it).
   */
  resumedHarness?(block: MetaBlock): { maxStepsPerSec?: number; sleepInterval?: number }
}

/** A newborn's cfg.meta.birth is not a block this life may write (runSession throws it before making a learner). */
export class BadMetaBlockError extends Error {
  constructor(learnerId: string, why: string) {
    super(`learner ${learnerId}: bad meta block: ${why}`)
    this.name = "BadMetaBlockError"
  }
}

export interface SessionConfig {
  learnerId: string
  codeSha: string
  knowledgeSha: string
  sessionsSeenAtBirth: number
  hostSig: LogFileV2["hostSig"]
  /** Birth facts for a newborn. A resumed learner keeps its own, from its checkpoint's learnerState. */
  seed: number
  widths: Widths
  /** Clamped (clampSettings) before use; the log records the clamped values. */
  learnerSettings: LearnerSettings
  /** Undefined (the default): the life never ends. Test knobs set one; its end is a final log. */
  lifetimeCU?: number
  /** Steps between sleeps. Default DEFAULT_SLEEP_INTERVAL. */
  sleepInterval?: number
  /** Default DEFAULT_MAX_STEPS_PER_SEC. */
  maxStepsPerSec?: number
  /**
   * At a sleep, a checkpoint is saved only if this long (ms, on the injected clock) has passed since the last save.
   * Default DEFAULT_CHECKPOINT_EVERY_MS; 0 = at every clean sleep. So a reload or restore can lose up to this much
   * learning (log-file crossings always save, so it never loses a logged one).
   */
  checkpointEveryMs?: number
  /**
   * The caregiver's settings: a birth fact (a resumed life uses its checkpoint's caregiver state). The caregiver is on
   * when these (or the checkpoint's caregiver) are set and the world has state, rooms and bringToFront.
   */
  caregiver?: CaregiverSettings
  /**
   * The meta-learner's block (Plan 05a). Absent: a newborn or a pre-05a resume gets no meta block (as before 05a), and
   * the short-lives switch does nothing. A checkpoint's own block is kept either way (see runSession).
   */
  meta?: SessionMeta
}

/**
 * What the session needs of a learner (a TrainableLearner has it). `onKnowledge`, if the learner has it, is handed
 * the manifest of each knowledge refresh (and, after a restore, the latest one); the 04a learner has none.
 */
export type SessionLearner = Pick<TrainableLearner, "step" | "stats" | "sleep" | "save" | "dispose"> & {
  onKnowledge?(m: Manifest): void
  /** Checkups need it (decision 0016); a learner without it gets none. */
  frozenRun?: TrainableLearner["frozenRun"]
  /** The filter snapshots (meta.filters at the life's budget levels) need it; a learner without it gets none. */
  readParams?: TrainableLearner["readParams"]
}
/** How the session asks for a learner: newborn, or from a checkpoint's state (createLearner's options minus the backend). */
export type LearnerInit = { seed: number; widths: Widths; settings: LearnerSettings } | { state: LearnerState }

/** Milliseconds. */
export interface Clock { now(): number; sleep(ms: number): Promise<void> }
const realClock: Clock = { now: () => performance.now(), sleep: ms => new Promise(r => setTimeout(r, ms)) }

export type SessionEvent =
  /**
   * Resumed from the checkpoint at `steps` and `cu`; `curveTail` is its curve since the latest log file. `lastLevel`: the
   * highest checkup level the life has had (0: none), from the checkpoint, so it is known when older log files are gone.
   */
  | { type: "resumed"; steps: number; cu: number; curveTail: CurvePoint[]; checkupTail: CheckupResult[]; lastLevel: number }
  /** After RESTORE_AFTER_SKIPS skipped updates in a row at `fromSteps`, the learner went back to its checkpoint at `toSteps`. */
  | { type: "restored"; fromSteps: number; toSteps: number; restores: number; lostCU: number }
  /** A log file was queued (with its checkpoint). A failed life's last file is final too (see runSession). */
  | { type: "logged"; seq: number; steps: number; cu: number; final: boolean }
  /** A checkup for log level `level` began (at a sleep). */
  | { type: "checkupStarted"; level: number }
  /** A checkup finished; its result goes into the next log file. (An abandoned one has no event of its own.) */
  | { type: "checkup"; result: CheckupResult }
  /** The caregiver carried the learner to room `to` after step `steps`. */
  | { type: "carried"; to: string; steps: number }
  /** The life's meta block, once per run (after the resumed event, or at birth). */
  | { type: "meta"; meta: MetaBlock }

/**
 * The page's requests to a running life (decision 0014), handled by runSession between steps, so never during a step
 * or a save. One inbox may outlive a run (the worker keeps it across GPU restores): requests wait for the next run.
 */
export class SessionInbox {
  #knowledge: Manifest | undefined
  #waiting: { resolve: (saved: boolean) => void; halt: boolean }[] = []
  #halted = false
  #shortLives = false

  /**
   * The manifest of refreshed knowledge, in force from the next clean step on (never while skipped updates are
   * outstanding): every later log file (and checkpoint) records its knowledgeSha. A new knowledgeSha is a boundary: the
   * points lived under the old one are first written in a log file of their own (runSession). The offline fallback's
   * (OFFLINE_KNOWLEDGE_SHA) is ignored.
   */
  setKnowledge(m: Manifest): void {
    if (m.knowledgeSha !== OFFLINE_KNOWLEDGE_SHA) this.#knowledge = m
  }
  /**
   * A checkpoint of the learner as it is now, outside the throttle (with no log file, unless `halt` or a crossing). True
   * once it is saved; false if it was not: skipped updates were outstanding (a save then would be of a learner going
   * non-finite), or the life ended first. The caller asks again later. With `halt`, a life that saves for it stops
   * right after the save, at that step (runSession returns `halted`): the page reloads onto exactly that checkpoint.
   * Once a run has halted, every request (one made during the halting save, one the page had given up waiting for and
   * asks again, any later one) is answered true at once: that save is the learner's current state, as nothing runs after it.
   */
  checkpointNow(opts: { halt?: boolean } = {}): Promise<boolean> {
    if (this.#halted) return Promise.resolve(true)
    return new Promise(resolve => this.#waiting.push({ resolve, halt: opts.halt ?? false }))
  }

  /**
   * The short-lives switch (decisions 0019, 0022), off by default. runSession reads it at clean sleeps only: a life with
   * a meta block ends at the first clean sleep where it is on, its budget checkup (lifeBudget(meta).budgetCU) is done and no checkup is
   * still due. So a life already past its budget ends at the next clean sleep once it is switched on; switching it off
   * never ends a life. Like the inbox, it outlives a run.
   */
  setShortLives(on: boolean): void {
    this.#shortLives = on
  }
  get shortLives(): boolean { return this.#shortLives }

  /** Whether a waiting request asks to halt (a running checkup is abandoned for it). */
  get haltWaiting(): boolean { return this.#waiting.some(r => r.halt) }

  /** Whether the last run halted at a forced save (and no run has started since). */
  get halted(): boolean { return this.#halted }

  /** runSession: the latest knowledge given (the same object until a newer one comes). */
  get knowledge(): Manifest | undefined { return this.#knowledge }
  /** runSession: the requests waiting now, taken (before the save that answers them: later ones wait for the next). */
  take(): { resolve: (saved: boolean) => void; halt: boolean }[] {
    return this.#waiting.splice(0)
  }
  /** runSession: answers every waiting request. */
  answer(saved: boolean): void {
    for (const r of this.take()) r.resolve(saved)
  }
  /** runSession: the run halted at a save that is current; every waiting request, and every later one, is answered true. */
  markHalted(): void {
    this.#halted = true
    this.answer(true)
  }
  /** runSession: a run starts; requests wait for it again. */
  markRunning(): void {
    this.#halted = false
  }
}

export interface SessionDeps {
  /**
   * drainTrace, if the world has one, is called after every action (nothing reads the trace yet). state, rooms and
   * bringToFront (privileged, harness only) are what the caregiver needs; without them it is off.
   */
  world: WorldView & { drainTrace?(): unknown } & Partial<Pick<World, "state" | "rooms" | "bringToFront">>
  createLearner(init: LearnerInit): Promise<SessionLearner>
  /** Holds the outbox and the checkpoints (one database, so a log and its checkpoint commit together). */
  idb: IDBFactory
  now: () => Date
  /** For the speed cap. Default: performance.now and setTimeout. */
  clock?: Clock
  /** Awaited after every step. */
  onTick?: (s: { steps: number; cu: number; obs: Observation }) => void | Promise<void>
  onEvent?: (e: SessionEvent) => void
  /**
   * Errors the caller recovers from by running the life again from its checkpoint (the page: a rejected GPU read on
   * WebGPU). They are rethrown as they are. Any other error ends the life: its final log is written first (runSession).
   */
  recoverable?: (e: unknown) => boolean
  /**
   * This learner's checkpoint record as stored, already read (by the startup sweep that adopted it): used instead of
   * reading it from `idb` again. Restores and later runs read the database as usual.
   */
  checkpointRecord?: unknown
  /** Refreshed knowledge and forced checkpoints from the page (SessionInbox). */
  inbox?: SessionInbox
  /**
   * Milestone checkups (decision 0016), on when this is set and the learner has frozenRun: a fresh canvas for each
   * variant world, and the suite (default MILESTONES_V1).
   */
  checkups?: { canvas(): CanvasLike; suite?: readonly Milestone[] }
}

export interface SessionResult {
  steps: number
  cu: number
  /** The next seq (= the number of log files this learner has written). */
  seq: number
  /** The checkpoint's step count when the life was resumed, else null. */
  resumedFrom: number | null
  /** The learner's last log file was already final: nothing ran (and any leftover checkpoint was removed). */
  alreadyFinal: boolean
  /** The life stopped after a forced save asked to halt (SessionInbox.checkpointNow): it is not over; its checkpoint is that save. */
  halted: boolean
  /** "budget": the short-lives switch ended the life at its budget checkup (decision 0019); its final file says so. */
  ended?: "budget"
}

/** base64 of the floats, little-endian float32 (meta.filters). */
function float32Base64(parts: readonly Float32Array[]): string {
  const n = parts.reduce((a, p) => a + p.length, 0)
  const view = new DataView(new ArrayBuffer(4 * n))
  let i = 0
  for (const p of parts) for (const x of p) view.setFloat32(4 * i++, x, true)
  const bytes = new Uint8Array(view.buffer)
  let bin = ""
  for (let j = 0; j < bytes.length; j += 0x8000) bin += String.fromCharCode(...bytes.subarray(j, j + 0x8000))
  return btoa(bin)
}

/**
 * The speed cap: awaited before every step, it holds consecutive steps at least 1000 / maxStepsPerSec ms apart
 * (never more than maxStepsPerSec a second, even after a stall). It always yields to the event loop.
 */
export function createPacer(maxStepsPerSec: number, clock: Clock = realClock): () => Promise<void> {
  const interval = 1000 / maxStepsPerSec
  let last = Number.NEGATIVE_INFINITY
  return async () => {
    await clock.sleep(Math.max(0, last + interval - clock.now()))
    last = clock.now()
  }
}

/** An error thrown by the caller's own callback (onTick, onEvent): the caller stopping the life, not the life failing. */
class Interrupted {
  constructor(readonly error: unknown) {}
}

/**
 * The curve point at `steps`, from the learner's stats: its own-stream losses and, at a sleep (`atSleep`, the default),
 * the held-out score that sleep just made, if the learner has one (decision 0015 §3). A point between sleeps (the end
 * of a life) carries no score: it would only repeat the last one. Null unless both own-stream losses pass isCurveLoss,
 * and the score is left out unless both of its values do: the same rule the parser applies, so every point written
 * parses. The one place a point is made (the session's log and the page's live curve).
 */
export function curvePoint(
  st: Pick<LearnerStats, "cu" | "predLoss" | "baselineLoss" | "heldOutLoss" | "heldOutBaseline">, steps: number, atSleep = true,
): CurvePoint | null {
  const { cu, predLoss, baselineLoss, heldOutLoss, heldOutBaseline } = st
  if (!isCurveLoss(predLoss) || !isCurveLoss(baselineLoss)) return null
  const scored = atSleep && isCurveLoss(heldOutLoss) && isCurveLoss(heldOutBaseline)
  return { cu, steps, predLoss, baselineLoss, ...(scored ? { heldOut: { loss: heldOutLoss, baseline: heldOutBaseline } } : {}) }
}

/**
 * Lives one learner's life (or the rest of it): for as long as it is run, or until it has used `lifetimeCU` compute units if there is one (a test knob).
 *
 * - Every `sleepInterval` steps the learner sleeps and a curve point (predLoss vs the copy-last baseline) is added if
 *   both are finite. A checkpoint is saved then if `checkpointEveryMs` have passed since the last save (on the
 *   injected clock); one that falls due during skips waits for the next clean sleep. A reload or restore therefore
 *   loses up to `checkpointEveryMs` of learning.
 * - A log file (v2) is queued each time CU crosses the next logThresholds() value (8, 16, ..., 512, then every 1024) and at the end of a test lifetime, carrying the
 *   curve points since the previous file. Each one is queued in one transaction with the checkpoint of that moment,
 *   so the checkpoint is never behind the latest log file: resuming never reuses a seq or goes back in steps or CU.
 * - No checkpoint is saved (and so no non-final log is written) while the learner has skipped updates outstanding;
 *   after RESTORE_AFTER_SKIPS in a row it is restored from the last checkpoint. Restores stay on the record: the log
 *   counts them (`restores`), the CU they threw away (`lostCU`) and their skipped updates (in `skippedUpdates`).
 * - The final log file removes the checkpoint in the same transaction.
 * - `inbox` (SessionInbox), checked after every clean step (none while skipped updates are outstanding): knowledge
 *   given there is in force from then on (each log file's knowledgeSha is the knowledge in force when it was written,
 *   and the checkpoint keeps it across a reload) and is handed to `learner.onKnowledge`. Knowledge with a new
 *   knowledgeSha is a boundary: a log file under the old sha (with its checkpoint, one transaction) is written first, so
 *   no log file holds points lived under two versions. A forced checkpoint is saved then, whatever the throttle, under
 *   the same rule as every other save: never while skipped updates are outstanding (it is answered false then, and when
 *   the life ends). One asked with `halt` (before a code reload) stops the life right after its save, and that save
 *   writes a log file with it (one transaction), so every point lived on this code is logged with this codeSha; the
 *   result says `halted`, no final log is written, and the inbox answers every later request true (markHalted).
 *   Boundary and halt logs are ordinary log files (the contributor's 16-per-PR cap counts them like any other).
 * - A life that fails (any error but the caller's recoverable ones and those its own onTick / onEvent throw: the
 *   restore limits, a checkpoint that will not load or resume, a full disk, the learner throwing) is ended on the
 *   record before the error is rethrown: a final log file built from its last checkpoint (steps, CU, seq, curve tail,
 *   restores; lostCU adds the CU lived since that checkpoint) goes into the outbox in the transaction that removes the
 *   checkpoint. Without a usable checkpoint it is built from the latest v2 log file, or from the life's counters if it
 *   has no log file yet; a life that took no step and wrote no log gets none (and its checkpoint is simply removed).
 *   Writing it is best effort: a failure there leaves the original error to be thrown.
 * - With a checkpoint for this learnerId the life resumes from it (its steps, CU, seq, curve and birth facts: seed,
 *   widths and settings from its learnerState, knowledgeSha and sessionsSeenAtBirth from the checkpoint; the config's
 *   are for newborns only); a learner with log files but
 *   no checkpoint cannot be resumed and throws, as does a checkpoint whose seq does not follow the latest log file.
 * - The world's room (decision 0017): each checkpoint records the room in front (when the world has state()). On a
 *   resume and on a restore, that room is brought to the front before the next frame, through
 *   world.bringToFront(room, "harness"): not a carry (no carried event, not in the caregiver's counts) and not a learner
 *   switch. The cursor and the rooms' own state are not restored. An older checkpoint, or a room the world lacks: as is.
 * - Checkups (decision 0016; on with `deps.checkups` and a learner with frozenRun): at a clean sleep, if the highest
 *   log level at or below cu has not been checked, the milestone suite runs on it through frozen runs (paced like the
 *   life, billed to evalCU only). One per sleep; its result goes into the next log file written and the checkpoint is
 *   saved at once. A halt asked during one abandons it (the level stays pending, so it runs once after the reload); a
 *   recoverable error inside one is rethrown as any other. Checkpoints carry the last level and the pending results.
 * - The caregiver (on with `cfg.caregiver` or a checkpoint's caregiver, in a world with state, rooms and
 *   bringToFront): after every step it observes the learner's last loss in the room that step's frame was seen in, when
 *   the frame before was seen in the same room (decision 0016: a transition across a carry or a taskbar switch is not a
 *   room's loss; nor is the first one of a run or after a restore), and every caregiverPeriod steps may carry the learner
 *   to another room through world.bringToFront (never during a checkup). Nothing it reads or
 *   decides reaches the learner. Its carries go into each log file, its settings into `settings`, its state into checkpoints.
 * - The meta block (Plan 05a): the checkpoint's if it has one; else, with `cfg.meta`, a resumed life's legacy block
 *   (cfg.meta.legacy of its flat settings, saved in a checkpoint at once; a failing or bad one is warned about and the
 *   life runs on without a block) or a newborn's `cfg.meta.birth` (checked: BadMetaBlockError); else none.
 *   It goes into every checkpoint and log file, and nothing of it reaches the learner. A crossing file at one of the life's budget levels
 *   carries `filters` (the first layer, through learner.readParams). With a meta block evalSeed (a confirmation life),
 *   checkups meet confirmSeed variants; with a meta block evalPanel, panelSeed variants (variantSeedOf). A failed life's final file has `end: "failed"`. With the inbox's short-lives switch on, at a
 *   clean sleep with its budget checkup (lifeBudget) done, no checkup due and no halt waiting, the life ends: its final file has
 *   `end: "budget"` and the result says `ended: "budget"`.
 */
export async function runSession(cfg: SessionConfig, deps: SessionDeps): Promise<SessionResult> {
  const lifetimeCU = cfg.lifetimeCU ?? Number.POSITIVE_INFINITY // no lifetime: the life ends only by failure or the caller
  // A resumed life's block may set both (cfg.meta.resumedHarness, below), once its checkpoint is read.
  let sleepInterval = cfg.sleepInterval ?? DEFAULT_SLEEP_INTERVAL
  let maxStepsPerSec = cfg.maxStepsPerSec ?? DEFAULT_MAX_STEPS_PER_SEC
  if (cfg.lifetimeCU !== undefined && !(Number.isFinite(cfg.lifetimeCU) && cfg.lifetimeCU > 0)) throw new RangeError(`lifetimeCU must be finite and > 0, got ${cfg.lifetimeCU}`)
  if (!(Number.isInteger(sleepInterval) && sleepInterval >= 1)) throw new RangeError(`sleepInterval must be an integer >= 1, got ${sleepInterval}`)
  if (!(Number.isFinite(maxStepsPerSec) && maxStepsPerSec > 0)) throw new RangeError(`maxStepsPerSec must be finite and > 0, got ${maxStepsPerSec}`)
  const checkpointEveryMs = cfg.checkpointEveryMs ?? DEFAULT_CHECKPOINT_EVERY_MS
  if (!(Number.isFinite(checkpointEveryMs) && checkpointEveryMs >= 0)) throw new RangeError(`checkpointEveryMs must be finite and >= 0, got ${checkpointEveryMs}`)
  const { learnerId: id } = cfg
  const { world, idb, now, onTick, onEvent } = deps
  const clock = deps.clock ?? realClock
  const queue = await openQueue(idb)
  const emit = (e: SessionEvent) => {
    try {
      onEvent?.(e)
    } catch (err) {
      throw new Interrupted(err)
    }
  }

  const latest = await queue.latest(id)
  if (latest?.final) {
    await deleteCheckpoint(idb, id)
    return { steps: latest.stepsTotal, cu: latest.schemaVersion === 2 ? latest.cu : 0, seq: latest.seq + 1, resumedFrom: null, alreadyFinal: true, halted: false }
  }
  // What a failed life's final log is built from, as far as the life got: its latest checkpoint, its birth facts and
  // its counters (live, once there is a learner).
  let cp: Checkpoint | undefined
  let birth: Birth | undefined
  let knowledgeSha = cfg.knowledgeSha
  let sessionsSeenAtBirth = cfg.sessionsSeenAtBirth
  let live: (() => Lived) | undefined
  /**
   * The caregiver's carries since `cp` was saved (decision 0017): counted as they happen and zeroed only once a save
   * has committed, so a save or log write that fails (after logFile took the caregiver's summary) or a learner too
   * broken to report cannot hide them. A restore and a failed life put them on the record as lost carries.
   */
  let carriesSinceSave = 0
  /** This run writes no meta block although the life has one (liveTheLife: its block is not what runs). */
  let metaOff = false

  /** The final log file of a life that failed (see runSession), or null if there is nothing to put on the record. */
  const failedLog = async (): Promise<LogFileV2 | null> => {
    let lived: Lived | undefined
    try {
      lived = live?.()
    } catch {
      lived = undefined // a learner too broken to report: its counters are not known
    }
    return finalLogOf({
      ctx: { learnerId: id, now, hostSig: cfg.hostSig, codeSha: cfg.codeSha, maxStepsPerSec, knowledgeSha, sessionsSeenAtBirth },
      cp, latest: await queue.latest(id), lived, birth, carriesSinceSave, end: "failed", ...(metaOff ? { noMeta: true } : {}),
    })
  }

  try {
    return await liveTheLife()
  } catch (e) {
    deps.inbox?.answer(false)
    if (e instanceof Interrupted) throw e.error
    if (deps.recoverable?.(e)) throw e
    try {
      const f = await failedLog()
      if (f) {
        await queue.put(f, { remove: id })
        onEvent?.({ type: "logged", seq: f.seq, steps: f.stepsTotal, cu: f.cu, final: true })
      } else {
        await deleteCheckpoint(idb, id)
      }
    } catch (err) {
      // Best effort (the disk may be full): the life's own error is the one to report. This one is said, not hidden.
      console.error(`learner ${id}: the failed life's final log was not written:`, err)
    }
    throw e
  }

  /**
   * A resumed pre-05a life's legacy block, or undefined (said on console.warn) if cfg.meta.legacy throws or gives a block
   * the parser refuses at the checkpoint's cu and steps, or one not of role legacy: a page bug never ends a forever life,
   * which then runs on without a block, as before 05a.
   */
  async function legacyBlock(m: SessionMeta, c: Checkpoint, flat: Record<string, number>): Promise<MetaBlock | undefined> {
    try {
      const block = await m.legacy(flat)
      const bad = block.role !== "legacy" ? `role ${JSON.stringify(block.role)}, not legacy` : parseMeta(block, { cu: c.cu, stepsTotal: c.steps, final: false })
      if (typeof bad === "string") throw new Error(bad)
      if (await genomeSha(block.genomeSchema, block.genome) !== block.genomeSha) throw new Error("its genomeSha does not match its genome")
      return block
    } catch (e) {
      console.warn(`learner ${id}: no legacy meta block (the life runs on without one):`, e)
      return undefined
    }
  }
  /**
   * A newborn's block, checked: the parser accepts it (no filters, end or heldOut; not a newborn reference nor a
   * confirmation newborn, which are frozen files), a confirmation life's seed is its evalSeed, its genome
   * lies in cfg.meta.space at that space's schema, its genomeSha is its genome's, and it decodes to the settings this
   * life runs and logs (`flat`). BadMetaBlockError otherwise, before any learner exists (the page falls back to another draw).
   */
  async function newbornBlock(m: SessionMeta, flat: Record<string, number>): Promise<MetaBlock> {
    const b = m.birth
    const bad = parseMeta(b, { cu: 0, stepsTotal: 0, final: false })
    if (typeof bad === "string") throw new BadMetaBlockError(id, bad)
    if (b.role === "legacy") throw new BadMetaBlockError(id, "a newborn's block cannot be role legacy")
    // A confirmation life's learner seed is its evalSeed (spec §4.5), as the gate checks (settings.seed = meta.evalSeed).
    if (b.role === "confirm" && flat.seed !== b.evalSeed) throw new BadMetaBlockError(id, `a confirmation life's seed ${flat.seed} is not its evalSeed ${b.evalSeed}`)
    const now = m.budget ?? CONFIG_BUDGET
    if (!b.budget || !sameBudget(b.budget, now)) {
      throw new BadMetaBlockError(id, `a newborn's block must carry the config's budget (${now.budgetCU} CU at ${now.levels.join(", ")})`)
    }
    if (b.genomeSchema !== m.space.schemaVersion) throw new BadMetaBlockError(id, `genomeSchema ${b.genomeSchema} is not the space's ${m.space.schemaVersion}`)
    const outside = checkGenome(m.space, b.genome)
    if (outside !== null) throw new BadMetaBlockError(id, `genome outside the space: ${outside}`)
    const off = settingsMismatch(m.space, b.genome, flat)
    if (off !== null) throw new BadMetaBlockError(id, `genome is not what runs: ${off}`)
    if (await genomeSha(b.genomeSchema, b.genome) !== b.genomeSha) throw new BadMetaBlockError(id, "its genomeSha does not match its genome")
    return b
  }

  /** A resumed life's block if it decodes to the settings this run logs (`flat`), as the gate checks; else undefined (warned). */
  function resumedBlockOrNone(space: GenomeSpace, b: MetaBlock, flat: Record<string, number>): MetaBlock | undefined {
    if (b.role === "legacy" || b.genomeSchema !== space.schemaVersion) return b // the gate checks neither against settings
    const off = settingsMismatch(space, b.genome, flat)
    if (off === null) return b
    console.warn(`learner ${id}: its block is not what this run runs (${off}); this run's log files go without it`)
    return undefined
  }

  async function liveTheLife(): Promise<SessionResult> {
    cp = deps.checkpointRecord !== undefined ? await fromRecord(deps.checkpointRecord, id) : await loadCheckpoint(idb, id)
    if (!cp && latest) throw new Error(`learner ${id} has log files up to seq ${latest.seq} but no checkpoint: its life cannot be resumed`)
    // Every log file is queued with its checkpoint, so the checkpoint's seq is exactly the one after the latest file.
    if (cp && cp.seq !== (latest?.seq ?? -1) + 1) {
      throw new Error(`learner ${id}: the checkpoint's seq ${cp.seq} does not follow its latest log file (seq ${latest?.seq ?? "none"})`)
    }
    if (cp?.meta && cfg.meta?.resumedHarness) {
      try {
        const h = cfg.meta.resumedHarness(structuredClone(cp.meta))
        const cap = h.maxStepsPerSec ?? maxStepsPerSec, every = h.sleepInterval ?? sleepInterval
        if (!(Number.isFinite(cap) && cap > 0) || !(Number.isInteger(every) && every >= 1)) throw new RangeError(`maxStepsPerSec ${cap}, sleepInterval ${every}`)
        maxStepsPerSec = cap
        sleepInterval = every
      } catch (e) {
        console.warn(`learner ${id}: its block's speed cap and sleep interval could not be used (the config's are):`, e)
      }
    }
    birth = cp ? birthOf(cp.learnerState, id) : { seed: cfg.seed, widths: cfg.widths, settings: clampSettings(cfg.learnerSettings) }
    const { seed, widths, settings } = birth
    // sessionsSeenAtBirth is a birth fact; knowledgeSha is the knowledge in force at the checkpoint (until the inbox
    // brings newer). A resumed life keeps the ones its checkpoint carries (older checkpoints do not; then the config's).
    knowledgeSha = cp?.knowledgeSha ?? cfg.knowledgeSha
    sessionsSeenAtBirth = cp?.sessionsSeenAtBirth ?? cfg.sessionsSeenAtBirth

    const inbox = deps.inbox
    inbox?.markRunning()
    /** The inbox's knowledge this run has put in force (none yet: the checkpoint's or the config's sha is). */
    let applied: Manifest | undefined
    /** Knowledge the inbox holds that this run has not put in force yet. */
    const freshKnowledge = (): Manifest | undefined => {
      const k = inbox?.knowledge
      return k && k !== applied ? k : undefined
    }
    const takeKnowledge = (k: Manifest) => {
      applied = k
      knowledgeSha = k.knowledgeSha
      learner.onKnowledge?.(k)
    }
    // Checkups (decision 0016): the highest level checked (0: none yet) and the results the next log file carries. A
    // checkpoint from before checkups has neither: lastLevel 0.
    // `tailSuite` is the suite the tail was recorded under (decision 0017: a checkpoint from before it has suite 1's);
    // it matters only while the tail is non-empty.
    const checkupsOf = (c: Checkpoint | undefined) => ({
      lastLevel: c?.checkups?.lastLevel ?? 0, tail: (c?.checkups?.tail ?? []).map(r => structuredClone(r)), tailSuite: c?.checkups?.suiteVersion ?? 1,
    })
    let { lastLevel, tail, tailSuite } = checkupsOf(cp)
    /** Checkpoints carry the checkups' state once a life has had them configured. */
    const keepCheckups = deps.checkups !== undefined || cp?.checkups !== undefined
    // The caregiver (a harness component; it acts on the world only, through bringToFront): its checkpoint's state, else
    // a newborn's from the config's settings, in a world it can carry the learner in.
    const careWorld = typeof world.state === "function" && typeof world.rooms === "function" && typeof world.bringToFront === "function"
    const careInit: Parameters<typeof createCaregiver>[0] | undefined = !careWorld ? undefined
      : cp?.caregiver ? { state: cp.caregiver }
      : cfg.caregiver ? { settings: clampCaregiverSettings(cfg.caregiver), seed: hashSeed(seed, "caregiver") }
      : undefined
    let caregiver: Caregiver | undefined = careInit ? createCaregiver(careInit) : undefined
    /** The life's settings as its log files record them (learner, speed cap, seed and the caregiver's). */
    const flatSettings = (): Record<string, number> => ({ ...settings, maxStepsPerSec, seed, ...(caregiver ? caregiver.state().settings : {}) })
    // The life's meta block (Plan 05a): its checkpoint's, kept whatever the config says (a life never changes its block,
    // and the gate holds it constant); else, with cfg.meta, a pre-05a life's legacy block or a newborn's draw.
    const block: MetaBlock | undefined = cp?.meta ?? (cfg.meta === undefined ? undefined : cp ? await legacyBlock(cfg.meta, cp, flatSettings()) : await newbornBlock(cfg.meta, flatSettings()))
    // What this run writes into its log files. A resumed block that no longer decodes to what runs (resumedHarness fell
    // back to config values, or the page has none) would be refused by the gate (settingsMismatch on every non-legacy
    // block of main's schema): this run's files go without it (said on console.warn), as does a failed life's final
    // file. Its checkpoints keep it, so a later run whose settings agree writes it again.
    const meta: MetaBlock | undefined = block && cp?.meta && cfg.meta ? resumedBlockOrNone(cfg.meta.space, block, flatSettings()) : block
    if (block && !meta) metaOff = true
    /**
     * The checkup variants (variantSeedOf): a confirmation life's confirmSeed, a panel's panelSeed, else checkupSeed.
     * From the life's block, not what this run writes: a run whose files go without the block (above) is still that
     * life, on its own worlds.
     */
    const variantSeed = variantSeedOf(block)
    /** The room the previous frame was seen in (undefined: no previous frame this run, or a restore since). */
    let prevRoom: string | undefined
    /**
     * Brings the checkpoint's room to the front (decision 0017), so the learner's first frame after a resume or restore
     * is seen in the room the life's next frame would have been seen in. A harness focus, not a carry (the caregiver is not told, and
     * no carried event) nor a learner switch (M6 counts source "learner" only). The learner sees only the frames that
     * follow. Nothing without cp.world (an older checkpoint) or for a room this world does not have.
     */
    const restoreRoom = (c: Checkpoint | undefined) => {
      const room = c?.world?.room
      if (room === undefined || typeof world.rooms !== "function" || typeof world.bringToFront !== "function") return
      if (world.rooms().includes(room as RoomKind)) world.bringToFront(room as RoomKind, "harness")
    }
    let learner = await deps.createLearner(cp ? { state: cp.learnerState } : { seed, widths, settings })
    try {
      let steps = cp?.steps ?? 0
      let seq = cp?.seq ?? 0
      let curve: CurvePoint[] = cp ? [...cp.curveTail] : []
      let crossing = nextCrossing(learner.stats().cu)
      let restoresSinceSave = 0
      // Restores stay on the record across the whole life (and reloads, through the checkpoint).
      let restores = cp?.restores ?? 0
      let lostCU = cp?.lostCU ?? 0
      let lostSkips = cp?.lostSkips ?? 0
      let lastSavedAt = clock.now()
      const resumedFrom = cp ? cp.steps : null
      live = () => ({ steps, seq, curve, restores, lostCU, lostSkips, stats: learner.stats() })
      restoreRoom(cp)
      if (cp) emit({ type: "resumed", steps: cp.steps, cu: cp.cu, curveTail: [...cp.curveTail], checkupTail: tail.map(r => structuredClone(r)), lastLevel })
      if (meta) emit({ type: "meta", meta: structuredClone(meta) })

      const checkpointNow = async (): Promise<Checkpoint> => {
        const c: Checkpoint = {
          learnerId: id, savedAt: now().toISOString(), steps, cu: learner.stats().cu, seq, curveTail: [...curve],
          restores, lostCU, lostSkips, knowledgeSha, sessionsSeenAtBirth, codeSha: cfg.codeSha,
          // suiteVersion only for a non-empty tail of a suite other than 1 (absent means suite 1): pre-04c code rejects
          // the key, so a checkpoint carries it only when it must (decision 0017, rollback).
          ...(keepCheckups ? { checkups: { lastLevel, tail: tail.map(r => structuredClone(r)), ...(tail.length > 0 && tailSuite !== 1 ? { suiteVersion: tailSuite } : {}) } } : {}),
          ...(caregiver ? { caregiver: caregiver.state() } : {}),
          ...(typeof world.state === "function" ? { world: { room: world.state().room } } : {}),
          ...(block ? { meta: structuredClone(block) } : {}),
          learnerState: await learner.save(),
        }
        // Never write a checkpoint that would not load (checkCheckpoint): a tail is billed inside the learner's evalCU.
        const tailError = c.checkups ? tailEvalCUError(c.checkups, c.learnerState) : null
        if (tailError) throw new Error(`learner ${id}: ${tailError}`)
        return c
      }
      const checkupsOn = () => deps.checkups !== undefined && typeof learner.frozenRun === "function"
      /**
       * meta.filters for a file at `cu` (spec §5.1): the first layer's filters (FIRST_LAYER_PARAMS) when levelAtOrBelow(cu)
       * is one of the life's own budget levels (lifeBudget(meta)), the life has a meta block and the learner can read its
       * params; else undefined.
       */
      const filtersAt = async (cu: number): Promise<string | undefined> => {
        const level = levelAtOrBelow(cu)
        if (!meta || !learner.readParams || level === null || !lifeBudget(meta).levels.includes(level)) return undefined
        const params = await learner.readParams()
        const parts = FIRST_LAYER_PARAMS.map(i => params[i])
        const n = parts.reduce((a, p) => a + (p?.length ?? 0), 0)
        // Never a file the parser refuses: exactly FILTER_FLOATS_PER_CONV1 per unit of widths.conv1.
        // A mismatch is said and the filters left out: a snapshot is not worth a life (decision 0022).
        if (parts.some(p => p === undefined) || n !== FILTER_FLOATS_PER_CONV1 * widths.conv1) {
          console.warn(`learner ${id}: the first layer has ${n} values, not ${FILTER_FLOATS_PER_CONV1}·${widths.conv1}; this file has no filters`)
          return undefined
        }
        return float32Base64(parts as Float32Array[])
      }
      /**
       * The log file of this moment. It takes the checkup results and the caregiver's carries since the last file. Its
       * meta block (if the life has one) carries `filters` and `end` when given.
       */
      const logFile = (final: boolean, filters?: string, end?: MetaEnd): LogFileV2 => {
        // checkupAtSleep never lets the tail grow past this; a file with more would be refused by the gate.
        if (tail.length > MAX_CHECKUPS_PER_FILE) throw new Error(`log file: ${tail.length} checkups, more than ${MAX_CHECKUPS_PER_FILE}`)
        const st = learner.stats()
        const checked = checkupsOn() || tail.length > 0 ? { checkups: tail, milestoneSuiteVersion: tail.length > 0 ? tailSuite : MILESTONE_SUITE_VERSION } : {}
        tail = []
        return {
          schemaVersion: 2, learnerId: id, seq, codeSha: cfg.codeSha, knowledgeSha, writtenAt: now().toISOString(),
          stepsTotal: steps, settings: flatSettings(), hostSig: { ...cfg.hostSig },
          sessionsSeenAtBirth, final, cu: st.cu, condition: "A", widths: { ...widths }, curve: [...curve],
          skippedUpdates: st.skippedUpdates + lostSkips, restores, lostCU, evalCU: st.evalCU,
          ...checked, ...(caregiver ? { caregiver: caregiver.takeSummary() } : {}),
          ...(meta ? { meta: { ...structuredClone(meta), ...(filters !== undefined ? { filters } : {}), ...(end ? { end } : {}) } } : {}),
        }
      }
      const addPoint = (atSleep: boolean) => {
        const p = curvePoint(learner.stats(), steps, atSleep)
        if (!p || curve.at(-1)?.steps === steps) return
        curve = thin([...curve, p])
      }
      /** After any save of the learner's current state. */
      const saved = () => {
        restoresSinceSave = 0
        carriesSinceSave = 0 // they are in the saved caregiver's carries now
        lastSavedAt = clock.now()
      }

      // A newborn's first checkpoint, so there is always one to restore. A resumed life given its block just now (a
      // legacy one) saves at once too, so every checkpoint it may restore or fail from has the block.
      if (!cp || (cp.meta === undefined && block !== undefined)) {
        const first = await checkpointNow()
        await saveCheckpoint(idb, first)
        cp = first
        saved()
      }
      let restorePoint: Checkpoint = cp
      const pace = createPacer(maxStepsPerSec, clock)
      let halted = false
      /**
       * At a clean sleep: the checkup for the highest log level due and not yet checked, if any (one per sleep). True
       * if one finished (its result waits in `tail` for the next log file; the caller saves a checkpoint at once). A halt
       * asked meanwhile abandons it: the level stays pending. Its frozen steps are paced like the life's own. The level
       * also stays pending while the tail is full (MAX_CHECKUPS_PER_FILE) or holds another suite's results (a resumed
       * tail: one file carries one suite), until the next log file takes the tail.
       */
      const checkupAtSleep = async (): Promise<boolean> => {
        const c = deps.checkups
        const l = learner
        if (!c || !l.frozenRun) return false
        if (tail.length >= MAX_CHECKUPS_PER_FILE || (tail.length > 0 && tailSuite !== MILESTONE_SUITE_VERSION)) return false
        const st = l.stats()
        if (st.consecutiveSkips !== 0) return false
        const level = levelAtOrBelow(st.cu)
        if (level === null || level <= lastLevel) return false
        emit({ type: "checkupStarted", level })
        const out = await runCheckup({
          learnerId: id, level, steps, learner: { frozenRun: o => l.frozenRun!(o), stats: () => l.stats() },
          canvas: () => c.canvas(), ...(c.suite ? { suite: c.suite } : {}),
          ...(variantSeed ? { variantSeed } : {}),
          ...(deps.recoverable ? { recoverable: deps.recoverable } : {}),
          shouldStop: () => inbox?.haltWaiting ?? false, yieldNow: pace,
        })
        if (out.status !== "done") return false
        tail.push(out.result)
        tailSuite = MILESTONE_SUITE_VERSION
        lastLevel = level
        emit({ type: "checkup", result: out.result })
        caregiver?.onCheckup(out.result.passed) // after the checkup event: it takes no part in the checkup itself
        return true
      }

      /** Whether a checkup is due that has not run (the highest level at or below cu is above the last one checked). */
      const checkupDue = (cu: number) => checkupsOn() && (levelAtOrBelow(cu) ?? 0) > lastLevel
      /** Set when the short-lives switch ends the life. */
      let ended: "budget" | undefined
      while (learner.stats().cu < lifetimeCU) {
        await pace()
        const room = caregiver ? world.state!().room : undefined // the room the coming frame is seen in
        const obs = world.observe()
        world.act(await learner.step(obs))
        world.drainTrace?.()
        steps++
        try {
          await onTick?.({ steps, cu: learner.stats().cu, obs })
        } catch (err) {
          throw new Interrupted(err)
        }
        let checkpointDue = false
        let saveNow = false // a checkup finished: its level is saved at once, so a reload never checks it again
        const slept = steps % sleepInterval === 0
        if (slept) {
          await learner.sleep()
          addPoint(true)
          checkpointDue = true
          saveNow = await checkupAtSleep()
        }
        if (caregiver) {
          if (room === prevRoom) caregiver.observe(room!, learner.stats().lastLoss) // within one room only
          prevRoom = room
          const to = caregiver.decide(world.state!().room, world.rooms!())
          if (to !== null) {
            world.bringToFront!(to as RoomKind)
            carriesSinceSave++
            emit({ type: "carried", to, steps })
          }
        }
        const st = learner.stats()
        if (st.consecutiveSkips >= RESTORE_AFTER_SKIPS) {
          if (restores >= MAX_RESTORES) throw new Error(`learner ${id}: ${MAX_RESTORES} restores, the most a log file may record`)
          if (++restoresSinceSave > MAX_RESTORES_PER_CHECKPOINT) {
            throw new Error(`learner ${id}: still skipping updates after ${MAX_RESTORES_PER_CHECKPOINT} restores from the checkpoint at step ${restorePoint.steps}`)
          }
          const back = await loadCheckpoint(idb, id)
          if (!back) throw new Error(`learner ${id}: no checkpoint to restore`)
          // Every log file carries its checkpoint, so nothing was logged after this one: seq must not move.
          if (back.seq !== seq) throw new Error(`learner ${id}: the checkpoint's seq ${back.seq} is not the next seq ${seq}`)
          learner.dispose()
          learner = await deps.createLearner({ state: back.learnerState })
          if (applied) learner.onKnowledge?.(applied)
          const now_ = learner.stats()
          restores++
          lostCU += st.cu - now_.cu
          lostSkips += st.skippedUpdates - now_.skippedUpdates
          // On the record at once, so a reload before the next save does not forget the restore.
          // Checkups and caregiver go back with it (a checkpoint from before them: as this run began).
          ;({ lastLevel, tail, tailSuite } = checkupsOf(back))
          if (careInit) {
            // The carries made since that checkpoint are thrown away with the learner's state: on the record as lost
            // (decision 0017). The checkpoint's caregiver keeps those earlier restores since the last file threw away.
            caregiver = createCaregiver(back.caregiver ? { state: back.caregiver } : careInit)
            caregiver.noteLost(carriesSinceSave)
          }
          carriesSinceSave = 0 // in the restored caregiver's lost count now
          prevRoom = undefined // the restored learner's last frame is not this one
          restoreRoom(back)
          // The restored caregiver (with its lost count) is saved, so a reload right after keeps the count.
          cp = restorePoint = { ...back, restores, lostCU, lostSkips, ...(caregiver ? { caregiver: caregiver.state() } : {}) }
          await saveCheckpoint(idb, cp)
          lastSavedAt = clock.now()
          emit({ type: "restored", fromSteps: steps, toSteps: back.steps, restores, lostCU })
          steps = back.steps
          curve = [...back.curveTail]
          crossing = nextCrossing(learner.stats().cu)
          continue
        }
        if (st.cu >= lifetimeCU) break
        if (st.consecutiveSkips > 0) {
          inbox?.answer(false) // not now: the page asks again later
          continue // no checkpoint now; a crossing waits until updates apply again
        }
        // Short lives (decisions 0019, 0022): the switch is read at clean sleeps only. A life with a meta block ends once
        // its budget checkup is done and no checkup is still due; a halt waiting goes first (the resumed run ends).
        // The budget is the life's own (born with it), never the current config's.
        if (slept && meta && inbox?.shortLives && !inbox.haltWaiting && lastLevel >= lifeBudget(meta).budgetCU && !checkupDue(st.cu)) {
          ended = "budget"
          break
        }
        // Taken before the save: a request made while it is written waits for the next step's.
        const forced = inbox?.take() ?? []
        const halt = forced.some(r => r.halt)
        const fresh = freshKnowledge()
        // A log file now: at a crossing; before a halt (a code reload: the points so far are this code's); and before a
        // new knowledgeSha comes into force (the points so far are the old knowledge's). One file covers all three.
        if (st.cu >= crossing || halt || (fresh !== undefined && fresh.knowledgeSha !== knowledgeSha)) {
          const filters = st.cu >= crossing ? await filtersAt(st.cu) : undefined // a crossing at 512, 1024 or 2048
          const f = logFile(false, filters) // under the code and knowledge the points in it were lived with
          seq++
          curve = []
          crossing = nextCrossing(st.cu)
          if (fresh) takeKnowledge(fresh) // the checkpoint saved with the file is already under the new knowledge
          const next = await checkpointNow()
          await queue.put(f, { save: next })
          cp = restorePoint = next
          saved()
          emit({ type: "logged", seq: f.seq, steps: f.stepsTotal, cu: f.cu, final: false })
        } else {
          if (fresh) takeKnowledge(fresh) // the same sha (new content): nothing to separate
          if (forced.length > 0 || saveNow || (checkpointDue && clock.now() - lastSavedAt >= checkpointEveryMs)) {
            const next = await checkpointNow()
            await saveCheckpoint(idb, next)
            cp = restorePoint = next
            saved()
          }
        }
        for (const r of forced) r.resolve(true)
        if (halt) {
          halted = true
          break
        }
      }

      if (halted) {
        inbox?.markHalted() // a request made during the save, or the page asking again, is answered true: the save is current
        return { steps, cu: learner.stats().cu, seq, resumedFrom, alreadyFinal: false, halted: true }
      }

      addPoint(false) // the end of the life: no sleep here, so no held-out score of its own
      // The final file is also a crossing file when it falls on one (a short life's budget sleep can).
      const cuNow = learner.stats().cu
      const end = logFile(true, cuNow >= crossing ? await filtersAt(cuNow) : undefined, ended)
      await queue.put(end, { remove: id })
      cp = undefined // the life is over: nothing left to fail
      live = undefined
      seq++
      inbox?.answer(false) // the life is over: there is no checkpoint now
      emit({ type: "logged", seq: end.seq, steps: end.stepsTotal, cu: end.cu, final: true })
      return { steps, cu: learner.stats().cu, seq, resumedFrom, alreadyFinal: false, halted: false, ...(ended ? { ended } : {}) }
    } finally {
      learner.dispose()
    }
  }
}
