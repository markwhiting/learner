// The learner's thread: owns the OffscreenCanvas, the world, the tensor backend, the learner and its session.
// The session writes log files and checkpoints to the shared IndexedDB; this thread tells the page what happened.
// It never sees the token.
import { WORLD } from "@blanktab/interface"
import { createWorld } from "@blanktab/world"
import { buildHeldOut } from "@blanktab/harness/heldout"
import { GPU_WIDTHS, LearnerStateVersionError, createLearner, type Widths } from "@blanktab/learner"
import { createWebGpuBackend } from "@blanktab/tensor"
import type { Manifest } from "@blanktab/harness/knowledge"
import { CONFIG_BUDGET, budgetError, lifeBudget, type CheckupResult, type CurvePoint, type EvalBudget, type LogFile } from "@blanktab/harness/log"
import { runNewborn } from "@blanktab/harness/newborn"
import { openQueue } from "@blanktab/harness/queue"
import {
  BadMetaBlockError, DEFAULT_MAX_STEPS_PER_SEC, SessionInbox, curvePoint, runSession, type LearnerInit, type SessionConfig, type SessionLearner, type SessionMeta,
  type SessionResult,
} from "@blanktab/harness/session"
import { hasCheckpoint } from "@blanktab/harness/checkpoint"
import { chooseBackend, type Chosen } from "./backend"
import { checkpointAnswers } from "./checkpoints"
import { birthFromDraw, drawContextOf, sessionMetaOf, withoutPosterior, type Birth, type DrawContext } from "./birth"
import { frameGate } from "./frames"
import { isRecoverable, liveOnBestBackend, resumeStatus, seedsOf, type Restore, type RunLife } from "./life"
import { checkupStatus, createCarryTally, fmtLevel, type CarryFile } from "./milestones"
import type { FromWorker, StartMessage, ToWorker } from "./protocol"
import { shortLivesGate } from "./shortlives"

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null
  postMessage(m: FromWorker, transfer?: Transferable[]): void
}
const post = (m: FromWorker, transfer: Transferable[] = []) => scope.postMessage(m, transfer)

const FRAME_MS = 33 // at most ~30 frames a second reach the page
/** No frames while the page is hidden (the page says when it is). */
const frames = frameGate(FRAME_MS)
const RATE_WINDOW_MS = 3000
/** The page's refreshed knowledge and forced checkpoints, for whichever run of the life is current (it outlives GPU restores). */
const inbox = new SessionInbox()
/** Forced checkpoints: through the inbox while a life runs, answered at once (not saved) while a confirmation newborn does. */
const checkpoints = checkpointAnswers(inbox, m => post(m))
/** The short-lives switch as the session gets it: a life resumed past its budget needs the page's confirm (shortlives.ts). */
let shortLives: ReturnType<typeof shortLivesGate> | undefined

/** The widths class the gate recomputes a draw with (tools/gate/meta.ts): "gpu" exactly at the GPU widths. */
const widthsClassOf = (w: Widths): "gpu" | "cpu" =>
  w.conv1 === GPU_WIDTHS.conv1 && w.conv2 === GPU_WIDTHS.conv2 && w.enc === GPU_WIDTHS.enc && w.hidden === GPU_WIDTHS.hidden ? "gpu" : "cpu"

/** The page's raw meta-learner files as a draw context (null: no valid space or centre). */
const contextOf = (draw: StartMessage["draw"]): Promise<DrawContext | null> =>
  draw ? drawContextOf({ meta: { genomeSpace: draw.spaceJson, genome: draw.genomeJson, posteriorText: draw.posteriorText } }) : Promise.resolve(null)

/** A newborn's evaluation budget: the config's, or the test knob's (one level, the budget itself). */
function budgetOf(budgetCU: number | undefined): EvalBudget {
  if (budgetCU === undefined) return CONFIG_BUDGET
  const b = { budgetCU, levels: [budgetCU] }
  const bad = budgetError(b)
  if (bad !== null) throw new RangeError(`a test budget of ${budgetCU} CU: ${bad}`)
  return b
}

/**
 * One way a newborn can be born: a drawn genome (its birth and the session's meta source), or none (no meta block,
 * the config's fallback settings). Tried in order (the draw, then the centre, then none) until the session accepts one.
 */
interface Candidate { name: string; birth?: Birth; meta?: SessionMeta; unlabelled?: string }

/**
 * A newborn's births: the candidates a session tries in order, and, when the draw was a confirmation's newborn arm,
 * that birth (with the context it was drawn in): it runs as one frozen newborn instead of a life (runConfirmNewborn),
 * and the candidates are its fallbacks.
 */
interface Births { list: Candidate[]; newbornArm?: { birth: Birth; ctx: DrawContext } }

/** `lrn_` + 16 hex from the crypto RNG: a newborn reference's own learnerId (it never takes over this tab's). */
function freshLearnerId(): string {
  return "lrn_" + Array.from(crypto.getRandomValues(new Uint8Array(8)), b => b.toString(16).padStart(2, "0")).join("")
}

/** Steps per second over the last RATE_WINDOW_MS. */
function rateMeter(): (steps: number) => number {
  const samples: [number, number][] = []
  return steps => {
    const t = performance.now()
    samples.push([t, steps])
    while (samples.length > 2 && samples[0]![0] < t - RATE_WINDOW_MS) samples.shift()
    const [t0, s0] = samples[0]!
    return t > t0 ? ((steps - s0) * 1000) / (t - t0) : 0
  }
}

/**
 * The birth candidates for life `id` at `widths` (the first run's): its draw (birthFromDraw), then a centre draw if the
 * draw was not already centre, then none. A candidate that cannot be built is skipped, with the reason on the console.
 * A draw of a confirmation's newborn arm is not a candidate (a session never runs it): it is `newbornArm`.
 */
async function candidatesOf(m: StartMessage, widths: Widths): Promise<Births> {
  const id = m.cfg.learnerId
  const ctx = await contextOf(m.draw)
  if (!ctx) return { list: [{ name: "none", unlabelled: m.draw ? "the genome files could not be read" : "no genome files" }] }
  const budget = budgetOf(m.overrides?.budgetCU)
  const out: Candidate[] = []
  let newbornArm: Births["newbornArm"]
  const add = async (name: string, c: DrawContext) => {
    try {
      const birth = await birthFromDraw(c, id, widthsClassOf(widths), budget)
      if (birth.reason !== undefined) console.warn(`blanktab: learner ${id}: the posterior gave the centre: ${birth.reason}`)
      if (birth.kind === "confirmNewborn") { newbornArm = { birth, ctx: c }; return }
      const knobs = { ...(m.overrides?.maxStepsPerSec !== undefined ? { maxStepsPerSec: m.overrides.maxStepsPerSec } : {}), ...(m.overrides?.sleepInterval !== undefined ? { sleepInterval: m.overrides.sleepInterval } : {}) }
      out.push({ name, birth, meta: sessionMetaOf(c, birth.meta, budget, knobs) })
    } catch (e) {
      console.warn(`blanktab: learner ${id}: no ${name} birth:`, e)
    }
  }
  await add("drawn", ctx)
  if (ctx.posterior !== null) await add("centre", withoutPosterior(ctx))
  out.push({ name: "none", unlabelled: "the session refused its genome blocks (see the console)" })
  return { list: out, ...(newbornArm ? { newbornArm } : {}) }
}

/**
 * The config a candidate runs: its decoded settings over the page's fallback ones (a resumed life keeps its own cap),
 * then the test knobs. A resumed life takes its settings and caregiver from its checkpoint whatever these say.
 */
function configOf(c: SessionConfig, m: StartMessage, cand: Candidate): SessionConfig {
  const b = cand.birth
  const o = m.overrides ?? {}
  return {
    ...c,
    ...(b ? { learnerSettings: b.learnerSettings, caregiver: b.caregiver, ...(m.ownCap ? {} : { maxStepsPerSec: b.maxStepsPerSec, sleepInterval: b.sleepInterval }) } : {}),
    // A confirmation life's learner seed is its evalSeed (a resumed life's comes from its checkpoint whatever this says).
    ...(b?.seed !== undefined ? { seed: b.seed } : {}),
    ...(cand.meta ? { meta: cand.meta } : {}),
    ...(o.maxStepsPerSec !== undefined ? { maxStepsPerSec: o.maxStepsPerSec } : {}),
    ...(o.sleepInterval !== undefined ? { sleepInterval: o.sleepInterval } : {}),
  }
}

/** A learner on `chosen`'s backend; a newborn's held-out set (decision 0015 §3) is recorded at birth, from a world of its own. */
const learnerOn = (chosen: Chosen, init: LearnerInit) => createLearner({
  backend: chosen.backend, ...init,
  ...("state" in init ? {} : { heldOut: buildHeldOut(init.seed, new OffscreenCanvas(WORLD.width, WORLD.height)) }),
})

async function start(m: StartMessage): Promise<void> {
  const { cfg, backend: force, widths, hidden, checkpointRecord } = m
  shortLives = shortLivesGate(inbox, { on: m.shortLives, consent: m.shortLivesConsent ?? false })
  const gate = shortLives
  frames.setHidden(hidden ?? false)
  if (m.newborns) return newborns(m, m.newborns)
  /** Used by the first run only: a restore after a GPU failure reads the (updated) checkpoint from the database. */
  let record = checkpointRecord
  const id = cfg.learnerId
  const world = createWorld({ seed: seedsOf(id).world, canvas: new OffscreenCanvas(WORLD.width, WORLD.height) })
  const queue = await openQueue(indexedDB)
  const rate = rateMeter()
  let steps = 0
  let cu = 0
  /**
   * The birth, decided at the first run (its widths class is known then) and kept: a restore on another backend runs
   * the same genome (and resumes from the checkpoint, which carries its block anyway).
   */
  let candidates: Promise<Births> | undefined
  let accepted: Candidate | undefined
  /** The newborn arm's frozen file was refused (BadMetaBlockError): later runs go to the candidates. */
  let armRefused = false
  /** Whether this life is a newborn: no checkpoint (handed over or stored) and no log file yet. */
  const isNewborn = async () => m.checkpointRecord === undefined && !(await hasCheckpoint(indexedDB, id)) && (await queue.files(id)).length === 0

  /** The log files' share of the caregiver line (createCarryTally). */
  const carryFiles = (files: readonly LogFile[]): CarryFile[] =>
    files.flatMap(f => (f.schemaVersion === 2 ? [{ seq: f.seq, carries: f.caregiver?.carries ?? 0, safetyNet: f.caregiver?.safetyNet ?? false }] : []))
  /** The caregiver line of the current run (made again by each run, from the files then in the queue). */
  let care = createCarryTally([], 0)
  const postCare = () => post({ type: "carried", ...care.line() })

  const runOnce = async (c: SessionConfig, chosen: Chosen, restore: Restore | null, cand: Candidate) => {
    // Every log file this learner has written comes before its checkpoint, so its curve (and its checkups) are the start
    // of the resumed one.
    const files = await queue.files(id)
    const prior: CurvePoint[] = files.flatMap(f => (f.schemaVersion === 2 ? f.curve : []))
    const priorCheckups: CheckupResult[] = files.flatMap(f => (f.schemaVersion === 2 ? f.checkups ?? [] : []))
    // The coming carries go into the file after the latest one (seq 0 for a newborn).
    // Files the last run saw are kept: the queue may have deleted some since. Carries it saw but not in a file are
    // dropped: a restore takes the life back to its checkpoint, whose own carries come with the next file.
    care = createCarryTally([...care.files(), ...carryFiles(files)], files.reduce((n, f) => Math.max(n, f.seq + 1), 0))
    const tally = care
    // Only a newborn's block can be refused (BadMetaBlockError), and a newborn has no record: a retry loses nothing.
    const adopted = record
    record = undefined
    const run = configOf(c, m, cand)
    gate.run()
    return runSession(run, {
      ...(adopted !== undefined ? { checkpointRecord: adopted } : {}),
      world, idb: indexedDB, now: () => new Date(), recoverable: isRecoverable(chosen), inbox,
      // Each checkup room is a fresh world on a canvas of its own (decision 0016).
      checkups: { canvas: () => new OffscreenCanvas(WORLD.width, WORLD.height) },
      createLearner: async (init): Promise<SessionLearner> => {
        // The session asks for a learner only once it has accepted the life's block (or has none to check).
        accepted = cand
        if (cand.unlabelled !== undefined) post({ type: "unlabelled", reason: cand.unlabelled })
        const l = await learnerOn(chosen, init)
        const birth = "state" in init ? init.state : init
        post({ type: "learner", settings: { ...birth.settings, maxStepsPerSec: run.maxStepsPerSec ?? DEFAULT_MAX_STEPS_PER_SEC, seed: birth.seed }, widths: { ...birth.widths } })
        if (!("state" in init)) {
          post({ type: "curve", points: [], reset: true })
          post({ type: "checkups", results: [], reset: true })
          postCare()
        }
        // Refreshed knowledge goes to the learner if it takes it (the 04a learner does not; its logs still record the sha in force).
        const takesKnowledge = l as { onKnowledge?(m: Manifest): void }
        return {
          step: o => l.step(o), stats: () => l.stats(), save: () => l.save(), dispose: () => l.dispose(),
          frozenRun: o => l.frozenRun(o), readParams: (...a) => l.readParams(...a),
          onKnowledge: k => takesKnowledge.onKnowledge?.(k),
          async sleep() {
            await l.sleep()
            const p = curvePoint(l.stats(), steps)
            if (p) post({ type: "curve", points: [p], reset: false })
          },
        }
      },
      onTick: ({ steps: n, cu: now, obs }) => {
        steps = n
        cu = now
        const perSec = rate(n)
        if (!frames.due(performance.now())) return
        // Copies: the learner may keep its observation, so its buffers are never transferred away.
        const periphery = obs.periphery.slice(), fovea = obs.fovea.slice()
        post({ type: "frame", steps: n, cu: now, stepsPerSec: perSec, periphery, fovea }, [periphery.buffer, fovea.buffer])
      },
      onEvent: e => {
        if (e.type === "resumed") {
          gate.resumed(e.lastLevel)
          steps = e.steps
          cu = e.cu
          post({ type: "curve", points: [...prior, ...e.curveTail], reset: true })
          post({ type: "checkups", results: [...priorCheckups, ...e.checkupTail], reset: true })
          postCare()
          post({ type: "status", message: resumeStatus(e.steps, restore) })
        } else if (e.type === "meta") {
          post({ type: "meta", meta: e.meta, lastLevel: gate.meta(lifeBudget(e.meta).budgetCU) })
        } else if (e.type === "restored") {
          post({ type: "status", message: `its updates kept failing at step ${e.fromSteps}: restored from its checkpoint at step ${e.toSteps} (${e.restores} restore${e.restores === 1 ? "" : "s"} so far)` })
        } else if (e.type === "logged") {
          post({ type: "queued", seq: e.seq, steps: e.steps, cu: e.cu, final: e.final })
          // The session emits this once the file is in the queue: its own count replaces the carries seen for it.
          tally.logged(e.seq)
          void queue.files(id).then(fs => {
            for (const f of carryFiles(fs)) tally.file(f)
            if (care === tally) postCare()
          }, () => {})
        } else if (e.type === "checkupStarted") {
          post({ type: "status", message: checkupStatus(e.level) })
        } else if (e.type === "checkup") {
          post({ type: "checkups", results: [e.result], reset: false })
          post({ type: "status", message: `checkup at ${fmtLevel(e.result.level)} CU done; learning again` })
        } else if (e.type === "carried") {
          tally.carried(e.to)
          postCare()
        }
      },
    })
  }

  /**
   * The first run tries each candidate until the session accepts one: a block it refuses (BadMetaBlockError) is
   * thrown before any learner, checkpoint or file exists, so the next candidate starts clean (said on the console).
   * Later runs (restores) run the accepted one.
   */
  const run: RunLife = async (c, chosen, restore) => {
    if (accepted) return runOnce(c, chosen, restore, accepted)
    if (!candidates) {
      // The draw runs once, before the learner starts (a ts draw conditions the model: a moment, up to a few seconds).
      post({ type: "status", message: "drawing a recipe…" })
      candidates = candidatesOf(m, c.widths)
      const fresh = await isNewborn()
      await candidates
      if (fresh) post({ type: "status", message: "newborn" })
    }
    const births = await candidates
    // A confirmation's newborn arm runs as one frozen newborn, in place of a life (a resumed life is never one).
    if (births.newbornArm && !armRefused && await isNewborn()) {
      try {
        return await runConfirmNewborn(births.newbornArm.birth, births.newbornArm.ctx, chosen)
      } catch (e) {
        if (!(e instanceof BadMetaBlockError)) throw e
        armRefused = true
        console.warn(`blanktab: ${e.message}; trying the ${births.list[0]!.name} birth instead`)
      }
    }
    const list = births.list
    for (const [i, cand] of list.entries()) {
      try {
        return await runOnce(c, chosen, restore, cand)
      } catch (e) {
        if (!(e instanceof BadMetaBlockError) || accepted || i === list.length - 1) throw e
        console.warn(`blanktab: ${e.message}; trying the ${list[i + 1]!.name} birth instead`)
      }
    }
    throw new Error("unreachable: the last candidate has no block to refuse")
  }

  /**
   * A confirmation's newborn arm (spec §4.5; decision 0024 "Confirmation-newborn files"): this tab's learner `id` is one
   * untrained learner of the candidate, seeded by its evalSeed, checked frozen on that seed's own variants, its one file
   * queued for the contributor (gate data: not local-only). About 4 minutes on a GPU. Nothing to resume: a restore after
   * a GPU failure runs it again from the start (its file is queued only at its end).
   */
  const runConfirmNewborn = async (birth: Birth, ctx: DrawContext, chosen: Chosen): Promise<SessionResult> => {
    const meta = birth.meta
    const { evalSeed, confirmation, drawSeed, posteriorSha } = meta
    if (evalSeed === null || confirmation === null || drawSeed === null || posteriorSha === null) throw new BadMetaBlockError(id, "a confirmation newborn's block needs its evalSeed, confirmation and draw")
    post({ type: "confirmNewborn", learnerId: id, evalSeed, done: false })
    post({ type: "meta", meta, lastLevel: 0 })
    post({ type: "status", message: `checking a confirmation newborn (seed ${evalSeed.toLocaleString("en-US")}) on its fresh worlds` })
    checkpoints.frozenStarts()
    const file = await runNewborn({
      learnerId: id, who: { kind: "confirm", evalSeed, confirmationId: confirmation.id, drawSeed, posteriorSha },
      codeSha: cfg.codeSha, knowledgeSha: cfg.knowledgeSha, hostSig: { ...cfg.hostSig, webgpu: chosen.kind === "webgpu" },
      widths: chosen.widths, settings: birth.learnerSettings, genome: { schema: meta.genomeSchema, values: { ...meta.genome }, sha: meta.genomeSha }, space: ctx.space,
    }, {
      createLearner: async init => {
        const l = await learnerOn(chosen, init)
        if (!("state" in init)) post({ type: "learner", settings: { ...init.settings, maxStepsPerSec: birth.maxStepsPerSec, seed: init.seed }, widths: { ...init.widths } })
        return l
      },
      checkupCanvas: () => new OffscreenCanvas(WORLD.width, WORLD.height), idb: indexedDB, now: () => new Date(),
    }).finally(() => checkpoints.frozenEnds())
    post({ type: "checkups", results: file.checkups ?? [], reset: true })
    post({ type: "queued", seq: file.seq, steps: 0, cu: 0, final: true })
    post({ type: "confirmNewborn", learnerId: id, evalSeed, done: true })
    return { steps: 0, cu: 0, seq: file.seq + 1, resumedFrom: null, alreadyFinal: false, halted: false }
  }

  const r = await liveOnBestBackend(cfg, {
    choose: () => chooseBackend({ createGpu: () => createWebGpuBackend({ powerPreference: "high-performance" }), ...(force ? { force } : {}), ...(widths ? { widths } : {}) }),
    run, idb: indexedDB, lastCU: () => cu,
    onBackend: c => post({ type: "backend", kind: c.kind, label: c.label }),
    onStatus: message => post({ type: "status", message }),
  })
  // Halted after a forced save: the life is not over, and the page reloads onto that save (even if it had stopped
  // waiting for the answer; any request it makes from now on is answered true at once by the inbox).
  post(r.halted ? { type: "halted" } : { type: "final", steps: r.steps, cu: r.cu, ...(r.ended ? { ended: r.ended } : {}) })
}

/**
 * Newborn reference mode (spec §2.3): newborns k = from…to, one after another, each an untrained learner of the centre
 * genome on this tab's backend, its one file queued in the outbox (runNewborn). Each is marked local-only first, so
 * the contributor never sends it: tools/eval/collect-newborns.ts collects them from this browser (controller ruling B).
 */
async function newborns(m: StartMessage, range: { from: number; to: number }): Promise<void> {
  const ctx = await contextOf(m.draw)
  if (!ctx) throw new Error("newborn reference mode needs the genome space and the centre (knowledge/genome-space.json, knowledge/genome.json)")
  const chosen = await chooseBackend({ createGpu: () => createWebGpuBackend({ powerPreference: "high-performance" }), ...(m.backend ? { force: m.backend } : {}), ...(m.widths ? { widths: m.widths } : {}) })
  post({ type: "backend", kind: chosen.kind, label: chosen.label })
  const queue = await openQueue(indexedDB)
  for (let k = range.from; k <= range.to; k++) {
    const learnerId = freshLearnerId()
    post({ type: "newborn", k, learnerId, done: false })
    await queue.setContribution({ learnerId, lastPr: 0, closedInARow: 0, localOnly: true })
    const centre = await birthFromDraw(withoutPosterior(ctx), learnerId, widthsClassOf(chosen.widths))
    const file = await runNewborn({
      learnerId, who: { kind: "reference", k }, codeSha: m.cfg.codeSha, knowledgeSha: m.cfg.knowledgeSha, hostSig: { ...m.cfg.hostSig, webgpu: chosen.kind === "webgpu" },
      widths: chosen.widths, settings: centre.learnerSettings, genome: { schema: ctx.space.schemaVersion, values: { ...ctx.centre }, sha: ctx.centreSha }, space: ctx.space,
    }, {
      createLearner: async init => learnerOn(chosen, init), checkupCanvas: () => new OffscreenCanvas(WORLD.width, WORLD.height),
      idb: indexedDB, now: () => new Date(),
    })
    const evalPanel = file.meta?.evalPanel
    post({ type: "newborn", k, learnerId, done: true, ...(typeof evalPanel === "number" ? { evalPanel } : {}) })
  }
  post({ type: "newbornsDone" })
}

let started = false
scope.onmessage = e => {
  const m = e.data
  if (m.type === "visibility") { frames.setHidden(m.hidden); return }
  if (m.type === "knowledge") { inbox.setKnowledge(m.manifest); return }
  if (m.type === "checkpointNow") { checkpoints.request(m); return }
  if (m.type === "shortLives") { shortLives?.switched(m.on, m.consent ?? false); return }
  if (m.type !== "start" || started) return
  started = true
  start(m).catch((err: unknown) => post({
    type: "error", message: err instanceof Error ? err.message : String(err),
    ...(err instanceof LearnerStateVersionError && err.older ? { olderVersion: true } : {}),
  }))
}
