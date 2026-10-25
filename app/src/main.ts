// The page's main thread: knowledge, identity, the GitHub contributor and the token live here.
// The learner itself runs in ./worker.ts; this thread only starts it and shows what it reports.
import {
  DB_BLOCKED_MESSAGE, DEFAULT_MAX_STEPS_PER_SEC, LOCAL_ONLY_MESSAGE, PERMISSION_MESSAGE, REJECTED_MESSAGE, adoptLearnerId, checkToken,
  clampCaregiverSettings, clearToken, createContributor, getToken, hasCheckpoint, lifeBudget, loadKnowledge, loadManifest, newLearnerId, openQueue,
  setToken, storedLearnerId, sweepCheckpoints, waitLock, webLock, MAX_UNSENT,
  type Knowledge, type LogFile, type Queue, type SweepResult,
} from "@blanktab/harness"
import { DEFAULT_SETTINGS } from "@blanktab/learner"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"
import { NEWBORN_SEED_BASE } from "@blanktab/harness/newborn"
import { drawContextOf } from "./birth"
import { seedsOf, type LifeConfig } from "./life"
import { caregiverText } from "./milestones"
import { newbornText, readNewborns } from "./newborns"
import type { FromWorker, StartMessage, ToWorker } from "./protocol"
import { readTestParams, type TestParams } from "./params"
import { flushPolicy } from "./schedule"
import { readConsent, writeConsent } from "./shortlives"
import { buffersOf } from "./transfer"
import { createUi, type Ui } from "./ui"
import { WORKER_LOAD_MESSAGE, onWorkerLoadFailure, startUpdates } from "./updates"

const NEWBORN_DELAY_MS = 2_000
const PRESENCE_CHANNEL = "blanktab:presence"
const PRESENCE_WAIT_MS = 250
/** Shown (as a notice, not an error) when a resumed learner's checkpoint is from an older learner state version. */
const OLDER_VERSION_MESSAGE = "This learner was born on an older version; a newborn takes its place."
/** A forced checkpoint not answered in this long counts as not saved (the reload is retried later). */
const CHECKPOINT_WAIT_MS = 60_000

// Test-only overrides. In the production build MODE is the literal "production" and this is dropped.
const test: TestParams = import.meta.env.MODE !== "production" ? readTestParams(location.search) : {}
const lifetimeCU = test.lifetimeCU // undefined: the learner lives as long as its tab
/** Test knobs the worker applies after the birth draw (the genome sets these otherwise). */
const overrides: NonNullable<StartMessage["overrides"]> = {
  ...(test.stepsPerSecond !== undefined ? { maxStepsPerSec: test.stepsPerSecond } : {}),
  ...(test.sleepInterval !== undefined ? { sleepInterval: test.sleepInterval } : {}),
  ...(test.budgetCU !== undefined ? { budgetCU: test.budgetCU } : {}),
}

/** The short-lives switch (decisions 0019, 0022): this browser's, off unless stored as "1". */
const SHORT_LIVES_KEY = "blanktab:shortLives"
const shortLivesOn = () => localStorage.getItem(SHORT_LIVES_KEY) === "1"
/**
 * Asked before switching short lives on for a life that has already had its budget checkup (ruling A, decision 0022),
 * and once when such a life is resumed with the switch on: only a yes ends it (at its next clean sleep), and the answer
 * is stored for that learnerId (shortlives.ts); the stored switch alone never ends it.
 */
const pastBudgetQuestion = (budgetCU: number) =>
  `This learner has had its ${budgetCU.toLocaleString("en-US")} CU checkup, so it will end at its next sleep and a newborn will start. Go on?`

const fetchFn: typeof fetch = (input, init) => fetch(input, init)

/** Every blanktab tab answers pings, including this one, so the count includes self. */
function answerPresence(): void {
  const ch = new BroadcastChannel(PRESENCE_CHANNEL)
  ch.onmessage = (e: MessageEvent<{ type?: string; from?: string }>) => {
    if (e.data?.type === "ping") ch.postMessage({ type: "pong", to: e.data.from })
  }
}
async function countTabs(): Promise<number> {
  const me = crypto.randomUUID()
  const ch = new BroadcastChannel(PRESENCE_CHANNEL)
  let n = 0
  ch.onmessage = (e: MessageEvent<{ type?: string; to?: string }>) => { if (e.data?.type === "pong" && e.data.to === me) n++ }
  ch.postMessage({ type: "ping", from: me })
  await new Promise(r => setTimeout(r, PRESENCE_WAIT_MS))
  ch.close()
  return Math.max(1, n)
}
/** webgpu is filled in by the worker: whether the learner actually runs on WebGPU. */
async function hostSig(): Promise<LifeConfig["hostSig"]> {
  const nav = navigator as Navigator & { deviceMemory?: number }
  return {
    cores: Number.isFinite(nav.hardwareConcurrency) ? nav.hardwareConcurrency : 1,
    deviceMemoryGb: typeof nav.deviceMemory === "number" && Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : null,
    tabsOnMachine: await countTabs(),
  }
}

/**
 * Birth facts are fixed for a learner's whole life. A resumed learner's seed, widths, learner settings, knowledgeSha
 * and sessionsSeenAtBirth come from its checkpoint (the session prefers them to these); its speed cap from its latest
 * log (`ownCap`). A newborn's genome is drawn in the worker (birth.ts): these settings (DEFAULT_SETTINGS, the
 * knowledge's caregiver prior, clamped, and the default cap) are only what a tab without genome files runs.
 */
function birth(latest: LogFile | undefined, knowledge: Knowledge): Pick<LifeConfig, "knowledgeSha" | "sessionsSeenAtBirth" | "learnerSettings" | "maxStepsPerSec" | "caregiver"> & { ownCap: boolean } {
  const fallback = { learnerSettings: { ...DEFAULT_SETTINGS }, caregiver: clampCaregiverSettings(knowledge.caregiverPrior), maxStepsPerSec: DEFAULT_MAX_STEPS_PER_SEC }
  if (!latest) return { ...fallback, ownCap: false, knowledgeSha: knowledge.manifest.knowledgeSha, sessionsSeenAtBirth: knowledge.summary.sessions.length }
  const cap = latest.settings.maxStepsPerSec
  const own = typeof cap === "number" && cap > 0
  return { ...fallback, ownCap: own, ...(own ? { maxStepsPerSec: cap } : {}), knowledgeSha: latest.knowledgeSha, sessionsSeenAtBirth: latest.sessionsSeenAtBirth }
}

/** The page's meta-learner files as the worker takes them (raw: it parses and draws). */
const drawOf = (k: Knowledge): StartMessage["draw"] =>
  k.meta ? { spaceJson: k.meta.genomeSpace, genomeJson: k.meta.genome, posteriorText: k.meta.posteriorText } : undefined

/**
 * One tab per learner. Chrome's "Duplicate tab" copies sessionStorage, so two tabs can start with the same
 * learnerId; the Web Lock (blanktab:learner:<id>, held until the learner's life ends) tells them apart. Resolves to a
 * release function, or null if another tab holds it. Without the Locks API (very old browsers) every tab simply proceeds.
 */
function lockLearner(id: string): Promise<(() => void) | null> {
  if (!("locks" in navigator)) return Promise.resolve(() => {})
  return webLock(navigator.locks)(id).then(h => h && h.release)
}

/**
 * This tab's own learner's lock (a reload, or a tab the browser restored): waited for, up to OWN_LOCK_WAIT_MS, since
 * another tab's sweep may be looking at it for a moment (and, seeing this request waiting, lets it go). Null if
 * another tab lives it (a duplicated tab).
 */
function lockOwn(id: string): Promise<(() => void) | null> {
  if (!("locks" in navigator)) return Promise.resolve(() => {})
  return waitLock(navigator.locks, id).then(h => h && h.release)
}

/** The life running in this tab, as the updates see it. */
interface Life {
  /** Its learnerId. */
  id: string
  /** Hands refreshed knowledge to the learner (its manifest only: the sha is what its logs record). */
  knowledge(k: Knowledge): void
  /**
   * A forced checkpoint: true once saved; false if not (skips outstanding, no answer in CHECKPOINT_WAIT_MS, the life
   * ended). With `halt` the learner stops right after the save.
   */
  checkpoint(halt: boolean): Promise<boolean>
  /** Stops the learner where it is (before a reload onto the checkpoint just saved). */
  halt(): void
  /** The short-lives switch changed; `confirmed`: the past-budget question was asked for this learner and answered yes. */
  shortLives(on: boolean, confirmed?: boolean): void
  /**
   * Its budget (CU) if switching short lives on would end it at its next sleep: it has a meta block and its budget
   * checkup is done (the session's end condition, session.ts; the level it resumed at is the session's own, so a
   * pruned outbox does not hide it); else null.
   */
  pastBudget(): number | null
}

/**
 * `knowledge`: the latest the page has (refreshed while it lives); newborns are born from it. `life`: the running one,
 * if any. `halted`: the running learner stopped at a forced save (startUpdates' halted: the page reloads onto it).
 */
interface Ctx { ui: Ui; queue: Queue; knowledge: Knowledge; queued: (final: boolean) => void; life: Life | null; halted: () => void }

const newborn = (ctx: Ctx) => live(newLearnerId(sessionStorage), ctx)

/**
 * `held`: this learner's lock, already taken (by the startup sweep). `record`: its checkpoint record, already read by
 * the sweep that adopted it (handed to the worker, so it is not read twice).
 */
async function live(id: string, ctx: Ctx, held?: () => void, record?: unknown): Promise<void> {
  const { ui, queue, queued } = ctx
  const knowledge = ctx.knowledge
  const release = held ?? (await lockLearner(id))
  if (release === null) return newborn(ctx) // another tab is living this learner's life
  const latest = await queue.latest(id)
  if (latest?.final) { release(); return newborn(ctx) } // this tab's learner already finished its life
  // Every log file is written with its checkpoint, so logs without one (e.g. the old flailer's) cannot be resumed.
  if (latest && !(await hasCheckpoint(indexedDB, id))) {
    release()
    ui.notice(`Learner ${id} has logs but no checkpoint, so its life cannot be resumed. A newborn takes its place.`)
    return newborn(ctx)
  }
  const { ownCap, ...b } = birth(latest, knowledge)
  const cfg: LifeConfig = {
    learnerId: id, codeSha: __CODE_SHA__, ...b, hostSig: await hostSig(), seed: seedsOf(id).learner,
    ...(lifetimeCU !== undefined ? { lifetimeCU } : {}),
  }
  ui.learner({ id, sessionsBefore: cfg.sessionsSeenAtBirth, ...(lifetimeCU !== undefined ? { lifetimeCU } : {}) })

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
  // A hidden page draws nothing: the worker stops posting frames until it is shown again.
  const visibility = () => worker.postMessage({ type: "visibility", hidden: document.hidden } satisfies ToWorker)
  document.addEventListener("visibilitychange", visibility)
  let over = false
  /** What the switch's question needs: the highest checkup level done, and the life's budget once its block is known. */
  let checkedTo = 0
  let budgetCU: number | null = null
  /** The highest checkup level the life had when this run started (the session's own; meta message). */
  let resumedAt = 0
  /** Whether the past-budget question was asked at this resume. */
  let askedAtResume = false
  /** The knowledge this life is drawn from (the worker draws from `draw` below): its recipe line reads it, not a later refresh. */
  const drawnFrom = drawContextOf(knowledge)
  let nextCheckpoint = 0
  const waiting = new Map<number, (saved: boolean) => void>()
  const life: Life = {
    id,
    knowledge: k => worker.postMessage({ type: "knowledge", manifest: k.manifest } satisfies ToWorker),
    checkpoint: halt => new Promise<boolean>(resolve => {
      if (over) return resolve(false)
      const id = nextCheckpoint++
      const timer = setTimeout(() => { waiting.delete(id); resolve(false) }, CHECKPOINT_WAIT_MS)
      waiting.set(id, saved => { clearTimeout(timer); waiting.delete(id); resolve(saved) })
      worker.postMessage({ type: "checkpointNow", id, halt } satisfies ToWorker)
    }),
    halt: () => worker.terminate(),
    shortLives: (on, confirmed = false) => {
      if (confirmed) writeConsent(localStorage, id, "yes") // the decision survives a code-update reload
      worker.postMessage({ type: "shortLives", on, ...(confirmed ? { consent: true } : {}) } satisfies ToWorker)
    },
    pastBudget: () => (budgetCU !== null && checkedTo >= budgetCU ? budgetCU : null),
  }
  ctx.life = life
  // Its logs were rejected too often (decision 0014): it lives on here, but no longer contributes.
  void queue.contribution(id).then(c => { if (c.localOnly && ctx.life === life) ui.notice(LOCAL_ONLY_MESSAGE) }, () => {})
  /**
   * The learner always runs: whether its life completed or failed, a newborn takes its place (unless `then` is
   * "stop": no newborn could run either, e.g. the worker's code itself cannot load).
   */
  const end = (show: () => void, then: "newborn" | "stop" = "newborn") => {
    if (over) return
    over = true
    if (ctx.life === life) ctx.life = null
    for (const answer of [...waiting.values()]) answer(false)
    document.removeEventListener("visibilitychange", visibility)
    worker.terminate()
    release()
    show()
    if (then === "stop") return
    setTimeout(() => void newborn(ctx).catch((e: unknown) => ui.error(e instanceof Error ? e.message : String(e))), NEWBORN_DELAY_MS)
  }
  /** Whether the worker has said anything yet: an error before that is its code failing to load (or to start). */
  let heard = false
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    heard = true
    const m = e.data
    if (m.type === "frame") ui.frame(m)
    else if (m.type === "backend") ui.backend(m.label)
    else if (m.type === "learner") ui.settings(m.settings)
    else if (m.type === "curve") ui.curve(m.points, m.reset)
    else if (m.type === "status") ui.status(m.message)
    else if (m.type === "checkups") {
      const before = checkedTo
      checkedTo = Math.max(m.reset ? resumedAt : checkedTo, resumedAt, ...m.results.map(r => r.level))
      // Its budget checkup done in this tab with the switch on: the session ends it at this sleep, as agreed. Should a
      // halt win that sleep, the resumed run must end too, so its end counts as confirmed.
      if (!m.reset && budgetCU !== null && before < budgetCU && checkedTo >= budgetCU && shortLivesOn()) writeConsent(localStorage, id, "yes")
      ui.checkups(m.results, m.reset)
    }
    else if (m.type === "carried") ui.caregiver(caregiverText(m))
    else if (m.type === "meta") {
      budgetCU = lifeBudget(m.meta).budgetCU
      resumedAt = m.lastLevel
      checkedTo = Math.max(checkedTo, m.lastLevel)
      // Resumed past its budget with the switch on, and never asked: the switch alone does not end it (ruling A); ask once.
      const past = life.pastBudget()
      if (past !== null && m.lastLevel >= past && shortLivesOn() && readConsent(localStorage, id) === null && !askedAtResume) {
        askedAtResume = true
        setTimeout(() => {
          if (over || ctx.life !== life) return
          const yes = confirm(pastBudgetQuestion(past))
          if (yes) life.shortLives(shortLivesOn(), true)
          else writeConsent(localStorage, id, "no")
        }, 0)
      }
      const meta = m.meta
      void drawnFrom.then(dc => { if (ctx.life === life) ui.recipe({ meta, ctx: dc }) })
    }
    else if (m.type === "unlabelled") ui.recipe({ unlabelled: m.reason })
    // A confirmation's newborn arm: one frozen file instead of a life; its `final` follows, then a newborn starts.
    else if (m.type === "confirmNewborn" && m.done) ui.status("the confirmation newborn's file is in the outbox; a newborn starts next")
    else if (m.type === "queued") {
      ui.progress(m.steps, m.cu)
      // Without a token (or for a local-only learner) nothing is ever sent: keep each learner's unsent files bounded.
      void queue.capUnsent(MAX_UNSENT).catch(() => {}).then(() => queued(m.final))
    }
    // Stopped at a forced save, perhaps after the page had stopped waiting for its answer: reload onto it now.
    else if (m.type === "halted") ctx.halted()
    else if (m.type === "checkpointed") waiting.get(m.id)?.(m.saved)
    else if (m.type === "error") end(() => (m.olderVersion ? ui.notice(OLDER_VERSION_MESSAGE) : ui.error(`${m.message}. A newborn takes its place.`)))
    else if (m.type === "final") {
      writeConsent(localStorage, id, null) // the life is over: its answer is no longer needed
      end(() => ui.lifeComplete(id, m.ended))
    }
  }
  worker.onerror = e => {
    const failed = () => end(() => ui.error(`${e.message || "worker failed"}. A newborn takes its place.`))
    if (heard || over) return failed()
    // Its code may be gone (a deploy replaced the hashed assets): reload onto the new code rather than retry forever.
    void onWorkerLoadFailure({
      codeSha: __CODE_SHA__, storage: sessionStorage, loadManifest: () => loadManifest(import.meta.env.BASE_URL, fetchFn),
      reload: () => location.reload(),
    }).then(r => {
      if (r === "same-code") failed()
      else end(() => (r === "gave-up" ? ui.notice(WORKER_LOAD_MESSAGE) : undefined), "stop")
    })
  }
  const draw = drawOf(knowledge)
  const start: ToWorker = {
    type: "start", cfg, hidden: document.hidden, ...(record !== undefined ? { checkpointRecord: record } : {}),
    ...(draw ? { draw } : {}), ownCap, shortLives: shortLivesOn(), shortLivesConsent: readConsent(localStorage, id) === "yes",
    ...(test.backend ? { backend: test.backend } : {}), ...(test.widths ? { widths: test.widths } : {}),
    ...(Object.keys(overrides).length > 0 ? { overrides } : {}),
  }
  worker.postMessage(start, buffersOf(record)) // moved, not copied
  // The knowledge in force from now on, for a resumed learner too (it was saved with what it had then).
  if (!knowledge.offline) life.knowledge(knowledge)
}

async function main(ui: Ui): Promise<void> {
  // Newborn reference mode: no life of this tab's (its learner, if it has one, is left for another tab to adopt).
  const newbornRange = readNewborns(location.search)
  // Before anything else (anything awaited): this tab's own learner, if it has one, is asked for at once, so no other
  // tab's startup sweep takes its checkpoint for an orphan while this page loads.
  const own = newbornRange ? null : storedLearnerId(sessionStorage)
  const ownLock = own === null ? Promise.resolve(null) : lockOwn(own)
  answerPresence()
  const sig = hostSig() // for the sweep's final logs: counting tabs takes a moment, so it starts now
  const showTokenState = () => {
    const stored = getToken(localStorage) !== null
    ui.tokenPanel(!stored)
    ui.changeToken(stored)
  }
  showTokenState()
  const [knowledge, queue] = await Promise.all([loadKnowledge(import.meta.env.BASE_URL, fetchFn), openQueue(indexedDB)])
  ui.knowledge(knowledge)

  let current: Ctx | null = null
  const contributor = createContributor({
    queue, ls: localStorage, fetchFn, now: () => new Date(),
    onLocalOnly: id => { if (current?.life?.id === id) ui.notice(LOCAL_ONLY_MESSAGE) },
    onStatus: s => {
      ui.contribution(s)
      // Anything the token itself is the cause of: offer to paste another one.
      if (s.state === "no-token" || s.message === REJECTED_MESSAGE || s.message === PERMISSION_MESSAGE) ui.tokenPanel(true)
    },
  })
  const flush = () => void contributor.flush().catch((e: unknown) => {
    ui.contribution({ state: "error", message: `Couldn't read the saved logs (${e instanceof Error ? e.message : String(e)}); will try again.`, pending: 0 })
  })
  const policy = flushPolicy(flush, undefined, () => void contributor.report())
  ui.onTokenSubmit(async token => {
    const r = await checkToken(token, fetchFn) // check first: a token is stored only once GitHub accepts it
    if (!r.ok) return r.reason
    setToken(localStorage, token)
    showTokenState()
    policy.tokenSaved()
    return null
  })
  ui.onChangeToken(() => {
    clearToken(localStorage)
    showTokenState()
    flush() // reports "add a token" with the number of logs waiting
  })
  flush() // logs left by earlier tabs or lives

  const ctx: Ctx = { ui, queue, knowledge, queued: policy.queued, life: null, halted: () => {} }
  current = ctx
  if (newbornRange) return runNewborns(ctx, newbornRange)
  ctx.halted = watchForUpdates(ctx).halted
  // The short-lives switch: read by each life at its start, and handed to the running one on each change.
  ui.shortLives(shortLivesOn())
  ui.onShortLives(on => {
    const past = on ? ctx.life?.pastBudget() ?? null : null
    if (past !== null && !confirm(pastBudgetQuestion(past))) return false
    if (on) localStorage.setItem(SHORT_LIVES_KEY, "1")
    else localStorage.removeItem(SHORT_LIVES_KEY)
    ctx.life?.shortLives(on, past !== null)
    return true
  })
  const release = await ownLock
  if (own !== null && release !== null) {
    await sweep(false, sig, own)
    return live(own, ctx, release)
  }
  // A new tab (or a duplicated one, whose learner lives in the original): adopt a closed tab's learner, if any.
  // The sweep never touches the learner in this tab's sessionStorage.
  const adopted = own === null ? await sweep(true, sig) : (await sweep(false, sig, own), null)
  if (adopted) {
    adoptLearnerId(sessionStorage, adopted.learnerId)
    ui.notice(`Adopted learner ${adopted.learnerId}: its tab was closed, so this tab lives the rest of its life.`)
    return live(adopted.learnerId, ctx, adopted.release, adopted.record)
  }
  return newborn(ctx)
}

/**
 * Newborn reference mode (?newborns=a-b; spec §2.3): the worker runs newborns k = a…b of the centre genome, one after
 * another, and queues each one's file in the outbox, marked local-only, for tools/eval/collect-newborns.ts (controller
 * ruling B: collected from this browser, not sent as PRs). No life runs in this tab meanwhile, and none after.
 */
async function runNewborns(ctx: Ctx, range: { from: number; to: number }): Promise<void> {
  const { ui, knowledge } = ctx
  const draw = drawOf(knowledge)
  if (!draw) return ui.error("newborn reference mode needs the genome files (knowledge/genome-space.json, knowledge/genome.json), and they could not be loaded")
  const cfg: LifeConfig = {
    learnerId: "lrn_0000000000000000", codeSha: __CODE_SHA__, knowledgeSha: knowledge.manifest.knowledgeSha, sessionsSeenAtBirth: 0,
    hostSig: await hostSig(), seed: 0, learnerSettings: { ...DEFAULT_SETTINGS },
  }
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
  ui.newbornMode()
  ui.notice(newbornText(range.from, range))
  let backend: string | null = null // shown again for each newborn (ui.learner resets the line)
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data
    if (m.type === "backend") { backend = m.label; ui.backend(m.label) }
    else if (m.type === "newborn" && !m.done) {
      ui.learner({ id: m.learnerId, sessionsBefore: 0 })
      if (backend !== null) ui.backend(backend)
      ui.recipe({ newborn: "a newborn reference" })
      ui.status(`checking newborn ${m.k} (seed ${NEWBORN_SEED_BASE + m.k}) on its panel`)
      ui.notice(newbornText(m.k, range))
    } else if (m.type === "newborn") {
      // Its panel as its file records it.
      if (m.evalPanel !== undefined) ui.recipe({ newborn: `a newborn reference (panel ${m.evalPanel})` })
      ui.status(`newborn ${m.k}'s file is in this browser's outbox`)
    }
    else if (m.type === "newbornsDone") {
      worker.terminate()
      ui.notice(`Newborn references ${range.from}–${range.to} are done. Their files stay in this browser, for tools/eval/collect-newborns.ts.`)
    } else if (m.type === "error") { worker.terminate(); ui.error(m.message) }
  }
  worker.onerror = e => { worker.terminate(); ui.error(e.message || "the newborn worker failed") }
  worker.postMessage({ type: "start", cfg, hidden: document.hidden, draw, shortLives: false, newborns: range, ...(test.backend ? { backend: test.backend } : {}), ...(test.widths ? { widths: test.widths } : {}) } satisfies ToWorker)
}

/**
 * Decision 0014: refreshed knowledge goes to the running learner (and to newborns); new code is reloaded onto if it can
 * resume this learner's checkpoint (the learner is saved first), and otherwise the learner lives on here and the page
 * says so. Nothing polls while the page is hidden.
 */
function watchForUpdates(ctx: Ctx): { halted(): void } {
  const visibility = {
    hidden: () => document.hidden,
    onChange: (fn: () => void) => {
      document.addEventListener("visibilitychange", fn)
      return () => document.removeEventListener("visibilitychange", fn)
    },
  }
  const poll = test.pollMs !== undefined ? { everyMs: test.pollMs, jitterMs: 0 } : undefined
  return startUpdates({
    env: {
      timers: { setTimeout: (fn, ms) => setTimeout(fn, ms), clearTimeout: h => clearTimeout(h as ReturnType<typeof setTimeout>) },
      visibility, random: Math.random, now: () => performance.now(),
    },
    initial: ctx.knowledge, codeSha: __CODE_SHA__, learnerStateVersion: LEARNER_STATE_VERSION, storage: sessionStorage,
    loadKnowledge: () => loadKnowledge(import.meta.env.BASE_URL, fetchFn),
    loadManifest: () => loadManifest(import.meta.env.BASE_URL, fetchFn),
    onKnowledge: k => {
      ctx.knowledge = k
      ctx.ui.knowledge(k)
      ctx.life?.knowledge(k)
    },
    // Halted at the save, so the reload loses no step. No learner running: nothing to save.
    checkpoint: () => ctx.life?.checkpoint(true) ?? Promise.resolve(true),
    reload: () => {
      ctx.life?.halt()
      location.reload()
    },
    notice: m => ctx.ui.update(m),
    ...(poll ? { knowledgePoll: poll, manifestPoll: poll, retryMs: poll.everyMs } : {}),
    ...(test.reloadJitterMs !== undefined ? { reloadJitterMs: test.reloadJitterMs } : {}),
  })
}

/**
 * Orphaned checkpoints (spec §10): adopts the newest one this tab may if `adopt`, and deletes those nobody will
 * resume, so they do not fill the disk (sweepCheckpoints), each with a final log file. Needs Web Locks to tell orphans from live learners; without
 * them, and on any error, it does nothing.
 */
async function sweep(adopt: boolean, sig: Promise<LifeConfig["hostSig"]>, exclude?: string): Promise<SweepResult["adopted"]> {
  if (!("locks" in navigator)) return null
  try {
    // A deleted orphan's final log (decision 0017) names the host it lived on (its latest file's); this tab's only for
    // one that never logged, and no learner runs on WebGPU in writing it.
    const final = { hostSig: { ...(await sig), webgpu: false }, codeSha: __CODE_SHA__, now: () => new Date() }
    const r = await sweepCheckpoints({ idb: indexedDB, tryLock: webLock(navigator.locks), now: () => new Date(), adopt, final, ...(exclude ? { exclude: [exclude] } : {}) })
    return r.adopted
  } catch (e) {
    console.warn("blanktab: the orphaned-checkpoint sweep failed", e)
    return null
  }
}

const ui = createUi()
void main(ui).catch((e: unknown) => {
  const message = e instanceof Error ? e.message : String(e)
  // An older tab without a versionchange handler holds the database open, so this page cannot upgrade it.
  if (message === DB_BLOCKED_MESSAGE) ui.notice(`${DB_BLOCKED_MESSAGE}: an older learner tab is holding this browser's saved logs open. Close it, then reload this page.`)
  else ui.error(message)
})
