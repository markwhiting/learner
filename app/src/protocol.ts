// Messages between the page (main thread) and the learner's worker. The token never crosses this boundary.
import type { Manifest } from "@blanktab/harness/knowledge"
import type { CheckupResult, CurvePoint, MetaBlock } from "@blanktab/harness/log"
import type { Widths } from "@blanktab/learner"
import type { BackendKind } from "./backend"
import type { LifeConfig } from "./life"

export interface StartMessage {
  type: "start"
  /**
   * The life's config. Its learnerSettings, caregiver, maxStepsPerSec and sleepInterval are the fallback for a tab
   * without a genome space or centre (DEFAULT_SETTINGS, the caregiver prior, the default cap): with `draw`, the worker's
   * birth draw replaces them (but not the cap of a resumed life: `ownCap`).
   */
  cfg: LifeConfig
  /**
   * The meta-learner's files as the page loaded them (Knowledge.meta, raw): the worker draws the birth genome from them
   * at the first run, once the backend (and so the widths class) is known. Absent: no genome files, so no meta block.
   */
  draw?: { spaceJson: unknown; genomeJson: unknown; posteriorText: string | null }
  /** cfg.maxStepsPerSec is the life's own (from its latest log file, a resumed life): the draw does not replace it. */
  ownCap?: boolean
  /** The short-lives switch as the page has it now (then ShortLivesMessage on each change; decisions 0019, 0022). */
  shortLives: boolean
  /**
   * This learner's end at its budget was confirmed in this browser (stored per learnerId): without it, a life resumed
   * past its budget checkup is never ended by the switch (shortlives.ts).
   */
  shortLivesConsent?: boolean
  /** Newborn reference mode (?newborns=a-b): runs newborns k = from…to in sequence instead of a life. */
  newborns?: { from: number; to: number }
  /** Test knobs (never set by the production build): skip WebGPU; a newborn's widths. */
  backend?: "cpu"
  widths?: Widths
  /**
   * Test knobs applied after the draw (never set by the production build): the speed cap, the steps between sleeps,
   * and a newborn's evaluation budget (its budget checkup, where a short life ends). A life whose genome a knob
   * contradicts is not born with a meta block (the session refuses it); these lives are dev builds, never rows.
   */
  overrides?: { maxStepsPerSec?: number; sleepInterval?: number; budgetCU?: number }
  /** document.hidden when the life starts (then VisibilityMessage on each change): no frames are posted while hidden. */
  hidden?: boolean
  /** The learner's checkpoint record, already read by the startup sweep that adopted it (its buffers transferred). */
  checkpointRecord?: unknown
}
export interface VisibilityMessage { type: "visibility"; hidden: boolean }
/** Refreshed knowledge (never offline), as its manifest: its knowledgeSha is in force for later log files; handed to the learner. */
export interface KnowledgeMessage { type: "knowledge"; manifest: Manifest }
/**
 * A forced checkpoint (before a code reload): answered by a `checkpointed` message with the same id. With `halt` the
 * learner stops right after the save, so the reload loses no step.
 */
export interface CheckpointNowMessage { type: "checkpointNow"; id: number; halt: boolean }
/** The short-lives switch changed (read by the session at clean sleeps only). `consent`: the page confirmed this learner's end. */
export interface ShortLivesMessage { type: "shortLives"; on: boolean; consent?: boolean }
export type ToWorker = StartMessage | VisibilityMessage | KnowledgeMessage | CheckpointNowMessage | ShortLivesMessage

export type FromWorker =
  | { type: "backend"; kind: BackendKind; label: string }
  /** The learner as it runs: its settings (clamped, with the speed cap and seed) and widths, newborn or from a checkpoint. */
  | { type: "learner"; settings: Record<string, number>; widths: Widths }
  | { type: "frame"; steps: number; cu: number; stepsPerSec: number; periphery: Uint8Array; fovea: Uint8Array }
  /** `reset`: the whole curve so far (birth or resume); otherwise points to add. */
  | { type: "curve"; points: CurvePoint[]; reset: boolean }
  | { type: "status"; message: string }
  /** Milestone checkups (decision 0016). `reset`: every one of the life so far (birth or resume); otherwise ones to add. */
  | { type: "checkups"; results: CheckupResult[]; reset: boolean }
  /**
   * The caregiver: `carries` in the life so far (from its log files and this run's carries since the latest), `to` the
   * room of the last one (null if not known, e.g. just after a reload), and whether it is in safety-net mode (as of the
   * latest log file).
   */
  | { type: "carried"; to: string | null; carries: number; safetyNet: boolean }
  | { type: "queued"; seq: number; steps: number; cu: number; final: boolean }
  /** The life's end. `ended: "budget"`: the short-lives switch ended it at its budget checkup. */
  | { type: "final"; steps: number; cu: number; ended?: "budget" }
  /**
   * The life's meta block (its genome and how it was drawn), as the session logs it: once per run. `lastLevel`: the
   * highest checkup level the life had when this run started (the session's, from its checkpoint; 0 for a newborn),
   * known even when the files that held those checkups are gone from the outbox.
   */
  | { type: "meta"; meta: MetaBlock; lastLevel: number }
  /** This life logs no meta block, and why (no genome files, or the session refused every block: a test knob). */
  | { type: "unlabelled"; reason: string }
  /**
   * Newborn reference mode: newborn `k` (learner `learnerId`) has started, or its file is in the outbox (`done`, with
   * the panel its file records).
   */
  | { type: "newborn"; k: number; learnerId: string; done: boolean; evalPanel?: number }
  /**
   * A confirmation's newborn arm (plan 05b, Task 11): this tab's newborn `learnerId` drew it, so instead of a life it
   * runs one frozen newborn of the candidate, seeded `evalSeed`, and queues its one file (gate data, contributed as any
   * other): started, or its file is in the outbox (`done`). A `final` message follows, and the page starts a newborn.
   */
  | { type: "confirmNewborn"; learnerId: string; evalSeed: number; done: boolean }
  /** Newborn reference mode: every newborn asked for is done. */
  | { type: "newbornsDone" }
  /** The life stopped at a forced save asked with `halt`: its checkpoint is current; the page reloads onto it. */
  | { type: "halted" }
  /** The answer to CheckpointNowMessage `id`: saved, or not (skipped updates outstanding, or the life ended): ask again later. */
  | { type: "checkpointed"; id: number; saved: boolean }
  /** `olderVersion`: its checkpoint is from an older learner state version (LearnerStateVersionError.older). */
  | { type: "error"; message: string; olderVersion?: boolean }
