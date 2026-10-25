// A newborn life: an untrained learner, scored once on its held-out set and checked frozen at NEWBORN_LEVELS. One log
// file, cu 0. Two kinds (decision 0024 "Confirmation-newborn files"):
// - a reference newborn (spec §2.3; plan 05a Task 9): the centre genome, on the shared panel a search life would meet;
// - a confirmation newborn (spec §4.5; plan 05b Task 7): the newborn arm of a confirmation, of the candidate's genome,
//   seeded by its evalSeed and checked on that seed's own variants (confirmSeed), as its other arms' lives are.
// What a newborn reads is harness instrumentation: the learner never sees its genome, panel, arm or scores.
import { clampSettings, type LearnerSettings, type Widths } from "@blanktab/learner"
import type { CanvasLike } from "@blanktab/world"
import { checkGenome, decodeGenome, genomeSha, settingsMismatch, type GenomeSpace } from "@blanktab/meta"
import { runCheckup, variantSeedOf } from "./checkup"
import {
  CONFIG_BUDGET, MAX_EVAL_PANEL, MILESTONE_SUITE_VERSION, NEWBORN_LEVELS, isCurveLoss, parseLogFile, parseMeta,
  type CheckupResult, type LogFileV2, type MetaBlock,
} from "./log"
import { openQueue } from "./queue"
import { BadMetaBlockError, type SessionDeps } from "./session"

/** Newborn k's learner seed is NEWBORN_SEED_BASE + k (spec §2.3: seeds 1,000–1,063). */
export const NEWBORN_SEED_BASE = 1000
/** The newborns of one reference: k = 0 … NEWBORN_COUNT − 1, 4 per panel × 16 panels (spec §2.3). */
export const NEWBORN_COUNT = 64

/**
 * Which newborn:
 * - `reference`: newborn k of the reference, 0 ≤ k < NEWBORN_COUNT; its seed is NEWBORN_SEED_BASE + k and its panel
 *   1 + (k mod 16);
 * - `confirm`: the newborn-arm life of confirmation `confirmationId` drawn with seed `evalSeed` (the draw's drawSeed and
 *   posteriorSha, as birthDraw gave them); its learner seed is evalSeed, it has no panel, and evalSeed's arm must be
 *   "newborn" (the parser's rule).
 */
export type NewbornWho =
  | { kind: "reference"; k: number }
  | { kind: "confirm"; evalSeed: number; confirmationId: number; drawSeed: number; posteriorSha: string }

export interface NewbornConfig {
  learnerId: string
  who: NewbornWho
  codeSha: string
  knowledgeSha: string
  hostSig: LogFileV2["hostSig"]
  widths: Widths
  /** The genome's learner settings (decoded by the page); they must be what `genome` decodes to. */
  settings: LearnerSettings
  /** The genome it runs (a reference: the centre; a confirmation newborn: the candidate): its schema, values and genomeSha. */
  genome: { schema: number; values: Record<string, number | string>; sha: string }
  /**
   * The genome space `genome` is of (main's knowledge/genome-space.json, as the page loads it). Not in the plan's
   * interface: the file's logged settings must decode from the genome (the gate's settingsMismatch), and the caregiver
   * genes and speed cap are known only through the space (agent's call, decision 0022).
   */
  space: GenomeSpace
}

export interface NewbornDeps {
  /** As for a life: an untrained learner with its held-out set (the page builds it with buildHeldOut). */
  createLearner: SessionDeps["createLearner"]
  /** A fresh canvas for each checkup world. */
  checkupCanvas(): CanvasLike
  idb: IDBFactory
  now(): Date
  /** Awaited after every frozen step (the page's pacer); without it recordRun yields now and then. */
  yieldNow?: () => Promise<void>
}

/** The learner seed of newborn `who` (RangeError for a reference k outside 0 … NEWBORN_COUNT − 1). */
function seedOf(who: NewbornWho): number {
  if (who.kind === "confirm") return who.evalSeed
  const { k } = who
  if (!(Number.isInteger(k) && k >= 0 && k < NEWBORN_COUNT)) throw new RangeError(`newborn k must be an integer in [0, ${NEWBORN_COUNT}), got ${k}`)
  return NEWBORN_SEED_BASE + k
}

/**
 * The meta block and logged settings newborn `cfg` would write, checked: the block is one the parser accepts in a
 * frozen file (for a confirmation newborn: its seed's arm is "newborn"), the genome lies in the space, its sha
 * recomputes, and it decodes to `cfg.settings` (clamped). BadMetaBlockError otherwise, before any learner exists.
 */
async function birthOf(cfg: NewbornConfig, seed: number): Promise<{ meta: MetaBlock; settings: Record<string, number> }> {
  const bad = (why: string): never => { throw new BadMetaBlockError(cfg.learnerId, why) }
  const { who } = cfg
  const { schema, values, sha } = cfg.genome
  if (schema !== cfg.space.schemaVersion) bad(`a newborn's genome schema ${schema} is not the space's ${cfg.space.schemaVersion}`)
  const outside = checkGenome(cfg.space, values)
  if (outside !== null) bad(`a newborn's genome is outside the genome space: ${outside}`)
  if (await genomeSha(schema, values) !== sha) bad("a newborn's genome sha does not match its genome")
  let d: ReturnType<typeof decodeGenome>
  try {
    d = decodeGenome(values, cfg.space)
  } catch (e) {
    return bad(e instanceof Error ? e.message : String(e))
  }
  const settings: Record<string, number> = {
    ...(clampSettings(cfg.settings) as unknown as Record<string, number>),
    maxStepsPerSec: d.harness.maxStepsPerSec, seed, ...d.caregiver,
  }
  const off = settingsMismatch(cfg.space, values, settings)
  if (off !== null) bad(`a newborn's genome is not what it runs: ${off}`)
  const common = {
    v: 1 as const, genomeSchema: schema, genome: { ...values }, genomeSha: sha, inheritance: { filters: "none" as const },
    budget: { budgetCU: CONFIG_BUDGET.budgetCU, levels: [...CONFIG_BUDGET.levels] },
  }
  const meta: MetaBlock = who.kind === "reference"
    ? { ...common, role: "newborn", method: "newborn", drawSeed: null, posteriorSha: null, evalPanel: 1 + (who.k % MAX_EVAL_PANEL), evalSeed: null, confirmation: null }
    : {
      ...common, role: "confirm", method: "confirm", drawSeed: who.drawSeed, posteriorSha: who.posteriorSha, evalPanel: null,
      evalSeed: who.evalSeed, confirmation: { id: who.confirmationId, arm: "newborn" },
    }
  const parsed = parseMeta(meta, { cu: 0, stepsTotal: 0, final: true })
  if (typeof parsed === "string") bad(parsed)
  return { meta, settings }
}

/**
 * Runs newborn `cfg.who` and queues its one log file (spec §2.3; "how a newborn runs" was left to the builder):
 * - its learner: seed NEWBORN_SEED_BASE + k (a reference) or evalSeed (a confirmation newborn), untrained (createLearner
 *   with the config's widths and settings);
 * - its held-out score: one sleep() on an empty buffer, which makes no update and only scores the held-out set,
 *   recorded as meta.heldOut (left out, with a warning, if the score is not finite: the file stays on the record);
 * - its checkups: runCheckup at each of NEWBORN_LEVELS, on variantSeedOf(its block): a reference's panel 1 + (k mod 16)
 *   (panelSeed), a confirmation newborn's evalSeed (confirmSeed);
 * - its file: seq 0, final, cu 0, stepsTotal 0, born under the config's budget; role and method "newborn" (a reference),
 *   or role and method "confirm" with confirmation { id, arm: "newborn" }, evalSeed and evalPanel null (a confirmation
 *   newborn). The held-out score and checkups are billed to evalCU. Queued with queue.put (no checkpoint: there is
 *   nothing to resume). A confirmation newborn's file is gate data, contributed like any other.
 * The file is checked with parseLogFile before it is queued, so the outbox never holds one the gate would refuse to parse.
 */
export async function runNewborn(cfg: NewbornConfig, deps: NewbornDeps): Promise<LogFileV2> {
  const seed = seedOf(cfg.who)
  const birth = await birthOf(cfg, seed)
  const variantSeed = variantSeedOf(birth.meta)!
  const learner = await deps.createLearner({ seed, widths: { ...cfg.widths }, settings: clampSettings(cfg.settings) })
  let file: LogFileV2
  try {
    if (!learner.frozenRun) throw new Error(`learner ${cfg.learnerId}: a newborn needs a learner with frozenRun (checkups)`)
    await learner.sleep()
    const st = learner.stats()
    if (st.cu !== 0 || st.steps !== 0) throw new Error(`learner ${cfg.learnerId}: a newborn's sleep must make no update (cu ${st.cu}, steps ${st.steps})`)
    const heldOut = isCurveLoss(st.heldOutLoss) && isCurveLoss(st.heldOutBaseline) ? { loss: st.heldOutLoss, baseline: st.heldOutBaseline } : undefined
    if (!heldOut) console.warn(`learner ${cfg.learnerId}: the newborn's held-out score is not finite; its file has none`)
    const checkups: CheckupResult[] = []
    const frozen = learner.frozenRun
    for (const level of NEWBORN_LEVELS) {
      const out = await runCheckup({
        learnerId: cfg.learnerId, level, steps: 0, learner: { frozenRun: o => frozen(o), stats: () => learner.stats() },
        canvas: () => deps.checkupCanvas(), variantSeed,
        ...(deps.yieldNow ? { yieldNow: deps.yieldNow } : {}),
      })
      // Nothing can stop it (no shouldStop), so a checkup always finishes.
      if (out.status !== "done") throw new Error(`learner ${cfg.learnerId}: newborn checkup at ${level} was abandoned`)
      checkups.push(out.result)
    }
    const end = learner.stats()
    if (end.cu !== 0) throw new Error(`learner ${cfg.learnerId}: a newborn's checkups must not change its cu (${end.cu})`)
    file = {
      schemaVersion: 2, learnerId: cfg.learnerId, seq: 0, codeSha: cfg.codeSha, knowledgeSha: cfg.knowledgeSha,
      writtenAt: deps.now().toISOString(), stepsTotal: 0, settings: birth.settings, hostSig: { ...cfg.hostSig },
      sessionsSeenAtBirth: 0, final: true, cu: 0, condition: "A", widths: { ...cfg.widths }, curve: [],
      skippedUpdates: end.skippedUpdates, restores: 0, lostCU: 0, evalCU: end.evalCU,
      checkups, milestoneSuiteVersion: MILESTONE_SUITE_VERSION,
      meta: { ...birth.meta, ...(heldOut ? { heldOut } : {}) },
    }
  } finally {
    learner.dispose()
  }
  const p = parseLogFile(JSON.parse(JSON.stringify(file)))
  if (!p.ok) throw new Error(`learner ${cfg.learnerId}: the newborn's file would not parse: ${p.error}`)
  await (await openQueue(deps.idb)).put(file)
  return file
}
