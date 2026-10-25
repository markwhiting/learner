// A newborn's genome (Plan 05a): the knowledge's meta-learner files parsed into a draw context, the seeded birth draw
// (@blanktab/meta birthDraw, which the gate recomputes) decoded into what each component runs, and the session's meta
// block source. Pure apart from hashing, so it is testable; the worker calls it once the backend (widths class) is known.
import { DEFAULT_SETTINGS, type LearnerSettings } from "@blanktab/learner"
import { CAREGIVER_DEFAULTS, type CaregiverSettings } from "@blanktab/harness/caregiver"
import { CONFIG_BUDGET, type EvalBudget, type MetaBlock } from "@blanktab/harness/log"
import type { Knowledge } from "@blanktab/harness/knowledge"
import type { SessionMeta } from "@blanktab/harness/session"
import {
  birthDraw, decodeGenome, genomeSha, legacyGenome, modelPosterior, parseCentre, parseGenomeSpace, parsePosterior, sha256Hex,
  type Genome, type GenomeSpace, type GpPosterior, type Posterior,
} from "@blanktab/meta"

/** What a birth draw needs from the knowledge: the space, the centre and its sha, and the posterior with the sha of its bytes. */
export interface DrawContext {
  space: GenomeSpace; centre: Genome; centreSha: string
  /** null together: absent, unreadable, or not a posterior this code reads (an unknown schemaVersion): the draw is centre. */
  posterior: Posterior | null; posteriorText: string | null; posteriorSha: string | null
}

/**
 * The draw context of `k`'s meta-learner files, or null when the genome space or the centre (knowledge/genome.json) is
 * missing or malformed, or does not decode: such a tab runs DEFAULT_SETTINGS and logs no meta block (Review Focus 5).
 * A posterior that does not parse gives posterior, posteriorText and posteriorSha null (the draw is then centre).
 */
export async function drawContextOf(k: Pick<Knowledge, "meta">): Promise<DrawContext | null> {
  if (!k.meta) return null
  let space: GenomeSpace, centre: Genome
  try {
    space = parseGenomeSpace(k.meta.genomeSpace)
    centre = parseCentre(k.meta.genome, space)
    decodeGenome(centre, space)
  } catch {
    return null
  }
  const centreSha = await genomeSha(space.schemaVersion, centre)
  const text = k.meta.posteriorText
  const posterior = text === null ? null : parsePosterior(text)
  if (posterior === null || text === null) return { space, centre, centreSha, posterior: null, posteriorText: null, posteriorSha: null }
  return { space, centre, centreSha, posterior, posteriorText: text, posteriorSha: await sha256Hex(new TextEncoder().encode(text)) }
}

/**
 * What a newborn of a drawn genome runs, and the block its logs carry. `reason`: why a present posterior gave centre.
 * `kind`: "life" (a life, run by the session), or "confirmNewborn" (a confirmation's newborn arm: one frozen file,
 * runNewborn, in place of a life). `seed`: the learner seed the draw sets (role confirm: its evalSeed); absent, the
 * learnerId's (seedsOf).
 */
export interface Birth {
  kind: "life" | "confirmNewborn"
  meta: MetaBlock; learnerSettings: LearnerSettings; caregiver: CaregiverSettings; maxStepsPerSec: number; sleepInterval: number
  seed?: number
  reason?: string
}

/** Each key of `decoded` replaces the default's; a key the component does not have is an error (it would not run). */
function overlay<T extends object>(defaults: T, decoded: Record<string, number>, what: string): T {
  const out: Record<string, number> = { ...(defaults as Record<string, number>) }
  for (const [k, v] of Object.entries(decoded)) {
    if (!Object.hasOwn(defaults, k)) throw new RangeError(`the genome's ${what} gene ${k} is not a ${what} setting`)
    out[k] = v
  }
  return out as T
}

/**
 * The meta block of a draw: its genome and sha at the space's schema, born under `budget` (decision 0020; none:
 * legacy). The draw's panel, evalSeed and confirmation pass through as drawn (role confirm: no panel, its evalSeed and
 * confirmation { id, arm }; every other role: a panel and both null).
 */
type BlockDraw = Pick<MetaBlock, "role" | "method" | "drawSeed" | "posteriorSha" | "evalPanel" | "evalSeed" | "confirmation" | "genome">
async function block(space: GenomeSpace, d: BlockDraw, budget: EvalBudget | null): Promise<MetaBlock> {
  return {
    v: 1, role: d.role, method: d.method, genomeSchema: space.schemaVersion, genome: { ...d.genome }, genomeSha: await genomeSha(space.schemaVersion, d.genome),
    drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, evalSeed: d.evalSeed,
    confirmation: d.confirmation === null ? null : { id: d.confirmation.id, arm: d.confirmation.arm }, inheritance: { filters: "none" },
    ...(budget ? { budget: { budgetCU: budget.budgetCU, levels: [...budget.levels] } } : {}),
  }
}

/**
 * The gpu model's GP posterior, conditioned once per posteriorSha (the latest one only: a tab draws under one
 * posterior at a time). birthDraw calls it only when it reaches the ts branch, which has checked the model first.
 */
let conditioned: { posteriorSha: string; post: GpPosterior | null } | null = null
function conditionedOf(ctx: DrawContext): (() => GpPosterior | null) | undefined {
  const { posterior, posteriorSha } = ctx
  const model = posterior?.models.gpu
  if (model === undefined || posteriorSha === null) return undefined
  return () => {
    if (conditioned?.posteriorSha !== posteriorSha) conditioned = { posteriorSha, post: modelPosterior(model) }
    return conditioned.post
  }
}

/**
 * A newborn's birth: birthDraw for `learnerId` in `ctx` at `widthsClass` (the gate's recomputation, spec §7.4), decoded
 * (decodeGenome): the learner and caregiver genes over DEFAULT_SETTINGS and CAREGIVER_DEFAULTS, the harness genes as the
 * speed cap and the sleep interval. The block carries `budget` (default the config's). A ts draw's GP conditioning is
 * memoised per posteriorSha. A confirm draw sets the learner seed to its evalSeed; its newborn arm is kind
 * "confirmNewborn" (a frozen file, not a life). Throws if the genome does not decode to settings the components have
 * (the worker then falls back to a centre draw).
 */
export async function birthFromDraw(ctx: DrawContext, learnerId: string, widthsClass: "gpu" | "cpu", budget: EvalBudget = CONFIG_BUDGET): Promise<Birth> {
  const memo = conditionedOf(ctx)
  const d = birthDraw({
    learnerId, space: ctx.space, centre: ctx.centre, centreSha: ctx.centreSha, widthsClass, posterior: ctx.posterior, posteriorSha: ctx.posteriorSha,
    ...(memo ? { conditioned: memo } : {}),
  })
  const dec = decodeGenome(d.genome, ctx.space)
  return {
    kind: d.role === "confirm" && d.confirmation?.arm === "newborn" ? "confirmNewborn" : "life",
    meta: await block(ctx.space, d, budget),
    learnerSettings: overlay(DEFAULT_SETTINGS, dec.learner, "learner"),
    caregiver: overlay(CAREGIVER_DEFAULTS, dec.caregiver, "caregiver"),
    maxStepsPerSec: dec.harness.maxStepsPerSec, sleepInterval: dec.harness.sleepInterval,
    ...(d.role === "confirm" && d.evalSeed !== null ? { seed: d.evalSeed } : {}),
    ...(d.reason !== undefined ? { reason: d.reason } : {}),
  }
}

/** The same context without its posterior: what a birth falls back to (a centre draw). */
export const withoutPosterior = (ctx: DrawContext): DrawContext => ({ ...ctx, posterior: null, posteriorText: null, posteriorSha: null })

/** Test knobs that win over a genome's harness genes (the worker's StartMessage.overrides). */
export interface HarnessKnobs { maxStepsPerSec?: number; sleepInterval?: number }

/**
 * SessionMeta.resumedHarness: a resumed life's speed cap and sleep interval from its checkpoint block's genome (decoded
 * in `ctx`'s space; a block of another genome schema gives none), leaving out the genes a legacy block did not record
 * (`missing`: the config's stand), with `knobs` over them.
 */
export function resumedHarnessOf(ctx: DrawContext, knobs: HarnessKnobs = {}): NonNullable<SessionMeta["resumedHarness"]> {
  return block => {
    const out: HarnessKnobs = {}
    if (block.genomeSchema === ctx.space.schemaVersion) {
      const h = decodeGenome(block.genome, ctx.space).harness
      const missing = new Set(block.missing ?? [])
      if (!missing.has("maxStepsPerSec")) out.maxStepsPerSec = h.maxStepsPerSec
      if (!missing.has("sleepInterval")) out.sleepInterval = h.sleepInterval
    }
    return { ...out, ...knobs }
  }
}

/**
 * The session's meta source (SessionMeta) for a life born with `birth` in `ctx`. A resumed pre-05a life gets a legacy
 * block (legacyGenome of its logged settings: role and method legacy, no panel, its unrecorded genes in `missing`). A
 * resumed life with a block runs its genome's speed cap and sleep interval (resumedHarnessOf, `knobs` over them).
 */
export function sessionMetaOf(ctx: DrawContext, birth: MetaBlock, budget: EvalBudget = CONFIG_BUDGET, knobs: HarnessKnobs = {}): SessionMeta {
  return {
    space: ctx.space, birth, budget, resumedHarness: resumedHarnessOf(ctx, knobs),
    async legacy(flat) {
      const { genome, missing } = legacyGenome(flat, ctx.space)
      // No budget: a legacy life runs under the v1 budget (lifeBudget), and the parser refuses one on it.
      const b = await block(ctx.space, { role: "legacy", method: "legacy", drawSeed: null, posteriorSha: null, evalPanel: null, evalSeed: null, confirmation: null, genome }, null)
      return { ...b, ...(missing.length > 0 ? { missing } : {}) }
    },
  }
}
