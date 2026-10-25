// @blanktab/evaluators: swappable, versioned evaluators (spec §2; decision 0020). Every evaluator is pure and
// deterministic. REGISTRY keys are "id@version"; old versions stay forever, so any merged life can be rescored under
// any of them. Which ones count is pinned in config/evaluation.json, which the gate never writes.
import { confirmFisher } from "./confirm-fisher.ts"
import { newbornBaseline } from "./newborn-baseline.ts"
import { searchComposite } from "./search-composite.ts"
import { signalCheck } from "./signal-check.ts"
import type { CohortContext, Evaluator } from "./types.ts"

export type {
  CohortContext, EvalBudget, EvalOutput, Evaluator, LifeInput, LifeOptions, LifeRole, LifeStanding, MilestoneKey, NewbornProvenance, NewbornReference, ReferenceParser, References,
} from "./types.ts"
export { GPU_WIDTHS_CLASS } from "./types.ts"
export { parseNewbornReference, PROVENANCE_TEXT_RE, REFERENCE_PARSERS } from "./reference.ts"
export { rankAverage, searchComposite } from "./search-composite.ts"
export { newbornBaseline } from "./newborn-baseline.ts"
export { signalCheck } from "./signal-check.ts"
export { confirmFisher } from "./confirm-fisher.ts"
export { ALPHA_LOOK, ICC_MIN, SIGNAL_LOOKS } from "./stats.ts"

export const REGISTRY: Readonly<Record<string, Evaluator>> = Object.freeze({
  "search-composite@1": searchComposite,
  "newborn-baseline@1": newbornBaseline,
  "signal-check@1": signalCheck,
  "confirm-fisher@1": confirmFisher,
})

/** The evaluator registered as `idAtVersion` (e.g. "search-composite@1"); throws on an unknown one. */
export function getEvaluator(idAtVersion: string): Evaluator {
  const e = Object.hasOwn(REGISTRY, idAtVersion) ? REGISTRY[idAtVersion] : undefined
  if (!e) throw new Error(`unknown evaluator ${JSON.stringify(idAtVersion)}`)
  return e
}

/**
 * What the cohort evaluators run with: the pinned search evaluator, REGISTRY[pins.search] (config/evaluation.json's
 * `search`). signal-check and confirm-fisher score lives only through it, so swapping the search evaluator is a pin edit.
 */
export function cohortContext(pins: { search: string }): CohortContext {
  return { search: getEvaluator(pins.search) }
}
