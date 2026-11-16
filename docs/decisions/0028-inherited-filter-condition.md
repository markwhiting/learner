# 0028 — The inherited-filter condition (spec §5.2)

- **Date:** 2026-11-16
- **Status:** proposed
- **Decided by:** agent proposal awaiting the user's decision. Nothing in it is built or switched on.
- **Resolves:** spec §5.2 ("specified, not built; gated on the probe and a later decision"); plan 05c Task 8

## Context

- **Decision 0018.** The user decided that learned material that generalises moves up the hierarchy. That the *first* artifact to move up is first-layer filters, gated by a convergence probe and always run beside fresh and matched-control conditions, is one of 0018's agent's calls, **not yet confirmed** (question 1 below).
- **The probe's result** (`docs/research/2026-11-16-filter-convergence.md`; calls in decision 0027). On 99 eligible fresh search lives (41 of the centre genome), at 2,048 CU:
  - across genomes: periphery **not converged** (ckaDelta −0.389, matched 0.000, p 1.0); fovea **not converged** (ckaDelta −0.029, matched 0.257, p 0.897);
  - the centre genome (n = 41): periphery **not converged** (ckaDelta −0.347, matched 0.000, p 0.9998); fovea **not converged** (ckaDelta −0.051, matched 0.295, p 0.934);
  - learned pairs are *less* alike than random-init pairs; same-genome lives are not more alike than different-genome lives (periphery p 0.140, fovea p 0.918); about half of a life's periphery units are dead on the probe frames by 2,048 CU.
- **Invariant 1.** A life born with inherited filters is a different condition from a fresh life, and is labelled as one.

## Decision (proposed)

### The gate on building it [agent's call, not yet confirmed]

The condition is built only if the pooled verdict is "converged" for at least the periphery encoder, and the one-genome (centre) verdict is converged too, or the user accepts building without it with a recorded reason.

**Neither holds today, so the proposal is "not now".** The spec's next candidates come first: a **distilled frame set**, then a **shared replay or curriculum file** [KUH K7, K10, K28]. Re-running the probe unchanged on later lives (for example after a learner change that stops periphery units dying) can reopen this; such a result is added to the research note with a new date and, if it changes the outcome, a new decision supersedes this one.

### What would be built if the gate opened [agent's call, from spec §5.2]

- **Arms.** **F (fresh):** the default, at least 40 % of births [guess]. **I (inherited):** loads `knowledge/filters-gpu-v1.json` at birth; the filters stay trainable. **C (matched control):** random filters with each filter's norm and bias matched to the artifact's. GPU widths only; CPU lives are always F.
- **Building the artifact.** Generation 0 aligns each eligible life's 2,048 CU filters to a reference life (the lowest learnerId of the centre) by `hungarianMax` on activation correlation over the probe set, then averages them. Later generations take a Reptile step toward the mean end state of the I arm. The artifact records the learnerIds, the probe manifest sha and the population CU it took.
- **Labels, and the code that would change** (none changed by this proposal):
  - `meta.inheritance.filters` ∈ {`none`, `<sha>`, `control:<sha>`}. Today `packages/harness/src/log.ts` accepts only `"none"`, and `search-composite@1` and `confirm-fisher@1` count only `"none"`, so I and C lives would never be search rows or confirmation lives unless a new evaluator version says otherwise.
  - The artifact joins `knowledgeSha` (`packages/harness/src/knowledge.ts`); the page's birth (`app/src/birth.ts`, `app/src/worker.ts`) would pick the arm and load it into `createModel`'s initial values for `FIRST_LAYER_PARAMS` only.
  - The GP treats the condition as fixed context, never as a gene; confirmations never pool conditions.
  - The learner's checkpoint format is unchanged: the filters set only initial values.
- **Readouts:** CU to the M1 bar; the held-out ratio at 512 and 1,024 CU on unseen variants; no worse than F at a long budget (the warm-start check, [KUH] K16); and the slope after a novel room (the plasticity check, K17). The population CU spent building the artifact is reported beside any gain (invariant 5).

## Alternatives considered

- **Build the I and C arms now anyway.** Rejected: nothing converged, so an averaged artifact would be an average of dissimilar (and largely dead) filters; C would test little.
- **Whole-network checkpoints** (Lamarckian inheritance): last, per [KUH].
- **A distilled frame set, or a shared replay or curriculum file:** no shared initial weights; the spec's next candidates when filters do not converge. Recommended next.
- **Doing nothing** beyond recording the result.

## Questions for the user (each with the agent's suggestion)

1. Confirm first-layer filters as the first artifact to move up (0018's agent's call), or name another? *Suggestion: given the result, move to a distilled frame set first.*
2. Build only on a converged pooled verdict, or also require the one-genome one? *Suggestion: require both.*
3. The F share (≥ 40 %) and the I:C split (1:1)? *Suggestion: as stated, if ever built.*
4. Periphery and fovea together, or only an encoder that converged? *Suggestion: only a converged encoder.*
5. Is a per-filter norm match enough for C, or should C also match the artifact's spatial frequency spectrum? *Suggestion: norm match first; spectrum match if I beats C.*
6. (New, from the result.) Should the dying periphery units be looked at as a learner question before any filter sharing is reconsidered? *Suggestion: yes, under its own plan.*

## Consequences

Nothing changes in `packages/`, `app/`, `knowledge/`, `config/` or `data/`. Every life stays `inheritance.filters: "none"`. `docs/open-questions.md` is updated when the user decides.
