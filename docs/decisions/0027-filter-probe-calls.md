# 0027 — The filter-convergence probe's calls

- **Date:** 2026-11-16
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (every call below is an agent's call, not yet confirmed). The plan it implements is `docs/plans/2026-10-28-05c-filter-probe.md`; the measurement itself is spec §5.1 (`docs/specs/2026-10-22-meta-learner-v1-design.md`).
- **Resolves:** the gaps plan 05c found in spec §5.1 ("Spec conflicts and gaps found while planning"), for the probe only

## Context

Decision 0018's user part binds: learned material that generalises moves up the hierarchy. That the first candidate is the learner's first-layer filters, gated by a convergence probe, is one of 0018's agent's calls, not yet confirmed. Spec §5.1 says what the probe measures (linear CKA, the median matched correlation after Hungarian matching, weight motion; against init-pair and between-genome controls; a bar over ≥ 10 lives of one genome) but leaves several choices open. Plan 05c fills them; this file records each fill as an agent's call so the user can confirm or change it.

All of these were fixed, tested and committed **before** the probe first ran on real lives (invariant 7). Changing any of them after that run needs a new recorded decision, and both results are kept.

## Decision

### The probe frames [agent's call, not yet confirmed]

- **256 frames:** 16 world seeds 20,000–20,015 (spec §5.1) × the observation after each of the first 16 actions. The actions are drawn per seed with `uniformAction(createRng(hashSeed(seed, "filter-probe/actions")))`, the held-out set's rule.
- **Start room `ROOM_KINDS[seed mod 4]`,** so each of the four rooms starts four seeds. With the default start all 256 frames would come from one room, since 16 random steps rarely leave it.
- **Rendered in Node** with `@napi-rs/canvas`, not the browser's canvas, so text pixels may differ from what tabs see. Every frame's sha256 is pinned; a canvas or world change that alters a frame makes the probe refuse to run, and regenerating the manifest is then a recorded decision.
- **The manifest lives in `tools/probe/filter-probe-set.json`,** not `knowledge/` (tabs fetch that) or `data/` (the gate writes that). Nothing in `packages/` or `app/` imports `tools/probe/` (a test), and a test checks that no committed life under `data/runs` or `data/newborns` has a training, held-out or checkup world with a probe seed.
- At generation, 236 of the 256 periphery frames and 124 of the 256 fovea frames are distinct (a step that does not move the cursor or change the room repeats a frame). Repeats are kept: the set is what the rule gives.

## Alternatives considered

Each section above names what it rejected.

## Consequences

The probe's result is reported favourable or not (`docs/research/`). Nothing here changes what a learner sees or does.
