# 0026 — Checkpoint verification progress: the verification-time bound

- **Date:** 2026-10-28
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (agent's call, not yet confirmed), under the user's 2026-10-28 delegation ("go ahead with all plans"). What it implements is decision 0024's planned mitigation; how it is bounded and sized is the agent's.
- **Resolves:** decision 0024's open item "The verification-time bound" (its planned mitigation, which 0024 recorded as not built)

## Context

The gate re-verifies every search and confirmation life against the posterior it was born under (spec §7.4), and records each verdict in `data/meta/verified.json` so that it is computed once (0024, "Verification cache"). A lost cache, or a `VERIFIER` bump, re-verifies every life. 0024 measured that at 0.33 s per ts life on the laptop at the 1,000-row cap (memoised conditioning), so about 11–17 min per build on CI for 1,000 ts lives. `commit-meta.ts` rebuilds up to 3 times when main moves, and `gate.yml` stops a run at 30 minutes, so past roughly 1,000 ts lives a lost cache could stop the gate from ever finishing a build.

0024's planned mitigation, quoted: "when fresh verification would exceed a time budget, the gate checkpoints its verification progress into `verified.json` and leaves `posterior.json` and `calibration.json` unchanged until verification completes, so the outputs change only once it is complete and the cache still never decides. It must be built before about 1,000 ts lives exist." There are no ts lives yet (the signal check has not answered).

## Decision

### The bound is a count of work, not a clock [agent's call, not yet confirmed]

- `tools/gate/meta.ts` exports `VERIFY_BUDGET = 200`. A build verifies fresh at most 200 lives that **may make a Thompson pick**: lives whose birth posterior is known, readable and carries a gpu model (`mayPick`). Only those can reach `birthDraw`'s ts branch, the conditioning and the pick that 0024 measured.
- Every other uncached life (born under a calibrate, flat or paused posterior, or offline) costs milliseconds to verify. It is always verified and not counted. So a lost cache of calibration lives still completes in one build.
- **Which lives are verified is fixed by the input.** Lives are taken in learnerId order. After the 200th counted fresh verification, every further counted, uncached life is left for the next build. The same main gives the same subset (tested).
- **No clock decides anything.** A wall-clock budget would make the subset, and so the commit, depend on the runner's speed. The count bounds the same cost without a clock. It decides only how much is verified in one build, never a verdict or an output.
- **Why 200.** From 0024's figures, a counted life costs at most 0.5 s on the laptop: 0.33 s for the candidate set and the pick, plus at most one conditioning per distinct posterior (about 0.17 s, from 500 s unmemoised against 331 s memoised for 1,000 lives). On CI (3×) that is 1.5 s. So 200 lives take at most 5 min per build, and three builds (commit-meta's rebuilds) at most 15 min. The fit adds under a minute per build, which is under 20 of the 30 minutes in total. Not re-measured for this decision.
- **Convergence.** A lost cache of 1,000 ts lives completes in 5 builds (about 5 hours on the hourly cron).

### A build with lives left unverified is a checkpoint [agent; 0024's planned mitigation]

- `buildOrCheckpoint(input, budget)` is the gate's build; `meta-cli.ts` calls it with `VERIFY_BUDGET`. When lives are left unverified it returns a `MetaCheckpoint`: `verified.json` with every verdict the build has (the cached ones and this build's fresh ones), and nothing else.
- In that case `meta-cli.ts` writes `verified.json` only. Everything else stays byte-identical to main's: `posterior.json`, `calibration.json`, `diagnostics.json`, `posterior-index.jsonl`, the confirmation files, `promotions.jsonl` and `knowledge/genome.json`. Nothing after verification runs: no look, no fit and no confirmation step. So no output is ever computed from part of the lives.
- That is broader than 0024's wording, which named only `posterior.json` and `calibration.json` [agent's call, not yet confirmed]. A confirmation decided, or a diagnostics count taken, on part of the lives would be an output of an incomplete verification too.
- `buildMeta(input)` is the same build with no bound. Checkpoints converge to it (below). It is what the tests compare against.
- **The cache still never decides.** A checkpointed verdict is written by the same rule, under the same `VERIFIER` and keyed on the same learnerId and posteriorSha, as any cached verdict (0024). So the build that verifies the last life writes exactly what one unbounded build writes on the same main.
- Tests check this byte for byte for every file over a chain of checkpoints, including a flagged forged life. They also check that every checkpointed verdict equals the fresh one. A checkpoint of another `VERIFIER` is no cache: a bump starts again from nothing, and main's outputs (built under the old verifier) stay until the new verifier has verified every life.
- `meta-cli.ts` prints a `::notice::` with the fresh, verified and pending counts, so a checkpoint shows in the gate's log.

### commit-meta commits a checkpoint [agent's call, not yet confirmed]

- `META_PATH_RE` already allowed `verified.json`. What changed is the commit message, which no longer reads `posterior.json` when `verified.json` is the only staged file: `meta: verified.json only, <n> lives verified; outputs unchanged`. That message works on a main with no `posterior.json` yet, and it does not name a posterior the commit leaves unchanged.
- Each checkpoint is pushed, so progress accumulates on main and the next run resumes from it.
- **A moved main.** When main moves under a push, commit-meta's reset discards this build's unpushed checkpoint, and the rebuild verifies the same next 200 again from main's `verified.json`. Work is repeated, but nothing is lost or counted twice: a verdict is a memo, and the bound is per build, so three builds cost at most 3 × 200.

## Alternatives considered

- **A wall-clock budget.** Its behaviour depends on runner speed and load, so the same main could give different commits. The count bounds the same work deterministically.
- **Counting every fresh verification.** That is simpler, but it would slow a lost cache of cheap calibration lives to 200 per build for no saving in time.
- **Carrying unpushed verdicts across a rebuild on a moved main.** That would save the repeated work, but the rebuild would then run on bytes not built from that main. The new main may also bump `VERIFIER`. Rejected: the bound already keeps three builds within the limit.
- **Writing partial diagnostics, or showing progress on the page.** Diagnostics are an output, so they wait. The `::notice::` line is enough for now.

## Consequences

- When a checkpoint is needed, the posterior, calibration looks, confirmations and promotions do not move until verification completes. Tabs keep drawing from main's posterior. A confirmation whose deadline passes meanwhile is decided by the build that completes, with that build's `builtAt`.
- If more than 200 counted lives merge between two hourly runs, those runs checkpoint and the outputs wait. If that rate were sustained, the outputs would never move. That is far above today's volume, and the `::notice::` would show it.
- **Trust boundary, unchanged from 0024.** A hand-edited `verified.json` with a forged clean verdict, under the right posteriorSha and `VERIFIER`, would be believed. Only main's writers can produce one (the gate builds it, `commit-meta` commits it, and PR content never reaches it, per 0019). The checkpoint does not widen this.
- Code: `tools/gate/meta.ts` (`VERIFY_BUDGET`, `mayPick`, `buildOrCheckpoint`, `MetaCheckpoint`), `tools/gate/meta-cli.ts`, `tools/gate/commit-meta.ts`. Tests: `tools/gate/test/meta.test.ts` ("the verification-time bound" and the meta-cli checkpoint run) and `tools/gate/test/commit-meta.test.ts`.
