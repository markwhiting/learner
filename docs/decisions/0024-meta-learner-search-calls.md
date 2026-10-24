# 0024 — Plan 05b's calls: the meta-learner search

- **Date:** 2026-10-24
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (every call here is an agent's call, not yet confirmed). "Inert until signal" is the user's condition (plan 05b); how it is made structural is the agent's.
- **Resolves:** the choices plan 05b (`docs/plans/2026-10-24-05b-meta-learner-search.md`) makes for the user, including its spec gaps

## Context

Plan 05b builds the meta-learner's search: the GP fit in the gate, Thompson-sampling and confirmation draws in the tab, and automatic promotion (spec `docs/specs/2026-10-22-meta-learner-v1-design.md` §4.1, §4.3–§4.5, §7). It is built and deployed while the signal check has not answered, so it must stay switched off until that check says "signal". Several choices along the way belong to the user but were not asked. This file records each as an agent's call, so it can be confirmed or reversed. Later 05b tasks add their sections here.

## Decision

### Posterior schema 2 [agent]

- **`data/meta/posterior.json` moves to `schemaVersion` 2.** `packages/meta/src/posterior.ts` exports `POSTERIOR_SCHEMA = 2`. A v2 file fills what 05a left empty:
  - `models.gpu`: the GP's genes (the cube's axes), its rows (learnerId and unit-cube x), its targets y, its hyperparameters and the fit's record.
  - `incumbent`: the posterior-mean argmax, its genome sha, its mean and P(beats centre).
  - `confirmation`: the confirmation being served, with its seed base (10,000 + 1,000·id), both genomes, and each arm's target, count so far and remaining need.
  - `page.rowsSinceDeclaration`: the eligible rows since the last confirmation was declared (spec §4.4's "≥ 100 rows" condition).
- **A v2 file is strict in every part.** Exact keys at every level. At most 1,000 rows, with learnerIds strictly increasing. x in [0, 1]. Noise ≥ 1e-3. Four fit starts. Arm needs never above their targets. A candidate that is not the centre. Any violation makes the whole file unreadable (`parsePosterior` → null), and the tab then draws `centre`.
- **05b code still reads v1.** A v1 file is checked exactly as 05a checks it. It reads as `models: {}`, `incumbent: null`, `confirmation: null` and `rowsSinceDeclaration: 0`. 05a wrote nothing else, so a 05b tab draws from the last v1 file exactly what a 05a tab drew. `packages/meta/test/fixtures/draws.json` is unchanged and still passes.
- **Old tabs draw centre from a v2 file until they reload (05a ruling, Task 4).** 05a's parser checks `schemaVersion !== 1 → null`, so a 05a tab treats every v2 file as unreadable and draws `centre`, which is always a valid draw. Once it reloads onto new code (0014), it reads v2.
- **Any other `schemaVersion`** (0, 3, a string) is unreadable.

### The mode guard [agent; implements the user's "inert until signal"]

"Inert until signal" does not rest on a flag. It is structural, at four points (plan 05b, Global Constraints):

1. **The gate computes the search only in `search`.** `buildMeta` computes `models`, `incumbent` and `confirmation`, writes confirmation or promotion files, and touches `knowledge/genome.json`, only when `mode === "search"`.
2. **Only the signal check can set `search`.** `mode` is `modeOf(calibration)`. `calibration.outcome` becomes `"signal"` only in `nextLook`, from the pinned signal evaluator's `score === 1`. No other code path sets it, and no flag, environment variable or option can. Tests and the simulation reach `search` only through data (planted calibration lives), never through a code switch.
3. **A tab refuses search state outside `search`.** `parsePosterior` returns null for a v2 posterior that carries a model, an incumbent or a confirmation while its mode is `calibrate`, `paused` or `flat`. The tab then draws `centre`. An incumbent also needs a model. So a hand-edited or mis-built posterior cannot switch a tab on. (Task 1; tested in `packages/meta/test/posterior.test.ts`.)
4. **The draw reaches the new methods only in `search`.** `birthDraw` reaches `ts` and `confirm` only for mode `search` on `gpu`.

### The GP model [agent]

Built in `packages/meta/src/gp.ts` and `packages/meta/src/linalg.ts` (Task 2); the fit that uses it is Task 3.

- **Noise has a smooth floor: σ_n² = 1e-3 + e^θ_n.** The spec says "a floor of 1e-3" and puts its prior on ln σ_n². A hard clamp would put a kink in the objective, which L-BFGS handles badly. So θ_n = ln(σ_n² − 1e-3) is the fitted parameter, and the spec's prior N(ln 0.5, 1) is placed on θ_n instead of ln σ_n². Above about 0.01 the two hardly differ.
- **The prior box.** The objective is undefined (null) when any θ is more than 6 prior sd from the prior mode. This keeps the fit away from regions where e^θ overflows or K is numerically singular, and gives every start the same bounded box.
- **Jitter is part of the model.** K is factored as K + jitter·I, with jitter = 1e-8·mean(diag)·10^k for the first k in 0..6 that factors (spec §4.3). The likelihood, α and every prediction use that same jittered K. The gradient includes the jitter's own dependence on σ_f² and σ_n², so it is the exact gradient of the computed objective. If no step factors, the result is "no model" (null), never a NaN.
- **Predictions are of the latent f.** `predict` gives mean, variance and (jointly) the covariance of f, without noise. A negative variance (rounding) is clamped at 0, but a NaN is kept, so a corrupted posterior shows as NaN and its callers refuse it (Task 4). `predict` returns null when a row of xs is not d finite numbers. The joint covariance is symmetric bit for bit, and its diagonal is the same number as the variance.
- **Malformed input is "no model".** Empty data, rows of the wrong length, or a non-finite x, y or hyperparameter gives null from `objective` and `conditionGp`, not an exception.
- **The overlap kernel for categoricals is specified, not built (plan 05b, conflict 4).** Genome schema 1 searches no categorical, and `parseGenomeSpace` rejects a categorical gene owned by `gp`, so no model can meet one and 05b has no overlap-kernel code. The schema decision that activates the first GP categorical must add:
  - the kernel factor exp(−Σ_c 1[x_c ≠ x'_c]/ℓ_c) on top of σ_f²·Matérn-5/2 of the continuous genes (spec §4.3), with ln ℓ_c ~ N(0, 1) [guess];
  - uniform levels for uniform candidates, and the parent's level kept with probability 0.9 for local ones;
  - a posterior schemaVersion bump. A schema change is a fresh fit anyway, so nothing is lost.
- **Bit-identity across engines is claimed, not yet shown.** Every operation is + − × ÷, `Math.sqrt` or `@blanktab/detmath`, in fixed loop order. `packages/meta/test/fixtures/gp-golden.json` pins the outputs in Node bit for bit. The cross-browser check is Task 6.

### The fit [agent; within spec §4.3]

Built in `packages/meta/src/lbfgs.ts` and `packages/meta/src/fit.ts` (Task 3). It maximises log marginal likelihood + log prior by minimising its negative.

- **Starts.** Start 0 is the prior mode. Starts 1–3 are mode + 1.5·sd·(2u − 1) for u in `sobolPoints(d + 3, 3, "gp-fit/1")`. More than 13 searched genes (d + 3 > 16 Sobol dimensions) throws: that needs a recorded decision that also adds direction numbers. A start whose objective is null at its θ0 is skipped and records 0 iterations. If all four are null the fit is "no model" (null).
- **L-BFGS.** Memory 10, two-loop recursion with H₀ = γI (γ = sᵀy/yᵀy of the newest pair). Backtracking Armijo line search with c₁ = 1e-4. The first step is 1/max(1, ‖g‖∞) on the first iteration and 1 after that. It halves at most 30 times. A null objective counts as no decrease. A pair with sᵀy ≤ 1e-10 is skipped. If rounding makes the two-loop direction non-descending, the memory is cleared and the step is along −g [agent's guard, not in the plan text].
- **Stopping.** |Δvalue| < 1e-6, ‖g‖∞ < 1e-8, a failed line search, or 100 iterations.
- **The winner.** The lowest negative objective, compared with a strict `<`, so ties go to the earliest start.
- **Determinism.** No randomness and fixed constants. `packages/meta/test/fixtures/fit-golden.json` pins two fits in Node bit for bit.

### Fit cost [measured]

`fitGp` on seeded, standardised rows with d = 12, Node v22.23.4, this machine (Apple silicon laptop), single run each:

| Rows | Wall time | Iterations per start | Winning start |
|---|---|---|---|
| 300 | 6.9 s | 90, 58, 100, 78 | 2 |
| 1,000 | 177 s | 80, 61, 100, 47 | 1 |

Both are under the 120 s stop rule at 300 rows and well within the gate job's 30-minute limit at the 1,000-row cap. The spec's estimate (§8: ~10 s at 300, 1–3 min at 1,000) holds. One start in each run hit the 100-iteration cap.

### Thompson sampling and the incumbent [agent; within spec §4.3–§4.4]

Built in `packages/meta/src/thompson.ts` (Task 4).

- **Candidate spread is capped: σ_j = min(0.1·ℓ_j, 0.1)** (ruling 4). The spec's 0.1·ℓ_j alone is about 1.4 at the prior's ℓ ≈ 14, so most children would land on the cube's faces.
- **Fewer than 16 training rows: the parents cycle in rank order**, so the set is always 256 uniform + 256 local = 512 candidates. Not in the spec.
- **Draw order.** 256·d uniform draws (row by row), then 256·d child draws (parent by parent, child by child, gene by gene), then, only if Σ + jitter factors, 512 draws for z. One draw u becomes a normal as normInv(u + 2^-33), which is finite for every rng() value.
- **The sample's jitter is absolute**: 1e-6·I, ×10 up to 5 steps (1e-6 … 1e-2), not scaled by σ_f². No step factoring gives null (the tab draws centre).
- **No NaN reaches a draw.** `thompsonPick` returns null, never a NaN or out-of-cube x, when a candidate is not d numbers in [0, 1], the prediction is null or not finite, or no jitter step factors.
- **The incumbent's x is rounded to 6 decimals before its mean and P(beats centre) are computed**, so the numbers the gate writes describe the x it writes. The argmax itself is taken on the unrounded candidates.
- **Seed.** The gate seeds the incumbent's candidate set with hashSeed(0, "incumbent/" + modelSha), modelSha = sha256 of the serialised model block (ruling 4; the spec's posteriorSha is circular).
- **Test fixture.** `packages/meta/test/fixtures/posterior-v2-search.json` is now a real fit (40 seeded rows over the 12 searched genes) with its incumbent from `incumbentOf`; `thompson.test.ts` rebuilds it and checks it byte for byte.

### Birth cost [measured]

One tab birth = `conditionGp` + `candidateSet` + `thompsonPick` (512 candidates, d = 12), Node v22.23.4, this machine, best of 3 runs; `incumbentOf` (gate only) separately:

| Rows | conditionGp | candidateSet | thompsonPick | Birth total | incumbentOf |
|---|---|---|---|---|---|
| 300 | 7 ms | 3 ms | 71 ms | 83 ms | 9 ms |
| 1,000 | 167 ms | 37 ms | 302 ms | 507 ms | 56 ms |

That is under the spec's estimate (§4.3: 1–3 s at birth in a worker). Browsers are not measured here (Task 6).

### Draw order [agent; within spec §4.1, §4.5]

`birthDraw` (`packages/meta/src/draw.ts`) keeps 05a's checks (genome schema, centre sha) and its first draw, u0 = rng(). After that, the rng is read in this exact order:

- **gpu + calibrate:** unchanged from 05a (one rng() picks a calibration genome by need).
- **gpu + search:**
  1. **u0 < 0.2 → `uniform`**, unchanged (d more draws).
  2. **A confirmation in flight and u0 < 0.7 (`CONFIRM_SHARE_END`):** j = ⌊rng()·1000⌋ (`CONFIRM_SEEDS`), arm = `ARM_OF_J[j mod 4]` = candidate, centre, newborn, newborn. If need[arm] > 0 the draw is `confirm`, role `confirm`, with evalSeed = seedBase + j, evalPanel null and confirmation { id, arm }. The candidate and newborn arms run `candidate.values`; the centre arm runs `centre.values`. If need[arm] = 0, the birth goes on to `ts` with the same rng, after j.
  3. **`ts`:** `candidateSet(post, rng)`, then `thompsonPick(post, cands, rng)`; the genome is `fromUnit(space, pick.x, centre)`. It falls back to `centre`, with a reason, when there is no gpu model, when the model's genes are not exactly `searched(space)`'s names in order, when `modelPosterior` is null, or when `thompsonPick` is null.
- **paused, flat, and every mode on cpu:** unchanged. CPU tabs never draw `ts` or `confirm`.
- **The mode is tested first.** The branches check `widthsClass === "gpu" && mode === "search"` before they read `models` or `confirmation`. So even a hand-built `Posterior` (bypassing the parser's mode guard) with a model or a confirmation in mode calibrate, paused or flat draws exactly as 05a.
- **A v1 file draws 05a's draws exactly.** It has no model, so its gpu search births with u0 ≥ 0.2 draw `centre` with no reason, as 05a did. A v2 search file without a gpu model gives `centre` with the reason "no gpu model". `evalSeed` and `confirmation` are null for every method but `confirm`, and 05a's `packages/meta/test/fixtures/draws.json` is unchanged (`fixtures/draws-05b.json` pins 20 ts and confirm draws).
- **A confirmation genome is checked like a calibration genome.** If `checkGenome` refuses it, the draw is `centre` with a reason, not a life the session would refuse.
- **`armOfSeed(evalSeed, id)` = `ARM_OF_J[(evalSeed − 10000 − 1000·id) mod 4]`** (the mod taken non-negative), so the arm can be recomputed from the seed alone.

## Alternatives considered

- **Keep `schemaVersion` 1 and add the new fields to it.** Rejected. 05a's parser accepts any object for `models` and `incumbent`, but checks the page's exact keys, so a new page field would make 05a tabs refuse the file anyway, and silently. A version bump says so on purpose, and both outcomes are the same safe draw (`centre`).
- **A "search enabled" flag in the posterior or the config.** Rejected. A flag is one more thing that can be set by hand. The guard instead makes the search's state itself unreadable outside the mode that only the signal check sets.

## Consequences

- A v2 posterior is all-or-nothing. A single bad number in the model makes every tab served that file draw `centre` until the gate writes a good one. That is the intended failure mode: unfavourable but safe.
- During a deploy, a 05a tab served a v2 file draws `centre` and a 05b tab served the last v1 file draws as 05a did (Review Focus 1). Task 13 checks the first pairing against the real 05a code.
- Later 05b tasks add their calls below this point.
