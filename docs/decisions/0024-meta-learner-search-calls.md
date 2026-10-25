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

### Confirmation-newborn files [agent; settles 05a's open note on the log shape]

A confirmation's newborn arm uses the reference newborn's frozen file shape, so `confirm-fisher@1` reads it unchanged and keeps its version. (Task 7; `packages/harness/src/log.ts`, `newborn.ts`.)

- **The shape.** A file with `role: "confirm"` and `confirmation.arm: "newborn"` is a frozen file: `seq: 0`, `cu: 0`, `stepsTotal: 0` and `final: true`, with exactly its budget levels' checkups, run above cu (`allowAhead`). It may carry `meta.heldOut`. `isFrozenNewborn` names the two frozen kinds (role `newborn`, and a confirmation newborn), and `meta.heldOut` is allowed for those two only.
- **Only that shape.** An arm-`newborn` block in any other file (a running life's, a later seq, too few checkups) is refused by the parser, and so by the session's `newbornBlock` and the checkpoint loader. A candidate or centre life never gets frozen checkups or `heldOut`. It may still end at cu 0 (an orphan swept right after birth), as any life may.
- **The runner.** `runNewborn({ who: { kind: "confirm", evalSeed, confirmationId, drawSeed, posteriorSha } })` runs it: the candidate's genome, learner seed = evalSeed, no panel, role and method `confirm`. It checks its block with `parseMeta` before making a learner, so a seed of another arm is refused there. Reference mode (`who: { kind: "reference", k }`) is unchanged. A confirmation newborn's file is gate data and is contributed normally (the page wiring is Task 11).

### The arm is the seed's [agent; implements 05a's note "the gate recomputes the arm from evalSeed mod 4"]

- `parseMeta` refuses a confirm block whose `confirmation.arm` is not `ARM_OF_J[(evalSeed − 10000 − 1000·id) mod 4]`. Every reader of a log (the tab's writer checks, the checkpoint loader, the gate's validator, the evaluators' fixtures) applies it, so a tab cannot choose its arm.
- `log.ts` imports no package, so the 4-entry table is copied there as `CONFIRM_ARM_OF_J` (`confirmArmOfSeed`). `tools/genome-centre.test.ts` pins it equal to `@blanktab/meta`'s `ARM_OF_J`, and `confirmArmOfSeed` equal to `armOfSeed` on every seed of two blocks.

### Confirmation seeds [agent; within spec §4.5]

- **Checkup variants.** A life whose block has `evalSeed` (every confirmation life, every arm) meets `confirmSeed(evalSeed, level, m) = hashSeed(0, "confirm/<evalSeed>/<level>/<m>")`. Lives with the same evalSeed meet the same worlds. The `confirm/` prefix keeps them apart from every shared panel's `panelSeed` (tested: disjoint on the budget levels for all 16 panels). `variantSeedOf(meta)` is the one rule, shared by the session and the newborn runner: confirmSeed, else panelSeed, else 05a's `checkupSeed`.
- **The learner seed is evalSeed** (spec §4.5). `validateSessionPr` refuses a confirm file whose `settings.seed` is not `meta.evalSeed`. The session refuses such a birth (`BadMetaBlockError`) before making a learner, so a tab never writes a file the gate would refuse. The world seed stays the learnerId's.

### Every schema checked against its own space [agent; closes 05a Task 6's older-schema bypass]

- In 05a, a non-legacy genome of an older schema than main's current one was checked for structure only. That let a file claim an old schema to skip the bounds check.
- Now `ValidateOptions.spaces` (it replaces `space`) maps each schemaVersion to its space. Every non-legacy block is checked against `spaces.get(meta.genomeSchema)`: bounds, settings (`settingsMismatch`) and sha. A schema not in the map is refused (`"unknown genome schema N"`). A schema newer than the newest in the map is refused as before. Legacy blocks keep their pre-05a values (plan gap 4).
- **Where the map comes from.** `tools/gate/history.ts` has three functions:
  - `fileHistory` lists the distinct committed texts of a file, newest first, from the commits that added or modified it;
  - `spacesBySchema` keeps the newest text of `knowledge/genome-space.json` per schemaVersion, and skips texts that do not parse;
  - `centresBySha` lists every `genome.json` main has had, for Task 8.

  `gate.ts` and `validate-cli.ts` build the map with `mainSpaces(root, git)`: that history over main's full history (both checkouts are `fetch-depth: 0`), plus the checkout's current space. They stop, as before, when the current space is missing or invalid.
- `centresBySha` hashes with `node:crypto` synchronously. `history.test.ts` pins its sha equal to `@blanktab/meta`'s `genomeSha`.
- **Tools with one space.** `tools/eval` (collect-newborns, newborn-reference) passes `spacesOf(space)` with the checkout's space only. Newborn files of an older schema would now be refused there rather than checked for structure only. None exist.

### Rows and the fit [agent; within spec §4.3]

Built in `tools/gate/search.ts` and `tools/gate/meta.ts` (Task 8).

- **Only in mode `search`.** In every other mode `buildMeta` calls nothing of the search: no rows are subset, no fit runs, and the posterior is 05a's bytes with `schemaVersion` 2 and `page.rowsSinceDeclaration` 0 (a test rebuilds 05a's golden this way, byte for byte, and a spy on `fitGp` sees no call in calibrate or flat with 200 rows). `diagnostics.json` is unchanged in those modes.
- **Rows.** An eligible life under `config.search`, of method `calibrate`, `uniform`, `ts` or `centre`, on gpu widths, at the space's schema, with inheritance `none`, that verifies. Confirmation lives are verified but are never rows. y is `targets(...)` over all rows; above 1,000 rows, `rowSubset` keeps the 1,000 with the largest `hashSeed(0, "subset/" + learnerId)`. x is `toUnit` of the **logged** genome, rounded to 6 decimals (and kept in [0, 1]).
- **No model, with a reason.** Fewer than d + 1 rows (13 at d = 12), a non-finite target, a fit that throws (ragged data, or more than 13 searched genes), a fit with no finite start, or a fitted model that `conditionGp` refuses: `models` is `{}`, the incumbent is null, and the posterior's `diagnostics.model` is `{ reason }`. Tabs then draw centre (and uniform).
- **Reuse.** When the previous posterior's model has the same genes, rows, x and y, its `hyper` and `fit` are reused without refitting. The fit is deterministic, so this changes no byte (tested). `FIT_ROWS` is not part of that key: a change of it takes effect at the next change of the rows.
- **The hyperparameters are fitted on at most 500 rows** (`FIT_ROWS`) [agent's call, not in the spec]. Above 500 rows, the fit uses the 500 model rows with the largest subset hash; the model still conditions on all its rows (up to 1,000), and tabs sample from that. Why: a gate run may build up to 3 times (commit-meta's rebuild on a moved main), and a full fit at 1,000 rows is 177 s on the laptop, so about 6–9 min on a CI runner (2–3× slower), up to 27 min for three, against gate.yml's 30-minute limit. Measured on the laptop (Node 22): `fitGp` 4.5 s at 300 rows and 19.7 s at 500; `buildModel` at the 1,000-row cap (fit on 500, condition on 1,000) 17.5 s, and `incumbentFor` 0.22 s. So one build's fit is under a minute on CI and three are under 3 minutes. The cap is deterministic: no clock decides anything. Only the gpu class is fitted, so this is the whole fit time per run.
- **What the fit did** goes in `diagnostics.json` `model` (rows, fitRows, refit, the starts that took no step, the reason), never in the posterior, whose bytes must not move between a refit and a reuse.
- **Rows since declaration** (controller ruling, Task 8 review): the rows born under a posterior whose mode is `search` (from each row's verdict, below), so offline or calibrate-era rows never count. Before the first declaration that is every search-era row (plan conflict 5); Task 9 counts from the last declaration's `rowsAtDeclaration`.
- **The incumbent** is `incumbentOf` seeded with `hashSeed(0, "incumbent/" + sha256(serialise(model)))` against the current centre; its genomeSha is that of `fromUnit(space, x*, centre)`. `triggered` holds when pBeatsCentre ≥ 0.95, ≥ 100 rows since the declaration, no confirmation in flight and an incumbent that is not the centre; Task 9 acts on it.
- **Posterior size.** At 1,000 rows the posterior is about 285 KB (the spec estimated 150 KB).

### Centre in force [agent; closes 05a Task 10's note; Review Focus 3]

- A tab draws with its own `knowledge/genome.json`, which a promotion may have moved. The gate therefore tries up to two centres when it recomputes a draw: the posterior's `centre` (the one it was built for), then, for a `centre` draw only, the logged genome itself when its sha is one of the centres main has had (`centresBySha` over `knowledge/genome.json`'s history, plus the checkout's own). The first that reproduces method, drawSeed, evalPanel, evalSeed and confirmation, with the exact genome, wins. A life born under P whose tab held an older centre B therefore verifies as a `centre` draw with the fallback "posterior built for another centre", and is a row on B.
- **A `centre` draw whose genome is no centre main ever had is flagged `"unknown centre"`**, checked right after the posterior, as an unknown posteriorSha is.
- **Copies are exact.** A `centre` or `confirm` draw must reproduce the genome exactly; only `uniform`, `ts` and `calibrate` (within the design) keep a row on a logged genome that differs from the recomputed one (spec §7.4). For a confirm life that means its genome is the confirmation's candidate (candidate and newborn arms) or its centre (centre arm), as the gate fixed them in the declaration (spec §4.5 "Integrity"); otherwise `"genome is not confirmation <c>'s candidate|centre"`.
- **Confirm lives** are verified like draws (evalSeed and `{ id, arm }` included) and kept for Task 9. Two lives with the same evalSeed both verify; Task 9 counts only the lower learnerId.

### Verification cache [agent; settles 05a's note on posteriorHistory growth]

- `data/meta/verified.json` = `{ schemaVersion: 1, verifier: 2, lives: { <learnerId>: { posteriorSha, posteriorMode, reason, genomeMismatch } } }`, written every build with the verdict of every life checked. A verdict holds only facts of the birth (the posterior's bytes and mode, the space, the centres main has had); `posteriorMode` is the mode of the posterior the life was born under (null offline or unknown), kept so rows since declaration need no old posterior.
- A life in it with the same posteriorSha is not verified again, and `wantedPosteriors` asks the history walk only for uncached lives' posteriors (of the space's schema, as `readLives` checks), so the walk stops at recent posteriors.
- **The cache never decides** (controller ruling). A missing or malformed cache, or one of another `verifier`, verifies every life again. The calibration-design check is not in the verdict: it runs every build after the cache lookup, so a calibration restart with another design flags the same lives with or without the cache (tested), as does every other output (a build from no cache gives the same files; tested). Scoring and rows are recomputed every build.
- **The verifier is pinned.** `VERIFIER` (2) is exported; `meta.test.ts` pins the sha256 of `meta.ts`'s text between the `verifier: begin` and `verifier: end` markers to it, so a change of the verification rule without a `VERIFIER` bump fails CI. (The draw itself is pinned by `packages/meta`'s draw goldens.)
- **Cost bounded by memoised conditioning** (controller ruling, Task 8 review; no cap). `birthDraw` accepts an optional `conditioned` thunk; the gate passes one memoised per posteriorSha, so the GP is conditioned once per distinct posterior and each ts life costs only its candidate set and Thompson pick. Measured (laptop, Node 22): re-verifying 1,000 ts lives under 20 distinct posteriors at 1,000 rows takes 331 s memoised (0.33 s per life: the candidate set and the Thompson pick, which no posterior can share), against 500 s unmemoised. On CI (2–3×) that is about 11–17 min for a lost cache: under the 30-minute limit, with the fit (under a minute) on top. The per-life pick, not the conditioning, is now the cost; about 1,800 ts lives at the 1,000-row cap would reach 30 min on a slow runner. **Open:** if the population gets there, a further bound is needed.
- `commit-meta.ts` commits `verified.json` once Task 10 widens `META_PATH_RE`.

### Calibration consistency [agent; Review Focus 2]

`keptCalibration` refuses a `calibration.json` whose outcome is not the one its looks give, so a hand-edited "signal" cannot switch the search on. Every look must be an eligible output of the pinned `config.signal`, scored 0 or 1. `signal` needs the last look scored 1 and every earlier one 0; `flat` needs all 3 looks, all 0; `pending` needs fewer than 3 looks, all 0. The gate then stops and writes nothing.

### Confirmation timing [agent; within spec §4.5; the 48 h is a guess]

Built in `tools/gate/confirm.ts` (Task 9), called by `buildMeta` in mode `search` only.

- **Declaration.** On `triggered`, with no confirmation open and none decided in the same build, the gate writes `data/meta/confirmations/<c>.json` (c = the number of earlier files, from 0): the incumbent's genome as the candidate, the current centre, `seedBase` 10,000 + 1,000c, targets 24/24/60, the pinned `confirm` and `search` ids, and the trigger's inputs (pBeatsCentre, the incumbent's mean, `rowsAtDeclaration`, the model's sha). The same build serves it in the posterior, so the declaration and its first serving land in one gate commit.
- **Rows since declaration** are the search-era rows (Rows and the fit) minus the last declaration's `rowsAtDeclaration`. The next trigger needs 100 more.
- **No declaration in a deciding build** [agent's call]. A build that appends a result declares nothing, so a promotion and the next declaration never share a commit; the next build may declare.
- **The deadline** is `declaredAt` + 21 days [guess, spec §4.5], both the build's `builtAt`. A confirmation is decided at the deadline only when `builtAt` is strictly later. These times live only in the confirmation files (and the index line), never in `posterior.json`.
- **The clock.** `builtAt` is meta-cli's one clock read, passed in. It also decides which running lives hold a slot, so a posterior's `need` can change with time alone.
- **Serving `need`.** need = max(0, target − counted − inFlight) per arm:
  - **counted** is the counted lives (complete or failed, as `confirm-fisher@1` counts them) among the lives `armGroups` keeps;
  - **inFlight** is the arm's lives, one per evalSeed, that are not final, have no budget checkup, and whose newest file was written within 48 h of `builtAt` [guess].

  This bounds overshoot when many tabs are born at once; extras beyond the target are still dropped by rule.
- **The centre in force.** `parsePosterior` refuses a v2 file whose `confirmation.centre` is not its `centre` (sha and values). A tab's centre-arm life therefore always runs the tab's centre. Suppose `knowledge/genome.json` changes by hand while a confirmation is open. The gate then decides that confirmation at once as `incomplete` (the output is still recorded) and never promotes it, because its comparison was against a centre no longer in force [agent's call].
- **The files' checks.** `meta-cli` reads `data/meta/confirmations/` as exactly `0.json` … `(n−1).json`, each with its own id, or it stops. `parseConfirmation` is strict, and only the last file may be open. An open confirmation must match the pins, the space's schema and its genomes' shas, or the gate stops: a pin or schema change is a recorded decision.

### Decision outcomes [agent; within spec §4.5, decision 0018]

- **The groups** (`armGroups`). First, per (arm, evalSeed), the lowest learnerId (the arm is `armOfSeed`'s). Then, per arm in learnerId order, the lives up to and including the target-th counted one. Lives among those that are not counted (abandoned, or still running) are passed too [agent's call]. `confirm-fisher@1` excludes them, and reports their abandonment rates, which it flags when the arms differ (spec §2.5).
- **When.** Once counted reaches the target in every arm, or once `builtAt` is past the deadline. The gate runs `getEvaluator(config.confirm).evaluateCohort(groups, refs, cohortContext(config))`, resolved through the registry (never imported), and appends `result` once:
  - full arms, score 1 → `pass`;
  - full arms, score 0 → `fail`;
  - full arms, output not eligible → `ineligible`;
  - the deadline first → `incomplete`, with the output still recorded.
- **What is recorded.** `result` = { outcome, decidedAt (`builtAt`), sourceCommit, the evaluator's output, the learnerIds passed per arm }.
- **Never changed again.** A file with a result is never rewritten. Fail, incomplete and ineligible change nothing else, and they stay on the record (invariant 7).
- **Lives never mix.** Confirmation lives are never rows, and rows are never confirmation lives.

### Promotion record [agent; within spec §4.5]

On a pass, the same build does four things:

- **`knowledge/genome.json`.** It returns the new file: `serialise({ schemaVersion: 1, genomeSchema, values: candidate, provenance: { source: "promotion", confirmation, decision: 0018, promotedAt, fisher, mannWhitneyP, lives } })`. The file must read back through `parseCentre` as the candidate, or the build throws.
- **`promotions.jsonl`.** One line: `{ confirmation, genomeSha, previous, promotedAt, sourceCommit }`.
- **The posterior.** It is built for the new centre, with its incumbent recomputed against it. `page.lastPromotion` is the last pass's `decidedAt`, kept by every later build.
- **Running lives** keep their birth genome, and their rows stay eligible: they verify against the centre in force at their birth.

`meta-cli` writes the files and appends the line. Committing them is Task 10's (`commit-meta.ts`). `knowledge/genome.json` is non-null only in the build that appends a passing result, so the two always share a commit.

## Alternatives considered

- **Keep `schemaVersion` 1 and add the new fields to it.** Rejected. 05a's parser accepts any object for `models` and `incumbent`, but checks the page's exact keys, so a new page field would make 05a tabs refuse the file anyway, and silently. A version bump says so on purpose, and both outcomes are the same safe draw (`centre`).
- **A "search enabled" flag in the posterior or the config.** Rejected. A flag is one more thing that can be set by hand. The guard instead makes the search's state itself unreadable outside the mode that only the signal check sets.

## Consequences

- A v2 posterior is all-or-nothing. A single bad number in the model makes every tab served that file draw `centre` until the gate writes a good one. That is the intended failure mode: unfavourable but safe.
- During a deploy, a 05a tab served a v2 file draws `centre` and a 05b tab served the last v1 file draws as 05a did (Review Focus 1). Task 13 checks the first pairing against the real 05a code.
- Later 05b tasks add their calls below this point.
