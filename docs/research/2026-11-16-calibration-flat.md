# Why calibration came out "flat"

**Date:** 2026-11-16. **Data:** `data/runs/` and `data/newborns/` at main `a30716f`. **Script:** `npx tsx tools/eval/flat-diagnosis.ts`. It is deterministic: all simulations are seeded, and two runs gave byte-identical output. It writes `.local/flat-diagnosis.md` and `.local/flat-diagnosis.json`, and every number below comes from that output. Lives are scored only by the registered evaluators (`search-composite@1`, `signal-check@1`, the pinned newborn reference). Nothing here changes an evaluator, a config or a decision. Any change listed under Options restarts calibration and needs a decision recorded by the user (invariant 7).

## Summary
- **Reproduced exactly.** Re-running `signal-check@1` on look 3's 60 learnerIds gives the recorded result: p 0.0241, ICC 0.165.
- **The score combines a component that varies between genomes with one that doesn't.** The behaviour part C differs clearly between genomes: ICC 0.40 [0.15, 0.73], Kruskal–Wallis (KW) p 0.001, F p 9e-5. The prediction part H does not: ICC 0.04 [0.00, 0.36], KW p 0.22. H is also the noisier part (var 0.32 against C's 0.09), and the rank average gives H equal weight. The test that sees both, KW on a_i, ends at p 0.024. That is just above the per-look α of 1/60.
- **Was flat at 60 lives bad luck?** Under the variance components fitted to these 60 lives, the `signal-check@1` rule has about **61% power at 6 repeats** (simulated). So flat was a ~40% outcome even if the fitted effects are real. The rule reaches 80% at about **10 repeats** (100 lives). The analytic F test on the score needs 15 repeats. Power levels off near 0.89, because with only 10 genomes the ICC ≥ 0.1 floor sometimes fails by chance.
- **No genome beat the centre.** Against all 41 centre-genome lives, every Sobol genome is equal or worse. The three worst (S1, S5, S9) all have high lr. C's rank correlation with lr is −0.61 (Holm p 0.029 over 36 tests). But lr is confounded with exploreEpsilon across the 10 points (ρ −0.76), and replayPerSleep and gamma have identical ranks (ρ 1.00).
- **At 2,048 CU the learners are not better than newborns on most of what the score measures.**
  - Trained lives pass M2, M3 and M5 *less* often than untrained newborns.
  - They pass M1 and M4 more often.
  - Their held-out prediction is worse than copy-last (median ρ̃ 1.13; only 32% of lives have ρ̃ < 1).
  - The between-genome signal in C comes from M2 and M3, which are the milestones training makes worse. It does not come from M4, which training improves but where genomes don't differ (ICC 0.01).
  - So C's signal mostly measures how far training pulls a learner away from newborn behaviour. That is a guess; see "What it doesn't support".
- **Data quality did not inflate the noise.** I found no checkup errors and no null measures. Lives on CPU, lives restored after a GPU failure, and lives that crossed a code change all look like ordinary lives. Panel balance is acceptable.
  - One learner-side finding: in 12 of 95 lives the first-layer filters barely move between 512 and 2,048 CU (< 1%), and in 2 lives they are bit-identical.
- **Recommendation [agent's call, not yet confirmed]:** option (b). Write a new evaluator version that ranks on C only, with H kept as a reported diagnostic, and run a fresh calibration of 10 genomes × 6 repeats. Do not widen the ranges. The details are under Options.

## What was run
- **Calibration set:** exactly the 60 learnerIds of look 3 (`data/meta/calibration.json`), grouped by genome as the look grouped them (5, 6 or 7 lives per genome).
- **Repeat-noise pool:** every eligible centre-genome life (`d22c1c…`). That is 41 lives: 6 from calibration and 35 born since flat (method `centre`).
- **The search score's parts,** read from `search-composite@1`'s own output:
  - score, C and H;
  - C per level (`perLevel`), C per milestone (`perMilestone`), and each q_{m,L};
  - M1's diagnostic q;
  - from the checkups' pass lists, pass indicators per milestone × level, sustained passes, and pass counts.
- **For each part:**
  - the method-of-moments σ²_b and σ²_w;
  - ICC(1) with its F-based 95% CI (Shrout–Fleiss/Donner, unbalanced via n0);
  - the KW and one-way ANOVA p across the 10 genomes;
  - the SD over the centre pool, with a χ² CI;
  - ICC*, which is the ICC with σ²_w taken from the centre pool.
- **Power, analytic:** the random-effects F test, using F/(1 + n·ICC/(1 − ICC)) ~ F(9, 10(n − 1)) at α = 1/60.
- **Power, simulated, of the actual rule.** The rule is KW on `rankAverage` from `search-composite@1` at α = 1/60, together with `icc1(score)` ≥ 0.1, all using the evaluators' code.
  - Genome effects (C, H) were drawn from N(0, Σ_b), with Σ_b fitted by MANOVA moments.
  - Residual (C, H) vectors were resampled from 95 real within-genome residuals, so their non-normal shape is kept.
  - 2,000 runs per n.
- **Single-part rules:** the same rule applied to one part alone. Continuous parts used normal effects plus resampled residuals; pass indicators used a beta-binomial. 1,000 runs per n.

## Numbers

### 1. Parts across the 10 genomes (60 lives)
Selected rows. The full table, about 50 parts, is in the script output.

| component | mean | ICC | 95% CI | KW p | F p | σ_w calibration | σ_w centre pool [CI] (n = 41) | ICC* |
|---|---|---|---|---|---|---|---|---|
| **score** | −0.713 | 0.165 | [0.00, 0.52] | 0.019 | 0.039 | 0.666 | 0.484 [0.40, 0.62] | 0.343 |
| **C** | −0.386 | **0.397** | [0.15, 0.73] | **0.001** | **9.3e-5** | 0.243 | 0.249 [0.20, 0.32] | 0.383 |
| **H** | −0.327 | 0.041 | [0.00, 0.36] | 0.221 | 0.285 | 0.558 | 0.353 [0.29, 0.45] | 0.261 |
| C@512 | −0.387 | 0.209 | [0.01, 0.57] | 0.015 | 0.016 | 0.389 | 0.295 | 0.367 |
| C@1024 | −0.367 | 0.223 | [0.02, 0.58] | 0.016 | 0.012 | 0.379 | 0.324 | 0.312 |
| C@2048 | −0.405 | 0.258 | [0.05, 0.62] | 0.005 | 0.005 | 0.364 | 0.443 | 0.153 |
| q.M2 | −0.711 | 0.235 | [0.03, 0.60] | 0.006 | 0.009 | 0.427 | 0.520 | 0.132 |
| q.M3 | −0.932 | 0.220 | [0.02, 0.58] | 0.003 | 0.012 | 0.665 | 0.673 | 0.213 |
| q.M4 | −0.289 | 0.011 | [0.00, 0.31] | 0.481 | 0.405 | 0.635 | 0.563 | 0.056 |
| q.M5 | −0.259 | 0.122 | [0.00, 0.47] | 0.125 | 0.085 | 0.202 | 0.243 | 0.044 |
| q.M6 | 0.259 | 0.139 | [0.00, 0.49] | 0.044 | 0.064 | 0.403 | 0.469 | 0.070 |
| q.M1 (diagnostic) | −0.138 | 0.178 | [0.00, 0.54] | 0.065 | 0.030 | 0.249 | 0.187 | 0.338 |
| passes@2048 (M2–M6) | 0.533 | −0.074 | [0.00, 0.15] | 0.800 | 0.801 | 0.672 | 0.805 | 0.000 |
| sustained count (M2–M6) | 0.450 | −0.062 | [0.00, 0.17] | 0.616 | 0.748 | 0.549 | 0.505 | 0.000 |

- **a_i, the rank average that `signal-check@1` tests:** KW p 0.024, ICC(a) 0.229. Spearman(C, H) over the 60 is 0.27.
- **Single q_{m,L} cells and pass indicators** are mostly at ICC ≈ 0. The best single cell is q.M3@512 at 0.24. Averaging over levels and milestones is what gives C its ICC.
- **Pass-based parts carry no detectable genome signal.** Sustained passes of M2 and M5 never happen in the 60 lives.
- **H's noise depends on the genome.** Its within-genome SD is 0.56 in calibration but 0.35 in the centre pool. The difference comes from occasional blow-ups: 11 of 95 lives have ρ̃ > 2, and 4 are clipped at H = −2. In calibration these sit in high-lr genomes (S1 2, S5 2). Part of the "noise" in H is therefore a genome's risk of diverging, which the one-way ICC model counts as within-genome variance.

### 2. Trained lives against newborns (95 search lives; the 64 reference newborns)
The p column is a two-sided Fisher test on passes per checkup. It is optimistic, because one life's three checkups are not independent.

| milestone | newborn passes / checkup | search passes / checkup | p | search sustained (newborn) | median q |
|---|---|---|---|---|---|
| M1 | 0/192 | 101/285 (0.35) | 7e-27, search higher | 0.34 (0) | −0.16 |
| M2 | 8/192 (0.04) | 2/285 (0.01) | 0.024, search lower | 0 (0) | −0.60 |
| M3 | 42/192 (0.22) | 25/285 (0.09) | 1e-4, search lower | 0.04 (0.11) | −0.98 |
| M4 | 36/192 (0.19) | 112/285 (0.39) | 2e-6, search higher | 0.41 (0.14) | −0.12 |
| M5 | 19/192 (0.10) | 11/285 (0.04) | 0.014, search lower | 0 (0.03) | −0.29 |
| M6 | 6/192 (0.03) | 13/285 (0.05) | 0.59 | 0.02 (0) | 0.00 |

- **Held-out ratio ρ̃:** median 1.131, range 0.83–175. The newborn reference is 1.000 (an untrained predictor is copy-last).
- **How much a measure belongs to the learner** (test–retest ICC of raw measures):
  - **Same frozen weights, three different worlds** (newborns): M2 0.09, M3 −0.02, M4 0.30, M5 −0.02, M6 −0.04.
  - **Same world, different weights** (4 newborns per panel × level cell): M2 0.16, M3 0.13, M4 0.45, M5 0.28, M6 0.15.
  - **One trained life's three checkups:** M2 0.20, M3 0.07, M4 0.27, M5 0.03, M6 0.27.
  - So a single checkup of M3 or M5 says almost nothing about the learner itself. The world draw dominates.

### 3. Genomes and genes
Each genome against the centre pool (n = 41, mean score −0.450). Welch 95% CI; Holm over 9 comparisons.

| genome | lr / exploreEpsilon | n | mean C | mean H | Δ score vs centre | 95% CI | Hedges g | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 3f8dd9 | 1.4e-3 / 0.033 | 5 | −0.557 | −0.859 | −0.97 | [−2.71, 0.78] | −1.51 | 0.20 (1.00) |
| S2 1e95ea | 6.8e-4 / 0.066 | 6 | −0.272 | −0.146 | +0.03 | [−0.47, 0.53] | 0.07 | 0.88 (1.00) |
| S3 179bc0 | 1.7e-4 / 0.264 | 6 | −0.208 | −0.210 | +0.03 | [−0.29, 0.36] | 0.07 | 0.83 (1.00) |
| S4 008ce7 | 2.4e-4 / 0.186 | 6 | −0.170 | −0.362 | −0.08 | [−0.71, 0.54] | −0.16 | 0.76 (1.00) |
| S5 56e0b0 | 9.6e-4 / 0.047 | 6 | −0.521 | −0.690 | −0.76 | [−1.31, −0.22] | −1.53 | 0.014 (0.12) |
| S6 e1aaaf | 1.9e-3 / 0.093 | 7 | −0.661 | −0.090 | −0.30 | [−0.60, −0.00] | −0.64 | 0.047 (0.38) |
| S7 aac76d | 4.8e-4 / 0.373 | 6 | −0.350 | −0.157 | −0.06 | [−0.38, 0.26] | −0.12 | 0.70 (1.00) |
| S8 fd6a26 | 4.1e-4 / 0.222 | 6 | −0.213 | −0.191 | +0.05 | [−0.34, 0.43] | 0.09 | 0.79 (1.00) |
| S9 0574db | 1.6e-3 / 0.055 | 6 | −0.746 | −0.513 | −0.81 | [−1.79, 0.17] | −1.44 | 0.09 (0.63) |

- **Gene correlations.** For each gene, the p comes from 20,000 permutations of genome labels; that test effectively has 10 points. Holm is over 12 genes × {score, C, H} = 36 tests.
  - Only **lr with C** survives Holm: ρ −0.61, p 8e-4, Holm 0.029.
  - lr with score: ρ −0.39, p 0.038, Holm 1.
  - exploreEpsilon with score: ρ +0.38, p 0.043, Holm 1.
- **The design cannot separate several genes.** Among the 10 points: replayPerSleep and gamma ρ 1.00; gamma and caregiverTau −0.89; headLrScale and bufferSize 0.87; exploreEpsilon and caregiverEpsilon 0.83; lr and exploreEpsilon −0.76.

### 4. Power: repeats per genome needed for 80% at α = 1/60 (10 genomes)
| score used | ICC | analytic power at 6 | analytic n for 80% (n at CI low end) | simulated rule: n = 3 / 6 / 10 / 15 / 30 |
|---|---|---|---|---|
| a_i + ICC(score), the `signal-check@1` rule | 0.165 (score) | — | — | 0.07 / **0.61** / **0.81** / 0.84 / 0.88 |
| score alone | 0.165 | 0.33 | 15 (never) | 0.05 / 0.48 / 0.78 / 0.83 / 0.86 |
| **C alone** | 0.397 | **0.85** | **6** (17) | 0.15 / **0.77** / 0.96 / 0.99 / 0.99 |
| H alone | 0.041 | 0.05 | 67 (never) | 0.01 / 0.14 / 0.21 / 0.17 / 0.12 |
| C@2048 alone | 0.258 | 0.59 | 9 (58) | 0.03 / 0.41 / 0.72 / 0.88 / 0.93 |
| q.M2 alone | 0.235 | 0.53 | 11 (86) | 0.03 / 0.38 / 0.72 / 0.84 / 0.91 |

**Assumptions:**
- Genome effects are random and normal. Repeats are exchangeable, and their residual shape is the one observed.
- The fitted ICC is taken as the truth. It comes from the same 60 lives, so these powers are optimistic, and the best single parts (picked from about 50) are biased upward by the winner's curse. C is one of the three headline parts, so it is the least affected.
- Ten genomes are drawn afresh in each simulation. That is why power levels off below 1: with 10 genomes, ICC(score) ≥ 0.1 sometimes fails.
- H alone loses power as n grows, because its true ICC (0.04) is below the 0.1 floor.

### 5. Data quality (the 60 calibration lives plus the centre pool, 95 lives)
| check | result |
|---|---|
| log files that do not parse | 0 |
| first parsed seq ≠ 0 / seq gaps | 11 / 0. All 102 eligible lives have checkups at all three levels. |
| checkups with `error: true`, null measures at 512/1,024/2,048 | 0, 0 |
| lives with a CPU file in the budget window | 4 (all after 2 GPU restores). Residuals of the 2 in calibration: −0.09, +0.18. The residual SD is 0.61. |
| lives restored from a checkpoint (GPU run failure, `app/src/life.ts`) | 19; median lostCU 128. Score minus genome mean: −0.06 on average. |
| lives crossing a code change in the window | 2 |
| build era (`369c5eb`, n = 28, against `249b7d0`, n = 32; score minus own-genome mean) | +0.08 against −0.07, KW p 0.22. Between the two builds only the PR-sending code changed (`contributor.ts`, `github.ts`, `main.ts`), none of the learner, world or milestones. |
| identical checkup measures at two levels; filters shared between lives | 0; 0 |
| conv1 filters, change 512 → 2,048 CU | median 14%; < 1% in 12 lives; bit-identical in 2 (`lrn_4c7ec98889ef9cbb`, `lrn_8f81e6c3dda13693`); 1,024 = 2,048 in 3. Not genome-dependent (ICC −0.03). Spearman with H −0.43, with C 0.00. |
| panels | 4–6 distinct panels per genome, none balanced. Panel effect on score within the centre pool: ICC 0.23 [0.00, 0.59], F p 0.11. |
| hosts | all 95 lives ran on one 10-core machine with one tab (the runner) |
| eligibility, all 109 search lives | calibrate: 61 eligible, 5 abandoned. centre: 35 eligible, 1 abandoned, 1 without a 2,048 checkup. uniform: 6 eligible. No legacy or failed lives among the rows. |

## What the evidence supports
- **Flat is mostly a score-design effect plus too few repeats, not "genomes don't differ."** C separates the 10 genomes at p 0.001. H adds noise and no detectable genome signal. Because the rank average weights the two equally, it diluted C below the look's α.
- **Under the fitted model, 6 repeats gave the current rule only about a 61% chance.** Flat was not surprising.
- **In the current space, the differences are mostly genomes that are worse than the centre.** High-lr points are worse. No point is credibly better (the best CI is +0.05 [−0.34, 0.43]).
- **Data quality is not the cause.** None of the issues checked shifts scores measurably.

## What it doesn't support (guesses marked)
- **That C would have signal on fresh lives.** The C-only power (0.77 simulated at 6 repeats) comes from the same 60 lives that suggested it. It needs fresh lives to confirm.
- **That lr is the cause.** lr is confounded with exploreEpsilon in this design, and the gene tests have 10 effective points.
- **What C's signal means.** *Guess:* C's genome signal mostly tracks how much training erodes newborn-level M2 and M3 behaviour; lower lr erodes less. If so, a search on C would push toward learning less. That is a waste of search, not a false promotion, because `confirm-fisher@1` still requires beating the candidate's own newborns.
- **That noise, rather than the learner, is the binding limit.** At 2,048 CU the learner is worse than copy-last on held-out prediction and worse than newborns on three of five behaviour milestones. *Guess:* better recipes within these 12 genes may not exist, whatever the score.
- **Why some lives' first-layer filters stop moving.** I did not investigate it. Possible explanations include dead units or a snapshot artefact; both are unverified.

## Options for the user
Each option restarts calibration, except (d). Each needs a decision the user records. The costs below are measured from the logs:
- A life takes a median of **1.47 h** from first log file to budget checkup (mean 3.7 h with stalls).
- Since flat, the runner has produced **5.1 eligible lives per day** on one machine.

- **(a) More repeats with the current score.** The rule reaches 80% at about 10 repeats (100 fresh lives, about 20 days), or 15 repeats by the analytic F test (150 lives, about 30 days). It levels off near 0.89, and H's noise stays in the score. If it signals, the GP would mostly learn that high lr is worse.
- **(b) A new evaluator version (`search-composite@2`).**
  - Rank on C only: the same q, levels and milestones as now.
  - Keep H, ρ̃ and the blow-up count as reported diagnostics, not in the ranking.
  - Leave pass indicators out of the search score: their ICC is about 0. The confirmation test keeps using them, as now.
  - Keep averaging over the 3 levels and 5 milestones. Any single level or milestone is weaker: ICC 0.21–0.26 against 0.40.
  - Fitted power is 0.77–0.85 at 6 repeats (60 fresh lives, about 12 days).
  - *Variant:* down-weight H rather than drop it. That is not evaluated here.
- **(c) A new genome schema.**
  - **Wider ranges** would mostly add points like S1, S5 and S9 (high lr, worse). There is no evidence that anything better lies further out.
  - **New genes** (the v2 mechanisms, spec §4.6) are the only route here to genomes that could beat the centre. They need learner code with bit-exact-at-centre goldens first.
  - *Guess:* a design with fewer confounded genes per point, such as more Sobol points with fewer repeats each, would make gene effects readable.
- **(d) Accept flat for now.** Births stay 80% centre, 20% uniform, which keeps adding repeat-noise data at no new cost. The next step would wait for learner-side work (held-out ρ̃ > 1, M2/M3/M5 below newborns, stalled filters).

**Recommendation [agent's call, not yet confirmed]:** (b), and not (a) or wide (c).
- (b) is the cheapest change that the data say would most likely have signal (about 12 days of one runner), and it needs no learner code.
- (a) costs 2–2.5× as much for the same diluted score.
- Run (b)'s calibration on fresh lives with the same 10-genome, 6-repeat design. The existing 60 cannot validate a score that was chosen by looking at them.
- Expect the search to favour lower lr, and expect confirmation to fail unless a genome beats its own newborns.
- If v2 genes are close to ready, bundle them with (b) into a single restart, so the 60-life calibration is paid once.
