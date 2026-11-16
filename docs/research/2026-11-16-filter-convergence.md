# Do fresh lives learn the same first-layer filters? (Plan 05c, the filter-convergence probe)

- **Date:** 2026-11-16
- **Plan:** `docs/plans/2026-10-28-05c-filter-probe.md`. **Calls:** decision 0027 (proposed). **Spec:** §5.1 of `docs/specs/2026-10-22-meta-learner-v1-design.md`.
- **Probe fixed at:** branch `plan-05c`, commit `1045693` (Task 6). **Data read:** `data/runs` at `a30716f` (main when the branch was cut; the branch changes nothing under `data/`).
- **Report:** `2026-11-16-filter-convergence/report.json`, byte for byte as the tool wrote it.

## The question

Decision 0018's agent's call (not yet confirmed) makes the learner's first-layer filters the first candidate to move up the hierarchy. That only makes sense if separate fresh lives learn roughly the same filters. Spec §5.1 asks: is the mean learned-pair CKA at 2,048 CU at least 0.1 above the mean init-pair CKA, is the median matched unit correlation at least 0.5, and is a permutation test over lives below p 0.05, over ≥ 10 lives of one genome and separately across genomes? (The bar is the spec's guess.)

## The answer

**Neither encoder's filters converge, by the spec's bar, either within one genome or across genomes.** Learning makes lives' first layers *less* alike than their random starting filters, not more. Each line is a verdict on its own (decision 0027: no combined verdict); the Holm verdicts agree with the raw ones everywhere.

- **Across genomes (pooled, n = 99): periphery — not converged** (ckaDelta −0.389, matched 0.000, p 1.0). **Fovea — not converged** (ckaDelta −0.029, matched 0.257, p 0.897).
- **One genome, the centre `d22c1c55` (n = 41, evaluable): periphery — not converged** (ckaDelta −0.347, matched 0.000, p 0.9998). **Fovea — not converged** (ckaDelta −0.051, matched 0.295, p 0.934).
- No other genome has 10 lives (5–6 each, not evaluable; shown below as description).

## The data

- 127 learner directories under `data/runs` at `a30716f`; **99 eligible**, 28 excluded (every one listed in `report.json`):
  - 18 `no meta` (pre-05a lives, including the one legacy life, whose early files have no meta block);
  - 5 `no snapshot at 512` (`lrn_0adb1fb0fabb8f49`, `lrn_1b06e2ef1a7a4a2e`, `lrn_245f90f55bad891a`, `lrn_6f8258735e6ae8ba`, `lrn_e98ab50d3b616cae`: ended or still running below 512 CU);
  - 2 `no snapshot at 2048` (`lrn_2897b594eb31a53a`, `lrn_5b874cfab7665c57`);
  - 3 `filters frozen after …` (Task 9 below): `lrn_4c7ec98889ef9cbb` and `lrn_8f81e6c3dda13693` after 512, `lrn_4d41b1c2f0e6eb7e` after 1,024.
  - No life was excluded for role, inheritance, widths, seed, or a shared seed.
- **Eligible per genome:** the centre `d22c1c55` 41 (6 calibration lives and 35 `centre` lives born after calibration ended `flat` on 2026-11-08); the nine other calibration genomes 6, 6, 6, 6, 6, 6, 6, 5, 5 (`0574db3e` and `3f8dd9b9` lose one frozen life each, `e1aaafa0` has 7 lives with snapshots less one frozen); six `uniform` explorer genomes with 1 life each (pooled only).
- So the one-genome bar is now reachable from committed data under the plan's option 1 (a `flat` outcome, no new births, no eligibility change): the centre has 41 ≥ 10 lives.
- **Probe set:** 256 frames from world seeds 20,000–20,015 (manifest sha256 `bf9ebe6b…`), 236 distinct periphery frames and 124 distinct fovea frames.

## Results

Columns: n, mean CKA (learned at 2,048 CU, init), ckaDelta, median matched correlation (learned, init), p (raw permutation p; "desc." = descriptive, the analysis is not evaluable), Holm p, the p floor, and the verdict (raw / Holm).

### Periphery (5×5 conv on the 50×80 frame)

| analysis | n | CKA L | CKA I | ckaDelta | matched L | matched I | p | Holm p | floor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **pooled** | 99 | 0.365 | 0.754 | −0.389 | 0.000 | 0.477 | 1.0000 | 1.0000 | 1e-4 | **not converged / not converged** |
| **centre `d22c1c55`** | 41 | 0.426 | 0.773 | −0.347 | 0.000 | 0.467 | 0.9998 | 1.0000 | 1e-4 | **not converged / not converged** |
| `008ce7c2` | 6 | 0.196 | 0.729 | −0.533 | 0.000 | 0.599 | 0.984 desc. | — | 0.0156 | not evaluable |
| `0574db3e` | 5 | 0.142 | 0.699 | −0.557 | 0.000 | 0.394 | 0.906 desc. | — | 0.0313 | not evaluable |
| `179bc010` | 6 | 0.251 | 0.762 | −0.511 | 0.000 | 0.440 | 1.000 desc. | — | 0.0156 | not evaluable |
| `1e95ea0c` | 6 | 0.715 | 0.661 | 0.054 | 0.000 | 0.309 | 0.422 desc. | — | 0.0156 | not evaluable |
| `3f8dd9b9` | 5 | 0.092 | 0.757 | −0.665 | 0.000 | 0.518 | 0.906 desc. | — | 0.0313 | not evaluable |
| `56e0b0f4` | 6 | 0.686 | 0.816 | −0.130 | 0.000 | 0.466 | 0.813 desc. | — | 0.0156 | not evaluable |
| `aac76dab` | 6 | 0.319 | 0.762 | −0.443 | 0.000 | 0.585 | 0.844 desc. | — | 0.0156 | not evaluable |
| `e1aaafa0` | 6 | 0.601 | 0.764 | −0.163 | 0.000 | 0.583 | 0.688 desc. | — | 0.0156 | not evaluable |
| `fd6a2683` | 6 | 0.213 | 0.683 | −0.470 | 0.000 | 0.437 | 0.906 desc. | — | 0.0156 | not evaluable |

### Fovea (3×3 conv on the 32×32 window)

| analysis | n | CKA L | CKA I | ckaDelta | matched L | matched I | p | Holm p | floor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **pooled** | 99 | 0.723 | 0.752 | −0.029 | 0.257 | 0.558 | 0.8973 | 1.0000 | 1e-4 | **not converged / not converged** |
| **centre `d22c1c55`** | 41 | 0.686 | 0.736 | −0.051 | 0.295 | 0.563 | 0.9341 | 1.0000 | 1e-4 | **not converged / not converged** |
| `008ce7c2` | 6 | 0.797 | 0.793 | 0.005 | 0.509 | 0.658 | 0.469 desc. | — | 0.0156 | not evaluable |
| `0574db3e` | 5 | 0.595 | 0.701 | −0.106 | 0.016 | 0.531 | 0.969 desc. | — | 0.0313 | not evaluable |
| `179bc010` | 6 | 0.846 | 0.700 | 0.146 | 0.508 | 0.533 | 0.0156 desc. | — | 0.0156 | not evaluable |
| `1e95ea0c` | 6 | 0.820 | 0.746 | 0.074 | 0.371 | 0.528 | 0.141 desc. | — | 0.0156 | not evaluable |
| `3f8dd9b9` | 5 | 0.700 | 0.846 | −0.147 | 0.015 | 0.397 | 0.781 desc. | — | 0.0313 | not evaluable |
| `56e0b0f4` | 6 | 0.847 | 0.799 | 0.047 | 0.041 | 0.636 | 0.234 desc. | — | 0.0156 | not evaluable |
| `aac76dab` | 6 | 0.605 | 0.852 | −0.246 | 0.302 | 0.550 | 0.859 desc. | — | 0.0156 | not evaluable |
| `e1aaafa0` | 6 | 0.716 | 0.648 | 0.069 | 0.078 | 0.401 | 0.359 desc. | — | 0.0156 | not evaluable |
| `fd6a2683` | 6 | 0.844 | 0.834 | 0.010 | 0.200 | 0.464 | 0.484 desc. | — | 0.0156 | not evaluable |

One small genome (`179bc010`, fovea, n = 6) would pass all three numbers descriptively (ckaDelta 0.146, matched 0.508, p at its floor 0.0156). It is one of 20 descriptive rows, n is below the spec's minimum, and it is not a verdict.

**Random filters already reach the matched bar on the fovea** (init matched 0.558 pooled, 0.563 centre ≥ 0.5) and nearly on the periphery (0.477). As the plan requires, the bar is not changed; it means the matched criterion on its own says little here, and ckaDelta carries the verdict.

### The between-genome control

Over the 10 genomes with ≥ 2 lives (93 lives): periphery within-genome 2,048 CKA 0.419 vs between 0.359, one-sided p 0.140; fovea within 0.696 vs between 0.731, p 0.918. Same-recipe lives are not detectably more alike than different-recipe lives either.

### Weight motion

Finite and non-zero for every life and level. Median ‖W − W₀‖/‖W₀‖ over the 99 lives: periphery 0.210 / 0.231 / 0.247 at 512 / 1,024 / 2,048 CU (range at 2,048: 0.100–0.394); fovea 0.225 / 0.268 / 0.319 (range 0.059–1.232). Centre lives at 2,048: periphery 0.241, fovea 0.319. Most motion happens before 512 CU. The filter sheets (`--filters-png`, one life per genome, init and 2,048) look like the random init with small changes: colour noise, no edges or blobs.

26 kept lives are flagged because an encoder's motion at 2,048 is not above its motion at 512 (14 periphery flags, 16 fovea flags; four lives have both): every one is listed in `report.json` `flags`. They stay in every analysis (decision 0027).

### What drives the drop: periphery units die

The flags also count units that are zero on every probe frame ("dead" on the probe set):

| | lives with any dead unit | median dead units (of 16) | mean | max |
|---|---|---|---|---|
| periphery, init | 66 of 99 | 1 | 1.35 | 5 |
| periphery, 2,048 CU | 98 of 99 | 8 | 8.29 | 16 (one life) |
| fovea, init | 47 of 99 | 0 | 0.64 | 4 |
| fovea, 2,048 CU | 73 of 99 | 2 | 2.13 | 12 |

By 2,048 CU half of a typical life's periphery units output zero on every probe frame. A dead unit's correlations are 0 (decision 0027), so with 8 or more dead units in either life the pair's median matched correlation is at most 0 — hence matched 0.000 on the periphery. CKA falls too because the surviving units are fewer and less alike than random ones. This is descriptive, not a new analysis; it suggests learning in v1 kills first-layer periphery units (a dying-ReLU pattern) rather than shaping them into shared detectors. Whether those units are also dead on the frames lives actually see (not only the probe's start-state frames) is not measured here.

## The identification check

Status `ok` (exit 0): every life's 512 CU filters correlate far more with its own recomputed init than with any other life's. min ownInit 0.886, max bestOther 0.120, smallest margin 0.810 — the same 0.810 that was measured while planning on the 38 lives then available.

## Real-artifact checks run (Task 7 Step 3)

- The probe ran twice on the same data; the two `report.json` files are byte-identical (`cmp`), as are the two `summary.md` files.
- Stage timings on this Mac (10 cores, single-threaded Node): render 0.1 s, read 127 lives 0.2 s, eligibility and init recompute 6 s, both Grams 567 s (second run; the first took 667 s while a CPU experiment ran beside it), analyses about 1 s. Maximum resident set 444 MB (`/usr/bin/time -l`, second run); no activations are kept.
- The life counts above were re-counted from the files by a separate scratch script and agree with the report (99 eligible; per-genome counts as listed).
- The 8 probe frame PNGs show the four rooms and differ from each other; the filter PNGs were opened (see Weight motion).

## Caveats

- **Node-rendered frames.** `@napi-rs/canvas` is not the browser's canvas; text pixels may differ from what tabs saw.
- **A probe set biased to start states:** 16 random steps from each room's start; 20 periphery and 132 fovea frames repeat. Units "dead on the probe set" may be alive elsewhere.
- **One machine** for the probe; lives ran on several hosts (all GPU tabs).
- **The bar is a guess** (spec §5.1); the matched criterion is already met by random fovea filters.
- **Multiplicity:** up to four verdict-bearing tests; raw p carries the bar, Holm reported beside it. Nothing is near significance, so this does not matter here.
- **Pooling methods in the centre analysis.** The centre's 41 lives are 6 calibration and 35 `centre` births: the same genome, role `search`, inheritance `none`, so the same training rule; only how the genome was drawn differs.

## Task 9: why three first layers froze

The plan found one life whose logged first layer never changed after 512 CU; at this commit there are three, all calibration lives on GPU tabs, all ended at budget: `lrn_4c7ec98889ef9cbb` (genome `3f8dd9b9`, code `369c5eb7`), `lrn_4d41b1c2f0e6eb7e` (`0574db3e`, code `249b7d03`) and `lrn_8f81e6c3dda13693` (`e1aaafa0`, code `249b7d03`). The same two code shas ran 106 other search lives at GPU widths whose filters did not freeze.

- **Their records (Step 1):** no restores, no skipped updates, no lost CU, one code sha per life (no reload between 512 and 1,024 CU), steps and CU rising normally, and the predictor's loss and held-out ratio still moving between files. So the rest of the model kept running while conv1 held bit-identical. (`lrn_4d41b1c2f0e6eb7e`'s held-out ratio is very poor, 7.6–39 from 512 CU on.)
- **Not a dead first layer:** at the frozen state 14–16 of 16 fovea units (1–10 periphery units) are active on the probe frames, so a fully dead conv1 does not explain a zero gradient.
- **CPU experiment (Step 2, a scratch run, not a committed test):** `lrn_4c7ec98889ef9cbb`'s seed and settings, GPU widths, CPU backend, 1,500 steps in its own world, `readParams()` every 50 steps. conv1 kept changing at every read (61–1,388 of 1,664 values per 50 steps), so on the CPU `readParams` is not stale and gradient reaches conv1 at this genome. But the number of values that change falls over the life, and 233 values never changed after birth (88 for a centre life run the same way): weights of units that die stop moving. A complete freeze was not reproduced within 1,500 CPU steps (the frozen lives froze between about 5,000 and 14,000 GPU steps).
- **Open:** whether the freeze is a GPU-only fault (a stale `readParams` on WebGPU, or a gradient that stops reaching conv1, e.g. dead units downstream with Adam's moments then decaying to exact zero) needs a WebGPU probe in Playwright or a checkpoint of a frozen life; neither was done here. The failing-first package tests the plan asks for were not written, because nothing failed on the CPU. The probe's rule (exclude bit-identical repeats) is unaffected; any fix belongs to its own plan.

## What follows (spec §5.1)

- Filters do not converge, so by the spec the inherited-filter condition is **not** the next step. The spec's next candidates are **a distilled frame set**, then **a shared replay or curriculum file**. Decision 0028 (proposed) drafts the §5.2 condition as the plan asks and recommends "not now".
- The periphery's dying units look like a learner question worth its own look before any filter sharing is reconsidered.
- The one-genome options in the plan are moot: the centre already has 41 eligible lives.

## Deviations from the plan

- **Merge to main and CI (Task 7 Step 1) not done:** the run read `data/runs` at `a30716f` on branch `plan-05c`, with the probe fixed at `1045693`; merging, pushing and CI are left to the controller.
- **More frozen lives than planned:** three, not one. The rule fixed in Task 2 excluded them unchanged.
- **`Verdict.pDescriptive`** was added: the plan makes `p` null for a non-evaluable analysis but also asks for its numbers as description; `pDescriptive` keeps the permutation p.
- **`analyse` takes the excluded list** as a fifth argument (the plan's signature had four) so the report lists exclusions; **`Identification.bestOther`/`bestOtherId`** are null when there is only one life.
- **Hungarian tie rule:** a free column beats an assigned one, then the lower index (the plan said "lowest column, then lowest row"); this is what makes the all-equal case give the identity in this algorithm (decision 0027).
- **Flags** also count dead units per state (the plan's Review Focus 4 asks dead units to be counted in the report; `flags` is where they are).
- **Widths:** eligibility checks the full `GPU_WIDTHS`, not only `conv1`, because the recomputed init depends on every width; the reason is `widths conv1 <c>` when conv1 differs.
- **Fixture helper** `fixture-lives.ts` was created in Task 2 (needed by its tests), not Task 6.
- **Task 9 Step 2** was a scratch CPU experiment, not committed failing-first tests (nothing failed).
- Runtime: about 10 min per run for 99 lives (the plan estimated a few minutes at 37; the Gram cost grows with the square of the lives).
