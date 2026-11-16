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

### Which lives count [agent's call, not yet confirmed; following spec §5.1 "fresh search lives"]

- **Eligible:** every file has a meta block with `role: "search"` and `inheritance.filters: "none"`; GPU widths (`conv1` 16, and the rest of `GPU_WIDTHS`, since the birth state depends on every width) in every snapshot file; one snapshot each at 512, 1,024 and 2,048 CU (by `levelAtOrBelow(cu)`, the lowest-seq file at a level); an integer `settings.seed`, the same in every file; no lower learnerId among the candidates with the same seed; and a first layer that changed between every pair of consecutive states.
- **Excluded with the first failing reason,** in this order: `no meta`, `role <r>`, `inheritance <x>`, `widths conv1 <c>` (or `widths <c1>/<c2>/<enc>/<hidden>` when only a later width differs), `no snapshot at <level>`, `no seed`, `seed shared with <learnerId>`, `filters frozen after <state>`. Every excluded life is listed in the report.
- **Shared seeds.** Two lives with one learner seed start from bit-identical first layers; their init pair would inflate the init control, and the identification check could not tell them apart. Only the lowest learnerId per seed is kept, as decision 0024 does for confirmation lives that share an `evalSeed`.
- **Frozen filters.** The states in order are init (recomputed), 512, 1,024 and 2,048. If any state's first layer (all four arrays) is bit-identical to the one before it, the life is excluded with `filters frozen after <the earlier state>`. A float32 layer trained with a non-zero learning rate does not repeat itself bit for bit over thousands of steps; a repeat means no gradient reached conv1 or the snapshot was a stale read, and either way the logged 2,048 state is not a 2,048 CU learned state. Only bit-identical repeats exclude: slow but non-zero motion is a finding, kept, and flagged in the report when periphery motion does not grow from 512 to 2,048. Plan 05c found one such life (`lrn_4c7ec98889ef9cbb`, `filters frozen after 512`); at the time of the first run there are three (the research note lists them).
- **Weight motion** is ‖(w, b) − (w₀, b₀)‖ / ‖(w₀, b₀)‖ per encoder, weights and biases together, in float64, at 512, 1,024 and 2,048 CU.
- **The birth state** is recomputed with `createModel(cpu, settings.seed, GPU_WIDTHS)`'s `FIRST_LAYER_PARAMS`, and tested bit for bit against a newborn learner's `readParams()`.

### Which activations [agent's call, not yet confirmed]

- **Post-ReLU conv1 output,** which is what the next layer receives, recomputed in float64 from each state's float32 weights. Pre-ReLU values are never computed, so there is no second set of numbers to choose between later.
- **Every (frame, output position) is one sample:** 256 × 1,000 for the periphery, 256 × 256 for the fovea.
- **Activations at init and 2,048 CU only,** as the spec compares; the 512 and 1,024 snapshots get weight motion only (and 512 the identification check, below).
- **A streamed two-pass centred Gram** over every (life, state, unit) column: pass 1 takes means, pass 2 accumulates centred cross-products; the activations themselves are never stored. A column that takes one value on every sample (a dead unit) is flagged, and its Gram row and column are exactly 0.

### The metrics [agent's call, not yet confirmed]

- **Linear CKA** between two states' 16-unit blocks of the centred Gram: ‖K_ab‖²_F / (‖K_aa‖_F · ‖K_bb‖_F) [KUH K27]; null (the pair skipped and counted) if a state's block is all zero.
- **Signed matching.** Units are matched one to one by Hungarian matching on their signed correlation, not its absolute value: a ReLU unit's negative is a different detector. The matched value of a pair is the median of the 16 matched correlations (the mean of the 8th and 9th); over pairs, the median again (spec gap 4).
- **Dead units** (one value on every probe sample) have correlation 0 with everything and are matched like any other unit.
- **Ties.** Hungarian (our own Kuhn–Munkres with potentials, on −c, float64) adds rows in order; when two columns tie in a shortest-path step, a free column beats an assigned one, then the lower column index wins. An all-equal matrix gives the identity. (The plan said "the lowest column index, then the lowest row"; a free-column-first rule is what makes the all-equal case give the identity in this algorithm, so it is the rule implemented.)

### The analyses, the controls and the bar [agent's call, not yet confirmed]

- **Per-encoder verdicts, no combined verdict.** The periphery and the fovea are different layers on different inputs, so one can converge while the other does not; each gets its own verdict, and neither is folded into an "overall" answer. The §5.2 draft's gate ("converged for at least the periphery") is a use of these verdicts and part of this call: the periphery sees the whole frame, so it is the first candidate for a shared start.
- **Analyses:** "pooled" (every eligible life, across genomes) and one per genome with ≥ 2 lives. **The centre's is the primary one-genome analysis;** any other genome reaching 10 lives is reported beside it, not instead of it.
- **Statistics over the pairs (i < j):** learned = the mean CKA at 2,048 and the median over pairs of the matched median at 2,048; init = the same on the same pairs at init; `ckaDelta` = learned mean CKA − init mean CKA. A pair with any null CKA is skipped and counted.
- **The permutation test (within-life label swap).** Under the null, learning does not change how alike lives are, so which of a life's two states is "learned" is exchangeable. T(s) is `ckaDelta` with life i's states swapped where s_i = 1. Exact over all 2ⁿ relabellings when n ≤ 16 (p counts s = 0; floor 2⁻ⁿ); otherwise 9,999 Monte Carlo relabellings from `createRng(hashSeed(0, "filter-probe/perm/<name>/<encoder>"))`, p = (1 + hits) / 10,000.
- **The bar** (spec §5.1, a guess; `BAR` in `tools/probe/filters/analyse.ts`, pinned by a test): an analysis with n < 10 lives is **not evaluable** (reasons `n = <n> < 10 lives`, plus `p cannot reach 0.05 (floor <f>)` when the floor is ≥ 0.05), with `converged` and `p` null and its numbers shown as description (`pDescriptive` keeps the permutation p). Otherwise it **converged** iff ckaDelta ≥ 0.1, matched ≥ 0.5 and p < 0.05. The init-pair matched value is always reported beside it; if random filters already reach 0.5 the note says so and the bar is not changed.
- **Multiplicity.** Raw p carries the bar (the spec gives one bar and one p < 0.05). A Holm-adjusted p over each analysis's two encoders is reported beside it, with the verdict it would give; the note states both whenever they differ.
- **The between-genome (recipe) control,** reported and never part of the bar: over lives whose genome has ≥ 2 lives, `within` is the mean 2,048 CKA of same-genome pairs and `between` that of different-genome pairs; p is one-sided, from 9,999 shuffles of the genome labels (group sizes kept), seed `"filter-probe/recipe/<encoder>"`. It tells "everyone converges" apart from "the same recipe converges".
- **Flags,** never exclusions: a kept life whose motion at 2,048 is not above its motion at 512 (per encoder), and any state with units that are dead on the probe set (counted).
- **The identification check** (the spec does not ask for it). For each life, `ownInit` is the Pearson correlation, over all 1,664 first-layer floats (periphery w, b, fovea w, b; `FIRST_LAYER_PARAMS` order; biases are drawn from the seed too), of its 512 CU state with its own recomputed init; `bestOther` is the largest such correlation with any other eligible life's init. **It fails if any life has `ownInit ≤ bestOther`:** a seed or decoding bug, not a finding. The report is still written, with `status` `"identification failed"`, every verdict withheld (not evaluable, `converged` and `convergedHolm` null, reason `identification failed for <k> lives`) and every number kept as description; the CLI exits 2.
- **No constant here changes after the first real run** without a new recorded decision; both results are then kept.

## Alternatives considered

Each section above names what it rejected.

## Consequences

The probe's result is reported favourable or not (`docs/research/`). Nothing here changes what a learner sees or does.
