# Plan 05c — The filter-convergence probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms
Decision 0018 says that learned material many lives acquire should move up the hierarchy, and that the first candidate is the learner's **first-layer filters**: the 16 small pixel detectors at the front of each eye (periphery and fovea). Before anything is shared, we have to know whether separate lives actually learn the same filters. If every fresh life ends up with roughly the same detectors, a shared starting set might save later lives work. If they don't, sharing filters makes no sense, and the next candidates are shared data instead.

This plan builds the measurement and runs it. It builds nothing that a learner sees:
- **A fixed set of probe pictures.** 256 frames, rendered by the world in Node from 16 world seeds that no life ever lives in (20,000–20,015), under seeded random actions. A committed manifest fixes every frame.
- **What each life's first layer does on those pictures.** Since Plan 05a, every life logs its first-layer filters at 512, 1,024 and 2,048 CU (`meta.filters`). The probe also recomputes each life's filters at birth from its logged seed. It runs both through the same first layer on the probe frames.
- **How alike two lives are.** Three numbers per pair of lives: linear CKA (a whole-layer similarity score), the median correlation after pairing each unit with its best partner (Hungarian matching), and how far the weights moved from birth.
- **Against the right baselines.** Random filters already look somewhat alike on the same pictures, so learned pairs are compared with the same lives' birth filters. Pairs within one recipe are compared with pairs across recipes, to separate "everyone converges" from "same recipe, same result".
- **A bar fixed before the run, and a write-up either way.** The spec's bar (a guess) is coded once, pinned by a test, and evaluated only where there is enough data. The result goes in a research note, favourable or not.
- **The inherited-filters condition is drafted, not built.** The last task writes it up as a proposed decision for the user.

**Goal:** ship spec §10 phase 05c: `tools/probe/filter-convergence.ts` over the logged snapshots, run on the committed lives, and the write-up; plus the §5.2 inherited-filter condition drafted as a decision for the user, not built.

**Architecture:**
- **Everything is in `tools/probe/`, offline.** No file in `packages/`, `app/`, `knowledge/`, `config/`, `data/` or `.github/` changes.
  - `tools/probe/filters/probe-set.ts`: renders the probe frames and checks them against the committed manifest `tools/probe/filter-probe-set.json`.
  - `tools/probe/filters/lives.ts`: reads committed lives (`tools/eval/lives.ts` `collectLives`), decides eligibility, decodes `meta.filters`, recomputes the init filters, and measures weight motion.
  - `tools/probe/filters/layer.ts`: the first layer (conv + ReLU) in float64, and the streamed centred Gram matrix over all (life, state) columns.
  - `tools/probe/filters/metrics.ts`: linear CKA, unit correlation, `hungarianMax`, matched-correlation medians.
  - `tools/probe/filters/analyse.ts`: the analyses, controls, permutation tests and the pinned bar.
  - `tools/probe/filters/png.ts`: filter sheets as PNGs, for looking at (Task 7), never part of the analysis.
  - `tools/probe/filter-convergence.ts`: the core (`runProbe`) and the CLI around it.
- **Reads:** `data/runs/**` (committed logs), `knowledge/genome.json` (which genome is the centre). **Writes:** `.local/filter-probe/` only. The research note and its summary file are written by hand in Task 7.
- **Imports:** `@blanktab/learner` (`createModel`, `GPU_WIDTHS`, `FIRST_LAYER_PARAMS`, `obsToInputs`), `@blanktab/tensor` (CPU backend, only to recompute init and in tests), `@blanktab/world` (`createWorld`, `ROOM_KINDS`), `@blanktab/harness/log` (`FILTER_FLOATS_PER_CONV1`, `levelAtOrBelow`, log types), `@blanktab/harness/heldout` (`uniformAction`), `@blanktab/detmath` (`hashSeed`, `createRng`), `@blanktab/meta` (`genomeSha`), `tools/gate/serialise.ts`. Nothing imports `tools/probe/`.

**Tech stack:** TypeScript, Vitest, ESLint flat config, `tsx`, `@napi-rs/canvas` (already a dependency of `packages/world` and `packages/harness`). No new npm dependency.

**Spec:** `docs/specs/2026-10-22-meta-learner-v1-design.md` §5.1, §5.2, §10 "05c" (and §1 goal 5). Decision 0018 is binding (user's): "the first artifact is first-layer filters, gated by a convergence probe and always run beside fresh and matched-control conditions". Decisions 0020–0024 are in force; this plan changes none of them. Literature: [KUH] `docs/research/2026-10-22-lit-knowledge-up-the-hierarchy.md` K26 (convergent learning, unit matching), K27 (CKA), §"Concrete first candidate".

**Planned against `main` at `7749ae6`.** `data/meta/posterior.json` is mode `calibrate`, with 34 eligible lives. Look 1 has answered "no signal yet" (p 0.059 against α 0.0167), so calibration continues to 6 repeats, with looks 2 and 3 at 45 and 60 lives.

## Why 05c can be built now
05c does not depend on the signal check's answer:
- **Its input already exists.** 05a logs `meta.filters` at 512, 1,024 and 2,048 CU for every life with a meta block, in every mode. Calibration lives have them now.
- **It changes nothing the meta-learner reads.** No birth, posterior, gate, evaluator, log format or learner code changes. Its verdict feeds no automatic decision; only the user acts on it (Task 8).
- **The answer changes only which lives come later.** Under `flat`, 80 % of births run the centre. Under `search`, births spread over many genomes, and the centre is drawn mainly in confirmation arms. It is also drawn, with role `search`, in two other cases (`packages/meta/src/draw.ts`):
  - **CPU births,** about 80 % of them in every mode. They fail this plan's GPU-width rule (Task 2), so they never count.
  - **GPU search births that fall back.** When the `ts` path has no GPU model, a model of other genes, no GP posterior or no Thompson pick, `tsDraw` returns the centre with a reason. These are fresh GPU search lives and would count. With a healthy model they should be rare.

  That matters for the one-genome bar (next section), not for building or running the probe.

## The data on `main` today (counted from the committed files at `7749ae6`)
Every life below is `role: "search"`, `method: "calibrate"`, inheritance `none`, GPU widths (`conv1` 16), and has a filter snapshot at each of 512, 1,024 and 2,048 CU. Genomes are numbered by their position in `data/meta/calibration.json`; genome 0 is the centre (`d22c1c55…`, the same sha as `knowledge/genome.json`).

| Calibration genome | 0 (centre) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lives with all three snapshots | 4 | 4 | 4 | 2 | 3 | 4 | 2 | 4 | 4 | 3 | **34** |

- **Not counted:** `lrn_c7bb5490d6c71b57` (genome 1, still running: snapshots at 512 and 1,024 only); `lrn_0adb1fb0fabb8f49` and `lrn_245f90f55bad891a` (genome 2, ended or stalled at 32 CU); one `legacy` life (failed at 268 CU); 17 pre-05a lives without a meta block. None of these has a 2,048 snapshot.
- **One of the 34 has a frozen first layer.** `lrn_4c7ec98889ef9cbb` (genome 1, `3f8dd9b9…`) logs the same `meta.filters` string, byte for byte, in `6.json` (515.7 CU, 5,000 steps), `7.json` (1,031.5 CU, 10,000 steps) and `8.json` (2,048.0 CU, 19,906 steps). Its genome has no setting that would hold conv1 still (lr 0.00136, headLrScale 0.35). It is the only life under `data/runs` with repeated snapshots. Its 2,048 snapshot is really a 512 state, or a stale read, so Task 2 excludes it by a rule fixed now, before any run (`filters frozen after 512`). That leaves **33 eligible lives** (genome 1: 3). Why it froze is looked at separately (Task 9); the answer does not change the probe.
- **The across-genome bar can be evaluated today:** 33 lives ≥ 10.
- **The one-genome bar cannot, and calibration alone will never reach it.** The most lives any one genome has is 4. The calibration design stops at 6 repeats per genome (60 lives, look 3), and 6 < 10. With fewer than 5 lives the within-life permutation test cannot even reach p < 0.05: its smallest possible p is 2⁻ⁿ, and 2⁻⁴ = 0.0625.
- **This plan does not lower the bar.** The probe reports the one-genome analysis per genome as "not evaluable (n < 10)", with its numbers shown as description only. How to get ≥ 10 lives of one genome is a question for the user, with options below.

### Open for the user: getting ≥ 10 lives of one genome
None of these is built by this plan. The agent's recommendation is marked.
1. **Wait for a `flat` outcome (no new code).** If look 3 answers "flat", 80 % of GPU births run the centre. The centre will have 6 lives with snapshots once calibration ends, so about 4 more centre lives (about 5 births, a few hours of runner time) reach 10. **[agent's call, not yet confirmed: the default.]**
2. **If the check answers "signal", use confirmation centre-arm lives.** These are fresh lives of the centre genome with inheritance `none`, so their training is the same as a search life's; only their checkup worlds and learner seed rule differ. The spec says "fresh search lives", so counting them needs the user's word, and two more things:
   - **An eligibility change, recorded as a decision.** Task 2 excludes `role confirm` by design, so this option is a code change to `probeLives` as well as a ruling.
   - **One life per learner seed.** A confirmation life's learner seed is its `evalSeed` (decision 0024, Confirmation seeds), and two centre-arm lives can share one. Their birth filters are then bit-identical: their init-pair CKA is 1, which inflates the init control, and the identification check (Task 5) would refuse the whole report. The rule would be 0024's: the lowest learnerId per `evalSeed`. Task 2's shared-seed exclusion already applies it to whatever lives are eligible.

   Also counted under `search`, without any change: the GPU fallback births of the centre ("Why 05c can be built now"). They are role `search` and eligible.
3. **A centre-only probe batch now.** About 6 more centre lives, run through a new birth mode like the newborn mode (a role that is never a row). This needs new code in the page, the log parser and the gate, and a recorded decision. It costs about 3 h of runner time on one machine, taken from calibration, which would delay looks 2 and 3. Choose this only if the within-genome answer is wanted before calibration ends.
- **Not options:** judging the one-genome bar at n = 4–6; pooling genomes and calling it "one genome"; changing calibration's repeats mid-run (invariant 7: hold the design fixed within a run).

## Global Constraints
- **Earlier plans still bind.** That covers the walls, the purity lint on `packages/{detmath,meta,evaluators}/src/**`, the log and checkpoint invariants, and decisions 0020–0024.
- **AGENTS.md invariant 2: harness-only.** Nothing the probe computes, renders or writes reaches a learner. The probe frames, the manifest, the activations and the report live in `tools/probe/` and `.local/`. Nothing in `packages/` or `app/` imports `tools/probe/`, and a test enforces this (Task 1). The probe does not go in `knowledge/` (which tabs fetch) or `data/` (which the gate writes).
- **AGENTS.md invariant 1.** No life is born with inherited filters in this plan. The §5.2 condition is a proposed decision only (Task 8).
- **AGENTS.md invariants 7 and 8: fixed before it runs.** The probe set, eligibility, metrics, controls, permutation scheme and the bar are fixed and committed (Tasks 1–6) **before** the probe runs on real lives (Task 7).
  - After the first real run, changing any of them needs a new recorded decision. Both results are then kept in the note.
  - The bar's constants live in one object, pinned by a test to the spec's values.
  - The note reports the result whether or not it is favourable, including "not evaluable".
- **Offline and read-only.** The tool reads committed files and writes only under `.local/filter-probe/`. It never writes `data/`, `knowledge/`, `config/` or the manifest (the manifest is written only by its own `--write-manifest` step in Task 1, reviewed and committed by hand).
- **Determinism.**
  - All metric arithmetic is float64: the first layer, Gram sums, CKA, correlations, Hungarian, medians and permutation statistics.
  - Fixed orders: frames in manifest order, positions row-major, lives sorted by learnerId, states as (init, 2048) per life, pairs (i < j).
  - Seeds come only from `hashSeed` with fixed salts (`"filter-probe/…"`). There is no `Math.random`, and no clock in any output.
  - Output is `serialise`d (sorted keys, one-space indent, final newline). Two runs give identical bytes (a test, and Task 7 on real data).
- **No silent exclusions.** Every committed life the probe does not use is listed in the report with its reason.
- **Agent's calls are labelled.** Every choice this plan makes for the user goes in one decision file, called **the probe-calls decision** in this plan, marked not yet confirmed. The §5.2 draft is a separate decision file, called **the inherited-filter draft** (Task 8).
  - **Numbers are assigned when each file is created,** not here: the next free number in `docs/decisions/` on the branch at that moment, after merging `main`. At `7749ae6` the last is 0024, and the unmerged `wf/gate-cadence` adds 0025, so the two are likely 0026 (`NNNN-filter-probe-calls.md`) and 0027 (`NNNN-inherited-filter-condition.md`). Never reuse a number another branch has taken.
  - Once a file exists, cite it by its number in the code comments, the research note and the plans README.
- **Do not edit `docs/open-questions.md`** in this plan. A user decision on Task 8's draft updates it later.
- **Hygiene and gates.** As plan 05a: no trailing whitespace and no machine paths in any file, docs included. Every task runs `npm test && npm run typecheck && npm run lint`. **After any push**, open the CI run and check it is green.

## Review Focus
1. **A wrong birth state makes the whole probe meaningless.** Examples: the wrong seed, the wrong widths, the wrong parameter indices, or decoding the floats in the wrong order. Guards:
   - the recomputed init equals a newborn learner's step-0 parameters bit for bit (Task 2);
   - the decoded snapshot re-encodes to the logged string (Task 2);
   - on real data, every life's 512 CU filters are closer to its own recomputed init than to any other life's (Task 7).
2. **Choices made after seeing the data.** The order of tasks is the guard: everything is fixed, tested and committed before Task 7's first real run. Any later change is a recorded decision, and both results are kept. (Tasks 5, 7)
3. **Probe frames that a life has seen, or that leak to a life.**
   - A test checks that no committed life (under `data/runs` and `data/newborns`) has a training world, held-out world or checkup variant with a seed in 20,000–20,015.
   - A test checks that nothing in `packages/` or `app/` imports `tools/probe/`.

   (Task 1)
4. **Degenerate units and ties.** A ReLU unit that is zero on every probe frame has no correlation. A correlation matrix can have tied entries. Each case gives a defined, deterministic result: a dead unit's correlations are 0, and it is counted in the report. Hungarian breaks ties by the lowest index. (Tasks 3, 4)
5. **Too few lives.** An analysis with n < 10 says "not evaluable" and why, and has no verdict. It never passes or fails on fewer lives than the spec's minimum. (Tasks 5, 7)

---

### Task 1: The probe set, its manifest, and the guards

**Files:**
- Create: `tools/probe/filters/probe-set.ts`, `tools/probe/filter-probe-set.json` (generated, then committed)
- Create: `tools/probe/filter-convergence.ts` with only `--write-manifest` and `--frames-png` (Task 6 adds the run)
- Test: `tools/probe/test/probe-set.test.ts`, `tools/probe/test/isolation.test.ts`
- Create: the probe-calls decision, `docs/decisions/NNNN-filter-probe-calls.md` (the next free number; Global Constraints), from `0000-template.md`. Later tasks add sections.

**Interfaces:**
- Produces:
  ```ts
  export const PROBE_SEEDS: readonly number[]          // 20000..20015 (spec §5.1)
  export const PROBE_STEPS = 16                         // frames per seed
  export interface ProbeFrameEntry { seed: number; step: number; action: Action; periphery: string; fovea: string } // sha256 hex of the bytes
  export interface ProbeManifest { schemaVersion: 1; seeds: number[]; steps: number; startRoom: "seed mod 4"; frames: ProbeFrameEntry[]; sha256: string }
  export function buildManifest(): Promise<ProbeManifest>                   // renders, draws the actions; used only by --write-manifest
  export function renderProbeSet(m: ProbeManifest): Promise<Observation[]>  // replays m's actions; throws on any frame sha mismatch
  export function writeFramePngs(frames: readonly Observation[], m: ProbeManifest, dir: string): Promise<string[]>
    // the periphery and fovea of 8 frames (two per start room, the first two of each room's first seed) as PNGs; returns the paths
  ```
- **The frames.**
  - For each seed s in 20,000–20,015: `createWorld({ seed: s, canvas: createCanvas(320, 200), start: ROOM_KINDS[s % 4] })`.
  - The actions are `uniformAction(createRng(hashSeed(s, "filter-probe/actions")))` (`@blanktab/harness/heldout`, the held-out set's rule).
  - For steps 1..16: `act`, `drainTrace`, `observe`. Each frame is the observation after step k, so there are 256 frames.
  - The manifest records every action, so later runs replay them rather than redrawing them. `sha256` covers all frames' bytes in order.
- **The start room** [agent's call]: `ROOM_KINDS[seed mod 4]`, so the 16 seeds cover each of the four rooms four times. With the default start, all 256 frames would come from one room, since 16 random steps rarely leave it.
- **Where the manifest lives** [agent's call]: `tools/probe/`, not `knowledge/` (tabs fetch that) or `data/` (the gate writes that).
- **Frames are Node renders.** `@napi-rs/canvas` is not the browser's canvas, so text pixels may differ from what tabs see. Pinning the sha makes the set fixed; a canvas upgrade that changes a frame makes the tool refuse to run, and regenerating the manifest is then a recorded decision.

- [ ] **Step 1: Write the failing tests**
  - `probe-set.test.ts`:
    - `"the committed manifest has 16 seeds 20000–20015, 16 steps each, 256 frames, and every start room four times"`.
    - `"renderProbeSet reproduces every committed frame sha (256 frames)"`.
    - `"a manifest with one frame sha changed makes renderProbeSet throw, naming the seed and step"`.
    - `"buildManifest run twice gives identical bytes"`.
    - `"no committed life has seen a probe world: for every life under data/runs and data/newborns, its training world seed (parseInt of learnerId hex 4–12, the page's seedsOf rule), hashSeed(learnerSeed, 'heldout'), and its checkup variant seeds lie outside 20000–20015"`.
      - The variant seeds come from `variantSeedOf(meta)` (`packages/harness/src/checkup.ts`: `panelSeed` for a panel reference, `confirmSeed` for a confirmation life or newborn) and from `checkupSeed(learnerId, …)` for a life without a panel or `evalSeed`. That is the same rule the page and `runNewborn` use, so newborn references are covered by their own rule.
      - It scans both trees: spec §5.1 says the probe frames are "never seen by any life", and newborn references act in worlds too (their held-out set and checkups).
      - It reads `data/runs` and `data/newborns`, which only grow. The property must hold for every life, so it is not a pinned fixture.
  - `isolation.test.ts`: `"no file under packages/ or app/ imports tools/probe (static or dynamic)"`, scanning `trackedFiles()`.
- [ ] **Step 2: Run them; they fail.** `npx vitest run tools/probe/test`. Expected: FAIL (no module).
- [ ] **Step 3: Implement**, then write the manifest once: `npx tsx tools/probe/filter-convergence.ts --write-manifest`. Then `npx tsx tools/probe/filter-convergence.ts --frames-png .local/filter-probe/frames` (it calls `writeFramePngs` and exits; Task 6 keeps the flag). Open the 8 PNGs, and check that they show the four rooms and differ from each other.
- [ ] **Step 4: Run them; they pass.** Create the probe-calls decision and start it with: the probe frames (start room rule, frames after each step, Node rendering) and where the manifest lives.
- [ ] **Step 5: Commit** `feat(probe): the filter probe set: 256 frames from reserved world seeds, fixed by a committed manifest; isolation guards`

### Task 2: Lives, their first-layer filters, and the birth state

**Files:**
- Create: `tools/probe/filters/lives.ts`
- Test: `tools/probe/test/lives.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export type Encoder = "periphery" | "fovea"
  export interface FirstLayer { periphery: { w: Float32Array; b: Float32Array }; fovea: { w: Float32Array; b: Float32Array } } // w [16, 75] / [16, 27], b [16]
  export interface ProbeLife {
    learnerId: string; genomeSha: string; seed: number
    init: FirstLayer; at: { 512: FirstLayer; 1024: FirstLayer; 2048: FirstLayer }
  }
  export interface Excluded { learnerId: string; reason: string }
  export function probeLives(lives: readonly LifeInput[]): { lives: ProbeLife[]; excluded: Excluded[] } // sorted by learnerId
  export function decodeFilters(base64: string, conv1: number): FirstLayer          // parameter order: pw, pb, fw, fb
  export function initFirstLayer(seed: number, widths: Widths): FirstLayer         // createModel(cpu, seed, widths), FIRST_LAYER_PARAMS
  export function weightMotion(at: FirstLayer, init: FirstLayer, e: Encoder): number // ‖(w,b) − (w₀,b₀)‖ / ‖(w₀,b₀)‖, float64
  ```
- **Eligible** [agent's call, following spec §5.1 "fresh search lives"]: every file has a meta block with `role: "search"` and `inheritance.filters: "none"`; GPU widths (`conv1` 16) in every snapshot file; one snapshot each at 512, 1,024 and 2,048 (by `levelAtOrBelow(cu)`); an integer `settings.seed`; no lower learnerId among the candidates with the same `settings.seed`; and a first layer that moved between every pair of consecutive states. Otherwise the life is excluded, with the first failing reason, in this order: `no meta`, `role <r>`, `inheritance <x>`, `widths conv1 <c>`, `no snapshot at <level>`, `no seed`, `seed shared with <learnerId>`, `filters frozen after <state>`.
- **Shared seeds** [agent's call]. Two lives with one learner seed start from bit-identical first layers. Their init pair would inflate the init control, and the identification check (Task 5) could not tell them apart. So only the lowest learnerId per seed is kept, as decision 0024 does for confirmation lives that share an `evalSeed`. No search life at `7749ae6` shares a seed; the rule is fixed now so that a later one fails with a clear reason, not an identification abort.
- **Frozen filters** [agent's call; fixed before any run, Review Focus 2]. The states in order are init (recomputed), 512, 1,024 and 2,048. If any state's `FirstLayer` (all four arrays) is bit-identical to the one before it, the life is excluded with `filters frozen after <the earlier state>`.
  - **Why exclude, not keep and flag.** A float32 first layer trained with a non-zero learning rate does not repeat itself bit for bit over 5,000 steps. A repeat means either no gradient reached conv1 or the snapshot was a stale read. In both cases the logged 2,048 state is not a 2,048 CU learned state, which is what the probe compares.
  - **Only bit-identical repeats.** Slow but non-zero motion is a finding, not a fault, so it is kept: a kept life whose weight motion does not grow from 512 to 2,048 is listed in the report's `flags` (Task 5), never excluded. A threshold on "near zero" would be one more guessed constant, and bit identity needs none.
  - **At `7749ae6` it excludes one life:** `lrn_4c7ec98889ef9cbb`, `filters frozen after 512` (its 512, 1,024 and 2,048 snapshots are one string). Why it froze is Task 9's question.
- **Files are read through `collectLives`** (`tools/eval/lives.ts`), so every file has passed `parseLogFile`, including its `meta.filters` length check.
- **Weight motion** covers weights and biases together, per encoder [agent's call].

- [ ] **Step 1: Write the failing tests**
  - `"initFirstLayer(seed, GPU_WIDTHS) equals FIRST_LAYER_PARAMS of a newborn learner's readParams() before any step (createLearner on the CPU backend), bit for bit, for 3 seeds"` (spec §5.1).
  - `"decodeFilters of a real committed 2048 snapshot re-encodes (float32 LE, base64) to the same string, and has 16·104 values"`. The test picks the first eligible life under `data/runs` at run time; it does not pin an id.
  - `"probeLives on fixture lives: a legacy life, a life without meta, a CPU-width life, a running life with no 2048 snapshot, and a confirm-role life are each excluded with their reason; an eligible life is kept"`.
  - `"probeLives excludes the higher learnerId of two lives with the same settings.seed as 'seed shared with <lower id>', and keeps the lower"`.
  - `"probeLives excludes a life whose 1024 and 2048 snapshots are the same string as 'filters frozen after 1024', one whose 512 snapshot equals its recomputed init as 'filters frozen after init', and keeps a life whose snapshots differ by one ulp in one weight"`.
  - `"probeLives on the committed lives excludes lrn_4c7ec98889ef9cbb as 'filters frozen after 512'"`. It is skipped, and says so, if that life's files are ever absent; it pins the one known case, not a count.
  - `"weightMotion is 0 at init and 1 for w = 2·w₀, b = 2·b₀"`.
- [ ] **Step 2: Run them; they fail.** `npx vitest run tools/probe/test/lives.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add to the probe-calls decision: eligibility, the exclusion order, the shared-seed rule, the frozen-filters rule (exclude bit-identical repeats, flag slow motion; with the one life it excludes at `7749ae6`), and weight motion over weights and biases.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(probe): eligible lives, decoded filter snapshots and the recomputed birth state`

### Task 3: The first layer in float64, and the streamed Gram matrix

**Files:**
- Create: `tools/probe/filters/layer.ts`
- Test: `tools/probe/test/layer.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface LayerShape { k: number; stride: number; pad: number; inH: number; inW: number; outH: number; outW: number }
  export const LAYER: Record<Encoder, LayerShape>  // periphery 5×5 s2 p2: 50×80 → 25×40; fovea 3×3 s2 p1: 32×32 → 16×16 (model.ts)
  export function firstLayer(x: Float32Array, f: { w: Float32Array; b: Float32Array }, s: LayerShape): Float64Array
    // relu(conv), [16, outH·outW], float64; input CHW 0..1 from obsToInputs; weight index order = the tensor package's im2col
  export interface Gram { cols: number; g: Float64Array; means: Float64Array; varZero: Uint8Array; samples: number }
  export function centredGram(frames: readonly Float32Array[], states: readonly { w: Float32Array; b: Float32Array }[], s: LayerShape): Gram
  ```
- **Which activation** [agent's call]: the post-ReLU output of conv1, which is what the next layer receives. Pre-ReLU values are not computed, so no second set of numbers is available to choose between later.
- **The samples.** Every (frame, output position) pair is one sample: 256 × 1,000 = 256,000 for the periphery and 256 × 256 = 65,536 for the fovea.
- **The Gram.**
  - The columns are 16 units × every (life, state) with state ∈ {init, 2048}. At 33 lives that is 1,056 columns, a 1,056² float64 matrix of about 8.9 MB.
  - **Two passes over the frames.** Pass 1 computes every column's mean. Pass 2 accumulates the centred products (upper triangle, then mirrored).
  - Only the Gram is kept, never the activations: those would be about 2 GB.
  - A column whose variance is exactly 0 (a dead unit) is flagged in `varZero`.
  - 512 and 1,024 snapshots get weight motion only. Activations are compared at init and 2,048, as the spec says.
- **Cost [estimate]:** about 1.5e11 multiply-adds for the periphery at 33 lives, so a few minutes in Node. Task 7 measures it.

- [ ] **Step 1: Write the failing tests**
  - `"firstLayer equals relu(conv2d(...)) from the CPU backend on 3 seeded inputs and filters, for both encoders, to 1e-5 absolute"` (the weight order must be the learner's).
  - `"LAYER's output sizes equal model.ts's (25×40 and 16×16)"`.
  - `"centredGram equals a naive two-pass float64 covariance on a 4-frame, 3-state fixture to 1e-12 relative"`.
  - `"a state whose bias is −100 has every unit flagged varZero"`.
  - `"centredGram run twice gives identical bytes"`.
- [ ] **Step 2: Run them; they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Add to the probe-calls decision: post-ReLU activations, and activations at init and 2,048 only.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(probe): the first layer in float64 and a streamed centred Gram over all lives and states`

### Task 4: The metrics: linear CKA, unit correlation, Hungarian matching

**Files:**
- Create: `tools/probe/filters/metrics.ts`
- Test: `tools/probe/test/metrics.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export function cka(G: Gram, a: number, b: number): number | null      // states a, b (column blocks of 16)
    // ‖K_ab‖²_F / (‖K_aa‖_F · ‖K_bb‖_F), K = the centred cross-product blocks [KUH K27]; null if a block is all zero
  export function unitCorrelation(G: Gram, a: number, b: number): Float64Array // 16×16; 0 where either unit is varZero
  export function hungarianMax(c: Float64Array, n: number): Int32Array    // row i → column, maximising Σ c[i, col(i)]
  export function matchedMedian(G: Gram, a: number, b: number): number    // median of the 16 matched correlations [KUH K26]
  ```
- **Hungarian** is our own O(n³) Kuhn–Munkres with potentials, on −c, in float64, with no dependency. Ties go to the lowest column index, then the lowest row.
- **Signed correlation** [agent's call]: it matches on signed correlation, not absolute. A ReLU unit's negative is a different detector.
- **The median of 16** is the mean of the 8th and 9th sorted values.

- [ ] **Step 1: Write the failing tests**
  - `"cka(X, X) = 1; cka is unchanged when one state's units are permuted, rotated by a seeded orthogonal 16×16 matrix, or scaled by 3"`.
  - `"cka of two independent seeded Gaussian blocks on 10,000 samples is below 0.05"`.
  - `"hungarianMax equals brute force over all permutations on 200 seeded matrices, n = 2..7, to 1e-12 in the matched sum"`.
  - `"hungarianMax on an all-equal matrix returns the identity; on a permuted identity it returns that permutation"`.
  - `"matchedMedian of a state against a unit-permuted copy of itself is 1"`.
  - `"a dead unit's correlations are 0 and it is matched like any other unit"`.
- [ ] **Step 2: Run them; they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Add to the probe-calls decision: signed matching, dead units at 0, and the tie rule.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(probe): linear CKA, unit correlation and deterministic Hungarian matching in float64`

### Task 5: The analyses, the controls, the permutation tests and the bar

**Files:**
- Create: `tools/probe/filters/analyse.ts`
- Test: `tools/probe/test/analyse.test.ts`

**Interfaces:**
- Produces:
  ```ts
  /** Spec §5.1's bar, a [guess]. Changing any value needs a recorded decision (invariant 7); a test pins them. */
  export const BAR = Object.freeze({ minLives: 10, ckaDelta: 0.1, matched: 0.5, alpha: 0.05 })
  export const PERMUTATIONS = 9_999            // Monte Carlo draws when exact enumeration exceeds 2^16
  export interface PairStats { learned: { cka: number; matched: number }; init: { cka: number; matched: number }; n: number; pairs: number; skipped: number }
  export interface Verdict {
    evaluable: boolean; reasons: string[]; converged: boolean | null; ckaDelta: number; matched: number; p: number | null; pFloor: number
    pHolm: number | null; convergedHolm: boolean | null   // Holm over the analysis's two encoders; reported, never the bar
  }
  export interface Analysis { name: string; encoder: Encoder; learnerIds: string[]; stats: PairStats; verdict: Verdict }
  export interface ProbeReport {
    schemaVersion: 1; manifestSha: string; centreSha: string; lives: number; excluded: Excluded[]
    convergence: Analysis[]                       // "pooled" and "genome:<sha>" (every genome with ≥ 2 lives), per encoder
    recipe: { encoder: Encoder; within: number; between: number; p: number; genomes: number }[]  // the between-genome control
    motion: { learnerId: string; genomeSha: string; periphery: number[]; fovea: number[] }[]     // at 512, 1024, 2048
    flags: { learnerId: string; flag: string }[]  // kept lives worth a look, e.g. "periphery motion did not grow 512→2048"
    identification: { learnerId: string; ownInit: number; bestOther: number }[]                  // Review Focus 1
  }
  export function analyse(lives: ProbeLife[], grams: Record<Encoder, Gram>, centreSha: string, manifestSha: string): ProbeReport
  export function permutationP(t: (s: Uint8Array) => number, n: number, how: "exact" | "monteCarlo", seed: number): { p: number; floor: number }
    // analyse uses "exact" when n ≤ 16, else "monteCarlo"
  ```
- **The convergence analyses.** Run for both encoders, each separately, and each with its own verdict [agent's call; the spec gives one bar and does not say per encoder]:
  - **"pooled"** (across genomes): every eligible life.
  - **One per genome** with ≥ 2 lives.
  - **The primary one-genome analysis is the centre's** [agent's call]. Other genomes reaching 10 are reported beside it, not instead of it.
  - **No combined verdict** [agent's call]. The periphery and fovea are different layers on different inputs, so one can converge while the other does not. The report and the note give each encoder's verdict on its own line; neither is folded into an "overall" answer. Task 8's gate ("converged for at least the periphery") is a use of these verdicts, and is part of the same call: the periphery sees the whole frame, so it is the first candidate for a shared start.
- **Statistics, over the pairs (i < j) of an analysis's lives.**
  - The learned value is the mean CKA at 2,048 and the median over pairs of `matchedMedian` at 2,048.
  - The init value is the same, on the same pairs at init.
  - `ckaDelta` = learned mean CKA − init mean CKA.
  - Pairs with a null CKA are skipped and counted.
- **The permutation test (within-life label swap)** [agent's call; spec: "a permutation test over lives"].
  - Under the null, learning does not change how alike lives are. So for each life, which of its two states is called "learned" is exchangeable.
  - A relabelling s ∈ {0,1}ⁿ swaps life i's states where s_i = 1. T(s) is `ckaDelta` recomputed with the swapped labels, from the four precomputed CKAs of each pair (init–init, init–2048, 2048–init, 2048–2048).
  - **Exact when n ≤ 16:** p = #{s : T(s) ≥ T(0)} / 2ⁿ, counting s = 0.
  - **Otherwise Monte Carlo:** p = (1 + #{k : T(s_k) ≥ T(0)}) / (1 + 9,999), with s_k drawn from `createRng(hashSeed(0, "filter-probe/perm/" + name + "/" + encoder))`.
  - `pFloor` is 2⁻ⁿ (exact) or 1/10,000.
- **The verdict.**
  - **Not evaluable** if n < `BAR.minLives`. Its reasons are `"n = <n> < 10 lives"`, plus `"p cannot reach 0.05 (floor <pFloor>)"` when pFloor ≥ 0.05. Then `converged` and `p` are null; the numbers are still shown, labelled descriptive.
  - **Otherwise converged** iff ckaDelta ≥ 0.1, matched ≥ 0.5 and p < 0.05.
  - The init-pair matched value is always reported beside it. If random filters already reach 0.5, the note says so, and the bar is not changed.
- **The between-genome control** [agent's call on the statistic]. Over lives whose genome has ≥ 2 lives:
  - `within` is the mean 2,048 CKA of same-genome pairs, and `between` that of different-genome pairs.
  - p is one-sided, from 9,999 seeded shuffles of the genome labels (group sizes kept), seed `"filter-probe/recipe/" + encoder`.
  - It is reported, never part of the bar. It tells "everyone converges" apart from "the same recipe converges".
- **Identification** (Review Focus 1). For each life: `ownInit` is the correlation of its flattened 512 CU weights with its own recomputed init, and `bestOther` is the largest correlation with any other life's init. The CLI refuses to write a report if any life has `ownInit ≤ bestOther`: that would mean a seed or decoding bug, not a finding. If a failure turns out to be genuine weight motion rather than a bug, relaxing the check is a recorded decision, not a code edit.
- **Multiplicity.** There are up to four verdict-bearing tests: two encoders × (pooled, centre). Raw p is used for the bar [agent's call; the spec says "p < 0.05" for one bar and does not address multiplicity, since it does not split by encoder]. A Holm-adjusted p over the two encoders of each analysis is reported beside it, with the verdict that p would give. The note states both verdicts whenever they differ.

- [ ] **Step 1: Write the failing tests**
  - `"BAR is exactly { minLives 10, ckaDelta 0.1, matched 0.5, alpha 0.05 } (spec §5.1)"`.
  - `"12 planted lives whose 2048 filters are one shared filter set plus small seeded noise, units permuted per life: pooled is evaluable and converged, matched near 1, p = exact floor"`.
  - `"12 planted lives whose 2048 filters are fresh random draws: evaluable, not converged"`.
  - `"4 lives of one genome: not evaluable, reasons 'n = 4 < 10 lives' and 'p cannot reach 0.05 (floor 0.0625)', converged null"`.
  - `"permutationP exact and monteCarlo agree within 0.01 on the same 14-life fixture"`.
  - `"the recipe control finds within > between on planted genome-specific filters, and p ≥ 0.05 when filters ignore the genome"`.
  - `"identification: a life whose seed is swapped with another's is caught (ownInit ≤ bestOther)"`.
  - `"a kept life whose periphery motion at 2048 is not above its motion at 512 is listed in flags, and stays in every analysis"`.
  - `"each verdict carries its Holm-adjusted p and the verdict at that p, beside the raw one"`.
  - `"analyse run twice gives identical serialised bytes"`.
- [ ] **Step 2: Run them; they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Add to the probe-calls decision: per-encoder verdicts with no combined verdict (and that Task 8's periphery gate rests on it), the centre as the primary one-genome analysis, the permutation scheme, the recipe control's statistic, raw p for the bar with the Holm verdict reported beside it, the `flags` list, and the rule that no constant may change after the first real run without a new decision.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(probe): convergence analyses, init and between-genome controls, permutation tests and the pinned bar`

### Task 6: The CLI and the golden test

**Files:**
- Modify: `tools/probe/filter-convergence.ts` (Task 1 created it)
- Create: `tools/probe/filters/png.ts`, `tools/probe/test/fixtures/filter-probe-golden.json`
- Test: `tools/probe/test/filter-convergence.test.ts`, `tools/probe/test/fixture-lives.ts` (a test helper, not a test file)

**Interfaces:**
- Produces:
  ```ts
  // tools/probe/filter-convergence.ts: one core that the CLI and the tests both call; main only gathers its inputs
  export interface ProbeInputs { frames: readonly Observation[]; manifestSha: string; lives: readonly LifeInput[]; centreSha: string }
  export class ProbeRefusal extends Error { readonly why: "identification" | "no eligible lives" }
  export function runProbe(inp: ProbeInputs): { report: ProbeReport; summary: string }
    // probeLives → both Grams → analyse → summary.md text; throws ProbeRefusal; writes nothing
  export function writeProbe(out: { report: ProbeReport; summary: string }, dir: string): void  // report.json (serialise), summary.md
  export function main(argv: readonly string[]): Promise<number>  // the exit code; the file's entry point is process.exit(await main(...))
  // tools/probe/filters/png.ts
  export function writeFilterPngs(life: ProbeLife, states: readonly ("init" | 512 | 1024 | 2048)[], dir: string): Promise<string[]>
    // the 16 periphery filters (5×5 RGB, each scaled to its own min..max) per state, one 4×4 sheet per state; returns the paths
  ```
- **Usage:**
  - `npx tsx tools/probe/filter-convergence.ts [--runs data/runs] [--out .local/filter-probe] [--manifest tools/probe/filter-probe-set.json] [--genome knowledge/genome.json]`. The defaults are shown; the tests point `--manifest` and `--out` at temp files.
  - `npx tsx tools/probe/filter-convergence.ts --write-manifest` (Task 1 only).
  - `npx tsx tools/probe/filter-convergence.ts --frames-png <dir>` (Task 1 Step 3): renders the probe set, calls `writeFramePngs`, and exits.
  - `npx tsx tools/probe/filter-convergence.ts --filters-png <dir>` (Task 7 Step 3): reads the lives, calls `writeFilterPngs` at init and 2,048 for the lowest eligible learnerId of each genome, and exits.
  - The PNG flags write only under `<dir>`, never `report.json`, and are not part of the analysis.
- **What `main` does** for a run.
  1. Loads `--manifest` and renders the probe set. `renderProbeSet` throws on any frame mismatch.
  2. Reads the lives with `collectLives(--runs)`, and gets `centreSha` from `--genome` through `genomeSha`.
  3. Calls `runProbe`, then `writeProbe` into `--out`. Nothing is written before `runProbe` returns.
  4. Prints the stage timings to stdout only, never into a file.
- **Exit codes.** `main` returns 1 and writes nothing on a manifest mismatch (the `renderProbeSet` throw), a `ProbeRefusal` (a failed identification or zero eligible lives), or a bad flag. Any other throw is a bug and is not caught.
- **The fixture lives** (`fixture-lives.ts`), so that synthetic lives take the same path as real ones:
  - `fixtureLife({ learnerId, genomeSha, seed, noise })` builds a `LifeInput` whose files pass `parseLogFile`: role `search`, inheritance `none`, GPU widths, and snapshots at 512, 1,024 and 2,048 CU. A test asserts that every fixture file parses.
  - Each snapshot is `initFirstLayer(seed, GPU_WIDTHS)` plus seeded noise from `createRng(hashSeed(seed, "filter-probe/fixture/" + level))`, growing with the level. So identification passes, motion grows, and no two states are bit-identical.
  - The 3 genome shas are fixed fixture strings, `centreSha` is the first of them, and `manifestSha` is a fixed fixture string. Nothing is read from `knowledge/` or `data/`.
  - The 8 frames are seeded synthetic `Observation`s (bytes from `createRng(hashSeed(0, "filter-probe/fixture-frames"))`), so no canvas is needed.
- **The golden fixture.** 3 genomes × 4 lives through `runProbe`, in seconds. `writeProbe`'s `report.json` is compared byte for byte with `fixtures/filter-probe-golden.json`. Regenerate with `FILTER_PROBE_WRITE=1`, and review the diff by eye.

- [ ] **Step 1: Write the failing tests**
  - `"every fixture life's files pass parseLogFile, and probeLives keeps all 12"`.
  - `"the fixture run's report.json equals filter-probe-golden.json byte for byte"`.
  - `"two fixture runs write identical report.json and summary.md bytes"`.
  - `"runProbe throws ProbeRefusal('identification') when two fixture lives' snapshots are swapped, and ProbeRefusal('no eligible lives') on lives that are all excluded"`.
  - `"main with --manifest pointing at a temp copy with one frame sha changed returns 1 and leaves --out empty"`. It renders frames up to the first mismatch, so it needs the canvas, as Task 1's tests do.
  - `"main returns 1 and writes nothing when runProbe refuses"`. The refusal is reached through `--runs` pointing at a temp dir of fixture lives written as files.
  - `"the run writes only under --out"`. Compare `git status --porcelain` before and after a fixture run of `main` against a temp `--out`.
  - `"writeFilterPngs writes one 4×4 sheet per state, and the same bytes twice"`.
- [ ] **Step 2: Run them; they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Generate the golden file, and read it: are the planted genome effects visible in `recipe`, and are all four-life genomes "not evaluable"?
- [ ] **Step 4: Run the full gates.** `npm test && npm run typecheck && npm run lint`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(probe): tools/probe/filter-convergence.ts with a golden report`. This commit fixes the probe. Everything after it runs on real data (Global Constraints).

### Task 7: Run it on the committed lives, check it against the real artifacts, and write it up

- [ ] **Step 1: Merge Tasks 1–6 to `main`, and check that CI is green.** Merge `main` into the branch first. If another branch has meanwhile taken the probe-calls decision's number, renumber the file and every citation of it in that merge (Global Constraints). Then merge the branch to `main` and push. Open the CI run on the pushed commit and wait for green. The probe is fixed from this commit on; record its sha.
- [ ] **Step 2: Run it on `main`.** Wrap the run in `caffeinate -i`. Run `npx tsx tools/probe/filter-convergence.ts` twice, then `cmp` the two `report.json` files: they must be identical. Record the stage timings, and the commit sha whose `data/runs` was read.
- [ ] **Step 3: Check it against the artifacts** (Review Focus 1, invariant 8). Each item is checked by running something, not by reading code:
  - **The life counts** equal a fresh count from the files: 33 eligible lives at `7749ae6` (the table above, less `lrn_4c7ec98889ef9cbb`), or more if more have merged since. List each excluded life and its reason.
  - **Identification:** every life has `ownInit > bestOther`. Report the smallest margin.
  - **Weight motion** is finite and non-zero at every level. Every life in the report's `flags` is named in the note; the rule that keeps them was fixed in Task 2, so nothing is excluded here.
  - **Look at the filters.** Run `npx tsx tools/probe/filter-convergence.ts --filters-png .local/filter-probe/filters` (one life per genome, at init and at 2,048), and open the PNGs. Do they have structure (edges, colour blobs) or only noise?
- [ ] **Step 4: Write `docs/research/2026-10-2x-filter-convergence.md`.** Keep it short:
  - **The question.**
  - **The data:** the lives, the per-genome counts and the exclusions.
  - **A table per encoder:** pooled and centre, plus each genome as description. Columns: n, learned and init CKA, ckaDelta, learned and init matched, p, Holm p, pFloor, verdict, and the verdict under Holm.
  - **The recipe control and weight motion.**
  - **The answer, favourable or not: one line per encoder,** with no combined verdict (Task 5). For example: "Across genomes, periphery filters converge / do not converge by the spec's bar; fovea filters converge / do not converge; the one-genome bar is not evaluable (centre n = 4)". Where the Holm verdict differs from the raw one, the line says so.
  - **Caveats:** Node-rendered frames; a probe set biased to start states; one machine; the bar is a guess; the multiplicity.
  - **What follows**, per spec §5.1. If the filters do not converge, the next candidates are a distilled frame set, then a shared replay or curriculum file. If they converge across genomes, the §5.2 condition goes to the user (Task 8). In both cases, the one-genome options above go to the user.

  Commit `report.json` itself, byte for byte, beside the note, as `docs/research/2026-10-2x-filter-convergence/report.json`. It holds no per-pair arrays: per-analysis aggregates, and per-life rows for `excluded`, `motion`, `flags` and `identification`, so it stays small. It can be rebuilt from the committed logs at the recorded sha.
- [ ] **Step 5: Set row 05c in `docs/plans/README.md`** to "done" with a link to the note, and its one-line result.
- [ ] **Step 6: Commit, push and check that CI is green.** Message: `docs: Plan 05c — the filter-convergence probe on <n> committed lives`.

### Task 8: Draft the §5.2 inherited-filter condition as a decision for the user (proposed, not accepted)

**Files:**
- Create: the inherited-filter draft, `docs/decisions/NNNN-inherited-filter-condition.md` (the next free number on `main` when it is written; Global Constraints), from `0000-template.md`
- **Status:** `proposed`. **Decided by:** `agent proposal awaiting the user's decision`. Nothing in it is built or switched on.

**Contents** (spec §5.2, written as the decision the user would be taking):
- **Context.**
  - Decision 0018: the first artifact is first-layer filters, gated by a convergence probe and always run beside fresh and matched-control conditions.
  - The probe's result (Task 7), favourable or not, with its numbers.
  - Invariant 1: an inherited start is a different condition from a fresh life, and is labelled.
- **The gate on building it** [agent's call]. It is built only if the pooled verdict is "converged" for at least the periphery encoder. The one-genome verdict should also be converged, or the user accepts building without it, with a recorded reason. If neither holds, the decision says "not now", and the spec's next candidates are named.
- **The arms.**
  - **F (fresh):** the default; at least 40 % of births [guess].
  - **I (inherited):** loads `knowledge/filters-gpu-v1.json` at birth. The filters stay trainable.
  - **C (matched control):** random filters with each filter's norm and bias matched to the artifact's.

  GPU widths only; CPU lives are always F.
- **Building the artifact.** Generation 0 aligns each eligible life's 2,048 CU filters to a reference life (the lowest learnerId of the centre) by `hungarianMax` on activation correlation over the probe set, then averages them. Later generations take a Reptile step toward the mean end state of the I arm. The artifact records the learnerIds, the probe manifest sha and the population CU it took.
- **Labels and the code that would change** (none of it changed by this plan):
  - `meta.inheritance.filters` ∈ {`none`, `<sha>`, `control:<sha>`}. Today `packages/harness/src/log.ts` accepts only `"none"`, and `search-composite@1` and `confirm-fisher@1` count only `"none"`. So I and C lives are never search rows or confirmation lives unless a new evaluator version says otherwise.
  - The artifact joins `knowledgeSha`.
  - The GP treats the condition as fixed context, never as a gene. Confirmations never pool conditions.
  - The learner's checkpoint format is unchanged: the filters set only the initial values.
- **The readouts:** CU to the M1 bar; the held-out ratio at 512 and 1,024 CU on unseen variants; no worse than F at a long budget (the warm-start check, [KUH] K16); and the slope after a novel room (the plasticity check, K17). The population CU spent building the artifact is reported beside any gain (invariant 5).
- **Alternatives:** whole-network checkpoints (Lamarckian; last, per [KUH]); a distilled frame set or a shared replay or curriculum file (no shared init; the spec's next candidates if filters do not converge); doing nothing.
- **Questions for the user**, each with the agent's suggestion:
  1. Build only on a converged pooled verdict, or also require the one-genome one?
  2. The F share (≥ 40 %) and the I:C split (1:1)?
  3. Periphery and fovea together, or only an encoder that converged?
  4. Is a per-filter norm match enough for C, or should C also match the artifact's spatial frequency spectrum?

- [ ] **Step 1: Write the draft.** Link it from the research note's "What follows" section.
- [ ] **Step 2: Check it.** It contradicts no accepted decision (0018–0024, and any accepted since). Every agent's call in it is labelled. It names the code paths that would change, and changes none of them:
  - `git diff --stat` and `git diff --stat --cached` before the commit, then `git show --stat HEAD` after it, list only the draft and the research note's link.
  - `git diff --stat <the Task 7 Step 1 merge commit> HEAD -- packages app knowledge config .github` is empty: nothing the learner, page, gate or knowledge reads has changed since the probe was fixed.
- [ ] **Step 3: Run the hygiene tests:** `npx vitest run tools/no-trailing-whitespace.test.ts tools/no-machine-paths.test.ts tools/tracked.test.ts`. Expected: PASS.
- [ ] **Step 4: Commit** `docs: draft decision <NNNN>, the inherited-filter condition (proposed, for the user)`, then push and check that CI is green.

### Task 9: Separate from the probe — why `lrn_4c7ec98889ef9cbb`'s first layer froze

The probe's rule for this life is fixed in Task 2 and does not wait on this task. This task asks whether the freeze is a learner or snapshot fault that could touch other lives, or other parameters. It may run any time after Task 2 is committed, beside Tasks 3–8. It changes no probe code.

- [ ] **Step 1: Read the life's own record.** For files 1–9, list `codeSha`, `hostSig`, `stepsTotal`, `cu`, `restores`, `skippedUpdates`, `lostCU` and the curve's predLoss. Did the rest of the model keep learning while conv1 held still? Did a restore or a code change fall between 512 and 1,024 CU?
- [ ] **Step 2: Test the two candidate mechanisms on the CPU backend,** each as a failing-first test in the package it concerns:
  - **A stale read.** `readParams()` after k steps differs from `readParams()` at step 0, in `FIRST_LAYER_PARAMS`, for the genome's values (`packages/learner`). Also that `filtersAt` in `packages/harness/src/session.ts` reads the learner live at each crossing, and does not reuse an earlier result.
  - **No gradient to conv1.** One step with this genome's settings gives a non-zero conv1 gradient (`packages/learner`).

  If neither fails on the CPU, say so: the life ran on a GPU tab, and a GPU-only fault needs a WebGPU probe in Playwright, recorded as such.
- [ ] **Step 3: Write it up** as a short section in the research note (Task 7), favourable or not: the cause if found, or what was ruled out. A fault found here is fixed under its own plan or decision, not in this one. Any fix changes future lives only; the committed logs and the probe's exclusion stay as they are.

**After this plan.**
- **When a genome reaches ≥ 10 eligible lives** (by whichever option the user picks above), re-run the probe unchanged. Add a dated section to the note with the one-genome verdict, and keep the first result.
  - **If the inherited-filter draft is still `proposed`,** amend its context with the new result.
  - **If the user has decided it,** do not edit it: decisions are append-only (AGENTS.md). Write a new decision that cites the new result and, if it changes the outcome, supersedes the old one.
- **The decision on the inherited-filter draft is the user's.** Building the I and C arms is a later plan, and only on an accepted decision.

---

## Spec conflicts and gaps found while planning (resolutions adopted above; each goes into the probe-calls decision as an agent's call)
1. **"≥ 10 lives of one genome" is out of reach of the calibration design** (at most 6 per genome). Resolution: no relaxation. The one-genome analysis is reported as "not evaluable", and getting the data is a question for the user, with options.
2. **The permutation test is unspecified.** Resolution: a within-life swap of init and 2,048 labels, exact up to n = 16. It cannot reach p < 0.05 below n = 5, which is one more reason to keep the spec's minimum of 10.
3. **"Activations"** could be before or after the ReLU. Resolution: after, and only that is computed.
4. **"The median matched correlation"** has two levels of median. Resolution: the median over each pair's 16 matched units, then the median over pairs.
5. **A probe set of 16 random steps from the default start** would all be one room. Resolution: start room = seed mod 4.
6. **Memory.** The activations of 68 states would be about 2 GB. Resolution: a streamed two-pass centred Gram; every metric comes from it.
7. **512 and 1,024 snapshots.** The spec compares at init and 2,048. Resolution: activations at those two only; weight motion at all three levels.
8. **The reserved world seeds 20,000–20,015 and confirmation learner seeds.** Confirmation id 10 would use learner seeds 20,000–20,999. Those are learner seeds, not world seeds (a confirmation life's world is its learnerId's: plan 05b conflict 9; decision 0024, Confirmation seeds). Resolution: the guard test checks the world seeds every committed life actually used, under `data/runs` and `data/newborns`.
9. **One bar, two encoders.** The spec gives one "converged" bar and does not say whether it applies per encoder, nor how many tests carry it. Resolution: a verdict per encoder, with no combined verdict; raw p for the bar, with the Holm-adjusted p and its verdict reported beside it. Task 8's periphery gate is part of the same call.
10. **A life whose first layer stops changing** (`lrn_4c7ec98889ef9cbb`, found while planning). The spec says nothing about it. Resolution: bit-identical consecutive states exclude the life, with the reason; slow motion is flagged and kept. Fixed before any run, so the choice is not made after seeing results.
11. **Lives that share a learner seed** start from the same filters. Resolution: the lowest learnerId per seed, as decision 0024 does for confirmation lives.

## Self-review
- **Spec coverage, §5.1 and §10 05c:**
  - the probe set, its seeds and the manifest → Task 1;
  - per-life activations at init and 2,048, with init recomputed from the seed and tested against a newborn → Tasks 2–3;
  - linear CKA, Hungarian median matched correlation and weight motion → Tasks 2, 4;
  - the init-pair and between-genome controls → Task 5;
  - the bar over ≥ 10 lives of one genome, and separately across genomes → Task 5, applied in Task 7;
  - the write-up, favourable or not → Task 7;
  - the §5.2 condition drafted as a decision, not built → Task 8;
  - the one frozen life: its rule → Task 2, its cause → Task 9 (outside the spec).
- **Invariants:** 1 (nothing inherited; Task 8 is a draft); 2 (harness-only; isolation test, Task 1); 5 (Task 8 asks for build cost beside gain); 7 (fixed before the run, pinned bar, no relaxation); 8 (identification, artifact checks, note either way).
- **Names across tasks:** `PROBE_SEEDS`, `ProbeManifest`, `renderProbeSet`, `writeFramePngs`; `ProbeLife`, `FirstLayer`, `probeLives`, `decodeFilters`, `initFirstLayer`, `weightMotion`; `LAYER`, `firstLayer`, `Gram`, `centredGram`; `cka`, `unitCorrelation`, `hungarianMax`, `matchedMedian`; `BAR`, `Verdict`, `Analysis`, `ProbeReport`, `analyse`; `ProbeInputs`, `ProbeRefusal`, `runProbe`, `writeProbe`, `main`, `writeFilterPngs`.
- **Order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Tasks 3 and 4 can run side by side after Task 2. Task 7 must not start until Task 6 is committed; its Step 1 merges Tasks 1–6 to `main`. Task 9 runs any time after Task 2 and blocks nothing.
- **Proportion:** signatures, constants and test names only. Detail is given only where the tests do not fix it: the frames, eligibility, the permutation scheme, the verdict rules and the identification check.
