# Plan 05b — Meta-learner search

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms
Plan 05a gave the meta-learner everything except its brain. Every new tab draws a recipe (a **genome**) by a seeded rule the gate can redo, every life is scored, and a noise check asks whether the score can tell recipes apart at all. This plan builds the brain, and it builds it **switched off**:
- **A model of which recipes work.** Once an hour the gate fits a Gaussian process (a smooth guess of "score as a function of recipe", with its uncertainty) to every scored life. The fit is plain arithmetic in our own code, so it gives the same bytes on every machine and every run.
- **Tabs draw from the model.** One birth in five stays a random explorer, forever. Most of the rest draw a "Thompson sample": the tab imagines one plausible version of the model and runs the recipe that version likes best. Recipes that look good get tried more, and recipes nobody has tried still get a chance.
- **A winner must prove itself on fresh worlds.** When the model is 95 % sure some recipe beats the current one, the gate declares a **confirmation**. Lives of that recipe, of the current recipe, and untrained newborns of that recipe run on seeds no search life ever saw. Only if the candidate beats newborn luck on milestones **and** beats the current recipe does the gate promote it. It then writes `knowledge/genome.json` itself, and new tabs are born from the new recipe.
- **Switched off until the noise check says "signal".** The model is fitted, served and sampled only when the meta-learner's mode is `search`. Only the 05a noise check can set that mode, from merged lives. Until it does, the deployed 05b code behaves exactly like 05a. If the check answers "flat", the code stays switched off and the next step is new genes (the spec's v2 mechanisms), not this search.

**Goal:** ship spec §10 phase 05b, built and deployed but inert until the signal check answers "signal": the deterministic GP fit in the gate, `posterior.json` schemaVersion 2 with the model, Thompson-sampling and confirmation draws in the tab, re-verification of every draw, the incumbent and trigger, confirmation declaration, serving and decision through the pinned `confirm-fisher@1`, automatic promotion of `knowledge/genome.json`, and the page lines.

**Architecture:**
- **`packages/meta`** (imports only `@blanktab/detmath`) gains the maths: `linalg.ts` (Cholesky with a fixed jitter schedule), `gp.ts` (Matérn-5/2 ARD kernel, MAP objective and gradient, prediction), `lbfgs.ts`, `fit.ts` (multi-start fit), `thompson.ts` (candidates, the sample, the incumbent). `posterior.ts` reads schemaVersion 1 and 2; `draw.ts` adds the `ts` and `confirm` branches.
- **The gate** (`tools/gate/meta.ts`) calls two new modules after re-verification: `search.ts` (rows, the fit, the incumbent, the trigger) and `confirm.ts` (declaration, need, decision, promotion). `history.ts` reads earlier versions of the knowledge files from git. `commit-meta.ts` widens its path set to the new gate files and `knowledge/genome.json`.
- **The harness** gains the confirmation-life rules: the frozen confirmation-newborn file, confirmation checkup variants, the arm-from-seed rule, and per-schema genome checks.
- **The page** draws `ts`/`confirm` in the worker, runs confirmation newborns, and shows the search, confirmation and promotion lines.

**Tech stack:** TypeScript, Vitest, ESLint flat config (the purity bans of plan 05a apply to all new `packages/meta` files), `@blanktab/*` workspaces, Playwright (Chromium in CI; WebKit and Firefox for the local determinism probe), GitHub Actions (`GITHUB_TOKEN`).

**Spec:** `docs/specs/2026-10-22-meta-learner-v1-design.md`: §4.1, §4.3–§4.5, §6, §7, §10 "05b". Decisions 0018 and 0019 are binding (user's). 0020–0023 are agent's calls in force; this plan must not contradict them without a new decision.

**Planned against `main` at `730f905`.** The signal check has not answered yet: `data/meta/posterior.json` is mode `calibrate`, and the newborn reference is pinned (0023).

## Global Constraints
- **Earlier plans still bind.** This covers the walls, the purity lint on `packages/{detmath,meta,evaluators}/src/**`, the log and checkpoint invariants, and every 05a review ruling recorded in 0020 and 0022.
- **Inert until signal (the user's condition).** Built now, active only after the noise check finds signal:
  - `buildMeta` computes `models`, `incumbent` and `confirmation`, writes confirmation or promotion files, and touches `knowledge/genome.json`, only when `mode === "search"`.
  - `mode` is `modeOf(calibration)`, and `calibration.outcome` becomes `"signal"` only in `nextLook`, from the pinned signal evaluator's `score === 1`. No other code path sets it, and no flag, environment variable or option can. Tests and the simulation reach `search` only through data (planted calibration lives), never through a code switch.
  - `parsePosterior` returns null for any posterior that carries a model, an incumbent or a confirmation outside mode `search`. A tab then draws `centre`.
  - `birthDraw` reaches `ts` and `confirm` only for mode `search` on `gpu`.
- **"Flat" leaves it inert.** Under `flat` (or `calibrate`), births stay `calibrate`/`uniform`/`centre`. The gate fits nothing, so it spends no fit time. No confirmation or promotion can exist. Leaving `flat` needs a recorded decision (a new genome schema or evaluator version), which restarts calibration (spec §4.6). 05b then switches on by itself only if that new calibration finds signal.
- **AGENTS.md invariant 2.** The learner never sees its genome, its draw, the model, its score or its confirmation arm. A confirmation life differs from a search life only in what the harness sets: its learner seed and its checkup variant seeds.
- **AGENTS.md invariant 5.** This is research-agent design work. Promotions are the first learner-population-discovered change, and each one is recorded with the confirmation that justified it.
- **AGENTS.md invariant 7.** Promotion uses only the pinned `config.confirm` (`confirm-fisher@1`, both tests). The gate never writes `config/`. Unfavourable results (fail, incomplete, flat) are appended and shown.
- **Determinism.** All GP and sampler arithmetic is `+ − × ÷`, `Math.sqrt` and `@blanktab/detmath` (`dexp`, `dlog`, `normCdf`, `normInv`). Loops run in a fixed order, with fixed seeds and iteration caps. No `Math.exp`, `Math.log`, `Math.pow`, `**` or `Math.random` appear in `packages/meta/src`. The gate's files are `serialise`d (sorted keys, one-space indent, final newline).
- **Nothing ends a forever life.** `LEARNER_STATE_VERSION` stays **2**, log `schemaVersion` stays 2 and `meta.v` stays 1. Every file under `data/runs/` and `data/newborns/` still parses and validates.
- **The posterior's schemaVersion becomes 2.** 05b code reads 1 and 2 and draws exactly as 05a from a v1 file. 05a code (strict `schemaVersion !== 1 → null`) draws `centre` from a v2 file until its tab reloads onto new code (0014). Ruling, 05a Task 4.
- **The gate's commits** contain only gate-computed files matching the widened `META_PATH_RE`. They are never forced, and they respect the lockfile guard. `knowledge/genome.json` appears only with a passing confirmation's result in the same commit.
- **Agent's calls are labelled.** Every choice this plan makes for the user goes in `docs/decisions/0024-meta-learner-search-calls.md`, marked not yet confirmed.
- **Hygiene, dependencies, gates.** As plan 05a. No new npm dependency. Playwright's WebKit and Firefox browser binaries may be installed locally (`npx playwright install webkit firefox`); they are not package dependencies. Every task runs `npm test && npm run typecheck && npm run lint`, plus `npm run e2e` when `app/` changed. **After any push**, open the CI run and check it is green.

## Review Focus
1. **Mixed code and posterior versions during a deploy.** Each pairing must behave as follows:
   - a 05a tab served a v2 posterior draws `centre` and never throws;
   - a 05b tab served the last v1 posterior draws exactly what 05a drew;
   - the gate re-verifies lives born under both versions.

   (Tasks 1, 5, 8)
2. **A model where there should be none.** For example: a hand-edited `calibration.json` that claims "signal", a v2 posterior carrying `models.gpu` in mode `calibrate`, or a `paused`/`flat` posterior with a confirmation. The guard is the same each time: no `ts` or `confirm` draw ever happens, the gate refuses an inconsistent calibration, and a tab treats such a posterior as unreadable. (Tasks 1, 5, 8)
3. **A promotion while tabs are open.** A tab still holding the old `genome.json` meets a posterior built for the new centre. It draws the old centre (fallback "posterior built for another centre"). The gate verifies that life against the old centre from `genome.json`'s history, and keeps it as a row. Running lives keep their birth genome. (Tasks 8, 9, 11)
4. **A messy confirmation.** For example: duplicate evalSeeds, a burst of births overshooting an arm, a failed candidate life, an arm that never fills by the 21-day deadline, or lives still running at the decision. The outcomes:
   - extras beyond N are ignored by rule;
   - a failed life counts as a non-pass;
   - a deadline gives `incomplete` and no promotion;
   - a result is appended exactly once.

   (Task 9)
5. **Numerically awkward rows.** For example: two rows with identical x, all targets equal, exactly d + 1 rows, or a kernel matrix that defeats every jitter step. Each gives a deterministic result: a fit, or "no model" with the reason recorded, and then centre draws. No NaN reaches a file or a draw. (Tasks 2–4, 8)

---

### Task 1: The posterior file, schemaVersion 2, and the mode guard

**Files:**
- Modify: `packages/meta/src/posterior.ts`, `packages/meta/src/index.ts`
- Test: `packages/meta/test/posterior.test.ts` (new; move the 05a `parsePosterior` cases here from `draw.test.ts`)
- Create: `docs/decisions/0024-meta-learner-search-calls.md` (from `0000-template.md`; sections added by later tasks)

**Interfaces:**
- Produces:
  ```ts
  export const POSTERIOR_SCHEMA = 2
  export interface GpHyper { lengthscales: number[]; outputscale: number; noise: number; mean: number }
  export interface GpModel {
    genes: string[]                                    // searched(space) names, in file order: the cube's axes
    rows: { learnerId: string; x: number[] }[]          // x rounded to 6 decimals, in [0, 1]
    y: number[]                                        // the search evaluator's targets, same order
    hyper: GpHyper
    fit: { objective: number; start: number; iterations: number[] }
  }
  export interface Incumbent { x: number[]; genomeSha: string; mean: number; pBeatsCentre: number }
  export type ArmName = "candidate" | "centre" | "newborn"
  export interface ServedConfirmation {
    id: number; seedBase: number                       // seedBase = 10_000 + 1_000·id
    candidate: { genomeSha: string; values: Genome }; centre: { genomeSha: string; values: Genome }
    target: Record<ArmName, number>; counted: Record<ArmName, number>; need: Record<ArmName, number>
  }
  export interface PosteriorPage {
    eligibleLives: number; lastPromotion: string | null; calibrationLives: number; calibrationTarget: number; looks: number
    rowsSinceDeclaration: number                       // v2 only; a v1 file reads as 0
  }
  export interface Posterior {
    schemaVersion: 1 | 2; genomeSchema: number; evaluator: string; sourceCommit: string; mode: Mode
    centre: { genomeSha: string; values: Genome }
    calibration: { genomes: { genomeSha: string; values: Genome }[]; need: number[] }
    models: { gpu?: GpModel }; incumbent: Incumbent | null; confirmation: ServedConfirmation | null
    diagnostics: Record<string, unknown>; page: PosteriorPage
  }
  export function parsePosterior(text: string): Posterior | null   // never throws
  ```
- **What v1 reads as.** A v1 file is checked exactly as 05a checks it. It is returned with `models: {}`, `incumbent: null`, `confirmation: null` and `page.rowsSinceDeclaration: 0`. 05a wrote nothing else, so its draws are unchanged.
- **The v2 strict rules (null on any violation):**
  - Keys are exactly v1's top keys. `page` has exactly v1's page keys plus `rowsSinceDeclaration`.
  - **The mode guard:** `models` must be `{}`, `incumbent` null and `confirmation` null unless `mode === "search"`.
  - `models` may have only the key `gpu`.
  - `GpModel`:
    - `genes` holds 1–40 distinct `SETTING_KEY_RE` names.
    - `rows.length === y.length` and lies in [1, 1000]. Each `x` has `genes.length` finite entries in [0, 1], and the learnerIds match `/^lrn_[0-9a-f]{16}$/` and strictly increase.
    - `y` is finite.
    - `hyper.lengthscales.length === genes.length`, with every entry finite and > 0. `outputscale` > 0, `noise ≥ 1e-3`, `mean` finite.
    - `fit.iterations` has 4 non-negative integers, and `fit.start` is in 0..3.
  - `Incumbent`: `x` has `genes.length` entries in [0, 1]; `genomeSha` is hex64; `pBeatsCentre` is in [0, 1]. It is non-null only if `models.gpu` exists.
  - `ServedConfirmation`:
    - `id` is a non-negative integer, and `seedBase === 10000 + 1000 * id`.
    - `candidate.genomeSha !== centre.genomeSha`.
    - Every `target`, `counted` and `need` is a count, and `need[a] ≤ target[a]`.

- [ ] **Step 1: Write the failing tests** (`posterior.test.ts`)
  - `"the committed data/meta/posterior.json (v1) parses, as models {}, incumbent null, confirmation null, rowsSinceDeclaration 0"`
  - `"a v2 search posterior with a model, an incumbent and a confirmation parses (fixture fixtures/posterior-v2-search.json)"`
  - `"mode guard: the same v2 file with mode calibrate, paused or flat → null; with models {} but a confirmation in calibrate → null"` (Review Focus 2)
  - `"v2 violations → null, one case each: rows/y length mismatch, x outside [0,1], 1001 rows, unsorted learnerIds, noise 5e-4, lengthscales of the wrong length, seedBase ≠ 10000 + 1000·id, need > target, candidate = centre, incumbent without a model"`
  - `"schemaVersion 3 → null"`
  - `"05a's parser refuses every v2 file: its schemaVersion is 2"`. Read the fixture's `schemaVersion` and assert it is `2`, which 05a's `v.schemaVersion !== 1` check refuses (Review Focus 1). Task 13 checks this against the real 05a code.
- [ ] **Step 2: Run them; they fail.** `npx vitest run packages/meta/test/posterior.test.ts`. Expected: FAIL (no v2).
- [ ] **Step 3: Implement.** Write the fixture by hand (3 genes, 4 rows); Task 4 regenerates it from a real fit. Start **0024** with these sections:
  - "Posterior schema 2". Old tabs draw centre from it until they reload (05a ruling).
  - "The mode guard". How "inert until signal" is structural: the four points of Global Constraints.
- [ ] **Step 4: Run them; they pass.** Then run `npx vitest run packages/meta`; `draw.test.ts`'s 05a fixtures stay byte-identical.
- [ ] **Step 5: Commit** `feat(meta): posterior schemaVersion 2 (model, incumbent, confirmation) with a mode guard; v1 still read; decision 0024 started`

### Task 2: Linear algebra and the GP model

**Files:**
- Create: `packages/meta/src/linalg.ts`, `packages/meta/src/gp.ts`
- Test: `packages/meta/test/gp.test.ts`

**Interfaces:**
- Produces:
  ```ts
  // linalg.ts (row-major Float64Array, n×n)
  export interface Chol { L: Float64Array; n: number; jitter: number }
  export function cholesky(a: Float64Array, n: number, baseJitter: number, steps: number): Chol | null
    // tries jitter = baseJitter·10^k for k = 0..steps−1 (added to the diagonal), the first that factors wins; null if none
  export function solveLower(c: Chol, b: Float64Array): Float64Array       // L z = b
  export function solveUpper(c: Chol, z: Float64Array): Float64Array       // Lᵀ x = z
  export function cholInverse(c: Chol): Float64Array                       // (L Lᵀ)⁻¹
  // gp.ts
  export const NOISE_FLOOR = 1e-3
  export const FIT_JITTER_STEPS = 7                                         // 1e-8·mean(diag) … 1e-2·mean(diag) (spec §4.3)
  export interface GpData { x: readonly (readonly number[])[]; y: readonly number[] }
  export function matern52(r: number): number                              // (1 + √5 r + 5r²/3)·dexp(−√5 r)
  export function thetaOf(h: GpHyper): Float64Array                        // [ln ℓ_1..ℓ_d, ln σ_f², ln(σ_n² − NOISE_FLOOR), m]
  export function hyperOf(theta: Float64Array, d: number): GpHyper
  export function priorOf(d: number): { mode: Float64Array; sd: Float64Array }
  export function objective(theta: Float64Array, data: GpData): { value: number; grad: Float64Array } | null
    // log marginal likelihood + log prior and its analytic gradient; null if θ is outside mode ± 6 sd or Cholesky fails
  export interface GpPosterior { data: GpData; hyper: GpHyper; chol: Chol; alpha: Float64Array }
  export function conditionGp(data: GpData, hyper: GpHyper): GpPosterior | null
  export function predict(post: GpPosterior, xs: readonly (readonly number[])[], joint: boolean): { mean: Float64Array; cov: Float64Array | null; variance: Float64Array }
    // latent f (no noise); cov is K×K when joint, else null
  ```
- **The model (spec §4.3).**
  - It has a constant mean m.
  - The kernel is σ_f² · Matérn-5/2, with r = √Σ_j ((x_j − x'_j)/ℓ_j)².
  - The noise is σ_n² = 1e-3 + dexp(θ_n), a smooth floor [agent].
- **The priors.**
  - ln ℓ_j ~ N(√2 + ½ dlog(d), √3).
  - ln σ_f² ~ N(0, 1).
  - θ_n ~ N(dlog(0.5), 1).
  - m ~ N(0, 1).
- **Gradient.** ∂/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ), plus the prior's gradient. Use `cholInverse` and loop over i < j once.
- **Categoricals.** None in Phase A; the parser still rejects `gp` categoricals. The overlap kernel is specified in 0024 and built with the first schema that activates one (conflict 4).

- [ ] **Step 1: Write the failing tests** (`gp.test.ts`)
  - `"cholesky of a known 3×3 SPD matrix equals the hand factor to 1e-15; a rank-1 4×4 matrix needs jitter > 0 and records it; a matrix with a NaN → null"`
  - `"matern52(0) = 1; matern52(1) = (1 + √5 + 5/3)·e^(−√5) to 1e-15; decreasing on [0, 10]"`
  - `"objective's gradient matches central finite differences (h 1e-6) to 1e-5 relative on 3 seeded datasets (d 3, 12; N 13, 40)"`
  - `"objective at a θ 7 prior sd from the mode → null"`
  - `"predict at a training point with noise 1e-3 is within 0.05 of its y on a smooth fixture; joint cov is symmetric and its diagonal equals variance"`
  - `"two identical x rows and all-equal y: objective and conditionGp return finite values"` (Review Focus 5)
  - `"every output on the fixture inputs equals fixtures/gp-golden.json bit for bit"`. Generate it with `GP_WRITE=1`.
- [ ] **Step 2: Run them; they fail.** `npx vitest run packages/meta/test/gp.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Then run `npm run lint`. Expected: no purity violations.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(meta): deterministic Cholesky and a Matérn-5/2 ARD GP with MAP objective and analytic gradient`

### Task 3: The deterministic fit (multi-start L-BFGS)

**Files:**
- Create: `packages/meta/src/lbfgs.ts`, `packages/meta/src/fit.ts`
- Test: `packages/meta/test/fit.test.ts`

**Interfaces:**
- Consumes: `objective`, `priorOf`, `hyperOf` (Task 2); `sobolPoints`, `SOBOL_MAX_DIMS` (detmath).
- Produces:
  ```ts
  export interface LbfgsResult { x: Float64Array; value: number; iterations: number; converged: boolean }
  export function minimise(f: (x: Float64Array) => { value: number; grad: Float64Array } | null, x0: Float64Array,
    opts: { memory: number; maxIters: number; tol: number }): LbfgsResult
  export const FIT_SEED = "gp-fit/1", FIT_STARTS = 4, FIT_MAX_ITERS = 100, FIT_TOL = 1e-6, LBFGS_MEMORY = 10, START_SPREAD_SD = 1.5
  export interface FitResult { hyper: GpHyper; objective: number; start: number; iterations: number[] }
  export function fitGp(data: GpData): FitResult | null   // null: every start failed at θ0
  ```
- **The algorithm** ([agent] within spec §4.3).
  - It minimises −objective.
  - **The starts.** Start 0 is the prior mode. Starts 1–3 are `mode + START_SPREAD_SD·sd·(2u − 1)` for u in `sobolPoints(d + 3, 3, FIT_SEED)`. `d + 3 > SOBOL_MAX_DIMS` throws: more than 13 searched genes needs a recorded decision that also adds direction numbers.
  - **The line search.** Two-loop recursion with memory 10, then backtracking Armijo with c₁ = 1e-4. The first step length is `1 / max(1, ‖g‖∞)` on the first iteration and 1 after that. It halves at most 30 times. A null objective counts as no decrease.
  - **Curvature pairs.** A pair with sᵀy ≤ 1e-10 is skipped.
  - **Stopping.** It stops when |Δvalue| < 1e-6, when ‖g‖∞ < 1e-8, when the line search fails, or after 100 iterations.
  - **The winner.** The best value wins. Ties go to the earliest start: compare with a strict `<`.

- [ ] **Step 1: Write the failing tests** (`fit.test.ts`)
  - `"minimise finds the Rosenbrock minimum (1, 1) to 1e-4 from (−1.2, 1)"`
  - `"fitGp on data drawn from a known GP (d 3, N 60, ℓ [0.2, 1, 5], noise 0.1, seeded) recovers ℓ_1 < ℓ_2 < ℓ_3 and noise within ×3"`
  - `"fitGp twice gives identical bytes (JSON.stringify of the FitResult); the result equals fixtures/fit-golden.json"`. Generate it with `FIT_WRITE=1`.
  - `"a start whose objective is null at θ0 is skipped; all four null → null"`
  - `"d = 14 → throws (more searched genes than Sobol dimensions)"`
- [ ] **Step 2: Run them; they fail.** `npx vitest run packages/meta/test/fit.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Then time `fitGp` on 300 seeded rows with d = 12 in Node 22 (`npx tsx` of a scratch script in `.local/`). Record the seconds in 0024 §"Fit cost". If it exceeds 120 s, stop and report.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(meta): deterministic multi-start L-BFGS fit of the GP's hyperparameters`

### Task 4: The Thompson sampler, the candidate set and the incumbent

**Files:**
- Create: `packages/meta/src/thompson.ts`
- Test: `packages/meta/test/thompson.test.ts`
- Modify: `packages/meta/test/fixtures/posterior-v2-search.json` (regenerated from a real fit)

**Interfaces:**
- Consumes: `conditionGp`, `predict` (Task 2); `normInv`, `normCdf`, `createRng`, `hashSeed` (detmath).
- Produces:
  ```ts
  export const TS_UNIFORM = 256, TS_PARENTS = 16, TS_PER_PARENT = 16, TS_LOCAL_SD = 0.1, TS_JITTER = 1e-6, TS_JITTER_STEPS = 5
  export function normalOf(u: number): number                         // normInv(u + 2^-33): rng() ∈ [0,1) never hits 0
  export function round6(v: number): number                           // Math.round(v·1e6)/1e6
  export function candidateSet(post: GpPosterior, rng: () => number): number[][]
  export function thompsonPick(post: GpPosterior, cands: readonly (readonly number[])[], rng: () => number): { index: number; x: number[] } | null
  export function incumbentOf(post: GpPosterior, centreX: readonly number[], seed: number): { x: number[]; mean: number; pBeatsCentre: number } | null
  export function modelPosterior(m: GpModel): GpPosterior | null       // conditionGp of the shipped rows and hyper
  ```
- **The candidate set** (spec §4.3, with one change). Each step draws from the rng in this order:
  1. 256 uniform points, d draws each.
  2. The 16 parents: the training rows with the highest posterior mean, ties to the lowest index.
  3. For each parent, 16 children: x_j + σ_j·`normalOf(rng())` per dimension, clipped to [0, 1].
  - **The change:** σ_j = min(TS_LOCAL_SD·ℓ_j, TS_LOCAL_SD) [agent; conflict 3]. The spec's 0.1·ℓ_j alone is about 1.4 at the prior's ℓ ≈ 14, which would clip most children to the cube's faces.
- **The sample.** f = μ + L·z over the 512 candidates. L is the Cholesky of Σ + jitter, starting at TS_JITTER and ×10 up to 5 steps. z_i = `normalOf(rng())`. The pick is the argmax, ties to the lowest index. If every jitter fails, the result is null.
- **The incumbent** (spec §4.4).
  - **Where it looks.** The argmax of the posterior mean over `candidateSet(post, createRng(seed))`. The gate passes `seed = hashSeed(0, "incumbent/" + modelSha)`, where `modelSha` is the sha256 of the serialised model. The spec's `"incumbent/" + posteriorSha` is circular: the posterior holds the incumbent (conflict 2).
  - **pBeatsCentre** = `normCdf((μ* − μ_c)/√max(v, 1e-12))`, with v = σ*² + σ_c² − 2·cov, all from one joint `predict` at [x*, centreX].

- [ ] **Step 1: Write the failing tests** (`thompson.test.ts`)
  - `"candidateSet has 512 points in [0,1]^d; the first 256 consume exactly 256·d rng draws; children stay within 4·σ_j of their parent unless clipped"`
  - `"thompsonPick over 400 seeds on a 1-d model with a clear peak at 0.7: ≥ 80 % of picks within 0.1 of 0.7, and some outside (exploration)"`
  - `"thompsonPick is a pure function of (model, seed): same seed → same index; 20 picks equal fixtures/ts-picks.json"`
  - `"incumbentOf on a model peaked away from the centre gives pBeatsCentre > 0.95; on a flat model ≈ 0.5 ± 0.1"`
  - `"a Σ that fails every jitter step → thompsonPick null"` (Review Focus 5)
- [ ] **Step 2: Run them; they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Regenerate `posterior-v2-search.json` with a real `fitGp` on a 12-gene fixture of 40 rows, so Task 1's fixture is a genuine model.
- [ ] **Step 4: Run them; they pass.** Run all of `packages/meta`.
- [ ] **Step 5: Commit** `feat(meta): Thompson sampling on a seeded candidate set, and the incumbent with P(beats centre)`

### Task 5: `birthDraw` gains the `ts` and `confirm` branches

**Files:**
- Modify: `packages/meta/src/draw.ts`, `packages/meta/src/index.ts`
- Test: `packages/meta/test/draw.test.ts`, `packages/meta/test/fixtures/draws-05b.json` (new; `draws.json` from 05a stays unchanged)

**Interfaces:**
- Consumes: `Posterior`, `ServedConfirmation` (Task 1); `modelPosterior`, `candidateSet`, `thompsonPick` (Task 4).
- Produces:
  ```ts
  export const CONFIRM_SHARE_END = 0.7
  export const CONFIRM_SEEDS = 1000
  export const ARM_OF_J: readonly ArmName[] = ["candidate", "centre", "newborn", "newborn"]   // j mod 4 (spec §4.5)
  export function armOfSeed(evalSeed: number, id: number): ArmName                           // ARM_OF_J[(evalSeed − 10000 − 1000·id) mod 4]
  export interface Draw {
    role: Role; method: Method; drawSeed: number | null; posteriorSha: string | null
    evalPanel: number | null                 // null exactly for role confirm
    evalSeed: number | null; confirmation: { id: number; arm: ArmName } | null
    genome: Genome; calibrationIndex?: number; reason?: string
  }
  ```
- **The draw**, after 05a's checks and `u0 = rng()`, in this exact rng order:
  - **`gpu` + `calibrate`:** unchanged.
  - **`gpu` + `search`:**
    1. `u0 < 0.2` → `uniform`, unchanged.
    2. Else if `confirmation !== null && u0 < 0.7`: j = ⌊rng()·1000⌋, arm = ARM_OF_J[j % 4]. If `need[arm] > 0`, the method is `confirm` (role `confirm`): evalSeed = seedBase + j, evalPanel null. The genome is `candidate.values` for the candidate and newborn arms and `centre.values` for the centre arm. Otherwise fall through to `ts` with the same rng.
    3. `ts`:
       - **No model to use.** No `models.gpu`, or `genes` ≠ `searched(space)` names, gives `centre` with a reason.
       - **No posterior or pick.** `modelPosterior` null or `thompsonPick` null gives `centre` with a reason.
       - **Otherwise** `ts`, with the genome `fromUnit(space, pick.x, centre)`.
  - **`paused`, `flat`, and every mode on `cpu`:** unchanged. CPU tabs never draw `ts` or `confirm`.
  - **A hand-built `Posterior` object** (bypassing the parser) with a model in mode `calibrate` draws exactly as 05a. The branches test `mode === "search"` before reading `models` or `confirmation`.
- 05a's `evalPanel` stays `evalPanelOf(learnerId)` for every role except `confirm`.

- [ ] **Step 1: Write the failing tests** (`draw.test.ts`, describe `"05b branches"`)
  - `"05a's 20 fixture draws (draws.json) are unchanged under 05b code"` (Review Focus 1)
  - `"a v1 search posterior (models {}): gpu births with u0 ≥ 0.2 draw centre, as in 05a"`
  - `"search with a model: over 2,000 ids, uniform 0.2 ± 0.03, ts 0.8 ± 0.03; with a confirmation: uniform 0.2, confirm ≈ 0.5, ts ≈ 0.3 (± 0.03)"`
  - `"confirm draws: evalSeed − seedBase = j, arm = ARM_OF_J[j mod 4]; candidate and newborn run candidate.values, centre runs centre.values; evalPanel null"`
  - `"an arm with need 0 falls through to ts with the same rng stream (the ts genome equals the one drawn when that arm's need is 0 by construction)"`
  - `"mode guard: a hand-built Posterior with mode calibrate/paused/flat and models.gpu + confirmation set → no draw is ts or confirm over 2,000 ids"` (Review Focus 2)
  - `"cpu in search with a model and a confirmation → only uniform or centre"`
  - `"a model whose genes differ from the space's searched genes → centre with reason"`
  - `"20 ts/confirm draws equal fixtures/draws-05b.json"`. Generate them with `DRAW_WRITE=1`.
- [ ] **Step 2: Run them; they fail.** Expected: FAIL.
- [ ] **Step 3: Implement.** Update the header comment of `draw.ts`. Add 0024 §"Draw order" (the rng order above).
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit** `feat(meta): Thompson-sampling and confirmation draws (search mode on gpu only), 05a draws unchanged`

### Task 6: The cross-engine determinism probe for the fit and the sampler

**Files:**
- Modify: `packages/detmath/test/golden-inputs.ts` (or create `packages/meta/test/gp-probe-inputs.ts` with `computeGp(): GpProbe`), `app/src/probe/detmath-probe.ts`, `app/src/probe/detmath-probe.worker.ts`, `tests/e2e/detmath.spec.ts`, `playwright.config.ts`
- Create: `packages/meta/test/fixtures/gp-probe-golden.json`, `docs/research/2026-10-2x-gp-determinism-probe.md`

**Interfaces:**
- Produces: `computeGp(): { fit: FitResult; picks: number[]; incumbent: { x: number[]; mean: number; pBeatsCentre: number } }`.
  - It runs `fitGp` on a seeded 12-gene, 60-row dataset.
  - It makes 20 `birthDraw` ts picks from the fixture posterior.
  - It computes `incumbentOf`.
- `playwright.config.ts` gains projects `webkit` and `firefox`, used only by `--project` on the probe spec (`grep: /determinism/`). CI keeps running Chromium only.

- [ ] **Step 1: Write the failing test.** In `tests/e2e/detmath.spec.ts`, add `"GP determinism: fit, ts picks and incumbent in the page (main thread and worker) equal the Node golden file bit for bit"`. Generate the golden with `GP_PROBE_WRITE=1 npx vitest run packages/meta/test/gp-probe.test.ts`, whose own test asserts Node twice gives the same bytes.
- [ ] **Step 2: Run it; it fails.** `npx playwright test detmath`. Expected: FAIL (no GP probe).
- [ ] **Step 3: Implement.** Run `npx playwright install webkit firefox`. That fetches browser binaries only; `git diff package-lock.json` must be empty. Then run `npx playwright test detmath --project=chromium --project=webkit --project=firefox`.
- [ ] **Step 4: Record the result.** Write the research note: each engine, PASS or the first differing value. Any engine that differs is **stop and report**: its draws would be flagged, and the user decides whether to ship. Also time one `birthDraw` ts draw in the Chromium worker at 300 rows, and record it (spec estimate 1–3 s).
- [ ] **Step 5: Commit** `test(meta): the GP fit and Thompson draws are bit-identical in Node, Chromium, WebKit and Firefox (probe note)`

### Task 7: Confirmation lives in logs, checkups, the newborn runner and the validator

**Files:**
- Modify: `packages/harness/src/log.ts` (`parseMeta`, `parseLogFile`), `packages/harness/src/checkup.ts` (`confirmSeed`), `packages/harness/src/session.ts` (variant seeds), `packages/harness/src/newborn.ts` (`runNewborn` gains a confirm mode)
- Modify: `tools/gate/validate.ts` (`ValidateOptions.spaces`, seed rule), `tools/gate/gate.ts`, `tools/gate/validate-cli.ts`
- Create: `tools/gate/history.ts`
- Test: `packages/harness/test/log.test.ts`, `checkup.test.ts`, `session.test.ts`, `newborn.test.ts`; `tools/gate/test/validate.test.ts`, `tools/gate/test/history.test.ts`

**Interfaces:**
- Produces:
  ```ts
  // checkup.ts
  export function confirmSeed(evalSeed: number, level: number, milestoneId: string): number // hashSeed(0, `confirm/${evalSeed}/${level}/${milestoneId}`)
  // newborn.ts: NewbornConfig.k is replaced by
  who: { kind: "reference"; k: number } | { kind: "confirm"; evalSeed: number; confirmationId: number; drawSeed: number; posteriorSha: string }
  // history.ts
  export function fileHistory(git: (args: string[]) => string, path: string): string[]       // distinct committed texts, newest first
  export function spacesBySchema(git: (args: string[]) => string): Map<number, GenomeSpace> // knowledge/genome-space.json's history; newest text per schemaVersion
  export function centresBySha(git: (args: string[]) => string, spaces: ReadonlyMap<number, GenomeSpace>): Map<string, Genome> // knowledge/genome.json's history
  // validate.ts
  export interface ValidateOptions { spaces?: ReadonlyMap<number, GenomeSpace>; budget?: EvalBudget }   // replaces `space`
  ```
- **The confirmation-newborn file shape** (settles 05a's note). It reuses the reference newborn's shape. The test fixture `confirm-fisher.test.ts` already builds it this way.
  - A file with `role: "confirm"` and `confirmation.arm: "newborn"` is a **frozen file**. It has `seq: 0`, `cu: 0`, `stepsTotal: 0` and `final: true`, and exactly the budget levels' checkups (`allowAhead`). It may carry `heldOut`.
  - A `confirm` life of another arm may not have that shape, and an arm-`newborn` file of any other shape is rejected.
  - `meta.heldOut` is allowed for role `newborn` and for confirm-newborn files.
- **The arm is the seed's.** `parseMeta` rejects `confirmation.arm !== armOfSeed(evalSeed, id)`. The rule is copied as a 4-entry table into log.ts, which cannot import meta. `tools/genome-centre.test.ts` pins the two tables equal. This puts the 05a note "the gate recomputes the arm from evalSeed mod 4" into every parser.
- **Seeds.**
  - A session whose meta block has `evalSeed !== null` uses `variantSeed: (l, m) => confirmSeed(evalSeed, l, m)`.
  - `validateSessionPr` rejects a `confirm` file whose `settings.seed !== meta.evalSeed`. The learner seed is evalSeed (spec §4.5), and the world seed stays the learnerId's [agent].
- **Closing the older-schema bypass** (05a Task 6 note).
  - Every non-legacy block is checked for bounds, settings and sha against `spaces.get(meta.genomeSchema)`.
  - A schema not in the map is rejected (`"unknown genome schema N"`), and a schema newer than the newest one is rejected as before.
  - `gate.ts` and `validate-cli.ts` build the map with `spacesBySchema` from main's full history (both checkouts are `fetch-depth: 0`). They stop, as now, when the current space is invalid.

- [ ] **Step 1: Write the failing tests**
  - log:
    - `"a confirm-newborn file (cu 0, final, three checkups above cu, heldOut) parses; the same shape with arm candidate fails; arm newborn at cu 600 fails"`;
    - `"confirmation.arm must equal ARM_OF_J[(evalSeed − 10000 − 1000·id) mod 4]: 13,004 candidate ok, 13,005 candidate fails, 13,006 newborn ok"`;
    - `"every committed file under data/runs and data/newborns still parses"`.
  - checkup: `"with variantSeed confirmSeed(e, …), two learners with evalSeed e meet identical variants, disjoint from every panelSeed(p, …) on the fixture levels"`.
  - session: `"a confirm block (candidate arm) runs checkups on confirmSeed variants and logs settings.seed = evalSeed"`.
  - newborn: `"runNewborn({ who: confirm }) writes one frozen file: role confirm, arm newborn, seed evalSeed, confirmSeed variants, evalPanel null; it parses and validates"`; reference mode is unchanged.
  - validate:
    - `"a confirm life whose settings.seed ≠ evalSeed is rejected"`;
    - `"a search file claiming genome schema 1 when the map has only schema 2 is rejected 'unknown genome schema 1'"`;
    - `"with spaces {1, 2}, an out-of-bounds schema-1 genome is rejected (the bypass is closed)"`.
  - history: `"fileHistory on a fixture repo with 3 commits of a file (one repeating an earlier text) returns 2 distinct texts, newest first; spacesBySchema maps each schemaVersion to its newest text"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/harness tools/gate/test/validate.test.ts tools/gate/test/history.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add these sections to 0024: "Confirmation-newborn files", "The arm is the seed's", "Confirmation seeds" and "Every schema checked against its own space".
- [ ] **Step 4: Run the full gates.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(harness,gate): confirmation lives (frozen newborn arm, confirm variants, arm from seed, seed = evalSeed); every genome checked against its own schema's space`

### Task 8: The gate fits the GP: rows, the centre in force, verification, the model and the incumbent

**Files:**
- Create: `tools/gate/search.ts`, `tools/gate/test/search.test.ts`
- Modify: `tools/gate/meta.ts` (`MetaInput`, `MetaFiles`, `verify`, `readLives`, `keptCalibration`, `buildMeta`, `wantedPosteriors`), `tools/gate/meta-cli.ts`
- Modify: `tools/gate/test/meta.test.ts` (the fixture helpers `bornLife`/`grow`/`planted` move to `tools/gate/test/meta-fixture.ts` for Tasks 9 and 12), `tools/gate/test/fixtures/meta/` (goldens)

**Interfaces:**
- Consumes: `fitGp`, `modelPosterior`, `incumbentOf`, `round6`, `birthDraw`, `toUnit`, `fromUnit` (Tasks 3–5); `centresBySha`, `fileHistory` (Task 7).
- Produces:
  ```ts
  // search.ts
  export const MAX_ROWS = 1000, MIN_ROWS = (d: number) => d + 1, TRIGGER_P = 0.95, TRIGGER_NEW_ROWS = 100
  export const ROW_METHODS = ["calibrate", "uniform", "ts", "centre"] as const
  export function rowSubset(ids: readonly string[]): string[]             // ≤ MAX_ROWS: the largest hashSeed(0, "subset/" + id), then sorted by id
  export function buildModel(rows: { learnerId: string; genome: Genome }[], y: number[], space: GenomeSpace, previous: GpModel | null): { model: GpModel | null; reason: string | null }
  export function incumbentFor(model: GpModel, space: GenomeSpace, centre: Genome): Promise<Incumbent | null>
  export function triggered(inc: Incumbent | null, centreSha: string, rowsSinceDeclaration: number, inFlight: boolean): boolean
  // meta.ts
  export const VERIFIED_PATH = "data/meta/verified.json"
  export interface VerifiedCache { schemaVersion: 1; verifier: 2; lives: Record<string, { posteriorSha: string | null; reason: string | null; genomeMismatch: boolean }> }
  // MetaInput gains: centres: ReadonlyMap<string, Genome> (centresBySha); previous.verified: VerifiedCache | null
  // MetaFiles gains: "data/meta/verified.json": string
  ```
- **Only in search.** With `mode !== "search"`, `buildMeta` writes `models: {}`, `incumbent: null`, `confirmation: null`, exactly as 05a, plus `schemaVersion: 2` and `rowsSinceDeclaration: 0`.
- **`keptCalibration` refuses an inconsistent record** (Review Focus 2):
  - `outcome "signal"` needs the last look's `output` to be `{ evaluator, version }` of `config.signal`, `eligible: true` and `score: 1`, with every earlier look at score 0;
  - `"flat"` needs 3 looks, all at score 0;
  - `"pending"` needs every look at score 0.
- **Rows** (spec §4.3).
  - A row is an eligible life under `config.search`. Its method is in `ROW_METHODS`, it is on gpu at the space's schema with inheritance `none`, and it is verified.
  - `y = search.targets(outs)` over all rows. Then `rowSubset` applies above 1,000.
  - `x = toUnit(space, logged genome).map(round6)`.
- **The fit.**
  - Below `MIN_ROWS`, or when `fitGp` is null, `models` is `{}` and `diagnostics.model` carries `{ reason }`.
  - When `previous`'s rows, x and y are identical, its `hyper` and `fit` are reused without refitting. That is safe because the fit is deterministic; a test pins that refitting gives the same bytes.
- **The centre in force at each birth** (05a Task 10 note).
  - **The candidates.** `verify` tries up to two centres: the posterior's `centre.values` (when there is a posterior), then the logged genome, but only when the method is `centre` and the logged genome's sha is in `input.centres` (every `knowledge/genome.json` ever on main).
  - **The rule.** A life verifies if `birthDraw` with either centre (and its sha) reproduces `method`, `drawSeed`, `evalPanel`, `evalSeed` and `confirmation`.
  - **Confirm lives** must also reproduce the genome exactly: the gate fixed it in the declaration (spec §4.5 "Integrity").
- **The verification cache** (05a: `posteriorHistory` growth).
  - **The file.** `data/meta/verified.json` records each life's birth verdict once.
  - **What is skipped.** Lives already in it, with the same `posteriorSha` and `verifier: 2`, are not verified again. `wantedPosteriors` lists only the uncached lives' shas, so the history walk stops at the recent posteriors.
  - **Invalidation.** A missing cache, or one of another `verifier`, verifies everything again.
  - **What the cache never covers.** Scoring and row selection are recomputed every build. Only the draw check is cached.
- **The incumbent and trigger** (spec §4.4): `incumbentFor` uses `seed = hashSeed(0, "incumbent/" + sha256(serialise(model)))`.
  - Its `genomeSha` is that of `fromUnit(space, x*, centre)`.
  - `triggered` holds when all four of these hold:
    - `pBeatsCentre ≥ 0.95`;
    - `rowsSinceDeclaration ≥ 100`, counted since the last declaration's `rowsAtDeclaration`, or else since the signal look's `lives`;
    - no confirmation in flight;
    - the incumbent's genomeSha ≠ the centre's.
  - Task 9 acts on it.

- [ ] **Step 1: Write the failing tests**
  - `search.test.ts`:
    - `"rowSubset keeps all ≤ 1000; of 1,200 ids keeps the 1,000 with the largest subset hash, sorted"`;
    - `"buildModel below d + 1 rows → no model, reason 'fewer than 13 rows'"`;
    - `"buildModel with previous identical rows reuses hyper; refitting gives byte-identical hyper"`;
    - `"triggered: each of the four conditions false alone → false"`.
  - `meta.test.ts`:
    - `"calibrate mode with 200 eligible rows → posterior schemaVersion 2, models {}, incumbent null, confirmation null; no fit is run (a spy on fitGp is not called)"` (the user's condition; Review Focus 2);
    - `"flat after 3 looks: births keep uniform/centre; still no model"`;
    - `"a calibration.json with outcome signal but last look score 0 → buildMeta throws, writing nothing"`;
    - `"30 planted calibration rows → signal → the next build fits models.gpu on all rows; ts lives born from it verify and become rows"`;
    - `"a search posterior built twice → identical bytes; equals fixtures/meta/search-posterior.json"` (spec §7.2 CI golden);
    - `"verification with a moved centre: a life born under posterior P (centre A) whose tab held centre B (in centres) verifies as centre/another-centre; one whose centre is not in centres → 'unknown centre'"` (Review Focus 3);
    - `"a confirm life whose genome differs from the declaration's → ineligible"`;
    - `"verified.json: a second build verifies only new lives (a spy counts birthDraw calls); a changed verifier re-verifies all"`;
    - `"a life born under the last v1 posterior verifies under 05b code"` (Review Focus 1).
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run tools/gate/test/search.test.ts tools/gate/test/meta.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `meta-cli.ts` reads `verified.json`, builds `centres` with `centresBySha(git, spacesBySchema(git))`, and writes `verified.json`. Add these sections to 0024: "Rows and the fit", "Centre in force", "Verification cache" and "Calibration consistency".
- [ ] **Step 4: Run them; they pass.** Then run the full gates.
- [ ] **Step 5: Commit** `feat(gate): fit the GP in search mode only; verify ts and confirm draws against the centre in force; cache verdicts; incumbent and trigger`

### Task 9: The gate declares, serves, decides and promotes

**Files:**
- Create: `tools/gate/confirm.ts`, `tools/gate/test/confirm.test.ts`
- Modify: `tools/gate/meta.ts` (`MetaInput.previous.confirmations`, `MetaFiles`), `tools/gate/meta-cli.ts`

**Interfaces:**
- Consumes: `triggered`, `incumbentFor` (Task 8); `cohortContext`, `getEvaluator` (evaluators); `armOfSeed` (Task 5).
- Produces:
  ```ts
  export const CONFIRM_TARGET = { candidate: 24, centre: 24, newborn: 60 } as const
  export const DEADLINE_DAYS = 21, IN_FLIGHT_HOURS = 48
  export const confirmationPath = (c: number) => `data/meta/confirmations/${c}.json`
  export const PROMOTIONS_PATH = "data/meta/promotions.jsonl", GENOME_PATH = "knowledge/genome.json"
  export interface ConfirmationFile {
    schemaVersion: 1; id: number; genomeSchema: number; confirm: string; search: string
    candidate: { genomeSha: string; values: Genome }; centre: { genomeSha: string; values: Genome }
    seedBase: number; target: typeof CONFIRM_TARGET; declaredAt: string; deadline: string; sourceCommit: string
    trigger: { pBeatsCentre: number; incumbentMean: number; rowsAtDeclaration: number; modelSha: string }
    result: null | { outcome: "pass" | "fail" | "incomplete" | "ineligible"; decidedAt: string; sourceCommit: string; output: EvalOutput; lives: Record<ArmName, string[]> }
  }
  export function parseConfirmation(text: string): ConfirmationFile     // throws on shape errors
  export function armGroups(lives: VerifiedConfirmLife[], c: ConfirmationFile): Record<ArmName, LifeInput[]>
    // lowest learnerId per (arm, evalSeed), then the first target[arm] counted lives per arm in learnerId order
  export function decide(c: ConfirmationFile, groups: Record<ArmName, LifeInput[]>, full: boolean, input: MetaInput): ConfirmationFile["result"]
  // MetaFiles gains: confirmations: Record<string, string> (changed files only); promotionLine: string | null; "knowledge/genome.json": string | null
  ```
- **Declare** (spec §4.5). On `triggered`, write confirmation c = (highest id) + 1, starting at 0, with:
  - `seedBase` = 10,000 + 1,000c;
  - `declaredAt = builtAt`, and `deadline` = builtAt + 21 days;
  - the candidate = the incumbent's genome, and the centre = the current centre.

  The same build serves it in the posterior. The declaration and its first serving land in one commit, before any deploy.
- **Serving `need`.**
  - `counted[a]` is the lives `armGroups` keeps that `confirm-fisher@1` would count: complete or failed.
  - `inFlight[a]` is the verified lives of that arm that are not final, have no budget checkup, and whose newest file has `writtenAt` within 48 h of `builtAt` [guess]. This bounds overshoot under bursts (05a note); extras are still dropped by rule.
  - `need[a] = max(0, target[a] − counted[a] − inFlight[a])`.
- **Decide.**
  - **When.** When `counted` reaches `target` in every arm, or `builtAt > deadline`.
  - **How.** Run `getEvaluator(config.confirm).evaluateCohort(armGroups(…), refs, cohortContext(config))` and append `result` once, with these outcomes:
    - full, `score 1` → `pass`;
    - full, `score 0` → `fail`;
    - full but `eligible: false` → `ineligible`;
    - the deadline passed first → `incomplete`, with the output still recorded.
  - A file with a result is never changed again.
- **Promote (pass only)**, in the same build:
  - **`knowledge/genome.json`.** `serialise({ schemaVersion: 1, genomeSchema, values: candidate.values, provenance: { source: "promotion", confirmation: c, decision: "docs/decisions/0018-one-meta-learner-over-a-learning-genome.md", promotedAt: builtAt, fisher, mannWhitneyP, lives } })`. It must pass `parseCentre`, or the build throws.
  - **`promotions.jsonl`.** One line: `{ confirmation, genomeSha, previous, sourceCommit, promotedAt }`.
  - **The posterior.** It is rebuilt with the new centre, and `page.lastPromotion` is set.
  - **Lives already running.** Their rows stay eligible (their genome is logged).
- **Lives never mix.** Confirm lives are never rows. Search rows are never confirmation lives.

- [ ] **Step 1: Write the failing tests** (`confirm.test.ts`, using `meta-fixture.ts` lives drawn by `birthDraw`)
  - `"a trigger declares confirmation 0: seedBase 10000, targets 24/24/60, deadline +21 days; the posterior serves it with need = target"`
  - `"armGroups: two lives with one (arm, evalSeed) → the lower learnerId; 30 candidate lives → the first 24 by learnerId"` (Review Focus 4)
  - `"need subtracts in-flight lives written within 48 h; a life silent for 49 h no longer holds a slot"`
  - `"full arms with a planted candidate advantage → pass: genome.json is the candidate (parseCentre ok), promotions.jsonl has one line, the posterior's centre is the candidate and lastPromotion is set"`
  - `"full arms, no advantage → fail; nothing else changes"`
  - `"deadline passed with arms short → incomplete; no promotion; result appended once (a later build changes nothing)"`
  - `"a failed candidate life counts as a non-pass; an abandoned one is excluded"`
  - `"no declaration while one is in flight; the next trigger needs 100 new rows after rowsAtDeclaration"`
  - `"buildMeta twice on the same input (builtAt included) → identical bytes for every file"`
- [ ] **Step 2: Run them; they fail.** `npx vitest run tools/gate/test/confirm.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `meta-cli.ts` does three new things:
  - it reads `data/meta/confirmations/*.json` (ids 0..n−1 with no gaps, or it throws);
  - it writes the changed confirmation files, appends `promotionLine` and writes `knowledge/genome.json` when it is non-null;
  - it prints `changed=`.

  Add these sections to 0024: "Confirmation timing" (deadline, in-flight 48 h), "Decision outcomes" and "Promotion record".
- [ ] **Step 4: Run them; they pass.** Then run the full gates.
- [ ] **Step 5: Commit** `feat(gate): declare, serve, decide (confirm-fisher@1) and promote: knowledge/genome.json written by the gate on a pass only`

### Task 10: The gate's commit of the new files and `knowledge/genome.json`; centre-dependent checks follow a promotion

**Files:**
- Modify: `tools/gate/commit-meta.ts` (`META_PATH_RE`, staging, `promotionErrors`, append-only check, commit message), `tools/gate/test/commit-meta.test.ts`
- Modify: `tools/genome-centre.test.ts`, `tools/eval/newborn-reference.ts` (centre check), `tools/eval/test/eval.test.ts`
- Modify: `tools/walls.test.ts` or the workflow test (the gate stages `knowledge/genome.json` only via `commit-meta.ts`)

**Interfaces:**
- Produces:
  ```ts
  export const META_PATH_RE = /^(data\/meta\/(posterior\.json|calibration\.json|diagnostics\.json|verified\.json|posterior-index\.jsonl|promotions\.jsonl|confirmations\/(0|[1-9][0-9]{0,5})\.json)|knowledge\/genome\.json)$/
  export function promotionErrors(show: (path: string) => string | null, staged: string[]): string[] // staged contents via `git show :<path>`
  ```
- **Staging.** `git add` of exactly `data/meta` and `knowledge/genome.json`. All of 05a's guards stay:
  - only matching paths;
  - only regular `100644` files, added or modified; `knowledge/genome.json` may only be modified;
  - the lockfile check before every build and commit;
  - HEAD equals origin/main;
  - no other change in the tree;
  - no force;
  - at most 3 tries, rebuilding on the new main.
- **`promotionErrors`** (defence in depth for "gate-computed only"). If `knowledge/genome.json` is staged, all of these must hold:
  - exactly one staged confirmation file changes, gaining `result.outcome: "pass"`;
  - its `candidate.values` deep-equal the staged `genome.json` values, and `provenance.confirmation` is its id;
  - `promotions.jsonl` is staged with exactly one new last line naming that id.

  Otherwise it throws and nothing is committed.
- **Append-only.** A staged change to an existing confirmation file may only replace `result: null` with a result. A staged change to `promotions.jsonl` or `posterior-index.jsonl` may only append lines.
- **Message.** `meta: <mode>, <n> eligible lives, posterior <sha8>`. Add `; declare confirmation <c>`, `; confirmation <c> <outcome>` or `; promote <genomeSha8>` when they apply.
- **What follows the centre.**
  - `genome-centre.test.ts` now pins the **space** centres to `DEFAULT_SETTINGS`, `CAREGIVER_DEFAULTS` and the harness defaults. It also checks `genome.json`:
    - it passes `parseCentre`;
    - with `provenance.source === "initial"` it equals the space centres;
    - with `"promotion"` it equals the candidate of `data/meta/confirmations/<provenance.confirmation>.json`, whose result is `pass`.
  - `newborn-reference.ts` checks each newborn's genome against the centre its lives ran, which must be one genome present in `centresBySha` history, instead of against today's `genome.json`. So `--check` stays green after a promotion.

- [ ] **Step 1: Write the failing tests** (`commit-meta.test.ts`, local bare repo as in 05a)
  - `"a declaration commit (confirmations/0.json added, posterior, verified) → committed"`
  - `"a promotion commit (genome.json modified + confirmations/0.json result pass + one promotions line) → committed; message ends '; promote <sha8>'"`
  - `"genome.json staged without a passing confirmation, or with values ≠ the candidate, or without a promotions line → throws, nothing committed"`
  - `"a confirmation file whose candidate changed, or whose existing result changed → throws"`
  - `"promotions.jsonl with a rewritten earlier line → throws"`
  - `"knowledge/genome-space.json or config/evaluation.json staged → throws"` (the 05a cases stay)
  - `"lockfile moved between npm ci and a promotion commit → throws before committing"`
  - `genome-centre.test.ts`: `"a promoted genome.json fixture with its passing confirmation file is accepted; one naming a failed confirmation is rejected"`
  - eval: `"newborn-reference --check passes when genome.json has been promoted (fixture)"`
- [ ] **Step 2: Run them; they fail.** `npx vitest run tools/gate/test/commit-meta.test.ts tools/genome-centre.test.ts tools/eval`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `gate.yml`, add a comment saying the deploy fires on `meta == 'committed'`, which covers a promotion. A promotion changes `knowledgeSha` by design (0019), and `deploy.yml` already copies `knowledge/*.json`. Add 0024 §"Gate commits in 05b".
- [ ] **Step 4: Run the full gates.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(gate): commit confirmations, promotions and knowledge/genome.json (gate-computed only, cross-checked, append-only); centre checks follow promotions`

### Task 11: The page — `ts` and `confirm` births, confirmation newborns, the search lines

**Files:**
- Modify: `app/src/birth.ts` (`block` takes the draw's `evalPanel`, `evalSeed`, `confirmation`; `Birth.seed?`), `app/src/worker.ts` (seed override; confirm-newborn run), `app/src/protocol.ts`, `app/src/main.ts`, `app/src/recipe.ts`, `app/src/ui.ts`
- Modify: `tools/gate/summary.ts`, `packages/harness/src/log.ts` (`SessionRow.recipe`, `parseSessionRow`)
- Test: `app/test/birth.test.ts`, `app/test/recipe.test.ts`, `tools/gate/test/summary.test.ts`, `tests/e2e/meta.spec.ts`

**Interfaces:**
- Produces:
  ```ts
  // birth.ts: Birth gains seed?: number (role confirm: evalSeed); kind: "life" | "confirmNewborn"
  // protocol.ts: FromWorker gains { type: "confirmNewborn"; learnerId: string; evalSeed: number; done: boolean }
  // log.ts: SessionRow gains recipe?: { role: MetaRole; method: MetaMethod }   (from the first meta block; dropped if malformed)
  // recipe.ts
  export function recipeText(meta: MetaBlock, posterior: Posterior | null): string   // new cases below
  export function metaStateText(posterior: Posterior | null): string                 // new cases below
  ```
- **The worker.**
  - **The seed.** A `confirm` draw of the candidate or centre arm runs with `seed: evalSeed`; `configOf` applies `birth.seed`.
  - **The newborn arm.** A `confirm` draw of arm `newborn` runs `runNewborn({ who: { kind: "confirm", … } })` in place of a life, which takes about 4 min. That file is contributed normally, not local-only: it is gate data. The page then starts a newborn as after any final file.
  - **The draw's cost.** The draw runs once, before the learner starts. The status line reads "drawing a recipe…" while it runs.
- **Copy** (spec §6):
  - `recipeText`:
    - ts: `"drawn by the meta-learner (a Thompson sample from 214 lives)"`, using `models.gpu.rows.length` of the posterior the life was drawn from;
    - confirm: `"a confirmation life (candidate arm, fresh world 10,347)"`, with arms `candidate arm`, `current-recipe arm` and `newborn arm`.
  - `metaStateText` in mode search:
    - with no model yet: `"Searching: 214 lives, not yet enough for the model."`;
    - with a model: `"Searching: 214 lives; its best guess is predicted to beat the current recipe with 72 % confidence."`;
    - confirming: `"Confirming recipe 3f2a1c9e on fresh worlds: 31 of 108 lives."`;
    - when `page.lastPromotion` is set, append `" Last promotion: 2026-11-03."`.
  - The 05a line `"Signal found: the search can start…"` is removed.
- **Progress** (spec §6, reduced; conflict 6). The HISTORY tab's sessions table gains two columns:
  - "Recipe", from `recipe.method`;
  - "Score", from `evals[pinned]`.

  A line above the table shows the current recipe: `"Current recipe: the initial one"` or `"Current recipe: promoted 2026-11-03 (confirmation 0)"`, from `genome.json`'s provenance. It changes only on a passed confirmation (Ladder-style). The chart is left for a later plan.

- [ ] **Step 1: Write the failing tests**
  - birth:
    - `"birthFromDraw of a confirm draw (candidate arm) → meta role confirm, evalSeed, confirmation, evalPanel null, seed = evalSeed; parseMeta accepts it"`;
    - `"birthFromDraw equals birthDraw + decodeGenome for 50 ids under the v2 search fixture (ts and confirm included)"`.
  - recipe: each copy line above, for its input. A v1 search posterior in a 05b tab reads `"Searching: 0 lives, not yet enough for the model."`.
  - summary: `"rows carry recipe {role, method} from the first meta block; parseSessionRow drops a malformed recipe, not the row"`.
  - `meta.spec.ts` (the stubbed `posterior.json` is the v2 search fixture; this is the only place tabs meet a search posterior before signal, and it is a Playwright route, which cannot reach production):
    - `"a search posterior: the page's first file has method ts or uniform, and its genome equals birthDraw in Node for the same learnerId and bytes"`;
    - `"a confirmation in the posterior and a learnerId whose draw is the newborn arm (chosen in the test by recomputing birthDraw): one frozen confirm file is queued and a new life starts"`;
    - `"the committed calibrate posterior: the page shows 'Calibrating: …' and the first file's method is calibrate, uniform or centre"` (inert);
    - `"a v2 posterior with mode calibrate and a model (mode guard): the first file has method centre and posteriorSha null; the meta-learner line reads 'Not reachable: this learner runs the current recipe.'"` (Review Focus 2).
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run app/test tools/gate/test/summary.test.ts && npm run e2e -- meta`. Expected: FAIL.
- [ ] **Step 3: Implement.** Then run `npm run dev` with `data/meta/posterior.json` temporarily replaced by the v2 search fixture (restore it after; `git status` clean). Take screenshots of the LEARNER and HISTORY tabs at 1280×800 and 390×844 to `.local/05b-page-*.png`. Look at them: the recipe line stays one line, and the chart keeps its height.
- [ ] **Step 4: Run the full gates and `npm run e2e`.** Expected: PASS, `loop.spec.ts` included.
- [ ] **Step 5: Commit** `feat(app): Thompson and confirmation births, confirmation newborns, the search/confirm/promotion lines, recipe and score in history`

### Task 12: The end-to-end simulation on a local fixture repo

**Files:**
- Create: `tools/sim/meta-sim.ts`, `tools/sim/test/meta-sim.test.ts`

**Interfaces:**
- Consumes: `meta-fixture.ts` (`bornLife` and its life builders, Task 8), `buildMeta` via the real `meta-cli.ts` and `commit-meta.ts`, `git-fixture.ts`, `tools/gate/manifest.ts` (knowledgeSha).
- Produces: `runSim(opts: { dir: string; quick: boolean; log(m: string): void }): Promise<SimReport>`, where `SimReport = { stages: { name: string; mode: Mode; commit: string; paths: string[] }[]; promoted: boolean; knowledgeShaBefore: string; knowledgeShaAfter: string }`.
- **The fixture repo.** It is a temporary clone of main's `knowledge/`, `config/` and `data/meta/` with an empty `data/runs/`, plus a bare "origin". The real `meta-cli.ts` and `commit-meta.ts` run against it from this checkout's code (`LOCK_BLOB` set to the fixture's lockfile blob).
- **The stages.** Each stage writes synthetic lives drawn with `birthDraw` from the fixture's current posterior, commits them as "merged", and runs a gate pass.
  1. **Calibrate.** Planted calibration lives, until look 1 gives `signal`. This is the only way to reach `search`: data through the real signal check (conflict 7).
  2. **Search.** `ts` and `uniform` lives whose score follows a planted smooth optimum, until the trigger declares confirmation 0.
  3. **Confirm.** Confirmation lives with a planted candidate advantage, plus newborn-arm frozen files, until the decision.
  4. **Promote.** Check that `knowledge/genome.json` changed and that the manifest's `knowledgeSha` differs.
- **Every stage asserts:**
  - the commit's paths match `META_PATH_RE`;
  - `git log` shows no force;
  - the posterior parses;
  - every life verifies.
- `quick: true` uses d = 3 (a reduced test space) for CI speed. The full run uses main's space and is run in Task 13.

- [ ] **Step 1: Write the failing test** (`meta-sim.test.ts`): `"quick sim: calibrate → signal → search → declare → pass → promote; knowledgeSha changes once; every gate commit touches only META_PATH_RE paths; no stage before signal writes a model"`.
- [ ] **Step 2: Run it; it fails.** `npx vitest run tools/sim`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it; it passes** in under 60 s. Run the full sim once: `npx tsx tools/sim/meta-sim.ts .local/sim`. Record its stage table and timings for Task 13.
- [ ] **Step 5: Commit** `test(sim): end-to-end meta-learner simulation on a local fixture repo (search reached by planted data only)`

### Task 13: Merge, deploy, CI green, the inertness live check, write-up

- [ ] **Step 1: Merge and deploy.**
  - Merge to `main` and push.
  - Open the CI run and wait for green.
  - Wait for the deploy. In the deployed `knowledge/manifest.json`, the `codeSha` is new, `learnerStateVersion` is 2, and `knowledgeSha` is **unchanged**: 05b changes no knowledge file.
- [ ] **Step 2: First 05b gate run.** Run `gate.yml` by `workflow_dispatch`.
  - **What it committed.** The log shows `meta=committed` (the posterior bytes changed to v2). `git show --stat` lists only `data/meta/{posterior.json,posterior-index.jsonl,verified.json,diagnostics.json}`. There is no `confirmations/`, no `promotions.jsonl` and no `knowledge/genome.json`.
  - **What is served.** `https://markwhiting.github.io/learner/knowledge/posterior.json` is `schemaVersion: 2`, `mode: "calibrate"` (or `flat`), `models: {}`, `incumbent: null` and `confirmation: null`.
- [ ] **Step 3: Old code against the new posterior** (Review Focus 1). Check out 05a's parser in a scratch worktree: `git worktree add .local/wt-05a 730f905`. Run its `parsePosterior` on the served v2 bytes with `npx tsx`; expected: `null`. Also run 05b's `parsePosterior` on the last v1 bytes; expected: non-null, with draws equal to 05a's for 20 learnerIds. Remove the worktree.
- [ ] **Step 4: Inert on the live page.** Wrap the run in `caffeinate -dims` and use Playwright Chromium (WebGPU) on the live page. Births stay 05a's:
  - Open 5 fresh tabs (fresh profiles).
  - Each newborn's first queued file has `meta.method` in {calibrate, uniform, centre} and `meta.posteriorSha` = the served sha.
  - Each genome equals `birthDraw` recomputed in Node.
  - The page shows "Calibrating: …". Take a screenshot and look at it.
- [ ] **Step 5: The simulated end-to-end.**
  - Run `npx tsx tools/sim/meta-sim.ts .local/sim` (main's space). Record the stage table, the fit seconds per build and the promotion.
  - Run `npx playwright test meta --grep "search posterior"` against the sim's final posterior (copied over the e2e stub). Expected: ts draws equal to Node.
- [ ] **Step 6: Write `docs/research/2026-10-2x-meta-search-live.md`.** Keep it short:
  - the checks above, the determinism probe (Task 6) and the sim table;
  - the measured fit and draw times;
  - caveats: inert until signal, one machine, Playwright WebKit is not Safari, the progress chart deferred.
  - **What switches it on:** the signal check's look; nothing else.
  - **What "flat" means:** the code stays inert, and the next step is v2 genes under a new genome schema (a recorded decision), which restarts calibration.

  Set row 05b in `docs/plans/README.md` to "built, inert until signal" with a link. Update Q14 in `docs/open-questions.md` to cite 0024.
- [ ] **Step 7: Commit, push and check that CI is green.** Message: `docs: Plan 05b live check — meta-learner search (inert until signal)`.

**After this plan.**
- **If look 1, 2 or 3 answers "signal":** the next hourly gate run fits the model, and tabs begin `ts` draws without any deploy. On that day, check the first model commit, the first `ts` life's verification, and the page's "Searching" line. Write it up, favourable or not.
- **If the check answers "flat":** nothing in 05b runs. Write the research note, and draft the v2-genes decision for the user.

---

## Spec conflicts and gaps found while planning (resolutions adopted above; each goes into 0024 as an agent's call)
1. **The incumbent's seed is circular.** Spec §4.4 seeds the incumbent's candidates with `"incumbent/" + posteriorSha`, but the posterior contains the incumbent. Resolution: the seed comes from `modelSha`, the sha256 of the serialised model block.
2. **Local candidates would sit on the cube's faces.** Spec §4.3 perturbs by N(0, (0.1·ℓ_j)²). Under the spec's own length-scale prior (ℓ ≈ 14 at d = 12), that is a standard deviation of about 1.4, so most children clip to 0 or 1. Resolution: σ_j = min(0.1·ℓ_j, 0.1).
3. **The noise floor's form.** The spec gives a floor of 1e-3 without saying how. Resolution: σ_n² = 1e-3 + e^θ, with the spec's prior on θ. It is smooth, so the gradient exists everywhere.
4. **Categorical genes.** Schema v1 has none searched, and the parser rejects `gp` categoricals. Resolution: no overlap-kernel code in 05b (YAGNI). 0024 specifies what the schema decision that activates the first one must add:
   - the kernel factor exp(−Σ_c 1[x_c ≠ x'_c]/ℓ_c), with ln ℓ_c ~ N(0, 1) [guess];
   - uniform levels for uniform candidates, and the parent's level kept with p 0.9 for local ones;
   - a posterior schemaVersion bump.

   A schema change is a fresh fit anyway, so nothing is lost.
5. **"≥ 100 rows since the last confirmation" before the first one.** Resolution: count from the signal look's `lives`.
6. **The progress chart (spec §6).** The page layout has needed three fixes already. Resolution: recipe and score columns in HISTORY, plus a Ladder-style "Current recipe" line. The chart waits for a later plan. **Open for the user:** is the table enough for now?
7. **"Force mode to search via a test-only path."** Any switch in `buildMeta` or `birthDraw` would be a production code path that could set `search` without the signal check. Resolution: there is no such path anywhere. The simulation reaches `search` with planted calibration data through the real `signal-check@1`. The e2e reaches it with a stubbed `posterior.json` served by a Playwright route, which exists only in `tests/e2e/`. **Open for the user:** confirm that "forced by data" meets "forced by a test-only path".
8. **`paused` mode.** 05a reserved it for 05b, but no 05b rule needs it: calibration stays `calibrate` with 6 repeats, and a failed fit just serves no model. Resolution: it stays reserved and is never written. The parser and the draw keep their 05a behaviour for it.
9. **The confirmation world seed.** The spec sets the learner seed to evalSeed but is silent on the training world. Resolution: the world seed stays the learnerId's. Only the checkup variants (`confirm/` namespace) and the learner init are fresh. Training worlds are already unique to each life.
10. **Confirmation-newborn files** (05a note). Resolution: role `confirm` with arm `newborn`, in the reference newborn's frozen shape (seq 0, cu 0, final, three checkups ahead of cu). This is what `confirm-fisher@1` already classifies, so its version does not change.
11. **The deadline needs a clock.** Resolution: `builtAt`, which `meta-cli.ts` already reads once, goes into confirmation files (`declaredAt`, `deadline`, `decidedAt`) and promotions, never into `posterior.json`. So tabs' draw seeds stay clock-free, and `buildMeta` stays a pure function of its input.
12. **Fixed genes after a promotion.** `checkGenome` holds fixed genes at the **space's** centre. A promoted `genome.json` can move only `gp` genes, so `fromUnit`'s non-searched genes are unchanged, as spec §4.5 ("ranges do not move") requires.

## Self-review
- **Spec coverage, §10 05b:**
  - the deterministic GP fit → Tasks 2–3;
  - `posterior.json` with index and fixtures → Tasks 1 and 8 (CI golden);
  - the cross-browser probe → Task 6;
  - TS draws in the tab → Tasks 4, 5 and 11;
  - re-verification → Task 8;
  - the incumbent and trigger → Tasks 4 and 8;
  - declaration and serving → Tasks 5 and 9;
  - the confirm evaluator wired to promotion → Task 9;
  - promotion records and the gate commit → Tasks 9–10;
  - page progress → Task 11 (reduced, conflict 6);
  - the live check → Task 13.
- **Spec §4.1:** the 1-in-5 uniform share is kept in search (Task 5), and confirm takes [0.2, 0.7) (Task 5).
- **Spec §4.2:** only the first 2,048 CU counts, unchanged (evaluators).
- **Spec §7.1:** every gate file has its path in `META_PATH_RE` (Task 10).
- **Spec §7.4:** seeds from protected namespaces (Tasks 5, 7); the TS recheck cost is bounded by the cache (Task 8).
- **05a ledger notes addressed:**
  - posterior bump and old tabs → Tasks 1 and 13;
  - confirmation-newborn shape → Task 7;
  - the centre in force at each birth → Task 8;
  - the gate filters confirmation lives → Task 9 (`armGroups`);
  - `META_PATH_RE` and `knowledge/genome.json` with both guards → Task 10;
  - the older-schema bypass → Task 7 (`spacesBySchema`);
  - need overshoot → Task 9 (in-flight). Calibration overshoot is moot once calibration has answered, and is noted in 0024;
  - `posteriorHistory` growth → Task 8 (`verified.json`);
  - the confirm arm from evalSeed mod 4 → Task 7 (parser) and Task 8 (verify).
- **Names across tasks:**
  - `GpModel`, `GpHyper`, `Incumbent`, `ServedConfirmation`, `ArmName`;
  - `cholesky`, `objective`, `conditionGp`, `predict`, `fitGp`;
  - `candidateSet`, `thompsonPick`, `incumbentOf`, `modelPosterior`;
  - `ARM_OF_J`, `armOfSeed`, `confirmSeed`, `spacesBySchema`, `centresBySha`;
  - `buildModel`, `incumbentFor`, `triggered`, `armGroups`, `decide`, `promotionErrors`, `runSim`.
- **Order:** 1 → 2 → 3 → 4 → 5 → 6. Task 7 needs Task 5 (`ArmName`). Task 8 needs Tasks 5 and 7. Then 9 → 10. Task 11 needs Tasks 5 and 7. Task 12 needs Tasks 10 and 11. Then 13. Task 6 can run beside Task 7.
- **Proportion:** signatures, constants and test names only. Algorithm detail is given only where the tests do not fix it: the line search, the jitter schedule, the candidate construction and rng order, the verification rule, and `need`.
