# Plan 06a — A predictor that keeps learning, and a fresh-world test

## In plain terms

Calibration came out flat because the learner barely learns, and the introspection note found why (`docs/research/2026-11-17-task-and-target-introspection.md`). The user then ruled (decision 0029) that the learner's drive stays intrinsic, that prediction on fresh worlds is the first competence to measure, and that lives may also run outside the browser.

This plan is the first step after those rulings. It builds the parts that do not depend on which interest signal wins:

- **The predictor's fixes, as mechanisms a genome can reach.** No decay on the predictor's residual, more replay, and a memory that can hold a sample of the whole life instead of only the recent past.
- **A test of prediction in worlds the learner has never lived in,** with enough of the rare events (a click that makes something respond, a navigation, a room switch) to score them separately. Every later interest signal is judged on it.
- **Offline measurements with both,** written up either way.

Nothing here changes what a live learner does. Each mechanism is off unless a setting turns it on, and the settings are not in the genome yet.

**Goal:** the mechanisms and the test exist in `packages/`, are tested, are measured offline, and are ready for the genome and evaluator changes of Plan 06d.

**Spec and decisions:** decision 0029 (user) sets the direction. Decision 0030 (agent's calls, proposed) records the design calls made here. 0021's rule holds: a mechanism lands with proof that the learner is bit-identical when it is off, and becomes a gene only under a new genome schema version. 0020's rule holds: a new search evaluator is a new version with its own decision.

## Tasks

- [x] **1. Whole-life replay in bounded memory.** `LearnerSettings.reservoirShare` (optional, absent or 0 is off): that share of the `bufferSize` slots is a reservoir, a uniform sample of every transition of the life so far; the rest stay the FIFO of recent transitions; a sleep replays uniformly over both. `packages/learner/src/replay.ts` (`Reservoir`, `reservoirSlots`, `sampleStored`), `learner.ts`, and `packages/harness/src/pack.ts` (checkpoints store the reservoir's frames compressed with the replay's). Off, the golden trajectory, the RNG stream and saved states are unchanged.
- [x] **2. Score any trajectory.** `TrainableLearner.scoreSet(set)`: the held-out scoring on a set the caller supplies, one loss per transition for each view (the wide periphery and the close-up fovea), forward only, billed to `evalCU`. A learner that scored a set continues bit-identically to one that did not.
- [x] **3. Ask the world what is at a point.** `World.inspect(x, y)`, privileged and harness only, like `state()`.
- [x] **4. The fresh-world set, version 1.** `packages/harness/src/fresh.ts`: 128 trajectories of 40 transitions (5,120 in all) in worlds of the harness's own seeds. 32 are driven by uniform random actions and 96 by a privileged prober that walks to things and clicks them. Each transition is labelled by what its action did (`switch`, `navigate`, `reveal`, `widget`, `respond`, `scroll`, `move`, `static`). `scoreFresh` groups a learner's losses by part and by label against copy-last, each view apart. A hash pins the set's content apart from its pixels (rooms, actions, labels, the cursor's path), which every platform builds alike; the frames are what each platform's canvas draws (corrected 2026-11-19: the first pin hashed the pixels too and failed in CI, where Linux renders a few anti-aliased pixels differently from macOS).
- [x] **5. The offline probe reports it.** `tools/probe/introspect-probe.ts --freshEvery N`, and `--reservoirShare`.
- [x] **6. Measure offline and write up.** `docs/research/2026-11-17-predictor-fixes-and-fresh-world-test.md`.
- [ ] **7. Checkups record the fresh-world score.** A new optional block on a checkup result, a log-schema and gate change, and the session scoring the set at each checkup. Not started: it belongs with the evaluator that reads it (Plan 06d).
- [ ] **8. Check the set in a real browser.** The pinned hash (rooms, actions, labels, cursor path) must come out of a browser too (a Playwright test). The pixels will not match exactly across platforms (macOS and Linux Node already differ in anti-aliased pixels), so before a tab's score is compared with an offline life's, measure how far one learner's scores differ between a browser-rendered and a Node-rendered set.
- [ ] **9. Genome schema 2.** Proposed ranges, not applied (changing `knowledge/genome-space.json` starts a new cohort and makes every life ineligible under `search-composite@1`, so it goes live together with the new search evaluator, Plan 06d):
  - `residualDecay`: reach 0 (a linear scale from 0, or fixed at 0).
  - `replayPerSleep`: up to one replay per step (250 at the default sleep interval).
  - `reservoirShare`: 0 to 1, a new gene.
  - `curiosityWeight`: reach 0, so that "no drive" is inside the box.

## Not in this plan

- **Interest signals** (Plan 06b): candidates developed offline and judged on the fresh-world test against a random explorer.
- **Lives outside the browser that count** (Plan 06c).
- **The new search evaluator and the calibration restart** (Plan 06d).
