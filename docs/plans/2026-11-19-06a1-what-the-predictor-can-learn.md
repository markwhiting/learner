# Plan 06a1 — What the predictor can learn

## In plain terms

Plan 06a ended on a finding: almost everything the predictor learns is one built-in skill (the close-up shifts when the cursor moves), and on the wide view it has never beaten "nothing changes" (`docs/research/2026-11-17-predictor-fixes-and-fresh-world-test.md`, open question Q35). An interest signal built on the predictor's error has little to work with while that holds. The user's instruction was to make the fixes and keep moving (2026-11-19).

This plan gives the predictor ways to express the changes this kind of world is made of, as optional mechanisms a genome can switch on:

- **What a move brings into the close-up comes from the wide view.** The close-up is a window onto the screen, and the wide view already shows, coarsely, what lies around it. Today the shift fills the incoming strip with black.
- **Part of the wide view can move as one piece,** as when a page scrolls.
- **How the close-up shifts can depend on the situation,** not only on the action.
- **The shift's weights can count for more per update,** so a kernel sharpens within a life.

Each is off unless a setting turns it on, and the settings are not in the genome yet. Nothing here changes what a live learner does.

**Goal:** the mechanisms exist in `packages/learner`, are tested (bit-identical when off, gradients checked, the same on WebGPU), are measured offline on the fresh-world test against Plan 06a's reference, and are proposed as genes for the genome schema that goes live with Plan 06d.

**Spec and decisions:** decision 0029 (user) sets the direction; 0018 (user) says improvements enter as genes whose centre is "off" and are found by the meta-learner, not fixed by hand; 0021's rule holds (bit-identical when off, a gene only under a new schema version). Decision 0031 (agent's calls, proposed) records the design calls made here. These are a research agent's design changes, not something a learner discovered (AGENTS.md invariant 5).

## Tasks

- [x] **1. Find where the change is.** On the fresh-world set, measure which events hold the copy-last error in each view, and what an oracle that knows the true motion could reach (`docs/research/2026-11-19-what-the-predictor-can-learn.md` §1–2).
- [x] **2. The close-up's surround.** `LearnerSettings.foveaSurround` (optional; absent or 0 is off). `encode.ts` `foveaSurround`: a 32×32 image at the close-up's target scale, centred on the cursor, whose middle is the fovea target and whose margin is the periphery around it (black off the screen). `shift.ts` `translateSurround`: the fovea shift reads from it, so what a move brings in is the wide view's content. No new parameters.
- [x] **3. A shift that depends on the situation.** `LearnerSettings.stateShift`: the fovea shift's kernels and gate also take a linear term in the predictor's hidden layer. Three zero-initialised parameters.
- [x] **4. The wide-view shift.** `LearnerSettings.wideShift`, `wide-shift.ts`: the periphery may be translated (±3 cells per axis) in the cells a learned mask selects, behind a learned gate. Eight zero-initialised parameters.
- [x] **5. The shift gain.** `LearnerSettings.shiftGain` (optional; absent is 1): the kernels' and the mask's learned logits count that many times over.
- [x] **6. Tests.** Off: the golden trajectory, the RNG stream and saved states are unchanged. On: every newborn still predicts exactly copy-last; gradient checks on each mechanism and on the whole predictor; save and resume bit-identical; no leaked buffers; the same learning on WebGPU as on the CPU (`packages/learner/test/gpu/learning.gpu.test.ts`).
- [x] **7. Measure offline and write up.** `docs/research/2026-11-19-what-the-predictor-can-learn.md`.
- [ ] **8. Genome schema 2** (with Plan 06a task 9 and Plan 06d): the mechanisms as genes with their centre at "off", `shiftGain` with its centre at 1.

## Not in this plan

- **In-place change** (a press or a hover recolours what is under the cursor without moving it). The note sizes it; no mechanism is built.
- **Interest signals** (Plan 06b), **lives outside the browser that count** (Plan 06c), **the new search evaluator and the calibration restart** (Plan 06d).
