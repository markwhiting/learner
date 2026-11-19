# Plan 06a1 — What the predictor can learn

## In plain terms

Plan 06a ended on a finding: almost everything the predictor learns is one built-in skill (the close-up shifts when the cursor moves), and on the wide view it has never beaten "nothing changes" (`docs/research/2026-11-17-predictor-fixes-and-fresh-world-test.md`, open question Q35). An interest signal built on the predictor's error has little to work with while that holds. The user's instruction was to make the fixes and keep moving (2026-11-19).

This plan gives the predictor ways to express the changes this kind of world is made of, as optional mechanisms a genome can switch on, and keeps only those that measure well:

- **Landed: what a move brings into the close-up comes from the wide view.** The close-up is a window onto the screen, and the wide view already shows, coarsely, what lies around it. Today the shift fills the incoming strip with black.
- **Landed: the shift's weights can count for more per update,** so a kernel sharpens early in a life instead of late.
- **Built and measured, not landed: part of the wide view moving as one piece** (a page scrolling), and **a shift that depends on the situation.** Both need the learner to know which situation it is in, and its recurrent state turns out not to say.

Each landed mechanism is off unless a setting turns it on, and the settings are not in the genome yet. Nothing here changes what a live learner does.

**Goal:** the mechanisms that work exist in `packages/learner`, are tested (bit-identical when off, gradients checked, the same on WebGPU), are measured offline on the fresh-world test against Plan 06a's reference, and are proposed as genes for the genome schema that goes live with Plan 06d. What did not work is written up with its cause.

**Spec and decisions:** decision 0029 (user) sets the direction; 0018 (user) says improvements enter as genes whose centre is "off" and are found by the meta-learner, not fixed by hand; 0021's rule holds (bit-identical when off, a gene only under a new schema version). Decision 0031 (agent's calls, proposed) records the design calls made here. These are a research agent's design changes, not something a learner discovered (AGENTS.md invariant 5).

## Tasks

- [x] **1. Find where the change is.** On the fresh-world set, which events hold the copy-last error in each view, and what an oracle that knows the true motion could reach (`docs/research/2026-11-19-what-the-predictor-can-learn.md` §1–2).
- [x] **2. The close-up's surround.** `LearnerSettings.foveaSurround` (optional; absent or 0 is off). `encode.ts` `foveaSurround`: a 32×32 image at the close-up's target scale, centred on the cursor, whose middle is the fovea target and whose margin is the periphery around it (black off the screen). `shift.ts` `translateSurround`: the fovea shift reads from it, so what a move brings in is the wide view's content. No new parameters.
- [x] **3. The shift gain.** `LearnerSettings.shiftGain` (optional; absent is 1, clamped to 1–64): the shift kernels' learned logits count that many times over. No new parameters.
- [x] **4. A wide-view shift and a situation-dependent gate: built, measured, not landed.** Two versions of each (an added term in the predictor's hidden layer; then a product with a gate on the core's state). The wide shift learns to scroll (0.15–0.25 on scrolls against an oracle's 0.11) but fires on scroll actions in every room; the gate on the close-up's shift measured worse in both forms. The code and its tests are on branch `diag/06a1-wide-shift` (note §5–7).
- [x] **5. Why: the room cannot be read from the core's state.** `tools/probe/room-readout.ts` (note §7). Open question Q36.
- [x] **6. Tests.** Off: the golden trajectory, the RNG stream and saved states are unchanged. On: every newborn still predicts exactly copy-last; gradient checks on the shift and on the whole predictor; save and resume bit-identical; no leaked buffers; the same learning on WebGPU as on the CPU (`packages/learner/test/gpu/learning.gpu.test.ts`).
- [x] **7. The built-in curiosity with the better predictor** (note §8): no longer ruinous, still worse than a random explorer.
- [x] **8. Write up.** `docs/research/2026-11-19-what-the-predictor-can-learn.md`.
- [ ] **9. Genome schema 2** (with Plan 06a task 9 and Plan 06d): `foveaSurround` as an off/on gene with its centre at off, `shiftGain` on a log scale with its centre at 1.

## Not in this plan

- **A state that says where the learner is** (Q36): what the wide-view shift and anything else that depends on the situation need first.
- **In-place change** (a press or a hover recolours what is under the cursor without moving it). The note sizes it; no mechanism is built.
- **Interest signals** (Plan 06b), **lives outside the browser that count** (Plan 06c), **the new search evaluator and the calibration restart** (Plan 06d).
