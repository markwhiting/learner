# 0031 — Plan 06a1 calls: two optional predictor mechanisms (the close-up's surround, a shift gain); a wide-view shift is not landed

- **Date:** 2026-11-19
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (agent's calls, not yet confirmed). The instruction to make the predictor's fixes is the user's (2026-11-19: "make the fixes", "keep moving forward"), after 0029.
- **Resolves:** the implementation choices of `docs/plans/2026-11-19-06a1-what-the-predictor-can-learn.md`; part of Q35

## Context

Plan 06a found that almost all the predictor learns is the fovea-shift prior of 0015, and that it never beats copy-last on the wide view (Q35). 0029 makes prediction on fresh worlds the first competence and wants interest signals judged by it, and a signal built on the predictor's error has little to work with while the predictor cannot learn what happens on the screen.

0018 (user) says improvements enter the design space as genes whose centre is "off" and are found by the meta-learner, not fixed by hand. 0021 says a mechanism lands bit-identical when off and becomes a gene only under a new schema version. Invariant 5 says an agent's design change is not something a learner discovered. Everything below is an agent's design change, entered that way.

## Decision

**Agent's calls (not yet confirmed):**

- **Two optional learner settings, neither in `DEFAULT_SETTINGS`.** Absent (or 0, or gain 1) the learner is bit-identical to before: same golden trajectory, RNG stream and saved state, and `LEARNER_STATE_VERSION` does not change. They are fixed for a life. Neither adds a parameter, so the checkpoint's parameter list is the same with them, and every newborn still predicts exactly copy-last.
- **`foveaSurround`: the fovea shift reads from the wide view around the close-up.**
  - `encode.ts` `foveaSurround` builds a 32×32 image at the close-up's target scale, centred on the cursor: the middle is the fovea target, each cell around it is the periphery cell under its centre, and cells off the screen are black as the fovea's own padding is.
  - **This adds a prior about the learner's own sensors, and nothing about the world:** the fovea is the window centred on the cursor position, and the periphery is the whole screen scaled down. Both are in `packages/interface` already, and the cursor position is a declared observation. No new modality or reward (invariant 2). Which action moves the view by how much stays learned (0015).
- **`shiftGain`: the fovea shift's learned kernel logits count that many times over** (1 to 64; absent is 1). An update moves a weight by about the learning rate at most, and one action's kernel row is updated on one step in 25, so at gain 1 a kernel takes most of a life to outgrow the prior that keeps it centred.
- **Proposed genes for genome schema 2** (Plan 06a1 task 9, with Plan 06a task 9 and Plan 06d): `foveaSurround` as off/on with the centre at off; `shiftGain` on a log scale from 1 with the centre at 1. Whether either is switched on in lives is the meta-learner's finding to make (0018), not this decision's.
- **A wide-view shift and a situation-dependent gate were built and measured, and are not landed.** A mechanism that is known to fire in the wrong place is not offered to the search. The code, its tests and the runs' settings are on branch `diag/06a1-wide-shift`, and the note (§5–7) has the measurements and the cause: the core's recurrent state does not say which room is in front (Q36).

## Evidence

Offline, four seeds, 25,000 steps, a random explorer with Plan 06a's fixed predictor (`docs/research/2026-11-19-what-the-predictor-can-learn.md`):

- **The surround** takes the fresh-world score on the probe part from 0.64–0.82 to 0.37–0.39 and removes the spread between seeds, for 3.4% more compute per step.
- **The gain** (16) brings the learner to 0.40–0.44 after 2,500 steps, where it otherwise is after about 20,000. The end is the same, and the path is less steady.

## Alternatives considered

- **A convolutional output head on the first conv layer's features** (Q35's first candidate). The measurements of where the change is (the note, §1–2) pointed elsewhere first: most of the close-up's error is what a move brings in, and most of the wide view's is scrolling. Both are translations, which a translation prior expresses with a few hundred parameters and which a small conv head would have to rediscover. Not built.
- **Letting the learner find where the fovea sits in the periphery** (a learned crop). Four numbers that the sensor's definition already fixes, bought with a differentiable sampler the tensor library does not have.
- **Black or mean-colour fill.** Measured by oracle: black fill is 9 times worse than copy-last on moves, mean colour no better than copy-last.
- **Raising the learning rate of the shift's weights through a third optimizer.** A gain inside the mechanism is one number with the same effect and no change to the checkpoint's optimizer layout.
- **Landing the wide-view shift switched off, as an experimental setting.** Every later change would have to keep it bit-identical and tested, for a mechanism that cannot work until the learner's state says where it is.

## Consequences

- Nothing is live: no setting is in the genome, and no life uses them. A push of this code changes `codeSha`, so open tabs reload once (0014) and carry on unchanged.
- The surround costs about 3% more compute per step (CPU widths); the gain costs nothing.
- **The wide view is still where it was:** no landed mechanism changes it, and the predictor still never beats copy-last there.
- In-place change (a press or a hover recolouring what is under the cursor) has no mechanism yet. It is 5 to 8% of the close-up's error on the fresh-world set.
- The measurements are offline, at CPU widths, on four seeds, with a random explorer. They say what the predictor can learn from broad experience, not what any interest signal will make of it.
