# What the predictor can learn: the close-up's surround, a wide-view shift, and forgetting

**Date:** 2026-11-19
**Status:** offline measurements by a research agent (AGENTS.md invariant 5) for Plan 06a1 (`docs/plans/2026-11-19-06a1-what-the-predictor-can-learn.md`). Nothing here is live. Four seeds per condition unless said otherwise: first measurements, not confirmations. Every mechanism is a design change made by an agent, not something a learner discovered.

## In plain words

@@SUMMARY@@

## Method

- **Lives:** `tools/probe/introspect-probe.ts`: the real learner on the CPU backend at CPU widths, the real four-room world, caregiver and held-out set, 25,000 steps, seeds 1 to 4. Deterministic.
- **Score:** the fresh-world set v1 of decision 0030 (5,120 transitions in worlds no life lives in: 1,280 under random actions, 3,840 under a prober that walks to things and clicks them), as loss ÷ copy-last. Below 1 is better than assuming nothing changes. The wide view (periphery) and the close-up (fovea) are also scored apart, and by what the action did.
- **Reference:** Plan 06a's best learner, a random explorer (`curiosityWeight` 0) with the "fixed predictor" (`residualDecay` 0, 250 replays per sleep, the last 2,000 steps replayed). Every arm below is that learner plus the named mechanisms, unless said otherwise.
- **Compute:** a life's compute units (CU) are given with each table. The mechanisms cost compute per step, so a mechanism's 25,000 steps are more CU than the reference's.
- **Oracles** (§2) use the true cursor displacement or try every shift. They are privileged analysis of the test set, not learners.
- **Raw output:** `.local/06b/` in the worktree (not committed); the windows of every run are in `2026-11-19-what-the-predictor-can-learn/runs.json`.

## 1. Where the change is

Share of the fresh-world set's copy-last error by what the action did (`.local/06b/fresh-anatomy.ts`):

| What the action did | transitions | share of the wide view's error | share of the close-up's error |
|---|---|---|---|
| plain cursor move | 4,256 | 8% | 81% |
| scroll | 275 | 64% | 9% |
| reveal (most are scrolls that uncover something) | 61 | 14% | 2% |
| room switch | 23 | 6% | 0.5% |
| click that made something respond | 216 | 4% | 5% |
| navigation | 51 | 2% | 0.4% |
| widget change | 111 | 1% | 3% |

- **The close-up is 83% of the combined error and the wide view 17%,** by the weights the learner trains with.
- **In the close-up, plain cursor moves are most of it.** Whatever limits the prediction of a move limits the whole score.
- **In the wide view, scrolling is most of it.** One room (hide-and-seek) scrolls its whole page by 16 px for any scroll action, wherever the cursor is. Under random actions, scrolls are 89% of the wide view's error and 36% of the close-up's.

## 2. What an oracle could reach

**Close-up.** The fovea-shift prior of 0015 translates the current close-up and fills what comes in with black. With the true cursor displacement (`.local/06b/fovea-oracle.ts`; ratios to copy-last):

| Prediction | all | plain moves | 16 px moves |
|---|---|---|---|
| shift, black fill (the prior's translation alone) | 8.77 | 9.65 | 10.48 |
| shift, fill with the close-up's mean colour | 1.15 | 1.14 | 1.06 |
| shift, fill from the wide view | 0.73 | 0.63 | 0.52 |
| the same two with the cursor arrow kept in place: mean colour | 0.69 | 0.63 | 0.65 |
| … from the wide view | **0.35** | **0.21** | **0.15** |

- **Most of a large move's close-up is new.** A 16 px move brings in half of the 32 px window, or three quarters for a diagonal one. The close-up alone cannot know what is there. The wide view shows it, at a quarter of the resolution.
- **The reference's 0.55 to 0.78 on plain moves is about what mean-colour fill allows (0.63).** It was at its ceiling. Filling from the wide view moves the ceiling to 0.21.

**Wide view.** Trying every vertical shift of up to 3 cells on each transition (`.local/06b/wide-oracle.ts`):

| Prediction | scroll | reveal | all |
|---|---|---|---|
| the best shift of the whole frame | 0.93 | 0.95 | 0.95 |
| the best shift of the rows above the taskbar only | **0.11** | **0.22** | **0.30** |

- **A shift has to be told where it applies.** Shifting the taskbar with the page costs almost as much as the shift gains.
- **A masked shift could take the wide view from 1.00 to about 0.30.**

## 3. The close-up's surround

`foveaSurround`: the fovea shift reads from a 32×32 image centred on the cursor whose middle is the close-up and whose margin is the wide view around it, so what a move brings in is the wide view's content instead of black. It adds no parameters, and uses only what the observation already declares (both frames and the cursor position) plus the fixed geometry of the two views.

Fresh-world score, both views, after 25,000 steps (random part / probe part):

| Learner | CU | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| reference | 970 | 0.83 / 0.64 | 0.83 / 0.67 | 0.83 / 0.64 | 0.95 / 0.82 |
| + surround | 1,003 | 0.67 / 0.38 | 0.67 / 0.37 | 0.68 / 0.38 | 0.69 / 0.39 |
| + surround, at 22,500 steps | 902 | 0.69 / 0.40 | 0.69 / 0.38 | 0.67 / 0.39 | 0.69 / 0.41 |

- **The probe part goes from 0.64–0.82 to 0.37–0.39,** and at less compute than the reference (902 CU) it is 0.38–0.41.
- **The spread between seeds is gone.** The reference's seed 4 never got its shift going (0.82). With the surround all four seeds end within 0.02 of each other.
- **Plain cursor moves in the close-up score 0.19–0.21** (reference 0.55–0.78), which is the oracle's 0.21.
- **The wide view is untouched** (1.00–1.01), as expected: nothing here changes it.
- **Cost:** 3.4% more compute per step.

## 4. The shift gain

`shiftGain`: the shift kernels' learned logits count that many times over. One update moves a weight by about the learning rate at most, and a kernel only sharpens once its logits outgrow the prior that keeps it centred. One action's row is updated on one step in 25, so at gain 1 that takes most of a life.

Probe part, both views, along the life (surround on):

| Steps | 2,500 | 5,000 | 10,000 | 15,000 | 20,000 | 25,000 |
|---|---|---|---|---|---|---|
| gain 1, seeds 1–4 | 0.66–0.73 | 0.57–0.65 | 0.46–0.55 | 0.43–0.46 | 0.39–0.43 | 0.37–0.39 |
| gain 16, seeds 1–4 | 0.40–0.44 | 0.39–0.50 | 0.38–0.48 | 0.39–0.44 | 0.37–0.42 | 0.36–0.38 |

- **With the gain the learner is at 0.40–0.44 after 2,500 steps (100 CU),** where it otherwise is after about 20,000 (800 CU). The end is the same.
- **It is less steady:** two of the four seeds wander up to 0.48–0.50 for a while before coming back. 16 is one value, picked by arithmetic, not tuned.

## 5. A shift that depends on the situation, first version: worse

`stateShift`, first version: the fovea shift's kernels and gate got an added linear term in the predictor's hidden layer. It was meant for the cases where the same action shifts the view in one situation and not in another (a move against the screen's edge; a scroll in a room that does not scroll).

| Learner (surround on) | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| without it, gain 1 | 0.67 / 0.38 | 0.67 / 0.37 | 0.68 / 0.38 | 0.69 / 0.39 |
| with it, gain 1 | 0.73 / 0.49 | 0.71 / 0.43 | 0.71 / 0.42 | 0.77 / 0.53 |
| with it, gain 16 | 0.86 / 0.56 | 0.74 / 0.43 | 0.75 / 0.43 | 0.75 / 0.44 |

- **Worse in all eight lives.** That version is removed. §7 has its replacement.

## 6. The wide-view shift, first version: it learns to scroll, then unlearns it

`wideShift`, first version: the periphery may be translated in the cells a mask selects, with the kernels and the gate depending on the action plus an added term in the predictor's hidden layer.

The wide view's scroll score at each 2,500 steps of four lives (gain 1; **bold** = the learner spent part of the preceding 2,500 steps in the room that scrolls):

| seed | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **0.66** | **0.59** | 0.88 | 0.91 | 0.89 | 0.95 | **0.84** | **0.32** | 0.76 | 0.84 |
| 2 | **0.85** | **0.87** | **0.87** | **0.86** | **0.83** | 0.93 | **0.77** | **0.82** | 0.91 | 0.95 |
| 3 | **0.70** | **0.57** | **0.54** | 0.88 | **0.32** | **0.47** | 0.62 | **0.46** | **0.40** | 0.59 |
| 4 | **0.82** | **0.71** | **0.78** | 0.92 | 0.91 | **0.62** | **0.45** | **0.47** | **0.42** | **0.41** |

- **This is the first time the wide view has beaten copy-last** in any offline run: 0.32 to 0.6 on scrolls while the learner is in the room that scrolls.
- **Away from that room the skill goes.** Within 2,500 to 7,500 steps elsewhere the score is most of the way back to 1 (0.59 to 0.95), and it is relearned on return.
- **It misfires on plain moves:** their wide-view score is 1.3 to 3.1 (on a small error, so the whole wide view still ends at 0.66–0.99 on the probe part).
- **A higher gain and whole-life replay made it worse, not better.** At gain 16 the mean wide-view score over a life was 0.93–1.17. With half the replay memory a whole-life reservoir it was 0.95–1.05, and scrolling was then never learned well (0.64–0.99).
- **So this is not plain forgetting.** The gate could not say "a scroll action, in this room". Replaying only the current room it learned "a scroll action scrolls", which is wrong everywhere else, so it was unlearned there. Replaying every room at once it learned the average, which is weak everywhere.

@@WIDE2@@

## 8. The built-in curiosity with the better predictor: no longer ruinous, still worse than random

Plan 06a found that with the decay off and recent-only replay, the built-in curiosity wrecked the predictor (1.3 to 2.7 times worse than copy-last, up to 5 times on the fresh-world test). The same comparison with the surround on, same predictor in every arm (fresh-world score, random part / probe part):

| Learner (surround on, gain 1) | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| random explorer, recent-only replay | 0.67 / 0.38 | 0.67 / 0.37 | 0.68 / 0.38 | 0.69 / 0.39 |
| random explorer, half whole-life replay | 0.68 / 0.39 | 0.70 / 0.41 | 0.67 / 0.38 | 0.67 / 0.40 |
| curiosity, recent-only replay | 0.79 / 0.50 | 0.82 / 0.54 | 0.89 / 0.58 | 0.97 / 0.58 |
| curiosity, half whole-life replay | 0.75 / 0.45 | 0.74 / 0.44 | 0.76 / 0.43 | 0.80 / 0.55 |

What the policies did over their last 10,000 steps:

| | random explorer | curiosity |
|---|---|---|
| share of moves that are 16 px jumps | 0.31–0.32 | 0.66–0.83 |
| share of steps with the cursor at a screen edge | 0.10–0.16 | 0.45–0.70 |
| clicks that made something respond, per 1,000 steps | 5–10 | 0–6 |

- **Curiosity no longer wrecks the predictor.** Every curious life ends below 1 on both parts.
- **It is still worse than doing things at random,** in all eight lives, on both parts. The bar of 0029 (beat a random explorer at equal compute) is not met.
- **Whole-life replay helps the curious learner** (three seeds of four end at 0.43–0.45 on the probe part against 0.50–0.58) and makes no difference to the random one. That is 06a's pattern again: where replay comes from matters once experience is narrow.
- **It still chases pixel change, and has found a new place for it.** The curious policy makes big jumps and spends half its life against the screen's edge, where the close-up is half black padding and every jump changes it most. It all but stops clicking things.
- **So a better predictor does not fix the drive.** The reward is the loss a step's update removes, which is largest where the error is largest, and the largest errors are still the largest pixel changes. Which signal replaces it is Plan 06b's question (Q31).

## What this changes

@@CHANGES@@

## What this does not establish

- **Four seeds, one world, half width, a random explorer.** These say what the predictor can learn from broad experience at CPU widths. GPU widths, live tabs and other policies are untested.
- **The gain, the kernel sizes and the mask prior are single values,** chosen by arithmetic and not tuned.
- **The oracles are ceilings for these mechanisms, not for prediction.** A predictor that also handled in-place change (a press or a hover recolouring what is under the cursor, 5 to 8% of the close-up's error) could go lower.
- **Nothing here is an interest signal.** A curious learner with this predictor still explores worse than a random one.
- **Frames differ slightly between platforms** (0030's correction), so these scores are comparable among themselves and not yet with a browser's.

## Sources

- Code: `packages/learner/src/encode.ts` (`foveaSurround`), `shift.ts`, `wide-shift.ts`, `model.ts` (`ModelOptions`), `learner.ts` (`MECHANISMS`, `shiftGain`); tests beside them and `packages/learner/test/gpu/learning.gpu.test.ts`.
- Probe: `tools/probe/introspect-probe.ts` (`--foveaSurround 1 --stateShift 1 --wideShift 1 --shiftGain 16 --reservoirShare 0.5`).
- Runs: `2026-11-19-what-the-predictor-can-learn/runs.json`. The arms of §5 and §6 ran the first versions of `stateShift` and `wideShift`, which are not in the merged code.
- Earlier notes: `2026-11-17-predictor-fixes-and-fresh-world-test.md` (the reference, the fresh-world test), `2026-11-17-task-and-target-introspection.md` (curiosity tracks pixel change).
- Decisions: 0015 (fovea-shift prior), 0018 (genes, centre off), 0021 (bit-identical when off), 0029, 0030, 0031 (proposed).


