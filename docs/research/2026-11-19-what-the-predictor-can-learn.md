# What the predictor can learn: the close-up's surround, a shift gain, and a wide view that cannot tell where it is

**Date:** 2026-11-19
**Status:** offline measurements by a research agent (AGENTS.md invariant 5) for Plan 06a1 (`docs/plans/2026-11-19-06a1-what-the-predictor-can-learn.md`). Nothing here is live. Four seeds per condition unless said otherwise: first measurements, not confirmations. Every mechanism is a design change made by an agent, not something a learner discovered.

## In plain words

1. **The close-up's biggest error was built in, and is fixed.** When the cursor moves, part of the close-up is new, and the predictor filled it with black. The wide view already shows what is there. Filling from it takes the fresh-world score from 0.64–0.82 to 0.37–0.39 (probe part), and every seed now ends in the same place.
2. **A gain on the shift's weights gets there in a tenth of the time.** The same score after 2,500 steps as otherwise after about 20,000. The end is the same.
3. **The wide view can learn to scroll, but not where.** A masked shift of the wide view learns scrolling to within reach of an oracle (0.15–0.25 against 0.11), the first time the wide view has beaten "nothing changes". It then scrolls in every room, and is wrong there by more than it gains. It is not landed.
4. **The reason is the learner's own state.** The recurrent state that everything reads says which room is in front only weakly (a fitted readout finds a little over half of the scrolling room's steps). A mechanism that depends on the situation has nothing reliable to read. This is a new open question (Q36).
5. **Whole-life replay helps a curious learner, and nothing else yet.** It did not keep the scrolling skill alive, and it makes no difference to a random explorer.
6. **Curiosity is no longer ruinous, and is still worse than acting at random.** With the better predictor it no longer wrecks prediction, but it still chases big pixel changes (now along the screen's edge) and all but stops clicking things. A better predictor does not fix the drive.

## Method

- **Lives:** `tools/probe/introspect-probe.ts`: the real learner on the CPU backend at CPU widths, the real four-room world, caregiver and held-out set, 25,000 steps, seeds 1 to 4. Deterministic.
- **Score:** the fresh-world set v1 of decision 0030 (5,120 transitions in worlds no life lives in: 1,280 under random actions, 3,840 under a prober that walks to things and clicks them), as loss ÷ copy-last. Below 1 is better than assuming nothing changes. The wide view (periphery) and the close-up (fovea) are also scored apart, and by what the action did.
- **Reference:** Plan 06a's best learner, a random explorer (`curiosityWeight` 0) with the "fixed predictor" (`residualDecay` 0, 250 replays per sleep, the last 2,000 steps replayed). Every arm below is that learner plus the named mechanisms, unless said otherwise.
- **Compute:** a life's compute units (CU) are given with each table. The mechanisms cost compute per step, so a mechanism's 25,000 steps are more CU than the reference's.
- **Oracles** (§2) use the true cursor displacement or try every shift. They are privileged analysis of the test set, not learners.
- **Raw output:** `.local/06b/` in the worktree (not committed); the windows of every run are in `2026-11-19-what-the-predictor-can-learn/runs.json`. The mechanisms of §5 to §7 are not in the merged code; they are on branch `diag/06a1-wide-shift`.

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

## 5. A close-up shift that depends on the situation: worse, in both forms

`stateShift` was meant for the cases where the same action shifts the view in one situation and not in another (a move against the screen's edge; a scroll in a room that does not scroll). Two forms were tried: an added linear term in the predictor's hidden layer on the kernels and the gate, and then a product, the action's gate times a sigmoid of a linear map of the core's state.

| Learner (surround on) | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| without it, gain 1 | 0.67 / 0.38 | 0.67 / 0.37 | 0.68 / 0.38 | 0.69 / 0.39 |
| added term, gain 1 | 0.73 / 0.49 | 0.71 / 0.43 | 0.71 / 0.42 | 0.77 / 0.53 |
| added term, gain 16 | 0.86 / 0.56 | 0.74 / 0.43 | 0.75 / 0.43 | 0.75 / 0.44 |
| product gate, gain 1 | 0.68 / 0.38 | 0.73 / 0.44 | 0.74 / 0.48 | 0.77 / 0.54 |

- **Worse in eleven lives of twelve, and equal in one.** It is not landed.

## 6. A wide-view shift, first version: it learns to scroll, then unlearns it

`wideShift`, first version: the periphery may be translated (±3 cells per axis) in the cells a learned mask selects, behind a gate, with the kernels and the gate depending on the action plus an added term in the predictor's hidden layer. It costs 12% more compute per step than the surround alone.

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

## 7. The wide-view shift with a product gate: it scrolls well, everywhere

Second version: kernels chosen by the action alone, one mask, and a gate that is a product, the action's gate times a sigmoid of a linear map of the core's state ("this action, in this situation"). 6% more compute per step than the surround alone.

The wide view's scroll score, and in brackets its score on plain moves, at each 2,500 steps (**bold** as in §6):

| Replay | seed | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recent only | 1 | **0.96** (1.1) | **0.78** (1.6) | **0.44** (3.7) | 0.97 (1.1) | **0.19** (10.8) | **0.62** (2.2) | 0.72 (1.5) | 0.66 (2.2) | **0.18** (9.7) | **0.18** (9.5) |
| recent only | 3 | **0.80** (2.5) | **0.56** (4.7) | **0.32** (18.7) | **0.50** (8.2) | **0.55** (5.8) | 0.84 (11.3) | **0.46** (14.3) | **0.34** (14.6) | **0.43** (18.4) | 0.84 (11.2) |
| half whole-life | 3 | **0.97** (1.2) | 0.88 (1.3) | **0.93** (1.2) | 0.97 (1.1) | **0.32** (6.1) | **0.45** (4.6) | **0.21** (7.2) | 0.95 (1.1) | **0.81** (1.3) | **0.18** (7.8) |
| half whole-life | 4 | **0.94** (1.4) | **0.47** (7.6) | **0.41** (6.9) | **0.33** (8.1) | **0.19** (10.3) | **0.18** (10.0) | **0.16** (10.3) | **0.15** (10.5) | **0.15** (10.3) | **0.21** (8.0) |

Mean wide-view score over the ten checkpoints of each life (probe part; random part):

| Arm | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| recent only, gain 1 | 0.79; 1.22 | 0.90; 0.99 | 1.78; 1.23 | 0.89; 1.08 |
| recent only, gain 4 | 0.83; 1.25 | 0.90; 1.00 | 1.07; 1.03 | 0.90; 1.01 |
| half whole-life, gain 1 | 0.93; 0.98 | 0.90; 1.04 | 0.84; 1.11 | 0.70; 1.78 |
| half whole-life, gain 4 | 0.94; 1.03 | 0.95; 1.01 | 0.81; 1.16 | 0.90; 0.91 |

- **It learns the scroll itself well:** 0.15 to 0.25 on scrolls, against the oracle's 0.11.
- **Whenever it scrolls well it is wrong on plain moves, by 4 to 19 times.** The two rise and fall together in every life.
- **It fires on scroll actions in every room.** A short life with the half whole-life arm, scored by starting room after 7,500 steps: plain moves whose action held a scroll score 88 in the crib, 77 in toys, 19 in places and 79 in hide-and-seek itself (the page was at its end), and moves without a scroll action score 1.2 to 1.4 everywhere. The state gate, read from the same life's weights at 5,000 steps, was 0.82 to 0.86 on average in all four rooms.
- **Whole-life replay does not hold the skill either.** In seed 3 the scroll score goes from 0.21 back to 0.95 within one window away, as it does with recent-only replay. Seed 4 keeps it (0.15 to 0.21 over its last 12,500 steps) only because the caregiver brought the scrolling room back in every window. And replaying every room together does not teach the gate where to fire: the wide view as a whole is no better (random part 0.91 to 1.78).
- **What looks like forgetting is the misfire being corrected.** In the other rooms every scroll action is a large error, and the quickest way to remove it is to lower the action's gate, which removes the skill everywhere.
- **So the wide view still has no mechanism that beats copy-last over a life.** This one is not landed.

**Why the gate does not learn where it is.** It reads the core's recurrent state, so the question is whether that state says which room is in front. `tools/probe/room-readout.ts` fits a linear readout of the room on half of the fresh-world trajectories and scores it on the other half, with the cursor's position as a control (random explorer, surround on, 25,000 steps):

| | seed 1 | seed 4 |
|---|---|---|
| which of four rooms (chance 0.25) | 0.54 | 0.56 |
| hide-and-seek or not: right (always "not" gives 0.75) | 0.84 | 0.79 |
| … share of hide-and-seek transitions found | 0.59 | 0.56 |
| control: the cursor's x, y (variance explained) | 0.91, 0.98 | 0.06, 0.11 |

- **The state says which room only weakly.** A readout fitted for the purpose finds a little over half of the hide-and-seek transitions. A gate whose every wrong firing costs some 80 times a plain move's error needs far better than that.
- **What the state holds differs between lives.** In one life the cursor's position reads out almost exactly, in the other hardly at all.
- **Limits of this check:** two lives, a linear readout, and 64 trajectories to fit on. It shows the room is not plainly there; it does not show it could not be extracted another way.
- **This matters beyond the wide view.** The policy and the value read the same state. Under a random explorer nothing trains it except what the predictor's output layers pass back, which is little.


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

- **Two mechanisms are landed, off by default:** the close-up's surround (`foveaSurround`) and the shift gain (`shiftGain`). Together they are a predictor that reaches 0.36–0.38 on the fresh-world probe part where the reference reached 0.64–0.82, and reaches 0.40–0.44 within a tenth of a life.
- **The reference to beat changes.** A random explorer with the surround ends at 0.67–0.69 on the random part and 0.37–0.39 on the probe part (1,003 CU). An interest signal should be judged against that, with the same predictor.
- **What is left in the close-up is mostly events.** With plain moves at 0.19–0.21, clicks that respond (0.88–0.90), widget changes (0.89–0.91) and scrolls (0.73–0.76) are now a large part of the remaining error. Before, they were hidden under the error of every move.
- **The wide view is still unlearned,** and the reason is now specific: translation is expressible and learnable (§7), but nothing tells a mechanism where it applies.
- **A new open question, Q36: what the core's state represents.** A gate, a policy or a value that should behave differently in different places needs a state that says where it is.
- **Whole-life replay helps the curious learner (§8) and did not rescue the wide-view skill (§7).** Under a random explorer it still makes no difference (06a, and the second row of §8's table).
- **Curiosity is unchanged as a drive** (§8). Plan 06b stands as planned.

## What this does not establish

- **Four seeds, one world, half width, a random explorer.** These say what the predictor can learn from broad experience at CPU widths. GPU widths, live tabs and other policies are untested.
- **The gain, the kernel sizes and the mask prior are single values,** chosen by arithmetic and not tuned.
- **The oracles are ceilings for these mechanisms, not for prediction.** A predictor that also handled in-place change (a press or a hover recolouring what is under the cursor; clicks that respond and widget changes are 8% of the close-up's error) could go lower.
- **Nothing here is an interest signal.** A curious learner with this predictor still explores worse than a random one.
- **Frames differ slightly between platforms** (0030's correction), so these scores are comparable among themselves and not yet with a browser's.

## Sources

- Code, landed: `packages/learner/src/encode.ts` (`foveaSurround`), `shift.ts` (`translateSurround`, the gain), `model.ts` (`ModelOptions`), `learner.ts` (`MECHANISMS`, `shiftGain`); tests beside them and `packages/learner/test/gpu/learning.gpu.test.ts`.
- Code, not landed: `wide-shift.ts` and the state gates, with their tests, on branch `diag/06a1-wide-shift` (its last commit has the second versions; `f56268b` on it has the first).
- Probes: `tools/probe/introspect-probe.ts` (`--foveaSurround 1 --shiftGain 16 --reservoirShare 0.5 --dumpParams file`; on the branch also `--stateShift 1 --wideShift 1`), `tools/probe/room-readout.ts`.
- Runs: `2026-11-19-what-the-predictor-can-learn/runs.json` (66 lives). The arms of §5 to §7 are reproducible only from the branch.
- Earlier notes: `2026-11-17-predictor-fixes-and-fresh-world-test.md` (the reference, the fresh-world test), `2026-11-17-task-and-target-introspection.md` (curiosity tracks pixel change).
- Decisions: 0015 (fovea-shift prior), 0018 (genes, centre off), 0021 (bit-identical when off), 0029, 0030, 0031 (proposed).


