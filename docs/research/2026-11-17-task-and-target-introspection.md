# What the learner is really optimising, and what to change

**Date:** 2026-11-17. **Asked by the user:** "introspect on the design of the task and training target … see if we could make tweaks that would get us closer to our goal."
**Follows:** `2026-11-16-calibration-flat.md` (why calibration came out flat) and `2026-11-16-filter-convergence.md`.
**Status:** analysis and offline measurements by a research agent (AGENTS.md invariant 5). Nothing here changes the product. Every proposal is an agent's proposal, not yet confirmed.

**Correction (2026-11-17, later the same day).** Point 3 below, section 3 and tweak 1 credit "replay drawn from the whole life". That arm changed how much is replayed and where it comes from at once, and a control run since separates them (`2026-11-17-predictor-fixes-and-fresh-world-test.md`):

- **For a random explorer, where replay comes from does not matter.** Replaying only the last 2,000 steps, at the same rate and with the decay off, does as well (0.51 and 0.65 against 0.51 and 0.64). The gain is from switching the decay off and replaying about once per step.
- **With curiosity on, it matters a great deal, and the decay turns out to be a real safeguard.** With the decay off and recent-only replay, the curiosity-driven learner's predictor ends 1.3 to 2.7 times worse than "nothing changes". Whole-life replay limits that.
- **The gain is narrower than this note says.** Nearly all of it is the close-up view shifting with the cursor (the fovea-shift prior). On the wide view the predictor never beats "nothing changes" in any run here.

## In plain words

1. **The learner is paid for making pixels change, so it learns to fling the cursor.** The "curiosity" reward is meant to be learning progress. As built, it tracks how much the picture changed. The policy's answer is to make the biggest diagonal jumps it can.
2. **It then forgets what it knew.** Over a 15-minute life, prediction on the learner's own stream keeps improving while prediction on a fixed, fair test gets worse. The learner specialises on its own flinging and loses its general skill.
3. **The predictor was built to forget, and the search could not undo that.** A decay that pulls the predictor back to "nothing changes" was added to limit the damage from point 2. With the decay off and replay drawn from the whole life, the predictor learns three to four times as much. Both settings lie outside the ranges the meta-learner was allowed to search.
4. **A "true" learning-progress reward is worse, not better.** Rewarding progress against an older copy of the model makes the policy collapse onto one or two moves within a few thousand steps: whatever it does most is what it improves at fastest, so that is what gets paid.
5. **The best learner available today is one with its drive switched off.** Acting at random, with the two predictor settings fixed, it predicts better than any life on record, and its behaviour stays a newborn's. On record, trained lives pass three of the five behaviour milestones less often than untrained newborns do.
6. **The policy can learn when the reward marks the right thing.** In a diagnostic where the harness paid for "a click made something respond", one run in four learned inside one life to sit on a shape and click it, ten times the newborn rate. The drive is the binding problem; reliability comes second.
7. **The score does not measure the question.** The project asks whether retaining, consolidating and reusing experience makes later learning cheaper. A life is scored on the spontaneous behaviour of a newborn in its first 2,048 CU. Nothing in that score has an "earlier" and a "later".

Flat was the right answer from the meta-learner: nothing inside its box mattered, because the box was drawn around a learner that does not learn. The approach (one meta-learner over a genome) is not refuted. The box and the score were wrong.

## Method

- **Live data:** the 106 search lives at GPU widths that reached their 2,048 CU budget (`data/runs` at `bdcec01`): their logged curve points and checkup measures, binned by CU.
- **Offline lives:** `tools/probe/introspect-probe.ts` on the branch `diag/learner-introspect` (worktree `.worktrees/introspect`; not merged). It runs the real learner on the CPU backend in the real four-room world, with the real caregiver and the real 64-transition held-out set, for 25,000 steps (the length of a live life). It is deterministic: two runs with the same inputs gave identical output. Every run's windows are in `2026-11-17-task-and-target-introspection/runs.json` (48 runs).
  - **Widths:** CPU widths (half size) unless marked. Two runs at GPU widths, the live size, agree with them.
  - **Instruments (harness side only, never fed to the learner):** the reward on each transition, the pixel change on it, what the action did (privileged trace), and the policy's action mix.
  - **Broad held-out set:** 800 transitions under uniform random actions in a differently seeded world, carried to a random room every 40 steps, so all four rooms are covered. The live held-out set is 64 transitions in one room. Ratios below are prediction loss ÷ copy-last ("nothing changes") on this set: below 1 is better than guessing no change.
- **Experimental switches** (on the scratch branch only, behind environment variables, off by default): a lagged learning-progress reward, on the fast and on a slow loss scale; an expected-surprise reward (section 7); a fovea-shift prior that reads only the move part of the action; an extra reward supplied by the harness.
- **Seeds:** 2 or 3 per condition. Treat every offline number as a first measurement, not a confirmation.

## What is happening

### 1. The reward pays for pixel change

The reward is `(loss before − loss after one update on the same transition) ÷ running loss` (`packages/learner/src/learner.ts`, `curiosityReward`). One gradient step on a transition lowers the loss on that same transition whether or not anything general was learned. So the reward is roughly the size of the error, and the error is roughly how much the picture changed.

| measured on the default learner (3 seeds, 25,000 steps each) | value |
|---|---|
| rank correlation of reward with pixel change, per 2,500-step window | 0.61–0.86 on two seeds, 0.27–0.61 on the third |
| share of steps with a positive reward | 94–99% |
| mean reward for a move of 1 px / 4 px / 16 px | 0.005 / 0.007 / 0.016 |
| mean reward for a click that made something respond | 0.011 |
| mean reward for a taskbar switch / a page navigation | 0.044 / 0.034 |
| how often a random policy gets a response / a taskbar switch / a navigation | 8 / 0.7 / 0.4 per 1,000 steps |

- A 16 px move pays more than a click that works, and it is available on every step with no skill.
- The events the milestones care about pay 2–3 times a big move, but they come once in a thousand steps and need several precise steps in a row.

**What the policy does with that.** The share of 16 px moves rises from the uniform 33% to 47–69% by the end of a life, mostly the four diagonals. The button and scroll parts of the policy stay at uniform: it learns nothing about clicking. At GPU widths the share reaches 55%.

**The live lives show the same.** Pixel change per step (the logged `baselineLoss`) more than doubles over a life: median 0.0077 in the first 64 CU, 0.0177 in the last 256. Across the 106 lives, more pixel change late in life goes with lower M2 (ρ −0.28), M3 (−0.17) and M5 (−0.21) and higher M4 (+0.20).

### 2. The learner specialises and forgets

From the 106 live lives (median across lives of each life's median in the bin):

| CU | held-out ratio (fixed test, random actions) | own-stream ratio | lives better than copy-last on held-out |
|---|---|---|---|
| 0–64 | 0.988 | 0.969 | 75% |
| 128–256 | 1.002 | 0.884 | 49% |
| 512–768 | 1.010 | 0.838 | 46% |
| 1,024–1,280 | 1.103 | 0.809 | 35% |
| 1,792–2,064 | 1.132 | 0.780 | 31% |

- On its own stream the learner keeps getting better. On the fixed test it gets worse and ends worse than a newborn.
- M1 ("predicts its world") is scored on the learner's own policy. It passes on 35% of checkups, and its measure follows the own-stream ratio (ρ +0.45), not the held-out one (+0.05). So M1 passes are mostly the specialisation, not general prediction.
- This is forgetting in the plain sense: a general skill present at 100 CU is gone by 1,000 CU. It is the phenomenon the project set out to study, happening inside every life, and no score tracks it as such.

### 3. The predictor is held back by its own safeguards

With the policy held random (curiosity weight 0), so that only the predictor's training differs. Broad held-out ratio at 25,000 steps, two seeds:

| predictor training | seed 1 | seed 2 | CU used |
|---|---|---|---|
| default: decay 1e-3, replay 64 per sleep from the last 2,000 steps | 0.86 | 0.91 | 671 |
| no decay | 0.84 | 0.82 | 671 |
| replay 250 per sleep from the whole life | 0.84 | 0.92 | 970 |
| **no decay and whole-life replay** | **0.51** | **0.64** | 970 |
| the same, with the shift prior reading only the move | 0.60 | 0.64 | 970 |

- Neither change does much alone. Together they take the gain over copy-last from about 0.12 to about 0.43, and the curve is still falling at 25,000 steps.
- On the live kind of test (64 transitions) the combined setting ends at 0.65 and 0.75. The best live life on record is at 0.83 and the median at 1.13 (live lives are at full width, these at half).
- At equal compute (671 CU, about step 17,300) the combined setting is at 0.60 and 0.69, against 0.86 and 0.91.
- **Why the decay matters.** `residualDecay` shrinks the parts of the predictor that say "something changes" on every update. It was added in decision 0015 so that "a residual must keep being re-earned", to counter the drift in point 2. By the code's own note it caps each shift-gate entry at about 0.5 and each output weight at about 0.05. It limits the damage by limiting the learning.
- **Why the replay matters.** Sleep replays 64 transitions from the last 2,000 steps, which is one update in five, all from the recent past. That is no protection once the policy has moved on.
- **The search could not reach this.** `knowledge/genome-space.json` allows `residualDecay` 2.5e-4 to 4e-3 (never 0) and `bufferSize` 500 to 4,000 (never the whole life). The ranges were centred on the learner as it stood.
- **The periphery never learns.** Its ratio is 1.00–1.06 in every condition. About half the network's weights sit in the periphery encoder, and nothing in the loss makes them useful.
- **Not supported:** my guess that the shift prior should read only the move part of the action. It did not help in any pairing.

### 4. Curiosity is what does the damage

Broad held-out ratio at 25,000 steps, two seeds, and how narrow the policy became:

| condition | seed 1 | seed 2 | share of 16 px moves | policy |
|---|---|---|---|---|
| random policy, fixed predictor (from the table above) | 0.51 | 0.64 | 32% | uniform |
| default learner, curiosity on | 0.98 | 0.88 | 69%, 50% | drifting to diagonals |
| curiosity on, fixed predictor | 0.65 | 0.98 | 71%, 89% | collapsing to diagonals |
| default learner, lagged-progress reward | 3.40 | 2.13 | 48%, 90% | two moves take 90% of steps |
| lagged-progress reward, fixed predictor | 1.71 | 1.13 | 93%, 94% | one or two moves take 90% |

- **The lagged-progress reward** pays `(loss of the model from 250–500 steps ago − loss now)` on each fresh transition, before training on it. It is closer to what "learning progress" means, and it no longer follows pixel change. It fails for a different reason: the learner improves fastest at whatever it does most, so the reward feeds on itself. Within a few thousand steps the policy repeats one or two moves, predicts them almost perfectly (own-stream ratio mostly below 0.2) and is worse than copy-last elsewhere (broad ratio 1.1–3.4).
- I did not tune the reward's scale. The lagged reward is roughly 10–50 times larger than the default one, which makes the collapse faster. The feedback loop itself does not depend on the scale.
- **Reading.** In this world, with this learner, progress in pixel prediction cannot be the drive. The predictable part of the world is shallow and mostly about the learner's own motion, and a learner that controls its own data can always manufacture progress by narrowing what it does.

### 5. Can the policy learn from any reward at all?

A diagnostic, not a proposal as it stands: the harness gave a reward of 1 whenever a press made something respond (read from the privileged trace), with curiosity off and the predictor fixed as in section 3. Two seeds per condition, 25,000 steps.

| condition | responses per 1,000 steps, first half → second half of life | time with the cursor over something interactive, last 5,000 steps |
|---|---|---|
| crib only, no reward (baseline) | 10 → 13 and 22 → 22 | 9% and 19% |
| crib only, rewarded | 8 → 6 and **14 → 156** | 6% and **68%** |
| all four rooms, rewarded | 6 → 7 and 9 → 4 | 7% and 3% |

- **One run of four learned the behaviour, and learned it strongly.** From about step 15,000 it parks on a shape and clicks it: 250 responses per 1,000 steps in its last windows (ten times the baseline), half of its presses land, and the cursor is over a shape two-thirds of the time. That behaviour would meet the measures of M2 and M3.
- **The other three learned only to press more often** (press rate up from a third of steps to a half or more), not where to press. One of them showed a late rise in its last window.
- **Reading.** The policy machinery is not the dead end. Given a reward that marks "I made something respond", milestone-like behaviour can appear inside one life. What it lacks is reliability: finding the rewarded situation is luck, and what it learned was to stay put, not to travel to things.
- So the binding problem for behaviour is the drive first, and exploration and credit second.

### 6. Why the milestone scores read as they do

- A newborn's policy is close to uniform, so it passes milestones by luck at the rates in decision 0016.
- Training moves the policy toward big jumps. That lowers the time spent over things (M2), the clicks that land on them (M3) and the pages visited (M5), and it raises scrolling's reach (M4) a little.
- So the behaviour score C, which compares each life with newborns, mostly measures how far a life has drifted from random. The one gene with a clear effect, the learning rate, fits: more learning, more drift, lower C.
- **This disagrees with the recommendation in `2026-11-16-calibration-flat.md`** to rank on C alone. That note already guessed that C's signal "tracks how far training pulls a learner away from newborn behaviour" and that a search on C "would push toward learning less". The measurements here support that guess. A search on C would select for not learning.

### 7. A first interest signal that can get bored (added the same day, after the user's direction)

The user ruled (decision 0029) that the drive stays intrinsic: interest without ground truth, "that was a surprising experience, lets try it more", not a point for clicking the right box. The two signals in section 4 fail that description for the same reason: neither gets bored. The default one pays for a big jump every time. The lagged one divides progress by a loss that shrinks as the learner specialises, so a mastered niche keeps paying.

**Candidate: "more surprising than I expected".** A small head predicts, from the state and the action, the log of the prediction loss that is about to occur. The reward is `clamp(log loss − expected log loss, −1, 1)`, from step 1,000 on. An error that is always large (a big move) comes to be expected and stops paying. So does an error the predictor cannot reduce, and so does one it learns to predict. (The idea is close to aleatoric-uncertainty-aware curiosity, Mavor-Parker et al. 2022, and to the confidence module in Schmidhuber 1991; cited from memory, not re-checked.)

**What it pays for** (all four rooms, 4 runs, mean over windows after the warm-up):

| what the action did | default reward (section 1) | expected-surprise reward |
|---|---|---|
| nothing changed | 0.005 | −0.21 |
| a move of 1 / 4 / 16 px | 0.005 / 0.007 / 0.016 | 0.01 / 0.00 / 0.00 |
| a click made something respond | 0.011 | +0.27 (noisy: −0.4 to +1.0 by window) |
| a scroll moved the page | 0.010 | +0.20 |
| a taskbar switch | 0.044 | +0.78 |
| a page navigation | 0.034 | +0.78 (2 windows only) |

**What the learner did with it** (25,000 steps, predictor fixed as in section 3, two seeds):

| condition | broad held-out ratio | 16 px share at the end | taskbar switches per 1,000 steps, first → second half | responses per 1,000 steps, second half |
|---|---|---|---|---|
| random policy (reference) | 0.51, 0.64 | 32%, 32% | 1.8 → 0.4, 2.2 → 1.8 | 8.5, 8.2 |
| old curiosity | 0.65, 0.98 | 71%, 89% | 0.2 → 0.0, 3.2 → 5.4 | 2.1, 1.0 |
| expected surprise, weight 1 | 0.80, 1.16 | 18%, 18% | 0.0 → 0.0, 6.4 → 3.3 | 1.4, 3.4 |
| expected surprise, weight 0.1 | 0.66, 0.91 | 37%, 31% | 0.2 → 0.1, 7.0 → 7.0 | 1.5, 3.4 |
| lagged progress on a slow scale, weight 0.1 | 0.82, 1.03 | 21%, 94% | 1.8 → 2.0, 1.7 → 0.4 | 1.6, 0.0 |

- **The flinging stops.** Big jumps earn nothing, and the policy stays broad (move entropy 2.7–3.2 of a possible 3.22).
- **One seed of two discovers the taskbar and keeps using it:** 3–7 switches per 1,000 steps, against about 2 for a random policy on the same seed. The other seed never finds it.
- **It does not click on things more.** Responses fall below the random rate. In a crib-only world the learner drifts to the taskbar strip and stays near the one icon there (36% and 70% of its time), where a hover lights the icon up. The expectation head cannot learn that pattern from what it sees, so that surprise never fades.
- **Prediction on fresh worlds is worse than with a random policy** (0.66–1.16 against 0.51–0.64), though the policy is far healthier than under the old curiosity.
- **Lagged progress fails again** on the slow scale: one move takes 70% of steps.

**Reading.** The signal is the right kind: it ranks events the way "interesting" should, and it can get bored. What it does not yet do is get bored of everything it should, or make the learner better at predicting fresh worlds. Two seeds; a first measurement.

**The bar this sets.** A random policy is, so far, the best explorer by the measure the user accepted (prediction on fresh worlds). An interest signal earns its place when its learner predicts fresh worlds better than a random one at equal compute, above all on the transitions random play rarely reaches: responses, reveals, navigations, room switches.

## What this says about the design

1. **The training target does not contain the evaluation target.** Nothing in the reward favours going to a thing and acting on it. The 12 searched genes are settings of one mechanism, so the search space holds no learner that passes milestones.
2. **Two parts of the learner fight each other.** The drive narrows the data. The decay throws away what was learned to limit the harm. The score then records the harm.
3. **Retention is the missing piece, and its absence is now measurable.** Replaying the whole life, without the decay, is a consolidation mechanism in the project's own sense. It is the difference between 0.86–0.91 and 0.51–0.64. The original design's conditions A, B and C (spec §5.6) were about exactly this and were never built.
4. **Curiosity and consolidation are coupled.** A progress-seeking drive only makes sense for a learner that keeps what it learns and cannot shrink its own world. Otherwise forgetting and narrowing both pay.
5. **The world asks for more than this policy can do in a life.**
   - The policy is a linear readout of a state trained only to predict pixels, with one-step credit and a 10-step horizon.
   - 45 of the 76 action dimensions are keys, and no room responds to any key. Scrolling does something in one room of four. A newborn presses a key on 98% of steps and scrolls on 67%.
   - Reaching a link or a taskbar icon takes many precise 16 px steps. Random play gets there about once in a thousand steps.
6. **The measurements are too noisy to steer by.** The live held-out set is 64 transitions in one room. A single checkup of M3 or M5 says almost nothing about the learner (`2026-11-16-calibration-flat.md` §2). The broad set used here gives smooth curves from the same weights.
7. **The loop is slow where it need not be.** The runner yields about 5 lives a day. This probe runs a full-length life at half width in about 10 minutes on one core, eight at a time, and is deterministic. At full width it is about 25 minutes a core.
8. **Nothing accumulates.** A life ends at 2,048 CU, no weights are inherited, and the inherited recipe tunes a mechanism that does not learn. "Learning gets cheaper over time" needs something that builds up, within a life or across lives. Today neither does.

## Tweaks, in the order I would do them

### Now: make the base learner honest (small, measured above)
1. **Predictor:** replay from the whole life at about one replay per step, and no residual decay. In the browser the whole-life buffer should be a uniform sample of the life (a reservoir) of a few thousand transitions, to bound memory. *Untested: the reservoir at that size.*
2. **Drive:** switch the curiosity reward off (weight 0) until a drive exists that is not pixel change. The learner then acts at random, which is what a newborn does, and learns to predict. *After decision 0029: the drive is replaced rather than switched off. The random-policy learner stays as the reference every interest signal must beat (section 7).*
3. **Measurement:** score prediction on a broad held-out set (several hundred transitions, all rooms), at the checkup levels. Keep the 64-transition set only for continuity.
4. **Search box:** let the genome reach the settings above (decay down to 0, buffer up to the whole life, curiosity down to 0) under a new schema version.
5. **Loop:** do learner experiments offline in Node first. Use the browser population to confirm and to run the meta-learner at scale once there is something to search.

### Next: give the policy something it can learn
6. **A drive about causing things,** not about pixels: reward change that the learner's hands caused beyond its own motion, and reward seeing something new. These are the reserved genes `ctrlWeight` and `noveltyWeight` from the v2 design. Section 5 shows that a reward marking caused responses can produce the behaviour; the open part is whether the learner can compute such a reward for itself from its own predictions. Test offline before anything goes live.
7. **An action interface a newborn can use.** Two candidates:
   - start the policy biased toward "none" for button, scroll and key, so it mostly moves and sometimes acts;
   - let it point: choose a place on the screen and jump there, so reaching a thing is one decision. This changes the action vocabulary and needs a recorded decision.
8. **A staged world.** Crib first; add rooms as prediction in the current ones settles. This is a curriculum prior and must be recorded.

### Re-aim the experiment at the question
9. **Score retention and transfer directly.** Give a life two phases: some rooms first, the rest introduced half-way. Score, by prediction on broad held-out sets:
   - how fast the new rooms are learned, against a newborn started in them (later learning cheaper);
   - how much of the early rooms is kept (retention).
   Both are deterministic given the weights, so they are low-noise, and they are what the research question asks.
10. **Make consolidation the first thing the meta-learner searches:** how much to replay, from where, and with what priority. There is now a measured forgetting effect for it to fix.
11. **Run the population offline** as well as in tabs, with the same code and the same gate checks, so the meta-learner sees hundreds of lives a day.

## Questions only the user can settle

**Answered 2026-11-17, recorded in decision 0029:** (1) no: the drive stays intrinsic, as interest without ground truth; (2) yes; (3) yes. The questions as asked:

1. **Must competence emerge from curiosity alone?** The research question needs learning that can be measured against a criterion. It does not need the criterion to be self-generated. A very simple rewarded task (the stage-2 "directive" idea) would give clean learning curves, retention tests and transfer tests now. It adds a reward channel, so it needs a recorded decision (invariant 2). *Agent's recommendation: bring a minimal version forward, alongside tweaks 1–5.*
2. **Is prediction on fresh worlds an acceptable first competence** for the retention and transfer experiments (tweak 9)? It is what the learner can demonstrably acquire once tweaks 1–3 are in.
3. **May the population also run offline** (tweak 11)? It changes where lives come from, not what they are.

## What this does not establish

- **Scale of evidence.** Two or three seeds per condition, half width for most runs, one machine. The live-data findings (sections 1 and 2) rest on 106 lives and are the firmest.
- **No tuning.** No reward scale, learning rate or replay ratio was tuned. The numbers show directions, not optima.
- **Milestones were not run offline.** "Behaves no worse than a newborn" for the random-policy learner follows from its policy staying uniform (action entropy at its maximum throughout). It was not checked with the checkup suite.
- **Longer lives.** No life on record is longer than 2,063 CU, and none was run here. Whether the fixed predictor keeps improving past 25,000 steps is not known; its curve was still falling.
- **The drives in tweak 6 are mostly untested.** Section 4 shows two drives that fail. Section 5 shows that a reward read from the privileged trace can work, in one run of four. Section 7 shows one self-computed signal that marks the right events and has not yet beaten a random policy.
- **The experimental switches are scratch code.** They are not checkpointed, not parity-tested on WebGPU and not fit to merge.

## Sources
- Code: `packages/learner/src/{learner,model,shift,replay}.ts`, `packages/tensor/src/optim.ts`, `packages/world/src/{world.ts,rooms/*}`, `packages/evaluators/src/search-composite.ts`, `knowledge/genome-space.json`.
- Data: `data/runs` at `bdcec01`; `data/meta/calibration.json`.
- Earlier notes: `2026-10-21-why-not-learning.md`, `2026-10-21-does-it-learn-2.md`, `2026-10-22-growing-up-live.md`, `2026-11-16-calibration-flat.md`, `2026-11-16-filter-convergence.md`; decisions 0013, 0015, 0016, 0018, 0020, 0021; specs `2026-10-19-blanktab-v1-design.md`, `2026-10-22-learner-v2-capabilities-design.md`.
