# The predictor's fixes and the fresh-world test, measured offline

**Date:** 2026-11-17
**Status:** offline measurements by a research agent (AGENTS.md invariant 5) for Plan 06a (`docs/plans/2026-11-17-06a-predictor-and-fresh-world-test.md`). Nothing here is live. Two to four seeds per condition: first measurements, not confirmations. It corrects part of `2026-11-17-task-and-target-introspection.md`.

## In plain words

1. **The earlier note credited the wrong thing.** It said the predictor learns much more with its decay off and "replay drawn from the whole life". With a random explorer, replaying only the last 2,000 steps does just as well. The gain comes from switching the decay off and replaying about once per step.
2. **A bounded whole-life memory is built, and is not needed yet.** The reservoir works and is off by default. It should matter when experience stops being broad, which a random explorer's never does.
3. **Nearly everything the predictor has learned is one skill: the close-up view shifts when the cursor moves.** That skill is a mechanism the design gave it (the fovea-shift prior, decision 0015). On the wide view the predictor has never beaten "nothing changes", in any of 70 offline runs.
4. **The new fresh-world test shows the same thing event by event.** For the best learner so far, every kind of event scores 1.00 to 1.03 on the wide view, which is no better than "nothing changes", and plain cursor moves score worse (1.03 to 1.26). The close-up improves, mostly on plain cursor moves (0.55 to 0.78).
5. **Lives differ a lot by chance.** Whether and when the shift skill takes off decides most of a life's score, so two seeds are not enough to compare two settings that are close.

## Method

- **Lives:** `tools/probe/introspect-probe.ts` on branch `feat/06a-predictor-retains`: the real learner on the CPU backend at CPU widths, the real four-room world, caregiver and 64-transition held-out set, 25,000 steps. Deterministic.
- **Scores,** all as loss ÷ copy-last (below 1 is better than assuming nothing changes):
  - **broad:** the 800-transition random-action set of the introspection note, split into the wide view (periphery) and the close-up (fovea).
  - **fresh:** the fresh-world set v1 of decision 0030 (5,120 transitions: 1,280 under random actions, 3,840 under the prober), by part and by what the action did. "Eventful" pools the five rarer labels (room switch, navigation, reveal, widget change, response).
- **"Fixed predictor"** means `residualDecay` 0 and 250 replays per sleep (one per step). **"Random explorer"** means `curiosityWeight` 0, which leaves the policy uniform.
- **Raw output:** `.local/06a/` in the worktree (not committed); the windows of every run are in `2026-11-17-predictor-fixes-and-fresh-world-test/runs.json` (34 files; the same settings and seed always gave the same numbers, and four of the files are such repeats). The fresh-world scores of the first two batches are both views together; the last batch scores each view.

## 1. Replay: how much matters, from where does not (random explorer)

Fixed predictor, random explorer. Broad ratio after 25,000 steps (970 CU):

| What the sleep replays from | seed 1 | seed 2 |
|---|---|---|
| the last 2,000 steps (today's FIFO) | 0.51 | 0.65 |
| the whole life (FIFO of 25,000) | 0.51 | 0.64 |
| a 2,000-slot reservoir of the whole life | 0.50 | 0.69 |
| a 4,000-slot reservoir | 0.59 | 0.64 |
| half recent, half reservoir, 2,000 slots | 0.87 | 0.70 |
| half recent, half reservoir, 4,000 slots | 0.82 | 0.97 |

- **The whole-life row reproduces the introspection note's 0.51 and 0.64 exactly,** now through the committed code path.
- **Recent-only replay matches it.** The introspection note's table had no such row: its "whole-life replay" arm changed the amount of replay (64 to 250 per sleep) and where it came from at once. The amount is what mattered.
- **The half-and-half rows are worse in three runs of four.** There is no mechanism for that, and two more seeds reverse it (section 5).

## 2. With curiosity on, where replay comes from matters, and the decay was a safeguard

Fixed predictor, curiosity on (weight 1, today's reward). After 25,000 steps (970 CU), seeds 1 and 2:

| What the sleep replays from | broad | fresh, random part | steps that are 16 px jumps |
|---|---|---|---|
| the last 2,000 steps | 2.73, 1.26 | 5.05, 2.25 | 84%, 78% |
| a 2,000-slot reservoir of the whole life | 0.86, 1.53 | 1.18, 1.69 | 85%, 81% |
| the whole life (FIFO of 25,000) | 0.65, 0.98 | 1.02, 1.34 | 77%, 87% |

- **With the decay off and recent-only replay, the curiosity-driven learner's predictor ends up to five times worse than assuming nothing changes.** The policy flings the cursor and the predictor follows it into that niche. The residual decay was added to limit exactly this (decision 0015). The introspection note called it a handicap; that holds for a random explorer only.
- **Whole-life replay limits the damage.** The whole-life row reproduces the introspection note's 0.65 and 0.98. The reservoir helps in one seed of two.
- **None of these beats a random explorer** (section 3).

## 3. The fresh-world test: who predicts fresh worlds, and what

End of life (25,000 steps), seeds 1 and 2, both views together:

| Learner | CU | random part | probe part | plain cursor moves | eventful |
|---|---|---|---|---|---|
| today's learner (curiosity on, default settings) | 671 | 1.15, 0.99 | 1.04, 0.90 | 1.07, 0.89 | 1.05, 0.97 |
| random explorer, default predictor settings | 671 | 0.99, 0.99 | 0.91, 0.91 | 0.90, 0.90 | 0.97, 0.97 |
| random explorer, fixed predictor | 970 | 0.83, 0.83 | 0.64, 0.67 | 0.56, 0.60 | 0.93, 0.94 |

- **The order is the introspection note's:** today's learner is at or worse than "nothing changes", a random explorer is slightly better, and a random explorer with the fixed predictor is clearly better.
- **The fixed predictor's life costs more compute** (970 CU against 671). At 20,000 steps (776 CU) it scored 0.82 and 0.89 on the random part and 0.62 and 0.70 on the probe part, so the order holds at near-equal compute.
- **It is not steady.** Seed 1 of the fixed predictor read 1.20 on the random part at 15,000 steps and 0.82 at 20,000: a relapse and recovery inside one life.
- **The gain is in plain cursor moves.** The eventful group barely moves (0.93 to 0.94), and the next table shows where even that comes from.

The random explorer with the fixed predictor, end of life, by view and by what the action did (four seeds; a re-run of seeds 1 and 2 with the views scored apart gave the same combined numbers as the table above):

| What the action did | transitions | wide view | close-up |
|---|---|---|---|
| whole random part | 1,280 | 1.00, 1.00, 1.01, 1.01 | 0.73, 0.72, 0.72, 0.91 |
| whole probe part | 3,840 | 1.00, 1.01, 1.03, 1.03 | 0.59, 0.63, 0.58, 0.79 |
| room switch | 23 | 1.00, 1.00, 1.00, 1.00 | 0.83, 0.89, 0.93, 0.97 |
| navigation | 51 | 1.00, 1.00, 1.01, 1.00 | 0.90, 0.98, 0.88, 0.84 |
| reveal | 61 | 1.00, 1.00, 1.00, 1.00 | 0.86, 0.84, 0.78, 0.88 |
| widget change | 111 | 1.00, 1.01, 1.03, 1.03 | 0.86, 0.94, 0.95, 1.07 |
| response | 216 | 1.00, 1.00, 1.01, 1.01 | 0.91, 0.91, 0.87, 0.93 |
| scroll | 275 | 1.00, 1.00, 1.00, 1.00 | 0.82, 0.80, 0.75, 0.88 |
| plain cursor move | 4,256 | 1.03, 1.08, 1.25, 1.26 | 0.55, 0.59, 0.55, 0.78 |

- **On the wide view nothing is learned.** Every group reads 1.00 to 1.03, and plain cursor moves are worse than "nothing changes". Across the ten runs scored by view (500 group readings over their lives), the lowest wide-view ratio is 0.996.
- **The same holds on the older broad set:** in all 70 offline runs with that measure (40 in the introspection note, 30 here), no window of any run has a wide-view ratio below 1.
- **The close-up carries the whole gain,** and most of it is plain cursor moves. Event steps gain less (0.75 to 1.07). This set cannot tell how much of that is the cursor moving on the same step.
- **Why the wide view cannot improve** (reading the code, not a measurement): the close-up has a mechanism for "the picture shifts when the cursor moves" (the fovea-shift prior). The wide view has only one linear layer from a summary vector to all 3,000 output values (`packages/learner/src/model.ts`, `headP`), which has no way to change one place and leave the rest.

## 4. Lives differ by chance, because the score is one skill taking off

In every run that ends well, the close-up ratio falls steeply at some point (for example from 0.80 to 0.62 between steps 7,500 and 10,000) and keeps falling; in runs that end near 0.85 it never does. Which run takes off, and when, changes with anything that changes the random stream, including whether a reservoir draws from it.

- Two settings that are close cannot be told apart at two seeds.
- A search that ranks lives by this score will mostly be ranking how reliably the shift skill takes off. That is a real signal, but a narrow one.

## 5. Two more seeds

Fixed predictor, random explorer. Broad ratio after 25,000 steps:

| What the sleep replays from | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| the last 2,000 steps | 0.51 | 0.65 | 0.65 | 0.85 |
| a 2,000-slot reservoir of the whole life | 0.50 | 0.69 | 0.70 | 0.84 |
| half recent, half reservoir, 2,000 slots | 0.87 | 0.70 | 0.61 | 0.68 |

- **No difference between them is visible.** The half-and-half rows that looked worse on seeds 1 and 2 are the best on seeds 3 and 4.
- **The spread between seeds (0.50 to 0.87) is larger than any difference between settings.**

Fresh-world test, both views together, end of life, four seeds:

| Learner | random part | probe part |
|---|---|---|
| today's learner (curiosity on, default settings) | 1.15, 0.99, 1.00, 1.11 | 1.04, 0.90, 0.90, 1.04 |
| random explorer, fixed predictor | 0.83, 0.83, 0.83, 0.95 | 0.64, 0.67, 0.64, 0.82 |

The second row is the reference an interest signal has to beat (decision 0029).

## What this changes

Agent's proposals, not yet confirmed.

1. **The genome box (Plan 06a task 9): keep `residualDecay` searchable down to 0, do not fix it at 0.** Off is best for a random explorer and harmful under today's curiosity with recent-only replay.
2. **The reservoir stays in the code, off.** It costs nothing off. Its real test is a life whose experience moves on.
3. **Score by view.** A search that ranks lives on the combined ratio ranks the cursor-shift skill. Learning about the world would show on the wide view and on events, and today nothing shows there.
4. **Fix what the predictor can learn before more interest signals.**
   - **Why first:** an interest signal built on the predictor's error cannot get bored of something the predictor can never learn. The "expected surprise" signal got stuck on the taskbar icon, where the surprise never faded (introspection note, section 7), which fits this.
   - **The cheapest candidate:** the first conv layer's feature map is already at the prediction's resolution (25×40 for the wide view, 16×16 for the close-up). A small conv output head on that map, conditioned on the action, can change one place without touching the rest. By my reading of the code it needs no operation the tensor library lacks; this is not yet tried.
   - **The test:** the fresh-world test's wide view, against the reference in section 5, offline, before anything goes live.
   - The other route is prediction in a learned latent space (the reserved `latentWeight` gene of the v2 design). Open question Q35.

## What this does not establish

- **Nothing about GPU widths or live tabs.** Every run is at CPU widths in Node.
- **The reservoir is untested where it should matter:** a life whose experience narrows or moves on (a learned policy that specialises, or a life that changes rooms for good). That is the retention experiment of decision 0029, point 2.
- **The prober's habits and the set's size are guesses** (decision 0030).

## Sources

- Code: `packages/learner/src/{learner,replay,model}.ts`, `packages/harness/src/fresh.ts`, `packages/world/src/world.ts`, `tools/probe/introspect-probe.ts`.
- `docs/research/2026-11-17-task-and-target-introspection.md` (the note this corrects), decisions 0015, 0029, 0030.
