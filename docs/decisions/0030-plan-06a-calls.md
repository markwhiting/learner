# 0030 — Plan 06a calls: an optional whole-life reservoir, and the fresh-world test v1

- **Date:** 2026-11-17
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (agent's calls, not yet confirmed). The direction is the user's (0029).
- **Resolves:** the implementation choices of `docs/plans/2026-11-17-06a-predictor-and-fresh-world-test.md`; gives Q31 its yardstick and Q33 its raw measurement

## Context

0029 makes prediction on fresh worlds the first competence, and asks that any interest signal be judged by it. Two things were missing: a way for the learner to keep a sample of its whole life in bounded memory, and a test set broad enough to hold the events that random play rarely reaches.

## Decision

**Agent's calls (not yet confirmed):**

- **Whole-life replay is an optional learner setting, `reservoirShare`.**
  - That share of the `bufferSize` slots is a reservoir (Algorithm R: a uniform sample of every transition of the life so far). The rest stay the FIFO of recent transitions. One slot always stays recent. A sleep replays uniformly over both.
  - Absent or 0 it is off, and the learner is bit-identical to before: same golden trajectory, same RNG stream, same saved state. So it is not one of `DEFAULT_SETTINGS`, and `LEARNER_STATE_VERSION` does not change (a running life's checkpoint still loads).
  - It becomes a gene only under a new genome schema version (0021).
- **The learner scores a set; the harness labels it.** `scoreSet` returns one loss per transition for each view and keeps nothing. Which transition was a navigation or a response is privileged, and stays on the harness side (AGENTS.md invariant 2).
- **`World.inspect(x, y)`** is privileged and harness only, like `state()`. The prober needs it to find things to click.
- **The fresh-world set, version 1.**
  - **One set for every life.** Its worlds are seeded `hashSeed(0, "fresh/1/<part>/<index>")`, so no life lives in them and every life is scored on the same transitions. A per-life set would add noise between lives for no gain.
  - **Two parts, each room in front equally often at the start.** 32 trajectories under uniform random actions and 96 under the prober, 40 transitions each, scored from the zero recurrent state.
  - **The prober** picks a target (an interactive element 65% of the time, a taskbar icon 20%, any point otherwise), walks the cursor to it and clicks what it finds. Its habits are constants in `packages/harness/src/fresh.ts`.
  - **One label per transition,** the first that applies: `switch`, `navigate`, `reveal`, `widget`, `respond`, `scroll`, `move`, `static`.
  - **A group's score is Σ loss ÷ Σ copy-last** over its transitions. Below 1 is better than assuming nothing changes.
  - **The wide view and the close-up are scored apart.** The close-up has the shift prior of 0015 and the wide view does not, and both together are dominated by the close-up (`research/2026-11-17-predictor-fixes-and-fresh-world-test.md` §3).
  - **The version is pinned by a hash of the set's content.** Any change to the worlds, the prober or the labels bumps `FRESH_SET_VERSION`; scores of two versions do not compare.
- **Privileged state is used here only to build and label the test,** which 0029 allows. Nothing of it reaches a learner, and no learner is rewarded by it.

## Alternatives considered

- **Replacing the FIFO with the reservoir, or bumping the state version.** Either would change every running life, or stop it resuming.
- **A larger FIFO instead of a reservoir.** A whole life of frames is about 400 MB at 25,000 steps, and lives are open-ended (0014).
- **Finding rare events by running random actions for longer.** Random play navigated about 1.5 times in 1,000 steps and never switched rooms in 3,840; a set with 30 of each would need tens of thousands of steps in every tab.
- **Exposing every element's rectangle.** A larger privileged surface in all four rooms; `inspect` plus sampling is enough.
- **A per-life or per-panel fresh set.** More noise, and the comparison between lives is the point.

## Consequences

- Nothing is live: no setting in the genome, no checkup records the score, no evaluator reads it. Those are Plan 06a tasks 7–9 and Plan 06d.
- The pinned hash is of frames rendered in Node. A browser must be shown to render the same set before a tab's score is compared with an offline life's.
- The prober's habits and the set's size are guesses. Changing them is a new set version.
