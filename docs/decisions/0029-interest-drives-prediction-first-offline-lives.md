# 0029 — Interest without ground truth; prediction on fresh worlds first; lives may also run offline

- **Date:** 2026-11-17
- **Status:** accepted
- **Decided by:** user (2026-11-17, answering the three questions of `docs/research/2026-11-17-task-and-target-introspection.md`). Details marked *agent's call* await confirmation.
- **Resolves:** the questions under "Questions only the user can settle" in that note; reshapes Q12 (what drives the learner) and Q24 (what follows "flat")

## Context

Calibration ended "flat" (`data/meta/calibration.json`, 2026-11-08). The introspection note found why: the curiosity reward tracks pixel change, the learner specialises on its own behaviour and forgets, and the behaviour score mostly measures drift away from a newborn's random play. It asked the user three things: whether competence must emerge from curiosity alone or a simple rewarded task could come forward; whether prediction on fresh worlds is an acceptable first competence; and whether lives may also run offline.

## Decision

**User's, in their words:**

1. **The drive stays intrinsic, as interest.** "i think curiosity should be rewarded, and valuable exploration should be rewarded as kind of interest style metrics that can occur when there is no ground truth to determine reward. e.g. that was a surprising experience, lets try it more... as opposed to, you clicked on the green box, one point."
2. **Prediction on fresh worlds is the first competence** for the retention and transfer experiments: "yes, i think thats a good metric".
3. **The population may also run offline,** with the same code and the same checks: "yes".
   - *What "offline" means here (agent's wording):* a life run as a command-line process in Node on the CPU, outside a browser tab, on the user's own computer unless a later decision says otherwise. It is not "without the internet": such a life would still send its log files to GitHub.
   - *The user asked what "offline" meant after answering (2026-11-17),* was given the meaning above, and replied "ok, lets go ahead with your proposal". The agent reads that as the "yes" standing with this meaning.

**What follows directly:**
- No reward that needs ground truth reaches the learner. The agent's recommendation to bring a minimal rewarded task forward is declined.
- The harness-reward run in the note (section 5) stays what it was: a diagnostic of the policy machinery. It is not a design.

**Agent's calls (not yet confirmed):**
- **An interest signal must be able to get bored.** "Lets try it more" is read as: return to what surprised, until it no longer does. The two signals measured so far cannot: the built-in one pays for every big jump, and progress against an older model pays for narrowing (note, sections 1 and 4).
- **First candidate: "more surprising than I expected"** (note, section 7). It ranks events as interest should and stops the cursor flinging. It has not yet beaten a random policy on prediction. Interest signals are developed offline and enter the genome as genes (0018, 0021); none goes live on this decision alone.
- **The bar for an interest signal:** its learner predicts fresh worlds better than a random-policy learner at equal compute, above all on transitions random play rarely reaches (responses, reveals, navigations, room switches). This ties the drive (the learner's own) to the metric the user accepted (the harness's).
- **The harness may still use privileged state to score** which held-out transitions are eventful. That is measurement, never an input or a reward (invariant 2).

## Alternatives considered

- **A minimal rewarded task now** (the agent's recommendation in the note). It would have given clean learning curves at once. Declined by the user: reward should come from interest, not from a judged outcome.
- **Switching the drive off** until a better one exists. The random-policy learner is the best predictor measured so far, but a learner with no drive is not the project's learner. It stays as the reference to beat.
- **Keeping the behaviour milestones as the first competence.** Their measures are too noisy at this stage, and the behaviour score falls as learning rises (`2026-11-16-calibration-flat.md` §2; note, section 6).

## Consequences

- **Left open: stage 2 of the v2 design.** `docs/specs/2026-10-22-learner-v2-capabilities-design.md` §10 gives directives "a harness success reward". This decision declines bringing that forward now. Whether a directive stage may later use a judged reward, or must also run on interest, is not settled here.
- **The search evaluator changes.** Ranking moves from the behaviour-plus-held-out composite to prediction on fresh worlds, on a broader held-out set than today's 64 transitions. That is a new evaluator version and its own decision (0020's rule), and it restarts calibration.
- **The genome changes.** The predictor's retention settings (no residual decay, replay from the whole life) and the interest genes need ranges the current schema excludes, so a new `schemaVersion` (0021's rule).
- **Offline lives need the gate's checks.** A life run in Node must produce the same log files and pass the same validation as a tab's. How offline lives are labelled and submitted is to be designed.
- **Behaviour milestones stay** as checkups and diagnostics. They are not the search target until a drive exists that moves them.
