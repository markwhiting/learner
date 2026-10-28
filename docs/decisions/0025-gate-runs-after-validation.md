# 0025 — The gate runs after each validation, with the hourly cron as a backstop

- **Date:** 2026-10-28
- **Status:** accepted (agent's call under the user's delegation; not yet confirmed in detail)
- **Decided by:** the direction is the user's; the mechanism and every detail below are the agent's calls [agent's call, not yet confirmed]
- **Resolves:** merges and the meta-learner's counts lagging by hours behind the PRs that carry them. Amends the "hourly gate" wording of `docs/specs/2026-10-19-blanktab-v1-design.md` §9.5 (`gate.yml (hourly cron, plus manual trigger …)`) and the heading of `docs/specs/2026-10-22-meta-learner-v1-design.md` §4.3 ("The gate's hourly refit").

## Context

`.github/workflows/gate.yml` ran on `cron: "17 * * * *"` and by hand. GitHub's scheduler skips cron runs under load: over 2026-10-20..28 the gaps between gate runs reached 6–8 h (`gh run list --workflow gate.yml`; for example 2026-10-27 06:19 → 13:15 → 18:45 UTC). A tab's PR waited that long to merge, and the meta-learner's eligible lives, posterior and calibration counts waited with it.

The user was told the plan "trigger the gate when a learner's PR opens, as well as on the timer", and said "go ahead with all plans" (2026-10-28).

## Decision

**User's:** the gate also runs soon after a learner's PR changes, not only on the timer.

**Agent's calls** [agent's call, not yet confirmed]:

- **The trigger.** `gate.yml` gains `workflow_run: { workflows: [validate], types: [completed] }`. The cron stays as a backstop and `workflow_dispatch` stays. `validate.yml` already runs on every `opened`, `synchronize`, `reopened`, `ready_for_review` and `edited` of a PR, so a tab's flush (one commit, `app/src/schedule.ts`) is followed by a gate run about as long after as validate takes.
- **Why `workflow_run` and not `pull_request_target`.** A `workflow_run` run always executes the workflow file as it is on the default branch, with the default branch's token, in the base repository. A PR, even one from a fork or one that edits `validate.yml` or `gate.yml`, cannot change what the run executes; it can only cause a run. The gate job still checks out `ref: main` and reads the open PRs itself through `tools/gate/gate.ts` (PR files as data only, decision 0017). Nothing from `github.event.workflow_run` (the PR's branch, title or sha) reaches any step. A test (`tools/gate/test/commit-meta.test.ts`, "workflows") holds all of this: the triggers, the concurrency group, no `github.event`, `github.head_ref` or `pull_request_target` outside comments, no job-level `if`, and `ref: main` on every checkout.
- **Every conclusion triggers a run (no job-level `if`).**
  - **A successful validate** is the common case: the gate merges the PR.
  - **A failed validate** is not a reason to wait. The gate re-validates every PR itself and never trusts validate's verdict (validate's own comment: a PR that edits the workflow runs its own copy). So an invalid PR is closed with its reason within minutes, and the tab learns it sooner.
  - **A cancelled or skipped validate** (a newer push superseded it; a PR not titled `run: `) gives a run with nothing new to do. It costs about a minute of runner time.
  - **Why not skip those at job level.** The concurrency group is workflow-level, so a run whose only job is skipped still takes the waiting slot (below). It could replace a waiting run that had a PR to merge. That PR would then wait for the next trigger, which for a quiet repo is the cron. Running on every conclusion avoids this case at the cost of a few short idle runs. (The slot behaviour is from GitHub's documentation as the agent knows it, not probed here.)
- **Concurrency is unchanged: group `gate`, `cancel-in-progress: false`.** At most one run is in progress and at most one waits. A newer trigger replaces the waiting run (GitHub cancels the older pending run in the group). Every gate run decides every open `run: ` PR, so the replaced run would have done nothing the newer one does not. A burst of pushes therefore queues one run, not one per push. The gate's merges, its `data/meta` commit and the deploy it calls stay serialised, as before.
- **A meta commit per run is acceptable.** `tools/gate/commit-meta.ts` commits only when `meta-cli.ts` computed a change. Since 2026-10-26 that has been once per run that brought a newly eligible life (`meta: calibrate, N eligible lives, …` commits). A log merge or a meta commit changes neither `codeSha` nor `knowledgeSha` (`tools/gate/manifest.ts`: `data/runs/**` and `data/meta/**` are not build inputs, and `posterior.json` is outside `PRIOR_FILES`). So tabs do not reload; at their next knowledge poll (15 ± 3 min, `app/src/updates.ts`) they read a fresher posterior. Deploys stay in the `pages` group, also without cancelling in progress.
- **The budget, estimated.** All figures are from runs measured on 2026-10-27. Per-run costs:
  - **Gate job.** About 35 s (run 40562958271: 32 s, merged 2 PRs, 8 gh calls in all).
  - **Deploy job.** About 20 s, only when something merged or a meta commit was made.
  - **Billing.** Each job is billed rounded up to a whole minute, so about 2 runner-minutes per run with a merge and 1 without.
  - **GitHub API.** 1 call to list the PRs, then per PR 1 to load, 1 for its owner (memoised per learner) and 2 to merge (decision 0017). The bound per PR is unchanged; a run sees fewer open PRs than before, because they merge sooner.

  Load at the request's assumption (about 2 PRs an hour):
  - **Runs.** About 48 triggered runs a day, plus up to 24 cron runs, for about 150 runner-minutes a day.
  - **API calls.** About 5–10 per run, roughly 20 an hour, against GITHUB_TOKEN's 1,000 REST requests an hour per repository.

  Load at the busiest day measured (2026-10-26: 170 validate runs, about 7 an hour):
  - **Runs.** About 400 runner-minutes a day.
  - **API calls.** About 70 an hour.

  The ceiling:
  - **Runs.** Serialised runs of about 1 min each cap the rate at roughly 60 an hour, however many tabs push.
  - **API calls.** That is about 600 calls an hour at the very most, still under 1,000.

  Runner minutes on standard runners are free for a public repository (`markwhiting/learner` is public). The 3-push limit of `commit-meta.ts`'s rebuild (decision 0022) holds per run as before. Runs are serialised, so the gate's own merges never move main under its push; only a human push can.
- **No worse for a PR mid-flush.**
  - **The merge is pinned.** The gate merges with `--match-head-commit <validated sha>` (`tools/gate/github.ts` `mergePr`).
  - **The contributor's race tests cover a merge at any point of a flush.** These are in `packages/harness/test/contributor.test.ts`:
    - "race: a tab commits after the gate validated; the pinned merge fails, the PR keeps every file, next run merges it";
    - "race: the gate merges (branch deleted | kept) while the tab is committing; no file is lost or duplicated";
    - "the gate merges right after the push (between the ref move and the check by number)".
  - **Races should be rarer.** A triggered run starts a minute or two after a flush, and a tab's next timed flush is 10 min away (`FLUSH_EVERY_MS`). An hourly run landed anywhere in that cycle.
  - **A run that leaves a PR for later is no worse off than before.** That covers a moved head, a base newer than the checkout, or a second PR for one learner. The next trigger or the cron decides it.

## Alternatives considered

- **Keep the cron only, or run it more often (`*/10`).** GitHub skips scheduled runs under load whatever the period, so the lag is not bounded; a shorter period mostly adds idle runs.
- **`pull_request_target` on gate.yml.** It also runs main's file with a write token, but it is the event that most invites checking out or interpolating PR content, and it would run the gate before validate has looked. `workflow_run` keeps validate first and gives the same safety.
- **Run only when validate succeeded.** Rejected above: it delays closing invalid PRs and, through the shared waiting slot, can delay a valid one.
- **A self-hosted or external scheduler.** This would need more infrastructure, and the user asked for minimal infrastructure.

## Consequences

- **More, smaller PRs.** A tab's PR now usually merges before its next flush, so each 10-minute flush tends to become its own PR and its own squash commit on main: about 6 an hour per active tab, where hourly gating gathered several flushes into one PR. The number of validate runs is unchanged (one per commit, as before). The tab opens a fresh branch and PR more often, using its own token.
- **Cancelled runs in the Actions list.** A waiting run replaced by a newer trigger shows as cancelled. That is expected, not a failure.
- **The cron is still the backstop.** It catches PRs that no validate completion follows: a fork PR whose workflows wait for approval, a validate that never ran, and PRs left for later.
- **Only from main.** `workflow_run` fires only for the workflow file on the default branch, so none of this takes effect until it is merged to main. It has not been observed live yet.
- **Unchanged.** Determinism (the gate's decisions depend only on main and the PRs' contents, not on when it runs) and the gate's rules and bounds (0011, 0014, 0017, 0019, 0022) do not change.
