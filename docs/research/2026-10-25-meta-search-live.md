# Meta-learner search, live and inert (Plan 05b)

**Date:** 2026-10-25. **Build:** ba91080, merged as e18ef2a through PR #30. CI passed on GitHub's runners and the deploy succeeded.

## Summary
- **Live and inert, as designed.** After the merge, a manual gate run merged a session PR (#31) and recomputed the meta-learner's state, committing `meta: calibrate, 0 eligible lives, posterior 8e81f3f6`. The site now serves a schemaVersion-2 posterior with mode `calibrate`, no models, no incumbent and no confirmation. knowledgeSha is unchanged (`9be1e368…`), so no genome was promoted.
- **Old tabs are safe.** A tab still running 05a code reads the v2 file as unknown and draws the centre. New code draws exactly as 05a did on a v1 file: 800 of 800 cases matched in the final review.
- **Search switches itself on.** It starts only when calibration's signal check scores 1 on real lives; no flag or switch can turn it on. Until then the gate fits nothing and tabs draw only calibrate, uniform or centre.
- **Verified before merge:**
  - an end-to-end simulation on a local fixture repo: calibrate → signal → search → declare → three-arm confirmation → pass → promote → a posterior for the new centre, plus a "flat" run that stayed inert;
  - bit-identical GP fits and draws in Node, Chromium, WebKit and Firefox;
  - CI on x86-64 Linux, against goldens made on arm64.

## Not done yet
- **Calibration has no lives yet.** It is at 0 of 30. The contributing tab was suspended in the background by Safari, so lives barely advanced. The fix is `tools/runner/run-lives.ts`: the ordinary page in a headless Chromium tab with its own profile, the short-lives switch on and `caffeinate` held, reloaded if it crashes or stalls for 10 minutes. Its token is pasted once into the page with `--setup` and lives only in that profile's localStorage; the script checks that one is there but never reads it [agent's call]. A 3-minute check on the live site without a token ran a calibration life (recipe 5 of 10) on WebGPU through its 16 and 32 CU checkups with the switch on.
- **Verification-time bound.** A lost `verified.json` with about 1,000 ts lives would take roughly 11–17 min per build on CI. The mitigation is planned in 0024 and must be built before that many ts lives exist.
- **Repeat confirmations are capped** at 3 per candidate (0024). Raising the cap needs a recorded decision.

## Runner, first lives (2026-10-26)
- **The loop closes.** With a token pasted through `--setup`, the runner finished an adopted calibration life at its budget checkup (`lrn_e90903d0871f32e5`, PR #32) and started a newborn (`lrn_7a6495cac26a7170`, PR #34). A manual gate run merged both and committed `meta: calibrate, 1 eligible lives, posterior 41b3e8a5`.
- **One file lost to a send race.** For about two minutes two runners shared the profile (an operator error: the first was not stopped). One tab opened PR #33 with `lrn_7a6495cac26a7170/0.json`; the other, not yet seeing it, reset the branch to main, so GitHub closed #33 as empty and the page dropped its file as rejected. The runner now refuses to start while another holds the profile (`af51e5f`).
- **It was not only two runners: one tab lost a file on almost every life (fixed, `249b7d0`).** Ten more PRs (#43–#70) were closed seconds after opening, with one runner. The page's GitHub calls used the browser's HTTP cache, and GitHub marks authenticated GETs cacheable for 60 s. A send within a minute of the last one read its own stale answers: "no open PR" (so it reset the branch of the PR it had just opened; GitHub closed it as empty, and the page dropped its files as rejected) and the old branch head (so its next commit was not a fast-forward: the repeated 422s). Each lost file was a newborn's `0.json` (its 16 CU checkup), so no life became ineligible; the files are not recoverable. Calls now bypass the cache. Two guards stay (`6ca352d`): a PR this browser opened is checked by number before any branch reset, and one tab of a browser sends at a time (a Web Lock), which also closes the two-tab race.
- **The hourly gate is late.** GitHub skipped scheduled runs: the gate ran at 21:49 and 00:24 UTC, not every hour. Merges and counts wait for it; nothing is lost.

## Runner, when idle (2026-10-26)
- **`run-lives --when-idle`** runs the learner only after 10 min without keyboard or mouse input and on AC power. It freezes the browser's processes (SIGSTOP) the moment someone is back, and holds the Mac awake only while running, at most 8 h after the last input. A life is measured in CU, so a frozen life resumes mid-step: a 3-minute freeze of the live tab stopped all 5 browser processes and the learner, and after the thaw it went on at the same rate through its next checkup. The pause rule is unit-tested; a real pause on returning input was not triggered by hand (that needs synthetic input).
- **`tools/runner/agent.ts install`** makes it a launchd agent: it starts at login and restarts if it exits. `uninstall` removes it.
- **Limits:** with the lid closed and no external display the Mac sleeps and nothing runs. A confirmation life (05b) keeps its slot 48 h while silent, so a pause longer than that (a weekend off power) could cost it the slot; calibration lives are not affected.

