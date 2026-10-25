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
