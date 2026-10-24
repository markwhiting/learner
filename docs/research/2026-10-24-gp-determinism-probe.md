# GP and Thompson-draw determinism across engines (probe, 2026-10-24)

**Question.** Do the gate (Node) and a tab (Chromium, WebKit, Firefox; main thread and module worker) get the same bits from the 05b pipeline: the GP fit, conditioning and prediction, the incumbent, and `birthDraw`'s `ts` and `confirm` draws? If they don't, the gate's re-verification of a tab's draw would fail for that engine (plan 05b, Task 6; spec §4.3–§4.5).

**Answer: yes, bit for bit, in all three engines, on both threads.** Every number in the probe's output matches the Node golden as float64 bits, compared as 16-hex-digit strings, so -0 and NaN would show too.

## What is compared
`packages/meta/test/gp-probe-inputs.ts` `computeGp()` runs in Node (`gp-probe.test.ts`, twice, same bytes) and in the e2e-only probe page `app/detmath-probe.html` (main thread and worker, `app/src/probe/`). `tests/e2e/detmath.spec.ts` "GP determinism: …" compares both against `packages/meta/test/fixtures/gp-probe-golden.json`. The output, with every number as its bits:
- `fit`: `fitGp` on a seeded 12-gene, 60-row dataset. All four L-BFGS starts run (36, 39, 45 and 40 iterations), start 3 wins, and the objective is −21.0201… The fit finds the four strong genes (0, 3, 6 and 9: ℓ ≈ 1.4–1.8; the others ℓ ≈ 10–29).
- `predict`: `conditionGp` with the fitted hyperparameters (its α, 60 values) and a joint `predict` at 6 seeded points (mean, the 6×6 covariance, and the variance).
- `incumbent`: `incumbentOf` on that posterior, centre at the cube's middle (pBeatsCentre 0.9972…).
- `picks`: 20 full `birthDraw` results, all `ts`, from `fixtures/posterior-v2-search.json` with its confirmation removed. Each runs `modelPosterior`, `candidateSet` (512 candidates) and `thompsonPick` (the joint 512×512 covariance, its Cholesky and the sample).
- `confirms`: 10 full `birthDraw` results, all `confirm`, from the same fixture with its confirmation in flight. All three arms appear: 8 newborn, 1 candidate, 1 centre.

**Sensitivity check.** Flipping the last bit of one gene in one golden pick (`picks[7].genome.lr`, `3f42611f2e1a23a1` → `…a0`) makes the Chromium test fail with exactly that diff. The golden was restored afterwards, and the Node test checks it byte for byte.

## Results (this machine: macOS 27.2, Apple Silicon)
| Engine | Version | detmath golden (05a) | GP golden (05b), main + worker | One `ts` birthDraw at 300 model rows, in the worker |
|---|---|---|---|---|
| Chromium | 156.0.8078.4 (Playwright) | PASS | PASS | 77 ms |
| WebKit | 27.2 (Playwright webkit-2370) | PASS | PASS | 87 ms |
| Firefox | 158.0 (stock, installed; WebDriver BiDi) | PASS | PASS | 105 ms |

Command: `npx playwright test detmath --project=chromium --project=webkit --project=firefox` → 6 passed (12.9 s).

**Timing.** The spec estimated 1–3 s for one ts draw at 300 rows. Measured: about 80–105 ms, one run each (`performance.now()` around `birthDraw` in the worker, after the probe's other work, so the JIT was warm). A cold first draw in a fresh tab may be slower. This timing was not measured.

## Caveats
- **Firefox is the stock browser, not Playwright's build.** Playwright's Firefox (`firefox-1555`, Firefox 157, from `npx playwright install firefox`) exits at launch with "Could not find profile folder" on this machine. It also does this when launched by hand with `-profile`, outside Playwright and outside any sandbox. The installed Firefox 158 launches fine. So the `firefox` project uses `channel: "moz-firefox"` (Playwright drives it over WebDriver BiDi). This tests the engine users actually run. But the project needs a Firefox installed in its standard location.
- **Coverage.** These results cover one machine (arm64) and one version of each engine. x86-64 and other versions were not run. The arithmetic is limited to IEEE-754 `+ − × ÷` and `Math.sqrt`, plus detmath (lint-enforced), which is why agreement is expected. This probe is evidence for that expectation, not proof for every engine.
- **CI runs Chromium only.** `webkit` and `firefox` are `default: false` projects, so `npx playwright test` and `npm run e2e` skip them. Re-run the command above after any change to `packages/detmath` or the GP and sampler code.
