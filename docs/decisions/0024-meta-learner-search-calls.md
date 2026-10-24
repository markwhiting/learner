# 0024 — Plan 05b's calls: the meta-learner search

- **Date:** 2026-10-24
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (every call here is an agent's call, not yet confirmed). "Inert until signal" is the user's condition (plan 05b); how it is made structural is the agent's.
- **Resolves:** the choices plan 05b (`docs/plans/2026-10-24-05b-meta-learner-search.md`) makes for the user, including its spec gaps

## Context

Plan 05b builds the meta-learner's search: the GP fit in the gate, Thompson-sampling and confirmation draws in the tab, and automatic promotion (spec `docs/specs/2026-10-22-meta-learner-v1-design.md` §4.1, §4.3–§4.5, §7). It is built and deployed while the signal check has not answered, so it must stay switched off until that check says "signal". Several choices along the way belong to the user but were not asked. This file records each as an agent's call, so it can be confirmed or reversed. Later 05b tasks add their sections here.

## Decision

### Posterior schema 2 [agent]

- **`data/meta/posterior.json` moves to `schemaVersion` 2.** `packages/meta/src/posterior.ts` exports `POSTERIOR_SCHEMA = 2`. A v2 file fills what 05a left empty:
  - `models.gpu`: the GP's genes (the cube's axes), its rows (learnerId and unit-cube x), its targets y, its hyperparameters and the fit's record.
  - `incumbent`: the posterior-mean argmax, its genome sha, its mean and P(beats centre).
  - `confirmation`: the confirmation being served, with its seed base (10,000 + 1,000·id), both genomes, and each arm's target, count so far and remaining need.
  - `page.rowsSinceDeclaration`: the eligible rows since the last confirmation was declared (spec §4.4's "≥ 100 rows" condition).
- **A v2 file is strict in every part.** Exact keys at every level. At most 1,000 rows, with learnerIds strictly increasing. x in [0, 1]. Noise ≥ 1e-3. Four fit starts. Arm needs never above their targets. A candidate that is not the centre. Any violation makes the whole file unreadable (`parsePosterior` → null), and the tab then draws `centre`.
- **05b code still reads v1.** A v1 file is checked exactly as 05a checks it. It reads as `models: {}`, `incumbent: null`, `confirmation: null` and `rowsSinceDeclaration: 0`. 05a wrote nothing else, so a 05b tab draws from the last v1 file exactly what a 05a tab drew. `packages/meta/test/fixtures/draws.json` is unchanged and still passes.
- **Old tabs draw centre from a v2 file until they reload (05a ruling, Task 4).** 05a's parser checks `schemaVersion !== 1 → null`, so a 05a tab treats every v2 file as unreadable and draws `centre`, which is always a valid draw. Once it reloads onto new code (0014), it reads v2.
- **Any other `schemaVersion`** (0, 3, a string) is unreadable.

### The mode guard [agent; implements the user's "inert until signal"]

"Inert until signal" does not rest on a flag. It is structural, at four points (plan 05b, Global Constraints):

1. **The gate computes the search only in `search`.** `buildMeta` computes `models`, `incumbent` and `confirmation`, writes confirmation or promotion files, and touches `knowledge/genome.json`, only when `mode === "search"`.
2. **Only the signal check can set `search`.** `mode` is `modeOf(calibration)`. `calibration.outcome` becomes `"signal"` only in `nextLook`, from the pinned signal evaluator's `score === 1`. No other code path sets it, and no flag, environment variable or option can. Tests and the simulation reach `search` only through data (planted calibration lives), never through a code switch.
3. **A tab refuses search state outside `search`.** `parsePosterior` returns null for a v2 posterior that carries a model, an incumbent or a confirmation while its mode is `calibrate`, `paused` or `flat`. The tab then draws `centre`. An incumbent also needs a model. So a hand-edited or mis-built posterior cannot switch a tab on. (Task 1; tested in `packages/meta/test/posterior.test.ts`.)
4. **The draw reaches the new methods only in `search`.** `birthDraw` reaches `ts` and `confirm` only for mode `search` on `gpu`.

## Alternatives considered

- **Keep `schemaVersion` 1 and add the new fields to it.** Rejected. 05a's parser accepts any object for `models` and `incumbent`, but checks the page's exact keys, so a new page field would make 05a tabs refuse the file anyway, and silently. A version bump says so on purpose, and both outcomes are the same safe draw (`centre`).
- **A "search enabled" flag in the posterior or the config.** Rejected. A flag is one more thing that can be set by hand. The guard instead makes the search's state itself unreadable outside the mode that only the signal check sets.

## Consequences

- A v2 posterior is all-or-nothing. A single bad number in the model makes every tab served that file draw `centre` until the gate writes a good one. That is the intended failure mode: unfavourable but safe.
- During a deploy, a 05a tab served a v2 file draws `centre` and a 05b tab served the last v1 file draws as 05a did (Review Focus 1). Task 13 checks the first pairing against the real 05a code.
- Later 05b tasks add their calls below this point.
