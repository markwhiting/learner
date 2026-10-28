# blanktab v1 — design

**Date:** 2026-10-19 · **Status:** approved by user 2026-10-19 · **Hosting:** `markwhiting.github.io/blanktab` (GitHub Pages, repo `markwhiting/blanktab`)

Background: `docs/research/2026-10-16-research-synthesis-v0.1.md`. Settled decisions: `docs/decisions/0001`–`0009`. Rules every agent must follow: `AGENTS.md`.

Labels: **[user]** = decided by the user. **[agent]** = agent's call, made on the user's instruction to "make best choices", open to revision. Unlabelled sections are mechanics that follow from the labelled choices.

---

## 1. Intent

A web page that, when opened, becomes a newborn learner. It starts with random weights and no world knowledge, sees only pixels, acts only through a cursor, clicks, scrolls and keys, and learns continuously. Each tab is one individual. Individuals report what they tried and how well they developed back to GitHub. The population's shared knowledge (settings, consolidation rules, caregiving rules) improves over time, so later newborns learn faster. **[user]**

The scientific question: does learning how to retain, consolidate and reuse experience make later learning cheaper on a small compute budget? The long-term ambition is a strategic reasoner that uses online resources to achieve nontrivial goals. v1 is the first rung: a developmental learner in a controlled world, plus the population loop around it. **[user]**

### Why v1 looks like this
- Plain web pages can't see or control other sites, so v1 brings its own world (a "playground") inside the page. A browser extension for the real web comes in stage 2. **[user, 0001]**
- The PR loop exists from day one, because early work is a population-wide search over how to learn, and every run should inform the next. **[user, 0003]**
- Everything is built from scratch: no existing harness, no ML libraries. **[user, 0003; agent for ML libraries, §6]**

## 2. Goals, non-goals, anti-goals

**Goals (v1 is done when all are true).** v1 is a **bounded engineering build** **[user]**. Goal 2 requires that checkups run and their results are reported honestly; a curve of all zeros is an acceptable first result. Making learners pass milestones is the next phase, and the machine itself helps with that search.
1. Opening the page starts a learner that runs unattended for hours, survives reloads and tab freezes, and resumes from its last checkpoint.
2. The learner is measurably developing: checkups show milestones passed on fresh, never-seen variants.
3. Each session's log reaches the repo through a PR made from the tab, without a human in the loop.
4. A GitHub Action validates submissions, merges them by the rules in §9, and rebuilds the shared summary.
5. New sessions read the shared summary and choose their settings from it.
6. The three consolidation conditions (A, B, C in §5.6) all run, and the logs support comparing them by milestones per unit of compute.

**Non-goals for v1:**
- The real web, the browser extension, and the mini-web room.
- A live session-to-session layer (0004); v1 only keeps it possible (§10).
- Goal-directed tasks or any external reward channel.
- An LLM observer or caregiver (0009).
- Sharing weights through GitHub.
- Resistance to deliberately dishonest contributors.

**Anti-goals (failure even if it "works"):**
- Any pretrained weight, embedding, tokenizer or learned competence entering the learner.
- Privileged world state (element identities, room names, DOM-like structure, milestone detectors) reaching the learner's inputs.
- Automatically merging code from sessions, or running PR code with secrets.
- Reporting progress that isn't backed by checkups.
- Hard limits on the learner's eventual action vocabulary or world size.

## 3. System overview

```
┌──────────────────────────── browser tab (one session) ─────────────────────────────┐
│ main thread: UI (live view, stats, token setup) ◄── telemetry (read-only copy)     │
│                                                                                    │
│ worker: ┌── harness ───────────────────────────────────────────────────────────┐   │
│         │ loop · sleep · checkpoint · caregiver · checkups · FLOP meter · logs │   │
│         │        │ step(action) → observation           ▲ privileged trace     │   │
│         │   ┌────▼────┐  interface  ┌──────────┐        │                      │   │
│         │   │ learner │◄───────────►│  world   │────────┘                      │   │
│         │   └─────────┘  (pixels/   └──────────┘                               │   │
│         │                 actions only)                                        │   │
│         └──────────────────────────────────────────────────────────────────────┘   │
│ contributor: queue (IndexedDB) → GitHub REST (token in localStorage)               │
└───────────────────────────────────────────────┬────────────────────────────────────┘
                                                │ PRs from the contributor's fork
                     ┌──────────────────────────▼──────────────────────────┐
                     │ GitHub repo markwhiting/blanktab                     │
                     │ data/runs/ (logs) · knowledge/ (shared) · code       │
                     │ Actions: validate on PR · hourly gate merges,        │
                     │ rebuilds knowledge/summary.json, deploys Pages       │
                     └──────────────────────────────────────────────────────┘
```

All three components run in one dedicated Web Worker, in lockstep. The separation between them is enforced at the code level (§4), not by threads. **[agent]**

## 4. Repository layout and walls [user: one repo, 0002; agent: details]

```
AGENTS.md  CLAUDE.md  README.md
docs/                      research, decisions, specs, plans, open-questions
packages/
  interface/               Observation, Action, StepResult types; the only shared contract
  tensor/                  our tensor + autodiff library; CPU and WebGPU backends; FLOP counting
  learner/                 model, objectives, plasticity tiers, consolidation, policy
  world/                   desktop shell, rooms, procedural variants, renderer, privileged trace
  harness/                 loop, sleep, checkpoints, caregiver, checkups, milestones, picker, logs, contributor
app/                       the page: UI, worker bootstrap, token onboarding
tools/gate/                Node scripts used by Actions: validate, judge, compile summary
knowledge/                 shared knowledge (see §8)
data/runs/<learnerId>/     merged session logs (see §9)
.github/workflows/         validate.yml, gate.yml, deploy.yml
```

- **Language and tooling [agent]:** TypeScript (strict), Vite, Vitest, Playwright, ESLint, npm workspaces. This departs from `life`'s plain JS because shape-heavy numeric code and the boundary checks benefit from types.
- **The wall (enforced in CI) [agent]:**
  - `learner` may import only `interface` and `tensor`.
  - `world` may import only `interface`.
  - `harness` may import all of them.
  - Use ESLint `no-restricted-imports` per package, or dependency-cruiser. A violation fails the build.
- **Milestone detectors** live in `harness/` and read the world's *privileged trace*: what was under the cursor, which elements responded, which content was revealed, which page is shown. The learner never receives the trace.
- **`.gitignore`:** remove `runs/` (logs now live in `data/runs/`). Keep `checkpoints/` and add `.local/` for local scratch output.

## 5. The learner

### 5.1 Cold start [agent; resolves Q3]
- **At birth:** every weight is drawn from a seeded initializer, and the seed is logged.
- **Supplied architecture priors:** the network structure; the objectives in §5.4; the optimizer; foveated vision; the plasticity tiers; the action space.
- **Inherited:** the prior files in `knowledge/` (`settings-space.json`, `consolidation-prior.json`, `caregiver-prior.json`). Their combined content hash is logged as `knowledgeSha`. This deliberately excludes the hourly-rebuilt `summary.json` and `best.json`, so cohorts of sessions sharing a `knowledgeSha` stay stable. `summary.json` is read but is data, not inheritance. **No weights are inherited.** **[user: "fine with no or limited weights in GH"]**
- **Curriculum** (which rooms exist and the caregiver's rule) is a recorded prior.

### 5.2 What it sees [agent; resolves Q16]
The world renders at **320×200 RGB**. Each step the learner receives:
- **Periphery:** the whole screen downsampled 4× to **80×50×3**.
- **Fovea:** a **32×32×3** native-resolution crop centred on the cursor. This is a biological prior: cheap wide vision plus sharp vision where it is "looking", and it makes small text readable at low cost.
- **Proprioception:** its own cursor position (normalized x, y) and button-held state.

Nothing else. The cursor is drawn into the frame as part of the world.

### 5.3 What it can do [agent; resolves Q16]
One factored action per step:
- `move`: none or one of 8 directions × magnitude {1, 4, 16} px (25 options)
- `button`: none / press / release, so dragging is possible
- `scroll`: none / up / down
- `key`: none, a–z, 0–9, space, enter, backspace, tab, ←↑→↓ (45 options)

The key set and magnitudes are config entries versioned with `schemaVersion`, so they can grow.

**Time is lockstep [agent]:** the world advances exactly one tick per learner step, with a nominal 10 ticks per world-second for animations. This makes runs reproducible and independent of device speed.

### 5.4 What drives it [agent; resolves Q12]
1. **Prediction:** predict the next periphery frame (downsampled again to 40×25) and the next fovea frame (16×16), given the current state and the chosen action. The loss is mean squared error in pixel space. Pixel targets are chosen over latent targets because they cannot collapse.
2. **Curiosity as learning progress:** the intrinsic reward for a transition is *how much the learner just learned from it*: `r = loss_before − loss_after` on that transition, measured around the fast-weight update (§5.5). Rewarding learning rather than raw error avoids the "noisy TV" trap, where unpredictable noise looks permanently interesting.
3. **Action selection:** an advantage actor-critic on `r`, with an entropy bonus. The curiosity/entropy balance is a setting.

There is no external reward. Goal-directed behaviour is out of scope for v1 (§11).

### 5.5 Architecture and plasticity tiers [agent; resolves Q13; **amended by 0013**: residual predictor, head lr scale, relative curiosity, TBPTT-1]
- **Encoders:** two small conv nets, periphery → 128-d and fovea → 128-d. Convs are implemented as im2col plus matmul.
- **Core:** a GRU with a 256-d state. This is the *working memory* tier.
- **Heads:** frame predictor (MLP decoders), policy, value.
- **Size:** roughly 1–3 M parameters at the default width. Width is a device-relative setting.

| Tier | Mechanism v1 | Updated |
|---|---|---|
| Working state | GRU hidden state | every step |
| Episodic memory | ring buffer of transitions (compressed frames, action, losses, r, room-blind) in RAM, mirrored to IndexedDB at sleep | every step (insert); eviction per §5.6 |
| Fast plastic state | low-rank adapters (rank is a setting) on the core and predictor, high learning rate, decaying toward zero | every step |
| Slow parameters | base weights, low learning rate | only during sleep |
| Learning policies | consolidation scorer (condition C), plus settings chosen from `knowledge/` | scorer: each sleep; settings: across lifetimes via the population |
| Shared inheritance | `knowledge/` | via the gate (§9) |

**Sleep:** every `sleepInterval` steps, acting pauses and the world freezes. The learner then:
1. Replays selected episodic items to train the slow weights.
2. Folds the fast adapters into the slow weights (distillation by replay), then shrinks the adapters.
3. Checkpoints.
4. Flushes its log.

The order is fixed by design.

### 5.6 Consolidation — the core experiment [agent; resolves Q4]
**The decision consolidation makes:** given the episodic buffer, choose (a) which items to replay during sleep, with what weight, and (b) which items to evict when the buffer is full.

Signals available for each item: prediction error at insert; learning progress `r`; age; how often it has been replayed; and *interference*, meaning the loss increase on that item since it was stored.

| Condition | Tiers | Replay and eviction rule |
|---|---|---|
| **A: plain continual learning** | single tier: no adapters, no sleep split; slow weights updated online plus uniform replay | FIFO eviction, uniform replay |
| **B: hierarchy + fixed rule** | full tiers and sleep | priority = fixed weighted sum of the signals; replay ∝ priority; evict lowest priority |
| **C: hierarchy + learned rule** | same as B | priority from a tiny scorer (linear, over the same signals) whose weights are learned |

**How C's scorer learns:**
- **Within a lifetime:** an antithetic evolution-strategies step each sleep. Score the sleep under perturbed scorer weights ±σ (split the replay budget in half per sign, on cloned slow weights). Measure each by the post-sleep loss on a **probe set**: a held-out sample of the buffer that is never trained on, stratified by age so it measures both retention and fresh acquisition. Step the weights toward the better sign.
- **Across lifetimes:** the scorer's final weights are a knowledge candidate (§9). Its starting weights come from `knowledge/consolidation-prior.json`.

Each session's condition is assigned by the picker (§8) with each condition's share held at ≥20%, so the A/B/C comparison stays balanced.

## 6. Compute: our own tensor library [agent; resolves Q15]
- `packages/tensor` provides n-d float32 tensors, a reverse-mode autodiff tape, the Adam optimizer, and only the ops the learner needs: matmul, add/mul/elementwise, activations, reductions, softmax/log-softmax, im2col, GRU cell, layernorm, gather.
- **Two backends:**
  - **CPU (plain TypeScript):** the reference implementation and the fallback.
  - **WebGPU (WGSL kernels):** the fast path.
- **Tests:** every op is checked for CPU/GPU parity (tolerance) and for gradients by finite differences.
- **Why build it ourselves:** the user's from-scratch preference; plasticity and consolidation need non-standard update rules; no hidden pretrained assets; a small bundle; and **exact FLOP counting**. Every op reports its FLOPs, which gives the device-independent **compute unit** used throughout measurement (1 CU = 10⁹ FLOPs).
- **Target:** at least 20 learner steps per second at default width on a recent laptop GPU. Spike 2 checks this before the architecture is locked. If it falls short, shrink widths first; resolution comes last.

## 7. The world [user: desktop, caregiver, checkups, 0008; agent: details]

### 7.1 Desktop shell and rooms
- **The screen:** 320×200. A persistent **taskbar** (bottom 16 px) holds one icon per room. Clicking an icon brings that room's window forward. The rest of the screen is the active room.
- **Same physics everywhere:** cursor, press/release, scroll wheel, keys.
- **Rendering [agent]:** Canvas2D on an OffscreenCanvas inside the worker. The pixel buffer is read with `getImageData` each step. One bundled, openly licensed font is loaded with `FontFace`, so text renders identically on every machine. The learner doesn't know how anything is drawn. **[user: "don't mind how it is drawn"]**
  - *Amendment (2026-10-20, Plan 03):* rendering text with our own bitmap font instead of a FontFace font (Plan 03).

**v1 rooms [agent]:**

| Room | Content | Variants generated from a seed |
|---|---|---|
| **Crib** | sparse shapes that change colour or shape on hover and click | positions, shapes, palettes, responses |
| **Toys** | buttons, toggles, sliders, draggable blocks; state persists | layouts, which widgets, response types |
| **Hide-and-seek** | long scrolling pages, collapsible sections, tabs, pop-ups that close | page lengths, hidden-content placement |
| **Places** | a small linked site, 6–20 pages with link text and back buttons | graph shape, page contents, link positions |

Picture books, tools, games and the mini-web come later. The shell and generator interfaces must make adding a room a self-contained change.

**Content:** v1 rooms use procedurally generated shapes, colours and words. Words are drawn from an openly licensed English word list, rendered as pixels only. Later rooms draw on public-domain and openly licensed text (Project Gutenberg, Simple English Wikipedia).

### 7.2 Privileged trace
Each tick, the world emits to the harness only: element under the cursor and whether it responded; scroll extents and revealed fraction; active room and page; widget state changes. Detectors and the caregiver use this trace. It never crosses into the learner.

### 7.3 Caregiver [agent; the user called this "a meta-learner landscape"; **amended by 0016**: relative LP, settings from caregiver-prior.json while the picker is random, decides every `caregiverPeriod` steps only (not also after each sleep), observes only transitions within one room]
- **Learning progress per room:** `LP(room) = EMA_slow(loss) − EMA_fast(loss)`, computed from the learner's prediction loss while in that room.
- **When to move:** every `caregiverPeriod` steps (*0016: no longer also after each sleep*), the caregiver may *carry* the learner. It picks a room by softmax over LP with temperature τ, plus ε-uniform exploration.
- **How it moves:** by bringing the room's window forward through a harness command. This is logged as a caregiver action, never as a learner action.
- **Handover:** once the learner passes the **M6 (uses the taskbar)** checkup twice in a row, the caregiver enters *safety-net mode*. It only carries the learner if `LP < threshold` everywhere it has been for `stuckSteps`.
- **Settings:** `caregiverPeriod`, τ, ε, `threshold`, `stuckSteps`. They come from `knowledge/caregiver-prior.json` and the picker.

## 8. Milestones, checkups, and scoring [user: ladder, 0005; agent: v1 suite and format; **amended by 0016**: exact suite v1, module signature and location, checkups on the log schedule, frozen runs]

### 8.1 Milestone spec format
Each milestone is a module in `packages/harness/milestones/` that exports:
- `id`, `version`, `description`
- `room` and `prerequisites`
- `makeVariant(seed)`: builds a fresh room instance
- `budgetSteps`: the checkup's length
- `passes(trace, learnerStats) → boolean`

The suite has a version number (`milestoneSuiteVersion`), logged with every result. New milestones are added as sessions routinely pass the existing ones.

**Changing a threshold or criterion [agent]** requires a recorded decision in `docs/decisions/` and a `milestoneSuiteVersion` bump. Results are compared only within a suite version, so success can't be quietly redefined (AGENTS.md invariant 7).

### 8.2 v1 suite
v1 milestones are **spontaneous behaviours**: things a curious learner does unprompted, like a baby at a checkup. They don't depend on understanding an instruction.

| ID | Milestone | Pass criterion (initial thresholds; tune once real data exists) |
|---|---|---|
| M1 | Predicts its world | on a fresh crib/toys variant, prediction loss < 0.8 × the copy-last-frame baseline |
| M2 | Moves with purpose | fraction of steps with the cursor over interactive elements ≥ 2 × their area fraction |
| M3 | Prefers things that respond | click rate on responsive elements ≥ 2 × the rate on unresponsive ones, both normalized by area |
| M4 | Uncovers hidden content | reveals ≥ 60 % of a fresh long page's hidden content within budget |
| M5 | Explores places efficiently | distinct pages visited per step ≥ 1.5 × a random-walk clicker on the same fresh site |
| M6 | Uses the taskbar | ≥ 3 self-initiated room switches through the taskbar within budget |

### 8.3 Checkups
- **When:** at compute milestones 2ᵏ CU (k = 0, 1, 2…). On the first sleep after crossing each one, the harness runs the full suite.
- **How:** each milestone gets a variant from a seed derived from `hash(learnerId, k, milestoneId)`, which the learner has never seen. Learning (all weight updates) is **frozen** during checkups, and state is restored afterwards. Checkups therefore measure what the learner already knows, and the learner never trains on test material.
- **Score:** `score_k` = the number of milestones passed at checkup level k. The log records the full curve `{k: passedIds}`.
- **The picker's target:** `score` at level `k*`, the median of the highest level reached by sessions under the current `schemaVersion`. Sessions that didn't reach `k*` are excluded from that fit and kept in the log.

## 9. Shared knowledge and the contribution loop

### 9.1 `knowledge/` [agent; resolves Q14]

| File | Contents | Who changes it |
|---|---|---|
| `settings-space.json` | named settings, ranges, which are device-relative, `schemaVersion` | humans and coding agents (code-reviewed) |
| `summary.json` | compiled from `data/runs/`: one row per session `{settings, hostSig, condition, curve, schemaVersion, knowledgeSha}` | Actions only |
| `consolidation-prior.json` | starting weights for C's scorer | gate, on accepted knowledge PRs: `prior ← prior + 0.3·(candidate − prior)`, so one lucky session can't overwrite it |
| `caregiver-prior.json` | centre of the caregiver settings' search range, and the values used while the picker is still sampling randomly | humans and coding agents only. Caregiver settings are improved by the picker (§9.2), not by knowledge PRs |
| `best.json` | current best record per condition, plus thresholds | Actions only |

**v1 settings:**
- `condition` ∈ {A, B, C}
- `fastLR`, `slowLR`
- `adapterRank`
- `sleepInterval`
- `replayPerSleep`
- `bufferFraction` (of device memory)
- `width` (device-relative)
- `curiosityWeight`, `entropyWeight`
- caregiver τ, ε, `caregiverPeriod`

As in `life`, device-relative settings are fractions of host capacity, so all devices pool into one fit.

### 9.2 "What do I do given what I know" — the picker
At birth the session loads `knowledge/*` from the deployed Pages site, at the same commit as the code. It then:
1. Picks a condition, keeping each condition's share at ≥20 %.
2. Fits a Gaussian process for that condition over normalized settings. The target is the score at `k*`.
3. Picks settings by UCB. It samples uniformly while fewer than 8 usable rows exist.

The method used (random or UCB) is logged. Rows from other `schemaVersion`s are ignored. This mirrors `life`'s `experiment/picker.js`; it is the v1 placeholder for a learned meta-learner. **[agent, following user's "what it knows slowly changes" framing]**

### 9.3 Tokens and onboarding [user, 0007]
- **On load:** if `localStorage['blanktab:githubToken']` is empty, the page shows a short how-to, a link to a pre-filled token-creation page, and a paste field.
- **Before saving:** the page calls `GET /user` and checks the permissions it needs. A clear error is shown if the token is insufficient.
- **The learner always runs.** Without a token, submissions queue in IndexedDB and are sent once a token is added.
- **Token type [user]:** whichever type spike 1 shows works with the least access. A classic `public_repo` token is acceptable. **Hard rule [user]:** the token lives only in this browser's `localStorage`. It is never written to IndexedDB, logs, checkpoints, the live layer, telemetry, the repo, or any network destination other than `api.github.com`. Enforce this with a test that scans submitted payloads and checkpoints for the token string.
- **Shared storage accepted:** all `markwhiting.github.io` projects share one storage origin. The user accepted this risk. All keys are prefixed `blanktab:`.

### 9.4 What a tab submits [user: rolling PRs, logs kept, 0006; agent: mechanics; **amended by 0011**: one commit per flush, no heartbeat.json; **amended by 0014**: log schedule, seq gaps]
> **Superseded in part by decision 0014.** Logs are written at CU 8, 16, …, 512, then every 1024 CU (not at each sleep), as `<seq>.json`, with no heartbeat. A learner has at most one open PR; once the gate merges or closes it, the next PR starts on a fresh branch. A closed PR's files are dropped and the learner goes on (seq gaps are allowed); after 3 closed PRs in a row it stops contributing.
- **Contributors in v1 [agent]:** the owner and invited collaborators; public contribution comes later.
- **Setup (once per contributor):** if the token's user has push access to `markwhiting/blanktab` (the owner, or a collaborator), branches are pushed to upstream directly. Otherwise the tab ensures a fork exists and pushes there. In both cases, PRs target upstream `main`.
- **Log stream:** one branch per learner, `run/<learnerId>`, in the contributor's fork, with one draft PR to upstream `main`.
  - At each sleep, the tab commits a new file `data/runs/<learnerId>/<seq>.jsonl` (≤1 MB). It contains settings, hostSig, `knowledgeSha`, code SHA, condition, CU counter, wall time, checkup curve, sleep summaries, and caregiver actions.
  - The tab also commits `heartbeat.json`.
- **Knowledge stream:**
  - **When it fires:** when a session's checkup score at its newest level beats `best.json` for its condition at that level.
  - **What it contains:** a separate branch `knowledge/<learnerId>-<n>` and PR holding its final scorer weights (condition C only).
  - **What else it must include:** a `claim.json` that points at the log files supporting it.

### 9.5 The gate (GitHub Actions) [user: "a simple action that decides what to merge"; agent: rules; resolves Q18, Q7; **amended by 0011**: liveness from newest log, 16-file cap, 7-day max age, dev builds closed, validate not a required check; **amended by 0014**: merge every valid PR, seq gaps]
> **Superseded in part by decision 0014.** The final / 48 h heartbeat rule below and 0011's 7-day rule no longer apply: every valid session PR is merged at each hourly run. "Counters only increase" still holds for CU, steps and restores; for seq it now means strictly above the highest seq already on `main`, unique in the PR, with gaps allowed. A learner belongs to the author of the PR that first put its logs on `main`: the gate closes another author's PR for it ("Learner <id> belongs to @<owner>."), requires the title `run: <learnerId>`, and leaves undecided a PR whose author or binding it cannot read. A learner with nothing merged yet can be claimed by the first author to get a merge in.
- **`validate.yml` (on `pull_request`):**
  - Declares `permissions: contents: read` explicitly, so it is read-only for same-repo branches as well as forks. It uses no secrets.
  - Runs only `tools/gate/validate` from **`main`'s** copy of the code, against the PR's files *as data*. It never executes PR code.
  - Checks:
    - The PR touches only its own `data/runs/<learnerId>/` or the allowed `knowledge/` files.
    - The schema is valid.
    - Each file is ≤1 MB.
    - Counters (CU, seq) only increase.
    - Plausibility: CU per wall-second is within device bounds; checkup levels match CU; numbers are finite and within the bounds in `settings-space.json`.
  - Posts a status check.
- **`gate.yml` (hourly cron, plus manual trigger; `contents: write`, `pull-requests: write`; **amended by 0025**: also runs when validate completes, the cron kept as a backstop):**
  - **Log PRs:** merged if they are valid and either (a) the session marked itself final (§10), or (b) the heartbeat is more than **48 h** old. That is past the 24 h adoption window, so no tab can still be writing to the branch. Logs of failed runs are merged too, because unfavourable results are kept.
  - **Knowledge PRs:** merged if they are valid, and the claim is *consistent with* the cited logs (the claimed score equals the logged checkup result at that level; independent re-evaluation is impossible without weights, and dishonest contributors are a v1 non-goal), and the claimed score beats the **75th percentile** of sessions that started from the current knowledge SHA under the same condition and level (with at least 5 such sessions; before that, it must beat their median). Otherwise the PR is closed with a reason.
  - **Any PR touching code or other paths:** labelled `needs-human`. Never merged automatically.
  - **After merging:** rebuild `summary.json` and `best.json`, commit, and deploy Pages in the same job. Merges made with `GITHUB_TOKEN` don't trigger other workflows, so the job must do this itself.
- **Branch protection on `main`:** require the validate check. Only the gate's token and humans may merge.

## 10. Persistence, lifecycle, and the live layer [agent; resolves Q17; **amended by 0014**: no fixed lifetime]
> **Superseded in part by decision 0014.** There is no lifetime: a learner lives as long as its tab (and, through adoption, a later tab). `final: true` is written only when a life fails (or a test lifetime ends). New compatible code is reloaded onto with the learner resumed from a forced checkpoint; incompatible code leaves the learner running on its old code. Checkpoints are written every 5 minutes and with every log file, not at every sleep.
- **Lifetime [agent]:** a learner lives until one of the following happens, then becomes *final*:
  - it reaches `lifetimeCU` (default 2^`maxLevel` CU; `maxLevel` is a setting, default 12);
  - the deployed `schemaVersion` changes (it retires rather than resuming under different semantics);
  - its checkpoint goes unadopted for 24 h.

  On becoming final, the tab writes a last log file with `"final": true`, submits its knowledge candidate if it qualifies, and the next newborn starts. Code changes within the same `schemaVersion` are allowed mid-life. Every log file records the code SHA it was written under.
- **Identity:** each tab's learner has a `learnerId`, kept in `sessionStorage` so it survives reloads of that tab.
- **Checkpoints:** written at every sleep to IndexedDB, keyed by `learnerId`. They hold slow weights, adapters, optimizer state, GRU state, the episodic buffer, RNG state, and all counters.
- **Liveness and adoption:** a live learner holds a Web Lock named `blanktab:learner:<id>`. When a tab opens with no `learnerId`, it adopts any checkpoint whose lock is free (an orphan from a closed tab) **and** which is not final, matches the current `schemaVersion`, and was last checkpointed less than 24 h ago. Only otherwise does it create a newborn. This works with any number of tabs on one machine. **[user's multi-tab case]**
- **Freezes and discards:** the step loop runs in the worker and tolerates throttling. A discarded page resumes from its last checkpoint, losing at most one sleep interval. Only active compute is counted.
- **Storage limits:** device-relative. A dead learner's checkpoint is deleted once its final log has been submitted.
- **Live layer (0004):** not built in v1. Learner state is fully serializable and the harness has a clean place to attach a peer channel later, so that imitation or distillation can emerge rather than be scripted. **[user: "make possible, not hard-coded"]**

## 11. Explicitly deferred
- Goal-directed milestones such as "reach a target page". These need a goal channel and reward, an explicit design change under AGENTS.md invariant 2.
- Rooms: picture books, tools, games, mini-web.
- The browser extension (stage 2), the live layer, and LLM roles.
- Sharing weights, and architecture changes by the population.

## 12. Browser support [agent; resolves Q19]
- **Target:** any browser with WebGPU. Developed and tested in Chrome first.
- **CPU fallback:** the CPU backend runs where WebGPU is missing, at reduced width. This is recorded in `hostSig`.
- **CI:** runs unit tests on the CPU backend and a Playwright smoke test (page boots, learner steps, checkpoint and resume). GPU parity tests run locally.

## 13. Build order and spikes

**Spikes first.** Throwaway; each produces a short note in `docs/research/`:
1. **PR from a page:** using a pasted token, fork, branch, commit and open a PR against a scratch repo. Find which token type does this with the least access, and whether a pre-filled creation link works for it.
2. **WebGPU training speed:** a matmul plus GRU forward/backward at the §5.5 size. Measure steps per second.
3. **Rendering:** draw a 320×200 OffscreenCanvas scene in a worker and read it back at ≥60 fps.
4. **Lifecycle:** background, freeze and reload the tab. Confirm the worker throttling behaviour and that IndexedDB checkpoint/resume and Web Lock adoption work.

**Then build, in order:**
1. Scaffold and the walls.
2. `tensor` on CPU.
3. `tensor` on WebGPU.
4. World: shell plus the four rooms plus the trace.
5. Learner condition A.
6. Harness: loop, sleep, checkpoint, caregiver, checkups, FLOP meter, logs.
7. Contributor and onboarding.
8. Gate and summary.
9. Picker.
10. Conditions B and C.
11. Unattended runs; first A/B/C comparison.

The first operational milestone is reached when a tab runs overnight unattended, its logs land in `main`, and the next tab's picker uses them.

## 14. Left to the builder
- Exact layer sizes, initializers and kernel tiling.
- UI layout and visual style. The page should show: what the learner sees, its age in steps and CU, its milestone curve, its current settings, and its contribution status.
- Exact generator designs for each room, within §7.1.
- Exact log JSON schema, within the fields in §9.4. Version it.
- Choice between ESLint and dependency-cruiser for the wall.
