import { test, expect, type Page, type Request } from "@playwright/test"
import { fakeGithub } from "../../packages/harness/test/fakeGithub"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"
import { createHash } from "node:crypto"
import { birthDraw, genomeSha, parsePosterior } from "@blanktab/meta"
import { spacesOf, validateSessionPr } from "../../tools/gate/validate"
import { GENOME_SPACE } from "../../packages/harness/test/helpers"
import { CENTRE, GENOME_JSON, SPACE_JSON, posteriorText } from "../../app/test/posteriors"

const TOKEN = "test-token"
const UP = "markwhiting/learner"
/** The real learner on the CPU at tiny widths: a 0.05 CU life is a few dozen steps, with a sleep (a curve point) every 5. */
const KNOBS = "testBackend=cpu&testWidths=tiny&testSleepInterval=5"
const LIFETIME_CU = 0.15
/** No speed knob: the genome's cap (30 steps a second), so the life runs the genome it logs. */
const PAGE = `/learner/?${KNOBS}&testLifetimeCU=${LIFETIME_CU}`
/** A life that outlasts the test. */
const LONG = `/learner/?${KNOBS}&testLifetimeCU=50&testStepsPerSecond=20`
const TINY = { conv1: 1, conv2: 1, enc: 2, hidden: 2 }
const ID_RE = /lrn_[0-9a-f]{16}/

const row = (id: string) => ({
  learnerId: id, settings: { moveProbability: 0.5 }, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0,
})
/**
 * The genome e2e learners are born from: the repo's space, with a centre of the repo's but a small memory and replay
 * (16 replays a sleep, a 500-frame buffer), so a life's length in steps is predictable.
 */
const GENOME = { ...(GENOME_JSON as object), values: { ...CENTRE, replayPerSleep: 16, bufferSize: 500 } }
const GENOME_SHA = await genomeSha(GENOME_SPACE.schemaVersion, GENOME.values)
/** A paused posterior built for that centre (CPU-class draws: the centre, or 1 in 5 uniform). */
const POSTERIOR = posteriorText("paused", { calibrationLives: 34, calibrationTarget: 45, looks: 1 }, undefined, { genomeSha: GENOME_SHA, values: GENOME.values })
const POSTERIOR_SHA = createHash("sha256").update(POSTERIOR).digest("hex")
/** A learnerId whose draw under POSTERIOR is the centre: the loop's learner is pinned to it, so its settings are known. */
const CENTRE_ID = Array.from({ length: 50 }, (_, i) => `lrn_${(0xc0de00 + i).toString(16).padStart(16, "0")}`).find(learnerId =>
  birthDraw({ learnerId, space: GENOME_SPACE, centre: GENOME.values, centreSha: GENOME_SHA, widthsClass: "cpu", posterior: parsePosterior(POSTERIOR), posteriorSha: POSTERIOR_SHA }).method === "centre")!
const KNOWLEDGE: Record<string, unknown> = {
  "summary.json": { schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa"), { ...row("lrn_bbbbbbbbbbbbbbbb"), cu: 1024.5, lossRatio: 0.9123, heldOutRatio: 1.0456 }] },
  // The e2e build's own code ("dev") and checkpoint format: no test here sees new code (tests/e2e/updates.spec.ts does).
  "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev", learnerStateVersion: LEARNER_STATE_VERSION },
  "genome-space.json": SPACE_JSON,
  "genome.json": GENOME,
  // Served as its bytes (a string), as deploy copies data/meta/posterior.json.
  "posterior.json": POSTERIOR,
  // As deployed (knowledge/caregiver-prior.json): the caregiver's settings for newborns.
  "caregiver-prior.json": { schemaVersion: 1, settings: { caregiverPeriod: 500, caregiverTau: 0.05, caregiverEpsilon: 0.1, caregiverThreshold: 0.01, caregiverStuckSteps: 2000 } },
}

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "authorization, content-type, x-github-api-version, accept",
  "access-control-allow-methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
}

/** Mocks knowledge/*.json and api.github.com, and records every request the page (or its worker) makes. */
async function setup(page: Page, opts: { knowledge: "ok" | "unreachable"; token?: string }) {
  const gh = fakeGithub({ token: TOKEN, login: "tester", canPush: true })
  const all: Request[] = []
  page.context().on("request", r => all.push(r))
  await page.route("**/knowledge/*.json", route => {
    if (opts.knowledge === "unreachable") return route.abort("internetdisconnected")
    const name = new URL(route.request().url()).pathname.split("/").pop()!
    const body = KNOWLEDGE[name]
    if (body === undefined) return route.fulfill({ status: 404 })
    return typeof body === "string" ? route.fulfill({ body, contentType: "application/json" }) : route.fulfill({ json: body })
  })
  await page.route("https://api.github.com/**", async route => {
    const req = route.request()
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS })
    const res = await gh.fetchFn(req.url(), { method: req.method(), headers: await req.allHeaders(), ...(req.postData() ? { body: req.postData()! } : {}) })
    return route.fulfill({ status: res.status, headers: { ...CORS, "content-type": "application/json" }, body: await res.text() })
  })
  if (opts.token !== undefined) {
    const token = opts.token
    await page.addInitScript(t => { if (localStorage.getItem("blanktab:githubToken") === null) localStorage.setItem("blanktab:githubToken", t) }, token)
  }
  const githubCalls = () => gh.requests.filter(r => r.method !== "OPTIONS")
  const commits = () => githubCalls().filter(r => r.method === "POST" && r.url === `https://api.github.com/repos/${UP}/git/commits`)
  return { gh, all, githubCalls, commits }
}

const stepsLived = async (page: Page) => Number((await page.getByTestId("steps").textContent())?.match(/^(\d+)/)?.[1] ?? NaN)
const cuLived = async (page: Page) => Number((await page.getByTestId("cu").textContent())?.match(/^([\d.]+) (?:of |CU)/)?.[1] ?? NaN)
/** The log files in this page's outbox for learner `id`, oldest first. */
const outboxOf = (page: Page, id: string) => page.evaluate(i => new Promise<{ seq: number; final: boolean; stepsTotal: number; codeSha: string }[]>((res, rej) => {
  const req = indexedDB.open("blanktab")
  req.onerror = () => rej(req.error)
  req.onsuccess = () => {
    const all = req.result.transaction("outbox").objectStore("outbox").getAll()
    all.onsuccess = () => {
      req.result.close()
      type F = { seq: number; final: boolean; stepsTotal: number; codeSha: string }
      res((all.result as { learnerId: string; file: F }[]).filter(r => r.learnerId === i).map(r => r.file).sort((a, b) => a.seq - b.seq))
    }
    all.onerror = () => rej(all.error)
  }
}), id)
const curvePoints = async (page: Page) => Number(await page.getByTestId("curve").getAttribute("data-points"))

/** Writes rows straight into the page's IndexedDB (as an earlier page version or a crashed life would have left them). */
async function seedDb(page: Page, rows: { outbox?: unknown[]; checkpoints?: unknown[] }) {
  await page.evaluate(r => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const tx = req.result.transaction(["outbox", "checkpoints"], "readwrite")
      for (const row of r.outbox ?? []) tx.objectStore("outbox").put(row)
      for (const row of r.checkpoints ?? []) tx.objectStore("checkpoints").put(row)
      tx.oncomplete = () => { req.result.close(); res() }
      tx.onerror = () => rej(tx.error)
    }
  }), rows)
}

test("the loop: knowledge in, the real learner lives its life in CU, its v2 log goes to GitHub as one PR, a newborn follows", async ({ page, baseURL }) => {
  const { gh, all, githubCalls } = await setup(page, { knowledge: "ok" })
  await page.addInitScript(i => { if (!sessionStorage.getItem("blanktab:learnerId")) sessionStorage.setItem("blanktab:learnerId", i) }, CENTRE_ID)
  await page.goto(PAGE)

  // Sessions before this one: the visible proof that the loop is closed.
  await expect(page.getByText("2 sessions before this one")).toBeVisible()
  await expect(page.getByText(/offline/i)).toHaveCount(0)

  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(/lrn_[0-9a-f]{16}/)
  const id = (await learner.textContent())!.match(/lrn_[0-9a-f]{16}/)![0]
  expect(await page.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(id)
  expect(id).toBe(CENTRE_ID)
  // The settings it was born with: the genome's (its draw under the posterior is the centre), as the learner uses them.
  await expect(page.getByTestId("setting")).toContainText("replayPerSleep = 16")
  await expect(page.getByTestId("setting")).toContainText("bufferSize = 500")
  await expect(page.getByTestId("setting")).toContainText("maxStepsPerSec = 30") // the genome's cap is the cap it lives at
  await expect(page.getByTestId("setting")).not.toContainText("moveProbability")
  await expect(page.getByTestId("recipe-text")).toHaveText("the current recipe")
  await expect(page.getByTestId("meta-state")).toHaveText("Paused: the score can't yet tell recipes apart; measuring more (34 of 45).")
  await expect(page.getByTestId("backend")).toHaveText(/^CPU/)

  // Steps, compute and the loss curve all move.
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
  await expect(page.getByTestId("cu")).toContainText(`of ${LIFETIME_CU} CU`)
  await expect.poll(() => cuLived(page)).toBeGreaterThan(0)
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
  await expect(page.getByTestId("ratio")).toHaveText(/^\d+(\.\d+)? \(its guesses are/)
  // The chart and the ratio are on the fixed held-out set (decision 0015 §3); the own-stream ratio is secondary text.
  await expect(page.getByTestId("heldout-label")).toHaveText("on a fixed test it never trains on")
  await expect(page.getByTestId("own-ratio")).toHaveText(/^On what it chose to look at \(which it also trains on\): \d+(\.\d+)?$/)
  await expect(page.getByTestId("rate")).toHaveText(/^[\d.]+ steps\/s$/)
  await expect(page.getByTestId("curve-legend")).toContainText("how wrong its guesses are")
  await expect(page.getByTestId("curve-legend")).toContainText("the lazy guess: nothing changes")

  // No token: the panel is offered as one folded line; opened, it explains how to get one; nothing has gone to GitHub.
  const panel = page.getByTestId("token-panel")
  await expect(panel).toBeVisible()
  await expect(panel.getByLabel(/GitHub token/i)).toBeHidden()
  await panel.getByText("Share this learner's logs").click()
  await expect(panel.getByLabel(/GitHub token/i)).toBeVisible()
  await expect(page.getByTestId("change-token")).toBeHidden()
  await page.screenshot({ path: test.info().outputPath("page-notoken.png"), fullPage: true })
  await expect(panel.getByRole("link")).toHaveAttribute("href", "https://github.com/settings/tokens/new?scopes=public_repo&description=learner")
  await expect(page.getByTestId("contribution")).toContainText(/token/i)
  expect(githubCalls()).toHaveLength(0)

  await panel.getByLabel(/GitHub token/i).fill(TOKEN)
  await panel.getByRole("button", { name: "Check & save" }).click()
  await expect(panel).toBeHidden()
  expect(githubCalls()[0]?.url).toBe("https://api.github.com/user") // checkToken ran before the token was used

  // The life goes final below the first crossing (8 CU = 2^3), so its one log file is the final one: data/runs/<id>/0.json, in one PR.
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 30_000 })
  const pulls = () => githubCalls().filter(r => r.method === "POST" && r.url === `https://api.github.com/repos/${UP}/pulls` && JSON.parse(r.body ?? "{}").head === `markwhiting:run/${id}`)
  await expect.poll(() => pulls().length, { timeout: 30_000 }).toBe(1)
  expect(gh.prFiles(gh.pulls[0]!)).toEqual([`data/runs/${id}/0.json`])
  expect(githubCalls().some(r => r.method === "PUT" || r.url.includes("/contents/") || (r.body ?? "").includes("heartbeat"))).toBe(false)
  await expect(page.getByTestId("change-token")).toBeVisible()
  const final = gh.file(UP, `run/${id}`, `data/runs/${id}/0.json`) as Record<string, unknown> & { cu: number; evalCU: number; curve: { steps: number; predLoss: number; baselineLoss: number; heldOut?: { loss: number; baseline: number } }[]; settings: Record<string, number> }
  expect(final).toMatchObject({
    schemaVersion: 2, seq: 0, final: true, sessionsSeenAtBirth: 2, knowledgeSha: "c".repeat(64), codeSha: "dev", condition: "A",
    widths: TINY, hostSig: { webgpu: false }, restores: 0, lostCU: 0,
  })
  expect(final.cu).toBeGreaterThanOrEqual(LIFETIME_CU)
  expect(final.curve.length).toBeGreaterThan(0)
  for (const p of final.curve) expect(Number.isFinite(p.predLoss) && Number.isFinite(p.baselineLoss)).toBe(true)
  // Every point at a sleep (every 5 steps) carries the held-out score, the end-of-life point none; its compute is
  // billed apart from the life's.
  const final_ = final.curve
  expect(final_.some(p => p.heldOut)).toBe(true)
  for (const p of final_) {
    if (p.steps % 5 === 0) expect(p.heldOut!.baseline > 0 && Number.isFinite(p.heldOut!.loss)).toBe(true)
    else expect(p.heldOut).toBeUndefined()
  }
  expect(final.evalCU).toBeGreaterThan(0)
  expect(final.settings).toMatchObject({ replayPerSleep: 16, bufferSize: 500, maxStepsPerSec: 30 })
  expect(final.settings).not.toHaveProperty("moveProbability")
  // Its genome block: the centre drawn under the posterior's bytes; the gate accepts the file as the learner's first PR.
  expect(final.meta).toMatchObject({ role: "search", method: "centre", posteriorSha: POSTERIOR_SHA, genome: GENOME.values, genomeSha: GENOME_SHA })
  const v = await validateSessionPr([{ path: `data/runs/${id}/0.json`, content: JSON.stringify(final, null, 2) }], [], { spaces: spacesOf(GENOME_SPACE) })
  expect(v.errors).toEqual([])

  // A newborn with a new learnerId takes its place.
  await expect(learner).not.toHaveText(id)
  await expect(learner).toHaveText(/lrn_[0-9a-f]{16}/)
  expect(await page.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).not.toBe(id)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)

  // Review Focus 1: no host but api.github.com and the page's own origin; the token is in no body, no file, no IndexedDB row.
  const origin = new URL(baseURL!).origin
  const strays = all.map(r => r.url()).filter(u => !u.startsWith("data:") && !u.startsWith("blob:") && new URL(u).origin !== origin && new URL(u).origin !== "https://api.github.com")
  expect(strays).toEqual([])
  for (const r of all) expect(r.postData() ?? "").not.toContain(TOKEN)
  for (const r of all) if (new URL(r.url()).origin !== "https://api.github.com") expect(JSON.stringify(await r.allHeaders())).not.toContain(TOKEN)
  for (const r of gh.requests) {
    const content = r.body ? (JSON.parse(r.body) as { content?: string }).content : undefined
    if (content) expect(gh.decode(content)).not.toContain(TOKEN)
  }
  for (const b of gh.blobs.values()) expect(gh.decode(b)).not.toContain(TOKEN)
  const idb = await page.evaluate(() => new Promise<string>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const get = req.result.transaction("outbox").objectStore("outbox").getAll()
      get.onsuccess = () => res(JSON.stringify(get.result))
    }
  }))
  expect(idb).toContain(id)
  expect(idb).not.toContain(TOKEN)
})

test("knowledge unreachable: the page says offline and the learner runs anyway", async ({ page }) => {
  const { githubCalls } = await setup(page, { knowledge: "unreachable" })
  await page.goto(LONG)
  await expect(page.getByTestId("knowledge")).toHaveText(/offline/i)
  await expect(page.getByText("0 sessions before this one")).toBeVisible()
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
  expect(githubCalls()).toHaveLength(0)
})

test("cadence: with a token, a whole life goes to GitHub as one commit, sent when the final log is queued", async ({ page }) => {
  const { gh, githubCalls, commits } = await setup(page, { knowledge: "ok", token: TOKEN })
  // At the CPU half widths (not tiny): far more compute per step, so 8 CU takes seconds, not a minute.
  await page.goto("/learner/?testBackend=cpu&testSleepInterval=50&testLifetimeCU=12&testStepsPerSecond=300") // a log at 8 CU (not final) and the final one, a third of the life later
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  await expect.poll(() => cuLived(page), { timeout: 60_000 }).toBeGreaterThan(8.05) // the non-final log is queued by now
  expect(githubCalls().filter(r => r.method !== "GET")).toEqual([]) // ...and nothing was written for it
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 60_000 })
  await expect.poll(() => gh.pulls.length, { timeout: 30_000 }).toBe(1)
  expect(commits()).toHaveLength(1)
  expect(gh.prFiles(gh.pulls[0]!)).toEqual([0, 1].map(s => `data/runs/${id}/${s}.json`))
  const files = [0, 1].map(s => gh.file(UP, `run/${id}`, `data/runs/${id}/${s}.json`) as { cu: number; final: boolean })
  expect(files.map(f => f.final)).toEqual([false, true])
  expect(files[0]!.cu).toBeGreaterThanOrEqual(8)
  expect(files[1]!.cu).toBeGreaterThanOrEqual(12)
})

test("token stored: a small 'change token' control clears it and brings the panel back", async ({ page }) => {
  await setup(page, { knowledge: "ok", token: TOKEN })
  await page.goto(LONG)
  const panel = page.getByTestId("token-panel")
  const change = page.getByTestId("change-token")
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  await expect(change).toBeVisible()
  await expect(panel).toBeHidden()
  await page.screenshot({ path: test.info().outputPath("page-token.png"), fullPage: true })
  await change.click()
  await expect(panel).toBeVisible()
  await expect(panel.getByLabel(/GitHub token/i)).toBeVisible() // asking to change the token opens the form
  await expect(change).toBeHidden()
  expect(await page.evaluate(() => localStorage.getItem("blanktab:githubToken"))).toBeNull()
  await expect(page.getByTestId("contribution")).toContainText(/token/i)
})

test("permission error: the token panel comes back (with 'change token') instead of a dead end", async ({ page }) => {
  const { gh } = await setup(page, { knowledge: "ok", token: TOKEN })
  gh.interceptors.push(r => (r.method === "POST" && r.url.endsWith("/git/blobs") ? new Response('{"message":"Resource not accessible by personal access token"}', { status: 403 }) : undefined))
  await page.goto(PAGE)
  await expect(page.getByTestId("token-panel")).toBeHidden()
  await expect(page.getByTestId("contribution")).toContainText("can't write to the repo", { timeout: 30_000 })
  await expect(page.getByTestId("token-panel")).toBeVisible()
  await expect(page.getByTestId("change-token")).toBeVisible()
  await page.screenshot({ path: test.info().outputPath("page-permission.png"), fullPage: true })
})

test("an unexpected failure while checking a token shows an error instead of hanging", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.addInitScript(() => {
    const orig = Storage.prototype.setItem
    Storage.prototype.setItem = function (k: string, v: string) { if (k === "blanktab:githubToken") throw new Error("storage is full"); return orig.call(this, k, v) }
  })
  await page.goto(LONG)
  const panel = page.getByTestId("token-panel")
  await panel.getByText("Share this learner's logs").click()
  await panel.getByLabel(/GitHub token/i).fill(TOKEN)
  await panel.getByRole("button", { name: "Check & save" }).click()
  await expect(page.locator("#token-error")).toHaveText(/went wrong/i)
  await expect(panel.getByRole("button", { name: "Check & save" })).toBeEnabled()
})

test("the learner always runs: a worker error starts a newborn", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const broken = "lrn_badbadbadbadbad0"
  // A checkpoint whose learner state the session refuses: the life throws at start.
  await seedDb(page, { checkpoints: [{ learnerId: broken, savedAt: new Date().toISOString(), steps: 10, cu: 0.01, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0, learnerState: { garbage: true } }] })
  await page.evaluate(id => sessionStorage.setItem("blanktab:learnerId", id), broken)
  await page.reload()
  await expect(page.getByTestId("life")).toContainText("The learner stopped")
  await expect(page.getByTestId("learner-id")).not.toHaveText(broken, { timeout: 10_000 })
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
})

test("a learner born on an older version (another learner state version): a plain notice, its life ended on the record, a newborn", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const older = "lrn_01de01de01de01de"
  const settings = { lr: 1e-3, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 500, replayPerSleep: 16, headLrScale: 0.1, exploreEpsilon: 0.1, residualDecay: 1e-3 }
  // An older version's learner state: its birth facts read fine, but the learner cannot load it.
  await seedDb(page, { checkpoints: [{ learnerId: older, savedAt: new Date().toISOString(), steps: 10, cu: 0.01, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0,
    codeSha: "a".repeat(40), learnerState: { version: LEARNER_STATE_VERSION - 1, seed: 5, widths: TINY, settings } }] })
  await page.evaluate(id => sessionStorage.setItem("blanktab:learnerId", id), older)
  await page.reload()
  const life = page.getByTestId("life")
  await expect(life).toHaveText("This learner was born on an older version; a newborn takes its place.")
  await expect(life).not.toHaveClass(/error/)
  await expect(page.getByTestId("learner-id")).not.toHaveText(older, { timeout: 10_000 })
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  // Its life is still ended on the record, under the code it was lived on.
  const files = await outboxOf(page, older)
  expect(files).toEqual([expect.objectContaining({ seq: 0, final: true, stepsTotal: 10, codeSha: "a".repeat(40) })])
})

test("the worker cannot load (its asset is gone after a deploy) and newer code is published: the page reloads, once per code", async ({ page }) => {
  const NEW = "e".repeat(40)
  await setup(page, { knowledge: "ok" })
  // The deployed manifest now names other code; this build's worker asset no longer exists.
  await page.route("**/knowledge/manifest.json", route => route.fulfill({ json: { ...(KNOWLEDGE["manifest.json"] as object), codeSha: NEW } }))
  let loads = 0
  page.on("load", () => { loads++ })
  await page.route(/\/assets\/worker-[^/]*\.js/, route => route.fulfill({ status: 404, body: "" }))
  await page.goto(LONG)
  // It reloads for the new code (the guard allows two tries per code), then gives up and says why, instead of
  // starting a newborn every 2 s forever.
  await expect.poll(() => loads, { timeout: 30_000 }).toBe(3)
  await expect(page.getByTestId("life")).toContainText("could not load", { timeout: 10_000 })
  expect(JSON.parse((await page.evaluate(() => sessionStorage.getItem("blanktab:reloadTarget")))!)).toEqual({ codeSha: NEW, tries: 2 })
  await page.waitForTimeout(3_000)
  expect(loads).toBe(3)
})

test("a learner with logs but no checkpoint (e.g. the old flailer's) cannot be resumed: a newborn takes its place", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const old = "lrn_01d01d01d01d01d0"
  const file = { schemaVersion: 1, learnerId: old, seq: 0, codeSha: "dev", knowledgeSha: "0".repeat(64), writtenAt: new Date().toISOString(), stepsTotal: 500,
    settings: { moveProbability: 0.5 }, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: false }
  await seedDb(page, { outbox: [{ learnerId: old, seq: 0, sent: false, file }] })
  await page.evaluate(id => sessionStorage.setItem("blanktab:learnerId", id), old)
  await page.reload()
  await expect(page.getByTestId("life")).toContainText("cannot be resumed")
  await expect(page.getByTestId("life")).not.toContainText("stopped")
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  await expect(page.getByTestId("learner-id")).not.toHaveText(old)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
})

test("a reload resumes the learner from its checkpoint; a duplicated tab (same sessionStorage) gets a newborn", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(ID_RE)
  const id = (await learner.textContent())!.match(ID_RE)![0]
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)

  await page.reload()
  await expect(learner).toHaveText(id)
  await expect(page.getByTestId("status")).toContainText("resumed from its checkpoint")
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)

  // Chrome's "Duplicate tab" copies sessionStorage, so the copy starts with the same learnerId.
  const dup = await context.newPage()
  await dup.addInitScript(i => { if (sessionStorage.getItem("blanktab:dup") === null) { sessionStorage.setItem("blanktab:learnerId", i); sessionStorage.setItem("blanktab:dup", "1") } }, id)
  await dup.goto(LONG)
  // The copy first waits up to OWN_LOCK_WAIT_MS (10 s) for "its" learner's lock, which the original holds, then a newborn.
  await expect(dup.getByTestId("learner-id")).toHaveText(ID_RE, { timeout: 20_000 })
  const dupId = (await dup.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  expect(dupId).not.toBe(id)
  expect(await dup.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(dupId)
  await expect(learner).toHaveText(id) // the original keeps its learner
  await expect.poll(() => stepsLived(dup)).toBeGreaterThan(0)
})

test("a local-only learner (3 PRs closed in a row): resumed, it keeps learning and the page says it no longer contributes", async ({ page }) => {
  await setup(page, { knowledge: "ok", token: TOKEN })
  await page.goto(LONG)
  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(ID_RE)
  const id = (await learner.textContent())!.match(ID_RE)![0]
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
  // What the contributor records after the gate closed three of its PRs unmerged (packages/harness contributor settle).
  await page.evaluate(i => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onsuccess = () => {
      const tx = req.result.transaction("contribution", "readwrite")
      tx.objectStore("contribution").put({ learnerId: i, lastPr: 3, closedInARow: 3, localOnly: true })
      tx.oncomplete = () => { req.result.close(); res() }
      tx.onerror = () => rej(tx.error)
    }
    req.onerror = () => rej(req.error)
  }), id)
  await page.reload()
  await expect(learner).toHaveText(id)
  await expect(page.getByTestId("life")).toHaveText("This learner's logs were rejected; it keeps learning here but no longer contributes.")
  const before = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(before)
  await page.screenshot({ path: test.info().outputPath("page-local-only.png"), fullPage: true })
})

test("a closed tab's learner is adopted by the next new tab; surplus and stale orphans are deleted", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0) // its birth checkpoint is written
  // Six more closed tabs' learners: copies of its checkpoint saved 1..5 hours ago, and one 25 hours ago (stale).
  const others = [1, 2, 3, 4, 5, 25].map(h => ({ learnerId: `lrn_${h.toString(16).padStart(16, "0")}`, hoursAgo: h }))
  await page.evaluate(({ id, others }) => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const db = req.result
      const get = db.transaction("checkpoints").objectStore("checkpoints").get(id)
      get.onsuccess = () => {
        const tx = db.transaction(["checkpoints", "checkpointMeta"], "readwrite")
        for (const o of others) {
          // Each as if it had lived some steps (the copied one may be the birth checkpoint, at step 0: a life with no
          // step and no log gets no final file).
          const rec = { ...get.result, learnerId: o.learnerId, savedAt: new Date(Date.now() - o.hoursAgo * 3_600_000).toISOString(), steps: Math.max(get.result.steps, 50) }
          tx.objectStore("checkpoints").put(rec)
          tx.objectStore("checkpointMeta").put({ learnerId: rec.learnerId, savedAt: rec.savedAt, version: rec.learnerState.version, steps: rec.steps, seq: rec.seq })
        }
        tx.oncomplete = () => { db.close(); res() }
        tx.onerror = () => rej(tx.error)
      }
    }
  }), { id, others })
  await page.close() // the tab closes mid-life: its learner is an orphan now

  const next = await context.newPage() // a new tab: no learnerId yet
  await setup(next, { knowledge: "ok" })
  await next.goto(LONG)
  await expect(next.getByTestId("learner-id")).toHaveText(id)
  await expect(next.getByTestId("life")).toContainText(`Adopted learner ${id}`)
  await expect(next.getByTestId("status")).toContainText("resumed from its checkpoint")
  expect(await next.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(id)
  await expect.poll(() => stepsLived(next)).toBeGreaterThan(0)
  await next.screenshot({ path: test.info().outputPath("page-adopted.png"), fullPage: true })
  // The newest three orphans stay (the adopted one among them); the older ones and the stale one are gone.
  const ids = await next.evaluate(() => new Promise<string[]>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const keys = req.result.transaction("checkpoints").objectStore("checkpoints").getAllKeys()
      keys.onsuccess = () => { req.result.close(); res(keys.result as string[]) }
    }
  }))
  expect(ids.sort()).toEqual([id, others[0]!.learnerId, others[1]!.learnerId].sort())
  // Each deleted one ends on the record (decision 0017): a final log file built from its checkpoint, in the outbox.
  const rows = await next.evaluate(() => new Promise<{ learnerId: string; seq: number; file: Record<string, unknown> }[]>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const all = req.result.transaction("outbox").objectStore("outbox").getAll()
      all.onsuccess = () => { req.result.close(); res(all.result) }
    }
  }))
  const deleted = others.slice(2).map(o => o.learnerId)
  const finals = rows.filter(r => deleted.includes(r.learnerId))
  expect(finals.map(r => r.learnerId).sort()).toEqual([...deleted].sort())
  for (const r of finals) {
    expect(r.seq).toBe(0)
    expect(r.file).toMatchObject({ schemaVersion: 2, learnerId: r.learnerId, seq: 0, final: true, condition: "A" })
    expect(r.file.stepsTotal as number).toBeGreaterThanOrEqual(50)
    // As the gate reads it: it parses and is accepted as the learner's first PR.
    const v = await validateSessionPr([{ path: `data/runs/${r.learnerId}/0.json`, content: JSON.stringify(r.file, null, 2) }], [], { spaces: spacesOf(GENOME_SPACE) })
    expect(v.errors).toEqual([])
  }
  expect(rows.filter(r => r.learnerId === others[0]!.learnerId || r.learnerId === others[1]!.learnerId)).toEqual([])
})

test("two tabs start at once, one reloading and one new: the reloading tab keeps its learner", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0) // its checkpoint is written: the only one, and the newest
  const fresh = await context.newPage()
  await setup(fresh, { knowledge: "ok" })
  // While the reload is under way its learner's lock is free: the new tab's sweep must not take it for an orphan.
  await Promise.all([page.reload(), fresh.goto(LONG)])
  await expect(page.getByTestId("learner-id")).toHaveText(id)
  await expect(page.getByTestId("status")).toContainText("resumed from its checkpoint")
  await expect(fresh.getByTestId("learner-id")).toHaveText(ID_RE)
  await expect(fresh.getByTestId("learner-id")).not.toHaveText(id)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  await expect.poll(() => stepsLived(fresh)).toBeGreaterThan(0)
})

test("a hidden page gets no frames; the learner lives on, and frames come back when the page is shown", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const setHidden = (h: boolean) => page.evaluate(hidden => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden })
    document.dispatchEvent(new Event("visibilitychange"))
  }, h)
  await setHidden(true)
  await page.waitForTimeout(300) // frames already in flight land
  const frozen = await stepsLived(page)
  await page.waitForTimeout(1_500) // ~30 steps at 20 steps/s
  expect(await stepsLived(page)).toBe(frozen)
  await setHidden(false)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(frozen + 10)
})

test("an older tab holding the database open: the page says to close other learner tabs", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  // An old page version (database v1, no versionchange handler) open in another tab of this origin.
  const old = await context.newPage()
  await old.route("**/knowledge/*.json", route => route.fulfill({ json: {} }))
  await old.goto("/learner/knowledge/manifest.json")
  await old.evaluate(() => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab", 1)
    req.onupgradeneeded = () => req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
    req.onsuccess = () => { (globalThis as { held?: unknown }).held = req.result; res() }
    req.onerror = () => rej(req.error)
  }))
  await page.goto(LONG)
  await expect(page.getByTestId("life")).toContainText("Close other learner tabs to update", { timeout: 20_000 })
})

test("without a speed knob the learner lives at most maxStepsPerSec = 30 steps a second, on the backend this browser offers", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto("/learner/?testWidths=tiny&testLifetimeCU=50") // no testBackend: the real WebGPU-or-CPU choice
  await expect(page.getByTestId("backend")).toHaveText(/^(WebGPU|CPU)/)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const t0 = Date.now(), s0 = await stepsLived(page)
  await page.waitForTimeout(5_000)
  const rate = (await stepsLived(page) - s0) / ((Date.now() - t0) / 1000)
  expect(rate).toBeGreaterThan(10)
  expect(rate).toBeLessThan(31)
})

const GH = `https://github.com/${UP}`

test("tabs: LEARNER by default; HISTORY lists past sessions newest first; ABOUT explains and links; links on every tab", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  const tab = (name: string) => page.getByRole("tab", { name })
  await expect(tab("Learner")).toHaveAttribute("aria-selected", "true")
  await expect(tab("History")).toHaveAttribute("aria-selected", "false")
  await expect(page.getByTestId("learner-id")).toBeVisible()
  await expect(page.getByTestId("history")).toBeHidden()
  await expect(page.getByTestId("build")).toHaveText(/^dev built \d{4}-\d\d-\d\d \d\d:\d\d UTC$/)
  await expect(page.getByTestId("build")).toHaveAttribute("href", GH)

  const footer = page.getByTestId("footer-links")
  const footerLinks = async () => footer.getByRole("link").evaluateAll(as => as.map(a => (a as HTMLAnchorElement).href))
  const expected = [GH, `${GH}/blob/main/docs/specs/2026-10-19-blanktab-v1-design.md`, `${GH}/tree/main/docs/decisions`,
    `${GH}/blob/main/docs/research/2026-10-16-research-synthesis-v0.1.md`, `${GH}/blob/main/docs/validate-plan-01.md`,
    `${GH}/pulls`, `${GH}/actions`, `${GH}/tree/main/data/runs`, "https://markwhiting.github.io/life/"]
  expect(await footerLinks()).toEqual(expected)

  await tab("History").click()
  await expect(tab("History")).toHaveAttribute("aria-selected", "true")
  await expect(tab("Learner")).toHaveAttribute("aria-selected", "false")
  await expect(page.getByTestId("learner-id")).toBeHidden()
  const rows = page.getByTestId("history").locator("tbody tr")
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText("lrn_bbbbbbbbbbbbbbbb")
  await expect(rows.nth(1)).toContainText("lrn_aaaaaaaaaaaaaaaa")
  await expect(rows.nth(0).getByRole("link")).toHaveAttribute("href", `${GH}/tree/main/data/runs/lrn_bbbbbbbbbbbbbbbb`)
  await expect(rows.nth(0)).toContainText("3000")
  await expect(rows.nth(0)).toContainText("moveProbability = 0.5")
  await expect(rows.nth(0)).toContainText("4 cores")
  await expect(rows.nth(0)).toContainText("0.912") // its own-stream ratio
  await expect(rows.nth(0)).toContainText("1.046") // its held-out ratio
  await expect(rows.nth(1)).toContainText("—") // a learner from before the held-out check
  await expect(rows.nth(0)).toContainText("1024.5") // its CU
  await expect(page.getByTestId("history").locator("thead")).toContainText(/held-out ratio/i)
  await expect(page.getByTestId("history").locator("thead")).toContainText(/own-stream ratio/i)
  await expect(page.getByTestId("history").locator("thead")).toContainText(/CU/)
  await expect(footer).toBeVisible()

  await tab("About").click()
  await expect(tab("About")).toHaveAttribute("aria-selected", "true")
  const about = page.getByTestId("about")
  await expect(about).toBeVisible()
  await expect(about).toContainText(/only pixels/i)
  await expect(about).toContainText(/api\.github\.com/)
  await expect(about).toContainText(/learns to predict what it sees/i)
  await expect(about).not.toContainText(/doesn't yet move between rooms by itself/i)
  await expect(about).toContainText(/caregiver/i)
  await expect(about).toContainText(/checkup/i)
  await expect(about).not.toContainText(/flails/i)
  expect(await about.getByRole("link").evaluateAll(as => as.map(a => (a as HTMLAnchorElement).href))).toEqual(expected)
  await expect(footer).toBeVisible()

  // The learner keeps living while another tab is shown.
  const before = await stepsLived(page)
  await tab("Learner").click()
  await expect(page.getByTestId("learner-id")).toBeVisible()
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(before)
})

test("growing up: a 'Has it grown up?' panel with no checkup yet, a caregiver line, and a Milestones column in HISTORY", async ({ page }) => {
  const errors: string[] = []
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()) })
  page.on("pageerror", e => errors.push(e.message))
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByRole("heading", { name: "Has it grown up?" })).toBeVisible()
  await expect(page.getByTestId("milestone-text")).toHaveText("no checkup yet (the first comes soon after it has lived 8 CU)")
  await expect(page.getByTestId("milestones")).toHaveAttribute("data-bars", "0")
  await expect(page.getByTestId("caregiver")).toHaveText(/^has (not carried it yet|carried it (once|\d+ times))/)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  await page.getByRole("tab", { name: "History" }).click()
  const head = page.getByTestId("history").locator("thead th")
  const names = await head.allTextContents()
  expect(names.indexOf("Milestones")).toBe(names.indexOf("Own-stream ratio") + 1)
  await expect(page.getByTestId("history").locator("tbody tr").nth(0)).toContainText("—")
  expect(errors).toEqual([])
})

test("charts redraw on resize: the curve and milestones canvases' width attributes follow a viewport change from 1280 to 700 px wide", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.goto(LONG)
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
  const widths = () => page.evaluate(() => ["curve", "milestones"].map(id => {
    const c = document.querySelector<HTMLCanvasElement>(`[data-testid="${id}"]`)!
    return { attr: c.width, css: c.clientWidth }
  }))
  const wide = await widths()
  expect(wide.map(w => w.attr === w.css && w.css > 0)).toEqual([true, true]) // a 1x display: one canvas pixel per CSS pixel
  // (the layout is one column below ~760 px, so the charts get wider, not narrower: either way the width attribute must follow)
  await page.setViewportSize({ width: 700, height: 800 })
  await expect.poll(async () => (await widths()).map(w => w.attr === w.css && w.css > 0 && w.css !== wide[0]!.css)).toEqual([true, true])
})

/** Standard laptop and desktop windows: the page is the window. */
const DESKTOPS = [{ width: 1280, height: 800 }, { width: 1440, height: 900 }, { width: 1536, height: 864 }, { width: 1920, height: 1080 }]

/**
 * Every tab fits the window: no page scroll either way, the panels fill it, and on LEARNER neither column scrolls inside.
 * `fontScale` makes every element's text that much larger (its computed font-size times the scale, set inline), standing
 * in for a platform whose fonts set taller than this one's (e.g. Linux CI): the layout must absorb it, not overflow.
 */
async function expectFits(page: Page, sizes: { width: number; height: number }[], fontScale = 1) {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
  if (fontScale !== 1) {
    await page.evaluate(scale => {
      const els = [...document.querySelectorAll<HTMLElement>("body, body *")]
      const px = els.map(e => parseFloat(getComputedStyle(e).fontSize)) // read all first: children inherit computed px
      els.forEach((e, i) => { e.style.fontSize = `${px[i]! * scale}px` })
    }, fontScale)
  }
  const fit = () => page.evaluate(() => {
    const se = document.scrollingElement!
    const panel = document.querySelector<HTMLElement>('[role="tabpanel"]:not([hidden])')!
    const boxes = [...panel.querySelectorAll<HTMLElement>(".stage, .side, .table-wrap, .about")].map(e => e.getBoundingClientRect())
    const inner = [...panel.querySelectorAll<HTMLElement>(".stage, .side")].map(e => e.scrollHeight - e.clientHeight)
    return {
      vScroll: se.scrollHeight - innerHeight, hScroll: se.scrollWidth - innerWidth,
      widthShare: (Math.max(...boxes.map(b => b.right)) - Math.min(...boxes.map(b => b.left))) / innerWidth,
      bottomGap: innerHeight - Math.max(...boxes.map(b => b.bottom)), innerOverflow: Math.max(0, ...inner),
    }
  })
  for (const size of sizes) {
    await page.setViewportSize(size)
    for (const name of ["Learner", "History", "About"]) {
      await page.getByRole("tab", { name }).click()
      const f = await fit()
      const at = `${name} at ${size.width}x${size.height}${fontScale !== 1 ? ` with text x${fontScale}` : ""}`
      expect(f.vScroll, at).toBeLessThanOrEqual(0)
      expect(f.hScroll, at).toBeLessThanOrEqual(0)
      expect(f.widthShare, at).toBeGreaterThanOrEqual(name === "About" ? 0.5 : 0.85)
      if (name === "Learner") {
        expect(f.bottomGap, at).toBeLessThanOrEqual(44) // only the footer's links under the panels
        expect(f.innerOverflow, at).toBe(0) // and neither column scrolls inside either
        // The recipe's "settings" toggle, opened: its box scrolls on its own; the chart keeps room and nothing overflows.
        await page.getByTestId("genome").locator("summary").click()
        await expect(page.getByTestId("settings-box")).toBeVisible()
        const open = await fit()
        const curveHeight = await page.getByTestId("curve").evaluate(c => c.clientHeight)
        expect(open.vScroll, `${at}, settings open`).toBeLessThanOrEqual(0)
        expect(open.innerOverflow, `${at}, settings open`).toBe(0)
        expect(curveHeight, `${at}, settings open: the curve's height`).toBeGreaterThanOrEqual(40)
        await page.getByTestId("genome").locator("summary").click()
        await expect(page.getByTestId("settings-box")).toBeHidden()
      }
    }
    await page.getByRole("tab", { name: "Learner" }).click()
    // The learner's views fill their column at their own aspect ratio, scaled up nearest-neighbour.
    const views = await page.evaluate(() => ["#periphery", "#fovea"].map(s => {
      const c = document.querySelector<HTMLCanvasElement>(s)!
      return { w: c.clientWidth, h: c.clientHeight, ratio: c.width / c.height, rendering: getComputedStyle(c).imageRendering }
    }))
    expect(views[0]!.w).toBeGreaterThan(size.width * 0.4)
    for (const v of views) {
      expect(Math.abs(v.w / v.h - v.ratio)).toBeLessThan(0.02)
      expect(v.rendering).toBe("pixelated")
    }
  }
}

test("on a laptop or desktop window each tab fits it: no page scroll either way, and the panels fill it", async ({ page }) => {
  await expectFits(page, DESKTOPS)
})

test("the fit absorbs taller text (fonts that set 8% larger, as on another platform): still no scroll, nothing overflows", async ({ page }) => {
  await expectFits(page, [{ width: 1280, height: 800 }, { width: 1440, height: 900 }], 1.08)
})

test("on a laptop-height window the right column's prose is clamped, and its \"more\" opens it in full", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  for (const id of ["curve-help", "short-lives-help"]) {
    const clipped = () => page.locator(`#${id}`).evaluate(e => e.scrollHeight - e.clientHeight)
    const more = page.locator(`button[aria-controls="${id}"]`)
    expect(await clipped(), id).toBeGreaterThan(0)
    await more.click()
    await expect(more).toHaveAttribute("aria-expanded", "true")
    expect(await clipped(), id).toBe(0)
    await more.click()
    await expect(more).toHaveText("more")
    expect(await clipped(), id).toBeGreaterThan(0)
  }
})

test("on a phone the LEARNER tab is one scrolling column", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const ids = ["#periphery", "#fovea", '[data-testid="learner-id"]', '[data-testid="curve"]', '[data-testid="milestones"]']
  const boxes = await page.evaluate(sel => sel.map(s => document.querySelector(s)!.getBoundingClientRect()).map(b => ({ left: b.left, right: b.right, top: b.top, bottom: b.bottom })), ids)
  for (let i = 1; i < boxes.length; i++) expect(boxes[i]!.top, `${ids[i]} under ${ids[i - 1]}`).toBeGreaterThanOrEqual(boxes[i - 1]!.bottom)
  for (const b of boxes) { expect(b.left).toBeGreaterThanOrEqual(0); expect(b.right).toBeLessThanOrEqual(390) }
  expect(await page.evaluate(() => document.scrollingElement!.scrollWidth)).toBeLessThanOrEqual(390)
  expect(await page.evaluate(() => document.scrollingElement!.scrollHeight)).toBeGreaterThan(844) // the page scrolls
})

test.describe("on a 2x display", () => {
  test.use({ deviceScaleFactor: 2 })
  test("charts drawn while LEARNER is hidden keep their size, and are drawn again when it is shown", async ({ page }) => {
    await setup(page, { knowledge: "ok" })
    await page.goto(LONG)
    await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
    const sizes = () => page.evaluate(() => ["curve", "milestones"].map(id => {
      const c = document.querySelector<HTMLCanvasElement>(`[data-testid="${id}"]`)!
      return [c.width, c.height]
    }))
    // The charts flex with the text around them (e.g. the ratio line), so wait until they are drawn at their laid-out size.
    const settled = () => page.evaluate(() => ["curve", "milestones"].every(id => {
      const c = document.querySelector<HTMLCanvasElement>(`[data-testid="${id}"]`)!
      return c.clientWidth > 0 && c.width === Math.round(c.clientWidth * devicePixelRatio) && c.height === Math.round(c.clientHeight * devicePixelRatio)
    }))
    await expect.poll(settled).toBe(true)
    const shown = await sizes()
    await page.getByRole("tab", { name: "History" }).click()
    // A curve point arrives at every sleep (every 5 steps): several draws while the canvases are hidden.
    const points = await curvePoints(page)
    await expect.poll(() => curvePoints(page), { timeout: 30_000 }).toBeGreaterThan(points + 3)
    expect(await sizes()).toEqual(shown)
    await page.getByRole("tab", { name: "Learner" }).click()
    expect(await sizes()).toEqual(shown)
    await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
  })
})

test("HISTORY says so when there are no past sessions", async ({ page }) => {
  await setup(page, { knowledge: "unreachable" })
  await page.goto(LONG)
  await page.getByRole("tab", { name: "History" }).click()
  await expect(page.getByTestId("history")).toContainText("No past sessions yet")
  await expect(page.getByTestId("history").locator("tbody tr")).toHaveCount(0)
})

test("a malformed summary row is dropped: the learner runs and HISTORY shows only the good rows, newest first", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.route("**/knowledge/summary.json", route => route.fulfill({ json: { schemaVersion: 1, sessions: [
    { ...row("lrn_aaaaaaaaaaaaaaaa"), lastWrittenAt: "2026-10-19T12:00:00.000Z" },
    { ...row("lrn_dddddddddddddddd"), hostSig: null },
    { learnerId: "garbage" },
    { ...row("lrn_bbbbbbbbbbbbbbbb"), lastWrittenAt: "2026-10-20T12:00:00.000Z" },
  ] } }))
  await page.goto(LONG)
  await expect(page.getByText("2 sessions before this one")).toBeVisible()
  await expect(page.getByTestId("knowledge")).not.toHaveText(/offline/i)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  await page.getByRole("tab", { name: "History" }).click()
  const rows = page.getByTestId("history").locator("tbody tr")
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText("lrn_bbbbbbbbbbbbbbbb")
  await expect(rows.nth(1)).toContainText("lrn_aaaaaaaaaaaaaaaa")
})
