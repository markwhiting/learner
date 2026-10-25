// Plan 05a Task 12: the page's birth draw, its recipe and meta-learner lines, the short-lives switch and newborn mode.
// Plan 05b Task 11: ts and confirm births under a search posterior, the confirmation newborn, and the mode guard.
import { test, expect, type Page } from "@playwright/test"
import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"
import { DEFAULT_SETTINGS } from "@blanktab/learner"
import { birthDraw, parsePosterior } from "@blanktab/meta"
import { NEWBORN_SEED_BASE } from "@blanktab/harness/newborn"
import { NEWBORN_LEVELS, parseLogFile } from "@blanktab/harness/log"
import { spacesOf, validateSessionPr } from "../../tools/gate/validate"
import { CENTRE, CENTRE_SHA, GENOME_JSON, SPACE, SPACE_JSON, V2_SEARCH_TEXT, posteriorText, v2SearchText } from "../../app/test/posteriors"
import { fmt } from "../../app/src/site"
import { recipeText } from "../../app/src/recipe"
import type { MetaBlock } from "@blanktab/harness/log"

const ID_RE = /lrn_[0-9a-f]{16}/
const sha = (s: string) => createHash("sha256").update(s).digest("hex")
type File = Record<string, unknown> & { seq: number; final: boolean; cu: number; settings: Record<string, number>; meta?: Record<string, unknown> & { genome: Record<string, number | string> }; checkups?: { level: number }[] }

/** Serves knowledge/*.json: the manifest and summary as the loop does, the genome files and posterior as given (strings as raw bytes; undefined: 404). */
async function serve(page: Page, files: Record<string, unknown>) {
  const all: Record<string, unknown> = {
    "summary.json": { schemaVersion: 1, sessions: [] },
    "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev", learnerStateVersion: LEARNER_STATE_VERSION },
    "genome-space.json": SPACE_JSON, "genome.json": GENOME_JSON, ...files,
  }
  await page.route("**/knowledge/*.json", route => {
    const body = all[new URL(route.request().url()).pathname.split("/").pop()!]
    if (body === undefined) return route.fulfill({ status: 404 })
    return typeof body === "string" ? route.fulfill({ body, contentType: "application/json" }) : route.fulfill({ json: body })
  })
  await page.route("https://api.github.com/**", route => route.abort())
  const errors: string[] = []
  page.on("console", m => { if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) errors.push(m.text()) })
  page.on("pageerror", e => errors.push(e.message))
  return { errors, files: all }
}

/** The outbox's files, oldest first (of learner `id`, or every learner's). */
const outbox = (page: Page, id?: string) => page.evaluate(i => new Promise<{ learnerId: string; file: unknown }[]>((res, rej) => {
  const req = indexedDB.open("blanktab")
  req.onerror = () => rej(req.error)
  req.onsuccess = () => {
    const all = req.result.transaction("outbox").objectStore("outbox").getAll()
    all.onsuccess = () => {
      req.result.close()
      const rows = all.result as { learnerId: string; seq: number; file: unknown }[]
      res(rows.filter(r => i === undefined || r.learnerId === i).sort((a, b) => a.seq - b.seq).map(r => ({ learnerId: r.learnerId, file: r.file })))
    }
    all.onerror = () => rej(all.error)
  }
}), id)
const filesOf = async (page: Page, id: string) => (await outbox(page, id)).map(r => r.file as File)
const learnerIdOf = async (page: Page) => {
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  return (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
}
const cuLived = async (page: Page) => Number((await page.getByTestId("cu").textContent())?.match(/^([\d.]+) (?:of |CU)/)?.[1] ?? NaN)
/** The full genome as the page's toggle lists it. */
const fmtGenome = (g: Record<string, number | string>) => Object.entries(g).map(([k, v]) => `${k} = ${fmt(v)}`).join(", ")

test("a calibrate posterior: the page shows 'a calibration life' and 'Calibrating: …'; the outbox's first file has meta.method calibrate and the genome the page shows", async ({ page }) => {
  const text = posteriorText("calibrate", { calibrationLives: 18, calibrationTarget: 30 })
  const { errors } = await serve(page, { "posterior.json": text })
  // The GPU widths on the CPU: a "gpu" widths class draw (calibration lives are GPU-width only), in a life of a step or
  // two and no sleep (a sleep at these widths on the CPU, replaying up to 256 samples, can take minutes).
  await page.goto("/learner/?testBackend=cpu&testWidths=gpu&testLifetimeCU=0.05&testSleepInterval=1000")
  const id = await learnerIdOf(page)
  await expect(page.getByTestId("meta-state")).toHaveText("Calibrating: 18 of 30 lives measured, to learn whether the score can tell recipes apart.")
  await expect(page.getByTestId("recipe-text")).toHaveText(/^a calibration life \(recipe \d+ of 10\)$/, { timeout: 30_000 })
  const shown = await page.getByTestId("genome-list").textContent()
  const shas = await page.getByTestId("genome-shas").textContent()
  await expect.poll(async () => (await filesOf(page, id)).length, { timeout: 60_000 }).toBeGreaterThan(0)
  const first = (await filesOf(page, id))[0]!
  expect(first.meta).toMatchObject({ role: "search", method: "calibrate", posteriorSha: sha(text) })
  expect(fmtGenome(first.meta!.genome)).toBe(shown)
  expect(shas).toBe(`genomeSha ${(first.meta!.genomeSha as string).slice(0, 8)} · posteriorSha ${sha(text).slice(0, 8)}`)
  const recipe = Number((await page.getByTestId("recipe-text").textContent())!.match(/recipe (\d+)/)![1])
  expect(parsePosterior(text)!.calibration.genomes[recipe - 1]!.genomeSha).toBe(first.meta!.genomeSha)
  expect(errors).toEqual([])
})

test("the draw in the page equals birthDraw in Node for the same learnerId and posterior bytes", async ({ context }) => {
  // Paused: CPU-class tabs draw centre or uniform (1 in 5). The bytes carry odd spacing: the sha is of the bytes.
  const text = posteriorText("paused", { calibrationLives: 34, calibrationTarget: 45, looks: 1 }).replace("{\n", "{  \n")
  const posterior = parsePosterior(text)!
  // learnerIds of both kinds, pinned through sessionStorage (this tab's learner), so the uniform branch is checked too.
  const ids = Array.from({ length: 400 }, (_, i) => `lrn_${(0xabc000 + i).toString(16).padStart(16, "0")}`)
  const drawOf = (learnerId: string) => birthDraw({ learnerId, space: SPACE, centre: CENTRE, centreSha: CENTRE_SHA, widthsClass: "cpu", posterior, posteriorSha: sha(text) })
  const picks = [ids.find(i => drawOf(i).method === "uniform")!, ids.find(i => drawOf(i).method === "centre")!]
  for (const learnerId of picks) {
    const page = await context.newPage()
    const { errors } = await serve(page, { "posterior.json": text })
    await page.addInitScript(i => { if (!sessionStorage.getItem("blanktab:learnerId")) sessionStorage.setItem("blanktab:learnerId", i) }, learnerId)
    await page.goto("/learner/?testBackend=cpu&testWidths=tiny&testLifetimeCU=0.05&testSleepInterval=5")
    await expect(page.getByTestId("learner-id")).toHaveText(learnerId)
    await expect(page.getByTestId("meta-state")).toHaveText("Paused: the score can't yet tell recipes apart; measuring more (34 of 45).")
    await expect.poll(async () => (await filesOf(page, learnerId)).length, { timeout: 30_000 }).toBeGreaterThan(0)
    const meta = (await filesOf(page, learnerId))[0]!.meta!
    const d = drawOf(learnerId)
    expect({ method: meta.method, drawSeed: meta.drawSeed, posteriorSha: meta.posteriorSha, evalPanel: meta.evalPanel, genome: meta.genome })
      .toEqual({ method: d.method, drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, genome: d.genome })
    await expect(page.getByTestId("recipe-text")).toHaveText(d.method === "uniform" ? "one of the 1-in-5 random explorers" : "the current recipe")
    if (d.method === "uniform") await expect(page.getByTestId("recipe-genes")).toHaveText(/^ · [a-z ]+ ×[\d.]+, [a-z ]+ ×[\d.]+, [a-z ]+ ×[\d.]+$/)
    expect(errors).toEqual([])
    await page.close()
  }
})

test("posterior.json 404 → 'the current recipe', posteriorSha null; genome-space.json garbage → no meta, DEFAULT_SETTINGS", async ({ page, context }) => {
  const { errors } = await serve(page, {})
  await page.goto("/learner/?testBackend=cpu&testWidths=tiny&testLifetimeCU=0.05&testSleepInterval=5")
  const id = await learnerIdOf(page)
  await expect(page.getByTestId("recipe-text")).toHaveText("the current recipe")
  await expect(page.getByTestId("meta-state")).toHaveText("Not reachable: this learner runs the current recipe.")
  await expect(page.getByTestId("genome-shas")).toHaveText(`genomeSha ${CENTRE_SHA.slice(0, 8)} · posteriorSha none`)
  await expect.poll(async () => (await filesOf(page, id)).length, { timeout: 30_000 }).toBeGreaterThan(0)
  expect((await filesOf(page, id))[0]!.meta).toMatchObject({ method: "centre", posteriorSha: null, drawSeed: null, genome: CENTRE })
  expect(errors).toEqual([])

  const other = await context.newPage()
  const o = await serve(other, { "genome-space.json": "<html>garbage", "posterior.json": posteriorText("calibrate") })
  await other.goto("/learner/?testBackend=cpu&testWidths=tiny&testLifetimeCU=0.05&testSleepInterval=5")
  const id2 = await learnerIdOf(other)
  await expect(other.getByTestId("recipe-text")).toHaveText(/^the default settings, not counted/)
  await expect(other.getByTestId("meta-state")).toHaveText("Not reachable: the genome files could not be read, so this learner runs the default settings and is not counted.")
  await expect(other.getByTestId("knowledge")).toHaveText(/^loaded/) // never offline for it
  await expect.poll(async () => (await filesOf(other, id2)).length, { timeout: 30_000 }).toBeGreaterThan(0)
  const f = (await filesOf(other, id2))[0]!
  expect(f.meta).toBeUndefined()
  expect(f.settings).toMatchObject({ ...DEFAULT_SETTINGS })
  expect(o.errors).toEqual([])
})

test("a life born without a block (before 05a) resumes as legacy: the recipe says so and names the genes its settings did not record", async ({ page }) => {
  const { errors, files } = await serve(page, { "genome-space.json": undefined, "genome.json": undefined })
  await page.goto("/learner/?testBackend=cpu&testWidths=tiny&testLifetimeCU=2&testSleepInterval=5")
  const id = await learnerIdOf(page)
  await expect(page.getByTestId("recipe-text")).toHaveText("the default settings, not counted (no genome files)")
  await expect.poll(async () => Number(await page.getByTestId("steps").textContent()), { timeout: 30_000 }).toBeGreaterThan(20)
  // The genome files are published; the reload resumes the same life from its checkpoint, which has no block.
  files["genome-space.json"] = SPACE_JSON
  files["genome.json"] = GENOME_JSON
  await page.reload()
  await expect(page.getByTestId("learner-id")).toHaveText(id)
  await expect(page.getByTestId("status")).toContainText("resumed from its checkpoint")
  await expect(page.getByTestId("recipe-text")).toHaveText("born before the meta-learner (its old settings)")
  await expect(page.getByTestId("recipe-genes")).toHaveText(" · some settings not recorded: sleepInterval")
  await expect.poll(async () => (await filesOf(page, id)).some(f => f.final), { timeout: 60_000 }).toBe(true)
  const files_ = await filesOf(page, id)
  const last = files_[files_.length - 1]!
  expect(last.meta).toMatchObject({ role: "legacy", method: "legacy", evalPanel: null, missing: ["sleepInterval"] })
  const v = await validateSessionPr(files_.map(f => ({ path: `data/runs/${id}/${f.seq}.json`, content: JSON.stringify(f, null, 2) })), [], { spaces: spacesOf(SPACE) })
  expect(v.errors).toEqual([])
  expect(errors).toEqual([])
})

test("'Score' appears once the summary row for this learner has its pinned evaluator's score", async ({ page }) => {
  const id = "lrn_5c0e5c0e5c0e5c0e"
  const row = (evals?: unknown) => ({
    learnerId: id, settings: {}, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, stepsTotal: 3000, final: false,
    codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0, ...(evals ? { evals } : {}),
  })
  const { errors, files } = await serve(page, { "summary.json": { schemaVersion: 1, sessions: [row()] } })
  await page.addInitScript(i => { if (!sessionStorage.getItem("blanktab:learnerId")) sessionStorage.setItem("blanktab:learnerId", i) }, id)
  await page.goto("/learner/?testBackend=cpu&testWidths=tiny&testLifetimeCU=50&testSleepInterval=5")
  await expect(page.getByTestId("learner-id")).toHaveText(id)
  await expect(page.getByTestId("recipe-text")).toHaveText("the current recipe")
  await expect(page.getByTestId("score")).toBeHidden()
  files["summary.json"] = { schemaVersion: 1, sessions: [row({ "search-composite@1": { eligible: true, score: 0.4123, C: 0.5, H: 0.3 } })] }
  await page.reload()
  await expect(page.getByTestId("score")).toHaveText("0.41 (search-composite@1)")
  await expect(page.getByTestId("score-label")).toBeVisible()
  expect(errors).toEqual([])
})

test("the short-lives switch: off by default; checking it stores blanktab:shortLives = 1, survives a reload, and posts shortLives to the worker", async ({ page }) => {
  const { errors } = await serve(page, {})
  // Every message the page posts to its worker, by type (and the switch's value).
  await page.addInitScript(() => {
    const sent: unknown[] = ((globalThis as { sent?: unknown[] }).sent = [])
    const post = Worker.prototype.postMessage
    Worker.prototype.postMessage = function (this: Worker, m: { type: string; on?: boolean; shortLives?: boolean }, ...rest: unknown[]) {
      sent.push(m.type === "start" ? { type: "start", shortLives: m.shortLives } : m.type === "shortLives" ? { type: "shortLives", on: m.on } : { type: m.type })
      return (post as (...a: unknown[]) => void).call(this, m, ...rest)
    } as Worker["postMessage"]
  })
  const sent = () => page.evaluate(() => ((globalThis as { sent?: { type: string }[] }).sent ?? []).filter(m => m.type === "start" || m.type === "shortLives"))
  await page.goto("/learner/?testBackend=cpu&testWidths=tiny&testLifetimeCU=50&testSleepInterval=5")
  const box = page.getByTestId("short-lives")
  await expect(page.getByText("Short lives")).toBeVisible()
  await expect(page.getByText("End each life at its 2,048 CU checkup and start a newborn (about every 30 minutes). The meta-learner only gets results at a useful rate from tabs with this on.")).toBeVisible()
  await expect(box).not.toBeChecked()
  expect(await page.evaluate(() => localStorage.getItem("blanktab:shortLives"))).toBeNull()
  await expect.poll(sent).toEqual([{ type: "start", shortLives: false }])
  await box.check()
  expect(await page.evaluate(() => localStorage.getItem("blanktab:shortLives"))).toBe("1")
  await expect.poll(sent).toEqual([{ type: "start", shortLives: false }, { type: "shortLives", on: true }])
  await page.reload()
  await expect(box).toBeChecked()
  await expect.poll(sent).toEqual([{ type: "start", shortLives: true }])
  await box.uncheck()
  expect(await page.evaluate(() => localStorage.getItem("blanktab:shortLives"))).toBeNull()
  await expect.poll(sent).toEqual([{ type: "start", shortLives: true }, { type: "shortLives", on: false }])
  expect(errors).toEqual([])
})

test("with a test budget of 8 CU: switching short lives on past it asks first; once on, the life ends at a clean sleep with meta.end budget and a newborn follows with its own draw", async ({ page }) => {
  test.setTimeout(240_000) // 8 CU and its checkup take about a minute on the CPU
  // A genome that runs as it says (a speed knob would contradict the genome, and such a life has no block).
  const { errors } = await serve(page, fastGenomeFiles())
  await page.goto("/learner/?testBackend=cpu&testSleepInterval=50&testBudgetCU=8")
  const id = await learnerIdOf(page)
  await expect(page.getByTestId("recipe-text")).toHaveText("the current recipe")
  // Its budget checkup (8 CU) is logged, and with the switch off it lives on past it.
  await expect.poll(async () => (await filesOf(page, id)).some(f => (f.checkups ?? []).some(c => c.level === 8)), { timeout: 90_000 }).toBe(true)
  await expect.poll(() => cuLived(page), { timeout: 30_000 }).toBeGreaterThan(9)
  expect((await filesOf(page, id)).some(f => f.final)).toBe(false)

  const box = page.getByTestId("short-lives")
  const asked: string[] = []
  page.once("dialog", d => { asked.push(d.message()); void d.dismiss() })
  await box.click()
  await expect(box).not.toBeChecked() // dismissed: nothing changes
  expect(await page.evaluate(() => localStorage.getItem("blanktab:shortLives"))).toBeNull()
  expect(asked).toEqual(["This learner has had its 8 CU checkup, so it will end at its next sleep and a newborn will start. Go on?"])
  page.once("dialog", d => { asked.push(d.message()); void d.accept() })
  await box.click()
  await expect(box).toBeChecked()
  expect(asked).toHaveLength(2)
  await expect(page.getByTestId("life")).toContainText(`This learner's life is complete (${id}): short lives are on`, { timeout: 30_000 })
  const files = await filesOf(page, id)
  const last = files[files.length - 1]!
  expect(last.final).toBe(true)
  expect(last.meta).toMatchObject({ end: "budget", budget: { budgetCU: 8, levels: [8] } })
  expect(files.filter(f => f.final)).toHaveLength(1)
  // The newborn: another learner, its own draw (its learnerId's panel), short lives still on.
  await expect(page.getByTestId("learner-id")).not.toHaveText(id, { timeout: 10_000 })
  const next = await learnerIdOf(page)
  await expect(page.getByTestId("recipe-text")).toHaveText("the current recipe")
  await expect.poll(async () => (await filesOf(page, next)).length, { timeout: 60_000 }).toBeGreaterThan(0)
  const born = (await filesOf(page, next))[0]!.meta!
  expect(born.evalPanel).toBe(birthDraw({ learnerId: next, space: SPACE, centre: CENTRE, centreSha: CENTRE_SHA, widthsClass: "cpu", posterior: null, posteriorSha: null }).evalPanel)
  await expect(box).toBeChecked()
  expect(errors).toEqual([])
})

/** A genome space whose speed cap is 300 steps a second (a fixed gene), so a half-width CPU life lives 8 CU in about a minute. */
function fastGenomeFiles() {
  const space = JSON.parse(JSON.stringify(SPACE_JSON)) as { genes: { name: string; min?: number; max?: number; centre: number | string }[] }
  for (const g of space.genes) if (g.name === "maxStepsPerSec") Object.assign(g, { min: 300, max: 300, centre: 300 })
  const genome = JSON.parse(JSON.stringify(GENOME_JSON)) as { values: Record<string, number> }
  genome.values.maxStepsPerSec = 300
  return { "genome-space.json": space, "genome.json": genome }
}

/** Leaves only learner `id`'s newest outbox file, without its checkups: as if the files that held them were sent and pruned. */
const pruneOutbox = (page: Page, id: string) => page.evaluate(i => new Promise<void>((res, rej) => {
  const req = indexedDB.open("blanktab")
  req.onerror = () => rej(req.error)
  req.onsuccess = () => {
    const tx = req.result.transaction("outbox", "readwrite")
    const store = tx.objectStore("outbox")
    const all = store.getAll()
    all.onsuccess = () => {
      const rows = (all.result as { learnerId: string; seq: number; file: Record<string, unknown> }[]).filter(r => r.learnerId === i).sort((a, b) => a.seq - b.seq)
      const newest = rows.pop()!
      for (const r of rows) store.delete([r.learnerId, r.seq])
      store.put({ ...newest, file: { ...newest.file, checkups: [] } })
    }
    tx.oncomplete = () => { req.result.close(); res() }
    tx.onerror = () => rej(tx.error)
  }
}), id)

test("final review: a life resumed past its budget with the switch stored on ends only after a confirm for that learner, stored across reloads; a pruned outbox still asks", async ({ page }) => {
  test.setTimeout(480_000) // the 8 CU checkup, then the 16 CU one (its learning paused for a while) on the CPU
  const { errors } = await serve(page, fastGenomeFiles())
  const asked: string[] = []
  let answer: "accept" | "dismiss" = "dismiss"
  page.on("dialog", d => { asked.push(d.message()); void (answer === "accept" ? d.accept() : d.dismiss()) })
  const question = "This learner has had its 8 CU checkup, so it will end at its next sleep and a newborn will start. Go on?"
  const steps = async () => Number(await page.getByTestId("steps").textContent())
  /** More than 110 more steps (at least two clean sleeps at 50 steps each), and the life is still this one, not final. */
  const livesOn = async (id: string) => {
    const from = await steps()
    await expect.poll(steps, { timeout: 120_000 }).toBeGreaterThan(from + 110) // a checkup may pause it meanwhile
    await expect(page.getByTestId("learner-id")).toHaveText(id)
    expect((await filesOf(page, id)).some(f => f.final)).toBe(false)
  }
  const resumed = async (id: string) => {
    await expect(page.getByTestId("learner-id")).toHaveText(id)
    await expect(page.getByTestId("status")).toContainText("resumed from its checkpoint", { timeout: 30_000 })
  }
  await page.goto("/learner/?testBackend=cpu&testSleepInterval=50&testBudgetCU=8")
  const id = await learnerIdOf(page)
  await expect.poll(async () => (await filesOf(page, id)).some(f => (f.checkups ?? []).some(c => c.level === 8)), { timeout: 90_000 }).toBe(true)
  await expect.poll(() => cuLived(page), { timeout: 30_000 }).toBeGreaterThan(9)

  // The switch is stored on (another tab, or a code-update reload): the resumed life is asked about once; no ends nothing.
  await page.evaluate(() => localStorage.setItem("blanktab:shortLives", "1"))
  await page.reload()
  await resumed(id)
  await expect.poll(() => asked.length).toBe(1)
  expect(asked).toEqual([question])
  expect(await page.evaluate(i => localStorage.getItem(`blanktab:shortLivesConsent:${i}`), id)).toBe("0")
  await expect(page.getByTestId("short-lives")).toBeChecked()
  await livesOn(id)
  // The answer survives a reload: not asked again, and still not ended.
  await page.reload()
  await resumed(id)
  await livesOn(id)
  expect(asked).toHaveLength(1)

  // The outbox pruned (the files with its checkups gone) and the switch off: switching it on still asks (the session's
  // own last level, not the outbox's), and a yes ends it at its next clean sleep.
  await page.evaluate(() => localStorage.removeItem("blanktab:shortLives"))
  await pruneOutbox(page, id)
  await page.reload()
  await resumed(id)
  await livesOn(id)
  expect((await filesOf(page, id)).flatMap(f => f.checkups ?? []).filter(c => c.level >= 8)).toEqual([])
  answer = "accept"
  await page.getByTestId("short-lives").click()
  await expect.poll(() => asked.length).toBe(2)
  expect(asked[1]).toBe(question)
  await expect(page.getByTestId("life")).toContainText(`This learner's life is complete (${id}): short lives are on`, { timeout: 30_000 })
  const last = (await filesOf(page, id)).at(-1)!
  expect(last.final).toBe(true)
  expect(last.meta).toMatchObject({ end: "budget" })
  expect(await page.evaluate(i => localStorage.getItem(`blanktab:shortLivesConsent:${i}`), id)).toBeNull()
  expect(errors).toEqual([])
})

test("?newborns=0-0 writes one newborn file that parses and validates", async ({ page }) => {
  const { errors } = await serve(page, { "posterior.json": posteriorText("calibrate") })
  await page.addInitScript(() => localStorage.setItem("blanktab:githubToken", "test-token")) // even with a token, nothing is sent
  const github: string[] = []
  page.on("request", r => { if (r.url().startsWith("https://api.github.com")) github.push(r.url()) })
  await page.goto("/learner/?newborns=0-0&testBackend=cpu&testWidths=tiny")
  await expect(page.getByTestId("life")).toContainText("Newborn reference 0 of 0–0")
  // No draw in this mode: the meta-learner line says so, and the switch (which ends lives) is not offered.
  await expect(page.getByTestId("meta-state")).toHaveText(/^Newborn reference run: no draw here\./)
  await expect(page.getByTestId("short-lives")).toBeHidden()
  await expect(page.getByTestId("life")).toContainText("Newborn references 0–0 are done", { timeout: 80_000 })
  await expect(page.getByTestId("recipe-text")).toHaveText("a newborn reference (panel 1)") // its file's panel
  const rows = await outbox(page)
  expect(rows).toHaveLength(1)
  const { learnerId, file } = rows[0]!
  const f = file as File
  expect(parseLogFile(JSON.parse(JSON.stringify(f))).ok).toBe(true)
  expect(f).toMatchObject({ seq: 0, final: true, cu: 0, stepsTotal: 0, settings: { seed: NEWBORN_SEED_BASE } })
  expect(f.meta).toMatchObject({ role: "newborn", method: "newborn", evalPanel: 1, genomeSha: CENTRE_SHA, genome: CENTRE })
  const v = await validateSessionPr([{ path: `data/runs/${learnerId}/0.json`, content: JSON.stringify(f, null, 2) }], [], { spaces: spacesOf(SPACE) })
  expect(v.errors).toEqual([])
  // Kept in this browser for the collector: marked local-only, never sent.
  const local = await page.evaluate(() => new Promise<unknown[]>(res => {
    const req = indexedDB.open("blanktab")
    req.onsuccess = () => { const g = req.result.transaction("contribution").objectStore("contribution").getAll(); g.onsuccess = () => res(g.result) }
  }))
  expect(local).toEqual([{ learnerId, lastPr: 0, closedInARow: 0, localOnly: true }])
  expect(github.filter(u => !u.endsWith("/user"))).toEqual([])
  expect(errors).toEqual([])
})

// ---- Plan 05b, Task 11. The stubbed posterior.json is the v2 search fixture (a real fit, confirmation 3 in flight), served
// by a Playwright route: the only place a tab meets a search posterior before the signal check answers (ruling 1: data
// only, no code switch). GPU widths on the CPU backend: a "gpu" widths class, which alone draws ts and confirm.
const GPU_CLASS = "/learner/?testBackend=cpu&testWidths=gpu&testLifetimeCU=0.05&testSleepInterval=1000"
const gpuDraw = (learnerId: string, text: string) =>
  birthDraw({ learnerId, space: SPACE, centre: CENTRE, centreSha: CENTRE_SHA, widthsClass: "gpu", posterior: parsePosterior(text), posteriorSha: sha(text) })
const POOL = Array.from({ length: 400 }, (_, i) => `lrn_${(0x5eac00 + i).toString(16).padStart(16, "0")}`)
/** This tab's learner is `learnerId` (its sessionStorage), so its draw is known in Node. */
const pin = (page: Page, learnerId: string) =>
  page.addInitScript(i => { if (!sessionStorage.getItem("blanktab:learnerId")) sessionStorage.setItem("blanktab:learnerId", i) }, learnerId)

test("a search posterior: the page's first file has method ts or uniform, and its genome equals birthDraw in Node for the same learnerId and bytes", async ({ context }) => {
  test.setTimeout(180_000)
  // A ts life, and (the same fixture) a confirmation life of the candidate arm, seeded by its evalSeed.
  const picks = [POOL.find(i => gpuDraw(i, V2_SEARCH_TEXT).method === "ts")!, POOL.find(i => gpuDraw(i, V2_SEARCH_TEXT).confirmation?.arm === "candidate")!]
  for (const learnerId of picks) {
    const page = await context.newPage()
    const { errors } = await serve(page, { "posterior.json": V2_SEARCH_TEXT })
    await pin(page, learnerId)
    await page.goto(GPU_CLASS)
    await expect(page.getByTestId("learner-id")).toHaveText(learnerId)
    const d = gpuDraw(learnerId, V2_SEARCH_TEXT)
    await expect(page.getByTestId("meta-state")).toHaveText(/^Confirming recipe [0-9a-f]{8} on fresh worlds: 70 of 108 lives\.$/)
    await expect(page.getByTestId("recipe-text")).toHaveText(d.method === "ts"
      ? "drawn by the meta-learner (a Thompson sample from 40 lives)"
      : `a confirmation life (candidate arm, fresh world ${d.evalSeed!.toLocaleString("en-US")})`, { timeout: 30_000 })
    await expect.poll(async () => (await filesOf(page, learnerId)).length, { timeout: 60_000 }).toBeGreaterThan(0)
    const first = (await filesOf(page, learnerId))[0]!
    const meta = first.meta!
    expect(["ts", "uniform", "confirm"]).toContain(meta.method)
    expect({ role: meta.role, method: meta.method, drawSeed: meta.drawSeed, posteriorSha: meta.posteriorSha, evalPanel: meta.evalPanel, evalSeed: meta.evalSeed, confirmation: meta.confirmation, genome: meta.genome })
      .toEqual({ role: d.role, method: d.method, drawSeed: d.drawSeed, posteriorSha: sha(V2_SEARCH_TEXT), evalPanel: d.evalPanel, evalSeed: d.evalSeed, confirmation: d.confirmation, genome: d.genome })
    // A confirmation life's learner seed is its evalSeed (the gate checks settings.seed = meta.evalSeed).
    if (d.role === "confirm") expect(first.settings.seed).toBe(d.evalSeed)
    expect(parseLogFile(JSON.parse(JSON.stringify(first))).ok).toBe(true)
    expect(errors).toEqual([])
    await page.close()
  }
})

test("a confirmation in the posterior and a learnerId whose draw is the newborn arm: one frozen confirm file is queued and a new life starts", async ({ page }) => {
  test.setTimeout(300_000) // a GPU-width newborn's three frozen checkups on the CPU (about a minute here)
  const learnerId = POOL.find(i => gpuDraw(i, V2_SEARCH_TEXT).confirmation?.arm === "newborn")!
  const d = gpuDraw(learnerId, V2_SEARCH_TEXT)
  const { errors } = await serve(page, { "posterior.json": V2_SEARCH_TEXT })
  await pin(page, learnerId)
  await page.goto(GPU_CLASS)
  await expect(page.getByTestId("learner-id")).toHaveText(learnerId)
  await expect(page.getByTestId("recipe-text")).toHaveText(`a confirmation life (newborn arm, fresh world ${d.evalSeed!.toLocaleString("en-US")})`, { timeout: 30_000 })
  await expect(page.getByTestId("status")).toHaveText(/^checking a confirmation newborn \(seed [\d,]+\) on its fresh worlds$/)
  // Its one file, then the page's next life: another learner, born from its own draw.
  await expect(page.getByTestId("learner-id")).not.toHaveText(learnerId, { timeout: 270_000 })
  const files = await filesOf(page, learnerId)
  expect(files).toHaveLength(1)
  const f = files[0]!
  expect(parseLogFile(JSON.parse(JSON.stringify(f))).ok).toBe(true)
  expect(f).toMatchObject({ seq: 0, final: true, cu: 0, stepsTotal: 0, settings: { seed: d.evalSeed } })
  expect(f.meta).toMatchObject({ role: "confirm", method: "confirm", evalPanel: null, evalSeed: d.evalSeed, confirmation: { id: 3, arm: "newborn" }, drawSeed: d.drawSeed, posteriorSha: sha(V2_SEARCH_TEXT), genome: d.genome })
  expect((f.checkups ?? []).map(c => c.level)).toEqual([...NEWBORN_LEVELS])
  // Gate data, contributed as any other file: not marked local-only.
  const contribution = await page.evaluate(i => new Promise<unknown>(res => {
    const req = indexedDB.open("blanktab")
    req.onsuccess = () => { const g = req.result.transaction("contribution").objectStore("contribution").get(i); g.onsuccess = () => res(g.result ?? null) }
  }), learnerId)
  // No contribution record at all: the queue's default for a learner, { localOnly: false }, so the contributor sends it
  // (a reference newborn's is written localOnly: true before its file, as the ?newborns test checks).
  expect(contribution).toBeNull()
  // The page's next life: its own draw under the same posterior, in the LEARNER tab's words.
  const next = await learnerIdOf(page)
  expect(next).not.toBe(learnerId)
  const nd = gpuDraw(next, V2_SEARCH_TEXT)
  await expect(page.getByTestId("recipe-text")).toHaveText(recipeText(nd as unknown as MetaBlock, parsePosterior(V2_SEARCH_TEXT)), { timeout: 30_000 })
  expect(errors).toEqual([])
})

test("new code while a confirmation newborn runs: the forced checkpoint is answered at once (not saved) and the page reloads within seconds, not CHECKPOINT_WAIT_MS", async ({ page }) => {
  test.setTimeout(120_000)
  const learnerId = POOL.find(i => gpuDraw(i, V2_SEARCH_TEXT).confirmation?.arm === "newborn")!
  const { files } = await serve(page, { "posterior.json": V2_SEARCH_TEXT })
  await pin(page, learnerId)
  await page.goto(`${GPU_CLASS}&testPollMs=500&testReloadJitterMs=300`)
  await expect(page.getByTestId("status")).toHaveText(/^checking a confirmation newborn \(seed [\d,]+\) on its fresh worlds$/, { timeout: 30_000 })
  let navigated = false
  page.on("framenavigated", f => { if (f === page.mainFrame()) navigated = true })
  files["manifest.json"] = { ...(files["manifest.json"] as object), codeSha: "b".repeat(40) }
  const t0 = Date.now()
  await expect.poll(() => navigated, { timeout: 20_000 }).toBe(true)
  expect(Date.now() - t0).toBeLessThan(20_000) // the page's own wait is 60 s
  await page.waitForLoadState("load")
  // Nothing of the newborn was queued: it runs again from its start on the reloaded page.
  expect(await filesOf(page, learnerId)).toEqual([])
  await expect(page.getByTestId("learner-id")).toHaveText(learnerId)
})

test("the committed calibrate posterior: the page shows 'Calibrating: …' and the first file's method is calibrate, uniform or centre", async ({ page }) => {
  const text = readFileSync(new URL("../../data/meta/posterior.json", import.meta.url), "utf8")
  expect(parsePosterior(text)!.mode).toBe("calibrate")
  const { errors } = await serve(page, { "posterior.json": text })
  await page.goto(GPU_CLASS)
  const id = await learnerIdOf(page)
  await expect(page.getByTestId("meta-state")).toHaveText(/^Calibrating: \d+ of \d+ lives measured, to learn whether the score can tell recipes apart\.$/)
  await expect.poll(async () => (await filesOf(page, id)).length, { timeout: 60_000 }).toBeGreaterThan(0)
  const meta = (await filesOf(page, id))[0]!.meta!
  expect(["calibrate", "uniform", "centre"]).toContain(meta.method)
  expect(meta).toMatchObject({ ...(({ role, method, drawSeed, evalPanel, genome }) => ({ role, method, drawSeed, evalPanel, genome }))(gpuDraw(id, text)), posteriorSha: sha(text) })
  expect(errors).toEqual([])
})

test("a v2 posterior with mode calibrate and a model (mode guard): the first file has method centre and posteriorSha null; the meta-learner line reads 'Not reachable: this learner runs the current recipe.'", async ({ page }) => {
  const text = v2SearchText(f => { f.mode = "calibrate"; f.confirmation = null })
  expect(parsePosterior(text)).toBeNull()
  const { errors } = await serve(page, { "posterior.json": text })
  await page.goto(GPU_CLASS)
  const id = await learnerIdOf(page)
  await expect(page.getByTestId("meta-state")).toHaveText("Not reachable: this learner runs the current recipe.")
  await expect(page.getByTestId("recipe-text")).toHaveText("the current recipe")
  await expect.poll(async () => (await filesOf(page, id)).length, { timeout: 60_000 }).toBeGreaterThan(0)
  expect((await filesOf(page, id))[0]!.meta).toMatchObject({ role: "search", method: "centre", posteriorSha: null, drawSeed: null, genome: CENTRE })
  expect(errors).toEqual([])
})
