// Plain DOM rendering. No framework: three tabs, a handful of labelled fields, four canvases and a table.
import { FOVEA, PERIPHERY } from "@blanktab/interface"
import { REPO, type CheckupResult, type ContributorStatus, type CurvePoint, type Knowledge, type MetaBlock } from "@blanktab/harness"
import evaluation from "../../config/evaluation.json" with { type: "json" }
import type { DrawContext } from "./birth"
import { appendPoints, fitCanvas, ownStreamText, ratioText, sparkline } from "./curve"
import { appendCheckups, milestoneBars, milestoneText } from "./milestones"
import { NEWBORN_RUN_TEXT, currentRecipeText, genomeHighlights, metaLineOf, missingText, recipeText, scoreText, unlabelledText } from "./recipe"
import { HISTORY_COLUMNS, buildLabel, fmt, fmtSettings, historyRows, siteLinks } from "./site"

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel)
  if (!el) throw new Error(`missing ${sel}`)
  return el
}
const byTest = (id: string) => $<HTMLElement>(`[data-testid="${id}"]`)

function painter(canvas: HTMLCanvasElement, w: number, h: number): (rgb: Uint8Array) => void {
  const ctx = canvas.getContext("2d")
  if (!ctx) throw new Error("2d context unavailable")
  const img = ctx.createImageData(w, h)
  return rgb => {
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      img.data[j] = rgb[i]!; img.data[j + 1] = rgb[i + 1]!; img.data[j + 2] = rgb[i + 2]!; img.data[j + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }
}

/**
 * A chart canvas sized to its CSS box at this display's pixel ratio, cleared, with a transform to CSS pixels. Null while
 * it is not laid out (its tab hidden): nothing is drawn or resized then (the tab redraws it when shown).
 */
function prepareCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const ctx = canvas.getContext("2d")
  const fit = fitCanvas(canvas.clientWidth, canvas.clientHeight, globalThis.devicePixelRatio || 1)
  if (!ctx || !fit) return null
  if (canvas.width !== fit.width || canvas.height !== fit.height) {
    canvas.width = fit.width
    canvas.height = fit.height
  }
  ctx.setTransform(fit.dpr, 0, 0, fit.dpr, 0, 0)
  ctx.clearRect(0, 0, fit.w, fit.h)
  return { ctx, w: fit.w, h: fit.h }
}

/** Compute units as shown: three decimals below 10 CU, one above. */
const fmtCU = (cu: number) => (cu < 10 ? cu.toFixed(3) : cu.toFixed(1))

/**
 * The loss curve on the held-out set (decision 0015 §3): both series on one scale, the learner's guesses solid, the
 * lazy guess dashed over them. data-points counts the points drawn (those with a held-out score).
 */
function drawCurve(canvas: HTMLCanvasElement, points: readonly CurvePoint[]): void {
  canvas.dataset.points = String(sparkline(points, 1, 1).pred.length)
  const p = prepareCanvas(canvas)
  if (!p) return
  const { ctx, w, h } = p
  const css = getComputedStyle(canvas)
  const line = (pts: [number, number][], color: string, dash: number[]) => {
    if (pts.length === 0) return
    ctx.beginPath()
    ctx.setLineDash(dash)
    ctx.strokeStyle = color
    ctx.lineWidth = 1.75
    ctx.lineJoin = "round"
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    if (pts.length === 1) ctx.lineTo(pts[0]![0] + 0.01, pts[0]![1])
    ctx.stroke()
  }
  const { pred, base } = sparkline(points, w, h, 4)
  // The lazy guess goes on top, dashed: where the learner is no better than it, the two lines coincide and both still show.
  line(pred, css.getPropertyValue("--curve-pred").trim() || "teal", [])
  line(base, css.getPropertyValue("--curve-base").trim() || "orange", [5, 4])
}

/**
 * The milestone curve: one bar per checkup, its height the share of the six tests passed; a dashed line at the top is
 * all six. data-bars counts the bars drawn.
 */
function drawMilestones(canvas: HTMLCanvasElement, results: readonly CheckupResult[]): void {
  canvas.dataset.bars = String(results.length)
  const p = prepareCanvas(canvas)
  if (!p) return
  const { ctx, w, h } = p
  const css = getComputedStyle(canvas)
  const pad = 4
  ctx.beginPath()
  ctx.setLineDash([3, 4])
  ctx.strokeStyle = css.getPropertyValue("--muted").trim() || "gray"
  ctx.globalAlpha = 0.5
  ctx.lineWidth = 1
  ctx.moveTo(pad, pad + 0.5)
  ctx.lineTo(w - pad, pad + 0.5)
  ctx.stroke()
  ctx.globalAlpha = 1
  ctx.setLineDash([])
  ctx.fillStyle = css.getPropertyValue("--accent-green").trim() || "green"
  for (const b of milestoneBars(results, w, h, pad)) ctx.fillRect(b.x, b.y, b.w, b.h)
}

/** Client-side tabs: real buttons with aria-selected, arrow keys move between them. Idempotent. `shown` runs after each switch. */
function setupTabs(shown: () => void): void {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')]
  const select = (t: HTMLButtonElement, focus: boolean) => {
    for (const x of tabs) {
      const on = x === t
      x.setAttribute("aria-selected", String(on))
      x.tabIndex = on ? 0 : -1
      $(`#${x.getAttribute("aria-controls")!}`).hidden = !on
    }
    if (focus) t.focus()
    shown()
  }
  tabs.forEach((t, i) => {
    t.onclick = () => select(t, false)
    t.onkeydown = e => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0
      if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length]!, true) }
    }
  })
}

/** Everything that doesn't depend on the learner: links, repo name, build id. */
function renderStatic(): void {
  for (const ul of document.querySelectorAll("[data-links]")) {
    ul.replaceChildren(...siteLinks(REPO).map(({ label, href }) => {
      const li = document.createElement("li"), a = document.createElement("a")
      a.href = href; a.textContent = label; a.target = "_blank"; a.rel = "noopener noreferrer"
      li.append(a)
      return li
    }))
  }
  for (const el of document.querySelectorAll("[data-repo]")) el.textContent = `${REPO.owner}/${REPO.name}`
  const b = buildLabel(__CODE_SHA__, __BUILD_TIME__, REPO), build = byTest("build") as HTMLAnchorElement
  build.textContent = b.text
  build.href = b.href
}

function renderHistory(k: Knowledge): void {
  const box = byTest("history"), rows = historyRows(k.summary, REPO, SEARCH_EVALUATOR)
  // The centre in force (knowledge/genome.json's provenance): it changes only on a passed confirmation.
  const current = byTest("current-recipe"), text = k.meta ? currentRecipeText(k.meta.genome) : null
  current.textContent = text ?? ""
  current.hidden = text === null
  box.querySelector("tbody")!.replaceChildren(...rows.map(r => {
    const tr = document.createElement("tr")
    const a = document.createElement("a")
    a.href = r.href; a.textContent = r.learnerId; a.target = "_blank"; a.rel = "noopener noreferrer"
    const values: (string | Node)[] = [a, r.steps, r.cu, r.heldOutRatio, r.lossRatio, r.milestones, r.recipe, r.score, r.final, r.setting, r.host]
    for (const [i, c] of values.entries()) {
      const { label, numeric } = HISTORY_COLUMNS[i]!
      const td = document.createElement("td")
      if (numeric) td.className = "num"
      td.dataset.label = label // the stacked mobile layout shows it
      td.append(c)
      tr.append(td)
    }
    return tr
  }))
  box.querySelector("table")!.hidden = rows.length === 0
  box.querySelector<HTMLElement>(".empty")!.hidden = rows.length > 0
}

/** What the recipe line shows: a life's block (with the knowledge's draw context, if readable), none, or a newborn reference. */
export type RecipeView = { meta: MetaBlock; ctx: DrawContext | null } | { unlabelled: string } | { newborn: string }

/** The pinned search evaluator (config/evaluation.json): the life's own score is shown under it (spec §6). */
const SEARCH_EVALUATOR = evaluation.search

export interface Ui {
  knowledge(k: Knowledge): void
  learner(o: { id: string; sessionsBefore: number; lifetimeCU?: number }): void
  /** The settings the learner runs with (as the worker reports them). */
  settings(settings: Record<string, number | string>): void
  backend(label: string): void
  frame(f: { steps: number; cu: number; stepsPerSec: number; periphery: Uint8Array; fovea: Uint8Array }): void
  progress(steps: number, cu: number): void
  /** `reset`: these points are the whole curve so far; otherwise they are added to it. */
  curve(points: CurvePoint[], reset: boolean): void
  /** Milestone checkups: `reset`, these are all of them so far; otherwise they are added (a level already shown is not). */
  checkups(results: CheckupResult[], reset: boolean): void
  /** The caregiver's line (what it has done for this learner). */
  caregiver(text: string): void
  /** The latest thing that happened to the learner (resumed, restored, moved to the CPU). */
  status(message: string): void
  /** A note in the life line that is not an error (e.g. a learner that cannot be resumed, a blocked database). */
  notice(message: string): void
  /** About this page's code (a newer version that cannot resume this learner): kept apart from the life notices, which replace each other. */
  update(message: string): void
  contribution(s: ContributorStatus): void
  /** `ended: "budget"`: the short-lives switch ended it at its budget checkup. */
  lifeComplete(id: string, ended?: "budget"): void
  /** The recipe line: how this life's genome was drawn, its furthest genes; its settings and full genome behind a toggle. */
  recipe(r: RecipeView): void
  /** Newborn reference mode: the meta-learner line says so, and the short-lives switch (which ends lives) is hidden. */
  newbornMode(): void
  /** The short-lives switch as stored. */
  shortLives(on: boolean): void
  /** The handler returns false to refuse the change (the box goes back). */
  onShortLives(handler: (on: boolean) => boolean): void
  error(message: string): void
  tokenPanel(visible: boolean): void
  /** The small "change token" control, shown whenever a token is stored. */
  changeToken(visible: boolean): void
  onChangeToken(handler: () => void): void
  /** The handler returns an error message to show, or null on success. */
  onTokenSubmit(handler: (token: string) => Promise<string | null>): void
}

export function createUi(): Ui {
  const paintPeriphery = painter($("#periphery"), PERIPHERY.width, PERIPHERY.height)
  const paintFovea = painter($("#fovea"), FOVEA.width, FOVEA.height)
  const panel = byTest("token-panel") as HTMLDetailsElement, input = $<HTMLInputElement>("#token"), button = $<HTMLButtonElement>("#token-save"), err = $("#token-error")
  const life = byTest("life"), change = byTest("change-token") as HTMLButtonElement
  const curveCanvas = byTest("curve") as HTMLCanvasElement
  const milestonesCanvas = byTest("milestones") as HTMLCanvasElement
  let lifetime: number | undefined
  /** The learner shown, and the latest knowledge: its score is its summary row's (once there is one). */
  let shownId: string | null = null
  let summary: Knowledge["summary"] = { schemaVersion: 1, sessions: [] }
  const showScore = () => {
    const text = scoreText(summary.sessions.find(r => r.learnerId === shownId), SEARCH_EVALUATOR)
    byTest("score").textContent = text ?? ""
    byTest("score").hidden = byTest("score-label").hidden = text === null
  }
  const shortLivesBox = byTest("short-lives") as HTMLInputElement
  let newborns = false
  let curve: CurvePoint[] = []
  let checkups: CheckupResult[] = []
  // Charts are not drawn while their tab is hidden (they have no size then): drawn again when it is shown.
  const redraw = () => {
    drawCurve(curveCanvas, curve)
    drawMilestones(milestonesCanvas, checkups)
  }
  setupTabs(redraw)
  // Clamped prose (shown in full only when asked, on a short window): its "more" button opens and closes it.
  for (const more of document.querySelectorAll<HTMLButtonElement>("button[data-more]")) {
    const target = $<HTMLElement>(`#${more.getAttribute("aria-controls")!}`)
    more.addEventListener("click", () => {
      const open = target.classList.toggle("expanded")
      more.setAttribute("aria-expanded", String(open))
      more.textContent = open ? "less" : "more"
    })
  }
  // A resize changes the canvases' CSS boxes: redraw at the new size (coalesced to one frame; a hidden canvas has no size and is skipped).
  let frame = 0
  const ro = new ResizeObserver(() => {
    if (frame) return
    frame = requestAnimationFrame(() => { frame = 0; redraw() })
  })
  ro.observe(curveCanvas)
  ro.observe(milestonesCanvas)
  renderStatic()

  return {
    knowledge(k) {
      byTest("knowledge").textContent = k.offline
        ? "offline: the shared knowledge could not be loaded, so this learner starts from nothing."
        : `loaded (knowledgeSha ${k.manifest.knowledgeSha.slice(0, 12)}…)`
      if (!newborns) byTest("meta-state").textContent = metaLineOf(k)
      summary = k.summary
      showScore()
      renderHistory(k)
    },
    learner({ id, sessionsBefore, lifetimeCU }) {
      lifetime = lifetimeCU
      shownId = id
      showScore()
      byTest("learner-id").textContent = id
      byTest("sessions").textContent = `${sessionsBefore} session${sessionsBefore === 1 ? "" : "s"} before this one`
      byTest("setting").textContent = "…"
      byTest("backend").textContent = "choosing…"
      byTest("rate").textContent = "…"
      byTest("status").textContent = "newborn"
      byTest("caregiver").textContent = "…"
      byTest("recipe-text").textContent = "drawing…"
      byTest("recipe-genes").textContent = ""
      byTest("genome-box").hidden = true
      this.progress(0, 0)
      this.curve([], true)
      this.checkups([], true)
    },
    settings(settings) {
      byTest("setting").textContent = fmtSettings(settings)
    },
    backend(label) {
      byTest("backend").textContent = label
    },
    frame({ steps, cu, stepsPerSec, periphery, fovea }) {
      paintPeriphery(periphery)
      paintFovea(fovea)
      this.progress(steps, cu)
      byTest("rate").textContent = `${stepsPerSec.toFixed(1)} steps/s`
    },
    progress(steps, cu) {
      byTest("steps").textContent = String(steps)
      byTest("cu").textContent = lifetime === undefined ? `${fmtCU(cu)} CU` : `${fmtCU(cu)} of ${fmt(lifetime)} CU`
    },
    curve(points, reset) {
      curve = appendPoints(reset ? [] : curve, points)
      drawCurve(curveCanvas, curve)
      byTest("ratio").textContent = ratioText(curve)
      byTest("own-ratio").textContent = ownStreamText(curve)
    },
    checkups(results, reset) {
      checkups = appendCheckups(reset ? [] : checkups, results)
      drawMilestones(milestonesCanvas, checkups)
      byTest("milestone-text").textContent = milestoneText(checkups)
    },
    caregiver(text) {
      byTest("caregiver").textContent = text
    },
    status(message) {
      byTest("status").textContent = message
    },
    notice(message) {
      life.hidden = false
      life.classList.remove("error")
      life.textContent = message
    },
    update(message) {
      const el = byTest("update")
      el.hidden = false
      el.textContent = message
    },
    contribution(s) {
      const waiting = s.pending > 0 ? ` (${s.pending} log${s.pending === 1 ? "" : "s"} waiting in this browser)` : ""
      byTest("contribution").textContent = s.message + waiting
    },
    lifeComplete(id, ended) {
      life.hidden = false
      life.classList.remove("error")
      life.textContent = ended === "budget"
        ? `This learner's life is complete (${id}): short lives are on, so it ended at its budget checkup. A newborn takes its place.`
        : `This learner's life is complete (${id}). A newborn takes its place.`
    },
    recipe(r) {
      const genes = byTest("recipe-genes"), box = byTest("genome-box")
      box.hidden = true
      genes.textContent = ""
      if ("unlabelled" in r) { byTest("recipe-text").textContent = unlabelledText(r.unlabelled); return }
      if ("newborn" in r) { byTest("recipe-text").textContent = r.newborn; return }
      const { meta, ctx } = r
      // A ts life's words name the model it was drawn from: only that posterior's (a resumed life's may have changed since).
      const drawnFrom = meta.method !== "ts" || ctx?.posteriorSha === meta.posteriorSha ? ctx?.posterior ?? null : null
      byTest("recipe-text").textContent = recipeText(meta, drawnFrom)
      // The highlights need the genome in the page's space (a legacy genome of an older schema has none).
      let highlights: string[] = []
      try {
        if (ctx && meta.genomeSchema === ctx.space.schemaVersion) highlights = genomeHighlights(meta.genome, ctx.centre, ctx.space)
      } catch {
        highlights = []
      }
      const missing = missingText(meta.missing ?? [])
      genes.textContent = [highlights.join(", "), missing].filter(Boolean).map(t => ` · ${t}`).join("")
      byTest("genome-list").textContent = Object.entries(meta.genome).map(([k, v]) => `${k} = ${fmt(v)}`).join(", ")
      byTest("genome-shas").textContent = `genomeSha ${meta.genomeSha.slice(0, 8)} · posteriorSha ${meta.posteriorSha === null ? "none" : meta.posteriorSha.slice(0, 8)}`
      box.hidden = false
    },
    newbornMode() {
      newborns = true
      byTest("meta-state").textContent = NEWBORN_RUN_TEXT
      byTest("short-lives-switch").hidden = true
    },
    shortLives(on) {
      shortLivesBox.checked = on
    },
    onShortLives(handler) {
      shortLivesBox.addEventListener("change", () => {
        if (!handler(shortLivesBox.checked)) shortLivesBox.checked = !shortLivesBox.checked
      })
    },
    error(message) {
      life.hidden = false
      life.classList.add("error")
      life.textContent = `The learner stopped: ${message}`
    },
    tokenPanel(visible) {
      panel.hidden = !visible
      if (!visible) panel.open = false // folded again the next time it is offered
    },
    changeToken(visible) {
      change.hidden = !visible
    },
    onChangeToken(handler) {
      // Asking to change the token opens the form at once.
      change.addEventListener("click", () => { panel.open = true; handler() })
    },
    onTokenSubmit(handler) {
      const submit = async () => {
        const token = input.value.trim()
        if (!token) { err.textContent = "Paste a token first."; return }
        button.disabled = true
        err.textContent = "Checking with GitHub…"
        try {
          const problem = await handler(token)
          err.textContent = problem ?? ""
          if (problem === null) input.value = ""
        } catch {
          // Never leave "Checking with GitHub…" up forever. The cause is not shown: it could echo the token.
          err.textContent = "Something went wrong while saving the token; try again."
        } finally {
          button.disabled = false
        }
      }
      button.addEventListener("click", () => void submit())
      input.addEventListener("keydown", e => { if (e.key === "Enter") void submit() })
    },
  }
}
