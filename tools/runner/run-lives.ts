// Usage: tsx tools/runner/run-lives.ts [pageUrl] [--profile .local/runner-profile] [--hours 16] [--setup] [--headed]
//   (run from the repo root; pageUrl defaults to the live site)
// Runs short lives for calibration in a headless Chromium tab that no browser can suspend in the background (the
// 2026-10-25 Safari stall). The tab is the ordinary page: it draws its own recipes, ends each life at its budget
// checkup and sends its logs as PRs with the token stored in this profile, exactly as a person's tab would.
//
// --setup opens the page headed once: paste a token in the page's own token panel. The token is stored by the page in
// this profile's localStorage and never read by this script, which only checks that one is there (decision 0010).
//
// The runner turns the short-lives switch on (it only ends lives born after it is on, or confirmed in the page), keeps
// macOS awake with caffeinate while it runs, and reloads the tab if it crashes or its compute stops advancing.
import { spawn } from "node:child_process"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"

export const SITE = "https://markwhiting.github.io/learner/"
const TOKEN_KEY = "blanktab:githubToken" // packages/harness/src/token.ts
const SHORT_LIVES_KEY = "blanktab:shortLives" // app/src/main.ts

export interface RunnerArgs { url: string; profile: string; hours: number; setup: boolean; headed: boolean }

export function parseArgs(argv: readonly string[]): RunnerArgs {
  const value = (name: string) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : undefined }
  const valued = new Set(["--profile", "--hours"])
  const positional = argv.filter((a, i) => !a.startsWith("--") && !valued.has(argv[i - 1] ?? ""))
  const hours = Number(value("hours") ?? 16)
  if (!(hours > 0 && hours <= 24 * 14)) throw new Error(`--hours must be in (0, 336], got ${JSON.stringify(value("hours"))}`)
  return {
    url: positional[0] ?? SITE,
    profile: value("profile") ?? ".local/runner-profile",
    hours,
    setup: argv.includes("--setup"),
    headed: argv.includes("--headed") || argv.includes("--setup"),
  }
}

/** A compute reading's CU as the page shows it ("1,234.5 CU", "…" before the first step), or null. */
export function cuOf(text: string): number | null {
  const m = /([\d,]+(?:\.\d+)?)/.exec(text)
  return m ? Number(m[1]!.replaceAll(",", "")) : null
}

/**
 * Stall watch: a tab whose learner and compute have not changed for `limitMs` is stuck (a newborn's first checkup and
 * a code-update reload each take well under 10 minutes on a GPU).
 */
export function stallWatch(limitMs: number) {
  let key = "", since = 0
  return (learner: string, cu: number | null, now: number): boolean => {
    const k = `${learner}|${cu}`
    if (k !== key) { key = k; since = now; return false }
    return now - since > limitMs
  }
}

interface Reading { learner: string; cu: string; status: string; contribution: string; meta: string; life: string; tokenStored: boolean }

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const log = (s: string) => console.log(`${new Date().toISOString()} ${s}`)
  // Keep the machine awake (idle sleep) for exactly as long as this process lives.
  if (process.platform === "darwin") spawn("caffeinate", ["-i", "-w", String(process.pid)], { stdio: "ignore", detached: true }).unref()
  const { chromium } = await import("@playwright/test")
  const context = await chromium.launchPersistentContext(resolve(args.profile), {
    headless: !args.headed, args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--enable-features=Vulkan"],
  })
  // Before any page script on every load (the page reloads itself onto new code): the switch is on.
  await context.addInitScript(([k]) => localStorage.setItem(k!, "1"), [SHORT_LIVES_KEY])
  const page = context.pages()[0] ?? await context.newPage()
  page.on("console", m => { if (m.type() === "error") log(`page error: ${m.text()}`) })
  // A string, not a function: tsx compiles functions with a __name helper the page does not have.
  const read = (): Promise<Reading> => page.evaluate(`(() => {
    const t = id => document.querySelector('[data-testid="' + id + '"]')?.textContent?.trim().replace(/\\s+/g, " ") ?? ""
    return {
      learner: t("learner-id"), cu: t("cu"), status: t("status"), contribution: t("contribution"), meta: t("meta-state"),
      life: t("life"), tokenStored: localStorage.getItem(${JSON.stringify(TOKEN_KEY)}) !== null,
    }
  })()`)
  try {
    await page.goto(args.url)
    if (args.setup) {
      log("paste a token (public_repo scope) in the page's token panel; the runner closes once the page has stored it")
      await page.waitForFunction(([k]) => localStorage.getItem(k!) !== null, [TOKEN_KEY], { timeout: 1_800_000, polling: 1_000 })
      log(`token stored in ${args.profile}; contribution: ${(await read()).contribution}`)
      return
    }
    if (!(await read()).tokenStored) throw new Error(`no token in ${args.profile}: run once with --setup`)
    const until = Date.now() + args.hours * 3_600_000
    const stalled = stallWatch(10 * 60_000)
    const seen = new Set<string>()
    let last = ""
    page.on("crash", () => log("tab crashed"))
    while (Date.now() < until) {
      await new Promise(r => setTimeout(r, 60_000))
      let r: Reading
      try { r = await read() } catch (e) { log(`read failed (${(e as Error).message.split("\n")[0]}); reloading`); await page.reload().catch(() => {}); continue }
      if (/^lrn_/.test(r.learner)) seen.add(r.learner)
      const line = `${r.learner} · ${r.cu} · ${r.status} · ${r.contribution} · meta: ${r.meta}${r.life ? ` · ${r.life}` : ""}`
      if (line !== last) { log(`${line} (lives seen: ${seen.size})`); last = line }
      if (stalled(r.learner, cuOf(r.cu), Date.now())) { log("no progress for 10 min; reloading"); await page.reload().catch(() => {}) }
    }
    log(`--hours ${args.hours} reached; ${seen.size} lives seen`)
  } finally {
    await context.close()
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
