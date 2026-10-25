// Usage: tsx tools/eval/collect-newborns.ts <pageUrl> [--newborns a-b] [--out data/newborns] [--profile .local/newborn-profile] [--collect-only] [--headed]
//   (run from the repo root; e.g. tsx tools/eval/collect-newborns.ts https://markwhiting.github.io/learner/ --newborns 0-63)
// Runs newborn reference lives in a real Chromium tab (?newborns=a-b, on WebGPU where the machine has it) and collects
// their files from that browser's IndexedDB outbox into <out>/<learnerId>/<seq>.json, the layout
// tools/eval/newborn-reference.ts reads (controller ruling B: collected locally, not sent as 64 PRs). The browser
// profile persists in --profile, so a run can be collected later with --collect-only. Every file is checked as the
// gate would check it (parseLogFile, then validateSessionPr against this checkout's knowledge/genome-space.json)
// before it is written; a file already there with other bytes is never overwritten.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { LEARNER_ID_RE, parseLogFile } from "@blanktab/harness/log"
import { NEWBORN_COUNT } from "@blanktab/harness/newborn"
import type { GenomeSpace } from "@blanktab/meta"
import { loadGenomeSpace } from "../gate/gate.ts"
import { spacesOf, validateSessionPr } from "../gate/validate.ts"

/** One outbox row as the page stores it (packages/harness/src/queue.ts). */
export interface OutboxRow { learnerId: string; seq: number; file: unknown }
export interface CollectResult { written: string[]; same: string[]; skipped: number; errors: string[] }

/** The file's bytes on disk: 2-space JSON and a final newline (the repo's JSON hygiene). */
const bytesOf = (file: unknown) => JSON.stringify(file, null, 2) + "\n"
const isNewborn = (file: unknown) => (file as { meta?: { role?: unknown } } | null)?.meta?.role === "newborn"

/**
 * Writes the outbox's newborn files (meta.role "newborn"; any other learner's are skipped) to <outDir>/<learnerId>/<seq>.json,
 * each checked first as the gate would check it as its learner's first PR. A file already on disk with the same bytes
 * is left (same); with other bytes it is an error, as is any file that does not parse or validate.
 */
export async function writeCollected(rows: readonly OutboxRow[], outDir: string, space: GenomeSpace): Promise<CollectResult> {
  const out: CollectResult = { written: [], same: [], skipped: 0, errors: [] }
  for (const { learnerId, seq, file } of rows) {
    if (!isNewborn(file)) { out.skipped++; continue }
    const at = `${learnerId}/${seq}.json`
    if (!LEARNER_ID_RE.test(learnerId) || !Number.isSafeInteger(seq) || seq < 0) { out.errors.push(`${at}: not a learnerId and seq`); continue }
    const parsed = parseLogFile(JSON.parse(JSON.stringify(file)))
    if (!parsed.ok) { out.errors.push(`${at}: ${parsed.error}`); continue }
    const bytes = bytesOf(file)
    const v = await validateSessionPr([{ path: `data/runs/${at}`, content: bytes }], [], { spaces: spacesOf(space) })
    if (v.errors.length > 0) { out.errors.push(...v.errors.map(e => `${at}: ${e}`)); continue }
    const path = join(outDir, learnerId, `${seq}.json`)
    if (existsSync(path)) {
      if (readFileSync(path, "utf8") === bytes) out.same.push(at)
      else out.errors.push(`${at}: ${path} already exists with other bytes (not overwritten)`)
      continue
    }
    mkdirSync(join(outDir, learnerId), { recursive: true })
    writeFileSync(path, bytes)
    out.written.push(at)
  }
  return out
}

/** ?newborns=a-b's range from "a-b" (0 ≤ a ≤ b < NEWBORN_COUNT), or a thrown error. */
export function rangeOf(s: string): { from: number; to: number } {
  const m = /^(\d+)-(\d+)$/.exec(s)
  const from = Number(m?.[1]), to = Number(m?.[2])
  if (!m || !(from <= to && to < NEWBORN_COUNT)) throw new Error(`--newborns must be a-b with 0 ≤ a ≤ b < ${NEWBORN_COUNT}, got ${JSON.stringify(s)}`)
  return { from, to }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.includes(`--${name}`)
  const value = (name: string, d: string) => { const i = args.indexOf(`--${name}`); return i >= 0 && args[i + 1] !== undefined ? args[i + 1]! : d }
  const url = args[0]
  if (!url || url.startsWith("--")) throw new Error("usage: tsx tools/eval/collect-newborns.ts <pageUrl> [--newborns a-b] [--out data/newborns] [--profile dir] [--collect-only] [--headed]")
  const root = process.cwd()
  const space = loadGenomeSpace(root)
  const range = rangeOf(value("newborns", `0-${NEWBORN_COUNT - 1}`))
  const { chromium } = await import("@playwright/test")
  // WebGPU on this machine's GPU where it has one (the reference is measured at the GPU widths class).
  const context = await chromium.launchPersistentContext(resolve(value("profile", ".local/newborn-profile")), {
    headless: !flag("headed"), args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--enable-features=Vulkan"],
  })
  try {
    const page = await context.newPage()
    page.on("console", m => { if (m.type() === "error" || m.type() === "warning") console.error(`page ${m.type()}: ${m.text()}`) })
    if (!flag("collect-only")) {
      const u = new URL(url)
      u.searchParams.set("newborns", `${range.from}-${range.to}`)
      await page.goto(u.href)
      await page.waitForFunction(() => !/choosing|…/.test(document.querySelector('[data-testid="backend"]')?.textContent ?? ""), null, { timeout: 120_000 }).catch(() => {})
      console.log(`running newborns ${range.from}–${range.to} at ${u.href} (${(await page.getByTestId("backend").textContent()) ?? ""})`)
      const life = page.getByTestId("life")
      const done = `Newborn references ${range.from}–${range.to} are done`
      // About a minute a newborn on a GPU (three frozen checkups and a held-out score); no limit beyond a day.
      await page.waitForFunction(([t]) => {
        const el = document.querySelector('[data-testid="life"]')
        return el !== null && (el.textContent?.includes(t!) || el.classList.contains("error"))
      }, [done], { timeout: 86_400_000, polling: 2_000 })
      const text = (await life.textContent()) ?? ""
      if (!text.includes(done)) throw new Error(`the page stopped: ${text}`)
    } else {
      // Any page of the origin reads its IndexedDB; a knowledge file starts no life in this profile.
      await page.goto(new URL("knowledge/manifest.json", url).href)
    }
    const rows = await page.evaluate(() => new Promise<{ learnerId: string; seq: number; file: unknown }[]>((res, rej) => {
      const req = indexedDB.open("blanktab")
      req.onerror = () => rej(req.error)
      req.onsuccess = () => {
        const all = req.result.transaction("outbox").objectStore("outbox").getAll()
        all.onsuccess = () => { req.result.close(); res(all.result as { learnerId: string; seq: number; file: unknown }[]) }
        all.onerror = () => rej(all.error)
      }
    }))
    const r = await writeCollected(rows, resolve(value("out", "data/newborns")), space)
    console.log(`${r.written.length} written, ${r.same.length} already there, ${r.skipped} other files skipped`)
    for (const e of r.errors) console.error(e)
    if (r.errors.length > 0) process.exitCode = 1
  } finally {
    await context.close()
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
