import { describe, it, expect } from "vitest"
import { execFileSync } from "node:child_process"
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { createRng, hashSeed } from "@blanktab/detmath"
import type { LifeInput } from "@blanktab/evaluators"
import { parseLogFile, type LogFileV2 } from "@blanktab/harness/log"
import { FOVEA, PERIPHERY, type Observation } from "@blanktab/interface"
import { GPU_WIDTHS } from "@blanktab/learner"
import { CPU } from "../../../packages/evaluators/test/fixtures/build.ts"
import { ProbeRefusal, main, runProbe, writeProbe } from "../filter-convergence.ts"
import { initFirstLayer, probeLives } from "../filters/lives.ts"
import { writeFilterPngs } from "../filters/png.ts"
import { FIXTURE_GENOMES, FIXTURE_MANIFEST_SHA, fixtureLife } from "./fixture-lives.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
const GOLDEN = join(REPO, "tools/probe/test/fixtures/filter-probe-golden.json")
const id = (k: number) => "lrn_" + (0xf00 + k).toString(16).padStart(16, "0")

/** 8 seeded synthetic observations: no canvas needed. */
function fixtureFrames(): Observation[] {
  const r = createRng(hashSeed(0, "filter-probe/fixture-frames"))
  const bytes = (n: number) => Uint8Array.from({ length: n }, () => Math.floor(r() * 256))
  return Array.from({ length: 8 }, () => ({
    periphery: bytes(PERIPHERY.width * PERIPHERY.height * 3), fovea: bytes(FOVEA.width * FOVEA.height * 3), cursor: { x: 0.5, y: 0.5 }, buttonHeld: false,
  }))
}
/** 3 genomes × 4 lives; each genome's lives are pulled toward that genome's own filter set by 2,048 CU. */
function fixtureLives(): LifeInput[] {
  const targets = FIXTURE_GENOMES.map((_, g) => initFirstLayer(90_000 + g, GPU_WIDTHS))
  return Array.from({ length: 12 }, (_, k) => fixtureLife({
    learnerId: id(k), genomeSha: FIXTURE_GENOMES[k % 3]!, seed: 300 + k, noise: 0.02, toward: { layer: targets[k % 3]!, at2048: 0.6 },
  }))
}
/** Two lives' meta.filters swapped in every file (a decoding or seed mix-up). */
function swapSnapshots(lives: LifeInput[], a: number, b: number): LifeInput[] {
  const out = lives.map(l => ({ ...l, files: l.files.map(f => structuredClone(f)) }))
  const fa = out[a]!.files as LogFileV2[], fb = out[b]!.files as LogFileV2[]
  fa.forEach((f, i) => { const t = f.meta!.filters; f.meta!.filters = fb[i]!.meta!.filters; fb[i]!.meta!.filters = t })
  return out
}
const run = (lives = fixtureLives()) => runProbe({ frames: fixtureFrames(), manifestSha: FIXTURE_MANIFEST_SHA, lives, centreSha: FIXTURE_GENOMES[0] })
/** Lives as files under a fresh runs dir: <runs>/<learnerId>/<seq>.json. */
function writeRuns(lives: LifeInput[]): string {
  const runs = mkdtempSync(join(tmpdir(), "probe-runs-"))
  for (const l of lives) {
    mkdirSync(join(runs, l.learnerId))
    for (const f of l.files) writeFileSync(join(runs, l.learnerId, `${f.seq}.json`), JSON.stringify(f))
  }
  return runs
}
const tmp = () => mkdtempSync(join(tmpdir(), "probe-out-"))

describe("the filter-convergence probe", () => {
  it("every fixture life's files pass parseLogFile, and probeLives keeps all 12", () => {
    const lives = fixtureLives()
    for (const l of lives) for (const f of l.files) expect(parseLogFile(JSON.parse(JSON.stringify(f)))).toMatchObject({ ok: true })
    expect(probeLives(lives).lives).toHaveLength(12)
  })

  it("the fixture run's report.json equals filter-probe-golden.json byte for byte", () => {
    const dir = tmp()
    writeProbe(run(), dir)
    const got = readFileSync(join(dir, "report.json"), "utf8")
    if (process.env.FILTER_PROBE_WRITE === "1") writeFileSync(GOLDEN, got)
    expect(got).toBe(readFileSync(GOLDEN, "utf8"))
  })

  it("two fixture runs write identical report.json and summary.md bytes", () => {
    const a = tmp(), b = tmp()
    writeProbe(run(), a)
    writeProbe(run(), b)
    for (const f of ["report.json", "summary.md"]) expect(readFileSync(join(a, f), "utf8")).toBe(readFileSync(join(b, f), "utf8"))
  })

  it("runProbe returns status 'identification failed', with every verdict withheld, when two fixture lives' snapshots are swapped; and throws ProbeRefusal('no eligible lives') on lives that are all excluded", () => {
    const r = run(swapSnapshots(fixtureLives(), 1, 4)).report
    expect(r.status).toBe("identification failed")
    for (const a of r.convergence) expect(a.verdict).toMatchObject({ evaluable: false, converged: null, convergedHolm: null })
    const cpu = fixtureLives().map(l => ({ ...l, files: l.files.map(f => ({ ...f, widths: CPU }) as LogFileV2) }))
    expect(() => run(cpu)).toThrow(ProbeRefusal)
    try { run(cpu) } catch (e) { expect((e as ProbeRefusal).why).toBe("no eligible lives") }
  })

  it("main with --manifest pointing at a temp copy with one frame sha changed returns 1 and leaves --out empty", async () => {
    const m = JSON.parse(readFileSync(join(REPO, "tools/probe/filter-probe-set.json"), "utf8"))
    m.frames[0].periphery = "0".repeat(64)
    const dir = tmp(), out = join(dir, "out"), manifest = join(dir, "m.json")
    writeFileSync(manifest, JSON.stringify(m))
    expect(await main(["--manifest", manifest, "--out", out, "--runs", writeRuns(fixtureLives().slice(0, 3))])).toBe(1)
    expect(readdirSync(dir)).toEqual(["m.json"])
  })

  it("main returns 1 and writes nothing when runProbe refuses", async () => {
    const dir = tmp(), out = join(dir, "out")
    const cpu = fixtureLives().slice(0, 2).map(l => ({ ...l, files: l.files.map(f => ({ ...f, widths: CPU }) as LogFileV2) }))
    expect(await main(["--runs", writeRuns(cpu), "--out", out])).toBe(1)
    expect(readdirSync(dir)).toEqual([])
  })

  it("main returns 2 and writes report.json (status 'identification failed') and summary.md when two fixture lives' snapshots are swapped", async () => {
    const out = join(tmp(), "out")
    expect(await main(["--runs", writeRuns(swapSnapshots(fixtureLives().slice(0, 3), 0, 2)), "--out", out])).toBe(2)
    expect(JSON.parse(readFileSync(join(out, "report.json"), "utf8")).status).toBe("identification failed")
    expect(readFileSync(join(out, "summary.md"), "utf8").split("\n")[0]).toMatch(/^\*\*Identification failed/)
  }, 60_000)

  it("the run writes only under --out", async () => {
    const status = () => execFileSync("git", ["status", "--porcelain", "--untracked-files=all"], { cwd: REPO, encoding: "utf8" })
    const before = status()
    const out = join(tmp(), "out")
    expect(await main(["--runs", writeRuns(fixtureLives().slice(0, 3)), "--out", out])).toBe(0)
    expect(status()).toBe(before)
    expect(readdirSync(out).sort()).toEqual(["report.json", "summary.md"])
  }, 60_000)

  it("writeFilterPngs writes one 4×4 sheet per state, and the same bytes twice", async () => {
    const life = probeLives(fixtureLives().slice(0, 1)).lives[0]!
    const a = tmp(), b = tmp()
    const pa = await writeFilterPngs(life, ["init", 512, 2048], a), pb = await writeFilterPngs(life, ["init", 512, 2048], b)
    expect(pa.map(p => p.slice(a.length + 1))).toEqual([`${life.learnerId}-init.png`, `${life.learnerId}-512.png`, `${life.learnerId}-2048.png`])
    pa.forEach((p, i) => expect(readFileSync(p).equals(readFileSync(pb[i]!))).toBe(true))
    const png = readFileSync(pa[0]!)
    expect(png.readUInt32BE(16)).toBe(4 * 40 + 5 * 2) // width: 4 cells of 5×8 px and 5 gaps
    expect(png.readUInt32BE(20)).toBe(4 * 40 + 5 * 2)
  })
})
