import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { gunzipSync, gzipSync } from "node:zlib"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { hashSeed } from "@blanktab/detmath"
import { variantSeedOf, checkupSeed } from "@blanktab/harness/checkup"
import { MILESTONE_IDS, type LogFileV2 } from "@blanktab/harness/log"
import { ROOM_KINDS } from "@blanktab/world"
import { collectLives } from "../../eval/lives.ts"
import { serialise } from "../../gate/serialise.ts"
import { PROBE_SEEDS, PROBE_STEPS, buildManifest, encodeFrames, loadProbeSet, renderProbeSet, type ProbeManifest } from "../filters/probe-set.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
const frames = (): Buffer => readFileSync(join(REPO, "tools/probe/filter-probe-frames.bin.gz"))
// The committed frames were rendered on macOS, and the canvas (Skia) draws other pixels elsewhere (CI's Linux, 2026-11-16):
// a render is compared with them only where they were made. Everywhere, the committed frames are checked (loadProbeSet).
const MADE_HERE = process.platform === "darwin"
const manifest = (): ProbeManifest => JSON.parse(readFileSync(join(REPO, "tools/probe/filter-probe-set.json"), "utf8")) as ProbeManifest

describe("the filter probe set", () => {
  it("the committed manifest has 16 seeds 20000–20015, 16 steps each, 256 frames, and every start room four times", () => {
    const m = manifest()
    expect(m.seeds).toEqual(Array.from({ length: 16 }, (_, i) => 20_000 + i))
    expect(m.seeds).toEqual([...PROBE_SEEDS])
    expect(m.steps).toBe(16)
    expect(PROBE_STEPS).toBe(16)
    expect(m.frames).toHaveLength(256)
    for (const s of m.seeds) expect(m.frames.filter(f => f.seed === s).map(f => f.step)).toEqual(Array.from({ length: 16 }, (_, i) => i + 1))
    const rooms = m.seeds.map(s => ROOM_KINDS[s % 4])
    for (const r of ROOM_KINDS) expect(rooms.filter(x => x === r)).toHaveLength(4)
  })

  it("the committed frames load, and every frame matches its manifest sha (256 frames)", () => {
    expect(loadProbeSet(manifest(), frames())).toHaveLength(256)
  })

  it("a frames file with one byte changed is refused, naming the frame's seed and step", () => {
    const m = manifest()
    const raw = gunzipSync(frames())
    const at = 37 * (raw.length / 256) + 5
    raw[at] = raw[at]! ^ 1
    expect(() => loadProbeSet(m, gzipSync(raw))).toThrow(`seed ${m.frames[37]!.seed} step ${m.frames[37]!.step}`)
    expect(() => loadProbeSet(m, gzipSync(raw.subarray(1)))).toThrow(/bytes, not 256 frames/)
  })

  it.runIf(MADE_HERE)("on macOS, rendering reproduces the committed frames byte for byte", async () => {
    expect(encodeFrames(await renderProbeSet(manifest())).equals(frames())).toBe(true)
  })

  it.runIf(MADE_HERE)("a manifest with one frame sha changed makes renderProbeSet throw, naming the seed and step", async () => {
    const m = manifest()
    m.frames[37] = { ...m.frames[37]!, fovea: "0".repeat(64) }
    await expect(renderProbeSet(m)).rejects.toThrow(`seed ${m.frames[37]!.seed} step ${m.frames[37]!.step}`)
  })

  it("buildManifest run twice gives identical bytes", async () => {
    const a = serialise(await buildManifest()), b = serialise(await buildManifest())
    expect(a).toBe(b)
    if (MADE_HERE) expect(a).toBe(readFileSync(join(REPO, "tools/probe/filter-probe-set.json"), "utf8"))
  })

  it("no committed life has seen a probe world: for every life under data/runs and data/newborns, its training world seed (parseInt of learnerId hex 4–12, the page's seedsOf rule), hashSeed(learnerSeed, 'heldout'), and its checkup variant seeds lie outside 20000–20015", () => {
    const reserved = new Set(PROBE_SEEDS)
    let lives = 0, checked = 0
    for (const tree of ["data/runs", "data/newborns"]) {
      for (const life of collectLives(join(REPO, tree))) {
        lives++
        const id = life.learnerId
        const seeds: number[] = [parseInt(id.slice(4, 12), 16)]
        for (const f of life.files) {
          const s = f.settings.seed
          const learnerSeed = typeof s === "number" && Number.isInteger(s) ? s : parseInt(id.slice(12, 20), 16)
          seeds.push(hashSeed(learnerSeed, "heldout"))
          if (f.schemaVersion !== 2) continue
          const v2 = f as LogFileV2
          const variant = variantSeedOf(v2.meta) ?? ((l: number, m: string) => checkupSeed(id, l, m))
          for (const c of v2.checkups ?? []) for (const m of MILESTONE_IDS[v2.milestoneSuiteVersion ?? 1] ?? []) seeds.push(variant(c.level, m))
        }
        for (const s of seeds) expect(reserved.has(s), `${id}: world seed ${s}`).toBe(false)
        checked += seeds.length
      }
    }
    expect(lives).toBeGreaterThan(0)
    expect(checked).toBeGreaterThan(lives)
  })
})
