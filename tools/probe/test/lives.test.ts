import { describe, it, expect } from "vitest"
import { existsSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { parseLogFile, type LogFileV2 } from "@blanktab/harness/log"
import { DEFAULT_SETTINGS, FIRST_LAYER_PARAMS, GPU_WIDTHS, createLearner } from "@blanktab/learner"
import { createCpuBackend } from "@blanktab/tensor"
import { CPU, confirmMeta, lid } from "../../../packages/evaluators/test/fixtures/build.ts"
import { collectLives } from "../../eval/lives.ts"
import { decodeFilters, encodeFilters, initFirstLayer, layerArrays, probeLives, weightMotion } from "../filters/lives.ts"
import { FIXTURE_GENOMES, fixtureLife, fixtureSnapshot, mapLayer } from "./fixture-lives.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
const RUNS = join(REPO, "data/runs")
const G = FIXTURE_GENOMES[0]

describe("filter probe lives", () => {
  it("initFirstLayer(seed, GPU_WIDTHS) equals FIRST_LAYER_PARAMS of a newborn learner's readParams() before any step (createLearner on the CPU backend), bit for bit, for 3 seeds", async () => {
    for (const seed of [1, 2314181819, 4041665406]) {
      const learner = await createLearner({ backend: createCpuBackend(), seed, widths: GPU_WIDTHS, settings: DEFAULT_SETTINGS })
      const params = await learner.readParams()
      learner.dispose()
      const mine = layerArrays(initFirstLayer(seed, GPU_WIDTHS))
      FIRST_LAYER_PARAMS.forEach((p, k) => expect(Buffer.from(mine[k]!.buffer).equals(Buffer.from(params[p]!.buffer.slice(params[p]!.byteOffset, params[p]!.byteOffset + params[p]!.byteLength))), `seed ${seed} param ${p}`).toBe(true))
    }
  })

  it("decodeFilters of a real committed 2048 snapshot re-encodes (float32 LE, base64) to the same string, and has 16·104 values", () => {
    const { lives } = probeLives(collectLives(RUNS))
    expect(lives.length).toBeGreaterThan(0)
    const files = collectLives(RUNS).find(l => l.learnerId === lives[0]!.learnerId)!.files as LogFileV2[]
    const s = files.find(f => f.meta?.filters !== undefined && f.cu >= 2048)!.meta!.filters!
    const f = decodeFilters(s, 16)
    expect(layerArrays(f).reduce((a, x) => a + x.length, 0)).toBe(16 * 104)
    expect(encodeFilters(f)).toBe(s)
  })

  it("probeLives on fixture lives: a legacy life, a life without meta, a CPU-width life, a running life with no 2048 snapshot, and a confirm-role life are each excluded with their reason; an eligible life is kept", () => {
    const life = (k: number) => fixtureLife({ learnerId: lid(k), genomeSha: G, seed: 100 + k, noise: 0.01 })
    const legacy = life(1); legacy.files = legacy.files.map(f => ({ ...f, meta: { ...(f as LogFileV2).meta!, role: "legacy", method: "legacy" } }) as LogFileV2)
    const noMeta = life(2); noMeta.files = noMeta.files.map(f => { const { meta: _m, ...rest } = f as LogFileV2; void _m; return rest as LogFileV2 })
    const cpu = life(3); cpu.files = cpu.files.map(f => ({ ...f, widths: CPU }) as LogFileV2)
    const running = life(4); running.files = running.files.slice(0, 2)
    const confirm = fixtureLife({ learnerId: lid(5), genomeSha: G, seed: 105, noise: 0.01, meta: confirmMeta("centre", 0) })
    const ok = life(6)
    for (const f of ok.files) expect(parseLogFile(JSON.parse(JSON.stringify(f))).ok).toBe(true)
    const r = probeLives([ok, confirm, running, cpu, noMeta, legacy])
    expect(r.lives.map(l => l.learnerId)).toEqual([lid(6)])
    expect(r.excluded).toEqual([
      { learnerId: lid(1), reason: "role legacy" },
      { learnerId: lid(2), reason: "no meta" },
      { learnerId: lid(3), reason: "widths conv1 8" },
      { learnerId: lid(4), reason: "no snapshot at 2048" },
      { learnerId: lid(5), reason: "role confirm" },
    ])
    expect(r.lives[0]!.seed).toBe(106)
    expect(r.lives[0]!.genomeSha).toBe(G)
  })

  it("probeLives excludes the higher learnerId of two lives with the same settings.seed as 'seed shared with <lower id>', and keeps the lower", () => {
    const r = probeLives([fixtureLife({ learnerId: lid(9), genomeSha: G, seed: 7, noise: 0.01 }), fixtureLife({ learnerId: lid(8), genomeSha: G, seed: 7, noise: 0.02 })])
    expect(r.lives.map(l => l.learnerId)).toEqual([lid(8)])
    expect(r.excluded).toEqual([{ learnerId: lid(9), reason: `seed shared with ${lid(8)}` }])
  })

  it("probeLives excludes a life whose 1024 and 2048 snapshots are the same string as 'filters frozen after 1024', one whose 512 snapshot equals its recomputed init as 'filters frozen after init', and keeps a life whose snapshots differ by one ulp in one weight", () => {
    const spec = (k: number) => ({ learnerId: lid(k), genomeSha: G, seed: 200 + k, noise: 0.01 })
    const s1 = fixtureSnapshot(spec(1), 1024)
    const frozen1024 = fixtureLife({ ...spec(1), snapshots: { 2048: s1 } })
    const frozenInit = fixtureLife({ ...spec(2), snapshots: { 512: initFirstLayer(202, GPU_WIDTHS) } })
    const s3 = fixtureSnapshot(spec(3), 1024)
    const ulp = mapLayer(s3, (v, a, i) => (a === 0 && i === 0 ? new Float32Array([v * (1 + 2 ** -23)])[0]! : v))
    expect(ulp.periphery.w[0]).not.toBe(s3.periphery.w[0])
    const kept = fixtureLife({ ...spec(3), snapshots: { 1024: s3, 2048: ulp } })
    const r = probeLives([frozen1024, frozenInit, kept])
    expect(r.lives.map(l => l.learnerId)).toEqual([lid(3)])
    expect(r.excluded).toEqual([
      { learnerId: lid(1), reason: "filters frozen after 1024" },
      { learnerId: lid(2), reason: "filters frozen after init" },
    ])
  })

  it.skipIf(!existsSync(join(RUNS, "lrn_4c7ec98889ef9cbb")))("probeLives on the committed lives excludes lrn_4c7ec98889ef9cbb as 'filters frozen after 512'", () => {
    const r = probeLives(collectLives(RUNS))
    expect(r.excluded).toContainEqual({ learnerId: "lrn_4c7ec98889ef9cbb", reason: "filters frozen after 512" })
  })

  it("weightMotion is 0 at init and 1 for w = 2·w₀, b = 2·b₀", () => {
    const init = initFirstLayer(5, GPU_WIDTHS)
    for (const e of ["periphery", "fovea"] as const) {
      expect(weightMotion(init, init, e)).toBe(0)
      expect(weightMotion(mapLayer(init, v => 2 * v), init, e)).toBeCloseTo(1, 12)
    }
  })
})
