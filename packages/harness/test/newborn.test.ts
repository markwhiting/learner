import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { createLearner, type LearnerSettings, type Widths } from "@blanktab/learner"
import { hashSeed } from "@blanktab/world"
import { decodeGenome, genomeSha } from "@blanktab/meta"
import { NEWBORN_LEVELS, parseLogFile } from "../src/log"
import { openQueue } from "../src/queue"
import { buildHeldOut } from "../src/heldout"
import { confirmSeed, panelSeed } from "../src/checkup"
import { MILESTONES_V1 } from "../src/milestones"
import { BadMetaBlockError, type LearnerInit, type SessionLearner } from "../src/session"
import { NEWBORN_COUNT, NEWBORN_SEED_BASE, runNewborn, type NewbornConfig } from "../src/newborn"
import { spacesOf, validateSessionPr } from "../../../tools/gate/validate"
import { CENTRE, GENOME_SPACE } from "./helpers"

const ID = "lrn_00000000000a0005"
const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const canvas = () => createCanvas(WORLD.width, WORLD.height)

async function configFor(k: number | NewbornConfig["who"], over: Partial<NewbornConfig> = {}): Promise<NewbornConfig> {
  return {
    learnerId: ID, who: typeof k === "number" ? { kind: "reference", k } : k, codeSha: "dev", knowledgeSha: "f".repeat(64),
    hostSig: { cores: 8, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 1 }, widths: TINY,
    settings: decodeGenome(CENTRE, GENOME_SPACE).learner as unknown as LearnerSettings,
    genome: { schema: GENOME_SPACE.schemaVersion, values: { ...CENTRE }, sha: await genomeSha(GENOME_SPACE.schemaVersion, CENTRE) },
    space: GENOME_SPACE, ...over,
  }
}

describe("runNewborn (Plan 05a Task 9; spec §2.3)", () => {
  it("runNewborn on the CPU backend at tiny widths writes one file that parseLogFile accepts and validateSessionPr accepts; its checkups' variants are the panel's; the learner's weights are unchanged (readParams before = after)", { timeout: 300_000 }, async () => {
    const idb = new IDBFactory()
    const k = 21 // panel 1 + 21 mod 16 = 6
    const inits: LearnerInit[] = []
    const seeds: number[] = []
    let before: Float32Array[] | undefined
    let kept: Awaited<ReturnType<typeof createLearner>> | undefined
    let disposed = false
    const make = async (init: LearnerInit): Promise<SessionLearner> => {
      inits.push(init)
      if ("state" in init) throw new Error("a newborn is never resumed")
      const l = await createLearner({ backend: createCpuBackend(), ...init, heldOut: buildHeldOut(init.seed, canvas()) })
      before = (await l.readParams()).map(p => p.slice())
      kept = l
      return {
        step: o => l.step(o), stats: () => l.stats(), sleep: () => l.sleep(), save: () => l.save(),
        dispose: () => { disposed = true }, // the test reads the weights after, then disposes
        frozenRun: o => { seeds.push(o.seed); return l.frozenRun(o) },
      }
    }
    const f = await runNewborn(await configFor(k), { createLearner: make, checkupCanvas: canvas, idb, now: () => new Date("2026-10-23T12:00:00.000Z") })

    // The learner: seed 1000 + k, born untrained with the config's widths and settings.
    expect(inits).toEqual([{ seed: NEWBORN_SEED_BASE + k, widths: TINY, settings: (await configFor(k)).settings }])
    expect(disposed).toBe(true)
    const after = await kept!.readParams()
    expect(after.map(p => [...p])).toEqual(before!.map(p => [...p]))
    kept!.dispose()

    // One file, in the outbox: seq 0, final, cu 0, role and method newborn, panel 1 + (k mod 16).
    expect(await (await openQueue(idb)).files(ID)).toEqual([f])
    expect(f).toMatchObject({ seq: 0, final: true, cu: 0, stepsTotal: 0, schemaVersion: 2, widths: TINY })
    expect(f.meta).toMatchObject({ role: "newborn", method: "newborn", evalPanel: 6, drawSeed: null, posteriorSha: null, evalSeed: null })
    expect(f.settings.seed).toBe(NEWBORN_SEED_BASE + k)
    // Its held-out score at birth, and what it cost, billed to evalCU with the checkups'.
    expect(f.meta!.heldOut!.loss).toBeGreaterThan(0)
    expect(f.meta!.heldOut!.baseline).toBeGreaterThan(0)
    const checkupCU = f.checkups!.reduce((a, c) => a + c.evalCU, 0)
    expect(f.evalCU!).toBeGreaterThan(checkupCU)
    // Checkups at NEWBORN_LEVELS, each milestone on the panel's variant (the frozen run's policy seed derives from it).
    expect(f.checkups!.map(c => c.level)).toEqual([...NEWBORN_LEVELS])
    expect(seeds).toEqual(NEWBORN_LEVELS.flatMap(L => MILESTONES_V1.map(m => hashSeed(panelSeed(6, L, m.id), "policy"))))

    const p = parseLogFile(JSON.parse(JSON.stringify(f)))
    expect(p.ok, p.ok ? "" : p.error).toBe(true)
    const v = await validateSessionPr([{ path: `data/runs/${ID}/0.json`, content: JSON.stringify(f, null, 2) }], [], { spaces: spacesOf(GENOME_SPACE) })
    expect(v.errors).toEqual([])
    expect(v.ok).toBe(true)
  })

  it("runNewborn({ who: confirm }) writes one frozen file: role confirm, arm newborn, seed evalSeed, confirmSeed variants, evalPanel null; it parses and validates", { timeout: 300_000 }, async () => {
    const idb = new IDBFactory()
    // Confirmation 2's seed 12,006: j = 6, arm newborn (ARM_OF_J[6 mod 4]).
    const who = { kind: "confirm" as const, evalSeed: 12_006, confirmationId: 2, drawSeed: 4_242, posteriorSha: "cd".repeat(32) }
    const inits: LearnerInit[] = []
    const seeds: number[] = []
    const make = async (init: LearnerInit): Promise<SessionLearner> => {
      inits.push(init)
      if ("state" in init) throw new Error("a newborn is never resumed")
      const l = await createLearner({ backend: createCpuBackend(), ...init, heldOut: buildHeldOut(init.seed, canvas()) })
      return {
        step: o => l.step(o), stats: () => l.stats(), sleep: () => l.sleep(), save: () => l.save(), dispose: () => l.dispose(),
        frozenRun: o => { seeds.push(o.seed); return l.frozenRun(o) },
      }
    }
    const cfg = await configFor(who)
    const f = await runNewborn(cfg, { createLearner: make, checkupCanvas: canvas, idb, now: () => new Date("2026-10-24T12:00:00.000Z") })
    expect(inits).toEqual([{ seed: 12_006, widths: TINY, settings: cfg.settings }])
    expect(await (await openQueue(idb)).files(ID)).toEqual([f])
    expect(f).toMatchObject({ seq: 0, final: true, cu: 0, stepsTotal: 0, schemaVersion: 2, widths: TINY })
    expect(f.settings.seed).toBe(12_006)
    expect(f.meta).toMatchObject({
      role: "confirm", method: "confirm", evalPanel: null, evalSeed: 12_006, confirmation: { id: 2, arm: "newborn" },
      drawSeed: 4_242, posteriorSha: "cd".repeat(32), genome: CENTRE, genomeSha: cfg.genome.sha,
    })
    expect(f.meta!.heldOut!.loss).toBeGreaterThan(0)
    expect(f.checkups!.map(c => c.level)).toEqual([...NEWBORN_LEVELS])
    // Its variants are its seed's (confirmSeed), never a panel's.
    expect(seeds).toEqual(NEWBORN_LEVELS.flatMap(L => MILESTONES_V1.map(m => hashSeed(confirmSeed(12_006, L, m.id), "policy"))))
    const p = parseLogFile(JSON.parse(JSON.stringify(f)))
    expect(p.ok, p.ok ? "" : p.error).toBe(true)
    const v = await validateSessionPr([{ path: `data/runs/${ID}/0.json`, content: JSON.stringify(f, null, 2) }], [], { spaces: spacesOf(GENOME_SPACE) })
    expect(v.errors).toEqual([])
  })

  it("refuses a confirmation newborn whose seed is not of the newborn arm, or outside its confirmation's block, before making a learner", async () => {
    let made = 0
    const deps = { createLearner: async (): Promise<SessionLearner> => { made++; throw new Error("unreachable") }, checkupCanvas: canvas, idb: new IDBFactory(), now: () => new Date(0) }
    const who = { kind: "confirm" as const, evalSeed: 12_006, confirmationId: 2, drawSeed: 1, posteriorSha: "cd".repeat(32) }
    await expect(runNewborn(await configFor({ ...who, evalSeed: 12_004 }), deps)).rejects.toThrow(/arm must be candidate/)
    await expect(runNewborn(await configFor({ ...who, evalSeed: 12_005 }), deps)).rejects.toThrow(BadMetaBlockError)
    await expect(runNewborn(await configFor({ ...who, evalSeed: 13_006 }), deps)).rejects.toThrow(/block/)
    expect(made).toBe(0)
  })

  it("refuses a k outside 0…NEWBORN_COUNT−1 and a genome that is not what it would run, before making a learner", async () => {
    let made = 0
    const deps = { createLearner: async (): Promise<SessionLearner> => { made++; throw new Error("unreachable") }, checkupCanvas: canvas, idb: new IDBFactory(), now: () => new Date(0) }
    for (const k of [-1, NEWBORN_COUNT, 1.5]) await expect(runNewborn(await configFor(k), deps)).rejects.toThrow(RangeError)
    const cfg = await configFor(0)
    await expect(runNewborn({ ...cfg, genome: { ...cfg.genome, sha: "0".repeat(64) } }, deps)).rejects.toThrow(BadMetaBlockError)
    await expect(runNewborn({ ...cfg, settings: { ...cfg.settings, lr: cfg.settings.lr * 2 } }, deps)).rejects.toThrow(BadMetaBlockError)
    await expect(runNewborn({ ...cfg, genome: { ...cfg.genome, schema: cfg.genome.schema + 1 } }, deps)).rejects.toThrow(BadMetaBlockError)
    expect(made).toBe(0)
  })
})
