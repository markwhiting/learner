import { describe, it, expect } from "vitest"
import { IDBFactory, IDBObjectStore as FakeObjectStore } from "fake-indexeddb"
import { checkpointMetas, deleteCheckpoint, hasCheckpoint, loadCheckpoint, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { openQueue } from "../src/queue"
import { DB_VERSION, openDb } from "../src/db"
import { mkFile } from "./helpers"
import { emptyObservation, PERIPHERY, decodeMove, type Action, type Observation } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, createLearner, type LearnerState } from "@blanktab/learner"
import { CHECKPOINTS, result } from "../src/db"
import { packLearnerState, unpackLearnerState } from "../src/pack"
import { CAREGIVER_DEFAULTS, createCaregiver } from "../src/caregiver"
import type { MetaBlock } from "../src/log"

const A = "lrn_aaaaaaaaaaaaaaaa", B = "lrn_bbbbbbbbbbbbbbbb"
const cp = (learnerId: string, steps: number): Checkpoint<{ w: Float32Array }> => ({
  learnerId, savedAt: new Date(0).toISOString(), steps, cu: steps / 100, seq: 1, curveTail: [{ cu: 0.5, steps: 50, predLoss: 0.1, baselineLoss: 0.2 }], restores: 0, lostCU: 0, lostSkips: 0,
  learnerState: { w: Float32Array.of(1, 2, 3) },
})

describe("checkpoints", () => {
  it("codeSha is optional (older checkpoints have none) and, when present, 40 lowercase hex chars or \"dev\"", async () => {
    const idb = new IDBFactory()
    for (const ok of ["a".repeat(40), "dev", undefined]) {
      await saveCheckpoint(idb, { ...cp(A, 1), ...(ok !== undefined ? { codeSha: ok } : {}) })
      expect((await loadCheckpoint(idb, A))!.codeSha).toBe(ok)
    }
    await saveCheckpoint(idb, { ...cp(A, 1), codeSha: "nope" })
    await expect(loadCheckpoint(idb, A)).rejects.toThrow(/codeSha/)
  })

  it("world.room is optional, matches /^[a-z]{1,16}$/; 'Toys', '' and 1 are RangeError", async () => {
    const idb = new IDBFactory()
    await saveCheckpoint(idb, cp(A, 1))
    expect((await loadCheckpoint(idb, A))!.world).toBeUndefined()
    for (const room of ["toys", "a", "a".repeat(16)]) {
      await saveCheckpoint(idb, { ...cp(A, 1), world: { room } })
      expect((await loadCheckpoint(idb, A))!.world).toEqual({ room })
    }
    for (const world of [{ room: "Toys" }, { room: "" }, { room: 1 }, { room: "a".repeat(17) }, {}, { room: "toys", x: 1 }, "toys", null]) {
      await saveCheckpoint(idb, { ...cp(A, 1), world: world as unknown as { room: string } })
      await expect(loadCheckpoint(idb, A), JSON.stringify(world)).rejects.toThrow(RangeError)
    }
  })

  it("a checkpoint with meta loads; one with a malformed meta is a RangeError; a pre-05a checkpoint (no meta) loads", async () => {
    const idb = new IDBFactory()
    const meta: MetaBlock = {
      v: 1, role: "search", method: "centre", genomeSchema: 1, genome: { lr: 5e-4 }, genomeSha: "ab".repeat(32),
      drawSeed: null, posteriorSha: null, evalPanel: 7, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
    }
    await saveCheckpoint(idb, cp(A, 1))
    expect((await loadCheckpoint(idb, A))!.meta).toBeUndefined()
    await saveCheckpoint(idb, { ...cp(A, 1), meta })
    expect((await loadCheckpoint(idb, A))!.meta).toEqual(meta)
    const legacy: MetaBlock = { ...meta, role: "legacy", method: "legacy", evalPanel: null, genome: { entropyWeight: 0.001 }, missing: ["entropyWeight"] }
    await saveCheckpoint(idb, { ...cp(A, 1), meta: legacy })
    expect((await loadCheckpoint(idb, A))!.meta).toEqual(legacy)
    const bad: [string, unknown][] = [
      ["meta.method", { ...meta, method: "guess" }],
      ["meta.genomeSha", { ...meta, genomeSha: "x" }],
      ["unknown key", { ...meta, extra: 1 }],
      // A checkpoint never stores filters (they go only into the log file at their level) or an end (final files only).
      ["meta.filters", { ...meta, filters: "AAAA" }],
      ["meta.end", { ...meta, end: "budget" }],
      ["meta must be an object", "x"],
    ]
    for (const [want, m] of bad) {
      await saveCheckpoint(idb, { ...cp(A, 1), meta: m as MetaBlock })
      await expect(loadCheckpoint(idb, A), want).rejects.toThrow(RangeError)
      await expect(loadCheckpoint(idb, A), want).rejects.toThrow(want)
    }
  })

  it("hasCheckpoint says whether a learner has one, without reading it", async () => {
    const idb = new IDBFactory()
    expect(await hasCheckpoint(idb, A)).toBe(false)
    await saveCheckpoint(idb, cp(A, 100))
    expect(await hasCheckpoint(idb, A)).toBe(true)
    expect(await hasCheckpoint(idb, B)).toBe(false)
    await deleteCheckpoint(idb, A)
    expect(await hasCheckpoint(idb, A)).toBe(false)
  })
  it("keeps only the latest checkpoint per learnerId, structured-clone exact, until deleted", async () => {
    const idb = new IDBFactory()
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    await saveCheckpoint(idb, cp(A, 100))
    await saveCheckpoint(idb, cp(A, 250))
    await saveCheckpoint(idb, cp(B, 7))
    const a = await loadCheckpoint<{ w: Float32Array }>(idb, A)
    expect(a).toEqual(cp(A, 250))
    expect(a!.learnerState.w).toBeInstanceOf(Float32Array)
    expect((await loadCheckpoint(idb, B))?.steps).toBe(7)
    await deleteCheckpoint(idb, A)
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    expect((await loadCheckpoint(idb, B))?.steps).toBe(7)
    await deleteCheckpoint(idb, A) // deleting nothing is fine
  })

  it("a log and a checkpoint change are queued in one transaction: both happen or neither", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0), { save: cp(A, 500) })
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
    expect((await loadCheckpoint(idb, A))?.steps).toBe(500)
    // A duplicate seq aborts the transaction: the checkpoint is not replaced either.
    await expect(q.put(mkFile(A, 0, { stepsTotal: 1 }), { save: cp(A, 999) })).rejects.toThrow()
    expect((await loadCheckpoint(idb, A))?.steps).toBe(500)
    await q.put(mkFile(A, 1, { final: true }), { remove: A })
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    expect((await q.pending()).map(f => f.seq)).toEqual([0, 1])
  })

  it("a checkpoint write that throws at once (a value that cannot be stored) does not let the log commit alone", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    const put = FakeObjectStore.prototype.put
    FakeObjectStore.prototype.put = function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
      if (this.name === "checkpoints") throw new DOMException("the disk is full", "QuotaExceededError")
      return put.call(this, value, key)
    } as typeof put
    try {
      await expect(q.put(mkFile(A, 0), { save: cp(A, 500) })).rejects.toMatchObject({ name: "QuotaExceededError" })
    } finally {
      FakeObjectStore.prototype.put = put
    }
    expect(await q.pending()).toEqual([])
    await q.put(mkFile(A, 0), { save: cp(A, 500) }) // and the seq is still free
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
  })

  it("rejects a checkpoint for another learner than the log it travels with", async () => {
    const q = await openQueue(new IDBFactory())
    await expect(q.put(mkFile(A, 0), { save: cp(B, 1) })).rejects.toThrow(/learner/)
    await expect(q.put(mkFile(A, 0), { remove: B })).rejects.toThrow(/learner/)
    expect(await q.pending()).toEqual([])
  })

  it("rejects a malformed checkpoint record on load", async () => {
    const idb = new IDBFactory()
    for (const bad of [{ steps: -1 }, { seq: 1.5 }, { cu: Number.NaN }, { curveTail: null }, { savedAt: 3 }, { restores: -1 }, { restores: undefined }, { lostCU: Number.NaN }, { lostSkips: 0.5 }, { knowledgeSha: "nope" }, { sessionsSeenAtBirth: -2 }]) {
      await saveCheckpoint(idb, { ...cp(A, 1), ...bad } as unknown as Checkpoint)
      await expect(loadCheckpoint(idb, A)).rejects.toThrow(/checkpoint/)
    }
    // Birth facts are optional (checkpoints written before they were added) but checked when present.
    await saveCheckpoint(idb, { ...cp(A, 1), knowledgeSha: "a".repeat(64), sessionsSeenAtBirth: 2 })
    expect(await loadCheckpoint(idb, A)).toMatchObject({ knowledgeSha: "a".repeat(64), sessionsSeenAtBirth: 2 })
  })

  it("checkups and caregiver (Plan 04b) are optional, and checked when present: RangeError when malformed", async () => {
    const idb = new IDBFactory()
    // A tail needs the learner's evalCU (Plan 04c): the stub state carries stats.evalCU.
    const base = { ...cp(A, 5000), cu: 100, learnerState: { w: Float32Array.of(1), stats: { evalCU: 0.5 } } }
    const result = { level: 64, steps: 4000, passed: ["M1"], measures: { M1: 0.5, M2: null, M3: 1, M4: 0, M5: 1, M6: 0 }, evalCU: 0.1 }
    const caregiver = createCaregiver({ settings: CAREGIVER_DEFAULTS, seed: 3 })
    caregiver.observe("crib", 0.5)
    for (let i = 0; i < CAREGIVER_DEFAULTS.caregiverPeriod; i++) caregiver.decide("crib", ["crib", "toys"])
    const good = { ...base, checkups: { lastLevel: 64, tail: [result] }, caregiver: caregiver.state() }
    await saveCheckpoint(idb, good)
    expect(await loadCheckpoint(idb, A)).toMatchObject({ checkups: good.checkups, caregiver: good.caregiver })
    await saveCheckpoint(idb, { ...base, checkups: { lastLevel: 0, tail: [] } })
    expect((await loadCheckpoint(idb, A))!.checkups).toEqual({ lastLevel: 0, tail: [] })
    const cg = good.caregiver
    for (const bad of [
      { checkups: null }, { checkups: { lastLevel: 12, tail: [] } }, { checkups: { lastLevel: 128, tail: [] } }, // off the schedule; above cu
      { checkups: { lastLevel: 32, tail: [result] } }, { checkups: { lastLevel: 64, tail: [{ ...result, passed: ["M9"] }] } },
      { checkups: { lastLevel: 64, tail: [{ ...result, steps: 6000 }] } }, { checkups: { lastLevel: 64, tail: {} } }, { checkups: { lastLevel: 64, tail: [], extra: 1 } },
      { caregiver: null }, { caregiver: { ...cg, version: 2 } }, { caregiver: { ...cg, rng: -1 } }, { caregiver: { ...cg, rooms: { crib: { fast: Number.NaN, slow: 1 } } } },
      { caregiver: { ...cg, settings: { ...cg.settings, caregiverTau: "x" } } }, { caregiver: { ...cg, carries: 1.5 } }, { caregiver: { ...cg, safetyNet: 1 } },
      { caregiver: { ...cg, to: { crib: -1 } } }, { caregiver: { ...cg, extra: 1 } }, { caregiver: { ...cg, rooms: { crib: null } } },
      { caregiver: { ...cg, to: { Crib: cg.carries } } }, { caregiver: { ...cg, to: {}, carries: cg.carries + 1 } },
    ]) {
      await saveCheckpoint(idb, { ...base, ...bad } as unknown as Checkpoint)
      await expect(loadCheckpoint(idb, A), JSON.stringify(bad)).rejects.toThrow(RangeError)
    }
  })

  describe("checkups.suiteVersion and the tail's evalCU (Plan 04c)", () => {
    const base = { ...cp(A, 5000), cu: 100, learnerState: { w: Float32Array.of(1), stats: { evalCU: 0.5 } } }
    const result = { level: 64, steps: 4000, passed: ["M1"], measures: { M1: 0.5, M2: null, M3: 1, M4: 0, M5: 1, M6: 0 }, evalCU: 0.1 }
    it("a pre-04c checkpoint (no suiteVersion) loads as suite 1", async () => {
      const idb = new IDBFactory()
      await saveCheckpoint(idb, { ...base, checkups: { lastLevel: 64, tail: [result] } })
      const c = (await loadCheckpoint(idb, A))!
      expect(c.checkups).toEqual({ lastLevel: 64, tail: [result] })
      // Suite 1's ids are the rule: a 7th id is not in it.
      await saveCheckpoint(idb, { ...base, checkups: { lastLevel: 64, tail: [{ ...result, measures: { ...result.measures, M7: 1 } }] } })
      await expect(loadCheckpoint(idb, A)).rejects.toThrow(RangeError)
      await saveCheckpoint(idb, { ...base, checkups: { lastLevel: 64, tail: [result], suiteVersion: 1 } })
      expect((await loadCheckpoint(idb, A))!.checkups).toEqual({ lastLevel: 64, tail: [result], suiteVersion: 1 })
    })
    it("unknown suiteVersion is a RangeError", async () => {
      const idb = new IDBFactory()
      for (const suiteVersion of [2, 0, -1, 1.5, "1", null]) {
        await saveCheckpoint(idb, { ...base, checkups: { lastLevel: 64, tail: [], suiteVersion } } as unknown as Checkpoint)
        await expect(loadCheckpoint(idb, A), String(suiteVersion)).rejects.toThrow(RangeError)
      }
    })
    it("a non-empty tail without the learner's evalCU is a RangeError", async () => {
      const idb = new IDBFactory()
      for (const learnerState of [{ w: 1 }, { stats: {} }, { stats: { evalCU: -1 } }, { stats: { evalCU: Number.NaN } }, { stats: { evalCU: "1" } }, null]) {
        await saveCheckpoint(idb, { ...base, learnerState, checkups: { lastLevel: 64, tail: [result] } } as unknown as Checkpoint)
        await expect(loadCheckpoint(idb, A), JSON.stringify(learnerState)).rejects.toThrow("checkups.tail without the learner's evalCU")
        // An empty tail needs none (a stub, or an older state).
        await saveCheckpoint(idb, { ...base, learnerState, checkups: { lastLevel: 64, tail: [] } } as unknown as Checkpoint)
        await expect(loadCheckpoint(idb, A)).resolves.toBeDefined()
      }
    })
  })

  it("upgrades a v1 database (outbox only): queued logs survive and the checkpoint store appears", async () => {
    const idb = new IDBFactory()
    await new Promise<void>((res, rej) => {
      const req = idb.open("blanktab", 1)
      req.onupgradeneeded = () => req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
      req.onerror = () => rej(req.error)
      req.onsuccess = () => {
        const db = req.result
        const tx = db.transaction("outbox", "readwrite")
        tx.objectStore("outbox").add({ learnerId: A, seq: 0, sent: false, file: mkFile(A, 0) })
        tx.oncomplete = () => { db.close(); res() }
        tx.onerror = () => rej(tx.error)
      }
    })
    const q = await openQueue(idb)
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
    await saveCheckpoint(idb, cp(A, 3))
    expect((await loadCheckpoint(idb, A))?.steps).toBe(3)
  })
})

describe("checkpoint meta records", () => {
  it("are written and removed with their checkpoints, and built for checkpoints a v2 database already holds", async () => {
    const idb = new IDBFactory()
    await new Promise<void>((res, rej) => {
      const req = idb.open("blanktab", 2)
      req.onupgradeneeded = () => {
        req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
        req.result.createObjectStore("checkpoints", { keyPath: "learnerId" })
      }
      req.onerror = () => rej(req.error)
      req.onsuccess = () => {
        const db = req.result
        const tx = db.transaction("checkpoints", "readwrite")
        tx.objectStore("checkpoints").put({ ...cp(A, 7), learnerState: { version: 1, w: Float32Array.of(1) } })
        tx.oncomplete = () => { db.close(); res() }
        tx.onerror = () => rej(tx.error)
      }
    })
    expect(await checkpointMetas(idb)).toEqual([{ learnerId: A, savedAt: cp(A, 7).savedAt, version: 1, steps: 7, seq: 1 }])
    await saveCheckpoint(idb, cp(B, 3))
    expect((await checkpointMetas(idb)).map(m => [m.learnerId, m.steps])).toEqual([[A, 7], [B, 3]])
    await (await openQueue(idb)).put(mkFile(B, 1, { final: true }), { remove: B })
    await deleteCheckpoint(idb, A)
    expect(await checkpointMetas(idb)).toEqual([])
  })
})

describe("the database connection", () => {
  it("an upgrade blocked by an old tab's open connection fails after a timeout with a message the page can show", async () => {
    const idb = new IDBFactory()
    const old = await new Promise<IDBDatabase>((res, rej) => {
      const req = idb.open("blanktab", 1) // old code: no versionchange handler, never closes
      req.onupgradeneeded = () => req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
      req.onsuccess = () => res(req.result)
      req.onerror = () => rej(req.error)
    })
    await expect(openDb(idb, 50)).rejects.toThrow("Close other learner tabs to update")
    old.close() // the old tab goes away: the next attempt upgrades
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0))
    expect((await q.pending()).length).toBe(1)
  })

  /** A database as newer code would leave it: version DB_VERSION + 1, with `stores` (and maybe more). */
  const newer = (idb: IDBFactory, stores: string[]) => new Promise<void>((res, rej) => {
    const req = idb.open("blanktab", DB_VERSION + 1)
    req.onupgradeneeded = () => {
      for (const name of stores) req.result.createObjectStore(name, { keyPath: name === "outbox" ? ["learnerId", "seq"] : "learnerId" })
      req.result.createObjectStore("fromTheFuture", { keyPath: "id" })
    }
    req.onsuccess = () => { req.result.close(); res() }
    req.onerror = () => rej(req.error)
  })

  it("a database a newer page already upgraded (VersionError for this code's version): used as it is, if every store this code needs is there", async () => {
    const idb = new IDBFactory()
    await newer(idb, ["outbox", "checkpoints", "checkpointMeta", "contribution"])
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0), { save: cp(A, 1) })
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
    expect((await loadCheckpoint(idb, A))?.steps).toBe(1)
    const db = await openDb(idb)
    expect(db.version).toBe(DB_VERSION + 1)
  })

  it("a newer database missing a store this code needs: fails as before", async () => {
    const idb = new IDBFactory()
    await newer(idb, ["outbox", "checkpoints"])
    await expect(openDb(idb)).rejects.toThrow(/VersionError|version|store/i)
  })

  it("steps aside for a version change (here: the database being deleted) and reopens on the next call", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0))
    await saveCheckpoint(idb, cp(A, 1))
    await new Promise<void>((res, rej) => { const r = idb.deleteDatabase("blanktab"); r.onsuccess = () => res(); r.onerror = () => rej(r.error); r.onblocked = () => rej(new Error("blocked")) })
    await q.put(mkFile(A, 1)) // the same Queue object keeps working
    expect((await q.pending()).map(f => f.seq)).toEqual([1])
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    await saveCheckpoint(idb, cp(A, 2))
    expect((await loadCheckpoint(idb, A))?.steps).toBe(2)
  })
})

/** A tiny stream whose frames depend on the actions (a bright patch the cursor moves). */
function stream() {
  let x = 40, y = 25, t = 0
  return {
    observe(): Observation {
      const o = emptyObservation()
      o.periphery.fill(30 + (t++ % 5))
      o.periphery.fill(220, (y * PERIPHERY.width + x) * 3, (y * PERIPHERY.width + x) * 3 + 3)
      o.fovea.fill(x % 7 * 30)
      o.cursor = { x: x / (PERIPHERY.width - 1), y: y / (PERIPHERY.height - 1) }
      o.buttonHeld = t % 3 === 0
      return o
    },
    act(a: Action) {
      const { dx, dy } = decodeMove(a.move)
      x = Math.min(PERIPHERY.width - 1, Math.max(0, x + Math.sign(dx)))
      y = Math.min(PERIPHERY.height - 1, Math.max(0, y + Math.sign(dy)))
    },
  }
}
const TINY = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
const rawRecord = async (idb: IDBFactory, id: string) =>
  result((await openDb(idb)).transaction(CHECKPOINTS).objectStore(CHECKPOINTS).get(id) as IDBRequest<Checkpoint<Record<string, unknown>>>)
const learnerCp = (learnerState: LearnerState): Checkpoint => ({ ...cp(A, learnerState.stats.steps), seq: 0, learnerState })

describe("checkpoint frames (write volume)", { timeout: 60_000 }, () => {
  async function lived(steps: number) {
    const world = stream()
    const l = await createLearner({ backend: createCpuBackend(), seed: 3, widths: TINY, settings: { ...DEFAULT_SETTINGS, bufferSize: 100, replayPerSleep: 4 } })
    for (let i = 0; i < steps; i++) {
      world.act(await l.step(world.observe()))
      if (i % 25 === 24) await l.sleep()
    }
    return { l, world }
  }

  it("stores the replay frames gzip-compressed, once per distinct observation, and loads the state back exactly", async () => {
    const { l } = await lived(120)
    const state = await l.save()
    l.dispose()
    const idb = new IDBFactory()
    await saveCheckpoint(idb, learnerCp(state))
    const stored = (await rawRecord(idb, A))!.learnerState
    const frames = stored.frames as { codec: string; count: number; data: Uint8Array }
    expect(frames.codec).toBe("gzip")
    expect(frames.count).toBe(state.replay.length + 1) // the transitions share their frames: n + 1 observations
    expect(frames.data.byteLength).toBeLessThan((frames.count * (state.replay[0]!.obs.periphery.length + state.replay[0]!.obs.fovea.length)) / 10)
    const back = (await loadCheckpoint(idb, A))!.learnerState
    expect(back).toEqual(state)
    for (let i = 1; i < back.replay.length; i++) expect(back.replay[i]!.obs).toBe(back.replay[i - 1]!.next)
    expect(back.last!.obs).toBe(back.replay.at(-1)!.next)
  })

  it("packs the held-out set's frames with the replay's and restores it exactly", async () => {
    const world = stream()
    const heldOut = { obs: [world.observe()], actions: [] as Action[] }
    for (let i = 0; i < 8; i++) { const a = { move: (i * 5) % 25, button: 0, scroll: 0, key: 0 }; world.act(a); heldOut.actions.push(a); heldOut.obs.push(world.observe()) }
    const l = await createLearner({ backend: createCpuBackend(), seed: 3, widths: TINY, settings: { ...DEFAULT_SETTINGS, bufferSize: 100, replayPerSleep: 4 }, heldOut })
    for (let i = 0; i < 30; i++) world.act(await l.step(world.observe()))
    await l.sleep()
    const state = await l.save()
    l.dispose()
    expect(state.heldOut).not.toBeNull()
    const idb = new IDBFactory()
    await saveCheckpoint(idb, learnerCp(state))
    const stored = (await rawRecord(idb, A))!.learnerState
    expect((stored.frames as { count: number }).count).toBe(state.replay.length + 1 + heldOut.obs.length)
    expect(JSON.stringify(Object.keys(stored.heldOut as object))).toBe('["obs","actions"]')
    expect((stored.heldOut as { obs: unknown[] }).obs.every(i => Number.isInteger(i))).toBe(true) // indices, not frames
    const back = (await loadCheckpoint(idb, A))!.learnerState
    expect(back).toEqual(state)
    const again = await createLearner({ backend: createCpuBackend(), state: back as LearnerState })
    again.dispose()
  })

  it("packs a whole-life reservoir's frames with the replay's (each distinct observation once) and restores it exactly", async () => {
    const world = stream()
    const l = await createLearner({ backend: createCpuBackend(), seed: 3, widths: TINY, settings: { ...DEFAULT_SETTINGS, bufferSize: 40, replayPerSleep: 4, reservoirShare: 0.5 } })
    for (let i = 0; i < 120; i++) {
      world.act(await l.step(world.observe()))
      if (i % 25 === 24) await l.sleep()
    }
    const state = await l.save()
    l.dispose()
    expect(state.reservoir!.items.length).toBe(20)
    const distinct = new Set([...state.replay, ...state.reservoir!.items].flatMap(tr => [tr.obs, tr.next]))
    expect(distinct.size).toBeGreaterThan(state.replay.length + 1) // the reservoir holds frames the recent ones do not
    const idb = new IDBFactory()
    await saveCheckpoint(idb, learnerCp(state))
    const stored = (await rawRecord(idb, A))!.learnerState
    const frames = stored.frames as { codec: string; count: number }
    expect(frames.codec).toBe("gzip")
    expect(frames.count).toBe(distinct.size)
    const packed = stored.reservoir as { items: { obs: unknown; next: unknown }[]; seen: number }
    expect(packed.seen).toBe(state.reservoir!.seen)
    expect(packed.items.every(tr => Number.isInteger(tr.obs) && Number.isInteger(tr.next))).toBe(true) // indices, not frames
    const back = (await loadCheckpoint(idb, A))!.learnerState as LearnerState
    expect(back).toEqual(state)
    // What was one object before is one object again: a transition held as recent and in the reservoir, and its frames.
    const shared = state.reservoir!.items.findIndex(tr => state.replay.includes(tr))
    if (shared >= 0) expect(back.replay).toContain(back.reservoir!.items[shared])
    const again = await createLearner({ backend: createCpuBackend(), state: back })
    again.dispose()
  })

  it("a learner resumed from a stored (packed) checkpoint goes on bit-identically to one resumed from its unstored state", async () => {
    const { l, world } = await lived(60)
    const state = await l.save()
    l.dispose()
    const idb = new IDBFactory()
    await (await openQueue(idb)).put({ ...mkFile(A, 0) }, { save: learnerCp(state) }) // the log + checkpoint path packs too
    expect(((await rawRecord(idb, A))!.learnerState.frames as { codec: string }).codec).toBe("gzip")
    const fromDb = await createLearner({ backend: createCpuBackend(), state: (await loadCheckpoint(idb, A))!.learnerState })
    const direct = await createLearner({ backend: createCpuBackend(), state })
    for (let i = 0; i < 60; i++) {
      const o = world.observe()
      const [a, b] = [await fromDb.step(o), await direct.step(o)]
      expect(a).toEqual(b)
      world.act(a)
      if (i % 25 === 24) { await fromDb.sleep(); await direct.sleep() }
    }
    expect(await fromDb.save()).toEqual(await direct.save())
    fromDb.dispose()
    direct.dispose()
  })

  it("without CompressionStream the frames are stored raw, and still load exactly", async () => {
    const { l } = await lived(30)
    const state = await l.save()
    l.dispose()
    const saved = globalThis.CompressionStream
    try {
      Object.defineProperty(globalThis, "CompressionStream", { value: undefined, configurable: true, writable: true })
      const idb = new IDBFactory()
      await saveCheckpoint(idb, learnerCp(state))
      expect(((await rawRecord(idb, A))!.learnerState.frames as { codec: string }).codec).toBe("raw")
      expect((await loadCheckpoint(idb, A))!.learnerState).toEqual(state)
    } finally {
      Object.defineProperty(globalThis, "CompressionStream", { value: saved, configurable: true, writable: true })
    }
  })

  it("checkpoints written before packing (raw learner states) still load; anything that is not a learner state is stored as is", async () => {
    const { l } = await lived(10)
    const state = await l.save()
    l.dispose()
    expect(await unpackLearnerState(state)).toBe(state)
    for (const odd of [{}, null, { replay: [{ obs: 1 }], last: null }, { w: Float32Array.of(1) }]) expect(await packLearnerState(odd)).toBe(odd)
    expect(await unpackLearnerState(await packLearnerState(state))).toEqual(state)
  })

  it("a corrupt packed frame buffer is a RangeError on load", async () => {
    const { l } = await lived(30)
    const state = await l.save()
    l.dispose()
    const idb = new IDBFactory()
    await saveCheckpoint(idb, learnerCp(state))
    const rec = (await rawRecord(idb, A))!
    const frames = rec.learnerState.frames as { data: Uint8Array }
    const tx = (await openDb(idb)).transaction(CHECKPOINTS, "readwrite")
    tx.objectStore(CHECKPOINTS).put({ ...rec, learnerState: { ...rec.learnerState, frames: { ...frames, data: frames.data.slice(0, 20) } } })
    await new Promise(r => { tx.oncomplete = r })
    await expect(loadCheckpoint(idb, A)).rejects.toThrow(RangeError)
  })
})
