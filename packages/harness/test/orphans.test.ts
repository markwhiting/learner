import { describe, it, expect, vi } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { DEFAULT_SETTINGS, LEARNER_STATE_VERSION, clampSettings, type Widths } from "@blanktab/learner"
import { checkpointIds, readCheckpointRecord, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { CHECKPOINTS, CHECKPOINT_META, OUTBOX, openDb } from "../src/db"
import { CAREGIVER_DEFAULTS, createCaregiver } from "../src/caregiver"
import { parseLogFile, type LogFileV2 } from "../src/log"
import { spacesOf, validateSessionPr } from "../../../tools/gate/validate"
import { openQueue } from "../src/queue"
import { MAX_ORPHANS, ORPHAN_MAX_AGE_MS, learnerLockName, sweepCheckpoints, waitLock, webLock } from "../src/orphans"
import { adoptLearnerId, storedLearnerId } from "../src/identity"
import { GENOME_SPACE, birthBlock, mkFile, stubStorage } from "./helpers"

/** When set, finalLogOf throws for a final built from a checkpoint (the sweep's fallback is tested with it). */
const finalSpy = vi.hoisted(() => ({ throwWithCp: false }))
vi.mock("../src/final", async orig => {
  const real = await orig<typeof import("../src/final")>()
  return {
    ...real,
    finalLogOf: (o: Parameters<typeof real.finalLogOf>[0]) => {
      if (finalSpy.throwWithCp && o.cp) throw new Error("cannot build it from the checkpoint")
      return real.finalLogOf(o)
    },
  }
})

const NOW = new Date("2026-10-20T12:00:00.000Z")
const HOUR = 60 * 60 * 1000
const id = (n: number) => `lrn_${n.toString(16).padStart(16, "0")}`

const cp = (learnerId: string, ageMs: number, over: Partial<Checkpoint<unknown>> = {}): Checkpoint<unknown> => ({
  learnerId, savedAt: new Date(NOW.getTime() - ageMs).toISOString(), steps: 100, cu: 1, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0,
  learnerState: { version: LEARNER_STATE_VERSION }, ...over,
})

/**
 * Web Locks as the browser keeps them, for every tab of the origin: exclusive locks, ifAvailable, waiting requests
 * (with an abort signal) and query(). `hold(id)` is another tab living that learner.
 */
class FakeLocks {
  readonly held = new Set<string>()
  readonly waiting = new Map<string, (() => void)[]>()
  failRequests = false
  async request(name: string, options: LockOptions, cb: (lock: Lock | null) => unknown): Promise<unknown> {
    if (this.failRequests) throw new DOMException("no locks here", "SecurityError")
    if (this.held.has(name)) {
      if (options.ifAvailable) return cb(null)
      await new Promise<void>((res, rej) => {
        const q = this.waiting.get(name) ?? []
        this.waiting.set(name, q)
        q.push(res)
        options.signal?.addEventListener("abort", () => {
          q.splice(q.indexOf(res), 1)
          rej(options.signal!.reason)
        })
      })
    }
    this.held.add(name)
    try {
      return await cb({ name, mode: "exclusive" } as Lock)
    } finally {
      this.held.delete(name)
      this.waiting.get(name)?.shift()?.()
    }
  }
  async query(): Promise<LockManagerSnapshot> {
    return {
      held: [...this.held].map(name => ({ name, mode: "exclusive" as const })),
      pending: [...this.waiting].flatMap(([name, q]) => q.map(() => ({ name, mode: "exclusive" as const }))),
    }
  }
  /** Another tab takes the learner's lock for its life; resolves to its release. */
  hold(learnerId: string): Promise<() => void> {
    return new Promise(granted => void this.request(learnerLockName(learnerId), {}, () => new Promise<void>(r => granted(r))))
  }
  /** Locks still held, as learnerIds. */
  ids(): string[] {
    return [...this.held].map(n => n.replace("blanktab:learner:", "")).sort()
  }
}
const locks = async (heldByOthers: string[] = []) => {
  const l = new FakeLocks()
  for (const i of heldByOthers) await l.hold(i)
  return { l, tryLock: webLock(l) }
}

describe("orphaned checkpoints", () => {
  it("six closed tabs and more: the newest eligible orphan is adopted, at most 3 orphans remain, live ones untouched", async () => {
    const idb = new IDBFactory()
    // Six tabs closed 1..6 hours ago (tab 1 the most recent).
    for (let n = 1; n <= 6; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    await saveCheckpoint(idb, cp(id(7), ORPHAN_MAX_AGE_MS)) // closed 24 h ago: stale
    await saveCheckpoint(idb, cp(id(8), 10 * 60 * 1000, { steps: -1 })) // malformed
    await saveCheckpoint(idb, cp(id(9), 10 * 60 * 1000, { learnerState: { version: LEARNER_STATE_VERSION + 1 } })) // other code
    await saveCheckpoint(idb, cp(id(10), 10 * 60 * 1000)) // its final log is queued already
    await (await openQueue(idb)).put(mkFile(id(10), 0, { final: true }))
    // Alive in other tabs (their locks held), however old or odd: never touched.
    await saveCheckpoint(idb, cp(id(11), 5 * 60 * 1000))
    await saveCheckpoint(idb, cp(id(12), 30 * HOUR, { steps: -1 }))
    const { l, tryLock } = await locks([id(11), id(12)])
    // A checkpoint written by hand (no meta record): not one this code wrote, so not one to resume.
    const db = await openDb(idb)
    await new Promise(r => { const tx = db.transaction(CHECKPOINTS, "readwrite"); tx.objectStore(CHECKPOINTS).put(cp(id(13), HOUR / 2)); tx.oncomplete = r })

    const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })
    expect(r.adopted?.learnerId).toBe(id(1))
    expect(r.verdicts).toEqual({
      [id(1)]: "adopted", [id(2)]: "kept", [id(3)]: "kept", [id(4)]: "surplus", [id(5)]: "surplus", [id(6)]: "surplus",
      [id(7)]: "stale", [id(8)]: "invalid", [id(9)]: "version", [id(10)]: "final", [id(11)]: "locked", [id(12)]: "locked", [id(13)]: "invalid",
    })
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3), id(11), id(12)].sort())
    const orphans = (await checkpointIds(idb)).filter(i => i !== id(11) && i !== id(12))
    expect(orphans.length).toBeLessThanOrEqual(MAX_ORPHANS)
    // This tab holds the adopted learner's lock (for its life); every other lock it took is released.
    expect(l.ids()).toEqual([id(1), id(11), id(12)].sort())
    r.adopted!.release()
    await Promise.resolve()
    expect(l.ids()).toEqual([id(11), id(12)].sort())
  })

  it("without adopting (this tab has a learner) it still bounds the orphans, and skips this tab's own learner", async () => {
    const idb = new IDBFactory()
    for (let n = 1; n <= 5; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    await saveCheckpoint(idb, cp(id(20), 40 * HOUR)) // this tab's (a reload): its lock is about to be taken
    const { l, tryLock } = await locks()
    const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: false, exclude: [id(20)] })
    expect(r.adopted).toBeNull()
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3), id(20)].sort())
    await Promise.resolve()
    expect(l.ids()).toEqual([])
  })

  it("an orphan another tab adopts meanwhile is not adopted twice", async () => {
    const idb = new IDBFactory()
    await saveCheckpoint(idb, cp(id(1), HOUR))
    const { tryLock } = await locks()
    const [a, b] = await Promise.all([1, 2].map(() => sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })))
    expect([a!.adopted?.learnerId, b!.adopted?.learnerId].filter(Boolean)).toEqual([id(1)])
    expect(await checkpointIds(idb)).toEqual([id(1)])
  })

  it("nothing to adopt: null, and nothing deleted", async () => {
    const idb = new IDBFactory()
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true })
    expect(r).toEqual({ adopted: null, verdicts: {} })
  })

  it("an adopted learnerId becomes this tab's", () => {
    const s = stubStorage()
    expect(storedLearnerId(s)).toBeNull()
    adoptLearnerId(s, id(1))
    expect(storedLearnerId(s)).toBe(id(1))
    expect(() => adoptLearnerId(s, "nope")).toThrow(RangeError)
    s.setItem("blanktab:learnerId", "garbage")
    expect(storedLearnerId(s)).toBeNull()
  })

  describe("two tabs start at once: one reloading (its own learner X in its sessionStorage), one new", () => {
    const setupX = async () => {
      const idb = new IDBFactory()
      await saveCheckpoint(idb, cp(id(1), 60 * 1000)) // X: the newest checkpoint, its lock free during the reload
      await saveCheckpoint(idb, cp(id(2), 2 * HOUR)) // a real orphan
      return { idb, ...(await locks()) }
    }

    it("the reloading tab asks first: the new tab's sweep leaves X alone and adopts the real orphan", async () => {
      const { idb, l, tryLock } = await setupX()
      const own = waitLock(l, id(1), 1_000)
      const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })
      expect((await own)).not.toBeNull()
      expect(r.adopted?.learnerId).toBe(id(2))
      expect(r.verdicts[id(1)]).toBe("locked")
      expect(await readCheckpointRecord(idb, id(1))).toBeDefined()
    })

    it("the new tab's sweep takes X's lock first: the reloading tab waits, the sweep sees it waiting and lets X go", async () => {
      const { idb, l } = await setupX()
      let own: Promise<unknown> | undefined
      // The reloading tab asks for X right after the sweep took X's lock (the worst moment).
      const tryLock = async (learnerId: string) => {
        const held = await webLock(l)(learnerId)
        if (learnerId === id(1)) own = waitLock(l, id(1), 1_000)
        return held
      }
      const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })
      expect(r.verdicts[id(1)]).toBe("wanted")
      expect(r.adopted?.learnerId).toBe(id(2))
      expect(await own).not.toBeNull() // the reloading tab keeps its learner
      expect(await readCheckpointRecord(idb, id(1))).toBeDefined()
    })

    it("nor is X deleted as surplus or stale while its tab waits for it", async () => {
      const idb = new IDBFactory()
      await saveCheckpoint(idb, cp(id(1), 30 * HOUR)) // stale by age, but its tab is reloading right now
      const l = new FakeLocks()
      let own: Promise<unknown> | undefined
      const tryLock = async (learnerId: string) => {
        const held = await webLock(l)(learnerId)
        own ??= waitLock(l, id(1), 1_000)
        return held
      }
      const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: false })
      expect(r.verdicts[id(1)]).toBe("wanted")
      expect(await own).not.toBeNull()
      expect(await readCheckpointRecord(idb, id(1))).toBeDefined()
    })
  })

  it("a sweep slow on its adoption candidate's full read holds no other lock meanwhile: a reloading tab's wait succeeds", async () => {
    const idb = new IDBFactory()
    await saveCheckpoint(idb, cp(id(1), HOUR)) // the adoption candidate (newest)
    await saveCheckpoint(idb, cp(id(2), 2 * HOUR)) // X: a tab reloading right now
    await saveCheckpoint(idb, cp(id(3), 3 * HOUR))
    const l = new FakeLocks()
    let own: Promise<unknown> | undefined
    let heldDuringRead: string[] = []
    const readRecord = async (db: IDBFactory, learnerId: string) => {
      heldDuringRead = l.ids()
      own = waitLock(l, id(2), 200) // X's tab asks while the sweep is busy reading (a ~29 MB record, slowly)
      await new Promise(r => setTimeout(r, 500))
      return readCheckpointRecord(db, learnerId)
    }
    const r = await sweepCheckpoints({ idb, tryLock: webLock(l), now: () => NOW, adopt: true, readRecord })
    expect(heldDuringRead).toEqual([id(1)])
    expect(await own).not.toBeNull()
    expect(r.adopted?.learnerId).toBe(id(1))
    expect(await readCheckpointRecord(idb, id(2))).toBeDefined()
  })

  it("waitLock gives up after its timeout when another tab lives the learner (a duplicated tab)", async () => {
    const { l } = await locks([id(1)])
    expect(await waitLock(l, id(1), 50)).toBeNull()
    expect(l.waiting.get(learnerLockName(id(1)))).toEqual([])
  })

  it("a lock request the browser rejects is 'not acquired', not a hang", async () => {
    const l = new FakeLocks()
    l.failRequests = true
    expect(await webLock(l)(id(1))).toBeNull()
    expect(await waitLock(l, id(1), 10_000)).toBeNull()
  })

  it("the sweep judges checkpoints by their meta records, and reads a full record only to adopt it", async () => {
    const idb = new IDBFactory()
    for (let n = 1; n <= 5; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    const metas = await (async () => { const db = await openDb(idb); return new Promise<unknown[]>(r => { const q = db.transaction(CHECKPOINT_META).objectStore(CHECKPOINT_META).getAll(); q.onsuccess = () => r(q.result) }) })()
    expect(metas).toHaveLength(5)
    expect(metas[0]).toEqual({ learnerId: id(1), savedAt: cp(id(1), HOUR).savedAt, version: LEARNER_STATE_VERSION, steps: 100, seq: 0 })
    const reads: string[] = []
    const { IDBObjectStore: Store } = await import("fake-indexeddb")
    const get = Store.prototype.get
    Store.prototype.get = function (this: IDBObjectStore, key: IDBValidKey | IDBKeyRange) {
      if (this.name === CHECKPOINTS) reads.push(String(key))
      return get.call(this, key)
    } as typeof get
    try {
      await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: false })
      expect(reads).toEqual([])
      await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true })
      expect(reads).toEqual([id(1)])
    } finally {
      Store.prototype.get = get
    }
  })
})

describe("orphans deleted by the sweep end on the record with a final log (Plan 04c)", () => {
  const HOST = { cores: 2, deviceMemoryGb: 4, webgpu: true, tabsOnMachine: 2 }
  const FINAL = { hostSig: HOST, codeSha: "b".repeat(40), now: () => NOW }
  const W: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
  const SETTINGS = clampSettings(DEFAULT_SETTINGS)
  const K = "c".repeat(64)
  /** A learner state birthOf reads (seed, widths, settings) with the stats a final file takes. */
  const state = (stats: Record<string, number>) => ({ version: LEARNER_STATE_VERSION, seed: 7, widths: W, settings: SETTINGS, stats })
  /** A v2 log file as a session writes it: CU doubles with seq (seq 0 at 8 CU), 100 steps per CU. */
  const v2 = (learnerId: string, seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => {
    const cu = 8 * 2 ** seq
    return {
      schemaVersion: 2, learnerId, seq, codeSha: "a".repeat(40), knowledgeSha: K, writtenAt: new Date(NOW.getTime() - 30 * HOUR).toISOString(),
      stepsTotal: 100 * cu, settings: { ...SETTINGS, maxStepsPerSec: 20, seed: 7 }, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
      sessionsSeenAtBirth: 2, final: false, cu, condition: "A", widths: { ...W }, curve: [{ cu, steps: 100 * cu, predLoss: 0.02, baselineLoss: 0.025 }],
      skippedUpdates: 1, restores: 0, lostCU: 0, evalCU: 0.5, ...over,
    }
  }
  const put = async (idb: IDBFactory, files: LogFileV2[]) => { const q = await openQueue(idb); for (const f of files) await q.put(f) }
  const filesOf = async (idb: IDBFactory, learnerId: string) => (await (await openQueue(idb)).files(learnerId)) as LogFileV2[]
  /** Writer/validator agreement: every file parses, and the gate accepts the final one after the files before it. */
  const gateAccepts = async (files: LogFileV2[]) => {
    for (const f of files) {
      const p = parseLogFile(JSON.parse(JSON.stringify(f)))
      expect(p.ok, p.ok ? "" : p.error).toBe(true)
    }
    const last = files.at(-1)!
    const v = await validateSessionPr([{ path: `data/runs/${last.learnerId}/${last.seq}.json`, content: JSON.stringify(last, null, 2) }], files.slice(0, -1), { spaces: spacesOf(GENOME_SPACE) })
    expect(v.errors).toEqual([])
    expect(v.ok).toBe(true)
  }

  it("a stale orphan with log files gets a final file (seq latest+1, final true, counters >= the latest's) in the transaction that removes its checkpoint; the gate's validateSessionPr accepts it after the existing files", async () => {
    const idb = new IDBFactory()
    const L = id(1)
    await put(idb, [v2(L, 0), v2(L, 1)]) // 8 and 16 CU
    const care = createCaregiver({ settings: CAREGIVER_DEFAULTS, seed: 3 })
    care.noteLost(2) // carries a restore threw away since the last file: what the checkpoint knows
    const tailPt = { cu: 20, steps: 2000, predLoss: 0.019, baselineLoss: 0.025 }
    await saveCheckpoint(idb, cp(L, 30 * HOUR, {
      steps: 2000, cu: 20, seq: 2, curveTail: [tailPt], restores: 1, lostCU: 1.5, lostSkips: 3, knowledgeSha: K, sessionsSeenAtBirth: 2,
      codeSha: "d".repeat(40), caregiver: care.state(), checkups: { lastLevel: 16, tail: [], suiteVersion: 1 }, world: { room: "paint" },
      learnerState: state({ cu: 20, skippedUpdates: 4, evalCU: 0.75 }),
    }))
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })
    expect(r.verdicts).toEqual({ [L]: "stale" })
    expect(await checkpointIds(idb)).toEqual([])
    const files = await filesOf(idb, L)
    expect(files.map(f => [f.seq, f.final])).toEqual([[0, false], [1, false], [2, true]])
    const f = files[2]!
    expect(f).toMatchObject({
      schemaVersion: 2, learnerId: L, seq: 2, final: true, stepsTotal: 2000, cu: 20, curve: [tailPt], restores: 1, lostCU: 1.5,
      skippedUpdates: 3 + 4, evalCU: 0.75, codeSha: "d".repeat(40), knowledgeSha: K, sessionsSeenAtBirth: 2, hostSig: files[1]!.hostSig, writtenAt: NOW.toISOString(),
      widths: W, checkups: [], milestoneSuiteVersion: 1,
      // maxStepsPerSec from the latest file's settings; the caregiver's settings with the learner's.
      settings: { ...SETTINGS, maxStepsPerSec: 20, seed: 7, ...CAREGIVER_DEFAULTS },
      caregiver: { carries: 0, to: {}, safetyNet: false, lostCarries: 2 },
    })
    for (const k of ["stepsTotal", "cu", "restores", "lostCU", "skippedUpdates", "evalCU"] as const) expect(f[k]!).toBeGreaterThanOrEqual(files[1]![k]!)
    await gateAccepts(files)
  })

  it("the final file and the checkpoint's removal are one transaction: a log that cannot be queued leaves the checkpoint", async () => {
    const idb = new IDBFactory()
    const L = id(1)
    await put(idb, [v2(L, 0)])
    await saveCheckpoint(idb, cp(L, 30 * HOUR, { steps: 900, cu: 9, seq: 1, knowledgeSha: K, sessionsSeenAtBirth: 2, learnerState: state({ cu: 9, skippedUpdates: 1, evalCU: 0.5 }) }))
    const { IDBObjectStore: Store } = await import("fake-indexeddb")
    const add = Store.prototype.add
    Store.prototype.add = function (this: IDBObjectStore, ...a: Parameters<typeof add>) {
      if (this.name === OUTBOX) throw new DOMException("disk full", "QuotaExceededError")
      return add.apply(this, a)
    } as typeof add
    try {
      await expect(sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: false, final: FINAL })).rejects.toThrow("disk full")
    } finally {
      Store.prototype.add = add
    }
    expect(await checkpointIds(idb)).toEqual([L])
    expect((await filesOf(idb, L)).map(f => f.seq)).toEqual([0])
  })

  it("a version orphan (learner state version 1) gets a final file from its latest log", async () => {
    const idb = new IDBFactory()
    const L = id(1)
    const care = createCaregiver({ settings: CAREGIVER_DEFAULTS, seed: 3 }).state()
    await put(idb, [v2(L, 0), v2(L, 1, { caregiver: { carries: 2, to: { paint: 2 }, safetyNet: false }, milestoneSuiteVersion: 1, checkups: [] })])
    await saveCheckpoint(idb, cp(L, HOUR, { steps: 3000, cu: 30, seq: 2, caregiver: care, learnerState: { version: 1, seed: 7, widths: W, stats: { cu: 30 } } }))
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })
    expect(r).toEqual({ adopted: null, verdicts: { [L]: "version" } })
    expect(await checkpointIds(idb)).toEqual([])
    const files = await filesOf(idb, L)
    const { checkups: _c, caregiver: _g, ...counters } = files[1]!
    void _c
    void _g
    // Its counters, an empty curve, final: not its checkups or carries (on the record already).
    expect(files[2]).toEqual({ ...counters, seq: 2, final: true, curve: [], writtenAt: NOW.toISOString() })
    await gateAccepts(files)
  })

  it("a version orphan whose old state has full birth facts still gets its final from the latest log: its settings are not read (or re-clamped) by this code", async () => {
    const idb = new IDBFactory()
    const L = id(1)
    await put(idb, [v2(L, 0), v2(L, 1)])
    // Version 1 with everything birthOf reads, and a setting this code's clamp would change.
    const old = { version: 1, seed: 7, widths: W, settings: { ...SETTINGS, lr: 1e9 }, stats: { cu: 30, skippedUpdates: 0, evalCU: 0.5 } }
    expect(clampSettings(old.settings as typeof SETTINGS)).not.toEqual(old.settings)
    await saveCheckpoint(idb, cp(L, HOUR, { steps: 3000, cu: 30, seq: 2, knowledgeSha: K, sessionsSeenAtBirth: 2, codeSha: "d".repeat(40), learnerState: old }))
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })
    expect(r.verdicts).toEqual({ [L]: "version" })
    const files = await filesOf(idb, L)
    expect(files[2]).toEqual({ ...files[1]!, seq: 2, final: true, curve: [], writtenAt: NOW.toISOString() })
    await gateAccepts(files)
  })

  it("an orphan with a meta block gets meta.end orphan: from its checkpoint's block, or its latest file's without that file's filters", async () => {
    const meta = await birthBlock(id(1))
    const filters = btoa(String.fromCharCode(...new Uint8Array(new Float32Array(104 * W.conv1).buffer)))
    // Seq 0..6 at 8..512 CU; the 512 file carries the filters.
    // The files of a life that ran the block's genome (its settings decode from it: the gate checks).
    const care = createCaregiver({ settings: CAREGIVER_DEFAULTS, seed: 3 }).state()
    const settings = { ...SETTINGS, maxStepsPerSec: 30, seed: 7, ...CAREGIVER_DEFAULTS }
    const files7 = (L: string) => Array.from({ length: 7 }, (_, s) => v2(L, s, { settings, meta: { ...meta, ...(s === 6 ? { filters } : {}) } }))
    // A version orphan: from its latest file.
    const idb = new IDBFactory()
    const L = id(1)
    await put(idb, files7(L))
    await saveCheckpoint(idb, cp(L, HOUR, { steps: 60_000, cu: 600, seq: 7, learnerState: { version: 1, seed: 7, widths: W, stats: { cu: 600 } } }))
    expect((await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })).verdicts).toEqual({ [L]: "version" })
    const files = await filesOf(idb, L)
    expect(files[7]!.meta).toEqual({ ...meta, end: "orphan" })
    await gateAccepts(files)
    // A stale orphan: from its checkpoint's block.
    const idb2 = new IDBFactory()
    await put(idb2, files7(L))
    await saveCheckpoint(idb2, cp(L, 30 * HOUR, { steps: 60_000, cu: 600, seq: 7, knowledgeSha: K, sessionsSeenAtBirth: 2, meta, caregiver: care, learnerState: state({ cu: 600, skippedUpdates: 1, evalCU: 0.5 }) }))
    expect((await sweepCheckpoints({ idb: idb2, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })).verdicts).toEqual({ [L]: "stale" })
    const files2 = await filesOf(idb2, L)
    expect(files2[7]).toMatchObject({ final: true, stepsTotal: 60_000, meta: { ...meta, end: "orphan" } })
    await gateAccepts(files2)
  })

  it("a final that cannot be built from the checkpoint is built from the latest log file instead (said on console.warn)", async () => {
    const idb = new IDBFactory()
    const L = id(1)
    await put(idb, [v2(L, 0), v2(L, 1)])
    await saveCheckpoint(idb, cp(L, 30 * HOUR, { steps: 2000, cu: 20, seq: 2, knowledgeSha: K, sessionsSeenAtBirth: 2, learnerState: state({ cu: 20, skippedUpdates: 4, evalCU: 0.75 }) }))
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    finalSpy.throwWithCp = true
    try {
      const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: false, final: FINAL })
      expect(r.verdicts).toEqual({ [L]: "stale" })
      expect(warn).toHaveBeenCalledTimes(1)
      expect(String(warn.mock.calls[0]![0])).toContain("built from its latest log file")
    } finally {
      finalSpy.throwWithCp = false
      warn.mockRestore()
    }
    expect(await checkpointIds(idb)).toEqual([])
    const files = await filesOf(idb, L)
    expect(files[2]).toEqual({ ...files[1]!, seq: 2, final: true, curve: [], writtenAt: NOW.toISOString() })
    await gateAccepts(files)
  })

  it("a surplus orphan gets one; an adopted or kept one does not", async () => {
    const idb = new IDBFactory()
    for (let n = 1; n <= 5; n++) {
      await put(idb, [v2(id(n), 0)])
      await saveCheckpoint(idb, cp(id(n), n * HOUR, { steps: 900, cu: 9, seq: 1, knowledgeSha: K, sessionsSeenAtBirth: 2, learnerState: state({ cu: 9, skippedUpdates: 1, evalCU: 0.5 }) }))
    }
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })
    expect(r.verdicts).toEqual({ [id(1)]: "adopted", [id(2)]: "kept", [id(3)]: "kept", [id(4)]: "surplus", [id(5)]: "surplus" })
    for (let n = 1; n <= 5; n++) {
      const files = await filesOf(idb, id(n))
      expect(files.map(f => f.final), id(n)).toEqual(n >= 4 ? [false, true] : [false])
      if (n >= 4) {
        expect(files[1]).toMatchObject({ seq: 1, stepsTotal: 900, cu: 9 })
        await gateAccepts(files)
      }
    }
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3)])
  })

  it("an invalid orphan (no meta, unreadable record) with log files gets a final from the latest file; without log files, none", async () => {
    const idb = new IDBFactory()
    await put(idb, [v2(id(1), 0)])
    const db = await openDb(idb)
    // Written by hand: no meta record, and a record that is not a checkpoint.
    await new Promise(r => { const tx = db.transaction(CHECKPOINTS, "readwrite"); tx.objectStore(CHECKPOINTS).put({ learnerId: id(1), steps: -1 }); tx.objectStore(CHECKPOINTS).put({ learnerId: id(2) }); tx.oncomplete = r })
    // Malformed, found only when its full record is read for adoption.
    await put(idb, [v2(id(3), 0)])
    await saveCheckpoint(idb, cp(id(3), HOUR, { steps: -1 }))
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })
    expect(r.verdicts).toEqual({ [id(1)]: "invalid", [id(2)]: "invalid", [id(3)]: "invalid" })
    expect(await checkpointIds(idb)).toEqual([])
    for (const n of [1, 3]) {
      const files = await filesOf(idb, id(n))
      expect(files.map(f => ({ seq: f.seq, final: f.final, stepsTotal: f.stepsTotal, cu: f.cu }))).toEqual([
        { seq: 0, final: false, stepsTotal: 800, cu: 8 }, { seq: 1, final: true, stepsTotal: 800, cu: 8 },
      ])
      await gateAccepts(files)
    }
    expect(await filesOf(idb, id(2))).toEqual([])
  })

  it("an orphan whose latest file is final gets nothing more", async () => {
    const idb = new IDBFactory()
    await put(idb, [v2(id(1), 0), v2(id(1), 1, { final: true })])
    await saveCheckpoint(idb, cp(id(1), HOUR, { seq: 2 }))
    // Stale and final at once: no file after the final one either.
    await put(idb, [v2(id(2), 0, { final: true })])
    await saveCheckpoint(idb, cp(id(2), 30 * HOUR, { seq: 1 }))
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true, final: FINAL })
    expect(r.verdicts).toEqual({ [id(1)]: "final", [id(2)]: "stale" })
    expect(await checkpointIds(idb)).toEqual([])
    expect((await filesOf(idb, id(1))).map(f => f.seq)).toEqual([0, 1])
    expect((await filesOf(idb, id(2))).map(f => f.seq)).toEqual([0])
  })

  it("an orphan that never logged and took no step gets none; one that took steps gets one from its checkpoint", async () => {
    const idb = new IDBFactory()
    await saveCheckpoint(idb, cp(id(1), 30 * HOUR, { steps: 0, cu: 0, knowledgeSha: K, sessionsSeenAtBirth: 0, learnerState: state({ cu: 0, skippedUpdates: 0, evalCU: 0 }) }))
    await saveCheckpoint(idb, cp(id(2), 30 * HOUR, { steps: 500, cu: 5, knowledgeSha: K, sessionsSeenAtBirth: 0, learnerState: state({ cu: 5, skippedUpdates: 0, evalCU: 0 }) }))
    await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: false, final: FINAL })
    expect(await checkpointIds(idb)).toEqual([])
    expect(await filesOf(idb, id(1))).toEqual([])
    const files = await filesOf(idb, id(2))
    expect(files).toHaveLength(1)
    // No latest file: maxStepsPerSec is the default; the code is the checkpoint's, else this page's; the host this tab's.
    expect(files[0]).toMatchObject({ seq: 0, final: true, stepsTotal: 500, cu: 5, codeSha: FINAL.codeSha, hostSig: HOST, settings: { maxStepsPerSec: 30 } })
    await gateAccepts(files)
  })

  it("without the final option the sweep writes no log (as before)", async () => {
    const idb = new IDBFactory()
    await put(idb, [v2(id(1), 0)])
    await saveCheckpoint(idb, cp(id(1), 30 * HOUR, { steps: 900, cu: 9, seq: 1, knowledgeSha: K, sessionsSeenAtBirth: 2, learnerState: state({ cu: 9, skippedUpdates: 1, evalCU: 0.5 }) }))
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: false })
    expect(r.verdicts).toEqual({ [id(1)]: "stale" })
    expect(await checkpointIds(idb)).toEqual([])
    expect((await filesOf(idb, id(1))).map(f => f.seq)).toEqual([0])
  })
})
