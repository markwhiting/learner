import { describe, it, expect, vi } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { openQueue, type Queue } from "../src/queue"
import { createContributor, type ContributorStatus } from "../src/contributor"
import { setToken } from "../src/token"
import { parseLogFile } from "../src/log"
import { GENOME_SPACE, stubStorage, mkFile } from "./helpers"
import { spacesOf, validateSessionPr } from "../../../tools/gate/validate"
import { fakeGithub } from "./fakeGithub"

const TOKEN = "ghp_SECRET_token_1234567890"
const ID = "lrn_0123456789abcdef"
const NOW = new Date(1_700_000_000_000)

async function setup(opts: { canPush?: boolean; withToken?: boolean; files?: number; existingForkAs?: string; onLocalOnly?: (id: string) => void; lock?: <T>(fn: () => Promise<T>) => Promise<T> } = {}) {
  const idb = new IDBFactory()
  const queue: Queue = await openQueue(idb)
  const n = opts.files ?? 6
  for (let s = 0; s < n; s++) await queue.put(mkFile(ID, s, { final: s === n - 1 }))
  const ls = stubStorage()
  if (opts.withToken !== false) setToken(ls, TOKEN)
  const gh = fakeGithub({ token: TOKEN, login: "ann", canPush: opts.canPush ?? true, ...(opts.existingForkAs ? { existingForkAs: opts.existingForkAs } : {}) })
  const statuses: ContributorStatus[] = []
  const scheduled: number[] = []
  const timers: Array<() => void> = []
  const c = createContributor({
    queue, ls, fetchFn: gh.fetchFn, now: () => NOW, onStatus: s => statuses.push(s),
    ...(opts.onLocalOnly ? { onLocalOnly: opts.onLocalOnly } : {}),
    ...(opts.lock ? { lock: opts.lock } : {}),
    sleep: async () => {},
    schedule: (fn, ms) => { scheduled.push(ms); timers.push(fn) },
  })
  return { idb, queue, ls, gh, c, statuses, scheduled, timers, last: () => statuses.at(-1)! }
}

const UP = "markwhiting/learner"
const BR = `run/${ID}`
const at = (r: { method: string; url: string }) => `${r.method} ${new URL(r.url).pathname}`
const writes = (reqs: { method: string; url: string }[]) => reqs.filter(r => r.method !== "GET").map(at)
const logPaths = (id: string, seqs: number[]) => seqs.map(s => `data/runs/${id}/${s}.json`)

describe("contributor: push access", () => {
  it("one commit per flush: 6 blobs, 1 tree on the branch head's tree, 1 commit, 1 ref move, 1 draft PR", async () => {
    const t = await setup()
    await t.c.flush()
    const reqs = t.gh.requests
    expect(writes(reqs)).toEqual([
      `POST /repos/${UP}/git/refs`,
      ...Array(6).fill(`POST /repos/${UP}/git/blobs`),
      `POST /repos/${UP}/git/trees`,
      `POST /repos/${UP}/git/commits`,
      `PATCH /repos/${UP}/git/refs/heads/${BR}`,
      `POST /repos/${UP}/pulls`,
    ])
    const body = (suffix: string) => JSON.parse(reqs.find(r => r.method !== "GET" && new URL(r.url).pathname.endsWith(suffix))!.body!)
    const commitOfMain = reqs.find(r => r.method === "GET" && r.url.endsWith(`/git/commits/${"a".repeat(40)}`))
    expect(commitOfMain).toBeTruthy()
    expect(body("/git/trees")).toMatchObject({ base_tree: expect.stringMatching(/^[0-9a-f]{40}$/) })
    expect(body("/git/trees").tree.map((e: { path: string }) => e.path)).toEqual(logPaths(ID, [0, 1, 2, 3, 4, 5]))
    expect(body("/git/commits").parents).toEqual(["a".repeat(40)])
    expect(body(`/git/refs/heads/${BR}`)).toMatchObject({ force: false })
    expect(reqs.some(r => r.method === "PUT" || r.url.includes("/contents/"))).toBe(false)
    expect(t.gh.paths(UP, BR)).toEqual(logPaths(ID, [0, 1, 2, 3, 4, 5]).sort())
    for (let s = 0; s < 6; s++) {
      const f = t.gh.file(UP, BR, `data/runs/${ID}/${s}.json`)
      expect(parseLogFile(f).ok).toBe(true)
      expect(f).toEqual(mkFile(ID, s, { final: s === 5 }))
    }
    expect(t.gh.pulls).toEqual([expect.objectContaining({ title: `run: ${ID}`, head: `markwhiting:${BR}`, base: "main", draft: true })])
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, [0, 1, 2, 3, 4, 5]).sort())
    expect(await t.queue.pending()).toEqual([])
    expect(t.last()).toMatchObject({ state: "idle", pending: 0 })
    expect(reqs.some(r => r.url.includes("/forks"))).toBe(false)
  })

  it("writes no heartbeat.json", async () => {
    const t = await setup()
    await t.c.flush()
    expect(t.gh.paths(UP, BR).some(p => p.includes("heartbeat"))).toBe(false)
    expect(t.gh.requests.some(r => (r.body ?? "").includes("heartbeat"))).toBe(false)
  })

  it("a second flush with nothing pending makes no requests", async () => {
    const t = await setup()
    await t.c.flush()
    const n = t.gh.requests.length
    await t.c.flush()
    expect(t.gh.requests.length).toBe(n)
  })

  it("reuses an existing branch and open PR: one more commit on top, fast-forward only", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    const head = t.gh.repos.get(UP)!.branches.get(BR)!
    await t.queue.put(mkFile(ID, 2))
    t.gh.requests.length = 0
    await t.c.flush()
    expect(writes(t.gh.requests)).toEqual([
      `POST /repos/${UP}/git/blobs`, `POST /repos/${UP}/git/trees`, `POST /repos/${UP}/git/commits`, `PATCH /repos/${UP}/git/refs/heads/${BR}`,
    ])
    const commit = JSON.parse(t.gh.requests.find(r => r.method === "POST" && r.url.endsWith("/git/commits"))!.body!)
    expect(commit.parents).toEqual([head])
    expect(t.gh.paths(UP, BR)).toEqual(logPaths(ID, [0, 1, 2]))
    expect(t.gh.pulls).toHaveLength(1)
    expect(await t.queue.pending()).toEqual([])
  })

  it("nothing is marked sent until the commit and the PR both exist; a retry completes without duplicating", async () => {
    const t = await setup({ files: 3 })
    t.gh.interceptors.push(r => (r.method === "POST" && r.url.endsWith("/pulls") ? new Response("{}", { status: 503 }) : undefined))
    await t.c.flush()
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([0, 1, 2])
    expect(t.gh.paths(UP, BR)).toEqual(logPaths(ID, [0, 1, 2])) // the commit landed
    expect(t.gh.pulls).toHaveLength(0)
    t.gh.interceptors.length = 0
    await t.c.flush()
    expect(await t.queue.pending()).toEqual([])
    expect(t.gh.pulls).toHaveLength(1)
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, [0, 1, 2]))
  })

  it("a failed commit marks nothing sent", async () => {
    const t = await setup({ files: 3 })
    t.gh.interceptors.push(r => (r.method === "POST" && r.url.endsWith("/git/commits") ? new Response("{}", { status: 503 }) : undefined))
    await t.c.flush()
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([0, 1, 2])
    expect(t.gh.pulls).toHaveLength(0)
  })

  it("a ref move that is not a fast-forward (another tab moved the branch) fails and retries; nothing marked", async () => {
    const t = await setup({ files: 1 })
    t.gh.interceptors.push(r => (r.method === "PATCH" ? new Response('{"message":"Update is not a fast forward"}', { status: 422 }) : undefined))
    await t.c.flush()
    expect((await t.queue.pending()).length).toBe(1)
    expect(t.last().state).toBe("error")
    expect(t.scheduled).toHaveLength(1)
  })

  it("files already on the open PR's branch (crash after commit) are marked sent without a new commit", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    // Simulate the crash: the same files are pending again.
    const idb2 = await openQueue(new IDBFactory())
    for (let s = 0; s < 2; s++) await idb2.put(mkFile(ID, s, { final: s === 1 }))
    const c2 = createContributor({ queue: idb2, ls: t.ls, fetchFn: t.gh.fetchFn, now: () => NOW, onStatus: () => {}, sleep: async () => {}, schedule: () => {} })
    t.gh.requests.length = 0
    await c2.flush()
    expect(t.gh.requests.some(r => r.method === "POST" && r.url.endsWith("/git/commits"))).toBe(false)
    expect(await idb2.pending()).toEqual([])
    expect(t.gh.pulls).toHaveLength(1)
  })

  it("groups several learners, one branch, commit and PR each", async () => {
    const t = await setup({ files: 2 })
    const other = "lrn_fedcba9876543210"
    await t.queue.put(mkFile(other, 0))
    await t.c.flush()
    expect(t.gh.pulls.map(p => p.head).sort()).toEqual([`markwhiting:run/${other}`, `markwhiting:${BR}`].sort())
    expect(t.gh.requests.filter(r => r.method === "POST" && r.url.endsWith("/git/commits"))).toHaveLength(2)
    expect(t.gh.paths(UP, `run/${other}`)).toEqual(logPaths(other, [0]))
    expect(await t.queue.pending()).toEqual([])
  })

  it("encodes the body as base64 of UTF-8 JSON", async () => {
    const t = await setup({ files: 1 })
    await t.queue.put(mkFile("lrn_aaaaaaaaaaaaaaaa", 0, { settings: { note: "héllo ✓" } }))
    await t.c.flush()
    expect(t.gh.file(UP, "run/lrn_aaaaaaaaaaaaaaaa", "data/runs/lrn_aaaaaaaaaaaaaaaa/0.json")).toMatchObject({ settings: { note: "héllo ✓" } })
  })
})

describe("contributor: fresh branch after merge (brief item 13)", () => {
  for (const deleteBranch of [false, true]) {
    for (const canPush of [true, false]) {
      it(`after a merge (${canPush ? "push" : "fork"}, branch ${deleteBranch ? "deleted" : "kept"}) the next PR proposes only the new files`, async () => {
        const t = await setup({ canPush, files: 2 })
        const repo = canPush ? UP : "ann/learner"
        await t.c.flush()
        t.gh.merge(t.gh.pulls[0]!, { deleteBranch })
        const main = t.gh.repos.get(UP)!.branches.get("main")!
        await t.queue.put(mkFile(ID, 2))
        await t.queue.put(mkFile(ID, 3, { final: true }))
        t.gh.requests.length = 0
        await t.c.flush()
        expect(t.gh.pulls).toHaveLength(2)
        expect(t.gh.pulls[1]).toMatchObject({ state: "open", draft: true })
        expect(t.gh.prFiles(t.gh.pulls[1]!)).toEqual(logPaths(ID, [2, 3]))
        if (!deleteBranch) {
          const reset = t.gh.requests.find(r => r.method === "PATCH" && JSON.parse(r.body!).force === true)
          expect(reset && new URL(reset.url).pathname).toBe(`/repos/${repo}/git/refs/heads/${BR}`)
          expect(JSON.parse(reset!.body!).sha).toBe(main)
        }
        expect(await t.queue.pending()).toEqual([])
      })
    }
  }

  it("once a life's final file is sent, the browser keeps only that file of it", async () => {
    const t = await setup({ files: 3 })
    await t.c.flush()
    expect(await t.queue.pending()).toEqual([])
    expect((await t.queue.files(ID)).map(f => [f.seq, f.final])).toEqual([[2, true]])
  })

  it("files that are already on main (PR merged before they were marked sent) are marked sent and no PR is opened", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    t.gh.merge(t.gh.pulls[0]!, { deleteBranch: false })
    const q2 = await openQueue(new IDBFactory())
    for (let s = 0; s < 2; s++) await q2.put(mkFile(ID, s, { final: s === 1 }))
    const c2 = createContributor({ queue: q2, ls: t.ls, fetchFn: t.gh.fetchFn, now: () => NOW, onStatus: () => {}, sleep: async () => {}, schedule: () => {} })
    await c2.flush()
    expect(await q2.pending()).toEqual([])
    expect(t.gh.pulls).toHaveLength(1)
  })
})

describe("contributor: the gate merges on a schedule (Review Focus 3)", () => {
  type T = Awaited<ReturnType<typeof setup>>
  const feed = async (t: T, seqs: number[]) => { for (const s of seqs) await t.queue.put(mkFile(ID, s)) }
  /** What the gate does: validate the head, then `gh pr merge --match-head-commit <that head>`. Returns the files that PR carried. */
  const gateMerge = (t: T, i: number, deleteBranch = true): string[] => {
    const pr = t.gh.pulls[i]!
    const carried = t.gh.prFiles(pr)
    t.gh.merge(pr, { deleteBranch, matchHeadCommit: t.gh.headOf(pr)! })
    return carried
  }

  for (const canPush of [true, false]) {
    it(`${canPush ? "push" : "fork"}: merge (branch deleted), more logs, flush: a new PR; every file on exactly one PR, three cycles`, async () => {
      const t = await setup({ canPush, files: 0 })
      const carried: string[] = []
      let next = 0
      for (let cycle = 0; cycle < 3; cycle++) {
        await feed(t, [next, next + 1])
        next += 2
        await t.c.flush()
        expect(t.gh.pulls).toHaveLength(cycle + 1)
        expect(t.gh.pulls[cycle]).toMatchObject({ state: "open", draft: true })
        carried.push(...gateMerge(t, cycle))
        expect(t.gh.repos.get(canPush ? UP : "ann/learner")!.branches.has(BR)).toBe(false)
      }
      expect(carried.sort()).toEqual(logPaths(ID, [0, 1, 2, 3, 4, 5]))
      expect(t.gh.paths(UP, "main")).toEqual(logPaths(ID, [0, 1, 2, 3, 4, 5]))
      expect(await t.queue.pending()).toEqual([])
    })
  }

  it("race: a tab commits after the gate validated; the pinned merge fails, the PR keeps every file, next run merges it", async () => {
    const t = await setup({ files: 0 })
    await feed(t, [0, 1])
    await t.c.flush()
    const validated = t.gh.headOf(t.gh.pulls[0]!)!
    await feed(t, [2])
    await t.c.flush() // the tab pushes while the gate is between validating and merging
    expect(() => t.gh.merge(t.gh.pulls[0]!, { deleteBranch: true, matchHeadCommit: validated })).toThrow(/Head branch was modified/)
    expect(t.gh.pulls[0]!.state).toBe("open")
    expect(t.gh.paths(UP, "main")).toEqual([])
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, [0, 1, 2]))
    const first = gateMerge(t, 0) // next run validates the new head and merges
    expect(t.gh.paths(UP, "main")).toEqual(logPaths(ID, [0, 1, 2]))
    await feed(t, [3])
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(2)
    expect([...first, ...t.gh.prFiles(t.gh.pulls[1]!)].sort()).toEqual(logPaths(ID, [0, 1, 2, 3]))
  })

  for (const deleteBranch of [true, false]) {
    it(`race: the gate merges (branch ${deleteBranch ? "deleted" : "kept"}) while the tab is committing; no file is lost or duplicated`, async () => {
      const t = await setup({ files: 0 })
      await feed(t, [0, 1])
      await t.c.flush()
      await feed(t, [2, 3])
      // The tab has seen an open PR and is about to commit; the gate merges at exactly that moment.
      const carried: string[] = []
      let fired = false
      t.gh.interceptors.push(r => {
        if (!fired && r.method === "POST" && new URL(r.url).pathname.endsWith("/git/blobs")) {
          fired = true
          carried.push(...gateMerge(t, 0, deleteBranch))
        }
        return undefined
      })
      await t.c.flush()
      // Whatever the tab did, the files it still owes reach a PR on later flushes (retry timers included).
      for (let i = 0; i < 3; i++) { for (const fn of t.timers.splice(0)) fn(); await t.c.flush() }
      expect(await t.queue.pending()).toEqual([])
      for (let i = 1; i < t.gh.pulls.length; i++) if (t.gh.pulls[i]!.state === "open") carried.push(...gateMerge(t, i, deleteBranch))
      expect(fired).toBe(true)
      expect(t.gh.paths(UP, "main")).toEqual(logPaths(ID, [0, 1, 2, 3]))
      expect(carried.sort()).toEqual(logPaths(ID, [0, 1, 2, 3]))
    })
  }
})

describe("contributor: the gate merges right after the push (between the ref move and the check by number)", () => {
  for (const deleteBranch of [true, false]) {
    it(`branch ${deleteBranch ? "deleted" : "kept"}: the merged PR carried the files; the next flush finds them on main and sends nothing twice`, async () => {
      const t = await setup({ files: 0 })
      for (const s of [0, 1]) await t.queue.put(mkFile(ID, s))
      await t.c.flush()
      for (const s of [2, 3]) await t.queue.put(mkFile(ID, s))
      let fired = false
      t.gh.interceptors.push(r => {
        if (!fired && r.method === "GET" && new URL(r.url).pathname === `/repos/${UP}/pulls/1`) {
          fired = true
          const pr = t.gh.pulls[0]!
          t.gh.merge(pr, { deleteBranch, matchHeadCommit: t.gh.headOf(pr)! })
        }
        return undefined
      })
      await t.c.flush()
      expect(fired).toBe(true)
      expect((await t.queue.pending()).map(f => f.seq)).toEqual([2, 3]) // not marked: the check saw the PR closed
      for (const fn of t.timers.splice(0)) fn()
      await t.c.flush()
      expect(await t.queue.pending()).toEqual([])
      expect(t.gh.pulls).toHaveLength(1) // no second PR: the files were already on main
      expect(t.gh.paths(UP, "main")).toEqual(logPaths(ID, [0, 1, 2, 3]))
    })
  }
})

describe("contributor: the PR file cap", () => {
  it("20 pending files: the first PR gets 16, the other 4 wait for the merge and then go in a fresh PR", async () => {
    const t = await setup({ files: 20 })
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(1)
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, Array.from({ length: 16 }, (_, i) => i)).sort())
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([16, 17, 18, 19])
    // A full PR takes nothing more: no commit, nothing sent.
    t.gh.requests.length = 0
    await t.c.flush()
    expect(writes(t.gh.requests)).toEqual([])
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toHaveLength(16)
    expect(await t.queue.pending()).toHaveLength(4)
    t.gh.merge(t.gh.pulls[0]!, { deleteBranch: false })
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(2)
    expect(t.gh.prFiles(t.gh.pulls[1]!)).toEqual(logPaths(ID, [16, 17, 18, 19]).sort())
    expect(await t.queue.pending()).toEqual([])
  })

  it("an open PR holding N files takes only 16 - N more, oldest first", async () => {
    const t = await setup({ files: 10 })
    await t.c.flush()
    for (let s = 10; s < 20; s++) await t.queue.put(mkFile(ID, s))
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(1)
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toHaveLength(16)
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([16, 17, 18, 19])
  })
})

describe("contributor: the PR file cap fails closed", () => {
  it("the count comes from the branch head: a lagging PR.changed_files cannot push a PR past 16", async () => {
    const t = await setup({ files: 10 })
    await t.c.flush()
    t.gh.interceptors.push(r => {
      const m = new URL(r.url).pathname.match(/\/pulls\/(\d+)$/)
      const live = { number: 1, state: "open", merged_at: null, head: { sha: t.gh.headOf(t.gh.pulls[0]!) } }
      return r.method === "GET" && m ? new Response(JSON.stringify({ ...live, changed_files: 0 }), { status: 200 }) : undefined
    })
    for (let s = 10; s < 20; s++) await t.queue.put(mkFile(ID, s))
    await t.c.flush()
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toHaveLength(16)
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([16, 17, 18, 19])
  })

  for (const [name, res] of [
    ["has no files field", () => new Response(JSON.stringify({}), { status: 200 })],
    ["files is not a list", () => new Response(JSON.stringify({ files: "many" }), { status: 200 })],
    ["fails with a 500", () => new Response("{}", { status: 500 })],
  ] as const) {
    it(`an unreadable count (compare ${name}) commits nothing that flush, and the next flush recovers`, async () => {
      const t = await setup({ files: 2 })
      await t.c.flush()
      await t.queue.put(mkFile(ID, 2))
      let broken = true
      t.gh.interceptors.push(r => (broken && r.method === "GET" && new URL(r.url).pathname.includes("/compare/") ? res() : undefined))
      t.gh.requests.length = 0
      await t.c.flush()
      expect(writes(t.gh.requests)).toEqual([])
      expect((await t.queue.pending()).map(f => f.seq)).toEqual([2])
      broken = false
      await t.c.flush()
      expect(t.gh.prFiles(t.gh.pulls[0]!)).toHaveLength(3)
      expect(await t.queue.pending()).toEqual([])
    })
  }
})

describe("contributor: no push access", () => {
  it("creates a fork first, polls until ready, writes to the fork, PR goes to upstream", async () => {
    const t = await setup({ canPush: false, files: 2 })
    await t.c.flush()
    const reqs = t.gh.requests.map(r => `${r.method} ${new URL(r.url).pathname}`)
    const forkAt = reqs.indexOf("POST /repos/markwhiting/learner/forks")
    expect(forkAt).toBeGreaterThan(-1)
    expect(reqs.indexOf("POST /repos/ann/learner/git/blobs")).toBeGreaterThan(forkAt)
    expect(reqs.filter(r => r === "GET /repos/ann/learner").length).toBeGreaterThan(2)
    expect(t.gh.file("ann/learner", `run/${ID}`, `data/runs/${ID}/1.json`)).toBeTruthy()
    expect(t.gh.pulls[0]).toMatchObject({ head: `ann:run/${ID}`, base: "main", draft: true })
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, [0, 1]))
    // upstream is only read and asked for a PR (and the fork): no git writes there
    expect(reqs.filter(r => r.startsWith("POST /repos/markwhiting/learner/git") || r.startsWith("PATCH /repos/markwhiting"))).toEqual([])
    expect(await t.queue.pending()).toEqual([])
  })

  it("uses the full_name an existing differently named fork, without creating another", async () => {
    const t = await setup({ canPush: false, files: 1, existingForkAs: "ann/learner-1" })
    await t.c.flush()
    expect(t.gh.file("ann/learner-1", `run/${ID}`, `data/runs/${ID}/0.json`)).toBeTruthy()
    expect(t.gh.pulls[0]!.head).toBe(`ann:run/${ID}`)
  })

  it("does not create a second fork when one already exists", async () => {
    const t = await setup({ canPush: false, files: 1, existingForkAs: "ann/learner" })
    await t.c.flush()
    expect(t.gh.requests.some(r => r.url.endsWith("/forks"))).toBe(false)
  })
})

describe("contributor: GitHub failures (Review Focus 4)", () => {
  it("401 leaves the queue intact, shows the rejection message, and does not auto-retry", async () => {
    const t = await setup()
    setToken(t.ls, "ghp_bad")
    await t.c.flush()
    expect((await t.queue.pending()).length).toBe(6)
    expect(t.last()).toEqual({ state: "error", message: "Your GitHub token was rejected — paste a new one.", pending: 6 })
    expect(t.scheduled).toEqual([])
    expect(t.gh.requests.some(r => r.method !== "GET")).toBe(false)
  })

  it("401 midway keeps every file pending (a learner's files are sent as one commit)", async () => {
    const t = await setup()
    let blobs = 0
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") && ++blobs === 3 ? new Response('{"message":"Bad credentials"}', { status: 401 }) : undefined))
    await t.c.flush()
    expect(t.last().message).toMatch(/rejected/)
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([0, 1, 2, 3, 4, 5])
  })

  it("403 with Retry-After: 30 schedules a retry of at least 30 s, queue intact", async () => {
    const t = await setup()
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response('{"message":"rate"}', { status: 403, headers: { "Retry-After": "30" } }) : undefined))
    await t.c.flush()
    expect(t.scheduled).toHaveLength(1)
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(30_000)
    expect(t.last().state).toBe("error")
    expect((await t.queue.pending()).length).toBe(6)
    // the scheduled retry works once the limit clears
    t.gh.interceptors.length = 0
    t.timers[0]!()
    await vi.waitFor(async () => expect(await t.queue.pending()).toEqual([]))
  })

  it("429 honours X-RateLimit-Reset", async () => {
    const t = await setup()
    const reset = Math.floor(NOW.getTime() / 1000) + 120
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response("{}", { status: 429, headers: { "X-RateLimit-Reset": String(reset), "X-RateLimit-Remaining": "0" } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(120_000)
  })

  it("offline (fetch throws) retries with exponential backoff from 5 s, capped at 10 min", async () => {
    const t = await setup()
    t.gh.interceptors.push(() => { throw new TypeError("Failed to fetch") })
    await t.c.flush()
    for (let i = 0; i < 9; i++) {
      t.timers.shift()!()
      await vi.waitFor(() => expect(t.timers.length).toBeGreaterThan(0)) // the retry has failed and re-armed
    }
    expect(t.scheduled).toEqual([5000, 10000, 20000, 40000, 80000, 160000, 320000, 600000, 600000, 600000])
    expect((await t.queue.pending()).length).toBe(6)
    expect(t.last()).toMatchObject({ state: "error", pending: 6 })
    t.gh.interceptors.length = 0
    t.timers.shift()!()
    await vi.waitFor(() => expect(t.last().state).toBe("idle"))
    expect(await t.queue.pending()).toEqual([])
  })

  it("a flush asked for mid-flight runs again afterwards, so a file queued meanwhile is sent", async () => {
    const t = await setup({ files: 1 })
    let again: Promise<void> | undefined
    t.gh.interceptors.push(r => {
      if (again === undefined && r.url.endsWith("/git/blobs")) {
        void t.queue.put(mkFile(ID, 1, { final: true }))
        again = t.c.flush()
      }
      return undefined
    })
    await t.c.flush()
    await again
    expect(t.gh.paths(UP, BR)).toEqual(logPaths(ID, [0, 1]))
    expect(t.gh.requests.filter(r => r.url.endsWith("/git/commits") && r.method === "POST")).toHaveLength(2)
    expect(await t.queue.pending()).toEqual([])
  })

  it("a follow-up flush still runs when the run in flight fails unexpectedly", async () => {
    const t = await setup({ files: 1 })
    const pending = t.queue.pending.bind(t.queue)
    let calls = 0
    t.queue.pending = async () => { if (++calls === 1) { await Promise.resolve(); void t.c.flush().catch(() => {}); throw new Error("idb broke") } return pending() }
    await expect(t.c.flush()).rejects.toThrow("idb broke")
    await vi.waitFor(async () => expect(await t.queue.pending()).toEqual([]))
    expect(t.gh.requests.filter(r => r.url.endsWith("/git/commits") && r.method === "POST")).toHaveLength(1)
  })

  it("concurrent flushes do not double-send", async () => {
    const t = await setup()
    await Promise.all([t.c.flush(), t.c.flush()])
    expect(t.gh.requests.filter(r => r.url.endsWith("/git/commits") && r.method === "POST")).toHaveLength(1)
    expect(t.gh.pulls).toHaveLength(1)
  })
})

describe("contributor: no token (Review Focus 5)", () => {
  it("queues with state no-token and no requests; after setToken all 6 send in seq order", async () => {
    const t = await setup({ withToken: false })
    await t.c.flush()
    expect(t.last()).toMatchObject({ state: "no-token", pending: 6 })
    expect(t.gh.requests).toEqual([])
    setToken(t.ls, TOKEN)
    await t.c.flush()
    expect(t.gh.requests.filter(r => r.method === "POST" && r.url.endsWith("/git/commits"))).toHaveLength(1)
    expect(t.gh.paths(UP, BR)).toEqual(logPaths(ID, [0, 1, 2, 3, 4, 5]).sort())
    expect(await t.queue.pending()).toEqual([])
  })
})

describe("contributor: token never leaks (Review Focus 1)", () => {
  it("only api.github.com; token only in Authorization header; never in a body or in IndexedDB", async () => {
    for (const canPush of [true, false]) {
      const t = await setup({ canPush })
      await t.c.flush()
      expect(t.gh.requests.length).toBeGreaterThan(10)
      for (const r of t.gh.requests) {
        expect(new URL(r.url).host).toBe("api.github.com")
        expect(new URL(r.url).protocol).toBe("https:")
        expect(r.body ?? "").not.toContain(TOKEN)
        expect(r.url).not.toContain(TOKEN)
        expect(r.headers.Authorization).toBe(`Bearer ${TOKEN}`)
        expect(r.headers["x-github-api-version"]).toBe("2022-11-28")
        const others = Object.entries(r.headers).filter(([k]) => k !== "Authorization")
        for (const [, v] of others) expect(v).not.toContain(TOKEN)
      }
      for (const s of t.statuses) expect(JSON.stringify(s)).not.toContain(TOKEN)
      expect(t.gh.blobs.size).toBe(6)
      for (const b of t.gh.blobs.values()) expect(t.gh.decode(b)).not.toContain(TOKEN)
      // dump every IndexedDB record, raw
      const db = await new Promise<IDBDatabase>((res, rej) => { const q = t.idb.open("blanktab"); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      const rows = await new Promise<unknown[]>((res, rej) => { const q = db.transaction("outbox").objectStore("outbox").getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      expect(rows).toHaveLength(1) // the life went final and was sent: only its final file is kept (prune)
      expect(JSON.stringify(rows)).not.toContain(TOKEN)
      expect(Array.from(db.objectStoreNames)).toEqual(["checkpointMeta", "checkpoints", "contribution", "outbox"])
      const contrib = await new Promise<unknown[]>((res, rej) => { const q = db.transaction("contribution").objectStore("contribution").getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      expect(JSON.stringify(contrib)).not.toContain(TOKEN)
      const metas = await new Promise<unknown[]>((res, rej) => { const q = db.transaction("checkpointMeta").objectStore("checkpointMeta").getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      expect(JSON.stringify(metas)).not.toContain(TOKEN)
      const cps = await new Promise<unknown[]>((res, rej) => { const q = db.transaction("checkpoints").objectStore("checkpoints").getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      expect(JSON.stringify(cps)).not.toContain(TOKEN)
    }
  })

  it("error statuses never contain the token", async () => {
    const t = await setup()
    t.gh.interceptors.push(() => new Response(JSON.stringify({ message: `bad ${TOKEN}` }), { status: 500 }))
    await t.c.flush()
    expect(t.statuses.length).toBeGreaterThan(0)
    for (const s of t.statuses) expect(JSON.stringify(s)).not.toContain(TOKEN)
  })
})

describe("contributor: review fixes", () => {
  it("opens a new PR when the earlier one was merged/closed", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(1)
    t.gh.pulls[0]!.state = "closed" // closed by the gate without merging: its files are not re-proposed
    await t.queue.put(mkFile(ID, 2))
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(2)
    expect(t.gh.pulls[1]).toMatchObject({ state: "open", draft: true })
    expect(t.gh.prFiles(t.gh.pulls[1]!)).toEqual(logPaths(ID, [2]))
    expect(await t.queue.pending()).toEqual([])
  })

  it("does not open a second PR while one is open", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    await t.queue.put(mkFile(ID, 2))
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(1)
  })

  it("403 with x-ratelimit-remaining 0 waits until the reset time", async () => {
    const t = await setup()
    const reset = Math.floor(NOW.getTime() / 1000) + 300
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response("{}", { status: 403, headers: { "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(reset) } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(300_000)
  })

  it("429 with no hints defaults to at least 60 s; reset is ignored while quota remains", async () => {
    const t = await setup()
    const reset = Math.floor(NOW.getTime() / 1000) + 1
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response("{}", { status: 429, headers: { "x-ratelimit-remaining": "40", "x-ratelimit-reset": String(reset) } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(60_000)
  })

  it("a 403 secondary rate limit is rate limiting (>= 60 s backoff), not a permissions error", async () => {
    for (const body of ['{"message":"You have exceeded a secondary rate limit. Please wait a few minutes before you try again."}', "You have exceeded a SECONDARY RATE LIMIT"]) {
      const t = await setup()
      t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response(body, { status: 403, headers: { "x-ratelimit-remaining": "4000" } }) : undefined))
      await t.c.flush()
      expect(t.last().message).toMatch(/limiting requests/)
      expect(t.last().message).not.toMatch(/permissions/)
      expect(t.scheduled).toHaveLength(1)
      expect(t.scheduled[0]!).toBeGreaterThanOrEqual(60_000)
    }
  })

  it("a secondary rate limit with a longer Retry-After waits that long", async () => {
    const t = await setup()
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response('{"message":"secondary rate limit"}', { status: 403, headers: { "retry-after": "120" } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(120_000)
  })

  it("a plain 403 is a permissions error: no auto-retry, queue intact", async () => {
    const t = await setup()
    t.gh.interceptors.push(r => (r.url.endsWith("/git/blobs") ? new Response('{"message":"Resource not accessible"}', { status: 403, headers: { "x-ratelimit-remaining": "4000" } }) : undefined))
    await t.c.flush()
    expect(t.last()).toEqual({ state: "error", message: "Your GitHub token can't write to the repo — check its permissions.", pending: 6 })
    expect(t.scheduled).toEqual([])
    expect((await t.queue.pending()).length).toBe(6)
  })

  it("treats a 422 'pull request already exists' (another tab) as success", async () => {
    const t = await setup({ files: 1 })
    t.gh.interceptors.push(r => {
      if (r.method !== "POST" || !r.url.endsWith("/pulls")) return undefined
      t.gh.pulls.push({ ...JSON.parse(r.body!), state: "open", number: t.gh.pulls.length + 1 }) // the other tab won the race
      return new Response('{"message":"A pull request already exists"}', { status: 422 })
    })
    await t.c.flush()
    expect(t.last()).toMatchObject({ state: "idle", pending: 0 })
    expect(t.gh.pulls).toHaveLength(1)
  })
})

describe("contributor: report() between sends", () => {
  it("shows how many logs are waiting, without any request", async () => {
    const t = await setup({ files: 2 })
    await t.c.report()
    expect(t.gh.requests).toHaveLength(0)
    expect(t.last()).toEqual({ state: "idle", message: "2 session logs waiting; they are sent every 10 minutes and when a life ends.", pending: 2 })
  })
  it("says all shared when nothing is waiting", async () => {
    const t = await setup({ files: 1 })
    await t.c.flush()
    await t.c.report()
    expect(t.last()).toEqual({ state: "idle", message: "All session logs are shared.", pending: 0 })
  })
  it("does not hide an error the user must act on", async () => {
    const t = await setup({ withToken: false, files: 1 })
    await t.c.report()
    expect(t.last().state).toBe("no-token")
  })
})

describe("contributor: PRs the gate closes (decision 0014: seq gaps, local-only)", () => {
  type T = Awaited<ReturnType<typeof setup>>
  const feed = async (t: T, seqs: number[]) => { for (const s of seqs) await t.queue.put(mkFile(ID, s, { codeSha: "a".repeat(40) })) }
  /** The gate's view of an open PR: its files as data, main's files for the learner as previous. */
  const gateValidates = (t: T, i: number) => {
    const pr = t.gh.pulls[i]!
    const files = t.gh.prFiles(pr).map(path => ({ path, content: JSON.stringify(t.gh.file(t.gh.repos.has("ann/learner") ? "ann/learner" : UP, BR, path)) }))
    const previous = t.gh.paths(UP, "main").map(path => parseLogFile(t.gh.file(UP, "main", path))).flatMap(p => (p.ok ? [p.value] : []))
    return validateSessionPr(files, previous, { spaces: spacesOf(GENOME_SPACE) })
  }

  for (const deleteBranch of [false, true]) {
    it(`one closed PR (branch ${deleteBranch ? "deleted" : "kept"}): its files are dropped, never resent; the next PR (a seq gap) validates and merges`, async () => {
      const t = await setup({ files: 0 })
      await feed(t, [0, 1])
      await t.c.flush()
      t.gh.close(t.gh.pulls[0]!, { deleteBranch })
      await feed(t, [2, 3])
      await t.c.flush()
      expect(t.gh.pulls).toHaveLength(2)
      expect(t.gh.prFiles(t.gh.pulls[1]!)).toEqual(logPaths(ID, [2, 3]))
      expect(await gateValidates(t, 1)).toMatchObject({ ok: true })
      t.gh.merge(t.gh.pulls[1]!, { deleteBranch: true, matchHeadCommit: t.gh.headOf(t.gh.pulls[1]!)! })
      expect(t.gh.paths(UP, "main")).toEqual(logPaths(ID, [2, 3]))
      expect((await t.queue.files(ID)).map(f => f.seq)).toEqual([3]) // 0 and 1 dropped (and pruned), 2 sent: only the newest is kept
      expect(await t.queue.contribution(ID)).toMatchObject({ lastPr: 1, closedInARow: 1, localOnly: false })
      // The life goes on: the next PR follows main, and the merge resets the count.
      await feed(t, [4])
      await t.c.flush()
      expect(t.gh.prFiles(t.gh.pulls[2]!)).toEqual(logPaths(ID, [4]))
      expect(await gateValidates(t, 2)).toMatchObject({ ok: true })
      expect(await t.queue.contribution(ID)).toMatchObject({ lastPr: 2, closedInARow: 0, localOnly: false })
      expect(await t.queue.pending()).toEqual([])
    })
  }

  it("three closed PRs in a row: the learner is local-only (kept in IndexedDB), sends nothing more, and the page is told", async () => {
    const told: string[] = []
    const t = await setup({ files: 0, onLocalOnly: id => told.push(id) })
    for (let i = 0; i < 3; i++) {
      await feed(t, [2 * i, 2 * i + 1])
      await t.c.flush()
      expect(t.gh.pulls).toHaveLength(i + 1)
      t.gh.close(t.gh.pulls[i]!)
    }
    await feed(t, [6, 7])
    t.gh.requests.length = 0
    await t.c.flush()
    expect(writes(t.gh.requests)).toEqual([]) // no fourth PR
    expect(t.gh.pulls).toHaveLength(3)
    expect(told).toEqual([ID])
    expect(await (await openQueue(t.idb)).contribution(ID)).toEqual({ learnerId: ID, lastPr: 3, closedInARow: 3, localOnly: true })
    expect(await t.queue.localOnly()).toEqual([ID])
    expect(await t.queue.pending()).toEqual([]) // never sent, so not waiting either
    expect(t.last()).toMatchObject({ state: "idle", pending: 0, message: "No session logs waiting; a learner here no longer contributes (its logs were rejected)." })
    // Later logs are kept locally (for the page) but never sent.
    await feed(t, [8])
    t.gh.requests.length = 0
    await t.c.flush()
    expect(t.gh.requests).toEqual([])
    expect((await t.queue.files(ID)).map(f => f.seq)).toEqual([6, 7, 8])
  })

  it("a merged PR between closed ones resets the count: closed, merged, closed, closed is not local-only", async () => {
    const t = await setup({ files: 0 })
    const outcome = ["close", "merge", "close", "close"]
    for (let i = 0; i < outcome.length; i++) {
      await feed(t, [i])
      await t.c.flush()
      const pr = t.gh.pulls[i]!
      if (outcome[i] === "merge") t.gh.merge(pr, { deleteBranch: true, matchHeadCommit: t.gh.headOf(pr)! })
      else t.gh.close(pr)
    }
    await feed(t, [4])
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(5)
    expect(await t.queue.contribution(ID)).toMatchObject({ lastPr: 4, closedInARow: 2, localOnly: false })
    expect(await gateValidates(t, 4)).toMatchObject({ ok: true })
  })

  it("another learner in the same browser keeps contributing while one is local-only", async () => {
    const t = await setup({ files: 0 })
    await t.queue.setContribution({ learnerId: ID, lastPr: 0, closedInARow: 3, localOnly: true })
    const OTHER = "lrn_bbbbbbbbbbbbbbbb"
    await feed(t, [0])
    await t.queue.put(mkFile(OTHER, 0))
    await t.c.flush()
    expect(t.gh.pulls.map(p => p.head)).toEqual([`markwhiting:run/${OTHER}`])
  })
})

describe("contributor: after a push the PR is checked by number (item 5)", () => {
  it("open with its head at the new commit: sent, even when the open-PR list lags behind", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    await t.queue.put(mkFile(ID, 2))
    let pushed = false
    t.gh.interceptors.push(r => {
      const u = new URL(r.url)
      if (r.method === "PATCH" && u.pathname.endsWith(`/git/refs/heads/${BR}`)) pushed = true
      // After the push the list says no open PR (GitHub's listing lags); the PR itself is open.
      return pushed && r.method === "GET" && u.pathname === `/repos/${UP}/pulls` ? new Response("[]", { status: 200 }) : undefined
    })
    await t.c.flush()
    expect(pushed).toBe(true)
    expect(await t.queue.pending()).toEqual([])
    expect(t.gh.pulls).toHaveLength(1)
    const check = t.gh.requests.filter(r => r.method === "GET" && new URL(r.url).pathname === `/repos/${UP}/pulls/1`)
    expect(check).toHaveLength(1)
  })

  for (const [name, change] of [
    ["merged meanwhile", (pr: Record<string, unknown>) => ({ ...pr, state: "closed", merged_at: "2026-10-21T12:00:00Z" })],
    ["closed meanwhile", (pr: Record<string, unknown>) => ({ ...pr, state: "closed", merged_at: null })],
    ["its head is another commit", (pr: Record<string, unknown>) => ({ ...pr, head: { sha: "f".repeat(40) } })],
  ] as const) {
    it(`${name}: the files are not marked sent; a later flush sends them`, async () => {
      const t = await setup({ files: 2 })
      await t.c.flush()
      await t.queue.put(mkFile(ID, 2))
      let broken = true
      t.gh.interceptors.push(r => {
        if (!broken || r.method !== "GET" || new URL(r.url).pathname !== `/repos/${UP}/pulls/1`) return undefined
        broken = false
        const pr = t.gh.pulls[0]!
        return new Response(JSON.stringify(change({ number: 1, state: "open", merged_at: null, head: { sha: t.gh.headOf(pr) } })), { status: 200 })
      })
      await t.c.flush()
      expect((await t.queue.pending()).map(f => f.seq)).toEqual([2])
      expect(t.last().state).toBe("error")
      await t.c.flush()
      expect(await t.queue.pending()).toEqual([])
    })
  }
})

describe("contributor: re-review fixes", () => {
  it("the open-PR list lags but the full list shows the PR still open: it is used as the open PR (the branch is not reset)", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    await t.queue.put(mkFile(ID, 2))
    t.gh.interceptors.push(r => {
      const u = new URL(r.url)
      return r.method === "GET" && u.pathname === `/repos/${UP}/pulls` && u.searchParams.get("state") === "open" ? new Response("[]", { status: 200 }) : undefined
    })
    t.gh.requests.length = 0
    await t.c.flush()
    expect(t.gh.requests.some(r => r.method === "PATCH" && JSON.parse(r.body!).force === true)).toBe(false)
    expect(t.gh.requests.some(r => r.method === "POST" && new URL(r.url).pathname === `/repos/${UP}/pulls`)).toBe(false)
    expect(t.gh.pulls).toHaveLength(1)
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, [0, 1, 2]))
    expect(await t.queue.pending()).toEqual([])
    expect(await t.queue.contribution(ID)).toMatchObject({ closedInARow: 0, localOnly: false })
  })

  it("both PR lists lag behind a PR this browser just opened: it is found by number, the branch is not reset, no second PR (2026-10-26)", async () => {
    // Live: a newborn's first PR was 9 s old when the next send listed none (open or all), reset its branch to main, so
    // GitHub closed it as empty and its 0.json was dropped as rejected, on almost every life.
    const t = await setup({ files: 1 })
    await t.c.flush()
    await t.queue.put(mkFile(ID, 1))
    t.gh.interceptors.push(r => {
      const u = new URL(r.url)
      return r.method === "GET" && u.pathname === `/repos/${UP}/pulls` ? new Response("[]", { status: 200 }) : undefined
    })
    t.gh.requests.length = 0
    await t.c.flush()
    expect(t.gh.requests.some(r => r.method === "PATCH" && JSON.parse(r.body!).force === true)).toBe(false)
    expect(t.gh.requests.some(r => r.method === "POST" && new URL(r.url).pathname === `/repos/${UP}/pulls`)).toBe(false)
    expect(t.gh.pulls).toHaveLength(1)
    expect(t.gh.prFiles(t.gh.pulls[0]!)).toEqual(logPaths(ID, [0, 1]))
    expect(await t.queue.pending()).toEqual([])
    expect(await t.queue.contribution(ID)).toMatchObject({ closedInARow: 0, localOnly: false })
  })

  it("the remembered PR was merged or closed: the next send starts afresh (a new branch from main, a new PR)", async () => {
    const t = await setup({ files: 1 })
    await t.c.flush()
    t.gh.pulls[0]!.state = "closed"
    t.gh.pulls[0]!.merged = true
    await t.queue.put(mkFile(ID, 1))
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(2)
    expect(t.gh.prFiles(t.gh.pulls[1]!)).toEqual(logPaths(ID, [1]))
    expect(await t.queue.pending()).toEqual([])
  })

  it("sends run under the cross-tab lock it is given, one at a time", async () => {
    const order: string[] = []
    let held: Promise<unknown> = Promise.resolve()
    const lock = <T>(fn: () => Promise<T>): Promise<T> => {
      const run = held.then(async () => { order.push("in"); try { return await fn() } finally { order.push("out") } })
      held = run.catch(() => {})
      return run
    }
    const t = await setup({ files: 2, lock })
    await t.c.flush()
    expect(order).toEqual(["in", "out"])
    expect(t.gh.pulls).toHaveLength(1)
  })

  it("every GitHub call bypasses the browser's HTTP cache (GitHub marks GETs cacheable for 60 s; 2026-10-26)", async () => {
    // A cached "no open PR" made a send reset the branch of a PR it had just opened (GitHub closed it, its files were
    // dropped), and a cached branch head made the next commit not a fast-forward (a 422 until the cache expired).
    const t = await setup({ files: 2 })
    const modes: (RequestCache | undefined)[] = []
    const inner = t.gh.fetchFn
    const c = createContributor({
      queue: t.queue, ls: t.ls, now: () => NOW, onStatus: () => {}, sleep: async () => {}, schedule: () => {},
      fetchFn: ((url: RequestInfo | URL, init?: RequestInit) => { modes.push(init?.cache); return inner(url, init) }) as typeof fetch,
    })
    await c.flush()
    expect(modes.length).toBeGreaterThan(5)
    expect(new Set(modes)).toEqual(new Set(["no-store"]))
  })

  it("a failure while reporting idle with nothing pending is a status, not a rejected flush", async () => {
    const t = await setup({ files: 0 })
    t.queue.localOnly = async () => { throw new Error("disk") }
    await expect(t.c.flush()).resolves.toBeUndefined()
    expect(t.last().state).toBe("error")
  })
})
