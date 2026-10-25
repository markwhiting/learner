import { afterEach, beforeEach, describe, it, expect } from "vitest"
import { execFileSync } from "node:child_process"
import { join } from "node:path"
import { act, decide, memoBound, runGate, type GateDeps, type GateOps, type OpenPr } from "../gate.ts"
import { gitRunner, MAX_PR_FILES } from "../github.ts"
import { FIXTURE_ENV, makeFixture, prObject, type Fixture } from "./git-fixture.ts"
import { spacesOf, type PrFile } from "../validate.ts"
import type { LogFile, LogFileV1 } from "@blanktab/harness/log"
import { SPACE_JSON } from "./git-fixture.ts"
import { parseGenomeSpace } from "@blanktab/meta"

const SPACE = spacesOf(parseGenomeSpace(JSON.parse(SPACE_JSON)))

const ID = "lrn_0123456789abcdef"
const NOW = new Date("2026-10-21T12:00:00.000Z")
const hoursAgo = (h: number) => new Date(NOW.getTime() - h * 3_600_000)
const log = (seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId: ID, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: hoursAgo(1).toISOString(), stepsTotal: 100 * (seq + 1), settings: { lr: 0.1 },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})
const file = (l: LogFile): PrFile => ({ path: `data/runs/${ID}/${l.seq}.json`, content: JSON.stringify(l) })
const written = (h: number) => ({ writtenAt: hoursAgo(h).toISOString() })
const pr = (over: Partial<OpenPr> = {}): OpenPr => ({ number: 1, title: `run: ${ID}`, headRef: `run/${ID}`, baseRef: "main", files: [file(log(0))], previous: [], author: "ann", boundAuthor: null, ...over })

describe("decide", () => {
  it("valid + final -> merge", async () => {
    const [d] = await decide([pr({ files: [file(log(0)), file(log(1, { final: true }))] })], SPACE)
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("a valid PR that is not final is merged, however fresh its logs", async () => {
    const [d] = await decide([pr({ files: [file(log(0, written(0.01)))] })], SPACE)
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("a valid PR is merged however old or new it is; decide never waits", async () => {
    for (const h of [0, 1, 47, 49, 24 * 8]) {
      expect((await decide([pr({ files: [file(log(0, written(h)))] })], SPACE))[0]!.action).toBe("merge")
    }
    expect((await decide([pr({ files: [file(log(0, written(-400 * 24)))] })], SPACE))[0]!.action).toBe("merge")
  })
  it("invalid -> close with the validator error in the reason", async () => {
    const [d] = await decide([pr({ files: [file(log(1))], previous: [log(0), log(1)] })], SPACE)
    expect(d!.action).toBe("close")
    expect(d!.reason).toContain("seq 1 is not above seq 1, the highest on main")
  })
  it("after a closed PR, the learner's next PR (a seq gap on main) is merged", async () => {
    // seq 0-1 merged; seq 2-3 went in a PR the gate closed; the learner goes on from seq 4.
    const [d] = await decide([pr({ files: [file(log(4)), file(log(5))], previous: [log(0), log(1)] })], SPACE)
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("code path -> label-needs-human", async () => {
    const [d] = await decide([pr({ files: [file(log(0)), { path: "app/src/main.ts", content: "x" }] })], SPACE)
    expect(d!.action).toBe("label-needs-human")
    expect(d!.reason).toContain("app/src/main.ts")
  })
  it("closes PRs whose base is not main, even if otherwise mergeable", async () => {
    const [d] = await decide([pr({ baseRef: "release", files: [file(log(0, { final: true }))] })], SPACE)
    expect(d).toMatchObject({ action: "close", reason: "Session PRs must target main." })
  })
  describe("development builds", () => {
    it("any log with codeSha dev -> close, even when final", async () => {
      const [d] = await decide([pr({ files: [file(log(0)), file(log(1, { codeSha: "dev", final: true }))] })], SPACE)
      expect(d).toEqual({ pr: 1, action: "close", reason: "Development builds don't contribute." })
    })
    it("dev build that is also old -> close, never merge", async () => {
      expect((await decide([pr({ files: [file(log(0, { codeSha: "dev", ...written(100) }))] })], SPACE))[0]).toMatchObject({ action: "close", reason: "Development builds don't contribute." })
    })
  })
})

describe("act", () => {
  const ops = () => {
    const calls: string[] = []
    const o: GateOps = {
      close: n => void calls.push(`close ${n}`), label: n => void calls.push(`label ${n}`),
      merge: (n, sha) => void calls.push(`merge ${n} ${sha}`), log: m => void calls.push(`log ${m}`), error: m => void calls.push(`error ${m}`),
    }
    return { o, calls }
  }
  it("merges a valid PR pinned to its validated head commit", async () => {
    const { o, calls } = ops()
    const p = { ...pr(), headSha: "c".repeat(40), learnerId: ID }
    expect(act(await decide([p], SPACE), [p], o)).toBe(1)
    expect(calls).toContain(`merge 1 ${"c".repeat(40)}`)
  })
  it("a PR loaded without a head commit is not merged (merge never sees undefined); it is logged and left for the next run", async () => {
    const { o, calls } = ops()
    const p = { ...pr(), learnerId: ID } // no headSha
    expect(act(await decide([p], SPACE), [p], o)).toBe(0)
    expect(calls.some(c => c.startsWith("merge"))).toBe(false)
    expect(calls.some(c => c.startsWith("error PR #1") && c.includes("no head commit"))).toBe(true)
  })
  it("merges at most one PR per learner per run", async () => {
    const { o, calls } = ops()
    const a = { ...pr({ number: 1 }), headSha: "c".repeat(40), learnerId: ID }, b = { ...pr({ number: 2 }), headSha: "d".repeat(40), learnerId: ID }
    expect(act(await decide([a, b], SPACE), [a, b], o)).toBe(1)
    expect(calls.filter(c => c.startsWith("merge"))).toEqual([`merge 1 ${"c".repeat(40)}`])
  })
})

describe("decide: a learner belongs to the author of its first merged PR", () => {
  it("bound to the PR's author (any letter case): merged", async () => {
    expect((await decide([pr({ author: "ann", boundAuthor: "ann", previous: [log(0)], files: [file(log(1))] })], SPACE))[0]).toMatchObject({ action: "merge" })
    expect((await decide([pr({ author: "Ann", boundAuthor: "aNN", previous: [log(0)], files: [file(log(1))] })], SPACE))[0]).toMatchObject({ action: "merge" })
  })
  it("bound to someone else: closed, saying whose it is", async () => {
    const [d] = await decide([pr({ author: "mallory", boundAuthor: "ann", previous: [log(0)], files: [file(log(1))] })], SPACE)
    expect(d).toEqual({ pr: 1, action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
  it("bound, but the PR's author is unknown: closed", async () => {
    expect((await decide([pr({ author: undefined, boundAuthor: "ann", previous: [log(0)], files: [file(log(1))] })], SPACE))[0]).toMatchObject({ action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
  it("not bound yet (nothing of it on main): anyone's valid first PR is merged", async () => {
    expect((await decide([pr({ author: "bob", boundAuthor: null })], SPACE))[0]).toMatchObject({ action: "merge" })
  })
  it("the title must name the files' learner", async () => {
    const [d] = await decide([pr({ title: "run: lrn_fedcba9876543210" })], SPACE)
    expect(d).toMatchObject({ action: "close" })
    expect(d!.reason).toContain(`run: ${ID}`)
    expect((await decide([pr({ title: "run: x" })], SPACE))[0]!.action).toBe("close")
  })
  it("two first PRs for one learner in one run, from two authors: one merges; next run the other is closed as someone else's", async () => {
    const { o, calls } = (() => {
      const calls: string[] = []
      const o: GateOps = { close: n => void calls.push(`close ${n}`), label: () => {}, merge: n => void calls.push(`merge ${n}`), log: () => {}, error: () => {} }
      return { o, calls }
    })()
    const a = { ...pr({ number: 1, author: "ann" }), headSha: "c".repeat(40), learnerId: ID }
    const b = { ...pr({ number: 2, author: "bob" }), headSha: "d".repeat(40), learnerId: ID }
    expect(act(await decide([a, b], SPACE), [a, b], o)).toBe(1)
    expect(calls).toEqual(["merge 1"])
    // Next run: PR 1's merge bound the learner to ann.
    const b2 = { ...b, boundAuthor: "ann", previous: [log(0)] }
    expect((await decide([b2], SPACE))[0]).toEqual({ pr: 2, action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
})

describe("oldest PR first (a later copy never wins an unbound learner's first merge)", () => {
  it("fed newest first, the older PR merges; next run the newer is closed as the older author's", async () => {
    const calls: string[] = []
    const o: GateOps = { close: n => void calls.push(`close ${n}`), label: () => {}, merge: n => void calls.push(`merge ${n}`), log: () => {}, error: () => {} }
    const older = { ...pr({ number: 5, author: "ann" }), headSha: "c".repeat(40), learnerId: ID }
    const copy = { ...pr({ number: 9, author: "mallory" }), headSha: "d".repeat(40), learnerId: ID }
    expect(act(await decide([copy, older], SPACE), [copy, older], o)).toBe(1)
    expect(calls).toEqual(["merge 5"])
    const next = { ...copy, boundAuthor: "ann", previous: [log(0)] }
    expect((await decide([next], SPACE))[0]).toEqual({ pr: 9, action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
})

describe("memoBound: one owner lookup per learner per run", () => {
  it("looks each learner up once, failures included", async () => {
    const asked: string[] = []
    const bound = memoBound(id => { asked.push(id); if (id === "lrn_2222222222222222") throw new Error("gh down"); return id === ID ? "ann" : null })
    expect(bound(ID)).toBe("ann")
    expect(bound(ID)).toBe("ann")
    expect(bound("lrn_1111111111111111")).toBeNull()
    expect(bound("lrn_1111111111111111")).toBeNull()
    expect(() => bound("lrn_2222222222222222")).toThrow("gh down")
    expect(() => bound("lrn_2222222222222222")).toThrow("gh down")
    expect(asked).toEqual([ID, "lrn_1111111111111111", "lrn_2222222222222222"])
  })
})

describe("runGate (a local origin over file://, a fake gh)", { timeout: 60_000 }, () => {
  let fx: Fixture
  beforeEach(() => { fx = makeFixture() })
  afterEach(() => fx.cleanup())
  const D = `data/runs/${ID}`
  const json = (l: LogFile) => JSON.stringify(l)
  /**
   * A fake gh: lists the given PRs, answers pulls/<n> with `heads[n]` (and `over`), commits/<sha>/pulls from `owners`
   * (a commit not in it was a direct push: []), and records ready/merge/close/label calls.
   */
  const deps = (heads: Record<number, string>, owners: Record<string, string> = {}, over: Record<string, unknown> = {}, workDir = fx.work, labels: Record<number, string[]> = {}) => {
    const calls: string[] = [], logs: string[] = []
    const d: GateDeps = {
      gh: a => {
        const j = a.join(" ")
        calls.push(j)
        if (a[0] === "pr" && a[1] === "list") return Buffer.from(JSON.stringify(Object.keys(heads).map(n => ({ number: Number(n), title: `run: ${ID}`, headRefName: `run/${n}`, baseRefName: "main", labels: (labels[Number(n)] ?? []).map(name => ({ name })) }))))
        let m = /^api repos\/o\/r\/pulls\/(\d+)$/.exec(j)
        if (m) return prObject(Number(m[1]), heads[Number(m[1])]!, over)
        m = /^api repos\/o\/r\/commits\/([0-9a-f]{40})\/pulls$/.exec(j)
        if (m) return Buffer.from(JSON.stringify(owners[m[1]!] ? [{ merged_at: "t", user: { login: owners[m[1]!] } }] : []))
        if (a[0] === "pr" || a[0] === "label") return Buffer.from("")
        throw new Error(`unexpected gh call: ${j}`)
      },
      git: a => execFileSync("git", a, { cwd: workDir, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, ...FIXTURE_ENV } }),
      gitBuf: gitRunner({ cwd: workDir, env: FIXTURE_ENV }),
      log: m => void logs.push(m), error: m => void logs.push(`error ${m}`),
    }
    return { d, calls, logs }
  }

  it("a learner whose oldest commit was a direct push is decided (merged when valid)", async () => {
    fx.commitMain({ [`${D}/0.json`]: json(log(0)) }) // a direct push: no PR
    const head = fx.pr(1, { [`${D}/1.json`]: json(log(1)) })
    const { d, calls } = deps({ 1: head })
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(1)
    expect(calls).toContain(`pr merge 1 --repo o/r --squash --delete-branch --match-head-commit ${head}`)
  })

  it("an owned learner: another author's PR is closed (the owner comes from the first merged PR)", async () => {
    const direct = fx.commitMain({ [`${D}/0.json`]: json(log(0)) })
    const viaPr = fx.commitMain({ [`${D}/1.json`]: json(log(1)) })
    const head = fx.pr(2, { [`${D}/2.json`]: json(log(2)) })
    const { d, calls } = deps({ 2: head }, { [viaPr]: "bob" })
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(0)
    expect(calls.some(c => c.startsWith("pr close 2") && c.includes(`belongs to @bob`))).toBe(true)
    expect(calls).toContain(`api repos/o/r/commits/${direct}/pulls`)
  })

  it("a head that moved since pulls/N: gate main leaves the PR open", async () => {
    fx.pr(1, { [`${D}/0.json`]: json(log(0)) })
    const { d, calls, logs } = deps({ 1: "f".repeat(40) })
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(0)
    expect(calls.filter(c => c.startsWith("pr ") && !c.startsWith("pr list"))).toEqual([]) // no close, no merge
    expect(logs.some(l => l.includes("PR #1") && l.includes("moved") && l.includes("next run"))).toBe(true)
  })

  it("a fetch over its bound (here: time) labels that PR needs-human at once, the run goes on, and later runs skip it", async () => {
    const h1 = fx.pr(1, { [`${D}/0.json`]: json(log(0)) })
    const h2 = fx.pr(2, { [`${D}/0.json`]: json(log(0)) })
    const { d, calls, logs } = deps({ 1: h1, 2: h2 })
    const real = d.gitBuf
    d.gitBuf = (a, env, limits) => a.includes(`+${h1}:refs/gate/pr-1`) ? gitRunner({ cwd: fx.work, env: FIXTURE_ENV, timeoutMs: 1 })(a, env, limits) : real(a, env, limits)
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(1)
    expect(logs.some(l => l.startsWith("error PR #1: fetching it took over 120 s") && l.includes("needs-human"))).toBe(true)
    expect(calls).toContain("pr edit 1 --repo o/r --add-label needs-human")
    expect(calls.filter(c => /^pr (close|merge|ready) 1 /.test(c))).toEqual([])
    expect(calls).toContain(`pr merge 2 --repo o/r --squash --delete-branch --match-head-commit ${h2}`)
    // Next run: PR 1 carries the label, so it is neither read nor fetched again.
    const next = deps({ 1: h1 }, {}, {}, fx.work, { 1: ["needs-human"] })
    const fetched: string[][] = []
    next.d.gitBuf = (a, env, limits) => { fetched.push(a); return real(a, env, limits) }
    expect(await runGate("o/r", join(fx.work, "data/runs"), next.d)).toBe(0)
    expect(next.calls).toEqual([next.calls[0]]) // only the PR list
    expect(fetched).toEqual([])
  })

  it("a git error that is not a bound (e.g. the network) leaves the PR for the next run, unlabelled", async () => {
    const h1 = fx.pr(1, { [`${D}/0.json`]: json(log(0)) })
    const { d, calls, logs } = deps({ 1: h1 })
    d.gitBuf = () => { throw new Error("fatal: unable to access origin") }
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(0)
    expect(logs.some(l => l.startsWith("error PR #1: could not load") && l.includes("next run"))).toBe(true)
    expect(calls.filter(c => c.startsWith("pr edit") || c.startsWith("label"))).toEqual([])
  })

  it("two PRs for one learner in one run: the older merges, the newer is deferred (not closed) to the next run", async () => {
    fx.commitMain({ [`${D}/0.json`]: json(log(0)) })
    const h1 = fx.pr(1, { [`${D}/1.json`]: json(log(1)) })
    const h2 = fx.pr(2, { [`${D}/2.json`]: json(log(2)) })
    const { d, calls, logs } = deps({ 1: h1, 2: h2 })
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(1)
    expect(calls).toContain(`pr merge 1 --repo o/r --squash --delete-branch --match-head-commit ${h1}`)
    expect(calls.filter(c => /^pr (close|merge|ready|edit) 2 /.test(c))).toEqual([])
    expect(logs).toContain(`PR #2: deferred, another PR for ${ID} merged this run`)
  })

  it("a PR based on a main newer than the gate's checkout: left open for the next run", async () => {
    fx.commitMain({ "other/file.txt": "x" }, [], false)
    const head = fx.pr(1, { [`${D}/0.json`]: json(log(0)) })
    const { d, calls, logs } = deps({ 1: head })
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(0)
    expect(calls.filter(c => c.startsWith("pr ") && !c.startsWith("pr list"))).toEqual([])
    expect(logs.some(l => l.includes("PR #1") && l.includes("next run"))).toBe(true)
  })

  it("a shallow checkout aborts the run before deciding anything", async () => {
    const head = fx.pr(1, { [`${D}/0.json`]: json(log(0)) })
    const origin = fx.git(["remote", "get-url", "origin"]).toString().trim()
    const shallow = join(fx.work, "..", "shallow")
    execFileSync("git", ["clone", "-q", "--depth", "1", origin, shallow], { stdio: "ignore" })
    const { d, calls } = deps({ 1: head }, {}, {}, shallow)
    await expect(runGate("o/r", join(shallow, "data/runs"), d)).rejects.toThrow("shallow")
    expect(calls).toEqual([])
  })

  it("a checkout whose genome space is missing or invalid aborts the run before deciding anything", async () => {
    const head = fx.pr(1, { [`${D}/0.json`]: json(log(0)) })
    fx.commitMain({ "knowledge/genome-space.json": "{ \"schemaVersion\": 0 }" })
    const bad = deps({ 1: head })
    await expect(runGate("o/r", join(fx.work, "data/runs"), bad.d)).rejects.toThrow("knowledge/genome-space.json on main is invalid")
    expect(bad.calls).toEqual([])
    fx.commitMain({}, ["knowledge/genome-space.json"])
    const gone = deps({ 1: head })
    await expect(runGate("o/r", join(fx.work, "data/runs"), gone.d)).rejects.toThrow("knowledge/genome-space.json could not be read")
    expect(gone.calls).toEqual([])
  })

  it("at the 16-file cap, gh calls per PR ≤ 3 before the merge", async () => {
    const first = fx.commitMain({ [`${D}/0.json`]: json(log(0)) })
    const files = Object.fromEntries(Array.from({ length: MAX_PR_FILES }, (_, i) => [`${D}/${i + 1}.json`, json(log(i + 1))]))
    const head = fx.pr(1, files)
    const { d, calls, logs } = deps({ 1: head }, { [first]: "ann" }, { changed_files: MAX_PR_FILES })
    expect(await runGate("o/r", join(fx.work, "data/runs"), d)).toBe(1)
    const before = calls.slice(1, calls.findIndex(c => c.startsWith("pr ready")))
    expect(before).toEqual(["api repos/o/r/pulls/1", `api repos/o/r/commits/${first}/pulls`])
    expect(before.length).toBeLessThanOrEqual(3)
    expect(logs).toContain("PR #1: 2 gh calls")
  })
})
