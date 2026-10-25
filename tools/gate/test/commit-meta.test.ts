// The gate's direct commit of data/meta/** (decision 0019; invariant 7): only the gate's own files, never forced.
import { execFileSync } from "node:child_process"
import { createHash } from "node:crypto"
import { chmodSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { afterEach, describe, expect, it } from "vitest"
import { commitMeta, META_PATH_RE, metaPathErrors, promotionErrors } from "../commit-meta.ts"
import { computeKnowledgeSha } from "../manifest.ts"
import { genomeShaSync } from "../history.ts"
import { serialise } from "../serialise.ts"
import { makeFixture, SPACE_JSON, type Fixture } from "./git-fixture.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
// No author or committer names in the environment: they would override the bot identity commitMeta sets.
const ENV: NodeJS.ProcessEnv = { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" }
for (const k of ["GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"]) delete ENV[k]

const posterior = (mode: string, eligibleLives: number, tag = "") => JSON.stringify({ mode, page: { eligibleLives }, tag }, null, 1) + "\n"
const sha8 = (text: string) => createHash("sha256").update(text, "utf8").digest("hex").slice(0, 8)
const META = (tag: string) => ({
  "data/meta/posterior.json": posterior("calibrate", 3, tag),
  "data/meta/calibration.json": `{"c":"${tag}"}\n`,
  "data/meta/diagnostics.json": `{"d":"${tag}"}\n`,
})

// A confirmation, a promotion and the genome files as the gate writes them (confirm.ts), for genome schema 1.
const GENOME_JSON = readFileSync(join(REPO, "knowledge/genome.json"), "utf8")
type Values = Record<string, number | string>
const CENTRE = (JSON.parse(GENOME_JSON) as { values: Values }).values
const CANDIDATE: Values = { ...CENTRE, lr: 1e-3 }
const shaOf = (v: Values) => genomeShaSync(1, v)
const AT = "2026-11-01T10:00:00.000Z", DEADLINE = "2026-11-22T10:00:00.000Z"
const confirmationText = (id: number, outcome: string | null, over: Record<string, unknown> = {}) => serialise({
  schemaVersion: 1, id, genomeSchema: 1, confirm: "confirm-fisher@1", search: "search-composite@1",
  candidate: { genomeSha: shaOf(CANDIDATE), values: CANDIDATE }, centre: { genomeSha: shaOf(CENTRE), values: CENTRE },
  seedBase: 10_000 + 1_000 * id, target: { candidate: 24, centre: 24, newborn: 60 }, declaredAt: AT, deadline: DEADLINE, sourceCommit: "a".repeat(40),
  trigger: { pBeatsCentre: 0.97, incumbentMean: 0.4, rowsAtDeclaration: 120, modelSha: "b".repeat(64) },
  result: outcome === null ? null : {
    outcome, decidedAt: "2026-11-10T10:00:00.000Z", sourceCommit: "c".repeat(40), output: { eligible: true, score: outcome === "pass" ? 1 : 0 },
    lives: { candidate: [], centre: [], newborn: [] },
  },
  ...over,
})
const promotedGenome = (values: Values = CANDIDATE, confirmation = 0) => serialise({
  schemaVersion: 1, genomeSchema: 1, values,
  provenance: { source: "promotion", confirmation, decision: "docs/decisions/0018-one-meta-learner-over-a-learning-genome.md", promotedAt: AT, fisher: null, mannWhitneyP: null, lives: { candidate: [], centre: [], newborn: [] } },
})
const promotionLine = (confirmation = 0, genomeSha = shaOf(CANDIDATE)) =>
  JSON.stringify({ confirmation, genomeSha, previous: shaOf(CENTRE), promotedAt: AT, sourceCommit: "c".repeat(40) }) + "\n"
const C0 = "data/meta/confirmations/0.json", PROMOTIONS = "data/meta/promotions.jsonl", GENOME = "knowledge/genome.json"
const SEARCH = (tag: string) => ({ ...META(tag), "data/meta/posterior.json": posterior("search", 140, tag), "data/meta/verified.json": `{"v":"${tag}"}\n` })
/** The files of the build that decides confirmation 0 with a pass and promotes its candidate. */
const PROMOTION = (tag = "p") => ({ ...SEARCH(tag), [C0]: confirmationText(0, "pass"), [PROMOTIONS]: promotionLine(), [GENOME]: promotedGenome() })

/** A forced push: --force, --force-with-lease, -f (alone or in a flag group) or a +refspec. */
const isForce = (a: string) => a.startsWith("--force") || /^-[a-z]*f/.test(a) || a.startsWith("+")

let fx: Fixture | null = null
afterEach(() => { fx?.cleanup(); fx = null })

function setup() {
  fx = makeFixture()
  const work = fx.work
  const calls: string[][] = []
  const git = (args: string[]): string => {
    calls.push(args)
    try {
      return execFileSync("git", args, { cwd: work, encoding: "utf8", env: ENV, stdio: ["ignore", "pipe", "pipe"] })
    } catch (e) {
      const err = e as { stderr?: string; message: string }
      throw new Error(`git ${args.join(" ")} failed: ${err.stderr ?? err.message}`, { cause: e })
    }
  }
  const write = (files: Record<string, string>) => {
    for (const [p, t] of Object.entries(files)) { mkdirSync(dirname(join(work, p)), { recursive: true }); writeFileSync(join(work, p), t) }
  }
  // git in the bare "origin": what main really is after a push.
  const origin = (...args: string[]) => execFileSync("git", ["--git-dir", join(work, "../origin.git"), ...args], { encoding: "utf8", env: ENV }).trim()
  // The lockfile npm ci "installed" from: main's, as the workflow's LOCK_BLOB records it.
  fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3}\n", "knowledge/genome.json": GENOME_JSON })
  const installedLock = git(["rev-parse", "HEAD:package-lock.json"]).trim()
  calls.length = 0
  const logs: string[] = []
  return { fx, work, git, calls, write, origin, logs, installedLock, log: (m: string) => { logs.push(m) } }
}

describe("metaPathErrors", () => {
  it("accepts exactly the gate-computed files: data/meta's seven kinds and knowledge/genome.json", () => {
    expect(metaPathErrors(["data/meta/posterior.json", "data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior-index.jsonl",
      "data/meta/verified.json", "data/meta/promotions.jsonl", "data/meta/confirmations/0.json", "data/meta/confirmations/17.json",
      "data/meta/confirmations/999999.json", "knowledge/genome.json"])).toEqual([])
    expect(metaPathErrors([])).toEqual([])
    for (const p of ["data/meta/genome.json", "data/meta/x/posterior.json", "data/meta/posterior.jsonx", "data/meta/../config/evaluation.json",
      "config/evaluation.json", "knowledge/genome-space.json", "knowledge/caregiver-prior.json", "knowledge/genome.jsonx", "xknowledge/genome.json",
      "data/meta/knowledge/genome.json", "data/runs/a/0.json", "data/meta", "xdata/meta/posterior.json", "data/meta/confirmations/00.json",
      "data/meta/confirmations/01.json", "data/meta/confirmations/-1.json", "data/meta/confirmations/1000000.json", "data/meta/confirmations/a.json",
      "data/meta/confirmations/0.json.tmp", "data/meta/confirmations/0/1.json", "data/meta/confirmations", "data/meta/promotions.json"]) {
      expect(META_PATH_RE.test(p)).toBe(false)
      expect(metaPathErrors(["data/meta/posterior.json", p])).toHaveLength(1)
    }
  })
})

describe("commitMeta", () => {
  it("only data/meta files staged → committed and pushed", async () => {
    const t = setup()
    const before = t.origin("rev-parse", "main")
    t.write({ ...META("a"), "data/meta/posterior-index.jsonl": `{"i":1}\n` })
    const rebuild = async () => { throw new Error("no rebuild without a rejected push") }
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).toBe("committed")
    const head = t.origin("rev-parse", "main")
    expect(head).not.toBe(before)
    expect(t.origin("rev-parse", "main^")).toBe(before)
    expect(t.origin("log", "-1", "--format=%an <%ae>|%cn <%ce>|%s", "main")).toBe(
      `github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|meta: calibrate, 3 eligible lives, posterior ${sha8(META("a")["data/meta/posterior.json"])}`)
    expect(t.origin("diff-tree", "--no-commit-id", "--name-only", "-r", "main").split("\n").sort()).toEqual(
      ["data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior-index.jsonl", "data/meta/posterior.json"])
    expect(t.origin("show", "main:data/meta/posterior.json") + "\n").toBe(META("a")["data/meta/posterior.json"])
  })

  for (const other of ["config/evaluation.json", "knowledge/genome-space.json", "knowledge/genome.json", "data/runs/abc/0.json", "data/meta/genome.json"]) {
    it(`a staged ${other} → throws, nothing committed`, async () => {
      const t = setup()
      const before = t.origin("rev-parse", "main")
      const localBefore = t.git(["rev-parse", "HEAD"]).trim()
      t.write({ ...META("a"), [other]: "{}\n" })
      t.git(["add", "--", other])
      await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(other)
      expect(t.origin("rev-parse", "main")).toBe(before)
      expect(t.git(["rev-parse", "HEAD"]).trim()).toBe(localBefore)
      expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
    })
  }

  it("another tracked file modified → throws", async () => {
    const t = setup()
    const before = t.origin("rev-parse", "main")
    t.write({ ...META("a"), "README.md": "changed by npm ci\n" })
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow("README.md")
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  it("an untracked file outside data/meta → throws", async () => {
    const t = setup()
    t.write({ ...META("a"), "stray.txt": "x\n" })
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow("stray.txt")
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  it("a local commit not on origin/main → throws, nothing pushed", async () => {
    const t = setup()
    const before = t.origin("rev-parse", "main")
    t.write({ "README.md": "local\n" })
    t.git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "local"])
    t.write(META("a"))
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/origin\/main/)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "push")).toBe(false)
  })

  it("a push rejected because main moved → fetch, rebuild on the new main, push; never --force", async () => {
    const t = setup()
    // The first build creates the index (untracked): it must not survive into the rebuilt commit.
    t.write({ ...META("stale"), "data/meta/posterior-index.jsonl": `{"stale":1}\n` })
    const moved = t.fx.commitMain({ "data/runs/abc/0.json": "{}\n" }, [], false)
    let rebuilds = 0
    const rebuild = async () => {
      rebuilds++
      // Task 10 again, on the new main: it must see the moved commit's files.
      expect(t.git(["rev-parse", "HEAD"]).trim()).toBe(moved)
      expect(readFileSync(join(t.work, "data/runs/abc/0.json"), "utf8")).toBe("{}\n")
      t.write(META("fresh"))
      return true
    }
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).toBe("committed")
    expect(rebuilds).toBe(1)
    expect(t.origin("rev-parse", "main^")).toBe(moved)
    expect(t.origin("show", "main:data/meta/calibration.json")).toBe(`{"c":"fresh"}`)
    expect(t.origin("ls-tree", "--name-only", "main", "data/meta/").split("\n").sort()).toEqual(
      ["data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior.json"])
    expect(t.origin("log", "-1", "--format=%s", "main")).toBe(`meta: calibrate, 3 eligible lives, posterior ${sha8(META("fresh")["data/meta/posterior.json"])}`)
    const pushes = t.calls.filter(a => a[0] === "push")
    expect(pushes).toEqual([["push", "origin", "HEAD:main"], ["push", "origin", "HEAD:main"]])
    expect(t.calls.filter(a => a[0] === "push").flat().some(isForce)).toBe(false)
  })

  it("main moving on every try → at most 3 pushes, then throws; never --force", async () => {
    const t = setup()
    t.write(META("a"))
    let n = 0
    t.fx.commitMain({ [`data/runs/abc/${n}.json`]: "{}\n" }, [], false)
    let before = ""
    const rebuild = async () => {
      n++
      t.fx.commitMain({ [`data/runs/abc/${n}.json`]: "{}\n" }, [], false) // main moves again before this push
      t.write(META(`r${n}`))
      before = t.origin("rev-parse", "main")
      return true
    }
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).rejects.toThrow(/3 tries/)
    expect(t.calls.filter(a => a[0] === "push")).toHaveLength(3)
    expect(n).toBe(2)
    expect(t.origin("log", "--format=%s", "main").split("\n").some(s => s.startsWith("meta:"))).toBe(false)
    expect(t.origin("rev-list", "--count", `${before}..main`)).toBe("0") // only the fixture's commits landed
    expect(t.calls.filter(a => a[0] === "push").flat().some(isForce)).toBe(false)
  })

  it("a push refused for another reason → throws at once, no retry", async () => {
    const t = setup()
    t.write(META("a"))
    execFileSync("git", ["-C", t.work, "remote", "set-url", "--push", "origin", join(t.work, "../nowhere.git")], { env: ENV })
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/push/)
    expect(t.calls.filter(a => a[0] === "push")).toHaveLength(1)
  })

  it("a rebuild on the new main that changes nothing → unchanged", async () => {
    const t = setup()
    t.write(META("a"))
    t.fx.commitMain(META("a"), [], false) // someone else already committed the same files
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => false, log: t.log })).toBe("unchanged")
    expect(t.origin("log", "-1", "--format=%s", "main")).toBe("c")
  })

  it("main's package-lock.json changed since npm ci (pulled in before the build) → throws, nothing committed", async () => {
    const t = setup()
    t.fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3,\"new\":1}\n" }) // the workflow's git pull brought it
    const before = t.origin("rev-parse", "main")
    t.write(META("a"))
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/package-lock\.json/)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  it("main's package-lock.json changed by the commit that rejected the push → throws before the rebuild, nothing pushed", async () => {
    const t = setup()
    t.write(META("a"))
    const moved = t.fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3,\"new\":1}\n" }, [], false)
    let rebuilds = 0
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => { rebuilds++; return true }, log: t.log })).rejects.toThrow(/package-lock\.json/)
    expect(rebuilds).toBe(0)
    expect(t.origin("rev-parse", "main")).toBe(moved)
    expect(t.calls.filter(a => a[0] === "push")).toHaveLength(1)
  })

  it("a tracked data/meta file deleted → throws, nothing committed", async () => {
    const t = setup()
    t.fx.commitMain(META("a"))
    const before = t.origin("rev-parse", "main")
    t.write({ "data/meta/calibration.json": `{"c":"b"}\n` })
    rmSync(join(t.work, "data/meta/diagnostics.json"))
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/diagnostics\.json" is D/)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  for (const [what, make] of [
    ["a tracked data/meta file turned into a symlink", (w: string) => { rmSync(join(w, "data/meta/posterior.json")); symlinkSync("../../README.md", join(w, "data/meta/posterior.json")) }],
    ["a new data/meta file that is a symlink", (w: string) => { symlinkSync("../../README.md", join(w, "data/meta/posterior-index.jsonl")) }],
    ["a data/meta file made executable", (w: string) => { chmodSync(join(w, "data/meta/calibration.json"), 0o755) }],
  ] as const) {
    it(`${what} → throws, nothing committed`, async () => {
      const t = setup()
      t.fx.commitMain(META("a"))
      const before = t.origin("rev-parse", "main")
      t.git(["config", "core.fileMode", "true"])
      make(t.work)
      await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/not a regular file added or modified/)
      expect(t.origin("rev-parse", "main")).toBe(before)
      expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
    })
  }

  it("nothing changed → unchanged, no commit", async () => {
    const t = setup()
    t.fx.commitMain(META("a"))
    const before = t.origin("rev-parse", "main")
    t.write(META("a"))
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).toBe("unchanged")
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })
})

// Each case runs real git against a bare origin (several setups in some): give them room when the suite runs in parallel.
describe("commitMeta: confirmations, promotions and knowledge/genome.json (plan 05b Task 10)", { timeout: 30_000 }, () => {
  /** commitMeta must refuse: it throws `msg`, nothing is committed or pushed, and main (and its genome.json) is unchanged. */
  async function refused(t: ReturnType<typeof setup>, msg: RegExp) {
    const before = t.origin("rev-parse", "main")
    const localBefore = t.git(["rev-parse", "HEAD"]).trim()
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(msg)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.git(["rev-parse", "HEAD"]).trim()).toBe(localBefore)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
    // main's genome.json (or its absence) is what it was.
    expect(t.origin("ls-tree", "main", "--", GENOME)).toBe(t.git(["ls-tree", localBefore, "--", GENOME]).trim())
  }
  const commit = (t: ReturnType<typeof setup>) =>
    commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => { throw new Error("no rebuild without a rejected push") }, log: t.log })

  it("a declaration commit (confirmations/0.json added, posterior, verified) → committed", async () => {
    const t = setup()
    t.fx.commitMain(SEARCH("a"))
    t.write({ ...SEARCH("b"), [C0]: confirmationText(0, null) })
    expect(await commit(t)).toBe("committed")
    expect(t.origin("diff-tree", "--no-commit-id", "--name-only", "-r", "main").split("\n").sort()).toEqual(
      [C0, "data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior.json", "data/meta/verified.json"].sort())
    expect(t.origin("log", "-1", "--format=%s", "main")).toBe(
      `meta: search, 140 eligible lives, posterior ${sha8(SEARCH("b")["data/meta/posterior.json"])}; declare confirmation 0`)
  })

  it("a promotion commit (genome.json modified + confirmations/0.json result pass + one promotions line) → committed; message ends '; promote <sha8>'", async () => {
    const t = setup()
    t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
    const knowledgeBefore = computeKnowledgeSha(join(t.work, "knowledge"))
    t.write(PROMOTION())
    expect(await commit(t)).toBe("committed")
    expect(t.origin("diff-tree", "--no-commit-id", "--name-only", "-r", "main").split("\n").sort()).toEqual(
      [C0, "data/meta/calibration.json", "data/meta/diagnostics.json", PROMOTIONS, "data/meta/posterior.json", "data/meta/verified.json", GENOME].sort())
    const subject = t.origin("log", "-1", "--format=%s", "main")
    expect(subject).toBe(`meta: search, 140 eligible lives, posterior ${sha8(PROMOTION()["data/meta/posterior.json"])}; confirmation 0 pass; promote ${shaOf(CANDIDATE).slice(0, 8)}`)
    expect(subject.endsWith(`; promote ${shaOf(CANDIDATE).slice(0, 8)}`)).toBe(true)
    expect(t.origin("show", `main:${GENOME}`) + "\n").toBe(promotedGenome())
    // The promotion is new knowledge: deploy.yml copies knowledge/*.json, so the deployed knowledgeSha moves (tabs reload, 0014).
    expect(computeKnowledgeSha(join(t.work, "knowledge"))).not.toBe(knowledgeBefore)
  })

  it("a confirmation decided without a pass (fail) → committed, no genome.json; message '; confirmation 0 fail'", async () => {
    const t = setup()
    t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
    t.write({ ...SEARCH("b"), [C0]: confirmationText(0, "fail") })
    expect(await commit(t)).toBe("committed")
    expect(t.origin("log", "-1", "--format=%s", "main")).toMatch(/; confirmation 0 fail$/)
  })

  it("a promotion whose push main moved past → reset drops the promoted genome.json, the rebuild's own promotion is pushed", async () => {
    const t = setup()
    t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
    t.write(PROMOTION("stale"))
    const moved = t.fx.commitMain({ "data/runs/abc/0.json": "{}\n" }, [], false)
    let rebuilds = 0
    const rebuild = async () => {
      rebuilds++
      expect(t.git(["rev-parse", "HEAD"]).trim()).toBe(moved)
      expect(readFileSync(join(t.work, GENOME), "utf8")).toBe(GENOME_JSON) // reset to the new main's genome.json
      t.write(PROMOTION("fresh"))
      return true
    }
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).toBe("committed")
    expect(rebuilds).toBe(1)
    expect(t.origin("rev-parse", "main^")).toBe(moved)
    expect(t.origin("show", "main:data/meta/calibration.json")).toBe(`{"c":"fresh"}`)
    expect(t.calls.filter(a => a[0] === "push").flat().some(isForce)).toBe(false)
  })

  describe("genome.json staged without a passing confirmation, or with values ≠ the candidate, or without a promotions line → throws, nothing committed", () => {
    const cases: [string, Record<string, string>, RegExp][] = [
      ["the confirmation still open (not staged)", { [C0]: confirmationText(0, null) }, /exactly one/],
      ["the confirmation failed", { [C0]: confirmationText(0, "fail") }, /pass/],
      ["the confirmation incomplete", { [C0]: confirmationText(0, "incomplete") }, /pass/],
      ["values ≠ the candidate", { [GENOME]: promotedGenome({ ...CANDIDATE, lr: 1.5e-3 }) }, /candidate/],
      ["values outside the space", { [GENOME]: promotedGenome({ ...CANDIDATE, lr: 1 }) }, /knowledge\/genome\.json/],
      ["provenance naming another confirmation", { [GENOME]: promotedGenome(CANDIDATE, 1) }, /provenance\.confirmation/],
      ["provenance source initial", { [GENOME]: serialise({ ...JSON.parse(promotedGenome()), provenance: { source: "initial" } }) }, /provenance/],
      ["not JSON", { [GENOME]: "{ no" }, /knowledge\/genome\.json/],
      ["no promotions line", { [PROMOTIONS]: "" }, /promotions\.jsonl/],
      ["a promotions line naming another confirmation", { [PROMOTIONS]: promotionLine(1) }, /promotions\.jsonl/],
      ["a promotions line naming another genome", { [PROMOTIONS]: promotionLine(0, "d".repeat(64)) }, /promotions\.jsonl/],
      ["two promotions lines", { [PROMOTIONS]: promotionLine() + promotionLine() }, /promotions\.jsonl/],
    ]
    for (const [what, over, msg] of cases) {
      it(what, async () => {
        const t = setup()
        t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
        const files: Record<string, string> = { ...PROMOTION(), ...over }
        if (over[PROMOTIONS] === "") delete files[PROMOTIONS]
        t.write(files)
        await refused(t, msg)
      })
    }
    it("a passing confirmation without genome.json (the pass would be lost) → throws", async () => {
      const t = setup()
      t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
      t.write({ ...SEARCH("b"), [C0]: confirmationText(0, "pass"), [PROMOTIONS]: promotionLine() })
      await refused(t, /knowledge\/genome\.json/)
    })
    it("a promotions line without genome.json → throws", async () => {
      const t = setup()
      t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
      t.write({ ...SEARCH("b"), [PROMOTIONS]: promotionLine() })
      await refused(t, /promotions\.jsonl/)
    })
    it("a promotion together with a declaration → throws", async () => {
      const t = setup()
      t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
      t.write({ ...PROMOTION(), "data/meta/confirmations/1.json": confirmationText(1, null) })
      await refused(t, /exactly one/)
    })
    it("knowledge/genome.json added (main has none) → throws: it may only be modified", async () => {
      const t = setup()
      t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) }, [GENOME])
      t.write(PROMOTION())
      await refused(t, /knowledge\/genome\.json" is A/)
    })
    it("knowledge/genome.json deleted → throws", async () => {
      const t = setup()
      t.fx.commitMain(SEARCH("a"))
      t.write(SEARCH("b"))
      rmSync(join(t.work, GENOME))
      await refused(t, /knowledge\/genome\.json" is D/)
    })
  })

  describe("a confirmation file whose candidate changed, or whose existing result changed → throws", () => {
    const cases: [string, string | null, string, RegExp][] = [
      ["the candidate changed with the result", null, confirmationText(0, "fail", { candidate: { genomeSha: shaOf({ ...CANDIDATE, lr: 1.5e-3 }), values: { ...CANDIDATE, lr: 1.5e-3 } } }), /only its result/],
      ["the deadline changed, still open", null, confirmationText(0, null, { declaredAt: "2026-11-02T10:00:00.000Z", deadline: "2026-11-23T10:00:00.000Z" }), /only its result/],
      ["an existing result changed (fail → pass)", "fail", confirmationText(0, "pass"), /already has a result/],
      ["an existing result removed", "fail", confirmationText(0, null), /already has a result/],
      ["not a confirmation", null, "{}\n", /confirmations\/0\.json/],
    ]
    for (const [what, headOutcome, staged, msg] of cases) {
      it(what, async () => {
        const t = setup()
        t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, headOutcome) })
        t.write({ ...SEARCH("b"), [C0]: staged })
        await refused(t, msg)
      })
    }
    it("a new confirmation file declared with a result, or under another id → throws", async () => {
      for (const [path, text] of [[C0, confirmationText(0, "fail")], [C0, confirmationText(1, null)]] as const) {
        const t = setup()
        t.fx.commitMain(SEARCH("a"))
        t.write({ ...SEARCH("b"), [path]: text })
        await refused(t, /confirmations\/0\.json/)
        t.fx.cleanup()
      }
    })
  })

  it("promotions.jsonl with a rewritten earlier line → throws", async () => {
    const t = setup()
    const earlier = JSON.stringify({ confirmation: 7, genomeSha: "e".repeat(64), previous: "f".repeat(64), promotedAt: AT, sourceCommit: "c".repeat(40) }) + "\n"
    t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null), [PROMOTIONS]: earlier })
    t.write({ ...PROMOTION(), [PROMOTIONS]: earlier.replace("7", "8") + promotionLine() })
    await refused(t, /promotions\.jsonl.*append/)
  })

  it("posterior-index.jsonl with a rewritten or dropped earlier line, or no final newline → throws", async () => {
    for (const staged of [`{"i":2}\n{"i":3}\n`, `{"i":2}\n`, `{"i":1}\n{"i":2}\n{"i":3}`]) {
      const t = setup()
      t.fx.commitMain({ ...SEARCH("a"), "data/meta/posterior-index.jsonl": `{"i":1}\n{"i":2}\n` })
      t.write({ ...SEARCH("b"), "data/meta/posterior-index.jsonl": staged })
      await refused(t, /posterior-index\.jsonl.*append/)
      t.fx.cleanup()
    }
  })

  it("lockfile moved between npm ci and a promotion commit → throws before committing", async () => {
    const t = setup()
    t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
    t.fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3,\"new\":1}\n" }) // pulled in before the build
    t.write(PROMOTION())
    await refused(t, /package-lock\.json/)
  })

  it("lockfile moved by the commit that rejected a promotion's push → throws before the rebuild; main keeps its genome.json", async () => {
    const t = setup()
    t.fx.commitMain({ ...SEARCH("a"), [C0]: confirmationText(0, null) })
    t.write(PROMOTION())
    const moved = t.fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3,\"new\":1}\n" }, [], false)
    let rebuilds = 0
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => { rebuilds++; return true }, log: t.log })).rejects.toThrow(/package-lock\.json/)
    expect(rebuilds).toBe(0)
    expect(t.origin("rev-parse", "main")).toBe(moved)
    expect(t.origin("show", `main:${GENOME}`) + "\n").toBe(GENOME_JSON)
  })
})

describe("promotionErrors", () => {
  const files = (over: Record<string, string | null> = {}) => {
    const staged: Record<string, string | null> = { ...PROMOTION(), ...over }
    const head: Record<string, string> = { ...SEARCH("a"), [C0]: confirmationText(0, null), [GENOME]: GENOME_JSON, "knowledge/genome-space.json": SPACE_JSON }
    const show = (p: string) => staged[p] ?? head[p] ?? null
    return { show, head: (p: string) => head[p] ?? null, staged: Object.keys(staged).filter(p => staged[p] !== null) }
  }
  it("a promotion with its passing confirmation and its line → no errors; nothing staged → no errors", () => {
    const f = files()
    expect(promotionErrors(f.show, f.staged, f.head)).toEqual([])
    expect(promotionErrors(f.show, [], f.head)).toEqual([])
  })
  it("genome.json staged with the wrong values → an error naming the candidate", () => {
    const f = files({ [GENOME]: promotedGenome({ ...CANDIDATE, lr: 1.5e-3 }) })
    expect(promotionErrors(f.show, f.staged, f.head).join("; ")).toMatch(/candidate/)
  })
  it("genome.json staged when main's genome.json is not the confirmation's centre → an error", () => {
    const f = files()
    const head = (p: string) => (p === GENOME ? promotedGenome({ ...CENTRE, lr: 2e-4 }, 5) : f.head(p))
    expect(promotionErrors(f.show, f.staged, head).join("; ")).toMatch(/centre/)
  })
})

describe("workflows", () => {
  const gate = readFileSync(join(REPO, ".github/workflows/gate.yml"), "utf8")
  const deploy = readFileSync(join(REPO, ".github/workflows/deploy.yml"), "utf8")
  it("gate.yml's deploy condition includes meta == 'committed'; deploy.yml copies data/meta/posterior.json", () => {
    const deployJob = gate.slice(gate.indexOf("\n  deploy:\n"))
    expect(deployJob).toMatch(/^ {4}needs: gate\n {4}if: \$\{\{ !cancelled\(\) && \(\(needs\.gate\.outputs\.merged != '' && needs\.gate\.outputs\.merged != '0'\) \|\| needs\.gate\.outputs\.meta == 'committed'\) \}\}\n {4}uses: \.\/\.github\/workflows\/deploy\.yml$/m)
    expect(gate).toMatch(/meta: \$\{\{ steps\.meta\.outputs\.meta \}\}/)
    expect(deploy).toContain("if [ -f data/meta/posterior.json ]; then cp data/meta/posterior.json app/dist/knowledge/posterior.json; fi")
  })
  it("gate.yml pulls main, builds data/meta and commits it after the gate step, with only contents and pull-requests write", () => {
    const at = (s: string) => { const i = gate.indexOf(s); expect(i, s).toBeGreaterThan(-1); return i }
    const order = [at("run: npm ci"), at(`LOCK_BLOB=$(git hash-object package-lock.json)`), at("npx tsx tools/gate/gate.ts"), at("git pull --ff-only origin main"),
      at(`"$(git rev-parse HEAD:package-lock.json)" = "$LOCK_BLOB"`), at("npx tsx tools/gate/meta-cli.ts"), at("npx tsx tools/gate/commit-meta.ts")]
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    const gateJob = gate.slice(gate.indexOf("  gate:"), gate.indexOf("  deploy:"))
    expect(gateJob).toMatch(/permissions:\n\s+contents: write\n\s+pull-requests: write\n\s+outputs:/)
    expect(gate).not.toMatch(/--force|push -f/)
  })
  it("the gate stages and commits knowledge/genome.json only through commit-meta.ts: no git add, commit or push in gate.yml or any other gate tool", () => {
    const steps = gate.split("\n").filter(l => !/^\s*#/.test(l)).join("\n")
    expect(steps).not.toMatch(/git (add|commit|push)|knowledge\//)
    const tools = execFileSync("git", ["ls-files", "--", "tools/**/*.ts"], { cwd: REPO, encoding: "utf8" }).split("\n")
      .filter(f => f !== "" && !f.includes("/test/") && !f.endsWith(".test.ts"))
    const writers = tools.filter(f => /\[\s*"(add|commit|push)"/.test(readFileSync(join(REPO, f), "utf8")))
    expect(writers).toEqual(["tools/gate/commit-meta.ts"])
    // The deploy the gate calls on meta == 'committed' copies every knowledge file, so a promotion reaches the page.
    expect(deploy).toContain("cp knowledge/*.json app/dist/knowledge/")
    expect(gate).toMatch(/promotion/)
  })
})
