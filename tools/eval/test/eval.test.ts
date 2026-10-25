import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { execFileSync } from "node:child_process"
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { CONFIG_BUDGET, MAX_EVAL_PANEL, NEWBORN_LEVELS, type LogFile, type LogFileV2 } from "@blanktab/harness/log"
import { NEWBORN_SEED_BASE } from "@blanktab/harness/newborn"
import { decodeGenome, genomeSha, parseCentre, parseGenomeSpace } from "@blanktab/meta"
import { parseNewbornReference } from "@blanktab/evaluators"
import { scoredLife, lid } from "../../../packages/evaluators/test/fixtures/build.ts"
import { loadEvalConfig, loadReferences } from "../config.ts"
import { collectLives } from "../lives.ts"
import { budgetWindow } from "../../../packages/evaluators/src/life.ts"
import { rescore } from "../rescore.ts"
import { newbornReference } from "../newborn-reference.ts"
import { rangeOf, writeCollected } from "../collect-newborns.ts"
import { serialise } from "../../gate/serialise.ts"
import { FIXTURE_ENV } from "../../gate/test/git-fixture.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
const SPACE = parseGenomeSpace(JSON.parse(readFileSync(join(REPO, "knowledge/genome-space.json"), "utf8")))
const CENTRE = parseCentre(JSON.parse(readFileSync(join(REPO, "knowledge/genome.json"), "utf8")), SPACE)
const GPU = { conv1: 16, conv2: 32, enc: 128, hidden: 256 }
const CODE = "c".repeat(40)

let root: string
let runs: string
/** git in the fixture checkout (isolated from the user's and the system's git config). */
const git = (...args: string[]) => execFileSync("git", args, { cwd: root, encoding: "utf8", env: { ...process.env, ...FIXTURE_ENV }, stdio: ["ignore", "pipe", "pipe"] })
beforeEach(() => {
  // A checkout (a git repository: newborn-reference reads knowledge/genome.json's history) with the pinned config and
  // the genome files committed, and nothing else.
  root = mkdtempSync(join(tmpdir(), "eval-"))
  for (const rel of ["config/evaluation.json", "knowledge/genome-space.json", "knowledge/genome.json"]) {
    mkdirSync(join(root, rel, ".."), { recursive: true })
    copyFileSync(join(REPO, rel), join(root, rel))
  }
  git("init", "-q", "-b", "main")
  git("add", "--", "config", "knowledge")
  git("commit", "-q", "-m", "initial")
  runs = join(root, "runs")
})
afterEach(() => { rmSync(root, { recursive: true, force: true }) })
const put = (dir: string, f: LogFile) => {
  mkdirSync(join(dir, f.learnerId), { recursive: true })
  writeFileSync(join(dir, f.learnerId, `${f.seq}.json`), JSON.stringify(f, null, 2))
}
const refPath = () => join(root, loadEvalConfig(root).references.newborn!)

/** Newborn k's file as the page writes it (runNewborn's shape), at GPU widths: the centre genome, its panel and seed. */
async function newbornFile(k: number, m3: number): Promise<LogFileV2> {
  const d = decodeGenome(CENTRE, SPACE)
  const ck = (level: number, i: number) => ({
    level, steps: 0, passed: m3 + i >= 2 ? ["M3"] : [], evalCU: 20,
    measures: { M1: 0.9, M2: 0.5 + k / 100, M3: m3 + i, M4: 0.1, M5: 0.6, M6: 0 },
  })
  return {
    schemaVersion: 2, learnerId: lid(0x1000 + k), seq: 0, codeSha: CODE, knowledgeSha: "e".repeat(64),
    writtenAt: `2026-10-2${k % 2 === 0 ? 3 : 4}T10:00:00.000Z`, stepsTotal: 0,
    settings: { ...d.learner, maxStepsPerSec: d.harness.maxStepsPerSec, seed: NEWBORN_SEED_BASE + k, ...d.caregiver },
    hostSig: { cores: 10, deviceMemoryGb: 8, webgpu: true, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: true,
    cu: 0, condition: "A", widths: GPU, curve: [], skippedUpdates: 0, restores: 0, lostCU: 0, evalCU: 61,
    checkups: NEWBORN_LEVELS.map(ck), milestoneSuiteVersion: 1,
    meta: {
      v: 1, role: "newborn", method: "newborn", genomeSchema: SPACE.schemaVersion, genome: { ...CENTRE }, genomeSha: await genomeSha(SPACE.schemaVersion, CENTRE),
      drawSeed: null, posteriorSha: null, evalPanel: 1 + (k % MAX_EVAL_PANEL), evalSeed: null, confirmation: null, inheritance: { filters: "none" },
      heldOut: { loss: 1.01 + k / 1000, baseline: 1 }, budget: { budgetCU: CONFIG_BUDGET.budgetCU, levels: [...CONFIG_BUDGET.levels] },
    },
  }
}
const PROV = { browser: "Chrome 141", machine: "Apple M2 laptop, 16 GB" }
/** Writing a reference from fewer than 64 newborns (as every test here does) needs allowMissing. */
const W = { check: false, allowMissing: true, ...PROV }

describe("collectLives", () => {
  it("every learner dir, its files parsed and sorted by seq; junk skipped", () => {
    for (const f of scoredLife(lid(7), { ms: {}, ratio: 1 }).reverse()) put(runs, f)
    mkdirSync(join(runs, "junk"))
    writeFileSync(join(runs, lid(7), "heartbeat.json"), "{}")
    const lives = collectLives(runs)
    expect(lives.map(l => l.learnerId)).toEqual([lid(7)])
    expect(lives[0]!.files.map(f => f.seq)).toEqual([0, 1, 2, 3])
    expect(collectLives(join(root, "nope"))).toEqual([])
  })
})

describe("rescore (spec §2.2)", () => {
  it("rescore writes one row per learner with the evaluator's id@version in the header", () => {
    const ref = JSON.parse(readFileSync(join(REPO, "packages/evaluators/test/fixtures/reference.json"), "utf8"))
    mkdirSync(join(refPath(), ".."), { recursive: true })
    writeFileSync(refPath(), JSON.stringify(ref))
    const S = lid(1), T = lid(2)
    for (const f of scoredLife(S, { ms: { M2: 1, M3: 2, M4: 0.25, M5: 0.7, M6: 0 }, ratio: 0.9 })) put(runs, f)
    for (const f of scoredLife(T, { ms: { M2: 1 }, ratio: 0.9 }).slice(0, 2)) put(runs, f) // no 2048 checkup yet
    const out = rescore("search-composite@1", runs, root)
    expect(out.path).toBe(join(root, ".local", "rescore-search-composite@1.tsv"))
    const lines = readFileSync(out.path, "utf8").replace(/\n$/, "").split("\n").map(l => l.split("\t"))
    expect(lines[0]).toEqual([
      "learnerId", "role", "method", "genomeSha",
      "search-composite@1:eligible", "search-composite@1:reason", "search-composite@1:score", "search-composite@1:C", "search-composite@1:H",
      "pinned:search-composite@1:score",
    ])
    expect(lines).toHaveLength(3)
    expect(lines.slice(1).map(r => r[0])).toEqual([S, T])
    const [s, t] = [lines[1]!, lines[2]!]
    expect(s.slice(1, 6)).toEqual(["search", "calibrate", "ab".repeat(32), "true", ""])
    expect(Number(s[6])).toBeCloseTo(Number(s[7]) + Number(s[8]), 12)
    expect(s[9]).toBe(s[6]) // the pinned one is this one
    expect(t.slice(4, 7)).toEqual(["false", "no 2048 checkup", ""])
    // A cohort evaluator scores no single life; an unknown one is refused.
    expect(() => rescore("newborn-baseline@1", runs, root)).toThrow(/life evaluator/)
    expect(() => rescore("nope@1", runs, root)).toThrow(/unknown evaluator/)
  })
})

describe("newborn-reference (spec §2.3; controller ruling B)", () => {
  it("newborn-reference --check passes on its own output and fails after one measure changes", async () => {
    const src = join(root, "newborns")
    for (let k = 0; k < 4; k++) put(src, await newbornFile(k, k % 2 === 0 ? 1 : 2))
    // Writing needs the provenance a log cannot give: the browser and the machine.
    expect((await newbornReference({ root, runsDir: src, check: false })).ok).toBe(false)
    expect(existsSync(refPath())).toBe(false)
    // Fewer than 64 newborns: refused unless allowMissing says so.
    const short = await newbornReference({ root, runsDir: src, check: false, ...PROV })
    expect(short.ok).toBe(false)
    expect(short.messages.join("\n")).toMatch(/60 newborns are missing: pass --allow-missing/)
    expect(short.messages.join("\n")).toMatch(/missing newborns k = 4, 5, /)
    expect(existsSync(refPath())).toBe(false)
    const built = await newbornReference({ root, runsDir: src, ...W })
    expect(built.messages.join("\n")).not.toMatch(/error/i)
    expect(built.ok).toBe(true)
    // The file the config pins: a reference the search score reads, naming its lives, with its provenance.
    const ref = loadReferences(root, loadEvalConfig(root)).newborn!
    expect(ref.learnerIds).toEqual([0, 1, 2, 3].map(k => lid(0x1000 + k)))
    expect(ref.milestones.M3).toEqual({ median: 2.5, passRate: 10 / 12, sustainedPassRate: 1 })
    expect(ref.provenance).toEqual({ codeSha: CODE, date: "2026-10-24", ...PROV })
    expect(readFileSync(refPath(), "utf8").endsWith("}\n")).toBe(true)

    expect((await newbornReference({ root, runsDir: src, check: true })).ok).toBe(true)
    // One measure changes (newborn 0's M3 at 512, 1 → 3; what it passed is unchanged): the M3 median moves from 2.5
    // to 3, so the committed file no longer matches its lives.
    const f = await newbornFile(0, 1)
    f.checkups![0]!.measures.M3 = 3
    put(src, f)
    const changed = await newbornReference({ root, runsDir: src, check: true })
    expect(changed.ok).toBe(false)
    expect(changed.messages.join("\n")).toMatch(/differs/)
  })

  it("refuses files the gate would refuse, and newborns of another genome, seed or panel", async () => {
    const src = join(root, "newborns")
    const good = await newbornFile(0, 1)
    put(src, good)
    expect((await newbornReference({ root, runsDir: src, ...W })).ok).toBe(true)
    const cases: [string, (f: LogFileV2) => void, RegExp][] = [
      ["a settings value the genome does not run", f => { f.settings.lr = 1 }, /not what ran/],
      ["a wrong genomeSha", f => { f.meta!.genomeSha = "0".repeat(64) }, /genomeSha/],
      ["a panel that is not 1 + k mod 16", f => { f.meta!.evalPanel = 2 }, /panel/],
      ["a seed outside 1000…1063", f => { f.settings.seed = 999 }, /seed/],
      ["not a newborn", f => { f.meta = { ...f.meta!, role: "search", method: "centre" }; delete f.meta.heldOut; f.checkups = [] }, /newborn/],
    ]
    for (const [what, edit, msg] of cases) {
      const f = structuredClone(good)
      edit(f)
      put(src, f)
      const r = await newbornReference({ root, runsDir: src, ...W })
      expect(r.ok, what).toBe(false)
      expect(r.messages.join("\n"), what).toMatch(msg)
    }
    // A duplicate k (two lives on one seed) is refused too.
    put(src, good)
    put(src, { ...structuredClone(good), learnerId: lid(0x2000) })
    const dup = await newbornReference({ root, runsDir: src, ...W })
    expect(dup.ok).toBe(false)
    expect(dup.messages.join("\n")).toMatch(/seed 1000/)
    expect(parseNewbornReference(JSON.parse(readFileSync(refPath(), "utf8")))).not.toBeNull() // the last good write stands
  })

  it("newborn-reference --check passes when genome.json has been promoted (fixture)", async () => {
    // The reference was measured under the initial centre; then a confirmation passed and the gate promoted a candidate.
    const src = join(root, "newborns")
    for (let k = 0; k < 4; k++) put(src, await newbornFile(k, k % 2 === 0 ? 1 : 2))
    expect((await newbornReference({ root, runsDir: src, ...W })).ok).toBe(true)
    const promote = (values: Record<string, number | string>) => {
      writeFileSync(join(root, "knowledge/genome.json"), serialise({ schemaVersion: 1, genomeSchema: SPACE.schemaVersion, values, provenance: { source: "promotion", confirmation: 0 } }))
      git("commit", "-q", "-am", "meta: promote")
    }
    promote({ ...CENTRE, lr: 1e-3 })
    const r = await newbornReference({ root, runsDir: src, check: true })
    expect(r.messages.join("\n")).not.toMatch(/error/)
    expect(r.ok).toBe(true)
    promote({ ...CENTRE, lr: 1.5e-3 }) // a second promotion: the newborns' centre is two genomes back, still in the history
    expect((await newbornReference({ root, runsDir: src, check: true })).ok).toBe(true)
  })

  it("refuses newborns of a genome knowledge/genome.json never was, and newborns of two different centres", async () => {
    const src = join(root, "newborns")
    const other = { ...CENTRE, lr: 1e-3 }
    const of = async (k: number, values: typeof CENTRE) => {
      const f = await newbornFile(k, 1)
      const d = decodeGenome(values, SPACE)
      f.settings = { ...f.settings, ...d.learner }
      f.meta = { ...f.meta!, genome: { ...values }, genomeSha: await genomeSha(SPACE.schemaVersion, values) }
      return f
    }
    put(src, await of(0, other))
    const never = await newbornReference({ root, runsDir: src, ...W })
    expect(never.ok).toBe(false)
    expect(never.messages.join("\n")).toMatch(/not a centre knowledge\/genome\.json has had/)
    // Promoted later, that genome is a centre; but a reference is one centre's newborns.
    writeFileSync(join(root, "knowledge/genome.json"), serialise({ schemaVersion: 1, genomeSchema: SPACE.schemaVersion, values: other, provenance: { source: "promotion", confirmation: 0 } }))
    git("commit", "-q", "-am", "meta: promote")
    expect((await newbornReference({ root, runsDir: src, ...W })).ok).toBe(true)
    put(src, await of(1, CENTRE))
    const mixed = await newbornReference({ root, runsDir: src, ...W })
    expect(mixed.ok).toBe(false)
    expect(mixed.messages.join("\n")).toMatch(/2 centres/)
  })

  it("never throws: a stray file named like a learner, a runsDir that is a file, a committed file that is not JSON, a broken config", async () => {
    const src = join(root, "newborns")
    put(src, await newbornFile(0, 1))
    writeFileSync(join(src, lid(0x3000)), "not a dir")
    const stray = await newbornReference({ root, runsDir: src, ...W })
    expect(stray.ok).toBe(false)
    expect(stray.messages.join("\n")).toMatch(/not a directory/)
    rmSync(join(src, lid(0x3000)))
    expect((await newbornReference({ root, runsDir: join(src, lid(0x1000), "0.json"), ...W })).messages.join("\n")).toMatch(/no such directory/)
    expect((await newbornReference({ root, runsDir: src, ...W })).ok).toBe(true)
    writeFileSync(refPath(), "{ not json")
    const bad = await newbornReference({ root, runsDir: src, check: true })
    expect(bad.ok).toBe(false)
    expect(bad.messages.join("\n")).toMatch(/is not JSON/)
    writeFileSync(join(root, "config/evaluation.json"), "{}")
    const broken = await newbornReference({ root, runsDir: src, ...W })
    expect(broken.ok).toBe(false)
    expect(broken.messages.join("\n")).toMatch(/^error: config\/evaluation.json/)
  })
})

describe("collect-newborns (controller ruling B: the outbox's newborn files, checked as the gate would)", () => {
  it("writes each newborn file once as <learnerId>/<seq>.json, skips other learners', leaves identical bytes and refuses to overwrite", async () => {
    const out = join(root, "newborns")
    const nb = await newbornFile(3, 1)
    const life = scoredLife(lid(9), { ms: {}, ratio: 1 })[0]!
    const rows = [{ learnerId: nb.learnerId, seq: 0, file: nb }, { learnerId: life.learnerId, seq: life.seq, file: life }]
    const first = await writeCollected(rows, out, SPACE)
    expect(first).toEqual({ written: [`${nb.learnerId}/0.json`], same: [], skipped: 1, errors: [] })
    const path = join(out, nb.learnerId, "0.json")
    expect(readFileSync(path, "utf8")).toBe(JSON.stringify(nb, null, 2) + "\n")
    expect((await writeCollected(rows, out, SPACE)).same).toEqual([`${nb.learnerId}/0.json`])
    // The collected files are what newborn-reference reads.
    expect(collectLives(out).map(l => l.learnerId)).toEqual([nb.learnerId])
    writeFileSync(path, "{}\n")
    const again = await writeCollected(rows, out, SPACE)
    expect(again.errors).toEqual([expect.stringMatching(/already exists with other bytes/)])
    expect(readFileSync(path, "utf8")).toBe("{}\n")
  })
  it("refuses a file the gate would refuse (its genome outside the space), and writes nothing for it", async () => {
    const nb = await newbornFile(4, 1)
    const bad = { ...nb, meta: { ...nb.meta!, genome: { ...nb.meta!.genome, lr: 1 } } }
    const r = await writeCollected([{ learnerId: nb.learnerId, seq: 0, file: bad }], join(root, "newborns"), SPACE)
    expect(r.written).toEqual([])
    expect(r.errors.length).toBeGreaterThan(0)
    expect(existsSync(join(root, "newborns", nb.learnerId))).toBe(false)
  })
  it("rangeOf reads a-b within the reference's 64 newborns", () => {
    expect(rangeOf("0-63")).toEqual({ from: 0, to: 63 })
    for (const s of ["5-4", "0-64", "x", "1"]) expect(() => rangeOf(s), s).toThrow(/--newborns/)
  })
})

describe("the committed lives in data/runs (final review: real log layout)", () => {
  it("every checkup a real life logged is inside budgetWindow at its level, though its crossing file is a fraction over it", () => {
    const lives = collectLives(join(REPO, "data/runs"))
    let checked = 0, later = 0
    for (const life of lives) {
      for (const f of life.files) {
        if (f.schemaVersion !== 2) continue
        for (const c of f.checkups ?? []) {
          const window = budgetWindow(life.files, c.level)
          expect(window.some(w => w.seq === f.seq), `${life.learnerId} level ${c.level}`).toBe(true)
          checked++
          const crossing = life.files.find(x => x.schemaVersion === 2 && x.cu >= c.level)!
          if (crossing.seq < f.seq) later++
        }
      }
    }
    // The committed data has such lives (a checkup in a file after its crossing file): the guard is not vacuous.
    expect(checked).toBeGreaterThan(0)
    expect(later).toBeGreaterThan(0)
  })
})
