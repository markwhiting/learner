// The end-to-end meta-learner simulation (plan 05b Task 12) on a local fixture repo: the real meta-cli.ts and
// commit-meta.ts against a bare origin, lives written as a session writes them. Search is reached by data only.
import { execFileSync } from "node:child_process"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { parsePosterior, type Posterior } from "@blanktab/meta"
import { META_PATH_RE } from "../../gate/commit-meta.ts"
import { runSim, type SimReport } from "../meta-sim.ts"

const dirs: string[] = []
afterEach(() => { for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }) })
const fresh = () => { const d = mkdtempSync(join(tmpdir(), "meta-sim-")); dirs.push(d); return d }

/** git in the fixture's bare origin: what main is. */
const originOf = (dir: string) => (...args: string[]) =>
  execFileSync("git", ["--git-dir", join(dir, "origin.git"), ...args], { encoding: "utf8", env: { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" } })
const posteriorAt = (git: ReturnType<typeof originOf>, commit: string): Posterior => {
  const p = parsePosterior(git("show", `${commit}:data/meta/posterior.json`))
  if (p === null) throw new Error(`the posterior at ${commit} does not parse`)
  return p
}
const touches = (r: SimReport, path: string) => r.stages.filter(s => s.paths.includes(path)).map(s => s.name)

describe("meta-learner simulation (plan 05b Task 12)", () => {
  it("quick sim: calibrate → signal → search → declare → pass → promote; knowledgeSha changes once; every gate commit touches only META_PATH_RE paths; no stage before signal writes a model", async () => {
    const dir = fresh()
    const r = await runSim({ dir, quick: true, log: () => {} })
    const git = originOf(dir)
    const names = r.stages.map(s => s.name)
    // The stages, in order, and the mode each gate run left main in.
    expect(names[0]).toBe("start")
    const firstSearch = r.stages.findIndex(s => s.mode === "search")
    expect(firstSearch).toBeGreaterThan(0)
    expect(r.stages.slice(0, firstSearch).every(s => s.mode === "calibrate")).toBe(true)
    expect(r.stages.slice(firstSearch).every(s => s.mode === "search")).toBe(true)
    expect(names.at(-1)).toBe("after promotion")
    // Every gate commit: only the gate's files.
    for (const s of r.stages) for (const p of s.paths) expect(p, `${s.name}: ${p}`).toMatch(META_PATH_RE)
    // Before signal: no model, incumbent or confirmation; the signal build serves a model (fitted on the calibration rows).
    for (const s of r.stages.slice(0, firstSearch)) {
      const p = posteriorAt(git, s.commit)
      expect(p.models, s.name).toEqual({})
      expect(p.incumbent, s.name).toBeNull()
      expect(p.confirmation, s.name).toBeNull()
      expect(JSON.parse(git("show", `${s.commit}:data/meta/diagnostics.json`)).model, s.name).toBeUndefined()
    }
    expect(posteriorAt(git, r.stages[firstSearch]!.commit).models.gpu).toBeDefined()
    // Declared once, decided once (with the promotion), and genome.json changes in that one commit only.
    const declared = touches(r, "data/meta/confirmations/0.json")
    expect(declared).toHaveLength(2)
    expect(touches(r, "knowledge/genome.json")).toEqual([declared[1]])
    expect(touches(r, "data/meta/promotions.jsonl")).toEqual([declared[1]])
    const promotion = r.stages.find(s => s.name === declared[1])!
    const c0 = JSON.parse(git("show", `${promotion.commit}:data/meta/confirmations/0.json`)) as { candidate: { genomeSha: string }; result: { outcome: string } }
    expect(c0.result.outcome).toBe("pass")
    expect(git("log", "-1", "--format=%s", promotion.commit).trim()).toMatch(new RegExp(`; confirmation 0 pass; promote ${c0.candidate.genomeSha.slice(0, 8)}$`))
    expect(r.promoted).toBe(true)
    expect(r.knowledgeShaAfter).not.toBe(r.knowledgeShaBefore)
    // After the promotion: the posterior is built for the new centre.
    const after = posteriorAt(git, r.stages.at(-1)!.commit)
    expect(after.centre.genomeSha).toBe(c0.candidate.genomeSha)
    expect(after.incumbent).not.toBeNull()
    // main's history is linear: every commit's first parent is the one before (nothing was forced past).
    const log = git("rev-list", "--parents", "main").trim().split("\n").map(l => l.split(" "))
    for (let i = 0; i + 1 < log.length; i++) expect(log[i]![1]).toBe(log[i + 1]![0])
  }, 300_000)

  it("quick sim, calibration finds no signal: flat after 3 looks, and inert to the end (no fit, no model, no confirmation, genome.json unchanged)", async () => {
    const dir = fresh()
    const r = await runSim({ dir, quick: true, calibration: "noise", log: () => {} })
    const git = originOf(dir)
    expect(r.stages.map(s => s.mode)).toEqual(["calibrate", "calibrate", "calibrate", "flat", "flat"])
    expect(r.stages.at(-1)!.name).toBe("after flat")
    for (const s of r.stages) {
      for (const p of s.paths) expect(p, `${s.name}: ${p}`).toMatch(/^data\/meta\/(posterior\.json|calibration\.json|diagnostics\.json|verified\.json|posterior-index\.jsonl)$/)
      const p = posteriorAt(git, s.commit)
      expect(p, s.name).toMatchObject({ models: {}, incumbent: null, confirmation: null })
      expect(JSON.parse(git("show", `${s.commit}:data/meta/diagnostics.json`)).model, s.name).toBeUndefined()
    }
    expect(git("ls-tree", "-r", "--name-only", "main", "data/meta").trim().split("\n").filter(p => /confirmations|promotions/.test(p))).toEqual([])
    expect(r.promoted).toBe(false)
    expect(r.knowledgeShaAfter).toBe(r.knowledgeShaBefore)
  }, 300_000)
})
