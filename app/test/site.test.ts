import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import type { SessionRow } from "@blanktab/harness"
import { HISTORY_COLUMNS, buildLabel, fmt, fmtSettings, historyRows, siteLinks } from "../src/site"

const REPO = { owner: "markwhiting", name: "learner" } as const
const row = (id: string, seen: number, over: Partial<SessionRow> = {}): SessionRow => ({
  learnerId: id, settings: { moveProbability: 0.123456 }, hostSig: { cores: 8, deviceMemoryGb: null, webgpu: true, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: seen, ...over,
})

describe("fmt", () => {
  it("shows a small setting to 3 significant digits, not rounded to 3 decimals (lr 5e-4 is not 0.001)", () => {
    expect(fmtSettings({ lr: 5e-4, headLrScale: 0.1, bufferSize: 500 })).toBe("lr = 0.0005, headLrScale = 0.1, bufferSize = 500")
    expect([0.123456, 0.0012345, -0.00049, 0, 1024.4567, 1.04321, "x"].map(fmt)).toEqual(["0.123", "0.00123", "-0.00049", "0", "1024.457", "1.043", "x"])
  })
})

describe("historyRows", () => {
  it("is empty for no sessions", () => {
    expect(historyRows({ schemaVersion: 1, sessions: [] }, REPO)).toEqual([])
  })
  it("lists newest first by lastWrittenAt; rows without it go last (born after more sessions first, then later entry first)", () => {
    const rows = historyRows({ schemaVersion: 1, sessions: [
      row("lrn_1111111111111111", 5),
      row("lrn_2222222222222222", 0, { lastWrittenAt: "2026-10-19T12:00:00.000Z" }),
      row("lrn_3333333333333333", 1),
      row("lrn_4444444444444444", 0, { lastWrittenAt: "2026-10-20T08:00:00.000Z" }),
      row("lrn_5555555555555555", 1),
    ] }, REPO)
    expect(rows.map(r => r.learnerId)).toEqual(["lrn_4444444444444444", "lrn_2222222222222222", "lrn_1111111111111111", "lrn_5555555555555555", "lrn_3333333333333333"])
  })
  it("formats each column and links the learner to its data folder", () => {
    const [r] = historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, { final: false, stepsTotal: 1500, hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 2 } })] }, REPO)
    expect(r).toEqual({
      learnerId: "lrn_aaaaaaaaaaaaaaaa",
      href: "https://github.com/markwhiting/learner/tree/main/data/runs/lrn_aaaaaaaaaaaaaaaa",
      steps: "1500", final: "no", setting: "moveProbability = 0.123", host: "4 cores · no webgpu", cu: "—", lossRatio: "—", heldOutRatio: "—", milestones: "—",
      recipe: "—", score: "—",
    })
    expect(historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, { settings: {} })] }, REPO)[0]).toMatchObject({ final: "yes", setting: "none", host: "8 cores · webgpu" })
  })
  it("shows a learner's compute lived and its last loss ratio (v2 logs), a dash for v1 learners", () => {
    const [r] = historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, { cu: 1024.4567, lossRatio: 0.87654, heldOutRatio: 1.04321 })] }, REPO)
    expect(r).toMatchObject({ cu: "1024.457", lossRatio: "0.877", heldOutRatio: "1.043" })
    expect(historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, { cu: null, lossRatio: null, heldOutRatio: null })] }, REPO)[0]).toMatchObject({ cu: "—", lossRatio: "—", heldOutRatio: "—" })
  })
  it("historyRows: a milestones column, '2 of 6 at 512 CU', and '—' without checkupScore", () => {
    const at = (over: Partial<SessionRow>) => historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, over)] }, REPO)[0]!.milestones
    expect(at({ checkupScore: { level: 512, score: 2 } })).toBe("2 of 6 at 512 CU")
    expect(at({ checkupScore: { level: 2048, score: 0 } })).toBe("0 of 6 at 2,048 CU")
    expect(at({ checkupScore: null })).toBe("—")
    expect(at({})).toBe("—")
  })
})

describe("historyRows recipe and score (plan 05b, Task 11)", () => {
  it("a recipe column in words from recipe.method, and a score column from evals[evaluator]; '—' without", () => {
    const at = (over: Partial<SessionRow>, evaluator?: string) => {
      const [r] = historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, over)] }, REPO, evaluator)
      return [r!.recipe, r!.score]
    }
    const evals = { "search-composite@1": { eligible: true, score: 0.41234, C: 0.5, H: 0.3 }, "other@2": { eligible: false, score: null, C: null, H: null } }
    expect(at({ recipe: { role: "search", method: "ts" }, evals }, "search-composite@1")).toEqual(["meta-learner", "0.41"])
    expect(at({ recipe: { role: "search", method: "uniform" }, evals }, "other@2")).toEqual(["random", "not eligible"])
    expect(at({ recipe: { role: "search", method: "uniform" }, evals }, "missing@1")).toEqual(["random", "—"])
    expect(at({ recipe: { role: "search", method: "uniform" }, evals })).toEqual(["random", "—"])
    const words = (["centre", "calibrate", "uniform", "ts", "confirm", "newborn", "legacy"] as const).map(method => at({ recipe: { role: "search", method } })[0])
    expect(words).toEqual(["centre", "calibration", "random", "meta-learner", "confirmation", "newborn", "legacy"])
    expect(at({})).toEqual(["—", "—"])
  })
})

describe("siteLinks", () => {
  it("builds every GitHub link from the repo, plus life", () => {
    const links = siteLinks(REPO)
    expect(links.map(l => l.label)).toEqual([
      "Source", "Design", "Decisions", "Research notes", "How to check it works", "Session pull requests", "Actions", "Session data", "Related: life",
    ])
    const gh = "https://github.com/markwhiting/learner"
    expect(Object.fromEntries(links.map(l => [l.label, l.href]))).toEqual({
      "Source": gh,
      "Design": `${gh}/blob/main/docs/specs/2026-10-19-blanktab-v1-design.md`,
      "Decisions": `${gh}/tree/main/docs/decisions`,
      "Research notes": `${gh}/blob/main/docs/research/2026-10-16-research-synthesis-v0.1.md`,
      "How to check it works": `${gh}/blob/main/docs/validate-plan-01.md`,
      "Session pull requests": `${gh}/pulls`,
      "Actions": `${gh}/actions`,
      "Session data": `${gh}/tree/main/data/runs`,
      "Related: life": "https://markwhiting.github.io/life/",
    })
  })
})

describe("buildLabel", () => {
  const time = "2026-10-20T01:45:07.000Z"
  it("shows the short sha and links to its commit", () => {
    expect(buildLabel("0123456789abcdef0123456789abcdef01234567", time, REPO)).toEqual({
      text: "0123456 built 2026-10-20 01:45 UTC",
      href: "https://github.com/markwhiting/learner/commit/0123456789abcdef0123456789abcdef01234567",
    })
  })
  it("a dev build links to the repo", () => {
    expect(buildLabel("dev", time, REPO)).toEqual({ text: "dev built 2026-10-20 01:45 UTC", href: "https://github.com/markwhiting/learner" })
  })
})

describe("HISTORY_COLUMNS", () => {
  it("the labels equal app/index.html's HISTORY <th> texts, in order; numeric exactly Steps, CU, Held-out ratio, Own-stream ratio, Score", () => {
    const html = readFileSync(join(__dirname, "../index.html"), "utf8")
    const head = html.match(/<thead><tr>(<th[^>]*>[^<]*<\/th>)+<\/tr><\/thead>/)![0]
    const ths = [...head.matchAll(/<th[^>]*>([^<]*)<\/th>/g)].map(m => m[1])
    expect(HISTORY_COLUMNS.map(c => c.label)).toEqual(ths)
    expect(HISTORY_COLUMNS.filter(c => c.numeric).map(c => c.label)).toEqual(["Steps", "CU", "Held-out ratio", "Own-stream ratio", "Score"])
  })
})
