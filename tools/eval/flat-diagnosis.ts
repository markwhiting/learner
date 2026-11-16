// Usage: npx tsx tools/eval/flat-diagnosis.ts   (run from the repo root; reads data/, config/, knowledge/; writes .local/)
// Why did calibration end "flat" (2026-11-08)? Decomposes search-composite@1 into its components on exactly the 60
// lives of signal-check look 3, estimates repeat noise from every centre-genome life, tests genomes and genes, checks
// data quality and simulates how many repeats signal-check@1 would need. Scores come only from the registered
// evaluators (search-composite@1, signal-check@1); nothing here is pinned, gated or written outside .local/.
// Deterministic: every simulation is seeded (SEED below). Output: markdown on stdout, the numbers in
// .local/flat-diagnosis.json. Research note: docs/research/2026-11-16-calibration-flat.md.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { hashSeed } from "@blanktab/detmath"
import { cohortContext, getEvaluator, ALPHA_LOOK, ICC_MIN, type EvalOutput, type LifeInput, type References } from "@blanktab/evaluators"
import type { LogFileV2 } from "@blanktab/harness/log"
import { parseGenomeSpace, searched, toUnit, type Genome } from "@blanktab/meta"
import { budgetWindow, isV2 } from "../../packages/evaluators/src/life.ts"
import { fisherOneSided, icc1, kruskalWallis } from "../../packages/evaluators/src/stats.ts"
import { loadEvalConfig, loadReferences } from "./config.ts"
import { collectLives } from "./lives.ts"
import {
  anova1, covComponents, chiSquareQuantile, fCdf, holmAdjust, iccCi, powerAnovaRandom, psd2, repeatsFor, sampler, spearman, tCrit,
} from "./flat-stats.ts"

const SEED = hashSeed(0, "flat-diagnosis/2026-11-16")
const LEVELS = [512, 1024, 2048] as const
const CM = ["M2", "M3", "M4", "M5", "M6"] as const
const ALL_M = ["M1", ...CM] as const
const K = 10
const REPEAT_GRID = [3, 6, 10, 15, 20, 30, 50, 80, 120]
const SIMS_MAIN = 2000
const SIMS_COMPONENT = 1000
const PERMS = 20000

type Row = {
  learnerId: string; genome: string; panel: number | null; method: string; out: EvalOutput
  comps: Record<string, number | null>
  files: LogFileV2[]; window: LogFileV2[]
}

const f3 = (v: number | null | undefined, d = 3): string => (v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d))
const pf = (p: number): string => (p < 0.001 ? p.toExponential(1) : p.toFixed(3))
const mean = (xs: readonly number[]): number => xs.reduce((s, x) => s + x, 0) / xs.length
const variance = (xs: readonly number[]): number => { const m = mean(xs); return xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1) }
const median = (xs: readonly number[]): number => { const s = [...xs].sort((a, b) => a - b); const h = s.length >> 1; return s.length % 2 ? s[h]! : (s[h - 1]! + s[h]!) / 2 }
const table = (head: string[], rows: (string | number)[][]): string =>
  [`| ${head.join(" | ")} |`, `|${head.map(() => "---").join("|")}|`, ...rows.map(r => `| ${r.join(" | ")} |`)].join("\n")

/** The first checkup at each budget level in the life's budget window, as search-composite@1 reads them. */
function checkupsAt(window: LogFileV2[]): Map<number, NonNullable<LogFileV2["checkups"]>[number]> {
  const m = new Map<number, NonNullable<LogFileV2["checkups"]>[number]>()
  for (const f of window) for (const c of f.checkups ?? []) if ((LEVELS as readonly number[]).includes(c.level) && !m.has(c.level)) m.set(c.level, c)
  return m
}

/**
 * Every component of search-composite@1 for one scored life, read from the evaluator's own output (score, C, H,
 * perLevel, perMilestone, q, qM1) plus the checkups' pass lists (the confirmation test's pass indicators).
 */
function componentsOf(out: EvalOutput, window: LogFileV2[]): Record<string, number | null> {
  const d = out.diagnostics as {
    perLevel?: Record<string, number>; perMilestone?: Record<string, number>; q?: Record<string, Record<string, number>>
    qM1?: Record<string, number | null>; heldOutRatio?: number
  }
  const c: Record<string, number | null> = { score: out.score, C: out.components.C ?? null, H: out.components.H ?? null }
  for (const L of LEVELS) c[`C@${L}`] = d.perLevel?.[String(L)] ?? null
  for (const m of CM) c[`q.${m}`] = d.perMilestone?.[m] ?? null
  for (const m of CM) for (const L of LEVELS) c[`q.${m}@${L}`] = d.q?.[String(L)]?.[m] ?? null
  const m1 = LEVELS.map(L => d.qM1?.[String(L)]).filter((v): v is number => typeof v === "number")
  c["q.M1 (diag)"] = m1.length ? mean(m1) : null
  const cks = checkupsAt(window)
  let sustainedCount = 0
  for (const m of ALL_M) {
    let n = 0
    for (const L of LEVELS) {
      const ck = cks.get(L)
      const v = ck ? (ck.passed.includes(m) ? 1 : 0) : null
      if (m !== "M1") c[`pass.${m}@${L}`] = v
      n += v ?? 0
    }
    c[`sustained.${m}`] = n >= 2 ? 1 : 0
    if (m !== "M1") sustainedCount += n >= 2 ? 1 : 0
  }
  for (const L of LEVELS) { const ck = cks.get(L); c[`passes@${L} (M2–M6)`] = ck ? ck.passed.filter(p => p !== "M1").length : null }
  c["sustained count (M2–M6)"] = sustainedCount
  return c
}

/** Component names in report order (the binary ones flagged). */
function componentNames(): { name: string; binary: boolean; group: string }[] {
  const out: { name: string; binary: boolean; group: string }[] = [
    { name: "score", binary: false, group: "headline" }, { name: "C", binary: false, group: "headline" }, { name: "H", binary: false, group: "headline" },
  ]
  for (const L of LEVELS) out.push({ name: `C@${L}`, binary: false, group: "C by level" })
  for (const m of CM) out.push({ name: `q.${m}`, binary: false, group: "C by milestone" })
  out.push({ name: "q.M1 (diag)", binary: false, group: "C by milestone" })
  for (const m of CM) for (const L of LEVELS) out.push({ name: `q.${m}@${L}`, binary: false, group: "q per milestone × level" })
  for (const L of LEVELS) out.push({ name: `passes@${L} (M2–M6)`, binary: false, group: "passes" })
  out.push({ name: "sustained count (M2–M6)", binary: false, group: "passes" })
  for (const m of ALL_M) out.push({ name: `sustained.${m}`, binary: true, group: "passes" })
  for (const m of CM) for (const L of LEVELS) out.push({ name: `pass.${m}@${L}`, binary: true, group: "pass per milestone × level" })
  return out
}

/** Groups of a component's values by genome (genomes in `order`), dropping nulls. */
const groupsOf = (rows: Row[], order: string[], name: string): number[][] =>
  order.map(g => rows.filter(r => r.genome === g).map(r => r.comps[name]).filter((v): v is number => typeof v === "number"))

// ---- simulations -------------------------------------------------------------------------------------------------

/** Pooled within-genome residuals of `names` (rows of vectors), each group's deviations inflated by √(n/(n−1)). */
function residualPool(groups: Row[][], names: string[]): number[][] {
  const out: number[][] = []
  for (const g of groups) {
    const vecs = g.map(r => names.map(n => r.comps[n])).filter((v): v is number[] => v.every(x => typeof x === "number"))
    if (vecs.length < 2) continue
    const m = names.map((_, j) => mean(vecs.map(v => v[j]!)))
    const s = Math.sqrt(vecs.length / (vecs.length - 1))
    for (const v of vecs) out.push(v.map((x, j) => (x - m[j]!) * s))
  }
  return out
}

/** A minimal search-composite@1 output, so the evaluator's own rankAverage ranks simulated (C, H). */
const fakeOut = (C: number, H: number): EvalOutput =>
  ({ evaluator: "search-composite", version: 1, eligible: true, score: C + H, components: { C, H }, diagnostics: {} })

/**
 * Power of signal-check@1's rule itself (KW on the rank-average a_i at ALPHA_LOOK and ICC(score) ≥ ICC_MIN) for K
 * genomes × n repeats: genome effects (C, H) ~ N(0, Σ_b), residual (C, H) vectors resampled from `pool`.
 */
function simulateSignalCheck(n: number, sigmaB: number[][], pool: number[][], sims: number, seed: number): { power: number; kwOnly: number } {
  const s = sampler(seed)
  const search = getEvaluator("search-composite@1")
  const l11 = Math.sqrt(Math.max(0, sigmaB[0]![0]!))
  const l21 = l11 > 0 ? sigmaB[1]![0]! / l11 : 0
  const l22 = Math.sqrt(Math.max(0, sigmaB[1]![1]! - l21 * l21))
  let hits = 0, kwHits = 0
  for (let it = 0; it < sims; it++) {
    const outs: EvalOutput[] = [], gid: number[] = []
    for (let g = 0; g < K; g++) {
      const z1 = s.normal(), z2 = s.normal()
      const bC = l11 * z1, bH = l21 * z1 + l22 * z2
      for (let i = 0; i < n; i++) { const e = pool[s.index(pool.length)]!; outs.push(fakeOut(bC + e[0]!, bH + e[1]!)); gid.push(g) }
    }
    const a = search.rankAverage!(outs)
    const ga = Array.from({ length: K }, () => [] as number[]), gs = Array.from({ length: K }, () => [] as number[])
    a.forEach((v, i) => { ga[gid[i]!]!.push(v); gs[gid[i]!]!.push(outs[i]!.score!) })
    const p = kruskalWallis(ga).p
    if (p < ALPHA_LOOK) { kwHits++; if (icc1(gs) >= ICC_MIN) hits++ }
  }
  return { power: hits / sims, kwOnly: kwHits / sims }
}

/**
 * Power of the same rule on one component used alone as the score (KW on it at ALPHA_LOOK and ICC ≥ ICC_MIN).
 * Continuous: genome effects N(0, σ_b²) plus resampled residuals. Binary: per-genome rate ~ Beta with the pooled rate
 * and intraclass correlation ρ (beta-binomial), lives Bernoulli.
 */
function simulateComponent(n: number, kind: { binary: true; rate: number; rho: number } | { binary: false; sb: number; pool: number[] }, sims: number, seed: number): number {
  const s = sampler(seed)
  let hits = 0
  for (let it = 0; it < sims; it++) {
    const groups: number[][] = []
    for (let g = 0; g < K; g++) {
      const xs: number[] = []
      if (kind.binary) {
        const ab = 1 / Math.max(kind.rho, 1e-9) - 1
        const pg = kind.rho > 0 ? s.beta(Math.max(kind.rate * ab, 1e-6), Math.max((1 - kind.rate) * ab, 1e-6)) : kind.rate
        for (let i = 0; i < n; i++) xs.push(s.u() < pg ? 1 : 0)
      } else {
        const b = kind.sb * s.normal()
        for (let i = 0; i < n; i++) xs.push(b + kind.pool[s.index(kind.pool.length)]!)
      }
      groups.push(xs)
    }
    const flat = groups.flat()
    if (flat.every(v => v === flat[0])) continue
    if (kruskalWallis(groups).p < ALPHA_LOOK && icc1(groups) >= ICC_MIN) hits++
  }
  return hits / sims
}

// ---- main --------------------------------------------------------------------------------------------------------

function main(): void {
  const root = process.cwd()
  const config = loadEvalConfig(root)
  const refs: References = loadReferences(root, config)
  const search = getEvaluator(config.search)
  const calib = JSON.parse(readFileSync(join(root, "data/meta/calibration.json"), "utf8")) as {
    outcome: string; looks: { look: number; lives: number; output: EvalOutput; learnerIds: string[] }[]
    genomes: { genomeSha: string; values: Genome }[]
  }
  const space = parseGenomeSpace(JSON.parse(readFileSync(join(root, "knowledge/genome-space.json"), "utf8")))
  const genes = searched(space).map(g => g.name)
  const centreSha = calib.genomes[0]!.genomeSha
  const look3 = calib.looks.find(l => l.look === 3)!
  const groupsById = look3.output.diagnostics.learnerIds as Record<string, string[]>
  const genomeOf = new Map<string, string>()
  for (const [g, ids] of Object.entries(groupsById)) for (const id of ids) genomeOf.set(id, g)
  const order = calib.genomes.map(g => g.genomeSha) // centre first, then the Sobol points in calibration order
  const short = (g: string) => (g === centreSha ? `centre ${g.slice(0, 6)}` : `S${order.indexOf(g)} ${g.slice(0, 6)}`)

  const lives = collectLives(join(root, "data/runs"))
  const byId = new Map(lives.map(l => [l.learnerId, l]))
  const out: string[] = []
  const json: Record<string, unknown> = { seed: SEED }
  const say = (s = "") => out.push(s)

  // 0. Reproduce look 3 with the registered signal-check@1 on the same groups.
  const groups: Record<string, LifeInput[]> = {}
  for (const [g, ids] of Object.entries(groupsById)) groups[g] = ids.map(id => byId.get(id)!).filter(Boolean)
  const rerun = getEvaluator(config.signal).evaluateCohort!(groups, refs, cohortContext(config))
  say("## 0. Reproduction of look 3")
  say(`recorded: p=${look3.output.components.p} icc=${look3.output.components.icc} n=${look3.lives}`)
  say(`rerun:    p=${rerun.components.p} icc=${rerun.components.icc} n=${(rerun.diagnostics as { n: number }).n} (signal-check@1, ${config.search}, pinned reference)`)
  json.reproduction = { recorded: look3.output.components, rerun: rerun.components }

  // Score every search life.
  const rowOf = (l: LifeInput): Row | null => {
    const files = l.files.filter(isV2)
    const meta = files.find(f => f.meta)?.meta
    if (!meta || meta.role !== "search") return null
    const o = search.evaluateLife!(l, refs)
    const window = budgetWindow(l.files, 2048).filter(isV2)
    return { learnerId: l.learnerId, genome: meta.genomeSha, panel: meta.evalPanel, method: meta.method, out: o, comps: o.eligible ? componentsOf(o, window) : {}, files, window }
  }
  const allRows = lives.map(rowOf).filter((r): r is Row => r !== null)
  const calRows = allRows.filter(r => genomeOf.has(r.learnerId))
  if (calRows.length !== 60 || calRows.some(r => !r.out.eligible || r.genome !== genomeOf.get(r.learnerId))) throw new Error("the 60 look-3 lives did not all load and score")
  const centrePool = allRows.filter(r => r.genome === centreSha && r.out.eligible)
  const postFlatCentre = centrePool.filter(r => !genomeOf.has(r.learnerId))
  const uniform = allRows.filter(r => r.method === "uniform")
  say()
  say(`search lives on disk: ${allRows.length}; look-3 calibration lives: ${calRows.length}; centre-genome eligible lives: ${centrePool.length} (${centrePool.length - postFlatCentre.length} calibration + ${postFlatCentre.length} since flat); uniform: ${uniform.length}`)

  // 1. Components: between vs within, ICC with CI, KW p, ANOVA p; centre-pool repeat noise.
  say()
  say("## 1. Components of search-composite@1 across the 10 calibration genomes (60 lives)")
  say("σ²_b, σ²_w: method-of-moments variance components (σ²_b may be negative); ICC(1) untruncated with its F-based 95% CI;")
  say("KW p: Kruskal–Wallis across genomes (signal-check@1's test, here on the component itself); F p: one-way ANOVA.")
  say("σ_w centre: SD over all centre-genome lives (pure repeat noise) with 95% χ² CI; ICC*: ICC recomputed with σ²_w from the centre pool.")
  const comps = componentNames()
  const compStats: Record<string, unknown>[] = []
  const rowsTable: (string | number)[][] = []
  for (const c of comps) {
    const gs = groupsOf(calRows, order, c.name)
    if (gs.some(g => g.length === 0)) continue
    const flat = gs.flat()
    if (flat.every(v => v === flat[0])) { rowsTable.push([c.name, f3(mean(flat)), "constant", "", "", "", "", "", "", "", ""]); continue }
    const a = anova1(gs)
    const [lo, hi] = iccCi(a)
    const kw = kruskalWallis(gs)
    const cp = centrePool.map(r => r.comps[c.name]).filter((v): v is number => typeof v === "number")
    const sw2c = variance(cp), dfc = cp.length - 1
    const swLo = Math.sqrt((dfc * sw2c) / chiSquareQuantile(0.975, dfc)), swHi = Math.sqrt((dfc * sw2c) / chiSquareQuantile(0.025, dfc))
    const sb2c = (a.msb - sw2c) / a.n0
    // No repeat noise in the centre pool (e.g. a pass no centre life ever made) leaves ICC* undefined.
    const iccStar = sw2c > 0 ? (sb2c > 0 ? sb2c / (sb2c + sw2c) : 0) : null
    compStats.push({ name: c.name, binary: c.binary, group: c.group, mean: a.grand, sb2: a.sb2, sw2: a.sw2, icc: a.icc, iccLo: lo, iccHi: hi, kwP: kw.p, fP: a.p, n0: a.n0, centreN: cp.length, centreSd: Math.sqrt(sw2c), centreSdLo: swLo, centreSdHi: swHi, iccStar, centreMean: mean(cp) })
    rowsTable.push([c.name, f3(a.grand), f3(a.sb2, 4), f3(a.sw2, 4), f3(a.icc), `[${f3(lo, 2)}, ${f3(hi, 2)}]`, pf(kw.p), pf(a.p), `${f3(Math.sqrt(sw2c))} [${f3(swLo, 2)}, ${f3(swHi, 2)}] (n=${cp.length})`, f3(Math.sqrt(Math.max(0, a.sw2))), iccStar === null ? "–" : f3(iccStar)])
  }
  say(table(["component", "mean", "σ²_b", "σ²_w", "ICC", "95% CI", "KW p", "F p", "σ_w centre [CI] (n)", "σ_w calib", "ICC*"], rowsTable))
  json.components = compStats

  // The a_i KW (what signal-check@1 tests) and how C and H relate.
  const aAll = search.rankAverage!(calRows.map(r => r.out))
  const aGroups = order.map(g => calRows.map((r, i) => (r.genome === g ? aAll[i]! : NaN)).filter(Number.isFinite))
  const aKw = kruskalWallis(aGroups)
  const Cs = calRows.map(r => r.comps.C!), Hs = calRows.map(r => r.comps.H!)
  say()
  say(`a_i (rank average of C and H): KW p = ${pf(aKw.p)} (H=${f3(aKw.H, 2)}), ICC(a) = ${f3(anova1(aGroups).icc)}; Spearman(C, H) over the 60 = ${f3(spearman(Cs, Hs))}`)
  say(`variance share of score: var(C) = ${f3(variance(Cs), 4)}, var(H) = ${f3(variance(Hs), 4)}, var(score) = ${f3(variance(calRows.map(r => r.out.score!)), 4)}`)
  json.aKw = aKw

  // Trained vs untrained: are the checkups above newborn chance at all?
  say()
  say("### Checkups against the pinned newborn reference")
  const ref = refs.newborn!
  const passRows: (string | number)[][] = []
  const pool = [...calRows, ...postFlatCentre]
  const nbAll = collectLives(join(root, "data/newborns")).map(l => l.files.find(isV2)!).filter(f => f.meta?.role === "newborn")
  for (const m of ALL_M) {
    // Per checkup (each life's three budget checkups): passes / checkups, newborns from their own files.
    const sCk = pool.flatMap(r => [...checkupsAt(r.window).values()].map(c => (c.passed.includes(m) ? 1 : 0)))
    const nCk = nbAll.flatMap(f => (f.checkups ?? []).filter(c => (LEVELS as readonly number[]).includes(c.level)).map(c => (c.passed.includes(m) ? 1 : 0)))
    const a = sCk.reduce<number>((x, y) => x + y, 0), c = nCk.reduce<number>((x, y) => x + y, 0)
    const pHi = fisherOneSided(a, sCk.length - a, c, nCk.length - c), pLo = fisherOneSided(c, nCk.length - c, a, sCk.length - a)
    const sus = mean(pool.map(r => r.comps[`sustained.${m}`]!))
    const q = m === "M1" ? pool.map(r => r.comps["q.M1 (diag)"]).filter((v): v is number => typeof v === "number") : pool.map(r => r.comps[`q.${m}`]!)
    const ms = ref.milestones[m as keyof typeof ref.milestones]
    passRows.push([m, `${c}/${nCk.length} = ${f3(c / nCk.length)}`, `${a}/${sCk.length} = ${f3(a / sCk.length)}`, `${pf(Math.min(1, 2 * Math.min(pHi, pLo)))} (${pHi < pLo ? "search higher" : "search lower"})`, f3(ms.sustainedPassRate), f3(sus), f3(median(q)), f3(mean(q))])
  }
  say(`search lives: the 60 calibration lives + ${postFlatCentre.length} centre lives since flat (n=${pool.length}); newborns: ${nbAll.length} (the pinned reference's).`)
  say("p: two-sided Fisher exact (2 × the smaller one-sided p) on per-checkup passes; checkups of one life are not independent, so this p is optimistic.")
  say(table(["milestone", "newborn passes / checkups", "search passes / checkups", "p", "newborn sustained", "search sustained", "median q (0 = newborn median)", "mean q"], passRows))
  const ho = pool.map(r => (r.out.diagnostics as { heldOutRatio: number }).heldOutRatio)
  say(`held-out ratio ρ̃ (search lives): median ${f3(median(ho), 4)}, range [${f3(Math.min(...ho), 4)}, ${f3(Math.max(...ho), 4)}]; newborn reference median ${f3(ref.heldOutMedian, 4)}`)
  say(`  share with ρ̃ < 1 (better than copy-last, i.e. H > 0): ${f3(ho.filter(v => v < 1).length / ho.length)}; ρ̃ > 2: ${ho.filter(v => v > 2).length}; H clipped at −2 (ρ̃ > e²): ${ho.filter(v => v > Math.exp(2)).length}`)
  const tail = calRows.filter(r => (r.out.diagnostics as { heldOutRatio: number }).heldOutRatio > 2)
  say(`  calibration lives with ρ̃ > 2, by genome: ${JSON.stringify(Object.fromEntries(order.map(g => [short(g), tail.filter(r => r.genome === g).length]).filter(([, n]) => n)))}`)
  say(`C over search lives: mean ${f3(mean(pool.map(r => r.comps.C!)))} (SD ${f3(Math.sqrt(variance(pool.map(r => r.comps.C!))))}); H: mean ${f3(mean(pool.map(r => r.comps.H!)))} (SD ${f3(Math.sqrt(variance(pool.map(r => r.comps.H!))))})`)

  // Checkup reliability: newborns (same frozen weights at 3 levels) and search lives (3 levels of one life).
  say()
  say("### Checkup test–retest (raw measures)")
  say("newborn: ICC across a newborn's 3 frozen checkups (same weights; different worlds and policy seeds) = how much the weights,")
  say("not the world draw, fix a measure. world: ICC across newborns sharing a panel × level cell (same world and policy seed,")
  say("different weights). life: ICC across a search life's 3 checkups (weights trained between them).")
  const newborns = collectLives(join(root, "data/newborns"))
  const nbFiles = newborns.map(l => l.files.find(isV2)!).filter(f => f.meta?.role === "newborn")
  const relRows: (string | number)[][] = []
  for (const m of ALL_M) {
    const byNb = nbFiles.map(f => LEVELS.map(L => f.checkups?.find(c => c.level === L)?.measures[m] ?? 0))
    const cells = new Map<string, number[]>()
    for (const f of nbFiles) for (const L of LEVELS) {
      const k = `${f.meta!.evalPanel}/${L}`
      const v = f.checkups?.find(c => c.level === L)?.measures[m] ?? 0
      cells.set(k, [...(cells.get(k) ?? []), v])
    }
    const byLife = pool.map(r => { const cks = checkupsAt(r.window); return LEVELS.map(L => cks.get(L)?.measures[m] ?? 0) })
    const safe = (gs: number[][]) => { const fl = gs.flat(); return fl.every(v => v === fl[0]) ? null : anova1(gs).icc }
    relRows.push([m, f3(safe(byNb)), f3(safe([...cells.values()])), f3(safe(byLife))])
  }
  say(table(["measure", "ICC newborn (same weights)", "ICC world (same panel×level)", "ICC life (3 levels)"], relRows))
  say(`(${nbFiles.length} newborns; null read as 0, as C does)`)

  // 2. Genomes: effect sizes; genes: rank correlations.
  say()
  say("## 2. Genome effects and gene correlations")
  const gRows: (string | number)[][] = []
  const centreScores = centrePool.map(r => r.out.score!)
  const effects: Record<string, unknown>[] = []
  const pRaw: number[] = []
  for (const g of order) {
    const xs = calRows.filter(r => r.genome === g)
    const sc = xs.map(r => r.out.score!), C = xs.map(r => r.comps.C!), H = xs.map(r => r.comps.H!)
    const gv = calib.genomes.find(x => x.genomeSha === g)!.values
    const genesCell = `${(gv.lr as number).toExponential(1)} / ${(gv.exploreEpsilon as number).toFixed(3)}`
    if (g === centreSha) {
      gRows.push([short(g), genesCell, xs.length, f3(mean(sc)), f3(mean(C)), f3(mean(H)), `(pool n=${centreScores.length}, mean ${f3(mean(centreScores))})`, "", "", ""])
      continue
    }
    const v1 = variance(sc) / sc.length, v2 = variance(centreScores) / centreScores.length
    const d = mean(sc) - mean(centreScores)
    const df = (v1 + v2) * (v1 + v2) / ((v1 * v1) / (sc.length - 1) + (v2 * v2) / (centreScores.length - 1))
    const se = Math.sqrt(v1 + v2), t = d / se
    const p = 1 - fCdf(t * t, 1, df)
    const sp = Math.sqrt(((sc.length - 1) * variance(sc) + (centreScores.length - 1) * variance(centreScores)) / (sc.length + centreScores.length - 2))
    const J = 1 - 3 / (4 * (sc.length + centreScores.length) - 9)
    pRaw.push(p)
    effects.push({ genome: g, n: sc.length, mean: mean(sc), diff: d, lo: d - tCrit(0.05, df) * se, hi: d + tCrit(0.05, df) * se, g: (J * d) / sp, p })
    gRows.push([short(g), genesCell, xs.length, f3(mean(sc)), f3(mean(C)), f3(mean(H)), f3(d), `[${f3(d - tCrit(0.05, df) * se, 2)}, ${f3(d + tCrit(0.05, df) * se, 2)}]`, f3((J * d) / sp, 2), pf(p)])
  }
  const holm = holmAdjust(pRaw)
  effects.forEach((e, i) => { e.pHolm = holm[i] })
  gRows.forEach((r, i) => { if (i > 0) r.push(pf(holm[i - 1]!)); else r.push("") })
  say("Each Sobol genome against the whole centre-genome pool (calibration + since flat), on score. Welch 95% CI; Hedges g on the pooled SD; Holm over 9.")
  say(table(["genome", "lr / exploreEpsilon", "n", "mean score", "mean C", "mean H", "Δ vs centre", "95% CI", "g", "p", "p Holm"], gRows))
  json.effects = effects

  // Gene rank correlations, with a cluster permutation (genome labels permuted, lives kept with their genome).
  const units = new Map(calib.genomes.map(g => [g.genomeSha, toUnit(space, g.values)]))
  const s = sampler(hashSeed(SEED, "perm"))
  const geneRows: (string | number)[][] = []
  const geneP: number[] = [], geneRes: Record<string, unknown>[] = []
  for (const outcome of ["score", "C", "H"] as const) {
    const y = calRows.map(r => r.comps[outcome]!)
    const gi = calRows.map(r => order.indexOf(r.genome))
    const perms = Array.from({ length: PERMS }, () => s.shuffle(order.map((_, i) => i)))
    genes.forEach((gene, j) => {
      const gv = order.map(g => units.get(g)![j]!)
      const rho = spearman(gi.map(i => gv[i]!), y)
      let ge = 0
      for (const p of perms) if (Math.abs(spearman(gi.map(i => gv[p[i]!]!), y)) >= Math.abs(rho) - 1e-12) ge++
      const p = (ge + 1) / (PERMS + 1)
      geneP.push(p)
      geneRes.push({ outcome, gene, rho, p })
    })
  }
  const geneHolm = holmAdjust(geneP)
  geneRes.forEach((r, i) => { r.pHolm = geneHolm[i] })
  for (const gene of genes) {
    const cell = (o: string) => { const r = geneRes.find(x => x.gene === gene && x.outcome === o)!; return `${f3(r.rho as number, 2)} (p ${pf(r.p as number)}, Holm ${pf(r.pHolm as number)})` }
    geneRows.push([gene, cell("score"), cell("C"), cell("H")])
  }
  say()
  say(`Spearman ρ of each searched gene (unit-cube value) with score, C and H over the 60 lives; p by ${PERMS} seeded permutations of genome labels`)
  say("(lives stay with their genome, so the test has 10 effective points); Holm over all 36 tests.")
  say(table(["gene", "score", "C", "H"], geneRows))
  json.genes = geneRes
  // How confounded the genes are across only 10 design points.
  const conf: { a: string; b: string; rho: number }[] = []
  genes.forEach((a, i) => genes.forEach((b, j) => {
    if (j <= i) return
    conf.push({ a, b, rho: spearman(order.map(g => units.get(g)![i]!), order.map(g => units.get(g)![j]!)) })
  }))
  conf.sort((x, y) => Math.abs(y.rho) - Math.abs(x.rho) || (x.a + x.b < y.a + y.b ? -1 : 1))
  say(`gene–gene Spearman ρ across the 10 calibration genomes, largest |ρ|: ${conf.slice(0, 6).map(c => `${c.a}~${c.b} ${f3(c.rho, 2)}`).join("; ")}`)
  json.geneConfounding = conf.slice(0, 10)

  // 3. Power.
  say()
  say("## 3. Power: repeats per genome (10 genomes) for 80% power at α = 1/60")
  say("Analytic: one-way random-effects F test, normal effects and errors, at the component's point ICC (and its CI ends).")
  const powRows: (string | number)[][] = []
  const powJson: Record<string, unknown>[] = []
  for (const cs of compStats as { name: string; icc: number; iccLo: number; iccHi: number; iccStar: number | null; binary: boolean }[]) {
    const n = repeatsFor(cs.icc, K, ALPHA_LOOK), nHi = repeatsFor(cs.iccHi, K, ALPHA_LOOK), nLo = repeatsFor(cs.iccLo, K, ALPHA_LOOK)
    const nStar = cs.iccStar === null ? null : repeatsFor(cs.iccStar, K, ALPHA_LOOK)
    const p6 = powerAnovaRandom(cs.icc, K, 6, ALPHA_LOOK)
    powJson.push({ name: cs.name, icc: cs.icc, nFor80: n, nFor80AtIccHi: nHi, nFor80AtIccLo: nLo, nFor80IccStar: nStar, powerAt6: p6 })
    powRows.push([cs.name, f3(cs.icc), f3(p6, 2), n ?? "never (ICC ≤ 0)", nHi ?? "–", nLo ?? "never", cs.iccStar === null ? "–" : nStar ?? "never"])
  }
  say(table(["component", "ICC", "power at 6 repeats", "n for 80%", "n at ICC CI high", "n at ICC CI low", "n at ICC* (centre σ_w)"], powRows))
  json.powerAnalytic = powJson

  // Simulated signal-check@1 rule.
  say()
  say(`Simulated signal-check@1 rule (KW on a_i at α=1/60 AND ICC(score) ≥ ${ICC_MIN}), ${SIMS_MAIN} seeded runs per n:`)
  say("genome effects (C, H) ~ N(0, Σ_b) with Σ_b the MANOVA moment estimate on the 60 (clipped to PSD); residual (C, H) vectors")
  say("resampled from the within-genome residuals of the 9 Sobol genomes plus the whole centre pool (non-normal shape kept).")
  const resGroups = [...order.filter(g => g !== centreSha).map(g => calRows.filter(r => r.genome === g)), centrePool]
  const poolCH = residualPool(resGroups, ["C", "H"])
  const cov = covComponents(order.map(g => calRows.filter(r => r.genome === g).map(r => [r.comps.C!, r.comps.H!])))
  const sigmaB = psd2(cov.between)
  say(`Σ_b (raw) = [[${cov.between.flat().map(v => f3(v, 5)).join(", ")}]]; PSD = [[${sigmaB.flat().map(v => f3(v, 5)).join(", ")}]]; Σ_w = [[${cov.within.flat().map(v => f3(v, 5)).join(", ")}]]; residual pool n=${poolCH.length}`)
  const simRows: (string | number)[][] = []
  const simJson: Record<string, unknown>[] = []
  for (const n of REPEAT_GRID) {
    const r = simulateSignalCheck(n, sigmaB, poolCH, SIMS_MAIN, hashSeed(SEED, `sc/${n}`))
    simJson.push({ n, ...r })
    simRows.push([n, K * n, f3(r.kwOnly, 3), f3(r.power, 3)])
  }
  say(table(["repeats n", "lives", "P(KW p < 1/60)", "P(signal: KW and ICC ≥ 0.1)"], simRows))
  say(`(Monte Carlo SE ≤ ${f3(Math.sqrt(0.25 / SIMS_MAIN), 3)})`)
  json.powerSignalCheck = simJson

  // Simulated single-component rules for the headline parts and the best components.
  const ranked = [...(compStats as { name: string; icc: number; binary: boolean; mean: number; sb2: number }[])].sort((x, y) => y.icc - x.icc)
  const picks = ["score", "C", "H", ...ranked.filter(c => !["score", "C", "H"].includes(c.name)).slice(0, 4).map(c => c.name)]
  say()
  say(`Simulated rule with one component as the whole score (KW at 1/60 and ICC ≥ ${ICC_MIN} on it), ${SIMS_COMPONENT} runs per n. The best`)
  say("components are picked post hoc from ~50 by their observed ICC, so their ICCs are biased upward (winner's curse).")
  const compSimRows: (string | number)[][] = []
  const compSimJson: Record<string, unknown>[] = []
  for (const name of picks) {
    const cs = ranked.find(c => c.name === name)!
    const kind = cs.binary
      ? { binary: true as const, rate: cs.mean, rho: Math.max(0, cs.icc) }
      : { binary: false as const, sb: Math.sqrt(Math.max(0, cs.sb2)), pool: residualPool(resGroups, [name]).map(v => v[0]!) }
    const ps = REPEAT_GRID.map(n => simulateComponent(n, kind, SIMS_COMPONENT, hashSeed(SEED, `comp/${name}/${n}`)))
    compSimJson.push({ name, icc: cs.icc, power: Object.fromEntries(REPEAT_GRID.map((n, i) => [n, ps[i]])) })
    compSimRows.push([name, f3(cs.icc), ...ps.map(p => f3(p, 2))])
  }
  say(table(["component", "ICC", ...REPEAT_GRID.map(n => `n=${n}`)], compSimRows))
  json.powerComponents = compSimJson

  // 4. Data quality.
  say()
  say("## 4. Data quality")
  dataQuality(root, allRows, calRows, centrePool, order, short, say, json)

  say()
  say("## 5. Cost")
  cost(allRows, calRows, postFlatCentre, say, json)

  mkdirSync(join(root, ".local"), { recursive: true })
  writeFileSync(join(root, ".local/flat-diagnosis.json"), JSON.stringify(json, null, 1) + "\n")
  writeFileSync(join(root, ".local/flat-diagnosis.md"), out.join("\n") + "\n")
  console.log(out.join("\n"))
  console.log("\nwrote .local/flat-diagnosis.md and .local/flat-diagnosis.json")
}

function dataQuality(root: string, all: Row[], cal: Row[], centre: Row[], order: string[], short: (g: string) => string, say: (s?: string) => void, json: Record<string, unknown>): void {
  const scope = [...new Map([...cal, ...centre].map(r => [r.learnerId, r])).values()]
  // Files that did not parse, and seq gaps.
  const issues: (string | number)[][] = []
  let unparsed = 0, gaps = 0, notZero = 0
  for (const r of scope) {
    const onDisk = readdirSync(join(root, "data/runs", r.learnerId)).filter(n => /^\d+\.json$/.test(n)).length
    unparsed += onDisk - r.files.length
    const seqs = r.files.map(f => f.seq)
    if (seqs[0] !== 0) notZero++
    for (let i = 1; i < seqs.length; i++) if (seqs[i] !== seqs[i - 1]! + 1) gaps++
  }
  // Checkup errors, null measures, CPU files, restores, code shas, identical filters / measures.
  let errLives = 0, errCks = 0, nullMeasures = 0, cpuLives = 0, restoreLives = 0, skipLives = 0, multiCode = 0, sameMeasures = 0
  const errReasons: Record<string, number> = {}
  const cpu: string[] = [], multi: string[] = []
  const filterSeen = new Map<string, string>()
  let dupFiltersAcross = 0
  const restored: Row[] = []
  for (const r of scope) {
    const cks = checkupsAt(r.window)
    let e = 0
    for (const c of cks.values()) {
      if (c.error) { e++; for (const [k, v] of Object.entries(c.errorReasons ?? {})) errReasons[`${k}: ${v}`] = (errReasons[`${k}: ${v}`] ?? 0) + 1 }
      nullMeasures += Object.values(c.measures).filter(v => v === null).length
    }
    if (e) { errLives++; errCks += e }
    if (r.window.some(f => !f.hostSig.webgpu)) { cpuLives++; cpu.push(r.learnerId) }
    const last = r.window[r.window.length - 1]!
    if (last.restores > 0) { restoreLives++; restored.push(r) }
    if (last.skippedUpdates > 0) skipLives++
    if (new Set(r.window.map(f => f.codeSha)).size > 1) { multiCode++; multi.push(r.learnerId) }
    const filters = r.window.map(f => f.meta?.filters).filter((v): v is string => typeof v === "string")
    for (const fl of filters) { const prev = filterSeen.get(fl); if (prev && prev !== r.learnerId) dupFiltersAcross++; filterSeen.set(fl, r.learnerId) }
    const ms = [...cks.values()].map(c => JSON.stringify(c.measures))
    if (new Set(ms).size < ms.length) sameMeasures++
  }
  issues.push(["log files on disk that do not parse (dropped by the loader)", unparsed])
  issues.push(["lives whose first parsed seq is not 0 / seq gaps", `${notZero} / ${gaps}`])
  issues.push(["lives with a checkup error:true at 512/1024/2048 (checkups)", `${errLives} (${errCks})`])
  issues.push(["null measures at 512/1024/2048 (read as 0)", nullMeasures])
  issues.push(["lives with a file in the budget window on CPU (webgpu false)", `${cpuLives}: ${cpu.join(", ")}`])
  issues.push(["lives with restores > 0 / skipped updates > 0 by the budget", `${restoreLives} / ${skipLives}`])
  issues.push(["lives spanning more than one codeSha in the budget window", `${multiCode}: ${multi.join(", ")}`])
  issues.push(["filter snapshots shared between different lives", dupFiltersAcross])
  issues.push(["lives with identical checkup measures at two levels", sameMeasures])
  say(`Scope: the 60 calibration lives and all centre-genome lives (${scope.length} distinct).`)
  say(table(["check", "count"], issues))
  if (Object.keys(errReasons).length) say(`error reasons: ${JSON.stringify(errReasons)}`)
  const rRes = restored.map(r => r.out.score! - mean(scope.filter(x => x.genome === r.genome).map(x => x.out.score!)))
  say(`restored lives (GPU run failures, app/src/life.ts; each restore redoes the CU since the last checkpoint): ${restored.length}, lostCU median ${f3(median(restored.map(r => r.window[r.window.length - 1]!.lostCU)), 0)}; their score minus their genome's mean: mean ${f3(mean(rRes))} (n=${rRes.length})`)

  // Filter snapshots (conv1 of both encoders, spec §5.1): how far the first layer moves from 512 to 2,048 CU.
  const snaps = scope.map(r => ({ r, f: filterSnapshots(r.window) })).filter(x => x.f.length === 3)
  const rel = (a: Float32Array, b: Float32Array) => { let d = 0, n = 0; for (let i = 0; i < a.length; i++) { d += (a[i]! - b[i]!) ** 2; n += a[i]! ** 2 } return Math.sqrt(d / n) }
  const same = (a: Float32Array, b: Float32Array) => a.every((v, i) => v === b[i])
  const moves = snaps.map(x => ({ r: x.r, m: rel(x.f[0]!, x.f[2]!), frozen512: same(x.f[0]!, x.f[2]!), frozen1024: same(x.f[1]!, x.f[2]!) }))
  say()
  say(`conv1 filter snapshots (${moves.length} lives with all three): relative L2 change 512 → 2,048 CU: median ${f3(median(moves.map(x => x.m)))}, `
    + `< 1%: ${moves.filter(x => x.m < 0.01).length}; bit-identical 512 = 2,048: ${moves.filter(x => x.frozen512).length} (${moves.filter(x => x.frozen512).map(x => x.r.learnerId).join(", ")}); `
    + `bit-identical 1,024 = 2,048: ${moves.filter(x => x.frozen1024).length}`)
  const calMoves = moves.filter(x => cal.includes(x.r))
  for (const k of ["score", "C", "H"]) say(`  Spearman(filter change, ${k}) over the 60: ${f3(spearman(calMoves.map(x => x.m), calMoves.map(x => x.r.comps[k]!)))}`)
  const mg = order.map(g => calMoves.filter(x => x.r.genome === g).map(x => x.m))
  const ma = anova1(mg)
  say(`  filter change across the 10 genomes: ICC ${f3(ma.icc)} [${iccCi(ma).map(v => f3(v, 2)).join(", ")}], KW p ${pf(kruskalWallis(mg).p)}`)
  json.filters = { median: median(moves.map(x => x.m)), under1pct: moves.filter(x => x.m < 0.01).length, frozen512: moves.filter(x => x.frozen512).map(x => x.r.learnerId), frozen1024: moves.filter(x => x.frozen1024).length, icc: ma.icc }

  // Roles / methods / eligibility over every search life.
  const reasons: Record<string, number> = {}
  for (const r of all) { const k = `${r.method}: ${r.out.eligible ? "eligible" : r.out.reason}`; reasons[k] = (reasons[k] ?? 0) + 1 }
  say()
  say(`every search life on disk by method and eligibility: ${JSON.stringify(reasons)}`)

  // Codes and hosts.
  const codeOf = (r: Row) => [...new Set(r.window.map(f => f.codeSha.slice(0, 7)))].join("+")
  const codes: Record<string, number> = {}, hosts: Record<string, number> = {}
  for (const r of scope) {
    codes[codeOf(r)] = (codes[codeOf(r)] ?? 0) + 1
    const h = r.window[r.window.length - 1]!.hostSig
    const k = `cores ${h.cores}, mem ${h.deviceMemoryGb}, tabs ${h.tabsOnMachine}`
    hosts[k] = (hosts[k] ?? 0) + 1
  }
  say(`codeSha in the budget window: ${JSON.stringify(codes)}; hosts: ${JSON.stringify(hosts)}`)

  // Era: the calibration lives ran on two page builds; does score differ by build within genome?
  const resid = (rows: Row[]) => rows.map(r => {
    const g = cal.filter(x => x.genome === r.genome).map(x => x.out.score!)
    return { r, e: r.out.score! - mean(g) }
  })
  const res = resid(cal)
  const early = res.filter(x => codeOf(x.r).includes("369c5eb")).map(x => x.e), late = res.filter(x => !codeOf(x.r).includes("369c5eb")).map(x => x.e)
  say(`build era (calibration lives, score minus own-genome mean): 369c5eb-era n=${early.length} mean ${f3(mean(early))}; 249b7d0-era n=${late.length} mean ${f3(mean(late))}`)
  const kwEra = kruskalWallis([early, late])
  say(`  KW p (era) = ${pf(kwEra.p)}`)
  const cpuRes = res.filter(x => x.r.window.some(f => !f.hostSig.webgpu)).map(x => x.e)
  if (cpuRes.length) say(`CPU-touched calibration lives: residual scores ${cpuRes.map(v => f3(v, 2)).join(", ")} (SD of all residuals ${f3(Math.sqrt(variance(res.map(x => x.e))))})`)

  // Panels: balance across genomes and the panel effect within the centre pool.
  say()
  say("Panels (shared checkup worlds, 1–16) per calibration genome:")
  say(table(["genome", "panels"], order.map(g => [short(g), cal.filter(r => r.genome === g).map(r => r.panel).join(", ")])))
  const byPanel = new Map<number, number[]>()
  for (const r of centre) byPanel.set(r.panel!, [...(byPanel.get(r.panel!) ?? []), r.out.score!])
  const pg = [...byPanel.values()].filter(g => g.length >= 1)
  const pa = anova1(pg)
  const pkw = kruskalWallis(pg)
  say(`panel effect on score within the centre pool (n=${centre.length}, ${pg.length} panels): ICC ${f3(pa.icc)} [${iccCi(pa).map(v => f3(v, 2)).join(", ")}], KW p ${pf(pkw.p)}, F p ${pf(pa.p)}`)
  for (const comp of ["C", "H"]) {
    const g2 = [...new Map(centre.map(r => [r.panel!, centre.filter(x => x.panel === r.panel).map(x => x.comps[comp]!)])).values()]
    const a2 = anova1(g2)
    say(`  on ${comp}: ICC ${f3(a2.icc)} [${iccCi(a2).map(v => f3(v, 2)).join(", ")}], F p ${pf(a2.p)}`)
  }
  // Panel-adjusted rerun of the KW on a_i (exploratory: panel means from the post-flat centre lives only).
  json.dataQuality = { unparsed, notZero, gaps, errLives, errCks, nullMeasures, cpu, restoreLives, skipLives, multi, dupFiltersAcross, sameMeasures, reasons, codes, hosts, era: { early: mean(early), late: mean(late), p: kwEra.p }, panelIcc: pa.icc, panelP: pa.p }
}

/** The life's conv1 filter snapshots at 512, 1,024 and 2,048 CU (meta.filters, float32 LE), in level order. */
function filterSnapshots(window: LogFileV2[]): Float32Array[] {
  return window.filter(f => typeof f.meta?.filters === "string").slice(0, 3).map(f => {
    const b = Buffer.from(f.meta!.filters!, "base64")
    return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
  })
}

function cost(all: Row[], cal: Row[], postFlat: Row[], say: (s?: string) => void, json: Record<string, unknown>): void {
  const hours = (r: Row) => (Date.parse(r.window[r.window.length - 1]!.writtenAt) - Date.parse(r.files[0]!.writtenAt)) / 3.6e6
  const eligible = all.filter(r => r.out.eligible)
  const h = eligible.map(hours)
  const recent = [...cal, ...postFlat].filter(r => r.window.every(f => f.codeSha.startsWith("249b7d0"))).map(hours)
  const q = (xs: number[], p: number) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]! }
  say(`wall time from a life's first log file to its 2,048-CU checkup file (the life starts a little before its first file):`)
  say(`  all eligible search lives n=${h.length}: median ${f3(median(h), 2)} h, IQR ${f3(q(h, 0.25), 2)}–${f3(q(h, 0.75), 2)} h, mean ${f3(mean(h), 2)} h`)
  say(`  on the current build (249b7d0) only n=${recent.length}: median ${f3(median(recent), 2)} h, mean ${f3(mean(recent), 2)} h`)
  // Throughput since flat: eligible search lives whose budget file landed after the look-3 commit's last life.
  const ends = eligible.map(r => Date.parse(r.window[r.window.length - 1]!.writtenAt)).sort((a, b) => a - b)
  const flatAt = Math.max(...cal.map(r => Date.parse(r.window[r.window.length - 1]!.writtenAt)))
  const since = ends.filter(t => t > flatAt)
  const days = (ends[ends.length - 1]! - flatAt) / 8.64e7
  say(`  since flat: ${since.length} eligible lives in ${f3(days, 1)} days = ${f3(since.length / days, 1)} lives/day (one runner machine; see hosts above)`)
  json.cost = { medianHours: median(h), meanHours: mean(h), medianHoursCurrent: median(recent), meanHoursCurrent: mean(recent), sinceFlat: since.length, days, perDay: since.length / days }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main()
