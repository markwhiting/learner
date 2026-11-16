// The filter probe's analyses (plan 05c Task 5): convergence per encoder, pooled across genomes and within each genome,
// against the init-pair control, with a within-life permutation test; the between-genome (recipe) control; weight
// motion; and the identification check that every life's logged filters are its own. Every rule here is an agent's
// call recorded in decision 0027 and fixed before the probe first ran on real lives (invariant 7).
import { createRng, hashSeed } from "@blanktab/detmath"
import type { Gram } from "./layer.ts"
import { ENCODERS, layerArrays, weightMotion, type Encoder, type Excluded, type FirstLayer, type ProbeLife } from "./lives.ts"
import { UNITS, cka, matchedMedian, median } from "./metrics.ts"

/** Spec §5.1's bar, a [guess]. Changing any value needs a recorded decision (invariant 7); a test pins them. */
export const BAR = Object.freeze({ minLives: 10, ckaDelta: 0.1, matched: 0.5, alpha: 0.05 })
/** Monte Carlo draws when exact enumeration would exceed 2^16 relabellings. */
export const PERMUTATIONS = 9_999
/** Exact enumeration up to this many lives (2^16 relabellings). */
export const EXACT_MAX_LIVES = 16

export interface PairStats { learned: { cka: number; matched: number }; init: { cka: number; matched: number }; n: number; pairs: number; skipped: number }
export interface Verdict {
  evaluable: boolean; reasons: string[]; converged: boolean | null; ckaDelta: number; matched: number; p: number | null; pFloor: number
  /** Holm over the analysis's two encoders; reported, never the bar. */
  pHolm: number | null; convergedHolm: boolean | null
  /** The permutation p whether or not the analysis is evaluable: description only when `p` is null. */
  pDescriptive: number
}
export interface Analysis { name: string; encoder: Encoder; learnerIds: string[]; stats: PairStats; verdict: Verdict }
export interface RecipeControl { encoder: Encoder; within: number; between: number; p: number; genomes: number }
export interface Identification { learnerId: string; ownInit: number; bestOther: number | null; bestOtherId: string | null }
export interface ProbeReport {
  schemaVersion: 1; manifestSha: string; centreSha: string; lives: number; excluded: Excluded[]
  /** "identification failed": every verdict is withheld; the numbers stay, descriptive. */
  status: "ok" | "identification failed"
  /** "pooled" and "genome:<sha>" (every genome with ≥ 2 lives; the centre's first), per encoder. */
  convergence: Analysis[]
  /** The between-genome control. */
  recipe: RecipeControl[]
  /** Weight motion at 512, 1,024 and 2,048 CU. */
  motion: { learnerId: string; genomeSha: string; periphery: number[]; fovea: number[] }[]
  /** Kept lives worth a look. */
  flags: { learnerId: string; flag: string }[]
  identification: Identification[]
}

/** In a Gram over lives' states, life i's init is state 2i and its 2,048 CU state is 2i + 1. */
export const stateOf = (life: number, learned: boolean): number => 2 * life + (learned ? 1 : 0)

/**
 * The permutation p of T: the share of relabellings s ∈ {0,1}ⁿ with T(s) ≥ T(0), counting s = 0. "exact" enumerates all
 * 2ⁿ (floor 2⁻ⁿ); "monteCarlo" draws PERMUTATIONS relabellings from createRng(seed), p = (1 + hits) / (1 + PERMUTATIONS).
 */
export function permutationP(t: (s: Uint8Array) => number, n: number, how: "exact" | "monteCarlo", seed: number): { p: number; floor: number } {
  const s = new Uint8Array(n)
  const t0 = t(s)
  if (how === "exact") {
    let hits = 0
    const total = 2 ** n
    for (let m = 0; m < total; m++) {
      for (let i = 0; i < n; i++) s[i] = (m >>> i) & 1
      if (t(s) >= t0) hits++
    }
    return { p: hits / total, floor: 1 / total }
  }
  const rnd = createRng(seed)
  let hits = 0
  for (let k = 0; k < PERMUTATIONS; k++) {
    for (let i = 0; i < n; i++) s[i] = rnd() < 0.5 ? 1 : 0
    if (t(s) >= t0) hits++
  }
  return { p: (1 + hits) / (1 + PERMUTATIONS), floor: 1 / (1 + PERMUTATIONS) }
}

/** Pearson correlation in float64. */
function pearson(x: Float64Array, y: Float64Array): number {
  const n = x.length
  let mx = 0, my = 0
  for (let i = 0; i < n; i++) { mx += x[i]!; my += y[i]! }
  mx /= n; my /= n
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) { const a = x[i]! - mx, b = y[i]! - my; sxy += a * b; sxx += a * a; syy += b * b }
  return sxy / Math.sqrt(sxx * syy)
}
/** All 1,664 first-layer floats of a state, in FIRST_LAYER_PARAMS order (periphery w, b, fovea w, b). */
const flat = (f: FirstLayer): Float64Array => Float64Array.from(layerArrays(f).flatMap(a => Array.from(a)))

/** Review Focus 1: each life's 512 CU state against its own recomputed init and against every other life's init. */
export function identify(lives: readonly ProbeLife[]): Identification[] {
  const inits = lives.map(l => flat(l.init))
  return lives.map((l, i) => {
    const at = flat(l.at[512])
    let bestOther: number | null = null, bestOtherId: string | null = null
    lives.forEach((o, j) => {
      if (j === i) return
      const r = pearson(at, inits[j]!)
      if (bestOther === null || r > bestOther) { bestOther = r; bestOtherId = o.learnerId }
    })
    return { learnerId: l.learnerId, ownInit: pearson(at, inits[i]!), bestOther, bestOtherId }
  })
}

/** Per encoder: CKA between every two states, and the matched medians of every pair at init and at 2,048. */
interface PairTable { ii: number | null; il: number | null; li: number | null; ll: number | null; mInit: number; mLearned: number }
function pairTable(G: Gram, n: number): PairTable[] {
  const out: PairTable[] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const I = (k: number) => stateOf(k, false), L = (k: number) => stateOf(k, true)
    out.push({
      ii: cka(G, I(i), I(j)), il: cka(G, I(i), L(j)), li: cka(G, L(i), I(j)), ll: cka(G, L(i), L(j)),
      mInit: matchedMedian(G, I(i), I(j)), mLearned: matchedMedian(G, L(i), L(j)),
    })
  }
  return out
}
/** The index of pair (i, j), i < j, in pairTable's order over n lives. */
const pairIndex = (i: number, j: number, n: number) => i * n - (i * (i + 1)) / 2 + (j - i - 1)

const mean = (xs: readonly number[]) => xs.reduce((a, v) => a + v, 0) / xs.length

/** One analysis over the lives `idx` (indices into the probe's lives) for one encoder: the statistics and the permutation p. */
function analyseSet(table: PairTable[], n: number, idx: readonly number[], name: string, encoder: Encoder): { stats: PairStats; ckaDelta: number; perm: { p: number; floor: number } } {
  const pairs: { a: number; b: number; t: PairTable }[] = []
  let skipped = 0
  for (let x = 0; x < idx.length; x++) for (let y = x + 1; y < idx.length; y++) {
    const t = table[pairIndex(idx[x]!, idx[y]!, n)]!
    if (t.ii === null || t.il === null || t.li === null || t.ll === null) skipped++
    else pairs.push({ a: x, b: y, t })
  }
  const learnedCka = mean(pairs.map(p => p.t.ll!)), initCka = mean(pairs.map(p => p.t.ii!))
  const stats: PairStats = {
    learned: { cka: learnedCka, matched: median(pairs.map(p => p.t.mLearned)) },
    init: { cka: initCka, matched: median(pairs.map(p => p.t.mInit)) },
    n: idx.length, pairs: pairs.length, skipped,
  }
  // T(s): ckaDelta with life k's two states swapped where s_k = 1 (the null: which state is "learned" is exchangeable).
  const d = pairs.map(p => [p.t.ll! - p.t.ii!, p.t.li! - p.t.il!, p.t.il! - p.t.li!, p.t.ii! - p.t.ll!])
  const T = (s: Uint8Array) => {
    let sum = 0
    for (let k = 0; k < pairs.length; k++) sum += d[k]![s[pairs[k]!.a]! * 2 + s[pairs[k]!.b]!]!
    return sum / pairs.length
  }
  const how = idx.length <= EXACT_MAX_LIVES ? "exact" : "monteCarlo"
  const perm = permutationP(T, idx.length, how, hashSeed(0, `filter-probe/perm/${name}/${encoder}`))
  return { stats, ckaDelta: learnedCka - initCka, perm }
}

/** Holm's adjustment of a family of p values (same order). */
function holm(ps: readonly number[]): number[] {
  const order = ps.map((p, i) => [p, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const out = new Array<number>(ps.length)
  let run = 0
  order.forEach(([p, i], k) => { run = Math.max(run, Math.min(1, (ps.length - k) * p)); out[i] = run })
  return out
}

/** The between-genome control for one encoder over lives whose genome has ≥ 2 lives, or null with fewer than 2 such genomes. */
function recipeControl(table: PairTable[], n: number, lives: readonly ProbeLife[], encoder: Encoder): RecipeControl | null {
  const count = new Map<string, number>()
  for (const l of lives) count.set(l.genomeSha, (count.get(l.genomeSha) ?? 0) + 1)
  const idx = lives.map((l, i) => i).filter(i => count.get(lives[i]!.genomeSha)! >= 2)
  const genomes = [...new Set(idx.map(i => lives[i]!.genomeSha))].sort()
  if (genomes.length < 2) return null
  const label = Int32Array.from(idx, i => genomes.indexOf(lives[i]!.genomeSha))
  const pairs: { a: number; b: number; v: number }[] = []
  for (let x = 0; x < idx.length; x++) for (let y = x + 1; y < idx.length; y++) {
    const v = table[pairIndex(idx[x]!, idx[y]!, n)]!.ll
    if (v !== null) pairs.push({ a: x, b: y, v })
  }
  const split = (lab: Int32Array) => {
    let ws = 0, wn = 0, bs = 0, bn = 0
    for (const p of pairs) if (lab[p.a] === lab[p.b]) { ws += p.v; wn++ } else { bs += p.v; bn++ }
    return { within: ws / wn, between: bs / bn }
  }
  const obs = split(label)
  const stat0 = obs.within - obs.between
  const rnd = createRng(hashSeed(0, `filter-probe/recipe/${encoder}`))
  const lab = Int32Array.from(label)
  let hits = 0
  for (let k = 0; k < PERMUTATIONS; k++) {
    lab.set(label)
    for (let i = lab.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = lab[i]!; lab[i] = lab[j]!; lab[j] = t }
    const s = split(lab)
    if (s.within - s.between >= stat0) hits++
  }
  return { encoder, within: obs.within, between: obs.between, p: (1 + hits) / (1 + PERMUTATIONS), genomes: genomes.length }
}

/**
 * The probe report from the eligible lives (sorted by learnerId) and each encoder's centred Gram over their states
 * (stateOf: init, then 2,048 CU, per life). Pure: no IO, no clock; the same inputs give the same serialised bytes.
 */
export function analyse(lives: ProbeLife[], grams: Record<Encoder, Gram>, centreSha: string, manifestSha: string, excluded: readonly Excluded[] = []): ProbeReport {
  const n = lives.length
  const identification = identify(lives)
  const misidentified = identification.filter(r => r.bestOther !== null && r.ownInit <= r.bestOther).length
  const status = misidentified > 0 ? "identification failed" : "ok"

  const byGenome = new Map<string, number[]>()
  lives.forEach((l, i) => byGenome.set(l.genomeSha, [...(byGenome.get(l.genomeSha) ?? []), i]))
  const genomeSets = [...byGenome.entries()].filter(([, ix]) => ix.length >= 2)
    .sort(([a], [b]) => (a === centreSha ? -1 : b === centreSha ? 1 : a < b ? -1 : a > b ? 1 : 0))
  const sets: [string, number[]][] = [["pooled", lives.map((_, i) => i)], ...genomeSets.map(([g, ix]) => [`genome:${g}`, ix] as [string, number[]])]

  const tables = Object.fromEntries(ENCODERS.map(e => [e, pairTable(grams[e], n)])) as Record<Encoder, PairTable[]>
  const convergence: Analysis[] = []
  for (const [name, idx] of sets) {
    const per = ENCODERS.map(e => ({ e, ...analyseSet(tables[e], n, idx, name, e) }))
    const nOk = idx.length >= BAR.minLives
    const pHolm = holm(per.map(r => r.perm.p))
    per.forEach((r, k) => {
      const reasons: string[] = []
      if (!nOk) reasons.push(`n = ${idx.length} < ${BAR.minLives} lives`)
      if (!nOk && r.perm.floor >= BAR.alpha) reasons.push(`p cannot reach ${BAR.alpha} (floor ${r.perm.floor})`)
      if (status !== "ok") reasons.push(`identification failed for ${misidentified} lives`)
      const evaluable = reasons.length === 0
      const matched = r.stats.learned.matched
      const pass = (p: number) => r.ckaDelta >= BAR.ckaDelta && matched >= BAR.matched && p < BAR.alpha
      convergence.push({
        name, encoder: r.e, learnerIds: idx.map(i => lives[i]!.learnerId), stats: r.stats,
        verdict: {
          evaluable, reasons, converged: evaluable ? pass(r.perm.p) : null, ckaDelta: r.ckaDelta, matched,
          p: evaluable ? r.perm.p : null, pFloor: r.perm.floor,
          pHolm: evaluable ? pHolm[k]! : null, convergedHolm: evaluable ? pass(pHolm[k]!) : null, pDescriptive: r.perm.p,
        },
      })
    })
  }

  const recipe = ENCODERS.map(e => recipeControl(tables[e], n, lives, e)).filter((r): r is RecipeControl => r !== null)
  const motion = lives.map(l => ({
    learnerId: l.learnerId, genomeSha: l.genomeSha,
    periphery: ([512, 1024, 2048] as const).map(k => weightMotion(l.at[k], l.init, "periphery")),
    fovea: ([512, 1024, 2048] as const).map(k => weightMotion(l.at[k], l.init, "fovea")),
  }))
  const flags: { learnerId: string; flag: string }[] = []
  lives.forEach((l, i) => {
    for (const e of ENCODERS) if (!(motion[i]![e][2]! > motion[i]![e][0]!)) flags.push({ learnerId: l.learnerId, flag: `${e} motion did not grow 512→2048` })
    for (const e of ENCODERS) for (const learned of [false, true]) {
      const s = stateOf(i, learned)
      let dead = 0
      for (let u = 0; u < UNITS; u++) dead += grams[e].varZero[s * UNITS + u]!
      if (dead > 0) flags.push({ learnerId: l.learnerId, flag: `${e} ${learned ? "2048" : "init"}: ${dead} dead units on the probe set` })
    }
  })
  return { schemaVersion: 1, manifestSha, centreSha, lives: n, excluded: [...excluded], status, convergence, recipe, motion, flags, identification }
}
