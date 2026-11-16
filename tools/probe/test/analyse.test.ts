import { describe, it, expect } from "vitest"
import { createRng, hashSeed } from "@blanktab/detmath"
import { GPU_WIDTHS } from "@blanktab/learner"
import { serialise } from "../../gate/serialise.ts"
import { BAR, PERMUTATIONS, analyse, permutationP, stateOf, type ProbeReport } from "../filters/analyse.ts"
import { LAYER, probeGrams } from "../filters/layer.ts"
import { initFirstLayer, layerArrays, type Encoder, type FirstLayer, type ProbeLife } from "../filters/lives.ts"
import { cka } from "../filters/metrics.ts"
import { mapLayer } from "./fixture-lives.ts"

const GA = "a1".repeat(32), GB = "b2".repeat(32), GC = "c3".repeat(32)
const id = (k: number) => "lrn_" + k.toString(16).padStart(16, "0")

/** Seeded smooth synthetic frames (CHW, 0..1): a few random gratings each, so neighbouring pixels correlate. */
function frames(e: Encoder, n: number): Float32Array[] {
  const s = LAYER[e]
  return Array.from({ length: n }, (_, f) => {
    const r = createRng(hashSeed(f, `analyse-test/${e}`))
    const waves = Array.from({ length: 4 }, () => ({ fx: r() * 0.8, fy: r() * 0.8, ph: r() * 6.28, c: [r(), r(), r()] }))
    const x = new Float32Array(3 * s.inH * s.inW)
    for (let c = 0; c < 3; c++) for (let y = 0; y < s.inH; y++) for (let xx = 0; xx < s.inW; xx++) {
      let v = 0.5
      for (const w of waves) v += 0.12 * w.c[c]! * Math.sin(w.fx * xx + w.fy * y + w.ph)
      x[(c * s.inH + y) * s.inW + xx] = Math.min(1, Math.max(0, v + 0.05 * (r() - 0.5)))
    }
    return x
  })
}
const INPUTS = { periphery: frames("periphery", 4), fovea: frames("fovea", 6) }

const noise = (f: FirstLayer, seed: number, amp: number) => { const r = createRng(seed); return mapLayer(f, v => v + amp * (2 * r() - 1)) }
/** f with its 16 units reordered by a seeded permutation (both encoders, weights and biases together). */
function permuteUnits(f: FirstLayer, seed: number): FirstLayer {
  const r = createRng(seed), pi = Array.from({ length: 16 }, (_, i) => i)
  for (let i = 15; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [pi[i], pi[j]] = [pi[j]!, pi[i]!] }
  const re = (w: Float32Array, k: number) => { const o = new Float32Array(w.length); pi.forEach((src, u) => o.set(w.subarray(src * k, src * k + k), u * k)); return o }
  return { periphery: { w: re(f.periphery.w, 75), b: re(f.periphery.b, 1) }, fovea: { w: re(f.fovea.w, 27), b: re(f.fovea.b, 1) } }
}
/** A life whose 2,048 CU state is `end(k)`; 512 and 1,024 move from init toward it. */
function life(k: number, genomeSha: string, end: (init: FirstLayer, k: number) => FirstLayer): ProbeLife {
  const seed = 1000 + k
  const init = initFirstLayer(seed, GPU_WIDTHS)
  const at2048 = end(init, k)
  const toward = (t: number) => { const a = layerArrays(init), b = layerArrays(at2048); return mapLayer(init, (v, x, i) => v + t * (b[x]![i]! - a[x]![i]!)) }
  return { learnerId: id(k), genomeSha, seed, init, at: { 512: toward(0.1), 1024: toward(0.5), 2048: at2048 } }
}
const SHARED = initFirstLayer(424242, GPU_WIDTHS)
const converged = (_: FirstLayer, k: number) => noise(permuteUnits(SHARED, k), hashSeed(k, "n"), 0.005)
const fresh = (_: FirstLayer, k: number) => initFirstLayer(50_000 + k, GPU_WIDTHS)
const run = (lives: ProbeLife[]): ProbeReport => analyse(lives, probeGrams(lives, INPUTS), GA, "f0".repeat(32))
const get = (r: ProbeReport, name: string, e: Encoder) => r.convergence.find(a => a.name === name && a.encoder === e)!

describe("filter probe analyses", () => {
  it("BAR is exactly { minLives 10, ckaDelta 0.1, matched 0.5, alpha 0.05 } (spec §5.1)", () => {
    expect({ ...BAR }).toEqual({ minLives: 10, ckaDelta: 0.1, matched: 0.5, alpha: 0.05 })
    expect(Object.isFrozen(BAR)).toBe(true)
    expect(PERMUTATIONS).toBe(9_999)
  })

  it("12 planted lives whose 2048 filters are one shared filter set plus small seeded noise, units permuted per life: pooled is evaluable and converged, matched near 1, p = exact floor", () => {
    const r = run(Array.from({ length: 12 }, (_, k) => life(k, GA, converged)))
    expect(r.status).toBe("ok")
    for (const e of ["periphery", "fovea"] as const) {
      const a = get(r, "pooled", e)
      expect(a.verdict.evaluable, e).toBe(true)
      expect(a.verdict.converged, e).toBe(true)
      expect(a.verdict.matched).toBeGreaterThan(0.95)
      expect(a.verdict.p).toBe(2 ** -12)
      expect(a.verdict.pFloor).toBe(2 ** -12)
      expect(a.stats).toMatchObject({ n: 12, pairs: 66, skipped: 0 })
    }
  })

  it("12 planted lives whose 2048 filters are fresh random draws: evaluable, not converged", () => {
    const r = run(Array.from({ length: 12 }, (_, k) => life(k, GA, fresh)))
    for (const e of ["periphery", "fovea"] as const) {
      const a = get(r, "pooled", e)
      expect(a.verdict.evaluable).toBe(true)
      expect(a.verdict.converged).toBe(false)
      expect(Math.abs(a.verdict.ckaDelta)).toBeLessThan(BAR.ckaDelta)
    }
  })

  it("4 lives of one genome: not evaluable, reasons 'n = 4 < 10 lives' and 'p cannot reach 0.05 (floor 0.0625)', converged null", () => {
    const r = run([...Array.from({ length: 4 }, (_, k) => life(k, GB, converged)), ...Array.from({ length: 8 }, (_, k) => life(20 + k, GC, fresh))])
    const a = get(r, `genome:${GB}`, "periphery").verdict
    expect(a.evaluable).toBe(false)
    expect(a.reasons).toEqual(["n = 4 < 10 lives", "p cannot reach 0.05 (floor 0.0625)"])
    expect(a.converged).toBeNull()
    expect(a.p).toBeNull()
    expect(a.pHolm).toBeNull()
  })

  it("5 lives of one genome: not evaluable with the one reason 'n = 5 < 10 lives' (floor 0.03125), converged null even when p < 0.05", () => {
    const r = run(Array.from({ length: 5 }, (_, k) => life(k, GB, converged)))
    const a = get(r, `genome:${GB}`, "periphery").verdict
    expect(a.reasons).toEqual(["n = 5 < 10 lives"])
    expect(a.pFloor).toBe(0.03125)
    expect(a.pDescriptive).toBeLessThan(0.05)
    expect(a.converged).toBeNull()
    expect(a.p).toBeNull()
  })

  it("permutationP exact and monteCarlo agree within 0.01 on the same 14-life fixture", () => {
    const lives = Array.from({ length: 14 }, (_, k) => life(k, GA, (init, kk) => (kk % 3 === 0 ? converged(init, kk) : fresh(init, kk))))
    const G = probeGrams(lives, INPUTS).periphery
    const c = (a: number, b: number) => cka(G, a, b)!
    const pairs: [number, number][] = []
    for (let i = 0; i < 14; i++) for (let j = i + 1; j < 14; j++) pairs.push([i, j])
    const T = (s: Uint8Array) => pairs.reduce((acc, [i, j]) => acc + c(stateOf(i, !s[i]), stateOf(j, !s[j])) - c(stateOf(i, !!s[i]), stateOf(j, !!s[j])), 0) / pairs.length
    const ex = permutationP(T, 14, "exact", 0), mc = permutationP(T, 14, "monteCarlo", hashSeed(0, "t"))
    expect(ex.floor).toBe(2 ** -14)
    expect(mc.floor).toBe(1 / 10_000)
    expect(ex.p).toBeGreaterThan(0.01)
    expect(Math.abs(ex.p - mc.p)).toBeLessThan(0.01)
  })

  it("the recipe control finds within > between on planted genome-specific filters, and p ≥ 0.05 when filters ignore the genome", () => {
    const sets = [GA, GB, GC].map((_, g) => initFirstLayer(7000 + g, GPU_WIDTHS))
    const planted = run(Array.from({ length: 12 }, (_, k) => life(k, [GA, GB, GC][k % 3]!, (_i, kk) => noise(permuteUnits(sets[kk % 3]!, kk), kk, 0.005))))
    const ignore = run(Array.from({ length: 12 }, (_, k) => life(k, [GA, GB, GC][k % 3]!, fresh)))
    for (const e of ["periphery", "fovea"] as const) {
      const p = planted.recipe.find(x => x.encoder === e)!, q = ignore.recipe.find(x => x.encoder === e)!
      expect(p.genomes).toBe(3)
      expect(p.within).toBeGreaterThan(p.between)
      expect(p.p).toBeLessThan(0.05)
      expect(q.p).toBeGreaterThanOrEqual(0.05)
    }
  })

  it("identification correlates all 1,664 floats in FIRST_LAYER_PARAMS order: a 512 state equal to the init gives ownInit 1", () => {
    const lives = [0, 1, 2].map(k => life(k, GA, converged))
    lives[1] = { ...lives[1]!, at: { ...lives[1]!.at, 512: lives[1]!.init } }
    expect(layerArrays(lives[1]!.init).reduce((a, x) => a + x.length, 0)).toBe(1664)
    const r = run(lives)
    expect(r.identification[1]!.ownInit).toBe(1)
    expect(r.identification[0]!.ownInit).toBeGreaterThan(0.9)
    expect(r.identification[0]!.bestOther).toBeLessThan(0.2)
  })

  it("identification: a life whose seed is swapped with another's is caught (ownInit ≤ bestOther, bestOtherId names the other); status is 'identification failed', every verdict has converged and convergedHolm null and the reason, and the numbers are still there", () => {
    const lives = Array.from({ length: 12 }, (_, k) => life(k, GA, converged))
    const a = lives[3]!, b = lives[7]!
    lives[3] = { ...a, init: b.init, seed: b.seed }
    lives[7] = { ...b, init: a.init, seed: a.seed }
    const r = run(lives)
    expect(r.status).toBe("identification failed")
    const i3 = r.identification[3]!
    expect(i3.ownInit).toBeLessThanOrEqual(i3.bestOther!)
    expect(i3.bestOtherId).toBe(id(7))
    for (const x of r.convergence) {
      expect(x.verdict.evaluable).toBe(false)
      expect(x.verdict.converged).toBeNull()
      expect(x.verdict.convergedHolm).toBeNull()
      expect(x.verdict.reasons).toContain("identification failed for 2 lives")
      expect(Number.isFinite(x.verdict.ckaDelta)).toBe(true)
      expect(Number.isFinite(x.stats.learned.cka)).toBe(true)
    }
  })

  it("a kept life whose periphery motion at 2048 is not above its motion at 512 is listed in flags, and stays in every analysis", () => {
    const lives = Array.from({ length: 4 }, (_, k) => life(k, GA, converged))
    lives[2] = { ...lives[2]!, at: { ...lives[2]!.at, 2048: lives[2]!.at[512] } }
    const r = run(lives)
    expect(r.flags).toContainEqual({ learnerId: id(2), flag: "periphery motion did not grow 512→2048" })
    expect(r.flags.filter(f => f.flag.includes("motion")).map(f => f.learnerId)).toEqual([id(2), id(2)])
    for (const x of r.convergence) expect(x.learnerIds).toContain(id(2))
    expect(r.motion[2]!.periphery[2]).toBe(r.motion[2]!.periphery[0])
  })

  it("each verdict carries its Holm-adjusted p and the verdict at that p, beside the raw one", () => {
    const r = run(Array.from({ length: 12 }, (_, k) => life(k, GA, k % 2 ? converged : fresh)))
    const [p, f] = [get(r, "pooled", "periphery").verdict, get(r, "pooled", "fovea").verdict]
    const [lo, hi] = p.p! <= f.p! ? [p, f] : [f, p]
    expect(lo.pHolm).toBe(Math.min(1, 2 * lo.p!))
    expect(hi.pHolm).toBe(Math.max(lo.pHolm!, hi.p!))
    for (const v of [p, f]) {
      expect(typeof v.convergedHolm).toBe("boolean")
      expect(v.convergedHolm).toBe(v.ckaDelta >= 0.1 && v.matched >= 0.5 && v.pHolm! < 0.05)
    }
  })

  it("analyse run twice gives identical serialised bytes", () => {
    const lives = Array.from({ length: 6 }, (_, k) => life(k, k < 3 ? GA : GB, k % 2 ? converged : fresh))
    expect(serialise(run(lives))).toBe(serialise(run(lives)))
  })
})
