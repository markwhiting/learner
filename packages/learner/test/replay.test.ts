import { describe, it, expect } from "vitest"
import { NOOP_ACTION, emptyObservation } from "@blanktab/interface"
import { ReplayBuffer, Reservoir, reservoirSlots, sampleStored } from "../src/replay"
import type { Transition } from "../src/replay"

const tr = (i: number): Transition => {
  const obs = emptyObservation()
  obs.periphery[0] = i
  return { obs, hIn: Float32Array.of(i), action: { ...NOOP_ACTION, move: i % 25 }, next: emptyObservation() }
}

describe("ReplayBuffer", () => {
  it("evicts the oldest entry first and never exceeds its capacity", () => {
    const b = new ReplayBuffer(3)
    expect(b.size).toBe(0)
    for (let i = 0; i < 10; i++) {
      b.push(tr(i))
      expect(b.size).toBe(Math.min(i + 1, 3))
      expect(b.size).toBeLessThanOrEqual(b.capacity)
      // Oldest first: at(0) is the earliest surviving transition, at(size-1) the newest.
      const ids = Array.from({ length: b.size }, (_, k) => b.at(k).hIn[0])
      expect(ids).toEqual(Array.from({ length: b.size }, (_, k) => Math.max(0, i - 2) + k))
    }
  })

  it("toArray lists the transitions oldest first, and pushing them in order rebuilds the same buffer", () => {
    const b = new ReplayBuffer(4)
    for (let i = 0; i < 7; i++) b.push(tr(i)) // wrapped: head is mid-ring
    const items = b.toArray()
    expect(items.map((t) => t.hIn[0])).toEqual([3, 4, 5, 6])
    const c = new ReplayBuffer(4)
    for (const t of items) c.push(t)
    expect(Array.from({ length: c.size }, (_, k) => c.at(k))).toEqual(items)
  })

  it("rejects an invalid capacity and out-of-range reads", () => {
    for (const c of [0, -1, 1.5, Number.NaN]) expect(() => new ReplayBuffer(c)).toThrow(RangeError)
    const b = new ReplayBuffer(2)
    b.push(tr(0))
    expect(() => b.at(1)).toThrow(RangeError)
    expect(() => b.at(-1)).toThrow(RangeError)
  })
})

/** A seeded uniform [0, 1) generator that counts its draws. */
const counted = (seed: number) => {
  let s = seed >>> 0
  const rng = () => {
    rng.draws++
    s = (s + 0x6d2b79f5) >>> 0
    let x = s
    x = Math.imul(x ^ (x >>> 15), x | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
  rng.draws = 0
  return rng
}

describe("Reservoir (Plan 06a)", () => {
  it("keeps everything while it has room, without drawing; then holds `capacity` and draws once per offer", () => {
    const rng = counted(1)
    const r = new Reservoir(4)
    for (let i = 0; i < 4; i++) r.offer(tr(i), rng)
    expect(rng.draws).toBe(0)
    expect(r.toArray().map(t => t.hIn[0])).toEqual([0, 1, 2, 3])
    for (let i = 4; i < 50; i++) {
      r.offer(tr(i), rng)
      expect(r.size).toBe(4)
      expect(rng.draws).toBe(i - 3)
    }
    expect(r.seen).toBe(50)
    expect(r.capacity).toBe(4)
  })

  it("is a uniform sample of everything offered: each of 200 offers survives in a 20-slot reservoir with probability 1/10", () => {
    const rng = counted(7)
    const kept = new Array<number>(200).fill(0)
    const TRIALS = 2000
    for (let trial = 0; trial < TRIALS; trial++) {
      const r = new Reservoir(20)
      for (let i = 0; i < 200; i++) r.offer(tr(i), rng)
      for (const t of r.toArray()) kept[t.hIn[0]!]!++
    }
    // Binomial(2000, 0.1): mean 200, sd 13.4. Every one of the 200 counts within 5 sd, early offers and late alike.
    expect(Math.min(...kept)).toBeGreaterThan(200 - 67)
    expect(Math.max(...kept)).toBeLessThan(200 + 67)
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(Math.abs(mean(kept.slice(0, 50)) - mean(kept.slice(150)))).toBeLessThan(8)
  })

  it("rebuilt from toArray() and seen, it continues exactly as the original", () => {
    const a = new Reservoir(5)
    const rngA = counted(3)
    for (let i = 0; i < 30; i++) a.offer(tr(i), rngA)
    const b = new Reservoir(5, { items: a.toArray(), seen: a.seen })
    const rngB = counted(99), rngA2 = counted(99)
    for (let i = 30; i < 80; i++) {
      a.offer(tr(i), rngA2)
      b.offer(tr(i), rngB)
    }
    expect(b.toArray().map(t => t.hIn[0])).toEqual(a.toArray().map(t => t.hIn[0]))
    expect(b.seen).toBe(80)
    expect(b.at(2).hIn[0]).toBe(a.at(2).hIn[0])
  })

  it("rejects an invalid capacity, an inconsistent state and out-of-range reads", () => {
    for (const c of [0, -1, 1.5, Number.NaN]) expect(() => new Reservoir(c)).toThrow(RangeError)
    const items = [tr(0), tr(1), tr(2)]
    expect(() => new Reservoir(2, { items, seen: 3 })).toThrow(RangeError) // more items than slots
    expect(() => new Reservoir(3, { items, seen: 2 })).toThrow(RangeError) // seen below what it holds
    expect(() => new Reservoir(4, { items, seen: 9 })).toThrow(RangeError) // not full, yet more were offered than kept
    expect(() => new Reservoir(3, { items, seen: 2.5 })).toThrow(RangeError)
    const r = new Reservoir(3, { items, seen: 9 })
    expect(() => r.at(3)).toThrow(RangeError)
    expect(() => r.at(-1)).toThrow(RangeError)
  })
})

describe("reservoirSlots (Plan 06a)", () => {
  it("share 0 or absent is none; a share of the buffer otherwise, always leaving one recent slot", () => {
    expect(reservoirSlots(2000, undefined)).toBe(0)
    expect(reservoirSlots(2000, 0)).toBe(0)
    expect(reservoirSlots(2000, 0.5)).toBe(1000)
    expect(reservoirSlots(2000, 1)).toBe(1999)
    expect(reservoirSlots(16, 0.26)).toBe(4)
    expect(reservoirSlots(1, 1)).toBe(0)
  })
})

describe("sampleStored (Plan 06a)", () => {
  it("without a reservoir it is the old draw: one rng call, recent.at(min(size − 1, floor(u · size)))", () => {
    const b = new ReplayBuffer(8)
    for (let i = 0; i < 8; i++) b.push(tr(i))
    const rng = counted(5), twin = counted(5)
    for (let k = 0; k < 40; k++) {
      const want = b.at(Math.min(b.size - 1, Math.floor(twin() * b.size)))
      expect(sampleStored(b, null, rng)).toBe(want)
    }
    expect(rng.draws).toBe(40)
  })

  it("with a reservoir it draws uniformly over the recent transitions followed by the kept ones", () => {
    const b = new ReplayBuffer(2)
    b.push(tr(10))
    b.push(tr(11))
    const r = new Reservoir(6)
    for (let i = 0; i < 6; i++) r.offer(tr(i), () => 0)
    const rng = counted(11)
    const hits = new Map<number, number>()
    for (let k = 0; k < 8000; k++) {
      const id = sampleStored(b, r, rng)!.hIn[0]!
      hits.set(id, (hits.get(id) ?? 0) + 1)
    }
    expect([...hits.keys()].sort((x, y) => x - y)).toEqual([0, 1, 2, 3, 4, 5, 10, 11])
    for (const n of hits.values()) expect(Math.abs(n - 1000)).toBeLessThan(150) // Binomial(8000, 1/8): sd 29.6
    expect(rng.draws).toBe(8000)
  })

  it("returns null when nothing is stored, without drawing", () => {
    const rng = counted(1)
    expect(sampleStored(new ReplayBuffer(3), null, rng)).toBeNull()
    expect(sampleStored(new ReplayBuffer(3), new Reservoir(3), rng)).toBeNull()
    expect(rng.draws).toBe(0)
  })
})
