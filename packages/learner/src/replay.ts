import type { Action, Observation } from "@blanktab/interface"

/**
 * One learning transition, kept as raw data so it can be replayed and checkpointed (no tensors).
 * Replaying it means: h = core(obsToInputs(obs), hIn), then predict(h, action, targetsFromObs(obs)) against
 * targetsFromObs(next) (the residual base is recomputed from `obs`, so it is not stored).
 * `obs` and `next` are the learner's own copies of the observations; consecutive transitions share the
 * frame between them (`next` of one is `obs` of the following), so a frame is stored once.
 */
export interface Transition {
  /** Observation fed to the core (inputs_{t-1}). */
  obs: Observation
  /** Recurrent state fed to the core with `obs` (h_{t-2}, detached), [hidden] floats. */
  hIn: Float32Array
  /** Action taken after `obs` (a_{t-1}). */
  action: Action
  /** Observation that followed (its targets are y_t). */
  next: Observation
}

/** FIFO ring of the last `capacity` transitions: pushing into a full buffer evicts the oldest. */
export class ReplayBuffer {
  readonly capacity: number
  private readonly items: Transition[] = []
  /** Index in `items` of the oldest transition once the ring is full. */
  private head = 0

  constructor(capacity: number) {
    if (!(Number.isInteger(capacity) && capacity >= 1)) throw new RangeError(`ReplayBuffer: capacity must be an integer >= 1, got ${capacity}`)
    this.capacity = capacity
  }

  get size(): number {
    return this.items.length
  }

  push(t: Transition): void {
    if (this.items.length < this.capacity) {
      this.items.push(t)
      return
    }
    this.items[this.head] = t
    this.head = (this.head + 1) % this.capacity
  }

  /** Every transition, oldest first (the same objects; for checkpoints). Pushing them in order rebuilds the buffer. */
  toArray(): Transition[] {
    return Array.from({ length: this.items.length }, (_, i) => this.at(i))
  }

  /** The i-th transition, oldest first (0 = oldest, size-1 = newest). */
  at(i: number): Transition {
    if (!(Number.isInteger(i) && i >= 0 && i < this.items.length)) throw new RangeError(`ReplayBuffer: index ${i} out of range [0, ${this.items.length})`)
    return this.items[(this.head + i) % this.items.length]!
  }
}

/**
 * A uniform sample of every transition ever offered, in `capacity` slots (reservoir sampling, Algorithm R): the first
 * `capacity` offers are kept; after that offer number n replaces a uniformly chosen kept one with probability
 * capacity / n. So at any moment each transition of the life so far is held with the same probability, however old.
 * It draws from `rng` only once it is full: exactly one draw per offer.
 */
export class Reservoir {
  readonly capacity: number
  private readonly items: Transition[] = []
  /** Transitions offered so far (kept or not). */
  seen = 0

  /** `state`: a reservoir to continue (its toArray() and seen). */
  constructor(capacity: number, state?: { items: readonly Transition[]; seen: number }) {
    if (!(Number.isInteger(capacity) && capacity >= 1)) throw new RangeError(`Reservoir: capacity must be an integer >= 1, got ${capacity}`)
    this.capacity = capacity
    if (!state) return
    const n = state.items.length
    if (n > capacity) throw new RangeError(`Reservoir: ${n} transitions exceed its ${capacity} slots`)
    // While it has room it has kept every offer; once full it has seen at least as many as it holds.
    if (!(Number.isSafeInteger(state.seen) && (n < capacity ? state.seen === n : state.seen >= n))) {
      throw new RangeError(`Reservoir: seen ${state.seen} does not fit ${n} kept of ${capacity} slots`)
    }
    this.items.push(...state.items)
    this.seen = state.seen
  }

  get size(): number {
    return this.items.length
  }

  offer(t: Transition, rng: () => number): void {
    this.seen++
    if (this.items.length < this.capacity) {
      this.items.push(t)
      return
    }
    const j = Math.floor(rng() * this.seen)
    if (j < this.capacity) this.items[j] = t
  }

  /** The kept transitions in slot order (the same objects; for checkpoints). */
  toArray(): Transition[] {
    return [...this.items]
  }

  at(i: number): Transition {
    if (!(Number.isInteger(i) && i >= 0 && i < this.items.length)) throw new RangeError(`Reservoir: index ${i} out of range [0, ${this.items.length})`)
    return this.items[i]!
  }
}

/**
 * How many of a learner's `bufferSize` slots are its whole-life reservoir at `reservoirShare` (absent: none): the
 * nearest whole number of slots, but always leaving one for the most recent transition. The rest stay the FIFO of
 * recent transitions.
 */
export function reservoirSlots(bufferSize: number, reservoirShare: number | undefined): number {
  return Math.min(bufferSize - 1, Math.round((reservoirShare ?? 0) * bufferSize))
}

/**
 * One stored transition for replay, uniform over the recent ones followed by the reservoir's (a transition held in
 * both is twice as likely); null, without drawing, when nothing is stored. One draw. Without a reservoir this is
 * exactly the draw the learner always made, so its RNG stream is unchanged.
 */
export function sampleStored(recent: ReplayBuffer, kept: Reservoir | null, rng: () => number): Transition | null {
  const total = recent.size + (kept?.size ?? 0)
  if (total === 0) return null
  const i = Math.min(total - 1, Math.floor(rng() * total))
  return i < recent.size ? recent.at(i) : kept!.at(i - recent.size)
}
