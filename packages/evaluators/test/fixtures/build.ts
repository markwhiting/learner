// Synthetic lives for the evaluator tests: LogFileV2 objects that parseLogFile accepts (the tests check it).
import type { CheckupResult, CurvePoint, LogFileV2, MetaBlock } from "@blanktab/harness/log"

export const GPU = { conv1: 16, conv2: 32, enc: 128, hidden: 256 }
export const CPU = { conv1: 8, conv2: 16, enc: 64, hidden: 128 }
export const IDS = ["M1", "M2", "M3", "M4", "M5", "M6"] as const
export type Mid = typeof IDS[number]

/** A valid learnerId for an integer k. */
export const lid = (k: number): string => "lrn_" + k.toString(16).padStart(16, "0")

export function meta(over: Partial<MetaBlock> = {}): MetaBlock {
  return {
    v: 1, role: "search", method: "calibrate", genomeSchema: 1, genome: { lr: 0.001, gamma: 0.9 }, genomeSha: "ab".repeat(32),
    drawSeed: 7, posteriorSha: "cd".repeat(32), evalPanel: 3, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
    ...over,
  }
}

/** A checkup at `level`: absent measures are null; `passed` defaults to none. */
export function checkup(level: number, measures: Partial<Record<Mid, number | null>>, passed: Mid[] = []): CheckupResult {
  return {
    level, steps: level * 10, passed: [...passed],
    measures: Object.fromEntries(IDS.map(id => [id, measures[id] ?? null])), evalCU: 1,
  }
}

/** A curve point at `cu`; with `ratio`, a held-out score of loss `ratio` against baseline 1. */
export const point = (cu: number, ratio?: number): CurvePoint => ({
  cu, steps: Math.round(cu * 10), predLoss: 1, baselineLoss: 1, ...(ratio !== undefined ? { heldOut: { loss: ratio, baseline: 1 } } : {}),
})

export interface FileSpec {
  learnerId: string; seq: number; cu: number; final?: boolean; curve?: CurvePoint[]; checkups?: CheckupResult[]
  meta?: MetaBlock | null; widths?: LogFileV2["widths"]; suite?: number
}

export function file(s: FileSpec): LogFileV2 {
  const checkups = s.checkups ?? []
  const m = s.meta === undefined ? meta() : s.meta
  return {
    schemaVersion: 2, learnerId: s.learnerId, seq: s.seq, codeSha: "dev", knowledgeSha: "ef".repeat(32),
    writtenAt: "2026-10-23T00:00:00.000Z", stepsTotal: Math.round(s.cu * 10), settings: { lr: 0.001 },
    hostSig: { cores: 8, deviceMemoryGb: 8, webgpu: true, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: s.final ?? false,
    cu: s.cu, condition: "A", widths: s.widths ?? GPU, curve: s.curve ?? [], skippedUpdates: 0, restores: 0, lostCU: 0,
    evalCU: 10 * (s.seq + 1),
    ...(checkups.length ? { checkups, milestoneSuiteVersion: s.suite ?? 1 } : {}),
    ...(m ? { meta: m } : {}),
  }
}

/**
 * The cu of the file that crosses `level`, as a real session writes it: at whatever step crosses the level, so a
 * fraction over it (never exactly on it). The checkup at that level runs at the next sleep, so it lands in a later file.
 */
export const crossed = (level: number): number => level + 0.03

/**
 * A typical eligible search life, laid out like a real log: crossing files at 512, 1024, 2048 and 3072 CU (each a fraction
 * over), each checkup in the file after its crossing's (the 2048 checkup in the 3072 file), held-out points in (1024, 2048].
 */
export function typicalLife(learnerId: string, over: { meta?: MetaBlock; widths?: LogFileV2["widths"] } = {}): LogFileV2[] {
  const ms = { M1: 0.9, M2: 1, M3: 0.2, M4: 0.3, M5: 0.8, M6: 0 }
  const f = (s: Omit<FileSpec, "learnerId">) => file({ learnerId, ...(over.meta ? { meta: over.meta } : {}), ...(over.widths ? { widths: over.widths } : {}), ...s })
  return [
    f({ seq: 0, cu: crossed(512), curve: [point(256, 1), point(512, 0.99)] }),
    f({ seq: 1, cu: crossed(1024), curve: [point(1024, 0.97)], checkups: [checkup(512, ms)] }),
    f({ seq: 2, cu: crossed(2048), curve: [point(1200, 0.95), point(1600, 0.94), point(2048, 0.93), point(crossed(2048), 0.5)], checkups: [checkup(1024, ms)] }),
    f({ seq: 3, cu: crossed(3072), curve: [point(3000, 0.9)], checkups: [checkup(2048, ms)] }),
  ]
}

/** A confirmation life's meta block (spec §4.5): role confirm, its confirmation and arm, a seed from its block, no panel. */
/**
 * A confirmation 0 life of `arm`, its `seed`-th: evalSeed 10,000 + j with j of the arm's (j mod 4: candidate 0, centre
 * 1, newborn 2; the parser recomputes the arm from the seed). Seeds wrap within the block of 1,000.
 */
export const confirmMeta = (arm: "candidate" | "centre" | "newborn", seed: number, over: Partial<MetaBlock> = {}): MetaBlock =>
  meta({ role: "confirm", method: "confirm", evalPanel: null, evalSeed: 10_000 + 4 * (seed % 250) + { candidate: 0, centre: 1, newborn: 2 }[arm], confirmation: { id: 0, arm }, ...over })

export interface ScoredLifeSpec {
  /** Measures at every level. */
  ms: Partial<Record<Mid, number | null>>
  /** Held-out ratio at the three points in (1024, 2048]. */
  ratio: number
  /** Milestones passed, per level 512, 1024, 2048. */
  passed?: [Mid[], Mid[], Mid[]]
  meta?: MetaBlock
}

/**
 * An eligible life with the same measures at 512, 1024 and 2048 and a held-out ratio, laid out like a real log: fractional
 * crossing files, each checkup in the file after its crossing's (the 2048 one in a reload's file at 2,100 CU).
 */
export function scoredLife(learnerId: string, s: ScoredLifeSpec): LogFileV2[] {
  const p = s.passed ?? [[], [], []]
  const f = (x: Omit<FileSpec, "learnerId">) => file({ learnerId, ...(s.meta ? { meta: s.meta } : {}), ...x })
  return [
    f({ seq: 0, cu: crossed(512), curve: [point(512, 1)] }),
    f({ seq: 1, cu: crossed(1024), curve: [point(1024, 1)], checkups: [checkup(512, s.ms, p[0])] }),
    f({ seq: 2, cu: crossed(2048), curve: [point(1200, s.ratio), point(1600, s.ratio), point(2048, s.ratio)], checkups: [checkup(1024, s.ms, p[1])] }),
    f({ seq: 3, cu: 2100, curve: [point(2100, s.ratio)], checkups: [checkup(2048, s.ms, p[2])] }),
  ]
}

/** A life that ended before its budget checkup: `end` "failed" or "orphan" (abandoned), at cu 1300. */
export function endedLife(learnerId: string, end: "failed" | "orphan", m: MetaBlock): LogFileV2[] {
  return [
    file({ learnerId, seq: 0, cu: crossed(512), meta: m, curve: [point(512, 1)] }),
    file({ learnerId, seq: 1, cu: 1300, final: true, meta: { ...m, end }, curve: [point(1300, 1)], checkups: [checkup(512, {})] }),
  ]
}
