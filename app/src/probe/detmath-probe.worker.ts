// The detmath probe's worker (see detmath-probe.ts): the golden fixtures' values computed in a module worker, then one
// ts birthDraw at 300 model rows timed (recorded by the spec, not compared).
import { compute, type Golden } from "../../../packages/detmath/test/golden-inputs"
import { computeGp, toBits, tsDrawInputAt } from "../../../packages/meta/test/gp-probe-inputs"
import { birthDraw } from "../../../packages/meta/src/index"

export type ProbeMessage =
  | { kind: "detmath"; golden: Golden }
  | { kind: "gp"; gp: unknown; tsDrawMs300: number }
  | { kind: "error"; error: string }

postMessage({ kind: "detmath", golden: compute() } satisfies ProbeMessage)
try {
  const gp = toBits(computeGp())
  const input = tsDrawInputAt(300)
  const t0 = performance.now()
  const d = birthDraw(input)
  const tsDrawMs300 = performance.now() - t0
  if (d.method !== "ts") throw new Error(`the timing draw is ${d.method}, not ts`)
  postMessage({ kind: "gp", gp, tsDrawMs300 } satisfies ProbeMessage)
} catch (err) {
  postMessage({ kind: "error", error: String(err) } satisfies ProbeMessage)
}
