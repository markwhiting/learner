// The e2e build's detmath probe page: the golden fixtures' values computed by the browser, on the main thread and in a
// module worker (where the page's birth draws run), for tests/e2e/detmath.spec.ts to compare bit for bit with
// packages/detmath/test/fixtures/golden.json and packages/meta/test/fixtures/gp-probe-golden.json (written by Node).
import { compute, type Golden } from "../../../packages/detmath/test/golden-inputs"
import { computeGp, toBits } from "../../../packages/meta/test/gp-probe-inputs"
import type { ProbeMessage } from "./detmath-probe.worker"

declare global {
  interface Window {
    __detmath?: { main: Golden; worker: Golden } | { error: string }
    __gpProbe?: { main: unknown; worker: unknown; tsDrawMs300: number } | { error: string }
  }
}

const out = document.getElementById("out")!
const worker = new Worker(new URL("./detmath-probe.worker.ts", import.meta.url), { type: "module" })
worker.onmessage = (e: MessageEvent<ProbeMessage>) => {
  const m = e.data
  if (m.kind === "error") { window.__gpProbe = { error: m.error }; return }
  if (m.kind === "detmath") {
    window.__detmath = { main: compute(), worker: m.golden }
    out.textContent = "detmath done; GP computing"
    return
  }
  // The GP probe blocks the main thread for a few seconds: run it after this message's task.
  setTimeout(() => {
    try {
      window.__gpProbe = { main: toBits(computeGp()), worker: m.gp, tsDrawMs300: m.tsDrawMs300 }
      out.textContent = "done"
    } catch (err) {
      window.__gpProbe = { error: String(err) }
    }
  }, 0)
}
worker.onerror = e => { window.__detmath = { error: e.message }; window.__gpProbe = { error: e.message } }
