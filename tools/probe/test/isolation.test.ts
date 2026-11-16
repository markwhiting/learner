// AGENTS.md invariant 2: the filter probe is harness-only. Nothing a learner, the page or the gate runs may reach it.
import { describe, it, expect } from "vitest"
import { readTracked, trackedFiles } from "../../tracked.ts"

const CODE = /\.(ts|tsx|js|mjs|cjs)$/
/** A static import/export-from, a dynamic import() or a require() whose specifier names tools/probe. */
const PROBE_IMPORT = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*)["'`][^"'`]*tools\/probe[^"'`]*["'`]/

describe("the filter probe is isolated", () => {
  it("no file under packages/ or app/ imports tools/probe (static or dynamic)", () => {
    const files = trackedFiles().filter(f => (f.startsWith("packages/") || f.startsWith("app/")) && CODE.test(f))
    expect(files.length).toBeGreaterThan(50)
    const offenders = files.filter(f => PROBE_IMPORT.test(readTracked(f) ?? ""))
    expect(offenders).toEqual([])
  })

  it("the pattern catches each import form", () => {
    for (const s of [`import { x } from "../../tools/probe/filters/lives.ts"`, `await import("../tools/probe/x")`, `require('../../tools/probe/a')`, `export * from "../tools/probe/b"`, `import "../tools/probe/c"`])
      expect(PROBE_IMPORT.test(s), s).toBe(true)
    expect(PROBE_IMPORT.test(`import { x } from "../tools/eval/lives.ts"`)).toBe(false)
  })
})
