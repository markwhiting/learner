// The deploy contract: the production build ignores every ?test* knob, ships no knowledge/, and stamps CODE_SHA.
import { describe, it, expect, beforeAll, afterAll } from "vitest"
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { build } from "vite"
import { TEST_KNOBS } from "../src/params"

const APP = fileURLToPath(new URL("..", import.meta.url))
const SHA = "0123456789abcdef0123456789abcdef01234567"
const files = (dir: string): string[] => readdirSync(dir).flatMap(n => {
  const p = join(dir, n)
  return statSync(p).isDirectory() ? files(p) : [p]
})

describe("production build", { timeout: 120_000 }, () => {
  let out: string
  let js: string
  beforeAll(async () => {
    out = mkdtempSync(join(tmpdir(), "blanktab-build-"))
    const prev = process.env.CODE_SHA
    process.env.CODE_SHA = SHA
    try {
      await build({ root: APP, configFile: join(APP, "vite.config.ts"), mode: "production", logLevel: "silent", build: { outDir: out, emptyOutDir: true } })
    } finally {
      if (prev === undefined) delete process.env.CODE_SHA
      else process.env.CODE_SHA = prev
    }
    js = files(out).filter(f => f.endsWith(".js")).map(f => readFileSync(f, "utf8")).join("\n")
  })
  afterAll(() => rmSync(out, { recursive: true, force: true }))

  it("drops every test-only knob", () => {
    expect(js.length).toBeGreaterThan(0)
    expect(TEST_KNOBS.length).toBeGreaterThan(0)
    for (const knob of TEST_KNOBS) expect(js).not.toContain(knob)
    expect(js).not.toMatch(/["'`]test[A-Z][A-Za-z]*["'`]/) // nor any other ?test… knob name
    expect(js).not.toContain("testLifetime")
  })
  it("ships no knowledge/ of its own and builds the worker", () => {
    expect(files(out).some(f => f.includes(`${join(out, "knowledge")}`))).toBe(false)
    expect(files(out).filter(f => /worker.*\.js$/.test(f))).toHaveLength(1)
  })
  it("ships no detmath probe page (the e2e build's only)", () => {
    expect(files(out).filter(f => f.endsWith(".html")).map(f => f.slice(out.length + 1))).toEqual(["index.html"])
    expect(js).not.toContain("__detmath")
    expect(js).not.toContain("__gpProbe") // nor the GP determinism probe (plan 05b, Task 6)
  })
  it("the worker carries the real learner on WebGPU (its WGSL kernels), not the flailer", () => {
    const worker = files(out).filter(f => /worker.*\.js$/.test(f)).map(f => readFileSync(f, "utf8")).join("\n")
    expect(worker).toContain("@compute")
  })
  it("stamps CODE_SHA", () => {
    expect(js).toContain(SHA)
  })
})
