import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { execFileSync } from "node:child_process"
import { cpSync, mkdirSync, mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { createHash } from "node:crypto"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"
import { BUILD_INPUTS, buildManifest, codeShaOf, computeKnowledgeSha, resolveCodeSha } from "../manifest.ts"

let dir: string
beforeEach(() => { dir = mkdtempSync(join(tmpdir(), "manifest-")) })
// Retries: a `git commit` in the temp repo can leave a background maintenance process still writing under .git/objects
// when the directory is removed (seen once as ENOTEMPTY under load). The git helper below also turns that off.
afterEach(() => { rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 }) })
const sha = (s: string) => createHash("sha256").update(s).digest("hex")

describe("computeKnowledgeSha", () => {
  it("hashes name + newline + content + newline for each present prior, sorted by name", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    writeFileSync(join(dir, "consolidation-prior.json"), "C")
    expect(computeKnowledgeSha(dir)).toBe(sha("consolidation-prior.json\nC\nsettings-space.json\nS\n"))
  })
  it("matches the 64-lowercase-hex rule and ignores other files (summary.json, manifest.json)", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    const a = computeKnowledgeSha(dir)
    writeFileSync(join(dir, "summary.json"), "x"); writeFileSync(join(dir, "manifest.json"), "y")
    expect(computeKnowledgeSha(dir)).toBe(a)
    expect(a).toMatch(/^[0-9a-f]{64}$/)
  })
  it("changes when a prior's content changes or a prior appears", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    const a = computeKnowledgeSha(dir)
    writeFileSync(join(dir, "caregiver-prior.json"), "G")
    const b = computeKnowledgeSha(dir)
    expect(b).not.toBe(a)
    writeFileSync(join(dir, "settings-space.json"), "S2")
    expect(computeKnowledgeSha(dir)).not.toBe(b)
  })
})

describe("the committed knowledge directory", () => {
  it("the committed knowledge/caregiver-prior.json is a hashed prior", () => {
    const committed = fileURLToPath(new URL("../../../knowledge", import.meta.url))
    cpSync(committed, dir, { recursive: true })
    const withIt = computeKnowledgeSha(dir)
    rmSync(join(dir, "caregiver-prior.json"))
    expect(computeKnowledgeSha(dir)).not.toBe(withIt)
  })
  it("knowledgeSha changes when genome-space.json or genome.json changes", () => {
    const committed = fileURLToPath(new URL("../../../knowledge", import.meta.url))
    cpSync(committed, dir, { recursive: true })
    const seen = new Set([computeKnowledgeSha(dir)])
    for (const name of ["genome-space.json", "genome.json"]) {
      writeFileSync(join(dir, name), "{}\n")
      seen.add(computeKnowledgeSha(dir))
      rmSync(join(dir, name))
      seen.add(computeKnowledgeSha(dir))
    }
    expect(seen.size).toBe(5)
  })
})

// Each case runs git several times in a temporary repository: slow on a busy machine, so a generous timeout.
describe("codeShaOf: the last commit that changed a build input (BUILD_INPUTS)", { timeout: 60_000 }, () => {
  const git = (...args: string[]) => execFileSync("git", ["-c", "gc.auto=0", "-c", "maintenance.auto=false", ...args], { cwd: dir, encoding: "utf8", env: { ...process.env, GIT_AUTHOR_NAME: "t", GIT_AUTHOR_EMAIL: "t@t", GIT_COMMITTER_NAME: "t", GIT_COMMITTER_EMAIL: "t@t" } }).trim()
  const commit = (path: string, text: string) => {
    mkdirSync(join(dir, path, ".."), { recursive: true })
    writeFileSync(join(dir, path), text)
    git("add", "-A")
    git("commit", "-q", "-m", path)
    return git("rev-parse", "HEAD")
  }
  beforeEach(() => { git("init", "-q") })

  it("the build inputs: the app, the packages, the root package files, the shared compiler settings and the evaluation config", () => {
    expect([...BUILD_INPUTS]).toEqual(["app", "packages", "package.json", "package-lock.json", "tsconfig.base.json", "config/evaluation.json"])
  })
  it("a commit touching only data/runs, docs, knowledge, README, AGENTS, tests, tools, workflows or config/ besides evaluation.json leaves it unchanged; one touching app/ changes it", () => {
    const code = commit("app/src/main.ts", "1")
    expect(codeShaOf(dir)).toBe(code)
    for (const path of [
      "data/runs/lrn_0123456789abcdef/0.json", "docs/decisions/0099-x.md", "knowledge/settings-space.json", "README.md", "AGENTS.md",
      "tests/e2e/x.spec.ts", "tools/gate/gate.ts", ".github/workflows/gate.yml", "tsconfig.json", "eslint.config.js", "playwright.config.ts",
      "config/references/newborn-baseline@1-gpu-suite1-genome1.json", "config/other.json",
    ]) {
      commit(path, `x ${path}`)
      expect(codeShaOf(dir), path).toBe(code)
    }
    const next = commit("app/src/main.ts", "2")
    expect(next).not.toBe(code)
    expect(codeShaOf(dir)).toBe(next)
    expect(next).toMatch(/^[0-9a-f]{40}$/)
  })
  it("a commit to any other build input changes it", () => {
    commit("app/a.ts", "1")
    for (const path of ["packages/learner/src/version.ts", "package.json", "package-lock.json", "tsconfig.base.json", "config/evaluation.json"]) {
      const c = commit(path, `x ${path}`)
      expect(codeShaOf(dir), path).toBe(c)
    }
  })
  it("a nested file named like a root input (app/package.json is in app/ anyway; docs/package.json is not code)", () => {
    const code = commit("app/a.ts", "1")
    commit("docs/package.json", "{}")
    commit("tools/tsconfig.base.json", "{}")
    expect(codeShaOf(dir)).toBe(code)
  })
  it("is dev outside a git repository, with no commits, or with no commit touching a build input", () => {
    expect(codeShaOf(dir)).toBe("dev") // git init, nothing committed
    commit("README.md", "x")
    expect(codeShaOf(dir)).toBe("dev")
    const bare = mkdtempSync(join(tmpdir(), "nogit-"))
    try { expect(codeShaOf(bare)).toBe("dev") } finally { rmSync(bare, { recursive: true, force: true }) }
  })
})

describe("resolveCodeSha", () => {
  it("uses a 40-hex CODE_SHA (the deploy computes it with codeShaOf), otherwise codeShaOf the repo", () => {
    expect(resolveCodeSha("b".repeat(40), dir)).toBe("b".repeat(40))
    expect(resolveCodeSha(undefined, dir)).toBe("dev") // not a repo
    expect(resolveCodeSha("nonsense", dir)).toBe("dev")
  })
})

describe("buildManifest", () => {
  it("carries knowledgeSha, codeSha and the learnerStateVersion this code reads and writes", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    expect(buildManifest(dir, "b".repeat(40))).toEqual({ knowledgeSha: computeKnowledgeSha(dir), codeSha: "b".repeat(40), learnerStateVersion: LEARNER_STATE_VERSION })
    expect(Number.isInteger(LEARNER_STATE_VERSION) && LEARNER_STATE_VERSION >= 1).toBe(true)
  })
})
