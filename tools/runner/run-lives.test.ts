import { existsSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { SITE, STALL_MS, claimProfile, cuOf, isAlive, isRunner, parseArgs, stallWatch } from "./run-lives"

describe("run-lives", () => {
  it("parses its arguments, with the live site and a 16 h run by default", () => {
    expect(parseArgs([])).toEqual({ url: SITE, profile: ".local/runner-profile", hours: 16, setup: false, headed: false, whenIdle: false, idleAfterS: 600 })
    expect(parseArgs(["--hours", "2", "http://localhost:5173/", "--profile", "p"])).toMatchObject({ url: "http://localhost:5173/", profile: "p", hours: 2 })
    expect(parseArgs(["--setup"])).toMatchObject({ setup: true, headed: true })
    expect(() => parseArgs(["--hours", "0"])).toThrow(/--hours/)
    expect(() => parseArgs(["--hours", "x"])).toThrow(/--hours/)
    expect(parseArgs(["--when-idle", "--idle-after", "60"])).toMatchObject({ whenIdle: true, idleAfterS: 60 })
    expect(() => parseArgs(["--idle-after", "-1"])).toThrow(/--idle-after/)
  })
  it("reads the page's compute line", () => {
    expect(cuOf("1,234.5 CU")).toBe(1234.5)
    expect(cuOf("…")).toBeNull()
  })
  it("calls a tab stalled only after its learner and compute stay the same past the limit", () => {
    const s = stallWatch(1000)
    expect(s("a", 1, 0)).toBe(false)
    expect(s("a", 1, 1000)).toBe(false)
    expect(s("a", 1, 1001)).toBe(true)
    expect(s("a", 2, 1002)).toBe(false)
    expect(s("b", 2, 5000)).toBe(false)
  })
  it("claims a profile for one runner at a time, taking over a claim whose process is gone", () => {
    const profile = join(mkdtempSync(join(tmpdir(), "runner-")), "profile")
    const release = claimProfile(profile, 10, () => true)
    expect(() => claimProfile(profile, 11, pid => pid === 10)).toThrow(/another runner \(pid 10\)/)
    expect(claimProfile(profile, 11, () => false)).toBeTypeOf("function") // pid 10 gone: 11 takes it
    release() // 10's release leaves 11's claim
    expect(existsSync(`${profile}.pid`)).toBe(true)
    writeFileSync(`${profile}.pid`, "garbage")
    claimProfile(profile, 12, () => true)()
    expect(existsSync(`${profile}.pid`)).toBe(false)
  })
  it("takes a claim over when its pid is alive but is not a runner (a reboot reuses pids)", () => {
    expect(isRunner("/opt/node --require x tools/runner/run-lives.ts --when-idle --hours 336")).toBe(true)
    expect(isRunner("/usr/libexec/some-daemon")).toBe(false)
    expect(isAlive(process.pid)).toBe(false) // this test process is alive, but it is not a runner
    expect(isAlive(2 ** 22 + 12345)).toBe(false)
  })
  it("leaves a tab mid-checkup alone: the limit is well above a checkup's 7-8 min", () => {
    expect(STALL_MS).toBeGreaterThanOrEqual(30 * 60_000)
  })
})
