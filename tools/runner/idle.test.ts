import { describe, expect, it } from "vitest"
import { DEFAULT_IDLE, descendantsOf, idleSecondsOf, onAcOf, wanted } from "./idle"

describe("idle", () => {
  it("reads the idle time and the power source as macOS prints them", () => {
    expect(idleSecondsOf(`    | |   "HIDIdleTime" = 133052447916\n`)).toBe(133)
    expect(idleSecondsOf("nothing")).toBeNull()
    expect(onAcOf("Now drawing from 'AC Power'\n -InternalBattery-0")).toBe(true)
    expect(onAcOf("Now drawing from 'Battery Power'")).toBe(false)
  })
  it("runs only unattended on AC power, pauses on any input, on battery, past the awake cap, or when unsure", () => {
    const at = (s: number | null, ac = true) => wanted(s, ac, DEFAULT_IDLE).want
    expect(at(599)).toBe("pause")
    expect(at(600)).toBe("run")
    expect(at(8 * 3600 - 1)).toBe("run")
    expect(at(8 * 3600)).toBe("pause")
    expect(at(3600, false)).toBe("pause")
    expect(at(null)).toBe("pause")
  })
  it("finds every process under the runner, but not caffeinate", () => {
    const ps = ["  10     1 node", "  11    10 /usr/bin/caffeinate", "  12    10 Chromium", "  13    12 Chromium Helper (GPU)", "  14    12 Chromium Helper (Renderer)", "  15     1 other"].join("\n")
    expect(descendantsOf(ps, 10).sort()).toEqual([12, 13, 14])
  })
})

describe("agent", () => {
  it("writes a plist that runs run-lives --when-idle from the checkout, logging to .local/runner.log", async () => {
    const { plistOf, LABEL } = await import("./agent")
    const p = plistOf("/repo/a & b", "/opt/node")
    expect(p).toContain(`<string>${LABEL}</string>`)
    expect(p).toContain("<string>/opt/node</string>\n    <string>/repo/a &amp; b/node_modules/tsx/dist/cli.mjs</string>\n    <string>tools/runner/run-lives.ts</string>\n    <string>--when-idle</string>")
    expect(p).toContain("<key>WorkingDirectory</key><string>/repo/a &amp; b</string>")
    expect(p).toContain("<string>/repo/a &amp; b/.local/runner.log</string>")
  })
})
