import { describe, expect, it } from "vitest"
import { SITE, cuOf, parseArgs, stallWatch } from "./run-lives"

describe("run-lives", () => {
  it("parses its arguments, with the live site and a 16 h run by default", () => {
    expect(parseArgs([])).toEqual({ url: SITE, profile: ".local/runner-profile", hours: 16, setup: false, headed: false })
    expect(parseArgs(["--hours", "2", "http://localhost:5173/", "--profile", "p"])).toMatchObject({ url: "http://localhost:5173/", profile: "p", hours: 2 })
    expect(parseArgs(["--setup"])).toMatchObject({ setup: true, headed: true })
    expect(() => parseArgs(["--hours", "0"])).toThrow(/--hours/)
    expect(() => parseArgs(["--hours", "x"])).toThrow(/--hours/)
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
})
