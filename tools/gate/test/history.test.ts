import { afterEach, describe, it, expect } from "vitest"
import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { genomeSha, parseCentre, parseGenomeSpace } from "@blanktab/meta"
import { centresBySha, fileHistory, genomeShaSync, spacesBySchema } from "../history.ts"
import { mainSpaces } from "../gate.ts"
import { FIXTURE_ENV, SPACE_JSON, makeFixture, type Fixture } from "./git-fixture.ts"

const GENOME_JSON = readFileSync(join(__dirname, "../../../knowledge/genome.json"), "utf8")
const SPACE = parseGenomeSpace(JSON.parse(SPACE_JSON))
const CENTRE = parseCentre(JSON.parse(GENOME_JSON), SPACE)

let fx: Fixture | undefined
afterEach(() => { fx?.cleanup(); fx = undefined })
/** git in the fixture's checkout of main, as the gate's string-returning git. */
const gitIn = (f: Fixture) => (args: string[]) => execFileSync("git", args, { cwd: f.work, encoding: "utf8", env: { ...process.env, ...FIXTURE_ENV }, stdio: ["ignore", "pipe", "pipe"] })

/** The space with schemaVersion `v` and gene `name`'s max moved to `max` (another text of the file). */
const spaceText = (v: number, max?: number) => {
  const o = JSON.parse(SPACE_JSON) as { schemaVersion: number; genes: { name: string; max?: number }[] }
  o.schemaVersion = v
  if (max !== undefined) o.genes.find(g => g.name === "entropyWeight")!.max = max
  return JSON.stringify(o, null, 1) + "\n"
}

describe("history.ts: earlier versions of the knowledge files (plan 05b Task 7)", { timeout: 60_000 }, () => {
  it("fileHistory on a fixture repo with 3 commits of a file (one repeating an earlier text) returns 2 distinct texts, newest first; spacesBySchema maps each schemaVersion to its newest text", () => {
    fx = makeFixture()
    const git = gitIn(fx)
    fx.commitMain({ "notes.txt": "a\n" })
    fx.commitMain({ "notes.txt": "b\n" })
    fx.commitMain({ "notes.txt": "a\n" })
    fx.commitMain({ "README.md": "y\n" }) // not a commit of notes.txt
    expect(fileHistory(git, "notes.txt")).toEqual(["a\n", "b\n"])
    expect(fileHistory(git, "never.txt")).toEqual([])
    // A deleted file's earlier texts are still there; the deletion adds none.
    fx.commitMain({}, ["notes.txt"])
    expect(fileHistory(git, "notes.txt")).toEqual(["a\n", "b\n"])

    // The fixture's main starts with today's space (schema 1). Schema 1 is then edited, schema 2 replaces it, and an
    // invalid text is committed last: each schema maps to its newest text, and the invalid one is skipped.
    const s1b = spaceText(1, 0.05), s2 = spaceText(2), s2b = spaceText(2, 0.045)
    fx.commitMain({ "knowledge/genome-space.json": s1b })
    fx.commitMain({ "knowledge/genome-space.json": s2 })
    fx.commitMain({ "knowledge/genome-space.json": s2b })
    fx.commitMain({ "knowledge/genome-space.json": "{ not json\n" })
    const spaces = spacesBySchema(git)
    expect([...spaces.keys()].sort()).toEqual([1, 2])
    expect(spaces.get(1)).toEqual(parseGenomeSpace(JSON.parse(s1b)))
    expect(spaces.get(2)).toEqual(parseGenomeSpace(JSON.parse(s2b)))
    expect(fileHistory(git, "knowledge/genome-space.json")).toHaveLength(5)
    // mainSpaces stops when the checkout's current space is invalid, as loadGenomeSpace does.
    expect(() => mainSpaces(fx!.work, git)).toThrow(/knowledge\/genome-space.json on main is invalid/)
    fx.commitMain({ "knowledge/genome-space.json": s2b })
    expect([...mainSpaces(fx.work, git).keys()].sort()).toEqual([1, 2])
  })

  it("centresBySha: every genome.json main has had, by genomeSha (equal to @blanktab/meta genomeSha), each parsed against its own schema's space", async () => {
    fx = makeFixture()
    const git = gitIn(fx)
    expect(centresBySha(git, spacesBySchema(git)).size).toBe(0) // no genome.json yet
    const moved = { ...CENTRE, lr: 1e-3 }
    const file = (values: object, genomeSchema = 1) => JSON.stringify({ schemaVersion: 1, genomeSchema, values, provenance: { source: "initial" } }, null, 1) + "\n"
    fx.commitMain({ "knowledge/genome.json": GENOME_JSON })
    fx.commitMain({ "knowledge/genome.json": file(moved) })
    fx.commitMain({ "knowledge/genome.json": file(CENTRE, 7) }) // a schema main never had: skipped
    fx.commitMain({ "knowledge/genome.json": file({ ...CENTRE, lr: 99 }) }) // out of bounds: skipped
    const centres = centresBySha(git, spacesBySchema(git))
    expect(centres.size).toBe(2)
    expect(centres.get(await genomeSha(1, CENTRE))).toEqual(CENTRE)
    expect(centres.get(await genomeSha(1, moved))).toEqual(parseCentre(JSON.parse(file(moved)), SPACE))
    for (const g of [CENTRE, moved, { b: "x", a: 2 }]) expect(genomeShaSync(3, g)).toBe(await genomeSha(3, g))
  })
})
