// The gate's JSON (spec §7.4): sorted keys at every level, one-space indent, a final newline, no trailing spaces. Every
// file the gate writes, and every sha it takes of a block (the incumbent's model sha), goes through serialise.

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)

/** `v` with the keys of every object sorted, so serialisation never depends on insertion order. */
function sortKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeys)
  if (isObj(v)) return Object.fromEntries(Object.keys(v).sort().map(k => [k, sortKeys(v[k])]))
  return v
}

/** The gate's JSON: sorted keys, one-space indent, a final newline, no trailing spaces. */
export const serialise = (v: unknown): string => JSON.stringify(sortKeys(v), null, 1) + "\n"
