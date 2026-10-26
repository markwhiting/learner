// Minimal GitHub REST client. The only place a token is read into a request, and only as an Authorization header
// to https://api.github.com. Error messages are built from status codes, never from response bodies or the token.

const BASE = "https://api.github.com"

export class GitHubError extends Error {
  constructor(readonly status: number, message: string, readonly retryAfterMs: number | null, readonly rateLimited = false) {
    super(message)
  }
}
export class NetworkError extends Error {}

export interface ApiResult { status: number; body: unknown; headers: Headers }
export interface Api {
  /** Resolves for 2xx and for statuses listed in `allow`; throws GitHubError otherwise, NetworkError if fetch throws. */
  call(method: string, path: string, body?: unknown, allow?: number[]): Promise<ApiResult>
}

/** GitHub's abuse protection: no quota headers, a 403 whose message says so. Waits at least this long. */
export const SECONDARY_RATE_LIMIT_MIN_MS = 60_000
const SECONDARY_RE = /secondary rate limit/i

/**
 * Rate-limit verdict for a 403/429. Reset time is trusted only when the quota is exhausted.
 * `bodyText` is read only to recognise a secondary rate limit; it never goes into an error message.
 */
export function rateLimit(status: number, h: Headers, nowMs: number, bodyText = ""): { limited: boolean; delayMs: number | null } {
  const primary = primaryRateLimit(status, h, nowMs)
  if (status === 403 && SECONDARY_RE.test(bodyText)) return { limited: true, delayMs: Math.max(primary.delayMs ?? 0, SECONDARY_RATE_LIMIT_MIN_MS) }
  return primary
}

function primaryRateLimit(status: number, h: Headers, nowMs: number): { limited: boolean; delayMs: number | null } {
  const num = (k: string): number | null => {
    const v = h.get(k)
    return v !== null && v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : null
  }
  const retryAfter = num("retry-after")
  if (retryAfter !== null) return { limited: true, delayMs: Math.max(0, retryAfter) * 1000 }
  const exhausted = h.get("x-ratelimit-remaining") === "0"
  const reset = num("x-ratelimit-reset")
  if (exhausted) return { limited: true, delayMs: reset === null ? null : Math.max(0, reset * 1000 - nowMs) }
  return { limited: status === 429, delayMs: null }
}

export function createApi(token: string, fetchFn: typeof fetch, now: () => number = Date.now): Api {
  return {
    async call(method, path, body, allow = []) {
      let res: Response
      try {
        res = await fetchFn(BASE + path, {
          method,
          // GitHub marks GET answers cacheable for 60 s; a send must see its own last push and PR, never a cached answer
          // from before them (2026-10-26: cached "no open PR" closed PRs and lost files; a cached branch head gave 422s).
          cache: "no-store",
          headers: {
            Authorization: `Bearer ${token}`,
            "X-GitHub-Api-Version": "2022-11-28",
            Accept: "application/vnd.github+json",
            ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        })
      } catch {
        throw new NetworkError("network request failed")
      }
      let text = ""
      try { text = await res.text() } catch { /* unreadable body */ }
      let parsed: unknown = null
      try { parsed = JSON.parse(text) } catch { /* empty or non-JSON body */ }
      if (res.ok || allow.includes(res.status)) return { status: res.status, body: parsed, headers: res.headers }
      const rl = res.status === 403 || res.status === 429 ? rateLimit(res.status, res.headers, now(), text) : { limited: false, delayMs: null }
      throw new GitHubError(res.status, `GitHub answered ${res.status}`, rl.delayMs, rl.limited)
    },
  }
}

/** UTF-8 safe base64 that works in browsers and Node. */
export function toBase64(s: string): string {
  const bytes = new TextEncoder().encode(s)
  let bin = ""
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(bin)
}
