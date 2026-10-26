import type { Contribution, Queue } from "./queue"
import type { LogFile } from "./log"
import { GitHubError, NetworkError, createApi, toBase64, type Api } from "./github"
import { REPO, REJECTED_MESSAGE, getToken, whoAmI, type Identity } from "./token"

export interface ContributorStatus { state: "no-token" | "idle" | "sending" | "error"; message: string; pending: number }

export interface ContributorDeps {
  queue: Queue
  ls: Storage
  fetchFn: typeof fetch
  now: () => Date
  onStatus: (s: ContributorStatus) => void
  /** A learner just became local-only (MAX_CLOSED_IN_A_ROW of its PRs closed unmerged): its logs are no longer sent. */
  onLocalOnly?: (learnerId: string) => void
  /** Injectable for tests. Default: real timers. */
  sleep?: (ms: number) => Promise<void>
  schedule?: (fn: () => void, ms: number) => void
  /**
   * Runs a send while no other tab of this browser is sending (the page passes a Web Lock). Two tabs share one outbox:
   * without it, one can reset a branch the other has just opened a PR on. Default: no lock.
   */
  lock?: <T>(fn: () => Promise<T>) => Promise<T>
}

const UPSTREAM = `${REPO.owner}/${REPO.name}`
const BACKOFF_START_MS = 5_000
const BACKOFF_MAX_MS = 600_000
const FORK_POLLS = 10
const FORK_POLL_MS = 2_000
const DEFAULT_RATE_LIMIT_MS = 60_000
/** The gate's cap: a PR over this many files is not merged. */
export const MAX_PR_FILES = 16
export const PERMISSION_MESSAGE = "Your GitHub token can't write to the repo — check its permissions."
/** After this many of a learner's PRs in a row are closed unmerged, it stops contributing (decision 0014). */
export const MAX_CLOSED_IN_A_ROW = 3
/** What the page shows for a local-only learner. */
export const LOCAL_ONLY_MESSAGE = "This learner's logs were rejected; it keeps learning here but no longer contributes."

class Wait extends Error { constructor(readonly ms: number, readonly why: string) { super(why) } }

const runPath = (id: string) => `data/runs/${id}`
const branchOf = (id: string) => `run/${id}`
/** localStorage: the number of the learner's PR this browser last opened or found open (shared by its tabs). */
export const prKey = (id: string): string => `blanktab:pr:${id}`

export function createContributor(deps: ContributorDeps): { flush(): Promise<void>; report(): Promise<void> } {
  const { queue, ls, fetchFn, now, onStatus } = deps
  const sleep = deps.sleep ?? (ms => new Promise<void>(r => setTimeout(r, ms)))
  const schedule = deps.schedule ?? ((fn, ms) => void setTimeout(fn, ms))
  let inFlight: Promise<void> | null = null
  let retryArmed = false
  let backoff = BACKOFF_START_MS

  let lastState: ContributorStatus["state"] | null = null
  const status = (state: ContributorStatus["state"], message: string, pending: number) => { lastState = state; onStatus({ state, message, pending }) }
  /** Idle with `pending` waiting. A local-only learner's logs are never waiting, so "all shared" would not be true then. */
  const idle = async (pending: number) => {
    const local = pending === 0 ? (await queue.localOnly()).length : 0
    status("idle", pending > 0
      ? `${pending} session log${pending === 1 ? "" : "s"} waiting; they are sent every 10 minutes and when a life ends.`
      : local > 0 ? "No session logs waiting; a learner here no longer contributes (its logs were rejected)." : "All session logs are shared.", pending)
  }

  const armRetry = (ms: number) => {
    if (retryArmed) return
    retryArmed = true
    schedule(() => { retryArmed = false; void contributor.flush() }, ms)
  }

  async function destination(api: Api, me: Identity): Promise<string> {
    if (me.canPush) return UPSTREAM
    const mine = await api.call("GET", `/repos/${me.login}/${REPO.name}`, undefined, [404])
    const b = mine.body as { full_name?: string; fork?: boolean; parent?: { full_name?: string } } | null
    if (mine.status === 200 && b?.fork === true && b.parent?.full_name === UPSTREAM && typeof b.full_name === "string") return b.full_name
    const created = await api.call("POST", `/repos/${UPSTREAM}/forks`, {})
    const full = (created.body as { full_name?: unknown } | null)?.full_name
    if (typeof full !== "string") throw new GitHubError(502, "unexpected fork response", null)
    for (let i = 0; i < FORK_POLLS; i++) {
      const r = await api.call("GET", `/repos/${full}`, undefined, [404])
      if (r.status === 200) return full
      await sleep(FORK_POLL_MS)
    }
    throw new Wait(BACKOFF_START_MS, "Your fork is still being created")
  }

  const shaOf = (body: unknown, where: string): string => {
    const b = body as { sha?: unknown; object?: { sha?: unknown } } | null
    const sha = b?.object?.sha ?? b?.sha
    if (typeof sha !== "string") throw new GitHubError(502, `unexpected ${where} response`, null)
    return sha
  }

  async function upstreamMain(api: Api): Promise<string> {
    return shaOf((await api.call("GET", `/repos/${UPSTREAM}/git/ref/heads/main`)).body, "ref")
  }

  /** The learner's OPEN PR's number, or null. Only an open one counts: once the gate merges or closes one, later logs need a fresh PR. */
  async function openPr(api: Api, owner: string, learnerId: string): Promise<number | null> {
    const head = `${owner}:${branchOf(learnerId)}`
    const found = await api.call("GET", `/repos/${UPSTREAM}/pulls?head=${encodeURIComponent(head)}&state=open`)
    const first = Array.isArray(found.body) ? (found.body[0] as { number?: unknown } | undefined) : undefined
    if (first === undefined) return null
    if (!(Number.isInteger(first.number) && (first.number as number) > 0)) throw new GitHubError(502, "unexpected pulls response", null)
    return first.number as number
  }

  /**
   * The PR this browser last opened or found open for the learner, if it is still open, asked for by number. GitHub's
   * PR lists (by head, open or all) lag a just-opened PR by seconds; a send that trusted them alone reset the branch of
   * a PR it had opened moments before, GitHub closed that PR as empty, and its files were dropped as rejected
   * (2026-10-26, on almost every life). A PR asked for by number is never stale that way.
   */
  async function knownOpenPr(api: Api, learnerId: string): Promise<number | null> {
    const n = Number(ls.getItem(prKey(learnerId)))
    if (!(Number.isSafeInteger(n) && n > 0)) return null
    const r = await api.call("GET", `/repos/${UPSTREAM}/pulls/${n}`, undefined, [404])
    if (r.status === 200 && (r.body as { state?: unknown } | null)?.state === "open") return n
    ls.removeItem(prKey(learnerId))
    return null
  }

  /**
   * Takes stock of the learner's PRs the gate has decided since `c.lastPr` (oldest first, up to an open one): a merge
   * resets the count of closed-unmerged PRs in a row; a PR closed unmerged has its files dropped from the outbox (they
   * are never proposed again: the gate allows seq gaps, so the learner's next PR goes on without them). The
   * MAX_CLOSED_IN_A_ROW-th in a row makes the learner local-only. Saved in the outbox's database. `open`: a PR this
   * list shows still open (the open-PR list lagged): it is the learner's open PR, and its branch is built on, not reset.
   */
  async function settle(api: Api, owner: string, c: Contribution): Promise<{ contribution: Contribution; open: number | null }> {
    const head = `${owner}:${branchOf(c.learnerId)}`
    const found = await api.call("GET", `/repos/${UPSTREAM}/pulls?head=${encodeURIComponent(head)}&state=all&per_page=100`)
    if (!Array.isArray(found.body)) throw new GitHubError(502, "unexpected pulls response", null)
    const prs = (found.body as { number?: unknown; state?: unknown; merged_at?: unknown }[])
      .filter(p => Number.isInteger(p.number) && (p.number as number) > c.lastPr)
      .sort((a, b) => (a.number as number) - (b.number as number))
    let next = { ...c }
    let open: number | null = null
    for (const p of prs) {
      if (p.state === "open") { open = p.number as number; break } // not decided yet: still the learner's open PR
      if (p.state !== "closed") break
      const n = p.number as number
      if (typeof p.merged_at === "string" && p.merged_at !== "") next = { ...next, lastPr: n, closedInARow: 0 }
      else {
        await queue.dropPr(c.learnerId, n)
        next = { ...next, lastPr: n, closedInARow: next.closedInARow + 1 }
      }
      if (next.closedInARow >= MAX_CLOSED_IN_A_ROW) next = { ...next, localOnly: true }
      await queue.setContribution(next) // after each: a dropped PR is never counted twice
      if (next.localOnly) break
    }
    return { contribution: next, open }
  }

  /** Whether PR `n` is open with its head at `commit`: then the files just pushed are on it. */
  async function carries(api: Api, n: number, commit: string): Promise<boolean> {
    const r = await api.call("GET", `/repos/${UPSTREAM}/pulls/${n}`)
    const b = r.body as { state?: unknown; head?: { sha?: unknown } } | null
    return b?.state === "open" && b.head?.sha === commit
  }

  /**
   * How many more files an open PR may take before it passes the gate's cap (MAX_PR_FILES). The count comes from the
   * branch head itself (compare with main), so it cannot lag a commit just pushed. Fails closed: if the count cannot
   * be read, the answer is 0 and the files wait for the next flush.
   */
  async function room(api: Api, owner: string, learnerId: string): Promise<number> {
    try {
      const r = await api.call("GET", `/repos/${UPSTREAM}/compare/main...${owner}:${branchOf(learnerId)}`, undefined, [404])
      const files = (r.body as { files?: unknown } | null)?.files
      if (r.status !== 200 || !Array.isArray(files)) return 0
      return Math.max(0, MAX_PR_FILES - files.length)
    } catch (e) {
      if (e instanceof GitHubError && (e.status === 401 || e.status === 403 || e.rateLimited)) throw e
      return 0
    }
  }

  /**
   * The commit to build on. With an open PR: the branch head. Without one the branch starts again from
   * upstream main, so files an earlier (merged or closed) PR carried are not proposed a second time.
   */
  async function branchBase(api: Api, dest: string, learnerId: string, prOpen: boolean): Promise<string> {
    const refPath = `/repos/${dest}/git/ref/heads/${branchOf(learnerId)}`
    const have = await api.call("GET", refPath, undefined, [404])
    if (have.status === 200 && prOpen) return shaOf(have.body, "ref")
    const main = await upstreamMain(api)
    if (have.status === 200) {
      await api.call("PATCH", `/repos/${dest}/git/refs/heads/${branchOf(learnerId)}`, { sha: main, force: true })
      return main
    }
    try {
      await api.call("POST", `/repos/${dest}/git/refs`, { ref: `refs/heads/${branchOf(learnerId)}`, sha: main })
      return main
    } catch (e) {
      if (!(e instanceof GitHubError && e.status === 422)) throw e
      return shaOf((await api.call("GET", refPath)).body, "ref") // 422: another tab created it first
    }
  }

  /** All of a learner's pending files as one commit on its branch. Resolves to that commit, or null when they were all there already. */
  async function commitFiles(api: Api, dest: string, learnerId: string, parent: string, files: LogFile[]): Promise<string | null> {
    const baseTree = (await api.call("GET", `/repos/${dest}/git/commits/${parent}`)).body as { tree?: { sha?: unknown } } | null
    const baseTreeSha = baseTree?.tree?.sha
    if (typeof baseTreeSha !== "string") throw new GitHubError(502, "unexpected commit response", null)
    const entries = []
    for (const f of files) {
      const blob = await api.call("POST", `/repos/${dest}/git/blobs`, { content: toBase64(JSON.stringify(f)), encoding: "base64" })
      entries.push({ path: `${runPath(learnerId)}/${f.seq}.json`, mode: "100644", type: "blob", sha: shaOf(blob.body, "blob") })
    }
    const tree = shaOf((await api.call("POST", `/repos/${dest}/git/trees`, { base_tree: baseTreeSha, tree: entries })).body, "tree")
    if (tree === baseTreeSha) return null // a crash after an earlier commit: nothing new to write
    const seqs = files.map(f => f.seq)
    const message = `log: ${learnerId} #${seqs.length === 1 ? seqs[0] : `${seqs[0]}-${seqs[seqs.length - 1]}`}`
    const commit = shaOf((await api.call("POST", `/repos/${dest}/git/commits`, { message, tree, parents: [parent] })).body, "commit")
    // force:false: if another tab moved the branch meanwhile this fails and the whole send is retried.
    await api.call("PATCH", `/repos/${dest}/git/refs/heads/${branchOf(learnerId)}`, { sha: commit, force: false })
    return commit
  }

  /** Opens the learner's PR; resolves to its number. */
  async function createPr(api: Api, dest: string, owner: string, learnerId: string): Promise<number> {
    try {
      const r = await api.call("POST", `/repos/${UPSTREAM}/pulls`, {
        title: `run: ${learnerId}`,
        head: `${owner}:${branchOf(learnerId)}`,
        base: "main",
        draft: true,
        body: `Session logs from learner ${learnerId}, submitted by the learner page on behalf of @${owner}. Branch: ${dest}@${branchOf(learnerId)}.`,
      })
      const n = (r.body as { number?: unknown } | null)?.number
      if (!(Number.isInteger(n) && (n as number) > 0)) throw new GitHubError(502, "unexpected pull request response", null)
      return n as number
    } catch (e) {
      // Two tabs share one outbox: the other tab may have opened it first.
      if (e instanceof GitHubError && e.status === 422) {
        const n = await openPr(api, owner, learnerId)
        if (n !== null) return n
      }
      throw e
    }
  }

  async function sendAll(token: string): Promise<void> {
    const pending = await queue.pending()
    if (pending.length === 0) return
    const api = createApi(token, fetchFn, () => now().getTime())
    const me = await whoAmI(token, fetchFn)
    const byLearner = new Map<string, LogFile[]>()
    for (const f of pending) byLearner.set(f.learnerId, [...(byLearner.get(f.learnerId) ?? []), f])
    const dest = await destination(api, me)
    const owner = dest.split("/")[0]!
    for (const [id, files] of byLearner) {
      files.sort((a, b) => a.seq - b.seq)
      let open = await openPr(api, owner, id) ?? await knownOpenPr(api, id)
      if (open === null) {
        // No open PR: the last one was decided. Closed unmerged, its files are dropped; too many in a row, local-only.
        const s = await settle(api, owner, await queue.contribution(id))
        if (s.contribution.localOnly) {
          deps.onLocalOnly?.(id)
          continue
        }
        open = s.open
      }
      const prOpen = open !== null
      // Never let an open PR pass the gate's file cap: send the oldest files that fit, the rest wait for the merge.
      const fit = prOpen ? await room(api, owner, id) : MAX_PR_FILES
      files.splice(fit)
      if (files.length === 0) continue
      status("sending", `Sending ${files.length} session log${files.length === 1 ? "" : "s"} of learner ${id}…`, pending.length)
      const parent = await branchBase(api, dest, id, prOpen)
      const commit = await commitFiles(api, dest, id, parent, files)
      let pr: number | undefined
      if (commit === null) pr = open ?? undefined // all there already: on the open PR's branch, or (no open PR) merged to main
      else if (open === null) {
        pr = await createPr(api, dest, owner, id)
        ls.setItem(prKey(id), String(pr)) // at once: the next send may come before GitHub lists it
      }
      // The gate merges (or closes) every hour and may have done so between the check above and our push. Then the
      // commit sits on a branch with no open PR: keep the files queued; the next send starts a fresh branch from main.
      // Checked on the PR itself (by number), not the open-PR list, which can lag.
      else if (await carries(api, open, commit)) pr = open
      else throw new Wait(BACKOFF_START_MS, "The pull request was merged or closed while sending")
      // Only now, with the commit and an open PR both in place (or the files already on main), are they sent.
      for (const f of files) await queue.markSent(f.learnerId, f.seq, pr)
    }
    await queue.prune() // sent files are on GitHub now (or rejected): the browser keeps only each learner's newest
  }

  async function run(): Promise<void> {
    const count = async () => (await queue.pending()).length
    const token = getToken(ls)
    if (token === null) return status("no-token", "Add a GitHub token to share your session logs.", await count())
    const before = await count()
    try {
      if (before === 0) return await idle(0)
      status("sending", `Sending ${before} session log${before === 1 ? "" : "s"} to GitHub…`, before)
      await (deps.lock ?? (fn => fn()))(() => sendAll(token))
      backoff = BACKOFF_START_MS
      await idle(await count())
    } catch (e) {
      const pending = await count()
      if (e instanceof GitHubError && e.status === 401) return status("error", REJECTED_MESSAGE, pending)
      let ms: number
      let msg: string
      if (e instanceof GitHubError && e.status === 403 && !e.rateLimited) return status("error", PERMISSION_MESSAGE, pending)
      if (e instanceof GitHubError && e.rateLimited) {
        ms = Math.max(e.retryAfterMs ?? DEFAULT_RATE_LIMIT_MS, 1000)
        msg = `GitHub is limiting requests; retrying in ${Math.ceil(ms / 1000)} s.`
      } else if (e instanceof Wait) {
        ms = e.ms
        msg = `${e.why}; retrying in ${Math.ceil(ms / 1000)} s.`
      } else {
        ms = backoff
        backoff = Math.min(backoff * 2, BACKOFF_MAX_MS)
        msg = e instanceof NetworkError
          ? `Can't reach GitHub; retrying in ${Math.ceil(ms / 1000)} s.`
          : `GitHub had a problem (${e instanceof GitHubError ? e.status : "unknown"}); retrying in ${Math.ceil(ms / 1000)} s.`
      }
      status("error", msg, pending)
      armRetry(ms)
    }
  }

  // One run at a time. A flush asked for while one is running is not dropped: it runs once more afterwards,
  // so a file queued mid-flight is not left waiting for the next timer.
  let again: Promise<void> | null = null
  const contributor = {
    /** Show how many logs are waiting, without sending. Leaves a running send or an error the user must act on alone. */
    async report(): Promise<void> {
      if (inFlight !== null || lastState === "error") return
      const pending = (await queue.pending()).length
      if (getToken(ls) === null) return status("no-token", "Add a GitHub token to share your session logs.", pending)
      await idle(pending)
    },
    flush(): Promise<void> {
      if (inFlight === null) {
        inFlight = run().finally(() => { inFlight = null })
        return inFlight
      }
      // Whether the current run resolves or rejects, the follow-up still runs and `again` is cleared.
      const after = () => { again = null; return contributor.flush() }
      again ??= inFlight.then(after, after)
      return again
    },
  }
  return contributor
}
