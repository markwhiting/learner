import { describe, it, expect } from "vitest"
import { SessionInbox } from "@blanktab/harness/session"
import { checkpointAnswers } from "../src/checkpoints"
import type { FromWorker } from "../src/protocol"

const flush = () => new Promise(r => setTimeout(r, 0))

describe("forced checkpoints while a confirmation newborn runs (final review)", () => {
  it("while it runs: answered saved false at once, and a halt request is answered halted; before and after, the session's inbox answers", async () => {
    const inbox = new SessionInbox()
    const posted: FromWorker[] = []
    const c = checkpointAnswers(inbox, m => posted.push(m))
    // A life runs: the request waits for the session (nothing posted until it answers).
    c.request({ type: "checkpointNow", id: 0, halt: false })
    await flush()
    expect(posted).toEqual([])
    // A halt request also waits, then the newborn starts: both waiting requests are answered, not saved, and the page is told to reload.
    c.request({ type: "checkpointNow", id: 1, halt: true })
    c.frozenStarts()
    await flush()
    expect(posted).toEqual([{ type: "halted" }, { type: "checkpointed", id: 0, saved: false }, { type: "checkpointed", id: 1, saved: false }])
    posted.length = 0
    // While it runs: at once (no 60 s wait), synchronously.
    c.request({ type: "checkpointNow", id: 2, halt: false })
    expect(posted).toEqual([{ type: "checkpointed", id: 2, saved: false }])
    c.request({ type: "checkpointNow", id: 3, halt: true })
    expect(posted.slice(1)).toEqual([{ type: "checkpointed", id: 3, saved: false }, { type: "halted" }])
    posted.length = 0
    // It ended: requests go to the session again.
    c.frozenEnds()
    c.request({ type: "checkpointNow", id: 4, halt: false })
    await flush()
    expect(posted).toEqual([])
    inbox.answer(true)
    await flush()
    expect(posted).toEqual([{ type: "checkpointed", id: 4, saved: true }])
  })

  it("a newborn that starts with no request waiting posts nothing", async () => {
    const posted: FromWorker[] = []
    const c = checkpointAnswers(new SessionInbox(), m => posted.push(m))
    c.frozenStarts()
    await flush()
    expect(posted).toEqual([])
  })
})
