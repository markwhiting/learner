// The worker's answers to the page's forced checkpoints (CheckpointNowMessage). While a life runs, the session answers
// them through its inbox. A confirmation newborn (worker.ts runConfirmNewborn) is no session: it has no checkpoint to
// save, and its one file is queued only at its end. While it runs, every request is answered at once, not saved; one
// that asks to halt is also answered `halted`, so the page reloads now instead of waiting CHECKPOINT_WAIT_MS for an
// answer that never comes. After the reload the tab draws again; a newborn arm runs again from its start.
import type { SessionInbox } from "@blanktab/harness/session"
import type { CheckpointNowMessage, FromWorker } from "./protocol"

type Inbox = Pick<SessionInbox, "checkpointNow" | "haltWaiting" | "answer">

export interface CheckpointAnswers {
  /** A CheckpointNowMessage from the page. */
  request(m: CheckpointNowMessage): void
  /** A confirmation newborn starts: requests already waiting are answered (not saved) as later ones will be. */
  frozenStarts(): void
  /** It ended (or threw): requests go to the session again. */
  frozenEnds(): void
}

export function checkpointAnswers(inbox: Inbox, post: (m: FromWorker) => void): CheckpointAnswers {
  let frozen = false
  return {
    request(m) {
      if (!frozen) {
        void inbox.checkpointNow({ halt: m.halt }).then(saved => post({ type: "checkpointed", id: m.id, saved }))
        return
      }
      post({ type: "checkpointed", id: m.id, saved: false })
      if (m.halt) post({ type: "halted" })
    },
    frozenStarts() {
      frozen = true
      const halt = inbox.haltWaiting
      inbox.answer(false) // resolves each waiting request: its own `checkpointed` answer follows
      if (halt) post({ type: "halted" })
    },
    frozenEnds() {
      frozen = false
    },
  }
}
