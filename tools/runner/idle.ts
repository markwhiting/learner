// --when-idle: the runner's learner runs only while the Mac is unattended (no keyboard or mouse for a while, on AC
// power) and is frozen in place (SIGSTOP of the browser's processes) the moment someone is back. Compute units, not
// wall time, measure a life, so a frozen life resumes mid-step with its checkups unchanged. macOS only.
import { execFileSync } from "node:child_process"

export interface IdlePolicy {
  /** Seconds with no keyboard or mouse input before the learner runs. */
  idleAfterS: number
  /** Seconds since the last input after which it stops holding the Mac awake, so the Mac's own sleep applies. */
  awakeCapS: number
}
export const DEFAULT_IDLE: IdlePolicy = { idleAfterS: 600, awakeCapS: 8 * 3600 }

/** Seconds since the last keyboard or mouse input, from `ioreg -c IOHIDSystem` (HIDIdleTime is in ns), or null. */
export function idleSecondsOf(ioreg: string): number | null {
  const m = /"HIDIdleTime"\s*=\s*(\d+)/.exec(ioreg)
  return m ? Math.floor(Number(m[1]) / 1e9) : null
}

/** Whether `pmset -g batt` says the Mac draws from AC power. */
export const onAcOf = (pmset: string): boolean => /'AC Power'/.test(pmset)

export type Wanted = "run" | "pause"
/** Why the learner should run or pause now; an unreadable idle time pauses (fail closed: never run over someone). */
export function wanted(idleS: number | null, ac: boolean, p: IdlePolicy): { want: Wanted; why: string } {
  if (idleS === null) return { want: "pause", why: "idle time unreadable" }
  if (!ac) return { want: "pause", why: "on battery" }
  if (idleS < p.idleAfterS) return { want: "pause", why: `in use (input ${idleS} s ago)` }
  if (idleS >= p.awakeCapS) return { want: "pause", why: `unattended for over ${Math.round(p.awakeCapS / 3600)} h; letting the Mac sleep` }
  return { want: "run", why: `unattended (no input for ${Math.round(idleS / 60)} min)` }
}

/** Every descendant of `root` in `ps -A -o pid=,ppid=,comm=` output, except caffeinate (it must not be frozen). */
export function descendantsOf(ps: string, root: number): number[] {
  const kids = new Map<number, { pid: number; comm: string }[]>()
  for (const line of ps.split("\n")) {
    const m = /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line)
    if (!m) continue
    const ppid = Number(m[2])
    kids.set(ppid, [...(kids.get(ppid) ?? []), { pid: Number(m[1]), comm: m[3]!.trim() }])
  }
  const out: number[] = []
  const walk = (pid: number) => {
    for (const k of kids.get(pid) ?? []) {
      if (/(^|\/)caffeinate$/.test(k.comm)) continue
      out.push(k.pid)
      walk(k.pid)
    }
  }
  walk(root)
  return out
}

const sh = (cmd: string, args: string[]) => { try { return execFileSync(cmd, args, { encoding: "utf8" }) } catch { return "" } }
export const readIdleSeconds = (): number | null => idleSecondsOf(sh("ioreg", ["-c", "IOHIDSystem", "-d", "4"]))
export const readOnAc = (): boolean => onAcOf(sh("pmset", ["-g", "batt"]))

/** Sends `sig` to every process under this one (the browser and its helpers), except caffeinate. */
export function signalBrowser(sig: "SIGSTOP" | "SIGCONT"): number {
  const pids = descendantsOf(sh("ps", ["-A", "-o", "pid=,ppid=,comm="]), process.pid)
  for (const pid of pids) { try { process.kill(pid, sig) } catch { /* exited meanwhile */ } }
  return pids.length
}
