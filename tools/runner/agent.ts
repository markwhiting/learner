// Usage: tsx tools/runner/agent.ts install|uninstall|status   (run from the repo root; macOS)
// Installs run-lives --when-idle as a launchd agent of this user: it starts at login, is restarted if it exits, and
// appends to .local/runner.log. The learner itself runs only while the Mac is unattended (tools/runner/idle.ts). The
// plist is written with this checkout's paths at install time, so none are tracked.
import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { join, resolve } from "node:path"
import { pathToFileURL } from "node:url"

export const LABEL = "io.github.markwhiting.learner-runner"

const esc = (s: string) => s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")

/** The agent's plist for a checkout at `repo`, run with the node at `node`. */
export function plistOf(repo: string, node: string): string {
  const args = [node, join(repo, "node_modules/tsx/dist/cli.mjs"), "tools/runner/run-lives.ts", "--when-idle", "--hours", "336"]
  const log = join(repo, ".local/runner.log")
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
${args.map(a => `    <string>${esc(a)}</string>`).join("\n")}
  </array>
  <key>WorkingDirectory</key><string>${esc(repo)}</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>300</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>${esc(log)}</string>
  <key>StandardErrorPath</key><string>${esc(log)}</string>
</dict>
</plist>
`
}

function main(): void {
  const cmd = process.argv[2]
  const plist = join(homedir(), "Library/LaunchAgents", `${LABEL}.plist`)
  const domain = `gui/${process.getuid!()}`
  const launchctl = (...a: string[]) => { try { return execFileSync("launchctl", a, { encoding: "utf8" }) } catch (e) { return (e as { stdout?: string }).stdout ?? "" } }
  if (cmd === "install") {
    const repo = resolve(".")
    if (!existsSync(join(repo, "tools/runner/run-lives.ts"))) throw new Error("run from the repo root")
    mkdirSync(join(homedir(), "Library/LaunchAgents"), { recursive: true })
    launchctl("bootout", `${domain}/${LABEL}`)
    writeFileSync(plist, plistOf(repo, process.execPath))
    launchctl("bootstrap", domain, plist)
    console.log(`installed ${plist}; log: ${join(repo, ".local/runner.log")}`)
  } else if (cmd === "uninstall") {
    launchctl("bootout", `${domain}/${LABEL}`)
    rmSync(plist, { force: true })
    console.log(`removed ${LABEL}`)
  } else if (cmd === "status") {
    const out = launchctl("print", `${domain}/${LABEL}`)
    console.log(out ? out.split("\n").filter(l => /state =|pid =|last exit/.test(l)).map(l => l.trim()).join("\n") : "not installed")
  } else throw new Error("usage: tsx tools/runner/agent.ts install|uninstall|status")
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main()
