// OpenCode's plugin for the env's date. The bridge starts the CLI on real time, so a session's records sort in the
// order they were written at any date the env's clock shows; what the CLI starts, a command, reads the env's time, so
// the date the CLI tells the model is read the same way. A read that fails fails the turn rather than show real time.
import { execFileSync } from "node:child_process"

const ENV_DATE = /(<env>[\s\S]*?\n {2}Today's date: )[^\n]*/
const SEARCH_YEAR = /The current year is -?\d+\./

function envNow() {
  const { AGENTENV_REAL_TIME: _, ...env } = process.env
  const [seconds, nanoseconds] = execFileSync("date", ["+%s %N"], { encoding: "utf8", env }).trim().split(" ")
  const now = new Date(Number(seconds) * 1000 + Math.floor(Number(nanoseconds) / 1e6))
  if (Number.isNaN(now.getTime())) throw new Error(`the env's date: unreadable "${seconds} ${nanoseconds}"`)
  return now
}

export default {
  id: "agentenv-env-date",
  server: async () => ({
    "experimental.chat.system.transform": async (_input, output) => {
      const today = envNow().toDateString()
      output.system.forEach((part, i) => {
        output.system[i] = part.replace(ENV_DATE, (_match, head) => head + today)
      })
    },
    "tool.definition": async (input, output) => {
      if (input.toolID !== "websearch") return
      const year = envNow().getFullYear()
      output.description = output.description.replace(SEARCH_YEAR, () => `The current year is ${year}.`)
    },
  }),
}
