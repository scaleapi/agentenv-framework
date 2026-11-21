# Letta Code A2A agent

The [Letta Code](https://github.com/letta-ai/letta-code) CLI wrapped with `agentenv_protocol.a2a_agent`, shipped
in the `agents` bundle as `@local/agentenv-framework/agents/letta-code`, with alias `letta-code`. This directory
is the whole build context. The Dockerfile pins the Python base by digest and Node by version and checksum;
`package-lock.json` pins `@letta-ai/letta-code` to 0.34.26.

Each turn runs `letta --backend local -p --output-format stream-json --yolo`. The bridge creates an agent with
`letta agents create`; the first turn uses `--agent <id> --new`, and subsequent turns use
`--conversation <id>`. Each context has a separate home under `$HOME/sessions/`. Different contexts
run concurrently; turns in one context and snapshot operations are synchronized.

The bridge connects the local backend with `letta connect openai-compatible`, to a per-turn loopback model relay
(`model_relay.py`, shared with other agents) that forwards requests to `$LITELLM_BASE_URL/v1`, with `LITELLM_API_KEY`,
`[model.headers]` (`LITELLM_EXTRA_HEADERS`), and `project_id` / `task_id` as `x-litellm-customer-id` and
`x-litellm-tags`. The model defaults to `openai/gpt-4.1-mini`. The CLI is connected with a placeholder key; the relay
answers its model list with the turn's model alone: the CLI's merge of a listed model scans every model listed, which
costs about 45 s of CPU per command over an endpoint that lists thousands. The relay counts each model call from its
answer's usage, so a task's `Usage` covers every call the CLI made, its subagents', summaries' and retries' too; a call
answered without usage is counted in `unreported_calls`, and a refused one counts nothing. When a task ends, the Letta
Code processes it left running (a background Workflow's server, a subagent stuck in a tool), found by a marker its
commands carry, are stopped, since their later calls would go to the next task's relay; the model's own background
jobs keep running. The CLI's environment, and so the model's shell, still has `LITELLM_*`: a call a tool makes to the
endpoint itself isn't counted. Registered MCP servers retain their names and headers;
a task role replaces `AgentEnv-Role`. Before each turn, `letta mcp tools --full` discovers the exact schemas. A context-local mod registers them as
direct Letta Code tools through `letta.tools.register`; each tool executes `letta mcp call` with its full
`mcp__<server>__<tool>` name. Peer tools use the same path.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | SDK set/get; `model`, `effort`, `system_prompt`, `output_format`, `project_id`, `task_id`, `timeout_seconds`. Before every turn, task instructions are committed to the existing agent's core memory, which the local backend compiles into its system prompt. JSON schema answers are validated locally. |
| `mcp-config/v1` | SDK registration and listing; servers passed to the context's agent settings each turn, with their discovered schemas exposed as direct tools through the public mods API. |
| `skill-config/v1` | Inline and bundle skills installed under `$HOME/.letta/skills/`, copied into each context's home before its turn. |
| `trajectory/v1` | Native stream-json events appended live, format `letta-code-events/1`. |
| `snapshot/v1` | Save archives the complete context home, including conversations, agent records, settings and git-backed memory; load stages it and restores it into the target context, rewriting stored home paths and binding the saved conversation. Optional workspace archives restore `/app`. The validator reports load/recall as false without its optional Claude Code rubric judge, as it does for the OpenCode baseline; a live cross-context recall probe verifies restore independently. GNU tar changelog captures task-level file increments, including removed roots, using the OpenCode template's implementation. |
| `peer-agents/v1` | SDK default; its `peer_agents` MCP server is registered alongside task servers. |
| `attribution-probe/v1` | Reports the project and task attribution used by the last CLI turn. |
| `install/v1` | Template installer brings checksum-verified Node and uv, the pinned CLI, and a private Python into a container or a Linux host with glibc. Installs git if missing. |
| `staging/v1` | SDK default. |

The image starts as root (`start-agent`), whose supervisord runs the A2A server as user `agent` in `/app`, with state
under `/data`; started as another user, it runs the server alone, as before. Either way the container ends when the
server exits, as before: supervisord never restarts it, and stops once it exits. It includes git and `pdftoppm`. PDF
attachments are rendered as page images; other attachments are materialized and their paths passed to the CLI. Headless
mode never loads node-pty, so installation skips dependency scripts and rebuilds only the CLI, sharp and protobufjs.
The end of CLI stderr is included in failures. Cancellation and shutdown kill the CLI's process tree.

The image follows a task env's clock as the agents README describes, only once a `sync_env_clock` sets `agent_clock`.
Letta Code's local backend runs inside each `letta` process, so the whole CLI follows the clock, the "Local time" line
of the environment context it gives the model included, as do the commands the model runs; the A2A server, its model
relay and its records stay on real time. The CLI reaches the model through that relay on loopback, directly, and the env's
MCP servers over TLS through the egress proxy, `letta mcp call` included. The workspace changelog's tar runs on real
time, as file times do. Nothing on the CLI's headless path ages conversations or tool output by file times. What still
reads the env's clock against other times:
- A commit to the git-backed memory, the bridge's or the CLI's, carries the env's date. Before 1970 git stores an
  overflowed time, and after 2099 a broken time zone, which `git fsck` flags; `commit`, `ls-tree`, `show`, `log` and
  `status`, which the bridge and the CLI run on it, work.
- The memory tool's listing of a folder shows its 10 most recently edited entries, dating committed ones by their
  commits, on the env's clock, and uncommitted ones by their files' times, which are real. A clock ahead of real time,
  or before 1970, lists committed entries first and one behind it uncommitted ones, which changes the 10 a larger
  folder shows.
- A jump back across a compaction makes the CLI's pre-turn compaction check estimate the context's size from its text
  rather than from the last reported usage.
- Messages written on 2026-01-01 UTC, the CLI's placeholder date, get timestamps interpolated from the transcript
  file's times when loaded, which changes neither their content nor their order.
- A `Retry-After` given as a date (LiteLLM and the providers give seconds) is read against the env's date, and the CLI
  waits at most 60 s.
- File locks time out, and those that don't go by their owner's process go stale, on the env's clock, so a rate above
  1 or a jump forward shortens both. A context's own commands are the only ones to take its locks. A jump back
  lengthens both: a lock a killed CLI left, the reflection transcript's say, goes stale only once the env's clock
  passes the time it was taken plus 90 s, and until then every turn waits 10 s for it at its end and leaves its messages
  out of that transcript, logging `Failed to append transcript delta` in the context's `.letta/logs/debug`. A jump back
  while a command waits for a held lock stretches the wait by the jump, and a frozen clock (rate 0) stretches it to the
  turn's timeout.
- On a frozen clock, every assistant message from a provider that gives no response id (LiteLLM gives one) takes the
  same id, made from the time.

Run `agent-env run agents --task letta-code-smoke` for the shell smoke task and
`agent-env a2a-agent validate --id letta-code --sandbox local` for the repository's validator.
