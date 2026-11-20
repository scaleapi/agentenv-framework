# Gemini A2A agent

Google's [Gemini CLI](https://github.com/google-gemini/gemini-cli) wrapped with the `agentenv_protocol.a2a_agent`
framework, shipped in the `agents` bundle as `@local/agentenv-framework/agents/gemini`. This directory is the whole
build context: the protocol package comes from PyPI (`requirements.txt`), Node is pinned by version and checksum in the
`Dockerfile`, and the Gemini CLI by `package-lock.json`.

Each task runs `gemini -p <prompt> --output-format stream-json --approval-mode yolo` once in `/workspace`, the CLI's
working directory and the boundary of its file tools. A context's first task starts a Gemini session, captured from the
stream's `init` event, and each later task resumes it with `-r <session>`. File parts are written under
`/workspace/inputs/<task>/` and attached to the prompt as `@path`.

Each run gets its own settings, as the CLI's system settings file (`GEMINI_CLI_SYSTEM_SETTINGS_PATH`), so tasks of
different roles or system prompts running at once share none:

- the registered MCP servers and the task's `peer_agents` tools as `httpUrl` servers, with the task's role as their
  `AgentEnv-Role` header, else the role their registration names. Gemini splits `mcp_<server>_<tool>` on the first
  `_`, so server names carry none: `peer_agents` is `peeragents`.
- the task's `system_prompt` as a context file the CLI layers on its own system prompt, beside any `GEMINI.md` in the
  workspace.

The CLI's model calls go to a proxy in the agent, `/litellm-gemini/<run>/...`, which adds `[model.headers]`
(`LITELLM_EXTRA_HEADERS`), then the run's `project_id` and `task_id` as `x-litellm-customer-id` and `x-litellm-tags`
(the run's tags join an `x-litellm-tags` that `[model.headers]` names, whose own keys, such as `taskId`, win),
and forwards to `$LITELLM_BASE_URL/gemini/...`, with TLS checks on. A model named with a provider prefix, such as `vertex_ai/global/gemini-3.1-pro-preview`, is handed to the
CLI by its last segment, and the proxy translates its generate calls to `/v1/chat/completions` and back.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `GeminiConfig`: `model`, `system_prompt`, `allowed_tools` (read back, but the CLI has no per-run tool allowlist), `project_id`, `task_id`, `timeout_seconds` (600 by default) |
| `mcp-config/v1` | SDK default, with names stripped of `_`; written to each run's settings |
| `skill-config/v1` | inline and bundle skills installed as `~/.gemini/skills/<name>/SKILL.md`, where the CLI finds them |
| `trajectory/v1` | the task's stream-json events, live while it runs (format `gemini-stream-json/1`) |
| `snapshot/v1` | `save` uploads the session's transcript, byte for byte, and an empty workspace object; `load` plants it under a fresh session id and binds the target context |
| `snapshot/v1` changelog | where the host has GNU tar: an AfterTool hook captures each tool call's writes as an incremental tar uploaded through the namespace grant; `apply` replays them in order and, with `resume_conversation`, binds the last applied tool call's session |
| `peer-agents/v1` | SDK default: the CLI reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the proxy last sent |
| `install/v1` | `install.sh`, into a task container (as root, the CLI working in `/workspace`) or onto a `vm` sandbox's Linux host: Node and uv checked against their checksums, the CLI from `package-lock.json` and the agent's requirements in a Python of its own, which the target's own Node and Python are never replaced by; the model's commands get the agent's where the target has none. Node's builds need glibc |
| `triggers/v1` | SDK default |
| `clock/v1` | the framework's `sync_time`, in the image, which runs the clock follower |

The image is `python:3.11-slim` with Node, the CLI, the agent and the Python packages `model-requirements.txt` pins for
the model's scripts. It starts as root, whose supervisord (`start-agent`, `root.conf`) runs the clock follower and the
egress proxy, each as its own user, and the A2A server as the `agent` user, on real time; the CLI's state is in
`/home/agent/.gemini` (`GEMINI_CLI_HOME`) whatever the uid. Started as another user, it runs the A2A server alone.

## The env's clock

The image can follow a task env's clock as Claude Code's does, with agent-env's libfaketime preloaded for the `agent`
user and the computer base's follower and egress proxy (`src/agent_env/agents/README.md`), and only once a
`sync_env_clock` sets `agent_clock`. Unarmed, the CLI runs exactly as it did before it could: the same environment, and
started as `gemini`. On an armed clock the CLI itself stays on real time, and only the commands the model runs follow
the clock:

- **The CLI, on real time.** Each turn starts the CLI's script with node itself, with agent-env's real-time marker and
  `GEMINI_CLI_NO_RELAUNCH`, so neither `gemini`'s `#!/usr/bin/env node` line nor the CLI's relaunch of itself hands node
  an environment the marker is gone from. It judges its `projects.json` lock stale, its sessions, tool outputs and logs
  old and its credentials expired by real file and server times, which would be wrong on the env's clock, and it checks
  certificates against real time, so it reaches its model relay and the env's MCP servers directly, an HTTPS server on
  Modal included, whatever the clock shows. It keeps Node's default heap, which its relaunch would have raised to half
  the host's memory.
- **The model's commands, on the env's clock.** Every bash the CLI starts, each command the model runs and each hook,
  first reads a file beside the run's settings (`BASH_ENV`) that exports the egress proxy by the rule every clocked
  agent shares (`agent_egress.egress_env`): every TLS host through the proxy but loopback, `host.docker.internal`, the
  deploy's `NO_PROXY` hosts and hosts reached over plain HTTP alone. The CLI itself is handed no proxy, since it would
  send every request through one, `NO_PROXY` or not, its loopback relay included.
- **The env's date.** The CLI words its "Today's date is" line from its own clock, so the model relay names the env's
  date there instead, read as the turn starts from `date`, which follows the clock as the model's commands do. A CLI
  that calls no relay, with no `LITELLM_BASE_URL` or with a `GOOGLE_GEMINI_BASE_URL` of its own, shows the real date
  there. The commands' `date` and the gateway's `get_time` show the env's time.
- **The changelog, on real time.** The capture hook and its `tar` run with the marker: the hook checks its grant's
  expiry, and tar the times its incremental snapshot compares, against real time, as both are.

A clock armed while a turn runs reaches that turn's CLI, which was started to follow it, without the proxy; the next
turn starts as above.
