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

The image is `python:3.11-slim` with Node, the CLI and the agent, running as the `agent` user; the CLI's state is in
`/home/agent/.gemini` (`GEMINI_CLI_HOME`) whatever the uid.
