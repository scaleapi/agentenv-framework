# OpenCode A2A agent

The [OpenCode](https://github.com/anomalyco/opencode) CLI wrapped with the `agentenv_protocol.a2a_agent` framework,
shipped in the `agents` bundle as `@local/agentenv-framework/agents/opencode`. This directory is the whole build
context: the protocol package comes from PyPI (`requirements.txt`), Node is pinned by version and checksum in the
`Dockerfile`, and the OpenCode CLI by `package-lock.json`.

Each task runs `opencode run --format json` once in `/app`. A context's first task starts an OpenCode session, and each
later task resumes it with `-s <session_id>`; each context keeps its sessions in a data directory of its own, under
`$HOME/sessions/`. Text parts are the prompt, after the task's `system_prompt`, and file parts are written under
`/app/inputs/<task>/` and attached with `--file`; a PDF goes as its pages' PNGs, rendered with `pdftoppm` where it is
installed. Turns of different contexts run at once, and a context's turns one at a time.

Each run's settings reach the CLI as `OPENCODE_CONFIG_CONTENT`, so nothing is written for it on disk:

- the model endpoint, `$LITELLM_BASE_URL/v1`, as an OpenAI-compatible provider with its key from `LITELLM_API_KEY`,
  `[model.headers]` (`LITELLM_EXTRA_HEADERS`) and the run's `project_id` and `task_id` as `x-litellm-customer-id` and
  `x-litellm-tags` on every model call. The run's tags join an `x-litellm-tags` that `[model.headers]` names, whose
  own keys, such as `taskId`, win. Each call asks LiteLLM to drop `tool_choice`, which some providers refuse.
- the model: the task's, else `OPENCODE_MODEL`, else `anthropic/claude-sonnet-4-6`;
- every registered MCP server, the task's `peer_agents` tools included, as a remote server. Its `AgentEnv-Role` header
  is the role its registration names, unless the task has a role of its own;
- every tool allowed, and the one that asks the user a question denied.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `OpenCodeConfig`: `model`, `effort` (the provider's `reasoningEffort`), `system_prompt`, `output_format` (a JSON schema the answer is checked against), `project_id`, `task_id`, `timeout_seconds` (1800 by default) |
| `mcp-config/v1` | SDK default; each server passed to the task's CLI |
| `skill-config/v1` | inline and bundle skills installed as `$XDG_CONFIG_HOME/opencode/skills/<name>/`, where OpenCode finds them |
| `trajectory/v1` | the task's `opencode run --format json` events, live while it runs (format `opencode-events/1`) |
| `snapshot/v1` | `save` exports the context's session with `opencode export` and archives `/app`, once no task runs; `load` imports the session into the target context with `opencode import` and replaces `/app`'s contents, so a failed import leaves the workspace as it was. Where `tar` is GNU tar, the changelog too: an incremental archive of its roots after each task, with the roots the task removed, uploaded through the namespace grant, and applied in order, the removed roots removed; an increment is a whole task's changes, since OpenCode has no hook after each tool call, and conversation state is outside the roots |
| `peer-agents/v1` | SDK default: the CLI reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the last CLI run sent |
| `install/v1` | `install.sh`, into a task container (as root, the CLI working in `/app`) or onto a `vm` sandbox's Linux host: Node and uv checked against their checksums, the CLI from `package-lock.json` and the agent's requirements in a Python of its own, which the target's own Node and Python are never replaced by; the model's commands get the agent's where the target has none. Node's builds need glibc |

The image is `python:3.11-slim` with Node, `pdftoppm`, the CLI and the agent, running as the `agent` user. OpenCode's
state is under `/data`, its `HOME`, whatever the uid.
