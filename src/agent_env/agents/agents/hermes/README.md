# Hermes Agent A2A agent

Nous Research's [Hermes Agent](https://github.com/NousResearch/hermes-agent) wrapped with the
`agentenv_protocol.a2a_agent` framework, shipped in the `agents` bundle as `@local/agentenv-framework/agents/hermes`.
This directory is the whole build context: the protocol package comes from PyPI (`requirements.txt`), and Hermes from
its own installer at a pinned release, checked against its checksum, in the `Dockerfile` (`HERMES_COMMIT`, the
`v0.22.5` release).

Each context is a Hermes profile of its own, `$HERMES_ROOT/profiles/ctx-<hash>/`, with its own config, sessions and
memory. Each task runs `hermes agentenv turn` once in `/app`, a command the agent's Hermes plugin, `agentenv_plugin/`,
adds: it runs the turn in its own process, as `hermes -z` does, rather than handing it to a Hermes gateway, so the turn
ends with that process and a timeout or cancel that kills it, and everything it started, stops the turn. A context's
first task starts a Hermes session, and each later task resumes it with `--resume <session_id>`. Text parts are the
prompt, and file parts are written under `/app/inputs/<task>/` and named in it, for Hermes' file and vision tools; a
PDF goes with its pages' PNGs, rendered with `pdftoppm`. Turns of different contexts run at once, as many as memory fits
at about 400 MB each past the first 1 GB (`HERMES_MAX_CONCURRENT_TURNS` sets the number), and a context's turns one at
a time.

The profile's `config.yaml` is written before each turn:

- the model endpoint, `$LITELLM_BASE_URL/v1`, as a custom OpenAI-compatible provider with its key from
  `LITELLM_API_KEY`, `[model.headers]` (`LITELLM_EXTRA_HEADERS`) and the run's `project_id` and `task_id` as
  `x-litellm-customer-id` and `x-litellm-tags` on every model call, Hermes' auxiliary calls (titles, compression,
  vision) included. The run's tags join an `x-litellm-tags` that `[model.headers]` names, whose own keys, such as
  `taskId`, win;
- the model: the task's, else `HERMES_MODEL`, else `anthropic/claude-sonnet-4-6`;
- every registered MCP server, the task's `peer_agents` tools included, as a remote server. Its `AgentEnv-Role` header
  is the role its registration names, unless the task has a role of its own;
- the skills agent-env installed, as an external skills folder, and the `agentenv` plugin enabled.

The task's `system_prompt` is the profile's `SOUL.md`, the slot Hermes' own identity fills, so it reaches the model as
the system prompt; a session keeps the system prompt it started with, as Hermes does. A turn with no system prompt gets
Hermes' own back. Hermes runs non-interactively: dangerous-command approvals are bypassed, and its `clarify` tool tells
the model no one is there to answer.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `HermesConfig`: `model`, `effort` (Hermes' `--reasoning`), `system_prompt`, `output_format` (a JSON schema the answer is checked against), `project_id`, `task_id`, `timeout_seconds` (1800 by default) |
| `mcp-config/v1` | SDK default; each server written to the turn's profile config |
| `skill-config/v1` | inline and bundle skills installed under `$HERMES_ROOT/agentenv-skills/<name>/`, which every profile reads as `skills.external_dirs` |
| `trajectory/v1` | the turn's Hermes plugin hook events (`on_session_start`, `pre_tool_call`, `post_tool_call`, `post_api_request` with each model response, its tool calls and usage, `api_request_error`, `post_llm_call`, `on_session_end`), live while it runs (format `hermes-hook-events/1`) |
| `snapshot/v1` | `save` archives the context's profile (its sessions database, memory and `SOUL.md`; not the config, caches or logs Hermes rebuilds) with its session id, and `/app`, once no task runs; `load` restores the profile into the target context and replaces `/app`'s contents, so a failed load leaves the workspace as it was. No changelog |
| `peer-agents/v1` | SDK default: Hermes reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the last turn sent |

A turn's usage is Hermes' own report for its main model calls: tokens, `api_calls`, and its estimated cost where the
endpoint's model list prices the model. The image is `python:3.11-slim` with Hermes, `pdftoppm` and the agent, running
as the `agent` user. Hermes' runtime (its own Python, Node and ripgrep) and the profiles are under `/data/.hermes`,
whatever the uid. It has no `install/v1`.
