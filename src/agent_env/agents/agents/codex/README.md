# Codex A2A agent

OpenAI's [Codex CLI](https://github.com/openai/codex) wrapped with the `agentenv_protocol.a2a_agent` framework,
shipped in the `agents` bundle as `@local/agentenv-framework/agents/codex`. This directory is the whole build
context: the protocol package comes from PyPI (`requirements.txt`), Node is pinned by version and checksum in the
`Dockerfile`, and the Codex CLI and `mcp-remote` by `package-lock.json`.

Each task runs `codex exec --json --dangerously-bypass-approvals-and-sandbox` once in `/app`, the prompt on stdin. A
context's first task starts a Codex thread, captured from its `thread.started` event, and each later task resumes it
with `codex exec resume <thread_id>`. A turn that fails on a transient model error (a 5xx, a dropped stream) is resumed
in the same thread, up to `CODEX_API_ERROR_RETRIES` times (2 by default) while the turn's deadline allows. A turn the
CLI stays silent through for 300 s, or 900 s at `xhigh` or `max` effort, is stopped as stalled;
`AGENT_CLI_INACTIVITY_TIMEOUT_S` sets another bound.

Codex reads its settings from `$CODEX_HOME/config.toml`, which the agent writes:

- At start, `model_providers.litellm` points Codex at `$LITELLM_BASE_URL/openai/v1` with the Responses API, its key
  from `LITELLM_API_KEY`, `[model.headers]` (`LITELLM_EXTRA_HEADERS`) on every model call, and a run's `project_id`
  and `task_id` as `x-litellm-customer-id` and `x-litellm-tags`.
- Before each task, every registered MCP server, the task's `peer_agents` tools included, as an `mcp-remote` stdio
  server with the role's `AgentEnv-Role` header. Codex hands such a server only a few of its own variables, so the
  agent writes in `NODE_EXTRA_CA_CERTS`, `NODE_TLS_REJECT_UNAUTHORIZED` and `SSL_CERT_FILE` when its environment has
  them.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `CodexConfig`: `model`, `effort` (Codex's `model_reasoning_effort`; a set without it clears it), `system_prompt` (Codex's `developer_instructions`), `output_format` (a JSON schema, as `--output-schema`), `project_id`, `task_id`, `timeout_seconds` |
| `mcp-config/v1` | SDK default; written to `config.toml` before each task |
| `skill-config/v1` | inline and bundle skills installed as `~/.agents/skills/<name>/SKILL.md`, where Codex finds them |
| `trajectory/v1` | the task's `codex exec --json` events, live while it runs (format `codex-exec-json/1`) |
| `snapshot/v1` | `save` uploads the thread's rollout JSONL and the workspace as a gzipped tar, each held to a size limit, once no task of the context runs; `load` writes the rollout under a fresh thread id, replaces the workspace's contents (extracted aside first) and binds the thread to the target context |
| `peer-agents/v1` | SDK default: the CLI reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the last CLI run sent |
| `install/v1` | `install.sh`, into a task container (as root, the CLI working in `/app`) or onto a `vm` sandbox's Linux host: Node and uv checked against their checksums, the CLIs from `package-lock.json` and the agent's requirements in a Python of its own. Node's builds need glibc |
| `triggers/v1` | SDK default |

The image is `python:3.11-slim` with Node, the CLIs and the agent, running as the `agent` user; Codex's state is in
`/home/agent/.codex` whatever the uid.
