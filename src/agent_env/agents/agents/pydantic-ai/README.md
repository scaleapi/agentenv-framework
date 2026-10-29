# Pydantic AI A2A agent

A [Pydantic AI](https://ai.pydantic.dev) agent wrapped with the `agentenv_protocol.a2a_agent` framework, shipped in
the `agents` bundle as `@local/agentenv-framework/agents/pydantic-ai`. This directory is the whole build context: the
protocol package, Pydantic AI and the [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) come from PyPI
(`requirements.txt`), and `install.sh` installs them in a Python of uv's own, with git for the agent's shell, both for
the image and for `install/v1`.

Instead of wrapping a CLI, the agent runs Pydantic AI in process. `pydantic_ai_agent.py` is the A2A side and imports
no Pydantic AI; `pydantic_ai_runtime.py` runs the agent. Each task is one run of an `Agent` with the harness's `Coder`
capability, the coding tools CLAI is built on (`shell`, `read_file`, `write_file`, `edit_file`, `list_files`, `grep`,
`delegate_task`), working in a `LocalWorkspace` on the workspace. Its shell commands get only `PATH`, `HOME` and the
locale from the agent's environment, never the model key, and the harness keeps their logs under
`.pydantic-ai-harness/` in the workspace, with a `.gitignore` of its own. The agent runs in a sandbox of its own, so the file tools
are not confined to the workspace and reach the files attached outside it. Pydantic AI's own A2A server
(`Agent.to_a2a`) is not used: it speaks plain A2A, while agent-env needs the protocol's extensions, which the framework
serves.

The model is called through the `[model]` endpoint as an OpenAI-compatible one, `<base>/chat/completions`, with
`LITELLM_EXTRA_HEADERS` sent on every call but kept out of the persisted messages, which a snapshot uploads. The model
name is passed through as is; a `vllm:` prefix, as in `vllm:qwen3.8-27b`, picks Pydantic AI's vLLM provider, which
fits the requests to the model's family (Qwen, DeepSeek, GLM, Llama, ...) as a vLLM server needs, and `openai:` names
the default.

A context maps to one conversation, whose id is derived from the context id, its message history persisted in
`~/.pydantic-ai-a2a/conversations/<id>/messages.json` once a turn succeeds, so follow-up tasks in the context continue
it, across restarts too. Images are sent to the model as images; every file part is also saved in the conversation's
directory under `inputs/<task id>/` and its path added to the prompt, for the agent's tools, and the conversation's
snapshots carry it. A file part larger than 256 MiB, inline or downloaded, fails the task as `pydantic_ai.attachment`.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `PydanticAIConfig`: `model`, `effort` (Pydantic AI's `thinking` setting), `system_prompt` (added to the agent's instructions), `request_limit` (model requests per turn, default 500), `timeout_seconds` |
| `mcp-config/v1` | SDK default; each server becomes an `MCPToolset` on its URL, with its headers, its tools named as the server names them |
| `skill-config/v1` | inline and bundle skills installed as `SKILL.md` directories under `~/.pydantic-ai-a2a/skills`, offered to the model through the harness's `Skills`, which it loads with `load_capability` |
| `trajectory/v1` | live: the task's Pydantic AI messages (`ModelMessagesTypeAdapter`, format `pydantic-ai-messages/v1`), appended as each request and response completes |
| `snapshot/v1` | `save` uploads the context's conversation directory, its messages and attached files, as a gzipped tar and, when asked, the workspace as another; `load` extracts both aside, validates the messages as Pydantic AI's and points the paths of the attached files at the target's directory before it replaces anything, then replaces the workspace's contents and installs the conversation under the target context's id, so an archive that fails to extract, or holds no conversation, changes nothing |
| `peer-agents/v1` | SDK default: the `peer_agents` MCP server reaches the agent like any other |
| `install/v1` | copies this directory into the task container as `/opt/pydantic-ai-a2a`, runs `install.sh` (Debian or Ubuntu, as root, with network access), writes the LiteLLM credentials and the lines that trust the local transfer CA to a 0600 `agent.env` through a heredoc and starts `start.sh`, which applies them |
| `triggers/v1` | SDK default |

The snapshot changelog (`enable_changelog`/`apply_changelog`) is not implemented, nor is `attribution-probe/v1`.

`PYDANTIC_AI_A2A_MODEL` sets the model when agent-config sets none (by default `anthropic/claude-sonnet-4-6`), as for a
peer that other agents message directly, and `PYDANTIC_AI_A2A_WORKSPACE` the workspace (default `/workspace`, and the
task container's working directory when it's installed there). `a2a_server.py` reads these and the `LITELLM_*`
variables once, at startup: a
`LITELLM_EXTRA_HEADERS` that is not a JSON object stops the agent there, while a missing `LITELLM_BASE_URL` fails each
task as `pydantic_ai.no_endpoint`.

Snapshot `load` answers 409 while a task is running, and 400 for an archive that holds no conversation. A snapshot
`save` waits for its context's running task to end; one that saves the workspace also waits for the tasks running in
other contexts, which work in the same workspace. When the workspace holds the agent's own `~/.pydantic-ai-a2a`, as an
install in a home directory's does, the workspace snapshot leaves it out and `load` keeps it as it is.

Usage is the run's: input, output and cached tokens, model requests and tool calls; no cost. A failed run's error code
says whose failure it was: `pydantic_ai.model_http_<status>` and `pydantic_ai.model_api` for the model endpoint and
`pydantic_ai.error` for anything else that stopped the run, such as an MCP server that can't be reached, are
`infra_error`s; `pydantic_ai.usage_limit` (the request limit) and `pydantic_ai.unexpected_model_behavior` are
`agent_error`s, and `pydantic_ai.history`, a persisted conversation that can't be read, an `infra_error`. A failed
turn leaves the conversation as it was. A timeout (`pydantic_ai.timeout`) counts reading the
task's file parts too and reports what the run had spent. A timeout, a `tasks/cancel` and the server shutting down each
cancel the run, which kills the commands its shell started. Errors are answered as `{"detail": ...}`.

```bash
docker build -t pydantic-ai-a2a src/agent_env/agents/agents/pydantic-ai
docker run -p 8000:8000 -e LITELLM_BASE_URL=... -e LITELLM_API_KEY=... pydantic-ai-a2a
pytest tst/unit/agents/pydantic_ai
```

Pydantic AI's version is pinned in `requirements.txt` and `PYDANTIC_AI_VERSION`, and the harness's beside it; a new
release is a pull request that bumps them.
