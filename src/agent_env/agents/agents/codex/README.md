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
  and `task_id` as `x-litellm-customer-id` and `x-litellm-tags`. The run's tags join an `x-litellm-tags` that
  `[model.headers]` names, whose own keys, such as `taskId`, win. Its `capabilities.remote_compaction = "v2"` makes
  Codex compact through the endpoint, with the turn's tools declared, instead of locally. `tools.update_plan.enabled`
  is set too, since Codex leaves `update_plan` off by default.
- Before each task, every registered MCP server, the task's `peer_agents` tools included, as an `mcp-remote` stdio
  server. Its `AgentEnv-Role` header is the role its registration names, unless the task has a role of its own, which
  reaches that task's CLI alone as `-c` overrides, so tasks of different roles running at once share the file. Codex
  hands such a server only a few of its own variables, so the entry names `NODE_EXTRA_CA_CERTS`,
  `NODE_TLS_REJECT_UNAUTHORIZED` and `SSL_CERT_FILE` for it to pass on, and the proxy variables an armed clock sets
  (below); `--enable-proxy` has mcp-remote take them, and connect directly when none is set.

Each CLI run calls the endpoint through a model relay of its own on loopback (`model_relay.py`, shared with the other
agents'), given to it as a `-c` override of the provider's `base_url`, which forwards each call and its answer as they
come.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `CodexConfig`: `model`, `effort` (Codex's `model_reasoning_effort`; a set without it clears it), `system_prompt` (Codex's `developer_instructions`), `output_format` (a JSON schema, as `--output-schema`), `project_id`, `task_id`, `timeout_seconds`, `builtin_tools`, `web_tools` (below) |
| `mcp-config/v1` | SDK default; written to `config.toml` before each task |
| `skill-config/v1` | inline and bundle skills installed as `~/.agents/skills/<name>/SKILL.md`, where Codex finds them |
| `trajectory/v1` | the task's `codex exec --json` events, live while it runs (format `codex-exec-json/1`) |
| `snapshot/v1` | `save` uploads the thread's rollout JSONL and the workspace as a gzipped tar, each held to a size limit, once no task of the context runs; `load` writes the rollout under a fresh thread id, replaces the workspace's contents (extracted aside first) and binds the thread to the target context |
| `peer-agents/v1` | SDK default: the CLI reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the last CLI run sent |
| `install/v1` | `install.sh`, into a task container (as root, the CLI working in `/app`) or onto a `vm` sandbox's Linux host: Node and uv checked against their checksums, the CLIs from `package-lock.json` and the agent's requirements in a Python of its own, which the target's own Node and Python are never replaced by; the model's commands get the agent's where the target has none. Node's builds need glibc |
| `triggers/v1` | SDK default |

The image is `python:3.11-slim` with Node, the CLIs, the agent and the Python packages `model-requirements.txt` pins for
the model's scripts. It starts as root (`start-agent`), whose supervisord runs the A2A server as the `agent` user;
started as another user, it runs the server alone, as before. Codex's state is in `/home/agent/.codex` whatever the uid.

The image follows a task env's clock as the agents README describes, once a `sync_env_clock` sets `agent_clock`. The
Codex CLI is a static binary, which the preloaded libfaketime never reaches, so the CLI itself stays on real time: its
threads, their rollouts and the times in its events keep real time, and a clock set behind them breaks nothing. The
commands the model runs, and the MCP clients the CLI starts, read the env's time. The CLI writes the date it tells the
model, the `<current_date>` of its environment context, and the time of the reminder it gives a turn at `persistent`
effort, from its own clock, so while the clock is armed the run's relay puts the env's date and time there, read from
`date`, in every message of each model call, a resumed thread's earlier ones included (`codex_date.py`). A call whose
date the relay can't read, or that writes a date or time in a form it doesn't know, is refused, so the turn fails rather
than tell the model the machine's date. The `<timezone>` the CLI names is the container's, which the clock doesn't
change. On an armed clock the CLI's commands and MCP clients go out through the egress proxy, its model calls go to the
relay directly, and the relay, on real time, reaches the endpoint directly. Without a `LITELLM_BASE_URL` the CLI has no
provider of agent-env's and no relay, so nothing tells it the env's date.

For a model in its catalog, such as `gpt-6-astra`, the CLI runs in code mode: the model calls its tools from JavaScript
it hands the `exec` tool, which runs in `codex-code-mode-host`, another static binary. The tools it calls there follow
the clock as above, but JavaScript's own `Date` in that code reads real time; the `clock` namespace's `sleep` only
waits, so it's unaffected.

A `deploy_agent` `builtin_tools` selection turns the CLI's own tools off with `-c` overrides on the task's CLI alone;
config.toml stays as it is. `{"disable": "*"}` turns off every setting below, `{"only": [...]}` every one but those whose
tools it names, and a list to turn off those whose tools it names. A list names all of a setting's tools or none:

A `deploy_agent` `web_tools: false` adds `-c web_search="disabled"`, which takes the CLI's web search tool off,
whatever `builtin_tools` selects.

| Tools | Override |
|---|---|
| `exec_command`, `write_stdin` | `features.shell_tool=false` |
| `update_plan` | `tools.update_plan.enabled=false` |
| `request_user_input` | `tools.experimental_request_user_input.enabled=false` |
| `view_image` | `features.view_image=false` |
| `spawn_agent`, `send_message`, `wait_agent`, `followup_task`, `interrupt_agent`, `list_agents` | `features.multi_agent_v2=false` and `features.multi_agent=false`, since the v1 tools come back behind `tool_search` otherwise |
| `create_goal`, `get_goal`, `update_goal` | `features.goals=false` |
| `web_search` | `web_search="disabled"` |

No setting turns off `apply_patch`: a model's entry in Codex's model catalog offers it (every GPT model 0.162.1 ships
with does). So a selection without it (`{"disable": "*"}`, a list naming it, or an `{"only": [...]}` that doesn't)
writes a copy of the catalog the CLI would use, with `apply_patch_tool_type` null for every model, to
`$CODEX_HOME/model_catalog_without_apply_patch.json`, and points the task's CLI at it with `-c model_catalog_json=...`.
The copy is made from `codex debug models --bundled`, or from the file config.toml's own `model_catalog_json` names.

The MCP tools agent-env gives the agent stay with any selection, and so do the tools Codex adds for them:
`tool_search`, `list_mcp_resources`, `list_mcp_resource_templates` and `read_mcp_resource`. The names are `CodexTool`'s
in `codex_config.py`; a test fails when package.json's `@openai/codex` moves, until `CodexTool` and
`TOOLS_CLI_VERSION` move with it.
