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
  `NODE_TLS_REJECT_UNAUTHORIZED` and `SSL_CERT_FILE` for it to pass on.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `CodexConfig`: `model`, `effort` (Codex's `model_reasoning_effort`; a set without it clears it), `system_prompt` (Codex's `developer_instructions`), `output_format` (a JSON schema, as `--output-schema`), `project_id`, `task_id`, `timeout_seconds`, `builtin_tools` (below) |
| `mcp-config/v1` | SDK default; written to `config.toml` before each task |
| `skill-config/v1` | inline and bundle skills installed as `~/.agents/skills/<name>/SKILL.md`, where Codex finds them |
| `trajectory/v1` | the task's `codex exec --json` events, live while it runs (format `codex-exec-json/1`) |
| `snapshot/v1` | `save` uploads the thread's rollout JSONL and the workspace as a gzipped tar, each held to a size limit, once no task of the context runs; `load` writes the rollout under a fresh thread id, replaces the workspace's contents (extracted aside first) and binds the thread to the target context |
| `peer-agents/v1` | SDK default: the CLI reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the last CLI run sent |
| `install/v1` | `install.sh`, into a task container (as root, the CLI working in `/app`) or onto a `vm` sandbox's Linux host: Node and uv checked against their checksums, the CLIs from `package-lock.json` and the agent's requirements in a Python of its own, which the target's own Node and Python are never replaced by; the model's commands get the agent's where the target has none. Node's builds need glibc |
| `triggers/v1` | SDK default |

The image is `python:3.11-slim` with Node, the CLIs, the agent and the Python packages `model-requirements.txt` pins for
the model's scripts. It starts as root, runs the env clock's follower and egress proxy under supervisord (`start-agent`,
`root.conf`) and the A2A server as the `agent` user; started as another user, it runs the server alone. Either way the
server's exit ends the container: supervisord never starts it again, and its `agent-exit` listener shuts supervisord
down, while the follower and the proxy restart in place. Codex's state is in `/home/agent/.codex` whatever the uid.

A task's `sync_env_clock` with `agent_clock` has the agent follow the env's clock (`clock/v1` `sync_time`). agent-env's
libfaketime, preloaded for the `agent` user, then gives the model's commands the env's time. The Codex CLI is a static
binary the library doesn't reach, so the CLI and what it keeps (rollouts, `state.db`, thread ids) stay on real time,
and it checks the model endpoint's certificate on real time and reaches it directly. On an armed clock each CLI run
also gets:

- each MCP server's `mcp-remote` started by Node itself (`-c mcp_servers.<name>.command` and `.args`) with
  `AGENTENV_REAL_TIME=1` in its environment, so it stays on real time: the expiry of its cookies and tokens and the age
  of its leases are server times or real file times. Its `#!/usr/bin/env node` would start `env` first, which takes the
  marker out of what it hands Node. On real time it checks certificates against the real date, so it reaches the
  servers directly.
- `-c shell_environment_policy.set.<name>=...` with the egress proxy's variables (`agent_egress.egress_env`), so the
  model's commands, which check certificates on the env's date, go out through the proxy, all but loopback, the hosts
  `NO_PROXY` names and the model endpoint's and MCP servers' hosts served over plain HTTP alone.
- the env's date and zone before the prompt (`<environment_clock>`), as `date` shows it: Codex puts the host's real
  date in its `<environment_context>`, and the note says so, and that a clock tool and `Date` in JavaScript the model
  runs in code mode read real time too.

Unarmed, a CLI run is as before. So is one of an agent installed with `install/v1`, which runs as root, whose
commands libfaketime doesn't fake, whatever the clock of a computer it's installed in.

A `deploy_agent` `builtin_tools` selection turns the CLI's own tools off with `-c` overrides on the task's CLI alone;
config.toml stays as it is. `{"disable": "*"}` turns off every setting below, `{"only": [...]}` every one but those whose
tools it names, and a list to turn off those whose tools it names. A list names all of a setting's tools or none:

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
