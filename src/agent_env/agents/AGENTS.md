# AGENTS.md: writing an A2A agent

How to write an agent that agent-env can deploy, prompt, inspect and validate. `README.md` here describes the agents
in this folder.

## Agent

An agent is a Docker image running an A2A server, built with the SDK, `agentenv_protocol.a2a_agent`
(`pip install 'agentenv-framework-protocol[agent]'`). You write one `AgentEnvAgent` subclass, decorated with
`@a2a_agent(...)`: its identity, a typed config, the extensions it supports, `run()` for one task, and handlers for
what only your runtime can do. The SDK serves the Agent Card, the A2A JSON-RPC endpoint, `/health` and every `/ext/*`
route. The end of `agents/claude-code/a2a_server.py` is the whole entry point:

```python
def build_app(agent: ClaudeCodeAgent) -> Starlette:
    return json_error_bodies(agent.create_app())


app = build_app(ClaudeCodeAgent())
AGENT_CARD = app.state.agentenv_a2a.card    # install/v1 imports this name
```

`agents/claude-code/entrypoint.sh` serves it with `exec uvicorn a2a_server:app --host 0.0.0.0 --port
"${A2A_PORT:-8000}"`, from the image's `CMD` (not `ENTRYPOINT`), as a non-root user that tolerates root, with
everything installed at build time.

Paths below are under `agents/claude-code/`. `kit.py` there holds the plumbing any CLI agent needs; copy it rather
than rewriting it.

| To | Read |
|---|---|
| declare the agent | `a2a_server.py`, the `@a2a_agent` block on `ClaudeCodeAgent` |
| map the model endpoint and `LITELLM_EXTRA_HEADERS` onto a CLI | `claude_bridge.py` `cli_env` |
| answer errors as `{"detail": ...}` | `kit.py` `json_error_bodies`, applied in `build_app` |
| stop running turns when the server stops | `_turns_stop_with_the_server`, passed as `lifespan=` |
| build the image | `Dockerfile`, `entrypoint.sh` |
| test the agent offline | `tst/unit/agents/claude_code/` (`conftest.py` builds the app with `TestClient`) |

`packages/agentenv-protocol/examples/` has smaller agents, one feature each.

## Agent loop: handling tasks

agent-env speaks A2A JSON-RPC to `/a2a`: `message/send` for each prompt, then `tasks/get` until the task ends, and
`tasks/cancel` to stop one (`src/agent_env/a2a_agent/protocol.py`). The SDK owns the task: it creates it
(`submitted`, then `working`), builds a `TaskRequest`, calls your `run()` and maps what it returns onto the task's
terminal state, `completed`, `failed` or `canceled`. Your `run()` is one turn.

- **In:** `TaskRequest` carries `task_id`, `context_id`, `parts` (`TextPart`, `FilePart` by `bytes` or `uri`,
  `DataPart`), `config` (your typed config, as it stood when the task was created), `mcp_servers`, `skills`,
  `metadata`, `session_ref`, `workspace` and `trajectory`.
- **Out:** a `TaskResult` from `TaskResult.text(...)`, `TaskResult.failure(code, message)` or `TaskResult.builder()`.
  Its parts become the task's status message, with `usage` as a data part; a failure adds
  `{error_type, error_code, error_message}`. A success needs at least one part, and `builder().failed(...)` adds no
  text part, so add one.
- **Errors:** `agent_error`, the default, means the agent tried and failed; `infra_error` means the run couldn't
  happen, and agent-env may retry it. An exception that escapes `run()` becomes an opaque `infra_error`, its text
  never sent.
- **Context and sessions:** turns of one `context_id` run one at a time; different contexts run concurrently, and the
  validator relies on it, so don't serialize them. Return your runtime's conversation id as `session_ref` and the next
  turn in that context gets it back; bind it before the runtime starts, so a live trajectory read or a mid-turn
  snapshot finds it.
- **Deadline and cancel:** bound the turn by `request.config.timeout_seconds`. On `tasks/cancel`, `run()` gets
  `CancelledError`; kill the runtime and anything it started before re-raising.
- **Streaming:** make `run()` an async generator that yields `TaskProgress` and ends with one `TaskResult`; the card
  then advertises streaming.

| To | Read |
|---|---|
| handle a turn: parts in, session bound, CLI run, result out | `ClaudeCodeAgent.run` and `_turn` |
| turn file parts into files the runtime reads | `kit.py` `materialize_file_part` |
| run the CLI within the deadline, streaming its events | `claude_bridge.py` `run_claude` |
| kill the CLI and its tools on timeout or cancel | `claude_bridge.py` `_kill_cli`, `kill_running_clis` |
| keep session refs past the SDK's 1,024-context LRU | `kit.py` `DurableSessionRefs` |
| test the loop | `tst/unit/agents/claude_code/test_message_contract.py`, `test_claude_bridge_cancel.py` |
| stream progress | `packages/agentenv-protocol/examples/streaming_agent.py` |

## Agent Card

The SDK builds the card from `@a2a_agent(...)` and serves it at `/.well-known/agent-card.json` and
`/.well-known/agent.json`, the path agent-env reads. You never write it by hand.

- **Identity:** `AgentIdentity(name, description, version, input_modes, output_modes, skills)`. List in
  `input_modes` only the media types `run()` handles; the validator sends every modality probe and records your
  declared modes beside the results. `skills` here describe the agent for discovery; they are not installed skills.
- **Derived:** the RPC `url` (`/a2a`), `capabilities.streaming` from the shape of `run()`, and
  `capabilities.extensions`, one entry per extension with `params.endpoint` and `params.methods`. For agent-config,
  `methods.set.request.supported` lists your config fields: agent-env sends only those. `staging/v1` is appended.
- **Live:** setting `name` or `description` through agent-config, or installing a skill, updates the served card.

Reference: the `identity=AgentIdentity(...)` block on `ClaudeCodeAgent`, and
`tst/unit/agents/claude_code/test_card_contract.py`, which pins the card.
`curl localhost:8000/.well-known/agent-card.json` shows a running agent's card.

## Extensions

Each extension is a `urn:agentenv:*` URI on the Agent Card, most with an `/ext/*` endpoint. An operation is
implemented by the SDK or by your runtime. List an extension in `extensions=(...)`, with `enable(EXT, ...)` for
options; binding a handler with `@extension(EXT.operation)` also turns it on. Overriding an SDK-owned operation
means overriding its whole group, and `self.default_handlers.call(...)` reaches the SDK's own handler from yours.
Shapes are in `packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/extensions.py`.

### `agent-config/v1`, `/ext/agent-config`

agent-env configures the agent: identity at deploy, then `model`, `system_prompt`, `timeout_seconds` and the rest
before each prompt. The SDK implements `set` and `get`; you declare an `AgentConfig` subclass with `config=` and read
`request.config` in `run()`. Only the fields you declare are sent to you, every field needs a default, and secrets
are `WriteOnly`. `prompt_agent` always sends `timeout_seconds` (600 unless the step sets one), and each `set` merges
into the last, so a field one prompt sets stays set until another sets it.

Reference: `ClaudeCodeConfig` and `config=ClaudeCodeConfig`. Overriding `set`/`get`:
`packages/agentenv-protocol/examples/advanced_agent.py`.

### `mcp-config/v1`, `/ext/mcp-config`

agent-env registers each env's MCP server as `{url, name, headers}`. The SDK implements `add` and `list`; you hand
`request.mcp_servers` to the runtime every turn, keeping the names and passing the headers through.

Reference: `enable(MCP_CONFIG_V1, ...)`, and `claude_bridge.py` where it builds `--mcp-config`.

### `trajectory/v1`, `/ext/trajectory`

agent-env reads what the agent did, and verifiers grade it. Call `request.trajectory.set_format(...)`, then append
each native event as it happens; with `enable(TRAJECTORY_V1, live=True)` it can be read while the turn runs. The SDK
serves the reads; bind `get.context` and `get.context_objects` to serve a whole conversation from your own record.

Reference: `enable(TRAJECTORY_V1, live=True, ...)`, `ClaudeCodeAgent.run` passing
`event_callback=request.trajectory.append`, and `get_context_trajectory` / `upload_context_trajectory`. A runtime
that only has a record at the end returns it with `native_trajectory(...)`: `agents/pi/pi_agent.py` `_result`.

### `skill-config/v1`, `/ext/skill-config`

agent-env installs skills, inline (`skill_md`) or as a bundle of files behind read grants. You implement both
`add.inline` and `add.bundle`; the SDK implements `list`, rejects duplicate names and passes installed skills in
`request.skills`.

Reference: `ClaudeCodeAgent.add_inline_skill` / `add_bundle_skill`, both calling `kit.py` `install_skill`.

### `snapshot/v1`, `/ext/snapshot`

agent-env saves an agent's conversation and workspace and restores them into another agent. You implement `save`,
uploading both the trajectory and the workspace object through the grants (an empty workspace if there is none),
and `load`, restoring them into `target_context_id`. `load` runs on a fresh agent, so create state directories at
startup. The optional changelog uploads one increment per tool call and replays them in order.

Reference: `ClaudeCodeAgent.save_snapshot` / `load_snapshot`. Changelog: `agents/pi/pi_agent.py`
`enable_changelog` / `apply_changelog`, and `agents/pi/changelog.py`, which captures in the agent's process; for a
CLI that runs its tools itself, `ClaudeCodeChangelogAgent` and `agents/claude-code/tar_changelog.py`, which capture from
the CLI's post-tool hook, and only where the host has GNU tar.

### `peer-agents/v1`, `/ext/peer-agents`

agent-env tells the agent about the task's other agents. Enabled with no handlers, the SDK (protocol 0.1.350 or
newer) serves six MCP peer tools and adds them to `request.mcp_servers` as `peer_agents`. To serve your own,
override `set` and `list` and register your MCP server through `default_handlers.call(MCP_CONFIG_V1.add, ...)` on a
`127.0.0.1` URL ending in `/mcp`.

Reference: `enable(PEER_AGENTS_V1, ...)`. Your own tools: `agents/pi/pi_agent.py` `set_peers` / `list_peers`, and
`agents/pi/peers.py`.

### `triggers/v1`, `/ext/triggers`

agent-env registers scripted triggers and asks, each turn, what an agent playing a user or stakeholder says next.
The SDK implements `register`, `decide` and `state`; enable it and do nothing else.

Reference: `enable(TRIGGERS_V1, ...)`. Overriding `decide` to change the decision:
`packages/agentenv-protocol/examples/advanced_agent.py`.

### `attribution-probe/v1`, `/ext/attribution-probe`

The agent reports the attribution it last sent with a model request. You implement `probe`, which also declares the
extension.

Reference: `agents/codex/a2a_server.py` `probe_attribution`, and
`test_attribution_probe_reports_what_the_agent_last_sent` in `packages/agentenv-protocol/tests/test_a2a_agent.py`.

### `install/v1` (no endpoint)

A card declaration that lets a task install the agent into a container it already runs, or onto a VM sandbox's host.
`install_agent` imports `AGENT_CARD` from the image's `a2a_server` module, with `/app` on `sys.path` and without your
entrypoint, then fills each `{param}` of the commands in, quoted. Pass the endpoint as environment on the command
that needs it, not as arguments; register the image with `--dockerfile`, since its build context is what gets
installed.

Reference: `enable(INSTALL_V1, ...)` with `_INSTALL_COMMANDS`, `_INSTALL_COMMANDS_HOST` and `_ENDPOINT`, and
`install.sh`.

### `staging/v1`, `/ext/staging`

An object store on the agent's own server, which agent-env uses when the agent can't reach its object store. It is
on by default and needs nothing from you; `AGENTENV_STAGING_MAX_BYTES=0` turns it off.

Reference: `packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/staging.py`.

### Your own extensions

Use your own URI, not `urn:agentenv:*`. Reference: `packages/agentenv-protocol/examples/custom_extensions_agent.py`.

## Adding your own agent

1. **Make a bundle with an agent folder.** A bundle is a folder; `agents/<name>/` in it, with a `Dockerfile`, is an
   agent named `<name>`. Start from a copy of `agents/claude-code/`: `a2a_server.py`, `kit.py`, `Dockerfile`,
   `entrypoint.sh`, and a `requirements.txt` with `agentenv-framework-protocol[agent]`.
2. **Declare the agent** (Agent, Agent Card): an `AgentConfig` subclass with only the fields your runtime honours,
   and `@a2a_agent(identity=AgentIdentity(...), config=..., extensions=(...))` with the `input_modes` `run()` handles.
3. **Write the turn** (Agent loop): `run()` turns `request.parts` into a prompt, runs your runtime within
   `request.config.timeout_seconds` with `request.mcp_servers` connected, records the trajectory, and returns a
   `TaskResult` with a `session_ref`. Replace `claude_bridge.py` with your runtime's bridge.
4. **Map the model endpoint**: read `LITELLM_BASE_URL`, `LITELLM_API_KEY` and `LITELLM_EXTRA_HEADERS` and hand them
   to your runtime, as `claude_bridge.py` `cli_env` does.
5. **Add extensions** (Extensions): `mcp-config` and `trajectory` first, then every other one your runtime can
   honour. Keep `AGENT_CARD = app.state.agentenv_a2a.card` at the end of `a2a_server.py`.
6. **Test it offline** with Starlette's `TestClient` and your runtime faked, as `tst/unit/agents/claude_code/` does:
   the card, one turn, one failure, a cancel.
7. **Add defaults and a smoke task.** An optional `agents/<name>/agent.toml` holds `[default_env_vars]` and
   `[metadata]` (`default_model`, `min_disk_size_gb`). `tasks/<name>-smoke.json` deploys the agent by folder name,
   prompts it once and checks the reply, as `tasks/claude-code-smoke.json` does.
8. **Run it**: `agent-env run ./my-bundle` builds the image, registers the agent and runs the bundle's tasks. It needs
   Docker and a `[model]` endpoint in `.agentenv/config.toml`.
9. **Register and validate it**: `agent-env a2a-agent put` stores the image and registers the agent under an id any
   task can name in `deploy_agent`'s `a2a_agent_id`, then runs the validator; fix what fails and validate again.
10. **Use it**: name it in a task's `deploy_agent` step, or make it the agent every `deploy_agent` without an id gets
    with `[agents] default_a2a_agent_id` in `.agentenv/config.toml`.

```bash
agent-env run ./my-bundle                                    # build, register under @local, run its tasks
agent-env a2a-agent put --id my-agent --dockerfile my-bundle/agents/my-agent/Dockerfile
agent-env a2a-agent validate --id my-agent                   # agent-env's built-in checks
agent-env a2a-agent get --id my-agent                        # the validation results
```

`validate` runs a fixed set of checks against the reference env `mail`, an inbox with a `search_emails` tool: the alias
of the `mail-mcp` example's env (`src/agent_env/examples/mail-mcp/`), which `validate` writes before it runs, with its
image context-only on a remote sandbox, when the store has no env of that id. The checks are in
`src/agent_env/a2a_agent/validator.py` and `src/agent_env/task_step/task_steps/a2a_agent_validator/`, one step each. It doesn't cover everything an agent can
advertise: `triggers/v1`, `attribution-probe/v1`, host-mode `install/v1` and custom extensions need tests of your
own.

## The protocol

- `extensions.py` (above) is the source of truth for the wire format. `tasks/v1.py` next to it is `TaskRequest`,
  `TaskResult` and `TrajectoryLog`; `agentenv_protocol/transfers.py` is `upload`, `download` and
  `NamespaceUploader` for object grants.
- `packages/agentenv-protocol/README.md`, section "A2A agent framework", is the prose reference.
- What agent-env sends you: `src/agent_env/task_step/task_steps/deploy_agent.py` (env vars, MCP registrations,
  identity) and `prompt_agent.py` (config fields, then `message/send`).
