# AGENTS.md: writing an A2A agent

How to write an agent that agent-env can deploy, prompt, inspect and validate. The reference is the Claude Code agent,
`agents/claude-code/`: agent-env's default agent and its most tested one. `agents/pi/` covers the few patterns Claude
Code doesn't use. `README.md` here describes what each of them does.

## The shape of an agent

An agent is a Docker image running an A2A server, built with the SDK, `agentenv_protocol.a2a_agent`
(`pip install 'agentenv-framework-protocol[agent]'`). You write one class: a typed config, the extensions it supports,
`run()` for one turn, and handlers for what only your runtime can do. The SDK serves the Agent Card, the JSON-RPC
endpoint and every `/ext/*` route. The end of `agents/claude-code/a2a_server.py` is the whole entry point:

```python
def build_app(agent: ClaudeCodeAgent) -> Starlette:
    return json_error_bodies(agent.create_app())


app = build_app(ClaudeCodeAgent())
AGENT_CARD = app.state.agentenv_a2a.card    # install/v1 imports this name
```

and `agents/claude-code/entrypoint.sh` serves it with
`exec uvicorn a2a_server:app --host 0.0.0.0 --port "${A2A_PORT:-8000}"`.

## Where each pattern lives

Paths are under `agents/claude-code/` unless they name `agents/pi/`.

| To | Read |
|---|---|
| declare the agent, its config and extensions | `a2a_server.py`: `ClaudeCodeConfig` and the `@a2a_agent` block |
| run a CLI per turn and stream its events as the trajectory | `ClaudeCodeAgent.run`, `claude_bridge.run_claude` (`event_callback=request.trajectory.append`) |
| map the model endpoint and `LITELLM_EXTRA_HEADERS` onto a CLI | `claude_bridge.cli_env` |
| hand `request.mcp_servers` to the runtime | `claude_bridge.py`, where it builds `--mcp-config` |
| read file parts | `kit.materialize_file_part` |
| install skills, inline and bundle | `kit.install_skill`, bound in `ClaudeCodeAgent.add_inline_skill` / `add_bundle_skill` |
| keep session refs past the SDK's 1,024-context LRU | `kit.DurableSessionRefs` |
| answer errors as `{"detail": ...}` | `kit.json_error_bodies`, applied in `build_app` |
| save and restore a conversation and workspace | `ClaudeCodeAgent.save_snapshot` / `load_snapshot` |
| serve a whole context's trajectory | `ClaudeCodeAgent.get_context_trajectory` (`TRAJECTORY_V1.get.context`) |
| delegate to peer agents | the `enable(PEER_AGENTS_V1, ...)` line: the SDK serves the peer tools |
| install into a running container or onto a VM host | `_INSTALL_COMMANDS`, `_INSTALL_COMMANDS_HOST` and `install.sh` |
| build the image | `Dockerfile` |
| test the agent offline | `tst/unit/agents/claude_code/` (`conftest.py`, `test_card_contract.py`, `test_extensions.py`) |
| return a trajectory once the run ends, not live | `agents/pi/pi_agent.py` `_result` (`native_trajectory`) |
| capture and replay a snapshot changelog | `agents/pi/pi_agent.py` `enable_changelog` / `apply_changelog`, `agents/pi/changelog.py` |
| replace the SDK's peer tools with your own | `agents/pi/pi_agent.py` `set_peers` / `list_peers`, `agents/pi/peers.py` |

`packages/agentenv-protocol/examples/` has smaller agents, one feature each (`basic_agent.py`, `streaming_agent.py`,
`multimodal_agent.py`, `custom_extensions_agent.py`, `advanced_agent.py`).

## The protocol

- `packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/extensions.py` defines every `urn:agentenv:*`
  extension: its endpoint, operations, request and response models, and whether the SDK or your runtime implements
  each operation. It is the source of truth for the wire format.
- `tasks/v1.py` there is `TaskRequest`, `TaskResult` and `TrajectoryLog`; `agentenv_protocol/transfers.py` is
  `upload`, `download` and `NamespaceUploader` for object grants.
- `packages/agentenv-protocol/README.md`, section "A2A agent framework", is the prose reference.
- What agent-env sends you: `src/agent_env/task_step/task_steps/deploy_agent.py` (env vars, MCP registrations,
  identity) and `prompt_agent.py` (config fields, then `message/send`).

## Rules that bite

- `prompt_agent` always sends `timeout_seconds` (600 unless the step sets one), and `agent-config` merges each post
  into the last: a field one prompt sets stays set until another sets it.
- Declare only the config fields the runtime honours, under the names `prompt_agent` sends (`model`, `system_prompt`,
  `effort`, `max_turns`, `max_thinking_tokens`, `output_format`, `timeout_seconds`, `task_id`, ...). Secrets are
  `WriteOnly`.
- `skill-config` needs both `add.inline` and `add.bundle`. Overriding an SDK-owned operation means overriding its
  whole group.
- A snapshot `save` uploads both the trajectory and the workspace object, an empty workspace if there is none;
  `load` runs on a fresh agent, so create state directories at startup.
- Bind the session before the runtime starts, as `ClaudeCodeAgent.run` does, so a live trajectory read or a snapshot
  taken mid-turn finds it.
- The SDK serves the peer tools from protocol 0.1.350. An agent that overrides `peer-agents` registers its own MCP
  server through `default_handlers.call(MCP_CONFIG_V1.add, ...)` on a `127.0.0.1` URL ending in `/mcp`.
- Turns of different contexts run concurrently, and the validator relies on it; don't serialize them.
- `install_agent` imports `AGENT_CARD` with `/app` on `sys.path` and without your entrypoint. Pass the endpoint as
  environment on the command that needs it, as `_ENDPOINT` does, not as arguments.
- Start the server from `CMD`, not `ENTRYPOINT`, on `0.0.0.0:$A2A_PORT`, as a non-root user that tolerates root, with
  everything installed at build time (`Dockerfile`, `entrypoint.sh`).
- `TaskResult.builder().failed(...)` adds no text part; add one. Use `infra_error` only when a retry could help.

## Registering and validating

In a bundle, a folder `agents/<name>/` with a Dockerfile is an agent, with optional defaults in `agent.toml`
(`[default_env_vars]`, `[metadata] default_model`, `min_disk_size_gb`). Give it a smoke task like
`tasks/claude-code-smoke.json`.

```bash
agent-env run ./my-bundle                                    # build, register under @local, run its tasks
agent-env a2a-agent put --id my-agent --dockerfile path/to/Dockerfile
agent-env a2a-agent validate --id my-agent                   # agent-env's built-in checks
agent-env a2a-agent get --id my-agent                        # the validation results
```

`validate` runs a fixed set of checks against the reference env `multi-caf64d`, which must exist in the document
store; the checks are in `src/agent_env/a2a_agent/validator.py` and
`src/agent_env/task_step/task_steps/a2a_agent_validator/`, one step each. It doesn't cover everything an agent can
advertise: `triggers/v1`, `attribution-probe/v1`, host-mode `install/v1` and custom extensions need tests of your
own.
