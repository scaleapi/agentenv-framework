# AGENTS.md: the shipped agents

Guidance for writing or changing an A2A agent in this bundle. The repository's `AGENTS.md` still applies; this file
only adds what is specific to agents. `README.md` here is the user-facing description of each agent.

## Layout

| Path | What it is |
|---|---|
| `agents/<name>/` | one agent: a Docker build context whose `Dockerfile` makes it an agent, deployed as `@local/agentenv-framework/agents/<name>` |
| `agents/<name>/a2a_server.py` | the module that builds the app and exports `AGENT_CARD`; `install_agent` imports it from the image |
| `agents/<name>/requirements.txt` | exactly one `agentenv-framework-protocol[agent]` line, which the release rewrites to a pin (`tst/unit/test_release_protocol_pins.py`) |
| `tasks/<name>-smoke.json` | deploy, one prompt, one `response_contains` check; `agent-env run agents --task <name>-smoke` |
| `tst/unit/agents/<name>/` | the agent's offline tests; `agent.py` there imports its modules from this folder by name |

The bundle is the `agents` entry in `[project.entry-points."agent_env.bundles"]`, so every file here ships in the
wheel.

## Where to look

Every agent is built on the SDK, `agentenv_protocol.a2a_agent`. Copy from the agent closer to your runtime.

| To see how to | Read |
|---|---|
| declare the agent, its config and extensions | `claude-code/a2a_server.py` (`ClaudeCodeConfig`, the `@a2a_agent` block), `pi/pi_agent.py` (`PiConfig`) |
| run a CLI per turn and stream its events as the trajectory | `ClaudeCodeAgent.run`, `claude_bridge.run_claude` (`event_callback=request.trajectory.append`) |
| return a trajectory once the run ends | `pi/pi_agent.py` `_result` (`native_trajectory`) |
| map the `[model]` endpoint and `LITELLM_EXTRA_HEADERS` onto a CLI | `claude_bridge.cli_env`; pi's `_models_json` for an OpenAI-compatible one |
| hand `request.mcp_servers` to the runtime | `claude_bridge.py` (`--mcp-config`), pi `_mcp_json` (`mcp.json`, headers as `${VAR}`) |
| read file parts | `kit.materialize_file_part` |
| install skills, inline and bundle | `kit.install_skill`, pi `add_inline_skill` / `add_bundle_skill` |
| keep session refs past the SDK's 1,024-context LRU | `kit.DurableSessionRefs` |
| answer errors as `{"detail": ...}` | `kit.json_error_bodies`, applied in `build_app` |
| save and restore a conversation and workspace | `ClaudeCodeAgent.save_snapshot` / `load_snapshot`, pi `save_snapshot` / `load_snapshot` |
| capture and replay a changelog | pi `enable_changelog` / `apply_changelog`, `pi/changelog.py` (`ChangelogCapture`, `apply`) |
| serve a whole context's trajectory | `ClaudeCodeAgent.get_context_trajectory` (`TRAJECTORY_V1.get.context`) |
| replace the SDK's peer tools with your own | pi `set_peers` / `list_peers`, `pi/peers.py` |
| install into a running container or a VM host | `_INSTALL_COMMANDS` / `_INSTALL_COMMANDS_HOST` and `install.sh` (Claude Code), `INSTALL_COMMANDS` and `start.sh` (pi) |
| test an agent offline | `tst/unit/agents/claude_code/` (`conftest.py`, `test_card_contract.py`), `tst/unit/agents/pi/` (`fake_pi.py`) |

Smaller single-feature agents are in `packages/agentenv-protocol/examples/`.

## The protocol

- `packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/extensions.py` defines every `urn:agentenv:*`
  extension, its endpoint, operations, request and response models, and which side owns each operation (`SDK` or
  `RUNTIME`). It is the source of truth for the wire format; read it before guessing a shape.
- `framework.py` there builds the app: card, routes, executor, default handlers. `tasks/v1.py` is `TaskRequest`,
  `TaskResult` and `TrajectoryLog`. `agentenv_protocol/transfers.py` is `upload`, `download` and `NamespaceUploader`.
- `packages/agentenv-protocol/README.md`, section "A2A agent framework", is the prose reference, including live
  trajectories, object transfer and staging.
- The control plane's side: `src/agent_env/a2a_agent/` (deploy, config negotiation, object transfer) and the steps
  in `src/agent_env/task_step/task_steps/` (`deploy_agent.py`, `prompt_agent.py`, `install_agent.py`).
- The validator: `src/agent_env/a2a_agent/validator.py` and `task_step/task_steps/a2a_agent_validator/`, one step per
  check.

A change to an extension's shape is a protocol change: make it in `packages/agentenv-protocol` with its tests, not by
working around the SDK in an agent. v1 routes are frozen; a new version gets its own path.

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
- An agent that overrides `peer-agents` registers its own MCP server through
  `default_handlers.call(MCP_CONFIG_V1.add, ...)` on a `127.0.0.1` URL ending in `/mcp`, as pi does.
- Turns of different contexts run concurrently, and the validator relies on it; don't serialize them.
- `install_agent` imports `AGENT_CARD` with `/app` on `sys.path` and without the entrypoint, and logs only each
  install template's first line; keep credentials out of templates (pi writes them through a heredoc).
- The image starts the server from `CMD`, not `ENTRYPOINT`, binds `0.0.0.0:$A2A_PORT`, runs as a non-root user that
  tolerates root, and installs everything at build time.
- `TaskResult.builder().failed(...)` adds no text part; add one. Use `infra_error` only when a retry could help.

## Checking a change

```bash
pytest tst/unit/agents/<name> -v                  # offline, with the runtime faked
agent-env run agents --task <name>-smoke          # builds the image; needs Docker and [model]
agent-env a2a-agent validate --id <name>          # the full check matrix, against the reference env
```

`agent-env a2a-agent validate` needs the reference env `multi-caf64d` in the document store. A new agent also gets a
paragraph in `README.md` here.
