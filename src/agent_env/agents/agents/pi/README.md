# pi A2A agent

The [pi coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) wrapped
with the `agentenv_protocol.a2a_agent` framework, shipped in the `agents` bundle as
`@local/agentenv-framework/agents/pi`. Its build context is this directory and what its `shared_sources.toml`
takes from agent-env's own sources: the clock library (`clock/`), the egress proxy (`egress/`) and the egress rule
every clocked agent shares (`agent_clock/`). The protocol package comes from PyPI (`requirements.txt`), and
`install.sh` installs Node, pinned by version and checksum, pi and the agent's environment, both for the image and for
`install/v1`.

Each task runs `pi --mode json` once in its workspace: `PI_A2A_WORKSPACE`, default `/workspace`, and the task
container's working directory when it's installed there. The prompt goes in on stdin. Images and text
files are attached as `@path`; any other file is saved to disk and its path is added to the prompt,
for pi's tools to use. A file fetched from a URL may be at most 256 MiB. The pi config dir is built fresh for each task:

- `models.json` points provider `agentenv` at `LITELLM_BASE_URL`. The key stays a `${LITELLM_API_KEY}`
  reference, so it is never written to disk.
- `mcp.json` holds the registered MCP servers with `exposure: "direct"`. Their header values are
  passed as environment variables, not written to the file.

A context's id becomes its pi session id (`--session-id`), so follow-up tasks resume the same session.
Sessions live in `~/.pi-a2a/sessions` and skills in `~/.pi-a2a/skills/<name>`.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `PiConfig`: `model`, `effort` (pi thinking level), `system_prompt`, `append_system_prompt`, `tools`, `context_window`, `max_tokens`, `model_params`, `timeout_seconds` |
| `mcp-config/v1` | SDK default; written to `mcp.json` per task |
| `skill-config/v1` | inline and bundle skills installed as `SKILL.md` directories, passed with `--skill` |
| `trajectory/v1` | the task's pi JSON events, with `message_update` / `tool_execution_update` deltas dropped (format `pi-json-events/v1`) |
| `snapshot/v1` | `save` uploads the context's pi session (JSONL) and, when asked, the workspace as a gzipped tar; `load` replaces the workspace's contents with the saved ones (extracted aside first, so an archive that fails to extract changes nothing), restores the session and binds it to the target context |
| `snapshot/v1` changelog | after enable, every tool result builds `NNNNNN.tar`: the files that changed under the roots, the deleted paths and the session so far. Increments upload in order; a failed one is kept, with those after it, and retried at the next tool result and when the task ends, which fails as `pi.changelog_incomplete` if any is still not uploaded. Past 256 MiB held by failed uploads, they are dropped and capture stops, so every later task fails that way until the changelog is enabled again. `apply` replays them in order, writing only under the replaying agent's own roots (symlinks resolved), and with `resume_conversation` binds the last session to the target context |
| `peer-agents/v1` | setting peers registers a loopback MCP server (`peers` at `http://127.0.0.1:$A2A_PORT/mcp`) whose `peer_list` and `peer_send_message` tools send A2A messages, one conversation per A2A context and peer. It answers only loopback clients that send a running task's token, which each task gets in its environment and pi sends as a header, so a task speaks only in its own context's conversations. A peer on the host's loopback is retried at `host.docker.internal` |
| `install/v1` | unpacks the build context into the task container as `/opt/pi-a2a`, runs `install.sh` (Debian or Ubuntu, as root, with network access), which also copies the shared egress rule and the clock page's format beside the bridge, as the image does, writes the LiteLLM credentials to a 0600 `agent.env` through a heredoc and starts `start.sh`, which sources it. An installed agent doesn't follow the env's clock: the target runs neither the follower nor agent-env's libfaketime. The credentials stay off each command's first line, the only one the installer logs, but core's install contract substitutes them into the script text, so a sandbox that logs whole scripts records them |
| `triggers/v1` | SDK default |

Model requests carry `model_params`, merged into the request body through pi's `samplingParams`. agent-env
reserves some request fields, such as `user` and `metadata`, and does not forward them as
`model_params`. A deployment that needs one, for example a LiteLLM key that requires project
attribution, sets it at registration through `PI_A2A_MODEL_PARAMS`, a JSON object that config
values override. Two more registration variables: `PI_A2A_MODEL`, the model when agent-config sets
none (as for a peer that other agents message directly), and `PI_A2A_CHANGELOG_ROOTS`, the
colon-separated changelog roots (default the workspace and `/app`). Roots an enable request names must lie
within them, replay writes only under them, and a replaying agent must share them with the capturing
one. While a changelog is enabled, tasks run one at a time:

```bash
agent-env a2a-agent put --id pi --dockerfile src/agent_env/agents/agents/pi/Dockerfile \
  --context src/agent_env/agents/agents/pi \
  --env-var 'PI_A2A_MODEL_PARAMS={"user":"<project id>"}'
```

An installed agent is started without registration env vars, so it does not get `PI_A2A_MODEL_PARAMS`.

Snapshot `load` and changelog `apply` answer 409 while a task is running, and tasks and snapshot saves wait while either restores the workspace.

## The env's clock

The image can follow a task env's clock as the other clocked agent images do (`src/agent_env/agents/README.md`):
root's supervisord (`start-agent`, `root.conf`) runs the clock follower and the egress proxy, each as its own user,
and the A2A server as `agent` with the real-time marker, and agent-env's libfaketime is preloaded for `agent`.

- **Off by default.** Without `agent_clock` on `sync_env_clock`, pi and every command it runs keep real time and go
  out directly, and the model reads the env's time from the gateway's `get_time`.
- **Opted in, the commands follow.** Once the agent follows an armed clock, every command pi runs for the model reads
  the env's time: `date`, `datetime.now()` and the timestamps its programs write. Files keep real modification times,
  as libfaketime is built without faking `stat`.
- **pi itself stays on real time.** The bridge starts pi with the real-time marker, which libfaketime takes out of
  pi's environment, so what pi starts follows the clock (`cli_env`). It hands pi's script to `node` itself
  (`PI_COMMAND`): run through its `#!/usr/bin/env node`, `env` would take the marker and start node without it. pi
  can't follow the clock: before 1970 its compaction throws, as it makes a UUIDv7 from its clock, and its file locks
  compare its clock with their modification times.
  pi 1.1.0's own prompt carries no date (pi dropped it upstream), so nothing pi shows the model gives another time
  than the env's; its trajectory's timestamps and its timings are real time.
- **Through the egress proxy.** While the clock is armed, pi and its commands go out by the rule every clocked agent
  shares (`agent_egress.egress_env`): every TLS host through the egress proxy, except loopback (the peers MCP server),
  `host.docker.internal`, the hosts `NO_PROXY` names and hosts reached by plain HTTP alone. pi honours `HTTP_PROXY`,
  `HTTPS_PROXY` and `NO_PROXY` itself, its model and MCP calls alike; Node programs the model runs do through
  `NODE_USE_ENV_PROXY`, which Node honours from 22.21.
- **The changelog needs no change.** Capture compares each file's modification time, size and mode with the previous
  scan's, in the server on real time, and never with a clock.

Started as a user other than root, the image runs the A2A server alone, on real time, as before.

## Usage, errors and building

Usage is summed over the assistant messages. `cost_usd` is only reported when the model has `cost`
metadata, and the generated `models.json` sets none. A `stopReason` of `error` is an `infra_error`
(`pi.model_error`), `aborted` is an `agent_error`, and a timeout kills pi's process group.

`agent-env a2a-agent put`, as above, builds the image from its build context, shared sources included. A plain
`docker build` of this directory lacks them; to build the image alone:

```bash
python -c 'from pathlib import Path; from agent_env.utils.docker.build import build_image
d = Path("src/agent_env/agents/agents/pi"); build_image(d / "Dockerfile", d, "pi-a2a", platform=None)'
docker run -p 8000:8000 -e LITELLM_BASE_URL=... -e LITELLM_API_KEY=... pi-a2a
pytest tst/unit/agents/pi
```
