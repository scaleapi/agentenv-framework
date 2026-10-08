# pi A2A agent

The [pi coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) wrapped
with the `agentenv_protocol.a2a_agent` framework. This directory is the whole build context: the
protocol package comes from PyPI (`requirements.txt`), and `install.sh` installs Node, pi and the
agent's environment both for the image and for `install/v1`.

Each task runs `pi --mode json` once in `/workspace`. The prompt goes in on stdin. Images and text
files are attached as `@path`; any other file is saved to disk and its path is added to the prompt,
for pi's tools to use. The pi config dir is built fresh for each task:

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
| `snapshot/v1` | `save` uploads the context's pi session (JSONL) and, when asked, the workspace as a gzipped tar; `load` restores both and binds the session to the target context |
| `snapshot/v1` changelog | after enable, every tool result uploads `NNNNNN.tar`: the files that changed under the roots (default `/workspace` and `/app`), the deleted paths and the session so far. `apply` replays them in order, writing only under each increment's roots, and with `resume_conversation` binds the last session to the target context |
| `peer-agents/v1` | setting peers registers a loopback MCP server (`peers` at `http://127.0.0.1:$A2A_PORT/mcp`) whose `peer_list` and `peer_send_message` tools send A2A messages, one conversation per peer. A peer on the host's loopback is retried at `host.docker.internal` |
| `install/v1` | copies this directory into the task container as `/opt/pi-a2a`, runs `install.sh` (Debian or Ubuntu, as root, with network access) and starts `start.sh` |
| `triggers/v1` | SDK default |

Model requests carry `model_params`, merged into the request body through pi's `samplingParams`. agent-env
reserves some request fields, such as `user` and `metadata`, and does not forward them as
`model_params`. A deployment that needs one, for example a LiteLLM key that requires project
attribution, sets it at registration through `PI_A2A_MODEL_PARAMS`, a JSON object that config
values override:

```bash
agent-env a2a-agent put --id pi --dockerfile plugins/agents/pi/Dockerfile --context plugins/agents/pi \
  --env-var 'PI_A2A_MODEL_PARAMS={"user":"<project id>"}'
```

An installed agent is started without registration env vars, so it does not get `PI_A2A_MODEL_PARAMS`.

Usage is summed over the assistant messages. `cost_usd` is only reported when the model has `cost`
metadata, and the generated `models.json` sets none. A `stopReason` of `error` is an `infra_error`
(`pi.model_error`), `aborted` is an `agent_error`, and a timeout kills pi's process group.

```bash
docker build -t pi-a2a plugins/agents/pi
docker run -p 8000:8000 -e LITELLM_BASE_URL=... -e LITELLM_API_KEY=... pi-a2a
pytest plugins/agents/pi
```
