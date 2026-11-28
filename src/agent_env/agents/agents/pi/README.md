# pi A2A agent

The [pi coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) wrapped
with the `agentenv_protocol.a2a_agent` framework, shipped in the `agents` bundle as
`@local/agentenv-framework/agents/pi`. This directory is the whole build context: the
protocol package comes from PyPI (`requirements.txt`), and `install.sh` installs Node, pi and the
agent's environment both for the image and for `install/v1`.

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
| `agent-config/v1` | `PiConfig`: `model`, `effort` (pi thinking level), `system_prompt`, `append_system_prompt`, `tools`, `context_window`, `max_tokens`, `model_params`, `timeout_seconds`, `web_tools` (pi's tools have no web search or fetch, so `false` changes nothing) |
| `mcp-config/v1` | SDK default; written to `mcp.json` per task |
| `skill-config/v1` | inline and bundle skills installed as `SKILL.md` directories, passed with `--skill` |
| `trajectory/v1` | the task's pi JSON events, appended as pi prints them and readable while the task runs, with `message_update` / `tool_execution_update` deltas dropped (format `pi-json-events/v1`); and `atif/1` when a run prescribes it, built from them by `pi_atif.py` (see below) |
| `snapshot/v1` | `save` uploads the context's pi session (JSONL) and, when asked, the workspace as a gzipped tar; `load` replaces the workspace's contents with the saved ones (extracted aside first, so an archive that fails to extract changes nothing), restores the session and binds it to the target context |
| `snapshot/v1` changelog | after enable, every tool result builds `NNNNNN.tar`: the files that changed under the roots, the deleted paths and the session so far. Increments upload in order; a failed one is kept, with those after it, and retried at the next tool result and when the task ends, which fails as `pi.changelog_incomplete` if any is still not uploaded. Past 256 MiB held by failed uploads, they are dropped and capture stops, so every later task fails that way until the changelog is enabled again. `apply` replays them in order, writing only under the replaying agent's own roots (symlinks resolved), and with `resume_conversation` binds the last session to the target context |
| `peer-agents/v1` | setting peers registers a loopback MCP server (`peers` at `http://127.0.0.1:$A2A_PORT/mcp`) whose `peer_list` and `peer_send_message` tools send A2A messages, one conversation per A2A context and peer. It answers only loopback clients that send a running task's token, which each task gets in its environment and pi sends as a header, so a task speaks only in its own context's conversations. A peer on the host's loopback is retried at `host.docker.internal` |
| `install/v1` | copies this directory into the task container as `/opt/pi-a2a`, runs `install.sh` (Debian or Ubuntu, as root, with network access), writes the LiteLLM credentials to a 0600 `agent.env` through a heredoc and starts `start.sh`, which sources it. The credentials stay off each command's first line, the only one the installer logs, but core's install contract substitutes them into the script text, so a sandbox that logs whole scripts records them |
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

Usage is summed over the assistant messages and each compaction's summary call, the prompt cache counted in the
input (`pi_atif.counted`, which the atif/1 copy's steps count by too). `cost_usd` is only reported when the model has
`cost` metadata, and the generated `models.json` sets none. A `stopReason` of `error` is an `infra_error`
(`pi.model_error`), `aborted` is an `agent_error`, and a timeout kills pi's process group. A cancelled or timed-out
task keeps the events pi printed before it stopped.

The atif/1 copy (`pi_atif.PiAtif`) has a step per model call, from each assistant `message_end`: its model (the one
the endpoint answered as), the time pi sent the request, its usage, its thinking as reasoning, its text and its tool
calls, each with the `toolResult` message after it, images as media files. The tools on the agent block are those pi's
system messages declare. A call pi ended with an error and no usage is one the endpoint refused, a step with no model
call holding the error, and each retry pi starts after one is a further attempt; a response that ended in an error or
was aborted has its calls left out, since pi runs none of them. A compaction is its summary call's step, timed by when the bridge saw it start. While
a task builds the copy, the bridge feeds it two aux records (`bridge`): each compaction's start, and, when a new copy
starts on a session pi resumes (after the agent restarts, say), the session's tools, which pi declares only as they
change.

```bash
docker build -t pi-a2a src/agent_env/agents/agents/pi
docker run -p 8000:8000 -e LITELLM_BASE_URL=... -e LITELLM_API_KEY=... pi-a2a
pytest tst/unit/agents/pi
```
