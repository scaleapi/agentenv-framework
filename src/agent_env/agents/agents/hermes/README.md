# Hermes Agent A2A agent

Nous Research's [Hermes Agent](https://github.com/NousResearch/hermes-agent) wrapped with the
`agentenv_protocol.a2a_agent` framework, shipped in the `agents` bundle as `@local/agentenv-framework/agents/hermes`.
This directory is the whole build context: the protocol package comes from PyPI (`requirements.txt`), and Hermes from
its own installer at a pinned release, checked against its checksum, in the `Dockerfile` (`HERMES_COMMIT`, the
`v0.22.5` release).

Each context is a Hermes profile of its own, `$HERMES_ROOT/profiles/ctx-<hash>/`, with its own config, sessions and
memory. Each task runs `hermes agentenv turn` once in `/app`, a command the agent's Hermes plugin, `agentenv_plugin/`,
adds: it runs the turn in its own process, as `hermes -z` does, rather than handing it to a Hermes gateway, so the turn
ends with that process and a timeout or cancel that kills it, and everything it started, stops the turn. A context's
first task starts a Hermes session, and each later task resumes it with `--resume <session_id>`. Text parts are the
prompt, and file parts are written under `/app/inputs/<task>/` and named in it, for Hermes' file and vision tools; a
PDF goes with its pages' PNGs, rendered with `pdftoppm`. Turns of different contexts run at once, as many as memory fits
at about 400 MB each past the first 1 GB (`HERMES_MAX_CONCURRENT_TURNS` sets the number), and a context's turns one at
a time; a task's files are downloaded and rendered only once it has a turn's slot, so they count against that number.

The profile's `config.yaml` is written before each turn:

- the model endpoint, `$LITELLM_BASE_URL/v1`, as a custom OpenAI-compatible provider with its key from
  `LITELLM_API_KEY`, `[model.headers]` (`LITELLM_EXTRA_HEADERS`) and the run's `project_id` and `task_id` as
  `x-litellm-customer-id` and `x-litellm-tags` on every model call, Hermes' auxiliary calls (titles, compression,
  vision) included. The run's tags join an `x-litellm-tags` that `[model.headers]` names, whose own keys, such as
  `taskId`, win;
- the model: the task's, else `HERMES_MODEL`, else `anthropic/claude-sonnet-4-6`;
- every registered MCP server, the task's `peer_agents` tools included, as a remote server. Its `AgentEnv-Role` header
  is the role its registration names, unless the task has a role of its own;
- the skills agent-env installed, as an external skills folder, and the `agentenv` plugin enabled.

The task's `system_prompt` is the profile's `SOUL.md`, the slot Hermes' own identity fills, so it reaches the model as
the system prompt; a session keeps the system prompt it started with, as Hermes does. A turn with no system prompt gets
Hermes' own back. Hermes runs non-interactively: dangerous-command approvals are bypassed, and its `clarify` tool tells
the model no one is there to answer.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `HermesConfig`: `model`, `effort` (Hermes' `--reasoning`), `system_prompt`, `output_format` (a JSON schema the answer is checked against), `project_id`, `task_id`, `timeout_seconds` (1800 by default) |
| `mcp-config/v1` | SDK default; each server written to the turn's profile config |
| `skill-config/v1` | inline and bundle skills installed under `$HERMES_ROOT/agentenv-skills/<name>/`, which every profile reads as `skills.external_dirs` |
| `trajectory/v1` | the turn's Hermes plugin hook events (`on_session_start`, `pre_tool_call`, `post_tool_call`, `post_api_request` with each model response, its tool calls and usage, `api_request_error`, `post_llm_call`, `on_session_end`), live while it runs (format `hermes-hook-events/1`) |
| `snapshot/v1` | `save` archives the context's profile (its sessions database, memory and `SOUL.md`; not the config, caches or logs Hermes rebuilds) with its session id, and `/app`, once no task runs; `load` restores the profile into the target context and replaces `/app`'s contents, both or neither, so a failed load leaves the context's profile, session and workspace as they were. No changelog |
| `peer-agents/v1` | SDK default: Hermes reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `attribution-probe/v1` | the `project_id` and `task_id` the last turn sent |

A turn's usage counts every model call the turn made: its main calls and its auxiliary ones (title generation,
compression), as Hermes' usage file reports them, and a `delegate_task` child's, from their `post_api_request` events,
which the file leaves out. A turn killed before it wrote the file keeps the calls its events counted, marked as a lower
bound. Hermes' estimated cost is kept only where it priced the calls. The profile turns Hermes' background review off:
its fork outlives the turn's process, which kills it before it reports what the proxy billed. The image is
`python:3.11-slim` with Hermes, `pdftoppm` and the agent. It starts as root, which runs the env clock's follower and
egress proxy under supervisord (`start-agent`, `root.conf`), and the A2A server and Hermes run as the `agent` user, on
Modal too; started as another user, it runs the A2A server alone as that user. A server that exits, for any reason,
stops supervisord and the container with it, as when the server was the container's process, so the run fails at
once; the follower and the proxy restart in place. Hermes' runtime (its own Python, Node and ripgrep) and the profiles
are under `/data/.hermes`, whatever the uid. It has no `install/v1`.

## The env's clock

The image follows a task env's clock as the agents README describes (`../../README.md`), with one difference: Hermes
itself stays on real time and the commands it runs follow the env's clock.

- **Off by default.** Without `agent_clock` on `sync_env_clock`, a turn runs as it would without the clock:
  `hermes agentenv turn` with the same flags, the server's environment and a profile config with nothing of the
  clock's, on real time, with no proxy variables. The server's environment is supervisord's, which adds its
  `SUPERVISOR_*` variables, `USER`, `LOGNAME` and `AGENTENV_EGRESS_PROXY`, and points TLS trust at agent-env's bundles
  (`root.conf`).
- **Opted in.** While the agent follows an armed clock, the bridge starts each turn with the command Hermes' launcher
  runs, Hermes' own Python, rather than the launcher, a shell script, so the real-time marker reaches Hermes and not the
  shell, and passes `--env-clock` (`hermes_bridge.cli_command`). Every command Hermes starts, the model's `date`,
  shell and `execute_code` scripts among them, reads the env's time, and goes out through the egress proxy with Hermes,
  as the shared rule (`agent_egress.egress_env`) has it for the model endpoint and every MCP server. `execute_code`
  hands its scripts only a few variables of its own choosing, so the turn's config names the proxy's, its CA's and the
  task's zone (`TZ`) for it (`terminal.env_passthrough`).
- **Why Hermes stays on real time.** Hermes compares its clock with times it didn't make on that clock in many places:
  its scratch folder (`TMPDIR`), pruned at each start of entries idle for 24 hours and the processes running in them;
  its spill files and large tool results, deleted past 7 days and `sessions.retention_days`; its checkpoints, past
  their retention; the turn's lease, which stops the turn when the clock jumps past the lease's deadline or runs a few
  times fast; and the stream stall check, which drops a model stream the clock jumps past. Each runs in Hermes' own
  process against real file times or real elapsed time, so on real time a conversation keeps its history and files at
  any date, jump or rate the env's clock takes.
- **The date the model sees.** With `--env-clock`, the plugin has Hermes read the date it shows the model, its
  `Conversation started` line (and `Today's date` after a rebuild) and the date its compaction summary carries, from a
  `date` it runs, which follows the env's clock; the session ids it mints, a compaction's new one included, whose stamp
  gives the start date, are stamped from it too. Hermes writes that line when a session starts and keeps it, as it does
  on any clock, so a session started before the clock was armed keeps the real date it started on until Hermes rebuilds
  its prompt. Other times Hermes reports itself, such as the dates `session_search` lists, are real, as files'
  modification times are.
- **A clock armed mid-turn.** The command and environment are chosen as a turn starts. A turn already running when the
  clock is armed was started as an unarmed turn is, through the launcher with no marker, so its Hermes follows the
  env's clock from then on, without the proxy. Its own model and MCP calls then check certificates on the env's
  clock: Hermes' Python carries its own OpenSSL, which agent-env's TLS shim doesn't reach, so at a date outside the
  endpoint's certificate they fail (`APIConnectionError: Connection error.`), and its turn lease or stall check may
  stop it too (`Session turn lease ...`). The turn ends with that error, the conversation keeps its history, and the
  next turn is armed.
