# Antigravity A2A agent

Google's [Antigravity SDK](https://pypi.org/project/google-antigravity/) (`google-antigravity`) wrapped with the
`agentenv_protocol.a2a_agent` framework, shipped in the `agents` bundle as
`@local/agentenv-framework/agents/antigravity`. This directory is the whole build context: the protocol package and the
SDK, whose wheel holds its harness, come from PyPI (`requirements.txt`).

Each task is one `Agent.chat()` of the SDK in the A2A server's process, in `/app`. The SDK runs its own harness,
`localharness`, a static Go binary that runs the tools and calls the model. A context's first task starts an SDK
conversation, and each later task resumes it by its id; the SDK keeps each conversation in a SQLite store under
`$ANTIGRAVITY_HOME/conversations/`. Text parts are the prompt, and file parts are written under `/app/inputs/<task>/`
and named in it; an image goes to the model inline too, as it was given, with its message in every later turn, and a PDF as its
pages' PNGs, rendered with `pdftoppm`. Turns of
different contexts run at once, a context's turns one at a time.

The SDK's `LocalOpenAIAgentConfig` takes a chat-completions `base_url` and nothing else for the endpoint, and its
harness sends no key, so each task's harness calls a model relay of the task's own on loopback (`model_relay.py`,
shared with other agents), which forwards each call to `$LITELLM_BASE_URL/v1/chat/completions` with:

- the endpoint's key from `LITELLM_API_KEY`, `[model.headers]` (`LITELLM_EXTRA_HEADERS`) and the run's `project_id` and
  `task_id` as `x-litellm-customer-id` and `x-litellm-tags`. The run's tags join an `x-litellm-tags` that
  `[model.headers]` names;
- the task's `effort` as `reasoning_effort` (`xhigh` and `max` as `high`) and its images, which the SDK has no setting
  for, recorded by the message they came with so the turns after carry them too;
- the env's time in the harness's "The current local time is" line while the agent follows an armed clock (below).

The relay counts each call from the answer's usage, every subagent's included, so the task's `Usage` is their sum; the
SDK reports none. A call answered without usage is counted in `unreported_calls`.

The rest of each task's config:

- the task's `system_prompt`, as the SDK's `system_instructions`, a section after the harness's own;
- the model: the task's, else `ANTIGRAVITY_MODEL`, else `anthropic/claude-sonnet-5-5`;
- every registered MCP server, the task's `peer_agents` tools included, as a streamable-HTTP server. Its
  `AgentEnv-Role` header is the role its registration names, unless the task has a role of its own;
- every tool call allowed (`policy.allow_all()`), autonomous behaviour, `ask_question` and `generate_image` off, and the
  SDK's own retries of a model call the endpoint answers with a transient error: six, from 5 s, doubling.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `AntigravityConfig`: `model`, `effort`, `system_prompt`, `project_id`, `task_id`, `timeout_seconds` (1800 by default; past it the task fails with `timeout`), `max_turns` (the model calls a task may make, the SDK's `max_model_calls`; a task stopped at it fails with `max_turns_exceeded`), `builtin_tools`, `web_tools` (below) |
| `mcp-config/v1` | SDK default; each server passed to the task's harness. One that fails to start fails the task with `mcp_load_failed`, an infra error |
| `skill-config/v1` | inline and bundle skills installed as `$ANTIGRAVITY_HOME/skills/<name>/`, the harness's skills path; an inline `SKILL.md` without frontmatter gets one from the skill's name and description, since the SDK lists a skill by its frontmatter alone |
| `trajectory/v1` | the SDK's streamed steps, a step's text and thought each as one record, and each tool call's result from the SDK's hooks after it, live while the task runs (format `antigravity-sdk/1`) |
| `snapshot/v1` | `save` archives the context's conversation store with its own turns' env times and images, never another context's, and `/app`, once no task runs; `load` restores the store under its own id, which the SDK holds inside it, bound to the target context, and replaces `/app`'s contents; a conversation another context of the same agent holds is refused (409), since the two would share one history. Where `tar` is GNU tar, the changelog too: an incremental archive of its roots after each task, uploaded through the namespace grant, and applied in order; an increment is a whole task's changes, since the harness has no hook this agent can hold a task at |
| `peer-agents/v1` | SDK default: the harness reaches the task's other agents through the SDK's `peer_agents` MCP tools |
| `triggers/v1` | SDK default |
| `attribution-probe/v1` | the `project_id` and `task_id` the last task sent |
| `install/v1` | `install.sh`, into a task container (as root, working in `/app`) or onto a `vm` sandbox's Linux host: uv checked against its checksum and the agent's requirements in a Python of its own, which the target's own Python is never replaced by. The SDK's wheels need glibc on x86_64 or aarch64 |

The image is `python:3.11-slim` with `pdftoppm`, git, the SDK and the agent, running as the `agent` user, its `HOME`
`/data`.

**The env's clock.** The image follows the env's clock as the other clocked agents do (the agents' README). The harness
is a static binary, which agent-env's libfaketime can't reach, so it keeps real time and its conversation store stays in
order at any env date, while the commands it runs read the env's time and go out through the egress proxy. The model
sees the harness's clock in two places, and while the clock is armed the relay rewrites both (`antigravity_date.py`).

- **The local-time line.** The relay puts the env's time, read from `date`, in each turn's "The current local time is:
  <RFC 3339>." line.
- **Command-result stamps.** It shifts the `Created At:` / `Completed At:` stamps on a command's result by the env's
  offset from real time, so the command keeps its duration.
- **A date change mid-turn.** When the env's date moves past the one the model was last told during a turn, the next
  command result gets "The date has changed. The current local time is now: …", as Claude Code tells its model.

Each rewrite is recorded beside the conversations, so a resumed conversation's earlier turns keep what they were
given. A call whose time the relay can't read is refused, so the task fails rather than tell the model the machine's
time. The harness's own file tools write real modification times, and its `schedule` timers run on real time and end
with the turn, since a turn's harness is started for it and stopped after it.

A `deploy_agent` `builtin_tools` selection picks the SDK's own tools: `{"disable": "*"}` and `{"only": [...]}` turn off
every tool but those named, a list to turn off turns off those alone; the tools of the MCP servers agent-env gives the
agent stay in every case, `finish`, which ends a task, stays on, and `ask_question` and `generate_image` stay off. The
names are `AntigravityTool`'s in `antigravity_bridge.py`, the SDK's own: `search_web` and `read_url_content` are its
web tools. A test fails when `requirements.txt`'s SDK pin moves, until `SDK_VERSION` moves with it.

A `deploy_agent` `web_tools: false` turns off the SDK's web tools, `search_web` and `read_url_content`, whatever
`builtin_tools` selects.
