Agents that come with agent-env, each an agent folder a bundle builds and deploys like its own.

`agents/claude-code/` runs Anthropic's [Claude Code](https://code.claude.com/docs/en/headless) CLI behind an A2A
server: the prompt goes to `claude -p`, the task env's MCP servers to `--mcp-config`, and its stream-json output is the
turn's trajectory. It's the default agent, so a `deploy_agent` or an agent judge that names no agent deploys it. Name it
from another bundle as `claude-code`, the alias `aliases.toml` gives it, or by its id,
`@local/agentenv-framework/agents/claude-code`; an agent of your own with the id `claude-code`, in your bundle or your
store, is the one `claude-code` names instead. `agent-env a2a-agent list` lists the agents a task can name, with
their aliases, and `agent-env run <bundle> --agent pi` runs a bundle's solver as another of them for one run, its
tasks left as written.

It needs Docker, and a model endpoint serving Anthropic's Messages API, `<base>/v1/messages`, which a LiteLLM gateway
does: the `[model]` base URL and key reach it as `ANTHROPIC_BASE_URL` and a bearer token, and `[model.headers]` are sent
on every model call, with the run's `project_id` and `task_id` as `x-litellm-customer-id` and `x-litellm-tags` (the
run's tags join an `x-litellm-tags` that `[model.headers]` names, whose own keys, such as `taskId`, win). The
`LITELLM_*` variables stay in its environment too, so a script the model runs can call the endpoint. The image holds
Claude Code, which is Anthropic's software under its
[Commercial Terms](https://www.anthropic.com/legal/commercial-terms); a release also publishes the image, CLI included,
at `ghcr.io/scaleapi/agentenv-claude-code:src-<hex>`, built from this folder, and again only when
the folder changes. A run that deploys the agent registers that image when it's published for the sources installed: each deploy
pulls it by its digest and builds it from this folder only when it can't, and a provider that runs images by name can
run it. The CLI's version is the Dockerfile's
`CLAUDE_CODE_VERSION`. The image gives the scripts the model runs
`node`, without npm or npx, and the Python packages `model-requirements.txt` pins.

`install_agent` installs it too, into a container a task already runs or onto a `vm` sandbox's Linux host, with
`install.sh` from its folder: Claude Code's native build for the target's CPU and libc, uv, and the agent's requirements
in a Python of its own, so the target's Python is left alone. The CLI runs its commands in bash or zsh, so a target with
neither, Alpine's for one, gets bash from its own package manager. The agent works in the container's working directory,
or in `/app` on the host.

`agents/pi/` runs the [pi coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) behind
the protocol's A2A agent framework: each turn is one `pi --mode json` run, the task env's MCP servers go to its
`mcp.json`, and its JSON events are the turn's trajectory. Name it as `pi`, its alias, or by its id,
`@local/agentenv-framework/agents/pi`. It calls
the `[model]` endpoint as an OpenAI-compatible one and takes its model from the prompt, so its smoke task needs
`[model] default` (or the `agent` role) set. It declares `install/v1` too, for Debian or Ubuntu targets.
Its folder's `README.md` covers the extensions it implements.

`agents/codex/` runs OpenAI's [Codex CLI](https://github.com/openai/codex) behind the protocol's A2A agent framework:
each turn is one `codex exec --json` run, resumed in the context's Codex thread, the task env's MCP servers go to its
`config.toml` through `mcp-remote`, and its JSON events are the turn's trajectory. Name it as `codex`, its alias, or by
its id, `@local/agentenv-framework/agents/codex`. It calls the `[model]` endpoint's OpenAI-compatible Responses API,
`<base>/openai/v1/responses`, which a LiteLLM gateway serves, and takes its model from the prompt, so its smoke task
needs `[model] default` (or the `agent` role) set. It declares `install/v1` too, into a container or onto a `vm`
sandbox's Linux host, with glibc. Its folder's `README.md` covers the extensions it implements.

`agents/gemini/` runs Google's [Gemini CLI](https://github.com/google-gemini/gemini-cli) behind the protocol's A2A
agent framework: each turn is one `gemini -p` run, resumed in the context's Gemini session, the task env's MCP servers
go to the run's own settings, and its stream-json events are the turn's trajectory. Name it as `gemini`, its alias, or
by its id, `@local/agentenv-framework/agents/gemini`. It calls the `[model]` endpoint's Gemini passthrough,
`<base>/gemini`, which a LiteLLM gateway serves, through a proxy in the agent that adds the run's attribution; a model
named with a provider prefix, such as `vertex_ai/...`, goes to `<base>/v1/chat/completions` instead. It takes its
model from the prompt, so its smoke task needs `[model] default` (or the `agent` role) set. It declares `install/v1`
too, into a container or onto a `vm` sandbox's Linux host, with glibc. Its folder's `README.md` covers the extensions
it implements.

`agents/opencode/` runs the [OpenCode](https://github.com/anomalyco/opencode) CLI behind the protocol's A2A agent
framework: each turn is one `opencode run --format json` run, resumed in the context's OpenCode session, the task env's
MCP servers go to the run's own inline config, and its JSON events are the turn's trajectory. Name it as `opencode`, its
alias, or by its id, `@local/agentenv-framework/agents/opencode`. It calls the `[model]` endpoint as an
OpenAI-compatible one, `<base>/v1/chat/completions`, and takes its model from the prompt, else
`anthropic/claude-sonnet-4-6`. It declares `install/v1` too, into a container or onto a `vm` sandbox's Linux host,
with glibc. Its folder's `README.md` covers the extensions it implements.

`agents/openhands/` runs the [OpenHands](https://github.com/OpenHands/software-agent-sdk) agent behind the
protocol's A2A agent framework, driving the OpenHands SDK in process rather than a CLI: each turn is one run of a
persisted SDK conversation, the task env's MCP servers go to its `mcp_config`, and its SDK events are the turn's
trajectory. Name it as `openhands`, its alias, or by its id, `@local/agentenv-framework/agents/openhands`. It
calls the `[model]` endpoint through LiteLLM and takes its model from the prompt, so its smoke task needs
`[model] default` (or the `agent` role) set. It declares `install/v1` too, for Debian or Ubuntu targets. Its folder's
`README.md` covers the extensions it implements.

A release publishes Codex's, Gemini's, OpenCode's, pi's and OpenHands' images too, at
`ghcr.io/scaleapi/agentenv-codex:src-<hex>`, `ghcr.io/scaleapi/agentenv-gemini:src-<hex>`,
`ghcr.io/scaleapi/agentenv-opencode:src-<hex>`, `ghcr.io/scaleapi/agentenv-pi:src-<hex>` and
`ghcr.io/scaleapi/agentenv-openhands:src-<hex>`, each built again only when its folder changes, and a run deploys
them as it does Claude Code's: it pulls the image
published for the sources installed, by its digest, and builds it from the agent's folder only when it can't.

`agents/proxy/` is the Episode API's, not an agent to pick for a task. It serves the protocol's `ProxyAgent`,
which has no model: an episode deploys it in the place of the agent its caller plays, and the caller's model acts
through it over `urn:agentenv:proxy/v1`. Env tool calls and `bash` run in it, where the agent would work, and it
records the turn. Its image works in `/app` as user `agent`. On a `vm` sandbox's host, `install.sh` installs it
where the agent it replaces declares it works (`workdir_host` and `user_host` in that agent's `install/v1`), with
the host's own Python in a venv; into a task's container, it installs as root in the container's working directory,
as an agent installed there works, with a private Python of uv's. Its `agent.toml` sets `takes_credentials = false`, so it's given no model key or
store credentials. A release publishes its image at `ghcr.io/scaleapi/agentenv-proxy:src-<hex>`, as it does the
others'. It has no alias and no smoke task: with nobody driving it, a prompt only waits out its deadline.

`agent-env run agents` runs every agent's smoke task, pulling or building each image; `--task claude-code-smoke` runs
Claude Code's alone, `--task codex-smoke` Codex's, `--task gemini-smoke` Gemini's, `--task opencode-smoke` OpenCode's,
`--task pi-smoke` pi's, `--task openhands-smoke` OpenHands'. 
