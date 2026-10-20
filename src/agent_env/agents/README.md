Agents that come with agent-env, each an agent folder a bundle builds and deploys like its own.

`agents/claude-code/` runs Anthropic's [Claude Code](https://code.claude.com/docs/en/headless) CLI behind an A2A
server: the prompt goes to `claude -p`, the task env's MCP servers to `--mcp-config`, and its stream-json output is the
turn's trajectory. It's the default agent, so a `deploy_agent` or an agent judge that names no agent deploys it. Name it
from another bundle as `@local/agentenv-framework/agents/claude-code`.

It needs Docker, and a model endpoint serving Anthropic's Messages API, `<base>/v1/messages`, which a LiteLLM gateway
does: the `[model]` base URL and key reach it as `ANTHROPIC_BASE_URL` and a bearer token, and `[model.headers]` are sent
on every model call. The `LITELLM_*` variables stay in its environment too, so a script the model runs can call the
endpoint. Building the image downloads Claude Code, which is Anthropic's software under its
[Commercial Terms](https://www.anthropic.com/legal/commercial-terms); the image is built where it runs and never
published. The CLI's version is the Dockerfile's `CLAUDE_CODE_VERSION`. The image gives the scripts the model runs
`node`, without npm or npx, and the Python packages `model-requirements.txt` pins.

`install_agent` installs it too, into a container a task already runs or onto a `vm` sandbox's Linux host, with
`install.sh` from its folder: Claude Code's native build for the target's CPU and libc, uv, and the agent's requirements
in a Python of its own, so the target's Python is left alone. The CLI runs its commands in bash or zsh, so a target with
neither, Alpine's for one, gets bash from its own package manager. The agent works in the container's working directory,
or in `/app` on the host.

`agents/pi/` runs the [pi coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) behind
the protocol's A2A agent framework: each turn is one `pi --mode json` run, the task env's MCP servers go to its
`mcp.json`, and its JSON events are the turn's trajectory. Name it as `@local/agentenv-framework/agents/pi`. It calls
the `[model]` endpoint as an OpenAI-compatible one and takes its model from the prompt, so its smoke task needs
`[model] default` (or the `agent` role) set. It declares `install/v1` too, for Debian or Ubuntu targets.
Its folder's `README.md` covers the extensions it implements.

`agent-env run agents` runs every agent's smoke task, building each image; `--task claude-code-smoke` runs Claude
Code's alone, `--task pi-smoke` pi's.
