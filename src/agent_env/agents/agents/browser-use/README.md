# Browser Use A2A agent

[Browser Use Pi](https://github.com/browser-use/browser-use-pi) wrapped with `agentenv_protocol.a2a_agent`, shipped in
the `agents` bundle as `@local/agentenv-framework/agents/browser-use`, with alias `browser-use`. Browser Use Pi drives
a real browser over CDP from a persistent JavaScript REPL. `package-lock.json` pins `@browser_use/pi` to 0.1.0.

Each turn runs `node agent.mjs` once. It calls the `[model]` endpoint as an OpenAI-compatible one
(`LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_EXTRA_HEADERS`), and the task env's MCP tools are the agent's tools
next to the browser, named `<server>__<tool>`. The browser is a [Browser Use Cloud](https://cloud.browser-use.com)
browser, which needs `BROWSER_USE_API_KEY`, or the Chrome DevTools endpoint `cdp_url` names. A context's turns
continue one conversation through a saved history file; each turn opens a fresh browser.

```bash
agent-env a2a-agent put --id browser-use --dockerfile src/agent_env/agents/agents/browser-use/Dockerfile \
    --default-model gpt-5.4-mini --env-var BROWSER_USE_API_KEY=...
agent-env run agents --task browser-use-smoke
```

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `model` (else `BROWSER_USE_A2A_MODEL`), `effort`, `system_prompt` (added to Browser Use Pi's instructions), `max_steps`, `cdp_url`, `timeout_seconds` |
| `mcp-config/v1` | SDK default; each server's tools are listed per turn and called over streamable HTTP |
| `trajectory/v1` | Browser Use Pi's agent events, live, format `browser-use-pi-events/1`, without the streamed `message_update` / `tool_execution_update` deltas |

A run that stops early (`max_steps`, `timeout`, `cost_limit`, ...) fails with code `browser_use.<status>` and its
last text. Skills, snapshots, peers and `install/v1` are not implemented, so the validator reports them unsupported.
