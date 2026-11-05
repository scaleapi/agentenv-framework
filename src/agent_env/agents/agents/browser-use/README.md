# Browser Use A2A agent

[Browser Use Pi](https://github.com/browser-use/browser-use-pi) wrapped with `agentenv_protocol.a2a_agent`, shipped in
the `agents` bundle as `@local/agentenv-framework/agents/browser-use`, with alias `browser-use`. Browser Use Pi drives
a real browser over CDP from a persistent JavaScript REPL. `package-lock.json` pins `@browser_use/pi` to 0.1.0, and
`package.json` overrides the `undici` and `brace-expansion` it pins with their patched releases.

Each turn runs `node agent.mjs` once. It calls the `[model]` endpoint as an OpenAI-compatible one
(`LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_EXTRA_HEADERS`), and the task env's MCP tools are the agent's tools
next to the browser, named `<server>__<tool>`.
- **The browser** is the one the task gives it through `cdp-config/v1`, as the `browser_connect_cdp` step does: the endpoint `browser` names, else the turn's only one.
- **Connecting:** it connects by the endpoint's `ws_url`, or by `http_url` with `connect_by = "http"`.
- **Its tab:** when the browser has one tab open, as a browser computer does, the agent drives that tab, which is the computer's screen, so its work shows there and in a recording. With several, Browser Use opens a tab of its own.
- **With no endpoint,** or several and none named, the turn fails with `cdp.no_endpoint` or `cdp.ambiguous`, an `infra_error`. It never drives a browser of its own.
- **An endpoint with headers** fails the turn with `browser_use.cdp_headers`, an `infra_error`, since Browser Use Pi connects by the address alone.
- **Across turns:** a context's turns continue one conversation through a saved history file, and each turn connects afresh.

It takes its model from the prompt, so a run needs `[model] default` (or the `agent` role) set. The smoke task deploys
`chromium` from the `browsers` bundle beside it, connects the agent to it, and has it read a synthetic page's title:

```bash
agent-env run agents --task browser-use-smoke
```

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `model` (else `BROWSER_USE_A2A_MODEL`), `effort`, `system_prompt` (added to Browser Use Pi's instructions), `max_steps`, `browser`, `connect_by` (`ws` or `http`), `timeout_seconds` |
| `cdp-config/v1` | SDK default; each turn reads its endpoint with `request.cdp_endpoint(browser)` |
| `mcp-config/v1` | SDK default; each server's tools are listed per turn and called over streamable HTTP |
| `trajectory/v1` | Browser Use Pi's agent events, live, format `browser-use-pi-events/1`, without the streamed `message_update` / `tool_execution_update` deltas |

A run that stops early (`max_steps`, `timeout`, `cost_limit`, ...) fails with code `browser_use.<status>` and its
last text. Skills, snapshots, peers and `install/v1` are not implemented, so the validator reports them unsupported.
