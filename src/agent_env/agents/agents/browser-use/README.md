# Browser Use A2A agent

[browser-use](https://github.com/browser-use/browser-use) 1.0 wrapped with `agentenv_protocol.a2a_agent`, shipped in
the `agents` bundle as `@local/agentenv-framework/agents/browser-use`, with alias `browser-use`. browser-use drives a
real browser over CDP from a persistent Python REPL. `browser-use-requirements.txt` pins it to 1.0.3, which pins every
dependency of its own exactly, `mcp` 2 among them, so the image installs it in an environment of its own,
`/opt/browser-use`, beside the A2A server's.

Each turn runs `agent.py` once in that environment.
- **The model** is the prompt's, on the `[model]` endpoint (`LITELLM_BASE_URL`, `LITELLM_API_KEY`,
  `LITELLM_EXTRA_HEADERS`). A Claude model goes through the endpoint's Anthropic Messages route, as browser-use runs
  Claude: `effort` is its adaptive thinking's, medium unset, browser-use's default, and its prompt is cached. Its
  profile comes from the bare model name, since a provider prefix (`anthropic/claude-...`) hides the model's adaptive
  thinking from Pydantic AI. Any other model goes through the OpenAI-compatible route, with `effort` sent as
  `reasoning_effort` and none sent unset.
- **The browser** is the one the task gives it through `cdp-config/v1`, as the `browser_connect_cdp` step does: the
  endpoint `browser` names, else the turn's only one. It connects by the endpoint's `ws_url`, or by `http_url` with
  `connect_by = "http"`.
- **Its tab:** when the browser has one tab open, as a browser computer does, the agent drives that tab, which is the
  computer's screen, so its work shows there and in a recording, and stays there when the turn ends. With several,
  browser-use opens a tab of its own. browser-use closes a tab of its own when it stops, so its last page would be out
  of reach after the turn. Either way the browser keeps running.
- **With no endpoint,** or several and none named, the turn fails with `cdp.no_endpoint` or `cdp.ambiguous`, an
  `infra_error`. It never drives a browser of its own.
- **An endpoint with headers** fails the turn with `browser_use.cdp_headers`, an `infra_error`, since browser-use
  connects by the address alone.
- **The task env's MCP tools** are the agent's tools beside its REPL, named `<server>__<tool>`.
- **Across turns:** a context's turns continue one conversation through a saved history file, and each turn connects
  afresh.
- **Limits:** a turn makes at most `max_steps` model requests (browser-use's 100 unset), its delivery included, and
  browser-use's run ends `DELIVERY_SECONDS` before the turn's `timeout_seconds`, which leaves it time to deliver.
- **Telemetry:** off. The image and each turn set `ANONYMIZED_TELEMETRY=false` and `BH_TELEMETRY=0`, so neither
  browser-use nor its browser daemon reports anything.

It takes its model from the prompt, so a run needs `[model] default` (or the `agent` role) set. The smoke task deploys
`chromium` from the `browsers` bundle beside it, connects the agent to it, and has it read a synthetic page's title:

```bash
agent-env run agents --task browser-use-smoke
```

| Extension | Implementation |
|---|---|
| `agent-config/v1` | `model` (else `BROWSER_USE_A2A_MODEL`), `effort` (`low`, `medium` or `high`), `mode` (`default` or `ultrafast`, browser-use's accessibility-tree helpers), `system_prompt` (added to browser-use's instructions), `max_steps`, `browser`, `connect_by` (`ws` or `http`), `timeout_seconds` |
| `cdp-config/v1` | SDK default; each turn reads its endpoint with `request.cdp_endpoint(browser)` |
| `mcp-config/v1` | SDK default; each server's tools are listed per turn and called over streamable HTTP |
| `trajectory/v1` | browser-use's messages, as Pydantic AI writes them, each turn's when it ends, format `browser-use-messages/1` |

A turn's usage reports browser-use's tokens, cached ones among its input, with its model requests, cache reads and
writes, and its cost when browser-use knows the model's price. A run that ends without delivering (`max_steps`,
`timeout`, `error`) fails with code `browser_use.<status>` and its error. One that reaches `max_steps` or its timeout
gets a forced delivery turn first, as browser-use does, and completes with that answer. Skills, snapshots, peers and
`install/v1` are not implemented, so the validator reports them unsupported.
