# AgentEnv Explorer (UI)

The web UI for the local AgentEnv control plane that `agent-env up` starts: browse environments,
universes, tasks and agents, start runs, and inspect their results, trajectories and triggers.
The user guide is at [www.agentenvframework.com/docs](https://www.agentenvframework.com/docs).

## Develop

With `agent-env up` running, start the dev server from this folder:

```bash
npm ci
npm run dev        # http://localhost:3000
```

The dev server proxies `/api/v1`, `/health` and `/openapi.json` to `http://127.0.0.1:8234`, the
explorer's default address. To point it somewhere else, copy `.env.example` to `.env` and set
`AGENT_ENV_HUB_BACKEND_PROXY_URL`.

## Build

```bash
npm run build:static   # writes out/
```

`agent-env up` serves the built UI when `[explorer] static_dir` in `.agentenv/config.toml` points
at `out/`. Checks: `npm run typecheck`, `npm run test:smoke` and `npm run lint`.
