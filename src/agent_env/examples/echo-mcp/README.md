An MCP server built from a Dockerfile, deployed through the gateway and checked by calling its tool.

It needs Docker. `envs/echo/` is the server: a `Dockerfile` and a `server.py` whose one tool, `echo`,
returns the text it's sent, named `echo` by its environment card. `deploy_env` builds it and deploys
it behind the gateway, and `env_outcome_verifier` runs `artifacts/check_echo/verify.py` against the
env's MCP endpoint. The first run also builds the infra it deploys on, the gateway and a service
database, which takes minutes. The run tears it all down when it ends.

Run it with `agent-env run echo-mcp`, or copy this folder and run the copy with
`agent-env run ./my-echo`.
