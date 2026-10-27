An MCP server with an inbox to search, deployed through the gateway: the env the A2A agent validator gives agents.

`agent-env a2a-agent validate` deploys it, by its alias, beside the agent it checks. A release publishes the env's
image, so a run on this folder's sources pulls it rather than building it; a local sandbox still needs
Docker. `envs/mail/` is
the server: a `Dockerfile` and a `server.py` whose one tool, `search_emails`, returns the emails holding every word of
its query from a fixed inbox of three, named `mail` by its environment card. `aliases.toml` gives it the alias
`mail`, so any task can name it `mail`. `tasks/smoke.json` deploys it and checks, with
`artifacts/check_mail/verify.py`, that a search for "invoice" finds the inbox's one invoice. The first run also builds
the gateway and service database it deploys on, which takes minutes.

Run it with `agent-env run mail-mcp`, or copy this folder and run the copy with `agent-env run ./my-mail`.
