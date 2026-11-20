The smallest bundle: load a greeting into a local sandbox and check it.

It needs no Docker, model or configuration. `deploy_sandbox` starts a local sandbox (a work
folder on this machine), `load_artifact` copies `artifacts/greeting/` into it, and
`verify_sandbox` checks the greeting with a file probe and a shell command. The run removes the
work folder when it ends; `agent-env run hello --keep` holds it up until Ctrl-C.

Run it with `agent-env run hello`, or copy this folder and run the copy with
`agent-env run ./my-hello`.
