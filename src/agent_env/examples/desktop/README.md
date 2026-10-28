A Linux desktop an agent uses through the `computer_*` tools: M0's smoke example of a computer env.

`envs/desktop/env.toml` is a computer env of the shipped `ubuntu-24.04` base computer and the `reference` driver,
named by their aliases. A run composes it first: it builds an image of the base with the driver installed for each
arch this machine's Docker builds, and probes the one this machine runs natively, which takes a few minutes the first
time; a later run reuses the image. Deploying it boots that image in a container on local Docker.

`tasks/probe.json` deploys it and runs the conformance probe at `full` against it, with `artifacts/check_probe/`. It
needs Docker and the local registry on `localhost:5000`, and no model.

`tasks/agent.json` has Claude Code, given the computer's skill (`add_skills` `computer_env_ids`), open the file
manager and make a folder named `Reports` in `Documents`, while the display is recorded; `artifacts/check_folder/`
checks the folder through the driver. It needs a model endpoint as well.

Run them with `agent-env run desktop`, or one with `agent-env run desktop --task probe`.
