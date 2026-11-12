A Linux desktop with LibreOffice that an agent writes Office files on through the `computer_*` tools.

`envs/desktop/env.toml` is a computer env of the shipped `ubuntu-24.04` base computer, the `reference` driver and the
`apps` bundle's `libreoffice`, named by their aliases. A run composes it first, which takes a few minutes the first time
and is reused after.

`tasks/probe.json` deploys it and runs the conformance probe at `full` against it, with `artifacts/check_probe/`. It
needs Docker and the local registry on `localhost:5000`, and no model.

`tasks/author.json` has Claude Code, given the computer's skills (`add_skills` `env_ids`), write a memo as a .docx and a
two-slide presentation as a .pptx whose second slide has a list with an entrance animation, in LibreOffice, while the
display is recorded; `artifacts/check_office/` reads both files inside the computer and checks the memo's title and
paragraphs, the slides' titles and list, and the animation. It needs a model endpoint as well.

Run them with `agent-env run office`, or one with `agent-env run office --task probe`.
