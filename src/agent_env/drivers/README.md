Computer drivers that come with agent-env, each a `computer_driver` env a computer env serves its tools through.

`envs/reference/` is the reference driver, for an X11 desktop on Linux (`amd64` and `arm64`): `mss` captures the
screen and XTEST, through `python-xlib`, sends the input, foreground only. It serves the seven `computer_*` tools, the
pointer and keyboard groups, and `computer-ready`, `computer-display`, `computer-spaces`, `computer-keyboard`, and
`computer-exec` and `data/add` through the base class's Linux defaults, and `computer-recording`, from FFmpeg's
`x11grab` through PyAV at up to 30 fps. Name it from another bundle as `reference`, the
alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/drivers/reference`.

A run that names it writes it: its files, and what its class declares for each platform it supports, read without
`mss`, `python-xlib` or PyAV installed. It deploys inside a computer, which agent-env can't create yet, so a deploy of it is
refused until then. `python -m agentenv_protocol.computer.conformance <url>` checks it, or any driver, where it runs.

This bundle has no tasks of its own: other bundles pull its drivers by alias, and `agent-env run drivers` has nothing
to run.
