Computer drivers that come with agent-env, each a `computer_driver` env a computer env serves its tools through.

`envs/reference/` is the reference driver, for an X11 desktop on Linux (`amd64` and `arm64`): `mss` captures the
screen and XTEST, through `python-xlib`, sends the input, foreground only. It serves the seven `computer_*` tools, the
pointer and keyboard groups, and `computer-ready`, `computer-display`, `computer-spaces`, `computer-keyboard`, and
`computer-exec` and `data/add` through the base class's Linux defaults, and `computer-recording`, from FFmpeg's
`x11grab` through PyAV at up to 30 fps. Name it from another bundle as `reference`, the
alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/drivers/reference`.

A run that names it writes it: its files, what its class declares for each platform it supports, read without
`mss`, `python-xlib` or PyAV installed, and how a computer installs it. Its `env.toml`'s `[install.linux]` has a
computer env's composer put the folder in `/opt/agentenv/drivers/reference/`, install `requirements.txt` into a venv
there, and run `agentenv_protocol.computer.serve driver.py:ReferenceDriver` as a session service on `port`, 18765. It deploys only inside a computer env, as
the shipped `desktop` example's driver. `python -m agentenv_protocol.computer.conformance <url>` checks it, or any
driver, where it runs.

`envs/cua-driver/` is trycua's [cua-driver](https://github.com/trycua/cua/tree/main/libs/cua-driver) as a driver,
for the same X11 desktop on Linux (`amd64` and `arm64`). cua-driver 0.38.0 runs on the computer as a daemon,
`cua-driver serve`, started by `daemon.sh` once X answers, with approvals off and no telemetry. Agents get its own 66
tools beside the `computer_*` tools, as `tools.json` lists them: element tokens from `get_window_state`, clicks and
typing by element or pixel, `run_script` and its browser tools, each call forwarded on its caller's own connection to the
daemon. The `computer_*` tools, gate 3's probe, `computer-ready` and recording run on its desktop-scope calls (XTest),
in a session of their own, so they never make an agent's element tokens stale. It has no pointer or keyboard hold group,
drags from a start to an end, and scrolls with no keys held. Its skill leads with cua-driver's own skill pack, as the
executable embeds it, its other files beside the SKILL.md. An env narrows which tools its agent sees with `[tools]`.
`requirements.txt` installs `cua-driver==0.38.0` from PyPI, whose wheel carries the executable, and `computer-ready`
checks that executable's sha256 and that the daemon lists the tools `tools.json` pins. Name it as `cua-driver`, the
alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/drivers/cua-driver`.

This bundle has no tasks of its own: other bundles pull its drivers by alias, and `agent-env run drivers` has nothing
to run.
