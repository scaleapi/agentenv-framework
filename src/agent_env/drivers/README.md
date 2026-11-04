Computer drivers that come with agent-env, each a `computer_driver` env a computer env serves its tools through.

`envs/reference/` is the reference driver, for an X11 desktop on Linux (`amd64` and `arm64`): `mss` captures the
screen and XTEST, through `python-xlib`, sends the input, foreground only. It serves the seven `computer_*` tools, the
pointer and keyboard groups, and `computer-ready`, `computer-display`, `computer-spaces`, `computer-keyboard`, and
`computer-exec` and `data/add` through the base class's Linux defaults, and `computer-recording`, from FFmpeg's
`x11grab` through PyAV at up to 30 fps. Name it from another bundle as `reference`, the
alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/drivers/reference`.

`envs/page/` is the page driver, for a computer with no display on Linux (`amd64` and `arm64`), such as `ubuntu-24.04-headless`:
its screen is the computer's one website browser's active tab, its `screen` feature `page`. A screenshot is the page,
and input reaches it as CDP input events, both through `browser/v1`'s `page/*` operations on the browser's own server;
a tab that paints nothing shows white. It serves the page group too: `computer_navigate`, `computer_history`,
`computer_tabs`, `computer_dialog` and `computer_upload`, and every frame names the page, its tab and what is open on
it. The browser ending on its own fails the computer for good: its ready check is
`final`, and its tools answer with why. Name it as `page`, or by its id, `@local/agentenv-framework/drivers/page`.

A run that names it writes it: its files, what its class declares for each platform it supports, read without
`mss`, `python-xlib` or PyAV installed, and how a computer installs it. Its `env.toml`'s `[install.linux]` has a
computer env's composer put the folder in `/opt/agentenv/drivers/reference/`, install `requirements.txt`, resolved and
fetched once at put, into a venv there, and run `agentenv_protocol.computer.serve driver.py:ReferenceDriver` as a session service on `port`, 18765. It deploys only inside a computer env, as
the shipped `desktop` example's driver. `python -m agentenv_protocol.computer.conformance <url>` checks it, or any
driver, where it runs.

`envs/cua-driver/` is trycua's [cua-driver](https://github.com/trycua/cua/tree/main/libs/cua-driver) as a driver, for
the same X11 desktop on Linux (`amd64` and `arm64`). cua-driver 0.38.0 runs on the computer as a daemon, `cua-driver
serve`, which `driver.py` starts when run as a script, once X answers, with approvals off and no telemetry. Agents get
what cua-driver's own MCP server (`cua-driver mcp`) serves, as `mcp.json` pins it: its instructions when they connect,
its 66 tools beside the `computer_*` tools, with their output schemas (element tokens from `get_window_state`, clicks
and typing by element or pixel, `run_script` and its browser tools), and its skill pack as `skill://cua-driver/`
resources, whose text is read from that server. A call that fails is answered as that server answers one,
`tool_invocation_failed` with its exit code. Each call is forwarded to the daemon in a cua-driver session of the
caller's own unless it names one, as cua-driver's own MCP server gives each client. Element tokens belong to a window,
not a session: an agent reading a window another agent has read makes the other's tokens for it stale. The `computer_*`
tools, gate 3's probe, `computer-ready` and recording run on its desktop-scope calls (XTest) in a session of their own;
a desktop capture leaves window snapshots alone, so it never makes an agent's element tokens stale. It has no pointer or
keyboard hold group, drags from a start to an end, and scrolls with no keys held. Its skill leads with cua-driver's own
skill pack, as the executable embeds it, its other files beside the SKILL.md. An env narrows which tools its agent sees
with `[tools]`. `requirements.txt` installs `cua-driver==0.38.0` from PyPI, whose wheel carries the executable, and
`computer-ready` checks that executable's sha256 and that the daemon lists the tools `mcp.json` pins. Name it as
`cua-driver`, the alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/drivers/cua-driver`.

On macOS (`arm64`, composed onto a VM bundle such as `macos-26`), cua-driver is CuaDriver.app, from the cua-driver-rs
0.38.0 release, its tarball checked by sha256 and its app put in `/Applications` by `install-app.sh`. Its daemon is a
direct LaunchAgent, CuaDriver.app's own executable, since macOS gives the app's TCC permissions, Accessibility, Post
Event and Screen Recording, which `[install.macos]` grants it, only to the executable launchd starts itself; its socket
is `/tmp/agentenv-cua-driver.sock`. `requirements-macos.txt` installs the protocol alone. Agents get the 62 tools
`mcp-macos.json` pins, as macOS's `cua-driver mcp` lists them; `computer-ready` checks CuaDriver.app's executable's
sha256, those tools, and that cua-driver reports Accessibility and Screen Recording granted. Its pointer's actions and
its typing go in the foreground, where cua-driver otherwise keeps from moving a person's pointer over a background app,
and a drag first rests the pointer on its start, since an app such as Tk reads where the pointer is when it handles the
press. cua-driver 0.38.0 fails two of the probe's checks on macOS for defects of its own: it posts each typed
character's key-up with the A key's code (P6, typed text arrives as key events), and a drag's mouse-down after the
pointer has moved along its path (P7). `env.toml`'s `[[known_failures.macos]]` names both with that reason, so gate 3
records them in the build's probe (`known_failures`) and doesn't fail the build for them; any other failure still
does. A driver's `known_failures`, by OS type, names a check by its id and name, with why.

To move it to another cua-driver version:

1. Pin the version in `requirements.txt`.
2. Regenerate `mcp.json` from what that version's MCP server, `cua-driver mcp`, says at connect and lists: its
   `instructions`, each tool's `name`, `description`, `inputSchema`, `annotations` and `outputSchema`, and each
   resource's `uri`, `name`, `description` and `mimeType`. That takes a running daemon, so do it on a Linux test
   machine, and `mcp-macos.json` the same way in a macOS computer.
3. Set `EXECUTABLES` in `driver.py` to the sha256 of `cua_driver/bin/cua-driver` in each arch's wheel, and of
   CuaDriver.app's executable for `macos/arm64`, and the release tarball's URL and sha256 in `env.toml`'s
   `[install.macos]` and its folder in `install-app.sh`.
4. Run `tst/unit/drivers/cua_driver_test.py`, then `tst/integration/computer/cua_driver_test.py` on Linux, where
   `computer-ready` names any tool or executable that doesn't match its pin.

This bundle has no tasks of its own: other bundles pull its drivers by alias, and `agent-env run drivers` has nothing
to run.
