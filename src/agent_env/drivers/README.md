Computer drivers that come with agent-env, each a `computer_driver` env a computer env serves its tools through.
A computer env's `tools` is what its agents get: `standard`, AgentEnv's `computer_*` tools alone;
`driver`, the driver's own alone; or `both`, the default. `agent-env run <bundle> --set envs/desktop.driver=cua-driver`
runs a bundle's `desktop` env with another driver for one run, and `--set envs/desktop.tools=driver` gives its agents
the driver's own tools; the bundle stays as written, and what changes is written under ids ending in the settings,
such as `desktop(desktop.driver cua-driver)`.

`envs/reference/` is the reference driver, for an X11 desktop on Linux (`amd64` and `arm64`): `X11Driver`
(`agentenv_protocol.computer._x11`, the protocol's `x11` extra) as it is, `mss` capturing the screen and XTEST, through
`python-xlib`, sending the input, foreground only. It serves the seven `computer_*` tools, the
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
sha256, those tools, and that cua-driver reports Accessibility and Screen Recording granted. Its pointer's actions go in
the foreground, where cua-driver otherwise keeps from moving a person's pointer over a background app,
and a click or a drag first rests the pointer where it presses, since an app such as Tk reads where the pointer is when it
handles the press, and a button the pointer hasn't yet entered loses the click (P4, a button and a checkbox take a click). cua-driver 0.38.0 fails two of the probe's checks on macOS for defects of its own: it posts each typed
character's key-up with the A key's code (P6, typed text arrives as key events), and a drag's mouse-down after the
pointer has moved along its path (P7). `env.toml`'s `[[known_failures.macos]]` names both with that reason, so gate 3
records them in the build's probe (`known_failures`) and doesn't fail the build for them; any other failure still
does. A driver's `known_failures`, by OS type, names a check by its id and name, with why. One that a probe didn't see
fail is recorded in its probe (`unused_known_failures`, with what it saw) and warned of, so a waiver doesn't outlive its
defect; P7's is intermittent, so a probe it passes doesn't show it fixed.

On Windows (`amd64`, composed onto a disk such as `windows-11`), cua-driver is the cua-driver-rs release's zip,
checked by sha256 and unpacked beside the driver (`install-windows.ps1`). Its daemon runs in the session user's
sign-in, where `agentenv-session` starts this file as a script, since cua-driver refuses Session 0, on the named pipe
`\\.\pipe\agentenv-cua-driver`, which takes the same line-delimited JSON as the Linux socket; a call that finds every
pipe instance taken, or none yet made, waits for one, as a Unix socket would queue it. Agents get the tools
`mcp-windows.json` pins, as Windows's `cua-driver mcp` lists them in a signed-in session; `computer-ready` checks the
zip's executable's sha256 and those tools. cua-driver 0.38.0 refuses a click (`foreground_unavailable`) when the
clicked window isn't in the foreground after it, and in a burst of clicks on a window that just opened, Windows passes
the foreground between the window and its frame, so a few of the first burst's clicks are refused (P4, 16 clicks sent
at once all land); `[[known_failures.windows]]` names it. The text of Windows 11's Notepad can't be read through
cua-driver's UI Automation tree, whose walk of it fails.

On a Hyprland desktop (Wayland), as Omarchy's is, cua-driver runs its Wayland backend (`CUA_DRIVER_RS_ENABLE_WAYLAND=1`)
and sends input through the seats of Hyprland's `cua-hyprland-plugin`. Hyprland's session turns this on
(`HYPRLAND_INSTANCE_SIGNATURE`), not any Wayland one, since the driver reaches Hyprland's IPC; `supports` names no
Wayland desktop yet, so gate 1 doesn't match it to one. The daemon starts once the compositor's socket answers, on a
socket in the session user's runtime folder. What the driver asks Hyprland (`hyprctl -j`) is in `hyprland.py`, beside
`driver.py`. Keys and text go to the window Hyprland has focused, in the foreground, and to the desktop with nothing
focused; clicks, drags and scrolls stay desktop-scope. `computer-ready` also checks that the plugin is loaded, enabled,
built for this Hyprland and taking input (`transport.ready`), and every input, a `computer_*` operation or one of
cua-driver's own tools that isn't read-only, checks it again first and fails as `device_unavailable`, sending nothing,
when it isn't. The runtime runs ready checks only until they first pass, and a Hyprland restarted after a crash runs in
safe mode with no plugins, where input is accepted but never arrives. A plugin loaded again into a running Hyprland has
no input seat until the desktop restarts, and a driver that outlives its Hyprland can't reach the next one, which has a
new instance signature, so the driver's services have to restart with the graphical session. Keys aimed at a window
skip Hyprland's own shortcuts, such as Super+2, and are refused while a layer such as Omarchy's menu holds the keyboard,
and non-ASCII text aimed at one is refused; sent to the desktop in the foreground, all three work, but Backspace, the
arrows and Home/End do nothing in a text box, so neither way alone covers both.

To move it to another cua-driver version:

1. Pin the version in `requirements.txt`.
2. Regenerate `mcp.json` from what that version's MCP server, `cua-driver mcp`, says at connect and lists: its
   `instructions`, each tool's `name`, `description`, `inputSchema`, `annotations` and `outputSchema`, and each
   resource's `uri`, `name`, `description` and `mimeType`. That takes a running daemon, so do it on a Linux test
   machine, `mcp-macos.json` the same way in a macOS computer, and `mcp-windows.json` in a Windows computer's signed-in
   session, on the driver's venv, since `cua-driver mcp` refuses Session 0, where ssh lands.
3. Set `EXECUTABLES` in `driver.py` to the sha256 of `cua_driver/bin/cua-driver` in each arch's wheel, and of
   CuaDriver.app's executable for `macos/arm64`, and the release tarball's URL and sha256 in `env.toml`'s
   `[install.macos]` and its folder in `install-app.sh`; for `windows/amd64`, the zip's `cua-driver.exe`, its URL and
   sha256 in `[install.windows]` and its folder in `install-windows.ps1`.
4. Run `tst/unit/drivers/cua_driver_test.py`, then `tst/integration/computer/cua_driver_test.py` on Linux, where
   `computer-ready` names any tool or executable that doesn't match its pin.

`envs/agent-desktop/` is Vercel's [agent-desktop](https://github.com/vercel-labs/agent-desktop) as a driver, for macOS
(`arm64`, composed onto a VM bundle such as `macos-26`) and for an X11 desktop on Linux (`amd64` and `arm64`).
agent-desktop 1.2.0 reads apps' accessibility trees, AT-SPI's on Linux, and acts on what it finds; its release's
tarball for the OS and arch, checked by sha256, is unpacked into the driver's `bin/` by `install.sh`.
Agents get what its own MCP server, `agent-desktop mcp`, serves, as `mcp.json` pins it, the same on both OSes: its
instructions and its 67 `desktop_*` tools, each call forwarded to a server of its own, and its skill, its references and its guide for the OS
leading the computer's skill. The agent doesn't share the computer's files, so each screenshot a call saves to an
`output_path`, in a batch too, comes back beside the answer (up to 8 MB in all, a line naming the rest), and the
instructions add a line saying other files it writes stay on the computer. The `computer_*` tools, gate 3's probe, `computer-ready` and recording run on
`MacOSDriver` (`agentenv_protocol.computer._macos`) on macOS: ScreenCaptureKit captures the display and Quartz events,
posted where the hardware's are, send the input, from the driver's own process. On Linux they run on `X11Driver`, as
the reference driver's do, and agent-desktop gets the session's display and bus, where it finds AT-SPI. `[install.macos]` grants Accessibility, Post
Event and Screen Recording to the driver's Python, its venv's own copy of the pinned one, and the LaunchAgent starts
the driver on it, so agent-desktop, which the driver starts, has them too, and nothing else does. A tool can't pass
agent-desktop's global flags, so the server's come from the driver's own setting, `headed`, which a deploy passes in
`driver_settings`: off, agent-desktop's default, it acts through the accessibility API alone and refuses what needs
physical input (double clicks, hovers, drags, the mouse commands); on, it allows those and prefers physical input,
moving the pointer and focus. `computer-ready` checks the two executables' sha256 for the OS and arch, that the
server lists the tools `mcp.json` pins, and the TCC permissions on macOS, XTEST on Linux. To move it to another version,
pin each tarball's URL and sha256 in `env.toml` and the executables' in `driver.py`'s `EXECUTABLES`, and take `mcp.json`
again from what that version's server says at connect and lists. Name it as `agent-desktop`, the alias `aliases.toml` gives it.

`envs/macos-harness/` is browser-use's [macOS harness](https://github.com/browser-use/macos-harness) as a driver, for
the same macOS computers. macos-harness 0.2.1, from PyPI into the driver's venv, reads an app as numbered text and acts
on a number or a pixel, in the background, through eleven commands its skill teaches an agent to run in a shell,
chaining the actions it's sure of and the read after them with `&&`. Agents get one tool, `macos_harness`, which runs
the harness on a list of command lines in order, the first that fails stopping the rest, with no shell between, and a
line that doesn't start with one of its commands running nothing; instructions that say how its shell lines map to the
tool and send the agent to its skill, `skill.md` as pinned, which the computer's skill also leads with (Claude Code
cuts a server's instructions at 4,096 characters, so the skill itself isn't in them); and a read's screenshot, a
`--keep` copy too, as an image beside its text, the instructions saying the files it names stay on the computer. Its `computer_*` tools and its permissions are agent-desktop's, from
`MacOSDriver` and a grant to its own Python. The harness gets none of the driver's environment, and its telemetry is off
(`DO_NOT_TRACK`). `computer-ready` checks its version, that it prints the pinned skill, and the TCC permissions. Name it
as `macos-harness`.

This bundle has no tasks of its own: other bundles pull its drivers by alias, and `agent-env run drivers` has nothing
to run.
