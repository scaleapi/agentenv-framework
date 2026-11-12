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

`envs/page/` is the page driver, for a computer with no display on Linux (`amd64` and `arm64`), such as `ubuntu-24.04-browser`:
its screen is the computer's one website browser's active tab, its `screen` feature `page`. A screenshot is the page,
and input reaches it as CDP input events, both through `browser/v1`'s `page/*` operations on the browser's own server;
a tab that paints nothing shows white. It serves the page group too: `computer_navigate`, `computer_history`,
`computer_tabs`, `computer_dialog` and `computer_upload`, and every frame names the page, its tab and what is open on
it. The browser ending on its own fails the computer for good: its ready check is
`final`, and its tools answer with why. Each call it makes to the browser for an agent's tool carries that call, its tool, id, arguments and the caller's role, in the `AgentEnv-Tool-Call` and `AgentEnv-Role` headers, so the browser's log credits the action to it. Each `browser_*` call is an entry too, begun and ended with `calls/begin` and `calls/end`, which Playwright MCP's CDP commands meanwhile, and the tab events they cause, name as `for`. Name it as `page`, or by its id, `@local/agentenv-framework/drivers/page`.

Beside those it serves Playwright MCP's `browser_*` tools (with `--caps vision`), as `playwright-mcp-tools.json` lists
them, from `playwright_mcp.py`. Its install lays Node and every npm package Playwright MCP needs out from pinned
downloads, refusing a package whose dependencies aren't all there, and it runs Playwright MCP itself, attached to the browser over a CDP address the browser's relay issues to
the user `playwright-mcp`, so the guard applies to it as to any CDP client; it never starts a browser of its own. Each
caller gets a session of its own there. It starts on the first call, again once it has exited, and again on a fresh
address after the browser's data is reset; it ends with the driver, gets none of the driver's environment, and its file
access is `~/Uploads`. `browser_evaluate` and `browser_run_code_unsafe`, which run the agent's own code, are hidden by
default: a task's tool access shows them by naming them. To move to another version, run
`python scripts/upgrade_playwright_mcp.py <version>`, which pins the downloads in `env.toml` and writes the manifest from
that version's `tools/list`, and look at each new tool it prints: hide it in `HIDDEN_TOOLS` if it runs the agent's own
code, and add it to the unit test's `REVIEWED_TOOLS`, which fails until you do. The browser computer's integration test
checks the installed version lists what the manifest does.

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
defect; P7's is intermittent, so a probe it passes doesn't show it fixed. An install or known failures can also be for
one distribution, keyed `"<os>/<distribution>"` by its /etc/os-release ID, as `[install."linux/omarchy"]` and
`[[known_failures."linux/omarchy"]]`: on that distribution they replace the OS type's, never merge with them, and
elsewhere the OS type's apply. A distribution no `supports` entry names is refused at put.

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
(`HYPRLAND_INSTANCE_SIGNATURE`), not any Wayland one, since the driver reaches Hyprland's IPC; `supports` names
Omarchy's Wayland desktop on amd64, which installs `[install."linux/omarchy"]`: `requirements-omarchy.txt`, cua-driver's
release tarball for Linux, whose executable `install-hyprland.sh` unpacks into the driver's `cua-driver` folder, and the
two services, with no Hyprland plugin: Omarchy's base carries the plugin of the same cua-driver release, built by its
kit. It runs cua-driver 0.47.4, its own pin, which takes background input from any native Wayland app through the
plugin's agent seats, where 0.38.0 refuses background input there, the delivery cua-driver's own tools default to; it
comes from the release, since 0.47.4's wheel isn't on PyPI, and moving the pin moves the plugin's release in
`computers/disk/omarchy/recipe.sh` with it. `computer-ready` checks the release's executable (`hyprland/amd64` in
`EXECUTABLES`) and the tools `mcp-hyprland.json` pins, while `requirements.txt` keeps 0.38.0 for the other desktops.

The daemon starts once the compositor's socket answers, on a socket in the session user's runtime folder. What the
driver asks Hyprland (`hyprctl -j`) is in `hyprland.py`, beside `driver.py`. Text goes to the desktop in the foreground,
where Hyprland's shortcuts, a menu holding the keyboard and non-ASCII text all work. A key goes to the window Hyprland
has focused, in the foreground, where Backspace, the arrows and Home/End edit text as a person's keys do, which at the
desktop they don't, pressed as on a US keyboard, the only layout the plugin takes there, so a shifted character goes as
Shift and the key it's on. Super and keys held with it, Hyprland's own shortcuts, go to the desktop, as does a key with
nothing focused and one the window path refuses because a layer such as Omarchy's menu holds the keyboard, punctuation
there by its keysym name: given the character, cua-driver falls back to libei, which can hang its input worker. The
computer skill tells agents so (`notes`): Super chords reach the desktop's shortcuts, which the protocol's skill lists
for Omarchy, and other chords, such as Alt+Tab, go to the focused window. Clicks, drags and scrolls stay desktop-scope.

Keys aimed at a window need the seat's keymap to be the US layout's, and an input method such as Omarchy's fcitx5 takes
on the keymap that desktop text uploads, after which cua-driver refuses them (`unsupported_layout`): the driver then
puts the layout back with a chord of the layout's own keys at the desktop, Super+Ctrl+Alt+Shift+F12, which it binds to
nothing in Hyprland first (`hyprctl eval`), so Hyprland keeps it from every window, and sends the key again. It presses
the chord only while its own bind alone has it: with the chord bound to anything else, the key fails as cua-driver
refused it.

`computer-ready` also checks that the plugin is loaded, enabled, built for this Hyprland and taking input
(`transport.ready`), and every input, a `computer_*` operation or one of cua-driver's own tools that isn't read-only,
checks it again first and fails as `device_unavailable`, sending nothing, when it isn't. The runtime runs ready checks
only until they first pass. A plugin loaded again into a running Hyprland has no input seat until the desktop restarts.
After a crash, Hyprland's watchdog brings it back within the same session, in safe mode with no plugins and under a new
instance signature, so nothing restarts the driver with it: once another Hyprland runs in place of the one the driver
started with (`hyprctl instances`), the plugin check fails for good (`final`) and every input fails as an infrastructure
failure. The driver's services otherwise restart with the graphical session, as after a logout.

To move it to another cua-driver version:

1. Pin the version in `requirements.txt`, and Hyprland's own in `[install."linux/omarchy"]`'s download with the plugin
   of the same release in `computers/disk/omarchy/recipe.sh`.
2. Regenerate `mcp.json` from what that version's MCP server, `cua-driver mcp`, says at connect and lists: its
   `instructions`, each tool's `name`, `description`, `inputSchema`, `annotations` and `outputSchema`, and each
   resource's `uri`, `name`, `description` and `mimeType`. That takes a running daemon, so do it on a Linux test
   machine, `mcp-hyprland.json` the same way on a Hyprland desktop at its own pin, `mcp-macos.json` in a macOS
   computer, and `mcp-windows.json` in a Windows computer's signed-in session, on the driver's venv, since
   `cua-driver mcp` refuses Session 0, where ssh lands.
3. Set `EXECUTABLES` in `driver.py` to the sha256 of `cua_driver/bin/cua-driver` in each arch's wheel, of the Linux
   release tarball's `cua-driver` at Hyprland's own pin for `hyprland/amd64`, with the tarball's URL and sha256 in
   `[install."linux/omarchy"]`, and of CuaDriver.app's executable for `macos/arm64`, and the release
   tarball's URL and sha256 in `env.toml`'s `[install.macos]` and its folder in `install-app.sh`; for `windows/amd64`,
   the zip's `cua-driver.exe`, its URL and sha256 in `[install.windows]` and its folder in `install-windows.ps1`.
4. Run `tst/unit/drivers/cua_driver_test.py`, then `tst/integration/computer/cua_driver_test.py` on Linux, where
   `computer-ready` names any tool or executable that doesn't match its pin.

`envs/agent-desktop/` is Vercel's [agent-desktop](https://github.com/vercel-labs/agent-desktop) as a driver, for macOS
(`arm64`, composed onto a VM bundle such as `macos-26`), for Windows (`amd64`, composed onto `windows-11`) and for an
X11 desktop on Linux (`amd64` and `arm64`). agent-desktop 1.2.0 reads apps' accessibility trees, UI Automation's on
Windows and AT-SPI's on Linux, and acts on what it finds; its release's tarball for the OS and arch, checked by sha256,
is unpacked into the driver's `bin/` by `install.sh`, on Windows by `install-windows.ps1`, which also copies beside it
the `VCRUNTIME140.dll` its MSVC build loads, from the venv's Python, since Windows doesn't ship it.
Agents get what its own MCP server, `agent-desktop mcp`, serves, as `mcp.json` pins it, the same on every OS: its
instructions and its 67 `desktop_*` tools, each call forwarded to a server of its own, and its skill, its references and its guide for the OS
leading the computer's skill. The agent doesn't share the computer's files, so each screenshot a call saves to an
`output_path`, in a batch too, comes back beside the answer (up to 8 MB in all, a line naming the rest), and the
instructions add a line saying other files it writes stay on the computer. The `computer_*` tools, gate 3's probe, `computer-ready` and recording run on
`MacOSDriver` (`agentenv_protocol.computer._macos`) on macOS: ScreenCaptureKit captures the display and Quartz events,
posted where the hardware's are, send the input, from the driver's own process. On Linux they run on `X11Driver`, as
the reference driver's do, and agent-desktop gets the session's display and bus, where it finds AT-SPI. On Windows they
run on `WindowsDriver` (`agentenv_protocol.computer._windows`): mss captures the screen and `SendInput` sends the input,
and agent-desktop gets the session user's Windows folders. `[install.macos]` grants Accessibility, Post
Event and Screen Recording to the driver's Python, its venv's own copy of the pinned one, and the LaunchAgent starts
the driver on it, so agent-desktop, which the driver starts, has them too, and nothing else does. A tool can't pass
agent-desktop's global flags, so the server's come from the driver's own setting, `headed`, which a deploy passes in
`driver_settings`: off, agent-desktop's default, it acts through the accessibility API alone and refuses what needs
physical input (double clicks, hovers, drags, the mouse commands); on, it allows those and prefers physical input,
moving the pointer and focus. `computer-ready` checks the two executables' sha256 for the OS and arch, that the
server lists the tools `mcp.json` pins, and the TCC permissions on macOS, the input desktop on Windows, XTEST on Linux. To move it to another version,
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

`envs/codex-computer-use/` is OpenAI's Codex Computer Use as a driver, for the same macOS computers. It is the ChatGPT
app's `unified-computer-use` plugin, whose MCP server, `cua_repl`, runs the JavaScript an agent sends against OpenAI's
`cua` API, which reads apps' accessibility trees and acts on them in the background through the plugin's helper app. The
ChatGPT version `driver.py` pins as `VERSION`, OpenAI's versioned download checked by sha256, is unpacked beside the
driver by `install.sh` and never run, so it never updates itself, and its helper app is copied to
`/opt/agentenv/codex-computer-use/`, where `[install.macos]` grants it Accessibility, Post Event and Screen Recording,
as it grants the driver's own Python. The driver starts `cua_repl` as the app would, on the app's own Node with the
environment the app builds and the computer surface alone, one server for each agent session, since the REPL keeps its
bindings between calls; one that stops is started afresh on the next call, and nothing says when a session ends, so one
left unused for 30 minutes, or past the eight kept, is closed, the least recently used first. Agents get the server's
instructions and its `js` and `js_reset` tools, as `mcp.json` pins them, `js` marked as acting on apps where Codex marks
it read-only; `js_add_node_module_dir` and `turn_ended`, which it serves the ChatGPT app, are hidden. Codex asks its
host before each action on an app, and the driver answers: it approves the apps its `allowed_apps` setting names, by
name or bundle identifier, case aside, as the request's metadata or its message gives them, or every app when it names
none, and nothing else, recording the computer's audio included, a request that names no app logged as a warning. Each
call runs under Codex's own sandbox, read-only but for `~/.codex` and the temporary folders and with no network, unless
its `js_sandbox` setting is off; the apps it drives are outside it, and it still lets the JavaScript read files and run
commands, so a bench that wants the GUI alone guards for that. No ChatGPT sign-in is needed. Its `computer_*` tools are
agent-desktop's, from `MacOSDriver`. `computer-ready` checks the executables' sha256, that `cua_repl` gives the
instructions and lists the tools `mcp.json` pins and answers `cua.getState()`, and the TCC permissions. To move it to
another ChatGPT version, set `VERSION`, pin the zip's URL and sha256 in `env.toml` and the executables' in `driver.py`'s
`EXECUTABLES` (a unit test checks that `env.toml`, `install.sh` and `driver.py` agree on the version and the helper's
path), and take `mcp.json` again from what that version's `cua_repl` says at connect and lists. Name it as
`codex-computer-use`.

`envs/aside/` is At Inc.'s Aside Computer Use as a driver, for the same macOS computers. Its MCP server is Aside's CLI,
`aside mcp --computer-use`, with one tool, `repl`. `repl` runs the agent's JavaScript in a QuickJS sandbox whose only
app API is the `computer` global, with no file system, network or shell. `computer` reads apps' accessibility trees and
acts on them in the background through Aside's daemon and its helper app. `driver.py` pins two of At Inc.'s published
releases: the daemon component as `VERSION`, the CRX Aside's browsers update in the background, and the CLI as
`CLI_VERSION`. The install downloads both from `releases.aside.com` and checks them by sha256, and `install.sh` unpacks
them to `/opt/agentenv/aside/`, links `bin/aside` into the CLI app, and writes the account's settings with usage
analytics off before the daemon first starts. The daemon runs as a direct LaunchAgent of its own executable, since macOS
gives an app's TCC permissions only to the executable launchd starts itself. `[install.macos]` grants Accessibility,
Post Event and Screen Recording to the helper app, to the copy the daemon runs from its home, and to the driver's own
Python. The driver starts one server for each agent session, since the REPL keeps its bindings between calls, with the
session user's environment, Aside's home and the CLI's update check off. A server that stops is started afresh on the
next call, and nothing says when a session ends, so one left unused for 30 minutes, or past the eight kept, is closed,
the least recently used first. Each server gets an `--app` for each app its `allowed_apps` setting names, by name or
bundle identifier, case aside, so `computer.getApp` opens only those, or every app when it names none. Agents get the
server's instructions and its `repl` tool, as `mcp.json` pins them, marked as acting on apps. No Aside sign-in is
needed. Its `computer_*` tools are agent-desktop's, from `MacOSDriver`. `computer-ready` checks the executables' sha256
and that the daemon's health endpoint answers ready. It turns usage analytics off (`aside settings analytics disable`)
and Computer Use on (`aside settings computer-use enable`), then checks that the server gives the instructions and lists
the tool `mcp.json` pins and answers `computer.listApps()`, and the TCC permissions. To move it to other Aside releases,
set `VERSION` and `CLI_VERSION`, pin both downloads' URLs and sha256 in `env.toml` and the executables' in `driver.py`'s
`EXECUTABLES`, and take `mcp.json` again from what that version's server says at connect and lists. A unit test checks
that `env.toml`, `install.sh` and `driver.py` agree on the versions, the folder, the helper and the daemon. Name it as
`aside`.

This bundle has no tasks of its own: other bundles pull its drivers by alias, and `agent-env run drivers` has nothing
to run.
