"""Aside Computer Use as a computer driver on macOS: Aside's CLI, ``aside mcp --computer-use``, serves one tool,
``repl``, which runs the JavaScript an agent sends in a QuickJS sandbox whose only app API is the ``computer``
global, which reads apps' accessibility trees and acts on them in the background through Aside's daemon and its
helper app. The pinned releases are unpacked to ``/opt/agentenv/aside/``; the daemon runs there as a service of its
own, with Aside's home in ``home/`` there, where the daemon runs its copy of the helper app from, and the driver
turns usage analytics off and Computer Use on once it answers. The driver starts the CLI's server with a minimal
environment and an ``--app`` for each app the ``allowed_apps`` setting names, or none for every app, one server for
each agent session, since the REPL keeps its bindings between calls, closed once it's left unused or past the most
kept, since nothing says when a session ends. Agents get the server's instructions and its ``repl`` tool as pinned
(``mcp.json``), marked as acting on apps. No Aside sign-in is needed. The computer/v1 operations are MacOSDriver's;
this process's Python is granted Accessibility, Post Event and Screen Recording, as the helper app is. It needs
``agentenv-framework-protocol[computer,macos]`` and the pinned releases."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CONNECTION_CLOSED, CallToolResult, InitializeResult, TextContent
from pydantic import Field

from agentenv_protocol import EnvironmentTool, ForwardedTool, ToolContext, server_instructions
from agentenv_protocol.computer import ComputerSettings, MacOSDriver, ReadyCheck, computer_driver

HERE = Path(__file__).resolve().parent
#: The pinned Aside releases, as At Inc. publishes them and install.sh unpacks them: the daemon component, the daemon
#: and the helper app that Aside's browsers update in the background, and the CLI; and where the CLI and the helper are.
VERSION = "1.26.1111.1046"
CLI_VERSION = "1.26.1111.1046"
PREFIX = Path("/opt/agentenv/aside")
CLI = Path("bin", "aside")
HELPER = "Aside Computer Use.app"
#: The daemon app's executable, which env.toml's daemon service runs.
DAEMON = Path("Aside Daemon.app", "Contents", "MacOS", "aside-daemon")
#: Aside's home, ``ASIDE_HOME`` for the daemon service and every CLI the driver starts, fixed so the daemon's runtime
#: copy of the helper, which install.sh places and env.toml grants, has a fixed path.
ASIDE_HOME = PREFIX / "home"
#: The sha256 of each executable at the pinned version, by its path under the prefix.
EXECUTABLES = {
    str(DAEMON): "40b91cd799b8ad404d71137a62535b8335b1c2368acdf87f3cb9945e4860572b",
    f"{HELPER}/Contents/MacOS/aside-computer-use": "768662135dac8727e41791469b59c3a0312551d727c2d1f70cfe907182033d3c",
    "bin/aside": "0c13c719966302a8bf6cb5382c41765512139d3dcd360d067a4fec2c56dcf374",
}
#: What ``aside mcp --computer-use`` says at connect and lists at the pinned version: its instructions and its tool.
MANIFEST = HERE / "mcp.json"
#: The hints agents get, where Aside gives none: ``repl`` acts on apps, and through them on anything.
ANNOTATIONS = {"repl": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}}
#: The daemon's health endpoint, which answers ``"ready": true`` once it serves, and how long it has to answer.
HEALTH, HEALTH_TIMEOUT_S = "http://127.0.0.1:21420/health", 5
#: How long an ``aside settings`` command has to finish.
SETTINGS_TIMEOUT_S = 30
#: How long a session's REPL is kept unused before it's closed, since nothing says when a session ends; how many are
#: kept at most, the least recently used closed first; and how often the idle ones are looked for.
IDLE_S, MAX_REPLS, SWEEP_S = 1800, 8, 60
#: How long a closed REPL's server has to end before it's cancelled.
CLOSE_TIMEOUT_S = 10
#: What of the driver's environment the CLI gets: the session user's own; none of a deploy's values.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
#: The PATH it runs with.
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
#: The JavaScript the readiness check runs: an inventory, which needs no app, printed as the number of apps.
PROBE = "console.log((await computer.listApps({ emit: false })).length)"


@computer_driver(name="aside",
                 supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}])
class AsideDriver(MacOSDriver):
    class Settings(ComputerSettings):
        #: The apps ``computer.getApp`` may open, by name or bundle identifier, case aside; None for every app, and
        #: never empty, which would pass no ``--app`` and so allow every app too.
        allowed_apps: list[str] | None = Field(default=None, min_length=1)

    def __init__(self) -> None:
        super().__init__()
        self._pinned: list[ReadyCheck] | None = None
        #: Each session's REPL, the least recently used first.
        self._repls: dict[str | None, _Repl] = {}
        self._sweeper: asyncio.Task | None = None
        #: The closes under way, held so they aren't collected before they end.
        self._closing: set[asyncio.Task] = set()

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool(EnvironmentTool(name=t["name"], description=t["description"],
                                              inputSchema=t["inputSchema"]),
                              functools.partial(self._forward, t["name"]),
                              {**(t.get("annotations") or {}), **ANNOTATIONS.get(t["name"], {})})
                for t in _manifest()["tools"]]

    @server_instructions
    def aside_instructions(self) -> str:
        return _manifest()["instructions"]

    async def ready_checks(self) -> list[ReadyCheck]:
        """The pinned executables, the daemon, analytics off and Computer Use on, what the server says and lists and
        that it answers, checked until they pass once, and MacOSDriver's TCC permissions every time."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            self._pinned = await self._pinned_checks()
        return [*self._pinned, *await super().ready_checks()]

    async def _forward(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """A call to Aside's tool on the caller's own REPL, answered as Aside answers it. Other sessions' REPLs left
        idle, or past the most kept, are closed. Nothing here awaits before the call is queued, so calls a session
        makes together find one REPL."""
        repl = self._repls.pop(caller := ToolContext.current().caller.session, None) or _Repl(self._server)
        self._evict(keep=MAX_REPLS - 1)
        self._repls[caller] = repl
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep())
        try:
            return await repl.call(tool, arguments)
        except Exception as e:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=(
                f"Aside's MCP server failed before it answered, so {tool} may have run: {type(e).__name__}: {e}"))])

    def _evict(self, keep: int = MAX_REPLS) -> None:
        """Take out the REPLs idle for ``IDLE_S`` and, past ``keep``, the least recently used, none of them in a call,
        and close them in the background."""
        now = time.monotonic()
        free = [caller for caller, repl in self._repls.items() if not repl.busy]
        idle = [caller for caller in free if now - self._repls[caller].used > IDLE_S]
        over = max(0, len(self._repls) - len(idle) - keep)
        spare = [caller for caller in free if caller not in idle][:over]
        for caller in [*idle, *spare]:
            closing = asyncio.create_task(self._repls.pop(caller).close())
            self._closing.add(closing)
            closing.add_done_callback(self._closing.discard)

    async def _sweep(self) -> None:
        """Close idle REPLs every ``SWEEP_S`` while there are any."""
        while self._repls:
            await asyncio.sleep(SWEEP_S)
            self._evict()

    @contextlib.asynccontextmanager
    async def _server(self) -> AsyncIterator[tuple[ClientSession, InitializeResult]]:
        """``aside mcp --computer-use`` for the allowed apps, connected and initialized, with what it said at connect;
        it ends with the block, as closing its stdin ends it."""
        server = StdioServerParameters(command=str(PREFIX / CLI), args=_arguments(self.settings.allowed_apps),
                                       env=_environment())
        async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
            yield session, await session.initialize()

    async def _pinned_checks(self) -> list[ReadyCheck]:
        digests = await asyncio.to_thread(_digests)
        differ = sorted(path for path, digest in digests.items() if digest != EXECUTABLES[path])
        pinned = f"Aside {VERSION}, CLI {CLI_VERSION}"
        checks = [ReadyCheck("Aside Computer Use", not differ, f"{pinned}, as pinned" if not differ else
                             f"these differ from the pinned {pinned}: {', '.join(differ)}")]
        if differ:
            return checks
        try:
            healthy = (await asyncio.to_thread(_health)).get("ready") is True
        except Exception as e:
            return [*checks, ReadyCheck("Aside daemon", False, f"{HEALTH} didn't answer: {type(e).__name__}: {e}")]
        checks.append(ReadyCheck("Aside daemon", healthy, "it answers ready" if healthy else
                                 f"{HEALTH} doesn't say it's ready"))
        if not healthy:
            return checks
        for name, words in (("Analytics off", ("analytics", "disable")),
                            ("Computer Use on", ("computer-use", "enable"))):
            code, said = await _setting(*words)
            checks.append(ReadyCheck(name, code == 0, said or f"it exited {code}"))
            if code != 0:
                return checks
        try:
            async with self._server() as (session, initialized):
                listed = (await session.list_tools()).tools
                probed = await session.call_tool("repl", {"code": PROBE})
        except Exception as e:
            return [*checks, ReadyCheck("Aside tools", False, f"aside mcp didn't answer: {type(e).__name__}: {e}")]
        running = {t.name: t.model_dump(mode="json", exclude_none=True) for t in listed}
        pinned = {t["name"]: t for t in _manifest()["tools"]}
        changed = ["instructions"] if initialized.instructions != _manifest()["instructions"] else []
        changed += sorted(name for name in running.keys() | pinned.keys() if name not in running or name not in pinned
                          or any(running[name].get(key) != value for key, value in pinned[name].items()))
        answer = " ".join(c.text for c in probed.content if isinstance(c, TextContent)).strip()
        listed_apps = not probed.isError and answer.isdigit() and int(answer) > 0
        return [*checks,
                ReadyCheck("Aside tools", not changed, f"its instructions and {len(running)} tool, as pinned"
                           if not changed else f"these differ from {MANIFEST.name}: {', '.join(changed)}"),
                ReadyCheck("Aside answers", listed_apps, f"it read the computer's {answer} apps" if listed_apps
                           else f"computer.listApps() gave no apps: {answer}")]


class _Repl:
    """A session's ``aside mcp``, one server at a time, each held open by a task of its own, since a stdio client is
    entered and left in one task, and reached from any through that server's queue, one call at a time; a call its
    caller gave up while it waited is dropped, never sent. A server that stops is let go at once, while it still ends,
    and the next call starts one afresh, without the last one's bindings; ``close`` ends them for good."""

    def __init__(self, open_server: Callable[[], contextlib.AbstractAsyncContextManager[
            tuple[ClientSession, InitializeResult]]]) -> None:
        self._open = open_server
        #: The queue of the server taking calls, None when none is; and the tasks of every server not yet ended.
        self._calls: asyncio.Queue | None = None
        self._serving: set[asyncio.Task] = set()
        #: When a call last ended, on the monotonic clock, and how many are queued or running.
        self.used, self.busy = time.monotonic(), 0

    async def call(self, tool: str, arguments: dict) -> CallToolResult:
        if self._calls is None:
            self._calls = asyncio.Queue()
            serving = asyncio.create_task(self._serve(self._calls))
            self._serving.add(serving)
            serving.add_done_callback(self._serving.discard)
        answer = asyncio.get_running_loop().create_future()
        self.busy += 1
        try:
            self._calls.put_nowait((tool, arguments, answer))
            return await answer
        finally:
            self.busy -= 1
            self.used = time.monotonic()

    async def close(self) -> None:
        """Every server ended, its process with it, waiting at most ``CLOSE_TIMEOUT_S`` before cancelling one."""
        if self._calls is not None:
            self._calls.put_nowait(None)
            self._calls = None
        if serving := set(self._serving):
            _, late = await asyncio.wait(serving, timeout=CLOSE_TIMEOUT_S)
            for task in late:
                task.cancel()

    async def _serve(self, calls: asyncio.Queue) -> None:
        stopped: BaseException = ConnectionError("aside mcp stopped")
        try:
            async with self._open() as (session, _):
                try:
                    while (call := await calls.get()) is not None:
                        tool, arguments, answer = call
                        if answer.done():
                            continue
                        try:
                            result = await session.call_tool(tool, arguments)
                        except McpError as e:
                            _settle(answer, error=e)
                            if e.error.code == CONNECTION_CLOSED:
                                raise
                            continue
                        except Exception as e:
                            _settle(answer, error=e)
                            raise
                        _settle(answer, result=result)
                finally:
                    self._let_go(calls)
        except Exception as e:
            stopped = e
        finally:
            self._let_go(calls)
            while not calls.empty():
                if (call := calls.get_nowait()) is not None:
                    _settle(call[-1], error=stopped)

    def _let_go(self, calls: asyncio.Queue) -> None:
        """The next call to a new server, not to this one, which takes no more."""
        if self._calls is calls:
            self._calls = None


def _settle(answer: asyncio.Future, *, result: Any = None, error: BaseException | None = None) -> None:
    if answer.done():
        return
    if error is not None:
        answer.set_exception(error)
    else:
        answer.set_result(result)


@functools.cache
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _arguments(allowed_apps: list[str] | None) -> list[str]:
    """The CLI's arguments for a server that may open ``allowed_apps``, every app for None."""
    return ["mcp", "--computer-use", *(word for app in allowed_apps or [] for word in ("--app", app))]


def _environment() -> dict[str, str]:
    """The CLI's environment: the session user's own, the system PATH and Aside's home, with no update check."""
    return {**{name: os.environ[name] for name in KEPT if name in os.environ}, "PATH": PATH,
            "ASIDE_HOME": str(ASIDE_HOME), "ASIDE_CLI_UPDATE_CHECK": "0"}


def _health() -> dict:
    """What the daemon's health endpoint answers."""
    with urllib.request.urlopen(HEALTH, timeout=HEALTH_TIMEOUT_S) as response:
        return json.load(response)


async def _setting(*words: str) -> tuple[int, str]:
    """``aside settings <words>``, which needs the daemon and is idempotent: its exit code and what it printed."""
    process = await asyncio.create_subprocess_exec(
        str(PREFIX / CLI), "settings", *words, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=_environment())
    try:
        out, _ = await asyncio.wait_for(process.communicate(), SETTINGS_TIMEOUT_S)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, f"aside settings {' '.join(words)} didn't finish within {SETTINGS_TIMEOUT_S} s"
    return process.returncode, out.decode(errors="replace").strip()


def _digests() -> dict[str, str | None]:
    """Each pinned executable's sha256, None for one that isn't there."""
    return {path: hashlib.sha256((PREFIX / path).read_bytes()).hexdigest() if (PREFIX / path).is_file() else None
            for path in EXECUTABLES}
