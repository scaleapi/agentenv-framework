"""OpenAI's Codex Computer Use as a computer driver on macOS: the ChatGPT app's ``unified-computer-use`` plugin, whose
MCP server, ``cua_repl``, runs the JavaScript an agent sends against OpenAI's ``cua`` API, which reads apps'
accessibility trees and acts on them in the background through the plugin's helper app. The pinned app is unpacked
beside the driver and never run, so it never updates itself; the driver starts ``cua_repl`` as the app would, on the
app's own Node with the environment the app builds, one server for each agent session, since the REPL keeps its
bindings between calls, closed once it's left unused or past the most kept, since nothing says when a session ends.
Agents get the server's instructions and its ``js`` and ``js_reset`` tools as pinned (``mcp.json``), ``js`` marked as
acting on apps where Codex marks it read-only; the two it serves only the ChatGPT app are hidden. Codex asks its host
before each action on an app, and the driver answers: it approves the apps the ``allowed_apps`` setting names, by name
or bundle identifier, or every app when it names none, and nothing else, recording the computer's audio included. Each
call runs under Codex's own sandbox, read-only but for Codex's folder and the temporary ones and with no network, unless
the ``js_sandbox`` setting is off; the apps it drives are outside it. No ChatGPT sign-in is needed. The computer/v1
operations are MacOSDriver's; this process's Python is granted Accessibility, Post Event and Screen Recording, as the
helper app is. It needs ``agentenv-framework-protocol[computer,macos]`` and the pinned ChatGPT app."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (CONNECTION_CLOSED, CallToolResult, ElicitRequestParams, ElicitResult, InitializeResult,
                       TextContent)

from agentenv_protocol import EnvironmentTool, ForwardedTool, ToolContext, server_instructions
from agentenv_protocol.computer import ComputerSettings, MacOSDriver, ReadyCheck, computer_driver

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
#: The pinned ChatGPT app, as install.sh unpacks it, and where in it what starts Codex's REPL is.
VERSION = "26.930.51102"
APP = HERE / "ChatGPT.app"
CUA_NODE = Path("Contents", "Resources", "cua_node")
NODE = CUA_NODE / "bin" / "node"
NODE_REPL = CUA_NODE / "bin" / "node_repl"
NODE_MODULES = CUA_NODE / "lib" / "node_modules"
LAUNCHER = NODE_MODULES / "@oai" / "cua-repl" / "bin" / "cua-repl.mjs"
CODEX_CLI = Path("Contents", "Resources", "codex-cli", "CodexCLI.app", "Contents", "MacOS", "codex")
#: Codex Computer Use's helper app, where install.sh copies it and the layer's grant names it.
HELPER = Path("/opt/agentenv/codex-computer-use/Codex Computer Use.app")
#: The sha256 of each executable at the pinned version, by its path under the app or the helper.
EXECUTABLES = {
    "ChatGPT.app/Contents/MacOS/ChatGPT": "1f2989ac3d702e48c740fe4c54b1b6279f9e48e0e4c0ead7d4501464e2ea976e",
    "ChatGPT.app/Contents/Resources/cua_node/bin/node":
        "59f603e9e6d81c8b6bdaf6bb35b1990e7b8048762ca07a524ac7e87ade6fea15",
    "ChatGPT.app/Contents/Resources/cua_node/bin/node_repl":
        "bbd1640851593a724e2a6bc558f3a16078f40b09d4fe76f2f6d89f8b895eb5ed",
    "Codex Computer Use.app/Contents/MacOS/SkyComputerUseService":
        "fa5b5d685f550af82902cff1a345db558a1da3eb526cf78cd4659fa0dcde01a5",
}
#: What ``cua_repl`` says at connect and lists at the pinned version: its instructions and its tools.
MANIFEST = HERE / "mcp.json"
#: The tools agents get; the others are the ChatGPT app's own.
FORWARDED = ("js", "js_reset")
#: The hints agents get where Codex's own undersell a tool: ``js`` acts on apps, and through them on anything.
ANNOTATIONS = {"js": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}}
#: How long a session's REPL is kept unused before it's closed, since nothing says when a session ends; how many are
#: kept at most, the least recently used closed first; and how often the idle ones are looked for.
IDLE_S, MAX_REPLS, SWEEP_S = 1800, 8, 60
#: How long a closed REPL's server has to end before it's cancelled.
CLOSE_TIMEOUT_S = 10
#: What of the driver's environment ``cua_repl`` gets: the session user's own; none of a deploy's values.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
#: The PATH it runs, and launches apps, with.
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
#: The JavaScript the readiness check runs: an inventory, which needs no app's approval.
PROBE = "await cua.getState({ emit: false }); nodeRepl.write('ready');"
#: The message of the approval Codex asks of its host before it acts on an app, which names it as the request's
#: metadata does.
APPROVAL =re.compile(r'Allow (?:Computer Use|Codex) to use "?(?P<app>[^"?]+)"?\?')
#: The key of the metadata Codex reads its sandbox from on each call.
SANDBOX_KEY = "codex/sandbox-state-meta"


@computer_driver(name="codex-computer-use",
                 supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}])
class CodexComputerUse(MacOSDriver):
    class Settings(ComputerSettings):
        #: The apps Codex may act on, by name or bundle identifier, case aside; None for every app.
        allowed_apps: list[str] | None = None
        js_sandbox: bool = True

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
                for t in _manifest()["tools"] if t["name"] in FORWARDED]

    @server_instructions
    def codex_instructions(self) -> str:
        return _manifest()["instructions"]

    async def ready_checks(self) -> list[ReadyCheck]:
        """The pinned executables, the tools ``cua_repl`` lists and that it answers, checked until they pass once, and
        MacOSDriver's TCC permissions every time."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            self._pinned = await self._pinned_checks()
        return [*self._pinned, *await super().ready_checks()]

    async def _forward(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """A call to one of Codex's tools on the caller's own REPL, answered as Codex answers it. Other sessions' REPLs
        left idle, or past the most kept, are closed. Nothing here awaits before the call is queued, so calls a session
        makes together find one REPL."""
        repl = self._repls.pop(caller := ToolContext.current().caller.session, None) or _Repl(self._server)
        self._evict(keep=MAX_REPLS - 1)
        self._repls[caller] = repl
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep())
        try:
            return await repl.call(tool, arguments, self._sandbox())
        except Exception as e:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=(
                f"Codex's cua_repl failed before it answered, so {tool} may have run: {type(e).__name__}: {e}"))])

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
        """``cua_repl`` connected and initialized, with what it said at connect, its approval requests answered; it
        ends with the block."""
        server = StdioServerParameters(command=str(APP / NODE), args=[str(APP / LAUNCHER)], env=_environment(),
                                       cwd=str(_workspace()))
        async with (stdio_client(server) as (read, write),
                    ClientSession(read, write, elicitation_callback=self._approve) as session):
            yield session, await session.initialize()

    async def _approve(self, context: RequestContext, params: ElicitRequestParams) -> ElicitResult:
        """Codex's request to act on an app, approved when ``allowed_apps`` names the app or names none; anything else
        it asks, to record the computer's audio say, is declined."""
        message = getattr(params, "message", "")
        names = _app_names(message, params.meta.model_dump() if params.meta else {})
        allowed = self.settings.allowed_apps
        approved = bool(names) and (allowed is None or bool(names & {app.lower() for app in allowed}))
        if names:
            logger.info("%s %r", "approved" if approved else "declined", message)
        else:
            logger.warning("declined %r, which names no app", message)
        return ElicitResult(action="accept", content={}) if approved else ElicitResult(action="decline")

    def _sandbox(self) -> dict:
        """The metadata that sets the sandbox Codex runs the call's JavaScript in."""
        profile = _read_only() if self.settings.js_sandbox else {"type": "disabled"}
        return {SANDBOX_KEY: {"sandboxCwd": _workspace().as_uri(), "permissionProfile": profile}}

    async def _pinned_checks(self) -> list[ReadyCheck]:
        digests = await asyncio.to_thread(_digests)
        differ = sorted(path for path, digest in digests.items() if digest != EXECUTABLES[path])
        checks = [ReadyCheck("Codex Computer Use", not differ, f"ChatGPT {VERSION}, as pinned" if not differ else
                             f"these differ from the pinned ChatGPT {VERSION}: {', '.join(differ)}")]
        if differ:
            return checks
        try:
            async with self._server() as (session, initialized):
                listed = (await session.list_tools()).tools
                probed = await session.call_tool("js", {"code": PROBE}, meta=self._sandbox())
        except Exception as e:
            return [*checks, ReadyCheck("Codex tools", False, f"cua_repl didn't answer: {type(e).__name__}: {e}")]
        running = {t.name: t.model_dump(mode="json", exclude_none=True) for t in listed}
        pinned = {t["name"]: t for t in _manifest()["tools"]}
        changed = ["instructions"] if initialized.instructions != _manifest()["instructions"] else []
        changed += sorted(name for name in running.keys() | pinned.keys() if name not in running or name not in pinned
                          or any(running[name].get(key) != value for key, value in pinned[name].items()))
        answer = " ".join(c.text for c in probed.content if isinstance(c, TextContent))
        return [*checks,
                ReadyCheck("Codex tools", not changed, f"its instructions and {len(running)} tools, as pinned"
                           if not changed else f"these differ from {MANIFEST.name}: {', '.join(changed)}"),
                ReadyCheck("Codex answers", not probed.isError and "ready" in answer,
                           "it read the computer's apps" if not probed.isError else f"cua.getState() failed: {answer}")]


class _Repl:
    """A session's ``cua_repl``, one server at a time, each held open by a task of its own, since a stdio client is
    entered and left in one task, and reached from any through that server's queue, one call at a time. A server that
    stops is let go at once, while it still ends, and the next call starts one afresh, without the last one's bindings;
    ``close`` ends them for good."""

    def __init__(self, open_server: Callable[[], contextlib.AbstractAsyncContextManager[
            tuple[ClientSession, InitializeResult]]]) -> None:
        self._open = open_server
        #: The queue of the server taking calls, None when none is; and the tasks of every server not yet ended.
        self._calls: asyncio.Queue | None = None
        self._serving: set[asyncio.Task] = set()
        #: When a call last ended, on the monotonic clock, and how many are queued or running.
        self.used, self.busy = time.monotonic(), 0

    async def call(self, tool: str, arguments: dict, meta: dict) -> CallToolResult:
        if self._calls is None:
            self._calls = asyncio.Queue()
            serving = asyncio.create_task(self._serve(self._calls))
            self._serving.add(serving)
            serving.add_done_callback(self._serving.discard)
        answer = asyncio.get_running_loop().create_future()
        self.busy += 1
        try:
            self._calls.put_nowait((tool, arguments, meta, answer))
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
        stopped: BaseException = ConnectionError("cua_repl stopped")
        try:
            async with self._open() as (session, _):
                try:
                    while (call := await calls.get()) is not None:
                        tool, arguments, meta, answer = call
                        try:
                            result = await session.call_tool(tool, arguments, meta=meta)
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


def _app_names(message: str, meta: dict) -> set[str]:
    """The app an approval request is for, case aside: its bundle identifier and name as the request's metadata gives
    them, and the name its message quotes; none for a request that isn't for an app."""
    shown = [field.get("value") for field in meta.get("tool_params_display") or [] if field.get("name") == "app"]
    named = APPROVAL.fullmatch(message.strip())
    names = [(meta.get("tool_params") or {}).get("app"), *shown, named and named["app"]]
    return {str(name).strip().lower() for name in names if name}


def _codex_home() -> Path:
    return Path.home() / ".codex"


def _read_only() -> dict:
    """Codex's managed sandbox profile: everything readable, only Codex's folder and the temporary ones writable, and
    no network."""
    special = [{"path": {"type": "special", "value": {"kind": kind}}, "access": access}
               for kind, access in (("root", "read"), ("tmpdir", "write"), ("slash_tmp", "write"))]
    codex = {"path": {"type": "path", "path": str(_codex_home())}, "access": "write"}
    return {"type": "managed", "network": "restricted",
            "file_system": {"type": "restricted", "entries": [*special, codex]}}


def _workspace() -> Path:
    """The folder ``cua_repl`` runs in, Codex's sandbox's working folder, inside the one it may write."""
    folder = _codex_home() / "agentenv"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _environment() -> dict[str, str]:
    """``cua_repl``'s environment, as the ChatGPT app builds it for the plugin, with the computer surface alone."""
    codex, modules = str(_codex_home()), str(APP / NODE_MODULES)
    return {**{name: os.environ[name] for name in KEPT if name in os.environ}, "PATH": PATH,
            "CODEX_HOME": codex,
            "CODEX_CLI_PATH": str(APP / CODEX_CLI),
            "CUA_REPL_ENABLED_SURFACES": "computer",
            "CUA_REPL_NODE_REPL_PATH": str(APP / NODE_REPL),
            "NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE": "Control desktop apps on macOS through Computer Use.",
            "NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS": "1000",
            "NODE_REPL_NODE_MODULE_DIRS": modules,
            "NODE_REPL_NODE_PATH": str(APP / NODE),
            "NODE_REPL_TRUSTED_CODE_PATHS": f"{codex}:{modules}",
            "NODE_REPL_TRUSTED_SERVICES": json.dumps({"sky": "@oai/sky/service"}),
            "SKY_CUA_SERVICE_PATH": str(HELPER)}


def _digests() -> dict[str, str | None]:
    """Each pinned executable's sha256, None for one that isn't there."""
    roots = {"ChatGPT.app": APP, "Codex Computer Use.app": HELPER}
    found = {}
    for path in EXECUTABLES:
        root, _, rest = path.partition("/")
        file = roots[root] / rest
        found[path] = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    return found
