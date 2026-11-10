"""OpenAI's Codex Computer Use as a computer driver on macOS and on Windows: the ChatGPT app's ``unified-computer-use``
plugin, whose MCP server, ``cua_repl``, runs the JavaScript an agent sends against OpenAI's ``cua`` API, which reads
apps' accessibility trees and acts on them through the plugin's service: in the background on macOS, and on Windows in
the foreground, where OpenAI runs it only on the active desktop. The pinned app is unpacked beside the driver and never
run, so it never updates itself; the driver starts ``cua_repl`` as the app would, on the app's own Node with the
environment the app builds, one server for each agent session, since the REPL keeps its bindings between calls, closed
once it's left unused or past the most kept, since nothing says when a session ends. Agents get the server's
instructions and its ``js`` and ``js_reset`` tools as pinned for the OS (``mcp.json``, ``mcp-windows.json``), ``js``
marked as acting on apps where Codex marks it read-only and its input schema one object where Codex's is a choice of
several, which Anthropic's API refuses; the two it serves only the ChatGPT app are hidden. Codex asks its host before
each action on an app, and the driver answers: it approves the apps the ``allowed_apps`` setting names, by name or
identifier as the request names it or by the name it shows (on Windows the shown name, since an app run from a renamed
Python is named as Python), or every app when it names none, and nothing else, recording the computer's audio
included. On macOS each call runs under Codex's own sandbox, read-only but for Codex's folder and the temporary ones
and with no network, unless the ``js_sandbox`` setting is off; the apps it drives are outside it. On Windows that
sandbox refuses to start Codex Computer Use's service, so the JavaScript runs unsandboxed there unless the setting
turns it on. No ChatGPT sign-in is needed. The computer/v1 operations are MacOSDriver's on macOS, where this process's
Python is granted Accessibility, Post Event and Screen Recording as the plugin's service is, and WindowsDriver's on
Windows. It needs ``agentenv-framework-protocol`` with ``[computer,macos]`` on macOS or ``[computer,windows]`` on
Windows, and the pinned ChatGPT app."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, NamedTuple

from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (CONNECTION_CLOSED, CallToolResult, ElicitRequestParams, ElicitResult, InitializeResult,
                       TextContent)

from agentenv_protocol import EnvironmentTool, ForwardedTool, ToolContext, computer, server_instructions
from agentenv_protocol.computer import ComputerSettings, ReadyCheck, computer_driver

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
#: Codex Computer Use's service on macOS, the plugin's helper app, where install.sh copies it and the layer's grant names
#: it; on Windows the plugin's executable, which needs no grant.
HELPER = Path("/opt/agentenv/codex-computer-use/Codex Computer Use.app")


class _OS(NamedTuple):
    """What differs by OS: the protocol driver the computer/v1 operations run on; the pinned ChatGPT app's version, the
    folder the install unpacks it into and its resources there; under them, the Node that starts Codex's REPL, its
    ``node_repl``, its modules, the Codex CLI the app names to it, and Codex Computer Use's service, None for the helper
    app; the sha256 of each executable at the pinned version, by its path under the driver's folder or the helper; what
    of the driver's environment ``cua_repl`` gets beside the session user's own and the PATH it runs, and launches apps,
    with, ``{SystemRoot}`` filled in from the environment, its list separator, how the app names the OS to Codex, the
    manifest pinned there; whether Codex's sandbox can hold the JavaScript, which on Windows refuses to start Codex
    Computer Use's own service; and the JavaScript the readiness check runs, one that starts that service."""
    driver: str
    version: str
    app: Path
    resources: str
    node: str
    node_repl: str
    node_modules: str
    codex_cli: str
    service: str | None
    executables: dict[str, str]
    kept: tuple[str, ...]
    path: str
    separator: str
    use_case: str
    manifest: str
    sandboxed: bool
    probe: str


OSES = {
    "macos": _OS(
        "MacOSDriver", "26.930.51102", HERE / "ChatGPT.app", "Contents/Resources",
        "cua_node/bin/node", "cua_node/bin/node_repl", "cua_node/lib/node_modules",
        "codex-cli/CodexCLI.app/Contents/MacOS/codex", None,
        {"ChatGPT.app/Contents/MacOS/ChatGPT": "1f2989ac3d702e48c740fe4c54b1b6279f9e48e0e4c0ead7d4501464e2ea976e",
         "ChatGPT.app/Contents/Resources/cua_node/bin/node":
             "59f603e9e6d81c8b6bdaf6bb35b1990e7b8048762ca07a524ac7e87ade6fea15",
         "ChatGPT.app/Contents/Resources/cua_node/bin/node_repl":
             "bbd1640851593a724e2a6bc558f3a16078f40b09d4fe76f2f6d89f8b895eb5ed",
         "Codex Computer Use.app/Contents/MacOS/SkyComputerUseService":
             "fa5b5d685f550af82902cff1a345db558a1da3eb526cf78cd4659fa0dcde01a5"},
        ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG"), "/usr/bin:/bin:/usr/sbin:/sbin", ":",
        "Control desktop apps on macOS through Computer Use.", "mcp.json", True,
        "await cua.getState({ emit: false }); nodeRepl.write('ready');"),
    "windows": _OS(
        "WindowsDriver", "26.1103.7773.0", HERE / "ChatGPT", "resources",
        "cua_node/bin/node.exe", "cua_node/bin/node_repl.exe", "cua_node/bin/node_modules", "codex.exe",
        "cua_node/bin/node_modules/@oai/sky/bin/windows/codex-computer-use.exe",
        {"ChatGPT/resources/cua_node/bin/node.exe": "10bc02b8078c896ad9f8cd8b6eed622e8d17a9425403159e98f5b896e26f933c",
         "ChatGPT/resources/cua_node/bin/node_repl.exe":
             "182c15f8994f20c1a5d8fe631f73b1f2ae3ae1a840e8e1c14d9b35231f3942b3",
         "ChatGPT/resources/cua_node/bin/node_modules/@oai/sky/bin/windows/codex-computer-use.exe":
             "0ebb47bfaaf729e22bcdb08dcfce1419eefeda80afaefed3edbc088c24a2e778",
         "ChatGPT/resources/cua_node/bin/node_modules/@oai/sky/bin/windows/swift/x64/codex-computer-use-swift.exe":
             "d43d0ff78567b212070efc8f62155b2226ef9305ec030c0c778c4aa4cee55020"},
        ("USERNAME", "USERDOMAIN", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
         "SystemRoot", "SystemDrive", "windir", "ComSpec", "PATHEXT", "ProgramData", "ProgramFiles",
         "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles", "PUBLIC", "ALLUSERSPROFILE", "COMPUTERNAME"),
        r"{SystemRoot}\System32;{SystemRoot};{SystemRoot}\System32\Wbem;{SystemRoot}\System32\WindowsPowerShell\v1.0",
        ";", "Control desktop apps on Windows through Computer Use.", "mcp-windows.json", False,
        "await cua.getState({ emit: false }); await cua.listWindows(); nodeRepl.write('ready');"),
}
OS_NAME = "windows" if sys.platform == "win32" else "macos"
OS = OSES[OS_NAME]
OSDriver = getattr(computer, OS.driver)
VERSION = OS.version
#: What ``cua_repl`` says at connect and lists at the pinned version: its instructions and its tools.
MANIFEST = HERE / OS.manifest
#: The tools agents get; the others are the ChatGPT app's own.
FORWARDED = ("js", "js_reset")
#: The hints agents get where Codex's own undersell a tool: ``js`` acts on apps, and through them on anything.
ANNOTATIONS = {"js": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}}
#: How long a session's REPL is kept unused before it's closed, since nothing says when a session ends; how many are
#: kept at most, the least recently used closed first; and how often the idle ones are looked for.
IDLE_S, MAX_REPLS, SWEEP_S = 1800, 8, 60
#: How long a closed REPL's server has to end before it's cancelled.
CLOSE_TIMEOUT_S = 10
#: The message of the approval Codex asks of its host before it acts on an app, which names it as the request's
#: metadata does.
APPROVAL =re.compile(r'Allow (?:Computer Use|Codex) to use "?(?P<app>[^"?]+)"?\?')
#: The key of the metadata Codex reads its sandbox from on each call.
SANDBOX_KEY = "codex/sandbox-state-meta"


@computer_driver(name="codex-computer-use",
                 supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]},
                           {"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}])
class CodexComputerUse(OSDriver):
    class Settings(ComputerSettings):
        #: The apps Codex may act on, by name or identifier, case aside; None for every app. One the approval request
        #: names matches, so on Windows, where it names the program an app runs from (``pythonw.exe`` for any app run
        #: from a copy of Python), name the app as Codex shows it, never a program several apps share.
        allowed_apps: list[str] | None = None
        #: Whether Codex's sandbox holds the JavaScript; None for the OS's way, on macOS and off on Windows.
        js_sandbox: bool | None = None

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
                                              inputSchema=_one_object(t["inputSchema"])),
                              functools.partial(self._forward, t["name"]),
                              {**(t.get("annotations") or {}), **ANNOTATIONS.get(t["name"], {})})
                for t in _manifest()["tools"] if t["name"] in FORWARDED]

    @server_instructions
    def codex_instructions(self) -> str:
        return _manifest()["instructions"]

    async def ready_checks(self) -> list[ReadyCheck]:
        """The pinned executables, the tools ``cua_repl`` lists and that it answers, checked until they pass once, and
        the protocol driver's own checks every time."""
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
        server = StdioServerParameters(command=str(_resources() / OS.node), args=[str(_launcher())],
                                       env=_environment(), cwd=str(_workspace()))
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
        sandboxed = OS.sandboxed if self.settings.js_sandbox is None else self.settings.js_sandbox
        profile = _read_only() if sandboxed else {"type": "disabled"}
        return {SANDBOX_KEY: {"sandboxCwd": _workspace().as_uri(), "permissionProfile": profile}}

    async def _pinned_checks(self) -> list[ReadyCheck]:
        digests = await asyncio.to_thread(_digests)
        differ = sorted(path for path, digest in digests.items() if digest != OS.executables[path])
        checks = [ReadyCheck("Codex Computer Use", not differ, f"ChatGPT {VERSION}, as pinned" if not differ else
                             f"these differ from the pinned ChatGPT {VERSION}: {', '.join(differ)}")]
        if differ:
            return checks
        try:
            async with self._server() as (session, initialized):
                listed = (await session.list_tools()).tools
                probed = await session.call_tool("js", {"code": OS.probe}, meta=self._sandbox())
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
                           "it read the computer's apps" if not probed.isError else
                           f"its probe of the computer's apps failed: {answer}")]


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


def _one_object(schema: dict) -> dict:
    """A tool's input schema as agents get it: Codex's own, or where that is a choice of objects at the top, which
    Anthropic's API refuses, one object with every choice's properties, requiring what all of them require."""
    choices = schema.get("oneOf")
    if not choices:
        return schema
    required = set.intersection(*(set(choice.get("required") or ()) for choice in choices))
    properties = {name: spec for choice in choices for name, spec in (choice.get("properties") or {}).items()}
    return {**{key: value for key, value in schema.items() if key != "oneOf"}, "type": "object",
            "properties": {**properties, **(schema.get("properties") or {})}, "required": sorted(required)}


def _app_names(message: str, meta: dict) -> set[str]:
    """The app an approval request is for, case aside: its identifier and name as the request's metadata gives them,
    and the name its message quotes; none for a request that isn't for an app."""
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


def _resources() -> Path:
    return OS.app / OS.resources


def _launcher() -> Path:
    return _resources() / OS.node_modules / "@oai" / "cua-repl" / "bin" / "cua-repl.mjs"


def _environment() -> dict[str, str]:
    """``cua_repl``'s environment, as the ChatGPT app builds it for the plugin, with the computer surface alone."""
    codex, modules = str(_codex_home()), str(_resources() / OS.node_modules)
    return {**{name: os.environ[name] for name in OS.kept if name in os.environ},
            "PATH": OS.path.format_map(os.environ),
            "CODEX_HOME": codex,
            "CODEX_CLI_PATH": str(_resources() / OS.codex_cli),
            "CUA_REPL_ENABLED_SURFACES": "computer",
            "CUA_REPL_NODE_REPL_PATH": str(_resources() / OS.node_repl),
            "NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE": OS.use_case,
            "NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS": "1000",
            "NODE_REPL_NODE_MODULE_DIRS": modules,
            "NODE_REPL_NODE_PATH": str(_resources() / OS.node),
            "NODE_REPL_TRUSTED_CODE_PATHS": f"{codex}{OS.separator}{modules}",
            "NODE_REPL_TRUSTED_SERVICES": json.dumps({"sky": "@oai/sky/service"}),
            "SKY_CUA_SERVICE_PATH": str(HELPER if OS.service is None else _resources() / OS.service)}


def _digests() -> dict[str, str | None]:
    """Each pinned executable's sha256, None for one that isn't there."""
    found = {}
    for path in OS.executables:
        root, _, rest = path.partition("/")
        file = HELPER / rest if root == HELPER.name else OS.app.parent / path
        found[path] = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    return found
