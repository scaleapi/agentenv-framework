"""OpenAI's Codex Computer Use as a computer driver on macOS: the ChatGPT app's ``unified-computer-use`` plugin, whose
MCP server, ``cua_repl``, runs the JavaScript an agent sends against OpenAI's ``cua`` API, which reads apps'
accessibility trees and acts on them in the background through the plugin's helper app. The pinned app is unpacked
beside the driver and never run, so it never updates itself; the driver starts ``cua_repl`` as the app would, on the
app's own Node with the environment the app builds, one server for each agent session, since the REPL keeps its
bindings between calls. Agents get the server's instructions and its ``js`` and ``js_reset`` tools as pinned
(``mcp.json``); the two it serves only the ChatGPT app are hidden. Codex asks its host before each action on an app,
and the driver answers: it approves the apps the ``allowed_apps`` setting names, by name or bundle identifier, or every
app when it names none, and nothing else, recording the computer's audio included. Each call runs under Codex's own
sandbox, read-only but for Codex's folder and the temporary ones and with no network, unless the ``js_sandbox`` setting
is off; the apps it drives are outside it. No ChatGPT sign-in is needed. The computer/v1 operations are MacOSDriver's;
this process's Python is granted Accessibility, Post Event and Screen Recording, as the helper app is. It needs
``agentenv-framework-protocol[computer,macos]`` and the pinned ChatGPT app."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import CONNECTION_CLOSED, CallToolResult, ElicitRequestParams, ElicitResult, TextContent

from agentenv_protocol import EnvironmentTool, ForwardedTool, ToolContext, server_instructions
from agentenv_protocol.computer import ComputerSettings, MacOSDriver, ReadyCheck, computer_driver

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
#: The pinned ChatGPT app, as install.sh unpacks it, and what of it starts Codex's REPL.
VERSION = "26.930.51102"
APP = HERE / "ChatGPT.app"
RESOURCES = APP / "Contents" / "Resources"
NODE = RESOURCES / "cua_node" / "bin" / "node"
NODE_REPL = RESOURCES / "cua_node" / "bin" / "node_repl"
NODE_MODULES = RESOURCES / "cua_node" / "lib" / "node_modules"
LAUNCHER = NODE_MODULES / "@oai" / "cua-repl" / "bin" / "cua-repl.mjs"
CODEX_CLI = RESOURCES / "codex-cli" / "CodexCLI.app" / "Contents" / "MacOS" / "codex"
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
#: What of the driver's environment ``cua_repl`` gets: the session user's own; none of a deploy's values.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
#: The PATH it runs, and launches apps, with.
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
#: The JavaScript the readiness check runs: an inventory, which needs no app's approval.
PROBE = "await cua.getState({ emit: false }); nodeRepl.write('ready');"
#: The approval Codex asks of its host before it acts on an app, which names it.
APPROVAL = re.compile(r'Allow (?:Computer Use|Codex) to use "?(?P<app>[^"?]+)"?\?')
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
        self._repls: dict[str | None, _Repl] = {}

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool(EnvironmentTool(name=t["name"], description=t["description"],
                                              inputSchema=t["inputSchema"]),
                              functools.partial(self._forward, t["name"]), t.get("annotations"))
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
        """A call to one of Codex's tools on the caller's own REPL, answered as Codex answers it."""
        repl = self._repls.get(caller := ToolContext.current().caller.session)
        if repl is None:
            repl = self._repls[caller] = _Repl(self._server)
        try:
            return await repl.call(tool, arguments, self._sandbox())
        except Exception as e:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=(
                f"Codex's cua_repl failed before it answered, so {tool} may have run: {type(e).__name__}: {e}"))])

    @contextlib.asynccontextmanager
    async def _server(self) -> AsyncIterator[ClientSession]:
        """``cua_repl`` connected and initialized, its approval requests answered; it ends with the block."""
        server = StdioServerParameters(command=str(NODE), args=[str(LAUNCHER)], env=_environment(),
                                       cwd=str(_workspace()))
        async with (stdio_client(server) as (read, write),
                    ClientSession(read, write, elicitation_callback=self._approve) as session):
            await session.initialize()
            yield session

    async def _approve(self, context: RequestContext, params: ElicitRequestParams) -> ElicitResult:
        """Codex's request to act on an app, approved when ``allowed_apps`` names the app or names none; anything else
        it asks is declined."""
        message = getattr(params, "message", "")
        approved = False
        if named := APPROVAL.fullmatch(message.strip()):
            meta = params.meta.model_dump() if params.meta else {}
            names = {named["app"].strip().lower(), str((meta.get("tool_params") or {}).get("app") or "").lower()}
            allowed = self.settings.allowed_apps
            approved = allowed is None or bool(names & {app.lower() for app in allowed})
        logger.info("%s %r", "approved" if approved else "declined", message)
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
            async with self._server() as session:
                listed = (await session.list_tools()).tools
                probed = await session.call_tool("js", {"code": PROBE}, meta=self._sandbox())
        except Exception as e:
            return [*checks, ReadyCheck("Codex tools", False, f"cua_repl didn't answer: {type(e).__name__}: {e}")]
        running = {t.name: t.inputSchema for t in listed}
        pinned = {t["name"]: t["inputSchema"] for t in _manifest()["tools"]}
        changed = sorted(name for name in running.keys() | pinned.keys() if running.get(name) != pinned.get(name))
        answer = " ".join(c.text for c in probed.content if isinstance(c, TextContent))
        return [*checks,
                ReadyCheck("Codex tools", not changed, f"{len(running)} tools, as pinned" if not changed else
                           f"these tools differ from {MANIFEST.name}: {', '.join(changed)}"),
                ReadyCheck("Codex answers", not probed.isError and "ready" in answer,
                           "it read the computer's apps" if not probed.isError else f"cua.getState() failed: {answer}")]


class _Repl:
    """One ``cua_repl`` server, held open by a task of its own, since a stdio client is entered and left in one task,
    and reached from any through a queue, one call at a time. A server that stops is started afresh on the next call,
    without the last one's bindings."""

    def __init__(self, open_server: Callable[[], contextlib.AbstractAsyncContextManager[ClientSession]]) -> None:
        self._open = open_server
        self._calls: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def call(self, tool: str, arguments: dict, meta: dict) -> CallToolResult:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._serve())
        answer = asyncio.get_running_loop().create_future()
        await self._calls.put((tool, arguments, meta, answer))
        return await answer

    async def _serve(self) -> None:
        stopped: BaseException = ConnectionError("cua_repl stopped")
        try:
            async with self._open() as session:
                while True:
                    tool, arguments, meta, answer = await self._calls.get()
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
        except Exception as e:
            stopped = e
        finally:
            while not self._calls.empty():
                *_, answer = self._calls.get_nowait()
                _settle(answer, error=stopped)


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
    codex, modules = str(_codex_home()), str(NODE_MODULES)
    return {**{name: os.environ[name] for name in KEPT if name in os.environ}, "PATH": PATH,
            "CODEX_HOME": codex,
            "CODEX_CLI_PATH": str(CODEX_CLI),
            "CUA_REPL_ENABLED_SURFACES": "computer",
            "CUA_REPL_NODE_REPL_PATH": str(NODE_REPL),
            "NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE": "Control desktop apps on macOS through Computer Use.",
            "NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS": "1000",
            "NODE_REPL_NODE_MODULE_DIRS": modules,
            "NODE_REPL_NODE_PATH": str(NODE),
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
