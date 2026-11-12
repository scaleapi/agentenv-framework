"""Playwright MCP for the page driver: its install, laid out beside the driver by ``install-playwright-mcp.sh`` from
the downloads ``scripts/upgrade_playwright_mcp.py`` pins; its tools, as its manifest lists them; and its process and
sessions, one per caller of the driver's tools, which the driver forwards those tools to."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import socket
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from agentenv_protocol import ForwardedTool
from agentenv_protocol.computer import DeviceUnavailable
from agentenv_protocol.tool_context import ToolContext

#: Playwright MCP, as its install lays it out beside the driver, and its tools as its version lists them with ``--caps
#: vision``, which ``scripts/upgrade_playwright_mcp.py`` writes and an integration test checks against the one installed.
HERE = Path(__file__).resolve().parent
NODE = HERE / "node" / "bin" / "node"
CLI = HERE / "node_modules" / "@playwright" / "mcp" / "cli.js"
TOOLS = HERE / "playwright-mcp-tools.json"
#: Playwright MCP's tools that run the agent's own code, in the page and in Playwright MCP's own process: hidden until
#: a task's tool access names them.
HIDDEN_TOOLS = frozenset({"browser_evaluate", "browser_run_code_unsafe"})
#: Whom the browser issues Playwright MCP's CDP address to, as its record names the connections.
CDP_USER = "playwright-mcp"
#: What Playwright MCP's process keeps of the driver's environment, which holds the deploy's tokens and settings.
KEPT_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ", "TMPDIR")
#: Where Playwright MCP writes what it names itself, and its log.
OUTPUT = Path(tempfile.gettempdir()) / "agentenv-playwright-mcp"
#: How long Playwright MCP may take to listen or to open a session, and one of its tools to answer.
START_S = 30.0
CALL_TIMEOUT_S = 300.0
#: How often a start checks whether Playwright MCP listens yet, and how long a stop waits for it to end on SIGTERM
#: before it's killed.
POLL_S = 0.1
STOP_S = 10.0
#: How long one call may take at most, from when it asks for its session to its answer, however long it waits for
#: another's start: a stop, a start and a session's opening, then its own bound.
CALL_BOUND_S = STOP_S + 2 * START_S + CALL_TIMEOUT_S
#: How long a caller's session on Playwright MCP is kept with no call: a caller's own MCP session ends with no word to
#: the driver, and each holds a connection to the browser.
SESSION_IDLE_S = 3600.0

logger = logging.getLogger(__name__)


def tools(call: Callable[..., Awaitable[CallToolResult]]) -> list[ForwardedTool]:
    """Playwright MCP's tools, as its manifest lists them, each called through ``call`` with its name first; those that
    run the agent's own code hidden until a task's tool access names them."""
    return [ForwardedTool.listed(t, functools.partial(call, t["name"]), hidden_by_default=t["name"] in HIDDEN_TOOLS)
            for t in _manifest()]


def acts(tool: str) -> bool:
    """Whether ``tool`` can change the browser or a page: any its manifest doesn't mark read-only."""
    return not next(t for t in _manifest() if t["name"] == tool)["annotations"].get("readOnlyHint", False)


@functools.cache
def _manifest() -> list[dict]:
    return json.loads(TOOLS.read_text(encoding="utf-8"))["tools"]


class PlaywrightMcp:
    """Playwright MCP, attached to the browser over the CDP address ``endpoint`` gives, and its sessions there, one per
    caller of the driver's tools, so each agent keeps its own snapshots and tab, each closed once idle for
    SESSION_IDLE_S. It starts on the first call, again once it has exited or failed to start, and again on a fresh
    address once the browser's data reset has ended the last; it ends with the driver, so a driver started again finds
    none of its predecessor's."""

    def __init__(self, endpoint: Callable[[], Awaitable[str]]) -> None:
        self._endpoint = endpoint
        self._lock = asyncio.Lock()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running_on: Optional[str] = None
        self._url = ""
        self._sessions: dict[Optional[str], _Session] = {}

    async def call(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """``tool`` called on Playwright MCP as it answers, or refused saying why it couldn't be, within CALL_BOUND_S
        of asking for its session. A call Playwright MCP or its session ends under, or that runs past its bound, fails
        then, and isn't sent again, since it may have acted."""
        try:
            async with asyncio.timeout(CALL_BOUND_S):
                return await self._answer(tool, **arguments)
        except TimeoutError:
            return _refused(f"Playwright MCP didn't answer {tool} within {CALL_BOUND_S:g} s of its asking for a "
                            "session, which may have acted")

    async def _answer(self, tool: str, /, **arguments: Any) -> CallToolResult:
        try:
            held, process = await self._session(ToolContext.current().caller.session)
        except DeviceUnavailable as e:
            return _refused(e.message)
        except Exception as e:
            return _refused(f"Playwright MCP didn't answer {tool}: {type(e).__name__}: {e}")
        called = asyncio.ensure_future(held.session.call_tool(
            tool, arguments, read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT_S)))
        exited = asyncio.ensure_future(process.wait())
        try:
            await asyncio.wait({called, exited, held.owner}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            called.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await called
            raise
        finally:
            exited.cancel()
        if not called.done():
            called.cancel()
            held.close.set()
            return _refused(f"Playwright MCP ended during {tool}, which may have acted before it did")
        try:
            return called.result()
        except Exception as e:
            held.close.set()
            return _refused(f"Playwright MCP didn't answer {tool}: {type(e).__name__}: {e}")
        finally:
            held.used_at = asyncio.get_running_loop().time()

    async def _session(self, caller: Optional[str]) -> tuple[_Session, asyncio.subprocess.Process]:
        async with self._lock:
            url = await self._running()
            now = asyncio.get_running_loop().time()
            for idle in [key for key, held in self._sessions.items()
                         if key != caller and now - held.used_at > SESSION_IDLE_S]:
                self._sessions.pop(idle).close.set()
            held = self._sessions.get(caller)
            if held is None or held.url != url or held.owner.done() or held.close.is_set():
                if held is not None:
                    held.close.set()
                try:
                    held = self._sessions[caller] = await _Session.open(url)
                except DeviceUnavailable:
                    await self._stop()
                    raise
            held.used_at = now
            return held, self._process

    async def _running(self) -> str:
        """Playwright MCP's MCP URL, started on the browser's current address for it unless it runs there already."""
        endpoint = await self._endpoint()
        if self._process is not None and self._process.returncode is None and endpoint == self._running_on:
            return self._url
        await self._stop()
        port = _free_port()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        uploads = Path.home() / "Uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / "playwright-mcp.log").open("ab") as log:
            self._process = await asyncio.create_subprocess_exec(
                *_command(endpoint, port), cwd=uploads, env=_environment(), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL, stderr=log)
        try:
            await _listening(port, self._process)
        except BaseException:
            await self._stop()
            raise
        self._running_on, self._url = endpoint, f"http://127.0.0.1:{port}/mcp"
        return self._url

    async def _stop(self) -> None:
        for held in self._sessions.values():
            held.close.set()
        self._sessions.clear()
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), STOP_S)
            if self._process.returncode is None:
                self._process.kill()
                await self._process.wait()
        self._process, self._running_on = None, None


class _Session:
    """One MCP session on Playwright MCP, held open by a task of its own until ``close`` is set, since mcp's client
    contexts are anyio task groups, which only the task that entered one can leave."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.close = asyncio.Event()
        self.session: Optional[ClientSession] = None
        self.owner: Optional[asyncio.Task] = None
        self.used_at = 0.0

    @classmethod
    async def open(cls, url: str) -> _Session:
        """A session on Playwright MCP at ``url``; DeviceUnavailable when it doesn't open one in time, wedged. One whose
        opening is given up on is closed, not left half open."""
        held = cls(url)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        held.owner = asyncio.create_task(held._own(ready))
        try:
            held.session = await asyncio.wait_for(ready, START_S)
        except BaseException as e:
            held.close.set()
            held.owner.cancel()
            if isinstance(e, TimeoutError):
                raise DeviceUnavailable(f"Playwright MCP didn't open a session within {START_S:.0f} s") from None
            raise
        return held

    async def _own(self, ready: asyncio.Future) -> None:
        try:
            timeout = httpx.Timeout(CALL_TIMEOUT_S, connect=START_S)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http, \
                    streamable_http_client(self.url, http_client=http) as (read, write, _), \
                    ClientSession(read, write) as session:
                await session.initialize()
                if ready.done():
                    return
                ready.set_result(session)
                await self.close.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            elif not isinstance(e, asyncio.CancelledError):
                logger.warning(f"a session on Playwright MCP ended: {type(e).__name__}: {e}")
            if not isinstance(e, Exception):
                raise


def _command(endpoint: str, port: int) -> list[str]:
    """Playwright MCP on loopback ``port``, attached to the browser at ``endpoint``, never a browser of its own, with
    its file access confined to its working folder, the session user's ~/Uploads; ``setpriv`` ends it with the driver."""
    return ["setpriv", "--pdeathsig", "TERM", "--", str(NODE), str(CLI), "--cdp-endpoint", endpoint,
            "--host", "127.0.0.1", "--port", str(port), "--caps", "vision", "--ignore-roots",
            "--output-dir", str(OUTPUT)]


def _environment() -> dict[str, str]:
    return {name: os.environ[name] for name in KEPT_ENVIRONMENT if name in os.environ}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _listening(port: int, process: asyncio.subprocess.Process) -> None:
    """Once Playwright MCP listens on ``port``; DeviceUnavailable when it exits first or takes too long."""
    deadline = asyncio.get_running_loop().time() + START_S
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise DeviceUnavailable(f"Playwright MCP exited at start with {process.returncode}; its log is in "
                                    f"{OUTPUT}")
        with contextlib.suppress(OSError):
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        await asyncio.sleep(POLL_S)
    raise DeviceUnavailable(f"Playwright MCP didn't listen within {START_S:.0f} s")


def _refused(message: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])
