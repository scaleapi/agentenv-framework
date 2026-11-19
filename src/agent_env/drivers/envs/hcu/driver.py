"""Hybrid Computer Use (HCU) on Linux and Windows: the pinned server's instructions and AX and GUI tools,
beside the protocol driver's computer_* tools. One stdio server retains its snapshots, refs and layout lease for the driver's
lifetime; calls reach it in order. A server that stops fails its pending calls, and the next call starts a fresh one."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

from agentenv_protocol import ForwardedTool, computer, server_instructions
from agentenv_protocol.computer import ReadyCheck, computer_driver

HERE = Path(__file__).resolve().parent
VERSION = "1.0.1"
WINDOWS = sys.platform == "win32"
OSDriver = getattr(computer, "WindowsDriver" if WINDOWS else "X11Driver")
EXECUTABLE = HERE / "bin" / ("computer-use.exe" if WINDOWS else "computer-use")
EXECUTABLES = {"linux/amd64": "3a4243b3801b4e49c3fb50d0191e31be5bcc590cc700d2c921c3ff864a7f3923",
               "windows/amd64": "76bfd73ba9a0a783b018a782df5d8638d3a6411a5e45068973ab86786fab8d08"}
PROVIDER = HERE / "bin" / "computer-use-windows-provider.exe"
PROVIDER_SHA256 = "0f41f120ee1e042b3a0b1a311443caecc7c38066e8cb933c083297f1a7a53f35"
INPUT_ACK = HERE / "bin" / "computer-use-input-ack.dll"
INPUT_ACK_SHA256 = "6d4b4384fec3ed49447e807c1566ee616731b7f600f2950bf967c8888fff63f4"
MANIFEST = HERE / ("mcp-windows.json" if WINDOWS else "mcp.json")
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR", "GTK_IM_MODULE", "QT_IM_MODULE", "XMODIFIERS")
PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
WINDOWS_KEPT = ("USERNAME", "USERDOMAIN", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
                "TEMP", "TMP", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "PROGRAMDATA",
                "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES", "PUBLIC",
                "ALLUSERSPROFILE", "COMPUTERNAME", "PROCESSOR_ARCHITECTURE")
WINDOWS_PATH = (r"{SYSTEMROOT}\System32;{SYSTEMROOT};{SYSTEMROOT}\System32\Wbem;"
                r"{SYSTEMROOT}\System32\WindowsPowerShell\v1.0")
SESSION_BUS = Path("/run/agentenv/bus")
# AX batches allow a minute of waits; leave time for input and final evidence.
CALL_TIMEOUT_S = 300


@computer_driver(name="hcu", supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]},
                                     {"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}])
class HybridComputerUse(OSDriver):
    def __init__(self) -> None:
        super().__init__()
        self._pinned: list[ReadyCheck] | None = None
        self._server = _Server()

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool.listed(tool, functools.partial(self._forward, tool["name"]))
                for tool in _manifest()["tools"]]

    @server_instructions
    def hcu_instructions(self) -> str:
        return _manifest()["instructions"]

    async def ready_checks(self) -> list[ReadyCheck]:
        """Cache successful pin checks; run the desktop driver's checks every time."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            self._pinned = await self._pinned_checks()
        return [*self._pinned, *await super().ready_checks()]

    async def _forward(self, tool: str, /, **arguments: Any) -> CallToolResult:
        try:
            return await self._server.call(tool, arguments)
        except Exception as error:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=(
                f"HCU's server failed before it answered, so {tool} may have run: {_reason(error)}"))])

    async def _pinned_checks(self) -> list[ReadyCheck]:
        pinned = EXECUTABLES.get(f"{self.platform.os}/{self.platform.arch}")
        digest = await asyncio.to_thread(_digest)
        matches = pinned is not None and digest == pinned
        checks = [ReadyCheck("hcu", matches, "the pinned computer-use" if matches else
                             f"bin/{EXECUTABLE.name} is {digest}, not the pinned {pinned}")]
        if WINDOWS:
            provider_digest = await asyncio.to_thread(_digest, PROVIDER)
            checks.append(ReadyCheck("hcu provider", provider_digest == PROVIDER_SHA256,
                                     "the pinned Windows provider" if provider_digest == PROVIDER_SHA256 else
                                     f"bin/{PROVIDER.name} is {provider_digest}, not the pinned {PROVIDER_SHA256}"))
            ack_digest = await asyncio.to_thread(_digest, INPUT_ACK)
            checks.append(ReadyCheck("hcu input acknowledgement", ack_digest == INPUT_ACK_SHA256,
                                     "the pinned input acknowledgement helper" if ack_digest == INPUT_ACK_SHA256 else
                                     f"bin/{INPUT_ACK.name} is {ack_digest}, not the pinned {INPUT_ACK_SHA256}"))
        if not all(check.ok for check in checks):
            return checks
        try:
            async with _session() as (session, initialized):
                listed = (await session.list_tools()).tools
        except Exception as error:
            return [*checks, ReadyCheck("hcu tools", False, f"its server didn't list them: {_reason(error)}")]
        running = {tool.name: tool.inputSchema for tool in listed}
        pinned_tools = {tool["name"]: tool["inputSchema"] for tool in _manifest()["tools"]}
        changed = ["instructions"] if (initialized.instructions or "") != _manifest()["instructions"] else []
        changed += sorted(name for name in running.keys() | pinned_tools.keys()
                          if running.get(name) != pinned_tools.get(name))
        return [*checks, ReadyCheck("hcu tools", not changed, f"{len(running)} tools, as pinned" if not changed
                                    else f"these differ from {MANIFEST.name}: {', '.join(changed)}")]


class _Server:
    """Keep the stdio client's entry and exit in one task, with a queue to serialize callers."""

    def __init__(self) -> None:
        self._calls: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None

    async def call(self, tool: str, arguments: dict) -> CallToolResult:
        if self._task is None or self._task.done():
            self._calls = asyncio.Queue()
            self._task = asyncio.create_task(self._serve(self._calls))
        answer: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._calls.put((tool, arguments, answer))
        return await answer

    async def _serve(self, calls: asyncio.Queue) -> None:
        pending = None
        try:
            async with _session() as (session, _):
                while True:
                    pending = await calls.get()
                    tool, arguments, answer = pending
                    if answer.cancelled():
                        pending = None
                        continue
                    try:
                        async with asyncio.timeout(CALL_TIMEOUT_S):
                            result = await session.call_tool(tool, arguments)
                    except TimeoutError as error:
                        raise TimeoutError(f"HCU did not answer within {CALL_TIMEOUT_S} seconds") from error
                    if not answer.done():
                        answer.set_result(result)
                    pending = None
        except BaseException as error:
            failed = [pending] if pending else []
            while not calls.empty():
                failed.append(calls.get_nowait())
            for _, _, answer in failed:
                if not answer.done():
                    answer.set_exception(error if isinstance(error, Exception) else RuntimeError(_reason(error)))
            if not isinstance(error, Exception):
                raise


@contextlib.asynccontextmanager
async def _session():
    """AgentEnv's container desktops have no logind session, so no /run/user/<uid>: HCU's Linux provider refuses to
    start without XDG_RUNTIME_DIR, and the session bus is AgentEnv's /run/agentenv/bus. Fill both only when unset.
    The MCP client inherits extra environment variables; env -i gives HCU only the kept values."""
    environment = _environment()
    with contextlib.ExitStack() as resources:
        if WINDOWS:
            server = StdioServerParameters(command=str(EXECUTABLE), args=["mcp"], env=environment)
        else:
            if "XDG_RUNTIME_DIR" not in environment:
                environment["XDG_RUNTIME_DIR"] = resources.enter_context(tempfile.TemporaryDirectory(prefix="hcu-runtime-"))
            server = StdioServerParameters(command="/usr/bin/env", args=[
                "-i", *(f"{name}={value}" for name, value in environment.items()), str(EXECUTABLE), "mcp"], env={})
        async with stdio_client(server) as (read, write), ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT_S)) as session:
            yield session, await session.initialize()


def _reason(error: BaseException) -> str:
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"


@functools.cache
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _environment() -> dict[str, str]:
    if WINDOWS:
        environment = {name: os.environ[name] for name in WINDOWS_KEPT if name in os.environ}
        return {**environment, "PATH": WINDOWS_PATH.format_map(environment)}
    environment = {**{name: os.environ[name] for name in KEPT if name in os.environ}, "PATH": PATH}
    if "DBUS_SESSION_BUS_ADDRESS" not in environment and SESSION_BUS.exists():
        environment["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={SESSION_BUS}"
    return environment


def _digest(path: Path | None = None) -> str | None:
    path = EXECUTABLE if path is None else path
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
