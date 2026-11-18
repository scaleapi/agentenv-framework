"""The env's clock, read for an agent's model. While agent-env's gateway runs a task env's clock
(``urn:agentenv:clock/v1``), it lists ``get_time`` on the env's MCP server, and the agent's own machine clock, and any
date its harness states from it, isn't the env's.

The framework reads the clock before each turn (``find_env_clock``): when a registered MCP server lists ``get_time``,
the turn's request ends with the env's time in an ``<ENV_CLOCK>`` note (unless the agent places it itself,
``AgentEnvAgent.env_clock_prompt_note``) and carries ``TaskRequest.env_clock``, which an agent's harness reads again
between steps (``EnvClock.read``, ``turn_note``), say after each tool call, so a clock moved during a turn, or running
faster than real time, never leaves its model on a stale date. A subprocess such as a CLI hook gets the server through
its environment (``ENV_CLOCK_SERVER_ENV``, ``ClockServer.from_env``). Unarmed, no server lists ``get_time`` and nothing
changes: the env runs on real time. A failed or slow read is logged and never fails the turn."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..tool_context import ROLE_HEADER

logger = logging.getLogger(__name__)

ENV_CLOCK_TOOL_NAME = "get_time"
# The server whose get_time a subprocess reads, as ``ClockServer.to_env`` JSON.
ENV_CLOCK_SERVER_ENV = "AGENTENV_ENV_CLOCK_SERVER"
# How long a turn waits for the clock before going without it.
FIND_TIMEOUT_S = 10.0
# How long a read between a turn's steps waits.
READ_TIMEOUT_S = 5.0
# How much of a bad get_time answer an error quotes.
_ERROR_TEXT_LIMIT = 200


@dataclass(frozen=True)
class ClockServer:
    """An MCP server's url and the headers to send it."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def to_env(self) -> str:
        return json.dumps({"url": self.url, "headers": self.headers})

    @classmethod
    def from_env(cls, value: str) -> ClockServer:
        raw = json.loads(value)
        return cls(url=raw["url"], headers=dict(raw.get("headers") or {}))


@dataclass(frozen=True)
class EnvClock:
    """An armed env clock: the server whose get_time reads it, and the time it read at the turn's start."""

    server: ClockServer
    time: str

    async def read(self, timeout_s: float = READ_TIMEOUT_S) -> str:
        """The env's time now."""
        return await read_env_clock(self.server, timeout_s)


def clock_servers(mcp_servers: Mapping[str, Any], role: str | None) -> dict[str, ClockServer]:
    """Each MCP server's url and headers, with the task's role, which the gateway filters tools by and which wins over
    one a registration names, as an agent sends them."""
    return {name: ClockServer(url=info["url"],
                              headers={**(info.get("headers") or {}), **({ROLE_HEADER: role} if role else {})})
            for name, info in mcp_servers.items()}


async def _read(server: ClockServer, *, find: bool) -> str | None:
    """The env's time from ``server``'s get_time; None when ``find`` and the server doesn't list it."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(server.url, headers=server.headers or None) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if find:
                cursor, listed = None, False
                while not listed:
                    page = await session.list_tools(cursor=cursor)
                    listed = any(t.name == ENV_CLOCK_TOOL_NAME for t in page.tools)
                    cursor = page.nextCursor
                    if not cursor:
                        break
                if not listed:
                    return None
            result = await session.call_tool(ENV_CLOCK_TOOL_NAME, {})
    if result.isError:
        raise ValueError(f"{ENV_CLOCK_TOOL_NAME} failed: {_text(result)[:_ERROR_TEXT_LIMIT]!r}")
    if isinstance(result.structuredContent, dict) and result.structuredContent.get("current_time"):
        return str(result.structuredContent["current_time"])
    text = _text(result)
    current = json.loads(text).get("current_time")
    if not current:
        raise ValueError(f"no current_time in {text[:_ERROR_TEXT_LIMIT]!r}")
    return str(current)


def _text(result: Any) -> str:
    return "\n".join(c.text for c in (result.content or []) if getattr(c, "text", None))


def failure_reason(e: BaseException) -> str:
    """What failed: the first error inside the MCP client's task groups, which say only how many there were."""
    # By its shape, not BaseExceptionGroup, which Python 3.10 has only as a backport.
    while isinstance(getattr(e, "exceptions", None), tuple) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"


async def find_env_clock(servers: Mapping[str, ClockServer], timeout_s: float = FIND_TIMEOUT_S) -> EnvClock | None:
    """The armed clock of the first of ``servers`` that lists get_time; None when none does or the read fails or
    takes longer than ``timeout_s``, which is logged."""

    async def search() -> EnvClock | None:
        for name, server in servers.items():
            try:
                current = await _read(server, find=True)
            except Exception as e:  # one server's failure is no reason to skip the others
                logger.warning(f"env clock: reading {ENV_CLOCK_TOOL_NAME} from MCP server {name!r} failed: "
                               f"{failure_reason(e)}")
                continue
            if current:
                return EnvClock(server=server, time=current)
        return None

    try:
        return await asyncio.wait_for(search(), timeout=timeout_s)
    except asyncio.TimeoutError:  # not TimeoutError before Python 3.11
        logger.warning(f"env clock: no reading within {timeout_s:g} s; the turn goes without it")
        return None


async def read_env_clock(server: ClockServer, timeout_s: float = READ_TIMEOUT_S) -> str:
    """The env's time now, from ``server``'s get_time."""
    current = await asyncio.wait_for(_read(server, find=False), timeout=timeout_s)
    assert current is not None
    return current


def prompt_note(current: str) -> str:
    # Same stance as get_time's own description: the machine's clock is not this environment's.
    return (f"<ENV_CLOCK>The current date and time in this environment is {current} (UTC). The clock on the machine "
            f"you are running on, and any date you were given from it, may not match it, so use this value, or call "
            f"{ENV_CLOCK_TOOL_NAME} for a fresh one before acting on a date.</ENV_CLOCK>")


def turn_note(current: str) -> str:
    return f"<ENV_CLOCK>Environment time now: {current} (UTC).</ENV_CLOCK>"
