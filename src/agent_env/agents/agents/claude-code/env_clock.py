"""The env's clock, read for the model. While agent-env's gateway runs an env's clock (urn:agentenv:clock/v1), it lists
``get_time`` on the env's MCP server, and the date the CLI states on its own ("Today's date is ...", from this
machine's clock) isn't the env's. So each prompt, a resume's too, carries the env's time, and a hook
(``env_clock_hook.py``) adds a fresh reading after each tool call, failed ones too: a clock moved between prompts, or
running faster than real time, never leaves the model on a stale date. Unarmed, no server lists ``get_time`` and
nothing is added: the env runs on real time, which is the CLI's date."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

ENV_CLOCK_TOOL_NAME = "get_time"
# The server whose get_time the hook reads, as {"url", "headers"} JSON in the CLI's environment while the clock is
# armed. The CLI's --mcp-config already carries the same url and headers on its command line.
ENV_CLOCK_SERVER_ENV = "AGENTENV_ENV_CLOCK_SERVER"
# How long a prompt waits for the clock before going without its note.
PROMPT_READ_TIMEOUT_S = 10.0
# How long the hook waits for the clock, and how long the CLI lets the hook run, a little longer.
HOOK_READ_TIMEOUT_S = 5.0
HOOK_TIMEOUT_S = 10
HOOK_SCRIPT = Path(__file__).with_name("env_clock_hook.py")
# After every tool call, whether it succeeded or failed: a long command that fails takes as much of the clock as one
# that succeeds.
HOOK_EVENTS = ("PostToolUse", "PostToolUseFailure")


@dataclass(frozen=True)
class ClockServer:
    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def to_env(self) -> str:
        return json.dumps({"url": self.url, "headers": self.headers})

    @classmethod
    def from_env(cls, value: str) -> ClockServer:
        raw = json.loads(value)
        return cls(url=raw["url"], headers=dict(raw.get("headers") or {}))


async def _read(server: ClockServer, *, find: bool) -> str | None:
    """The env's current time from ``server``'s get_time; None when ``find`` and the server doesn't list it."""
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
        raise ValueError(f"{ENV_CLOCK_TOOL_NAME} failed: {_text(result)[:200]!r}")
    if isinstance(result.structuredContent, dict) and result.structuredContent.get("current_time"):
        return str(result.structuredContent["current_time"])
    text = _text(result)
    current = json.loads(text).get("current_time")
    if not current:
        raise ValueError(f"no current_time in {text[:200]!r}")
    return str(current)


def failure_reason(e: BaseException) -> str:
    """What failed: the first error inside the MCP client's task groups, which say only how many there were."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"


def _text(result) -> str:
    return "\n".join(c.text for c in (result.content or []) if getattr(c, "text", None))


async def find_env_clock(servers: Mapping[str, ClockServer]) -> tuple[ClockServer, str] | None:
    """The server listing get_time and the env's time it read, trying ``servers`` in order; None when none lists it or
    the read fails, which is logged and never fails the prompt."""

    async def search() -> tuple[ClockServer, str] | None:
        for name, server in servers.items():
            try:
                current = await _read(server, find=True)
            except Exception as e:  # one server's failure is no reason to skip the others
                logger.warning(f"env clock: reading {ENV_CLOCK_TOOL_NAME} from MCP server {name!r} failed: "
                               f"{failure_reason(e)}")
                continue
            if current:
                return server, current
        return None

    try:
        return await asyncio.wait_for(search(), timeout=PROMPT_READ_TIMEOUT_S)
    except TimeoutError:
        logger.warning(f"env clock: no reading within {PROMPT_READ_TIMEOUT_S:g} s; the prompt goes without it")
        return None


async def read_env_clock(server: ClockServer, timeout_s: float = HOOK_READ_TIMEOUT_S) -> str:
    return await asyncio.wait_for(_read(server, find=False), timeout=timeout_s)


def prompt_note(current: str) -> str:
    # Same stance as get_time's own description: this machine's clock is not this environment's.
    return (f"<ENV_CLOCK>The current date and time in this environment is {current} (UTC). The clock on the machine "
            f"you are running on, and the date you were given with it, may not match it, so use this value, or call "
            f"{ENV_CLOCK_TOOL_NAME} for a fresh one before acting on a date.</ENV_CLOCK>")


def turn_note(current: str) -> str:
    return f"<ENV_CLOCK>Environment time now: {current} (UTC).</ENV_CLOCK>"


def hook_settings() -> str:
    """The CLI's --settings for an armed clock: the hook after each tool call, succeeded or failed. On the command line rather than in the
    user settings, so a snapshot of the CLI's state never carries it to an agent with no clock or no such script."""
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(HOOK_SCRIPT))}"
    entry = {"matcher": "", "hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_S}]}
    return json.dumps({"hooks": {event: [entry] for event in HOOK_EVENTS}})
