"""The env's date in OpenCode's model calls. Where the image follows the env's clock, the CLI runs on real time to keep
its sessions in order, and the turn's model relay puts the env's date where the CLI put its own. The patterns below
are where the pinned CLI, TOOLS_CLI_VERSION, writes its date; tst/integration/agent/opencode_env_date_test.py checks
them against it."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import time

from agent_egress import clock_armed
from model_relay import Refusal

# The <env> block's date line in the CLI's system prompt (src/session/system.ts) and the websearch tool's year
# (src/tool/websearch.txt).
_ENV_DATE = re.compile(r"(<env>\n(?:  [^\n]*\n)*?  Today's date: )[^\n]*")
_SEARCH_YEAR = re.compile(r"The current year is -?\d+\.")
_DATE_TIMEOUT_S = 5
# How much of what a failing date printed its refusal quotes.
_SHOWN_CHARS = 200
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


async def env_date_rewrite(body: bytes) -> bytes:
    """The model call with the env's date in place of the CLI's, while this agent follows an armed clock; unchanged
    otherwise. A call that isn't JSON, an <env> block without a date line this knows, or a date that can't be read is
    refused, so the model is never told the machine's date instead."""
    if not clock_armed():
        return body
    try:
        request = json.loads(body)
    except ValueError as exc:
        raise Refusal("the model call isn't JSON this agent can read, so the model can't be told the env's date") \
            from exc
    if not isinstance(request, dict):
        return body
    system = [(holder, key) for holder, key in _texts(request, ("system", "developer")) if "<env>\n" in holder[key]]
    search = [(holder, key) for holder, key in _search_descriptions(request) if _SEARCH_YEAR.search(holder[key])]
    if not system and not search:
        return body
    if any(not _ENV_DATE.search(holder[key]) for holder, key in system):
        raise Refusal("OpenCode's <env> block has no date line this agent knows, so the model can't be told the env's "
                      "date")
    today, year = await _env_today()
    # Every block, the CLI's and any a prompt or its instructions add.
    for holder, key in system:
        holder[key] = _ENV_DATE.sub(lambda match: match.group(1) + today, holder[key])
    for holder, key in search:
        holder[key] = _SEARCH_YEAR.sub(f"The current year is {year}.", holder[key], count=1)
    # Escaped, so a lone surrogate an MCP server's tool sent, which UTF-8 can't encode, goes as it came.
    return json.dumps(request, separators=(",", ":")).encode()


def _texts(request: dict, roles: tuple[str, ...]):
    """Each text of the request's messages in ``roles``, as (holder, key), so it can be replaced in place."""
    for message in request.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") not in roles:
            continue
        content = message.get("content")
        if isinstance(content, str):
            yield message, "content"
        elif isinstance(content, list):
            yield from ((part, "text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))


def _search_descriptions(request: dict):
    for tool in request.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == "websearch" \
                and isinstance(function.get("description"), str):
            yield function, "description"


async def _env_today() -> tuple[str, int]:
    """The env's date as the CLI's ``Date.toDateString()`` writes one in this zone, and its year. It comes from
    ``date``: this server runs on real time, but what it starts reads the env's clock, as the model's commands do."""
    def unread(reason: str) -> Refusal:
        return Refusal(f"the env's date couldn't be read for the model: {reason}")

    try:
        code, out, err = await _read_date()
    except TimeoutError:
        raise unread(f"date gave no answer within {_DATE_TIMEOUT_S} s") from None
    except OSError as exc:
        raise unread(f"{type(exc).__name__}: {exc}") from exc
    if code != 0:
        raise unread(f"date exited {code}: {err.decode(errors='replace').strip()[:_SHOWN_CHARS]}")
    try:
        seconds, _ = out.decode().split()
        day = time.localtime(int(seconds))
    except (ValueError, OverflowError, OSError) as exc:
        raise unread(f"date printed {out.decode(errors='replace')[:_SHOWN_CHARS]!r}") from exc
    return f"{_DAYS[day.tm_wday]} {_MONTHS[day.tm_mon - 1]} {day.tm_mday:02d} {day.tm_year:04d}", day.tm_year


async def _read_date() -> tuple[int, bytes, bytes]:
    """``date``'s exit code and output. Past the timeout its process group is killed, which reaches anything it left
    holding its output, even after it exited itself."""
    environ = {name: value for name, value in os.environ.items() if name != "AGENTENV_REAL_TIME"}
    proc = await asyncio.create_subprocess_exec("date", "+%s %N", stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, env=environ, start_new_session=True)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_DATE_TIMEOUT_S)
    except BaseException:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _DATE_TIMEOUT_S)
        raise
    return proc.returncode, out, err
