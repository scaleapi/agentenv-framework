"""The env's date in Codex's model calls. The Codex CLI is a static binary, which no preloaded library reaches, so it
stays on real time and writes the machine's date where it tells the model the date; while this agent follows an armed
clock, each CLI run's model relay puts the env's date there instead. The patterns below are where the pinned CLI,
TOOLS_CLI_VERSION, writes its date; tst/integration/agent/codex_env_date_test.py checks them against it."""

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

# The date line of the environment context the CLI gives the model (codex-rs core/src/context/world_state/
# environment.rs, from turn_context.rs's local_time_context), and the time reminder it gives a persistent turn
# (core/src/context/current_time_reminder.rs). The CLI writes each as a text of its own, between its markers, so a
# text that only quotes one, a task's prompt say, is left as written.
_CONTEXT_MARKERS = ("<environment_context>", "</environment_context>")
_REMINDER_MARKERS = ("<current_time_reminder>", "</current_time_reminder>")
_CURRENT_DATE = re.compile(r"<current_date>[^<]*</current_date>")
_DATE_UNAVAILABLE = '<current_date status="unavailable" />'
_REMINDER_TIME = re.compile(r"It is [^<]* UTC\.")
_TIME_REMINDER = re.compile(r"\s*<current_time_reminder>" + _REMINDER_TIME.pattern + r"</current_time_reminder>\s*")
_ROLES = ("user", "developer", "system")
_DATE_TIMEOUT_S = 5
# How much of what a failing date printed its refusal quotes.
_SHOWN_CHARS = 200


async def env_date_rewrite(body: bytes) -> bytes:
    """The model call with the env's date in place of the CLI's, while this agent follows an armed clock; unchanged
    otherwise. A call that isn't JSON, a date or time this doesn't know how to place, or a date that can't be read is
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
    contexts = [(holder, key) for holder, key in _texts(request)
                if _between(holder[key], _CONTEXT_MARKERS) and "<current_date" in holder[key]]
    reminders = [(holder, key) for holder, key in _texts(request) if _between(holder[key], _REMINDER_MARKERS)]
    if not contexts and not reminders:
        return body
    for holder, key in contexts:
        context = holder[key]
        if context.count("<current_date") != len(_CURRENT_DATE.findall(context)) + context.count(_DATE_UNAVAILABLE):
            raise Refusal("Codex's environment context has a date this agent doesn't know how to place, so the model "
                          "can't be told the env's date")
    if any(not _TIME_REMINDER.fullmatch(holder[key]) for holder, key in reminders):
        raise Refusal("Codex's time reminder has a time this agent doesn't know how to place, so the model can't be "
                      "told the env's time")
    today, now_utc = await _env_now()
    for holder, key in contexts:
        holder[key] = _CURRENT_DATE.sub(f"<current_date>{today}</current_date>", holder[key])
    for holder, key in reminders:
        holder[key] = _REMINDER_TIME.sub(f"It is {now_utc} UTC.", holder[key], count=1)
    # Escaped, so a lone surrogate an MCP server's tool sent, which UTF-8 can't encode, goes as it came.
    return json.dumps(request, separators=(",", ":")).encode()


def _texts(request: dict):
    """Each text of the request's input messages in ``_ROLES``, as (holder, key), so it can be replaced in place."""
    for item in request.get("input") or []:
        if not isinstance(item, dict) or item.get("role") not in _ROLES or item.get("type", "message") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            yield item, "content"
        elif isinstance(content, list):
            yield from ((part, "text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))


def _between(text: str, markers: tuple[str, str]) -> bool:
    """Whether ``text`` is one of the CLI's fragments: its whole text between the fragment's markers."""
    stripped = text.strip()
    return stripped.startswith(markers[0]) and stripped.endswith(markers[1])


async def _env_now() -> tuple[str, str]:
    """The env's date as the CLI writes its own, in this zone, and its time in UTC as the CLI's reminder writes it. It
    comes from ``date``: this server runs on real time, but what it starts reads the env's clock, as the model's
    commands do."""
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
        day, utc = time.localtime(int(seconds)), time.gmtime(int(seconds))
    except (ValueError, OverflowError, OSError) as exc:
        raise unread(f"date printed {out.decode(errors='replace')[:_SHOWN_CHARS]!r}") from exc
    return (f"{day.tm_year:04d}-{day.tm_mon:02d}-{day.tm_mday:02d}",
            f"{utc.tm_year:04d}-{utc.tm_mon:02d}-{utc.tm_mday:02d} "
            f"{utc.tm_hour:02d}:{utc.tm_min:02d}:{utc.tm_sec:02d}")


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
